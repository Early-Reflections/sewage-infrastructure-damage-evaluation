"""
evaluate.py — Run inference and compute metrics on the test split.

Usage:
    python evaluation/evaluate.py \\
        --config configs/dinov2_mask2former_crack.yaml \\
        --checkpoint outputs/crack/best/hf_model

The script loads the model from a saved HuggingFace checkpoint, runs inference
on the configured test split, and prints COCO AP and (optionally) Sewer-ML
F2CIW / F1Normal metrics.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from data.crack_dataset import build_crack_dataset
from data.csdd_dataset import build_csdd_splits
from data.transforms import build_eval_transforms
from evaluation.metrics import CocoInstanceMetrics
from training.collate import mask2former_collate_fn

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Mask2Former checkpoint")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to directory saved by model.save_pretrained().",
    )
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def build_test_dataset(cfg, split: str):
    eval_tfm = build_eval_transforms(cfg)
    dataset_name = cfg.data.dataset

    if dataset_name == "crack":
        return build_crack_dataset(cfg.data.root, split=split, transform=eval_tfm)
    elif dataset_name == "csdd":
        _, val_ds, test_ds = build_csdd_splits(
            root=cfg.data.root,
            eval_transform=eval_tfm,
            train_frac=cfg.data.get("train_split", 0.80),
            val_frac=cfg.data.get("val_split", 0.10),
            seed=cfg.training.seed,
            min_area=cfg.data.get("min_instance_area", 64),
        )
        return val_ds if split == "val" else test_ds
    raise ValueError(f"Unknown dataset '{dataset_name}'")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
    )

    base_cfg = OmegaConf.load("configs/base.yaml")
    exp_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, exp_cfg)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Load model ─────────────────────────────────────────────────────────────
    logger.info("Loading model from %s", args.checkpoint)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.checkpoint)
    model.to(device)
    model.eval()

    # ── Image processor for post-processing ───────────────────────────────────
    # Use a compatible pretrained processor for size/normalisation parameters.
    try:
        processor = AutoImageProcessor.from_pretrained(args.checkpoint)
    except Exception:
        processor = AutoImageProcessor.from_pretrained(
            "facebook/mask2former-swin-small-coco-instance"
        )

    # ── Dataset ────────────────────────────────────────────────────────────────
    dataset = build_test_dataset(cfg, args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=False,
        collate_fn=mask2former_collate_fn,
    )

    id2label: dict[int, str] = {int(k): v for k, v in cfg.data.id2label.items()}
    metrics = CocoInstanceMetrics()
    metrics.set_categories(id2label)

    target_h, target_w = cfg.data.image_size

    # ── Inference loop ─────────────────────────────────────────────────────────
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating ({args.split})"):
            pixel_values = batch["pixel_values"].to(device)
            outputs = model(pixel_values=pixel_values)

            target_sizes = [(target_h, target_w)] * pixel_values.size(0)
            predictions = processor.post_process_instance_segmentation(
                outputs,
                target_sizes=target_sizes,
                threshold=0.5,
            )

            # Move predictions to CPU and convert segmentation maps.
            cpu_preds = []
            for pred in predictions:
                cpu_preds.append(
                    {
                        "segmentation": pred["segmentation"].cpu(),
                        "segments_info": pred["segments_info"],
                    }
                )

            # Reconstruct targets in per-image list format for the metrics accumulator.
            targets = [
                {
                    "mask_labels": batch["mask_labels"][i],
                    "class_labels": batch["class_labels"][i],
                }
                for i in range(len(cpu_preds))
            ]
            metrics.update(cpu_preds, targets)

    # ── Results ────────────────────────────────────────────────────────────────
    results = metrics.compute()
    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        print(f"  {k:8s}: {v:.4f}")

    results_path = Path(args.checkpoint) / f"eval_{args.split}.json"
    import json
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
