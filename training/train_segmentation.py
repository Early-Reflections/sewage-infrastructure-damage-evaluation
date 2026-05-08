"""
train_segmentation.py — Mask2Former + DINOv2 fine-tuning with HuggingFace Accelerate.

Usage:
    python training/train_segmentation.py --config configs/dinov2_mask2former_crack.yaml
    python training/train_segmentation.py --config configs/dinov2_mask2former_csdd.yaml
    python training/train_segmentation.py --config configs/dinov2_mask2former_crack.yaml --dry-run

The script is designed for a single-GPU DGX Spark (GB10 Grace Blackwell).
BF16 mixed precision is used by default (native Blackwell BF16 support).
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.crack_dataset import build_crack_dataset
from data.csdd_dataset import build_csdd_splits
from data.transforms import build_eval_transforms, build_train_transforms
from model.segmentor import build_model, load_dinov2_weights
from training.collate import mask2former_collate_fn

logger = logging.getLogger(__name__)


# ─── Argument parsing ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Mask2Former + DINOv2")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to experiment YAML config."
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint directory to resume from."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 2 training steps and 2 eval steps then exit (sanity check).",
    )
    return parser.parse_args()


# ─── Dataset builder ──────────────────────────────────────────────────────────

def build_datasets(cfg):
    train_tfm = build_train_transforms(cfg)
    eval_tfm = build_eval_transforms(cfg)
    dataset_name = cfg.data.dataset

    if dataset_name == "crack":
        train_ds = build_crack_dataset(cfg.data.root, split="train", transform=train_tfm)
        val_ds = build_crack_dataset(cfg.data.root, split="val", transform=eval_tfm)
    elif dataset_name == "csdd":
        train_ds, val_ds, _ = build_csdd_splits(
            root=cfg.data.root,
            train_transform=train_tfm,
            eval_transform=eval_tfm,
            train_frac=cfg.data.get("train_split", 0.80),
            val_frac=cfg.data.get("val_split", 0.10),
            seed=cfg.training.seed,
            min_area=cfg.data.get("min_instance_area", 64),
        )
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Set cfg.data.dataset to 'crack' or 'csdd'.")

    return train_ds, val_ds


# ─── LR scheduler ─────────────────────────────────────────────────────────────

def build_scheduler(optimizer, cfg, steps_per_epoch: int):
    total_steps = cfg.training.num_epochs * steps_per_epoch
    warmup_steps = cfg.training.warmup_epochs * steps_per_epoch

    if cfg.training.lr_scheduler == "cosine":
        from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

        warmup = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    elif cfg.training.lr_scheduler == "multistep":
        from torch.optim.lr_scheduler import MultiStepLR

        milestones = [
            int(cfg.training.num_epochs * 0.4) * steps_per_epoch,
            int(cfg.training.num_epochs * 0.7) * steps_per_epoch,
        ]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    else:
        raise ValueError(f"Unknown lr_scheduler '{cfg.training.lr_scheduler}'")

    return scheduler


# ─── Main training loop ───────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Load config: merge base → experiment
    base_cfg = OmegaConf.load("configs/base.yaml")
    exp_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, exp_cfg)

    # Accelerator handles device placement, mixed precision, and gradient accumulation.
    accelerator = Accelerator(
        mixed_precision=cfg.training.mixed_precision,
        gradient_accumulation_steps=cfg.training.grad_accum_steps,
        log_with="wandb" if cfg.logging.get("wandb_entity") else None,
        project_dir=cfg.training.output_dir,
    )

    if accelerator.is_main_process:
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            level=logging.INFO,
        )
        os.makedirs(cfg.training.output_dir, exist_ok=True)
        OmegaConf.save(cfg, Path(cfg.training.output_dir) / "config.yaml")

    set_seed(cfg.training.seed)

    # ── Datasets and DataLoaders ───────────────────────────────────────────────
    train_ds, val_ds = build_datasets(cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=mask2former_collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=mask2former_collate_fn,
        drop_last=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg)
    load_dinov2_weights(model, cfg.model.backbone_name)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    # Use a lower LR for the pretrained backbone, higher for the randomly
    # initialised head — standard fine-tuning practice.
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if "backbone" in name or "encoder" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    effective_batch = cfg.training.batch_size * cfg.training.grad_accum_steps
    base_lr = cfg.training.base_lr * effective_batch / 16.0

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": base_lr * 0.1},
            {"params": head_params, "lr": base_lr},
        ],
        weight_decay=cfg.training.weight_decay,
        betas=(cfg.training.adam_beta1, cfg.training.adam_beta2),
    )

    steps_per_epoch = math.ceil(len(train_ds) / (cfg.training.batch_size * cfg.training.grad_accum_steps))
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)

    # ── Accelerate prepare ────────────────────────────────────────────────────
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # ── Optional W&B init ─────────────────────────────────────────────────────
    if accelerator.is_main_process and cfg.logging.get("wandb_entity"):
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": {"name": cfg.logging.run_name, "entity": cfg.logging.wandb_entity}},
        )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        accelerator.load_state(args.resume)
        start_epoch = int(Path(args.resume).name.replace("epoch_", ""))
        logger.info("Resumed from epoch %d", start_epoch)

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step = start_epoch * len(train_loader)
    best_val_loss = float("inf")

    for epoch in range(start_epoch, cfg.training.num_epochs):
        model.train()
        epoch_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{cfg.training.num_epochs}",
            disable=not accelerator.is_main_process,
        )

        for step, batch in enumerate(progress):
            if args.dry_run and step >= 2:
                break

            with accelerator.accumulate(model):
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    mask_labels=batch["mask_labels"],
                    class_labels=batch["class_labels"],
                )
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.detach().float().item()
            global_step += 1

            if global_step % cfg.training.log_every == 0 and accelerator.is_main_process:
                lr = scheduler.get_last_lr()[0]
                progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")
                accelerator.log({"train/loss": loss.item(), "train/lr": lr}, step=global_step)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_step, batch in enumerate(val_loader):
                if args.dry_run and val_step >= 2:
                    break
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    mask_labels=batch["mask_labels"],
                    class_labels=batch["class_labels"],
                )
                val_loss += outputs.loss.detach().float().item()

        val_loss /= max(len(val_loader), 1)
        train_loss = epoch_loss / max(len(train_loader), 1)

        if accelerator.is_main_process:
            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f",
                epoch + 1, cfg.training.num_epochs, train_loss, val_loss,
            )
            accelerator.log(
                {"val/loss": val_loss, "epoch": epoch + 1}, step=global_step
            )

        # ── Checkpoint ────────────────────────────────────────────────────────
        if (epoch + 1) % cfg.training.save_every == 0 and accelerator.is_main_process:
            ckpt_dir = Path(cfg.training.output_dir) / f"epoch_{epoch + 1}"
            accelerator.save_state(str(ckpt_dir))
            logger.info("Checkpoint saved to %s", ckpt_dir)

        if val_loss < best_val_loss and accelerator.is_main_process:
            best_val_loss = val_loss
            best_dir = Path(cfg.training.output_dir) / "best"
            accelerator.save_state(str(best_dir))
            # Also save the unwrapped model in HF format for easy loading.
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(str(best_dir / "hf_model"))
            logger.info("New best model saved (val_loss=%.4f)", best_val_loss)

        if args.dry_run:
            logger.info("--dry-run: exiting after first epoch.")
            break

    accelerator.end_training()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
