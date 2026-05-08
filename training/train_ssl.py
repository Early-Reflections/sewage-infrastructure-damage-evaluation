"""
train_ssl.py — DINO self-supervised pretraining on Sewer-ML frames.

This script pre-trains a ViT backbone using the DINO self-distillation method
(Caron et al., 2021) on the Sewer-ML image corpus.  The resulting backbone
weights can then be used in place of the standard DINOv2 Hub weights when
fine-tuning Mask2Former — useful when domain adaptation from in-the-wild images
to sewer CCTV footage is expected to yield a meaningful benefit.

NOTE: Training from Meta's released DINOv2 weights is strongly recommended
as a first experiment.  This script provides an optional path for researchers
who want to study the effect of domain-specific SSL pretraining.

The implementation follows the DINO training recipe described in the paper
and as implemented in the solo-learn library (da Costa et al., 2022).

Usage:
    python training/train_ssl.py --config configs/base.yaml \
        --sewer-ml-root datasets/sewer_ml \
        --backbone facebook/dinov2-base \
        --epochs 35 \
        --output-dir outputs/ssl_pretraining/

Requires Sewer-ML data (request via https://forms.gle/hBaPtoweZumZAi4u9).
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import Dinov2Config, Dinov2Model

from data.sewer_ml_dataset import SewerMLDataset

logger = logging.getLogger(__name__)


# ─── Argument parsing ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DINO SSL pretraining on Sewer-ML")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--sewer-ml-root", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="facebook/dinov2-base",
                        help="HF Hub backbone to initialise from.")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="outputs/ssl_pretraining")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


# ─── DINO augmentations ───────────────────────────────────────────────────────

def build_dino_augmentations(image_size: int = 224) -> transforms.Compose:
    """
    Two global crop augmentations as used in DINO (no multi-crop, matching the
    paper's choice for Sewer-ML where defects vary in size).
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


# ─── DINO projection head ─────────────────────────────────────────────────────

class DINOHead(nn.Module):
    """MLP projection head followed by L2-normalised prototype layer."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 2048,
        out_dim: int = 256,
        n_prototypes: int = 32768,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.last_layer = nn.utils.weight_norm(nn.Linear(out_dim, n_prototypes, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


# ─── Exponential moving average update ───────────────────────────────────────

@torch.no_grad()
def update_ema(student: nn.Module, teacher: nn.Module, decay: float) -> None:
    for s_param, t_param in zip(student.parameters(), teacher.parameters()):
        t_param.data.mul_(decay).add_(s_param.data, alpha=1.0 - decay)


# ─── DINO loss ────────────────────────────────────────────────────────────────

class DINOLoss(nn.Module):
    def __init__(self, out_dim: int, n_prototypes: int, teacher_temp: float = 0.04) -> None:
        super().__init__()
        self.teacher_temp = teacher_temp
        self.register_buffer("center", torch.zeros(1, n_prototypes))

    def forward(
        self,
        student_out: torch.Tensor,
        teacher_out: torch.Tensor,
        center_momentum: float = 0.9,
    ) -> torch.Tensor:
        student_prob = F.log_softmax(student_out / 0.1, dim=-1)
        teacher_prob = F.softmax((teacher_out - self.center) / self.teacher_temp, dim=-1).detach()
        loss = -(teacher_prob * student_prob).sum(dim=-1).mean()
        # Update center (EMA of teacher outputs)
        self.center = self.center * center_momentum + teacher_out.mean(dim=0, keepdim=True) * (1 - center_momentum)
        return loss


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=4,
        project_dir=args.output_dir,
    )

    if accelerator.is_main_process:
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            level=logging.INFO,
        )
        os.makedirs(args.output_dir, exist_ok=True)

    set_seed(cfg.training.seed)

    # ── Dataset ───────────────────────────────────────────────────────────────
    augment = build_dino_augmentations(image_size=224)
    dataset = SewerMLDataset(
        root=args.sewer_ml_root,
        split="Train",
        transform=augment,
        ssl_views=True,
        max_samples=None if not args.dry_run else 512,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=False,
        drop_last=True,
    )

    # ── Student and teacher backbones ─────────────────────────────────────────
    student_backbone = Dinov2Model.from_pretrained(args.backbone)
    teacher_backbone = copy.deepcopy(student_backbone)
    for p in teacher_backbone.parameters():
        p.requires_grad = False

    embed_dim = student_backbone.config.hidden_size  # e.g. 768 for ViT-B

    student_head = DINOHead(in_dim=embed_dim)
    teacher_head = DINOHead(in_dim=embed_dim)
    teacher_head.load_state_dict(student_head.state_dict())
    for p in teacher_head.parameters():
        p.requires_grad = False

    dino_loss = DINOLoss(out_dim=256, n_prototypes=32768)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    total_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = 5e-5 * args.batch_size / 256.0

    optimizer = torch.optim.AdamW(
        list(student_backbone.parameters()) + list(student_head.parameters()),
        lr=base_lr,
        weight_decay=0.04,
        betas=(0.9, 0.999),
    )

    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
    warmup_sched = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_steps])

    # ── Accelerate prepare ────────────────────────────────────────────────────
    (
        student_backbone, teacher_backbone,
        student_head, teacher_head,
        dino_loss, optimizer, loader, scheduler,
    ) = accelerator.prepare(
        student_backbone, teacher_backbone,
        student_head, teacher_head,
        dino_loss, optimizer, loader, scheduler,
    )

    # ── EMA decay schedule (linearly annealed from base_decay to final_decay) ─
    base_decay, final_decay = 0.996, 0.999

    global_step = 0
    for epoch in range(args.epochs):
        student_backbone.train()
        student_head.train()

        progress = tqdm(
            loader,
            desc=f"SSL Epoch {epoch + 1}/{args.epochs}",
            disable=not accelerator.is_main_process,
        )

        for step, (view1, view2) in enumerate(progress):
            if args.dry_run and step >= 2:
                break

            with accelerator.accumulate(student_backbone):
                # Forward student
                s_out1 = student_head(student_backbone(view1).last_hidden_state[:, 0])
                s_out2 = student_head(student_backbone(view2).last_hidden_state[:, 0])

                # Forward teacher (no grad)
                with torch.no_grad():
                    t_out1 = teacher_head(teacher_backbone(view1).last_hidden_state[:, 0])
                    t_out2 = teacher_head(teacher_backbone(view2).last_hidden_state[:, 0])

                loss = (dino_loss(s_out1, t_out2) + dino_loss(s_out2, t_out1)) / 2
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(student_backbone.parameters(), 3.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # EMA update
            decay = final_decay - (final_decay - base_decay) * (
                math.cos(math.pi * global_step / total_steps) + 1
            ) / 2
            update_ema(student_backbone, teacher_backbone, decay)
            update_ema(student_head, teacher_head, decay)

            global_step += 1
            if global_step % 200 == 0 and accelerator.is_main_process:
                progress.set_postfix(loss=f"{loss.item():.4f}", decay=f"{decay:.4f}")

        if args.dry_run:
            break

    # ── Save backbone weights ─────────────────────────────────────────────────
    if accelerator.is_main_process:
        save_path = Path(args.output_dir) / "student_backbone"
        accelerator.unwrap_model(student_backbone).save_pretrained(str(save_path))
        logger.info("SSL pretraining complete. Backbone saved to %s", save_path)
        logger.info(
            "To use this backbone with Mask2Former fine-tuning, pass "
            "--backbone %s to train_segmentation.py and update load_dinov2_weights() "
            "to load from this local path.", save_path,
        )

    accelerator.end_training()


if __name__ == "__main__":
    main()
