"""
CrackDataset — PyTorch Dataset for binary crack segmentation datasets.

Supported sources (all stored under a common root, see data/download_datasets.sh):
  crack500/    – CRACK500 (Yang et al., T-ITS 2019)
  deepcrack/   – DeepCrack (Liu et al., Neurocomputing 2019)
  khanhha/     – khanhha merged crack dataset (12 public datasets, 448×448)

Each source directory is expected to contain at minimum:
    images/  (or train/images/, test/images/)
    masks/   (or train/masks/,  test/masks/ ) — PNG uint8, 0=background 255=crack

The dataset converts every binary semantic mask to a list of instance masks by
treating the entire crack region as a single instance (crack segmentation data
does not have instance-level annotations).  This is compatible with the
Mask2Former training format.

Targets dict returned per sample:
    mask_labels  : List[np.ndarray shape (H, W)]   – one binary mask per instance
    class_labels : List[int]                        – all 1 (crack) here
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset


# ─── Utilities ────────────────────────────────────────────────────────────────

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

CRACK_CLASS_ID = 1


def _find_pairs(
    img_dir: Path, mask_dir: Path
) -> list[tuple[Path, Path]]:
    """Return (image_path, mask_path) pairs where both files exist."""
    pairs = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in _IMG_EXTS:
            continue
        # masks may be saved as .png regardless of image extension
        mask_path = mask_dir / (img_path.stem + ".png")
        if not mask_path.exists():
            mask_path = mask_dir / img_path.name
        if mask_path.exists():
            pairs.append((img_path, mask_path))
    return pairs


def _mask_to_instances(
    mask_array: np.ndarray, threshold: int = 128
) -> tuple[list[np.ndarray], list[int]]:
    """Convert a binary/grey mask to a single-instance list.

    Crack datasets do not distinguish between individual crack instances, so we
    treat the whole foreground region as one instance.  Downstream, Mask2Former
    will learn to segment the crack region as a single mask.
    """
    binary = (mask_array >= threshold).astype(np.uint8)
    if binary.sum() == 0:
        return [], []
    return [binary], [CRACK_CLASS_ID]


# ─── Single-source Dataset ────────────────────────────────────────────────────

class _CrackSourceDataset(Dataset):
    """Loads (image, targets) pairs from one image/mask directory pair."""

    def __init__(
        self,
        img_dir: Path,
        mask_dir: Path,
        transform: Callable | None = None,
    ) -> None:
        self.pairs = _find_pairs(img_dir, mask_dir)
        if not self.pairs:
            raise FileNotFoundError(
                f"No image/mask pairs found.\n  images: {img_dir}\n  masks:  {mask_dir}"
            )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        mask_array = np.array(Image.open(mask_path).convert("L"))
        mask_labels, class_labels = _mask_to_instances(mask_array)
        targets = {"mask_labels": mask_labels, "class_labels": class_labels}
        if self.transform is not None:
            image, targets = self.transform(image, targets)
        return image, targets


# ─── Per-source factory functions ─────────────────────────────────────────────

def _load_crack500(root: Path, split: str, transform) -> Dataset:
    """
    CRACK500 directory layout (after unzip):
        crack500/
          train/images/  train/masks/
          val/images/    val/masks/
          test/images/   test/masks/
    If the split subdirectory doesn't exist, fall back to flat images/masks dirs.
    """
    base = root / "crack500"
    split_dir = base / split
    if split_dir.exists():
        return _CrackSourceDataset(split_dir / "images", split_dir / "masks", transform)
    # Flat layout fallback
    return _CrackSourceDataset(base / "images", base / "masks", transform)


def _load_deepcrack(root: Path, split: str, transform) -> Dataset | None:
    """
    DeepCrack directory layout (after unzip from DeepCrack.zip):
        deepcrack/
          train_img/   train_lab/
          test_img/    test_lab/
    """
    base = root / "deepcrack"
    if not base.exists():
        return None
    if split == "train":
        img_dir, mask_dir = base / "train_img", base / "train_lab"
    else:
        img_dir, mask_dir = base / "test_img", base / "test_lab"
    if img_dir.exists() and mask_dir.exists():
        return _CrackSourceDataset(img_dir, mask_dir, transform)
    return None


def _load_khanhha(root: Path, split: str, transform) -> Dataset | None:
    """
    khanhha merged layout (all images in one flat directory):
        khanhha/
          images/
          masks/
    No official split — caller handles train/val split upstream.
    """
    base = root / "khanhha"
    if not base.exists():
        return None
    img_dir = base / "images" if (base / "images").exists() else base / "image"
    mask_dir = base / "masks" if (base / "masks").exists() else base / "mask"
    if img_dir.exists() and mask_dir.exists():
        return _CrackSourceDataset(img_dir, mask_dir, transform)
    return None


# ─── Public API ───────────────────────────────────────────────────────────────

def build_crack_dataset(
    root: str | Path,
    split: str = "train",
    transform: Callable | None = None,
) -> Dataset:
    """
    Build a merged crack segmentation dataset from all available sources.

    Args:
        root:      Path to the datasets/crack directory created by download_datasets.sh.
        split:     "train" | "val" | "test"
        transform: Callable(image, targets) → (image, targets)

    Returns:
        A ``torch.utils.data.Dataset`` (possibly a ``ConcatDataset`` of multiple sources).
    """
    root = Path(root)
    parts: list[Dataset] = []

    # CRACK500
    try:
        ds = _load_crack500(root, split, transform)
        parts.append(ds)
        print(f"[CrackDataset] CRACK500 ({split}): {len(ds)} samples")
    except FileNotFoundError as exc:
        print(f"[CrackDataset] CRACK500 not found — skipping. ({exc})")

    # DeepCrack (only train/test, no val split)
    if split != "val":
        ds = _load_deepcrack(root, split, transform)
        if ds is not None:
            parts.append(ds)
            print(f"[CrackDataset] DeepCrack ({split}): {len(ds)} samples")
        else:
            print("[CrackDataset] DeepCrack not found — skipping.")

    # khanhha merged (train only — large dataset)
    if split == "train":
        ds = _load_khanhha(root, split, transform)
        if ds is not None:
            parts.append(ds)
            print(f"[CrackDataset] khanhha merged (train): {len(ds)} samples")
        else:
            print("[CrackDataset] khanhha merged not found — skipping.")

    if not parts:
        raise RuntimeError(
            f"No crack dataset sources found under '{root}'. "
            "Run 'make download-data' first, or place datasets manually."
        )

    dataset = ConcatDataset(parts) if len(parts) > 1 else parts[0]
    print(f"[CrackDataset] Total ({split}): {len(dataset)} samples from {len(parts)} source(s)")
    return dataset
