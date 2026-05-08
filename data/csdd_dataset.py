"""
CSDDDataset — PyTorch Dataset for the Culvert-Sewer Defects Dataset (CSDD).

Reference: arXiv:2312.14053  (Ferdaus et al., UNO / USACE, 2023)
Data access: mferdaus@uno.edu

Expected directory layout after receiving the data:
    datasets/csdd/
        images/   ← RGB CCTV frames (.jpg or .png)
        masks/    ← PNG semantic masks (uint8), one file per image

Mask encoding: Each pixel value encodes a class id.
Background (unlabeled) = 0, defect classes = 1..9 (as defined in the config).

This module converts semantic segmentation masks to instance segmentation
annotations by running connected-component labelling on each class layer.
Each connected component becomes one instance.  This is a common approach when
only semantic labels are available and is sufficient to train Mask2Former for
instance segmentation.

Targets dict returned per sample:
    mask_labels  : List[np.ndarray shape (H, W)]   – binary, one per instance
    class_labels : List[int]
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from skimage import measure
from torch.utils.data import Dataset, Subset


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
BACKGROUND_ID = 0


# ─── Connected-component instance extraction ──────────────────────────────────

def semantic_mask_to_instances(
    semantic_mask: np.ndarray,
    ignore_classes: frozenset[int] = frozenset({BACKGROUND_ID}),
    min_area: int = 64,
) -> tuple[list[np.ndarray], list[int]]:
    """
    Convert a per-pixel semantic mask to a list of binary instance masks.

    For each foreground class, connected components are extracted via
    ``skimage.measure.label``.  Small components below ``min_area`` pixels
    are discarded.

    Args:
        semantic_mask : H×W uint8 array of class ids.
        ignore_classes: Class ids to skip (background by default).
        min_area      : Minimum pixel area to keep a component.

    Returns:
        (mask_labels, class_labels)
        mask_labels  : List of bool H×W arrays, one per instance.
        class_labels : Corresponding class id (int) for each instance.
    """
    instance_masks: list[np.ndarray] = []
    class_ids: list[int] = []

    for class_id in np.unique(semantic_mask):
        if int(class_id) in ignore_classes:
            continue
        class_layer = (semantic_mask == class_id).astype(np.uint8)
        labelled = measure.label(class_layer, connectivity=2)
        for region in measure.regionprops(labelled):
            if region.area < min_area:
                continue
            binary_mask = (labelled == region.label).astype(np.uint8)
            instance_masks.append(binary_mask)
            class_ids.append(int(class_id))

    return instance_masks, class_ids


# ─── Dataset ──────────────────────────────────────────────────────────────────

class CSDDDataset(Dataset):
    """
    CSDD instance segmentation dataset.

    Args:
        root      : Path to datasets/csdd/ (contains images/ and masks/).
        transform : Callable(image: PIL.Image, targets: dict) → (image, targets).
        min_area  : Minimum connected-component pixel area to keep.
    """

    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
        min_area: int = 64,
    ) -> None:
        self.root = Path(root)
        self.img_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        self.transform = transform
        self.min_area = min_area

        if not self.img_dir.exists():
            raise FileNotFoundError(
                f"CSDD images directory not found: {self.img_dir}\n"
                "Request the dataset from mferdaus@uno.edu (arXiv:2312.14053)."
            )
        if not self.mask_dir.exists():
            raise FileNotFoundError(
                f"CSDD masks directory not found: {self.mask_dir}"
            )

        self.samples = self._collect_samples()
        if not self.samples:
            raise FileNotFoundError(
                f"No image/mask pairs found under {self.root}."
            )
        print(f"[CSDDDataset] Loaded {len(self.samples)} samples from {self.root}")

    def _collect_samples(self) -> list[tuple[Path, Path]]:
        pairs = []
        for img_path in sorted(self.img_dir.iterdir()):
            if img_path.suffix.lower() not in _IMG_EXTS:
                continue
            mask_path = self.mask_dir / (img_path.stem + ".png")
            if not mask_path.exists():
                mask_path = self.mask_dir / img_path.name
            if mask_path.exists():
                pairs.append((img_path, mask_path))
        return pairs

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        semantic_mask = np.array(Image.open(mask_path))

        mask_labels, class_labels = semantic_mask_to_instances(
            semantic_mask, min_area=self.min_area
        )
        # If there are no instances after filtering, return an empty annotation
        # so the collate function can handle it gracefully.
        targets = {"mask_labels": mask_labels, "class_labels": class_labels}

        if self.transform is not None:
            image, targets = self.transform(image, targets)
        return image, targets


# ─── Public API ───────────────────────────────────────────────────────────────

def build_csdd_splits(
    root: str | Path,
    train_transform: Callable | None = None,
    eval_transform: Callable | None = None,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    seed: int = 42,
    min_area: int = 64,
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Build train / val / test splits from a flat CSDD directory.

    The base dataset is loaded with ``eval_transform`` for val/test, and with
    ``train_transform`` for the training split.  Splitting is done by index
    permutation with a fixed seed for reproducibility.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    rng = random.Random(seed)
    full_indices = list(range(len(CSDDDataset(root, min_area=min_area))))
    rng.shuffle(full_indices)

    n = len(full_indices)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = full_indices[:n_train]
    val_idx = full_indices[n_train : n_train + n_val]
    test_idx = full_indices[n_train + n_val :]

    train_ds = CSDDDataset(root, transform=train_transform, min_area=min_area)
    eval_ds = CSDDDataset(root, transform=eval_transform, min_area=min_area)

    return (
        Subset(train_ds, train_idx),
        Subset(eval_ds, val_idx),
        Subset(eval_ds, test_idx),
    )
