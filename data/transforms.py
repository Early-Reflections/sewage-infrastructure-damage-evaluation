"""
Shared image augmentation pipeline.

All transforms operate on (PIL.Image, dict) pairs where the dict carries:
  mask_labels  : List[np.ndarray]  – binary H×W masks, one per instance
  class_labels : List[int]         – class id for each instance mask

The final step, ``HFProcessorTransform``, converts the augmented PIL image
and instance masks into the dict expected by
``Mask2FormerForUniversalSegmentation``:
  pixel_values : Tensor (C, H, W)
  mask_labels  : List[Tensor (N, H, W)] per image (one element = one image)
  class_labels : List[Tensor (N,)]      per image
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision.transforms import functional as TF


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _flip_masks(masks: list[np.ndarray], horizontal: bool) -> list[np.ndarray]:
    if not horizontal:
        return masks
    return [np.fliplr(m).copy() for m in masks]


def _resize_masks(
    masks: list[np.ndarray],
    size: tuple[int, int],
) -> list[np.ndarray]:
    """Nearest-neighbour resize for binary masks to avoid interpolation artefacts."""
    return [
        np.array(Image.fromarray(m.astype(np.uint8)).resize(size[::-1], Image.NEAREST))
        for m in masks
    ]


# ─── Transform classes ────────────────────────────────────────────────────────

class RandomHorizontalFlip:
    def __init__(self, prob: float = 0.5) -> None:
        self.prob = prob

    def __call__(
        self, image: Image.Image, targets: dict[str, Any]
    ) -> tuple[Image.Image, dict[str, Any]]:
        if random.random() < self.prob:
            image = TF.hflip(image)
            targets["mask_labels"] = _flip_masks(targets["mask_labels"], horizontal=True)
        return image, targets


class RandomResize:
    """Scale the shorter side to a random length in [min_size, max_size]."""

    def __init__(self, min_size: int = 384, max_size: int = 640) -> None:
        self.min_size = min_size
        self.max_size = max_size

    def __call__(
        self, image: Image.Image, targets: dict[str, Any]
    ) -> tuple[Image.Image, dict[str, Any]]:
        w, h = image.size
        scale = random.uniform(self.min_size, self.max_size) / min(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        image = image.resize((new_w, new_h), Image.BILINEAR)
        targets["mask_labels"] = _resize_masks(targets["mask_labels"], (new_h, new_w))
        return image, targets


class FixedResize:
    def __init__(self, size: tuple[int, int]) -> None:
        # size = (H, W)
        self.size = size

    def __call__(
        self, image: Image.Image, targets: dict[str, Any]
    ) -> tuple[Image.Image, dict[str, Any]]:
        h, w = self.size
        image = image.resize((w, h), Image.BILINEAR)
        targets["mask_labels"] = _resize_masks(targets["mask_labels"], self.size)
        return image, targets


class RandomCrop:
    def __init__(self, size: tuple[int, int]) -> None:
        # size = (H, W)
        self.size = size

    def __call__(
        self, image: Image.Image, targets: dict[str, Any]
    ) -> tuple[Image.Image, dict[str, Any]]:
        crop_h, crop_w = self.size
        img_w, img_h = image.size
        top = random.randint(0, max(img_h - crop_h, 0))
        left = random.randint(0, max(img_w - crop_w, 0))
        image = TF.crop(image, top, left, crop_h, crop_w)
        targets["mask_labels"] = [
            m[top : top + crop_h, left : left + crop_w] for m in targets["mask_labels"]
        ]
        return image, targets


class ColorJitter:
    def __init__(
        self,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.1,
    ) -> None:
        from torchvision import transforms
        self._jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(
        self, image: Image.Image, targets: dict[str, Any]
    ) -> tuple[Image.Image, dict[str, Any]]:
        image = self._jitter(image)
        return image, targets


class GaussianBlur:
    """Apply Gaussian blur with probability p."""

    def __init__(self, prob: float = 0.3, radius_range: tuple[float, float] = (0.1, 2.0)) -> None:
        self.prob = prob
        self.radius_range = radius_range

    def __call__(
        self, image: Image.Image, targets: dict[str, Any]
    ) -> tuple[Image.Image, dict[str, Any]]:
        if random.random() < self.prob:
            radius = random.uniform(*self.radius_range)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        return image, targets


class FilterEmptyInstances:
    """Remove instances whose mask has been cropped/resized away."""

    def __init__(self, min_area: int = 32) -> None:
        self.min_area = min_area

    def __call__(
        self, image: Image.Image, targets: dict[str, Any]
    ) -> tuple[Image.Image, dict[str, Any]]:
        kept_masks, kept_labels = [], []
        for mask, label in zip(targets["mask_labels"], targets["class_labels"]):
            if mask.sum() >= self.min_area:
                kept_masks.append(mask)
                kept_labels.append(label)
        targets["mask_labels"] = kept_masks
        targets["class_labels"] = kept_labels
        return image, targets


class ToTensor:
    """Convert numpy masks to torch tensors and image to float tensor."""

    def __call__(
        self, image: Image.Image, targets: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        image_tensor = TF.to_tensor(image)  # (C, H, W), float [0,1]
        targets["mask_labels"] = [
            torch.from_numpy(m.astype(np.bool_)) for m in targets["mask_labels"]
        ]
        targets["class_labels"] = torch.tensor(targets["class_labels"], dtype=torch.long)
        return image_tensor, targets


class Normalize:
    """ImageNet-style normalisation (matches DINOv2 preprocessing)."""

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __call__(
        self, image: torch.Tensor, targets: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        image = TF.normalize(image, mean=self.MEAN, std=self.STD)
        return image, targets


# ─── Composed pipelines ───────────────────────────────────────────────────────

class Compose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(
        self, image: Image.Image, targets: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        for t in self.transforms:
            image, targets = t(image, targets)
        return image, targets


def build_train_transforms(cfg) -> Compose:
    size = tuple(cfg.data.image_size)  # (H, W)
    jitter = cfg.data.get("color_jitter", 0.4)
    return Compose(
        [
            RandomResize(min_size=int(size[0] * 0.75), max_size=int(size[0] * 1.25)),
            RandomCrop(size),
            RandomHorizontalFlip(prob=cfg.data.get("flip_prob", 0.5)),
            ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter),
            GaussianBlur(prob=0.3),
            FilterEmptyInstances(min_area=cfg.data.get("min_instance_area", 32)),
            ToTensor(),
            Normalize(),
        ]
    )


def build_eval_transforms(cfg) -> Compose:
    size = tuple(cfg.data.image_size)  # (H, W)
    return Compose(
        [
            FixedResize(size),
            FilterEmptyInstances(min_area=cfg.data.get("min_instance_area", 32)),
            ToTensor(),
            Normalize(),
        ]
    )
