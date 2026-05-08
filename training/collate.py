"""
collate.py — Custom DataLoader collate function for Mask2Former.

CRITICAL: Mask2Former's ``forward()`` expects:
  pixel_values : Tensor (B, C, H, W)
  mask_labels  : List[Tensor (N_i, H, W)]   ← variable N per image; must NOT be stacked
  class_labels : List[Tensor (N_i,)]         ← same

Using the default ``default_collate`` will fail because it tries to stack
tensors with different first dimensions.  This module provides a safe
``mask2former_collate_fn`` that handles the variable-length lists.

Samples returned by the dataset are expected to be:
    (image_tensor: Tensor (C, H, W), targets: dict)

where targets contains:
    mask_labels  : List[Tensor (N, H, W)] or empty list
    class_labels : Tensor (N,)            or empty tensor (shape (0,))
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def mask2former_collate_fn(batch: list[tuple[Tensor, dict[str, Any]]]) -> dict[str, Any]:
    """
    Collate a list of (image_tensor, targets) samples.

    Returns a dict with:
        pixel_values : Tensor  (B, C, H, W)
        mask_labels  : List[Tensor (N_i, H, W)]   length B
        class_labels : List[Tensor (N_i,)]         length B
    """
    images, mask_labels_list, class_labels_list = [], [], []

    for image, targets in batch:
        images.append(image)

        masks = targets.get("mask_labels", [])
        classes = targets.get("class_labels", torch.zeros(0, dtype=torch.long))

        if len(masks) == 0:
            # Image with no annotated instances — provide empty tensors.
            # Mask2Former can handle zero-instance images during training.
            h, w = image.shape[-2], image.shape[-1]
            mask_labels_list.append(torch.zeros((0, h, w), dtype=torch.bool))
            class_labels_list.append(torch.zeros((0,), dtype=torch.long))
        else:
            # masks may already be Tensors (from ToTensor transform) or numpy arrays.
            if isinstance(masks, list):
                stacked = torch.stack(
                    [m if isinstance(m, Tensor) else torch.from_numpy(m) for m in masks],
                    dim=0,
                )  # (N, H, W)
            else:
                stacked = masks  # already stacked Tensor

            mask_labels_list.append(stacked.bool())

            if isinstance(classes, Tensor):
                class_labels_list.append(classes.long())
            else:
                class_labels_list.append(torch.tensor(classes, dtype=torch.long))

    return {
        "pixel_values": torch.stack(images, dim=0),
        "mask_labels": mask_labels_list,
        "class_labels": class_labels_list,
    }
