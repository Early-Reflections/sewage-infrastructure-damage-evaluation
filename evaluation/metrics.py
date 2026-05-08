"""
metrics.py — Evaluation metrics for sewer defect instance segmentation.

Provides:
  CocoInstanceMetrics  — wraps pycocotools for standard COCO AP/AR evaluation
  SewerMLMetrics       — F2CIW and F1Normal scores from the Sewer-ML benchmark
                         (computed from instance predictions mapped to image-level
                          classification labels, for comparison with Table 1 / Table 2)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as coco_mask_utils


# ─── COCO Instance Segmentation Metrics ──────────────────────────────────────

class CocoInstanceMetrics:
    """
    Accumulates Mask2Former predictions and ground-truth masks, then computes
    COCO-style AP and AR via pycocotools.

    Usage:
        metrics = CocoInstanceMetrics()
        for batch in dataloader:
            outputs = model(...)
            predictions = processor.post_process_instance_segmentation(
                outputs, target_sizes=target_sizes
            )
            metrics.update(predictions, ground_truth_batch)
        results = metrics.compute()
        print(results["AP"])
    """

    def __init__(self) -> None:
        self._gt_annotations: list[dict] = []
        self._dt_annotations: list[dict] = []
        self._gt_images: list[dict] = []
        self._gt_categories: list[dict] | None = None
        self._image_id = 0
        self._ann_id = 0

    def set_categories(self, id2label: dict[int, str]) -> None:
        self._gt_categories = [
            {"id": int(cid), "name": name} for cid, name in id2label.items()
        ]

    def update(
        self,
        predictions: list[dict],
        targets: list[dict],
    ) -> None:
        """
        Args:
            predictions: List of dicts per image (output of HF post_process_instance_segmentation):
                         {"segments_info": [{"id", "label_id", "score"}], "segmentation": Tensor H×W}
            targets: List of dicts per image:
                     {"mask_labels": List[Tensor (H, W)], "class_labels": Tensor (N,)}
        """
        for pred, target in zip(predictions, targets):
            self._image_id += 1
            img_id = self._image_id
            h, w = pred["segmentation"].shape if hasattr(pred["segmentation"], "shape") else (0, 0)
            self._gt_images.append({"id": img_id, "height": h, "width": w})

            # Ground truth
            for mask, label in zip(target["mask_labels"], target["class_labels"]):
                self._ann_id += 1
                binary = mask.numpy().astype(np.uint8) if isinstance(mask, torch.Tensor) else mask
                rle = coco_mask_utils.encode(np.asfortranarray(binary))
                rle["counts"] = rle["counts"].decode("utf-8")
                area = int(binary.sum())
                x, y, bw, bh = _rle_to_bbox(binary)
                self._gt_annotations.append(
                    {
                        "id": self._ann_id,
                        "image_id": img_id,
                        "category_id": int(label),
                        "segmentation": rle,
                        "area": area,
                        "bbox": [x, y, bw, bh],
                        "iscrowd": 0,
                    }
                )

            # Predictions
            seg_map = pred["segmentation"]
            if isinstance(seg_map, torch.Tensor):
                seg_map = seg_map.numpy()

            for seg_info in pred.get("segments_info", []):
                binary = (seg_map == seg_info["id"]).astype(np.uint8)
                if binary.sum() == 0:
                    continue
                rle = coco_mask_utils.encode(np.asfortranarray(binary))
                rle["counts"] = rle["counts"].decode("utf-8")
                area = int(binary.sum())
                x, y, bw, bh = _rle_to_bbox(binary)
                self._dt_annotations.append(
                    {
                        "image_id": img_id,
                        "category_id": int(seg_info["label_id"]),
                        "segmentation": rle,
                        "score": float(seg_info.get("score", 1.0)),
                        "area": area,
                        "bbox": [x, y, bw, bh],
                    }
                )

    def compute(self) -> dict[str, float]:
        """Return a dict with AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100."""
        if not self._gt_annotations:
            return {k: 0.0 for k in ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100"]}

        categories = self._gt_categories or _infer_categories(self._gt_annotations)

        gt_json = {
            "images": self._gt_images,
            "annotations": self._gt_annotations,
            "categories": categories,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(gt_json, f)
            gt_path = f.name

        coco_gt = COCO(gt_path)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self._dt_annotations, f)
            dt_path = f.name

        coco_dt = coco_gt.loadRes(dt_path)
        evaluator = COCOeval(coco_gt, coco_dt, iouType="segm")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

        Path(gt_path).unlink(missing_ok=True)
        Path(dt_path).unlink(missing_ok=True)

        stats = evaluator.stats
        return {
            "AP": float(stats[0]),
            "AP50": float(stats[1]),
            "AP75": float(stats[2]),
            "APs": float(stats[3]),
            "APm": float(stats[4]),
            "APl": float(stats[5]),
            "AR1": float(stats[6]),
            "AR10": float(stats[7]),
            "AR100": float(stats[8]),
        }

    def reset(self) -> None:
        self.__init__()


# ─── Sewer-ML Classification Metrics ─────────────────────────────────────────

# Class Importance Weights — Table 2, Haurum & Moeslund 2021.
_CIW: dict[str, float] = {
    "RB": 0.0660, "OB": 0.0524, "PF": 0.0994, "DE": 0.0587,
    "FS": 0.0997, "IS": 0.0403, "RO": 0.0575, "IN": 0.0873,
    "AF": 0.0944, "BE": 0.0340, "FO": 0.0423, "GR": 0.0296,
    "PH": 0.0616, "PB": 0.0830, "OS": 0.0993, "OP": 0.0943,
}

_DEFECT_CLASSES = list(_CIW.keys())


class SewerMLMetrics:
    """
    Compute F2CIW (defect recall-weighted F2) and F1Normal (non-defect F1).

    Intended for use when the model outputs image-level classification logits
    (e.g. during SSL linear probe or when mapping instance predictions back
    to image-level labels).

    Usage:
        metrics = SewerMLMetrics()
        metrics.update(predictions, targets)  # shape (B, C) multi-hot
        results = metrics.compute()
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._preds: list[np.ndarray] = []
        self._targets: list[np.ndarray] = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Args:
            preds  : (B, C) float tensor of sigmoid logits or probabilities.
            targets: (B, C) float tensor of binary ground-truth labels.
        """
        binary = (preds.sigmoid() >= self.threshold).cpu().numpy()
        self._preds.append(binary)
        self._targets.append(targets.cpu().numpy())

    def compute(self) -> dict[str, float]:
        if not self._preds:
            return {"F2CIW": 0.0, "F1Normal": 0.0}

        preds = np.concatenate(self._preds, axis=0)    # (N, C)
        targets = np.concatenate(self._targets, axis=0)  # (N, C)

        f2ciw = _compute_f2ciw(preds, targets)
        f1normal = _compute_f1normal(preds, targets)
        return {"F2CIW": f2ciw, "F1Normal": f1normal}

    def reset(self) -> None:
        self._preds = []
        self._targets = []


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _fbeta(tp: float, fp: float, fn: float, beta: float) -> float:
    denom = (1 + beta ** 2) * tp + beta ** 2 * fn + fp
    return (1 + beta ** 2) * tp / denom if denom > 0 else 0.0


def _compute_f2ciw(preds: np.ndarray, targets: np.ndarray) -> float:
    """Weighted F2 over defect classes using CIW weights."""
    # Assume column order matches _DEFECT_CLASSES (16 defect columns).
    if preds.shape[1] < len(_DEFECT_CLASSES):
        return 0.0
    total_weight = sum(_CIW.values())
    f2ciw = 0.0
    for i, cls in enumerate(_DEFECT_CLASSES):
        tp = float(np.sum((preds[:, i] == 1) & (targets[:, i] == 1)))
        fp = float(np.sum((preds[:, i] == 1) & (targets[:, i] == 0)))
        fn = float(np.sum((preds[:, i] == 0) & (targets[:, i] == 1)))
        f2 = _fbeta(tp, fp, fn, beta=2.0)
        f2ciw += _CIW[cls] / total_weight * f2
    return f2ciw * 100.0


def _compute_f1normal(preds: np.ndarray, targets: np.ndarray) -> float:
    """F1 for the 'Normal' (no defect) class — first column."""
    tp = float(np.sum((preds[:, 0] == 1) & (targets[:, 0] == 1)))
    fp = float(np.sum((preds[:, 0] == 1) & (targets[:, 0] == 0)))
    fn = float(np.sum((preds[:, 0] == 0) & (targets[:, 0] == 1)))
    return _fbeta(tp, fp, fn, beta=1.0) * 100.0


def _infer_categories(annotations: list[dict]) -> list[dict]:
    ids = sorted({a["category_id"] for a in annotations})
    return [{"id": i, "name": str(i)} for i in ids]


def _rle_to_bbox(binary: np.ndarray) -> tuple[int, int, int, int]:
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    if not rows.any():
        return 0, 0, 0, 0
    rmin, rmax = int(np.where(rows)[0][[0, -1]])
    cmin, cmax = int(np.where(cols)[0][[0, -1]])
    return cmin, rmin, cmax - cmin + 1, rmax - rmin + 1
