"""
SewerMLDataset — PyTorch Dataset for DINO SSL pretraining on Sewer-ML.

Reference: Haurum & Moeslund, CVPR 2021 (arXiv:2103.10895)
Data access: https://forms.gle/hBaPtoweZumZAi4u9

Expected directory layout after receiving the data:
    datasets/sewer_ml/
        images/        ← JPEG frames (flat directory or subdirectories)
        annotations/
            Train.csv
            Valid.csv
            Test.csv

CSV format (one row per image):
    Filename,Normal,RB,OB,PF,DE,FS,IS,RO,IN,AF,BE,FO,GR,PH,PB,OS,OP,OK
    path/to/img.jpg,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

For SSL (DINO) pretraining the labels are not used — only the images are needed.
Two independent random augmented views of each image are returned as a pair
(view1, view2), which is the input format for the DINO self-distillation loss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


# Defect class names in column order (excluding 'Filename').
SEWER_ML_CLASSES = [
    "Normal", "RB", "OB", "PF", "DE", "FS", "IS", "RO",
    "IN", "AF", "BE", "FO", "GR", "PH", "PB", "OS", "OP", "OK",
]

# Class Importance Weights (CIW) from Haurum & Moeslund 2021 (Table 2).
# Used for the F2CIW metric — stored here for reference.
CLASS_IMPORTANCE_WEIGHTS: dict[str, float] = {
    "RB": 0.0660, "OB": 0.0524, "PF": 0.0994, "DE": 0.0587,
    "FS": 0.0997, "IS": 0.0403, "RO": 0.0575, "IN": 0.0873,
    "AF": 0.0944, "BE": 0.0340, "FO": 0.0423, "GR": 0.0296,
    "PH": 0.0616, "PB": 0.0830, "OS": 0.0993, "OP": 0.0943,
    "OK": 0.0000,  # No defect class — not used in F2CIW
}


class SewerMLDataset(Dataset):
    """
    Sewer-ML dataset loader.

    In SSL mode (``ssl_views=True``, default), returns a tuple of two
    independently augmented views (view1, view2) for DINO pretraining.

    In classification mode (``ssl_views=False``), returns (image, label_vector)
    where label_vector is a float32 tensor of length 18 (binary multi-label).

    Args:
        root        : Path to datasets/sewer_ml/.
        split       : "Train" | "Valid" | "Test".
        transform   : Augmentation callable for a single view (PIL → Tensor).
        ssl_views   : If True, apply transform twice independently for SSL.
        max_samples : Optional cap for fast debugging.
    """

    _SPLIT_FILES = {"Train": "Train.csv", "Valid": "Valid.csv", "Test": "Test.csv"}

    def __init__(
        self,
        root: str | Path,
        split: str = "Train",
        transform: Callable | None = None,
        ssl_views: bool = True,
        max_samples: int | None = None,
    ) -> None:
        if split not in self._SPLIT_FILES:
            raise ValueError(f"split must be one of {list(self._SPLIT_FILES)}; got '{split}'")

        self.root = Path(root)
        self.img_dir = self.root / "images"
        self.ssl_views = ssl_views
        self.transform = transform

        ann_path = self.root / "annotations" / self._SPLIT_FILES[split]
        if not ann_path.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {ann_path}\n"
                "Request Sewer-ML from https://forms.gle/hBaPtoweZumZAi4u9"
            )

        df = pd.read_csv(ann_path)
        if max_samples is not None:
            df = df.head(max_samples)
        self.df = df.reset_index(drop=True)

        # Resolve the filename column (first column regardless of name)
        self._filename_col = df.columns[0]

        print(
            f"[SewerMLDataset] split={split}, "
            f"ssl_views={ssl_views}, "
            f"samples={len(self.df)}"
        )

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, idx: int) -> Image.Image:
        rel_path = self.df.at[idx, self._filename_col]
        img_path = self.img_dir / rel_path
        return Image.open(img_path).convert("RGB")

    def __getitem__(self, idx: int):
        image = self._load_image(idx)

        if self.ssl_views:
            # Return two independently augmented views for DINO self-distillation.
            view1 = self.transform(image) if self.transform else image
            view2 = self.transform(image) if self.transform else image
            return view1, view2

        # Classification mode: return image + multi-hot label vector.
        import torch
        label_cols = [c for c in SEWER_ML_CLASSES if c in self.df.columns]
        labels = torch.tensor(
            self.df.loc[idx, label_cols].values.astype("float32"),
            dtype=torch.float32,
        )
        img_tensor = self.transform(image) if self.transform else image
        return img_tensor, labels
