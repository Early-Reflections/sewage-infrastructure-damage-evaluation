#!/usr/bin/env bash
# Download public crack segmentation datasets used as proxy pretraining data.
# Usage: bash data/download_datasets.sh [TARGET_DIR]
# Default TARGET_DIR: datasets/crack

set -euo pipefail

TARGET_DIR="${1:-datasets/crack}"
mkdir -p "$TARGET_DIR"

cd "$TARGET_DIR"

echo "=== Downloading DeepCrack dataset ==="
# Liu et al., Neurocomputing 2019  |  github.com/yhlleo/DeepCrack
if [ ! -d "deepcrack" ]; then
    mkdir -p deepcrack
    wget -q --show-progress -O deepcrack.zip \
        "https://github.com/yhlleo/DeepCrack/releases/download/dataset/DeepCrack.zip" \
        || { echo "DeepCrack direct download failed. Visit https://github.com/yhlleo/DeepCrack and download manually to $TARGET_DIR/deepcrack/"; }
    if [ -f "deepcrack.zip" ]; then
        unzip -q deepcrack.zip -d deepcrack_raw
        mv deepcrack_raw deepcrack
        rm -f deepcrack.zip
        echo "DeepCrack: OK"
    fi
else
    echo "DeepCrack: already present, skipping."
fi

echo ""
echo "=== Downloading CRACK500 dataset ==="
# Yang et al., T-ITS 2019  |  github.com/fyangneil/pavement-crack-detection
# Files hosted on Google Drive; gdown is required (installed in container).
if [ ! -d "crack500" ]; then
    mkdir -p crack500
    # Train set
    gdown --fuzzy "https://drive.google.com/file/d/1hbzFOSBpHujzHGs971LGDgivqXiALgHg" \
        -O crack500/train.zip || echo "CRACK500 train: manual download needed."
    # Test set
    gdown --fuzzy "https://drive.google.com/file/d/1XAet0webpaWKH4LVHqx1KBHeAsFCsVFh" \
        -O crack500/test.zip  || echo "CRACK500 test: manual download needed."
    for z in crack500/*.zip; do
        [ -f "$z" ] && unzip -q "$z" -d crack500/ && rm -f "$z"
    done
    echo "CRACK500: OK"
else
    echo "CRACK500: already present, skipping."
fi

echo ""
echo "=== Downloading khanhha merged crack dataset (~11,200 images) ==="
# github.com/khanhha/crack_segmentation  |  ~2.7 GB
if [ ! -d "khanhha" ]; then
    mkdir -p khanhha
    gdown --fuzzy "https://drive.google.com/file/d/1xrOqv0-3uMHjZyEUrerOYiYXW_E8SUMP" \
        -O khanhha/dataset.zip || echo "khanhha dataset: manual download needed from github.com/khanhha/crack_segmentation"
    if [ -f "khanhha/dataset.zip" ]; then
        unzip -q khanhha/dataset.zip -d khanhha/
        rm -f khanhha/dataset.zip
        echo "khanhha: OK"
    fi
else
    echo "khanhha: already present, skipping."
fi

echo ""
echo "=== Done ==="
echo "Datasets stored in: $(pwd)"
echo ""
echo "CSDD (Culvert-Sewer Defects Dataset) must be requested from the authors:"
echo "  Contact: mferdaus@uno.edu  (Md Meftahul Ferdaus, University of New Orleans)"
echo "  Reference: arXiv:2312.14053"
echo "  Once received, place at: datasets/csdd/{images/,masks/}"
echo ""
echo "SEWER-ML (1.3M images, classification labels for SSL pretraining):"
echo "  Request via: https://forms.gle/hBaPtoweZumZAi4u9"
echo "  Once received, place at: datasets/sewer_ml/"
