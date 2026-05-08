#!/usr/bin/env bash
# Run DINO SSL pretraining on Sewer-ML frames.
# Usage: bash scripts/run_pretraining.sh [EXTRA_ARGS...]
#
# Requires Sewer-ML data at datasets/sewer_ml/
# Request via: https://forms.gle/hBaPtoweZumZAi4u9

set -euo pipefail

SEWER_ML_ROOT="${SEWER_ML_ROOT:-datasets/sewer_ml}"
BACKBONE="${BACKBONE:-facebook/dinov2-base}"
EPOCHS="${EPOCHS:-35}"
BATCH_SIZE="${BATCH_SIZE:-64}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ssl_pretraining}"

if [ ! -d "$SEWER_ML_ROOT" ]; then
    echo "ERROR: Sewer-ML directory not found at '$SEWER_ML_ROOT'."
    echo "  Request the dataset from: https://forms.gle/hBaPtoweZumZAi4u9"
    echo "  Then set SEWER_ML_ROOT=/path/to/sewer_ml and rerun."
    exit 1
fi

accelerate launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    training/train_ssl.py \
    --config configs/base.yaml \
    --sewer-ml-root "$SEWER_ML_ROOT" \
    --backbone "$BACKBONE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --output-dir "$OUTPUT_DIR" \
    "$@"
