#!/usr/bin/env bash
# Run evaluation on the test split.
# Usage: bash scripts/run_evaluation.sh [CONFIG_PATH] [CHECKPOINT_PATH] [SPLIT]

set -euo pipefail

CONFIG="${1:-configs/dinov2_mask2former_crack.yaml}"
CHECKPOINT="${2:-outputs/crack/best/hf_model}"
SPLIT="${3:-test}"

if [ ! -d "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint directory not found: '$CHECKPOINT'"
    echo "  Run fine-tuning first: make train-seg"
    exit 1
fi

python evaluation/evaluate.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --split "$SPLIT"
