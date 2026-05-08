#!/usr/bin/env bash
# Run Mask2Former segmentation fine-tuning.
# Usage: bash scripts/run_finetuning.sh [CONFIG_PATH] [EXTRA_ARGS...]
# Example: bash scripts/run_finetuning.sh configs/dinov2_mask2former_crack.yaml

set -euo pipefail

CONFIG="${1:-configs/dinov2_mask2former_crack.yaml}"
shift || true   # remaining args are passed through

# DGX Spark (GB10 Grace Blackwell) has a single GPU — single-process launch.
# Accelerate handles device placement and BF16 mixed precision.
accelerate launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    training/train_segmentation.py \
    --config "$CONFIG" \
    "$@"
