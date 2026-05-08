# Sewage Infrastructure Damage Evaluation

## CV Model Training

This repository contains a full training scaffold for instance segmentation of sewer/pipe defects using Mask2Former with DINOv2 backbones, optimized for NVIDIA DGX Spark (GB10 Grace Blackwell, ARM64, CUDA 13).

- Supports fine-tuning on public crack segmentation datasets and the CSDD sewer dataset (request access required)
- Optional DINO SSL pretraining on Sewer-ML (classification-only, request access)
- Pure PyTorch/HuggingFace implementation — no custom CUDA ops, runs out-of-the-box on ARM64

See configs/ and data/ for dataset setup instructions.
