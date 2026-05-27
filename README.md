# Sewage Infrastructure Damage Evaluation (SIDE) AI System

## CV Model Training

This repository contains a full training scaffold for instance segmentation of sewer/pipe defects using Mask2Former with DINOv2 backbones, optimized for NVIDIA DGX Spark (GB10 Grace Blackwell, ARM64, CUDA 13).

- Supports fine-tuning on public crack segmentation datasets and the CSDD sewer dataset (request access required)
- Optional DINO SSL pretraining on Sewer-ML (classification-only, request access)
- Pure PyTorch/HuggingFace implementation — no custom CUDA ops, runs out-of-the-box on ARM64

### On DGX Spark inside ~/side-train:
- make build
- make download-data
- SEWER_ML_ROOT=/path/to/sewer_ml make train-ssl
- [Without SSL -> Meta's public dinov2 weights]: make train-seg
- [With SSL] make train-seg CONFIG=configs/dinov2_mask2former_csdd.yaml -> but first edit the config's backbone_name to point to the local path: backbone_name: "outputs/ssl_pretraining/student_backbone"
- make eval
