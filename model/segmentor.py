"""
segmentor.py — Mask2Former model construction with a DINOv2 backbone.

Public API:
    build_model(cfg)  → Mask2FormerForUniversalSegmentation
    load_dinov2_weights(model, backbone_name)  → None  (loads pretrained backbone weights)
"""

from __future__ import annotations

import logging

import torch
from transformers import (
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
)

from model.backbone import build_dinov2_backbone_config

logger = logging.getLogger(__name__)


def build_model(cfg) -> Mask2FormerForUniversalSegmentation:
    """
    Instantiate Mask2FormerForUniversalSegmentation with a DINOv2 backbone.

    The model is built from scratch (random weights) for the Mask2Former head
    and pixel decoder.  DINOv2 backbone weights are loaded from the Hub
    separately via ``load_dinov2_weights``.

    Args:
        cfg: OmegaConf config with at minimum:
               cfg.model.backbone_name   (str)
               cfg.model.num_queries     (int)
               cfg.model.hidden_dim      (int)
               cfg.model.encoder_layers  (int)
               cfg.model.decoder_layers  (int)
               cfg.data.id2label         (dict  int → str)

    Returns:
        Mask2FormerForUniversalSegmentation
    """
    id2label: dict[int, str] = {int(k): v for k, v in cfg.data.id2label.items()}
    label2id: dict[str, int] = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)

    backbone_config = build_dinov2_backbone_config(cfg.model.backbone_name)

    m2f_config = Mask2FormerConfig(
        backbone_config=backbone_config,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        # Transformer decoder
        num_queries=cfg.model.num_queries,
        hidden_dim=cfg.model.hidden_dim,
        encoder_layers=cfg.model.encoder_layers,
        decoder_layers=cfg.model.decoder_layers,
        num_attention_heads=8,
        # Pixel decoder
        feature_strides=list(cfg.model.feature_strides),
        enforce_input_projection=cfg.model.get("enforce_input_projection", True),
        # Loss weights (Mask2Former defaults work well for instance seg)
        class_weight=2.0,
        mask_weight=5.0,
        dice_weight=5.0,
        # Task type
        use_auxiliary_loss=True,
    )

    model = Mask2FormerForUniversalSegmentation(m2f_config)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(
        "Built Mask2Former with DINOv2 backbone '%s': %.1f M parameters",
        cfg.model.backbone_name,
        n_params,
    )
    return model


def load_dinov2_weights(
    model: Mask2FormerForUniversalSegmentation,
    backbone_name: str,
) -> None:
    """
    Load pretrained DINOv2 weights into the backbone of a Mask2Former model.

    Only the backbone parameters are updated; the Mask2Former pixel decoder
    and transformer decoder are left with their randomly initialised weights.

    Args:
        model:         A Mask2Former model built with a Dinov2Config backbone.
        backbone_name: HuggingFace Hub id, e.g. "facebook/dinov2-large".
    """
    from transformers import Dinov2Model

    logger.info("Loading pretrained DINOv2 weights from '%s' ...", backbone_name)
    pretrained = Dinov2Model.from_pretrained(backbone_name)

    # HF Mask2Former stores the backbone under model.model.pixel_level_module.encoder
    # The exact attribute path depends on the HF version; find it dynamically.
    backbone = _find_backbone(model)
    if backbone is None:
        raise RuntimeError(
            "Could not locate the DINOv2 backbone sub-module inside the "
            "Mask2Former model.  Check transformers version compatibility."
        )

    missing, unexpected = backbone.load_state_dict(
        pretrained.state_dict(), strict=False
    )

    if missing:
        logger.warning(
            "Missing keys when loading DINOv2 weights (%d): %s ...",
            len(missing),
            missing[:5],
        )
    if unexpected:
        logger.warning(
            "Unexpected keys when loading DINOv2 weights (%d): %s ...",
            len(unexpected),
            unexpected[:5],
        )
    logger.info(
        "DINOv2 weights loaded. Missing=%d, Unexpected=%d", len(missing), len(unexpected)
    )


def _find_backbone(model: torch.nn.Module) -> torch.nn.Module | None:
    """Walk the module tree to find the ViT/DINOv2 sub-module."""
    # Try known HF Transformers attribute paths first.
    candidate_paths = [
        "model.pixel_level_module.encoder",
        "pixel_level_module.encoder",
        "backbone",
        "encoder",
    ]
    for path in candidate_paths:
        try:
            sub = model
            for attr in path.split("."):
                sub = getattr(sub, attr)
            return sub
        except AttributeError:
            continue

    # Fallback: return the first sub-module whose class name contains "Dinov2"
    for _, module in model.named_modules():
        if "Dinov2" in type(module).__name__:
            return module

    return None
