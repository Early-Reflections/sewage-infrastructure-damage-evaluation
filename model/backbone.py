"""
backbone.py — DINOv2 backbone configuration helpers for HuggingFace Mask2Former.

``Mask2FormerConfig`` accepts a ``backbone_config`` parameter.  Passing a
``Dinov2Config`` here causes the Mask2Former pixel decoder to receive
multi-scale feature maps from different DINOv2 ViT layers via the
``out_features`` / ``out_indices`` mechanism introduced in HF Transformers ≥4.37.

Mask2Former's pixel decoder (MSDeformAttnPixelDecoder) expects 4 feature
levels.  Plain ViT only has one output scale, so we expose 4 intermediate
block outputs and use the built-in ``enforce_input_projection=True`` in
Mask2Former to project them all to ``hidden_dim``.

Layer indices for DINOv2 variants (0-indexed):
    ViT-S/14 (12 blocks): out_indices = [2, 5, 8, 11]
    ViT-B/14 (12 blocks): out_indices = [2, 5, 8, 11]
    ViT-L/14 (24 blocks): out_indices = [5, 11, 17, 23]
    ViT-g/14 (40 blocks): out_indices = [9, 19, 29, 39]
"""

from __future__ import annotations

from transformers import Dinov2Config

# ─── Layer indices per model ──────────────────────────────────────────────────

_OUT_INDICES: dict[str, list[int]] = {
    "facebook/dinov2-small":  [2, 5, 8, 11],
    "facebook/dinov2-base":   [2, 5, 8, 11],
    "facebook/dinov2-large":  [5, 11, 17, 23],
    "facebook/dinov2-giant":  [9, 19, 29, 39],
}

_OUT_FEATURES: dict[str, list[str]] = {
    name: [f"stage{i}" for i in indices]
    for name, indices in _OUT_INDICES.items()
}


def build_dinov2_backbone_config(backbone_name: str) -> Dinov2Config:
    """
    Return a ``Dinov2Config`` ready for use as the backbone of Mask2Former.

    The ``out_features`` and ``out_indices`` fields are set so that HF's
    ``Dinov2BackboneWithProjection`` (used internally by Mask2Former) emits
    4 feature maps at different spatial scales.

    Args:
        backbone_name: HuggingFace Hub id, e.g. "facebook/dinov2-large".

    Returns:
        A ``Dinov2Config`` instance.
    """
    if backbone_name not in _OUT_INDICES:
        raise ValueError(
            f"Unknown DINOv2 backbone '{backbone_name}'. "
            f"Supported: {list(_OUT_INDICES)}"
        )

    config = Dinov2Config.from_pretrained(backbone_name)

    # Tell HF to expose intermediate block outputs as named stages.
    config.out_features = _OUT_FEATURES[backbone_name]
    config.out_indices = _OUT_INDICES[backbone_name]

    # Reshape is not applied inside Mask2Former's backbone wrapper — keep off.
    config.apply_layernorm = True
    config.reshape_hidden_states = False  # Mask2Former handles spatial reshape

    return config
