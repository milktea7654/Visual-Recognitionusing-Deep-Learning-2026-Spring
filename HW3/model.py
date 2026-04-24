"""
model.py – Mask2Former model wrapper for 4-class medical cell segmentation.

Uses Hugging Face ``transformers`` Mask2Former with a ConvNeXt-V2-Base backbone.

Weight initialisation policy
-----------------------------
* **Backbone (ConvNeXt-V2-Base)**: ImageNet-1K pretrained weights loaded from
  ``facebook/convnextv2-base-1k-224``.
* **Pixel Decoder, Transformer Decoder, class/mask heads**: randomly
  initialised – no COCO or any other external-dataset weights used.

This satisfies the homework constraint that only ImageNet-1K pretrained
weights are permitted.

Architecture overview
---------------------
ConvNeXt-V2-Base backbone → Pixel Decoder (multi-scale deformable attention) →
Transformer Decoder (N queries) → per-query class + binary mask heads.

Why ConvNeXt-V2-Base over V1:
  * Global Response Normalization (GRN) prevents feature collapse across
    channels, giving richer multi-scale representations.
  * Same parameter count (~89 M backbone), same API, drop-in replacement.
  * ImageNet-1K top-1: 86.8 % vs 85.8 % for V1-Base.
"""

from __future__ import annotations

from types import MethodType
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import (
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
)


# ---------------------------------------------------------------------------
# Class labels (0 = no-object / background, 1-4 = cell classes)
# ---------------------------------------------------------------------------

NUM_CLASSES = 4  # foreground classes; the model adds 1 for the "no-object" cls


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_mask2former(
    pretrained: bool = True,
    num_classes: int = NUM_CLASSES,
    num_queries: int = 100,
) -> Mask2FormerForUniversalSegmentation:
    """
    Instantiate a Mask2Former with a ConvNeXt-Base backbone.

    The full architecture is **always built from scratch** (no COCO weights).
    When ``pretrained=True``, only the ConvNeXt-Base backbone is loaded with
    ImageNet-1K weights from ``facebook/convnext-base-224``.
    The Pixel Decoder, Transformer Decoder, and all heads remain randomly
    initialised.

    Args:
        pretrained:  If True, load ImageNet-1K backbone weights (permitted).
                     If False, fully random initialisation.
        num_classes: Number of *foreground* classes.
        num_queries: Number of object queries.

    Returns:
        model: ready-to-train ``Mask2FormerForUniversalSegmentation``.
    """
    m2f_config = _build_config(
        num_classes=num_classes, num_queries=num_queries
    )
    model = Mask2FormerForUniversalSegmentation(m2f_config)

    if pretrained:
        _load_imagenet_backbone(model)

    _patch_mask_decoder(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[model] trainable parameters: {n_params / 1e6:.1f} M  "
        f"(limit: 200 M)"
    )
    assert n_params < 200e6, (
        f"Model has {n_params / 1e6:.1f} M params – exceeds 200 M cap!"
    )

    return model


def _patch_mask_decoder(
    model: Mask2FormerForUniversalSegmentation,
) -> None:
    """
    Patch the decoder to rebuild the masked-attention map for the current
    feature level right before each cross-attention call.

    Some transformers builds only officially support Swin backbones for
    Mask2Former. With ConvNeXt, the stock decoder can end up feeding an
    attention mask sized for the next feature level into the current level's
    cross-attention, which triggers a source-length mismatch on rectangular
    images.

    This patch keeps the architecture unchanged and only makes the decoder's
    mask scheduling explicit: for decoder layer ``idx``, the cross-attention
    mask is always generated with ``feature_size_list[idx % num_feature_levels]``.
    """
    from transformers.models.mask2former.modeling_mask2former import (
        Mask2FormerMaskedAttentionDecoderOutput,
    )

    decoder = model.model.transformer_module.decoder

    def _forward_patched(
        self,
        inputs_embeds: torch.Tensor,
        multi_stage_positional_embeddings: list,
        pixel_embeddings: torch.Tensor,
        encoder_hidden_states: list,
        query_position_embeddings: Optional[torch.Tensor] = None,
        feature_size_list: Optional[list] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
    ):
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        intermediate = ()
        attentions = () if output_attentions else None
        intermediate_mask_predictions = ()

        intermediate_hidden_states = self.layernorm(inputs_embeds)
        intermediate += (intermediate_hidden_states,)

        predicted_mask, _ = self.mask_predictor(
            intermediate_hidden_states,
            pixel_embeddings,
            feature_size_list[0],
        )
        intermediate_mask_predictions += (predicted_mask,)

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            dropout_probability = torch.rand([])
            if self.training and (dropout_probability < self.layerdrop):
                continue

            level_index = idx % self.num_feature_levels
            _, attention_mask = self.mask_predictor(
                intermediate_hidden_states,
                pixel_embeddings,
                feature_size_list[level_index],
            )

            where = (attention_mask.sum(-1) != attention_mask.shape[-1]).to(
                attention_mask.dtype
            )
            attention_mask = attention_mask * where.unsqueeze(-1)

            layer_outputs = decoder_layer(
                hidden_states,
                level_index,
                None,
                multi_stage_positional_embeddings,
                query_position_embeddings,
                encoder_hidden_states,
                encoder_attention_mask=attention_mask,
                output_attentions=output_attentions,
            )

            intermediate_hidden_states = self.layernorm(layer_outputs[0])
            predicted_mask, _ = self.mask_predictor(
                intermediate_hidden_states,
                pixel_embeddings,
                feature_size_list[(idx + 1) % self.num_feature_levels],
            )
            intermediate_mask_predictions += (predicted_mask,)
            intermediate += (intermediate_hidden_states,)
            hidden_states = layer_outputs[0]

            if output_attentions:
                attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        hidden_states = hidden_states.transpose(1, 0)
        if not return_dict:
            outputs = [
                hidden_states,
                all_hidden_states,
                attentions,
                intermediate,
                intermediate_mask_predictions,
            ]
            return tuple(v for v in outputs if v is not None)

        return Mask2FormerMaskedAttentionDecoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=attentions,
            intermediate_hidden_states=intermediate,
            masks_queries_logits=intermediate_mask_predictions,
        )

    decoder.forward = MethodType(_forward_patched, decoder)


def _load_imagenet_backbone(
    model: Mask2FormerForUniversalSegmentation,
) -> None:
    """
    Copy ImageNet-1K pretrained ConvNeXt-V2 (or V1) weights into the backbone.

    The backbone model id is read from ``config.BACKBONE``
    (e.g. ``"convnextv2-base-1k-224"`` → ``facebook/convnextv2-base-1k-224``).
    """
    import config
    from transformers import ConvNextV2Model, ConvNextModel

    backbone_id = getattr(config, "BACKBONE", "convnextv2-base-1k-224")
    hf_id = f"facebook/{backbone_id}"
    use_v2 = "convnextv2" in backbone_id

    print(f"[model] loading ImageNet-1K backbone: {hf_id}")
    if use_v2:
        imagenet_convnext = ConvNextV2Model.from_pretrained(hf_id, use_safetensors=True)
    else:
        imagenet_convnext = ConvNextModel.from_pretrained(hf_id, use_safetensors=True)

    backbone = model.model.pixel_level_module.encoder

    # strict=False: skips pooler / classifier keys absent in Mask2Former.
    missing, unexpected = backbone.load_state_dict(
        imagenet_convnext.state_dict(), strict=False
    )

    n_loaded = len(imagenet_convnext.state_dict()) - len(unexpected)
    print(f"[model] backbone weights loaded: {n_loaded} tensors")
    del imagenet_convnext  # free memory immediately


def _build_config(
    num_classes: int,
    num_queries: int,
) -> Mask2FormerConfig:
    """Build a from-scratch Mask2Former config driven by config.py."""
    import config
    from transformers import ConvNextV2Config, ConvNextConfig

    backbone_id = getattr(config, "BACKBONE", "convnextv2-base-1k-224")
    use_v2 = "convnextv2" in backbone_id

    if use_v2:
        backbone_config = ConvNextV2Config(
            num_channels=3,
            depths=config.BACKBONE_DEPTHS,
            hidden_sizes=config.BACKBONE_HIDDEN_SIZES,
            drop_path_rate=config.BACKBONE_DROP_PATH,
            out_features=["stage1", "stage2", "stage3", "stage4"],
        )
    else:
        backbone_config = ConvNextConfig(
            num_channels=3,
            depths=config.BACKBONE_DEPTHS,
            hidden_sizes=config.BACKBONE_HIDDEN_SIZES,
            drop_path_rate=config.BACKBONE_DROP_PATH,
            out_features=["stage1", "stage2", "stage3", "stage4"],
        )

    config_ = Mask2FormerConfig(
        backbone_config=backbone_config,
        num_labels=num_classes,
        num_queries=num_queries,
        feature_size=config.DECODER_FEATURE_SIZE,
        mask_feature_size=config.DECODER_MASK_FEATURE_SIZE,
        hidden_dim=config.DECODER_HIDDEN_DIM,
        encoder_layers=config.DECODER_ENCODER_LAYERS,
        decoder_layers=config.DECODER_DECODER_LAYERS,
        num_attention_heads=config.DECODER_ATTENTION_HEADS,
        dropout=config.DECODER_DROPOUT,
        dim_feedforward=config.DECODER_DIM_FEEDFORWARD,
        pre_norm=False,
        enforce_input_projection=False,
        common_stride=4,
    )
    return config_


# ---------------------------------------------------------------------------
# Lightweight wrapper (optional) – freezes backbone early in training
# ---------------------------------------------------------------------------


class Mask2FormerWrapper(nn.Module):
    """
    Thin wrapper around the HF model that exposes
    ``freeze_backbone`` / ``unfreeze_backbone`` helpers.
    """

    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = NUM_CLASSES,
        num_queries: int = 100,
    ) -> None:
        super().__init__()
        self.model = build_mask2former(
            pretrained=pretrained,
            num_classes=num_classes,
            num_queries=num_queries,
        )

    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Freeze ConvNeXt-Base backbone weights (warm-up phase)."""
        for param in self.model.model.pixel_level_module.encoder.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze ConvNeXt-Base backbone for full fine-tuning."""

        for param in self.model.model.pixel_level_module.encoder.parameters():
            param.requires_grad = True

    # ------------------------------------------------------------------

    def forward(
        self,
        pixel_values: torch.Tensor,
        mask_labels: Optional[list] = None,
        class_labels: Optional[list] = None,
    ) -> Dict:
        """
        Args:
            pixel_values:  (B, 3, H, W).
            mask_labels:   list of (N_i, H, W) float tensors per image
                           (required for training).
            class_labels:  list of (N_i,) int64 tensors per image
                           (required for training).

        Returns:
            HF ``Mask2FormerForUniversalSegmentationOutput`` (dict-like).
            During training this contains ``loss`` and its components.
            During inference it contains ``masks_queries_logits`` and
            ``class_queries_logits``.
        """
        outputs = self.model(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )
        return outputs
