"""
model.py – ConvNeXt(V1/V2)-Base + FPN + Mask R-CNN.

Improvements over baseline:
* Multi-scale training (random min_size in MULTI_SCALE_MIN_SIZES per iter)
* Higher-resolution mask head (28x28 RoIAlign → 56x56 mask logits)
* Optional ConvNeXt-V2-Base backbone (HF, ImageNet-1K only weights)
* Optional Cascade Mask R-CNN (3 IoU stages: 0.5 → 0.6 → 0.7)

Weight policy: backbone uses ImageNet-1K weights only; FPN/RPN/heads are
randomly initialised.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection._utils import (
    BalancedPositiveNegativeSampler,
    BoxCoder,
    Matcher,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads
from torchvision.models.detection.roi_heads import fastrcnn_loss, maskrcnn_loss
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import (
    FeaturePyramidNetwork,
    MultiScaleRoIAlign,
    batched_nms,
    box_iou,
    clip_boxes_to_image,
)
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

import config


# ---------------------------------------------------------------------------
# Backbone variants
# ---------------------------------------------------------------------------


def _build_convnext_v1_backbone() -> nn.Module:
    """torchvision ConvNeXt-Base, ImageNet-1K-V1 weights."""
    net = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
    body = create_feature_extractor(net, return_nodes=config.BACKBONE_RETURN_NODES)
    return body


def _build_convnextv2_base_backbone() -> nn.Module:
    """
    HF ConvNeXt-V2-Base (facebook/convnextv2-base-1k-224).

    Returns a module whose forward returns OrderedDict("0".."3") matching
    the V1 feature-extractor signature, so it drops into FPN.
    """
    from transformers import ConvNextV2Model

    hf_model = ConvNextV2Model.from_pretrained(
        "facebook/convnextv2-base-1k-224", output_hidden_states=True
    )

    class V2Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            out = self.m(pixel_values=x)
            # hidden_states: (stem, stage1, stage2, stage3, stage4)
            hs = out.hidden_states
            return OrderedDict([
                ("0", hs[1]),
                ("1", hs[2]),
                ("2", hs[3]),
                ("3", hs[4]),
            ])

    return V2Wrapper(hf_model)


class ConvNeXtFPNBackbone(nn.Module):
    """ConvNeXt(V1 or V2) + FPN backbone for torchvision MaskRCNN."""

    def __init__(self) -> None:
        super().__init__()
        if config.BACKBONE_NAME == "convnext_base":
            self.body = _build_convnext_v1_backbone()
        elif config.BACKBONE_NAME == "convnextv2_base":
            self.body = _build_convnextv2_base_backbone()
        else:
            raise ValueError(f"Unknown BACKBONE_NAME: {config.BACKBONE_NAME}")

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=config.BACKBONE_IN_CHANNELS,
            out_channels=config.BACKBONE_OUT_CHANNELS,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = config.BACKBONE_OUT_CHANNELS

    def forward(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        feats = self.body(x)
        return self.fpn(feats)


# ---------------------------------------------------------------------------
# Cascade Mask R-CNN RoI Heads
# ---------------------------------------------------------------------------


class CascadeRoIHeads(nn.Module):
    """
    Cascade Mask R-CNN RoI heads (Cai & Vasconcelos, CVPR 2018).

    Three cascade stages with IoU thresholds (0.5 → 0.6 → 0.7):
    - Training  : independent IoU assignment + detection loss per stage,
                  mask loss computed on final-stage positive proposals.
    - Inference : sequential box refinement, ensemble class scores (avg
                  softmax), per-class batched NMS, mask on final boxes.

    Drop-in replacement for torchvision RoIHeads; same (result, losses)
    interface so GeneralizedRCNN / MaskRCNN forward needs no changes.
    """

    CASCADE_IOUS: Tuple[float, ...] = (0.5, 0.6, 0.7)

    def __init__(
        self,
        box_roi_pool: MultiScaleRoIAlign,
        box_heads: nn.ModuleList,       # one TwoMLPHead per stage
        box_predictors: nn.ModuleList,  # one FastRCNNPredictor per stage
        num_classes: int,
        batch_size_per_image: int = 512,
        positive_fraction: float = 0.25,
        bbox_reg_weights: Tuple[float, ...] = (10.0, 10.0, 5.0, 5.0),
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        detections_per_img: int = 500,
        mask_roi_pool: Optional[MultiScaleRoIAlign] = None,
        mask_head: Optional[nn.Module] = None,
        mask_predictor: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        assert len(box_heads) == len(self.CASCADE_IOUS)
        assert len(box_predictors) == len(self.CASCADE_IOUS)

        self.box_roi_pool = box_roi_pool
        self.box_heads = box_heads
        self.box_predictors = box_predictors
        self.mask_roi_pool = mask_roi_pool
        self.mask_head = mask_head
        self.mask_predictor = mask_predictor

        self.num_classes = num_classes
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

        self.box_coder = BoxCoder(weights=bbox_reg_weights)
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction
        )
        # Matcher is not an nn.Module — just store as Python list
        self._cascade_matchers: List[Matcher] = [
            Matcher(iou_thr, iou_thr - 0.1, allow_low_quality_matches=False)
            for iou_thr in self.CASCADE_IOUS
        ]

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _assign(
        self,
        proposals: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        matcher: Matcher,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Return (matched_idxs, labels, regression_targets) per image."""
        matched_idxs_list, labels_list, reg_list = [], [], []

        for props, tgt in zip(proposals, targets):
            gt_boxes = tgt["boxes"]
            gt_labels = tgt["labels"]
            device = props.device
            N = props.shape[0]

            if gt_boxes.numel() == 0:
                matched_idxs_list.append(torch.zeros(N, dtype=torch.int64, device=device))
                labels_list.append(torch.zeros(N, dtype=torch.int64, device=device))
                reg_list.append(torch.zeros((N, 4), device=device))
                continue

            iou_mat = box_iou(gt_boxes, props)        # [num_gt, N]
            matched = matcher(iou_mat)                 # [N]
            clamped = matched.clamp(min=0)

            lbls = gt_labels[clamped].to(torch.int64)
            lbls[matched == Matcher.BELOW_LOW_THRESHOLD] = 0    # background
            lbls[matched == Matcher.BETWEEN_THRESHOLDS] = -1  # ignore

            reg = self.box_coder.encode([gt_boxes[clamped]], [props])[0]  # [N, 4]

            matched_idxs_list.append(clamped)
            labels_list.append(lbls)
            reg_list.append(reg)

        return matched_idxs_list, labels_list, reg_list

    def _refine(
        self,
        box_regression: torch.Tensor,     # [N_total, num_classes * 4]
        proposals: List[torch.Tensor],
        image_shapes: List[Tuple[int, int]],
    ) -> List[torch.Tensor]:
        """Decode regression deltas → refined proposals (avg over fg classes)."""
        decoded = self.box_coder.decode(box_regression, list(proposals))  # [N_total, C*4]
        N = sum(len(p) for p in proposals)
        decoded = decoded.view(N, self.num_classes, 4)            # [N_total, C, 4]
        refined = decoded[:, 1:, :].mean(dim=1)                   # [N_total, 4]  avg fg

        new_props = []
        for r, shape in zip(refined.split([len(p) for p in proposals]), image_shapes):
            new_props.append(clip_boxes_to_image(r.detach(), shape))
        return new_props

    # ------------------------------------------------------------------ #
    # Forward                                                              #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[torch.Tensor],
        image_shapes: List[Tuple[int, int]],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Tuple[List[Dict], Dict[str, torch.Tensor]]:
        if self.training:
            assert targets is not None
            return self._train(features, proposals, image_shapes, targets)
        return self._eval(features, proposals, image_shapes)

    # ------------------------------------------------------------------ #
    # Training pass                                                        #
    # ------------------------------------------------------------------ #

    def _train(self, features, proposals, image_shapes, targets):
        losses: Dict[str, torch.Tensor] = {}
        cur_props = [p.detach() for p in proposals]

        last_pos_props: Optional[List[torch.Tensor]] = None
        last_matched_idxs: Optional[List[torch.Tensor]] = None

        for s, (matcher, bh, bp) in enumerate(
            zip(self._cascade_matchers, self.box_heads, self.box_predictors)
        ):
            # 1. Assign GT with stage-specific IoU threshold
            matched_idxs, labels, reg_targets = self._assign(cur_props, targets, matcher)

            # 2. Sample balanced pos/neg
            pos_idxs, neg_idxs = self.fg_bg_sampler(labels)

            # 3. Pool ALL current proposals (reuse for refinement step 6)
            box_feats = self.box_roi_pool(features, cur_props, image_shapes)
            box_feats = bh(box_feats)
            all_cls_logits, all_box_reg = bp(box_feats)

            # 4. Select sampled subset for loss
            props_per = [len(p) for p in cur_props]
            offset = 0
            sampled_cls, sampled_box, sampled_lbl, sampled_reg = [], [], [], []
            for i, (pi, ni) in enumerate(zip(pos_idxs, neg_idxs)):
                keep_i = (pi | ni).nonzero(as_tuple=False).squeeze(1)
                g = keep_i + offset
                sampled_cls.append(all_cls_logits[g])
                sampled_box.append(all_box_reg[g])
                sampled_lbl.append(labels[i][keep_i])
                sampled_reg.append(reg_targets[i][keep_i])
                offset += props_per[i]

            loss_cls, loss_box = fastrcnn_loss(
                torch.cat(sampled_cls), torch.cat(sampled_box),
                sampled_lbl, sampled_reg,
            )
            losses[f"loss_classifier_s{s}"] = loss_cls
            losses[f"loss_box_reg_s{s}"] = loss_box

            # 5. Save final-stage positives for mask head
            if s == len(self.CASCADE_IOUS) - 1:
                # Use stage-2 REFINED boxes (output) for mask RoI, not input proposals
                refined_for_mask = self._refine(all_box_reg.detach(), cur_props, image_shapes)
                last_pos_props, last_matched_idxs = [], []
                for i, (pi, mi) in enumerate(zip(pos_idxs, matched_idxs)):
                    pos_i = pi.nonzero(as_tuple=False).squeeze(1)
                    last_pos_props.append(refined_for_mask[i][pos_i])
                    last_matched_idxs.append(mi[pos_i])

            # 6. Refine proposals for next stage
            if s < len(self.CASCADE_IOUS) - 1:
                cur_props = self._refine(all_box_reg.detach(), cur_props, image_shapes)

        # Mask head on final-stage positive proposals
        if (
            self.mask_roi_pool is not None
            and last_pos_props is not None
            and any(p.shape[0] > 0 for p in last_pos_props)
        ):
            gt_masks = [t["masks"] for t in targets]
            gt_labels = [t["labels"] for t in targets]
            mf = self.mask_roi_pool(features, last_pos_props, image_shapes)
            mf = self.mask_head(mf)
            mask_logits = self.mask_predictor(mf)
            losses["loss_mask"] = maskrcnn_loss(
                mask_logits, last_pos_props, gt_masks, gt_labels, last_matched_idxs
            )

        return [{} for _ in proposals], losses

    # ------------------------------------------------------------------ #
    # Inference pass                                                       #
    # ------------------------------------------------------------------ #

    def _eval(self, features, proposals, image_shapes):
        all_cls: List[torch.Tensor] = []
        cur_props = proposals
        final_box_reg: Optional[torch.Tensor] = None

        for s, (bh, bp) in enumerate(zip(self.box_heads, self.box_predictors)):
            bf = self.box_roi_pool(features, cur_props, image_shapes)
            bf = bh(bf)
            cls_logits, box_reg = bp(bf)
            all_cls.append(cls_logits)
            final_box_reg = box_reg
            if s < len(self.CASCADE_IOUS) - 1:
                cur_props = self._refine(box_reg, cur_props, image_shapes)

        det_boxes, det_scores, det_labels = self._postprocess(
            all_cls, final_box_reg, cur_props, image_shapes
        )

        # Mask head
        result = []
        device = final_box_reg.device
        empty_mask = torch.zeros(
            (0, 1, config.MASK_ROI_OUTPUT_SIZE, config.MASK_ROI_OUTPUT_SIZE), device=device
        )

        if self.mask_roi_pool is not None and sum(b.shape[0] for b in det_boxes) > 0:
            mf = self.mask_roi_pool(features, det_boxes, image_shapes)
            mf = self.mask_head(mf)
            mask_logits = self.mask_predictor(mf)  # [total, C, H, W]

            dets_per = [b.shape[0] for b in det_boxes]
            offset = 0
            for b, s_scores, lbls, n in zip(det_boxes, det_scores, det_labels, dets_per):
                ml = mask_logits[offset: offset + n]          # [n, C, H, W]
                idx = torch.arange(n, device=device)
                sel = ml.sigmoid()[idx, lbls.long()]           # [n, H, W]
                result.append({
                    "boxes": b, "scores": s_scores, "labels": lbls,
                    "masks": sel.unsqueeze(1),                 # [n, 1, H, W]
                })
                offset += n
        else:
            for b, s_scores, lbls in zip(det_boxes, det_scores, det_labels):
                result.append({
                    "boxes": b, "scores": s_scores, "labels": lbls,
                    "masks": empty_mask[:b.shape[0]],
                })

        return result, {}

    def _postprocess(
        self,
        all_cls_logits: List[torch.Tensor],
        box_regression: torch.Tensor,
        proposals: List[torch.Tensor],
        image_shapes: List[Tuple[int, int]],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Average ensemble scores, decode final boxes, batched NMS."""
        device = box_regression.device

        # Ensemble: average softmax across stages → [N_total, num_classes]
        avg_scores = torch.stack(
            [F.softmax(lg, dim=-1) for lg in all_cls_logits], dim=0
        ).mean(0)

        # Decode final-stage boxes → [N_total, C, 4]
        pred_boxes = self.box_coder.decode(box_regression, list(proposals))  # [N_total, C*4]
        pred_boxes = pred_boxes.view(-1, self.num_classes, 4)             # [N_total, C, 4]

        props_per = [len(p) for p in proposals]
        boxes_split = pred_boxes.split(props_per, dim=0)
        scores_split = avg_scores.split(props_per, dim=0)

        all_boxes, all_scores, all_labels = [], [], []

        for pb, sc, img_shape in zip(boxes_split, scores_split, image_shapes):
            img_b, img_s, img_l = [], [], []
            for c in range(1, self.num_classes):
                sc_c = sc[:, c]
                bc = clip_boxes_to_image(pb[:, c, :], img_shape)
                keep = sc_c > self.score_thresh
                if not keep.any():
                    continue
                img_b.append(bc[keep])
                img_s.append(sc_c[keep])
                img_l.append(torch.full((keep.sum(),), c, dtype=torch.int64, device=device))

            if img_b:
                ib = torch.cat(img_b)
                is_ = torch.cat(img_s)
                il = torch.cat(img_l)
                keep = batched_nms(ib, is_, il, self.nms_thresh)
                keep = keep[:self.detections_per_img]
                all_boxes.append(ib[keep])
                all_scores.append(is_[keep])
                all_labels.append(il[keep])
            else:
                all_boxes.append(torch.zeros((0, 4), device=device))
                all_scores.append(torch.zeros(0, device=device))
                all_labels.append(torch.zeros(0, dtype=torch.int64, device=device))

        return all_boxes, all_scores, all_labels


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_maskrcnn(
    pretrained_backbone: bool = True,
    multi_scale: bool = True,
) -> MaskRCNN:
    """
    Build Mask R-CNN with ConvNeXt+FPN backbone, larger mask head,
    and (optional) multi-scale training resolution.
    """
    _ = pretrained_backbone  # always loaded inside backbone factory
    backbone = ConvNeXtFPNBackbone()

    anchor_generator = AnchorGenerator(
        sizes=config.RPN_ANCHOR_SIZES,
        aspect_ratios=config.RPN_ASPECT_RATIOS,
    )

    # Higher-resolution mask RoIAlign for sharper boundaries
    mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=config.MASK_ROI_OUTPUT_SIZE,
        sampling_ratio=2,
    )

    if multi_scale:
        min_size = config.MULTI_SCALE_MIN_SIZES
        max_size = config.MULTI_SCALE_MAX_SIZE
    else:
        min_size = config.INFERENCE_MIN_SIZE
        max_size = config.INFERENCE_MAX_SIZE

    model = MaskRCNN(
        backbone=backbone,
        num_classes=config.NUM_CLASSES_WITH_BG,
        rpn_anchor_generator=anchor_generator,
        box_detections_per_img=config.BOX_DETECTIONS_PER_IMG,
        min_size=min_size,
        max_size=max_size,
        mask_roi_pool=mask_roi_pool,
    )
    # Deeper mask head: default torchvision is 4-conv; replace with MASK_HEAD_LAYERS-conv
    model.roi_heads.mask_head = MaskRCNNHeads(
        config.BACKBONE_OUT_CHANNELS,
        [256] * config.MASK_HEAD_LAYERS,
        dilation=1,
    )
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_cascade_maskrcnn(
    pretrained_backbone: bool = True,
    multi_scale: bool = True,
) -> nn.Module:
    """
    Build Cascade Mask R-CNN with ConvNeXt-V2-Base + FPN backbone.

    Replaces the standard torchvision RoIHeads with 3-stage CascadeRoIHeads
    (IoU thresholds 0.5 → 0.6 → 0.7).

    Parameter budget:
        Standard Mask R-CNN  ≈ 107 M
        + 2 extra box stages ≈  26 M  (TwoMLPHead + FastRCNNPredictor × 2)
        Cascade total        ≈ 133 M  → well under the 200 M limit
    """
    base = build_maskrcnn(pretrained_backbone=pretrained_backbone, multi_scale=multi_scale)
    rh = base.roi_heads

    # Three cascade stages: stage 0 reuses existing heads; 1 and 2 are deep-copies
    box_heads = nn.ModuleList([
        rh.box_head,
        copy.deepcopy(rh.box_head),
        copy.deepcopy(rh.box_head),
    ])
    box_predictors = nn.ModuleList([
        rh.box_predictor,
        copy.deepcopy(rh.box_predictor),
        copy.deepcopy(rh.box_predictor),
    ])

    base.roi_heads = CascadeRoIHeads(
        box_roi_pool=rh.box_roi_pool,
        box_heads=box_heads,
        box_predictors=box_predictors,
        num_classes=config.NUM_CLASSES_WITH_BG,
        batch_size_per_image=512,
        positive_fraction=0.25,
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=config.BOX_DETECTIONS_PER_IMG,
        mask_roi_pool=rh.mask_roi_pool,
        mask_head=rh.mask_head,
        mask_predictor=rh.mask_predictor,
    )
    return base


if __name__ == "__main__":
    m = build_cascade_maskrcnn(pretrained_backbone=False)
    n = count_parameters(m)
    print(f"Cascade Mask R-CNN params: {n / 1e6:.2f} M  (limit 200 M)")
