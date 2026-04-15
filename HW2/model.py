"""Deformable DETR with ResNet-50 backbone and multi-scale feature maps.

Key changes from standard DETR:
1. Multi-scale features (C3, C4, C5) from ResNet-50
2. Deformable attention (sparse sampling instead of global attention)
3. Sigmoid classification (focal loss) instead of softmax + background class
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_utils
from torchvision.models import resnet50, ResNet50_Weights


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _inverse_sigmoid(x, eps=1e-5):
    """Inverse of sigmoid: log(x / (1-x)), clipped for numerical stability."""
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))


class MLP(nn.Module):
    """Simple multi-layer perceptron (feed-forward network)."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)]
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionEmbeddingSine(nn.Module):
    """2-D sine/cosine positional encoding (follows original DETR paper)."""

    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, mask):
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=mask.device
        )
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack(
            [pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()], dim=4
        ).flatten(3)
        pos_y = torch.stack(
            [pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()], dim=4
        ).flatten(3)

        pos = torch.cat([pos_y, pos_x], dim=3).permute(0, 3, 1, 2)
        return pos


# ---------------------------------------------------------------------------
# Multi-scale ResNet-50 backbone
# ---------------------------------------------------------------------------

class BackboneResNet50(nn.Module):
    """ResNet-50 backbone returning multi-scale features (C2, C3, C4).

    Uses lower-stride features for small object (digit) detection:
    - C2 (stride 4, 256 ch): preserves fine digit edges
    - C3 (stride 8, 512 ch): main feature level
    - C4 (stride 16, 1024 ch): context
    Drops C5 (stride 32) which gives <1.5px per digit — useless.
    """

    def __init__(self, pretrained=True, train_backbone=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet50(weights=weights)

        self.stem = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool
        )
        self.layer1 = net.layer1   # stride 4,  256 ch  → C2
        self.layer2 = net.layer2   # stride 8,  512 ch  → C3
        self.layer3 = net.layer3   # stride 16, 1024 ch → C4

        self.num_channels = [256, 512, 1024]
        self.train_backbone = train_backbone

        if not train_backbone:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        return [c2, c3, c4]

    def train(self, mode=True):
        """Keep BatchNorm frozen to avoid stats noise at small batch sizes."""
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


# ---------------------------------------------------------------------------
# Multi-Scale Deformable Attention (pure PyTorch, no custom CUDA ops)
# ---------------------------------------------------------------------------

def _ms_deform_attn_core(value, spatial_shapes, sampling_locations, attention_weights):
    """
    Pure PyTorch multi-scale deformable attention via grid_sample.

    Args:
        value: [B, sum(Hi*Wi), n_heads, head_dim]
        spatial_shapes: list of (H, W) tuples, one per level
        sampling_locations: [B, Lq, n_heads, n_levels, n_points, 2] in [0, 1]
        attention_weights: [B, Lq, n_heads, n_levels, n_points]
    Returns:
        output: [B, Lq, n_heads * head_dim]
    """
    B, _, n_heads, head_dim = value.shape
    _, Lq, _, n_levels, n_points, _ = sampling_locations.shape

    # Split value by levels
    split_sizes = [H * W for H, W in spatial_shapes]
    value_list = value.split(split_sizes, dim=1)

    # Convert [0,1] → [-1,1] for grid_sample
    sampling_grids = 2 * sampling_locations - 1

    sampling_value_list = []
    for lid, (H, W) in enumerate(spatial_shapes):
        # [B, H*W, n_heads, head_dim] → [B*n_heads, head_dim, H, W]
        value_l = (
            value_list[lid]
            .permute(0, 2, 3, 1)
            .reshape(B * n_heads, head_dim, H, W)
        )
        # [B, Lq, n_heads, n_points, 2] → [B*n_heads, Lq, n_points, 2]
        sampling_grid_l = (
            sampling_grids[:, :, :, lid]
            .permute(0, 2, 1, 3, 4)
            .reshape(B * n_heads, Lq, n_points, 2)
        )
        # → [B*n_heads, head_dim, Lq, n_points]
        sampling_value_l = F.grid_sample(
            value_l, sampling_grid_l,
            mode="bilinear", padding_mode="zeros", align_corners=False,
        )
        sampling_value_list.append(sampling_value_l)

    # [B*n_heads, head_dim, Lq, n_levels * n_points]
    sampling_values = torch.stack(sampling_value_list, dim=-1).reshape(
        B * n_heads, head_dim, Lq, n_levels * n_points
    )
    # [B*n_heads, 1, Lq, n_levels * n_points]
    attn = (
        attention_weights
        .permute(0, 2, 1, 3, 4)
        .reshape(B * n_heads, 1, Lq, n_levels * n_points)
    )
    # Weighted sum → [B*n_heads, head_dim, Lq]
    output = (sampling_values * attn).sum(-1)
    # → [B, Lq, n_heads * head_dim]
    output = output.reshape(B, n_heads * head_dim, Lq).permute(0, 2, 1)
    return output


class MSDeformAttn(nn.Module):
    """Multi-Scale Deformable Attention Module."""

    def __init__(self, d_model=256, n_levels=3, n_heads=8, n_points=4):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        self.sampling_offsets = nn.Linear(
            d_model, n_heads * n_levels * n_points * 2
        )
        self.attention_weights = nn.Linear(
            d_model, n_heads * n_levels * n_points
        )
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        # Initialize biases so offsets fan out like a grid
        thetas = torch.arange(
            self.n_heads, dtype=torch.float32
        ) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        grid_init = (
            grid_init
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self, query, reference_points, input_flatten,
        input_spatial_shapes, input_padding_mask=None,
    ):
        """
        Args:
            query:                [B, Lq, C]
            reference_points:     [B, Lq, n_levels, 2] normalised [0,1]
            input_flatten:        [B, sum(Hi*Wi), C]
            input_spatial_shapes: list of (H, W) tuples
            input_padding_mask:   [B, sum(Hi*Wi)] bool or None
        """
        B, Lq, _ = query.shape

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.0)
        value = value.view(B, -1, self.n_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).view(
            B, Lq, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            B, Lq, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            B, Lq, self.n_heads, self.n_levels, self.n_points
        )

        # Normalise offsets by spatial resolution
        offset_normalizer = torch.tensor(
            [[W, H] for H, W in input_spatial_shapes],
            dtype=torch.float32, device=query.device,
        )  # [n_levels, 2]

        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + sampling_offsets
            / offset_normalizer[None, None, None, :, None, :]
        )

        output = _ms_deform_attn_core(
            value, input_spatial_shapes, sampling_locations, attention_weights
        )
        return self.output_proj(output)


# ---------------------------------------------------------------------------
# Deformable Transformer layers
# ---------------------------------------------------------------------------

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
        self, d_model=256, d_ffn=1024, dropout=0.1,
        n_levels=3, n_heads=8, n_points=4,
    ):
        super().__init__()
        # Deformable self-attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, pos, reference_points, spatial_shapes, padding_mask=None):
        src2 = self.self_attn(
            src + pos, reference_points, src, spatial_shapes, padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self, d_model=256, d_ffn=1024, dropout=0.1,
        n_levels=3, n_heads=8, n_points=4,
    ):
        super().__init__()
        # Self-attention (standard MHA over object queries)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # Cross-attention (deformable, queries attend to encoder memory)
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self, tgt, query_pos, reference_points,
        src, src_spatial_shapes, src_padding_mask=None,
    ):
        # Self-attention among object queries
        q = k = tgt + query_pos
        tgt2, _ = self.self_attn(q, k, tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # Deformable cross-attention
        tgt2 = self.cross_attn(
            tgt + query_pos, reference_points,
            src, src_spatial_shapes, src_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # FFN
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt


# ---------------------------------------------------------------------------
# Deformable Transformer (encoder + decoder)
# ---------------------------------------------------------------------------

class DeformableTransformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        n_levels=3,
        n_points=4,
        use_checkpoint=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.use_checkpoint = use_checkpoint

        enc_layer = DeformableTransformerEncoderLayer(
            d_model, dim_feedforward, dropout, n_levels, nhead, n_points
        )
        self.encoder = nn.ModuleList(
            [copy.deepcopy(enc_layer) for _ in range(num_encoder_layers)]
        )

        dec_layer = DeformableTransformerDecoderLayer(
            d_model, dim_feedforward, dropout, n_levels, nhead, n_points
        )
        self.decoder = nn.ModuleList(
            [copy.deepcopy(dec_layer) for _ in range(num_decoder_layers)]
        )

        # Learnable level embedding added to each scale
        self.level_embed = nn.Parameter(torch.Tensor(n_levels, d_model))

        # Project query embedding → reference point (x, y)
        self.reference_points_linear = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.level_embed)
        nn.init.xavier_uniform_(self.reference_points_linear.weight)
        nn.init.constant_(self.reference_points_linear.bias, 0.0)

    @staticmethod
    def _get_encoder_reference_points(spatial_shapes, device):
        """Grid centres for each feature-map cell, replicated over all levels."""
        ref_list = []
        for H, W in spatial_shapes:
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, device=device) / H,
                torch.linspace(0.5, W - 0.5, W, device=device) / W,
                indexing="ij",
            )
            ref = torch.stack([ref_x.flatten(), ref_y.flatten()], dim=-1)
            ref_list.append(ref)
        reference_points = torch.cat(ref_list, dim=0)  # [sum(Hi*Wi), 2]
        # → [1, sum(Hi*Wi), n_levels, 2]
        reference_points = (
            reference_points[None, :, None, :]
            .repeat(1, 1, len(spatial_shapes), 1)
        )
        return reference_points

    def forward(self, srcs, masks, pos_embeds, query_embed):
        """
        Args:
            srcs:       list of [B, C, Hi, Wi] projected feature maps
            masks:      list of [B, Hi, Wi] boolean masks
            pos_embeds: list of [B, C, Hi, Wi] positional encodings
            query_embed: [Q, 2*C] (split into query_pos + query_tgt)
        Returns:
            hs: [B, Q, C] decoder output
        """
        # --- flatten multi-scale features --------------------------------
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []

        for lvl, (src, mask, pos) in enumerate(zip(srcs, masks, pos_embeds)):
            B, C, H, W = src.shape
            spatial_shapes.append((H, W))
            src_flat = src.flatten(2).permute(0, 2, 1)    # [B, H*W, C]
            mask_flat = mask.flatten(1)                    # [B, H*W]
            pos_flat = pos.flatten(2).permute(0, 2, 1)    # [B, H*W, C]
            lvl_pos = pos_flat + self.level_embed[lvl].view(1, 1, -1)
            src_flatten.append(src_flat)
            mask_flatten.append(mask_flat)
            lvl_pos_embed_flatten.append(lvl_pos)

        src_flatten = torch.cat(src_flatten, dim=1)               # [B, Σ, C]
        mask_flatten = torch.cat(mask_flatten, dim=1)             # [B, Σ]
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, dim=1)

        # --- encoder -----------------------------------------------------
        enc_ref = self._get_encoder_reference_points(
            spatial_shapes, src_flatten.device
        ).expand(B, -1, -1, -1)

        memory = src_flatten
        for layer in self.encoder:
            if self.use_checkpoint and self.training:
                def _fwd(mem, pos, ref, mask, _layer=layer, _shapes=spatial_shapes):
                    return _layer(mem, pos, ref, _shapes, mask)
                memory = checkpoint_utils.checkpoint(
                    _fwd, memory, lvl_pos_embed_flatten, enc_ref, mask_flatten,
                    use_reentrant=False,
                )
            else:
                memory = layer(
                    memory, lvl_pos_embed_flatten, enc_ref,
                    spatial_shapes, mask_flatten,
                )

        # --- decoder -----------------------------------------------------
        query_pos, query_tgt = query_embed.split(C, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(B, -1, -1)
        query_tgt = query_tgt.unsqueeze(0).expand(B, -1, -1)

        # Reference points from query_pos → sigmoid → [0, 1]
        dec_ref = self.reference_points_linear(query_pos).sigmoid()
        # → [B, Q, n_levels, 2]
        dec_ref = dec_ref[:, :, None, :].repeat(1, 1, self.n_levels, 1)

        hs = query_tgt
        intermediate = []
        intermediate_ref = []
        for lid, layer in enumerate(self.decoder):
            hs = layer(
                hs, query_pos, dec_ref,
                memory, spatial_shapes, mask_flatten,
            )
            intermediate.append(hs)
            intermediate_ref.append(dec_ref[:, :, 0, :])  # [B, Q, 2]

            # Iterative bbox refinement: update reference points in logit space
            if self.bbox_refine is not None:
                tmp = self.bbox_refine[lid](hs)  # [B, Q, 4]
                # ref is in [0,1]; convert to logit space, add offset, sigmoid back
                new_ref = (_inverse_sigmoid(dec_ref[:, :, 0, :]) + tmp[..., :2]).sigmoid()
                dec_ref = new_ref[:, :, None, :].repeat(1, 1, self.n_levels, 1).detach()

        # [n_decoder_layers, B, Q, C]
        hs_stack = torch.stack(intermediate)
        ref_stack = torch.stack(intermediate_ref)  # [n_decoder_layers, B, Q, 2]

        return hs_stack, ref_stack


# ---------------------------------------------------------------------------
# Deformable DETR
# ---------------------------------------------------------------------------

class DETR(nn.Module):
    """
    Deformable DETR for digit detection.

    Changes from standard DETR:
    - Multi-scale features (C3, C4, C5 from ResNet-50)
    - Deformable attention in encoder and decoder cross-attention
    - Sigmoid classification head (no explicit background class)
    - Iterative bounding box refinement
    - Auxiliary decoding losses from all decoder layers
    """

    def __init__(
        self,
        num_classes,
        num_queries=20,
        hidden_dim=256,
        nheads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        n_levels=3,
        n_points=4,
        pretrained_backbone=True,
        use_checkpoint=False,
        aux_loss=True,
        iterative_refine=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.n_levels = n_levels
        self.aux_loss = aux_loss
        self.iterative_refine = iterative_refine

        # --- backbone -------------------------------------------------
        self.backbone = BackboneResNet50(pretrained=pretrained_backbone)

        # One 1×1 conv per backbone level → hidden_dim
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            )
            for ch in self.backbone.num_channels
        ])

        # --- positional encoding --------------------------------------
        self.pos_enc = PositionEmbeddingSine(hidden_dim // 2)

        # --- transformer ----------------------------------------------
        self.transformer = DeformableTransformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            n_levels=n_levels,
            n_points=n_points,
            use_checkpoint=use_checkpoint,
        )

        # query_embed stores both positional part and content part
        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)

        # --- output heads ---------------------------------------------
        # Sigmoid classification — num_classes only (no background slot)
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # --- iterative bbox refinement heads (one per decoder layer) ---
        if iterative_refine:
            self.class_embed = _get_clones(self.class_embed, num_decoder_layers)
            self.bbox_embed = _get_clones(self.bbox_embed, num_decoder_layers)
            # Tell transformer to use bbox refinement
            self.transformer.bbox_refine = self.bbox_embed
        else:
            self.transformer.bbox_refine = None

        self._init_weights()

    def _init_weights(self):
        # Classification bias → low initial probability (focal-loss friendly)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        if self.iterative_refine:
            for cls_layer in self.class_embed:
                nn.init.constant_(cls_layer.bias, bias_value)
            for bbox_layer in self.bbox_embed:
                nn.init.constant_(bbox_layer.layers[-1].weight, 0.0)
                nn.init.constant_(bbox_layer.layers[-1].bias, 0.0)
        else:
            nn.init.constant_(self.class_embed.bias, bias_value)
            nn.init.constant_(self.bbox_embed.layers[-1].weight, 0.0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias, 0.0)
        # Input projections
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0.0)

    def forward(self, samples, masks):
        """
        Args:
            samples: [B, 3, H, W] padded batch of images
            masks:   [B, H, W] bool (True = padding pixel)
        Returns:
            dict with:
                'pred_logits': [B, num_queries, num_classes]  (raw logits)
                'pred_boxes':  [B, num_queries, 4]            (sigmoid cxcywh)
                'aux_outputs': list of dicts (one per intermediate decoder layer)
        """
        # 1. Multi-scale backbone features
        features = self.backbone(samples)   # [C3, C4, C5]

        srcs, ms_masks, pos_embeds = [], [], []
        for lvl, feat in enumerate(features):
            src = self.input_proj[lvl](feat)
            B, C, H, W = src.shape
            mask = (
                F.interpolate(
                    masks.float().unsqueeze(1), size=(H, W), mode="nearest"
                )
                .bool()
                .squeeze(1)
            )
            pos = self.pos_enc(mask)
            srcs.append(src)
            ms_masks.append(mask)
            pos_embeds.append(pos)

        # 2. Deformable transformer
        hs_stack, ref_stack = self.transformer(
            srcs, ms_masks, pos_embeds, self.query_embed.weight
        )   # hs_stack: [n_dec, B, Q, C], ref_stack: [n_dec, B, Q, 2]

        # 3. Prediction heads
        outputs = {}
        if self.iterative_refine:
            # Each decoder layer has its own head
            all_cls = []
            all_box = []
            for lid in range(hs_stack.shape[0]):
                all_cls.append(self.class_embed[lid](hs_stack[lid]))
                raw_box = self.bbox_embed[lid](hs_stack[lid])
                ref = ref_stack[lid]  # [B, Q, 2] in [0,1]
                # cx, cy: offset in logit space → sigmoid back to [0,1]
                box_xy = (_inverse_sigmoid(ref) + raw_box[..., :2]).sigmoid()
                # w, h: direct sigmoid (absolute prediction, no reference)
                box_wh = raw_box[..., 2:].sigmoid()
                all_box.append(torch.cat([box_xy, box_wh], dim=-1))

            outputs["pred_logits"] = all_cls[-1]
            outputs["pred_boxes"] = all_box[-1]

            if self.aux_loss and self.training:
                outputs["aux_outputs"] = [
                    {"pred_logits": c, "pred_boxes": b}
                    for c, b in zip(all_cls[:-1], all_box[:-1])
                ]
        else:
            hs = hs_stack[-1]  # last decoder layer
            outputs["pred_logits"] = self.class_embed(hs)
            outputs["pred_boxes"] = self.bbox_embed(hs).sigmoid()

        return outputs


# ---------------------------------------------------------------------------
# Load COCO-pretrained Deformable DETR weights (HuggingFace SenseTime format)
# ---------------------------------------------------------------------------

def load_pretrained_deformable_detr(model, ckpt_path, verbose=True):
    """
    Load COCO-pretrained Deformable DETR weights into our model.

    Handles key name mapping between HuggingFace SenseTime/deformable-detr
    checkpoint and our custom model. Skips keys with shape mismatches
    (n_levels=4→3, num_classes=91→10, num_queries=300→N).

    Args:
        model: our DETR model instance
        ckpt_path: path to the HF pytorch_model.bin
        verbose: print loading statistics
    Returns:
        (loaded_keys, skipped_keys) tuple
    """
    import torch

    pretrained = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_sd = model.state_dict()

    # Build key mapping: pretrained_key → our_key
    mapped = {}

    for p_key, p_val in pretrained.items():
        our_key = _map_hf_key(p_key)
        if our_key is None:
            continue
        mapped[our_key] = (p_key, p_val)

    # Handle decoder self-attention: HF uses separate q/k/v_proj,
    # our nn.MultiheadAttention uses fused in_proj_weight/bias
    for lid in range(6):
        prefix = f"model.decoder.layers.{lid}.self_attn"
        our_prefix = f"transformer.decoder.{lid}.self_attn"

        # Fuse q/k/v weights → in_proj_weight [768, 256]
        q_w = pretrained.get(f"{prefix}.q_proj.weight")
        k_w = pretrained.get(f"{prefix}.k_proj.weight")
        v_w = pretrained.get(f"{prefix}.v_proj.weight")
        if q_w is not None and k_w is not None and v_w is not None:
            fused_w = torch.cat([q_w, k_w, v_w], dim=0)
            mapped[f"{our_prefix}.in_proj_weight"] = (
                f"{prefix}.{{q,k,v}}_proj.weight", fused_w
            )

        q_b = pretrained.get(f"{prefix}.q_proj.bias")
        k_b = pretrained.get(f"{prefix}.k_proj.bias")
        v_b = pretrained.get(f"{prefix}.v_proj.bias")
        if q_b is not None and k_b is not None and v_b is not None:
            fused_b = torch.cat([q_b, k_b, v_b], dim=0)
            mapped[f"{our_prefix}.in_proj_bias"] = (
                f"{prefix}.{{q,k,v}}_proj.bias", fused_b
            )

    # Load compatible weights
    loaded_keys = []
    skipped_keys = []

    for our_key, (p_key, p_val) in mapped.items():
        if our_key not in model_sd:
            skipped_keys.append((our_key, "not in model"))
            continue
        if model_sd[our_key].shape != p_val.shape:
            skipped_keys.append(
                (our_key, f"shape {tuple(p_val.shape)}→{tuple(model_sd[our_key].shape)}")
            )
            continue
        model_sd[our_key] = p_val
        loaded_keys.append(our_key)

    # Special: level_embed partial load (4→3 rows, truncate d_model if needed)
    p_le = pretrained.get("model.level_embed")
    our_le_key = "transformer.level_embed"
    if p_le is not None and our_le_key in model_sd:
        tgt_shape = model_sd[our_le_key].shape
        n = tgt_shape[0]
        d = tgt_shape[1]
        model_sd[our_le_key] = p_le[:n, :d]
        loaded_keys.append(our_le_key + f" (partial)")

    # Special: partial load of deformable attention params (4 levels → 3, d_model adapt)
    n_heads = 8
    n_points = 4
    src_levels = 4   # pretrained
    tgt_levels = 3   # ours

    # Build list of (pretrained_key, our_key, param_type) for partial loading
    partial_pairs = []
    for lid in range(6):
        # Encoder self-attn
        partial_pairs.append((
            f"model.encoder.layers.{lid}.self_attn",
            f"transformer.encoder.{lid}.self_attn",
        ))
        # Decoder cross-attn
        partial_pairs.append((
            f"model.decoder.layers.{lid}.encoder_attn",
            f"transformer.decoder.{lid}.cross_attn",
        ))

    for p_prefix, our_prefix in partial_pairs:
        # sampling_offsets weight: [n_heads * n_levels * n_points * 2, d_model]
        p_w = pretrained.get(f"{p_prefix}.sampling_offsets.weight")
        our_k = f"{our_prefix}.sampling_offsets.weight"
        if p_w is not None and our_k in model_sd:
            src_d = p_w.shape[1]
            tgt_d = model_sd[our_k].shape[1]
            w = p_w.reshape(n_heads, src_levels, n_points, 2, src_d)
            w = w[:, :tgt_levels, :, :, :tgt_d].reshape(-1, tgt_d)
            model_sd[our_k] = w
            loaded_keys.append(our_k + " (partial)")

        p_b = pretrained.get(f"{p_prefix}.sampling_offsets.bias")
        our_k = f"{our_prefix}.sampling_offsets.bias"
        if p_b is not None and our_k in model_sd:
            b = p_b.reshape(n_heads, src_levels, n_points, 2)
            model_sd[our_k] = b[:, :tgt_levels].reshape(-1)
            loaded_keys.append(our_k + " (partial)")

        # attention_weights weight: [n_heads * n_levels * n_points, d_model]
        p_w = pretrained.get(f"{p_prefix}.attention_weights.weight")
        our_k = f"{our_prefix}.attention_weights.weight"
        if p_w is not None and our_k in model_sd:
            src_d = p_w.shape[1]
            tgt_d = model_sd[our_k].shape[1]
            w = p_w.reshape(n_heads, src_levels, n_points, src_d)
            w = w[:, :tgt_levels, :, :tgt_d].reshape(-1, tgt_d)
            model_sd[our_k] = w
            loaded_keys.append(our_k + " (partial)")

        p_b = pretrained.get(f"{p_prefix}.attention_weights.bias")
        our_k = f"{our_prefix}.attention_weights.bias"
        if p_b is not None and our_k in model_sd:
            b = p_b.reshape(n_heads, src_levels, n_points)
            model_sd[our_k] = b[:, :tgt_levels].reshape(-1)
            loaded_keys.append(our_k + " (partial)")

    # query_embed: skipped (random init — COCO spatial priors hurt small digits)

    # Remove previously-skipped keys that we just handled via partial loading
    skipped_keys = [
        (k, r) for k, r in skipped_keys
        if "sampling_offsets" not in k
        and "attention_weights" not in k
    ]

    model.load_state_dict(model_sd, strict=False)

    if verbose:
        total = len(model_sd)
        print(f"[Pretrained] Loaded {len(loaded_keys)}/{total} params "
              f"({len(skipped_keys)} skipped)")
        if skipped_keys:
            print(f"  Skipped: {[f'{k} ({r})' for k, r in skipped_keys[:10]]}")

    return loaded_keys, skipped_keys


def _map_hf_key(p_key):
    """Map a single HuggingFace SenseTime key to our model key. Returns None to skip."""
    # --- Backbone: model.backbone.conv_encoder.model.X → backbone.X ---
    bb_prefix = "model.backbone.conv_encoder.model."
    if p_key.startswith(bb_prefix):
        suffix = p_key[len(bb_prefix):]
        # conv1/bn1 → stem.0/stem.1
        if suffix.startswith("conv1."):
            return "backbone.stem.0." + suffix[len("conv1."):]
        if suffix.startswith("bn1."):
            return "backbone.stem.1." + suffix[len("bn1."):]
        # layerN → backbone.layerN
        if suffix.startswith("layer"):
            return "backbone." + suffix
        return None  # skip fc, etc.

    # --- Input projection: skip (channel dims changed with C2/C3/C4 backbone) ---
    if p_key.startswith("model.input_proj."):
        return None

    # --- Encoder: model.encoder.layers.{i}.X → transformer.encoder.{i}.X ---
    if p_key.startswith("model.encoder.layers."):
        suffix = p_key[len("model.encoder.layers."):]
        # fc1/fc2 → linear1/linear2
        suffix = suffix.replace(".fc1.", ".linear1.").replace(".fc2.", ".linear2.")
        # self_attn_layer_norm → norm1, final_layer_norm → norm2
        suffix = suffix.replace(".self_attn_layer_norm.", ".norm1.")
        suffix = suffix.replace(".final_layer_norm.", ".norm2.")
        return "transformer.encoder." + suffix

    # --- Decoder: model.decoder.layers.{i}.X → transformer.decoder.{i}.X ---
    if p_key.startswith("model.decoder.layers."):
        suffix = p_key[len("model.decoder.layers."):]
        # Skip separate q/k/v_proj (handled by fusion above)
        if ".self_attn.q_proj." in suffix or ".self_attn.k_proj." in suffix or ".self_attn.v_proj." in suffix:
            return None
        # encoder_attn → cross_attn
        suffix = suffix.replace(".encoder_attn.", ".cross_attn.")
        # fc1/fc2 → linear1/linear2
        suffix = suffix.replace(".fc1.", ".linear1.").replace(".fc2.", ".linear2.")
        # norm names
        suffix = suffix.replace(".self_attn_layer_norm.", ".norm1.")
        suffix = suffix.replace(".encoder_attn_layer_norm.", ".norm2.")
        suffix = suffix.replace(".final_layer_norm.", ".norm3.")
        return "transformer.decoder." + suffix

    # --- Reference points: skip (query_embed is random, so pretrained ref_points is inconsistent) ---
    if p_key.startswith("model.reference_points."):
        return None

    # --- Query embed (skip: COCO spatial priors hurt small-digit detection) ---
    if p_key == "model.query_position_embeddings.weight":
        return None

    # --- Level embed (handled specially) ---
    if p_key == "model.level_embed":
        return None  # handled in main function

    # --- bbox_embed (skip: random init 0.0 is better for iterative refinement) ---
    if p_key.startswith("bbox_embed."):
        return None

    # --- class_embed (skip: 91 vs 10 classes) ---
    if p_key.startswith("class_embed."):
        return None

    return None
