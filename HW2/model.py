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


@torch.no_grad()
def prepare_for_cdn(targets, dn_number, label_noise_ratio, box_noise_scale,
                    num_queries, num_classes, device):
    """Prepare Contrastive De-Noising (CDN) training queries (DINO).

    Creates noised copies of GT labels/boxes as extra decoder queries.
    Even-indexed groups are positive (reconstruct GT), odd-indexed are negative
    (flipped labels → should predict low confidence).

    Args:
        targets: list[dict] with 'labels' [N_i] (1-indexed) and 'boxes' [N_i, 4]
        dn_number: controls number of denoising groups
        label_noise_ratio: probability of flipping a GT label
        box_noise_scale: noise magnitude relative to box w/h
        num_queries: number of normal detection queries
        num_classes: number of object classes
        device: torch device

    Returns:
        dn_label_ids:  [B, pad_size] int64 (0-indexed class IDs)
        dn_bbox:       [B, pad_size, 4] float (noised cxcywh in [0, 1])
        attn_mask:     [pad_size + Q, pad_size + Q] bool (True = blocked)
        dn_meta:       dict with 'pad_size', 'is_valid', 'gt_indices', 'is_negative'
    """
    batch_size = len(targets)
    known_num = [len(t['labels']) for t in targets]
    max_gt = max(known_num) if known_num else 0

    if max_gt == 0 or dn_number <= 0:
        return None, None, None, {}

    # Number of groups: half positive, half negative
    num_groups = max(dn_number // max_gt, 1)
    if num_groups % 2 == 1:
        num_groups += 1
    num_groups = max(num_groups, 2)

    single_pad = max_gt
    pad_size = num_groups * single_pad

    # Build repeated GT labels and boxes
    input_label = torch.zeros(batch_size, pad_size, dtype=torch.long, device=device)
    input_bbox = torch.zeros(batch_size, pad_size, 4, device=device)
    is_valid = torch.zeros(batch_size, pad_size, dtype=torch.bool, device=device)
    gt_indices = torch.zeros(batch_size, pad_size, dtype=torch.long, device=device)

    for b in range(batch_size):
        n = known_num[b]
        if n == 0:
            continue
        for g in range(num_groups):
            s = g * single_pad
            input_label[b, s:s + n] = targets[b]['labels']       # 1-indexed
            input_bbox[b, s:s + n] = targets[b]['boxes']
            is_valid[b, s:s + n] = True
            gt_indices[b, s:s + n] = torch.arange(n, device=device)

    # Label noise: flip with probability label_noise_ratio
    noised_label = input_label.clone()
    if label_noise_ratio > 0:
        flip = torch.rand(batch_size, pad_size, device=device) < label_noise_ratio
        flip = flip & is_valid
        rand_label = torch.randint(1, num_classes + 1, (batch_size, pad_size), device=device)
        noised_label[flip] = rand_label[flip]

    # Convert to 0-indexed for embedding; clamp invalid to 0
    dn_label_ids = (noised_label - 1).clamp(0, num_classes - 1)

    # Box noise: add noise scaled by box w/h
    noised_bbox = input_bbox.clone()
    if box_noise_scale > 0:
        box_wh = input_bbox[..., 2:].clamp(min=1e-4)  # [B, pad, 2]
        noise = (torch.rand_like(noised_bbox) * 2 - 1) * box_noise_scale
        noise[..., :2] *= box_wh   # cx, cy noise proportional to w, h
        noise[..., 2:] *= box_wh   # w, h noise proportional to w, h
        noised_bbox = (noised_bbox + noise).clamp(0, 1)
    noised_bbox[~is_valid] = 0

    # Attention mask: block cross-group and DN↔detection attention
    attn_mask = torch.ones(pad_size + num_queries, pad_size + num_queries,
                           dtype=torch.bool, device=device)
    # Detection queries attend to each other
    attn_mask[pad_size:, pad_size:] = False
    # Within each group: attend freely
    for g in range(num_groups):
        s = g * single_pad
        e = s + single_pad
        attn_mask[s:e, s:e] = False

    # Negative groups are odd-indexed (1, 3, 5, ...)
    group_per_pos = torch.arange(pad_size, device=device) // single_pad
    is_negative = (group_per_pos % 2 == 1).unsqueeze(0).expand(batch_size, -1)

    dn_meta = {
        'pad_size': pad_size,
        'single_pad': single_pad,
        'num_groups': num_groups,
        'is_valid': is_valid,        # [B, pad_size]
        'gt_indices': gt_indices,    # [B, pad_size]
        'is_negative': is_negative,  # [B, pad_size]
    }
    return dn_label_ids, noised_bbox, attn_mask, dn_meta


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
    """ResNet-50 backbone returning multi-scale features (C3, C4, C5).

    Standard Deformable DETR feature levels:
    - C3 (stride 8,  512 ch):  fine detail
    - C4 (stride 16, 1024 ch): main feature level
    - C5 (stride 32, 2048 ch): context / large receptive field
    """

    def __init__(self, pretrained=True, train_backbone=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet50(weights=weights)

        self.stem = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool
        )
        self.layer1 = net.layer1   # stride 4,  256 ch  → C2 (not returned)
        self.layer2 = net.layer2   # stride 8,  512 ch  → C3
        self.layer3 = net.layer3   # stride 16, 1024 ch → C4
        self.layer4 = net.layer4   # stride 32, 2048 ch → C5

        self.num_channels = [512, 1024, 2048]
        self.train_backbone = train_backbone

        if not train_backbone:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c3, c4, c5]

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
        self_attn_mask=None,
    ):
        # Self-attention among object queries
        q = k = tgt + query_pos
        tgt2, _ = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)
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
        two_stage=False,
        two_stage_num_proposals=300,
        num_classes=10,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.use_checkpoint = use_checkpoint
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

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

        if two_stage:
            # Two-stage: encoder output → proposal class + bbox
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.enc_cls_head = nn.Linear(d_model, num_classes)
            self.enc_bbox_head = MLP(d_model, d_model, 4, 3)
            # Project selected encoder features → query_pos + query_tgt
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            # Project query embedding → reference point (x, y)
            self.reference_points_linear = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.level_embed)
        if self.two_stage:
            nn.init.xavier_uniform_(self.enc_output.weight)
            nn.init.xavier_uniform_(self.pos_trans.weight)
        else:
            nn.init.xavier_uniform_(self.reference_points_linear.weight)
            nn.init.constant_(self.reference_points_linear.bias, 0.0)

    def _get_proposal_pos_embed(self, proposals):
        """Generate sinusoidal positional embedding from proposal boxes [B, N, 4]."""
        num_pos_feats = self.d_model // 2
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

        # proposals: [B, N, 4] (cx, cy, w, h) in [0, 1]
        proposals = proposals * scale
        # [B, N, 4, num_pos_feats]
        pos = proposals[:, :, :, None] / dim_t
        # sin/cos interleaved → [B, N, 4*num_pos_feats] = [B, N, 2*d_model]
        pos = torch.stack([pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()], dim=4)
        pos = pos.flatten(2)
        return pos

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

    def _get_reference_pos_embed(self, ref_points):
        """Convert [B, N, 2] reference points → [B, N, C] sinusoidal positional embedding."""
        num_pos_feats = self.d_model // 2   # C/2 for x, C/2 for y
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=ref_points.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

        ref = ref_points * scale    # [B, N, 2]
        pos_x = ref[..., 0:1] / dim_t  # [B, N, feats]
        pos_y = ref[..., 1:2] / dim_t
        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(-2)
        return torch.cat([pos_y, pos_x], dim=-1)  # [B, N, C]

    def forward(self, srcs, masks, pos_embeds, query_embed,
                dn_label_embed=None, dn_bbox=None, dn_attn_mask=None,
                tgt_embed_override=None):
        """
        Args:
            srcs:       list of [B, C, Hi, Wi] projected feature maps
            masks:      list of [B, Hi, Wi] boolean masks
            pos_embeds: list of [B, C, Hi, Wi] positional encodings
            query_embed: [Q, 2*C] (split into query_pos + query_tgt)
            dn_label_embed: [B, pad_size, C] denoising label embeddings (optional)
            dn_bbox:        [B, pad_size, 4] noised boxes for DN queries (optional)
            dn_attn_mask:   [pad+Q, pad+Q] bool attention mask (optional)
            tgt_embed_override: [B, Q, C] learnable target embedding (optional)
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
        enc_outputs_dict = None
        if self.two_stage:
            # Score each encoder token and select top-K proposals
            enc_out = self.enc_output_norm(self.enc_output(memory))  # [B, Σ, C]
            enc_cls_logits = self.enc_cls_head(enc_out)  # [B, Σ, num_classes]
            enc_bbox_raw = self.enc_bbox_head(enc_out)   # [B, Σ, 4]

            # Build reference point per encoder token from grid centers
            enc_ref_xy = self._get_encoder_reference_points(
                spatial_shapes, memory.device
            )[:, :, 0, :]  # [1, Σ, 2]
            enc_ref_xy = enc_ref_xy.expand(B, -1, -1)  # [B, Σ, 2]
            # Bbox proposal: offset + reference → sigmoid
            enc_proposals = torch.cat([
                (_inverse_sigmoid(enc_ref_xy) + enc_bbox_raw[..., :2]).sigmoid(),
                enc_bbox_raw[..., 2:].sigmoid(),
            ], dim=-1)  # [B, Σ, 4] cxcywh in [0,1]

            # Select top-K by max class score (across classes)
            topk_score = enc_cls_logits.max(-1)[0]  # [B, Σ]
            # Mask out padding positions
            topk_score = topk_score.masked_fill(mask_flatten, float('-inf'))
            topk_k = min(self.two_stage_num_proposals, topk_score.shape[1])
            topk_indices = topk_score.topk(topk_k, dim=1)[1]  # [B, K]

            # Gather selected proposals and features
            topk_proposals = torch.gather(
                enc_proposals, 1,
                topk_indices.unsqueeze(-1).expand(-1, -1, 4)
            )  # [B, K, 4]
            topk_features = torch.gather(
                enc_out, 1,
                topk_indices.unsqueeze(-1).expand(-1, -1, C)
            )  # [B, K, C]

            # Reference points from proposals (detach to stop gradient)
            dec_ref = topk_proposals[..., :2].detach()  # [B, K, 2]
            dec_ref = dec_ref[:, :, None, :].repeat(1, 1, self.n_levels, 1)

            # Generate query_pos + query_tgt from proposals
            pos_embed = self._get_proposal_pos_embed(topk_proposals.detach())
            query_both = self.pos_trans_norm(self.pos_trans(pos_embed))
            query_pos, query_tgt = query_both.split(C, dim=-1)

            # embed_init_tgt: use learnable content instead of encoder-derived
            if tgt_embed_override is not None:
                query_tgt = tgt_embed_override

            enc_outputs_dict = {
                "enc_cls_logits": enc_cls_logits,
                "enc_bbox_proposals": enc_proposals,
                "topk_indices": topk_indices,
            }
        else:
            query_pos, query_tgt = query_embed.split(C, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(B, -1, -1)
            query_tgt = query_tgt.unsqueeze(0).expand(B, -1, -1)

            # embed_init_tgt: use learnable content instead of query_embed split
            if tgt_embed_override is not None:
                query_tgt = tgt_embed_override

            # Reference points from query_pos → sigmoid → [0, 1]
            dec_ref = self.reference_points_linear(query_pos).sigmoid()
            # → [B, Q, n_levels, 2]
            dec_ref = dec_ref[:, :, None, :].repeat(1, 1, self.n_levels, 1)

        # --- prepend denoising queries (CDN) ----------------------------
        if dn_label_embed is not None:
            dn_ref = dn_bbox[..., :2]  # box centers [B, pad, 2]
            dn_query_pos = self._get_reference_pos_embed(dn_ref)
            dn_ref_expanded = dn_ref[:, :, None, :].repeat(1, 1, self.n_levels, 1)

            query_tgt = torch.cat([dn_label_embed, query_tgt], dim=1)
            query_pos = torch.cat([dn_query_pos, query_pos], dim=1)
            dec_ref = torch.cat([dn_ref_expanded, dec_ref], dim=1)

        hs = query_tgt
        intermediate = []
        intermediate_ref = []
        for lid, layer in enumerate(self.decoder):
            hs = layer(
                hs, query_pos, dec_ref,
                memory, spatial_shapes, mask_flatten,
                self_attn_mask=dn_attn_mask,
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

        return hs_stack, ref_stack, enc_outputs_dict


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
        two_stage=False,
        two_stage_num_proposals=300,
        use_dn=False,
        dn_number=100,
        dn_label_noise_ratio=0.5,
        dn_box_noise_scale=1.0,
        embed_init_tgt=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.n_levels = n_levels
        self.aux_loss = aux_loss
        self.iterative_refine = iterative_refine
        self.two_stage = two_stage
        self.use_dn = use_dn
        self.dn_number = dn_number
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_box_noise_scale = dn_box_noise_scale
        self.embed_init_tgt = embed_init_tgt

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
        # Extra conv levels: stride-64 from C5 (standard Deformable DETR)
        num_backbone_levels = len(self.backbone.num_channels)
        if n_levels > num_backbone_levels:
            for _ in range(n_levels - num_backbone_levels):
                in_ch = self.backbone.num_channels[-1] if len(self.input_proj) == num_backbone_levels else hidden_dim
                self.input_proj.append(nn.Sequential(
                    nn.Conv2d(in_ch, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))

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
            two_stage=two_stage,
            two_stage_num_proposals=two_stage_num_proposals,
            num_classes=num_classes,
        )

        # query_embed: only needed for one-stage
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        else:
            self.query_embed = None

        # --- denoising training (CDN) ---------------------------------
        if use_dn:
            self.label_enc = nn.Embedding(num_classes, hidden_dim)

        # --- learnable target embedding (DINO embed_init_tgt) ---------
        if embed_init_tgt:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
            nn.init.normal_(self.tgt_embed.weight)

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

    def forward(self, samples, masks, targets=None):
        """
        Args:
            samples: [B, 3, H, W] padded batch of images
            masks:   [B, H, W] bool (True = padding pixel)
            targets: list[dict] with 'labels' and 'boxes' (training only, for DN)
        Returns:
            dict with:
                'pred_logits': [B, num_queries, num_classes]  (raw logits)
                'pred_boxes':  [B, num_queries, 4]            (sigmoid cxcywh)
                'aux_outputs': list of dicts (one per intermediate decoder layer)
                'dn_outputs':  dict (training only, for DN loss)
                'dn_meta':     dict (training only, for DN loss)
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

        # Extra feature levels from additional conv layers (stride 64, 128, ...)
        num_backbone_levels = len(features)
        if self.n_levels > num_backbone_levels:
            last_feat = features[-1]
            for lvl in range(num_backbone_levels, self.n_levels):
                if lvl == num_backbone_levels:
                    src = self.input_proj[lvl](last_feat)
                else:
                    src = self.input_proj[lvl](srcs[-1])
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

        # 2. Prepare DN queries (training only)
        dn_label_embed = dn_bbox = dn_attn_mask = None
        dn_meta = {}
        if self.use_dn and self.training and targets is not None:
            dn_label_ids, dn_bbox, dn_attn_mask, dn_meta = prepare_for_cdn(
                targets, self.dn_number, self.dn_label_noise_ratio,
                self.dn_box_noise_scale, self.num_queries,
                self.num_classes, samples.device,
            )
            if dn_label_ids is not None:
                dn_label_embed = self.label_enc(dn_label_ids)  # [B, pad, C]
                dn_label_embed[~dn_meta['is_valid']] = 0

        # 3. Learnable target embedding (embed_init_tgt)
        tgt_override = None
        if self.embed_init_tgt:
            tgt_override = self.tgt_embed.weight.unsqueeze(0).expand(B, -1, -1)

        # 4. Deformable transformer
        query_embed_w = self.query_embed.weight if self.query_embed is not None else None
        hs_stack, ref_stack, enc_outputs_dict = self.transformer(
            srcs, ms_masks, pos_embeds, query_embed_w,
            dn_label_embed=dn_label_embed, dn_bbox=dn_bbox,
            dn_attn_mask=dn_attn_mask, tgt_embed_override=tgt_override,
        )   # hs_stack: [n_dec, B, pad+Q, C], ref_stack: [n_dec, B, pad+Q, 2]

        # 5. Split DN queries from detection queries
        dn_hs_stack = None
        dn_ref_stack = None
        pad_size = dn_meta.get('pad_size', 0)
        if pad_size > 0:
            dn_hs_stack = hs_stack[:, :, :pad_size, :]
            hs_stack = hs_stack[:, :, pad_size:, :]
            dn_ref_stack = ref_stack[:, :, :pad_size, :]
            ref_stack = ref_stack[:, :, pad_size:, :]

        # 6. Prediction heads (detection queries only)
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

        # Two-stage encoder output loss (for training)
        if self.two_stage and enc_outputs_dict is not None and self.training:
            outputs["enc_outputs"] = enc_outputs_dict

        # 7. DN output predictions (for loss computation during training)
        if dn_hs_stack is not None and self.training:
            if self.iterative_refine:
                dn_cls_all = []
                dn_box_all = []
                for lid in range(dn_hs_stack.shape[0]):
                    dn_cls_all.append(self.class_embed[lid](dn_hs_stack[lid]))
                    raw_box = self.bbox_embed[lid](dn_hs_stack[lid])
                    ref = dn_ref_stack[lid]
                    box_xy = (_inverse_sigmoid(ref) + raw_box[..., :2]).sigmoid()
                    box_wh = raw_box[..., 2:].sigmoid()
                    dn_box_all.append(torch.cat([box_xy, box_wh], dim=-1))
                outputs['dn_outputs'] = {
                    'pred_logits': dn_cls_all[-1],
                    'pred_boxes': dn_box_all[-1],
                }
                if self.aux_loss:
                    outputs['dn_outputs']['aux_outputs'] = [
                        {'pred_logits': c, 'pred_boxes': b}
                        for c, b in zip(dn_cls_all[:-1], dn_box_all[:-1])
                    ]
            else:
                dn_hs = dn_hs_stack[-1]
                outputs['dn_outputs'] = {
                    'pred_logits': self.class_embed(dn_hs),
                    'pred_boxes': self.bbox_embed(dn_hs).sigmoid(),
                }
            outputs['dn_meta'] = dn_meta

        return outputs


