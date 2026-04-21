"""
DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection.

Complete single-file implementation: backbone, positional encoding, deformable
transformer with 4D reference points, denoising training, matcher, criterion,
and postprocessor.
"""
import copy
import math
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.boxes import nms
from scipy.optimize import linear_sum_assignment

# ── inlined utilities (box ops, image batching, helpers) ─────────────────────


class ImageBatch:
    """Padded batch of images together with boolean padding masks."""
    __slots__ = ["tensors", "mask"]

    def __init__(self, pixels, pad_mask):
        self.tensors = pixels
        self.mask = pad_mask

    def to(self, device):
        return ImageBatch(
            self.tensors.to(device),
            self.mask.to(device) if self.mask is not None else None)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return repr(self.tensors)


def make_image_batch(images: List[torch.Tensor]) -> ImageBatch:
    """Pad a list of [C,H,W] tensors into a single batch with masks."""
    mh = max(im.shape[1] for im in images)
    mw = max(im.shape[2] for im in images)
    out = images[0].new_zeros(len(images), 3, mh, mw)
    masks = torch.ones(len(images),
                       mh,
                       mw,
                       dtype=torch.bool,
                       device=images[0].device)
    for k, im in enumerate(images):
        _, h, w = im.shape
        out[k, :, :h, :w] = im
        masks[k, :h, :w] = False
    return ImageBatch(out, masks)


def _cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w * .5, cy - h * .5, cx + w * .5, cy + h * .5], -1)


def _pairwise_iou(a, b):
    a1 = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    a2 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    top_left = torch.max(a[:, None, :2], b[None, :, :2])
    bot_right = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (bot_right - top_left).clamp(min=0)
    overlap = wh[..., 0] * wh[..., 1]
    union = a1[:, None] + a2[None, :] - overlap
    return overlap / union, union


def _generalized_iou(a, b):
    assert (a[:, 2:] >= a[:, :2]).all()
    assert (b[:, 2:] >= b[:, :2]).all()
    iou, union = _pairwise_iou(a, b)
    top_left = torch.min(a[:, None, :2], b[None, :, :2])
    bot_right = torch.max(a[:, None, 2:], b[None, :, 2:])
    wh = (bot_right - top_left).clamp(min=0)
    enclosing = wh[..., 0] * wh[..., 1]
    return iou - (enclosing - union) / enclosing


def _inv_sigmoid(t, eps=1e-5):
    t = t.clamp(0., 1.)
    return torch.log(t.clamp(min=eps) / (1 - t).clamp(min=eps))


@torch.no_grad()
def _topk_accuracy(logits, labels, k=(1,)):
    if labels.numel() == 0:
        return [torch.zeros([], device=logits.device)]
    mk = max(k)
    _, pred = logits.topk(mk, 1, True, True)
    hit = pred.t().eq(labels.view(1, -1).expand_as(pred.t()))
    return [
        hit[:ki].reshape(-1).float().sum() * (100. / labels.size(0)) for ki in k
    ]


def _is_distributed():
    return False


def _world_size():
    return 1


# ===========================================================================
# Building Blocks
# ===========================================================================


class MLP(nn.Module):
    """Feed-forward network with ReLU activations."""

    def __init__(self, in_dim, hid_dim, out_dim, depth):
        super().__init__()
        self.depth = depth
        dims = [in_dim] + [hid_dim] * (depth - 1) + [out_dim]
        self.layers = nn.ModuleList(
            nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.depth - 1 else layer(x)
        return x


def _sinusoidal_pe(pos_tensor, half_dim=128):
    """Sine positional embedding for 2D or 4D position tensors.

    Args:
        pos_tensor: [B, N, 2] or [B, N, 4] with values in [0,1]
        half_dim: number of features per spatial dimension
    Returns:
        [B, N, 2*half_dim] or [B, N, 4*half_dim]
    """
    scale = 2 * math.pi
    freq = torch.arange(half_dim, dtype=torch.float32, device=pos_tensor.device)
    freq = 10000**(2 * (freq // 2) / half_dim)

    x_enc = pos_tensor[:, :, 0] * scale
    y_enc = pos_tensor[:, :, 1] * scale
    px = x_enc[:, :, None] / freq
    py = y_enc[:, :, None] / freq
    px = torch.stack([px[:, :, 0::2].sin(), px[:, :, 1::2].cos()],
                     dim=3).flatten(2)
    py = torch.stack([py[:, :, 0::2].sin(), py[:, :, 1::2].cos()],
                     dim=3).flatten(2)

    ndim = pos_tensor.size(-1)
    if ndim == 2:
        return torch.cat([py, px], dim=2)
    elif ndim == 4:
        w_enc = pos_tensor[:, :, 2] * scale
        h_enc = pos_tensor[:, :, 3] * scale
        pw = w_enc[:, :, None] / freq
        ph = h_enc[:, :, None] / freq
        pw = torch.stack([pw[:, :, 0::2].sin(), pw[:, :, 1::2].cos()],
                         dim=3).flatten(2)
        ph = torch.stack([ph[:, :, 0::2].sin(), ph[:, :, 1::2].cos()],
                         dim=3).flatten(2)
        return torch.cat([py, px, pw, ph], dim=2)
    else:
        raise ValueError(f"pos_tensor last dim must be 2 or 4, got {ndim}")


def _generate_grid_proposals(memory, pad_mask, spatial_shapes, learned_wh=None):
    """Generate 4D (cx,cy,w,h) proposals from encoder output grid.

    For two-stage DINO: each encoder token gets a proposal box centered
    at its grid position with level-dependent default size.
    """
    B, S, C = memory.shape
    proposals = []
    offset = 0
    for lvl, (H, W) in enumerate(spatial_shapes):
        level_mask = pad_mask[:, offset:offset + H * W].view(B, H, W, 1)
        valid_h = torch.sum(~level_mask[:, :, 0, 0], 1)
        valid_w = torch.sum(~level_mask[:, 0, :, 0], 1)

        gy, gx = torch.meshgrid(torch.linspace(0,
                                               H - 1,
                                               H,
                                               dtype=torch.float32,
                                               device=memory.device),
                                torch.linspace(0,
                                               W - 1,
                                               W,
                                               dtype=torch.float32,
                                               device=memory.device),
                                indexing='ij')
        grid = torch.cat([gx.unsqueeze(-1), gy.unsqueeze(-1)], -1)
        scale = torch.cat([valid_w.unsqueeze(-1),
                           valid_h.unsqueeze(-1)], 1).view(B, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(B, -1, -1, -1) + 0.5) / scale

        if learned_wh is not None:
            wh = torch.ones_like(grid) * learned_wh.sigmoid() * (2.0**lvl)
        else:
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)

        prop = torch.cat([grid, wh], -1).view(B, -1, 4)
        proposals.append(prop)
        offset += H * W

    all_proposals = torch.cat(proposals, 1)
    valid_mask = ((all_proposals > 0.01) & (all_proposals < 0.99)).all(
        -1, keepdim=True)

    # Convert to logit space (inverse sigmoid)
    all_proposals = torch.log(all_proposals / (1 - all_proposals))
    all_proposals = all_proposals.masked_fill(pad_mask.unsqueeze(-1),
                                              float('inf'))
    all_proposals = all_proposals.masked_fill(~valid_mask, float('inf'))

    out_memory = memory.masked_fill(pad_mask.unsqueeze(-1), 0.0)
    out_memory = out_memory.masked_fill(~valid_mask, 0.0)
    return out_memory, all_proposals


def _copy_layers(module, n, share=False):
    """Create n copies of a module (shared or deep-copied)."""
    if share:
        return nn.ModuleList([module] * n)
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


# ===========================================================================
# Multi-Scale Deformable Attention (Pure PyTorch)
# ===========================================================================


def _ms_deform_forward(value, spatial_shapes, sampling_locs, attn_weights):
    """Pure PyTorch multi-scale deformable attention via grid_sample.

    Args:
        value: [B, sum(HiWi), nheads, head_dim]
        spatial_shapes: list of (H, W) per level
        sampling_locs: [B, Lq, nheads, n_levels, n_points, 2] in [0,1]
        attn_weights: [B, Lq, nheads, n_levels, n_points]
    """
    B, _, nheads, hdim = value.shape
    _, Lq, _, n_lvl, n_pts, _ = sampling_locs.shape

    splits = [H * W for H, W in spatial_shapes]
    vals_per_level = value.split(splits, dim=1)

    grids = 2.0 * sampling_locs - 1.0  # [0,1] -> [-1,1]

    sampled = []
    for lid, (H, W) in enumerate(spatial_shapes):
        val = (vals_per_level[lid].permute(0, 2, 3,
                                           1).reshape(B * nheads, hdim, H, W))
        grid = (grids[:, :, :, lid].permute(0, 2, 1, 3,
                                            4).reshape(B * nheads, Lq, n_pts,
                                                       2))
        s = F.grid_sample(val,
                          grid,
                          mode='bilinear',
                          padding_mode='zeros',
                          align_corners=False)
        sampled.append(s)

    # [BH, hdim, Lq, n_lvl*n_pts]
    sampled = torch.stack(sampled, dim=-1).reshape(B * nheads, hdim, Lq,
                                                   n_lvl * n_pts)

    weights = (attn_weights.permute(0, 2, 1, 3,
                                    4).reshape(B * nheads, 1, Lq,
                                               n_lvl * n_pts))

    out = (sampled * weights).sum(-1)
    return out.reshape(B, nheads * hdim, Lq).permute(0, 2, 1)


class MSDeformAttn(nn.Module):
    """Multi-Scale Deformable Attention (pure PyTorch, supports 2D and 4D refs)."""

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        self.sampling_offsets = nn.Linear(d_model,
                                          n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model,
                                           n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        angles = torch.arange(
            self.n_heads, dtype=torch.float32) * (2 * math.pi / self.n_heads)
        grid = torch.stack([angles.cos(), angles.sin()], -1)
        grid = grid / grid.abs().max(-1, keepdim=True)[0]
        grid = grid.view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels,
                                                       self.n_points, 1)
        for i in range(self.n_points):
            grid[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid.view(-1))
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(self,
                query,
                reference_points,
                value,
                spatial_shapes,
                level_start_index,
                padding_mask=None):
        """
        query: [B, Lq, C]
        reference_points: [B, Lq, n_levels, 2] or [B, Lq, n_levels, 4]
        value: [B, sum(HiWi), C]
        spatial_shapes: [(H0,W0), ...]  (list of tuples or Tensor)
        """
        B, Lq, _ = query.shape

        val = self.value_proj(value)
        if padding_mask is not None:
            val = val.masked_fill(padding_mask[..., None], 0.0)
        val = val.view(B, -1, self.n_heads, self.head_dim)

        offsets = self.sampling_offsets(query).view(B, Lq, self.n_heads,
                                                    self.n_levels,
                                                    self.n_points, 2)
        weights = self.attention_weights(query).view(
            B, Lq, self.n_heads, self.n_levels * self.n_points)
        weights = F.softmax(weights, -1).view(B, Lq, self.n_heads,
                                              self.n_levels, self.n_points)

        # Handle spatial_shapes as tensor or list
        if isinstance(spatial_shapes, torch.Tensor):
            shapes_list = [(int(spatial_shapes[i, 0]), int(spatial_shapes[i,
                                                                          1]))
                           for i in range(spatial_shapes.shape[0])]
            offset_norm = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1).float()
        else:
            shapes_list = spatial_shapes
            offset_norm = torch.tensor([[W, H] for H, W in spatial_shapes],
                                       dtype=torch.float32,
                                       device=query.device)

        ref_dim = reference_points.shape[-1]
        if ref_dim == 2:
            sampling_locs = (
                reference_points[:, :, None, :, None, :] +
                offsets / offset_norm[None, None, None, :, None, :])
        elif ref_dim == 4:
            sampling_locs = (reference_points[:, :, None, :, None, :2] +
                             offsets / self.n_points *
                             reference_points[:, :, None, :, None, 2:] * 0.5)
        else:
            raise ValueError(
                f"reference_points last dim must be 2 or 4, got {ref_dim}")

        out = _ms_deform_forward(val, shapes_list, sampling_locs, weights)
        return self.output_proj(out)


# ===========================================================================
# Positional Encoding (Sine with separate H/W temperatures)
# ===========================================================================


class SinePositionEmbed2D(nn.Module):
    """2D sine positional encoding with independent H and W temperatures."""

    def __init__(self,
                 num_feats=64,
                 tempH=10000,
                 tempW=10000,
                 normalize=False,
                 scale=None):
        super().__init__()
        self.num_feats = num_feats
        self.tempH = tempH
        self.tempW = tempW
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError("normalize must be True when scale is given")
        self.scale = scale or 2 * math.pi

    def forward(self, nested: ImageBatch):
        x = nested.tensors
        mask = nested.mask
        assert mask is not None
        not_mask = ~mask
        y_cum = not_mask.cumsum(1, dtype=torch.float32)
        x_cum = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_cum = y_cum / (y_cum[:, -1:, :] + eps) * self.scale
            x_cum = x_cum / (x_cum[:, :, -1:] + eps) * self.scale

        freq_x = torch.arange(self.num_feats,
                              dtype=torch.float32,
                              device=x.device)
        freq_x = self.tempW**(2 * (freq_x // 2) / self.num_feats)
        freq_y = torch.arange(self.num_feats,
                              dtype=torch.float32,
                              device=x.device)
        freq_y = self.tempH**(2 * (freq_y // 2) / self.num_feats)

        pe_x = x_cum[:, :, :, None] / freq_x
        pe_y = y_cum[:, :, :, None] / freq_y

        pe_x = torch.stack(
            [pe_x[:, :, :, 0::2].sin(), pe_x[:, :, :, 1::2].cos()],
            dim=4).flatten(3)
        pe_y = torch.stack(
            [pe_y[:, :, :, 0::2].sin(), pe_y[:, :, :, 1::2].cos()],
            dim=4).flatten(3)

        return torch.cat([pe_y, pe_x], dim=3).permute(0, 3, 1, 2)


# ===========================================================================
# Backbone: ResNet with FrozenBatchNorm2d
# ===========================================================================


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d with frozen running stats and affine parameters."""

    def __init__(self, n):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        k = prefix + 'num_batches_tracked'
        if k in state_dict:
            del state_dict[k]
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        s = w * (rv + 1e-5).rsqrt()
        return x * s + (b - rm * s)


class BackboneBase(nn.Module):
    """Wraps a torchvision backbone to return intermediate features."""

    def __init__(self, backbone, train_backbone, num_channels, return_indices):
        super().__init__()
        for name, param in backbone.named_parameters():
            if not train_backbone or ('layer2' not in name and
                                      'layer3' not in name and
                                      'layer4' not in name):
                param.requires_grad_(False)
        # Map return_indices to layer names
        # return_indices [1,2,3] -> layer2, layer3, layer4
        layer_map = {}
        for idx, li in enumerate(return_indices):
            layer_name = f"layer{li + 1}"
            layer_map[layer_name] = str(idx)
        self.body = IntermediateLayerGetter(backbone, return_layers=layer_map)
        self.num_channels = num_channels

    def forward(self, tensor_list: ImageBatch):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, ImageBatch] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            mask = F.interpolate(m[None].float(),
                                 size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = ImageBatch(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with FrozenBatchNorm2d and multi-scale outputs."""

    def __init__(self,
                 name,
                 train_backbone,
                 dilation,
                 return_indices,
                 norm_layer=FrozenBatchNorm2d):
        net = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            weights="IMAGENET1K_V1",
            norm_layer=norm_layer)
        assert name in ('resnet50', 'resnet101')
        all_ch = [256, 512, 1024, 2048]
        ch = [all_ch[i] for i in return_indices]
        super().__init__(net, train_backbone, ch, return_indices)


class Joiner(nn.Sequential):
    """Joins backbone and positional encoding."""

    def __init__(self, backbone, pos_enc):
        super().__init__(backbone, pos_enc)

    def forward(self, tensor_list: ImageBatch):
        features = self[0](tensor_list)
        out = []
        pos = []
        for name, x in features.items():
            out.append(x)
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos


# ===========================================================================
# Deformable Transformer
# ===========================================================================


class DefEncoderBlock(nn.Module):

    def __init__(self,
                 d_model=256,
                 d_ffn=1024,
                 dropout=0.1,
                 activation="relu",
                 n_levels=4,
                 n_heads=8,
                 n_points=4):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.drop1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ffn)
        self.act = nn.ReLU() if activation == "relu" else nn.GELU()
        self.drop2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ffn, d_model)
        self.drop3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def _add_pos(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self,
                src,
                pos,
                ref_pts,
                spatial_shapes,
                level_start_index,
                pad_mask=None):
        q = self._add_pos(src, pos)
        src2 = self.self_attn(q, ref_pts, src, spatial_shapes,
                              level_start_index, pad_mask)
        src = src + self.drop1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.drop2(self.act(self.fc1(src))))
        src = src + self.drop3(src2)
        src = self.norm2(src)
        return src


class DefDecoderBlock(nn.Module):

    def __init__(self,
                 d_model=256,
                 d_ffn=1024,
                 dropout=0.1,
                 activation="relu",
                 n_levels=4,
                 n_heads=8,
                 n_points=4,
                 module_seq=None):
        super().__init__()
        self.module_seq = module_seq or ['sa', 'ca', 'ffn']

        # Cross-attention (deformable)
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.drop_ca = nn.Dropout(dropout)
        self.norm_ca = nn.LayerNorm(d_model)

        # Self-attention (standard MHA)
        self.self_attn = nn.MultiheadAttention(d_model,
                                               n_heads,
                                               dropout=dropout)
        self.drop_sa = nn.Dropout(dropout)
        self.norm_sa = nn.LayerNorm(d_model)

        # FFN
        self.fc1 = nn.Linear(d_model, d_ffn)
        self.act = nn.ReLU() if activation == "relu" else nn.GELU()
        self.drop_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ffn, d_model)
        self.drop_ffn = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(d_model)

    @staticmethod
    def _add_pos(t, p):
        return t if p is None else t + p

    def _do_sa(self, tgt, qpos=None, self_attn_mask=None, **kw):
        q = k = self._add_pos(tgt, qpos)
        tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
        tgt = tgt + self.drop_sa(tgt2)
        return self.norm_sa(tgt)

    def _do_ca(self,
               tgt,
               qpos=None,
               ref_pts=None,
               mem=None,
               mem_pad_mask=None,
               mem_start_idx=None,
               mem_shapes=None,
               **kw):
        tgt2 = self.cross_attn(
            self._add_pos(tgt, qpos).transpose(0, 1),
            ref_pts.transpose(0, 1).contiguous(), mem.transpose(0, 1),
            mem_shapes, mem_start_idx, mem_pad_mask).transpose(0, 1)
        tgt = tgt + self.drop_ca(tgt2)
        return self.norm_ca(tgt)

    def _do_ffn(self, tgt, **kw):
        tgt2 = self.fc2(self.drop_fc(self.act(self.fc1(tgt))))
        tgt = tgt + self.drop_ffn(tgt2)
        return self.norm_ffn(tgt)

    def forward(self,
                tgt,
                qpos=None,
                query_sine=None,
                tgt_key_padding_mask=None,
                ref_pts=None,
                mem=None,
                mem_pad_mask=None,
                mem_start_idx=None,
                mem_shapes=None,
                mem_pos=None,
                self_attn_mask=None,
                cross_attn_mask=None):
        kw = dict(qpos=qpos,
                  ref_pts=ref_pts,
                  mem=mem,
                  mem_pad_mask=mem_pad_mask,
                  mem_start_idx=mem_start_idx,
                  mem_shapes=mem_shapes,
                  mem_pos=mem_pos,
                  self_attn_mask=self_attn_mask)
        for fn_name in self.module_seq:
            if fn_name == 'sa':
                tgt = self._do_sa(tgt, **kw)
            elif fn_name == 'ca':
                tgt = self._do_ca(tgt, **kw)
            elif fn_name == 'ffn':
                tgt = self._do_ffn(tgt, **kw)
        return tgt


class MSEncoder(nn.Module):

    def __init__(self,
                 layer,
                 num_layers,
                 norm=None,
                 d_model=256,
                 deformable=True,
                 layer_share=False):
        super().__init__()
        if num_layers > 0:
            self.layers = _copy_layers(layer, num_layers, share=layer_share)
        else:
            self.layers = []
        self.num_layers = num_layers
        self.norm = norm
        self.d_model = d_model
        self.deformable = deformable

    @staticmethod
    def _build_ref_points(spatial_shapes, valid_ratios, device):
        """MSEncoder reference points as 2D grid centers, scaled by valid_ratios."""
        refs = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ry, rx = torch.meshgrid(torch.linspace(0.5,
                                                   H - 0.5,
                                                   H,
                                                   dtype=torch.float32,
                                                   device=device),
                                    torch.linspace(0.5,
                                                   W - 0.5,
                                                   W,
                                                   dtype=torch.float32,
                                                   device=device),
                                    indexing='ij')
            ry = ry.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            rx = rx.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            refs.append(torch.stack([rx, ry], -1))
        refs = torch.cat(refs, 1)  # [B, sum(HW), 2]
        return refs[:, :,
                    None] * valid_ratios[:, None]  # [B, sum(HW), n_levels, 2]

    def forward(self, src, pos, spatial_shapes, level_start_index, valid_ratios,
                pad_mask, **kw):
        out = src
        if self.num_layers > 0 and self.deformable:
            ref_pts = self._build_ref_points(spatial_shapes,
                                             valid_ratios,
                                             device=src.device)
        for layer in self.layers:
            out = layer(src=out,
                        pos=pos,
                        ref_pts=ref_pts,
                        spatial_shapes=spatial_shapes,
                        level_start_index=level_start_index,
                        pad_mask=pad_mask)
        if self.norm is not None:
            out = self.norm(out)
        return out


class MSDecoder(nn.Module):

    def __init__(self,
                 layer,
                 num_layers,
                 norm=None,
                 return_intermediate=False,
                 d_model=256,
                 query_dim=4,
                 n_feature_levels=1,
                 deformable=True,
                 use_detached_boxes=False):
        super().__init__()
        if num_layers > 0:
            self.layers = _copy_layers(layer, num_layers)
        else:
            self.layers = []
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.query_dim = query_dim
        self.d_model = d_model
        self.deformable = deformable
        self.use_detached_boxes = use_detached_boxes

        # Convert 4D sine embeddings -> query_pos
        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        self.bbox_embed = None
        self.class_embed = None

    def forward(self,
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                pos=None,
                refpoints_unsigmoid=None,
                level_start_index=None,
                spatial_shapes=None,
                valid_ratios=None):
        output = tgt
        intermediate = []
        ref_sig = refpoints_unsigmoid.sigmoid()
        ref_pts_list = [ref_sig]

        for lid, layer in enumerate(self.layers):
            if ref_sig.shape[-1] == 4:
                ref_input = (
                    ref_sig[:, :, None] *
                    torch.cat([valid_ratios, valid_ratios], -1)[None, :])
            else:
                ref_input = ref_sig[:, :, None] * valid_ratios[None, :]

            q_sine = _sinusoidal_pe(ref_input[:, :, 0, :], self.d_model // 2)

            raw_qpos = self.ref_point_head(q_sine)

            output = layer(tgt=output,
                           qpos=raw_qpos,
                           query_sine=q_sine,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           ref_pts=ref_input,
                           mem=memory,
                           mem_pad_mask=memory_key_padding_mask,
                           mem_start_idx=level_start_index,
                           mem_shapes=spatial_shapes,
                           mem_pos=pos,
                           self_attn_mask=tgt_mask,
                           cross_attn_mask=memory_mask)

            # Iterative box refinement
            if self.bbox_embed is not None:
                ref_logit = _inv_sigmoid(ref_sig)
                delta = self.bbox_embed[lid](output)
                new_ref = (delta + ref_logit).sigmoid()
                ref_sig = new_ref.detach()
                if self.use_detached_boxes:
                    ref_pts_list.append(ref_sig)
                else:
                    ref_pts_list.append(new_ref)

            intermediate.append(self.norm(output))

        return [
            [item.transpose(0, 1) for item in intermediate],
            [item.transpose(0, 1) for item in ref_pts_list],
        ]


class DeformableTransformer(nn.Module):
    """Full DINO transformer: encoder + 4D decoder with two-stage support."""

    def __init__(self,
                 d_model=256,
                 nhead=8,
                 num_queries=300,
                 num_enc_layers=6,
                 num_dec_layers=6,
                 dim_feedforward=2048,
                 dropout=0.0,
                 activation="relu",
                 normalize_before=False,
                 return_intermediate_dec=True,
                 query_dim=4,
                 num_feature_levels=1,
                 enc_n_points=4,
                 dec_n_points=4,
                 two_stage_type='no',
                 two_stage_learn_wh=False,
                 two_stage_keep_all_tokens=False,
                 two_stage_pat_embed=0,
                 two_stage_add_query_num=0,
                 random_refpoints_xy=False,
                 dec_layer_number=None,
                 decoder_sa_type='sa',
                 module_seq=None,
                 embed_init_tgt=False,
                 use_detached_boxes=False):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.num_queries = num_queries
        self.d_model = d_model

        enc_layer = DefEncoderBlock(d_model, dim_feedforward, dropout,
                                    activation, num_feature_levels, nhead,
                                    enc_n_points)
        enc_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = MSEncoder(enc_layer,
                                 num_enc_layers,
                                 enc_norm,
                                 d_model=d_model,
                                 deformable=True)

        dec_layer = DefDecoderBlock(d_model,
                                    dim_feedforward,
                                    dropout,
                                    activation,
                                    num_feature_levels,
                                    nhead,
                                    dec_n_points,
                                    module_seq=module_seq)
        dec_norm = nn.LayerNorm(d_model)
        self.decoder = MSDecoder(dec_layer,
                                 num_dec_layers,
                                 dec_norm,
                                 return_intermediate=return_intermediate_dec,
                                 d_model=d_model,
                                 query_dim=query_dim,
                                 n_feature_levels=num_feature_levels,
                                 deformable=True,
                                 use_detached_boxes=use_detached_boxes)

        self.nhead = nhead
        self.dec_layers = num_dec_layers

        if num_feature_levels > 1:
            self.level_embed = nn.Parameter(
                torch.Tensor(num_feature_levels,
                             d_model)) if num_enc_layers > 0 else None
        else:
            self.level_embed = None

        self.embed_init_tgt = embed_init_tgt
        if (two_stage_type != 'no' and
                embed_init_tgt) or two_stage_type == 'no':
            self.tgt_embed = nn.Embedding(num_queries, d_model)
            nn.init.normal_(self.tgt_embed.weight.data)
        else:
            self.tgt_embed = None

        self.two_stage_type = two_stage_type
        self.two_stage_add_query_num = two_stage_add_query_num
        self.two_stage_learn_wh = two_stage_learn_wh

        if two_stage_type == 'standard':
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            if two_stage_learn_wh:
                self.two_stage_wh_embedding = nn.Embedding(1, 2)
            else:
                self.two_stage_wh_embedding = None
            if two_stage_add_query_num > 0:
                self.tgt_embed = nn.Embedding(two_stage_add_query_num, d_model)

        if two_stage_type == 'no':
            self._init_ref_points(num_queries, random_refpoints_xy)

        self.enc_out_class_embed = None
        self.enc_out_bbox_embed = None
        self.dec_layer_number = dec_layer_number
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if self.level_embed is not None:
            nn.init.normal_(self.level_embed)
        if self.two_stage_learn_wh:
            nn.init.constant_(self.two_stage_wh_embedding.weight,
                              math.log(0.05 / (1 - 0.05)))

    def _init_ref_points(self, n, random_xy=False):
        self.refpoint_embed = nn.Embedding(n, 4)
        if random_xy:
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = _inv_sigmoid(
                self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

    def _valid_ratio(self, mask):
        _, H, W = mask.shape
        vh = torch.sum(~mask[:, :, 0], 1).float() / H
        vw = torch.sum(~mask[:, 0, :], 1).float() / W
        return torch.stack([vw, vh], -1)

    def forward(self,
                srcs,
                masks,
                refpoint_embed,
                pos_embeds,
                tgt,
                attn_mask=None):
        # Flatten multi-scale features
        src_flat, mask_flat, lvl_pos_flat = [], [], []
        spatial_shapes = []
        for lvl, (s, m, p) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = s.shape
            spatial_shapes.append((h, w))
            s = s.flatten(2).transpose(1, 2)
            m = m.flatten(1)
            p = p.flatten(2).transpose(1, 2)
            if self.level_embed is not None:
                p = p + self.level_embed[lvl].view(1, 1, -1)
            src_flat.append(s)
            mask_flat.append(m)
            lvl_pos_flat.append(p)

        src_flat = torch.cat(src_flat, 1)
        mask_flat = torch.cat(mask_flat, 1)
        lvl_pos_flat = torch.cat(lvl_pos_flat, 1)
        sp_shapes = torch.as_tensor(spatial_shapes,
                                    dtype=torch.long,
                                    device=src_flat.device)
        lvl_start = torch.cat(
            [sp_shapes.new_zeros((1,)),
             sp_shapes.prod(1).cumsum(0)[:-1]])
        valid_ratios = torch.stack([self._valid_ratio(m) for m in masks], 1)

        # MSEncoder
        memory = self.encoder(src_flat,
                              pos=lvl_pos_flat,
                              spatial_shapes=spatial_shapes,
                              level_start_index=lvl_start,
                              valid_ratios=valid_ratios,
                              pad_mask=mask_flat)

        # Two-stage proposal selection
        if self.two_stage_type == 'standard':
            wh_input = self.two_stage_wh_embedding.weight[0] \
                if self.two_stage_learn_wh else None
            out_mem, out_props = _generate_grid_proposals(
                memory, mask_flat, spatial_shapes, wh_input)
            out_mem = self.enc_output_norm(self.enc_output(out_mem))

            enc_cls = self.enc_out_class_embed(out_mem)
            enc_box = self.enc_out_bbox_embed(out_mem) + out_props

            topk = self.num_queries
            topk_idx = torch.topk(enc_cls.max(-1)[0], topk, dim=1)[1]

            refpoint_undetach = torch.gather(
                enc_box, 1,
                topk_idx.unsqueeze(-1).repeat(1, 1, 4))
            refpoint_ = refpoint_undetach.detach()
            init_box = torch.gather(out_props, 1,
                                    topk_idx.unsqueeze(-1).repeat(1, 1,
                                                                  4)).sigmoid()

            tgt_undetach = torch.gather(
                out_mem, 1,
                topk_idx.unsqueeze(-1).repeat(1, 1, self.d_model))

            if self.embed_init_tgt:
                tgt_ = self.tgt_embed.weight[:,
                                             None, :].repeat(1, bs,
                                                             1).transpose(0, 1)
            else:
                tgt_ = tgt_undetach.detach()

            if refpoint_embed is not None:
                refpoint_embed = torch.cat([refpoint_embed, refpoint_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                refpoint_embed, tgt = refpoint_, tgt_

        elif self.two_stage_type == 'no':
            tgt_ = self.tgt_embed.weight[:, None, :].repeat(1, bs,
                                                            1).transpose(0, 1)
            refpoint_ = self.refpoint_embed.weight[:, None, :].repeat(
                1, bs, 1).transpose(0, 1)
            if refpoint_embed is not None:
                refpoint_embed = torch.cat([refpoint_embed, refpoint_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                refpoint_embed, tgt = refpoint_, tgt_
            init_box = refpoint_.sigmoid()
        else:
            raise NotImplementedError(f"two_stage_type={self.two_stage_type}")

        # MSDecoder
        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=memory.transpose(0, 1),
            memory_key_padding_mask=mask_flat,
            pos=lvl_pos_flat.transpose(0, 1),
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=lvl_start,
            spatial_shapes=sp_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=attn_mask)

        if self.two_stage_type == 'standard':
            hs_enc = tgt_undetach.unsqueeze(0)
            ref_enc = refpoint_undetach.sigmoid().unsqueeze(0)
        else:
            hs_enc = ref_enc = None

        return hs, references, hs_enc, ref_enc, init_box


# ===========================================================================
# Denoising (CDN) Components
# ===========================================================================


def prepare_for_cdn(dn_args, training, num_queries, num_classes, hidden_dim,
                    label_enc):
    """Contrastive De-Noising query preparation for DINO training."""
    if training:
        targets, dn_number, label_noise_ratio, box_noise_scale = dn_args
        dn_number = dn_number * 2
        known = [torch.ones_like(t['labels']).cuda() for t in targets]
        batch_size = len(known)
        known_num = [int(k.sum()) for k in known]

        if int(max(known_num)) == 0:
            dn_number = 1
        else:
            if dn_number >= 100:
                dn_number = dn_number // (int(max(known_num) * 2))
            elif dn_number < 1:
                dn_number = 1
        if dn_number == 0:
            dn_number = 1

        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t['labels'] for t in targets])
        boxes = torch.cat([t['boxes'] for t in targets])
        batch_idx = torch.cat([
            torch.full_like(t['labels'].long(), i)
            for i, t in enumerate(targets)
        ])

        known_idx = torch.nonzero(unmask_label + unmask_bbox).view(-1)
        known_idx = known_idx.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        known_labels_exp = known_labels.clone()
        known_bbox_exp = known_bboxs.clone()

        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_exp.float())
            flip = torch.nonzero(p < label_noise_ratio * 0.5).view(-1)
            new_label = torch.randint_like(flip, 0, num_classes)
            known_labels_exp.scatter_(0, flip, new_label)

        single_pad = int(max(known_num))
        pad_size = int(single_pad * 2 * dn_number)

        pos_idx = torch.tensor(range(
            len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number, 1)
        pos_idx += (torch.tensor(range(dn_number)) * len(boxes) *
                    2).long().cuda().unsqueeze(1)
        pos_idx = pos_idx.flatten()
        neg_idx = pos_idx + len(boxes)

        if box_noise_scale > 0:
            known_bb = torch.zeros_like(known_bboxs)
            known_bb[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            known_bb[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2
            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            diff[:, 2:] = known_bboxs[:, 2:] / 2
            sign = torch.randint_like(known_bboxs, 0, 2,
                                      dtype=torch.float32) * 2.0 - 1.0
            rand = torch.rand_like(known_bboxs)
            rand[neg_idx] += 1.0
            rand *= sign
            known_bb = known_bb + \
                torch.mul(rand, diff).cuda() * box_noise_scale
            known_bb = known_bb.clamp(0.0, 1.0)
            known_bbox_exp[:, :2] = (known_bb[:, :2] + known_bb[:, 2:]) / 2
            known_bbox_exp[:, 2:] = known_bb[:, 2:] - known_bb[:, :2]

        m = known_labels_exp.long().to('cuda')
        input_label_embed = label_enc(m)
        input_bbox_embed = _inv_sigmoid(known_bbox_exp)

        pad_label = torch.zeros(pad_size, hidden_dim).cuda()
        pad_bbox = torch.zeros(pad_size, 4).cuda()
        input_query_label = pad_label.repeat(batch_size, 1, 1)
        input_query_bbox = pad_bbox.repeat(batch_size, 1, 1)

        map_idx = torch.tensor([]).to('cuda')
        if len(known_num):
            map_idx = torch.cat([torch.tensor(range(num)) for num in known_num])
            map_idx = torch.cat([
                map_idx + single_pad * i for i in range(2 * dn_number)
            ]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_idx)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_idx)] = input_bbox_embed

        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        attn_mask[pad_size:, :pad_size] = True
        for i in range(dn_number):
            if i == 0:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1),
                          single_pad * 2 * (i + 1):pad_size] = True
            if i == dn_number - 1:
                attn_mask[single_pad * 2 * i:single_pad * 2 *
                          (i + 1), :single_pad * i * 2] = True
            else:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1),
                          single_pad * 2 * (i + 1):pad_size] = True
                attn_mask[single_pad * 2 * i:single_pad * 2 *
                          (i + 1), :single_pad * 2 * i] = True

        dn_meta = {'pad_size': pad_size, 'num_dn_group': dn_number}
    else:
        input_query_label = input_query_bbox = attn_mask = dn_meta = None

    return input_query_label, input_query_bbox, attn_mask, dn_meta


def dn_post_process(outputs_class, outputs_coord, dn_meta, aux_loss,
                    _set_aux_loss):
    """Split denoising outputs from detection outputs after decoder."""
    if dn_meta and dn_meta['pad_size'] > 0:
        output_known_cls = outputs_class[:, :, :dn_meta['pad_size'], :]
        output_known_box = outputs_coord[:, :, :dn_meta['pad_size'], :]
        outputs_class = outputs_class[:, :, dn_meta['pad_size']:, :]
        outputs_coord = outputs_coord[:, :, dn_meta['pad_size']:, :]
        out = {
            'pred_logits': output_known_cls[-1],
            'pred_boxes': output_known_box[-1]
        }
        if aux_loss:
            out['aux_outputs'] = _set_aux_loss(output_known_cls,
                                               output_known_box)
        dn_meta['output_known_lbs_bboxes'] = out
    return outputs_class, outputs_coord


# ===========================================================================
# DINO Model
# ===========================================================================


class DINO(nn.Module):
    """DINO: DETR with Improved DeNoising Anchor Boxes."""

    def __init__(self,
                 backbone,
                 transformer,
                 num_classes,
                 num_queries,
                 aux_loss=False,
                 num_feature_levels=1,
                 nheads=8,
                 two_stage_type='no',
                 dec_pred_class_embed_share=True,
                 dec_pred_bbox_embed_share=True,
                 two_stage_class_embed_share=True,
                 two_stage_bbox_embed_share=True,
                 dn_number=100,
                 dn_box_noise_scale=0.4,
                 dn_label_noise_ratio=0.5,
                 dn_labelbook_size=100):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.label_enc = nn.Embedding(dn_labelbook_size + 1, hidden_dim)

        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        # Input projections (multi-scale)
        if num_feature_levels > 1:
            n_backbone = len(backbone.num_channels)
            projs = []
            for i in range(n_backbone):
                projs.append(
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[i], hidden_dim, 1),
                        nn.GroupNorm(32, hidden_dim)))
            in_ch = backbone.num_channels[-1]
            for _ in range(num_feature_levels - n_backbone):
                projs.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, hidden_dim, 3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim)))
                in_ch = hidden_dim
            self.input_proj = nn.ModuleList(projs)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-1], hidden_dim, 1),
                    nn.GroupNorm(32, hidden_dim))
            ])

        self.backbone = backbone
        self.aux_loss = aux_loss

        # Prediction heads
        _cls = nn.Linear(hidden_dim, num_classes)
        _box = MLP(hidden_dim, hidden_dim, 4, 3)
        prior = 0.01
        _cls.bias.data = torch.ones(num_classes) * (-math.log(
            (1 - prior) / prior))
        nn.init.constant_(_box.layers[-1].weight.data, 0)
        nn.init.constant_(_box.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_layers = [_box] * transformer.num_dec_layers
        else:
            box_layers = [
                copy.deepcopy(_box) for _ in range(transformer.num_dec_layers)
            ]
        if dec_pred_class_embed_share:
            cls_layers = [_cls] * transformer.num_dec_layers
        else:
            cls_layers = [
                copy.deepcopy(_cls) for _ in range(transformer.num_dec_layers)
            ]

        self.bbox_embed = nn.ModuleList(box_layers)
        self.class_embed = nn.ModuleList(cls_layers)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        self.two_stage_type = two_stage_type
        if two_stage_type != 'no':
            if two_stage_bbox_embed_share:
                self.transformer.enc_out_bbox_embed = _box
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_box)
            if two_stage_class_embed_share:
                self.transformer.enc_out_class_embed = _cls
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_cls)

        self._reset_parameters()

    def _reset_parameters(self):
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def forward(self, samples: ImageBatch, targets: List = None):
        if isinstance(samples, (list, torch.Tensor)):
            samples = make_image_batch(samples)
        features, poss = self.backbone(samples)

        srcs, masks = [], []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)

        if self.num_feature_levels > len(srcs):
            n_src = len(srcs)
            for l in range(n_src, self.num_feature_levels):
                if l == n_src:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(),
                                     size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](ImageBatch(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        # Denoising queries
        if self.dn_number > 0 or targets is not None:
            input_query_label, input_query_bbox, attn_mask, dn_meta = \
                prepare_for_cdn(
                    dn_args=(targets, self.dn_number, self.dn_label_noise_ratio,
                             self.dn_box_noise_scale),
                    training=self.training, num_queries=self.num_queries,
                    num_classes=self.num_classes, hidden_dim=self.hidden_dim,
                    label_enc=self.label_enc)
        else:
            input_query_bbox = input_query_label = attn_mask = dn_meta = None

        hs, reference, hs_enc, ref_enc, init_box = \
            self.transformer(srcs, masks, input_query_bbox, poss,
                             input_query_label, attn_mask)

        # Ensure label_enc gradient flows
        hs[0] += self.label_enc.weight[0, 0] * 0.0

        # Box predictions with iterative refinement
        outputs_coord_list = []
        for lid, (layer_ref, layer_box, layer_hs) in enumerate(
                zip(reference[:-1], self.bbox_embed, hs)):
            delta = layer_box(layer_hs)
            coord = (delta + _inv_sigmoid(layer_ref)).sigmoid()
            outputs_coord_list.append(coord)
        outputs_coord_list = torch.stack(outputs_coord_list)

        outputs_class = torch.stack([
            cls_head(layer_hs)
            for cls_head, layer_hs in zip(self.class_embed, hs)
        ])

        # DN post-processing
        if self.dn_number > 0 and dn_meta is not None:
            outputs_class, outputs_coord_list = dn_post_process(
                outputs_class, outputs_coord_list, dn_meta, self.aux_loss,
                self._set_aux_loss)

        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord_list[-1]
        }

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class,
                                                    outputs_coord_list)

        # Two-stage intermediate outputs
        if hs_enc is not None:
            interm_coord = ref_enc[-1]
            interm_cls = self.transformer.enc_out_class_embed(hs_enc[-1])
            out['interm_outputs'] = {
                'pred_logits': interm_cls,
                'pred_boxes': interm_coord
            }
            out['interm_outputs_for_matching_pre'] = {
                'pred_logits': interm_cls,
                'pred_boxes': init_box
            }

        out['dn_meta'] = dn_meta
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{
            'pred_logits': a,
            'pred_boxes': b
        } for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


# ===========================================================================
# Loss Components
# ===========================================================================


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha=0.25, gamma=2):
    """Sigmoid focal loss for classification."""
    prob = inputs.sigmoid()
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t)**gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_boxes


class BipartiteMatcher(nn.Module):
    """Optimal bipartite matching with focal-loss class cost."""

    def __init__(self,
                 cost_class=1,
                 cost_bbox=1,
                 cost_giou=1,
                 focal_alpha=0.25):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_alpha = focal_alpha

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, nq = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        alpha, gamma = self.focal_alpha, 2.0
        neg = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
        pos = alpha * ((1 - out_prob)**gamma) * (-(out_prob + 1e-8).log())
        cost_cls = pos[:, tgt_ids] - neg[:, tgt_ids]
        cost_bb = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -_generalized_iou(_cxcywh_to_xyxy(out_bbox),
                                      _cxcywh_to_xyxy(tgt_bbox))

        C = (self.cost_bbox * cost_bb + self.cost_class * cost_cls +
             self.cost_giou * cost_giou).view(bs, nq, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(C.split(sizes, -1))
        ]
        return [(torch.as_tensor(i, dtype=torch.int64),
                 torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class DINOCriterion(nn.Module):
    """DINO loss: labels + boxes + cardinality + DN + interm."""

    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2],
                                    self.num_classes,
                                    dtype=torch.int64,
                                    device=src_logits.device)
        target_classes[idx] = target_classes_o

        onehot = torch.zeros([*src_logits.shape[:2], src_logits.shape[2] + 1],
                             dtype=src_logits.dtype,
                             device=src_logits.device)
        onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        onehot = onehot[:, :, :-1]

        loss = sigmoid_focal_loss(
            src_logits, onehot, num_boxes, alpha=self.focal_alpha,
            gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss}
        if log:
            losses['class_error'] = 100 - _topk_accuracy(
                src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs['pred_logits']
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets],
                                      device=pred_logits.device)
        card = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        return {
            'cardinality_error': F.l1_loss(card.float(), tgt_lengths.float())
        }

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat(
            [t['boxes'][i] for t, (_, i) in zip(targets, indices)])
        l1 = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {'loss_bbox': l1.sum() / num_boxes}
        giou = 1 - torch.diag(
            _generalized_iou(_cxcywh_to_xyxy(src_boxes),
                             _cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat(
            [torch.full_like(s, i) for i, (s, _) in enumerate(indices)])
        src_idx = torch.cat([s for (s, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kw):
        fn_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes
        }
        return fn_map[loss](outputs, targets, indices, num_boxes, **kw)

    def forward(self, outputs, targets, return_indices=False):
        outputs_wo_aux = {
            k: v
            for k, v in outputs.items()
            if k != 'aux_outputs' and k != 'interm_outputs' and
            k != 'interm_outputs_for_matching_pre' and k != 'dn_meta'
        }
        device = next(iter(outputs.values())).device
        indices = self.matcher(outputs_wo_aux, targets)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes],
                                    dtype=torch.float,
                                    device=device)
        if _is_distributed():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / _world_size(), min=1).item()

        losses = {}

        # DN losses
        dn_meta = outputs.get('dn_meta', None)
        if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
            output_known, single_pad, scalar = self._prep_dn(dn_meta)
            dn_pos_idx, dn_neg_idx = [], []
            for i in range(len(targets)):
                if len(targets[i]['labels']) > 0:
                    t = torch.arange(0, len(targets[i]['labels'])).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    out_idx = ((torch.tensor(range(scalar)) *
                                single_pad).long().cuda().unsqueeze(1) +
                               t).flatten()
                else:
                    out_idx = tgt_idx = torch.tensor([]).long().cuda()
                dn_pos_idx.append((out_idx, tgt_idx))
                dn_neg_idx.append((out_idx + single_pad // 2, tgt_idx))

            for loss in self.losses:
                kw = {'log': False} if 'labels' in loss else {}
                l_dict = self.get_loss(loss, output_known, targets, dn_pos_idx,
                                       num_boxes * scalar, **kw)
                losses.update({k + '_dn': v for k, v in l_dict.items()})
        else:
            losses.update({
                'loss_bbox_dn': torch.tensor(0., device=device),
                'loss_giou_dn': torch.tensor(0., device=device),
                'loss_ce_dn': torch.tensor(0., device=device),
                'cardinality_error_dn': torch.tensor(0., device=device),
            })

        # Main losses
        for loss in self.losses:
            losses.update(
                self.get_loss(loss, outputs, targets, indices, num_boxes))

        # Auxiliary decoder layer losses
        if 'aux_outputs' in outputs:
            for i, aux in enumerate(outputs['aux_outputs']):
                aux_idx = self.matcher(aux, targets)
                for loss in self.losses:
                    kw = {'log': False} if loss == 'labels' else {}
                    l = self.get_loss(loss, aux, targets, aux_idx, num_boxes,
                                      **kw)
                    losses.update({k + f'_{i}': v for k, v in l.items()})

                if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    aux_known = output_known['aux_outputs'][i]
                    for loss in self.losses:
                        kw = {'log': False} if 'labels' in loss else {}
                        l = self.get_loss(loss, aux_known, targets, dn_pos_idx,
                                          num_boxes * scalar, **kw)
                        losses.update({k + f'_dn_{i}': v for k, v in l.items()})
                else:
                    losses.update({
                        f'loss_bbox_dn_{i}':
                            torch.tensor(0., device=device),
                        f'loss_giou_dn_{i}':
                            torch.tensor(0., device=device),
                        f'loss_ce_dn_{i}':
                            torch.tensor(0., device=device),
                        f'cardinality_error_dn_{i}':
                            torch.tensor(0., device=device),
                    })

        # Intermediate encoder output loss (two-stage)
        if 'interm_outputs' in outputs:
            interm = outputs['interm_outputs']
            interm_idx = self.matcher(interm, targets)
            for loss in self.losses:
                kw = {'log': False} if loss == 'labels' else {}
                l = self.get_loss(loss, interm, targets, interm_idx, num_boxes,
                                  **kw)
                losses.update({k + '_interm': v for k, v in l.items()})

        return losses

    def _prep_dn(self, dn_meta):
        output_known = dn_meta['output_known_lbs_bboxes']
        n_groups, pad_size = dn_meta['num_dn_group'], dn_meta['pad_size']
        assert pad_size % n_groups == 0
        return output_known, pad_size // n_groups, n_groups


# ===========================================================================
# DetPostProcessor
# ===========================================================================


class DetPostProcessor(nn.Module):
    """Sigmoid-based top-k post-processing."""

    def __init__(self, num_select=300, nms_iou_threshold=-1):
        super().__init__()
        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold

    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False, test=False):
        logits = outputs['pred_logits']
        boxes = outputs['pred_boxes']
        assert len(logits) == len(target_sizes)

        prob = logits.sigmoid()
        topk_vals, topk_idxs = torch.topk(prob.view(logits.shape[0], -1),
                                          self.num_select,
                                          dim=1)

        scores = topk_vals
        topk_boxes = topk_idxs // logits.shape[2]
        labels = topk_idxs % logits.shape[2]

        if not_to_xyxy:
            out_boxes = boxes
        else:
            out_boxes = _cxcywh_to_xyxy(boxes)

        if test:
            out_boxes[:, :, 2:] = out_boxes[:, :, 2:] - out_boxes[:, :, :2]

        out_boxes = torch.gather(out_boxes, 1,
                                 topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        img_h, img_w = target_sizes.unbind(1)
        scale = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        out_boxes = out_boxes * scale[:, None, :]

        if self.nms_iou_threshold > 0:
            keep = [
                nms(b, s, iou_threshold=self.nms_iou_threshold)
                for b, s in zip(out_boxes, scores)
            ]
            results = [{
                'scores': s[k],
                'labels': l[k],
                'boxes': b[k]
            } for s, l, b, k in zip(scores, labels, out_boxes, keep)]
        else:
            results = [{
                'scores': s,
                'labels': l,
                'boxes': b
            } for s, l, b in zip(scores, labels, out_boxes)]
        return results


# ===========================================================================
# Build Functions
# ===========================================================================


def build_backbone_and_pos(config):
    """Construct backbone + positional encoding from config dict."""
    n_steps = config['hidden_dim'] // 2
    pos_enc = SinePositionEmbed2D(n_steps,
                                  tempH=config.get('pe_temperatureH', 20),
                                  tempW=config.get('pe_temperatureW', 20),
                                  normalize=True)

    train_bb = config['lr_backbone'] > 0
    ret_idx = config.get('return_interm_indices', [1, 2, 3])
    bb_net = Backbone(config.get('backbone', 'resnet50'),
                      train_bb,
                      config.get('dilation', False),
                      ret_idx,
                      norm_layer=FrozenBatchNorm2d)
    model = Joiner(bb_net, pos_enc)
    model.num_channels = bb_net.num_channels
    return model


def build_model(config):
    """Build complete DINO model + criterion + postprocessors from config."""
    num_classes = config['num_classes']
    device = torch.device(config.get('device', 'cuda'))

    backbone = build_backbone_and_pos(config)

    transformer = DeformableTransformer(
        d_model=config['hidden_dim'],
        nhead=config['nheads'],
        num_queries=config['num_queries'],
        num_enc_layers=config['enc_layers'],
        num_dec_layers=config['dec_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config.get('dropout', 0.0),
        activation=config.get('transformer_activation', 'relu'),
        normalize_before=config.get('pre_norm', False),
        return_intermediate_dec=True,
        query_dim=4,
        num_feature_levels=config.get('num_feature_levels', 4),
        enc_n_points=config.get('enc_n_points', 4),
        dec_n_points=config.get('dec_n_points', 4),
        two_stage_type=config.get('two_stage_type', 'standard'),
        two_stage_learn_wh=config.get('two_stage_learn_wh', False),
        two_stage_keep_all_tokens=config.get('two_stage_keep_all_tokens',
                                             False),
        two_stage_pat_embed=config.get('two_stage_pat_embed', 0),
        two_stage_add_query_num=config.get('two_stage_add_query_num', 0),
        random_refpoints_xy=config.get('random_refpoints_xy', False),
        dec_layer_number=config.get('dec_layer_number', None),
        decoder_sa_type=config.get('decoder_sa_type', 'sa'),
        module_seq=config.get('decoder_module_seq', ['sa', 'ca', 'ffn']),
        embed_init_tgt=config.get('embed_init_tgt', True),
        use_detached_boxes=config.get('use_detached_boxes_dec_out', False),
    )

    dn_labelbook_size = config.get('dn_labelbook_size', num_classes)

    model = DINO(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=config['num_queries'],
        aux_loss=config.get('aux_loss', True),
        num_feature_levels=config.get('num_feature_levels', 4),
        nheads=config['nheads'],
        two_stage_type=config.get('two_stage_type', 'standard'),
        dec_pred_class_embed_share=config.get('dec_pred_class_embed_share',
                                              True),
        dec_pred_bbox_embed_share=config.get('dec_pred_bbox_embed_share', True),
        two_stage_class_embed_share=config.get('two_stage_class_embed_share',
                                               False),
        two_stage_bbox_embed_share=config.get('two_stage_bbox_embed_share',
                                              False),
        dn_number=config.get('dn_number', 100)
        if config.get('use_dn', True) else 0,
        dn_box_noise_scale=config.get('dn_box_noise_scale', 0.4),
        dn_label_noise_ratio=config.get('dn_label_noise_ratio', 0.5),
        dn_labelbook_size=dn_labelbook_size,
    )

    matcher = BipartiteMatcher(
        cost_class=config.get('set_cost_class', 2.0),
        cost_bbox=config.get('set_cost_bbox', 5.0),
        cost_giou=config.get('set_cost_giou', 2.0),
        focal_alpha=config.get('focal_alpha', 0.25),
    )

    weight_dict = {
        'loss_ce': config.get('cls_loss_coef', 1.0),
        'loss_bbox': config.get('bbox_loss_coef', 5.0),
        'loss_giou': config.get('giou_loss_coef', 2.0),
    }
    clean_wo_dn = copy.deepcopy(weight_dict)

    if config.get('use_dn', True):
        weight_dict['loss_ce_dn'] = config.get('cls_loss_coef', 1.0)
        weight_dict['loss_bbox_dn'] = config.get('bbox_loss_coef', 5.0)
        weight_dict['loss_giou_dn'] = config.get('giou_loss_coef', 2.0)
    clean_wd = copy.deepcopy(weight_dict)

    if config.get('aux_loss', True):
        for i in range(config['dec_layers'] - 1):
            weight_dict.update({k + f'_{i}': v for k, v in clean_wd.items()})

    if config.get('two_stage_type', 'standard') != 'no':
        no_interm_box = config.get('no_interm_box_loss', False)
        coeff = {
            'loss_ce': 1.0,
            'loss_bbox': 0.0 if no_interm_box else 1.0,
            'loss_giou': 0.0 if no_interm_box else 1.0
        }
        interm_coef = config.get('interm_loss_coef', 1.0)
        weight_dict.update({
            k + '_interm': v * interm_coef * coeff[k]
            for k, v in clean_wo_dn.items()
        })

    criterion = DINOCriterion(num_classes,
                              matcher=matcher,
                              weight_dict=weight_dict,
                              focal_alpha=config.get('focal_alpha', 0.25),
                              losses=['labels', 'boxes', 'cardinality'])
    criterion.to(device)

    postprocessors = {
        'bbox':
            DetPostProcessor(num_select=config.get('num_select', 300),
                             nms_iou_threshold=config.get(
                                 'nms_iou_threshold', -1))
    }

    return model, criterion, postprocessors
