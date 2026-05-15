"""PromptIR-CPLFreq model for all-in-one rain/snow restoration.

This implementation keeps the PromptIR idea: learn degradation-aware visual prompts
and inject them into a restoration backbone. It adds sparse prompt routing and
frequency-aware feature enhancement for the HW4 rain/snow restoration task.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class BiasFreeLayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for 4D image tensors."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        x_perm = x.permute(0, 2, 3, 1)
        var = x_perm.var(dim=-1, keepdim=True, unbiased=False)
        x_perm = x_perm / torch.sqrt(var + self.eps)
        x_perm = x_perm * self.weight
        return x_perm.permute(0, 3, 1, 2)


class WithBiasLayerNorm2d(nn.Module):
    """Channel-wise LayerNorm with bias for 4D image tensors."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        x_perm = x.permute(0, 2, 3, 1)
        mean = x_perm.mean(dim=-1, keepdim=True)
        var = x_perm.var(dim=-1, keepdim=True, unbiased=False)
        x_perm = (x_perm - mean) / torch.sqrt(var + self.eps)
        x_perm = x_perm * self.weight + self.bias
        return x_perm.permute(0, 3, 1, 2)


class FeedForward(nn.Module):
    """Gated depthwise feed-forward network used in restoration transformers."""

    def __init__(self, channels: int, expansion: float = 2.66, bias: bool = False) -> None:
        super().__init__()
        hidden = int(channels * expansion)
        self.project_in = nn.Conv2d(channels, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            kernel_size=3,
            padding=1,
            groups=hidden * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden, channels, kernel_size=1, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class MultiDconvHeadTransposedAttention(nn.Module):
    """MDTA attention from Restormer-style image restoration transformers."""

    def __init__(self, channels: int, num_heads: int, bias: bool = False) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by heads={num_heads}")
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            channels * 3,
            channels * 3,
            kernel_size=3,
            padding=1,
            groups=channels * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        head_dim = channels // self.num_heads

        q = q.reshape(batch, self.num_heads, head_dim, height * width)
        k = k.reshape(batch, self.num_heads, head_dim, height * width)
        v = v.reshape(batch, self.num_heads, head_dim, height * width)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = out.reshape(batch, channels, height, width)
        return self.project_out(out)


class TransformerBlock(nn.Module):
    """Restoration transformer block."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        ffn_expansion: float = 2.66,
        bias: bool = False,
        layer_norm_type: str = "with_bias",
    ) -> None:
        super().__init__()
        norm = WithBiasLayerNorm2d if layer_norm_type == "with_bias" else BiasFreeLayerNorm2d
        self.norm1 = norm(channels)
        self.attn = MultiDconvHeadTransposedAttention(channels, num_heads, bias=bias)
        self.norm2 = norm(channels)
        self.ffn = FeedForward(channels, expansion=ffn_expansion, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Downsample(nn.Module):
    """Downsample by 2 and double channels."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class Upsample(nn.Module):
    """Upsample by 2 and halve channels."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class FrequencyEnhancementBlock(nn.Module):
    """Frequency-aware modulation block inspired by recent AiOIR frequency methods.

    The block decouples feature maps into adaptive low/high-frequency components in
    the Fourier domain, fuses them with depthwise convolutions, and injects the
    result back through a learnable residual gate.
    """

    def __init__(self, channels: int, cutoff: float = 0.22, sharpness: float = 24.0) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.sharpness = sharpness
        self.low_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
        )
        self.high_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, max(channels // 4, 8), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(max(channels // 4, 8), channels * 2, kernel_size=1),
            nn.Sigmoid(),
        )
        self.fuse = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def _low_high(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        original_dtype = x.dtype
        x_float = x.float()
        _, _, height, width = x_float.shape

        fy = torch.fft.fftfreq(height, device=x.device).view(height, 1)
        fx = torch.fft.rfftfreq(width, device=x.device).view(1, width // 2 + 1)
        radius = torch.sqrt(fy.square() + fx.square())
        low_mask = torch.sigmoid((self.cutoff - radius) * self.sharpness)
        low_mask = low_mask.view(1, 1, height, width // 2 + 1)

        spectrum = torch.fft.rfft2(x_float, norm="ortho")
        low = torch.fft.irfft2(spectrum * low_mask, s=(height, width), norm="ortho")
        high = x_float - low
        return low.to(original_dtype), high.to(original_dtype)

    def forward(self, x: Tensor) -> Tensor:
        low, high = self._low_high(x)
        low = self.low_proj(low)
        high = self.high_proj(high)
        fused = torch.cat([low, high], dim=1)
        gated = fused * self.gate(fused)
        return x + self.gamma * self.fuse(gated)


class SparsePromptBlock(nn.Module):
    """Input-adaptive sparse visual prompt injection block."""

    def __init__(
        self,
        channels: int,
        prompt_size: int = 16,
        num_prompts: int = 6,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        if top_k > num_prompts:
            raise ValueError("top_k cannot be larger than num_prompts")
        self.channels = channels
        self.num_prompts = num_prompts
        self.top_k = top_k
        self.prompt_bank = nn.Parameter(
            torch.randn(num_prompts, channels, prompt_size, prompt_size) * 0.02
        )
        hidden = max(channels // 4, 16)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, num_prompts, kernel_size=1),
        )
        self.task_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        self.prompt_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.alpha = nn.Parameter(torch.zeros(1))

    def _sparse_weights(self, logits: Tensor) -> Tensor:
        if self.top_k == self.num_prompts:
            return logits.softmax(dim=-1)
        values, indices = logits.topk(self.top_k, dim=-1)
        sparse_logits = torch.full_like(logits, torch.finfo(logits.dtype).min)
        sparse_logits.scatter_(dim=-1, index=indices, src=values)
        return sparse_logits.softmax(dim=-1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        batch, _, height, width = x.shape
        logits = self.router(x).flatten(1)
        weights = self._sparse_weights(logits)
        prompt = torch.einsum("bn,nchw->bchw", weights, self.prompt_bank)
        prompt = F.interpolate(prompt, size=(height, width), mode="bilinear", align_corners=False)
        prompt = self.prompt_refine(prompt)
        fused = self.fuse(torch.cat([x, prompt], dim=1))
        out = x + self.alpha * fused
        aux = {
            "router_logits": logits,
            "router_weights": weights,
            "task_logits": self.task_head(x),
            "prompt_bank": self.prompt_bank,
        }
        return out, aux


class PromptIRCPLFreq(nn.Module):
    """PromptIR with sparse contrastive prompts and frequency enhancement."""

    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: Tuple[int, int, int, int] = (4, 6, 6, 8),
        num_refinement_blocks: int = 4,
        heads: Tuple[int, int, int, int] = (1, 2, 4, 8),
        ffn_expansion: float = 2.66,
        bias: bool = False,
        layer_norm_type: str = "with_bias",
        num_prompts: int = 6,
        top_k: int = 2,
        use_frequency: bool = True,
    ) -> None:
        super().__init__()
        self.use_frequency = use_frequency
        self.patch_embed = nn.Conv2d(inp_channels, dim, kernel_size=3, padding=1, bias=bias)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim, heads[0], ffn_expansion, bias, layer_norm_type)
            for _ in range(num_blocks[0])
        ])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[1], ffn_expansion, bias, layer_norm_type)
            for _ in range(num_blocks[1])
        ])
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim * 4, heads[2], ffn_expansion, bias, layer_norm_type)
            for _ in range(num_blocks[2])
        ])
        self.down3_4 = Downsample(dim * 4)
        self.latent = nn.Sequential(*[
            TransformerBlock(dim * 8, heads[3], ffn_expansion, bias, layer_norm_type)
            for _ in range(num_blocks[3])
        ])

        self.freq_latent = FrequencyEnhancementBlock(dim * 8) if use_frequency else nn.Identity()

        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=bias)
        self.prompt3 = SparsePromptBlock(dim * 4, num_prompts=num_prompts, top_k=top_k)
        self.freq3 = FrequencyEnhancementBlock(dim * 4) if use_frequency else nn.Identity()
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim * 4, heads[2], ffn_expansion, bias, layer_norm_type)
            for _ in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias)
        self.prompt2 = SparsePromptBlock(dim * 2, num_prompts=num_prompts, top_k=top_k)
        self.freq2 = FrequencyEnhancementBlock(dim * 2) if use_frequency else nn.Identity()
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[1], ffn_expansion, bias, layer_norm_type)
            for _ in range(num_blocks[1])
        ])

        self.up2_1 = Upsample(dim * 2)
        self.reduce_chan_level1 = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.prompt1 = SparsePromptBlock(dim, num_prompts=num_prompts, top_k=top_k)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim, heads[0], ffn_expansion, bias, layer_norm_type)
            for _ in range(num_blocks[0])
        ])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim, heads[0], ffn_expansion, bias, layer_norm_type)
            for _ in range(num_refinement_blocks)
        ])
        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=bias)

        self.embedding_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim * 8, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, 128),
        )

    def _decode_prompt(
        self,
        x: Tensor,
        prompt_block: SparsePromptBlock,
        freq_block: nn.Module,
        aux_list: List[Dict[str, Tensor]],
    ) -> Tensor:
        x, aux = prompt_block(x)
        x = freq_block(x)
        aux_list.append(aux)
        return x

    def forward(self, inp_img: Tensor, return_aux: bool = False) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        aux_list: List[Dict[str, Tensor]] = []
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)
        embedding = self.embedding_head(latent)
        latent = self.freq_latent(latent)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], dim=1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        inp_dec_level3 = self._decode_prompt(inp_dec_level3, self.prompt3, self.freq3, aux_list)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        inp_dec_level2 = self._decode_prompt(inp_dec_level2, self.prompt2, self.freq2, aux_list)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        inp_dec_level1 = self._decode_prompt(inp_dec_level1, self.prompt1, nn.Identity(), aux_list)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        out = self.output(out_dec_level1) + inp_img

        if not return_aux:
            return out
        aux = {
            "prompt_aux": aux_list,
            "embedding": F.normalize(embedding, dim=1),
        }
        return out, aux

    def prompt_parameters(self) -> List[nn.Parameter]:
        """Return prompt-bank parameters for diversity regularization."""
        return [
            self.prompt1.prompt_bank,
            self.prompt2.prompt_bank,
            self.prompt3.prompt_bank,
        ]


def _get(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def build_model(cfg) -> PromptIRCPLFreq:
    """Factory used by train.py and inference.py.

    A new architecture can be added by creating models/<name>.py with the same
    build_model(cfg) function and setting model.name in a YAML config.
    """
    return PromptIRCPLFreq(
        dim=_get(cfg, "dim", 48),
        num_blocks=tuple(_get(cfg, "num_blocks", [4, 6, 6, 8])),
        num_refinement_blocks=_get(cfg, "num_refinement_blocks", 4),
        heads=tuple(_get(cfg, "heads", [1, 2, 4, 8])),
        num_prompts=_get(cfg, "num_prompts", 6),
        top_k=_get(cfg, "top_k", 2),
        use_frequency=not _get(cfg, "no_frequency", False),
    )
