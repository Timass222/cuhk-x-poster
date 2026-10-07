# -*- coding: utf-8 -*-
"""VideoMAE-style ViT-S/16 (tubelet 2) for the CUHK-X video branches.

Vendored re-implementation of VideoMAE(v2) `modeling_finetune` (no timm /
transformers dependency): PatchEmbed Conv3d(in_ch, 384, (2,16,16)) ->
12 pre-LN blocks (qkv with separate q/v bias, SDPA attention, MLP 4x,
drop-path) -> mean pool -> fc_norm -> head. Position embedding is the fixed
sin-cos table of VideoMAE, generated for the actual token grid (T/2, H/16,
W/16) so any input size works; it is a non-persistent buffer.

Pretrained weights: data/external/videomaev2/vit_s_k710_dl_from_giant.pth
(OpenGVLab VideoMAEv2, ViT-S distilled from ViT-g on K710; 'module' dict,
fp16). The 3-channel patch-embed kernel is extended to `in_channels`: RGB
kernels stay on the depth-palette channels, the extra (IR) channel gets the
channel-mean kernel, and the whole kernel is rescaled by 3/in_channels so the
pre-activation magnitude matches the pretrained regime (there is no BN to
absorb the change, unlike the R(2+1)D stem).

Input contract (same as every other branch): x of shape (N, T, C, H, W) in
[-0.5, 0.5] (VideoDS gives uint8/255 - 0.5). ImageNet normalisation is applied
inside forward: mean/std per channel, IR channel uses the RGB average.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS = Path("data/external/videomaev2/vit_s_k710_dl_from_giant.pth")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def sincos_pos_embed(n_pos: int, dim: int) -> torch.Tensor:
    """VideoMAE's get_sinusoid_encoding_table: (1, n_pos, dim)."""
    pos = np.arange(n_pos)[:, None]
    i = np.arange(dim)[None, :]
    angle = pos / np.power(10000, 2 * (i // 2) / dim)
    table = np.zeros((n_pos, dim), np.float32)
    table[:, 0::2] = np.sin(angle[:, 0::2])
    table[:, 1::2] = np.cos(angle[:, 1::2])
    return torch.from_numpy(table)[None]


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        n, l, d = x.shape
        bias = torch.cat([self.q_bias, torch.zeros_like(self.v_bias), self.v_bias])
        qkv = F.linear(x, self.qkv.weight, bias).reshape(n, l, 3, self.heads, d // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)            # (3, N, heads, L, hd)
        out = F.scaled_dot_product_attention(q, k, v)    # flash / mem-efficient
        return self.proj(out.transpose(1, 2).reshape(n, l, d))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, drop_path: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential()
        self.mlp.fc1 = nn.Linear(dim, hidden)
        self.mlp.act = nn.GELU()
        self.mlp.fc2 = nn.Linear(hidden, dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VideoViT(nn.Module):
    def __init__(self, in_channels: int = 4, num_classes: int = 40, n_frames: int = 16,
                 dim: int = 384, depth: int = 12, heads: int = 6, mlp_ratio: float = 4.0,
                 patch: int = 16, tubelet: int = 2, drop_path: float = 0.1,
                 pretrained: bool = True, use_checkpoint: bool = False,
                 weights=None, norm: str = "imagenet"):
        super().__init__()
        self.in_channels, self.n_frames, self.dim = in_channels, n_frames, dim
        self.weights = Path(weights) if weights is not None else WEIGHTS
        if norm == "clip":                      # UMT / CLIP-teacher checkpoints
            base_mean, base_std = CLIP_MEAN, CLIP_STD
        else:
            base_mean, base_std = IMAGENET_MEAN, IMAGENET_STD
        self.patch, self.tubelet, self.use_checkpoint = patch, tubelet, use_checkpoint
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv3d(in_channels, dim, (tubelet, patch, patch), (tubelet, patch, patch))
        dpr = [float(v) for v in np.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio, dpr[i]) for i in range(depth)])
        self.fc_norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, num_classes)
        mean = base_mean + [float(np.mean(base_mean))] * max(0, in_channels - 3)
        std = base_std + [float(np.mean(base_std))] * max(0, in_channels - 3)
        self.register_buffer("in_mean", torch.tensor(mean[:in_channels]).view(1, 1, in_channels, 1, 1), persistent=False)
        self.register_buffer("in_std", torch.tensor(std[:in_channels]).view(1, 1, in_channels, 1, 1), persistent=False)
        self._pos_cache = {}
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        if pretrained:
            self.load_pretrained(self.weights)

    def load_pretrained(self, path: Path):
        sd = torch.load(path, map_location="cpu", weights_only=False)
        sd = sd.get("module", sd.get("model", sd))
        sd = {k: v.float() for k, v in sd.items()
              if not k.startswith("head.") and k not in ("pos_embed",)}
        w = sd["patch_embed.proj.weight"]                         # (384, 3, 2, 16, 16)
        if self.in_channels != 3:
            new = torch.zeros(w.shape[0], self.in_channels, *w.shape[2:])
            new[:, :min(3, self.in_channels)] = w[:, :min(3, self.in_channels)]
            if self.in_channels > 3:
                new[:, 3:] = w.mean(1, keepdim=True).expand(-1, self.in_channels - 3, -1, -1, -1)
            sd["patch_embed.proj.weight"] = new * (3.0 / self.in_channels)
        missing, unexpected = self.load_state_dict(sd, strict=False)
        missing = [m for m in missing if not m.startswith("head.")]
        assert not unexpected and not missing, (missing, unexpected)

    def pos_embed(self, n_pos: int, device, dtype):
        key = (n_pos, str(device))
        if key not in self._pos_cache:
            self._pos_cache[key] = sincos_pos_embed(n_pos, self.dim).to(device)
        return self._pos_cache[key].to(dtype)

    def forward(self, x):                      # (N, T, C, H, W) in [-0.5, 0.5]
        x = (x + 0.5 - self.in_mean) / self.in_std
        x = x.permute(0, 2, 1, 3, 4)            # (N, C, T, H, W)
        x = self.patch_embed.proj(x)            # (N, D, T/2, H/16, W/16)
        x = x.flatten(2).transpose(1, 2)        # (N, L, D)
        x = x + self.pos_embed(x.shape[1], x.device, x.dtype)
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return self.head(self.fc_norm(x.mean(1)))


def param_groups_llrd(model: VideoViT, base_lr: float, weight_decay: float, layer_decay: float = 0.75):
    """AdamW param groups: layer-wise lr decay (patch-embed deepest, head = base),
    no weight decay on biases / norms / q,v bias."""
    depth = len(model.blocks)
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("patch_embed"):
            layer = 0
        elif name.startswith("blocks."):
            layer = int(name.split(".")[1]) + 1
        else:
            layer = depth + 1
        scale = layer_decay ** (depth + 1 - layer)
        no_decay = p.ndim == 1 or name.endswith("_bias")
        key = (layer, no_decay)
        if key not in groups:
            groups[key] = {"params": [], "lr": base_lr * scale, "weight_decay": 0.0 if no_decay else weight_decay}
        groups[key]["params"].append(p)
    return list(groups.values())


# ======================= InternVideo2 (single-modality distilled S/14, B/14) =======================
# Vendored port of OpenGVLab InternVideo2 single_modality `internvideo2.py` (no flash_attn / fused
# RMSNorm): PatchEmbed Conv3d(in_ch, D, (1,14,14), bias) -> [cls] + learned pos_embed (1, 1+8*16*16, D)
# -> 12 blocks {RMSNorm(1e-6) -> Attention(qkv no bias, q/k RMSNorm over full dim, SDPA) -> LayerScale
# -> RMSNorm -> MLP(GELU) -> LayerScale} -> clip_projector (mean-query cross-attention pooling, 16 heads,
# LayerNorm eps 1e-5, out_dim 768) -> fc_norm LayerNorm(768) -> head. Input 8 frames x 224, ImageNet
# mean/std (InternVideo2 kinetics pipeline). Checkpoints: data/external/internvideo2/{S14,B14}_ft_k710_ft_k400_f8.bin (bf16).
IV2_WEIGHTS = {"iv2_s": Path("data/external/internvideo2/S14_ft_k710_ft_k400_f8.bin"),
               "iv2_b": Path("data/external/internvideo2/B14_ft_k710_ft_k400_f8.bin"),
               "iv2_s_ssv2": Path("data/external/internvideo2/S14_ft_ssv2_f8.bin")}   # SSv2-файнтюн (руки и предметы), Apache
IV2_CFG = {"iv2_s": dict(dim=384, heads=6), "iv2_b": dict(dim=768, heads=12), "iv2_s_ssv2": dict(dim=384, heads=6)}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(dt)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class IV2Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm = RMSNorm(dim)
        self.k_norm = RMSNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        n, l, d = x.shape
        q, k, v = self.qkv(x).reshape(n, l, 3, d).unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)          # qk-norm over the full dim (before head split)
        hd = d // self.heads
        q, k, v = (t.reshape(n, l, self.heads, hd).transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(n, l, d))


class IV2Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, drop_path: float):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = IV2Attention(dim, heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = RMSNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential()
        self.mlp.fc1 = nn.Linear(dim, hidden)
        self.mlp.act = nn.GELU()
        self.mlp.fc2 = nn.Linear(hidden, dim)
        self.ls2 = LayerScale(dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x


class IV2CrossAttention(nn.Module):
    """clip_projector.cross_attn: q/k/v Linear without bias + separate biases, proj -> out_dim."""
    def __init__(self, dim: int, heads: int, out_dim: int):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.k_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))
        self.proj = nn.Linear(dim, out_dim)

    def forward(self, xq, xk, xv):
        n, lq, d = xq.shape
        hd = d // self.heads
        q = F.linear(xq, self.q.weight, self.q_bias).reshape(n, lq, self.heads, hd).transpose(1, 2)
        k = F.linear(xk, self.k.weight, self.k_bias).reshape(n, -1, self.heads, hd).transpose(1, 2)
        v = F.linear(xv, self.v.weight, self.v_bias).reshape(n, -1, self.heads, hd).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(n, lq, d))


class IV2AttentionPool(nn.Module):
    """AttentionPoolingBlock: query = mean of tokens; norms LayerNorm(eps=1e-5)."""
    def __init__(self, dim: int, heads: int, out_dim: int):
        super().__init__()
        self.norm1_q = nn.LayerNorm(dim, eps=1e-5)
        self.norm1_k = nn.LayerNorm(dim, eps=1e-5)
        self.norm1_v = nn.LayerNorm(dim, eps=1e-5)
        self.cross_attn = IV2CrossAttention(dim, heads, out_dim)

    def forward(self, x):
        xq = x.mean(1, keepdim=True)
        return self.cross_attn(self.norm1_q(xq), self.norm1_k(x), self.norm1_v(x)).squeeze(1)


class VideoInternViT(nn.Module):
    def __init__(self, variant: str = "iv2_s", in_channels: int = 4, num_classes: int = 40, n_frames: int = 8,
                 depth: int = 12, mlp_ratio: float = 4.0, patch: int = 14, img: int = 224, out_dim: int = 768,
                 pool_heads: int = 16, drop_path: float = 0.1, pretrained: bool = True, use_checkpoint: bool = False,
                 weights=None):
        super().__init__()
        cfg = IV2_CFG[variant]
        dim, heads = cfg["dim"], cfg["heads"]
        self.in_channels, self.n_frames, self.dim, self.patch = in_channels, n_frames, dim, patch
        self.use_checkpoint = use_checkpoint
        self.weights = Path(weights) if weights is not None else IV2_WEIGHTS[variant]
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv3d(in_channels, dim, (1, patch, patch), (1, patch, patch))
        n_tok = n_frames * (img // patch) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tok + 1, dim))
        dpr = [float(v) for v in np.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([IV2Block(dim, heads, mlp_ratio, dpr[i]) for i in range(depth)])
        self.clip_projector = IV2AttentionPool(dim, pool_heads, out_dim)
        self.fc_norm = nn.LayerNorm(out_dim)
        self.head = nn.Linear(out_dim, num_classes)
        mean = IMAGENET_MEAN + [float(np.mean(IMAGENET_MEAN))] * max(0, in_channels - 3)
        std = IMAGENET_STD + [float(np.mean(IMAGENET_STD))] * max(0, in_channels - 3)
        self.register_buffer("in_mean", torch.tensor(mean[:in_channels]).view(1, 1, in_channels, 1, 1), persistent=False)
        self.register_buffer("in_std", torch.tensor(std[:in_channels]).view(1, 1, in_channels, 1, 1), persistent=False)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        if pretrained:
            self.load_pretrained(self.weights)

    def load_pretrained(self, path: Path):
        sd = torch.load(path, map_location="cpu", weights_only=False)
        sd = sd.get("module", sd.get("model", sd))
        sd = {k: v.float() for k, v in sd.items() if not k.startswith("head.")}
        w = sd["patch_embed.proj.weight"]                              # (D, 3, 1, 14, 14)
        if self.in_channels != 3:
            new = torch.zeros(w.shape[0], self.in_channels, *w.shape[2:])
            new[:, :min(3, self.in_channels)] = w[:, :min(3, self.in_channels)]
            if self.in_channels > 3:
                new[:, 3:] = w.mean(1, keepdim=True).expand(-1, self.in_channels - 3, -1, -1, -1)
            sd["patch_embed.proj.weight"] = new * (3.0 / self.in_channels)
        pe = sd["pos_embed"]
        if tuple(pe.shape) != tuple(self.pos_embed.shape):
            raise ValueError(f"pos_embed {tuple(pe.shape)} != {tuple(self.pos_embed.shape)}: use n_frames=8, img=224")
        missing, unexpected = self.load_state_dict(sd, strict=False)
        missing = [m for m in missing if not m.startswith("head.")]
        assert not unexpected and not missing, (missing, unexpected)

    def forward(self, x):                      # (N, T, C, H, W) in [-0.5, 0.5]
        x = (x + 0.5 - self.in_mean) / self.in_std
        x = x.permute(0, 2, 1, 3, 4)
        x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)        # (N, L, D)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], 1) + self.pos_embed
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = self.clip_projector(x)
        return self.head(self.fc_norm(x))


def param_groups_llrd_iv2(model: VideoInternViT, base_lr: float, weight_decay: float, layer_decay: float = 0.75):
    depth = len(model.blocks)
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(("patch_embed", "cls_token", "pos_embed")):
            layer = 0
        elif name.startswith("blocks."):
            layer = int(name.split(".")[1]) + 1
        else:
            layer = depth + 1
        scale = layer_decay ** (depth + 1 - layer)
        no_decay = p.ndim == 1 or name.endswith("_bias") or name in ("cls_token", "pos_embed")
        key = (layer, no_decay)
        if key not in groups:
            groups[key] = {"params": [], "lr": base_lr * scale, "weight_decay": 0.0 if no_decay else weight_decay}
        groups[key]["params"].append(p)
    return list(groups.values())
