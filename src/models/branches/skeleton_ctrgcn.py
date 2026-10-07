"""CTR-GCN (Channel-wise Topology Refinement GCN, ICCV 2021), compact.

Differences from ST-GCN that matter: instead of a fixed adjacency shared
by all channels, each block *infers* a per-sample, per-channel refinement
of the topology from pairwise feature differences, so distant joints
(hand-head, hand-hip) can couple when the motion demands it. Temporal
part is a multi-scale conv (two dilations + max-pool + 1x1 branches).

~1.5M params at base width 64 for COCO-17. Input (N, C, T, V).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .skeleton_gcn import build_adjacency, V  # COCO-17 graph, FLIP_PERM lives there


class CTRGC(nn.Module):
    def __init__(self, c_in, c_out, rel_reduction=8):
        super().__init__()
        rel = 8 if c_in <= 16 else c_in // rel_reduction
        self.conv_k = nn.Conv2d(c_in, rel, 1)
        self.conv_q = nn.Conv2d(c_in, rel, 1)
        self.conv_v = nn.Conv2d(c_in, c_out, 1)
        self.expand = nn.Conv2d(rel, c_out, 1)
        self.tanh = nn.Tanh()

    def forward(self, x, A, alpha):
        # x (N,C,T,V); A (V,V) static partition
        k = self.conv_k(x).mean(-2)                 # (N,rel,V)
        q = self.conv_q(x).mean(-2)
        v = self.conv_v(x)                          # (N,C_out,T,V)
        dyn = self.tanh(k.unsqueeze(-1) - q.unsqueeze(-2))   # (N,rel,V,V)
        adj = self.expand(dyn) * alpha + A.view(1, 1, V, V)  # (N,C_out,V,V)
        return torch.einsum("ncuv,nctv->nctu", adj, v)


class SpatialUnit(nn.Module):
    """Sum of CTRGC over the 3 static partitions."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.register_buffer("A", build_adjacency())         # (3,V,V)
        self.gcns = nn.ModuleList(CTRGC(c_in, c_out) for _ in range(3))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(c_out)
        self.down = (nn.Sequential(nn.Conv2d(c_in, c_out, 1),
                                   nn.BatchNorm2d(c_out))
                     if c_in != c_out else nn.Identity())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = None
        for i, gcn in enumerate(self.gcns):
            z = gcn(x, self.A[i], self.alpha)
            y = z if y is None else y + z
        return self.relu(self.bn(y) + self.down(x))


class TemporalBranch(nn.Module):
    def __init__(self, c_in, c_out, kernel=5, stride=1, dilation=1):
        super().__init__()
        pad = (kernel + (kernel - 1) * (dilation - 1) - 1) // 2
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 1), nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, (kernel, 1), (stride, 1), (pad, 0),
                      dilation=(dilation, 1)),
            nn.BatchNorm2d(c_out))

    def forward(self, x):
        return self.net(x)


class MultiScaleTCN(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        b = c_out // 4
        self.branches = nn.ModuleList([
            TemporalBranch(c_in, b, 5, stride, 1),
            TemporalBranch(c_in, b, 5, stride, 2),
            nn.Sequential(nn.Conv2d(c_in, b, 1), nn.BatchNorm2d(b),
                          nn.ReLU(inplace=True),
                          nn.MaxPool2d((3, 1), (stride, 1), (1, 0)),
                          nn.BatchNorm2d(b)),
            nn.Sequential(nn.Conv2d(c_in, c_out - 3 * b, 1, (stride, 1)),
                          nn.BatchNorm2d(c_out - 3 * b)),
        ])

    def forward(self, x):
        return torch.cat([br(x) for br in self.branches], 1)


class Block(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.spatial = SpatialUnit(c_in, c_out)
        self.temporal = MultiScaleTCN(c_out, c_out, stride)
        self.res = (nn.Identity() if c_in == c_out and stride == 1 else
                    nn.Sequential(nn.Conv2d(c_in, c_out, 1, (stride, 1)),
                                  nn.BatchNorm2d(c_out)))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.temporal(self.spatial(x)) + self.res(x))


class CTRGCN(nn.Module):
    def __init__(self, in_channels=5, num_classes=40, base=64):
        super().__init__()
        self.bn_in = nn.BatchNorm1d(in_channels * V)
        cfg = [(in_channels, base, 1), (base, base, 1), (base, base, 1),
               (base, base, 1), (base, 2 * base, 2), (2 * base, 2 * base, 1),
               (2 * base, 2 * base, 1), (2 * base, 4 * base, 2),
               (4 * base, 4 * base, 1), (4 * base, 4 * base, 1)]
        self.blocks = nn.ModuleList(Block(a, b, s) for a, b, s in cfg)
        self.head = nn.Linear(4 * base, num_classes)

    def forward(self, x):                          # (N, C, T, V)
        n, c, t, v = x.shape
        x = self.bn_in(x.permute(0, 1, 3, 2).reshape(n, c * v, t))
        x = x.view(n, c, v, t).permute(0, 1, 3, 2).contiguous()
        for blk in self.blocks:
            x = blk(x)
        return self.head(x.mean(dim=(2, 3)))
