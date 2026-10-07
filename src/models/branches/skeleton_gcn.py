"""Compact ST-GCN for COCO-17 skeletons, trained from scratch.

Input  (N, C, T, V): C channels over T resampled frames, V = 17 joints.
Spatial graph convolution over 3 partitions (self, inward to nose, outward)
with a learnable edge-importance mask, then temporal 9x1 convs; 6 blocks,
~1.1M params at width 64-128-256 -- far under the 100 MB cap.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# The dataset skeleton is a LIFTED 3D pose in H36M-17 order (verified 5 Sep:
# joint 0 = pelvis at (0,0), channel 2 = height rebased to 0, scores == 1):
# 0 pelvis, 1/2/3 R hip/knee/ankle, 4/5/6 L hip/knee/ankle, 7 spine,
# 8 thorax, 9 neck, 10 head, 11/12/13 L shoulder/elbow/wrist,
# 14/15/16 R shoulder/elbow/wrist. Channels = [x lateral, depth, height].
EDGES = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8),
         (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15),
         (15, 16)]
V = 17
# left-right joint swap for mirror augmentation (x -> -x)
FLIP_PERM = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]


def build_adjacency() -> torch.Tensor:
    """(3, V, V): identity, inward (toward nose along tree), outward."""
    import collections
    adj = collections.defaultdict(set)
    for a, b in EDGES:
        adj[a].add(b)
        adj[b].add(a)
    # BFS depth from nose
    depth = {0: 0}
    queue = [0]
    while queue:
        u = queue.pop(0)
        for w in adj[u]:
            if w not in depth:
                depth[w] = depth[u] + 1
                queue.append(w)
    A = torch.zeros(3, V, V)
    A[0] = torch.eye(V)
    for a, b in EDGES:
        hi, lo = (a, b) if depth[a] > depth[b] else (b, a)
        A[1, hi, lo] = 1.0     # inward: from deeper joint to shallower
        A[2, lo, hi] = 1.0     # outward
    # column-normalise each partition
    for p in range(3):
        d = A[p].sum(0, keepdim=True).clamp(min=1)
        A[p] = A[p] / d
    return A


class STGCNBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1, dropout=0.1):
        super().__init__()
        self.A = None  # set by parent
        self.gcn = nn.Conv2d(c_in, c_out * 3, 1)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, (9, 1), (stride, 1), (4, 0)),
            nn.BatchNorm2d(c_out), nn.Dropout(dropout),
        )
        self.res = (nn.Identity() if c_in == c_out and stride == 1 else
                    nn.Conv2d(c_in, c_out, 1, (stride, 1)))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        n, _, t, v = x.shape
        res = self.res(x)
        x = self.gcn(x).view(n, 3, -1, t, v)
        x = torch.einsum("npctv,pvw->nctw", x, A)
        x = self.tcn(x)
        return self.relu(x + res)


class STGCN(nn.Module):
    def __init__(self, in_channels=5, num_classes=40, width=64):
        super().__init__()
        self.register_buffer("A_base", build_adjacency())
        self.edge_mask = nn.Parameter(torch.ones(3, V, V))
        self.bn_in = nn.BatchNorm1d(in_channels * V)
        cfg = [(in_channels, width, 1), (width, width, 1),
               (width, width * 2, 2), (width * 2, width * 2, 1),
               (width * 2, width * 4, 2), (width * 4, width * 4, 1)]
        self.blocks = nn.ModuleList(STGCNBlock(a, b, s) for a, b, s in cfg)
        self.head = nn.Linear(width * 4, num_classes)

    def forward(self, x):                     # (N, C, T, V)
        n, c, t, v = x.shape
        x = self.bn_in(x.permute(0, 1, 3, 2).reshape(n, c * v, t))
        x = x.view(n, c, v, t).permute(0, 1, 3, 2).contiguous()
        A = self.A_base * self.edge_mask
        for blk in self.blocks:
            x = blk(x, A)
        x = x.mean(dim=(2, 3))
        return self.head(x)
