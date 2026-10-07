"""FrameNet -- the per-frame CNN + mean-pooling baseline from the public
"14th place 0.8+ thermal" notebook, reimplemented under our harness.

Per frame: a small ResNet (32-64-128-256, four stride-2 stages), then the
40-way head is applied to every frame embedding and logits are averaged
over frames. No temporal modelling at all -- the honest comparison point
for the TSM branch on the same frozen folds.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, a, b, s=1):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(a, b, 3, s, 1, bias=False), nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
            nn.Conv2d(b, b, 3, 1, 1, bias=False), nn.BatchNorm2d(b))
        self.d = (nn.Sequential(nn.Conv2d(a, b, 1, s, bias=False),
                                nn.BatchNorm2d(b))
                  if (a != b or s != 1) else nn.Identity())

    def forward(self, x):
        return torch.relu(self.c(x) + self.d(x))


class FrameNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=40, n_frames=8, width=32):
        super().__init__()
        self.n_frames = n_frames
        w = width
        self.f = nn.Sequential(
            nn.Conv2d(in_channels, w, 7, 2, 3, bias=False),
            nn.BatchNorm2d(w), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1),
            Block(w, w), Block(w, 2 * w, 2), Block(2 * w, 4 * w, 2),
            Block(4 * w, 8 * w, 2), nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(8 * w, num_classes)

    def forward(self, x):                          # (N, T, C, H, W)
        n, t, c, h, w = x.shape
        z = self.f(x.reshape(n * t, c, h, w)).flatten(1)
        return self.fc(z).reshape(n, t, -1).mean(1)
