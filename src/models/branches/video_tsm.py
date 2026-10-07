"""Slim ResNet + Temporal Shift Module, trained from scratch.

2D convolutions over T sampled frames; TSM shifts 1/8 of channels one step
forward and 1/8 backward along time before each residual block, giving
temporal modelling at zero parameter cost. Width 32: ~2.8M params (~11 MB
fp32) for the 2-channel Depth+IR branch -- far under the 100 MB cap even
summed with every other branch.

Input (N, T, C, H, W) -> logits (N, num_classes).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def tsm(x: torch.Tensor, n_frames: int, fold_div: int = 8) -> torch.Tensor:
    """Temporal shift: (N*T, C, H, W) -> same, channels rolled along T."""
    nt, c, h, w = x.shape
    n = nt // n_frames
    x = x.view(n, n_frames, c, h, w)
    fold = c // fold_div
    out = torch.zeros_like(x)
    out[:, 1:, :fold] = x[:, :-1, :fold]          # shift forward
    out[:, :-1, fold:2 * fold] = x[:, 1:, fold:2 * fold]  # shift back
    out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
    return out.view(nt, c, h, w)


class TSMBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1, n_frames=16):
        super().__init__()
        self.n_frames = n_frames
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or c_in != c_out:
            self.down = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride, bias=False),
                nn.BatchNorm2d(c_out))

    def forward(self, x):
        res = x if self.down is None else self.down(x)
        x = tsm(x, self.n_frames)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + res)


class VideoTSM(nn.Module):
    def __init__(self, in_channels=2, num_classes=40, width=32, n_frames=16):
        super().__init__()
        self.n_frames = n_frames
        w = width
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, w, 7, 2, 3, bias=False),
            nn.BatchNorm2d(w), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1))
        layers = []
        for c_in, c_out, stride in [(w, w, 1), (w, w, 1),
                                    (w, 2 * w, 2), (2 * w, 2 * w, 1),
                                    (2 * w, 4 * w, 2), (4 * w, 4 * w, 1),
                                    (4 * w, 8 * w, 2), (8 * w, 8 * w, 1)]:
            layers.append(TSMBlock(c_in, c_out, stride, n_frames))
        self.blocks = nn.ModuleList(layers)
        self.head = nn.Linear(8 * w, num_classes)

    def forward(self, x):                          # (N, T, C, H, W)
        n, t, c, h, w = x.shape
        assert t == self.n_frames, (t, self.n_frames)
        x = x.view(n * t, c, h, w)
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        x = x.mean(dim=(2, 3)).view(n, t, -1).mean(1)
        return self.head(x)
