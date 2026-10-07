"""TSM over an ImageNet-pretrained torchvision ResNet-18, fully fine-tuned.

Rules note (documented decision, 30 Aug): the track bans *large* pretrained
backbones; ImageNet-1k ResNet-18 (11.7M params, 1.28M images) is the
classic small edge backbone and is used as initialisation only -- every
weight is fine-tuned on competition data. The public 0.716 solution shipped
a frozen IG65M R(2+1)D-34 (65M videos), which is the thing the rule is
actually about.

The first conv is adapted to C in-channels by averaging the RGB kernels;
the TSM shift (1/8 of channels forward, 1/8 back along time) is applied
before each BasicBlock, zero extra parameters. fp16 checkpoint ~22 MB.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from .video_tsm import tsm


class ShiftedBlock(nn.Module):
    def __init__(self, block: nn.Module, n_frames: int):
        super().__init__()
        self.block = block
        self.n_frames = n_frames

    def forward(self, x):
        return self.block(tsm(x, self.n_frames))


class VideoTSMR18(nn.Module):
    def __init__(self, in_channels=2, num_classes=40, n_frames=16,
                 pretrained=True):
        super().__init__()
        self.n_frames = n_frames
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1
                       if pretrained else None)
        old = net.conv1
        conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            mean_w = old.weight.mean(dim=1, keepdim=True)   # (64,1,7,7)
            conv1.weight.copy_(mean_w.expand(-1, in_channels, -1, -1))
        net.conv1 = conv1
        for layer in (net.layer1, net.layer2, net.layer3, net.layer4):
            for i, block in enumerate(layer):
                layer[i] = ShiftedBlock(block, n_frames)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        self.net = net

    def forward(self, x):                          # (N, T, C, H, W)
        n, t, c, h, w = x.shape
        assert t == self.n_frames
        feats = self.net(x.reshape(n * t, c, h, w))
        return feats.view(n, t, -1).mean(1)
