"""R(2+1)D-18 with Kinetics-400 weights, fully fine-tuned.

The cardinal video move: a VIDEO-pretrained backbone. The public 0.716
solution is a single Depth+IR branch on IG65M R(2+1)D-34 (63.5M params,
65M-video pretrain). This is the 18-layer sibling from torchvision with
K400 weights: 33M params, 240K-clip pretrain -- half their size, two
orders less pretrain data, same defence line as our ImageNet decision.

Stem conv is adapted to C in-channels by averaging the RGB kernels.
Native input resolution is 112x112, 16 frames -- exactly our pipeline.
Checkpoint ~63 MB fp16 (watch the 100 MB budget: ship 1-2 folds).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights


class VideoR2P1D(nn.Module):
    def __init__(self, in_channels=2, num_classes=40, n_frames=16,
                 pretrained=True):
        super().__init__()
        self.n_frames = n_frames
        net = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1
                          if pretrained else None)
        old = net.stem[0]                      # Conv3d(3, 45, (1,7,7), ...)
        conv = nn.Conv3d(in_channels, old.out_channels, old.kernel_size,
                         old.stride, old.padding, bias=False)
        with torch.no_grad():
            mean_w = old.weight.mean(dim=1, keepdim=True)
            if in_channels >= 3:
                # first 3 channels are colormap-RGB: keep the pretrained RGB
                # kernels verbatim (E290 recipe), extra channels get the mean
                conv.weight[:, :3] = old.weight
                if in_channels > 3:
                    conv.weight[:, 3:] = mean_w.expand(
                        -1, in_channels - 3, -1, -1, -1)
            else:
                conv.weight.copy_(mean_w.expand(-1, in_channels, -1, -1, -1))
        net.stem[0] = conv
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        self.net = net

    def forward(self, x):                      # (N, T, C, H, W)
        return self.net(x.permute(0, 2, 1, 3, 4))
