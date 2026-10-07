"""R(2+1)D-34 (IG65M -> Kinetics-400 finetune), fully fine-tuned.

The backbone of the public LB-0.711 notebook and the E290 lineage: 63.5M
params, pretrained on 65M Instagram videos then finetuned on K400 (clip32
variant). Architecture source vendored from moabitcoin/ig65m-pytorch at
the pinned commit fc749e2 (MIT); weights downloaded once to
data/external/ig65m/ (disclosed in the report as an external pretrain).

Stem conv is adapted to C in-channels the same way as VideoR2P1D:
colormap-RGB keeps the pretrained kernels, extra channels get the mean.
fp16 checkpoint ~127 MB -- packaging needs int5/6 or distillation.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .ig65m_models import r2plus1d_34_32_kinetics

WEIGHTS = Path("data/external/ig65m/"
               "r2plus1d_34_clip32_ft_kinetics_from_ig65m.pth")


class VideoR2P1D34(nn.Module):
    def __init__(self, in_channels=2, num_classes=40, n_frames=16,
                 pretrained=True):
        super().__init__()
        self.n_frames = n_frames
        net = r2plus1d_34_32_kinetics(num_classes=400, pretrained=False)
        if pretrained:
            state = torch.load(WEIGHTS, map_location="cpu",
                               weights_only=True)
            net.load_state_dict(state)
        old = net.stem[0]                      # Conv3d(3, 45, (1,7,7))
        conv = nn.Conv3d(in_channels, old.out_channels, old.kernel_size,
                         old.stride, old.padding, bias=False)
        with torch.no_grad():
            mean_w = old.weight.mean(dim=1, keepdim=True)
            if in_channels >= 3:
                conv.weight[:, :3] = old.weight
                if in_channels > 3:
                    conv.weight[:, 3:] = mean_w.expand(
                        -1, in_channels - 3, -1, -1, -1)
            else:
                conv.weight.copy_(mean_w.expand(-1, in_channels,
                                                -1, -1, -1))
        net.stem[0] = conv
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        self.net = net

    def forward(self, x):                      # (N, T, C, H, W)
        return self.net(x.permute(0, 2, 1, 3, 4))
