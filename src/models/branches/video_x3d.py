"""X3D-M (Kinetics-400, pytorchvideo hub), fully fine-tuned.

3.8M parameters -- the packaging-friendly backbone: fp16 checkpoint
~7.6 MB, five folds fit trivially under the 100 MB cap. Efficiency-first
architecture (progressive expansion); native 16x224, we feed 16x128 so
the head pool is swapped for AdaptiveAvgPool3d.

Stem conv adapted to C in-channels like the other video branches:
first 3 channels keep the pretrained RGB kernels, extras get the mean.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VideoX3D(nn.Module):
    def __init__(self, in_channels=2, num_classes=40, n_frames=16,
                 pretrained=True):
        super().__init__()
        # imported here: the final packs do not use X3D, so inference does not need pytorchvideo
        from pytorchvideo.models.hub import x3d_m
        self.n_frames = n_frames
        net = x3d_m(pretrained=pretrained)
        # первый Conv3d с in_channels=3 — пространственный конв стема
        stem_conv, parent, name = None, None, None
        for mod in net.modules():
            for cname, child in mod.named_children():
                if (isinstance(child, nn.Conv3d)
                        and child.in_channels == 3):
                    stem_conv, parent, name = child, mod, cname
                    break
            if stem_conv is not None:
                break
        assert stem_conv is not None, "стем-конв не найден"
        conv = nn.Conv3d(in_channels, stem_conv.out_channels,
                         stem_conv.kernel_size, stem_conv.stride,
                         stem_conv.padding, bias=stem_conv.bias
                         is not None)
        with torch.no_grad():
            mean_w = stem_conv.weight.mean(dim=1, keepdim=True)
            if in_channels >= 3:
                conv.weight[:, :3] = stem_conv.weight
                if in_channels > 3:
                    conv.weight[:, 3:] = mean_w.expand(
                        -1, in_channels - 3, -1, -1, -1)
            else:
                conv.weight.copy_(mean_w.expand(-1, in_channels,
                                                -1, -1, -1))
        setattr(parent, name, conv)
        # голова: у ProjectedPool заменяем только внутренний AvgPool3d
        # (фикс-ядро 16x7x7 рассчитано на вход 224; у нас 128)
        head = net.blocks[-1]
        head.pool.pool = nn.AdaptiveAvgPool3d(1)
        assert isinstance(head.proj, nn.Linear)
        head.proj = nn.Linear(head.proj.in_features, num_classes)
        self.net = net

    def forward(self, x):                      # (N, T, C, H, W)
        return self.net(x.permute(0, 2, 1, 3, 4))
