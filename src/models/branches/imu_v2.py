# -*- coding: utf-8 -*-
"""IMU v2 branch (docs/imu_v2_arch.md, final configuration after review).

Raw input per clip: (5, 16, 64) on a fixed 10 Hz grid — channels per device: acc 3 (g), gyro 3 (deg/s),
angle 3 (deg, unused), mag 3 (unused), quaternion 4 [w, x, y, z] derived from the Euler angles (sensor -> world,
world z = up). NaN = stream absent or grid step outside the stream span. Training cuts a random 32-step window,
inference averages windows (stride 16).

Feature layer (inside the model, so train/test share one implementation), per device, 21 channels, all yaw-invariant:
  lin_s 3 (acc - R^T z), tilt 3 (R^T z), gyro 3, |gyro| 1, |acc| 1, |lin| 1, |d lin_s/dt| 1,
  (R lin)_z 1, |(R lin)_xy| 1, rel6D 6 (limbs: first two columns of R_waist^T R_dev; waist: zeros)
+ present 1 + one-hot device 5 + log-duration 1 = 28 input channels.
Model: shared ResCNN stem (28->32 pool2 ->64) per device -> concat 320 -> dilated temporal CNN 160 (d 1,2,4)
-> attention pooling (+ masked mean) -> head 320->128->40.  ~440k parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICES = ["WTLA", "WTRA", "WTC", "WTLL", "WTRL"]
WAIST = 2
DT = 0.1
N_PHYS = 21                 # physical features per device
N_IN = N_PHYS + 1 + 5 + 1   # + present + one-hot device + log-duration = 28
WIN = 32


# ---------------------------------------------------------------- quaternion helpers (w, x, y, z)
def q_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def q_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], -1)


def q_to_mat(q: torch.Tensor) -> torch.Tensor:
    """(..., 4) -> (..., 3, 3) with v_world = R v_sensor."""
    w, x, y, z = q.unbind(-1)
    r = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], -1)
    return r.reshape(*q.shape[:-1], 3, 3)


def q_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """axis (..., 3) unit, angle (...) rad -> (..., 4) [w, x, y, z]."""
    half = angle[..., None] / 2
    return torch.cat([torch.cos(half), axis * torch.sin(half)], -1)


# ---------------------------------------------------------------- feature layer
class IMUFeatures(nn.Module):
    """raw (B,5,16,T), valid (B,5,T) bool, present (B,5), logdur (B,) -> (B,5,28,T)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(5, N_PHYS, 1))
        self.register_buffer("std", torch.ones(5, N_PHYS, 1))

    @staticmethod
    def physical(x: torch.Tensor) -> torch.Tensor:
        """(B,5,16,T) with NaN -> 0 already applied -> (B,5,21,T)."""
        B, D, C, T = x.shape
        acc = x[:, :, 0:3].permute(0, 1, 3, 2)                       # (B,5,T,3)
        gyro = x[:, :, 3:6].permute(0, 1, 3, 2)
        q = q_normalize(x[:, :, 12:16].permute(0, 1, 3, 2))
        R = q_to_mat(q)                                                # (B,5,T,3,3)
        g = torch.zeros_like(acc)
        g[..., 2] = 1.0
        tilt = torch.einsum("bdtji,bdtj->bdti", R, g)                  # R^T z: gravity in the sensor frame
        lin_s = acc - tilt
        lin_w = torch.einsum("bdtij,bdtj->bdti", R, lin_s)             # world-frame linear acceleration
        lin_vert = lin_w[..., 2:3]
        lin_horiz = lin_w[..., :2].norm(dim=-1, keepdim=True)
        acc_mag = acc.norm(dim=-1, keepdim=True)
        lin_mag = lin_s.norm(dim=-1, keepdim=True)
        jerk = torch.diff(lin_s, dim=2, prepend=lin_s[:, :, :1]).norm(dim=-1, keepdim=True) / DT
        gyro_mag = gyro.norm(dim=-1, keepdim=True)
        R_w = R[:, WAIST:WAIST + 1]
        R_rel = R_w.transpose(-1, -2) @ R                              # (B,5,T,3,3)
        rel6d = R_rel[..., :, :2].reshape(B, D, T, 6)
        rel6d = torch.cat([rel6d[:, :WAIST], torch.zeros_like(rel6d[:, WAIST:WAIST + 1]), rel6d[:, WAIST + 1:]], 1)
        f = torch.cat([lin_s, tilt, gyro, gyro_mag, acc_mag, lin_mag, jerk, lin_vert, lin_horiz, rel6d], -1)
        return f.permute(0, 1, 3, 2)                                   # (B,5,21,T)

    def forward(self, x, valid, present, logdur):
        B, D, C, T = x.shape
        f = self.physical(torch.nan_to_num(x, nan=0.0))
        f = (f - self.mean) / self.std
        # zero after standardisation: absent devices, invalid steps, and every device if the waist is absent (rel6D)
        keep = valid.float() * present[:, :, None]
        f = f * keep[:, :, None, :]
        f[:, :, 15:21] = f[:, :, 15:21] * present[:, WAIST, None, None, None]
        const = torch.cat([present[:, :, None], torch.eye(5, device=x.device)[None].expand(B, -1, -1),
                           logdur[:, None, None].expand(-1, 5, 1)], -1)  # (B,5,7)
        return torch.cat([f, const[:, :, :, None].expand(-1, -1, -1, T)], 2)  # (B,5,28,T)


# ---------------------------------------------------------------- blocks
class SE(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        h = max(ch // r, 4)
        self.fc = nn.Sequential(nn.Linear(ch, h), nn.ReLU(inplace=True), nn.Linear(h, ch), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x.mean(-1))[:, :, None]


class ResCNNBlock(nn.Module):
    def __init__(self, cin, cout, k=3, pool=2, drop=0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True),
            nn.Conv1d(cout, cout, k, padding=k // 2, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True))
        self.se = SE(cout)
        self.short = None if cin == cout else nn.Sequential(nn.Conv1d(cin, cout, 1, bias=False), nn.BatchNorm1d(cout))
        self.pool = nn.MaxPool1d(pool) if pool else nn.Identity()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        y = self.se(self.cnn(x)) + (x if self.short is None else self.short(x))
        return self.drop(self.pool(F.relu(y)))


class TempBlock(nn.Module):
    def __init__(self, cin, cout, k=3, dil=1, drop=0.2):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(cin, cout, k, padding=dil * (k // 2), dilation=dil, bias=False),
                                  nn.BatchNorm1d(cout), nn.ReLU(inplace=True), nn.Dropout(drop))
        self.short = None if cin == cout else nn.Conv1d(cin, cout, 1, bias=False)

    def forward(self, x):
        return self.conv(x) + (x if self.short is None else self.short(x))


class AttnPool(nn.Module):
    def __init__(self, ch, hid=20):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(ch, hid), nn.Tanh(), nn.Linear(hid, 1))

    def forward(self, x, tmask):                    # x (B,C,T'), tmask (B,T') float
        h = x.transpose(1, 2)
        a = self.att(h).squeeze(-1).masked_fill(tmask < 0.5, -1e4)
        a = torch.softmax(a, 1)[..., None]
        mean = (h * tmask[..., None]).sum(1) / tmask.sum(1, keepdim=True).clamp_min(1.0)
        return torch.cat([(a * h).sum(1), mean], -1)


class IMUv2(nn.Module):
    def __init__(self, n_classes=40, stem=(32, 64), temp=160, head=128, drop=0.2, shared_stem=True):
        super().__init__()
        self.features = IMUFeatures()
        self.shared_stem = shared_stem
        mk = lambda: nn.Sequential(ResCNNBlock(N_IN, stem[0], 3, 2, drop), ResCNNBlock(stem[0], stem[1], 3, None, drop))
        self.stems = nn.ModuleList([mk()] if shared_stem else [mk() for _ in range(5)])
        self.temporal = nn.Sequential(TempBlock(5 * stem[1], temp, 3, 1, drop), TempBlock(temp, temp, 3, 2, drop), TempBlock(temp, temp, 3, 4, drop))
        self.pool = AttnPool(temp)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * temp, head, bias=False), nn.BatchNorm1d(head),
                                  nn.ReLU(inplace=True), nn.Dropout(drop), nn.Linear(head, n_classes))

    def forward(self, x, valid, present, logdur):
        f = self.features(x, valid, present, logdur)                  # (B,5,28,T)
        outs = [self.stems[0 if self.shared_stem else d](f[:, d]) for d in range(5)]
        h = self.temporal(torch.cat(outs, 1))                          # (B,temp,T/2)
        tm = F.max_pool1d((valid.float() * present[:, :, None]).amax(1)[:, None], 2).squeeze(1)  # (B,T/2)
        return self.head(self.pool(h, tm))


def count_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    m = IMUv2()
    x = torch.randn(2, 5, 16, WIN)
    x[:, :, 12:16] = F.normalize(torch.randn(2, 5, 4, WIN), dim=2)
    valid = torch.ones(2, 5, WIN, dtype=torch.bool)
    print("params", count_params(m), "| out", m(x, valid, torch.ones(2, 5), torch.zeros(2)).shape)
    print("params (5 стемов)", count_params(IMUv2(shared_stem=False)))
