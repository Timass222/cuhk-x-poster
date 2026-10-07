"""Parse WitMotion IMU CSVs into fixed-grid per-clip tensors.

Raw format (both up(LA+RA+C).csv and down(LL+RL).csv, UTF-8 with BOM,
Chinese headers -- columns are read by POSITION):
    0 time "2025-06-13 15:03:44.930"
    1 device "WTLA(D2:5B:8F:2B:49:EE)" -> WTLA/WTRA/WTC/WTLL/WTRL
    2-4 accel XYZ (g)   5-7 gyro XYZ (deg/s)   8-10 angle XYZ (deg)
    11-13 mag XYZ  14-17 quaternion  18 temp  19 version  20 battery

Devices sample at ~10 Hz each, multiplexed in file order. Streams can be
missing (no file / empty file / device absent) -- the tensor is NaN there.

Output per clip: float32 (5 devices, 9 channels, T_GRID) resampled by linear
interpolation onto a uniform grid over the clip's own IMU time span.
Channels: accel XYZ, gyro XYZ, angle XYZ. Mag/quat dropped (redundant with
angle, phone-grade mag is noisy indoors).
"""

from __future__ import annotations

import io
from datetime import datetime

import numpy as np

DEVICES = ["WTLA", "WTRA", "WTC", "WTLL", "WTRL"]
N_CH = 9
N_CH_V2 = 16        # v2: accel 3, gyro 3, angle 3, mag 3, quaternion 4 [w, x, y, z] (columns 2-17)
T_GRID = 32
T_GRID_V2 = 64      # v2: fixed 10 Hz grid from the clip's first IMU sample (6.4 s covers 93% of clips), NaN past the stream end
HZ_V2 = 10.0


def _parse_rows(text: str, n_vals: int = N_CH):
    rows = []
    for line in text.splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 2 + n_vals:
            continue
        try:
            ts = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            continue
        dev = parts[1].split("(")[0]
        try:
            vals = [float(v) for v in parts[2:2 + n_vals]]
        except ValueError:
            continue
        rows.append((ts, dev, vals))
    return rows


def clip_tensor(up_text: str | None, down_text: str | None,
                t_grid: int = T_GRID, n_ch: int = N_CH) -> np.ndarray:
    """(5, n_ch, t_grid) float32, NaN where a device stream is absent.
    n_ch=9: accel/gyro/angle (v1, unchanged); n_ch=16 (v2): + mag 3 + quaternion 4 [w,x,y,z]
    (quaternion sign made continuous along time, re-normalised after interpolation)."""
    rows = []
    if up_text:
        rows += _parse_rows(up_text, n_ch)
    if down_text:
        rows += _parse_rows(down_text, n_ch)
    out = np.full((len(DEVICES), n_ch, t_grid), np.nan, dtype=np.float32)
    if not rows:
        return out
    t0 = min(r[0] for r in rows)
    t1 = max(r[0] for r in rows)
    span = max((t1 - t0).total_seconds(), 1e-3)
    grid = np.linspace(0.0, span, t_grid)
    for d, dev in enumerate(DEVICES):
        sub = [(r[0], r[2]) for r in rows if r[1] == dev]
        if len(sub) < 2:
            continue
        sub.sort()
        t = np.array([(ts - t0).total_seconds() for ts, _ in sub])
        v = np.array([vals for _, vals in sub], dtype=np.float32)
        # unwrap angle channels (deg, can jump 360)
        v[:, 6:9] = np.unwrap(v[:, 6:9], period=360.0, axis=0)
        if n_ch >= 16:
            q = v[:, 12:16]
            for i in range(1, len(q)):          # hemisphere continuity: q and -q are the same rotation
                if np.dot(q[i], q[i - 1]) < 0:
                    q[i] = -q[i]
            v[:, 12:16] = q
        for c in range(n_ch):
            out[d, c] = np.interp(grid, t, v[:, c])
        if n_ch >= 16:
            nrm = np.linalg.norm(out[d, 12:16], axis=0, keepdims=True)
            out[d, 12:16] = out[d, 12:16] / np.where(nrm > 1e-6, nrm, 1.0)
    return out


def summary_features(x: np.ndarray) -> np.ndarray:
    """Fixed-length stats vector from (5, 9, T): mean/std/min/max/energy of
    value and first difference per device-channel -> 5*9*10 = 450 dims.
    NaN streams yield NaN features (imputed downstream)."""
    d = np.diff(x, axis=-1)
    feats = [
        x.mean(-1), x.std(-1), x.min(-1), x.max(-1), (x ** 2).mean(-1),
        d.mean(-1), d.std(-1), np.abs(d).mean(-1), d.min(-1), d.max(-1),
    ]
    return np.stack(feats, -1).reshape(-1).astype(np.float32)


def clip_tensor_v2(up_text: str | None, down_text: str | None,
                   n_steps: int = T_GRID_V2, hz: float = HZ_V2) -> np.ndarray:
    """(5, 16, n_steps) float32 on a FIXED grid t0 + k/hz (t0 = first IMU sample of the clip, any device).
    Channels: accel 3, gyro 3, angle 3 (deg, unwrapped), mag 3, quaternion 4 [w,x,y,z] DERIVED FROM THE EULER
    ANGLES (scipy from_euler('xyz')): the raw WitMotion quaternion updates only every ~2.5 samples while the
    angles update every sample. NaN where the device stream is absent or the grid step lies outside the
    stream's own time span (+-1.5 samples tolerance); a fully absent stream is NaN everywhere."""
    from scipy.spatial.transform import Rotation as R
    rows = []
    if up_text:
        rows += _parse_rows(up_text, N_CH_V2)
    if down_text:
        rows += _parse_rows(down_text, N_CH_V2)
    out = np.full((len(DEVICES), N_CH_V2, n_steps), np.nan, dtype=np.float32)
    if not rows:
        return out
    t0 = min(r[0] for r in rows)
    grid = np.arange(n_steps) / hz
    tol = 1.5 / hz
    for d, dev in enumerate(DEVICES):
        sub = [(r[0], r[2]) for r in rows if r[1] == dev]
        if len(sub) < 2:
            continue
        sub.sort()
        t = np.array([(ts - t0).total_seconds() for ts, _ in sub])
        v = np.array([vals for _, vals in sub], dtype=np.float64)
        v[:, 6:9] = np.unwrap(v[:, 6:9], period=360.0, axis=0)
        q = R.from_euler("xyz", v[:, 6:9], degrees=True).as_quat()[:, [3, 0, 1, 2]]  # -> [w, x, y, z]
        for i in range(1, len(q)):
            if np.dot(q[i], q[i - 1]) < 0:
                q[i] = -q[i]
        v[:, 12:16] = q
        valid = (grid >= t[0] - tol) & (grid <= t[-1] + tol)
        if valid.sum() == 0:
            continue
        idx = np.where(valid)[0]
        for c in range(N_CH_V2):
            out[d, c, idx] = np.interp(grid[idx], t, v[:, c])
        blk = out[d][12:16][:, idx]                      # (4, n_valid): explicit view, no mixed fancy indexing
        nrm = np.linalg.norm(blk, axis=0, keepdims=True)
        out[d][12:16][:, idx] = blk / np.where(nrm > 1e-6, nrm, 1.0)
    return out

