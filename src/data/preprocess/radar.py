"""mmWave radar point-cloud CSV -> fixed-length clip features.

Raw format (radar_output_T*.csv, one file per clip, ~48% of clips non-empty):
    timestamp,frame,DetObj#,x,y,z,v,snr,noise
Rows are detected points, several per frame. Empty clips = header only.

Features (per clip, NaN vector when no points):
  global point stats: counts, xyz/r moments, doppler |v| stats + histogram,
  snr/noise stats; per-frame series: point count, centroid, spread, mean|v|
  -> temporal stats (movement energy separates walk/squat from sit/read).
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

V_BINS = np.array([0.0, 0.05, 0.15, 0.3, 0.6, 1.0, 2.0, np.inf])
N_FEAT = 75


def clip_features(text: str | None) -> np.ndarray:
    """Fixed-length float32 vector from one radar CSV's text."""
    out = np.full(N_FEAT, np.nan, np.float32)
    if not text:
        return out
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception:
        return out
    if len(df) == 0 or "frame" not in df.columns:
        return out
    x = df["x"].to_numpy(float)
    y = df["y"].to_numpy(float)
    z = df["z"].to_numpy(float)
    v = df["v"].to_numpy(float)
    snr = df["snr"].to_numpy(float)
    noise = df["noise"].to_numpy(float)
    r = np.sqrt(x * x + y * y + z * z)
    av = np.abs(v)

    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    dur = max((ts.max() - ts.min()).total_seconds(), 1e-3)

    g = df.groupby("frame")
    cnt = g.size().to_numpy(float)                       # points per frame
    cx = g["x"].mean().to_numpy()
    cy = g["y"].mean().to_numpy()
    cz = g["z"].mean().to_numpy()
    spread = g.apply(
        lambda fr: float(np.sqrt(fr["x"].var() + fr["y"].var()
                                 + fr["z"].var()))
        if len(fr) > 1 else 0.0).to_numpy(float)
    fav = g["v"].apply(lambda s: float(np.abs(s).mean())).to_numpy(float)

    def stats(a):                                        # 5 dims
        if len(a) == 0 or np.all(~np.isfinite(a)):
            return [np.nan] * 5
        return [np.nanmean(a), np.nanstd(a), np.nanmin(a), np.nanmax(a),
                float(np.nanpercentile(a, 90))]

    def dstats(a):                                       # 3 dims
        if len(a) < 2:
            return [np.nan] * 3
        d = np.diff(a)
        return [float(np.nanstd(d)), float(np.nanmean(np.abs(d))),
                float(np.nanmax(np.abs(d)))]

    hist = np.histogram(av, bins=V_BINS)[0].astype(float)
    hist = hist / max(hist.sum(), 1.0)                   # 7 dims

    feats = [
        float(len(df)), float(len(cnt)), float(len(df)) / len(cnt), dur,
        float(len(df)) / dur,
        *stats(x), *stats(y), *stats(z), *stats(r),      # 20
        *stats(av), float((av > 0.05).mean()),           # 6
        *hist,                                           # 7
        *stats(snr), *stats(noise),                      # 10
        *stats(cnt), *dstats(cnt),                       # 8
        *dstats(cx), *dstats(cy), *dstats(cz),           # 9
        float(np.nanmax(cz) - np.nanmin(cz)) if len(cz) else np.nan,
        *stats(spread),                                  # 5
        *dstats(fav), float(np.nanmax(fav)) if len(fav) else np.nan,
    ]
    arr = np.asarray(feats, np.float32)
    assert arr.shape == (N_FEAT,), arr.shape
    return arr
