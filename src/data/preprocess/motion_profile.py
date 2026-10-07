# -*- coding: utf-8 -*-
"""Профиль движения по клипам видеокэша (для семплера кадров «по движению», IsoGD).
Для каждого клипа <dir>/<key>.npy (T, S, S[, 3]) uint8: e[t] = mean |f_t − f_{t−1}| на прореженной сетке (::step), e[0] = e[1].
Пишет npz {key: float32 (T,)} — детерминированно, без обучаемых частей.
  python code/data/preprocess/motion_profile.py --dir C:/cuhkx_cache/IRcrop6h_train --out data/cache/motion_IRcrop6h_train.npz
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def profile(arr: np.ndarray, step: int = 4) -> np.ndarray:
    a = np.asarray(arr[:, ::step, ::step], dtype=np.float32)
    if a.ndim == 4:
        a = a.mean(-1)
    if len(a) < 2:
        return np.ones(len(a), np.float32)
    e = np.abs(np.diff(a, axis=0)).mean(axis=(1, 2))
    return np.concatenate([[e[0]], e]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=4)
    args = ap.parse_args()
    d = Path(args.dir); files = sorted(d.glob("*.npy")); t0 = time.time(); out = {}
    for i, f in enumerate(files):
        out[f.stem] = profile(np.load(f, mmap_mode="r"), args.step)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(files)} ({time.time() - t0:.0f}s)", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    lens = np.array([len(v) for v in out.values()])
    print(f"{len(out)} клипов, кадров медиана {np.median(lens):.0f} [{lens.min()}, {lens.max()}] -> {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
