"""Person bounding boxes per clip via temporal motion.

The rig is static, only the person moves: per-pixel std over ~16 sampled
frames lights up the action tube. Mask = smoothed std above 25% of its
max; largest connected component; square box with the LB-validated
constants of the public 0.716 notebook (margin 1.40, min side 0.35).

IR reuses the Depth_Color boxes (frame-synchronous, same camera).

Output: data/meta/bbox_<mod>_<split>.json  {key: [y0, y1, x0, x1]} as
fractions of the frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

import os
CACHE = Path(os.environ.get("CUHKX_CACHE", "data/cache"))
META = Path(os.environ.get("CUHKX_META", "data/meta"))


def motion_bbox(arr: np.ndarray, margin: float = 1.40,
                min_side: float = 0.35):
    t, s, _ = arr.shape
    idx = np.linspace(0, t - 1, min(t, 16)).astype(int)
    v = arr[idx].astype(np.float32)
    std = ndimage.gaussian_filter(v.std(0), 2)
    thr = max(std.max() * 0.25, 4.0)
    mask = std > thr
    if mask.sum() < 10:
        return None
    lab, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    big = lab == (1 + int(np.argmax(sizes)))
    ys, xs = np.where(big)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    side = max(y1 - y0, x1 - x0) * margin
    side = min(max(side, min_side * s), s)
    y0, y1 = cy - side / 2, cy + side / 2
    x0, x1 = cx - side / 2, cx + side / 2
    if y0 < 0:
        y1 -= y0; y0 = 0
    if x0 < 0:
        x1 -= x0; x0 = 0
    if y1 > s:
        y0 -= (y1 - s); y1 = s
    if x1 > s:
        x0 -= (x1 - s); x1 = s
    return [round(max(0, y0) / s, 4), round(y1 / s, 4),
            round(max(0, x0) / s, 4), round(x1 / s, 4)]


def run(mod: str, split: str) -> None:
    src = CACHE / f"{mod}_{split}"
    out = {}
    files = sorted(src.glob("*.npy"))
    for i, p in enumerate(files):
        bb = motion_bbox(np.load(p))
        if bb:
            out[p.stem] = bb
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)
    dst = META / f"bbox_{mod}_{split}.json"
    json.dump(out, open(dst, "w"))
    sides = [b[1] - b[0] for b in out.values()]
    print(f"{dst}: {len(out)}/{len(files)} boxes, median side "
          f"{np.median(sides):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mod", required=True)
    ap.add_argument("--split", required=True)
    a = ap.parse_args()
    run(a.mod, a.split)
