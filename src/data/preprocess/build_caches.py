"""Materialise per-clip IMU tensors and skeleton sequences into data/cache/.

Sources: the test zip directly (no extraction needed), and the extracted
training tree data/extracted/HAR/data/. Outputs:

    data/cache/imu_test.npy        (405, 5, 9, 32) float32
    data/cache/imu_test_feat.npy   (405, 450)
    data/cache/skeleton_test.npz   clip_id -> (T, 17, 3)
    data/cache/imu_train.npy       (3036, 5, 9, 32) aligned to clips_train.csv
    data/cache/imu_train_feat.npy  (3036, 450)
    data/cache/skeleton_train.npz  "action|user|trial" -> (T, 17, 3)

Clips with no data at all stay NaN -- the OOF contract marks them unscored.
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from imu import clip_tensor, clip_tensor_v2, summary_features, DEVICES, N_CH, N_CH_V2, T_GRID, T_GRID_V2  # noqa
from frames import open_archive, is_junk  # noqa: E402
from skeleton import stack_clip  # noqa

import os
CACHE = Path(os.environ.get("CUHKX_CACHE", "data/cache"))
META = Path(os.environ.get("CUHKX_META", "data/meta"))


def build_test(zip_path: Path) -> None:
    z = open_archive(zip_path)
    names = [n for n in z.namelist() if not is_junk(n)]
    clips = sorted(pd.read_csv(META / "clips_test.csv")["clip"])
    base_of = {}
    for n in names:
        m = re.search(r"^(.*?)(?:^|/)([^/]+)/(?:IMU|Skeleton|Depth_Color|IR|Thermal|Radar)/", n)
        if m and m.group(2) not in base_of:
            base_of[m.group(2)] = (m.group(1).rstrip("/") + "/" if m.group(1) else "") + m.group(2)
    imu = np.full((len(clips), len(DEVICES), N_CH, T_GRID), np.nan, np.float32)
    imu2 = np.full((len(clips), len(DEVICES), N_CH_V2, T_GRID_V2), np.nan, np.float32)
    feats = np.full((len(clips), 450), np.nan, np.float32)
    sk = {}
    by_clip_sk = {}
    for n in names:
        m = re.search(r"(?:^|/)([^/]+)/Skeleton/predictions/[^/]*\.json$", n)
        if m:
            by_clip_sk.setdefault(m.group(1), []).append(n)
    for i, clip in enumerate(clips):
        base = base_of.get(clip, clip)
        def read(rel):
            try:
                return z.read(f"{base}/{rel}").decode("utf-8-sig", "replace")
            except KeyError:
                return None
        up_t, down_t = read("IMU/up(LA+RA+C).csv"), read("IMU/down(LL+RL).csv")
        x = clip_tensor(up_t, down_t)
        imu[i] = x
        imu2[i] = clip_tensor_v2(up_t, down_t)
        feats[i] = summary_features(x)
        sk_names = sorted(by_clip_sk.get(clip, []))
        if sk_names:
            sk[clip] = stack_clip([z.read(n) for n in sk_names])
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / "imu_test.npy", imu)
    np.save(CACHE / "imu_test_feat.npy", feats)
    np.save(CACHE / "imu_v2_test.npy", imu2)
    np.savez_compressed(CACHE / "skeleton_test.npz", **sk)
    total_streams = imu.shape[0] * len(DEVICES)
    nan_streams = int(np.isnan(imu).all(axis=(2, 3)).sum())
    print(f"test: {len(clips)} clips | imu NaN streams {nan_streams}/{total_streams} "
          f"| skeleton clips {len(sk)}")


def build_train(root: Path) -> None:
    clips = pd.read_csv(META / "clips_train.csv")
    n = len(clips)
    imu = np.full((n, len(DEVICES), N_CH, T_GRID), np.nan, np.float32)
    imu2 = np.full((n, len(DEVICES), N_CH_V2, T_GRID_V2), np.nan, np.float32)
    feats = np.full((n, 450), np.nan, np.float32)
    sk = {}
    for i, row in clips.iterrows():
        a, u, t = row["action"], row["user"], row["trial"]
        imu_dir = root / "IMU" / a / u / t
        def read(p: Path):
            return p.read_text(encoding="utf-8-sig", errors="replace") if p.is_file() else None
        up_t, down_t = read(imu_dir / "up(LA+RA+C).csv"), read(imu_dir / "down(LL+RL).csv")
        x = clip_tensor(up_t, down_t)
        imu[i] = x
        imu2[i] = clip_tensor_v2(up_t, down_t)
        feats[i] = summary_features(x)
        pred_dir = root / "Skeleton" / a / u / t / "predictions"
        if pred_dir.is_dir():
            files = sorted(pred_dir.glob("*.json"))
            if files:
                sk[f"{a}|{u}|{t}"] = stack_clip([f.read_bytes() for f in files])
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n}")
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / "imu_train.npy", imu)
    np.save(CACHE / "imu_train_feat.npy", feats)
    np.save(CACHE / "imu_v2_train.npy", imu2)
    np.savez_compressed(CACHE / "skeleton_train.npz", **sk)
    nan_streams = int(np.isnan(imu).all(axis=(2, 3)).sum())
    print(f"train: {n} clips | imu NaN streams {nan_streams}/{n*len(DEVICES)} "
          f"| skeleton clips {len(sk)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-zip", type=Path)
    ap.add_argument("--train-root", type=Path,
                    default=Path("data/extracted/HAR/data"))
    ap.add_argument("--train", action="store_true")
    args = ap.parse_args()
    if args.test_zip:
        build_test(args.test_zip)
    if args.train:
        build_train(args.train_root)
