# -*- coding: utf-8 -*-
"""IMU HGB v2: gradient boosting on clip statistics of the v2 physical features (same 21 channels per device
as the IMU v2 network: gravity-free acceleration, tilt, gyro, magnitudes, jerk, world vertical/horizontal
linear acceleration, orientation relative to the waist), computed on the fixed 10 Hz grid over the valid span.

Features per clip: 5 devices x 21 channels x 10 stats (mean, std, min, max, energy of the value; mean, std,
mean|.|, min, max of the first difference) = 1050, + present flags 5 + valid length + log-duration = 1057.
Absent device -> NaN block (HGB handles NaN natively). Same OOF contract as imu_stats / imu_hgb_*:
  experiments/<out>/oof.npy (N, 40) NaN rows = no IMU; test.npy (405, 40) from the full refit; model_full.pkl.
  python code/train/train_imu_hgb_v2.py --tag c150l15 [--feat core_yaw] [--stats 0 1 4] [--devs 0 1 2]
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")
from models.branches.imu_v2 import IMUFeatures  # noqa: E402

HZ = 10.0
CFGS = {
    "c150l15": dict(max_iter=150, max_leaf_nodes=15, learning_rate=0.15),
    "c100l8": dict(max_iter=100, max_leaf_nodes=8, learning_rate=0.2),
    "orig400": dict(max_iter=400, max_leaf_nodes=31, learning_rate=0.08),
}


def clip_features(X: np.ndarray) -> np.ndarray:
    """(N, 5, 16, 64) -> (N, 1057) float32 with NaN for absent devices."""
    N = len(X)
    valid = ~np.isnan(X[:, :, 0, :])                       # (N,5,64)
    present = valid.any(-1)
    last = np.where(valid.any(1), np.arange(X.shape[-1])[None, :], -1).max(-1) + 1
    out = np.full((N, 5 * 21 * 10 + 7), np.nan, np.float32)
    with torch.no_grad():
        for b in range(0, N, 256):
            xb = torch.from_numpy(np.nan_to_num(X[b:b + 256], nan=0.0))
            f = IMUFeatures.physical(xb).numpy()          # (B,5,21,64)
            for i in range(len(f)):
                feats = []
                for d in range(5):
                    m = valid[b + i, d]
                    if m.sum() < 2:
                        feats.append(np.full(21 * 10, np.nan, np.float32)); continue
                    v = f[i, d][:, m]                       # (21, L)
                    dv = np.diff(v, axis=1)
                    st = [v.mean(1), v.std(1), v.min(1), v.max(1), (v ** 2).mean(1),
                          dv.mean(1), dv.std(1), np.abs(dv).mean(1), dv.min(1), dv.max(1)]
                    feats.append(np.stack(st, 1).reshape(-1))
                row = np.concatenate(feats + [present[b + i].astype(np.float32), [last[b + i], np.log(max(last[b + i], 1) / HZ)]])
                out[b + i] = row
    return out


def yaw_features(X: np.ndarray) -> np.ndarray:
    """(N,5,16,64) -> (N, 20): mean/std of sin(yaw) and cos(yaw) per device (absolute heading = room geometry)."""
    yaw = np.deg2rad(X[:, :, 8, :])
    with np.errstate(all="ignore"):
        return np.stack([np.nanmean(np.sin(yaw), -1), np.nanstd(np.sin(yaw), -1), np.nanmean(np.cos(yaw), -1), np.nanstd(np.cos(yaw), -1)], -1).reshape(len(X), -1).astype(np.float32)


def select_features(F: np.ndarray, Xraw: np.ndarray, feat: str, stats, devs) -> np.ndarray:
    """feat: all (1057) | core (lin_s, tilt, gyro = 9 channels) | core_yaw (core + yaw sin/cos stats). stats/devs = index lists or None."""
    N = len(F); body = F[:, :-7].reshape(N, 5, 21, 10); extra = F[:, -7:]
    ch = list(range(21)) if feat == "all" else list(range(9))
    st = list(range(10)) if stats is None else list(stats); dv = list(range(5)) if devs is None else list(devs)
    parts = [body[:, dv][:, :, ch][:, :, :, st].reshape(N, -1)]
    if feat == "core_yaw":
        parts.append(yaw_features(Xraw)[:, [4 * d + j for d in dv for j in range(4)]])
    parts.append(extra)
    return np.concatenate(parts, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="c150l15", choices=list(CFGS))
    ap.add_argument("--feat", default="all", choices=["all", "core", "core_yaw"])
    ap.add_argument("--stats", type=int, nargs="*", default=None, help="индексы статистик 0..9 (mean,std,min,max,energy,dmean,dstd,dabs,dmin,dmax)")
    ap.add_argument("--devs", type=int, nargs="*", default=None, help="индексы датчиков 0..4 (LA,RA,C,LL,RL)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = ROOT / (args.out or f"experiments/imu_hgb_v2_{args.tag}"); out.mkdir(parents=True, exist_ok=True)
    clips = pd.read_csv(ROOT / "data/meta/clips_train.csv"); y = clips["action_id"].to_numpy()
    fm = pd.read_csv(ROOT / "data/meta/folds.csv"); u2f = dict(zip(fm["user_id"], fm["fold"])); fold = np.array([u2f[int(u)] for u in clips["user_id"]])
    t0 = time.time()
    Xr = np.load(ROOT / "data/cache/imu_v2_train.npy"); Xrt = np.load(ROOT / "data/cache/imu_v2_test.npy")
    X = select_features(clip_features(Xr), Xr, args.feat, args.stats, args.devs)
    Xt = select_features(clip_features(Xrt), Xrt, args.feat, args.stats, args.devs)
    print(f"признаки: {args.feat} stats={args.stats} devs={args.devs}", flush=True)
    np.save(out / "feat_train.npy", X); np.save(out / "feat_test.npy", Xt)
    has = ~np.isnan(X[:, -7:-2]).all(1) & (X[:, -7:-2].sum(1) > 0)
    print(f"признаки {X.shape} за {time.time() - t0:.0f}s | клипов с IMU {has.sum()} | тест {Xt.shape}", flush=True)
    cfg = CFGS[args.tag]
    oof = np.full((len(y), 40), np.nan, np.float32)
    for k in range(5):
        tr = has & (fold != k); te = has & (fold == k)
        clf = HistGradientBoostingClassifier(l2_regularization=1.0, random_state=42, **cfg).fit(X[tr], y[tr])
        prob = clf.predict_proba(X[te])
        oof[np.where(te)[0][:, None], clf.classes_[None, :]] = prob.astype(np.float32)
        pr = oof[te].argmax(1); yy = y[te]
        print(f"  fold {k}: acc {(pr == yy).mean():.4f} bal {np.mean([(pr[yy == c] == c).mean() for c in np.unique(yy)]):.4f} ({time.time() - t0:.0f}s)", flush=True)
    m = has; pr = oof[m].argmax(1); yy = y[m]
    print(f"OOF: покрытие {m.sum()} acc {(pr == yy).mean():.4f} bal {np.mean([(pr[yy == c] == c).mean() for c in np.unique(yy)]):.4f}", flush=True)
    np.save(out / "oof.npy", oof)
    clf = HistGradientBoostingClassifier(l2_regularization=1.0, random_state=42, **cfg).fit(X[has], y[has])
    has_t = ~np.isnan(Xt[:, -7:-2]).all(1) & (Xt[:, -7:-2].sum(1) > 0)
    test = np.full((len(Xt), 40), np.nan, np.float32)
    test[np.where(has_t)[0][:, None], clf.classes_[None, :]] = clf.predict_proba(Xt[has_t]).astype(np.float32)
    np.save(out / "test.npy", test)
    with open(out / "model_full.pkl", "wb") as fh:
        pickle.dump(clf, fh)
    import gzip
    print(f"test: {has_t.sum()}/{len(Xt)} | model_full.pkl gz {len(gzip.compress(pickle.dumps(clf), 9)) / 1048576:.2f} МиБ -> {out}", flush=True)


if __name__ == "__main__":
    main()
