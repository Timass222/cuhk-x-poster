# -*- coding: utf-8 -*-
"""IMU v2 training (docs/imu_v2_arch.md, final configuration).

Data: data/cache/imu_v2_train.npy (N, 5, 16, 64) fixed 10 Hz grid, NaN outside the stream span;
      data/meta/clips_train.csv (action_id, user_id), data/meta/folds.csv (user_id -> fold).
Train: random 32-step window, device dropout (limbs .10, waist .05), tilt jitter +-10 deg per device,
       gaussian noise (acc .02 g, gyro 2 deg/s), gyro scale +-5%, TimeStretch [.8, 1.25] p .35,
       mixup alpha .4 p .5 on the feature tensor; balanced softmax + soft CE + label smoothing .1;
       Adam 5e-4 wd 3e-4, warmup 5 + cosine, EMA .995, select by val bal (min epoch 40).
Eval : windows with stride 16 averaged (softmax), EMA weights.
Out  : experiments/<out>/fold{k}.pth (EMA state incl. feature norm buffers), oof.npy, test.npy, test_f{k}.npy.
  python code/train/train_imu_v2.py --folds 0 1 2 3 4 --seeds 0 1 2 --epochs 120 --out experiments/imu_v2
"""
from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")
from models.branches.imu_v2 import IMUv2, IMUFeatures, WAIST, WIN, N_PHYS, count_params  # noqa: E402

K = 40
HZ = 10.0


# ---------------------------------------------------------------- data helpers
def load_train():
    X = np.load(ROOT / "data/cache/imu_v2_train.npy")                      # (N,5,16,64)
    clips = pd.read_csv(ROOT / "data/meta/clips_train.csv")
    y = clips["action_id"].to_numpy()
    fm = pd.read_csv(ROOT / "data/meta/folds.csv")
    u2f = dict(zip(fm["user_id"], fm["fold"]))
    fold = np.array([u2f[int(u)] for u in clips["user_id"]])
    return X, y, fold


def clip_meta(X):
    valid = ~np.isnan(X[:, :, 0, :])                                       # (N,5,T)
    present = valid.any(-1)                                                # (N,5)
    last = np.where(valid.any(1), np.arange(X.shape[-1])[None, :], -1).max(-1) + 1  # (N,) valid length in steps
    logdur = np.log(np.maximum(last, 1) / HZ)
    return valid, present, last, logdur.astype(np.float32)


def window_starts(L: int, stride: int = 16):
    if L <= WIN:
        return [0]
    s = list(range(0, L - WIN + 1, stride))
    if s[-1] != L - WIN:
        s.append(L - WIN)
    return s


def cut(X1, valid1, start):
    """(5,16,64),(5,64) -> window (5,16,32),(5,32) zero-padded."""
    xw = np.zeros((5, 16, WIN), np.float32)
    vw = np.zeros((5, WIN), bool)
    seg = X1[:, :, start:start + WIN]
    n = seg.shape[-1]
    xw[:, :, :n] = np.nan_to_num(seg, nan=0.0)
    vw[:, :n] = valid1[:, start:start + WIN]
    return xw, vw


# ---------------------------------------------------------------- augmentations (raw, numpy)
def tilt_jitter(x, valid, max_deg=10.0, rng=None):
    """q <- q (x) q_D, acc/gyro <- R_D^T v, per device (the sensor is 'worn slightly differently')."""
    for d in range(5):
        m = valid[d]
        if m.sum() == 0:
            continue
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis) + 1e-9
        ang = np.deg2rad(rng.uniform(-max_deg, max_deg))
        RD = Rot.from_rotvec(axis * ang)
        q = x[d, 12:16][:, m].T[:, [1, 2, 3, 0]]                             # scipy (x,y,z,w)
        qn = (Rot.from_quat(q) * RD).as_quat()[:, [3, 0, 1, 2]]
        x[d, 12:16][:, m] = qn.T
        x[d, 0:3][:, m] = RD.apply(x[d, 0:3][:, m].T, inverse=True).T
        x[d, 3:6][:, m] = RD.apply(x[d, 3:6][:, m].T, inverse=True).T
    return x


def time_stretch(x, valid, rate):
    """Resample the 64-grid by `rate` (>1 = faster): new length round(L/rate)."""
    T = x.shape[-1]
    out = np.full_like(x, np.nan); vo = np.zeros_like(valid)
    for d in range(5):
        idx = np.where(valid[d])[0]
        if len(idx) < 2:
            continue
        t0, t1 = idx[0], idx[-1]
        L = t1 - t0 + 1
        Ln = max(2, int(round(L / rate)))
        src = np.linspace(t0, t1, Ln)
        n = min(Ln, T - t0)
        for c in range(16):
            out[d, c, t0:t0 + n] = np.interp(src[:n], idx, x[d, c, idx])
        vo[d, t0:t0 + n] = True
        nrm = np.linalg.norm(out[d][12:16][:, t0:t0 + n], axis=0, keepdims=True)
        out[d][12:16][:, t0:t0 + n] /= np.where(nrm > 1e-6, nrm, 1.0)
    return out, vo


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, X, valid, present, logdur, y, aug: dict, seed=0):
        self.X, self.valid, self.present, self.logdur, self.y, self.aug = X, valid, present, logdur, y, aug
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        rng = self.rng
        x = self.X[i].copy(); valid = self.valid[i].copy(); present = self.present[i].copy()
        if self.aug.get("stretch_p", 0) > 0 and rng.random() < self.aug["stretch_p"]:
            x, valid = time_stretch(x, valid, rng.uniform(*self.aug["stretch_range"]))
        L = int(np.where(valid.any(0), np.arange(x.shape[-1]), -1).max() + 1)
        start = 0 if L <= WIN else int(rng.integers(0, L - WIN + 1))
        xw, vw = cut(x, valid, start)
        if self.aug.get("tilt_deg", 0) > 0:
            xw = tilt_jitter(xw, vw, self.aug["tilt_deg"], rng)
        if self.aug.get("noise", True):
            xw[:, 0:3] += rng.normal(0, 0.02, xw[:, 0:3].shape).astype(np.float32)
            xw[:, 3:6] += rng.normal(0, 2.0, xw[:, 3:6].shape).astype(np.float32)
            xw[:, 3:6] *= rng.uniform(0.95, 1.05)
        if self.aug.get("dev_drop", True):
            for d in range(5):
                p = 0.05 if d == WAIST else 0.10
                if present[d] and rng.random() < p:
                    present[d] = False; xw[d] = 0.0; vw[d] = False
        yv = np.zeros(K, np.float32); yv[self.y[i]] = 1.0
        return (torch.from_numpy(xw), torch.from_numpy(vw), torch.from_numpy(present.astype(np.float32)),
                torch.tensor(self.logdur[i]), torch.from_numpy(yv))


# ---------------------------------------------------------------- eval
@torch.no_grad()
def predict_clips(model, X, valid, present, logdur, last, dev, bs=256):
    model.eval()
    out = np.full((len(X), K), np.nan, np.float32)
    items = []
    for i in range(len(X)):
        if not present[i].any():
            continue
        for s in window_starts(int(last[i])):
            items.append((i, s))
    for b in range(0, len(items), bs):
        chunk = items[b:b + bs]
        xs, vs, ps, ls = [], [], [], []
        for i, s in chunk:
            xw, vw = cut(X[i], valid[i], s)
            xs.append(xw); vs.append(vw); ps.append(present[i].astype(np.float32)); ls.append(logdur[i])
        logits = model(torch.from_numpy(np.stack(xs)).to(dev), torch.from_numpy(np.stack(vs)).to(dev),
                       torch.from_numpy(np.stack(ps)).to(dev), torch.tensor(ls).to(dev))
        pr = torch.softmax(logits, 1).cpu().numpy()
        for (i, s), p in zip(chunk, pr):
            out[i] = p if np.isnan(out[i]).any() else out[i] + p
    cnt = np.zeros(len(X));
    for i, s in items:
        cnt[i] += 1
    ok = cnt > 0
    out[ok] /= cnt[ok, None]
    return out


def bal_acc(p, y):
    pr = p.argmax(1)
    return float(np.mean([(pr[y == c] == c).mean() for c in np.unique(y)])), float((pr == y).mean())


# ---------------------------------------------------------------- train one fold/seed
def fit_norm(model, X, valid, present, logdur, tr_idx, dev):
    feats, masks = [], []
    with torch.no_grad():
        for b in range(0, len(tr_idx), 256):
            idx = tr_idx[b:b + 256]
            x = torch.from_numpy(np.nan_to_num(X[idx], nan=0.0)).to(dev)
            f = IMUFeatures.physical(x)                                    # (B,5,21,64)
            m = (torch.from_numpy(valid[idx]).to(dev).float() * torch.from_numpy(present[idx]).to(dev).float()[:, :, None])
            feats.append(f.cpu()); masks.append(m.cpu())
    f = torch.cat(feats); m = torch.cat(masks)[:, :, None, :]
    s = (f * m).sum((0, 3)); n = m.sum((0, 3)).clamp_min(1)
    mean = s / n
    var = (((f - mean[None, :, :, None]) ** 2) * m).sum((0, 3)) / n
    model.features.mean.copy_(mean[:, :, None].to(dev))
    model.features.std.copy_((var.sqrt() + 1e-3)[:, :, None].to(dev))


def train_fold(args, X, y, fold, valid, present, last, logdur, k, seed, dev, Xte=None, te_meta=None):
    torch.manual_seed(seed); np.random.seed(seed)
    usable = present.any(1)
    tr = np.where((fold != k) & usable)[0]; va = np.where((fold == k) & usable)[0]
    model = IMUv2(stem=tuple(args.stem), temp=args.temp, head=args.head, drop=args.drop, shared_stem=not args.sep_stems).to(dev)
    fit_norm(model, X, valid, present, logdur, tr, dev)
    prior = torch.log(torch.tensor(np.bincount(y[tr], minlength=K) / len(tr) + 1e-6, dtype=torch.float32, device=dev))
    aug = dict(stretch_p=0.0 if args.no_stretch else 0.35, stretch_range=(0.8, 1.25), tilt_deg=0.0 if args.no_tilt else 10.0,
               noise=True, dev_drop=not args.no_devdrop)
    ds = TrainSet(X[tr], valid[tr], present[tr], logdur[tr], y[tr], aug, seed=seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * len(dl); warm = args.warmup * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    best, best_state, best_ep, hist = -1.0, None, -1, []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tl, tn = 0.0, 0
        for xw, vw, pw, lw, yv in dl:
            xw, vw, pw, lw, yv = xw.to(dev), vw.to(dev), pw.to(dev), lw.to(dev), yv.to(dev)
            mix = None
            if not args.no_mixup and np.random.rand() < 0.5:
                lam = float(np.random.beta(0.4, 0.4)); perm = torch.randperm(len(xw), device=dev)
                mix = (lam, perm); yv = lam * yv + (1 - lam) * yv[perm]
            logits = forward_mix(model, xw, vw, pw, lw, mix)
            target = (1 - args.ls) * yv + args.ls / K
            loss = -(target * F.log_softmax(logits + prior, 1)).sum(1).mean()
            tl += float(loss) * len(xw); tn += len(xw)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(args.ema).add_(pm.detach(), alpha=1 - args.ema)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
        pv = predict_clips(ema, X[va], valid[va], present[va], logdur[va], last[va], dev)
        okv = ~np.isnan(pv).any(1)
        bal, acc = bal_acc(pv[okv], y[va][okv]); hist.append(bal)
        if ep >= args.min_epoch and bal > best:
            best, best_ep, best_state = bal, ep, copy.deepcopy(ema.state_dict())
        if ep % 10 == 0 or ep == args.epochs:
            print(f"    fold {k} seed {seed} ep {ep}: train loss {tl / max(tn, 1):.3f} | val bal {bal:.4f} acc {acc:.4f} (best {best:.4f} @ {best_ep}) {time.time() - t0:.0f}s", flush=True)
    last_bal = float(np.mean(hist[-5:]))
    print(f"  fold {k} seed {seed}: best {best:.4f} @ {best_ep} | last5 {last_bal:.4f} | best-last {best - last_bal:+.4f}", flush=True)
    ema.load_state_dict(best_state)
    pv = predict_clips(ema, X[va], valid[va], present[va], logdur[va], last[va], dev)
    pt = predict_clips(ema, *te_meta, dev) if Xte is not None else None
    return ema, va, pv, pt, best


def forward_mix(model, xw, vw, pw, lw, mix):
    f = model.features(xw, vw, pw, lw)
    tm = (vw.float() * pw[:, :, None]).amax(1)
    if mix is not None:
        lam, perm = mix
        f = lam * f + (1 - lam) * f[perm]
        tm = torch.maximum(tm, tm[perm])
    outs = [model.stems[0 if model.shared_stem else d](f[:, d]) for d in range(5)]
    h = model.temporal(torch.cat(outs, 1))
    tm2 = F.max_pool1d(tm[:, None], 2).squeeze(1)
    return model.head(model.pool(h, tm2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--min-epoch", type=int, default=40)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=3e-4)
    ap.add_argument("--ema", type=float, default=0.995)
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--stem", type=int, nargs=2, default=[32, 64])
    ap.add_argument("--temp", type=int, default=160)
    ap.add_argument("--head", type=int, default=128)
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--sep-stems", action="store_true")
    ap.add_argument("--no-mixup", action="store_true")
    ap.add_argument("--no-stretch", action="store_true")
    ap.add_argument("--no-tilt", action="store_true")
    ap.add_argument("--no-devdrop", action="store_true")
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--out", default="experiments/imu_v2")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, fold = load_train()
    valid, present, last, logdur = clip_meta(X)
    print(f"train {len(X)} clips, usable {present.any(1).sum()} | device {dev} | params {count_params(IMUv2(stem=tuple(args.stem), temp=args.temp, head=args.head, shared_stem=not args.sep_stems))}", flush=True)
    Xte = te_meta = None
    if not args.no_test:
        Xte = np.load(ROOT / "data/cache/imu_v2_test.npy")
        vt, pt_, lt, ldt = clip_meta(Xte)
        te_meta = (Xte, vt, pt_, ldt, lt)
    out = ROOT / args.out; out.mkdir(parents=True, exist_ok=True)
    oof = np.full((len(X), K), np.nan, np.float32); oof_n = np.zeros(len(X))
    test = None; test_n = 0
    for k in args.folds:
        pt_f, best_seed, best_val = None, None, -1
        for seed in args.seeds:
            ema, va, pv, pt, best = train_fold(args, X, y, fold, valid, present, last, logdur, k, seed, dev, Xte, te_meta)
            ok = ~np.isnan(pv).any(1)
            oof[va[ok]] = np.where(np.isnan(oof[va[ok]]), 0, oof[va[ok]]) + pv[ok]; oof_n[va[ok]] += 1
            if pt is not None:
                pt_f = pt if pt_f is None else pt_f + pt
            if best > best_val:
                best_val, best_seed = best, seed
                torch.save(ema.state_dict(), out / f"fold{k}.pth")
        if pt_f is not None:
            pt_f = pt_f / len(args.seeds); np.save(out / f"test_f{k}.npy", pt_f)
            test = pt_f if test is None else test + pt_f; test_n += 1
        m = (fold == k) & (oof_n > 0)
        bal, acc = bal_acc(oof[m] / oof_n[m][:, None], y[m])
        print(f"fold {k}: OOF (среднее по сидам) bal {bal:.4f} acc {acc:.4f} | в пак сид {best_seed} (val bal {best_val:.4f})", flush=True)
    m = oof_n > 0
    oof[m] /= oof_n[m][:, None]
    np.save(out / "oof.npy", oof)
    bal, acc = bal_acc(oof[m], y[m])
    print(f"OOF: покрытие {m.sum()} acc {acc:.4f} bal {bal:.4f} -> {out / 'oof.npy'}", flush=True)
    if test is not None:
        np.save(out / "test.npy", test / test_n); print(f"test: {test_n} фолдов -> {out / 'test.npy'}", flush=True)


if __name__ == "__main__":
    main()
