"""IMU branch v2: temporal CNN on the raw resampled streams.

Input (5 devices x 9 channels, 32 time steps) -> flatten devices into 45
channels; NaN streams become zeros plus a 5-dim presence mask appended as
constant channels (total 50). 1D residual conv stack, ~0.4M params.
Augmentation: device dropout (matches real stream holes), time shift,
scale jitter, gaussian noise. Balanced softmax + label smoothing.

Writes the OOF contract: experiments/imu_tcn/{oof.npy,test.npy}.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

META = Path("data/meta")
CACHE = Path("data/cache")


def to_input(x: np.ndarray) -> np.ndarray:
    """(5, 9, 32) with NaN streams -> (50, 32) float32."""
    present = ~np.isnan(x).all(axis=(1, 2))            # (5,)
    filled = np.nan_to_num(x, nan=0.0)
    flat = filled.reshape(-1, x.shape[-1])             # (45, T)
    mask = np.repeat(present.astype(np.float32)[:, None], x.shape[-1], 1)
    return np.concatenate([flat, mask]).astype(np.float32)


class ImuDS(Dataset):
    def __init__(self, X, y, train):
        self.X, self.y, self.train = X, y, train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i].copy()                           # (50, 32)
        if self.train:
            rng = np.random
            if rng.rand() < 0.3:                       # device dropout
                d = rng.randint(0, 5)
                x[d * 9:(d + 1) * 9] = 0
                x[45 + d] = 0
            if rng.rand() < 0.5:                       # time shift
                x[:45] = np.roll(x[:45], rng.randint(-4, 5), axis=1)
            x[:45] *= 1 + (rng.rand() - 0.5) * 0.2     # scale jitter
            x[:45] += rng.normal(0, 0.02, x[:45].shape).astype(np.float32)
        return torch.from_numpy(x), int(self.y[i])


class Block(nn.Module):
    def __init__(self, a, b, s=1):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv1d(a, b, 5, s, 2, bias=False), nn.BatchNorm1d(b),
            nn.ReLU(inplace=True),
            nn.Conv1d(b, b, 5, 1, 2, bias=False), nn.BatchNorm1d(b))
        self.d = (nn.Sequential(nn.Conv1d(a, b, 1, s, bias=False),
                                nn.BatchNorm1d(b))
                  if (a != b or s != 1) else nn.Identity())

    def forward(self, x):
        return F.relu(self.c(x) + self.d(x))


class ImuTCN(nn.Module):
    def __init__(self, num_classes=40, w=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(50, w, 5, 1, 2, bias=False), nn.BatchNorm1d(w),
            nn.ReLU(inplace=True),
            Block(w, w), Block(w, 2 * w, 2), Block(2 * w, 2 * w),
            Block(2 * w, 4 * w, 2), nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Linear(4 * w, num_classes)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--out", default="experiments/imu_tcn")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = pd.read_csv(META / "clips_train.csv")
    folds = pd.read_csv(META / "folds.csv").set_index("user_id")["fold"]
    f = clips["user_id"].map(folds).to_numpy()
    y = clips["action_id"].to_numpy()
    raw = np.load(CACHE / "imu_train.npy")
    raw_t = np.load(CACHE / "imu_test.npy")
    has = ~np.isnan(raw).all(axis=(1, 2, 3))
    has_t = ~np.isnan(raw_t).all(axis=(1, 2, 3))
    X = np.stack([to_input(x) for x in raw])
    X_t = np.stack([to_input(x) for x in raw_t])
    # normalise per channel over train
    mu = X[has].mean((0, 2), keepdims=True)
    sd = X[has].std((0, 2), keepdims=True) + 1e-6
    X = (X - mu) / sd
    X_t = (X_t - mu) / sd
    print(f"imu clips: {has.sum()}/{len(y)}  test {has_t.sum()}/{len(X_t)}")

    prior = np.bincount(y, minlength=40) / len(y)
    prior_log = torch.log(torch.tensor(prior, dtype=torch.float32,
                                       device=device).clamp(1e-8))

    oof = np.full((len(y), 40), np.nan, np.float32)
    accs = []
    for k in sorted(np.unique(f).tolist()):
        tr = np.where((f != k) & has)[0]
        va = np.where((f == k) & has)[0]
        tr_dl = DataLoader(ImuDS(X[tr], y[tr], True), 128, shuffle=True,
                           drop_last=True)
        va_dl = DataLoader(ImuDS(X[va], y[va], False), 512)
        model = ImuTCN().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=2e-3, total_steps=args.epochs * len(tr_dl),
            pct_start=0.1)
        best_acc, best_state = 0.0, None
        for ep in range(args.epochs):
            model.train()
            for xb, yb in tr_dl:
                xb, yb = xb.to(device), yb.to(device)
                loss = F.cross_entropy(model(xb) + prior_log, yb,
                                       label_smoothing=0.1)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for xb, yb in va_dl:
                    correct += (model(xb.to(device)).argmax(1).cpu()
                                == yb).sum().item()
                    total += len(yb)
            acc = correct / total
            if acc > best_acc:
                best_acc = acc
                best_state = {k2: v.clone() for k2, v in
                              model.state_dict().items()}
        model.load_state_dict(best_state)
        accs.append(best_acc)
        print(f"fold {k}: best val {best_acc:.4f}", flush=True)
        with torch.no_grad():
            p = F.softmax(model(torch.from_numpy(X[va]).to(device)), 1)
            oof[va] = p.cpu().numpy()
        torch.save(model.state_dict(), out_dir / f"fold{k}.pth")

    np.save(out_dir / "oof.npy", oof)
    print("fold accs:", [round(a, 4) for a in accs],
          "mean %.4f" % float(np.mean(accs)))

    probs = np.zeros((len(X_t), 40), np.float32)
    for k in sorted(np.unique(f).tolist()):
        model = ImuTCN().to(device)
        model.load_state_dict(torch.load(out_dir / f"fold{k}.pth",
                                         weights_only=True))
        model.eval()
        with torch.no_grad():
            probs[has_t] += F.softmax(
                model(torch.from_numpy(X_t[has_t]).to(device)), 1).cpu().numpy()
    test = np.full((len(X_t), 40), np.nan, np.float32)
    test[has_t] = probs[has_t] / probs[has_t].sum(1, keepdims=True)
    np.save(out_dir / "test.npy", test)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
