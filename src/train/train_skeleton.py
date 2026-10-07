"""Train the ST-GCN skeleton branch on the frozen GroupKFold split.

Input pipeline (from data/cache/skeleton_train.npz):
  (T, 17, 3) [x, depth, height] (lifted 3D, H36M-17) -> NaN frames linearly interpolated ->
  resample T -> 48 -> channels [x, depth, height, dx, ddepth] -> (5, 48, 17)
Augmentation: mirror (x -> -x + left/right joint permutation), temporal
random crop-and-stretch, small rotation of xy, joint dropout.
Loss: cross-entropy + label smoothing 0.1 + balanced-softmax prior shift
(train prior spans 12..365 clips/class).

Writes the OOF contract: experiments/skeleton_stgcn/{oof.npy,test.npy}.
Run: .venv/Scripts/python.exe code/train/train_skeleton.py [--epochs 60]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.branches.skeleton_gcn import STGCN, FLIP_PERM, EDGES  # noqa: E402
from models.branches.skeleton_ctrgcn import CTRGCN  # noqa: E402


def make_model(arch, width):
    if arch == "stgcn":
        return STGCN(in_channels=5, width=width)
    if arch == "ctrgcn":
        return CTRGCN(in_channels=5, base=width)
    raise ValueError(arch)

# parent of each joint on the tree rooted at the pelvis (BFS over EDGES)
def _parents():
    import collections
    adj = collections.defaultdict(set)
    for a, b in EDGES:
        adj[a].add(b); adj[b].add(a)
    par = {0: 0}
    q = [0]
    while q:
        u = q.pop(0)
        for w in adj[u]:
            if w not in par:
                par[w] = u
                q.append(w)
    return [par[j] for j in range(17)]

PARENTS = _parents()
# первый ребёнок каждого сустава в дереве (для листьев — сам сустав)
CHILD = [next((b for a, b in EDGES if a == j), j) for j in range(17)]
TORSO = (0, 8)          # ось торса: таз -> грудь


def angle_features(out: np.ndarray) -> np.ndarray:
    """(T,17,3) [x, depth, height] -> (T,17,5): cos(кость к родителю, кость к ребёнку), cos(кость, ось торса),
    cos(кость, вертикаль), |кость|/|торс|, d/dt cos(родитель-ребёнок). Инвариантно к повороту вокруг вертикали и к зеркалу."""
    par = np.array(PARENTS); ch = np.array(CHILD)
    bp = out - out[:, par, :]                                   # кость к родителю (у таза 0)
    bp[:, 0, :] = out[:, 0, :] - out[:, 7, :]                   # таз: кость таз-позвоночник
    bc = out[:, ch, :] - out                                    # кость к первому ребёнку (у листьев 0)
    leaf = ch == np.arange(17)
    bc[:, leaf, :] = bp[:, leaf, :]
    # симметрия зеркала: у таза угол между двумя бёдрами, у груди — между двумя плечами (первый ребёнок зависит от стороны)
    bp_sym = bp.copy(); bc_sym = bc.copy()
    bp_sym[:, 0, :] = out[:, 1, :] - out[:, 0, :]; bc_sym[:, 0, :] = out[:, 4, :] - out[:, 0, :]
    bp_sym[:, 8, :] = out[:, 11, :] - out[:, 8, :]; bc_sym[:, 8, :] = out[:, 14, :] - out[:, 8, :]
    torso = out[:, TORSO[1], :] - out[:, TORSO[0], :]           # (T,3)
    up = np.zeros_like(torso); up[:, 2] = 1.0

    def cos(a, b):
        return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-6)
    f1 = cos(bp_sym, bc_sym)
    f2 = cos(bp, torso[:, None, :])
    f3 = cos(bp, up[:, None, :])
    f4 = np.linalg.norm(bp, axis=-1) / (np.linalg.norm(torso, axis=-1)[:, None] + 1e-6)
    f5 = np.diff(f1, axis=0, prepend=f1[:1])
    # у таза/груди кость к родителю для f2-f4: таз -> позвоночник, грудь -> позвоночник (симметричны)

    return np.stack([f1, f2, f3, f4, f5], -1).astype(np.float32)

META = Path("data/meta")
CACHE = Path("data/cache")
T_FIX = 48


def prep_clip(arr: np.ndarray, stream: str = "joint") -> np.ndarray:
    """(T,17,3) raw -> (5, T_FIX, 17) float32."""
    t = arr.shape[0]
    flat = arr.reshape(t, -1).astype(np.float64)
    bad = np.isnan(flat)
    if bad.any():
        idx = np.arange(t)
        for j in range(flat.shape[1]):
            col = flat[:, j]
            ok = ~np.isnan(col)
            flat[:, j] = (np.interp(idx, idx[ok], col[ok])
                          if ok.any() else 0.0)
    arr = flat.reshape(t, 17, 3)
    src = np.linspace(0, 1, t) if t > 1 else np.array([0.0])
    dst = np.linspace(0, 1, T_FIX)
    out = np.empty((T_FIX, 17, 3), np.float32)
    for j in range(17):
        for c in range(3):
            out[:, j, c] = np.interp(dst, src, arr[:, j, c])
    if stream == "angle":                     # угловой поток (UAV-Human): 5 инвариантных каналов на сустав
        return angle_features(out).transpose(2, 0, 1).astype(np.float32)
    if stream == "bone":                      # 3D bone vectors (H36M tree, root = pelvis)
        par = np.array(PARENTS)
        out = out - out[:, par, :]
    vel = np.diff(out[..., :2], axis=0, prepend=out[:1, :, :2])   # velocities of x, depth
    x = np.concatenate([out, vel], axis=2)          # (T,17,5)
    return x.transpose(2, 0, 1).astype(np.float32)  # (5,T,17)


class SkeletonDS(Dataset):
    def __init__(self, keys, arrays, labels, train, stream="joint"):
        self.keys, self.arrays, self.labels, self.train = keys, arrays, labels, train
        self.stream = stream

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        x = self.arrays[self.keys[i]].copy()        # (5,T,17)
        if self.train and self.stream == "angle":   # угловой поток: только перестановка суставов при зеркале, кроп-стретч, дропаут
            rng = np.random
            if rng.rand() < 0.5:
                x = x[:, :, FLIP_PERM]
            if rng.rand() < 0.3:
                lo = rng.randint(0, T_FIX // 4)
                hi = T_FIX - rng.randint(0, T_FIX // 4)
                idx = np.clip(np.linspace(lo, hi - 1, T_FIX).round().astype(int), 0, T_FIX - 1)
                x = x[:, idx]
            if rng.rand() < 0.2:
                drop = rng.choice(17, rng.randint(1, 3), replace=False)
                x[:, :, drop] = 0
            return torch.from_numpy(np.ascontiguousarray(x)), self.labels[i]
        if self.train:
            rng = np.random
            if rng.rand() < 0.5:                    # mirror
                x = x[:, :, FLIP_PERM]
                x[0] = -x[0]
                x[3] = -x[3]
            th = (rng.rand() - 0.5) * 0.5           # rotate xy
            c, s = math.cos(th), math.sin(th)
            xy = x[:2].copy()
            x[0], x[1] = c * xy[0] - s * xy[1], s * xy[0] + c * xy[1]
            dxy = x[3:5].copy()
            x[3], x[4] = c * dxy[0] - s * dxy[1], s * dxy[0] + c * dxy[1]
            if rng.rand() < 0.3:                    # temporal crop-stretch
                lo = rng.randint(0, T_FIX // 4)
                hi = T_FIX - rng.randint(0, T_FIX // 4)
                idx = np.clip(np.linspace(lo, hi - 1, T_FIX).round().astype(int),
                              0, T_FIX - 1)
                x = x[:, idx]
            if rng.rand() < 0.2:                    # joint dropout
                drop = rng.choice(17, rng.randint(1, 3), replace=False)
                x[:, :, drop] = 0
            x = x * (1 + (rng.rand() - 0.5) * 0.2)  # scale jitter
        return torch.from_numpy(np.ascontiguousarray(x)), self.labels[i]


def focal_loss(logits, y, gamma=2.0):
    logp = F.log_softmax(logits, 1).gather(1, y[:, None]).squeeze(1)
    return (-(1 - logp.exp()) ** gamma * logp).mean()


def train_fold(model, tr_dl, va_dl, prior_log, epochs, device, lr=1e-3,
               init_keys=None, init_mult=1.0, loss_kind="ce", focal_gamma=2.0):
    if init_keys and init_mult != 1.0:      # претрейн-параметры учатся медленнее (lr * mult)
        pre = [p for n, p in model.named_parameters() if n in init_keys]
        new = [p for n, p in model.named_parameters() if n not in init_keys]
        opt = torch.optim.AdamW([{"params": pre, "lr": lr * init_mult}, {"params": new, "lr": lr}],
                                weight_decay=1e-4)
        max_lr = [lr * init_mult, lr]
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        max_lr = lr
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=max_lr, total_steps=epochs * len(tr_dl), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    best_acc, best_state = 0.0, None
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                logits = model(xb) + prior_log      # balanced softmax
                loss = (focal_loss(logits.float(), yb, focal_gamma) if loss_kind == "focal"
                        else F.cross_entropy(logits, yb, label_smoothing=0.1))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in va_dl:
                out = model(xb.to(device))          # no prior shift at eval
                correct += (out.argmax(1).cpu() == yb).sum().item()
                total += len(yb)
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        if (ep + 1) % 10 == 0:
            print(f"    ep {ep+1}: val {acc:.4f} (best {best_acc:.4f})",
                  flush=True)
    model.load_state_dict(best_state)
    return best_acc


@torch.no_grad()
def predict(model, keys, arrays, device, bs=256):
    model.eval()
    out = []
    for i in range(0, len(keys), bs):
        xb = torch.stack([torch.from_numpy(arrays[k]) for k in keys[i:i + bs]])
        out.append(F.softmax(model(xb.to(device)), dim=1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--out", default="experiments/skeleton_stgcn")
    ap.add_argument("--stream", default="joint", choices=["joint", "bone", "angle"])
    ap.add_argument("--loss", default="ce", choices=["ce", "focal"], help="ce = CE+LS.1 (+balanced softmax); focal = focal γ (+balanced softmax)")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--arch", default="stgcn", choices=["stgcn", "ctrgcn"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--init-lr-mult", type=float, default=1.0, help="множитель lr для параметров из --init")
    ap.add_argument("--init", default=None,
                    help="checkpoint to initialise from (head reinitialised)")
    ap.add_argument("--folds", type=int, nargs="*", default=None,
                    help="subset of folds for screening; default all")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = pd.read_csv(META / "clips_train.csv")
    folds = pd.read_csv(META / "folds.csv").set_index("user_id")["fold"]
    f = clips["user_id"].map(folds).to_numpy()
    y_all = clips["action_id"].to_numpy()
    keys_all = (clips["action"] + "|" + clips["user"] + "|"
                + clips["trial"]).tolist()

    raw = np.load(CACHE / "skeleton_train.npz")
    print("preprocessing", len(raw.files), "skeleton clips ...", flush=True)
    arrays = {k: prep_clip(raw[k], args.stream) for k in raw.files}
    has = np.array([k in arrays for k in keys_all])
    print(f"skeleton available: {has.sum()}/{len(keys_all)}", flush=True)

    prior = np.bincount(y_all, minlength=40) / len(y_all)
    prior_log = torch.log(torch.tensor(prior, dtype=torch.float32,
                                       device=device).clamp(1e-8))

    oof = np.full((len(keys_all), 40), np.nan, np.float32)
    accs = []
    run_folds = (sorted(np.unique(f).tolist()) if args.folds is None
                 else args.folds)
    for k in run_folds:
        tr = np.where((f != k) & has)[0]
        va = np.where((f == k) & has)[0]
        print(f"fold {k}: train {len(tr)} val {len(va)}", flush=True)
        tr_ds = SkeletonDS([keys_all[i] for i in tr], arrays, y_all[tr], True, args.stream)
        va_ds = SkeletonDS([keys_all[i] for i in va], arrays, y_all[va], False, args.stream)
        tr_dl = DataLoader(tr_ds, 64, shuffle=True, drop_last=True)
        va_dl = DataLoader(va_ds, 256)
        model = make_model(args.arch, args.width).to(device)
        init_keys = None
        if args.init:
            ck = torch.load(args.init, map_location="cpu", weights_only=True)
            state = ck.get("state_dict", ck)
            state = {kk: v for kk, v in state.items()
                     if not kk.startswith("head.")}
            missing, unexpected = model.load_state_dict(state, strict=False)
            init_keys = set(state.keys())
            print(f"  init from {args.init}: "
                  f"missing {len(missing)} unexpected {len(unexpected)}",
                  flush=True)
        acc = train_fold(model, tr_dl, va_dl, prior_log, args.epochs, device,
                         lr=args.lr, init_keys=init_keys, init_mult=args.init_lr_mult,
                         loss_kind=args.loss, focal_gamma=args.focal_gamma)
        accs.append(acc)
        print(f"  fold {k} best val acc {acc:.4f}", flush=True)
        oof[va] = predict(model, [keys_all[i] for i in va], arrays, device)
        torch.save(model.state_dict(), out_dir / f"fold{k}.pth")

    np.save(out_dir / "oof.npy", oof)
    print("fold accs:", [round(a, 4) for a in accs],
          "mean %.4f" % float(np.mean(accs)), flush=True)

    # test predictions: mean of the 5 fold models
    test_raw = np.load(CACHE / "skeleton_test.npz")
    t_keys = sorted(pd.read_csv(META / "clips_test.csv")["clip"])
    t_arr = {k: prep_clip(test_raw[k], args.stream) for k in test_raw.files}
    have = [c for c in t_keys if c in t_arr]
    pos = {c: i for i, c in enumerate(t_keys)}
    probs = np.zeros((len(t_keys), 40), np.float32)
    n_models = 0
    for k in run_folds:
        model = make_model(args.arch, args.width).to(device)
        model.load_state_dict(torch.load(out_dir / f"fold{k}.pth",
                                         weights_only=True))
        p = predict(model, have, t_arr, device)
        for c, row in zip(have, p):
            probs[pos[c]] += row
        n_models += 1
    test = np.full((len(t_keys), 40), np.nan, np.float32)
    scored = probs.sum(1) > 0
    test[scored] = probs[scored] / probs[scored].sum(1, keepdims=True)
    np.save(out_dir / "test.npy", test)
    print(f"wrote {out_dir}/oof.npy test.npy", flush=True)


if __name__ == "__main__":
    main()
