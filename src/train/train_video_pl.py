"""Pseudo-label finetune of a video branch on confident fused test clips.

Legal check (findings 30 aug): only "API / LLM labeling" is banned; using our
own models' predictions as labels is standard self-training.

Takes an already-trained branch (--init <expdir> with fold{k}.pth), adds test
clips whose fused probability exceeds --pl-thresh as hard-labelled samples,
and finetunes each fold model briefly. Writes the usual OOF contract to --out
so the result can be scored/fused like any branch.

    python code/train/train_video_pl.py --mods Thermal --arch r2p1d \
        --init experiments/thermal_r2p1d --out experiments/thermal_r2p1d_pl \
        --pl-probs experiments/fusion_v9_test.npy --folds 0 2 \
        --epochs 6 --bs 16 --lr 5e-5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train.train_video as tv  # noqa: E402
from train.train_video import (VideoDS, make_model, train_fold,  # noqa: E402
                               predict)

META = Path("data/meta")
CACHE = Path("data/cache")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mods", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", required=True,
                    help="experiment dir with fold{k}.pth to start from")
    ap.add_argument("--pl-probs", required=True,
                    help="fused test probs npy, rows = sorted clips_test")
    ap.add_argument("--pl-thresh", type=float, default=0.75)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--arch", default="r2p1d")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 2])
    args = ap.parse_args()
    tv.N_FRAMES = args.frames
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = pd.read_csv(META / "clips_train.csv")
    folds = pd.read_csv(META / "folds.csv").set_index("user_id")["fold"]
    f = clips["user_id"].map(folds).to_numpy()
    y_all = clips["action_id"].to_numpy()
    keys_all = (clips["action"] + "~" + clips["user"] + "~"
                + clips["trial"]).tolist()
    tr_dirs = [CACHE / f"{m}_train" for m in args.mods]
    te_dirs = [CACHE / f"{m}_test" for m in args.mods]
    has = np.array([any((d / (k + ".npy")).exists() for d in tr_dirs)
                    for k in keys_all])
    widths = []
    for d in tr_dirs:
        probe = np.load(next(d.glob("*.npy")), mmap_mode="r")
        widths.append(3 if probe.ndim == 4 else 1)
    in_ch = sum(widths)

    # pseudo-labelled test clips
    t_keys = sorted(pd.read_csv(META / "clips_test.csv")["clip"])
    P = np.load(args.pl_probs)
    assert P.shape[0] == len(t_keys), P.shape
    conf = np.nan_to_num(P, nan=0.0).max(1)
    lab = np.nan_to_num(P, nan=0.0).argmax(1)
    has_te = np.array([any((d / (k + ".npy")).exists() for d in te_dirs)
                       for k in t_keys])
    sel = (conf >= args.pl_thresh) & has_te
    pl_keys = [k for k, s in zip(t_keys, sel) if s]
    pl_y = lab[sel]
    print(f"pseudo-labels: {sel.sum()}/{len(t_keys)} clips "
          f"(thresh {args.pl_thresh}, class spread "
          f"{np.bincount(pl_y, minlength=40).min()}.."
          f"{np.bincount(pl_y, minlength=40).max()})", flush=True)

    prior = np.bincount(y_all, minlength=40) / len(y_all)
    prior_log = torch.log(torch.tensor(prior, dtype=torch.float32,
                                       device=device).clamp(1e-8))

    oof = np.full((len(keys_all), 40), np.nan, np.float32)
    for k in args.folds:
        tr = np.where((f != k) & has)[0]
        va = np.where((f == k) & has)[0]
        tr_ds = VideoDS([keys_all[i] for i in tr], y_all[tr], tr_dirs, True,
                        None, widths)
        pl_ds = VideoDS(pl_keys, pl_y, te_dirs, True, None, widths)
        va_ds = VideoDS([keys_all[i] for i in va], y_all[va], tr_dirs, False,
                        None, widths)
        mix = ConcatDataset([tr_ds, pl_ds])
        print(f"fold {k}: train {len(tr)} + PL {len(pl_ds)}, val {len(va)}",
              flush=True)
        tr_dl = DataLoader(mix, args.bs, shuffle=True, drop_last=True,
                           num_workers=2, persistent_workers=True)
        va_dl = DataLoader(va_ds, args.bs, num_workers=2,
                           persistent_workers=True)
        model = make_model(args.arch, in_ch, 32, args.frames).to(device)
        model.load_state_dict(torch.load(
            Path(args.init) / f"fold{k}.pth", weights_only=True))
        bal = train_fold(model, tr_dl, va_dl, prior_log, args.epochs,
                         device, args.lr)
        print(f"  fold {k} best val bal {bal:.4f}", flush=True)
        oof[va] = predict(model, va_ds, device)
        torch.save(model.state_dict(), out_dir / f"fold{k}.pth")
    np.save(out_dir / "oof.npy", oof)

    # test: mean of finetuned fold models
    has_t = [k for k in t_keys if any((d / (k + ".npy")).exists()
                                      for d in te_dirs)]
    te_ds = VideoDS(has_t, np.zeros(len(has_t), int), te_dirs, False, None,
                    widths)
    pos = {c: i for i, c in enumerate(t_keys)}
    probs = np.zeros((len(t_keys), 40), np.float32)
    for k in args.folds:
        model = make_model(args.arch, in_ch, 32, args.frames).to(device)
        model.load_state_dict(torch.load(out_dir / f"fold{k}.pth",
                                         weights_only=True))
        p = predict(model, te_ds, device)
        for c, row in zip(has_t, p):
            probs[pos[c]] += row
    test = np.full((len(t_keys), 40), np.nan, np.float32)
    scored = probs.sum(1) > 0
    test[scored] = probs[scored] / probs[scored].sum(1, keepdims=True)
    np.save(out_dir / "test.npy", test)
    print(f"wrote {out_dir}/oof.npy test.npy", flush=True)


if __name__ == "__main__":
    main()
