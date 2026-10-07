# -*- coding: utf-8 -*-
"""OOF + test predictor for trained video branches with arbitrary
checkpoint sources, input mods and optional weight quantization.
  --ckpt DIR:f,f,...   fold checkpoints (repeatable), e.g.
        --ckpt video_r34_crop5:0,2 --ckpt r34c134:1,3,4
  --mods M1 M2         cache mods (train/test dirs)
  --bits N             per-output-channel symmetric int-N on conv/linear
                       weights (dequantized in fp32) -> size estimate
  --tta                6-view test TTA (roll -1/0/+1 x hflip)
Writes <out>/oof.npy, <out>/test.npy or test_tta.npy."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.stdout.reconfigure(encoding="utf-8")
torch.backends.cudnn.benchmark = True
from train.train_video import VideoDS, make_model  # noqa: E402
import train.train_video as tv  # noqa: E402

import os  # noqa: E402
ROOT = Path(__file__).resolve().parents[2]
META = Path(os.environ.get("CUHKX_META", ROOT / "data/meta"))
CACHE = Path(os.environ.get("CUHKX_CACHE", ROOT / "data/cache"))


class RollDS(VideoDS):
    crop = 112  # overridden after construction (self.crop from VideoDS when given)
    roll = 0
    shift = (0, 0)      # пространственный сдвиг окна кропа (dy, dx) в пикселях, TTA

    def load_clip(self, key):
        clip = super().load_clip(key)
        if clip is not None and self.roll:
            clip = np.roll(clip, self.roll, axis=0)
        return clip

    def __getitem__(self, i):
        if self.shift == (0, 0):
            return super().__getitem__(i)
        clip = self.load_clip(self.keys[i])
        t, c, s, _ = clip.shape
        C = self.crop
        off = (s - C) // 2
        top = min(max(off + self.shift[0], 0), s - C)
        left = min(max(off + self.shift[1], 0), s - C)
        clip = clip[:, :, top:top + C, left:left + C]
        x = torch.from_numpy(np.ascontiguousarray(clip)).float() / 255.0
        return x - 0.5, self.labels[i]


QCLIPS = [1.0, 0.97, 0.94, 0.9, 0.86, 0.82, 0.78, 0.74, 0.7]


def _q_rows(flat, qmax, clip_search):
    amax = flat.abs().max(1).values.clamp_min(1e-8)
    best_err, best = None, None
    for c in (QCLIPS if clip_search else [1.0]):
        scale = amax * c / qmax
        deq = torch.round(flat / scale[:, None]).clamp(-qmax, qmax) * scale[:, None]
        err = ((deq - flat) ** 2).sum(1)
        if best is None:
            best_err, best = err, deq
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err); best = torch.where(better[:, None], deq, best)
    return best


def quantize_sd(sd, bits, group=0, clip_search=False):
    """Per-out-channel symmetric int-N for weights with ndim>=2 (group>0: групповые масштабы по входной оси,
    clip_search: MSE-оптимальный клиппинг); returns dequantized sd and byte estimate."""
    qmax = 2 ** (bits - 1) - 1
    nbytes, n_q = 0, 0
    out = {}
    for k, v in sd.items():
        if v.ndim >= 2 and k.endswith("weight") and v.dtype == torch.float32:
            w = v.float()
            flat = w.reshape(w.shape[0], -1)
            if group and flat.shape[1] % group == 0 and flat.shape[1] >= group:
                deq = _q_rows(flat.reshape(-1, group), qmax, clip_search).reshape(w.shape)
                n_scale = flat.numel() // group
            else:
                deq = _q_rows(flat, qmax, clip_search).reshape(w.shape)
                n_scale = flat.shape[0]
            out[k] = deq
            nbytes += flat.numel() * bits / 8 + n_scale * (1 if group else 4)
            n_q += flat.numel()
        else:
            out[k] = v
            nbytes += v.numel() * v.element_size()
    return out, nbytes / 1024 ** 2, n_q


@torch.no_grad()
def predict(model, ds, device, hflip=False, bs=16):
    model.eval()
    dl = DataLoader(ds, bs, num_workers=2)
    res = []
    for xb, _ in dl:
        xb = xb.to(device)
        if hflip:
            xb = torch.flip(xb, dims=[-1])
        with torch.amp.autocast("cuda", enabled=device == "cuda"):
            res.append(F.softmax(model(xb).float(), 1).cpu().numpy())
    return np.concatenate(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--mods", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="r2p1d")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--bits", type=int, default=0)
    ap.add_argument("--qgroup", type=int, default=0, help="групповые масштабы квантования (0 = per-row)")
    ap.add_argument("--qclip", action="store_true", help="MSE-оптимальный клиппинг при квантовании")
    ap.add_argument("--crop", type=int, default=112, help="пространственный кроп из кэша (ViT: 224)")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--no-flip", action="store_true", help="TTA только сдвигами (без hflip)")
    ap.add_argument("--no-roll", action="store_true", help="TTA только отражением (без сдвигов)")
    ap.add_argument("--oof-tta", action="store_true", help="считать OOF теми же TTA-видами, что и тест (честный гейт TTA)")
    ap.add_argument("--tta-shift", type=int, default=0, help="доп. TTA: 4 диагональных сдвига окна кропа на ±N px (0 = выкл)")
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--feat-out", default=None, help="каталог для fc_norm-признаков: feat_oof.npy (N x D, каждая строка от модели своего фолда), feat_test_f{k}.npy")
    args = ap.parse_args()
    tv.N_FRAMES = args.frames
    tv.CROP = args.crop
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    ck = {}
    for spec in args.ckpt:
        d, fs = spec.split(":")
        for k in fs.split(","):
            ck[int(k)] = ROOT / "experiments" / d / f"fold{k}.pth"

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
    print(f"mods {args.mods} widths {widths} in_ch {in_ch} | folds "
          f"{sorted(ck)} | bits {args.bits or 'fp32'}", flush=True)

    t_keys = sorted(pd.read_csv(META / "clips_test.csv")["clip"])
    has_t = [k for k in t_keys if any((d / (k + ".npy")).exists()
                                      for d in te_dirs)]
    pos = {c: i for i, c in enumerate(t_keys)}
    acc_t = np.zeros((len(t_keys), 40), np.float64)
    n_views = 0

    oof = np.full((len(keys_all), 40), np.nan, np.float32)
    FEATS = []
    feat_oof = None
    for k in sorted(ck):
        model = make_model(args.arch, in_ch, 32, args.frames, pretrained=False).to(device)
        if args.feat_out:
            Path(ROOT / args.feat_out).mkdir(parents=True, exist_ok=True)
            model.fc_norm.register_forward_hook(lambda m, i, o: FEATS.append(o.detach().float().cpu().numpy()))
        sd = torch.load(ck[k], weights_only=True, map_location=device)
        if args.bits:
            if args.qgroup or args.qclip:
                from pack_checkpoint import pack_sd, unpack_sd
                packed = pack_sd({k: v.cpu() for k, v in sd.items()}, args.bits, args.qgroup, args.qclip)
                mb = sum(a.nbytes for a in packed.values() if hasattr(a, "nbytes")) / 2 ** 20
                sd = {k: v.to(device) for k, v in unpack_sd(packed).items()}; nq = 0
            else:
                sd, mb, nq = quantize_sd(sd, args.bits)
            print(f"  fold {k}: int{args.bits} -> {mb:.1f} МБ "
                  f"({nq / 1e6:.1f}M квантованных параметров)", flush=True)
        model.load_state_dict(sd)
        va = np.where((f == k) & has)[0]
        va_ds = VideoDS([keys_all[i] for i in va], y_all[va], tr_dirs,
                        False, None, widths)
        if args.oof_tta:
            # OOF теми же видами, что и тест (честный гейт TTA)
            d0 = args.tta_shift
            sh0 = [(0, 0)] + ([(-d0, -d0), (-d0, d0), (d0, -d0), (d0, d0)] if d0 else [])
            acc_v, nv = np.zeros((len(va), 40), np.float64), 0
            for roll in ((-1, 0, 1) if (args.tta and not args.no_roll) else (0,)):
                vds = RollDS([keys_all[i] for i in va], y_all[va], tr_dirs, False, None, widths)
                vds.roll = roll
                for sh in sh0:
                    vds.shift = sh
                    for hf in ((False, True) if (args.tta and not args.no_flip) else (False,)):
                        acc_v += predict(model, vds, device, hf); nv += 1
            p = (acc_v / nv).astype(np.float32)
            print(f"  fold {k}: OOF по {nv} видам", flush=True)
        else:
            FEATS.clear()
            p = predict(model, va_ds, device)
            if args.feat_out:
                fv = np.concatenate(FEATS)
                if feat_oof is None:
                    feat_oof = np.full((len(keys_all), fv.shape[1]), np.nan, np.float32)
                feat_oof[va] = fv
        oof[va] = p
        acc = (p.argmax(1) == y_all[va]).mean()
        bal = np.mean([(p.argmax(1)[y_all[va] == c] == c).mean()
                       for c in np.unique(y_all[va])])
        print(f"  fold {k}: val n={len(va)} acc {acc:.4f} bal {bal:.4f}",
              flush=True)
        if args.no_test:
            continue
        rolls = (-1, 0, 1) if (args.tta and not args.no_roll) else (0,)
        flips = (False, True) if (args.tta and not args.no_flip) else (False,)
        d = args.tta_shift
        shifts = [(0, 0)] + ([(-d, -d), (-d, d), (d, -d), (d, d)] if d else [])
        for roll in rolls:
            ds = RollDS(has_t, np.zeros(len(has_t), int), te_dirs, False,
                        None, widths)
            ds.roll = roll
            for sh in shifts:
                ds.shift = sh
                for hf in flips:
                    FEATS.clear()
                    pt = predict(model, ds, device, hf)
                    if args.feat_out and roll == 0 and sh == (0, 0) and not hf:
                        ft = np.full((len(t_keys), FEATS[0].shape[1]), np.nan, np.float32)
                        fv = np.concatenate(FEATS)
                        for c, row in zip(has_t, fv):
                            ft[pos[c]] = row
                        np.save(ROOT / args.feat_out / f"feat_test_f{k}.npy", ft)
                    if roll == 0 and sh == (0, 0) and not hf:
                        # 1-видовое предсказание этого фолда отдельно (для пер-фолдовых паков/калибровки)
                        tf = np.full((len(t_keys), 40), np.nan, np.float32)
                        for c, row in zip(has_t, pt):
                            tf[pos[c]] = row
                        np.save(out_dir / f"test_f{k}.npy", tf)
                    for c, row in zip(has_t, pt):
                        acc_t[pos[c]] += row
                    n_views += 1
        print(f"  fold {k}: test views done ({n_views})", flush=True)

    np.save(out_dir / "oof.npy", oof)
    if args.feat_out and feat_oof is not None:
        np.save(ROOT / args.feat_out / "feat_oof.npy", feat_oof)
        print(f"feat: OOF {(~np.isnan(feat_oof).any(1)).sum()} x {feat_oof.shape[1]} -> {ROOT / args.feat_out / 'feat_oof.npy'}", flush=True)
    ok = ~np.isnan(oof).any(1)
    pr = oof[ok].argmax(1)
    bal = np.mean([(pr[y_all[ok] == c] == c).mean()
                   for c in np.unique(y_all[ok])])
    print(f"OOF: покрытие {ok.sum()} acc {(pr == y_all[ok]).mean():.4f} "
          f"bal {bal:.4f} -> {out_dir / 'oof.npy'}", flush=True)
    if not args.no_test:
        test = np.full((len(t_keys), 40), np.nan, np.float32)
        sc = acc_t.sum(1) > 0
        test[sc] = (acc_t[sc] / acc_t[sc].sum(1, keepdims=True)).astype(
            np.float32)
        name = "test_tta.npy" if args.tta else "test.npy"
        np.save(out_dir / name, test)
        print(f"test: {n_views} видов, {sc.sum()}/{len(t_keys)} -> "
              f"{out_dir / name}", flush=True)


if __name__ == "__main__":
    main()
