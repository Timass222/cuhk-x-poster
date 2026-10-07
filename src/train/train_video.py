"""Train a video branch (Depth+IR joint, or Thermal) on the frozen folds.

Data: per-clip uint8 frame caches from preprocess/frames.py
(data/cache/<mod>_<split>/<key>.npy, (T, S, S)). Depth and IR are
frame-synchronous, so the joint branch stacks them as 2 channels; frames
where one is shorter are clamped to the shorter length. 16 frames are
sampled uniformly (train: jittered segment sampling, TSN-style).

Writes the OOF contract: experiments/<out>/{oof.npy,test.npy}.

    python code/train/train_video.py --mods Depth_Color IR --out experiments/video_dir
    python code/train/train_video.py --mods Thermal --out experiments/video_thermal
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

# fixed input shapes -> autotuned cudnn algos; the default heuristics pick
# pathological conv3d kernels on this build (measured 6.3x: 1114->176 ms/step)
torch.backends.cudnn.benchmark = True

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.branches.video_tsm import VideoTSM  # noqa: E402
from models.branches.framenet import FrameNet  # noqa: E402
from models.branches.video_tsm_r18 import VideoTSMR18  # noqa: E402
from models.branches.video_r2p1d import VideoR2P1D  # noqa: E402
from models.branches.video_r2p1d34 import VideoR2P1D34  # noqa: E402
from models.branches.video_x3d import VideoX3D  # noqa: E402
from models.branches.video_vit import VideoViT, param_groups_llrd, VideoInternViT, param_groups_llrd_iv2  # noqa: E402

META = Path("data/meta")
CACHE = Path("data/cache")
N_FRAMES = 16          # overridden by --frames (main process only: Windows workers re-import
                       # this module, so VideoDS keeps its own copies -- see VideoDS.__init__)
CROP = 112             # overridden by --crop
DENSE = 0              # --dense S: contiguous window of N_FRAMES frames with stride S (0 = TSN sampling)


def make_model(arch, in_channels, width, n_frames, pretrained=True,
               drop_path=0.1, grad_ckpt=False):
    if arch == "vit_s":
        return VideoViT(in_channels=in_channels, num_classes=40, n_frames=n_frames,
                        drop_path=drop_path, pretrained=pretrained, use_checkpoint=grad_ckpt)
    if arch == "vit_b":            # VideoMAEv2 ViT-B/16 distilled (K400 86.6), 16 frames tubelet 2
        return VideoViT(in_channels=in_channels, num_classes=40, n_frames=n_frames, dim=768, heads=12,
                        drop_path=drop_path, pretrained=pretrained, use_checkpoint=grad_ckpt,
                        weights="data/external/videomaev2/vit_b_k710_dl_from_giant.pth")
    if arch in ("iv2_s", "iv2_b", "iv2_s_ssv2"):   # InternVideo2 distilled S/14 or B/14 (K400 or SSv2 ft), 8 frames tubelet 1, patch 14
        return VideoInternViT(variant=arch, in_channels=in_channels, num_classes=40, n_frames=n_frames,
                              drop_path=drop_path, pretrained=pretrained, use_checkpoint=grad_ckpt)
    if arch == "umt_b":            # UMT-B/16 (CLIP teacher), 8 frames tubelet 1, ImageNet normalisation (UMT ft pipeline)
        return VideoViT(in_channels=in_channels, num_classes=40, n_frames=n_frames, dim=768, heads=12,
                        tubelet=1, drop_path=drop_path, pretrained=pretrained,
                        use_checkpoint=grad_ckpt, weights="data/external/umt/umt_b16_k400_f8.pth")
    if arch == "tsm":
        return VideoTSM(in_channels=in_channels, width=width,
                        n_frames=n_frames)
    if arch == "framenet":
        return FrameNet(in_channels=in_channels, width=width,
                        n_frames=n_frames)
    if arch == "tsm_r18":
        return VideoTSMR18(in_channels=in_channels, n_frames=n_frames)
    if arch == "r2p1d":
        return VideoR2P1D(in_channels=in_channels, n_frames=n_frames,
                          pretrained=pretrained)
    if arch == "r2p1d34":
        return VideoR2P1D34(in_channels=in_channels, n_frames=n_frames,
                            pretrained=pretrained)
    if arch == "x3d":
        return VideoX3D(in_channels=in_channels, n_frames=n_frames)
    raise ValueError(arch)


def sample_idx(t: int, n: int, jitter: bool) -> np.ndarray:
    """TSN segment sampling: n indices over t frames."""
    if t <= 0:
        return np.zeros(n, dtype=int)
    edges = np.linspace(0, t, n + 1)
    if jitter:
        pos = edges[:-1] + np.random.rand(n) * np.maximum(np.diff(edges), 1e-6)
    else:
        pos = (edges[:-1] + edges[1:]) / 2
    return np.clip(pos.astype(int), 0, t - 1)


class PrefetchLoader:
    """DataLoader(num_workers=0) + фоновый поток-префетчер: перекрытие чтения кэша и GPU без
    процессов-воркеров (Windows: spawn копирует память, shared-memory батчи -> ошибка 1455 / хэнг)."""

    def __init__(self, dl, depth=4):
        self.dl, self.depth = dl, depth

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        import queue
        import threading
        q = queue.Queue(self.depth)
        sentinel = object()

        def run():
            try:
                for item in self.dl:
                    q.put(item)
            except BaseException as e:  # noqa: BLE001
                q.put(e)
            q.put(sentinel)

        threading.Thread(target=run, daemon=True).start()
        while True:
            item = q.get()
            if item is sentinel:
                return
            if isinstance(item, BaseException):
                raise item
            yield item


class VideoDS(Dataset):
    def __init__(self, keys, labels, dirs, train: bool, ram=None, widths=None,
                 n_frames=None, crop=None, dense=None):
        self.keys, self.labels, self.dirs, self.train = keys, labels, dirs, train
        self.ram = ram          # optional {(dir_idx, key): ndarray} preload
        self.widths = widths or [1] * len(dirs)   # channels per dir (3 = RGB)
        # own copies: DataLoader workers on Windows re-import the module and
        # would otherwise see the default globals (F1 bug, 5 Sep)
        self.n_frames = N_FRAMES if n_frames is None else n_frames
        self.crop = CROP if crop is None else crop
        self.dense = DENSE if dense is None else dense
        self.mod_drop = MOD_DROP if train else None   # (dir_idx, p) — случайно гасить канал(ы) модальности
        self.zero_dirs = set()                        # всегда гасить эти модальности (оценка «без IR»)
        self.blur_p = BLUR_P if train else 0.0

    def __len__(self):
        return len(self.keys)

    def load_clip(self, key):
        chans = []
        t_min = None
        for di, d in enumerate(self.dirs):
            if self.ram is not None:
                arr = self.ram.get((di, key))
                if arr is not None:
                    t_min = arr.shape[0] if t_min is None else min(t_min, arr.shape[0])
                chans.append(arr)
                continue
            p = d / (key + ".npy")
            if p.exists():
                arr = np.load(p, mmap_mode="r")
                t_min = arr.shape[0] if t_min is None else min(t_min, arr.shape[0])
                chans.append(arr)
            else:
                chans.append(None)
        if t_min is None:
            return None
        n_fr, dense = self.n_frames, self.dense
        if dense > 0 and t_min >= n_fr * dense:
            span = n_fr * dense
            start = np.random.randint(0, t_min - span + 1) if self.train else (t_min - span) // 2
            idx = start + np.arange(n_fr) * dense
        else:
            idx = sample_idx(t_min, n_fr, self.train)
        s = chans[0].shape[1] if chans[0] is not None else next(
            c for c in chans if c is not None).shape[1]
        out = np.zeros((n_fr, sum(self.widths), s, s), np.uint8)
        off = 0
        for ci, arr in enumerate(chans):
            w = self.widths[ci]
            if arr is not None:
                a = arr[idx]
                if a.ndim == 4:               # (n, s, s, 3) RGB cache
                    out[:, off:off + w] = a.transpose(0, 3, 1, 2)
                else:
                    out[:, off] = a
            off += w
        kill = set(self.zero_dirs)
        if self.mod_drop is not None and np.random.rand() < self.mod_drop[1]:
            kill.add(self.mod_drop[0])
        if kill:
            offs = np.cumsum([0] + list(self.widths))
            for di in kill:
                out[:, offs[di]:offs[di + 1]] = 0
        return out

    def __getitem__(self, i):
        clip = self.load_clip(self.keys[i])           # (T, C, S, S) uint8
        t, c, s, _ = clip.shape
        if self.train:
            # random crop + horizontal flip
            top = np.random.randint(0, s - self.crop + 1)
            left = np.random.randint(0, s - self.crop + 1)
            clip = clip[:, :, top:top + self.crop, left:left + self.crop]
            if np.random.rand() < 0.5:
                clip = clip[:, :, :, ::-1]
        else:
            off = (s - self.crop) // 2
            clip = clip[:, :, off:off + self.crop, off:off + self.crop]
        x = torch.from_numpy(np.ascontiguousarray(clip)).float() / 255.0
        x = x - 0.5
        if self.blur_p > 0 and np.random.rand() < self.blur_p:
            sz = x.shape[-1]; lo = max(8, int(round(sz * 0.7)))
            x = F.interpolate(F.interpolate(x, size=(lo, lo), mode="bilinear", antialias=True, align_corners=False),
                              size=(sz, sz), mode="bilinear", align_corners=False)
        return x, self.labels[i]


def eval_balanced(model, va_dl, device):
    """(balanced_acc, plain_acc) on the val loader."""
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for xb, yb in va_dl:
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                out = model(xb.to(device))
            preds.append(out.argmax(1).cpu().numpy())
            ys.append(yb.numpy())
    preds, ys = np.concatenate(preds), np.concatenate(ys)
    recalls = [np.mean(preds[ys == c] == c) for c in np.unique(ys)]
    return float(np.mean(recalls)), float(np.mean(preds == ys))


MIXUP_ALPHA, MIXUP_P = 0.0, 0.0   # выставляется из main (--mixup ALPHA --mixup-p P)
MOD_DROP = None                    # (dir_idx, p): модальностный дропаут в трейне (--mod-dropout IDX P)
BLUR_P = 0.0                       # --blur-aug P: даунскейл x0.7 + апскейл (мягкость HAU-mp4 не должна быть признаком)


def soft_ce(logits, target, smoothing=0.1):
    """кросс-энтропия с мягкими целями (mixup) + label smoothing."""
    n = logits.shape[1]
    t = target * (1 - smoothing) + smoothing / n
    return -(t * F.log_softmax(logits.float(), 1)).sum(1).mean()


class FakeQuantRow(torch.nn.Module):
    """QAT: per-out-channel symметричный int-N (как pack_sd) со straight-through estimator."""
    def __init__(self, bits):
        super().__init__(); self.qmax = 2 ** (bits - 1) - 1

    def forward(self, w):
        flat = w.reshape(w.shape[0], -1)
        scale = flat.detach().abs().amax(1).clamp_min(1e-8) / self.qmax
        q = torch.round(flat / scale[:, None]).clamp(-self.qmax, self.qmax) * scale[:, None]
        return (flat + (q - flat).detach()).reshape(w.shape)


def apply_qat(model, bits):
    import torch.nn.utils.parametrize as P
    n = 0
    for mod in model.modules():
        w = getattr(mod, "weight", None)
        if isinstance(w, torch.nn.Parameter) and w.ndim >= 2 and w.dtype == torch.float32:
            P.register_parametrization(mod, "weight", FakeQuantRow(bits)); n += 1
    return n


def remove_qat(model):
    import torch.nn.utils.parametrize as P
    for mod in model.modules():
        if P.is_parametrized(mod, "weight"):
            P.remove_parametrizations(mod, "weight", leave_parametrized=False)


def train_fold(model, tr_dl, va_dl, prior_log, epochs, device, lr,
               ema_decay=0.999, wd=1e-4, layer_decay=1.0, select="best"):
    if isinstance(model, VideoInternViT) and layer_decay < 1.0:
        groups = param_groups_llrd_iv2(model, lr, wd, layer_decay)
        opt = torch.optim.AdamW(groups, lr=lr)
        max_lr = [g["lr"] for g in groups]
    elif isinstance(model, VideoViT) and layer_decay < 1.0:
        groups = param_groups_llrd(model, lr, wd, layer_decay)
        opt = torch.optim.AdamW(groups, lr=lr)
        max_lr = [g["lr"] for g in groups]
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        max_lr = lr
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=max_lr, total_steps=epochs * len(tr_dl), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # live views + fused updates: a per-key python loop is ~600 tiny CUDA
    # launches per step and starves the GPU (measured: 82W at "100%" util)
    live = model.state_dict()
    f_keys = [k for k, v in live.items() if v.dtype.is_floating_point]
    ema_f = [ema[k] for k in f_keys]
    live_f = [live[k] for k in f_keys]
    i_keys = [k for k in live if k not in set(f_keys)]
    best_bal, best_state, best_tag = 0.0, None, ""
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                if MIXUP_ALPHA > 0 and np.random.rand() < MIXUP_P:
                    lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
                    perm = torch.randperm(xb.shape[0], device=xb.device)
                    xm = lam * xb + (1 - lam) * xb[perm]
                    yo = F.one_hot(yb, 40).float()
                    tgt = lam * yo + (1 - lam) * yo[perm]
                    loss = soft_ce(model(xm) + prior_log, tgt)
                else:
                    loss = F.cross_entropy(model(xb) + prior_log, yb,
                                           label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            with torch.no_grad():
                torch._foreach_mul_(ema_f, ema_decay)
                torch._foreach_add_(ema_f, live_f, alpha=1 - ema_decay)
        with torch.no_grad():
            for k in i_keys:
                ema[k].copy_(live[k])
        bal, acc = eval_balanced(model, va_dl, device)
        raw_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema)
        bal_e, acc_e = eval_balanced(model, va_dl, device)
        model.load_state_dict(raw_state)
        for cand_bal, cand_state, tag in ((bal, raw_state, "raw"),
                                          (bal_e, ema, "ema")):
            if cand_bal > best_bal:
                best_bal = cand_bal
                best_state = {k: v.clone() for k, v in cand_state.items()}
                best_tag = tag
        print(f"    ep {ep+1}: bal {bal:.4f}/{bal_e:.4f} acc {acc:.4f}/{acc_e:.4f}"
              f" (best {best_bal:.4f} {best_tag})", flush=True)
    if select == "last":
        # чекпойнт по плато: финальная EMA вместо лучшей эпохи (ранний всплеск val = недоученная модель)
        model.load_state_dict(ema)
        print(f"    select=last: финальная EMA (bal {bal_e:.4f}), best был {best_bal:.4f} {best_tag}", flush=True)
        return bal_e
    model.load_state_dict(best_state)
    return best_bal


@torch.no_grad()
def predict(model, ds, device, bs=16):
    # bs=16: on 8GB cards bs>=24 loses cuDNN workspaces even in inference
    # (measured: "100% util" at ~82W = slow fallback conv3d algorithms)
    model.eval()
    dl = DataLoader(ds, bs)
    out = []
    for xb, _ in dl:
        with torch.amp.autocast("cuda", enabled=device == "cuda"):
            out.append(F.softmax(model(xb.to(device)).float(), 1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mods", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--select", default="best", choices=["best", "last"], help="чекпойнт: лучшая эпоха по val bal (best) или финальная EMA (last)")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--arch", default="tsm",
                    choices=["tsm", "framenet", "tsm_r18", "r2p1d", "r2p1d34", "x3d", "vit_s", "vit_b", "umt_b", "iv2_s", "iv2_b", "iv2_s_ssv2"])
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--layer-decay", type=float, default=1.0, help="LLRD для vit_s (0.75)")
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--prefetch", type=int, default=0, help="поток-префетчер при num-workers 0 (глубина очереди)")
    ap.add_argument("--cache-root", default=None, help="каталог кэшей (дефолт data/cache или env CUHKX_CACHE)")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--crop", type=int, default=112, help="spatial crop from the cache (cache 128 -> 112; 160 -> 128/144)")
    ap.add_argument("--dense", type=int, default=0, help="stride of a contiguous window (0 = TSN segments over the clip)")
    ap.add_argument("--preload", action="store_true")
    ap.add_argument("--folds", type=int, nargs="*", default=None,
                    help="subset of folds for screening; default all")
    ap.add_argument("--seed", type=int, default=0, help="сид torch/numpy (0 = как раньше: без явной фиксации)")
    ap.add_argument("--blur-aug", type=float, default=0.0, help="HAU: вероятность мягкого размытия трейн-клипов (внешние клипы не размываются)")
    ap.add_argument("--mod-dropout", type=float, nargs=2, default=None, metavar=("DIR_IDX", "P"), help="HAU: гасить модальность DIR_IDX (индекс в --mods) с вероятностью P в трейне; в конце — val без неё")
    ap.add_argument("--mixup", type=float, default=0.0, help="R2: mixup alpha (0 = выкл); мягкие цели + LS .1")
    ap.add_argument("--mixup-p", type=float, default=0.5, help="доля батчей с mixup")
    ap.add_argument("--drop-train-folds", type=int, nargs="*", default=[],
                    help="кривая обучения: исключить эти фолды из ТРЕЙНА (val-фолд не трогается)")
    ap.add_argument("--val-users", nargs="*", default=None,
                    help="валидация на перечисленных юзерах (user9 …), трейн = все остальные (минус --drop-train-folds); чекпойнт foldu.pth")
    ap.add_argument("--val-only-classes", type=int, nargs="*", default=None,
                    help="с --val-users: val = только эти классы val-юзеров; их остальные клипы уходят в ТРЕЙН")
    ap.add_argument("--init", default=None, help="стартовый чекпойнт (state_dict) вместо претрейна; с --no-pretrained")
    ap.add_argument("--pl-csv", default=None, help="самообучение: CSV clip,label — тест-клипы с псевдо-метками, добавляются в трейн (кэш *_test)")
    ap.add_argument("--extra-csv", default=None, help="HAU: CSV key,label[,weight] — внешние клипы (кэши <extra-root>/<mod>_extra/<key>.npy; нет файла модальности = нули)")
    ap.add_argument("--extra-root", default=None, help="каталог кэшей внешних клипов (дефолт = cache-root)")
    ap.add_argument("--qat-bits", type=int, default=0, help="QAT: fake-quant int-N per-row (как pack_sd) в forward; val = квантованная модель")
    ap.add_argument("--full", action="store_true",
                    help="полнотрейновая модель: трейн = все клипы, val = 128 клипов трейна (только для лога); чекпойнт fold_full.pth; use --select last")
    args = ap.parse_args()
    global MIXUP_ALPHA, MIXUP_P, MOD_DROP, BLUR_P
    BLUR_P = args.blur_aug
    if args.mod_dropout: MOD_DROP = (int(args.mod_dropout[0]), float(args.mod_dropout[1]))
    MIXUP_ALPHA, MIXUP_P = args.mixup, args.mixup_p
    if args.seed:
        torch.manual_seed(args.seed); np.random.seed(args.seed)
    global N_FRAMES, CROP, DENSE, CACHE
    N_FRAMES = args.frames
    CROP = args.crop
    DENSE = args.dense
    import os
    if args.cache_root or os.environ.get("CUHKX_CACHE"):
        CACHE = Path(args.cache_root or os.environ["CUHKX_CACHE"])
    print(f"cache root: {CACHE} | frames {N_FRAMES} crop {CROP} dense {DENSE}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("device:", device, "| mods:", args.mods, flush=True)

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
    print(f"clips with data: {has.sum()}/{len(keys_all)}", flush=True)

    widths = []
    for d in tr_dirs:
        probe = np.load(next(d.glob("*.npy")), mmap_mode="r")
        widths.append(3 if probe.ndim == 4 else 1)
    in_ch = sum(widths)
    print("channel widths:", widths, "-> in_channels", in_ch, flush=True)

    NW = 0 if args.preload else args.num_workers   # preload: без воркеров (Windows spawn копирует RAM в каждый)
    ram = None
    if args.preload:
        import time
        t0 = time.time()
        ram = {}
        for di, d in enumerate(tr_dirs + te_dirs):
            for p in d.glob("*.npy"):
                ram[(di % len(tr_dirs) if di < len(tr_dirs) else di - len(tr_dirs),
                     p.stem) if False else (di, p.stem)] = np.load(p)
        print(f"preloaded {len(ram)} arrays in {time.time()-t0:.0f}s", flush=True)

    prior = np.bincount(y_all, minlength=40) / len(y_all)
    prior_log = torch.log(torch.tensor(prior, dtype=torch.float32,
                                       device=device).clamp(1e-8))

    oof = np.full((len(keys_all), 40), np.nan, np.float32)
    accs = []
    run_folds = (sorted(np.unique(f).tolist()) if args.folds is None
                 else args.folds)
    if args.full:
        run_folds = ["full"]
    if args.val_users:
        run_folds = ["u"]
        is_val_user = clips["user"].isin(args.val_users).to_numpy()
    for k in run_folds:
        if k == "full":
            tr = np.where(has)[0]
            va = tr[np.linspace(0, len(tr) - 1, 128).astype(int)]     # in-sample, только для лога
        elif k == "u":
            if args.val_only_classes:
                in_val_cls = np.isin(y_all, args.val_only_classes)
                tr = np.where(((~is_val_user) | (~in_val_cls)) & has & ~np.isin(f, args.drop_train_folds))[0]
                va = np.where(is_val_user & in_val_cls & has)[0]
                print(f"val-only-classes {args.val_only_classes}: в трейн ушло {int((is_val_user & ~in_val_cls & has).sum())} клипов val-юзеров", flush=True)
            else:
                tr = np.where(~is_val_user & has & ~np.isin(f, args.drop_train_folds))[0]
                va = np.where(is_val_user & has)[0]
            print(f"val-users {args.val_users}: юзеров в трейне {clips['user'].iloc[tr].nunique()}", flush=True)
        else:
            tr = np.where((f != k) & has & ~np.isin(f, args.drop_train_folds))[0]
            va = np.where((f == k) & has)[0]
        print(f"fold {k}: train {len(tr)} val {len(va)}" + (f" (без фолдов {args.drop_train_folds} в трейне)" if args.drop_train_folds else ""), flush=True)
        tr_ram = None if ram is None else {(i, k): v for (i, k), v in ram.items()
                                           if i < len(tr_dirs)}
        tr_ds = VideoDS([keys_all[i] for i in tr], y_all[tr], tr_dirs, True,
                        tr_ram, widths)
        va_ds = VideoDS([keys_all[i] for i in va], y_all[va], tr_dirs, False,
                        tr_ram, widths)
        if args.extra_csv:
            ex = pd.read_csv(args.extra_csv)
            ex_root = Path(args.extra_root) if args.extra_root else CACHE
            ex_dirs = [ex_root / f"{m}_extra" for m in args.mods]
            ex_ds = VideoDS(ex["key"].tolist(), ex["label"].to_numpy(), ex_dirs, True, None, widths)
            ex_ds.blur_p = 0.0
            print(f"  EXTRA: +{len(ex_ds)} внешних клипов из {args.extra_csv} (кэши {ex_root}; классов {ex['label'].nunique()}, min/max на класс {ex['label'].value_counts().min()}/{ex['label'].value_counts().max()})", flush=True)
            tr_ds = ConcatDataset([tr_ds, ex_ds])
        if args.pl_csv:
            pl = pd.read_csv(args.pl_csv)
            pl = pl[[any((d / (c + ".npy")).exists() for d in te_dirs) for c in pl["clip"]]]
            te_ram_pl = None if ram is None else {(i - len(tr_dirs), kk): v for (i, kk), v in ram.items() if i >= len(tr_dirs)}
            pl_ds = VideoDS(pl["clip"].tolist(), pl["label"].to_numpy(), te_dirs, True, te_ram_pl, widths)
            print(f"  PL: +{len(pl_ds)} тест-клипов из {args.pl_csv} (классов {pl['label'].nunique()}, min/max на класс {pl['label'].value_counts().min()}/{pl['label'].value_counts().max()})", flush=True)
            tr_ds = ConcatDataset([tr_ds, pl_ds])
        tr_dl = DataLoader(tr_ds, args.bs, shuffle=True, drop_last=True,
                           num_workers=NW, persistent_workers=NW > 0)
        va_dl = DataLoader(va_ds, args.bs, num_workers=NW,
                           persistent_workers=NW > 0)
        if NW == 0 and args.prefetch > 0:
            tr_dl, va_dl = PrefetchLoader(tr_dl, args.prefetch), PrefetchLoader(va_dl, args.prefetch)
        model = make_model(args.arch, in_ch, args.width, N_FRAMES,
                           pretrained=not args.no_pretrained,
                           drop_path=args.drop_path,
                           grad_ckpt=args.grad_ckpt).to(device)
        if args.init:
            model.load_state_dict(torch.load(args.init, map_location=device, weights_only=True))
            print(f"  init <- {args.init}", flush=True)
        if args.qat_bits:
            print(f"  QAT int{args.qat_bits}: параметризовано матриц {apply_qat(model, args.qat_bits)}", flush=True)
        acc = train_fold(model, tr_dl, va_dl, prior_log, args.epochs,
                         device, args.lr, wd=args.wd,
                         layer_decay=args.layer_decay, select=args.select)
        accs.append(acc)
        print(f"  fold {k} best val acc {acc:.4f}", flush=True)
        if args.mod_dropout:
            va0 = VideoDS([keys_all[i] for i in va], y_all[va], tr_dirs, False, tr_ram, widths)
            va0.zero_dirs = {int(args.mod_dropout[0])}
            b0, a0 = eval_balanced(model, DataLoader(va0, args.bs, num_workers=0), device)
            print(f"  fold {k} val БЕЗ модальности {args.mods[int(args.mod_dropout[0])]}: bal {b0:.4f} acc {a0:.4f}", flush=True)
        oof[va] = predict(model, va_ds, device)
        if args.qat_bits:
            remove_qat(model)          # в чекпойнт — латентные веса; pack_sd квантует их тем же правилом
        torch.save(model.state_dict(), out_dir / f"fold{k}.pth")

    np.save(out_dir / "oof.npy", oof)
    print("fold accs:", [round(a, 4) for a in accs],
          "mean %.4f" % float(np.mean(accs)), flush=True)

    # test: mean of fold models
    t_keys = sorted(pd.read_csv(META / "clips_test.csv")["clip"])
    has_t = [k for k in t_keys if any((d / (k + ".npy")).exists()
                                      for d in te_dirs)]
    te_ram = None if ram is None else {(i - len(tr_dirs), k): v
                                       for (i, k), v in ram.items()
                                       if i >= len(tr_dirs)}
    te_ds = VideoDS(has_t, np.zeros(len(has_t), int), te_dirs, False, te_ram,
                    widths)
    pos = {c: i for i, c in enumerate(t_keys)}
    probs = np.zeros((len(t_keys), 40), np.float32)
    for k in run_folds:
        model = make_model(args.arch, in_ch, args.width, N_FRAMES,
                           pretrained=False).to(device)
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
