# -*- coding: utf-8 -*-
"""Реальный упаковщик: per-out-channel симметричный int-N с бит-паком
(np.packbits) + fp16-скейлы; всё остальное (BN, bias, 1-D) — fp16.
pack_sd(sd, bits) -> dict of numpy; unpack_sd(packed) -> fp32 state_dict,
побитово равный quantize_sd() из predict_video_oof.py (гейт = прод).
CLI: python pack_checkpoint.py <ckpt.pth> <bits> [--out file.npz]
     печатает размер файла и максимальную разницу с гейт-деквантом."""
import argparse
import io
import sys
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRATCH))
from predict_video_oof import quantize_sd  # noqa: E402


GCLIPS = [1.0, 0.97, 0.94, 0.9, 0.86, 0.82, 0.78, 0.74, 0.7]


def _group_codes(flat, scale, group, qmax, clip):
    """uint8-подмасштабы на группу из `group` весов вдоль входной оси: g_scale = scale_row * u/255 (u >= 1).
    clip=True: MSE-оптимальный клиппинг по сетке GCLIPS на группу. Возвращает (u uint8 [rows, ngroups], q int [rows, cols])."""
    rows, cols = flat.shape
    g = flat.reshape(rows, cols // group, group)
    gmax = g.abs().amax(2)                                            # [rows, ngroups]
    best_u, best_err, best_q = None, None, None
    for c in (GCLIPS if clip else [1.0]):
        u = torch.ceil(gmax * c / scale[:, None] * 255 / qmax).clamp(1, 255)   # ceil: без переполнения при c=1
        gs = scale[:, None] * u / 255                                 # [rows, ngroups]
        q = torch.round(g / gs[:, :, None]).clamp(-qmax, qmax)
        err = ((q * gs[:, :, None] - g) ** 2).sum(2)
        if best_u is None:
            best_u, best_err, best_q = u, err, q
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err); best_u = torch.where(better, u, best_u)
            best_q = torch.where(better[:, :, None], q, best_q)
    return best_u.to(torch.uint8).numpy(), best_q.reshape(rows, cols)


def pack_sd(sd, bits, group=0, clip=False):
    qmax = 2 ** (bits - 1) - 1
    out = {"__bits__": np.int64(bits)}
    for k, v in sd.items():
        if v.ndim >= 2 and k.endswith("weight") and v.dtype == torch.float32:
            w = v.float()
            flat = w.reshape(w.shape[0], -1)
            scale = flat.abs().max(1).values.clamp_min(1e-8) / qmax
            if group and flat.shape[1] % group == 0 and flat.shape[1] > group:
                u8, q = _group_codes(flat, scale, group, qmax, clip)
                out[k + "|g"] = u8; out[k + "|gs"] = np.int64(group)
            else:
                q = torch.round(flat / scale[:, None]).clamp(-qmax, qmax)
            u = (q + qmax).to(torch.uint8).numpy()          # 0..2*qmax
            # бит-пак: каждый элемент -> `bits` бит, MSB-first
            bitsarr = np.unpackbits(u[:, :, None], axis=2)[:, :, 8 - bits:]
            out[k + "|q"] = np.packbits(bitsarr.reshape(-1))
            out[k + "|s"] = scale.numpy().astype(np.float32)
            out[k + "|shape"] = np.array(w.shape, np.int64)
        else:
            out[k + "|f"] = (v.numpy().astype(np.float32)
                             if v.is_floating_point() else v.numpy())
    return out


def unpack_sd(p):
    bits = int(p["__bits__"])
    qmax = 2 ** (bits - 1) - 1
    sd = {}
    keys = {k.rsplit("|", 1)[0] for k in p if k != "__bits__" and not k.endswith("|gs")}
    for k in keys:
        if k + "|q" in p:
            shape = tuple(int(x) for x in p[k + "|shape"])
            n = int(np.prod(shape))
            u = np.unpackbits(p[k + "|q"])[: n * bits].reshape(n, bits)
            u = np.packbits(np.concatenate(
                [np.zeros((n, 8 - bits), np.uint8), u], 1), axis=1)[:, 0]
            q = u.astype(np.float32) - qmax
            scale = p[k + "|s"].astype(np.float32)
            if k + "|g" in p:
                gsz = int(p[k + "|gs"]); u8 = p[k + "|g"].astype(np.float32)          # [rows, ngroups]
                gs = scale[:, None] * u8 / 255.0
                w = (q.reshape(shape[0], -1, gsz) * gs[:, :, None]).reshape(shape[0], -1)
            else:
                w = q.reshape(shape[0], -1) * scale[:, None]
            sd[k] = torch.from_numpy(w.reshape(shape))
        else:
            sd[k] = torch.from_numpy(np.ascontiguousarray(p[k + "|f"]))
            if sd[k].dtype == torch.float16:
                sd[k] = sd[k].float()
    return sd


def save_packed(packed, path):
    np.savez(path, **packed)          # без zip-сжатия: честный размер
    return Path(path).stat().st_size / 2 ** 20


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("bits", type=int)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sd = torch.load(a.ckpt, weights_only=True, map_location="cpu")
    packed = pack_sd(sd, a.bits)
    out = a.out or str(SCRATCH / f"packed_int{a.bits}.npz")
    mb = save_packed(packed, out)
    ref, est, _ = quantize_sd(sd, a.bits)
    back = unpack_sd(np.load(out))
    worst = 0.0
    for k in ref:
        d = (ref[k].float() - back[k].float()).abs().max().item()
        worst = max(worst, d)
    print(f"{Path(a.ckpt).name} int{a.bits}: файл {mb:.1f} МБ (оценка "
          f"{est:.1f}), макс. расхождение с гейтом {worst:.2e}")
