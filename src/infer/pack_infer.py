# -*- coding: utf-8 -*-
"""Инференс ТОЛЬКО из единого файла пака (npz): модели (бит-пак), HGB,
нормировка TCN, fusion-конфиг, таблицы декодера. Вход — кэши теста
(детерминированный препроцессинг; YOLO-боксы уже запечены в кропы),
imu_test_feat.npy / imu_test.npy / imu_v2_test.npy (HGB v2), skeleton_test.npz, sessions_test.csv.
Выход — submission CSV; сравнение с эталоном (--ref)."""
import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

SC = Path(__file__).resolve().parent
ROOT = SC.parents[1]
sys.path.insert(0, str(SC))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "train"))
sys.stdout.reconfigure(encoding="utf-8")
from pack_checkpoint import unpack_sd  # noqa: E402
from predict_video_oof import RollDS  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


@torch.no_grad()
def predict(model, ds, device, hflip=False, bs=16):
    # num_workers=0: скрипт без __main__-гарда, Windows spawn (урок 01.09)
    model.eval()
    res = []
    for xb, _ in DataLoader(ds, bs, num_workers=0):
        xb = xb.to(device)
        if hflip:
            xb = torch.flip(xb, dims=[-1])
        with torch.amp.autocast("cuda", enabled=device == "cuda"):
            res.append(F.softmax(model(xb).float(), 1).cpu().numpy())
    return np.concatenate(res)
from train.train_video import make_model  # noqa: E402
from models.fusion import apply_fusion  # noqa: E402
import train_imu_tcn as tt  # noqa: E402
import train_skeleton as ts  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pack", default="experiments/pack1A_checkpoint.npz")
ap.add_argument("--probs-cache", default=None, help="npz: если есть — взять вероятности видеоветок и признаки адаптации из него (без GPU); если нет — сохранить после расчёта")
ap.add_argument("--out", default="submissions/submission_pack_infer.csv")
ap.add_argument("--ref", default=None, help="эталонный CSV для сверки (опционально)")
ap.add_argument("--test-csv", default=None, help="CSV с колонкой path (дефолт: META/test_paths.csv, иначе ROOT/test.csv)")
args = ap.parse_args()
import os  # noqa: E402
CACHE = Path(os.environ.get("CUNHX_CACHE", ROOT / "data/cache"))
META = Path(os.environ.get("CUNHX_META", ROOT / "data/meta"))
print(f"кэши {CACHE} | мета {META}")
dev = "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"  # CUDA_VISIBLE_DEVICES= on Windows: is_available() True, 0 devices
EPS, K = 1e-12, 40
BETA, THR, GAMMA, CGATE = 0.25, 30.0, 0.3, 0.5

z = np.load(ROOT / args.pack)
files = set(z.files)


def comp(prefix):
    return {k[len(prefix) + 1:]: z[k] for k in files if k.startswith(prefix + "/")}


cfg = json.loads(bytes(z["fusion_cfg"]).decode())
TAU = cfg["tau"]
A_in, A1_in, A1_b = z["tables/A_in"], z["tables/A1_in"], z["tables/A1_b"]
lift = z["tables/lift"]
lift3 = z["tables/lift3"] if "tables/lift3" in files else None
LIFT_ORD = bool(cfg.get("lift_ord", False)) and lift3 is not None
REPAIR_MODE = cfg.get("repair_mode", "hungarian")
if REPAIR_MODE == "map":
    from xt_layers import map_distinct
print(f"пак {args.pack}: компонентов {len(files)}, ветки {cfg['names']}, tau={TAU}")

t_keys = sorted(pd.read_csv(META / "clips_test.csv")["clip"])
pos = {c: i for i, c in enumerate(t_keys)}
probs = {}


ADAPT = cfg.get("adapt") or None            # ярус (b): прототипы в признаках ViT-ветки, второй проход декодера
ADAPT_PREFIX = (ADAPT or {}).get("prefix", "vit")
FEATS_OUT = {}                                # prefix -> (N, D) L2-нормированные fc_norm-признаки (NaN = нет клипа)


def video_branch(prefix, arch, mods, frames=16, crop=112):
    sd = unpack_sd(comp(prefix))
    te_dirs = [CACHE / f"{m}_test" for m in mods]
    if any((not d.exists()) or next(d.glob("*.npy"), None) is None for d in te_dirs):
        print(f"  {prefix}: нет кэшей {mods} -> ветка пропущена (NaN)", flush=True)
        return np.full((len(t_keys), 40), np.nan, np.float32)
    widths = []
    for d in te_dirs:
        probe = np.load(next(d.glob("*.npy")), mmap_mode="r")
        widths.append(3 if probe.ndim == 4 else 1)
    model = make_model(arch, sum(widths), 32, frames, pretrained=False).to(dev)
    model.load_state_dict(sd)
    has_t = [k for k in t_keys if any((d / (k + ".npy")).exists()
                                      for d in te_dirs)]
    acc = np.zeros((len(t_keys), 40), np.float64)
    feats, facc = [], None
    if ADAPT is not None and prefix == ADAPT_PREFIX and hasattr(model, "fc_norm"):
        model.fc_norm.register_forward_hook(lambda m, i, o: feats.append(o.detach().float().cpu().numpy()))
    view = BRANCH_VIEWS.get(prefix, "tta6")
    rolls = (0,) if view in ("1", "flip2") else (-1, 0, 1)
    flips = (False, True) if view in ("tta6", "flip2") else (False,)
    nv = 0
    for roll in rolls:
        ds = RollDS(has_t, np.zeros(len(has_t), int), te_dirs, False, None,
                    widths, n_frames=frames, crop=crop)
        ds.roll = roll
        for hf in flips:
            feats.clear()
            p = predict(model, ds, dev, hf)
            for c, row in zip(has_t, p):
                acc[pos[c]] += row
            if ADAPT is not None and prefix == ADAPT_PREFIX and feats:
                fv = np.concatenate(feats)
                if facc is None:
                    facc = np.zeros((len(t_keys), fv.shape[1]), np.float64)
                for c, row in zip(has_t, fv):
                    facc[pos[c]] += row
            nv += 1
    out = np.full((len(t_keys), 40), np.nan, np.float32)
    sc = acc.sum(1) > 0
    out[sc] = (acc[sc] / acc[sc].sum(1, keepdims=True)).astype(np.float32)
    print(f"  {prefix}: {sc.sum()}/{len(t_keys)} клипов, виды {view} ({nv})", flush=True)
    if facc is not None:
        fz = np.full(facc.shape, np.nan, np.float64)
        fz[sc] = facc[sc] / (np.linalg.norm(facc[sc], axis=1, keepdims=True) + 1e-9)
        FEATS_OUT[prefix] = fz
        print(f"  {prefix}: признаки fc_norm {sc.sum()} x {facc.shape[1]} для адаптации", flush=True)
    return out


# входные кэши веток: самоописание из пака (branch_mods), иначе дефолты pack1
BRANCH_MODS = {"r34c5": ["Depth_ColorRGBcrop5", "IRcrop5"],
               "r34c2": ["Depth_ColorRGBcrop2", "IRcrop2"],
               "dir": ["Depth_Colorcrop", "IRcrop"], "thermal": ["Thermal"]}
if "branch_mods" in files:
    BRANCH_MODS.update(json.loads(bytes(z["branch_mods"]).decode()))
BRANCH_VIEWS = json.loads(bytes(z["branch_views"]).decode()) if "branch_views" in files else {}
# самоописание веток (arch/mods/frames/crop); дефолт = таблица pack1
BRANCH_SPEC = {"r34c5": {"prefix": "r34c5", "arch": "r2p1d34", "mods": BRANCH_MODS["r34c5"], "frames": 16, "crop": 112},
               "r34c2": {"prefix": "r34c5", "arch": "r2p1d34", "mods": BRANCH_MODS["r34c2"], "frames": 16, "crop": 112},
               "dir_r2p1d": {"prefix": "dir", "arch": "r2p1d", "mods": BRANCH_MODS["dir"], "frames": 16, "crop": 112},
               "thermal_pl2": {"prefix": "thermal", "arch": "r2p1d", "mods": BRANCH_MODS["thermal"], "frames": 16, "crop": 112}}
if "branch_spec" in files:
    BRANCH_SPEC.update(json.loads(bytes(z["branch_spec"]).decode()))
print(f"  спецификация веток: { {k: (v['arch'], v['crop']) for k, v in BRANCH_SPEC.items()} }")
print(f"  виды TTA: {BRANCH_VIEWS or 'tta6 (дефолт)'}")
print(f"  входы веток: {BRANCH_MODS}")
W = cfg.get("weights", {})
PC = np.load(args.probs_cache) if args.probs_cache and Path(args.probs_cache).exists() else None
video_names = []
for name, spec in BRANCH_SPEC.items():
    if name not in cfg["names"]:
        continue
    if W.get(name, 1.0) <= 0:
        print(f"  {name}: вес 0 -> не считается", flush=True)
        probs[name] = np.full((len(t_keys), 40), 1.0 / 40, np.float32)
    elif f"{spec['prefix']}/__bits__" not in files:
        print(f"  {name}: компонента {spec['prefix']} нет в паке -> NaN", flush=True)
        probs[name] = np.full((len(t_keys), 40), np.nan, np.float32)
    elif PC is not None and f"probs/{name}" in PC.files:
        probs[name] = PC[f"probs/{name}"]
        if f"feats/{spec['prefix']}" in PC.files:
            FEATS_OUT[spec["prefix"]] = PC[f"feats/{spec['prefix']}"]
        print(f"  {name}: вероятности из кэша {args.probs_cache}", flush=True)
    else:
        probs[name] = video_branch(spec["prefix"], spec["arch"], spec["mods"], spec.get("frames", 16), spec.get("crop", 112))
        video_names.append(name)
if args.probs_cache and PC is None and video_names:
    np.savez(args.probs_cache, **{f"probs/{n}": probs[n] for n in video_names}, **{f"feats/{p}": v for p, v in FEATS_OUT.items()})
    print(f"  вероятности видеоветок {video_names} и признаки {list(FEATS_OUT)} -> {args.probs_cache}", flush=True)

# HGB: v1 = 450 статфич из imu_test_feat.npy; hgb_spec в паке = признаки v2 (физические, из imu_v2_test.npy)
try:
    clf = pickle.loads(gzip.decompress(bytes(z["hgb_gz"])))
    if "hgb_spec" in files:
        hspec = json.loads(bytes(z["hgb_spec"]).decode())
        from train_imu_hgb_v2 import clip_features as hgb2_clip_features, select_features as hgb2_select_features  # noqa: E402
        Xr2 = np.load(CACHE / "imu_v2_test.npy")
        X_feat = hgb2_select_features(hgb2_clip_features(Xr2), Xr2, hspec["feat"], hspec.get("stats"), hspec.get("devs"))
        X_feat[X_feat[:, -7:-2].sum(1) <= 0] = np.nan          # клип без IMU -> NaN-строка (как has в обучении)
        print(f"  imu_hgb: признаки v2 {hspec['feat']} {X_feat.shape}", flush=True)
    else:
        X_feat = np.load(CACHE / "imu_test_feat.npy")
    has_f = ~np.isnan(X_feat).all(1)
    h = np.full((len(X_feat), 40), np.nan, np.float32)
    h[np.where(has_f)[0][:, None], clf.classes_[None, :]] = clf.predict_proba(
        X_feat[has_f])
    probs["imu_hgb"] = h
    print(f"  imu_hgb: {has_f.sum()} клипов")
except Exception as e:  # noqa: BLE001
    print(f"  imu_hgb: ОШИБКА {type(e).__name__}: {e} -> ветка пропущена (NaN)", flush=True)
    probs["imu_hgb"] = np.full((len(t_keys), 40), np.nan, np.float32)

# IMU v2 (сеть по рецепту CMI: физические признаки в модели, окна 32 шага, усреднение окон; фолды усредняются)
if "imu_v2_spec" in files and "imu_v2" in cfg["names"]:
    try:
        from models.branches.imu_v2 import IMUv2  # noqa: E402
        from train_imu_v2 import clip_meta as imu2_clip_meta, predict_clips as imu2_predict  # noqa: E402
        spec2 = json.loads(bytes(z["imu_v2_spec"]).decode())
        X2 = np.load(CACHE / "imu_v2_test.npy")
        valid2, present2, last2, logdur2 = imu2_clip_meta(X2)
        acc2 = np.zeros((len(X2), 40), np.float64); n2 = 0
        for k in range(5):
            if f"imuv2_{k}/__bits__" not in files:
                continue
            m2 = IMUv2(stem=tuple(spec2.get("stem", (32, 64))), temp=spec2.get("temp", 160), head=spec2.get("head", 128),
                       drop=spec2.get("drop", 0.2), shared_stem=spec2.get("shared_stem", True)).to(dev)
            m2.load_state_dict(unpack_sd(comp(f"imuv2_{k}")))
            m2.eval()
            with torch.no_grad():
                acc2 += np.nan_to_num(imu2_predict(m2, X2, valid2, present2, logdur2, last2, dev), nan=0.0)
            n2 += 1
        has2 = present2.any(1)
        p2 = np.full((len(X2), 40), np.nan, np.float32)
        p2[has2] = (acc2[has2] / max(n2, 1)).astype(np.float32)
        probs["imu_v2"] = p2
        print(f"  imu_v2: {n2} фолдов, {has2.sum()} клипов", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  imu_v2: ОШИБКА {type(e).__name__}: {e} -> ветка пропущена (NaN)", flush=True)
        probs["imu_v2"] = np.full((len(t_keys), 40), np.nan, np.float32)

# TCN x3 int8 + нормировка из пака
try:
    raw_t = np.load(CACHE / "imu_test.npy")
    has_t = ~np.isnan(raw_t).all(axis=(1, 2, 3))
    X_t = np.stack([tt.to_input(x) for x in raw_t])
    X_t = (X_t - z["tcn_norm/mu"]) / z["tcn_norm/sd"]
    acc = np.zeros((len(X_t), 40), np.float32)
    n_t = 0
    for k in range(5):
        if f"tcn{k}/__bits__" not in files:
            continue
        m = tt.ImuTCN().to(dev)
        m.load_state_dict(unpack_sd(comp(f"tcn{k}")))
        m.eval()
        with torch.no_grad():
            acc[has_t] += F.softmax(m(torch.from_numpy(X_t[has_t]).to(dev)),
                                    1).cpu().numpy()
        n_t += 1
    tcn = np.full((len(X_t), 40), np.nan, np.float32)
    tcn[has_t] = acc[has_t] / acc[has_t].sum(1, keepdims=True)
    probs["imu_tcn"] = tcn
    print(f"  imu_tcn: {n_t} моделей")
except Exception as e:  # noqa: BLE001
    print(f"  imu_tcn: ОШИБКА {type(e).__name__}: {e} -> ветка пропущена (NaN)", flush=True)
    probs["imu_tcn"] = np.full((len(t_keys), 40), np.nan, np.float32)

# CTR-GCN x3 int8
try:
    test_raw = np.load(CACHE / "skeleton_test.npz")
    t_arr = {k: ts.prep_clip(test_raw[k], "joint") for k in test_raw.files}
    have = [c for c in t_keys if c in t_arr]
    acc = np.zeros((len(t_keys), 40), np.float32)
    n_s = 0
    for k in range(5):
        if f"ctrgcn{k}/__bits__" not in files:
            continue
        m = ts.make_model("ctrgcn", 64).to(dev)
        m.load_state_dict(unpack_sd(comp(f"ctrgcn{k}")))
        p = ts.predict(m, have, t_arr, dev)
        for c, row in zip(have, p):
            acc[pos[c]] += row
        n_s += 1
    sk = np.full((len(t_keys), 40), np.nan, np.float32)
    sc = acc.sum(1) > 0
    sk[sc] = acc[sc] / acc[sc].sum(1, keepdims=True)
    probs["ctrgcn"] = sk
    print(f"  ctrgcn: {n_s} моделей")
except Exception as e:  # noqa: BLE001
    print(f"  ctrgcn: ОШИБКА {type(e).__name__}: {e} -> ветка пропущена (NaN)", flush=True)
    probs["ctrgcn"] = np.full((len(t_keys), 40), np.nan, np.float32)

fusedT = apply_fusion(cfg, {n: probs[n].astype(np.float64)
                            for n in cfg["names"]})
PRIOR_ALPHA = cfg.get("prior_alpha", 0.0)
if PRIOR_ALPHA:
    okA = ~np.isnan(fusedT).any(1)
    qA = fusedT[okA].mean(0)
    qA = qA / qA.sum()
    pA = np.clip(fusedT[okA], EPS, 1) / np.power(qA + EPS, PRIOR_ALPHA)
    fusedT[okA] = pA / pA.sum(1, keepdims=True)
    print(f"  приор-коррекция alpha={PRIOR_ALPHA}")

# ---- декодер (порт decode_final --repair --align, таблицы из пака) ----
ct = pd.read_csv(META / "clips_test.csv").set_index("clip").loc[t_keys].reset_index()
def _files(col, scale=1.0):  # a modality absent from the whole sample has no column in clips_test.csv
    return ct[col].to_numpy(float) / scale if col in ct.columns else np.zeros(len(ct))
fr = _files("Depth_Color_files")
fr = np.where(fr > 0, fr, _files("IR_files"))
fr = np.where(fr > 0, fr, _files("Thermal_files", 2.45))
fr[fr <= 0] = np.nan
lf_te = np.log(fr)
fin_te = np.isfinite(lf_te)
edges_te = np.quantile(lf_te[fin_te], np.linspace(0, 1, 9)[1:-1])
bn_te = np.digitize(lf_te, edges_te)
st = pd.read_csv(META / "sessions_test.csv")
st["first_ts"] = pd.to_datetime(st["first_ts"]).astype("datetime64[ns]")
st["sec"] = st["first_ts"].astype("int64") / 1e9


def fb2(logE, bnd):
    n = logE.shape[0]
    E = np.exp(logE - logE.max(1, keepdims=True))
    E /= E.sum(1, keepdims=True)
    if n == 1:
        return E
    al = np.zeros((n, K, K))
    A1f = A1_b if bnd[1] else A1_in
    al[1] = (E[0][:, None] / K) * A1f * E[1][None, :]
    al[1] /= max(al[1].sum(), EPS)
    for t in range(2, n):
        if bnd[t]:
            v = (al[t - 1].sum(0)[:, None] * A1_b) * E[t][None, :]
        elif bnd[t - 1]:
            v = np.einsum("ab,bc->bc", al[t - 1], A1_in) * E[t][None, :]
        else:
            v = np.einsum("ab,abc->bc", al[t - 1], A_in) * E[t][None, :]
        al[t] = v / max(v.sum(), EPS)
    be = np.ones((n, K, K))
    for t in range(n - 2, 0, -1):
        if bnd[t + 1]:
            r = (A1_b * E[t + 1][None, :] * be[t + 1]).sum(1)
            be[t] = np.tile(r[None, :], (K, 1))
        elif bnd[t]:
            r = (A1_in * E[t + 1][None, :] * be[t + 1]).sum(1)
            be[t] = np.tile(r[None, :], (K, 1))
        else:
            be[t] = np.einsum("abc,c,bc->ab", A_in, E[t + 1], be[t + 1])
        be[t] /= max(be[t].max(), EPS)
    post = np.zeros((n, K))
    for t in range(1, n):
        pc = (al[t] * be[t]).sum(0)
        post[t] = pc / max(pc.sum(), EPS)
    p0 = (al[1] * be[1]).sum(1)
    post[0] = p0 / max(p0.sum(), EPS)
    return post


def cosm(E, a, b):
    num = (E[a] * E[b]).sum(1)
    den = np.linalg.norm(E[a], axis=1) * np.linalg.norm(E[b], axis=1) + EPS
    return num / den


def decode_all(fusedT):
    """Порт decode_final --repair --align: сессии -> блоки (600 с) -> чанки -> сглаживание/пулинг/репэйр."""
    out = fusedT.copy()
    for s, gr in st.groupby("session"):
        gr = gr.sort_values("sec")
        rows_all = gr["clip"].map(pos).to_numpy()
        secs_all = gr["sec"].to_numpy()
        blocks, cur = [], [0]
        for i in range(1, len(rows_all)):
            if secs_all[i] - secs_all[i - 1] > 600:
                blocks.append(np.array(cur))
                cur = []
            cur.append(i)
        blocks.append(np.array(cur))
        for bl in blocks:
            rows, secs = rows_all[bl], secs_all[bl]
            logE = np.where(np.isnan(fusedT[rows]), np.log(1.0 / K),
                            np.log(fusedT[rows] + EPS)) * TAU
            logE = logE + BETA * np.where(fin_te[rows][:, None], lift[bn_te[rows]], 0.0)
            gaps = np.diff(secs)
            bnd = np.concatenate([[False], gaps > THR])
            E = np.exp(logE - logE.max(1, keepdims=True))
            E /= E.sum(1, keepdims=True)
            cut = np.where(gaps > THR)[0]
            ch, s0 = [], 0
            for c in cut:
                ch.append(np.arange(s0, c + 1))
                s0 = c + 1
            ch.append(np.arange(s0, len(rows)))
            if LIFT_ORD:
                ords = np.zeros(len(rows), int)
                g0 = 0
                while g0 < len(ch):
                    o = 0
                    while g0 + 1 < len(ch) and len(ch[g0 + 1]) == len(ch[g0]):
                        if cosm(E, ch[g0], ch[g0 + 1]).mean() < CGATE:
                            break
                        g0 += 1
                        o += 1
                        ords[ch[g0]] = min(o, 2)
                    g0 += 1
                logE = np.where(np.isnan(fusedT[rows]), np.log(1.0 / K),
                                np.log(fusedT[rows] + EPS)) * TAU
                logE = logE + BETA * np.where(fin_te[rows][:, None],
                                              lift3[bn_te[rows], ords], 0.0)
                E = np.exp(logE - logE.max(1, keepdims=True))
                E /= E.sum(1, keepdims=True)
            pooled = logE.copy()
            g = 0
            while g < len(ch) - 1:
                grp = [ch[g]]
                while g + 1 < len(ch) and len(ch[g + 1]) == len(ch[g]):
                    if cosm(E, ch[g], ch[g + 1]).mean() < CGATE:
                        break
                    grp.append(ch[g + 1])
                    g += 1
                if len(grp) > 1:
                    for i in range(len(grp[0])):
                        ri = [c[i] for c in grp]
                        ssum = sum(logE[r] for r in ri)
                        for r in ri:
                            pooled[r] = logE[r] + GAMMA * (ssum - logE[r])
                g += 1
            for a, b in zip(ch[:-1], ch[1:]):          # I2b |dL|=1
                if abs(len(a) - len(b)) != 1 or min(len(a), len(b)) < 2:
                    continue
                sh, lg = (a, b) if len(a) < len(b) else (b, a)
                best, bscore = None, -1.0
                for kk in range(len(lg)):
                    idx_l = [j for j in range(len(lg)) if j != kk]
                    scv = float(cosm(E, sh, lg[idx_l]).mean())
                    if scv > bscore:
                        best, bscore = idx_l, scv
                if bscore < CGATE:
                    continue
                for i, j in zip(range(len(sh)), best):
                    pooled[sh[i]] = pooled[sh[i]] + GAMMA * logE[lg[j]]
                    pooled[lg[j]] = pooled[lg[j]] + GAMMA * logE[sh[i]]
            for a, b in zip(ch[:-1], ch[1:]):          # align dl2 + guard
                dL = abs(len(a) - len(b))
                if dL < 2 or min(len(a), len(b)) < 2:
                    continue
                la = set(logE[a].argmax(1).tolist())
                lb2 = set(logE[b].argmax(1).tolist())
                if len(la & lb2) / len(la | lb2) < 0.25:
                    continue
                sh, lg = (a, b) if len(a) < len(b) else (b, a)
                best_o, bscore = None, -1.0
                for o in range(dL + 1):
                    scv = float(cosm(E, sh, lg[o:o + len(sh)]).mean())
                    if scv > bscore:
                        best_o, bscore = o, scv
                if bscore < CGATE:
                    continue
                for i in range(len(sh)):
                    j = lg[best_o + i]
                    pooled[sh[i]] = pooled[sh[i]] + GAMMA * logE[j]
                    pooled[j] = pooled[j] + GAMMA * logE[sh[i]]
            for i2, c2 in enumerate(ch):               # solo
                if len(c2) != 1:
                    continue
                cands = []
                if i2 > 0 and len(ch[i2 - 1]) >= 2:
                    cands.extend(ch[i2 - 1].tolist())
                if i2 + 1 < len(ch) and len(ch[i2 + 1]) >= 2:
                    cands.extend(ch[i2 + 1].tolist())
                if not cands:
                    continue
                ca = np.array(cands)
                scv = (E[c2[0]][None, :] * E[ca]).sum(1) / (
                    np.linalg.norm(E[c2[0]]) * np.linalg.norm(E[ca], axis=1) + EPS)
                jb = int(np.argmax(scv))
                if scv[jb] < 0.8:
                    continue
                j = cands[jb]
                pooled[c2[0]] = pooled[c2[0]] + GAMMA * logE[j]
                pooled[j] = pooled[j] + GAMMA * logE[c2[0]]
            post = fb2(pooled, bnd)
            groups, gi2 = [], 0                        # repair
            while gi2 < len(ch):
                grp2 = [ch[gi2]]
                while gi2 + 1 < len(ch) and len(ch[gi2 + 1]) == len(ch[gi2]):
                    if cosm(E, ch[gi2], ch[gi2 + 1]).mean() < CGATE:
                        break
                    grp2.append(ch[gi2 + 1])
                    gi2 += 1
                groups.append(grp2)
                gi2 += 1
            lp = np.log(post + EPS)
            pd_bl = post.argmax(1)
            for grp2 in groups:
                L = len(grp2[0])
                if L < 2:
                    continue
                Cg = sum(lp[c] for c in grp2)
                for c in grp2:
                    lab = pd_bl[c]
                    if len(set(lab.tolist())) == L:
                        continue
                    if REPAIR_MODE == "map":
                        lab_new = map_distinct(Cg, A_in, A1_in, w_tr=0.0)
                        if lab_new is not None:
                            for i_, t in enumerate(c):
                                old, new = pd_bl[t], lab_new[i_]
                                if new != old:
                                    post[t, [old, new]] = post[t, [new, old]]
                        continue
                    vals, cnts_ = np.unique(lab, return_counts=True)
                    dup = set(vals[cnts_ > 1].tolist())
                    confl = [i for i in range(L) if lab[i] in dup]
                    keep = {lab[i] for i in range(L) if i not in confl}
                    cand = [k for k in range(K) if k not in keep]
                    sub = Cg[np.ix_(confl, cand)]
                    r_i, c_i = linear_sum_assignment(-sub)
                    for ri, ci in zip(r_i, c_i):
                        t = c[confl[ri]]
                        old, new = pd_bl[t], cand[ci]
                        if new != old:
                            post[t, [old, new]] = post[t, [new, old]]
            out[rows] = post
    return out


out = decode_all(fusedT)
fused_final = fusedT

# ---- ярус (b): сессионные прототипы в признаках ViT + второй проход декодера ----
if ADAPT is not None:
    if "tables/proto_src" not in files or ADAPT_PREFIX not in FEATS_OUT:
        print(f"  adapt: нет прототипов/признаков ({ADAPT_PREFIX}) -> пропуск", flush=True)
    else:
        kappa, Tp, w_ad = float(ADAPT.get("kappa", 3.0)), float(ADAPT.get("tp", 0.05)), float(ADAPT.get("w", 0.3))
        Fz = FEATS_OUT[ADAPT_PREFIX]
        C_src = z["tables/proto_src"].astype(np.float64)
        C_src /= np.linalg.norm(C_src, axis=1, keepdims=True) + 1e-9
        pred1 = out.argmax(1)
        fusedA = fusedT.copy()
        n_sess, n_clips = 0, 0
        for s_, gr in st.groupby("session"):
            rows = gr["clip"].map(pos).to_numpy()
            ok = rows[~np.isnan(Fz[rows]).any(1) & ~np.isnan(fusedT[rows]).any(1)]
            if len(ok) < 2:
                continue
            lab = pred1[ok]
            cs = np.zeros((K, Fz.shape[1])); cnt = np.zeros(K)
            for i_, k_ in enumerate(lab):
                cs[k_] += Fz[ok[i_]]; cnt[k_] += 1
            C = (cs + kappa * C_src) / (cnt[:, None] + kappa)
            C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
            lg = (Fz[ok] @ C.T) / Tp
            lg -= lg.max(1, keepdims=True)
            pp = np.exp(lg); pp /= pp.sum(1, keepdims=True)
            pa = np.clip(fusedT[ok], EPS, 1) * np.power(pp + EPS, w_ad)
            fusedA[ok] = pa / pa.sum(1, keepdims=True)
            n_sess += 1; n_clips += len(ok)
        out2 = decode_all(fusedA)
        ch_pre = int((fusedA.argmax(1) != fusedT.argmax(1)).sum()); ch_post = int((out2.argmax(1) != pred1).sum())
        print(f"  adapt: kappa={kappa:g} tp={Tp:g} w={w_ad:g} | сессий {n_sess}, клипов {n_clips} | изменено до декодера {ch_pre}, после {ch_post}", flush=True)
        out = out2
        fused_final = fusedA

# ---- EM переходов на тестовых сессиях (SHL/Ion Switching): счёт переходов по декодированным меткам ->
# ---- смесь с трейновыми P (tables/P1i,P2i,P1b) -> те же преобразования, что в паке (^lam_pow, mu) -> повторный декод ----
EM = cfg.get("em")
if EM is not None:
    if "tables/P1i" not in files:
        print("  em: нет сырых таблиц переходов в паке -> пропуск", flush=True)
    else:
        lam_em, em_iters, em_what = float(EM.get("lam", 0.3)), int(EM.get("iters", 1)), EM.get("what", "all")
        lam_pow, mu_em = float(EM.get("lam_pow", 0.75)), float(EM.get("mu", 0.75))
        P1i, P2i, P1b = (z[f"tables/{k}"].astype(np.float64) for k in ("P1i", "P2i", "P1b"))
        pred_cur = out.argmax(1)
        for it_ in range(em_iters):
            c1i = np.full((K, K), 0.5); c2i = np.full((K, K, K), 0.1); c1b = np.full((K, K), 0.5)
            for s_, gr in st.groupby("session"):
                gr = gr.sort_values("sec")
                rows = gr["clip"].map(pos).to_numpy(); secs = gr["sec"].to_numpy(); lab = pred_cur[rows]; gaps = np.diff(secs)
                for t in range(1, len(rows)):
                    if gaps[t - 1] <= THR:
                        c1i[lab[t - 1], lab[t]] += 1
                        if t >= 2 and gaps[t - 2] <= THR:
                            c2i[lab[t - 2], lab[t - 1], lab[t]] += 1
                    else:
                        c1b[lab[t - 1], lab[t]] += 1
            H1i, H2i, H1b = c1i / c1i.sum(1, keepdims=True), c2i / c2i.sum(2, keepdims=True), c1b / c1b.sum(1, keepdims=True)
            Q1i = (1 - lam_em) * P1i + lam_em * H1i if em_what in ("all", "in") else P1i
            Q2i = (1 - lam_em) * P2i + lam_em * H2i if em_what in ("all", "in") else P2i
            Q1b = (1 - lam_em) * P1b + lam_em * H1b if em_what in ("all", "b") else P1b
            A_ = (mu_em * Q2i + (1 - mu_em) * Q1i[None, :, :]) ** lam_pow; A_in = A_ / A_.sum(2, keepdims=True)
            A1_ = Q1i ** lam_pow; A1_in = A1_ / A1_.sum(1, keepdims=True)
            Ab_ = Q1b ** 0.75; A1_b = Ab_ / Ab_.sum(1, keepdims=True)
            out3 = decode_all(fused_final); pred_new = out3.argmax(1)
            print(f"  em: lam={lam_em:g} what={em_what} it {it_ + 1}/{em_iters} | переходов внутри {int(c1i.sum() - 0.5 * K * K)}, на границах {int(c1b.sum() - 0.5 * K * K)} | изменено {int((pred_new != pred_cur).sum())}", flush=True)
            pred_cur = pred_new; out = out3

tp = Path(args.test_csv) if args.test_csv else META / "test_paths.csv"
if not tp.exists():
    tp = ROOT / "test.csv"
test_csv = pd.read_csv(tp)
clip_of = test_csv["path"].str.rstrip("/").str.split("/").str[-1]
known = clip_of.map(pos)
if known.isna().any():
    print(f"  ВНИМАНИЕ: {int(known.isna().sum())} путей из {tp.name} нет среди клипов -> prediction 36 (Walk)")
pred = np.where(known.isna(), 36, out[known.fillna(0).astype(int).to_numpy()].argmax(1))
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"path": test_csv["path"], "prediction": pred}).to_csv(args.out, index=False)
print(f"wrote {args.out} ({len(pred)} строк)")
if args.ref:
    ref = pd.read_csv(args.ref)
    m = ref.merge(pd.read_csv(args.out), on="path", suffixes=("_ref", "_pk"))
    d = (m["prediction_ref"] != m["prediction_pk"]).sum()
    print(f"отличий от {args.ref}: {d}/{len(m)}")
