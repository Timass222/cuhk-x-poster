# -*- coding: utf-8 -*-
"""Сборка ЕДИНОГО файла пака: квантованные веса (бит-пак, fp32-скейлы),
компакт-HGB (gz-пикл), YOLO fp16 как есть, fusion-конфиг, таблицы
декодера. Один .npz без zip-сжатия (честный размер)."""
import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import torch

SC = Path(__file__).resolve().parent
sys.path.insert(0, str(SC))
sys.stdout.reconfigure(encoding="utf-8")
from pack_checkpoint import pack_sd  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--r34-bits", type=int, default=5)
ap.add_argument("--dir-bits", type=int, default=5)
ap.add_argument("--thermal-bits", type=int, default=5)
ap.add_argument("--tcn-folds", type=int, default=5)
ap.add_argument("--ctrgcn-folds", type=int, default=5)
ap.add_argument("--tcn-fold-ids", type=int, nargs="*", default=None, help="явный список фолдов TCN (переопределяет --tcn-folds)")
ap.add_argument("--ctrgcn-fold-ids", type=int, nargs="*", default=None, help="явный список фолдов CTR-GCN (переопределяет --ctrgcn-folds)")
ap.add_argument("--hgb", default="c150l15")
ap.add_argument("--hgb-dir", default=None, help="каталог experiments/<dir> с model_full.pkl (переопределяет imu_hgb_<hgb>)")
ap.add_argument("--hgb-feat", default="v1", choices=["v1", "all", "core", "core_yaw"], help="признаки HGB: v1 = imu_test_feat.npy (450); иначе — из imu_v2_test.npy (train_imu_hgb_v2.select_features)")
ap.add_argument("--hgb-stats", type=int, nargs="*", default=None)
ap.add_argument("--hgb-devs", type=int, nargs="*", default=None)
ap.add_argument("--imu-v2-src", default=None, help="experiments/<dir> с fold{k}.pth сети IMU v2 (компоненты imuv2_{k}, int8)")
ap.add_argument("--imu-v2-fold-ids", type=int, nargs="*", default=[0, 1, 2, 3, 4])
ap.add_argument("--imu-v2-spec", default='{"stem": [32, 64], "temp": 160, "head": 128, "drop": 0.2, "shared_stem": true}', help="kwargs IMUv2 (JSON)")
ap.add_argument("--em-tables", default=None, help="experiments/decoder_tables_raw.npz (P1i, P2i, P1b) -> tables/ для EM переходов на тестовых сессиях (cfg em)")
ap.add_argument("--r34-src", default="r34c134/fold3.pth")
ap.add_argument("--dir-src", default="video_dircrop_r2p1d/fold0.pth")
ap.add_argument("--thermal-src", default="thermal_pl2/fold0.pth")
ap.add_argument("--out", default="pack1_checkpoint.npz")
ap.add_argument("--cfg", default="fusion_pack1.json")
ap.add_argument("--r34-mods", nargs="+", default=["Depth_ColorRGBcrop5", "IRcrop5"],
                help="входные кэши R34 (crop5 / crop6) — самоописание пака для pack_infer")
ap.add_argument("--r34c2-mods", nargs="+", default=["Depth_ColorRGBcrop2", "IRcrop2"])
ap.add_argument("--dir-mods", nargs="+", default=["Depth_Colorcrop", "IRcrop"])
ap.add_argument("--thermal-mods", nargs="+", default=["Thermal"])
ap.add_argument("--vit-src", default=None, help="чекпоинт ViT-S (experiments/...): добавляет ветку vit")
ap.add_argument("--vit-bits", type=int, default=6)
ap.add_argument("--vit-qgroup", type=int, default=0, help="групповые uint8-подмасштабы квантования ViT (128: +0.64 МиБ, ошибка int6 вдвое меньше)")
ap.add_argument("--vit-qclip", action="store_true", help="MSE-клиппинг групп при квантовании ViT")
ap.add_argument("--vit-mods", nargs="+", default=["Depth_ColorRGBcrop6h", "IRcrop6h"])
ap.add_argument("--vit-crop", type=int, default=224)
ap.add_argument("--vit-name", default="vit_c6h", help="имя ветки во fusion cfg")
ap.add_argument("--no-r34", action="store_true", help="не класть R34 в пак")
ap.add_argument("--no-dir", action="store_true", help="не класть dir-r18 в пак (слот dir -> ViT)")
ap.add_argument("--no-thermal", action="store_true", help="не класть thermal в пак")
ap.add_argument("--vit-arch", default="vit_s", help="архитектура ViT-ветки: vit_s | vit_b")
ap.add_argument("--ir-src", default=None, help="чекпоинт второй ViT-ветки (IR-only): добавляет компоненту vit_ir")
ap.add_argument("--ir-bits", type=int, default=5)
ap.add_argument("--ir-mods", nargs="+", default=["IRcrop6h"])
ap.add_argument("--ir-arch", default="vit_s")
ap.add_argument("--ir-name", default="ir", help="имя IR-ветки во fusion cfg")
ap.add_argument("--proto-feat", default=None, help="OOF-признаки (experiments/.../feat_oof.npy) -> tables/proto_src (40 x D fp16, средние по классам) для ярусa (b)")
ap.add_argument("--ctrgcn-src", default="skeleton_ctrgcn", help="experiments/<dir>/fold{k}.pth для CTR-GCN")
ap.add_argument("--views", nargs="*", default=[], help="виды TTA на тесте по веткам: r34c5=1|roll3|tta6 thermal=... dir=... (дефолт tta6)")
A = ap.parse_args()
ROOT = Path(r"d:\cuhk-x")
E = ROOT / "experiments"
SPEC = ([] if A.no_r34 else [("r34c5", E / A.r34_src, A.r34_bits)]) + ([] if A.no_dir else [("dir", E / A.dir_src, A.dir_bits)]) + ([] if A.no_thermal else [("thermal", E / A.thermal_src, A.thermal_bits)])
if A.vit_src:
    SPEC.append(("vit", E / A.vit_src, A.vit_bits))
if A.ir_src:
    SPEC.append(("vit_ir", E / A.ir_src, A.ir_bits))
TCN_IDS = A.tcn_fold_ids if A.tcn_fold_ids is not None else list(range(A.tcn_folds))
CTR_IDS = A.ctrgcn_fold_ids if A.ctrgcn_fold_ids is not None else list(range(A.ctrgcn_folds))
SPEC += [(f"tcn{k}", E / f"imu_tcn/fold{k}.pth", 8) for k in TCN_IDS]
SPEC += [(f"ctrgcn{k}", E / f"{A.ctrgcn_src}/fold{k}.pth", 8) for k in CTR_IDS]
if A.imu_v2_src:
    SPEC += [(f"imuv2_{k}", E / f"{A.imu_v2_src}/fold{k}.pth", 8) for k in A.imu_v2_fold_ids if (E / f"{A.imu_v2_src}/fold{k}.pth").exists()]

pack, sizes = {}, {}


def add_bytes(name, b):
    pack[name] = np.frombuffer(b, np.uint8)
    sizes[name] = len(b) / 2 ** 20


for name, ck, bits in SPEC:
    sd = torch.load(ck, weights_only=True, map_location="cpu")
    p = pack_sd(sd, bits, A.vit_qgroup, A.vit_qclip) if name == "vit" and A.vit_qgroup else pack_sd(sd, bits)
    for k, v in p.items():
        pack[f"{name}/{k}"] = v
    sizes[name] = sum(v.nbytes for v in p.values()) / 2 ** 20

with open(E / (A.hgb_dir or f"imu_hgb_{A.hgb}") / "model_full.pkl", "rb") as fh:
    add_bytes("hgb_gz", gzip.compress(fh.read(), 9))
if A.hgb_feat != "v1":
    add_bytes("hgb_spec", json.dumps({"feat": A.hgb_feat, "stats": A.hgb_stats, "devs": A.hgb_devs}).encode())
add_bytes("yolo11n", (SC / "yolo11n.pt").read_bytes())
add_bytes("fusion_cfg", json.dumps(
    json.load(open(E / A.cfg))).encode())
add_bytes("branch_mods", json.dumps({"r34c5": A.r34_mods, "r34c2": A.r34c2_mods,
                                     "dir": A.dir_mods, "thermal": A.thermal_mods}).encode())
VIEWS = {"r34c5": "tta6", "r34c2": "tta6", "dir": "tta6", "thermal": "tta6", "vit": "1"}
BSPEC = {"r34c5": {"prefix": "r34c5", "arch": "r2p1d34", "mods": A.r34_mods, "frames": 16, "crop": 112},
         "r34c2": {"prefix": "r34c5", "arch": "r2p1d34", "mods": A.r34c2_mods, "frames": 16, "crop": 112},
         "dir_r2p1d": {"prefix": "dir", "arch": "r2p1d", "mods": A.dir_mods, "frames": 16, "crop": 112},
         "thermal_pl2": {"prefix": "thermal", "arch": "r2p1d", "mods": A.thermal_mods, "frames": 16, "crop": 112}}
if A.vit_src:
    BSPEC[A.vit_name] = {"prefix": "vit", "arch": A.vit_arch, "mods": A.vit_mods, "frames": 16, "crop": A.vit_crop}
if A.ir_src:
    BSPEC[A.ir_name] = {"prefix": "vit_ir", "arch": A.ir_arch, "mods": A.ir_mods, "frames": 16, "crop": A.vit_crop}
    VIEWS["vit_ir"] = "1"
add_bytes("branch_spec", json.dumps(BSPEC).encode())
if A.imu_v2_src:
    add_bytes("imu_v2_spec", A.imu_v2_spec.encode())
for kv in A.views:
    k, v = kv.split("="); assert k in VIEWS and v in ("1", "roll3", "flip2", "tta6"), kv; VIEWS[k] = v
add_bytes("branch_views", json.dumps(VIEWS).encode())
print("виды TTA:", VIEWS)
norm = np.load(E / "tcn_norm.npz")
for k in norm.files:
    pack[f"tcn_norm/{k}"] = norm[k]
sizes["tcn_norm"] = sum(norm[k].nbytes for k in norm.files) / 2 ** 20
tabs = np.load(E / "decoder_tables.npz")
for k in tabs.files:
    pack[f"tables/{k}"] = tabs[k]
if A.em_tables:
    rawt = np.load(E / A.em_tables)
    for k in ("P1i", "P2i", "P1b"):
        pack[f"tables/{k}"] = rawt[k].astype(np.float32)
    print(f"EM: сырые таблицы переходов {A.em_tables} -> tables/P1i,P2i,P1b ({sum(rawt[k].nbytes for k in ('P1i', 'P2i', 'P1b')) / 1024:.0f} КБ)")
if A.proto_feat:
    import pandas as pd
    Fo = np.load(E / A.proto_feat).astype(np.float64)
    yo = pd.read_csv(ROOT / "data/meta/clips_train.csv")["action_id"].to_numpy()
    okf = ~np.isnan(Fo).any(1)
    Fo[~okf] = 0.0; Fo /= np.linalg.norm(Fo, axis=1, keepdims=True) + 1e-9
    Cp = np.zeros((40, Fo.shape[1]))
    for k in range(40):
        sel = okf & (yo == k)
        if sel.sum():
            Cp[k] = Fo[sel].mean(0)
    Cp /= np.linalg.norm(Cp, axis=1, keepdims=True) + 1e-9
    pack["tables/proto_src"] = Cp.astype(np.float16)
    print(f"прототипы источника: {A.proto_feat} -> tables/proto_src {Cp.shape} fp16 ({Cp.size * 2 / 1024:.0f} КБ), клипов {okf.sum()}")
sizes["tables"] = sum(tabs[k].nbytes for k in tabs.files) / 2 ** 20

out = E / A.out
np.savez(out, **pack)
total = out.stat().st_size / 2 ** 20
for k, v in sizes.items():
    print(f"  {k:12s} {v:6.2f} МиБ")
dec = total * 1.048576
print(f"ИТОГО {out.name}: {total:.2f} МиБ = {dec:.2f} МБ десятичных "
      f"({'ВЛЕЗАЕТ' if dec <= 100 else 'ПРЕВЫШЕНИЕ'} консервативного лимита)")
