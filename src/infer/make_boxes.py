# -*- coding: utf-8 -*-
"""Person boxes for a raw sample archive -> data/meta/bbox_hybrid6flat_test.json
(the crop window the R34-crop6 branch was trained on).

Rule (= build_hybrid6 + hybrid6b of the Kaggle pipeline):
  YOLO11n (weights taken from the pack, key 'yolo11n') on 12 evenly spaced RAW
  IR frames per clip, imgsz 640, conf 0.25, class person, best box per frame.
    n_det >= 2 & mean conf >= 0.40  -> square(union)         src 'yolo'
    n_det >= 1 & mean conf >= 0.25  -> square(union)         src 'yolo_soft'
    else: second pass imgsz 1280, conf 0.15; >= 1 det        src 'yolo_1280'
    else: motion box from bbox_Depth_Color_test.json          src 'fb_motion'
    else: full frame                                          src 'fb_full'
  square(): margin 1.40, min side 0.35, clipped to the frame.
Kaggle test: 401/405 clips are YOLO-sourced; the 4 fallbacks used a
session-background model there (fb_h3) which this Stage-2 script replaces by
the motion box (documented deviation, <1% of clips).
"""
import argparse
import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data" / "preprocess"))
from frames import frame_sort_key, open_archive, is_junk  # noqa: E402

META = Path(os.environ.get("CUHKX_META", ROOT / "data/meta"))
N_PROBE, CONF, BATCH = 12, 0.25, 48
MARGIN, MIN_SIDE = 1.4, 0.35


def sq(y0, y1, x0, x1):
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    side = max(y1 - y0, x1 - x0) * MARGIN
    side = max(side, MIN_SIDE)
    side = min(side, 1.0)
    h = side / 2
    cy = min(max(cy, h), 1 - h)
    cx = min(max(cx, h), 1 - h)
    return [round(cy - h, 4), round(cy + h, 4), round(cx - h, 4), round(cx + h, 4)]


def sample(names):
    idx = np.unique(np.linspace(0, len(names) - 1, min(len(names), N_PROBE)).astype(int))
    return [names[i] for i in idx]


def clip_frames(z):
    pat = re.compile(r"(?:^|/)([^/]+)/IR/[^/]+\.(?:png|jpg)$")
    by = {}
    for n in z.namelist():
        if is_junk(n):
            continue
        m = pat.search(n)
        if m:
            by.setdefault(m.group(1), []).append(n)
    for clip, names in sorted(by.items()):
        names.sort(key=frame_sort_key)
        frs = []
        for n in sample(names):
            try:
                frs.append(np.asarray(Image.open(io.BytesIO(z.read(n))).convert("L")))
            except Exception:  # noqa: BLE001
                pass
        yield clip, frs


try:
    import torch as _torch
    YOLO_DEVICE = 0 if (_torch.cuda.is_available() and _torch.cuda.device_count() > 0) else "cpu"
except Exception:  # noqa: BLE001
    YOLO_DEVICE = "cpu"


def detect(model, frames, imgsz, conf):
    dets = model.predict([np.stack([f] * 3, -1) for f in frames], classes=[0], conf=conf,
                         imgsz=imgsz, verbose=False, device=YOLO_DEVICE)
    boxes = []
    for r in dets:
        if len(r.boxes):
            h, w = r.orig_shape
            j = int(r.boxes.conf.argmax())
            x0, y0, x1, y1 = r.boxes.xyxy[j].tolist()
            boxes.append([y0 / h, y1 / h, x0 / w, x1 / w, float(r.boxes.conf[j])])
    return boxes


def union(boxes):
    b = np.array([d[:4] for d in boxes])
    return sq(float(b[:, 0].min()), float(b[:, 1].max()), float(b[:, 2].min()), float(b[:, 3].max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--pack", type=Path, required=True, help="pack npz holding key 'yolo11n' (yolo11n.pt bytes)")
    ap.add_argument("--motion-json", type=Path, default=None,
                    help="fallback boxes (bbox_Depth_Color_test.json); default META/bbox_Depth_Color_test.json")
    args = ap.parse_args()
    from ultralytics import YOLO  # noqa: E402
    z = np.load(args.pack)
    with tempfile.TemporaryDirectory() as td:
        wp = Path(td) / "yolo11n.pt"
        wp.write_bytes(bytes(z["yolo11n"]))
        model = YOLO(str(wp))
    motion_p = args.motion_json or (META / "bbox_Depth_Color_test.json")
    motion = json.load(open(motion_p)) if motion_p.exists() else {}
    arc = open_archive(args.src)
    out, flat, n_src = {}, {}, {}
    pending = []           # (clip, frames) batched for the 640 pass

    def finish(clip, frames, boxes):
        tight = boxes
        if len(boxes) >= 2 and np.mean([d[4] for d in boxes]) >= 0.40:
            box, src = union(boxes), "yolo"
        elif len(boxes) >= 1 and np.mean([d[4] for d in boxes]) >= 0.25:
            box, src = union(boxes), "yolo_soft"
        else:
            b2 = detect(model, frames, 1280, 0.15) if frames else []
            if b2:
                box, src = union(b2), "yolo_1280"
                tight = b2
            elif clip in motion:
                box, src = motion[clip], "fb_motion"
            else:
                box, src = [0.0, 1.0, 0.0, 1.0], "fb_full"
        # per-coordinate median of the tight per-frame detections (used by the upper-body ROI crop)
        med = [float(np.median([d[i] for d in tight])) for i in range(4)] if tight else None
        out[clip] = {"box": box, "src": src, "n_det": len(boxes),
                     "conf": float(np.mean([d[4] for d in boxes])) if boxes else 0.0,
                     "median": med, "n_tight": len(tight)}
        flat[clip] = box
        n_src[src] = n_src.get(src, 0) + 1

    n = 0
    for clip, frames in clip_frames(arc):
        boxes = detect(model, frames, 640, CONF) if frames else []
        finish(clip, frames, boxes)
        n += 1
        if n % 100 == 0:
            print(f"  {n} clips", flush=True)
    META.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(META / "bbox_hybrid6_test.json", "w"))
    json.dump(flat, open(META / "bbox_hybrid6flat_test.json", "w"))
    print(f"boxes: {len(out)} clips; sources {n_src} -> {META / 'bbox_hybrid6flat_test.json'}", flush=True)


if __name__ == "__main__":
    main()
