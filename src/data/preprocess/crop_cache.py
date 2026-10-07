"""Cropped per-clip frame caches: apply the per-clip person box to the
ORIGINAL frames (640x480 / 320x240), then resize to --size. This is where
the crop pays: the person goes from ~4% of the pixels to ~15-20%.

Boxes come from data/meta/bbox_<bbox-mod>_<split>.json (fractions).
IR has no boxes of its own -- pass --bbox-mod Depth_Color (same camera).
Clips without a box fall back to the full frame.

    python code/data/preprocess/crop_cache.py --mod Depth_Color --split train
    python code/data/preprocess/crop_cache.py --mod IR --bbox-mod Depth_Color --split train
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from colormap import rgb_to_depth  # noqa: E402
from frames import encode, frame_sort_key, open_archive, is_junk  # noqa: E402

import os
META = Path(os.environ.get("CUHKX_META", "data/meta"))
CACHE = Path(os.environ.get("CUHKX_CACHE", "data/cache"))


def encode_rgb(img: Image.Image, mod: str, size: int) -> np.ndarray:
    """Raw colormap RGB, no hue decoding: (S, S, 3) uint8.

    The E290 lesson: feeding the palette as RGB lets a video backbone's
    pretrained RGB stem see it with its original kernels intact.
    """
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def crop_img(img: Image.Image, bb) -> Image.Image:
    if bb is None:
        return img
    w, h = img.size
    y0, y1, x0, x1 = bb
    return img.crop((round(x0 * w), round(y0 * h),
                     round(x1 * w), round(y1 * h)))


def build_train(mod: str, bbox_mod: str, size: int,
                shard: int = 0, nshards: int = 1, rgb: bool = False,
                bbox_json: str | None = None, tag: str = "") -> None:
    boxes = json.load(open(bbox_json or META / f"bbox_{bbox_mod}_train.json"))
    root = Path("data/extracted/HAR/data") / mod
    enc = encode_rgb if rgb else encode
    out_dir = CACHE / f"{mod}{'RGB' if rgb else ''}crop{tag}_train"
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = pd.read_csv(META / "clips_train.csv")
    todo = clips[clips[f"{mod}_files"] > 0] if f"{mod}_files" in clips.columns else clips.iloc[0:0]  # modality absent from the whole sample
    print(f"{mod} crop train: {len(todo)} clips")
    for i, (_, r) in enumerate(todo.iterrows()):
        if i % nshards != shard:
            continue
        key = f"{r['action']}~{r['user']}~{r['trial']}"
        out = out_dir / (key + ".npy")
        if out.exists():
            continue
        bb = boxes.get(key)
        d = root / r["action"] / r["user"] / r["trial"]
        files = sorted((p for p in d.iterdir()
                        if p.suffix.lower() in (".png", ".jpg")),
                       key=lambda p: frame_sort_key(p.name))
        arrs = []
        for p in files:
            try:
                arrs.append(enc(crop_img(Image.open(p), bb), mod, size))
            except Exception:
                pass
        if arrs:
            np.save(out, np.stack(arrs))
        if (i + 1) % 300 == 0:
            print(f"  {i+1}/{len(todo)}", flush=True)
    print(f"{mod} crop train done")


def build_test(mod: str, bbox_mod: str, size: int, zip_path: Path,
               shard: int = 0, nshards: int = 1, rgb: bool = False,
               bbox_json: str | None = None, tag: str = "") -> None:
    boxes = json.load(open(bbox_json or META / f"bbox_{bbox_mod}_test.json"))
    z = open_archive(zip_path)
    enc = encode_rgb if rgb else encode
    out_dir = CACHE / f"{mod}{'RGB' if rgb else ''}crop{tag}_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    pat = re.compile(rf"(?:^|/)([^/]+)/{mod}/[^/]+\.(?:png|jpg)$")
    by_clip: dict[str, list[str]] = {}
    for n in z.namelist():
        if is_junk(n):
            continue
        m = pat.search(n)
        if m:
            by_clip.setdefault(m.group(1), []).append(n)
    print(f"{mod} crop test: {len(by_clip)} clips")
    for i, (clip, names) in enumerate(sorted(by_clip.items())):
        if i % nshards != shard:
            continue
        out = out_dir / (clip + ".npy")
        if out.exists():
            continue
        bb = boxes.get(clip)
        names.sort(key=frame_sort_key)
        arrs = []
        for n in names:
            try:
                img = Image.open(io.BytesIO(z.read(n)))
                arrs.append(enc(crop_img(img, bb), mod, size))
            except Exception:
                pass
        if arrs:
            np.save(out, np.stack(arrs))
    print(f"{mod} crop test done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mod", required=True)
    ap.add_argument("--bbox-mod", default=None)
    ap.add_argument("--split", required=True)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--zip", type=Path,
                    default=Path("data/raw/small_model_track_test.zip"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--rgb", action="store_true",
                    help="raw colormap RGB (no hue decode), 3-channel cache")
    ap.add_argument("--bbox-json", default=None,
                    help="override box file (e.g. bbox_hybrid3_train.json)")
    ap.add_argument("--tag", default="",
                    help="suffix for the cache dir name (e.g. '2')")
    a = ap.parse_args()
    bbox_mod = a.bbox_mod or a.mod
    if a.split == "train":
        build_train(a.mod, bbox_mod, a.size, a.shard, a.nshards, a.rgb,
                    a.bbox_json, a.tag)
    else:
        build_test(a.mod, bbox_mod, a.size, a.zip, a.shard, a.nshards, a.rgb,
                   a.bbox_json, a.tag)
