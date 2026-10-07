"""Decode Depth_Color / IR / Thermal frames into per-clip uint8 arrays.

Depth_Color: rainbow palette inverted to scalar depth (colormap.py), so the
net sees geometry, not palette RGB. IR: grayscale as-is. Thermal: RGB ->
hue-inverted scalar the same way (the palette is also a rainbow ramp).
Every clip becomes data/cache/<mod>_<split>/<key>.npy of shape (T, S, S)
uint8, T = all frames of the clip in timestamp order, S = --size (128).

Run after extraction:
    python code/data/preprocess/frames.py --mod Depth_Color --split train
    python code/data/preprocess/frames.py --mod IR --split test --zip data/raw/small_model_track_test.zip
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from colormap import rgb_to_depth  # noqa: E402

import os
META = Path(os.environ.get("CUHKX_META", "data/meta"))
CACHE = Path(os.environ.get("CUHKX_CACHE", "data/cache"))


def encode(img: Image.Image, mod: str, size: int) -> np.ndarray:
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img)
    if mod == "Depth_Color" and arr.ndim == 3:
        depth, valid = rgb_to_depth(arr)
        return (depth * 255).astype(np.uint8)
    if mod == "Thermal" and arr.ndim == 3:
        # ironbow palette is monotonic in luminance -- grayscale keeps the
        # temperature ordering (hue inversion breaks: the ramp wraps red)
        lum = (0.299 * arr[..., 0] + 0.587 * arr[..., 1]
               + 0.114 * arr[..., 2])
        return lum.astype(np.uint8)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8)


def frame_sort_key(name: str):
    m = re.search(r"_(\d{8})", name)
    if m:
        return int(m.group(1))
    m = re.search(r"frame_(\d+)", name)
    return int(m.group(1)) if m else name


def build_train(mod: str, size: int) -> None:
    root = Path("data/extracted/HAR/data") / mod
    out_dir = CACHE / f"{mod}_train"
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = pd.read_csv(META / "clips_train.csv")
    todo = clips[clips[f"{mod}_files"] > 0] if f"{mod}_files" in clips.columns else clips.iloc[0:0]  # modality absent from the whole sample
    print(f"{mod} train: {len(todo)} clips")
    done = 0
    for _, r in todo.iterrows():
        key = f"{r['action']}~{r['user']}~{r['trial']}"
        out = out_dir / (key + ".npy")
        if out.exists():
            done += 1
            continue
        d = root / r["action"] / r["user"] / r["trial"]
        files = sorted((p for p in d.iterdir() if p.suffix.lower() in
                        (".png", ".jpg")), key=lambda p: frame_sort_key(p.name))
        if not files:
            continue
        arrs = []
        for p in files:
            try:
                arrs.append(encode(Image.open(p), mod, size))
            except Exception:
                pass                      # corrupt frame -- skip
        if arrs:
            np.save(out, np.stack(arrs))
        done += 1
        if done % 300 == 0:
            print(f"  {done}/{len(todo)}", flush=True)
    print(f"{mod} train done: {done}")


class DirArchive:
    """zipfile-like view of an extracted sample directory (namelist/read)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._names = [p.relative_to(self.root).as_posix()
                       for p in self.root.rglob("*") if p.is_file()]

    def namelist(self):
        return self._names

    def read(self, name: str) -> bytes:
        try:
            return (self.root / name).read_bytes()
        except FileNotFoundError:
            raise KeyError(name) from None  # same contract as zipfile.ZipFile.read


def open_archive(path: Path):
    path = Path(path)
    return DirArchive(path) if path.is_dir() else zipfile.ZipFile(path)

def is_junk(name: str) -> bool:
    """AppleDouble/Finder junk inside archives (__MACOSX/, ._*, .DS_Store)."""
    b = name.rsplit("/", 1)[-1]
    return "__MACOSX/" in name or b.startswith("._") or b == ".DS_Store"



def build_test(mod: str, size: int, zip_path: Path) -> None:
    z = open_archive(zip_path)
    out_dir = CACHE / f"{mod}_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    by_clip: dict[str, list[str]] = {}
    pat = re.compile(rf"(?:^|/)([^/]+)/{mod}/[^/]+\.(?:png|jpg)$")
    for n in z.namelist():
        if is_junk(n):
            continue
        m = pat.search(n)
        if m:
            by_clip.setdefault(m.group(1), []).append(n)
    print(f"{mod} test: {len(by_clip)} clips")
    for i, (clip, names) in enumerate(sorted(by_clip.items())):
        out = out_dir / (clip + ".npy")
        if out.exists():
            continue
        names.sort(key=frame_sort_key)
        arrs = []
        for n in names:
            try:
                arrs.append(encode(Image.open(io.BytesIO(z.read(n))), mod, size))
            except Exception:
                pass
        if arrs:
            np.save(out, np.stack(arrs))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(by_clip)}", flush=True)
    print(f"{mod} test done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mod", required=True,
                    choices=["Depth_Color", "IR", "Thermal"])
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--zip", type=Path,
                    default=Path("data/raw/small_model_track_test.zip"))
    args = ap.parse_args()
    if args.split == "train":
        build_train(args.mod, args.size)
    else:
        build_test(args.mod, args.size, args.zip)
