"""Build per-clip indices: clips_train.csv from the HAR.zip central directory,
clips_test.csv + availability.csv from the test archive (zip or extracted dir).

A "clip" is one (action, user, trial) in training, one SM_test_XXXX in test.
For every clip we record, per modality: number of files and total bytes.
This is the clip x modality availability matrix demanded by eda.md (1.1, 2.7)
-- computed from the archives, never hardcoded.

Usage
-----
    python code/data/index.py --train-archive data/raw/HAR.zip
    python code/data/index.py --test-zip data/raw/small_model_track_test.zip
"""

from __future__ import annotations

import argparse
import collections
import csv
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from inventory import iter_central_directory  # noqa: E402

MODALITIES = ["Depth_Color", "IR", "Thermal", "Skeleton", "IMU", "Radar"]

TRAIN_RE = re.compile(
    r"HAR/data/(?P<modality>[^/]+)/(?P<action>[^/]+)/(?P<user>[^/]+)/"
    r"(?P<trial>[^/]+)/(?:[^/]+/)?(?P<file>[^/]+)$"
)
TEST_RE = re.compile(
    r"(?:^|/)(?P<clip>[^/]+)/(?P<modality>Depth_Color|IR|Thermal|Skeleton|IMU|Radar)/"
    r"(?:[^/]+/)?(?P<file>[^/]+)$"
)


def build_train_index(archive: Path, out_csv: Path) -> None:
    files = collections.Counter()   # (action, user, trial, modality) -> n files
    nbytes = collections.Counter()
    for name, size in iter_central_directory(archive):
        m = TRAIN_RE.search(name)
        if m is None:
            continue
        key = (m.group("action"), m.group("user"), m.group("trial"),
               m.group("modality"))
        files[key] += 1
        nbytes[key] += size

    clips = sorted({k[:3] for k in files})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        header = ["action", "action_id", "user", "user_id", "trial",
                  "t1", "t2", "t3"]
        for mod in MODALITIES:
            header += [f"{mod}_files", f"{mod}_bytes"]
        writer.writerow(header)
        for action, user, trial in clips:
            action_id = int(action.split("_", 1)[0])
            user_id = int(re.search(r"\d+", user).group())
            parts = trial.split("-")
            t1, t2, t3 = (parts + ["", "", ""])[:3]
            row = [action, action_id, user, user_id, trial, t1, t2, t3]
            for mod in MODALITIES:
                key = (action, user, trial, mod)
                row += [files.get(key, 0), nbytes.get(key, 0)]
            writer.writerow(row)
    print(f"wrote {out_csv}  ({len(clips)} clips)")


def build_test_index(zip_path: Path, out_csv: Path, avail_csv: Path) -> None:
    files = collections.Counter()
    nbytes = collections.Counter()
    if Path(zip_path).is_dir():
        entries = ((p.relative_to(zip_path).as_posix(), p.stat().st_size)
                   for p in Path(zip_path).rglob("*") if p.is_file())
    else:
        with zipfile.ZipFile(zip_path) as zf:
            entries = [(i.filename, i.file_size) for i in zf.infolist()]
    for name, size in entries:
        b = name.rsplit("/", 1)[-1]
        if "__MACOSX/" in name or b.startswith("._") or b == ".DS_Store":
            continue
        m = TEST_RE.search(name)
        if m is None:
            continue
        key = (m.group("clip"), m.group("modality"))
        files[key] += 1
        nbytes[key] += size

    # clip directories with no files at all (or only junk) are still clips: index them with zero files
    empty = set()
    if Path(zip_path).is_dir():
        # a clip is a directory that is a sibling of directories holding modality folders (same parent)
        parents = {}
        for p in Path(zip_path).rglob("*"):
            if p.is_dir() and any(x.is_dir() and x.name in MODALITIES for x in p.iterdir()):
                parents[p.parent] = parents.get(p.parent, 0) + 1
        for par in parents:
            for d in par.iterdir():
                if d.is_dir() and d.name not in MODALITIES and not d.name.startswith(".") and not any(x.is_file() for x in d.rglob("*")):
                    empty.add(d.name)
    else:
        with zipfile.ZipFile(zip_path) as zf:
            infos = [i.filename for i in zf.infolist() if "__MACOSX" not in i.filename]
            dirs = [n for n in infos if n.endswith("/")]
            with_files = set()
            for n in infos:
                if not n.endswith("/"):
                    parts = n.split("/")
                    for k in range(1, len(parts)):
                        with_files.add("/".join(parts[:k]) + "/")
            dirs = [n for n in dirs if n not in with_files]  # keep only directories with no file beneath them
            clip_dirs = {n for n in dirs if TEST_RE.search(n + "x") is None and n.count("/") >= 1}
            has_mod = {n.rsplit("/", 2)[0] + "/" for n in dirs if n.rstrip("/").rsplit("/", 1)[-1] in MODALITIES}
            depth = {n.count("/") for n in has_mod}
            for n in clip_dirs:
                if n.count("/") in depth and n not in has_mod and not n.rstrip("/").rsplit("/", 1)[-1].startswith("."):
                    empty.add(n.rstrip("/").rsplit("/", 1)[-1])
    clips = sorted({k[0] for k in files} | empty)
    if empty:
        print(f"NOTE: {len(empty)} clip folder(s) without any files, indexed with zero modalities: {sorted(empty)[:5]}")
    modalities = sorted({k[1] for k in files})
    extra = [m for m in modalities if m not in MODALITIES]
    if extra:
        print(f"NOTE: test contains modalities outside the known six: {extra}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        header = ["clip"]
        for mod in modalities:
            header += [f"{mod}_files", f"{mod}_bytes"]
        writer.writerow(header)
        for clip in clips:
            row = [clip]
            for mod in modalities:
                row += [files.get((clip, mod), 0), nbytes.get((clip, mod), 0)]
            writer.writerow(row)
    print(f"wrote {out_csv}  ({len(clips)} clips)")

    with open(avail_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["clip"] + modalities)
        for clip in clips:
            writer.writerow([clip] + [int(files.get((clip, m), 0) > 0)
                                      for m in modalities])
    print(f"wrote {avail_csv}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-archive", type=Path)
    parser.add_argument("--test-zip", type=Path)
    parser.add_argument("--meta", type=Path, default=Path("data/meta"))
    args = parser.parse_args()

    if args.train_archive:
        build_train_index(args.train_archive, args.meta / "clips_train.csv")
    if args.test_zip:
        build_test_index(args.test_zip, args.meta / "clips_test.csv",
                         args.meta / "availability.csv")
    if not (args.train_archive or args.test_zip):
        parser.error("nothing to do: pass --train-archive and/or --test-zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
