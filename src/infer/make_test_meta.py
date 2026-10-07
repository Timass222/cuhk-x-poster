# -*- coding: utf-8 -*-
"""Stage-2 / test meta from a raw sample archive (zip or extracted dir):
  data/meta/clips_test.csv      (clip x modality files/bytes; via code/data/index.py)
  data/meta/sessions_test.csv   (clip, first_ts, session)
first_ts = earliest timestamp parsed from Depth_Color / IR frame names
(Depth_YYYY-MM-DD_HH-MM-SS.mmm_idx_Color.png, IR_..._idx.png); fallback: Radar
file name radar_output_T<ts>.csv; fallback: Skeleton json names; last resort: the
`time` column inside the IMU csv files (content, not names). Session = new
when the gap between consecutive clips (sorted by first_ts) exceeds --gap
seconds (1800 = rule that reproduces the 16 Kaggle test sessions). Clips with
no timestamp at all get their own singleton sessions (decoder falls back to
per-clip argmax there). Verified against the Kaggle test zip: identical
partition."""
import argparse
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data"))
sys.path.insert(0, str(ROOT / "src" / "data" / "preprocess"))
from frames import is_junk  # noqa: E402

TS = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})\.(\d{3})")
CLIP_RE = re.compile(r"^(?P<pre>(?:[^/]+/)*?)(?P<clip>[^/]+)/(?P<mod>Depth_Color|IR|Radar|Skeleton|Thermal|IMU)/(?:[^/]+/)?(?P<file>[^/]+)$")


def parse_ts(name):
    m = TS.search(name)
    if not m:
        return None
    d, hh, mm, ss, ms = m.groups()
    return pd.Timestamp(f"{d} {hh}:{mm}:{ss}.{ms}")


def imu_content_ts(src: Path, name: str, zf=None):
    """Earliest parseable value of the `time` column in one IMU csv (e.g. `2025-6-17 14:34:47.47`).
    `zf`: an already open ZipFile when `src` is an archive (avoids re-reading the central directory)."""
    try:
        if src.is_dir():
            with open(src / name, "r", encoding="utf-8", errors="ignore") as f:
                head = [f.readline() for _ in range(4)]
        else:
            z = zf or zipfile.ZipFile(src)
            with z.open(name) as f:
                head = [f.readline().decode("utf-8", "ignore") for _ in range(4)]
            if zf is None:
                z.close()
    except Exception:
        return None
    # header may be English (`time`) or Chinese (`时间`): do not rely on it, parse the first cells
    best = None
    for line in head:
        cell = line.split(",")[0].strip()
        if not cell:
            continue
        try:
            ts = pd.Timestamp(cell)
        except Exception:
            continue
        if pd.isna(ts):
            continue
        if best is None or ts < best:
            best = ts
    return best


def iter_names(src: Path):
    if src.is_dir():
        for p in src.rglob("*"):
            if p.is_file():
                yield p.relative_to(src).as_posix()
    else:
        with zipfile.ZipFile(src) as z:
            for n in z.namelist():
                if not n.endswith("/"):
                    yield n


def build_sessions(src: Path, clips, gap: float):
    first, paths, imu_files = {}, {}, {}
    prio = {"Depth_Color": 0, "IR": 0, "Radar": 1, "Skeleton": 2, "Thermal": 9, "IMU": 9}
    for n in iter_names(src):
        if is_junk(n):
            continue
        m = CLIP_RE.match(n)
        if not m or m.group("clip") not in clips:
            continue
        paths.setdefault(m.group("clip"), m.group("pre") + m.group("clip") + "/")
        if m.group("mod") == "IMU" and m.group("file").lower().endswith(".csv"):
            imu_files.setdefault(m.group("clip"), []).append(n)
        if prio[m.group("mod")] >= 9:
            continue
        ts = parse_ts(m.group("file"))
        if ts is None:
            continue
        key = m.group("clip")
        pr = prio[m.group("mod")]
        cur = first.get(key)
        if cur is None or pr < cur[0] or (pr == cur[0] and ts < cur[1]):
            first[key] = (pr, ts)
    # last resort (names carry no timestamp at all): first `time` cell of the IMU csv content
    todo = [c for c in clips if c not in first and c in imu_files]
    zf = None if (src.is_dir() or not todo) else zipfile.ZipFile(src)
    for c in todo:
        for n in imu_files[c]:
            ts = imu_content_ts(src, n, zf)
            if ts is not None and (c not in first or ts < first[c][1]):
                first[c] = (3, ts)
    if zf is not None:
        zf.close()
    rows = [(c, first[c][1]) for c in clips if c in first]
    missing = [c for c in clips if c not in first]
    # deterministic order: ties on first_ts (clips sharing a timestamp) broken by clip name
    df = pd.DataFrame(rows, columns=["clip", "first_ts"]).sort_values(["first_ts", "clip"], kind="stable").reset_index(drop=True)
    gaps = pd.to_datetime(df["first_ts"]).diff().dt.total_seconds().fillna(0.0)
    df["session"] = (gaps > gap).cumsum().astype(int)
    nxt = int(df["session"].max()) + 1 if len(df) else 0
    for c in missing:
        df.loc[len(df)] = [c, pd.NaT, nxt]
        nxt += 1
    # клипы без таймстампа: синтетические метки далеко от остальных -> одиночные сессии
    if missing:
        base = pd.to_datetime(df["first_ts"]).max() if len(df) - len(missing) else pd.Timestamp("2000-01-01")
        for i, c in enumerate(missing):
            df.loc[df["clip"] == c, "first_ts"] = base + pd.Timedelta(days=1, hours=i)
    df["first_ts"] = df["first_ts"].apply(lambda t: pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%S.%f"))
    return df, missing, paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="sample zip or extracted directory")
    ap.add_argument("--meta", type=Path, default=ROOT / "data/meta")
    ap.add_argument("--gap", type=float, default=1800.0)
    ap.add_argument("--check", action="store_true", help="compare with existing meta instead of writing")
    args = ap.parse_args()
    from index import build_test_index  # noqa: E402
    out_clips = args.meta / ("clips_test.check.csv" if args.check else "clips_test.csv")
    out_avail = args.meta / ("availability.check.csv" if args.check else "availability.csv")
    build_test_index(args.src, out_clips, out_avail)
    clips = pd.read_csv(out_clips)["clip"].tolist()
    df, missing, paths = build_sessions(args.src, set(clips), args.gap)
    out_s = args.meta / ("sessions_test.check.csv" if args.check else "sessions_test.csv")
    df.to_csv(out_s, index=False)
    # a clip without any file has no derived path: use the prefix shared by the other clips (Kaggle layout <prefix>/<clip>/)
    prefixes = [v[: -len(c) - 1] for c, v in paths.items() if v.endswith(c + "/")]
    common = max(set(prefixes), key=prefixes.count) if prefixes else ""
    pd.DataFrame({"clip": clips, "path": [paths.get(c, common + c + "/") for c in clips]}).to_csv(
        args.meta / ("test_paths.check.csv" if args.check else "test_paths.csv"), index=False)
    print(f"clips {len(clips)} | sessions {df['session'].nunique()} | clips without timestamp {len(missing)} -> {out_s}")
    if args.check:
        a = pd.read_csv(args.meta / "clips_test.csv"); b = pd.read_csv(out_clips)
        print("clips_test identical:", a.equals(b))
        sa = pd.read_csv(args.meta / "sessions_test.csv"); sb = pd.read_csv(out_s)
        ma = sa.set_index("clip")["session"]; mb = sb.set_index("clip")["session"].loc[ma.index]
        pairs = pd.crosstab(ma, mb)
        same_part = (pairs.astype(bool).sum(1) == 1).all() and (pairs.astype(bool).sum(0) == 1).all()
        same_ids = bool((ma == mb).all())
        ta = pd.to_datetime(sa.set_index("clip")["first_ts"]); tb = pd.to_datetime(sb.set_index("clip")["first_ts"]).loc[ta.index]
        print("session partition identical:", same_part, "| ids identical:", same_ids, "| first_ts identical:", bool((ta == tb).all()))
