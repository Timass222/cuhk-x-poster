"""Radar branch: point-cloud summary features + gradient boosting.

Mirrors train_imu_quick.py. Coverage ~48% of clips (empty CSVs stay NaN and
are handled by the fusion NaN contract). Writes the standard OOF contract:

    experiments/radar_hgb/oof.npy   (3036, 40) NaN rows = no radar points
    experiments/radar_hgb/test.npy  (405, 40)

Run:  python code/train/train_radar_hgb.py
Score: python code/eval/score.py experiments/radar_hgb
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).parents[1] / "data" / "preprocess"))
from radar import clip_features, N_FEAT  # noqa: E402

META = Path("data/meta")
CACHE = Path("data/cache")
OUT = Path("experiments/radar_hgb")
TRAIN_ROOT = Path("data/extracted/HAR/data/Radar")
TEST_ZIP = Path("data/raw/small_model_track_test.zip")


def build_features() -> tuple[np.ndarray, np.ndarray]:
    tr_p, te_p = CACHE / "radar_train_feat.npy", CACHE / "radar_test_feat.npy"
    if tr_p.is_file() and te_p.is_file():
        return np.load(tr_p), np.load(te_p)

    clips = pd.read_csv(META / "clips_train.csv")
    X = np.full((len(clips), N_FEAT), np.nan, np.float32)
    for i, row in clips.iterrows():
        d = TRAIN_ROOT / row["action"] / row["user"] / row["trial"]
        files = sorted(d.glob("radar_output_*.csv")) if d.is_dir() else []
        if files:
            X[i] = clip_features(files[0].read_text(encoding="utf-8-sig",
                                                    errors="replace"))
        if (i + 1) % 500 == 0:
            print(f"  train {i + 1}/{len(clips)}")

    test_clips = sorted(pd.read_csv(META / "clips_test.csv")["clip"])
    Xt = np.full((len(test_clips), N_FEAT), np.nan, np.float32)
    with zipfile.ZipFile(TEST_ZIP) as zf:
        by_clip = {}
        for n in zf.namelist():
            parts = n.split("/")
            if (len(parts) >= 4 and parts[2] == "Radar"
                    and parts[-1].endswith(".csv")):
                by_clip.setdefault(parts[1], []).append(n)
        for i, clip in enumerate(test_clips):
            names = sorted(by_clip.get(clip, []))
            if names:
                Xt[i] = clip_features(
                    zf.read(names[0]).decode("utf-8-sig", "replace"))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(tr_p, X)
    np.save(te_p, Xt)
    return X, Xt


def main() -> None:
    clips = pd.read_csv(META / "clips_train.csv")
    folds = pd.read_csv(META / "folds.csv").set_index("user_id")["fold"]
    f = clips["user_id"].map(folds).to_numpy()
    y = clips["action_id"].to_numpy()
    X, X_test = build_features()

    has = ~np.isnan(X).all(axis=1)
    print(f"clips with radar: {has.sum()}/{len(y)}")

    oof = np.full((len(y), 40), np.nan, np.float32)
    for k in sorted(np.unique(f)):
        tr = (f != k) & has
        te = (f == k) & has
        clf = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.08, l2_regularization=1.0,
            random_state=42)
        clf.fit(X[tr], y[tr])
        prob = clf.predict_proba(X[te])
        cols = clf.classes_
        oof[np.where(te)[0][:, None], cols[None, :]] = prob.astype(np.float32)
        acc = (cols[prob.argmax(1)] == y[te]).mean()
        print(f"fold {k}: n={te.sum()}  acc={acc:.4f}")

    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.08, l2_regularization=1.0,
        random_state=42)
    clf.fit(X[has], y[has])
    test = np.full((len(X_test), 40), np.nan, np.float32)
    has_t = ~np.isnan(X_test).all(axis=1)
    prob = clf.predict_proba(X_test[has_t])
    test[np.where(has_t)[0][:, None], clf.classes_[None, :]] = prob
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "oof.npy", oof)
    np.save(OUT / "test.npy", test)
    print(f"wrote {OUT}/oof.npy and test.npy")


if __name__ == "__main__":
    main()
