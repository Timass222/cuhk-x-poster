"""IMU branch v1: summary-statistics features + gradient boosting.

Day-one calibration model (exploration, not the submission): 450 stats
features per clip (see preprocess/imu.py), HistGradientBoosting per fold of
the frozen GroupKFold split. Writes the standard OOF contract:

    experiments/imu_stats/oof.npy   (3036, 40) NaN rows = clip has no IMU
    experiments/imu_stats/test.npy  (405, 40)  model refit on all folds

Run:  python code/train/train_imu_quick.py
Score: python code/eval/score.py experiments/imu_stats
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

META = Path("data/meta")
CACHE = Path("data/cache")
OUT = Path("experiments/imu_stats")


def main() -> None:
    clips = pd.read_csv(META / "clips_train.csv")
    folds = pd.read_csv(META / "folds.csv").set_index("user_id")["fold"]
    f = clips["user_id"].map(folds).to_numpy()
    y = clips["action_id"].to_numpy()
    X = np.load(CACHE / "imu_train_feat.npy")
    X_test = np.load(CACHE / "imu_test_feat.npy")

    has_imu = ~np.isnan(X).all(axis=1)
    print(f"clips with IMU: {has_imu.sum()}/{len(y)}")

    oof = np.full((len(y), 40), np.nan, np.float32)
    for k in sorted(np.unique(f)):
        tr = (f != k) & has_imu
        te = (f == k) & has_imu
        clf = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.08, max_depth=None,
            l2_regularization=1.0, random_state=42)
        clf.fit(X[tr], y[tr])
        prob = clf.predict_proba(X[te])
        cols = clf.classes_
        oof[np.where(te)[0][:, None], cols[None, :]] = prob.astype(np.float32)
        acc = (prob.argmax(1) == pd.Series(y[te]).map(
            {c: i for i, c in enumerate(cols)}).to_numpy()).mean()
        print(f"fold {k}: n={te.sum()}  acc={acc:.4f}")

    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.08, l2_regularization=1.0,
        random_state=42)
    clf.fit(X[has_imu], y[has_imu])
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
