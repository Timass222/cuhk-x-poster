"""The validation split: GroupKFold by user, fixed forever.

18 training users -> 5 folds, balanced by clip count (users are assigned
greedily, largest first, to the currently lightest fold). Deterministic:
depends only on data/meta/clips_train.csv. The assignment is written to
data/meta/folds.csv and MUST NOT change for the rest of the competition --
every model's out-of-fold predictions must be comparable.

Why GroupKFold by user: the test subjects (10, 11, 25, 26 + 8 private) are
disjoint from training subjects (verified by inventory.py -- no overlap).
The public 0.716 notebook showed subject-held-out CV predicts LB to the third
decimal. Random splits leak subject identity and overestimate wildly.

Usage:
    python code/train/cv.py            # writes data/meta/folds.csv if absent
    python code/train/cv.py --force    # rewrite (only before any model runs!)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

META = Path("data/meta")
FOLDS_CSV = META / "folds.csv"
N_FOLDS = 5


def make_folds(clips_csv: Path = META / "clips_train.csv") -> pd.DataFrame:
    df = pd.read_csv(clips_csv)
    per_user = df.groupby("user_id").size().sort_values(ascending=False)
    fold_load = [0] * N_FOLDS
    fold_users: list[list[int]] = [[] for _ in range(N_FOLDS)]
    for user, count in per_user.items():        # largest user first
        lightest = min(range(N_FOLDS), key=lambda i: (fold_load[i], i))
        fold_users[lightest].append(int(user))
        fold_load[lightest] += int(count)
    rows = [
        {"user_id": u, "fold": i}
        for i, users in enumerate(fold_users)
        for u in sorted(users)
    ]
    out = pd.DataFrame(rows).sort_values(["fold", "user_id"]).reset_index(drop=True)
    return out, fold_load


def load_folds() -> pd.Series:
    """user_id -> fold. The one true split; import this everywhere."""
    return pd.read_csv(FOLDS_CSV).set_index("user_id")["fold"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if FOLDS_CSV.exists() and not args.force:
        print(f"{FOLDS_CSV} already exists -- the split is frozen. "
              "Use --force only if no model has trained yet.")
        return 0

    folds, load = make_folds()
    folds.to_csv(FOLDS_CSV, index=False)
    print(f"wrote {FOLDS_CSV}")
    for i in range(N_FOLDS):
        users = folds[folds["fold"] == i]["user_id"].tolist()
        print(f"  fold {i}: users {users}  ({load[i]} clips)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
