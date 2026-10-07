"""Score any model's out-of-fold predictions against the frozen fold split.

The contract every branch must follow
-------------------------------------
A trained branch writes one directory under experiments/:

    experiments/<name>/
        oof.npy     float32 (3036, 40)  out-of-fold probabilities, row order
                    = data/meta/clips_train.csv row order; rows the model
                    could not score (missing modality) must be all-NaN
        test.npy    float32 (405, 40)   optional, row order = test.csv

Scoring is then instant and uniform for every model and every fusion:

    python code/eval/score.py experiments/imu_minirocket
    python code/eval/score.py experiments/a experiments/b --fuse 0.6 0.4

What it reports: overall accuracy, per-fold (per held-out-user-group)
accuracy, coverage (share of clips the model scored), balanced accuracy,
majority-class baseline, and the worst confusions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

META = Path("data/meta")


def load_truth():
    clips = pd.read_csv(META / "clips_train.csv")
    folds = pd.read_csv(META / "folds.csv").set_index("user_id")["fold"]
    y = clips["action_id"].to_numpy()
    f = clips["user_id"].map(folds).to_numpy()
    return clips, y, f


def fuse(prob_list: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Weighted late fusion with per-clip renormalisation over the models
    that actually scored the clip (NaN rows = modality missing)."""
    n, c = prob_list[0].shape
    acc = np.zeros((n, c))
    wsum = np.zeros((n, 1))
    for probs, weight in zip(prob_list, weights):
        ok = ~np.isnan(probs).any(axis=1)
        acc[ok] += weight * probs[ok]
        wsum[ok, 0] += weight
    out = np.full((n, c), np.nan)
    scored = wsum[:, 0] > 0
    out[scored] = acc[scored] / wsum[scored]
    return out


def report(name: str, probs: np.ndarray, y: np.ndarray, f: np.ndarray,
           clips: pd.DataFrame) -> None:
    scored = ~np.isnan(probs).any(axis=1)
    pred = probs[scored].argmax(axis=1)
    yt = y[scored]
    acc = (pred == yt).mean()
    cover = scored.mean()

    # balanced accuracy
    classes = np.unique(yt)
    recalls = [(pred[yt == c] == c).mean() for c in classes]

    # majority baseline on the same coverage
    maj = pd.Series(yt).mode()[0]
    maj_acc = (yt == maj).mean()

    print(f"== {name}")
    print(f"   accuracy {acc:.4f}   balanced {np.mean(recalls):.4f}   "
          f"coverage {cover:.3f} ({scored.sum()}/{len(y)})   "
          f"majority-baseline {maj_acc:.4f}")
    for k in sorted(np.unique(f)):
        m = scored & (f == k)
        if m.sum() == 0:
            continue
        fa = (probs[m].argmax(axis=1) == y[m]).mean()
        print(f"   fold {k}: {fa:.4f}  (n={m.sum()})")

    # worst confusions
    names = clips.drop_duplicates("action_id").set_index("action_id")["action"]
    err = pd.DataFrame({"true": yt, "pred": pred})
    err = err[err["true"] != err["pred"]]
    top = err.value_counts().head(6)
    if len(top):
        print("   worst confusions:")
        for (t, p), cnt in top.items():
            print(f"     {cnt:>3}  {names[t]} -> {names[p]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path,
                        help="experiments/<name> dirs containing oof.npy")
    parser.add_argument("--fuse", nargs="*", type=float, default=None,
                        help="weights; if given, also score the weighted fusion")
    args = parser.parse_args()

    clips, y, f = load_truth()
    probs_all = []
    for run in args.runs:
        probs = np.load(run / "oof.npy")
        assert probs.shape == (len(y), 40), f"{run}: shape {probs.shape}"
        probs_all.append(probs)
        report(run.name, probs, y, f, clips)

    if args.fuse is not None and len(args.runs) > 1:
        weights = args.fuse or [1.0] * len(probs_all)
        fused = fuse(probs_all, weights)
        report(f"FUSION {weights}", fused, y, f, clips)
    return 0


if __name__ == "__main__":
    sys.exit(main())
