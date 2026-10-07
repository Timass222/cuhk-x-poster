"""Late fusion over per-branch probability matrices, missing-modality aware.

Design (why late fusion first)
------------------------------
Branches are unimodal and produce calibrated per-clip probabilities. Fusion
then happens in probability space:

    p_fused(clip) = sum_m w_m * p_m(clip)  /  sum_{m present} w_m

renormalised over the modalities actually present for that clip -- this is
the honest way to handle "not every clip has every modality" without
imputation. All fusion decisions (weights, temperatures, prior correction)
are fit on OUT-OF-FOLD predictions, never on training folds, so the local
CV estimate of the fused model stays unbiased.

Pipeline pieces, each optional, applied in this order:
  1. temperature per branch  -- fixes over/under-confidence (fit on OOF NLL)
  2. logit adjustment        -- subtract log(train prior): the train set is
                                imbalanced 12..365 clips/class, test is ~10
                                per class; branches inherit the skew
  3. weights on the simplex  -- coordinate ascent on OOF accuracy
  4. Sinkhorn balancing      -- optional, test-time only: push the predicted
                                class histogram toward uniform (405/40 ~ 10)

Usage from python (the scoring CLI stays in code/eval/score.py):

    from models.fusion import fit_fusion, apply_fusion
    fus = fit_fusion({"skel": oof_a, "imu": oof_b}, y)
    fused_test = apply_fusion(fus, {"skel": test_a, "imu": test_b})
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def _nan_rows(p: np.ndarray) -> np.ndarray:
    return np.isnan(p).any(axis=1)


def temperature_scale(probs: np.ndarray, y: np.ndarray,
                      grid=np.linspace(0.25, 4.0, 46)) -> float:
    """Pick T minimising NLL of probs**(1/T) (renormalised) on scored rows."""
    ok = ~_nan_rows(probs)
    p = np.clip(probs[ok], EPS, 1.0)
    yt = y[ok]
    best_t, best_nll = 1.0, np.inf
    logp = np.log(p)
    for t in grid:
        q = np.exp(logp / t)
        q /= q.sum(axis=1, keepdims=True)
        nll = -np.log(np.clip(q[np.arange(len(yt)), yt], EPS, 1)).mean()
        if nll < best_nll:
            best_t, best_nll = t, nll
    return float(best_t)


def apply_temperature(probs: np.ndarray, t: float) -> np.ndarray:
    out = np.full_like(probs, np.nan)
    ok = ~_nan_rows(probs)
    q = np.exp(np.log(np.clip(probs[ok], EPS, 1)) / t)
    out[ok] = q / q.sum(axis=1, keepdims=True)
    return out


def logit_adjust(probs: np.ndarray, train_prior: np.ndarray,
                 tau: float = 1.0) -> np.ndarray:
    """Divide by train prior^tau (in prob space) -- balanced posterior."""
    out = np.full_like(probs, np.nan)
    ok = ~_nan_rows(probs)
    q = np.clip(probs[ok], EPS, 1) / np.power(train_prior + EPS, tau)
    out[ok] = q / q.sum(axis=1, keepdims=True)
    return out


def weighted_fuse(prob_list, weights) -> np.ndarray:
    n, c = prob_list[0].shape
    acc = np.zeros((n, c))
    wsum = np.zeros(n)
    for p, w in zip(prob_list, weights):
        ok = ~_nan_rows(p)
        acc[ok] += w * p[ok]
        wsum[ok] += w
    out = np.full((n, c), np.nan)
    scored = wsum > 0
    out[scored] = acc[scored] / wsum[scored, None]
    return out


def geometric_fuse(prob_list, weights, beta: float = 0.0) -> np.ndarray:
    """Weighted power-mean pooling. beta=0 -> geometric (log-space) mean:
    optimal for quasi-independent experts; beta=1 -> arithmetic (see
    weighted_fuse). Missing-modality aware like weighted_fuse."""
    n, c = prob_list[0].shape
    acc = np.zeros((n, c))
    wsum = np.zeros(n)
    for p, w in zip(prob_list, weights):
        if w <= 0:
            continue
        ok = ~_nan_rows(p)
        q = np.clip(p[ok], EPS, 1.0)
        acc[ok] += w * (np.log(q) if beta == 0 else np.power(q, beta))
        wsum[ok] += w
    out = np.full((n, c), np.nan)
    scored = wsum > 0
    m = acc[scored] / wsum[scored, None]
    q = np.exp(m) if beta == 0 else np.power(np.clip(m, EPS, None), 1 / beta)
    out[scored] = q / q.sum(axis=1, keepdims=True)
    return out


def balanced_acc(pred, yt):
    classes = np.unique(yt)
    return float(np.mean([(pred[yt == c] == c).mean() for c in classes]))


def fit_weights(prob_list, y, iters: int = 60, seed: int = 42,
                metric: str = "balanced") -> np.ndarray:
    """Coordinate ascent over the weight simplex. metric="balanced" targets
    mean per-class recall -- the honest proxy for the ~uniform test prior;
    "plain" targets raw OOF accuracy (imbalanced train prior)."""
    rng = np.random.default_rng(seed)
    k = len(prob_list)
    w = np.ones(k) / k

    def acc(weights):
        f = weighted_fuse(prob_list, weights)
        ok = ~_nan_rows(f)
        if metric == "balanced":
            return balanced_acc(f[ok].argmax(1), y[ok])
        return (f[ok].argmax(1) == y[ok]).mean()

    best = acc(w)
    for _ in range(iters):
        i = rng.integers(k)
        for step in (0.25, 0.1, 0.05, -0.25, -0.1, -0.05):
            trial = w.copy()
            trial[i] = np.clip(trial[i] + step, 0.0, None)
            if trial.sum() == 0:
                continue
            trial /= trial.sum()
            a = acc(trial)
            if a > best:
                w, best = trial, a
                break
    return w


def sinkhorn_balance(probs: np.ndarray, target_prior=None,
                     n_iter: int = 30) -> np.ndarray:
    """Test-time only: rescale columns so the soft class mass matches the
    target prior (uniform by default), keeping rows normalised."""
    ok = ~_nan_rows(probs)
    p = np.clip(probs[ok], EPS, 1).copy()
    n, c = p.shape
    target = (np.ones(c) / c if target_prior is None
              else np.asarray(target_prior, float))
    for _ in range(n_iter):
        p /= p.sum(axis=1, keepdims=True)
        col = p.sum(axis=0) / n
        p *= (target / np.clip(col, EPS, None))
    p /= p.sum(axis=1, keepdims=True)
    out = np.full_like(probs, np.nan)
    out[ok] = p
    return out


def fit_fusion(oof: dict[str, np.ndarray], y: np.ndarray,
               train_prior: np.ndarray | None = None,
               adjust: set[str] | None = None,
               metric: str = "balanced") -> dict:
    """Fit temperatures + weights on OOF. Returns a config dict.

    `adjust` names the branches whose probabilities still carry the train
    prior (e.g. sklearn models fit on the imbalanced set) -- ONLY those get
    logit adjustment. Branches trained with balanced softmax are already
    uniform-prior; adjusting them twice destroys them (measured: 0.53->0.38).
    """
    names = sorted(oof)
    adjust = adjust or set()
    temps = {m: temperature_scale(oof[m], y) for m in names}
    cal = []
    for m in names:
        p = apply_temperature(oof[m], temps[m])
        if m in adjust and train_prior is not None:
            p = logit_adjust(p, train_prior)
        cal.append(p)
    w = fit_weights(cal, y, metric=metric)
    return {"names": names, "temps": temps,
            "weights": {m: float(wi) for m, wi in zip(names, w)},
            "train_prior": None if train_prior is None else list(map(float, train_prior)),
            "adjust": sorted(adjust)}


def apply_fusion(cfg: dict, probs: dict[str, np.ndarray]) -> np.ndarray:
    names = cfg["names"]
    cal = []
    for m in names:
        p = apply_temperature(probs[m], cfg["temps"][m])
        if m in set(cfg.get("adjust", [])):
            p = logit_adjust(p, np.asarray(cfg["train_prior"]))
        cal.append(p)
    return weighted_fuse(cal, [cfg["weights"][m] for m in names])
