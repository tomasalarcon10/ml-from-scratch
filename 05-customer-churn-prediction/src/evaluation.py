"""
Evaluation written from scratch: precision-recall curve, average precision, gains and lift,
profit curve for a retention campaign and permutation importance.
"""

import numpy as np


def pr_curve(y, score):
    order = np.argsort(-score, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    precision = tp / np.arange(1, len(ys) + 1)
    recall = tp / ys.sum()
    return np.r_[1.0, precision], np.r_[0.0, recall], score[order]


def average_precision(y, score):
    precision, recall, _ = pr_curve(y, score)
    return float(np.sum(np.diff(recall) * precision[1:]))


def at_threshold(y, score, t):
    pred = score >= t
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
    return {"threshold": float(t), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(1e-12, precision + recall), "accuracy": (tp + tn) / len(y)}


def gains(y, score, points=101):
    """Share of all churners captured when contacting the top x% riskiest customers."""
    ys = y[np.argsort(-score, kind="mergesort")]
    share = np.linspace(0, 1, points)
    captured = np.array([ys[: int(round(s * len(ys)))].sum() for s in share]) / ys.sum()
    return share, captured


def profit_curve(y, score, value_saved, contact_cost, success_rate):
    """
    Campaign profit when contacting the top k customers by risk, for every k.
    A contacted churner is retained with probability success_rate and saves value_saved;
    every contact costs contact_cost.
    """
    ys = y[np.argsort(-score, kind="mergesort")]
    k = np.arange(0, len(ys) + 1)
    churners_reached = np.r_[0, np.cumsum(ys)]
    profit = churners_reached * success_rate * value_saved - k * contact_cost
    return k, profit


def permutation_importance(predict, X, y, metric, n_repeats=10, seed=0):
    """Drop in the metric when one column is shuffled, breaking its link with the target."""
    rng = np.random.default_rng(seed)
    base = metric(y, predict(X))
    out = np.zeros((X.shape[1], n_repeats))
    for j in range(X.shape[1]):
        for r in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            out[j, r] = base - metric(y, predict(Xp))
    return out.mean(1), out.std(1)
