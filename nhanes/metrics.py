"""Metrics, threshold selection, bootstrap intervals and calibration. Pure numpy / sklearn."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def threshold_for_recall(y: np.ndarray, p: np.ndarray, target: float) -> float:
    """Highest threshold t such that flagging p >= t catches at least `target` of the positives.

    Computed on the data passed in. The caller must pass validation data, never test data.
    """
    pos = np.sort(np.asarray(p)[np.asarray(y) == 1])
    if len(pos) == 0:
        raise ValueError("no positives to set a recall threshold on")
    # the small epsilon guards float error, e.g. (1 - 0.8) * 100 = 19.999999999999996
    k = int(np.floor((1.0 - target) * len(pos) + 1e-9))
    return float(pos[k])


def confusion(y: np.ndarray, p: np.ndarray, t: float) -> dict[str, int]:
    pred = (np.asarray(p) >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def threshold_metrics(y: np.ndarray, p: np.ndarray, t: float) -> dict[str, float]:
    """Accuracy, precision, recall, F1 and the cost of false alarms at threshold t."""
    y = np.asarray(y)
    pred = (np.asarray(p) >= t).astype(int)
    cm = confusion(y, p, t)
    tp, fp, tn = cm["tp"], cm["fp"], cm["tn"]
    flagged = tp + fp
    return {
        "threshold": float(t),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
        "flagged_share": float(flagged / len(y)),
        # the "cost" of high recall: healthy people flagged per sick person found
        "false_alarms_per_case_found": float(fp / tp) if tp else float("nan"),
        "people_flagged_per_case_found": float(flagged / tp) if tp else float("nan"),
        **cm,
    }


def rank_metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, p, sample_weight=w)),
        "pr_auc": float(average_precision_score(y, p, sample_weight=w)),
        "brier": float(brier_score_loss(y, p, sample_weight=w)),
    }


def bootstrap_ci(
    y: np.ndarray,
    scores: dict[str, np.ndarray],
    stats,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Paired bootstrap over rows. Every model sees the same resampled rows each round.

    `stats` maps a statistic name to f(y, p) -> float. Returns point estimate and 95% interval
    (2.5th and 97.5th percentiles) per model and statistic.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = len(y)
    draws = {m: {s: [] for s in stats} for m in scores}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        for m, p in scores.items():
            pb = np.asarray(p)[idx]
            for s, f in stats.items():
                draws[m][s].append(f(yb, pb))
    out: dict[str, dict[str, tuple[float, float, float]]] = {}
    for m, p in scores.items():
        out[m] = {}
        for s, f in stats.items():
            d = np.asarray(draws[m][s])
            out[m][s] = (
                float(f(y, np.asarray(p))),
                float(np.nanpercentile(d, 2.5)),
                float(np.nanpercentile(d, 97.5)),
            )
    return out


def paired_difference_ci(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    stat,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """stat(a) - stat(b) with a paired bootstrap 95% interval."""
    rng = np.random.default_rng(seed)
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(stat(y[idx], p_a[idx]) - stat(y[idx], p_b[idx]))
    point = stat(y, p_a) - stat(y, p_b)
    return float(point), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    """Equal-count bins: mean predicted risk against observed rate."""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(p)
    rows = []
    for chunk in np.array_split(order, n_bins):
        rows.append(
            {
                "n": int(len(chunk)),
                "mean_predicted": float(p[chunk].mean()),
                "observed_rate": float(y[chunk].mean()),
            }
        )
    return rows


def calibration_summary(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    """Brier score, expected calibration error, and calibration intercept and slope.

    Slope below 1 means predictions are too extreme (overconfident); a perfectly calibrated
    model has intercept 0 and slope 1. Mean predicted vs observed rate shows overall drift.
    """
    y, p = np.asarray(y), np.asarray(p)
    tab = calibration_table(y, p, n_bins)
    ece = sum(r["n"] * abs(r["mean_predicted"] - r["observed_rate"]) for r in tab) / len(y)
    eps = 1e-6
    logit = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps))).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit, y)
    return {
        "brier": float(brier_score_loss(y, p)),
        "ece": float(ece),
        "intercept": float(lr.intercept_[0]),
        "slope": float(lr.coef_[0, 0]),
        "mean_predicted": float(p.mean()),
        "observed_rate": float(y.mean()),
    }
