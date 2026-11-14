"""Charts. Okabe-Ito colours (colour-blind safe); lines also differ in style, not colour alone."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from nhanes import metrics as M  # noqa: E402
from nhanes.features import FEATURES, LABELS, TARGET  # noqa: E402

C = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermillion": "#D55E00",
    "sky": "#56B4E9", "purple": "#CC79A7", "yellow": "#F0E442", "black": "#000000", "grey": "#7F7F7F",
}  # fmt: skip
MODEL_STYLE = {
    "age_sex_baseline": {"color": C["grey"], "ls": ":", "label": "Age + sex only"},
    "logistic_regression": {"color": C["blue"], "ls": "-", "label": "Logistic regression"},
    "random_forest": {"color": C["orange"], "ls": "--", "label": "Random forest"},
    "xgboost": {"color": C["vermillion"], "ls": "-.", "label": "XGBoost"},
}

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
    }
)


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval for a proportion."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return centre - half, centre + half


def plot_missingness(df: pd.DataFrame):
    miss = df.groupby("split")[FEATURES].apply(lambda g: g.isna().mean() * 100).T
    miss = miss[["train", "valid", "test"]].sort_values("train")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    y = np.arange(len(miss))
    h = 0.27
    for i, (s, col) in enumerate(
        zip(("train", "valid", "test"), (C["blue"], C["orange"], C["green"]), strict=True)
    ):
        ax.barh(y + (i - 1) * h, miss[s], height=h, color=col, label=s)
    ax.set_yticks(y, [LABELS.get(f, f.replace("_", " ")) for f in miss.index])
    ax.set_xlabel("% of adults with the value missing")
    ax.set_title("Missing values by feature and split")
    ax.legend(title="split", frameon=False)
    return fig


def _rate_bars(ax, df, col, order=None, title=None, rotate=0):
    g = df.groupby(col, observed=True)[TARGET].agg(["sum", "count"])
    if order is not None:
        g = g.reindex([o for o in order if o in g.index])
    rate = g["sum"] / g["count"] * 100
    ci = np.array([wilson(int(k), int(n)) for k, n in zip(g["sum"], g["count"], strict=True)]) * 100
    x = np.arange(len(g))
    ax.bar(x, rate, color=C["blue"], width=0.65)
    ax.errorbar(
        x,
        rate,
        yerr=[rate - ci[:, 0], ci[:, 1] - rate],
        fmt="none",
        ecolor="black",
        capsize=3,
        lw=1,
    )
    top = ci[:, 1]
    for xi, hi, n in zip(x, top, g["count"], strict=True):
        ax.text(
            xi,
            hi + top.max() * 0.02,
            f"n={n:,}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#444444",
        )
    ax.set_ylim(0, top.max() * 1.15)
    ax.set_xticks(x, [str(i) for i in g.index], rotation=rotate)
    ax.set_ylabel("% with heart disease")
    ax.set_title(title or col)


def add_risk_factor_bins(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["age_group"] = pd.cut(
        d.age_years,
        [19, 29, 39, 49, 59, 69, 80],
        labels=["20s", "30s", "40s", "50s", "60s", "70-80"],
    )
    d["bmi_group"] = pd.cut(
        d.bmi,
        [0, 18.5, 25, 30, 35, 100],
        labels=["<18.5", "18.5-25", "25-30", "30-35", "35+"],
        right=False,
    )
    d["sbp_group"] = pd.cut(
        d.sbp_mean,
        [0, 120, 130, 140, 160, 400],
        labels=["<120", "120-129", "130-139", "140-159", "160+"],
        right=False,
    )
    return d  # fmt: skip


def plot_rate_panels(df: pd.DataFrame):
    d = add_risk_factor_bins(df)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    _rate_bars(axes[0, 0], d, "age_group", title="By age")
    _rate_bars(axes[0, 1], d, "sex", title="By sex")
    _rate_bars(axes[0, 2], d, "bmi_group", title="By BMI")
    _rate_bars(axes[1, 0], d, "sbp_group", title="By systolic blood pressure (mmHg)")
    _rate_bars(
        axes[1, 1], d, "smoking_status", order=["never", "former", "current"], title="By smoking"
    )
    _rate_bars(
        axes[1, 2],
        d,
        "diabetes",
        order=["no", "borderline", "yes"],
        title="By diabetes (self-reported)",
    )
    fig.suptitle(
        "Share of adults reporting heart disease, with 95% intervals", fontweight="bold", y=1.01
    )
    fig.tight_layout()
    return fig


def plot_distributions(
    df: pd.DataFrame, cols=("age_years", "bmi", "sbp_mean", "total_chol", "hdl", "hba1c")
):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, c in zip(axes.ravel(), cols, strict=False):
        lo, hi = df[c].quantile([0.005, 0.995])
        bins = np.linspace(lo, hi, 36)
        for lab, col, name in (
            (0, C["blue"], "no heart disease"),
            (1, C["vermillion"], "heart disease"),
        ):
            ax.hist(df.loc[df[TARGET] == lab, c].dropna().clip(lo, hi), bins=bins, density=True,
                    histtype="step", lw=2, color=col, label=name)  # fmt: skip
        ax.set_title(LABELS.get(c, c))
        ax.set_ylabel("density")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Distributions by outcome (each curve sums to 1)", fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_prevalence_by_cycle(df: pd.DataFrame):
    g = df.groupby(["cycle", "survey_years", "split"])[TARGET].agg(["sum", "count"]).reset_index()
    rate = g["sum"] / g["count"] * 100
    ci = np.array([wilson(int(k), int(n)) for k, n in zip(g["sum"], g["count"], strict=True)]) * 100
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    colors = {"train": C["blue"], "valid": C["orange"], "test": C["green"]}
    ax.bar(g.survey_years, rate, color=[colors[s] for s in g.split], width=0.6)
    ax.errorbar(
        g.survey_years,
        rate,
        yerr=[rate - ci[:, 0], ci[:, 1] - rate],
        fmt="none",
        ecolor="black",
        capsize=3,
    )
    for x, r, s in zip(range(len(g)), rate, g.split, strict=True):
        ax.text(x, r + 0.9, s, ha="center", fontsize=8)
    ax.set_ylabel("% with heart disease")
    ax.set_title("Prevalence by survey cycle (colour = split role)")
    ax.set_ylim(0, max(rate) + 3)
    return fig


def plot_roc_pr(y, probs: dict[str, np.ndarray]):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for m, p in probs.items():
        st = MODEL_STYLE[m]
        fpr, tpr, _ = roc_curve(y, p)
        a.plot(fpr, tpr, color=st["color"], ls=st["ls"], lw=2, label=st["label"])
        pr, rc, _ = precision_recall_curve(y, p)
        b.plot(rc, pr, color=st["color"], ls=st["ls"], lw=2, label=st["label"])
    a.plot([0, 1], [0, 1], color="black", lw=0.8, ls=":")
    a.set(
        xlabel="False positive rate",
        ylabel="Recall (true positive rate)",
        title="ROC curve (test cycle)",
    )
    b.axhline(float(np.mean(y)), color="black", lw=0.8, ls=":", label="No skill (prevalence)")
    b.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve (test cycle)")
    a.legend(frameon=False, loc="lower right")
    b.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_confusions(
    y, probs: dict[str, np.ndarray], thresholds: dict[str, float], names: list[str]
):
    fig, axes = plt.subplots(1, len(names), figsize=(4.3 * len(names), 4.2))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, names, strict=True):
        cm = M.confusion(y, probs[m], thresholds[m])
        mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        ax.imshow(mat, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i, j]:,}", ha="center", va="center", fontsize=13,
                        color="white" if mat[i, j] > mat.max() / 2 else "black")  # fmt: skip
        ax.set_xticks([0, 1], ["flagged: no", "flagged: yes"])
        ax.set_yticks([0, 1], ["healthy", "heart disease"])
        ax.grid(False)
        ax.set_title(f"{MODEL_STYLE[m]['label']}\nthreshold {thresholds[m]:.1%}")
    fig.suptitle("Test cycle confusion matrices at the 90% recall threshold", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_calibration(y, probs: dict[str, np.ndarray]):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
    mx = 0.0
    for m, p in probs.items():
        t = pd.DataFrame(M.calibration_table(y, p))
        st = MODEL_STYLE[m]
        a.plot(
            t.mean_predicted * 100,
            t.observed_rate * 100,
            marker="o",
            color=st["color"],
            ls=st["ls"],
            label=st["label"],
        )
        mx = max(mx, t.mean_predicted.max(), t.observed_rate.max())
        b.hist(np.asarray(p) * 100, bins=40, histtype="step", lw=2, color=st["color"], ls=st["ls"], label=st["label"])  # fmt: skip
    lim = mx * 100 * 1.05
    a.plot([0, lim], [0, lim], color="black", lw=0.8, ls=":", label="Perfect")
    a.set(xlabel="Mean predicted risk in decile (%)", ylabel="Observed rate in decile (%)",
          title="Calibration by decile (test cycle)")  # fmt: skip
    a.legend(frameon=False)
    b.set(xlabel="Predicted risk (%)", ylabel="People", title="Distribution of predicted risk")
    fig.tight_layout()
    return fig


def plot_threshold_tradeoff(y_valid, p_valid, y_test, p_test, targets=(0.8, 0.85, 0.9, 0.95)):
    ts = np.linspace(0.005, 0.5, 200)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.7))
    for y, p, ls, tag in ((y_valid, p_valid, "--", "validation"), (y_test, p_test, "-", "test")):
        rows = [M.threshold_metrics(y, p, t) for t in ts]
        a.plot(
            ts * 100,
            [r["recall"] for r in rows],
            color=C["blue"],
            ls=ls,
            lw=2,
            label=f"recall, {tag}",
        )
        a.plot(
            ts * 100,
            [r["precision"] for r in rows],
            color=C["vermillion"],
            ls=ls,
            lw=2,
            label=f"precision, {tag}",
        )
        a.plot(
            ts * 100,
            [r["flagged_share"] for r in rows],
            color=C["green"],
            ls=ls,
            lw=2,
            label=f"share flagged, {tag}",
        )
    for i, tg in enumerate(targets[::-1]):
        t = M.threshold_for_recall(y_valid, p_valid, tg)
        a.axvline(t * 100, color="black", lw=0.7, ls=":")
        a.text(t * 100, 1.02 + 0.06 * (i % 2), f"{tg:.0%}", ha="center", fontsize=8)
    a.set(xlabel="Decision threshold (flag if predicted risk is at least this, %)", ylabel="Share", ylim=(0, 1.16),
          title="Thresholds picked on validation (dotted, labelled by target recall)")  # fmt: skip
    a.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.95))
    rec, fa, flagged = [], [], []
    for tg in targets:
        t = M.threshold_for_recall(y_valid, p_valid, tg)
        tm = M.threshold_metrics(y_test, p_test, t)
        rec.append(tm["recall"] * 100)
        fa.append(tm["false_alarms_per_case_found"])
        flagged.append(tm["flagged_share"] * 100)
    x = np.arange(len(targets))
    b.bar(x, fa, color=C["orange"], width=0.55)
    for xi, f, r, fl in zip(x, fa, rec, flagged, strict=True):
        b.text(
            xi, f + 0.15, f"{f:.1f}\n(recall {r:.0f}%,\nflag {fl:.0f}%)", ha="center", fontsize=8
        )
    b.set_xticks(x, [f"{t:.0%}" for t in targets])
    b.set(xlabel="Target recall used to pick the threshold on validation",
          ylabel="False alarms per sick person found (test)", title="Cost of catching more sick people",
          ylim=(0, max(fa) * 1.35))  # fmt: skip
    fig.tight_layout()
    return fig


def plot_importance(perm: list[dict], shap_table: list[dict], top: int = 10):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8))
    p = perm[:top][::-1]
    a.barh([LABELS.get(r["feature"], r["feature"]) for r in p], [r["auc_drop_mean"] for r in p],
           xerr=[r["auc_drop_std"] for r in p], color=C["blue"])  # fmt: skip
    a.set(
        xlabel="Drop in test ROC-AUC when the feature is shuffled", title="Permutation importance"
    )
    s = shap_table[:top][::-1]
    b.barh([LABELS.get(r["feature"], r["feature"].replace("_", " ")) for r in s],
           [r["mean_abs_shap"] for r in s], color=C["orange"])  # fmt: skip
    b.set(xlabel="Mean |SHAP| (log-odds)", title="SHAP, XGBoost")
    fig.tight_layout()
    return fig


def plot_shap_beeswarm(shap_out: dict, top: int = 10, seed: int = 42):
    names, vals, X = shap_out["names"], shap_out["values"], shap_out["X"]
    order = np.argsort(-np.abs(vals).mean(axis=0))[:top][::-1]
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for row, j in enumerate(order):
        x = X[:, j]
        rank = pd.Series(x).rank(pct=True).to_numpy()
        sc = ax.scatter(
            vals[:, j], row + rng.uniform(-0.3, 0.3, len(x)), c=rank, cmap="viridis", s=5, alpha=0.6
        )
    ax.set_yticks(
        range(len(order)), [LABELS.get(names[j], names[j].replace("_", " ")) for j in order]
    )
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel("SHAP value (effect on log-odds of heart disease)")
    ax.set_title("SHAP values, XGBoost (colour = feature value, low to high)")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_ticks([0, 1], labels=["low", "high"])
    return fig


def plot_subgroups(sub: dict):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8))
    rows = []
    for dim in ("sex", "age_band"):
        for r in sub[dim]:
            if "roc_auc" in r:
                rows.append({**r, "label": f"{dim.replace('_', ' ')}: {r['group']}"})
    d = pd.DataFrame(rows)[::-1].reset_index(drop=True)
    y = np.arange(len(d))
    a.errorbar(
        d.roc_auc,
        y,
        xerr=[d.roc_auc - d.roc_auc_lo, d.roc_auc_hi - d.roc_auc],
        fmt="o",
        color=C["blue"],
        capsize=3,
    )
    a.set_yticks(
        y,
        [
            f"{lab} (n={n:,}, cases={p})"
            for lab, n, p in zip(d.label, d.n, d.positives, strict=True)
        ],
    )
    a.set(xlabel="ROC-AUC with bootstrap 95% interval", title="Ranking quality within each group")
    b.barh(y, d.recall * 100, color=C["green"], height=0.55, label="recall (sick people flagged)")
    b.scatter(
        d.flagged_share * 100,
        y,
        color=C["vermillion"],
        zorder=3,
        marker="D",
        label="share of group flagged",
    )
    b.set_yticks(y, [""] * len(d))
    b.set(xlabel="%", title="At the 90% recall threshold (test cycle)")
    b.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig
