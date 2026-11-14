"""Explanations: permutation importance (any model), SHAP (XGBoost) and logistic odds ratios."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from nhanes.config import SEED
from nhanes.features import FEATURES, NUMERIC, TARGET, feature_names


def permutation_table(
    pipe: Pipeline, test: pd.DataFrame, n_repeats: int = 20
) -> list[dict[str, float]]:
    """Drop in test ROC-AUC when one feature's values are shuffled (bigger drop = more used)."""
    r = permutation_importance(
        pipe, test[FEATURES], test[TARGET], scoring="roc_auc",
        n_repeats=n_repeats, random_state=SEED, n_jobs=1,
    )  # fmt: skip
    rows = [
        {"feature": f, "auc_drop_mean": float(m), "auc_drop_std": float(s)}
        for f, m, s in zip(FEATURES, r.importances_mean, r.importances_std, strict=True)
    ]
    return sorted(rows, key=lambda d: -d["auc_drop_mean"])


def shap_for_xgboost(pipe: Pipeline, test: pd.DataFrame) -> dict:
    """SHAP values (log-odds scale) for the XGBoost model on the test rows."""
    import shap

    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    X = pre.transform(test[FEATURES])
    names = feature_names(pre)
    values = shap.TreeExplainer(clf).shap_values(X)
    values = np.asarray(values)
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(-mean_abs)
    return {
        "names": names,
        "values": values,
        "X": X,
        "table": [{"feature": names[i], "mean_abs_shap": float(mean_abs[i])} for i in order],
    }


def logistic_odds_ratios(pipe: Pipeline) -> list[dict[str, float]]:
    """Odds ratio for a 1 standard deviation increase (numeric) or for the category present."""
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    names = feature_names(pre)
    coef = clf.coef_[0]
    rows = [
        {"feature": n, "coefficient": float(c), "odds_ratio": float(np.exp(c))}
        for n, c in zip(names, coef, strict=True)
        if n in NUMERIC or not n.endswith("_missing")
    ]
    return sorted(rows, key=lambda d: -abs(d["coefficient"]))
