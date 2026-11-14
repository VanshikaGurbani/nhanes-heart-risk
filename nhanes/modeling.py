"""Stage 5b: fit and tune the models. Every choice is made on the validation cycle only."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from nhanes.config import SEED
from nhanes.features import FEATURES, TARGET, make_preprocessor

LR_GRID = [{"C": c} for c in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)]
RF_GRID = [
    {"min_samples_leaf": leaf, "max_features": mf}
    for leaf in (10, 20, 40, 80, 160)
    for mf in ("sqrt", 0.5)
]
XGB_GRID = [{"max_depth": d, "min_child_weight": w} for d in (1, 2, 3, 4) for w in (1, 5)]


@dataclass
class Fitted:
    name: str
    strategy: str
    pipeline: Pipeline
    params: dict
    valid_pr_auc: float
    grid: list[dict] = field(default_factory=list)  # every setting tried, with its valid PR-AUC

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X[FEATURES])[:, 1]


def _lr(params: dict) -> LogisticRegression:
    return LogisticRegression(C=params["C"], max_iter=5000, random_state=SEED)


def _rf(params: dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        n_jobs=-1,
        random_state=SEED,
    )


def _xgb(params: dict, early_stopping: bool) -> XGBClassifier:
    kw = dict(
        n_estimators=params.get("n_estimators", 1000),
        learning_rate=0.05,
        max_depth=params["max_depth"],
        min_child_weight=params["min_child_weight"],
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=4,
        random_state=SEED,
    )
    if early_stopping:
        kw["early_stopping_rounds"] = 50
    return XGBClassifier(**kw)


def _fit_xgb(params, pre, train, valid, early_stopping=True):
    Xt = pre.fit_transform(train[FEATURES])
    Xv = pre.transform(valid[FEATURES])
    clf = _xgb(params, early_stopping)
    if early_stopping:
        clf.fit(Xt, train[TARGET], eval_set=[(Xv, valid[TARGET])], verbose=False)
    else:
        clf.fit(Xt, train[TARGET])
    return clf


def fit_model(
    name: str,
    params: dict,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    strategy: str = "median_indicator",
) -> Pipeline:
    """Fit one model with fixed hyperparameters and return a ready-to-predict pipeline."""
    if name == "logistic_regression":
        pre = make_preprocessor(strategy, scale=True)
        return Pipeline([("pre", pre), ("clf", _lr(params))]).fit(train[FEATURES], train[TARGET])
    if name == "random_forest":
        pre = make_preprocessor(strategy, scale=False)
        return Pipeline([("pre", pre), ("clf", _rf(params))]).fit(train[FEATURES], train[TARGET])
    if name == "xgboost":
        pre = make_preprocessor(strategy, scale=False)
        early = "n_estimators" not in params
        clf = _fit_xgb(params, pre, train, valid, early_stopping=early)
        return Pipeline([("pre", pre), ("clf", clf)])
    raise ValueError(name)


def _valid_pr(pipe: Pipeline, valid: pd.DataFrame) -> float:
    return float(average_precision_score(valid[TARGET], pipe.predict_proba(valid[FEATURES])[:, 1]))


def tune(
    name: str,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    strategy: str = "median_indicator",
) -> Fitted:
    """Grid search on validation PR-AUC (average precision). The test cycle is never touched."""
    grid = {"logistic_regression": LR_GRID, "random_forest": RF_GRID, "xgboost": XGB_GRID}[name]
    best: tuple[float, dict, Pipeline] | None = None
    tried = []
    for params in grid:
        pipe = fit_model(name, params, train, valid, strategy)
        score = _valid_pr(pipe, valid)
        info = dict(params)
        if name == "xgboost":
            info["n_estimators"] = int(pipe.named_steps["clf"].best_iteration) + 1
        tried.append({**info, "valid_pr_auc": score})
        if best is None or score > best[0]:
            best = (score, info, pipe)
    assert best is not None
    return Fitted(name, strategy, best[2], best[1], best[0], tried)


def fit_baseline_age_sex(train: pd.DataFrame, valid: pd.DataFrame) -> Fitted:
    """Reference model: logistic regression on age and sex only. Shows what the other
    features add beyond the two things everyone already knows. The column selector in the
    preprocessor ignores every other column, so it takes the same full feature frame."""
    pre = make_preprocessor("median", scale=True, numeric=["age_years"], categorical=["sex"])
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(C=1.0, max_iter=5000))])
    pipe.fit(train[FEATURES], train[TARGET])
    return Fitted("age_sex_baseline", "median", pipe, {"C": 1.0}, _valid_pr(pipe, valid))


def refit_with_strategy(fitted: Fitted, strategy: str, train, valid) -> Pipeline:
    """Refit a tuned model under another missing-value strategy, same hyperparameters."""
    params = dict(fitted.params)
    return fit_model(fitted.name, params, train, valid, strategy)
