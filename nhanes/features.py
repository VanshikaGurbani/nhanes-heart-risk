"""Stage 5a: feature lists and preprocessing pipelines.

Features are only things a person could know or have measured BEFORE or independently of a heart
disease diagnosis workflow: age, sex, body size, blood pressure, standard blood tests, smoking,
self-reported diabetes and alcohol. Excluded on purpose (leakage or consequence of diagnosis):
heart, blood pressure, cholesterol and diabetes medication questions, congestive heart failure,
stroke, and every column that is part of the label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from nhanes.config import SEED
from nhanes.db import read_sql

NUMERIC = [
    "age_years",
    "bmi",
    "waist_cm",
    "sbp_mean",
    "dbp_mean",
    "total_chol",
    "hdl",
    "hba1c",
    "fasting_glucose",
    "drinks_per_day",
]
CATEGORICAL = ["sex", "smoking_status", "diabetes", "ever_heavy_drinker"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "heart_disease"

# Columns that must never be used as inputs. A test asserts none of them is in FEATURES.
FORBIDDEN = [
    "heart_disease", "chd", "angina", "heart_attack", "chf", "stroke",
    "exam_weight", "psu", "strata", "seqn", "cycle", "split", "survey_years", "race_ethnicity",
    "was_examined", "age_band",
]  # fmt: skip

MISSING_STRATEGIES = ("median_indicator", "median", "iterative", "native")

LABELS = {
    "age_years": "Age (years)",
    "bmi": "BMI (kg/m2)",
    "waist_cm": "Waist (cm)",
    "sbp_mean": "Systolic BP (mmHg)",
    "dbp_mean": "Diastolic BP (mmHg)",
    "total_chol": "Total cholesterol (mg/dL)",
    "hdl": "HDL cholesterol (mg/dL)",
    "hba1c": "HbA1c (%)",
    "fasting_glucose": "Fasting glucose (mg/dL)",
    "drinks_per_day": "Drinks per drinking day",
}


def load_model_table(engine) -> pd.DataFrame:
    return read_sql(engine, "SELECT * FROM model.heart_risk ORDER BY seqn")


def split_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {s: df[df.split == s].reset_index(drop=True) for s in ("train", "valid", "test")}


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each numeric column to the training 0.5th and 99.5th percentiles (NaN stays NaN)."""

    def __init__(self, lower: float = 0.005, upper: float = 0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, names=None):
        return np.asarray(names)


def make_preprocessor(
    strategy: str = "median_indicator",
    scale: bool = True,
    numeric: list[str] | None = None,
    categorical: list[str] | None = None,
) -> ColumnTransformer:
    """Build the preprocessing step for one missing-value strategy.

    median_indicator: median fill plus a 0/1 "was missing" column for each numeric feature
    median:           median fill only
    iterative:        each feature predicted from the others (BayesianRidge), no indicators
    native:           leave NaN in place (only models that accept NaN, i.e. XGBoost)
    """
    numeric = NUMERIC if numeric is None else numeric
    categorical = CATEGORICAL if categorical is None else categorical
    steps: list[tuple[str, object]] = [("clip", Winsorizer())]
    if strategy == "median_indicator":
        steps.append(("impute", SimpleImputer(strategy="median", add_indicator=True)))
    elif strategy == "median":
        steps.append(("impute", SimpleImputer(strategy="median")))
    elif strategy == "iterative":
        steps.append(
            ("impute", IterativeImputer(max_iter=10, random_state=SEED, sample_posterior=False))
        )
    elif strategy == "native":
        pass
    else:
        raise ValueError(f"unknown missing-value strategy: {strategy}")
    if scale:
        steps.append(("scale", StandardScaler()))
    cat = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", Pipeline(steps), numeric), ("cat", cat, categorical)],
        verbose_feature_names_out=False,
    )


def feature_names(pre: ColumnTransformer, numeric: list[str] | None = None) -> list[str]:
    """Readable output column names after a fitted preprocessor (indicators get a _missing tag)."""
    numeric = NUMERIC if numeric is None else numeric
    num_pipe = pre.named_transformers_["num"]
    names = list(numeric)
    imp = num_pipe.named_steps.get("impute")
    if hasattr(imp, "indicator_") and imp.indicator_ is not None:
        idx = imp.indicator_.features_
        names += [f"{numeric[i]}_missing" for i in idx]
    cat = pre.named_transformers_["cat"].named_steps["onehot"]
    names += list(cat.get_feature_names_out(CATEGORICAL))
    return names
