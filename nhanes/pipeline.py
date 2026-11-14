"""Stage 6: one function that runs everything, from download to docs/RESULTS.md.

Idempotent: it rebuilds the raw, stg and model schemas from scratch each time and every random
step uses a fixed seed, so a second run on the same raw files gives the same numbers.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nhanes import plots
from nhanes.config import (
    DATA_DIR,
    MANIFEST_PATH,
    RAW_DIR,
    ROOT,
    Outputs,
    db_url,
    fixture_db_url,
)
from nhanes.db import ensure_database, get_engine, read_sql, wait_for_db
from nhanes.evaluate import Experiment, run_experiment
from nhanes.extract import fetch_all
from nhanes.features import CATEGORICAL, FEATURES, NUMERIC, TARGET, load_model_table
from nhanes.fixture import write_fixture
from nhanes.load import apply_transforms, load_raw
from nhanes.report import render_results, update_readme_headline
from nhanes.validate import Thresholds, assert_valid

DISCLAIMER = (
    "Educational demo, not medical advice. Trained on self-reported survey answers from the "
    "U.S. NHANES survey. It does not diagnose anything."
)


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _clean(o):
    """Make results strict JSON: NaN and infinity become null."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def save_figures(exp: Experiment, df: pd.DataFrame, out: Outputs) -> list[Path]:
    te = exp.frames["test"]
    y = te[TARGET].to_numpy()
    names = ["logistic_regression", "random_forest", "xgboost"]
    probs = {m: exp.fitted[m].predict(te) for m in ["age_sex_baseline", *names]}
    thr = {m: exp.results["models"][m]["recall_target"]["threshold"] for m in probs}
    va = exp.frames["valid"]
    sel = exp.selected
    p_va = exp.fitted[sel].predict(va)
    figs = {
        "missingness": plots.plot_missingness(df),
        "prevalence_by_cycle": plots.plot_prevalence_by_cycle(df),
        "rate_by_risk_factor": plots.plot_rate_panels(df),
        "distributions": plots.plot_distributions(df),
        "roc_pr": plots.plot_roc_pr(y, probs),
        "confusion_matrices": plots.plot_confusions(y, probs, thr, names),
        "calibration": plots.plot_calibration(y, probs),
        "threshold_tradeoff": plots.plot_threshold_tradeoff(
            va[TARGET].to_numpy(), p_va, y, probs[sel]
        ),
        "importance": plots.plot_importance(
            exp.results["importance"]["permutation_selected"],
            exp.results["importance"]["shap_xgboost_mean_abs"],
        ),
        "shap_beeswarm": plots.plot_shap_beeswarm(exp.shap),
        "subgroups": plots.plot_subgroups(exp.results["subgroups_selected"]),
    }
    return [plots.save(f, out.fig_dir / f"{k}.png") for k, f in figs.items()]


def model_card(exp: Experiment, df: pd.DataFrame) -> dict:
    """Everything the Streamlit app needs besides the model: input ranges and the threshold."""
    train = exp.frames["train"]
    rt = exp.results["models"][exp.selected]["recall_target"]
    ranges = {}
    for c in NUMERIC:
        s = train[c].dropna()
        ranges[c] = {
            "min": float(s.min()),
            "p01": float(s.quantile(0.01)),
            "median": float(s.median()),
            "p99": float(s.quantile(0.99)),
            "max": float(s.max()),
        }
    return {
        "model": exp.selected,
        "disclaimer": DISCLAIMER,
        "threshold": exp.threshold,
        "target_recall": rt["target"],
        "valid_recall_at_threshold": rt["valid"]["recall"],
        "test_recall_at_threshold": rt["test"]["recall"],
        "test_share_flagged": rt["test"]["flagged_share"],
        "test_false_alarms_per_case_found": rt["test"]["false_alarms_per_case_found"],
        "test_roc_auc": exp.results["models"][exp.selected]["test"]["roc_auc"],
        "base_rate_train": float(train[TARGET].mean()),
        "trained_on": exp.results["data"]["splits"]["train"]["survey_years"],
        "tested_on": exp.results["data"]["splits"]["test"]["survey_years"],
        "numeric_ranges": ranges,
        "categories": {c: sorted(train[c].dropna().unique().tolist()) for c in CATEGORICAL},
        "features": FEATURES,
        "versions": {
            "python": platform.python_version(),
            "sklearn": exp.results["meta"]["sklearn"],
            "xgboost": exp.results["meta"]["xgboost"],
            "pandas": exp.results["meta"]["pandas"],
        },
    }


def save_artifacts(exp: Experiment, df: pd.DataFrame, out: Outputs, checks, funnel) -> None:
    R = exp.results
    R["data"]["funnel"] = funnel.to_dict("records")
    R["validation"] = [
        {"name": c.name, "passed": bool(c.passed), "detail": c.detail} for c in checks
    ]
    out.results_json.parent.mkdir(parents=True, exist_ok=True)
    out.results_json.write_text(
        json.dumps(_clean(R), indent=2, default=_json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    out.results_md.write_text(render_results(R), encoding="utf-8")
    out.artifacts.mkdir(parents=True, exist_ok=True)
    exp.predictions.to_csv(out.artifacts / "predictions.csv", index=False)
    joblib.dump(
        {m: f.pipeline for m, f in exp.fitted.items()}, out.artifacts / "models.joblib", compress=3
    )
    out.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(exp.fitted[exp.selected].pipeline, out.model_dir / "final_model.joblib", compress=3)
    (out.model_dir / "model_card.json").write_text(
        json.dumps(model_card(exp, df), indent=2, default=_json_default) + "\n", encoding="utf-8"
    )


def run_all(
    fixture: bool = False,
    out_dir: Path | None = None,
    n_boot: int = 1000,
    quick: bool = False,
    skip_fetch: bool = False,
) -> Experiment:
    t0 = time.time()
    if fixture:
        if out_dir is None:
            raise ValueError("fixture runs need an output directory")
        out = Outputs(Path(out_dir), Path(out_dir) / "artifacts")
        url = fixture_db_url()
        ensure_database(url)
        raw_dir = write_fixture(Path(out_dir) / "fixture_raw")
        thresholds = Thresholds.fixture()
    else:
        out = Outputs(ROOT, DATA_DIR / "artifacts")
        url = db_url()
        raw_dir = RAW_DIR
        thresholds = Thresholds()
        if not skip_fetch:
            print("[1/6] extract: download and verify checksums")
            fetch_all(RAW_DIR, MANIFEST_PATH)
    engine = get_engine(url)
    wait_for_db(engine)
    print("[2/6] load raw tables into PostgreSQL (schema raw)")
    print("     ", load_raw(engine, raw_dir))
    print("[3/6] transform in SQL (schemas stg, model)")
    print("     ", apply_transforms(engine))
    print("[4/6] validate data quality")
    checks = assert_valid(engine, thresholds)
    print("[5/6] train and evaluate")
    df = load_model_table(engine)
    funnel = read_sql(engine, "SELECT * FROM model.cohort_funnel")
    exp = run_experiment(df, n_boot=n_boot, quick=quick)
    print("[6/6] write results, figures, model")
    save_artifacts(exp, df, out, checks, funnel)
    save_figures(exp, df, out)
    if not fixture:
        update_readme_headline(exp.results, ROOT / "README.md")
    print(f"done in {time.time() - t0:.0f}s, results in {out.results_md}")
    return exp


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
