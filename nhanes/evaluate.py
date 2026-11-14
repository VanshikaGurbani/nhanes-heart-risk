"""Stage 5c: train every model, evaluate on the held-out latest cycle, return one results dict.

Rules that keep the evaluation honest:
  * hyperparameters, the model choice and the decision threshold use the validation cycle only;
  * the test cycle is scored once per fitted model and never feeds back into any choice;
  * every number in docs/RESULTS.md is read from the dict this module returns.
"""

from __future__ import annotations

import platform
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from nhanes import metrics as M
from nhanes.config import CYCLES, SEED
from nhanes.explain import logistic_odds_ratios, permutation_table, shap_for_xgboost
from nhanes.features import CATEGORICAL, FEATURES, NUMERIC, TARGET
from nhanes.modeling import (
    Fitted,
    fit_baseline_age_sex,
    fit_model,
    refit_with_strategy,
    tune,
)

MODEL_ORDER = ["logistic_regression", "random_forest", "xgboost"]  # simplest first
RECALL_TARGETS = (0.80, 0.85, 0.90, 0.95)
PRIMARY_RECALL = 0.90


@dataclass
class Experiment:
    results: dict
    fitted: dict[str, Fitted]
    selected: str
    threshold: float
    frames: dict[str, pd.DataFrame]
    predictions: pd.DataFrame
    shap: dict | None


def choose_model(
    fitted: dict[str, Fitted],
    y_valid: np.ndarray,
    p_valid: dict[str, np.ndarray],
    n_boot: int = 1000,
) -> tuple[str, dict]:
    """Pick the final model using the validation cycle only.

    A more complex model replaces logistic regression only if its validation PR-AUC is higher
    AND the paired bootstrap 95% interval of the gap (model minus logistic regression) is above
    zero. Otherwise the simplest model wins. Among several that qualify, the best PR-AUC wins.
    This rule was written down after a first exploratory run showed the three models within
    about 1 point of each other; applying it uses validation data and nothing else.
    """
    evidence = {}
    eligible = []
    for m in MODEL_ORDER[1:]:
        d, lo, hi = M.paired_difference_ci(
            y_valid, p_valid[m], p_valid["logistic_regression"], _ap, n_boot, SEED
        )
        evidence[m] = {"pr_auc_gap_vs_logistic": d, "lo": lo, "hi": hi, "qualifies": bool(lo > 0)}
        if lo > 0:
            eligible.append(m)
    if not eligible:
        return "logistic_regression", evidence
    return max(eligible, key=lambda m: fitted[m].valid_pr_auc), evidence


def _auc(y, p):
    return roc_auc_score(y, p)


def _ap(y, p):
    return average_precision_score(y, p)


def _split_summary(df: pd.DataFrame) -> dict:
    out = {}
    for s in ("train", "valid", "test"):
        d = df[df.split == s]
        out[s] = {
            "cycles": sorted(d.cycle.unique().tolist()),
            "survey_years": ", ".join(sorted(d.survey_years.unique().tolist())),
            "n": int(len(d)),
            "positives": int(d[TARGET].sum()),
            "prevalence": float(d[TARGET].mean()),
        }
    return out


def _missingness(df: pd.DataFrame) -> dict:
    cols = FEATURES
    return {
        "overall": {c: float(df[c].isna().mean()) for c in cols},
        "by_split": {
            s: {c: float(df.loc[df.split == s, c].isna().mean()) for c in cols}
            for s in ("train", "valid", "test")
        },
        "examined_share": float(df.was_examined.mean()),
        "complete_case_share": float(df[FEATURES].notna().all(axis=1).mean()),
    }


def _subgroups(test: pd.DataFrame, p: np.ndarray, t: float, n_boot: int) -> dict:
    out = {}
    rng = np.random.default_rng(SEED)
    for col in ("sex", "age_band", "race_ethnicity"):
        rows = []
        for level, d in test.assign(_p=p).groupby(col, observed=True):
            y, s = d[TARGET].to_numpy(), d["_p"].to_numpy()
            row = {
                "group": str(level),
                "n": int(len(d)),
                "positives": int(y.sum()),
                "prevalence": float(y.mean()),
            }
            if 0 < y.sum() < len(y):
                row["roc_auc"] = float(roc_auc_score(y, s))
                boots = []
                for _ in range(n_boot):
                    idx = rng.integers(0, len(y), len(y))
                    if y[idx].min() != y[idx].max():
                        boots.append(roc_auc_score(y[idx], s[idx]))
                row["roc_auc_lo"], row["roc_auc_hi"] = (
                    float(np.percentile(boots, 2.5)),
                    float(np.percentile(boots, 97.5)),
                )
                tm = M.threshold_metrics(y, s, t)
                row.update(
                    recall=tm["recall"],
                    precision=tm["precision"],
                    flagged_share=tm["flagged_share"],
                    false_alarms_per_case_found=tm["false_alarms_per_case_found"],
                )
            rows.append(row)
        out[col] = rows
    return out


def _weighted_sensitivity(fitted: dict[str, Fitted], sel: str, frames, p_test: np.ndarray) -> dict:
    """Do the conclusions change if the survey weights are used?

    1. Score the (unweighted) selected model with weighted metrics on examined test people.
    2. Retrain logistic regression with exam weights as sample weights and compare.
    """
    tr, va, te = frames["train"], frames["valid"], frames["test"]
    mask = (te.was_examined & (te.exam_weight > 0)).to_numpy()
    y = te[TARGET].to_numpy()[mask]
    w = te.exam_weight.to_numpy()[mask]
    out = {
        "n_examined_test": int(mask.sum()),
        "selected_unweighted_roc_auc": float(roc_auc_score(y, p_test[mask])),
        "selected_weighted_roc_auc": float(roc_auc_score(y, p_test[mask], sample_weight=w)),
        "selected_unweighted_pr_auc": float(average_precision_score(y, p_test[mask])),
        "selected_weighted_pr_auc": float(
            average_precision_score(y, p_test[mask], sample_weight=w)
        ),
        "weighted_prevalence": float(np.average(y, weights=w)),
        "unweighted_prevalence": float(y.mean()),
    }
    # weighted training of logistic regression (examined people only: others have weight 0)
    trm = tr[tr.was_examined & (tr.exam_weight > 0)]
    base = fitted["logistic_regression"]
    pipe = fit_model("logistic_regression", base.params, trm, va)
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    clf_w = LogisticRegression(C=base.params["C"], max_iter=5000, random_state=SEED)
    clf_w.fit(
        pre.transform(trm[FEATURES]),
        trm[TARGET],
        sample_weight=trm.exam_weight / trm.exam_weight.mean(),
    )
    pw = clf_w.predict_proba(pre.transform(te[FEATURES]))[:, 1][mask]
    pu = clf.predict_proba(pre.transform(te[FEATURES]))[:, 1][mask]
    out.update(
        lr_unweighted_training_weighted_auc=float(roc_auc_score(y, pu, sample_weight=w)),
        lr_weighted_training_weighted_auc=float(roc_auc_score(y, pw, sample_weight=w)),
        lr_unweighted_training_unweighted_auc=float(roc_auc_score(y, pu)),
        lr_weighted_training_unweighted_auc=float(roc_auc_score(y, pw)),
    )
    return out


def _missing_strategy_table(fitted, frames, n_boot: int) -> dict:
    tr, va, te = frames["train"], frames["valid"], frames["test"]
    y = te[TARGET].to_numpy()
    rows = []
    stats = {"roc_auc": _auc, "pr_auc": _ap}
    plan = {
        "logistic_regression": ("median_indicator", "median", "iterative"),
        "xgboost": ("median_indicator", "median", "iterative", "native"),
    }
    warnings.filterwarnings("ignore", message=".*IterativeImputer.*")
    for name, strategies in plan.items():
        params = dict(fitted[name].params)
        for strat in strategies:
            if strat == fitted[name].strategy:
                pipe = fitted[name].pipeline
            else:
                pipe = refit_with_strategy(fitted[name], strat, tr, va)
            p = pipe.predict_proba(te[FEATURES])[:, 1]
            ci = M.bootstrap_ci(y, {"m": p}, stats, n_boot=n_boot, seed=SEED)["m"]
            rows.append(
                {
                    "model": name,
                    "strategy": strat,
                    "roc_auc": ci["roc_auc"][0],
                    "roc_auc_lo": ci["roc_auc"][1],
                    "roc_auc_hi": ci["roc_auc"][2],
                    "pr_auc": ci["pr_auc"][0],
                    "pr_auc_lo": ci["pr_auc"][1],
                    "pr_auc_hi": ci["pr_auc"][2],
                    "params": params,
                }
            )
    # complete-case: train and test only on rows with no missing feature at all
    cc_tr = tr[tr[FEATURES].notna().all(axis=1)]
    cc_te = te[te[FEATURES].notna().all(axis=1)]
    lr_params = dict(fitted["logistic_regression"].params)
    cc_pipe = fit_model("logistic_regression", lr_params, cc_tr, va, "median")
    p_cc = cc_pipe.predict_proba(cc_te[FEATURES])[:, 1]
    p_main = fitted["logistic_regression"].pipeline.predict_proba(cc_te[FEATURES])[:, 1]
    yc = cc_te[TARGET].to_numpy()
    complete = {
        "n_train_complete": int(len(cc_tr)),
        "n_train_total": int(len(tr)),
        "n_test_complete": int(len(cc_te)),
        "n_test_total": int(len(te)),
        "positives_test_complete": int(yc.sum()),
        "complete_case_model_roc_auc": float(roc_auc_score(yc, p_cc)),
        "complete_case_model_pr_auc": float(average_precision_score(yc, p_cc)),
        "median_indicator_model_roc_auc_same_rows": float(roc_auc_score(yc, p_main)),
        "median_indicator_model_pr_auc_same_rows": float(average_precision_score(yc, p_main)),
    }
    return {"rows": rows, "complete_case": complete}


def run_experiment(df: pd.DataFrame, n_boot: int = 1000, quick: bool = False) -> Experiment:
    frames = {s: df[df.split == s].reset_index(drop=True) for s in ("train", "valid", "test")}
    tr, va, te = frames["train"], frames["valid"], frames["test"]
    y_va, y_te = va[TARGET].to_numpy(), te[TARGET].to_numpy()

    fitted: dict[str, Fitted] = {name: tune(name, tr, va) for name in MODEL_ORDER}
    baseline = fit_baseline_age_sex(tr, va)
    all_models = {"age_sex_baseline": baseline, **fitted}
    p_va = {m: f.predict(va) for m, f in all_models.items()}
    selected, selection_evidence = choose_model(fitted, y_va, p_va, n_boot)
    p_te = {m: f.predict(te) for m, f in all_models.items()}

    # thresholds come from validation scores only
    thr = {m: M.threshold_for_recall(y_va, p_va[m], PRIMARY_RECALL) for m in all_models}

    stats = {"roc_auc": _auc, "pr_auc": _ap}
    ci = M.bootstrap_ci(y_te, p_te, stats, n_boot=n_boot, seed=SEED)

    def at_thr_ci(m):
        t = thr[m]
        rec = lambda y, p: float(((p >= t) & (y == 1)).sum() / max((y == 1).sum(), 1))  # noqa: E731
        prec = lambda y, p: float(((p >= t) & (y == 1)).sum() / max((p >= t).sum(), 1))  # noqa: E731
        return M.bootstrap_ci(y_te, {m: p_te[m]}, {"recall": rec, "precision": prec}, n_boot, SEED)[
            m
        ]

    models = {}
    for m, f in all_models.items():
        t_ci = at_thr_ci(m)
        models[m] = {
            "params": f.params,
            "strategy": f.strategy,
            "valid": {**M.rank_metrics(y_va, p_va[m])},
            "test": {
                **M.rank_metrics(y_te, p_te[m]),
                "roc_auc_ci": list(ci[m]["roc_auc"][1:]),
                "pr_auc_ci": list(ci[m]["pr_auc"][1:]),
            },
            "default_threshold_test": M.threshold_metrics(y_te, p_te[m], 0.5),
            "recall_target": {
                "target": PRIMARY_RECALL,
                "threshold": thr[m],
                "valid": M.threshold_metrics(y_va, p_va[m], thr[m]),
                "test": M.threshold_metrics(y_te, p_te[m], thr[m]),
                "test_recall_ci": list(t_ci["recall"][1:]),
                "test_precision_ci": list(t_ci["precision"][1:]),
            },
            "calibration_test": M.calibration_summary(y_te, p_te[m]),
            "calibration_table_test": M.calibration_table(y_te, p_te[m]),
            "grid": f.grid,
        }

    # paired differences, one pair per question the README answers
    diffs = {}
    for a, b in (
        ("xgboost", "logistic_regression"),
        ("random_forest", "logistic_regression"),
        ("logistic_regression", "age_sex_baseline"),
        ("xgboost", "age_sex_baseline"),
    ):
        diffs[f"{a}_minus_{b}"] = {
            "roc_auc": list(M.paired_difference_ci(y_te, p_te[a], p_te[b], _auc, n_boot, SEED)),
            "pr_auc": list(M.paired_difference_ci(y_te, p_te[a], p_te[b], _ap, n_boot, SEED)),
        }

    # threshold trade-off table for the selected model (validation picks, test reports)
    trade = []
    for tgt in RECALL_TARGETS:
        t = M.threshold_for_recall(y_va, p_va[selected], tgt)
        tm = M.threshold_metrics(y_te, p_te[selected], t)
        vm = M.threshold_metrics(y_va, p_va[selected], t)
        trade.append({"target_recall": tgt, "threshold": t, "valid": vm, "test": tm})

    # recalibration check (Platt scaling fitted on validation, applied to test)
    eps = 1e-6
    lg = lambda p: np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps))).reshape(-1, 1)  # noqa: E731
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(lg(p_va[selected]), y_va)
    p_te_platt = platt.predict_proba(lg(p_te[selected]))[:, 1]
    recal = {
        "raw": M.calibration_summary(y_te, p_te[selected]),
        "platt_on_valid": M.calibration_summary(y_te, p_te_platt),
    }

    results = {
        "meta": {
            "seed": SEED,
            "n_boot": n_boot,
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "cycles": [{"letter": c.letter, "years": c.years, "split": c.split} for c in CYCLES],
            "features_numeric": NUMERIC,
            "features_categorical": CATEGORICAL,
        },
        "data": {
            "splits": _split_summary(df),
            "n_total": int(len(df)),
            "positives_total": int(df[TARGET].sum()),
            "missingness": _missingness(df),
            "label_components": {
                k: int(df[k].fillna(False).astype(bool).sum())
                for k in ("chd", "angina", "heart_attack", "chf", "stroke")
            },
            "majority_class_accuracy_test": float(1 - y_te.mean()),
            "prevalence_by_cycle": {c: float(g[TARGET].mean()) for c, g in df.groupby("cycle")},
        },
        "selection": {
            "rule": (
                "a model replaces logistic regression only if the paired bootstrap 95% interval "
                "of its validation PR-AUC gap is above zero"
            ),
            "evidence": selection_evidence,
            "selected": selected,
            "valid_pr_auc": {m: all_models[m].valid_pr_auc for m in all_models},
        },
        "models": models,
        "paired_differences_test": diffs,
        "threshold_tradeoff_selected": trade,
        "recalibration_selected": recal,
        "subgroups_selected": _subgroups(te, p_te[selected], thr[selected], 300 if quick else 1000),
        "weighted_sensitivity": _weighted_sensitivity(fitted, selected, frames, p_te[selected]),
        "missing_value_strategies": _missing_strategy_table(fitted, frames, 300 if quick else 1000),
        "odds_ratios_logistic": logistic_odds_ratios(fitted["logistic_regression"].pipeline),
    }

    shap_out = shap_for_xgboost(fitted["xgboost"].pipeline, te)
    results["importance"] = {
        "permutation_selected": permutation_table(
            fitted[selected].pipeline, te, 5 if quick else 20
        ),
        "permutation_xgboost": permutation_table(
            fitted["xgboost"].pipeline, te, 5 if quick else 20
        ),
        "shap_xgboost_mean_abs": shap_out["table"],
    }

    pred = pd.concat(
        [
            d[["seqn", "cycle", "split", "sex", "age_band", "age_years", "race_ethnicity",
               "was_examined", "exam_weight", TARGET]].assign(
                **{f"p_{m}": p for m, p in zip(
                    all_models, ([p_va[m] for m in all_models] if k == "valid"
                                 else [p_te[m] for m in all_models]), strict=True)}
            )
            for k, d in (("valid", va), ("test", te))
        ],
        ignore_index=True,
    )  # fmt: skip
    return Experiment(
        results=results,
        fitted=all_models,
        selected=selected,
        threshold=thr[selected],
        frames=frames,
        predictions=pred,
        shap=shap_out,
    )
