"""Write the five notebooks (without outputs). Run `nhanes run-notebooks` afterwards to execute
them and save the outputs. Keeping the cell source here makes the notebooks reviewable as plain text.

Every number a reader sees in the notebooks is printed or formatted by code. Markdown cells hold
only questions and method; the "what this means" and "what is weak" cells are code cells that
format their sentences from computed values.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks"

SETUP = """\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from nhanes import nb, plots
from nhanes.features import CATEGORICAL, FEATURES, LABELS, NUMERIC, TARGET

ctx = nb.context()
df, R = ctx.df, ctx.R
pct = nb.pct
pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.width", 160)
print("mode:", "synthetic fixture (CI)" if ctx.fixture else "real NHANES data")
print(f"{len(df):,} adults, {int(df[TARGET].sum()):,} reporting heart disease")
"""


def build(name: str, cells: list[tuple[str, str]]) -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    for kind, src in cells:
        nb.cells.append(
            nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
        )
    nbf.write(nb, OUT / name)


# --------------------------------------------------------------------------------------------
NB1 = [
    (
        "md",
        """\
# 1. The data and its missing values

**Question.** What is in the NHANES tables, which adults end up in the analysis, how is the heart
disease label built, and what is missing (and does the missingness itself carry information)?

How to read this notebook: each section runs a query or a calculation and prints the result. The
last two cells, *what this means* and *what is weak*, are code cells so that every number in their
sentences is computed, not typed. Lines starting with `Check:` compare a number recomputed here
with the one the pipeline wrote to `docs/results/results.json`.
""",
    ),
    ("code", SETUP),
    ("md", "## 1.1 Who is in the analysis (a SQL query on the model schema)"),
    (
        "code",
        """\
funnel = ctx.sql("SELECT * FROM model.cohort_funnel ORDER BY cycle")
display(funnel)
nb.check("labelled adults", funnel.adults_with_label.sum(), R["data"]["n_total"])
dropped = funnel.adults_20_plus.sum() - funnel.adults_with_label.sum()
print(f"Adults dropped because the label is unknown (no yes, and at least one refused / don't know / missing): {dropped:,}")
""",
    ),
    (
        "md",
        """\
## 1.2 How refused and don't-know answers look in the raw table

The CDC codes refused as 7 and don't know as 9. If these were left as numbers they would look like
"more than yes". The raw table below is read straight from PostgreSQL (`raw.mcq`), before any cleaning.
""",
    ),
    (
        "code",
        """\
codes = ctx.sql('''
    SELECT cycle,
           COUNT(*) FILTER (WHERE mcq160c = 1)          AS yes,
           COUNT(*) FILTER (WHERE mcq160c = 2)          AS no,
           COUNT(*) FILTER (WHERE mcq160c IN (7, 9))    AS refused_or_dont_know,
           COUNT(*) FILTER (WHERE mcq160c IS NULL)      AS not_asked_or_missing
    FROM raw.mcq GROUP BY cycle ORDER BY cycle
''')
display(codes)
model_codes = ctx.sql("SELECT COUNT(*) AS n FROM model.heart_risk WHERE heart_disease NOT IN (0, 1)")
print("Rows in the model table whose label is not 0 or 1:", int(model_codes.n[0]))
""",
    ),
    (
        "md",
        """\
## 1.3 What the label is made of

Label = yes to coronary heart disease, angina or heart attack. Congestive heart failure and stroke
are different conditions and are **not** part of the label. They are also never used as features,
because they are consequences of (or entangled with) the diagnosis we are predicting.
""",
    ),
    (
        "code",
        """\
comp = df[["chd", "angina", "heart_attack"]].fillna(False).astype(bool)
cases = comp[df[TARGET] == 1]
combo = cases.value_counts().rename("adults").reset_index()
combo["share_of_cases"] = combo.adults / combo.adults.sum()
display(combo)
other = (df[["chf", "stroke"]].fillna(False).astype(bool)
         .groupby(df[TARGET]).mean().rename(index={0: "no heart disease", 1: "heart disease"}))
display(other.rename(columns={"chf": "share with heart failure", "stroke": "share with stroke"}))
""",
    ),
    (
        "md",
        "## 1.4 Splits by survey cycle (earlier cycles train, a later one validates, the latest tests)",
    ),
    (
        "code",
        """\
splits = pd.DataFrame(R["data"]["splits"]).T[["cycles", "survey_years", "n", "positives", "prevalence"]]
display(splits)
nb.show(plots.plot_prevalence_by_cycle(df))
""",
    ),
    ("md", "## 1.5 Missing values"),
    (
        "code",
        """\
nb.show(plots.plot_missingness(df))
miss = df[FEATURES].isna()
share_complete = float((~miss.any(axis=1)).mean())
nb.check("share of adults with every feature", share_complete, R["data"]["missingness"]["complete_case_share"])
patterns = (miss.apply(lambda r: ", ".join(c for c in FEATURES if r[c]) or "nothing missing", axis=1)
            .value_counts(normalize=True).head(6).rename("share of adults").to_frame())
display(patterns)
""",
    ),
    (
        "md",
        """\
## 1.6 Is missingness informative?

If people with a feature missing have a different disease rate from people with it present, then how
we fill in missing values can matter. Fisher's exact test per feature (a small p-value only says the
rates differ; it does not say why).
""",
    ),
    (
        "code",
        """\
from scipy.stats import fisher_exact

rows = []
for c in FEATURES:
    m = df[c].isna()
    if m.sum() < 20 or (~m).sum() < 20:
        continue
    a, b = int(df.loc[m, TARGET].sum()), int((~df.loc[m, TARGET].astype(bool)).sum())
    c1, d1 = int(df.loc[~m, TARGET].sum()), int((~df.loc[~m, TARGET].astype(bool)).sum())
    p = fisher_exact([[a, b], [c1, d1]])[1]
    rows.append({"feature": c, "n_missing": int(m.sum()), "rate_when_missing": df.loc[m, TARGET].mean(),
                 "rate_when_present": df.loc[~m, TARGET].mean(), "p_value": p})
miss_tab = pd.DataFrame(rows).sort_values("p_value")
display(miss_tab)
informative = miss_tab[miss_tab.p_value < 0.05].feature.tolist()
""",
    ),
    ("md", "## What this means"),
    (
        "code",
        """\
ms = R["data"]["missingness"]
top_miss = max(ms["overall"], key=ms["overall"].get)
lab_share = ms["overall"]["total_chol"]
nb.say(
    f"- {pct(ms['examined_share'])} of adults were examined, and only {pct(share_complete)} have **every** feature, "
    f"mainly because **{LABELS.get(top_miss, top_miss)}** is missing for {pct(ms['overall'][top_miss])} "
    f"(it exists only for the fasting morning subsample).\\n"
    f"- Total cholesterol is missing for {pct(lab_share)} of adults. The most common pattern other than "
    f"'nothing missing' is: {patterns.index[patterns.index != 'nothing missing'][0]} "
    f"({pct(patterns.loc[patterns.index[patterns.index != 'nothing missing'][0], 'share of adults'])} of adults).\\n"
    f"- Features whose missing group has a different disease rate (p < 5%): "
    f"{', '.join(informative) if informative else 'none'}.\\n"
    f"- Unknown labels removed {dropped:,} adults ({pct(dropped / funnel.adults_20_plus.sum(), 2)} of adults)."
)
""",
    ),
    ("md", "## What is weak"),
    (
        "code",
        """\
cc = R["missing_value_strategies"]["complete_case"]
nb.say(
    f"- Dropping people with any missing feature would keep only {pct(cc['n_train_complete'] / cc['n_train_total'])} of the "
    f"training rows, so the pipeline fills values in instead (strategies compared in notebook 3 and RESULTS.md section 7).\\n"
    "- The alcohol questions skip people who did not drink in the past year; SQL sets their drinks per day to 0, "
    "which is an assumption about the skip pattern, not a measured value.\\n"
    "- The label is self-reported: it depends on people remembering and reporting what a doctor said, and on having "
    "seen a doctor at all. People without healthcare access may be labelled healthy by mistake (the data cannot tell us how often).\\n"
    "- A test of many features against the outcome will flag some by chance; treat the table above as a hint, not proof."
)
""",
    ),
]

# --------------------------------------------------------------------------------------------
NB2 = [
    (
        "md",
        """\
# 2. Distributions and risk factors

**Question.** Who reports heart disease: how does the rate change with age, sex, BMI, blood pressure,
smoking and diabetes, and do the measurements differ between people with and without it?

Counts and rates here describe the **unweighted** sample of respondents, with 95% Wilson intervals.
Section 2.5 shows what survey weights do to the same rates.
""",
    ),
    ("code", SETUP),
    ("md", "## 2.1 Rate by risk factor"),
    (
        "code",
        """\
fig = plots.plot_rate_panels(df)
nb.show(fig)
d = plots.add_risk_factor_bins(df)
tab = d.groupby("age_group", observed=True)[TARGET].agg(["mean", "sum", "count"]).rename(
    columns={"mean": "rate", "sum": "cases", "count": "adults"})
display(tab)
""",
    ),
    ("md", "## 2.2 Distributions by outcome"),
    (
        "code",
        """\
nb.show(plots.plot_distributions(df))
summary = df.groupby(TARGET)[NUMERIC].median().T.rename(columns={0: "median, no heart disease", 1: "median, heart disease"})
display(summary)
""",
    ),
    (
        "md",
        """\
## 2.3 Age and sex together, and age and smoking together

Age is the strongest driver, so the other patterns must be read within an age group.
""",
    ),
    (
        "code",
        """\
d = plots.add_risk_factor_bins(df)
by_sex = d.pivot_table(index="age_group", columns="sex", values=TARGET, aggfunc="mean", observed=True)
by_smoke = d.pivot_table(index="age_group", columns="smoking_status", values=TARGET, aggfunc="mean", observed=True)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, t, title, cols in ((axes[0], by_sex, "By age and sex", {"female": "#E69F00", "male": "#0072B2"}),
                           (axes[1], by_smoke, "By age and smoking", {"never": "#009E73", "former": "#D55E00", "current": "#CC79A7"})):
    for c in t.columns:
        ax.plot(t.index.astype(str), t[c] * 100, marker="o", label=c, color=cols[c])
    ax.set(title=title, ylabel="% with heart disease")
    ax.legend(frameon=False)
fig.tight_layout()
nb.show(fig)
display(by_sex)
""",
    ),
    (
        "md",
        """\
## 2.4 Cholesterol points the "wrong" way

Textbooks say higher cholesterol raises heart risk. Within an age band, compare the average total
cholesterol of people with and without a reported diagnosis.
""",
    ),
    (
        "code",
        """\
chol = df.groupby(["age_band", TARGET]).total_chol.mean().unstack().rename(columns={0: "no heart disease", 1: "heart disease"})
chol["difference (mg/dL)"] = chol["heart disease"] - chol["no heart disease"]
display(chol)
hdl = df.groupby(["age_band", TARGET]).hdl.mean().unstack().rename(columns={0: "no heart disease", 1: "heart disease"})
display(hdl)
""",
    ),
    ("md", "## 2.5 Survey weights: do rates change?"),
    (
        "code",
        """\
ex = df[df.was_examined & (df.exam_weight > 0)]
w = ex.groupby("age_band").apply(lambda g: np.average(g[TARGET], weights=g.exam_weight), include_groups=False)
u = ex.groupby("age_band")[TARGET].mean()
wt = pd.DataFrame({"unweighted": u, "weighted": w})
wt["ratio"] = wt.weighted / wt.unweighted
display(wt)
ws = R["weighted_sensitivity"]
nb.check("overall weighted prevalence, test examined", float(np.average(
    df[(df.split == "test") & df.was_examined & (df.exam_weight > 0)][TARGET],
    weights=df[(df.split == "test") & df.was_examined & (df.exam_weight > 0)].exam_weight)), ws["weighted_prevalence"])
""",
    ),
    ("md", "## 2.6 How related are the features to each other?"),
    (
        "code",
        """\
corr = df[NUMERIC].corr(method="spearman")
fig, ax = plt.subplots(figsize=(6.8, 5.6))
im = ax.imshow(corr, cmap="PuOr", vmin=-1, vmax=1)
ax.set_xticks(range(len(NUMERIC)), [LABELS[c] for c in NUMERIC], rotation=70, ha="right")
ax.set_yticks(range(len(NUMERIC)), [LABELS[c] for c in NUMERIC])
ax.grid(False)
fig.colorbar(im, label="Spearman correlation")
nb.show(fig)
pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack().sort_values(key=abs, ascending=False).head(5)
display(pairs.rename("spearman").to_frame())
""",
    ),
    ("md", "## What this means"),
    (
        "code",
        """\
age = tab["rate"]
sx = d.groupby("sex")[TARGET].mean()
sm = d.groupby("smoking_status")[TARGET].mean()
ages = d.groupby("smoking_status").age_years.mean()
dm = d.groupby("diabetes")[TARGET].mean()
c60 = chol.loc["60+", "difference (mg/dL)"] if "60+" in chol.index else float("nan")
nb.say(
    f"- The rate climbs from {pct(age.iloc[0])} in the youngest group to {pct(age.iloc[-1])} in the oldest "
    f"({age.iloc[-1] / max(age.iloc[0], 1e-9):.0f} times higher).\\n"
    f"- Men report heart disease at {pct(sx['male'])} and women at {pct(sx['female'])}.\\n"
    f"- By smoking: never {pct(sm['never'])}, former {pct(sm['former'])}, current {pct(sm['current'])}. "
    f"Mean age is {ages['never']:.1f} for never, {ages['former']:.1f} for former and {ages['current']:.1f} for current smokers, "
    f"so compare the groups within an age band (section 2.3).\\n"
    f"- Adults told they have diabetes report heart disease at {pct(dm['yes'])} against {pct(dm['no'])} for those who are not.\\n"
    f"- In the 60+ band, people with a reported diagnosis have average total cholesterol {c60:+.1f} mg/dL "
    f"relative to people without one."
)
""",
    ),
    ("md", "## What is weak"),
    (
        "code",
        """\
nb.say(
    "- These are **associations in a cross-section**. Smoking, BMI and blood pressure were measured at the same visit "
    "as the question, often years after any diagnosis, so the arrow of cause is unknown.\\n"
    "- Cholesterol (and often blood pressure) may look better in people with heart disease because many take "
    "cholesterol-lowering or blood pressure drugs after diagnosis. The survey has those medication questions; "
    "this project deliberately does not use them as features because they are consequences of the diagnosis.\\n"
    f"- Weighted and unweighted rates differ (the 60+ weighted-to-unweighted ratio is {wt.weighted.iloc[-1] / wt.unweighted.iloc[-1]:.2f}), "
    "so the unweighted figures describe the people surveyed, not the U.S. population.\\n"
    f"- The most correlated pair of measurements is {pairs.index[0][0]} and {pairs.index[0][1]} (Spearman {pairs.iloc[0]:.2f}, see 2.6). "
    "Correlated inputs make individual coefficients unstable even when predictions are fine."
)
""",
    ),
]

# --------------------------------------------------------------------------------------------
NB3 = [
    (
        "md",
        """\
# 3. Model comparison

**Question.** Trained on 2011 to 2014, validated on 2015 to 2016 and tested on 2017 to 2018: does XGBoost
beat the logistic regression baseline, and by enough to matter? Random forest and a model using only
age and sex are included for reference.

Rules: hyperparameters and the final model choice use the validation cycle only. The test cycle is
scored once. Intervals are bootstrap 95% intervals over test rows.
""",
    ),
    ("code", SETUP),
    ("md", "## 3.1 Test-cycle scores, recomputed from the saved predictions"),
    (
        "code",
        """\
from sklearn.metrics import average_precision_score, roc_auc_score

pred = ctx.pred
test = pred[pred.split == "test"]
valid = pred[pred.split == "valid"]
models = ["age_sex_baseline", "logistic_regression", "random_forest", "xgboost"]
rows = []
for m in models:
    r = R["models"][m]["test"]
    rows.append({
        "model": m,
        "valid PR-AUC": average_precision_score(valid[TARGET], valid[f"p_{m}"]),
        "test ROC-AUC": roc_auc_score(test[TARGET], test[f"p_{m}"]),
        "ROC-AUC 95% low": r["roc_auc_ci"][0], "ROC-AUC 95% high": r["roc_auc_ci"][1],
        "test PR-AUC": average_precision_score(test[TARGET], test[f"p_{m}"]),
        "PR-AUC 95% low": r["pr_auc_ci"][0], "PR-AUC 95% high": r["pr_auc_ci"][1],
        "Brier": r["brier"],
    })
    nb.check(f"{m} test ROC-AUC", rows[-1]["test ROC-AUC"], r["roc_auc"])
    nb.check(f"{m} test PR-AUC", rows[-1]["test PR-AUC"], r["pr_auc"])
table = pd.DataFrame(rows).set_index("model")
display(table)
print(f"Share of test adults with the outcome: {pct(test[TARGET].mean())}. A model that says 'no' to everyone scores {pct(1 - test[TARGET].mean())} accuracy.")
""",
    ),
    ("md", "## 3.2 ROC and precision-recall curves"),
    (
        "code",
        """\
y = test[TARGET].to_numpy()
probs = {m: test[f"p_{m}"].to_numpy() for m in models}
nb.show(plots.plot_roc_pr(y, probs))
""",
    ),
    (
        "md",
        "## 3.3 Confusion matrices at the recall-target threshold (set on validation, see notebook 4)",
    ),
    (
        "code",
        """\
thr = {m: R["models"][m]["recall_target"]["threshold"] for m in models}
nb.show(plots.plot_confusions(y, probs, thr, ["logistic_regression", "random_forest", "xgboost"]))
""",
    ),
    ("md", "## 3.4 Calibration: do predicted risks match observed rates?"),
    (
        "code",
        """\
nb.show(plots.plot_calibration(y, probs))
cal = pd.DataFrame({m: R["models"][m]["calibration_test"] for m in models}).T[["brier", "ece", "intercept", "slope", "mean_predicted", "observed_rate"]]
display(cal)
""",
    ),
    ("md", "## 3.5 Paired differences between models (same test rows, paired bootstrap)"),
    (
        "code",
        """\
diff = pd.DataFrame(
    {k: {"ROC-AUC gap": v["roc_auc"][0], "ROC-AUC low": v["roc_auc"][1], "ROC-AUC high": v["roc_auc"][2],
         "PR-AUC gap": v["pr_auc"][0], "PR-AUC low": v["pr_auc"][1], "PR-AUC high": v["pr_auc"][2]}
     for k, v in R["paired_differences_test"].items()}
).T
display(diff)
""",
    ),
    ("md", "## 3.6 How sensitive are the tree models to their settings?"),
    (
        "code",
        """\
for m in ["random_forest", "xgboost"]:
    g = pd.DataFrame(R["models"][m]["grid"]).sort_values("valid_pr_auc", ascending=False)
    print(m, "validation PR-AUC across", len(g), "settings: best", pct(g.valid_pr_auc.max()), "worst", pct(g.valid_pr_auc.min()))
    display(g.head(4))
""",
    ),
    (
        "md",
        """\
## 3.7 Does the time-based split matter?

A random split mixes survey years, so train and test come from the same period. Compare the
logistic regression's AUC under a random 5-fold split over all four cycles with its AUC on the
later test cycle.
""",
    ),
    (
        "code",
        """\
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from nhanes.features import make_preprocessor

C = R["models"]["logistic_regression"]["params"]["C"]
pipe = Pipeline([("pre", make_preprocessor("median_indicator")), ("clf", LogisticRegression(C=C, max_iter=5000))])
p_cv = cross_val_predict(pipe, df[FEATURES], df[TARGET], cv=StratifiedKFold(5, shuffle=True, random_state=42), method="predict_proba")[:, 1]
auc_random = roc_auc_score(df[TARGET], p_cv)
auc_time = R["models"]["logistic_regression"]["test"]["roc_auc"]
print(f"Logistic regression ROC-AUC, random 5-fold over all cycles: {pct(auc_random)}")
print(f"Logistic regression ROC-AUC, trained earlier and tested on the latest cycle: {pct(auc_time)}")
""",
    ),
    ("md", "## What this means"),
    (
        "code",
        """\
lr, xg, rf, base = (R["models"][m]["test"] for m in ["logistic_regression", "xgboost", "random_forest", "age_sex_baseline"])
dx = R["paired_differences_test"]["xgboost_minus_logistic_regression"]["roc_auc"]
dr = R["paired_differences_test"]["random_forest_minus_logistic_regression"]["roc_auc"]
db = R["paired_differences_test"]["logistic_regression_minus_age_sex_baseline"]["roc_auc"]
verdict = "does not beat" if dx[1] <= 0 <= dx[2] else ("beats" if dx[0] > 0 else "is worse than")
nb.say(
    f"- XGBoost **{verdict}** logistic regression on the test cycle: ROC-AUC {pct(xg['roc_auc'])} against {pct(lr['roc_auc'])}, "
    f"paired gap {dx[0] * 100:+.1f} points (95% interval {dx[1] * 100:+.1f} to {dx[2] * 100:+.1f}). "
    f"For the random forest the gap is {dr[0] * 100:+.1f} points ({dr[1] * 100:+.1f} to {dr[2] * 100:+.1f}).\\n"
    f"- Age and sex alone reach {pct(base['roc_auc'])}; logistic regression on all features differs from that by "
    f"{db[0] * 100:+.1f} points (paired 95% interval {db[1] * 100:+.1f} to {db[2] * 100:+.1f}).\\n"
    f"- Calibration: logistic regression predicts a mean risk of {pct(R['models']['logistic_regression']['calibration_test']['mean_predicted'])} "
    f"against an observed {pct(R['models']['logistic_regression']['calibration_test']['observed_rate'])}.\\n"
    f"- A random split gives {pct(auc_random)} against {pct(auc_time)} on the later cycle, "
    f"a difference of {(auc_random - auc_time) * 100:+.1f} points."
)
""",
    ),
    ("md", "## What is weak"),
    (
        "code",
        """\
nb.say(
    f"- Only {int(test[TARGET].sum())} test cases: intervals are wide ({(R['models']['logistic_regression']['test']['roc_auc_ci'][1] - R['models']['logistic_regression']['test']['roc_auc_ci'][0]) * 100:.1f} points wide for logistic regression), "
    "so small gaps between models cannot be resolved.\\n"
    "- The model-selection rule was written down after an exploratory run in which I had seen test scores (see DESIGN.md section 6), "
    "and the tuning grids were chosen by hand, so treat the choice as lightly tuned.\\n"
    "- The features are few and tabular; on such data logistic regression is often hard to beat. "
    "That says nothing about XGBoost on richer data.\\n"
    "- Intervals treat respondents as independent draws and ignore NHANES clustering, so they are a little too narrow."
)
""",
    ),
]

# --------------------------------------------------------------------------------------------
NB4 = [
    (
        "md",
        """\
# 4. Threshold and the recall trade-off

**Question.** Missing a person with heart disease is worse than a false alarm. If the rule must catch
at least 90% of cases, what threshold does that need, what share of people does it flag, and how many
false alarms does each case found cost?

The threshold is picked on the **validation** cycle only and then applied unchanged to the test cycle.
""",
    ),
    ("code", SETUP),
    ("md", "## 4.1 Picking the threshold on validation data (code, not a lookup)"),
    (
        "code",
        """\
from nhanes.metrics import threshold_for_recall, threshold_metrics

sel = R["selection"]["selected"]
pred = ctx.pred
valid, test = pred[pred.split == "valid"], pred[pred.split == "test"]
yv, pv = valid[TARGET].to_numpy(), valid[f"p_{sel}"].to_numpy()
yt, pt = test[TARGET].to_numpy(), test[f"p_{sel}"].to_numpy()
t90 = threshold_for_recall(yv, pv, 0.90)
nb.check("90% recall threshold", t90, R["models"][sel]["recall_target"]["threshold"])
mv, mt = threshold_metrics(yv, pv, t90), threshold_metrics(yt, pt, t90)
print(f"Selected model: {sel}. Threshold on validation for 90% recall: {pct(t90, 2)} predicted risk")
print(f"Validation recall {pct(mv['recall'])}, then on the test cycle: recall {pct(mt['recall'])}, precision {pct(mt['precision'])}, share flagged {pct(mt['flagged_share'])}")
""",
    ),
    ("md", "## 4.2 The whole trade-off curve"),
    (
        "code",
        """\
nb.show(plots.plot_threshold_tradeoff(yv, pv, yt, pt))
trade = pd.DataFrame([
    {"target recall": t["target_recall"], "threshold": t["threshold"], "valid recall": t["valid"]["recall"],
     "test recall": t["test"]["recall"], "test precision": t["test"]["precision"],
     "share flagged": t["test"]["flagged_share"], "false alarms per case found": t["test"]["false_alarms_per_case_found"]}
    for t in R["threshold_tradeoff_selected"]
]).set_index("target recall")
display(trade)
""",
    ),
    ("md", "## 4.3 What 1,000 screened adults look like"),
    (
        "code",
        """\
scale = 1000 / len(yt)
per1000 = pd.Series({
    "adults screened": 1000,
    "with heart disease (self-reported)": mt["tp"] * scale + mt["fn"] * scale,
    "cases caught": mt["tp"] * scale,
    "cases missed": mt["fn"] * scale,
    "flagged in total": (mt["tp"] + mt["fp"]) * scale,
    "false alarms": mt["fp"] * scale,
}).round(0).astype(int)
display(per1000.to_frame("per 1,000 adults"))
""",
    ),
    (
        "md",
        """\
## 4.4 If the cost of a miss is a known multiple of a false alarm

Suppose one missed case is as bad as *r* false alarms. For each *r*, choose the threshold that
minimises expected cost **on validation**, then report what it does on the test cycle.
""",
    ),
    (
        "code",
        """\
grid = np.unique(np.quantile(pv, np.linspace(0, 0.999, 400)))
rows = []
for r in (1, 5, 10, 20, 50):
    costs = [r * threshold_metrics(yv, pv, t)["fn"] + threshold_metrics(yv, pv, t)["fp"] for t in grid]
    t = float(grid[int(np.argmin(costs))])
    m = threshold_metrics(yt, pt, t)
    rows.append({"miss costs r false alarms": r, "threshold": t, "test recall": m["recall"],
                 "share flagged": m["flagged_share"], "false alarms per case found": m["false_alarms_per_case_found"]})
costtab = pd.DataFrame(rows).set_index("miss costs r false alarms")
display(costtab)
""",
    ),
    ("md", "## 4.5 Who gets flagged: subgroups at the 90% recall threshold"),
    (
        "code",
        """\
nb.show(plots.plot_subgroups(R["subgroups_selected"]))
sub = pd.DataFrame(R["subgroups_selected"]["age_band"]).set_index("group")[["n", "positives", "prevalence", "recall", "precision", "flagged_share"]]
display(sub)
sub_sex = pd.DataFrame(R["subgroups_selected"]["sex"]).set_index("group")[["n", "positives", "prevalence", "recall", "precision", "flagged_share"]]
display(sub_sex)
""",
    ),
    ("md", "## What this means"),
    (
        "code",
        """\
lo = sub.recall.idxmin()
hi = sub.recall.idxmax()
nb.say(
    f"- To catch {pct(0.90, 0)} of cases on validation the rule must flag everyone with predicted risk of at least {pct(t90, 1)}. "
    f"On the test cycle that caught {pct(mt['recall'])} of cases while flagging {pct(mt['flagged_share'])} of all adults, "
    f"which is {mt['fp'] / mt['tp']:.1f} false alarms for every case found.\\n"
    f"- Per 1,000 adults: about {int(per1000['flagged in total'])} flagged, {int(per1000['cases caught'])} cases caught and "
    f"{int(per1000['cases missed'])} missed.\\n"
    f"- Raising the target from 80% to 95% moves false alarms per case found from "
    f"{trade['false alarms per case found'].iloc[0]:.1f} to {trade['false alarms per case found'].iloc[-1]:.1f}.\\n"
    f"- The rule treats age bands very differently: recall is {pct(sub.recall[hi])} in the {hi} band (with {pct(sub.flagged_share[hi])} of that band flagged) "
    f"and {pct(sub.recall[lo])} in the {lo} band ({int(sub.positives[lo])} cases)."
)
""",
    ),
    ("md", "## What is weak"),
    (
        "code",
        """\
nb.say(
    f"- Validation recall was {pct(mv['recall'])} and test recall {pct(mt['recall'])} (a difference of {(mt['recall'] - mv['recall']) * 100:+.1f} points). "
    f"The threshold is not guaranteed to carry over to a later cycle: prevalence was {pct(yv.mean())} on validation and {pct(yt.mean())} on test, "
    "so in a new period recall can land below the target.\\n"
    f"- The overall target is met partly by how the rule treats the oldest band: {pct(sub.flagged_share[hi])} of the {hi} band is flagged, "
    f"against {pct(sub.flagged_share[lo])} of the {lo} band. One global threshold on a model dominated by age behaves like an age cut-off, "
    "and that has a high false alarm cost.\\n"
    "- The cost ratios in 4.4 are illustrations. The data contain no costs; a real deployment would need real ones.\\n"
    "- Small subgroups (see the case counts) give recall figures that move a lot with a handful of people."
)
""",
    ),
]

# --------------------------------------------------------------------------------------------
NB5 = [
    (
        "md",
        """\
# 5. What do the models use?

**Question.** Which inputs drive the predictions, in which direction, and is the story the same for
logistic regression and XGBoost? Importance tells us what the model uses, **not** what causes disease.
""",
    ),
    ("code", SETUP),
    ("md", "## 5.1 Permutation importance and SHAP"),
    (
        "code",
        """\
from nhanes.explain import shap_for_xgboost

models = ctx.models()
test = df[df.split == "test"].reset_index(drop=True)
sh = shap_for_xgboost(models["xgboost"], test)
nb.check("top mean |SHAP| value", sh["table"][0]["mean_abs_shap"], R["importance"]["shap_xgboost_mean_abs"][0]["mean_abs_shap"])
nb.show(plots.plot_importance(R["importance"]["permutation_selected"], sh["table"]))
nb.show(plots.plot_shap_beeswarm(sh))
""",
    ),
    ("md", "## 5.2 Logistic regression: odds ratios"),
    (
        "code",
        """\
orr = pd.DataFrame(R["odds_ratios_logistic"])
display(orr.head(10))
""",
    ),
    ("md", "## 5.3 Does the model's risk track observed risk across age?"),
    (
        "code",
        """\
sel = R["selection"]["selected"]
pr = ctx.pred
tt = pr[pr.split == "test"].copy()
tt["age_group"] = pd.cut(tt.age_years, [19, 29, 39, 49, 59, 69, 80], labels=["20s", "30s", "40s", "50s", "60s", "70-80"])
byage = tt.groupby("age_group", observed=True).agg(observed=(TARGET, "mean"), predicted=(f"p_{sel}", "mean"), adults=(TARGET, "size"))
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(byage.index.astype(str), byage.observed * 100, marker="o", color="#0072B2", label="observed rate")
ax.plot(byage.index.astype(str), byage.predicted * 100, marker="s", ls="--", color="#D55E00", label="mean predicted risk")
ax.set(ylabel="%", title="Test cycle, by age group")
ax.legend(frameon=False)
nb.show(fig)
display(byage)
""",
    ),
    ("md", "## 5.4 SHAP dependence: total cholesterol"),
    (
        "code",
        """\
names = sh["names"]
j = names.index("total_chol")
fig, ax = plt.subplots(figsize=(6.5, 4.2))
older = (test.age_years >= 60).to_numpy()
ax.scatter(test.total_chol[~older], sh["values"][~older, j], s=6, alpha=0.5, color="#0072B2", label="under 60")
ax.scatter(test.total_chol[older], sh["values"][older, j], s=6, alpha=0.5, color="#D55E00", label="60 and over")
ax.axhline(0, color="black", lw=0.7)
ax.set(xlabel="Total cholesterol (mg/dL), missing values omitted", ylabel="SHAP value (log-odds)", title="XGBoost: effect of total cholesterol")
ax.legend(frameon=False)
nb.show(fig)
""",
    ),
    (
        "md",
        """\
## 5.5 Take features away and refit logistic regression

Does the cholesterol finding come from a treatment effect, and does the model still work without lab
tests (which a person may not have at hand)? Same data, same tuned C, only the inputs change.
""",
    ),
    (
        "code",
        """\
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from nhanes.features import make_preprocessor

C = R["models"]["logistic_regression"]["params"]["C"]
tr, te = df[df.split == "train"], df[df.split == "test"]
labs = ["total_chol", "hdl", "hba1c", "fasting_glucose"]
variants = {
    "all features": (NUMERIC, CATEGORICAL),
    "without total cholesterol": ([c for c in NUMERIC if c != "total_chol"], CATEGORICAL),
    "without any blood test": ([c for c in NUMERIC if c not in labs], CATEGORICAL),
    "age and sex only": (["age_years"], ["sex"]),
}
rows = []
for name, (num, cat) in variants.items():
    pipe = Pipeline([("pre", make_preprocessor("median_indicator", True, num, cat)), ("clf", LogisticRegression(C=C, max_iter=5000))])
    pipe.fit(tr[FEATURES], tr[TARGET])
    rows.append({"inputs": name, "test ROC-AUC": roc_auc_score(te[TARGET], pipe.predict_proba(te[FEATURES])[:, 1])})
abl = pd.DataFrame(rows).set_index("inputs")
display(abl)
nb.check("all-features LR test ROC-AUC", abl.loc["all features", "test ROC-AUC"], R["models"]["logistic_regression"]["test"]["roc_auc"])
""",
    ),
    ("md", "## 5.6 Do the two models agree on what matters?"),
    (
        "code",
        """\
from scipy.stats import spearmanr

a = pd.DataFrame(R["importance"]["permutation_selected"]).set_index("feature").auc_drop_mean
b = pd.DataFrame(R["importance"]["permutation_xgboost"]).set_index("feature").auc_drop_mean.reindex(a.index)
rho = spearmanr(a, b)[0]
print(f"Spearman rank correlation of permutation importances, {R['selection']['selected']} vs XGBoost: {rho:.2f}")
""",
    ),
    ("md", "## What this means"),
    (
        "code",
        """\
top = R["importance"]["permutation_selected"]
first, second = top[0], top[1]
chol_or = orr.loc[orr.feature == "total_chol", "odds_ratio"]
nb.say(
    f"- **{LABELS.get(first['feature'], first['feature'])}** is the most used input: shuffling it lowers test ROC-AUC by "
    f"{first['auc_drop_mean'] * 100:.1f} points, against {second['auc_drop_mean'] * 100:.1f} for the next, "
    f"{LABELS.get(second['feature'], second['feature'])}.\\n"
    f"- Using only age and sex gives {pct(abl.loc['age and sex only', 'test ROC-AUC'])}; all features give {pct(abl.loc['all features', 'test ROC-AUC'])}; "
    f"dropping every blood test gives {pct(abl.loc['without any blood test', 'test ROC-AUC'])}.\\n"
    f"- The two models rank features similarly (Spearman {rho:.2f}).\\n"
    + (f"- The logistic odds ratio for total cholesterol is {chol_or.iloc[0]:.2f} per standard deviation, "
       f"{'below' if chol_or.iloc[0] < 1 else 'above'} 1, which is {'the opposite of' if chol_or.iloc[0] < 1 else 'consistent with'} the textbook direction." if len(chol_or) else "")
)
""",
    ),
    ("md", "## What is weak"),
    (
        "code",
        """\
nb.say(
    "- **Importance is not causation.** If total cholesterol points in the opposite direction from the textbook (see the odds ratio above), one "
    "plausible reason is treatment: people with a diagnosis are often on cholesterol-lowering drugs. This is a hypothesis; the data here cannot test it. "
    "The model uses such patterns as predictors, which is fine for prediction on a similar survey and wrong as health advice.\\n"
    "- SHAP and permutation importance split credit among correlated inputs (BMI, waist) arbitrarily, so single-feature rankings below the top are fragile.\\n"
    f"- The SHAP plot explains XGBoost. The model selected for the demo is {R['selection']['selected'].replace('_', ' ')} (RESULTS.md section 5); "
    "for logistic regression the odds-ratio table is its explanation.\\n"
    "- Nothing here shows that changing a feature (losing weight, quitting smoking) would change anyone's risk."
)
""",
    ),
]

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    build("01_data_and_missingness.ipynb", NB1)
    build("02_distributions_and_risk_factors.ipynb", NB2)
    build("03_model_comparison.ipynb", NB3)
    build("04_threshold_and_recall.ipynb", NB4)
    build("05_explanation.ipynb", NB5)
    print("wrote 5 notebooks to", OUT)
