# Design notes

Why the project is built the way it is, what the trade-offs were, and what I would change next.
Numbers are not repeated here; they live in [RESULTS.md](RESULTS.md), which code writes.

## 1. Data choices

**Cycles.** NHANES releases two-year cycles. I use G (2011 to 2012), H, I and J (2017 to 2018). All four
use the same auscultatory blood pressure exam (`BPX`), the same medical conditions questionnaire
(`MCQ160B` to `MCQ160F`), and nearly the same alcohol, smoking and diabetes questions. The CDC's
2017 to March 2020 pre-pandemic files (`P_` prefix) are not used for two reasons: they replace `BPX`
with an oscillometric device (`BPXO`), which would change what "systolic blood pressure" means between
train and test, and they already contain the 2017 to 2018 participants, so using both J and P would put
the same people in two places. I checked that every file name exists on the CDC site before writing
any code (11 tables times 4 cycles is 44 files), and the manifest pins each file's SHA-256.

**Tables.** Demographics, body measures, blood pressure, total and HDL cholesterol, HbA1c, fasting glucose,
smoking, diabetes, alcohol, and the medical conditions questionnaire. They are joined on `SEQN`.
Alcohol is the one place the questionnaire changed: cycle J redesigned it, so only the two questions that
exist in all four cycles (`ALQ130` average drinks on a drinking day, `ALQ151` ever heavy drinker) are used.

**Population.** Adults 20 and older, because the heart questions are only asked from age 20. Age is
top-coded at 80 by the CDC, so "80" means "80 or older".

## 2. The label

| Question | In the label? | Why |
|---|---|---|
| Coronary heart disease (`MCQ160C`) | yes | ischemic heart disease |
| Angina (`MCQ160D`) | yes | ischemic heart disease |
| Heart attack (`MCQ160E`) | yes | ischemic heart disease |
| Congestive heart failure (`MCQ160B`) | no | different condition, often a consequence of the others |
| Stroke (`MCQ160F`) | no | cerebrovascular, not cardiac |

A person is positive on any yes, negative only if all three are no, and dropped if the answers are
ambiguous (no yes, but a refused or don't-know). Dropping rather than guessing keeps the negative class
clean. The label is a self-report, which I say everywhere it matters.

## 3. Leakage

A feature leaks if it exists because of the diagnosis. Excluded on purpose:

* all medication questions (heart, blood pressure, cholesterol, insulin, diabetes pills): treatment follows diagnosis;
* congestive heart failure and stroke answers: consequences of, or entangled with, the outcome;
* every column of the label itself;
* survey weights, design variables, cycle and ids.

`tests/test_unit.py` asserts that no forbidden column is in the feature list and that no feature name
contains words like medication, stroke or angina.

One leak I cannot remove: lab values are affected by treatment (cholesterol-lowering drugs make
cholesterol look better in people with the disease). The notebooks show this explicitly and
notebook 5 refits the model without cholesterol and without any blood test.

Education and income are in the CDC files but not in the model table. They are plausible predictors, but a
demo that asks for your income to estimate your heart risk is a bad idea, and they are proxies for access
to care, which also drives who gets diagnosed. Race and ethnicity are kept only to report performance by
group, never as an input.

## 4. Schemas and SQL

```
raw     one table per CDC table, all cycles stacked, every column as published, plus `cycle`
ref     cycle -> split role (train, valid, test)
stg     one VIEW per questionnaire or exam topic: cleaned, recoded, still keyed by (seqn, cycle)
model   heart_risk: one row per adult, all features and the label. cohort_funnel: who was dropped and why
```

* **Raw stays raw.** The only thing done at load time is decoding how SAS transport files store an exact
  zero (see section 8). Cleaning happens in SQL so the logic is visible, reviewable and diffable.
* **Views for staging, a table for the model.** Staging views cost nothing and are always consistent with
  `raw`; `model.heart_risk` is materialised because training reads it repeatedly and it carries a primary key.
* **Special codes in SQL.** Refused (7, 9, 77, 99, 777, 999) become NULL with `NULLIF`, written next to the
  question they belong to, because the code value differs by question (for example `ALQ130` uses 777 and 999,
  `SMQ020` uses 7 and 9). A test builds people with every code and asserts what comes out.
* **Blood pressure.** Up to four readings per person are averaged; a reading of 0 is treated as "no reading"
  and excluded.
* **Left joins for exam data.** People who were interviewed but not examined keep a row with NULL labs.
  Dropping them would discard some people and would hide the missingness that the notebooks examine.
* **Everything is re-runnable.** `load_raw` drops `stg` and `model`, replaces `raw` in one transaction and
  `apply_transforms` rebuilds the rest. `tests` run the load and transform twice and compare tables.

PostgreSQL rather than SQLite or DuckDB because the request was a real server-based SQL pipeline; Docker
Compose makes it one command, and CI uses a service container.

## 5. Validation (stage 4)

Checks fail the run rather than warn. They are grouped as: raw row counts equal file row counts and no
duplicate `(cycle, seqn)`; model row uniqueness and funnel arithmetic; core columns complete; value ranges
set at **impossible** values, not unusual ones (for example systolic pressure above 280), so real outliers
do not stop the pipeline; allowed categories and no code leaking through as a value; missingness per cycle
(catches a variable vanishing from one cycle); label equals the OR of its components; plausible prevalence
per cycle; temporal order of the split; enough positives per split. Tests corrupt a clean database in five
different ways and assert the right check fails.

## 6. Evaluation design

* **Temporal split.** Train 2011 to 2014, validate 2015 to 2016, test 2017 to 2018. A random split would
  mix survey years and flatter the model; notebook 3 measures how much.
* **Three uses of validation, zero of test.** Hyperparameters (small grids), the model-selection rule and the
  decision threshold all use the validation cycle. The test cycle is scored once per model.
* **Metrics.** Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier score, calibration slope and intercept,
  confusion matrices. Accuracy is reported with the "predict everyone healthy" baseline next to it, because
  the classes are imbalanced. PR-AUC is the tuning metric because it is the one sensitive to the minority class.
* **Reference model.** A logistic regression on age and sex alone. Without it a headline AUC cannot be judged,
  because age alone already ranks people well.
* **Bootstrap.** 1,000 resamples of the test rows. Model comparisons use a **paired** bootstrap (the same
  resampled rows for both models), which is the right way to compare two models scored on the same people.
* **Threshold.** The highest threshold that catches at least 90% of the cases on the validation cycle. The
  test cycle then shows what really happens. The false alarm cost is stated as false alarms per case found
  and as the share of all adults flagged. Other targets (80%, 85%, 95%) are shown too.
* **Model selection rule.** A model replaces logistic regression only if the paired bootstrap interval for its
  validation PR-AUC gap is entirely above zero. Honest disclosure: I wrote this rule after a first exploratory
  run showed the three models were close, and I had seen that run's test scores; the rule itself uses only
  validation data. It keeps the simplest adequate model for the app.
* **Missing values.** Compared on the same tuned hyperparameters: median fill with missing flags (default),
  median only, iterative imputation, and XGBoost's native handling, plus a complete-case run on the rows that
  have every feature. Numeric features are also clipped to the training 0.5th and 99.5th percentiles (a few
  respondents report implausible drink counts).
* **Survey weights.** Models are unweighted because the goal is to rank the people in the sample, and the
  weights are for population estimates. A sensitivity section shows weighted prevalence, weighted AUC and a
  weighted-training logistic regression. Not done: design-based variance (strata and PSUs).
* **Explanation.** Permutation importance for any model, SHAP for XGBoost, odds ratios for logistic regression.

## 7. Trade-offs I made on purpose

| Decision | Alternative | Why this way |
|---|---|---|
| Unweighted models | survey-weighted | simpler, and the sensitivity check shows the effect |
| Tune on validation, no CV | nested cross-validation | keeps the temporal logic, cheaper, but fewer tuning degrees of freedom |
| Median fill with flags as default | iterative or native | not distinguishable from the others in the comparison, and the app can reuse it for a "Not sure" box |
| App uses the selected model, not the best on test | pick the best test score | would turn the test cycle into a tuning set |
| One recall target (90%) as the headline | cost-based threshold | the brief asked for a recall target; notebook 4 also shows cost ratios |
| CI uses a synthetic fixture | download real data in CI | CDC downloads are slow and could break CI; the fixture has the same columns, codes and skip patterns |

## 8. Bugs found and fixed while building (worth knowing)

* **SAS transport zeros.** pandas reads a stored 0 in an `.XPT` file as 5.4e-79, a tiny positive number, not 0.
  A "greater than 0" test would have counted a diastolic reading of 0, or an age of 0, as a valid value. The
  loader now sets any value with absolute size below 1e-30 to exactly 0; a unit test covers it and the
  blood pressure SQL excludes the zeros.
* **Float error in the threshold.** `(1 - 0.8) * 100` is 19.999999999999996, which silently gave a slightly
  higher recall than asked; a small epsilon and a test now pin the behaviour.
* **The validator caught my own fixture.** The first synthetic data had systolic below diastolic pressure for
  a few rows; the validation stage failed the run, which is the intended behaviour.

## 9. What I would do next

* Design-based standard errors with the survey strata and PSUs.
* Add the 2017 to March 2020 files as a second, separate test (with a harmonised blood pressure variable).
* A calibration-aware, subgroup-aware threshold rather than one global number.
* Real costs for a miss versus a false alarm, from a clinical collaborator.
* A model card and drift monitoring if this were ever deployed (it should not be, in its present form).
