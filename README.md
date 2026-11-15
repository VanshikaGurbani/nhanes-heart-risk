# NHANES Heart Risk: predicting self-reported heart disease from CDC survey data

An end-to-end data pipeline and modeling project: it downloads public CDC health survey files, cleans them in PostgreSQL, trains three models, tests them on a later survey period, and shows the result in a small web app. Educational only, not medical advice.

**Live demo:** https://heartrisk-prediction-system.streamlit.app/ (the app may take a few seconds to wake up if nobody has used it for a while)

![Trade-off between catching cases and false alarms](docs/figures/threshold_tradeoff.png)

## What is this?

Every two years the CDC runs a health survey called **NHANES**: people answer questions, get measured
(weight, waist, blood pressure) and give a blood sample. One question asks whether a doctor ever told
them they had coronary heart disease, angina or a heart attack.

This project asks: **from age, sex, body measurements, blood pressure, blood tests, smoking, diabetes
and alcohol answers, how well can we tell who reports heart disease?** It covers the whole path:
download and checksum the CDC files, load them into PostgreSQL, clean and join them with SQL, run 42
automated data quality checks, train a logistic regression, a random forest and XGBoost, and test them
on a *later* survey period than the one they learned from. The numbers below are written by code, not
typed.

The labels are **self-reported**, this is **not a clinical diagnosis**, and results **do not transfer to
hospital patients** (see [Limitations](#limitations)). Independent work, not affiliated with the CDC.

## Key results

Data: 22,512 U.S. adults from four survey cycles, 1,637 of whom reported heart disease. Models learned
from 2011 to 2014, were tuned on 2015 to 2016, and were scored once on 2017 to 2018.

<!-- PLAIN:START -->
**The task.** The survey asked adults whether a doctor ever told them they had coronary heart disease, angina or a heart attack. Using only age, sex, body measurements, blood pressure, blood tests, smoking, diabetes and alcohol answers, can a model tell who said yes? In the 2017-2018 test group, 8.3% of adults said yes (457 of 5,534).

**How good is the ranking?** Pick one person who said yes and one who said no, at random. The model gives the higher risk score to the person who said yes in about 83 of 100 such pairs (that is the ROC-AUC of 82.9%). A coin flip would get 50 of 100. A model that only knows age and sex gets about 80. So the other health measurements add a real but modest gain.

**Is the fancy model better?** The three models are statistically tied: logistic regression (the simple one), random forest and XGBoost all score about 83 of 100. XGBoost minus logistic regression is +0.1 pp, and the 95% interval (-0.4 pp to +0.6 pp) includes zero, so the data cannot show a winner. The simple model is used because it is easier to explain.

**What matters most.** Age, by a wide margin (the top feature is `age_years`). Older people report heart disease far more often, so a model that knows age already does most of the work.

**Why not just report accuracy?** Saying 'no heart disease' to everyone is right 91.7% of the time and finds nobody. So the model is judged on how many real cases it finds (recall) and how many people it flags who did not say yes (false alarms).

**The trade-off at the chosen setting.** The cut-off was set on the earlier 2015-2016 data so that at least 90% of cases would be caught, then applied once to the 2017-2018 data. Out of every 1,000 adults in the test group:

* about 83 said yes, and the model flagged 75 of them;
* it missed about 7;
* it also flagged about 397 people who did not say yes (false alarms), which is 5.3 false alarms for every case found;
* in total it flagged 47% of adults, so catching almost every case means flagging nearly half of everyone.

**Who gets flagged.** Mostly older adults: 94% of the 60+ group is flagged and 99% of their cases are caught, but only 1% of the 20-39 group is flagged and just 18% of their cases are caught (17 cases, too few to trust). The recall target is met mostly by flagging most older adults.

**What this is not.** It is not a diagnosis and not a screening tool. The answers are self-reported, the data cover the U.S. household population, and the results do not transfer to hospital patients.

**Words used in the tables.** *ROC-AUC*: the pair-ranking score above (50% is chance, 100% is perfect). *PR-AUC*: a score that focuses on how clean the flagged group is, useful when cases are rare. *Recall*: the share of real cases the model flags. *Flagged*: scored above the cut-off. *False alarm*: flagged but did not say yes. *95% interval*: the range the true value probably falls in, found by re-running the test on 1,000 reshuffled copies of the data.
<!-- PLAIN:END -->

The same results as a table:

<!-- TABLE:START -->
| Model | Test ROC-AUC (95% CI) | Test PR-AUC (95% CI) | Recall at the 90% threshold | Share of adults flagged | False alarms per case found |
|---|---|---|---|---|---|
| Age and sex only (reference) | 79.5% (77.5% to 81.3%) | 22.7% (19.8% to 25.8%) | 92.1% | 53.1% | 6.0 |
| Logistic regression | 82.9% (81.1% to 84.5%) | 27.6% (24.3% to 31.5%) | 91.2% | 47.2% | 5.3 |
| Random forest | 82.9% (81.2% to 84.6%) | 28.4% (24.9% to 32.5%) | 88.2% | 41.8% | 4.7 |
| XGBoost | 83.0% (81.2% to 84.6%) | 27.7% (24.3% to 31.5%) | 90.8% | 45.8% | 5.1 |
<!-- TABLE:END -->

Full numbers, confidence intervals, subgroups and sensitivity checks: [docs/RESULTS.md](docs/RESULTS.md).

## Tech stack

* **Language and ML:** Python, pandas, scikit-learn, XGBoost, SHAP
* **Data and SQL:** PostgreSQL 17, SQLAlchemy, plain SQL transforms
* **Packaging and CI:** Docker Compose, pytest, ruff, GitHub Actions (Postgres service container)
* **Delivery:** Streamlit demo, five Jupyter notebooks with saved outputs

## Quick start

Needs Python 3.12 or 3.13 (3.14 is excluded: some pinned packages do not support it yet), Docker, and
about 100 MB of disk for the CDC files.

```bash
git clone https://github.com/VanshikaGurbani/nhanes-heart-risk.git
cd nhanes-heart-risk
python -m venv .venv
.venv/Scripts/activate            # macOS and Linux: source .venv/bin/activate
pip install -e ".[serve,dev,notebooks]"
docker compose up -d              # PostgreSQL 17 on 127.0.0.1:5433
nhanes run-all                    # download, load, SQL, validate, train, report (about 3 minutes)
streamlit run app/streamlit_app.py
```

`nhanes run-all` is safe to run again. Other commands: `nhanes fetch`, `nhanes load`,
`nhanes transform`, `nhanes validate`, `nhanes run-notebooks`, and `nhanes run-all --fixture --out DIR`
(a small synthetic dataset, used by CI).

## Challenges and trade-offs

* **A silent data bug.** The CDC files store zero in a way that pandas reads as 5.4e-79, so a blood
  pressure of 0 looked like a valid tiny reading. The loader now decodes it, and a unit test covers it.
* **Keeping the label honest.** The label is yes to coronary heart disease, angina or heart attack.
  Heart failure and stroke are left out of both the label and the features, and no medication question
  is used, because treatment follows a diagnosis and would leak the answer.
* **Not fooling myself on the test set.** Tuning, model choice and the decision threshold all use only the
  2015 to 2016 data; the 2017 to 2018 data is scored once. The result is that XGBoost did not beat the
  simpler logistic regression, and the README says so.
* **A recall target has a price.** Catching 90% of cases means flagging nearly half of all adults, mostly
  by flagging almost every older adult. The results report that cost instead of hiding it.

## How it works

```
CDC website ──> 1 extract ──> 2 load ──> 3 transform (SQL) ──> 4 validate ──> 5 features + models ──> 6 report
 .xpt files     sha256 per    PostgreSQL   sql/*.sql builds      42 automated    logistic regression,    docs/RESULTS.md
 (11 tables x   file, raw     schema raw   stg views and the     checks, any     random forest,          figures, model,
  4 cycles)     files kept                 model.heart_risk      failure stops   XGBoost, bootstrap      Streamlit app
                untouched                  table                 the run         intervals, SHAP
```

| Stage | Module | What it does |
|---|---|---|
| 1 Extract | [`nhanes/extract.py`](nhanes/extract.py) | Downloads 44 SAS transport files (11 tables for each of 4 survey cycles), records a SHA-256 for each in [`manifest/nhanes_sha256.json`](manifest/nhanes_sha256.json), and refuses to continue if a later download no longer matches. Raw files are never modified. |
| 2 Load | [`nhanes/load.py`](nhanes/load.py) | Bulk loads every table into PostgreSQL schema `raw` with `COPY`, tagged by cycle. Rebuilds from scratch each time. |
| 3 Transform | [`sql/`](sql) | Plain SQL files: special codes (7, 9, 77, 99, 777, 999) become NULL, blood pressure readings are averaged, the label is defined, tables are joined into `model.heart_risk` (one row per adult). |
| 4 Validate | [`nhanes/validate.py`](nhanes/validate.py) | 42 checks: row counts against the files, duplicates, value ranges, allowed categories, no refused-code leaks, missingness per cycle, label consistency, temporal split order. A failure stops the run with a non-zero exit code. |
| 5 Models | [`nhanes/features.py`](nhanes/features.py), [`modeling.py`](nhanes/modeling.py), [`evaluate.py`](nhanes/evaluate.py), [`explain.py`](nhanes/explain.py) | Preprocessing, tuning on validation only, evaluation on the test cycle, bootstrap intervals, calibration, subgroups, SHAP. |
| 6 One command | [`nhanes/pipeline.py`](nhanes/pipeline.py) | `nhanes run-all` runs everything, idempotently. |

The design reasons (why three schemas, why SQL for cleaning, why a temporal split, what was
excluded and why) are in [docs/DESIGN.md](docs/DESIGN.md).

### Which survey years

| Role | Cycle | Years |
|---|---|---|
| Train | G, H | 2011 to 2012, 2013 to 2014 |
| Validate (tuning, model choice, threshold) | I | 2015 to 2016 |
| Test (scored once) | J | 2017 to 2018 |

The 2017 to March 2020 "pre-pandemic" files (prefix `P_`) are deliberately not used: they swap the
blood pressure device (oscillometric `BPXO` instead of auscultatory `BPX`) and they already contain
the 2017 to 2018 data, so they would overlap with cycle J.

### The label

`heart_disease = 1` if the person said yes to coronary heart disease, angina **or** heart attack.
It is 0 only if all three were answered no. If there is no yes and any answer is refused, don't know or
missing, the label is unknown and the person is left out. Congestive heart failure and stroke are
different conditions: they are not part of the label and are **never used as features**. No
medication question (heart, blood pressure, cholesterol, diabetes) is used either, because treatment
is a consequence of a diagnosis.

### Survey weights

NHANES gives each person a weight (how many Americans they represent). **The models are unweighted.**
[RESULTS.md section 9](docs/RESULTS.md) checks what weights change: weighted prevalence and weighted
AUC, and a logistic regression trained with weights. Design-based standard errors (strata and PSUs)
are not computed.

## Notebooks

Five notebooks, committed **with outputs from the real data**. Each opens with a plain-English
question and closes with *what this means* and *what is weak*. Numbers in their text are computed by
code (the closing cells are code cells that format their sentences), and every number a notebook
recomputes that the pipeline also reports prints a `Check:` line that must say `matches`.

| Notebook | The question it answers |
|---|---|
| [01 Data and missingness](notebooks/01_data_and_missingness.ipynb) | Who is in the analysis, how is the label built, what is missing and is the missingness informative? |
| [02 Distributions and risk factors](notebooks/02_distributions_and_risk_factors.ipynb) | How do heart disease rates change with age, sex, BMI, blood pressure, smoking and diabetes, and what do survey weights do? |
| [03 Model comparison](notebooks/03_model_comparison.ipynb) | Does XGBoost beat logistic regression? ROC, PR, confusion matrices, calibration, paired differences. |
| [04 Threshold and recall](notebooks/04_threshold_and_recall.ipynb) | What does a 90% recall target cost in false alarms, per 1,000 adults and per subgroup? |
| [05 Explanation](notebooks/05_explanation.ipynb) | What do the models use, and why must importance not be read as cause? |

## Streamlit demo

```bash
streamlit run app/streamlit_app.py
```

Enter made-up health values and get an estimate, the decision threshold in use, and a plain-language
note. It is labelled **educational demo, not medical advice** at the top, collects nothing, stores
nothing and sends nothing anywhere: the model runs inside the Streamlit process. Fields you leave
as "Not sure" are passed to the model as missing values.

## Tests and CI

* `pytest`: threshold and bootstrap maths, a checksum manifest that detects a changed file, SQL transform
  tests on hand-made people covering every refused code and skip pattern, validation checks that
  must fail on corrupted data, a check that features contain nothing forbidden, a check that thresholds
  come from validation scores only, pipeline idempotency, Streamlit app test, and a scan for em dashes.
  `pytest` needs PostgreSQL running (`docker compose up -d`); tests use their own database
  (`nhanes_test`) and never touch the real one.
* `ruff check` and `ruff format --check`.
* GitHub Actions ([ci.yml](.github/workflows/ci.yml)) runs all of the above on Python 3.12 and 3.13 with a
  PostgreSQL service container, then runs the whole pipeline and all five notebooks on the **synthetic
  fixture** (not the real CDC data). A notebook `Check:` line that says `DIFFERS` fails the job.

## Limitations

* **Self-reported labels.** "A doctor told me" depends on memory, on having seen a doctor, and on
  what the doctor said. It is not a measured diagnosis.
* **Not clinical.** NHANES covers the U.S. household population. Nothing here transfers to hospital
  patients, other countries, or to anyone's real health decisions.
* **Mostly age.** Age carries most of the signal (the age-and-sex-only reference model is reported next to
  every result). The other features add a measured but modest gain.
* **Treatment contaminates the labs.** People with heart disease are often treated, which lowers
  cholesterol and blood pressure. The models learn that pattern, so it is prediction on this survey,
  not cause.
* **The recall target is met mostly by flagging most older adults**, and the 20 to 39 band has very few
  cases, so recall there is unreliable (see RESULTS.md section 8).
* **One test cycle, few cases.** Intervals are wide enough that small model differences cannot be resolved,
  and they ignore NHANES clustering, so they are a little too narrow.
* **The base rate moves.** Prevalence changes from cycle to cycle, so a threshold set on one cycle does not hit
  its target exactly on the next.
* **Alcohol skip pattern.** People who did not drink in the past year are given 0 drinks per day by assumption.
* **Education, income and race and ethnicity are not features** (race and ethnicity appear only in the
  descriptive subgroup table), so the demo asks for things a person can measure or know.
* **No external validation** on any other dataset or hospital, and no prospective test.
* Tested on Python 3.12 and 3.13 in CI (fixture data); the full real-data run was done locally on 3.13.

## Data and credits

NHANES public-use files from the U.S. CDC National Center for Health Statistics, downloaded from
`https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/`. The data are in the public domain. Cycle letters
G, H, I and J correspond to 2011 to 2012, 2013 to 2014, 2015 to 2016 and 2017 to 2018.
