"""Streamlit demo: enter health information, get an estimate of how common heart disease reports
are among people like you in NHANES survey data. Educational demo, not medical advice.

Nothing is stored or sent anywhere: the model runs inside this process and the inputs live only
in the browser session. Run it with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("NHANES_MODEL_DIR", ROOT / "models"))
FEATURES = [
    "age_years", "bmi", "waist_cm", "sbp_mean", "dbp_mean", "total_chol", "hdl", "hba1c",
    "fasting_glucose", "drinks_per_day", "sex", "smoking_status", "diabetes", "ever_heavy_drinker",
]  # fmt: skip

st.set_page_config(page_title="NHANES heart disease estimate", page_icon=None, layout="centered")


@st.cache_resource
def load_model():
    card = json.loads((MODEL_DIR / "model_card.json").read_text(encoding="utf-8"))
    return joblib.load(MODEL_DIR / "final_model.joblib"), card


def optional_number(label: str, lo: float, hi: float, default: float, step: float, help_: str):
    """A number input with an 'I do not know' box, which sends a missing value to the model."""
    c1, c2 = st.columns([3, 1])
    unknown = c2.checkbox("Not sure", key=f"na_{label}", help="Leave this out of the estimate")
    val = c1.number_input(label, min_value=lo, max_value=hi, value=default, step=step, help=help_,
                          disabled=unknown)  # fmt: skip
    return np.nan if unknown else float(val)


def risk_band(p: float, base: float) -> str:
    ratio = p / base if base else 1.0
    if ratio < 0.5:
        return "well below"
    if ratio < 0.9:
        return "somewhat below"
    if ratio <= 1.1:
        return "about the same as"
    if ratio <= 2:
        return "somewhat above"
    return "well above"


st.title("Heart disease estimate from survey data")
st.error(
    "**Educational demo, not medical advice.** This is a statistics project built on public "
    "survey answers. It cannot diagnose anything and must not guide any health decision. "
    "If you are worried about your heart, talk to a doctor."
)

try:
    pipe, card = load_model()
except FileNotFoundError:
    st.warning("No trained model found. Run `nhanes run-all` first, then reload this page.")
    st.stop()

st.caption(
    "Privacy: nothing you enter is saved, logged or sent anywhere. The model runs in this session "
    "only. Please do not enter real personal medical records anyway; use made-up values to explore."
)

with st.form("inputs"):
    st.subheader("About you")
    c1, c2 = st.columns(2)
    age = c1.number_input(
        "Age (years)", 20, 80, 55, help="The survey tops out at 80 (80 means 80 or older)."
    )
    sex = c2.selectbox("Sex", ["female", "male"])
    st.subheader("Measurements")
    bmi = optional_number(
        "BMI (kg/m2)", 12.0, 80.0, 27.0, 0.1, "Weight in kg divided by height in m, squared."
    )
    waist = optional_number(
        "Waist (cm)", 50.0, 180.0, 94.0, 0.5, "Measured just above the hip bone."
    )
    sbp = optional_number("Systolic blood pressure (mmHg)", 70.0, 240.0, 124.0, 1.0, "Top number.")
    dbp = optional_number(
        "Diastolic blood pressure (mmHg)", 20.0, 140.0, 76.0, 1.0, "Bottom number."
    )
    st.subheader("Blood tests")
    chol = optional_number("Total cholesterol (mg/dL)", 70.0, 500.0, 190.0, 1.0, "")
    hdl = optional_number("HDL cholesterol (mg/dL)", 10.0, 150.0, 52.0, 1.0, "")
    a1c = optional_number("HbA1c (%)", 3.5, 15.0, 5.4, 0.1, "Average blood sugar marker.")
    glu = optional_number(
        "Fasting glucose (mg/dL)", 40.0, 400.0, 98.0, 1.0, "Only if you fasted for the test."
    )
    st.subheader("Habits and history")
    smoking = st.selectbox(
        "Smoking",
        ["never", "former", "current"],
        help="'Never' means fewer than 100 cigarettes in your life.",
    )
    diabetes = st.selectbox("Told by a doctor you have diabetes", ["no", "borderline", "yes"])
    drinks = optional_number(
        "Drinks on a typical drinking day", 0.0, 30.0, 1.0, 1.0, "0 if you do not drink."
    )
    heavy = st.selectbox("Ever had 4 or 5 drinks nearly every day", ["no", "yes"])
    go = st.form_submit_button("Show estimate")

if go:
    row = pd.DataFrame(
        [
            {
                "age_years": float(age),
                "bmi": bmi,
                "waist_cm": waist,
                "sbp_mean": sbp,
                "dbp_mean": dbp,
                "total_chol": chol,
                "hdl": hdl,
                "hba1c": a1c,
                "fasting_glucose": glu,
                "drinks_per_day": drinks,
                "sex": sex,
                "smoking_status": smoking,
                "diabetes": diabetes,
                "ever_heavy_drinker": heavy,
            }
        ]
    )[FEATURES]
    p = float(pipe.predict_proba(row)[0, 1])
    thr = card["threshold"]
    base = card["base_rate_train"]
    flagged = p >= thr

    st.header(f"Estimated chance: {p:.1%}")
    st.write(
        f"Among adults in the NHANES survey answering like you, about **{p:.1%}** reported that a doctor "
        f"had told them they had coronary heart disease, angina or a heart attack. That is "
        f"{risk_band(p, base)} the average in the data ({base:.1%}). This is a survey statistic about "
        "groups of people, not a prediction about you."
    )
    st.subheader("Screening flag (demo)")
    st.write(
        f"Decision threshold used: **{thr:.1%}**. It was chosen on one set of survey years so that "
        f"{card['target_recall']:.0%} of people reporting heart disease would be flagged "
        f"(validation years). On the later test years it flagged "
        f"{card['test_recall_at_threshold']:.0%} of them, and it flagged {card['test_share_flagged']:.0%} "
        f"of all adults, with about {card['test_false_alarms_per_case_found']:.1f} false alarms for every "
        "person with heart disease it found."
    )
    if flagged:
        st.warning(
            "Your estimate is at or above the threshold, so this demo screening rule would flag you. "
            "A flag is a prompt to ask a doctor, not a finding. Many flagged people are healthy."
        )
    else:
        st.success(
            "Your estimate is below the threshold, so this demo rule would not flag you. That does not "
            "mean you are healthy: the rule still misses some people."
        )
    n_missing = int(row.isna().sum(axis=1).iloc[0])
    if n_missing:
        st.info(f"{n_missing} value(s) were left out; the model filled them with typical values.")

with st.expander("What this model is and its limits"):
    st.markdown(
        f"""
* Model: **{card["model"].replace("_", " ")}**, trained on NHANES {card["trained_on"]} survey years and
  tested on {card["tested_on"]} (ROC-AUC {card["test_roc_auc"]:.1%} on the test years).
* The outcome is **self-reported**: whether a person said a doctor had told them they had coronary heart
  disease, angina or a heart attack. It is not a clinical diagnosis.
* NHANES samples people living in U.S. households. Results do **not** transfer to hospital patients or to
  other countries.
* People who already have heart disease are often on cholesterol or blood pressure treatment, which
  changes those numbers. The model learns from that mix, so it cannot say what causes what.
* Age explains most of the estimate. See the README for measured results and limitations.
"""
    )
