"""A small synthetic stand-in for the NHANES files, for tests and CI only.

It has the same table names, column names, special codes (7, 9, 77, 99, 777, 999), skip patterns
and missingness shapes as the real files, so the real extract-free pipeline code runs on it
unchanged. The numbers are random draws from a made-up risk model. They carry no medical meaning
and no result from this fixture is ever reported as a finding.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nhanes.config import CYCLES


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_cycle(letter: str, idx: int, n: int, rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    seqn = 100000 + idx * 20000 + np.arange(n)
    age = np.where(rng.random(n) < 0.25, rng.integers(0, 20, n), rng.integers(20, 81, n))
    adult = age >= 20
    male = rng.random(n) < 0.49
    examined = rng.random(n) < 0.92
    diabetes_true = rng.random(n) < _sigmoid(-4.0 + 0.035 * age)
    smoker = rng.random(n) < 0.18
    ever_smoked = smoker | (rng.random(n) < 0.22)
    bmi = np.clip(rng.normal(28 + 0.02 * (age - 40), 6), 14, 70)
    waist = np.clip(bmi * 3.4 + rng.normal(0, 7, n), 55, 170)
    sbp = np.clip(rng.normal(105 + 0.4 * age, 14), 75, 220)
    dbp = np.minimum(np.clip(rng.normal(68 + 0.05 * age, 9), 35, 120), sbp - 15)
    chol = np.clip(rng.normal(185 + 0.2 * age, 38), 90, 400)
    hdl = np.clip(rng.normal(53 - 6 * male, 14), 20, 120)
    a1c = np.clip(rng.normal(5.3 + 1.6 * diabetes_true, 0.5), 4, 14)
    glu = np.clip(rng.normal(98 + 45 * diabetes_true, 15), 60, 400)

    logit = (
        -7.6 + 0.075 * age + 0.7 * male + 0.9 * diabetes_true + 0.5 * smoker
        + 0.012 * (sbp - 120) + 0.1 * idx
    )  # fmt: skip
    disease = (rng.random(n) < _sigmoid(logit)) & adult
    which = rng.integers(1, 8, n)  # bit mask over chd, angina, heart attack

    def code(flag, no_value=2.0):
        out = np.where(flag, 1.0, no_value)
        refuse = rng.random(n) < 0.005
        out = np.where(refuse, np.where(rng.random(n) < 0.7, 9.0, 7.0), out)
        return np.where(adult, out, np.nan)

    demo = pd.DataFrame(
        {
            "SEQN": seqn,
            "RIDSTATR": np.where(examined, 2, 1),
            "RIAGENDR": np.where(male, 1, 2),
            "RIDAGEYR": age,
            "RIDRETH3": rng.choice([1, 2, 3, 4, 6, 7], n),
            "WTMEC2YR": np.where(examined, rng.uniform(5000, 90000, n), 0.0),
            "SDMVPSU": rng.integers(1, 3, n),
            "SDMVSTRA": rng.integers(90, 150, n),
        }
    )

    def exam_only(df: pd.DataFrame, cols: list[str], p_missing: float) -> pd.DataFrame:
        df = df[examined].copy()
        for c in cols:
            df.loc[rng.random(len(df)) < p_missing, c] = np.nan
        return df

    bmx = exam_only(
        pd.DataFrame({"SEQN": seqn, "BMXBMI": bmi, "BMXWAIST": waist,
                      "BMXHT": rng.normal(168, 10, n), "BMXWT": bmi * 2.9}),
        ["BMXBMI", "BMXWAIST"], 0.05,
    )  # fmt: skip
    bpx_cols = {}
    for k in range(1, 5):
        bpx_cols[f"BPXSY{k}"] = np.where(
            rng.random(n) < (0.05 * k), np.nan, sbp + rng.normal(0, 3, n)
        )
        bpx_cols[f"BPXDI{k}"] = np.where(
            rng.random(n) < (0.05 * k), np.nan, dbp + rng.normal(0, 3, n)
        )
    bpx = exam_only(pd.DataFrame({"SEQN": seqn, **bpx_cols}), ["BPXSY1", "BPXDI1"], 0.04)
    tchol = exam_only(pd.DataFrame({"SEQN": seqn, "LBXTC": chol}), ["LBXTC"], 0.05)
    hdl_df = exam_only(pd.DataFrame({"SEQN": seqn, "LBDHDD": hdl}), ["LBDHDD"], 0.0)
    ghb = exam_only(pd.DataFrame({"SEQN": seqn, "LBXGH": a1c}), ["LBXGH"], 0.03)
    glu_df = pd.DataFrame({"SEQN": seqn, "LBXGLU": glu})[examined & (rng.random(n) < 0.45)]

    smq = pd.DataFrame(
        {
            "SEQN": seqn,
            "SMQ020": np.where(adult, code(ever_smoked), np.nan),
            "SMQ040": np.where(adult & ever_smoked, np.where(smoker, 1.0, 3.0), np.nan),
        }
    )
    diq = pd.DataFrame(
        {
            "SEQN": seqn,
            "DIQ010": np.where(diabetes_true, 1.0, np.where(rng.random(n) < 0.04, 3.0, 2.0)),
        }
    )
    drinker = adult & (rng.random(n) < 0.7)
    alq130 = np.where(drinker, rng.integers(1, 9, n).astype(float), np.nan)
    alq130 = np.where(drinker & (rng.random(n) < 0.01), 999.0, alq130)
    alq = pd.DataFrame(
        {
            "SEQN": seqn,
            "ALQ130": alq130,
            "ALQ151": np.where(adult, code(rng.random(n) < 0.1), np.nan),
        }
    )[adult]
    mcq = pd.DataFrame(
        {
            "SEQN": seqn,
            "MCQ160B": code(rng.random(n) < 0.03),
            "MCQ160C": code(disease & ((which & 1) > 0)),
            "MCQ160D": code(disease & ((which & 2) > 0)),
            "MCQ160E": code(disease & ((which & 4) > 0)),
            "MCQ160F": code(rng.random(n) < 0.03),
        }
    )
    return {
        "DEMO": demo, "BMX": bmx, "BPX": bpx, "TCHOL": tchol, "HDL": hdl_df, "GHB": ghb,
        "GLU": glu_df, "SMQ": smq, "DIQ": diq, "ALQ": alq, "MCQ": mcq,
    }  # fmt: skip


def write_fixture(out_dir: Path, n_per_cycle: int = 1500, seed: int = 7) -> Path:
    """Write DEMO_G.csv ... MCQ_J.csv into out_dir. Returns out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i, cycle in enumerate(CYCLES):
        for table, df in make_cycle(cycle.letter, i, n_per_cycle, rng).items():
            df.to_csv(out_dir / f"{table}_{cycle.letter}.csv", index=False)
    return out_dir
