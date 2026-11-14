"""SQL transform tests on hand-made people that exercise every special code and skip pattern."""

import numpy as np
import pandas as pd
import pytest

from nhanes.config import CYCLES, TABLES
from nhanes.db import read_sql
from nhanes.load import apply_transforms, load_raw

NAN = np.nan


def _edge_tables() -> dict[str, pd.DataFrame]:
    """People seqn 1..12 in cycle G, one adult in each other cycle (so every cycle exists)."""
    ids = list(range(1, 13))
    age = {1: 50, 2: 50, 3: 50, 4: 50, 5: 15, 6: 60, 7: 60, 8: 60, 9: 60, 10: 60, 11: 60, 12: 60}
    demo = pd.DataFrame(
        {
            "SEQN": ids,
            "RIDSTATR": [2] * 10 + [1, 2],
            "RIAGENDR": [1, 2] * 6,
            "RIDAGEYR": [age[i] for i in ids],
            "RIDRETH3": [3] * 12,
            "WTMEC2YR": [1000.0] * 10 + [0.0, 1000.0],
            "SDMVPSU": [1] * 12,
            "SDMVSTRA": [100] * 12,
        }
    )
    # MCQ160B, C, D, E, F per person
    mcq_vals = {
        1: (2, 1, 2, 2, 2),  # coronary heart disease yes -> 1
        2: (2, 2, 2, 2, 2),  # all no -> 0
        3: (2, 2, 9, 2, 2),  # don't know on angina, no yes -> unknown, dropped
        4: (2, 9, 2, 1, 2),  # heart attack yes, don't know on CHD -> 1
        5: (NAN, NAN, NAN, NAN, NAN),  # child
        6: (2, 2, 2, 2, 2),
        7: (2, 2, 2, 2, 2),
        8: (2, 2, 2, 2, 2),
        9: (1, 2, 2, 2, 2),  # congestive heart failure only -> label 0
        10: (2, 2, 2, 2, 1),  # stroke only -> label 0
        11: (2, 7, 2, 2, 2),  # refused on CHD, no yes -> unknown, dropped
        12: (2, 2, 2, 2, 2),
    }
    mcq = pd.DataFrame(
        [
            {
                "SEQN": i,
                **dict(
                    zip(["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"], v, strict=True)
                ),
            }
            for i, v in mcq_vals.items()
        ]
    )
    smq = pd.DataFrame(
        {
            "SEQN": ids,
            "SMQ020": [2, 2, 2, 2, NAN, 7, 1, 1, 1, 1, 1, 9],
            "SMQ040": [NAN, NAN, NAN, NAN, NAN, NAN, 3, 1, 2, 3, 9, NAN],
        }
    )
    diq = pd.DataFrame({"SEQN": ids, "DIQ010": [2, 2, 2, 2, 2, 9, 1, 3, 2, 2, 7, 2]})
    alq = pd.DataFrame(
        {
            "SEQN": ids,
            "ALQ130": [NAN, NAN, NAN, NAN, NAN, 999, NAN, 3, 777, NAN, NAN, 4],
            "ALQ151": [2, 2, 2, 2, NAN, 2, 2, 1, 7, 9, NAN, 2],
        }
    )
    bpx = pd.DataFrame(
        {
            "SEQN": ids,
            "BPXSY1": [120.0] * 12,
            "BPXSY2": [NAN] * 12,
            "BPXSY3": [NAN] * 12,
            "BPXSY4": [NAN] * 12,
            "BPXDI1": [80.0] * 12,
            "BPXDI2": [NAN] * 12,
            "BPXDI3": [NAN] * 12,
            "BPXDI4": [NAN] * 12,
        }
    )
    # person 8: readings 120, 0 (a real zero as the loader decodes it), 130, missing / 80, 0, 70, missing
    bpx.loc[bpx.SEQN == 8, ["BPXSY2", "BPXSY3"]] = [0.0, 130.0]
    bpx.loc[bpx.SEQN == 8, ["BPXDI1", "BPXDI2", "BPXDI3"]] = [80.0, 0.0, 70.0]
    exam = [i for i in ids if i != 11]
    return {
        "DEMO": demo,
        "BMX": pd.DataFrame(
            {"SEQN": exam, "BMXBMI": 25.0, "BMXWAIST": 90.0, "BMXHT": 170.0, "BMXWT": 70.0}
        ),
        "BPX": bpx[bpx.SEQN.isin(exam)],
        "TCHOL": pd.DataFrame({"SEQN": exam, "LBXTC": 190.0}),
        "HDL": pd.DataFrame({"SEQN": exam, "LBDHDD": 50.0}),
        "GHB": pd.DataFrame({"SEQN": exam, "LBXGH": 5.4}),
        "GLU": pd.DataFrame({"SEQN": [1, 2], "LBXGLU": [95.0, 101.0]}),
        "SMQ": smq,
        "DIQ": diq,
        "ALQ": alq,
        "MCQ": mcq,
    }


@pytest.fixture(scope="module")
def model(engine, tmp_path_factory):
    raw = tmp_path_factory.mktemp("edge_raw")
    g = _edge_tables()
    for i, cycle in enumerate(CYCLES):
        for table in TABLES:
            if cycle.letter == "G":
                df = g[table]
            else:  # one clean adult per other cycle
                sid = 1000 * (i + 1)
                df = g[table][g[table].SEQN == 2].copy()
                df["SEQN"] = sid
            df.to_csv(raw / f"{table}_{cycle.letter}.csv", index=False)
    load_raw(engine, raw)
    apply_transforms(engine)
    return read_sql(
        engine, "SELECT * FROM model.heart_risk WHERE cycle = 'G' ORDER BY seqn"
    ).set_index("seqn")


def test_label_rules(model):
    assert model.loc[1, "heart_disease"] == 1  # coronary heart disease
    assert model.loc[2, "heart_disease"] == 0  # all no
    assert model.loc[4, "heart_disease"] == 1  # one yes beats a don't know
    assert model.loc[9, "heart_disease"] == 0 and bool(model.loc[9, "chf"])  # CHF is not the label
    assert model.loc[10, "heart_disease"] == 0 and bool(model.loc[10, "stroke"])  # nor is stroke


def test_unknown_labels_and_children_are_dropped(model):
    assert 3 not in model.index  # don't know on angina, no yes
    assert 11 not in model.index  # refused on CHD, no yes
    assert 5 not in model.index  # child, never asked
    assert set(model.index) == {1, 2, 4, 6, 7, 8, 9, 10, 12}


def test_special_codes_become_null_never_numbers(model):
    assert pd.isna(model.loc[6, "smoking_status"])  # SMQ020 = 7 refused
    assert pd.isna(model.loc[6, "diabetes"])  # DIQ010 = 9 don't know
    assert pd.isna(model.loc[6, "drinks_per_day"])  # ALQ130 = 999
    assert pd.isna(model.loc[9, "drinks_per_day"])  # ALQ130 = 777
    assert pd.isna(model.loc[9, "ever_heavy_drinker"])  # ALQ151 = 7
    assert pd.isna(model.loc[12, "smoking_status"])  # SMQ020 = 9
    assert model.drinks_per_day.dropna().max() <= 10  # no 777 / 999 survived


def test_recodes_and_skip_patterns(model):
    assert model.loc[1, "smoking_status"] == "never"
    assert model.loc[7, "smoking_status"] == "former"  # smoked 100+, now not at all
    assert model.loc[8, "smoking_status"] == "current"
    assert model.loc[1, "drinks_per_day"] == 0  # alcohol section answered, ALQ130 skipped
    assert model.loc[8, "drinks_per_day"] == 3
    assert model.loc[7, "diabetes"] == "yes" and model.loc[8, "diabetes"] == "borderline"
    assert model.loc[1, "sex"] == "male" and model.loc[2, "sex"] == "female"


def test_blood_pressure_averages_valid_readings_and_drops_zero(model):
    assert model.loc[8, "sbp_mean"] == pytest.approx(125.0)  # (120 + 130) / 2, the 0 is excluded
    assert model.loc[8, "dbp_mean"] == pytest.approx(75.0)  # (80 + 70) / 2
    assert model.loc[1, "sbp_mean"] == pytest.approx(120.0)


def test_people_without_exam_keep_a_row_with_null_labs(model):
    assert model.loc[9, "total_chol"] == 190.0
    assert pd.isna(model.loc[1, "fasting_glucose"]) is False  # person 1 has a fasting glucose
    assert pd.isna(model.loc[4, "fasting_glucose"])  # not in the fasting subsample
    assert model.loc[1, "split"] == "train"


def test_xpt_zero_decoding():
    from nhanes.load import fix_xport_zeros

    df = pd.DataFrame({"a": [5.397605346934028e-79, 1.0, np.nan, 0.0, 120.0]})
    out = fix_xport_zeros(df)
    assert out.a.tolist()[:2] == [0.0, 1.0] and np.isnan(out.a[2]) and out.a[4] == 120.0
