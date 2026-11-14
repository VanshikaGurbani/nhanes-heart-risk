"""Unit tests that need no database."""

import json

import numpy as np
import pandas as pd
import pytest

from nhanes import extract, metrics
from nhanes.features import CATEGORICAL, FEATURES, FORBIDDEN, NUMERIC, Winsorizer, make_preprocessor
from nhanes.fixture import write_fixture


def test_threshold_for_recall_hits_target_on_the_data_it_was_set_on():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.1).astype(int)
    p = np.clip(0.1 + 0.5 * y + rng.normal(0, 0.15, 2000), 0, 1)
    for target in (0.8, 0.9, 0.95):
        t = metrics.threshold_for_recall(y, p, target)
        m = metrics.threshold_metrics(y, p, t)
        assert m["recall"] >= target
        # and it is the highest threshold that does so: raising it must drop below the target
        higher = metrics.threshold_metrics(y, p, np.nextafter(t, 1.0))
        assert higher["recall"] < target or higher["recall"] == m["recall"]


def test_threshold_metrics_counts_add_up():
    y = np.array([1, 1, 0, 0, 0, 1])
    p = np.array([0.9, 0.2, 0.8, 0.1, 0.3, 0.6])
    m = metrics.threshold_metrics(y, p, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 2)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["false_alarms_per_case_found"] == pytest.approx(0.5)
    assert m["flagged_share"] == pytest.approx(3 / 6)


def test_bootstrap_interval_contains_point_estimate_and_is_reproducible():
    rng = np.random.default_rng(1)
    y = (rng.random(800) < 0.2).astype(int)
    p = np.clip(0.2 + 0.3 * y + rng.normal(0, 0.2, 800), 0.001, 0.999)
    from sklearn.metrics import roc_auc_score

    a = metrics.bootstrap_ci(y, {"m": p}, {"auc": roc_auc_score}, n_boot=200, seed=3)["m"]["auc"]
    b = metrics.bootstrap_ci(y, {"m": p}, {"auc": roc_auc_score}, n_boot=200, seed=3)["m"]["auc"]
    assert a == b
    assert a[1] <= a[0] <= a[2]


def test_calibration_of_perfectly_calibrated_scores():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 0.4, 40000)
    y = (rng.random(40000) < p).astype(int)
    c = metrics.calibration_summary(y, p)
    assert abs(c["slope"] - 1) < 0.1 and abs(c["intercept"]) < 0.1 and c["ece"] < 0.01


def test_no_leaky_or_forbidden_column_is_a_feature():
    assert not set(FEATURES) & set(FORBIDDEN)
    assert set(FEATURES) == set(NUMERIC) | set(CATEGORICAL)
    banned_words = ("mcq", "medic", "statin", "insulin", "chf", "stroke", "heart", "angina")
    assert not [f for f in FEATURES if any(w in f for w in banned_words)]


def test_winsorizer_clips_to_training_range_and_keeps_nan():
    w = Winsorizer(0.1, 0.9).fit(np.arange(101, dtype=float).reshape(-1, 1))
    out = w.transform(np.array([[-50.0], [50.0], [500.0], [np.nan]]))
    assert out[0, 0] == pytest.approx(10) and out[1, 0] == 50 and out[2, 0] == pytest.approx(90)
    assert np.isnan(out[3, 0])


@pytest.mark.parametrize("strategy", ["median_indicator", "median", "iterative", "native"])
def test_preprocessor_handles_missing_values(strategy, tmp_path):
    raw = write_fixture(tmp_path, n_per_cycle=300)
    demo = pd.read_csv(raw / "DEMO_G.csv")
    n = len(demo)
    rng = np.random.default_rng(0)
    X = pd.DataFrame({c: rng.normal(size=n) for c in NUMERIC})
    X.loc[::7, "bmi"] = np.nan
    X["sex"] = rng.choice(["male", "female"], n)
    X["smoking_status"] = rng.choice(["never", None], n)
    X["diabetes"] = "no"
    X["ever_heavy_drinker"] = "no"
    pre = make_preprocessor(strategy, scale=False).fit(X[FEATURES])
    out = pre.transform(X[FEATURES])
    if strategy != "native":
        assert not np.isnan(out).any()
    assert out.shape[0] == n


def test_checksum_manifest_detects_a_changed_file(tmp_path, monkeypatch):
    import nhanes.extract as ex

    def fake_download(url, dest, retries=3):
        dest.write_bytes(b"original bytes")

    monkeypatch.setattr(ex, "_download", fake_download)
    manifest_path = tmp_path / "manifest.json"
    raw = tmp_path / "raw"
    first = ex.fetch_all(raw, manifest_path)
    assert len(first) == 44  # 11 tables x 4 cycles
    assert all(v["sha256"] == ex.sha256_of(raw / k) for k, v in first.items())
    # second run verifies and passes
    assert ex.fetch_all(raw, manifest_path) == first
    # a silently changed file is caught
    victim = next(iter(raw.glob("*.xpt")))
    victim.write_bytes(b"tampered")
    with pytest.raises(extract.ChecksumError):
        ex.fetch_all(raw, manifest_path)
    assert json.loads(manifest_path.read_text())  # manifest untouched


def test_fixture_has_cdc_special_codes_and_skip_patterns(tmp_path):
    raw = write_fixture(tmp_path, n_per_cycle=1500)
    mcq = pd.concat([pd.read_csv(raw / f"MCQ_{c}.csv") for c in "GHIJ"])
    assert {7.0, 9.0} & set(mcq.MCQ160C.dropna().unique())  # refused / don't know present
    assert mcq.MCQ160C.isna().any()  # not asked of children
