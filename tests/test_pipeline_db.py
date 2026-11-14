"""Validation and end-to-end tests. They need PostgreSQL (docker compose up -d) and are skipped
when it is not reachable. They use a separate database, never the real one."""

import json
import os
import re

import pandas as pd
import pytest
from sqlalchemy import text

from nhanes.config import ROOT, fixture_db_url
from nhanes.db import read_sql
from nhanes.fixture import write_fixture
from nhanes.load import apply_transforms, load_raw
from nhanes.metrics import threshold_for_recall
from nhanes.validate import Thresholds, ValidationError, assert_valid, run_checks

pytestmark = pytest.mark.db

EM_DASH, EN_DASH = chr(0x2014), chr(0x2013)  # built from code points so this file has none


@pytest.fixture()
def built(engine, tmp_path):
    raw = write_fixture(tmp_path / "raw", n_per_cycle=1200)
    load_raw(engine, raw)
    apply_transforms(engine)
    return engine


def _failed(engine) -> set[str]:
    return {c.name for c in run_checks(engine, Thresholds.fixture()) if not c.passed}


def test_validation_passes_on_clean_fixture(built):
    assert _failed(built) == set()
    assert_valid(built, Thresholds.fixture())


def test_out_of_range_value_fails_the_run(built):
    with built.begin() as c:
        c.execute(
            text(
                "UPDATE model.heart_risk SET sbp_mean = 900 WHERE seqn = (SELECT MIN(seqn) FROM model.heart_risk)"
            )
        )
    assert "range_sbp_mean" in _failed(built)
    with pytest.raises(ValidationError):
        assert_valid(built, Thresholds.fixture())


def test_duplicate_raw_rows_fail_the_run(built):
    with built.begin() as c:
        c.execute(text("INSERT INTO raw.demo SELECT * FROM raw.demo LIMIT 1"))
    failed = _failed(built)
    assert "raw_demo_no_duplicate_seqn" in failed
    assert "raw_row_counts_match_files" in failed


def test_a_variable_vanishing_from_one_cycle_fails_the_run(built):
    with built.begin() as c:
        c.execute(text("UPDATE model.heart_risk SET total_chol = NULL WHERE cycle = 'I'"))
    assert "missingness_total_chol" in _failed(built)


def test_refused_code_leaking_into_a_value_fails_the_run(built):
    with built.begin() as c:
        c.execute(
            text(
                "UPDATE model.heart_risk SET smoking_status = '7' WHERE seqn = (SELECT MIN(seqn) FROM model.heart_risk)"
            )
        )
        c.execute(
            text(
                "UPDATE model.heart_risk SET drinks_per_day = 999 WHERE seqn = (SELECT MAX(seqn) FROM model.heart_risk)"
            )
        )
    failed = _failed(built)
    assert "categories_smoking_status" in failed
    assert "no_special_codes_in_drinks" in failed


def test_label_that_disagrees_with_components_fails(built):
    with built.begin() as c:
        c.execute(
            text(
                "UPDATE model.heart_risk SET heart_disease = 1 - heart_disease WHERE seqn = (SELECT MIN(seqn) FROM model.heart_risk)"
            )
        )
    assert "label_matches_components" in _failed(built)


def test_reloading_and_retransforming_is_idempotent(built, tmp_path):
    before = read_sql(built, "SELECT * FROM model.heart_risk ORDER BY seqn")
    apply_transforms(built)  # run the SQL again on the same raw tables
    after = read_sql(built, "SELECT * FROM model.heart_risk ORDER BY seqn")
    pd.testing.assert_frame_equal(before, after)
    load_raw(built, tmp_path / "raw")  # and the load again, from the same files
    apply_transforms(built)
    again = read_sql(built, "SELECT * FROM model.heart_risk ORDER BY seqn")
    pd.testing.assert_frame_equal(before, again)


@pytest.fixture(scope="module")
def fixture_run(engine, tmp_path_factory):
    """Run the whole pipeline once on the fixture, twice, into two folders."""
    from nhanes import config
    from nhanes.pipeline import run_all

    out1 = tmp_path_factory.mktemp("run1")
    out2 = tmp_path_factory.mktemp("run2")
    old = os.environ.get("NHANES_FIXTURE_DB_URL")
    os.environ["NHANES_FIXTURE_DB_URL"] = engine.url.render_as_string(hide_password=False)
    try:
        assert config.fixture_db_url() == os.environ["NHANES_FIXTURE_DB_URL"]
        run_all(fixture=True, out_dir=out1, n_boot=60, quick=True)
        run_all(fixture=True, out_dir=out2, n_boot=60, quick=True)
    finally:
        if old is None:
            os.environ.pop("NHANES_FIXTURE_DB_URL")
        else:
            os.environ["NHANES_FIXTURE_DB_URL"] = old
    return out1, out2


def _assert_close(a, b, path=""):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            _assert_close(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (u, v) in enumerate(zip(a, b, strict=True)):
            _assert_close(u, v, f"{path}[{i}]")
    elif isinstance(a, float):
        assert a == pytest.approx(b, rel=1e-9, abs=1e-12), path
    else:
        assert a == b, path


def test_pipeline_is_idempotent_and_writes_everything(fixture_run):
    out1, out2 = fixture_run
    r1 = json.loads((out1 / "docs/results/results.json").read_text())
    r2 = json.loads((out2 / "docs/results/results.json").read_text())
    _assert_close(r1, r2)  # same data and seeds give the same numbers (up to float round-off)
    # the report only prints rounded values, so the two reports are identical text
    t1 = (out1 / "docs/RESULTS.md").read_text(encoding="utf-8")
    assert t1 == (out2 / "docs/RESULTS.md").read_text(encoding="utf-8")
    for name in ("docs/RESULTS.md", "models/final_model.joblib", "models/model_card.json",
                 "docs/figures/roc_pr.png", "docs/figures/shap_beeswarm.png", "artifacts/predictions.csv"):  # fmt: skip
        assert (out1 / name).exists(), name


def test_threshold_comes_from_validation_scores_not_test(fixture_run):
    out1, _ = fixture_run
    R = json.loads((out1 / "docs/results/results.json").read_text())
    pred = pd.read_csv(out1 / "artifacts/predictions.csv")
    valid = pred[pred.split == "valid"]
    for m, v in R["models"].items():
        t = threshold_for_recall(valid.heart_disease.to_numpy(), valid[f"p_{m}"].to_numpy(), 0.9)
        assert v["recall_target"]["threshold"] == pytest.approx(t)
        assert v["recall_target"]["valid"]["recall"] >= 0.9


def test_train_valid_test_are_disjoint_in_time_and_people(fixture_run):
    out1, _ = fixture_run
    pred = pd.read_csv(out1 / "artifacts/predictions.csv")
    cycles = pred.groupby("split").cycle.agg(set).to_dict()
    assert cycles == {"valid": {"I"}, "test": {"J"}}
    assert pred.seqn.is_unique


def test_results_are_written_without_em_dashes_and_with_percentages(fixture_run):
    out1, _ = fixture_run
    md = (out1 / "docs/RESULTS.md").read_text(encoding="utf-8")
    assert EM_DASH not in md and EN_DASH not in md
    assert re.search(r"ROC-AUC of \*\*\d+\.\d%", md)
    assert "not a clinical diagnosis" in md


def test_streamlit_app_runs_and_shows_the_disclaimer(fixture_run, monkeypatch):
    from streamlit.testing.v1 import AppTest

    out1, _ = fixture_run
    monkeypatch.setenv("NHANES_MODEL_DIR", str(out1 / "models"))
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
    assert any("not medical advice" in e.value.lower() for e in at.error)
    at.button[0].click().run()  # the form's submit button
    assert not at.exception
    headers = [h.value for h in at.header]
    assert any(h.startswith("Estimated chance") for h in headers)
    assert any("threshold" in m.value.lower() for m in at.markdown)


def test_no_em_dashes_anywhere_in_the_repo():
    bad = []
    skip = {
        ".venv",
        ".git",
        "data",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "nhanes_heart_risk.egg-info",
    }
    for p in ROOT.rglob("*"):
        if not p.is_file() or set(p.relative_to(ROOT).parts) & skip:
            continue
        if p.suffix.lower() not in {
            ".py",
            ".md",
            ".sql",
            ".yml",
            ".yaml",
            ".toml",
            ".ipynb",
            ".json",
            ".txt",
        }:
            continue
        if EM_DASH in p.read_text(encoding="utf-8", errors="ignore"):
            bad.append(str(p.relative_to(ROOT)))
    assert not bad, f"em dash found in {bad}"


def test_fixture_db_url_never_points_at_the_real_database(monkeypatch):
    monkeypatch.delenv("NHANES_FIXTURE_DB_URL", raising=False)
    assert fixture_db_url().endswith("/nhanes_fixture")
