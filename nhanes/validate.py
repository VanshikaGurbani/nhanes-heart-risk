"""Stage 4: automated data quality checks on the PostgreSQL tables. Any failure stops the run.

Checks cover row counts, duplicates, value ranges, allowed categories, missingness, the label
definition, and the temporal split. Hard bounds are about impossible values (a systolic blood
pressure of 900), not about unusual ones, so real outliers do not stop the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import Engine

from nhanes.config import CYCLES, TABLES
from nhanes.db import read_sql


class ValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Thresholds:
    """Bounds used by the checks. Real-data defaults; tests use a smaller fixture variant."""

    min_adults_per_cycle: int = 4000
    min_positives_per_split: int = 200
    prevalence_range: tuple[float, float] = (0.02, 0.20)
    max_missing_core: float = 0.0  # age, sex, split, label may never be missing
    max_missing_exam: float = 0.25  # bmi, bp, labs from the exam, per cycle
    max_missing_fasting_glucose: float = 0.75  # morning fasting subsample only
    max_missing_questionnaire: float = 0.05  # smoking, diabetes
    max_missing_alcohol: float = 0.40
    ranges: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "age_years": (20, 80),
            "bmi": (10, 90),
            "waist_cm": (40, 200),
            "sbp_mean": (50, 280),
            "dbp_mean": (0, 160),
            "total_chol": (40, 900),
            "hdl": (5, 250),
            "hba1c": (2.5, 20),
            "fasting_glucose": (20, 800),
            "drinks_per_day": (0, 100),
        }
    )

    @classmethod
    def fixture(cls) -> Thresholds:
        return cls(
            min_adults_per_cycle=100,
            min_positives_per_split=5,
            prevalence_range=(0.01, 0.40),
            max_missing_exam=0.40,
            max_missing_fasting_glucose=0.90,
            max_missing_alcohol=0.60,
        )


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


CATEGORIES = {
    "sex": {"male", "female"},
    "split": {"train", "valid", "test"},
    "age_band": {"20-39", "40-59", "60+"},
    "smoking_status": {"never", "former", "current"},
    "diabetes": {"yes", "no", "borderline"},
    "ever_heavy_drinker": {"yes", "no"},
}
FEATURES_EXAM = ["bmi", "waist_cm", "sbp_mean", "dbp_mean", "total_chol", "hdl", "hba1c"]


def run_checks(engine: Engine, t: Thresholds | None = None) -> list[Check]:
    t = t or Thresholds()
    checks: list[Check] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append(Check(name, bool(passed), detail))

    # 1. Raw layer: every file's rows arrived, and (cycle, seqn) is unique in DEMO.
    log = read_sql(engine, "SELECT table_name, cycle, n_rows FROM raw.load_log")
    bad = []
    for table in TABLES:
        name = table.lower()
        actual = read_sql(engine, f"SELECT cycle, COUNT(*) AS n FROM raw.{name} GROUP BY cycle")
        merged = log[log.table_name == name].merge(actual, on="cycle", how="left")
        mism = merged[merged.n_rows != merged.n.fillna(-1)]
        bad += [f"{name}/{r.cycle}: file {r.n_rows} vs table {r.n}" for r in mism.itertuples()]
    add("raw_row_counts_match_files", not bad, "; ".join(bad) or "all raw tables match their files")

    dup_raw = read_sql(
        engine,
        "SELECT COUNT(*) AS n FROM "
        "(SELECT 1 FROM raw.demo GROUP BY seqn, cycle HAVING COUNT(*) > 1) x",
    ).n[0]
    add("raw_demo_no_duplicate_seqn", dup_raw == 0, f"{dup_raw} duplicated (cycle, seqn) pairs")

    # 2. Model table: row counts, funnel, duplicates.
    m = read_sql(engine, "SELECT * FROM model.heart_risk")
    funnel = read_sql(engine, "SELECT * FROM model.cohort_funnel")
    add("model_nonempty", len(m) > 0, f"{len(m)} rows")
    add(
        "model_unique_seqn",
        m.seqn.is_unique,
        f"{int(m.seqn.duplicated().sum())} duplicated seqn values",
    )
    per_cycle = m.groupby("cycle").size()
    small = {c.letter: int(per_cycle.get(c.letter, 0)) for c in CYCLES}
    small = {k: v for k, v in small.items() if v < t.min_adults_per_cycle}
    add(
        "adults_per_cycle_minimum",
        not small,
        f"cycles below {t.min_adults_per_cycle}: {small}"
        if small
        else f"every cycle has at least {t.min_adults_per_cycle} labelled adults "
        f"(min {int(per_cycle.min())})",
    )
    add(
        "funnel_reconciles",
        int(funnel.adults_with_label.sum()) == len(m)
        and (funnel.adults_with_label <= funnel.adults_with_mcq).all()
        and (funnel.adults_with_mcq <= funnel.adults_20_plus).all(),
        f"funnel labelled adults {int(funnel.adults_with_label.sum())} vs model rows {len(m)}",
    )

    # 3. Core columns never missing.
    core = ["seqn", "cycle", "split", "age_years", "sex", "age_band", "heart_disease"]
    miss_core = {c: float(m[c].isna().mean()) for c in core}
    add(
        "core_columns_complete",
        all(v <= t.max_missing_core for v in miss_core.values()),
        f"missing share: { {k: round(v, 4) for k, v in miss_core.items() if v > 0} or 'none' }",
    )

    # 4. Ranges (hard physical bounds).
    for col, (lo, hi) in t.ranges.items():
        s = m[col].dropna()
        out = int(((s < lo) | (s > hi)).sum())
        add(
            f"range_{col}",
            out == 0,
            f"{out} values outside [{lo}, {hi}] (observed {s.min():.4g} to {s.max():.4g})"
            if len(s)
            else "column is entirely missing",
        )
    add(
        "systolic_above_diastolic",
        int((m.sbp_mean <= m.dbp_mean).sum()) == 0,
        f"{int((m.sbp_mean <= m.dbp_mean).sum())} rows with systolic <= diastolic",
    )

    # 5. Categories, and no refused / don't know code leaked through as a value.
    for col, allowed in CATEGORIES.items():
        values = set(m[col].dropna().unique())
        extra = values - allowed
        add(
            f"categories_{col}", not extra, f"unexpected values: {sorted(extra)}" if extra else "ok"
        )
    leaked = int(m.drinks_per_day.isin([77, 99, 777, 999]).sum())
    add("no_special_codes_in_drinks", leaked == 0, f"{leaked} rows carry 77/99/777/999 as drinks")

    # 6. Missingness per cycle, so a cycle where a variable vanished is caught.
    def miss_by_cycle(col: str, limit: float) -> None:
        rates = m.groupby("cycle")[col].apply(lambda s: s.isna().mean())
        worst = rates.idxmax()
        add(
            f"missingness_{col}",
            bool((rates <= limit).all()),
            f"worst cycle {worst}: {rates.max():.1%} missing (limit {limit:.0%})",
        )

    for col in FEATURES_EXAM:
        miss_by_cycle(col, t.max_missing_exam)
    miss_by_cycle("fasting_glucose", t.max_missing_fasting_glucose)
    miss_by_cycle("smoking_status", t.max_missing_questionnaire)
    miss_by_cycle("diabetes", t.max_missing_questionnaire)
    miss_by_cycle("drinks_per_day", t.max_missing_alcohol)

    # 7. Label: binary, consistent with its components, plausible rate.
    add(
        "label_binary",
        set(m.heart_disease.unique()) <= {0, 1},
        f"values {sorted(int(v) for v in m.heart_disease.unique())}",
    )
    comp = (
        m.chd.fillna(False).astype(bool)
        | m.angina.fillna(False).astype(bool)
        | m.heart_attack.fillna(False).astype(bool)
    ).astype(int)
    inconsistent = int((comp != m.heart_disease).sum())
    add(
        "label_matches_components",
        inconsistent == 0,
        f"{inconsistent} rows disagree with CHD/angina/MI",
    )
    lo, hi = t.prevalence_range
    prev = m.groupby("cycle").heart_disease.mean()
    add(
        "prevalence_plausible",
        bool(((prev >= lo) & (prev <= hi)).all()),
        "per-cycle prevalence " + ", ".join(f"{k} {v:.1%}" for k, v in prev.items()),
    )

    # 8. Temporal split: train cycles strictly before valid before test, enough positives each.
    order = {c.letter: i for i, c in enumerate(CYCLES)}
    m["_o"] = m.cycle.map(order)
    rng = m.groupby("split")["_o"].agg(["min", "max"])
    ok_order = (
        {"train", "valid", "test"} <= set(rng.index)
        and rng.loc["train", "max"] < rng.loc["valid", "min"]
        and rng.loc["valid", "max"] < rng.loc["test", "min"]
    )
    add("temporal_split_order", ok_order, "train < valid < test by survey cycle")
    pos = m.groupby("split").heart_disease.sum()
    short = {k: int(v) for k, v in pos.items() if v < t.min_positives_per_split}
    add(
        "positives_per_split",
        not short and {"train", "valid", "test"} <= set(pos.index),
        f"positives {pos.astype(int).to_dict()} (minimum {t.min_positives_per_split})",
    )
    exam = m[m.was_examined]
    add(
        "exam_weights_positive",
        bool((exam.exam_weight > 0).all()),
        f"{int((exam.exam_weight <= 0).sum())} examined people with weight <= 0",
    )
    return checks


def assert_valid(engine: Engine, t: Thresholds | None = None) -> list[Check]:
    """Run all checks, print a report, raise ValidationError if any failed."""
    checks = run_checks(engine, t)
    for c in checks:
        print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}")
    failed = [c for c in checks if not c.passed]
    if failed:
        raise ValidationError(
            f"{len(failed)} of {len(checks)} data quality checks failed: "
            + ", ".join(c.name for c in failed)
        )
    return checks
