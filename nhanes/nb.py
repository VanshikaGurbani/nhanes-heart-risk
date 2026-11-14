"""Helpers shared by the notebooks: load the model table, results and predictions, and print
numbers that are computed (never typed) together with a check against docs/results/results.json."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nhanes import plots
from nhanes.config import Outputs, db_url, default_outputs, fixture_db_url
from nhanes.db import get_engine, read_sql

CHECKS: list[tuple[str, bool]] = []


@dataclass
class Context:
    out: Outputs
    engine: object
    df: pd.DataFrame
    R: dict
    pred: pd.DataFrame
    fixture: bool

    def models(self) -> dict:
        """All fitted pipelines saved by `nhanes run-all`."""
        return joblib.load(self.out.artifacts / "models.joblib")

    def sql(self, query: str) -> pd.DataFrame:
        return read_sql(self.engine, query)


def is_fixture() -> bool:
    return os.environ.get("NHANES_NOTEBOOK_MODE", "real") == "fixture"


def context() -> Context:
    out = default_outputs()
    fixture = is_fixture()
    engine = get_engine(fixture_db_url() if fixture else db_url())
    df = read_sql(engine, "SELECT * FROM model.heart_risk ORDER BY seqn")
    R = json.loads(out.results_json.read_text(encoding="utf-8"))
    pred = pd.read_csv(out.artifacts / "predictions.csv")
    return Context(out, engine, df, R, pred, fixture)


def say(text: str) -> None:
    """Show Markdown whose numbers were formatted by code."""
    from IPython.display import Markdown, display

    display(Markdown(text))


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


def check(name: str, recomputed: float, reported: float, tol: float = 1e-6) -> None:
    """Print whether a number computed in the notebook matches the one in results.json."""
    ok = bool(np.isclose(recomputed, reported, rtol=tol, atol=tol))
    CHECKS.append((name, ok))
    flag = "matches" if ok else "DIFFERS"
    print(f"Check: {name}: notebook {recomputed:.6g} vs results.json {reported:.6g} ({flag})")


def show(fig, name: str | None = None) -> None:
    """Display a figure inline and optionally save a PNG next to the pipeline's figures."""
    if name:
        out = default_outputs()
        out.fig_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out.fig_dir / f"{name}.png", bbox_inches="tight", facecolor="white", dpi=150)
    from IPython.display import display

    display(fig)
    plt.close(fig)


__all__ = ["Context", "Path", "check", "context", "pct", "plots", "say", "show"]
