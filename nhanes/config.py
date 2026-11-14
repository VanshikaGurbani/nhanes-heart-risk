"""Central configuration: survey cycles, file names, database URL, paths.

Everything that names a CDC file or a split lives here so the extract, load and SQL stages
cannot drift apart.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("NHANES_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
FIXTURE_DIR = DATA_DIR / "fixture_raw"
SQL_DIR = ROOT / "sql"
DOCS_DIR = ROOT / "docs"
FIG_DIR = DOCS_DIR / "figures"
MODEL_DIR = ROOT / "models"
MANIFEST_PATH = ROOT / "manifest" / "nhanes_sha256.json"

# The default points at the Docker Compose service in docker-compose.yml. These are throwaway
# local credentials for a database that only listens on 127.0.0.1, not secrets.
DEFAULT_DB_URL = "postgresql+psycopg://nhanes:nhanes_local@127.0.0.1:5433/nhanes"
CDC_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"
SEED = 42


def db_url() -> str:
    return os.environ.get("NHANES_DB_URL", DEFAULT_DB_URL)


def fixture_db_url() -> str:
    """Database used by fixture runs, so a test run never overwrites the real data.

    NHANES_FIXTURE_DB_URL wins if set (CI uses its own service database). Otherwise the same
    server is used with the database name `nhanes_fixture`.
    """
    explicit = os.environ.get("NHANES_FIXTURE_DB_URL")
    if explicit:
        return explicit
    base = db_url()
    head, _, _ = base.rpartition("/")
    return f"{head}/nhanes_fixture"


@dataclass(frozen=True)
class Outputs:
    """Where a run writes its results. Real runs write into the repo (docs/, models/);
    fixture runs write into a throwaway directory so the committed results are never touched."""

    base: Path
    artifacts: Path

    @property
    def docs(self) -> Path:
        return self.base / "docs"

    @property
    def fig_dir(self) -> Path:
        return self.docs / "figures"

    @property
    def results_json(self) -> Path:
        return self.docs / "results" / "results.json"

    @property
    def results_md(self) -> Path:
        return self.docs / "RESULTS.md"

    @property
    def model_dir(self) -> Path:
        return self.base / "models"


def default_outputs() -> Outputs:
    """Real run, or whatever NHANES_OUT / NHANES_ARTIFACTS point at (notebooks use this)."""
    base = Path(os.environ.get("NHANES_OUT", ROOT))
    default_art = DATA_DIR / "artifacts" if base == ROOT else base / "artifacts"
    return Outputs(base, Path(os.environ.get("NHANES_ARTIFACTS", default_art)))


@dataclass(frozen=True)
class Cycle:
    letter: str  # file suffix used by the CDC, e.g. G for DEMO_G.xpt
    years: str  # human label
    start_year: int  # directory the CDC files live in
    split: str  # train / valid / test (temporal)
    weight_col: str  # exam weight column name in this cycle's DEMO file


# Four consecutive 2-year cycles that all use the same blood pressure method (auscultatory BPX).
# The 2017 to March 2020 "P_" files are NOT used: they replace BPX with an oscillometric device
# (BPXO), and they contain the 2017 to 2018 cycle again, so they overlap with cycle J.
CYCLES: tuple[Cycle, ...] = (
    Cycle("G", "2011-2012", 2011, "train", "WTMEC2YR"),
    Cycle("H", "2013-2014", 2013, "train", "WTMEC2YR"),
    Cycle("I", "2015-2016", 2015, "valid", "WTMEC2YR"),
    Cycle("J", "2017-2018", 2017, "test", "WTMEC2YR"),
)

# CDC table name -> raw table name in PostgreSQL (lower case, schema raw)
TABLES: tuple[str, ...] = (
    "DEMO",  # demographics and survey weights
    "BMX",  # body measures
    "BPX",  # blood pressure (auscultatory)
    "TCHOL",  # total cholesterol (lab)
    "HDL",  # HDL cholesterol (lab)
    "GHB",  # glycohemoglobin / HbA1c (lab)
    "GLU",  # fasting plasma glucose (lab, morning fasting subsample)
    "SMQ",  # smoking questionnaire
    "DIQ",  # diabetes questionnaire
    "ALQ",  # alcohol questionnaire
    "MCQ",  # medical conditions questionnaire (the outcome lives here)
)


def file_stem(table: str, cycle: Cycle) -> str:
    return f"{table}_{cycle.letter}"


def file_url(table: str, cycle: Cycle) -> str:
    return f"{CDC_BASE}/{cycle.start_year}/DataFiles/{file_stem(table, cycle)}.xpt"
