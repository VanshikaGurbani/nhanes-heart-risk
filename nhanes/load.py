"""Stage 2: load the raw files into PostgreSQL (schema raw), one table per NHANES table.

Cycles are stacked into one table and tagged with a `cycle` column (the CDC file suffix).
Nothing is cleaned here. Re-running replaces the raw tables inside one transaction, so a failed
load leaves the previous state intact and a successful one is identical every time.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, text

from nhanes.config import CYCLES, RAW_DIR, TABLES, file_stem
from nhanes.db import run_sql_files, sql_files
from nhanes.extract import sha256_of


def fix_xport_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """Decode the way SAS transport files store an exact zero.

    pandas reads a stored zero as 5.397605346934028e-79 (the smallest IBM float). Left alone, that
    value passes a "greater than 0" test, so a diastolic reading of 0 or an age of 0 would look like
    a tiny positive number. Anything with 0 < |x| < 1e-30 is set to exactly 0. No real NHANES value
    is that small, and nothing else in the file is touched.
    """
    out = df.copy()
    for c in out.select_dtypes("number").columns:
        tiny = (out[c].abs() < 1e-30) & (out[c] != 0)
        if tiny.any():
            out.loc[tiny, c] = 0.0
    return out


def read_raw_file(table: str, letter: str, raw_dir: Path) -> pd.DataFrame:
    """Read one CDC file. Real data is .xpt (SAS transport); test fixtures are .csv."""
    stem = f"{table}_{letter}"
    xpt = raw_dir / f"{stem}.xpt"
    csv = raw_dir / f"{stem}.csv"
    if xpt.exists():
        df = pd.read_sas(xpt, format="xport", encoding="utf-8")
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        raise FileNotFoundError(f"{stem}.xpt not found in {raw_dir}. Run `nhanes fetch` first.")
    df.columns = [c.lower() for c in df.columns]
    df = fix_xport_zeros(df)
    df["seqn"] = df["seqn"].astype("int64")
    df["cycle"] = letter
    return df


def _copy_frame(conn, df: pd.DataFrame, name: str) -> None:
    """Create raw.<name> from the frame's dtypes, then bulk load with COPY (fast, exact)."""
    df.head(0).to_sql(name, conn, schema="raw", if_exists="replace", index=False)
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, na_rep="")
    buf.seek(0)
    cols = ", ".join(f'"{c}"' for c in df.columns)
    cur = conn.connection.cursor()
    with cur.copy(f"COPY raw.{name} ({cols}) FROM STDIN WITH (FORMAT csv, NULL '')") as copy:
        while chunk := buf.read(1 << 20):
            copy.write(chunk)


def load_raw(engine: Engine, raw_dir: Path = RAW_DIR) -> dict[str, int]:
    """(Re)create schemas, reference tables and raw tables. Returns rows loaded per table."""
    counts: dict[str, int] = {}
    log_rows = []
    frames: dict[str, pd.DataFrame] = {}
    for table in TABLES:
        parts = []
        for cycle in CYCLES:
            df = read_raw_file(table, cycle.letter, raw_dir)
            path = next(
                p
                for p in (
                    raw_dir / f"{file_stem(table, cycle)}.xpt",
                    raw_dir / f"{file_stem(table, cycle)}.csv",
                )
                if p.exists()
            )
            log_rows.append(
                {
                    "table_name": table.lower(),
                    "cycle": cycle.letter,
                    "file_name": path.name,
                    "sha256": sha256_of(path),
                    "n_rows": len(df),
                }
            )
            parts.append(df)
        frames[table] = pd.concat(parts, ignore_index=True)

    ref = pd.DataFrame(
        [
            {"letter": c.letter, "years": c.years, "split": c.split, "weight_col": c.weight_col}
            for c in CYCLES
        ]
    )
    with engine.begin() as conn:
        # Downstream layers depend on the raw tables, so they are dropped here and rebuilt by
        # apply_transforms(). This is what makes the whole pipeline safe to re-run.
        conn.exec_driver_sql(
            "DROP SCHEMA IF EXISTS model CASCADE; DROP SCHEMA IF EXISTS stg CASCADE;"
        )
        conn.exec_driver_sql(sql_files(0, 0)[0].read_text(encoding="utf-8"))  # 00_schemas.sql
        for table, df in frames.items():
            _copy_frame(conn, df, table.lower())
            conn.execute(text(f'CREATE INDEX ON raw.{table.lower()} (seqn, "cycle")'))
            counts[table.lower()] = len(df)
        ref.to_sql("cycle", conn, schema="ref", if_exists="replace", index=False)
        pd.DataFrame(log_rows).to_sql(
            "load_log", conn, schema="raw", if_exists="replace", index=False
        )
    return counts


def apply_transforms(engine: Engine) -> list[str]:
    """Stage 3: run the numbered SQL files (stg views, then the model table)."""
    return run_sql_files(engine, sql_files(10, 99))
