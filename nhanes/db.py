"""Database helpers: engine creation, waiting for the server, running the SQL files."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import OperationalError

from nhanes.config import SQL_DIR, db_url


def get_engine(url: str | None = None) -> Engine:
    return create_engine(url or db_url(), pool_pre_ping=True, future=True)


def wait_for_db(engine: Engine, timeout_s: int = 60) -> None:
    """Block until the server accepts connections (docker compose may still be starting)."""
    deadline = time.time() + timeout_s
    while True:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return
        except OperationalError:
            if time.time() > deadline:
                raise
            time.sleep(1)


def sql_files(prefix_min: int = 0, prefix_max: int = 99, sql_dir: Path = SQL_DIR) -> list[Path]:
    """SQL files named NN_something.sql, selected by their numeric prefix, in order."""
    out = []
    for p in sorted(sql_dir.glob("*.sql")):
        n = int(p.name.split("_", 1)[0])
        if prefix_min <= n <= prefix_max:
            out.append(p)
    return out


def run_sql_files(engine: Engine, files: list[Path]) -> list[str]:
    """Run each file in a single transaction (all or nothing). Returns the file names run."""
    ran = []
    with engine.begin() as conn:
        for p in files:
            conn.exec_driver_sql(p.read_text(encoding="utf-8"))
            ran.append(p.name)
    return ran


def read_sql(engine: Engine, query: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql_query(text(query), conn)


def ensure_database(url: str) -> None:
    """Create the database named in `url` if it does not exist (used for fixture runs)."""
    from sqlalchemy.engine import make_url

    u = make_url(url)
    admin = create_engine(u.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": u.database}
        ).scalar()
        if not exists:
            conn.execute(text(f'CREATE DATABASE "{u.database}"'))
    admin.dispose()
