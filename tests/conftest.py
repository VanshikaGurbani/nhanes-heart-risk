import os

import pytest
from sqlalchemy.engine import make_url

from nhanes.config import fixture_db_url
from nhanes.db import ensure_database, get_engine


def _test_url() -> str:
    """A database that is never the real one. CI sets NHANES_FIXTURE_DB_URL to its service DB."""
    if os.environ.get("NHANES_FIXTURE_DB_URL"):
        return os.environ["NHANES_FIXTURE_DB_URL"]
    return make_url(fixture_db_url()).set(database="nhanes_test").render_as_string(False)


@pytest.fixture(scope="session")
def engine():
    url = _test_url()
    try:
        ensure_database(url)
        eng = get_engine(url)
        with eng.connect():
            pass
    except Exception as e:  # no PostgreSQL available
        pytest.skip(f"PostgreSQL not reachable ({type(e).__name__}); run `docker compose up -d`")
    return eng
