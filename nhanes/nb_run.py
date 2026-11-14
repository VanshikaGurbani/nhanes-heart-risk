"""Execute the notebooks top to bottom and save them with their outputs.

Real mode runs against the real database and writes the outputs into notebooks/ in the repo.
Fixture mode (CI) copies the notebooks into the output directory of `run-all --fixture` and runs
them there against the fixture database, so the committed notebooks are never touched.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from nhanes.config import ROOT, fixture_db_url


def run_notebooks(fixture: bool = False, out: Path | None = None) -> int:
    src = ROOT / "notebooks"
    if fixture:
        if out is None:
            raise SystemExit("--fixture needs --out pointing at the run-all --fixture directory")
        dest = Path(out) / "notebooks"
        dest.mkdir(parents=True, exist_ok=True)
        for p in src.glob("*.ipynb"):
            shutil.copy(p, dest / p.name)
        os.environ["NHANES_NOTEBOOK_MODE"] = "fixture"
        os.environ["NHANES_OUT"] = str(Path(out))
        os.environ["NHANES_FIXTURE_DB_URL"] = fixture_db_url()
    else:
        dest = src
        os.environ["NHANES_NOTEBOOK_MODE"] = "real"
    failures = 0
    for path in sorted(dest.glob("*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        client = NotebookClient(
            nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(dest)}}
        )
        print(f"running {path.name} ...", flush=True)
        client.execute()
        bad = _differs(nb)
        if bad:
            failures += 1
            print(f"  {path.name}: {len(bad)} check line(s) say DIFFERS: {bad[:2]}")
        nbformat.write(nb, path)
    return 1 if failures else 0


def _differs(nb) -> list[str]:
    out = []
    for cell in nb.cells:
        for o in cell.get("outputs", []):
            text = o.get("text", "")
            out += [
                line
                for line in text.splitlines()
                if line.startswith("Check:") and "DIFFERS" in line
            ]
    return out
