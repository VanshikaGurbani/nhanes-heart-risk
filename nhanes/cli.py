"""Command line: `nhanes <command>`. `nhanes run-all` does everything."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nhanes.config import MANIFEST_PATH, RAW_DIR, db_url


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="nhanes", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="download the NHANES .xpt files and verify checksums")
    sub.add_parser("load", help="load raw files into PostgreSQL (schema raw)")
    sub.add_parser("transform", help="run the SQL files (schemas stg and model)")
    sub.add_parser("validate", help="run data quality checks; non-zero exit on failure")
    ra = sub.add_parser("run-all", help="fetch, load, transform, validate, train, report")
    ra.add_argument("--fixture", action="store_true", help="use a small synthetic dataset (CI)")
    ra.add_argument("--out", type=Path, help="output directory (required with --fixture)")
    ra.add_argument("--n-boot", type=int, default=1000)
    ra.add_argument("--quick", action="store_true", help="fewer bootstrap and permutation repeats")
    ra.add_argument("--skip-fetch", action="store_true", help="assume data/raw is already complete")
    sub.add_parser("refresh-readme", help="rewrite the generated README blocks from results.json")
    nb = sub.add_parser("run-notebooks", help="execute the notebooks in place (saves outputs)")
    nb.add_argument("--fixture", action="store_true")
    nb.add_argument("--out", type=Path, help="fixture output directory from run-all --fixture")
    args = ap.parse_args(argv)

    if args.cmd == "fetch":
        from nhanes.extract import fetch_all

        m = fetch_all(RAW_DIR, MANIFEST_PATH)
        print(f"{len(m)} files verified against {MANIFEST_PATH}")
        return 0
    if args.cmd in ("load", "transform", "validate"):
        from nhanes.db import get_engine, wait_for_db

        engine = get_engine(db_url())
        wait_for_db(engine)
        if args.cmd == "load":
            from nhanes.load import load_raw

            print(load_raw(engine))
        elif args.cmd == "transform":
            from nhanes.load import apply_transforms

            print(apply_transforms(engine))
        else:
            from nhanes.validate import ValidationError, assert_valid

            try:
                assert_valid(engine)
            except ValidationError as e:
                print(f"VALIDATION FAILED: {e}", file=sys.stderr)
                return 2
        return 0
    if args.cmd == "run-all":
        from nhanes.pipeline import run_all
        from nhanes.validate import ValidationError

        try:
            run_all(
                fixture=args.fixture,
                out_dir=args.out,
                n_boot=args.n_boot,
                quick=args.quick,
                skip_fetch=args.skip_fetch,
            )
        except ValidationError as e:
            print(f"VALIDATION FAILED: {e}", file=sys.stderr)
            return 2
        return 0
    if args.cmd == "refresh-readme":
        import json

        from nhanes.config import ROOT, default_outputs
        from nhanes.report import update_readme_headline

        R = json.loads(default_outputs().results_json.read_text(encoding="utf-8"))
        print(
            "README updated"
            if update_readme_headline(R, ROOT / "README.md")
            else "README unchanged"
        )
        return 0
    if args.cmd == "run-notebooks":
        from nhanes.nb_run import run_notebooks

        return run_notebooks(fixture=args.fixture, out=args.out)
    return 1


if __name__ == "__main__":
    sys.exit(main())
