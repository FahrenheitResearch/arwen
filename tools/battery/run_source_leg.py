"""Run source-dependent release gates, refusing missing inputs and any skip.

Run only inside an authorized GPU queue. This script does not reserve a card.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
FILES = [
    "tests/test_cldfra2_gpu.py",
    "tests/test_cldfra2_coupled_gpu.py",
    "tests/test_ysu_generation_source.py",
    "tests/test_ysu_generation_batch_gpu.py",
    "tests/test_g3_halo.py",
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrf461", required=True, type=Path)
    parser.add_argument("--wrf480-source", required=True, type=Path)
    parser.add_argument("--ysu-fixture", required=True, type=Path)
    parser.add_argument("--coupled-fixture", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    required = [args.wrf461 / "phys/module_radiation_driver.F", args.wrf480_source,
                args.ysu_fixture, args.coupled_fixture]
    for path in required:
        if not path.exists():
            parser.error(f"source leg requires {path}")
    args.out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, WRF_SOURCE_ROOT=str(args.wrf461.resolve()),
               YSU_WRF480_SOURCE=str(args.wrf480_source.resolve()),
               YSU_COLUMN_FIXTURE=str(args.ysu_fixture.resolve()),
               GPUWM_CLDFRA2_COUPLED_ROOT=str(args.coupled_fixture.resolve()),
               PYTEST_ADDOPTS="")
    records = []
    for exact in ("0", "1"):
        report = args.out.resolve() / f"source-exact-{exact}.xml"
        # An interrupted invocation cannot borrow an earlier passing report.
        report.unlink(missing_ok=True)
        command = [sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider",
                   "-o", "addopts=", "-o", "junit_family=xunit1",
                   f"--junitxml={report}", *FILES]
        with (args.out / f"source-exact-{exact}.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, env=dict(env, GPUWM_WRF_EXACT=exact),
                                    stdout=log, stderr=subprocess.STDOUT)
        report_error = None
        try:
            cases = list(ET.parse(report).iter("testcase"))
        except (OSError, ET.ParseError) as error:
            cases = []
            report_error = str(error)
        skipped = sum(case.find("skipped") is not None for case in cases)
        failed = sum(case.find("failure") is not None or case.find("error") is not None
                     for case in cases)
        covered = {case.get("file", "").replace("\\", "/") for case in cases}
        missing = sorted(set(FILES) - covered)
        record = {"command": command, "exact": exact, "returncode": result.returncode,
                  "tests": len(cases), "skipped": skipped, "failed": failed,
                  "missing_files": missing, "report_error": report_error,
                  "passed": result.returncode == 0 and bool(cases) and skipped == 0
                  and failed == 0 and not missing and report_error is None}
        records.append(record)
    (args.out / "source-leg.json").write_text(json.dumps({
        "sources": {key: env[key] for key in ("WRF_SOURCE_ROOT", "YSU_WRF480_SOURCE",
                     "YSU_COLUMN_FIXTURE", "GPUWM_CLDFRA2_COUPLED_ROOT")},
        "runs": records}, indent=2) + "\n")
    return 0 if all(row["passed"] for row in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
