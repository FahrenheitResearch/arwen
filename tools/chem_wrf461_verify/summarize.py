"""Fold one run_parity.sh output folder into per-test-file ULP maxima.

    python tools/chem_wrf461_verify/summarize.py OUT_DIR/<mode> > summary.json

Reads ulp.jsonl and junit.xml; prints, per test file, the number of oracle
comparisons, compared values, values that differ, the largest ULP distance and
the pytest outcome counts.
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


def main(folder: str) -> dict:
    folder = Path(folder)
    files: dict[str, dict] = defaultdict(lambda: {
        "comparisons": 0, "values": 0, "values_differing": 0, "max_ulp": 0,
        "passed": 0, "failed": 0, "skipped": 0, "errors": 0})
    log = folder / "ulp.jsonl"
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = (row["test"] or "?").split("::")[0]
            entry = files[key]
            entry["comparisons"] += 1
            entry["values"] += row["n"]
            entry["values_differing"] += row["n_nonzero"]
            entry["max_ulp"] = max(entry["max_ulp"], row["max_ulp"])
    junit = folder / "junit.xml"
    if junit.exists():
        for case in ET.parse(junit).getroot().iter("testcase"):
            key = case.get("file") or (case.get("classname", "").replace(".", "/") + ".py")
            entry = files[key]
            if case.find("skipped") is not None:
                entry["skipped"] += 1
            elif case.find("failure") is not None:
                entry["failed"] += 1
            elif case.find("error") is not None:
                entry["errors"] += 1
            else:
                entry["passed"] += 1
    total = {"comparisons": sum(v["comparisons"] for v in files.values()),
             "values": sum(v["values"] for v in files.values()),
             "values_differing": sum(v["values_differing"] for v in files.values()),
             "max_ulp": max((v["max_ulp"] for v in files.values()), default=0),
             "passed": sum(v["passed"] for v in files.values()),
             "failed": sum(v["failed"] for v in files.values()),
             "skipped": sum(v["skipped"] for v in files.values()),
             "errors": sum(v["errors"] for v in files.values())}
    return {"total": total, "files": dict(sorted(files.items()))}


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1]), indent=2))
