"""Re-record tools/battery/list_census.json from a real collection pass.

    python -m tools.battery.record_list_census            # rewrite
    python -m tools.battery.record_list_census --check    # compare only

This is the procedure the census file's ``_how_to_re_record`` field
describes, as a program instead of a sentence: for every CPU leg the
census records, collect the leg's list under the leg's own marker
expression and environment, count what each listed file contributes,
and rewrite the file with the counts, the commit and the date.

THE BREAKAGE THIS PREVENTS
--------------------------
tests/test_battery_list_census.py pins a bijection between the battery
lists and this census, so the re-record runs at every closing tip.  Until
2026-09-01 it was re-done by hand each time from a scratch script that
did not survive the session; one such script was pointed at the wrong
tree (a lane worktree instead of the merge tip) and recorded a census
that was not the tree's.  A recorder that lives in the tree reads the
tree it lives in.

A listed file that collects nothing under the leg's marker expression is
REFUSED here rather than recorded as zero: that is the silent-deselection
class tools/battery/no_silent_deselection.py exists for, and the census
test ``test_no_census_row_records_zero_tests`` says the same thing.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CENSUS_PATH = REPO_ROOT / "tools" / "battery" / "list_census.json"


def _listed(list_path: Path) -> list[str]:
    entries = []
    for raw in list_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            entries.append(line)
    return entries


def _collect(entries: list[str], markexpr: str, environment: dict) -> dict:
    env = dict(os.environ)
    env.update({k: str(v) for k, v in environment.items()})
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
           "--collect-only", "-m", markexpr, *entries]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    counts: dict = OrderedDict((e, 0) for e in entries)
    for line in proc.stdout.splitlines():
        if "::" not in line:
            continue
        file_part = line.split("::", 1)[0].replace("\\", "/")
        if file_part in counts:
            counts[file_part] += 1
    if proc.returncode not in (0, 5) and sum(counts.values()) == 0:
        sys.stderr.write(proc.stdout[-4000:])
        sys.stderr.write(proc.stderr[-4000:])
        raise SystemExit(f"collection failed (rc={proc.returncode})")
    return counts


def _record() -> dict:
    census = json.loads(CENSUS_PATH.read_text(encoding="utf-8"))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                          capture_output=True, text=True).stdout.strip()
    new = OrderedDict()
    for key, value in census.items():
        if key.startswith("_"):
            new[key] = value
    new["_how_to_re_record"] = (
        "python -m tools.battery.record_list_census  (collects every leg's "
        "list under its recorded markexpr and environment with "
        "--collect-only and rewrites this file).  A count that FELL is "
        "coverage that left: say in the commit which tests went and where "
        "they went to.  A count that rose needs no argument.")
    new["recorded_at_commit"] = head
    new["recorded_on"] = _dt.date.today().isoformat()
    legs = OrderedDict()
    for leg, body in census["legs"].items():
        entries = _listed(REPO_ROOT / body["list"])
        counts = _collect(entries, body["markexpr"], body.get("environment", {}))
        zero = [e for e, n in counts.items() if n == 0]
        if zero:
            raise SystemExit(
                f"REFUSED: {leg} lists files that collect nothing under "
                f"{body['markexpr']!r}: {zero} -- fix the list or the marks, "
                "do not record a zero")
        legs[leg] = OrderedDict([
            ("list", body["list"]),
            ("markexpr", body["markexpr"]),
            ("environment", body.get("environment", {})),
            ("collected_total", sum(counts.values())),
            ("files", counts),
        ])
    new["legs"] = legs
    return new


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="collect and compare; exit 1 on drift, write nothing")
    args = parser.parse_args(argv)
    new = _record()
    old = json.loads(CENSUS_PATH.read_text(encoding="utf-8"))
    drift = []
    for leg, body in new["legs"].items():
        before = old.get("legs", {}).get(leg, {}).get("files", {})
        for entry, count in body["files"].items():
            if before.get(entry) != count:
                drift.append(f"{leg}: {entry} {before.get(entry)} -> {count}")
        for entry in before:
            if entry not in body["files"]:
                drift.append(f"{leg}: {entry} {before[entry]} -> (unlisted)")
    for line in drift:
        print(line)
    if args.check:
        print("census: " + ("DRIFT" if drift else "at fixed point"))
        return 1 if drift else 0
    CENSUS_PATH.write_text(json.dumps(new, indent=1) + "\n", encoding="utf-8",
                           newline="\n")
    totals = {leg: body["collected_total"] for leg, body in new["legs"].items()}
    print(f"census: recorded at {new['recorded_at_commit'][:9]} -> {totals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
