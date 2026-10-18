"""Check the real CPU --score-only CLI against all round-3 replay hashes.

No GPU, raw model arrays or regeneration is needed. Source replays stay read-only.
The destination must be a new directory owned by this lane.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

EXPECTED = {"FAIL": 351, "INFO-MISS": 135, "WOOF-ONLY-OK": 60,
            "WOOF-REFUSED": 32, "UNSOUND-REF": 49, "WOOF-CRASH": 8, "WRF-REFUSED": 12}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, required=True)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=Path(__file__).parent /
                        "combo_sweep/round3-outcomes.json")
    args = parser.parse_args()
    baseline = {(r["combo"], r["case"]): r["outcome"]
                for r in json.loads(args.baseline.read_text())}
    args.out.mkdir(parents=True, exist_ok=False)
    for arm in ("woof", "woof-repeat"):
        for path in (args.replays / arm).glob("*/*/hashes.json"):
            target = args.out / path.relative_to(args.replays)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    command = [sys.executable, "-m", "gpuwm.cli", "verify-exact", "--score-only",
               "--recordings", str(args.recordings), "--out", str(args.out),
               "--combos-file", str(Path(__file__).parent / "combo_sweep/combos-v2.json"),
               "--combos", ",".join(sorted({combo for combo, _ in baseline}))]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", GPUWM_NO_LOCAL_GPU="1")
    with (args.out / "rescore.log").open("w") as log:
        run = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    rows = []
    for path in (args.out / "results").glob("*.jsonl"):
        rows.extend(row for row in map(json.loads, path.read_text().splitlines())
                    if "combo" in row)
    actual = {(r["combo"], r["case"]): r["outcome"] for r in rows}
    changes = [{"combo": k[0], "case": k[1], "before": old, "after": actual.get(k)}
               for k, old in sorted(baseline.items()) if actual.get(k) != old]
    counts = dict(Counter(actual.values()))
    ok = (len(rows) == len(actual) == len(baseline) == 647 and counts == EXPECTED
          and actual.keys() == baseline.keys() and len(changes) == 17
          and all(r["before"] == "WOOF-CRASH" and r["after"] == "UNSOUND-REF"
                  for r in changes) and run.returncode == 1)
    receipt = {"passed": ok, "command": command, "exit": run.returncode,
               "rows": len(rows), "counts": counts, "expected": EXPECTED, "changes": changes}
    (args.out / "check.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
