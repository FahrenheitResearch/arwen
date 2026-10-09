"""Grade kernels/sfclay.cu (option 1) against the WRF oracle words, bitwise.

    python tools/sfclayrev_wrf461_oracle/compare.py [--outputs PATH] [--json PATH]
                                                     [--lanes N]

Runs the real kernel through gpuwm.core.sfclay on the fixture's columns and
prints, per output field, the worst float32 ULP distance and worst absolute
difference over every arm, step and column.  Set GPUWM_WRF_EXACT=1 for the
strict build; leave it unset for default arithmetic.  Exit status 0 means
every compared word is bitwise WRF's.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

from _sfclayrev_oracle import ARMS, ORACLE_DIR, load_fixture, measure, port_outputs  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", type=Path, default=ORACLE_DIR / "sfclayrev-outputs.hex")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--lanes", type=int, default=40)
    args = ap.parse_args(argv)
    # An absolute name replaces ORACLE_DIR in load_fixture's join, so any
    # --outputs file is read, including one that sits beside the reference
    # words (the as-built receipt does).
    fixture = load_fixture(ORACLE_DIR, outputs_name=str(args.outputs.resolve()))
    port = port_outputs(fixture)
    report = measure(fixture, port)
    mode = "strict (GPUWM_WRF_EXACT=1)" if os.environ.get("GPUWM_WRF_EXACT") == "1" else "default"
    print(f"mode: {mode}; columns {len(fixture.cases)}; arms {ARMS}; steps 3")
    print(f"{'field':8s} {'max_ulp':>12s} {'max_abs':>12s} {'differ':>7s} {'lanes':>6s}")
    for name, v in report["fields"].items():
        print(f"{name:8s} {v['max_ulp']:12d} {v['max_abs']:12.4g} {v['lanes_differ']:7d} {v['lanes']:6d}")
    print(f"MAX ULP over every field: {report['max_ulp']}")
    for lane in sorted(report["lanes"], key=lambda x: -x[5])[:args.lanes]:
        name, a, s, case, label, ulp, w, p = lane
        print(f"  {name:7s} arm{a + 1} {ARMS[a]} step{s + 1} case{case:4d} {label:28s} "
              f"ulp={ulp} wrf={w!r} port={p!r}")
    if args.json:
        args.json.write_text(json.dumps({"mode": mode, **report}, indent=1, default=str))
    return 0 if report["max_ulp"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
