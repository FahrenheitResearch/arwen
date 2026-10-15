"""Print the MYJ kernel's distance from the WRF v4.6.1 column oracle.

    python tools/myjpbl_wrf461_oracle/compare_myjpbl.py [FIXTURE.npz] [--json OUT]

Needs a CUDA GPU.  Run it twice: with GPUWM_WRF_EXACT=1 (the strict build:
no FMA contraction, no FTZ, IEEE division and square root) and without it
(default arithmetic).  Asserts nothing; tests/test_myjpbl_wrf461_parity.py
is the gate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gpuwm.verify import myjpbl_oracle  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("fixture", nargs="?", default=str(myjpbl_oracle.FIXTURE))
    ap.add_argument("--json")
    ap.add_argument("--zero-ht", action="store_true",
                    help="negative control: hand the kernel HT = 0")
    args = ap.parse_args()
    fixture = myjpbl_oracle.load(args.fixture)
    report = myjpbl_oracle.measure(fixture, zero_ht=args.zero_ht)
    summary = myjpbl_oracle.summarize(report)
    mode_name = ("strict (GPUWM_WRF_EXACT=1)"
                 if os.environ.get("GPUWM_WRF_EXACT") == "1" else "default")
    ncols = sum(int(fx["meta"][0]) for fx in fixture.values())
    if args.zero_ht:
        mode_name += " -- CONTROL: HT zeroed"
    print(f"arithmetic: {mode_name}; cases {sorted(fixture)}; "
          f"{ncols} column-calls x 3 steps")
    for mode, worst in summary.items():
        print(f"== {mode}")
        for name, st in worst.items():
            print(f"  {name:10s} max_ulp {st['max_ulp']:>12d}  "
                  f"bit_mismatch {st['bit_mismatch']:>6d}/{st['n']:<7d} "
                  f"max_abs {st['max_abs']:.3e}  beyond_ftz "
                  f"{st['beyond_ftz']}  flushed_subnormal "
                  f"{st['flushed_subnormal']}")
    print("== replay, per case (fields with any bit mismatch)")
    for case, by_step in report["replay"].items():
        for step, fields in by_step.items():
            bad = {n: (s["max_ulp"], s["bit_mismatch"]) for n, s in
                   fields.items() if s["bit_mismatch"]}
            if bad:
                print(f"  {case} call {step}: {bad}")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"arithmetic": mode_name, "summary": summary, "report": {
                m: {c: {str(s): f for s, f in st.items()}
                    for c, st in r.items()} for m, r in report.items()}},
            indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
