#!/usr/bin/env python3
"""Show which gwd_opt = 1 branches the column oracle reaches, on the GPU.

    python branch_mutations.py [--json OUT]

For every control in ``gpuwm.verify.gwdo_kim_oracle.BRANCH_MUTATIONS`` it
compiles the production terrain-drag unit with that one gwdo_column
statement changed and counts the WRF words each grid's oracle then misses.
A control the oracle catches is a branch the fixture's columns reach; one it
does not catch is reported as such, not hidden.  Run under the GPU mutex.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpuwm.verify import gwdo_kim_oracle as go  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    base = {grid: sum(v["not_bit_identical"]
                      for v in go.measure(grid).values())
            for grid in go.GRIDS}
    report = {"production": base, "controls": {}}
    # The kernel as it was before lane/verify-gwdo-kim: the double atan2
    # rounded once, measured per field on the same fixture.
    from gpuwm.core import terrain_drag
    source = terrain_drag.module_source()
    head = "__device__ float td_atan2f(float y, float x)\n{\n"
    before = source.replace(head, head + "    return __double2float_rn("
                            "atan2((double)y, (double)x));\n")
    with go.kernel_source(before):
        report["before_fix_double_atan2"] = {
            grid: {k: {f: v[f] for f in ("words", "not_bit_identical",
                                         "max_ulp", "max_abs_diff",
                                         "columns_missed_by_regime")}
                   for k, v in go.measure(grid).items()}
            for grid in go.GRIDS}
    for name, _, _, reaches in go.BRANCH_MUTATIONS:
        missed = go.measure_mutation(name)
        report["controls"][name] = {"reaches": reaches, "missed": missed,
                                    "caught": any(missed.values())}
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if not any(base.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
