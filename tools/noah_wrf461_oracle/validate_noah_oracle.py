#!/usr/bin/env python3
"""Measure gpuwm's Noah CUDA kernel against the unmodified WRF v4.6.1 driver.

Run it on the CSVs build.sh produced, or with no argument on the packaged
fixture under ``gpuwm/data/noah/oracle``::

    python3 tools/noah_wrf461_oracle/validate_noah_oracle.py [FIXTURE_DIR]

It prints one line per output field per fixture and asserts nothing.  The
point of this script is the number, not a verdict;
``tests/test_noah_wrf461_parity.py`` is the gate.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpuwm.verify.noah_oracle import (  # noqa: E402
    NOAH_ORACLE_FILES,
    glacial_divergence,
    load_noah_oracle,
    measure_noah_parity,
    noah_port_outputs,
)


def main(argv: list[str]) -> int:
    directory = Path(argv[1]) if len(argv) > 1 else None
    overall = 0
    for name in NOAH_ORACLE_FILES:
        fixture = load_noah_oracle(name, directory)
        port = noah_port_outputs(fixture)
        parity = measure_noah_parity(fixture, port)
        glacial = int(fixture.glacial.sum())
        print(f"=== {name}  {fixture.switches}  "
              f"{fixture.ncase} columns ({glacial} glacial, excluded) ===")
        for field in sorted(parity, key=lambda f: -parity[f].max_ulp):
            print("   ", parity[field])
        worst = max(p.max_ulp for p in parity.values())
        overall = max(overall, worst)
        print(f"    worst max_ulp: {worst}")
        gaps = glacial_divergence(fixture, port)
        if gaps:
            worst_gap = sorted(gaps.items(), key=lambda kv: -kv[1])[:5]
            print("    SFLX_GLACIAL columns gpuwm does not compute, "
                  "absolute gap:",
                  ", ".join(f"{k}={v:.6g}" for k, v in worst_gap))
    print(f"worst max_ulp over every fixture and field: {overall}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
