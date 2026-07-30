#!/usr/bin/env python3
"""Measure gpuwm's YSU CUDA kernel against the unmodified WRF v4.6.1 module.

Run it on the CSVs build.sh produced, or with no argument on the packaged
fixture under ``gpuwm/data/ysu/oracle``::

    python3 tools/ysu_wrf461_oracle/validate_ysu_oracle.py [FIXTURE_DIR]

It prints one line per output field and exits non-zero only if a field is
*worse* than the recorded baseline in ``tests/test_ysu_wrf461_parity.py``; it
does not decide whether the baseline is acceptable.  The point of this script
is the number, not a verdict.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpuwm.verify.ysu_oracle import (  # noqa: E402
    kpbl_disagreements,
    load_ysu_oracle,
    measure_ysu_parity,
    ysu_port_outputs,
)


def main(argv: list[str]) -> int:
    directory = Path(argv[1]) if len(argv) > 1 else None
    fixture = load_ysu_oracle(directory)
    print(f"fixture: {fixture.ncase} columns x {fixture.nz} levels,"
          f" dt={fixture.dt}, cases={list(fixture.cases)}")
    port = ysu_port_outputs(fixture)
    parity = measure_ysu_parity(fixture, port)
    for name in sorted(parity):
        print(parity[name])
    bad = kpbl_disagreements(fixture, port)
    if bad:
        print("kpbl disagreements (case: got, want):", bad)
    else:
        print("kpbl: every case agrees with WRF")
    worst = max(p.max_ulp for p in parity.values())
    print(f"worst max_ulp over all fields: {worst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
