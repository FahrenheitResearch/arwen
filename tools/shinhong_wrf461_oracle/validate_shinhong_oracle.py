#!/usr/bin/env python3
"""Measure gpuwm's Shin-Hong float32 authority against unmodified WRF v4.6.1.

Run it on the CSVs build.sh produced, or with no argument on the packaged
fixture under ``gpuwm/data/shinhong/oracle``::

    python tools/shinhong_wrf461_oracle/validate_shinhong_oracle.py [FIXTURE_DIR]

It prints one line per output field for both oracle arms -- max ULP over the
non-NaN lanes, the NaN-lane counts and whether the NaN geometry matches --
plus the five partition functions against their direct probe.  It asserts
nothing and always exits 0: the point of this script is the number, and
``tests/test_shinhong_wrf461_parity.py`` is the gate that pins it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpuwm.core.fp32_ulp import fp32_ulp_distance  # noqa: E402
from gpuwm.verify.shinhong_ref import (  # noqa: E402
    np_pq,
    np_pthl,
    np_pthnl,
    np_ptke,
    np_pu,
)
from gpuwm.verify.shinhong_oracle import (  # noqa: E402
    ARM_A,
    ARM_B,
    kpbl_disagreements,
    load_shinhong_oracle,
    measure_shinhong_parity,
    shinhong_ref_outputs,
)

_PARTITION = {"pu": np_pu, "pq": np_pq, "pthnl": np_pthnl,
              "pthl": np_pthl, "ptke": np_ptke}


def main(argv: list[str]) -> int:
    directory = Path(argv[1]) if len(argv) > 1 else None
    fixture = load_shinhong_oracle(directory)
    print(f"fixture: {fixture.ncol} columns ({len(fixture.case_ids)} cases"
          f" x 6 dx) x {fixture.nz} levels,"
          f" partition probe rows: {fixture.partition['d'].size}")

    print("\npartition functions vs the direct probe:")
    d, h = fixture.partition["d"], fixture.partition["h"]
    for name, fn in _PARTITION.items():
        got = np.asarray([fn(d[i], h[i]) for i in range(d.size)], np.float32)
        distance = fp32_ulp_distance(got, fixture.partition[name])
        print(f"  {name:<6s} max_ulp={int(distance.max())}"
              f" differing={int(np.count_nonzero(distance))}/{d.size}")

    for label, arm in (("arm A (ctopo=ctopo2=1)", ARM_A),
                       ("arm B (ctopo=0.85, ctopo2=0.7)", ARM_B)):
        print(f"\n{label}:")
        port = shinhong_ref_outputs(fixture, arm=arm)
        parity = measure_shinhong_parity(fixture, port, arm=arm)
        for name in sorted(parity):
            print(f"  {parity[name]}")
        bad = kpbl_disagreements(fixture, port, arm=arm)
        if bad:
            print(f"  kpbl disagreements ((case, dxi): got, want): {bad}")
        else:
            print("  kpbl: every column agrees with WRF")
        worst = max(p.max_ulp for p in parity.values())
        print(f"  worst max_ulp over all fields: {worst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
