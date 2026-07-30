#!/usr/bin/env python3
"""Compare the CPU phim/phih transcription with the official WRF CSV.

Both arms are bitwise.  The unstable arm was not while the port's ``atanf``
rounded an FP64 evaluation: glibc's ``atanf`` is faithfully rounded rather
than correctly rounded, so that shim was a third function, and the
``(1 - phi_m)/zet`` cancellation amplified the disagreement to 80 ULP on 22
of 814 ``phim`` rows and 84 ULP on 9 ``phih`` rows.  Pointing ``_atanf`` at
the verified glibc 2.39 transcription in ``gpuwm/core/noahmp_libm.py`` drives
all four counts to zero.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

from gpuwm.core.fp32_ulp import monotone_fp32_key
from gpuwm.core.mynn_pbl import mynn_phih, mynn_phim

EXPECTED_ROWS = 814

#: Measured over the whole fixture on the verified glibc 2.39 ``atanf``.
#: These were 80/84/22/9 with the FP64-then-round shim; they are a ratchet,
#: so a regression in either arm now trips instead of hiding under a budget
#: sized for a shim the port no longer uses.
STABLE_ULP = 0
UNSTABLE_PHIM_ULP = 0
UNSTABLE_PHIH_ULP = 0
UNSTABLE_PHIM_MISSES = 0
UNSTABLE_PHIH_MISSES = 0


def _ulp(got: np.float32, want: np.float32) -> int:
    a = monotone_fp32_key(np.asarray([got], dtype=np.float32))
    b = monotone_fp32_key(np.asarray([want], dtype=np.float32))
    return int(np.abs(a - b)[0])


def main(path: str) -> None:
    with Path(path).open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != EXPECTED_ROWS:
        raise SystemExit(f"expected {EXPECTED_ROWS} rows, found {len(rows)}")
    stable = {"phim": 0, "phih": 0, "rows": 0}
    unstable = {"phim": 0, "phih": 0, "rows": 0}
    misses = {"phim": 0, "phih": 0}
    for row in rows:
        zet = np.float32(row["zet"])
        got = {"phim": mynn_phim(zet), "phih": mynn_phih(zet)}
        bucket = stable if zet >= np.float32(0.0) else unstable
        bucket["rows"] += 1
        for name in ("phim", "phih"):
            distance = _ulp(got[name], np.float32(row[name]))
            bucket[name] = max(bucket[name], distance)
            if distance and bucket is unstable:
                misses[name] += 1
    if stable["phim"] > STABLE_ULP or stable["phih"] > STABLE_ULP:
        raise AssertionError(f"the stable arm is no longer bitwise: {stable}")
    if unstable["phim"] > UNSTABLE_PHIM_ULP:
        raise AssertionError(f"phim unstable arm: {unstable['phim']} ULP")
    if unstable["phih"] > UNSTABLE_PHIH_ULP:
        raise AssertionError(f"phih unstable arm: {unstable['phih']} ULP")
    if misses["phim"] > UNSTABLE_PHIM_MISSES:
        raise AssertionError(f"phim now misses {misses['phim']} rows")
    if misses["phih"] > UNSTABLE_PHIH_MISSES:
        raise AssertionError(f"phih now misses {misses['phih']} rows")
    if stable["rows"] == 0 or unstable["rows"] == 0:
        raise AssertionError("the fixture must cover both arms")
    print(json.dumps({
        "status": "PASS", "rows": len(rows),
        "stable": stable, "unstable": unstable, "unstable_misses": misses,
    }, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_stfunc_oracle.py stfunc.csv")
    main(sys.argv[1])
