#!/usr/bin/env python3
"""Replace the RRTM longwave's "no oracle" warning with what the oracle measured.

The registry is generated -- ``tools/build_registry.py`` must still reproduce
it byte for byte -- so the edit goes through the builder's own
``canonical_json``.  Only the first warning of
``components.radiation.options.wrf-rrtm-dudhia`` changes; the maturity stays
``implemented-unverified``, because that rung is about a matched forecast
trajectory and because the row's Dudhia shortwave half has no oracle here.

    python tools/rrtm_lw_wrf461_oracle/patch_registry.py IN.json OUT.json

Idempotent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT))

from build_registry import canonical_json  # noqa: E402

OLD_PREFIX = "WRF RRTM LONGWAVE IS IMPLEMENTED-UNVERIFIED."

RRTM_LW_WARNING = (
    "WRF RRTM LONGWAVE IS BITWISE AGAINST WRF ON ITS COLUMN ORACLE. "
    "gpuwm/core/rrtm_lw.py, gpuwm/core/rrtm_taumol.py and "
    "gpuwm/core/rrtm_tables.py transcribe phys/module_ra_rrtm.F (O3DATA, "
    "MM5ATM, SETCOEF, CMBGB1-16, TAUGB1-16, GASABS, RTRN, RRTM/RRTMLWRAD) "
    "and the coefficient tables are the packaged RRTM_DATA records reduced "
    "256 g-points to 140 by the module's own CMBGB weighting. "
    "tools/rrtm_lw_wrf461_oracle compiles the byte-unmodified module at the "
    "pinned WRF v4.6.1 commit (gfortran 13.3.0, glibc 2.39, -O0) and drives "
    "RRTMLWRAD on 79 columns in 7 calls: deep convection, stable nights, "
    "snow and ice, ocean, 580-705 hPa terrain and edge cases (a 1085 hPa "
    "surface, 340 K and 342 K surfaces, SETCOEF's pressure switches hit "
    "exactly, negative and subnormal mixing ratios, icloud=0, Kessler's "
    "flags), with 25, 13, 3, 1 and 0 Cavallo buffer layers. On that fixture "
    "every RTHRATEN, RTHRATENC, GLW and OLR word, every MM5ATM/SETCOEF/"
    "GASABS/RTRN intermediate and every rrtm_lookuptable table equals WRF's "
    "(0 ULP) on the NumPy path and on the GPU under GPUWM_WRF_EXACT=1; under "
    "default GPU arithmetic the outputs are also 0 ULP and only TAUCLOUD on "
    "10 subnormal-mixing-ratio layers differs (flushed to zero). "
    "tests/test_rrtm_lw_wrf461_parity.py is the gate. Reaching it took "
    "WRF's float32 BPADE and FLUXFAC, WOOF's own float32 libm for EXP, LOG "
    "and **, the Fortran accumulation order for O3DATA's and RTRN's sums, "
    "P/100 as a division, the REAL(8) trace gases computed in float32, one "
    "association in eight TAUGB bands, and the 340 K Planck row (an OLR "
    "error of 2.3 W m-2 on a 340 K surface before). WRF's own -O2 "
    "-ftree-vectorize build moves 5 TAUCLOUD words by 1 ULP through "
    "libmvec's vector powf and no output. NOT COVERED: cal_cldfra1 and the "
    "MYNN cloud merge that produce CLDFRA upstream of RRTMLWRAD (radiation "
    "driver code, graded with the driver), level or layer temperatures "
    "outside the 160-340 K Planck table (WRF reads out of bounds there and "
    "the port holds the end row), and any matched WRF forecast trajectory.")


def patch(registry: dict) -> dict:
    row = registry["components"]["radiation"]["options"]["wrf-rrtm-dudhia"]
    warnings = row["warnings"]
    for index, text in enumerate(warnings):
        if text.startswith(OLD_PREFIX) or text == RRTM_LW_WARNING:
            warnings[index] = RRTM_LW_WARNING
            return registry
    raise SystemExit("wrf-rrtm-dudhia carries neither the old nor the new "
                     "RRTM longwave warning")


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    registry = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    text = canonical_json(patch(registry))
    Path(argv[2]).write_bytes((text + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
