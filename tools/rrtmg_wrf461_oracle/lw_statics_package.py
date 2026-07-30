"""Package the 13 LW compile-time module-DATA tables as a data asset.

WRF v4.6.1's module_ra_rrtmg_lw carries lwavplank (totplnk/totplk16),
lwatmref (preflog/tref/chi_mls), lwcldpr (absice0/absice1/absice2/
absice3/absliq0/absliq1) and the rrlw_wvn delwave/ngb rosters as
COMPILE-TIME DATA statements -- they are part of the algorithm, not of
the RRTMG_LW_DATA coefficient file.  This script reads them from the
oracle module dump (lw_coeffs.bin, the compiled UNMODIFIED Fortran's
own module state, the same authority every LW gate uses) and writes

    gpuwm/data/wrf_radiation/rrtmg_lw_statics.npz

one C-ordered little-endian member per table, member order exactly the
roster below (= gpuwm.core.rrtmg_legacy._LW_STATIC_SPECS), then prints
the file's SHA-256 for the pin in gpuwm/core/rrtmg_legacy.py
(RRTMG_LW_STATICS_SHA256).

Usage:
    python tools/rrtmg_wrf461_oracle/lw_statics_package.py [fixdir]

``fixdir`` defaults to the GPUWM_RRTMG_LW_FIXTURES location used by
every other LW oracle gate (see lw_gate.DEFAULT_FIXDIR).
"""

import hashlib
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from lw_fixtures import read_fixture            # noqa: E402
from lw_gate import DEFAULT_FIXDIR              # noqa: E402

#: (packaged key, oracle dump key) roster, in packaged member order.
NAMES = (
    ("wvn/totplnk", "rrlw_wvn/totplnk"),
    ("wvn/totplk16", "rrlw_wvn/totplk16"),
    ("wvn/delwave", "rrlw_wvn/delwave"),
    ("wvn/ngb", "rrlw_wvn/ngb"),
    ("ref/preflog", "rrlw_ref/preflog"),
    ("ref/tref", "rrlw_ref/tref"),
    ("ref/chi_mls", "rrlw_ref/chi_mls"),
    ("cld/absice0", "rrlw_cld/absice0"),
    ("cld/absice1", "rrlw_cld/absice1"),
    ("cld/absice2", "rrlw_cld/absice2"),
    ("cld/absice3", "rrlw_cld/absice3"),
    ("cld/absliq0", "rrlw_cld/absliq0"),
    ("cld/absliq1", "rrlw_cld/absliq1"),
)

#: Total scalar count across the 13 tables (fail-closed completeness).
TOTAL_VALUES = 6129


def main(argv):
    fixdir = argv[1] if len(argv) > 1 else DEFAULT_FIXDIR
    cfx = read_fixture(os.path.join(fixdir, "lw_coeffs.bin"))
    tables = {}
    total = 0
    for key, src in NAMES:
        arr = np.asarray(cfx[src])
        assert arr.dtype in (np.dtype("<f4"), np.dtype("<i4")), \
            (key, arr.dtype)
        tables[key] = np.ascontiguousarray(arr) if arr.ndim else arr
        total += int(arr.size)
    assert total == TOTAL_VALUES, total

    out_path = os.path.join(_REPO, "gpuwm", "data", "wrf_radiation",
                            "rrtmg_lw_statics.npz")
    with open(out_path, "wb") as stream:
        np.savez(stream, **tables)
    with open(out_path, "rb") as stream:
        digest = hashlib.sha256(stream.read()).hexdigest()

    for key, _src in NAMES:
        arr = tables[key]
        print(f"  {key}: shape={tuple(arr.shape)} dtype={arr.dtype.name}")
    print(f"{total} values -> {out_path}")
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
