"""Assert the gpuwm CPU reference against the DX > 5 km MYNN oracle.

Run under ``PYTHONPATH=<repo root>`` so ``gpuwm.core.mynn_surface`` imports.
Reports max_abs / max_rel / max_ulp per output column and fails closed on a
dead branch or a parity regression.

The gate is the measured FP32 ULP residue, column by column, not a relative
tolerance: the point of an oracle gate is to fail when the port stops
matching WRF, and a tolerance an order of magnitude wider than the real error
cannot do that.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
import sys

import numpy as np

from gpuwm.core.fp32_ulp import fp32_ulp_distance
from gpuwm.core.mynn_surface import mynn_surface_layer_default


INPUT_ALIASES = {
    "hfx": "hfx_input",
    "qfx": "qfx_input",
    "znt": "znt_input",
    "qsfc": "qsfc_input",
    "ust": "ust_input",
}
INPUT_NAMES = (
    "u1", "v1", "t1", "qv1", "p1", "rho1", "dz1",
    "u2", "v2", "dz2", "psfc", "tsk", "pblh", "mavail",
    "hfx", "qfx", "znt", "qsfc", "ust", "xland", "snowh",
)
COLUMN_OUTPUTS = (
    "regime", "zol", "rmol", "ust", "ustm", "mol", "psim", "psih",
    "chs", "chs2", "cqs2", "ch", "flhc", "flqc", "qgh", "qsfc",
    "hfx", "qfx", "lh", "u10", "v10", "th2", "t2", "q2",
    "gz1oz0", "wspd", "br", "ck", "cka", "cd", "cda", "wstar",
    "qstar", "cpm", "znt",
)
EXPECTED_CASES = (
    "strong_stable_land",
    "clipped_stable_land",
    "damped_stable_land",
    "neutral_land",
    "free_convective_land",
    "land_qsfc_unset",
    "thin_land_level2_wind",
    "thin_land_log10_wind",
    "midres_water",
    "coarse_water",
)
EXPECTED_DX = (3000.0, 5000.0, 5001.0, 12000.0, 27000.0)
#: The three SFCLAY1D_mynn stages the widened fixture records, now swept over
#: DX as well.  ISFFLX=0 is the arm of module_sf_mynn.F:1027-1044 that leaves
#: the exchange coefficients and fluxes at zero instead of diagnosing them;
#: the other twenty-two outputs still run the VSGD-widened WSPD.
EXPECTED_STAGES = ((1, 1), (2, 1), (1, 0))

#: module_sf_mynn.F:1027-1044.  ISFFLX<1 assigns these thirteen outputs the
#: literal constant 0 and leaves the other twenty-two on the ISFFLX=1 code
#: path, so an ISFFLX=0 stage reuses its ``(dx, 1)`` table for the twenty-two
#: and 0 for these thirteen.  Both halves are checked below.
ISFFLX0_ZEROED = (
    "hfx", "qfx", "flhc", "flqc", "lh", "chs", "ch", "chs2", "cqs2",
    "ck", "cd", "cka", "cda",
)

#: DX = 5000 is on the threshold, so ``max()`` clamps VSGD to zero and WRF
#: writes byte-identical rows to DX = 3000; one table covers both exactly.
DX_TABLE_ALIAS = {5000.0: 3000.0}

#: The FP32 ULP distance of the CPU reference from this oracle, measured per
#: (stage, output, column) -- one integer per compared number, in
#: ``EXPECTED_CASES`` order, where a stage is one (dx, itimestep).  An output
#: a stage does not name is bitwise on every column there and must stay
#: bitwise: the lookup returns zeros.  There is no margin to justify, because
#: there is no margin: each entry is the residue measured at that element and
#: the gate is ``residue <= entry``.
#:
#: The numbers are the elementwise maximum over THREE NumPy builds -- Windows
#: NumPy 2.2.6 / CPython 3.13.7, WSL Ubuntu-24.04 NumPy 2.4.3 / CPython 3.12.3,
#: and Ubuntu-22.04 NumPy 2.5.1 / CPython 3.12.13 -- because NumPy's float32
#: ``arctan`` and ``**`` are not the same function on all three, and
#: ``build_coarse.sh`` runs this script under a Linux build while pytest runs
#: under the Windows one.  They disagree at 300 of 5150 CPU elements, 212 of
#: them here.  The revision before last was measured on Windows only and failed
#: 13 comparisons here; the last was measured on Windows and WSL and failed six
#: under NumPy 2.5.1.  That is why the union is what is recorded.  The
#: derivation -- which primitive moves which column, and why NumPy 2.5.1 is the
#: build closest to the glibc functions gfortran linked -- is in
#: ``tests/test_mynn_surface.py``.
#:
#: These mirror ``tests/test_mynn_surface_coarse.py``, which asserts them
#: equal -- two copies of one measurement drift otherwise.  A ratchet to
#: lower, never to raise.
CPU_ULP = {
    (3000.0, 1): {
        "zol": (1, 0, 0, 0, 1, 1, 0, 0, 0, 0),
        "rmol": (0, 0, 0, 0, 1, 1, 0, 0, 0, 0),
        "ust": (2, 0, 0, 0, 0, 1, 0, 0, 0, 0),
        "ustm": (2, 0, 0, 0, 0, 1, 0, 0, 0, 0),
        "psim": (1, 0, 0, 0, 1, 3, 0, 0, 1, 0),
        "psih": (1, 0, 0, 0, 0, 1, 2, 0, 1, 0),
        "chs": (1, 0, 0, 0, 0, 2, 0, 0, 0, 0),
        "chs2": (1, 0, 0, 0, 0, 1, 0, 0, 0, 0),
        "cqs2": (1, 0, 0, 0, 0, 1, 0, 0, 0, 0),
        "ch": (2, 0, 0, 0, 0, 1, 0, 0, 0, 0),
        "flhc": (2, 0, 0, 0, 0, 1, 0, 0, 0, 0),
        "flqc": (1, 0, 0, 0, 0, 2, 0, 0, 0, 0),
        "qgh": (0, 0, 0, 0, 0, 1, 0, 1, 3, 0),
        "qsfc": (0, 0, 0, 0, 0, 1, 0, 0, 1, 0),
        "hfx": (2, 0, 0, 0, 0, 1, 0, 0, 0, 0),
        "qfx": (1, 0, 0, 0, 0, 2, 0, 0, 10, 0),
        "lh": (1, 0, 0, 0, 0, 3, 0, 0, 6, 0),
        "u10": (1, 0, 1, 0, 2, 1, 0, 0, 0, 0),
        "v10": (1, 0, 1, 0, 2, 2, 0, 0, 0, 0),
        # land_qsfc_unset: 0 on NumPy 2.2.6/2.4.3, 1 on 2.5.1.        +2.5.1
        "ck": (0, 0, 1, 0, 2, 1, 0, 0, 0, 0),
        "cka": (1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        "cd": (0, 0, 2, 0, 3, 0, 0, 0, 0, 0),
        "cda": (3, 0, 0, 0, 0, 2, 0, 0, 0, 0),
        "qstar": (0, 0, 0, 0, 0, 0, 0, 0, 5, 0),
    },
    (3000.0, 2): {
        "zol": (0, 0, 2, 0, 1, 2, 0, 0, 0, 0),
        "rmol": (0, 0, 2, 0, 1, 3, 0, 0, 0, 0),
        "ust": (0, 0, 0, 0, 0, 1, 0, 0, 0, 0),
        "ustm": (0, 0, 0, 0, 0, 1, 0, 0, 0, 0),
        "mol": (0, 0, 2, 0, 0, 1, 0, 0, 0, 0),
        "psim": (0, 0, 2, 0, 1, 2, 0, 0, 1, 1),
        "psih": (0, 0, 2, 0, 0, 2, 0, 0, 1, 0),
        "chs": (0, 0, 1, 0, 0, 2, 0, 0, 0, 0),
        "chs2": (0, 0, 1, 0, 0, 1, 1, 0, 0, 0),
        "cqs2": (0, 0, 0, 0, 0, 1, 1, 0, 0, 0),
        "ch": (0, 0, 2, 0, 0, 2, 0, 0, 0, 0),
        "flhc": (0, 0, 2, 0, 0, 1, 0, 0, 0, 0),
        "flqc": (0, 0, 0, 0, 0, 1, 0, 0, 0, 0),
        "qgh": (0, 0, 0, 0, 0, 1, 0, 1, 3, 0),
        "qsfc": (0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
        "hfx": (0, 0, 3, 0, 0, 1, 0, 0, 0, 0),
        "qfx": (0, 0, 0, 0, 0, 2, 0, 0, 8, 0),
        "lh": (0, 0, 0, 0, 0, 3, 0, 0, 10, 0),
        "u10": (0, 0, 0, 0, 1, 1, 0, 0, 2, 0),
        "v10": (0, 0, 0, 0, 1, 1, 0, 0, 2, 0),
        "ck": (0, 0, 0, 0, 2, 2, 0, 0, 0, 0),
        "cka": (0, 0, 0, 0, 1, 1, 0, 0, 0, 0),
        "cd": (0, 0, 0, 0, 2, 0, 0, 0, 0, 0),
        "cda": (0, 0, 0, 0, 1, 1, 0, 0, 0, 0),
        "qstar": (0, 0, 0, 0, 0, 1, 0, 0, 5, 0),
    },
    (5001.0, 1): {
        "zol": (4, 0, 0, 0, 0, 2, 0, 0, 1, 0),
        "rmol": (3, 0, 0, 0, 0, 1, 0, 0, 1, 0),
        "ust": (3, 0, 0, 0, 1, 0, 0, 0, 0, 0),
        "ustm": (3, 0, 0, 0, 1, 0, 0, 0, 0, 0),
        "mol": (1, 0, 0, 0, 0, 2, 1, 0, 0, 0),
        "psim": (2, 0, 0, 0, 1, 4, 0, 0, 1, 1),
        "psih": (1, 0, 0, 0, 0, 1, 1, 0, 1, 0),
        "chs": (3, 0, 0, 0, 1, 2, 1, 0, 0, 0),
        "chs2": (2, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        "cqs2": (2, 0, 0, 0, 1, 1, 0, 0, 0, 0),
        "ch": (3, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        "flhc": (4, 0, 0, 0, 1, 1, 1, 0, 0, 0),
        "flqc": (3, 0, 0, 0, 1, 0, 1, 0, 0, 0),
        "qgh": (0, 0, 0, 0, 0, 1, 0, 1, 3, 0),
        "qsfc": (0, 0, 0, 0, 0, 1, 0, 0, 1, 0),
        "hfx": (4, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        "qfx": (3, 0, 0, 0, 0, 2, 1, 0, 11, 0),
        "lh": (4, 0, 0, 0, 0, 2, 1, 0, 7, 0),
        # free_convective_land: 0 on NumPy 2.2.6/2.4.3, 1 on 2.5.1.   +2.5.1
        "u10": (0, 0, 0, 0, 1, 1, 0, 0, 0, 0),
        "v10": (1, 0, 0, 0, 1, 2, 0, 0, 0, 0),
        "ck": (1, 0, 0, 0, 2, 1, 1, 0, 0, 0),
        "cka": (3, 0, 0, 0, 1, 1, 1, 0, 0, 0),
        "cd": (4, 0, 0, 0, 3, 0, 2, 0, 0, 0),
        "cda": (5, 0, 0, 0, 1, 1, 0, 0, 0, 0),
        "qstar": (0, 0, 0, 0, 0, 2, 0, 0, 5, 0),
    },
    (5001.0, 2): {
        "zol": (0, 0, 2, 0, 0, 1, 0, 0, 0, 0),
        "rmol": (0, 0, 2, 0, 0, 1, 0, 0, 0, 0),
        "mol": (0, 0, 2, 0, 0, 1, 0, 0, 1, 0),
        "psim": (0, 0, 2, 0, 0, 1, 0, 0, 1, 1),
        "psih": (0, 0, 3, 0, 0, 0, 0, 0, 1, 0),
        "chs": (0, 0, 1, 0, 0, 2, 0, 0, 1, 0),
        "chs2": (0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
        "cqs2": (0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
        "ch": (0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
        "flhc": (0, 0, 1, 0, 0, 0, 0, 0, 1, 0),
        "flqc": (0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
        "qgh": (0, 0, 0, 0, 0, 1, 0, 1, 3, 0),
        "qsfc": (0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
        "hfx": (0, 0, 1, 0, 0, 0, 0, 0, 1, 0),
        "qfx": (0, 0, 0, 0, 0, 0, 0, 0, 8, 0),
        "lh": (0, 0, 0, 0, 0, 0, 0, 0, 10, 0),
        "u10": (0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
        "v10": (0, 0, 1, 0, 1, 0, 0, 0, 0, 0),
        "ck": (0, 0, 1, 0, 0, 0, 0, 0, 1, 0),
        "cka": (0, 0, 2, 0, 0, 0, 0, 0, 0, 0),
        "cd": (0, 0, 2, 0, 0, 0, 0, 0, 2, 0),
        "cda": (0, 0, 3, 0, 0, 0, 0, 0, 0, 0),
        # midres_water re-measured 4 -> 5 when the psi tables moved onto the
        # glibc transcriptions; identical on all three builds.       +psi_libm
        "qstar": (0, 0, 0, 0, 0, 1, 0, 0, 5, 0),
    },
    (12000.0, 1): {
        "zol": (0, 0, 0, 0, 1, 3, 1, 2, 0, 0),
        "rmol": (0, 0, 0, 0, 1, 3, 1, 3, 0, 0),
        "ust": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "ustm": (0, 0, 0, 0, 0, 0, 0, 1, 1, 0),
        "mol": (0, 0, 0, 0, 1, 0, 3, 1, 1, 0),
        # land_qsfc_unset: 1 on NumPy 2.2.6/2.4.3, 2 on 2.5.1.        +2.5.1
        "psim": (0, 0, 0, 0, 1, 2, 0, 2, 1, 1),
        "psih": (2, 0, 0, 0, 1, 2, 3, 4, 1, 0),
        "chs": (0, 0, 0, 0, 1, 0, 2, 2, 1, 0),
        "chs2": (2, 0, 0, 0, 0, 0, 0, 4, 0, 0),
        "cqs2": (2, 0, 0, 0, 0, 0, 1, 2, 0, 0),
        "ch": (0, 0, 0, 0, 2, 0, 2, 2, 1, 0),
        "flhc": (0, 0, 0, 0, 2, 0, 2, 2, 1, 0),
        "flqc": (1, 0, 0, 0, 1, 0, 2, 2, 0, 0),
        "qgh": (0, 0, 0, 0, 0, 1, 0, 1, 3, 0),
        "qsfc": (0, 0, 0, 0, 0, 1, 0, 0, 1, 0),
        "hfx": (0, 0, 0, 0, 2, 0, 2, 3, 1, 0),
        "qfx": (1, 0, 0, 0, 0, 2, 2, 2, 10, 0),
        "lh": (1, 0, 0, 0, 0, 3, 3, 2, 6, 0),
        "u10": (0, 0, 0, 0, 0, 1, 0, 0, 0, 0),
        "v10": (0, 0, 0, 0, 1, 1, 0, 0, 0, 0),
        "ck": (0, 0, 0, 0, 2, 3, 1, 1, 0, 0),
        "cka": (1, 0, 0, 0, 1, 0, 1, 4, 2, 0),
        "cd": (0, 0, 0, 0, 2, 2, 3, 0, 0, 0),
        "cda": (0, 0, 0, 0, 0, 0, 0, 2, 4, 0),
        "qstar": (0, 0, 0, 0, 0, 0, 0, 0, 5, 0),
    },
    (12000.0, 2): {
        "zol": (0, 0, 2, 0, 1, 0, 0, 1, 1, 0),
        "rmol": (0, 0, 2, 0, 1, 0, 0, 2, 1, 0),
        "ust": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "ustm": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "psim": (0, 0, 1, 0, 1, 1, 1, 2, 1, 1),
        "psih": (0, 0, 1, 0, 0, 0, 0, 1, 1, 0),
        "chs": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "chs2": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "cqs2": (0, 0, 0, 0, 0, 1, 0, 1, 0, 0),
        "ch": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "flhc": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "qgh": (0, 0, 0, 0, 0, 1, 0, 1, 3, 0),
        "qsfc": (0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
        "qfx": (0, 0, 0, 0, 0, 0, 0, 0, 8, 0),
        "lh": (0, 0, 0, 0, 0, 0, 0, 0, 10, 0),
        "u10": (0, 0, 1, 0, 1, 0, 0, 0, 0, 0),
        "v10": (0, 0, 1, 0, 1, 0, 0, 0, 1, 0),
        "ck": (0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
        "cka": (0, 0, 1, 0, 0, 0, 0, 2, 0, 0),
        "cd": (0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
        "cda": (0, 0, 1, 0, 0, 0, 0, 2, 0, 0),
        "qstar": (0, 0, 0, 0, 0, 0, 0, 0, 5, 0),
    },
    (27000.0, 1): {
        "zol": (3, 0, 0, 0, 0, 0, 0, 1, 1, 0),
        "rmol": (2, 0, 0, 0, 0, 0, 0, 1, 1, 0),
        "ust": (1, 0, 0, 0, 0, 1, 0, 1, 0, 0),
        "ustm": (1, 0, 0, 0, 0, 1, 0, 1, 0, 0),
        "psim": (1, 0, 0, 0, 0, 6, 0, 3, 1, 0),
        "psih": (1, 0, 0, 0, 0, 0, 0, 1, 1, 0),
        "chs": (0, 0, 0, 0, 0, 2, 0, 2, 0, 0),
        # strong_stable_land 1 -> 2, gale_land 1 -> 0: the same re-measurement,
        # net zero budget on this row.                               +psi_libm
        "chs2": (2, 0, 0, 0, 0, 0, 0, 2, 0, 0),
        "cqs2": (2, 0, 0, 0, 0, 1, 0, 3, 0, 2),
        "ch": (1, 0, 0, 0, 0, 1, 0, 1, 0, 0),
        "flhc": (1, 0, 0, 0, 0, 1, 0, 1, 0, 0),
        "flqc": (1, 0, 0, 0, 0, 3, 0, 1, 0, 0),
        "qgh": (0, 0, 0, 0, 0, 1, 0, 1, 3, 0),
        "qsfc": (0, 0, 0, 0, 0, 1, 0, 0, 1, 0),
        "hfx": (1, 0, 0, 0, 0, 1, 0, 2, 0, 0),
        # land_qsfc_unset 1 -> 2, strong_stable_land 1 -> 0; net zero on
        # this row.                                                  +psi_libm
        "qfx": (0, 0, 0, 0, 0, 2, 0, 1, 11, 0),
        # land_qsfc_unset 1 -> 2, strong_stable_land 1 -> 0; net zero on
        # this row.                                                  +psi_libm
        "lh": (0, 0, 0, 0, 0, 2, 0, 2, 6, 0),
        "u10": (0, 0, 0, 0, 1, 1, 0, 0, 0, 0),
        "v10": (1, 0, 0, 0, 1, 2, 0, 0, 0, 0),
        "ck": (2, 0, 0, 0, 1, 0, 1, 1, 0, 0),
        "cka": (1, 0, 0, 0, 0, 0, 0, 2, 0, 0),
        "cd": (3, 0, 0, 0, 2, 0, 0, 1, 0, 0),
        "cda": (2, 0, 0, 0, 0, 1, 0, 2, 0, 0),
        "qstar": (0, 0, 0, 0, 0, 0, 0, 0, 5, 0),
    },
    (27000.0, 2): {
        "zol": (0, 0, 1, 0, 0, 0, 0, 2, 0, 0),
        "rmol": (0, 0, 1, 0, 0, 0, 0, 2, 0, 0),
        "ust": (0, 0, 2, 0, 0, 1, 0, 1, 0, 0),
        "ustm": (0, 0, 2, 0, 0, 1, 0, 1, 0, 0),
        "mol": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "psim": (0, 0, 1, 0, 0, 3, 0, 4, 1, 1),
        "psih": (0, 0, 0, 0, 0, 0, 0, 2, 1, 0),
        "chs": (0, 0, 1, 0, 0, 1, 0, 2, 0, 0),
        "chs2": (0, 0, 1, 0, 0, 1, 0, 1, 0, 0),
        "cqs2": (0, 0, 1, 0, 0, 1, 1, 1, 0, 0),
        "ch": (0, 0, 1, 0, 0, 1, 0, 0, 0, 0),
        "flhc": (0, 0, 2, 0, 0, 1, 0, 1, 0, 0),
        # thin_land_log10_wind 0 -> 1, and two columns down; net -2 ULP on
        # this row.                                                  +psi_libm
        "flqc": (0, 0, 0, 0, 1, 0, 0, 1, 0, 0),
        "qgh": (0, 0, 0, 0, 0, 1, 0, 1, 3, 0),
        "qsfc": (0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
        "hfx": (0, 0, 3, 0, 0, 1, 0, 1, 0, 0),
        # QFX/LH at the 27 km spacing are the FLQC*(QSFCMR-QV1) and XLV*QFX
        # cancellation columns named in the note above; re-measuring them on
        # the glibc psi tables moves midres_water 6 -> 8 and
        # thin_land_log10_wind 0 -> 1.  Identical on all three builds, and
        # paid for on this stage by cka below.                       +psi_libm
        "qfx": (0, 0, 0, 0, 0, 0, 0, 1, 8, 0),
        # Same two columns as qfx: 7 -> 9 and 0 -> 1.                +psi_libm
        "lh": (0, 0, 0, 0, 0, 0, 0, 1, 9, 0),
        "u10": (0, 0, 1, 0, 1, 1, 0, 0, 0, 0),
        "v10": (0, 0, 1, 0, 1, 2, 0, 0, 0, 0),
        "ck": (2, 0, 0, 0, 1, 0, 0, 2, 0, 0),
        # Re-measured 6 -> 2 ULP summed.                             +psi_libm
        "cka": (0, 0, 0, 0, 1, 0, 0, 1, 0, 0),
        "cd": (0, 0, 0, 0, 2, 0, 0, 1, 0, 0),
        "cda": (0, 0, 2, 0, 0, 2, 0, 0, 2, 0),
        "qstar": (0, 0, 0, 0, 0, 0, 0, 0, 5, 0),
    },
}

#: Outputs WRF leaves untouched when VSGD switches on, measured per stage.
#: At step 2 even GZ1OZ0/WSTAR/ZNT move, because the DX branch has by then
#: propagated into the carried UST/MOL that charnock_1955 (:657) reads; with
#: ISFFLX=0 the thirteen zeroed outputs cannot respond to anything.
DX_INSENSITIVE = {
    (1, 1): ("regime", "qgh", "qsfc", "q2", "gz1oz0", "wstar", "cpm", "znt"),
    (2, 1): ("regime", "qgh", "qsfc", "q2", "cpm"),
    (1, 0): (
        "regime", "qgh", "qsfc", "q2", "gz1oz0", "wstar", "cpm", "znt",
    ) + ISFFLX0_ZEROED,
}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="ascii") as stream:
        return list(csv.DictReader(stream))


def _fields(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float32)
        for key in rows[0]
        if key not in ("case", "itimestep", "isfflx", "dx")
    }


def _expected_vsgd(dx: float) -> float:
    f = np.float32
    return float(
        f(0.32) * max(f(dx) / f(5000.0) - f(1.0), f(0.0)) ** f(0.33)
    )


def _table(dx: float, itimestep: int):
    return CPU_ULP[(DX_TABLE_ALIAS.get(dx, dx), itimestep)]


def _budget(table, name, zeroed=()) -> np.ndarray:
    row = () if name in zeroed else table.get(name, ())
    if not row:
        return np.zeros(len(EXPECTED_CASES), dtype=np.int64)
    if len(row) != len(EXPECTED_CASES):
        raise ValueError(f"{name}: table row is {len(row)} wide")
    return np.asarray(row, dtype=np.int64)


def _compare(label, table, zeroed, actual, fields, report, failures) -> None:
    for name in COLUMN_OUTPUTS:
        got = np.asarray(actual[name], dtype=np.float32)
        ref = fields[name]
        abs_err = np.abs(got - ref)
        denom = np.abs(ref)
        rel = np.where(denom > 0.0, abs_err / np.where(denom > 0.0, denom, 1.0),
                       0.0)
        residue = fp32_ulp_distance(got, ref)
        budget = _budget(table, name, zeroed)
        report.append((label, name, float(abs_err.max()), float(rel.max()),
                       int(residue.max()), int(budget.max())))
        if name == "regime":
            if not np.array_equal(got, ref):
                failures.append(f"{label}/regime is {got}, expected {ref}")
            continue
        for index in np.flatnonzero(residue > budget):
            failures.append(
                f"{label}/{name}[{EXPECTED_CASES[index]}]: "
                f"{int(residue[index])} ULP from the unmodified WRF oracle "
                f"exceeds the measured {int(budget[index])}"
            )


def _assert_branch_coverage(stages, failures) -> None:
    """The whole point of this fixture is that VSGD is live somewhere."""

    if _expected_vsgd(3000.0) != 0.0 or _expected_vsgd(5000.0) != 0.0:
        failures.append("VSGD is not clamped to zero at or below DX=5000")

    moved_any = False
    for itimestep, isfflx in EXPECTED_STAGES:
        base = stages[(3000.0, itimestep, isfflx)]
        at_threshold = stages[(5000.0, itimestep, isfflx)]
        for name in COLUMN_OUTPUTS:
            if not np.array_equal(at_threshold[name], base[name]):
                failures.append(
                    f"DX=5000 moved {name} at stage {(itimestep, isfflx)};"
                    " max() should have clamped, and DX_TABLE_ALIAS assumes"
                    " it did"
                )
        frozen = set(DX_INSENSITIVE[(itimestep, isfflx)])
        for dx in (5001.0, 12000.0, 27000.0):
            if _expected_vsgd(dx) <= 0.0:
                failures.append(f"DX={dx} did not produce a positive VSGD")
            stage = stages[(dx, itimestep, isfflx)]
            still = {
                name for name in COLUMN_OUTPUTS
                if np.array_equal(stage[name], base[name])
            }
            if still != frozen:
                failures.append(
                    f"DX={dx} stage {(itimestep, isfflx)}: outputs unmoved by"
                    f" VSGD are {sorted(still)}, measured {sorted(frozen)}"
                )
            else:
                moved_any = True
            # module_sf_mynn.F:556 + :585-586 in full.  WSTAR is read from the
            # fixture rather than assumed constant, because at itimestep=2 it
            # has itself moved with DX.
            quadrature = np.maximum(
                np.sqrt(
                    stage["u1"].astype(np.float64) ** 2
                    + stage["v1"].astype(np.float64) ** 2
                    + stage["wstar"].astype(np.float64) ** 2
                    + _expected_vsgd(dx) ** 2
                ),
                0.1,   # wmin, module_sf_mynn.F:82
            )
            if not np.allclose(stage["wspd"], quadrature, rtol=2.0e-7):
                failures.append(
                    f"DX={dx} stage {(itimestep, isfflx)} WSPD is not the"
                    " VSGD quadrature sum"
                )
    if not moved_any:
        failures.append("the DX>5 km branch is not exercised at all")

    # ISFFLX<1 is a pure post-processing branch (:1027-1044): thirteen outputs
    # become the constant 0 and nothing else changes.  Both halves are checked
    # at every spacing, because that is what licenses reusing the (dx, 1)
    # table for the ISFFLX=0 stage.
    for dx in EXPECTED_DX:
        off = stages[(dx, 1, 0)]
        on = stages[(dx, 1, 1)]
        for name in ISFFLX0_ZEROED:
            if np.any(off[name] != 0.0):
                failures.append(f"DX={dx} isfflx=0 does not zero {name}")
            if not np.any(on[name] != 0.0):
                failures.append(
                    f"DX={dx} {name} is zero on the isfflx=1 stage too, so"
                    " the isfflx=0 zeroing check discriminates nothing"
                )
        for name in COLUMN_OUTPUTS:
            if name in ISFFLX0_ZEROED:
                continue
            if not np.array_equal(off[name], on[name]):
                failures.append(
                    f"DX={dx} isfflx=0 moved {name}, which :1027-1044 leaves"
                    " alone"
                )


def validate(path: Path) -> None:
    failures: list[str] = []
    report: list[tuple[str, str, float, float, int, int]] = []

    rows = _read(path)
    stages: dict[tuple[float, int, int], dict[str, np.ndarray]] = {}
    keys = sorted(
        {(float(r["dx"]), int(r["itimestep"]), int(r["isfflx"])) for r in rows}
    )
    for key in keys:
        selected = [
            r for r in rows
            if (float(r["dx"]), int(r["itimestep"]), int(r["isfflx"])) == key
        ]
        names = tuple(r["case"] for r in selected)
        if names != EXPECTED_CASES:
            raise ValueError(f"stage {key} cases are {names}")
        stages[key] = _fields(selected)
    for dx in EXPECTED_DX:
        for itimestep, isfflx in EXPECTED_STAGES:
            if (dx, itimestep, isfflx) not in stages:
                raise ValueError(
                    f"missing stage dx={dx} step={itimestep} isfflx={isfflx}"
                )
    if len(stages) != len(EXPECTED_DX) * len(EXPECTED_STAGES):
        raise ValueError(f"unexpected stage set: {sorted(stages)}")

    for (dx, itimestep, isfflx), fields in stages.items():
        if not all(math.isfinite(v) for a in fields.values() for v in a):
            raise ValueError(
                f"stage dx={dx} step={itimestep} isfflx={isfflx} not finite"
            )
        values = {
            name: fields[INPUT_ALIASES.get(name, name)] for name in INPUT_NAMES
        }
        actual = mynn_surface_layer_default(
            values, dx=dx, itimestep=itimestep, isfflx=isfflx,
            mol=fields["mol_input"], ustm=fields["ustm_input"],
        )
        _compare(
            f"sfclay1d/dx{dx:.0f}/step{itimestep}/isfflx{isfflx}",
            _table(dx, itimestep),
            ISFFLX0_ZEROED if isfflx == 0 else (),
            actual, fields, report, failures,
        )

    _assert_branch_coverage(stages, failures)

    width = max(len(f"{label}/{name}") for label, name, *_ in report)
    for label, name, abs_err, rel, ulp, budget in report:
        print(f"{label}/{name:<{width}}  max_abs {abs_err:.6e}  "
              f"max_rel {rel:.6e}  max_ulp {ulp}  gate {budget}")
    if failures:
        for line in failures:
            print(f"FAIL {line}", file=sys.stderr)
        raise SystemExit(1)
    print(f"OK: {len(stages)} stages x {len(EXPECTED_CASES)} columns "
          f"within the measured per-column ULP residue")


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        raise SystemExit("usage: validate_coarse_oracle.py COARSE.csv")
    validate(Path(argv[1]))


if __name__ == "__main__":
    main(sys.argv)
