"""WRF-oracle parity for the over-water ISFTCFLX branches of MYNN.

``module_sf_mynn.F`` reads ISFTCFLX in exactly three places, and the third is
not where the header comment or the routine names suggest::

    :631-662   the aerodynamic roughness z0 over water
    :680-710   the thermal/moisture roughness lengths zt and zq over water
    :1067-1072 the AHW dissipative-heating term added to HFX over water

The selection, read from the source rather than from the :35-41 header (which
does not mention that ISFTCFLX=4 exists)::

    ISFTCFLX  z0                     zt/zq                  HFX term
    0         charnock_1955          fairall_etal_2003      no
    1         davis_etal_2008        fairall_etal_2003      yes
    2         davis_etal_2008        garratt_1992           yes
    3         Taylor_Yelland_2001    fairall_etal_2003      yes
    4         charnock_1955          NEVER ASSIGNED         yes

``z_t``/``z_q`` are undecorated local automatics (:474) and :680-702 has no
arm for 4, so ISFTCFLX=4 is undefined and both the CPU reference and the
kernel launcher reject it rather than invent a value.  The COARE 3.5 half of
the 0/1/3 arms -- ``edson_etal_2013`` and ``fairall_etal_2014`` -- is dead
under the ``REAL, PARAMETER :: COARE_OPT=3.0`` at :85, so it can only be
oracled at its own entry points, which is what the leaf fixture does.

Two fixtures, both from the same pinned WRF v4.6.1 commit d66e442f and the
same unmodified ``phys/module_sf_mynn.F``
(``tools/mynn_wrf461_oracle/run_surface_layer_water.F90``, built by
``build_water.sh``):

* ``surface-layer-water.csv`` -- ``SFCLAY1D_mynn`` over twelve columns for
  ISFTCFLX = 0/1/2/3, advanced across two timesteps so the in-place ZNT
  rewrite of every water branch persists into the next step.  Nine columns are
  water; ``control_land`` and ``control_snow_land`` are negative controls that
  must not move with ISFTCFLX at all; ``xland_exactly_1p5`` sits on the
  boundary, where :625's ``.GE. 0`` sends it to the water roughness while
  :1065/:1073's ``.GT.``/``.LT.`` pair leaves its HFX untouched and
  ``garratt_1992``'s own ``landsea-1.5 .GT. 0`` sends it to that leaf's LAND
  arm.
* ``surface-layer-water-leaf.csv`` -- all seven roughness leaves called
  directly over 32 samples chosen to bind every clamp and every internal arm
  they have, plus ``garratt_1992`` at landsea 2.0, 1.5 and 1.0.

Two ULP budgets, and they are not the same kind of number.

``LEAF`` has no budget: every one of the 288 leaf rows is bitwise identical to
the unmodified module on all three platforms.  That is not luck.  The water
leaves in ``gpuwm/core/mynn_surface.py`` route ``exp``/``log``/``**`` through
``gpuwm.core.noahmp_libm``, so they return the glibc 2.39 words gfortran
linked against instead of whatever the host NumPy computes, and the
three-platform union collapses to zero by construction rather than by
measurement.  This is the follow-up the rest of that file still owes.

``WATER_ULP`` is the measured three-platform union for the whole column --
Windows NumPy 2.2.6 / CPython 3.13.7, WSL Ubuntu 24.04 NumPy 2.4.3 / CPython
3.12.3, and the rented Linux box NumPy 2.5.1 / CPython 3.12.13 -- one integer
per (stage, output, column), every entry the elementwise maximum of the three
with no slack.  The three platforms disagree on 129 of the 3,360 compared
elements, so the union is doing real work.  What is left in it is the psi
tables, the saturation ``exp`` and the 10 m ``log`` -- all still NumPy, all
still land-and-water shared, and none of it this lane's ISFTCFLX code: ``br``
and ``znt`` do not appear in the table at all, at any stage, on any column,
which is the tightest statement available that the branches themselves are
exact.

``WATER_CUDA_ULP`` is the same table for the kernel, measured once on the
RTX 5090 (cupy 14.1.1), because cupy runs nowhere else.  It is much larger,
and the fixture says why rather than leaving it to be argued: ``br`` is
bitwise on every CPU row and reaches 696 ULP on ``gale_water`` on the device.
That column has TSK - T1 = 1 K, so DTHVDZ = THV1 - THVGB (:559-560) is a
~1 K difference of ~289 K numbers, a 300x cancellation; one ULP of disagreement
between CUDA's ``powf`` and glibc's in the Exner factor above it is enough to
move BR by hundreds of ULP, and ZOL/PSIM/PSIH/HFX inherit it.  HFX is worst at
ISFTCFLX=0 (2,390) and *smaller* at 1/2/3 (286-539), because the dissipative
term the non-default identities add is large next to the cancelling
difference -- so the new branches reduce this residue rather than cause it.
ZNT, the direct output of the ported leaves, is 0 ULP on the CPU everywhere
and at most 10 on the device, all of it in the ISFTCFLX=3 arm where
``powf(hs/Lp, 4.5)`` amplifies the device shim's error by 4.5x.

Both tables are ratchets: lower them as the shims are unified, never raise.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest

from gpuwm.core.fp32_ulp import fp32_ulp_distance
from gpuwm.core.mynn_surface import (
    _charnock_1955,
    _davis_etal_2008,
    _edson_etal_2013,
    _fairall_etal_2003,
    _fairall_etal_2014,
    _garratt_1992,
    _taylor_yelland_2001,
    mynn_surface_layer_default,
)


ORACLE_DIR = Path(__file__).parents[1] / "gpuwm" / "data" / "mynn" / "oracle"
WATER_ORACLE = ORACLE_DIR / "surface-layer-water.csv"
LEAF_ORACLE = ORACLE_DIR / "surface-layer-water-leaf.csv"
HARNESS = (
    Path(__file__).parents[1] / "tools" / "mynn_wrf461_oracle"
    / "run_surface_layer_water.F90"
)

#: Identity of the fixtures and of the harness that produced them, so a
#: regenerated or hand-edited oracle cannot be passed off as WRF's answer.
#: The CSVs are covered by ``gpuwm/data/** -text``; the .F90 is not, and git
#: hands it back with CRLF on Windows, so its hash is over LF-normalized bytes.
WATER_ORACLE_SHA256 = (
    "242f85d8394e8586499c49b51ca772e4908f5eb64ed9efd5894e2c2ef4829b0d"
)
LEAF_ORACLE_SHA256 = (
    "6a81bdfb348f8ac564537e60433b77c27d1a598f914e7b28440d4020f4881916"
)
HARNESS_LF_SHA256 = (
    "3b960914af5000f48e7612410f18d2df7bd858706d3059ee2dfa9ef8236157e0"
)

#: The four identities WRF defines over water, and the two stages each is
#: advanced through.
ISFTCFLX_SWEEP = (0, 1, 2, 3)
STAGES = ((1, 1), (2, 1))
CASES = (
    "calm_water", "light_water", "moderate_water", "breezy_water",
    "windy_water", "gale_water", "hurricane_water", "cold_water",
    "extreme_water", "xland_exactly_1p5", "control_land",
    "control_snow_land",
)
WATER_CASES = tuple(name for name in CASES if name.endswith("_water"))

INPUT_NAMES = (
    "u1", "v1", "t1", "qv1", "p1", "rho1", "dz1",
    "u2", "v2", "dz2", "psfc", "tsk", "pblh", "mavail",
    "hfx", "qfx", "znt", "qsfc", "ust", "xland", "snowh",
)
INPUT_ALIASES = {
    "hfx": "hfx_input", "qfx": "qfx_input", "znt": "znt_input",
    "qsfc": "qsfc_input", "ust": "ust_input",
}
OUTPUT_NAMES = (
    "regime", "zol", "rmol", "ust", "ustm", "mol", "psim", "psih",
    "chs", "chs2", "cqs2", "ch", "flhc", "flqc", "qgh", "qsfc",
    "hfx", "qfx", "lh", "u10", "v10", "th2", "t2", "q2", "gz1oz0",
    "wspd", "br", "ck", "cka", "cd", "cda", "wstar", "qstar", "cpm", "znt",
)

#: The clamps WRF applies in the roughness leaves, as exact FP32 words, so a
#: coverage assertion can say "this sample is ON the clamp" instead of "near".
Z0_FLOOR = np.float32(1.27e-7)
Z0_CEILING = np.float32(2.85e-3)
ZTQ_FAIRALL_CEILING = np.float32(1.0e-4)
ZTQ_GARRATT_CEILING = np.float32(5.5e-5)
ZTQ_FLOOR = np.float32(2.0e-9)
ZT_FAIRALL2014_CEILING = np.float32(1.6e-4)
WATER_ULP = {
    (0, 1, 1): {
        "cda": (0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0),
        "ch": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1),
        "chs": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1),
        "chs2": (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        "cka": (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        "cqs2": (1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
        "flhc": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1),
        "hfx": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1),
        "lh": (3, 0, 0, 9, 5, 0, 0, 0, 2, 0, 0, 0),
        "mol": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1),
        "psih": (0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0),
        "psim": (0, 2, 1, 3, 32, 47, 32, 2, 32, 2, 0, 0),
        "qfx": (6, 0, 0, 7, 4, 0, 0, 0, 2, 0, 0, 0),
        "qgh": (3, 3, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0),
        "qsfc": (1, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0),
        "qstar": (1, 1, 0, 3, 2, 1, 1, 0, 1, 0, 0, 0),
        "u10": (1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0),
        "v10": (0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0),
    },
    (0, 2, 1): {
        "cd": (0, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3),
        "cda": (0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 2, 0),
        "ch": (2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
        "chs": (1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0),
        "chs2": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0),
        "ck": (1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4),
        "cka": (1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0),
        "cqs2": (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
        "flhc": (2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0),
        "flqc": (1, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
        "gz1oz0": (0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0),
        "hfx": (2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0),
        "lh": (4, 2, 0, 9, 4, 0, 0, 0, 4, 0, 0, 0),
        "mol": (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        "psih": (1, 0, 3, 0, 0, 0, 0, 0, 0, 1, 1, 1),
        # The psi tables now run on the glibc transcriptions, so the 32/48/32
        # unstable-table residue collapses: re-measured 148 ULP -> 1, with
        # light_water going 0 -> 1.  Identical on all three builds. +psi_libm
        "psim": (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        "qfx": (4, 2, 0, 8, 3, 0, 0, 0, 3, 0, 0, 0),
        "qgh": (3, 3, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0),
        "qsfc": (1, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0),
        "qstar": (2, 1, 0, 3, 1, 1, 1, 0, 1, 0, 0, 0),
        "rmol": (0, 2, 3, 0, 0, 0, 0, 0, 0, 0, 1, 1),
        "u10": (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1),
        "ust": (0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
        "ustm": (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
        "v10": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1),
        "zol": (0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 1),
    },
    (1, 1, 1): {
        "cd": (0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0),
        "cda": (2, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0),
        "ch": (0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 1, 1),
        "chs": (0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 1),
        "ck": (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
        "cka": (1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        "cqs2": (0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1),
        "flhc": (0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1),
        "flqc": (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
        "gz1oz0": (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
        "hfx": (0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1),
        "lh": (3, 0, 0, 9, 4, 1, 0, 0, 2, 0, 0, 0),
        "mol": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1),
        "psih": (0, 0, 1, 2, 0, 3, 0, 0, 0, 1, 1, 0),
        "psim": (1, 1, 0, 9, 32, 45, 32, 1, 32, 0, 0, 0),
        "qfx": (6, 0, 0, 7, 4, 1, 0, 0, 2, 0, 0, 0),
        "qgh": (3, 3, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0),
        "qsfc": (1, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0),
        "qstar": (1, 1, 0, 4, 1, 1, 1, 0, 1, 0, 0, 0),
        "rmol": (0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0),
        "u10": (0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0),
        "ust": (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
        "ustm": (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
        "v10": (0, 0, 0, 0, 0, 1, 0, 1, 1, 2, 0, 0),
        "zol": (0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0),
    },
    (1, 2, 1): {
        "cd": (0, 0, 0, 2, 0, 1, 0, 0, 0, 2, 0, 3),
        "cda": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0),
        "ch": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
        "chs": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0),
        "chs2": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0),
        "ck": (1, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 4),
        "cka": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
        "cqs2": (1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0),
        "flhc": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0),
        "hfx": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0),
        "lh": (3, 0, 0, 9, 4, 0, 0, 0, 4, 0, 0, 0),
        "psih": (0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1),
        "psim": (0, 1, 1, 8, 32, 28, 32, 0, 32, 4, 3, 0),
        "q2": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0),
        "qfx": (3, 0, 0, 7, 3, 0, 0, 0, 3, 0, 0, 0),
        "qgh": (3, 3, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0),
        "qsfc": (1, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0),
        "qstar": (1, 0, 0, 4, 1, 1, 1, 0, 1, 0, 0, 0),
        "rmol": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1),
        "u10": (1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1),
        "ust": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
        "ustm": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
        "v10": (0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1),
        "zol": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1),
    },
    (2, 1, 1): {
        "cda": (2, 0, 0, 0, 0, 3, 1, 2, 0, 0, 0, 0),
        "ch": (0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 1, 1),
        "chs": (0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 1, 1),
        "chs2": (0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0),
        "cka": (0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0),
        "cqs2": (0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1),
        "flhc": (0, 0, 0, 1, 0, 0, 2, 1, 0, 0, 1, 1),
        "flqc": (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
        "gz1oz0": (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
        "hfx": (0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 1, 1),
        "lh": (3, 0, 0, 4, 4, 0, 1, 0, 1, 0, 0, 0),
        "mol": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1),
        "psih": (0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0),
        "psim": (1, 1, 2, 10, 32, 46, 27, 1, 32, 0, 0, 0),
        "qfx": (5, 0, 0, 7, 4, 0, 1, 0, 1, 0, 0, 0),
        "qgh": (3, 3, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0),
        "qsfc": (1, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0),
        "qstar": (1, 1, 0, 4, 1, 1, 1, 0, 1, 0, 0, 0),
        "rmol": (0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0),
        "u10": (0, 0, 0, 0, 0, 1, 2, 1, 0, 1, 0, 0),
        "ust": (0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0),
        "ustm": (0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0),
        "v10": (0, 0, 0, 0, 0, 1, 1, 1, 0, 2, 0, 0),
        "zol": (0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0),
    },
    (2, 2, 1): {
        "cd": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3),
        "cda": (0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 2, 0),
        "ch": (0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0),
        "chs": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0),
        "chs2": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0),
        "ck": (0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 4),
        "cka": (0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0),
        "cqs2": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0),
        "flhc": (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0),
        "flqc": (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        "hfx": (0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 2, 0),
        "lh": (4, 0, 0, 9, 3, 0, 0, 0, 2, 0, 0, 0),
        "psih": (0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1),
        "psim": (0, 2, 0, 7, 32, 28, 29, 1, 32, 0, 3, 0),
        "q2": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0),
        "qfx": (3, 0, 0, 7, 3, 0, 0, 0, 2, 0, 0, 0),
        "qgh": (3, 3, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0),
        "qsfc": (1, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0),
        "qstar": (1, 1, 0, 4, 2, 1, 1, 0, 0, 0, 0, 0),
        "rmol": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1),
        "u10": (1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1),
        "ust": (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0),
        "ustm": (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0),
        "v10": (0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1),
        "zol": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 1),
    },
    (3, 1, 1): {
        "cda": (0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0),
        "ch": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1),
        "chs": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1),
        "chs2": (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        "cka": (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        "cqs2": (1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1),
        "flhc": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1),
        "hfx": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1),
        "lh": (3, 0, 0, 8, 3, 0, 0, 0, 2, 0, 0, 0),
        "mol": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1),
        "psih": (0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 1, 0),
        "psim": (0, 4, 3, 5, 32, 51, 32, 0, 32, 2, 0, 0),
        "qfx": (5, 0, 0, 7, 3, 0, 0, 0, 2, 0, 0, 0),
        "qgh": (3, 3, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0),
        "qsfc": (1, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0),
        "qstar": (1, 0, 0, 4, 2, 1, 1, 0, 1, 0, 0, 0),
        "rmol": (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "u10": (0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0),
        "v10": (0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0),
        "zol": (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
    },
    (3, 2, 1): {
        "cd": (0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3),
        "cda": (0, 0, 3, 0, 0, 0, 0, 0, 0, 2, 2, 0),
        "ch": (0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 1, 0),
        "chs": (0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 3, 0),
        # light_water 1 -> 2 and two columns down; net -1 ULP.        +psi_libm
        "chs2": (1, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0),
        "ck": (0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 4),
        "cka": (0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0),
        "cqs2": (1, 1, 2, 0, 0, 0, 0, 0, 0, 1, 1, 0),
        "flhc": (0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 2, 0),
        "flqc": (0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "gz1oz0": (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "hfx": (0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0),
        "lh": (4, 0, 2, 9, 4, 0, 0, 0, 4, 1, 0, 0),
        "psih": (0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 1, 1),
        "psim": (0, 2, 3, 5, 32, 52, 32, 0, 32, 0, 3, 0),
        "qfx": (3, 0, 2, 7, 3, 0, 0, 0, 3, 1, 0, 0),
        "qgh": (3, 3, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0),
        "qsfc": (1, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0),
        "qstar": (1, 1, 0, 3, 2, 1, 1, 0, 1, 0, 0, 0),
        "rmol": (0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 1),
        "u10": (0, 1, 0, 0, 2, 0, 0, 1, 0, 2, 1, 1),
        "ust": (0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0),
        "ustm": (0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0),
        "v10": (0, 0, 0, 0, 1, 0, 0, 1, 0, 2, 1, 1),
        "zol": (0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1),
    },
}

def _sha256(path: Path, *, normalize_newlines: bool = False) -> str:
    data = path.read_bytes()
    if normalize_newlines:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _column_rows() -> dict[tuple[int, int, int], list[dict[str, str]]]:
    groups: dict[tuple[int, int, int], list[dict[str, str]]] = {}
    for row in _rows(WATER_ORACLE):
        key = (int(row["isftcflx"]), int(row["itimestep"]), int(row["isfflx"]))
        groups.setdefault(key, []).append(row)
    return groups


def _f32(rows, name) -> np.ndarray:
    return np.asarray([np.float32(row[name]) for row in rows],
                      dtype=np.float32)


def _evaluate(key, rows) -> dict[str, np.ndarray]:
    isftcflx, itimestep, isfflx = key
    values = {
        name: _f32(rows, INPUT_ALIASES.get(name, name))
        for name in INPUT_NAMES
    }
    return mynn_surface_layer_default(
        values,
        dx=float(np.float32(rows[0]["dx"])),
        itimestep=itimestep,
        isfflx=isfflx,
        isftcflx=isftcflx,
        mol=_f32(rows, "mol_input"),
        ustm=_f32(rows, "ustm_input"),
    )


def _budget(table, key, name, count):
    return np.asarray(table.get(key, {}).get(name, (0,) * count),
                      dtype=np.int64)


def test_the_water_fixtures_and_their_harness_are_the_pinned_bytes():
    assert _sha256(WATER_ORACLE) == WATER_ORACLE_SHA256
    assert _sha256(LEAF_ORACLE) == LEAF_ORACLE_SHA256
    assert _sha256(HARNESS, normalize_newlines=True) == HARNESS_LF_SHA256
    groups = _column_rows()
    assert sorted(groups) == sorted(
        (flx, step, flux) for flx in ISFTCFLX_SWEEP for step, flux in STAGES
    )
    for key, rows in groups.items():
        assert tuple(row["case"] for row in rows) == CASES, key


@pytest.mark.parametrize("isftcflx", ISFTCFLX_SWEEP)
@pytest.mark.parametrize("itimestep,isfflx", STAGES)
def test_cpu_reference_matches_the_water_oracle(isftcflx, itimestep, isfflx):
    key = (isftcflx, itimestep, isfflx)
    rows = _column_rows()[key]
    actual = _evaluate(key, rows)
    for name in OUTPUT_NAMES:
        expected = _f32(rows, name)
        residue = fp32_ulp_distance(actual[name], expected)
        budget = _budget(WATER_ULP, key, name, len(rows))
        over = np.nonzero(residue > budget)[0]
        assert not over.size, (
            f"{name} at isftcflx={isftcflx} step={itimestep} "
            f"isfflx={isfflx} exceeds its measured budget on "
            + ", ".join(
                f"{CASES[i]} ({int(residue[i])} > {int(budget[i])})"
                for i in over
            )
        )


def test_the_new_branches_leave_znt_and_br_bitwise_on_every_column():
    """ZNT is what the ported leaves compute; BR is what feeds every branch.

    Neither has a budget entry at any stage, so state that as a test instead
    of as a claim about a table's contents.
    """

    for key, rows in _column_rows().items():
        actual = _evaluate(key, rows)
        for name in ("znt", "br"):
            residue = fp32_ulp_distance(actual[name], _f32(rows, name))
            assert not residue.any(), (key, name, residue.tolist())


def _leaf_rows() -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in _rows(LEAF_ORACLE):
        name = row["leaf"]
        if name == "garratt_1992":
            name = f"garratt_1992@{float(np.float32(row['landsea']))}"
        grouped.setdefault(name, []).append(row)
    return grouped


LEAF_NAMES = (
    "charnock_1955", "edson_etal_2013", "davis_etal_2008",
    "taylor_yelland_2001", "fairall_etal_2003", "fairall_etal_2014",
    "garratt_1992@2.0", "garratt_1992@1.5", "garratt_1992@1.0",
)


def _evaluate_leaf(name, row):
    ustar = np.float32(row["ustar"])
    wsp10 = np.float32(row["wsp10"])
    visc = np.float32(row["visc"])
    zu = np.float32(row["zu"])
    ren = np.float32(row["ren"])
    z0_in = np.float32(row["z0_in"])
    landsea = np.float32(row["landsea"])
    if name == "charnock_1955":
        return {"z0_out": _charnock_1955(ustar, wsp10, visc, zu)}
    if name == "edson_etal_2013":
        return {"z0_out": _edson_etal_2013(ustar, wsp10, visc, zu)}
    if name == "davis_etal_2008":
        return {"z0_out": _davis_etal_2008(ustar)}
    if name == "taylor_yelland_2001":
        return {"z0_out": _taylor_yelland_2001(wsp10)}
    if name == "fairall_etal_2003":
        zt, zq = _fairall_etal_2003(ren)
    elif name == "fairall_etal_2014":
        zt, zq = _fairall_etal_2014(ren)
    else:
        zt, zq = _garratt_1992(z0_in, ren, landsea)
    return {"zt_out": zt, "zq_out": zq}


@pytest.mark.parametrize("leaf", LEAF_NAMES)
def test_cpu_leaves_are_bitwise_identical_to_the_unmodified_module(leaf):
    """max_ulp 0, not a budget: these leaves do not call NumPy at all."""

    rows = _leaf_rows()[leaf]
    assert len(rows) == 32
    for row in rows:
        for name, value in _evaluate_leaf(leaf, row).items():
            expected = np.float32(row[name])
            residue = int(fp32_ulp_distance(
                np.asarray([value], dtype=np.float32),
                np.asarray([expected], dtype=np.float32),
            )[0])
            assert residue == 0, (
                f"{leaf} sample {row['sample']} {name}: "
                f"{float(value)!r} != {float(expected)!r}"
            )


def test_the_leaf_sweep_binds_every_arm_of_every_ported_leaf():
    """Branch coverage, read off the oracle's own outputs.

    Every count below is a number of the 32 fixture samples that land ON the
    arm named, so a sweep that stopped exercising an arm fails here instead of
    quietly passing with the arm dead.
    """

    grouped = _leaf_rows()
    z0 = {name: _f32(grouped[name], "z0_out") for name in
          ("charnock_1955", "edson_etal_2013", "davis_etal_2008",
           "taylor_yelland_2001")}
    ustar = _f32(grouped["davis_etal_2008"], "ustar")
    wsp10 = _f32(grouped["taylor_yelland_2001"], "wsp10")
    ren = _f32(grouped["fairall_etal_2003"], "ren")

    # charnock_1955: the CZC ramp is MIN(MAX((wsp10m-10)/8, 0), 1), so all
    # three of its arms have to appear.
    czc = np.asarray([
        np.float32(np.float32(row["wsp10"])
                   * np.float32(np.log(np.float32(1.0e5)))
                   / np.float32(np.log(np.float32(row["zu"])
                                       / np.float32(1.0e-4))))
        for row in grouped["charnock_1955"]
    ], dtype=np.float32)
    assert (czc < 10.0).sum() >= 4
    assert ((czc >= 10.0) & (czc <= 18.0)).sum() >= 4
    assert (czc > 18.0).sum() >= 4
    # edson_etal_2013: MIN(19., wsp10m) and MAX(CZC, 0.) both live.
    assert (czc > 19.0).sum() >= 4
    assert (np.float32(0.0017) * np.minimum(czc, np.float32(19.0))
            - np.float32(0.005) < 0.0).sum() >= 2
    # ...and its ustar floor is 0.07, not charnock's 0.05, so both sides of
    # both floors are swept.
    assert (ustar < 0.05).sum() >= 4
    assert ((ustar >= 0.05) & (ustar < 0.07)).sum() >= 1
    assert (ustar >= 0.07).sum() >= 4

    # davis_etal_2008: ZW saturates at u* = 1.06.
    assert (ustar < 1.06).sum() >= 4
    assert (ustar >= 1.06).sum() >= 4
    assert (z0["davis_etal_2008"] == Z0_CEILING).sum() >= 1
    # OZO = 1.59e-5 holds ZN1 above the floor and ZN2 is larger still once
    # ZW == 1, so davis' MAX(Z_0, 1.27e-7) cannot bind.  Recorded, not
    # claimed as covered.
    assert (z0["davis_etal_2008"] == Z0_FLOOR).sum() == 0

    # Taylor_Yelland_2001: both clamps and the MAX(wsp10, 0.1) arm.
    assert (wsp10 < 0.1).sum() >= 1
    assert (z0["taylor_yelland_2001"] == Z0_FLOOR).sum() >= 1
    assert (z0["taylor_yelland_2001"] == Z0_CEILING).sum() >= 1
    assert ((z0["taylor_yelland_2001"] > Z0_FLOOR)
            & (z0["taylor_yelland_2001"] < Z0_CEILING)).sum() >= 4
    # charnock and edson reach the same ceiling from the other side -- 7 and
    # 11 of the 32 samples -- and neither can reach the floor, because both
    # add 0.11*visc/MAX(u*, floor), which is 2.4e-5 at the largest u* swept.
    assert (z0["charnock_1955"] == Z0_CEILING).sum() >= 4
    assert (z0["edson_etal_2013"] == Z0_CEILING).sum() >= 4
    assert (z0["charnock_1955"] == Z0_FLOOR).sum() == 0
    assert (z0["edson_etal_2013"] == Z0_FLOOR).sum() == 0
    for name, values in z0.items():
        assert ((values > Z0_FLOOR) & (values < Z0_CEILING)).sum() >= 20, name

    # fairall_etal_2003: the Ren<=2 test at :1442 has two arms that compute
    # the same expression, so both sides are swept to prove it cannot matter.
    assert (ren <= 2.0).sum() >= 4
    assert (ren > 2.0).sum() >= 4
    zt03 = _f32(grouped["fairall_etal_2003"], "zt_out")
    assert (zt03 == ZTQ_FAIRALL_CEILING).sum() >= 1
    assert (zt03 == ZTQ_FLOOR).sum() >= 1
    zt14 = _f32(grouped["fairall_etal_2014"], "zt_out")
    assert (zt14 == ZT_FAIRALL2014_CEILING).sum() >= 1
    assert (zt14 == ZTQ_FLOOR).sum() >= 1
    assert ((zt14 > ZTQ_FLOOR) & (zt14 < ZT_FAIRALL2014_CEILING)).sum() >= 4

    # garratt_1992 water arm: independent zt and zq, both clamps live.
    water = grouped["garratt_1992@2.0"]
    zt = _f32(water, "zt_out")
    zq = _f32(water, "zq_out")
    assert (zt == ZTQ_GARRATT_CEILING).sum() >= 1
    assert (zt == ZTQ_FLOOR).sum() >= 1
    assert ((zt > ZTQ_FLOOR) & (zt < ZTQ_GARRATT_CEILING)).sum() >= 4
    assert (zt != zq).sum() >= 4
    # ...and its land arm, which the ISFTCFLX=2 caller reaches at XLAND=1.5.
    # The land arm sets Zq = Z_0/e**2 and Zt = Zq, so zt == zq there and the
    # Reynolds number cannot appear at all; the water arm is the opposite on
    # both counts.  Its exact FP32 form is pinned bitwise by the leaf test.
    for landsea in ("garratt_1992@1.5", "garratt_1992@1.0"):
        rows = grouped[landsea]
        assert np.array_equal(_f32(rows, "zt_out"), _f32(rows, "zq_out"))
        assert not np.array_equal(_f32(rows, "zt_out"), zt)


def test_every_isftcflx_arm_moves_znt_away_from_the_others():
    """Each new arm has to be bound by a case, not merely implemented."""

    znt = {}
    for isftcflx in ISFTCFLX_SWEEP:
        rows = _column_rows()[(isftcflx, 1, 1)]
        znt[isftcflx] = {row["case"]: np.float32(row["znt"]) for row in rows}
    water = [name for name in CASES if name != "control_land"
             and name != "control_snow_land"]
    for a, b in ((0, 1), (0, 2), (0, 3), (1, 3), (2, 3)):
        moved = [c for c in water if znt[a][c] != znt[b][c]]
        assert len(moved) >= 5, (a, b, moved)
    # 1 and 2 share davis_etal_2008 for z0, so ZNT is identical between them
    # on every column and only the zt/zq selection can separate them -- which
    # is what the next test uses to bind the garratt arm.
    assert all(znt[1][c] == znt[2][c] for c in CASES)


def test_the_zt_zq_selection_separates_isftcflx_1_from_2():
    one = _column_rows()[(1, 1, 1)]
    two = _column_rows()[(2, 1, 1)]
    differing = [
        row_one["case"]
        for row_one, row_two in zip(one, two)
        if np.float32(row_one["chs"]) != np.float32(row_two["chs"])
    ]
    assert set(differing) >= set(WATER_CASES)


def test_land_columns_never_move_with_isftcflx():
    """:625 sends land past the whole ISFTCFLX block; prove it, per output."""

    base = {
        row["case"]: row for row in _column_rows()[(0, 1, 1)]
    }
    for isftcflx in (1, 2, 3):
        rows = {row["case"]: row for row in _column_rows()[(isftcflx, 1, 1)]}
        for case in ("control_land", "control_snow_land"):
            for name in OUTPUT_NAMES:
                assert (np.float32(rows[case][name])
                        == np.float32(base[case][name])), (
                    isftcflx, case, name
                )


def test_the_xland_1p5_column_takes_water_roughness_and_keeps_its_hfx():
    """The two predicates around 1.5 are not the same predicate.

    :625 is ``(XLAND-1.5).GE.0`` -- water.  :1065 is ``XLAND-1.5.GT.0`` and
    :1073 is ``.LT.0`` -- neither.  A column at exactly 1.5 therefore gets a
    water ZNT and an untouched HFX, which is the only way to tell a faithful
    transcription from an if/else that collapsed the pair.
    """

    for isftcflx in ISFTCFLX_SWEEP:
        for itimestep, isfflx in STAGES:
            key = (isftcflx, itimestep, isfflx)
            rows = _column_rows()[key]
            row = next(r for r in rows if r["case"] == "xland_exactly_1p5")
            assert np.float32(row["xland"]) == np.float32(1.5)
            assert np.float32(row["hfx"]) == np.float32(row["hfx_input"])
            assert np.float32(row["znt"]) != np.float32(row["znt_input"])
            actual = _evaluate(key, rows)
            index = CASES.index("xland_exactly_1p5")
            assert actual["hfx"][index] == np.float32(row["hfx_input"])


def test_the_dissipative_heating_term_is_what_separates_hfx():
    """:1067-1072 is ISFTCFLX-selected, water-only, and additive.

    Two negative controls, both against the oracle: dropping the term where
    WRF adds it, and adding it where WRF does not, each has to move HFX by a
    large fraction of the term itself -- orders of magnitude outside any ULP
    budget in this file.
    """

    def _term(row, ustm):
        rho1 = np.float32(row["rho1"])
        u1, v1 = np.float32(row["u1"]), np.float32(row["v1"])
        wspdi = max(np.float32(np.sqrt(u1 * u1 + v1 * v1)), np.float32(0.1))
        return np.float32(np.float32(np.float32(rho1 * ustm) * ustm) * wspdi)

    default = _evaluate((0, 1, 1), _column_rows()[(0, 1, 1)])
    for index, row in enumerate(_column_rows()[(0, 1, 1)]):
        if not row["case"].endswith("_water"):
            continue
        term = _term(row, default["ustm"][index])
        assert term > 0.0
        expected = np.float32(row["hfx"])
        assert abs(float(default["hfx"][index] + term - expected))             >= 0.5 * float(term), row["case"]

    for isftcflx in (1, 2, 3):
        key = (isftcflx, 1, 1)
        rows = _column_rows()[key]
        actual = _evaluate(key, rows)
        for index, row in enumerate(rows):
            if not row["case"].endswith("_water"):
                continue
            term = _term(row, actual["ustm"][index])
            expected = np.float32(row["hfx"])
            assert abs(float(actual["hfx"][index] - term - expected))                 >= 0.5 * float(term), (isftcflx, row["case"])


@pytest.mark.parametrize("isftcflx", [4, 5, -1, 0.0, True, "0"])
def test_undefined_isftcflx_is_refused_by_the_cpu_reference(isftcflx):
    """4 is the interesting one: WRF accepts it and then reads z_t/z_q
    uninitialized (:644 sets a z0, :680-702 has no arm), so there is no
    behaviour to port and gpuwm must not invent one."""

    rows = _column_rows()[(0, 1, 1)]
    values = {
        name: _f32(rows, INPUT_ALIASES.get(name, name))
        for name in INPUT_NAMES
    }
    with pytest.raises(ValueError, match="isftcflx"):
        mynn_surface_layer_default(values, isftcflx=isftcflx)


def test_the_default_identity_is_unchanged_by_the_new_parameter():
    rows = _column_rows()[(0, 1, 1)]
    values = {
        name: _f32(rows, INPUT_ALIASES.get(name, name))
        for name in INPUT_NAMES
    }
    explicit = mynn_surface_layer_default(
        values, mol=_f32(rows, "mol_input"), ustm=_f32(rows, "ustm_input"),
        isftcflx=0,
    )
    implicit = mynn_surface_layer_default(
        values, mol=_f32(rows, "mol_input"), ustm=_f32(rows, "ustm_input"),
    )
    for name in OUTPUT_NAMES:
        assert np.array_equal(explicit[name], implicit[name]), name
