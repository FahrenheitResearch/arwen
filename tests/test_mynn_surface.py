"""WRF-oracle checks for the default MYNN surface-layer CPU reference."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gpuwm.core import mynn_surface
from gpuwm.core.fp32_ulp import fp32_ulp_distance
from gpuwm.core.mynn_surface import (
    mynn_sfclay_first_step_state,
    mynn_surface_layer_default,
)


ORACLE_DIR = Path(__file__).parents[1] / "gpuwm" / "data" / "mynn" / "oracle"
ORACLE = ORACLE_DIR / "surface-layer.csv"
WIDE_ORACLE = ORACLE_DIR / "surface-layer-wide.csv"
WRAPPER_ORACLE = ORACLE_DIR / "surface-layer-wrapper.csv"
WIDE_VALIDATOR = (
    Path(__file__).parents[1] / "tools" / "mynn_wrf461_oracle"
    / "validate_wide_oracle.py"
)

NARROW_CASES = (
    "stable_land", "unstable_land", "neutral_land", "snow_land",
    "stable_water", "unstable_water",
)
WIDE_CASES = (
    "strong_stable_land", "clipped_stable_land", "damped_stable_land",
    "neutral_land", "free_convective_land", "land_qsfc_unset",
    "thin_land_level2_wind", "thin_land_log10_wind", "midres_water",
    "coarse_water",
)

# The FP32 ULP distance of the CPU reference from the unmodified WRF v4.6.1
# oracle, measured per (stage, output, column) -- one integer per compared
# number, in fixture column order (``NARROW_CASES`` / ``WIDE_CASES`` above).
# An output a stage does not name is bitwise on every column of that stage and
# is required to stay bitwise: the lookup returns zeros.
#
# There is no margin here to justify, because there is no margin: each entry is
# the residue that was measured at that element, and the gate is
# ``residue <= entry`` column by column.  The previous revision budgeted per
# *fixture* -- one table shared by the widened fixture's three stages, maxed
# over stages and over columns -- which on this machine left 3398 ULP of
# unearned margin, 41 of 205 (stage, output) comparisons unable to fail on a
# 1-ULP regression, and single elements as much as 10 ULP loose.  Intermediate
# schemes were measured and rejected: budgeting per (stage, output) and per
# (stage, column) and taking the smaller of the two still leaves 1068 ULP and
# only catches 49% of single-element 1-ULP moves, for the same table volume as
# this.
#
# The columns make the physics legible in a way a field-wide scalar cannot:
# QFX/LH lead because ``FLQC*(QSFCMR-QV1)`` and ``XLV*QFX``
# (module_sf_mynn.F:1004-1006) cancel three or four significant digits, and the
# residue sits in the two water columns and ``land_qsfc_unset``.  The rest
# tracks NumPy's float32 exp/log/pow not being the glibc functions gfortran
# linked.  BR, WSPD, TH2, T2, Q2, GZ1OZ0, CPM and ZNT are bitwise everywhere.
#
# These are CPU-reference numbers, so they are the elementwise maximum over
# THREE NumPy builds, and the only elements in this file carrying any slack at
# all are the 300 of 5150 -- 88 here, 212 in ``test_mynn_surface_coarse.py`` --
# where those three disagree:
#
#   * Windows NumPy 2.2.6 / CPython 3.13.7, which runs pytest on this box;
#   * WSL Ubuntu-24.04 NumPy 2.4.3 / CPython 3.12.3, which
#     ``tools/mynn_wrf461_oracle/build*.sh`` runs its validators under;
#   * Ubuntu-22.04 NumPy 2.5.1 / CPython 3.12.13 on the second RTX 5090 host.
#
# The disagreement has exactly two roots, both established against the glibc
# 2.39 functions gfortran linked when it built the oracle -- reached directly
# with ``ctypes.CDLL("libm.so.6")``, and fingerprinted by
# ``test_the_two_platform_dependent_float32_primitives_are_a_known_build``:
#
#   * ``np.arctan`` on float32.  It reaches these tables through
#     ``_psim_unstable_full`` / ``_psih_unstable_full``
#     (gpuwm/core/mynn_surface.py:63-71, :83-85) and so through the whole
#     ``_PSIM_UNSTAB`` / ``_PSIH_UNSTAB`` lookup, which is why the slack sits in
#     the unstable columns -- ``land_qsfc_unset`` and ``free_convective_land``.
#     NumPy 2.5.1 reproduces glibc ``atanf`` bit for bit on all 600060 sampled
#     arguments; 2.4.3 differs on 158548 of them and 2.2.6 differs differently
#     again.  It is the NumPy version, not the SIMD dispatch: forcing 2.4.3 off
#     AVX512 with ``NPY_DISABLE_CPU_FEATURES`` changes not one bit.
#   * ``**`` on float32, which NumPy hands to the platform ``powf``.  Both
#     Linux builds are glibc-exact at every exponent this module uses; Windows
#     gets the MSVC runtime's ``powf`` instead and differs at 2 of the 6006
#     psi-table arguments and at one ``cda`` column of the narrow fixture.
#
# ``np.log`` and ``np.exp`` on float32 are NumPy's own code on all three builds
# and are byte-identical between them, so they are a *constant* contribution to
# the residue rather than a platform axis: each differs from glibc by the same
# amount everywhere.  That is what the rest of these numbers is made of.
#
# So NumPy 2.5.1 is the build closest to WRF, not furthest.  Over the 5150 CPU
# elements this file and the coarse file measure it is 1540 ULP from the oracle
# against 1753 for 2.2.6 and 1866 for 2.4.3, and it is strictly the lowest of
# the three at 281 elements and strictly the highest at 19.  Only 7 of those 19
# clear the older pair's maximum, each by exactly 1 ULP, because at those
# elements the older ``atanf`` error happened to cancel downstream -- a
# primitive being closer to glibc does not make every composed output closer.
# They are the four entries marked ``+2.5.1`` (one here, three in the coarse
# file), and admitting them costs the other two builds 7 ULP of bite in total:
# Windows slack over the 5150 goes 161 -> 168 ULP, WSL 48 -> 55.
#
# A table measured on one platform alone is red on the others -- the
# pre-existing ``validate_coarse_oracle.py`` tables were, and its build script
# had been failing 13 comparisons under WSL before an earlier revision.
#
# These are ratchets: lower them as the FP32 shims are unified, never raise.
#
# HALF UNIFIED (2026-07-26).  The unification this note asked for has happened
# for ``psi_init``'s four lookup tables and only for those: they now build
# through ``gpuwm.core.noahmp_libm``'s ``atanf``/``logf``/``powf`` -- which the
# Noah-MP leaf lanes added after this note was written -- and are pinned by
# ``mynn_surface.PSI_TABLE_SHA256``.  What that measured, over the same 9678
# CPU elements this file, the coarse file and the water file compare:
#
#   distance from the WRF oracle   Windows 2.2.6   WSL 2.4.3   Ubuntu 2.5.1
#     NumPy-built psi tables            3343         (n/m)         2207
#     glibc-transcribed psi tables      1469          1463         1463
#
# and the three builds, which disagreed at 243 of those elements before, now
# disagree at 6 -- all of them ``cda``/``cka``/``psim`` water columns fed by
# the ``**`` and ``np.exp`` leaves this lane did NOT route.  Twelve entries
# were re-measured for it, in ``CPU_ULP`` and ``WATER_ULP``; seven columns rose
# by 1-2 ULP where the old NumPy error had been cancelling downstream, 152 ULP
# of budget came off, and every raised value is identical on all three builds.
#
# The rest of these tables were NOT re-measured and are now knowably loose --
# most of all ``WATER_ULP``'s ``psim`` rows, whose 32/48/32 unstable-table
# residue was the psi tables' error and is now 1.  Closing that is mechanical
# but wants the CUDA halves and ``WRAPPER_ULP`` re-measured on all three hosts
# too, which this lane did not do.  Until then: still ratchets, still never
# raise.
NARROW_ULP = {
    "zol": (4, 0, 0, 0, 0, 0),
    "rmol": (3, 0, 0, 0, 0, 0),
    "ust": (1, 0, 0, 0, 0, 1),
    "ustm": (1, 0, 0, 0, 0, 0),
    "mol": (1, 0, 0, 1, 0, 0),
    "psim": (4, 0, 0, 0, 0, 1),
    "psih": (4, 0, 0, 1, 0, 1),
    "chs": (3, 0, 0, 1, 0, 1),
    "chs2": (2, 0, 0, 0, 0, 1),
    "cqs2": (1, 0, 0, 0, 0, 1),
    "ch": (2, 0, 0, 1, 0, 1),
    "flhc": (2, 0, 0, 1, 0, 2),
    "flqc": (2, 0, 0, 1, 0, 2),
    "qgh": (0, 3, 0, 0, 1, 0),
    "hfx": (3, 0, 0, 0, 0, 2),
    "qfx": (2, 0, 0, 0, 0, 2),
    "lh": (2, 0, 0, 0, 0, 1),
    "u10": (1, 0, 0, 0, 0, 1),
    "v10": (1, 0, 0, 0, 0, 1),
    "ck": (4, 0, 0, 0, 0, 0),
    "cka": (6, 0, 0, 1, 0, 1),
    "cd": (1, 0, 0, 0, 0, 0),
    "cda": (4, 1, 0, 0, 0, 2),
}

WIDE_ULP = {
    (1, 1): {
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
    (2, 1): {
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
}

WRAPPER_ULP = {
    1: {
        "zol": (5, 0, 1, 0, 1, 1, 0, 2, 0, 0),
        "rmol": (4, 0, 1, 0, 1, 0, 0, 2, 0, 0),
        "ust": (1, 0, 0, 0, 0, 1, 0, 0, 0, 0),
        "ustm": (1, 0, 0, 0, 0, 1, 0, 0, 0, 0),
        "mol": (2, 1, 0, 0, 0, 0, 1, 1, 0, 0),
        "psim": (2, 0, 1, 0, 2, 2, 1, 3, 0, 0),
        "psih": (3, 0, 1, 0, 2, 1, 1, 2, 0, 0),
        "chs": (4, 1, 0, 0, 0, 1, 1, 1, 0, 0),
        "chs2": (2, 0, 0, 0, 1, 1, 1, 1, 0, 0),
        "cqs2": (2, 0, 0, 0, 0, 1, 0, 2, 0, 0),
        "ch": (3, 1, 0, 0, 0, 1, 1, 1, 0, 0),
        "flhc": (2, 1, 0, 0, 0, 2, 1, 1, 0, 0),
        "flqc": (1, 0, 0, 0, 0, 3, 1, 1, 0, 0),
        "qgh": (0, 0, 0, 0, 0, 1, 0, 1, 3, 0),
        "qsfc": (0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
        "hfx": (2, 1, 0, 0, 0, 2, 1, 1, 0, 0),
        "qfx": (1, 0, 0, 0, 0, 0, 1, 1, 7, 0),
        "lh": (1, 0, 0, 0, 0, 0, 1, 1, 8, 0),
        "u10": (1, 0, 0, 0, 0, 1, 0, 0, 0, 0),
        "v10": (0, 0, 0, 0, 0, 2, 0, 0, 1, 0),
        "ck": (1, 0, 0, 0, 0, 0, 0, 2, 0, 0),
        "cka": (4, 0, 0, 0, 0, 1, 2, 1, 0, 0),
        "cd": (2, 0, 0, 0, 0, 0, 0, 2, 0, 0),
        "cda": (5, 0, 0, 0, 0, 3, 2, 0, 0, 0),
    },
    2: {
        "zol": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "rmol": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "ust": (0, 0, 0, 0, 0, 0, 1, 1, 0, 0),
        "ustm": (0, 0, 0, 0, 0, 0, 1, 1, 0, 0),
        "psim": (0, 0, 0, 0, 1, 2, 1, 3, 0, 0),
        "psih": (1, 0, 0, 0, 0, 0, 0, 2, 0, 0),
        "chs": (0, 0, 0, 1, 0, 0, 1, 2, 0, 0),
        "chs2": (0, 0, 3, 0, 0, 0, 1, 1, 0, 0),
        "cqs2": (0, 0, 0, 0, 0, 0, 1, 1, 0, 0),
        "ch": (0, 0, 0, 1, 0, 0, 1, 4, 0, 0),
        "flhc": (0, 0, 0, 1, 0, 0, 1, 2, 0, 0),
        "flqc": (1, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        "qgh": (0, 0, 0, 0, 0, 1, 0, 1, 3, 0),
        "qsfc": (0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
        "hfx": (0, 0, 0, 0, 0, 0, 1, 3, 0, 0),
        "qfx": (1, 0, 0, 0, 0, 0, 1, 0, 6, 0),
        "lh": (1, 0, 0, 0, 0, 0, 1, 0, 8, 0),
        "u10": (0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
        "v10": (0, 0, 0, 0, 1, 0, 0, 0, 1, 2),
        "ck": (2, 0, 0, 0, 0, 0, 0, 0, 2, 1),
        "cka": (1, 0, 0, 0, 0, 0, 0, 3, 0, 0),
        "cd": (0, 0, 0, 0, 2, 0, 0, 0, 2, 2),
        "cda": (0, 0, 0, 0, 0, 0, 0, 2, 0, 0),
    },
}

#: module_sf_mynn.F:1027-1044.  With ISFFLX<1 WRF assigns these thirteen
#: outputs the literal constant 0 and leaves the other twenty-two on the same
#: code path the ISFFLX=1 first step takes.  So the ISFFLX=0 stage needs no
#: table of its own: it reuses ``WIDE_ULP[(1, 1)]`` for the twenty-two, which
#: is exact because the outputs are bit-identical, and 0 for these thirteen,
#: which is exact because they are 0.0 on both sides.  Both halves are pinned
#: by ``test_the_isfflx0_stage_zeroes_thirteen_outputs_and_shares_the_rest``.
#:
#: This is where the old per-fixture table was worst: it handed these thirteen
#: the ISFFLX=1 budgets, so on the CUDA side CH and CHS measured 0 ULP against
#: a budget of 77.
ISFFLX0_ZEROED = (
    "hfx", "qfx", "flhc", "flqc", "lh", "chs", "ch", "chs2", "cqs2",
    "ck", "cd", "cka", "cda",
)

#: What the tables above are a union *of*, made checkable instead of asserted.
#:
#: ``gpuwm/core/mynn_surface.py`` is FP32 arithmetic plus five NumPy elementary
#: functions: ``log``, ``exp``, ``sqrt``, ``arctan`` and ``**``.  Each row here
#: is one build's answer for the four that are not exact, over a fixed 4000
#: point float32 sweep, reduced to the first 16 hex digits of the SHA-256 of
#: the raw little-endian bytes.
#:
#: ``glibc`` is not a NumPy row.  It is
#: ``ctypes.CDLL("libm.so.6").{atanf,logf,expf,powf}`` under glibc 2.39 --
#: the functions gfortran linked when it produced these oracles -- recorded on
#: the two Linux hosts, which agree.  Reading down the ``arctan`` column is the
#: whole finding: NumPy 2.5.1 *is* glibc there and the older two builds are
#: not, so the four ``+2.5.1`` entries above were raised by the *more* faithful
#: build, not by a regression.  Reading down ``log``/``exp`` is the other half:
#: all three NumPy builds agree with each other and none agrees with glibc, so
#: those two are a constant term in every number above rather than a platform
#: axis.
GLIBC = "glibc"
FP32_FINGERPRINTS = {
    "glibc 2.39 -- the libm gfortran linked, via ctypes": {
        "arctan": "4a06e679c8a51e79", "log": "561cab742170f61d",
        "exp": "205ba7f723b5c78f", "pow2": "a95cf0d901f059b5",
        "pow0.33": "5d14a0a8bd99ffd2",
    },
    "NumPy 2.5.1 / CPython 3.12.13 / Ubuntu-22.04 (second 5090 host)": {
        "arctan": "4a06e679c8a51e79", "log": "511300744c384c9a",
        "exp": "6d216c097c764b80", "pow2": "a95cf0d901f059b5",
        "pow0.33": "5d14a0a8bd99ffd2",
    },
    "NumPy 2.4.3 / CPython 3.12.3 / WSL Ubuntu-24.04 (oracle builds)": {
        "arctan": "fa7ab2cff17a47ad", "log": "511300744c384c9a",
        "exp": "6d216c097c764b80", "pow2": "a95cf0d901f059b5",
        "pow0.33": "5d14a0a8bd99ffd2",
    },
    "NumPy 2.2.6 / CPython 3.13.7 / Windows 11 (pytest here)": {
        "arctan": "628753a5a564c404", "log": "511300744c384c9a",
        "exp": "6d216c097c764b80", "pow2": "6293278d9ba62405",
        "pow0.33": "1ee31ffeb21a6f67",
    },
}

INPUT_ALIASES = {
    "hfx": "hfx_input", "qfx": "qfx_input", "znt": "znt_input",
    "qsfc": "qsfc_input", "ust": "ust_input",
}
INPUT_NAMES = (
    "u1", "v1", "t1", "qv1", "p1", "rho1", "dz1",
    "u2", "v2", "dz2", "psfc", "tsk", "pblh", "mavail",
    "hfx", "qfx", "znt", "qsfc", "ust", "xland", "snowh",
)


def _rows(path: Path):
    with path.open(newline="", encoding="ascii") as stream:
        return list(csv.DictReader(stream))


def _fields(rows):
    skip = ("case", "itimestep", "isfflx")
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float32)
        for key in rows[0] if key not in skip
    }


def _oracle():
    rows = _rows(ORACLE)
    return rows, _fields(rows)


def _stage(path: Path, itimestep: int, isfflx: int | None = None):
    rows = [
        row for row in _rows(path)
        if int(row["itimestep"]) == itimestep
        and (isfflx is None or int(row["isfflx"]) == isfflx)
    ]
    assert tuple(row["case"] for row in rows) == WIDE_CASES
    return _fields(rows)


def _inputs(fields):
    return {name: fields[INPUT_ALIASES.get(name, name)] for name in INPUT_NAMES}


def _budget(table, name, cases, zeroed=()):
    """The measured per-column residue for ``name``, or zeros."""

    row = () if name in zeroed else table.get(name, ())
    if not row:
        return np.zeros(len(cases), dtype=np.int64)
    assert len(row) == len(cases), name
    return np.asarray(row, dtype=np.int64)


def _assert_ulp(actual, expected, name, budget, cases):
    """Fail when any column of ``name`` passes its measured ULP residue.

    ``budget`` is per column, so the message names the case that drifted
    instead of only the output: a field-wide number cannot say which of the
    ten columns moved, and that is the information a regression needs.
    """

    residue = fp32_ulp_distance(np.ravel(actual), np.ravel(expected))
    budget = np.broadcast_to(
        np.asarray(budget, dtype=np.int64), residue.shape
    )
    over = np.flatnonzero(residue > budget)
    if over.size:
        worst = int(over[np.argmax(residue[over] - budget[over])])
        raise AssertionError(
            f"{name}[{cases[worst]}]: {int(residue[worst])} ULP from the "
            f"unmodified WRF oracle exceeds the measured {int(budget[worst])} "
            f"({over.size} of {residue.size} columns over)"
        )


def _assert_parity(actual, fields, names, table, cases, zeroed=()):
    np.testing.assert_array_equal(actual["regime"], fields["regime"])
    for name in names:
        if name == "regime":
            continue
        _assert_ulp(
            actual[name], fields[name], name,
            _budget(table, name, cases, zeroed), cases,
        )


def test_default_surface_layer_matches_six_official_wrf_regimes():
    rows, fields = _oracle()
    inputs = _inputs(fields)
    actual = mynn_surface_layer_default(inputs)

    assert tuple(row["case"] for row in rows) == NARROW_CASES
    _assert_parity(
        actual, fields,
        tuple(name for name in actual if name != "znt"),
        NARROW_ULP, NARROW_CASES,
    )
    # This fixture predates the ZNT output column, so it can only pin the
    # land invariant: charnock_1955 (module_sf_mynn.F:657) runs over water
    # only, so land ZNT must come back exactly as it went in.
    land = fields["xland"] < 1.5
    np.testing.assert_array_equal(
        actual["znt"][land], fields["znt_input"][land]
    )
    assert np.all(actual["znt"][~land] != fields["znt_input"][~land])


@pytest.mark.parametrize(
    ("itimestep", "isfflx"), ((1, 1), (2, 1), (1, 0)),
)
def test_default_surface_layer_matches_widened_wrf_oracle(itimestep, isfflx):
    fields = _stage(WIDE_ORACLE, itimestep, isfflx)
    actual = mynn_surface_layer_default(
        _inputs(fields), itimestep=itimestep, isfflx=isfflx,
        mol=fields["mol_input"], ustm=fields["ustm_input"],
    )
    _assert_parity(
        actual, fields, tuple(actual), WIDE_ULP[(itimestep, 1)], WIDE_CASES,
        zeroed=ISFFLX0_ZEROED if isfflx == 0 else (),
    )


def test_the_isfflx0_stage_zeroes_thirteen_outputs_and_shares_the_rest():
    """What licenses reusing ``WIDE_ULP[(1, 1)]`` for the ISFFLX=0 stage.

    module_sf_mynn.F:1027-1044 makes ISFFLX<1 a pure post-processing branch:
    thirteen outputs become the constant 0 and nothing else changes.  Both
    halves are pinned here, on the fixture AND on the reference, so the reuse
    is a measured consequence of the branch rather than a borrowed budget.
    """

    off = _stage(WIDE_ORACLE, 1, 0)
    on = _stage(WIDE_ORACLE, 1, 1)
    assert len(ISFFLX0_ZEROED) == 13

    for name in ISFFLX0_ZEROED:
        np.testing.assert_array_equal(
            off[name], np.zeros_like(off[name]), err_msg=f"oracle {name}"
        )
        # ...and ISFFLX=1 is *not* zero there, or this proves nothing.
        assert np.any(on[name] != 0.0), name

    inputs = _inputs(off)
    kwargs = dict(mol=off["mol_input"], ustm=off["ustm_input"])
    diagnosed = mynn_surface_layer_default(
        inputs, itimestep=1, isfflx=1, **kwargs
    )
    prescribed = mynn_surface_layer_default(
        inputs, itimestep=1, isfflx=0, **kwargs
    )
    for name in diagnosed:
        if name in ISFFLX0_ZEROED:
            np.testing.assert_array_equal(
                prescribed[name], np.zeros_like(off[name]),
                err_msg=f"reference {name}",
            )
            continue
        np.testing.assert_array_equal(
            off[name], on[name], err_msg=f"oracle {name}"
        )
        np.testing.assert_array_equal(
            prescribed[name], diagnosed[name], err_msg=f"reference {name}"
        )


def test_widened_oracle_covers_the_branches_the_narrow_fixture_missed():
    step1 = _stage(WIDE_ORACLE, 1, 1)
    step2 = _stage(WIDE_ORACLE, 2, 1)

    assert set(step1["regime"]) | set(step2["regime"]) == {1.0, 2.0, 3.0, 4.0}
    assert step1["br"].max() > 0.2
    assert np.any(step2["mol_input"] != 0.0)

    land = step1["xland"] < 1.5
    assert np.any(step1["qsfc_input"][land] <= 0.0)

    za = 0.5 * step1["dz1"]
    za2 = step1["dz1"] + 0.5 * step1["dz2"]
    assert np.any((za <= 7.0) & (za2 > 7.0) & (za2 < 13.0))
    assert np.any((za <= 7.0) & ~((za2 > 7.0) & (za2 < 13.0)))
    assert np.any((za > 7.0) & (za < 13.0))
    assert np.any(za >= 13.0)

    # TH2 = THGB + 2*(TH1-THGB)/ZA is the bracketing fallback (:1142).
    thgb = step1["tsk"] * (100.0 / (step1["psfc"] / 1000.0)) ** (287.0 / 1004.5)
    th1 = step1["t1"] * (100.0 / (step1["p1"] / 1000.0)) ** (287.0 / 1004.5)
    fallback = thgb + 2.0 * (th1 - thgb) / za
    assert np.any(np.abs(step1["th2"] - fallback) < 1.0e-3)


def test_znt_evolves_over_water_and_is_carried_into_the_next_step():
    step1 = _stage(WIDE_ORACLE, 1, 1)
    step2 = _stage(WIDE_ORACLE, 2, 1)
    water = step1["xland"] > 1.5
    assert water.any()

    actual1 = mynn_surface_layer_default(
        _inputs(step1), itimestep=1,
        mol=step1["mol_input"], ustm=step1["ustm_input"],
    )
    # The oracle carries WRF's own updated ZNT into step 2; the reference has
    # to produce the same value or the persistence chain is broken.  Measured
    # at 0 ULP on every column, so the chain is pinned bitwise.
    _assert_ulp(
        actual1["znt"], step2["znt_input"], "carried znt", 0, WIDE_CASES
    )
    assert np.all(actual1["znt"][water] != step1["znt_input"][water])
    assert np.array_equal(actual1["znt"][~water], step1["znt_input"][~water])


def test_land_column_with_unset_qsfc_takes_the_wrf_land_height_scale():
    """module_sf_mynn.F:573 re-tests QSFC *after* the :533 update."""

    fields = _stage(WIDE_ORACLE, 1, 1)
    index = WIDE_CASES.index("land_qsfc_unset")
    assert fields["xland"][index] < 1.5
    assert fields["qsfc_input"][index] <= 0.0
    assert fields["wstar"][index] > 0.0

    actual = mynn_surface_layer_default(
        _inputs(fields), itimestep=1,
        mol=fields["mol_input"], ustm=fields["ustm_input"],
    )
    # WSTAR is bitwise on this fixture, so the discriminator below is being
    # compared against an exact reproduction of WRF, not a nearby number.
    _assert_ulp(
        actual["wstar"][index:index + 1], fields["wstar"][index:index + 1],
        "land_qsfc_unset wstar", 0, WIDE_CASES[index:index + 1],
    )
    # The stale-flag bug picks PBLH instead of MIN(1.5*PBLH,4000); that is a
    # >10% WSTAR error here, so the assertion above is a real discriminator.
    inputs = _inputs(fields)
    water_scale = np.float32(
        1.25 * (9.81 / inputs["tsk"][index] * inputs["pblh"][index]
                * (inputs["hfx"][index] / inputs["rho1"][index] / 1004.5
                   + 0.60777 * 306.6 * inputs["qfx"][index]
                   / inputs["rho1"][index])) ** 0.33
    )
    assert abs(water_scale - fields["wstar"][index]) > 0.1


@pytest.mark.parametrize("itimestep", (1, 2))
def test_sfclay_wrapper_seeding_matches_wrf(itimestep):
    fields = _stage(WRAPPER_ORACLE, itimestep)
    values = _inputs(fields)
    mol = fields["mol_input"]
    if itimestep == 1:
        seed = mynn_sfclay_first_step_state(
            fields["u1"], fields["v1"], fields["qv1"]
        )
        # The fixture enters from WRF's module_physics_init.F cold start, so
        # the seeding is observable rather than a no-op.
        assert not np.allclose(seed["ust"], fields["ust_input"])
        assert np.all(seed["mol"] == 0.0)
        assert np.all(seed["qstar"] == 0.0)
        values["ust"] = seed["ust"]
        values["qsfc"] = seed["qsfc"]
        mol = seed["mol"]
    actual = mynn_surface_layer_default(
        values, itimestep=itimestep, isfflx=1,
        mol=mol, ustm=fields["ustm_input"],
    )
    # SFCLAY_mynn keeps wstar/qstar as wrapper locals and never returns them.
    _assert_parity(
        actual, fields,
        tuple(name for name in actual if name not in ("wstar", "qstar")),
        WRAPPER_ULP[itimestep], WIDE_CASES,
    )


def test_every_table_row_is_the_right_width_and_carries_a_measurement():
    """A row of the wrong width, or of zeros, is a table that lost meaning.

    ``_budget`` asserts the width when it is used, but only for outputs a test
    actually compares; this covers every row in the file, and rejects an
    all-zero row -- which would mean the output is bitwise and the row should
    have been deleted rather than left as decoration.
    """

    tables = [(NARROW_ULP, NARROW_CASES)]
    tables += [(WIDE_ULP[key], WIDE_CASES) for key in WIDE_ULP]
    tables += [(WRAPPER_ULP[key], WIDE_CASES) for key in WRAPPER_ULP]
    assert len(tables) == 5
    for table, cases in tables:
        assert table
        for name, row in table.items():
            assert len(row) == len(cases), name
            assert all(isinstance(v, int) and v >= 0 for v in row), name
            assert any(row), f"{name} is all zero; delete the row"


def _fp32_fingerprint():
    """This build's answer for the four inexact float32 primitives."""

    probe = (
        np.arange(1, 4001, dtype=np.float32) * np.float32(0.001)
    ).astype(np.float32)

    def digest(values):
        return hashlib.sha256(
            np.asarray(values, dtype=np.float32).astype("<f4").tobytes()
        ).hexdigest()[:16]

    f = np.float32
    return {
        "arctan": digest([f(np.arctan(v)) for v in probe]),
        "log": digest([f(np.log(v)) for v in probe]),
        "exp": digest([f(np.exp(v)) for v in probe]),
        "pow2": digest([f(v ** f(2.0)) for v in probe]),
        "pow0.33": digest([f(v ** f(0.33)) for v in probe]),
    }


def test_the_ulp_tables_name_the_float32_builds_they_were_measured_on():
    """The union above is over these builds, and only these builds.

    Three claims, all of them checked rather than written down:

    1. ``log`` and ``exp`` are identical on every NumPy build recorded and
       none of them is glibc, so they contribute a *constant* to every number
       in these tables and are not what makes the tables a union.
    2. ``arctan`` is different on all three, and NumPy 2.5.1's is glibc's
       ``atanf`` exactly -- which is why 2.5.1 raised four entries: at those
       elements the older builds' ``atanf`` error cancelled downstream.
       ``**`` splits Linux from Windows the same way, via the platform
       ``powf``.
    3. The build running this test is one of the three the union covers.  A
       fourth is not a bug, but these budgets were never measured against it,
       and a passing gate would not mean they hold.
    """

    glibc = FP32_FINGERPRINTS[
        "glibc 2.39 -- the libm gfortran linked, via ctypes"
    ]
    numpy_rows = {
        name: row for name, row in FP32_FINGERPRINTS.items()
        if not name.startswith(GLIBC)
    }
    assert len(numpy_rows) == 3, sorted(numpy_rows)

    # (1) log/exp: one NumPy answer, and it is not glibc's.
    for name in ("log", "exp"):
        seen = {row[name] for row in numpy_rows.values()}
        assert len(seen) == 1, (
            f"float32 {name} is no longer one function across the recorded"
            f" builds ({sorted(seen)}); it has become a platform axis and"
            " every number in these tables is now suspect"
        )
        assert glibc[name] not in seen, (
            f"float32 {name} now agrees with glibc {name}f; the tables were"
            " measured while it did not, so re-measure -- this is a"
            " tightening, not a widening"
        )

    # (2) arctan/pow: the axes.  Three distinct arctans, exactly one of which
    # is glibc's; two distinct pows, split Linux/Windows.
    seen = {row["arctan"] for row in numpy_rows.values()}
    assert len(seen) == 3, (
        "float32 arctan is meant to be what separates all three recorded"
        f" builds, and now takes only {len(seen)} values"
    )
    on_glibc_atan = [
        name for name, row in numpy_rows.items()
        if row["arctan"] == glibc["arctan"]
    ]
    assert on_glibc_atan == [
        "NumPy 2.5.1 / CPython 3.12.13 / Ubuntu-22.04 (second 5090 host)"
    ], (
        "exactly one recorded build reproduces glibc atanf, and it is the one"
        f" that raised the four +2.5.1 entries; got {on_glibc_atan}"
    )
    on_glibc_pow = sorted(
        name for name, row in numpy_rows.items()
        if (row["pow2"], row["pow0.33"])
        == (glibc["pow2"], glibc["pow0.33"])
    )
    assert len(on_glibc_pow) == 2 and all(
        "Windows" not in name for name in on_glibc_pow
    ), (
        "float32 ** is the platform powf: glibc's on the two Linux builds and"
        f" the MSVC runtime's on Windows; got {on_glibc_pow}"
    )

    # (3) ...and this build is one of them.
    here = _fp32_fingerprint()
    matches = [name for name, row in numpy_rows.items() if row == here]
    assert len(matches) == 1, (
        "this NumPy's float32 primitives are a fourth build: "
        + ", ".join(
            f"{key}={here[key]}"
            for key in sorted(here)
            if here[key] not in {row[key] for row in numpy_rows.values()}
        )
        + f" (numpy {np.__version__}).  The ULP tables in this file and in"
        " tests/test_mynn_surface_coarse.py are the elementwise maximum over"
        f" {sorted(numpy_rows)} and were never measured here; re-measure"
        " before trusting a green run, and extend both tables and this"
        " fingerprint together."
    )


def _wide_validator():
    spec = importlib.util.spec_from_file_location(
        "_mynn_wide_oracle_validator", WIDE_VALIDATOR
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_wide_oracle_validator_gates_on_these_same_measurements():
    """One measurement, two consumers.

    ``tools/mynn_wrf461_oracle/validate_wide_oracle.py`` is what the oracle
    build runs, and it is a standalone script that cannot import a test
    module, so it carries its own copy of these tables.  Two copies of a
    measurement drift; this pins them equal instead of hoping.  It also pins
    the defect out: that script used to compute ``max_ulp``, print it, and
    then gate on ``np.allclose(rtol=3e-6, atol=1e-8)`` anyway.
    """

    module = _wide_validator()
    assert module.WIDE_ULP == WIDE_ULP
    assert module.WRAPPER_ULP == WRAPPER_ULP
    assert module.ISFFLX0_ZEROED == ISFFLX0_ZEROED
    assert module.EXPECTED_CASES == WIDE_CASES
    for gone in ("RTOL", "ATOL"):
        assert not hasattr(module, gone), (
            f"validate_wide_oracle.py still carries {gone}; it gates on"
            " measured ULP now"
        )


def test_surface_layer_rejects_shape_and_option_drift():
    _, fields = _oracle()
    inputs = _inputs(fields)
    with pytest.raises(ValueError, match="isfflx"):
        mynn_surface_layer_default(inputs, isfflx=2)
    bad = dict(inputs)
    bad["u1"] = bad["u1"][:-1]
    with pytest.raises(ValueError, match="equal-length"):
        mynn_surface_layer_default(bad)


def test_first_step_seed_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="one shape"):
        mynn_sfclay_first_step_state(
            np.zeros(3, np.float32), np.zeros(3, np.float32),
            np.zeros(2, np.float32),
        )


# ---------------------------------------------------------------------------
# psi_init's four lookup tables: pinned words, and the drift they replaced
# ---------------------------------------------------------------------------

#: Every NumPy elementary function that could reach the psi tables.
#: ``+ - * /`` are left alone: correctly rounded in binary32, and the same
#: function everywhere.
_FORBIDDEN_NUMPY_TRANSCENDENTALS = (
    "arctan", "arctan2", "log", "log2", "log10", "log1p", "exp", "exp2",
    "expm1", "power", "float_power", "sqrt", "cbrt",
)

#: Words 0, 500 and 1000 of each table, as raw FP32 bit patterns, so a
#: digest mismatch says *which* table moved.
#: The stable pair starts at NEGATIVE zero -- ``-6.1 * logf(1)`` -- which is
#: the word WRF's table carries and the word the CUDA kernel recomputes.
_PSI_TABLE_PROBES = {
    "psim_stable": (0x80000000, 0xC161143B, 0xC19238F9),
    "psih_stable": (0x80000000, 0xC1498947, 0xC180848C),
    "psim_unstable": (0x00000000, 0x4008D7AE, 0x402A252E),
    "psih_unstable": (0x00000000, 0x40457542, 0x4069F756),
}


def _numpy_built_psi_tables():
    """The four tables as ``mynn_surface.py`` built them before this lane:
    scalar ``np.arctan``/``np.log``/``**`` on ``np.float32``, verbatim from
    the revision this replaced (gpuwm/core/mynn_surface.py:84-145 at
    fd57224).  Kept here, not in the module, precisely so the module has no
    NumPy transcendental left on this path.
    """

    F = np.float32

    def psim_stable(zolf):
        zolf = F(zolf)
        return F(-F(6.1) * np.log(
            zolf + (F(1.0) + zolf ** F(2.5)) ** F(1.0 / 2.5)))

    def psih_stable(zolf):
        zolf = F(zolf)
        return F(-F(5.3) * np.log(
            zolf + (F(1.0) + zolf ** F(1.1)) ** F(1.0 / 1.1)))

    def psim_unstable(zolf):
        zolf = F(zolf)
        x = F((F(1.0) - F(16.0) * zolf) ** F(0.25))
        psimk = F(F(2.0) * np.log(F(0.5) * (F(1.0) + x))
                  + np.log(F(0.5) * (F(1.0) + x * x))
                  - F(2.0) * np.arctan(x) + F(2.0) * np.arctan(F(1.0)))
        ym = F((F(1.0) - F(10.0) * zolf) ** F(0.33))
        psimc = F(F(1.5) * np.log((ym * ym + ym + F(1.0)) / F(3.0))
                  - np.sqrt(F(3.0)) * np.arctan(
                      (F(2.0) * ym + F(1.0)) / np.sqrt(F(3.0)))
                  + F(4.0) * np.arctan(F(1.0)) / np.sqrt(F(3.0)))
        return F((psimk + zolf * zolf * psimc) / (F(1.0) + zolf * zolf))

    def psih_unstable(zolf):
        zolf = F(zolf)
        y = F((F(1.0) - F(16.0) * zolf) ** F(0.5))
        psihk = F(F(2.0) * np.log((F(1.0) + y) / F(2.0)))
        yh = F((F(1.0) - F(34.0) * zolf) ** F(0.33))
        psihc = F(F(1.5) * np.log((yh * yh + yh + F(1.0)) / F(3.0))
                  - np.sqrt(F(3.0)) * np.arctan(
                      (F(2.0) * yh + F(1.0)) / np.sqrt(F(3.0)))
                  + F(4.0) * np.arctan(F(1.0)) / np.sqrt(F(3.0)))
        return F((psihk + zolf * zolf * psihc) / (F(1.0) + zolf * zolf))

    return tuple(
        np.asarray([full(sign * F(n) * F(0.01)) for n in range(1001)],
                   dtype=np.float32)
        for full, sign in ((psim_stable, F(1.0)), (psih_stable, F(1.0)),
                           (psim_unstable, F(-1.0)), (psih_unstable, F(-1.0)))
    )


def test_the_psi_tables_are_the_pinned_glibc_words():
    """The four 1001-entry tables are data now, digested, not a side effect.

    Every word is a pure function of this repository's own glibc 2.39
    transcriptions, which are integer bit arithmetic, so this digest does not
    depend on the host NumPy or the platform ``powf`` -- measured equal on
    Windows NumPy 2.2.6, WSL NumPy 2.4.3 and Ubuntu NumPy 2.5.1.
    """

    tables = mynn_surface.psi_tables()
    assert len(tables) == 4
    for table in tables:
        assert table.shape == (1001,) and table.dtype == np.float32
    assert mynn_surface.psi_table_digest() == mynn_surface.PSI_TABLE_SHA256
    words = dict(zip(_PSI_TABLE_PROBES, tables))
    for name, probes in _PSI_TABLE_PROBES.items():
        got = words[name].view(np.uint32)
        assert tuple(int(got[i]) for i in (0, 500, 1000)) == probes, name


def test_the_psi_tables_replaced_words_numpy_actually_got_wrong():
    """The failing form of the pin above, and the size of the defect.

    ``mynn_surface.py`` used to build these tables with scalar ``np.arctan``,
    ``np.log`` and ``**`` on ``np.float32`` at import time.  This rebuilds
    them that way and shows they are NOT the glibc words gfortran linked, so
    the digest is guarding a real difference rather than restating one
    computation twice.
    """

    old = _numpy_built_psi_tables()
    new = mynn_surface.psi_tables()
    joined = hashlib.sha256(b"".join(
        np.ascontiguousarray(t).tobytes() for t in old)).hexdigest()
    assert joined != mynn_surface.PSI_TABLE_SHA256, (
        "the NumPy-built tables now digest to the pinned words; either the "
        "host NumPy became glibc-exact at every argument these tables use "
        "-- re-measure and say so -- or the module stopped using "
        "noahmp_libm"
    )
    residue = np.concatenate([
        fp32_ulp_distance(a, b) for a, b in zip(old, new)
    ])
    assert residue.size == 4004
    differing = int(np.count_nonzero(residue))
    # Measured on Windows NumPy 2.2.6 / CPython 3.13.7: 364 of 4004 words,
    # worst 32 ULP in _PSIM_UNSTAB where psimk cancels near zolf = 0.  The
    # count is host-dependent (it is NumPy's error, not ours), so this
    # asserts the defect is present and large, not its exact size.
    assert 100 <= differing <= 1200, differing
    assert 8 <= int(residue.max()) <= 4096, int(residue.max())


def test_building_the_psi_tables_calls_no_numpy_transcendental(monkeypatch):
    """Negative control: break every NumPy elementary function and rebuild.

    If a future edit reaches for ``np.arctan`` instead of
    ``gpuwm.core.noahmp_libm.atanf``, this fails at once instead of the next
    time somebody upgrades NumPy.
    """

    called: list[str] = []

    def forbid(name):
        def trap(*args, **kwargs):
            called.append(name)
            raise AssertionError(
                f"the psi tables called np.{name}; WRF's psi_init is binary32 "
                "and gfortran lowers it to glibc atanf/logf/powf.  Use "
                "gpuwm.core.noahmp_libm."
            )
        return trap

    for name in _FORBIDDEN_NUMPY_TRANSCENDENTALS:
        monkeypatch.setattr(np, name, forbid(name), raising=True)
    # The trap is live: prove it fires before trusting that it did not.
    with pytest.raises(AssertionError, match=r"called np\.arctan"):
        np.arctan(np.float32(1.0))
    called.clear()

    mynn_surface.psi_tables.cache_clear()
    try:
        assert mynn_surface.psi_table_digest() == (
            mynn_surface.PSI_TABLE_SHA256)
    finally:
        mynn_surface.psi_tables.cache_clear()
    assert called == []
    # And the pre-routing construction does trip it, so the control is not
    # vacuous.
    with pytest.raises(AssertionError, match=r"called np\.log"):
        _numpy_built_psi_tables()


def test_the_psi_tables_are_not_built_at_import():
    """An import-time table cannot be regenerated, diffed or pinned.

    The coupled MYNN runtime imports this module only for
    ``ISFTCFLX_DEFINED``, so building 4004 words through the Python
    transcriptions at import is 33 ms of measured work nothing on that path
    wants.  A fresh interpreter must leave the cache cold.
    """

    probe = (
        "import gpuwm.core.mynn_surface as m;"
        "print(m.psi_tables.cache_info().currsize)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True,
        cwd=str(Path(__file__).parents[1]), check=True,
        env={**os.environ, "GPUWM_NO_LOCAL_GPU": "1"},
    )
    assert out.stdout.strip() == "0", out.stdout + out.stderr
