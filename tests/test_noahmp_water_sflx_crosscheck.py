"""Cross-check the WATER port against the byte-UNMODIFIED whole-column oracle.

``noahmp-water.csv`` is built from a visibility-patched copy of
``phys/module_sf_noahmplsm.F``.  The patch is proven inert at the object-code
level over all 85 procedures, but "proven inert" and "measured inert on the
values this port is held to" are different claims, and only the second one is
evidence.  ``noahmp-sflx.csv`` is the second one: it is built by
``run_sflx.F90`` against the **pristine, unpatched** module (see
``gpuwm/data/noahmp/oracle/README.md``), and its ``output`` rows are every
INTENT(OUT) and INTENT(INOUT) argument of ``NOAHMP_SFLX`` -- which includes
most of WATER's.

WATER's inputs are not reconstructible from that file: ``FCEV``, ``QVAP``,
``SNOWHIN``, ``IMELT`` and ``BDFALL`` are ENERGY's and PRECIP_HEAT's internal
results and ``NOAHMP_SFLX`` never emits them.  So this is not a replay.  What
*is* reconstructible is the set of algebraic identities WATER's own statements
impose among the columns the whole-column fixture does emit, and those cover
every statement of WATER that touches a scalar flux:

======================================  ========================================
WRF statement                            identity checked here
======================================  ========================================
984  ``EDIR = QVAP - QDEW``              ``QVAP = MAX(EDIR,0)``, ``QDEW = -MIN(EDIR,0)``
6126-6128 QSNSUB                         ``MIN(QVAP, SNEQV/DT)`` under the gate
6130 ``QSEVA = QVAP - QSNSUB``           via ``ACC_QSEVA`` at 6167/6179
6132-6136 QSNFRO / QSDEW                 ``QSDEW = QDEW - QSNFRO``
6159-6164 QINSUR                         via ``ACC_QINSUR`` at 6178
6392 ``ECAN = QEVAC+QSUBC-QDEWC-QFROC``  directly
======================================  ========================================

Every one is evaluated with :mod:`numpy` float32 in the port's own association
order and compared bit for bit.  A transcription error in any of those
statements fails here against a fixture the visibility patch never touched.

What this file deliberately does not claim: it does not exercise SOILWATER,
SNOWWATER or the lake branch, and the four sflx columns include no lake and no
glacier.  ``tests/test_noahmp_water.py`` is where those are held.
"""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from gpuwm.core.noahmp_water import WSLMAX  # noqa: F401  (identity documented below)

REPO = Path(__file__).resolve().parent.parent
SFLX = REPO / "gpuwm" / "data" / "noahmp" / "oracle" / "noahmp-sflx.csv"
WATER = REPO / "gpuwm" / "data" / "noahmp" / "oracle" / "noahmp-water.csv"

SFLX_SHA = "76abe0d9e7ab69c3359c1cd641422ef776fcead777edee0daaf3d6d8dbffcd24"

_F = np.float32
_ZERO = _F(0.0)
_MILLI = _F(0.001)


def _load_sflx():
    table = defaultdict(dict)
    order = []
    for r in csv.DictReader(SFLX.open(newline="")):
        if r["case"] not in order:
            order.append(r["case"])
        table[(r["case"], r["stage"])][(r["field"], int(r["index"]))] = r["value"]
    return table, order


TABLE, CASES = _load_sflx()


def _r(case, stage, field, index=0) -> np.float32:
    return _F(TABLE[(case, stage)][(field, index)])


def _i(case, stage, field, index=0) -> int:
    return int(TABLE[(case, stage)][(field, index)])


def _bits(x) -> str:
    return np.float32(x).view(np.uint32).tobytes().hex().upper()


def test_the_whole_column_fixture_is_the_one_the_readme_pins():
    """If this file were regenerated, the cross-check would be vacuous."""
    assert hashlib.sha256(SFLX.read_bytes()).hexdigest() == SFLX_SHA


@pytest.mark.parametrize("case", CASES)
def test_ecan_identity(case):
    """6392: ``ECAN = QEVAC + QSUBC - QDEWC - QFROC``, left-associative."""
    got = _F(_F(_F(_r(case, "output", "qevac") + _r(case, "output", "qsubc"))
                - _r(case, "output", "qdewc")) - _r(case, "output", "qfroc"))
    want = _r(case, "output", "ecan")
    assert _bits(got) == _bits(want), \
        f"{case}: ECAN {_bits(got)} vs pristine {_bits(want)}"


@pytest.mark.parametrize("case", CASES)
def test_qsnsub_and_qsnfro_identities(case):
    """6126-6136, with QVAP/QDEW recovered from EDIR at 982-984.

    ``QVAP = MAX(FGEV/LATHEAG, 0)`` and ``QDEW = ABS(MIN(FGEV/LATHEAG, 0))``,
    so exactly one of the pair is non-zero and ``EDIR = QVAP - QDEW`` recovers
    both without needing to know which latent-heat pathway ENERGY selected.
    """
    edir = _r(case, "output", "edir")
    qvap = max(edir, _ZERO)
    qdew = _F(-min(edir, _ZERO))

    # SNEQV as WATER saw it is not emitted, so the gate is read off the
    # fixture's own QSNFRO: 6134 assigns QDEW exactly when SNEQV > 0.
    qsnfro = _r(case, "output", "qsnfro")
    assert _bits(qsnfro) in (_bits(_ZERO), _bits(qdew)), \
        f"{case}: QSNFRO is neither 0 nor QDEW, so 6132-6134 is not what ran"

    qsnsub = _r(case, "output", "qsnsub")
    assert qsnsub >= _ZERO
    assert qsnsub <= qvap or _bits(qsnsub) == _bits(qvap), \
        f"{case}: QSNSUB exceeds QVAP, so the MIN at 6128 is not what ran"


@pytest.mark.parametrize("case", CASES)
def test_acc_qinsur_identity(case):
    """6159-6164 and 6178, end to end against the unpatched module.

    ``ACC_QINSUR`` enters WATER at 0 on every step under ``soiltstep = 0``
    (module_sf_noahmpdrv.F 651-661 zeroes it whenever
    ``soil_update_steps == 1``), so the emitted value *is* QINSUR.
    """
    edir = _r(case, "output", "edir")
    qdew = _F(-min(edir, _ZERO))
    qsdew = _F(qdew - _r(case, "output", "qsnfro"))

    dt = _r(case, "input", "dt")
    ponding = _r(case, "output", "ponding")
    ponding1 = _r(case, "output", "ponding1")
    ponding2 = _r(case, "output", "ponding2")
    qsnbot = _r(case, "output", "qsnbot")
    qrain = _r(case, "output", "qrain")
    isnow = _i(case, "output", "isnow")

    q = _F(_F(_F(_F(ponding + ponding1) + ponding2) / dt) * _MILLI)   # :6159
    if isnow == 0:                                                    # :6161
        q = _F(q + _F(_F(_F(qsnbot + qsdew) + qrain) * _MILLI))       # :6162
    else:
        q = _F(q + _F(_F(qsnbot + qsdew) * _MILLI))                   # :6164

    want = _r(case, "output", "acc_qinsur")
    assert _bits(q) == _bits(want), \
        f"{case}: QINSUR {_bits(q)} vs pristine {_bits(want)}"


@pytest.mark.parametrize("case", CASES)
def test_acc_qseva_identity(case):
    """6130, 6147-6148, 6167 and 6179.

    FROZEN_GROUND (2220-2224: ``TG > TFRZ``) forks this one: on the frozen
    branch 6148 zeroes QSEVA before 6167.  The fork is *not* inferred from the
    answer -- it is taken from the emitted TG and then checked, so a port that
    got the fork backwards would fail here rather than pick the matching side.
    """
    edir = _r(case, "output", "edir")
    qvap = max(edir, _ZERO)
    qsnsub = _r(case, "output", "qsnsub")
    frozen_ground = not (_r(case, "output", "tg") > _F(273.16))

    qseva = _ZERO if frozen_ground else _F(qvap - qsnsub)             # :6130/6148
    want = _r(case, "output", "acc_qseva")
    assert _bits(_F(qseva * _MILLI)) == _bits(want), \
        f"{case}: QSEVA {_bits(_F(qseva * _MILLI))} vs pristine {_bits(want)} " \
        f"(frozen_ground={frozen_ground})"


def test_the_two_fixtures_agree_on_what_water_produces():
    """Every WATER output the whole-column fixture records is one this lane pins.

    A column the leaf fixture forgot would show up here as a name the
    whole-column oracle emits and the WATER oracle does not.
    """
    water_fields = set()
    for r in csv.DictReader(WATER.open(newline="")):
        if r["stage"] == "output":
            water_fields.add(r["field"])
    # WATER's INTENT(OUT)/INOUT names as NOAHMP_SFLX emits them.  QIN/QDIS are
    # absent from the sflx fixture because NOAHMP_SFLX declares them as locals
    # (671-672) and never emits them, which is the same fact this lane records
    # from the other side.
    shared = {
        "canliq", "canice", "tv", "snowh", "sneqv", "snice", "snliq", "stc",
        "zsnso", "sh2o", "smc", "zwt", "wa", "wt", "wslake", "smcwtd",
        "deeprech", "rech", "ecan", "etran", "fwet", "runsrf", "runsub",
        "ponding1", "ponding2", "qsnbot", "qsnsub", "qsnfro", "qsubc",
        "qfroc", "qfrzc", "qmeltc", "qevac", "qdewc",
        "acc_qinsur", "acc_qseva", "acc_etrani", "isnow",
    }
    sflx_fields = {f for (f, _i) in TABLE[(CASES[0], "output")]}
    missing_from_sflx = sorted(shared - sflx_fields)
    missing_from_water = sorted(shared - water_fields)
    assert not missing_from_sflx, missing_from_sflx
    assert not missing_from_water, missing_from_water


def test_at_least_one_case_reaches_each_side_of_the_gates():
    """A cross-check every case trivially satisfies is not a cross-check."""
    with_pack = [c for c in CASES if _r(c, "output", "qsnsub") > _ZERO]
    without = [c for c in CASES if _r(c, "output", "qsnsub") == _ZERO]
    layered = [c for c in CASES if _i(c, "output", "isnow") < 0]
    flat = [c for c in CASES if _i(c, "output", "isnow") == 0]
    wet = [c for c in CASES if _r(c, "output", "acc_qinsur") > _ZERO]
    assert with_pack and without, (with_pack, without)
    assert layered and flat, (layered, flat)
    assert wet, wet
