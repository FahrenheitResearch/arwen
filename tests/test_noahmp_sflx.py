"""Bitwise gate for the Noah-MP NOAHMP_SFLX composition.

The whole-column fixture ``gpuwm/data/noahmp/oracle/noahmp-sflx.csv`` is the
only artefact in this tree built by calling the **byte-unmodified**
``phys/module_sf_noahmplsm.F``: NOAHMP_SFLX is one of the module's two public
entry points, so its harness needs no visibility lift at all.  Every other
Noah-MP fixture here is produced from the accessibility-patched copy.  Holding
the composed port to it is therefore the one check that can catch drift between
the patched-visibility leaf path and the pristine column.

This module makes three independent claims and keeps them separable:

1. ``test_energy_boundary_bitwise`` -- everything NOAHMP_SFLX computes *before*
   ENERGY (ATM, DZSNSO, TROOT, PHENOLOGY, FVEG, PRECIP_HEAT, and the argument
   marshalling itself) reproduces ENERGY's own fixture, ``noahmp-energy.csv``,
   at ``max_ulp 0`` on its ``in`` and ``seed`` roles.
2. ``test_replayed_column_bitwise`` -- everything *after* ENERGY (SICE,
   QVAP/QDEW/EDIR, WATER, ERROR, the urban override, the snow clamp and the
   albedo sentinel) reproduces ``noahmp-sflx.csv`` at ``max_ulp 0`` with
   ENERGY's outputs **replayed from its fixture** rather than computed.
3. ``test_whole_column_bitwise`` -- and the same holds with the real
   :func:`gpuwm.core.noahmp_energy.energy` in the loop.

Keeping (2) and (3) apart is the point: (2) cannot be passed by a pair of
compensating errors either side of ENERGY, and a future ENERGY regression shows
up as (3) failing while (2) still passes.

A fourth claim has its own fixture, because the whole column cannot carry it:
``test_error_bitwise`` holds ERROR to ``noahmp-sflx-error.csv``, which calls
ERROR directly over sixteen cases.  ERRWAT is a NOAHMP_SFLX local that reaches
no output, the ``IST /= 1`` lake branch appears in no column here, and the two
irrigation terms are identically zero under ``opt_irr = 0``: none of the three
is observable from ``noahmp-sflx.csv`` at all.

``test_output_inventory_is_covered`` refuses to let the comparison go vacuous:
every one of the 171 values the fixture records per column is either compared
bitwise against the port or is on the pass-through list, and each pass-through
is *proved* pass-through out of the fixture itself rather than asserted.
"""

from __future__ import annotations

import csv
import hashlib
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from gpuwm.core.noahmp import load_noahmp_parameters, transfer_mp_parameters
from gpuwm.core.noahmp_energy import EnergyState
from gpuwm.core.noahmp_snow import SnowColumn
from gpuwm.core.noahmp_sflx import (EnergyBalanceError, NIGHT_ALBEDO,
                                    ShortwaveBalanceError, SflxParameters,
                                    WaterBalanceError, error, sflx, sflx_post,
                                    sflx_pre)

REPO = Path(__file__).resolve().parent.parent
ORACLE = REPO / "gpuwm" / "data" / "noahmp" / "oracle"
SFLX_CSV = ORACLE / "noahmp-sflx.csv"
ENERGY_CSV = ORACLE / "noahmp-energy.csv"

#: A fixture that can be regenerated silently is not a gate.
PINNED = {
    "noahmp-sflx.csv":
        "76abe0d9e7ab69c3359c1cd641422ef776fcead777edee0daaf3d6d8dbffcd24",
}

NSOIL = 4
NSNOW = 3
CASES = ("veg_warm_day_dry", "veg_warm_night_rain",
         "snowpack_frozen_soil", "bare_thin_snow_melt")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _bits(x) -> str:
    return struct.pack(">f", np.float32(x)).hex().upper()


def _from_hex(h: str) -> np.float32:
    return np.float32(struct.unpack(">f", bytes.fromhex(h))[0])


def _int_from_hex(h: str) -> int:
    v = int(h, 16)
    return v - (1 << 32) if v >= (1 << 31) else v


def _load_sflx():
    """``noahmp-sflx.csv`` is decimal, not hex; ``test_fixture_round_trips``
    proves the encoding is lossless before anything else reads it."""
    table = defaultdict(dict)
    for row in csv.DictReader(SFLX_CSV.open(newline="")):
        key = (row["field"], int(row["index"]))
        table[(row["case"], row["stage"])][key] = np.float32(float(row["value"]))
    return table


def _load_energy():
    table = defaultdict(dict)
    for row in csv.DictReader(ENERGY_CSV.open(newline="")):
        table[(row["case"], row["role"])][row["name"]] = row["hex"].strip()
    return table


SF = _load_sflx()
EN = _load_energy()
BUNDLE = load_noahmp_parameters()


def _params(x) -> SflxParameters:
    p = transfer_mp_parameters(
        BUNDLE, vegtype=int(x[("vegtyp", 0)]),
        soiltype=[int(x[("soiltype", 0)])] * NSOIL,
        slopetype=int(x[("slopetype", 0)]),
        soilcolor=int(x[("soilcolor", 0)]),
        croptype=int(x[("croptype", 0)]))
    return SflxParameters.from_transferred(p, nsoil=NSOIL)


def _vec(d, name, lo, hi) -> np.ndarray:
    return np.asarray([d[(name, k)] for k in range(lo, hi + 1)],
                      dtype=np.float32)


def _column(x) -> SnowColumn:
    return SnowColumn(
        nsnow=NSNOW, nsoil=NSOIL, isnow=int(x[("isnow", 0)]),
        snowh=x[("snowh", 0)], sneqv=x[("sneqv", 0)],
        snice=_vec(x, "snice", -NSNOW + 1, 0),
        snliq=_vec(x, "snliq", -NSNOW + 1, 0),
        stc=_vec(x, "stc", -NSNOW + 1, NSOIL),
        zsnso=_vec(x, "zsnso", -NSNOW + 1, NSOIL),
        sh2o=_vec(x, "sh2o", 1, NSOIL))


#: NOAHMP_SFLX's scalar forcing and entry state, by the fixture's own names.
_FORCING = ("lat yearlen julian cosz dt dx dz8w shdmax vegtyp ice ist croptype "
            "sfctmp sfcprs psfc uu vv q2 qc soldn lwdn prcpconv prcpnonc "
            "prcpshcv prcpsnow prcpgrpl prcphail tbot co2air o2air foln zlvl "
            "albold sneqvo tah eah canliq canice tv tg qsfc lai sai cm ch "
            "tauss pgs").split()
_INTEGER_FORCING = {"yearlen", "vegtyp", "ice", "ist", "croptype", "pgs"}


def _forcing(x, **override) -> dict:
    kw = {f: (int(x[(f, 0)]) if f in _INTEGER_FORCING else x[(f, 0)])
          for f in _FORCING}
    kw["zsoil"] = _vec(x, "zsoil", 1, NSOIL)
    kw.update(override)
    return kw


def _run_pre(case, **override):
    x = SF[(case, "input")]
    p = _params(x)
    col = _column(x)
    smc = _vec(x, "smc", 1, NSOIL)
    pre = sflx_pre(p, col, smc, wa=x[("wa", 0)], acc_ssoil=np.float32(0.0),
                   **_forcing(x, **override))
    return p, col, smc, pre, x


def _run_post(case, p, col, pre, en, x):
    return sflx_post(
        p, col, pre, en, dt=x[("dt", 0)], ist=int(x[("ist", 0)]),
        zsoil=_vec(x, "zsoil", 1, NSOIL),
        ficeold=_vec(x, "ficeold", -NSNOW + 1, 0),
        wa=x[("wa", 0)], wslake=x[("wslake", 0)],
        acc_qinsur=np.float32(0.0), acc_qseva=np.float32(0.0),
        acc_etrani=np.zeros(NSOIL, dtype=np.float32),
        acc_dwater=np.float32(0.0), acc_prcp=np.float32(0.0),
        acc_ecan=np.float32(0.0), acc_etran=np.float32(0.0),
        acc_edir=np.float32(0.0))


def _run_whole(case, **override):
    x = SF[(case, "input")]
    p = _params(x)
    col = _column(x)
    r = sflx(p, col, _vec(x, "smc", 1, NSOIL),
             ficeold=_vec(x, "ficeold", -NSNOW + 1, 0),
             wa=x[("wa", 0)], wslake=x[("wslake", 0)],
             **_forcing(x, **override))
    return col, r


# ---------------------------------------------------------------------------
# ENERGY replay, straight out of the pristine fixture
# ---------------------------------------------------------------------------

_E_SCALARS = (
    "z0wrf t2m fsno sav sag qmelt fsa fsr taux tauy fira fsh fcev fgev fctr "
    "trad psn apar ssoil btran ponding ts latheav latheag tv tg snowh eah tah "
    "sneqvo sneqv albold cm ch tauss laisun laisha rb qsfc t2mv t2mb fsrv "
    "fsrg rssun rssha bgap wgap tgv tgb q1 q2v q2b q2e chv chb emissi pah "
    "canhs shg shc shb evg evb ghv ghb irg irc irb tr evc chleaf chuc chv2 "
    "chb2 eflxb acc_ssoil").split()
_E_ARRAYS = {"stc": (-NSNOW + 1, NSOIL), "sh2o": (1, NSOIL),
             "smc": (1, NSOIL), "snice": (-NSNOW + 1, 0),
             "snliq": (-NSNOW + 1, 0), "albsnd": (1, 2), "albsni": (1, 2),
             "hcpct": (-NSNOW + 1, NSOIL), "btrani": (1, NSOIL),
             "imelt": (-NSNOW + 1, NSOIL)}


def _key(name: str, index: int) -> str:
    return f"{name}_{'+' if index >= 0 else '-'}{abs(index)}"


def _replayed_energy(case) -> EnergyState:
    """ENERGY's exit state for ``case``, read out of ``noahmp-energy.csv``.

    ``out`` and ``undef`` are merged: the ENERGY lane files a slot under
    ``undef`` when WRF leaves it genuinely unwritten on that column's path and
    pins it at the defined ``0.0``, which is the value this composition needs
    and the value :mod:`gpuwm.core.noahmp_energy` produces.
    """
    d = dict(EN[(case, "out")])
    d.update(EN[(case, "undef")])
    kw = {f: _from_hex(d[f.upper()]) for f in _E_SCALARS}
    for name, (lo, hi) in _E_ARRAYS.items():
        raw = [d[_key(name.upper(), k)] for k in range(lo, hi + 1)]
        kw[name] = (tuple(_int_from_hex(v) for v in raw) if name == "imelt"
                    else tuple(_from_hex(v) for v in raw))
    kw["frozen_canopy"] = bool(int(d["FROZEN_CANOPY"], 16))
    kw["frozen_ground"] = bool(int(d["FROZEN_GROUND"], 16))
    return EnergyState(**kw)


# ---------------------------------------------------------------------------
# what the composition claims, and what the fixture carries
# ---------------------------------------------------------------------------

#: Everything the port returns as a scalar, keyed as ``noahmp-sflx.csv`` keys
#: it.  Every name here is compared bitwise.
RESULT_SCALARS = (
    "z0wrf fsa fsr fira fsh ssoil fcev fgev fctr ecan etran edir trad tgb tgv "
    "t2mv t2mb q2v q2b runsrf runsub apar psn sav sag fsno nee gpp npp fveg "
    "albedo qsnbot ponding ponding1 ponding2 rssun rssha bgap wgap chv chb "
    "emissi shg shc shb evg evb ghv ghb irg irc irb tr evc chleaf chuc chv2 "
    "chb2 fpice pahv pahg pahb pah laisun laisha rb qints qintr qdrips qdripr "
    "qthros qthror qsnsub qsnfro qsubc qfroc qfrzc qmeltc qevac qdewc qmelt "
    "rain snow eflxb canhs albold sneqvo tah eah fwet canliq canice tv tg "
    "qsfc qsnow qrain wslake lai sai cm ch tauss acc_ssoil acc_qinsur "
    "acc_qseva acc_dwater acc_prcp acc_ecan acc_etran acc_edir").split()
RESULT_ARRAYS = {"albsnd": (1, 2), "albsni": (1, 2), "smc": (1, NSOIL),
                 "acc_etrani": (1, NSOIL), "hcpct": (-NSNOW + 1, NSOIL)}
COLUMN_ARRAYS = {"sh2o": (1, NSOIL), "stc": (-NSNOW + 1, NSOIL),
                 "zsnso": (-NSNOW + 1, NSOIL), "snice": (-NSNOW + 1, 0),
                 "snliq": (-NSNOW + 1, 0)}

#: The fourteen outputs no live statement writes under this option identity:
#: DVEG = 4 kills CARBON (the six pools), OPT_CROP = 0 kills the crop stage,
#: OPT_RUN = 3 kills GROUNDWATER and SHALLOWWATERTABLE (ZWT, WA, WT, SMCWTD,
#: DEEPRECH, RECH), and nothing in the routine assigns ZLVL.
PASS_THROUGH = ("lfmass rtmass stmass wood stblcp fastcp pgs "
                "zwt wa wt smcwtd deeprech rech zlvl").split()


def _compare(case, col, r) -> tuple[int, list]:
    want = SF[(case, "output")]
    bad, n = [], 0
    for name in RESULT_SCALARS:
        n += 1
        got, exp = _bits(getattr(r, name)), _bits(want[(name, 0)])
        if got != exp:
            bad.append((name, got, exp))
    for name, (lo, hi) in RESULT_ARRAYS.items():
        arr = getattr(r, name)
        for j, k in enumerate(range(lo, hi + 1)):
            if (name, k) not in want:
                continue
            n += 1
            got, exp = _bits(arr[j]), _bits(want[(name, k)])
            if got != exp:
                bad.append((f"{name}({k})", got, exp))
    for name, (lo, hi) in COLUMN_ARRAYS.items():
        arr = getattr(col, name)
        for j, k in enumerate(range(lo, hi + 1)):
            if (name, k) not in want:
                continue
            n += 1
            got, exp = _bits(arr[j]), _bits(want[(name, k)])
            if got != exp:
                bad.append((f"{name}({k})", got, exp))
    n += 1
    if int(col.isnow) != int(want[("isnow", 0)]):
        bad.append(("isnow", int(col.isnow), int(want[("isnow", 0)])))
    for name, value in (("snowh", col.snowh), ("sneqv", col.sneqv)):
        n += 1
        got, exp = _bits(value), _bits(want[(name, 0)])
        if got != exp:
            bad.append((name, got, exp))
    return n, bad


# ---------------------------------------------------------------------------
# 0. the fixture itself
# ---------------------------------------------------------------------------

def test_fixture_is_byte_pinned():
    for name, digest in PINNED.items():
        got = hashlib.sha256((ORACLE / name).read_bytes()).hexdigest()
        assert got == digest, f"{name} changed under the port: {got}"


def test_fixture_round_trips():
    """``run_sflx.F90`` writes ``g0``, not hex.  Every comparison in this file
    assumes that decimal is lossless for binary32; prove it rather than
    inherit it."""
    checked = 0
    for row in csv.DictReader(SFLX_CSV.open(newline="")):
        text = row["value"].strip()
        if "." not in text and "E" not in text:
            continue                     # an INTEGER field
        value = np.float32(float(text))
        assert np.float32(float(f"{float(value):.9g}")) == value, text
        checked += 1
    assert checked > 1100


def test_case_inventory():
    order = tuple(dict.fromkeys(
        row["case"] for row in csv.DictReader(SFLX_CSV.open(newline=""))))
    assert order == CASES
    for case in CASES:
        assert EN[(case, "out")], f"noahmp-energy.csv lacks {case}"


# ---------------------------------------------------------------------------
# 1. the ENERGY argument boundary
# ---------------------------------------------------------------------------

_CALL_INTS = {"ice": "ICE", "ist": "IST", "isnow": "ISNOW"}
_CALL_SCALARS = {
    "dt": "DT", "rhoair": "RHOAIR", "sfcprs": "SFCPRS", "qair": "QAIR",
    "sfctmp": "SFCTMP", "thair": "THAIR", "lwdn": "LWDN", "uu": "UU",
    "vv": "VV", "zref": "ZREF", "co2air": "CO2AIR", "o2air": "O2AIR",
    "cosz": "COSZ", "igs": "IGS", "eair": "EAIR", "tbot": "TBOT",
    "elai": "ELAI", "esai": "ESAI", "fwet": "FWET", "foln": "FOLN",
    "fveg": "FVEG", "pahv": "PAHV", "pahg": "PAHG", "pahb": "PAHB",
    "qsnow": "QSNOW", "lat": "LAT", "canliq": "CANLIQ", "canice": "CANICE",
    "tv": "TV", "tg": "TG", "snowh": "SNOWH", "eah": "EAH", "tah": "TAH",
    "sneqvo": "SNEQVO", "sneqv": "SNEQV", "albold": "ALBOLD", "cm": "CM",
    "ch": "CH", "dx": "DX", "dz8w": "DZ8W", "q2": "Q2", "tauss": "TAUSS",
    "laisun": "LAISUN", "laisha": "LAISHA", "rb": "RB", "qc": "QC",
    "qsfc": "QSFC", "psfc": "PSFC", "acc_ssoil": "ACC_SSOIL",
    "julian": "JULIAN", "prcp": "PRCP", "fb": "FB"}
#: ``SWDOWN`` is an ENERGY argument but ``noahmp-energy.csv`` files it under
#: ``seed``, because ATM produces it rather than the ENERGY harness.
_CALL_FROM_SEED = {"swdown": "SWDOWN"}
_CALL_ARRAYS = {"solad": ("SOLAD", 1, 2), "solai": ("SOLAI", 1, 2),
                "zsnso": ("ZSNSO", -NSNOW + 1, NSOIL),
                "dzsnso": ("DZSNSO", -NSNOW + 1, NSOIL),
                "stc": ("STC", -NSNOW + 1, NSOIL),
                "zsoil": ("ZSOIL", 1, NSOIL), "sh2o": ("SH2O", 1, NSOIL),
                "smc": ("SMC", 1, NSOIL), "snice": ("SNICE", -NSNOW + 1, 0),
                "snliq": ("SNLIQ", -NSNOW + 1, 0)}
#: The pre-ENERGY locals ``noahmp-energy.csv`` records under role ``seed``.
_SEED = {"troot": "TROOT", "bdfall": "BDFALL", "rain": "RAIN",
         "snow": "SNOWFALL", "fp": "FP", "fpice": "FPICE", "qrain": "QRAIN",
         "snowhin": "SNOWHIN", "cmc": "CMC", "qintr": "QINTR",
         "qdripr": "QDRIPR", "qthror": "QTHROR", "qints": "QINTS",
         "qdrips": "QDRIPS", "qthros": "QTHROS", "swdown": "SWDOWN",
         "qprecc": "QPRECC", "qprecl": "QPRECL"}


def _compare_boundary(case, pre) -> tuple[int, list]:
    ein, seed = EN[(case, "in")], EN[(case, "seed")]
    bad, n = [], 0
    for field, name in _CALL_INTS.items():
        n += 1
        got = f"{int(getattr(pre.call, field)) & 0xFFFFFFFF:08X}"
        if got != ein[name]:
            bad.append((f"in:{name}", got, ein[name]))
    for field, name in _CALL_SCALARS.items():
        n += 1
        got = _bits(getattr(pre.call, field))
        if got != ein[name]:
            bad.append((f"in:{name}", got, ein[name]))
    for field, (name, lo, hi) in _CALL_ARRAYS.items():
        arr = getattr(pre.call, field)
        for j, k in enumerate(range(lo, hi + 1)):
            # ZSNSO's buried slots are SNOW_INIT's stack residue in the
            # whole-column fixture and were zeroed by ENERGY's own harness.
            # NOAHMP_SFLX forwards ZSNSO verbatim, so the port reproduces the
            # residue; test_buried_slots_are_inert covers what that costs.
            if name == "ZSNSO" and k <= pre.call.isnow:
                continue
            n += 1
            got = _bits(arr[j])
            if got != ein[_key(name, k)]:
                bad.append((f"in:{_key(name, k)}", got, ein[_key(name, k)]))
    for field, name in _CALL_FROM_SEED.items():
        n += 1
        got = _bits(getattr(pre.call, field))
        if got != seed[name]:
            bad.append((f"in:{name}", got, seed[name]))
    for field, name in _SEED.items():
        n += 1
        got = _bits(getattr(pre, field))
        if got != seed[name]:
            bad.append((f"seed:{name}", got, seed[name]))
    return n, bad


@pytest.mark.parametrize("case", CASES)
def test_energy_boundary_bitwise(case):
    _p, _col, _smc, pre, _x = _run_pre(case)
    n, bad = _compare_boundary(case, pre)
    assert n >= 110, f"only {n} boundary values compared"
    assert not bad, f"{case}: {len(bad)} of {n} differ: {bad[:8]}"


def test_energy_boundary_comparator_can_fail():
    """A comparator that cannot reject is not a gate."""
    case = CASES[0]
    _p, _col, _smc, pre, _x = _run_pre(case)
    pre.call.fveg = np.float32(float(pre.call.fveg) * 1.0000001)
    _n, bad = _compare_boundary(case, pre)
    assert any(name == "in:FVEG" for name, *_ in bad)


# ---------------------------------------------------------------------------
# 2. the composition with ENERGY replayed from its own fixture
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", CASES)
def test_replayed_column_bitwise(case):
    p, col, _smc, pre, x = _run_pre(case)
    r = _run_post(case, p, col, pre, _replayed_energy(case), x)
    n, bad = _compare(case, col, r)
    assert n >= 150, f"only {n} outputs compared"
    assert not bad, f"{case}: {len(bad)} of {n} differ: {bad[:8]}"


def test_replayed_comparator_can_fail():
    case = CASES[0]
    p, col, _smc, pre, x = _run_pre(case)
    en = _replayed_energy(case)
    # CHLEAF is a pure diagnostic: it moves the comparison without disturbing
    # any of ERROR's three budgets, so the negative control tests the
    # comparator rather than the abort gates.
    en.chleaf = np.float32(float(en.chleaf) + 1.0)
    r = _run_post(case, p, col, pre, en, x)
    _n, bad = _compare(case, col, r)
    assert "chleaf" in {name for name, *_ in bad}


# ---------------------------------------------------------------------------
# 3. the whole column, from ported code end to end
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", CASES)
def test_whole_column_bitwise(case):
    col, r = _run_whole(case)
    n, bad = _compare(case, col, r)
    assert n >= 150, f"only {n} outputs compared"
    assert not bad, f"{case}: {len(bad)} of {n} differ: {bad[:8]}"


# ---------------------------------------------------------------------------
# 4. the comparison is not vacuous
# ---------------------------------------------------------------------------

def test_output_inventory_is_covered():
    """Every value the fixture records is either compared or proved inert."""
    covered = {(name, 0) for name in RESULT_SCALARS}
    for name, (lo, hi) in {**RESULT_ARRAYS, **COLUMN_ARRAYS}.items():
        covered |= {(name, k) for k in range(lo, hi + 1)}
    covered |= {("isnow", 0), ("snowh", 0), ("sneqv", 0)}
    claimed = covered | {(name, 0) for name in PASS_THROUGH}

    for case in CASES:
        recorded = set(SF[(case, "output")])
        missing = sorted(recorded - claimed)
        assert not missing, f"{case}: {missing} is recorded but never checked"
        assert len(recorded & covered) >= 150


@pytest.mark.parametrize("case", CASES)
def test_pass_through_state(case):
    """The fourteen outputs this identity kills leave the column unchanged.

    Proved out of the fixture -- entry bits equal exit bits -- rather than out
    of the port, so it is a statement about WRF and not about the transcription
    that omits them.
    """
    x, o = SF[(case, "input")], SF[(case, "output")]
    for name in PASS_THROUGH:
        assert _bits(x[(name, 0)]) == _bits(o[(name, 0)]), name


# ---------------------------------------------------------------------------
# 5. the undefined slots, and the arguments the identity makes dead
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", CASES)
def test_buried_dzsnso_slots_are_inert(case):
    """``DZSNSO`` above ``ISNOW`` is stack residue in WRF.  The port writes the
    defined ``0.0``; prove nothing downstream can tell."""
    p, col, _smc, pre, x = _run_pre(case)
    baseline = _run_post(case, p, col, pre, _replayed_energy(case), x)

    p2, col2, _s2, pre2, x2 = _run_pre(case)
    buried = pre2.call.isnow + NSNOW            # count of unwritten slots
    if buried == 0:
        pytest.skip("this column fills every snow slot")
    poison = np.float32(-1.0e30)
    pre2.dzsnso[:buried] = poison
    pre2.call.dzsnso[:buried] = poison
    perturbed = _run_post(case, p2, col2, pre2, _replayed_energy(case), x2)

    for name in RESULT_SCALARS:
        assert _bits(getattr(baseline, name)) == _bits(getattr(perturbed, name)), \
            f"{name} moved when a buried DZSNSO slot changed"


@pytest.mark.parametrize("case", CASES[:2])
def test_entry_ponding_and_fwet_are_dead(case):
    """``PONDING`` is INTENT(OUT) and ENERGY writes it before WATER reads it;
    ``FWET`` is INOUT and PRECIP_HEAT (:1554) writes it before ENERGY reads
    it.  Neither entry value can reach an output, and neither is an argument
    of :func:`sflx_pre`; drive ``sflx`` with a wild ``ponding`` and require the
    answer not to move."""
    _col, a = _run_whole(case)

    x = SF[(case, "input")]
    p = _params(x)
    col = _column(x)
    c = sflx(p, col, _vec(x, "smc", 1, NSOIL),
             ficeold=_vec(x, "ficeold", -NSNOW + 1, 0),
             wa=x[("wa", 0)], wslake=x[("wslake", 0)],
             ponding=np.float32(1.0e9), **_forcing(x))
    for name in RESULT_SCALARS:
        assert _bits(getattr(a, name)) == _bits(getattr(c, name)), name


def test_structural_refusals():
    x = SF[(CASES[0], "input")]
    p, col = _params(x), _column(x)
    smc = _vec(x, "smc", 1, NSOIL)
    with pytest.raises(ValueError, match="croptype"):
        sflx_pre(p, col, smc, wa=x[("wa", 0)], acc_ssoil=np.float32(0.0),
                 **_forcing(x, croptype=1))
    with pytest.raises(ValueError, match="irrfra"):
        sflx_pre(p, col, smc, wa=x[("wa", 0)], acc_ssoil=np.float32(0.0),
                 irrfra=np.float32(0.5), **_forcing(x))
    with pytest.raises(ValueError, match="iramtfi"):
        sflx_pre(p, col, smc, wa=x[("wa", 0)], acc_ssoil=np.float32(0.0),
                 iramtfi=np.float32(1e-3), **_forcing(x))


def test_night_albedo_sentinel():
    """WRF assigns -999.9 when SWDOWN is exactly zero (:1075); the value is a
    sentinel and must not be reachable by division."""
    _col, r = _run_whole("veg_warm_night_rain")
    assert _bits(r.albedo) == _bits(NIGHT_ALBEDO)
    _col, r = _run_whole("veg_warm_day_dry")
    assert 0.0 < float(r.albedo) < 1.0


# ---------------------------------------------------------------------------
# 6. ERROR
# ---------------------------------------------------------------------------

def _balanced(**over):
    """An ERROR call that closes all three budgets exactly."""
    kw = dict(
        swdown=np.float32(800.0), fsa=np.float32(600.0), fsr=np.float32(200.0),
        sav=np.float32(400.0), sag=np.float32(200.0),
        fira=np.float32(100.0), fsh=np.float32(200.0), fcev=np.float32(50.0),
        fgev=np.float32(150.0), fctr=np.float32(60.0), ssoil=np.float32(40.0),
        beg_wb=np.float32(500.0), canliq=np.float32(0.0),
        canice=np.float32(0.0), sneqv=np.float32(0.0), wa=np.float32(0.0),
        smc=np.full(NSOIL, np.float32(0.25)),
        dzsnso=np.asarray([0.1, 0.3, 0.6, 1.0], dtype=np.float32),
        prcp=np.float32(0.0), ecan=np.float32(0.0), etran=np.float32(0.0),
        edir=np.float32(0.0), runsrf=np.float32(0.0), runsub=np.float32(0.0),
        dt=np.float32(60.0), qtldrn=np.float32(0.0), ist=1,
        acc_dwater=np.float32(0.0), acc_prcp=np.float32(0.0),
        acc_ecan=np.float32(0.0), acc_etran=np.float32(0.0),
        acc_edir=np.float32(0.0))
    kw.update(over)
    return kw


def test_error_closes_the_reference_budget():
    """END_WB is CANLIQ+CANICE+SNEQV+WA plus SUM(SMC*DZ)*1000, i.e.
    0.25 * (0.1+0.3+0.6+1.0) * 1000 = 500 mm for the reference column."""
    r = error(**_balanced())
    assert float(r.end_wb) == 500.0
    assert float(r.acc_dwater) == 0.0
    assert float(r.errwat) == 0.0
    assert float(r.errsw) == 0.0
    assert float(r.erreng) == 0.0


def test_error_shortwave_gate():
    """Bracketed to within 0.1% of the threshold, so that a transcription
    carrying the wrong tolerance constant cannot pass.  ERRSW is driven through
    SWDOWN, whose ulp at 800 W/m2 quantises the residual to 6.1e-5; the two
    values below are the neighbouring representable ones either side."""
    error(**_balanced(swdown=np.float32(800.0 - 0.0099)))
    with pytest.raises(ShortwaveBalanceError):
        error(**_balanced(swdown=np.float32(800.0 + 0.010002)))
    with pytest.raises(ShortwaveBalanceError):
        error(**_balanced(fsr=np.float32(200.0 + 0.02)))


def test_error_energy_gate():
    """ERRENG is ``(SAV+SAG) - sink + PAH`` and the reference column makes the
    first two cancel exactly, so PAH drives the residual with no cancellation
    and the gate can be bracketed at 0.02%."""
    error(**_balanced(pah=np.float32(0.009998)))
    with pytest.raises(EnergyBalanceError):
        error(**_balanced(pah=np.float32(0.010002)))
    with pytest.raises(EnergyBalanceError):
        error(**_balanced(ssoil=np.float32(40.0 + 0.02)))


def test_error_water_gate():
    """The water gate is the only one of the three with a 0.1 threshold, and
    the only one whose message distinguishes gain from loss."""
    assert float(error(**_balanced()).errwat) == 0.0
    error(**_balanced(beg_wb=np.float32(500.0 - 0.09)))
    with pytest.raises(WaterBalanceError, match="gaining"):
        error(**_balanced(beg_wb=np.float32(500.0 - 0.2)))
    with pytest.raises(WaterBalanceError, match="losing"):
        error(**_balanced(beg_wb=np.float32(500.0 + 0.2)))
    # Bracketed at 0.02%.  With END_WB == BEG_WB the residual is exactly
    # RUNSRF, so the threshold can be approached without cancellation.
    error(**_balanced(runsrf=np.float32(0.09998)))
    with pytest.raises(WaterBalanceError, match="gaining"):
        error(**_balanced(runsrf=np.float32(0.10002)))


def test_error_lake_branch_reports_zero():
    """IST /= 1 skips the whole water block: ERRWAT is 0 and the five
    accumulators leave unchanged (:1733-1734)."""
    r = error(**_balanced(ist=2, beg_wb=np.float32(1.0e6),
                          acc_prcp=np.float32(7.0)))  # would abort at IST=1
    assert float(r.errwat) == 0.0
    assert float(r.acc_prcp) == 7.0
    assert float(r.acc_dwater) == 0.0


def test_error_accumulates_across_steps():
    """The five ACC_* are INOUT: a second call must add to the first, which is
    what makes them a soil-timestep accumulation rather than a diagnostic."""
    kw = _balanced(prcp=np.float32(1.0e-3),
                   ecan=np.float32(1.0e-4), etran=np.float32(2.0e-4),
                   edir=np.float32(3.0e-4), runsrf=np.float32(0.03))
    first = error(**kw)
    kw2 = dict(kw)
    kw2.update(acc_dwater=first.acc_dwater, acc_prcp=first.acc_prcp,
               acc_ecan=first.acc_ecan, acc_etran=first.acc_etran,
               acc_edir=first.acc_edir)
    second = error(**kw2)
    assert float(second.acc_prcp) == pytest.approx(2.0 * float(first.acc_prcp))
    assert float(second.acc_edir) == pytest.approx(2.0 * float(first.acc_edir))


def test_error_irrigation_terms_are_signed_into_the_residual():
    """IRFIRATE and IRMIRATE are m/timestep and enter ERRWAT scaled by 1000.
    They are identically zero under OPT_IRR = 0, so the only way to keep the
    transcription honest is to exercise them directly."""
    base = error(**_balanced())
    with_irr = error(**_balanced(irfirate=np.float32(1.0e-5)))
    assert float(base.errwat) == 0.0
    assert float(with_irr.errwat) == pytest.approx(-0.01, abs=1e-6)


def test_error_skips_the_residual_without_calculate_soil():
    """``calculate_soil`` false leaves ERRWAT unassigned in WRF; the port
    defines it as 0.0 and still accumulates, which is the whole point of the
    ACC_* variables on a non-soil substep."""
    r = error(**_balanced(beg_wb=np.float32(1.0e6)), calculate_soil=False)
    assert float(r.errwat) == 0.0
    assert float(r.acc_dwater) != 0.0


# ---------------------------------------------------------------------------
# 7. ERROR against its own oracle, ``noahmp-sflx-error.csv``
# ---------------------------------------------------------------------------
#
# Built by tools/noahmp_wrf461_oracle/build_sflx_compose.sh, which calls the
# accessibility-lifted module's ERROR directly.  It exists because three things
# ERROR owns are invisible from the whole column: ERRWAT (a NOAHMP_SFLX local
# that reaches no output), the IST /= 1 lake branch, and the IRFIRATE/IRMIRATE
# terms that opt_irr = 0 pins at zero.  All four build levels -- -O0,
# -ffp-contract=off, -finit-real=snan and WRF's own FCOPTIM -- emit a
# byte-identical fixture; ERROR evaluates no transcendental at any option
# setting, and stage 5 of the build proves it with nm -u.

ERROR_CSV = ORACLE / "noahmp-sflx-error.csv"
ERROR_PINNED = \
    "40ba41db5e50fbdcc8f635b7b88af167062274f21e2eb18b4b816d21c1f00f56"

#: The harness poisons ERRWAT before every call, as the binary32 it becomes.
ERRWAT_POISON = np.float32(-9.0e30)


def _load_error():
    table = defaultdict(dict)
    for row in csv.DictReader(ERROR_CSV.open(newline="")):
        key = (row["field"], int(row["index"]))
        table[(row["case"], row["stage"])][key] = (
            int(row["value"]) if row["dtype"] == "int"
            else _from_hex(row["bits"]))
    return table


ERR = _load_error()
ERROR_CASES = tuple(dict.fromkeys(
    row["case"] for row in csv.DictReader(ERROR_CSV.open(newline=""))))


def test_error_fixture_is_byte_pinned():
    got = hashlib.sha256(ERROR_CSV.read_bytes()).hexdigest()
    assert got == ERROR_PINNED, f"noahmp-sflx-error.csv changed: {got}"
    assert len(ERROR_CASES) == 16


@pytest.mark.parametrize("case", ERROR_CASES)
def test_error_bitwise(case):
    """max_ulp 0 on ERRWAT and all five accumulators, every case."""
    i = ERR[(case, "input")]
    o = ERR[(case, "output")]
    r = error(
        swdown=i[("swdown", 0)], fsa=i[("fsa", 0)], fsr=i[("fsr", 0)],
        fira=i[("fira", 0)], fsh=i[("fsh", 0)], fcev=i[("fcev", 0)],
        fgev=i[("fgev", 0)], fctr=i[("fctr", 0)], ssoil=i[("ssoil", 0)],
        sav=i[("sav", 0)], sag=i[("sag", 0)], beg_wb=i[("beg_wb", 0)],
        canliq=i[("canliq", 0)], canice=i[("canice", 0)],
        sneqv=i[("sneqv", 0)], wa=i[("wa", 0)],
        smc=_vec(i, "smc", 1, NSOIL), dzsnso=_vec(i, "dzsnso", 1, NSOIL),
        prcp=i[("prcp", 0)], ecan=i[("ecan", 0)], etran=i[("etran", 0)],
        edir=i[("edir", 0)], runsrf=i[("runsrf", 0)],
        runsub=i[("runsub", 0)], dt=i[("dt", 0)], qtldrn=i[("qtldrn", 0)],
        ist=int(i[("ist", 0)]), acc_dwater=i[("acc_dwater", 0)],
        acc_prcp=i[("acc_prcp", 0)], acc_ecan=i[("acc_ecan", 0)],
        acc_etran=i[("acc_etran", 0)], acc_edir=i[("acc_edir", 0)],
        pah=i[("pah", 0)], firr=i[("firr", 0)], canhs=i[("canhs", 0)],
        irmirate=i[("irmirate", 0)], irfirate=i[("irfirate", 0)],
        nsoil=int(i[("nsoil", 0)]), nsnow=int(i[("nsnow", 0)]),
        calculate_soil=bool(i[("calculate_soil", 0)]))

    for name in ("acc_dwater", "acc_prcp", "acc_ecan", "acc_etran",
                 "acc_edir"):
        assert _bits(getattr(r, name)) == _bits(o[(name, 0)]), name

    if int(i[("ist", 0)]) == 1 and not i[("calculate_soil", 0)]:
        # WRF leaves ERRWAT unassigned here (1708-1731 is the only statement
        # that writes it on the IST == 1 path), so the fixture records the
        # harness's poison.  The port defines it as 0.0, which is the defined
        # value; asserting the poison instead would be asserting stack state.
        assert _bits(o[("errwat", 0)]) == _bits(ERRWAT_POISON)
        assert float(r.errwat) == 0.0
    else:
        assert _bits(r.errwat) == _bits(o[("errwat", 0)]), "errwat"


def test_error_oracle_covers_the_branches_the_column_cannot():
    """The fixture must actually reach what noahmp-sflx.csv cannot, or it is
    16 more copies of a check that already passes."""
    lake = [c for c in ERROR_CASES if ERR[(c, "input")][("ist", 0)] != 1]
    assert len(lake) >= 2, "no lake column"
    irr = [c for c in ERROR_CASES
           if ERR[(c, "input")][("irfirate", 0)] != 0.0
           or ERR[(c, "input")][("irmirate", 0)] != 0.0]
    assert len(irr) >= 3, "the irrigation terms are never exercised"
    nonzero = [c for c in ERROR_CASES
               if ERR[(c, "input")][("ist", 0)] == 1
               and ERR[(c, "input")][("calculate_soil", 0)]
               and float(ERR[(c, "output")][("errwat", 0)]) != 0.0]
    assert len(nonzero) >= 2, "ERRWAT is zero on every case"
    assert any(float(ERR[(c, "output")][("errwat", 0)]) > 0 for c in nonzero)
    assert any(float(ERR[(c, "output")][("errwat", 0)]) < 0 for c in nonzero)
    substep = [c for c in ERROR_CASES
               if not ERR[(c, "input")][("calculate_soil", 0)]]
    assert substep, "the non-soil substep is never exercised"


def test_error_comparator_can_fail():
    """Every ERROR argument the port consumes must move at least one output on
    at least one case; a fixture that cannot tell is not a gate."""
    case = "err_second_substep"
    i = ERR[(case, "input")]

    def run(**over):
        kw = dict(
            swdown=i[("swdown", 0)], fsa=i[("fsa", 0)], fsr=i[("fsr", 0)],
            fira=i[("fira", 0)], fsh=i[("fsh", 0)], fcev=i[("fcev", 0)],
            fgev=i[("fgev", 0)], fctr=i[("fctr", 0)], ssoil=i[("ssoil", 0)],
            sav=i[("sav", 0)], sag=i[("sag", 0)], beg_wb=i[("beg_wb", 0)],
            canliq=i[("canliq", 0)], canice=i[("canice", 0)],
            sneqv=i[("sneqv", 0)], wa=i[("wa", 0)],
            smc=_vec(i, "smc", 1, NSOIL), dzsnso=_vec(i, "dzsnso", 1, NSOIL),
            prcp=i[("prcp", 0)], ecan=i[("ecan", 0)], etran=i[("etran", 0)],
            edir=i[("edir", 0)], runsrf=i[("runsrf", 0)],
            runsub=i[("runsub", 0)], dt=i[("dt", 0)],
            qtldrn=i[("qtldrn", 0)], ist=int(i[("ist", 0)]),
            acc_dwater=i[("acc_dwater", 0)], acc_prcp=i[("acc_prcp", 0)],
            acc_ecan=i[("acc_ecan", 0)], acc_etran=i[("acc_etran", 0)],
            acc_edir=i[("acc_edir", 0)])
        kw.update(over)
        return error(**kw)

    base = run()

    def moved(**over) -> bool:
        """A perturbation counts as observed if ERRWAT changes *or* the water
        gate fires.  An abort is the strongest possible observation, and
        refusing to count it would force the perturbations to be small enough
        to stop discriminating."""
        try:
            return _bits(run(**over).errwat) != _bits(base.errwat)
        except WaterBalanceError:
            return True

    for name in ("beg_wb", "canliq", "canice", "sneqv", "wa", "prcp", "ecan",
                 "etran", "edir", "runsrf", "runsub", "qtldrn", "dt"):
        assert moved(**{name: np.float32(float(i[(name, 0)]) + 0.25)}), \
            f"{name} moves no output; the transcription is unconstrained"
    for k in range(NSOIL):
        smc = _vec(i, "smc", 1, NSOIL).copy()
        smc[k] = np.float32(float(smc[k]) + 0.01)
        assert moved(smc=smc), f"smc({k + 1})"
        dz = _vec(i, "dzsnso", 1, NSOIL).copy()
        dz[k] = np.float32(float(dz[k]) + 0.01)
        assert moved(dzsnso=dz), f"dzsnso({k + 1})"
    # The energy- and shortwave-balance arguments carry no ERROR output at all:
    # the only way a transcription error in either residual is observable is
    # the gate, so each argument must be shown to reach it.
    for name in ("fira", "fsh", "fcev", "fgev", "fctr", "ssoil", "sav", "sag",
                 "pah", "firr", "canhs"):
        with pytest.raises(EnergyBalanceError):
            run(**{name: np.float32(float(i[(name, 0)]) + 1.0)})
    for name in ("fsa", "fsr", "swdown"):
        with pytest.raises(ShortwaveBalanceError):
            run(**{name: np.float32(float(i[(name, 0)]) + 1.0)})


# ---------------------------------------------------------------------------
# the staged whole-column continuation
# ---------------------------------------------------------------------------
def _staged_call(case, **override):
    """The arguments :func:`_run_whole` uses, without running anything."""
    x = SF[(case, "input")]
    p = _params(x)
    col = _column(x)
    return (p, col, _vec(x, "smc", 1, NSOIL)), dict(
        ficeold=_vec(x, "ficeold", -NSNOW + 1, 0),
        wa=x[("wa", 0)], wslake=x[("wslake", 0)],
        **_forcing(x, **override))


def _drive_interleaved(cases):
    """Every case's whole column, run in lockstep with batched leaves.

    This is the shape ``gpuwm.core.noahmp_runtime`` uses around the CUDA
    batches, with the CPython leaves substituted: each round collects the leaf
    every still-running column is paused at, answers those calls together, and
    resumes.  Nothing is restarted and nothing is recomputed.
    """
    from gpuwm.core.noahmp_energy import LeafRequest, answer_on_host
    from gpuwm.core.noahmp_sflx import sflx_steps

    calls = [_staged_call(case) for case in cases]
    steps = [sflx_steps(*args, **kwargs) for args, kwargs in calls]
    requests = [gen.send(None) for gen in steps]
    results = [None] * len(steps)
    active = list(range(len(steps)))
    rounds = []
    while active:
        buckets = {}
        for index in active:
            buckets.setdefault(requests[index].leaf, []).append(index)
        rounds.append({leaf: len(idx) for leaf, idx in buckets.items()})
        answers = {}
        for leaf, indices in buckets.items():
            for index in indices:
                request = requests[index]
                answers[index] = answer_on_host(
                    LeafRequest(leaf, request.args, request.kwargs))
        still = []
        for index in active:
            try:
                requests[index] = steps[index].send(answers[index])
            except StopIteration as stop:
                results[index] = stop.value
            else:
                still.append(index)
        active = still
    return [(args[1], result) for (args, _kw), result in zip(calls, results)], \
        rounds


def test_the_staged_whole_column_is_bitwise_against_the_fixture():
    """The batched runtime shape, held to the same unmodified-WRF fixture.

    ``test_whole_column_bitwise`` runs one column at a time through ``sflx``.
    This runs all four interleaved through ``sflx_steps`` with their RADIATION,
    VEGE_FLUX, BARE_FLUX and WATER calls collected and answered in batches --
    the exact control flow the device path uses -- and requires the same bit
    comparison against ``noahmp-sflx.csv``.  A staging bug that paired a result
    with the wrong column would fail here and nowhere else.
    """
    staged, rounds = _drive_interleaved(CASES)
    for case, (col, r) in zip(CASES, staged):
        n, bad = _compare(case, col, r)
        assert not bad, (case, bad[:10])
        assert n >= 60, (case, n)

    # Every column reaches every leaf on its path exactly once.  The round
    # count itself is a scheduling detail -- a bare column runs ahead of a
    # vegetated one by the round VEGE_FLUX costs -- so it is not asserted; the
    # totals are what a mis-staged seam would break.
    total = {}
    for entry in rounds:
        for leaf, count in entry.items():
            total[leaf] = total.get(leaf, 0) + count
    for leaf in ("sflx_pre", "thermoprop", "radiation", "bare_flux",
                 "tsnosoi", "phasechange", "water"):
        assert total[leaf] == len(CASES), (leaf, rounds)
    assert 0 < total["vege_flux"] <= len(CASES), rounds
    # NOAHMP_SFLX's own first half is the first thing every column pauses at:
    # ATM, PHENOLOGY and PRECIP_HEAT run before ENERGY is entered at all.
    assert rounds[0] == {"sflx_pre": len(CASES)}, rounds
    assert rounds[1] == {"thermoprop": len(CASES)}, rounds


def test_the_staged_whole_column_comparison_can_fail():
    """Misroute one WATER answer between columns; the fixture must reject it.

    WATER mutates the caller's column in place, so a mispaired answer is the
    defect most likely to look plausible: the arrays are all the right shape
    and the wrong column's water balance.

    Measured outcome: on this fixture the mispairing is caught *before* the bit
    comparison, by NOAHMP_SFLX's own ERRWAT check at :1713 -- 115 kg/m2 of
    water appears from nowhere.  That is a stronger rejection than a differing
    bit, so the assertion admits either and says which one fired.
    """
    from gpuwm.core.noahmp_energy import LeafRequest, answer_on_host
    from gpuwm.core.noahmp_sflx import NoahmpBalanceError, sflx_steps

    calls = [_staged_call(case) for case in CASES]
    steps = [sflx_steps(*args, **kwargs) for args, kwargs in calls]
    requests = [gen.send(None) for gen in steps]
    results = [None] * len(steps)
    active = list(range(len(steps)))
    raised = None
    try:
        while active:
            answers = {}
            water_indices = [i for i in active if requests[i].leaf == "water"]
            for index in active:
                request = requests[index]
                # Answer the last column's WATER with the first column's call.
                if request.leaf == "water" and len(water_indices) > 1 \
                        and index == water_indices[-1]:
                    request = requests[water_indices[0]]
                answers[index] = answer_on_host(
                    LeafRequest(request.leaf, request.args, request.kwargs))
            still = []
            for index in active:
                try:
                    requests[index] = steps[index].send(answers[index])
                except StopIteration as stop:
                    results[index] = stop.value
                else:
                    still.append(index)
            active = still
    except NoahmpBalanceError as error:
        raised = error

    differing = []
    for case, (args, _kw), result in zip(CASES, calls, results):
        if result is None:
            continue
        _n, bad = _compare(case, args[1], result)
        if bad:
            differing.append(case)
    assert raised is not None or differing, (
        "misrouting a WATER answer between columns went undetected; the "
        "staged whole-column comparison is not sensitive to pairing")
