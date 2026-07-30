"""max_ulp 0 gate for the Noah-MP ENERGY assembly against the WRF v4.6.1 oracle.

``gpuwm/data/noahmp/oracle/noahmp-energy.csv`` was produced by
``tools/noahmp_wrf461_oracle/build_energy.sh`` from the pinned WRF tree with
nothing but a ``private::`` -> ``public::`` accessibility lift, at ``-O0``.
Every value is stored as its IEEE-754 binary32 bit pattern, so equality here is
bitwise; there is no tolerance to widen.

What this file is for, beyond the bit gate: ENERGY is a composition, so
``max_ulp 0`` on nine columns proves the arithmetic and proves nothing about
whether the composition reads its arguments.  ``test_energy_fixture_kills_*``
is the part that does.
"""

from __future__ import annotations

import csv
import hashlib
import pathlib
import struct

import numpy as np
import pytest

from gpuwm.core.fp32_ulp import max_ulp
from gpuwm.core.noahmp import load_noahmp_parameters, transfer_mp_parameters
from gpuwm.core.noahmp_energy import (
    EnergyOptions,
    EnergyParameters,
    PINNED_OPTIONS,
    energy,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "gpuwm" / "data" / "noahmp" / "oracle" / "noahmp-energy.csv"

#: The fixture this gate is pinned to.  A regenerated CSV must land here.
FIXTURE_SHA256 = "c99cf60c5e58c07f3292c49d14ed950a6f11dcd66ed511f5e8ed22a4a6bc189c"

NSNOW = 3
NSOIL = 4

CASES = (
    "veg_warm_day_dry",
    "veg_warm_night_rain",
    "snowpack_frozen_soil",
    "bare_thin_snow_melt",
    "veg_calm_desert_dry",
    "veg_deep_snow_saturated",
    "veg_subfreezing_canopy",
    "urban_snowfree",
    "veg_single_snow_layer",
)


# --------------------------------------------------------------------------
def _from_bits(h: str) -> float:
    return struct.unpack("<f", struct.pack("<I", int(h, 16)))[0]


def _to_int(h: str) -> int:
    raw = int(h, 16)
    return raw - (1 << 32) if raw >= (1 << 31) else raw


def _ulp_gap(a: float, b: float) -> int:
    fa = np.asarray([a], dtype=np.float32)
    fb = np.asarray([b], dtype=np.float32)
    return max_ulp(fa, fb)


def load_fixture():
    if not FIXTURE.exists():                                # pragma: no cover
        pytest.skip(f"oracle fixture missing: {FIXTURE}")
    table: dict = {}
    with FIXTURE.open(newline="") as fh:
        for row in csv.DictReader(fh):
            case = table.setdefault(
                row["case"].strip(),
                {"opt": {}, "cfg": {}, "par": {}, "seed": {}, "in": {},
                 "out": {}, "undef": {}})
            case[row["role"].strip()][row["name"].strip()] = row["hex"].strip()
    return table


FX = load_fixture()


def _vec(case, role, name, lo, hi):
    """Fortran ``name(lo:hi)`` as a 0-based list."""
    return [_from_bits(FX[case][role][f"{name}_{k:+d}"])
            for k in range(lo, hi + 1)]


def _soil(case, role, name):
    return [_from_bits(FX[case][role][f"{name}_{k:+d}"])
            for k in range(1, NSOIL + 1)]


def _parameters(case) -> EnergyParameters:
    cfg = FX[case]["cfg"]
    bundle = load_noahmp_parameters()
    transferred = transfer_mp_parameters(
        bundle,
        vegtype=_to_int(cfg["VEGTYP"]),
        soiltype=[_to_int(cfg["SOILTYPE"])] * NSOIL,
        slopetype=_to_int(cfg["SLOPETYPE"]),
        soilcolor=_to_int(cfg["SOILCOLOR"]),
        croptype=_to_int(cfg["CROPTYPE"]),
    )
    return EnergyParameters.from_transferred(transferred, nsoil=NSOIL)


def _call(case, *, overrides=None, options=PINNED_OPTIONS):
    """Return ``(parameters, kwargs)`` for the fixture's pinned entry state."""
    ent = FX[case]["in"]
    kwargs = dict(
        ice=_to_int(ent["ICE"]), vegtyp=_to_int(FX[case]["cfg"]["VEGTYP"]),
        ist=_to_int(ent["IST"]), isnow=_to_int(ent["ISNOW"]),
        iloc=_to_int(ent["ILOC"]), jloc=_to_int(ent["JLOC"]),
        dt=_from_bits(ent["DT"]), rhoair=_from_bits(ent["RHOAIR"]),
        sfcprs=_from_bits(ent["SFCPRS"]), qair=_from_bits(ent["QAIR"]),
        sfctmp=_from_bits(ent["SFCTMP"]), thair=_from_bits(ent["THAIR"]),
        lwdn=_from_bits(ent["LWDN"]), uu=_from_bits(ent["UU"]),
        vv=_from_bits(ent["VV"]), zref=_from_bits(ent["ZREF"]),
        co2air=_from_bits(ent["CO2AIR"]), o2air=_from_bits(ent["O2AIR"]),
        solad=[_from_bits(ent["SOLAD_+1"]), _from_bits(ent["SOLAD_+2"])],
        solai=[_from_bits(ent["SOLAI_+1"]), _from_bits(ent["SOLAI_+2"])],
        cosz=_from_bits(ent["COSZ"]), igs=_from_bits(ent["IGS"]),
        eair=_from_bits(ent["EAIR"]), tbot=_from_bits(ent["TBOT"]),
        zsnso=_vec(case, "in", "ZSNSO", -NSNOW + 1, NSOIL),
        dzsnso=_vec(case, "in", "DZSNSO", -NSNOW + 1, NSOIL),
        stc=_vec(case, "in", "STC", -NSNOW + 1, NSOIL),
        zsoil=_soil(case, "in", "ZSOIL"),
        sh2o=_soil(case, "in", "SH2O"), smc=_soil(case, "in", "SMC"),
        snice=_vec(case, "in", "SNICE", -NSNOW + 1, 0),
        snliq=_vec(case, "in", "SNLIQ", -NSNOW + 1, 0),
        elai=_from_bits(ent["ELAI"]), esai=_from_bits(ent["ESAI"]),
        fwet=_from_bits(ent["FWET"]), foln=_from_bits(ent["FOLN"]),
        fveg=_from_bits(ent["FVEG"]),
        pahv=_from_bits(ent["PAHV"]), pahg=_from_bits(ent["PAHG"]),
        pahb=_from_bits(ent["PAHB"]), qsnow=_from_bits(ent["QSNOW"]),
        lat=_from_bits(ent["LAT"]), canliq=_from_bits(ent["CANLIQ"]),
        canice=_from_bits(ent["CANICE"]),
        tv=_from_bits(ent["TV"]), tg=_from_bits(ent["TG"]),
        snowh=_from_bits(ent["SNOWH"]), eah=_from_bits(ent["EAH"]),
        tah=_from_bits(ent["TAH"]), sneqvo=_from_bits(ent["SNEQVO"]),
        sneqv=_from_bits(ent["SNEQV"]), albold=_from_bits(ent["ALBOLD"]),
        cm=_from_bits(ent["CM"]), ch=_from_bits(ent["CH"]),
        dx=_from_bits(ent["DX"]), dz8w=_from_bits(ent["DZ8W"]),
        q2=_from_bits(ent["Q2"]), tauss=_from_bits(ent["TAUSS"]),
        qc=_from_bits(ent["QC"]), qsfc=_from_bits(ent["QSFC"]),
        psfc=_from_bits(ent["PSFC"]),
        acc_ssoil=_from_bits(ent["ACC_SSOIL"]),
        options=options,
    )
    if overrides:
        kwargs.update(overrides)
    return _parameters(case), kwargs


def _run(case, *, overrides=None, options=PINNED_OPTIONS):
    """Drive ENERGY on the fixture's pinned entry state for ``case``."""
    parameters, kwargs = _call(case, overrides=overrides, options=options)
    return energy(parameters, **kwargs)


# --------------------------------------------------------------------------
# how the state maps onto the fixture's output names
# --------------------------------------------------------------------------
_SCALARS = (
    "Z0WRF", "T2M", "FSNO", "SAV", "SAG", "QMELT", "FSA", "FSR", "TAUX",
    "TAUY", "FIRA", "FSH", "FCEV", "FGEV", "FCTR", "TRAD", "PSN", "APAR",
    "SSOIL", "BTRAN", "PONDING", "TS", "LATHEAV", "LATHEAG", "TV", "TG",
    "SNOWH", "EAH", "TAH", "SNEQVO", "SNEQV", "ALBOLD", "CM", "CH", "TAUSS",
    "LAISUN", "LAISHA", "RB", "QSFC", "T2MV", "T2MB", "FSRV", "FSRG",
    "RSSUN", "RSSHA", "BGAP", "WGAP", "TGV", "TGB", "Q1", "Q2V", "Q2B",
    "Q2E", "CHV", "CHB", "EMISSI", "PAH", "CANHS", "SHG", "SHC", "SHB",
    "EVG", "EVB", "GHV", "GHB", "IRG", "IRC", "IRB", "TR", "EVC", "CHLEAF",
    "CHUC", "CHV2", "CHB2", "EFLXB", "ACC_SSOIL",
)
_LAYER_ARRAYS = (("HCPCT", -NSNOW + 1, NSOIL), ("STC", -NSNOW + 1, NSOIL),
                 ("SNICEV", -NSNOW + 1, 0), ("SNLIQV", -NSNOW + 1, 0),
                 ("EPORE", -NSNOW + 1, 0), ("SNICE", -NSNOW + 1, 0),
                 ("SNLIQ", -NSNOW + 1, 0), ("BTRANI", 1, NSOIL),
                 ("SH2O", 1, NSOIL), ("SMC", 1, NSOIL),
                 ("ALBSND", 1, 2), ("ALBSNI", 1, 2))
_LOGICALS = ("FROZEN_CANOPY", "FROZEN_GROUND")


def _port_pairs(case, st):
    """(name, got, want) over every output the fixture pins as defined."""
    out = FX[case]["out"]
    pairs = []
    for name in _SCALARS:
        if name not in out:                     # role 'undef' for this case
            continue
        pairs.append((name, float(getattr(st, name.lower())),
                      _from_bits(out[name])))
    for name, lo, hi in _LAYER_ARRAYS:
        seq = getattr(st, name.lower())
        for k in range(lo, hi + 1):
            key = f"{name}_{k:+d}"
            if key not in out:
                continue
            idx = k - lo if lo == 1 else k + NSNOW - 1
            pairs.append((key, float(seq[idx]), _from_bits(out[key])))
    return pairs


# --------------------------------------------------------------------------
def test_fixture_is_the_pinned_one():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    assert tuple(FX) == CASES


def test_option_identity_is_the_registry_default():
    opt = FX[CASES[0]]["opt"]
    want = {"DVEG": 4, "OPT_CRS": 1, "OPT_BTR": 1, "OPT_RUN": 3,
            "OPT_SFC": 1, "OPT_FRZ": 1, "OPT_INF": 1, "OPT_RAD": 3,
            "OPT_ALB": 2, "OPT_SNF": 1, "OPT_TBOT": 2, "OPT_STC": 1,
            "OPT_RSF": 1, "OPT_SOIL": 1, "OPT_PEDO": 1, "OPT_CROP": 0,
            "OPT_IRR": 0, "OPT_IRRM": 0, "OPT_INFDV": 0, "OPT_TDRN": 0,
            "SOIL_UPDATE_STEPS": 1, "CALCULATE_SOIL": 1}
    assert {k: _to_int(v) for k, v in opt.items()} == want


def test_transferred_parameters_match_the_fixture_echo():
    """The port rebuilds parameters from the tables; the oracle read them.

    If those two ever diverge, every ``max_ulp 0`` below would be measuring
    the parameter transfer rather than ENERGY, so check the overlap first.
    """
    for case in CASES:
        p = _parameters(case)
        par = FX[case]["par"]
        assert p.urban_flag == bool(_to_int(par["URBAN_FLAG"])), case
        assert p.nroot == _to_int(par["NROOT"]), case
        for name in ("MFSNO", "SCFFAC", "Z0SNO", "Z0MVT", "HVT", "CWPVT",
                     "SNOW_EMIS", "RSURF_EXP", "RSURF_SNOW"):
            assert float(getattr(p, name.lower())) == _from_bits(par[name]), \
                (case, name)
        for k in range(1, 3):
            assert float(p.eg[k - 1]) == _from_bits(par[f"EG_{k:+d}"]), case
        for k in range(1, NSOIL + 1):
            for name in ("SMCMAX", "SMCREF", "SMCWLT", "PSISAT", "BEXP"):
                assert float(getattr(p, name.lower())[k - 1]) == \
                    _from_bits(par[f"{name}_{k:+d}"]), (case, name, k)


@pytest.mark.parametrize("case", CASES)
def test_energy_bit_exact(case):
    st = _run(case)
    pairs = _port_pairs(case, st)
    assert len(pairs) >= 90, case
    bad = [(n, g, w) for n, g, w in pairs
           if struct.pack("<f", np.float32(g)) !=
           struct.pack("<f", np.float32(w))]
    assert not bad, (case, bad[:12])
    got = np.asarray([g for _, g, _ in pairs], dtype=np.float32)
    want = np.asarray([w for _, _, w in pairs], dtype=np.float32)
    assert max_ulp(got, want) == 0


@pytest.mark.parametrize("case", CASES)
def test_integer_and_logical_outputs_bit_exact(case):
    st = _run(case)
    out = FX[case]["out"]
    for k in range(-NSNOW + 1, NSOIL + 1):
        key = f"IMELT_{k:+d}"
        if key in out:
            assert st.imelt[k + NSNOW - 1] == _to_int(out[key]), (case, key)
    assert int(st.frozen_canopy) == _to_int(out["FROZEN_CANOPY"]), case
    assert int(st.frozen_ground) == _to_int(out["FROZEN_GROUND"]), case


@pytest.mark.parametrize("case", CASES)
def test_undefined_slots_get_the_defined_zero(case):
    """WRF leaves these unassigned; the port must produce exactly 0.0.

    The fixture never pins the residue -- see
    ``gpuwm/data/noahmp/oracle/PROVENANCE-energy.md`` -- so this is the only
    statement of what the port is allowed to do there.
    """
    st = _run(case)
    isnow = _to_int(FX[case]["in"]["ISNOW"])
    nroot = _to_int(FX[case]["par"]["NROOT"])
    cosz = _from_bits(FX[case]["in"]["COSZ"])
    undef = set(FX[case]["undef"])
    assert undef, case if isnow < 0 or cosz <= 0 or nroot < NSOIL else True

    for name in undef:
        if name in ("FSRV", "FSRG", "BGAP", "WGAP"):
            assert float(getattr(st, name.lower())) == 0.0, (case, name)
            continue
        base, idx = name.rsplit("_", 1)
        k = int(idx)
        seq = getattr(st, base.lower())
        pos = k - 1 if base == "BTRANI" else k + NSNOW - 1
        assert float(seq[pos]) == 0.0, (case, name)


def test_every_live_branch_is_taken_somewhere():
    """The composition's branches, recorded by the port as it runs them.

    The oracle asserts the same coverage from the entry state
    (``validate_energy_oracle.py``); this asserts the port actually reaches
    the arm the entry state selects, which is a different claim.
    """
    seen: dict[str, set] = {}
    for case in CASES:
        for key, value in _run(case).branches.items():
            seen.setdefault(key, set()).add(value)
    assert seen["veg"] == {True, False}
    assert seen["tile_veg"] == {True, False}
    assert seen["fsno_snow"] == {True, False}
    assert seen["ur_clamped"] == {True, False}
    assert seen["zpd_snow"] == {True, False}
    assert seen["urban"] == {True, False}
    assert seen["frozen_canopy"] == {True, False}
    assert seen["frozen_ground"] == {True, False}
    assert seen["rsurf_saturated"] == {True, False}
    assert seen["rsurf_dry_cap"] == {True}
    assert seen["rsurf_urban_cap"] == {True}
    assert seen["btran_gx_low_clamp"] == {True}
    assert seen["btran_gx_high_clamp"] == {True}
    assert seen["isnow"] == {0, -1, -2, -3}


# --------------------------------------------------------------------------
# the part that max_ulp 0 does not prove
# --------------------------------------------------------------------------
#: Entry-state slots whose perturbation must change some pinned output, and
#: the case to test each one on.  A slot that survives being moved is a slot
#: the fixture cannot tell the port reads.
_MUTABLE_SCALARS = (
    "DT", "RHOAIR", "SFCPRS", "QAIR", "SFCTMP", "LWDN", "UU", "VV",
    "ZREF", "CO2AIR", "O2AIR", "COSZ", "IGS", "EAIR", "TBOT", "ELAI", "ESAI",
    "FWET", "FOLN", "FVEG", "PAHV", "PAHG", "PAHB", "CANLIQ",
    "CANICE", "TV", "TG", "SNOWH", "EAH", "TAH", "SNEQVO", "SNEQV", "ALBOLD",
    "TAUSS", "PSFC", "ACC_SSOIL", "QSNOW",
)

#: Arguments that are genuinely inert under the pinned identity, each with the
#: WRF line numbers that make it so.  These are *not* exempt from being probed
#: -- the study asserts they move nothing, which is the stronger claim.  Every
#: reason below was read out of the pinned source, not assumed.
_INERT = {
    "DX": "in the argument list of both flux routines (:3590, :4182) and "
          "referenced nowhere in either body",
    "DZ8W": "same as DX (:3590, :4182)",
    "QC": "same (:3594, :4183); it exists for CALHUM, dead under opt_crs=1",
    "Q2": "same (:3588, :4179)",
    "THAIR": "reaches only SFCDIF2 (:3898, :4368), dead under opt_sfc=1",
    "LAT": "reaches only THERMOPROP, whose body never reads it "
           "(the leaves fixture's zero-probe sweep is the proof)",
    "CM": "SFCDIF1 (:3894, :4364) writes it before the first read at "
          ":3905/:4375, so the entry value is dead",
    "CH": "same as CM (:3894/:3904, :4364/:4374)",
    "QSFC": "VEGE_FLUX rewrites it at :3851 and BARE_FLUX at :4435, both "
            "before any read, so the entry value is dead",
}

#: ENERGY declares these INTENT(INOUT) and then overwrites them before any
#: read, so no entry value exists to pass.  The port does not accept them at
#: all, and that absence is asserted rather than described.
_UNREADABLE_INOUT = {
    "rb": "ENERGY zeroes RB at :2053 before RADIATION or VEGE_FLUX runs",
    "laisun": "RADIATION writes LAISUN at :2118 before any read",
    "laisha": "RADIATION writes LAISHA at :2118 before any read",
}


def test_write_before_read_inouts_are_not_arguments():
    import inspect

    params = set(inspect.signature(energy).parameters)
    for name, why in _UNREADABLE_INOUT.items():
        assert name not in params, (
            f"{name} is accepted as an argument but {why}; accepting it "
            "would invite a caller to believe the entry value is observable")
        assert hasattr(_run(CASES[0]), name)


def _perturb(value: float) -> float:
    """A relative nudge, or an absolute one when the slot is zero.

    A zero-probe is not admissible here: ENERGY aborts on ``FIRE <= 0`` and
    divides by SFCPRS, SMCMAX and RSURF, so zeroing an input can take the
    composition off a cliff rather than test it.
    """
    v = np.float32(value)
    if v == np.float32(0.0):
        return float(np.float32(1.0e-3))
    return float(np.float32(v * np.float32(1.0009765625)))


def _signature(st) -> tuple:
    out = []
    for name in _SCALARS:
        out.append(float(getattr(st, name.lower())))
    for name, lo, hi in _LAYER_ARRAYS:
        out.extend(float(v) for v in getattr(st, name.lower()))
    out.extend([float(st.frozen_canopy), float(st.frozen_ground)])
    out.extend(float(v) for v in st.imelt)
    return tuple(np.float32(v).tobytes() for v in out)


@pytest.mark.parametrize("slot", _MUTABLE_SCALARS)
def test_every_live_input_moves_an_output(slot):
    """No pinned input may be droppable.

    ``max_ulp 0`` on the fixture is not evidence the port reads its
    arguments: a sibling lane reached it on 29 columns and then found 13 of 14
    argument-drop mutants still reproduced the CSV.  This walks every live
    entry slot and requires the composition's answer to move.
    """
    moved = []
    for case in CASES:
        base = _signature(_run(case))
        value = _from_bits(FX[case]["in"][slot])
        try:
            got = _signature(_run(case, overrides={
                slot.lower(): _perturb(value)}))
        except (ValueError, ZeroDivisionError, OverflowError):
            moved.append(case)          # the abort is itself a change
            continue
        if got != base:
            moved.append(case)
    assert moved, (
        f"{slot} moved no output in any of the {len(CASES)} cases; the "
        "fixture cannot tell whether the port reads it")


@pytest.mark.parametrize("slot", sorted(_INERT))
def test_declared_inert_inputs_move_nothing(slot):
    """The inert list is a measurement, not an annotation."""
    for case in CASES:
        base = _signature(_run(case))
        probe = _perturb(_from_bits(FX[case]["in"][slot]))
        got = _signature(_run(case, overrides={slot.lower(): probe}))
        assert got == base, (
            f"{slot} is declared inert ({_INERT[slot]}) but moved an output "
            f"in {case}")


@pytest.mark.parametrize("layer_slot", ("STC", "SH2O", "SMC", "DZSNSO",
                                        "ZSNSO", "ZSOIL", "SNICE", "SNLIQ",
                                        "SOLAD", "SOLAI"))
def test_every_live_layer_input_moves_an_output(layer_slot):
    spans = {"STC": (-NSNOW + 1, NSOIL), "DZSNSO": (-NSNOW + 1, NSOIL),
             "ZSNSO": (-NSNOW + 1, NSOIL), "SNICE": (-NSNOW + 1, 0),
             "SNLIQ": (-NSNOW + 1, 0), "SH2O": (1, NSOIL),
             "SMC": (1, NSOIL), "ZSOIL": (1, NSOIL),
             "SOLAD": (1, 2), "SOLAI": (1, 2)}
    lo, hi = spans[layer_slot]
    moved = []
    for case in CASES:
        isnow = _to_int(FX[case]["in"]["ISNOW"])
        base = _signature(_run(case))
        seq = _vec(case, "in", layer_slot, lo, hi)
        for k in range(lo, hi + 1):
            if lo < 1 and k <= isnow:
                continue                 # buried layer: WRF never reads it
            probe = list(seq)
            probe[k - lo] = _perturb(seq[k - lo])
            try:
                got = _signature(_run(case, overrides={
                    layer_slot.lower(): probe}))
            except (ValueError, ZeroDivisionError, OverflowError):
                moved.append((case, k))
                continue
            if got != base:
                moved.append((case, k))
    assert moved, f"{layer_slot} moved no output anywhere"


# --------------------------------------------------------------------------
# what the pinned identity makes dead
# --------------------------------------------------------------------------
@pytest.mark.parametrize("field,dead", [
    ("opt_sfc", 2), ("opt_crs", 2), ("opt_alb", 1), ("opt_btr", 2),
    ("opt_btr", 3), ("opt_rsf", 2), ("opt_rsf", 3), ("opt_rsf", 4),
    ("opt_stc", 2), ("opt_stc", 3), ("opt_crop", 1), ("opt_irr", 1),
    ("dveg", 2),
])
def test_dead_options_are_refused_not_approximated(field, dead):
    options = EnergyOptions(**{**PINNED_OPTIONS.__dict__, field: dead})
    with pytest.raises(NotImplementedError):
        _run(CASES[0], options=options)


@pytest.mark.parametrize("slot,value", [("ice", 1), ("ist", 2)])
def test_columns_outside_the_admitted_slice_are_refused(slot, value):
    with pytest.raises(NotImplementedError):
        _run(CASES[0], overrides={slot: value})


# --------------------------------------------------------------------------
# negative controls for the two traps the module docstring names
# --------------------------------------------------------------------------
def test_uu_squared_through_powf_is_a_null_result_not_a_trap():
    """``UU**2.0`` at :2057 compiles to ``powf``, and it does not matter.

    The exponent is a REAL literal, so gfortran emits a ``powf`` call rather
    than a multiply -- ``nm -u`` on a two-line probe shows the undefined
    ``powf`` symbol.  The obvious inference is that a port writing ``uu*uu``
    would miss ``max_ulp 0``.  It would not: swept over the 167,177,618 FP32
    values in [1e-3, 1e3], glibc's ``powf(x, 2.0f)`` is **bit-identical** to
    ``x*x`` on every one.

    So this records a measured null result rather than asserting a trap that
    is not there.  The port keeps the ``powf`` spelling because it is what the
    Fortran compiles to, not because anything can see the difference; this
    test exists so the next reader does not re-derive the question, and so
    that if a future glibc changes ``powf`` the assumption is written down.
    """
    import gpuwm.core.noahmp_energy as mod

    real_powf = mod.powf

    def square_by_multiply(x, y):
        if float(y) == 2.0:
            return np.float32(np.float32(x) * np.float32(x))
        return real_powf(x, y)

    mod.powf = square_by_multiply
    try:
        differs = [case for case in CASES
                   if any(struct.pack("<f", np.float32(g)) !=
                          struct.pack("<f", np.float32(w))
                          for _, g, w in _port_pairs(case, _run(case)))]
    finally:
        mod.powf = real_powf
    assert not differs, (
        "powf(x,2.0) and x*x disagreed on a fixture column, which the "
        "167,177,618-input sweep says cannot happen; re-measure before "
        "trusting either spelling")


def test_gate_fails_if_tanh_comes_from_an_fp64_shim():
    """``TANH`` at :2072 is glibc's ``tanhf``, which an FP64 shim is not.

    Over the 204,064,836 FP32 inputs in [1e-6, 22] the two disagree on 23.8%
    of them, so a port that reaches for ``math.tanh`` is a different function.

    Getting this control to fire took a fixture change, and that is the point
    worth recording: with case 9's snow water equivalent at a round 8.0 mm,
    all four snow columns happened to land on arguments where glibc and the
    shim agree, and this test passed with the wrong ``TANH``.  ``SNEQV`` is now
    8.75 mm -- a snow density of exactly 250 kg/m3 -- which puts the FSNO
    argument on 1.75, where they differ.  ``max_ulp 0`` on nine columns is not
    evidence that a fixture can tell right from plausible.
    """
    import math

    import gpuwm.core.noahmp_energy as mod

    real_tanhf = mod.tanhf
    mod.tanhf = lambda x: np.float32(math.tanh(float(x)))
    try:
        differs = [case for case in CASES
                   if any(struct.pack("<f", np.float32(g)) !=
                          struct.pack("<f", np.float32(w))
                          for _, g, w in _port_pairs(case, _run(case)))]
    finally:
        mod.tanhf = real_tanhf
    assert differs, (
        "an FP64 tanh shim reproduced the fixture on every case; either no "
        "case carries snow or the gate is not reading FSNO")


# --------------------------------------------------------------------------
# the port against the whole-column fixture
# --------------------------------------------------------------------------
SFLX = ROOT / "gpuwm" / "data" / "noahmp" / "oracle" / "noahmp-sflx.csv"

#: ENERGY writes these and WATER then writes them again, so NOAHMP_SFLX's
#: output is not ENERGY's.  Everything else must agree bit for bit.
_REWRITTEN_DOWNSTREAM = (
    {f"sh2o_{k}" for k in range(1, NSOIL + 1)}
    | {f"smc_{k}" for k in range(1, NSOIL + 1)}
    | {f"snice_{k}" for k in range(-NSNOW + 1, 1)}
    | {f"snliq_{k}" for k in range(-NSNOW + 1, 1)}
    | {"sneqv", "sneqvo", "snowh"}
)


def _load_sflx():
    if not SFLX.exists():                                   # pragma: no cover
        pytest.skip(f"whole-column fixture missing: {SFLX}")
    table: dict = {}
    with SFLX.open(newline="") as fh:
        for row in csv.DictReader(fh):
            field = row["field"].strip().lower()
            index = row["index"].strip()
            key = field if index == "0" else f"{field}_{index}"
            table.setdefault(
                (row["case"].strip(), row["stage"].strip()), {})[key] = \
                float(row["value"])
    return table


@pytest.mark.parametrize("case", CASES[:4])
def test_port_matches_the_whole_column_fixture(case):
    """The four mirror columns, checked against a fixture built years apart.

    Cases 1-4 of the ENERGY fixture stand on the same four columns as
    ``noahmp-sflx.csv``, which pins the *whole* NOAHMP_SFLX call.  Running the
    port on the ENERGY entry state therefore has to reproduce the column's
    outputs for everything ENERGY writes and nothing downstream touches.  This
    is the check that having both fixtures buys.
    """
    column = _load_sflx()[(case, "output")]
    st = _run(case)
    matched, bad = 0, []
    for name, got, _want in _port_pairs(case, st):
        key = name.lower().replace("_+", "_")
        if key not in column or key in _REWRITTEN_DOWNSTREAM:
            continue
        if struct.pack("<f", np.float32(got)) != \
                struct.pack("<f", np.float32(column[key])):
            bad.append((key, got, column[key]))
        else:
            matched += 1
    assert not bad, (case, bad[:10])
    assert matched >= 70, (case, matched)


# --------------------------------------------------------------------------
# the staged continuation
# --------------------------------------------------------------------------
def _state_bits(st) -> dict:
    """Every field of an :class:`EnergyState`, as raw bytes."""
    out = {}
    for name in dir(st):
        if name.startswith("_"):
            continue
        value = getattr(st, name)
        if callable(value):
            continue
        out[name] = repr(value) if isinstance(value, (dict, bool)) else \
            _bytes_of(value)
    return out


def _bytes_of(value):
    if isinstance(value, (tuple, list)):
        return tuple(_bytes_of(v) for v in value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return struct.pack("<f", np.float32(value))
    return repr(value)


def _drive_interleaved(calls, *, answer=None):
    """Run several ENERGY generators in lockstep, batching each paused leaf.

    This is exactly what ``gpuwm.core.noahmp_runtime`` does around the device
    batches, with the CPython leaves in place of CUDA: every round collects the
    leaf each still-running column is paused at, evaluates those calls
    together, and resumes.  If suspension could let one column observe another
    column's partially mutated state, this would not reproduce the direct call.
    """
    from gpuwm.core.noahmp_energy import (LeafRequest, answer_on_host,
                                          energy_steps)

    if answer is None:
        answer = answer_on_host
    steps = [energy_steps(parameters, **kwargs)
             for parameters, kwargs in calls]
    requests = [gen.send(None) for gen in steps]
    results = [None] * len(steps)
    active = list(range(len(steps)))
    leaves_seen = []
    while active:
        buckets = {}
        for index in active:
            buckets.setdefault(requests[index].leaf, []).append(index)
        leaves_seen.append({leaf: len(idx) for leaf, idx in buckets.items()})
        answers = {}
        for leaf, indices in buckets.items():
            for index in indices:
                request = requests[index]
                answers[index] = answer(
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
    return results, leaves_seen


def test_the_staged_continuation_is_bitwise_identical_to_the_direct_call():
    """Suspending ENERGY at its leaves must not move a single bit.

    The runtime pauses every land column at RADIATION, at VEGE_FLUX and again
    at BARE_FLUX so the calls can be batched onto the device.  That is a
    control-flow change in the hottest 75% of the column, and the only
    acceptable evidence that it is answer-preserving is a bit comparison of the
    whole ENERGY state against the unstaged call, on columns that take
    different branches.
    """
    calls = [_call(case) for case in CASES]
    staged, leaves_seen = _drive_interleaved(calls)
    for case, got in zip(CASES, staged):
        want = _run(case)
        assert _state_bits(got) == _state_bits(want), case

    # The shape of the rounds is the claim, and it is asserted as totals
    # rather than as a fixed round count: how many rounds it takes depends on
    # how far apart a bare column and a vegetated column drift, which is a
    # scheduling detail, whereas "every column pauses at each leaf on its path
    # exactly once" is the invariant a mis-staged seam would break.
    total = {}
    for entry in leaves_seen:
        for leaf, count in entry.items():
            total[leaf] = total.get(leaf, 0) + count
    for leaf in ("thermoprop", "radiation", "bare_flux", "tsnosoi",
                 "phasechange"):
        assert total[leaf] == len(CASES), (leaf, leaves_seen)
    # VEGE_FLUX is the one leaf a column can skip.
    assert 0 < total["vege_flux"] < len(CASES), leaves_seen
    # THERMOPROP is first on every path and RADIATION second, so the first two
    # rounds are pure: nothing has drifted yet.
    assert leaves_seen[0] == {"thermoprop": len(CASES)}, leaves_seen
    assert leaves_seen[1] == {"radiation": len(CASES)}, leaves_seen
    # The vegetated columns are exactly the ones that pause a second time at
    # BARE_FLUX, one round behind the bare ones.
    assert leaves_seen[2]["vege_flux"] == total["vege_flux"], leaves_seen
    assert leaves_seen[3]["bare_flux"] == total["vege_flux"], leaves_seen


def test_the_staging_comparison_can_fail():
    """A gate that has never been observed to fail is not evidence.

    Route one column's leaf answer to the wrong column and the same comparison
    must reject it.  Without this, the test above would pass for a staging loop
    that silently paired results with the wrong columns -- which is the single
    most likely defect in a batched seam.
    """
    from gpuwm.core.noahmp_energy import answer_on_host

    calls = [_call(case) for case in CASES]
    seen = []

    def misrouted(request):
        seen.append(request)
        # Answer the VEGE_FLUX of the *previous* vegetated call once the
        # second one arrives.  Physical, well-typed, and wrong.
        if request.leaf == "vege_flux" and len(seen) > 1:
            for earlier in reversed(seen[:-1]):
                if earlier.leaf == "vege_flux":
                    return answer_on_host(earlier)
        return answer_on_host(request)

    staged, _ = _drive_interleaved(calls, answer=misrouted)
    differing = [case for case, got in zip(CASES, staged)
                 if _state_bits(got) != _state_bits(_run(case))]
    assert differing, (
        "misrouting a VEGE_FLUX result between columns went undetected; the "
        "staging comparison is not sensitive to result/column pairing")
