"""max_ulp 0 gate for the Noah-MP VEGE_FLUX subtree against the WRF v4.6.1 oracle.

The fixture in ``gpuwm/data/noahmp/oracle/noahmp-vegeflux.csv`` was produced by
``tools/noahmp_wrf461_oracle/build_vegeflux.sh`` from the pinned WRF tree with
nothing but a ``private::`` -> ``public::`` accessibility lift.  Every value is
stored as its IEEE-754 binary32 bit pattern, so equality here is bitwise; there
is no tolerance to widen.
"""

from __future__ import annotations

import csv
import pathlib
import struct

import numpy as np
import pytest

from gpuwm.core.fp32_ulp import max_ulp
from gpuwm.core.noahmp_vegeflux import (
    R4, VegeFluxParameters, esat, ragrb, sfcdif1, stomata, vege_flux,
)

FIXTURE = (pathlib.Path(__file__).resolve().parents[1]
           / "gpuwm" / "data" / "noahmp" / "oracle" / "noahmp-vegeflux.csv")

NSNOW = 3
NSOIL = 4


# --------------------------------------------------------------------------
def _bits(x) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _from_bits(h: str) -> float:
    return struct.unpack("<f", struct.pack("<I", int(h, 16)))[0]


def _ulp_gap(a: int, b: int) -> int:
    """Distance in the FP32 total order.

    Delegates to ``gpuwm.core.fp32_ulp``; the map is derived once in the tree
    and a second, subtly different copy here is exactly how two gates end up
    measuring different things while both report zero.
    """
    fa = np.frombuffer(struct.pack("<I", a & 0xFFFFFFFF), dtype=np.float32)
    fb = np.frombuffer(struct.pack("<I", b & 0xFFFFFFFF), dtype=np.float32)
    return max_ulp(fa, fb)


def load_fixture():
    """Return {leaf: {case: {'in': {...}, 'out': {...}, 'opt': {...}}}}."""
    if not FIXTURE.exists():
        pytest.skip(f"oracle fixture missing: {FIXTURE}")
    table: dict = {}
    with FIXTURE.open(newline="") as fh:
        for row in csv.DictReader(fh):
            leaf = table.setdefault(row["leaf"], {})
            case = leaf.setdefault(row["case"], {"in": {}, "out": {}, "opt": {}})
            case[row["role"]][row["name"]] = row["hex"]
    return table


TABLE = load_fixture()


def _f(case, role, name) -> float:
    return _from_bits(case[role][name])


def _i(case, role, name) -> int:
    v = int(case[role][name], 16)
    return v - (1 << 32) if v >> 31 else v


def _check(results, case, leaf, tag, integer_names=()):
    """Every recorded output must match bit for bit (max_ulp 0)."""
    bad = []
    for name, want_hex in case["out"].items():
        if name not in results:
            continue
        got = _bits(results[name])
        want = int(want_hex, 16)
        if got != want:
            bad.append(f"{name}: got 0x{got:08x} want 0x{want:08x} "
                       f"(ulp gap {_ulp_gap(got, want)})")
    assert not bad, f"{leaf}/{tag} max_ulp != 0:\n  " + "\n  ".join(bad)
    missing = sorted(set(case["out"]) - set(results) - set(integer_names))
    assert not missing, (
        f"{leaf}/{tag}: transcription did not produce {missing}")


# --------------------------------------------------------------------------
def test_option_identity_is_the_registry_default():
    opts = TABLE["options"]["registry"]["opt"]
    got = {k: int(v, 16) for k, v in opts.items()}
    assert got == {
        "DVEG": 4, "OPT_CRS": 1, "OPT_BTR": 1, "OPT_RUN": 3, "OPT_SFC": 1,
        "OPT_STC": 1, "OPT_CROP": 0, "OPT_IRR": 0, "OPT_TDRN": 0,
        "OPT_ALB": 2, "OPT_RAD": 3,
    }


@pytest.mark.parametrize("tag", sorted(TABLE.get("esat", {})))
def test_esat_max_ulp_zero(tag):
    case = TABLE["esat"][tag]
    esw, esi, desw, desi = esat(_f(case, "in", "T"))
    _check({"ESW": esw, "ESI": esi, "DESW": desw, "DESI": desi},
           case, "esat", tag)


@pytest.mark.parametrize("tag", sorted(TABLE.get("ragrb", {})))
def test_ragrb_max_ulp_zero(tag):
    case = TABLE["ragrb"][tag]
    mozg, fhg, ramg, rahg, rawg, rb = ragrb(
        _i(case, "in", "ITER"), _f(case, "in", "VAI"), _f(case, "in", "RHOAIR"),
        _f(case, "in", "HG"), _f(case, "in", "TAH"), _f(case, "in", "ZPD"),
        _f(case, "in", "Z0MG"), _f(case, "in", "Z0HG"), _f(case, "in", "HCAN"),
        _f(case, "in", "UC"), _f(case, "in", "Z0H"), _f(case, "in", "FV"),
        _f(case, "in", "CWP"), _f(case, "in", "MPE"), _f(case, "in", "TV"),
        _f(case, "in", "MOZG"), _f(case, "in", "FHG"),
        _f(case, "in", "P_DLEAF"))
    _check({"MOZG": mozg, "FHG": fhg, "RAMG": ramg, "RAHG": rahg,
            "RAWG": rawg, "RB": rb}, case, "ragrb", tag)


@pytest.mark.parametrize("tag", sorted(TABLE.get("sfcdif1", {})))
def test_sfcdif1_max_ulp_zero(tag):
    case = TABLE["sfcdif1"][tag]
    moz, mozsgn, fm, fh, fm2, fh2, cm, ch, fv, ch2 = sfcdif1(
        _i(case, "in", "ITER"), _f(case, "in", "SFCTMP"),
        _f(case, "in", "RHOAIR"), _f(case, "in", "H"), _f(case, "in", "QAIR"),
        _f(case, "in", "ZLVL"), _f(case, "in", "ZPD"), _f(case, "in", "Z0M"),
        _f(case, "in", "Z0H"), _f(case, "in", "UR"), _f(case, "in", "MPE"),
        _f(case, "in", "MOZ"), _i(case, "in", "MOZSGN"), _f(case, "in", "FM"),
        _f(case, "in", "FH"), _f(case, "in", "FM2"), _f(case, "in", "FH2"),
        _f(case, "in", "FV"))
    assert mozsgn == _i(case, "out", "MOZSGN")
    _check({"MOZ": moz, "FM": fm, "FH": fh, "FM2": fm2, "FH2": fh2,
            "CM": cm, "CH": ch, "FV": fv, "CH2": ch2},
           case, "sfcdif1", tag, integer_names=("MOZSGN",))


def _params_from(case, prefix="P_") -> VegeFluxParameters:
    kw = {}
    for name in VegeFluxParameters.__dataclass_fields__:
        key = prefix + name
        if key in case["in"]:
            kw[name] = R4(_f(case, "in", key))
    return VegeFluxParameters(**kw)


@pytest.mark.parametrize("tag", sorted(TABLE.get("stomata", {})))
def test_stomata_max_ulp_zero(tag):
    case = TABLE["stomata"][tag]
    p = _params_from(case)
    rs, psn = stomata(
        p, _f(case, "in", "MPE"), _f(case, "in", "APAR"),
        _f(case, "in", "FOLN"), _f(case, "in", "TV"), _f(case, "in", "EI"),
        _f(case, "in", "EA"), _f(case, "in", "SFCTMP"),
        _f(case, "in", "SFCPRS"), _f(case, "in", "FVEG"), _f(case, "in", "O2"),
        _f(case, "in", "CO2"), _f(case, "in", "IGS"), _f(case, "in", "BTRAN"),
        _f(case, "in", "RB"))
    _check({"RS": rs, "PSN": psn}, case, "stomata", tag)


def call_vege_flux(case):
    p = _params_from(case)
    dz = {k: _f(case, "in", f"DZSNSO_{k:+d}") for k in range(-NSNOW + 1, NSOIL + 1)}
    stc = {k: _f(case, "in", f"STC_{k:+d}") for k in range(-NSNOW + 1, NSOIL + 1)}
    df = {k: _f(case, "in", f"DF_{k:+d}") for k in range(-NSNOW + 1, NSOIL + 1)}
    return vege_flux(
        p, NSNOW, NSOIL, _i(case, "in", "ISNOW"),
        _f(case, "in", "DT"), _f(case, "in", "SAV"), _f(case, "in", "SAG"),
        _f(case, "in", "LWDN"), _f(case, "in", "UR"), _f(case, "in", "UU"),
        _f(case, "in", "VV"), _f(case, "in", "SFCTMP"), _f(case, "in", "THAIR"),
        _f(case, "in", "QAIR"), _f(case, "in", "EAIR"), _f(case, "in", "RHOAIR"),
        _f(case, "in", "SNOWH"), _f(case, "in", "VAI"), _f(case, "in", "GAMMAV"),
        _f(case, "in", "GAMMAG"), _f(case, "in", "FWET"), _f(case, "in", "LAISUN"),
        _f(case, "in", "LAISHA"), _f(case, "in", "CWP"), dz,
        _f(case, "in", "ZLVL"), _f(case, "in", "ZPD"), _f(case, "in", "Z0M"),
        _f(case, "in", "FVEG"), _f(case, "in", "Z0MG"), _f(case, "in", "EMV"),
        _f(case, "in", "EMG"), _f(case, "in", "CANLIQ"), _f(case, "in", "FSNO"),
        _f(case, "in", "CANICE"), stc, df, _f(case, "in", "RSURF"),
        _f(case, "in", "LATHEAV"), _f(case, "in", "LATHEAG"),
        _f(case, "in", "PARSUN"), _f(case, "in", "PARSHA"), _f(case, "in", "IGS"),
        _f(case, "in", "FOLN"), _f(case, "in", "CO2AIR"), _f(case, "in", "O2AIR"),
        _f(case, "in", "BTRAN"), _f(case, "in", "SFCPRS"), _f(case, "in", "RHSUR"),
        _f(case, "in", "Q2"), _f(case, "in", "PAHV"), _f(case, "in", "PAHG"),
        _f(case, "in", "EAH"), _f(case, "in", "TAH"), _f(case, "in", "TV"),
        _f(case, "in", "TG"), _f(case, "in", "CM"), _f(case, "in", "CH"),
        _f(case, "in", "QC"), _f(case, "in", "QSFC"), _f(case, "in", "PSFC"),
        _f(case, "in", "FSR"))


@pytest.mark.parametrize("tag", sorted(TABLE.get("vegeflux", {})))
def test_vege_flux_max_ulp_zero(tag):
    case = TABLE["vegeflux"][tag]
    st = call_vege_flux(case)
    got = {name: getattr(st, name) for name in case["out"] if hasattr(st, name)}
    _check(got, case, "vegeflux", tag)


def test_dead_branches_are_refused_not_guessed():
    """The pinned option identity kills SFCDIF2, CANRES, gecros and opt_stc=3.

    None of them is transcribed; asking for them must raise rather than return
    a number nothing verified.
    """
    case = TABLE["vegeflux"]["vegeflux01"]
    p = _params_from(case)
    dz = {k: 0.1 for k in range(-NSNOW + 1, NSOIL + 1)}
    base = dict(p=p, nsnow=NSNOW, nsoil=NSOIL, isnow=0)
    for kw in ({"opt_sfc": 2}, {"opt_crs": 2}, {"opt_crop": 2}, {"opt_stc": 3}):
        with pytest.raises(NotImplementedError):
            vege_flux(
                base["p"], NSNOW, NSOIL, 0,
                30.0, 320.0, 90.0, 355.0, 4.2, 3.4, -2.5, 295.0, 296.4,
                0.0102, 1580.0, 1.1745, 0.0, 3.5, 65.0, 65.0, 0.12, 1.4, 2.1,
                3.0, dz, 30.0, 6.7, 1.0, 0.78, 0.01, 0.95, 0.96, 0.35, 0.0,
                0.0, dz, dz, 90.0, 2.5104E06, 2.5104E06, 260.0, 60.0, 1.0,
                1.0, 38.0, 20460.0, 0.8, 97000.0, 0.92, 0.0102, 0.0, 0.0,
                1600.0, 295.5, 297.0, 296.0, 0.02, 0.02, 0.0, 0.0102,
                97000.0, 0.0, **kw)


# --------------------------------------------------------------------------
# the arithmetic carrier itself
# --------------------------------------------------------------------------
#: A binary32 corpus chosen for the places FP32 arithmetic goes wrong, not for
#: coverage of the physical range.  Both zeros; the smallest and largest
#: subnormals; the subnormal/normal boundary from both sides; one ULP either
#: side of a power of two; exact halfway cases between adjacent binary32
#: values, which is where round-to-nearest-EVEN differs from
#: round-half-away; the binary32 extremes; binary64 values that overflow
#: binary32; both infinities; and a NaN.  The Noah-MP physical constants are
#: in there too so the sweep is not purely pathological.
def _r4_corpus():
    import math

    tiny = 2.0 ** -149                      # smallest positive subnormal
    values = [
        0.0, -0.0, tiny, -tiny, 2.0 ** -148, 3.0 * tiny,
        (2.0 ** -126) * (1.0 - 2.0 ** -24),  # largest subnormal
        2.0 ** -126, -(2.0 ** -126),         # subnormal/normal boundary
        1.0, -1.0, 2.0, 0.5, 4.0,
        1.0 + 2.0 ** -23, 1.0 - 2.0 ** -24,  # one ULP either side of 1
        1.0 + 2.0 ** -24,                    # exact tie: rounds to 1.0 (even)
        1.0 + 3.0 * 2.0 ** -24,              # exact tie: rounds up
        3.4028234663852886e38,               # largest finite binary32
        -3.4028234663852886e38,
        3.4028235677973366e38,               # first binary64 that overflows
        1e39, -1e39, 1e300, 1e-320,
        math.inf, -math.inf, math.nan,
        0.1, 0.622, 0.0098, 273.16, 1004.64, 4.188e6, 9.80616, 5.67e-8,
        0.4, 1000.0, 917.0, 2.5104e6, 2.8348e6, 1.0e-6, 1.0e6,
    ]
    # Adjacent binary32 neighbours of a handful of them, which is where a
    # carrier that rounds the wrong operand first shows up.
    out = list(values)
    for v in (1.0, 273.16, 0.622, 2.0 ** -126, 3.4028234663852886e38):
        b = _bits(v)
        for delta in (-1, 1):
            out.append(_from_bits("%08x" % ((b + delta) & 0xFFFFFFFF)))
    # And the binary64 midpoints between adjacent binary32 values, the inputs
    # a single rounding and a double rounding disagree about.
    for v in (1.0, 273.16, 0.1, 1004.64, 2.0 ** -126, tiny):
        b = _bits(v)
        lo = _from_bits("%08x" % b)
        hi = _from_bits("%08x" % (b + 1))
        out.append((lo + hi) * 0.5)
        out.append((lo + hi) * 0.5 * (1.0 + 2.0 ** -52))
        out.append((lo + hi) * 0.5 * (1.0 - 2.0 ** -52))
    # A deterministic spread of binary32 bit patterns, so the sweep is not
    # only the values a human thought of.  The seed is fixed: a sweep whose
    # corpus moves between runs cannot be cited in a docstring.  Exponents are
    # drawn across the whole range including the subnormal band, because the
    # subnormal band is where this port has actually been bitten.
    rng = np.random.default_rng(20260726)
    for exponent in range(0, 255, 6):
        for _ in range(6):
            mantissa = int(rng.integers(0, 1 << 23))
            sign = int(rng.integers(0, 2)) << 31
            bits = sign | (exponent << 23) | mantissa
            if (bits >> 23) & 0xFF == 0xFF:          # keep inf/NaN scripted
                continue
            out.append(_from_bits("%08x" % bits))
    return tuple(out)


_R4_CORPUS = _r4_corpus()

_R4_OPS = (
    ("add", lambda a, b: a + b),
    ("sub", lambda a, b: a - b),
    ("mul", lambda a, b: a * b),
    ("div", lambda a, b: a / b),
)


def _reference(op, lhs, rhs):
    """The three-call spelling ``R4`` used before the operators were flattened.

    Deliberately written out rather than imported: this is the definition the
    fast operators must reproduce, and if it lived in the module it would be
    changed by the same edit it is supposed to police.
    """
    from gpuwm.core.noahmp_libm import f32
    a = f32(lhs)
    b = f32(rhs)
    if op == "add":
        return f32(float.__add__(a, b))
    if op == "sub":
        return f32(float.__sub__(a, b))
    if op == "mul":
        return f32(float.__mul__(a, b))
    return f32(float.__truediv__(a, b))


def _outcome(fn):
    """(kind, payload) so a raise compares equal to a raise."""
    try:
        value = fn()
    except OverflowError:
        return ("OverflowError", None)
    except ZeroDivisionError:
        return ("ZeroDivisionError", None)
    if value != value:
        return ("nan", None)
    return ("value", _bits(value))


def test_the_fast_r4_operators_are_the_three_call_spelling():
    """Every ``(op, lhs, rhs)`` over the corpus, both operand orders.

    ``R4.__mul__`` and friends round the right-hand operand in place and then
    inline ``f32`` as ``struct`` pack/unpack.  That is a performance rewrite of
    the hottest code in the Noah-MP host column, so it gets held to the
    spelling it replaced rather than to a tolerance.  A mismatch here is a
    changed forecast.
    """
    lefts = []
    rejected = 0
    for lhs in _R4_CORPUS:
        try:
            lefts.append(R4(lhs))
        except OverflowError:
            # A binary64 outside binary32 cannot BE an R4; it can only be a
            # right-hand operand, and it stays in the corpus as one.
            rejected += 1
    assert rejected >= 4, (
        "the corpus lost its out-of-range values, so the overflow contract is "
        "no longer swept")

    pairs = 0
    mismatches = []
    for name, op in _R4_OPS:
        for left in lefts:
            for rhs in _R4_CORPUS:
                pairs += 1
                want = _outcome(lambda: _reference(name, float(left), rhs))
                got = _outcome(lambda: op(left, rhs))
                if want != got:
                    mismatches.append((name, float(left), rhs, want, got))
    assert pairs > 60_000, f"the sweep shrank to {pairs} pairs"
    assert not mismatches, (
        f"{len(mismatches)} of {pairs} pairs disagree with the three-call "
        f"spelling; first five: {mismatches[:5]}")


def test_the_r4_sweep_catches_a_carrier_that_is_wrong():
    """The negative control.  A gate that has never failed is not evidence.

    Two plausible-looking carriers are run through the same sweep:

    * one that flushes subnormals to zero, which is what ``-ftz=true`` does on
      the device and has already caused two real bugs in this port;
    * one that rounds the LEFT operand to binary32 after the operation instead
      of the right one before it, which is the mistake the flattening could
      have made and which agrees on every value whose product is exact.

    Both must be caught, or the sweep above proves nothing.
    """
    from gpuwm.core.noahmp_libm import f32

    def ftz(op, lhs, rhs):
        def flush(x):
            y = f32(x)
            return 0.0 if 0.0 < abs(y) < 2.0 ** -126 else y
        a, b = flush(lhs), flush(rhs)
        raw = {"add": float.__add__, "sub": float.__sub__,
               "mul": float.__mul__, "div": float.__truediv__}[op](a, b)
        return flush(raw)

    def unrounded_rhs(op, lhs, rhs):
        a = f32(lhs)
        raw = {"add": float.__add__, "sub": float.__sub__,
               "mul": float.__mul__, "div": float.__truediv__}[op](a, rhs)
        return f32(raw)

    for label, wrong in (("flush-to-zero", ftz),
                         ("right operand unrounded", unrounded_rhs)):
        caught = 0
        for name, _ in _R4_OPS:
            for lhs in _R4_CORPUS:
                for rhs in _R4_CORPUS:
                    want = _outcome(lambda: _reference(name, lhs, rhs))
                    got = _outcome(lambda: wrong(name, lhs, rhs))
                    if want != got:
                        caught += 1
        assert caught > 0, f"the sweep did not notice {label}"
