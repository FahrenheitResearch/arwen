"""max_ulp-0 gate for the Noah-MP shortwave-radiation leaves.

Every row of every fixture under ``gpuwm/data/noahmp/oracle/`` was produced by
the unmodified (visibility-patched only) WRF-4.6.1 module; see
``tools/noahmp_wrf461_oracle/build_radiation.sh``.  Inputs and outputs are
stored as IEEE-754 binary32 bit patterns, so nothing here parses a decimal.

The gate is bitwise identity, asserted through
:func:`gpuwm.core.fp32_ulp.assert_bit_exact`, which is strictly stronger than
``max_ulp == 0`` (it also separates -0.0 from +0.0).
"""

from __future__ import annotations

import csv
import struct
from pathlib import Path

import numpy as np
import pytest

from gpuwm.core.fp32_ulp import assert_bit_exact, max_ulp
from gpuwm.core.noahmp_radiation import (
    PINNED_OPTIONS,
    albedo,
    groundalb,
    snow_age,
    snowalb_class,
    surrad,
    twostream,
)

ORACLE = Path(__file__).resolve().parents[1] / "gpuwm" / "data" / "noahmp" / "oracle"

F = np.float32


def _f(hexstr: str) -> np.float32:
    """float32 from the fixture's ``0xXXXXXXXX`` bit pattern."""
    return np.float32(struct.unpack("<f", struct.pack("<I", int(hexstr, 16)))[0])


def _rows(leaf: str):
    path = ORACLE / f"noahmp-radiation-{leaf}.csv"
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows, f"{path} is empty"
    return rows


def _vec(row, base: str, n: int = 2):
    return tuple(_f(row[f"{base}{i}"]) for i in range(1, n + 1))


# --------------------------------------------------------------------------
# leaf 1: SNOW_AGE
# --------------------------------------------------------------------------
def test_snow_age_bit_exact():
    rows = _rows("snow_age")
    assert len(rows) >= 12
    got_t, got_f, want_t, want_f = [], [], [], []
    for r in rows:
        t, fa = snow_age(
            _f(r["tau0"]), _f(r["grain_growth"]), _f(r["extra_growth"]),
            _f(r["dirt_soot"]), _f(r["swemx"]),
            _f(r["dt"]), _f(r["tg"]), _f(r["sneqvo"]), _f(r["sneqv"]),
            _f(r["tauss_in"]),
        )
        got_t.append(t); got_f.append(fa)
        want_t.append(_f(r["tauss_out"])); want_f.append(_f(r["fage"]))
    assert_bit_exact(np.array(got_t, F), np.array(want_t, F), "snow_age.tauss")
    assert_bit_exact(np.array(got_f, F), np.array(want_f, F), "snow_age.fage")


def test_snow_age_binds_every_branch():
    """Each documented branch is bound by at least one fixture row."""
    rows = _rows("snow_age")
    seen = {"nosnow": 0, "arg_neg": 0, "arg_pos": 0, "arg_zero": 0,
            "accum": 0, "noaccum": 0, "sge_clamped": 0}
    for r in rows:
        sneqv, sneqvo = _f(r["sneqv"]), _f(r["sneqvo"])
        tg = _f(r["tg"])
        if sneqv <= 0:
            seen["nosnow"] += 1
            continue
        if tg < F(273.16):
            seen["arg_neg"] += 1
        elif tg > F(273.16):
            seen["arg_pos"] += 1
        else:
            seen["arg_zero"] += 1
        if sneqv > sneqvo:
            seen["accum"] += 1
        else:
            seen["noaccum"] += 1
        if F(sneqv - sneqvo) > _f(r["swemx"]):
            seen["sge_clamped"] += 1
    missing = [k for k, v in seen.items() if v == 0]
    assert not missing, f"unbound SNOW_AGE branches: {missing}"


# --------------------------------------------------------------------------
# leaf 2: SNOWALB_CLASS
# --------------------------------------------------------------------------
def test_snowalb_class_bit_exact():
    rows = _rows("snowalb_class")
    got, want = [], []
    for r in rows:
        alb, albsnd, albsni = snowalb_class(
            _f(r["swemx"]), int(r["nband"]), _f(r["qsnow"]), _f(r["dt"]),
            _f(r["albold"]),
        )
        got.extend([alb, albsnd[0], albsnd[1], albsni[0], albsni[1]])
        want.extend([_f(r["alb"]), _f(r["albsnd1"]), _f(r["albsnd2"]),
                     _f(r["albsni1"]), _f(r["albsni2"])])
    assert_bit_exact(np.array(got, F), np.array(want, F), "snowalb_class")


def test_snowalb_class_binds_qsnow_branches():
    rows = _rows("snowalb_class")
    pos = [r for r in rows if _f(r["qsnow"]) > 0]
    zero = [r for r in rows if _f(r["qsnow"]) <= 0]
    assert pos and zero
    # MIN(QSNOW, SWEMX/DT) must pick each arm at least once.
    picks_qsnow = [r for r in pos if _f(r["qsnow"]) < F(_f(r["swemx"]) / _f(r["dt"]))]
    picks_cap = [r for r in pos if _f(r["qsnow"]) > F(_f(r["swemx"]) / _f(r["dt"]))]
    assert picks_qsnow and picks_cap


def test_snowalb_class_rejects_unreachable_nband():
    with pytest.raises(ValueError):
        snowalb_class(1.0, 1, 0.0, 90.0, 0.5)


# --------------------------------------------------------------------------
# leaf 3: GROUNDALB
# --------------------------------------------------------------------------
def _groundalb_row(r):
    return groundalb(
        albsat=(_f(r["albsat1"]), _f(r["albsat2"])),
        albdry=(_f(r["albdry1"]), _f(r["albdry2"])),
        alblak=(_f(r["alblak1"]), _f(r["alblak2"])),
        nsoil=int(r["nsoil"]), nband=int(r["nband"]),
        ice=int(r["ice"]), ist=int(r["ist"]),
        fsno=_f(r["fsno"]), smc=(_f(r["smc1"]),),
        albsnd=_vec(r, "albsnd"), albsni=_vec(r, "albsni"),
        cosz=_f(r["cosz"]), tg=_f(r["tg"]),
    )


def test_groundalb_bit_exact():
    rows = _rows("groundalb")
    got, want = [], []
    for r in rows:
        albgrd, albgri = _groundalb_row(r)
        got.extend([*albgrd, *albgri])
        want.extend([*_vec(r, "albgrd"), *_vec(r, "albgri")])
    assert_bit_exact(np.array(got, F), np.array(want, F), "groundalb")


def test_groundalb_binds_every_branch():
    rows = _rows("groundalb")
    tfrz = F(273.16)
    soil = [r for r in rows if int(r["ist"]) == 1]
    lake_warm = [r for r in rows if int(r["ist"]) != 1 and _f(r["tg"]) > tfrz]
    lake_cold = [r for r in rows if int(r["ist"]) != 1 and _f(r["tg"]) <= tfrz]
    assert soil and lake_warm and lake_cold
    # INC clamp: both arms of MAX(0.11 - 0.40*SMC1, 0)
    inc_pos = [r for r in soil if F(F(0.11) - F(F(0.40) * _f(r["smc1"]))) > 0]
    inc_zero = [r for r in soil if F(F(0.11) - F(F(0.40) * _f(r["smc1"]))) <= 0]
    assert inc_pos and inc_zero
    # MIN(ALBSAT+INC, ALBDRY): both arms
    picks_sat, picks_dry = [], []
    for r in soil:
        inc = max(F(F(0.11) - F(F(0.40) * _f(r["smc1"]))), F(0.0))
        for b in (1, 2):
            if F(_f(r[f"albsat{b}"]) + inc) < _f(r[f"albdry{b}"]):
                picks_sat.append(r)
            else:
                picks_dry.append(r)
    assert picks_sat and picks_dry
    # MAX(0.01, COSZ): both arms, in the powf branch
    assert any(_f(r["cosz"]) < F(0.01) for r in lake_warm)
    assert any(_f(r["cosz"]) > F(0.01) for r in lake_warm)


# --------------------------------------------------------------------------
# leaf 4: SURRAD
# --------------------------------------------------------------------------
def _surrad_row(r):
    return surrad(
        _f(r["mpe"]), _f(r["fsun"]), _f(r["fsha"]), _f(r["elai"]), _f(r["vai"]),
        _f(r["laisun"]), _f(r["laisha"]),
        _vec(r, "solad"), _vec(r, "solai"), _vec(r, "fabd"), _vec(r, "fabi"),
        _vec(r, "ftdd"), _vec(r, "ftid"), _vec(r, "ftii"),
        _vec(r, "albgrd"), _vec(r, "albgri"), _vec(r, "albd"), _vec(r, "albi"),
        _vec(r, "frevd"), _vec(r, "frevi"), _vec(r, "fregd"), _vec(r, "fregi"),
    )


_SURRAD_OUT = ("parsun", "parsha", "sav", "sag", "fsa", "fsr", "fsrv", "fsrg")


def test_surrad_bit_exact():
    rows = _rows("surrad")
    got, want = [], []
    for r in rows:
        got.extend(_surrad_row(r))
        want.extend(_f(r[k]) for k in _SURRAD_OUT)
    assert_bit_exact(np.array(got, F), np.array(want, F), "surrad")


def test_surrad_binds_every_branch():
    rows = _rows("surrad")
    assert any(_f(r["fsun"]) > 0 for r in rows)
    assert any(_f(r["fsun"]) <= 0 for r in rows)
    assert any(_f(r["vai"]) < _f(r["mpe"]) for r in rows)      # MAX(VAI, MPE)
    assert any(_f(r["vai"]) > _f(r["mpe"]) for r in rows)
    assert any(_f(r["laisun"]) < _f(r["mpe"]) for r in rows)   # MAX(LAISUN, MPE)
    assert any(_f(r["laisun"]) > _f(r["mpe"]) for r in rows)
    assert any(_f(r["laisha"]) < _f(r["mpe"]) for r in rows)   # MAX(LAISHA, MPE)
    assert any(_f(r["laisha"]) > _f(r["mpe"]) for r in rows)


# --------------------------------------------------------------------------
# leaf 5: TWOSTREAM
# --------------------------------------------------------------------------
_TS_OUT = ("fab1", "fab2", "fre1", "fre2", "ftd1", "ftd2", "fti1", "fti2",
           "gdir", "frev1", "frev2", "freg1", "freg2", "bgap", "wgap")


def _twostream_row(r, options=PINNED_OPTIONS):
    fab, fre, ftd, fti, gdir, frev, freg, bgap, wgap = twostream(
        xl=_f(r["xl"]), omegas=_vec(r, "omegas"),
        betads=_f(r["betads"]), betais=_f(r["betais"]),
        ib=int(r["ib"]), ic=int(r["ic"]),
        cosz=_f(r["cosz"]), vai=_f(r["vai"]), fwet=_f(r["fwet"]),
        t=_f(r["t"]),
        albgrd=_vec(r, "albgrd"), albgri=_vec(r, "albgri"),
        rho=_vec(r, "rho"), tau=_vec(r, "tau"), fveg=_f(r["fveg"]),
        fab=_vec(r, "fab_in"), fre=_vec(r, "fre_in"),
        ftd=_vec(r, "ftd_in"), fti=_vec(r, "fti_in"),
        gdir=_f(r["gdir_in"]),
        frev=_vec(r, "frev_in"), freg=_vec(r, "freg_in"),
        bgap=_f(r["bgap_in"]), wgap=_f(r["wgap_in"]),
        options=options,
    )
    return [fab[0], fab[1], fre[0], fre[1], ftd[0], ftd[1], fti[0], fti[1],
            gdir, frev[0], frev[1], freg[0], freg[1], bgap, wgap]


def test_twostream_bit_exact():
    rows = _rows("twostream")
    assert len(rows) >= 24
    got, want = [], []
    for r in rows:
        got.extend(_twostream_row(r))
        want.extend(_f(r[k]) for k in _TS_OUT)
    assert_bit_exact(np.array(got, F), np.array(want, F), "twostream")


def test_twostream_binds_every_branch():
    rows = _rows("twostream")
    tfrz = F(273.16)
    assert any(_f(r["vai"]) == 0 for r in rows)           # GAP = KOPEN = 1
    assert any(_f(r["vai"]) != 0 for r in rows)           # GAP = 1 - FVEG
    assert any(_f(r["t"]) > tfrz for r in rows)           # no intercepted snow
    assert any(_f(r["t"]) <= tfrz for r in rows)          # intercepted snow
    assert any(int(r["ic"]) == 0 for r in rows)
    assert any(int(r["ic"]) == 1 for r in rows)
    assert any(int(r["ib"]) == 1 for r in rows)
    assert any(int(r["ib"]) == 2 for r in rows)
    assert any(_f(r["cosz"]) < F(0.001) for r in rows)    # COSZI clamp
    assert any(_f(r["cosz"]) > F(0.001) for r in rows)
    assert any(_f(r["xl"]) < F(-0.4) for r in rows)       # CHIL low clamp
    assert any(_f(r["xl"]) > F(0.6) for r in rows)        # CHIL high clamp
    assert any(abs(min(max(_f(r["xl"]), F(-0.4)), F(0.6))) <= F(0.01)
               for r in rows)                            # CHIL -> 0.01
    assert any(_f(r["fveg"]) == F(1.0) for r in rows)     # GAP = 0


def test_twostream_rejects_dead_opt_rad():
    from gpuwm.core.noahmp_radiation import RadiationOptions
    r = _rows("twostream")[0]
    for dead in (1, 2):
        with pytest.raises(NotImplementedError):
            _twostream_row(r, RadiationOptions(opt_rad=dead, opt_alb=2))


# --------------------------------------------------------------------------
# leaf 6: ALBEDO
# --------------------------------------------------------------------------
_ALB_SCALARS = ("fage", "albold", "tauss", "fsun", "bgap", "wgap")
_ALB_VECTORS = ("albgrd", "albgri", "albd", "albi", "fabd", "fabi",
                "ftdd", "ftid", "ftii", "frevd", "frevi", "fregd", "fregi",
                "albsnd", "albsni")


def _albedo_row(r, options=PINNED_OPTIONS):
    return albedo(
        tau0=_f(r["tau0"]), grain_growth=_f(r["grain_growth"]),
        extra_growth=_f(r["extra_growth"]), dirt_soot=_f(r["dirt_soot"]),
        swemx=_f(r["swemx"]),
        albsat=_vec(r, "albsat"), albdry=_vec(r, "albdry"),
        alblak=_vec(r, "alblak"),
        rhol=_vec(r, "rhol"), rhos=_vec(r, "rhos"),
        taul=_vec(r, "taul"), taus=_vec(r, "taus"), xl=_f(r["xl"]),
        omegas=_vec(r, "omegas"), betads=_f(r["betads"]), betais=_f(r["betais"]),
        vegtyp=int(r["vegtyp"]), ist=int(r["ist"]), ice=int(r["ice"]),
        nsoil=int(r["nsoil"]),
        dt=_f(r["dt"]), cosz=_f(r["cosz"]), fage=_f(r["fage_in"]),
        elai=_f(r["elai"]), esai=_f(r["esai"]),
        tg=_f(r["tg"]), tv=_f(r["tv"]), snowh=_f(r["snowh"]),
        fsno=_f(r["fsno"]), fwet=_f(r["fwet"]),
        smc=_vec(r, "smc", 4),
        sneqvo=_f(r["sneqvo"]), sneqv=_f(r["sneqv"]), qsnow=_f(r["qsnow"]),
        fveg=_f(r["fveg"]),
        albold=_f(r["albold_in"]), tauss=_f(r["tauss_in"]),
        frevd_in=_vec(r, "frevd_in"), frevi_in=_vec(r, "frevi_in"),
        fregd_in=_vec(r, "fregd_in"), fregi_in=_vec(r, "fregi_in"),
        options=options,
    )


def test_albedo_bit_exact():
    rows = _rows("albedo")
    assert len(rows) >= 10
    got, want = [], []
    for r in rows:
        out = _albedo_row(r)
        for k in _ALB_SCALARS:
            got.append(out[k]); want.append(_f(r[k]))
        for k in _ALB_VECTORS:
            got.extend(out[k])
            want.extend(_vec(r, k))
    assert_bit_exact(np.array(got, F), np.array(want, F), "albedo")
    assert max_ulp(np.array(got, F), np.array(want, F)) == 0


def test_albedo_binds_every_branch():
    rows = _rows("albedo")
    tfrz = F(273.16)
    assert any(_f(r["cosz"]) == 0 for r in rows)          # COSZ == 0 exit
    assert any(_f(r["cosz"]) < 0 for r in rows)           # COSZ < 0 exit
    assert any(_f(r["cosz"]) > 0 for r in rows)
    assert any(int(r["ist"]) == 1 for r in rows)
    assert any(int(r["ist"]) == 2 for r in rows)
    assert any(_f(r["tv"]) > tfrz for r in rows)
    assert any(_f(r["tv"]) <= tfrz for r in rows)
    assert any(_f(r["qsnow"]) > 0 for r in rows)
    assert any(_f(r["sneqv"]) > 0 for r in rows)          # SNOW_AGE live branch
    assert any(_f(r["sneqv"]) <= 0 for r in rows)         # SNOW_AGE reset
    # FSUN < 0.01 clamp: at least one daytime row lands on zero
    day = [r for r in rows if _f(r["cosz"]) > 0]
    assert any(_f(r["fsun"]) == 0 for r in day)
    assert any(_f(r["fsun"]) > 0 for r in day)


def test_albedo_rejects_dead_opt_alb():
    from gpuwm.core.noahmp_radiation import RadiationOptions
    r = _rows("albedo")[2]
    with pytest.raises(NotImplementedError):
        _albedo_row(r, RadiationOptions(opt_rad=3, opt_alb=1))


# --------------------------------------------------------------------------
# fixture hygiene
# --------------------------------------------------------------------------
#: The only module allowed to transcribe a glibc FP32 elementary function.
#: This lane and the VEGE_FLUX lane each arrived with one, written
#: independently from the same disassembly; they were shown to agree bit for
#: bit and then merged, because two copies of a bit-exact kernel is exactly
#: the situation the ``fp32_ulp`` single-source rule exists to prevent.
_LIBM_OWNER = "gpuwm/core/noahmp_libm.py"

#: Dotted module path of the owner, as an import target.
_LIBM_OWNER_MODULE = "gpuwm.core.noahmp_libm"

#: Names whose *definition* outside the owner means a transcription has been
#: re-derived.  Importing them is fine and is the point; ``def logf`` is not.
_LIBM_ENTRY_POINTS = frozenset(
    {"logf", "log10f", "expf", "powf", "atanf", "sqrtf", "f32"}
)


def _libm_owner_aliases(tree) -> frozenset[str]:
    """Every dotted name THIS module binds to the owner module.

    ``import gpuwm.core.noahmp_libm as _libm`` binds ``_libm``;
    ``from gpuwm.core import noahmp_libm`` binds ``noahmp_libm`` (or its
    asname); a bare ``import gpuwm.core.noahmp_libm`` binds the full
    dotted path.  ``from gpuwm.core.noahmp_libm import expf`` binds no
    MODULE name at all, so it grants no delegation base -- the exempt
    shape is attribute access on the imported owner, nothing looser.
    """
    import ast

    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == _LIBM_OWNER_MODULE:
                    aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "gpuwm.core":
                for alias in node.names:
                    if alias.name == "noahmp_libm":
                        aliases.add(alias.asname or alias.name)
    return frozenset(aliases)


def _float32_cast_names(tree) -> frozenset[str]:
    """Module-level names bound to ``np.float32`` (the ``F`` convention)."""
    import ast

    names: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "float32"
                and isinstance(node.value.value, ast.Name)):
            names.update(t.id for t in node.targets
                         if isinstance(t, ast.Name))
    return frozenset(names)


def _dotted_name(expr) -> str | None:
    """``a.b.c`` as a string, or None for anything not a pure name chain."""
    import ast

    parts = []
    while isinstance(expr, ast.Attribute):
        parts.append(expr.attr)
        expr = expr.value
    if not isinstance(expr, ast.Name):
        return None
    parts.append(expr.id)
    return ".".join(reversed(parts))


def _is_owner_delegation(node, owner_aliases, cast_names) -> bool:
    """Exactly the one exempt shape: ``return owner.<name>(...)``, at most
    wrapped in one ``np.float32``-bound cast call (``F(...)``).

    The single return expression must BE the delegated call -- a Call whose
    func is an Attribute named after the wrapped function on a base that
    resolves to an imported owner-module binding of THIS file.  Anything
    else -- arithmetic around the call (``owner.expf(x) + correction(x)``),
    a call on a module that is not the owner (``bogus.expf(x)``), a bare
    attribute that is never called -- is a fork, not a calling convention.
    """
    import ast

    body = [stmt for stmt in node.body
            if not (isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str))]
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return False
    value = body[0].value
    # Peel the float32-out carrier cast, exactly once: a single-positional-
    # argument call to a name this module binds to np.float32 (or to
    # np.float32 spelled inline).  Any other outer call is arithmetic.
    if (isinstance(value, ast.Call)
            and len(value.args) == 1 and not value.keywords
            and not isinstance(value.args[0], ast.Starred)
            and ((isinstance(value.func, ast.Name)
                  and value.func.id in cast_names)
                 or (isinstance(value.func, ast.Attribute)
                     and value.func.attr == "float32"
                     and isinstance(value.func.value, ast.Name)))):
        value = value.args[0]
    if not isinstance(value, ast.Call):
        return False
    func = value.func
    if not (isinstance(func, ast.Attribute) and func.attr == node.name):
        return False
    base = _dotted_name(func.value)
    return base is not None and base in owner_aliases


def _libm_fork_offenders(tree) -> list[str]:
    """Entry-point definitions in ``tree`` that are not owner delegations."""
    import ast

    owner_aliases = _libm_owner_aliases(tree)
    cast_names = _float32_cast_names(tree)
    return sorted(
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in _LIBM_ENTRY_POINTS
        and not _is_owner_delegation(node, owner_aliases, cast_names)
    )


def test_logf_has_not_forked_between_the_two_libm_modules():
    """One transcription of each glibc entry point, imported everywhere.

    The merge is only worth anything if a sixth lane cannot quietly start a
    seventh copy, so this reads the tree rather than trusting that it stayed
    merged: any module under ``gpuwm/`` that *defines* one of the entry points
    instead of importing it is a fork.

    One shape is exempt, precisely: a single-``return`` wrapper whose body
    IS a call to the owner's entry point of the same name, at most wrapped
    in one ``np.float32`` carrier cast (the legacy-RRTMG lane's
    ``F(_libm.expf(float(x)))`` float32-out convention).  That is one
    transcription with a calling convention, not a second transcription --
    a fork by hand would carry constants and arithmetic, and any body other
    than that one delegated ``return`` still fails here (see the predicate
    counterexample tests below for the shapes the codex step-1 audit showed
    the earlier walk-and-search predicate accepted).
    """
    import ast

    root = Path(__file__).resolve().parents[1]
    offenders = {}
    for path in sorted((root / "gpuwm").rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel == _LIBM_OWNER:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        defined = _libm_fork_offenders(tree)
        if defined:
            offenders[rel] = defined
    assert offenders == {}, (
        "these modules define a glibc FP32 entry point instead of importing"
        f" it from {_LIBM_OWNER}: {offenders}"
    )


def _fork_offenders_of(source: str) -> list[str]:
    import ast

    return _libm_fork_offenders(ast.parse(source))


def test_no_fork_predicate_rejects_arithmetic_around_the_owner_call():
    """Codex step-1 audit counterexample (i): a local correction added to
    the delegated call is a modified implementation, not a delegation."""
    source = (
        "from gpuwm.core import noahmp_libm as owner\n"
        "def local_correction(x):\n"
        "    return x\n"
        "def expf(x):\n"
        "    return owner.expf(x) + local_correction(x)\n"
    )
    assert _fork_offenders_of(source) == ["expf"]


def test_no_fork_predicate_rejects_a_non_owner_base_module():
    """Codex step-1 audit counterexample (ii): an owner import elsewhere in
    the file must not vouch for a call on some OTHER module."""
    source = (
        "from gpuwm.core import noahmp_libm as owner\n"
        "import bogus\n"
        "def expf(x):\n"
        "    return bogus.expf(x)\n"
    )
    assert _fork_offenders_of(source) == ["expf"]


def test_no_fork_predicate_accepts_the_shipped_wrapper_shapes():
    """Positive controls: the exact conventions gpuwm/core/rrtmg_lw.py and
    rrtmg_sw.py ship (F-cast, float carrier, two-argument powf), plus the
    uncast single-return delegation."""
    source = (
        "import numpy as np\n"
        "from gpuwm.core import noahmp_libm as _libm\n"
        "F = np.float32\n"
        "def logf(x):\n"
        "    return F(_libm.logf(float(x)))\n"
        "def expf(x):\n"
        "    return F(_libm.expf(float(x)))\n"
        "def powf(x, y):\n"
        "    return F(_libm.powf(float(x), float(y)))\n"
        "def atanf(x):\n"
        "    return _libm.atanf(x)\n"
    )
    assert _fork_offenders_of(source) == []
    # ... and the two shipped wrapper modules themselves stay accepted.
    root = Path(__file__).resolve().parents[1]
    for rel in ("gpuwm/core/rrtmg_lw.py", "gpuwm/core/rrtmg_sw.py"):
        text = (root / rel).read_text(encoding="utf-8")
        assert _fork_offenders_of(text) == [], rel


@pytest.mark.parametrize(
    "leaf", ["snow_age", "snowalb_class", "groundalb", "surrad", "twostream",
             "albedo"],
)
def test_fixture_case_names_unique(leaf):
    rows = _rows(leaf)
    names = [r["case"] for r in rows]
    assert len(set(names)) == len(names)
    for r in rows:
        for k, v in r.items():
            if v.startswith("0x"):
                assert len(v) == 10, f"{leaf}:{r['case']}:{k} = {v!r}"
