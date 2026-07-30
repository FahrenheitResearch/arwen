"""Acceptance gate for the WRF v4.6.1 Noah-MP BARE_FLUX transcription.

The bar is max_ulp 0 -- bitwise identity -- against a fixture produced by the
unmodified WRF module (visibility-patched only).  Nothing here relaxes that.

Layout of the evidence:

* :func:`test_fixture_provenance` -- the fixture header still names the pinned
  tree, sha256 and option identity, so a silently regenerated fixture cannot
  slip through.
* :func:`test_bitwise_against_oracle` -- the gate itself.
* :func:`test_branch_coverage_cases_present` -- each branch this lane claims to
  bind has a named case, and the case really does land on that branch
  (asserted through observable outputs, not through the note text).
* :func:`test_negative_controls` and :func:`test_source_mutants_are_all_killed`
  -- deliberately wrong transcriptions that the gate must reject.  A passing
  suite is not evidence unless it can fail.
* :func:`test_min_max_tie_semantics_are_unobservable` -- the one thing this
  fixture provably cannot pin, recorded as a limit rather than hidden.
* :func:`test_mutation_survivors_are_the_expected_dead_arguments` -- the
  mutation study, pinned to the exact survivor set that the pinned option
  identity predicts.
"""

from __future__ import annotations

import os
import struct
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOOLS = os.path.join(_ROOT, "tools", "noahmp_wrf461_oracle")
for _p in (_ROOT, _TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gpuwm.core.fp32_ulp import assert_bit_exact, max_ulp     # noqa: E402
from gpuwm.core import noahmp_bareflux as bf                  # noqa: E402

import validate_bareflux_oracle as vbo                        # noqa: E402
from gen_bareflux_cases import OUT_FIELDS, unhexf             # noqa: E402


@pytest.fixture(scope="module")
def rows():
    return vbo.load_fixture()


def _bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def test_fixture_provenance():
    with open(vbo.FIXTURE, "r") as handle:
        header = "".join(ln for ln in handle if ln.startswith("#"))
    for needle in (
        "d66e442fccc04111067e29274c9f9eaccc3cef28",
        "bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282",
        "opt_sfc=1",
        "opt_stc=1",
        "NITERB   5",
        "gfortran 13.3.0",
        "GLIBC 2.39",
        "noopt pristine",
    ):
        assert needle in header, f"fixture header lost {needle!r}"


def test_bitwise_against_oracle(rows):
    assert len(rows) >= 16
    for row in rows:
        got = vbo.call_row(row)
        for name in OUT_FIELDS:
            want = unhexf(row[name])
            assert_bit_exact([got[name]], [want], f"{row['case']}.{name}")


def _read_csv(path):
    import csv
    with open(path, newline="") as handle:
        lines = [ln for ln in handle if not ln.startswith("#")]
    return list(csv.DictReader(lines))


def test_cpu_libm_matches_live_glibc():
    """noahmp_libm's logf/atanf/powf against the live glibc 2.39 symbols.

    The reference is ``noahmp-bareflux-libm.csv``, produced by running
    ``tools/noahmp_wrf461_oracle/libm_ref_dump.c`` against libm.so.6 on the
    pinned host -- not against a correctly-rounded library and not against
    this transcription.  These are the functions gfortran calls for LOG, ATAN
    and **0.25 on a REAL(4).
    """
    from gpuwm.core.noahmp_libm import atanf, logf, powf

    ref = _read_csv(os.path.join(os.path.dirname(vbo.FIXTURE),
                                 "noahmp-bareflux-libm.csv"))
    assert len(ref) >= 2000
    for row in ref:
        x = unhexf(row["x"])
        for name, fn in (("logf", lambda: logf(x)),
                         ("atanf", lambda: atanf(x)),
                         ("powf025", lambda: powf(x, 0.25))):
            assert _bits(fn()) == int(row[name], 16), (
                f"{name}(0x{row['x']}) = 0x{_bits(fn()):08X}, "
                f"glibc says 0x{int(row[name], 16):08X}"
            )


def test_cpu_esat_matches_the_fortran():
    """ESAT over 12007 temperatures against the Fortran ESAT itself."""
    ref = _read_csv(os.path.join(os.path.dirname(vbo.FIXTURE),
                                 "noahmp-bareflux-esat.csv"))
    assert len(ref) >= 10000
    for row in ref:
        t = unhexf(row["t"])
        got = bf.esat(t)
        for j, name in enumerate(("esw", "esi", "desw", "desi")):
            assert _bits(got[j]) == int(row[name], 16), (
                f"esat({t!r}).{name} = 0x{_bits(got[j]):08X}, "
                f"Fortran says 0x{int(row[name], 16):08X}"
            )


def test_niterb_is_five_and_not_an_option():
    """NITERB is a DATA-initialised local, so it cannot be configured away."""
    assert bf.NITERB == 5


def test_dead_options_are_refused(rows):
    """SFCDIF2 and the opt_stc==3 blend are asserted off, not ported."""
    row = rows[0]
    with pytest.raises(NotImplementedError):
        vbo.call_row(row, {"opt_sfc": "2"})
    with pytest.raises(NotImplementedError):
        vbo.call_row(row, {"opt_stc": "3"})


def test_branch_coverage_cases_present(rows):
    """Every branch this lane claims is bound by an observable in some row."""
    by_case = {row["case"]: row for row in rows}
    expected = {
        "bf01_unstable_day", "bf02_stable_night", "bf03_ice_branch",
        "bf04_snow_reset", "bf05_lowwind", "bf06_highwind_ramb_floor",
        "bf07_urban", "bf08_tdc_clamp_high", "bf09_tdc_clamp_low",
        "bf10_degenerate_cmfm", "bf11_degenerate_z0m_2m", "bf12_fh_clamp",
        "bf13_pahb", "bf14_isnow_shallow", "bf15_zpd_offset",
        "bf16_negative_wind_components",
    }
    assert expected <= set(by_case)

    # EHB2 < 1e-5 arm: T2MB collapses onto TGB and Q2B onto QSFC.
    low = by_case["bf05_lowwind"]
    assert low["t2mb"] == low["tgb_out"]
    assert low["q2b"] == low["qsfc_out"]
    assert unhexf(low["ehb2"]) < 1.0e-5

    # The normal arm must not do that, or the test above proves nothing.
    day = by_case["bf01_unstable_day"]
    assert day["t2mb"] != day["tgb_out"]
    assert unhexf(day["ehb2"]) >= 1.0e-5

    # RAMB/RAHB floor: CH is returned as EHB = 1/RAHB, so RAHB == 1 shows up
    # as CH == 1.0 exactly.
    assert unhexf(by_case["bf06_highwind_ramb_floor"]["ch_out"]) == 1.0
    assert unhexf(day["ch_out"]) != 1.0

    # Snow reset: opt_stc==1 pins TGB to TFRZ exactly.
    assert unhexf(by_case["bf04_snow_reset"]["tgb_out"]) == bf.TFRZ
    # ...and the SNOWH <= 0.05 companion case must not be pinned.
    assert unhexf(by_case["bf14_isnow_shallow"]["tgb_out"]) != bf.TFRZ

    # urban_flag forces Q2B onto QSFC even though EHB2 is large.
    urban = by_case["bf07_urban"]
    assert urban["q2b"] == urban["qsfc_out"]
    assert unhexf(urban["ehb2"]) >= 1.0e-5

    # Degenerate roughness drives CM through the |CMFM| <= MPE guard, which
    # makes CM = VKC**2 / MPE**2 = 1.6e11.
    assert unhexf(by_case["bf10_degenerate_cmfm"]["cm_out"]) > 1.0e10

    # Ice vs water arm of ESAT: the ice case must stay below freezing.
    assert unhexf(by_case["bf03_ice_branch"]["tgb_out"]) < bf.TFRZ
    assert unhexf(day["tgb_out"]) > bf.TFRZ

    # TDC clamps.
    assert unhexf(by_case["bf08_tdc_clamp_high"]["tgb_out"]) > bf.TFRZ + 50.0
    assert unhexf(by_case["bf09_tdc_clamp_low"]["tgb_out"]) < bf.TFRZ - 50.0

    # Wind-stress sign follows UU/VV.
    assert unhexf(by_case["bf16_negative_wind_components"]["tauxb"]) == \
        -unhexf(day["tauxb"])


def _detecting_cases(rows, attr: str, replacement) -> list[str]:
    """Cases whose outputs move when ``bf.<attr>`` is replaced."""
    original = getattr(bf, attr)
    setattr(bf, attr, replacement)
    try:
        hits = []
        for row in rows:
            try:
                got = vbo.call_row(row)
            except Exception:            # a crashing mutant is also detected
                hits.append(row["case"])
                continue
            if any(_bits(got[k]) != _bits(unhexf(row[k])) for k in OUT_FIELDS):
                hits.append(row["case"])
        return hits
    finally:
        setattr(bf, attr, original)


def test_negative_controls(rows):
    """The gate must reject transcriptions that are wrong in realistic ways.

    Each control is a plausible way to get this leaf subtly wrong.  The
    libm ones matter most: glibc's logf/atanf/powf agree with a
    correctly-rounded float32 answer on the great majority of arguments, so a
    fixture that happens to miss the disagreements would let ``math.log`` and
    ``x ** 0.25`` through.  ``bf17``..``bf22`` exist precisely to close that,
    and this test is what proves they do.
    """
    import math

    controls = [
        ("logf -> correctly-rounded float32 log", "logf",
         lambda x: bf.f32(math.log(bf.f32(x)))),
        ("atanf -> correctly-rounded float32 atan", "atanf",
         lambda x: bf.f32(math.atan(bf.f32(x)))),
        ("powf -> correctly-rounded float32 pow", "powf",
         lambda x, y: bf.f32(math.pow(bf.f32(x), bf.f32(y)))),
        ("powf -> the sqrt(sqrt(x)) shortcut", "powf",
         lambda x, y: bf.f32(math.sqrt(bf.f32(math.sqrt(bf.f32(x)))))),
        ("every float32 rounding dropped", "f32", lambda x: float(x)),
        ("x**4 evaluated as ((x*x)*x)*x", "_powi4",
         lambda x: bf.f32(bf.f32(bf.f32(bf.f32(x * x) * x) * x))),
        ("x**4 evaluated in double and rounded once", "_powi4",
         lambda x: bf.f32(float(x) ** 4)),
        ("x**3 evaluated in double and rounded once", "_powi3",
         lambda x: bf.f32(float(x) ** 3)),
    ]
    for label, attr, replacement in controls:
        hits = _detecting_cases(rows, attr, replacement)
        assert hits, f"negative control went undetected: {label}"

    # NITERB is not a function, so it gets its own control.
    saved = bf.NITERB
    bf.NITERB = 4
    try:
        got = vbo.call_row(rows[0])
        assert max_ulp([got[k] for k in OUT_FIELDS],
                       [unhexf(rows[0][k]) for k in OUT_FIELDS]) > 0, \
            "NITERB=4 went undetected -- the loop count is not being tested"
    finally:
        bf.NITERB = saved

    # And the gate is clean again afterwards.
    for row in rows:
        again = vbo.call_row(row)
        for k in OUT_FIELDS:
            assert _bits(again[k]) == _bits(unhexf(row[k]))


def _source_mutant(old: str, new: str):
    """Load a copy of the transcription with one expression rewritten.

    Attribute swaps can only reach things the module exposes as names; the
    interesting ways to get this leaf wrong are inside expressions.  Rewriting
    the source and exec'ing it reaches those, and asserting the substring is
    present first means a refactor cannot silently turn a mutant into a no-op.
    """
    import types

    path = bf.__file__
    with open(path, "r", encoding="utf-8") as handle:
        src = handle.read()
    assert src.count(old) == 1, f"mutation target {old!r} is not unique"
    name = "noahmp_bareflux_mutant"
    module = types.ModuleType(name)
    module.__file__ = path
    # @dataclass resolves annotations through sys.modules[cls.__module__].
    sys.modules[name] = module
    try:
        exec(compile(src.replace(old, new), path, "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module


SOURCE_MUTANTS = [
    # WRF really divides CH by CMFM*CHFH, not CHFH*CHFH.  This is the single
    # most likely "tidy-up" a reader would make.
    ("f32(VKC * VKC) / f32(cmfm * chfh)", "f32(VKC * VKC) / f32(chfh * chfh)"),
    # Stability-correction clamp coefficient.
    ("state.fh = _fmin(state.fh, f32(f32(0.9) * tmpch))",
     "state.fh = _fmin(state.fh, f32(f32(0.95) * tmpch))"),
    # Ratio of molecular weights in QSFC.
    ("f32(0.622) * er", "f32(0.623) * er"),
    ("f32(0.378) * er", "f32(0.379) * er"),
    # The hard-coded pi/2 in FMNEW.
    ("- f32(2.0 * atanf(tmp1))) + f32(1.5707963))",
     "- f32(2.0 * atanf(tmp1))) + f32(1.5707964))"),
    # Monin-Obukhov unstable-profile coefficient.
    ("f32(16.0 * state.moz)", "f32(15.9 * state.moz)"),
    # TDC clamp bounds.
    ("_fmin(f32(50.0), _fmax(f32(-50.0)", "_fmin(f32(51.0), _fmax(f32(-50.0)"),
    ("_fmax(f32(-50.0), f32(t - TFRZ))", "_fmax(f32(-49.0), f32(t - TFRZ))"),
    # Snow-reset depth threshold.
    ("snowh > f32(0.05)", "snowh > f32(0.04)"),
    # 2 m diagnostic cut-off.
    ("ehb2 < f32(1.0e-5)", "ehb2 < f32(1.0e-6)"),
    # EHB2 must be corrected by FH2, not FH.
    ("z0h)) - st.fh2)", "z0h)) - st.fh)"),
    # The stability weighting is a half, not a two-thirds.
    ("f32(0.5) * f32(state.fh + fhnew)", "f32(0.51) * f32(state.fh + fhnew)"),
]


def test_source_mutants_are_all_killed(rows):
    """Every deliberate one-expression error must move at least one output."""
    survivors = []
    for old, new in SOURCE_MUTANTS:
        module = _source_mutant(old, new)
        saved = vbo.bare_flux
        vbo.bare_flux = module.bare_flux
        try:
            hit = False
            for row in rows:
                try:
                    got = vbo.call_row(row)
                except Exception:
                    hit = True
                    break
                if any(_bits(got[k]) != _bits(unhexf(row[k]))
                       for k in OUT_FIELDS):
                    hit = True
                    break
        finally:
            vbo.bare_flux = saved
        if not hit:
            survivors.append(old)
    assert not survivors, (
        "the fixture cannot see these source mutations: " + repr(survivors)
    )


def test_min_max_tie_semantics_are_unobservable(rows):
    """Honest statement of a limit: MIN/MAX tie-breaking cannot be pinned.

    Every ``MIN``/``MAX`` in BARE_FLUX and SFCDIF1 compares two values that are
    equal only when they are the same value, or when they are ``-0.0`` and
    ``+0.0`` -- and in that case both answers flow into a subtraction that
    erases the sign before anything observable.  Swapping the Fortran
    ``minss``/``maxss`` semantics for Python's therefore changes nothing, and
    this test records that rather than pretending it is a passing control.
    """
    for attr, replacement in (("_fmin", min), ("_fmax", max)):
        assert _detecting_cases(rows, attr, replacement) == [], (
            f"{attr} tie semantics became observable -- the note in this test "
            "is now wrong and the claim needs redoing"
        )


def test_ch2_path_is_unobservable_through_bare_flux(rows):
    """SFCDIF1's CH2 output, and the two guards that feed it, cannot be seen.

    ``branch_coverage_bareflux.sh`` reports lines 4732 and 4733 -- the
    ``ABS(CM2FM2) <= MPE`` and ``ABS(CH2FH2) <= MPE`` guards -- as untaken by
    the whole 27-case deck, and no case can take them observably: CM2FM2 is
    written and then read only by the commented-out CH2 formula on line 4736,
    and CH2FH2 is read only by ``CH2 = VKC*FV/CH2FH2``, whose value BARE_FLUX
    declares, receives as INTENT(OUT) and never uses.  Both are transcribed
    anyway; this test records that the fixture cannot police them, so nobody
    later reads their coverage gap as a fixture defect.
    """
    import types

    path = bf.__file__
    with open(path, encoding="utf-8") as handle:
        src = handle.read()
    old = "    state.ch2 = f32(f32(VKC * state.fv) / ch2fh2)"
    assert src.count(old) == 1
    name = "noahmp_bareflux_ch2_mutant"
    module = types.ModuleType(name)
    module.__file__ = path
    sys.modules[name] = module
    try:
        exec(compile(src.replace(old, "    state.ch2 = f32(-12345.0)"),
                     path, "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)

    saved = vbo.bare_flux
    vbo.bare_flux = module.bare_flux
    try:
        for row in rows:
            got = vbo.call_row(row)
            for k in OUT_FIELDS:
                assert _bits(got[k]) == _bits(unhexf(row[k])), (
                    "CH2 became observable through BARE_FLUX; the coverage "
                    "note in branch_coverage_bareflux.sh needs redoing"
                )
    finally:
        vbo.bare_flux = saved


def test_libm_controls_are_bound_by_named_cases(rows):
    """The libm discriminator cases must be the ones doing the detecting."""
    import math

    by_attr = {
        "logf": (lambda x: bf.f32(math.log(bf.f32(x))), "bf17_libm_logf_1"),
        "atanf": (lambda x: bf.f32(math.atan(bf.f32(x))), "bf19_libm_atanf_1"),
        "powf": (lambda x, y: bf.f32(math.pow(bf.f32(x), bf.f32(y))),
                 "bf21_libm_powf_1"),
    }
    for attr, (replacement, expected) in by_attr.items():
        hits = _detecting_cases(rows, attr, replacement)
        assert expected in hits, (
            f"{expected} no longer discriminates glibc {attr}; the fixture has "
            "lost its only guard against a correctly-rounded substitute"
        )


def test_mozsgn_reset_branch_is_reached(rows):
    """The MOZSGN>=2 reset really fires, and only on the cases built for it."""
    reached = {}
    original = bf.sfcdif1

    def probe(state, it, *a, **kw):
        original(state, it, *a, **kw)
        reached[probe.case] = max(reached.get(probe.case, 0), state.mozsgn)

    bf.sfcdif1 = probe
    try:
        for row in rows:
            probe.case = row["case"]
            vbo.call_row(row)
    finally:
        bf.sfcdif1 = original

    hot = {c for c, v in reached.items() if v >= 2}
    assert hot == {"bf23_mozsgn_reset_1", "bf24_mozsgn_reset_2"}, hot


# Arguments BARE_FLUX accepts and provably never lets reach an output under
# the pinned option identity.  See the module docstring of
# gpuwm/core/noahmp_bareflux.py for the line-by-line argument.
EXPECTED_MUTATION_SURVIVORS = {
    "dt",        # never referenced in the body
    "thair",     # only read by SFCDIF2 (opt_sfc == 2, dead)
    "q2",        # never referenced in the body
    "dx",        # never referenced in the body
    "dz8w",      # never referenced in the body
    "qc",        # never referenced in the body
    "sfcprs",    # never referenced in the body (QSFC uses PSFC)
    "fsno",      # only read by the opt_stc == 3 blend (dead)
    "cm",        # INOUT, but SFCDIF1 writes it INTENT(OUT) before any read
    "ch",        # INOUT, same
    "qsfc",      # INOUT, but assigned in the loop before any read
    "ivgtyp",    # declared, never referenced
    "iloc",      # passed to SFCDIF1 only for its error message
    "jloc",      # same
}


@pytest.mark.slow
def test_mutation_survivors_are_the_expected_dead_arguments(rows):
    """One mutant per argument; survivors must be exactly the dead arguments.

    A survivor that is not on the list means the fixture cannot see that
    argument being dropped -- a fixture defect, not a pass.
    """
    survivors = vbo.mutation_survivors(rows)
    unexpected = survivors - EXPECTED_MUTATION_SURVIVORS
    missing = EXPECTED_MUTATION_SURVIVORS - survivors
    assert not unexpected, (
        f"fixture cannot detect these arguments being ignored: {sorted(unexpected)}"
    )
    assert not missing, (
        f"expected-dead arguments that turned out to be live: {sorted(missing)}"
    )


# --------------------------------------------------------------------------
# the runtime device packing
# --------------------------------------------------------------------------
def _physical_kwargs(row: dict) -> dict:
    """The physical BARE_FLUX keyword call for one fixture row."""
    from gen_bareflux_cases import (ARRAY_FIELDS, INT_FIELDS, NSNOW, NSOIL,
                                    REAL_FIELDS)

    kw = {k: int(row[k]) for k in INT_FIELDS if k != "iurban"}
    kw["urban_flag"] = bool(int(row["iurban"]))
    for name in REAL_FIELDS:
        kw[name] = unhexf(row[name])
    for name in ARRAY_FIELDS:
        kw[name] = [unhexf(row[f"{name}{k}"])
                    for k in range(-NSNOW + 1, NSOIL + 1)]
    kw["nsnow"] = NSNOW
    kw["nsoil"] = NSOIL
    kw["opt_sfc"] = int(row["opt_sfc"])
    kw["opt_stc"] = int(row["opt_stc"])
    return kw


def _fixture_rows_flat(rows):
    """The flat device row the kernel documents, read straight off the CSV."""
    import numpy as np

    from gen_bareflux_cases import (ARRAY_FIELDS, INT_FIELDS, NSNOW, NSOIL,
                                    REAL_FIELDS)

    order = REAL_FIELDS + [f"{name}{k}" for name in ARRAY_FIELDS
                           for k in range(-NSNOW + 1, NSOIL + 1)]
    flat = np.array([[unhexf(row[name]) for name in order] for row in rows],
                    dtype=np.float32)
    ints = np.array([[int(row[name]) for name in INT_FIELDS] for row in rows],
                    dtype=np.int32)
    return flat, ints


def test_the_runtime_packing_matches_the_kernel_layout(rows):
    """The device batch's row must be the row ``noahmp_bareflux.cu`` reads.

    ``gpuwm.core.noahmp_bareflux_gpu`` packs a *physical* keyword call, while
    the kernel indexes a flat row by position.  Those two orders are related by
    nothing the compiler can check, so they are checked here against the oracle
    generator's own column order -- an authority outside this lane.  A silent
    transposition of two scalars is otherwise a plausible-looking forecast.
    """
    import numpy as np

    from gpuwm.core.noahmp_bareflux_gpu import pack_bare_flux_calls

    calls = [((), _physical_kwargs(row)) for row in rows]
    packed, packed_ints = pack_bare_flux_calls(calls)
    want, want_ints = _fixture_rows_flat(rows)
    assert packed.shape == want.shape
    bad = np.flatnonzero(packed.ravel().view(np.uint32)
                         != want.ravel().view(np.uint32))
    assert bad.size == 0, f"{bad.size} packed words differ, first at {bad[0]}"
    np.testing.assert_array_equal(packed_ints, want_ints)


def test_the_packing_gate_can_fail(rows, monkeypatch):
    """Show the failing form before the passing one.

    Transpose two scalar slots in the packer and the comparison above must
    reject it.  ``sag`` and ``lwdn`` are both live in the kernel, so this is a
    transposition that would change the answer rather than one the pinned
    identity makes inert.
    """
    import numpy as np

    from gpuwm.core import noahmp_bareflux_gpu as gpu

    names = list(gpu.SCALAR_NAMES)
    a, b = names.index("sag"), names.index("lwdn")
    names[a], names[b] = names[b], names[a]
    monkeypatch.setattr(gpu, "SCALAR_NAMES", tuple(names))

    calls = [((), _physical_kwargs(row)) for row in rows]
    packed, _ = gpu.pack_bare_flux_calls(calls)
    want, _ = _fixture_rows_flat(rows)
    assert (packed.ravel().view(np.uint32)
            != want.ravel().view(np.uint32)).any(), (
        "transposing sag and lwdn in the packer went undetected")
