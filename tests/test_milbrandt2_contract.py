"""Milbrandt-Yau (WRF ``mp_physics = 9``) contract: no device touched.

Everything here is a statement about the SHIPPED declarations -- the
constant table, config admission and the pinned identity, the transported
species set, the precipitation-slot row, the ring-guard budget, the
radiation coupling and the registry row.  The column smoke that runs the
scheme lives in ``tests/test_milbrandt2.py``, which imports cupy and is
therefore marked ``gpu`` in full.

SCOPE: nothing in either file is an oracle.  No comparison against the WRF
v4.6.1 Fortran has been run -- no ULP table, no column oracle, no matched
trajectory.
"""


from __future__ import annotations

import re

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.config import (RunConfig, validate_milbrandt2_options,
                          validate_run_config)
from gpuwm.core import constants as c


MASS_NAMES = ("qv", "qc", "qr", "qi", "qs", "qg", "qh")
NUMBER_NAMES = ("nc", "nr", "ni", "ns", "ng", "nh")


def _cfg(**overrides):
    values = dict(
        nx=6, ny=5, nz=24, dx=1000.0, dy=1000.0, ztop=15000.0,
        dt=10.0, run_seconds=10.0, moist=True, mp_physics=9)
    values.update(overrides)
    return RunConfig(**values)


def _regime_column(regime: str, nz: int = 24):
    """FP32 WK82-like column spanning one requested phase regime.

    Returns float64 profiles; the caller casts.  Deliberately NOT a WRF
    fixture -- there is no oracle to compare against, so this only has to
    be a physically ordinary sounding that reaches the branches.
    """
    z = np.linspace(200.0, 14200.0, nz, dtype=np.float64)
    p = 98000.0 * np.exp(-z / 8000.0)
    if regime == "warm":
        temp = np.maximum(298.0 - 0.0045 * z, 276.0)
    elif regime == "mixed":
        temp = 289.0 - 0.0065 * z
    elif regime == "glaciated":
        temp = 258.0 - 0.0030 * z
    else:
        raise ValueError(regime)
    return z, p, temp


# ---------------------------------------------------------------------------
# The constant table (no GPU needed)
# ---------------------------------------------------------------------------

def test_scheme_gamma_is_the_four_term_truncation_not_a_true_gamma():
    """module_mp_milbrandt2mom.F:184 is ``do j=1,4``, not ``do j=1,6``.

    gamma(3) = 2 exactly for a true gamma.  The scheme's truncated Lanczos
    returns 1.9999542, and every distribution constant is built from it, so
    a "fixed" gamma would move all 154 entries away from what WRF
    integrates with.
    """
    from gpuwm.core.milbrandt2_constants import CONSTANTS, gamma_my2

    value = float(gamma_my2(3.0))
    assert value != 2.0
    assert value == pytest.approx(1.9999542, abs=1e-7)
    # GC13 = gamma(3*(1/MUc) + 1 + alpha_c) = gamma(3) with MUc=3, alpha_c=1.
    assert float(CONSTANTS["GC13"]) == value


def test_constant_table_is_finite_and_kernel_ordered():
    from gpuwm.core.milbrandt2_constants import (
        CK_INDEX, CK_ORDER, CONSTANTS, ck_vector)

    vector = ck_vector()
    assert vector.dtype == np.float32
    assert vector.shape == (len(CK_ORDER),)
    assert np.all(np.isfinite(vector)), "a non-finite constant would poison "\
        "every column that reads it"
    for name, index in CK_INDEX.items():
        assert vector[index] == CONSTANTS[name]
    # The values that pin the two compiled-in switches.
    assert float(CONSTANTS["N_c_SM"]) == 2.0e8      # CCNtype = 2, :3615
    assert float(CONSTANTS["cms"]) == pytest.approx(0.1597)   # :1290 Brandes
    assert float(CONSTANTS["dms"]) == pytest.approx(2.078)


def test_cuda_define_block_still_matches_the_shipped_translation_unit():
    """The kernel indexes CK by position; a reordered table is silent death.

    ``cuda_define_block()`` is the generator for the ``#define`` rows at the
    top of ``kernels/milbrandt2.cu``.  If someone inserts a constant in the
    middle of ``_build()`` and does not regenerate, every macro after the
    insertion point reads a different number and NOTHING raises -- the run
    just produces wrong weather.  This is the guard.
    """
    from pathlib import Path

    from gpuwm.core.milbrandt2_constants import cuda_define_block
    import gpuwm.core.kernels as kernels

    source = (Path(kernels.__file__).parent / "milbrandt2.cu").read_text(
        encoding="utf-8")
    expected = cuda_define_block()
    assert expected in source, (
        "kernels/milbrandt2.cu's #define block has drifted from "
        "milbrandt2_constants.CK_ORDER; regenerate it")


# ---------------------------------------------------------------------------
# The two numeric-identity decisions, pinned (no GPU needed)
# ---------------------------------------------------------------------------
#
# Both fixes below changed the numbers this scheme produces, both were made
# for a WRF oracle campaign that has NOT been run, and until it is there is
# nothing downstream that would notice either one being undone.  Reverting
# the rain-evaporation chain to double left the whole mp9 suite green, and
# swapping powf back to __powf would too: they move results by ULPs, and
# every gate in this lane is a budget or a range.  So these are guards on
# the SOURCE, in the house pattern (tests/test_noahmp_sflx_cuda.py:150,
# tests/test_sase_gpu.py:5731), not on any computed value.


def _kernel_source() -> str:
    from pathlib import Path

    import gpuwm.core.kernels as kernels

    return (Path(kernels.__file__).parent / "milbrandt2.cu").read_text(
        encoding="utf-8")


def _code(text: str) -> str:
    """The kernel with its prose removed.

    The header block at ``milbrandt2.cu:30-31`` NAMES the four intrinsics
    the test below forbids, in order to record that this file once shipped
    90 of them.  Scanning raw text would make that note trip its own gate,
    which is how a check gets deleted instead of fixed.
    """
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return "\n".join(re.sub(r"//.*$", "", line)
                     for line in text.splitlines())


def test_transcendentals_are_ieee_not_fast_math():
    """No ``__``-prefixed device intrinsic may re-enter this kernel.

    The port briefly shipped 90 fast-math sites -- ``__powf`` 63, ``__expf``
    18, ``__log10f`` 5, ``__logf`` 4 -- and was the only microphysics kernel
    in the tree using any.  ``__powf`` alone diverges from ``powf`` by up to
    tens of ULPs over this scheme's own argument ranges, which would put a
    reduced-precision floor under the WRF oracle campaign before it starts.
    The swap back to IEEE is invisible to every test in this lane, so this
    is what holds it: the intrinsics are banned by name, and the general
    ``__``-prefixed form is banned too so the next one to arrive is caught
    without this list being edited first.
    """
    code = _code(_kernel_source())
    # Counted, not membership-tested: a bare ``in`` puts the whole 2,400-line
    # translation unit in the assertion's introspection output, which buries
    # the one word that says what broke.
    named = {banned: code.count(banned)
             for banned in ("__powf", "__expf", "__logf", "__log10f",
                            "__fdividef", "__frcp_rn", "__fsqrt_rn",
                            "__sinf", "__cosf", "__tanf", "__fmul_rn",
                            "__fadd_rn", "__saturatef")
             if code.count(banned)}
    assert named == {}, (
        f"fast-math intrinsics re-entered milbrandt2.cu: {named} -- the "
        "IEEE transcendentals are the whole reason this port can be "
        "compared to WRF")
    # The general form, so the NEXT intrinsic is caught without this list
    # being edited first.  Nothing in this kernel legitimately calls a
    # double-underscore function.
    intrinsics = sorted(set(re.findall(r"\b__[a-z][a-z0-9_]*(?=\s*\()",
                                       code)))
    assert intrinsics == [], (
        f"unlisted device intrinsics appeared: {intrinsics}")
    # The header note must survive too -- it is the only record of why.
    assert "TRANSCENDENTALS ARE IEEE, NOT FAST-MATH." in _kernel_source()


def test_rain_evaporation_is_single_precision_like_the_fortran():
    """WRF's :3058 is REAL(4) throughout; the port's shipped chain was not.

    ``module_mp_milbrandt2mom.F`` declares every factor of the QREVP
    expression single -- ``No_r`` and ``PI2`` in the ``real, save`` list
    (:1013, :1016), ``ssat`` in the ``real ::`` local block (:994), ``dt``
    ``real, intent(in)`` (:850).  The ONE double in the neighbourhood is
    the :3054 ``No_r`` product, and WRF's own comment at :3052 says why --
    "the following coding of 'No_r=...' prevents overflow" -- after which
    the double result is rounded back to REAL(4) on assignment.

    The port originally carried the whole chain in double, which made it
    MORE accurate than WRF and would have surfaced in the oracle as an
    unexplained divergence.  Reverting it today leaves the mp9 suite green,
    so the two rows are pinned verbatim: the overflow guard must stay
    double AND must land in a float, and the rate must be one float
    expression in the Fortran's left-to-right order.
    """
    # Counted rather than membership-tested, for the same reason as the
    # intrinsic gate above: ``in`` dumps the whole translation unit.
    code = _code(_kernel_source())

    guard = ("float No_r = (float)((double)nr\n"
             "                             * pow((double)LAMr,\n"
             "                                   (double)(1.0f + "
             "MY2_alpha_r))\n"
             "                             * (double)iGR31);")
    assert code.count(guard) == 1, (
        "the :3054 overflow guard must compute in double and be rounded "
        "back to float on assignment, exactly as No_r is REAL(4) in WRF")

    rate = ("float rate = -(dt * (ide * PI2 * ssat * No_r * VENTr / ABw));\n"
            "        float QREVP = fminf(qr, rate);")
    assert code.count(rate) == 1, (
        "the :3058 QREVP chain must be evaluated in float in Fortran's "
        "left-to-right order")

    reverted = {form: code.count(form)
                for form in ("double rate", "double No_r", "(float)fmin(")
                if code.count(form)}
    assert reverted == {}, (
        f"{reverted} is the double-precision rain-evaporation chain this "
        "lane removed; it is more accurate than WRF, not less")


# ---------------------------------------------------------------------------
# Config admission and the pinned identity (no GPU needed)
# ---------------------------------------------------------------------------

def test_config_admits_mp_physics_9():
    cfg = validate_run_config(_cfg())
    assert cfg.mp_physics == 9
    validate_milbrandt2_options(cfg)          # must not raise


def test_config_refuses_mp9_without_moisture():
    with pytest.raises(ValueError, match="requires moist=true"):
        validate_run_config(_cfg(moist=False))


def test_config_refuses_the_rrtmgp_pairing_and_names_the_way_through():
    with pytest.raises(NotImplementedError) as excinfo:
        validate_run_config(_cfg(ra_lw_physics=4, ra_sw_physics=4))
    message = str(excinfo.value)
    assert "has_reqc" in message
    assert "rrtmg_legacy" in message


def test_config_accepts_mp9_on_the_legacy_rrtmg_variant():
    cfg = validate_run_config(
        _cfg(ra_lw_physics=4, ra_sw_physics=4,
             ra_rrtmg_variant="rrtmg_legacy"))
    assert cfg.mp_physics == 9


def test_pinned_identity_table_cites_a_line_for_every_switch():
    from gpuwm.config import MILBRANDT2_FIXED_IDENTITY

    assert MILBRANDT2_FIXED_IDENTITY["ccntype"][0] == 2
    assert MILBRANDT2_FIXED_IDENTITY["snow_spherical"][0] is False
    for name, (_value, citation) in MILBRANDT2_FIXED_IDENTITY.items():
        assert re.search(r"module_mp_milbrandt2mom\.F:\d+", citation), (
            f"{name} is pinned without naming the Fortran line that pins it")


def test_pinned_identity_refuses_a_disagreeing_attribute():
    """The refusal is live, not decorative: a knob that appears and
    disagrees is refused by name."""
    class _Shim:
        mp_physics = 9
        ccntype = 1                      # maritime -- WRF cannot produce it
        ra_lw_physics = 0
        ra_sw_physics = 0
        ra_rrtmg_variant = "rte-rrtmgp"

    with pytest.raises(NotImplementedError, match="ccntype=1"):
        validate_milbrandt2_options(_Shim())


# ---------------------------------------------------------------------------
# The transported-species contract (no GPU needed)
# ---------------------------------------------------------------------------

def test_moist_species_discriminates_mp9_from_nssl_on_nh():
    """Both schemes allocate ``qh``; only mp=9 allocates ``nh``."""
    from gpuwm.core.moist import MY2_SPECIES, NSSL_SPECIES, extra_moist_species

    class _My2:
        qi = qs = qg = qh = object()
        nc = nr = ni = ns = ng = nh = object()

    class _Nssl:
        qi = qs = qg = qh = object()
        nh = None
        qndrop = qnr = qni = qns = qng = qnh = qnn = object()
        qvolg = qvolh = object()

    assert extra_moist_species(_My2()) == MY2_SPECIES
    assert extra_moist_species(_Nssl()) == NSSL_SPECIES
    assert "nh" in MY2_SPECIES and "nh" not in NSSL_SPECIES


def test_precipitation_slot_row_carries_hail():
    from gpuwm.core.physics import microphysics_scratch_slots

    slots = dict(microphysics_scratch_slots(9))
    assert slots["hailnc"] == "mp_hailnc"
    assert slots["hailncv"] == "mp_hailncv"
    assert slots["graupelnc"] == "mp_graupelnc"
    assert dict(microphysics_scratch_slots(9)) == \
        dict(microphysics_scratch_slots(18))


def test_ring_guard_registers_every_mp9_field_it_will_capture():
    from gpuwm.core.microphysics import spec_zone_ring_save_slots

    cfg = _cfg(nested=True, spec_zone=2)
    slots = spec_zone_ring_save_slots(cfg)
    for key in ("qh", "nc", "nr", "ni", "ns", "ng", "nh"):
        assert any(name.startswith(f"mp_ring_save_{key}_")
                   for name in slots), f"{key} is captured but unbudgeted"
    for key in ("mp_hailnc", "mp_hailncv"):
        assert any(name.startswith(f"mp_ring_save_{key}_")
                   for name in slots)


def test_legacy_radiation_declares_no_scheme_radii_but_stays_ice_active():
    """WRF's own pairing: has_req* = 0 and F_QI/F_QS = true."""
    from gpuwm.core.rrtmg_legacy import (legacy_ice_active,
                                         legacy_scheme_declares_radii)

    assert legacy_scheme_declares_radii(9, use_mp_re=1) is False
    assert legacy_ice_active(9) is True


def test_namelist_naming_mp9_imports_it_natively():
    from gpuwm.namelist_import import _MP_MAP

    mapped, wrf_name, gp_name = _MP_MAP[9]
    assert mapped == 9, "mapping 9 to anything else would be a substitution"
    assert wrf_name == gp_name == "Milbrandt-Yau 2-moment"


def test_registry_row_is_implemented_unverified_and_says_so():
    import json
    from pathlib import Path

    import gpuwm

    registry = json.loads(
        (Path(gpuwm.__file__).parent / "physics_registry_v2.json").read_text(
            encoding="utf-8"))
    row = registry["components"]["microphysics"]["options"]["milbrandt2mom-mp9"]
    assert row["implemented"] is True
    assert row["maturity"] == "implemented-unverified"
    assert row["reachability"]["state"] == "component-override"
    assert row["selectors"] == {"mp_physics": 9}
    assert any("NO ORACLE HAS BEEN RUN" in w for w in row["warnings"])
