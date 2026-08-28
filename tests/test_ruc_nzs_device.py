"""The RUC_NZS tier on the card.

``tests/test_ruc_nzs_tier.py`` proves the nine-level translation unit did not
change, on the CPU, three ways.  This file asks the remaining question, which
only a device can answer: **does compiling through the tiered loader change
the numbers?**

That is not the same question.  The CPU proof compares the source the
UNSPECIALIZED loader assembles against the pre-lift source.  It says nothing
about the arm of the mechanism a six-level run actually takes -- the
``#if RUC_NZS == 9`` ladder as seen by NVRTC with a define injected, the
derived ``RUC_NZS_M1/M2/M3/DTDZS_LEN`` macros, and the ``#if``-selected
``__constant__`` table.  So the gate here runs the whole 43-field driver
comparison with the tier FORCED ON at the shipped geometry: same count, other
code path.  Max ULP 0, or the ladder is not inert.

This is the closest substitute for a six-level forecast oracle that exists on
this machine.  It validates the MACHINERY at a geometry that HAS an oracle,
which leaves only the geometry itself untested -- see
``docs/wrf_ruc_runtime_admission.md`` for what that does and does not buy.

The file then asks the *other* half, which nothing else here asks: at SIX
levels, do the host lane and the device lane agree?  They are line-for-line
transcriptions of the same Fortran, so agreement cannot detect an error
present in both -- but a disagreement is a transcription error, and the
first run of that comparison found one.  ``ruc_soil_finalize`` computed
``dzstop`` from a hardcoded ``0.01f - 0.0f``, WRF's nine-level
``zsmain(2) - zsmain(1)``, and at six levels returned grdflx -337.1 W m-2
where the host returned -67.4 -- a factor of exactly 5, which is
0.05 / 0.01.  Nothing in the tree could have seen it: every other RUC gate
runs at nine, where the literal is right.  That is what this comparison is
for, and it is why "it compiles at six" was never the claim to make.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from gpuwm.core import ruc_tier  # noqa: E402
from gpuwm.core.kernels import (module_source,  # noqa: E402
                                module_source_int_defines)
from gpuwm.core.ruc import (RUC_DRIVER_COLUMN_STATE,  # noqa: E402
                            RUC_DRIVER_PROFILE_STATE,
                            RucLandSurfaceStep, ruc_land_surface_step)
from gpuwm.core.ruc_gpu import (RUC_SFCTMP_DEVICE_LEAVES,  # noqa: E402
                                RUC_SFCTMP_DEVICE_LEAVES_SNOW_FREE,
                                RUC_SFCTMP_DEVICE_STAGES)
from gpuwm.core.ruc_contract import (  # noqa: E402
    WRF_SUPPORTED_NUM_SOIL_LAYERS)
from gpuwm.core.ruc_runtime import (C1SN, C2SN, DEFINED_ILNB,  # noqa: E402
                                    ISNCOVR_OPT, RucRuntimeParameters)
from test_ruc_column_batching import _columns  # noqa: E402
from test_ruc_device_column import _max_ulp  # noqa: E402

_PARAMS = RucRuntimeParameters()


def _run(values, ivgtyp, isltyp, *, leaves=None, stages=None, params=None):
    params = _PARAMS if params is None else params
    return ruc_land_surface_step(
        values, dt=12.0, ktau=2, zs=params.zs, ivgtyp=ivgtyp, isltyp=isltyp,
        em_core=0, ilnb=DEFINED_ILNB, ilnb_chain=False, c1sn=C1SN, c2sn=C2SN,
        isncovr_opt=ISNCOVR_OPT, mminlu=params.dataset_identifier,
        parameters=params.bundle, leaves=leaves, stages=stages)


def _worst_gap(host, device) -> dict[str, int]:
    worst = {}
    for name in sorted(RucLandSurfaceStep.__dataclass_fields__):
        gap = _max_ulp(getattr(host, name), getattr(device, name))
        if gap:
            worst[name] = gap
    return worst


@pytest.fixture
def forced_nine_level_tier(monkeypatch):
    """Route every RUC launcher through the SPECIALIZED loader at nine.

    ``ruc_module_defines(9)`` is empty by design, so the shipped nine-level
    run never touches the tiered path.  This makes it take it anyway, at the
    same geometry, which is the only way to exercise the ladder, the derived
    macros and the ``#if``-selected table against something that has an
    oracle.
    """
    real = ruc_tier.ruc_module_defines

    def always_specialize(nzs: int) -> tuple[tuple[str, int], ...]:
        real(nzs)                       # keep the admitted-set refusal
        return (("RUC_NZS", int(nzs)),)

    monkeypatch.setattr(ruc_tier, "ruc_module_defines", always_specialize)
    return always_specialize


def test_the_forced_tier_really_is_a_different_translation_unit(
        forced_nine_level_tier):
    """The fixture must actually change what NVRTC is handed.

    Without this the bitwise gate below could pass because the monkeypatch
    did nothing at all.
    """
    assert ruc_tier.ruc_module_defines(9) == (("RUC_NZS", 9),)
    specialized = module_source_int_defines("ruc", (("RUC_NZS", 9),))
    unspecialized = module_source("ruc")
    assert specialized != unspecialized
    # TWO, not one: the injected define, plus the ladder's own `#define
    # RUC_NZS 9` inside the `#ifndef` guard that the injection now skips.
    # That second one is exactly what makes the guard a guard.
    assert specialized.count("#define RUC_NZS 9\n") == 2
    assert unspecialized.count("#define RUC_NZS 9\n") == 1
    assert specialized.replace("#define RUC_NZS 9\n", "", 1) == unspecialized
    assert ruc_tier.ruc_kernel_source(9) == specialized


@pytest.mark.parametrize("snow,water,ice", [
    (False, 0, 0), (False, 5, 4), (True, 6, 5),
], ids=["snow-free", "snow-free+water+ice", "snow+water+ice"])
def test_the_tier_ladder_is_arithmetically_inert_at_nine(
        forced_nine_level_tier, snow, water, ice):
    """Every returned field, compiled through the tier, against the host.

    The host lane is untouched by the tier -- it is Python -- so this is the
    same 43-field oracle-backed comparison ``test_ruc_device_column.py``
    makes, with the only difference being which of the two loaders assembled
    the device module.  A single ULP here would mean the ``#if`` ladder or a
    derived macro changed the arithmetic, and the nine-level bit-identity
    claim in the commit messages would be false.
    """
    leaves = (RUC_SFCTMP_DEVICE_LEAVES if snow
              else RUC_SFCTMP_DEVICE_LEAVES_SNOW_FREE)
    values, ivgtyp, isltyp = _columns(48, snow=snow, water=water, ice=ice)
    host = _run(values, ivgtyp, isltyp)
    device = _run(values, ivgtyp, isltyp, leaves=leaves,
                  stages=RUC_SFCTMP_DEVICE_STAGES)

    worst = _worst_gap(host, device)
    assert worst == {}, (
        f"the RUC_NZS ladder moved the answer at the shipped geometry: "
        f"{worst}.  Acceptance is max_ulp 0 -- the ladder is supposed to be "
        "a preprocessor no-op at nine, and this is the leg that says so on "
        "the hardware rather than in PTX")


def test_the_forced_tier_gate_can_fail(forced_nine_level_tier):
    """Mutation control for the gate above.

    A comparison that cannot fail proves nothing.  Perturbing one input field
    must move at least one output, which shows the harness is actually
    plumbed to the device result rather than comparing a value with itself.
    """
    values, ivgtyp, isltyp = _columns(48, snow=False, water=0, ice=0)
    host = _run(values, ivgtyp, isltyp)

    perturbed = dict(values)
    field = "tso"
    bumped = np.array(perturbed[field], dtype=np.float32, copy=True)
    bumped[0, 0] = np.float32(bumped[0, 0] + np.float32(0.5))
    perturbed[field] = bumped
    device = _run(perturbed, ivgtyp, isltyp,
                  leaves=RUC_SFCTMP_DEVICE_LEAVES_SNOW_FREE,
                  stages=RUC_SFCTMP_DEVICE_STAGES)

    assert _worst_gap(host, device) != {}, (
        "a half-kelvin perturbation of one soil temperature changed nothing, "
        "so the comparison above is not reading the device result")


# ---------------------------------------------------------------------------
# The six-level geometry, host against device
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nzs", sorted(WRF_SUPPORTED_NUM_SOIL_LAYERS))
@pytest.mark.parametrize("snow,water,ice", [
    (False, 0, 0), (False, 5, 4), (True, 6, 5),
], ids=["snow-free", "snow-free+water+ice", "snow+water+ice"])
def test_the_host_and_device_columns_agree_at_every_admitted_geometry(
        nzs, snow, water, ice):
    """The same 43-field comparison, at SIX levels as well as nine.

    What this can and cannot say, stated so a later reader does not have to
    reconstruct it:

    * it CANNOT say gpuwm's six-level column is WRF's.  There is no
      six-level forecast oracle in this tree and this comparison is not one:
      ``gpuwm.core.ruc`` and ``gpuwm/core/kernels/ruc.cu`` are transcriptions
      of the same Fortran, so an error present in both is invisible to it.
    * it CAN say no error entered ONE of them, which is the failure mode a
      geometry lift actually has -- a level count or a depth that reached one
      lane and not the other.  It found exactly that on its first run (see
      the module docstring), so this is a gate with a body count, not a
      formality.

    Nine is parametrized alongside six deliberately: without it a green here
    would not distinguish "the six-level lanes agree" from "the comparison
    is not reading the device".
    """
    params = RucRuntimeParameters(num_soil_layers=nzs)
    values, ivgtyp, isltyp = _columns(48, snow=snow, water=water, ice=ice,
                                      nzs=nzs)
    host = _run(values, ivgtyp, isltyp, params=params)
    leaves = (RUC_SFCTMP_DEVICE_LEAVES if snow
              else RUC_SFCTMP_DEVICE_LEAVES_SNOW_FREE)
    device = _run(values, ivgtyp, isltyp, params=params, leaves=leaves,
                  stages=RUC_SFCTMP_DEVICE_STAGES)

    worst = _worst_gap(host, device)
    assert worst == {}, (
        f"host and device disagree at {nzs} soil levels: {worst}.  At six "
        "this is how a nine-level constant left behind in ruc.cu shows "
        "itself; at nine it would mean the tier moved the shipped answer")


@pytest.mark.parametrize("nzs", sorted(WRF_SUPPORTED_NUM_SOIL_LAYERS))
def test_the_geometry_comparison_can_fail(nzs):
    """Mutation control, at both geometries.

    A half-kelvin bump of one soil temperature must move at least one output
    field, or the comparison above is reading a constant.
    """
    params = RucRuntimeParameters(num_soil_layers=nzs)
    values, ivgtyp, isltyp = _columns(48, snow=False, water=0, ice=0, nzs=nzs)
    host = _run(values, ivgtyp, isltyp, params=params)

    perturbed = dict(values)
    bumped = np.array(perturbed["tso"], dtype=np.float32, copy=True)
    bumped[0, 0] = np.float32(bumped[0, 0] + np.float32(0.5))
    perturbed["tso"] = bumped
    device = _run(perturbed, ivgtyp, isltyp, params=params,
                  leaves=RUC_SFCTMP_DEVICE_LEAVES_SNOW_FREE,
                  stages=RUC_SFCTMP_DEVICE_STAGES)
    assert _worst_gap(host, device) != {}


def test_the_six_level_geometry_is_the_one_wrf_tabulates():
    """The comparison above must be running on WRF's six-level grid.

    Cheap, and it closes the hole where the two lanes agree beautifully on a
    geometry that is nobody's.  The depths come from
    ``gpuwm.core.ruc.ruc_soil_geometry``, which
    ``tests/test_soil_layer_geometry.py`` pins against WRF 4.7.1 real.exe.
    """
    six = RucRuntimeParameters(num_soil_layers=6)
    np.testing.assert_array_equal(
        np.asarray(six.zs, dtype=np.float32),
        np.array([0.0, 0.05, 0.20, 0.40, 1.60, 3.00], dtype=np.float32))
    assert six.restart_identity()["num_soil_layers"] == 6


def test_the_carried_state_names_are_the_ones_this_file_compares():
    """Guards against the field set shrinking underneath the gate."""
    carried = set(RUC_DRIVER_COLUMN_STATE) | set(RUC_DRIVER_PROFILE_STATE)
    assert carried <= set(RucLandSurfaceStep.__dataclass_fields__)
    assert len(RucLandSurfaceStep.__dataclass_fields__) >= 43


@pytest.mark.parametrize("nzs", sorted(WRF_SUPPORTED_NUM_SOIL_LAYERS))
def test_the_root_zone_refusal_names_the_geometry_it_refuses_for(nzs):
    """A refusal a user cannot act on, at both admitted geometries.

    The root-zone BOUND was un-pinned with the soil geometry -- the device
    lane rejects ``nroot >= nzs`` -- but its MESSAGE kept the nine-level
    literal, so at six levels ``nroot=6`` was refused with "is outside
    1..8", naming the rejected value as inside the range that rejected it.
    The host lane
    (:func:`gpuwm.core.ruc._root_count_field`) always read the message from
    ``nzs``, so the two lanes also disagreed about the same refusal.

    This pins the ceiling to the geometry on BOTH lanes and pins them to
    each other, which is what stops the literal coming back the next time
    only one of the two is edited.
    """
    from gpuwm.core.ruc import _root_count_field as host_root_count_field
    from gpuwm.core.ruc_gpu import _root_count_field as device_root_count_field

    shape = (2, 2)
    ceiling = nzs - 1
    valid = np.full(shape, ceiling, dtype=np.int32)
    # The ceiling itself is admitted on both lanes, so the bound is not
    # merely a message: it is off-by-one-checked here too.
    assert int(np.asarray(host_root_count_field(valid, shape, nzs=nzs)).max()) \
        == ceiling
    assert int(cp.asnumpy(
        device_root_count_field(valid, shape, nzs=nzs)).max()) == ceiling

    rejected = np.full(shape, nzs, dtype=np.int32)
    with pytest.raises(ValueError) as host_error:
        host_root_count_field(rejected, shape, nzs=nzs)
    with pytest.raises(ValueError) as device_error:
        device_root_count_field(rejected, shape, nzs=nzs)

    expected = f"RUC nroot {nzs} is outside 1..{ceiling}"
    assert str(host_error.value) == expected
    assert str(device_error.value) == expected
    # The defect this retires, stated as the thing that must not reappear:
    # the nine-level ceiling quoted at a six-level geometry.
    if nzs != 9:
        assert "1..8" not in str(device_error.value)
