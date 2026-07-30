"""The RUC column with its four arithmetic leaves on the card.

Batching ``LSMRUC``'s ``DO i`` (see ``tests/test_ruc_column_batching.py``)
took the host column from 1.711 to 0.479 ms warm and from 3.261 to 1.254 over
snow, and left it **flat**.  That flatness is the point: what remains is
genuine per-element work, and ``cProfile`` over the batched call names it --
``ruc_soil_properties`` 45%, ``ruc_soil_moisture_step`` 16%,
``_ruc_phase_partition`` 11%, ``ruc_soil_temperature_step`` 11%,
``_ruc_vilka`` 5%.  Every one of those sits inside ``ruc_soil_step`` or
``ruc_snow_soil_step``, both of which have had oracle-matched CUDA leaves for
weeks with nothing able to launch them at width.

Batching is what made them reachable.  ``ruc_surface_temperature_step`` now
takes a ``leaves`` mapping, and ``gpuwm.core.ruc_gpu.RUC_SFCTMP_DEVICE_LEAVES``
is the device set.  This file is the evidence that using it does not change
the answer.

The gates, in the order they matter:

1. the CUDA and host result dataclasses agree field for field and in order,
   which is what makes the host-facing wrapper a rename-free transfer rather
   than a mapping nobody checks;
2. the device leaves reproduce the host leaves **bitwise** on a whole
   ``sfctmp`` call -- max ULP is reported, not assumed;
3. a six-step driver trajectory with the device leaves is bitwise against
   the same trajectory on the host, warm and snow-covered, with water and
   sea-ice columns present; and
4. the per-column cost across the published widths, so "is it still flat"
   is answered with a number.

Every comparison is paired in-process against the same tree.  A device run
compared against a recorded digest would also pass if both had drifted.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from gpuwm.core.ruc import (RUC_DRIVER_COLUMN_STATE,  # noqa: E402
                            RUC_DRIVER_PROFILE_STATE,
                            RUC_SFCTMP_HOST_LEAVES, RUC_SFCTMP_HOST_STAGES,
                            RucLandSurfaceStep,
                            RucSeaIceStep, RucSnowSeaIceStep, RucSnowSoilStep,
                            RucSoilStep, ruc_land_surface_step)
from gpuwm.core.ruc_gpu import (RUC_DEVICE_ARRAYS,  # noqa: E402
                                RUC_SFCTMP_DEVICE_LEAVES,
                                RUC_SFCTMP_DEVICE_LEAVES_RESIDENT,
                                RUC_SFCTMP_DEVICE_LEAVES_SNOW_FREE,
                                RUC_SFCTMP_DEVICE_STAGES,
                                RUC_SFCTMP_DEVICE_STAGES_RESIDENT,
                                RucSeaIceStepCuda, RucSnowSeaIceStepCuda,
                                RucSnowSoilStepCuda, RucSoilStepCuda,
                                ruc_tanhf_glibc)
from gpuwm.core.ruc_runtime import (C1SN, C2SN, DEFINED_ILNB,  # noqa: E402
                                    ISNCOVR_OPT, RucRuntimeParameters)
from test_ruc_column_batching import _columns  # noqa: E402

_PARAMS = RucRuntimeParameters()
_PRODUCTION_D04_COLUMNS = 600 * 600


def _run(values, ivgtyp, isltyp, *, ktau=2, leaves=None, stages=None, dt=12.0):
    return ruc_land_surface_step(
        values, dt=dt, ktau=ktau, zs=_PARAMS.zs, ivgtyp=ivgtyp,
        isltyp=isltyp, ilnb=DEFINED_ILNB, ilnb_chain=False, c1sn=C1SN,
        c2sn=C2SN, isncovr_opt=ISNCOVR_OPT,
        mminlu=_PARAMS.dataset_identifier, parameters=_PARAMS.bundle,
        leaves=leaves, stages=stages)


_CARRIED = tuple(RUC_DRIVER_COLUMN_STATE) + tuple(RUC_DRIVER_PROFILE_STATE)


def _advance(values, result):
    out = dict(values)
    for name in _CARRIED:
        out[name] = np.array(
            getattr(result, name), dtype=np.float32, copy=True)
    return out


def _digest(result) -> str:
    h = hashlib.sha256()
    for name in sorted(RucLandSurfaceStep.__dataclass_fields__):
        array = np.ascontiguousarray(np.asarray(getattr(result, name)))
        h.update(name.encode())
        h.update(str(array.dtype).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def _max_ulp(left, right) -> int:
    """Largest ULP gap over two float32 fields, or -1 if bitwise equal."""
    a = np.asarray(left)
    b = np.asarray(right)
    assert a.shape == b.shape and a.dtype == b.dtype
    if a.dtype != np.float32:
        return 0 if np.array_equal(a, b) else 1 << 30
    if a.tobytes() == b.tobytes():
        return 0
    ia = a.view(np.int32).astype(np.int64)
    ib = b.view(np.int32).astype(np.int64)
    # Map the sign-magnitude int32 ordering onto a monotone integer line so a
    # gap that straddles zero is counted once, not as 2**31.
    ia = np.where(ia < 0, np.int64(1) << 31, np.int64(0)) * 0 + np.where(
        ia < 0, -(ia & 0x7FFFFFFF), ia)
    ib = np.where(ib < 0, -(ib & 0x7FFFFFFF), ib)
    return int(np.abs(ia - ib).max())


# ---------------------------------------------------------------------------
# 1.  the wrapper is a rename-free transfer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("host,device", [
    (RucSoilStep, RucSoilStepCuda),
    (RucSeaIceStep, RucSeaIceStepCuda),
    (RucSnowSoilStep, RucSnowSoilStepCuda),
    (RucSnowSeaIceStep, RucSnowSeaIceStepCuda),
])
def test_the_cuda_and_host_leaf_results_agree_field_for_field(host, device):
    """``_host_facing_leaf`` transfers by name; that name list must match.

    Checked in ORDER as well as by set, because a reordering is exactly the
    kind of change that keeps every name valid and silently moves a column of
    numbers into the wrong field.
    """
    assert (list(host.__dataclass_fields__)
            == list(device.__dataclass_fields__)), (
        f"{host.__name__} and {device.__name__} have drifted; "
        "gpuwm.core.ruc_gpu._host_facing_leaf rebuilds the host result from "
        "the device one by name and would silently mis-bind")


def test_the_device_leaf_set_covers_exactly_the_host_leaf_set() -> None:
    assert set(RUC_SFCTMP_DEVICE_LEAVES) == set(RUC_SFCTMP_HOST_LEAVES)


def test_an_unknown_leaf_override_is_refused() -> None:
    """Fail-closed, so a typo does not silently leave a leaf on the host."""
    values, ivgtyp, isltyp = _columns(4, snow=False)
    with pytest.raises(KeyError, match="unknown RUC sfctmp leaf"):
        _run(values, ivgtyp, isltyp, leaves={"sol": object()})


# ---------------------------------------------------------------------------
# 2 and 3.  the answer does not change
# ---------------------------------------------------------------------------

#: The four fields a snow-covered column MOVED on an RTX 5090 through the
#: batched driver before 2026-07-26, and the ULP each moved.  Kept because the
#: gate below now asserts they are all zero, and a bare ``worst == {}`` says
#: nothing about what it is defending against.
#:
#: All four came out of ``SOILMOIST`` -- ``infiltr`` is its infiltration
#: capacity and ``runoff1`` its surface runoff, and ``sfcrunoff``/``acrunoff``
#: are the driver's accumulators of ``runoff1`` at ``:1038-1044``.  The cause
#: was ``ruc.cu``'s ``powf(acrt, 2.0f)``: the one ``**`` site in the file left
#: on the CUDA device libm while the host used the float64-evaluated,
#: rounded-once form, amplified by the ``1 - exp(-acrt)*SUM`` cancellation at
#: ``:5985``.  See ``docs/wrf_ruc_runtime_admission.md`` and
#: ``tests/test_ruc_gpu.py::test_ruc_soil_moisture_cuda_agrees_off_fixture``.
_SNOW_COLUMN_DIVERGENCE_BEFORE_2026_07_26 = {
    "infiltr": 10, "acrunoff": 4, "sfcrunoff": 4, "runoff1": 2,
}


@pytest.mark.parametrize("water,ice", [(0, 0), (5, 4)],
                         ids=["all-land", "water+ice"])
@pytest.mark.parametrize("stages,stage_label", [
    (None, "host stages"), (RUC_SFCTMP_DEVICE_STAGES, "device stages"),
])
def test_the_snow_free_device_leaves_reproduce_the_host_call_bitwise(
        water, ice, stages, stage_label):
    """Snow-free columns, every returned field, both stage sets.

    ``soil`` and ``sea_ice`` on the device, on a grid with no snow.  Max ULP 0
    across all 43 returned fields, including the sea-ice columns, with the
    snow-preparation stage on the host and on the card.
    """
    values, ivgtyp, isltyp = _columns(64, snow=False, water=water, ice=ice)
    host = _run(values, ivgtyp, isltyp)
    device = _run(values, ivgtyp, isltyp,
                  leaves=RUC_SFCTMP_DEVICE_LEAVES_SNOW_FREE, stages=stages)

    worst = {}
    for name in sorted(RucLandSurfaceStep.__dataclass_fields__):
        gap = _max_ulp(getattr(host, name), getattr(device, name))
        if gap:
            worst[name] = gap
    assert worst == {}, (
        f"the snow-free device leaves with {stage_label} are not bitwise "
        f"against the host: {worst}.  Acceptance for a RUC leaf is max_ulp 0; "
        "report the number, never widen the gate")


@pytest.mark.parametrize("leaves,label", [
    (RUC_SFCTMP_DEVICE_LEAVES_SNOW_FREE, "snow-free subset"),
    (RUC_SFCTMP_DEVICE_LEAVES, "full set"),
])
@pytest.mark.parametrize("stages,stage_label", [
    (None, "host stages"), (RUC_SFCTMP_DEVICE_STAGES, "device stages"),
])
def test_a_snow_covered_column_is_bitwise_on_the_device(
        leaves, label, stages, stage_label):
    """The gate that was two-sided against a divergence until 2026-07-26.

    Every RUC CUDA leaf was ``max_ulp 0`` against its own WRF fixture and they
    still disagreed here, by up to 10 ULP of ``infiltr`` -- the "a mirror is
    not an oracle" trap, and only driver-level composition over a perturbed
    snow grid could surface it.  It was ONE token: ``powf(acrt, 2.0f)`` in
    ``ruc_soil_moisture_step``, the single ``**`` site in ``ruc.cu`` still on
    the CUDA device libm while the host used the float64-evaluated,
    rounded-once form that the rest of the file and the whole snow lane agree
    on.  ``soilmoist.csv`` could not see it: three of its four cases carry
    ``soilice = 0``, so ``:5972`` is false, and the fourth multiplies the
    frozen reduction by an ``infmax1`` of exactly zero.

    Parametrised over BOTH device leaf sets because that is what located the
    fault -- the snow-free subset diverged too, so it was ``soil`` on the
    mosaic snow-free fraction and never ``snowsoil``.
    """
    values, ivgtyp, isltyp = _columns(48, snow=True, water=6, ice=5)
    host = _run(values, ivgtyp, isltyp)
    device = _run(values, ivgtyp, isltyp, leaves=leaves, stages=stages)

    worst = {}
    for name in sorted(RucLandSurfaceStep.__dataclass_fields__):
        gap = _max_ulp(getattr(host, name), getattr(device, name))
        if gap:
            worst[name] = gap

    assert worst == {}, (
        f"the {label} with {stage_label} left the host on a snow column: "
        f"{worst}.  Before 2026-07-26 this measured "
        f"{_SNOW_COLUMN_DIVERGENCE_BEFORE_2026_07_26}; if those fields are "
        "back, the acrt square in gpuwm/core/kernels/ruc.cu has regressed to "
        "the device libm.  Acceptance is max_ulp 0; never widen the gate")


# ---------------------------------------------------------------------------
# 2b.  the stage seam
# ---------------------------------------------------------------------------

def test_the_device_stage_set_covers_exactly_the_host_stage_set() -> None:
    assert set(RUC_SFCTMP_DEVICE_STAGES) == set(RUC_SFCTMP_HOST_STAGES)


def test_an_unknown_stage_override_is_refused() -> None:
    """Fail-closed, so a typo does not silently leave a stage on the host."""
    values, ivgtyp, isltyp = _columns(4, snow=False)
    with pytest.raises(KeyError, match="unknown RUC sfctmp stage"):
        _run(values, ivgtyp, isltyp, stages={"snowprep": object()})


def test_the_device_snow_prep_stage_refuses_a_foreign_parameter_bundle():
    """The kernel indexes tables built from the DEFAULT bundle.

    A caller who supplies a different one must be refused rather than
    silently given the default tables.
    """
    import dataclasses

    stage = RUC_SFCTMP_DEVICE_STAGES["snow_prep"]
    table = _PARAMS.bundle.vegetation_for(_PARAMS.dataset_identifier)
    rows = list(table.rows)
    rows[0] = dataclasses.replace(rows[0], z0=np.float32(9.0))
    mapping = dict(_PARAMS.bundle.vegetation)
    for key, value in mapping.items():
        if value is table:
            mapping[key] = dataclasses.replace(table, rows=tuple(rows))
            break
    else:  # pragma: no cover - the bundle always holds the table it hands out
        pytest.fail("the parameter bundle did not contain its own table")
    bundle = dataclasses.replace(_PARAMS.bundle, vegetation=mapping)

    with pytest.raises(ValueError, match="differs in z0tbl"):
        stage({}, delt=12.0, ivgtyp=None, iland=None,
              mminlu=_PARAMS.dataset_identifier, bundle=bundle)


def test_a_six_step_snow_free_trajectory_is_bitwise_on_the_device() -> None:
    """Paired in-process, so neither side is a recorded string.

    Six steps with the driver's own output fed back in, so the accumulators
    (SFCEVP, ACSNOW, SFCRUNOFF, UDRUNOFF, ACRUNOFF) and the carried soil
    state are compared too, not just one call's fluxes.
    """
    host_values, ivgtyp, isltyp = _columns(48, snow=False, water=6, ice=5)
    device_values = dict(host_values)

    for ktau in range(1, 7):
        host = _run(host_values, ivgtyp, isltyp, ktau=ktau)
        device = _run(device_values, ivgtyp, isltyp, ktau=ktau,
                      leaves=RUC_SFCTMP_DEVICE_LEAVES_SNOW_FREE)
        assert _digest(host) == _digest(device), (
            f"step {ktau}: the device trajectory left the host trajectory")
        host_values = _advance(host_values, host)
        device_values = _advance(device_values, device)


def test_the_bitwise_gate_sees_a_leaf_left_on_the_host() -> None:
    """The falsification.

    A gate that compares two runs of the same code passes for free.  Move one
    leaf back to the host while perturbing nothing else and the comparison
    must still agree -- that is the point -- so the falsification instead
    perturbs an input and requires the comparison to reject it.
    """
    values, ivgtyp, isltyp = _columns(24, snow=False)
    device = _run(values, ivgtyp, isltyp,
                  leaves=RUC_SFCTMP_DEVICE_LEAVES_SNOW_FREE)

    nudged = dict(values)
    moisture = np.array(values["soilmois"], dtype=np.float32, copy=True)
    moisture[0, 0] = np.nextafter(moisture[0, 0], np.float32(1.0))
    nudged["soilmois"] = moisture

    other = _run(nudged, ivgtyp, isltyp,
                 leaves=RUC_SFCTMP_DEVICE_LEAVES_SNOW_FREE)
    assert _digest(other) != _digest(device), (
        "one ULP of the entry soilmois changed nothing on the device path, "
        "so the bitwise comparisons above cannot see a real difference")

    # ...and moving a single leaf back to the host must NOT change anything,
    # which is the composability claim the device set rests on.
    partial = _run(values, ivgtyp, isltyp,
                   leaves={"soil": RUC_SFCTMP_DEVICE_LEAVES["soil"]})
    assert _digest(partial) == _digest(device), (
        "running only the `soil` leaf on the device disagreed with running "
        "the snow-free pair, on a warm all-land grid where `sea_ice` is never "
        "called; the leaf set is not composable")


# ---------------------------------------------------------------------------
# 4.  is it still flat?
# ---------------------------------------------------------------------------

_COST_COUNTS = (48, 192, 768, 3072)


def test_the_device_column_cost_across_width(capsys) -> None:
    """Reported, not asserted against a threshold.

    The absolute number belongs to this card and this driver.  What the gate
    asserts is only that the device path is not CATASTROPHICALLY slower than
    the host one, because the interesting failure mode of a per-batch
    transfer boundary is that it costs more than the work it moved.  The
    numbers themselves go in ``docs/wrf_ruc_runtime_admission.md``.
    """
    rows = []
    for n in _COST_COUNTS:
        values, ivgtyp, isltyp = _columns(n, snow=False)
        _run(values, ivgtyp, isltyp,
             leaves=RUC_SFCTMP_DEVICE_LEAVES_SNOW_FREE)
        cp.cuda.runtime.deviceSynchronize()

        def timed(leaves):
            start = time.perf_counter()
            _run(values, ivgtyp, isltyp, leaves=leaves)
            cp.cuda.runtime.deviceSynchronize()
            return (time.perf_counter() - start) / n * 1e3

        host = min(timed(None) for _ in range(2))
        device = min(
            timed(RUC_SFCTMP_DEVICE_LEAVES_SNOW_FREE) for _ in range(2))
        rows.append((n, host, device))

    with capsys.disabled():
        print()
        for n, host, device in rows:
            print(f"  {n:6d} columns: host {host:7.4f}  device {device:7.4f} "
                  f"ms/column  ({host / device:.2f}x)")
        widest = rows[-1]
        print(f"  one {_PRODUCTION_D04_COLUMNS}-column call: "
              f"{widest[2] * _PRODUCTION_D04_COLUMNS / 1e3:.0f} s device, "
              f"{widest[1] * _PRODUCTION_D04_COLUMNS / 1e3:.0f} s host")

    assert rows[-1][2] < 4.0 * rows[-1][1], (
        f"the device path costs {rows[-1][2] / rows[-1][1]:.1f}x the host "
        "path at the widest point measured; the per-batch transfer boundary "
        "is costing more than the work it moved")


# ---------------------------------------------------------------------------
# 5.  the whole column device-resident
# ---------------------------------------------------------------------------
#
# Everything above keeps the DISPATCH on the host: the leaves and the stage
# run as kernels, but every field they take crosses the boundary going in and
# every field they return crosses it coming back, once per field, each copy
# separately synchronised.  At 24,576 columns that was more than three hundred
# crossings for one land-surface call and 46% of a warm call's wall clock,
# 67% of a snow one's -- and the host dispatch also still ran
# ``_ruc_tanh_array``, a Python loop over fdlibm's tanh, once per snow-covered
# column.
#
# ``arrays=RUC_DEVICE_ARRAYS`` with the RESIDENT leaf and stage sets removes
# the boundary entirely: the driver's masking, gathers, scatters and mosaic
# recombination are kernels over the same device arrays the leaves wrote.
# These are the gates that say doing that did not change the answer.

_RESIDENT_WIDTHS = (512, 4096, 24576)


def _to_device(values):
    return {name: cp.asarray(np.ascontiguousarray(array))
            for name, array in values.items()}


def _resident(values, ivgtyp, isltyp, *, ktau=2, dt=12.0):
    return ruc_land_surface_step(
        _to_device(values), dt=dt, ktau=ktau, zs=_PARAMS.zs,
        ivgtyp=cp.asarray(ivgtyp), isltyp=cp.asarray(isltyp),
        ilnb=DEFINED_ILNB, ilnb_chain=False, c1sn=C1SN, c2sn=C2SN,
        isncovr_opt=ISNCOVR_OPT, mminlu=_PARAMS.dataset_identifier,
        parameters=_PARAMS.bundle,
        leaves=RUC_SFCTMP_DEVICE_LEAVES_RESIDENT,
        stages=RUC_SFCTMP_DEVICE_STAGES_RESIDENT,
        arrays=RUC_DEVICE_ARRAYS)


def _as_host(result):
    return RucLandSurfaceStep(**{
        name: cp.asnumpy(cp.asarray(getattr(result, name)))
        for name in RucLandSurfaceStep.__dataclass_fields__})


def test_the_device_tanh_is_the_host_tanh(capsys) -> None:
    """``ruc.cu``'s ``ruc_tanhf_glibc_array`` against ``ruc._f32_tanh``.

    The snow-cover rebuild at ``module_sf_ruclsm.F:2087`` evaluates
    ``TANH(snhei / (2.5*min(0.2,znt) * (rhosn/rhonewsn)))``.  The arguments it
    reaches are small and positive, which is the branch of fdlibm's reduction
    that goes through ``expm1(-2x)`` -- so this sweeps that range densely
    rather than sampling it, plus the two branch boundaries at 1.0 and 22.0
    and the subnormal cutoff, where the reduction changes form.

    The host spelling is not glibc's ``tanhf`` and does not claim to be; see
    ``_f32_tanh``'s docstring.  What must hold is that the two spellings of
    the SAME reduction agree bit for bit, because otherwise a snow-covered
    forecast answers differently depending on where the dispatch ran.
    """
    from gpuwm.core.ruc import _ruc_tanh_array

    arguments = np.concatenate([
        np.linspace(0.0, 4.0, 40001, dtype=np.float32),
        np.linspace(0.9999, 1.0001, 2001, dtype=np.float32),
        np.linspace(21.99, 22.01, 2001, dtype=np.float32),
        np.array([0.0, 2.0 ** -29, 2.0 ** -28, -2.0 ** -28, 1.0, -1.0,
                  22.0, -22.0, 1e-30, 100.0, -100.0], dtype=np.float32),
        -np.linspace(0.0, 4.0, 4001, dtype=np.float32),
    ]).astype(np.float32)

    host = _ruc_tanh_array(arguments)
    device = cp.asnumpy(ruc_tanhf_glibc(cp.asarray(arguments)))
    gap = _max_ulp(host, device)
    with capsys.disabled():
        print(f"\n  ruc_tanhf_glibc over {arguments.size} arguments: "
              f"max_ulp {gap}")
    assert gap == 0, (
        f"the CUDA tanh reduction is {gap} ULP from the host one; the two "
        "are transcriptions of one fdlibm reduction and acceptance is "
        "max_ulp 0.  Never widen the gate")


@pytest.mark.parametrize("snow", [False, True], ids=["warm", "snow"])
@pytest.mark.parametrize("n", _RESIDENT_WIDTHS)
def test_the_device_resident_driver_is_bitwise_at_width(n, snow, capsys):
    """The whole ``LSMRUC`` call on the card, against the host driver.

    Paired in-process against the same tree, so neither side is a recorded
    digest, and reported as a ULP rather than asserted as a boolean so a
    regression says how far it moved.

    Width matters here for a reason this lane paid for once already: every
    bitwise gate it had run was 48 or 64 columns, and widening to 512 exposed
    four ``**`` sites whose host and device spellings disagreed on 6% of the
    arguments they span.  A gate that only ever runs narrow cannot see a
    divergence whose probability per column is small.
    """
    values, ivgtyp, isltyp = _columns(
        n, snow=snow, water=max(4, n // 64), ice=max(3, n // 128))
    host = _run(values, ivgtyp, isltyp)
    device = _as_host(_resident(values, ivgtyp, isltyp))

    worst = {}
    for name in sorted(RucLandSurfaceStep.__dataclass_fields__):
        gap = _max_ulp(getattr(host, name), getattr(device, name))
        if gap:
            worst[name] = gap
    with capsys.disabled():
        print(f"\n  {n} columns {'snow' if snow else 'warm'}: "
              f"{len(RucLandSurfaceStep.__dataclass_fields__)} fields, "
              f"max_ulp {max(worst.values()) if worst else 0}")
    assert worst == {}, (
        f"the device-resident driver left the host driver at {n} columns "
        f"({'snow' if snow else 'warm'}): {worst}.  Acceptance is max_ulp 0; "
        "never widen the gate")


def test_the_resident_gate_can_fail() -> None:
    """A comparison that has never rejected anything is not evidence.

    One ULP of the entry soil moisture, and nothing else, must move the
    device-resident answer.

    The perturbed column has to be a LAND column, and that is not a detail:
    the first version of this test nudged column 0 of a grid whose first
    eight columns are water, and it FAILED -- correctly.  ``LSMRUC``'s water
    arm (``:824-850``) never reads ``SMOIS`` and overwrites the whole soil
    profile before returning, so a one-ULP change there is genuinely
    unobservable.  ``_columns`` lays the grid out as water, then sea ice,
    then land, so the index is derived from those counts rather than
    assumed, and the layout is asserted before the perturbation.
    """
    water_columns, ice_columns = 8, 6
    values, ivgtyp, isltyp = _columns(
        512, snow=True, water=water_columns, ice=ice_columns)
    land = water_columns + ice_columns
    assert values["xland"][land] < np.float32(1.5)
    assert values["xice"][land] < np.float32(0.5)

    reference = _as_host(_resident(values, ivgtyp, isltyp))

    nudged = dict(values)
    moisture = np.array(values["soilmois"], dtype=np.float32, copy=True)
    moisture[0, land] = np.nextafter(moisture[0, land], np.float32(1.0))
    nudged["soilmois"] = moisture
    other = _as_host(_resident(nudged, ivgtyp, isltyp))

    assert _digest(reference) != _digest(other), (
        "one ULP of the entry soilmois on a LAND column changed nothing on "
        "the device-resident path, so the bitwise gates above cannot see a "
        "real difference")

    # ...and the water arm really is the reason the first spelling failed,
    # so the paragraph above is a measurement rather than an explanation.
    water_nudge = dict(values)
    moisture = np.array(values["soilmois"], dtype=np.float32, copy=True)
    moisture[0, 0] = np.nextafter(moisture[0, 0], np.float32(1.0))
    water_nudge["soilmois"] = moisture
    assert _digest(_as_host(_resident(water_nudge, ivgtyp, isltyp))) ==         _digest(reference), (
        "a one-ULP change on a WATER column moved the answer; LSMRUC:824-850 "
        "overwrites the soil profile without reading it, so either the water "
        "arm has changed or this grid is not laid out the way it says")


def test_the_resident_driver_returns_device_arrays() -> None:
    """The claim is residency, so check the arrays never came home.

    A resident driver that quietly returned host arrays would still be
    bitwise and would have paid the boundary anyway, which is the whole cost
    this conversion exists to remove.
    """
    values, ivgtyp, isltyp = _columns(64, snow=True, water=4, ice=3)
    result = _resident(values, ivgtyp, isltyp)
    host_side = [name for name in RucLandSurfaceStep.__dataclass_fields__
                 if not isinstance(getattr(result, name), cp.ndarray)]
    assert host_side == [], (
        f"the device-resident driver returned host arrays for {host_side}")


def test_the_resident_driver_refuses_the_ilnb_chain() -> None:
    """Fail-closed rather than silently slow.

    ``ilnb_chain=True`` reproduces WRF's uninitialised ``ilnb``, which is a
    genuine per-column sequential dependence: column i's seed is column
    i-1's answer.  Dispatching one column at a time to the card would be
    slower than the host and is not what a forecast runs, so it is refused
    rather than accepted and quietly ruinous.
    """
    values, ivgtyp, isltyp = _columns(16, snow=False)
    with pytest.raises(ValueError, match="ilnb_chain"):
        ruc_land_surface_step(
            _to_device(values), dt=12.0, ktau=2, zs=_PARAMS.zs,
            ivgtyp=cp.asarray(ivgtyp), isltyp=cp.asarray(isltyp),
            ilnb=DEFINED_ILNB, ilnb_chain=True, c1sn=C1SN, c2sn=C2SN,
            isncovr_opt=ISNCOVR_OPT, mminlu=_PARAMS.dataset_identifier,
            parameters=_PARAMS.bundle,
            leaves=RUC_SFCTMP_DEVICE_LEAVES_RESIDENT,
            stages=RUC_SFCTMP_DEVICE_STAGES_RESIDENT,
            arrays=RUC_DEVICE_ARRAYS)


def test_a_six_step_resident_trajectory_is_bitwise() -> None:
    """Six steps, so the accumulators and the carried soil state agree too."""
    host_values, ivgtyp, isltyp = _columns(512, snow=True, water=8, ice=6)
    device_values = dict(host_values)

    for ktau in range(1, 7):
        host = _run(host_values, ivgtyp, isltyp, ktau=ktau)
        device = _as_host(_resident(device_values, ivgtyp, isltyp, ktau=ktau))
        assert _digest(host) == _digest(device), (
            f"step {ktau}: the device-resident trajectory left the host one")
        host_values = _advance(host_values, host)
        device_values = _advance(device_values, device)
