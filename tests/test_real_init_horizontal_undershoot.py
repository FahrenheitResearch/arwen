"""The horizontal operator's own undershoot on the direct specific-humidity lane.

An ERA5 (keyless ARCO) window sized for a 16 GiB card refused in
initialization with ``specific humidity must be finite in [0, 1)`` while
the same centre sized for an 8 GiB card ran end to end.  Nothing in the
forcing was out of range: all 1,497,834 specific-humidity values in the
published file are positive, the smallest being 2.533852239139378e-09
(``proof/source-spfh-extremes-tip.txt``, which carries the fetch command,
the published file's sha256 and the count, minimum and maximum read back
through this tree's own decoder).  The refused values are made by WPS's
overlapping-parabolic ``sixteen_pt``, which this tree runs to map that
source onto the model grid and which overshoots in both directions.
Whether a window met a cell where it undershoots is a question of how
much ground the window covers, so the SIZING decided whether the forcing
was admissible.

The numbers below are measured, at 27.5N 82.5W, on the 12/3 km ladder,
ERA5 cycle 2025-08-15T00.  The mapped probes were taken on an RTX 5070 Ti
and the device-memory peaks on both that card and an RTX 4090; each one
names the receipt it came from.
"""

from __future__ import annotations

import dataclasses
import datetime

import numpy as np
import pytest

from gpuwm.config import RunConfig
from gpuwm.core.grid import make_vertical_coord
from gpuwm.ingest import real as real_module
from gpuwm.ingest.horiz import (
    WPS_PARABOLIC_NEGATIVE_WEIGHT,
    HorizontalSnapshot,
    parabolic_undershoot_floor,
)
from gpuwm.verify.wps_masked_oracle import _wps_oned
from gpuwm.ingest.real import (
    _specific_humidity_to_mixing_ratio,
    initialize_real,
)

#: The root domain the 16 GiB sizing emits at that centre, and its nest:
#: 278x222 / 560x448 since A163 measured the forecast margin at 1.13 of
#: the subtotal (276x220 / 552x440 on the plan's 1.15).
SIZED_16GIB_LAYOUTS = [(278, 222), (560, 448)]
#: The plan whose run measured MEASURED_PEAK_BYTES: the 16 GiB sizing's
#: layouts at the plan's 1.15 margin.
MEASURED_16GIB_LAYOUTS = [(276, 220), (552, 440)]
#: The 8 GiB sizing of the same centre, which never met the undershoot
#: (154x124 / 304x248 on the plan's 1.15 margin).
SIZED_8GIB_LAYOUTS = [(156, 124), (312, 248)]
#: The source field the mapper was handed for the root's first valid time
#: (ERA5 SPFH, 37 levels, 117x173 source cells, 748,917 values), which is
#: also where the whole file's two extremes fall: its other valid time
#: runs 7.580820238217711e-08 to 0.021298227831721306, and neither time
#: holds a negative or a non-finite value.  Receipt:
#: proof/source-spfh-extremes-tip.txt, re-fetched and re-read at this
#: branch's tip.
SOURCE_SPFH_MAX = 0.02342507615685463
SOURCE_SPFH_MIN = 2.533852239139378e-09
#: What ``sixteen_pt`` made of it on the 276x220 root: 48 cells below zero
#: in 28 columns, on levels 15-19 and 23-24
#: (proof/measure-16gib-mapped-spfh.log).
MAPPED_SPFH_MIN = -6.367064634e-05
MAPPED_SPFH_MAX = 0.02349000052
#: ... and what it made of the 154x124 root of the same cycle: nothing
#: (proof/measure-8gib-mapped-spfh.log).
MAPPED_8GIB_SPFH_MIN = 2.884659807e-06
#: Peak device memory the 16 GiB-sized plan actually took, sampled once a
#: second across a 120 model-second run of both of its domains on an
#: otherwise idle card.  The plan has been run five times: 11542 MiB on an
#: RTX 5070 Ti, then 11766, 11724, 11974 and 11638 MiB on an RTX 4090
#: whose device pool was capped at this card class for the run.  The
#: sampled peak moves a couple of hundred MiB between runs of the same
#: plan, so the pin is the largest of the five, 11974 MiB
#: (proof/gpu-peak-mib-16gib-tip.txt; the fifth is
#: proof/gpu-peak-mib-16gib-tip3.txt).
MEASURED_PEAK_BYTES = 11974 * 1024 * 1024

NY, NX, NZ, P_TOP = 3, 4, 40, 5000.0


def _worst_case_patch(maximum: float, minimum: float = 1.0e-20
                     ) -> np.ndarray:
    """The 4x4 source stencil that drives ``sixteen_pt`` to its floor.

    The tensor-product weight of a stencil point is negative exactly when
    one of its two axis indices is an OUTER one, so the operator reaches
    its lower envelope when those eight points carry the source maximum
    and the other eight, whose weights sum to ``1 + 9/32``, carry the
    source minimum.  ``_wps_oned`` collapses a stencil whose inner points
    are exactly zero, the same degeneracy the mapping plan substitutes a
    tiny value for, so a source minimum of zero is that tiny value here.
    """
    patch = np.full((4, 4), minimum)
    for j in range(4):
        for i in range(4):
            if (j in (0, 3)) ^ (i in (0, 3)):
                patch[j, i] = maximum
    return patch


def test_the_envelope_is_the_operators_own_and_not_a_borrowed_constant():
    """WPS ``oned`` twice is ``sixteen_pt``; its floor is -9/32 of the max.

    This is the pin: the admission bound and the operator that makes the
    values it admits cannot drift apart, because the bound is evaluated
    here against the operator itself.
    """
    patch = _worst_case_patch(SOURCE_SPFH_MAX)
    rows = [_wps_oned(0.5, *patch[j]) for j in range(4)]
    worst = float(_wps_oned(0.5, *rows))
    assert worst == pytest.approx(
        -WPS_PARABOLIC_NEGATIVE_WEIGHT * SOURCE_SPFH_MAX, rel=1e-12)
    assert worst / SOURCE_SPFH_MAX == pytest.approx(-0.28125, rel=1e-12)
    # The recorded floor sits at or just below what the operator can do,
    # by the FP32 slack and nothing more.
    floor = parabolic_undershoot_floor(patch)
    assert floor <= worst
    assert floor == pytest.approx(worst, rel=2.0e-5)
    # The measured undershoot is INSIDE that envelope by two orders of
    # magnitude: 0.27 percent of the source maximum where 28 is allowed.
    assert floor < MAPPED_SPFH_MIN < 0.0
    assert MAPPED_SPFH_MIN / SOURCE_SPFH_MAX == pytest.approx(
        -0.002718, rel=1e-3)


def test_the_envelope_reads_a_windowed_operand_the_plan_will_actually_map():
    """A windowed source is the case this envelope exists for.

    ``WindowedAtmosphericSnapshot.operand`` hands the mapper a typed
    wrapper holding the values over the union of every support the plan
    can select, so that ``for_support`` can address it in original-source
    index space.  It is not an array: reading ``min`` straight off it
    raised ``AttributeError`` and took down preparation for every source
    small enough to window, which is the same sizing question that
    produced this lane.  The envelope is the one the values give.
    """
    from gpuwm.ingest.atmospheric_window import (AtmosphericWindow,
                                                 WindowedAtmosphericField)

    values = np.array(_worst_case_patch(SOURCE_SPFH_MAX),
                      dtype=np.float32)[None, :, :]
    window = AtmosphericWindow((8, 9), (2, 6), (3, 7))
    operand = WindowedAtmosphericField(values, window)
    assert parabolic_undershoot_floor(operand) == parabolic_undershoot_floor(
        values)


def test_the_envelope_is_two_sided_and_the_mapper_claims_nothing():
    """A source of its own is mapped; the range claim is made downstream.

    The envelope lowers by what the operator can do to the source
    MINIMUM as well, ``1 + 9/32`` of it, so a forcing carrying a small
    negative of its own -- a decode or a spectral-transform residual --
    is mapped and judged on the mapped field, which is where this lane
    reproduced anything at all.  No run has produced a negative source
    specific humidity: the measured ARCO file's minimum is
    +2.533852239139378e-09, and every route without specific-humidity
    authority mapped and initialized such a value before this envelope
    existed.
    """
    source = np.array([[SOURCE_SPFH_MIN, SOURCE_SPFH_MAX]])
    assert parabolic_undershoot_floor(source) < 0.0

    decode_residual = -1.0e-9
    patch = _worst_case_patch(SOURCE_SPFH_MAX, decode_residual)
    rows = [_wps_oned(0.5, *patch[j]) for j in range(4)]
    worst = float(_wps_oned(0.5, *rows))
    assert worst == pytest.approx(
        (1.0 + WPS_PARABOLIC_NEGATIVE_WEIGHT) * decode_residual
        - WPS_PARABOLIC_NEGATIVE_WEIGHT * SOURCE_SPFH_MAX, rel=1e-12)
    floor = parabolic_undershoot_floor(patch)
    assert floor <= worst
    assert floor == pytest.approx(worst, rel=2.0e-5)
    # ... and it is BELOW the envelope of the same source without that
    # value, by the amount the operator can amplify it to.
    assert floor < parabolic_undershoot_floor(
        _worst_case_patch(SOURCE_SPFH_MAX))

    # A POSITIVE minimum does not lift it.  The recording is taken from
    # the whole array, before the plan crops it to its proven support and
    # before the operator substitutes 1e-20 for an exact zero, and both
    # of those can only RAISE a minimum.  Clamping the recorded minimum
    # at zero therefore leaves a bound that holds for every operand the
    # plan can build from the array, and a dry source is not handed a
    # tighter bound than the recording can support: the envelope stays
    # -9/32 of the maximum.
    assert (parabolic_undershoot_floor(np.array([0.004, SOURCE_SPFH_MAX]))
            == parabolic_undershoot_floor(np.array([0.0, SOURCE_SPFH_MAX])))

    # A source that is not finite has no envelope.  The mapped field
    # carries the non-finite values into the one check that every
    # initialization passes through, and that check names them.
    assert parabolic_undershoot_floor(np.array([np.nan, 0.01])) is None
    with pytest.raises(ValueError, match="non_finite=1"):
        _specific_humidity_to_mixing_ratio(
            np.array([np.nan, 0.01]), allow_wps_undershoot=True,
            undershoot_floor=None)


def test_a_snapshot_refuses_an_envelope_that_is_not_one():
    for bad in (1.0e-9, float("nan")):
        with pytest.raises(ValueError, match="non-positive"):
            HorizontalSnapshot(
                valid_time=datetime.datetime(2025, 8, 15),
                levels_hpa=np.array([1000.0]),
                fields={}, specific_humidity_undershoot_floor=bad)


def test_the_direct_qv_lane_admits_what_that_operator_made():
    """The conversion, on both lanes, against the recorded envelope."""
    floor = parabolic_undershoot_floor(_worst_case_patch(SOURCE_SPFH_MAX))
    mapped = np.array([MAPPED_SPFH_MIN, 0.0, 1.0e-3, MAPPED_SPFH_MAX])
    admitted = _specific_humidity_to_mixing_ratio(
        mapped, allow_wps_undershoot=True, undershoot_floor=floor)
    np.testing.assert_allclose(
        admitted, mapped / (1.0 - mapped), rtol=0.0, atol=0.0)
    # Past the envelope the refusal stands, and names its evidence.
    with pytest.raises(ValueError, match=(
            r"specific humidity must be finite in \[-0\.00658837, 1\) \| "
            r"observed: outside_cells=1 of 1 in the field, of which "
            r"non_finite=0, below=1, at_or_above_one=0")):
        _specific_humidity_to_mixing_ratio(
            np.array([floor * 1.000001]), allow_wps_undershoot=True,
            undershoot_floor=floor)
    # A value WPS's 0..0.1 SPECHUMD gate would have admitted, but this
    # source's operator could not have made, is refused: the envelope is
    # the source's own.
    with pytest.raises(ValueError, match="outside_cells=1"):
        _specific_humidity_to_mixing_ratio(
            np.array([real_module._WPS_SPFH_UNDERSHOOT_LOWER_BOUND]),
            allow_wps_undershoot=True, undershoot_floor=floor)
    # Nonsense is still nonsense on either lane.
    for bad in (np.nan, np.inf, 1.0, 2.0):
        with pytest.raises(ValueError, match="specific humidity must be"):
            _specific_humidity_to_mixing_ratio(
                np.array([bad]), allow_wps_undershoot=True,
                undershoot_floor=floor)


def test_a_snapshot_with_no_recorded_envelope_keeps_wpss_own_gate():
    """met_em: WPS mapped it from a source this process never saw."""
    bound = real_module._WPS_SPFH_UNDERSHOOT_LOWER_BOUND
    assert bound == pytest.approx(
        -WPS_PARABOLIC_NEGATIVE_WEIGHT * 0.1, abs=2.0e-6)
    admitted = _specific_humidity_to_mixing_ratio(
        np.array([bound]), allow_wps_undershoot=True, undershoot_floor=None)
    assert admitted[0] == bound / (1.0 - bound)
    with pytest.raises(ValueError, match=r"finite in \[-0\.028126, 1\)"):
        _specific_humidity_to_mixing_ratio(
            np.array([-0.028127]), allow_wps_undershoot=True,
            undershoot_floor=None)


def test_the_control_arm_meets_no_undershoot_and_so_cannot_move():
    """The 8 GiB half of the measurement, in an assertion.

    Its mapped minimum is POSITIVE, which is why the rule that was
    changed cannot touch it: the pre-fix rule (refuse any negative on the
    direct-qv lane) and the shipped one (admit down to the operator's
    envelope) convert that field to the same values, and the 8 GiB
    control arm is bit-identical pre and post fix on every variable of
    every frame.
    """
    assert MAPPED_8GIB_SPFH_MIN > 0.0 > MAPPED_SPFH_MIN
    floor = parabolic_undershoot_floor(_worst_case_patch(SOURCE_SPFH_MAX))
    mapped = np.array([MAPPED_8GIB_SPFH_MIN, 1.0e-3, SOURCE_SPFH_MAX])
    refusing = _specific_humidity_to_mixing_ratio(
        mapped, allow_wps_undershoot=False)
    admitting = _specific_humidity_to_mixing_ratio(
        mapped, allow_wps_undershoot=True, undershoot_floor=floor)
    np.testing.assert_array_equal(refusing, admitting)


def _column_snapshot(spfh_minimum: float, *, floor: float | None):
    """A direct-qv snapshot whose mapped SPFH carries one undershoot cell."""
    levels_hpa = np.linspace(1013.0, 20.0, NZ, dtype=np.float64)
    pressure = np.broadcast_to(
        levels_hpa[:, None, None] * 100.0, (NZ, NY, NX)).copy()
    temperature = np.broadcast_to(
        np.linspace(298.0, 208.0, NZ)[:, None, None], (NZ, NY, NX)).copy()
    height = np.broadcast_to(
        np.linspace(100.0, 16_000.0, NZ)[:, None, None], (NZ, NY, NX)).copy()
    spfh = np.broadcast_to(
        (SOURCE_SPFH_MAX * np.linspace(1.0, 1.0e-4, NZ))[:, None, None],
        (NZ, NY, NX)).copy()
    spfh[NZ // 2, 1, 2] = spfh_minimum

    def f32(value):
        return np.ascontiguousarray(value, dtype=np.float32)

    fields = {
        "TT": f32(temperature), "PRES": f32(pressure), "SPFH": f32(spfh),
        "GHT": f32(height),
        "UU": f32(np.full((NZ, NY, NX + 1), 4.0)),
        "VV": f32(np.full((NZ, NY + 1, NX), -2.0)),
        "PSFC": f32(np.full((NY, NX), 101_000.0)),
        "T2": f32(np.full((NY, NX), 299.0)),
        "Q2": f32(np.full((NY, NX), 0.018)),
        "U10": f32(np.full((NY, NX + 1), 3.0)),
        "V10": f32(np.full((NY + 1, NX), -1.0)),
    }
    return HorizontalSnapshot(
        valid_time=datetime.datetime(2025, 8, 15, 0, 0),
        levels_hpa=levels_hpa, fields=fields,
        specific_humidity_authority=True, analyzed_species=(),
        specific_humidity_undershoot_floor=floor)


def _initialize(snapshot):
    from test_real_init_dry_slot import _ReferencePreprocessBackend

    cfg = RunConfig(
        nx=NX, ny=NY, nz=NZ, dx=12_000.0, dy=12_000.0, ztop=20_000.0,
        dt=60.0, run_seconds=120.0, hybrid_opt=2, etac=0.2,
        hypsometric_opt=2, moist=True, terrain_opt=1, mp_physics=10)
    coord = make_vertical_coord(NZ, stretch=1.6, hybrid_opt=2, etac=0.2)
    terrain = np.zeros((NY, NX), dtype=np.float64)
    return initialize_real(
        snapshot, cfg, coord, terrain, source_orography=terrain,
        p_top=P_TOP, sfcp_to_sfcp=True, column_workers=1,
        preprocess_backend=_ReferencePreprocessBackend(),
        state_backend="preprocess")


def test_the_lane_that_refused_the_window_now_initializes_it():
    """The defect at the size of one cell: the same value, both outcomes."""
    floor = parabolic_undershoot_floor(_worst_case_patch(SOURCE_SPFH_MAX))
    result = _initialize(_column_snapshot(MAPPED_SPFH_MIN, floor=floor))
    qv = np.asarray(result.state.qv, dtype=np.float64)
    assert np.isfinite(qv).all()
    assert float(qv.min()) >= real_module._WRF_QV_MIN_VALUE
    # The same snapshot whose mapped field that operator could not have
    # made is still refused, so this is an envelope and not an amnesty.
    with pytest.raises(ValueError, match="specific humidity must be finite"):
        _initialize(_column_snapshot(floor * 1.01, floor=floor))


@pytest.fixture
def version_identity_bound(monkeypatch):
    """Stand down ONE unrelated, pre-existing refusal for the sizing door.

    The fit loop loads every candidate through ``build_experiment``,
    which calls :func:`gpuwm.provenance_gate.require_version_identity`.
    That refuses whenever the running tree's ``pyproject.toml`` version
    and the version the installed distribution reports disagree, which is
    every editable install made before a version bump.  This branch opens
    2.7.6, so a batch run across the bump commit refused here and the
    failure landed on this file.  It is a real refusal doing its real job,
    it is not about sizing, and the remedy it names (reinstall) is the
    operator's, not this test's.  Same idiom, same wording, as
    ``tests/test_p3_front_door.py``, ``tests/test_chain_deep_ladder.py``
    and ``tests/test_shipped_configs_inflow_applicability.py`` already
    use, and self-retiring: once the tree under test is bound to its own
    metadata the early return fires and nothing is patched.
    """
    import gpuwm.provenance_gate as gate

    if gate.version_identity_refusal() is None:
        return                                  # already bound; no patch
    monkeypatch.setattr(gate, "version_identity_refusal",
                        lambda prov=None: None)


def test_the_16_gib_sizing_prices_at_or_above_what_its_run_measured(
        version_identity_bound):
    """The sizer's price and the initializer's peak, pinned to each other.

    The refusal this file is named for was not a memory refusal, and this
    is the measurement that says so: the plan the 16 GiB sizing emits is
    priced above what a run of it takes on the card it was sized for.
    """
    from gpuwm import domain_wizard as dw
    from gpuwm.core.preflight import GIB, estimate_experiment

    projection = {
        "map_proj": "lambert", "ref_lat": 27.5, "ref_lon": -82.5,
        "truelat1": 17.5, "truelat2": 37.5, "stand_lon": -82.5,
    }
    free_gib = dw.card_assumed_free_gib(16.0)
    dims, exp = dw.fit_ladder(
        ladder="12-3", free_bytes=int(free_gib * GIB), vram_gib=16.0,
        hours=6, start_time=datetime.datetime(2025, 8, 15, 0, 0),
        projection=projection, source="era5", name="arco-sizing")
    assert [tuple(int(v) for v in pair) for pair in dims] == [
        tuple(pair) for pair in SIZED_16GIB_LAYOUTS]
    # The other half of the pair that produced the defect: the same
    # centre, the same ladder, one card class down, and a window small
    # enough that it never meets a cell where the operator undershoots.
    small_free_gib = dw.card_assumed_free_gib(8.0)
    small_dims, _ = dw.fit_ladder(
        ladder="12-3", free_bytes=int(small_free_gib * GIB), vram_gib=8.0,
        hours=6, start_time=datetime.datetime(2025, 8, 15, 0, 0),
        projection=projection, source="era5", name="arco-sizing")
    assert [tuple(int(v) for v in pair) for pair in small_dims] == [
        tuple(pair) for pair in SIZED_8GIB_LAYOUTS]
    # The price is read on the plan that was RUN, so a sizing that now
    # emits a different layout still has to price the measured one at or
    # above what it took.
    measured_plan = dataclasses.replace(exp, domains=tuple(
        dataclasses.replace(dc, run=dataclasses.replace(dc.run, nx=nx, ny=ny))
        for dc, (nx, ny) in zip(exp.domains, MEASURED_16GIB_LAYOUTS)))
    estimate = estimate_experiment(
        measured_plan, vram_gib=16.0,
        forcing_interval_seconds=dw.source_forcing_interval_seconds("era5"))
    priced = int(estimate.peak_envelope_bytes)
    assert priced >= MEASURED_PEAK_BYTES, (
        f"the sizer prices {priced / GIB:.2f} GiB for a plan that measured "
        f"{MEASURED_PEAK_BYTES / GIB:.2f} GiB on the card it was sized for")
    assert priced <= int(free_gib * GIB)
