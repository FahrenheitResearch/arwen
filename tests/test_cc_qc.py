"""Correlation-coefficient QC: the per-moment mask, and what it must not eat.

The disaster case for this project is the mask working exactly as
configured and deleting the storm: hail cores run RhoHV 0.85-0.95, the
melting layer 0.90-0.97, and the tornadic debris signature falls below
0.8 at the mesocyclone couplet itself.  So the reflectivity survival
tests here are not smoke tests, they are the specification: high
reflectivity is never CC-dropped from the *echo*, whatever its RhoHV.
Each survival case ships beside its perturbed control -- the same RhoHV
with the reflectivity lowered, which must die -- so a test cannot pass
by the mask simply doing nothing.

Velocity is the other half of the specification and it runs the other
way.  The 2026-08-05 owner ruling is purity-first: a low-RhoHV gate
loses its velocity at every reflectivity, debris core included, because
debris is centrifuged and does not track the air.  These tests pin that
asymmetry moment by moment -- the same gate, kept in one field and
dropped in the other -- so that a future edit cannot quietly restore
the shield to velocity or quietly withdraw it from reflectivity.

The absence tests encode the other contract: a volume, sweep or gate
with no RhoHV passes through byte-identical to the pre-dual-pol
behaviour, and the absence is counted, never converted into a verdict.

The split-cut tests encode what a real 2026 KDMX volume proved: at
split-cut tilts the dual-pol moments ride the surveillance (CS) half
only, so the Doppler (CD) half -- the only velocity there is at the
lowest tilts -- must borrow its neighbour's RhoHV or go unmasked.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpuwm.obs.cc_qc import (CcQcParams, CcQcParamsError,
                             DROP_LOW_RHO, DROP_LOW_RHO_BELOW_SHIELD,
                             DROP_RHO_FLOOR, REASON_COLOCATED,
                             REASON_COMPANION, REASON_NO_RHO)
from gpuwm.obs.sweeps import Moment, RadarSite, RadarVolume, Sweep
from gpuwm.obs.superob import (SuperobParams, SuperobParamsError,
                               merge_contributions, superob_volume)
from gpuwm.obs.target_grid import TargetGrid
from gpuwm.static.lambert import LambertGrid

GATE_SIZE = 250.0
FIRST_GATE = 2125.0


def _grid(nx: int = 41, ny: int = 41, dx: float = 2000.0, nz: int = 10,
          top_m: float = 10000.0) -> TargetGrid:
    projection = LambertGrid(
        ref_lat=35.3331, ref_lon=-97.2778, truelat1=33.0, truelat2=37.0,
        stand_lon=-97.2778, dx=dx, dy=dx, e_we=nx + 1, e_sn=ny + 1)
    return TargetGrid.from_projection(
        projection, z_w=np.linspace(0.0, top_m, nz + 1), name="analytic")


def _plane(values, radials: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.tile(values[None, :], (radials, 1))


def _moment(product: str, values, *, radials: int,
            first_gate: float = FIRST_GATE,
            gate_size: float = GATE_SIZE) -> Moment:
    data = _plane(values, radials)
    unit = {"REF": "dBZ", "VEL": "m/s", "RHO": ""}[product]
    return Moment(product, unit, data.shape[1], first_gate, gate_size, data)


def _sweep(index: int, moments: dict[str, Moment], *,
           elevation: float = 0.5, azimuths=(88.0, 90.0, 92.0),
           nyquist: float | None = 32.0) -> Sweep:
    azimuth = np.asarray(azimuths, dtype=np.float32)
    return Sweep(
        sweep_index=index, elevation_number=index + 1,
        elevation_angle_deg=elevation, nyquist_velocity_ms=nyquist,
        start_status=3, end_status=2, cut_sector=0, complete=True,
        azimuth_deg=azimuth,
        elevation_deg=np.full(azimuth.size, elevation, dtype=np.float32),
        moments=moments)


def _volume(grid: TargetGrid, sweeps) -> RadarVolume:
    centre_j, centre_i = grid.ny // 2, grid.nx // 2
    return RadarVolume(
        site=RadarSite(id="KTLX", name="synthetic",
                       lat_deg=float(grid.lat[centre_j, centre_i]),
                       lon_deg=float(grid.lon[centre_j, centre_i]),
                       alt_m=0.0, source="test"),
        valid_time="2026-07-28T20:03:16Z", station_id="KTLX",
        volume_file="KTLX20260728_200316_V06",
        volume_sha256="0" * 64, volume_bytes=8102058,
        pack_path=__import__("pathlib").Path("synthetic.rdrpack"),
        pack_sha256="1" * 64,
        params={"moments": ["REF", "VEL", "RHO"], "max_range_km": 250.0},
        framing={"magic": "AR2V0006", "block_count": 1},
        sweeps=tuple(sweeps))


def _assert_identical(one, other) -> None:
    """Two contributions agree to the byte, counters included."""

    for name in ("z_linear_sum", "z_count", "z_max_dbz", "z_sumsq_dbz",
                 "z_sum_dbz", "vr_sum", "vr_sumsq", "vr_count", "vr_min",
                 "vr_max", "beam_east", "beam_north", "beam_up",
                 "nyquist_min", "vr_rejected"):
        left = getattr(one, name)
        right = getattr(other, name)
        assert left.dtype == right.dtype, name
        assert np.array_equal(left, right, equal_nan=True), name
    assert one.counts.to_payload() == other.counts.to_payload()


# ---------------------------------------------------------------------------
# The hazard that matters most: severe weather must survive the mask -- as
# echo.  Its velocity is a separate question with the opposite answer.
# ---------------------------------------------------------------------------

def test_hail_core_keeps_its_echo_and_loses_its_velocity():
    """High-Z low-CC (hail, 0.90) keeps Z; every low-CC gate loses V.

    Gate values are the cited signatures: hail cores run RhoHV 0.85-0.95
    (Balakrishnan & Zrnic 1990; Kumjian 2013) at reflectivities far above
    any biota, while bird layers run RhoHV well below 0.9 at low
    reflectivity (Zrnic & Ryzhkov 1998).

    The hail core is where the 2026-08-05 ruling costs something and the
    cost is pinned here rather than argued about later: its RhoHV of
    0.90 is below ``rho_min``, so under purity-first its velocity dies
    with the birds' even though its 55 dBZ echo is kept.  Tumbling,
    fast-falling, wet-coated hail is not a tracer of the air any more
    than debris is.  ``rho_min_velocity`` is the knob that buys that
    velocity back, and the second half of this test proves it works
    without touching the biota verdict.
    """

    grid = _grid()
    radials = 3
    #                 hail   rain   bird  bird
    ref = [55.0, 30.0, 10.0, 12.0]
    vel = [20.0, 10.0, 8.0, 6.0]
    rho = [0.90, 0.99, 0.75, 0.60]
    sweeps = lambda: [_sweep(0, {                                # noqa: E731
        "REF": _moment("REF", ref, radials=radials),
        "VEL": _moment("VEL", vel, radials=radials),
        "RHO": _moment("RHO", rho, radials=radials),
    })]
    on = superob_volume(_volume(grid, sweeps()), grid,
                        params=SuperobParams(cc_qc=CcQcParams()))
    off = superob_volume(_volume(grid, sweeps()), grid,
                         params=SuperobParams())

    # Reflectivity: the two bird gates died, the hail core and the rain
    # did not.  Velocity: the hail core died with them -- three gates.
    assert on.counts.cc_reflectivity_gates_rejected == 2 * radials
    assert on.counts.cc_velocity_gates_rejected == 3 * radials
    # And exactly the hail core's velocity is the shield's lost ground.
    assert on.counts.cc_velocity_gates_rejected_shielded_z == radials
    assert on.counts.cc_sweeps_masked == 1
    assert int(on.z_count.sum()) == int(off.z_count.sum()) - 2 * radials
    assert int(on.vr_count.sum()) == int(off.vr_count.sum()) - 3 * radials
    # The retained maxima still contain the hail core: no high-Z echo died.
    assert float(on.z_max_dbz.max()) == 55.0
    # And the bird reflectivity is gone from the grid entirely.
    retained = on.z_max_dbz[np.isfinite(on.z_max_dbz)]
    assert not np.any(np.isin(retained, [10.0, 12.0]))
    # CC-dropped gates read as holes downstream, by design.
    expected_holes = (2 + 3) * radials   # two REF gates, three VEL gates
    assert (on.counts.gates_nonfinite
            == off.counts.gates_nonfinite + expected_holes)
    # The QC ran before the shear scan: dropped gates break the pair chain.
    assert (on.counts.velocity_gate_pairs_tested
            < off.counts.velocity_gate_pairs_tested)
    # Provenance says why, per moment, disjointly.
    [record] = on.cc_qc["sweeps"]
    assert record["gates_dropped_reason"] == {
        "REF": {DROP_LOW_RHO_BELOW_SHIELD: 2 * radials},
        "VEL": {DROP_LOW_RHO: 3 * radials},
    }
    assert record["gates_dropped_shielded_z"] == {"REF": 0, "VEL": radials}

    # The knob that revisits the ruling: at a 0.80 velocity threshold --
    # the TDS criterion of Ryzhkov et al. (2005) and Bodine et al. (2013)
    # -- the hail core keeps its velocity and the birds still lose theirs.
    lenient = superob_volume(
        _volume(grid, sweeps()), grid,
        params=SuperobParams(cc_qc=CcQcParams(rho_min_velocity=0.80)))
    assert lenient.counts.cc_velocity_gates_rejected == 2 * radials
    assert lenient.counts.cc_velocity_gates_rejected_shielded_z == 0
    assert lenient.counts.cc_reflectivity_gates_rejected == 2 * radials


def test_the_shield_decides_reflectivity_and_only_the_correlation_decides_v():
    """Perturbed control: same RhoHV 0.75, only reflectivity differs.

    In reflectivity the shield decides, so the 55 dBZ gate lives and the
    10 dBZ gate dies -- if the mask were bare CC, both would die and this
    test would catch the hail/TDS deletion the compound rule exists to
    prevent (Tang et al. 2014 protect high-Z echo in MRMS for the same
    reason).  In velocity nothing shields, so the identical pair of
    RhoHV values produces the identical verdict at both reflectivities.
    """

    grid = _grid()
    ref = [55.0, 10.0]      # identical CC below; only Z differs
    rho = [0.75, 0.75]
    vel = [20.0, 8.0]
    sweeps = [_sweep(0, {
        "REF": _moment("REF", ref, radials=3),
        "VEL": _moment("VEL", vel, radials=3),
        "RHO": _moment("RHO", rho, radials=3),
    })]
    on = superob_volume(_volume(grid, sweeps), grid,
                        params=SuperobParams(cc_qc=CcQcParams()))
    assert on.counts.cc_reflectivity_gates_rejected == 3   # low-Z gate only
    assert float(on.z_max_dbz.max()) == 55.0
    assert on.counts.cc_velocity_gates_rejected == 6       # both gates
    assert int(on.vr_count.sum()) == 0


def test_tornadic_debris_keeps_its_echo_and_loses_velocity_with_no_couplet():
    """TDS at RhoHV 0.65 (Ryzhkov et al. 2005; Bodine et al. 2013).

    Three debris gates straddling the shield: 34.5 dBZ just below it,
    35.0 dBZ exactly on it, 45 dBZ well above.  The reflectivity verdict
    follows the shield and its boundary is inclusive -- ``z >=
    ref_shield_dbz`` survives -- so 35.0 lives and 34.5 does not.  The
    velocity verdict ignores all of that: every one of the three dies,
    including the 45 dBZ core.

    That last clause is the ruling, not an accident.  Debris is lofted,
    non-Rayleigh and centrifuged outward by the vortex, so its radial
    velocity is biased away from the air motion exactly where a
    tornadic wind analysis would lean on it hardest.  A missing
    observation beats a confidently wrong one.

    The velocity field here is uniform -- 20 m/s everywhere, no couplet
    anywhere -- which is why the 2026-08-12 fringe exemption does not
    fire on the 34.5 dBZ gate.  Low RhoHV in the debris band is not by
    itself evidence of debris; the rotation is the other half of the
    conjunction, and this test is the control that proves it binds.
    """

    grid = _grid()
    sweeps = lambda: [_sweep(0, {                                # noqa: E731
        "REF": _moment("REF", [34.5, 35.0, 45.0], radials=3),
        "VEL": _moment("VEL", [20.0, 20.0, 20.0], radials=3),
        "RHO": _moment("RHO", [0.65, 0.65, 0.65], radials=3),
    })]
    on = superob_volume(_volume(grid, sweeps()), grid,
                        params=SuperobParams(cc_qc=CcQcParams()))
    off = superob_volume(_volume(grid, sweeps()), grid,
                         params=SuperobParams())

    # Reflectivity: only the sub-shield fringe gate dies.
    assert on.counts.cc_reflectivity_gates_rejected == 3
    assert int(on.z_count.sum()) == int(off.z_count.sum()) - 3
    assert float(on.z_max_dbz.max()) == 45.0
    # Velocity: all three, at every reflectivity, shield or no shield.
    assert on.counts.cc_velocity_gates_rejected == 9
    assert int(on.vr_count.sum()) == 0
    # Six of those nine were protected echo -- the price, counted.
    assert on.counts.cc_velocity_gates_rejected_shielded_z == 6
    [record] = on.cc_qc["sweeps"]
    assert record["gates_dropped_reason"] == {
        "REF": {DROP_LOW_RHO_BELOW_SHIELD: 3},
        "VEL": {DROP_LOW_RHO: 9},
    }
    # The exemption was live and kept nothing, because there is no
    # couplet: three fringe gates asked and the rotation criterion
    # turned all three away.
    assert on.counts.cc_velocity_gates_exempt_tds_fringe == 0
    assert on.counts.cc_couplet_seed_gates == 0
    assert on.counts.cc_velocity_tds_no_couplet_nearby == 3
    assert on.counts.cc_velocity_tds_at_or_above_shield == 6


def test_a_floor_would_eat_the_debris_echo_which_is_why_it_defaults_off():
    """``rho_floor`` deletes the debris signature from the reflectivity field.

    The ruling keeps debris as echo -- it is real scatter, and the TDS
    is a diagnostic this project wants.  An unconditional floor above
    0.65 deletes it anyway, shield or no shield, which is WHY the floor
    defaults to None: this test is the record of that decision.  It also
    pins the floor's second property under the per-moment policy -- it
    is now a reflectivity knob, because velocity already dropped every
    one of those gates on the correlation alone.
    """

    grid = _grid()
    sweeps = lambda: [_sweep(0, {                                # noqa: E731
        "REF": _moment("REF", [34.5, 35.0, 45.0], radials=3),
        "VEL": _moment("VEL", [20.0, 20.0, 20.0], radials=3),
        "RHO": _moment("RHO", [0.65, 0.65, 0.65], radials=3),
    })]
    floored = superob_volume(
        _volume(grid, sweeps()), grid,
        params=SuperobParams(cc_qc=CcQcParams(rho_floor=0.7)))
    assert floored.counts.cc_reflectivity_gates_rejected == 9
    assert not np.any(np.isfinite(floored.z_max_dbz))     # the storm is gone
    assert floored.counts.cc_velocity_gates_rejected == 9
    [record] = floored.cc_qc["sweeps"]
    assert record["gates_dropped_reason"] == {
        "REF": {DROP_LOW_RHO_BELOW_SHIELD: 3, DROP_RHO_FLOOR: 6},
        # Nothing left for the floor to do: the velocity rule got there
        # first, on every gate.
        "VEL": {DROP_LOW_RHO: 9, DROP_RHO_FLOOR: 0},
    }


def test_bright_band_survives_via_the_shield_and_the_weak_edge_is_the_cost():
    """Melting layer: RhoHV dips to 0.90-0.97 (Giangrande et al. 2008).

    The bright-band peak carries reflectivity above the shield and
    survives as echo.  The stratiform edge below the shield does not,
    and this test records that honestly rather than hiding it: the
    residual risk lives at 25-30 dBZ bright-band edges, which is the
    argument for tuning ``ref_shield_dbz`` down, not for trusting CC
    alone.  Both gates lose their velocity -- melting, mixed-phase
    hydrometeors in the bright band are poor tracers, and the ruling
    does not carve out an exception for them either.
    """

    grid = _grid()
    sweeps = [_sweep(0, {
        "REF": _moment("REF", [42.0, 28.0], radials=3),
        "VEL": _moment("VEL", [12.0, 11.0], radials=3),
        "RHO": _moment("RHO", [0.92, 0.92], radials=3),
    })]
    on = superob_volume(_volume(grid, sweeps), grid,
                        params=SuperobParams(cc_qc=CcQcParams()))
    assert float(on.z_max_dbz.max()) == 42.0                  # peak lives
    assert on.counts.cc_reflectivity_gates_rejected == 3      # edge dies
    assert on.counts.cc_velocity_gates_rejected == 6          # both, in V
    assert on.counts.cc_velocity_gates_rejected_shielded_z == 3


# ---------------------------------------------------------------------------
# Absence never fabricates a verdict.
# ---------------------------------------------------------------------------

def test_off_is_the_identity_even_when_rho_planes_are_present():
    grid = _grid()
    with_rho = [_sweep(0, {
        "REF": _moment("REF", [55.0, 10.0], radials=3),
        "VEL": _moment("VEL", [20.0, 8.0], radials=3),
        "RHO": _moment("RHO", [0.90, 0.75], radials=3),
    })]
    without = [_sweep(0, {
        "REF": _moment("REF", [55.0, 10.0], radials=3),
        "VEL": _moment("VEL", [20.0, 8.0], radials=3),
    })]
    params = SuperobParams()
    assert "cc_qc" not in params.to_payload()
    one = superob_volume(_volume(grid, with_rho), grid, params=params)
    other = superob_volume(_volume(grid, without), grid, params=params)
    _assert_identical(one, other)
    assert one.cc_qc == {} and other.cc_qc == {}


def test_a_volume_with_no_rho_changes_nothing_and_says_so():
    """Pre-2013 archive: QC on, no dual-pol -- identical output, counted."""

    grid = _grid()
    sweeps = lambda: [_sweep(0, {                                # noqa: E731
        "REF": _moment("REF", [55.0, 10.0], radials=3),
        "VEL": _moment("VEL", [20.0, 8.0], radials=3),
    })]
    on = superob_volume(_volume(grid, sweeps()), grid,
                        params=SuperobParams(cc_qc=CcQcParams()))
    off = superob_volume(_volume(grid, sweeps()), grid,
                         params=SuperobParams())
    for name in ("z_linear_sum", "z_count", "z_max_dbz", "vr_sum",
                 "vr_count", "vr_rejected"):
        assert np.array_equal(getattr(on, name), getattr(off, name),
                              equal_nan=True), name
    assert on.counts.cc_sweeps_without_rho == 1
    assert on.counts.cc_sweeps_masked == 0
    assert on.counts.cc_velocity_gates_rejected == 0
    assert on.counts.cc_reflectivity_gates_rejected == 0
    [record] = on.cc_qc["sweeps"]
    assert record["applied"] is False
    assert record["reason"] == REASON_NO_RHO


def test_gate_level_absence_passes_open_and_matches_by_range_not_index():
    """RHO shorter than REF, offset by one gate, with one censored gate.

    REF has 20 gates; RHO starts one gate later and carries 10.  So REF
    gate 0 has no RhoHV (before RHO's first gate), REF gates 1..10 map to
    RHO gates 0..9 BY SLANT RANGE (an index-matched mask would be off by
    one), and REF gates 11..19 are beyond RHO's extent.  All reflectivity
    is low-Z and all present RhoHV is 0.70, except RHO gate 5 which is
    censored (NaN).  Exactly the range-mapped, finite-RhoHV gates die:
    the off-by-one is caught because gate 11 (beyond extent) must live
    while gate 10 (mapped to RHO gate 9) must die.
    """

    grid = _grid()
    ref_values = np.full(20, 10.0)
    rho_values = np.full(10, 0.70)
    rho_values[5] = np.nan
    sweeps = [_sweep(0, {
        "REF": _moment("REF", ref_values, radials=1),
        "RHO": _moment("RHO", rho_values, radials=1,
                       first_gate=FIRST_GATE + GATE_SIZE),
    }, azimuths=(90.0,))]
    on = superob_volume(_volume(grid, sweeps), grid,
                        params=SuperobParams(cc_qc=CcQcParams()))
    off = superob_volume(_volume(grid, sweeps), grid,
                         params=SuperobParams())
    # 9 finite RhoHV gates die; gate 0, the censored gate's target, and
    # the 9 beyond-extent gates all pass open -- 11 missing in total.
    assert on.counts.cc_reflectivity_gates_rejected == 9
    assert on.counts.cc_gates_rho_missing == 11
    assert int(on.z_count.sum()) == int(off.z_count.sum()) - 9


# ---------------------------------------------------------------------------
# Split cuts: the measured 2026 layout (CS carries RHO, CD carries VEL).
# ---------------------------------------------------------------------------

def test_split_cut_velocity_borrows_the_surveillance_rhohv():
    """CD half (REF+VEL, no RHO) is masked via the CS half's RHO plane.

    The CS azimuth grid is offset a quarter degree, the way real
    adjacent cuts are, so this only passes if the pairing matches
    radials by nearest azimuth rather than by row index.
    """

    grid = _grid()
    az_cd = (88.0, 90.0, 92.0)
    az_cs = (88.25, 90.25, 92.25)
    cs = _sweep(0, {
        "REF": _moment("REF", [10.0, 30.0], radials=3),
        "RHO": _moment("RHO", [0.75, 0.99], radials=3),
    }, elevation=0.27, azimuths=az_cs, nyquist=None)
    cd = _sweep(1, {
        "REF": _moment("REF", [10.0, 30.0], radials=3),
        "VEL": _moment("VEL", [8.0, 15.0], radials=3),
    }, elevation=0.53, azimuths=az_cd)
    on = superob_volume(_volume(grid, [cs, cd]), grid,
                        params=SuperobParams(cc_qc=CcQcParams()))
    assert on.counts.cc_sweeps_masked == 2
    assert on.counts.cc_sweeps_paired_companion == 1
    # The bird gate died on BOTH halves: CS reflectivity co-located, CD
    # reflectivity and velocity through the companion.
    assert on.counts.cc_reflectivity_gates_rejected == 2 * 3
    assert on.counts.cc_velocity_gates_rejected == 3
    records = {r["sweep_index"]: r for r in on.cc_qc["sweeps"]}
    assert records[0]["reason"] == REASON_COLOCATED
    assert records[1]["reason"] == REASON_COMPANION
    assert records[1]["rho_companion_sweep_index"] == 0


def test_a_different_tilt_is_never_mistaken_for_a_companion():
    """Adjacent sweep at 1.8 deg cannot lend RhoHV to a 0.5 deg cut."""

    grid = _grid()
    high = _sweep(0, {
        "REF": _moment("REF", [10.0], radials=3),
        "RHO": _moment("RHO", [0.70], radials=3),
    }, elevation=1.8, nyquist=None)
    low = _sweep(1, {
        "REF": _moment("REF", [10.0], radials=3),
        "VEL": _moment("VEL", [8.0], radials=3),
    }, elevation=0.5)
    on = superob_volume(_volume(grid, [high, low]), grid,
                        params=SuperobParams(cc_qc=CcQcParams()))
    assert on.counts.cc_sweeps_paired_companion == 0
    assert on.counts.cc_sweeps_without_rho == 1
    assert on.counts.cc_velocity_gates_rejected == 0
    records = {r["sweep_index"]: r for r in on.cc_qc["sweeps"]}
    assert records[1]["reason"] == REASON_NO_RHO


def test_pairing_can_be_declined():
    grid = _grid()
    cs = _sweep(0, {
        "REF": _moment("REF", [10.0], radials=3),
        "RHO": _moment("RHO", [0.70], radials=3),
    }, elevation=0.27, nyquist=None)
    cd = _sweep(1, {
        "REF": _moment("REF", [10.0], radials=3),
        "VEL": _moment("VEL", [8.0], radials=3),
    }, elevation=0.53)
    on = superob_volume(
        _volume(grid, [cs, cd]), grid,
        params=SuperobParams(
            cc_qc=CcQcParams(pair_companion_sweeps=False)))
    assert on.counts.cc_sweeps_paired_companion == 0
    assert on.counts.cc_sweeps_without_rho == 1
    assert on.counts.cc_velocity_gates_rejected == 0
    # The CS half still cleans its own reflectivity.
    assert on.counts.cc_reflectivity_gates_rejected == 3


# ---------------------------------------------------------------------------
# Parameters: refusal and payload identity.
# ---------------------------------------------------------------------------

def test_parameters_refuse_what_cannot_work():
    with pytest.raises(CcQcParamsError):
        CcQcParams(rho_min=0.0)
    with pytest.raises(CcQcParamsError):
        CcQcParams(rho_min=1.5)
    with pytest.raises(CcQcParamsError):
        CcQcParams(ref_shield_dbz=float("nan"))
    with pytest.raises(CcQcParamsError):
        CcQcParams(rho_floor=0.99)          # above rho_min
    with pytest.raises(CcQcParamsError):
        CcQcParams(rho_min_velocity=0.0)    # silently masks nothing
    with pytest.raises(CcQcParamsError):
        CcQcParams(rho_min_velocity=1.5)
    with pytest.raises(CcQcParamsError):
        CcQcParams(pair_companion_sweeps=1)  # a flag, not an int
    with pytest.raises(CcQcParamsError):
        CcQcParams(companion_elevation_tolerance_deg=0.0)
    with pytest.raises(CcQcParamsError):
        CcQcParams(companion_azimuth_tolerance_deg=-1.0)
    with pytest.raises(SuperobParamsError):
        SuperobParams(cc_qc="on")           # the mask needs thresholds


def test_payload_carries_cc_qc_only_when_configured():
    assert "cc_qc" not in SuperobParams().to_payload()
    payload = SuperobParams(cc_qc=CcQcParams()).to_payload()
    assert payload["cc_qc"]["rho_min"] == 0.95
    assert payload["cc_qc"]["ref_shield_dbz"] == 35.0
    assert payload["cc_qc"]["rho_floor"] is None
    assert payload["cc_qc"]["pair_companion_sweeps"] is True
    # None: velocity uses rho_min, which is the ruling's default.  The
    # key is present and null rather than absent, so a payload can never
    # be read as "the velocity policy was not recorded".
    assert payload["cc_qc"]["rho_min_velocity"] is None


def test_merge_carries_the_cc_account_through():
    grid = _grid()
    sweeps = [_sweep(0, {
        "REF": _moment("REF", [10.0], radials=3),
        "RHO": _moment("RHO", [0.70], radials=3),
    }, nyquist=None)]
    params = SuperobParams(cc_qc=CcQcParams())
    contribution = superob_volume(_volume(grid, sweeps), grid,
                                  params=params)
    observations = merge_contributions([contribution], grid, params=params)
    [account] = observations.cc_qc
    assert account["params"]["rho_min"] == 0.95
    assert account["sweeps"][0]["applied"] is True

    off = merge_contributions(
        [superob_volume(_volume(grid, sweeps), grid,
                        params=SuperobParams())],
        grid, params=SuperobParams())
    assert off.cc_qc == [{}]


# ---------------------------------------------------------------------------
# The debris-signature fringe (owner ruling 2026-08-12).  A weak or distant
# TDS paints 30-35 dBZ, below the reflectivity shield, and under the strict
# velocity rule that whole band lost the rotation a tornadic analysis needs.
# The exemption keeps it -- but only in conjunction, so each test below
# ships with the control that proves the conjunct it is about actually
# binds.  Low RhoHV alone must never be enough.
# ---------------------------------------------------------------------------

FRINGE_RADIALS = 8
FRINGE_AZIMUTHS = tuple(88.0 + 0.5 * index for index in range(FRINGE_RADIALS))
COUPLET_GATES = 6          # gates 0-5 carry the velocity couplet itself
FRINGE_GATES = (12, 18)    # gates 12-17: the debris fringe, inside the reach
DISTANT_GATES = (30, 36)   # gates 30-35: 25+ gates out, past the 12-gate reach
GATE_COUNT = 40
PATCH_GATES = FRINGE_GATES[1] - FRINGE_GATES[0]


def _moment2d(product: str, data) -> Moment:
    data = np.asarray(data, dtype=np.float32)
    unit = {"REF": "dBZ", "VEL": "m/s", "RHO": ""}[product]
    return Moment(product, unit, data.shape[1], FIRST_GATE, GATE_SIZE, data)


def _fringe_planes(*, near_dbz: float, near_rho: float,
                   far_dbz: float = 32.0, far_rho: float = 0.65,
                   couplet: bool = True) -> dict:
    """One sweep: a couplet, the fringe around it, a look-alike patch far off.

    The near patch is the debris fringe just beyond the couplet, inside
    its reach; the far patch carries the same reflectivity and the same
    RhoHV 23 gates further out, which is the in-sweep control for the
    rotation criterion -- identical dual-pol evidence, no rotation beside
    it.  The couplet's own gates are meteorological (RhoHV 0.99) so that
    what the exemption keeps is unambiguously the fringe.
    """

    shape = (FRINGE_RADIALS, GATE_COUNT)
    reflectivity = np.full(shape, 5.0, dtype=np.float32)
    rho = np.full(shape, 0.99, dtype=np.float32)
    velocity = np.full(shape, 4.0, dtype=np.float32)
    reflectivity[:, :COUPLET_GATES] = 45.0
    fringe = slice(*FRINGE_GATES)
    distant = slice(*DISTANT_GATES)
    reflectivity[:, fringe] = near_dbz
    rho[:, fringe] = near_rho
    velocity[:, fringe] = 8.0
    reflectivity[:, distant] = far_dbz
    rho[:, distant] = far_rho
    if couplet:
        # Inbound half meets outbound half between radials 3 and 4: one
        # adjacent pair, opposite signs, 40 m/s apart.
        velocity[:FRINGE_RADIALS // 2, :COUPLET_GATES] = -20.0
        velocity[FRINGE_RADIALS // 2:, :COUPLET_GATES] = 20.0
    return {
        "REF": _moment2d("REF", reflectivity),
        "VEL": _moment2d("VEL", velocity),
        "RHO": _moment2d("RHO", rho),
    }


def _fringe_volume(grid, **kwargs) -> RadarVolume:
    return _volume(grid, [_sweep(0, _fringe_planes(**kwargs),
                                 azimuths=FRINGE_AZIMUTHS)])


def test_the_debris_fringe_keeps_its_velocity_beside_a_couplet():
    """32 dBZ at RhoHV 0.65, one beam from a 40 m/s couplet: velocity kept.

    This is the seam task #58 opened.  Under the strict 2026-08-05 rule
    every one of these gates lost its velocity because RhoHV alone
    decided, and the fringe is exactly where a distant or weak tornado's
    signature lives, so the couplet went unobserved.  The control is the
    identical build with ``tds_fringe_exempt=False``: same volume, same
    thresholds, and the difference between the two arms is the ruling.
    """

    grid = _grid()
    kept = superob_volume(
        _fringe_volume(grid, near_dbz=32.0, near_rho=0.65), grid,
        params=SuperobParams(cc_qc=CcQcParams()))
    strict = superob_volume(
        _fringe_volume(grid, near_dbz=32.0, near_rho=0.65), grid,
        params=SuperobParams(cc_qc=CcQcParams(tds_fringe_exempt=False)))

    exempted = kept.counts.cc_velocity_gates_exempt_tds_fringe
    assert exempted == FRINGE_RADIALS * PATCH_GATES == 48
    assert kept.counts.cc_couplet_seed_gates > 0
    # Every exempted gate is a gate the strict arm deleted.
    assert (strict.counts.cc_velocity_gates_rejected
            - kept.counts.cc_velocity_gates_rejected) == exempted
    assert strict.counts.cc_velocity_gates_exempt_tds_fringe == 0
    # ...and the velocity reaches the grid rather than merely surviving
    # a counter.
    assert int(kept.vr_count.sum()) > int(strict.vr_count.sum())


def test_the_same_fringe_dies_where_no_couplet_is_beside_it():
    """The rotation conjunct, both directions, in one sweep and across two.

    In-sweep: the far patch carries the same 32 dBZ and the same RhoHV
    0.65 as the near patch but sits 29 gates from the couplet, past the
    12-gate reach, and it is dropped.  Across builds: with the couplet
    removed from the velocity plane entirely, nothing is exempted at all.
    """

    grid = _grid()
    kept = superob_volume(
        _fringe_volume(grid, near_dbz=32.0, near_rho=0.65), grid,
        params=SuperobParams(cc_qc=CcQcParams()))
    # The far patch is fringe-eligible on dual-pol evidence and still
    # dies: it is counted against the rotation criterion, by name.
    far_gates = FRINGE_RADIALS * PATCH_GATES
    assert kept.counts.cc_velocity_tds_no_couplet_nearby == far_gates == 48
    assert kept.counts.cc_velocity_gates_rejected == far_gates

    flat = superob_volume(
        _fringe_volume(grid, near_dbz=32.0, near_rho=0.65, couplet=False),
        grid, params=SuperobParams(cc_qc=CcQcParams()))
    assert flat.counts.cc_couplet_seed_gates == 0
    assert flat.counts.cc_velocity_gates_exempt_tds_fringe == 0
    assert flat.counts.cc_velocity_tds_no_couplet_nearby == 96


def test_noise_below_the_debris_floor_dies_beside_the_couplet():
    """RhoHV 0.30 is receiver noise, and rotation does not redeem it.

    The perturbed control is the exemption test above: the same gate at
    RhoHV 0.65 in the same place is kept, so this one cannot pass by the
    mask simply doing nothing.
    """

    grid = _grid()
    noise = superob_volume(
        _fringe_volume(grid, near_dbz=32.0, near_rho=0.30), grid,
        params=SuperobParams(cc_qc=CcQcParams()))
    assert noise.counts.cc_couplet_seed_gates > 0      # rotation was there
    assert noise.counts.cc_velocity_gates_exempt_tds_fringe == 0
    assert noise.counts.cc_velocity_tds_rho_below_floor == 48


def test_biota_below_the_debris_reflectivity_dies_beside_the_couplet():
    """20 dBZ at RhoHV 0.65 is a bird layer, couplet or no couplet.

    Roosting-bird echo peaks near 25-30 dBZ, which is why the band opens
    at 30 and not lower.  Same construction as the exempted case with
    the reflectivity dropped: the perturbed control that keeps this test
    honest.
    """

    grid = _grid()
    biota = superob_volume(
        _fringe_volume(grid, near_dbz=20.0, near_rho=0.65), grid,
        params=SuperobParams(cc_qc=CcQcParams()))
    assert biota.counts.cc_couplet_seed_gates > 0
    assert biota.counts.cc_velocity_gates_exempt_tds_fringe == 0
    assert biota.counts.cc_velocity_tds_below_reflectivity == 48


def test_the_debris_core_above_the_shield_still_loses_its_velocity():
    """45 dBZ at RhoHV 0.65 beside the couplet: the 2026-08-05 ruling holds.

    The exemption is a band, not a new shield.  At and above 35 dBZ the
    centrifuging argument is at its strongest and the surrounding
    meteorological gates are densest, so the core velocity is still
    deleted -- and counted against the ceiling criterion, so a reader can
    see the ruling applied rather than forgotten.
    """

    grid = _grid()
    core = superob_volume(
        _fringe_volume(grid, near_dbz=45.0, near_rho=0.65), grid,
        params=SuperobParams(cc_qc=CcQcParams()))
    assert core.counts.cc_couplet_seed_gates > 0
    assert core.counts.cc_velocity_gates_exempt_tds_fringe == 0
    assert core.counts.cc_velocity_tds_at_or_above_shield == 48
    # ...and its echo is still kept, which is the other half of 08-05.
    assert float(core.z_max_dbz.max()) == 45.0


def test_the_exemption_off_is_the_pre_ruling_build():
    """``tds_fringe_exempt=False`` restores the 2026-08-05 verdict.

    The switch has to be a true selector rather than a near-miss, or a
    measurement of what the ruling costs is a measurement of two
    different things.  The observation arrays match a pre-ruling build;
    the receipt gains the new counters reading zero, which is the
    difference between a rule that did nothing and a rule that was not
    there.
    """

    grid = _grid()
    strict = superob_volume(
        _fringe_volume(grid, near_dbz=32.0, near_rho=0.65), grid,
        params=SuperobParams(cc_qc=CcQcParams(tds_fringe_exempt=False)))
    assert strict.counts.cc_velocity_gates_exempt_tds_fringe == 0
    assert strict.counts.cc_couplet_seed_gates == 0
    assert strict.counts.cc_velocity_gates_rejected == 96
    [record] = strict.cc_qc["sweeps"]
    assert record["gates_dropped_reason"]["VEL"] == {DROP_LOW_RHO: 96}
    assert record["gates_exempt_tds_fringe"] == {}


def test_the_four_refusals_and_the_exemption_account_for_every_gate():
    """The criteria are disjoint and complete -- the audit property.

    Every velocity gate the strict rule would drop is either exempted or
    turned away by exactly one criterion, so a reader can reconstruct
    the strict-rule total from the receipt without rerunning anything.
    """

    grid = _grid()
    run = superob_volume(
        _fringe_volume(grid, near_dbz=32.0, near_rho=0.65), grid,
        params=SuperobParams(cc_qc=CcQcParams()))
    counts = run.counts
    turned_away = (counts.cc_velocity_tds_rho_below_floor
                   + counts.cc_velocity_tds_below_reflectivity
                   + counts.cc_velocity_tds_at_or_above_shield
                   + counts.cc_velocity_tds_no_couplet_nearby)
    assert turned_away == counts.cc_velocity_gates_rejected
    strict = superob_volume(
        _fringe_volume(grid, near_dbz=32.0, near_rho=0.65), grid,
        params=SuperobParams(cc_qc=CcQcParams(tds_fringe_exempt=False)))
    assert (turned_away + counts.cc_velocity_gates_exempt_tds_fringe
            == strict.counts.cc_velocity_gates_rejected)


def test_a_lone_pair_is_not_a_couplet_and_a_beam_gap_is_not_shear():
    """The two guards on the seed rule, tested against the rule working.

    A single opposite-signed pair is two noisy gates, so the cluster
    minimum turns it away; and radials further apart than the azimuth
    gap are not adjacent beams, so a sector boundary cannot seed at all.
    """

    from gpuwm.obs.cc_qc import couplet_seed_mask, rotation_association_mask

    params = CcQcParams()
    azimuths = np.array(FRINGE_AZIMUTHS, dtype=np.float64)
    velocity = np.full((FRINGE_RADIALS, GATE_COUNT), 4.0)
    velocity[:FRINGE_RADIALS // 2, :COUPLET_GATES] = -20.0
    velocity[FRINGE_RADIALS // 2:, :COUPLET_GATES] = 20.0
    assert rotation_association_mask(velocity, azimuths, params).any()

    lone = np.full((FRINGE_RADIALS, GATE_COUNT), 4.0)
    lone[3, 0] = -20.0
    lone[4, 0] = 20.0
    assert couplet_seed_mask(lone, azimuths, params).any()      # it seeds
    assert not rotation_association_mask(lone, azimuths, params).any()

    apart = azimuths.copy()
    apart[FRINGE_RADIALS // 2:] += 90.0        # a sector boundary
    assert not couplet_seed_mask(velocity, apart, params).any()


def test_the_exemption_band_refuses_to_be_empty():
    """A band with no interior would fire never, and say nothing about it."""

    with pytest.raises(CcQcParamsError, match="tds_ref_min_dbz"):
        CcQcParams(tds_ref_min_dbz=35.0)                  # equals the shield
    with pytest.raises(CcQcParamsError, match="tds_rho_floor"):
        CcQcParams(tds_rho_floor=0.95)                    # equals rho_min
    with pytest.raises(CcQcParamsError, match="tds_couplet_min_seeds"):
        CcQcParams(tds_couplet_min_seeds=0)
    with pytest.raises(CcQcParamsError, match="tds_couplet_lobe_ms"):
        CcQcParams(tds_couplet_lobe_ms=20.0)              # delta-V never binds
    with pytest.raises(CcQcParamsError, match="tds_fringe_exempt"):
        CcQcParams(tds_fringe_exempt="yes")
