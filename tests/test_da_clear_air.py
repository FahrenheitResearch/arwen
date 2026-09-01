"""Clear-air ("zero") observations: what makes one, and what refuses one.

A clear-air zero tells the filter that storms are NOT somewhere.  Built
correctly it is the observation that suppresses spurious convection;
built carelessly it is a licence to erase real storms over an arbitrarily
large area, because "the radar never looked here" and "the radar looked
and saw nothing" are the same absence of echo in a gridded file.

So the majority of this file is refusals.  The single most important one
is ``test_a_nan_gate_is_never_clear_air``: the Level-II decoder collapses
the Message-31 "below threshold" code (raw 0, genuinely clear) and the
"range folded" code (raw 1, possibly a storm) to one NaN, so NaN cannot
be evidence of anything downstream of it.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpuwm.da import obsop
from gpuwm.da.obs_radar import (CLEAR_AIR_NAME, RadarObsAdapterError,
                                radar_grid_to_gridded_obs)
from gpuwm.da.letkf import Localization
from gpuwm.da.radar_assimilation import (RadarAssimilationConfig,
                                         RadarAssimilationError)
from gpuwm.obs.radar_grid import (CLEAR_AIR_SOURCE, CLEAR_AIR_SOURCE_CENSOR,
                                  CLEAR_AIR_VARIABLES,
                                  RadarGridSchemaError, read_radar_grid,
                                  write_radar_grid)
from gpuwm.obs.superob import (SuperobParams, SuperobParamsError,
                               merge_contributions, superob_volume)
from gpuwm.obs.sweeps import (SWEEPS_SCHEMA_CENSOR, Censor, Moment, RadarSite,
                              RadarVolume, Sweep)
from gpuwm.obs.target_grid import TargetGrid
from gpuwm.static.lambert import LambertGrid

import pathlib


def _grid(nx: int = 41, ny: int = 41, dx: float = 2000.0, nz: int = 10,
          top_m: float = 10000.0) -> TargetGrid:
    projection = LambertGrid(
        ref_lat=35.3331, ref_lon=-97.2778, truelat1=33.0, truelat2=37.0,
        stand_lon=-97.2778, dx=dx, dy=dx, e_we=nx + 1, e_sn=ny + 1)
    return TargetGrid.from_projection(
        projection, z_w=np.linspace(0.0, top_m, nz + 1), name="analytic")


def _volume(grid: TargetGrid, *, reflectivity, velocity=None, nyquist=32.0,
            azimuths=(90.0,), elevation=0.5, gate_size=250.0,
            first_gate=2125.0, site_id="AAAA", censor=None,
            velocity_censor=None) -> RadarVolume:
    """A synthetic volume whose gate values the caller controls exactly."""

    centre_j, centre_i = grid.ny // 2, grid.nx // 2
    azimuth = np.asarray(azimuths, dtype=np.float32)

    def plane(values):
        values = np.asarray(values, dtype=np.float32)
        return np.tile(values[None, :], (azimuth.size, 1))

    def codes(values):
        values = np.asarray(values, dtype=np.uint8)
        return np.tile(values[None, :], (azimuth.size, 1))

    moments = {}
    if reflectivity is not None:
        data = plane(reflectivity)
        moments["REF"] = Moment("REF", "dBZ", data.shape[1], first_gate,
                                gate_size, data,
                                None if censor is None else codes(censor))
    if velocity is not None:
        data = plane(velocity)
        moments["VEL"] = Moment(
            "VEL", "m/s", data.shape[1], first_gate, gate_size, data,
            None if velocity_censor is None else codes(velocity_censor))
    sweep = Sweep(
        sweep_index=0, elevation_number=1, elevation_angle_deg=elevation,
        nyquist_velocity_ms=nyquist, start_status=3, end_status=2,
        cut_sector=0, complete=True, azimuth_deg=azimuth,
        elevation_deg=np.full(azimuth.size, elevation, dtype=np.float32),
        moments=moments)
    return RadarVolume(
        site=RadarSite(id=site_id, name="synthetic",
                       lat_deg=float(grid.lat[centre_j, centre_i]),
                       lon_deg=float(grid.lon[centre_j, centre_i]),
                       alt_m=0.0, source="test"),
        valid_time="2026-07-28T20:03:16Z", station_id=site_id,
        volume_file=f"{site_id}20260728_200316_V06",
        volume_sha256="0" * 64, volume_bytes=8102058,
        pack_path=pathlib.Path("synthetic.rdrpack"), pack_sha256="1" * 64,
        params={"moments": ["REF"], "max_range_km": 250.0},
        framing={"magic": "AR2V0006", "block_count": 1}, sweeps=(sweep,),
        pack_schema=(SWEEPS_SCHEMA_CENSOR if censor is not None
                     else RadarVolume.pack_schema))


def _gridded(grid, volumes, params=None, censored=False):
    params = params or SuperobParams()
    contributions = [superob_volume(volume, grid, params=params,
                                    clear_air_from_censor=censored)
                     for volume in volumes]
    return merge_contributions(contributions, grid, params=params), params


def _censored_volume(grid, code, *, gates=60, params_gate_value=np.nan,
                     **kwargs):
    """A volume whose every reflectivity gate carries one censor code.

    The moment plane is NaN wherever the code is not MEASURED, exactly as
    the decoder produces it -- so a test that admits such a gate can only
    have done so via the censor plane.
    """

    values = np.full(gates, params_gate_value, dtype=np.float32)
    return _volume(grid, reflectivity=values,
                   censor=np.full(gates, code, dtype=np.uint8), **kwargs)


# ---------------------------------------------------------------------------
# how a zero is constructed
# ---------------------------------------------------------------------------


def test_a_nan_gate_is_never_clear_air():
    """The refusal the whole product rests on.

    ``rw-nexrad`` decodes BOTH the Message-31 "below threshold" code
    (raw 0 -- the radar looked and detected nothing) and the "range
    folded" code (raw 1 -- an ambiguous second-trip return that may be a
    storm) to the same NaN, and NaN-fills any radial that did not carry
    the moment at all.  Downstream of that decode the three are
    indistinguishable.

    A real KDMX volume carries about 9.7 million non-finite reflectivity
    gates against about 1.1e4 finite below-floor ones, so a rule that
    read NaN as clear air would not be slightly wrong -- it would invent
    three orders of magnitude more observations than it measured.
    """

    grid = _grid()
    params = SuperobParams(clear_air_min_gates=1.0)
    gates = np.full(60, np.nan, dtype=np.float32)
    volume = _volume(grid, reflectivity=gates)
    observations, _ = _gridded(grid, [volume], params)

    assert observations.z0_count.sum() == 0, (
        "a NaN gate contributed to a clear-air count")
    assert observations.z0_mask.sum() == 0
    assert observations.z_mask.sum() == 0

    # This test must not be able to pass because the fixture's geometry
    # missed the grid.  The gates WERE processed and were rejected for
    # being non-finite specifically.
    contribution = superob_volume(volume, grid, params=params)
    assert contribution.counts.gates_considered > 0
    assert contribution.counts.gates_nonfinite == \
        contribution.counts.gates_considered
    assert contribution.counts.gates_below_floor == 0

    # And the guard is load-bearing rather than incidental: ``NaN >= floor``
    # is False, so a NaN gate satisfies the below-floor predicate.  Only the
    # finite filter upstream keeps it out of the clear-air count.
    assert not (np.float32(np.nan) >= params.min_reflectivity_dbz)

    # The A/B: the identical fixture with finite below-floor gates DOES
    # produce clear air, so the difference measured here is finiteness and
    # nothing else.
    finite_obs, _ = _gridded(
        grid, [_volume(grid, reflectivity=np.full(60, -30.0, np.float32))],
        params)
    assert finite_obs.z0_mask.sum() > 0


# ---------------------------------------------------------------------------
# the censored regime: the decoder now says WHY a gate is not a number
# ---------------------------------------------------------------------------


def test_range_folded_gates_are_never_clear_air_under_any_configuration():
    """The permanent pin.

    Message-31 raw code 1 is *range folded*: a return the RDA cannot place
    in range because it is ambiguous between trips.  It may be a storm.
    Reading it as clear air would assimilate "no echo here" into cells that
    have one, which is the single worst thing this product can do -- worse
    than yielding nothing, because it is confidently wrong.

    So the exclusion is swept rather than asserted once: every parameter
    that could plausibly widen the clear-air net is driven to its most
    permissive legal value, in both regimes, and the yield stays zero.
    """

    grid = _grid()
    volume = _censored_volume(grid, Censor.RANGE_FOLDED)

    permissive = [
        SuperobParams(clear_air_min_gates=1.0),
        SuperobParams(clear_air_min_gates=1.0, min_reflectivity_dbz=1e30),
        SuperobParams(clear_air_min_gates=1.0, min_reflectivity_dbz=-1e30),
        SuperobParams(clear_air_min_gates=1.0, clear_air_error_dbz=1e-6),
        SuperobParams(clear_air_min_gates=1.0, max_range_km=1e4,
                      max_elevation_deg=90.0),
    ]
    for params in permissive:
        for censored in (False, True):
            observations, _ = _gridded(grid, [volume], params,
                                       censored=censored)
            assert observations.z0_count.sum() == 0, (
                f"a range-folded gate was counted as clear air "
                f"(censored={censored}, params={params})")
            assert observations.z0_mask.sum() == 0
            assert observations.z_mask.sum() == 0

    # And the refusal is visible, not silent: the pass saw these gates and
    # counted them as refused rather than never reaching them.
    contribution = superob_volume(volume, grid,
                                  params=SuperobParams(clear_air_min_gates=1.0),
                                  clear_air_from_censor=True)
    assert contribution.counts.gates_considered > 0
    assert contribution.counts.censor.reflectivity_range_folded > 0
    assert contribution.counts.censor.range_folded_gates_refused == \
        contribution.counts.censor.reflectivity_range_folded
    assert contribution.counts.censor.clear_air_gates_admitted == 0

    # The A/B that proves the fixture's geometry reaches the grid at all:
    # the identical volume with BELOW_THRESHOLD in place of RANGE_FOLDED
    # does yield clear air.  The difference measured is the code and
    # nothing else.
    clear = _censored_volume(grid, Censor.BELOW_THRESHOLD)
    observations, _ = _gridded(grid, [clear],
                               SuperobParams(clear_air_min_gates=1.0),
                               censored=True)
    assert observations.z0_mask.sum() > 0


def test_a_never_collected_gate_is_not_a_clear_air_observation():
    """The third NaN source, and the one that covers whole azimuths.

    A radial that never carried the moment is NaN-filled into the pack's
    rectangle.  It is not a measurement of anything; admitting it would
    claim clear air over every azimuth a split cut did not scan.
    """

    grid = _grid()
    volume = _censored_volume(grid, Censor.NOT_COLLECTED)
    params = SuperobParams(clear_air_min_gates=1.0)
    observations, _ = _gridded(grid, [volume], params, censored=True)
    assert observations.z0_count.sum() == 0
    assert observations.z0_mask.sum() == 0

    contribution = superob_volume(volume, grid, params=params,
                                  clear_air_from_censor=True)
    assert contribution.counts.censor.reflectivity_not_collected > 0
    assert contribution.counts.censor.clear_air_gates_admitted == 0


def test_below_threshold_gates_are_clear_air_only_in_the_censored_regime():
    """The fix, stated as a difference.

    The same volume, the same parameters; the only variable is whether the
    decoder's reason for each NaN was preserved.  Without it the gates are
    ambiguous and yield nothing; with it they are what the radar actually
    reported -- a detection of no significant return.
    """

    grid = _grid()
    params = SuperobParams(clear_air_min_gates=1.0)
    volume = _censored_volume(grid, Censor.BELOW_THRESHOLD)

    plain, _ = _gridded(grid, [volume], params, censored=False)
    assert plain.z0_count.sum() == 0
    assert plain.z0_mask.sum() == 0
    assert plain.clear_air_source == CLEAR_AIR_SOURCE

    censored, _ = _gridded(grid, [volume], params, censored=True)
    assert censored.z0_count.sum() > 0
    assert censored.z0_mask.sum() > 0
    assert censored.clear_air_source == CLEAR_AIR_SOURCE_CENSOR

    # An echo gate still vetoes the cell: the censored regime widens what
    # counts as evidence of clear air, not what survives a contradiction.
    mixed = np.full(60, np.nan, dtype=np.float32)
    mixed[0] = 40.0
    codes = np.full(60, Censor.BELOW_THRESHOLD, dtype=np.uint8)
    codes[0] = Censor.MEASURED
    vetoed, _ = _gridded(
        grid, [_volume(grid, reflectivity=mixed, censor=codes)], params,
        censored=True)
    assert not (vetoed.z_mask.astype(bool)
                & vetoed.z0_mask.astype(bool)).any()


def test_the_default_regime_is_unchanged_by_the_presence_of_censor_planes():
    """Do no harm, at the observation level rather than the byte level.

    A v2 pack read WITHOUT asking for the censored regime must produce the
    identical product a v1 pack produces: same arrays, same counters, same
    clear_air_source.  If it did not, every already-published observation
    file would silently change meaning the moment the decoder was upgraded.
    """

    grid = _grid()
    params = SuperobParams(clear_air_min_gates=1.0)
    values = np.concatenate([
        np.full(30, -30.0, dtype=np.float32),      # finite, below floor
        np.full(30, np.nan, dtype=np.float32),     # censored
    ])
    codes = np.concatenate([
        np.full(30, Censor.MEASURED, dtype=np.uint8),
        np.full(15, Censor.BELOW_THRESHOLD, dtype=np.uint8),
        np.full(15, Censor.RANGE_FOLDED, dtype=np.uint8),
    ])
    with_planes = _volume(grid, reflectivity=values, censor=codes)
    without = _volume(grid, reflectivity=values)

    a = superob_volume(with_planes, grid, params=params)
    b = superob_volume(without, grid, params=params)
    assert a.clear_air_source == b.clear_air_source == CLEAR_AIR_SOURCE
    np.testing.assert_array_equal(a.z0_count, b.z0_count)
    np.testing.assert_array_equal(a.z_count, b.z_count)
    np.testing.assert_array_equal(a.z_linear_sum, b.z_linear_sum)
    assert a.counts.to_payload() == b.counts.to_payload()
    # The counters payload is serialized verbatim into every observation
    # file's provenance, so "no new keys" is what keeps committed obs
    # digests reproducible.
    assert "censor" not in a.counts.to_payload()
    assert a.counts.censor is None


def test_asking_for_the_censored_regime_without_the_planes_is_refused():
    """A silent downgrade would publish a thin product under the wide
    regime's clear_air_source, claiming coverage it does not have."""

    grid = _grid()
    volume = _volume(grid, reflectivity=np.full(60, -30.0, dtype=np.float32))
    with pytest.raises(ValueError, match="--censor-flags"):
        superob_volume(volume, grid, clear_air_from_censor=True)


def test_two_radars_established_differently_refuse_to_merge():
    """z0_counts from the two regimes cover different fractions of the
    domain, so their sum describes no coverage in particular."""

    grid = _grid()
    params = SuperobParams(clear_air_min_gates=1.0)
    plain = _volume(grid, reflectivity=np.full(60, -30.0, dtype=np.float32),
                    site_id="AAAA")
    censored = _censored_volume(grid, Censor.BELOW_THRESHOLD, site_id="BBBB")
    contributions = [
        superob_volume(plain, grid, params=params),
        superob_volume(censored, grid, params=params,
                       clear_air_from_censor=True),
    ]
    with pytest.raises(ValueError, match="different means"):
        merge_contributions(contributions, grid, params=params)


def test_the_censored_source_round_trips_and_the_adapter_accepts_it(tmp_path):
    """End to end: the writer publishes the new regime, the reader accepts
    it, and the DA adapter -- which refuses regimes it has not been taught
    -- assimilates it."""

    grid = _grid()
    params = SuperobParams(clear_air_min_gates=1.0)
    volume = _censored_volume(grid, Censor.BELOW_THRESHOLD)
    observations, _ = _gridded(grid, [volume], params, censored=True)
    path = tmp_path / "censored.nc"
    write_radar_grid(path, observations, grid,
                     valid_time="2026-07-28T20:03:16Z", params=params)
    document = read_radar_grid(path)
    assert document["clear_air_source"] == CLEAR_AIR_SOURCE_CENSOR

    floor = obsop.clear_air_floor_dbz(6)
    batches, provenance = radar_grid_to_gridded_obs(
        document, expected_grid=grid,
        clear_air_simulated=_hx(document, floor),
        clear_air_value_dbz=floor,
        clear_air_localization=Localization(horizontal_m=6000.0,
                                            vertical_m=2000.0))
    assert [batch.name for batch in batches] == [CLEAR_AIR_NAME]
    assert batches[0].mask.sum() > 0
    assert provenance["clear_air_source"] == CLEAR_AIR_SOURCE_CENSOR
    entry, = provenance["batches"]
    assert entry["clear_air_source"] == CLEAR_AIR_SOURCE_CENSOR


def test_measured_below_floor_gates_are_clear_air():
    """The positive case: finite, in range, on grid, below the floor."""

    grid = _grid()
    params = SuperobParams(clear_air_min_gates=1.0)
    gates = np.full(60, -30.0, dtype=np.float32)  # below the -15 floor
    observations, _ = _gridded(grid, [_volume(grid, reflectivity=gates)],
                               params)

    assert observations.z0_mask.sum() > 0
    assert observations.z0_count.sum() > 0
    # A clear-air cell is never also an echo cell.
    assert not np.any(observations.z0_mask.astype(bool)
                      & observations.z_mask.astype(bool))
    # The error is the clear-air sigma, not the echo sigma.
    clear = observations.z0_mask.astype(bool)
    assert np.allclose(observations.z0_err[clear], params.clear_air_error_dbz)


def test_a_cell_the_beam_never_reached_is_not_clear_air():
    """Absence of a gate is absence of an observation, not a zero.

    One short radial leaves the overwhelming majority of the grid
    untouched.  Those cells must end the pass with no clear-air claim --
    they are exactly the "never looked" population that a careless
    implementation converts into confident zeroes.
    """

    grid = _grid()
    params = SuperobParams(clear_air_min_gates=1.0)
    gates = np.full(8, -30.0, dtype=np.float32)
    observations, _ = _gridded(grid, [_volume(grid, reflectivity=gates)],
                               params)

    clear = observations.z0_mask.astype(bool)
    assert clear.sum() > 0, "the fixture produced no clear air at all"
    assert clear.sum() < clear.size / 10, (
        "a single short radial claimed clear air over the whole grid")
    assert np.all(observations.z0_count[~clear] == 0)


def test_an_echo_gate_vetoes_the_cell_even_beside_clear_gates():
    """A cell containing any echo is a cell with echo in it.

    The gates alternate strong echo and clear within one radial, so the
    same cells accumulate both.  Reflectivity must win: half-clear is not
    clear, and averaging the two would be inventing a third thing.
    """

    grid = _grid()
    params = SuperobParams(clear_air_min_gates=1.0)
    gates = np.empty(60, dtype=np.float32)
    gates[0::2] = 40.0
    gates[1::2] = -30.0
    observations, _ = _gridded(grid, [_volume(grid, reflectivity=gates)],
                               params)

    echo = observations.z_mask.astype(bool)
    clear = observations.z0_mask.astype(bool)
    assert echo.sum() > 0 and observations.z0_count.sum() > 0, (
        "the fixture did not produce both echo and below-floor gates")
    assert not np.any(echo & clear), (
        "a cell carrying echo gates was also reported clear")


def test_one_radar_seeing_echo_vetoes_another_radar_calling_it_clear():
    """The cross-radar veto, which is a merge property not a gate one.

    Reflectivity is summed across radars, so a cell any radar found echo
    in has a true ``z_mask``; the clear-air mask must defer to it.  This
    is also the physically right call -- the radar seeing the storm is
    usually the nearer one, and the other is looking through it or over
    it.
    """

    grid = _grid()
    params = SuperobParams(clear_air_min_gates=1.0)
    clear_volume = _volume(grid, reflectivity=np.full(60, -30.0, np.float32),
                           site_id="AAAA")
    echo_volume = _volume(grid, reflectivity=np.full(60, 45.0, np.float32),
                          site_id="BBBB")

    alone, _ = _gridded(grid, [clear_volume], params)
    both, _ = _gridded(grid, [clear_volume, echo_volume], params)

    assert alone.z0_mask.sum() > 0, "the clear radar alone produced no zeroes"
    overlap = alone.z0_mask.astype(bool) & both.z_mask.astype(bool)
    assert overlap.sum() > 0, "the fixture's two radars did not overlap"
    assert not np.any(both.z0_mask.astype(bool) & both.z_mask.astype(bool))
    assert both.z0_mask.sum() < alone.z0_mask.sum(), (
        "the second radar's echo did not withdraw any clear-air claim")


def test_too_few_supporting_gates_is_not_a_clear_air_observation():
    """One clipped gate in a cell corner is geometry, not a measurement."""

    grid = _grid()
    gates = np.full(60, -30.0, dtype=np.float32)
    volume = _volume(grid, reflectivity=gates)

    lenient, _ = _gridded(grid, [volume], SuperobParams(clear_air_min_gates=1.0))
    strict, _ = _gridded(grid, [volume],
                         SuperobParams(clear_air_min_gates=1000.0))

    assert lenient.z0_mask.sum() > 0
    assert strict.z0_mask.sum() == 0, (
        "a 1000-gate threshold still admitted cells")


def test_a_min_gate_threshold_below_one_is_refused():
    """Below 1 every unmeasured cell satisfies the threshold with its 0."""

    with pytest.raises(SuperobParamsError, match="clear_air_min_gates"):
        SuperobParams(clear_air_min_gates=0.0)


def test_a_zero_clear_air_error_is_refused_as_infinite_confidence():
    with pytest.raises(SuperobParamsError, match="clear_air_error_dbz"):
        SuperobParams(clear_air_error_dbz=0.0)


# ---------------------------------------------------------------------------
# the file contract
# ---------------------------------------------------------------------------


def test_clear_air_round_trips_through_the_writer_and_reader(tmp_path):
    grid = _grid()
    params = SuperobParams(clear_air_min_gates=1.0)
    observations, _ = _gridded(grid, [_volume(
        grid, reflectivity=np.full(60, -30.0, np.float32))], params)
    path = tmp_path / "obs.nc"
    write_radar_grid(path, observations, grid,
                     valid_time="2026-01-01T00:00:00Z", params=params)

    document = read_radar_grid(path, expected_grid=grid)
    assert document["clear_air_source"] == CLEAR_AIR_SOURCE
    for name in CLEAR_AIR_VARIABLES:
        assert name in document["variables"], name
    assert np.array_equal(document["variables"]["z0_mask"].astype(bool),
                          observations.z0_mask.astype(bool))
    assert int(document["variables"]["z0_count"].sum()) == int(
        observations.z0_count.sum())


def test_a_file_without_the_clear_air_extension_still_reads(tmp_path):
    """Backward compatibility, which is a correctness property here.

    Every observation file written before this capability -- including
    the ones the cycling receipts were produced from -- has no z0
    variables.  Promoting them to canonical would make those files
    unreadable and the runs that consumed them unreproducible.
    """

    grid = _grid()
    observations, params = _gridded(grid, [_volume(
        grid, reflectivity=np.full(60, 30.0, np.float32))])
    observations.z0_mask = None
    observations.z0_count = None
    observations.z0_err = None
    path = tmp_path / "legacy.nc"
    write_radar_grid(path, observations, grid,
                     valid_time="2026-01-01T00:00:00Z", params=params)

    document = read_radar_grid(path, expected_grid=grid)
    assert document["clear_air_source"] is None
    assert "z0_mask" not in document["variables"]


# ---------------------------------------------------------------------------
# the adapter's refusals
# ---------------------------------------------------------------------------


def _document(tmp_path, *, reflectivity, min_gates=1.0):
    grid = _grid()
    params = SuperobParams(clear_air_min_gates=min_gates)
    observations, _ = _gridded(grid, [_volume(grid,
                                              reflectivity=reflectivity)],
                               params)
    path = tmp_path / "obs.nc"
    write_radar_grid(path, observations, grid,
                     valid_time="2026-01-01T00:00:00Z", params=params)
    return grid, read_radar_grid(path, expected_grid=grid)


def _hx(document, value):
    shape = tuple(document["variables"]["z_mask"].shape)
    return np.full((3,) + shape, value, dtype=np.float64)


def test_clear_air_builds_a_second_batch_at_the_schemes_floor(tmp_path):
    grid, document = _document(tmp_path,
                               reflectivity=np.full(60, -30.0, np.float32))
    floor = obsop.clear_air_floor_dbz(6)

    batches, provenance = radar_grid_to_gridded_obs(
        document, expected_grid=grid,
        clear_air_simulated=_hx(document, floor),
        clear_air_value_dbz=floor,
        clear_air_localization=Localization(horizontal_m=6000.0,
                                            vertical_m=2000.0))

    names = [batch.name for batch in batches]
    assert names == [CLEAR_AIR_NAME]
    batch = batches[0]
    assert np.all(batch.values[batch.mask] == floor)
    assert batch.localization.horizontal_m == 6000.0
    entry, = provenance["batches"]
    assert entry["kind"] == "clear_air_reflectivity"
    assert entry["value_dbz"] == floor
    assert entry["observed_points"] == int(batch.mask.sum())


def test_a_file_with_no_clear_air_assessment_is_refused(tmp_path):
    """Not silently downgraded to "derive zeroes from z_mask"."""

    grid = _grid()
    observations, params = _gridded(grid, [_volume(
        grid, reflectivity=np.full(60, 30.0, np.float32))])
    observations.z0_mask = None
    observations.z0_count = None
    observations.z0_err = None
    path = tmp_path / "legacy.nc"
    write_radar_grid(path, observations, grid,
                     valid_time="2026-01-01T00:00:00Z", params=params)
    document = read_radar_grid(path, expected_grid=grid)

    with pytest.raises(RadarObsAdapterError, match="without a clear-air"):
        radar_grid_to_gridded_obs(
            document, expected_grid=grid,
            clear_air_simulated=_hx(document, -35.0),
            clear_air_value_dbz=-35.0)


def test_clear_air_without_a_stated_value_is_refused(tmp_path):
    grid, document = _document(tmp_path,
                               reflectivity=np.full(60, -30.0, np.float32))
    with pytest.raises(RadarObsAdapterError,
                       match="clear_air_value_dbz is required"):
        radar_grid_to_gridded_obs(
            document, expected_grid=grid,
            clear_air_simulated=_hx(document, -35.0))


def test_an_unrecognised_clear_air_source_is_refused(tmp_path):
    grid, document = _document(tmp_path,
                               reflectivity=np.full(60, -30.0, np.float32))
    document = dict(document)
    document["clear_air_source"] = "below_threshold_flag"
    with pytest.raises(RadarObsAdapterError, match="clear_air_source"):
        radar_grid_to_gridded_obs(
            document, expected_grid=grid,
            clear_air_simulated=_hx(document, -35.0),
            clear_air_value_dbz=-35.0)


def test_a_cell_marked_both_echo_and_clear_is_refused(tmp_path):
    grid, document = _document(tmp_path,
                               reflectivity=np.full(60, -30.0, np.float32))
    variables = dict(document["variables"])
    variables["z_mask"] = variables["z0_mask"].copy()
    document = dict(document, variables=variables)

    with pytest.raises(RadarObsAdapterError, match="both as reflectivity"):
        radar_grid_to_gridded_obs(
            document, expected_grid=grid,
            clear_air_simulated=_hx(document, -35.0),
            clear_air_value_dbz=-35.0)


def test_a_clear_air_mask_with_no_supporting_gates_is_refused(tmp_path):
    grid, document = _document(tmp_path,
                               reflectivity=np.full(60, -30.0, np.float32))
    variables = dict(document["variables"])
    variables["z0_count"] = np.zeros_like(variables["z0_count"])
    document = dict(document, variables=variables)

    with pytest.raises(RadarObsAdapterError, match="no supporting gates"):
        radar_grid_to_gridded_obs(
            document, expected_grid=grid,
            clear_air_simulated=_hx(document, -35.0),
            clear_air_value_dbz=-35.0)


def test_a_partial_clear_air_variable_set_is_refused(tmp_path):
    """All three variables or none; a mask with no error is unusable."""

    grid = _grid()
    params = SuperobParams(clear_air_min_gates=1.0)
    observations, _ = _gridded(grid, [_volume(
        grid, reflectivity=np.full(60, -30.0, np.float32))], params)
    path = tmp_path / "obs.nc"
    write_radar_grid(path, observations, grid,
                     valid_time="2026-01-01T00:00:00Z", params=params)

    import netCDF4
    with netCDF4.Dataset(path, "a") as dataset:
        dataset.renameVariable("z0_err", "z0_err_removed")

    with pytest.raises(RadarGridSchemaError, match="clear-air variables"):
        read_radar_grid(path, expected_grid=grid)


# ---------------------------------------------------------------------------
# the scheme floor
# ---------------------------------------------------------------------------


def test_the_clear_air_floor_is_per_scheme_and_never_guessed():
    assert obsop.clear_air_floor_dbz(6) == -35.0
    assert obsop.clear_air_floor_dbz(8) == -35.0
    assert obsop.clear_air_floor_dbz(10) == -35.0
    # NSSL is the one that is not floor-interchangeable with the rest.
    assert obsop.clear_air_floor_dbz(18) == 0.0
    with pytest.raises(ValueError, match="no clear-air reflectivity floor"):
        obsop.clear_air_floor_dbz(99)


def test_p3s_clear_air_floor_is_refused_by_name_and_not_by_absence():
    """mp=50 is OUT of the table on purpose, and says so.

    P3 is a shipped, front-doored scheme, so its absence from a scalar
    lookup has to read as a decision rather than as a scheme nobody got
    to.  ``-36.9897`` is what its H(x) reads at a clear LEVEL and
    ``-99.0`` is what it reads through a column that holds no hydrometeor
    anywhere; both are in the same field, so no entry in a
    ``dict[int, float]`` can be right for P3 and the refusal has to say
    which two numbers it is between.
    """
    assert 50 not in obsop.CLEAR_AIR_FLOOR_DBZ
    assert 50 in obsop.CLEAR_AIR_FLOOR_IS_NOT_ONE_NUMBER

    with pytest.raises(ValueError) as excinfo:
        obsop.clear_air_floor_dbz(50)
    message = str(excinfo.value)
    # Not the generic "nobody read this one yet" refusal.
    assert "no clear-air reflectivity floor is recorded" not in message
    assert "no single clear-air reflectivity floor" in message
    # The scheme, the two values, and what each wrong choice costs.
    assert "P3" in message
    assert "-36.9897" in message
    assert "-99.0" in message
    assert "+64 dB" in message and "-62 dB" in message
    assert "module_mp_p3.F" in message

    # And the front door carries it: the config refuses at construction,
    # not mid-cycle.
    with pytest.raises(ValueError, match="no single clear-air"):
        _config(clear_air=True, mp_physics=50,
                analysis_fields=("thp", "qv", "qr", "u", "v"))


def test_p3s_two_clear_air_values_are_measured_and_not_asserted():
    """Derive the refusal's two numbers from P3 itself.

    The reason paragraph is only worth having if it cannot rot, so run the
    scheme rather than restate it: adjacent columns, one carrying rain and
    two bone dry, must come back with -36.9897 dBZ in the cloudy column's
    clear levels and -99.0 dBZ throughout the dry ones.
    """
    from gpuwm.core.p3 import p3_main

    ni, nk = 3, 6
    zeros = lambda: np.zeros((ni, nk), np.float32)  # noqa: E731
    qc, nc, qr, nr = zeros(), zeros(), zeros(), zeros()
    qi, qir, nitot, qib, ssat = (zeros() for _ in range(5))
    pres = np.tile(np.linspace(100000.0, 50000.0, nk, dtype=np.float32),
                   (ni, 1))
    dzq = np.full((ni, nk), 500.0, np.float32)
    # Bone dry and warm: no level is within 5% of saturation over water or
    # ice, so p3_main's entry test finds neither hydrometeors nor possible
    # nucleation in the two clear columns.
    th = np.full((ni, nk), 320.0, np.float32)
    qv = np.full((ni, nk), 1.0e-8, np.float32)
    qr[1, 1] = np.float32(1.0e-3)
    nr[1, 1] = np.float32(1.0e4)
    ze, effc, effi = zeros(), zeros(), zeros()
    vmi, di, rhoi = zeros(), zeros(), zeros()

    p3_main(qc, nc, qr, nr, th.copy(), th, qv.copy(), qv, 30.0,
            qi, qir, nitot, qib, ssat, pres, dzq, 1,
            np.zeros(ni, np.float32), np.zeros(ni, np.float32),
            ze, effc, effi, vmi, di, rhoi)

    dry_column_value = np.float32(-99.0)
    clear_level_value = np.float32(-36.9897)
    assert np.all(ze[0] == dry_column_value)
    assert np.all(ze[2] == dry_column_value)
    # The cloudy column's own clear levels take the OTHER value, in the
    # same field, one grid cell away.
    assert np.all(ze[1, 2:] == clear_level_value)
    assert ze[1, 0] > 0.0 and ze[1, 1] > 0.0

    # Which is the whole point: 62 dB apart, and -35 is neither of them.
    assert float(dry_column_value) != -35.0
    assert float(clear_level_value) != -35.0
    assert float(clear_level_value) - float(dry_column_value) > 60.0


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def _config(**overrides):
    kwargs = dict(localization=Localization(horizontal_m=12000.0,
                                            vertical_m=3000.0),
                  rtps_alpha=0.9,
                  analysis_fields=("thp", "qv", "u", "v", "qr", "qnr"),
                  positivity_policy="clip", mp_physics=10)
    kwargs.update(overrides)
    return RadarAssimilationConfig(**kwargs)


def test_clear_air_is_off_by_default():
    assert _config().clear_air is False


def test_clear_air_against_a_wind_only_state_vector_is_refused():
    with pytest.raises(RadarAssimilationError, match="wind-only state"):
        _config(clear_air=True, analysis_fields=("u", "v"),
                positivity_policy=None)


def test_clear_air_with_neither_a_value_nor_a_scheme_is_refused():
    with pytest.raises(RadarAssimilationError, match="clear_air_value_dbz"):
        _config(clear_air=True, mp_physics=None)


def test_clear_air_with_an_unknown_scheme_is_refused_at_config_time():
    with pytest.raises(ValueError, match="no clear-air reflectivity floor"):
        _config(clear_air=True, mp_physics=99)


def test_clear_air_error_inflation_below_one_is_refused():
    with pytest.raises(RadarAssimilationError,
                       match="clear_air_error_inflation"):
        _config(clear_air=True, clear_air_error_inflation=0.5)


def test_clear_air_thinning_defaults_above_one_and_refuses_zero():
    assert _config().clear_air_thinning_cells > 1, (
        "clear air is the majority of a volume; an unthinned default would "
        "put thousands of near-identical numbers in one localisation lens")
    with pytest.raises(RadarAssimilationError,
                       match="clear_air_thinning_cells"):
        _config(clear_air=True, clear_air_thinning_cells=0)


def test_a_clear_air_file_leaves_the_velocity_only_batches_bit_identical(
        tmp_path):
    """Velocity-only stays exactly reproducible, which is the contract.

    A cycle re-run with the old settings must give the old answer.  The
    filter is deterministic given its prior and its observation batches,
    so identical batches are the property to pin -- and the risk this
    change introduces is precisely that new variables in the file perturb
    the velocity path that ignores them.
    """

    grid = _grid()
    params = SuperobParams(clear_air_min_gates=1.0)
    volume = _volume(grid, reflectivity=np.full(60, -30.0, np.float32),
                     velocity=np.full(60, 7.5, np.float32))
    observations, _ = _gridded(grid, [volume], params)

    with_zeroes = tmp_path / "with-zeroes.nc"
    write_radar_grid(with_zeroes, observations, grid,
                     valid_time="2026-01-01T00:00:00Z", params=params)

    observations.z0_mask = None
    observations.z0_count = None
    observations.z0_err = None
    without = tmp_path / "without.nc"
    write_radar_grid(without, observations, grid,
                     valid_time="2026-01-01T00:00:00Z", params=params)

    def velocity_batches(path):
        document = read_radar_grid(path, expected_grid=grid)
        shape = tuple(document["variables"]["z_mask"].shape)

        def simulated(index, radar):
            return np.zeros((3,) + shape, dtype=np.float64)

        batches, _ = radar_grid_to_gridded_obs(
            document, expected_grid=grid, velocity_simulated=simulated)
        return batches

    new, old = velocity_batches(with_zeroes), velocity_batches(without)
    assert [b.name for b in new] == [b.name for b in old]
    assert new, "the fixture produced no velocity batch to compare"
    for fresh, legacy in zip(new, old):
        # Bit-for-bit, not allclose: this is a reproducibility claim.
        assert np.array_equal(fresh.values, legacy.values)
        assert np.array_equal(fresh.errors, legacy.errors)
        assert np.array_equal(fresh.mask, legacy.mask)


def test_clear_air_thinning_is_a_no_op_at_unit_settings(tmp_path):
    """At 1 cell and no inflation the stage returns the caller's document.

    The early return is what keeps a configuration that asks for nothing
    from silently rewriting arrays, and it is the identity a velocity-only
    or reflectivity-only cycle relies on.
    """

    from gpuwm.da.radar_assimilation import _thinned_clear_air_document

    grid, document = _document(tmp_path,
                               reflectivity=np.full(60, -30.0, np.float32))
    same, receipt = _thinned_clear_air_document(
        document, _config(clear_air=True, clear_air_thinning_cells=1,
                          clear_air_error_inflation=1.0))
    assert same is document
    assert receipt is None


def test_clear_air_thinning_reduces_the_assimilated_count(tmp_path):
    from gpuwm.da.radar_assimilation import _thinned_clear_air_document

    grid, document = _document(tmp_path,
                               reflectivity=np.full(60, -30.0, np.float32))
    before = int(np.asarray(document["variables"]["z0_mask"]).sum())
    thinned, receipt = _thinned_clear_air_document(
        document, _config(clear_air=True, clear_air_thinning_cells=3))
    after = int(np.asarray(thinned["variables"]["z0_mask"]).sum())

    assert before > 0
    assert after < before
    assert receipt["points_before"] == before
    assert receipt["points_after"] == after
    # The caller's document is never mutated in place.
    assert int(np.asarray(document["variables"]["z0_mask"]).sum()) == before


def test_clear_air_error_inflation_is_applied_exactly_once(tmp_path):
    """Inflation lives in the document, not in the document AND the adapter.

    Both stages accept an inflation factor; applying it twice would square
    it silently, which is a four-fold error in variance at 2.0.
    """

    from gpuwm.da.radar_assimilation import _thinned_clear_air_document

    grid, document = _document(tmp_path,
                               reflectivity=np.full(60, -30.0, np.float32))
    base = np.asarray(document["variables"]["z0_err"], dtype=np.float64)
    thinned, _ = _thinned_clear_air_document(
        document, _config(clear_air=True, clear_air_thinning_cells=1,
                          clear_air_error_inflation=2.0))
    inflated = np.asarray(thinned["variables"]["z0_err"], dtype=np.float64)
    assert np.allclose(inflated, base * 2.0)

    # And the adapter, handed an already-inflated document, must not
    # inflate again -- which is why the cycle passes 1.0 there.
    batches, _ = radar_grid_to_gridded_obs(
        thinned, expected_grid=grid,
        clear_air_simulated=_hx(thinned, -35.0),
        clear_air_value_dbz=-35.0, clear_air_error_inflation=1.0)
    batch, = batches
    assert np.allclose(batch.errors[batch.mask], inflated[batch.mask])


def test_clear_air_and_reflectivity_are_independently_switchable():
    """Ablation depends on this: the two do opposite things."""

    assert _config(clear_air=True).reflectivity is False
    assert _config(reflectivity=True).clear_air is False
    both = _config(clear_air=True, reflectivity=True)
    assert both.clear_air and both.reflectivity
