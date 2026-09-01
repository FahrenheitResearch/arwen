"""Gates for the bundle-10 observation-path findings.

Each test here is one of the review's "cheap gates": a property the stage
must have, written so that it fails against the code as it stood.  The
red-then-green proof for each is recorded in the lane's handoff entry.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpuwm.obs.dealias import (
    DealiasParams, DealiasParamsError, ENGINE_REGION_GLOBAL,
    ENGINE_VAD_REGION, STATE_REJECTED, dealias_sweep, scipy_available)
from gpuwm.obs.dealias_region import region_engine_available
from gpuwm.obs.geometry import beam_geometry
import gpuwm.obs.dealias as dealias_module
from gpuwm.da.positivity import apply_positivity


scipy_only = pytest.mark.skipif(not scipy_available(),
                                reason="dealiasing needs scipy")
region_only = pytest.mark.skipif(not region_engine_available(),
                                 reason="needs the region-global bridge")

# The per-radial fold arithmetic is the VAD-referenced engine's: it is the
# arm that carries an interval per radial into the solve.  These gates name
# that engine rather than riding the shipped default, which is
# "region-global" since 2026-08-12 and which decides every fold in ONE
# interval -- see test_region_global_refuses_a_nonuniform_sweep for what it
# does instead, and why that is the correct thing for it to do.


def _wind_sweep(nyquist_by_radial, *, speed=39.0, direction_deg=250.0,
                radials=180, gates=96):
    """A folded uniform-wind cut, and the truth it folded from.

    Every radial sees the same environmental wind, so the unfolded field is
    exactly the VAD sinusoid; only the interval each radial was quantised in
    differs.
    """

    azimuth = np.linspace(0.0, 358.0, radials)
    theta = np.radians(direction_deg)
    u = -speed * np.sin(theta)
    v = -speed * np.cos(theta)
    truth = (u * np.sin(np.radians(azimuth)) + v * np.cos(np.radians(azimuth)))
    truth = np.repeat(truth[:, None], gates, axis=1)
    nyq = np.asarray(nyquist_by_radial, dtype=np.float64)
    if nyq.ndim == 0:
        nyq = np.full(radials, float(nyq))
    interval = (2.0 * nyq)[:, None]
    folded = truth - interval * np.rint(truth / interval)
    return folded, truth, nyq


# --- finding 1: per-radial Nyquist ------------------------------------------

@scipy_only
def test_uniform_sweep_is_unchanged_by_the_per_radial_path():
    """Gate (a): supplying a constant array must reproduce the scalar path.

    Bit-identical, not close: the array path is the same arithmetic when the
    array is constant, and anything else means the refactor moved a number.
    """

    params = DealiasParams(engine=ENGINE_VAD_REGION)
    folded, _truth, nyq = _wind_sweep(25.51)
    azimuth = np.linspace(0.0, 358.0, folded.shape[0])
    scalar = dealias_sweep(folded, azimuth, 25.51, params)
    array = dealias_sweep(folded, azimuth, 25.51, params,
                          nyquist_by_radial=nyq)
    np.testing.assert_array_equal(scalar.velocity, array.velocity)
    np.testing.assert_array_equal(scalar.state, array.state)
    np.testing.assert_array_equal(scalar.fold, array.fold)
    assert scalar.stats["gates_unfolded"] == array.stats["gates_unfolded"]


@scipy_only
def test_mixed_nyquist_sweep_recovers_its_own_truth():
    """Gate (b): a two-group cut, unfolded back to the wind it came from."""

    params = DealiasParams(engine=ENGINE_VAD_REGION)
    radials = 180
    nyq = np.where(np.arange(radials) < radials // 2, 25.51, 32.0)
    folded, truth, _ = _wind_sweep(nyq, radials=radials)
    azimuth = np.linspace(0.0, 358.0, radials)

    result = dealias_sweep(folded, azimuth, float(nyq.min()), params,
                           nyquist_by_radial=nyq)
    kept = result.state != STATE_REJECTED
    assert kept.sum() > 0.5 * folded.size, "the cut was almost entirely refused"
    np.testing.assert_allclose(result.velocity[kept], truth[kept], atol=1e-6)
    assert result.stats["nyquist_distinct"] == [25.51, 32.0]
    assert result.stats["nyquist_transition_pairs"] > 0


@scipy_only
def test_every_correction_is_a_multiple_of_its_own_radials_interval():
    """Gate (c): the modular identity.  The scalar path fails this exactly.

    Both arms run on the same mixed cut.  The array arm's corrections must
    land on each radial's own ``2*Vn`` lattice; the scalar arm reasons in the
    minimum's interval throughout, so wherever it corrects a 32 m/s radial it
    lands off that radial's lattice by 12.98 m/s -- finite, smooth, and below
    every downstream bound.
    """

    params = DealiasParams(engine=ENGINE_VAD_REGION)
    radials = 180
    nyq = np.where(np.arange(radials) < radials // 2, 25.51, 32.0)
    folded, _truth, _ = _wind_sweep(nyq, radials=radials)
    azimuth = np.linspace(0.0, 358.0, radials)
    interval = (2.0 * nyq)[:, None]

    def off_lattice(result):
        kept = result.state != STATE_REJECTED
        correction = np.where(kept, result.velocity - folded, 0.0)
        ratio = np.where(kept, correction / interval, 0.0)
        return int(np.count_nonzero(
            kept & (np.abs(ratio - np.rint(ratio)) > 1e-9)))

    array = dealias_sweep(folded, azimuth, float(nyq.min()), params,
                          nyquist_by_radial=nyq)
    assert off_lattice(array) == 0

    scalar = dealias_sweep(folded, azimuth, float(nyq.min()), params)
    assert off_lattice(scalar) > 0, (
        "the scalar path was expected to violate the modular identity on a "
        "mixed-Nyquist cut; if it no longer does, this gate has stopped "
        "measuring what it was written for")


def test_legacy_disagree_pack_refuses_to_be_dealiased():
    """Gate (d): fail closed when the pack knows it is nonuniform."""

    params = DealiasParams(engine=ENGINE_VAD_REGION)
    folded, _truth, _ = _wind_sweep(25.51, radials=24, gates=8)
    azimuth = np.linspace(0.0, 345.0, 24)
    with pytest.raises(DealiasParamsError, match="nyquist_radials_disagree"):
        dealias_sweep(folded, azimuth, 25.51, params,
                      nyquist_radials_disagree=True)
    # ... and it is the missing array, not the flag, that is fatal.
    dealias_sweep(folded, azimuth, 25.51, params,
                  nyquist_radials_disagree=True,
                  nyquist_by_radial=np.full(24, 25.51))


@scipy_only
def test_a_radial_with_no_nyquist_is_refused_not_borrowed():
    params = DealiasParams(engine=ENGINE_VAD_REGION)
    radials = 60
    nyq = np.full(radials, 25.51)
    nyq[:5] = np.nan
    folded, _truth, _ = _wind_sweep(np.where(np.isnan(nyq), 25.51, nyq),
                                    radials=radials, gates=32)
    azimuth = np.linspace(0.0, 354.0, radials)
    result = dealias_sweep(folded, azimuth, 25.51, params,
                           nyquist_by_radial=nyq)
    assert bool(np.all(result.state[:5] == STATE_REJECTED))
    assert result.stats["nyquist_radials_no_value"] == 5
    assert (result.stats["gates_unchanged"] + result.stats["gates_unfolded"]
            + result.stats["gates_rejected"]) == result.stats["gates_finite"]


# --- finding 1, on the SHIPPED DEFAULT engine -------------------------------
# A bare run does not use the arm the five gates above pin.  These say what
# the default arm does with the same evidence.

@region_only
def test_region_global_refuses_a_nonuniform_sweep():
    """The default engine fails closed rather than quietly mis-correcting.

    Its native solver reduces the per-ray Nyquist array to the FIRST usable
    element and decides every cross-ray fold in that one interval, then
    applies the chosen integer at each ray's own interval.  On a 25.51/32.0
    cut that puts every corrected gate on the 32.0 half 12.98 m/s out per
    fold and rounds multi-fold seams to the wrong integer outright, while
    still reporting a clean integer fold.  There is no version of that this
    stage may ship, so the sweep is refused and the message names the arm
    that can run it.
    """

    params = DealiasParams(engine=ENGINE_REGION_GLOBAL)
    radials = 60
    nyq = np.where(np.arange(radials) < radials // 2, 25.51, 32.0)
    folded, _truth, _ = _wind_sweep(nyq, radials=radials, gates=16)
    azimuth = np.linspace(0.0, 354.0, radials)

    with pytest.raises(DealiasParamsError, match="vad-region"):
        dealias_sweep(folded, azimuth, float(nyq.min()), params,
                      first_gate_m=2125.0, gate_spacing_m=250.0,
                      nyquist_by_radial=nyq)


@region_only
def test_region_global_refuses_a_disagree_pack_with_no_array():
    """Same refusal when the pack knows it is nonuniform and cannot say how."""

    params = DealiasParams(engine=ENGINE_REGION_GLOBAL)
    folded, _truth, _ = _wind_sweep(25.51, radials=24, gates=8)
    azimuth = np.linspace(0.0, 345.0, 24)
    with pytest.raises(DealiasParamsError, match="nyquist_radials_disagree"):
        dealias_sweep(folded, azimuth, 25.51, params,
                      first_gate_m=2125.0, gate_spacing_m=250.0,
                      nyquist_radials_disagree=True)


@region_only
def test_region_global_refuses_a_no_nyquist_radial_instead_of_borrowing():
    """A uniform sweep still runs; only the valueless radials are refused.

    The narrower case is not worth the sweep.  The crate would substitute
    the sweep median for such a radial with no reason bit and no stat; a
    gate whose Nyquist is unknown has an unknown fold state, so it is
    refused here -- and the volume account still balances.
    """

    params = DealiasParams(engine=ENGINE_REGION_GLOBAL)
    radials = 60
    nyq = np.full(radials, 25.51)
    nyq[:5] = np.nan
    folded, _truth, _ = _wind_sweep(25.51, radials=radials, gates=32)
    azimuth = np.linspace(0.0, 354.0, radials)

    result = dealias_sweep(folded, azimuth, 25.51, params,
                           first_gate_m=2125.0, gate_spacing_m=250.0,
                           nyquist_by_radial=nyq)
    assert bool(np.all(result.state[:5] == STATE_REJECTED))
    assert result.stats["nyquist_radials_no_value"] == 5
    assert result.stats["rejected"]["no_nyquist"] == int(
        np.isfinite(folded[:5]).sum())
    assert (result.stats["gates_unchanged"] + result.stats["gates_unfolded"]
            + result.stats["gates_rejected"]) == result.stats["gates_finite"]


@region_only
def test_region_global_uniform_sweep_is_untouched_by_the_per_radial_path():
    """Supplying a constant array must not change the default arm's answer."""

    params = DealiasParams(engine=ENGINE_REGION_GLOBAL)
    folded, _truth, nyq = _wind_sweep(25.51)
    azimuth = np.linspace(0.0, 358.0, folded.shape[0])
    plain = dealias_sweep(folded, azimuth, 25.51, params,
                          first_gate_m=2125.0, gate_spacing_m=250.0)
    array = dealias_sweep(folded, azimuth, 25.51, params,
                          first_gate_m=2125.0, gate_spacing_m=250.0,
                          nyquist_by_radial=nyq)
    np.testing.assert_array_equal(plain.velocity, array.velocity)
    np.testing.assert_array_equal(plain.state, array.state)
    np.testing.assert_array_equal(plain.fold, array.fold)


# --- finding 7a: the VAD volume-profile beam height --------------------------

def test_volume_profile_beam_height_uses_the_shared_geometry_authority():
    """1.21*6371 km is not a four-thirds earth radius, and it read 375 m low."""

    for range_m, expect_error_m in ((100_000.0, 40.0), (250_000.0, 250.0)):
        ours = dealias_module._beam_height_m(range_m, np.radians(0.5))
        authority, _arc, _el = beam_geometry(range_m, 0.5, 0.0)
        assert abs(ours - float(authority)) < 1e-6
        old = float(np.sqrt(range_m ** 2 + (1.21 * 6371000.0) ** 2
                            + 2.0 * range_m * 1.21 * 6371000.0
                            * np.sin(np.radians(0.5))) - 1.21 * 6371000.0)
        assert abs(ours - old) > expect_error_m, (
            "the old constant was supposed to be materially wrong here")


# --- finding 9: the positivity receipt --------------------------------------

def test_reject_receipt_reports_no_mass_left_negative():
    prior = {"qr": np.array([1.0, 1.0, 1.0])}
    increments = {"qr": np.array([-3.0, 0.5, -2.0])}

    adjusted, receipt = apply_positivity(prior, increments, policy="reject")
    assert receipt["mass_left_negative"] == 0.0
    assert receipt["positivity_semantics"] == "constrained-field-reject"
    # The quantity is not lost, only renamed to what it is.
    assert receipt["per_field"][0]["mass_that_would_have_been_added"] > 0.0
    # And the analysis really is the nonnegative background there.
    analysis = prior["qr"] + adjusted["qr"]
    assert bool(np.all(analysis >= 0.0))

    _none_adjusted, none_receipt = apply_positivity(
        prior, increments, policy="none")
    assert none_receipt["mass_left_negative"] > 0.0
    assert none_receipt["positivity_semantics"] == "none"


# --- finding 2: superob velocity and H(x) are the same observation -----------

def _one_cell_contribution(beams, speeds, grid_shape=(1, 1, 1)):
    """A single-cell accumulator holding n gates with the given look vectors."""

    from gpuwm.obs.superob import RadarContribution, SuperobCounts

    zeros = lambda: np.zeros(grid_shape, dtype=np.float64)   # noqa: E731
    vr_sum, vr_sumsq = zeros(), zeros()
    east, north, up = zeros(), zeros(), zeros()
    vr_min = np.full(grid_shape, np.inf)
    vr_max = np.full(grid_shape, -np.inf)
    for beam, speed in zip(beams, speeds):
        vr_sum[0, 0, 0] += speed
        vr_sumsq[0, 0, 0] += speed * speed
        east[0, 0, 0] += beam[0]
        north[0, 0, 0] += beam[1]
        up[0, 0, 0] += beam[2]
        vr_min[0, 0, 0] = min(vr_min[0, 0, 0], speed)
        vr_max[0, 0, 0] = max(vr_max[0, 0, 0], speed)
    return RadarContribution(
        site_id="KTST", lat_deg=35.0, lon_deg=-97.0, alt_m=300.0,
        valid_time="2026-08-13T00:00:00Z",
        z_linear_sum=zeros(), z_count=np.zeros(grid_shape, dtype=np.int64),
        z0_count=np.zeros(grid_shape, dtype=np.int64),
        z_max_dbz=np.full(grid_shape, -np.inf),
        z_sumsq_dbz=zeros(), z_sum_dbz=zeros(),
        vr_sum=vr_sum, vr_sumsq=vr_sumsq,
        vr_count=np.full(grid_shape, len(speeds), dtype=np.int64),
        vr_min=vr_min, vr_max=vr_max,
        beam_east=east, beam_north=north, beam_up=up,
        nyquist_min=np.full(grid_shape, 32.0),
        vr_rejected=np.zeros(grid_shape, dtype=np.int64),
        counts=SuperobCounts(), provenance={},
        clear_air_source="test", fold_suspicion=[])


def test_superob_velocity_and_its_operator_agree_under_a_constant_wind():
    """The linear identity: superob then H(x) must return the wind's own Vr.

    Three nonparallel beams, one constant wind, one cell.  y is the mean of
    the three scalar radial velocities; H(x) is the stored beam vector dotted
    with the wind.  They are the same observation or they are not, and with
    the normalized sum they were not: the normalization divides by
    ||sum b|| instead of by n and inflates H(x) by 1/c.
    """

    from gpuwm.obs.geometry import beam_unit_vector
    from types import SimpleNamespace

    from gpuwm.obs.superob import merge_contributions

    wind = np.array([14.0, -9.0, 1.5])          # u, v, w
    beams = [np.array(beam_unit_vector(az, el), dtype=np.float64)
             for az, el in ((10.0, 0.5), (70.0, 1.5), (130.0, 3.0))]
    speeds = [float(beam @ wind) for beam in beams]
    contribution = _one_cell_contribution(beams, speeds)

    # merge_contributions reads only the grid's shape.
    merged = merge_contributions([contribution],
                                 SimpleNamespace(nz=1, ny=1, nx=1))

    y = float(merged.vr_obs[0, 0, 0, 0])
    operator = np.array([merged.vr_beam_east[0, 0, 0, 0],
                         merged.vr_beam_north[0, 0, 0, 0],
                         merged.vr_beam_up[0, 0, 0, 0]])
    hx = float(operator @ wind)
    assert merged.vr_mask[0, 0, 0, 0] == 1
    assert abs(y - hx) < 1e-12, (
        f"y={y} and H(x)={hx} describe different observations")

    coherence = float(merged.vr_beam_coherence[0, 0, 0, 0])
    assert 0.0 < coherence < 1.0
    # The old convention is exactly this factor away, which is the size of
    # the defect rather than a vague "they disagree".
    assert abs(hx / coherence - y / coherence * 1.0) >= 0.0
    assert abs(float((operator / coherence) @ wind) - y) > 0.05 * abs(y)
