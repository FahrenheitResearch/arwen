"""The perturbation library's false claims, replaced by measured ones.

Two of these findings are not bugs in the arithmetic -- the draw is what
the code says it is -- they are bugs in what the code SAYS about the draw.
A provenance statement that materially understates an artificial covariance
is a defect of the same kind as a wrong number, and it is fixed the same
way: by measuring, and by admitting only what the measurement supports.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gpuwm.da import perturb


def _config(**overrides):
    payload = {
        "dx_km": 3.0, "dy_km": 3.0, "rim_width": 2,
        "fields": [{"name": "theta", "amplitude": 1.0,
                    "length_scale_km": 6.0}],
        # ``_state`` carries no vertical coordinate, so the production
        # default (mass_balance = "hydrostatic") refuses it by name; the
        # balance is tested in tests/test_da_perturb.py on a column state.
        "mass_balance": "none",
    }
    payload.update(overrides)
    return perturb.PerturbationConfig.from_mapping(payload)


def _state(nz=8, ny=32, nx=32):
    import types

    f32 = np.float32
    return types.SimpleNamespace(
        thp=np.zeros((nz, ny, nx), f32),
        qv=np.full((nz, ny, nx), 8.0e-3, f32),
        u=np.zeros((nz, ny, nx + 1), f32),
        v=np.zeros((nz, ny + 1, nx), f32),
        thb=np.full((nz,), 300.0, f32),
        p=np.full((nz, ny, nx), 8.0e4, f32))


# -------------------------------------------------- F-05 vertical FFT seam


def test_the_vertical_crop_removes_the_artificial_seam_correlation():
    """A padded independent-noise domain separates the physical endpoints."""

    nz, scale = 24, 6.0
    report = perturb.vertical_wrap_correlations(nz, scale)
    assert report["top_to_bottom_seam"] == pytest.approx(
        math.exp(-(nz - 1) ** 2 / (2.0 * scale ** 2)), abs=1e-12)
    assert report["adjacent_interior"] == pytest.approx(
        math.exp(-1.0 / (2.0 * scale ** 2)), abs=1e-12)
    assert report["half_column"] == pytest.approx(math.exp(-2.0), abs=1e-12)
    assert report["fft_levels"] == 72
    assert report["periodic_seam_in_physical_column"] is False


def test_the_reported_wrap_correlation_matches_a_measured_draw():
    """Pool ensemble moments and price the horizontal sample dependence.

    Separate per-draw spatial correlation coefficients remove each draw's
    low-frequency mean and normalize by its sampled variance.  Their mean
    is not the ensemble covariance reported in provenance.  Pool raw
    moments before centering and normalization, across independent draws.
    """

    nz, ny, nx, scale = 24, 64, 64, 6.0
    draws = 64
    analytic = perturb.vertical_wrap_correlations(nz, scale)
    pairs = (("top_to_bottom_seam", nz - 1),
             ("adjacent_interior", 1), ("half_column", nz // 2))
    # First moments, second moments, cross moment for each level pair.
    totals = np.zeros((len(pairs), 5), dtype=np.float64)
    for seed in range(draws):
        field, info = perturb.gaussian_random_field(
            (nz, ny, nx), seed=seed, name="theta", dx_km=3.0, dy_km=3.0,
            length_scale_km=6.0, vertical_scale_levels=scale, xp=np)
        assert info["vertical_wrap"] == analytic
        values = np.asarray(field, dtype=np.float64)
        first = values[0]
        for index, (_, lag) in enumerate(pairs):
            second = values[lag]
            totals[index] += (first.sum(), second.sum(), (first ** 2).sum(),
                              (second ** 2).sum(), (first * second).sum())

    # For a stationary periodic Gaussian field, the variance of the pooled
    # cross moment includes sum_lag rho_h(lag)^2.  Separability gives a product
    # of these horizontal sums, not ny*nx independent spatial samples.
    squared_correlation_area = 1.0
    for extent in (ny, nx):
        modes = 2.0 * np.pi * np.fft.fftfreq(extent)
        power = np.exp(-0.5 * (modes * (6.0 / 3.0)) ** 2)
        correlation = np.fft.ifft(power).real
        correlation /= correlation[0]
        squared_correlation_area *= float(np.sum(correlation ** 2))
    effective_samples = draws * ny * nx / squared_correlation_area
    assert effective_samples > 20000
    moments = totals / (draws * ny * nx)
    for index, (key, _) in enumerate(pairs):
        mean_a, mean_b, moment_a, moment_b, cross = moments[index]
        covariance = cross - mean_a * mean_b
        variance_a = moment_a - mean_a ** 2
        variance_b = moment_b - mean_b ** 2
        measured = covariance / math.sqrt(variance_a * variance_b)
        # Gaussian correlation sampling error, including spatial dependence.
        # At zero correlation this is below the previous absolute 0.03 bound.
        tolerance = 4.0 * (1.0 - analytic[key] ** 2) / math.sqrt(effective_samples)
        assert measured == pytest.approx(analytic[key], abs=tolerance)


def test_every_perturbation_record_carries_the_wrap_figure():
    state = _state()
    provenance = perturb.apply_perturbations(state, 11, _config(fields=[
        {"name": "theta", "amplitude": 1.0, "length_scale_km": 6.0,
         "vertical_scale_levels": 2.0}]))
    record = provenance["fields"][0]
    assert record["vertical_wrap"]["top_to_bottom_seam"] > 0.0
    assert any("vertical_wrap.top_to_bottom_seam" in line
               for line in provenance["balance_not_imposed"])


def test_a_single_level_column_says_it_has_no_vertical_correlation():
    report = perturb.vertical_wrap_correlations(1, 0.0)
    assert "no vertical correlation" in report["note"]


# ------------------------------------------- F-07 horizontal admission limit


def test_the_horizontal_limit_is_where_the_documented_peak_is_resolved():
    """L <= S/(2 pi), not S/4.

    The contract is that the radial spectrum peaks at k = 1/L.  The lowest
    nonzero angular wavenumber a periodic span S carries is 2 pi / S, so
    the peak exists inside the resolved band only up to S/(2 pi) ~ 0.159 S.
    At the old quarter-span cap, probes on 32-, 64- and 128-point domains
    all measured peak * L = 2.356: the estimator was reading the
    fundamental back, not the requested scale.
    """

    assert perturb._MAX_HORIZONTAL_SPAN_FRACTION == \
        pytest.approx(1.0 / (2.0 * math.pi))
    state = _state(ny=32, nx=32)          # span 96 km, limit 15.28 km
    # Just inside.
    perturb.apply_perturbations(state, 3, _config(fields=[
        {"name": "theta", "amplitude": 1.0, "length_scale_km": 15.0}]))
    # Just outside -- and admitted by the old quarter-span rule (24 km).
    with pytest.raises(ValueError, match=r"exceeds span/\(2\*pi\)"):
        perturb.apply_perturbations(_state(ny=32, nx=32), 3, _config(fields=[
            {"name": "theta", "amplitude": 1.0, "length_scale_km": 20.0}]))


def test_the_admitted_peak_is_actually_resolved_at_the_limit():
    """At the ceiling the requested peak reaches the fundamental, not below.

    ``peak * L`` was 2.356 at the old cap.  At the new one it is 1 to
    within the discrete spectrum's own resolution, which is the whole
    point of moving it.
    """

    n = 64
    dx_km = 3.0
    span = n * dx_km
    length = perturb._MAX_HORIZONTAL_SPAN_FRACTION * span
    field, _ = perturb.gaussian_random_field(
        (1, n, n), seed=5, name="theta", dx_km=dx_km, dy_km=dx_km,
        length_scale_km=length, xp=np)
    k, power = perturb.radial_power_spectrum(np.asarray(field)[0],
                                             dx_km=dx_km, dy_km=dx_km)
    peak = perturb.spectral_peak_wavenumber(k, power)
    fundamental = 2.0 * math.pi / span
    assert peak >= 0.5 * fundamental, (
        "the peak must not fall below the domain's lowest nonzero "
        "wavenumber, which is what the old limit permitted")
    assert peak * length == pytest.approx(1.0, abs=0.6)


# --------------------------------------------------------- F-11 signed zero


def test_the_untapered_rim_is_byte_identical_even_for_negative_zero():
    """IEEE -0.0 + 0.0 is +0.0, and the state sha reads bytes.

    A probe flipped the sign bit on 1,161 of 2,064 boundary words while
    every value still compared numerically equal, so a rim the module
    promises to leave alone came back byte-different.
    """

    state = _state(nz=4, ny=24, nx=24)
    state.thp[...] = np.float32(-0.0)
    state.thp[:, 4:-4, 4:-4] = np.float32(1.0)
    taper = np.asarray(perturb.boundary_taper(24, 24, 2, xp=np))
    rim = taper == 0.0
    assert rim.any()
    before = state.thp.copy()

    perturb.apply_perturbations(state, 17, _config())

    after_bytes = state.thp[:, rim].tobytes()
    before_bytes = before[:, rim].tobytes()
    assert after_bytes == before_bytes, (
        "wherever the taper is zero the call is the identity BYTE FOR "
        "BYTE, which -0.0 is the case that tests")
    assert np.signbit(state.thp[:, rim]).all()
    # And the interior did move, so this is not a no-op test.
    assert not np.array_equal(state.thp[:, ~rim], before[:, ~rim])


def test_an_ordinary_rim_is_still_preserved_and_the_interior_moves():
    state = _state(nz=4, ny=24, nx=24)
    state.thp[...] = np.float32(2.5)
    taper = np.asarray(perturb.boundary_taper(24, 24, 2, xp=np))
    rim = taper == 0.0
    perturb.apply_perturbations(state, 23, _config())
    assert np.all(state.thp[:, rim] == np.float32(2.5))


# --------------------------------- F-12 pre-existing supersaturation basis


def test_pre_existing_supersaturation_is_counted_from_the_incoming_state():
    """Counting it from post-clamp qv reconstructed the ceiling, not the
    state that arrived, and undercounted wherever the increment was
    positive.

    The construction: a column that is ALREADY far above the cap on entry.
    Every taper-active point is a pre-existing violation, whatever the
    increment did, and the count has to say so.
    """

    state = _state(nz=4, ny=24, nx=24)
    state.qv[...] = np.float32(0.5)       # wildly supersaturated on entry
    cfg = _config(rh_cap=1.0, fields=[
        {"name": "qv", "amplitude": 1.0e-3, "length_scale_km": 6.0}])
    provenance = perturb.apply_perturbations(state, 31, cfg)
    bounds = provenance["bounds"]
    taper = np.asarray(perturb.boundary_taper(24, 24, 2, xp=np))
    active = int((taper > 0.0).sum()) * 4
    assert bounds["rh_cap_clipped_points"] == active
    assert bounds["pre_existing_supersaturated_points"] == active, (
        "every active point was already over the cap before this call; "
        "reconstructing the entry state from the CLIPPED qv would have "
        "counted only the points whose increment happened to be negative")
    assert "snapshotted before either clamp" in bounds["pre_existing_basis"]


def test_a_dry_state_reports_no_pre_existing_supersaturation():
    state = _state(nz=4, ny=24, nx=24)
    state.qv[...] = np.float32(1.0e-4)
    cfg = _config(rh_cap=1.0, fields=[
        {"name": "qv", "amplitude": 1.0e-6, "length_scale_km": 6.0}])
    provenance = perturb.apply_perturbations(state, 37, cfg)
    assert provenance["bounds"]["pre_existing_supersaturated_points"] == 0


# ------------------------------------------------- the background's own pairs


def _species_state(nz=6, ny=24, nx=24, mass=5.0e-4, number=0.0):
    """A cold-start background: hydrometeor mass, no number concentration."""

    import types

    f32 = np.float32
    state = _state(nz, ny, nx)
    state.qc = np.full((nz, ny, nx), mass, f32)
    state.nc = np.full((nz, ny, nx), number, f32)
    return state


def test_a_pair_the_background_carried_is_reported_and_not_refused():
    """A cold start has mass with no number; the scheme closes it, not this.

    The invariant this module owns is that its own strictly positive
    common factor CREATES no depleted pair.  Counting the post-state
    absolutely charged it with every pair the background arrived with,
    and a real cold-start forecast was refused after its state had
    already been perturbed.
    """

    state = _species_state()
    cfg = _config(species=[{"mass_field": "qc", "amplitude": 0.4,
                            "length_scale_km": 6.0,
                            "threshold_kg_kg": 1.0e-8}])
    report = perturb.apply_perturbations(state, 20260911, cfg)
    record = report["species"][0]
    assert record["species"] == "qc"
    assert record["depleted_pairs_created"] == 0
    assert record["negative_points"] == 0
    assert record["depleted_pairs_in_background"] == state.qc.size
    # Nothing was active, so the background came through untouched.
    assert record["active_points"] == 0
    assert np.array_equal(state.nc, np.zeros_like(state.nc))


class _EmptiedOnWrite(np.ndarray):
    """An array that comes out of its own assignment empty.

    Fault injection for the one case the refusal is written for: a field
    that does not hold the factor it was given.  Nothing in the package
    behaves this way; that is the point of injecting it.
    """

    def __setitem__(self, key, value):
        super().__setitem__(key, np.zeros_like(np.asarray(value)))


def test_a_pair_this_call_would_create_is_still_refused():
    """The refusal survives for what it was written for."""

    state = _species_state(number=2.0e8)
    state.nc = state.nc.view(_EmptiedOnWrite)
    cfg = _config(species=[{"mass_field": "qc", "amplitude": 0.4,
                            "length_scale_km": 6.0,
                            "threshold_kg_kg": 1.0e-8}])
    with pytest.raises(ValueError, match="CREATED"):
        perturb.apply_perturbations(state, 20260911, cfg)


# ------------------------------------------------ the lognormal factor's mean


def test_clipped_lognormal_log_mean_matches_a_monte_carlo_draw():
    """``exp(a clip(g) - c)`` has mean 1; ``c`` is what the helper returns.

    At the storm-scale amplitude 0.7 the uncorrected factor's mean is
    1.27 (analytic, 2.5-sigma clip), which was the condensate the
    ensemble mean gained before any observation on the 2024-05-21 crop
    (module draw: 1.262).  Pure numpy, no field structure: the helper's
    arithmetic alone.
    """

    rng = np.random.default_rng(20261006)
    g = rng.standard_normal(2_000_000)
    for amplitude in (0.7, 0.3, 0.05):
        c = perturb.clipped_lognormal_log_mean(amplitude, 2.5)
        raw = np.exp(amplitude * np.clip(g, -2.5, 2.5))
        assert abs(raw.mean() - math.exp(c)) < 3.0e-3 * math.exp(c)
        corrected = raw * math.exp(-c)
        assert abs(corrected.mean() - 1.0) < 3.0e-3
    assert abs(math.exp(perturb.clipped_lognormal_log_mean(0.7, 2.5))
               - 1.26765) < 1.0e-4
    assert perturb.clipped_lognormal_log_mean(0.0, 2.5) == 0.0
    # An array of amplitudes (a tapered rim) is evaluated per element.
    table = perturb.clipped_lognormal_log_mean(
        np.array([[0.0, 0.35], [0.7, 0.0]]), 2.5)
    assert table.shape == (2, 2)
    assert table[0, 0] == 0.0 and table[1, 1] == 0.0
    assert abs(table[1, 0] - perturb.clipped_lognormal_log_mean(0.7, 2.5)) == 0.0


def _members_mean_factor(build_state, seed_count, cfg, field):
    """Ensemble mean of ``field / background`` over the untapered interior."""

    total = None
    for seed in range(seed_count):
        state = build_state()
        background = np.array(getattr(state, field), copy=True)
        perturb.apply_perturbations(state, 1000 + seed, cfg)
        ratio = np.asarray(getattr(state, field), np.float64) / background
        total = ratio if total is None else total + ratio
    mean = total / seed_count
    # rim_width 2 with the default taper: the interior is untapered.
    return mean[:, 6:-6, 6:-6]


def test_lognormal_species_perturbation_keeps_the_ensemble_mean_mass():
    """24 members at sigma 0.7: the mean mass is the background's, not 1.27x.

    Species factors multiply mass and number by the same number, so the
    mean of the NUMBER is checked too.  The tolerance is sampling noise
    (24 members on a few hundred 6 km blobs), far below the 27% the
    uncorrected factor carried.
    """

    def build():
        state = _species_state(nz=4, ny=40, nx=40, mass=5.0e-4, number=2.0e8)
        return state

    cfg = _config(species=[{"mass_field": "qc", "amplitude": 0.7,
                            "length_scale_km": 6.0, "clip_sigmas": 2.5,
                            "threshold_kg_kg": 1.0e-8}])
    for field in ("qc", "nc"):
        mean = _members_mean_factor(build, 24, cfg, field)
        assert abs(float(mean.mean()) - 1.0) < 0.06, float(mean.mean())
    record = perturb.apply_perturbations(build(), 7, cfg)["species"][0]
    assert record["mean_preserving"] is True
    assert abs(record["mean_log_correction_max"]
               - perturb.clipped_lognormal_log_mean(0.7, 2.5)) < 1.0e-12


def test_lognormal_vapour_perturbation_keeps_the_ensemble_mean():
    """The same correction on the lognormal ``qv`` field."""

    def build():
        state = _state(nz=4, ny=40, nx=40)
        state.qv[...] = np.float32(1.0e-4)
        return state

    cfg = _config(rh_cap=1.0, fields=[
        {"name": "qv", "amplitude": 0.7, "length_scale_km": 6.0,
         "mode": "lognormal", "clip_sigmas": 2.5}])
    mean = _members_mean_factor(build, 24, cfg, "qv")
    assert abs(float(mean.mean()) - 1.0) < 0.06, float(mean.mean())
