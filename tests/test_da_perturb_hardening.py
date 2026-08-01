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


def test_the_vertical_seam_correlation_is_reported_and_is_not_exp_minus_two():
    """The cap was justified by exp(-(nz/2)^2/(2 Lv^2)) ~ 0.14.

    That number is the correlation at the MAXIMUM circular separation.  On
    a periodic column the top and bottom levels are ONE interval apart, so
    the seam correlation is about exp(-1/(2 Lv^2)) -- 0.98 at the admitted
    cap, which is "locked together", not "near 0.14".
    """

    nz, scale = 24, 6.0
    report = perturb.vertical_wrap_correlations(nz, scale)
    assert report["top_to_bottom_seam"] == pytest.approx(0.983, abs=0.005)
    assert report["adjacent_interior"] == pytest.approx(0.984, abs=0.005)
    # The old claim, for contrast.
    assert report["top_to_bottom_seam"] > 5.0 * math.exp(-2.0)
    # Even the half-column value -- the quantity the cap DOES bound --
    # exceeds the Gaussian figure, because the periodic images contribute
    # covariance.  0.27 exactly (a single sampled draw measured 0.297).
    assert report["half_column"] == pytest.approx(0.270, abs=0.01)
    assert report["half_column"] > 1.9 * math.exp(-2.0)


def test_the_reported_wrap_correlation_matches_a_measured_draw():
    """Analytic, but checked against the sample it describes."""

    nz, ny, nx, scale = 24, 32, 32, 6.0
    analytic = perturb.vertical_wrap_correlations(nz, scale)
    seam, adjacent, half = [], [], []
    for seed in range(24):
        field, info = perturb.gaussian_random_field(
            (nz, ny, nx), seed=seed, name="theta", dx_km=3.0, dy_km=3.0,
            length_scale_km=6.0, vertical_scale_levels=scale, xp=np)
        assert info["vertical_wrap"] == analytic
        values = np.asarray(field, dtype=np.float64)
        flat = values.reshape(nz, -1)
        seam.append(np.corrcoef(flat[0], flat[-1])[0, 1])
        adjacent.append(np.corrcoef(flat[0], flat[1])[0, 1])
        half.append(np.corrcoef(flat[0], flat[nz // 2])[0, 1])
    assert np.mean(seam) == pytest.approx(analytic["top_to_bottom_seam"],
                                          abs=0.03)
    assert np.mean(adjacent) == pytest.approx(analytic["adjacent_interior"],
                                              abs=0.03)
    assert np.mean(half) == pytest.approx(analytic["half_column"], abs=0.06)


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
