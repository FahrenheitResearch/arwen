"""The spectral class recovers a known spectrum, and separates two.

Controls, named:

* **monotonicity** -- THIS package's failure control.  The distance between
  two fields of analytically different spectra strictly exceeds the distance
  between two identical fields, and grows with the separation between the two
  spectra.  A distance that recovered a spectrum perfectly and still could not
  tell two apart would pass every recovery test in this file and measure
  nothing;
* **variance-matched separation** -- two fields normalized to the same mean
  and the same variance, differing only in how that variance is distributed
  across scales, still separate, and by more than two fields of the same
  spectrum with different phases.  A distance that had quietly become an RMSE
  would report zero here;
* **pin tamper** -- the pins' hash is committed against a literal below, and
  the envelope registration's hash covers the spectral pins, so an edited
  window or bin rule fails rather than silently re-defining what an already
  published number measured;
* **field-selection refusal** -- a registration that scores its reflectivity
  on some other variable cannot carry the spectral class.

The synthetic-spectrum recovery tests are positive controls and are not
counted as this package's failure control.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from gpuwm.verify import chaos_envelope, spectral

#: The pins' hash, committed here before any receipt records one.  If this
#: literal and the module disagree, a pin moved: decide the change, re-measure
#: anything already published under the old pins, and only then re-cut this.
COMMITTED_PINS_SHA256 = (
    "f3d1d17f80742c7114be8984cdfb55c10db117ecaa6e926b3ccc27414834062e")

DX = 1000.0


def exact_power_law(n: int, beta: float, seed: int, dx: float = DX
                    ) -> np.ndarray:
    """A real field whose 2-D periodogram is exactly ``|k|**-beta``.

    Random phases alone, made antisymmetric under ``k -> -k`` so the inverse
    transform is real and every Fourier magnitude is exactly the power law.
    A field built from random *amplitudes* would carry chi-square scatter and
    a recovery tolerance would then be measuring the realization rather than
    the estimator.
    """
    ky = np.fft.fftfreq(n, d=dx)
    kx = np.fft.fftfreq(n, d=dx)
    magnitude = np.hypot(ky[:, None], kx[None, :])
    amplitude = np.zeros_like(magnitude)
    nonzero = magnitude > 0.0
    amplitude[nonzero] = magnitude[nonzero] ** (-beta / 2.0)
    raw = np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, (n, n))
    mirrored = raw[(-np.arange(n)) % n][:, (-np.arange(n)) % n]
    field = np.fft.ifft2(amplitude * np.exp(0.5j * (raw - mirrored)))
    assert np.max(np.abs(field.imag)) < 1e-9 * np.max(np.abs(field.real))
    return np.asarray(field.real, dtype=np.float64)


def standardized(field: np.ndarray) -> np.ndarray:
    centred = field - np.mean(field)
    return centred / np.std(centred)


# --------------------------------------------------------------------------
# the pins
# --------------------------------------------------------------------------


def test_the_pins_hash_matches_the_committed_literal():
    assert spectral.PINS_SHA256 == COMMITTED_PINS_SHA256
    assert spectral.registration()["pins_sha256"] == COMMITTED_PINS_SHA256


def test_every_argued_choice_is_pinned():
    for key in ("fields", "detrend", "window", "periodogram",
                "radial_bin_width", "radial_bin_assignment", "retained_band",
                "log_floor", "distance"):
        assert key in spectral.PINS, key
    assert [pin["variable"] for pin in spectral.FIELD_PINS] == ["W", "REFL_10CM"]
    assert spectral.pinned_variable("composite_reflectivity") == "REFL_10CM"
    assert spectral.pinned_variable("vertical_velocity") == "W"
    with pytest.raises(ValueError, match="no spectral field is pinned"):
        spectral.pinned_variable("humidity")


def test_a_moved_pin_moves_the_hash():
    """CONTROL: the hash is what makes a post-hoc pin edit visible."""
    tampered = json.loads(json.dumps(spectral.PINS))
    tampered["window"] = "no window"
    assert spectral._canonical_hash(tampered) != spectral.PINS_SHA256


# --------------------------------------------------------------------------
# detrend, window, binning
# --------------------------------------------------------------------------


def test_the_detrend_removes_a_plane_exactly():
    y, x = np.indices((12, 17))
    plane = 3.5 - 0.25 * x + 0.75 * y
    assert np.max(np.abs(spectral.detrend_plane(plane))) < 1e-9
    residual = spectral.detrend_plane(plane + np.cos(x))
    assert np.max(np.abs(residual)) > 0.1


def test_the_window_has_unit_mean_square():
    window = spectral.hann_window(24, 32)
    assert math.isclose(float(np.mean(window * window)), 1.0, rel_tol=1e-12)


def test_the_band_drops_dc_and_stops_at_the_isotropic_nyquist():
    bins = spectral.radial_bins(64, 64, DX)
    assert bins["first"] == 1
    wavenumbers = bins["wavenumbers"]
    assert float(wavenumbers[0]) > 0.0
    assert float(wavenumbers[-1]) <= 0.5 / DX
    assert float(wavenumbers[-1] + bins["bin_width"]) > 0.5 / DX
    assert np.all(np.asarray(bins["counts"]) > 0)


def test_a_mode_on_a_bin_edge_lands_where_its_integers_say():
    """CONTROL: the bin rule is arithmetic, not a rounding of 1/(n*dx).

    A Pythagorean mode -- (8, 15) on a 64-wide grid, magnitude exactly 17
    bin widths -- sits on a bin edge.  Forming ``|k|/dk`` from the physical
    frequencies rounds that ratio to just under 17 and files the mode one bin
    low, which moves power between two bins and depends on how the transform
    library happened to build its frequency table.
    """
    index = np.asarray(spectral.radial_bins(64, 64, DX)["index"])
    assert int(index[8, 15]) == 17
    assert int(index[15, 8]) == 17
    assert int(index[16, 30]) == 34
    for m in range(1, 32):
        assert int(index[0, m]) == m
        assert int(index[m, 0]) == m


def test_the_binning_does_not_move_with_the_grid_spacing():
    reference = np.asarray(spectral.radial_bins(14, 16, 250.0)["counts"])
    for dx in (1.0, 3.7, 1000.0, 12000.0, 1000.0 / 3.0):
        assert np.array_equal(
            np.asarray(spectral.radial_bins(14, 16, dx)["counts"]), reference)


def test_a_plane_too_small_to_bin_is_refused():
    with pytest.raises(ValueError, match="too small to bin radially"):
        spectral.scored_plane(np.zeros((3, 8)), "column_max")
    with pytest.raises(ValueError, match="fewer than the pinned minimum"):
        spectral.radial_bins(8, 8, DX)
    with pytest.raises(ValueError, match="unpinned plane reduction"):
        spectral.scored_plane(np.zeros((8, 8)), "column_mean")
    with pytest.raises(ValueError, match="empty/non-finite"):
        spectral.scored_plane(np.full((8, 8), np.nan), "column_max")


def test_the_periodogram_integrates_to_the_windowed_variance():
    """The pinned normalization is the one the pins state."""
    field = exact_power_law(64, 5.0 / 3.0, 3)
    windowed = spectral.detrend_plane(field) * spectral.hann_window(64, 64)
    power = (DX * DX / (64 * 64)) * np.abs(np.fft.fft2(windowed)) ** 2
    cell = (1.0 / (64 * DX)) ** 2
    assert math.isclose(float(np.sum(power) * cell),
                        float(np.mean(windowed * windowed)), rel_tol=1e-9)


def test_the_two_plane_reductions_are_what_they_say():
    field = np.zeros((3, 8, 9))
    field[1, 2, 3] = -7.0
    field[2, 2, 3] = 4.0
    assert spectral.scored_plane(field, "column_max")[2, 3] == 4.0
    assert spectral.scored_plane(field, "column_max_abs")[2, 3] == 7.0
    plane = np.arange(72, dtype=np.float64).reshape(8, 9)
    assert np.array_equal(spectral.scored_plane(plane, "column_max"), plane)


# --------------------------------------------------------------------------
# recovery (positive controls)
# --------------------------------------------------------------------------

#: Stated tolerance for the recovered log-log slope, in decades of power per
#: decade of wavenumber.  Measured worst case over eight phase realizations
#: and three slopes is 0.076; this is that, doubled.
SLOPE_TOLERANCE = 0.15


@pytest.mark.parametrize("beta", [5.0 / 3.0, 2.0, 3.0])
def test_a_known_power_law_comes_back_out(beta: float):
    psd = spectral.radial_psd(exact_power_law(256, beta, 0), DX)
    count = int(np.asarray(psd["power"]).size)
    slope = spectral.log_log_slope(psd, bins=range(count // 8, count // 2))
    assert abs(slope + beta) <= SLOPE_TOLERANCE, (beta, slope)


def test_a_single_wavenumber_lands_in_its_own_bin():
    """A wave of ``cycles`` per domain width belongs to radial bin ``cycles``."""
    n, dx = 64, 500.0
    bin_width = 1.0 / (n * dx)
    for cycles in (4, 8, 13):
        column = np.cos(2.0 * np.pi * cycles * np.arange(n) / n)
        psd = spectral.radial_psd(np.tile(column, (n, 1)), dx)
        peak_bin = int(np.argmax(np.asarray(psd["power"]))) + 1
        assert peak_bin == cycles
        centre = float(np.asarray(psd["wavenumber_cycles_per_m"])[peak_bin - 1])
        assert centre == pytest.approx((cycles + 0.5) * bin_width)
        assert float(psd["bin_width_cycles_per_m"]) == pytest.approx(bin_width)


# --------------------------------------------------------------------------
# monotonicity (THE failure control)
# --------------------------------------------------------------------------


def test_different_spectra_strictly_exceed_identical_ones():
    """CONTROL: the distance separates what the recovery test cannot."""
    reference = exact_power_law(128, 5.0 / 3.0, 5)
    assert spectral.log_spectral_distance(
        reference, reference, dx_m=DX) == 0.0
    other = exact_power_law(128, 3.0, 5)
    assert spectral.log_spectral_distance(reference, other, dx_m=DX) > 0.0


def test_the_distance_grows_with_the_separation_between_spectra():
    """CONTROL: monotone in the slope difference, not merely non-zero."""
    reference = exact_power_law(128, 5.0 / 3.0, 5)
    previous = spectral.log_spectral_distance(reference, reference, dx_m=DX)
    for step in (0.25, 0.5, 1.0, 2.0):
        other = exact_power_law(128, 5.0 / 3.0 + step, 5)
        distance = spectral.log_spectral_distance(reference, other, dx_m=DX)
        assert distance > previous, (step, distance, previous)
        previous = distance


def test_a_variance_matched_pair_still_separates():
    """CONTROL: a distance that had become an RMSE would report zero here."""
    reference = standardized(exact_power_law(128, 5.0 / 3.0, 5))
    steeper = standardized(exact_power_law(128, 3.0, 5))
    rephased = standardized(exact_power_law(128, 5.0 / 3.0, 6))
    assert math.isclose(float(np.std(reference)), float(np.std(steeper)),
                        rel_tol=1e-12)
    slope_change = spectral.log_spectral_distance(reference, steeper, dx_m=DX)
    phase_change = spectral.log_spectral_distance(reference, rephased, dx_m=DX)
    assert slope_change > phase_change > 0.0


def test_the_distance_is_symmetric_and_non_negative():
    left = exact_power_law(64, 5.0 / 3.0, 1)
    right = exact_power_law(64, 2.5, 2)
    forward = spectral.log_spectral_distance(left, right, dx_m=DX)
    backward = spectral.log_spectral_distance(right, left, dx_m=DX)
    assert forward == backward > 0.0


def test_mismatched_planes_and_bad_spacings_are_refused():
    left = exact_power_law(64, 2.0, 1)
    with pytest.raises(ValueError, match="share one plane shape"):
        spectral.log_spectral_distance(left, left[:32], dx_m=DX)
    with pytest.raises(ValueError, match="grid spacing must be positive"):
        spectral.log_spectral_distance(left, left, dx_m=0.0)
    with pytest.raises(ValueError, match="both spectra are empty"):
        spectral.log_spectral_distance(
            np.zeros((32, 32)), np.zeros((32, 32)), dx_m=DX)


def test_the_pinned_pair_scores_both_fields():
    left = {"W": np.stack([exact_power_law(64, 2.0, 1)] * 3),
            "REFL_10CM": np.stack([exact_power_law(64, 2.0, 2)] * 3)}
    right = {"W": np.stack([exact_power_law(64, 3.0, 1)] * 3),
             "REFL_10CM": np.stack([exact_power_law(64, 2.0, 2)] * 3)}
    distances = spectral.pair_distances(left, right, dx_m=DX)
    assert set(distances) == {"W", "REFL_10CM"}
    assert distances["W"] > 0.0
    assert distances["REFL_10CM"] == 0.0
    with pytest.raises(ValueError, match="require REFL_10CM on both sides"):
        spectral.pair_distances({"W": left["W"]}, {"W": right["W"]}, dx_m=DX)


# --------------------------------------------------------------------------
# the envelope's half of the wiring
# --------------------------------------------------------------------------


def test_the_registration_hash_covers_the_spectral_pins():
    """CONTROL: an edited window invalidates a registration, not a receipt."""
    registration = chaos_envelope.make_registration(
        start_time="2001-02-03T04:00:00", domain_dx_m={"d01": 12000.0},
        state_fields=("T",), leads_seconds=(600,), cadence_seconds=600,
        reflectivity_field="REFL_10CM", reflectivity_threshold=40.0,
        low_pass_physical_width_m=6000.0,
        low_pass_interior_exclusion_cells=1, boundary_width_cells=2,
        fss_radius_m=5000.0, object_min_area_km2=25.0, object_connectivity=8,
        evaluator_commit="b" * 40)
    parameters = registration["parameters"]
    assert parameters["spectral_pins_sha256"] == spectral.PINS_SHA256
    assert parameters["spectral"]["fields"][0]["variable"] == "W"
    assert parameters["unscheduled_metric_classes"][0]["metric_class"] == (
        "distributional")
    tampered = json.loads(json.dumps(registration))
    tampered["parameters"]["spectral"]["window"] = "no window"
    with pytest.raises(ValueError, match="hash does not match"):
        chaos_envelope.validate_registration(tampered)


def test_a_registration_on_another_reflectivity_cannot_carry_the_class():
    """CONTROL: the pinned field selection is not substitutable."""
    with pytest.raises(ValueError, match="cannot carry the spectral class"):
        chaos_envelope.require_spectral_agreement("REFLECTIVITY")
    with pytest.raises(ValueError, match="cannot carry the spectral class"):
        chaos_envelope.make_registration(
            start_time="2001-02-03T04:00:00", domain_dx_m={"d01": 12000.0},
            state_fields=("T",), leads_seconds=(600,), cadence_seconds=600,
            reflectivity_field="REFLECTIVITY", reflectivity_threshold=40.0,
            low_pass_physical_width_m=6000.0,
            low_pass_interior_exclusion_cells=1, boundary_width_cells=2,
            fss_radius_m=5000.0, object_min_area_km2=25.0,
            object_connectivity=8, evaluator_commit="b" * 40)


def test_the_spectral_class_is_a_metric_class_and_the_gap_is_named():
    assert "spectral_log_distance" in chaos_envelope.METRIC_CLASSES
    assert chaos_envelope.split_key(
        "spectral_log_distance:W|d02|1200") == ("spectral_log_distance",
                                                "d02", 1200)
    unscheduled = chaos_envelope.UNSCHEDULED_METRIC_CLASSES
    assert [entry["metric_class"] for entry in unscheduled] == ["distributional"]
    assert all(entry["status"] == "unscheduled" for entry in unscheduled)
