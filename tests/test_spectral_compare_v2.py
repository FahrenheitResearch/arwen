"""Failure controls for the additive scale-resolved comparison class."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gpuwm.verify import spectral_compare

COMMITTED_V2_PINS_SHA256 = (
    "51517e31336c7b415efe4cc969f62f740b003d95bc2ad14ea88f54f7849b0f5c")

DX = 1000.0
N = 128
BANDS = (
    spectral_compare.WavelengthBand("small", 2.0, 8.0),
    spectral_compare.WavelengthBand("mesoscale", 8.0, 32.0),
    spectral_compare.WavelengthBand("large", 32.0, None),
)


def wave(cycles: int, *, axis: str = "x", phase: float = 0.0) -> np.ndarray:
    coordinate = np.arange(N, dtype=np.float64)
    one_d = np.cos(2.0 * np.pi * cycles * coordinate / N + phase)
    return (np.tile(one_d, (N, 1)) if axis == "x"
            else np.tile(one_d[:, None], (1, N)))


def row(result: dict, name: str, component: str = "scalar") -> dict:
    for item in spectral_compare.metric_rows(result):
        if item["band"] == name and item["component"] == component:
            return item
    raise AssertionError((name, component))


def test_implementation_pins_are_hash_bound_and_v1_is_not_reused():
    registration = spectral_compare.implementation_registration()
    assert (registration["pins_sha256"]
            == spectral_compare.IMPLEMENTATION_PINS_SHA256
            == COMMITTED_V2_PINS_SHA256)
    assert spectral_compare._canonical_hash(registration["pins"]) == (
        spectral_compare.IMPLEMENTATION_PINS_SHA256)
    assert "additive" in registration["pins"]["relationship_to_v1"]


def test_bands_refuse_overlap_and_duplicate_names():
    with pytest.raises(ValueError, match="overlap"):
        spectral_compare.parse_bands([
            {"name": "a", "minimum_km": 10.0, "maximum_km": 30.0},
            {"name": "b", "minimum_km": 20.0, "maximum_km": 40.0},
        ])
    with pytest.raises(ValueError, match="unique"):
        spectral_compare.parse_bands([
            {"name": "a", "minimum_km": 10.0, "maximum_km": 20.0},
            {"name": "a", "minimum_km": 20.0, "maximum_km": 30.0},
        ])


def test_identical_scalar_is_exactly_zero_error_and_unit_agreement():
    field = wave(8) + 0.25 * wave(16, axis="y")
    result = spectral_compare.compare_scalar(
        field, field, dx_m=DX, bands=BANDS)
    target = row(result, "mesoscale")
    assert target["power_ratio"] == pytest.approx(1.0, abs=1e-14)
    assert target["amplitude_ratio"] == pytest.approx(1.0, abs=1e-14)
    assert target["error_power"] == 0.0
    assert target["phase_location_error_power"] == 0.0
    assert target["spectral_correlation"] == pytest.approx(1.0, abs=1e-14)
    assert target["coherence_squared"] == pytest.approx(1.0, abs=1e-14)
    assert target["weighted_mean_absolute_phase_error_degrees"] == pytest.approx(
        0.0, abs=1e-12)
    assert result["left_parseval"]["relative_error"] < 1e-12
    assert result["reference_parseval"]["relative_error"] < 1e-12


def test_amplitude_only_change_is_not_misdiagnosed_as_phase_error():
    reference = wave(8)
    candidate = 2.0 * reference
    result = spectral_compare.compare_scalar(
        candidate, reference, dx_m=DX, bands=BANDS)
    target = row(result, "mesoscale")
    assert target["power_ratio"] == pytest.approx(4.0, rel=1e-12)
    assert target["amplitude_ratio"] == pytest.approx(2.0, rel=1e-12)
    assert target["spectral_correlation"] == pytest.approx(1.0, abs=1e-12)
    assert target["normalized_error_power"] == pytest.approx(1.0, rel=1e-12)
    assert target["phase_location_error_power"] == pytest.approx(0.0, abs=1e-12)
    assert target["weighted_mean_absolute_phase_error_degrees"] == pytest.approx(
        0.0, abs=1e-10)
    assert target["amplitude_mismatch_power"] == pytest.approx(
        target["error_power"], rel=1e-12)


def test_sign_reversal_has_equal_power_but_maximal_signed_phase_error():
    reference = wave(8)
    candidate = -reference
    result = spectral_compare.compare_scalar(
        candidate, reference, dx_m=DX, bands=BANDS)
    target = row(result, "mesoscale")
    assert target["power_ratio"] == pytest.approx(1.0, rel=1e-12)
    assert target["spectral_correlation"] == pytest.approx(-1.0, abs=1e-12)
    assert target["coherence_squared"] == pytest.approx(1.0, abs=1e-12)
    assert target["weighted_mean_absolute_phase_error_degrees"] == pytest.approx(
        180.0, abs=1e-10)
    assert target["normalized_error_power"] == pytest.approx(4.0, rel=1e-12)
    assert target["amplitude_mismatch_power"] == pytest.approx(0.0, abs=1e-12)
    assert target["phase_location_fraction_of_error"] == pytest.approx(
        1.0, abs=1e-12)


def test_added_grid_scale_structure_lands_in_the_short_wavelength_band():
    reference = wave(8)
    candidate = reference + 3.0 * wave(32)
    result = spectral_compare.compare_scalar(
        candidate, reference, dx_m=DX, bands=BANDS)
    short = row(result, "small")
    middle = row(result, "mesoscale")
    large = row(result, "large")
    assert short["error_power"] > 25.0 * middle["error_power"]
    assert short["error_power"] > 100.0 * large["error_power"]
    assert middle["spectral_correlation"] > 0.99


def test_crop_changes_scored_shape_and_refuses_an_erased_domain():
    field = wave(8)
    result = spectral_compare.compare_scalar(
        field, field, dx_m=DX, bands=BANDS, crop_cells=4)
    assert result["input_shape"] == [N, N]
    assert result["scored_shape"] == [N - 8, N - 8]
    with pytest.raises(ValueError, match="entire"):
        spectral_compare.compare_scalar(
            np.zeros((16, 16)), np.zeros((16, 16)), dx_m=DX,
            bands=BANDS, crop_cells=8)


def test_pure_divergent_and_pure_rotational_winds_separate():
    # A wave vector along x with velocity parallel to x is divergent; the
    # same wave with velocity along y is rotational.
    parallel = wave(8)
    zero = np.zeros_like(parallel)
    divergent = spectral_compare.compare_vector(
        parallel, zero, parallel, zero, dx_m=DX, bands=BANDS)
    rotational = spectral_compare.compare_vector(
        zero, parallel, zero, parallel, dx_m=DX, bands=BANDS)
    div_band = next(item for item in divergent["bands"]
                    if item["name"] == "mesoscale")
    rot_band = next(item for item in rotational["bands"]
                    if item["name"] == "mesoscale")
    assert div_band["left_divergent_energy_fraction"] > 0.99
    assert rot_band["left_divergent_energy_fraction"] < 0.01
    assert divergent["helmholtz_mode_closure"]["maximum_relative_error"] < 1e-12
    assert rotational["helmholtz_mode_closure"]["maximum_relative_error"] < 1e-12
    assert divergent["left_parseval"]["relative_error"] < 1e-12


def test_vector_amplitude_change_is_visible_in_every_relevant_component():
    u = wave(8)
    v = 0.5 * wave(8, axis="y")
    result = spectral_compare.compare_vector(
        1.5 * u, 1.5 * v, u, v, dx_m=DX, bands=BANDS)
    total = row(result, "mesoscale", "total")
    assert total["power_ratio"] == pytest.approx(2.25, rel=1e-12)
    assert total["amplitude_ratio"] == pytest.approx(1.5, rel=1e-12)
    assert total["spectral_correlation"] == pytest.approx(1.0, abs=1e-12)


def test_anisotropic_spacing_uses_the_smaller_physical_nyquist():
    geometry = spectral_compare.mode_geometry(
        64, 64, dx_m=1000.0, dy_m=2000.0)
    assert geometry["isotropic_nyquist_cycles_per_m"] == pytest.approx(
        1.0 / 4000.0)
    assert geometry["valid_mode_count"] > 0


def test_nonfinite_and_mismatched_inputs_fail_closed():
    field = wave(8)
    with pytest.raises(ValueError, match="share one shape"):
        spectral_compare.compare_scalar(
            field, field[:64], dx_m=DX, bands=BANDS)
    bad = field.copy()
    bad[0, 0] = math.nan
    with pytest.raises(ValueError, match="non-finite"):
        spectral_compare.compare_scalar(bad, field, dx_m=DX, bands=BANDS)
