"""Scale-resolved model-to-reference comparison for limited-area fields.

This module is deliberately additive to :mod:`gpuwm.verify.spectral`.
The existing ``gpuwm.spectral-pins/v1`` metric is a frozen chaos-envelope
class for column-maximum W and reflectivity.  It must not be widened or
re-pinned to serve a different question.  This module answers the broader
model-verification question instead:

* how much variance/kinetic energy each run carries in physical wavelength
  bands;
* whether the amplitudes agree;
* whether the structures are in phase/in the same places;
* how much error remains after amplitude mismatch is removed; and
* for horizontal wind, whether that energy is rotational or divergent.

The transform is a two-dimensional Fourier analysis of a detrended,
unit-mean-square Hann-windowed regional plane.  Only the circular disk through
the smaller axis Nyquist is admitted, so every retained radial wavenumber has
complete directional support.  The DC mode is excluded.  All reductions and
sums are float64.

No empirical pass threshold lives here.  A campaign pre-registers wavelength
bands and any gates in a receipt registration; this module only supplies the
pinned arithmetic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Iterable, Mapping, Sequence

import numpy as np

COMPARISON_SCHEMA = "gpuwm.spectral-comparison/v1"
IMPLEMENTATION_PINS_SCHEMA = "gpuwm.spectral-comparison-pins/v1"


@dataclass(frozen=True)
class WavelengthBand:
    """One half-open physical wavelength interval, in kilometres.

    ``minimum_km`` is inclusive.  ``maximum_km`` is exclusive.  ``None``
    leaves that side open.  The zero-wavenumber/infinite-wavelength mode is
    always excluded before bands are applied.
    """

    name: str
    minimum_km: float | None = None
    maximum_km: float | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("a wavelength band needs a non-empty name")
        for label, value in (("minimum", self.minimum_km),
                             ("maximum", self.maximum_km)):
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise ValueError(
                    f"wavelength-band {label} must be positive and finite")
        if (self.minimum_km is not None and self.maximum_km is not None
                and self.minimum_km >= self.maximum_km):
            raise ValueError(
                "wavelength-band minimum must be below its maximum")


IMPLEMENTATION_PINS: dict[str, object] = {
    "schema": IMPLEMENTATION_PINS_SCHEMA,
    "relationship_to_v1": (
        "additive: gpuwm.verify.spectral and its PINS_SHA256 are unchanged; "
        "this class is registered and receipted independently"),
    "input_geometry": (
        "2-D regional planes on one common rectilinear grid; horizontal crop "
        "is applied before preprocessing"),
    "detrend": (
        "least-squares plane a+b*x+c*y removed using zero-centred integer "
        "cell coordinates and float64 projections"),
    "window": (
        "separable symmetric Hann over both axes, normalized so its 2-D mean "
        "square is exactly one to rounding"),
    "fft": "numpy-compatible unnormalized complex fft2 in float64/complex128",
    "mode_power": (
        "|F|^2/N^2, where N=ny*nx, so summing all modes equals the spatial "
        "mean square by Parseval"),
    "kinetic_energy": (
        "0.5*(|Fu|^2+|Fv|^2)/N^2; Helmholtz projections use the physical "
        "kx,ky unit vector at every nonzero mode"),
    "retained_modes": (
        "k>0 and k<=min(1/(2*dx),1/(2*dy)); this is the complete isotropic "
        "disk through the smaller axis Nyquist"),
    "wavelength_band_membership": (
        "minimum wavelength inclusive, maximum wavelength exclusive; bands "
        "may leave gaps but may not overlap"),
    "cross_spectrum": "sum(F_left*conj(F_reference))/N^2 within each band",
    "spectral_correlation": (
        "real(cross)/sqrt(P_left*P_reference), signed and bounded [-1,1]"),
    "coherence_squared": (
        "|cross|^2/(P_left*P_reference), bounded [0,1]"),
    "weighted_mean_absolute_phase_error_degrees": (
        "sum(|cross_mode|*abs(arg(cross_mode)))/sum(|cross_mode|) within "
        "each band, in [0,180] degrees; unlike the aggregate complex phase, "
        "the absolute per-mode phase does not cancel across conjugate pairs"),
    "error_power": "sum(|F_left-F_reference|^2)/N^2 within each band",
    "amplitude_mismatch_power": (
        "(sqrt(P_left)-sqrt(P_reference))^2, the minimum error possible if "
        "band phase/location were perfectly aligned"),
    "phase_location_error_power": (
        "error_power-amplitude_mismatch_power, clipped only for roundoff"),
    "precision": "float64 inputs/reductions; complex128 FFT/cross sums",
    "missing_nonfinite_or_zero_reference_policy": (
        "fail on missing/nonfinite input; a band with no admitted modes is "
        "unresolved; ratios requiring zero reference power are null"),
}


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


IMPLEMENTATION_PINS_SHA256 = _canonical_hash(IMPLEMENTATION_PINS)


def implementation_registration() -> dict[str, object]:
    """Deep-copyable implementation pins and their identity."""

    return {
        "pins": json.loads(json.dumps(IMPLEMENTATION_PINS)),
        "pins_sha256": IMPLEMENTATION_PINS_SHA256,
    }


def parse_bands(records: Iterable[Mapping[str, object]]) -> tuple[WavelengthBand, ...]:
    """Build and validate named, non-overlapping wavelength bands."""

    bands = tuple(
        WavelengthBand(
            name=str(record["name"]),
            minimum_km=(None if record.get("minimum_km") is None
                        else float(record["minimum_km"])),
            maximum_km=(None if record.get("maximum_km") is None
                        else float(record["maximum_km"])),
        )
        for record in records
    )
    if not bands:
        raise ValueError("spectral comparison needs at least one wavelength band")
    names = [band.name for band in bands]
    if len(names) != len(set(names)):
        raise ValueError("wavelength-band names must be unique")
    for index, left in enumerate(bands):
        left_lo = 0.0 if left.minimum_km is None else left.minimum_km
        left_hi = math.inf if left.maximum_km is None else left.maximum_km
        for right in bands[index + 1:]:
            right_lo = 0.0 if right.minimum_km is None else right.minimum_km
            right_hi = math.inf if right.maximum_km is None else right.maximum_km
            if max(left_lo, right_lo) < min(left_hi, right_hi):
                raise ValueError(
                    f"wavelength bands {left.name!r} and {right.name!r} overlap")
    return bands


def bands_as_records(bands: Sequence[WavelengthBand]) -> list[dict[str, object]]:
    return [asdict(band) for band in bands]


# ---------------------------------------------------------------------------
# Preprocessing and geometry
# ---------------------------------------------------------------------------


def _as_finite_plane(value: np.ndarray, *, label: str) -> np.ndarray:
    plane = np.asarray(value, dtype=np.float64)
    if plane.ndim != 2 or plane.size == 0:
        raise ValueError(f"{label} must be one non-empty 2-D plane")
    if not np.all(np.isfinite(plane)):
        raise ValueError(f"{label} contains non-finite values")
    return plane


def crop_plane(plane: np.ndarray, crop_cells: int) -> np.ndarray:
    """Remove one equal-width frame from all four edges."""

    if int(crop_cells) != crop_cells or crop_cells < 0:
        raise ValueError("crop_cells must be a non-negative integer")
    values = _as_finite_plane(plane, label="spectral plane")
    crop = int(crop_cells)
    if crop == 0:
        result = values
    else:
        if values.shape[0] <= 2 * crop or values.shape[1] <= 2 * crop:
            raise ValueError("crop removes the entire spectral plane")
        result = values[crop:-crop, crop:-crop]
    if min(result.shape) < 8:
        raise ValueError("cropped spectral plane must be at least 8x8")
    return np.asarray(result, dtype=np.float64)


def detrend_plane(plane: np.ndarray) -> np.ndarray:
    """Remove the least-squares plane ``a + b*x + c*y`` in float64."""

    values = _as_finite_plane(plane, label="detrend input")
    ny, nx = values.shape
    y = np.arange(ny, dtype=np.float64) - 0.5 * (ny - 1)
    x = np.arange(nx, dtype=np.float64) - 0.5 * (nx - 1)
    result = values - float(np.mean(values, dtype=np.float64))
    yy = float(np.sum(y * y, dtype=np.float64)) * nx
    xx = float(np.sum(x * x, dtype=np.float64)) * ny
    if yy > 0.0:
        result = result - (
            float(np.sum(result * y[:, None], dtype=np.float64)) / yy
        ) * y[:, None]
    if xx > 0.0:
        result = result - (
            float(np.sum(result * x[None, :], dtype=np.float64)) / xx
        ) * x[None, :]
    return result


def hann_window(ny: int, nx: int) -> np.ndarray:
    """Symmetric separable Hann normalized to unit 2-D mean square."""

    if ny < 2 or nx < 2:
        raise ValueError("a Hann window needs at least two cells per axis")
    window = np.outer(np.hanning(ny), np.hanning(nx))
    mean_square = float(np.mean(window * window, dtype=np.float64))
    if not math.isfinite(mean_square) or mean_square <= 0.0:
        raise ValueError("Hann window has no finite power")
    return window / math.sqrt(mean_square)


def preprocess_plane(plane: np.ndarray, *, crop_cells: int = 0) -> np.ndarray:
    cropped = crop_plane(plane, crop_cells)
    return detrend_plane(cropped) * hann_window(*cropped.shape)


def _validate_spacing(dx_m: float, dy_m: float) -> tuple[float, float]:
    dx = float(dx_m)
    dy = float(dy_m)
    if (not math.isfinite(dx) or not math.isfinite(dy)
            or dx <= 0.0 or dy <= 0.0):
        raise ValueError("dx_m and dy_m must be positive and finite")
    return dx, dy


def mode_geometry(ny: int, nx: int, *, dx_m: float, dy_m: float
                  ) -> dict[str, np.ndarray | float | int]:
    """Physical Fourier geometry and the complete isotropic support disk."""

    if ny < 8 or nx < 8:
        raise ValueError("spectral geometry needs at least 8x8 cells")
    dx, dy = _validate_spacing(dx_m, dy_m)
    ky = np.fft.fftfreq(ny, d=dy)
    kx = np.fft.fftfreq(nx, d=dx)
    ky2d = ky[:, None]
    kx2d = kx[None, :]
    magnitude = np.hypot(ky2d, kx2d)
    isotropic_nyquist = min(0.5 / dx, 0.5 / dy)
    # One ulp-scaled allowance keeps an exactly-on-Nyquist mode from moving
    # across the boundary because dx was represented by a nearby float.
    allowance = np.finfo(np.float64).eps * max(1.0, isotropic_nyquist) * 8.0
    valid = (magnitude > 0.0) & (magnitude <= isotropic_nyquist + allowance)
    wavelength_m = np.full_like(magnitude, np.inf, dtype=np.float64)
    wavelength_m[valid] = 1.0 / magnitude[valid]
    return {
        "kx_cycles_per_m": np.broadcast_to(kx2d, (ny, nx)),
        "ky_cycles_per_m": np.broadcast_to(ky2d, (ny, nx)),
        "magnitude_cycles_per_m": magnitude,
        "wavelength_m": wavelength_m,
        "valid": valid,
        "isotropic_nyquist_cycles_per_m": float(isotropic_nyquist),
        "valid_mode_count": int(np.count_nonzero(valid)),
    }


def _band_mask(geometry: Mapping[str, object], band: WavelengthBand) -> np.ndarray:
    wavelength = np.asarray(geometry["wavelength_m"], dtype=np.float64)
    mask = np.asarray(geometry["valid"], dtype=bool).copy()
    if band.minimum_km is not None:
        mask &= wavelength >= band.minimum_km * 1000.0
    if band.maximum_km is not None:
        mask &= wavelength < band.maximum_km * 1000.0
    return mask


def _parseval(windowed: np.ndarray, transform: np.ndarray) -> dict[str, float]:
    spatial = float(np.mean(windowed * windowed, dtype=np.float64))
    count = windowed.size
    spectral = float(np.sum(np.abs(transform) ** 2, dtype=np.float64)
                     / (count * count))
    scale = max(abs(spatial), abs(spectral), np.finfo(np.float64).tiny)
    return {
        "spatial_mean_square": spatial,
        "spectral_mean_square": spectral,
        "absolute_error": abs(spatial - spectral),
        "relative_error": abs(spatial - spectral) / scale,
    }


# ---------------------------------------------------------------------------
# Band arithmetic
# ---------------------------------------------------------------------------


def _finite_or_none(value: float | complex | None) -> float | None:
    if value is None:
        return None
    result = float(value.real if isinstance(value, complex) else value)
    return result if math.isfinite(result) else None


def _band_metrics(*, left_power_modes: np.ndarray,
                  reference_power_modes: np.ndarray,
                  cross_modes: np.ndarray, error_power_modes: np.ndarray,
                  mask: np.ndarray, left_valid_power: float,
                  reference_valid_power: float) -> dict[str, object]:
    mode_count = int(np.count_nonzero(mask))
    if mode_count == 0:
        return {
            "status": "unresolved",
            "mode_count": 0,
            "note": "no retained Fourier mode falls in this wavelength band",
        }

    left_power = float(np.sum(left_power_modes[mask], dtype=np.float64))
    reference_power = float(
        np.sum(reference_power_modes[mask], dtype=np.float64))
    cross = complex(np.sum(cross_modes[mask], dtype=np.complex128))
    error_power = float(np.sum(error_power_modes[mask], dtype=np.float64))
    if any(not math.isfinite(value) or value < 0.0
           for value in (left_power, reference_power, error_power)):
        raise ValueError("band reduction produced invalid power")

    denominator = math.sqrt(left_power * reference_power)
    cross_selected = np.asarray(cross_modes[mask], dtype=np.complex128)
    phase_weights = np.abs(cross_selected)
    phase_weight_sum = float(np.sum(phase_weights, dtype=np.float64))
    if denominator > 0.0:
        spectral_correlation = max(-1.0, min(1.0, cross.real / denominator))
        coherence_squared = max(
            0.0, min(1.0, (abs(cross) ** 2) / (left_power * reference_power)))
    else:
        spectral_correlation = None
        coherence_squared = None
    if phase_weight_sum > 0.0:
        mean_absolute_phase_error_degrees = float(np.sum(
            phase_weights * np.abs(np.angle(cross_selected)),
            dtype=np.float64) / phase_weight_sum * (180.0 / math.pi))
        mean_absolute_phase_error_degrees = max(
            0.0, min(180.0, mean_absolute_phase_error_degrees))
    else:
        mean_absolute_phase_error_degrees = None

    amplitude_mismatch = (
        math.sqrt(left_power) - math.sqrt(reference_power)
    ) ** 2
    # Cauchy-Schwarz makes this non-negative analytically.  A very small
    # negative is cancellation; a material one is an arithmetic defect.
    phase_location = error_power - amplitude_mismatch
    roundoff = 256.0 * np.finfo(np.float64).eps * max(
        left_power, reference_power, error_power, 1.0)
    if phase_location < -roundoff:
        raise ValueError("phase/location error is negative beyond roundoff")
    phase_location = max(0.0, phase_location)

    result: dict[str, object] = {
        "status": "ok",
        "mode_count": mode_count,
        "left_power": left_power,
        "reference_power": reference_power,
        "error_power": error_power,
        "amplitude_mismatch_power": amplitude_mismatch,
        "phase_location_error_power": phase_location,
        "left_fraction_of_valid_power": (
            left_power / left_valid_power if left_valid_power > 0.0 else None),
        "reference_fraction_of_valid_power": (
            reference_power / reference_valid_power
            if reference_valid_power > 0.0 else None),
        "power_ratio": (
            left_power / reference_power if reference_power > 0.0 else None),
        "amplitude_ratio": (
            math.sqrt(left_power / reference_power)
            if reference_power > 0.0 else None),
        "normalized_error_power": (
            error_power / reference_power if reference_power > 0.0 else None),
        "phase_location_fraction_of_error": (
            phase_location / error_power if error_power > 0.0 else 0.0),
        "spectral_correlation": _finite_or_none(spectral_correlation),
        "coherence_squared": _finite_or_none(coherence_squared),
        "cross_spectrum_real": float(cross.real),
        "cross_spectrum_imag": float(cross.imag),
        "weighted_mean_absolute_phase_error_degrees": _finite_or_none(
            mean_absolute_phase_error_degrees),
    }
    return result


def _uncovered_modes(geometry: Mapping[str, object],
                     bands: Sequence[WavelengthBand]) -> dict[str, int]:
    valid = np.asarray(geometry["valid"], dtype=bool)
    covered = np.zeros_like(valid)
    for band in bands:
        covered |= _band_mask(geometry, band)
    return {
        "valid_mode_count": int(np.count_nonzero(valid)),
        "covered_mode_count": int(np.count_nonzero(valid & covered)),
        "uncovered_mode_count": int(np.count_nonzero(valid & ~covered)),
    }


def _band_record(band: WavelengthBand, metrics: Mapping[str, object]
                 ) -> dict[str, object]:
    return {
        "name": band.name,
        "minimum_wavelength_km": band.minimum_km,
        "maximum_wavelength_km": band.maximum_km,
        **dict(metrics),
    }


# ---------------------------------------------------------------------------
# Scalar comparison
# ---------------------------------------------------------------------------


def compare_scalar(left: np.ndarray, reference: np.ndarray, *,
                   dx_m: float, dy_m: float | None = None,
                   bands: Sequence[WavelengthBand], crop_cells: int = 0
                   ) -> dict[str, object]:
    """Compare one scalar plane against a reference by wavelength band."""

    dy = float(dx_m if dy_m is None else dy_m)
    dx, dy = _validate_spacing(dx_m, dy)
    left_raw = _as_finite_plane(left, label="left scalar field")
    reference_raw = _as_finite_plane(reference, label="reference scalar field")
    if left_raw.shape != reference_raw.shape:
        raise ValueError("scalar comparison operands must share one shape")
    bands = tuple(bands)
    # Round-trip through the public validator so a programmatic caller cannot
    # bypass the same overlap/name contract the registration path uses.
    bands = parse_bands(bands_as_records(bands))

    left_windowed = preprocess_plane(left_raw, crop_cells=crop_cells)
    reference_windowed = preprocess_plane(reference_raw, crop_cells=crop_cells)
    left_fft = np.fft.fft2(left_windowed)
    reference_fft = np.fft.fft2(reference_windowed)
    ny, nx = left_windowed.shape
    count_squared = float((ny * nx) ** 2)
    left_power_modes = np.abs(left_fft) ** 2 / count_squared
    reference_power_modes = np.abs(reference_fft) ** 2 / count_squared
    cross_modes = left_fft * np.conjugate(reference_fft) / count_squared
    error_modes = np.abs(left_fft - reference_fft) ** 2 / count_squared
    geometry = mode_geometry(ny, nx, dx_m=dx, dy_m=dy)
    valid = np.asarray(geometry["valid"], dtype=bool)
    left_valid_power = float(np.sum(left_power_modes[valid], dtype=np.float64))
    reference_valid_power = float(
        np.sum(reference_power_modes[valid], dtype=np.float64))

    rows = []
    for band in bands:
        rows.append(_band_record(
            band,
            _band_metrics(
                left_power_modes=left_power_modes,
                reference_power_modes=reference_power_modes,
                cross_modes=cross_modes,
                error_power_modes=error_modes,
                mask=_band_mask(geometry, band),
                left_valid_power=left_valid_power,
                reference_valid_power=reference_valid_power,
            ),
        ))

    return {
        "schema": COMPARISON_SCHEMA,
        "kind": "scalar",
        "implementation_pins_sha256": IMPLEMENTATION_PINS_SHA256,
        "input_shape": list(left_raw.shape),
        "scored_shape": [ny, nx],
        "dx_m": dx,
        "dy_m": dy,
        "crop_cells": int(crop_cells),
        "isotropic_nyquist_cycles_per_m": geometry[
            "isotropic_nyquist_cycles_per_m"],
        "mode_coverage": _uncovered_modes(geometry, bands),
        "left_parseval": _parseval(left_windowed, left_fft),
        "reference_parseval": _parseval(reference_windowed, reference_fft),
        "left_valid_disk_power": left_valid_power,
        "reference_valid_disk_power": reference_valid_power,
        "bands": rows,
    }


# ---------------------------------------------------------------------------
# Vector/Helmholtz comparison
# ---------------------------------------------------------------------------


def _vector_component_modes(*, left_u_fft: np.ndarray,
                            left_v_fft: np.ndarray,
                            reference_u_fft: np.ndarray,
                            reference_v_fft: np.ndarray,
                            geometry: Mapping[str, object],
                            count_squared: float
                            ) -> dict[str, dict[str, np.ndarray]]:
    kx = np.asarray(geometry["kx_cycles_per_m"], dtype=np.float64)
    ky = np.asarray(geometry["ky_cycles_per_m"], dtype=np.float64)
    magnitude = np.asarray(geometry["magnitude_cycles_per_m"], dtype=np.float64)
    nonzero = magnitude > 0.0
    unit_x = np.zeros_like(magnitude)
    unit_y = np.zeros_like(magnitude)
    unit_x[nonzero] = kx[nonzero] / magnitude[nonzero]
    unit_y[nonzero] = ky[nonzero] / magnitude[nonzero]

    def projected(u_fft: np.ndarray, v_fft: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
        divergent = unit_x * u_fft + unit_y * v_fft
        rotational = -unit_y * u_fft + unit_x * v_fft
        divergent[~nonzero] = 0.0
        rotational[~nonzero] = 0.0
        return rotational, divergent

    left_rot, left_div = projected(left_u_fft, left_v_fft)
    reference_rot, reference_div = projected(reference_u_fft, reference_v_fft)

    def scalar_modes(left_fft: np.ndarray, reference_fft: np.ndarray,
                     factor: float) -> dict[str, np.ndarray]:
        return {
            "left_power": factor * np.abs(left_fft) ** 2 / count_squared,
            "reference_power": (
                factor * np.abs(reference_fft) ** 2 / count_squared),
            "cross": (factor * left_fft * np.conjugate(reference_fft)
                      / count_squared),
            "error": (factor * np.abs(left_fft - reference_fft) ** 2
                      / count_squared),
        }

    total = {
        "left_power": (0.5 * (np.abs(left_u_fft) ** 2
                              + np.abs(left_v_fft) ** 2) / count_squared),
        "reference_power": (
            0.5 * (np.abs(reference_u_fft) ** 2
                   + np.abs(reference_v_fft) ** 2) / count_squared),
        "cross": (0.5 * (left_u_fft * np.conjugate(reference_u_fft)
                         + left_v_fft * np.conjugate(reference_v_fft))
                  / count_squared),
        "error": (0.5 * (np.abs(left_u_fft - reference_u_fft) ** 2
                         + np.abs(left_v_fft - reference_v_fft) ** 2)
                  / count_squared),
    }
    return {
        "total": total,
        "rotational": scalar_modes(left_rot, reference_rot, 0.5),
        "divergent": scalar_modes(left_div, reference_div, 0.5),
    }


def _vector_parseval(u_windowed: np.ndarray, v_windowed: np.ndarray,
                     u_fft: np.ndarray, v_fft: np.ndarray) -> dict[str, float]:
    spatial = float(np.mean(
        0.5 * (u_windowed * u_windowed + v_windowed * v_windowed),
        dtype=np.float64))
    count_squared = float(u_windowed.size ** 2)
    spectral = float(np.sum(
        0.5 * (np.abs(u_fft) ** 2 + np.abs(v_fft) ** 2),
        dtype=np.float64) / count_squared)
    scale = max(abs(spatial), abs(spectral), np.finfo(np.float64).tiny)
    return {
        "spatial_mean_kinetic_energy": spatial,
        "spectral_mean_kinetic_energy": spectral,
        "absolute_error": abs(spatial - spectral),
        "relative_error": abs(spatial - spectral) / scale,
    }


def compare_vector(left_u: np.ndarray, left_v: np.ndarray,
                   reference_u: np.ndarray, reference_v: np.ndarray, *,
                   dx_m: float, dy_m: float | None = None,
                   bands: Sequence[WavelengthBand], crop_cells: int = 0
                   ) -> dict[str, object]:
    """Compare horizontal wind KE and its rotational/divergent partitions."""

    dy = float(dx_m if dy_m is None else dy_m)
    dx, dy = _validate_spacing(dx_m, dy)
    raw = [
        _as_finite_plane(left_u, label="left U"),
        _as_finite_plane(left_v, label="left V"),
        _as_finite_plane(reference_u, label="reference U"),
        _as_finite_plane(reference_v, label="reference V"),
    ]
    if len({item.shape for item in raw}) != 1:
        raise ValueError("all four vector operands must share one shape")
    bands = parse_bands(bands_as_records(tuple(bands)))

    prepared = [preprocess_plane(item, crop_cells=crop_cells) for item in raw]
    left_u_w, left_v_w, reference_u_w, reference_v_w = prepared
    transforms = [np.fft.fft2(item) for item in prepared]
    left_u_f, left_v_f, reference_u_f, reference_v_f = transforms
    ny, nx = left_u_w.shape
    geometry = mode_geometry(ny, nx, dx_m=dx, dy_m=dy)
    valid = np.asarray(geometry["valid"], dtype=bool)
    count_squared = float((ny * nx) ** 2)
    modes = _vector_component_modes(
        left_u_fft=left_u_f, left_v_fft=left_v_f,
        reference_u_fft=reference_u_f, reference_v_fft=reference_v_f,
        geometry=geometry, count_squared=count_squared)

    valid_power: dict[str, tuple[float, float]] = {}
    for component, values in modes.items():
        valid_power[component] = (
            float(np.sum(values["left_power"][valid], dtype=np.float64)),
            float(np.sum(values["reference_power"][valid], dtype=np.float64)),
        )

    rows: list[dict[str, object]] = []
    for band in bands:
        mask = _band_mask(geometry, band)
        record: dict[str, object] = {
            "name": band.name,
            "minimum_wavelength_km": band.minimum_km,
            "maximum_wavelength_km": band.maximum_km,
            "components": {},
        }
        for component in ("total", "rotational", "divergent"):
            values = modes[component]
            left_valid, reference_valid = valid_power[component]
            record["components"][component] = _band_metrics(
                left_power_modes=values["left_power"],
                reference_power_modes=values["reference_power"],
                cross_modes=values["cross"],
                error_power_modes=values["error"],
                mask=mask,
                left_valid_power=left_valid,
                reference_valid_power=reference_valid,
            )
        left_total = record["components"]["total"].get("left_power")
        left_div = record["components"]["divergent"].get("left_power")
        ref_total = record["components"]["total"].get("reference_power")
        ref_div = record["components"]["divergent"].get("reference_power")
        record["left_divergent_energy_fraction"] = (
            float(left_div) / float(left_total)
            if left_total not in (None, 0.0) and left_div is not None else None)
        record["reference_divergent_energy_fraction"] = (
            float(ref_div) / float(ref_total)
            if ref_total not in (None, 0.0) and ref_div is not None else None)
        rows.append(record)

    left_total_modes = modes["total"]["left_power"]
    left_parts = (modes["rotational"]["left_power"]
                  + modes["divergent"]["left_power"])
    reference_total_modes = modes["total"]["reference_power"]
    reference_parts = (modes["rotational"]["reference_power"]
                       + modes["divergent"]["reference_power"])
    closure_scale = max(
        float(np.max(left_total_modes[valid], initial=0.0)),
        float(np.max(reference_total_modes[valid], initial=0.0)),
        np.finfo(np.float64).tiny,
    )
    helmholtz_error = max(
        float(np.max(np.abs(left_total_modes[valid] - left_parts[valid]),
                     initial=0.0)),
        float(np.max(np.abs(reference_total_modes[valid]
                            - reference_parts[valid]), initial=0.0)),
    )

    return {
        "schema": COMPARISON_SCHEMA,
        "kind": "vector",
        "implementation_pins_sha256": IMPLEMENTATION_PINS_SHA256,
        "input_shape": list(raw[0].shape),
        "scored_shape": [ny, nx],
        "dx_m": dx,
        "dy_m": dy,
        "crop_cells": int(crop_cells),
        "isotropic_nyquist_cycles_per_m": geometry[
            "isotropic_nyquist_cycles_per_m"],
        "mode_coverage": _uncovered_modes(geometry, bands),
        "left_parseval": _vector_parseval(
            left_u_w, left_v_w, left_u_f, left_v_f),
        "reference_parseval": _vector_parseval(
            reference_u_w, reference_v_w, reference_u_f, reference_v_f),
        "helmholtz_mode_closure": {
            "maximum_absolute_error": helmholtz_error,
            "maximum_relative_error": helmholtz_error / closure_scale,
        },
        "left_valid_disk_kinetic_energy": valid_power["total"][0],
        "reference_valid_disk_kinetic_energy": valid_power["total"][1],
        "bands": rows,
    }


def metric_rows(result: Mapping[str, object]) -> list[dict[str, object]]:
    """Flatten a comparison into gate-addressable band/component rows."""

    kind = str(result.get("kind"))
    rows: list[dict[str, object]] = []
    for band in result.get("bands", []):
        if kind == "scalar":
            rows.append({"band": band["name"], "component": "scalar", **band})
        elif kind == "vector":
            for component, metrics in band.get("components", {}).items():
                rows.append({
                    "band": band["name"],
                    "component": component,
                    **metrics,
                })
        else:
            raise ValueError(f"unknown spectral comparison kind {kind!r}")
    return rows


__all__ = [
    "COMPARISON_SCHEMA", "IMPLEMENTATION_PINS", "IMPLEMENTATION_PINS_SHA256",
    "IMPLEMENTATION_PINS_SCHEMA", "WavelengthBand", "bands_as_records",
    "compare_scalar", "compare_vector", "crop_plane", "detrend_plane",
    "hann_window", "implementation_registration", "metric_rows",
    "mode_geometry", "parse_bands", "preprocess_plane",
]
