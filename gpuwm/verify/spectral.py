"""Radially-averaged 2-D power spectra, and the distance between two of them.

The other metric classes ask whether two runs hold the same values in the same
places.  This one asks something the others cannot: whether they hold their
variance at the same *scales*.  Two forecasts can carry equal domain RMSE while
one of them has moved its energy from 10 km features to 2 km noise, and only a
spectrum says so.

Everything a spectrum can be argued about after the fact is pinned here as
data, in :data:`PINS`, and hashed into :data:`PINS_SHA256`: which fields are
scored, how each 3-D field becomes the 2-D plane whose spectrum is taken, the
detrend, the window, the periodogram normalization, the radial bin width and
assignment, which bins are retained, the floor under the logarithm, and the
distance.  The hash is committed against a literal in the test suite, so a pin
edited after a receipt exists changes the hash and fails a test rather than
quietly re-defining what an already-published number measured.

The distance is an integrated log-spectral difference: the integral over the
retained radial band of ``|log10 P_left - log10 P_right|`` in ``dk``, divided
by the width of that band.  It is read in decades.  Logarithms rather than
differences of power because a spectrum spans orders of magnitude across the
band and a linear difference would be a statement about the largest scales
alone; an integral rather than a peak because a spectral defect is usually a
tilt over a range of scales rather than one bad wavenumber.

Three properties this module exists to keep honest:

* **Zero means identical, and nothing else does.**  The distance between a
  field and itself is exactly ``0.0``, and any bin-wise difference is strictly
  positive.  A distance that could not tell two analytically different spectra
  apart would pass a recovery test and still measure nothing, which is why the
  monotonicity control -- not the recovery test -- is this module's failure
  control.
* **The DC bin is not scored.**  A plane's mean is removed by the detrend
  before the transform, so the zero-wavenumber bin carries only the residue of
  that removal.  Scoring it would report the detrend's rounding as physics.
* **The band stops at the isotropic Nyquist.**  Beyond ``1/(2 dx)`` the
  Fourier grid covers only the corners of k-space, so a radial average there
  is an average over the diagonal directions alone and is not isotropic.

Nothing here is a campaign artifact.  These pins are the general case, taken
by whatever pair of like-configured runs the caller points them at, with the
grid spacing an argument.  The frozen N5S campaign pins in
:mod:`gpuwm.verify.n5s_metrics` are a separate, closed record of one campaign
and are deliberately not parameterized to reach this module: a gate record
that was measured under those pins keeps the numbers it was measured with, and
the general case is carried here.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Sequence

import numpy as np

SPECTRAL_SCHEMA = "gpuwm.spectral-pins/v1"

#: The scored plane of a 3-D field, by name.  ``column_max`` is the ordinary
#: composite of a reflectivity-like field; ``column_max_abs`` keeps the
#: strongest vertical motion in a column whichever way it is going, so an
#: updraft and a downdraft of the same strength are the same feature to the
#: spectrum.  A field that is already 2-D is its own plane.
PLANE_REDUCTIONS = ("column_max", "column_max_abs")

#: Field selection, pinned: the history variable, the role it plays, and the
#: reduction to the plane whose spectrum is taken.  A caller does not choose
#: these -- that choice is the pre-registration.
FIELD_PINS: tuple[dict[str, str], ...] = (
    {"role": "vertical_velocity", "variable": "W", "plane": "column_max_abs"},
    {"role": "composite_reflectivity", "variable": "REFL_10CM",
     "plane": "column_max"},
)

#: Every argument-after-the-fact, decided once and hashed.
PINS: dict[str, object] = {
    "schema": SPECTRAL_SCHEMA,
    "fields": [dict(pin) for pin in FIELD_PINS],
    "detrend": (
        "least-squares plane a + b*x + c*y removed from the scored plane, "
        "with x and y the zero-centred cell indices"),
    "window": (
        "separable symmetric Hann applied over both axes, scaled so the 2-D "
        "window has unit mean square"),
    "periodogram": (
        "P = dx*dy/(ny*nx) * |fft2(detrended, windowed)|^2, so the sum of P "
        "over the Fourier grid times dkx*dky is the windowed plane's mean "
        "square"),
    "grid_spacing": "isotropic: dy is taken equal to dx",
    "radial_bin_width": "dk = 1/(max(ny, nx) * dx)",
    "radial_bin_assignment": (
        "a Fourier mode of magnitude |k| falls in bin floor(|k|/dk), "
        "evaluated as floor(hypot(jy*max(ny, nx)/ny, jx*max(ny, nx)/nx)) over "
        "the signed integer frequency indices so that a mode sitting on a bin "
        "edge lands on the same side of it however |k| was formed; a bin's "
        "value is the arithmetic mean of P over the modes it holds"),
    "retained_band": (
        "bins 1 through (max(ny, nx) - 1)//2, which is the last bin whose "
        "centre (b + 0.5)*dk does not exceed the isotropic Nyquist 1/(2*dx); "
        "the DC bin is dropped because the detrend has already emptied it"),
    "minimum_retained_bins": 4,
    "empty_retained_bin_policy": "fail",
    "log_floor": (
        "before the logarithm both spectra are floored at 1e-12 times the "
        "larger of their retained-band maxima"),
    "log_floor_relative": 1e-12,
    "distance": (
        "integral over the retained band of |log10 P_left - log10 P_right| "
        "dk, divided by the band width; units are decades"),
    "missing_or_nonfinite_policy": "fail",
}


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


#: The pins' identity.  A receipt records this beside its numbers; the test
#: suite holds the literal it must equal.
PINS_SHA256: str = _canonical_hash(PINS)


def pinned_variable(role: str) -> str:
    """The history variable pinned for ``role``."""
    for pin in FIELD_PINS:
        if pin["role"] == role:
            return pin["variable"]
    raise ValueError(f"no spectral field is pinned for role {role!r}")


def registration() -> dict[str, object]:
    """The pins and their hash, as a caller embeds them in its own."""
    return {"pins": json.loads(json.dumps(PINS)),
            "pins_sha256": PINS_SHA256}


# --------------------------------------------------------------------------
# plane, detrend, window
# --------------------------------------------------------------------------


def scored_plane(field: np.ndarray, reduction: str) -> np.ndarray:
    """Reduce a 2-D or 3-D field to the 2-D plane whose spectrum is taken."""
    if reduction not in PLANE_REDUCTIONS:
        raise ValueError(f"unpinned plane reduction: {reduction!r}")
    values = np.asarray(field, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("spectral input plane is empty/non-finite")
    if values.ndim == 2:
        plane = values
    elif values.ndim == 3:
        plane = (np.max(np.abs(values), axis=0)
                 if reduction == "column_max_abs" else np.max(values, axis=0))
    else:
        raise ValueError("spectral input must be 2-D or 3-D after Time")
    if min(plane.shape) < 4:
        raise ValueError("spectral input plane is too small to bin radially")
    return plane


def detrend_plane(plane: np.ndarray) -> np.ndarray:
    """Remove the least-squares plane ``a + b*x + c*y``.

    On a full regular grid the centred index coordinates are orthogonal to one
    another and to the constant, so the least-squares coefficients are the
    three separable projections rather than a solve.
    """
    values = np.asarray(plane, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("detrend operates on a 2-D plane")
    ny, nx = values.shape
    y = np.arange(ny, dtype=np.float64) - 0.5 * (ny - 1)
    x = np.arange(nx, dtype=np.float64) - 0.5 * (nx - 1)
    result = values - float(np.mean(values, dtype=np.float64))
    yy = float(np.sum(y * y, dtype=np.float64)) * nx
    xx = float(np.sum(x * x, dtype=np.float64)) * ny
    if yy > 0.0:
        result = result - (float(np.sum(result * y[:, None], dtype=np.float64))
                           / yy) * y[:, None]
    if xx > 0.0:
        result = result - (float(np.sum(result * x[None, :], dtype=np.float64))
                           / xx) * x[None, :]
    return result


def hann_window(ny: int, nx: int) -> np.ndarray:
    """Separable symmetric Hann, scaled to unit mean square."""
    if ny < 2 or nx < 2:
        raise ValueError("a Hann window needs at least two cells per axis")
    window = np.outer(np.hanning(ny), np.hanning(nx))
    mean_square = float(np.mean(window * window, dtype=np.float64))
    if mean_square <= 0.0:
        raise ValueError("the window has no power")
    return window / math.sqrt(mean_square)


# --------------------------------------------------------------------------
# the spectrum
# --------------------------------------------------------------------------


def _frequency_indices(count: int) -> np.ndarray:
    """Signed integer Fourier frequency indices, in the transform's order."""
    indices = np.arange(count, dtype=np.int64)
    return np.where(indices > count // 2, indices - count,
                    indices).astype(np.float64)


def radial_bins(ny: int, nx: int, dx_m: float) -> dict[str, object]:
    """The pinned radial binning of the Fourier grid for this plane shape.

    The magnitude is formed in units of the bin width from the integer
    frequency indices rather than from the physical wavenumbers.  Along the
    longer axis the scale factor is exactly one, so an axis-aligned mode's
    magnitude is exactly the integer it should be and lands on the same side
    of the bin edge on every platform.  Forming ``|k|/dk`` from the physical
    frequencies instead leaves that on one rounding of ``1/(n*dx)``, and a
    mode crossing a bin edge moves the two bins it sits between.
    """
    if dx_m <= 0.0 or not math.isfinite(dx_m):
        raise ValueError("grid spacing must be positive and finite")
    span = max(ny, nx)
    bin_width = 1.0 / (span * dx_m)
    last = (span - 1) // 2
    if last < int(PINS["minimum_retained_bins"]):
        raise ValueError(
            f"a {ny}x{nx} plane retains {max(last, 0)} radial bins, fewer "
            f"than the pinned minimum of {PINS['minimum_retained_bins']}")
    units_y = _frequency_indices(ny) * (float(span) / ny)
    units_x = _frequency_indices(nx) * (float(span) / nx)
    index = np.floor(np.hypot(units_y[:, None],
                              units_x[None, :])).astype(np.int64)
    counts = np.bincount(index.ravel(), minlength=last + 1)
    empty = [b for b in range(1, last + 1) if counts[b] == 0]
    if empty:
        raise ValueError(f"radial bins {empty} hold no Fourier mode")
    centres = (np.arange(1, last + 1, dtype=np.float64) + 0.5) * bin_width
    return {"index": index, "first": 1, "last": last,
            "bin_width": float(bin_width), "counts": counts[1:last + 1],
            "wavenumbers": centres}


def radial_psd(plane: np.ndarray, dx_m: float) -> dict[str, object]:
    """Radially-averaged 2-D PSD of one plane, over the retained band."""
    values = np.asarray(plane, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("a PSD is taken over a 2-D plane")
    if not np.all(np.isfinite(values)):
        raise ValueError("spectral input plane is non-finite")
    ny, nx = values.shape
    windowed = detrend_plane(values) * hann_window(ny, nx)
    transform = np.fft.fft2(windowed)
    power = (dx_m * dx_m / (ny * nx)) * np.abs(transform) ** 2
    bins = radial_bins(ny, nx, dx_m)
    index = bins["index"]
    last = int(bins["last"])
    totals = np.bincount(index.ravel(), weights=power.ravel(),
                         minlength=last + 1)[1:last + 1]
    means = totals / np.asarray(bins["counts"], dtype=np.float64)
    if not np.all(np.isfinite(means)):
        raise ValueError("radially-averaged PSD is non-finite")
    return {"wavenumber_cycles_per_m": np.asarray(bins["wavenumbers"]),
            "power": means,
            "mode_count": np.asarray(bins["counts"]),
            "bin_width_cycles_per_m": float(bins["bin_width"])}


# --------------------------------------------------------------------------
# the distance
# --------------------------------------------------------------------------


def log_spectral_distance(left: np.ndarray, right: np.ndarray, *,
                          dx_m: float) -> float:
    """Integrated log-spectral difference between two planes, in decades."""
    left_plane = np.asarray(left, dtype=np.float64)
    right_plane = np.asarray(right, dtype=np.float64)
    if left_plane.shape != right_plane.shape:
        raise ValueError("spectral operands must share one plane shape")
    left_psd = radial_psd(left_plane, dx_m)
    right_psd = radial_psd(right_plane, dx_m)
    ceiling = max(float(np.max(left_psd["power"])),
                  float(np.max(right_psd["power"])))
    if not math.isfinite(ceiling) or ceiling <= 0.0:
        raise ValueError("both spectra are empty over the retained band")
    floor = float(PINS["log_floor_relative"]) * ceiling
    difference = np.abs(np.log10(np.maximum(left_psd["power"], floor))
                        - np.log10(np.maximum(right_psd["power"], floor)))
    bin_width = float(left_psd["bin_width_cycles_per_m"])
    band_width = bin_width * difference.size
    distance = float(np.sum(difference, dtype=np.float64) * bin_width
                     / band_width)
    if not math.isfinite(distance) or distance < 0.0:
        raise ValueError("log-spectral distance is non-finite or negative")
    return distance


def field_distance(left_field: np.ndarray, right_field: np.ndarray, *,
                   plane: str, dx_m: float) -> float:
    """Distance between two raw fields under one pinned plane reduction."""
    return log_spectral_distance(
        scored_plane(left_field, plane), scored_plane(right_field, plane),
        dx_m=dx_m)


def pair_distances(left_fields: Mapping[str, np.ndarray],
                   right_fields: Mapping[str, np.ndarray], *,
                   dx_m: float) -> dict[str, float]:
    """One distance per pinned field, keyed by that field's variable name."""
    distances: dict[str, float] = {}
    for pin in FIELD_PINS:
        variable = pin["variable"]
        if variable not in left_fields or variable not in right_fields:
            raise ValueError(
                f"the spectral pins require {variable} on both sides")
        distances[variable] = field_distance(
            left_fields[variable], right_fields[variable],
            plane=pin["plane"], dx_m=dx_m)
    return distances


def log_log_slope(psd: Mapping[str, np.ndarray], *,
                  bins: Sequence[int] | None = None) -> float:
    """Least-squares slope of ``log10 P`` against ``log10 k``.

    Not part of the distance.  It exists so a test can state what spectrum a
    synthetic field was built with and check that the same number comes back
    out of the retained band.
    """
    wavenumbers = np.asarray(psd["wavenumber_cycles_per_m"], dtype=np.float64)
    power = np.asarray(psd["power"], dtype=np.float64)
    if bins is not None:
        selection = np.asarray(list(bins), dtype=np.int64)
        wavenumbers = wavenumbers[selection]
        power = power[selection]
    if wavenumbers.size < 2 or np.any(power <= 0.0):
        raise ValueError("a log-log slope needs two or more positive bins")
    x = np.log10(wavenumbers)
    y = np.log10(power)
    x = x - np.mean(x, dtype=np.float64)
    return float(np.sum(x * y, dtype=np.float64)
                 / np.sum(x * x, dtype=np.float64))


__all__ = [
    "FIELD_PINS", "PINS", "PINS_SHA256", "PLANE_REDUCTIONS",
    "SPECTRAL_SCHEMA", "detrend_plane", "field_distance", "hann_window",
    "log_log_slope", "log_spectral_distance", "pair_distances",
    "pinned_variable", "radial_bins", "radial_psd", "registration",
    "scored_plane",
]
