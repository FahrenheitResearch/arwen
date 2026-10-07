"""Radar reflectivity observation classes, error model and vertical thinning.

DA design 2026-10-05, component A1 (lane 2).  The breakages this prevents,
named and measured on the 2026-10-01 19Z CONUS 9 km snapshot:

* The echo batch used the in-cell MAXIMUM of the gates (``z_obs`` was
  ``z_max`` in every cell).  It sat 6.6 dB above the in-cell linear mean
  (p90 12.6 dB), so O-B was +10.8 dBZ with the maximum against +5.2 with the
  mean, and the observed 35 dBZ area on the 9 km grid was 8.1 times MRMS's
  (2.4 times with the mean).  The filter was told storms were far wider and
  stronger than radar saw.  The mean is now the default reduction
  (:data:`gpuwm.da.obs_radar.Z_SOURCES`), because the model's H(x) is also
  the logarithm of a volume reflectivity.
* The observation error shrank with gate count (5/sqrt(n), floored at 2 dB),
  which is right for a mean of independent gates and wrong for correlated
  ones beside a 9 km model: representativeness dominates.  A fixed error
  can be stated per class.
* Weak returns had no class of their own.  Here every echo observation is
  exactly one of:

  - precipitation, ``y >= echo_floor``: observation and H(x) both floored at
    ``echo_floor``;
  - dead band, ``clear_floor <= y < echo_floor``: not assimilated;
  - clear, ``y < clear_floor``: observation set to ``clear_floor`` and H(x)
    floored at ``clear_floor``, the same floor the clear-air (zero) batch
    carries.

  A no-echo observation over a no-echo member therefore gives exactly zero
  innovation in every class, which :func:`assert_shared_floor` checks on the
  arrays the filter will read.
* Correlated radar data were over-weighted: about 730k values per cycle.
  Echo is kept on every ``echo_level_stride``-th level and clear air on
  every ``clear_level_stride``-th, and nothing above ``top_pa`` (11 km in
  the standard atmosphere) is assimilated.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

SCHEMA = "gpuwm-da.radar-classes.v1"

#: The engine's precipitation-echo threshold (shared with
#: gpuwm.da.reflectivity_echo and gpuwm.da.hotstart).
ECHO_FLOOR_DBZ = 15.0
#: Clear-air floor shared by observation and H(x).  The design's replay arms
#: are 0, 5 and 15 dBZ.
DEFAULT_CLEAR_FLOOR_DBZ = 0.0
DEFAULT_DEAD_BAND = True
#: Clear-air observation error, dBZ (design A1).
DEFAULT_CLEAR_ERROR_DBZ = 5.0
#: Echo error: None keeps the file's per-cell error.
DEFAULT_ECHO_ERROR_DBZ = None
DEFAULT_ECHO_LEVEL_STRIDE = 2
DEFAULT_CLEAR_LEVEL_STRIDE = 4
#: 11 km in the ICAO standard atmosphere, Pa.
DEFAULT_TOP_PA = 22632.0


class RadarClassError(ValueError):
    """A class setting this module cannot honour."""


def check_settings(*, clear_floor, echo_error, clear_error, echo_stride,
                   clear_stride, top_pa, huber_c):
    for label, value in (("reflectivity_clear_floor_dbz", clear_floor),
                         ("reflectivity_error_dbz", echo_error),
                         ("clear_air_error_dbz", clear_error),
                         ("radar_top_pa", top_pa),
                         ("reflectivity_huber_c", huber_c)):
        if value is not None and not np.isfinite(float(value)):
            raise RadarClassError(f"{label} must be finite or None, got {value!r}")
    for label, value in (("reflectivity_error_dbz", echo_error),
                         ("clear_air_error_dbz", clear_error),
                         ("reflectivity_huber_c", huber_c)):
        if value is not None and float(value) <= 0.0:
            raise RadarClassError(f"{label} must be positive, got {value!r}")
    if clear_floor is not None and float(clear_floor) > ECHO_FLOOR_DBZ:
        raise RadarClassError(
            f"reflectivity_clear_floor_dbz {clear_floor} is above the "
            f"{ECHO_FLOOR_DBZ} dBZ echo floor; the classes would overlap")
    for label, value in (("reflectivity_level_stride", echo_stride),
                         ("clear_air_level_stride", clear_stride)):
        if int(value) < 1:
            raise RadarClassError(f"{label} must be >= 1, got {value!r}")


def level_keep(pressure, stride: int, top_pa) -> np.ndarray:
    """``(nz, ny, nx)`` mask: every ``stride``-th model level from the
    surface, and only where the pressure is at or above ``top_pa``."""
    p = np.asarray(pressure, dtype=np.float64)
    keep = np.zeros(p.shape, dtype=bool)
    keep[::int(stride)] = True
    if top_pa is not None:
        keep &= p >= float(top_pa)
    return keep


def _masked(batch):
    mask = np.asarray(batch.mask, dtype=bool)
    values = np.asarray(batch.values, dtype=np.float64)
    errors = np.asarray(batch.errors, dtype=np.float64)
    if errors.ndim == 0:
        errors = np.full(values.shape, float(errors))
    return mask, values, errors


def classify_echo_batch(batch, *, clear_floor, dead_band: bool, echo_error,
                        clear_error, keep) -> tuple:
    """``(batch, floor, receipt)``: the echo batch split into the three
    classes, with ``floor`` the per-point H(x) floor (the value each class's
    observation and H(x) share).  ``clear_floor=None`` leaves the legacy
    behaviour (no classes) and returns ``floor=None``."""
    mask, values, errors = _masked(batch)
    receipt = {"observations_in": int(mask.sum())}
    if keep is not None:
        dropped = mask & ~keep
        receipt["thinned_by_level_or_top"] = int(dropped.sum())
        mask = mask & keep
    if clear_floor is None:
        receipt["classes"] = None
        out = replace(batch, mask=mask)
        if echo_error is not None:
            errors = errors.copy()
            errors[mask] = float(echo_error)
            out = replace(out, errors=errors)
        receipt["observations_out"] = int(mask.sum())
        return out, None, receipt
    clear_floor = float(clear_floor)
    precip = mask & (values >= ECHO_FLOOR_DBZ)
    band = mask & (values >= clear_floor) & (values < ECHO_FLOOR_DBZ)
    clear = mask & (values < clear_floor)
    new_values = values.copy()
    new_errors = errors.copy()
    floor = np.full(values.shape, ECHO_FLOOR_DBZ)
    if dead_band:
        new_mask = precip | clear
    else:
        # band observations stay as "no significant echo" against the echo
        # floor (the A7 behaviour)
        new_mask = precip | band | clear
        new_values[band] = ECHO_FLOOR_DBZ
    new_values[clear] = clear_floor
    floor[clear] = clear_floor
    if echo_error is not None:
        new_errors[precip | band] = float(echo_error)
    if clear_error is not None:
        new_errors[clear] = float(clear_error)
    receipt["classes"] = {
        "precipitation": int(precip.sum()),
        "dead_band": int(band.sum()),
        "dead_band_assimilated": not dead_band,
        "clear": int(clear.sum()),
        "echo_floor_dbz": ECHO_FLOOR_DBZ,
        "clear_floor_dbz": clear_floor,
    }
    receipt["observations_out"] = int(new_mask.sum())
    return (replace(batch, values=new_values, errors=new_errors,
                    mask=new_mask), floor, receipt)


def classify_clear_batch(batch, *, clear_floor, clear_error, keep) -> tuple:
    """``(batch, floor, receipt)`` for the clear-air (zero) batch: its value
    and H(x) floor become ``clear_floor`` (None keeps the scheme floor the
    adapter stated), its error ``clear_error`` (None keeps the file's)."""
    mask, values, errors = _masked(batch)
    receipt = {"observations_in": int(mask.sum())}
    if keep is not None:
        receipt["thinned_by_level_or_top"] = int((mask & ~keep).sum())
        mask = mask & keep
    new_values, new_errors = values, errors
    floor = None
    if clear_floor is not None:
        new_values = np.full(values.shape, float(clear_floor))
        floor = np.full(values.shape, float(clear_floor))
    if clear_error is not None:
        new_errors = errors.copy()
        new_errors[mask] = float(clear_error)
    receipt["observations_out"] = int(mask.sum())
    receipt["clear_floor_dbz"] = None if clear_floor is None else float(clear_floor)
    return (replace(batch, values=new_values, errors=new_errors, mask=mask),
            floor, receipt)


def apply_floor(batch, floor):
    """The batch with H(x) floored at ``floor`` (per point) on a NEW array:
    the clear-air and echo batches share one simulated array, which must
    not be written."""
    if floor is None:
        return batch
    mask = np.asarray(batch.mask, dtype=bool)
    sim = np.array(batch.simulated, dtype=np.float64, copy=True)
    sim[:, mask] = np.maximum(sim[:, mask], floor[mask][None, :])
    return replace(batch, simulated=sim)


def assert_shared_floor(batch, floor) -> None:
    """Every assimilated observation and every member's H(x) sit at or above
    the one floor of the observation's class, and an observation AT its floor
    over a member at its floor has zero innovation (asserted, not
    configured: the -15/-35 mismatch was a configuration)."""
    if floor is None:
        return
    mask = np.asarray(batch.mask, dtype=bool)
    if not mask.any():
        return
    y = np.asarray(batch.values, dtype=np.float64)[mask]
    hx = np.asarray(batch.simulated, dtype=np.float64)[:, mask]
    f = floor[mask]
    if np.any(y < f) or np.any(hx < f[None, :]):
        raise RadarClassError(
            f"{batch.name}: an observation or H(x) sits below its class "
            "floor; observation and H(x) must share one floor")


def huber_errors(batch, c) -> tuple:
    """``(batch, receipt)``: Huber weighting folded into sigma_o.  Beyond
    ``|d| > c * s`` (``s = sqrt(spread^2 + sigma_o^2)``) R is inflated by
    ``|d| / (c s)`` (design B1.3), so the observation's weight falls like
    1/|d| instead of being cut off."""
    mask, values, errors = _masked(batch)
    receipt = {"c": None if c is None else float(c), "inflated": 0}
    if c is None or not mask.any():
        return batch, receipt
    sim = np.asarray(batch.simulated, dtype=np.float64)[:, mask]
    d = values[mask] - sim.mean(axis=0)
    spread2 = sim.var(axis=0, ddof=1) if sim.shape[0] > 1 else 0.0 * d
    e = errors[mask]
    s = np.sqrt(spread2 + e ** 2)
    factor = np.where(np.abs(d) > float(c) * s, np.abs(d) / (float(c) * s), 1.0)
    new = errors.copy()
    new[mask] = e * np.sqrt(factor)
    receipt["inflated"] = int(np.count_nonzero(factor > 1.0))
    return replace(batch, errors=new), receipt
