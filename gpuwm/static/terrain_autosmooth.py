"""Bound terrain to the measured slope envelope before initialization.

Only over-limit fields enter the existing Rust WPS 1-2-1 smoother. The
unchanged branch keeps the original arrays, receipt and prepared identity.
The clock still checks crest and wind; a slope bound alone is not a claim
that every wind or time step is stable.
"""
from __future__ import annotations

import hashlib
import numpy as np

from gpuwm.acoustic_adaptation import (STABLE_SLOPE_BY_OFFCENTERING,
                                       steepest_slope, stable_slopes)
from .terrain_smoothing import TerrainSmoothing, smooth_terrain

SCHEMA = "gpuwm-terrain-autosmooth-v1"
MAX_PASSES = 32


class SmoothedFields(dict):
    """Numeric field mapping with a receipt outside the array inventory."""

    def __init__(self, fields, receipt):
        super().__init__(fields)
        self.terrain_autosmooth = receipt


def field_hash(terrain):
    return hashlib.sha256(np.asarray(terrain, dtype="<f8").tobytes()).hexdigest()


def measured_slope_limit(grid, run=None):
    """The slope envelope of the same combined map the local clock reads.

    Preparation has no forecast wind yet. Read the largest slope with a
    held six-substep entry at each bounding spacing and dynamics arm,
    capped by the acoustic map's strongest measured off-centering row.
    The clock subsequently reads the actual crest, wind and step.
    """
    from gpuwm.config import RunConfig
    from gpuwm.terrain_clock_local import candidate_map, dynamics_arms

    run = (RunConfig(nx=2, ny=2, nz=2, dx=grid.dx, dy=grid.dy,
                     ztop=20000., dt=1., run_seconds=1.)
           if run is None else run)
    arms = dynamics_arms(run)
    table = candidate_map(arms)
    spacings = sorted({row.dx_m for row in table.rows})
    limits = [stable_slopes(STABLE_SLOPE_BY_OFFCENTERING[-1][0])[2]]
    for spacing in (float(grid.dx), float(grid.dy)):
        below = [dx for dx in spacings if dx <= spacing]
        above = [dx for dx in spacings if dx >= spacing]
        bracket = {max(below) if below else spacings[0],
                   min(above) if above else spacings[-1]}
        for dx in bracket:
            held = [row.slope for row in table.rows
                    if row.dx_m == dx and row.sound_steps == 6
                    and any(value is not None for value in row.stable)]
            if not held:
                raise ValueError(
                    f"terrain at {dx:g} m has no held six-substep slope "
                    "in the measured clock map; choose a measured grid "
                    "spacing before preparing a forecast")
            limits.append(max(held))
    return min(limits), arms


def smooth_to_limit(terrain, grid, *, domain_id, run=None,
                    max_passes=MAX_PASSES):
    """Return (terrain, receipt), with no copy or receipt inside the limit."""
    if type(max_passes) is not int or not 0 <= max_passes <= MAX_PASSES:
        raise ValueError(f"terrain smoothing pass bound must be 0..{MAX_PASSES}")
    h = np.asarray(terrain, dtype=np.float64)
    if h.ndim != 2 or h.size == 0 or not np.isfinite(h).all():
        raise ValueError(
            f"d{int(domain_id):02d} terrain must be a non-empty finite plane; "
            "non-finite heights cannot define a stable terrain slope")
    if np.all(h == h.flat[0]):
        return terrain, None
    label = f"d{int(domain_id):02d}"
    limit, arms = measured_slope_limit(grid, run)
    factors = dict(msfu=grid.mapfac_u(), msfv=grid.mapfac_v(), label=label)
    before = steepest_slope(terrain, grid.dx, grid.dy, **factors)
    # The acoustic map defines stability BELOW its bound. Equality must
    # be smoothed too, or the acoustic door would immediately refuse it.
    if before.slope < limit:
        return terrain, None
    # Stage an owned edge-extended plane. Rust performs every numerical
    # smoothing sweep; the cropped field is checked after each one.
    extended = np.pad(h, 1, mode="edge")
    after = before
    for passes in range(1, max_passes + 1):
        extended = smooth_terrain(extended, TerrainSmoothing("1-2-1", 1))
        candidate = extended[1:-1, 1:-1].copy()
        after = steepest_slope(candidate, grid.dx, grid.dy, **factors)
        if after.slope < limit:
            from . import rust_bridge
            workaround = ("GPUWM_STATIC_PYTHON=1"
                          if rust_bridge.python_fallback_requested()
                          else rust_bridge.unavailable_reason())
            return candidate, {
                "schema": SCHEMA, "domain_id": int(domain_id),
                "smooth_option": "1-2-1", "smooth_passes": passes,
                "max_passes": max_passes, "slope_limit": limit,
                "slope_before": before.slope, "slope_after": after.slope,
                "degrees_before": before.degrees, "degrees_after": after.degrees,
                "face_before": list(before.face), "face_after": list(after.face),
                "terrain_before_sha256": field_hash(h),
                "terrain_after_sha256": field_hash(candidate),
                "measured_map": "local-face combined map and acoustic slope map",
                "dynamics_arms": list(arms),
                "static_compute": ("rust static-fields bridge" if workaround is None
                                   else f"pure-Python workaround ({workaround})"),
            }
    raise ValueError(
        f"{label} terrain slope {before.slope:.6g} remains {after.slope:.6g} "
        f"after {max_passes} 1-2-1 passes, outside the measured-stable "
        f"limit {limit:.6g}; refusing preparation before GPU time because "
        "this ground can produce non-finite state and runaway vertical "
        "velocity. Use a coarser grid or smoother terrain input, then "
        "prepare again.")


def receipt_of(fields):
    return getattr(fields, "terrain_autosmooth", None)


def run_line(receipt):
    return (f"terrain: d{receipt['domain_id']:02d} auto-smoothed with "
            f"1-2-1 x{receipt['smooth_passes']}; slope "
            f"{receipt['slope_before']:.6g} -> {receipt['slope_after']:.6g} "
            f"(measured-stable limit {receipt['slope_limit']:.6g})")


def prepare_fields(fields, grid, *, domain_id, run=None, announce=print):
    """Finalize terrain and its dependent TMN before building coordinates."""
    if "HGT_M" not in fields:
        raise ValueError(
            f"d{int(domain_id):02d} static fields lack HGT_M; refusing to "
            "prepare a coordinate without checking its terrain slope")
    terrain, receipt = smooth_to_limit(
        fields["HGT_M"], grid, domain_id=domain_id, run=run)
    if receipt is None:
        return fields
    from .highres import merge_terrain_override
    merged, _ = merge_terrain_override(fields, {"HGT_M": terrain})
    if announce is not None:
        announce(run_line(receipt))
    return SmoothedFields(merged, receipt)


def bind_receipt(fields, receipt):
    """Add the applied derivation only; leave every unchanged receipt alone."""
    applied = receipt_of(fields)
    if applied is None:
        return receipt
    return {**(receipt or {}), "terrain_autosmooth": applied}


def verify_receipt(receipt, terrain, *, domain_id):
    """A stored derivation must bind the exact terrain that will run."""
    if receipt is None:
        return None
    if (not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA
            or receipt.get("domain_id") != int(domain_id)
            or type(receipt.get("smooth_passes")) is not int
            or not 1 <= receipt["smooth_passes"] <= MAX_PASSES
            or receipt.get("smooth_option") != "1-2-1"
            or receipt.get("terrain_after_sha256") != field_hash(terrain)):
        raise ValueError(
            f"d{int(domain_id):02d} terrain auto-smoothing receipt does not "
            "bind its prepared terrain; prepare again to avoid running "
            "different terrain under the recorded slope and run identity")
    return receipt


def survey_domain_terrain(exp, geog_root, *, smoothing_settings=()):
    """The CPU domain door checks locally staged terrain before publishing."""
    from dataclasses import replace
    from .build import build_terrain, GeogSelection
    from .projection import grids_from_projection_config
    grids = grids_from_projection_config(exp)
    selection = GeogSelection.fallback(geog_root)
    rows = []
    for index, (domain, grid) in enumerate(zip(exp.domains, grids)):
        selected = (replace(selection, terrain_smoothing=smoothing_settings[
            min(index, len(smoothing_settings) - 1)])
            if smoothing_settings else selection)
        terrain = build_terrain(grid, geog_root, selection=selected)
        _, receipt = smooth_to_limit(
            terrain, grid, domain_id=domain.grid_id, run=domain.run)
        if receipt is not None:
            print(run_line(receipt) + "; local terrain survey, rechecked after prepare overlays")
            rows.append(receipt)
    return rows
