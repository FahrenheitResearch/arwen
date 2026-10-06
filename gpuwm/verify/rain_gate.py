"""Regional-rain/v1 hourly measurements with conservative common support.

Rust owns projection, corner recovery, polygon overlap, time integration and
physical-width FSS. Python records metadata and schedules those kernels. A
measurement can be complete while the scientific gate remains pending: one
event and two seeds cannot establish event-blocked confidence bounds.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Literal
import numpy as np
from . import rain_gate_bridge as native

SCHEMA = "regional-rain/v1"
SPACING_M = 3000.0
EARTH_RADIUS_M = 6370000.0
THRESHOLDS_MM = (1.0, 5.0, 10.0)
WIDTHS_KM = (10.0, 25.0, 50.0)
LEADS_HOURS = (1, 3, 6)


@dataclass(frozen=True)
class RectGrid:
    x_edges_m: np.ndarray
    y_edges_m: np.ndarray
    projection_id: str

    def __post_init__(self):
        for name in ("x_edges_m", "y_edges_m"):
            edge = native.array(getattr(self, name))
            if edge.ndim != 1 or edge.size < 2 or not np.all(np.isfinite(edge)) or not np.all(np.diff(edge) > 0):
                raise ValueError("rectangular grid edges must be finite and increasing")
            object.__setattr__(self, name, edge)
        if not self.projection_id:
            raise ValueError("equal-area projection identity is required")

    @property
    def shape(self):
        return len(self.y_edges_m)-1, len(self.x_edges_m)-1

    def corners(self):
        return np.meshgrid(self.x_edges_m, self.y_edges_m)


@dataclass(frozen=True)
class QuadGrid:
    x_corners_m: np.ndarray
    y_corners_m: np.ndarray
    projection_id: str

    def __post_init__(self):
        x, y = native.array(self.x_corners_m), native.array(self.y_corners_m)
        if x.shape != y.shape or x.ndim != 2 or min(x.shape) < 2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("quadrilateral grid needs finite common (ny+1,nx+1) corners")
        object.__setattr__(self, "x_corners_m", x)
        object.__setattr__(self, "y_corners_m", y)
        if not self.projection_id:
            raise ValueError("equal-area projection identity is required")

    @property
    def shape(self):
        return self.x_corners_m.shape[0]-1, self.x_corners_m.shape[1]-1

    def corners(self):
        return self.x_corners_m, self.y_corners_m


@dataclass
class Series:
    grid: RectGrid | QuadGrid
    times_seconds: np.ndarray
    rain_accum_mm: np.ndarray | None = None
    rate_mm_h: np.ndarray | None = None
    rain_valid: np.ndarray | None = None
    reset_ids: np.ndarray | None = None
    reset_carry_mm: np.ndarray | None = None
    echo_dbz: np.ndarray | None = None
    echo_valid: np.ndarray | None = None
    echo_times_seconds: np.ndarray | None = None
    footprint_masks: dict[str, np.ndarray] | None = None


def project_laea(lat, lon, center_lon, center_lat):
    lat, lon = native.array(lat), native.array(lon)
    if lat.shape != lon.shape:
        raise ValueError("latitude and longitude shapes differ")
    x, y = np.empty_like(lat), np.empty_like(lat)
    native.call("laea", lat, lon, lat.size, float(center_lon), float(center_lat), x, y)
    return x, y


def corners_from_centers(x, y):
    x, y = native.array(x), native.array(y)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("projected centers must have a common 2-D shape")
    shape = x.shape[0]+1, x.shape[1]+1
    xc, yc = np.empty(shape, np.float64), np.empty(shape, np.float64)
    native.call("corners", x, *x.shape, xc)
    native.call("corners", y, *y.shape, yc)
    return xc, yc


def common_grid(*grids, spacing_m=SPACING_M, max_cells=2_000_000):
    if spacing_m != SPACING_M:
        raise ValueError("regional-rain/v1 requires a 3000-m equal-area grid")
    if not grids or len({g.projection_id for g in grids}) != 1:
        raise ValueError("all source grids must share one equal-area projection")
    bounds = [g.corners() for g in grids]
    xmin = spacing_m*math.floor(min(float(x.min()) for x, _ in bounds)/spacing_m)
    xmax = spacing_m*math.ceil(max(float(x.max()) for x, _ in bounds)/spacing_m)
    ymin = spacing_m*math.floor(min(float(y.min()) for _, y in bounds)/spacing_m)
    ymax = spacing_m*math.ceil(max(float(y.max()) for _, y in bounds)/spacing_m)
    nx, ny = int(round((xmax-xmin)/spacing_m)), int(round((ymax-ymin)/spacing_m))
    if nx*ny > max_cells:
        raise ValueError("truth region is too large; provide a pinned case region containing all observed storm objects and 50-km neighborhoods")
    return RectGrid(xmin+np.arange(nx+1)*spacing_m, ymin+np.arange(ny+1)*spacing_m, grids[0].projection_id)


def conservative_remap(field, valid, source, target):
    field, valid = native.array(field), native.array(valid, np.uint8)
    if field.shape != source.shape or valid.shape != source.shape:
        raise ValueError("source field/mask shape differs from its cell geometry")
    if source.projection_id != target.projection_id or not isinstance(target, RectGrid):
        raise ValueError("remap needs one equal-area projection and rectangular target cells")
    sx, sy = (native.array(v) for v in source.corners())
    value = np.empty(target.shape, np.float64)
    good = np.empty(target.shape, np.uint8)
    coverage = np.empty(target.shape, np.float64)
    r = np.empty(3, np.float64)
    native.call("remap", field, valid, sx, sy, *source.shape,
                target.x_edges_m, target.y_edges_m, *target.shape, value, good, coverage, r)
    error = abs(float(r[1])-float(r[0]))/max(abs(float(r[0])), 1e-30)
    receipt = {"operator": "Rust projected quadrilateral overlap", "projection_id": target.projection_id,
               "matched_source_integral_mm_m2": float(r[0]), "destination_integral_mm_m2": float(r[1]),
               "valid_area_m2": float(r[2]), "volume_error_fraction": error,
               "coverage_min": float(coverage.min()), "coverage_max": float(coverage.max())}
    if error > 0.001:
        raise ValueError("conservative remap volume closure exceeds 0.1 percent")
    return value, good.view(np.bool_), receipt


def integrate_hour(times_seconds, field, valid, lead_hours, *, mode: Literal["accum", "rate", "echo"],
                   reset_ids=None, reset_carry_mm=None, max_gap_seconds=120):
    t, f = native.array(times_seconds), native.array(field)
    v = native.array(valid, np.uint8)
    if t.ndim != 1 or f.ndim != 3 or f.shape != v.shape or f.shape[0] != len(t):
        raise ValueError("series needs times (t), fields and quality masks (t,y,x)")
    if mode not in ("accum", "rate", "echo"):
        raise ValueError("unknown hourly quantity")
    if mode == "accum" and reset_ids is None:
        raise ValueError("cumulative rain requires explicit reset_ids for each frame")
    ids = native.array(np.zeros(len(t), np.int64) if reset_ids is None else reset_ids, np.int64)
    if ids.shape != t.shape:
        raise ValueError("reset IDs must match times")
    carry = native.array(np.full(f.shape, np.nan) if reset_carry_mm is None else reset_carry_mm)
    if carry.shape != f.shape:
        raise ValueError("reset carry must match cumulative rain shape")
    out, good, footprint = np.empty(f.shape[1:]), np.empty(f.shape[1:], np.uint8), np.empty(f.shape[1:], np.uint8)
    r = np.empty(3)
    start, end = (float(lead_hours)-1)*3600, float(lead_hours)*3600
    try:
        native.call("hour", t, f, v, ids, carry, len(t), out.size, start, end,
                    float(max_gap_seconds), ("accum", "rate", "echo").index(mode), out, good, footprint, r)
    except ValueError as error:
        out.fill(np.nan)
        good.fill(0)
        footprint.fill(0)
        return out, good.view(np.bool_), {"status": "pending", "reason": str(error),
                "window_seconds": [start, end]}, footprint.view(np.bool_)
    receipt = {"status": "complete" if bool(good.all()) else "pending",
               "reason": None if bool(good.all()) else "missing quality support, unexplained counter decrease or unrecorded reset carry",
               "window_seconds": [start, end], "segments": int(r[0]), "resets": int(r[1]),
               "max_gap_seconds": float(r[2]), "integration": "endpoint difference plus recorded reset carry" if mode == "accum" else "trapezoidal two-minute bracketing series clipped to exact hourly window"}
    return out, good.view(np.bool_), receipt, footprint.view(np.bool_)


def hourly_accumulation(times_seconds, accum_mm, reset_ids, reset_carry_mm, lead_hours, *, max_gap_seconds=120, valid=None):
    if valid is None:
        valid = np.isfinite(accum_mm)
    result, valid, receipt, _ = integrate_hour(times_seconds, accum_mm, valid, lead_hours,
        mode="accum", reset_ids=reset_ids, reset_carry_mm=reset_carry_mm, max_gap_seconds=max_gap_seconds)
    return result, valid, receipt


def fss_exact(model, obs, valid, threshold_mm, width_km, *, dx_m=SPACING_M):
    model, obs, valid = native.array(model), native.array(obs), native.array(valid, np.uint8)
    if model.shape != obs.shape or model.shape != valid.shape or model.ndim != 2:
        raise ValueError("FSS fields and masks need one 2-D shape")
    r = np.empty(5)
    support = np.empty(model.shape, np.uint8)
    native.call("fss", model, obs, valid, *model.shape, float(threshold_mm), float(width_km)*1000, float(dx_m), r, support)
    reason = "no complete common neighborhood support" if r[2] == 0 else "empty observed wet event" if r[3] == 0 else None
    value = None if reason is not None or r[1] == 0 else 1.0-float(r[0])/float(r[1])
    return {"numerator": float(r[0]), "denominator": float(r[1]), "value": value,
            "scored_cells": int(r[2]), "support_area_m2": float(r[4]),
            "support_hash": hashlib.sha256(support.tobytes()).hexdigest(),
            "threshold_mm_h": float(threshold_mm), "scale_km": float(width_km),
            "status": "pending" if reason else "complete", "reason": reason}


def _support_record(mask, area_m2):
    return {"support_hash": hashlib.sha256(np.ascontiguousarray(mask, np.uint8).tobytes()).hexdigest(),
            "support_count": int(np.count_nonzero(mask)), "support_area_m2": float(np.count_nonzero(mask))*area_m2}


def _ratio(metric, numerator, denominator, mask, area_m2, reason=None, interval=None):
    if denominator <= 0 and reason is None:
        reason = "empty observed wet event"
    value = numerator/denominator if reason is None else None
    screen = None if value is None or interval is None else "pass" if interval[0] <= value <= interval[1] else "fail"
    return {"metric": metric, "numerator": numerator, "denominator": denominator, "value": value,
            "status": "pending" if reason else "complete", "reason": reason,
            "screen_status": screen, "screen_interval": interval, "qualification_status": "pending",
            **_support_record(mask, area_m2)}


def score_hour(model_rain, obs_rain, model_valid, obs_valid, model_echo_fraction,
               obs_echo_fraction, echo_valid, *, footprint=None, grid, footprint_incomplete=False,
               footprint_observed_rain=None):
    if not isinstance(grid, RectGrid) or not np.allclose(np.diff(grid.x_edges_m), SPACING_M) or not np.allclose(np.diff(grid.y_edges_m), SPACING_M):
        raise ValueError("score grid must have exact 3000-m equal-area square cells")
    arrays = [np.asarray(x) for x in (model_rain, obs_rain, model_valid, obs_valid, model_echo_fraction, obs_echo_fraction, echo_valid)]
    if any(x.shape != grid.shape for x in arrays):
        raise ValueError("all hourly fields and masks must match the common grid")
    model_rain, obs_rain, mv, ov, me, oe, ev = arrays
    valid = mv.astype(bool) & ov.astype(bool) & np.isfinite(model_rain) & np.isfinite(obs_rain)
    ev = ev.astype(bool) & np.isfinite(me) & np.isfinite(oe)
    fraction = (oe > 0).astype(float) if footprint is None else np.asarray(footprint, float)
    if fraction.shape != grid.shape or not np.all(np.isfinite(fraction)) or np.any(fraction < 0) or np.any(fraction > 1+1e-7):
        raise ValueError("observed footprint shape differs from common grid")
    foot = fraction > 0
    touches = bool(np.any(foot[0]) or np.any(foot[-1]) or np.any(foot[:, 0]) or np.any(foot[:, -1]))
    incomplete = footprint_incomplete or touches
    area_m2 = SPACING_M**2
    model_footrain = native.combine(np.stack((model_rain, fraction)), mode="product")
    observed_footrain = native.combine(np.stack((obs_rain, fraction)), mode="product") if footprint_observed_rain is None else np.asarray(footprint_observed_rain)
    if observed_footrain.shape != grid.shape:
        raise ValueError("native masked footprint rain shape differs from common grid")
    rain_num = native.sum_masked(model_footrain, foot & valid)*area_m2
    rain_den = native.sum_masked(observed_footrain, foot & ov.astype(bool) & np.isfinite(observed_footrain))*area_m2
    reason = "observed storm object crosses a truth-region edge" if incomplete else "missing observed footprint support" if np.any(foot & ~valid) else None
    rows = [_ratio("footprint_rain_ratio", rain_num, rain_den, foot, area_m2, reason, [0.8, 1.2]),
            _ratio("domain_rain_ratio", native.sum_masked(model_rain, valid)*area_m2,
                   native.sum_masked(obs_rain, valid)*area_m2, valid, area_m2,
                   "no common covered domain" if not np.any(valid) else None, [0.8, 1.2]),
            _ratio("echo_area_multiple", native.sum_masked(me, ev)*area_m2*3600,
                   native.sum_masked(oe, ev)*area_m2*3600, ev, area_m2,
                   "incomplete observed echo footprint" if incomplete or np.any(foot & ~ev) else None, [0.7, 1.3])]
    rows[0]["support_area_m2"] = native.sum_masked(fraction, foot)*area_m2
    rows[0]["support_hash"] = hashlib.sha256(native.array(fraction).tobytes()).hexdigest()
    rows[0]["footprint_definition"] = "conservative area fractions of the native observed 35-dBZ hourly union; truth rain masked before remap"
    for threshold in THRESHOLDS_MM:
        for width in WIDTHS_KM:
            f = fss_exact(model_rain, obs_rain, valid, threshold_mm=threshold, width_km=width)
            rows.append({"metric": "fss", **f,
                         "qualification_status": "pending", "screen_status": None,
                         "support_count": f["scored_cells"], "support_area_m2": f["support_area_m2"]})
    return rows


def score_series(forecast: Series, truth: Series, *, leads_hours=LEADS_HOURS, target=None):
    target = common_grid(forecast.grid, truth.grid) if target is None else target
    allrows, diagnostics = [], {}
    for lead in leads_hours:
        receipts = {}
        fields = []
        for label, series in (("forecast", forecast), ("truth", truth)):
            rain = series.rain_accum_mm if series.rain_accum_mm is not None else series.rate_mm_h
            mode = "accum" if series.rain_accum_mm is not None else "rate"
            if rain is None or series.rain_valid is None or series.echo_dbz is None or series.echo_valid is None:
                raise ValueError(f"{label} requires rain, echo and explicit native quality masks")
            # Archive filenames include product-completion second jitter.
            # Thirty seconds around the nominal two-minute cadence admits
            # that documented jitter while a lost two-minute frame fails.
            gap_limit = 120 if label == "forecast" else 150
            r, rv, receipt, _ = integrate_hour(series.times_seconds, rain, series.rain_valid, lead, mode=mode,
                                              reset_ids=series.reset_ids, reset_carry_mm=series.reset_carry_mm,
                                              max_gap_seconds=gap_limit)
            echo_times = series.times_seconds if series.echo_times_seconds is None else series.echo_times_seconds
            e, ev, ereceipt, foot = integrate_hour(echo_times, series.echo_dbz, series.echo_valid, lead, mode="echo",
                                                  max_gap_seconds=gap_limit)
            receipt["configured_max_gap_seconds"] = gap_limit
            ereceipt["configured_max_gap_seconds"] = gap_limit
            frame_missing = receipt.get("segments") is None or ereceipt.get("segments") is None
            if label == "truth" and series.footprint_masks is not None:
                if str(lead) not in series.footprint_masks:
                    raise ValueError(f"explicit observed footprint has no lead {lead}")
                foot = np.asarray(series.footprint_masks[str(lead)], bool)
                if foot.shape != series.grid.shape:
                    raise ValueError("explicit observed footprint does not match native truth grid")
            incomplete = bool(foot[0].any() or foot[-1].any() or foot[:, 0].any() or foot[:, -1].any())
            rr, rrv, remap_r = conservative_remap(r, rv, series.grid, target)
            ee, eev, remap_e = conservative_remap(e, ev, series.grid, target)
            # Construct the observed union using native echo before model coverage.
            fp, _, remap_f = conservative_remap(foot.astype(np.float64), np.ones(foot.shape, bool), series.grid, target)
            # Mask truth rain on its native cells before changing grids. This
            # retains subcell covariance instead of multiplying two averages.
            native_footrain = native.combine(np.stack((r, foot.astype(float))), mode="product")
            footrain, _, remap_fr = conservative_remap(native_footrain, rv, series.grid, target)
            fields.append((rr, rrv, ee, eev, fp, incomplete, frame_missing, footrain))
            receipts[label] = {"rain_hour": receipt, "echo_hour": ereceipt,
                               "rain_remap": remap_r, "echo_remap": remap_e, "footprint_remap": remap_f,
                               "native_masked_footprint_rain_remap": remap_fr}
        m, o = fields
        rows = score_hour(m[0], o[0], m[1], o[1], m[2], o[2], m[3]&o[3],
                          footprint=o[4], grid=target, footprint_incomplete=o[5], footprint_observed_rain=o[7])
        missing = m[6] or o[6]
        for row in rows:
            row.update({"schema": SCHEMA, "lead_hours": float(lead), "window_seconds": [(lead-1)*3600, lead*3600],
                        "projection_id": target.projection_id, "grid_spacing_m": SPACING_M,
                        "confidence_bounds": None, "paired_baseline": None})
            if missing:
                row.update({"status": "pending", "reason": "missing required hourly endpoint or two-minute frame",
                            "value": None, "screen_status": None})
        allrows.extend(rows)
        diagnostics[str(lead)] = receipts
    return allrows, diagnostics
