"""Put the dealiasing engines side by side on one real volume.

Five arms of the observation front door over the same decoded sweep pack,
so every difference is attributable to the engine and to nothing else:

    off                            masking only, no unfolding
    vad-region                     gpuwm.obs.dealias, the VAD abstainer
    region-global                  the vendored Rust crate
    region-global+refine           the same with the refinement pass --
                                   the shipped default
    region-global+refine+nobound   the same with the physical speed bound
                                   lifted, which is how what the bound
                                   refuses is measured rather than assumed

Each arm reports its wall clock, the three-state gate account, the
front door's own rejection counters, and statistics of the gridded ``vr``
field the DA adapter would ingest.  The gridded fields are then differenced
pairwise, per cell, because "the engines disagree" is a number of
observations and not an adjective.

A fifth section answers the couplet question directly: it locates the
strongest azimuthal velocity couplet in the volume from the data itself --
no site, storm or case is named anywhere here -- and prints what each arm
did to the gates in and around it.

    python tools/dealias_engine_compare.py --pack <sweeps> --out report.json

Nothing is written inside the source tree unless ``--out`` says so.
EXPERIMENTAL, like the rest of this lane.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

#: How far either side of a couplet's centre the "couplet-adjacent"
#: window reaches, in gates and in rays.  A mesocyclone couplet at these
#: ranges is a few kilometres across; this is deliberately wider, because
#: the question is whether the gates AROUND the couplet were resolved
#: consistently with it, not only the two extrema themselves.
COUPLET_HALF_GATES = 24
COUPLET_HALF_RAYS = 16


def grid_around(lat: float, lon: float, *, nx: int, ny: int, nz: int,
                dx_m: float, top_m: float):
    """A plain Lambert box centred on the radar, at the asked-for spacing.

    Same construction ``tools/perf_obs_timing.py`` uses, so the timings
    here sit on the same grid as the receipt they are compared against.
    """

    from gpuwm.obs.target_grid import TargetGrid
    from gpuwm.static.lambert import LambertGrid

    projection = LambertGrid(
        ref_lat=lat, ref_lon=lon, truelat1=lat, truelat2=lat, stand_lon=lon,
        dx=dx_m, dy=dx_m, e_we=nx + 1, e_sn=ny + 1)
    z_w = np.linspace(0.0, top_m, nz + 1)
    return TargetGrid.from_projection(projection, z_w=z_w, name="target",
                                      source="dealias-engine-compare")


#: The bound, lifted.  Not "off": a parameter that cannot be switched off
#: is asked for a number no atmosphere reaches, so the arm that measures
#: what the bound refuses is still running the same code path as the arm
#: that refuses it.
NO_BOUND_MS = 1.0e6


def arm_params(name: str):
    """``SuperobParams`` for one arm, differing ONLY in the engine.

    ``region-global+refine`` is the shipped default and is spelled out
    here anyway, because an arm that reads its own configuration from a
    default is an arm nobody can tell apart from a mistake.
    """

    from gpuwm.obs.dealias import (ENGINE_REGION_GLOBAL, ENGINE_VAD_REGION,
                                   DealiasParams)
    from gpuwm.obs.superob import SuperobParams

    if name == "off":
        dealias = None
    elif name == "vad-region":
        dealias = DealiasParams(engine=ENGINE_VAD_REGION)
    elif name == "region-global":
        dealias = DealiasParams(engine=ENGINE_REGION_GLOBAL,
                                refinement=False)
    elif name == "region-global+refine":
        dealias = DealiasParams(engine=ENGINE_REGION_GLOBAL, refinement=True)
    elif name == "region-global+refine+nobound":
        dealias = DealiasParams(engine=ENGINE_REGION_GLOBAL, refinement=True,
                                max_speed_ms=NO_BOUND_MS)
    else:                                        # pragma: no cover - guarded
        raise SystemExit(f"unknown arm {name!r}")
    return SuperobParams(dealias=dealias)


def gridded_vr(contribution):
    """The per-cell mean ``vr`` a DA adapter would read, NaN where empty."""

    count = np.asarray(contribution.vr_count)
    total = np.asarray(contribution.vr_sum, dtype=np.float64)
    mean = np.full(count.shape, np.nan, dtype=np.float64)
    filled = count > 0
    mean[filled] = total[filled] / count[filled]
    return mean, filled


def field_stats(mean, filled) -> dict:
    values = mean[filled]
    if values.size == 0:                         # pragma: no cover - empty
        return {"cells": 0}
    return {
        "cells": int(filled.sum()),
        "min_ms": round(float(values.min()), 4),
        "max_ms": round(float(values.max()), 4),
        "mean_ms": round(float(values.mean()), 4),
        "abs_mean_ms": round(float(np.abs(values).mean()), 4),
        "std_ms": round(float(values.std()), 4),
        "cells_above_25ms": int((np.abs(values) > 25.0).sum()),
        "cells_above_40ms": int((np.abs(values) > 40.0).sum()),
    }


def run_arm(name: str, pack, grid, *, repeat: int) -> dict:
    """One arm: time it, count it, and keep its gridded field."""

    from gpuwm.obs.superob import superob_volume
    from gpuwm.obs.sweeps import read_sweep_pack

    params = arm_params(name)
    volume = read_sweep_pack(pack)
    best = float("inf")
    contribution = None
    for _ in range(max(1, repeat)):
        started = time.perf_counter()
        contribution = superob_volume(volume, grid, params=params)
        best = min(best, time.perf_counter() - started)
    mean, filled = gridded_vr(contribution)
    counts = contribution.counts
    record = {
        "arm": name,
        "seconds": round(best, 4),
        "sweeps": len(volume.sweeps),
        "counts": {
            "velocity_gates_rejected_nyquist":
                int(counts.velocity_gates_rejected_nyquist),
            "velocity_gates_rejected_no_nyquist":
                int(counts.velocity_gates_rejected_no_nyquist),
            "velocity_gates_rejected_shear":
                int(counts.velocity_gates_rejected_shear),
            "velocity_cells_rejected_spread":
                int(counts.velocity_cells_rejected_spread),
            "velocity_fold_boundaries": int(counts.velocity_fold_boundaries),
            "velocity_gate_pairs_tested":
                int(counts.velocity_gate_pairs_tested),
        },
        "vr_field": field_stats(mean, filled),
    }
    if contribution.dealias:
        totals = contribution.dealias.get("totals")
        record["dealias_totals"] = totals
        rejected = (totals or {}).get("rejected") or {}
        # The physical bound's own line in the account.  Pulled out by name
        # so the table can print what the bound refused without a reader
        # having to know which counter it lands in.
        record["speed_rejections"] = int(rejected.get("speed_out_of_range", 0))
        record["max_speed_ms"] = float(
            params.dealias.max_speed_ms) if params.dealias else None
    return record, mean, filled


def diff_fields(a, a_filled, b, b_filled) -> dict:
    """What the DA layer would ingest differently between two arms."""

    both = a_filled & b_filled
    delta = np.zeros(a.shape, dtype=np.float64)
    delta[both] = b[both] - a[both]
    moved = both & (np.abs(delta) > 1e-6)
    payload = {
        "cells_in_both": int(both.sum()),
        "cells_only_in_first": int((a_filled & ~b_filled).sum()),
        "cells_only_in_second": int((b_filled & ~a_filled).sum()),
        "cells_differing": int(moved.sum()),
    }
    if moved.any():
        values = np.abs(delta[moved])
        payload.update({
            "max_abs_delta_ms": round(float(values.max()), 4),
            "mean_abs_delta_ms": round(float(values.mean()), 4),
            "cells_delta_above_10ms": int((values > 10.0).sum()),
        })
    return payload


#: Where a mesocyclone couplet can be looked for at all, in km.
#:
#: The near bound is the cone of silence and the first few gates; the far
#: bound is sampling.  A 0.5-degree cut's rays are 360/720 = 0.5 degrees
#: apart, so adjacent beams are 0.26 km apart at 30 km and 1.55 km apart
#: at 178 km: past about 120 km a two-beam velocity difference is no
#: longer resolving a couplet, it is straddling one.
COUPLET_RANGE_KM = (5.0, 120.0)

#: Reflectivity a couplet's two gates must both carry, dBZ.  A
#: mesocyclone lives inside an echo; a pair of opposite-signed velocities
#: in clear air is noise, and in this volume the debris signature carries
#: 60+.
COUPLET_MIN_DBZ = 20.0

#: Elevations searched.  A tornadic couplet is a low-level feature and the
#: cuts above this see the mid-level mesocyclone at best.
COUPLET_MAX_ELEVATION_DEG = 3.5


def _reflectivity_on(sweep, ranges):
    """The sweep's REF plane sampled onto the velocity range axis.

    The two moments have their own gate counts on a WSR-88D split cut, so
    a reflectivity floor applied by raw index would compare a velocity
    gate against a reflectivity gate somewhere else.
    """

    moment = sweep.moments.get("REF")
    if moment is None:
        return None
    reflectivity = np.asarray(moment.data, dtype=np.float64)
    index = np.clip(np.searchsorted(moment.slant_range_m(), ranges), 0,
                    reflectivity.shape[1] - 1)
    return reflectivity[:, index]


def find_couplet(pack, *, engine_params, top: int = 10):
    """The strongest adjacent-ray velocity couplet in the volume.

    Located from the data, never named: no site, storm or case appears
    here or anywhere else in this tool.

    **The search runs on a FOLD-RESOLVED field, deliberately.**  Run on
    raw velocities the strongest opposite-signed adjacent pair in this
    volume is at 178 km with both gates pinned at -26.0 and +26.0 against
    a 26.12 m/s Nyquist -- which is the Nyquist seam, not a couplet, and
    it outranks the real feature because an alias is a bigger number than
    a tornado.  Resolving folds first is what makes the maximum mean
    something.  The engine used for the search is stated in the report;
    the top candidates are printed with their locations so that a reader
    can see the answer is vertically coherent across cuts rather than one
    cut's speckle.

    The floors are stated as constants above with their reasons, so a
    candidate is refused for a named physical reason rather than by taste.
    """

    from gpuwm.obs.dealias import dealias_sweep
    from gpuwm.obs.sweeps import read_sweep_pack

    volume = read_sweep_pack(pack)
    candidates = []
    for sweep in volume.sweeps:
        moment = sweep.moments.get("VEL")
        if moment is None:
            continue
        if sweep.elevation_angle_deg > COUPLET_MAX_ELEVATION_DEG:
            continue
        raw = np.asarray(moment.data, dtype=np.float64)
        if raw.shape[0] < 2:                     # pragma: no cover - guarded
            continue
        resolved = dealias_sweep(
            raw, sweep.azimuth_deg, sweep.nyquist_velocity_ms, engine_params,
            first_gate_m=moment.first_gate_range_m,
            gate_spacing_m=moment.gate_size_m).velocity
        ranges = moment.slant_range_m()
        delta = resolved[1:, :] - resolved[:-1, :]
        # Opposite-signed neighbours only: a large difference between two
        # same-signed velocities is shear, not rotation.
        keep = np.isfinite(delta)
        keep &= np.sign(resolved[1:, :]) * np.sign(resolved[:-1, :]) < 0
        keep &= ((ranges[None, :] >= COUPLET_RANGE_KM[0] * 1000.0)
                 & (ranges[None, :] <= COUPLET_RANGE_KM[1] * 1000.0))
        echo = _reflectivity_on(sweep, ranges)
        if echo is not None:
            keep &= (np.isfinite(echo[1:, :]) & np.isfinite(echo[:-1, :])
                     & (echo[1:, :] >= COUPLET_MIN_DBZ)
                     & (echo[:-1, :] >= COUPLET_MIN_DBZ))
        scored = np.where(keep, np.abs(delta), 0.0)
        if not scored.any():
            continue
        ray, gate = np.unravel_index(int(np.argmax(scored)), scored.shape)
        candidates.append({
            "sweep_index": int(sweep.sweep_index),
            "elevation_angle_deg": float(sweep.elevation_angle_deg),
            "ray": int(ray),
            "gate": int(gate),
            "delta_v_ms": round(float(scored[ray, gate]), 3),
            "rotational_velocity_ms": round(float(scored[ray, gate]) / 2.0, 3),
            "azimuth_deg": round(float(sweep.azimuth_deg[ray]), 3),
            "range_km": round(float(ranges[gate]) / 1000.0, 3),
            "beam_separation_km": round(
                float(ranges[gate]) / 1000.0
                * np.radians(360.0 / raw.shape[0]), 4),
            "resolved_ms": [round(float(resolved[ray, gate]), 3),
                            round(float(resolved[ray + 1, gate]), 3)],
            "raw_ms": [round(float(raw[ray, gate]), 3),
                       round(float(raw[ray + 1, gate]), 3)],
            "reflectivity_dbz": (None if echo is None
                                 else round(float(echo[ray, gate]), 2)),
            "nyquist_ms": (None if sweep.nyquist_velocity_ms is None
                           else float(sweep.nyquist_velocity_ms)),
        })
    if not candidates:
        return None, []
    candidates.sort(key=lambda entry: -entry["delta_v_ms"])
    return candidates[0], candidates[:top]


#: How close to a WHOLE Nyquist interval a gate's departure from its own
#: neighbourhood must sit to be called a fold error, as a fraction of the
#: interval.
#:
#: The band, and not a plain threshold, is the whole instrument.  A fold
#: error displaces a gate by exactly one interval; a tornado's velocity
#: extremum displaces it by whatever the wind is.  A "large departure"
#: test cannot tell those apart, and measured on the real couplet in this
#: volume it flagged the couplet's own -58.7 m/s peak (0.75 intervals
#: from its neighbourhood median) as an error -- an instrument that
#: reports the feature it was pointed at as a defect.
FOLD_ISLAND_BAND = 0.2

#: Whole-interval offsets a fold error can plausibly sit at.  Two is
#: included because a region can be double-folded; past that a
#: neighbourhood median is not a meaningful reference any more.
FOLD_ISLAND_OFFSETS = (1.0, 2.0)


def fold_islands(plane, nyquist: float, *, window=None):
    """Gates that sit a WHOLE Nyquist interval off their own neighbourhood.

    A dealiased velocity field is continuous where the atmosphere is, so a
    single gate sitting one interval away from the median of the eight
    gates touching it is a misresolution rather than a measurement --
    whichever engine produced it.  This is the only engine-independent
    check available on real data without a truth field, and it is
    deliberately weak.  Its three stated limits:

    * A whole region resolved onto the wrong branch is continuous with
      itself and invisible here.
    * A gate next to a strong gradient is compared against a median the
      gradient has already pulled, so an error there can fall short of the
      band.
    * It measures INCONSISTENCY, never truth.  A field with no islands can
      still be uniformly wrong.

    Both directions of it are tested in
    ``tests/test_obs_dealias_region.py``, including the couplet-peak false
    positive that motivated the band.

    Returns ``(count, mask)``.
    """

    interval = 2.0 * float(nyquist)
    finite = np.isfinite(plane)
    padded = np.where(finite, plane, np.nan)
    stack = []
    for dray in (-1, 0, 1):
        for dgate in (-1, 0, 1):
            if dray == 0 and dgate == 0:
                continue
            shifted = np.roll(padded, dray, axis=0)          # rays wrap
            shifted = np.roll(shifted, dgate, axis=1)
            if dgate > 0:
                shifted[:, :dgate] = np.nan                  # gates do not
            elif dgate < 0:
                shifted[:, dgate:] = np.nan
            stack.append(shifted)
    neighbours = np.stack(stack)
    counted = np.sum(np.isfinite(neighbours), axis=0)
    # A gate with no finite neighbour at all is not judged (the count gate
    # below drops it), and nanmedian's all-NaN warning would be about
    # exactly those gates.  Silence it here rather than at the call site,
    # where a reader would have to work out which slice it meant.
    with np.errstate(invalid="ignore"), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(neighbours, axis=0)
    # Four finite neighbours is the fewest that makes a median a statistic
    # rather than a coin toss between two of them.
    departure = np.abs(padded - median) / interval
    near_whole = np.zeros(plane.shape, dtype=bool)
    for offset in FOLD_ISLAND_OFFSETS:
        near_whole |= np.abs(departure - offset) <= FOLD_ISLAND_BAND
    island = finite & (counted >= 4) & near_whole
    if window is not None:
        inside = np.zeros(plane.shape, dtype=bool)
        inside[window] = True
        island &= inside
    return int(island.sum()), island


#: Schema of the speed-bound fixture written by ``--speed-bound-fixture``.
SPEED_BOUND_FIXTURE_SCHEMA = "gpuwm-dealias.speed-bound-gates.v1"


def speed_bound_fixture(pack, *, bound_ms: float, couplet) -> dict:
    """Every gate the region-global solver puts past the physical bound.

    Measured with the bound lifted, so the record is what the SOLVER
    produced rather than what the bound left behind: each gate's raw
    velocity, the whole-interval fold applied to it, and the speed it
    landed at.  The couplet's own two gates are recorded beside them, from
    the same volume and the same solve, because "the bound rejects
    misresolutions and not storms" is a claim about both sets at once.

    This is the generator for ``tests/data/dealias_speed_bound_gates.json``.
    The test that consumes it runs on those numbers with no radar volume
    and no library present, which is the only way a real-data regression
    fixture survives into a CPU-only suite.
    """

    from gpuwm.obs.dealias import (ENGINE_REGION_GLOBAL, DealiasParams,
                                   dealias_sweep)
    from gpuwm.obs.sweeps import read_sweep_pack

    # The bound is what is being measured, so it must not be in force
    # during the measurement.
    unbounded = DealiasParams(engine=ENGINE_REGION_GLOBAL, refinement=True,
                              max_speed_ms=1.0e6)
    volume = read_sweep_pack(pack)
    cuts, beyond_rows, couplet_rows = [], [], []
    offered = 0
    for sweep in volume.sweeps:
        moment = sweep.moments.get("VEL")
        if moment is None:
            continue
        raw = np.asarray(moment.data, dtype=np.float64)
        result = dealias_sweep(
            raw, sweep.azimuth_deg, sweep.nyquist_velocity_ms, unbounded,
            first_gate_m=moment.first_gate_range_m,
            gate_spacing_m=moment.gate_size_m)
        ranges = moment.slant_range_m()
        finite = np.isfinite(raw)
        offered += int(finite.sum())
        beyond = np.isfinite(result.velocity) & (
            np.abs(result.velocity) > float(bound_ms))
        cuts.append({
            "sweep_index": int(sweep.sweep_index),
            "elevation_angle_deg": round(
                float(sweep.elevation_angle_deg), 2),
            "nyquist_ms": (None if sweep.nyquist_velocity_ms is None
                           else round(float(sweep.nyquist_velocity_ms), 3)),
            "gates_offered": int(finite.sum()),
            "gates_beyond_bound": int(beyond.sum()),
        })

        def _record(ray: int, gate: int) -> dict:
            return {
                "sweep_index": int(sweep.sweep_index),
                "elevation_angle_deg": round(
                    float(sweep.elevation_angle_deg), 2),
                "ray": int(ray), "gate": int(gate),
                "azimuth_deg": round(float(sweep.azimuth_deg[ray]), 2),
                "range_km": round(float(ranges[gate]) / 1000.0, 3),
                "nyquist_ms": round(float(sweep.nyquist_velocity_ms), 3),
                "raw_ms": round(float(raw[ray, gate]), 3),
                "unfolded_ms": round(float(result.velocity[ray, gate]), 3),
                "fold": int(result.fold[ray, gate]),
            }

        beyond_rows.extend(_record(ray, gate)
                           for ray, gate in np.argwhere(beyond))
        if couplet is not None \
                and int(sweep.sweep_index) == couplet["sweep_index"]:
            rows = raw.shape[0]
            couplet_rows.extend(
                _record(ray, couplet["gate"])
                for ray in (couplet["ray"], (couplet["ray"] + 1) % rows))
    return {
        "schema": SPEED_BOUND_FIXTURE_SCHEMA,
        "produced_by": ("tools/dealias_engine_compare.py "
                        "--speed-bound-fixture"),
        "bound_ms": float(bound_ms),
        "engine": "region-global",
        "refinement": True,
        "volume": {"valid_time": str(volume.valid_time),
                   "velocity_cuts": len(cuts), "gates_offered": offered},
        "cuts": cuts,
        "gates_beyond_bound": beyond_rows,
        "couplet_gates": couplet_rows,
    }


def bound_parity(pack) -> dict:
    """Prove the bound only abstains: same run, one parameter moved.

    Digests the fold plane, the state plane and the velocity field of the
    shipped configuration and of the same configuration with the bound
    lifted, **excluding the gates the bound refused**.  Equal digests are
    the statement that the layer is an abstention and not a filter with
    an opinion about the field: it may remove a gate, and it may not move
    one.

    Per sweep and over the volume, because a digest that matched only in
    aggregate could hide two sweeps that cancelled.
    """

    from gpuwm.obs.dealias import (STATE_REJECTED, DealiasParams,
                                   dealias_sweep)
    from gpuwm.obs.sweeps import read_sweep_pack

    shipped = DealiasParams()
    lifted = DealiasParams(engine=shipped.engine,
                           refinement=shipped.refinement,
                           max_speed_ms=NO_BOUND_MS)
    volume = read_sweep_pack(pack)
    digests = {name: hashlib.sha256()
               for name in ("fold_bounded", "fold_lifted", "state_bounded",
                            "state_lifted", "vr_bounded", "vr_lifted")}
    refused_total = 0
    disagreeing_sweeps = []
    for sweep in volume.sweeps:
        moment = sweep.moments.get("VEL")
        if moment is None:
            continue
        raw = np.asarray(moment.data, dtype=np.float64)
        geometry = {"first_gate_m": moment.first_gate_range_m,
                    "gate_spacing_m": moment.gate_size_m}
        held = dealias_sweep(raw, sweep.azimuth_deg,
                             sweep.nyquist_velocity_ms, shipped, **geometry)
        free = dealias_sweep(raw, sweep.azimuth_deg,
                             sweep.nyquist_velocity_ms, lifted, **geometry)
        refused = ((held.state == STATE_REJECTED)
                   & (free.state != STATE_REJECTED))
        refused_total += int(refused.sum())
        kept = ~refused
        agree = (np.array_equal(held.fold[kept], free.fold[kept])
                 and np.array_equal(held.state[kept], free.state[kept])
                 and np.array_equal(np.isnan(held.velocity[kept]),
                                    np.isnan(free.velocity[kept]))
                 and np.array_equal(
                     np.nan_to_num(held.velocity[kept], nan=0.0),
                     np.nan_to_num(free.velocity[kept], nan=0.0)))
        if not agree:
            disagreeing_sweeps.append(int(sweep.sweep_index))
        for label, result in (("bounded", held), ("lifted", free)):
            digests[f"fold_{label}"].update(
                np.ascontiguousarray(result.fold[kept]).tobytes())
            digests[f"state_{label}"].update(
                np.ascontiguousarray(result.state[kept]).tobytes())
            digests[f"vr_{label}"].update(np.ascontiguousarray(
                np.nan_to_num(result.velocity[kept], nan=-9999.0)).tobytes())
    hexed = {name: digest.hexdigest() for name, digest in digests.items()}
    return {
        "bound_ms": float(shipped.max_speed_ms),
        "gates_refused": refused_total,
        "sweeps_disagreeing_off_the_refused_gates": disagreeing_sweeps,
        "digests": hexed,
        "fold_planes_match": hexed["fold_bounded"] == hexed["fold_lifted"],
        "state_planes_match": hexed["state_bounded"] == hexed["state_lifted"],
        "velocities_match": hexed["vr_bounded"] == hexed["vr_lifted"],
    }


def _panel(plane, rays, gates) -> list[str]:
    """One ray per row, one gate per column, m/s to the nearest integer.

    ``.`` is a gate the stage in question has no number for -- no data on
    the raw panel, a refusal on an engine's.
    """

    out = []
    for ray in rays:
        cells = []
        for gate in gates:
            value = plane[ray, gate]
            cells.append("    ." if not np.isfinite(value)
                         else f"{value:5.0f}")
        out.append(" ".join(cells))
    return out


def couplet_window(pack, couplet, arms: dict) -> dict:
    """What each arm did to the gates in and around the couplet.

    The window is a rectangle in (ray, gate) around the couplet centre.
    Rays wrap; gates do not.
    """

    from gpuwm.obs.dealias import STATE_REJECTED, STATE_UNFOLDED
    from gpuwm.obs.sweeps import read_sweep_pack

    volume = read_sweep_pack(pack)
    sweep = next(s for s in volume.sweeps
                 if int(s.sweep_index) == couplet["sweep_index"])
    moment = sweep.moments["VEL"]
    raw = np.asarray(moment.data, dtype=np.float64)
    rows, gates = raw.shape
    rays = [(couplet["ray"] + k) % rows
            for k in range(-COUPLET_HALF_RAYS, COUPLET_HALF_RAYS + 1)]
    lo = max(0, couplet["gate"] - COUPLET_HALF_GATES)
    hi = min(gates, couplet["gate"] + COUPLET_HALF_GATES + 1)
    window = (np.array(rays)[:, None], np.arange(lo, hi)[None, :])

    report = {
        "window": {"rays": len(rays), "gates": int(hi - lo),
                   "gate_span": [int(lo), int(hi)]},
        "raw": {
            "gates_finite": int(np.isfinite(raw[window]).sum()),
            "min_ms": round(float(np.nanmin(raw[window])), 3),
            "max_ms": round(float(np.nanmax(raw[window])), 3),
        },
        "arms": {},
    }
    planes = {}
    for name, params in arms.items():
        if params.dealias is None:
            continue
        from gpuwm.obs.dealias import dealias_sweep
        result = dealias_sweep(
            raw, sweep.azimuth_deg, sweep.nyquist_velocity_ms, params.dealias,
            first_gate_m=moment.first_gate_range_m,
            gate_spacing_m=moment.gate_size_m)
        planes[name] = result
        patch = result.velocity[window]
        finite = np.isfinite(patch)
        here = (couplet["ray"], couplet["gate"])
        there = ((couplet["ray"] + 1) % rows, couplet["gate"])
        pair = [float(result.velocity[here]), float(result.velocity[there])]
        offered = int(np.isfinite(raw[window]).sum())
        rejected = int((result.state[window] == STATE_REJECTED).sum())
        report["arms"][name] = {
            "gates_offered_in_window": offered,
            "gates_unfolded_in_window":
                int((result.state[window] == STATE_UNFOLDED).sum()),
            "gates_rejected_in_window": rejected,
            "gates_surviving_in_window": offered - rejected,
            "survival_fraction": (round((offered - rejected) / offered, 4)
                                  if offered else None),
            "min_ms": (round(float(patch[finite].min()), 3)
                       if finite.any() else None),
            "max_ms": (round(float(patch[finite].max()), 3)
                       if finite.any() else None),
            # The two gates the couplet was found on, by name: whether each
            # SURVIVED is the question the DA layer cares about, and a
            # rejected gate is None rather than a number.
            "couplet_gates_ms": [None if not np.isfinite(v) else round(v, 3)
                                 for v in pair],
            "couplet_delta_v_ms": (round(abs(pair[1] - pair[0]), 3)
                                   if all(np.isfinite(pair)) else None),
        }
        if sweep.nyquist_velocity_ms:
            in_window, mask = fold_islands(
                result.velocity, sweep.nyquist_velocity_ms, window=window)
            whole, _ = fold_islands(result.velocity,
                                    sweep.nyquist_velocity_ms)
            report["arms"][name]["fold_islands_in_window"] = in_window
            report["arms"][name]["fold_islands_on_the_cut"] = whole
            report["arms"][name]["fold_island_gates"] = [
                {"ray": int(r), "gate": int(g),
                 "value_ms": round(float(result.velocity[r, g]), 3),
                 "raw_ms": round(float(raw[r, g]), 3)}
                for r, g in np.argwhere(mask)[:12]]
    # A small before/after panel around the couplet.  Text, and on purpose:
    # this is a diagnostic for a ruling, not a product plot, and the
    # product renderer is the Rust one -- a matplotlib figure here would be
    # neither.
    panel_rays = [(couplet["ray"] + k) % rows for k in (-3, -2, -1, 0, 1, 2)]
    panel_gates = range(max(0, couplet["gate"] - 4),
                        min(gates, couplet["gate"] + 5))
    panels = {"raw": _panel(raw, panel_rays, panel_gates)}
    for name, result in planes.items():
        panels[name] = _panel(result.velocity, panel_rays, panel_gates)
    report["panels"] = panels
    report["panel_index"] = {
        "rays": [int(r) for r in panel_rays],
        "gates": [int(g) for g in panel_gates],
        "centre": {"ray": int(couplet["ray"]), "gate": int(couplet["gate"])},
    }

    names = list(planes)
    report["pairwise"] = {}
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            a, b = planes[first], planes[second]
            differing = (a.fold != b.fold) & np.isfinite(raw)
            in_window = np.zeros(raw.shape, dtype=bool)
            in_window[window] = True
            report["pairwise"][f"{first} vs {second}"] = {
                "folds_differing_whole_sweep": int(differing.sum()),
                "folds_differing_in_window":
                    int((differing & in_window).sum()),
            }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pack", type=Path, required=True,
                        help="a decoded gpuwm-obs.radar-sweeps pack")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--nx", type=int, default=200)
    parser.add_argument("--ny", type=int, default=200)
    parser.add_argument("--nz", type=int, default=40)
    parser.add_argument("--dx-m", type=float, default=3000.0)
    parser.add_argument("--top-m", type=float, default=16000.0)
    parser.add_argument("--repeat", type=int, default=1,
                        help="run each arm this many times, keep the fastest")
    parser.add_argument("--arm", action="append", default=None,
                        help="restrict to these arms (repeatable)")
    parser.add_argument("--bound-parity", action="store_true",
                        help="digest the shipped configuration against the "
                             "same configuration with the physical bound "
                             "lifted, off the gates the bound refused: "
                             "equal digests are the proof that the bound "
                             "abstains and moves nothing")
    parser.add_argument("--speed-bound-fixture", type=Path, default=None,
                        help="write the speed-bound regression fixture -- "
                             "every gate the solver puts past the bound, "
                             "with the couplet's own gates beside them -- "
                             "to this path (regenerates "
                             "tests/data/dealias_speed_bound_gates.json)")
    args = parser.parse_args(argv)

    from gpuwm.obs.sweeps import read_sweep_pack

    volume = read_sweep_pack(args.pack)
    site = volume.site
    grid = grid_around(site.lat_deg, site.lon_deg, nx=args.nx, ny=args.ny,
                       nz=args.nz, dx_m=args.dx_m, top_m=args.top_m)
    names = args.arm or ["off", "vad-region", "region-global",
                         "region-global+refine",
                         "region-global+refine+nobound"]

    records, fields = [], {}
    for name in names:
        record, mean, filled = run_arm(name, args.pack, grid,
                                       repeat=args.repeat)
        records.append(record)
        fields[name] = (mean, filled)
        print(f"{name:>30}  {record['seconds']:8.3f} s  "
              f"vr cells {record['vr_field']['cells']:>7}  "
              f"|vr| mean {record['vr_field']['abs_mean_ms']:6.2f}  "
              f"max {record['vr_field']['max_ms']:7.2f}  "
              f"speed-rejected {record.get('speed_rejections', 0):>5} "
              f"(bound {record.get('max_speed_ms')})", flush=True)

    diffs = {}
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            diffs[f"{first} vs {second}"] = diff_fields(
                *fields[first], *fields[second])

    search_engine = arm_params("region-global").dealias
    couplet, ranked = find_couplet(args.pack, engine_params=search_engine)
    if args.speed_bound_fixture is not None:
        fixture = speed_bound_fixture(
            args.pack, bound_ms=arm_params("region-global").dealias.max_speed_ms,
            couplet=couplet)
        args.speed_bound_fixture.parent.mkdir(parents=True, exist_ok=True)
        args.speed_bound_fixture.write_text(
            json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
        print(f"\nspeed-bound fixture: "
              f"{len(fixture['gates_beyond_bound'])} of "
              f"{fixture['volume']['gates_offered']} offered gates past "
              f"{fixture['bound_ms']} m/s -> {args.speed_bound_fixture}")
    couplet_report = None
    if couplet is not None:
        print("\ncouplet search (on the fold-resolved region-global field; "
              "one candidate per low cut, ranked):")
        print(f"  {'dV':>7} {'cut':>4} {'elev':>6} {'az':>7} {'km':>7} "
              f"{'sep_km':>7} {'v_in':>7} {'v_out':>7} {'dBZ':>6}")
        for entry in ranked:
            low, high = sorted(entry["resolved_ms"])
            print(f"  {entry['delta_v_ms']:7.1f} {entry['sweep_index']:4d} "
                  f"{entry['elevation_angle_deg']:6.2f} "
                  f"{entry['azimuth_deg']:7.1f} {entry['range_km']:7.1f} "
                  f"{entry['beam_separation_km']:7.3f} {low:7.1f} "
                  f"{high:7.1f} {entry['reflectivity_dbz']:6.1f}")
        couplet_report = couplet_window(
            args.pack, couplet,
            {name: arm_params(name) for name in names})
        couplet_report["ranked_candidates"] = ranked
        print(f"\ncouplet window ({couplet_report['window']['rays']} rays x "
              f"{couplet_report['window']['gates']} gates around it):")
        for name, block in couplet_report["arms"].items():
            print(f"  {name:>22}  offered "
                  f"{block['gates_offered_in_window']:>5}  survived "
                  f"{block['gates_surviving_in_window']:>5} "
                  f"({block['survival_fraction']})  unfolded "
                  f"{block['gates_unfolded_in_window']:>5}  vr "
                  f"[{block['min_ms']}, {block['max_ms']}]  couplet "
                  f"{block['couplet_gates_ms']} dV "
                  f"{block['couplet_delta_v_ms']}  fold-islands "
                  f"{block.get('fold_islands_in_window')} in window / "
                  f"{block.get('fold_islands_on_the_cut')} on the cut")
        for pair, block in couplet_report["pairwise"].items():
            print(f"  {pair}: {block['folds_differing_whole_sweep']} folds "
                  f"differ on the cut, "
                  f"{block['folds_differing_in_window']} in the window")
        index = couplet_report["panel_index"]
        print(f"\ncouplet panel, m/s, rows = rays {index['rays']}, "
              f"columns = gates {index['gates'][0]}-{index['gates'][-1]} "
              f"(centre ray {index['centre']['ray']}, "
              f"gate {index['centre']['gate']}):")
        for name, rows_out in couplet_report["panels"].items():
            print(f"  -- {name}")
            for line in rows_out:
                print(f"     {line}")

    parity = None
    if args.bound_parity:
        parity = bound_parity(args.pack)
        print(f"\nbound parity ({parity['bound_ms']} m/s): "
              f"{parity['gates_refused']} gates refused; off them the folds "
              f"{'MATCH' if parity['fold_planes_match'] else 'DIFFER'}, the "
              f"states "
              f"{'MATCH' if parity['state_planes_match'] else 'DIFFER'}, the "
              f"velocities "
              f"{'MATCH' if parity['velocities_match'] else 'DIFFER'} "
              f"({len(parity['sweeps_disagreeing_off_the_refused_gates'])} "
              "sweeps disagree)")
        print(f"  fold digest {parity['digests']['fold_bounded'][:16]} / "
              f"{parity['digests']['fold_lifted'][:16]}")
        print(f"  vr   digest {parity['digests']['vr_bounded'][:16]} / "
              f"{parity['digests']['vr_lifted'][:16]}")

    payload = {
        "schema": "gpuwm-dealias.engine-compare.v1",
        "pack": str(args.pack),
        "site": site.id,
        "valid_time": str(volume.valid_time),
        "grid": {"nx": grid.nx, "ny": grid.ny, "nz": grid.nz,
                 "dx_m": grid.dx_m},
        "arms": records,
        "gridded_vr_differences": diffs,
        "couplet": couplet,
        "couplet_window": couplet_report,
        "bound_parity": parity,
    }
    text = json.dumps(payload, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    for pair, block in diffs.items():
        print(f"{pair}: {block['cells_differing']} of "
              f"{block['cells_in_both']} shared cells differ"
              + (f", max |delta| {block['max_abs_delta_ms']} m/s"
                 if "max_abs_delta_ms" in block else "")
              + f"; +{block['cells_only_in_second']} / "
              f"-{block['cells_only_in_first']} cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
