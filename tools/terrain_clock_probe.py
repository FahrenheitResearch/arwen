"""The probe behind ``gpuwm/terrain_clock_map.json``, the terrain clock's map.

It measures what the map's ``what`` text says: a bell ridge across the grid
in a uniform cross-ridge wind, integrated through the production ``step()``
with the generated dynamics (the 49-level ladder the domain wizard emits,
the etac the vertical survey derives for the crest, ``epssm`` 0.5,
``smdiv`` 0.1, ``emdiv`` 0.01, ``w_damping`` 1, the slope-tapered
sixth-order filter at 0.12 and the Rayleigh lid from 5 km below the 20 km
top).  A step HELD when the run finished and the peak vertical velocity
stayed under ``4 x wind x slope + 20`` m/s, the slope being the grid-read
steepest slope the rule itself reads.  The domain is periodic, eight rows
deep and ``(wind x seconds + 10 ridge half-widths) / spacing`` columns wide
(never fewer than 96), so the flow cannot come back round to the ridge in
the time it is integrated.

Subcommands:

* ``geometry``: the grid-read slope, etac and thinnest-layer fraction of
  each row, on the CPU.  These are the row keys the map records, so this
  is how a rebuilt probe proves it builds the map's own ridges.
* ``cell``: one ridge, one wind, one step, for a stated time; prints held
  and the peak vertical velocity.  How a single map entry is re-checked.
* ``extend``: longer steps on rows whose entry is the map's longest
  measured step.  Each wind is tried from the step the weaker wind held,
  down the rung list, over the map's half hour; a wind whose entry is
  already below the map's longest step keeps it, and so does every
  stronger wind.  Every step held that way is then run three hours, and
  where one stops the entry walks down the rungs (three hours each) to
  the longest that held, or back to the map's own entry.  Writes the
  measured rows as JSON.
* ``merge``: writes an ``extend`` result into the map: the row's entries
  at the measured winds, and ``top_s_per_km``, the longest step tried at
  each wind, which is what the rule treats as "held everything tried".
* ``fixed``: every listed step at every listed substep count on the listed
  ridges and winds, each for the stated seconds (three hours unless told),
  one JSON line per run.  Every pair is run, so a stop is seen at every
  step it happens at, not only above the first that held.
* ``adaptive``: the same ridge on the production adaptive clock instead of
  a fixed step: :class:`gpuwm.core.adaptive_timestep.AdaptiveTimestepController`
  fed the dycore's own WRF CFL reduction after every step, from a stated
  first step up to each listed ``max_time_step``, WRF's ``3 x dx`` floor
  under it, landing on each hour as ``step_to_output_time`` does, and the
  substep count :func:`gpuwm.core.adaptive_clock.adaptive_sound_steps`
  derives from the live step.  Held means the same as for a fixed step.
  One JSON line per run, with the steps the clock took.
* ``adaptive-extend``: a map row's adaptive entries: per wind, the
  longest ``max_time_step`` from a ladder (s/km, longest first) that held
  three hours at every CFL target pair of :data:`ADAPTIVE_TARGETS`, each
  wind tried from the step the weaker wind held.  ``merge`` writes them on
  the row's four-substep line with the map's ``adaptive`` block.

Needs CuPy for ``cell``, ``extend``, ``fixed``, ``adaptive`` and
``adaptive-extend``; ``geometry`` and ``merge`` do not.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAP_PATH = Path(__file__).resolve().parents[1] / "gpuwm" / "terrain_clock_map.json"

#: The map's half hour.
SECONDS = 1800.0
#: The three-hour check every extended entry passes.
CHECK_SECONDS = 10800.0
#: Rows of the periodic probe domain (the ridge is uniform along them).
ROWS = 8
#: Fewest columns of a probe domain.
MIN_COLUMNS = 96


@dataclass(frozen=True)
class Ridge:
    dx: float
    crest: float
    ridge_slope: float

    @property
    def halfwidth(self) -> float:
        """The half-width whose steepest point has ``ridge_slope``."""
        return (3.0 * math.sqrt(3.0) / 8.0) * self.crest / self.ridge_slope

    def columns(self, wind: float, seconds: float) -> int:
        """Columns the flow cannot wrap in ``seconds`` (always even, so the
        crest sits on a face exactly as on the map's domains)."""
        span = float(wind) * float(seconds) + 10.0 * self.halfwidth
        n = max(MIN_COLUMNS, int(math.ceil(span / self.dx)))
        return n + (n % 2)


def _eta():
    from gpuwm.domain_wizard import _ETA_LEVELS

    return tuple(float(v) for v in _ETA_LEVELS)


def _config(ridge: Ridge, *, nx: int, dt: float, sound_steps: int,
            etac: float, seconds: float, zadvect_implicit: int = 0):
    from gpuwm.config import RunConfig

    eta = _eta()
    return RunConfig(
        nx=int(nx), ny=ROWS, nz=len(eta) - 1, dx=ridge.dx, dy=ridge.dx,
        ztop=20000.0, dt=float(dt), run_seconds=float(seconds),
        time_step_sound=int(sound_steps), epssm=0.5, smdiv=0.1, emdiv=0.01,
        damp_opt=3, zdamp=5000.0, dampcoef=0.2, w_damping=1, terrain_opt=1,
        hill_height=ridge.crest, hill_halfwidth=ridge.halfwidth,
        hybrid_opt=2, etac=float(etac), top_lid=False, diff_6th_opt=2,
        diff_6th_factor=0.12, diff_6th_slopeopt=1, h_sca_adv_order=5,
        eta_levels=eta, zadvect_implicit=int(zadvect_implicit))


def _sounding(z):
    from gpuwm.core import constants as c

    return 290.0 * np.exp(1.0e-4 * np.asarray(z, dtype=np.float64) / c.G)


def _base(cfg, etac, terrain):
    from gpuwm.core.grid import make_base_state, make_vertical_coord

    coord = make_vertical_coord(cfg.nz, hybrid_opt=2, etac=float(etac),
                                eta_levels=np.asarray(cfg.eta_levels))
    base = make_base_state(coord, _sounding, p_surf=cfg.p_surf,
                           ztop=cfg.ztop, terrain_z=terrain)
    return coord, base


def geometry(ridge: Ridge, *, nx: int | None = None) -> dict:
    """The row keys of ``ridge``: grid-read slope, etac, thinnest layer."""
    from gpuwm.acoustic_adaptation import steepest_slope
    from gpuwm.core.terrain import bell_hill
    from gpuwm.vertical_adaptation import (TerrainField,
                                           survey_vertical_coordinate)

    nx = ridge.columns(0.0, 0.0) if nx is None else int(nx)
    cfg = _config(ridge, nx=nx, dt=1.0, sound_steps=4, etac=0.2,
                  seconds=1.0)
    terrain = bell_hill(cfg)
    coord, _base_state = _base(cfg, 0.2, terrain)
    survey = survey_vertical_coordinate(
        np.asarray(cfg.eta_levels, dtype=np.float64), 2, 0.2,
        float(coord.p_top), [TerrainField("ridge", terrain)])
    etac = 0.2 if survey is None or survey.etac is None else survey.etac
    fraction = None if survey is None else survey.layer_fraction
    slope = steepest_slope(terrain, cfg.dx, cfg.dy, label="ridge").slope
    return {"slope": round(float(slope), 4), "etac": round(float(etac), 3),
            "etac_exact": float(etac),
            "thinnest_layer_fraction": (None if fraction is None
                                        else round(float(fraction), 4))}


def run_cell(ridge: Ridge, *, wind: float, per_km: float, sound_steps: int,
             seconds: float, etac: float | None = None,
             zadvect_implicit: int = 0) -> dict:
    """One ridge, one wind, one step through the production ``step()``."""
    import cupy as cp

    from gpuwm.acoustic_adaptation import steepest_slope
    from gpuwm.core.dycore import set_w_surface, step
    from gpuwm.core.state import init_at_rest
    from gpuwm.core.terrain import bell_hill

    if etac is None:
        etac = geometry(ridge)["etac_exact"]
    dt = float(per_km) * ridge.dx / 1000.0
    nx = ridge.columns(wind, seconds)
    cfg = _config(ridge, nx=nx, dt=dt, sound_steps=sound_steps, etac=etac,
                  seconds=seconds, zadvect_implicit=zadvect_implicit)
    terrain = bell_hill(cfg)
    slope = steepest_slope(terrain, cfg.dx, cfg.dy, label="ridge").slope
    bound = 4.0 * float(wind) * float(slope) + 20.0
    coord, base = _base(cfg, etac, terrain)
    state = init_at_rest(cfg, coord, base, terrain_z=base.terrain_z)
    state.u[...] = cp.float32(wind)
    set_w_surface(state, cfg)
    state.w[1:] = state.w[0][None] * (state.znw[1:, None, None] ** 2)
    steps = int(round(float(seconds) / dt))
    peak = 0.0
    held = True
    started = time.perf_counter()
    stopped_at = None
    step_seconds = []
    for index in range(steps):
        cp.cuda.Device().synchronize()
        tick = time.perf_counter()
        step(state, cfg)
        value = float(cp.abs(state.w).max())
        step_seconds.append(time.perf_counter() - tick)
        if not math.isfinite(value) or value > bound:
            held = False
            peak = value if math.isfinite(value) else float("inf")
            stopped_at = (index + 1) * dt
            break
        peak = max(peak, value)
    del state
    cp.get_default_memory_pool().free_all_blocks()
    timed = sorted(step_seconds[1:]) or step_seconds
    return {"held": held, "peak_w": peak, "bound": bound, "dt": dt,
            "per_km": float(per_km), "wind": float(wind), "nx": nx,
            "steps": steps, "stopped_at_s": stopped_at,
            "zadvect_implicit": int(zadvect_implicit),
            "median_step_ms": (round(1000.0 * timed[len(timed) // 2], 3)
                               if timed else None),
            "wall_s": round(time.perf_counter() - started, 2)}


def run_adaptive_cell(ridge: Ridge, *, wind: float, max_step: float,
                      start_step: float, seconds: float,
                      target_cfl: float = 1.2, target_hcfl: float = 0.84,
                      increase_pct: int = 5, sound_floor: int = 0,
                      alarm_s: float = 3600.0,
                      etac: float | None = None) -> dict:
    """One ridge, one wind, on the production adaptive clock.

    The loop is :class:`gpuwm.core.adaptive_clock.AdaptiveClockDriver`'s
    for a single domain: the controller reads the CFL the last step
    measured (the dycore's ``w_cfl_stat`` fold, switched on as the driver
    switches it on), proposes the next step, lands on every ``alarm_s`` and
    on the run's end, and the step runs with the substep count the clock
    derives from it, raised to ``sound_floor`` (a terrain rule's
    ``min_time_step_sound``).  ``min_time_step`` is WRF's ``3 x dx``
    fill-in, as on a run that leaves it at -1.
    """
    from dataclasses import replace
    from fractions import Fraction

    import cupy as cp

    from gpuwm.acoustic_adaptation import steepest_slope
    from gpuwm.core.adaptive_clock import (wrf_default_clamps,
                                           wrf_num_sound_steps)
    from gpuwm.core.adaptive_timestep import AdaptiveTimestepController
    from gpuwm.core.dycore import (enable_wrf_cfl_recording,
                                   reset_wrf_cfl_recording, set_w_surface,
                                   step, take_wrf_cfl)
    from gpuwm.core.state import init_at_rest
    from gpuwm.core.terrain import bell_hill

    if etac is None:
        etac = geometry(ridge)["etac_exact"]
    precision = 100

    def lattice(value: Fraction) -> Fraction:
        return Fraction(math.floor(value * precision), precision)

    def count(dt: Fraction) -> int:
        return max(wrf_num_sound_steps(float(dt), ridge.dx, ridge.dx),
                   int(sound_floor))

    start = Fraction(str(start_step))
    upper = Fraction(str(max_step))
    lower = Fraction(wrf_default_clamps(ridge.dx, ridge.dx)[2])
    nx = ridge.columns(wind, seconds)
    cfg = _config(ridge, nx=nx, dt=float(start), sound_steps=count(start),
                  etac=etac, seconds=seconds)
    terrain = bell_hill(cfg)
    slope = steepest_slope(terrain, cfg.dx, cfg.dy, label="ridge").slope
    bound = 4.0 * float(wind) * float(slope) + 20.0
    coord, base = _base(cfg, etac, terrain)
    state = init_at_rest(cfg, coord, base, terrain_z=base.terrain_z)
    state.u[...] = cp.float32(wind)
    set_w_surface(state, cfg)
    state.w[1:] = state.w[0][None] * (state.znw[1:, None, None] ** 2)
    ctl = AdaptiveTimestepController(
        target_cfl=float(target_cfl), target_hcfl=float(target_hcfl),
        max_step_increase_pct=int(increase_pct), starting_dt=start,
        min_dt=lower, max_dt=upper)
    total = Fraction(str(seconds))
    alarm = Fraction(str(alarm_s))
    elapsed = Fraction(0)
    applied_before = None
    steps = []
    counts: dict[int, int] = {}
    peak = 0.0
    held = True
    stopped_at = None
    started = time.perf_counter()
    reset_wrf_cfl_recording()
    enable_wrf_cfl_recording()
    try:
        while elapsed < total:
            vert, horiz = take_wrf_cfl(int(cfg.grid_id))
            if (ctl.started and not ctl.stepping_to_time
                    and applied_before is not None
                    and applied_before != ctl.last_dt):
                scale = float(ctl.last_dt / applied_before)
                vert, horiz = vert * scale, horiz * scale
            dt = (ctl.next_dt(max_vert_cfl=vert, max_horiz_cfl=horiz)
                  if ctl.started else ctl.first_step())
            to_alarm = alarm - (elapsed % alarm)
            dt, stepping = ctl.step_to_time(dt, to_alarm, quantise=lattice)
            left = total - elapsed
            if 0 < left < dt:
                dt, stepping = left, True
            proposed = dt
            dt = lattice(dt)
            if dt <= 0:
                raise RuntimeError(f"the clock proposed {proposed} s")
            sound = count(dt)
            cfg = replace(cfg, dt=float(dt), time_step_sound=sound)
            step(state, cfg)
            applied_before = dt
            ctl.accept(proposed, max_vert_cfl=vert, max_horiz_cfl=horiz,
                       stepping_to_time=stepping)
            elapsed += dt
            steps.append(float(dt))
            counts[sound] = counts.get(sound, 0) + 1
            value = float(cp.abs(state.w).max())
            if not math.isfinite(value) or value > bound:
                held = False
                peak = value if math.isfinite(value) else float("inf")
                stopped_at = float(elapsed)
                break
            peak = max(peak, value)
    finally:
        reset_wrf_cfl_recording()
        del state
        cp.get_default_memory_pool().free_all_blocks()
    return {"held": held, "peak_w": peak, "bound": bound,
            "max_time_step_s": float(upper), "start_step_s": float(start),
            "min_time_step_s": float(lower), "wind": float(wind), "nx": nx,
            "target_cfl": float(target_cfl),
            "target_hcfl": float(target_hcfl),
            "increase_pct": int(increase_pct),
            "sound_floor": int(sound_floor), "steps": len(steps),
            "mean_step_s": (float(elapsed) / len(steps) if steps else None),
            "longest_step_s": max(steps) if steps else None,
            "shortest_step_s": min(steps) if steps else None,
            "steps_at_max": sum(1 for v in steps
                                if abs(v - float(upper)) < 1e-9),
            "sound_steps_taken": {str(k): v for k, v in sorted(
                counts.items())},
            "stopped_at_s": stopped_at,
            "wall_s": round(time.perf_counter() - started, 2)}


def _map_row(document, ridge: Ridge, sound_steps: int):
    for row in document["rows"]:
        if (float(row["dx_m"]) == ridge.dx
                and float(row["crest_m"]) == ridge.crest
                and abs(float(row["ridge_slope"]) - ridge.ridge_slope) < 1e-9
                and int(row["sound_steps"]) == int(sound_steps)):
            return row
    raise SystemExit(f"no map row for {ridge} on {sound_steps} substeps")


def extend_row(ridge: Ridge, sound_steps: int, winds, rungs, document,
               log=print) -> dict:
    """Longer steps on one map row, at ``winds``, from ``rungs`` (s/km,
    longest first); the map's own entries stand where they are below its
    longest measured step."""
    row = _map_row(document, ridge, sound_steps)
    top = float(document["ladder_s_per_km"][0])
    map_winds = [float(w) for w in document["winds_m_s"]]
    shape = geometry(ridge)
    old = dict(zip(map_winds, row["stable_s_per_km"]))
    runs = []
    half_hour = {}
    start = 0
    for wind in winds:
        entry = old[float(wind)]
        if start is None or entry is None or float(entry) < top:
            half_hour[wind] = entry
            start = None
            continue
        held_index = None
        for index in range(start, len(rungs)):
            result = run_cell(ridge, wind=wind, per_km=rungs[index],
                              sound_steps=sound_steps, seconds=SECONDS,
                              etac=shape["etac_exact"])
            runs.append({"check": "half hour", **result})
            log(f"  {ridge.crest:.0f} m slope {ridge.ridge_slope:g} x{sound_steps} "
                f"{wind:g} m/s {rungs[index]:.3f} s/km: "
                f"{'held' if result['held'] else 'stopped'} "
                f"(peak w {result['peak_w']:.1f}, bound {result['bound']:.1f}, "
                f"{result['wall_s']} s)")
            if result["held"]:
                held_index = index
                break
        if held_index is None:
            half_hour[wind] = entry
            start = None
        else:
            half_hour[wind] = rungs[held_index]
            start = held_index
    final = {}
    ceiling = None
    for wind in winds:
        value = half_hour[wind]
        if value is None or float(value) <= top:
            final[wind] = value
            ceiling = top if value is not None else ceiling
            continue
        candidates = [r for r in rungs if r <= float(value) + 1e-9
                      and (ceiling is None or r <= ceiling + 1e-9)]
        chosen = None
        for rung in candidates:
            result = run_cell(ridge, wind=wind, per_km=rung,
                              sound_steps=sound_steps, seconds=CHECK_SECONDS,
                              etac=shape["etac_exact"])
            runs.append({"check": "three hours", **result})
            log(f"  3 h {ridge.crest:.0f} m slope {ridge.ridge_slope:g} "
                f"x{sound_steps} {wind:g} m/s {rung:.3f} s/km: "
                f"{'held' if result['held'] else 'stopped'} "
                f"(peak w {result['peak_w']:.1f}, {result['wall_s']} s)")
            if result["held"]:
                chosen = rung
                break
        final[wind] = chosen if chosen is not None else old[float(wind)]
        ceiling = float(final[wind])
    stable = []
    tops = []
    measured = {float(w) for w in winds}
    for wind in map_winds:
        if wind in measured:
            stable.append(final[wind])
            tops.append(float(rungs[0]))
        else:
            stable.append(old[wind])
            tops.append(top)
    return {"dx_m": ridge.dx, "crest_m": ridge.crest,
            "ridge_slope": ridge.ridge_slope, "sound_steps": int(sound_steps),
            "slope": shape["slope"], "etac": shape["etac"],
            "thinnest_layer_fraction": shape["thinnest_layer_fraction"],
            "map_slope": row["slope"], "map_etac": row["etac"],
            "map_thinnest_layer_fraction": row["thinnest_layer_fraction"],
            "stable_s_per_km": stable, "top_s_per_km": tops,
            "half_hour": [half_hour.get(float(w)) for w in map_winds
                          if float(w) in measured],
            "runs": runs}


#: The adaptive clock's settings every adaptive entry held under: WRF's
#: default CFL targets and the longer 1.4 / 0.98 pair, each with the 5
#: percent growth bound, from a first step of the ladder's top.
ADAPTIVE_TARGETS = ((1.2, 0.84), (1.4, 0.98))
ADAPTIVE_INCREASE_PCT = 5


def adaptive_extend_row(ridge: Ridge, winds, ladder, document,
                        log=print) -> dict:
    """The longest ``max_time_step`` (s/km, from ``ladder``, longest first)
    that held three hours on the adaptive clock at every target pair of
    :data:`ADAPTIVE_TARGETS`, per wind.  Each wind is tried from the step
    the weaker wind held; past a wind that held none, the stronger winds
    are left unmeasured."""
    _map_row(document, ridge, 4)
    top = float(document["ladder_s_per_km"][0])
    map_winds = [float(w) for w in document["winds_m_s"]]
    shape = geometry(ridge)
    km = ridge.dx / 1000.0
    runs = []
    held = {}
    start = 0
    for wind in winds:
        if start is None:
            break
        chosen = None
        for index in range(start, len(ladder)):
            every = True
            for target_cfl, target_hcfl in ADAPTIVE_TARGETS:
                result = run_adaptive_cell(
                    ridge, wind=wind, max_step=round(ladder[index] * km, 2),
                    start_step=round(top * km, 2), seconds=CHECK_SECONDS,
                    target_cfl=target_cfl, target_hcfl=target_hcfl,
                    increase_pct=ADAPTIVE_INCREASE_PCT,
                    etac=shape["etac_exact"])
                runs.append({"per_km": ladder[index], **result})
                log(f"  adaptive {ridge.dx:.0f} m {ridge.crest:.0f} m slope "
                    f"{ridge.ridge_slope:g} {wind:g} m/s max "
                    f"{ladder[index]:.3f} s/km cfl {target_cfl}/"
                    f"{target_hcfl}: "
                    f"{'held' if result['held'] else 'stopped'} "
                    f"(peak w {result['peak_w']:.1f}, mean step "
                    f"{result['mean_step_s']}, {result['wall_s']} s)")
                if not result["held"]:
                    every = False
                    break
            if every:
                chosen = index
                break
        if chosen is None:
            held[float(wind)] = None
            start = None
        else:
            held[float(wind)] = float(ladder[chosen])
            start = chosen
    entries = [held.get(w) for w in map_winds]
    tried = [float(ladder[0]) if w in held else None for w in map_winds]
    return {"dx_m": ridge.dx, "crest_m": ridge.crest,
            "ridge_slope": ridge.ridge_slope, "sound_steps": 4,
            "slope": shape["slope"], "etac": shape["etac"],
            "adaptive_s_per_km": entries, "adaptive_top_s_per_km": tried,
            "runs": runs}


#: The map's account of its adaptive entries (its ``adaptive`` block).
ADAPTIVE_WHAT = (
    "The same ridges on the production adaptive clock: "
    "gpuwm.core.adaptive_timestep.AdaptiveTimestepController fed the "
    "dycore's own WRF CFL after every step, WRF's substep count from the "
    "live step (map factor 1), min_time_step 3 x dx, landing on every "
    "hour, from a first step of the ladder's top (5 s/km), for the stated "
    "seconds under the same held criterion.  adaptive_s_per_km is, per "
    "wind, the longest max_time_step from ladder_s_per_km (longest first) "
    "that held at every CFL target pair in targets, with growth bounded at "
    "max_step_increase_pct; each wind was tried from the step the weaker "
    "wind held, and past a wind that held none the stronger winds were not "
    "run.  adaptive_top_s_per_km is the longest max_time_step tried at "
    "each wind (null where the wind was not run); an entry below it saw a "
    "longer one stop, and a null entry under a tried range means none "
    "tried held.  Four-substep rows only: the adaptive clock reads the map "
    "at the count it takes at its shortest steps.")


def adaptive_block(measured: dict) -> dict:
    """The ``adaptive`` block an ``adaptive-extend`` result measured."""
    return {"what": ADAPTIVE_WHAT,
            "targets": [list(pair) for pair in measured["targets"]],
            "max_step_increase_pct": int(measured["increase_pct"]),
            "ladder_s_per_km": list(measured["ladder_s_per_km"]),
            "seconds": float(measured["check_seconds"])}


def merge(document: dict, rows, *, what: str | None = None,
          adaptive: dict | None = None) -> dict:
    """Write measured rows into the map document (in place, returned).
    A fixed-step row replaces the row's entries and tried range; an
    adaptive row adds its adaptive entries and their tried range, and
    ``adaptive`` is the block saying how they were measured."""
    if adaptive is not None:
        if "adaptive" in document and document["adaptive"] != adaptive:
            raise SystemExit("adaptive entries measured another way than "
                             "the map's own; a map holds one measurement")
        document["adaptive"] = adaptive
    for measured in rows:
        ridge = Ridge(float(measured["dx_m"]), float(measured["crest_m"]),
                      float(measured["ridge_slope"]))
        row = _map_row(document, ridge, int(measured["sound_steps"]))
        if "adaptive_s_per_km" in measured:
            row["adaptive_s_per_km"] = list(measured["adaptive_s_per_km"])
            row["adaptive_top_s_per_km"] = list(
                measured["adaptive_top_s_per_km"])
            continue
        row["stable_s_per_km"] = list(measured["stable_s_per_km"])
        row["top_s_per_km"] = list(measured["top_s_per_km"])
    if what is not None:
        document["what"] = what
    return document


def write_map(document: dict, path: Path = MAP_PATH) -> None:
    """The map's own layout: one row per line."""
    head = {key: value for key, value in document.items() if key != "rows"}
    lines = ["{"]
    for key, value in head.items():
        lines.append(f" {json.dumps(key)}: "
                     f"{json.dumps(value, indent=1).replace(chr(10), chr(10) + ' ')},")
    lines.append(' "rows": [')
    rows = [" " + " " + json.dumps(row) for row in document["rows"]]
    lines.append(",\n".join(rows))
    lines.append(" ]")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _floats(text):
    return [float(v) for v in str(text).split(",") if v.strip()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("geometry")
    g.add_argument("--dx", type=float, required=True)
    g.add_argument("--crests", type=_floats, required=True)
    g.add_argument("--ridge-slopes", type=_floats, required=True)
    c = sub.add_parser("cell")
    c.add_argument("--dx", type=float, required=True)
    c.add_argument("--crest", type=float, required=True)
    c.add_argument("--ridge-slope", type=float, required=True)
    c.add_argument("--wind", type=float, required=True)
    c.add_argument("--per-km", type=float, required=True)
    c.add_argument("--sound-steps", type=int, required=True)
    c.add_argument("--seconds", type=float, default=SECONDS)
    c.add_argument("--zadvect-implicit", type=int, default=0,
                   help="WRF's implicit-explicit vertical advection (1)")
    e = sub.add_parser("extend")
    e.add_argument("--dx", type=float, required=True)
    e.add_argument("--crests", type=_floats, required=True)
    e.add_argument("--ridge-slopes", type=_floats, required=True)
    e.add_argument("--winds", type=_floats, required=True)
    e.add_argument("--sound-steps", type=_floats, default=[4, 6])
    e.add_argument("--rung-seconds", type=_floats, required=True,
                   help="steps to try, in seconds, longest first")
    e.add_argument("--out", type=Path, required=True)
    m = sub.add_parser("merge")
    m.add_argument("measured", type=Path, nargs="+")
    m.add_argument("--what", type=Path, default=None,
                   help="a text file holding the map's new 'what' text")
    a = sub.add_parser("adaptive-extend")
    a.add_argument("--dx", type=float, required=True)
    a.add_argument("--crests", type=_floats, required=True)
    a.add_argument("--ridge-slopes", type=_floats, required=True)
    a.add_argument("--winds", type=_floats, required=True)
    a.add_argument("--ladder", type=_floats, required=True,
                   help="max_time_step values in s/km, longest first")
    a.add_argument("--out", type=Path, required=True)
    for name in ("fixed", "adaptive"):
        s = sub.add_parser(name)
        s.add_argument("--dx", type=float, required=True)
        s.add_argument("--crest", type=float, required=True)
        s.add_argument("--ridge-slopes", type=_floats, required=True)
        s.add_argument("--winds", type=_floats, required=True)
        s.add_argument("--seconds", type=float, default=CHECK_SECONDS)
        s.add_argument("--out", type=Path, required=True,
                       help="JSON lines, appended")
        if name == "fixed":
            s.add_argument("--steps", type=_floats, required=True,
                           help="fixed steps in seconds")
            s.add_argument("--sound-steps", type=_floats, default=[4, 6])
        else:
            s.add_argument("--max-steps", type=_floats, required=True,
                           help="max_time_step values in seconds")
            s.add_argument("--start-step", type=float, required=True,
                           help="the clock's first step in seconds")
            s.add_argument("--target-cfl", type=float, default=1.2)
            s.add_argument("--target-hcfl", type=float, default=0.84)
            s.add_argument("--increase-pct", type=int, default=5)
            s.add_argument("--sound-floor", type=int, default=0)
    args = parser.parse_args(argv)

    def append(path: Path, record: dict) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    if args.command == "adaptive-extend":
        document = json.loads(MAP_PATH.read_text(encoding="utf-8"))
        rows = []
        for crest in args.crests:
            for slope in args.ridge_slopes:
                rows.append(adaptive_extend_row(
                    Ridge(args.dx, crest, slope), args.winds, args.ladder,
                    document, log=lambda text: print(text, flush=True)))
                args.out.write_text(json.dumps(
                    {"dx_m": args.dx, "winds_m_s": args.winds,
                     "ladder_s_per_km": args.ladder,
                     "targets": [list(pair) for pair in ADAPTIVE_TARGETS],
                     "increase_pct": ADAPTIVE_INCREASE_PCT,
                     "check_seconds": CHECK_SECONDS, "rows": rows},
                    indent=1), encoding="utf-8")
        return 0
    if args.command in ("fixed", "adaptive"):
        for ridge_slope in args.ridge_slopes:
            ridge = Ridge(args.dx, args.crest, ridge_slope)
            shape = geometry(ridge)
            key = {"dx_m": ridge.dx, "crest_m": ridge.crest,
                   "ridge_slope": ridge_slope, "slope": shape["slope"],
                   "etac": shape["etac"], "seconds": args.seconds}
            for wind in args.winds:
                if args.command == "fixed":
                    for count in args.sound_steps:
                        for seconds_step in args.steps:
                            result = run_cell(
                                ridge, wind=wind,
                                per_km=seconds_step * 1000.0 / ridge.dx,
                                sound_steps=int(count),
                                seconds=args.seconds,
                                etac=shape["etac_exact"])
                            record = {"arm": "fixed", **key,
                                      "sound_steps": int(count),
                                      "step_s": seconds_step, **result}
                            append(args.out, record)
                            print(json.dumps(record), flush=True)
                    continue
                for upper in args.max_steps:
                    result = run_adaptive_cell(
                        ridge, wind=wind, max_step=upper,
                        start_step=args.start_step, seconds=args.seconds,
                        target_cfl=args.target_cfl,
                        target_hcfl=args.target_hcfl,
                        increase_pct=args.increase_pct,
                        sound_floor=args.sound_floor,
                        etac=shape["etac_exact"])
                    record = {"arm": "adaptive", **key, **result}
                    append(args.out, record)
                    print(json.dumps(record), flush=True)
        return 0

    if args.command == "geometry":
        for crest in args.crests:
            for slope in args.ridge_slopes:
                print(json.dumps({"dx_m": args.dx, "crest_m": crest,
                                  "ridge_slope": slope,
                                  **geometry(Ridge(args.dx, crest, slope))}))
        return 0
    if args.command == "cell":
        print(json.dumps(run_cell(
            Ridge(args.dx, args.crest, args.ridge_slope), wind=args.wind,
            per_km=args.per_km, sound_steps=args.sound_steps,
            seconds=args.seconds, zadvect_implicit=args.zadvect_implicit)))
        return 0
    if args.command == "extend":
        document = json.loads(MAP_PATH.read_text(encoding="utf-8"))
        rungs = [s * 1000.0 / args.dx for s in args.rung_seconds]
        rows = []
        for crest in args.crests:
            for slope in args.ridge_slopes:
                for count in args.sound_steps:
                    rows.append(extend_row(
                        Ridge(args.dx, crest, slope), int(count), args.winds,
                        rungs, document,
                        log=lambda text: print(text, flush=True)))
                    args.out.write_text(json.dumps(
                        {"dx_m": args.dx, "winds_m_s": args.winds,
                         "rung_seconds": args.rung_seconds,
                         "rungs_s_per_km": rungs, "seconds": SECONDS,
                         "check_seconds": CHECK_SECONDS, "rows": rows},
                        indent=1), encoding="utf-8")
        return 0
    if args.command == "merge":
        document = json.loads(MAP_PATH.read_text(encoding="utf-8"))
        rows = []
        blocks = []
        for path in args.measured:
            measured = json.loads(path.read_text(encoding="utf-8"))
            rows.extend(measured["rows"])
            if "targets" in measured:
                blocks.append(adaptive_block(measured))
        if any(block != blocks[0] for block in blocks):
            raise SystemExit("the adaptive results were measured different "
                             "ways; merge one measurement at a time")
        what = (None if args.what is None
                else args.what.read_text(encoding="utf-8").strip())
        write_map(merge(document, rows, what=what,
                        adaptive=blocks[0] if blocks else None))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
