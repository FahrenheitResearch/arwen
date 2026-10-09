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


#: The dynamics settings a probe runs under.  ``generated`` is what the map
#: was measured at (the domain wizard's generated dynamics).  ``ncar`` is
#: NCAR's v4.4 CONUS benchmark namelist as WOOF's WRF-input door resolves
#: it (receipts of the 2026-10-06 benchmark check: epssm unset, so WRF's
#: 0.1; diff_6th_opt 0; damp_opt 3, zdamp 5000, dampcoef 0.2; w_damping
#: 1).  The breakage the second arm exists for: the map that halved NCAR's
#: steps was measured more damped (epssm 0.5, sixth-order filter on) than
#: the runs it judged, so its entries said nothing about them.
SETTINGS = {
    "generated": {"epssm": 0.5, "smdiv": 0.1, "emdiv": 0.01, "damp_opt": 3,
                  "zdamp": 5000.0, "dampcoef": 0.2, "w_damping": 1,
                  "diff_6th_opt": 2, "diff_6th_factor": 0.12,
                  "diff_6th_slopeopt": 1},
    "ncar": {"epssm": 0.1, "smdiv": 0.1, "emdiv": 0.01, "damp_opt": 3,
             "zdamp": 5000.0, "dampcoef": 0.2, "w_damping": 1,
             "diff_6th_opt": 0, "diff_6th_factor": 0.12,
             "diff_6th_slopeopt": 0},
}


def _config(ridge: Ridge, *, nx: int, dt: float, sound_steps: int,
            etac: float, seconds: float, zadvect_implicit: int = 0,
            settings: str = "generated"):
    from gpuwm.config import RunConfig

    eta = _eta()
    return RunConfig(
        nx=int(nx), ny=ROWS, nz=len(eta) - 1, dx=ridge.dx, dy=ridge.dx,
        ztop=20000.0, dt=float(dt), run_seconds=float(seconds),
        time_step_sound=int(sound_steps), terrain_opt=1,
        hill_height=ridge.crest, hill_halfwidth=ridge.halfwidth,
        hybrid_opt=2, etac=float(etac), top_lid=False, h_sca_adv_order=5,
        eta_levels=eta, zadvect_implicit=int(zadvect_implicit),
        **SETTINGS[settings])


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


#: Under ``criterion="blowup"`` a run stops where w passes this many times
#: the map's held bound (or is non-finite), so no less than 200 m/s, a
#: vertical velocity no flow over these ridges carries.  Named breakage:
#: in three-hour runs the map's bound stopped cells at every step down to
#: 3 s/km with w only 1.00-1.06 x the bound at the stop and the WRF
#: vertical Courant number falling with the step (2 km, 4.5 km crest,
#: slope 0.09-0.15, 40-50 m/s, both settings arms, 2026-10-07): a wave
#: grown past the bound and a run going unstable stop alike there.  This
#: criterion runs such a cell the whole three hours to tell them apart;
#: ``hourly_peak_w`` says whether w was still growing at the end.
BLOWUP_FACTOR = 10.0

#: Under ``criterion="blowup"`` (and "steady") a run also stops, and so
#: does not hold, where peak |w| reaches this many m/s: a probe run holds
#: only where it stayed finite AND its peak vertical velocity stayed under
#: it (lead decision 3 after step 5, 2026-10-07).  Breakage it prevents:
#: under :data:`BLOWUP_FACTOR` alone a run counted as held at any peak w
#: up to ten times the old bound (200 m/s and more), and entries rested
#: on runs that reached 108 to 388 m/s: LA Santa Ana's parent-grid cap
#: came from a 70 m/s cell held at 119 m/s (3 km, 4500 m crest, ridge
#: 0.40, 4 substeps, 4.5 s/km), and two adaptive entries on 203 to
#: 276 m/s runs (fix/step5/RULINGS.md ruling 6).  Ten times the bound is
#: never under 200 m/s, so this is the limit that acts.
BLOWUP_PEAK_W = 100.0


def stops(value: float, bound: float, criterion: str) -> bool:
    """Whether a run's ``|w|`` of ``value`` stops it: non-finite, past the
    map's ``bound`` under "map", and under "blowup" or "steady" past
    :data:`BLOWUP_FACTOR` times it or at :data:`BLOWUP_PEAK_W` and up."""
    if not math.isfinite(value):
        return True
    if criterion == "map":
        return value > bound
    return value > BLOWUP_FACTOR * bound or value >= BLOWUP_PEAK_W


def held_under_peak_rule(run: dict) -> bool:
    """A recorded blow-up-criterion run re-read under
    :data:`BLOWUP_PEAK_W`: held only where it held and its peak |w| stayed
    under that.  A run that held past it is the run the probe now stops
    when w first reaches it, so the two say the same."""
    peak = run.get("peak_w")
    return bool(run["held"]) and peak is not None and math.isfinite(
        float(peak)) and float(peak) < BLOWUP_PEAK_W

#: Under ``criterion="steady"`` a run that stayed finite and under
#: :data:`BLOWUP_FACTOR` times the bound still does not hold where its last
#: hour's peak w is more than this factor over the hour before: it is
#: still growing when the probe stops, and may run away past the probe's
#: three hours.  Measured: over the 163 finite blow-up re-runs of
#: 2026-10-07, the last-hour growth is at most 1.03 x on every steady cell
#: and 1.13 to 1.47 x on the five still growing (2 km 3000 m crest slope
#: 0.095-0.1 at 30 m/s; 3 km 3000 m crest slope 0.35-0.4 at 60 m/s;
#: 12 km 4500 m crest slope 0.3 at 30 m/s), so 1.10 sat in the gap THERE.
#: The full sweep under this criterion (236 rows, 2026-10-07 03:05-03:27Z,
#: CLOCK-CHECK-NCAR-2026-10-06/fix/step2/rows-steady) showed it does not
#: separate anything: on gentle 2 and 3 km ridges at 20 and 30 m/s the
#: mountain wave is still spinning up from the impulsive start in hour 3
#: (peak w 6 -> 18 -> 21 m/s), adjacent steps fall either side of 1.10
#: with the same wave (21.1 and 21.4 m/s "growing" at 6.5 and 6.0 s/km,
#: 19.7 "steady" at 5.5), so the walk's entry is set by noise around the
#: threshold.  The terrain clock's local-face candidate does NOT use it:
#: its rows read the blow-up criterion over every run
#: (tools/terrain_clock_candidate.py --evidence).  The option stays only
#: because its sweep's runs are evidence under that criterion.
STEADY_GROWTH = 1.10

#: Full levels above the ground whose geometric vertical Courant number is
#: reported as "near ground" (the lowest four layers; on the probe's
#: 49-level ladder they span roughly the lowest 250 m over the crest).
NEAR_GROUND_LEVELS = 4


def run_cell(ridge: Ridge, *, wind: float, per_km: float, sound_steps: int,
             seconds: float, etac: float | None = None,
             zadvect_implicit: int = 0, settings: str = "generated",
             courant: bool = False, criterion: str = "map") -> dict:
    """One ridge, one wind, one step through the production ``step()``.

    With ``courant``, every step also records the dycore's own WRF-form
    vertical Courant number (the one ``w_damping`` acts on above 1, read
    from the ``w_cfl_stat`` fold), the count of cells it damped, and the
    geometric Courant number ``|w| dt / dz`` from the actual vertical
    velocity, over every level and over the lowest
    :data:`NEAR_GROUND_LEVELS` full levels.

    ``criterion`` "map" stops a run where the map's held bound is passed.
    "blowup" stops it only where w is non-finite, passes
    :data:`BLOWUP_FACTOR` times that bound or reaches
    :data:`BLOWUP_PEAK_W`, and reports whether the map's
    bound was passed (``over_map_bound``, ``first_over_map_bound_s``): it
    re-checks a cell the map's bound stopped, to tell a run going unstable
    from a finite wave that grew past the bound.  "steady" stops a run as
    "blowup" does and, where it ran the whole time, still does not hold it
    where the last hour's peak w is more than :data:`STEADY_GROWTH` times
    the hour before (``stop`` "growing")."""
    import cupy as cp

    from gpuwm.acoustic_adaptation import steepest_slope
    from gpuwm.core import constants as c
    from gpuwm.core.dycore import set_w_surface, step
    from gpuwm.core.state import init_at_rest
    from gpuwm.core.terrain import bell_hill

    if courant:
        from gpuwm.core.dycore import (enable_wrf_cfl_recording,
                                       reset_wrf_cfl_recording,
                                       take_wrf_cfl)
    if etac is None:
        etac = geometry(ridge)["etac_exact"]
    dt = float(per_km) * ridge.dx / 1000.0
    nx = ridge.columns(wind, seconds)
    cfg = _config(ridge, nx=nx, dt=dt, sound_steps=sound_steps, etac=etac,
                  seconds=seconds, zadvect_implicit=zadvect_implicit,
                  settings=settings)
    terrain = bell_hill(cfg)
    slope = steepest_slope(terrain, cfg.dx, cfg.dy, label="ridge").slope
    bound = 4.0 * float(wind) * float(slope) + 20.0
    coord, base = _base(cfg, etac, terrain)
    if courant:
        reset_wrf_cfl_recording()
        enable_wrf_cfl_recording()
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
    wrf_vc = geo_vc = geo_vc_low = 0.0
    wrf_vc_at = geo_vc_low_at = None
    damped = 0
    over_at = None
    hourly: list[float] = []
    if criterion not in ("map", "blowup", "steady"):
        raise ValueError(f"criterion {criterion!r}: map, blowup or steady")
    gravity = cp.float32(c.G)
    try:
        for index in range(steps):
            cp.cuda.Device().synchronize()
            tick = time.perf_counter()
            step(state, cfg)
            value = float(cp.abs(state.w).max())
            step_seconds.append(time.perf_counter() - tick)
            if courant:
                vert, _horiz = take_wrf_cfl(int(cfg.grid_id))
                if math.isfinite(vert) and vert > wrf_vc:
                    wrf_vc, wrf_vc_at = vert, (index + 1) * dt
                if not math.isfinite(vert):
                    wrf_vc = float("inf")
                damped += _damped_cells(int(cfg.grid_id))
                phi = state.php + state.phb
                dz = (phi[1:] - phi[:-1]) / gravity
                # w on full level k against the thinner layer beside it.
                thin = cp.minimum(dz[:-1], dz[1:])
                ratio = cp.abs(state.w[1:-1]) * cp.float32(dt) / thin
                every = float(ratio.max())
                low = float(ratio[:NEAR_GROUND_LEVELS].max())
                geo_vc = max(geo_vc, every) if math.isfinite(every) \
                    else float("inf")
                if not math.isfinite(low):
                    geo_vc_low = float("inf")
                elif low > geo_vc_low:
                    geo_vc_low, geo_vc_low_at = low, (index + 1) * dt
            if math.isfinite(value) and value > bound and over_at is None:
                over_at = (index + 1) * dt
            # (criterion "steady" stops where "blowup" does; its growth
            # test is taken once the run is over.)
            if stops(value, bound, criterion):
                held = False
                peak = value if math.isfinite(value) else float("inf")
                stopped_at = (index + 1) * dt
                break
            peak = max(peak, value)
            hour = int(index * dt // 3600.0)
            while len(hourly) <= hour:
                hourly.append(0.0)
            hourly[hour] = max(hourly[hour], value)
    finally:
        if courant:
            reset_wrf_cfl_recording()
        del state
        cp.get_default_memory_pool().free_all_blocks()
    growing = (len(hourly) >= 2 and hourly[-2] > 0.0
               and hourly[-1] > STEADY_GROWTH * hourly[-2])
    stop = None if held else "runaway"
    if held and criterion == "steady" and growing:
        held, stop = False, "growing"
    if criterion == "map" and stop is not None:
        stop = "map bound"
    timed = sorted(step_seconds[1:]) or step_seconds
    record = {"held": held, "peak_w": peak, "bound": bound, "dt": dt,
              "per_km": float(per_km), "wind": float(wind), "nx": nx,
              "steps": steps, "stopped_at_s": stopped_at,
              "zadvect_implicit": int(zadvect_implicit),
              "settings": settings, "criterion": criterion,
              "stop": stop, "growing_last_hour": bool(growing),
              "over_map_bound": over_at is not None,
              "first_over_map_bound_s": over_at,
              "hourly_peak_w": [round(v, 3) for v in hourly],
              "median_step_ms": (round(1000.0 * timed[len(timed) // 2], 3)
                                 if timed else None),
              "wall_s": round(time.perf_counter() - started, 2)}
    if courant:
        record.update({"peak_wrf_vertical_courant": wrf_vc,
                       "peak_wrf_vertical_courant_at_s": wrf_vc_at,
                       "w_damping_cell_visits": int(damped),
                       "peak_geometric_vertical_courant": geo_vc,
                       "peak_near_ground_vertical_courant": geo_vc_low,
                       "peak_near_ground_vertical_courant_at_s":
                           geo_vc_low_at})
    return record


def _damped_cells(grid_id: int) -> int:
    """Cells above the w_damping onset in this step's CFL fold (word 1)."""
    import cupy as cp

    from gpuwm.core import dycore
    from gpuwm.core.cfl_inventory import WRF_CFL_SLOTS

    buf = dycore._wrf_cfl_bank("_WRF_CFL_STAT").get(int(grid_id))
    calls = dycore._wrf_cfl_bank("_WRF_CFL_CALLS").get(int(grid_id), 0)
    if buf is None or calls == 0:
        return 0
    slot = ((calls - 1) // 3) % WRF_CFL_SLOTS
    return int(cp.asnumpy(buf[slot, 1]))


def run_adaptive_cell(ridge: Ridge, *, wind: float, max_step: float,
                      start_step: float, seconds: float,
                      target_cfl: float = 1.2, target_hcfl: float = 0.84,
                      increase_pct: int = 5, sound_floor: int = 0,
                      alarm_s: float = 3600.0,
                      etac: float | None = None, criterion: str = "map",
                      settings: str = "generated") -> dict:
    """One ridge, one wind, on the production adaptive clock.

    The loop is :class:`gpuwm.core.adaptive_clock.AdaptiveClockDriver`'s
    for a single domain: the controller reads the CFL the last step
    measured (the dycore's ``w_cfl_stat`` fold, switched on as the driver
    switches it on), proposes the next step, lands on every ``alarm_s`` and
    on the run's end, and the step runs with the substep count the clock
    derives from it, raised to ``sound_floor`` (a terrain rule's
    ``min_time_step_sound``).  ``min_time_step`` is WRF's ``3 x dx``
    fill-in, as on a run that leaves it at -1.

    ``criterion`` is :func:`run_cell`'s: "map" stops the run where the
    map's held bound is passed, "blowup" only where w is non-finite,
    passes :data:`BLOWUP_FACTOR` times it or reaches
    :data:`BLOWUP_PEAK_W`.  Under either, the record keeps
    the hourly peak w, whether the map's bound was passed and when, and
    the peak WRF vertical Courant number the controller read.
    """
    if criterion not in ("map", "blowup"):
        raise ValueError(f"criterion {criterion!r}: map or blowup")
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
                  etac=etac, seconds=seconds, settings=settings)
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
    hourly: list[float] = []
    over_at = None
    peak_vert = 0.0
    started = time.perf_counter()
    reset_wrf_cfl_recording()
    enable_wrf_cfl_recording()
    try:
        while elapsed < total:
            vert, horiz = take_wrf_cfl(int(cfg.grid_id))
            if ctl.started:
                peak_vert = (max(peak_vert, float(vert))
                             if math.isfinite(vert) else float("inf"))
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
            began = elapsed
            step(state, cfg)
            applied_before = dt
            ctl.accept(proposed, max_vert_cfl=vert, max_horiz_cfl=horiz,
                       stepping_to_time=stepping)
            elapsed += dt
            steps.append(float(dt))
            counts[sound] = counts.get(sound, 0) + 1
            value = float(cp.abs(state.w).max())
            if math.isfinite(value) and value > bound and over_at is None:
                over_at = float(elapsed)
            if stops(value, bound, criterion):
                held = False
                peak = value if math.isfinite(value) else float("inf")
                stopped_at = float(elapsed)
                break
            peak = max(peak, value)
            hour = int(began // 3600)
            while len(hourly) <= hour:
                hourly.append(0.0)
            hourly[hour] = max(hourly[hour], value)
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
            "stopped_at_s": stopped_at, "criterion": criterion,
            "settings": settings, "seconds": float(seconds),
            "over_map_bound": over_at is not None,
            "first_over_map_bound_s": over_at,
            "hourly_peak_w": [round(v, 3) for v in hourly],
            "peak_wrf_vertical_courant": peak_vert,
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


def sweep_row(ridge: Ridge, sound_steps: int, winds, rungs, *,
              settings: str, seconds: float = CHECK_SECONDS,
              log=print, on_run=None, criterion: str = "map") -> dict:
    """One candidate row, measured from scratch at ``settings``.

    Each wind, weakest first, walks ``rungs`` (s/km, longest first) from
    the step the weaker wind held, each step run ``seconds`` (the map's
    three-hour check) on a domain the flow cannot wrap, with the
    per-step Courant numbers recorded; the entry is the first step that
    held, ``None`` where none did, and past a wind that held none the
    stronger winds are not run (``tried`` ``None``).  Every run is kept.

    ``criterion`` is :func:`run_cell`'s: "map" holds a run whose w stays
    under the map's bound, "blowup" one that stays finite, under
    :data:`BLOWUP_FACTOR` times it and under :data:`BLOWUP_PEAK_W` for
    the whole ``seconds``, "steady"
    such a run whose w is not still growing at the end."""
    shape = geometry(ridge)
    runs = []
    entries = []
    tried = []
    start = 0
    for wind in winds:
        if start is None:
            entries.append(None)
            tried.append(None)
            continue
        chosen = None
        for index in range(start, len(rungs)):
            result = run_cell(ridge, wind=wind, per_km=rungs[index],
                              sound_steps=sound_steps, seconds=seconds,
                              etac=shape["etac_exact"], settings=settings,
                              courant=True, criterion=criterion)
            runs.append(result)
            if on_run is not None:
                on_run(result)
            log(f"  {settings} {ridge.dx:.0f} m {ridge.crest:.0f} m slope "
                f"{ridge.ridge_slope:g} x{sound_steps} {wind:g} m/s "
                f"{rungs[index]:.3f} s/km: "
                f"{'held' if result['held'] else 'stopped'} (peak w "
                f"{result['peak_w']:.2f}, bound {result['bound']:.1f}, "
                f"vc {result['peak_wrf_vertical_courant']:.2f}, "
                f"{result['wall_s']} s)")
            if result["held"]:
                chosen = index
                break
        tried.append(float(rungs[start]))
        if chosen is None:
            entries.append(None)
            start = None
        else:
            entries.append(float(rungs[chosen]))
            start = chosen
    return {"dx_m": ridge.dx, "crest_m": ridge.crest,
            "ridge_slope": ridge.ridge_slope,
            "sound_steps": int(sound_steps), "slope": shape["slope"],
            "etac": shape["etac"],
            "thinnest_layer_fraction": shape["thinnest_layer_fraction"],
            "settings": settings, "dynamics": dict(SETTINGS[settings]),
            "criterion": criterion,
            "seconds": float(seconds), "winds_m_s": [float(w) for w in winds],
            "rungs_s_per_km": [float(r) for r in rungs],
            "stable_s_per_km": entries, "top_s_per_km": tried,
            "runs": runs}


def sweep_key(job: dict, criterion: str = "map") -> str:
    key = (f"{job['settings']}-dx{float(job['dx']):g}-c{float(job['crest']):g}"
           f"-r{float(job['ridge_slope']):g}-x{int(job['sound_steps'])}")
    # The map criterion keeps the keys its rows were written under.
    return key if criterion == "map" else f"{key}-{criterion}"


def _claim(claims: Path, key: str) -> bool:
    """Atomic claim of one job on this host's file system (a directory
    holding the claimant's pid); a claim whose pid is dead is taken
    over, so a killed worker's job is re-run, never skipped."""
    import os

    target = claims / key
    try:
        target.mkdir()
    except FileExistsError:
        try:
            pid = int((target / "pid").read_text().strip())
            os.kill(pid, 0)
            return False
        except (OSError, ValueError):
            pass
    (target / "pid").write_text(str(os.getpid()))
    return True


def sweep_worker(jobs_path: Path, out_dir: Path, winds, rungs, *,
                 deadline: float, seconds: float, provenance: dict,
                 criterion: str = "map") -> int:
    """Take unclaimed jobs from ``jobs_path`` until none is left or the
    wall clock passes ``deadline`` (epoch seconds; no new job starts after
    it).  Each finished job writes ``<out_dir>/<key>.json``; the runs of
    an unfinished one stream to ``<key>.partial.jsonl``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    claims = out_dir / "claims"
    claims.mkdir(exist_ok=True)
    jobs = [json.loads(line) for line in jobs_path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    done = 0
    for job in jobs:
        if time.time() > deadline:
            break
        key = sweep_key(job, criterion)
        final = out_dir / f"{key}.json"
        if final.exists() or not _claim(claims, key):
            continue
        partial = out_dir / f"{key}.partial.jsonl"
        partial.write_text("", encoding="utf-8")

        def keep(result, partial=partial):
            with partial.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result) + "\n")

        began = time.time()
        row = sweep_row(Ridge(float(job["dx"]), float(job["crest"]),
                              float(job["ridge_slope"])),
                        int(job["sound_steps"]), winds, rungs,
                        settings=job["settings"], seconds=seconds,
                        log=lambda text: print(text, flush=True),
                        on_run=keep, criterion=criterion)
        row["provenance"] = {**provenance, "key": key,
                             "started_utc": time.strftime(
                                 "%Y-%m-%dT%H:%M:%SZ", time.gmtime(began)),
                             "finished_utc": time.strftime(
                                 "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                             "wall_s": round(time.time() - began, 1)}
        tmp = final.with_suffix(".tmp")
        tmp.write_text(json.dumps(row, indent=1), encoding="utf-8")
        tmp.replace(final)
        partial.unlink()
        done += 1
    return done


#: The ladder the shipped adaptive entries walked (the map's ``adaptive``
#: block), and the rungs below it an adaptive sweep may go on to, down to
#: WRF's ``3 x dx`` min_time_step fill-in, past which max_time_step would
#: sit under the clock's own floor.
SHIPPED_ADAPTIVE_LADDER = (15.0, 40.0 / 3.0, 12.0, 11.0, 10.0, 9.0, 8.0,
                           22.0 / 3.0, 20.0 / 3.0, 6.0, 5.5, 5.0)
ADAPTIVE_LADDER_BELOW = (4.5, 4.0, 3.5, 3.0)


def _run_key(wind, per_km, pair) -> str:
    return f"{float(wind):g}|{float(per_km):.6f}|{pair[0]:g}/{pair[1]:g}"


def adaptive_sweep_row(ridge: Ridge, winds, ladder, *, seconds: float,
                       criterion: str, settings: str = "generated",
                       done: dict | None = None, on_run=None,
                       deadline: float = float("inf"), log=print) -> dict:
    """One row's adaptive entries, measured from scratch: per wind
    (weakest first), the longest ``max_time_step`` from ``ladder`` (s/km,
    longest first) that held ``seconds`` at every pair of
    :data:`ADAPTIVE_TARGETS`, each wind walked from the step the weaker
    wind held; past a wind that held none the stronger winds are not run.
    The first step is the shipped measurement's (5 s/km, or the max where
    that is shorter).  ``done`` holds runs already made (``_run_key``),
    which are reused, so a killed job resumes; no new run starts after
    ``deadline`` (the row is then returned with ``complete`` False)."""
    shape = geometry(ridge)
    km = ridge.dx / 1000.0
    done = {} if done is None else done
    runs = []
    entries, tops = [], []
    start = 0
    complete = True
    for wind in winds:
        if start is None or not complete:
            entries.append(None)
            tops.append(None)
            continue
        chosen = None
        for index in range(start, len(ladder)):
            every = True
            for pair in ADAPTIVE_TARGETS:
                key = _run_key(wind, ladder[index], pair)
                result = done.get(key)
                if result is None:
                    if time.time() > deadline:
                        complete = False
                        break
                    upper = round(ladder[index] * km, 2)
                    result = run_adaptive_cell(
                        ridge, wind=wind, max_step=upper,
                        start_step=min(round(5.0 * km, 2), upper),
                        seconds=seconds, target_cfl=pair[0],
                        target_hcfl=pair[1],
                        increase_pct=ADAPTIVE_INCREASE_PCT,
                        etac=shape["etac_exact"], criterion=criterion,
                        settings=settings)
                    result = {"key": key, "per_km": float(ladder[index]),
                              **result}
                    done[key] = result
                    if on_run is not None:
                        on_run(result)
                runs.append(result)
                log(f"  adaptive {ridge.dx:.0f} m {ridge.crest:.0f} m slope "
                    f"{ridge.ridge_slope:g} {wind:g} m/s max "
                    f"{ladder[index]:.3f} s/km cfl {pair[0]}/{pair[1]}: "
                    f"{'held' if result['held'] else 'stopped'} (peak w "
                    f"{result['peak_w']:.1f}, bound {result['bound']:.1f}, "
                    f"mean step {result['mean_step_s']}, "
                    f"{result['wall_s']} s)")
                if not result["held"]:
                    every = False
                    break
            if not complete:
                break
            if every:
                chosen = index
                break
        if not complete:
            entries.append(None)
            tops.append(None)
            continue
        tops.append(float(ladder[start]))
        if chosen is None:
            entries.append(None)
            start = None
        else:
            entries.append(float(ladder[chosen]))
            start = chosen
    return {"dx_m": ridge.dx, "crest_m": ridge.crest,
            "ridge_slope": ridge.ridge_slope, "sound_steps": 4,
            "slope": shape["slope"], "etac": shape["etac"],
            "thinnest_layer_fraction": shape["thinnest_layer_fraction"],
            "settings": settings, "criterion": criterion,
            "seconds": float(seconds),
            "winds_m_s": [float(w) for w in winds],
            "ladder_s_per_km": [float(v) for v in ladder],
            "targets": [list(pair) for pair in ADAPTIVE_TARGETS],
            "increase_pct": ADAPTIVE_INCREASE_PCT,
            "adaptive_s_per_km": entries, "adaptive_top_s_per_km": tops,
            "complete": complete, "runs": runs}


def adaptive_sweep_worker(jobs_path: Path, out_dir: Path, winds, ladder, *,
                          deadline: float, seconds: float, criterion: str,
                          provenance: dict, tag: str = "adaptive") -> int:
    """:func:`sweep_worker` for adaptive rows: each job (dx, crest,
    ridge_slope) writes ``<out_dir>/<key>.json`` when its walk is
    complete; every run streams to ``<key>.partial.jsonl`` and is reused
    when a later worker takes the job up again."""
    import shutil

    out_dir.mkdir(parents=True, exist_ok=True)
    claims = out_dir / "claims"
    claims.mkdir(exist_ok=True)
    jobs = [json.loads(line) for line in jobs_path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    finished = 0
    for job in jobs:
        if time.time() > deadline:
            break
        key = (f"{tag}-{job.get('settings', 'generated')}-dx"
               f"{float(job['dx']):g}-c{float(job['crest']):g}"
               f"-r{float(job['ridge_slope']):g}-{criterion}")
        final = out_dir / f"{key}.json"
        if final.exists() or not _claim(claims, key):
            continue
        partial = out_dir / f"{key}.partial.jsonl"
        done = {}
        if partial.exists():
            for line in partial.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    done[record["key"]] = record

        def keep(result, partial=partial):
            with partial.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result) + "\n")

        began = time.time()
        row = adaptive_sweep_row(
            Ridge(float(job["dx"]), float(job["crest"]),
                  float(job["ridge_slope"])), winds, ladder,
            seconds=seconds, criterion=criterion,
            settings=job.get("settings", "generated"), done=done,
            on_run=keep, deadline=deadline,
            log=lambda text: print(text, flush=True))
        if not row["complete"]:
            shutil.rmtree(claims / key, ignore_errors=True)
            break
        row["provenance"] = {**provenance, "key": key,
                             "finished_utc": time.strftime(
                                 "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                             "wall_s_this_worker": round(
                                 time.time() - began, 1)}
        tmp = final.with_suffix(".tmp")
        tmp.write_text(json.dumps(row, indent=1), encoding="utf-8")
        tmp.replace(final)
        finished += 1
    return finished


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
    w = sub.add_parser("sweep")
    w.add_argument("--jobs", type=Path, required=True,
                   help="JSON lines: dx, crest, ridge_slope, sound_steps, "
                        "settings (generated or ncar)")
    w.add_argument("--out-dir", type=Path, required=True)
    w.add_argument("--winds", type=_floats, required=True)
    w.add_argument("--rungs-per-km", type=_floats, required=True,
                   help="steps to try, s/km, longest first")
    w.add_argument("--seconds", type=float, default=CHECK_SECONDS)
    w.add_argument("--deadline", type=float, default=float("inf"),
                   help="epoch seconds after which no new job starts")
    w.add_argument("--provenance", type=Path, default=None,
                   help="a JSON file recorded on every row")
    w.add_argument("--criterion", choices=("map", "blowup", "steady"),
                   default="map",
                   help="what holds: the map's bound; finite, under "
                        "BLOWUP_FACTOR times it and under BLOWUP_PEAK_W m/s "
                        "for the whole run; or that "
                        "and not still growing (STEADY_GROWTH)")
    v = sub.add_parser("adaptive-sweep")
    v.add_argument("--jobs", type=Path, required=True,
                   help="JSON lines: dx, crest, ridge_slope (settings)")
    v.add_argument("--out-dir", type=Path, required=True)
    v.add_argument("--winds", type=_floats, required=True)
    v.add_argument("--ladder", type=_floats, default=None,
                   help="max_time_step values, s/km, longest first "
                        "(default: the shipped ladder and the rungs below)")
    v.add_argument("--seconds", type=float, default=CHECK_SECONDS)
    v.add_argument("--deadline", type=float, default=float("inf"))
    v.add_argument("--provenance", type=Path, default=None)
    v.add_argument("--criterion", choices=("map", "blowup"),
                   default="blowup")
    v.add_argument("--tag", default="adaptive")
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
            s.add_argument("--settings", choices=sorted(SETTINGS),
                           default="generated")
            s.add_argument("--criterion",
                           choices=("map", "blowup", "steady"),
                           default="map")
            s.add_argument("--courant", action="store_true")
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

    if args.command == "sweep":
        provenance = ({} if args.provenance is None else json.loads(
            args.provenance.read_text(encoding="utf-8")))
        count = sweep_worker(args.jobs, args.out_dir, args.winds,
                             args.rungs_per_km, deadline=args.deadline,
                             seconds=args.seconds, provenance=provenance,
                             criterion=args.criterion)
        print(json.dumps({"sweep_jobs_finished": count}), flush=True)
        return 0
    if args.command == "adaptive-sweep":
        provenance = ({} if args.provenance is None else json.loads(
            args.provenance.read_text(encoding="utf-8")))
        ladder = (list(SHIPPED_ADAPTIVE_LADDER + ADAPTIVE_LADDER_BELOW)
                  if args.ladder is None else args.ladder)
        count = adaptive_sweep_worker(
            args.jobs, args.out_dir, args.winds, ladder,
            deadline=args.deadline, seconds=args.seconds,
            criterion=args.criterion, provenance=provenance, tag=args.tag)
        print(json.dumps({"adaptive_sweep_jobs_finished": count}),
              flush=True)
        return 0
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
                                etac=shape["etac_exact"],
                                settings=args.settings,
                                criterion=args.criterion,
                                courant=args.courant)
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
