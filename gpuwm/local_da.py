"""Author, price, publish and launch a local cycling experiment.

Planning is read-only. Forecast stepping, analysis publication and restart
recovery remain with the existing regional ensemble engine.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
import tomllib
from typing import Callable

SCHEMA = "arwen.local-da-plan.v1"
REQUEST_SCHEMA = "arwen.local-da-request.v1"
#: What a published case directory contains, as (filename, review key).
#: The WPS namelist is named for the experiment beside it because that is
#: the name the preparation owner reads (gpuwm.go_cli.plan_from_config
#: derives "<config stem>.namelist.wps"); a bare "namelist.wps" made the
#: publisher and the preparation door disagree about one configuration and
#: every real launch stopped at its first stage.
PUBLISHED_FILES = (("experiment.toml", "experiment"),
                   ("experiment.namelist.wps", "wps"),
                   ("ensemble.toml", "ensemble"))
#: Where the companion protocol is written down, relative to the tree.
COMPANION_DOCUMENT = "docs/local-da-companion-protocol.md"
#: Every top-level field of a review document, in one place so the door,
#: the discovery contract and the written protocol cannot drift apart.
#: :func:`_build_plan` checks its own result against this before the
#: document leaves the door, so a field added without a line in the
#: protocol fails a test rather than a desktop.
REVIEW_FIELDS = ("alternatives", "analysis_times", "background", "cadence_overrun",
                 "cadence_settings", "changed_scale", "clock",
                 "configuration", "forecast_started", "inputs", "memory",
                 "observation_slots", "observations", "region_checks",
                 "request", "review_sha256", "schema", "selected", "status",
                 "wall", "warnings")
#: Every key of one ``alternatives`` row, in one place, because the
#: companion protocol tells a consumer to draw every row and two of the
#: rows the ladder can produce were never priced at all: the run that
#: stopped before the rung could be priced still has to say so in the
#: same shape as a rung that was.  An unpriced row carries ``priced``
#: false and ``None`` in every price field; a consumer reads ``priced``
#: and never a missing key.  :func:`_ladder_row` builds every row and
#: refuses one that drifts from this roster, so a row added on one path
#: and not the other fails a test rather than a desktop.
ALTERNATIVE_FIELDS = ("cadence", "cycle_seconds", "cycles", "dx_m", "fits",
                      "members", "peak_bytes", "priced", "reasons",
                      "remedies", "scale", "total_seconds")
#: Optional responses to an estimated overrun, beside the comparison they
#: answer. They never change the requested configuration or gate launch.
#: Every ladder row carries reasons and remedies in matching index order.
REMEDY_HOST_MEMORY = "Declare more host memory with --host-gib, or select a smaller region."
REMEDY_CARD_MEMORY = "Declare more free memory with --free-gib, or select a smaller region."
REMEDY_TIME_BUDGET = "Raise the time budget with --budget-seconds, or shorten the forecast."
REMEDY_CADENCE = "Use fewer members, a coarser nest, or a longer cadence."
REMEDY_FORCING = "Shorten the cycle run or select a later initial cycle."
#: Every refusal code this door can print, in three groups, because what
#: a consumer must do next differs by group and by nothing else.  All of
#: them print one JSON document with ``schema``, ``error``, ``code``,
#: ``details`` and ``forecast_started``; a refusal is told apart from a
#: review by the presence of ``error``, never by ``schema``, which both
#: carry.
#:
#: REVIEW means the request was decided before anything was written,
#: fetched or allocated: no case directory, no execution document.
#:
#: LAUNCH means a published case failed an integrity check that runs
#: before its execution document is opened, so a case directory exists,
#: the refusal writes no execution document, and ``forecast_started`` is
#: false.
#:
#: RUN means a published case failed a check that runs after its execution
#: document is opened, so the case directory holds an
#: ``arwen.local-da-execution.v1`` document with status ``FAILED`` and
#: ``forecast_started`` says whether a member had already begun, which it
#: can.  A run refusal is a coded run failure, and naming it that way is
#: deliberate: these checks read what an earlier stage published, so they
#: cannot be made at plan review, and a consumer that was told they could
#: would look for a document that is there and trust a flag that is not.
#:
#: :data:`FALLBACK_REFUSAL_CODE` carries anything the door did not raise
#: itself, so a consumer always has an arm to land in; its group is
#: whichever stage it escaped from, so read the case directory.
#:
#: ``tests/test_local_da_plan.py`` checks this roster against every
#: ``code=`` literal in the three modules, classified by the function that
#: raises it, and against the written protocol, so a code added without a
#: line in the protocol, or a check moved across the execution document,
#: fails a test rather than a desktop.
REVIEW_REFUSAL_CODES = ("FORCING_HORIZON", "INVALID_PLAN",
                        "MISSING_DOMAIN", "REVIEW_CONTRACT")
LAUNCH_REFUSAL_CODES = ("CONFIGURATION_CHANGED",
                        "MISSING_GEOGRAPHY", "OBSERVATION_CHANGED",
                        "PLAN_SCHEMA", "REVIEW_CHANGED", "ROSTER_CHANGED")
RUN_REFUSAL_CODES = ("ANALYSIS_ROSTER", "CONTINUOUS_WINDOW_FAILED", "CWP_OPERATOR_UNAVAILABLE", "MISSING_MANIFEST", "MISSING_SURFACE",
                     "OBSERVATION_WINDOW_CHANGED", "OUTPUT_CHANGED",
                     "PREPARATION_CHANGED", "REFERENCE_CHANGED",
                     "SURFACE_CHANGED")
FALLBACK_REFUSAL_CODE = "LOCAL_DA_ERROR"
REFUSAL_CODES = tuple(sorted(REVIEW_REFUSAL_CODES + LAUNCH_REFUSAL_CODES
                             + RUN_REFUSAL_CODES + (FALLBACK_REFUSAL_CODE,)))
#: What each group leaves behind, published beside the roster so the page,
#: the door and the desktop cannot hold three different beliefs about it:
#: (group, the refusal writes an execution document, ``forecast_started``
#: can be true).
REFUSAL_EFFECTS = (("review", False, False),
                   ("launch", False, False),
                   ("run", True, True))
#: The modules the roster is checked against, relative to the tree, by the
#: door that owns them.  The launch door's own two groups are split by
#: :data:`PRE_EXECUTION_FUNCTIONS` rather than by module, because one
#: module raises both.
REFUSAL_SOURCES = (("review", "gpuwm/local_da.py"),
                   ("launch", "gpuwm/local_da_runtime.py"),
                   ("launch", "gpuwm/local_da_fetch.py"))
#: The functions in the launch modules that run BEFORE the case's
#: execution document is opened.  A coded raise lexically inside one of
#: these is a LAUNCH refusal; a coded raise anywhere else in those modules
#: is a RUN refusal, because :func:`gpuwm.local_da_runtime.launch` has
#: written ``execution.json`` by the time it escapes.
PRE_EXECUTION_FUNCTIONS = ("read_plan", "preflight")
GIB = 1 << 30
# Policy choices, not engine limits. Static samples are evaluated only at
# analysis time; they are never integrated as forecast members.
STATIC_SAMPLES = 8
BASE_SPACING_M = 3000.
BASE_SPAN_M = 192000.
DEFAULT_BUDGET_SECONDS = 3600.
DEFAULT_FORECAST_SECONDS = 1800.
DA_SCRATCH_MIB = 256


class PlanError(ValueError):
    def __init__(self, message: str, *, code: str = "INVALID_PLAN", details=None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


def _positive(value, name: str) -> float:
    if isinstance(value, bool):
        raise PlanError(f"{name} is a boolean, not a quantity; enter a positive number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PlanError(f"{name} is not numeric; enter a positive number.") from exc
    if not math.isfinite(result) or result <= 0:
        raise PlanError(f"{name}={value!r} is not finite and positive; correct the quantity.")
    return result


def utc(value: str) -> datetime:
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise PlanError(f"epoch {value!r} is not an ISO timestamp; include a UTC offset.") from exc
    if stamp.tzinfo is None:
        raise PlanError("epoch has no timezone; append Z or an explicit UTC offset.")
    stamp = stamp.astimezone(timezone.utc)
    if stamp.microsecond:
        raise PlanError("epoch has fractional seconds; choose a whole-second boundary.")
    return stamp


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


@dataclass(frozen=True)
class Card:
    vram_gib: float
    free_gib: float | None = None
    host_gib: float = 32.
    speed_factor: float = 1.
    name: str = "declared card"

    def __post_init__(self):
        for field in ("vram_gib", "host_gib", "speed_factor"):
            object.__setattr__(self, field, _positive(getattr(self, field), field))
        if self.free_gib is not None:
            object.__setattr__(self, "free_gib", _positive(self.free_gib, "free_gib"))
            if self.free_gib > self.vram_gib:
                raise PlanError("Free VRAM exceeds total VRAM; correct the declared card memory.")
        if not isinstance(self.name, str) or not self.name.strip():
            raise PlanError("The card name is empty; provide a label for the timing assumption.")

    @property
    def budget_bytes(self):
        # Same reserve as the domain author, rather than a second card rule.
        from gpuwm import domain_wizard as dw
        free = self.vram_gib if self.free_gib is None else self.free_gib
        return max(0, dw.sizing_budget_bytes(None, free_bytes=int(free * GIB),
            vram_gib=self.vram_gib, forcing_interval_seconds=1.))


@dataclass(frozen=True)
class Request:
    epoch: str
    card: Card
    point: tuple[float, float] | None = None
    region: tuple[float, float, float, float] | None = None
    scale: int = 1
    budget_seconds: float = DEFAULT_BUDGET_SECONDS
    forecast_seconds: float = DEFAULT_FORECAST_SECONDS
    cadence_seconds: float | None = None
    source: str | None = None
    source_cycle: str | None = None
    source_product: str | None = None
    source_member: str | int | None = None
    source_provider: str | None = None
    forcing_cadence_hours: int | None = None
    source_root: str | None = None
    prepared_root: str | None = None
    prepared_config: str | None = None
    prepared_namelist: str | None = None
    source_inputs: dict[str, str] | None = None
    supplements: tuple[str, ...] = ()
    profile: str | None = None
    obs_tables: tuple[str, ...] = ()
    radar_grids: tuple[str, ...] = ()
    satellite_grids: tuple[str, ...] = ()
    base_seed: int = 0
    continuous_windows: int = 0

    def validate(self):
        if type(self.continuous_windows) is not int or self.continuous_windows < 0:
            raise PlanError('continuous_windows must be a whole number of analysis windows: 0 runs the reviewed finite cycle, N runs N continuous windows.')
        utc(self.epoch)
        from gpuwm.regional_preparation import validate_request
        validate_request(self)
        if not isinstance(self.card, Card):
            raise PlanError("The card declaration is missing; supply total VRAM and timing basis.")
        if type(self.scale) is not int or self.scale < 1:
            raise PlanError("scale must be a positive integer; start with rung 1.")
        if type(self.base_seed) is not int or self.base_seed < 0:
            raise PlanError("base_seed must be a nonnegative integer; use 0 for repeatable defaults.")
        for name in ('budget_seconds', 'forecast_seconds', 'cadence_seconds'):
            value = getattr(self, name)
            if value is None and name == 'cadence_seconds':
                continue
            if type(value) not in (int, float):
                raise PlanError(f'{name} needs a numeric duration; correct the request value.')
            _positive(value, name)
        if (self.point is None) == (self.region is None):
            raise PlanError("Choose exactly one point or region; remove the other location input.")
        if self.point is not None:
            if not isinstance(self.point, (tuple, list)) or len(self.point) != 2:
                raise PlanError("A point needs latitude,longitude; provide both coordinates.")
            lat, lon = self.point
            if not all(type(v) in (int, float) and math.isfinite(v) for v in self.point) or not -90 < lat < 90 or not -180 <= lon <= 180:
                raise PlanError("The point is outside finite geographic coordinates; correct latitude and longitude.")
        if self.region is not None:
            if not isinstance(self.region, (tuple, list)) or len(self.region) != 4 or not all(type(v) in (int, float) and math.isfinite(v) for v in self.region):
                raise PlanError("A region needs finite west,south,east,north coordinates; correct the bounds.")
            w, s, e, n = self.region
            if not (-180 <= w <= 180 and -180 <= e <= 180 and -90 < s < n < 90) or w == e:
                raise PlanError("The region is empty or outside geographic coordinates; correct its bounds.")
            if (e - w) % 360 >= 180:
                raise PlanError("The region spans at least half the globe; select a smaller local region.")
        for paths in (self.obs_tables, self.radar_grids, self.satellite_grids):
            for path in paths:
                if not Path(path).is_file():
                    raise PlanError(f"Observation input {path} is missing; supply an existing file or omit that input.")


def _location(request: Request):
    if request.point is not None:
        return tuple(float(v) for v in request.point)
    w, s, e, n = request.region
    return ((s + n) / 2, (w + ((e - w) % 360) / 2 + 180) % 360 - 180)


def derive_rung(request: Request, scale: int) -> dict:
    from gpuwm.ensemble import config as owner
    from gpuwm.da.cadence import TUNED_BASELINE_INTERVAL_S
    # Saturate the member count without constructing a giant integer.
    exponent = min(scale - 1, math.ceil(math.log2(owner.MAX_MEMBERS)))
    members = min(owner.MAX_MEMBERS, 2 ** exponent)
    refinement = 1 + (scale - 1) // 3
    dx = BASE_SPACING_M / refinement
    cadence = (request.cadence_seconds if request.cadence_seconds is not None
               else TUNED_BASELINE_INTERVAL_S / refinement)
    # Derived defaults use the whole-second output timestamp contract.
    # An explicit value must never be silently shortened to that contract.
    cadence = float(math.floor(cadence) if request.cadence_seconds is None else cadence)
    from gpuwm.experiment import _check_whole_second_cadence
    for value, label in ((cadence, 'cadence_seconds'), (request.forecast_seconds, 'forecast_seconds')):
        _positive(value, label)
        try:
            _check_whole_second_cadence(label, Fraction(str(value)), 1, 'local cycling')
        except ValueError as exc:
            raise PlanError(f'{exc} Choose positive whole-second durations for analysis and forecast-output timestamps.') from exc
    point = _location(request)
    from gpuwm import domain_wizard as dw
    projection = dw._projection_entries(*point, "auto")
    span = BASE_SPAN_M * (1 + .125 * (scale - 1))
    nx = ny = dw._even(math.ceil(span / dx))
    checks = {"contained": True, "method": "point at projected center"}
    if request.region is not None:
        import numpy as np
        w, s, e, n = request.region
        east = w + (e - w) % 360
        # Densify all four geographic sides. Projection curvature is not
        # represented by the four corners alone.
        count = max(17, int(math.ceil(max(n - s, east - w) * 32)) + 1)
        lon = np.linspace(w, east, count)
        lat = np.linspace(s, n, count)
        ys = np.r_[np.full(count, s), np.full(count, n), lat, lat]
        xs = np.r_[lon, lon, np.full(count, w), np.full(count, east)]
        grid = dw._root_grid(projection, nx, ny, dx)
        ii, jj = grid.latlon_to_ij(ys, (xs + 180) % 360 - 180)
        # Keep the region inside the relaxed boundary strip, not merely
        # inside the outer grid edge.
        margin = dw._SPEC_BDY_WIDTH + dw._BLEND_WIDTH
        half_i = float(np.max(np.abs(ii - (nx + 1) / 2)))
        half_j = float(np.max(np.abs(jj - (ny + 1) / 2)))
        nx = max(nx, dw._even(math.ceil(2 * (half_i + margin + 1))))
        ny = max(ny, dw._even(math.ceil(2 * (half_j + margin + 1))))
        grid = dw._root_grid(projection, nx, ny, dx)
        ii, jj = grid.latlon_to_ij(ys, (xs + 180) % 360 - 180)
        contained = bool(np.all((ii >= 1 + margin) & (ii <= nx - margin)
                                & (jj >= 1 + margin) & (jj <= ny - margin)))
        if not contained:
            raise PlanError("The projected region crosses the boundary strip; select a smaller region.")
        checks = {"contained": contained, "samples": len(xs), "boundary_margin_cells": margin,
                  "method": "densified geographic perimeter in native projection"}
    return dict(scale=scale, members=members, covariance_members=STATIC_SAMPLES if members == 1 else members,
                analysis="static-covariance-oi" if members == 1 else "letkf",
                nx=nx, ny=ny, dx_m=dx, point=list(point), projection=projection,
                cadence_seconds=cadence, cycles=1 if scale == 1 else scale,
                forecast_seconds=float(request.forecast_seconds), region_checks=checks)


def configuration(request: Request, rung: dict, *, background_probe=None):
    from gpuwm import domain_wizard as dw
    from gpuwm.regional_preparation import review_background
    from gpuwm.starter_template import render_tables
    background = review_background(request, rung, probe=background_probe)
    selected = background['selection']
    source = selected['source']
    stamp = utc(selected['init'])
    duration = rung['cycles'] * rung['cadence_seconds'] + rung['forecast_seconds']
    interval = selected['forcing_interval_seconds']
    dims = [(rung['nx'], rung['ny'])]
    # The rung's own grid picks the default, as at every other door
    # (gpuwm.physics_menu.default_profile_for): read with no grid, a 750 m
    # rung bound the source's own suite where the sub-km row binds.
    profile = dw.resolved_physics_profile(source, request.profile,
        finest_dx_m=dw.finest_spacing_m(rung['dx_m'], ()), domains=len(dims))
    dw._pole_clearance_refusal(rung['projection'], rung['nx'], rung['ny'], rung['dx_m'])
    area = dw.fetch_area_hint(rung['projection'], rung['nx'], rung['ny'],
                             source=source, root_dx_m=rung['dx_m'])
    if background['fetch_hints'] is not None:
        from gpuwm.fetch import fetch_accepts_area
        if background['inputs']['kind'] == 'automatic':
            background['fetch_hints'] = dict(background['fetch_hints'], out='data/background')
        if fetch_accepts_area(source):
            background['fetch_hints']['area'] = area
    text = dw.render_config(name="Local rapid cycling", start_time=stamp,
        hours=max(1, math.ceil(duration / 3600)), projection=rung['projection'],
        dims=dims, ratios=(), root_dx_m=rung['dx_m'],
        profile=profile, tiles="off", cumulus_requested=False,
        fetch_hints=background['fetch_hints'],
        case_data=None, history_interval_s=rung['cadence_seconds'])
    raw = tomllib.loads(text)
    raw['experiment']['run_seconds'] = duration
    raw['experiment']['restart_interval_s'] = rung['cadence_seconds']
    # Ask the existing clock author again after replacing the rounded-hour
    # duration. Bind every physics period it emitted, then require the
    # cycle spine's exact millisecond lattice as well.
    shared = raw['shared']
    root = raw['domain'][0]
    dt = dw.derived_time_step_s(rung['point'][0], rung['dx_m'],
        run_seconds=duration, ratios=(), history_interval_s=rung['cadence_seconds'],
        restart_interval_s=rung['cadence_seconds'],
        physics_periods_s=dw._root_physics_periods(root, shared))
    # A rational WRF step finer than one millisecond cannot be represented
    # by CycleClock. Choose the largest integral-millisecond divisor.
    from gpuwm.cycle.contracts import TICK_HZ
    if (dt * TICK_HZ).denominator != 1:
        periods = [duration, rung['cadence_seconds']]
        periods += [float(p) for p in dw._root_physics_periods(root, shared)]
        ticks = [round(p * TICK_HZ) for p in periods]
        common = math.gcd(*ticks)
        maximum = max(1, math.floor(dt * TICK_HZ))
        candidate = min(common, maximum)
        while common % candidate:
            candidate -= 1
        dt = Fraction(candidate, TICK_HZ)
    for key in ('time_step', 'time_step_fract_num', 'time_step_fract_den'):
        root.pop(key, None)
    root.update(dw._clock_keys(dt))
    text = '# Local rapid cycling configuration.\n' + render_tables(raw)
    exp = dw.experiment_from_text(text, source="experiment.toml")
    wps = dw.render_wps_namelist(rung['projection'], [(rung['nx'], rung['ny'])], (),
                               root_dx_m=rung['dx_m'], source=source,
                               forcing_interval_seconds=interval)
    from gpuwm.regional_preparation import validate_authored_background
    validate_authored_background(background, exp)
    return text, wps, exp, background


def observation_routes(request: Request, rung: dict, experiment) -> list[dict]:
    from gpuwm.local_da_observations import inspect_routes
    return inspect_routes(request, rung, experiment)


def price_rung(request: Request, rung: dict, exp, streams, *, forcing_interval_seconds=None) -> dict:
    import numpy as np
    from gpuwm.core.preflight import estimate_phases
    from gpuwm.core.pace import estimate_pace
    from gpuwm.da.background import DEFAULT_BACKGROUND_SOURCE, resolve_background_source
    from gpuwm.da.letkf import solve_bytes_per_point
    source = request.source or DEFAULT_BACKGROUND_SOURCE
    interval = forcing_interval_seconds or resolve_background_source(source).forcing_interval_seconds
    phases = estimate_phases(exp, source=source, vram_gib=request.card.vram_gib,
                             forcing_interval_seconds=interval, preprocess_backend="cpu")
    # An unpriced rung is admitted and priced from the most conservative
    # rate on record, never refused: a configuration nobody has timed is
    # not a configuration nobody may run. The substitution is named in
    # the basis by its owner and warned about once in the review.
    pace = estimate_pace(exp, streamed=None, conservative=True)
    if pace is None:
        raise PlanError("The authored experiment carries no domain to price; author a domain before review.", code="MISSING_DOMAIN")
    substituted = bool(getattr(pace, 'substituted', False))
    ncells = rung['nx'] * rung['ny'] * exp.domains[0].run.nz
    n = rung['covariance_members']
    enabled = sum(row['status'] in ('candidate', 'ready') for row in streams)
    # Reserve simultaneous prior, posterior, increments, diagnosed states
    # and observation-space copies. Sequential forecast state is not
    # multiplied by the member count.
    from gpuwm.ensemble.state_sha import serialized_state_attrs
    from gpuwm.da.letkf import GridGeometry, Localization, _horizontal_stencil
    from gpuwm import domain_wizard as dw
    fields = len(serialized_state_attrs())
    stored_members = n + 1 if rung['members'] == 1 else n
    batches = max(1, sum(int(row.get('batch_bound', 4)) for row in streams
                         if row['status'] in ('candidate', 'ready')))
    arrays = ncells * 8 * (4 * fields * stored_members + batches * (2 * stored_members + 4))
    scratch = DA_SCRATCH_MIB * (1 << 20)
    projection = dw._root_grid(rung['projection'], rung['nx'], rung['ny'], rung['dx_m'])
    lat, lon = projection.latlon_mass()
    geometry = GridGeometry(dx_m=rung['dx_m'], dy_m=rung['dx_m'],
        heights_m=np.arange(exp.domains[0].run.nz, dtype=float),
        lat_deg=lat, lon_deg=lon)
    horizontal = len(_horizontal_stencil(Localization(15000., 3000.), geometry,
                                         rung['nx'], rung['ny'])[0])
    # The actual terrain columns are not prepared yet. Every possible
    # vertical offset bounds the owner's height-dependent stencil; thinning
    # does not shrink its rectangular gather allocation.
    vertical = 2 * exp.domains[0].run.nz - 1
    slots = batches * horizontal * vertical
    scratch = max(scratch, solve_bytes_per_point(slots, n, 8))
    analysis_peak = arrays + scratch + int(.75 * GIB)
    # Existing LETKF source records 71,829 points/s for R=30, 405 slots,
    # chunk=512 on its reference card. Halve that rate for this budget and
    # operator/staging allowance, then scale the two operation terms.
    solve_scale = max(1., (n / 30.) ** 3 + (slots / 405.) * (n / 30.) ** 2)
    analysis_seconds = (ncells / (71829. / 2) * solve_scale + 15. * max(1, enabled)) / request.card.speed_factor
    per_sim = pace.wall_seconds_high / exp.run_seconds / request.card.speed_factor
    cycle_seconds = rung['members'] * rung['cadence_seconds'] * per_sim + analysis_seconds
    return dict(forecast_peak_bytes=phases.peak_envelope_bytes,
                analysis_peak_bytes=analysis_peak,
                host_peak_bytes=arrays * 2 + phases.peak_envelope_bytes + 16 * sum(Path(p).stat().st_size for p in request.obs_tables),
                cycle_seconds=cycle_seconds,
                forecast_seconds=rung['members'] * rung['forecast_seconds'] * per_sim,
                preparation_seconds=120., measured=False, observation_slots=slots,
                pace_substituted=substituted,
                solve_memory_mib=math.ceil(scratch / (1 << 20)),
                basis=[pace.basis, f"forecast reference card: {pace.reference_card}",
                       "DA basis: existing LETKF 30-member/405-slot/512-point measurement, rate halved; cubic and slot-quadratic extrapolation",
                       "120 s preparation allowance is an assumption; network wait is not predictable",
                       "Declared speed factor applies to compute, never inferred from VRAM",
                       "CPU preprocessing; resident sequential forecast; simultaneous analysis arrays and scratch",
                       f"DA bound: {fields} restart fields, {stored_members} stored states, {batches} batches, {horizontal} horizontal x {vertical} vertical slots per batch"])


def cadence_projection(*, epoch, cadence_seconds, cycles, dt_seconds,
                       cycle_cost_seconds, measured=False):
    """Project queue lag without changing the requested cycle schedule."""
    from gpuwm.da.cadence import CadencePlan, Cycle, check_overrun
    anchor = utc(epoch)
    cadence = float(cadence_seconds)
    plan = CadencePlan(mode='fixed', anchor=anchor, dt_seconds=float(dt_seconds),
        cycles=tuple(Cycle(elapsed_seconds=cadence * (index + 1), leg_seconds=cadence,
            valid_time=anchor + timedelta(seconds=cadence * (index + 1)),
            volume_time=None, quantization_shift_seconds=0.) for index in range(int(cycles))))
    _, record = check_overrun(plan, cycle_cost_seconds=float(cycle_cost_seconds),
        policy='queue', cost_basis='measured' if measured else 'estimated')
    return record


def cadence_verdict(request: Request, rung: dict, exp, cycle_seconds: float,
                    *, measured: bool = False) -> dict:
    """Keep every requested cycle and report its projected lag as advice."""
    return cadence_projection(epoch=request.epoch, cadence_seconds=rung['cadence_seconds'],
        cycles=rung['cycles'], dt_seconds=exp.domains[0].run.dt,
        cycle_cost_seconds=cycle_seconds, measured=measured)


def _ladder_row(scale: int, *, fits: bool, causes: list[tuple[str, str]],
                priced: bool = False, **price_fields) -> dict:
    """One ladder row, in the one shape the published protocol promises.

    ``causes`` pairs each comparison with an optional response, in the
    order found. ``fits`` describes estimates and never gates a ready
    plan. A rung missing required forcing is unpriced and carries
    ``None`` in every price field rather than dropping the keys.
    """
    row = dict(scale=int(scale), fits=bool(fits), priced=bool(priced),
               reasons=[reason for reason, _ in causes],
               remedies=[remedy for _, remedy in causes],
               members=None, dx_m=None, cycles=None, peak_bytes=None,
               cycle_seconds=None, total_seconds=None, cadence=None)
    row.update(price_fields)
    if sorted(row) != sorted(ALTERNATIVE_FIELDS):
        raise PlanError(
            "a ladder row carries keys the companion protocol does not "
            f"describe: {sorted(set(row) ^ set(ALTERNATIVE_FIELDS))}; add "
            f"them to ALTERNATIVE_FIELDS and to {COMPANION_DOCUMENT} in the "
            "same change, so a companion never meets an undescribed row.",
            code="REVIEW_CONTRACT")
    return row


def _advisory_text(opening: str, causes: list[tuple[str, str]], *,
                  lead: str | None = None) -> str:
    """Quote one reason per distinct remedy, including simultaneous limits."""
    if lead is not None:
        causes = ([pair for pair in causes if pair[1] == lead]
                  + [pair for pair in causes if pair[1] != lead])
    distinct = {}
    for reason, remedy in causes:
        distinct.setdefault(remedy, reason)
    return (opening + '; '.join(distinct.values()) + '. '
            + '; '.join(remedy.rstrip('.') for remedy in distinct) + '.')


def build_plan(request: Request, *, availability: Callable | None = None,
               price: Callable | None = None, background_probe=None) -> dict:
    """Inspect all owners without repairing bridge caches or keeping bindings."""
    from gpuwm.local_da_observations import registry_inspection
    with registry_inspection():
        return _build_plan(request, availability=availability, price=price, background_probe=background_probe)


def _build_plan(request: Request, *, availability: Callable | None = None,
                price: Callable | None = None, background_probe=None) -> dict:
    request.validate()
    availability = availability or observation_routes
    budget = request.card.budget_bytes
    alternatives = []
    chosen = None
    for scale in range(1, request.scale + 1):
        rung = derive_rung(request, scale)
        try:
            text, wps, exp, background = configuration(request, rung, background_probe=background_probe)
        except PlanError as exc:
            if exc.code == "FORCING_HORIZON":
                alternatives.append(_ladder_row(scale, fits=False,
                                                causes=[(str(exc), REMEDY_FORCING)]))
                if scale == request.scale:
                    raise PlanError(str(exc), code="FORCING_HORIZON",
                                    details={"alternatives": alternatives}) from exc
                continue
            raise
        streams = availability(request, rung, exp)
        costs = (price(request, rung, exp, streams) if price is not None else
                 price_rung(request, rung, exp, streams,
                            forcing_interval_seconds=background['selection']['forcing_interval_seconds']))
        for key in ('forecast_peak_bytes', 'analysis_peak_bytes', 'host_peak_bytes',
                    'cycle_seconds', 'forecast_seconds', 'preparation_seconds'):
            _positive(costs[key], f"price.{key}")
        total = costs['preparation_seconds'] + costs['forecast_seconds'] + rung['cycles'] * costs['cycle_seconds']
        peak = max(costs['forecast_peak_bytes'], costs['analysis_peak_bytes'])
        found: list[tuple[str, str]] = []
        if peak > budget:
            found.append((f"estimated peak {peak / GIB:.2f} GiB exceeds usable {budget / GIB:.2f} GiB",
                          REMEDY_CARD_MEMORY))
        if costs['host_peak_bytes'] > request.card.host_gib * GIB:
            found.append(("estimated analysis host arrays exceed declared host RAM", REMEDY_HOST_MEMORY))
        overrun = cadence_verdict(request, rung, exp, costs['cycle_seconds'],
                                  measured=bool(costs['measured']))
        if overrun['outcome'] != 'clear':
            found.append((f"cycle cost {costs['cycle_seconds']:.1f} s exceeds cadence "
                          f"{rung['cadence_seconds']:.1f} s: {overrun['detail']}", REMEDY_CADENCE))
        if total > request.budget_seconds:
            found.append((f"total {total:.1f} s exceeds time budget {request.budget_seconds:.1f} s",
                          REMEDY_TIME_BUDGET))
        alternatives.append(_ladder_row(scale, fits=not found, causes=found, priced=True,
                                        members=rung['members'], dx_m=rung['dx_m'],
                                        cycles=rung['cycles'], peak_bytes=int(peak),
                                        cycle_seconds=costs['cycle_seconds'],
                                        total_seconds=total, cadence=overrun))
        if scale == request.scale:
            chosen = rung, text, wps, exp, background, streams, costs, peak, total, overrun, found
    rung, text, wps, exp, background, streams, costs, peak, total, overrun, advisories = chosen
    from gpuwm.cycle.clock import CycleClock
    clock = CycleClock.build(epoch_anchor=utc(request.epoch), parent_dt_seconds=exp.domains[0].run.dt,
                             cycle_seconds=rung['cadence_seconds'], n_cycles=rung['cycles'])
    from gpuwm.da.cadence import scaled_settings
    settings = scaled_settings(cycle_interval_s=rung['cadence_seconds'], rtps_alpha=.8,
        error_inflation=1., horizontal_loc_m=15000., vertical_loc_m=3000.)
    # The radial-velocity dispersion gate's thresholds ride beside the
    # relaxation, unscaled, so the analysis reads them from the reviewed
    # plan and every analysis records them (gpuwm.da.velocity_dispersion).
    from gpuwm.da.velocity_dispersion import (DEFAULT_VELOCITY_DISPERSION_BATCH_RATIO,
                                              DEFAULT_VELOCITY_DISPERSION_RATIO)
    settings['applied'].update(velocity_dispersion_ratio=DEFAULT_VELOCITY_DISPERSION_RATIO,
                               velocity_dispersion_batch_ratio=DEFAULT_VELOCITY_DISPERSION_BATCH_RATIO)
    from gpuwm.da.static_covariance import perturbation_options
    ensemble = {"ensemble": dict(base_config="experiment.toml", n_members=rung['members'],
        base_seed=request.base_seed, perturbation="none" if rung['members'] == 1 else "gpuwm.da.perturb",
        perturbation_options=perturbation_options(mp_physics=exp.domains[0].run.mp_physics))}
    from gpuwm.starter_template import render_tables
    warnings = []
    if advisories:
        warnings.append(_advisory_text('Resource estimates are advisory; the requested settings are retained: ', advisories))
    if not costs['measured']:
        warnings.append("Timing is an estimate from the stated basis, not a measurement of this configuration.")
    if costs.get('pace_substituted'):
        warnings.append("No recorded pace row names this configuration's physics rung, so the forecast cost is "
                        "priced from the slowest rate on record; the review states that basis and the rung runs.")
    if not any(row['status'] in ('candidate', 'ready') for row in streams):
        warnings.append("No observation route is available; cycles publish explicit zero increments and are reported as forecast-only.")
    warnings.extend(["One externally forced regional domain; no concurrent parent or child-domain assimilation.",
                     "Correlated errors, covariance tuning and forecast benefit need independent validation.",
                     "The requested scale, cycle count and cadence are retained; estimates do not decide permission to run."])
    result = dict(schema=SCHEMA, status="ready", forecast_started=False,
        request=asdict(request), background=background, selected=rung, changed_scale=rung['scale'] != request.scale,
        region_checks=rung['region_checks'], alternatives=alternatives, observations=streams,
        clock=clock.to_json(), analysis_times=[clock.valid_time(i).isoformat() for i in range(1, rung['cycles'] + 1)],
        cadence_settings=settings, cadence_overrun=overrun, memory=dict(policy="advisory", peak_bytes=int(peak), budget_bytes=budget,
            forecast_peak_bytes=int(costs['forecast_peak_bytes']), analysis_peak_bytes=int(costs['analysis_peak_bytes']),
            host_peak_bytes=int(costs['host_peak_bytes']), host_budget_bytes=int(request.card.host_gib * GIB),
            solve_memory_mib=int(costs['solve_memory_mib'])),
        wall=dict(policy='advisory', budget_seconds=request.budget_seconds,
                  budget_semantics='comparison target, not an enforced deadline; the finite cycle count and forecast duration are unchanged',
                  cycle_seconds=costs['cycle_seconds'], total_seconds=total,
                  forecast_seconds=costs['forecast_seconds'], preparation_seconds=costs['preparation_seconds'],
                  measured=costs['measured'], basis=costs['basis']),
        warnings=warnings, observation_slots=costs['observation_slots'],
        configuration=dict(experiment=text, wps=wps, ensemble=render_tables(ensemble)),
        inputs={path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for paths in (request.obs_tables, request.radar_grids, request.satellite_grids) for path in paths})
    if request.continuous_windows:
        from gpuwm.local_da_controller import review_contract
        result['continuous'] = review_contract(request.continuous_windows)
    result['review_sha256'] = digest(result)
    if sorted(result) != sorted(set(REVIEW_FIELDS) | ({'continuous'} if request.continuous_windows else set())):
        raise PlanError(
            "the review document carries fields the companion protocol does "
            f"not describe: {sorted(set(result) ^ set(REVIEW_FIELDS))}; add "
            f"them to REVIEW_FIELDS and to {COMPANION_DOCUMENT} in the same "
            "change, so a companion never meets an undocumented field.",
            code="REVIEW_CONTRACT")
    return result


def _route_companions(plan: dict, root: Path):
    """Every file beside the configuration that its route reads.

    Asked of the route module, which answers from the published
    configuration's own ``[fetch].source`` -- the key the dispatcher
    reads -- so this publisher and the run cannot disagree about what a
    published case directory must hold.  The WPS namelist is left out
    because :data:`PUBLISHED_FILES` already publishes it, under the name
    this answer would give it.
    """
    from gpuwm import domain_wizard as dw
    from gpuwm.hrrr_route_inputs import candidate_companions, route_input_paths

    text = plan['configuration']['experiment']
    config_path = root / PUBLISHED_FILES[0][0]
    wps_name = route_input_paths(config_path)['wps_namelist'].name
    exp = dw.experiment_from_text(text, source=str(config_path))
    return [(path, content) for path, content in candidate_companions(
        config_path, exp, wps_text=plan['configuration']['wps'],
        source=(tomllib.loads(text).get('fetch') or {}).get('source'))
        if path.name != wps_name]


def publish(plan: dict, directory: str | Path) -> dict:
    from gpuwm.starter_template import _publish_new_files
    root = Path(directory).expanduser().resolve()
    if root.exists():
        raise PlanError(f"Output directory {root} exists; choose a new directory or launch its saved plan.")
    document = dict(plan)
    document['files'] = {name: hashlib.sha256(plan['configuration'][key].encode()).hexdigest()
                         for name, key in PUBLISHED_FILES}
    # Whatever else the published configuration's route reads beside it,
    # rendered from that configuration rather than assumed absent: the
    # background source decides the route, and a case directory whose
    # route reads namelists it has not got is refused at the prepare
    # stage of its first cycle, before anything is fetched.
    companions = _route_companions(plan, root)
    document['files'].update({path.name: hashlib.sha256(content.encode()).hexdigest()
                              for path, content in companions})
    root.mkdir(parents=True, exist_ok=False)
    try:
        _publish_new_files(tuple(companions) +
            tuple((root / name, plan['configuration'][key])
            for name, key in PUBLISHED_FILES) +
            ((root / 'local-da.json', json.dumps(document, indent=2, allow_nan=False) + '\n'),))
    except BaseException:
        # The publisher rolls back its own files; remove an empty directory
        # only, never erase another writer's content.
        try:
            root.rmdir()
        except OSError:
            pass
        raise
    result = dict(plan_path=str(root / 'local-da.json'), review_sha256=plan['review_sha256'], forecast_started=False)
    if plan.get('continuous', {}).get('enabled'):
        result.update(status_path=str(root / plan['continuous']['status_relative_path']),
                      control_path=str(root / plan['continuous']['control_relative_path']))
    return result


def request_from_json(raw: dict) -> Request:
    if not isinstance(raw, dict) or raw.get('schema') != REQUEST_SCHEMA:
        raise PlanError(f"Request schema is not {REQUEST_SCHEMA}; use the current local DA form.")
    allowed = set(Request.__dataclass_fields__) | {'schema'}
    unknown = set(raw) - allowed
    if unknown:
        raise PlanError(f"Unknown request fields {sorted(unknown)}; remove them rather than silently dropping settings.")
    data = {k: v for k, v in raw.items() if k != 'schema'}
    try:
        data['card'] = Card(**data['card'])
        return Request(**data)
    except (KeyError, TypeError) as exc:
        raise PlanError(f"The request is missing required fields or has unknown card keys: {exc}; correct the form.") from exc


def protocol_document() -> dict:
    """Static companion discovery; no observation, device or pricing work."""
    from gpuwm.ensemble.config import MAX_MEMBERS
    from gpuwm.da.cadence import TUNED_BASELINE_INTERVAL_S
    from gpuwm.background_contract import catalog
    from gpuwm.local_da_controller import capability_contract
    from gpuwm.local_da_score import SUMMARY_SCHEMA as NOWCAST_SUMMARY_SCHEMA
    from gpuwm.verify.obs.nowcast import (
        DEFAULT_LEAD_MINUTES as NOWCAST_LEAD_MINUTES,
        LEAD_STATUSES as NOWCAST_LEAD_STATUSES,
        NOWCAST_SCORE_SCHEMA)
    return dict(schema='arwen.companion-local-da.v1', request_schema=REQUEST_SCHEMA,
        continuous=capability_contract(), optional_review_fields=['continuous'],
        background_catalog=catalog(), review_schema=SCHEMA, request_fields=list(Request.__dataclass_fields__),
        card_fields=list(Card.__dataclass_fields__),
        review_fields=sorted(REVIEW_FIELDS), document=COMPANION_DOCUMENT,
        alternative_fields=sorted(ALTERNATIVE_FIELDS),
        unpriced_row_discriminator='priced',
        refusal_codes=dict(review=sorted(REVIEW_REFUSAL_CODES),
                           launch=sorted(LAUNCH_REFUSAL_CODES),
                           run=sorted(RUN_REFUSAL_CODES),
                           fallback=FALLBACK_REFUSAL_CODE),
        refusal_effects={group: dict(execution_document=document,
                                     forecast_started_can_be_true=started)
                         for group, document, started in REFUSAL_EFFECTS},
        refusal_fields=['schema', 'error', 'code', 'details', 'forecast_started'],
        refusal_discriminator='error',
        point_order=['latitude', 'longitude'], region_order=['west', 'south', 'east', 'north'],
        maximum_forecast_members=MAX_MEMBERS, baseline_cadence_seconds=TUNED_BASELINE_INTERVAL_S,
        defaults=dict(scale=1, forecast_seconds=DEFAULT_FORECAST_SECONDS,
            budget_seconds=DEFAULT_BUDGET_SECONDS, host_gib=32., speed_factor=1.),
        nowcast_score=dict(schema=NOWCAST_SCORE_SCHEMA, summary_schema=NOWCAST_SUMMARY_SCHEMA,
            default=True, receipt_relative_path='nowcast-score.json',
            window_receipt_relative_path='continuous/window_{index:06d}/nowcast-score.json',
            lead_minutes=list(NOWCAST_LEAD_MINUTES), lead_statuses=list(NOWCAST_LEAD_STATUSES),
            baseline='radar persistence at the same lead, with the difference',
            summary_lead_fields=['lead_minutes', 'valid_time', 'status', 'primary_fss',
                'persistence_primary_fss', 'difference_primary',
                'primary_observed_base_rate', 'primary_model_base_rate', 'reason'],
            empty_box_note=('an FSS of 1 with primary_observed_base_rate 0.0 means the radar '
                'found no echo at the primary threshold in the scored interior, not a perfect '
                'forecast: persistence scores 1 there too and the difference is 0.0'),
            score_command=['local-da', '--score', '{plan_path}']),
        commands=dict(review=['local-da', '--request-json', '-', '--dry-run'],
            publish=['local-da', '--request-json', '-', '--out', '{directory}'],
            launch=['local-da', '--launch', '{plan_path}'],
            score=['local-da', '--score', '{plan_path}'],
            read_review=['local-da', '--launch', '{plan_path}', '--dry-run']),
        transport='one JSON document on stdout; diagnostics on stderr; argv without a shell',
        confirmation='Display the published review and require explicit launch approval.')


def _source_role_arguments(values):
    result = {}
    for binding in values:
        role, separator, path = binding.partition('=')
        if not role or not separator or not path or role in result:
            raise PlanError('Each original source input needs one unique ROLE=PATH binding; correct the repeated or incomplete role.')
        result[role] = path
    return result


def main(args) -> int:
    from gpuwm.go_cli import GoStageFailed
    try:
        status_plan, stop_plan = getattr(args, 'status', None), getattr(args, 'stop', None)
        score_plan = getattr(args, 'score', None)
        if status_plan or stop_plan or score_plan:
            if (len([v for v in (status_plan, stop_plan, score_plan) if v]) > 1
                    or args.capabilities or args.run or args.launch
                    or args.request_json or args.out or args.point or args.region or args.dry_run):
                raise PlanError('--status, --stop and --score each name one saved plan; remove other review or execution arguments.')
            if score_plan:
                from gpuwm.local_da_score import score_case
                result = score_case(score_plan)
            else:
                from gpuwm.local_da_controller import status_for_plan, stop_plan as stop_continuous
                result = stop_continuous(stop_plan) if stop_plan else status_for_plan(status_plan)
            print(json.dumps(result, allow_nan=False))
            return 0
        if args.capabilities:
            if args.run or args.launch or args.request_json or args.out or args.point or args.region:
                raise PlanError('--capabilities is inspection only; remove execution or location arguments.')
            print(json.dumps(protocol_document(), allow_nan=False))
            return 0
        if args.continuous is not None and args.continuous < 1:
            raise PlanError('--continuous takes a positive number of windows; omit it for the reviewed finite cycle.')
        if args.continuous and args.launch:
            raise PlanError('--continuous is decided at review and saved in the plan; --launch runs the saved plan as it was published.')
        if args.run and (args.dry_run or args.launch):
            raise PlanError('--run contradicts --dry-run or --launch; select review, publish-and-run, or launch-existing.')
        if args.launch and (args.request_json or args.point is not None or args.region is not None or args.out is not None):
            raise PlanError('--launch already names a reviewed configuration; remove new location, request or output arguments.')
        with contextlib.redirect_stdout(sys.stderr):
            if args.launch:
                if args.dry_run:
                    from gpuwm.local_da_runtime import read_plan
                    result = read_plan(args.launch)
                else:
                    from gpuwm.local_da_runtime import launch
                    result = launch(args.launch)
            else:
                if args.request_json:
                    raw = json.loads(sys.stdin.read() if str(args.request_json) == '-' else Path(args.request_json).read_text())
                    request = request_from_json(raw)
                else:
                    if not args.epoch or args.vram_gib is None:
                        raise PlanError("epoch and vram-gib are required; declare the initial time and card before review.")
                    request = Request(epoch=args.epoch, point=args.point, region=args.region, scale=args.scale,
                        card=Card(args.vram_gib, args.free_gib, args.host_gib, args.speed_factor, args.card_name),
                        budget_seconds=args.budget_seconds, forecast_seconds=args.forecast_seconds,
                        cadence_seconds=args.cadence_seconds, source=args.source, profile=args.profile,
                        source_cycle=args.source_cycle, source_product=args.source_product,
                        source_member=args.source_member, source_provider=args.source_provider,
                        forcing_cadence_hours=args.forcing_cadence_hours,
                        source_root=args.source_root, prepared_root=args.prepared_root,
                        prepared_config=args.prepared_config, prepared_namelist=args.prepared_namelist,
                        source_inputs=_source_role_arguments(args.source_input),
                        supplements=tuple(args.supplement),
                        obs_tables=tuple(args.obs_table), radar_grids=tuple(args.radar_grid),
                        satellite_grids=tuple(args.satellite_grid), base_seed=args.seed,
                        continuous_windows=args.continuous or 0)
                from functools import lru_cache
                from gpuwm.fetch import _head_answer
                # One review samples each URL once; launch keeps its exact
                # selected cycle and rechecks payloads through the fetch owner.
                # A host not heard answers None, which is not "not published".
                result = build_plan(request, background_probe=lru_cache(maxsize=None)(_head_answer))
                if not args.dry_run:
                    if args.out is None:
                        raise PlanError("No output directory was declared; pass --out or use --dry-run for review only.")
                    result['publication'] = publish(result, args.out)
                    if args.run:
                        from gpuwm.local_da_runtime import launch
                        result['execution'] = launch(result['publication']['plan_path'])
                        result['forecast_started'] = result['execution'].get('forecast_started', False)
        print(json.dumps(result, allow_nan=False, default=str))
        return 0
    except (ValueError, RuntimeError, OSError, ImportError, MemoryError, GoStageFailed) as exc:
        message = str(exc) + (' ' + exc.recovery if getattr(exc, 'recovery', None) else '')
        code = FALLBACK_REFUSAL_CODE if isinstance(exc, GoStageFailed) else getattr(exc, 'code', FALLBACK_REFUSAL_CODE)
        details = {'stage_exit_code': exc.code} if isinstance(exc, GoStageFailed) else getattr(exc, 'details', {})
        print(json.dumps(dict(schema=SCHEMA, error=message, code=code,
                              details=details, forecast_started=bool(getattr(exc, 'forecast_started', False))), allow_nan=False))
        return 1


def _tuple(text, n):
    try:
        values = tuple(float(v) for v in text.split(','))
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Use comma-separated numeric coordinates.') from exc
    if len(values) != n:
        raise argparse.ArgumentTypeError(f'Expected {n} comma-separated coordinates.')
    return values


def register_cli(subparsers):
    p = subparsers.add_parser('local-da', help='review and run hardware-scaled local radar, surface and satellite cycling')
    where = p.add_mutually_exclusive_group()
    where.add_argument('--point', type=lambda s: _tuple(s, 2), help='latitude,longitude')
    where.add_argument('--region', type=lambda s: _tuple(s, 4), help='west,south,east,north; east<west crosses the dateline')
    p.add_argument('--epoch', help='initial UTC timestamp on a whole-second boundary, including Z or an explicit offset')
    p.add_argument('--scale', type=int, default=1, help='requested rung of the derived ladder; its domain, resolution, members and cycle count are preserved, with lower rungs shown as alternatives')
    p.add_argument('--vram-gib', type=float, help='declared card memory in GiB; required unless a request document supplies the card')
    p.add_argument('--free-gib', type=float, help='free card memory in GiB when less than the whole card is available; defaults to the declared total')
    p.add_argument('--host-gib', type=float, default=32., help='declared host RAM in GiB for advisory comparison with estimated analysis arrays and observation tables')
    p.add_argument('--speed-factor', type=float, default=1., help='compute speed relative to the printed reference card, not memory capacity')
    p.add_argument('--card-name', default='declared card', help='label recorded beside the timing basis so a review names the card it was priced for')
    p.add_argument('--budget-seconds', type=float, default=DEFAULT_BUDGET_SECONDS, help='advisory wall-time target for preparation, forecast and cycles; does not reduce cycles or impose a runtime deadline')
    p.add_argument('--forecast-seconds', type=float, default=DEFAULT_FORECAST_SECONDS, help='length of the forecast that follows the last analysis, in seconds')
    p.add_argument('--cadence-seconds', type=float, help='exact requested whole-second cadence; otherwise derived from scale; never shortened to meet a cost estimate')
    p.add_argument('--source', help='background source name resolved by its owner; defaults to the package background source')
    p.add_argument('--source-cycle', help='explicit source cycle in UTC; omitted selects one published cycle covering the entire window')
    p.add_argument('--source-product', help='source product selector; source membership never changes the default product implicitly')
    p.add_argument('--source-member', help='one source member, separate from the regional ensemble member count')
    p.add_argument('--source-provider', help='provider supported by the selected product owner')
    p.add_argument('--forcing-cadence-hours', type=int, help='whole-hour source boundary spacing, separate from the whole-second analysis cadence')
    p.add_argument('--source-root', help='local source directory with the existing preparation handoff or native member inventory')
    p.add_argument('--prepared-root', help='existing portable single-domain prepared bundle, verified by the ordinary forecast reader')
    p.add_argument('--prepared-config', help='configuration authority consumed by the supplied prepared bundle')
    p.add_argument('--prepared-namelist', help='WPS authority consumed by the supplied prepared bundle')
    p.add_argument('--source-input', action='append', default=[], help='ROLE=PATH for original source bytes needed to verify a supplied member identity; repeatable')
    p.add_argument('--supplement', action='append', default=[], help='ROLE=PATH consumed by the preparation composition; repeatable')
    p.add_argument('--profile', help='physics profile name resolved by the authoring authority; defaults to the profile that authority selects at the grid spacing of the selected rung')
    p.add_argument('--seed', type=int, default=0, help='base seed for member perturbation and the static covariance samples; the same seed reproduces the same analysis')
    p.add_argument('--obs-table', action='append', default=[], help='existing neutral observation table; repeatable')
    p.add_argument('--radar-grid', action='append', default=[], help='existing radar-grid observation file; repeatable')
    p.add_argument('--satellite-grid', action='append', default=[], help='existing cloud-water-path grid; repeatable')
    p.add_argument('--capabilities', action='store_true', help='print the companion command and field contract without pricing')
    p.add_argument('--request-json', type=Path, help=f'{REQUEST_SCHEMA} file, or - for stdin')
    p.add_argument('--out', type=Path, help='new directory to publish experiment.toml, ensemble.toml, experiment.namelist.wps, every other file the published configuration is read with on its own input route, and local-da.json into; refused if it exists')
    p.add_argument('--dry-run', action='store_true', help='review only; no writes, downloads or device allocation')
    p.add_argument('--json', action='store_true', help='emit the review as one JSON document on stdout, which this door always does; accepted so a companion can state it')
    p.add_argument('--run', action='store_true', help='launch after publishing the reviewed configuration')
    p.add_argument('--launch', type=Path, help='launch or resume an existing local-da.json')
    p.add_argument('--continuous', type=int, metavar='WINDOWS', help='cycle continuously for WINDOWS analysis windows at the reviewed cadence: each window restarts from the previous analysis, assimilates, forecasts and renders, and the boundary forcing is renewed from the same source cycle when a window reaches past it; --status and --stop address the saved plan')
    p.add_argument('--status', type=Path, metavar='PLAN', help='print the continuous status document of a saved plan with its controller liveness, and exit')
    p.add_argument('--stop', type=Path, metavar='PLAN', help='ask the running continuous controller of a saved plan to stop after its current operation; the request is durable, and a launch made while no controller runs clears it and resumes, so ask again after that launch to stop it')
    p.add_argument('--score', type=Path, metavar='PLAN', help='score every still-unscored nowcast lead of a saved plan now and exit: each completed window is graded against the MRMS composite nearest 15, 30, 45 and 60 minutes after its analysis, beside the radar-persistence baseline and the difference, and each window receipt is rewritten; a run scores its own leads by default, so this is for the leads whose valid time had not arrived when the run finished')
    p.set_defaults(func=main)
    return p
