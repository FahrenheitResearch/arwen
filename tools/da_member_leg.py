"""One public DA trajectory leg, shared by serial and packed controllers.

The worker restores and publishes the complete tree restart through its
existing owner. It returns host mirrors and observation-operator arrays.
Input restart deletion belongs to the controller after the member barrier.
No device owner escapes the call and CUDA imports remain lazy.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import functools
import dataclasses
import math
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np

CONTEXT_SCHEMA = "gpuwm-da.member-leg-context.v1"
RESULT_SCHEMA = "gpuwm-da.member-leg-result.v1"
CONTROL = "control"
PHYSICS_VOCABULARY_FIELDS = ("maturity", "registry_sha256", "registry_physics")


@dataclass
class MemberContext:
    """Everything a process needs to run any member's leg.

    Built once per process.  The serial driver builds one; each worker
    builds its own from the same arguments, which is why a worker's
    ``wire`` is the driver's ``wire`` and not a copy of it.
    """
    args: argparse.Namespace
    inputs: object
    exp: object
    cfg: object
    dt: float
    cfg_perturb: object
    hot_cfg: object
    perturbation_report: dict
    vocabulary_divergence: dict
    #: per-leg observation documents, cached by leg index
    _obs_cache: dict = dataclasses.field(default_factory=dict)

def build_member_context(args) -> MemberContext:
    """Run the prepared-authority preflight and build the ensemble config.

    This is the driver's own front-door binding, executed identically in
    the driver and in every worker.  It is deliberately not passed
    across the process boundary: re-deriving it means a worker verifies
    the same hashes the parent did rather than trusting them.
    """
    from gpuwm.da import moments, perturb
    from gpuwm.da.hotstart import HotStartConfig
    import gpuwm.prepared_single_domain_forecast as psdf
    from gpuwm.prepared_single_domain_forecast import (
        preflight_prepared_forecast)

    authority = (args.authority_dir if args.authority_dir is not None
                 else args.prepared_root.parent / "authority")

    _orig_check = psdf._validate_front_door_physics_proof
    vocabulary_divergence: dict = {}

    def _tolerant_check(proof, *, source, profile, cfg):
        try:
            return _orig_check(proof, source=source, profile=profile,
                               cfg=cfg)
        except ValueError as error:
            if (not args.tolerate_physics_vocabulary_drift
                    or "physics selection differs" not in str(error)):
                raise
            selected = dict(proof["physics"])
            expected = dict(psdf.validate_single_domain_physics_profile(
                profile, config=cfg,
                expert_acknowledgements=tuple(
                    selected["acknowledgements"]),
                acknowledgement_provenance=selected[
                    "acknowledgement_provenance"]))
            diverged = {}
            for key in PHYSICS_VOCABULARY_FIELDS:
                if selected.get(key) != expected.get(key):
                    diverged[key] = {"proof": selected.pop(key, None),
                                     "branch": expected.pop(key, None)}
            if selected != expected:
                raise
            vocabulary_divergence.update(diverged)
            return dict(proof["physics"])

    psdf._validate_front_door_physics_proof = _tolerant_check
    try:
        inputs = preflight_prepared_forecast(
            source=args.source, prepared_root=args.prepared_root,
            proof_sha256=args.proof_sha256,
            source_manifest_sha256=args.source_manifest_sha256,
            prepared_content_sha256=args.prepared_content_sha256,
            experiment_config=authority / "experiment.toml",
            wps_namelist=authority / "namelist.wps",
            physics_profile=args.physics_profile,
            run_seconds=args.run_seconds,
            history_interval_seconds=args.history_interval_seconds)
    finally:
        psdf._validate_front_door_physics_proof = _orig_check

    exp = inputs.experiment
    cfg = exp.root.run

    perturb_fields = [
        {"name": "u", "amplitude": args.wind_sigma_ms,
         "length_scale_km": args.length_scale_km},
        {"name": "v", "amplitude": args.wind_sigma_ms,
         "length_scale_km": args.length_scale_km},
    ]
    perturb_species: list[dict] = []
    perturbation_report: dict = {}
    if args.hydrometeors:
        scheme = moments.scheme_moments(int(cfg.mp_physics))
        perturb_fields += [
            {"name": "theta", "amplitude": args.theta_sigma_k,
             "length_scale_km": args.thermo_length_scale_km,
             "vertical_scale_levels": args.thermo_vertical_levels},
            {"name": "qv", "amplitude": args.qv_log_sigma,
             "length_scale_km": args.thermo_length_scale_km,
             "vertical_scale_levels": args.thermo_vertical_levels,
             "mode": "lognormal", "clip_sigmas": args.clip_sigmas},
        ]
        perturb_species = [
            {"mass_field": name, "amplitude": args.hydro_log_sigma,
             "length_scale_km": args.thermo_length_scale_km,
             "vertical_scale_levels": args.thermo_vertical_levels,
             "clip_sigmas": args.clip_sigmas,
             "threshold_kg_kg": scheme.q_threshold}
            for name in scheme.mass_fields
            if name in perturb.SUPPORTED_SPECIES]
        perturbation_report = {
            "scheme": scheme.name, "mp_physics": scheme.mp_physics,
            "species": [spec["mass_field"] for spec in perturb_species]}
    cfg_perturb = perturb.PerturbationConfig.from_mapping({
        "dx_km": float(cfg.dx) / 1000.0, "dy_km": float(cfg.dy) / 1000.0,
        "rim_width": 5,
        "fields": perturb_fields,
        "species": perturb_species,
        "wind_mode": str(getattr(args, "wind_perturbation", "rotational")),
        "hypsometric_opt": int(cfg.hypsometric_opt),
    })

    return MemberContext(
        args=args, inputs=inputs, exp=exp, cfg=cfg, dt=float(cfg.dt),
        cfg_perturb=cfg_perturb, hot_cfg=HotStartConfig(),
        perturbation_report=perturbation_report,
        vocabulary_divergence=vocabulary_divergence)


@dataclass
class MemberLegContext:
    args: Any
    inputs: Any
    identity: Any
    cfg_perturb: Any
    hot_cfg: Any
    leg: int
    absolute_leg_number: int
    t_start: float
    t_end: float
    stage_root: Path
    out: Path
    resumed: bool = False
    restart: Path | None = None
    pending: Mapping[str, np.ndarray] | None = None
    nest_child_dc: Any | None = None
    nested: bool = False
    nest_birth: float | None = None
    document: dict | None = None
    analysis_due: bool = False
    tten_reflectivity: np.ndarray | None = None
    obs_path: Path | None = None
    surface_enabled: bool = False
    setup_arrays: dict[str, np.ndarray] | None = None
    thb_host: np.ndarray | None = None
    #: --iau-mode 4d: this leg starts half a window before its analysis,
    #: from the member's own restart there (:class:`Iau4dLegRunner`).
    iau_rewound: bool = False

    def validate(self, name) -> None:
        if int(self.leg) != self.leg or self.leg < 0 or int(self.absolute_leg_number) != self.absolute_leg_number or self.absolute_leg_number < 0:
            raise ValueError("member leg indices must be nonnegative exact integers")
        if not math.isfinite(self.t_start) or not math.isfinite(self.t_end) or self.t_start < 0 or self.t_end <= self.t_start:
            raise ValueError("member leg clocks must be finite with 0 <= start < end")
        if name != CONTROL and (not isinstance(name, (int, np.integer)) or isinstance(name, bool) or int(name) < 0 or int(name) >= int(self.identity.members)):
            raise ValueError("member identity is outside the declared ensemble")
        if self.nested and self.nest_child_dc is None:
            raise ValueError("a nested member leg requires its child domain configuration")
        if self.nest_birth is not None and (not math.isfinite(self.nest_birth) or self.nest_birth < 0 or self.nest_birth > self.t_start):
            raise ValueError("child birth must be a finite elapsed clock at or before this leg")
        if (self.leg != 0 or self.resumed) and self.restart is None:
            raise ValueError("a continued member leg requires its complete tree restart")
        if self.restart is not None and self.leg == 0 and not self.resumed:
            raise ValueError("a fresh first member leg cannot silently ignore a supplied restart")
        if self.leg == 0 and not self.resumed and self.t_start != 0:
            raise ValueError("a fresh first member leg starts at elapsed zero; a later clock requires a restart")
        if not self.resumed and self.leg == 0 and self.pending:
            raise ValueError("pending analysis requires a restored member, not a fresh first leg")


@dataclass
class MemberLegResult:
    restart: Path
    #: The in-process host mirror of the leg-end state.  ``None`` once the
    #: result has crossed the worker boundary: the restart is that state.
    snapshot: dict[str, np.ndarray] | None
    H_Z: np.ndarray | None
    H_surface: dict[str, np.ndarray] | None
    hot_pending: dict[str, np.ndarray] | None
    setup_arrays: dict[str, np.ndarray]
    thb_host: np.ndarray | None
    nest_birth: float | None
    record: dict
    consume_restart: Path | None
    pending_consumed: bool


def capture_surface_diagnostics(driver, *, dewpoint=False):
    """Take the live surface H(x); PSFC is the surface driver's Pa field."""
    from tools.da_cycle_prepared import to_host
    names = ("t2", "u10", "v10") + (("q2", "psfc") if dewpoint else ())
    absent = [name for name in names if name not in driver.fields]
    if absent:
        raise RuntimeError(f"surface observations need live driver diagnostics {absent}")
    return {name: to_host(driver.fields[name]).astype(np.float64) for name in names}


class HistoryWriter:
    """One background thread that writes a leg's history frames in order.

    The frame is read from the live state at the alarm; compressing it and
    writing its wrfout copy happen here, beside the next steps.
    :meth:`close` waits for every frame and raises the first failure, so a
    leg never reports success over a frame that was not written.
    """

    def __init__(self):
        from concurrent.futures import ThreadPoolExecutor

        self._pool = ThreadPoolExecutor(max_workers=1,
                                        thread_name_prefix="history-writer")
        self._futures = []

    def submit(self, function):
        future = self._pool.submit(function)
        self._futures.append(future)
        return future

    def abandon(self):
        """Wait for the thread without raising: the leg already failed."""
        self._pool.shutdown(wait=True)
        self._futures = []

    def close(self):
        try:
            for future in self._futures:
                future.result()
        finally:
            self._pool.shutdown(wait=True)
            self._futures = []


def rain_history_handler(*, context, name, drivers, writer=None, native=None,
                         first_frame=None):
    """Read-only public rain/echo output at scheduled native history clocks.

    ``native``, when given, is a dict this handler fills with the latest
    scheme-native REFL_10CM column maximum per grid id, ``{gid: (ticks,
    colmax)}``, at EVERY history alarm, retained frame or not: the leg-end
    composite reads it so the leg frames and the free-forecast frames carry
    one reflectivity product.  Passing it also makes the leg request the
    native producer when ``--rain-history`` is off.

    ``first_frame``, when given, names the composite written at the leg's
    FIRST history alarm (the analysed state's first native frame, two
    minutes after an analysis on the CONUS cases), whether or not the
    rain history retains that frame: the echo an analysis inserted, as the
    model first shows it, beside the leg's start composite.
    """
    if not getattr(context.args, "rain_history", False) and native is None:
        return None
    pending_first = [first_frame]
    retain = bool(getattr(context.args, "rain_history", False))
    start = float(context.args.rain_forecast_start_seconds)
    from tools.da_cycle_prepared import _column_max_float32, _write_rain_composite
    from gpuwm.core.refl import consume_refl_10cm, domain_start_ticks_of
    from gpuwm.prepared_single_domain_forecast import _consume_due_native_refl_10cm
    def write(model, node, ticks):
        # The executor requested a native producer at every history alarm.
        # Drain it even when this check does not retain that frame. Skipping
        # its consumer leaves the previous field staged at the next MP call.
        refl = _consume_due_native_refl_10cm(node.state, ticks, consume_refl_10cm,
            domain_start_ticks=domain_start_ticks_of(node))
        gid = int(node.cfg.grid_id)
        if native is not None and refl is not None:
            native[gid] = (int(ticks), _column_max_float32(refl))
        elapsed = float(node.clock.elapsed_seconds)
        if (pending_first[0] is not None and gid == 1 and refl is not None
                and elapsed > float(context.t_start)):
            _write_rain_composite(
                Path(context.out)/"composites"/pending_first[0],
                state=node.state, driver=drivers[gid], grid=node.grid,
                cfg=node.cfg.run, elapsed_seconds=elapsed,
                exp=context.inputs.experiment,
                label=f"first frame after the analysis, member {name}",
                reflectivity=refl, writer=writer)
            pending_first[0] = None
        if not retain or elapsed <= start:
            return
        suffix = "" if gid == 1 else f"_d{gid:02d}"
        path = Path(context.out)/"composites"/(
            f"history{int(ticks):012d}t{int(node.clock.tick_den)}_{name}{suffix}.npz")
        _write_rain_composite(path, state=node.state, driver=drivers[gid],
            grid=node.grid, cfg=node.cfg.run, elapsed_seconds=elapsed,
            exp=context.inputs.experiment, label=f"free forecast member {name}",
            domain=None if gid == 1 else node.cfg, reflectivity=refl,
            writer=writer)
    return write


class WireOverride(Protocol):
    """Typed in-process fixture seam, never represented by production JSON."""
    def __call__(self, run_seconds_total: float, *, child_dc=None): ...


class _WorkerStage:
    """Select output directories and record a source-consumption intent."""
    def __init__(self, root):
        self.root = Path(root) / "restart"
        self.consume_intent: Path | None = None

    def directory(self, leg_number, name):
        from tools import da_ensemble_state as ens_state
        return self.root / f"leg{int(leg_number):03d}" / ens_state.trajectory_key(name)

    def consume(self, source):
        # A generation outside this stage is durable and has no intent.
        path = Path(source)
        if path.resolve().is_relative_to(self.root.resolve()):
            self.consume_intent = path
        return False


#: 4d: the hydrometeor increments' own window, from the analysis time.
HYDROMETEOR_WINDOW_SECONDS = 300.0


def insert_analysis(state, to_apply, *, mode, window_seconds, t_start,
                    hydrostatic, cfg, iau_fields, apply_increments,
                    refresh) -> dict:
    """Put one member's analysis into its restored state (DA lane 7).

    ``oneshot`` (with the hydrostatic rebalance; the plain one-shot stays
    the caller's own path): the applier writes the increment, then
    ``php`` is re-integrated so the analysed columns are hydrostatic at
    their own dry mass (gpuwm.da.hydrostatic).

    ``3d``: the hydrometeor part goes in now (spread, the scheme removes
    between fractions the rain the analysis takes out, and the rest clips
    at zero; the box E smoke clipped about 50,000 qr cells a step), with
    its loading's ``php`` change when the rebalance is on; the dynamics part
    has its vapour bounded as the applier bounds it and, with the
    rebalance, gains its own ``php`` change; it is then returned as an
    :class:`gpuwm.da.iau.IauForcing` over ``[t_start, t_start + window]``
    for the caller to attach for the integration.

    Returns ``{"forcing", "spread", "record"}``.
    """
    from gpuwm.da import hydrostatic as hydro
    from gpuwm.da.iau import (IauForcing, IauNode, box_density,
                              cap_vapour_increment, split_fields)

    hopt = int(cfg.hypsometric_opt)
    record = {"insertion": mode, "hydrostatic_rebalance": bool(hydrostatic)}
    if mode == "oneshot":
        before = hydro.capture_column(state)
        receipt = apply_increments(state, to_apply, mp_physics=cfg.mp_physics)
        refresh(state, hypsometric_opt=hopt)
        record["apply_fields"] = receipt["field_count"]
        record["hydrostatic"] = hydro.rebalance_after_insertion(
            state, before, hypsometric_opt=hopt, names=sorted(to_apply))
        del before
        return {"forcing": None, "spread": None, "record": record}
    if mode not in ("3d", "4d"):
        raise NotImplementedError(
            f"--iau-mode {mode} is not wired in the member leg")
    # 4d: the caller's leg starts half a window before the analysis
    # (Iau4dLegRunner), so the same box from the leg start is the window
    # centred on the analysis time.
    spread, at_once = split_fields(to_apply, iau_fields)
    hydrometeor_node = None
    if at_once and mode == "4d":
        # The analysis is valid at the window centre, half a window after
        # this leg's start: its hydrometeors go in there, over five
        # minutes (a short window, so the scheme removes little between
        # its steps and the clip the box E smoke found stays small), with
        # their loading's php.
        refresh(state, hypsometric_opt=hopt)
        at_centre = dict(at_once)
        if hydrostatic:
            dphp, record["hydrostatic_hydrometeors"] = (
                hydro.geopotential_increment(state, at_once,
                                             hypsometric_opt=hopt))
            if dphp is not None:
                at_centre["php"] = dphp
        hydrometeor_node = IauNode(
            at_centre, box_density(t_start + 0.5 * float(window_seconds),
                                   HYDROMETEOR_WINDOW_SECONDS),
            label="hydrometeors at the analysis time", group="hydrometeors")
        record["hydrometeor_window_seconds"] = [
            t_start + 0.5 * float(window_seconds),
            t_start + 0.5 * float(window_seconds)
            + HYDROMETEOR_WINDOW_SECONDS]
    elif at_once:
        before = hydro.capture_column(state) if hydrostatic else None
        apply_increments(state, at_once, mp_physics=cfg.mp_physics)
        refresh(state, hypsometric_opt=hopt)
        if hydrostatic:
            record["hydrostatic_at_start"] = hydro.rebalance_after_insertion(
                state, before, hypsometric_opt=hopt, names=sorted(at_once))
        del before
    else:
        refresh(state, hypsometric_opt=hopt)
    spread, record["vapour_cap"] = cap_vapour_increment(state, spread)
    if hydrostatic:
        dphp, record["hydrostatic_spread"] = hydro.geopotential_increment(
            state, spread, hypsometric_opt=hopt)
        if dphp is not None:
            spread = dict(spread)
            if "php" in spread:
                dphp = dphp + spread["php"]
            spread["php"] = dphp
    forcing = None
    nodes = []
    if spread:
        nodes.append(IauNode(spread, box_density(t_start, window_seconds),
                             label="analysis"))
    if hydrometeor_node is not None:
        nodes.append(hydrometeor_node)
    if nodes:
        forcing = IauForcing(
            state, nodes,
            provenance={"mode": mode, "window_seconds": float(window_seconds),
                        "window_start_seconds": float(t_start)})
    record["apply_fields"] = len(to_apply)
    record["iau_split"] = {"spread": sorted(spread),
                           ("at_analysis_time" if mode == "4d"
                            else "at_leg_start"): sorted(at_once)}
    return {"forcing": forcing, "spread": spread, "record": record}


def carry_iau_to_child(parent_state, child_state, spread, parent_forcing, *,
                       child_dc_leg, cfg, hydrostatic, t_start,
                       positivity_policy, nest_entry):
    """The child's share of a spread analysis, as its own forcing.

    The same operator the one-shot carry-down uses
    (:func:`gpuwm.da.nested_forecast.nest_down_analysis`: SINT of the
    analysed parent minus SINT of the background parent, never the
    interpolated increment), with the analysed side the live parent plus
    the spread increment it has not received yet.  ``php`` is the parent's
    hydrostatic response and is not carried; the child forms its own from
    its correction.  The child's forcing uses the parent's time density.
    """
    import cupy as cp

    from gpuwm.da import hydrostatic as hydro
    from gpuwm.da import nested_forecast
    from gpuwm.da.iau import IauForcing, IauNode
    from gpuwm.ensemble.member import refresh_diagnostics
    from tools.da_cycle_prepared import bound_child_correction, to_host

    fields = sorted(name for name in spread if name != "php"
                    and getattr(child_state, name, None) is not None)
    if not fields or parent_forcing is None:
        return None
    background = {name: to_host(getattr(parent_state, name))
                  for name in fields}
    analysed = {name: getattr(parent_state, name)
                + cp.asarray(spread[name]).astype(
                    getattr(parent_state, name).dtype)
                for name in fields}
    correction = nested_forecast.nest_down_analysis(
        background, analysed, child_dc_leg, cfg, array_module=cp)
    del background, analysed
    correction, positivity = bound_child_correction(
        child_state, correction, policy=positivity_policy, array_module=cp)
    correction = {name: delta for name, delta in correction.items()
                  if float(cp.abs(delta).max()) > 0.0}
    record = {"how": ("sint(live parent + spread increment) - sint(live "
                      "parent), forced over the parent's window"),
              "fields": sorted(correction)}
    if positivity is not None:
        record["positivity_on_correction"] = positivity
    if not correction:
        nest_entry["iau_carried_down"] = record
        return None
    hopt = int(child_dc_leg.run.hypsometric_opt)
    refresh_diagnostics(child_state, hypsometric_opt=hopt)
    if hydrostatic:
        dphp, record["hydrostatic"] = hydro.geopotential_increment(
            child_state, correction, hypsometric_opt=hopt)
        if dphp is not None:
            correction["php"] = dphp
    nodes = [IauNode(correction, node.knots, label=node.label)
             for node in parent_forcing.nodes]
    if len(nodes) != 1:
        raise NotImplementedError(
            "a multi-slot forcing is not carried into a nest yet")
    forcing = IauForcing(child_state, nodes,
                         provenance={"carried_from_parent": True})
    record["max_abs_correction"] = {
        name: float(cp.abs(delta).max()) for name, delta in correction.items()}
    nest_entry["iau_carried_down"] = record
    return forcing


def attach_iau(state, forcing, *, t_start: float) -> None:
    """Attach ``forcing`` after checking the state's clock is the one its
    window was stated on: a forcing whose window the integration never
    reaches would add nothing and record nothing wrong."""
    from gpuwm.da.iau import IauError, attach

    elapsed = float(state.elapsed_seconds)
    if abs(elapsed - float(t_start)) > 1e-6:
        raise IauError(
            f"the state's clock reads {elapsed} s and the forcing's window "
            f"was stated from the leg start {t_start} s")
    attach(state, forcing)
@functools.lru_cache(maxsize=8)
def _wrfout_grid_identity(path: str) -> str:
    from gpuwm.obs.target_grid import TargetGrid
    return TargetGrid.from_wrfout(Path(path)).identity_sha256()


def window_grid_identity(args, leg: int) -> str:
    """The grid identity a leg's heating windows are read against: the
    leg's own observation grid (``--grid-wrfout`` for this leg), the grid
    its observation document is already held to.  ``tools/
    radar_tten_windows.py --grid-wrfout`` on the same file writes windows
    with this identity; a window gridded anywhere else is refused."""
    grids = list(getattr(args, "grid_wrfout", None) or ())
    if int(leg) >= len(grids):
        raise ValueError(
            f"leg {leg} has no --grid-wrfout to read its heating windows "
            "against: a window read without the leg's grid identity may hold "
            "other columns")
    return _wrfout_grid_identity(str(Path(grids[int(leg)]).resolve()))


def _leg_radar_forcing(radar_tten, args, state, name, leg_reflectivity,
                       obs_path, *, leg_start_valid, leg_minutes: float,
                       grid_identity: str | None = None):
    """The radar forcing one trajectory carries through one observed leg.

    With ``--radar-tten-windows`` the leg is split into 15-minute windows
    (design E3), each heated from its own gridded volume valid at the
    window's end; without it, NOAA's one-slot research arrangement this
    line had: the leg-end observation file for the whole leg.  Heating
    (``--radar-tten-mode heating``) uses NOAA's builder with the run's
    options, drawn per member when ``--radar-tten-perturb-members`` is on;
    ``lhn`` is the latent heat nudging comparison arm.
    """
    import cupy as cp

    windows_root = getattr(args, "radar_tten_windows", None)
    if windows_root:
        windows = max(int(round(float(leg_minutes) / 15.0)), 1)
        paths = radar_tten.window_paths(windows_root, leg_start_valid,
                                        leg_minutes, windows)
        refs, sources = [], []
        for path in paths:
            # The strict reader: schema, status, digest and the leg's grid
            # identity, on every window (lane 6 code map, section 1).
            ref, receipt = radar_tten.read_window_reflectivity(
                path, identity_sha256=grid_identity)
            refs.append(ref)
            sources.append({"path": str(path),
                            "valid_time": (receipt.get("window") or {}).get("end"),
                            "counts": receipt.get("counts")})
        times = radar_tten.window_end_minutes(leg_minutes, windows)
        observations = f"{windows} gridded windows under {windows_root}"
    else:
        reflectivity = leg_reflectivity
        if isinstance(reflectivity, np.ndarray):
            reflectivity = cp.asarray(np.ascontiguousarray(
                reflectivity, dtype=np.float32))
        refs, sources = [reflectivity], None
        times = (float(leg_minutes),)
        observations = str(obs_path)
    provenance = {"observations": observations,
                  "background": "trajectory state at leg start",
                  "trajectory": str(name)}
    if getattr(args, "radar_tten_mode", "heating") == "lhn":
        return radar_tten.build_nudging(state, refs, times, sources=sources,
                                        provenance=provenance)
    base = radar_tten.RadarTtenConfig(
        latent_heat_period_min=float(
            getattr(args, "radar_tten_dt_cond", None) or 20.0),
        pbl_extension=bool(getattr(args, "radar_tten_pbl_extension", False)),
        strict_suppression=bool(
            getattr(args, "radar_tten_strict_suppression", False)))
    member = None
    if name != CONTROL and getattr(args, "radar_tten_perturb_members", False):
        member = int(str(name).lstrip("m"))
    config, perturbation = radar_tten.member_config(
        base, member, int(getattr(args, "seed", 0) or 0))
    provenance["member_parameters"] = perturbation
    return radar_tten.build_forcing(
        state, refs, times, config, sources=sources, provenance=provenance)


#: One member's own boundary tables per process and setting: a serial
#: roster runs every member's legs in one process, a packed worker one leg.
_MEMBER_BOUNDARIES: dict = {}


def _member_boundaries(name, state, cfg_perturb, args):
    """``(tables or None, record)``: the member's perturbed lateral boundary
    tables (gpuwm.da.perturb.perturbed_lateral_boundaries), or ``None`` with
    the stated reason when the state has no eager tables to perturb."""
    from gpuwm.da import perturb

    key = (int(name), int(args.seed), float(args.boundary_perturbation),
           float(args.boundary_perturbation_hours))
    if key not in _MEMBER_BOUNDARIES:
        shared = getattr(state, "lateral_boundaries", None)
        reason = perturb.boundary_perturbation_unavailable(shared)
        if reason is not None:
            _MEMBER_BOUNDARIES[key] = (None, {
                "schema": perturb.BOUNDARY_PERTURBATION_SCHEMA,
                "perturbed": False, "reason": reason})
        else:
            _MEMBER_BOUNDARIES[key] = perturb.perturbed_lateral_boundaries(
                shared, cfg_perturb, seed=args.seed + int(name),
                coupling=perturb.boundary_coupling_weights(state),
                scale=float(args.boundary_perturbation),
                time_scale_hours=float(args.boundary_perturbation_hours))
    return _MEMBER_BOUNDARIES[key]


def run_member_leg(context: MemberLegContext, name, *,
                   wire_override: WireOverride | None = None,
                   assemble_override: Callable | None = None) -> MemberLegResult:
    """Restore, apply pending increments, forecast, then return restart + H(x).

    The forecast body and default wiring were extracted from the public
    prepared-cycle driver at a9ade5f32, retaining its numerical call order.
    ``wire_override`` and ``assemble_override`` are explicit in-process GPU
    fixture seams. No production manifest can name or import either one.
    """
    context.validate(name)
    vts_enabled = bool(getattr(context.args, "spread_repair_vtsm", False)
                       and context.analysis_due and name != CONTROL)
    if vts_enabled:
        from gpuwm.da.spread_vts import validate_request
        validate_request(analysis_seconds=context.t_end,
                         origin_seconds=context.t_start,
                         history_seconds=context.args.history_interval_seconds,
                         observation_heating=context.tten_reflectivity is not None)
    import dataclasses
    import time
    from datetime import timedelta
    from types import MappingProxyType, SimpleNamespace

    import cupy as cp
    from gpuwm.core.clock import build_schedule, resolve_clock
    from gpuwm.core.health import StateHealthValidator
    from gpuwm.core.model import DomainNode, ExperimentState, ModelRuntimeStatus, execute_experiment
    from gpuwm.da import nested_forecast, obsop, perturb, radar_tten
    from gpuwm.da.hotstart import hotstart_increments
    from gpuwm.ensemble.increments import apply_increments
    from gpuwm.ensemble.member import refresh_diagnostics
    from gpuwm.ingest.hrrr_physics import initialize_prepared_physics
    from gpuwm.ingest.prepared_cache import restore_prepared_cache
    from gpuwm.io.restart import tree_restart_members
    from gpuwm.runtime import declared_constant_glw
    from gpuwm.state_serialization_contract import STATE_SERIALIZED_ATTRS
    from tools.da_cycle_prepared import (
        REFL_PRODUCT_NATIVE, REFL_PRODUCT_OPERATOR,
        _write_composite_wrfout, _write_rain_composite, bound_child_correction,
        keep_moment_pairs, rain_snapshot, restart_domain_ids, restore_leg_restart,
        to_host, trajectory_fingerprint, trajectory_identity, write_leg_restart,
    )

    args, inputs, identity = context.args, context.inputs, context.identity
    exp = inputs.experiment
    cfg = exp.root.run
    cfg_perturb, hot_cfg = context.cfg_perturb, context.hot_cfg
    leg, t_start, t_end = context.leg, context.t_start, context.t_end
    resumed_from = {} if context.resumed else None
    restarts = {name: context.restart}
    pending = {name: context.pending}
    hot_pending, member_dbz, member_sfc, snapshots = {}, {}, {}, {}
    nest_child_dc = context.nest_child_dc
    nest_birth = {name: context.nest_birth}
    setup_arrays, thb_host = context.setup_arrays, context.thb_host
    document, analysis_due = context.document, context.analysis_due
    tten_reflectivity, obs_path = context.tten_reflectivity, context.obs_path
    surface_cfg = True if context.surface_enabled else None
    out = Path(context.out)
    stage = _WorkerStage(context.stage_root)
    leg_record = {"trajectories": {}}
    z_obs_cp = z_mask_cp = None
    if document is not None and not args.no_hotstart:
        z_obs_cp = cp.asarray(np.asarray(document["variables"]["z_obs"], np.float32))
        z_mask_cp = cp.asarray(np.asarray(document["variables"]["z_mask"]).astype(bool))

    def leg_number(index):
        return context.absolute_leg_number + int(index) - context.leg

    def leg_length(index):
        if index != leg:
            raise ValueError("member worker owns only the current leg duration")
        return t_end - t_start

    def nests_this_leg(index, trajectory):
        return context.nested

    def child_config_for(trajectory, born_at):
        return nested_forecast.child_born_at(nest_child_dc, exp, born_at)

    def wire(run_seconds_total: float, *, child_dc=None):
        """Build one leg's parent model, and the clocks the child needs.

        With ``child_dc`` the clock and the schedule are the TWO-domain
        ones, but the child itself is NOT built here.  It is derived from
        the parent's state by SINT, and which parent state that is
        depends on the leg: a child born this leg is built from the
        ANALYSED parent after the restart set is restored and the
        increment applied, and a child the trajectory already carries is
        built before the restore so the restore can fill it.  Building a
        newborn from a pre-analysis parent would hand the nest a
        forecast nobody corrected, which is the very failure this whole
        design exists to avoid.  :func:`assemble` closes the model once
        the child is real.

        The root's external boundary clock is BOUND here, as
        ``run_prepared_forecast`` binds it, so a leg's Davies relaxation
        consumes WRF's post-increment ``dtbc`` recurrence from the clock
        the restart owner places rather than the retired elapsed-seconds
        calculation, and the checkpoint header records that semantic.
        """
        from gpuwm.ingest.lateral_bc import bind_lateral_boundary_clock

        exp_leg = dataclasses.replace(exp,
                                      run_seconds=float(run_seconds_total))
        live_born = ()
        if child_dc is not None:
            exp_leg = nested_forecast.nested_experiment(exp_leg, child_dc)
            live_born = (int(child_dc.grid_id),)
        restored = restore_prepared_cache(
            inputs.prepared_cache_path, expected_identity=inputs.cache_identity,
            cfg=cfg, static=inputs.static)
        driver = initialize_prepared_physics(
            restored.initial_result, cfg, restored.met, restored.surface,
            inputs.static, inputs.landuse_identity, inputs.grid,
            exp.start_time,
            constant_glw_wm2=declared_constant_glw(exp))
        tick = resolve_clock(
            exp_leg, lbc_interval_s=float(inputs.boundary_interval_seconds),
            live_born_children=live_born)
        schedule = build_schedule(exp_leg, tick)
        clocks = tick.clocks()
        node = DomainNode(exp.root, inputs.grid,
                          restored.initial_result.state, clocks[1],
                          None, [], None)
        if getattr(cfg, "specified", False):
            bind_lateral_boundary_clock(node.state, node.clock)
        return SimpleNamespace(node=node, restored=restored, driver=driver,
                               clocks=clocks, schedule=schedule,
                               child_dc=child_dc)

    def assemble(wired, *, name, child_node=None):
        """Turn the wired pieces into the ExperimentState the executor runs.

        The fingerprint is the TRAJECTORY's (:func:`trajectory_fingerprint`),
        so the restart owner refuses a checkpoint set that belongs to
        another member, another ensemble or another case before it reads
        an array, and names the component that differs.
        """
        nodes = {1: wired.node}
        if child_node is not None:
            nodes[child_node.cfg.grid_id] = child_node
        model = ExperimentState(wired.node, MappingProxyType(nodes),
                                wired.schedule, None,
                                trajectory_fingerprint(identity, name))
        model._experiment_fingerprint_components = trajectory_identity(
            identity, name)
        model._runtime_status = ModelRuntimeStatus()
        model._resumed = False
        model._resume_committed_history_grid_ids = frozenset()
        # The shared scratch arena hands every domain a prefix VIEW of one
        # backing buffer, on the premise that the schedule steps exactly
        # one domain at a time; the shared dycore workspace is the same
        # bargain.  Both stay None on this route -- with a nest attached
        # that is no longer a memory question but a correctness one, and
        # it is asserted rather than assumed.
        model._scratch_arena = None
        model._dycore_state_workspace = None
        if model._scratch_arena is not None \
                or model._dycore_state_workspace is not None:
            raise RuntimeError(
                "the DA cycling path requires per-domain scratch: a "
                "shared arena would let parent and child write the same "
                "bytes with no exception raised")
        model._io_manager = None
        model._last_checkpoint = None
        model._prepared_by_grid_id = MappingProxyType({
            1: SimpleNamespace(static_fields=inputs.static,
                               geog_selection=None,
                               initial_result=wired.restored.initial_result)})
        return model

    def run_trajectory(name) -> None:
        """One trajectory's leg, in a scope that ends with it.

        Every device owner the leg builds -- the restored state, the
        physics driver, the model, the child and its driver, the
        leg-end diagnostics -- is a local of this call, so returning
        drops the last reference to each of them.  What the leg hands
        on leaves as host data only, through the run's own tables:
        the restart set, the host mirror, the leg-end H(x), the hot
        start increments and the leg record.  The trajectory loop body
        used to run in :func:`cycle`'s own scope, where those names
        stayed bound after the old ``teardown`` deleted only its loop
        variable, so the previous trajectory's whole model was still
        alive while the next one was restored beside it, and a domain
        whose one trajectory fits the card ran out of memory building
        its second.
        """
        nonlocal setup_arrays, thb_host
        t_leg = time.time()
        # Wall clock of the leg's phases, in order, so a leg boundary's host
        # cost can be told from the integration itself (the 9 km CONUS
        # members stepped about 25 s of a 70 s worker).
        phases: dict = {}
        mark = [time.perf_counter()]

        def phase(label):
            now = time.perf_counter()
            phases[label] = round(phases.get(label, 0.0) + now - mark[0], 3)
            mark[0] = now
        nested_leg = nests_this_leg(leg, name)
        # -- what this leg starts from ---------------------------------
        #
        # Leg 0 of a fresh run starts from the prepared background
        # (perturbed for a member).  Every other leg starts from the
        # trajectory's own restart set: the one the previous leg
        # wrote, or the one the resumed generation carried.
        source = None
        child_in_checkpoint = False
        born_at = None
        if not (leg == 0 and resumed_from is None):
            source = restarts[name]
            if source is None:
                raise RuntimeError(
                    f"leg {leg} {name}: no restart set to continue "
                    "from; the previous leg wrote none")
            stored_ids = restart_domain_ids(source)
            if nested_leg:
                child_in_checkpoint = (
                    int(nest_child_dc.grid_id) in stored_ids)
            elif len(stored_ids) > 1:
                raise RuntimeError(
                    f"leg {leg} {name}: the restart set carries "
                    f"domains {list(stored_ids)} and this leg runs "
                    "the root alone; a child cannot be dropped at "
                    "a leg boundary silently")
        if nested_leg:
            born_at = (nest_birth[name] if child_in_checkpoint
                       else float(t_start))
        child_dc_leg = (child_config_for(name, born_at)
                        if nested_leg else None)
        phase("before_wire")
        wired = wire(t_end, child_dc=child_dc_leg)
        phase("wire_prepared_state_and_physics")
        node, restored, driver = wired.node, wired.restored, wired.driver
        state = node.state
        if (name != CONTROL
                and float(getattr(args, "boundary_perturbation", 0.0)) > 0.0):
            # The member's own boundaries (audit S4), onto the freshly
            # restored prepared state, before the restart set is restored
            # over it: the attach resets the forcing clock's elapsed time and
            # keeps its binding, which is what a state at time 0 needs.  The
            # draw is the same every leg (seed plus member), so the restart's
            # setup fingerprint, which hashes the attached tables, is stable.
            from gpuwm.ingest.lateral_bc import attach_lateral_boundaries
            tables, boundary_record = _member_boundaries(
                name, state, cfg_perturb, args)
            leg_record["trajectories"].setdefault(str(name), {})[
                "member_lateral_boundaries"] = boundary_record
            if tables is not None:
                attach_lateral_boundaries(state, tables)
        if setup_arrays is None:
            # The eta coordinate arrays and the base column mass a
            # checkpoint does not serialize (STATE_SETUP_ARRAYS in
            # gpuwm/state_serialization_contract.py). The CWP operator
            # integrates in the model's own mass measure and cannot
            # rebuild them from the npz, so they are captured here off
            # a live state, once, exactly as the reflectivity provider
            # would need thb.
            setup_arrays = {
                "c1h": to_host(state.c1h).astype(np.float64),
                "c2h": to_host(state.c2h).astype(np.float64),
                "dnw": to_host(state.dnw).astype(np.float64),
                "mub2d": to_host(state.mub2d).astype(np.float64),
            }
        child_node = child_driver = None
        nest_entry = None
        if nested_leg:
            nest_entry = leg_record["trajectories"].setdefault(
                str(name), {}).setdefault("nest", {})
            nest_entry["grid_id"] = nest_child_dc.grid_id
            nest_entry["born_at_seconds"] = float(born_at)

        def _build_child():
            """The child object, from the parent's live state.

            Its clock is placed at the child's own birth first, and
            the builder refreshes the model time from it before the
            physics is attached, so the child's driver counts its
            ITIMESTEP from its activation.
            """
            clock = wired.clocks[nest_child_dc.grid_id]
            nested_forecast.place_newborn_clock(clock)
            return nested_forecast.build_nested_child(
                node, child_dc_leg,
                static=inputs.static, surface=restored.surface,
                landuse_identity=inputs.landuse_identity,
                valid_time=exp.start_time, clock=clock,
                parent_driver=driver,
                constant_glw_wm2=declared_constant_glw(exp))

        # A child the trajectory already carries is built BEFORE the
        # restore, because the restart owner restores the whole set
        # into the whole tree: the SINT below only builds the object
        # and its base state, and every array it holds is then
        # overwritten from the checkpoint.
        if child_in_checkpoint:
            child_node, child_driver, _land_receipt = _build_child()
        # `resumed` drives model._resumed below, which stops the
        # resumed leg from rewriting history the previous process
        # already committed.  `resumed_from is None` is the separate
        # and more dangerous condition: a generation carried in from
        # --resume-ensemble is ALREADY perturbed, and perturbing it
        # again at leg 0 would throw away the analysis it carried with
        # no error raised anywhere.  Both guards are required; they are
        # not the same question.
        resumed = False
        #: The parent's fields the pending increment names, as the
        #: restart set restored them and BEFORE the increment: the
        #: background side of the correction a carried child takes.
        restored_background: dict = {}
        #: --iau-window-seconds: the increment becomes a forcing over the
        #: leg's first window seconds (gpuwm.da.iau) instead of an
        #: addition at its start.
        iau = None
        iau_window = getattr(args, "iau_window_seconds", None)
        #: --iau-mode 3d/4d: the increment as a held tendency inside every
        #: RK stage (gpuwm.da.iau.IauForcing), attached for the
        #: integration; ``iau_spread`` is what it spreads, for the nest.
        iau_mode = getattr(args, "iau_mode", None) or (
            "split" if iau_window is not None else "oneshot")
        hydrostatic = bool(getattr(args, "hydrostatic_rebalance", False))
        iau_forcing = None
        iau_spread = None
        child_forcing = None
        maspt = None
        maspt_minutes = float(getattr(args, "maspt_minutes", 0.0) or 0.0)
        if source is None:
            if name != CONTROL:
                if maspt_minutes > 0:
                    from gpuwm.da.maspt import MasptRecorder
                    maspt = MasptRecorder.for_config(
                        cfg, start_seconds=float(t_start),
                        horizon_seconds=60.0 * maspt_minutes)
                    maspt.before_insertion(state)
                # The member's perturbation is an insertion too.  The
                # module balances it itself by default (2026-10-06): one
                # streamfunction for the winds, php re-integrated after
                # the thermodynamic draws (gpuwm.da.perturb, wind_mode
                # and mass_balance); its provenance carries both
                # receipts.  The refresh below folds the new php into p.
                leg_record["trajectories"].setdefault(str(name), {})[
                    "perturbation"] = perturb.apply_perturbations(
                        state, args.seed + int(name), cfg_perturb)
                phase("perturb")
                refresh_diagnostics(
                    state, hypsometric_opt=cfg.hypsometric_opt)
                phase("refresh_diagnostics")
        else:
            model = assemble(wired, name=name, child_node=child_node)
            restart_info = restore_leg_restart(
                model, source, expected_seconds=t_start)
            phase("restore_restart")
            resumed = True
            entry_restore = leg_record["trajectories"].setdefault(
                str(name), {})
            entry_restore["restored_from"] = {
                "root_member": Path(source).name,
                "domain_ids": list(restart_domain_ids(source)),
                "elapsed_seconds": (restart_info.elapsed_ticks
                                    / restart_info.tick_den),
            }
            # A staged set is consumed by exactly one leg; a
            # generation's set is the durable record and stays.
            entry_restore["staged_set_removed"] = stage.consume(source)
            phase("consume_staged_set")
            if pending[name] and maspt_minutes > 0:
                # The background's surface before the analysis goes in:
                # the insertion's own jump is measured against it
                # (gpuwm.da.maspt; reads only).
                from gpuwm.da.maspt import MasptRecorder
                maspt = MasptRecorder.for_config(
                    cfg, start_seconds=float(t_start),
                    horizon_seconds=60.0 * maspt_minutes)
                maspt.before_insertion(state)
            if pending[name]:
                if child_in_checkpoint:
                    for field in sorted(pending[name]):
                        live = getattr(state, field, None)
                        if live is None or getattr(
                                child_node.state, field, None) is None:
                            continue
                        restored_background[field] = to_host(live)
                to_apply, pair_report = keep_moment_pairs(
                    state, pending[name], mp_physics=cfg.mp_physics)
                phase("keep_moment_pairs")
                entry_insert = leg_record["trajectories"].setdefault(
                    str(name), {})
                if iau_mode in ("3d", "4d") or (
                        iau_mode == "oneshot" and hydrostatic):
                    inserted = insert_analysis(
                        state, to_apply, mode=iau_mode,
                        window_seconds=iau_window, t_start=float(t_start),
                        hydrostatic=hydrostatic, cfg=cfg,
                        iau_fields=getattr(args, "iau_fields", "dynamics"),
                        apply_increments=apply_increments,
                        refresh=refresh_diagnostics)
                    iau_forcing = inserted["forcing"]
                    iau_spread = inserted["spread"]
                    entry_insert.update(inserted["record"])
                elif iau_window is not None:
                    if nested_leg:
                        # The child's carried correction is the parent's
                        # analysed state minus its restored one, both read
                        # at the leg start; under IAU the parent is not
                        # analysed yet there, so the child would get none.
                        raise RuntimeError(
                            f"leg {leg} {name}: --iau-window-seconds with a "
                            "nest this leg would carry no analysis into the "
                            "child (its correction is read at the leg start, "
                            "before IAU has added anything); run IAU on "
                            "root-only legs")
                    from gpuwm.da.iau import IncrementalUpdate, split_fields
                    spread, at_once = split_fields(
                        to_apply, getattr(args, "iau_fields", "dynamics"))
                    if at_once:
                        # Hydrometeors (and their moments) go in at the
                        # leg start as before: spread a fraction at a time
                        # the microphysics removes the rain the radar did
                        # not see between fractions, and the remaining
                        # negative fractions clip at zero (the box E smoke
                        # clipped about 50,000 qr cells per step).
                        receipt = apply_increments(
                            state, at_once, mp_physics=cfg.mp_physics)
                        refresh_diagnostics(
                            state, hypsometric_opt=cfg.hypsometric_opt)
                    if hydrostatic:
                        # The php change spreads with the dynamics.
                        from gpuwm.da.hydrostatic import \
                            geopotential_increment
                        refresh_diagnostics(
                            state, hypsometric_opt=cfg.hypsometric_opt)
                        dphp, hydro_record = geopotential_increment(
                            state, spread,
                            hypsometric_opt=int(cfg.hypsometric_opt))
                        if dphp is not None:
                            spread = dict(spread, php=dphp)
                        leg_record["trajectories"].setdefault(
                            str(name), {})["hydrostatic_spread"] = (
                                hydro_record)
                    iau = IncrementalUpdate(
                        spread, start_seconds=float(t_start),
                        window_seconds=float(iau_window),
                        hypsometric_opt=int(cfg.hypsometric_opt))
                    leg_record["trajectories"].setdefault(str(name), {})[
                        "apply_fields"] = len(to_apply)
                    leg_record["trajectories"][str(name)][
                        "iau_split"] = {"spread": sorted(spread),
                                        "at_leg_start": sorted(at_once)}
                else:
                    receipt = apply_increments(
                        state, to_apply, mp_physics=cfg.mp_physics)
                    phase("apply_increments")
                    leg_record["trajectories"].setdefault(str(name), {})[
                        "apply_fields"] = receipt["field_count"]
                if pair_report["conditioned"]:
                    leg_record["trajectories"].setdefault(str(name), {})[
                        "moment_pairs_kept"] = pair_report
                if iau is None and iau_forcing is None:
                    refresh_diagnostics(
                        state, hypsometric_opt=cfg.hypsometric_opt)
                    phase("refresh_diagnostics")
        phase("restore_and_apply_increments")
        health = StateHealthValidator(state).validate(
            phase=f"leg{leg}.{name}")
        if not health.ok:
            raise FloatingPointError(
                f"leg {leg} {name}: pre-leg health failed: "
                f"{vars(health)}")

        # -- the fine nest ------------------------------------------------
        #
        # A child born this leg is built AFTER the restore, the
        # increment application and the health gate: the parent state
        # it is derived from has to be the ANALYSED one, or the nest
        # would inherit a forecast nobody corrected and the whole
        # exercise would be worth less than interpolating the output.
        if nested_leg:
            if child_node is None:
                child_node, child_driver, _land_receipt = _build_child()
                nest_birth[name] = float(born_at)
                nest_entry["initialization"] = "parent-live-state-sint"
                if iau_spread:
                    # Born from the live parent, which the forcing has not
                    # moved yet: the child takes the spread part of the
                    # analysis as its own forcing over the same window.
                    child_forcing = carry_iau_to_child(
                        state, child_node.state, iau_spread, iau_forcing,
                        child_dc_leg=child_dc_leg, cfg=cfg,
                        hydrostatic=hydrostatic, t_start=float(t_start),
                        positivity_policy=args.positivity_policy,
                        nest_entry=nest_entry)
            else:
                # A later nested leg: the child already has fine
                # structure of its own, restored above from the
                # trajectory's own checkpoint set; flattening it back
                # to a parent interpolation every leg boundary would
                # throw away exactly what the nest is for.
                nest_entry["initialization"] = "restart-set"
                # The child is not analysed itself -- the filter runs
                # on the parent ensemble, and a 1 km member set is a
                # different (and much larger) experiment.  What the
                # child gets is the correction the analysis made to
                # its parent, carried down by the operator it was
                # born through: SINT(analysed) - SINT(background),
                # differenced rather than interpolated as one
                # increment because SINT is monotonicity-limited and
                # therefore not linear.  Both sides are raw
                # interpolations of the PARENT -- the restored parent
                # before the increment, and the live parent after it.
                # Over exactly the fields the increment names: a
                # moment the applier repaired on a field the analysis
                # did not name stays on the parent, and a leg with no
                # analysis leaves the child bitwise alone.
                child_state = child_node.state
                analysed_parent = {
                    field: getattr(state, field)
                    for field in sorted(restored_background)}
                carried = {}
                if restored_background:
                    correction = nested_forecast.nest_down_analysis(
                        restored_background, analysed_parent,
                        child_dc_leg, cfg, array_module=cp)
                    # The parent's analysis was bounded against the
                    # PARENT's background; the child's background is
                    # its own evolved state, so the same correction
                    # can drive a positive-definite species below
                    # zero there.  Same policy, child's background.
                    correction, child_positivity = (
                        bound_child_correction(
                            child_state, correction,
                            policy=args.positivity_policy,
                            array_module=cp))
                    if child_positivity is not None:
                        nest_entry["positivity_on_correction"] = (
                            child_positivity)
                    for field, delta in correction.items():
                        largest = float(cp.abs(delta).max())
                        if largest == 0.0:
                            continue
                        target = getattr(child_state, field)
                        target[...] = target + delta.astype(
                            target.dtype)
                        carried[field] = largest
                    del correction
                    refresh_diagnostics(
                        child_state,
                        hypsometric_opt=child_dc_leg.run.hypsometric_opt)
                del analysed_parent
                if iau_spread:
                    child_forcing = carry_iau_to_child(
                        state, child_state, iau_spread, iau_forcing,
                        child_dc_leg=child_dc_leg, cfg=cfg,
                        hydrostatic=hydrostatic, t_start=float(t_start),
                        positivity_policy=args.positivity_policy,
                        nest_entry=nest_entry)
                nest_entry["analysis_carried_down"] = {
                    "how": ("sint(analysed parent) - sint(restored "
                            "parent), the child's own state kept"),
                    "fields": sorted(carried),
                    "max_abs_correction": {
                        field: carried[field]
                        for field in sorted(carried)},
                }
            child_state = child_node.state
            child_health = StateHealthValidator(child_state).validate(
                phase=f"leg{leg}.{name}.nest")
            if not child_health.ok:
                raise FloatingPointError(
                    f"leg {leg} {name}: nest pre-leg health failed: "
                    f"{vars(child_health)}")
        del restored_background

        model = assemble(wired, name=name, child_node=child_node)
        if resumed:
            model._resumed = True
            model._resume_committed_history_grid_ids = frozenset(
                model.nodes_by_grid_id)

        # An exact issuance endpoint is needed before differencing
        # accumulations. The first free leg starts after the final
        # pending analysis has been applied, so its reflectivity is
        # the issued analysis rather than the pre-analysis forecast.
        # Every leg that starts from an analysis also writes its start
        # composite (and, below, its first native frame): the analysed
        # members' echo per analysis, read against the leg-end composite
        # of the leg before (the prior).  The 9 km CONUS cycle showed
        # 4.8x MRMS's echo area two minutes after an analysis and 1.4x
        # without DA; these frames say whether the analysis put it there.
        # A 4D-IAU rewound leg (lane 7) is the same leg started again,
        # not a new start, so it writes no second start composite.
        analysed_start = bool(source is not None and pending[name])
        if args.save_composites and not context.iau_rewound and (
                leg == 0 or leg == len(args.obs)
                or analysed_start
                or (getattr(args, "rain_history", False)
                    and t_start == args.rain_forecast_start_seconds)):
            start_name = (out / "composites" /
                          f"start{leg_number(leg):02d}_{name}.npz")
            _write_rain_composite(
                start_name, state=state, driver=driver,
                grid=inputs.grid, cfg=cfg, elapsed_seconds=t_start,
                exp=exp, label=f"start leg {leg_number(leg)} member {name}")
            if child_node is not None:
                _write_rain_composite(
                    start_name.with_name(
                        start_name.stem + f"_d{nest_child_dc.grid_id:02d}.npz"),
                    state=child_node.state, driver=child_driver,
                    grid=child_node.grid, cfg=child_dc_leg.run,
                    elapsed_seconds=t_start, exp=exp,
                    label=f"start leg {leg_number(leg)} member {name} nest",
                    domain=child_dc_leg)

        # -- radar latent heating for this leg --------------------------
        # External data, like the boundary data: attached for the
        # integration and detached after it, so the next leg attaches its
        # own windows on a fresh clock.  (The attribute is restart INFRA,
        # so a manifest taken while it is attached skips it.)  Members
        # only; the control stays the no-DA arm.
        tten_forcing = None
        if tten_reflectivity is not None and (
                name != CONTROL or getattr(args, "radar_tten_control", False)):
            tten_forcing = _leg_radar_forcing(
                radar_tten, args, state, name, tten_reflectivity, obs_path,
                leg_start_valid=exp.start_time + timedelta(
                    seconds=float(t_start)),
                leg_minutes=leg_length(leg) / 60.0,
                grid_identity=(window_grid_identity(args, leg)
                               if getattr(args, "radar_tten_windows", None)
                               else None))
            radar_tten.attach(state, tten_forcing, cfg)
        history_drivers = {1: driver}
        if child_node is not None:
            history_drivers[int(child_node.cfg.grid_id)] = child_driver
        history_writer = HistoryWriter()
        native_frames = {} if args.save_composites else None
        history = rain_history_handler(
            context=context, name=name, drivers=history_drivers,
            writer=history_writer, native=native_frames,
            first_frame=(f"first{leg_number(leg):02d}_{name}.npz"
                         if args.save_composites and analysed_start
                         else None))
        retained = None
        if vts_enabled:
            from gpuwm.da import moments
            from gpuwm.da.spread_vts import ForecastRetention
            retained = ForecastRetention(
                socket_path=args.spread_vts_socket, member=name,
                analysis_seconds=t_end, origin_seconds=t_start,
                grid_identity_sha256=args.spread_vts_grid_identity,
                fields=moments.analysis_fields(int(cfg.mp_physics)),
                surface_fields=(("t2", "u10", "v10", "q2", "psfc")
                                if context.surface_enabled else ()))
            history = retained.history(history, driver=driver, cfg=cfg)
            if node.clock.elapsed_seconds == t_end-900:
                retained.capture(node, driver, cfg, offset_seconds=-900)
        steppers = None
        root_step = None
        if iau is not None:
            from gpuwm.core.dycore import step as dycore_step
            root_step = iau.stepper(dycore_step)
        if maspt_minutes > 0:
            # Reads the state only (gpuwm.da.maspt): every leg logs its
            # start's surface pressure tendency, the balance measure the
            # promotion gate compares against the no-DA start.
            from gpuwm.core.dycore import step as dycore_step
            from gpuwm.da.maspt import MasptRecorder
            if maspt is None:
                maspt = MasptRecorder.for_config(
                    cfg, start_seconds=float(t_start),
                    horizon_seconds=60.0 * maspt_minutes)
            root_step = maspt.stepper(root_step or dycore_step)
        if root_step is not None:
            steppers = {int(node.cfg.grid_id): root_step}
        if iau_forcing is not None:
            attach_iau(state, iau_forcing, t_start=float(t_start))
        if child_forcing is not None:
            attach_iau(child_node.state, child_forcing,
                       t_start=float(t_start))
        try:
            phase("health_nest_composite_and_forcing")
            execute_experiment(model, history_handler=history,
                               progress_callback=None,
                               validate_state=True,
                               skip_feedback_path=True,
                               steppers=steppers)
        except BaseException:
            history_writer.abandon()
            raise
        finally:
            if tten_forcing is not None:
                radar_tten.detach(state)
            if iau_forcing is not None:
                from gpuwm.da.iau import detach as detach_iau
                detach_iau(state)
            if child_forcing is not None:
                from gpuwm.da.iau import detach as detach_iau
                detach_iau(child_node.state)
        entry_run = leg_record["trajectories"].setdefault(str(name), {})
        if iau_forcing is not None:
            entry_run["iau_forcing"] = iau_forcing.receipt()
            unfinished = 1.0 - iau_forcing.applied_fraction
            if unfinished > 1e-9:
                # The window reaches past this leg's end: the rest of the
                # increment was never added.  The restart carries no
                # forcing, so it is lost, and the record must say so.
                entry_run["iau_forcing"]["unapplied_fraction"] = unfinished
                print(f"leg {leg} {name}: IAU window longer than the leg; "
                      f"{unfinished:.3f} of the increment was not added",
                      flush=True)
            del iau_forcing
        if child_forcing is not None:
            entry_run.setdefault("nest", {})["iau_forcing"] = (
                child_forcing.receipt())
            del child_forcing
        if maspt is not None:
            entry_run["maspt"] = maspt.receipt()
            del maspt
        # Every frame of the leg is on disk before the leg goes on.
        history_writer.close()
        phase("integrate")
        if iau is not None:
            iau_record = iau.finish(state)
            leg_record["trajectories"].setdefault(str(name), {})[
                "iau"] = iau_record
            if iau_record["remainder_at_end"] > 1e-9:
                # Only a leg shorter than the window ends with the
                # increment unfinished; the remainder went in at the leg
                # end, which the receipt states.
                print(f"leg {leg} {name}: IAU window longer than the leg; "
                      f"{iau_record['remainder_at_end']:.3f} of the "
                      "increment was added at the leg end", flush=True)
            del iau
        cp.cuda.Stream.null.synchronize()
        if tten_forcing is not None:
            tten_record = tten_forcing.receipt()
            leg_record["trajectories"].setdefault(str(name), {})[
                "radar_tten"] = tten_record
            if not (sum(tten_record["calls_by_slot"])
                    + tten_record["calls_skipped_no_mp_heating"]):
                raise RuntimeError(
                    f"leg {leg} {name}: the radar heating was attached "
                    "for the leg and no microphysics call read it, so "
                    "the member ran unforced while this record says "
                    "forced")
            del tten_forcing

        # -- the leg join: this trajectory's restart set --------------
        #
        # Written FIRST, before any diagnostic reads the state, so the
        # set is the model exactly as the integration left it.  The
        # writer refuses anything but a period boundary with nothing
        # pending, which is what a completed integration is.
        restart_root = stage.directory(leg_number(leg), name)
        restarts[name] = write_leg_restart(
            model, restart_root,
            auto_epssm=(nested_forecast.nested_experiment(
                exp, nest_child_dc).auto_epssm
                if len(model.nodes_by_grid_id) > 1 else exp.auto_epssm),
            valid_time=exp.start_time + timedelta(seconds=float(
                node.clock.ticks / node.clock.tick_den)))
        written_members = tree_restart_members(restarts[name])
        entry = leg_record["trajectories"].setdefault(str(name), {})
        entry["pool_trim"] = model._pool_trim_policy
        entry["restart"] = {
            "root_member": restarts[name].name,
            "domain_ids": sorted(int(gid) for gid in written_members),
            "bytes": int(sum(member.stat().st_size
                             for member in written_members.values())),
            "elapsed_seconds": float(node.clock.elapsed_seconds),
        }

        phase("write_restart")
        if retained is not None:
            from gpuwm.da.spread_vts import continue_and_restore
            retained.capture(node, driver, cfg, offset_seconds=0)
            restoration = continue_and_restore(
                model, restart=restarts[name], experiment=exp,
                boundary_interval_seconds=inputs.boundary_interval_seconds,
                analysis_seconds=t_end,
                child_domain=(child_dc_leg if child_node is not None else None),
                capture=lambda: retained.capture(node, driver, cfg,
                                                   offset_seconds=900))
            entry["spread_vts"] = retained.receipt()
            entry["spread_vts"]["central_restore"] = restoration
            phase("vts_future_and_central_restore")
        thb_live = getattr(state, "thb", None)
        thb_snapshot = to_host(thb_live) if thb_live is not None \
            else None

        # -- leg-end diagnostics on the live device state ---------------
        refl = obsop.simulated_reflectivity(state, cfg)
        refl_host = to_host(refl).astype(np.float32)
        if name != CONTROL:
            member_dbz[int(name)] = refl_host.astype(np.float64)
        if surface_cfg is not None and name != CONTROL:
            # Leg-END surface diagnostics off the live driver: the
            # 2m/10m fields the surface layer diagnosed for the very
            # state that was mirrored, taken here beside member_dbz
            # and nowhere else.
            member_sfc[int(name)] = capture_surface_diagnostics(
                driver, dewpoint=(
                    getattr(args, "sfc_td_sigma_k", None) is not None
                    or bool(getattr(args, "obs_table", None))))
        retained_history = (getattr(args, "rain_history", False)
            and t_end > args.rain_forecast_start_seconds)
        comp_dir = out / "composites"

        def _leg_end_colmax(gid, clock, operator_colmax):
            # The leg-end frame carries the model's own REFL_10CM from the
            # output-due microphysics call at this very tick, the product
            # the free-forecast frames and WOOF's maps carry.  The DA
            # forward operator (``refl`` above, still what the analysis
            # reads) stands in only when no alarm rang at the leg end.
            # Mixing the two put the CONUS first-light DA members on the
            # operator and a proof run on native, 1,739 against 1,368
            # strong-echo cells on one identical control state.
            got = (native_frames or {}).get(int(gid))
            if got is not None and got[0] == int(clock.ticks):
                return got[1], REFL_PRODUCT_NATIVE
            return operator_colmax, REFL_PRODUCT_OPERATOR

        if args.save_composites and not retained_history:
            comp_dir = out / "composites"
            comp_dir.mkdir(parents=True, exist_ok=True)
            composite_npz = (comp_dir /
                             f"leg{leg_number(leg):02d}_{name}.npz")
            rain = rain_snapshot(driver)
            leg_colmax, leg_product = _leg_end_colmax(
                1, node.clock, refl_host.max(axis=0))
            np.savez_compressed(
                composite_npz,
                refl_colmax=leg_colmax,
                **rain, rain_reset_id=np.int64(0),
                elapsed_seconds=np.float64(
                    node.clock.elapsed_seconds),
                refl_product=np.str_(leg_product))
            # ...and the same composite as a real wrfout beside it, so
            # `gpuwm render --engine rust` -- the renderer the render
            # law names -- can draw this member.  The npz is what the
            # cycle's own analysis reads and is untouched; this is a
            # second copy in the format a different tool reads.
            _write_composite_wrfout(
                composite_npz, leg_colmax, inputs.grid, cfg,
                node.clock.elapsed_seconds, exp,
                label=f"leg {leg_number(leg):02d} member {name}", rain=rain)
        # -- the nest's own leg-end product ------------------------------
        #
        # Written under a d02 name beside the parent's rather than
        # replacing it: the point of the exercise is a fine view OF the
        # parent's forecast, and a reader has to be able to see both.
        refl_nest = None
        if child_node is not None:
            refl_nest = obsop.simulated_reflectivity(
                child_node.state, child_dc_leg.run)
            refl_nest_host = to_host(refl_nest).astype(np.float32)
            nest_entry["refl_max_dbz"] = float(refl_nest_host.max())
            nest_entry["elapsed_seconds"] = float(
                child_node.clock.elapsed_seconds)
            nest_entry["step_count"] = int(child_node.clock.step_count)
            nest_entry["domain_start_offset_seconds"] = float(
                child_node.state.domain_start_offset)
            if args.save_composites and not retained_history:
                # ``leg_number(leg)``, the same author as every other
                # leg-named artifact this driver writes.  The child's
                # frame and the parent's frame of the SAME leg have to
                # carry the same number, or a run given a leg-number
                # offset writes a nest under a leg the parent frames
                # never use and nothing finds it.
                nest_npz = (comp_dir /
                            f"leg{leg_number(leg):02d}_{name}_d"
                            f"{nest_child_dc.grid_id:02d}.npz")
                nest_colmax, nest_product = _leg_end_colmax(
                    nest_child_dc.grid_id, child_node.clock,
                    refl_nest_host.max(axis=0))
                nest_rain = rain_snapshot(child_driver)
                np.savez_compressed(
                    nest_npz,
                    refl_colmax=nest_colmax,
                    refl_product=np.str_(nest_product),
                    **nest_rain, rain_reset_id=np.int64(0),
                    elapsed_seconds=np.float64(
                        child_node.clock.elapsed_seconds),
                    dx_m=np.float64(nest_child_dc.run.dx),
                    i_parent_start=np.int32(
                        nest_child_dc.i_parent_start),
                    j_parent_start=np.int32(
                        nest_child_dc.j_parent_start),
                    parent_grid_ratio=np.int32(
                        nest_child_dc.parent_grid_ratio))
                # ...and the child's own wrfout beside it, for the
                # same reason the parent gets one and by the same
                # call.  Without it the nest's product had no route
                # to the renderer the render law names: an ``.npz``
                # states no geolocation, so rw_wrfbatch cannot read
                # one, and a door that can ask for a nest whose
                # picture nobody can draw is not a shipped door.
                _write_composite_wrfout(
                    nest_npz, nest_colmax, child_node.grid,
                    nest_child_dc.run,
                    child_node.clock.elapsed_seconds, exp,
                    label=(f"leg {leg_number(leg):02d} member {name} "
                           f"d{nest_child_dc.grid_id:02d}"),
                    domain=nest_child_dc, rain=nest_rain)
        phase("leg_end_diagnostics_and_composites")
        entry["wall_seconds"] = round(time.time() - t_leg, 1)
        entry["elapsed_seconds"] = float(node.clock.elapsed_seconds)
        if document is not None:
            z_mask = np.asarray(
                document["variables"]["z_mask"]).astype(bool)
            z_obs = np.asarray(document["variables"]["z_obs"],
                               np.float64)
            inside = z_mask
            model_in_mask = refl_host[inside].astype(np.float64)
            entry["z_obs_space"] = {
                "points": int(inside.sum()),
                "obs_mean_dbz": float(z_obs[inside].mean()),
                "model_mean_dbz": float(model_in_mask.mean()),
                "model_max_dbz": float(model_in_mask.max()),
                "model_cols_gt35_in_echo": int(
                    (refl_host.max(axis=0) >= 35.0)[
                        z_mask.any(axis=0)].sum()),
                "obs_cols_gt35": int(
                    ((z_obs * z_mask).max(axis=0) >= 35.0).sum()),
                "innovation_mean_dbz": float(
                    (z_obs[inside] - model_in_mask).mean()),
            }
        if (analysis_due and name != CONTROL
                and not args.no_hotstart):
            increments_hot, hot_prov = hotstart_increments(
                state, z_obs_cp, z_mask_cp, hot_cfg,
                simulated_dbz=refl)
            hot_pending[name] = {
                field: to_host(values).astype(np.float32)
                for field, values in increments_hot.items()}
            entry["hotstart"] = {
                key: hot_prov[key] for key in hot_prov
                if isinstance(hot_prov[key], (int, float, str))}

        # The host mirror the FILTER reads.  Not the leg join: that
        # is the restart set above, which carries this and everything
        # the mirror does not.
        snapshot = {}
        for field in STATE_SERIALIZED_ATTRS:
            value = getattr(state, field, None)
            if value is not None:
                snapshot[field] = to_host(value)
        snapshots[name] = snapshot
        entry["analysis_state_fields"] = sorted(snapshot)
        if thb_host is None and thb_snapshot is not None:
            thb_host = thb_snapshot
        pending[name] = None
        phase("hotstart_and_mirror")
        entry["phase_seconds"] = dict(phases)
        print(f"leg {leg} {name}: {entry['wall_seconds']} s, "
              f"elapsed {entry['elapsed_seconds']:.0f} s", flush=True)

    if wire_override is not None:
        wire = wire_override
    if assemble_override is not None:
        assemble = assemble_override
    run_trajectory(name)
    record = leg_record["trajectories"][str(name)]
    if stage.consume_intent is not None:
        record["staged_set_consume_pending"] = True
    return MemberLegResult(
        restart=restarts[name], snapshot=snapshots[name],
        H_Z=member_dbz.get(int(name)) if name != CONTROL else None,
        H_surface=member_sfc.get(int(name)) if name != CONTROL else None,
        hot_pending=hot_pending.get(name), setup_arrays=setup_arrays,
        thb_host=thb_host, nest_birth=nest_birth.get(name), record=record,
        consume_restart=stage.consume_intent,
        pending_consumed=context.pending is not None,
    )


# Legacy pure dispatch and clock references stay importable for existing tests.
# Neither one is used to place clocks or join a new complete-tree member leg.
def jump_clock(clock, start_seconds: float, dt: float) -> None:
    """Place a freshly built clock at a leg boundary.

    A leg after the first restores a host snapshot into a state the
    prepared cache just rebuilt, so its clock starts at zero while the
    trajectory it carries is already ``start_seconds`` old.  Three things
    have to move together.

    ``ticks`` and ``step_count`` are the integer calendar and are exact:
    the tick lattice is what every alarm evaluates on, so setting seconds
    would be setting a derived quantity.

    ``dtbc_fp32`` is WRF's REAL boundary-tendency accumulator, and it is
    NOT derivable in closed form.  It recurs as
    ``fl32(dtbc + dt)`` once per step and resets at every external-LBC
    seam (``gpuwm.core.clock.DomainClock.prepare_step`` /
    :meth:`mark_force`), so its value carries the accumulated FP32
    rounding of every step since the last seam.  ``steps_since_seam * dt``
    is a different number in the last bits, and the boundary relaxation
    consumes this one.  So it is REPLAYED with the same recurrence rather
    than computed -- at most one boundary interval of iterations, and
    bit-exact against a clock that stepped there.
    """
    steps = int(round(start_seconds / dt))
    clock.ticks = steps * clock.spec.step_ticks
    clock.step_count = steps
    interval = clock.spec.lbc_interval_ticks
    since = steps
    if interval is not None:
        per_seam = interval // clock.spec.step_ticks
        since = steps % per_seam
        # A leg boundary that lands exactly ON a seam has a FULL interval
        # accumulated, not zero.  The reset is top-of-step work
        # (``lbc_reset_due`` -> ``mark_force``), so the integrator applies
        # it to this clock on its own first step; zeroing it here would
        # hand the integrator a state it never produces, and the two only
        # happen to agree because the reset lands next.  Reproducing what
        # the integrator would have HAD is the rule -- a clock this worker
        # places must be indistinguishable from one that stepped there,
        # and ``tests/test_da_member_leg_clock.py`` compares them bit for
        # bit at exactly this instant.
        if since == 0 and steps > 0:
            since = per_seam
    value = np.float32(0.0)
    for _ in range(since):
        value = np.float32(value + clock.spec.dt_fp32)
    clock.dtbc_fp32 = value

class MemberPool:
    """``width`` persistent worker processes, each running whole legs.

    Persistent because a worker's CUDA context, its NVRTC-compiled
    kernel modules and its prepared-authority preflight are each worth
    about a second, and paying that per member-leg would eat the whole
    margin concurrency buys.

    Completion order is deliberately NOT propagated anywhere: results
    are collected into a dict keyed by trajectory name, and every
    downstream consumer -- the LETKF above all, which reads its
    backgrounds in ``sorted()`` index order -- sees exactly the
    ordering the serial driver produced.
    """

    def __init__(self, argv, width: int, workdir: Path):
        raise RuntimeError(
            "the retired member pool carries atmosphere-only snapshots and loses "
            "the land/physics state and complete restart clocks; use the complete-tree "
            "packed member-leg controller rather than its old stdin task protocol")

    @staticmethod
    def _stderr_tail(proc, limit: int = 4000) -> str:
        """Whatever the worker managed to say before it stopped.

        Read non-blockingly where possible: a worker that is merely slow
        must not turn a diagnostic into a hang.
        """
        if proc.poll() is None:
            return "(worker still running; no stderr collected)"
        try:
            return (proc.stderr.read() or "")[-limit:]
        except (OSError, ValueError):
            return "(stderr unavailable)"

    def run_leg(self, tasks: list) -> dict:
        """Run every task in ``tasks``, return ``{name: payload}``.

        A worker is refilled as soon as its result is taken, so with
        more trajectories than workers the pool stays busy instead of
        draining to empty between batches.  The parent blocks on the
        LOWEST-INDEXED busy worker rather than on whichever finishes
        first: with homogeneous members that costs nothing measurable,
        and it keeps the parent's control flow independent of worker
        timing, which is one less thing for the identity claim to rest
        on.
        """
        pending = list(tasks)
        inflight: dict = {}
        results: dict = {}
        free = list(range(self.width))
        while pending or inflight:
            while pending and free:
                index = free.pop(0)
                task = pending.pop(0)
                proc = self.procs[index]
                try:
                    proc.stdin.write(json.dumps(task) + "\n")
                    proc.stdin.flush()
                except (OSError, ValueError) as error:
                    # A worker that died between legs fails here, on the
                    # write, rather than on the read.  Report which
                    # trajectory was being handed over and what the
                    # worker last said; a bare EPIPE names neither.
                    raise RuntimeError(
                        f"member worker {index} is gone (exit "
                        f"{proc.poll()}) and cannot take trajectory "
                        f"{task.get('name')!r}: {error}\n"
                        f"{self._stderr_tail(proc)}") from error
                inflight[index] = task
            if not inflight:
                break
            index = min(inflight)
            proc = self.procs[index]
            line = proc.stdout.readline()
            if not line:
                # The same defect class as the write arm above, on the
                # platform where a write to a dead child's pipe still
                # buffers: the death then surfaces HERE, on the read, and
                # this arm must name the trajectory just as that one does.
                raise RuntimeError(
                    f"member worker {index} died running trajectory "
                    f"{inflight[index].get('name')!r} "
                    f"({inflight[index]!r}, exit {proc.poll()}): "
                    f"{self._stderr_tail(proc)}")
            payload = json.loads(line)
            if not payload.get("ok"):
                raise RuntimeError(
                    f"member worker {index} failed on trajectory "
                    f"{payload.get('name')}: {payload.get('error')}\n"
                    f"{payload.get('traceback')}")
            results[payload["name"]] = payload
            del inflight[index]
            free.append(index)
        return results

    def close(self) -> None:
        for proc in self.procs:
            try:
                proc.stdin.write(json.dumps({"stop": True}) + "\n")
                proc.stdin.flush()
                proc.stdin.close()
            except (OSError, ValueError):
                pass
        for proc in self.procs:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                # Deliberately not killed: a worker still holding the
                # device is reported, never terminated.
                print(f"member worker {proc.pid} did not exit within 30 s "
                      f"of being told to stop; leaving it alone",
                      flush=True)
