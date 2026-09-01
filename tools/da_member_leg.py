"""One member's forecast leg, and the process pool that runs several.

The cycling driver advances every trajectory serially: eleven fresh
models built and torn down per leg for a ten-member run.  This module
holds the body of that loop so it can be called either in the driver's
own process (the default, unchanged) or in a worker process (opt-in via
``--member-workers``).  Both paths call :func:`run_member_leg`, so the
concurrent arm is not a reimplementation that has to be argued equal to
the serial one -- it is the same function.

**Why processes and not streams.**  Measured on the live-fire-3 shape
(132x132x49, dt=15, 60 steps a leg, ``tools/da_probe_*.py``): a leg's
wall time, its CUDA event span and its cProfile total are all 5.02 s.
The leg is host-dispatch bound with an idle device -- Dudhia shortwave
alone is 1.94 s of a 49-level Python loop issuing thousands of tiny
CuPy operations.  Threads on non-blocking streams therefore measured
0.97x at width 2 and 0.83x at width 4: the GIL serializes the dispatch
and contention makes it worse.  Separate processes have no GIL and do
overlap, but saturate early -- 0.287, 0.400, 0.414, 0.419, 0.414 legs/s
at width 1, 2, 3, 4, 6 on a 32-core box, so the limit is WDDM
time-slicing between CUDA contexts rather than the host.  The ceiling is
~1.45x and width 2 already buys 1.40x of it.  None of that is a reason
to fold a member axis into the kernels: the device is not the busy part.

**Why this is bit-identical.**  A member's leg reads only its own state.
Perturbations come from a host Philox keyed by a SHA-256 of the field
and seed (``gpuwm.da.perturb``), so they do not depend on evaluation
order; there is no device RNG and no float atomic anywhere in the step
path; and the analysis reads members back from disk in ``sorted()``
index order, so completion order cannot reach it.  A worker process
wires the model exactly as the serial path does and runs the same
kernels on the same inputs.  What changes between the two arms is
allocator history and process identity, neither of which any computed
value depends on -- every device buffer in the step path is
zero-allocated.  ``tests/test_da_member_leg_identity.py`` is the proof
rather than the argument.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np

#: Name of the unassimilated trajectory in every report.
CONTROL = "control"

#: The physics-receipt fields that are registry VOCABULARY rather than
#: physics; see the driver's module docstring.
PHYSICS_VOCABULARY_FIELDS = ("maturity", "registry_sha256")


def to_host(value) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    return np.ascontiguousarray(np.asarray(value))


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
        # the integrator would have HAD is the rule -- a clock this driver
        # places must be indistinguishable from one that stepped there,
        # and ``tests/test_da_cycle_prepared.py`` compares them bit for
        # bit at exactly this instant.
        if since == 0 and steps > 0:
            since = per_seam
    value = np.float32(0.0)
    for _ in range(since):
        value = np.float32(value + clock.spec.dt_fp32)
    clock.dtbc_fp32 = value


# ---------------------------------------------------------------------------
# the per-process context
# ---------------------------------------------------------------------------

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
    })

    return MemberContext(
        args=args, inputs=inputs, exp=exp, cfg=cfg, dt=float(cfg.dt),
        cfg_perturb=cfg_perturb, hot_cfg=HotStartConfig(),
        perturbation_report=perturbation_report,
        vocabulary_divergence=vocabulary_divergence)


def wire(ctx: MemberContext, run_seconds_total: float):
    """Build a complete fresh model for one member-leg.

    ``model._scratch_arena`` and ``model._dycore_state_workspace`` are
    set to None here, and that is what makes member isolation true: with
    them None ``execute_experiment``'s arena path is inert and
    ``domain_turn`` degrades to a nullcontext, so every buffer a member
    touches hangs off its own ``DomainState``.  A shared arena hands
    every domain a prefix view of ONE backing allocation, so two members
    drawing from one would write the same bytes with no exception
    raised.  :func:`run_member_leg` asserts both stay None.
    """
    from gpuwm.core.clock import build_schedule, resolve_clock
    from gpuwm.core.model import (DomainNode, ExperimentState,
                                  ModelRuntimeStatus)
    from gpuwm.ingest.hrrr_physics import initialize_prepared_physics
    from gpuwm.ingest.prepared_cache import restore_prepared_cache

    inputs, exp, cfg = ctx.inputs, ctx.exp, ctx.cfg
    exp_leg = dataclasses.replace(exp, run_seconds=float(run_seconds_total))
    restored = restore_prepared_cache(
        inputs.prepared_cache_path, expected_identity=inputs.cache_identity,
        cfg=cfg, static=inputs.static)
    driver = initialize_prepared_physics(
        restored.initial_result, cfg, restored.met, restored.surface,
        inputs.static, inputs.landuse_identity, inputs.grid,
        exp.start_time)
    tick = resolve_clock(
        exp_leg, lbc_interval_s=float(inputs.boundary_interval_seconds))
    schedule = build_schedule(exp_leg, tick)
    clocks = tick.clocks()
    node = DomainNode(exp.root, inputs.grid,
                      restored.initial_result.state, clocks[1],
                      None, [], None)
    model = ExperimentState(node, MappingProxyType({1: node}), schedule,
                            None, "da-cycle-shakedown")
    model._runtime_status = ModelRuntimeStatus()
    model._resumed = False
    model._resume_committed_history_grid_ids = frozenset()
    model._scratch_arena = None
    model._dycore_state_workspace = None
    model._io_manager = None
    model._last_checkpoint = None
    model._prepared_by_grid_id = MappingProxyType({
        1: SimpleNamespace(static_fields=inputs.static,
                           geog_selection=None,
                           initial_result=restored.initial_result)})
    return model, node, restored, driver


def teardown(*objects) -> None:
    import cupy as cp
    for obj in objects:
        del obj
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def leg_observations(ctx: MemberContext, leg: int):
    """``(document, grid)`` for a leg, read once per process per leg.

    Cached for the CURRENT leg only.  Every trajectory in a leg wants
    the same document, so re-reading it per member would be waste; but
    holding all twelve legs' documents would retain a few hundred MiB of
    z_obs/z_mask per process, times the pool width, for nothing -- a
    finished leg's observations are never read again.
    """
    if leg in ctx._obs_cache:
        return ctx._obs_cache[leg]
    from gpuwm.da.obs_radar import read_document
    from gpuwm.obs.target_grid import TargetGrid

    args = ctx.args
    if leg >= len(args.obs):
        entry = (None, None)
    else:
        obs_path = Path(args.obs[leg])
        if not obs_path.is_file():
            raise FileNotFoundError(
                f"leg {leg}: no observation file at {obs_path}")
        grid_h = TargetGrid.from_wrfout(Path(args.grid_wrfout[leg]))
        entry = (read_document(obs_path, expected_grid=grid_h), grid_h)
    ctx._obs_cache.clear()
    ctx._obs_cache[leg] = entry
    return entry


@dataclass
class MemberLegResult:
    entry: dict
    snapshot: dict
    refl_host: np.ndarray
    hot_increments: dict | None
    thb_snapshot: np.ndarray | None


def run_member_leg(ctx: MemberContext, *, leg: int, name, t_start: float,
                   t_end: float, analysis_due: bool,
                   snapshot_in: dict | None,
                   pending_in: dict | None) -> MemberLegResult:
    """Advance ONE trajectory across ONE leg.

    The body of the driver's member loop, unchanged in substance and
    called by both the serial and the concurrent path.  ``name`` is
    :data:`CONTROL` or an integer member index; ``snapshot_in`` is the
    host state this trajectory ended the previous leg with (None at leg
    0) and ``pending_in`` the increments waiting to be applied to it.
    """
    import cupy as cp

    from gpuwm.core.health import StateHealthValidator
    from gpuwm.core.model import execute_experiment
    from gpuwm.da import obsop, perturb
    from gpuwm.da.hotstart import hotstart_increments
    from gpuwm.ensemble.increments import apply_increments
    from gpuwm.ensemble.member import refresh_diagnostics
    from gpuwm.state_serialization_contract import STATE_SERIALIZED_ATTRS

    args, cfg = ctx.args, ctx.cfg
    document, _grid_h = leg_observations(ctx, leg)

    t_leg = time.time()
    model, node, restored, driver = wire(ctx, t_end)
    # The two lines that make member isolation true.  They are set in
    # wire(); a future edit that reintroduces a shared arena or the
    # shared dycore-state workspace would silently alias members'
    # buffers, so the concurrent path refuses to start rather than
    # produce quietly wrong numbers.
    if model._scratch_arena is not None:
        raise AssertionError(
            "a member-leg model must not carry a shared scratch arena: "
            "ScratchArena hands every domain a prefix view of ONE "
            "backing allocation, so two trajectories drawing from it "
            "would overwrite each other with no exception raised")
    if model._dycore_state_workspace is not None:
        raise AssertionError(
            "a member-leg model must not carry a shared dycore-state "
            "workspace")
    t_wired = time.time()

    state = node.state
    entry: dict = {}
    if leg == 0:
        if name != CONTROL:
            perturb.apply_perturbations(
                state, args.seed + int(name), ctx.cfg_perturb)
            refresh_diagnostics(state, hypsometric_opt=cfg.hypsometric_opt)
    else:
        jump_clock(node.clock, t_start, ctx.dt)
        model._resumed = True
        model._resume_committed_history_grid_ids = frozenset({1})
        for field, host in snapshot_in.items():
            getattr(state, field)[...] = cp.asarray(
                host, dtype=getattr(state, field).dtype)
        if pending_in:
            receipt = apply_increments(
                state, pending_in, mp_physics=cfg.mp_physics)
            entry["apply_fields"] = receipt["field_count"]
            refresh_diagnostics(state, hypsometric_opt=cfg.hypsometric_opt)
    health = StateHealthValidator(state).validate(phase=f"leg{leg}.{name}")
    if not health.ok:
        raise FloatingPointError(
            f"leg {leg} {name}: pre-leg health failed: {vars(health)}")

    t_pre = time.time()
    execute_experiment(model, history_handler=None,
                       progress_callback=None, validate_state=True,
                       skip_feedback_path=True,
                       pool_trim_per_period=args.pool_trim_per_period)
    # Per-member synchronisation.  The serial driver used the NULL
    # stream, which is a device-wide barrier; a worker process has its
    # own context and synchronises only itself.
    cp.cuda.Stream.null.synchronize()
    t_integrated = time.time()

    thb_live = getattr(state, "thb", None)
    thb_snapshot = to_host(thb_live) if thb_live is not None else None

    # -- leg-end diagnostics on the live device state ------------------
    refl = obsop.simulated_reflectivity(state, cfg)
    refl_host = to_host(refl).astype(np.float32)
    if args.save_composites:
        comp_dir = Path(args.out) / "composites"
        comp_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            comp_dir / f"leg{leg:02d}_{name}.npz",
            refl_colmax=refl_host.max(axis=0),
            elapsed_seconds=np.float64(node.clock.elapsed_seconds))
    entry["wall_seconds"] = round(time.time() - t_leg, 1)
    entry["phase_seconds"] = {
        "wire": round(t_wired - t_leg, 2),
        "pre_integrate": round(t_pre - t_wired, 2),
        "integrate": round(t_integrated - t_pre, 2),
    }
    entry["elapsed_seconds"] = float(node.clock.elapsed_seconds)
    if document is not None:
        z_mask = np.asarray(document["variables"]["z_mask"]).astype(bool)
        z_obs = np.asarray(document["variables"]["z_obs"], np.float64)
        inside = z_mask
        model_in_mask = refl_host[inside].astype(np.float64)
        entry["z_obs_space"] = {
            "points": int(inside.sum()),
            "obs_mean_dbz": float(z_obs[inside].mean()),
            "model_mean_dbz": float(model_in_mask.mean()),
            "model_max_dbz": float(model_in_mask.max()),
            "model_cols_gt35_in_echo": int(
                (refl_host.max(axis=0) >= 35.0)[z_mask.any(axis=0)].sum()),
            "obs_cols_gt35": int(
                ((z_obs * z_mask).max(axis=0) >= 35.0).sum()),
            "innovation_mean_dbz": float(
                (z_obs[inside] - model_in_mask).mean()),
        }
    hot_increments = None
    if analysis_due and name != CONTROL and not args.no_hotstart:
        z_obs_cp = cp.asarray(np.asarray(
            document["variables"]["z_obs"], np.float32))
        z_mask_cp = cp.asarray(np.asarray(
            document["variables"]["z_mask"]).astype(bool))
        increments_hot, hot_prov = hotstart_increments(
            state, z_obs_cp, z_mask_cp, ctx.hot_cfg, simulated_dbz=refl)
        hot_increments = {field: to_host(values).astype(np.float32)
                          for field, values in increments_hot.items()}
        entry["hotstart"] = {key: hot_prov[key] for key in hot_prov
                             if isinstance(hot_prov[key], (int, float, str))}

    snapshot = {}
    for field in STATE_SERIALIZED_ATTRS:
        value = getattr(state, field, None)
        if value is not None:
            snapshot[field] = to_host(value)
    entry["snapshot_fields"] = sorted(snapshot)
    entry["phase_seconds"]["post"] = round(time.time() - t_integrated, 2)

    teardown(model, node, restored, driver, state, refl)
    return MemberLegResult(entry=entry, snapshot=snapshot,
                           refl_host=refl_host,
                           hot_increments=hot_increments,
                           thb_snapshot=thb_snapshot)


# ---------------------------------------------------------------------------
# device budget
# ---------------------------------------------------------------------------

#: Device residency of one worker process, MEASURED on this shape rather
#: than modelled: 3151, 3154, 3175, 3236 MiB per process at widths 1, 2,
#: 3 and 6 (tools/da_probe_procs.py, RTX 5090 / Windows, 132x132x49).
#: The estimator's own itemisation for the same case is 432 MiB of CUDA
#: context + 2044 MiB of local-memory backing + ~650 MiB of pool, i.e.
#: ~3126 MiB, so the two agree to about 3%.  The measurement is what is
#: used; the estimate is what explains it.
MEASURED_PROCESS_MIB_MARGIN = 1.10


def worker_device_budget(ctx: MemberContext, width: int) -> dict:
    """Price ``width`` worker processes against the card, before spawning.

    The DA driver hand-builds its ``ExperimentState`` and so has never
    run a VRAM gate at all -- ``estimate_experiment`` is reached only
    through ``build_experiment``, which this route bypasses.  This is
    that gate, for the one thing concurrency actually changes: the
    number of CUDA contexts and local-memory backing stores resident at
    once.  Each is per PROCESS and does not shrink with width, which is
    why the refusal is worth having rather than letting a mid-leg
    out-of-memory crash report it.
    """
    import cupy as cp
    from gpuwm.core.preflight import (estimate_experiment,
                                      non_pool_device_bytes)

    mib = 1024.0 ** 2
    non_pool = non_pool_device_bytes(ctx.exp) / mib
    estimate = estimate_experiment(ctx.exp)
    pool = float(getattr(estimate, "alloc_estimate_bytes", 0)) / mib
    per_process = (non_pool + pool) * MEASURED_PROCESS_MIB_MARGIN
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    free_mib = free_bytes / mib
    required = per_process * width
    return {
        "width": width,
        "per_process_mib": round(per_process, 1),
        "non_pool_mib": round(non_pool, 1),
        "pool_estimate_mib": round(pool, 1),
        "required_mib": round(required, 1),
        "device_free_mib": round(free_mib, 1),
        "device_total_mib": round(total_bytes / mib, 1),
        "fits": required <= free_mib,
    }


def refuse_if_too_wide(ctx: MemberContext, width: int) -> dict:
    budget = worker_device_budget(ctx, width)
    if not budget["fits"]:
        affordable = max(1, int(budget["device_free_mib"]
                                // budget["per_process_mib"]))
        raise SystemExit(
            f"--member-workers {width} does not fit this card: each "
            f"worker process holds its own CUDA context and local-memory "
            f"backing store, priced at {budget['per_process_mib']:.0f} "
            f"MiB, so {width} of them need "
            f"{budget['required_mib']:.0f} MiB against "
            f"{budget['device_free_mib']:.0f} MiB free. "
            f"Use --member-workers {affordable} or fewer. (Concurrency "
            f"measured 1.40x at width 2 and saturates by width 4 on this "
            f"box, so a narrower pool costs very little throughput.)")
    return budget


# ---------------------------------------------------------------------------
# the worker process
# ---------------------------------------------------------------------------

def _save_snapshot(path: Path, snapshot: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{f"state/{k}": v for k, v in snapshot.items()})


def load_snapshot(path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key[len("state/"):]: np.asarray(data[key])
                for key in data.files if key.startswith("state/")}


def _worker_main(argv_file: Path) -> int:
    """Read tasks on stdin, run them, answer on stdout -- one line each.

    The process stays alive across legs so the CUDA context, the
    compiled kernel modules and the prepared-authority preflight are
    paid once rather than per member.
    """
    import traceback

    argv = json.loads(Path(argv_file).read_text(encoding="utf-8"))
    try:
        from tools.da_cycle_prepared import build_parser
    except ImportError:  # invoked with tools/ itself on sys.path
        from da_cycle_prepared import build_parser
    args = build_parser().parse_args(argv)
    ctx = build_member_context(args)
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        task = json.loads(line)
        if task.get("stop"):
            break
        try:
            name = task["name"]
            name = CONTROL if name == CONTROL else int(name)
            snapshot_in = (load_snapshot(task["snapshot_in"])
                           if task["snapshot_in"] else None)
            pending_in = None
            if task["pending_in"]:
                with np.load(task["pending_in"], allow_pickle=False) as data:
                    pending_in = {key: np.asarray(data[key])
                                  for key in data.files}
            result = run_member_leg(
                ctx, leg=task["leg"], name=name,
                t_start=task["t_start"], t_end=task["t_end"],
                analysis_due=task["analysis_due"],
                snapshot_in=snapshot_in, pending_in=pending_in)
            _save_snapshot(Path(task["snapshot_out"]), result.snapshot)
            payload = {"ok": True, "entry": result.entry,
                       "name": task["name"]}
            if task["refl_out"]:
                np.save(task["refl_out"], result.refl_host)
                payload["refl_out"] = task["refl_out"]
            if result.hot_increments is not None:
                Path(task["hot_out"]).parent.mkdir(parents=True,
                                                   exist_ok=True)
                np.savez(task["hot_out"], **result.hot_increments)
                payload["hot_out"] = task["hot_out"]
            if result.thb_snapshot is not None and task["thb_out"]:
                np.save(task["thb_out"], result.thb_snapshot)
                payload["thb_out"] = task["thb_out"]
        except BaseException as error:  # noqa: BLE001
            payload = {"ok": False, "name": task.get("name"),
                       "error": f"{type(error).__name__}: {error}",
                       "traceback": traceback.format_exc()[-4000:]}
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()
    return 0


# ---------------------------------------------------------------------------
# the parent-side pool
# ---------------------------------------------------------------------------

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
        self.width = int(width)
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        argv_file = self.workdir / "worker-argv.json"
        argv_file.write_text(json.dumps(list(argv)), encoding="utf-8")
        import os
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        root = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = (root + os.pathsep + env["PYTHONPATH"]
                             if env.get("PYTHONPATH") else root)
        command = [sys.executable, str(Path(__file__).resolve()),
                   "--worker", "--argv-file", str(argv_file)]
        self.procs = [subprocess.Popen(
            command, env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            bufsize=1) for _ in range(self.width)]
        for index, proc in enumerate(self.procs):
            line = proc.stdout.readline()
            if not line or not json.loads(line).get("ready"):
                raise RuntimeError(
                    f"member worker {index} did not come up: "
                    f"{proc.stderr.read()[-2000:]}")

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--argv-file", type=Path)
    args = parser.parse_args()
    if not args.worker:
        parser.error("this module is a library; --worker is its only "
                     "standalone entry point")
    return _worker_main(args.argv_file)


if __name__ == "__main__":
    raise SystemExit(main())
