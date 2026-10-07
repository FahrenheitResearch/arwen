"""Leg-0 balance measurement: perturbed members against the control.

Runs N perturbed members, or the unperturbed control, from ONE prepared
single-domain case on one card, for the first ``--minutes`` of the
forecast, and records what the member start does to the model:

* MASPT (:mod:`gpuwm.da.maspt`): the interior mean absolute surface
  pressure tendency per step, the balance measure DA gate G4 compares
  against the no-DA start, plus the perturbation's own insertion jump.
* Level snapshots of ``u``, ``v``, ``theta'`` and ``qv`` at the chosen
  minutes, so the report can measure how much of the inserted spread the
  model keeps (spread retention) rather than radiates away.

Arms (``--arm``), all on the same seeds so the draws are comparable:

* ``control``: no perturbation.
* ``rotational``: the default member start since 2026-10-06 (one
  streamfunction for the winds, hydrostatic ``php`` after the
  thermodynamic draw).
* ``independent``: independent u/v draws, hydrostatic mass.
* ``rotational-nomass``: streamfunction winds, no mass balance.
* ``independent-nomass``: the pre-2026-10-06 member start.

The perturbation amplitudes default to the DA cycle tool's
(``tools/da_cycle_prepared.py``): 1.5 m/s at 150 km, 0.5 K and 5 percent
vapour at 60 km over three levels, species at 0.7 log-sigma.

The forecast body (preflight, restore, physics, clock, node, model) is
the member leg's (``tools/da_member_leg.py``), without the restart and
analysis machinery.  Nothing here writes history; the run folder holds
``maspt.json``, ``provenance.json`` and ``snapshots.npz`` per trajectory.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np

ARMS = {
    "control": None,
    "rotational": {"wind_mode": "rotational", "mass_balance": "hydrostatic"},
    "independent": {"wind_mode": "independent", "mass_balance": "hydrostatic"},
    "rotational-nomass": {"wind_mode": "rotational", "mass_balance": "none"},
    "independent-nomass": {"wind_mode": "independent", "mass_balance": "none"},
}


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--prepared", type=Path, required=True,
                   help="the prepared single-domain case (holds proof.json)")
    p.add_argument("--experiment-config", type=Path, required=True)
    p.add_argument("--wps-namelist", type=Path, required=True)
    p.add_argument("--source", default=None,
                   help="the preparation's source (default: proof.json's)")
    p.add_argument("--arm", choices=sorted(ARMS), required=True)
    p.add_argument("--members", type=int, default=4)
    p.add_argument("--first-seed", type=int, default=20260731)
    p.add_argument("--minutes", type=float, default=60.0)
    p.add_argument("--snapshot-minutes", default="0,10,20,30,45,60")
    p.add_argument("--levels", default="0,2,5,10,20,30")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--wind-sigma-ms", type=float, default=1.5)
    p.add_argument("--length-scale-km", type=float, default=150.0)
    p.add_argument("--theta-sigma-k", type=float, default=0.5)
    p.add_argument("--qv-log-sigma", type=float, default=0.05)
    p.add_argument("--hydro-log-sigma", type=float, default=0.7)
    p.add_argument("--thermo-length-scale-km", type=float, default=60.0)
    p.add_argument("--thermo-vertical-levels", type=float, default=3.0)
    p.add_argument("--clip-sigmas", type=float, default=2.5)
    p.add_argument("--no-hydrometeors", action="store_true",
                   help="skip the theta/qv/species draws (winds only)")
    p.add_argument("--preflight-only", action="store_true",
                   help="bind the prepared case, build the arm's "
                        "perturbation config and stop (no card needed)")
    return p.parse_args(argv)


def perturbation_config(args, cfg, arm_options):
    """The DA cycle tool's member perturbation, with the arm's balance."""
    from gpuwm.da import moments, perturb

    fields = [
        {"name": "u", "amplitude": args.wind_sigma_ms,
         "length_scale_km": args.length_scale_km},
        {"name": "v", "amplitude": args.wind_sigma_ms,
         "length_scale_km": args.length_scale_km},
    ]
    species: list[dict] = []
    if not args.no_hydrometeors:
        scheme = moments.scheme_moments(int(cfg.mp_physics))
        fields += [
            {"name": "theta", "amplitude": args.theta_sigma_k,
             "length_scale_km": args.thermo_length_scale_km,
             "vertical_scale_levels": args.thermo_vertical_levels},
            {"name": "qv", "amplitude": args.qv_log_sigma,
             "length_scale_km": args.thermo_length_scale_km,
             "vertical_scale_levels": args.thermo_vertical_levels,
             "mode": "lognormal", "clip_sigmas": args.clip_sigmas},
        ]
        species = [
            {"mass_field": name, "amplitude": args.hydro_log_sigma,
             "length_scale_km": args.thermo_length_scale_km,
             "vertical_scale_levels": args.thermo_vertical_levels,
             "clip_sigmas": args.clip_sigmas,
             "threshold_kg_kg": scheme.q_threshold}
            for name in scheme.mass_fields
            if name in perturb.SUPPORTED_SPECIES]
    return perturb.PerturbationConfig.from_mapping({
        "dx_km": float(cfg.dx) / 1000.0, "dy_km": float(cfg.dy) / 1000.0,
        "rim_width": 5, "fields": fields, "species": species,
        "hypsometric_opt": int(cfg.hypsometric_opt), **arm_options})


class Snapshots:
    """Host copies of a few levels of the prognostics at chosen minutes."""

    FIELDS = ("u", "v", "thp", "qv")

    def __init__(self, levels, minutes):
        self.levels = [int(k) for k in levels]
        self.pending = sorted(float(m) * 60.0 for m in minutes)
        self.taken: dict[str, np.ndarray] = {}
        self.times: list[float] = []

    def capture(self, state, elapsed):
        tag = f"{int(round(elapsed / 60.0)):03d}"
        for name in self.FIELDS:
            array = getattr(state, name)[self.levels]
            self.taken[f"{name}_{tag}"] = np.asarray(
                getattr(array, "get", lambda: array)(), dtype=np.float32)
        p0 = state.p[0]
        self.taken[f"psfc_{tag}"] = np.asarray(
            getattr(p0, "get", lambda: p0)(), dtype=np.float32)
        self.times.append(float(elapsed))

    def stepper(self, step):
        def snapshot_step(state, run, **kwargs):
            result = step(state, run, **kwargs)
            after = float(state.elapsed_seconds)
            dt = float(getattr(run, "dt", 1.0))
            while self.pending and self.pending[0] <= after + 0.5 * dt:
                target = self.pending.pop(0)
                if target > 0.0:
                    self.capture(state, target)
            return result
        return snapshot_step


def run_trajectory(args, inputs, name, seed, arm_options, out: Path):
    import cupy as cp  # noqa: F401 - the card must be here
    from gpuwm.core.clock import build_schedule, resolve_clock
    from gpuwm.core.dycore import step as dycore_step
    from gpuwm.core.model import (DomainNode, ExperimentState,
                                  ModelRuntimeStatus, execute_experiment)
    from gpuwm.da import perturb
    from gpuwm.da.maspt import MasptRecorder
    from gpuwm.ensemble.member import refresh_diagnostics
    from gpuwm.ingest.hrrr_physics import initialize_prepared_physics
    from gpuwm.ingest.lateral_bc import bind_lateral_boundary_clock
    from gpuwm.ingest.prepared_cache import restore_prepared_cache
    from gpuwm.runtime import declared_constant_glw

    exp = inputs.experiment
    cfg = exp.root.run
    seconds = float(args.minutes) * 60.0
    t0 = time.time()
    phases = {}

    exp_leg = dataclasses.replace(exp, run_seconds=seconds)
    restored = restore_prepared_cache(
        inputs.prepared_cache_path, expected_identity=inputs.cache_identity,
        cfg=cfg, static=inputs.static)
    driver = initialize_prepared_physics(
        restored.initial_result, cfg, restored.met, restored.surface,
        inputs.static, inputs.landuse_identity, inputs.grid, exp.start_time,
        constant_glw_wm2=declared_constant_glw(exp))
    tick = resolve_clock(
        exp_leg, lbc_interval_s=float(inputs.boundary_interval_seconds),
        live_born_children=())
    schedule = build_schedule(exp_leg, tick)
    clocks = tick.clocks()
    node = DomainNode(exp.root, inputs.grid, restored.initial_result.state,
                      clocks[1], None, [], None)
    if getattr(cfg, "specified", False):
        bind_lateral_boundary_clock(node.state, node.clock)
    state = node.state
    phases["build"] = time.time() - t0

    model = ExperimentState(node, MappingProxyType({1: node}), schedule, None,
                            f"da-leg0-balance:{args.arm}:{name}")
    model._experiment_fingerprint_components = {
        "tool": "da_leg0_balance", "arm": args.arm, "trajectory": name}
    model._runtime_status = ModelRuntimeStatus()
    model._resumed = False
    model._resume_committed_history_grid_ids = frozenset()
    model._scratch_arena = None
    model._dycore_state_workspace = None
    model._io_manager = None
    model._last_checkpoint = None
    model._prepared_by_grid_id = MappingProxyType({
        1: SimpleNamespace(static_fields=inputs.static, geog_selection=None,
                           initial_result=restored.initial_result)})

    maspt = MasptRecorder.for_config(cfg, start_seconds=0.0,
                                     horizon_seconds=seconds)
    provenance = None
    if arm_options is not None:
        maspt.before_insertion(state)
        cfg_perturb = perturbation_config(args, cfg, arm_options)
        provenance = perturb.apply_perturbations(state, int(seed), cfg_perturb)
        provenance["refresh"] = refresh_diagnostics(
            state, hypsometric_opt=int(cfg.hypsometric_opt))
        phases["perturb"] = time.time() - t0 - phases["build"]
    snaps = Snapshots([int(k) for k in args.levels.split(",")],
                      [float(m) for m in args.snapshot_minutes.split(",")])
    snaps.capture(state, 0.0)
    snaps.pending = [t for t in snaps.pending if t > 0.0]

    root_step = snaps.stepper(maspt.stepper(dycore_step))
    t1 = time.time()
    execute_experiment(model, history_handler=None, progress_callback=None,
                       validate_state=True, skip_feedback_path=True,
                       steppers={int(node.cfg.grid_id): root_step})
    phases["integrate"] = time.time() - t1

    out.mkdir(parents=True, exist_ok=True)
    (out / "maspt.json").write_text(json.dumps(maspt.receipt(), indent=1))
    np.savez_compressed(out / "snapshots.npz", times=np.asarray(snaps.times),
                        levels=np.asarray(snaps.levels), **snaps.taken)
    if provenance is not None:
        (out / "provenance.json").write_text(
            json.dumps(provenance, indent=1, default=str))
    (out / "done.json").write_text(json.dumps({
        "arm": args.arm, "trajectory": name, "seed": seed,
        "minutes": args.minutes, "phases_s": phases,
        "wall_s": time.time() - t0, "grid": [int(cfg.nz), int(cfg.ny),
                                             int(cfg.nx)],
        "dt": float(cfg.dt), "hypsometric_opt": int(cfg.hypsometric_opt)}))
    del model, node, state, driver, restored
    print(f"{args.arm} {name}: done in {time.time() - t0:.0f} s "
          f"(maspt30 {maspt.receipt()['maspt_first_30min']:.2f} hPa/h)",
          flush=True)


def bind_prepared(prepared: Path) -> dict:
    """The digests the forecast preflight binds, the way ``gpuwm sim``
    binds them: a sealed proof by its own three digests, a chained
    preparation (a ``boundary-stream`` under the prepared root, whose
    proof carries no top-level input manifest digest) by its head.  The
    first box S attempt (2026-10-06 16:23Z) bound only the former and
    was refused on the chained case before any arm ran."""
    from gpuwm.go_cli import GoRefusal, head_digests, proof_digests
    from gpuwm.ingest.boundary_stream import (
        BoundaryStreamError, head_sha256, read_head)

    try:
        return proof_digests(prepared)
    except GoRefusal as refusal:
        try:
            head = read_head(prepared)
        except (BoundaryStreamError, OSError) as error:
            raise SystemExit(
                f"{prepared}: not a sealed single-domain proof ({refusal}) "
                f"and no readable boundary-stream head ({error})") from None
        return head_digests(prepared, head_sha256(head))


def preflight(args):
    from gpuwm.prepared_single_domain_forecast import (
        preflight_prepared_forecast)

    digests = bind_prepared(args.prepared)
    proof = json.loads((args.prepared / "proof.json").read_text())
    source = args.source
    if not source:
        initial = proof.get("initial_source")
        source = initial if isinstance(initial, str) else (
            initial.get("source") if isinstance(initial, dict) else None)
    if not source:
        raise SystemExit("--source: proof.json names no single source; "
                         "pass the preparation's (e.g. rap-native)")
    inputs = preflight_prepared_forecast(
        source=str(source), prepared_root=args.prepared,
        proof_sha256=digests.get("proof"),
        source_manifest_sha256=digests.get("source_manifest"),
        prepared_content_sha256=digests.get("prepared_content"),
        prepared_head_sha256=digests.get("prepared_head"),
        experiment_config=args.experiment_config,
        wps_namelist=args.wps_namelist,
        run_seconds=float(args.minutes) * 60.0,
        history_interval_seconds=float(args.minutes) * 60.0)
    return inputs, source, digests


def main(argv=None) -> int:
    args = parse(argv)
    inputs, source, digests = preflight(args)
    if args.preflight_only:
        cfg = inputs.experiment.root.run
        options = ARMS[args.arm]
        built = (None if options is None
                 else dataclasses.asdict(perturbation_config(args, cfg,
                                                             options)))
        print(json.dumps({
            "preflight": "ok", "binding": digests, "source": str(source),
            "grid": [int(cfg.nz), int(cfg.ny), int(cfg.nx)],
            "dx_km": float(cfg.dx) / 1000.0,
            "hypsometric_opt": int(cfg.hypsometric_opt),
            "perturbation": built}, indent=1, default=str))
        return 0
    arm_options = ARMS[args.arm]
    out = args.out / args.arm
    out.mkdir(parents=True, exist_ok=True)
    (out / "arm.json").write_text(json.dumps({
        "arm": args.arm, "options": arm_options, "members": args.members,
        "first_seed": args.first_seed, "minutes": args.minutes,
        "source": str(source), "prepared": str(args.prepared),
        "argv": sys.argv[1:]}, indent=1))
    if arm_options is None:
        run_trajectory(args, inputs, "control", 0, None, out / "control")
        return 0
    for k in range(int(args.members)):
        name = f"m{k + 1:02d}"
        if (out / name / "done.json").exists():
            print(f"{args.arm} {name}: already done", flush=True)
            continue
        run_trajectory(args, inputs, name, args.first_seed + k + 1,
                       arm_options, out / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
