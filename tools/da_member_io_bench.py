"""Measurement tooling: packed DA member data movement between legs.

Drives the real code of whichever tree is on sys.path: member_transport
write_common/write_request(s)/write_result/load_result, member_wave
validate_results, member_roster.forecast_roster (packed route, with the
process wave replaced by in-process worker publication), the controller's
analysis-input staging and the analysis' own member read + float64 prior
build, then ensemble_state.write_generation (the per-leg recovery
generation).  Same synthetic bytes for both trees.

Usage: python -m tools.da_member_io_bench --root DIR --members 32 [--nx 241 --ny 241 --nz 49]
Prints one JSON line of phase timings and RSS.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
try:
    import resource
except ImportError:
    resource = None
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def rss_gib():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024**2
    except OSError:
        pass
    return float("nan")


def peak_gib():
    if resource is None:
        return float("nan")
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


SNAPSHOT_3D = ["al", "alt", "effc", "effi", "effs", "h_diabatic", "nc", "ni",
               "nifa", "nr", "nwfa", "p", "qc", "qg", "qi", "qr", "qs", "qv",
               "thp"]
ANALYSIS = ("thp", "qv", "u", "v", "qs", "qg", "qc", "nc", "qr", "nr", "qi",
            "ni", "nwfa", "nifa")


def member_state(rng, nz, ny, nx):
    f = lambda *s: rng.random(s, dtype=np.float32)
    state = {name: f(nz, ny, nx) for name in SNAPSHOT_3D}
    state["u"] = f(nz, ny, nx + 1)
    state["v"] = f(nz, ny + 1, nx)
    state["w"] = f(nz + 1, ny, nx)
    state["php"] = f(nz + 1, ny, nx)
    state["mup"] = f(ny, nx)
    state["nwfa2d"] = f(ny, nx)
    state["nifa2d"] = f(ny, nx)
    return state


IDENTITY = None


def write_restart(path, member, state, extra, seconds):
    from gpuwm.io.restart import RESTART_FORMAT_VERSION
    from tools.da_cycle_prepared import trajectory_fingerprint
    header = {"format_version": RESTART_FORMAT_VERSION, "domain_ids": [1],
              "elapsed_ticks": int(seconds), "tick_den": 1,
              "elapsed_seconds": seconds, "step_count": int(seconds // 20),
              "domain_start_ticks": 0, "domain_lifecycle": "ACTIVE",
              "dtbc_fp32_bits": 0,
              "experiment_fingerprint_components": {"trajectory": str(member)},
              "experiment_fingerprint": trajectory_fingerprint(IDENTITY, member),
              "is_stub": True}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez(stream, __gpuwm_restart_header__=np.frombuffer(
            json.dumps(header).encode(), np.uint8),
            **{f"state/{k}": v for k, v in state.items()},
            **{f"driver/x{i:02d}": a for i, a in enumerate(extra)})
        stream.flush()
        os.fsync(stream.fileno())
    return header


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--members", type=int, default=32)
    parser.add_argument("--nx", type=int, default=241)
    parser.add_argument("--ny", type=int, default=241)
    parser.add_argument("--nz", type=int, default=49)
    parser.add_argument("--extra-3d", type=int, default=36)
    parser.add_argument("--obs-mib", type=int, default=300)
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)

    from gpuwm.da import member_transport as transport
    from gpuwm.da import member_wave as wave
    from gpuwm.da import member_roster
    from gpuwm.da import radar_assimilation as ra
    from tools import da_ensemble_state as ens
    from tools.da_member_leg import MemberLegContext, MemberLegResult
    from tools.da_ensemble_state import EnsembleIdentity

    new_tree = hasattr(ra, "CheckpointStateView")
    nz, ny, nx, R = args.nz, args.ny, args.nx, args.members
    root = args.root
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    rng = np.random.default_rng(7)
    out = {"label": args.label, "tree": "new" if new_tree else "base",
           "members": R, "shape": [nz, ny, nx]}
    t_end = 3600.0

    # -- shared fixtures ---------------------------------------------------
    extra = [rng.random((nz, ny, nx), dtype=np.float32)
             for _ in range(args.extra_3d)]
    base_state = member_state(rng, nz, ny, nx)
    obs_path, grid_path = root / "obs.nc", root / "grid.nc"
    obs_path.write_bytes(rng.bytes(args.obs_mib * 1024**2))
    grid_path.write_bytes(rng.bytes(64 * 1024**2))
    n_obs = args.obs_mib * 1024**2 // 4 // 8
    document = {"schema": "synthetic-radar", "variables": {
        f"v{i}": rng.random(n_obs, dtype=np.float32) for i in range(8)}}
    source = root / "proof.json"
    source.write_text('{"source":"bench"}')
    identity = EnsembleIdentity(members=R, nx=nx, ny=ny, nz=nz, dt_s=20.0,
                                mp_physics=28, physics_profile="bench",
                                prepared_content_sha256="0" * 64)
    global IDENTITY
    IDENTITY = identity
    cli = argparse.Namespace(prepared_root=root, authority_dir=None,
                             obs=[obs_path], grid_wrfout=[grid_path],
                             radar_tten=False, members=R,
                             physics_profile="bench",
                             prepared_content_sha256="0" * 64)
    inputs = SimpleNamespace(proof_path=source,
                             cache_identity={"case_sha256": "0" * 64},
                             experiment=SimpleNamespace(root=SimpleNamespace(
                                 run=SimpleNamespace(nx=nx, ny=ny, nz=nz,
                                                     dt=20.0, mp_physics=28))))

    @dataclasses.dataclass
    class Perturbation:
        amplitude: float = 0.25

    @dataclasses.dataclass
    class HotStart:
        active: bool = False

    # Leg-start restarts (the previous leg's accepted sets) and pending.
    stage_root = root / "out" / "stage"
    stage = __import__("tools.da_cycle_prepared", fromlist=["x"]).StagedRestarts(
        stage_root, default=True)
    previous = {}
    for name in ["control", *range(R)]:
        state = {k: v + np.float32(0.001 * (hash(str(name)) % 97))
                 for k, v in base_state.items()}
        path = stage.directory(0, name) / "gpuwmrst_d01_start.npz"
        write_restart(path, name, state, extra, 0.0)
        previous[name] = path
    pending = {m: {f: (rng.random(base_state[f].shape) - 0.5) * 1e-3
                   for f in ANALYSIS} for m in range(R)}
    out["setup_rss_gib"] = rss_gib()

    def context_for(name):
        return MemberLegContext(
            args=cli, inputs=inputs, identity=identity,
            cfg_perturb=Perturbation(), hot_cfg=HotStart(), leg=0,
            absolute_leg_number=1, t_start=0.0, t_end=t_end,
            stage_root=stage_root, out=root / "out", resumed=True,
            restart=previous[name],
            pending=None if name == "control" else pending[name],
            document=document, analysis_due=True, obs_path=obs_path,
            surface_enabled=False,
            setup_arrays={"c1h": np.ones(nz)}, thb_host=None)

    hz = {m: rng.random((nz, ny, nx)).astype(np.float64) for m in range(R)}
    timings = {}

    def control_leg(context, name):
        state = {k: v + np.float32(0.5) for k, v in base_state.items()}
        directory = stage.directory(1, name)
        path = directory / "gpuwmrst_d01_end.npz"
        write_restart(path, name, state, extra, t_end)
        return MemberLegResult(
            restart=path, snapshot=state, H_Z=None, H_surface=None,
            hot_pending=None, setup_arrays={"c1h": np.ones(nz)},
            thb_host=np.ones(nz, np.float32), nest_birth=None,
            record={"elapsed_seconds": t_end, "restart": {}},
            consume_restart=None, pending_consumed=False)

    def fake_wave(jobs, devices, members_per_card, output, timeout_seconds,
                  **kwargs):
        # Worker publication, as each worker process does it at leg end.
        started = time.perf_counter()
        jobs = [job if isinstance(job, dict) else job.result() for job in jobs]
        for job in jobs:
            member = job["member"]
            jroot = Path(job["output_root"])
            state = {k: v + np.float32(0.01 * (member + 1))
                     for k, v in base_state.items()}
            restart = jroot / "stage" / "gpuwmrst_d01_end.npz"
            write_restart(restart, member, state, extra, t_end)
            # The worker's context carries the run's arguments and the
            # leg's analysis flag: write_result reads both to decide
            # whether the member owes its shifted VTSM slots (4dcca7d16).
            context = SimpleNamespace(t_start=0.0, t_end=t_end, restart=None,
                                      args=cli, analysis_due=True)
            result = MemberLegResult(
                restart=restart, snapshot=state, H_Z=hz[member],
                H_surface=None, hot_pending=None,
                setup_arrays={"c1h": np.ones(nz)}, thb_host=None,
                nest_birth=None,
                record={"elapsed_seconds": t_end, "restart": {}},
                consume_restart=None, pending_consumed=True)
            transport.write_result(job["result_path"], context, member,
                                   result, job["request_hash"])
        timings["worker_publish_serial_s"] = time.perf_counter() - started
        timings["worker_publish_per_member_s"] = (
            timings["worker_publish_serial_s"] / len(jobs))
        started = time.perf_counter()
        results = wave.validate_results(jobs)
        timings["barrier_validate_s"] = time.perf_counter() - started
        return results

    real_run_wave = wave.run_wave
    wave.run_wave = fake_wave
    member_roster_started = time.perf_counter()
    try:
        results, _ = member_roster.forecast_roster(
            ["control", *range(R)], context_for=context_for, packed=True,
            devices=[], members_per_card=2, member_peak_bytes=0,
            workdir=stage_root / "packed" / "leg001", stage=stage,
            run_leg=control_leg, release=lambda: None, timeout_seconds=3600)
    finally:
        wave.run_wave = real_run_wave
    roster_wall = time.perf_counter() - member_roster_started
    out["roster_wall_s"] = roster_wall
    out.update(timings)
    # Controller-side roster cost: everything except the worker writes,
    # which run inside 16 concurrent worker processes in production.
    out["controller_roster_s"] = roster_wall - timings["worker_publish_serial_s"]
    out["rss_after_barrier_gib"] = rss_gib()

    # -- controller: analysis inputs ------------------------------------------
    started = time.perf_counter()
    snapshots = {}
    if new_tree:
        from tools.da_cycle_prepared import (analysis_backgrounds,
                                             member_background)
        for name, result in results.items():
            snapshots[name] = member_background(result)
        checkpoints = analysis_backgrounds(snapshots, R, directory=root / "x",
                                           t_end=t_end, files=False)
    else:
        for name, result in results.items():
            snapshots[name] = result.snapshot
        shm_leg = stage.analysis_directory(1)
        checkpoints = {}
        for index in range(R):
            member_dir = shm_leg / f"member_{index:03d}"
            member_dir.mkdir(parents=True)
            path = member_dir / f"gpuwmrst_d01_{int(t_end):06d}.npz"
            np.savez(path, **{f"state/{k}": v
                              for k, v in snapshots[index].items()})
            checkpoints[index] = path
    out["stage_analysis_inputs_s"] = time.perf_counter() - started

    # The controller's spread screen, then the analysis' member read and
    # float64 prior, exactly as assimilate_radar_grid builds them.
    started = time.perf_counter()
    carried = set(snapshots[0])
    for name in [n for n in ANALYSIS if n in carried]:
        stack = np.stack([snapshots[i][name] for i in range(R)]).astype(
            np.float64)
        float(np.abs(stack).max())
        del stack
    out["spread_screen_s"] = time.perf_counter() - started
    started = time.perf_counter()
    if new_tree:
        states = ra.member_states(checkpoints)
    else:
        states = {i: ra.read_checkpoint_state(checkpoints[i])
                  for i in range(R)}
    out["analysis_member_read_s"] = time.perf_counter() - started
    started = time.perf_counter()
    prior = {}
    for name in ANALYSIS:
        prior[name] = np.stack([ra._mass_field(name, states[i], "bench")
                                for i in range(R)])
    for i in range(R):
        for name in ("w", "p", "alt"):
            np.asarray(states[i][name])
    out["prior_build_s"] = time.perf_counter() - started
    out["rss_at_prior_gib"] = rss_gib()
    out["peak_rss_through_prior_gib"] = peak_gib()
    del prior, states
    if not new_tree:
        shutil.rmtree(stage.analysis_directory(1), ignore_errors=True)

    # -- controller: per-leg recovery generation ------------------------------
    started = time.perf_counter()
    ens.write_generation(root / "out" / "packed-recovery" / "slot",
                         identity=identity, elapsed_seconds=t_end,
                         leg_number=1,
                         restarts={n: r.restart for n, r in results.items()},
                         pending={"control": None, **pending})
    out["recovery_generation_s"] = time.perf_counter() - started
    out["peak_rss_gib"] = peak_gib()
    out["controller_total_s"] = (out["controller_roster_s"]
                                 + out["stage_analysis_inputs_s"]
                                 + out["analysis_member_read_s"]
                                 + out["recovery_generation_s"])
    print(json.dumps(out), flush=True)
    for background in snapshots.values():
        close = getattr(background, "close", None)
        if callable(close):
            close()
    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
