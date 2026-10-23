"""NextGen standalone command door; Python enqueues, Rust writes products."""
from __future__ import annotations

from pathlib import Path
import math
import time


def register_cli(subparsers):
    parser = subparsers.add_parser("hydro", help="run GPU NextGen catchments and river routing")
    commands = parser.add_subparsers(dest="hydro_command", required=True)
    run = commands.add_parser("run", help="run a prepared NextGen realization")
    run.add_argument("config", type=Path)
    run.add_argument("--seconds", required=True, type=float)
    run.add_argument("--device", type=int, default=0)
    run.add_argument("--outdir", required=True, type=Path)
    run.add_argument("--restart", type=Path)
    run.set_defaults(func=run_main)
    ahead = commands.add_parser(
        "ahead", help="build (or reuse) the covering fabric `gpuwm go CONFIG --hydro on` binds, before that launch")
    ahead.add_argument("config", type=Path)
    ahead.set_defaults(func=ahead_main)
    from gpuwm.hydro_demo import register_demo_cli
    register_demo_cli(commands, runner=run_prepared)
    from gpuwm.hydrofabric import register_cli as register_fabric
    from gpuwm.hydro_preprocess import register_cli as register_preparation
    from .verify_usgs import register_cli as register_verification
    from gpuwm.hydro_retrospective import register_cli as register_retrospective
    register_fabric(commands)
    register_preparation(commands)
    register_verification(commands)
    register_retrospective(commands)
    from .initial_state import register_cli as register_initial_state
    register_initial_state(commands)
    from .forcing_inputs import register_cli as register_forcing
    register_forcing(commands)
    from .calibrate import register_cli as register_calibration
    register_calibration(commands)


def ahead_main(args):
    """Build the fabric ``gpuwm go CONFIG --hydro on`` would build, into the same cache.

    The fabric cache is keyed by the domain and the preparation request
    (gpuwm.hydro.door.materialize), so the launch that follows reuses this
    build instead of making it. Breakage it prevents: a forecast from a
    prepared bundle (WOOF prepare-ahead, Oklahoma City 2026-10-23) waited 64 s
    for the fabric of a domain new to the box, with no fetch or preparation to
    build it beside; the box can build it while it waits for the bundle.
    Exit 0 when hydro is enabled, 1 when no fabric covers CONFIG.
    """
    import json
    from .runtime import _launch_settings
    started = time.perf_counter()
    settings, _path, decision = _launch_settings("", args.config)
    decision = {key: value for key, value in decision.items() if not key.startswith("_")}
    print(json.dumps({"enabled": bool(settings.get("enabled")), "reason": decision.get("reason"),
                      "build": decision.get("build"), "seconds": round(time.perf_counter() - started, 3)},
                     default=str, sort_keys=True), flush=True)
    return 0 if settings.get("enabled") else 1


def run_main(args):
    run_prepared(settings=args.config, seconds=args.seconds, outdir=args.outdir,
                 device=args.device, restart=args.restart)
    return 0


def run_prepared(*, settings, seconds, outdir, device=0, restart=None):
    from .runtime import _prepare_resolved, read_settings, resolve_settings
    settings = read_settings(settings)
    settings["enabled"] = True
    settings = resolve_settings(settings)
    from gpuwm import numerics
    numerics.select(settings.get("numerics", "fast"))
    interval = float(settings["interval_seconds"])
    duration = float(seconds)
    if (not math.isfinite(duration) or not math.isfinite(interval)
            or duration <= 0 or interval <= 0 or duration % interval != 0):
        raise ValueError("--seconds must be a positive whole number of hydro intervals")
    import cupy as cp
    with cp.cuda.Device(device):
        driver = _prepare_resolved(settings, device_id=device)
        # Output lane owns Rust snapshot/restore. The selected realization
        # binds this boundary once; no output method runs in the step loop.
        import importlib
        target = settings.get("output_factory", "gpuwm.hydro.output:prepare")
        module, separator, name = target.partition(":")
        if not separator or not module.startswith("gpuwm."):
            raise ValueError("hydro output_factory must name a gpuwm module:callable")
        writer = getattr(importlib.import_module(module), name)(
            settings=settings, driver=driver, outdir=Path(outdir))
        if restart is not None:
            writer.restore(Path(restart), driver)
        stream = cp.cuda.get_current_stream()
        stream.synchronize()
        started = time.perf_counter()
        for _ in range(int(duration / interval)):
            driver.before_lsm(stream=stream)
            driver.advance(stream=stream)
        stream.synchronize()
        completed = time.perf_counter() - started
        receipt = {"hydro_completed_seconds": completed,
                   "hydro_memory": getattr(driver, "memory_admission", None),
                   "numerics": numerics.receipt(),
                                      "steps": int(duration / interval),
                                      "parallelism": "GPU; host prebound launch sequence",
                                      "boundary_synchronizations": 2,
                   "driver_per_step_host_data_work": False,
                   "stages": [{"name": stage.name, "parallelism": stage.parallelism}
                              for stage in driver.prepared_stages]}
        writer.finish(driver, receipt=receipt)
    return receipt
