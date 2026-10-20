"""Qualify the mass worker's real spawn/queue with a routing-only control.

The atmosphere, checkpoint input and exclusive-card preflight are injected.
Routing factories, GPU state, clocks, mass worker, pipes, output capture and
Rust checkpoint/product transport are real. Run inside the canonical pool.
This tool neither launches a weather ensemble nor changes the public launcher.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime, timedelta
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch


class AtmosphereBoundary:
    def __init__(self, *args):
        self.frames = []

    def __call__(self, **row):
        self.frames.append({"grid_id": row["grid_id"], "valid_time": str(row["valid_time"])})

    def finish(self):
        return self.frames


def load_boundary(parent, directory, hours):
    return (SimpleNamespace(experiment_config=parent["experiment_config"]), None,
            SimpleNamespace(path=Path(parent["checkpoint"]), sha256="injected-header-no-restart-proof"))


def routing_forecast(inputs, *, output_directory, **kwargs):
    # _card_worker has selected CUDA visibility before importing CuPy.
    import cupy as cp
    from gpuwm.ensemble.runtime_context import current_capture
    from gpuwm.hydro import runtime
    from gpuwm.hydro.assembly import prepare
    from tools.nextgen_ensemble_network import write_member

    capture = current_capture()
    member = capture.member_id
    settings = runtime.current_settings()
    if settings["member_tag"] != member or capture.provenance["parent"]["source"] != "routing-control":
        raise RuntimeError("spawned worker lost member settings/provenance")
    settings = json.loads(Path(settings["routing_settings"]).read_text())
    started = time.perf_counter()
    driver = prepare(settings=settings, state=None, physics=None, device_id=0, mode="standalone")
    prepare_ms = (time.perf_counter() - started) * 1000
    state = SimpleNamespace(physics=SimpleNamespace(hydro=driver))
    interval = settings["interval_seconds"]
    steps = driver.components["history"].capacity_steps
    cp.cuda.get_current_stream().synchronize()
    begin, end = cp.cuda.Event(), cp.cuda.Event()
    begin.record()
    started = time.perf_counter()
    for _ in range(steps):
        driver.before_lsm()
        driver.advance(interval)
    end.record()
    end.synchronize()
    wall_ms = (time.perf_counter() - started) * 1000
    cuda_ms = cp.cuda.get_elapsed_time(begin, end)
    valid_time = datetime.fromisoformat(settings["time_origin"].replace("Z", "+00:00")) + timedelta(seconds=steps*interval)
    started = time.perf_counter()
    capture.submit(state=state, valid_time=valid_time.isoformat().replace("+00:00", "Z"),
                   grid_id=1, streamed=None, metadata={}, refl_field=None)
    capture_enqueue_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    write_member(Path(output_directory) / "routing-control.nc", driver)
    member_write_ms = (time.perf_counter() - started) * 1000
    return {"scope": "real routing-only control; injected atmosphere/header/preflight",
            "pid": os.getpid(), "parent_pid": os.getppid(), "member_id": member,
            "settings_member_tag": member, "card": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "numerics": driver.restart_identity["numerics"], "steps": steps,
            "prepare_ms": prepare_ms, "routing_history_cuda_ms_per_step": cuda_ms / steps,
            "routing_history_wall_ms_per_step": wall_ms / steps,
            "capture_enqueue_ms": capture_enqueue_ms, "member_write_rust_ms": member_write_ms,
            "driver_identity_sha256": hashlib.sha256(json.dumps(driver.restart_identity, sort_keys=True).encode()).hexdigest(),
            "cuda_graphs": False, "per_step_host_field_work": False}


def spawned_worker(target, args):
    # The injected module import can import CUDA helpers: bind visibility
    # first, exactly as the actual target does before its own CuPy import.
    os.environ["CUDA_VISIBLE_DEVICES"] = args[1]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    import gpuwm.ensemble.mass_launcher as launcher
    with patch.object(launcher, "_load_inputs", side_effect=load_boundary), \
         patch.object(launcher, "ThinCapture", AtmosphereBoundary), \
         patch("gpuwm.prepared_domain_tree_forecast.run_prepared_tree", side_effect=routing_forecast):
        target(*args)


def run(settings_path, out, card):
    from gpuwm import run_stamp, supervisor
    import gpuwm.ensemble.mass_launcher as launcher
    from gpuwm.ensemble.restart_roster import atomic_json

    out.mkdir(parents=True, exist_ok=False)
    settings_path = settings_path.resolve()
    parents = []
    for member in (0, 1):
        config = out / f"member-{member}.toml"
        config.write_text(f'[hydro]\nenabled=true\nmember_tag={member}\nrouting_settings={json.dumps(str(settings_path))}\n')
        checkpoint = out / f"header-boundary-{member}.nc"
        checkpoint.touch()  # Explicitly not a physical checkpoint/restart.
        parents.append({"parent_id": member, "checkpoint": str(checkpoint), "prepared_root": str(out),
            "experiment_config": str(config), "provenance": {"source": "routing-control", "cycle": "2023-04-02T00:00:00Z"}})
    parent_file = out / "parents.json"
    atomic_json(parent_file, parents)
    policy = {"thresholds_m3_s": [0.005, 0.15, 10.0], "device_budget_bytes": 1048576}
    config = out / "launch.toml"
    config.write_text('[hydro]\nenabled=false\n[ensemble.hydro]\nthresholds_m3_s=[0.005,0.15,10.0]\ndevice_budget_bytes=1048576\n')
    destination = out / "set"
    context = multiprocessing.get_context("spawn")

    class Context:
        Pipe = staticmethod(context.Pipe)

        @staticmethod
        def Process(*, target, args):
            return context.Process(target=spawned_worker, args=(target, args))

    args = SimpleNamespace(branch_from=parent_file, members=2, config=config, wind_amplitude=0.,
        ensemble_seed=0, branch_surface=False, branch_hours=1., dry_run=False,
        cards="all", outdir=str(destination), keep_member_files=False)
    # The canonical pool already owns this lease. Public exclusive-card
    # admission is deliberately outside this routing/worker qualification.
    with patch.object(launcher, "select_cards", return_value=(SimpleNamespace(uuid=card),)), \
         patch.object(supervisor, "query_gpus", return_value=[]), \
         patch.object(supervisor, "GPUFileLock", side_effect=lambda *a, **kw: nullcontext()), \
         patch.object(supervisor, "preflight_exclusive_gpu"), \
         patch.object(run_stamp, "resolve", side_effect=lambda path, **kw: Path(path)), \
         patch.dict("sys.modules", {"gpuwm.ensemble.mass_products": SimpleNamespace(reduce_set=lambda *a: [])}), \
         patch.object(multiprocessing, "get_context", return_value=Context()):
        assert launcher.run(args) == 0
    receipt = json.loads((destination / "ensemble-set.json").read_text())
    assert receipt["nextgen_options"] == policy
    rows = sorted(receipt["completed"], key=lambda row: row["member_id"])
    forecasts = [json.loads(Path(row["receipt"]).read_text())["forecast"] for row in rows]
    assert all(row["pid"] != os.getpid() and row["parent_pid"] == os.getpid() for row in forecasts)
    assert len({row["pid"] for row in forecasts}) == 1  # One persistent worker, two queued members.
    assert [row["settings_member_tag"] for row in forecasts] == [0, 1]
    assert all(row["nextgen_points"]["options"] == policy for row in rows)
    assert all(len(row["frames"]) == 1 for row in rows)
    import cupy as cp
    from gpuwm.netcdf_bridge import Dataset
    histories = []
    for member in (0, 1):
        with Dataset(destination / "members" / f"member-{member:05d}" / "routing-control.nc") as source:
            source.set_auto_maskandscale(False)
            histories.append(cp.asarray(source.variables["discharge_m3_s"][:], dtype=cp.float32))
    assert bool(cp.array_equal(histories[0].view(cp.uint32), histories[1].view(cp.uint32)))
    q = histories[0][-1]
    threshold = cp.asarray(policy["thresholds_m3_s"], dtype=cp.float64).astype(cp.float32)
    expected_count = (q[None, :] >= threshold[:, None]).astype(cp.uint32) * 2
    expected_finite = cp.isfinite(q).astype(cp.uint32) * 2
    expected_probability = cp.where(expected_finite[None, :] == 2, expected_count.astype(cp.float32) / 2, cp.float32("nan"))
    with Dataset(receipt["nextgen"]["products"][0]) as source:
        source.set_auto_maskandscale(False)
        for variable, expected in (("discharge_threshold", threshold), ("exceedance_count", expected_count),
                                   ("finite_count", expected_finite), ("probability", expected_probability)):
            actual = cp.asarray(source.variables[variable][:], dtype=expected.dtype)
            assert bool(cp.array_equal(actual, expected, equal_nan=True)), variable
    result = {"status": "PASS", "scope": "real spawn/queue/routing/CUDA/Rust product control, no weather or checkpoint restore",
        "requested_member_ids": [0, 1], "parent_pid": os.getpid(), "worker_forecasts": forecasts,
        "product_options": policy, "member_history_differences": 0, "history_words_per_member": int(histories[0].size),
        "probability_oracle": "exact counts/finite/fractions on CUDA from actual final Q", "canonical_pool_shared": True,
        "physical_card": card, "process_start_method": "spawn", "card_workers": 1,
        "worker_members": "serial, fresh driver each; previous member released before next",
        "mass_worker_queue_seconds": receipt["forecast_wall_seconds"],
        "parent_product_seconds": receipt["product_seconds"], "launcher_wall_seconds": receipt["wall_seconds"],
        "parallelism": "MC128threads272levels;19LP17subsets128threads;history/count/fraction256threads;2parent-prefetch host threads",
        "cuda_graphs": False, "production_mass_attachment": "unapplied owner patch staged against 63d4106",
        "native_comparison": "no new native replay; compare saved CDF SHA to prior control in receipt", "set": str(destination)}
    atomic_json(out / "probe-result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--card", required=True)
    options = parser.parse_args()
    run(options.settings, options.out.resolve(), options.card)
