"""Resume real resident routing and ensemble output from a mid-run checkpoint.

Two identical captured-input members, each with independent routing storage.
The atmospheric collector is a metadata stub; this does not qualify weather
restart, meteorological spread, a mass launch or physical runoff coupling.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time
from types import SimpleNamespace

import cupy as cp

from gpuwm.ensemble.nextgen_capture import NextGenCollector
from gpuwm.ensemble.nextgen_inventory import ResidentInventory
from gpuwm.ensemble.restart_roster import atomic_json, digest_file
from gpuwm.ensemble.runtime_context import MemberOutputCapture, member_output_scope, current_capture
from gpuwm.hydro.assembly import prepare
from gpuwm.hydro.output import Writer
from tools.nextgen_ensemble_network import write_member


def collector(root):
    original = SimpleNamespace(root=root, submit=lambda **kw: None,
        require_complete=lambda: None, finish_run=lambda: [],
        save_resume=lambda: {"files": []}, restore_resume=lambda: {})
    return NextGenCollector(original, (0, 1), array_module=cp,
        options={"thresholds_m3_s": [.005, .15, 10.], "device_budget_bytes": 1 << 20})


def compare(left, right):
    """Compare every carrier byte on CUDA, including int64 IDs and NaNs."""
    assert left.checkpoint_metadata() == right.checkpoint_metadata()
    assert set(left.arrays) == set(right.arrays)
    rows = []
    for name, value in left.arrays.items():
        other = right.arrays[name]
        assert value.shape == other.shape and value.dtype == other.dtype
        different = int(cp.count_nonzero(value.reshape(-1).view(cp.uint8)
                                       != other.reshape(-1).view(cp.uint8)).item())
        rows.append(dict(name=name, bytes=value.nbytes, different_bytes=different))
    assert not any(row["different_bytes"] for row in rows)
    return dict(carriers=len(rows), logical_bytes=sum(row["bytes"] for row in rows),
                different_bytes=0, arrays=rows)


def run(settings_path, out):
    started = time.perf_counter()
    settings = json.loads(settings_path.read_text())
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "settings.json", settings)
    baseline_root, resume_root = out / "baseline", out / "resumed"
    baseline = collector(baseline_root)
    pending = collector(resume_root)
    drivers, restored, setup, timings, checkpoints = [], [], [], [], []
    inventory = ResidentInventory()

    def build(member, owner, group):
        begin = time.perf_counter()
        driver = prepare(settings=settings, state=None, physics=None, device_id=0, mode="standalone")
        owner.bind_member(member_id=member, state=SimpleNamespace(physics=SimpleNamespace(hydro=driver)))
        inventory.inspect(driver, member_id=2*group+member, device_id=0)
        setup.append(dict(group=group, member_id=member, wall_ms=(time.perf_counter()-begin)*1000))
        return driver

    for member in (0, 1):
        drivers.append(build(member, baseline, 0))
    steps = drivers[0].components["history"].capacity_steps
    split = steps // 2
    assert 0 < split < steps
    interval = settings["interval_seconds"]
    origin = datetime.fromisoformat(settings["time_origin"].replace("Z", "+00:00"))

    def capture(owner, member, driver, step):
        valid = (origin + timedelta(seconds=step*interval)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        before = time.perf_counter()
        with member_output_scope(MemberOutputCapture(owner.submit, member)):
            current_capture().submit(state=SimpleNamespace(physics=SimpleNamespace(hydro=driver)),
                valid_time=valid, grid_id=1, streamed=None, metadata={}, refl_field=None)
        return (time.perf_counter()-before)*1000

    def advance(driver, member, phase, count):
        first, last = cp.cuda.Event(), cp.cuda.Event()
        first.record()
        wall = time.perf_counter()
        for _ in range(count):
            driver.before_lsm()
            driver.advance(interval)
        last.record()
        last.synchronize()
        timings.append(dict(member_id=member, phase=phase, steps=count,
            routing_history_cuda_ms_per_step=cp.cuda.get_elapsed_time(first, last)/count,
            routing_history_wall_ms_per_step=(time.perf_counter()-wall)*1000/count))

    for member, driver in enumerate(drivers):
        advance(driver, member, "first_half", split)
        capture(baseline, member, driver, split)
        capture(pending, member, driver, split)
        before = time.perf_counter()
        receipt = Writer(settings=settings, driver=driver, outdir=out/f"checkpoint-{member}").finish(driver, {})
        path = out/f"checkpoint-{member}"/"restart.nc"
        checkpoints.append(dict(member_id=member, path=str(path), sha256=digest_file(path),
            wall_ms=(time.perf_counter()-before)*1000, receipt=receipt))
    saved = pending.save_resume()
    atomic_json(out / "collector-checkpoint.json", saved)
    resumed = collector(resume_root)
    before = time.perf_counter()
    resumed.restore_resume()
    collector_restore_ms = (time.perf_counter()-before)*1000
    initial_checks, restore_ms = [], []
    for member, original in enumerate(drivers):
        driver = build(member, resumed, 1)
        before = time.perf_counter()
        Writer(settings=settings, driver=driver, outdir=out/f"unused-{member}").restore(
            out/f"checkpoint-{member}"/"restart.nc", driver)
        restore_ms.append((time.perf_counter()-before)*1000)
        initial_checks.append(compare(original, driver))
        # The real history manager can emit its sealed checkpoint deadline once.
        capture(resumed, member, driver, split)
        restored.append(driver)

    final_checks, output_ms = [], []
    for member, (original, driver) in enumerate(zip(drivers, restored)):
        advance(original, member, "uninterrupted_second_half", steps-split)
        advance(driver, member, "resumed_second_half", steps-split)
        final_checks.append(compare(original, driver))
        output_ms.append(dict(member_id=member,
            baseline=capture(baseline, member, original, steps),
            resumed=capture(resumed, member, driver, steps)))

    products = []
    for owner in (baseline, resumed):
        before = time.perf_counter()
        files = owner.finish_run()
        products.append(dict(wall_ms=(time.perf_counter()-before)*1000, files=files))
    left = baseline.channel_depth_archive.snapshot()["samples"]
    right = resumed.channel_depth_archive.snapshot()["samples"]
    assert [(r["member_id"], r["valid_time"], r["source"]) for r in left] == [
        (r["member_id"], r["valid_time"], r["source"]) for r in right]
    product_checks = []
    for first, second in zip(baseline.nextgen_points.products[:-1], resumed.nextgen_points.products[:-1]):
        assert digest_file(first) == digest_file(second)
        product_checks.append(dict(baseline=first, resumed=second, sha256=digest_file(first)))
    member_files = []
    for member, driver in enumerate(restored):
        path = out/f"member-{member}.nc"
        write_member(path, driver)
        member_files.append(dict(member_id=member, path=str(path), sha256=digest_file(path)))
    receipt = dict(status="PASS", scope=__doc__, numerics=drivers[0].restart_identity["numerics"],
        card=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(), shared_pool=True,
        identical_inputs=True, weather_restart=False, requested_member_ids=[0, 1],
        steps=steps, split_step=split, interval_seconds=interval, setup=setup, timings=timings,
        checkpoints=checkpoints, collector_restore_ms=collector_restore_ms, driver_restore_ms=restore_ms,
        restored_state_checks=initial_checks, continued_state_checks=final_checks,
        depth_samples_per_run=len(left), depth_source_sha_equal=True,
        output_capture_ms=output_ms, products=products, probability_files=product_checks,
        member_files=member_files, resident_inventory=inventory.receipt(),
        parallelism="four independent live drivers; members serial; MC128 threads/block,272levels;19LP17subsets128threads;history/count/fraction256threads; CUDA byte comparisons at checkpoint/output boundaries",
        cuda_graphs=False, routing_step_host_field_work=False,
        wall_seconds=time.perf_counter()-started)
    atomic_json(out/"receipt.json", receipt)
    print(json.dumps({k:receipt[k] for k in ("status", "numerics", "steps", "split_step", "depth_samples_per_run", "depth_source_sha_equal", "timings", "driver_restore_ms", "wall_seconds")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.settings, args.out)
