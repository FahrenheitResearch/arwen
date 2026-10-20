"""Two independent resident routing members on captured inputs.

This is an identical-input control, not a meteorological ensemble. Settings
select the owners' real factories; Python handles metadata and CUDA launches.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone

import cupy as cp

from gpuwm.ensemble.nextgen_capture import NextGenCollector
from gpuwm.ensemble.runtime_context import MemberOutputCapture, current_capture, member_output_scope
from gpuwm.ensemble.restart_roster import atomic_json
from gpuwm.hydro.assembly import prepare
from gpuwm.io.classic_product import ClassicProduct
from gpuwm.netcdf_bridge import Dataset


def write_member(path, driver):
    history = driver.components["history"]
    channels = driver.channel_output()
    with ClassicProduct(path) as file:
        file.setncattr("scope", "identical captured-input routing control; no atmosphere or ensemble spread")
        file.createDimension("time", history.capacity)
        file.createDimension("feature", history.reaches)
        file.createDimension("channel", channels["reach_id"].size)
        file.createDimension("word", 2)
        for name, value, dims, units in (
                ("reach_id", history.arrays["reach_id"], ("feature",), "1"),
                ("reach_id_words", history.arrays["reach_id"].view(cp.uint32).reshape(-1, 2), ("feature", "word"), "1"),
                ("discharge_m3_s", history.arrays["flow"], ("time", "feature"), "m3 s-1"),
                ("channel_reach_id", channels["reach_id"], ("channel",), "1"),
                ("channel_reach_id_words", channels["reach_id"].view(cp.uint32).reshape(-1, 2), ("channel", "word"), "1"),
                ("channel_depth_m", channels["channel_depth_m"], ("channel",), "m")):
            variable = file.createVariable(name, value.dtype, dims)
            variable.setncattr("units", units)
            variable[:] = cp.asnumpy(value)  # Boundary transport to Rust, no numeric Python processing.


def run(settings_path, out):
    settings = json.loads(settings_path.read_text())
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "settings.json", settings)
    original = SimpleNamespace(root=out, submit=lambda **kw: None,
                               require_complete=lambda: None, finish_run=lambda: [])
    channels_seen = []
    def depth_sink(**sample):
        # Borrow native MC-only storage; the owner excludes reservoir elevations.
        channels_seen.append(sample)
    collector = NextGenCollector(original, (0, 1), array_module=cp, channel_depth_sink=depth_sink)
    states, preparation, timelines = [], [], []
    # Both factories are called while both states remain live. Inventory must
    # reject sharing even when separate factory objects alias writable storage.
    for member in (0, 1):
        started = time.perf_counter()
        driver = prepare(settings=settings, state=None, physics=None, device_id=0, mode="standalone")
        state = SimpleNamespace(physics=SimpleNamespace(hydro=driver))
        collector.bind_member(member_id=member, state=state)
        states.append(state)
        preparation.append((time.perf_counter() - started) * 1000)
    interval = settings["interval_seconds"]
    steps = states[0].physics.hydro.components["history"].capacity_steps
    cp.cuda.get_current_stream().synchronize()
    for member, state in enumerate(states):
        driver = state.physics.hydro
        begin, end = cp.cuda.Event(), cp.cuda.Event()
        begin.record()
        wall = time.perf_counter()
        for _ in range(steps):
            driver.before_lsm()
            driver.advance(interval)
        end.record()
        end.synchronize()
        timelines.append({"member_id": member, "routing_history_cuda_ms_per_step":
            cp.cuda.get_elapsed_time(begin, end) / steps,
            "routing_history_wall_ms_per_step": (time.perf_counter() - wall) * 1000 / steps})
    a, b = (state.physics.hydro for state in states)
    qa, qb = (driver.components["history"].arrays["flow"] for driver in (a, b))
    assert bool(cp.array_equal(qa.view(cp.uint32), qb.view(cp.uint32)))
    # A real write in member zero must not alter any bit of member one's state.
    first, second = (driver.components["routing"].arrays["state"] for driver in (a, b))
    saved_first, saved_second = first.copy(), second.copy()
    first[0, 0] += cp.float32(1)
    assert bool(cp.array_equal(second.view(cp.uint32), saved_second.view(cp.uint32)))
    cp.copyto(first, saved_first)
    origin = datetime.fromisoformat(settings["time_origin"].replace("Z", "+00:00"))
    valid_time = (origin + timedelta(seconds=steps * interval)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.perf_counter()
    for member, state in enumerate(states):
        with member_output_scope(MemberOutputCapture(collector.submit, member)):
            current_capture().submit(state=state, valid_time=valid_time, grid_id=1,
                                     streamed=None, metadata={}, refl_field=None)
    assert [row["member_id"] for row in channels_seen] == [0, 1]
    for row, state in zip(channels_seen, states):
        sample = state.physics.hydro.channel_output()
        assert row["feature_id"] is sample["reach_id"]
        assert row["channel_depth_m"] is sample["channel_depth_m"]
    products = collector.finish_run()
    publication_ms = (time.perf_counter() - started) * 1000
    with Dataset(products[0]) as source:
        source.set_auto_maskandscale(False)
        count = cp.asarray(source.variables["exceedance_count"][:], dtype=cp.uint32)
        finite = cp.asarray(source.variables["finite_count"][:], dtype=cp.uint32)
        probability = cp.asarray(source.variables["probability"][:], dtype=cp.float32)
    q = a.reach_output()["discharge_m3_s"]
    thresholds = cp.asarray(collector.nextgen_points.options["thresholds_m3_s"], dtype=cp.float64).astype(cp.float32)
    expected_count = (q[None, :] >= thresholds[:, None]).astype(cp.uint32) * 2
    expected_finite = cp.isfinite(q).astype(cp.uint32) * 2
    expected_probability = cp.where(expected_finite[None, :] == 2,
                                    expected_count.astype(cp.float32) / 2, cp.float32("nan"))
    assert bool(cp.array_equal(count, expected_count))
    assert bool(cp.array_equal(finite, expected_finite))
    assert bool(cp.array_equal(probability, expected_probability, equal_nan=True))
    for member, state in enumerate(states):
        write_member(out / f"member-{member}.nc", state.physics.hydro)
    # Choose a visible trace on CUDA at the output boundary; only its index
    # crosses to metadata. No Python numerical decoding of history values.
    trace_index = int(cp.argmax(cp.max(qa, axis=0) - cp.min(qa, axis=0)).item())
    receipt = {"pass": True, "scope": "real captured-input routing; two independent identical controls",
        "spread": "none: identical captured initial and lateral inputs; not a meteorological ensemble",
        "card": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(), "shared_pool": True,
        "numerics": a.restart_identity["numerics"], "requested_member_ids": [0, 1],
        "reach_axis": {"file": "member-0.nc", "variable": "reach_id",
                       "exact_transport_words": "reach_id_words", "size": int(qa.shape[1]),
                       "semantics": "native output IDs; distinct from internal collapsed routing IDs"},
        "origin": settings["time_origin"], "interval_seconds": interval,
        "suggested_hydrograph_reach_id": int(a.reach_output()["reach_id"][trace_index].item()),
        "steps": steps, "valid_time": valid_time, "prepare_ms": preparation, "timelines": timelines,
        "output_capture_merge_rust_publish_ms": publication_ms,
        "parallelism": "members serial on one card; MC 128 threads/block, dependency levels; reservoir subsets; history/reduction 256 threads/block",
        "cuda_graphs": False, "routing_step_host_data_work": False,
        "state_isolation": "both live inventories disjoint; actual member0 write leaves member1 words unchanged",
        "history_control_comparison": {"differing_words": 0, "words": qa.size},
        "probability_check": "all counts/finite counts/fractions equal direct CUDA comparison of actual final discharge",
        "resident_inventory": collector.resident_inventory.receipt(),
        "products": products, "contract": collector.nextgen_points.published_buffers(),
        "driver_identity_sha256": [hashlib.sha256(json.dumps(driver.restart_identity, sort_keys=True).encode()).hexdigest() for driver in (a, b)],
        "native_ensemble_replay": "not yet compared against native history; routing owners own that replay",
        "capture_api": "root history submit -> current_capture().submit_hydro; once per member/deadline, never in routing step",
        "driver_channel_provider": "setup-cached driver.channel_output; both collector callbacks borrow exact MC-only IDs/depth storage",
        "channel_reaches": [int(row["feature_id"].size) for row in channels_seen]}
    atomic_json(out / "receipt.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("pass", "scope", "card", "numerics", "timelines", "output_capture_merge_rust_publish_ms", "history_control_comparison")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.settings, args.out)
