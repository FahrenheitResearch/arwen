"""Run the published terrain and inundation factories on sealed engineering inputs.

SPDX-License-Identifier: Apache-2.0
Python orchestrates only. CUDA carries fields/history; Rust prepares expectations
and compares exports. This imposed-water experiment is not a flood-event run.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from raw_cuda import RawCuda


def run(args):
    import cupy as cp
    from gpuwm import numerics
    from gpuwm.nextgen import terrain, inundation

    source = json.loads(args.input.read_text())
    manifest = json.loads(Path(source["dem_manifest"]["path"]).read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    raw = RawCuda(args.library)
    # Verify metadata seals before interpreting any input shape/control.
    for name in ("dem_manifest", "terrain_receipt"):
        if raw.sha256(source[name]["path"]) != source[name]["sha256"]:
            raise ValueError(f"incorrect {name} seal")
    receipt = json.loads(Path(source["terrain_receipt"]["path"]).read_text())
    params = receipt["parameters"]
    period, dt = receipt["prepared_period_s"], receipt["substep_s"]
    periods = receipt["measured_periods"]
    shape = source["shape"]
    if periods < 1 or shape != [manifest["grid"]["ny"], manifest["grid"]["nx"]]:
        raise ValueError("invalid engineering duration/grid")
    specs = {
        "surface_depth": dict(source["initial_depth"], dtype="f32", shape=shape, order="c"),
        "active": dict(source["active"], dtype="i32", shape=shape, order="c"),
        "elevation": dict(path=str(args.elevation), sha256=manifest["arrays"]["elevation_m"]["sha256"],
                          dtype="f32", shape=shape, order="c"),
    }
    stream = cp.cuda.Stream(non_blocking=True)
    started = time.perf_counter()
    arrays = raw.load_many(specs, stream=stream)
    with stream:
        for name, parameter in {
            "groundwater_storage": "groundwater_initial_m", "soil_depth": "soil_depth_m",
            "specific_yield": "specific_yield", "manning": "Manning_s_m-1_3",
            "conductivity": "conductivity_m_s", "surface_source": "surface_excess_m_s",
            "recharge": "recharge_m_s", "outlet_width": "outlet_width_m",
        }.items():
            arrays[name] = cp.full(shape, params[parameter], dtype="float32")
        arrays["outlet_stage"] = cp.zeros(shape, dtype="float32")
        provider = SimpleNamespace(arrays=arrays, restart_identity={
            "engineering_inputs": source, "terrain_revision": args.terrain_revision,
            "forcing": "imposed uniform excess; not physical event", "grid": manifest["grid"],
        })
        common = dict(settings={}, state=None, physics=None,
                      device_id=cp.cuda.runtime.getDevice(), mode="standalone")
        routing = terrain.prepare(**common, components={"inputs": provider}, options={
            "dx": manifest["grid"]["dx"], "dt": dt, "period_seconds": period,
            "identity": {"origin": "live engineering contract check"},
            "fields": {name: {"component": "inputs", "array": name} for name in arrays},
        })
        diagnostic = inundation.prepare(**common, components={"terrain": routing},
                                        options={"terrain_component": "terrain"})
        history = cp.empty((periods, *shape), dtype="float32")
        frames = [history[index] for index in range(periods)]
        copy = cp.ElementwiseKernel("float32 x", "float32 y", "y=x", "nextgen_verify_snapshot")
        copy(arrays["surface_depth"], frames[0], block_size=256)  # Compile outside timing.
    stream.synchronize()
    setup_ms = (time.perf_counter() - started) * 1000
    pointers = {group: {name: a.data.ptr for name, a in item.arrays.items()}
                for group, item in (("terrain", routing), ("inundation", diagnostic))}
    started = time.perf_counter()
    for frame in frames:
        routing.advance(period, stream=stream)
        diagnostic.advance(period, stream=stream)
        copy(arrays["surface_depth"], frame, stream=stream, block_size=256)
    stream.synchronize()
    batch_ms = (time.perf_counter() - started) * 1000
    stable = pointers == {group: {name: a.data.ptr for name, a in item.arrays.items()}
                          for group, item in (("terrain", routing), ("inundation", diagnostic))}
    if not stable:
        raise AssertionError("resident pointers moved")
    exports = {"history": history, "depth_m": diagnostic.arrays["depth_m"],
               "maximum_depth_m": diagnostic.arrays["maximum_depth_m"],
               "health": diagnostic.arrays["health"].view("uint8"),
               "terrain_health": routing.arrays["health"].view("uint8")}
    raw.store_many([(a, args.out / f"{name}.raw") for name, a in exports.items()], stream=stream)
    expectation_input = dict(source, history={"frames": periods, "depth": {
        "path": str(args.out / "history.raw"), "sha256": raw.sha256(args.out / "history.raw")}})
    # The Rust preparer checks the history's last frame against the independent
    # terrain-owner export and computes maxima over initial plus every period.
    expectation_input["source_identity"] = routing.restart_identity
    (args.out / "expectation-input.json").write_text(json.dumps(expectation_input, indent=2) + "\n")
    subprocess.run([str(args.preparer), str(args.out / "expectation-input.json"),
                    str(args.out / "fixture")], check=True)
    fixture_path = args.out / "fixture/fixture.json"
    fixture = json.loads(fixture_path.read_text())
    comparisons = []
    for name, key in (("depth_m", "expected_depth"), ("maximum_depth_m", "expected_maximum"),
                      ("health", "healthy"), ("terrain_health", "healthy")):
        spec = fixture["arrays"][key]
        comparisons.append(dict(name=name, dtype=spec["dtype"], shape=spec["shape"], order="c",
            expected={"path": spec["path"], "sha256": spec["sha256"]},
            actual=str(args.out / f"{name}.raw")))
    code_paths = [Path(terrain.__file__), Path(inundation.__file__),
                  Path(terrain.__file__).parents[1] / "core/kernels/nextgen_terrain.cu",
                  Path(inundation.__file__).parents[1] / "core/kernels/nextgen_inundation.cu"]
    request = dict(schema="nextgen.raw.compare/v1", unit="terrain-inundation-live",
        case="real-dem-imposed-water-2400s", numerics=numerics.mode(), candidate_revision=args.revision,
        command=["check_live_terrain.py", *args.argv],
        provenance={"origin": "engineering-live-resident-contract", "sources": [
            {"name": "original terrain", "revision": args.terrain_revision, "licence": "Apache-2.0"},
            {"name": "original diagnostic", "revision": args.revision, "licence": "Apache-2.0"}],
            "evidence": [{"path": str(fixture_path), "sha256": raw.sha256(fixture_path)}]},
        fields=comparisons)
    (args.out / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    subprocess.run([str(args.comparator), str(args.out / "request.json"),
                    str(args.out / "verdict.json")], check=True)
    card = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())["name"]
    if isinstance(card, bytes):
        card = card.decode()
    execution = dict(schema="nextgen.terrain-inundation-live/v1", candidate_revision=args.revision,
        terrain_revision=args.terrain_revision, card=card, numerics=numerics.mode(),
        source_code=[{"path": str(p), "sha256": raw.sha256(p)} for p in code_paths],
        grid_shape=shape, stable_pointers=stable, live_terrain_health=True, factories_executed=True,
        periods=periods, elapsed_seconds=periods * period, observations=1 + periods,
        setup_ms=setup_ms, completed_batch_ms=batch_ms, completed_period_ms=batch_ms / periods,
        parallelism={"cells": shape[0] * shape[1], "terrain_threads_per_block": 128,
                     "diagnostic_history_threads_per_block": 256,
                     "terrain_grids_per_period": 2 * round(period / dt),
                     "diagnostic_history_grids_per_period": 2},
        history_launches=periods, history_scope="preallocated CUDA snapshots at 300s output cadence",
        per_period_syncs=0, per_period_events=0, cuda_graphs=False, shared_gpu_pool=True,
        boundary="sealed rectangle; no channel backwater",
        terrain_identity=routing.restart_identity, diagnostic_identity=diagnostic.restart_identity,
        scope="Live terrain plus diagnostic and history engineering integration only; no event, observation skill, native reference or whole-hydro speed acceptance.")
    (args.out / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
    print(json.dumps(execution, indent=2))


if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "elevation", "library", "preparer", "comparator", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("revision", "terrain-revision"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    args.argv = sys.argv[1:]
    run(args)
