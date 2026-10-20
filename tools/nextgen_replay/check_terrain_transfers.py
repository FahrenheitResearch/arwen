"""Check live nonzero terrain volume carriers and immediate same-stream consumers.

SPDX-License-Identifier: Apache-2.0
Python handles metadata/launches only; Rust raw I/O and comparison, CUDA fields.
Analytical one-cell cases are not physical-event or native replay evidence.
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
    from gpuwm.nextgen import terrain

    args.out.mkdir(exist_ok=False)
    raw = RawCuda(args.library)
    fixture = json.loads(args.fixture.read_text())
    if fixture["schema"] == "nextgen.terrain-transfer.fixture/v2":
        return run_intervals(args, raw, fixture, cp, terrain, numerics)
    if fixture["schema"] != "nextgen.terrain-transfer.fixture/v1":
        raise ValueError("wrong fixture schema")
    stream = cp.cuda.Stream(non_blocking=True)
    volumes = ("surface_outlet_volume", "groundwater_outlet_volume", "exfiltration_volume")
    consume = cp.ElementwiseKernel(
        "float64 s, float64 g, float64 e", "float64 rs, float64 rg, float64 re",
        "rs=s; rg=g; re=e", "nextgen_verify_immediate_transfer")
    prepared = []
    started = time.perf_counter()
    for case in fixture["cases"]:
        inputs = raw.load_many(case["inputs"], stream=stream)
        with stream:
            provider = SimpleNamespace(arrays=inputs, restart_identity={
                "fixture": {"path": str(args.fixture), "sha256": raw.sha256(args.fixture)},
                "case": case["name"], "origin": fixture["origin"],
            })
            routing = terrain.prepare(settings={}, state=None, physics=None,
                device_id=cp.cuda.runtime.getDevice(), mode="standalone",
                components={"inputs": provider}, options={
                    "dx": fixture["dx"], "dt": fixture["dt"],
                    "period_seconds": fixture["period_seconds"],
                    "identity": {"case": case["name"], "terrain_revision": args.terrain_revision},
                    "fields": {name: {"component": "inputs", "array": name} for name in inputs},
                })
            receivers = {name: cp.empty_like(inputs[name]) for name in volumes}
            consume(*(inputs[name] for name in volumes), *(receivers[name] for name in volumes),
                    block_size=128)  # Compile consumer before the measured trajectory.
            routing._flow.compile()
            routing._update.compile()
        if any(routing.arrays[name] is not inputs[name] for name in volumes):
            raise AssertionError("factory replaced caller-owned cumulative buffers")
        prepared.append((case, routing, receivers))
    stream.synchronize()
    setup_ms = (time.perf_counter() - started) * 1000

    def inventory():
        return {case["name"]: {
            "terrain": {name: a.data.ptr for name, a in routing.arrays.items()},
            "flux": routing._flux.data.ptr,
            "receivers": {name: a.data.ptr for name, a in receivers.items()},
        } for case, routing, receivers in prepared}

    pointers = inventory()
    started = time.perf_counter()
    for case, routing, receivers in prepared:
        routing.advance(fixture["period_seconds"], stream=stream)
        # This downstream device consumer runs before any host observation/sync.
        consume(*(routing.arrays[name] for name in volumes),
                *(receivers[name] for name in volumes), stream=stream, block_size=128)
    stream.synchronize()
    batch_ms = (time.perf_counter() - started) * 1000
    if inventory() != pointers:
        raise AssertionError("resident pointers moved")
    fields = []
    exports = []
    for case, routing, receivers in prepared:
        for name, spec in case["expected"].items():
            actual = args.out / f'{case["name"]}-{name}.raw'
            array = routing.arrays[name]
            if name == "health":
                array = array.view("uint8")
            exports.append((array, actual))
            fields.append(dict(name=f'{case["name"]}/{name}', dtype=spec["dtype"],
                shape=spec["shape"], order=spec["order"],
                expected={key: spec[key] for key in ("path", "sha256")}, actual=str(actual)))
        for name in volumes:
            spec = case["expected"][name]
            actual = args.out / f'{case["name"]}-received-{name}.raw'
            exports.append((receivers[name], actual))
            fields.append(dict(name=f'{case["name"]}/received/{name}', dtype=spec["dtype"],
                shape=spec["shape"], order=spec["order"],
                expected={key: spec[key] for key in ("path", "sha256")}, actual=str(actual)))
    raw.store_many(exports, stream=stream)
    request = dict(schema="nextgen.raw.compare/v1", unit="terrain-transfer-carriers",
        case="three-one-cell-nonzero-transfers", numerics=numerics.mode(),
        candidate_revision=args.revision, command=["check_terrain_transfers.py", *args.argv],
        provenance={"origin": fixture["origin"], "sources": [
            {"name": "original terrain", "revision": args.terrain_revision, "licence": "Apache-2.0"},
            {"name": "original verification", "revision": args.revision, "licence": "Apache-2.0"}],
            "evidence": [{"path": str(args.fixture), "sha256": raw.sha256(args.fixture)}]},
        fields=fields)
    (args.out / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    result = subprocess.run([str(args.comparator), str(args.out / "request.json"),
                             str(args.out / "verdict.json")])
    card = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())["name"]
    if isinstance(card, bytes):
        card = card.decode()
    code_paths = [Path(__file__), Path(__file__).with_name("raw_cuda.py"), Path(terrain.__file__),
                  Path(terrain.__file__).parents[1] / "core/kernels/nextgen_terrain.cu"]
    execution = dict(schema="nextgen.terrain-transfer.execution/v1", card=card,
        shared_gpu_pool=True, numerics=numerics.mode(), candidate_revision=args.revision,
        terrain_revision=args.terrain_revision, source_code=[
            {"path": str(p), "sha256": raw.sha256(p)} for p in code_paths],
        fixture={"path": str(args.fixture), "sha256": raw.sha256(args.fixture)},
        setup_ms=setup_ms, completed_three_case_batch_ms=batch_ms,
        pointer_inventory=pointers, stable_pointers=True, supplied_cumulative_buffers_preserved=True,
        immediate_same_stream_consumers=True, period_seconds=fixture["period_seconds"],
        parallelism="three independent 1x1 grids; each two terrain launches1blockx128threads plus one consumer1blockx128threads",
        terrain_launches=6, consumer_launches=3, warmup_updates=0,
        per_advance_syncs=0, per_advance_events=0, per_advance_field_readbacks=0,
        cuda_graphs=False, comparator_exit=result.returncode,
        scope="Analytical live transfer and cumulative-buffer contract only; not physical-event/native replay/whole-step speed acceptance")
    (args.out / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
    print(json.dumps(execution, indent=2))
    if result.returncode:
        raise SystemExit(result.returncode)


def run_intervals(args, raw, fixture, cp, terrain, numerics):
    stream = cp.cuda.Stream(non_blocking=True)
    volumes = ("surface_outlet_volume", "groundwater_outlet_volume", "exfiltration_volume",
               "surface_interval_volume", "groundwater_interval_volume")
    consume = cp.ElementwiseKernel(
        "float64 s,float64 g,float64 e,float64 si,float64 gi",
        "float64 rs,float64 rg,float64 re,float64 rsi,float64 rgi",
        "rs=s;rg=g;re=e;rsi=si;rgi=gi", "nextgen_verify_interval_consumer")
    common = dict(settings={}, state=None, physics=None,
                  device_id=cp.cuda.runtime.getDevice(), mode="standalone")
    started = time.perf_counter()
    mapping = SimpleNamespace(arrays=raw.load_many(fixture["mapping"], stream=stream),
                              restart_identity={"origin": "sealed analytical three-row CSR"})

    def lateral_for(source, inverse):
        return terrain.prepare_lateral(**common, components={"terrain": source, "map": inverse},
            options={"terrain_component": "terrain", "mapping_component": "map",
                     "offsets_array": "offsets", "indices_array": "indices",
                     "identity": {"fixture_sha256": raw.sha256(args.fixture)}})

    cases = []
    exports = []
    fields = []

    def export(name, spec, array):
        path = args.out / (name.replace("/", "-") + ".raw")
        exports.append((array, path))
        fields.append(dict(name=name, dtype=spec["dtype"], shape=spec["shape"], order=spec["order"],
            expected={key: spec[key] for key in ("path", "sha256")}, actual=str(path)))

    for case in fixture["cases"]:
        inputs = raw.load_many(case["inputs"], stream=stream)
        provider = SimpleNamespace(arrays=inputs, restart_identity={"case": case["name"],
            "fixture_sha256": raw.sha256(args.fixture)})
        with stream:
            routing = terrain.prepare(**common, components={"inputs": provider}, options={
                "dx": fixture["dx"], "dt": fixture["dt"], "period_seconds": fixture["period_seconds"],
                "identity": {"origin": fixture["origin"], "case": case["name"]},
                "fields": {name: {"component": "inputs", "array": name} for name in inputs}})
            lateral = lateral_for(routing, mapping)
            observations = []
            for step, expected in enumerate(case["observations"]):
                copies = []
                for name, spec in expected.items():
                    source = (lateral.arrays["health"] if name == "lateral_health" else
                              lateral.arrays[name] if name == "terrain_lateral" else routing.arrays[name])
                    if name in ("health", "lateral_health"):
                        source = source.view("uint8")
                    target = cp.empty_like(source)
                    copies.append((target.data.ptr, source.data.ptr, source.nbytes))
                    export(f'{case["name"]}/step{step}/{name}', spec, target)
                receivers = {name: cp.empty_like(inputs[name]) for name in volumes}
                for name in volumes:
                    export(f'{case["name"]}/step{step}/received/{name}', expected[name], receivers[name])
                observations.append((receivers, copies))
            consume(*(inputs[name] for name in volumes),
                    *(observations[0][0][name] for name in volumes), block_size=128)
            routing._flow.compile()
            routing._update.compile()
            lateral._reduce.compile()
        if any(routing.arrays[name] is not inputs[name] for name in volumes):
            raise AssertionError("factory replaced caller-owned cumulative/interval buffers")
        cases.append((case, routing, lateral, observations))

    witness_spec = fixture["ordered_witness"]
    witness_arrays = raw.load_many(witness_spec["inputs"], stream=stream)
    with stream:
        witness_source = SimpleNamespace(depth_m=cp.empty((1, 3), dtype="float32"),
            arrays=witness_arrays, _period=1., restart_identity={"origin": "analytical ordered-sum witness"})
        witness_mapping = SimpleNamespace(arrays=witness_arrays, restart_identity={"origin": "sealed ascending CSR"})
        witness = lateral_for(witness_source, witness_mapping)
        export("ordered-witness/terrain_lateral", witness_spec["expected"], witness.terrain_lateral)
        export("ordered-witness/health", witness_spec["healthy"], witness.health.view("uint8"))
    stream.synchronize()
    setup_ms = (time.perf_counter() - started) * 1000

    def inventory():
        return dict(terrain={case["name"]: {name: a.data.ptr for name, a in routing.arrays.items()}
                    for case, routing, lateral, observations in cases},
            flux={case["name"]: routing._flux.data.ptr for case, routing, lateral, observations in cases},
            lateral={case["name"]: {name: a.data.ptr for name, a in lateral.arrays.items()}
                     for case, routing, lateral, observations in cases},
            exports={str(path): a.data.ptr for a, path in exports},
            witness={name: a.data.ptr for name, a in witness.arrays.items()})

    pointers = inventory()
    started = time.perf_counter()
    for case, routing, lateral, observations in cases:
        for receivers, copies in observations:
            routing.advance(fixture["period_seconds"], stream=stream)
            lateral.advance(fixture["period_seconds"], stream=stream)
            consume(*(routing.arrays[name] for name in volumes),
                    *(receivers[name] for name in volumes), stream=stream, block_size=128)
            for dst, src, size in copies:
                cp.cuda.runtime.memcpyAsync(dst, src, size, cp.cuda.runtime.memcpyDeviceToDevice, stream.ptr)
    witness.advance(1., stream=stream)
    stream.synchronize()
    batch_ms = (time.perf_counter() - started) * 1000
    if pointers != inventory():
        raise AssertionError("resident pointers moved")
    raw.store_many(exports, stream=stream)
    request = dict(schema="nextgen.raw.compare/v1", unit="terrain-interval-carriers",
        case="two-advances-and-ordered-csr", numerics=numerics.mode(), candidate_revision=args.revision,
        command=["check_terrain_transfers.py", *args.argv], provenance={"origin": fixture["origin"],
            "sources": [{"name": "original terrain", "revision": args.terrain_revision, "licence": "Apache-2.0"},
                        {"name": "original verification", "revision": args.revision, "licence": "Apache-2.0"}],
            "evidence": [{"path": str(args.fixture), "sha256": raw.sha256(args.fixture)}]}, fields=fields)
    (args.out / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    result = subprocess.run([str(args.comparator), str(args.out / "request.json"), str(args.out / "verdict.json")])
    code_paths = [Path(__file__), Path(__file__).with_name("raw_cuda.py"), Path(terrain.__file__),
                  Path(terrain.__file__).parents[1] / "core/kernels/nextgen_terrain.cu"]
    card = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())["name"]
    if isinstance(card, bytes):
        card = card.decode()
    execution = dict(schema="nextgen.terrain-transfer.execution/v2", card=card, shared_gpu_pool=True,
        numerics=numerics.mode(), candidate_revision=args.revision, terrain_revision=args.terrain_revision,
        fixture={"path": str(args.fixture), "sha256": raw.sha256(args.fixture)},
        source_code=[{"path": str(p), "sha256": raw.sha256(p)} for p in code_paths],
        setup_ms=setup_ms, completed_three_case_and_witness_batch_ms=batch_ms,
        stable_pointers=True, supplied_cumulative_and_interval_buffers_preserved=True,
        immediate_same_stream_consumers=True, observations_per_case=2, warmup_updates=0,
        terrain_launches=12, lateral_launches=7, consumer_launches=6,
        history_d2d_copies=sum(len(copies) for case, routing, lateral, observations in cases
                              for receivers, copies in observations),
        parallelism="three independent1x1 terrain grids and three-row lateral outputs; ordered witness1x3; all kernels1blockx128threads",
        per_advance_syncs=0, per_advance_events=0, per_advance_field_readbacks=0,
        cuda_graphs=False, comparator_exit=result.returncode, pointer_inventory=pointers,
        scope=fixture["scope"])
    (args.out / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
    print(json.dumps(execution, indent=2))
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "library", "comparator", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("revision", "terrain-revision"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    args.argv = sys.argv[1:]
    run(args)
