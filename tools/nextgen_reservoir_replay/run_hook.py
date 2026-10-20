"""Real native leaf trajectory through the prepared MC reservoir level hook.

This checks composition on one source-owned reservoir, not full-network parity.
Raw payload read/write/hash/comparison is the fresh shared Rust replay tool.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("fixture", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--bridge", required=True)
    p.add_argument("--comparator", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--mc-revision", required=True)
    p.add_argument("--native-receipt", required=True)
    args = p.parse_args()
    import cupy as cp
    from numpy import int32
    from gpuwm.hydro.reservoirs import prepare, PARAMETERS, CONTROLS, SOURCE_PIN
    from gpuwm.hydro.troute_mc import MuskingumCunge, ReservoirLevel
    from gpuwm.core.ieee_module import compile_module
    from gpuwm.numerics import is_strict
    if not is_strict():
        raise ValueError("this native hook proof requires explicit GPUWM_NUMERICS=strict")
    spec = importlib.util.spec_from_file_location("raw_cuda", Path(__file__).with_name("raw_cuda.py"))
    raw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(raw)
    io = raw.RawCuda(args.bridge)
    args.output.mkdir(parents=True, exist_ok=False)
    payload_dir = args.output / "setup-payloads"
    payload_dir.mkdir()
    started = time.perf_counter()
    count = (args.fixture / "input.f32").stat().st_size // 56

    def load(name, shape):
        return io.load({"path": name, "dtype": "f32", "shape": shape,
                        "sha256": io.sha256(args.fixture/name)}, args.fixture)

    captured = load("input.f32", [14, count])
    arrays = {"parameters": load("parameters.f32", [9, 1]), "height": load("height.f32", [1])}
    for name in ("previous_inflow", "inflow", "lateral", "initial_height", "levelpool_release", "levelpool_height"):
        arrays[name] = cp.zeros(1, dtype="float32")
    arrays["types"] = cp.ones(1, dtype="int32")
    for name in ("indices", "dynamic_type", "active_source", "fault"):
        arrays[name] = cp.zeros(1, dtype="int32")
    arrays["offsets"] = cp.zeros(2, dtype="int64")
    arrays["controls"] = cp.zeros((8, 1), dtype="float64")
    arrays["carriers"] = cp.zeros((3, 1), dtype="float64")
    for name in ("clock", "release", "assimilated", "series", "times"):
        arrays[name] = cp.zeros(1, dtype="float64")
    # Seal every prepared input, including zero DA carriers initialized on GPU.
    # These harness payloads are setup data, not additional native DA captures.
    hashes = {}
    for name, value in arrays.items():
        io.store(value, payload_dir / name)
        hashes["reservoir_"+name] = io.sha256(payload_dir/name)
    declaration = dict(arrays={name: "reservoir_"+name for name in arrays},
                       source_pin=SOURCE_PIN, parameter_columns=PARAMETERS,
                       control_columns=CONTROLS, validated_csr=True,
                       reservoir_order_sha256="shipped-source-lake-4185105",
                       time_origin="shipped-test-first-inflow")
    source = SimpleNamespace(
        arrays={"reservoir_"+name: value for name, value in arrays.items()},
        manifest={"reservoirs": declaration},
        restart_identity={"source_sha256": io.sha256(args.fixture/"input.f32"),
                          "payload_sha256": hashes, "scope": "single-source-reservoir composition harness"})
    reservoirs = prepare(settings={}, options={"network_component": "fabric"}, state=None,
                         physics=None, device_id=cp.cuda.runtime.getDevice(), mode="standalone",
                         components={"fabric": source})
    # One reservoir node, no MC reaches. Source inflows enter as the native
    # boundary; the actual MC gather/scatter and reservoir subset run unchanged.
    router = MuskingumCunge(
        dt_seconds=300., parameters=cp.zeros((1, 8), dtype="float32"),
        state=cp.zeros((1, 3), dtype="float32"), lateral=cp.zeros(1, dtype="float32"),
        upstream_offsets=cp.zeros(2, dtype="int32"),
        upstream_indices=cp.empty(0, dtype="int32"), ordered_nodes=cp.empty(0, dtype="int32"),
        level_offsets=[0, 0], boundary_previous=cp.zeros(1, dtype="float32"),
        boundary_current=cp.zeros(1, dtype="float32"),
        reach_ids=cp.full(1, 4185105, dtype="int64"),
        identity=dict(network_sha256=source.restart_identity["source_sha256"],
                      parameters_sha256=hashes["reservoir_parameters"],
                      ordering_sha256="single-reservoir-no-MC-reaches",
                      time_origin="shipped-test-first-inflow"))
    rows = cp.zeros(1, dtype="int32")
    hook = ReservoirLevel(router, reservoirs, rows=rows, node_indices=rows,
                          selection_sha256="single-row-zero", mapping_sha256="single-node-zero",
                          validated_mapping=True)
    router.bind_level_hooks({0: [hook]})
    if not all("reservoir/"+name in router.arrays for name in arrays):
        raise AssertionError("MC checkpoint inventory omitted reservoir carriers")
    helpers = compile_module(r'''
extern "C" __global__ void feed(int n,const float *input,const double *clock,float *prev,float *now) {
 if(threadIdx.x==0) {int j=int(clock[0]/300.);prev[0]=now[0]=input[n+j];}
}
extern "C" __global__ void record(int n,const double *clock,const float *published,float *output) {
 if(threadIdx.x==0) {int j=int(clock[0]/300.)-1;output[j]=published[2];output[n+j]=published[0];}
}
''', ("--std=c++17", "--ftz=false"), "nextgen_reservoir_mc_hook_fixture")
    feed, record = helpers.get_function("feed"), helpers.get_function("record")
    output = cp.empty((2, count), dtype="float32")
    cp.cuda.get_current_stream().synchronize()
    setup_ms = (time.perf_counter()-started)*1e3
    started = time.perf_counter()
    for _ in range(count):
        feed((1,), (1,), (int32(count), captured, arrays["clock"],
                          router.arrays["boundary_previous"], router.arrays["boundary_current"]))
        router.launch(300.)
        record((1,), (1,), (int32(count), arrays["clock"], router.arrays["published"], output))
    cp.cuda.get_current_stream().synchronize()
    trajectory_ms = (time.perf_counter()-started)*1e3
    io.store(output, args.output/"trajectory.f32")
    request = dict(schema="nextgen.raw.compare/v1", unit="reservoirs",
                   case="t-route-shipped-Pocono-4185105-287x300s-MC-hook",
                   numerics="strict", candidate_revision=args.revision,
                   command=["run_hook.py", str(args.fixture), str(args.output)],
                   provenance={"origin": "native-module", "sources": [
                       dict(name="NOAA-OWP/t-route", revision=SOURCE_PIN, licence="Apache-2.0")],
                       "evidence": [dict(path=args.native_receipt, sha256=io.sha256(args.native_receipt))]},
                   fields=[dict(name="height_m/outflow_m3_per_s", dtype="f32", shape=[2, count], order="c",
                                expected=dict(path=str(args.fixture/"expected.f32"), sha256=io.sha256(args.fixture/"expected.f32")),
                                actual=str(args.output/"trajectory.f32"))])
    request_path = args.output/"request.json"
    request_path.write_text(json.dumps(request, indent=2)+"\n")
    result = subprocess.run([args.comparator, str(request_path), str(args.output/"verdict.json")], capture_output=True, text=True)
    receipt = dict(candidate_revision=args.revision, mc_revision=args.mc_revision, source_pin=SOURCE_PIN,
                   scope="one source-owned native reservoir trajectory through real prepared factory and MC hook; not full network or DA acceptance",
                   card=cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())["name"].decode(),
                   calls=count, compared_words=2*count, bitwise_match=result.returncode == 0,
                   comparison_output=result.stdout+result.stderr,
                   cuda_graphs=False, setup_ms=setup_ms, strict_trajectory_ms=trajectory_ms,
                   parallelism="one reservoir row in 128-thread block; zero MC reach threads; GPU feed/record one thread",
                   timing_scope="eight direct launches/update: feed,snapshot,gather,levelpool,assimilation_commit,scatter,publish,record; final sync; no production speed claim",
                   expected_sha256=io.sha256(args.fixture/"expected.f32"),
                   trajectory_sha256=io.sha256(args.output/"trajectory.f32"),
                   checkpoint_arrays=list(router.arrays), identity=router.restart_identity)
    (args.output/"receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2))
    if result.returncode:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
