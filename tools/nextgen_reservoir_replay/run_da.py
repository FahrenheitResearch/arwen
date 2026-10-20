"""Offline native-call replay. Python sequences Rust I/O and CUDA launches.
SPDX-License-Identifier: Apache-2.0
"""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--comparator", required=True)
    parser.add_argument("--native-receipt", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    import cupy as cp
    from numpy import int32, float64, uint64
    from raw_cuda import RawCuda
    from gpuwm.hydro.reservoirs import kernel_source, SOURCE_PIN
    from gpuwm.core.ieee_module import compile_module
    io = RawCuda(args.bridge)
    args.output.mkdir()
    metadata = json.loads((args.fixture / "manifest.json").read_text())
    n, columns = metadata["calls"], metadata["output_columns"]

    def load(name, dtype, shape):
        return io.load(dict(path=name, dtype=dtype, shape=shape,
                            sha256=metadata["files"][name]["sha256"]), args.fixture)

    source = kernel_source(True) + Path(__file__).with_name("da_replay.cu").read_text()
    start = time.perf_counter()
    module = compile_module(source, ("--std=c++17", "--ftz=false", "--fmad=false",
                                    "--prec-div=true", "--prec-sqrt=true"), "nextgen_reservoir_da_replay")
    kernel = module.get_function("ngen_reservoir_da_replay")
    values = load("input.f64", "f64", [18, n])
    offsets = load("offsets.i64", "i64", [n+1])
    series = load("series.f64", "f64", [metadata["files"]["series.f64"]["bytes"]//8])
    times = load("times.f64", "f64", [metadata["files"]["times.f64"]["bytes"]//8])
    actual = cp.empty((columns, n), dtype="float64")
    setup_ms = (time.perf_counter()-start)*1000
    events = [cp.cuda.Event(), cp.cuda.Event()]
    events[0].record()
    kernel(((n+127)//128,), (128,), (int32(n), int32(4 if metadata["kind"] == "rfc" else 2),
                                   values, offsets, series, times, actual))
    events[1].record()
    events[1].synchronize()
    io.store(actual, args.output / "actual.f64")
    request = dict(schema="nextgen.raw.compare/v1", unit="reservoirs", case="actual-native-"+metadata["kind"]+"-calls",
                   numerics="strict", candidate_revision=args.revision,
                   command=["run_da.py", str(args.fixture), str(args.output)],
                   provenance=dict(origin="native-module", sources=[dict(name="NOAA-OWP/t-route", revision=SOURCE_PIN, licence="Apache-2.0")],
                                   evidence=[dict(path=str(args.native_receipt), sha256=io.sha256(args.native_receipt))]),
                   fields=[dict(name="native_return_fields", dtype="f64", shape=[columns,n], order="c",
                                expected=dict(path=str(args.fixture / "expected.f64"), sha256=metadata["files"]["expected.f64"]["sha256"]),
                                actual=str(args.output / "actual.f64"))])
    path = args.output / "request.json"
    path.write_text(json.dumps(request, indent=2)+"\n")
    command = [args.comparator, str(path), str(args.output / "verdict.json")]
    result = subprocess.run(command, capture_output=True, text=True)
    # Exercise the production global assimilation/commit kernel as well as
    # its inline function. Initial values and LP results are real captures.
    a = {}
    for name, shape, dtype in (("parameters",(9,n),"float32"),("inflow",(n,),"float32"),
        ("initial",(n,),"float32"),("lpq",(n,),"float32"),("lph",(n,),"float32"),
        ("types",(n,),"int32"),("controls",(8,n),"float64"),("clock",(n,),"float64"),
        ("carriers",(3,n),"float64"),("indices",(n,),"int32"),("height",(n,),"float32"),
        ("release",(n,),"float64"),("assimilated",(n,),"float64"),("dynamic_type",(n,),"int32"),
        ("source",(n,),"int32"),("fault",(n,),"int32")):
        a[name]=cp.empty(shape,dtype=dtype)
    committed = cp.empty((columns,n),dtype="float64")
    stages=[cp.cuda.Event() for _ in range(4)]
    kind=int32(4 if metadata["kind"]=="rfc" else 2)
    launch=((n+127)//128,)
    stages[0].record()
    module.get_function("ngen_reservoir_da_feed")(launch,(128,),(int32(n),kind,values,
        a["parameters"],a["inflow"],a["initial"],a["lpq"],a["lph"],a["types"],
        a["controls"],a["clock"],a["carriers"],a["indices"]))
    stages[1].record()
    module.get_function("ngen_reservoir_assimilate")(launch,(128,),(int32(n),uint64(0),int32(n),
        a["parameters"],a["inflow"],a["initial"],a["lpq"],a["lph"],a["types"],offsets,
        series,times,a["controls"],float64(metadata["packing"]["routing_period_seconds"]),a["clock"],
        a["carriers"],a["indices"],a["height"],a["release"],a["assimilated"],a["dynamic_type"],a["source"],a["fault"]))
    stages[2].record()
    module.get_function("ngen_reservoir_da_record")(launch,(128,),(int32(n),kind,a["release"],a["height"],
        a["carriers"],a["indices"],a["dynamic_type"],a["assimilated"],a["source"],committed))
    stages[3].record();stages[3].synchronize()
    io.store(committed,args.output/"actual-commit.f64")
    request["case"] += "-native-carrier-commit"
    request["fields"][0].update(name="native_committed_return_fields",
        expected=dict(path=str(args.fixture/"expected-commit.f64"),sha256=metadata["files"]["expected-commit.f64"]["sha256"]),
        actual=str(args.output/"actual-commit.f64"))
    commit_path=args.output/"commit-request.json"
    commit_path.write_text(json.dumps(request,indent=2)+"\n")
    commit_command=[args.comparator,str(commit_path),str(args.output/"commit-verdict.json")]
    commit_result=subprocess.run(commit_command,capture_output=True,text=True)
    receipt = dict(kind=metadata["kind"], calls=n, compared_words=columns*n,
                   card=cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())["name"].decode(),
                   candidate_revision=args.revision, source_pin=SOURCE_PIN, bitwise_match=result.returncode == 0,
                   comparison_command=command, comparison_result=result.stdout+result.stderr,
                   comparison_returncode=result.returncode, setup_ms=setup_ms,
                   replay_kernel_ms=cp.cuda.get_elapsed_time(events[0], events[1]),
                   parallelism="one thread per independent native call,128 threads/block",
                   timing_scope="one cold independent-call replay kernel; not resident trajectory or isolated production speed",
                   cuda_graphs=False, warmup_steps=0,
                   actual_sha256=io.sha256(args.output / "actual.f64"),
                   expected_sha256=metadata["files"]["expected.f64"]["sha256"],
                   capture_manifest_sha256=io.sha256(args.fixture / "manifest.json"))
    receipt.update(commit_bitwise_match=commit_result.returncode==0,
                   commit_comparison_command=commit_command,
                   commit_comparison_result=commit_result.stdout+commit_result.stderr,
                   committed_actual_sha256=io.sha256(args.output/"actual-commit.f64"),
                   committed_expected_sha256=metadata["files"]["expected-commit.f64"]["sha256"],
                   separate_boundary_stages_ms=dict(feed=cp.cuda.get_elapsed_time(stages[0],stages[1]),
                       production_assimilation_commit=cp.cuda.get_elapsed_time(stages[1],stages[2]),
                       record=cp.cuda.get_elapsed_time(stages[2],stages[3])),
                   commit_scope="production global kernel from independent captured native inputs; no coupled DA trajectory claim")
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2))
    result.check_returncode()
    commit_result.check_returncode()


if __name__ == "__main__":
    main()
