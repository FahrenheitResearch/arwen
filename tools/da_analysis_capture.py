"""Capture the LETKF solves of a real DA run, then replay them on the host.

WHY.  A cycle's analysis is only reproducible inside the cycle: its prior,
H(x) and observation batches exist for a few seconds between a forecast
leg and the applier.  Proving that a faster host solve gives the same bytes
as the current one on the REAL case needs those exact inputs on disk, and
timing worker counts against each other needs them replayable without a
forecast leg in front.

    # 1. capture: run any DA driver under the hook; every solve the analysis
    #    performs (the main one and any dispersion-gated re-solve) is written
    #    to DIR/call-NNN before it runs, then runs unchanged.
    python -m tools.da_analysis_capture capture --dir DIR -- \\
        -m tools.da_cycle_prepared <its arguments>

    # 2. replay one captured solve on the host, timed, increments saved
    python -m tools.da_analysis_capture replay --call DIR/call-000 \\
        --workers 64 --out inc.npz --record rec.json

    # 3. compare two replays (or a replay and a reference) byte for byte
    python -m tools.da_analysis_capture compare A.npz B.npz

Replay always solves with numpy (the host path).  ``--workers`` sets
``LetkfConfig.host_workers`` when the engine under test has it; the
reference arm of an older engine is taken with ``GPUWM_DA_HOST_WORKERS=1``
and no ``--workers``.  This tool reads and writes files only; nothing in
the forecast or DA path imports it.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pickle
import runpy
import sys
import time
from pathlib import Path

import numpy as np

SCHEMA = "gpuwm-da.analysis-capture.v1"


def _host(value):
    if type(value).__module__.split(".")[0] == "cupy":
        return value.get()
    return value


def _host_batch(batch):
    return dataclasses.replace(
        batch, values=_host(batch.values), errors=_host(batch.errors),
        simulated=_host(batch.simulated), mask=_host(batch.mask))


def install(directory: Path) -> None:
    """Wrap the analysis seam so every solve is written before it runs."""
    from gpuwm.da import radar_assimilation as ra      # noqa: PLC0415

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    original = ra._execute_analysis
    counter = {"n": 0}

    def capturing(solver, prior, batches, geometry, config, **kwargs):
        call = directory / f"call-{counter['n']:03d}"
        counter["n"] += 1
        started = time.perf_counter()
        call.mkdir(parents=True, exist_ok=False)
        np.savez(call / "prior.npz",
                 **{name: np.asarray(_host(value)) for name, value in prior.items()})
        with open(call / "inputs.pkl", "wb") as stream:
            pickle.dump({"batches": [_host_batch(b) for b in batches],
                         "geometry": geometry, "config": config},
                        stream, protocol=5)
        record = {"schema": SCHEMA, "call": call.name,
                  "device": kwargs.get("device"),
                  "fields": list(prior), "batches": len(batches),
                  "write_seconds": round(time.perf_counter() - started, 3)}
        out = original(solver, prior, batches, geometry, config, **kwargs)
        record["solve_wall_seconds_in_run"] = round(
            time.perf_counter() - started - record["write_seconds"], 3)
        record["storage_in_run"] = out[4]
        # The increments the run itself applies, by field, so a replay can
        # be held against the live solve byte for byte, not only another
        # replay.  Hashed after the solve; not counted in its wall time.
        record["sha256_in_run"] = {name: _digest(_host(value))
                                   for name, value in out[0].items()}
        (call / "capture.json").write_text(json.dumps(record, indent=2) + "\n")
        return out

    ra._execute_analysis = capturing


def load(call: Path):
    call = Path(call)
    with np.load(call / "prior.npz") as data:
        prior = {name: np.ascontiguousarray(data[name]) for name in data.files}
    with open(call / "inputs.pkl", "rb") as stream:
        inputs = pickle.load(stream)
    return prior, inputs["batches"], inputs["geometry"], inputs["config"]


def _digest(array) -> str:
    array = np.ascontiguousarray(array)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def replay(call: Path, *, workers: int | None, out: Path | None,
           record_path: Path | None) -> dict:
    from gpuwm.da import letkf                          # noqa: PLC0415

    t0 = time.perf_counter()
    prior, batches, geometry, config = load(call)
    load_seconds = time.perf_counter() - t0
    if workers is not None:
        config = dataclasses.replace(config, host_workers=int(workers))
    diag = letkf.LetkfDiagnostics()
    t1 = time.perf_counter()
    increments = letkf.analyze(prior, batches, geometry, config, diag,
                               solve_namespace=np)
    wall = time.perf_counter() - t1
    record = {
        "schema": SCHEMA, "call": str(call), "engine": letkf.__file__,
        "workers_requested": workers,
        "env_host_workers": os.environ.get("GPUWM_DA_HOST_WORKERS"),
        "load_seconds": round(load_seconds, 3),
        "analyze_wall_seconds": round(wall, 3),
        "diagnostics": {k: v for k, v in vars(diag).items()
                        if isinstance(v, (int, float, str, bool, dict))},
        "sha256": {name: _digest(value) for name, value in increments.items()},
        "dtype": {name: str(value.dtype) for name, value in increments.items()},
    }
    if out is not None:
        np.savez(out, **increments)
    if record_path is not None:
        Path(record_path).write_text(
            json.dumps(record, indent=2, default=str) + "\n")
    return record


def compare(a: Path, b: Path) -> dict:
    rows = {}
    with np.load(a) as left, np.load(b) as right:
        if sorted(left.files) != sorted(right.files):
            raise SystemExit(f"field sets differ: {left.files} vs {right.files}")
        for name in left.files:
            x, y = left[name], right[name]
            same = x.dtype == y.dtype and x.shape == y.shape and \
                x.tobytes() == y.tobytes()
            row = {"identical_bytes": bool(same)}
            if not same and x.shape == y.shape:
                d = x.astype(np.float64) - y.astype(np.float64)
                row.update(max_abs=float(np.abs(d).max()),
                           rms=float(np.sqrt(np.mean(d * d))),
                           differing=int(np.count_nonzero(d)))
            rows[name] = row
    return {"a": str(a), "b": str(b),
            "all_identical": all(r["identical_bytes"] for r in rows.values()),
            "fields": rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--dir", type=Path, required=True)
    cap.add_argument("target", nargs=argparse.REMAINDER,
                     help="-- -m module args...  or  -- script.py args...")
    rep = sub.add_parser("replay")
    rep.add_argument("--call", type=Path, required=True)
    rep.add_argument("--workers", type=int, default=None)
    rep.add_argument("--out", type=Path, default=None)
    rep.add_argument("--record", type=Path, default=None)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("a", type=Path)
    cmp_.add_argument("b", type=Path)
    args = parser.parse_args(argv)
    if args.command == "capture":
        target = list(args.target)
        if target and target[0] == "--":
            target = target[1:]
        if not target:
            parser.error("capture needs a target after --")
        install(args.dir)
        if target[0] == "-m":
            sys.argv = [target[1], *target[2:]]
            runpy.run_module(target[1], run_name="__main__", alter_sys=True)
        else:
            sys.argv = target
            runpy.run_path(target[0], run_name="__main__")
        return 0
    if args.command == "replay":
        record = replay(args.call, workers=args.workers, out=args.out,
                        record_path=args.record)
        print(json.dumps({k: record[k] for k in
                          ("analyze_wall_seconds", "load_seconds", "sha256")},
                         indent=1))
        return 0
    result = compare(args.a, args.b)
    print(json.dumps(result, indent=1))
    return 0 if result["all_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
