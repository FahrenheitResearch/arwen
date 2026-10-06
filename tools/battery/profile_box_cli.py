"""Run the genuine production prep command with fixture inputs and RSS trace.

Target horizontal dimensions alone are scaled. The original CLI options,
native 50 level RAP and HRRR source geometry, physics and static defaults
remain. Inputs have synthetic constant values. No production callable is
replaced. A two-core affinity contains desktop CPU use without changing
the command's 48 preprocessing workers.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import psutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--static-source", type=Path)
    parser.add_argument("--static-library", type=Path)
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--max-gib", type=float, default=30)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    engine, case, inputs, out = (p.resolve() for p in (args.engine, args.case, args.inputs, args.out))
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(engine)
    env["GPUWM_NO_LOCAL_GPU"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONUNBUFFERED"] = "1"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS", "GPUWM_MAPPED_ENGINE_THREADS"):
        env[name] = str(args.threads)
    # Match production cache resolver on Windows and POSIX, while each
    # measured run starts with no derived mosaic and uses identical tiles.
    cache = out / "cache"
    original = case / "cache" / "gpuwm" / "highres-cache" / "copernicus_dem_glo30"
    destination = cache / "gpuwm" / "highres-cache" / "copernicus_dem_glo30"
    destination.mkdir(parents=True)
    for path in original.iterdir():
        if path.suffix in (".tif", ".sha256") or path.name.endswith(".sha256.json"):
            os.link(path, destination / path.name)
    env["LOCALAPPDATA"] = str(cache)
    env["XDG_CACHE_HOME"] = str(cache)
    scratch = out / "scratch"
    scratch.mkdir()
    env["TMPDIR"] = str(scratch)
    env["TEMP"] = str(scratch)
    env["TMP"] = str(scratch)
    env["GPUWM_COMPOSE_SCRATCH"] = str(scratch)
    # This imports exact pinned production modules, with no callable or
    # algorithm replacements. The outer production CLI remains unchanged.
    # The current highres_fetch module retains the separate tile race fix.
    if args.static_source is not None:
        overlay = out / "python-overlay"
        overlay.mkdir()
        source = args.static_source.resolve()
        loader = "import importlib.util, pathlib, sys\nimport gpuwm.static\n"
        loader += f"source = pathlib.Path({str(source)!r})\n"
        loader += "for name in ('rust_bridge', 'highres'):\n"
        loader += "    canonical = 'gpuwm.static.' + name\n"
        loader += "    spec = importlib.util.spec_from_file_location(canonical, source / 'gpuwm' / 'static' / (name + '.py'))\n"
        loader += "    module = importlib.util.module_from_spec(spec)\n"
        loader += "    sys.modules[canonical] = module\n"
        loader += "    setattr(gpuwm.static, name, module)\n"
        loader += "    spec.loader.exec_module(module)\n"
        (overlay / "sitecustomize.py").write_text(loader)
        env["PYTHONPATH"] = os.pathsep.join((str(overlay), str(engine)))
    if args.static_library is not None:
        env["GPUWM_STATIC_BRIDGE"] = str(args.static_library.resolve())
    native_binary_candidates = (
        "tools/rw_wps/target/release/gpuwm_mapped_engine.exe",
        "tools/grib1_bridge/target/release/gpuwm_preprocess_cpu.dll",
        "tools/grib1_bridge/target/release/grib2_inventory.exe",
        "tools/grib1_bridge/target/release/grib2_dump.exe")
    native_binaries = {name: hashlib.sha256((engine / name).read_bytes()).hexdigest()
        for name in native_binary_candidates if (engine / name).is_file()}
    mapped_engine = engine / native_binary_candidates[0]
    if mapped_engine.is_file():
        env["GPUWM_MAPPED_ENGINE_BIN"] = str(mapped_engine)
    command = [sys.executable, "-m", "gpuwm.cli", "prep", "--source", "rap-native"]
    for lead in (0, 3):
        command += ["--input", str(inputs / f"rap-native-f{lead:02}.grib2")]
    for lead in (0, 3):
        command += ["--supplement", str(inputs / f"rap-native-f{lead:02}.grib2")]
    command += ["--author-input-manifest", str(out / "proof-inputs.json"),
        "--initial-inputs", str(inputs / "initial-inputs.json"),
        "--experiment-config", str(case / "experiment.toml"),
        "--wps-namelist", str(case / "namelist.wps"),
        "--geog-root", str(case / "geog"), "--preprocess-backend", "cpu",
        "--preprocess-workers", "48", "--no-stock-wrf-export",
        "--output-root", str(out / "prepared.partial")]
    sources = {name: hashlib.sha256((engine / name).read_bytes()).hexdigest()
        for name in ("gpuwm/static/highres.py", "gpuwm/static/rust_bridge.py",
                     "gpuwm/static/highres_fetch.py", "gpuwm/static/highres_production.py",
                     "gpuwm/mapped_direct.py")}
    if args.static_source is not None:
        for name in ("gpuwm/static/highres.py", "gpuwm/static/rust_bridge.py"):
            sources[name] = hashlib.sha256((args.static_source / name).read_bytes()).hexdigest()
    receipt = {"schema": "static-prep-real-cli-profile-v1", "command": command,
        "cwd": str(engine), "fixture": str(case), "input_receipt": str(inputs / "input-receipt.json"),
        "config_sha256": {name: hashlib.sha256((case / name).read_bytes()).hexdigest()
                          for name in ("experiment.toml", "namelist.wps")},
        "source_sha256": sources, "environment": {k: env[k] for k in ("GPUWM_NO_LOCAL_GPU", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS", "GPUWM_MAPPED_ENGINE_THREADS", "LOCALAPPDATA", "XDG_CACHE_HOME")},
        "worker_option": 48, "cpu_affinity": [0, 1], "synthetic_source_values": True,
        "decode_or_transform_mocked": False, "max_gib": args.max_gib}
    receipt["native_binary_sha256"] = native_binaries
    receipt["static_source_root"] = None if args.static_source is None else str(args.static_source.resolve())
    receipt["static_library"] = None if args.static_library is None else {
        "path": str(args.static_library.resolve()),
        "sha256": hashlib.sha256(args.static_library.read_bytes()).hexdigest()}
    if args.static_library is not None:
        library = ctypes.CDLL(str(args.static_library.resolve()))
        limits = {}
        previous_rayon = os.environ.get("RAYON_NUM_THREADS")
        os.environ["RAYON_NUM_THREADS"] = env["RAYON_NUM_THREADS"]
        try:
            for name, result_type in (
                ("gpuwm_static_continuous_warp_worker_cap", ctypes.c_uint32),
                ("gpuwm_static_continuous_warp_workers", ctypes.c_uint32),
                ("gpuwm_static_continuous_warp_source_window_bytes", ctypes.c_uint64),
                ("gpuwm_static_continuous_warp_reader_cache_bytes", ctypes.c_uint64)):
                if hasattr(library, name):
                    function = getattr(library, name)
                    function.argtypes = []
                    function.restype = result_type
                    limits[name] = function()
        finally:
            if previous_rayon is None:
                os.environ.pop("RAYON_NUM_THREADS", None)
            else:
                os.environ["RAYON_NUM_THREADS"] = previous_rayon
        receipt["native_static_limits"] = limits
    (out / "command.json").write_text(json.dumps(receipt, indent=2) + "\n")
    log = (out / "prep.log").open("wb")
    started = time.perf_counter()
    proc = subprocess.Popen(command, cwd=engine, env=env, stdout=log, stderr=subprocess.STDOUT)
    ps = psutil.Process(proc.pid)
    ps.cpu_affinity([0, 1])
    peak = 0
    stop_reason = None
    stages = {}
    stage = "startup"
    static_rollup = {"first_sample_seconds": None, "last_sample_seconds": None,
        "peak_tree_rss_bytes": 0, "peak_process_rss_bytes": 0,
        "peak_process_os_rss_bytes": 0, "samples": 0}
    process_inventory = {}
    with (out / "rss.tsv").open("w") as trace, (out / "rss-per-pid.tsv").open("w") as pid_trace:
        trace.write("seconds\ttree_rss_bytes\tprocesses\tstage\troot_static_active\n")
        pid_trace.write("seconds\tpid\tppid\trss_bytes\tprivate_bytes\tpeak_wset_bytes\tlaunch_parent\troot_static_active\tstage\n")
        while proc.poll() is None:
            processes = [ps]
            try:
                processes += ps.children(recursive=True)
            except psutil.Error:
                pass
            seconds = time.perf_counter() - started
            current_log = (out / "prep.log").read_text(errors="replace")
            static_active = False
            # Reconstruct the open outer interval from observed progress.
            # Nested fetch/warp/merge events cannot close that interval.
            for line in current_log.splitlines():
                if line == "prep: Prepare root static fields":
                    static_active = True
                elif line.startswith(("prep: Prepare root static fields: done",
                                      "prep: Prepare root static fields: failed")):
                    static_active = False
                if line.startswith("prep: "):
                    match = re.match(r"prep: ([A-Z][^:]*?)(?: \(.*|:.*)?$", line)
                    if match:
                        stage = match.group(1)
            rss = 0
            process_peak = process_os_peak = 0
            for child in processes:
                try:
                    child.cpu_affinity([0, 1])
                    memory = child.memory_info()
                    rss += memory.rss
                    process_peak = max(process_peak, memory.rss)
                    process_os_peak = max(process_os_peak, getattr(memory, "peak_wset", 0))
                    parent_pid = child.ppid()
                    if child.pid not in process_inventory:
                        process_inventory[child.pid] = {"ppid": parent_pid,
                            "name": child.name(), "command": child.cmdline(),
                            "launch_parent": child.pid == proc.pid,
                            "peak_rss_bytes": 0, "peak_os_rss_bytes": 0}
                    record = process_inventory[child.pid]
                    record["peak_rss_bytes"] = max(record["peak_rss_bytes"], memory.rss)
                    record["peak_os_rss_bytes"] = max(record["peak_os_rss_bytes"], getattr(memory, "peak_wset", 0))
                    pid_trace.write(f"{seconds:.6f}\t{child.pid}\t{parent_pid}\t{memory.rss}\t{getattr(memory, 'private', 0)}\t{getattr(memory, 'peak_wset', 0)}\t{int(child.pid == proc.pid)}\t{int(static_active)}\t{stage}\n")
                except psutil.Error:
                    pass
            if static_active:
                if static_rollup["first_sample_seconds"] is None:
                    static_rollup["first_sample_seconds"] = seconds
                static_rollup["last_sample_seconds"] = seconds
                static_rollup["peak_tree_rss_bytes"] = max(static_rollup["peak_tree_rss_bytes"], rss)
                static_rollup["peak_process_rss_bytes"] = max(static_rollup["peak_process_rss_bytes"], process_peak)
                static_rollup["peak_process_os_rss_bytes"] = max(static_rollup["peak_process_os_rss_bytes"], process_os_peak)
                static_rollup["samples"] += 1
            peak = max(peak, rss)
            phase = stages.setdefault(stage, {"first_sample_seconds": seconds,
                "last_sample_seconds": seconds, "peak_tree_rss_bytes": 0})
            phase["last_sample_seconds"] = seconds
            phase["peak_tree_rss_bytes"] = max(phase["peak_tree_rss_bytes"], rss)
            trace.write(f"{seconds:.6f}\t{rss}\t{len(processes)}\t{stage}\t{int(static_active)}\n")
            trace.flush()
            pid_trace.flush()
            if rss > args.max_gib * 1024 ** 3 or seconds > args.timeout:
                stop_reason = "rss ceiling" if rss > args.max_gib * 1024 ** 3 else "timeout"
                for child in reversed(processes):
                    try:
                        child.kill()
                    except psutil.Error:
                        pass
                break
            time.sleep(0.05)
    returncode = proc.wait()
    log.close()
    final_log = (out / "prep.log").read_text(errors="replace")
    outer_done = re.search(r"prep: Prepare root static fields: (done|failed) \(([0-9.]+) s\)", final_log)
    if outer_done:
        static_rollup["reported_status"] = outer_done.group(1)
        static_rollup["reported_wall_seconds"] = float(outer_done.group(2))
    events = []
    for details in out.glob("prepared.partial-prep-*.log"):
        for line in details.read_text(errors="replace").splitlines():
            if line.startswith("GPUWM_PREP_EVENT "):
                event = json.loads(line[len("GPUWM_PREP_EVENT "):])
                events.append(event)
                if event.get("stage") == "root_static" and event.get("event") in ("finished", "failed"):
                    static_rollup["wall_seconds"] = event["elapsed_seconds"]
    receipt.update({"returncode": returncode, "wall_seconds": time.perf_counter() - started,
                    "peak_tree_rss_bytes": peak, "peak_tree_rss_gib": peak / 1024 ** 3,
                    "stop_reason": stop_reason})
    receipt["stages"] = stages
    receipt["root_static_rollup"] = static_rollup
    receipt["process_inventory"] = process_inventory
    receipt["preparation_events"] = events
    (out / "profile.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("returncode", "wall_seconds", "peak_tree_rss_gib", "stop_reason")}))
    return 0 if returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
