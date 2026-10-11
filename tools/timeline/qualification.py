"""Real-run C1 qualification orchestration. No model array reading.

Run `prepare` on CPU. Run `run` only inside the authorized GPU mutex.
The source tree, immutable input pin and qualification duration are explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def prefix_config(text, seconds):
    """Alter only clock/output metadata, retaining all domain and physics values."""
    original = tomllib.loads(text)
    total = original["experiment"]["run_seconds"]
    if type(seconds) is not int or seconds <= 0 or seconds > total or seconds % 3600:
        raise ValueError("qualification must be a positive whole-hour prefix of the pinned clock")
    replacements = {
        "experiment": {"run_seconds": float(seconds), "restart_interval_s": float(seconds)},
        "domain": {"history_interval_s": float(seconds)},
        "fetch": {"hours": seconds // 3600},
    }
    section = None
    lines = []
    for line in text.splitlines(keepends=True):
        match = re.match(r"\s*\[\[?([^\]]+)\]\]?\s*$", line.strip())
        if match:
            section = match[1]
        key = re.match(r"\s*([A-Za-z_][A-Za-z_0-9]*)\s*=", line)
        if key and key[1] in replacements.get(section, {}):
            value = replacements[section][key[1]]
            line = f"{key[1]} = {value}\n"
        lines.append(line)
    result = "".join(lines)
    changed = tomllib.loads(result)
    expected = json.loads(json.dumps(original, default=str))
    actual = json.loads(json.dumps(changed, default=str))
    expected["experiment"].update(replacements["experiment"])
    for domain in expected["domain"]:
        domain.update(replacements["domain"])
    expected["fetch"].update(replacements["fetch"])
    if expected != actual:
        raise ValueError("qualification rewrite changed non-clock metadata")
    return result


def cache_env(cache):
    names = {"CUPY_CACHE_DIR": "cupy", "CUDA_CACHE_PATH": "cuda",
             "XDG_CACHE_HOME": "xdg", "GPUWM_DATA_STORE_CACHE": "data-store",
             "GPUWM_CHEM_SOURCE_CACHE": "chem", "GPUWM_CACHE_ROOT": "gpuwm",
             "GPUWM_STATIC_SOURCE_ROOT": "static_sources", "GPUWM_BENCH_INPUT_ROOT": "raw",
             "PYTHONPYCACHEPREFIX": "python", "TMPDIR": "tmp"}
    result = {}
    for key, relative in names.items():
        path = cache / relative
        path.mkdir(parents=True, exist_ok=True)
        result[key] = str(path)
    (cache / "home").mkdir(exist_ok=True)
    return result


def prepare(args):
    from tools.bench.protocol import pack, seed_inputs
    pin, _ = pack(args.input_pack.parent, args.input_pack.name, args.pin_sha)
    if args.outdir.exists():
        raise ValueError("qualification input directory already exists; do not overwrite its authority")
    args.outdir.mkdir(parents=True)
    text = (args.input_pack / "experiment.toml").read_text()
    qualified = prefix_config(text, args.seconds)
    config = args.outdir / "experiment.toml"
    config.write_text(qualified)
    nml = args.outdir / "experiment.namelist.wps"
    nml.write_bytes((args.input_pack / "experiment.namelist.wps").read_bytes())
    caches = cache_env(args.cache)
    seeds = seed_inputs(pin, args.input_pack, args.cache)
    result = {"schema": "c1-qualification-input-v1", "parent_pin_sha256": args.pin_sha,
              "parent_input_pack": str(args.input_pack), "case": pin["case"],
              "parent_forecast_seconds": pin["forecast_seconds"], "qualification_seconds": args.seconds,
              "config": str(config), "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
              "wps_namelist": str(nml), "wps_sha256": hashlib.sha256(nml.read_bytes()).hexdigest(),
              "cache": str(args.cache), "cache_environment": caches, "raw_aliases": seeds,
              "changed_metadata": ["experiment.run_seconds", "experiment.restart_interval_s",
                                   "domain.history_interval_s", "fetch.hours"]}
    write(args.outdir / "input-receipt.json", result)
    print(json.dumps({"input_receipt": str(args.outdir / "input-receipt.json"),
                      "seconds": args.seconds, "config_sha256": result["config_sha256"]}))


def bridge_env(base):
    """Select the existing shared binaries via the engine's own artifact table."""
    from gpuwm.bridge_assets import BUNDLED_ARTIFACTS, artifact_filename, host_platform
    result = {}
    for artifact in BUNDLED_ARTIFACTS:
        path = base / artifact.crate / "target" / "release" / artifact_filename(artifact, host_platform())
        if path.is_file():
            result[artifact.env_var] = str(path)
    mapped = base / "tools/rw_wps/target/release/gpuwm_mapped_engine"
    if mapped.is_file():
        result["GPUWM_MAPPED_ENGINE_BIN"] = str(mapped)
    return result


def physical_card(visible):
    """Mutexes may grant UUIDs. Preserve unknown physical indexes as null."""
    first = visible.split(",")[0]
    return int(first) if first.isdecimal() else None


def profile_command(argv, source):
    if Path(source).resolve() != ROOT:
        raise ValueError("C1 profile source must match its tool tree; profiling another tree would misidentify the measured engine")
    if argv[1:3] != ["-m", "gpuwm.cli"]:
        raise ValueError("C1 call budget requires the real gpuwm CLI launcher; a different entry point would misidentify the profile")
    return [argv[0], str(ROOT / "tools/timeline/profile_cli.py"), *argv[3:]]


def run(args):
    from gpuwm.timeline import Ledger
    from gpuwm.stage_timing import ForecastLedger
    from gpuwm.timeline.taxonomy import RUN_STAGES
    from tools.bench.protocol import pack, expand
    authority = json.loads((args.input_dir / "input-receipt.json").read_text())
    pin, _ = pack(Path(authority["parent_input_pack"]).parent, authority["case"], authority["parent_pin_sha256"])
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("qualification run requires the mutex-granted visible card set")
    if args.outdir.exists():
        raise ValueError("run output already exists; a fresh path is required")
    args.outdir.mkdir(parents=True)
    source = args.source.resolve()
    ledger_path = args.outdir / "ledger.jsonl"
    run_id = args.outdir.name
    env = dict(os.environ, **authority["cache_environment"],
               PYTHONPATH=str(source) + ":" + str(source / "tests"),
               GPUWM_TIME_LEDGER=str(ledger_path), GPUWM_RUN_ID=run_id, GPUWM_PRODUCT="regional",
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", RAYON_NUM_THREADS="4")
    env.update(bridge_env(args.engine_base))
    variables = {"python": sys.executable, "input": authority["parent_input_pack"],
                 "cache": authority["cache"], "output": str(args.outdir),
                 "source": str(source), "engine_base": str(args.engine_base),
                 "ledger": str(ledger_path), "cards": "1"}
    argv = expand(pin["argv"], variables)
    original_config = str(Path(authority["parent_input_pack"]) / "experiment.toml")
    if argv.count(original_config) != 1:
        raise ValueError("pinned launcher must bind one declared experiment config")
    argv[argv.index(original_config)] = authority["config"]
    if args.prepared_root:
        argv += ["--prepared-root", str(args.prepared_root), "--wps-namelist", authority["wps_namelist"]]
    if args.skeleton_flags:
        argv += ["--stage-only", "unused", "--warm-kernels", "--static-cache-root", authority["cache"],
                 "--static-cache-key", "unused", "--readiness-fold", "--live-products-width", "1"]
    if args.profile_c1:
        argv = profile_command(argv, source)
        env["GPUWM_C1_PROFILE_RECEIPT"] = str(args.outdir / "call-budget.json")
    isolated = ["bash", str(ROOT / "tools/bench/isolate.sh"),
                str(Path(authority["cache"]) / "home"), str(Path.home()), "--", *argv]
    ledger = Ledger(ledger_path, run_id=run_id, product="regional",
                    card=physical_card(os.environ["CUDA_VISIBLE_DEVICES"]))
    receipt = {"schema": "c1-real-qualification-v1", "source": str(source),
               "source_tip": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
               "parent_pin_sha256": authority["parent_pin_sha256"],
               "qualification_config_sha256": authority["config_sha256"],
               "seconds": authority["qualification_seconds"], "argv": argv,
               "isolation_argv": isolated, "cache_roots": authority["cache_environment"],
               "bridges": {key: value for key, value in env.items() if key.startswith("GPUWM_")
                           and value.startswith(str(args.engine_base))},
               "visible_cards": os.environ["CUDA_VISIBLE_DEVICES"], "status": "RUNNING"}
    write(args.outdir / "receipt.json", receipt)
    ledger.emit("worker.claim", attrs={"basis": "bench proxy envelope", "label": run_id})
    started = time.monotonic()
    with (args.outdir / "run.log").open("w") as log:
        completed = subprocess.run(isolated, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
    receipt["wall_seconds"] = time.monotonic() - started
    receipt["returncode"] = completed.returncode
    receipt["status"] = "ENGINE_COMPLETE" if completed.returncode == 0 else "ENGINE_FAILED"
    ledger.emit("finish", attrs={"basis": "bench proxy envelope", "status": receipt["status"]})
    if args.legacy_ledger:
        # b287 did not have C1. Relay its existing stage and step clocks after
        # stopping the measured process timer, preserving the measurement basis.
        events = args.outdir / "artifacts/events.jsonl"
        forecast_started = None
        if events.is_file():
            for line in events.read_text().splitlines():
                row = json.loads(line)
                if row["event"] in ("stage_started", "stage_finished"):
                    unix = row["emitted_unix_ms"] / 1000.0
                    mono = time.monotonic() + unix - time.time()
                    stage = RUN_STAGES[row["stage"]]
                    phase = "begin" if row["event"] == "stage_started" else "end"
                    ledger.emit(stage, phase, attrs={"basis": "b287.event_relay", "role": "envelope"},
                                t_mono=mono, t_unix=unix)
                    if row["stage"] == "forecast" and phase == "begin":
                        forecast_started = (mono, unix)
        progress = args.outdir / "artifacts/chain/run"
        relay = ForecastLedger(ledger, progress)
        if forecast_started:
            mono, unix = forecast_started
            ledger.emit("forecast.construct", "begin", attrs={"basis": "b287.event_relay"},
                        t_mono=mono, t_unix=unix)
            relay.construct_open = True
        relay.read()
        relay.finish()
    write(args.outdir / "receipt.json", receipt)
    pack(Path(authority["parent_input_pack"]).parent, authority["case"], authority["parent_pin_sha256"])
    print(json.dumps({"output": str(args.outdir), "status": receipt["status"],
                      "wall_seconds": receipt["wall_seconds"]}))
    return completed.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("prepare")
    p.add_argument("--input-pack", type=Path, required=True)
    p.add_argument("--pin-sha", required=True)
    p.add_argument("--seconds", type=int, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    r = subs.add_parser("run")
    r.add_argument("--source", type=Path, required=True)
    r.add_argument("--engine-base", type=Path, required=True)
    r.add_argument("--input-dir", type=Path, required=True)
    r.add_argument("--outdir", type=Path, required=True)
    r.add_argument("--prepared-root", type=Path)
    r.add_argument("--legacy-ledger", action="store_true")
    r.add_argument("--skeleton-flags", action="store_true")
    r.add_argument("--profile-c1", action="store_true", help="diagnostic C1 method budget; retains the real CLI and scientific arguments")
    args = parser.parse_args()
    return prepare(args) if args.command == "prepare" else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
