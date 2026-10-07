"""Plan and run the public packed regional DA and matched no-DA arms."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tools.regional_rain_score import source_revision


def utc(value):
    date = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if date.tzinfo is None or date.utcoffset().total_seconds() != 0:
        raise ValueError("case clocks must state UTC explicitly")
    return date


def plan(case, *, seed, arm="both", devices=None):
    required = ("source", "prepared_root", "authority_dir", "physics_profile",
                "proof_sha256", "source_manifest_sha256", "prepared_content_sha256",
                "run_seconds", "history_interval_seconds", "slots", "surface_obs", "out")
    missing = [key for key in required if key not in case]
    if missing:
        raise ValueError("recent case lacks prepared authorities: " + ", ".join(missing))
    if case.get("engine_sha") != source_revision():
        raise ValueError("recent case and executing engine source revisions differ")
    smoke = bool(case.get("smoke_only", False))
    members = int(case.get("members", 32))
    if members != (4 if smoke else 32):
        raise ValueError("recent operational roster is 32 members; declared small smoke uses four")
    for key in ("proof_sha256", "source_manifest_sha256", "prepared_content_sha256"):
        value = case[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"{key} must be an actual SHA-256 digest")
    if float(case["history_interval_seconds"]) != 120:
        raise ValueError("regional rain requires the authority's 120-second history clock")
    slots = case["slots"]
    if len(slots) not in ((1,) if smoke else (3, 4, 5, 6)):
        raise ValueError("recent DA uses three to six hourly slots; the small smoke uses one")
    origin = utc(case["model_start_utc"])
    elapsed = 0.0
    observations, grids, durations = [], [], []
    for slot in slots:
        seconds = float(slot["leg_seconds"])
        if not math.isfinite(seconds) or seconds != 3600:
            raise ValueError("recent analysis legs must be exactly one hour")
        elapsed += seconds
        if (utc(slot["analysis_time"])-origin).total_seconds() != elapsed:
            raise ValueError("recent slot timestamp disagrees with its cumulative clock")
        for key in ("obs", "grid_wrfout"):
            if not Path(slot[key]).is_file():
                raise ValueError("missing prepared observation input: " + slot[key])
        observations.append(slot["obs"])
        grids.append(slot["grid_wrfout"])
        durations.append(seconds)
    if utc(case["forecast_fork_utc"]) != utc(slots[-1]["analysis_time"]):
        raise ValueError("forecast issuance must equal the final analysis")
    free = float(case.get("free_forecast_seconds", 21600))
    if free != (120 if smoke else 21600):
        raise ValueError("recent forecasts are six hours; the declared smoke advances two minutes")
    if not math.isfinite(float(case["run_seconds"])) or elapsed+free > float(case["run_seconds"]):
        raise ValueError("prepared forcing horizon does not cover issuance plus the complete forecast")
    if not Path(case["surface_obs"]).is_file():
        raise ValueError("hourly surface observations are missing")
    out = Path(case["out"])/f"seed-{int(seed)}"
    # The 33-point integration fixture cannot resolve a 40 km spectral
    # perturbation. Keep the operational scales and both arms identical.
    # A case authored at another spacing names its own scales; a manifest
    # without them is the deck's 3 km case.
    length_scale = str(case.get("perturbation_length_scale_km", 12 if smoke else 40))
    loc = {"horizontal_loc_m": 12000.0, "vertical_loc_m": 6000.0,
           "sfc_horizontal_loc_m": 12000.0, "sfc_vertical_loc_m": 3000.0, **case.get("localization", {})}
    radius = lambda key: f"{float(loc[key]):g}"
    common = [sys.executable, "-m", "tools.da_cycle_prepared",
        "--source", case["source"], "--prepared-root", case["prepared_root"],
        "--authority-dir", case["authority_dir"], "--physics-profile", case["physics_profile"],
        "--proof-sha256", case["proof_sha256"], "--source-manifest-sha256", case["source_manifest_sha256"],
        "--prepared-content-sha256", case["prepared_content_sha256"],
        "--run-seconds", str(case["run_seconds"]), "--history-interval-seconds", "120",
        "--members", str(members), "--seed", str(seed), "--solve-device", "cuda",
        "--memory-budget-mib", str(case.get("analysis_memory_budget_mib", 14336)),
        "--forecast-members-per-card", "auto", "--forecast-leg-timeout-seconds", "3600",
        "--wind-sigma-ms", "1.5", "--length-scale-km", length_scale,
        "--theta-sigma-k", "0.5", "--qv-log-sigma", "0.05", "--hydro-log-sigma", "0.7",
        "--thermo-length-scale-km", length_scale, "--horizontal-loc-m", radius("horizontal_loc_m"),
        "--vertical-loc-m", radius("vertical_loc_m"),
        "--hydrometeors", "--reflectivity-analysis", "--clear-air-analysis",
        # No positivity override: the deck takes the door's own default
        # (mean-preserving, 7a38f128a).  This deck was written one day
        # before that default and kept naming clip, so every campaign
        # analysis raised the undershooting members to zero and lowered
        # none: condensate the filter never proposed, added at every
        # analysis, in exactly the cells the reflectivity batch had just
        # pulled on.  The receipt key positivity_after_merge.mass_added_by_clip
        # counts it.
        "--velocity-dispersion-gate", "2",
        "--velocity-dispersion-batch-gate", "3", "--no-hotstart", "--save-composites",
        "--rain-history", "--rain-forecast-start-seconds", str(elapsed)]
    if smoke:
        common += ["--packed-smoke"]
    if devices:
        common += ["--forecast-device-uuids", *devices]
    slot_flags = [value for obs, grid in zip(observations, grids, strict=True)
                  for value in ("--obs", obs, "--grid-wrfout", grid)]
    da = common + slot_flags + [
        "--leg-durations-seconds", *map(str, durations+[free]), "--free-legs", "1",
        "--surface-obs", case["surface_obs"], "--sfc-t2-sigma-k", "2",
        "--sfc-wspd-sigma-ms", "2", "--sfc-td-sigma-k", str(case.get("surface_dewpoint_error_k", 1)),
        "--sfc-max-age-s", "900", "--sfc-elev-max-diff-m", "200",
        "--sfc-horizontal-loc-m", radius("sfc_horizontal_loc_m"), "--sfc-vertical-loc-m", radius("sfc_vertical_loc_m"),
        "--save-ensemble", str(out/"issuance-generation"), "--out", str(out/"da")]
    control = common + ["--free-legs", str(len(durations)+1),
        "--leg-durations-seconds", *map(str, durations+[free]), "--out", str(out/"no-da")]
    commands = [{"arm": "da", "argv": da}, {"arm": "no-da", "argv": control}]
    return [row for row in commands if arm == "both" or row["arm"] == arm]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--arm", choices=("da", "no-da", "both"), default="both")
    parser.add_argument("--device-uuids", nargs="+")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--wall-budget-seconds", type=float, default=4800)
    args = parser.parse_args(argv)
    if not math.isfinite(args.wall_budget_seconds) or args.wall_budget_seconds <= 0:
        parser.error("wall budget must be finite and positive")
    case = json.loads(args.manifest.read_text())
    commands = plan(case, seed=args.seed, arm=args.arm, devices=args.device_uuids)
    for row in commands:
        print(json.dumps(row), flush=True)
    if not args.execute:
        return 0
    root = Path(case["out"])/f"seed-{args.seed}"
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    env = {**os.environ, "GPUWM_MAPPED_ENGINE_THREADS": "12",
        "GPUWM_MAPPED_ENGINE_MEMORY_BUDGET_BYTES": str(5*1024**3),
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    for row in commands:
        log = root/f"{row['arm']}.log"
        before = time.monotonic()
        remaining = args.wall_budget_seconds-(before-started)
        if remaining <= 0:
            return 124
        with log.open("xb") as stream:
            process = subprocess.Popen(row["argv"], env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=remaining)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                # INT gives the packed controller its owned-worker cleanup.
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                code = 124
        receipt = {**row, "exit_code": code, "log": str(log),
            "wall_seconds": time.monotonic()-before, "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "engine_sha": source_revision(), "status": "complete" if code == 0 else "incomplete",
            "scientific_gate": "pending"}
        with (root/f"receipts-{args.arm}.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt)+"\n")
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
