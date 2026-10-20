#!/usr/bin/env python3
"""Run one example, retaining its command, result, wall time and device samples."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import fcntl
import json
import os
import shlex
from pathlib import Path
import subprocess
import time

import psutil


def device() -> dict:
    values = subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip().splitlines()
    if len(values) != 1:
        raise RuntimeError("proof runner requires exactly one visible device")
    used, free, util = (int(v.strip()) for v in values[0].split(","))
    return dict(used_mib=used, free_mib=free, utilization_percent=util)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--vram-mib", type=int, required=True)
    parser.add_argument("--mutex-dir", type=Path)
    parser.add_argument("--bounded-minutes", type=int, default=180, help="planned upper wall allowance; never kills a whole run")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.vram_mib < 0 or args.bounded_minutes <= 0 or (args.vram_mib and args.mutex_dir is None):
        parser.error("provide a command, a nonnegative budget, and --mutex-dir for GPU runs")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    if args.receipt.exists():
        parser.error("receipt already exists; choose a new path for this run")
    log = args.receipt.with_suffix(".log")
    admission = (args.mutex_dir / "OWNER.lock").open("a") if args.vram_mib else nullcontext()
    with admission as lock:
        if args.vram_mib:
            print("Waiting for the card OWNER lock", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
        print("Card admitted" if args.vram_mib else "CPU-only command", flush=True)
        before = device()
        if before["free_mib"] < args.vram_mib:
            raise RuntimeError(f"need {args.vram_mib} MiB; free {before['free_mib']} MiB")
        receipt = dict(command=command, cwd=str(Path.cwd()),
                       started_utc=datetime.now(timezone.utc).isoformat(),
                       vram_budget_mib=args.vram_mib, device_before=before,
                       bounded_minutes=args.bounded_minutes,
                       mutex_dir=str(args.mutex_dir) if args.vram_mib else None,
                       admission="OWNER.lock" if args.vram_mib else "CPU-only, no GPU allocation",
                       log=str(log), samples=[])
        def ledger(event):
            if args.vram_mib:
                with (args.mutex_dir / "OWNER").open("a") as out:
                    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    detail = (f"bounded {args.bounded_minutes} min ({shlex.join(command)})"
                              if event == "start" else f"rc {receipt.get('exit_code', 'unrecorded')}")
                    out.write(f"e2c-woof-docs {stamp} {event} pid {os.getpid()} {detail}\n")
        ledger("start")
        started = time.monotonic()
        max_idle = idle_since = None
        max_single_idle = single_idle_since = None
        processes = {}
        try:
            with log.open("w") as output:
                child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
                root = psutil.Process(child.pid)
                while child.poll() is None:
                    sample = dict(seconds=round(time.monotonic() - started, 3), **device())
                    cpu = []
                    try:
                        for process in [root, *root.children(recursive=True)]:
                            key = (process.pid, process.create_time())
                            tracked = processes.setdefault(key, process)
                            cpu.append(tracked.cpu_percent())
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                    sample["max_process_cpu_percent"] = round(max(cpu, default=0), 1)
                    sample["process_cpu_percent"] = [round(value, 1) for value in cpu]
                    receipt["samples"].append(sample)
                    if max(cpu, default=0) >= 90 and sample["utilization_percent"] < 5:
                        idle_since = time.monotonic() if idle_since is None else idle_since
                        max_idle = max(max_idle or 0, time.monotonic() - idle_since)
                    else:
                        idle_since = None
                    if any(90 <= value <= 110 for value in cpu) and sample["utilization_percent"] < 5:
                        single_idle_since = time.monotonic() if single_idle_since is None else single_idle_since
                        max_single_idle = max(max_single_idle or 0, time.monotonic() - single_idle_since)
                    else:
                        single_idle_since = None
                    time.sleep(1)
                receipt["exit_code"] = child.wait()
        finally:
            receipt["wall_seconds"] = round(time.monotonic() - started, 3)
            receipt["device_after"] = device()
            samples = [before, receipt["device_after"], *receipt["samples"]]
            receipt["peak_device_vram_mib"] = max(s["used_mib"] for s in samples)
            receipt["peak_vram_mib"] = (max(0, receipt["peak_device_vram_mib"] - before["used_mib"])
                                        if args.vram_mib else 0)
            receipt["peak_vram_method"] = "device used minus baseline, sampled each second" if args.vram_mib else "CPU-only command"
            receipt["cpu_busy_gpu_idle_max_seconds"] = round(max_idle or 0, 3)
            receipt["single_core_cpu_busy_gpu_idle_max_seconds"] = round(max_single_idle or 0, 3)
            args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
            ledger("release")
    print(json.dumps({k: v for k, v in receipt.items() if k != "samples"}, indent=2))
    return receipt["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
