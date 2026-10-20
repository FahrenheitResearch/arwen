#!/usr/bin/env python3
"""Command orchestration and passive CPU/VRAM receipts; no weather processing."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import subprocess
import time


def process_sample(root):
    table = {}
    hz = os.sysconf("SC_CLK_TCK")
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            table[int(path.parent.name)] = (int(fields[1]),
                (int(fields[11]) + int(fields[12])) / hz, int(fields[17]))
        except (OSError, ValueError, IndexError):
            pass
    found = {root}
    while True:
        expanded = found | {pid for pid, row in table.items() if row[0] in found}
        if expanded == found:
            break
        found = expanded
    return [{"pid": pid, "cpu_seconds": table[pid][1], "threads": table[pid][2]}
            for pid in sorted(found) if pid in table]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.receipt.exists():
        parser.error("provide a command and a new receipt path")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    clock = time.monotonic()
    samples = []
    with args.receipt.with_suffix(".log").open("x") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            row = {"elapsed_s": time.monotonic()-clock,
                   "processes": process_sample(process.pid)}
            try:
                query = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
                row["gpu_csv"] = query.stdout.strip().splitlines()
            except (OSError, subprocess.TimeoutExpired) as error:
                row["gpu_query_error"] = str(error)
            samples.append(row)
            # Incremental evidence survives interruption or loss of the box.
            with args.receipt.with_suffix(".samples.jsonl").open("a") as stream:
                stream.write(json.dumps(row)+"\n")
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt = {"schema": "trailer-command-measurement-v1", "command": command,
        "cwd": str(Path.cwd()), "started_utc": started,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.monotonic()-clock, "returncode": process.returncode,
        "child_user_seconds": after.ru_utime-before.ru_utime,
        "child_system_seconds": after.ru_stime-before.ru_stime,
        "child_maxrss_kib": after.ru_maxrss,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_mutex_cards": os.environ.get("GPU_MUTEX_CARDS"),
        "sampling": "passive /proc and nvidia-smi every five seconds; no device synchronization",
        "samples": samples}
    args.receipt.write_text(json.dumps(receipt, indent=2)+"\n")
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
