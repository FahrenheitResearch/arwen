"""CPU-only emitter microbenchmark, separate from forecast overhead qualification."""

import argparse
import json
import os
from pathlib import Path
import time

from gpuwm.timeline import Ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--events", type=int, default=20000)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    path = args.outdir / "microbenchmark.jsonl"
    if path.exists():
        raise SystemExit("refusing to overwrite an existing measurement")
    with (args.outdir / "MANIFEST.txt").open("a") as manifest:
        manifest.write(str(path) + "\n")
    ledger = Ledger(path, run_id="emitter_microbenchmark")
    start = time.perf_counter()
    for index in range(args.events):
        if not ledger.emit("boot.emitter_benchmark", attrs={"index": index}):
            raise SystemExit("emitter failed; no throughput result")
    duration = time.perf_counter() - start
    size = path.stat().st_size
    print(json.dumps({"kind": "emitter_microbenchmark", "events": args.events,
                      "seconds": duration, "microseconds_per_event": duration * 1e6 / args.events,
                      "bytes": size, "pid": os.getpid(), "forecast_overhead": None}))
    with (args.outdir / "deleted-files.jsonl").open("a") as receipt:
        receipt.write(json.dumps({"path": str(path), "bytes": size,
                                 "reason": "throughput recorded"}) + "\n")
    path.unlink()


if __name__ == "__main__":
    main()
