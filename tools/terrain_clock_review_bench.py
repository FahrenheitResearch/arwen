"""Time the actual default streamed guard on full arrays, with no GPU work.

Run from the engine root with PYTHONPATH=.:tests. The deterministic input
fixture uses the real coupled boundary builder and prepared-cache reader.
Only timings and receipts are written, never model data.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import sys

# This checkout-only review uses the same test fixture as its declared
# PYTHONPATH=.:tests invocation, without depending on the caller's path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from gpuwm import terrain_clock as tc
from gpuwm.prepared_domain_tree_forecast import StreamedClockGuard
from test_terrain_clock_review import _stream_fixture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=1901)
    parser.add_argument("--ny", type=int, default=1301)
    parser.add_argument("--uncached", action="store_true")
    args = parser.parse_args()
    basis, bounds = _stream_fixture(args.ny, args.nx)
    starts = {1: tc.start_winds_from_cache(basis.readers[1], "d01")}
    initial = tc.clock_for_domains(basis.experiment, basis.acoustic,
        statics=basis.statics, starts=starts, announce=False)[1]
    guard = StreamedClockGuard(basis, run_clock=tc.clock_receipt(initial),
                               root_grid_id=1, run_seconds=10800.)
    if args.uncached:
        guard.local_cache = None
    samples = []
    for interval in bounds.intervals:
        before = time.perf_counter()
        guard(interval)
        samples.append(time.perf_counter() - before)
    print(json.dumps({"nx": args.nx, "ny": args.ny,
        "faces": args.ny * (args.nx - 1) + (args.ny - 1) * args.nx,
        "default_clock": basis.experiment.root.run.terrain_clock,
        "cached": not args.uncached, "interval_seconds": samples,
        "checked": guard.checked,
        "clock_sha256": hashlib.sha256(json.dumps(guard.expected,
            sort_keys=True).encode()).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
