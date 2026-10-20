#!/usr/bin/env python3
"""Measure an owner's resident NextGen factory without instrumenting production.

Example: python tools/nextgen_speed.py --factory package.module:factory
    --steps 64 --output /data/lane/receipt.json
Or: python tools/nextgen_speed.py --settings /data/lane/prepared-stages.json
    --steps 12 --warmup 0 --output /data/lane/receipt.json
The factory accepts strict=False/True and returns a ResidentSequence. It binds
real inputs before returning, compiles eagerly, and never advances physical time.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--factory", help="importable module:callable")
    source.add_argument("--settings", type=Path,
                        help="JSON with executable standalone owner stage rows")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--strict", action="store_true", help="explicit proof numerics")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0:
        parser.error("steps must be positive; warmup must be nonnegative")
    from gpuwm import numerics

    requested_mode = "strict" if args.strict else "fast"
    numerics.select(requested_mode)
    from gpuwm.nextgen_execution import measure

    if args.settings is not None:
        from gpuwm.nextgen_execution import prepare_standalone

        settings_bytes = args.settings.read_bytes()
        settings = json.loads(settings_bytes)
        # The CLI mode switch governs compilation and the sealed driver identity.
        settings["numerics"] = requested_mode
        def build(*, strict):
            return prepare_standalone(settings, device_id=args.device)
    else:
        module, name = args.factory.rsplit(":", 1)
        build = getattr(importlib.import_module(module), name)

    def factory():
        sequence = build(strict=args.strict)
        if sequence.metadata.get("numerics") != requested_mode:
            raise ValueError("factory metadata must name the requested numerics")
        return sequence

    import cupy as cp
    with cp.cuda.Device(args.device):
        receipt = measure(factory, steps=args.steps, warmup=args.warmup)
    receipt["command"] = sys.argv
    receipt["factory"] = args.factory
    if args.settings is not None:
        receipt["settings"] = {"path": str(args.settings.resolve()),
                               "sha256": hashlib.sha256(settings_bytes).hexdigest()}
    receipt["numerics"] = numerics.receipt()
    receipt["recorded_unix_ns"] = time.time_ns()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
