"""Solo-GPU bitwise/performance A/B for configured real74 dynamics steps.

Run the same command from trunk and the candidate worktree.  Each requested
step is one configured 7.5-second internal real74 dynamics/physics step; setup
time is excluded.  The script synchronizes around every step, prints its wall
time, then hashes the final C-order FP32 bytes of every active prognostic.

Usage:  python tools/ab_dyn_state_hash.py 4
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import time

# Resolve gpuwm from the checkout used as the current working directory.  This
# lets one absolute copy of this branch's tool drive both trunk and candidate.
sys.path.insert(0, str(Path.cwd()))

from gpuwm.core.moist import moist_species
from gpuwm.verify.cases.real74_d01 import (
    phase3_config, phase3_integration_config, prepare_phase3_case)


_DRY_PROGNOSTICS = ("u", "v", "w", "thp", "php", "mup")


def _head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short=7", "HEAD"], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "steps", type=int,
        help="number of configured 7.5-second internal real74 steps")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be >= 1")

    import cupy as cp
    from gpuwm.core.dycore import step

    configured = phase3_config()
    prepared = prepare_phase3_case(configured)
    integration_cfg = phase3_integration_config(prepared.cfg)
    state = prepared.initial_result.state
    state.physics.bldt_seconds = integration_cfg.dt
    state.physics.stepbl = 1

    print(f"HEAD {_head()}")
    print(f"CONFIGURED_DT_SECONDS {integration_cfg.dt:.17g}")
    cp.cuda.runtime.deviceSynchronize()
    for index in range(args.steps):
        started = time.perf_counter()
        step(state, integration_cfg)
        cp.cuda.runtime.deviceSynchronize()
        elapsed = time.perf_counter() - started
        print(f"STEP {index + 1} WALL_SECONDS {elapsed:.9f}", flush=True)

    for name in _DRY_PROGNOSTICS + moist_species(state):
        array = getattr(state, name)
        if array.dtype != cp.float32:
            raise TypeError(f"{name} is {array.dtype}, expected float32")
        host = cp.asnumpy(array)
        digest = hashlib.sha256(host.tobytes(order="C")).hexdigest()
        print(f"HASH {name} {digest}")


if __name__ == "__main__":
    main()
