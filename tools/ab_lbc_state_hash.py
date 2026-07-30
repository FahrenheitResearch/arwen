"""Hash a short configured real74 trajectory for solo-GPU trunk/branch A/B.

Run this file from the repository/worktree being tested.  Passing the same
script path from both worktrees is supported: imports are rooted at the
current directory when it contains the ``gpuwm`` package.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np


def _repo_root() -> Path:
    cwd = Path.cwd().resolve()
    return cwd if (cwd / "gpuwm").is_dir() else Path(__file__).parents[1]


sys.path.insert(0, str(_repo_root()))


PROGNOSTIC_NAMES = (
    "u", "v", "w", "thp", "php", "mup",
    "qv", "qc", "qr", "qi", "qs", "qg",
    "nc", "nr", "ni", "ns", "ng",
)


def _digest(array) -> str:
    import cupy as cp

    host = np.ascontiguousarray(cp.asnumpy(array))
    if host.dtype != np.float32:
        raise TypeError(f"expected FP32 prognostic, got {host.dtype}")
    return hashlib.sha256(host.tobytes(order="C")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "steps", type=int,
        help="number K of real74 internal dynamics steps (7.5 s each)")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be >= 1")

    import cupy as cp

    from gpuwm.core.dycore import step
    from gpuwm.verify.cases.real74_d01 import (
        phase3_config,
        phase3_integration_config,
        prepare_phase3_case,
    )

    prepared = prepare_phase3_case(phase3_config())
    cfg = phase3_integration_config(prepared.cfg)
    state = prepared.initial_result.state
    # Match _integrate_configured_case's fast-physics clock setup.
    state.physics.bldt_seconds = cfg.dt
    state.physics.stepbl = 1
    for _ in range(args.steps):
        step(state, cfg)
    cp.cuda.runtime.deviceSynchronize()

    print(f"steps={args.steps} dt={cfg.dt} elapsed={state.elapsed_seconds}")
    seen = set()
    for name in PROGNOSTIC_NAMES:
        array = getattr(state, name, None)
        if array is None:
            continue
        seen.add(id(array))
        print(f"state.{name} {_digest(array)}")

    # Surface/physics arrays also evolve and feed later atmospheric steps.
    # Hash every unique FP32 array owned directly or in the driver's field
    # registry so an A/B cannot miss a latent divergence outside DomainState.
    driver = state.physics
    for owner_name, owner in (("physics", vars(driver)),
                              ("physics.fields", driver.fields)):
        for name, array in sorted(owner.items()):
            if (not isinstance(array, cp.ndarray)
                    or array.dtype != cp.float32 or id(array) in seen):
                continue
            seen.add(id(array))
            print(f"{owner_name}.{name} {_digest(array)}")


if __name__ == "__main__":
    main()
