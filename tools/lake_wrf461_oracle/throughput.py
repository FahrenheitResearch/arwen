"""Lake kernel throughput at forecast scale: the regime columns tiled to
``--columns`` lake columns, timed over ``--steps`` steps after one warm-up.

    python throughput.py OUT [--columns 32768] [--steps 60]

Run it once with PYTHONPATH naming the tree to time.  Prints milliseconds
per step and per million column-steps.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--columns", type=int, default=32768)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--campaign", default="dt60")
    a = p.parse_args()
    import cupy as cp
    import gpuwm  # noqa: F401
    from gpuwm.core.kernels import get_kernel
    from gpuwm.core.lake import lake_arena, launch_lake_columns
    from gpuwm.core.lake_schema import LAKE_OUTPUT_WORDS, LAKE_STATE_WORDS, LAKE_STATIC_WORDS
    data = np.load(a.out / f"inputs-{a.campaign}.npz")
    seed0, forcing0, dt = data["seed"], data["forcing"], np.float32(data["dt"])
    reps = -(-a.columns // seed0.shape[1])
    seed = np.tile(seed0, (1, reps))[:, : a.columns]
    n = seed.shape[1]
    init, step = get_kernel("lake", "lake_init_columns"), get_kernel("lake", "lake_step_columns")
    state = cp.zeros((LAKE_STATE_WORDS, n), cp.float32)
    static = cp.zeros((LAKE_STATIC_WORDS, n), cp.float32)
    output = cp.zeros((LAKE_OUTPUT_WORDS, n), cp.float32)
    errors = cp.zeros(n, cp.int32)
    arena = lake_arena(n)
    thr = np.float32(0.5)
    launch_lake_columns(init, n, (n, cp.asarray(seed), state, static, 1, 1, np.float32(50), thr, errors), arena)
    forcing = [cp.asarray(np.ascontiguousarray(np.tile(forcing0[s], (1, reps))[:, :n])) for s in range(a.steps + 1)]
    launch_lake_columns(step, n, (n, forcing[0], state, static, output, dt, thr, errors), arena)
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    for s in range(1, a.steps + 1):
        launch_lake_columns(step, n, (n, forcing[s], state, static, output, dt, thr, errors), arena)
    cp.cuda.Device().synchronize()
    sec = time.perf_counter() - t0
    assert not cp.asnumpy(errors).any()
    print(json.dumps(dict(columns=n, steps=a.steps, ms_per_step=round(1e3 * sec / a.steps, 3),
                          seconds_per_million_column_steps=round(sec / (n * a.steps) * 1e6, 3),
                          device=cp.cuda.runtime.getDeviceProperties(0)["name"].decode())))


if __name__ == "__main__":
    main()
