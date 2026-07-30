"""The per-land-column cost of a Noah-MP call, measured at several widths.

Two things are measured here and they answer different questions.

``test_the_column_cost_is_flat_or_it_is_not`` is the **width sweep**.  A
per-column cost that does not fall as columns are added is the signature of
per-column host work: it is not a launch cost, not a transfer cost and not an
under-occupied kernel.  That is how the original 7.24 ms/column diagnosis was
made, and it is the only measurement that can say whether a conversion moved
work to the device *at width* rather than merely on one column.  It runs both
modes on the same process and the same inputs -- the paired host authority
(:func:`gpuwm.core.noahmp_runtime.evaluate_leaf_batch_on_host`, the CPython
leaves the WRF fixtures pin) and the device batches -- so the ratio is not
contaminated by machine differences.

It is **opt-in**: ``GPUWM_NOAHMP_WIDTH_SWEEP=1``.  At 352 land columns the host
authority alone is several seconds per step, so this is not something the
ordinary suite should pay for.  ``GPUWM_NOAHMP_WIDTH_MAX`` caps the largest
grid and ``GPUWM_NOAHMP_WIDTH_REPS`` the repetition count.

Absolute milliseconds are a property of the *machine*, not of this port; the
project has already recorded different figures for the local RTX 5090 and a
rented Linux one.  Print the host and device figures together, name the box,
and do not merge two boxes into one number.
"""

from __future__ import annotations

import os
import time

import pytest

from conftest import requires_gpu

import cupy as cp

from test_noahmp_runtime import _build

#: (nx, ny) and the land-column count each produces with no water columns.
#: The historical series this port has always reported.
WIDTHS = ((2, 1), (4, 2), (8, 6), (16, 10), (22, 16))

_SWEEP_ENV = "GPUWM_NOAHMP_WIDTH_SWEEP"


def _requires_sweep():
    if os.environ.get(_SWEEP_ENV, "") != "1":
        pytest.skip(f"set {_SWEEP_ENV}=1 to run the width sweep")


def _time_one(nx, ny, *, reps, evaluator):
    """Warmed wall clock per land column for one grid and one leaf evaluator."""
    import gpuwm.core.noahmp_runtime as runtime_module
    from gpuwm.core.dycore import step

    previous = runtime_module.LEAF_BATCH_EVALUATOR
    runtime_module.LEAF_BATCH_EVALUATOR = evaluator
    try:
        state, cfg, driver = _build(nx=nx, ny=ny, water_columns=0)
        step(state, cfg)                    # warm the parameter-transfer cache
        columns = driver.last_noahmp_census["land"]
        assert columns == nx * ny, (columns, nx * ny)
        cp.cuda.runtime.deviceSynchronize()
        started = time.perf_counter()
        for _ in range(reps):
            step(state, cfg)
        cp.cuda.runtime.deviceSynchronize()
        elapsed = time.perf_counter() - started
    finally:
        runtime_module.LEAF_BATCH_EVALUATOR = previous
    return columns, elapsed / (reps * columns)


@requires_gpu
def test_the_column_cost_is_flat_or_it_is_not(capsys):
    _requires_sweep()
    import gpuwm.core.noahmp_runtime as runtime_module

    limit = int(os.environ.get("GPUWM_NOAHMP_WIDTH_MAX", "352"))
    reps = int(os.environ.get("GPUWM_NOAHMP_WIDTH_REPS", "3"))

    rows = []
    for nx, ny in WIDTHS:
        if nx * ny > limit:
            continue
        columns, host = _time_one(
            nx, ny, reps=reps,
            evaluator=runtime_module.evaluate_leaf_batch_on_host)
        _columns, device = _time_one(
            nx, ny, reps=reps,
            evaluator=runtime_module.evaluate_leaf_batch_on_device)
        rows.append((columns, host * 1e3, device * 1e3))

    with capsys.disabled():
        print("\n| land columns | host authority ms/col | device leaves ms/col"
              " | speedup |")
        print("|---:|---:|---:|---:|")
        for columns, host, device in rows:
            print(f"| {columns} | {host:.2f} | {device:.2f} | "
                  f"{host / device:.2f}x |")

    assert rows, "the sweep produced no widths"
    for columns, host, device in rows:
        assert device > 0.0 and host > 0.0, (columns, host, device)

# The cheap always-on bracket on the per-column cost is
# ``tests/test_noahmp_runtime.py::test_the_column_cost_is_what_the_registry_says``
# and is deliberately not duplicated here.  Two overlapping brackets on the
# same quantity is one more thing that can drift apart.
