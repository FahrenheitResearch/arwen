"""The executor's loop holds nothing per step.

The 2026-09-15 VRAM-growth lane found no leak in the engine: on the 5 h
nested pair the forecast process sat flat at 8.2-8.6 GB (NVML
per-process) while a second process filled the card.  This pin holds the
executor to that going forward: with a dycore step that allocates
transient arrays on every call, and a history handler that
receives every frame, the number of live NumPy arrays and the bytes they
hold, read after each root step once the loop is warm, must not move.
A per-step accumulation anywhere in ``execute_experiment`` -- a retained
history frame, a validation copy kept for a later pass, a diagnostic
appended per period -- would show as a rising series here.

CPU tier: the host-only fixture of ``tests/test_model.py``; the same loop
runs the CuPy state on the card, where the receipt's ``cupy_pool_used``
and NVML per-process peaks carry the equivalent reading.
"""

from __future__ import annotations

import gc

import numpy as np

from gpuwm.core.model import execute_experiment
from test_model import _model


def _live_numpy():
    gc.collect()
    arrays = [obj for obj in gc.get_objects()
              if isinstance(obj, np.ndarray)]
    return len(arrays), sum(int(a.nbytes) for a in arrays)


def test_live_array_count_and_bytes_stay_flat_across_root_steps(monkeypatch):
    _exp, model = _model()
    frames = []

    def allocating_step(state, cfg, *, refl_10cm_due=False, **kwargs):
        # Transients the size of a small field, dropped on return: what a
        # scheme's per-step scratch looks like to the loop.
        scratch = [np.ones((64, 64), dtype=np.float32) for _ in range(8)]
        state.elapsed_seconds += 0.0 * float(scratch[0][0, 0])

    monkeypatch.setattr("gpuwm.core.dycore.step", allocating_step)

    series = []

    def observe(**event):
        if event["grid_id"] == model.root.cfg.grid_id:
            series.append(_live_numpy())

    def history_handler(tree, node, ticks):
        # The handler sees a fresh array per frame and keeps only a
        # digest of it, the way a writer stages then releases a frame.
        frame = np.full((16, 16), float(ticks), dtype=np.float32)
        frames.append(float(frame.sum()))

    report = execute_experiment(
        model, validate_state=False, pool_trim_per_period=False,
        history_handler=history_handler, step_observer=observe)
    assert report.steps == 20
    assert len(series) == 10, "one reading per root step"
    assert frames, "the history handler ran"
    warm = series[2:]
    counts = {count for count, _ in warm}
    nbytes = {size for _, size in warm}
    assert len(counts) == 1, f"live NumPy array count moved: {series}"
    assert len(nbytes) == 1, f"live NumPy bytes moved: {series}"
