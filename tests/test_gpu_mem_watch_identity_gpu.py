"""Moving nvidia-smi off the main thread leaves every history byte alone.

The memory watcher observes; it never writes model state.  The fix for
the 2026-10-05 defect (the boundary ``sample()`` ran two nvidia-smi
subprocesses on the forecast's main thread every 2 s, 240 s per forecast
hour against about 24 on a busy shared box) changes WHERE and WHEN the
NVML views are read, so this pins that the history a run writes is
identical bytewise across three arms on a real GPU integration:

* no watcher at all;
* the BEFORE pattern: the NVML probes read inline by every boundary
  ``sample()`` (what the runners did until this fix);
* the AFTER pattern: the probes exactly as the runners now build them,
  NVML on its own background thread, boundary samples in-process only.

A frame is ``gpuwm.io.wrfout.state_frame`` -- the arrays a wrfout frame
carries -- taken every few steps of the WK82 moist bubble.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from conftest import requires_gpu

_STEPS = 24
_FRAME_EVERY = 6


def _run_arm(watcher_factory):
    import cupy as cp

    from gpuwm.config import validate_run_config
    from gpuwm.core.dycore import run_steps
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.io.wrfout import state_frame
    from gpuwm.verify.cases import moist_bubble
    from gpuwm.verify.cases.wk82 import wk82_sounding

    cfg = validate_run_config(moist_bubble.default_config())
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: wk82_sounding(z)[0],
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = moist_bubble.build(cfg, coord, base)
    watcher = None if watcher_factory is None else watcher_factory()
    frames = []
    if watcher is not None:
        watcher.start()
    try:
        for step in range(1, _STEPS + 1):
            run_steps(state, cfg, n=1)
            if watcher is not None:
                watcher.sample()  # the runners' step-boundary call
            if step % _FRAME_EVERY == 0:
                cp.cuda.Stream.null.synchronize()
                frames.append({name: np.array(value, copy=True)
                               for name, value in state_frame(state).items()})
    finally:
        if watcher is not None:
            watcher.stop()
            watcher.sample()  # the runners' end-of-run call
    return frames, watcher


def _before_watcher():
    """The pre-fix shape: every NVML probe read inline by sample()."""
    from gpuwm.core.gpu_mem_watch import (
        GpuPeakMemoryWatcher, default_cupy_probes, nvidia_smi_process_probes)

    inline = tuple(dataclasses.replace(probe, background_only=False)
                   for probe in nvidia_smi_process_probes())
    return GpuPeakMemoryWatcher(default_cupy_probes() + inline)


def _after_watcher():
    """Exactly the construction both prepared runners use."""
    from gpuwm.core.gpu_mem_watch import (
        GpuPeakMemoryWatcher, default_cupy_probes, nvidia_smi_process_probes)

    return GpuPeakMemoryWatcher(
        default_cupy_probes() + nvidia_smi_process_probes())


@requires_gpu
@pytest.mark.gpu
def test_history_bytes_are_identical_with_the_watcher_before_and_after():
    reference, _ = _run_arm(None)
    before, before_watch = _run_arm(_before_watcher)
    after, after_watch = _run_arm(_after_watcher)
    assert len(reference) == _STEPS // _FRAME_EVERY
    for arm_name, arm in (("before", before), ("after", after)):
        assert len(arm) == len(reference)
        for index, (want, got) in enumerate(zip(reference, arm)):
            assert want.keys() == got.keys(), (arm_name, index)
            for field, value in want.items():
                assert value.dtype == got[field].dtype
                assert value.tobytes() == got[field].tobytes(), (
                    f"{arm_name} watcher changed {field} in frame {index}")
    # Both watchers really observed the run they sat beside.
    for watch in (before_watch, after_watch):
        assert watch.peak_bytes("cuda_device_used") > 0
        assert watch.peak_bytes("cupy_pool_used") > 0
    assert after_watch.summary()["probes"]["nvml_device_used"]["read_on"] == (
        "background-thread-only")
