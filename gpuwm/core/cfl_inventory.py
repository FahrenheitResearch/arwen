"""Runtime-free inventory of the per-grid WRF CFL diagnostic ring."""
from __future__ import annotations

import os

WRF_CFL_SLOTS = 32768
CFL_HIST_BINS = 32
WRF_CFL_WORDS = 4 + CFL_HIST_BINS
WRF_CFL_SHAPE = (WRF_CFL_SLOTS, WRF_CFL_WORDS)
WRF_CFL_BYTES = WRF_CFL_SLOTS * WRF_CFL_WORDS * 4


def wrf_cfl_recording_requested(cfg=None, *, adaptive=None) -> bool:
    """The run controller or explicit diagnostic switch enables recording.

    Adaptive stepping is experiment-wide. A caller pricing an experiment
    passes its root switch, just as the model driver uses the root switch.
    Standalone domain/planner callers use their inherited RunConfig.
    """
    if adaptive is None:
        adaptive = bool(getattr(cfg, "use_adaptive_time_step", False))
    value = os.environ.get("GPUWM_WRF_CFL_PROBE", "")
    return bool(adaptive) or value.strip().lower() not in ("", "0", "false", "no", "off")


def fold_cfl_words(rows):
    """Fold CUDA atomic words: unsigned maxima and modular uint32 sums."""
    import numpy as np
    values = np.asarray(rows, dtype=np.uint32)
    if values.ndim < 2 or values.shape[-1] != WRF_CFL_WORDS or not len(values):
        raise ValueError("CFL fold needs nonempty rows of WRF_CFL_WORDS words")
    folded = values.sum(axis=0, dtype=np.uint32)
    folded[..., 0] = values[..., 0].max(axis=0)
    folded[..., 3] = values[..., 3].max(axis=0)
    return folded
