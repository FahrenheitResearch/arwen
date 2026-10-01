"""Cached bindings must preserve live scalars, aliases and release behavior."""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.core.ntiedtke import (
    NT_STAGE_SIGNATURE, NtPipeline, NtWorkspace, _GEOMETRY_TAIL,
)


def test_cached_arguments_take_current_scalars_and_base():
    pytest.importorskip("cupy")
    pipeline = NtPipeline(ncol=8, nz=10, dt=15.0)
    stage = "ntiedtke_cuascn"
    before = pipeline.args_for(stage)
    pipeline.retime(9.25)
    pipeline.scalars["llo3"] = np.int32(1)
    after = pipeline.args_for(stage)
    names = [name for name in NT_STAGE_SIGNATURE[stage]
             if name not in _GEOMETRY_TAIL]
    for index, name in enumerate(names):
        if name in pipeline.scalars:
            assert after[index] is pipeline.scalars[name]
        else:
            assert after[index] is before[index]
    assert after[names.index("llo3")] == 1
    assert after[names.index("ztmst")] == np.float32(9.25)
    pipeline._base[stage] = 0
    shifted = pipeline.args_for(stage)
    index = names.index("ptu")
    assert shifted[index].data.ptr == pipeline.w.bind("ptu", 0).data.ptr
    assert shifted[index].data.ptr != before[index].data.ptr


def test_cached_views_keep_alias_storage_and_release_drops_references():
    pytest.importorskip("cupy")
    workspace = NtWorkspace(ncol=8, nz=10)
    view = workspace.bind("pten", 0)
    assert workspace.bind("pten", 0) is view
    assert view.data.ptr == workspace.bind("ztp1", 0).data.ptr
    workspace.bind("ztp1", 1)[2, 0] = np.float32(7.0)
    assert float(view[1, 0]) == 7.0
    del view
    allocated = workspace.bytes_allocated()
    assert workspace.release() == allocated
    assert workspace.bytes_allocated() == 0
    assert not workspace._bindings
    assert not workspace._stage_args
    with pytest.raises(KeyError):
        workspace.bind("pten", 0)
    assert workspace.release() == 0


def test_release_clears_cached_stage_arguments():
    pytest.importorskip("cupy")
    pipeline = NtPipeline(ncol=8, nz=10, dt=15.0)
    pipeline.args_for("ntiedtke_cutypen")
    assert pipeline.w._stage_args
    pipeline.w.release()
    assert not pipeline.w._stage_args
    assert not pipeline.w._bindings
    assert pipeline.w.bytes_allocated() == 0
