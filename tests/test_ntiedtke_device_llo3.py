"""Deferred integer masks must preserve the hoist check and suppress errors."""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.core.ntiedtke import NT_STAGE_POSTRUN, NtPipeline


@pytest.mark.parametrize("ncol", [7, 129])
def test_deferred_masks_preserve_clear_and_active_chunks(ncol):
    cp = pytest.importorskip("cupy")
    pipeline = NtPipeline(ncol=ncol, nz=7, dt=15.0)
    labels = pipeline.w.bind("klab", 1)
    pipeline.begin_llo3_batch()
    labels.fill(0)
    pipeline.reduce_llo3()
    labels[7, -1] = np.int32(2)
    pipeline.reduce_llo3()
    masks = cp.asnumpy(cp.stack(pipeline._llo3_pending))
    np.testing.assert_array_equal(masks, np.array([0, 3], dtype=np.int32))
    pipeline.end_llo3_batch()
    assert pipeline._llo3_pending is None
    assert pipeline.scalars["llo3"] == np.int32(1)
    assert int(pipeline.stages._llo3_mask[0]) == 0


@pytest.mark.parametrize("ncol", [7, 129])
def test_invalid_mask_blocks_later_kernels_before_float_work(ncol):
    cp = pytest.importorskip("cupy")
    pipeline = NtPipeline(ncol=ncol, nz=7, dt=15.0)
    labels = pipeline.w.bind("klab", 1)
    labels.fill(0)
    labels[2, -1] = np.int32(1)
    output = pipeline.w.bind("rthcuten", 0)
    output.fill(np.float32(7.0))
    pipeline.begin_llo3_batch()
    pipeline.reduce_llo3()
    assert int(pipeline.stages._llo3_mask[0]) == 2
    pipeline.run_stage("ntiedtke_post_run")
    actual = cp.asnumpy(output).view(np.uint32)
    expected = np.full(output.shape, np.float32(7.0)).view(np.uint32)
    np.testing.assert_array_equal(actual, expected)
    assert int(pipeline.stages.order_report[NT_STAGE_POSTRUN]) == -1
    with pytest.raises(ValueError, match="llo3's hoist is unsound"):
        pipeline.end_llo3_batch()
    assert pipeline._llo3_pending is None
    assert int(pipeline.stages._llo3_mask[0]) == 0


def test_immediate_hoist_validation_is_preserved():
    pytest.importorskip("cupy")
    pipeline = NtPipeline(ncol=7, nz=7, dt=15.0)
    labels = pipeline.w.bind("klab", 1)
    labels.fill(0)
    assert pipeline.reduce_llo3() == 0
    labels[7, -1] = np.int32(2)
    assert pipeline.reduce_llo3() == 1
    labels[7, -1] = np.int32(0)
    labels[2, -1] = np.int32(1)
    with pytest.raises(ValueError, match="llo3's hoist is unsound"):
        pipeline.reduce_llo3()
