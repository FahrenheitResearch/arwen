"""Fusion must preserve the standalone parcel buffers and stage reports."""
import numpy as np
import pytest


def test_fused_group_preserves_full_fixture_workspace(monkeypatch):
    cp = pytest.importorskip("cupy")
    import test_ntiedtke_pipeline_boundaries as fixture
    from gpuwm.core import ntiedtke
    from gpuwm.core.ntiedtke_fused import run_fused, source_plan

    reference, _ = fixture.pipeline.__wrapped__()
    launch = ntiedtke.NtPipeline.run_stage
    _, _, group = source_plan()

    def grouped(pipeline, stage):
        if stage == group[0]:
            run_fused(pipeline)
        elif stage not in group:
            launch(pipeline, stage)

    monkeypatch.setattr(ntiedtke.NtPipeline, "run_stage", grouped)
    candidate, _ = fixture.pipeline.__wrapped__()
    for old, new in ((reference.w._level, candidate.w._level),
                     (reference.w._surface, candidate.w._surface)):
        assert old.keys() == new.keys()
        for name in old:
            np.testing.assert_array_equal(cp.asnumpy(old[name]).view(np.uint32),
                                          cp.asnumpy(new[name]).view(np.uint32),
                                          err_msg=name)
    ids = {"ntiedtke_" + name: i for i, name in ntiedtke.NT_STAGE_NAMES.items()}
    candidate.stages.check_order([ids[stage] for stage in ntiedtke.NT_CALL_ORDER])
    np.testing.assert_array_equal(cp.asnumpy(reference.stages.geom_report),
                                  cp.asnumpy(candidate.stages.geom_report))
    np.testing.assert_array_equal(cp.asnumpy(reference.stages.order_report),
                                  cp.asnumpy(candidate.stages.order_report))
    reference.w.release()
    candidate.w.release()


def test_small_chunk_uses_only_standalone_stages(monkeypatch):
    from types import SimpleNamespace
    from gpuwm.core import ntiedtke

    calls = []
    pipeline = ntiedtke.NtPipeline.__new__(ntiedtke.NtPipeline)
    pipeline.w = SimpleNamespace(ncol=16384)
    pipeline.zero_run_head = lambda: None
    pipeline.snapshot_forcing = lambda: None
    pipeline.snapshot_momentum = lambda: None
    pipeline.reduce_llo3 = lambda: None
    pipeline.run_stage = calls.append
    monkeypatch.setattr(ntiedtke, "_nt_fused_walk",
                        lambda p: pytest.fail("small chunks must avoid fusion"))
    pipeline.run_chunk()
    assert tuple(calls) == ntiedtke.NT_CALL_ORDER
