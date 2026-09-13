"""Launch-grid sizing follows device capacity and the actual field lengths."""
import pytest


def test_device_metadata_is_cached_separately_per_device(monkeypatch):
    from gpuwm.core import mynn_pbl_gpu as gpu
    calls = []
    def properties(device):
        calls.append(device)
        return {"multiProcessorCount": (17, 73)[device]}
    gpu._validation_block_budget.cache_clear()
    monkeypatch.setattr(gpu.cp.cuda.runtime, "getDeviceProperties", properties)
    try:
        assert gpu._validation_block_budget(0) == 544
        assert gpu._validation_block_budget(1) == 2336
        assert gpu._validation_block_budget(0) == 544
        assert gpu._validation_block_budget(1) == 2336
        assert calls == [0, 1]
    finally:
        gpu._validation_block_budget.cache_clear()


@pytest.mark.parametrize("longest,count,expected", [(0, 6, 1), (1, 50, 1), (129, 6, 2),
    (100003, 6, 97), (100003, 50, 11), (100003, 64, 9)])
def test_grid_shares_the_device_budget_and_caps_short_inputs(monkeypatch, longest, count, expected):
    from gpuwm.core import mynn_pbl_gpu as gpu
    seen = []
    def budget(device):
        seen.append(device)
        return 584
    monkeypatch.setattr(gpu, "_validation_block_budget", budget)
    assert gpu._validation_grid_blocks(longest, count, 3) == expected
    assert seen == [3]
