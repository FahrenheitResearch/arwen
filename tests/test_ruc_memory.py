"""RUC admission counts the actual soil/snow workspace and output lifetimes."""

import pytest

from gpuwm.core import ruc_memory as memory
from gpuwm.core.ruc_sfctmp_layout import _SFCTMP_SLOTS


@pytest.mark.parametrize("nzs", [6, 9])
@pytest.mark.parametrize("ncol", [0, 1, 65536, 65537, 500000])
def test_generated_soil_and_snow_slots_are_all_priced(nzs, ncol):
    offsets, count = memory.sfctmp_scratch_layout(ncol, nzs)
    expected = 0
    for offset, kind in zip(offsets, _SFCTMP_SLOTS):
        assert offset == expected
        expected += ncol * (nzs if kind == 9 else 1)
    assert count == expected
    assert memory.sfctmp_workspace_allocations(ncol, nzs)["scratch"] == ((expected,), "float32")


@pytest.mark.parametrize("nzs", [6, 9])
def test_full_width_solve_prices_outputs_alongside_both_workspaces(nzs):
    ncol = 500000
    receipt = memory.ruc_runtime_memory_bytes(ncol, nzs)
    assert receipt["sfctmp_workspace"] == memory.allocation_bytes(
        memory.sfctmp_workspace_allocations(ncol, nzs))
    assert receipt["sfctmp_outputs"] == memory.allocation_bytes(
        memory.sfctmp_output_allocations(ncol, nzs))
    assert receipt["driver_workspace"] == memory.allocation_bytes(
        memory.driver_workspace_allocations(ncol, nzs))
    assert sum(receipt.values()) > receipt["sfctmp_workspace"] + receipt["sfctmp_outputs"]


def test_invalid_soil_ladder_cannot_silently_price_a_different_kernel():
    with pytest.raises(ValueError, match="6 or 9 soil levels"):
        memory.ruc_runtime_memory_bytes(100, 4)


@pytest.mark.parametrize("nzs", [6, 9])
def test_pinned_host_mirrors_cover_the_three_column_fields_and_uploads(nzs):
    ncol = 500001
    allocations = memory.ruc_pinned_host_allocations(ncol, nzs)
    for name in ("driver_psfc", "driver_scale", "driver_inverse"):
        assert allocations[name] == ((ncol,), "float32")
    assert allocations["sfctmp_pointer_upload"] == allocations["sfctmp_pointers"]
    assert allocations["sfctmp_error_reset"] == allocations["sfctmp_reset"]
    assert memory.ruc_pinned_host_bytes(ncol, nzs) > 3 * ncol * 4
