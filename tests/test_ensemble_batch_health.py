"""Health allocation and reduction-tree preservation without a GPU."""

from pathlib import Path

import pytest

from gpuwm.ensemble.batch_health import stability_memory_plan, strided_health_source, _layout_words
from gpuwm.ensemble.batch_kernel import KernelSpec, PointerSpec, generate_batch_source


def test_health_plan_uses_each_scalar_domains_original_partition():
    plan, blocks = stability_memory_plan(members=10, u_shape=(49, 150, 151),
                                        w_shape=(50, 150, 150), theta_shape=(49, 150, 150))
    assert blocks == 256
    rows = {row["name"]: row for row in plan.inventory(10)}
    assert rows["health:partial"]["shape"] == (10, 256, 9)
    assert rows["health:result"]["shape"] == (10, 8)
    assert all(row["ownership"] == "member" for row in rows.values())
    assert plan.required_bytes(10) == 92672


def test_health_generated_body_retains_block_barriers_and_reduction_order():
    source = "typedef float real;\n" + (Path(__file__).parents[1] / "gpuwm/core/kernels/health.cu").read_text()
    for entry, pointers in (("health_partial", ("u", "w", "thp", "ph", "phb", "partial")),
                            ("health_final", ("partial", "result"))):
        spec = KernelSpec("health", entry, tuple(PointerSpec(name, "member") for name in pointers))
        assert generate_batch_source(source, spec, 1) == source
        batch = generate_batch_source(source, spec, 10)
        assert source.count("__syncthreads()") == batch.count("__syncthreads()")
        assert source.count("update_max(") == batch.count("update_max(")
        assert source.count("real dz =") == batch.count("real dz =")


def test_health_plan_refuses_an_empty_maximum():
    with pytest.raises(ValueError, match="no maximum identity"):
        stability_memory_plan(members=4, u_shape=(0,), w_shape=(1,), theta_shape=(1,))


def test_full_state_stride_adapter_keeps_native_checks_and_atomically_private_records():
    source = "typedef float real;\n" + (Path(__file__).parents[1] / "gpuwm/core/kernels/health.cu").read_text()
    changed = strided_health_source(source)
    assert "integer_values[physical_index] : values[physical_index]" in changed
    assert "auxiliary[member_health_offset(aux_index" in changed
    assert changed.count("atomicMin(result + 1, packed)") == source.count("atomicMin(result + 1, packed)")
    assert changed.count("value +=") == source.count("value +=")
    assert changed.count("!isfinite(value)") == source.count("!isfinite(value)")


def test_layout_words_encode_noncontiguous_member_columns_without_copying():
    import numpy as np
    packed = np.zeros((5, 4, 8, 9), np.float32)
    view = packed[:, 2]
    assert _layout_words(view) == (5, 8, 9, 288, 9, 1)
