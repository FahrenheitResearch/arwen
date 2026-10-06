"""Allocation and alias contracts for packed native MYNN leaves."""

from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core.physics_inventory import MYNN_SURFACE_OUTPUTS
from gpuwm.ensemble.batch_mynn import (
    _pbl_options, _DeclaredScratchState, _validate_surface_aliases,
    mynn_pbl_output_plan, mynn_pbl_transient_bytes,
    mynn_predict_output_plan, mynn_surface_output_plan)
from gpuwm.ensemble.batch_storage import BatchStorage


@pytest.mark.parametrize("members", [2, 4, 8])
def test_surface_plan_prices_every_result(members):
    plan = mynn_surface_output_plan(ny=3, nx=5, reserved_bytes=4096)
    storage = BatchStorage(plan, members, array_module=np, available_bytes=1 << 20)
    assert {spec.name for spec in plan.arrays} == {f"mynn_surface:{name}" for name in MYNN_SURFACE_OUTPUTS}
    assert storage.payload_bytes == len(MYNN_SURFACE_OUTPUTS) * members * 3 * 5 * 4
    assert plan.required_bytes(members) >= storage.payload_bytes + 4096
    assert all(spec.ownership == "member" for spec in plan.arrays)


@pytest.mark.parametrize("members", [2, 4, 8])
def test_predictor_plan_prices_all_four_products_and_ten_vectors(members):
    plan = mynn_predict_output_plan(nz=12, ny=2, nx=5, reserved_bytes=8192)
    storage = BatchStorage(plan, members, array_module=np, available_bytes=1 << 20)
    assert storage.payload_bytes == (4 + 10) * members * 12 * 2 * 5 * 4
    assert plan.required_bytes(members) >= storage.payload_bytes + 8192
    with pytest.raises(MemoryError, match="exhaust device memory"):
        BatchStorage(plan, members, array_module=np, available_bytes=plan.required_bytes(members) - 1)


@pytest.mark.parametrize("bad", [0, -1, True])
def test_invalid_extent_refuses_before_gpu(bad):
    with pytest.raises((TypeError, ValueError)):
        mynn_surface_output_plan(ny=bad, nx=5)
    with pytest.raises((TypeError, ValueError)):
        mynn_predict_output_plan(nz=12, ny=2, nx=bad)


def pointer(address, size=64):
    return SimpleNamespace(data=SimpleNamespace(ptr=address), nbytes=size)


def test_native_same_name_aliases_and_read_only_input_aliases_are_valid():
    ust, mol, ustm = pointer(1000), pointer(2000), pointer(3000)
    _validate_surface_aliases({"ust": ust, "u1": pointer(4000), "v1": pointer(4000)},
                             {"ust": ust, "mol": mol, "ustm": ustm}, mol=mol, ustm=ustm)


@pytest.mark.parametrize("output", [pointer(1010), pointer(1000, 32)])
def test_partial_same_field_alias_refuses_a_destructive_store(output):
    with pytest.raises(ValueError, match="overlaps.*overwrite a different field"):
        _validate_surface_aliases({"ust": pointer(1000)}, {"ust": output},
                                 mol=pointer(2000), ustm=pointer(3000))


def test_cross_field_output_alias_refuses_before_a_launch():
    with pytest.raises(ValueError, match="output ust overlaps output mol"):
        _validate_surface_aliases({}, {"ust": pointer(1000), "mol": pointer(1000)},
                                 mol=pointer(2000), ustm=pointer(3000))


@pytest.mark.parametrize("members", [2, 4, 8])
def test_pbl_plan_covers_complete_native_scratch_and_six_member_rates(members):
    from gpuwm.core.mynn_pbl_scratch import (
        mynn_pbl_scratch_bytes, mynn_pbl_scratch_shapes,
        mynn_pbl_index_shapes, mynn_pbl_flag_shapes)
    plan = mynn_pbl_output_plan(nz=50, ny=6, nx=8, members=members, column_chunk=61)
    storage = BatchStorage(plan, members, array_module=np, available_bytes=1 << 28)
    slots = {*mynn_pbl_scratch_shapes(61, 50), *mynn_pbl_index_shapes(61, 50), *mynn_pbl_flag_shapes()}
    assert slots <= {name.removeprefix("mynn_pbl:") for name in storage.arrays}
    assert storage.payload_bytes == mynn_pbl_scratch_bytes(61, 50) + 6 * members * 50 * 6 * 8 * 4
    wide = mynn_pbl_output_plan(nz=50, ny=6, nx=8, members=members, column_chunk=10000)
    assert sum(spec.slab_bytes for spec in wide.arrays if spec.ownership == "shared") == mynn_pbl_scratch_bytes(members * 48, 50)


@pytest.mark.parametrize("options", [{"bl_mynn_mixscalars": 1}, {"closure": 3},
    {"bl_mynn_cloudpdf": 1}, {"bl_mynn_mixlength": True}, {"spp_pbl": 1}])
def test_pbl_unbound_or_unimplemented_option_refuses_before_state_staging(options):
    with pytest.raises(ValueError, match="MYNN PBL"):
        _pbl_options(options)
    assert _pbl_options({"bl_mynn_mixlength": 2})["bl_mynn_mixlength"] == 2


def test_fixed_scratch_owner_refuses_hidden_growth_or_new_slots():
    owner = _DeclaredScratchState({"field": np.zeros((12,), np.float32)})
    assert owner.scratch((12,), "field") is owner.buffers["field"]
    for shape, slot, dtype in (((13,), "field", np.float32), ((12,), "missing", np.float32),
                               ((12,), "field", np.int32)):
        with pytest.raises(ValueError, match="absent from its allocation plan"):
            owner.scratch(shape, slot, dtype)


@pytest.mark.parametrize("members", [2, 4, 8])
def test_gsd_plan_prices_ten_plumes_and_both_simultaneous_gsd_working_sets(members):
    from gpuwm.core.mynn_pbl_scratch import (
        mynn_pbl_scratch_bytes, mynn_pbl_scratch_shapes, SLOT_PLUME_WORK, SLOT_PLUME_SCRATCH,
        SLOT_GSD41_CONDENSATION_WORK, SLOT_GSD41_THVL)
    nz, chunk = 50, 61
    plan = mynn_pbl_output_plan(nz=nz, ny=6, nx=8, members=members, column_chunk=chunk,
                                bl_mynn_version="gsd_41", reserved_bytes=8192)
    storage = BatchStorage(plan, members, array_module=np, available_bytes=1 << 28)
    shapes = mynn_pbl_scratch_shapes(chunk, nz, bl_mynn_version="gsd_41")
    assert shapes[SLOT_PLUME_WORK] == (80 * chunk * (nz + 1),)
    assert shapes[SLOT_PLUME_SCRATCH] == (13 * chunk * nz,)
    # The driver's thvl and the condensation's five work columns coexist,
    # so both are their own slots of the workspace, not reserved transients.
    assert shapes[SLOT_GSD41_THVL] == (chunk * nz,)
    assert shapes[SLOT_GSD41_CONDENSATION_WORK] == (5 * chunk * nz,)
    assert f"mynn_pbl:{SLOT_GSD41_THVL}" in storage.arrays
    assert f"mynn_pbl:{SLOT_GSD41_CONDENSATION_WORK}" in storage.arrays
    assert storage.payload_bytes == mynn_pbl_scratch_bytes(
        chunk, nz, bl_mynn_version="gsd_41") + 6 * members * nz * 6 * 8 * 4
    assert plan.reserved_bytes == 8192
    assert mynn_pbl_transient_bytes(nz=nz, column_chunk=chunk, bl_mynn_version="gsd_41") == 0
    assert mynn_pbl_transient_bytes(nz=nz, column_chunk=chunk) == 0
    with pytest.raises(MemoryError, match="exhaust device memory"):
        BatchStorage(plan, members, array_module=np, available_bytes=plan.required_bytes(members) - 1)


@pytest.mark.parametrize("bad", [
    {"bl_mynn_version": "unknown"}, {"bl_mynn_version": 1},
    {"bl_mynn_gsd41_unsquared_qtke": 0}, {"bl_mynn_gsd41_unsquared_qtke": np.bool_(True)},
    {"bl_mynn_cloud_tendency_form": "unknown"}, {"bl_mynn_cloud_tendency_form": True},
    {"bl_mynn_cloud_tendency_form": "gsd_41"},
])
def test_generation_selector_refuses_invalid_or_unbound_native_arithmetic(bad):
    with pytest.raises(ValueError, match="MYNN PBL"):
        _pbl_options(bad)


@pytest.mark.parametrize("unsquared", [False, True])
@pytest.mark.parametrize("cloud", ["wrf_461", "gsd_41"])
def test_gsd_options_preserve_each_explicit_native_identity(unsquared, cloud):
    result = _pbl_options({"bl_mynn_version": "gsd_41", "bl_mynn_gsd41_unsquared_qtke": unsquared,
                           "bl_mynn_cloud_tendency_form": cloud, "bl_mynn_mixlength": 2})
    assert result["bl_mynn_version"] == "gsd_41"
    assert result["bl_mynn_gsd41_unsquared_qtke"] is unsquared
    assert result["bl_mynn_cloud_tendency_form"] == cloud
    assert result["bl_mynn_mixlength"] == 2


def test_pbl_plan_preserves_invalid_reservation_refusal_with_gsd_transients():
    with pytest.raises(TypeError, match="reserved_bytes"):
        mynn_pbl_output_plan(nz=50, ny=6, nx=8, members=2, column_chunk=61,
                             bl_mynn_version="gsd_41", reserved_bytes=True)
