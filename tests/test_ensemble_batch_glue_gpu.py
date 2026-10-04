"""Scalar operation oracles for inventoried member-wide dycore glue."""
from dataclasses import replace

import numpy as np
import pytest

from conftest import requires_gpu
from test_ensemble_batch_bigstep_gpu import _inputs, _scalar_state
from gpuwm.ensemble.batch_glue import (
    prepare_face_masses, prepare_periodic_alias, prepare_total_mass,
    prepare_total_theta, workspace_specs,
)
from gpuwm.ensemble.batch_state import BatchedDomainState, SHARED_STATE_CANDIDATES
from gpuwm.wrf_exact import ENABLED

pytestmark = [pytest.mark.gpu, requires_gpu]


def _prepared(count, *, terrain=False, specified=False):
    from gpuwm.ensemble.batch_diagnostics import diagnostics_specs
    inputs = _inputs(count, terrain=terrain, specified=specified)
    workspace = workspace_specs(inputs[0].cfg)
    specs = diagnostics_specs(inputs[0].cfg) + workspace
    return tuple(replace(member, arrays=dict(member.arrays,
                 **{spec.name: np.zeros(spec.shape, spec.dtype) for spec in workspace}))
                 for member in inputs), specs


def _reference(member, operation):
    from gpuwm.core import dycore, ieva
    # Existing DomainState construction knows scalar fields, not batch extras.
    arrays = {name: value for name, value in member.arrays.items()
              if not name.startswith("batch_")}
    state = _scalar_state(replace(member, arrays=arrays))
    if operation == "mass":
        return {"batch_mass": state.total_mu()}
    if operation == "theta":
        # Full scalar DomainState step chooses its existing fused helper.
        from gpuwm.core.bandwidth_glue import total_theta
        return {"batch_theta": total_theta(state)}
    if operation == "faces":
        mass = state.total_mu()
        mux, muy = ieva.stage_face_masses(state, member.cfg, mass)
        return {"batch_mass": mass, "batch_mux": mux, "batch_muy": muy}
    dycore.close_periodic_alias(state, member.cfg)
    return {"u": state.u, "v": state.v}


@pytest.mark.parametrize("count", [1, 4, 10, 20, 40])
@pytest.mark.parametrize("operation", ["mass", "theta", "faces", "alias"])
@pytest.mark.parametrize("terrain", [False, True])
@pytest.mark.parametrize("share", [False, True])
def test_glue_exact_against_scalar_operations(count, operation, terrain, share):
    import cupy as cp
    inputs, specs = _prepared(count, terrain=terrain)
    shared = tuple(sorted(SHARED_STATE_CANDIDATES & inputs[0].arrays.keys())) if share else ()
    batch = BatchedDomainState.from_prepared(
        inputs, array_module=cp, available_bytes=2**30, shared_fields=shared,
        extra_specs=specs)
    before = {name: cp.asnumpy(value).tobytes()
              for name, value in batch.storage.arrays.items()}
    if operation == "mass":
        launch = prepare_total_mass(batch)
    elif operation == "theta":
        launch = prepare_total_theta(batch)
    elif operation == "faces":
        mass, faces = prepare_total_mass(batch), prepare_face_masses(batch)
        def launch():
            mass()
            faces()
    else:
        launch = prepare_periodic_alias(batch)
    launch()
    cp.cuda.get_current_stream().synchronize()
    pool = cp.get_default_memory_pool()
    live_bytes = pool.used_bytes()
    launch()
    cp.cuda.get_current_stream().synchronize()
    assert pool.used_bytes() == live_bytes
    touched = set()
    for member_index, member in enumerate(inputs):
        reference = _reference(member, operation)
        for name, expected in reference.items():
            touched.add(name)
            assert cp.asnumpy(batch.member_view(name, member_index)).tobytes() == cp.asnumpy(expected).tobytes(), (member_index, name)
    for name, previous in before.items():
        if name not in touched:
            assert cp.asnumpy(batch.storage.arrays[name]).tobytes() == previous, name


@pytest.mark.parametrize("count", [1, 4, 10, 20, 40])
def test_face_boundary_copies_use_each_members_own_edge_cells(count):
    import cupy as cp
    inputs, specs = _prepared(count, specified=True)
    batch = BatchedDomainState.from_prepared(inputs, array_module=cp,
                                             available_bytes=2**30, extra_specs=specs)
    prepare_total_mass(batch)()
    prepare_face_masses(batch)()
    for index, member in enumerate(inputs):
        for name, expected in _reference(member, "faces").items():
            assert cp.asnumpy(batch.member_view(name, index)).tobytes() == cp.asnumpy(expected).tobytes()


@pytest.mark.parametrize("count", [1, 4, 10, 20, 40])
def test_alias_word_copy_retains_nan_payloads_signed_zero_and_member_isolation(count):
    import cupy as cp
    inputs, specs = _prepared(count)
    words = np.array([0x80000000, 0x7FC00123, 0x00000001, 0x7F800000], np.uint32)
    for index, member in enumerate(inputs):
        for name in ("u", "v"):
            value = member.arrays[name].view(np.uint32)
            value[...] = np.resize(np.roll(words, index), value.size).reshape(value.shape)
    batch = BatchedDomainState.from_prepared(inputs, array_module=cp,
                                             available_bytes=2**30, extra_specs=specs)
    prepare_periodic_alias(batch)()
    for index, member in enumerate(inputs):
        for name, expected in _reference(member, "alias").items():
            assert cp.asnumpy(batch.member_view(name, index)).tobytes() == cp.asnumpy(expected).tobytes()


@pytest.mark.parametrize("count", [1, 4, 10, 20, 40])
@pytest.mark.parametrize("boundary", ["periodic", "open_x", "open_y", "specified", "nested"])
@pytest.mark.parametrize("share_base", [False, True])
def test_strict_face_round_points_boundaries_and_no_temporary_allocations(count, boundary, share_base, monkeypatch):
    import cupy as cp
    if not ENABLED:
        pytest.skip("strict calc_mu_uv round points require a separate strict arithmetic process")
    inputs, specs = _prepared(count)
    prepared = []
    for index, member in enumerate(inputs):
        cfg = replace(member.cfg, open_x=boundary == "open_x", open_y=boundary == "open_y",
                      specified=boundary == "specified", nested=boundary == "nested")
        arrays = dict(member.arrays)
        ny, nx = cfg.ny, cfg.nx
        base = np.resize(np.array([95344.6796875, 93474.4375, 93001.25], np.float32), ny * nx).reshape(ny, nx)
        perturbation = np.resize(np.array([15.01163101196289, 26.48706817626953, -17.25], np.float32), ny * nx).reshape(ny, nx)
        arrays["mub2d"] = base.copy()
        arrays["mup"] = perturbation + np.float32(0.03125 * index)
        prepared.append(replace(member, cfg=cfg, arrays=arrays))
    batch = BatchedDomainState.from_prepared(prepared, array_module=cp,
                                             available_bytes=2**30, extra_specs=specs,
                                             shared_fields=("mub2d",) if share_base else ())
    launch = prepare_face_masses(batch)
    class FieldAllocationHook(cp.cuda.MemoryHook):
        name = "StrictFaceFieldAllocationHook"
        def __init__(self):
            self.requests = []
        def malloc_preprocess(self, device_id, size, mem_size):
            self.requests.append((int(size), int(mem_size)))
    hook = FieldAllocationHook()
    def copied(*args, **kwargs):
        pytest.fail("strict face closure must not request an overlapping-view CuPy copy")
    with monkeypatch.context() as closing_check:
        closing_check.setattr(cp, "copyto", copied)
        with hook:
            launch()
            launch()
    cp.cuda.get_current_stream().synchronize()
    assert not hook.requests, "strict face submission allocated unpriced temporary GPU storage"
    for index, member in enumerate(prepared):
        reference = _reference(member, "faces")
        for name in ("batch_mux", "batch_muy"):
            assert cp.asnumpy(batch.member_view(name, index)).tobytes() == cp.asnumpy(reference[name]).tobytes(), (index, name, boundary)
    # An ordinary total-mass average has the wrong FP32 result at this face.
    # This fixed negative control makes the strict operand order observable.
    p, b = prepared[0].arrays["mup"], prepared[0].arrays["mub2d"]
    ordinary = np.float32(0.5) * ((b[0, 1] + p[0, 1]) + (b[0, 0] + p[0, 0]))
    strict = np.float32(0.5) * (((p[0, 1] + p[0, 0]) + b[0, 1]) + b[0, 0])
    assert ordinary.view(np.uint32) != strict.view(np.uint32)
    assert cp.asnumpy(batch.member_view("batch_mux", 0))[0, 1].view(np.uint32) == strict.view(np.uint32)


@pytest.mark.parametrize("count", [1, 4, 10, 20, 40])
@pytest.mark.parametrize("specified", [False, True])
@pytest.mark.parametrize("share_base", [False, True])
def test_strict_face_round_points_preserve_special_operand_words(count, specified, share_base):
    import cupy as cp
    if not ENABLED:
        pytest.skip("strict special operand words require a separate strict arithmetic process")
    inputs, specs = _prepared(count, specified=specified)
    words = np.array([0x80000000, 0x00000000, 0x00000001, 0x80000001,
                      0x7FC01122, 0xFFC03344, 0x7FA05577, 0xFFA06688,
                      0x7F800000, 0xFF800000,
                      0x3F800000, 0xBF800000], np.uint32)
    for index, member in enumerate(inputs):
        base = member.arrays["mub2d"]
        perturbation = member.arrays["mup"]
        base.view(np.uint32)[...] = np.resize(words, base.size).reshape(base.shape)
        perturbation.view(np.uint32)[...] = np.resize(
            np.roll(words, index + 2), perturbation.size).reshape(perturbation.shape)
    batch = BatchedDomainState.from_prepared(inputs, array_module=cp,
                                             available_bytes=2**30, extra_specs=specs,
                                             shared_fields=("mub2d",) if share_base else ())
    prepare_face_masses(batch)()
    for index, member in enumerate(inputs):
        reference = _reference(member, "faces")
        for name in ("batch_mux", "batch_muy"):
            assert cp.asnumpy(batch.member_view(name, index)).tobytes() == cp.asnumpy(reference[name]).tobytes(), (index, name)
