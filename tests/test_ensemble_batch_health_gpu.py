"""Every member's compact health words equal the stock domain reduction."""

import math

import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


@pytest.mark.parametrize("members", [1, 4, 10])
@pytest.mark.parametrize("terrain", [False, True])
@pytest.mark.parametrize("width", [None, 1])
@pytest.mark.parametrize("failure", [None, "u_nan", "w_nan", "theta_nan", "folded_layer", "tie"])
def test_batched_health_records_are_word_identical_and_never_mix_members(members, terrain, width, failure):
    import cupy as cp
    from test_ensemble_batch_bigstep_gpu import _inputs
    from gpuwm.core.dycore import stability_report
    from gpuwm.ensemble.batch_dycore import member_domain_view
    from gpuwm.ensemble.batch_health import PreparedBatchStability
    from gpuwm.ensemble.batch_state import BatchedDomainState, SHARED_STATE_CANDIDATES
    inputs = _inputs(members, terrain=terrain, mapped=True)
    cfg = inputs[0].cfg
    batch = BatchedDomainState.from_prepared(inputs, array_module=cp, available_bytes=1 << 28,
             shared_fields=tuple(SHARED_STATE_CANDIDATES & inputs[0].arrays.keys()))
    target = members - 1
    if failure in ("u_nan", "w_nan", "theta_nan"):
        name = {"u_nan": "u", "w_nan": "w", "theta_nan": "thp"}[failure]
        batch.member_view(name, target).reshape(-1)[17] = np.float32("nan")
    elif failure == "folded_layer":
        view = member_domain_view(batch, target)
        base = view.phb
        base = base[:, None, None] if base.ndim == 1 else base
        view.php[1] = view.php[0] + base[0] - base[1] - np.float32(100)
    elif failure == "tie":
        view = batch.member_view("w", target)
        view.fill(np.float32(0))
        view.reshape(-1)[1] = np.float32(-19)
        view.reshape(-1)[5] = np.float32(19)
    health = PreparedBatchStability(batch, cfg, boundary_width=width, available_bytes=1 << 24)
    before = {name: array.get().tobytes() for name, array in batch.storage.arrays.items()}
    got = health()
    records = health.storage.arrays["health:result"].get().view(np.uint32)
    for member in range(members):
        scalar = member_domain_view(batch, member)
        expected = stability_report(scalar, cfg, boundary_width=width)
        expected_record = scalar.existing_scratch("integration_health_result").get().view(np.uint32)
        assert records[member].tobytes() == expected_record.tobytes(), (member, failure, terrain, width)
        assert got[member].keys() == expected.keys()
        for name, reference in expected.items():
            actual = got[member][name]
            assert math.isnan(actual) if isinstance(reference, float) and math.isnan(reference) else actual == reference
        if failure == "tie" and member == target:
            assert got[member]["w_argmax"] == 1
    assert {name: array.get().tobytes() for name, array in batch.storage.arrays.items()} == before
    cp.cuda.get_current_stream().synchronize()
    pool_before = cp.get_default_memory_pool().used_bytes()
    health()
    cp.cuda.get_current_stream().synchronize()
    assert cp.get_default_memory_pool().used_bytes() == pool_before


@pytest.mark.parametrize("members", [1, 4, 10])
def test_health_without_cfl_configuration_keeps_the_native_maximum_record(members):
    import cupy as cp
    from test_ensemble_batch_bigstep_gpu import _inputs
    from gpuwm.core.dycore import stability_report
    from gpuwm.ensemble.batch_dycore import member_domain_view
    from gpuwm.ensemble.batch_health import PreparedBatchStability
    from gpuwm.ensemble.batch_state import BatchedDomainState
    batch = BatchedDomainState.from_prepared(_inputs(members), array_module=cp, available_bytes=1 << 28)
    health = PreparedBatchStability(batch, available_bytes=1 << 24)
    reports = health()
    for member in range(members):
        scalar = member_domain_view(batch, member)
        assert reports[member] == stability_report(scalar)
        assert health.storage.arrays["health:result"][member].get().tobytes() == scalar.existing_scratch("integration_health_result").get().tobytes()


@pytest.mark.parametrize("members", [1, 4, 10])
@pytest.mark.parametrize("failure", [None, "theta", "mass", "packed_soil", "held_nan", "integer"])
def test_full_state_member_records_match_native_policy_with_strided_physics(members, failure):
    import cupy as cp
    from dataclasses import replace
    from test_ensemble_batch_bigstep_gpu import _inputs
    from gpuwm.core.health import StateHealthValidator, field_from_array, collect_state_fields
    from gpuwm.ensemble.batch_dycore import member_domain_view
    from gpuwm.ensemble.batch_health import PreparedBatchStateHealth
    from gpuwm.ensemble.batch_state import BatchedDomainState, SHARED_STATE_CANDIDATES
    inputs = _inputs(members, terrain=True, mapped=True)
    batch = BatchedDomainState.from_prepared(inputs, array_module=cp, available_bytes=1 << 28,
             shared_fields=tuple(SHARED_STATE_CANDIDATES & inputs[0].arrays.keys()))
    nz, ny, nx = batch.cfg.nz, batch.cfg.ny, batch.cfg.nx
    soil = cp.full((4, members * ny, nx), np.float32(280))
    held = cp.full((nz, members * ny, nx), np.float32(1e-4))
    integer = cp.zeros((members * ny, nx), np.int32)
    target = members - 1
    if failure == "theta":
        batch.member_view("thp", target)[0, 2, 3] = np.float32(-500)
    elif failure == "mass":
        batch.member_view("mup", target)[2, 3] = -batch.mub2d[2, 3] - np.float32(1)
    elif failure == "packed_soil":
        soil[2, target * ny + 2, 3] = np.float32(450)
    elif failure == "held_nan":
        held[1, target * ny + 2, 3] = np.float32("nan")
    elif failure == "integer":
        integer[target * ny + 2, 3] = np.int32(4)
    validator = PreparedBatchStateHealth(batch, available_bytes=1 << 26)
    validator.field_provider = lambda member: collect_state_fields(member_domain_view(batch, member), backend="gpu") + provider_tail(member)
    def provider_tail(member):
        return (field_from_array("surface.tslb", soil[:, member * ny:(member + 1) * ny]),
                field_from_array("held.pbl.rtheta", held[:, member * ny:(member + 1) * ny]),
                field_from_array("surface.ebal", integer[member * ny:(member + 1) * ny]))
    before = {name: array.get().tobytes() for name, array in batch.storage.arrays.items()}
    reports = validator.validate(phase="identity")
    actual_records = validator.storage.arrays["health:validate:result"].get()
    for member in range(members):
        fields = validator.field_provider(member)
        copies = tuple(replace(field, values=cp.ascontiguousarray(field.values),
                        auxiliary=None if field.auxiliary is None else cp.ascontiguousarray(field.auxiliary))
                       for field in fields)
        scalar = member_domain_view(batch, member)
        native = StateHealthValidator(scalar, field_provider=lambda: copies)
        expected = native.validate(phase="identity")
        assert actual_records[member].tobytes() == native._result.get().view(np.uint64).tobytes(), (member, failure)
        for name, reference in vars(expected).items():
            actual = getattr(reports[member], name)
            assert math.isnan(actual) if isinstance(reference, float) and math.isnan(reference) else actual == reference
    assert {name: array.get().tobytes() for name, array in batch.storage.arrays.items()} == before
    assert validator.receipt["field_input_copy_bytes"] == 0
