"""Prepared-host allocation coverage, byte ownership and clock compatibility."""
from dataclasses import replace
import gc
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from gpuwm.config import RunConfig
from gpuwm.core.device_inventory import state_array_shapes
from gpuwm.core.state import DomainState
from gpuwm.ensemble.batch_state import (
    BatchStateUnsupported, BatchedDomainState, PreparedHostMember,
    SHARED_STATE_CANDIDATES, batch_from_members, state_array_specs,
)
from gpuwm.ensemble.batch_storage import BatchArraySpec


def config(**kwargs):
    return RunConfig(nx=4, ny=3, nz=2, dx=1000.0, dy=1000.0,
                     ztop=10000.0, dt=3.0, run_seconds=30.0, **kwargs)


def clock(**kwargs):
    result = dict(ticks=0, step_ticks=3, tick_den=1, run_ticks=30,
                  step_count=0, dt_fp32=np.float32(3), dtbc_fp32=np.float32(0))
    result.update(kwargs)
    return result


def prepared(cfg=None, *, offset=0, metadata=None, timing=None, scratch=None):
    cfg = cfg or config()
    state = DomainState(cfg, array_module=np)
    arrays = {name: getattr(state, name).copy() for name in state_array_shapes(cfg)}
    arrays["u"].fill(offset)
    scalars = {name: value for name, value in vars(state).items()
               if name not in arrays and name not in {
                   "physics", "lateral_boundaries", "_scratch", "_scratch_arena",
                   "_host_setup_state", "_phb_host"}}
    scalars.update(metadata or {})
    return PreparedHostMember(cfg, arrays, scalars, timing or clock(), scratch=scratch)


def pack(members, **kwargs):
    return BatchedDomainState.from_prepared(members, array_module=np,
                                            available_bytes=10**7, **kwargs)


def test_shapes_and_default_ownership_follow_the_existing_inventory():
    for cfg in (config(), config(terrain_opt=1), config(moist=True, mp_physics=10)):
        specs = state_array_specs(cfg)
        assert {spec.name: spec.shape for spec in specs} == state_array_shapes(cfg)
        assert all(spec.ownership == "member" for spec in specs)
        assert all(np.dtype(spec.dtype) == np.dtype(np.float32) for spec in specs)
    assert "p" not in SHARED_STATE_CANDIDATES
    assert {"ht", "msft", "thb", "phb", "dnw"} <= SHARED_STATE_CANDIDATES


def test_additional_member_carriers_are_copied_and_priced_before_allocation():
    members = (prepared(), prepared())
    shape = members[0].arrays["p"].shape
    specs = (BatchArraySpec("p_perturbation", shape, "member"),
             BatchArraySpec("status_words", (3,), "member", dtype=np.uint32))
    enriched = []
    for index, member in enumerate(members):
        arrays = dict(member.arrays)
        arrays["p_perturbation"] = np.full(shape, index + 1, np.float32)
        arrays["status_words"] = np.array([0x80000000, 0x7FC00123, index], np.uint32)
        enriched.append(replace(member, arrays=arrays))
    baseline = pack(members)
    result = pack(enriched, extra_specs=specs)
    for index, member in enumerate(enriched):
        for spec in specs:
            assert result.member_view(spec.name, index).tobytes() == member.arrays[spec.name].tobytes()
            assert not np.shares_memory(result.member_view(spec.name, index), member.arrays[spec.name])
    added = sum(((2 * np.prod(spec.shape) * np.dtype(spec.dtype).itemsize + 511) // 512) * 512
                for spec in specs)
    assert result.plan.required_bytes(2) - baseline.plan.required_bytes(2) == added
    with pytest.raises(MemoryError):
        BatchedDomainState.from_prepared(enriched, array_module=np,
                                        available_bytes=result.plan.required_bytes(2) - 1,
                                        extra_specs=specs)
    with pytest.raises(BatchStateUnsupported, match="inventory differs"):
        pack(members, extra_specs=specs)
    with pytest.raises(BatchStateUnsupported, match="inventory differs"):
        pack(enriched)


@pytest.mark.parametrize("name", ["u", "storage", "scratch", "physics", "_scratch", "scratch:new"])
def test_additional_carriers_cannot_replace_controls_or_existing_fields(name):
    with pytest.raises(ValueError, match="collides"):
        pack((prepared(),), extra_specs=(BatchArraySpec(name, (1,), "member"),))


def test_additional_carriers_reject_duplicates_and_unqualified_sharing():
    spec = BatchArraySpec("p_perturbation", (2, 3, 4), "member")
    with pytest.raises(ValueError, match="collides"):
        pack((prepared(),), extra_specs=(spec, spec))
    with pytest.raises(BatchStateUnsupported, match="independent member"):
        pack((prepared(),), extra_specs=(BatchArraySpec("new_shared", (1,), "shared"),))
    with pytest.raises(TypeError, match="BatchArraySpec"):
        pack((prepared(),), extra_specs=("new",))


def test_same_invalid_integer_clock_cannot_change_the_numerical_step():
    invalid = clock(step_ticks=1)
    with pytest.raises(BatchStateUnsupported, match="integer clock step differs"):
        pack((prepared(timing=invalid), prepared(timing=invalid)))


def test_every_state_field_gets_a_member_backing_without_input_aliases():
    inputs = (prepared(offset=1), prepared(offset=7))
    result = pack(inputs)
    for name, shape in state_array_shapes(inputs[0].cfg).items():
        assert getattr(result, name).shape == (2,) + shape
        for member in range(2):
            got = result.member_view(name, member)
            assert got.tobytes() == inputs[member].arrays[name].tobytes()
            assert not np.shares_memory(got, inputs[member].arrays[name])
        assert not np.shares_memory(result.member_view(name, 0), result.member_view(name, 1))
    result.member_view("u", 0).fill(99)
    assert np.all(result.member_view("u", 1) == 7)
    assert np.all(inputs[0].arrays["u"] == 1)
    assert result.storage.payload_bytes == sum(row["payload_bytes"] for row in result.plan.inventory(2))


def test_shared_candidates_require_bytes_before_allocation_and_remain_read_only():
    inputs = (prepared(), prepared())
    inputs[0].arrays["phb"].view(np.uint32)[0] = np.uint32(0x7FC00123)
    inputs[1].arrays["phb"].view(np.uint32)[0] = np.uint32(0x7FC00123)
    result = pack(inputs, shared_fields=("phb", "msft"))
    assert result.phb.shape == inputs[0].arrays["phb"].shape
    assert result.phb.tobytes() == inputs[0].arrays["phb"].tobytes()
    assert result.storage.pointer_stride_bytes("phb") == 0
    with pytest.raises(ValueError, match="read-only"):
        result.member_view("phb", 1)[0] = 2
    inputs[1].arrays["phb"].view(np.uint32)[0] = np.uint32(0x7FC00456)
    with pytest.raises(BatchStateUnsupported, match="shared phb differs"):
        pack(inputs, shared_fields=("phb",))
    inputs[0].arrays["msft"].fill(0)
    inputs[1].arrays["msft"].fill(-0.0)
    with pytest.raises(BatchStateUnsupported, match="shared msft differs"):
        pack(inputs, shared_fields=("msft",))
    with pytest.raises(BatchStateUnsupported, match="not audited"):
        pack(inputs, shared_fields=("u",))


def test_distinct_reference_bases_preserve_arbitrary_words_as_member_fields():
    inputs = (prepared(), prepared())
    words = (np.uint32(0x80000000), np.uint32(0x7FC00001))
    for member, word in zip(inputs, words):
        member.arrays["pb"].view(np.uint32)[0] = word
    result = pack(inputs)
    for member in range(2):
        assert result.member_view("pb", member).tobytes() == inputs[member].arrays["pb"].tobytes()


def test_prepared_host_base_cache_is_detached_without_reconstruction():
    members = (prepared(), prepared())
    host = np.array([0x8000000000000000, 0x7FF8000000000123, 0x3FF0000000000000], np.uint64).view(np.float64)
    members = tuple(replace(member, phb_host=host.copy()) for member in members)
    result = pack(members)
    for cache in result.phb_host_members:
        assert cache.tobytes() == host.tobytes()
        assert not np.shares_memory(cache, host)
        assert not cache.flags.writeable


def test_scratch_has_independent_member_and_slot_backings_with_registry_shapes():
    cfg = config()
    values = [np.full((cfg.nz + 1, cfg.ny, cfg.nx), m + 1, np.float32) for m in range(2)]
    members = tuple(prepared(cfg, scratch={"rk_ww": value}) for value in values)
    result = pack(members, scratch_slots={"adv_ru": np.float32})
    assert result.scratch(values[0].shape, "rk_ww").shape == (2,) + values[0].shape
    assert not np.shares_memory(result.scratch_member_view("rk_ww", 0), result.scratch_member_view("rk_ww", 1))
    assert not np.shares_memory(result.existing_scratch("rk_ww"), result.existing_scratch("adv_ru"))
    result.scratch_member_view("rk_ww", 0).fill(9)
    assert np.all(result.scratch_member_view("rk_ww", 1) == 2)
    assert result.existing_scratch("missing") is None
    with pytest.raises(BatchStateUnsupported, match="not planned"):
        result.scratch((2,), "missing")
    with pytest.raises(ValueError, match="shape/dtype"):
        result.scratch((2,), "rk_ww")
    with pytest.raises(BatchStateUnsupported, match="unclassified scratch"):
        pack(members, scratch_slots={"unknown": np.float32})


@pytest.mark.parametrize("kind", ["extra", "missing", "wrong_shape", "wrong_dtype"])
def test_unknown_or_incomplete_array_inventory_fails_before_allocation(kind):
    first, second = prepared(), prepared()
    arrays = dict(second.arrays)
    if kind == "extra":
        arrays["unregistered"] = np.zeros((1,), np.float32)
    elif kind == "missing":
        arrays.pop("u")
    elif kind == "wrong_shape":
        arrays["u"] = np.zeros((1,), np.float32)
    else:
        arrays["u"] = arrays["u"].astype(np.float64)
    second = replace(second, arrays=arrays)
    class NoAllocator:
        def zeros(self, *args, **kwargs):
            raise AssertionError("validation allocated before it refused")
    with pytest.raises((BatchStateUnsupported, ValueError)):
        BatchedDomainState.from_prepared((first, second), array_module=NoAllocator(), available_bytes=10**7)


def test_config_grid_clock_and_scalar_bytes_cannot_be_silently_unified():
    first = prepared()
    with pytest.raises(BatchStateUnsupported, match="configuration/grid differs"):
        pack((first, replace(prepared(), cfg=replace(first.cfg, dx=999.0))))
    with pytest.raises(BatchStateUnsupported, match="clock differs"):
        pack((first, prepared(timing=clock(ticks=3, step_count=1))))
    with pytest.raises(BatchStateUnsupported, match="scalar metadata differs"):
        pack((first, prepared(metadata={"cf1": np.float32(-0.0)})))
    with pytest.raises(BatchStateUnsupported, match="fixed common clock"):
        pack((prepared(config(use_adaptive_time_step=True)), prepared(config(use_adaptive_time_step=True))))
    with pytest.raises(BatchStateUnsupported, match="clock snapshot omits"):
        pack((replace(first, clock={"ticks": 0}), prepared()))


@pytest.mark.parametrize("bad", [np.float32(0), np.float32(-1), np.float32(np.nan), np.float32(np.inf)])
def test_equal_invalid_clocks_cannot_pass_pack_admission(bad):
    with pytest.raises(ValueError, match="clock"):
        pack((prepared(timing=clock(dt_fp32=bad)), prepared(timing=clock(dt_fp32=bad))))


def test_equal_clocks_still_bind_the_configuration_and_initial_boundary_time():
    with pytest.raises(BatchStateUnsupported, match="clock step differs from configuration"):
        pack((prepared(timing=clock(dt_fp32=np.float32(2))),
              prepared(timing=clock(dt_fp32=np.float32(2)))))
    with pytest.raises(BatchStateUnsupported, match="elapsed seconds differ from clock ticks"):
        pack((prepared(timing=clock(ticks=3, step_count=1)),
              prepared(timing=clock(ticks=3, step_count=1))))


def test_scalar_nan_payload_and_dtype_are_preserved_and_compared_exactly():
    nan = np.array([0x7FC00123], np.uint32).view(np.float32)[0]
    first, second = prepared(metadata={"cf1": nan}), prepared(metadata={"cf1": nan})
    result = pack((first, second))
    assert result.cf1.tobytes() == nan.tobytes()
    with pytest.raises(BatchStateUnsupported, match="scalar metadata differs"):
        pack((first, prepared(metadata={"cf1": np.float64(nan)})))


def test_from_members_preserves_n1_identity_and_rejects_unbatched_physics():
    cfg = config()
    state = DomainState(cfg, array_module=np)
    before = dict(vars(state))
    result = batch_from_members((state,), (cfg,), array_module=np, available_bytes=0)
    assert result is state
    assert all(vars(state)[name] is value for name, value in before.items())
    state.physics = object()
    with pytest.raises(BatchStateUnsupported, match="mutable caches"):
        batch_from_members((state,), (cfg,), array_module=np, available_bytes=0)


def test_from_members_retains_no_original_states_and_rejects_device_or_forcing():
    cfg = config()
    states = (DomainState(cfg, array_module=np), DomainState(cfg, array_module=np))
    result = batch_from_members(states, (cfg, cfg), clocks=(clock(), clock()),
                                array_module=np, available_bytes=10**7)
    states[0].u.fill(14)
    assert np.all(result.u == 0)
    assert all(value is not states[0] and value is not states[1] for value in vars(result).values())
    states[0].unregistered = np.zeros((1,), np.float32)
    with pytest.raises(BatchStateUnsupported, match="unclassified member array"):
        batch_from_members(states, (cfg, cfg), clocks=(clock(), clock()), array_module=np, available_bytes=10**7)
    proxy = SimpleNamespace(physics=None, _host_setup_state=False)
    with pytest.raises(BatchStateUnsupported, match="double the unpriced GPU"):
        batch_from_members((proxy, proxy), (cfg, cfg), clocks=(clock(), clock()), array_module=np, available_bytes=10**7)
    states[0].lateral_boundaries = object()
    with pytest.raises(BatchStateUnsupported, match="member tables/caches"):
        batch_from_members(states, (cfg, cfg), clocks=(clock(), clock()), array_module=np, available_bytes=10**7)


def test_named_state_admission_refuses_before_any_allocation():
    members = (prepared(), prepared())
    class NoAllocator:
        def zeros(self, *args, **kwargs):
            raise AssertionError("underbudget state reached allocation")
    with pytest.raises(MemoryError, match="resident ensemble members need"):
        BatchedDomainState.from_prepared(members, array_module=NoAllocator(), available_bytes=1)


def test_original_host_states_can_be_released_after_packing():
    cfg = config()
    states = (DomainState(cfg, array_module=np), DomainState(cfg, array_module=np))
    references = tuple(weakref.ref(state) for state in states)
    result = batch_from_members(states, (cfg, cfg), clocks=(clock(), clock()),
                                array_module=np, available_bytes=10**7)
    del states
    gc.collect()
    assert all(reference() is None for reference in references)
    assert result.u.shape[0] == 2
