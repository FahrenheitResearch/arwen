"""CPU metadata and ownership contracts for joined RRTMGP columns."""

from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.ensemble.batch_rrtmgp import (
    _scalar, _workspace_less_chunks, PackedRRTMGPColumns, rrtmgp_call_memory_inventory, rrtmgp_geometry_plan,
    validate_rrtmgp_binding_metadata, validate_rrtmgp_member_metadata)
from gpuwm.ensemble.batch_storage import BatchStorage


def _members(count=4):
    adapter = SimpleNamespace(start_time=datetime(2026, 9, 1, 12), longwave=True,
        shortwave=True, column_chunk=13, validation_mode="fused",
        trace_gas_overrides=None, trace_vmr={"co2": 420e-6}, update_count=7)
    state = SimpleNamespace(elapsed_seconds=720.0, p_top=10000.0,
                            physics=SimpleNamespace(microphysics_updates=3))
    cfg = SimpleNamespace(nz=49, nx=3, ny=2, dt=6.0, time_step_sound=4, mp_physics=8,
                          radt=12.0, radt_minutes=12.0,
                          use_adaptive_time_step=True, bl_pbl_physics=5, icloud_bl=1)
    return ([deepcopy(adapter) for _ in range(count)], [deepcopy(state) for _ in range(count)],
            [deepcopy(cfg) for _ in range(count)], deepcopy(state))


@pytest.mark.parametrize("count", [2, 4, 8])
def test_member_geometry_owned_privately_and_metadata_clock_exact(count):
    adapters, states, cfgs, packed = _members(count)
    got = validate_rrtmgp_member_metadata(adapters, states, cfgs, packed_state=packed)
    assert got["elapsed_seconds"] == 720 and got["p_top"] == 10000
    assert got["microphysics_updates"] == 3
    plan = rrtmgp_geometry_plan(ny=2, nx=3, reserved_bytes=1024)
    storage = BatchStorage(plan, count, array_module=np, available_bytes=1 << 20)
    assert storage.payload_bytes == count * 2 * 2 * 3 * 4
    storage.member_view("rrtmgp:latitude", count - 1).fill(5)
    assert not np.any(storage.member_view("rrtmgp:latitude", 0))
    assert plan.required_bytes(count) >= storage.payload_bytes + 1024


@pytest.mark.parametrize("target,key,value,reason", [
    ("adapter", "longwave", False, "spectra"),
    ("adapter", "start_time", datetime(2026, 9, 1, 13), "solar-date"),
    ("adapter", "trace_vmr", {"co2": 400e-6}, "trace"),
    ("adapter", "validation_mode", "full", "validation"),
    ("adapter", "column_chunk", 11, "policy"),
    ("cfg", "dt", 5.0, "configuration differs"),
    ("cfg", "bl_pbl_physics", 1, "cloud coupling"),
    ("cfg", "use_adaptive_time_step", False, "cadence"),
    ("state", "elapsed_seconds", np.nextafter(720.0, 721.0), "wrong time"),
    ("state", "p_top", np.nextafter(10000.0, 10001.0), "above-model cap"),
])
def test_different_member_authority_refuses_before_a_native_call(target, key, value, reason):
    adapters, states, cfgs, packed = _members()
    bank = {"adapter": adapters, "state": states, "cfg": cfgs}[target]
    setattr(bank[3], key, value)
    with pytest.raises(ValueError, match=reason):
        validate_rrtmgp_member_metadata(adapters, states, cfgs, packed_state=packed)
    assert all(adapter.update_count == 7 for adapter in adapters)


def test_microphysics_and_packed_clock_authorities_refuse():
    adapters, states, cfgs, packed = _members()
    states[1].physics.microphysics_updates = 0
    with pytest.raises(ValueError, match="effective-radius selection"):
        validate_rrtmgp_member_metadata(adapters, states, cfgs, packed_state=packed)
    states[1].physics.microphysics_updates = 3
    packed.elapsed_seconds = 721
    with pytest.raises(ValueError, match="packed clock/top"):
        validate_rrtmgp_member_metadata(adapters, states, cfgs, packed_state=packed)


def test_each_member_retains_its_original_counter_without_affecting_column_arithmetic():
    adapters, states, cfgs, packed = _members()
    adapters[3].update_count = 100
    validate_rrtmgp_member_metadata(adapters, states, cfgs, packed_state=packed)
    assert adapters[0].update_count == 7 and adapters[3].update_count == 100


def test_common_live_adaptive_step_is_validated_without_freezing_binding():
    adapters, states, cfgs, packed = _members()
    bound = validate_rrtmgp_member_metadata(adapters, states, cfgs, packed_state=packed)
    for cfg in cfgs:
        cfg.dt = 7.5
        cfg.time_step_sound = 8
    for state in [*states, packed]:
        state.elapsed_seconds = 727.5
    current = validate_rrtmgp_member_metadata(adapters, states, cfgs, packed_state=packed)
    assert current["config"] != bound["config"]
    validate_rrtmgp_binding_metadata(current, bound)
    cfgs[-1].time_step_sound = 9
    with pytest.raises(ValueError, match="configuration differs"):
        validate_rrtmgp_member_metadata(adapters, states, cfgs, packed_state=packed)


@pytest.mark.parametrize("field,value", [("nz", 50), ("radt_minutes", 15.0),
    ("mp_physics", 6), ("use_adaptive_time_step", False)])
def test_common_immutable_policy_change_still_refuses_bound_workspace(field, value):
    adapters, states, cfgs, packed = _members()
    bound = validate_rrtmgp_member_metadata(adapters, states, cfgs, packed_state=packed)
    for cfg in cfgs:
        setattr(cfg, field, value)
    current = validate_rrtmgp_member_metadata(adapters, states, cfgs, packed_state=packed)
    with pytest.raises(ValueError, match="immutable adapter/config/top authority"):
        validate_rrtmgp_binding_metadata(current, bound)


def test_common_top_change_refuses_stale_above_model_workspace():
    adapters, states, cfgs, packed = _members()
    bound = validate_rrtmgp_member_metadata(adapters, states, cfgs, packed_state=packed)
    for state in [*states, packed]:
        state.p_top = 9000.0
    current = validate_rrtmgp_member_metadata(adapters, states, cfgs, packed_state=packed)
    with pytest.raises(ValueError, match="top authority"):
        validate_rrtmgp_binding_metadata(current, bound)


@pytest.mark.parametrize("field,value", [("dt", 7.5), ("time_step_sound", 8)])
def test_fixed_clock_common_step_change_retains_exact_binding_identity(field, value):
    adapters, states, cfgs, packed = _members()
    for cfg in cfgs:
        cfg.use_adaptive_time_step = False
    bound = validate_rrtmgp_member_metadata(adapters, states, cfgs, packed_state=packed)
    for cfg in cfgs:
        setattr(cfg, field, value)
    current = validate_rrtmgp_member_metadata(adapters, states, cfgs, packed_state=packed)
    with pytest.raises(ValueError, match="immutable adapter/config/top authority"):
        validate_rrtmgp_binding_metadata(current, bound)


def test_metadata_preserves_float_words_and_inactive_control_types():
    assert _scalar(-0.0) != _scalar(0.0)
    assert _scalar(True) != _scalar(1.0)
    assert _scalar(np.float32(6.0)) == _scalar(6.0)


def test_native_workspace_policy_cannot_change_or_differ_between_members():
    adapters, states, cfgs, packed = _members()
    bound = validate_rrtmgp_member_metadata(adapters, states, cfgs, packed_state=packed)
    workspace = SimpleNamespace(nz=49, column_chunk=13, p_top=10000.0)
    adapters[0].chunk_workspace = workspace
    with pytest.raises(ValueError, match="policy"):
        validate_rrtmgp_member_metadata(adapters, states, cfgs, packed_state=packed)
    for adapter in adapters:
        adapter.chunk_workspace = deepcopy(workspace)
    current = validate_rrtmgp_member_metadata(adapters, states, cfgs, packed_state=packed)
    with pytest.raises(ValueError, match="immutable adapter/config/top authority"):
        validate_rrtmgp_binding_metadata(current, bound)


def test_unset_and_explicit_none_workspace_follow_the_same_native_policy():
    adapters, states, cfgs, packed = _members()
    assert all(not hasattr(adapter, "chunk_workspace") for adapter in adapters)
    unset = validate_rrtmgp_member_metadata(adapters, states, cfgs, packed_state=packed)
    bound = object.__new__(PackedRRTMGPColumns)
    bound.member_adapters = tuple(adapters)
    bound._table_owners = {}
    bound._setup_owners = {"chunk_workspace": (None,) * len(adapters)}
    bound._validate_table_owners()
    for adapter in adapters:
        adapter.chunk_workspace = None
    explicit = validate_rrtmgp_member_metadata(adapters, states, cfgs, packed_state=packed)
    assert unset == explicit
    bound._validate_table_owners()
    adapters[0].chunk_workspace = object()
    with pytest.raises(ValueError, match="setup owner changed"):
        bound._validate_table_owners()


def test_workspace_less_envelope_prices_unfused_cloud_and_planck_products():
    plan = _workspace_less_chunks(nz=49, column_chunk=5, p_top=10000.0,
        metadata={"ngpt_lw": 256, "nband_lw": 16, "ngas_lw": 7,
                  "ngpt_sw": 224, "nband_sw": 14, "ngas_sw": 5})
    assert plan["lw/planck_lay"] == ((5, 74, 256), 4)
    assert plan["lw/planck_lev"] == ((5, 75, 256), 4)
    assert plan["lw/mcica_mask"] == ((5, 74, 256), 1)
    assert plan["sw/finalized_asy"] == ((5, 50, 224), 4)
    assert plan["sw/gas_ssa"] == ((5, 50, 224), 4)


def test_memory_basis_uses_all_columns_and_declares_mynn_result_and_daylight_buffers(monkeypatch):
    from gpuwm.core import preflight
    observed = []
    def columns(cfg, p_top, *, column_chunk):
        observed.append((cfg.ny, cfg.nx, p_top, column_chunk))
        return {"columns/play": ((cfg.ny * cfg.nx, cfg.nz), 4)}
    monkeypatch.setattr(preflight, "rrtmgp_column_shapes", columns)
    monkeypatch.setattr(preflight, "rrtmgp_workspace_shapes",
                        lambda nz, chunk, p_top: {"lw/work": ((chunk, nz, 256), 4)})
    _adapters, _states, cfgs, _packed = _members(8)
    cfg = cfgs[0]
    plan = rrtmgp_call_memory_inventory(cfg, members=8, ny=2, nx=3, p_top=10000, column_chunk=13)
    assert observed == [(16, 3, 10000, 13)] and cfg.ny == 2
    assert plan["column_transient_bytes"] == 8 * 2 * 3 * 49 * 4
    assert plan["solver_workspace_bytes"] == 13 * 49 * 256 * 4
    extra = plan["additional_call_buffers"]
    assert extra["mynn/qc_bl"] == ((48, 49), 4)
    assert extra["mynn/supplied_ice"] == ((48, 49), 1)
    assert extra["returned/rthratenlw"] == ((49, 48), 4)
    assert extra["shortwave/daylight_indices"] == ((48,), 8)
    assert extra["shortwave/scatter_up"] == ((13, 50), 4)
