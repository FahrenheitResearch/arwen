"""Real all-member RTE calls versus the same native adapters run alone."""

from copy import copy
import dataclasses
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.ensemble.batch_rrtmgp import prepare_rrtmgp_column_batch

pytestmark = [pytest.mark.gpu, requires_gpu]


def _fixture(members, light, *, longwave=True, shortwave=True, native_workspace=True):
    import cupy as cp
    from test_rrtmgp import _chunk_invariance_case
    from gpuwm.core.rrtmgp import RRTMGPRadiation
    from gpuwm.core.model import SharedRRTMGPChunkWorkspace
    first = _chunk_invariance_case()
    nz, ny, nx = 49, 2, 3
    adapters, atmospheres, fields, states, configs = [], [], [], [], []
    for member in range(members):
        _, latitude, longitude, source, surface, state, _ = first
        atmosphere = {name: cp.ascontiguousarray(array[:, :ny, :nx]) for name, array in source.items()}
        surface = {name: cp.ascontiguousarray(array[:ny, :nx]) for name, array in surface.items()}
        surface["tsk"] += cp.float32(member * 0.125)
        surface["albedo"] += cp.float32(member * 0.005)
        surface.update(xland=cp.ones((ny, nx), cp.float32), glw=cp.full((ny, nx), 321, cp.float32),
                       swddir=cp.zeros((ny, nx), cp.float32), swddif=cp.zeros((ny, nx), cp.float32))
        surface["xland"][:, -1] = 2
        surface["qc_bl"] = cp.full((nz, ny, nx), 2e-5, cp.float32)
        surface["qi_bl"] = cp.full((nz, ny, nx), 3e-7, cp.float32)
        surface["cldfra_bl"] = cp.full((nz, ny, nx), 0.65, cp.float32)
        transported = SimpleNamespace(elapsed_seconds=0.0, p_top=10000.0,
            qc=atmosphere["qc"], qi=atmosphere["qi"], qr=cp.zeros((nz, ny, nx), cp.float32),
            qs=cp.zeros((nz, ny, nx), cp.float32), effc=cp.full((nz, ny, nx), 8, cp.float32),
            effi=cp.full((nz, ny, nx), 30, cp.float32), effs=cp.full((nz, ny, nx), 40, cp.float32),
            physics=SimpleNamespace(microphysics_updates=0))
        cfg = SimpleNamespace(nz=nz, ny=ny, nx=nx, mp_physics=8, dt=6.0, time_step_sound=4, radt=12.0,
            radt_minutes=12.0, ra_physics=4, ra_lw_physics=4, ra_sw_physics=4,
            ra_rrtmg_variant="rte-rrtmgp", bl_pbl_physics=5, icloud_bl=1,
            wrf_rrtmg_compatibility="none", use_adaptive_time_step=True)
        # Same absolute clock, different native geography. Mixed light crosses
        # both member and chunk boundaries, with ascending daylight gathers.
        longitude = cp.full((ny, nx), -90 if light == "day" else 90, cp.float32)
        if light == "mixed":
            longitude[:, :1] = -90
            if member % 2:
                longitude[:, 1:] = -90
        adapter = RRTMGPRadiation(datetime(1974, 4, 3, 18),
            cp.ascontiguousarray(latitude[:ny, :nx] + cp.float32(member)), longitude,
            column_chunk=5, longwave=longwave, shortwave=shortwave,
            trace_gas_overrides={"co2": 330e-6})
        adapter.update_count = member * 3
        if native_workspace:
            adapter.chunk_workspace = SharedRRTMGPChunkWorkspace(nz, adapter.column_chunk, transported.p_top)
        adapters.append(adapter)
        atmospheres.append(atmosphere)
        fields.append(surface)
        states.append(transported)
        configs.append(cfg)
    packed_atmosphere = {name: cp.concatenate([atmosphere[name] for atmosphere in atmospheres], axis=1)
                         for name in atmospheres[0]}
    packed_fields = {name: cp.concatenate([field[name] for field in fields], axis=1 if fields[0][name].ndim == 3 else 0)
                     for name in fields[0]}
    packed_state = SimpleNamespace(elapsed_seconds=0.0, p_top=10000.0,
        physics=SimpleNamespace(microphysics_updates=0))
    for name in ("qc", "qi", "qr", "qs", "effc", "effi", "effs"):
        setattr(packed_state, name, cp.concatenate([getattr(state, name) for state in states], axis=1))
    return adapters, atmospheres, fields, states, configs, packed_atmosphere, packed_fields, packed_state


@pytest.mark.parametrize("members", [2, 4, 8])
@pytest.mark.parametrize("light", ["day", "night", "mixed"])
def test_real_packed_rte_all_heating_flux_words_match_cold_and_carried_members(members, light, monkeypatch):
    _prove(members, light, monkeypatch)


@pytest.mark.parametrize("longwave,shortwave", [(True, False), (False, True)])
def test_independent_spectrum_preserves_declared_glw_and_optional_fluxes(longwave, shortwave, monkeypatch):
    _prove(2, "mixed", monkeypatch, longwave=longwave, shortwave=shortwave)


@pytest.mark.parametrize("light", ["day", "night", "mixed"])
def test_workspace_less_native_policy_matches_its_own_ordinary_branch(light, monkeypatch):
    _prove(2, light, monkeypatch, native_workspace=False)


def _prove(members, light, monkeypatch, *, longwave=True, shortwave=True, native_workspace=True):
    import cupy as cp
    from gpuwm.core.rrtmgp import RRTMGPRadiation
    adapters, atmospheres, fields, states, cfgs, packed_atmosphere, packed_fields, packed_state = _fixture(
        members, light, longwave=longwave, shortwave=shortwave, native_workspace=native_workspace)
    bound = prepare_rrtmgp_column_batch(adapters, member_states=states, member_configs=cfgs,
        atmosphere=packed_atmosphere, fields=packed_fields, packed_state=packed_state,
        members=members, ny=2, nx=3, available_bytes=1 << 30)
    # Separate counter owners are the independent oracle; the bound member
    # metadata is advanced only by the packed call, exactly once per member.
    ordinary_adapters = [copy(adapter) for adapter in adapters]
    native = RRTMGPRadiation.__call__
    observed = []
    def observe(self, **arguments):
        observed.append(arguments["atmosphere"]["pressure"].shape)
        return native(self, **arguments)
    monkeypatch.setattr(RRTMGPRadiation, "__call__", observe)
    before_atmosphere = {name: array.get().tobytes() for name, array in packed_atmosphere.items()}
    before_state = {name: getattr(packed_state, name).get().tobytes() for name in ("qc", "qi", "qr", "qs", "effc", "effi", "effs")}
    workspace_pointer = None if bound.workspace is None else int(bound.workspace.storage.data.ptr)
    try:
        for step, elapsed in enumerate((0.0, 720.0), 1):
            packed_state.elapsed_seconds = elapsed
            for state in states:
                state.elapsed_seconds = elapsed
            # These are the current per-member adaptive clock outputs, not
            # a fixed replacement timestep. The packed and ordinary calls
            # receive the same changed complete configurations.
            for cfg in cfgs:
                cfg.dt = 6.0 if step == 1 else 7.5
                cfg.time_step_sound = 4 if step == 1 else 8
            before_calls = len(observed)
            result = bound()
            assert observed[before_calls:] == [(49, members * 2, 3)]
            for member in range(members):
                ordinary = ordinary_adapters[member](atmosphere=atmospheres[member], fields=fields[member],
                                                     state=states[member], cfg=cfgs[member])
                stripe = slice(member * 2, (member + 1) * 2)
                for field in dataclasses.fields(result):
                    got, expected = getattr(result, field.name), getattr(ordinary, field.name)
                    if expected is None:
                        assert got is None
                    else:
                        selected = got[:, stripe] if got.ndim == 3 else got[stripe]
                        assert selected.get().tobytes() == expected.get().tobytes(), (member, step, light, field.name)
                        assert bool(cp.all(cp.isfinite(selected)))
                assert adapters[member].update_count == ordinary_adapters[member].update_count == member * 3 + step
            assert (None if bound.workspace is None else int(bound.workspace.storage.data.ptr)) == workspace_pointer
            assert {name: array.get().tobytes() for name, array in packed_atmosphere.items()} == before_atmosphere
            assert {name: getattr(packed_state, name).get().tobytes() for name in before_state} == before_state
        assert bound.receipt["native_calls"] == 2
        assert bound.receipt["last_config"]["dt"] != bound.receipt["metadata"]["config"]["dt"]
        assert bound.receipt["workspace_payload_bytes"] == bound.receipt["memory"]["solver_workspace_bytes"]
        assert bound.receipt["memory"]["column_transient_bytes"] > 0
        if shortwave:
            assert result.swddir is not None and result.swddif is not None
        if light == "night":
            assert not bool(cp.any(result.swdown))
    finally:
        bound.close()
    assert bound.storage is None and bound.workspace is None
    with pytest.raises(RuntimeError, match="closed"):
        bound()


def test_clock_drift_refuses_before_native_call_or_counter_mutation(monkeypatch):
    from gpuwm.core.rrtmgp import RRTMGPRadiation
    args = _fixture(2, "mixed")
    adapters, atmospheres, fields, states, cfgs, packed_atmosphere, packed_fields, packed_state = args
    bound = prepare_rrtmgp_column_batch(adapters, member_states=states, member_configs=cfgs,
        atmosphere=packed_atmosphere, fields=packed_fields, packed_state=packed_state,
        members=2, ny=2, nx=3, available_bytes=1 << 30)
    monkeypatch.setattr(RRTMGPRadiation, "__call__", lambda *unused, **kwargs: pytest.fail("native call reached after clock drift"))
    before = [adapter.update_count for adapter in adapters]
    states[1].elapsed_seconds = 1
    try:
        with pytest.raises(ValueError, match="wrong time"):
            bound()
        assert [adapter.update_count for adapter in adapters] == before
    finally:
        bound.close()
