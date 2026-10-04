"""Original member physics with real device arrays and persistent drivers."""
from dataclasses import fields, is_dataclass

import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


def _snapshot(state, cfg, driver):
    import cupy as cp
    from gpuwm.core.device_inventory import state_array_shapes
    result = {}
    def collect(value, name):
        if isinstance(value, cp.ndarray):
            result[name] = value.get().tobytes()
        elif isinstance(value, dict):
            for key, item in value.items():
                collect(item, f"{name}/{key}")
        elif is_dataclass(value):
            for field in fields(value):
                collect(getattr(value, field.name), f"{name}/{field.name}")
            collect(getattr(value, "extra_scalars", None), f"{name}/extra_scalars")
    for name in state_array_shapes(cfg):
        collect(getattr(state, name), "state/" + name)
    collect(driver.fields, "fields")
    for name in ("tendencies", "pbl_tendencies", "radiation_tendencies", "cumulus_tendencies",
                 "microphysics", "rainc", "cu_nca", "cu_pratec", "cu_expiring", "cu_rates",
                 "_pending_rainbl"):
        collect(getattr(driver, name, None), "driver/" + name)
    result["call_counts"] = dict(driver.call_counts)
    result["microphysics_updates"] = driver.microphysics_updates
    return result


@pytest.mark.parametrize("members", [1, 4])
@pytest.mark.parametrize("mp,pbl,sfclay,cumulus", [
    (8, 1, 1, 0), (6, 5, 5, 0), (8, 9, 1, 0), (10, 1, 91, 1),
])
def test_member_local_real_driver_and_microphysics_match_each_independent_call(members, mp, pbl, sfclay, cumulus):
    import cupy as cp
    from test_physics_driver import _full_state
    from gpuwm.core.dycore import update_diagnostics
    from gpuwm.core.microphysics import apply
    from gpuwm.ensemble.physics_execution import MemberPhysicsBinding, MemberPhysicsExecutor

    originals, bindings = [], []
    for member in range(members):
        pair = []
        for _ in range(2):
            state, cfg, driver = _full_state(nx=6, ny=5, nz=24, dt=6.0,
                mp_physics=mp, bl_pbl_physics=pbl, sf_sfclay_physics=sfclay,
                cu_physics=cumulus, bldt=0.0, radt=0.0)
            state.u[...] = cp.float32(6.0 + member * 0.25)
            state.v[...] = cp.float32(0.5 - member * 0.125)
            # Separate domains may be at different cadence positions.
            state.elapsed_seconds = float(member * 6)
            pair.append((state, cfg, driver))
        originals.append(pair[0])
        state, cfg, driver = pair[1]
        bindings.append(MemberPhysicsBinding(state, cfg, driver, member))
    executor = MemberPhysicsExecutor(bindings)
    for _ in range(2):
        held = executor.compute()
        for member, (state, cfg, driver) in enumerate(originals):
            driver.compute(state, cfg)
            assert held[member] is bindings[member].driver.tendencies
        executor.apply_microphysics()
        for member, (state, cfg, driver) in enumerate(originals):
            result = apply(state, cfg, cfg.dt, refl_10cm_due=False)
            driver.accept_microphysics(result, dt=cfg.dt)
            update_diagnostics(state, cfg.hypsometric_opt)
            update_diagnostics(bindings[member].state, cfg.hypsometric_opt)
            actual = _snapshot(bindings[member].state, bindings[member].cfg, bindings[member].driver)
            expected = _snapshot(state, cfg, driver)
            assert actual.keys() == expected.keys()
            for field in expected:
                assert actual[field] == expected[field], (member, field, mp, pbl, sfclay, cumulus)
            state.elapsed_seconds += cfg.dt
            bindings[member].state.elapsed_seconds += bindings[member].cfg.dt


@pytest.mark.parametrize("layout", ["outermost", "column"])
def test_device_tendency_word_handoff_has_no_allocations_and_no_member_reduction(layout):
    import cupy as cp
    from types import SimpleNamespace
    from gpuwm.ensemble.physics_execution import AdmittedTendencyTarget, TENDENCY_FIELDS
    members, nz, ny, nx = 4, 3, 5, 6
    rng = np.random.default_rng(903)
    sources = []
    for member in range(members):
        words = rng.integers(0, 2**32, (nz, ny, nx), dtype=np.uint32)
        sources.append(SimpleNamespace(**{name: cp.asarray(words.view(np.float32)) for name in TENDENCY_FIELDS},
                                       extra_scalars={"ni": cp.asarray(words.view(np.float32))}))
    shape = (members, nz, ny, nx) if layout == "outermost" else (nz, members * ny, nx)
    arrays = {name: cp.empty(shape, cp.float32) for name in TENDENCY_FIELDS}
    target = AdmittedTendencyTarget(arrays, members=members, layout=layout,
        extra_scalars={"ni": cp.empty(shape, cp.float32)}, array_module=cp)
    target(sources)
    cp.cuda.get_current_stream().synchronize()
    before = cp.get_default_memory_pool().used_bytes()
    result = target(sources)
    cp.cuda.get_current_stream().synchronize()
    assert cp.get_default_memory_pool().used_bytes() == before
    for member, source in enumerate(sources):
        for name in TENDENCY_FIELDS:
            assert target._view(getattr(result, name), member).get().tobytes() == getattr(source, name).get().tobytes()
        assert target._view(result.extra_scalars["ni"], member).get().tobytes() == source.extra_scalars["ni"].get().tobytes()
