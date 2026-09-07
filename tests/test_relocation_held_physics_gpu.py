"""Relocated terrain/base-state physics, through the Roch transplant seams."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import requires_gpu

cp = pytest.importorskip("cupy")
pytestmark = [pytest.mark.gpu, requires_gpu]


class _DeclaredHeating:
    # A declared input isolates transport of radiation heating from radiation
    # algorithm accuracy; the real radiation-driver coupling seam consumes it.
    restart_identity = {"algorithm": "relocation-declared-heating-v1"}

    def __call__(self, *, atmosphere, fields, state, cfg):
        from gpuwm.core.physics import RadiationResult
        shape = state.p.shape
        k, j, i = cp.indices(shape, dtype=cp.float32)
        return RadiationResult(
            cp.ascontiguousarray(-1e-5 * (1 + k / cfg.nz + i / cfg.nx)),
            cp.ascontiguousarray(2e-5 * (1 + j / cfg.ny)),
            cp.full(shape[1:], 250., cp.float32),
            cp.full(shape[1:], 330., cp.float32))


def _build(cfg, geo):
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import init_at_rest
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.physics import initialize_physics
    from tilestream.harness import install_geography
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: 296. + .004 * np.asarray(z),
                           p_surf=cfg.p_surf, ztop=cfg.ztop, terrain_z=geo.terrain)
    state = init_at_rest(cfg, coord, base)
    state.qv[...] = cp.float32(.020) * cp.exp(-cp.asarray(state.height_half()) / cp.float32(3500.))
    update_diagnostics(state, cfg.hypsometric_opt)
    install_geography(state, geo)
    state.u[...] = 5. + .05 * cp.arange(cfg.nx + 1, dtype=cp.float32)[None, None, :]
    state.v[...] = 2. + .03 * cp.arange(cfg.ny + 1, dtype=cp.float32)[None, :, None]
    state.u[..., -1] = state.u[..., 0]
    state.v[:, -1] = state.v[:, 0]
    driver = initialize_physics(state, cfg, landmask=1., tsk=305.,
                               radiation=_DeclaredHeating())
    return state, driver


def _snapshot(state):
    from tilestream.physics_inventory import carrier_manifest
    return {key: cp.asnumpy(value) for key, value in carrier_manifest(state).items()}


def _same(a, b):
    assert a.keys() == b.keys()
    bad = [key for key in a if not np.array_equal(a[key], b[key])]
    assert not bad, bad


@pytest.mark.parametrize("cadence", [0., 2.])
def test_relocated_base_recouples_last_rates_without_advancing_cadence(cadence):
    from gpuwm.config import RunConfig, validate_run_config
    from gpuwm.core.nest_relocation import Placement, plan_relocation, transplant_overlap
    from gpuwm.core.physics import couple_column_tendencies, couple_ysu_tendencies
    from gpuwm.core.physics_continuation import (
        capture_continuation, restore_continuation, shift_continuation)
    from gpuwm.ingest.relocation_init import rederive_after_transplant
    from gpuwm.runtime import RealRelocationChildPreparer
    from tilestream import harness
    from tilestream.physics_inventory import carrier_scalars, set_carrier_scalars

    cfg = validate_run_config(RunConfig(
        nx=20, ny=16, nz=40, dx=12000., dy=12000., ztop=18000.,
        dt=20., time_step_sound=4, terrain_opt=1, map_proj=1,
        run_seconds=0., moist=True, mp_physics=10,
        bl_pbl_physics=1, sf_sfclay_physics=91, bldt=cadence,
        ra_physics=90, radt_minutes=cadence))
    geo = harness.make_geography(cfg)
    source, original_driver = _build(cfg, geo)
    harness.run_steps(source, cfg, 2)
    if cadence:
        # The diagnostic dict retains views of the canonical rates, so the
        # persistent memory census and transport own each physical array once.
        assert all(original_driver.last_ysu[name] is value
                   for name, value in original_driver.pbl_raw_rates.items())
        assert sum(value.nbytes for value in original_driver.pbl_raw_rates.values()) == (
            6 * cfg.nz * cfg.ny * cfg.nx * 4)
    captured = capture_continuation(source, original_driver)
    clock = carrier_scalars(source)
    plan = plan_relocation(
        placement_from=Placement(grid_id=2, i_parent_start=10, j_parent_start=10),
        placement_to=Placement(grid_id=2, i_parent_start=11, j_parent_start=10, generation=1),
        parent_grid_ratio=3, child_nx=cfg.nx, child_ny=cfg.ny)
    # Like the existing Roch test_relocation_real_init fixture: a cell
    # entering a different blend-frame row changes its effective base.
    terrain = np.roll(geo.terrain, -plan.shift_i, axis=1).copy()
    terrain[:, :5] += 150.
    moved_geo = replace(geo, terrain=terrain)
    shifted = shift_continuation(captured, plan)
    states = []
    for mode in ("production", "explicit-reference", "lost-tendencies"):
        state, driver = _build(cfg, moved_geo)
        transplant_overlap(source_state=source, target_state=state, plan=plan)
        receipt = rederive_after_transplant(
            source_state=source, target_state=state, plan=plan, cfg=cfg)
        assert any(receipt["base_changed_cells"].values())
        assert not bool(cp.array_equal(state.total_mu(), source.total_mu()))
        # Keep every other surface value and clock identical between arms;
        # only the fate of held PBL/radiation tendencies is under test.
        for name, old in original_driver.fields.items():
            target = driver.fields[name]
            window = plan.window(old.shape)
            if window is not None:
                (dj, sj), (di, si) = window
                target[..., dj, di] = old[..., sj, si]
        set_carrier_scalars(state, clock)
        restore_continuation(state, driver, shifted)
        before = dict(driver.call_counts)
        if mode == "production":
            RealRelocationChildPreparer._recouple_moved_cumulus(
                SimpleNamespace(state=state, cfg=SimpleNamespace(run=cfg)))
        elif mode == "explicit-reference":
            if driver.pbl_raw_rates:
                driver.pbl_tendencies = couple_ysu_tendencies(
                    state, cfg, driver.pbl_raw_rates)
            driver.radiation_tendencies = couple_column_tendencies(
                state, cfg, rtheta=driver.rthratenlw + driver.rthratensw)
        assert driver.call_counts == before
        states.append((state, driver))
    moved, moved_driver = states[0]
    if cadence:
        # Independent scalar arithmetic verifies the *new* mass and map
        # factor, not just agreement of two calls to the same coupler.
        mass = moved.c1h[:, None, None] * moved.total_mu()[None] + moved.c2h[:, None, None]
        expected = mass * moved_driver.pbl_raw_rates["dtheta"] / moved.msft[None]
        assert bool(cp.array_equal(expected, moved_driver.pbl_tendencies.rtheta))
        assert float(cp.max(cp.abs(expected))) > 0
        for name in ("du", "dv"):
            assert float(cp.max(cp.abs(moved_driver.pbl_raw_rates[name]))) > 0
        old_coupled = cp.asnumpy(original_driver.pbl_tendencies.rtheta)
        (dj, sj), (di, si) = plan.window(old_coupled.shape)
        assert not np.array_equal(cp.asnumpy(expected)[..., dj, di], old_coupled[..., sj, si])
    for state, driver in states:
        harness.run_steps(state, cfg, 1)
        delta = 1 if cadence == 0 else 0
        assert driver.call_counts["ysu"] == original_driver.call_counts["ysu"] + delta
        assert driver.call_counts["radiation"] == original_driver.call_counts["radiation"] + delta
    actual, reference, lost = (_snapshot(state) for state, _ in states)
    _same(actual, reference)
    changed = [key for key in actual if key.startswith("state/")
               and not np.array_equal(actual[key], lost[key])]
    if cadence:
        assert changed
    else:
        _same(actual, lost)
    print(f"cadence={cadence}: changed-base transplant matches raw-rate reference; "
          f"omission control changes {len(changed)} state fields")


@pytest.mark.parametrize("cumulus", [3, 16])
def test_streamed_relocation_keeps_real_held_pbl_carriers(cumulus):
    from gpuwm.config import RunConfig, validate_run_config
    from gpuwm.core.nest_relocation import Placement, plan_relocation
    from gpuwm.core.physics_continuation import (
        capture_continuation, restore_continuation, shift_continuation)
    from gpuwm.io.restart import DRIVER_HELD_FORCING_ATTRS
    from tilestream import harness
    from tilestream.physics_inventory import streaming_inventory

    cfg = validate_run_config(RunConfig(
        nx=20, ny=16, nz=40, dx=12000., dy=12000., ztop=18000.,
        dt=20., time_step_sound=4, terrain_opt=1, map_proj=1,
        run_seconds=0., moist=True, mp_physics=10, cu_physics=cumulus,
        cudt_minutes=0., bl_pbl_physics=1, sf_sfclay_physics=91, bldt=2.,
        ra_physics=90, radt_minutes=2.))
    geo = harness.make_geography(cfg)
    source, driver = _build(cfg, geo)
    harness.run_steps(source, cfg, 2)
    # The real GPU producer owns both the names and the values in this store.
    store = {key: cp.asnumpy(value) for key, value in streaming_inventory(source).items()}
    expected = capture_continuation(source, driver)
    held = [name for name in DRIVER_HELD_FORCING_ATTRS
            if getattr(driver, name, None) is not None]
    assert held and any(np.any(store[f"held/{name}"]) for name in held)
    for name in held:
        assert f"driver/{name}" not in store
        getattr(driver, name).fill(-99.)  # A row template is no longer authoritative.
    captured = capture_continuation(source, driver, store=store)
    _same(captured, expected)
    missing = dict(store)
    del missing[f"held/{held[0]}"]
    with pytest.raises(ValueError, match="canonical carrier held/"):
        capture_continuation(source, driver, store=missing)

    plan = plan_relocation(
        placement_from=Placement(grid_id=2, i_parent_start=10, j_parent_start=10),
        placement_to=Placement(grid_id=2, i_parent_start=11, j_parent_start=10, generation=1),
        parent_grid_ratio=3, child_nx=cfg.nx, child_ny=cfg.ny)
    shifted = shift_continuation(captured, plan)
    restored, restored_driver = _build(cfg, geo)
    restore_continuation(restored, restored_driver, shifted)
    for name in held:
        key = f"held/{name}"
        # Independently copy the physical overlap, leaving the entering strip cold.
        reference = np.zeros_like(expected[key])
        (dj, sj), (di, si) = plan.window(reference.shape)
        reference[..., dj, di] = expected[key][..., sj, si]
        np.testing.assert_array_equal(cp.asnumpy(getattr(restored_driver, name)), reference)
