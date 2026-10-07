"""The CAMS rows' lateral boundary (WRF-Chem ``have_bcs_chem``).

A row whose first enabled boundary source is a data-store source is filled
on every forcing frame (gpuwm.chem_source_init), so the root's sealed
forcing carries it beside water vapour, UNCOUPLED as WRF-Chem's
``chem_b``/``chem_bt`` are, and the chem flow boundary takes its inflow
from the table's outermost row:
``max(epsilc, chem_b + chem_bt*(dt_rk + dtbc))``
(chem/module_input_chem_data.F:1531-2031, :2331-2357; dyn_em/solve_em.F:2604).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from gpuwm.config import RunConfig

CAMS_ROWS = ("chem_o3", "chem_no2", "chem_co", "chem_so2")


def _cfg(**changes):
    values = dict(nx=13, ny=12, nz=4, dx=12000., dy=12000., ztop=10000.,
                  dt=1., run_seconds=120., moist=True, mp_physics=8,
                  bl_pbl_physics=1, sf_sfclay_physics=1, specified=True,
                  chem_sets="cams_aq", chem_sources="cams-global")
    return RunConfig(**(values | changes))


def _table(cfg):
    from gpuwm.chem_table import load
    return load(cfg)


def test_the_filled_rows_are_the_rows_a_data_store_source_fills():
    from gpuwm.chem_source_init import chem_boundary_fields
    cfg = _cfg()
    assert chem_boundary_fields(_table(cfg), cfg) == CAMS_ROWS
    # Without the source enabled nothing is filled and no row is carried.
    quiet = _cfg(chem_sources="")
    assert chem_boundary_fields(_table(quiet), quiet) == ()


def test_the_config_door_admits_a_data_store_source_and_refuses_any_other(
        monkeypatch):
    import gpuwm.config as config
    config.validate_chem_config(_cfg())
    real = config._catalog_sources()
    unfilled = dict(real)
    unfilled["cams-global"] = replace(real["cams-global"], acquisition=None)
    monkeypatch.setattr(config, "_catalog_sources", lambda: unfilled)
    with pytest.raises(ValueError, match=r"cams-global.*no forcing-frame fill"):
        config.validate_chem_config(_cfg())


def test_a_sealed_inventory_carries_exactly_the_filled_rows():
    from gpuwm.boundary_fields import admissible_boundary_inventory
    cfg = _cfg()
    assert admissible_boundary_inventory(cfg, ("qv", *CAMS_ROWS))
    # A table sealed before the source was enabled would run the rows on
    # their default inflow: refused, and so is a partial one.
    assert not admissible_boundary_inventory(cfg, ("qv",))
    assert not admissible_boundary_inventory(cfg, ("qv", *CAMS_ROWS[:3]))
    # A chem-off configuration admits no chem table at all.
    plain = _cfg(chem_sets="", chem_sources="")
    assert admissible_boundary_inventory(plain, ("qv",))
    assert not admissible_boundary_inventory(plain, ("qv", *CAMS_ROWS))


def test_the_pricing_counts_the_carried_rows():
    from gpuwm.core.device_inventory import lbc_interval_values
    from gpuwm.core import preflight as pf
    cfg = _cfg()
    plain = _cfg(chem_sets="", chem_sources="")
    width, nz, ny, nx = cfg.spec_bdy_width, cfg.nz, cfg.ny, cfg.nx
    per_row = 2 * (2 * nz * ny * width + 2 * nz * width * nx)
    assert (lbc_interval_values(cfg) - lbc_interval_values(plain)
            == len(CAMS_ROWS) * per_row)
    slots = pf.scratch_slot_registry(cfg, n_lbc_intervals=2)
    rows = len(_table(cfg).transported)
    assert slots["lbc_chem_bxs"] == (rows, nz, ny)
    assert slots["lbc_chem_btye"] == (rows, nz, nx)
    assert "lbc_chem_bxs" not in pf.scratch_slot_registry(
        _cfg(chem_sources=""), n_lbc_intervals=2)
    for slot in slots:
        if slot.startswith("lbc_chem_"):
            assert pf.scratch_slot_lifetime(slot).kind == "carrying"


def _frame(cfg, seed):
    from gpuwm.chem_source_init import chem_boundary_fields
    from gpuwm.core.state import DomainState
    state = DomainState(cfg, array_module=np)
    rng = np.random.default_rng(seed)
    state.mup[...] = rng.uniform(-500.0, 500.0, state.mup.shape)
    state.qv[...] = rng.uniform(1e-3, 1e-2, state.qv.shape)
    for name in CAMS_ROWS:
        getattr(state, name)[...] = rng.uniform(1e-3, 5e-2,
                                                getattr(state, name).shape)
    # What fill_boundary_sources records on the frame it filled.
    state._external_scalar_boundary_fields = (
        "qv", *chem_boundary_fields(state.chem.table, cfg))
    return state


def test_the_forcing_carries_the_rows_uncoupled_and_the_launch_reads_the_outer_row():
    from gpuwm.core.chem_bdy import EPSILC
    from gpuwm.verify.chem_bdy_ref import apply_chem_flow_boundaries_cpu
    from gpuwm.ingest.lateral_bc import (
        CHEM_BOUNDARY_SLOTS, attach_lateral_boundaries,
        build_state_lateral_boundaries, chem_boundary_rows,
        domain_boundary_snapshot)
    cfg = _cfg()
    first, second = _frame(cfg, 1), _frame(cfg, 2)
    snapshot = domain_boundary_snapshot(first)
    for name in CAMS_ROWS:
        # WRF-Chem's chem_b is the mixing ratio itself.
        np.testing.assert_array_equal(
            snapshot[name], getattr(first, name).astype(np.float64))
    mu = np.asarray(first.total_mu())
    chm = first.c1h[:, None, None] * mu[None] + first.c2h[:, None, None]
    np.testing.assert_allclose(snapshot["qv"], chm * first.qv, rtol=1e-6)

    duration = 3600.0
    boundaries = build_state_lateral_boundaries(
        [first, second], [0.0, duration], spec_bdy_width=cfg.spec_bdy_width,
        spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone)
    state = _frame(cfg, 3)
    attach_lateral_boundaries(state, boundaries)
    state.elapsed_seconds = 600.0
    names = [row.state_attr for row in state.chem.transported]
    has, tables, dtbc = chem_boundary_rows(state, cfg, names)
    assert has == [1 if name in CAMS_ROWS else 0 for name in names]
    assert len(tables) == len(CHEM_BOUNDARY_SLOTS)
    assert float(dtbc) == 600.0
    for index, name in enumerate(names):
        if not has[index]:
            continue
        a = getattr(first, name).astype(np.float64)
        b = getattr(second, name).astype(np.float64)
        rate = (b - a) / duration
        outer = (
            (a[:, :, 0], rate[:, :, 0]), (a[:, :, -1], rate[:, :, -1]),
            (a[:, 0, :], rate[:, 0, :]), (a[:, -1, :], rate[:, -1, :]))
        for side, (value, tendency) in enumerate(outer):
            np.testing.assert_array_equal(tables[2 * side][index],
                                          value.astype(np.float32))
            np.testing.assert_array_equal(tables[2 * side + 1][index],
                                          tendency.astype(np.float32))

    # The launch: inflow cells of the outer ring take b + bt*dt, WRF's
    # float32 expression; an all-inflow flux makes every ring cell one.
    nz, ny, nx = state.p.shape
    fields = [np.array(getattr(state, name)) for name in names]
    ru = np.zeros((nz, ny, nx + 1), np.float32)
    rv = np.zeros((nz, ny + 1, nx), np.float32)
    ru[:, :, 0], ru[:, :, -1] = 1.0, -1.0      # west in, east in
    rv[:, 0, :], rv[:, -1, :] = 1.0, -1.0      # south in, north in
    dt_rk = np.float32(1.0 / 3.0)
    dt = np.float32(dt_rk + np.float32(dtbc))
    apply_chem_flow_boundaries_cpu(
        fields, ru, rv, cfg.spec_zone, has,
        [row.default_inflow for row in state.chem.transported], *tables, dt)
    for index, name in enumerate(names):
        if not has[index]:
            continue
        west = np.maximum(EPSILC, tables[0][index]
                          + np.float32(tables[1][index] * dt))
        # Y owns the corners; the west column's interior rows are X's.
        np.testing.assert_array_equal(fields[index][:, 1:-1, 0],
                                      west[:, 1:-1])
        south = np.maximum(EPSILC, tables[4][index]
                           + np.float32(tables[5][index] * dt))
        np.testing.assert_array_equal(fields[index][:, 0, :], south)


def test_a_forcing_without_the_filled_rows_is_refused_at_launch():
    """A chem run whose sealed forcing lacks a filled row names it rather
    than running that row on its default inflow."""
    from types import SimpleNamespace

    from gpuwm.core import chem_transport
    from gpuwm.ingest.lateral_bc import (
        attach_lateral_boundaries, build_state_lateral_boundaries)
    cfg = _cfg()
    frames = []
    for seed in (1, 2):
        frame = _frame(cfg, seed)
        frame._external_scalar_boundary_fields = ("qv",)
        frames.append(frame)
    boundaries = build_state_lateral_boundaries(
        frames, [0.0, 3600.0], spec_bdy_width=cfg.spec_bdy_width,
        spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone)
    state = _frame(cfg, 3)
    attach_lateral_boundaries(state, boundaries)
    state.elapsed_seconds = 0.0
    nz, ny, nx = state.p.shape
    with pytest.raises(RuntimeError, match=r"chem_co.*re-prepare"):
        chem_transport.apply_chem_flow_boundaries(
            state, cfg, np.zeros((nz, ny, nx + 1), np.float32),
            np.zeros((nz, ny + 1, nx), np.float32), 1.0)


def test_a_wesely_run_keeps_the_radiation_gsw():
    """Wesely's stomatal resistance reads GSW, which the driver held only for
    RUC: a Noah run with the CAMS rows stopped at its first chem step
    ("chem process drydep.wesely reads ['gsw']").  The plane is kept and
    priced from the same answer."""
    from gpuwm.core import preflight as pf
    from gpuwm.core.chem_context import chem_physics_reads
    cfg = _cfg(sf_surface_physics=2)
    assert "gsw" in chem_physics_reads(cfg)
    assert "gsw" in pf.physics_field_names_2d(cfg)
    plain = _cfg(sf_surface_physics=2, chem_sets="", chem_sources="")
    assert chem_physics_reads(plain) == frozenset()
    assert "gsw" not in pf.physics_field_names_2d(plain)
