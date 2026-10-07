"""Chem species transport through the engine's own scalar path (DESIGN 2.2).

GPU.  What is proven here, on the moist bubble case the moisture transport
tests use:

* PASSIVE: a run carrying the passive test tracer leaves every dynamics and
  moisture array byte-identical to the same run without chem, with the
  6th-order filter and the 2-D Smagorinsky mixing on and off.
* SAME OPERATOR: on a periodic domain a tracer started equal to qv stays
  equal to qv, bit for bit, through every RK stage (no microphysics or PBL
  touches qv here), which is the claim that chem transport IS the moist
  transport, including the positive-definite final stage and the held
  forward tendency.
* D11: an initially uniform tracer stays uniform (SK2008 design constraint).
* MONOTONIC (chem_adv_opt = 2): the final stage runs WRF's
  advect_scalar_mono, moves no other byte, and keeps a blob inside its
  initial bounds where the positive-definite stage overshoots them, with
  the ledger still closing.
* SPECIFIED DOMAIN: the WRF-Chem flow-dependent boundary matches an
  independent transcription of WRF's four sequential loops word for word,
  and a tracer blob that never reaches the boundary keeps its mass, which
  the ledger's transport bucket then reads as zero.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


def _bubble_case(**overrides):
    from gpuwm.config import RunConfig
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import init_moist_balanced

    nx, ny, nz = 16, 12, 16
    values = dict(nx=nx, ny=ny, nz=nz, dx=500.0, dy=500.0, ztop=8000.0,
                  dt=2.0, run_seconds=0.0, moist=True)
    values.update(overrides)
    cfg = RunConfig(**values)
    vc = make_vertical_coord(nz)

    def th(z):
        return 300.0 * np.exp(1e-4 * np.asarray(z, float) / 9.81)

    b = make_base_state(vc, th, p_surf=cfg.p_surf, ztop=cfg.ztop)

    def qv_prof(z):
        return 0.012 * np.exp(-np.asarray(z, float) / 2500.0)

    def bubble(x, z):
        zz = z[:, None, None] if np.ndim(z) == 1 else z
        L = np.sqrt((x[None, None, :] / 2000.0) ** 2
                    + ((zz - 2000.0) / 1500.0) ** 2)
        return (np.where(L < 1.0, 5.0 * np.cos(np.pi * L / 2) ** 2, 0.0)
                * np.ones((nz, ny, nx)))

    state = init_moist_balanced(cfg, vc, b, qv_prof, thp_func=bubble)
    ic = (np.arange(nx) + 0.5) / nx
    jc = (np.arange(ny) + 0.5) / ny
    iu = np.arange(nx + 1) / nx
    jv = np.arange(ny + 1) / ny

    def fxy(xi, yj):
        return (1.0 + 0.04 * np.sin(2 * np.pi * xi)[None, :]
                * np.cos(2 * np.pi * yj)[:, None])

    state.set_map_coriolis(msft=fxy(ic, jc), msfu=fxy(iu, jc),
                           msfv=fxy(ic, jv))
    return cfg, state


_MET = ("u", "v", "w", "thp", "php", "mup", "p", "al", "alt",
        "qv", "qc", "qr")


@pytest.mark.parametrize("mixing", [
    dict(),
    dict(diff_6th_opt=2, diff_6th_factor=0.12),
    dict(km_opt=4, diff_6th_opt=2, diff_6th_factor=0.12),
], ids=["no-mixing", "diff6", "smag-diff6"])
@pytest.mark.parametrize("vorder", [3, 5])
def test_a_passive_tracer_moves_no_other_byte(mixing, vorder):
    import cupy as cp
    from gpuwm.core.dycore import run_steps

    mixing = dict(mixing, v_sca_adv_order=vorder)
    cfg0, plain = _bubble_case(**mixing)
    cfg1 = dataclasses.replace(cfg0, chem_sets="tracer_test")
    _, chem = _bubble_case(**mixing, chem_sets="tracer_test")
    rng = np.random.default_rng(3)
    chem.chem_passive_1[...] = cp.asarray(
        rng.random(chem.chem_passive_1.shape).astype(np.float32))
    run_steps(plain, cfg0, 6)
    run_steps(chem, cfg1, 6)
    for name in _MET:
        a = cp.asnumpy(getattr(plain, name))
        b = cp.asnumpy(getattr(chem, name))
        assert a.tobytes() == b.tobytes(), name
    assert float(cp.abs(chem.chem_passive_1).sum()) > 0


@pytest.mark.parametrize("mixing", [
    dict(),
    dict(diff_6th_opt=2, diff_6th_factor=0.12),
    dict(km_opt=4, diff_6th_opt=2, diff_6th_factor=0.12),
], ids=["no-mixing", "diff6", "smag-diff6"])
@pytest.mark.parametrize("vorder", [3, 5])
def test_a_tracer_started_as_qv_stays_qv_bit_for_bit(mixing, vorder):
    import cupy as cp
    from gpuwm.core.dycore import run_steps

    cfg, state = _bubble_case(**mixing, chem_sets="tracer_test",
                              v_sca_adv_order=vorder)
    state.chem_passive_1[...] = state.qv
    run_steps(state, cfg, 8)
    assert float(cp.abs(state.w).max()) > 0.5
    a = cp.asnumpy(state.qv)
    b = cp.asnumpy(state.chem_passive_1)
    assert a.tobytes() == b.tobytes()


def test_chem_mix6_off_removes_the_filter_from_chem_only():
    import cupy as cp
    from gpuwm.core.dycore import run_steps

    mixing = dict(diff_6th_opt=2, diff_6th_factor=0.12)
    cfg, on = _bubble_case(**mixing, chem_sets="tracer_test")
    cfg_off, off = _bubble_case(**mixing, chem_sets="tracer_test",
                                chem_mix6_off=True)
    for s in (on, off):
        s.chem_passive_1[...] = s.qv
    run_steps(on, cfg, 6)
    run_steps(off, cfg_off, 6)
    assert cp.asnumpy(on.qv).tobytes() == cp.asnumpy(off.qv).tobytes()
    assert (cp.asnumpy(on.chem_passive_1).tobytes()
            != cp.asnumpy(off.chem_passive_1).tobytes())


def test_a_uniform_tracer_stays_uniform_d11():
    import cupy as cp
    from gpuwm.core.dycore import run_steps

    cfg, state = _bubble_case(chem_sets="tracer_test")
    state.chem_passive_1[...] = np.float32(7.5)
    run_steps(state, cfg, 10)
    dev = float(cp.abs(state.chem_passive_1 - np.float32(7.5)).max()) / 7.5
    assert dev < 2e-5, dev


@pytest.mark.parametrize("vorder", [3, 5])
def test_a_tracer_layer_keeps_its_mass_on_a_periodic_domain(vorder):
    """No boundary, no source, no sink: every kilogram the ledger's
    transport bucket books is created or destroyed by the transport itself.
    A sharp tracer layer straddling the rising bubble has empty air above
    and below it, so the positive-definite final stage meets empty upstream
    cells on both faces every step.  At vertical order 5 the low-order eta
    flux once took the downstream cell at Courant numbers below 1, drained
    those empty cells, and the final clamp turned the drained mass into new
    tracer (on a 12 h 3 km smoke forecast, +52.5 t in transport against
    722 t emitted, with no inflow)."""
    import cupy as cp
    from gpuwm.core.chem_driver import ledger_report
    from gpuwm.core.dycore import run_steps

    cfg, state = _bubble_case(chem_sets="tracer_test", v_sca_adv_order=vorder)
    layer = np.zeros(state.chem_passive_1.shape, dtype=np.float32)
    layer[3:6] = 40.0
    state.chem_passive_1[...] = cp.asarray(layer)
    run_steps(state, cfg, 60)
    report = ledger_report(state)["rows"]["passive_1"]
    print("periodic tracer layer, order", vorder, report)
    assert report["initial_kg"] > 0
    assert abs(report["transport_kg"]) <= 2e-6 * report["initial_kg"], report
    assert abs(report["closure_kg"]) <= 1e-12 * report["initial_kg"]


def _wrf_flow_dep_reference(q, ru, rv, sz, inflow):
    """WRF flow_dep_bdy_chem's four loops, transcribed sequentially.

    chem/module_input_chem_data.F:1685-2008 with 1-based ibs=jbs=1,
    ibe=nx, jbe=ny mapped to 0-based indices, u/v read at WRF's own face
    indices (u(i,k,j) west face of cell i, u(i+1,k,j) its east face).
    """
    q = q.copy()
    nz, ny, nx = q.shape
    ibs, ibe, jbs, jbe = 0, nx - 1, 0, ny - 1
    for j in range(jbs, jbs + sz):                       # Y-start
        b = j - jbs
        for k in range(nz):
            for i in range(max(0, b + ibs), min(nx - 1, ibe - b) + 1):
                ii = min(max(i, ibs + sz), ibe - sz)
                q[k, j, i] = (q[k, jbs + sz, ii] if rv[k, j, i] < 0
                              else inflow)
    for j in range(jbe - sz + 1, jbe + 1):               # Y-end
        b = jbe - j
        for k in range(nz):
            for i in range(max(0, b + ibs), min(nx - 1, ibe - b) + 1):
                ii = min(max(i, ibs + sz), ibe - sz)
                q[k, j, i] = (q[k, jbe - sz, ii] if rv[k, j + 1, i] > 0
                              else inflow)
    for i in range(ibs, ibs + sz):                       # X-start
        b = i - ibs
        for k in range(nz):
            for j in range(max(0, b + jbs + 1), min(ny - 1, jbe - b - 1) + 1):
                jj = min(max(j, jbs + sz), jbe - sz)
                q[k, j, i] = (q[k, jj, ibs + sz] if ru[k, j, i] < 0
                              else inflow)
    for i in range(ibe - sz + 1, ibe + 1):               # X-end
        b = ibe - i
        for k in range(nz):
            for j in range(max(0, b + jbs + 1), min(ny - 1, jbe - b - 1) + 1):
                jj = min(max(j, jbs + sz), jbe - sz)
                q[k, j, i] = (q[k, jj, ibe - sz] if ru[k, j, i + 1] > 0
                              else inflow)
    return q


@pytest.mark.parametrize("sz", [1, 2, 3])
def test_flow_boundaries_match_wrf_loops_word_for_word(sz):
    import cupy as cp
    from types import SimpleNamespace
    from gpuwm.chem_table import load_sets
    from gpuwm.core.chem_transport import apply_chem_flow_boundaries

    nz, ny, nx = 4, 11, 13
    rng = np.random.default_rng(sz)
    q = rng.random((nz, ny, nx)).astype(np.float32)
    ru = rng.standard_normal((nz, ny, nx + 1)).astype(np.float32)
    rv = rng.standard_normal((nz, ny + 1, nx)).astype(np.float32)
    ru[0, 3, 0] = 0.0          # exact zeros are inflow on every side
    rv[1, 0, 5] = 0.0
    table = load_sets(["tracer_test"])
    row = table.rows[0]
    field = cp.asarray(q)
    state = SimpleNamespace(
        p=cp.zeros((nz, ny, nx), cp.float32),
        chem=SimpleNamespace(transported=(row,), table=table,
                             fields=lambda _s: (field,)))
    setattr(state, row.state_attr, field)
    # A specified root with no boundary source enabled: WRF's
    # have_bcs_chem false, every row on its default inflow.
    cfg = SimpleNamespace(spec_zone=sz, specified=True, chem_sources="")
    apply_chem_flow_boundaries(state, cfg, cp.asarray(ru), cp.asarray(rv))
    want = _wrf_flow_dep_reference(q, ru, rv, sz,
                                   np.float32(row.default_inflow))
    got = cp.asnumpy(getattr(state, row.state_attr))
    assert got.tobytes() == want.tobytes()


def test_specified_domain_tracer_blob_conserves_and_the_ledger_agrees():
    import cupy as cp
    from gpuwm.config import RunConfig
    from gpuwm.core.chem_driver import ledger_report
    from gpuwm.core.dycore import run_steps
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import init_at_rest
    from gpuwm.ingest.lateral_bc import (attach_lateral_boundaries,
                                         build_state_lateral_boundaries)

    nx, ny, nz = 30, 26, 16
    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=500.0, dy=500.0, ztop=8000.0,
                    dt=2.0, run_seconds=0.0, moist=True, specified=True,
                    chem_sets="tracer_test")
    vc = make_vertical_coord(nz)
    b = make_base_state(vc, lambda z: 300.0 * np.exp(
        1e-4 * np.asarray(z, float) / 9.81), p_surf=cfg.p_surf,
        ztop=cfg.ztop)
    s = init_at_rest(cfg, vc, b)
    boundaries = build_state_lateral_boundaries([s, s], [0.0, 3600.0])
    attach_lateral_boundaries(s, boundaries)
    xg = np.arange(nx + 1) * cfg.dx - 0.5 * nx * cfg.dx
    yg = np.arange(ny + 1) * cfg.dy - 0.5 * ny * cfg.dy
    R = 3.0 * cfg.dx
    psi = 5.0 * R * np.exp(-(xg[None, :] ** 2 + yg[:, None] ** 2)
                           / (2.0 * R * R))
    u2d = -(psi[1:, :] - psi[:-1, :]) / cfg.dy
    v2d = (psi[:, 1:] - psi[:, :-1]) / cfg.dx
    s.u[...] = cp.asarray(np.broadcast_to(u2d, (nz, ny, nx + 1)),
                          dtype=cp.float32)
    s.v[...] = cp.asarray(np.broadcast_to(v2d, (nz, ny + 1, nx)),
                          dtype=cp.float32)
    blob = np.zeros((nz, ny, nx), dtype=np.float32)
    blob[5:11, ny // 2 - 2:ny // 2 + 2, nx // 2 - 2:nx // 2 + 2] = 40.0
    s.chem_passive_1[...] = cp.asarray(blob)

    run_steps(s, cfg, 25)
    report = ledger_report(s)["rows"]["passive_1"]
    assert report["initial_kg"] > 0
    rel = abs(report["current_kg"] - report["initial_kg"]) / report[
        "initial_kg"]
    assert rel < 1e-4, report
    assert abs(report["transport_kg"]) / report["initial_kg"] < 1e-4
    assert abs(report["closure_kg"]) <= 1e-12 * report["initial_kg"]
    assert float(s.chem_passive_1.min()) >= 0.0


def test_inflow_brings_the_default_value_in_and_the_ledger_books_it():
    """A west-to-east mean flow on a specified domain: passive_2's inflow
    rows take its default_inflow (1 ug/kg), the tracer advances into the
    domain, the east outflow rows copy their inner neighbours, and the
    ledger files the gain under transport while passive_1 (zero inflow,
    zero start) stays exactly zero."""
    import cupy as cp
    from gpuwm.config import RunConfig
    from gpuwm.core.chem_driver import ledger_report
    from gpuwm.core.dycore import run_steps
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import init_at_rest
    from gpuwm.ingest.lateral_bc import (attach_lateral_boundaries,
                                         build_state_lateral_boundaries)

    nx, ny, nz = 30, 20, 12
    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=1000.0, dy=1000.0, ztop=8000.0,
                    dt=4.0, run_seconds=0.0, moist=True, specified=True,
                    chem_sets="tracer_test")
    vc = make_vertical_coord(nz)
    b = make_base_state(vc, lambda z: 300.0 * np.exp(
        1e-4 * np.asarray(z, float) / 9.81), p_surf=cfg.p_surf,
        ztop=cfg.ztop)
    s = init_at_rest(cfg, vc, b)
    s.u[...] = np.float32(10.0)
    boundaries = build_state_lateral_boundaries([s, s], [0.0, 3600.0])
    attach_lateral_boundaries(s, boundaries)
    run_steps(s, cfg, 30)
    q2 = cp.asnumpy(s.chem_passive_2)
    q1 = cp.asnumpy(s.chem_passive_1)
    assert not q1.any()
    sz = cfg.spec_zone
    # West spec columns (away from the Y-owned corners) are inflow.
    assert (q2[:, sz + 1:ny - sz - 1, :sz] == np.float32(1.0)).all()
    # The tracer has entered the interior (120 s at 10 m/s is about one
    # 1 km cell, so the front is smeared over the first few columns and
    # decreases eastward) but has not crossed the domain.
    mid = q2[:, ny // 2, :]
    assert mid[:, sz].min() > 0.1
    assert (mid[:, sz] > mid[:, sz + 2]).all()
    assert mid[:, nx - sz - 2].max() < 1e-3
    report = ledger_report(s)["rows"]
    assert report["passive_1"]["current_kg"] == 0.0
    assert report["passive_2"]["transport_kg"] > 0.0
    assert report["passive_2"]["initial_kg"] >= 0.0
    assert abs(report["passive_2"]["closure_kg"]) <= 1e-9 * (
        report["passive_2"]["current_kg"])


@pytest.mark.parametrize("vorder", [3, 5])
def test_the_monotonic_final_stage_moves_no_other_byte(vorder):
    import cupy as cp
    from gpuwm.core.dycore import run_steps

    mixing = dict(diff_6th_opt=2, diff_6th_factor=0.12,
                  v_sca_adv_order=vorder)
    cfg0, plain = _bubble_case(**mixing)
    cfg1, chem = _bubble_case(**mixing, chem_sets="tracer_test",
                              chem_adv_opt=2)
    chem.chem_passive_1[...] = chem.qv
    run_steps(plain, cfg0, 6)
    run_steps(chem, cfg1, 6)
    for name in _MET:
        a = cp.asnumpy(getattr(plain, name))
        b = cp.asnumpy(getattr(chem, name))
        assert a.tobytes() == b.tobytes(), name
    # The monotonic stage is a different operator from qv's PD one.
    assert (cp.asnumpy(chem.qv).tobytes()
            != cp.asnumpy(chem.chem_passive_1).tobytes())


def _vortex_blob(chem_adv_opt):
    import cupy as cp
    from gpuwm.config import RunConfig
    from gpuwm.core.dycore import run_steps
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import init_at_rest
    from gpuwm.ingest.lateral_bc import (attach_lateral_boundaries,
                                         build_state_lateral_boundaries)

    nx, ny, nz = 30, 26, 16
    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=500.0, dy=500.0, ztop=8000.0,
                    dt=2.0, run_seconds=0.0, moist=True, specified=True,
                    chem_sets="tracer_test", chem_adv_opt=chem_adv_opt)
    vc = make_vertical_coord(nz)
    b = make_base_state(vc, lambda z: 300.0 * np.exp(
        1e-4 * np.asarray(z, float) / 9.81), p_surf=cfg.p_surf,
        ztop=cfg.ztop)
    s = init_at_rest(cfg, vc, b)
    boundaries = build_state_lateral_boundaries([s, s], [0.0, 3600.0])
    attach_lateral_boundaries(s, boundaries)
    xg = np.arange(nx + 1) * cfg.dx - 0.5 * nx * cfg.dx
    yg = np.arange(ny + 1) * cfg.dy - 0.5 * ny * cfg.dy
    R = 3.0 * cfg.dx
    psi = 5.0 * R * np.exp(-(xg[None, :] ** 2 + yg[:, None] ** 2)
                           / (2.0 * R * R))
    u2d = -(psi[1:, :] - psi[:-1, :]) / cfg.dy
    v2d = (psi[:, 1:] - psi[:, :-1]) / cfg.dx
    s.u[...] = cp.asarray(np.broadcast_to(u2d, (nz, ny, nx + 1)),
                          dtype=cp.float32)
    s.v[...] = cp.asarray(np.broadcast_to(v2d, (nz, ny + 1, nx)),
                          dtype=cp.float32)
    blob = np.zeros((nz, ny, nx), dtype=np.float32)
    blob[5:11, ny // 2 - 3:ny // 2 + 3, nx // 2 - 1:nx // 2 + 4] = 40.0
    s.chem_passive_1[...] = cp.asarray(blob)
    run_steps(s, cfg, 25)
    return s


def test_the_monotonic_stage_keeps_a_blob_inside_its_bounds():
    """Positive definite bounds the minimum only, so a sharp-edged blob
    swept by a vortex overshoots its 40 plateau; the monotonic limiter
    (Smolarkiewicz's two-sided renormalization, module_advect_em.F:
    10374-10447) holds both bounds, up to float32 rounding of the coupled
    update, and the ledger still closes."""
    from gpuwm.core.chem_driver import ledger_report

    pd = _vortex_blob(1)
    mono = _vortex_blob(2)
    pd_max = float(pd.chem_passive_1.max())
    mono_max = float(mono.chem_passive_1.max())
    print("vortex blob maximum after 25 steps, PD", pd_max, "mono", mono_max)
    assert pd_max > 40.0 * (1 + 1e-3), pd_max
    assert mono_max <= 40.0 * (1 + 1e-5), mono_max
    assert float(mono.chem_passive_1.min()) >= 0.0
    report = ledger_report(mono)["rows"]["passive_1"]
    rel = abs(report["current_kg"] - report["initial_kg"]) / report[
        "initial_kg"]
    assert rel < 1e-4, report
    assert abs(report["closure_kg"]) <= 1e-12 * report["initial_kg"]
