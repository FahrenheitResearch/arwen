"""Host-side strict-mode pieces of combo-sweep round 3, against WRF 4.6.1.

Each test pins one host-side number or order the localize pair (WRF with dump
hooks against WOOF's strict engine) showed WRF forms differently.  Run in a
process started with ``GPUWM_WRF_EXACT=1``; the module flags are read at
import, so a default process skips these.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

f32 = np.float32

pytestmark = pytest.mark.skipif(os.environ.get("GPUWM_WRF_EXACT") != "1",
                                reason="requires a process started with GPUWM_WRF_EXACT=1")


def test_specified_relaxation_weights_follow_lbc_fcx_gcx_in_real():
    """lbc_fcx_gcx, specified branch: fcx = 0.1/dt*(sz+rz-loop)/(rz-1) and
    gcx = 1.0/dt/50.*(...)/(rz-1) in REAL, times exp(-(loop-(sz+1))*spec_exp)."""
    from gpuwm.ingest.lateral_bc import _weights

    for dt in (15.0, 18.0, 20.0, 7.5, 12.5):
        fcx, gcx = _weights(5, 1, 4, dt, 0.0)
        for loop in range(2, 6):
            n, d = f32(5 - loop), f32(3)
            f = f32(f32(f32(f32(0.1) / f32(dt)) * n) / d)
            g = f32(f32(f32(f32(f32(1.0) / f32(dt)) / f32(50.0)) * n) / d)
            assert fcx[loop - 1].view(np.uint32) == f.view(np.uint32), (dt, loop)
            assert gcx[loop - 1].view(np.uint32) == g.view(np.uint32), (dt, loop)
    # The float64 law rounds once at the end and is a different word for
    # some dt, which is why the strict path exists.
    differs = []
    for dt in np.arange(5.0, 60.0, 0.5):
        for loop in range(2, 6):
            ramp = (5 - loop) / 3
            f = f32(f32(f32(f32(0.1) / f32(dt)) * f32(5 - loop)) / f32(3))
            differs.append(f32(0.1 / dt * ramp * 1.0) != f)
    assert any(differs)


def test_dampmag_is_formed_in_real():
    """advance_w: dampmag = dts*dampcoef, both default REAL (every build
    since 2.8.8; tests/test_upper_damping_wrf461_oracle.py holds it in the
    default process)."""
    from gpuwm.core.acoustic import damp_magnitude

    cfg = SimpleNamespace(damp_opt=3, dampcoef=0.2)
    for dtau in (5.0, 3.75, 3.0, 2.5, 6.666666666666667):
        want = f32(f32(dtau) * f32(0.2))
        assert np.float32(damp_magnitude(cfg, dtau)).view(np.uint32) == want.view(np.uint32)
    assert damp_magnitude(SimpleNamespace(damp_opt=0, dampcoef=0.2), 5.0) == 0.0


def test_external_mode_scale_is_formed_in_real():
    """advance_uv: mudf_xy = -emdiv*dx*(...), -emdiv*dx a REAL product."""
    from gpuwm.core.acoustic import _uv_emdiv_args

    state = SimpleNamespace(mup=object(), msfu=object(), msfv=object(), has_msf=True)
    cfg = SimpleNamespace(emdiv=0.01, dx=3000.0, dy=2500.0)
    args = _uv_emdiv_args(state, cfg, mudf=np.zeros(1, np.float32))
    assert args[3] == f32(f32(-0.01) * f32(3000.0))
    assert args[4] == f32(f32(-0.01) * f32(2500.0))
    assert int(args[5]) == 1
    assert int(_uv_emdiv_args(state, SimpleNamespace(emdiv=0.0, dx=1.0, dy=1.0),
                              mudf=None)[5]) == 0


def test_strict_moisture_rule_keeps_wrfs_own_negative_residuals():
    """WRF's scalar update leaves qv = -3.19e-13 after step 1 and carries
    -2.4808048e-07 at step 11 of the round-3 pair (stock 4.6.1 and strict
    WOOF hold the same word); strict mode keeps the ceiling only."""
    from gpuwm.core.health import rule_for_field

    rule = rule_for_field("qv")
    assert rule.lower is None
    assert rule.upper == 1.0
    assert rule.bound_note


def test_folded_relaxation_rows_are_rk_addtend_drys():
    """rk_addtend_dry folds u_save, v_save, t_save and w_save (nest or
    relax_w) into *_tendf; ph_tendf holds nothing else."""
    from gpuwm.ingest.lateral_bc import folded_relaxation_rows

    plain = SimpleNamespace(nested=False, specified=True, relax_w=False)
    nest = SimpleNamespace(nested=True, specified=False, relax_w=False)
    assert folded_relaxation_rows(plain) == ("u", "v", "theta")
    assert folded_relaxation_rows(nest) == ("u", "v", "theta", "w")


def test_step_end_follows_solve_ems_order(monkeypatch):
    """calc_p_rho_phi, microphysics and its calc_p_rho_phi, then spec_bdy_final
    and set_w_surface (solve_em); the default order diagnosed after the
    boundary values and moved P on the specified ring."""
    from gpuwm.core import dycore

    calls = []
    monkeypatch.setattr(dycore, "update_diagnostics",
                        lambda state, h, muts=None: calls.append(("diag", muts)))
    monkeypatch.setattr(dycore, "apply_microphysics",
                        lambda state, cfg, dt, refl_10cm_due=False: calls.append(("mp",)) or "r")
    monkeypatch.setattr(dycore, "apply_state_boundary_values",
                        lambda state, cfg, t: calls.append(("bdy",)))
    monkeypatch.setattr(dycore, "set_w_surface", lambda state, cfg: calls.append(("wsfc",)))
    monkeypatch.setattr(dycore, "_wrf_muts", lambda state: "muts")
    physics = SimpleNamespace(accept_microphysics=lambda r, dt: calls.append(("accept",)))
    mu = np.ones((3, 4), np.float32)
    halo = np.zeros_like(mu)
    state = SimpleNamespace(physics=physics, elapsed_seconds=0.0, mup=mu,
                            scratch=lambda shape, name: halo)
    cfg = SimpleNamespace(hypsometric_opt=2, mp_physics=6, dt=15.0, nwp_diagnostics=0,
                          specified=True, nested=False)
    dycore._wrf_step_epilogue(state, cfg, False)
    assert calls == [("diag", "muts"), ("mp",), ("accept",), ("diag", "muts"), ("bdy",), ("wsfc",)]
    np.testing.assert_array_equal(halo, mu)


def test_physics_fold_adds_each_held_row_once(monkeypatch):
    """update_phy_ten's coupled tendencies join the held *_tendf (rk_addtend_dry
    divides the sum once); scalar rows by species name."""
    from gpuwm.core import dycore

    slots = {name: np.zeros(3, np.float32) for name in
             ("smag_ru", "smag_rv", "smag_rw", "smag_rth", "smag_rqv", "smag_rqc")}
    state = SimpleNamespace(scratch=lambda shape, name: slots[name])
    f0 = np.zeros(3, np.float32)
    specs = [(f0, None, None, None, None, name, "") for name in slots]
    pt = SimpleNamespace(ru=np.full(3, 1, np.float32), rv=np.full(3, 2, np.float32),
                         rw=None, rtheta=np.full(3, 3, np.float32),
                         scalar_for=lambda n: {"qv": np.full(3, 4, np.float32)}.get(n))
    dycore._fold_physics_wrf(state, specs, pt)
    assert [float(slots[k][0]) for k in ("smag_ru", "smag_rv", "smag_rw", "smag_rth",
                                          "smag_rqv", "smag_rqc")] == [1, 2, 0, 3, 4, 0]
