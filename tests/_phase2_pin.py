"""Shared state builders for the Phase 3 Task 3 msf==1/f==0 bitwise pin.

These builders are imported both by the one-shot generator that captured
``tests/data/phase2_step_regression.npz`` at the pre-Task-3 tree (Phase 2
code, commit add26dc) and by ``tests/test_coriolis_map.py``'s regression
test after the map-factor/Coriolis wiring landed.  They must therefore
never acquire Task-3 features: the states they return carry the DEFAULT
map factors (1) and Coriolis parameters (0), which the plan pins to be
bitwise-identical to Phase 2 through any number of full ``dycore.step``
calls.

Each builder returns ``(state, cfg)`` ready for ``run_steps``.
"""

from __future__ import annotations

import numpy as np

from gpuwm.config import RunConfig

#: Fields compared bitwise after the pinned number of steps, per case.
PIN_FIELDS = ("u", "v", "w", "thp", "php", "mup")
PIN_STEPS = 5


def _bubble(amp=2.0, zc=2000.0, rz=1500.0, rx=2000.0):
    def thp(x, z):
        zz = z[:, None, None] if np.ndim(z) == 1 else z
        L = np.sqrt((x[None, None, :] / rx) ** 2 + ((zz - zc) / rz) ** 2)
        return np.where(L < 1.0, amp * np.cos(np.pi * L / 2) ** 2, 0.0) \
            * np.ones((zz.shape[0] if zz.ndim == 3 else len(z), 1, 1))
    return thp


def _bubble3(cfg, amp=2.0, zc=2000.0, rz=1500.0, rx=2000.0, ry=2000.0):
    y = (np.arange(cfg.ny) + 0.5) * cfg.dy - 0.5 * cfg.ny * cfg.dy

    def thp(x, z):
        zz = z[:, None, None] if np.ndim(z) == 1 else z
        L = np.sqrt((x[None, None, :] / rx) ** 2
                    + (y[None, :, None] / ry) ** 2 + ((zz - zc) / rz) ** 2)
        return np.where(L < 1.0, amp * np.cos(np.pi * L / 2) ** 2, 0.0) \
            * np.ones((cfg.nz, cfg.ny, cfg.nx))
    return thp


def _shear_u(s, cfg, u0=5.0):
    """Deterministic weak shear + y-variation on u (FP32 device fill)."""
    import cupy as cp
    z = s.height_half()
    zz = z if np.ndim(z) == 3 else np.broadcast_to(
        np.asarray(z)[:, None, None], (cfg.nz, cfg.ny, cfg.nx))
    prof = u0 * np.tanh(zz / 3000.0)                       # (nz, ny, nx)
    jvar = 1.0 + 0.1 * np.sin(2 * np.pi * np.arange(cfg.ny) / cfg.ny)
    u = prof * jvar[None, :, None]
    u = np.concatenate([u, u[:, :, :1]], axis=2)           # periodic dup
    s.u[...] = cp.asarray(u, dtype=cp.float32)


def build_dry_flat():
    """Dry flat periodic acoustic case (Phase 1 core paths)."""
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import init_theta_perturbation
    cfg = RunConfig(nx=14, ny=12, nz=16, dx=500.0, dy=500.0, ztop=8000.0,
                    dt=2.0, run_seconds=0.0)
    vc = make_vertical_coord(cfg.nz)
    th = lambda z: 300.0 * np.exp(1e-4 * np.asarray(z, float) / 9.81)
    b = make_base_state(vc, th, p_surf=cfg.p_surf, ztop=cfg.ztop)
    s = init_theta_perturbation(cfg, vc, b, _bubble3(cfg))
    _shear_u(s, cfg)
    return s, cfg


def build_moist():
    """Moist Kessler + PD transport + km_opt=4 + diff6 (Phase 2 paths)."""
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import init_moist_balanced
    cfg = RunConfig(nx=14, ny=12, nz=16, dx=500.0, dy=500.0, ztop=8000.0,
                    dt=2.0, run_seconds=0.0, moist=True, mp_physics=1,
                    diff_6th_opt=2, diff_6th_factor=0.12, km_opt=4)
    vc = make_vertical_coord(cfg.nz)
    th = lambda z: 300.0 * np.exp(1e-4 * np.asarray(z, float) / 9.81)
    b = make_base_state(vc, th, p_surf=cfg.p_surf, ztop=cfg.ztop)
    qv = lambda z: 0.012 * np.exp(-np.asarray(z, float) / 2500.0)
    s = init_moist_balanced(cfg, vc, b, qv, thp_func=_bubble3(cfg, amp=3.0))
    _shear_u(s, cfg)
    return s, cfg


def build_terrain():
    """Bell-hill terrain + hybrid coordinate (Phase 2 Task 4 paths)."""
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import init_theta_perturbation
    from gpuwm.core.terrain import bell_hill
    cfg = RunConfig(nx=16, ny=10, nz=16, dx=1000.0, dy=1000.0, ztop=10000.0,
                    dt=2.0, run_seconds=0.0, terrain_opt=1, hybrid_opt=2,
                    hill_height=300.0, hill_halfwidth=3000.0)
    vc = make_vertical_coord(cfg.nz, hybrid_opt=cfg.hybrid_opt, etac=cfg.etac)
    th = lambda z: 300.0 * np.exp(1e-4 * np.asarray(z, float) / 9.81)
    tz = bell_hill(cfg)
    b = make_base_state(vc, th, p_surf=cfg.p_surf, ztop=cfg.ztop,
                        terrain_z=tz)
    s = init_theta_perturbation(cfg, vc, b, _bubble3(cfg, zc=3000.0))
    _shear_u(s, cfg, u0=3.0)
    from gpuwm.core.dycore import set_w_surface
    set_w_surface(s, cfg)
    return s, cfg


def build_open():
    """Open lateral boundaries + emdiv + w_damping + smag/diff6 (Task 9-11
    paths); dry."""
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import init_theta_perturbation
    cfg = RunConfig(nx=16, ny=14, nz=16, dx=500.0, dy=500.0, ztop=8000.0,
                    dt=2.0, run_seconds=0.0, open_x=True, open_y=True,
                    emdiv=0.01, w_damping=1, diff_6th_opt=2, km_opt=4)
    vc = make_vertical_coord(cfg.nz)
    th = lambda z: 300.0 * np.exp(1e-4 * np.asarray(z, float) / 9.81)
    b = make_base_state(vc, th, p_surf=cfg.p_surf, ztop=cfg.ztop)
    s = init_theta_perturbation(cfg, vc, b, _bubble3(cfg))
    _shear_u(s, cfg, u0=2.0)
    return s, cfg


CASES = {"dry_flat": build_dry_flat, "moist": build_moist,
         "terrain": build_terrain, "open": build_open}
