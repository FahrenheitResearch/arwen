"""Step-1 state dumps from WOOF's dycore, by wrapping the module-level calls dycore.step makes.
Import before the run; dumps go to $WOOF_LOCDUMP as <seq>_<tag>.npz holding every device array on the state
(plus the call's extra arrays). Only while state.elapsed_seconds < cfg.dt * $WOOF_LOCDUMP_STEPS (default 1).
No arithmetic is changed: wrappers call the original and copy arrays to host."""
import os, numpy as np, cupy as cp
from pathlib import Path
import gpuwm.core.dycore as D
OUT = Path(os.environ["WOOF_LOCDUMP"]); OUT.mkdir(parents=True, exist_ok=True)
NSTEPS = int(os.environ.get("WOOF_LOCDUMP_STEPS", "1"))
SEQ = [0]
CTX = {"stage": 0, "sub": 0}

FROM = int(os.environ.get("WOOF_LOCDUMP_FROM", "1"))
def active(state, cfg):
    return cfg.dt * (FROM - 1) - 1e-6 < state.elapsed_seconds < cfg.dt * NSTEPS - 1e-6

def dump(state, tag, **extra):
    cp.cuda.runtime.deviceSynchronize()
    arrays = {}
    for k, v in vars(state).items():
        if isinstance(v, cp.ndarray) and v.size < 50_000_000:
            arrays[k] = cp.asnumpy(v)
    for k, v in extra.items():
        if isinstance(v, cp.ndarray):
            arrays["x_" + k] = cp.asnumpy(v)
    step = int(round(state.elapsed_seconds / max(state._loc_dt, 1e-9))) + 1
    name = f"{SEQ[0]:04d}_s{step:03d}_rk{CTX['stage']}_it{CTX['sub']:02d}_{tag}.npz"
    np.savez(OUT / name, **arrays)
    SEQ[0] += 1

def wrap(modname, attr, after_tag=None, before_tag=None, extra_from_result=None):
    orig = getattr(D, attr)
    def w(state, cfg, *a, **k):
        on = active(state, cfg)
        state._loc_dt = cfg.dt
        if on and before_tag:
            dump(state, before_tag)
        r = orig(state, cfg, *a, **k)
        if on and after_tag:
            extra = extra_from_result(r) if extra_from_result else {}
            dump(state, after_tag, **extra)
        return r
    setattr(D, attr, w)

# stage counter: zero_tendencies happens first in each stage; we track via stage_fluxes
orig_sf = D.stage_fluxes
def sf(state, cfg, *a, **k):
    r = orig_sf(state, cfg, *a, **k)
    state._loc_dt = cfg.dt
    if active(state, cfg):
        CTX["stage"] += 1; CTX["sub"] = 0
        ru, rv, ww = r
        dump(state, "b_prep", ru=ru, rv=rv, ww=ww)
    return r
D.stage_fluxes = sf

orig_pft = D.prepare_fixed_tendencies
def pft(state, cfg, *a, **k):
    state._loc_dt = cfg.dt
    on = active(state, cfg)
    if on:
        dump(state, "a_start")
    r = orig_pft(state, cfg, *a, **k)
    if on:
        extra = {}
        for row in D._smag2d_specs(state, None, None, time_t=True):
            f0, slot = row[0], row[5]
            extra[slot] = state.scratch(f0.shape, slot)
        dump(state, "c_fixed", **extra)
    return r
D.prepare_fixed_tendencies = pft
orig_slow = D._add_slow_tendencies
def slow(state, cfg, ru, rv, ww, **k):
    r = orig_slow(state, cfg, ru, rv, ww, **k)
    if active(state, cfg):
        dump(state, "d_slow")
    return r
D._add_slow_tendencies = slow
wrap("D", "add_fixed_dry_tendencies", after_tag="d2_fixedadded")
orig_wd = D.apply_w_damping
def wd(state, cfg, ww, *a, **k):
    r = orig_wd(state, cfg, ww, *a, **k)
    if active(state, cfg):
        dump(state, "d3_wdamp")
    return r
D.apply_w_damping = wd
orig_lat = D.apply_state_lateral_boundaries
def lat(state, cfg, *a, **k):
    r = orig_lat(state, cfg, *a, **k)
    if active(state, cfg):
        dump(state, "e_lateral")
    return r
D.apply_state_lateral_boundaries = lat

def wrap_factory(attr, tag, count_sub=False):
    orig = getattr(D, attr)
    def fac(state, cfg, *a, **k):
        launch = orig(state, cfg, *a, **k)
        if launch is None:
            return None
        def run(*la, **lk):
            r = launch(*la, **lk)
            if active(state, cfg):
                if count_sub:
                    CTX["sub"] += 1
                dump(state, tag)
            return r
        return run
    setattr(D, attr, fac)
wrap_factory("_prepare_small_step_init_launch", "f_ssinit")
wrap_factory("prepare_acoustic_substep_launch", "g_ss", count_sub=True)
wrap_factory("_prepare_small_step_finish_launch", "h_finish")

orig_bv = D.apply_state_boundary_values
def bv(state, cfg, *a, **k):
    on = getattr(state, "_loc_dt", None) is not None and active(state, cfg)
    if on:
        dump(state, "y_before_bdyvalues")
    r = orig_bv(state, cfg, *a, **k)
    if on:
        dump(state, "y_after_bdyvalues")
    return r
D.apply_state_boundary_values = bv
orig_sw = D.set_w_surface
def sw(state, cfg, *a, **k):
    r = orig_sw(state, cfg, *a, **k)
    if getattr(state, "_loc_dt", None) is not None and active(state, cfg):
        CTX["stage"] = 9
        dump(state, "z_after_wsurface")
    return r
D.set_w_surface = sw

orig_smag = D._compute_wrf_smag_tendencies
def smag(state, cfg, *a, **k):
    r = orig_smag(state, cfg, *a, **k)
    if active(state, cfg):
        extra = {}
        for row in D._smag2d_specs(state, None, None, time_t=True):
            extra[row[5]] = state.scratch(row[0].shape, row[5])
        dump(state, "c0_smag", **extra)
    return r
D._compute_wrf_smag_tendencies = smag

orig_km = D.launch_wrf_smag2d_km
def smagkm(state, cfg, xkmh, xkhh, *a, **k):
    r = orig_km(state, cfg, xkmh, xkhh, *a, **k)
    if active(state, cfg):
        d11, d22, d12 = r
        dump(state, "c00_deform", d11=d11, d22=d22, d12=d12, xkmh=xkmh, xkhh=xkhh)
    return r
D.launch_wrf_smag2d_km = smagkm
