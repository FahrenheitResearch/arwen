"""Extra WOOF dumps for the base4 dycore lane (copies only, no arithmetic):
the Smagorinsky momentum rows before and after the vertical pass, the acoustic
coefficients, and the folded relaxation rows."""
import cupy as cp
import gpuwm.core.dycore as D
import woof_hook as H


def _on(state, cfg):
    return getattr(state, "_loc_dt", None) is not None and H.active(state, cfg)


orig_vert = D.launch_wrf_smag2d_vertical


def vert(state, cfg, km, **k):
    on = _on(state, cfg)
    if on:
        H.dump(state, "c01_hsmag", **{"h_" + n: k[n].copy() for n in ("ru", "rv", "rw", "rth") if k.get(n) is not None})
    r = orig_vert(state, cfg, km, **k)
    if on:
        H.dump(state, "c02_vsmag", **{"hv_" + n: k[n].copy() for n in ("ru", "rv", "rw", "rth") if k.get(n) is not None})
    return r


D.launch_wrf_smag2d_vertical = vert

orig_coef = D.prepare_acoustic_coefficients


def coef(state, cfg, dtau, **k):
    r = orig_coef(state, cfg, dtau, **k)
    if _on(state, cfg):
        H.dump(state, "f0_coef", c2a=r[0], a=r[1], alpha=r[2], gam=r[3])
    return r


D.prepare_acoustic_coefficients = coef

orig_cap = D.capture_folded_relaxation


def cap(state, cfg):
    r = orig_cap(state, cfg)
    if _on(state, cfg):
        H.dump(state, "e0_relax", **{"relax_" + n: v.copy() for n, v in r.items()})
    return r


D.capture_folded_relaxation = cap

# --- moisture path (qv only) -------------------------------------------------
import gpuwm.core.moist as M

orig_adv_sc = D.advance_scalars_stage


def adv_sc(state, cfg, ru, rv, ww, *a, **k):
    if _on(state, cfg):
        H.dump(state, "q0_fluxes", ru_m=ru, rv_m=rv, ww_m=ww)
    r = orig_adv_sc(state, cfg, ru, rv, ww, *a, **k)
    if _on(state, cfg):
        H.dump(state, "q9_updated")
    return r


D.advance_scalars_stage = adv_sc
_CUR = {}

orig_fd = M.launch_flux_div_scalar


def fd(q, ru, rv, ww, tend, state, *a, **k):
    r = orig_fd(q, ru, rv, ww, tend, state, *a, **k)
    if q is getattr(state, "qv", None) and getattr(state, "_loc_dt", None) is not None \
            and H.active(state, type("C", (), {"dt": state._loc_dt})()):
        H.dump(state, "q1_advect", advect=tend.copy())
        _CUR["state"] = state
    return r


M.launch_flux_div_scalar = fd

orig_up = M._update_scalar_in_place


def up(q, q0, tend, *a, **k):
    st = _CUR.get("state")
    if st is not None and q is getattr(st, "qv", None):
        extra = {"tend_in": tend.copy()}
        for n in ("physics", "fixed"):
            if k.get(n) is not None:
                extra[n] = k[n].copy()
        H.dump(st, "q2_update_in", **extra)
    return orig_up(q, q0, tend, *a, **k)


M._update_scalar_in_place = up

# --- coupled physics tendencies handed to the strict fold -------------------
orig_pft2 = D.prepare_fixed_tendencies


def pft2(state, cfg, physics_tendencies=None):
    state._loc_dt = cfg.dt
    if physics_tendencies is not None and _on(state, cfg):
        ex = {n: getattr(physics_tendencies, n).copy() for n in ("ru", "rv", "rtheta", "rqv", "rqc")
              if getattr(physics_tendencies, n, None) is not None}
        H.dump(state, "c_phys", **ex)
    return orig_pft2(state, cfg, physics_tendencies)


D.prepare_fixed_tendencies = pft2
