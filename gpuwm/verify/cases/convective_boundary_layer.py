# gpuwm/verify/cases/convective_boundary_layer.py
"""Dry convective boundary layer under the km_opt=3 LES closure.

A doubly periodic, flat, dry free-convection case in the shape of WRF's
em_les reference (test/em_les: prescribed kinematic surface heat flux
driving a CBL, PBL off, diff_opt=2, double periodic, README.les), run
with the 3-D Smagorinsky closure this engine ports (km_opt=3,
mix_isotropic=1, c_s=0.18 -- the em_les-recommended constant) and the
prescribed-flux lower boundary (isfflx=0: constant tke_heat_flux heat,
constant tke_drag_coefficient drag, no moisture flux -- WRF
vertical_diffusion_2 CASE(0)).

Two entry points:

``run(outdir)`` -- the VERIFY smoke: a small short integration whose GATES
are validity-only (finite fields, bounded w and CFL, dry-mass closure on
the periodic domain).  The convective-statistics numbers (heat-flux
profile, z_i, resolved TKE) are computed and FILED in the metrics dict as
receipts, not gated: pass bands for LES statistics are measure-then-commit
against the WRF oracle per the LES program spec, and inventing them here
would calibrate gates to their own first measurement.

``main(argv)`` -- the sized LES driver: the full case (default 96 x 96 x
64 at dx = 100 m, two simulated hours) with periodic profile sampling,
writing one receipts JSON plus one compact npz (slab-mean profiles per
sample and a single final-state w slab) sized in kilobytes-to-megabytes,
never per-step 3-D history.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from gpuwm.config import RunConfig
from gpuwm.core import tke_budget
from gpuwm.core.grid import make_base_state, make_vertical_coord

#: Validity gates only (see module docstring).  LES-statistics bands are
#: measure-then-commit vs the WRF oracle; nothing here pretends otherwise.
GATES = {
    "w_max": (None, 50.0),
    "cfl_max": (None, 1.0),
    "mass_drift_rel": (None, 1.0e-4),
}

#: Receipt fields the dual-run determinism screen must SKIP, because
#: they measure the run's ENVIRONMENT rather than the run.  The screen
#: ("compare every numeric receipt field", ARWEN-REALISATION-SPREAD.md)
#: is the corruption detector on cards without ECC, so its field set
#: must contain only quantities two healthy same-seed runs are bound to
#: agree on.  Wall-clock timings differ between identical runs by
#: construction.  The device-wide VRAM figure ((total - free) from
#: ``cudaMemGetInfo``) counts every co-tenant process on the card: the
#: sm_89 (RTX 4090) stress lane reproduced two same-seed reps that were
#: byte-identical in every other field and differed ONLY here (0.72 vs
#: 1.60 GiB) while a forecast shared the card -- a false positive of
#: the screen, not of the engine.  ``vram_gib`` is the name that figure
#: carried before the split (receipts through cbl-2026-08-02); it stays
#: classified so committed receipts partition without being edited.
#: The per-process ``vram_pool_gib`` is deliberately NOT here: this
#: process's live pool bytes are a deterministic property of the run
#: and belong on the screened side (see :func:`_vram_fields`).
ENVIRONMENTAL_FIELDS = frozenset({
    "wall_seconds",
    "steps_per_second",
    "vram_device_gib",
    "vram_gib",  # pre-split name of the device-wide figure
})


def partition_receipt_fields(receipt: dict) -> tuple[dict, dict]:
    """Split a receipt's fields into ``(screened, environmental)``.

    Screened fields are the determinism-comparison surface: two
    same-seed runs on one card must agree on every one of them, and a
    disagreement is evidence of silent corruption.  Environmental
    fields describe the machine at the moment of measurement --
    comparing them makes the screen fail on healthy runs whenever
    anything else holds VRAM on the card.  Every field lands in exactly
    one bucket, so nothing can silently fall out of the screen.
    """
    screened = {k: v for k, v in receipt.items()
                if k not in ENVIRONMENTAL_FIELDS}
    environmental = {k: v for k, v in receipt.items()
                     if k in ENVIRONMENTAL_FIELDS}
    return screened, environmental


def _vram_fields(*, pool_used_bytes, device_free_total) -> dict:
    """Both VRAM receipt figures, from injected readers so the
    accounting is CPU-testable (the ``gpu_mem_watch`` probe pattern).

    ``vram_pool_gib`` -- this process's live CuPy default-pool bytes:
    a property of the run itself, identical between two same-seed runs
    whatever else the card is doing, so it belongs to the determinism
    screen.

    ``vram_device_gib`` -- whole-card occupancy as the CUDA runtime
    reports it, ``(total - free)`` from ``cudaMemGetInfo``: co-tenant
    processes inflate it wherever the platform exposes them, so it is
    an ENVIRONMENTAL field (:data:`ENVIRONMENTAL_FIELDS`) and the
    screen must not compare it.  It stays on the receipt because it is
    the number an operator sizing a card wants -- it was just mislabeled
    as a run property under its old name ``vram_gib``.

    A failing reader yields ``None`` for its own field and leaves the
    other alone: a missing memory figure must not kill a finished
    integration.
    """
    fields: dict = {"vram_pool_gib": None, "vram_device_gib": None}
    try:
        fields["vram_pool_gib"] = float(pool_used_bytes()) / 2.0**30
    except Exception:
        pass
    try:
        free_b, total_b = device_free_total()
        fields["vram_device_gib"] = float(total_b - free_b) / 2.0**30
    except Exception:
        pass
    return fields

#: em_les reference forcing (test/em_les/namelist.input: tke_heat_flux =
#: 0.24 K m/s; README.les "The turbulence of the free CBL is
#: driven/maintained by the surface heat flux").  The drag coefficient is
#: the em_les SGP variant's 0.0013 -- the canonical prescribed-drag value.
HEAT_FLUX = 0.24
DRAG_COEFFICIENT = 0.0013
C_S = 0.18
#: Trailing samples averaged for the citable entrainment ratio.  20 at the
#: default 60 s sampling is 20 minutes -- long enough that the statistic is
#: stable to ~3% and short enough to stay inside quasi-steady CBL growth.
_ENTRAINMENT_WINDOW = 20


def _theta_profile(z):
    """Mixed layer capped by a stable lapse: 300 K to 1000 m, then
    +3 K/km (a free-convection CBL sounding in the em_les shape)."""
    z = np.asarray(z, float)
    return 300.0 + np.where(z > 1000.0, 0.003 * (z - 1000.0), 0.0)


def make_config(*, nx=96, ny=96, nz=64, dx=100.0, ztop=2400.0, dt=0.5,
                minutes=120.0, heat_flux=HEAT_FLUX,
                drag_coefficient=DRAG_COEFFICIENT, c_s=C_S,
                km_opt=3, c_k=0.10, surface="prescribed",
                tke_budget=0, restart_interval_s=0.0,
                lateral="periodic") -> RunConfig:
    if surface not in ("prescribed", "most"):
        raise ValueError(
            f"surface must be 'prescribed' (isfflx=0 constant drag/heat) "
            f"or 'most' (isfflx=2: SFCLAY USTM/QFX with the constant "
            f"heat flux -- the em_les hybrid), got {surface!r}")
    if lateral not in ("periodic", "specified"):
        raise ValueError(
            f"lateral must be 'periodic' (the em_les topology) or "
            f"'specified' (a walled domain that exercises WRF's TKE "
            f"lateral arm, flow_dep_bdy), got {lateral!r}")
    most = surface == "most"
    return RunConfig(
        nx=nx, ny=ny, nz=nz, dx=dx, dy=dx, ztop=ztop, dt=dt,
        run_seconds=minutes * 60.0,
        km_opt=km_opt, mix_isotropic=1, c_s=c_s, c_k=c_k,
        bl_pbl_physics=0,
        sf_sfclay_physics=1 if most else 0,
        # em_les surface analog: MOST needs a marine-ish moisture field
        # for SFCLAY's flux branches; mp stays off (moist transport only).
        moist=most, mp_physics=0,
        isfflx=2 if most else 0, tke_heat_flux=heat_flux,
        tke_drag_coefficient=0.0 if most else drag_coefficient,
        time_step_sound=4,
        specified=lateral == "specified",
        tke_budget=tke_budget, restart_interval_s=restart_interval_s)


def build(cfg: RunConfig, seed: int = 0):
    """State at rest with seeded random theta perturbations on the lowest
    four levels (README.les: "A random perturbation is imposed initially
    on the mean temperature field at the lowest four grid levels").

    The MOST surface mode (isfflx=2) additionally carries a small marine
    moisture profile and a live SFCLAY over an idealized water surface
    (landmask 0, fixed skin temperature at the sounding surface theta,
    mavail 1 -- the em_les shalconv-variant analog), so USTM and QFX come
    from the real surface-layer scheme while the heat flux stays the
    prescribed constant.

    The ``specified`` lateral topology additionally attaches a frozen
    specified-boundary forcing built from the initial state itself, so the
    domain is a walled CBL rather than a doubly periodic one.  Its purpose
    is to put WRF's TKE lateral arm (``flow_dep_bdy``) on a live
    trajectory: the carrier has no ``b`` Registry flag, so no boundary
    stream forces it and a frozen wall is exactly the forcing WRF would
    see for TKE either way."""
    from gpuwm.core.state import init_theta_perturbation
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, _theta_profile, p_surf=cfg.p_surf,
                           ztop=cfg.ztop)

    def thp_func(x, z):
        rng = np.random.default_rng(seed)
        pert = np.zeros((cfg.nz, cfg.ny, cfg.nx))
        pert[:4] = 0.1 * rng.standard_normal((4, cfg.ny, cfg.nx))
        return pert

    if not cfg.moist:
        return _wall(init_theta_perturbation(cfg, coord, base, thp_func),
                     cfg)

    from gpuwm.core.moist import init_moist_balanced
    from gpuwm.core.physics import initialize_physics
    state = init_moist_balanced(
        cfg, coord, base,
        lambda z: 0.005 * np.exp(-np.asarray(z, float) / 2000.0),
        thp_func=thp_func)
    if cfg.sf_sfclay_physics != 0:
        initialize_physics(state, cfg, landmask=0.0, tsk=300.0,
                           mavail=1.0)
    return _wall(state, cfg)


def _wall(state, cfg: RunConfig):
    """Attach frozen specified-boundary forcing on the walled topology."""
    if not cfg.specified:
        return state
    from gpuwm.ingest.lateral_bc import (attach_lateral_boundaries,
                                         build_state_lateral_boundaries)
    attach_lateral_boundaries(
        state,
        build_state_lateral_boundaries(
            [state, state], [0.0, cfg.run_seconds + cfg.dt],
            spec_bdy_width=cfg.spec_bdy_width, spec_zone=cfg.spec_zone,
            relax_zone=cfg.relax_zone))
    return state


def _spec_zone_is_flow_dependent(field, spec_zone: int) -> bool:
    """WRF ``flow_dep_bdy``'s invariant, checkable from a finished state.

    Every specified-zone cell is either exactly zero (inflow face) or
    exactly its interior donor value (outflow face, zero gradient) --
    share/module_bc.F:2335-2456, with the Y sides owning the four corners.
    Exact equality, not a tolerance: the arm assigns, it does not blend.
    """
    field = np.asarray(field, dtype=np.float64)
    nz, ny, nx = field.shape
    sz = int(spec_zone)
    ok = True
    for d in range(sz):
        cols = np.arange(d, nx - d)
        inner_i = np.clip(cols, sz, nx - 1 - sz)
        for row, donor in ((d, sz), (ny - 1 - d, ny - 1 - sz)):
            value = field[:, row, cols]
            ok &= bool(np.all((value == 0.0)
                              | (value == field[:, donor, inner_i])))
        rows = np.arange(d + 1, ny - d - 1)
        inner_j = np.clip(rows, sz, ny - 1 - sz)
        for col, donor in ((d, sz), (nx - 1 - d, nx - 1 - sz)):
            value = field[:, rows, col]
            ok &= bool(np.all((value == 0.0)
                              | (value == field[:, inner_j, donor])))
    return ok


def _column_mass(state) -> float:
    """Domain-mean total dry column mass (periodic: no lateral flux, so
    this is the exact-conservation observable)."""
    import cupy as cp
    return float(cp.asnumpy(
        (state.mub2d + state.mup).astype(cp.float64).mean()))


def _slab_profiles(state, cfg) -> dict:
    """Horizontal-slab statistics: theta, resolved w'th', resolved TKE,
    and the SGS heat flux -Kh_v * dtheta/dz on w levels."""
    import cupy as cp
    thb = state.thb
    th = (thb if thb.ndim == 3 else thb[:, None, None]) + state.thp
    th = th.astype(cp.float64)
    wm = 0.5 * (state.w[:-1] + state.w[1:]).astype(cp.float64)
    uc = state.u[:, :, :cfg.nx].astype(cp.float64)
    vc = state.v[:, :cfg.ny, :].astype(cp.float64)

    def slab(a):
        return a.mean(axis=(1, 2))

    th_bar = slab(th)
    w_bar = slab(wm)
    thp = th - th_bar[:, None, None]
    wp = wm - w_bar[:, None, None]
    up = uc - slab(uc)[:, None, None]
    vp = vc - slab(vc)[:, None, None]
    wth_res = slab(wp * thp)
    tke_res = 0.5 * (slab(up * up) + slab(vp * vp) + slab(wp * wp))

    # Prognostic SGS TKE (km_opt=2): slab-mean profile and volume mean.
    e_sgs = np.zeros(cfg.nz)
    e_sgs_volume = 0.0
    if getattr(state, "tke", None) is not None:
        e64 = state.tke.astype(cp.float64)
        e_sgs = cp.asnumpy(e64.mean(axis=(1, 2)))
        e_sgs_volume = float(e64.mean())

    # SGS down-gradient flux on interior w levels from the live K field.
    khv = state.existing_scratch("smag_khv")
    wth_sgs = np.zeros(cfg.nz + 1)
    if khv is not None:
        khv = khv.astype(cp.float64)
        phb = state.phb
        phi = ((phb if phb.ndim == 3 else phb[:, None, None])
               + state.php).astype(cp.float64)
        fnm = state.fnm.astype(cp.float64)
        fnp = state.fnp.astype(cp.float64)
        for kw in range(1, cfg.nz):
            rdz = 2.0 * 9.81 / (phi[kw + 1] - phi[kw - 1])
            k_w = fnm[kw] * khv[kw] + fnp[kw] * khv[kw - 1]
            flux = -k_w * (th[kw] - th[kw - 1]) * rdz
            wth_sgs[kw] = float(flux.mean())
    return {
        "theta": cp.asnumpy(th_bar),
        "wth_res": cp.asnumpy(wth_res),
        "wth_sgs": wth_sgs,
        "tke_res": cp.asnumpy(tke_res),
        "e_sgs": e_sgs,
        "e_sgs_volume": e_sgs_volume,
    }


def _zi_from_flux(profile_total, z_mass) -> float:
    """Boundary-layer depth: height of the minimum total heat flux."""
    k = int(np.argmin(profile_total))
    return float(z_mass[min(k, len(z_mass) - 1)])


def _integrate(cfg: RunConfig, seed: int, sample_every_s: float,
               progress=None) -> dict:
    import cupy as cp

    from gpuwm.core.dycore import run_steps, stability_report

    state = build(cfg, seed=seed)
    z = state.height_half()
    z_mass = np.asarray(cp.asnumpy(z) if hasattr(z, "get") else z, float)
    if z_mass.ndim == 3:
        z_mass = z_mass.mean(axis=(1, 2))

    n_total = int(round(cfg.run_seconds / cfg.dt))
    n_chunk = max(int(round(sample_every_s / cfg.dt)), 1)
    mass0 = _column_mass(state)
    samples = []
    cfl_max = 0.0
    w_max = 0.0
    nan = False
    wall0 = time.monotonic()
    done = 0
    while done < n_total:
        n = min(n_chunk, n_total - done)
        run_steps(state, cfg, n)
        done += n
        rep = stability_report(state, cfg)
        nan = nan or bool(rep["nan"])
        if rep["cfl"] is not None:
            cfl_max = max(cfl_max, float(rep["cfl"]))
        w_max = max(w_max, float(rep["w_max"]))
        prof = _slab_profiles(state, cfg)
        prof["t_seconds"] = done * cfg.dt
        prof["w_max"] = float(rep["w_max"])
        prof["cfl"] = rep["cfl"]
        prof["budget"] = tke_budget.drain(state, cfg)
        samples.append(prof)
        if progress is not None:
            progress(done, n_total, rep)
        if nan:
            break
    wall = time.monotonic() - wall0

    last = samples[-1]
    total_flux = last["wth_res"].copy()
    total_flux += 0.5 * (last["wth_sgs"][:-1] + last["wth_sgs"][1:])
    zi = _zi_from_flux(total_flux, z_mass)

    # The entrainment minimum, TIME-AVERAGED.  The single-frame minimum is
    # not a reproducible statistic and must not be published as though it
    # were: measured on one 2 h run, the instantaneous
    # `wth_total_min_over_qs` wanders between -0.0721 and -0.1843 over the
    # final hour -- an 81% spread about its own mean, sd 0.0229 -- because
    # it is the extremum of a small, fluctuating residual between two much
    # larger fluxes.  Two committed receipts for nominally the same
    # configuration differ by 64% on it, and 72% of the individual minutes
    # in a single run fall inside that gap, so the gap is sampling noise
    # rather than irreproducibility.  (The engine itself is bit-reproducible:
    # two independent runs of this case are byte-identical, and tke_budget
    # is byte-inert.)  Averaged over the trailing window the same quantity
    # is stable to ~3%, so that is what a reader should cite.
    tail = samples[-_ENTRAINMENT_WINDOW:] if len(samples) >= 2 else samples
    tail_flux = np.stack([
        s["wth_res"] + 0.5 * (s["wth_sgs"][:-1] + s["wth_sgs"][1:])
        for s in tail])
    wth_total_min_mean = float(
        (tail_flux.min(axis=1) / cfg.tke_heat_flux).mean())
    wth_total_min_sd = float(
        (tail_flux.min(axis=1) / cfg.tke_heat_flux).std())
    mass1 = _column_mass(state)

    # Structure receipt: updraft area fraction at ~0.3 z_i (CBL updrafts
    # occupy the minority of the area -- a plume field, not noise).
    k_low = int(np.argmin(np.abs(z_mass - 0.3 * max(zi, z_mass[1]))))
    wm = 0.5 * (state.w[k_low] + state.w[k_low + 1])
    updraft_fraction = float((cp.asnumpy(wm) > 0.0).mean())

    # Two VRAM figures with opposite screen classifications -- the
    # per-process pool figure is screened, the device-wide figure is
    # environmental and excluded (see ENVIRONMENTAL_FIELDS).
    vram = _vram_fields(
        pool_used_bytes=cp.get_default_memory_pool().used_bytes,
        device_free_total=cp.cuda.runtime.memGetInfo)

    metrics = {
        "nan": nan,
        "w_max": w_max,
        "cfl_max": cfl_max,
        "mass_drift_rel": abs(mass1 - mass0) / abs(mass0),
        # --- receipts (never gated here; see module docstring) ---
        "zi_m": zi,
        "surface_heat_flux": cfg.tke_heat_flux,
        "wth_res_max": float(last["wth_res"].max()),
        "wth_res_max_over_qs": float(
            last["wth_res"].max() / cfg.tke_heat_flux),
        # Single-frame value, retained for continuity with receipts written
        # before the instability above was measured.  Do not cite it.
        "wth_total_min_over_qs": float(
            total_flux.min() / cfg.tke_heat_flux),
        # The citable entrainment ratio: mean over the trailing window,
        # with the spread that makes the single-frame number unusable.
        "wth_total_min_over_qs_mean": wth_total_min_mean,
        "wth_total_min_over_qs_sd": wth_total_min_sd,
        "entrainment_window_samples": len(tail),
        "tke_res_max": float(last["tke_res"].max()),
        "e_sgs_volume_final": float(last["e_sgs_volume"]),
        "e_sgs_max": float(np.asarray(last["e_sgs"]).max()),
        # TKE-based resolved fraction: DEFINED ONLY where a prognostic
        # SGS TKE carrier exists, i.e. km_opt=2.  `e_sgs` is filled from
        # state.tke, which km_opt=3 does not have, so under the 3-D
        # Smagorinsky closure this ratio evaluated to exactly 1.000 --
        # not because the run resolved everything, but because its
        # denominator was absent.  A metric that reads "perfect" when its
        # input is missing is worse than no metric, so it is null there.
        # The closure-independent measure is `wth_res_max_over_qs`: the
        # resolved buoyancy flux as a fraction of the surface flux, whose
        # SGS counterpart comes from the live khv field and is therefore
        # available under BOTH closures.
        "resolved_fraction_ml": (
            float(last["tke_res"][1:max(2, int(0.7 * cfg.nz))].mean()
                  / max(last["tke_res"][1:max(2, int(0.7 * cfg.nz))].mean()
                        + np.asarray(
                            last["e_sgs"])[1:max(2, int(0.7 * cfg.nz))].mean(),
                        1e-30))
            if cfg.km_opt == 2 else None),
        "resolved_fraction_basis": (
            "prognostic SGS TKE (km_opt=2)" if cfg.km_opt == 2
            else "unavailable: no prognostic SGS TKE carrier under "
                 f"km_opt={cfg.km_opt}; use wth_res_max_over_qs"),
        "updraft_fraction_low": updraft_fraction,
        # Walled topology: the mass observable above is NOT a conservation
        # statement (lateral flux is physical here), and the TKE carrier's
        # lateral arm becomes checkable from the finished state.  The
        # topology itself is recorded in the receipt's ``config`` block.
        "tke_spec_zone_flow_dependent": (
            None if not (cfg.specified and cfg.km_opt == 2)
            else _spec_zone_is_flow_dependent(
                cp.asnumpy(state.tke), cfg.spec_zone)),
        "wall_seconds": wall,
        "steps": done,
        "steps_per_second": done / wall if wall > 0 else None,
        **vram,
    }
    budgets = [s["budget"] for s in samples if s.get("budget") is not None]
    if budgets:
        # AC-L5.1 receipt: the LAST window's volume-integrated budget (the
        # statistically developed one), committed as a measurement.  The
        # closure residual is reported, never scored against a band cut
        # from itself -- the sensitivity control lives in the R0 battery.
        final_budget = budgets[-1]
        metrics["tke_budget_terms"] = dict(final_budget["volume"])
        metrics["tke_budget_residual"] = final_budget["residual"]
        metrics["tke_budget_residual_rel"] = final_budget["residual_rel"]
        metrics["tke_budget_window_steps"] = final_budget["steps"]
    return {"metrics": metrics, "samples": samples, "z_mass": z_mass,
            "state": state, "budgets": budgets}


def run(outdir: Path | None = None) -> dict:
    """VERIFY smoke: small grid, 15 simulated minutes, validity gates."""
    cfg = make_config(nx=48, ny=48, nz=40, dx=100.0, ztop=2000.0,
                      dt=0.5, minutes=15.0)
    result = _integrate(cfg, seed=0, sample_every_s=60.0)
    metrics = dict(result["metrics"])
    metrics["nan"] = bool(metrics["nan"])
    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        _write_outputs(result, cfg, outdir, tag="smoke")
    metrics.pop("state", None)
    return metrics


def _write_outputs(result, cfg, outdir: Path, tag: str) -> None:
    import cupy as cp
    samples = result["samples"]
    arrays = {
        "z_mass": result["z_mass"],
        "t_seconds": np.array([s["t_seconds"] for s in samples]),
        "theta": np.stack([s["theta"] for s in samples]),
        "wth_res": np.stack([s["wth_res"] for s in samples]),
        "wth_sgs": np.stack([s["wth_sgs"] for s in samples]),
        "tke_res": np.stack([s["tke_res"] for s in samples]),
        "e_sgs": np.stack([s["e_sgs"] for s in samples]),
        "e_sgs_volume": np.array([s["e_sgs_volume"] for s in samples]),
        "w_max": np.array([s["w_max"] for s in samples]),
    }
    state = result["state"]
    zi = result["metrics"]["zi_m"]
    z_mass = result["z_mass"]
    k_low = int(np.argmin(np.abs(z_mass - 0.3 * max(zi, z_mass[1]))))
    arrays["w_slab_final"] = cp.asnumpy(
        0.5 * (state.w[k_low] + state.w[k_low + 1])).astype(np.float32)
    arrays["w_slab_height_m"] = np.array([z_mass[k_low]])
    # Term-by-term TKE budget series: one (nsample, nz) plane per term plus
    # the closure residual, in coupled units (kernel currency), so a
    # consumer can rebuild the identity without re-running anything.
    budgets = result.get("budgets") or []
    if budgets:
        arrays["budget_t_seconds"] = np.array(
            [s["t_seconds"] for s in samples if s.get("budget") is not None])
        arrays["budget_window_steps"] = np.array(
            [b["steps"] for b in budgets])
        for name in tke_budget.TERMS:
            arrays["budget_" + name] = np.stack(
                [b["profiles"][name] for b in budgets])
        arrays["budget_residual"] = np.stack(
            [b["residual_profile"] for b in budgets])
    np.savez_compressed(outdir / f"cbl_{tag}_profiles.npz", **arrays)
    receipt = {k: v for k, v in result["metrics"].items()}
    receipt["config"] = {
        "nx": cfg.nx, "ny": cfg.ny, "nz": cfg.nz, "dx": cfg.dx,
        "ztop": cfg.ztop, "dt": cfg.dt, "run_seconds": cfg.run_seconds,
        "km_opt": cfg.km_opt, "mix_isotropic": cfg.mix_isotropic,
        "sf_sfclay_physics": cfg.sf_sfclay_physics, "moist": cfg.moist,
        "c_s": cfg.c_s, "c_k": cfg.c_k, "isfflx": cfg.isfflx,
        "tke_heat_flux": cfg.tke_heat_flux,
        "tke_drag_coefficient": cfg.tke_drag_coefficient,
        "tke_upper_bound": cfg.tke_upper_bound,
        "tke_budget": cfg.tke_budget,
        "bl_pbl_physics": cfg.bl_pbl_physics,
        "periodic": not (cfg.specified or cfg.open_x or cfg.open_y
                         or cfg.nested),
    }
    (outdir / f"cbl_{tag}_receipt.json").write_text(
        json.dumps(receipt, indent=1, default=float) + "\n",
        encoding="utf-8")


def _restart_check(cfg: RunConfig, *, seed: int, split_minutes: float,
                   outdir: Path, tag: str) -> int:
    """Run the engine's restart standard against a live LES trajectory.

    Straight-through vs stop/checkpoint/restore-into-a-fresh-state/finish,
    compared member by member on the restart archives themselves -- every
    serialized state array, every serialized scratch accumulator, and the
    clock.  Byte equality is the standard; anything else is a failure that
    names the fields that moved, never a tolerance.
    """
    import hashlib

    import cupy as cp

    from gpuwm.core.dycore import run_steps
    from gpuwm.io import restart as restart_io

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    n_total = int(round(cfg.run_seconds / cfg.dt))
    n_split = int(round(split_minutes * 60.0 / cfg.dt))
    if not 0 < n_split < n_total:
        raise ValueError(
            f"--restart-check {split_minutes} min is not inside the "
            f"{cfg.run_seconds / 60.0} min run")

    wall0 = time.monotonic()
    straight = build(cfg, seed=seed)
    run_steps(straight, cfg, n_total)
    reference = restart_io.write_restart(
        outdir / f"cbl_{tag}_reference.npz", straight, cfg)

    split = build(cfg, seed=seed)
    run_steps(split, cfg, n_split)
    checkpoint = restart_io.write_restart(
        outdir / f"cbl_{tag}_checkpoint.npz", split, cfg)

    # A FRESH state, exactly as a resumed process would build it: the
    # carrier starts at its cold-start zero and must be overwritten by the
    # archive, not by the trajectory.
    resumed_state = build(cfg, seed=seed)
    cold_tke = float(cp.abs(resumed_state.tke).max()) \
        if getattr(resumed_state, "tke", None) is not None else None
    info = restart_io.restore_restart(checkpoint, resumed_state, cfg)
    restored_tke = float(cp.abs(resumed_state.tke).max()) \
        if getattr(resumed_state, "tke", None) is not None else None
    run_steps(resumed_state, cfg, n_total - n_split)
    resumed = restart_io.write_restart(
        outdir / f"cbl_{tag}_resumed.npz", resumed_state, cfg)
    wall = time.monotonic() - wall0

    def _members(path):
        with np.load(path, allow_pickle=False) as data:
            return {key: bytes(np.ascontiguousarray(data[key]).tobytes())
                    for key in data.files
                    if key != "__gpuwm_restart_header__"}

    a, b = _members(reference), _members(resumed)
    only_reference = sorted(set(a) - set(b))
    only_resumed = sorted(set(b) - set(a))
    differing = sorted(key for key in set(a) & set(b) if a[key] != b[key])
    identical = bool(not only_reference and not only_resumed
                     and not differing)

    def _content_sha256(members: dict[str, bytes]) -> str:
        """Hash the STATE, not the container.

        An .npz is a zip: its bytes carry per-entry metadata, so two
        archives holding identical arrays hash differently and always
        will.  Hashing the file therefore proves nothing about the
        trajectory, and printing those two never-equal digests beside
        ``bit_identical: true`` invited the reader to conclude the exact
        opposite of the measurement.  This digests the member names and
        their raw array bytes in sorted order, so it corroborates the
        member-by-member verdict instead of contradicting it.
        """
        digest = hashlib.sha256()
        for key in sorted(members):
            digest.update(key.encode("utf-8"))
            digest.update(b"\0")
            digest.update(members[key])
            digest.update(b"\0")
        return digest.hexdigest()

    receipt = {
        "mode": "restart-identity",
        "steps_total": n_total,
        "steps_before_checkpoint": n_split,
        "checkpoint_elapsed_seconds": info.elapsed_seconds,
        "members_compared": len(set(a) & set(b)),
        "members_only_in_reference": only_reference,
        "members_only_in_resumed": only_resumed,
        "members_differing": differing,
        "bit_identical": identical,
        # The comparison must be non-trivial: the carrier had to be
        # non-zero at the checkpoint and the fresh state had to be cold.
        "tke_max_cold_start": cold_tke,
        "tke_max_after_restore": restored_tke,
        "carrier_was_restored": (
            None if restored_tke is None
            else bool(restored_tke > 0.0 and cold_tke == 0.0)),
        # State-content digests (see _content_sha256): these are EQUAL
        # exactly when bit_identical is true.
        "reference_state_sha256": _content_sha256(a),
        "resumed_state_sha256": _content_sha256(b),
        "state_sha256_match": _content_sha256(a) == _content_sha256(b),
        "wall_seconds": wall,
        "config": {
            "nx": cfg.nx, "ny": cfg.ny, "nz": cfg.nz, "dx": cfg.dx,
            "dt": cfg.dt, "run_seconds": cfg.run_seconds,
            "km_opt": cfg.km_opt, "c_k": cfg.c_k,
            "tke_upper_bound": cfg.tke_upper_bound,
            "isfflx": cfg.isfflx, "tke_heat_flux": cfg.tke_heat_flux,
        },
    }
    (outdir / f"cbl_{tag}_restart_receipt.json").write_text(
        json.dumps(receipt, indent=1, default=float) + "\n",
        encoding="utf-8")
    print(json.dumps(receipt, indent=1, default=float))
    if not identical:
        print(f"RESTART IDENTITY FAIL: {len(differing)} members differ")
        return 1
    if cold_tke is not None and not receipt["carrier_was_restored"]:
        print("RESTART IDENTITY INCONCLUSIVE: the TKE carrier was zero at "
              "the checkpoint, so this run does not exercise it")
        return 1
    print("RESTART IDENTITY PASS")
    return 0


def main(argv=None) -> int:
    """Sized LES driver (see module docstring)."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Dry convective boundary layer, km_opt=3 LES")
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--ny", type=int, default=96)
    parser.add_argument("--nz", type=int, default=64)
    parser.add_argument("--dx", type=float, default=100.0)
    parser.add_argument("--ztop", type=float, default=2400.0)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--minutes", type=float, default=120.0)
    parser.add_argument("--heat-flux", type=float, default=HEAT_FLUX)
    parser.add_argument("--drag", type=float, default=DRAG_COEFFICIENT)
    parser.add_argument("--cs", type=float, default=C_S)
    parser.add_argument("--km-opt", type=int, default=3, choices=(2, 3),
                        help="LES closure: 3 = 3-D Smagorinsky, 2 = "
                             "1.5-order prognostic TKE")
    parser.add_argument("--surface", default="prescribed",
                        choices=("prescribed", "most"),
                        help="lower boundary: prescribed = isfflx=0 "
                             "constant drag/heat; most = isfflx=2 SFCLAY "
                             "USTM/QFX + constant heat (em_les hybrid)")
    parser.add_argument("--ck", type=float, default=0.10,
                        help="km_opt=2 c_k (em_les reference 0.10)")
    parser.add_argument("--lateral", default="periodic",
                        choices=("periodic", "specified"),
                        help="lateral topology: periodic = the em_les "
                             "double-periodic domain; specified = a walled "
                             "domain with frozen boundary forcing, which "
                             "puts WRF's TKE lateral arm (flow_dep_bdy) on "
                             "a live trajectory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-every", type=float, default=60.0,
                        help="profile sampling interval, seconds")
    parser.add_argument("--tke-budget", action="store_true",
                        help="accumulate the term-by-term km_opt=2 TKE "
                             "budget on device and file it in the receipt")
    parser.add_argument("--restart-check", type=float, default=None,
                        metavar="MINUTES",
                        help="instead of one integration, prove restart "
                             "bit-identity: run the full length straight "
                             "through, then run it again stopping at "
                             "MINUTES, checkpointing, restoring into a "
                             "fresh process state and finishing -- then "
                             "byte-compare the two end states")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tag", default="les")
    args = parser.parse_args(argv)

    cfg = make_config(nx=args.nx, ny=args.ny, nz=args.nz, dx=args.dx,
                      ztop=args.ztop, dt=args.dt, minutes=args.minutes,
                      heat_flux=args.heat_flux, drag_coefficient=args.drag,
                      c_s=args.cs, km_opt=args.km_opt, c_k=args.ck,
                      surface=args.surface, lateral=args.lateral,
                      tke_budget=1 if args.tke_budget else 0)

    if args.restart_check is not None:
        return _restart_check(cfg, seed=args.seed,
                              split_minutes=args.restart_check,
                              outdir=args.out, tag=args.tag)

    def progress(done, total, rep):
        if done % max(total // 40, 1) < max(
                int(round(args.sample_every / cfg.dt)), 1):
            print(f"  t={done * cfg.dt:8.1f}s  w_max={rep['w_max']:6.2f}"
                  f"  cfl={rep['cfl'] if rep['cfl'] is None else round(rep['cfl'], 3)}",
                  flush=True)

    result = _integrate(cfg, seed=args.seed,
                        sample_every_s=args.sample_every,
                        progress=progress)
    args.out.mkdir(parents=True, exist_ok=True)
    _write_outputs(result, cfg, args.out, tag=args.tag)
    m = result["metrics"]
    print(json.dumps({k: v for k, v in m.items()}, indent=1, default=float))
    failed = bool(m["nan"])
    for key, (lo, hi) in GATES.items():
        value = m.get(key)
        if value is None:
            continue
        if key == "mass_drift_rel" and cfg.specified:
            # Dry-mass closure is the PERIODIC domain's exact-conservation
            # observable.  A walled domain exchanges mass through its
            # specified zone by design, so the number is filed as a
            # receipt and not gated -- gating it here would be scoring a
            # boundary condition as if it were a bug.
            print(f"  (mass_drift_rel = {value} filed, not gated: the "
                  f"specified topology exchanges lateral mass)")
            continue
        if (lo is not None and value < lo) or \
                (hi is not None and value > hi):
            print(f"GATE FAIL: {key} = {value} outside ({lo}, {hi})")
            failed = True
    if m.get("tke_spec_zone_flow_dependent") is False:
        # WRF's whole TKE lateral treatment, checked on the finished state.
        print("GATE FAIL: the TKE specified zone is not flow-dependent "
              "(WRF flow_dep_bdy, share/module_bc.F:2335-2456)")
        failed = True
    print("VALIDITY", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
