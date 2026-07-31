"""Weisman-Klemp (1982) quarter-circle supercell benchmark (Phase 2).

Task 10 provides the analytic WK82 sounding below; Task 11 completes the
module with the full 3-D quarter-circle-hodograph supercell case.

Sounding (Weisman & Klemp 1982 MWR, eqs. (1)-(2); the plan's Task 10
formulas transcribed verbatim):

    theta(z) = theta_0 + (theta_tr - theta_0) * (z/z_tr)^(5/4)   z <= z_tr
             = theta_tr * exp(g * (z - z_tr) / (cp * T_tr))      z >  z_tr
    RH(z)    = 1 - 0.75 * (z/z_tr)^(5/4)                         z <= z_tr
             = 0.25                                              z >  z_tr
    qv(z)    = min(RH(z) * qvs(p, T), 14 g/kg)

with theta_0 = 300 K, theta_tr = 343 K, z_tr = 12 km, T_tr = 213 K.

Provenance note: the local WRF v4.6.1 quarter_ss initializer
(``dyn_em/module_initialize_ideal.F``, compile-time aliased to
module_initialize_quarter_ss.F) has NO analytic sounding branch -- its
``get_sounding`` reads the tabulated ``test/em_quarter_ss/input_sounding``,
which is itself the WK82 analytic sounding evaluated by NCAR offline.  The
formulas above are therefore transcribed from WK82 directly (as the plan
specifies), and ``wk82_sounding`` is verified against that shipped
tabulation in tests/test_moist.py: theta below the tropopause matches to
6e-5 K (the formula is exact -- e.g. the file's theta(10 m) = 300.0061 is
300 + 43*(10/12000)^1.25), the 14 g/kg cap releases in the same 1.2-1.3 km
layer, and qv agrees within 3.5% everywhere (the residual is the file
generator's unknown saturation formula; it is not in the WRF tree).

RH -> qv closure: the saturation mixing ratio uses the model's OWN Teten
formula -- the constants of ``phys/module_mp_kessler.F`` already in
``gpuwm.core.constants`` (SVP1/SVP2/SVP3/SVPT0, EP2 = Rd/Rv) -- so that
"relative humidity" means the same thing to the initial state and to the
Kessler scheme that later acts on it.  Since qvs depends on the pressure
and the pressure (hydrostatically) on qv, the profile is closed by a
fixed-point iteration in the style of the local ``get_sounding``'s own
hydrostatic loops (a fixed 10-iteration sweep, ``do it=1,10``): integrate
the Exner function with the exact virtual potential temperature
``theta_v = theta * (1 + (Rv/Rd) qv) / (1 + qv)`` (the ratio comes from
the constants -- never a hardcoded 1.61/0.61) on a 10 m fine grid that
includes every requested height, then refresh qv = min(RH*qvs, 14 g/kg).
Float64 throughout (setup-time code).
"""

from __future__ import annotations

import numpy as np

from gpuwm.core import constants as c

#: WK82 sounding parameters (Weisman & Klemp 1982, eqs. (1)-(2)).
THETA_0 = 300.0     # surface potential temperature (K)
THETA_TR = 343.0    # tropopause potential temperature (K)
Z_TR = 12000.0      # tropopause height (m)
T_TR = 213.0        # tropopause temperature (K)
QV0 = 0.014         # boundary-layer mixing-ratio cap (kg/kg): 14 g/kg

#: Fine-grid spacing (m) and fixed-point sweep count of the hydrostatic
#: qv closure (the sweep count mirrors get_sounding's ``do it=1,10``).
_DZ_FINE = 10.0
_N_ITER = 10


def wk82_theta(z) -> np.ndarray:
    """WK82 eq. (1): potential temperature (K) at height ``z`` (m)."""
    z = np.asarray(z, dtype=np.float64)
    return np.where(z <= Z_TR,
                    THETA_0 + (THETA_TR - THETA_0) * (z / Z_TR) ** 1.25,
                    THETA_TR * np.exp(c.G * (z - Z_TR) / (c.CP * T_TR)))


def wk82_relhum(z) -> np.ndarray:
    """WK82 eq. (2): relative humidity (0-1) at height ``z`` (m)."""
    z = np.asarray(z, dtype=np.float64)
    return np.where(z <= Z_TR, 1.0 - 0.75 * (z / Z_TR) ** 1.25, 0.25)


def saturation_mixing_ratio(p, T) -> np.ndarray:
    """Saturation mixing ratio (kg/kg) over water at pressure ``p`` (Pa)
    and temperature ``T`` (K), exactly as the local Kessler scheme
    (phys/module_mp_kessler.F) diagnoses it: Teten
    ``es = 1000*svp1*exp(svp2*(T - svpt0)/(T - svp3))`` (Pa),
    ``qvs = ep2*es/(p - es)``."""
    es = 1000.0 * c.SVP1 * np.exp(c.SVP2 * (np.asarray(T, np.float64)
                                            - c.SVPT0)
                                  / (np.asarray(T, np.float64) - c.SVP3))
    return c.EP2 * es / (np.asarray(p, np.float64) - es)


def wk82_sounding(z, p_surf: float = 1.0e5, full: bool = False):
    """WK82 analytic sounding at heights ``z`` (m): ``(theta, qv)``.

    ``theta`` (K) and ``qv`` (kg/kg) are float64 arrays shaped like ``z``.
    ``full=True`` appends the closure's hydrostatic ``(p, T)`` (Pa, K) at
    the same heights -- the pressure the RH -> qv conversion used (the
    module docstring describes the fixed-point construction).  ``p_surf``
    defaults to the WK82/em_quarter_ss 1000 hPa surface.
    """
    z_in = np.asarray(z, dtype=np.float64)
    zq = np.atleast_1d(z_in).ravel()
    if zq.size == 0 or (zq < 0.0).any():
        raise ValueError("wk82_sounding needs nonempty heights >= 0, got "
                         f"range [{zq.min() if zq.size else np.nan}, "
                         f"{zq.max() if zq.size else np.nan}]")

    # Integration grid: 10 m spacing from the surface past the highest
    # requested level, with every requested height inserted exactly so no
    # output value is interpolated.
    zmax = float(zq.max())
    zg = np.union1d(np.arange(0.0, zmax + _DZ_FINE, _DZ_FINE), zq)
    th = wk82_theta(zg)
    rh = wk82_relhum(zg)
    dz = np.diff(zg)

    qv = np.zeros_like(zg)
    pi = np.empty_like(zg)
    for _ in range(_N_ITER):
        thv = th * (1.0 + c.RVOVRD * qv) / (1.0 + qv)
        thv_mid = 0.5 * (thv[:-1] + thv[1:])
        pi[0] = (p_surf / c.P0) ** c.RCP
        pi[1:] = pi[0] - np.cumsum(c.G * dz / (c.CP * thv_mid))
        p = c.P0 * pi ** (1.0 / c.RCP)
        T = th * pi
        qv = np.minimum(rh * saturation_mixing_ratio(p, T), QV0)

    idx = np.searchsorted(zg, zq)                  # exact: zq is in zg
    pick = lambda a: a[idx].reshape(z_in.shape)
    if full:
        return pick(th), pick(qv), pick(p), pick(T)
    return pick(th), pick(qv)


# ---------------------------------------------------------------------------
# Task 11: the full 3-D quarter-circle supercell benchmark.
# ---------------------------------------------------------------------------

#: WK82 quarter_ss thermal (module_initialize_ideal.F ideal_pert CASE
#: (quarter_ss), lines 1088-1135): amplitude (K), horizontal radius (m),
#: center height / half-depth (m).  The Fortran uses 1500 m (the plan
#: text's "1.4 km" reflects the WK82 paper; the plan defers the init to
#: the bundle's initializer, which is transcribed here verbatim).
BUBBLE_DELT = 3.0
BUBBLE_RADIUS = 10000.0
BUBBLE_ZC = 1500.0

#: Quarter-circle hodograph: the (z [m], u, v [m/s]) wind columns of the
#: local WRF v4.6.1 ``test/em_quarter_ss/input_sounding``, verbatim, up to
#: 7 km; every shipped row above 7 km holds (27.15, 3.5) exactly, so the
#: 30 km sentinel row reproduces the file's constant-above-7-km profile
#: under linear interpolation.  The constant storm-motion frame translation
#: that keeps the right mover centered is already baked into these columns
#: (surface wind (-8.10, -4.94), not zero) -- used as-is, nothing further
#: subtracted (plan Task 11; no umove/vmove defaults exist in this tree).
HODOGRAPH_ZUV = np.array([
    (10.0, -8.0998, -4.9419), (35.0, -8.0976, -4.7968),
    (100.0, -8.0802, -4.4199), (200.0, -8.0208, -3.8426),
    (300.0, -7.9222, -3.2706), (400.0, -7.7848, -2.7067),
    (500.0, -7.6092, -2.1535), (600.0, -7.3963, -1.6136),
    (700.0, -7.1470, -1.0894), (800.0, -6.8626, -0.5835),
    (900.0, -6.5442, -0.0982), (1000.0, -6.1935, 0.3642),
    (1100.0, -5.8121, 0.8017), (1200.0, -5.4017, 1.2121),
    (1300.0, -4.9642, 1.5935), (1400.0, -4.5018, 1.9442),
    (1500.0, -4.0165, 2.2626), (1600.0, -3.5106, 2.5470),
    (1700.0, -2.9864, 2.7963), (1800.0, -2.4465, 3.0092),
    (1900.0, -1.8933, 3.1848), (2000.0, -1.3294, 3.3222),
    (2100.0, -0.7574, 3.4208), (2200.0, -0.1801, 3.4802),
    (2300.0, 0.4000, 3.5000), (2400.0, 0.9691, 3.5000),
    (2500.0, 1.5383, 3.5000), (2600.0, 2.1074, 3.5000),
    (2700.0, 2.6766, 3.5000), (2800.0, 3.2457, 3.5000),
    (2900.0, 3.8149, 3.5000), (3000.0, 4.3840, 3.5000),
    (3100.0, 4.9532, 3.5000), (3200.0, 5.5223, 3.5000),
    (3300.0, 6.0915, 3.5000), (3400.0, 6.6606, 3.5000),
    (3500.0, 7.2298, 3.5000), (3600.0, 7.7989, 3.5000),
    (3700.0, 8.3681, 3.5000), (3800.0, 8.9372, 3.5000),
    (3900.0, 9.5064, 3.5000), (4000.0, 10.0755, 3.5000),
    (4100.0, 10.6447, 3.5000), (4200.0, 11.2138, 3.5000),
    (4300.0, 11.7830, 3.5000), (4400.0, 12.3521, 3.5000),
    (4500.0, 12.9213, 3.5000), (4600.0, 13.4904, 3.5000),
    (4700.0, 14.0596, 3.5000), (4800.0, 14.6287, 3.5000),
    (4900.0, 15.1979, 3.5000), (5000.0, 15.7670, 3.5000),
    (5100.0, 16.3362, 3.5000), (5200.0, 16.9053, 3.5000),
    (5300.0, 17.4745, 3.5000), (5400.0, 18.0436, 3.5000),
    (5500.0, 18.6128, 3.5000), (5600.0, 19.1819, 3.5000),
    (5700.0, 19.7511, 3.5000), (5800.0, 20.3202, 3.5000),
    (5900.0, 20.8894, 3.5000), (6000.0, 21.4585, 3.5000),
    (6100.0, 22.0277, 3.5000), (6200.0, 22.5968, 3.5000),
    (6300.0, 23.1660, 3.5000), (6400.0, 23.7351, 3.5000),
    (6500.0, 24.3043, 3.5000), (6600.0, 24.8734, 3.5000),
    (6700.0, 25.4426, 3.5000), (6800.0, 26.0117, 3.5000),
    (6900.0, 26.5809, 3.5000), (7000.0, 27.1500, 3.5000),
    (30000.0, 27.1500, 3.5000),
])

#: Analysis times (s) for the split geometry and plan-view PNGs.
ANALYSIS_TIMES = (3600.0, 5400.0, 7200.0)

#: Updraft-core threshold (m/s) and analysis height (m) for the split
#: metrics (plan: "w > 15 m/s clusters at z ~ 5 km").
CORE_W = 15.0
CORE_Z = 5000.0

#: UH-proxy integration bounds (m): plan "int w*zeta dz, 2-5 km".
UH_ZBOT, UH_ZTOP = 2000.0, 5000.0

#: Dry-mass drift sampling cadence (steps) for the ``mass_drift_max``
#: metric (Phase 2 final review T11-1: the original 100-step cadence could
#: miss short-lived boundary-exchange spikes between samples; the FP64
#: device-sum reduction is cheap next to a model step, so every 10 steps
#: costs nothing measurable).
MASS_SAMPLE_EVERY = 10


def mass_sample_steps(n_total: int) -> range:
    """1-based step indices at which :func:`run` samples the dry-mass
    drift for ``mass_drift_max``: every :data:`MASS_SAMPLE_EVERY` steps.
    (``run()`` additionally folds the final post-run drift into the
    metric, so an ``n_total`` off the cadence still gates its last step.)
    """
    return range(MASS_SAMPLE_EVERY, n_total + 1, MASS_SAMPLE_EVERY)

#: Pass gates, metric -> (lo, hi) with strict bounds, None = unbounded on
#: that side.  Single-sourced (Task 12): gpuwm.cli._GATES and
#: tests/test_case_wk82.py both consume this export.  Published WK82/ARW
#: behavior (plan Task 11 acceptance; mass gates adjudicated 2026-07-14
#: for the open lateral boundaries -- see tests/test_case_wk82.py): the
#: storm splits by t = 3600 s into two persistent w > 15 m/s cores at
#: z ~ 5 km (n_cores_T >= 2, i.e. strictly > 1) whose meridional
#: separation grows through the analysis hour; the right mover's motion
#: projects onto the right-hand normal of the 0-6 km mean-shear vector;
#: >95% of post-spin-up samples exceed 25 m/s while the peak stays below
#: 70 m/s; qr_max > 5 g/kg; dry-mass exchange is bounded and its independent
#: boundary-flux budget closes to 1e-5 relative.
#:
#: Separation is gated on the early interval plus TOTAL 3600->7200 growth,
#: not per-interval monotonicity: under h_diabatic the storm field grows
#: secondary flanking-line cells (5 cores by 7200 s) and the two-strongest
#: mover picker can swap cells between snapshots while the same-cell-tracked
#: movers keep separating (adjudicated GATE_ARTIFACT, 2026-07-16 --
#: .superpowers/sdd/codex/wk82-hdiab-adjudication.md: tracked separation
#: 41445 -> 70329 m over the interval whose picker metric measured -8339 m).
GATES = {
    "n_cores_3600": (1, None),
    "n_cores_5400": (1, None),
    "n_cores_7200": (1, None),
    "sep_3600": (0.0, None),
    "sep_growth_3600_5400": (0.0, None),
    "sep_growth": (0.0, None),
    "right_of_shear_m": (0.0, None),
    "w_max": (25.0, 70.0),
    "w_sustained_fraction": (0.95, None),
    "qr_max": (5.0e-3, None),
    "mass_drift_max": (None, 2.0e-3),
    "mass_drift": (None, 3.0e-4),
    "mass_closure_residual_max": (None, 1.0e-5),
}


def hodograph_uv(coord, ztop: float):
    """u/v at the model half levels, consumed exactly as the local WRF
    initializer consumes ``get_sounding``'s wind columns
    (module_initialize_ideal.F:1536-1551, flat terrain): linear
    interpolation against the FULL (moist) sounding pressure at
    ``p_level = c3h*(p_surf - p_top) + c4h + p_top``, with ``p_surf``/
    ``p_top`` the sounding pressures at z = 0 and z = ztop (WRF
    ``grid%p_top = interp_0(p_in, zk, ztop)``).  Below the file's lowest
    wind row (10 m) the interpolation clamps, as WRF's ``interp_0`` does.
    Returns float64 ``(u, v)``, each ``(nz,)``.
    """
    zs = HODOGRAPH_ZUV[:, 0]
    _, _, p_all, _ = wk82_sounding(np.concatenate([zs, [0.0, ztop]]),
                                   full=True)
    p_snd, p_surf, p_top = p_all[:-2], p_all[-2], p_all[-1]
    p_half = coord.c3h * (p_surf - p_top) + coord.c4h + p_top
    # np.interp needs ascending x; pressure decreases with height.
    u = np.interp(p_half, p_snd[::-1], HODOGRAPH_ZUV[::-1, 1])
    v = np.interp(p_half, p_snd[::-1], HODOGRAPH_ZUV[::-1, 2])
    return u, v


def default_config():
    """Plan Task 11 setup: 168 x 168 x 20 km at dx = dy = 1 km, nz = 60
    (~333 m), dt = 6 s, open x/y lateral boundaries, implicit-w Rayleigh
    top over the top 5 km, the production dissipation package (km_opt = 4
    Smagorinsky + monotonic 6th-order filter), Kessler warm rain, f = 0
    (the model carries no Coriolis); WRF's external-mode divergence
    damping at the em_quarter_ss namelist value (emdiv = 0.01) -- the
    stock stabilizer of the open-boundary external mode.  wrfout frames
    every 1800 s (t = 0/1800/3600/5400/7200: the analysis times; the
    ~60 MB 3-D frames make the WRF-default 5-min output pointlessly heavy
    for a benchmark)."""
    from gpuwm.config import RunConfig

    return RunConfig(nx=168, ny=168, nz=60, dx=1000.0, dy=1000.0,
                     ztop=20000.0, dt=6.0, run_seconds=7200.0,
                     moist=True, mp_physics=1, moist_adv_opt=1,
                     km_opt=4, c_s=0.25, diff_6th_opt=2,
                     diff_6th_factor=0.12,
                     damp_opt=3, zdamp=5000.0, dampcoef=0.2,
                     open_x=True, open_y=True, emdiv=0.01,
                     output_interval_s=1800.0, case="wk82")


def build(cfg, coord, base):
    """Balanced moist WK82 state + quarter-circle hodograph + 3 K thermal.

    Transcribed from the local quarter_ss initializer: the WK82 qv profile
    rides the dry base state through ``init_moist_balanced`` (the WRF
    moist rebalance), the thermal is ``delt*cos(pi*RAD/2)^2`` inside the
    unit ellipsoid RAD centered at (domain center, 1.5 km) with 10 km
    horizontal and 1.5 km vertical radii (the geopotential is re-integrated
    through the bubble's alpha by the same builder), and the winds are the
    shipped hodograph columns broadcast to the staggered u/v fields (flat
    terrain: one column everywhere, as in the Fortran's z_at_u == 0 path).
    """
    import cupy as cp

    from gpuwm.core.moist import init_moist_balanced
    from gpuwm.core.state import DTYPE

    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    y = (np.arange(ny) + 0.5) * cfg.dy - 0.5 * ny * cfg.dy

    def thp_func(x, z):
        xrad = x[None, None, :] / BUBBLE_RADIUS
        yrad = y[None, :, None] / BUBBLE_RADIUS
        zrad = (z[:, None, None] - BUBBLE_ZC) / BUBBLE_ZC
        rad = np.sqrt(xrad ** 2 + yrad ** 2 + zrad ** 2)
        thp = np.where(rad <= 1.0,
                       BUBBLE_DELT * np.cos(0.5 * np.pi * rad) ** 2, 0.0)
        return np.broadcast_to(thp, (nz, ny, nx))

    s = init_moist_balanced(cfg, coord, base,
                            lambda z: wk82_sounding(z)[1], thp_func)
    u, v = hodograph_uv(coord, cfg.ztop)
    s.u[...] = cp.asarray(u, dtype=DTYPE)[:, None, None]
    s.v[...] = cp.asarray(v, dtype=DTYPE)[:, None, None]
    return s


def _clusters(mask, min_cells=3):
    """4-connected clusters of True cells of a 2-D mask: list of index
    arrays ``(rows, cols)``, ignoring clusters below ``min_cells``."""
    seen = np.zeros(mask.shape, dtype=bool)
    out = []
    for j0, i0 in zip(*np.nonzero(mask)):
        if seen[j0, i0]:
            continue
        stack = [(j0, i0)]
        seen[j0, i0] = True
        cells = []
        while stack:
            j, i = stack.pop()
            cells.append((j, i))
            for jj, ii in ((j + 1, i), (j - 1, i), (j, i + 1), (j, i - 1)):
                if (0 <= jj < mask.shape[0] and 0 <= ii < mask.shape[1]
                        and mask[jj, ii] and not seen[jj, ii]):
                    seen[jj, ii] = True
                    stack.append((jj, ii))
        if len(cells) >= min_cells:
            out.append(np.asarray(cells).T)
    return out


def sustained_updraft_fraction(history, threshold=25.0, start=3600.0,
                               end=7200.0) -> float:
    """Fraction of finite ``w_max`` samples above threshold in a time span."""
    history = np.asarray(history, dtype=np.float64)
    selected = history[(history[:, 0] >= start) & (history[:, 0] <= end), 1]
    if selected.size == 0 or not np.isfinite(selected).all():
        return float("nan")
    return float(np.mean(selected > threshold))


def separation_growth_intervals(separation) -> tuple[float, float]:
    """Meridional separation changes across 3600-5400 and 5400-7200 s."""
    return (float(separation[5400.0] - separation[3600.0]),
            float(separation[7200.0] - separation[5400.0]))


def right_of_shear_distance(start_xy, end_xy, shear_uv) -> float:
    """Signed displacement (m) along the right normal of ``shear_uv``."""
    du, dv = np.asarray(shear_uv, dtype=np.float64)
    magnitude = float(np.hypot(du, dv))
    if not np.isfinite(magnitude) or magnitude == 0.0:
        return float("nan")
    dx, dy = np.asarray(end_xy, dtype=np.float64) - np.asarray(
        start_xy, dtype=np.float64)
    return float((dx * dv - dy * du) / magnitude)


def _movers(w2d, dx, dy, shear_uv, threshold=CORE_W):
    """Split diagnostics from a plan-view w slice (ny, nx).

    Returns the number of cores plus x/y centroids of the right and left
    members among the two strongest.  Right/left are classified by each
    centroid's projection on the right normal of ``shear_uv`` rather than
    by a hard-coded south/north proxy.
    """
    clusters = _clusters(w2d > threshold)
    if not clusters:
        return 0, np.nan, np.nan, np.nan, np.nan
    stats = []
    for rows, cols in clusters:
        wgt = w2d[rows, cols]
        stats.append((float(wgt.max()),
                      float((cols * wgt).sum() / wgt.sum()) * dx,
                      float((rows * wgt).sum() / wgt.sum()) * dy))
    stats.sort(reverse=True)               # strongest cores first
    if len(stats) < 2:
        return len(stats), np.nan, np.nan, np.nan, np.nan
    du, dv = np.asarray(shear_uv, dtype=np.float64)
    right_normal = np.array([dv, -du], dtype=np.float64)
    pair = sorted(stats[:2], key=lambda item:
                  item[1] * right_normal[0] + item[2] * right_normal[1],
                  reverse=True)
    _, xr, yr = pair[0]
    _, xl, yl = pair[1]
    return len(stats), xr, yr, xl, yl


def _uh_proxy(state, cfg, z_half):
    """UH proxy max(int w*zeta dz) over 2-5 km, float64 (ny, nx) max.

    zeta = dv/dx - du/dy with the staggered winds averaged to mass points
    and centered differences (open boundaries: one-sided edges via
    np.gradient); w averaged to half levels; the eta-layer depths come
    from the base-state half-level heights.
    """
    import cupy as cp

    sel = (z_half >= UH_ZBOT) & (z_half <= UH_ZTOP)
    dz = np.gradient(z_half)[sel]
    u = cp.asnumpy(state.u[sel]).astype(np.float64)
    v = cp.asnumpy(state.v[sel]).astype(np.float64)
    wf = cp.asnumpy(state.w).astype(np.float64)
    wh = 0.5 * (wf[:-1] + wf[1:])[sel]
    um = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
    vm = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
    zeta = (np.gradient(vm, cfg.dx, axis=2)
            - np.gradient(um, cfg.dy, axis=1))
    uh = np.sum(wh * zeta * dz[:, None, None], axis=0)
    return float(uh.max())


def _plot_planviews(snaps, key, x, y, path, label, cmap, vmin, vmax):
    """One row of plan views (t = 3600/5400/7200 s) of ``snaps[t][key]``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(snaps), figsize=(15.0, 4.8),
                             sharey=True)
    xk, yk = x / 1000.0, y / 1000.0
    for ax, (t, snap) in zip(np.atleast_1d(axes), sorted(snaps.items())):
        pm = ax.pcolormesh(xk, yk, snap[key], cmap=cmap,
                           vmin=vmin, vmax=vmax, shading="auto")
        ax.set_title(f"t = {int(t)} s")
        ax.set_xlabel("x (km)")
        ax.set_aspect("equal")
    np.atleast_1d(axes)[0].set_ylabel("y (km)")
    fig.colorbar(pm, ax=list(np.atleast_1d(axes)), label=label,
                 shrink=0.85)
    fig.suptitle(f"wk82: {label}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_hodograph(coord, ztop, z_half, path):
    """Quarter-circle hodograph: model-level winds + the shipped columns."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    u, v = hodograph_uv(coord, ztop)
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.plot(HODOGRAPH_ZUV[:-1, 1], HODOGRAPH_ZUV[:-1, 2], "-", color="0.7",
            label="input_sounding columns")
    sel = z_half <= 12000.0
    sc = ax.scatter(u[sel], v[sel], c=z_half[sel] / 1000.0, cmap="viridis",
                    s=18, zorder=3, label="model half levels")
    for zz in (1000.0, 3000.0, 6000.0):
        k = int(np.argmin(np.abs(z_half - zz)))
        ax.annotate(f"{zz / 1000:.0f} km", (u[k], v[k]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.axhline(0.0, color="0.85", lw=0.8)
    ax.axvline(0.0, color="0.85", lw=0.8)
    ax.set_xlabel("u (m/s)")
    ax.set_ylabel("v (m/s)")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8)
    fig.colorbar(sc, ax=ax, label="z (km)")
    ax.set_title("wk82 quarter-circle hodograph (storm-motion frame)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(outdir=None) -> dict:
    """Build the WK82 supercell case, integrate 2 h, return the metrics.

    Metrics (plan Task 11): ``w_max_hist`` (per-step (t, max w) history),
    ``w_max`` (peak after 1800 s spin-up), ``w_sustained_fraction`` (the
    fraction of t = [3600, 7200] samples above 25 m/s),
    ``qr_max`` (kg/kg, peak over the run), split diagnostics at
    t = 3600/5400/7200 s (``n_cores_T``, right/left mover centroid
    ``y_right_T``/``y_left_T`` from w > 15 m/s clusters at z ~ 5 km,
    separations ``sep_T``, drifts ``right_dy``/``left_dy``), the
    ``uh_proxy`` max(int w*zeta dz, 2-5 km) at the final time, ``nan``,
    and relative dry-mass drift ``mass_drift`` (FP64) plus the maximum
    relative closure residual against an independently integrated lateral
    boundary flux.  Separation is gated at 3600 s, across the early
    3600->5400 s interval, and in total from 3600->7200 s; the later
    interval remains informational.  Right-mover displacement is projected
    against the 0-6 km mean-shear vector.  With
    an ``outdir``, writes the three PNGs (wk82_w_z5km/wk82_qr_z1km plan
    views at the analysis times, and the hodograph) and
    ``wrfout_wk82.nc`` (a frame every ``cfg.output_interval_s``, moisture
    scalars and accumulated RAINNC included).
    """
    import cupy as cp
    from pathlib import Path

    from gpuwm.core.dycore import stability_report, step
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.config import validate_run_config

    # This idealized case has no PBL, so km_opt=4 selects the ported WRF
    # vertical_diffusion_2 stresses and surface-flux branch in addition to
    # horizontal Smagorinsky diffusion.
    cfg = validate_run_config(default_config())
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: wk82_sounding(z)[0],
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = build(cfg, coord, base)

    z_half = state.height_half()
    kw5 = int(np.argmin(np.abs(base.phb / c.G - CORE_Z)))  # w level ~ 5 km
    kq1 = int(np.argmin(np.abs(z_half - 1000.0)))          # half level ~ 1 km

    def dry_mass():
        return float(cp.sum((state.mub2d + state.mup).astype(cp.float64)))

    writer = None
    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        from gpuwm.io.wrfout import WrfoutWriter, state_frame, wrf_time_str
        writer = WrfoutWriter(outdir / "wrfout_wk82.nc", nx=cfg.nx,
                              ny=cfg.ny, nz=cfg.nz, dx=cfg.dx, dy=cfg.dy,
                              title="gpuwm WK82 quarter-circle supercell")
        rainnc = state.scratch((cfg.ny, cfg.nx), "mp_rainnc")

    def write_frame(t_s: float) -> None:
        writer.write_frame(wrf_time_str(t_s), state_frame(state)
                           | {"RAINNC": cp.asnumpy(rainnc)})

    mass0 = dry_mass()
    mass_drift_max = 0.0
    boundary_flux_integral = 0.0
    mass_closure_residual_max = 0.0
    n_total = int(round(cfg.run_seconds / cfg.dt))
    n_frame = max(int(round(cfg.output_interval_s / cfg.dt)), 1)
    snap_steps = {int(round(t / cfg.dt)): t for t in ANALYSIS_TIMES}
    mass_steps = set(mass_sample_steps(n_total))       # T11-1: 10-step cadence

    w_hist = []
    qr_max = 0.0
    snaps = {}

    def accumulate_boundary_flux(increment: float) -> None:
        nonlocal boundary_flux_integral
        boundary_flux_integral += float(increment)

    try:
        if writer is not None:
            write_frame(0.0)
        for n in range(1, n_total + 1):
            step(state, cfg, mass_flux_observer=accumulate_boundary_flux)
            t = n * cfg.dt
            w_hist.append((t, float(state.w.max())))
            qr_max = max(qr_max, float(state.qr.max()))
            if n in mass_steps:
                mass_now = dry_mass()
                mass_drift_max = max(
                    mass_drift_max, abs(mass_now - mass0) / mass0)
                mass_closure_residual_max = max(
                    mass_closure_residual_max,
                    abs((mass_now - mass0) - boundary_flux_integral) / mass0)
            if n in snap_steps:
                snaps[snap_steps[n]] = {
                    "w5": cp.asnumpy(state.w[kw5]).astype(np.float64),
                    "qr1": 1000.0
                    * cp.asnumpy(state.qr[kq1]).astype(np.float64),
                }
            if writer is not None and n % n_frame == 0:
                write_frame(t)
    finally:
        if writer is not None:
            writer.close()

    report = stability_report(state, cfg)
    finite = all(bool(cp.isfinite(q).all()) for q in
                 (state.qv, state.qc, state.qr))
    mass1 = dry_mass()
    mass_drift_max = max(mass_drift_max, abs(mass1 - mass0) / mass0)
    mass_closure_residual_max = max(
        mass_closure_residual_max,
        abs((mass1 - mass0) - boundary_flux_integral) / mass0)

    th = np.asarray(w_hist, dtype=np.float64)
    after = th[th[:, 0] > 1800.0]
    sustained_fraction = sustained_updraft_fraction(th)

    metrics = {
        "nan": bool(report["nan"] or not finite),
        "w_max": float(after[:, 1].max()),
        "w_max_sustained": float(th[(th[:, 0] >= 3600.0), 1].min()),
        "w_sustained_fraction": sustained_fraction,
        "w_max_hist": [tuple(row) for row in th],
        "qr_max": qr_max,
        "uh_proxy": _uh_proxy(state, cfg, z_half),
        "mass_drift": abs(mass1 - mass0) / mass0,
        "mass_drift_max": mass_drift_max,
        "mass_closure_residual_max": mass_closure_residual_max,
        "boundary_flux_exchange": abs(boundary_flux_integral) / mass0,
    }
    u_hod, v_hod = hodograph_uv(coord, cfg.ztop)
    k0 = int(np.argmin(np.abs(z_half)))
    k6 = int(np.argmin(np.abs(z_half - 6000.0)))
    shear_uv = (u_hod[k6] - u_hod[k0], v_hod[k6] - v_hod[k0])
    for t, snap in snaps.items():
        n_cores, x_right, y_right, x_left, y_left = _movers(
            snap["w5"], cfg.dx, cfg.dy, shear_uv)
        tag = str(int(t))
        metrics["n_cores_" + tag] = n_cores
        metrics["x_right_" + tag] = x_right
        metrics["y_right_" + tag] = y_right
        metrics["x_left_" + tag] = x_left
        metrics["y_left_" + tag] = y_left
        metrics["sep_" + tag] = y_left - y_right
    metrics["right_dy"] = metrics["y_right_7200"] - metrics["y_right_3600"]
    metrics["left_dy"] = metrics["y_left_7200"] - metrics["y_left_3600"]
    # Gate-ready derived forms (GATES bounds single metrics, so the
    # cross-metric acceptance criteria become metrics of their own).
    growth_1, growth_2 = separation_growth_intervals({
        t: metrics["sep_" + str(int(t))] for t in ANALYSIS_TIMES})
    metrics["sep_growth_3600_5400"] = growth_1
    metrics["sep_growth_5400_7200"] = growth_2
    metrics["sep_growth"] = metrics["sep_7200"] - metrics["sep_3600"]
    metrics["right_rel_dy"] = metrics["right_dy"] - metrics["left_dy"]
    metrics["right_of_shear_m"] = right_of_shear_distance(
        (metrics["x_right_3600"], metrics["y_right_3600"]),
        (metrics["x_right_7200"], metrics["y_right_7200"]), shear_uv)

    if outdir is not None:
        x = (np.arange(cfg.nx) + 0.5) * cfg.dx - 0.5 * cfg.nx * cfg.dx
        y = (np.arange(cfg.ny) + 0.5) * cfg.dy - 0.5 * cfg.ny * cfg.dy
        wmax = max(np.abs(s["w5"]).max() for s in snaps.values()) or 1.0
        _plot_planviews(snaps, "w5", x, y, outdir / "wk82_w_z5km.png",
                        "w at z = 5 km (m/s)", "RdBu_r", -wmax, wmax)
        qmax = max(s["qr1"].max() for s in snaps.values()) or 0.1
        _plot_planviews(snaps, "qr1", x, y, outdir / "wk82_qr_z1km.png",
                        "qr at z = 1 km (g/kg)", "Greens", 0.0, qmax)
        _plot_hodograph(coord, cfg.ztop, z_half, outdir / "wk82_hodograph.png")
    return metrics
