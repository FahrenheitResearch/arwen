#!/usr/bin/env python3
"""Named WRF v4.6.1 rules checked on hand-built columns, through the
production adapters on the host.

The real-column fixture (``fixture_check.py``) grades the port against
WRF's own answers where the saved states happen to exercise a rule.  Some
rules act on cells a saved state rarely holds -- a column with no
microphysics at all, vapour at exactly zero, condensate at or below R1 at a
melting level -- so each is also checked here on a column built to hold it,
against what the rule itself says the cell must be.  Every check runs the
production adapter unmodified (``gpuwm.core.microphysics_aerosol.
_apply_thompson_aerosol`` for mp=28, ``gpuwm.core.microphysics.
_apply_thompson`` for mp=8) through the host backend, exactly as
``real_column_parity`` does.

Bare line numbers are WRF v4.6.1's ``phys/module_mp_thompson.F``.

usage: rule_checks.py [CHECK ...]      prints one JSON object
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import real_column_parity as R  # noqa: E402

f32 = np.float32
G = 9.81
R1 = 1.0e-12


def _qvs(temperature, pressure):
    """Liquid saturation mixing ratio (Bolton), for building columns."""
    es = 611.2 * np.exp(17.67 * (temperature - 273.15)
                        / (temperature - 29.65))
    return 0.622 * es / (pressure - es)


def columns(ncol, nz=12, dz=500.0, rh=0.5, t_surface=300.0):
    """``ncol`` identical columns: a 6.5 K/km troposphere, relative humidity
    ``rh`` over liquid, no condensate, aerosol at 1e9 per kg."""
    zi = np.arange(nz + 1, dtype=np.float64) * dz
    zm = 0.5 * (zi[1:] + zi[:-1])
    temperature = t_surface - 6.5e-3 * zm
    pressure = 1.0e5 * (temperature / t_surface) ** (G / (287.04 * 6.5e-3))
    theta = temperature * (1.0e5 / pressure) ** (287.0 / 1004.0)
    qv = rh * _qvs(temperature, pressure)

    def lev(a):
        return np.repeat(np.asarray(a, np.float64)[None, :], ncol,
                         axis=0).astype(f32)

    cols = {
        "p": lev(pressure), "th": lev(theta), "geop": lev(zi * G),
        "w": np.zeros((ncol, nz + 1), f32), "qv": lev(qv),
        "nwfa2d": np.zeros((ncol,), f32), "nifa2d": np.zeros((ncol,), f32),
        "dt": f32(5.0),
    }
    for name in ("qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc"):
        cols[name] = np.zeros((ncol, nz), f32)
    cols["nwfa"] = np.full((ncol, nz), 1.0e9, f32)
    cols["nifa"] = np.full((ncol, nz), 1.0e6, f32)
    return cols


def run(cols, mp):
    """One adapter call; the final state as ``{name: (ncol, nz) float64}``."""
    dt = float(cols["dt"])
    out = R.run_port(R.prepare(cols, mp), dt, rates=False, mp=mp)
    return out["final"]


def check_no_micro_column(mp):
    """:2020.  Column 0 has no condensate above R1 and is nowhere
    supersaturated over ice, so WRF returns before the source loop: its
    top-level vapour stays at exactly zero (no :3974 floor), its aerosol
    above the 9999.E6 ceiling stays (no :3979 clamp), and its orphans at or
    below R1 leave as the entry rewrite leaves them, zero.  Column 1 is the
    same plus 0.1 g/kg of cloud at the lowest level: it has microphysics,
    so its zero vapour leaves floored at 1.E-10 and its aerosol clamped."""
    cols = columns(2)
    top = cols["qv"].shape[1] - 1
    cols["qv"][:, top] = 0.0
    cols["qc"][:, 2] = 5.0e-13
    cols["qi"][:, 9] = 5.0e-13
    cols["ni"][:, 9] = 100.0
    cols["nwfa"][:, :] = 2.0e10
    cols["qc"][1, 0] = 1.0e-4
    fin = run(cols, mp)
    out = {"qv_top": [float(v) for v in fin["qv"][:, top]],
           "qc_orphan": [float(v) for v in fin["qc"][:, 2]],
           "qi_orphan": [float(v) for v in fin["qi"][:, 9]],
           "ni_orphan": [float(v) for v in fin["ni"][:, 9]]}
    if mp == 28:
        out["nwfa_max"] = [float(v) for v in fin["nwfa"].max(axis=1)]
    return out


def check_terminal_graupel_zero():
    """:4058-4063.  The terminal apply writes graupel at or below R1 as zero,
    mass and number, in every column; the graupel number finalize is the
    last writer of the graupel mass on both schemes, so it carries that
    zero.  Levels: 0.5 R1, exactly R1, one float32 unit above R1, 2 R1,
    zero, 1e-4."""
    from gpuwm.core.thompson import launch_classic_graupel_number_finalize
    nz = 6
    qg = np.array([0.5e-12, 1.0e-12, np.nextafter(f32(1.0e-12), f32(1.0)),
                   2.0e-12, 0.0, 1.0e-4], f32).reshape(nz, 1, 1)
    temperature = np.full((nz, 1, 1), 260.0, f32)
    pressure = np.full((nz, 1, 1), 60000.0, f32)
    qv = np.full((nz, 1, 1), 1.0e-3, f32)
    ng = np.full((nz, 1, 1), 50.0, f32)
    launch_classic_graupel_number_finalize(qg, temperature, pressure, qv, ng)
    return {"qg": [float(v) for v in qg.ravel()],
            "ng_zero": [bool(v == 0.0) for v in ng.ravel()]}


def check_condensate_at_or_below_r1_melts(mp):
    """:3943-3966 then :4007-4009 and :4023-4027.  WRF melts ANY positive
    ice at a level above 0 C into cloud water, and only the terminal apply
    after that removes cloud and ice at or below R1.  So cloud and ice that
    each sit at or below R1 when the fallout is done, but together exceed
    it, leave as cloud.  Level 0 holds 1e-4 kg/kg of cloud, which opens the
    cloud fallout's column gate; level 2 holds 0.8e-12 kg/kg each of cloud
    and ice, level 3 0.4e-12 each, all near 280 K.  The fallout and the
    phase cleanup run as each adapter runs them, on the state as the sources
    would hand it over (an adapter call would zero these at entry, :1844 and
    :1871, so they are built past it).  Returned: the cloud and ice each
    level leaves with."""
    from gpuwm.core import thompson as T
    nz = 4
    shape = (nz, 1, 1)
    temperature = np.full(shape, 280.0, f32)
    pressure = np.full(shape, 90000.0, f32)
    qv = np.full(shape, 5.0e-3, f32)
    rho = (0.622 * pressure / (287.04 * temperature * (qv + 0.622))).astype(f32)
    qc = np.zeros(shape, f32)
    qi = np.zeros(shape, f32)
    ni = np.zeros(shape, f32)
    qc[0] = 1.0e-4
    qc[2], qi[2], ni[2] = 0.8e-12, 0.8e-12, 1.0
    qc[3], qi[3], ni[3] = 0.4e-12, 0.4e-12, 1.0
    dz = np.full(shape, 200.0, f32)
    w = np.zeros(shape, f32)
    sfc = [np.zeros((1, 1), f32) for _ in range(4)]
    gate = np.ones((1, 1), f32)
    dt = 5.0
    if mp == 28:
        from gpuwm.core import thompson_aerosol_sed as S
        nc = np.zeros(shape, f32)
        nc[0] = 1.0e8
        ncten = np.zeros(shape, f32)
        S.launch_aa_cloud_sedimentation(
            qc, nc, ncten, temperature, pressure, qv, w, dz, dt,
            reference_density=rho, rain_active_columns=np.zeros((1, 1), f32),
            cloud_active_columns=gate)
        T.launch_ice_sedimentation(qi, ni, temperature, pressure, qv, dz,
                                   *sfc, dt, reference_density=rho)
        S.launch_aa_final_phase_cleanup(
            qc, qi, ni, temperature, nc, ni.copy(), ncten, pressure, qv, dt)
    else:
        T.launch_cloud_sedimentation(
            qc, temperature, pressure, qv, w, dz, dt, reference_density=rho,
            rain_active_columns=np.zeros((1, 1), f32),
            cloud_active_columns=gate)
        T.launch_ice_sedimentation(qi, ni, temperature, pressure, qv, dz,
                                   *sfc, dt, reference_density=rho)
        T.launch_final_phase_cleanup(qc, qi, ni, temperature, pressure, qv,
                                     micro_columns=gate)
    return {"qc": [float(v) for v in qc.ravel()],
            "qi": [float(v) for v in qi.ravel()]}


def check_melting_snow_blend(mp):
    """:3612-3634 and :3722-3724.  Melting snow (prr_sml > 0) falls at
    vtsk = vts*SR + (1-SR)*vtrk(k) with SR = rs/(rs+rr), where rr(k) and
    vtrk(k) are the rain pass's own: a level with no rain has rr = R1 and
    takes vtrk from the level above, and a column with no rain has vtrk = 0.
    Level 2 of a warm column (276 K, dz 200 m) holds 5e-12 kg/kg of snow and
    no rain; three arms run the snow fallout once, as each adapter runs it:
    ``still`` without the melting marker (vtsk = vts), ``dry`` melting with
    no rain in the column, ``rain_above`` melting with 2e-4 kg/kg of rain at
    level 5.  With one substep the snow each arm removes from level 2 is
    vtsk*qs*dt/dz, so ``still`` measures vts and the rule fixes the other
    two.  Returned: the snow each arm removes, and the two the rule says."""
    from gpuwm.core import thompson as T
    nz, dz, dt = 8, 200.0, 5.0
    shape = (nz, 1, 1)
    temperature = np.full(shape, 276.0, f32)
    pressure = np.full(shape, 85000.0, f32)
    qv = np.full(shape, 5.0e-3, f32)
    rho = (0.622 * pressure / (287.04 * temperature * (qv + 0.622))).astype(
        f32)
    dzs = np.full(shape, dz, f32)
    boost = np.ones(shape, f32)
    qs0 = f32(5.0e-12)

    def arm(melting, rain):
        qs = np.zeros(shape, f32)
        qs[2] = qs0
        marker = np.zeros(shape, f32)
        if melting:
            marker[2] = 1.0
        qr = np.zeros(shape, f32)
        nr = np.zeros(shape, f32)
        if rain:
            qr[5], nr[5] = 2.0e-4, 2.0e3
        # mp=28's rain evaporation writes L_qr into the rain fallout's
        # density (zero where it failed); mp=8 carries the density itself.
        density = (np.where(qr > R1, rho, f32(0.0)).astype(f32)
                   if mp == 28 else rho.copy())
        sfc = [np.zeros((1, 1), f32) for _ in range(4)]
        T.launch_snow_sedimentation(
            qs, temperature, pressure, qv, dzs, *sfc, dt,
            reference_density=rho, reference_temperature=temperature,
            snow_melt_marker=marker, melt_rain_qr=qr, melt_rain_nr=nr,
            velocity_boost=boost, melt_rain_density=density,
            melt_rain_density_carries_presence=(mp == 28))
        return float(qs0) - float(qs[2, 0, 0]), qr, nr

    still, _, _ = arm(False, False)
    dry, _, _ = arm(True, False)
    rain_above, qr, nr = arm(True, True)
    rho2 = float(rho[2, 0, 0])
    rs = float(qs0) * rho2
    sr = rs / (rs + R1)
    # vtrk(5): WRF's rain mass fall speed (:3619-3620) for the rain at level
    # 5, which levels 4 to 2 inherit (:3630).
    rho5 = float(rho[5, 0, 0])
    rr = float(qr[5, 0, 0]) * rho5
    nn = max(1.0e-6, float(nr[5, 0, 0]) * rho5)
    lam = (np.pi * 1000.0 / 6.0 * 6.0 * nn / rr) ** (1.0 / 3.0)
    rho_not = 101325.0 / (287.05 * 298.0)
    vtrk = (np.sqrt(rho_not / rho5) * 4854.0 * 24.0 / 6.0 * lam ** 4
            * (lam + 195.0) ** -5)
    vts = still * dz / (float(qs0) * dt)
    return {"removed": {"still": still, "dry": dry,
                        "rain_above": rain_above},
            "rule": {"dry": sr * still,
                     "rain_above": (sr * vts + (1.0 - sr) * vtrk)
                     * float(qs0) * dt / dz},
            "solid_fraction": sr, "vtrk": vtrk, "vts": vts}


def check_presence_is_the_mixing_ratio(mp):
    """:1827-1949 with :2783-2825.  WRF's L_qs and L_qg are the MIXING RATIO
    tests of the entry block, and its warm-level melting runs wherever they
    hold, rs(k) and rg(k) being whatever q*rho is.  In thin air a mixing
    ratio just above R1 is a concentration at or below it: snow of 1.1e-12
    kg/kg at level 7 (about 276 K, rho about 0.8) and graupel of 1.1e-12 at
    level 6 (about 279 K) melt there.  Returned: the snow and graupel each
    level leaves with, the rain the melt made there, and the densities."""
    cols = columns(1, rh=0.97)
    cols["qs"][0, 7] = 1.1e-12
    cols["qg"][0, 6] = 1.1e-12
    inp = R.prepare(cols, mp)
    fin = run(cols, mp)
    rho = (inp["p"] / (287.04 * inp["T"]
                       * (cols["qv"].astype(np.float64) + 0.622)) * 0.622)
    return {"temperature": [float(inp["T"][0, k]) for k in (7, 6)],
            "concentration": [float(1.1e-12 * rho[0, k]) for k in (7, 6)],
            "qs": float(fin["qs"][0, 7]), "qg": float(fin["qg"][0, 6]),
            "qr": [float(fin["qr"][0, k]) for k in (7, 6)]}


def check_classic_rain_presence_export():
    """:3236, :3252-3253 and :3501-3572.  The rain fallout forms WRF's rr(k)
    from L_qr and from whichever density built it: the TAU+1 one of :3237
    at every level with L_qr, rewritten from the post-condensation density
    (and floored at R1) at :3568 only where the rain evaporation ran.  The
    classic rain evaporation carries both into the fallout's density, as the
    mp=28 one does: zero where L_qr failed, the held density where rain did
    not evaporate, the negative of its own density where it did.  Level 0
    holds 5e-13 kg/kg of rain (L_qr fails), level 1 1e-4 kg/kg in air at 50
    percent humidity (it evaporates), level 2 1e-4 kg/kg in air just
    supersaturated (it does not).  Returned: the density the kernel leaves
    at each level, the held density and the kernel's own."""
    from gpuwm.core import thompson as T
    nz = 3
    shape = (nz, 1, 1)
    temperature = np.full(shape, 285.0, f32)
    pressure = np.full(shape, 90000.0, f32)
    qvs = _qvs(285.0, 90000.0)
    qv = np.array([0.5 * qvs, 0.5 * qvs, 1.01 * qvs], f32).reshape(shape)
    qr = np.array([5.0e-13, 1.0e-4, 1.0e-4], f32).reshape(shape)
    nr = np.array([10.0, 1.0e4, 1.0e4], f32).reshape(shape)
    held = np.full(shape, 1.05, f32)
    rho = (f32(0.622) * pressure
           / (f32(287.04) * temperature * (np.maximum(qv, f32(1.0e-10))
                                           + f32(0.622)))).astype(f32)
    density = np.full(shape, np.nan, f32)
    marker = np.zeros(shape, f32)
    melt = np.zeros(shape, f32)
    T.launch_rain_evaporation(
        qr, nr, temperature, pressure, qv, 5.0,
        reference_density=density, graupel_melt_marker=melt,
        source_density=held, condensation_marker=marker,
        density_carries_rain_presence=True)
    return {"density": [float(v) for v in density.ravel()],
            "held": [float(v) for v in held.ravel()],
            "own": [float(v) for v in rho.ravel()],
            "qr": [float(v) for v in qr.ravel()]}


def _ice_number_in_bounds(qi, ni, temperature, pressure, qv):
    """WRF's cloud ice mass/number balance (:3033-3055) on an entry level
    no source process touched (qiten = niten = 0), per kilogram, float64:
    the number that keeps the mass-weighted mean size between 5 and 300
    microns, and no more than 999 per litre where the 5 micron end binds."""
    rho = 0.622 * pressure / (287.04 * temperature
                              * (max(qv, 1.0e-10) + 0.622))
    am_i = np.pi * 890.0 / 6.0
    ri = qi * rho
    xni = max(1.0e-6, ni * rho)
    diameter = 4.0 / (am_i * 6.0 * xni / ri) ** (1.0 / 3.0)
    if diameter < 5.0e-6:
        xni = min(999.0e3, ri / (6.0 * am_i) * (4.0 / 5.0e-6) ** 3)
    elif diameter > 300.0e-6:
        xni = ri / (6.0 * am_i) * (4.0 / 300.0e-6) ** 3
    return min(xni, 999.0e3) / rho


def check_warm_ice_number(mp):
    """:3033-3055.  The cloud ice mass/number balance runs at every level of
    a column with microphysics, above 0 C as well as below, so ice that sits
    at a warm level keeps its number inside the 5 to 300 micron bounds when
    the fallout reads it: ice arriving there with no number takes the 300
    micron number, ice with far too many crystals the 5 micron one.  Level 5
    (about 282 K) holds 2e-5 kg/kg of ice and no number, level 4 (about
    285 K) 1e-5 kg/kg and 1e9 per kg.  Returned: the ice number the port
    holds after its source stage at both levels, and the balance's own
    answer."""
    cols = columns(1)
    cols["qi"][0, 5] = 2.0e-5
    cols["ni"][0, 5] = 0.0
    cols["qi"][0, 4] = 1.0e-5
    cols["ni"][0, 4] = 1.0e9
    inp = R.prepare(cols, mp)
    out = R.run_port(inp, float(cols["dt"]), rates=False, mp=mp)
    got = out["stages"]["cold"]["ni"][0]
    levels = (5, 4)
    want = [_ice_number_in_bounds(
        float(cols["qi"][0, k]), float(cols["ni"][0, k]),
        float(inp["T"][0, k]), float(inp["p"][0, k]),
        float(cols["qv"][0, k])) for k in levels]
    if mp == 28:
        # mp=28 (v4.6.1) carries the balance as WRF does, in tendency form:
        # niten = (xni - ni1d*rho)*odts*orho, then ni1d + niten*DT, all REAL.
        # Where the entry number is far above the bound (level 4: 1e9 per
        # kg against ~1.07e6) that subtraction cancels to the float32 ulp of
        # the entry, so WRF's own answer sits ~9e-5 from the exact target;
        # the expectation is WRF's arithmetic, not the exact target.
        f32 = np.float32
        dt = f32(cols["dt"])
        tendency_form = []
        for k, target in zip(levels, want):
            temp = float(inp["T"][0, k])
            rho = f32(f32(0.622) * f32(inp["p"][0, k]) / (
                f32(287.04) * f32(temp)
                * (f32(max(float(cols["qv"][0, k]), 1.0e-10)) + f32(0.622))))
            ni1d = f32(cols["ni"][0, k])
            xni = f32(target * float(rho))
            niten = f32(f32(f32(xni - f32(ni1d * rho)) * f32(f32(1.0) / dt))
                        * f32(f32(1.0) / rho))
            tendency_form.append(float(f32(ni1d + f32(niten * dt))))
        want = tendency_form
    return {"temperature": [float(inp["T"][0, k]) for k in levels],
            "ni_after_sources": [float(got[k]) for k in levels],
            "ni_balance": want}


CHECKS = {
    "G_no_micro_column_mp28": lambda: check_no_micro_column(28),
    "G_no_micro_column_mp8": lambda: check_no_micro_column(8),
    "T_terminal_graupel_zero": check_terminal_graupel_zero,
    "Awarm_ice_number_mp28": lambda: check_warm_ice_number(28),
    "Awarm_ice_number_mp8": lambda: check_warm_ice_number(8),
    "J_melt_before_r1_mp28": lambda: check_condensate_at_or_below_r1_melts(28),
    "J_melt_before_r1_mp8": lambda: check_condensate_at_or_below_r1_melts(8),
    "I_melting_snow_blend_mp28": lambda: check_melting_snow_blend(28),
    "I_melting_snow_blend_mp8": lambda: check_melting_snow_blend(8),
    "P_presence_mp28": lambda: check_presence_is_the_mixing_ratio(28),
    "P_presence_mp8": lambda: check_presence_is_the_mixing_ratio(8),
    "R_classic_rain_presence": check_classic_rain_presence_export,
}


def main(argv):
    names = argv[1:] or list(CHECKS)
    unknown = [n for n in names if n not in CHECKS]
    if unknown:
        raise SystemExit(f"unknown checks {unknown}; known {list(CHECKS)}")
    print(json.dumps({name: CHECKS[name]() for name in names}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
