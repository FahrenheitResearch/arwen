"""Column oracle for WOOF's RTE+RRTMGP against the upstream Fortran.

Three steps, each a subcommand:

  make     build the column set (72 columns x 60 layers, seven regimes plus
           edge cases), draw the McICA masks with WOOF's own generator, and
           write both an .npz (WOOF's input) and the stream file the Fortran
           driver reads.  The masks and the TOA source are INPUTS: both
           sides receive the same bits.
  woof     run WOOF's GPU path on the .npz and save every output.  Two
           compositions: "fused" is the production path with a chunk
           workspace (gas optics, band cloud optics, then solvers that
           derive the Planck sources and fold the clouds in registers);
           "staged" is the workspace-less production path (standalone
           Planck kernel, finalize kernels, then the solvers).  A third arm
           re-runs the fused solvers with GPUWM_RTE_FOLD=0 (partials plus
           reduce kernel instead of the in-block fold).
  compare  bitwise comparison of every WOOF output with the Fortran's,
           reported in ULP (distance in the ordered binary32 lattice).

The upstream reference is earth-system-radiation/rte-rrtmgp fa107a16 built
single precision by build.sh; see README.md for the full recipe.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

F32 = np.float32
NCOL_PER_REGIME = 9
NLAY = 60
REGIMES = ("tropical-convective", "midlatitude-summer", "stable-night",
           "polar-snow-ice", "ocean-stratocumulus", "high-terrain",
           "dry-subtropical", "edge-cases")

GAS_NAMES = ("h2o", "co2", "o3", "n2o", "co", "ch4", "o2", "n2", "ccl4",
             "cfc11", "cfc12", "cfc22", "hfc143a", "hfc125", "hfc23",
             "hfc32", "hfc134a", "cf4", "no2")


# ---------------------------------------------------------------- columns
def _levels(ps, ptop, nlay, stretch):
    s = np.linspace(0.0, 1.0, nlay + 1)
    return np.exp(np.log(ps) + (np.log(ptop) - np.log(ps)) * s ** stretch)


def _temperature(play, ps, tsfc, lapse, tropo_p, strat_warm, inversion=0.0,
                 inv_depth=0.0):
    z = 7000.0 * np.log(ps / play)              # approximate height, m
    ztrop = 7000.0 * np.log(ps / tropo_p)
    t = np.where(z < ztrop, tsfc - lapse * z, tsfc - lapse * ztrop)
    above = np.maximum(z - ztrop, 0.0)
    t = t + strat_warm * above / 1000.0 * (above < 32000.0) \
        + strat_warm * 32.0 * (above >= 32000.0)
    if inversion:
        t = t + inversion * np.clip(1.0 - z / inv_depth, 0.0, 1.0) \
            + lapse * np.minimum(z, inv_depth) * (z < inv_depth)
    return np.clip(t, 160.5, 354.5)


def make_columns(seed=20261007):
    import json as _json
    from gpuwm.core import rrtmgp as R

    rng = np.random.default_rng(seed)
    clim = R.load_trace_climatology()
    trace = dict(clim.trace_vmr)
    o3_logp = np.log(clim.pressure_layer_pa)
    o3_vmr = clim.ozone_vmr
    order = np.argsort(o3_logp)
    o3_logp, o3_vmr = o3_logp[order], o3_vmr[order]
    sw_tables = R.load_gas_tables("sw")
    cld = R.load_cloud_tables("lw")
    p_trop32 = F32(sw_tables.press_ref_trop)
    p_min32 = F32(np.min(sw_tables.press_ref))

    cols = []
    for regime in REGIMES:
        for k in range(NCOL_PER_REGIME):
            u = rng.random(8)
            c = {"regime": regime}
            ptop = 1.0052 * (1.0 + 4.0 * u[0])     # 1 to 5 Pa
            ps, tsfc, lapse, trop, warm = 101325.0, 288.0, 0.0065, 20000.0, 2.0
            qsfc, inv, invd = 0.010, 0.0, 0.0
            emis, alb, mu = 0.98, 0.15, 0.6
            liq = ice = None
            if regime == "tropical-convective":
                ps, tsfc, lapse, trop, warm = 100800.0, 300.0 + 3 * u[1], \
                    0.0062, 10000.0, 2.5
                qsfc = 0.018 + 0.004 * u[2]
                liq, ice = (60000.0, 95000.0, 40.0 + 120 * u[3]), \
                    (15000.0, 50000.0, 20.0 + 60 * u[4])
                emis, alb, mu = 0.97, 0.12, 0.3 + 0.7 * u[5]
            elif regime == "midlatitude-summer":
                ps, tsfc, lapse, trop, warm = 101000.0, 292.0 + 6 * u[1], \
                    0.0065, 20000.0, 2.0
                qsfc = 0.010 + 0.004 * u[2]
                liq = (70000.0, 90000.0, 5.0 + 30 * u[3])
                ice = (25000.0, 40000.0, 2.0 + 10 * u[4])
                emis, alb, mu = 0.96, 0.18, 0.4 + 0.6 * u[5]
            elif regime == "stable-night":
                ps, tsfc, lapse, trop, warm = 100500.0, 262.0 + 8 * u[1], \
                    0.0060, 22000.0, 2.0
                qsfc, inv, invd = 0.002 + 0.002 * u[2], 8.0 + 6 * u[3], 400.0
                emis, alb, mu = 0.99, 0.25, -0.3
            elif regime == "polar-snow-ice":
                ps, tsfc, lapse, trop, warm = 100000.0, 228.0 + 15 * u[1], \
                    0.0045, 30000.0, 1.0
                qsfc, inv, invd = 0.0004 + 0.0004 * u[2], 6.0, 800.0
                ice = (50000.0, 90000.0, 1.0 + 15 * u[3])
                emis, alb, mu = 0.995, 0.78 + 0.1 * u[4], 0.05 + 0.25 * u[5]
            elif regime == "ocean-stratocumulus":
                ps, tsfc, lapse, trop, warm = 101800.0, 289.0 + 4 * u[1], \
                    0.0065, 18000.0, 2.0
                qsfc, inv, invd = 0.009 + 0.002 * u[2], 0.0, 0.0
                liq = (88000.0, 96000.0, 30.0 + 120 * u[3])
                emis, alb, mu = 0.985, 0.06, 0.2 + 0.8 * u[5]
            elif regime == "high-terrain":
                ps, tsfc, lapse, trop, warm = 60000.0 + 10000 * u[1], \
                    272.0 + 12 * u[2], 0.0070, 18000.0, 2.0
                qsfc = 0.004 + 0.003 * u[3]
                ice = (30000.0, 45000.0, 5.0 + 20 * u[4])
                emis, alb, mu = 0.95, 0.25 + 0.5 * u[5], 0.5 + 0.5 * u[6]
            elif regime == "dry-subtropical":
                ps, tsfc, lapse, trop, warm = 101200.0, 305.0 + 10 * u[1], \
                    0.0080, 15000.0, 2.0
                qsfc = 0.0005 * u[2]
                emis, alb, mu = 0.90 + 0.05 * u[3], 0.35, 0.7 + 0.3 * u[4]
            c.update(ps=ps, ptop=ptop, tsfc=tsfc, lapse=lapse, trop=trop,
                     warm=warm, qsfc=qsfc, inv=inv, invd=invd, emis=emis,
                     alb=alb, mu=mu, liq=liq, ice=ice, k=k)
            cols.append(c)
    ncol = len(cols)
    shape = (ncol, NLAY)
    play = np.empty(shape)
    plev = np.empty((ncol, NLAY + 1))
    tlay = np.empty(shape)
    tlev = np.empty((ncol, NLAY + 1))
    tsfc = np.empty(ncol)
    qv = np.empty(shape)
    o3 = np.empty(shape)
    clwp = np.zeros(shape)
    ciwp = np.zeros(shape)
    reliq = np.full(shape, 10.0)
    dgice = np.full(shape, 60.0)
    cldfra = np.zeros(shape)
    emis = np.empty(ncol)
    alb = np.empty(ncol)
    mu0 = np.empty(ncol)
    for i, c in enumerate(cols):
        lev = _levels(c["ps"], c["ptop"], NLAY, 0.75 + 0.1 * (i % 3))
        plev[i] = lev
        play[i] = 0.5 * (lev[:-1] + lev[1:])
        tl = _temperature(play[i], c["ps"], c["tsfc"], c["lapse"], c["trop"],
                          c["warm"], c["inv"], c["invd"])
        tlay[i] = tl
        tv = _temperature(lev, c["ps"], c["tsfc"], c["lapse"], c["trop"],
                          c["warm"], c["inv"], c["invd"])
        tlev[i] = tv
        tsfc[i] = np.clip(c["tsfc"] + (-3.0 if c["inv"] else 1.0), 160.5,
                          354.5)
        qv[i] = c["qsfc"] * (play[i] / c["ps"]) ** 3.2 + 3.0e-6
        o3[i] = np.interp(np.log(play[i]), o3_logp, o3_vmr)
        emis[i], alb[i], mu0[i] = c["emis"], c["alb"], c["mu"]
        for kind, spec in (("liq", c["liq"]), ("ice", c["ice"])):
            if spec is None:
                continue
            pbot_ = max(spec[0], spec[1])
            ptop_ = min(spec[0], spec[1])
            inside = (play[i] <= pbot_) & (play[i] >= ptop_)
            n = max(int(inside.sum()), 1)
            frac = np.where(inside, 0.3 + 0.7 * rng.random(NLAY), 0.0)
            path = np.where(inside, spec[2] / n * (0.5 + rng.random(NLAY)),
                            0.0)
            if kind == "liq":
                clwp[i] += path
                reliq[i] = np.where(inside, 4.0 + 12.0 * rng.random(NLAY),
                                    reliq[i])
            else:
                ciwp[i] += path
                dgice[i] = np.where(inside, 20.0 + 140.0 * rng.random(NLAY),
                                    dgice[i])
            cldfra[i] = np.maximum(cldfra[i], frac)

    # ---- edge cases (the last regime block), one property per column
    e0 = REGIMES.index("edge-cases") * NCOL_PER_REGIME
    # (0) layer temperatures exactly at the table ends, surface at the max
    tlay[e0 + 0, 0] = 355.0
    tlev[e0 + 0, 0] = 355.0
    tsfc[e0 + 0] = 355.0
    tlay[e0 + 0, 45] = 160.0
    tlev[e0 + 0, 46] = 160.0
    # (1) a layer exactly at the troposphere/stratosphere split pressure and
    #     its float32 neighbours, plus a top layer at the table minimum
    #     (the neighbours go to columns 5 and 6, in the layer whose
    #     interfaces bracket the split, so every play stays inside its layer)
    for col, value in ((e0 + 1, p_trop32),
                       (e0 + 5, np.nextafter(p_trop32, F32(1e6))),
                       (e0 + 6, np.nextafter(p_trop32, F32(0)))):
        j = int(np.flatnonzero((plev[col, :-1] > float(value))
                               & (plev[col, 1:] < float(value)))[0])
        play[col, j] = float(value)
    plev[e0 + 1, -1] = 0.5
    play[e0 + 1, -1] = float(p_min32)
    # (2) completely dry column, thick liquid at the table radius bounds
    qv[e0 + 2] = 0.0
    clwp[e0 + 2, 2:8] = 500.0
    reliq[e0 + 2, 2:5] = float(cld.radliq_lwr)
    reliq[e0 + 2, 5:8] = float(cld.radliq_upr)
    cldfra[e0 + 2, 2:8] = 1.0
    # (3) ice at the diameter bounds, tiny water paths
    ciwp[e0 + 3, 30:36] = [1e-30, 1e-12, 1e-6, 1e-3, 200.0, 2000.0]
    dgice[e0 + 3, 30:33] = float(cld.diamice_lwr)
    dgice[e0 + 3, 33:36] = float(cld.diamice_upr)
    cldfra[e0 + 3, 30:36] = 1.0
    # (4) very moist hot surface, emissivity 1, albedo 0, sun overhead
    qv[e0 + 4] = 0.03 * (play[e0 + 4] / play[e0 + 4, 0]) ** 3.0 + 3e-6
    tlay[e0 + 4, :3] = 320.0
    emis[e0 + 4], alb[e0 + 4], mu0[e0 + 4] = 1.0, 0.0, 1.0
    # (5) sun below the solver's mu floor; albedo 1
    mu0[e0 + 5], alb[e0 + 5] = 1.0e-5, 1.0
    # (6) sun exactly at the floor sqrt(epsilon); full overcast all layers
    mu0[e0 + 6] = float(np.sqrt(F32(np.finfo(F32).eps)))
    clwp[e0 + 6, :20] = 15.0
    cldfra[e0 + 6, :] = 1.0
    ciwp[e0 + 6, 20:40] = 5.0
    # (7) very thin layers near the surface (dp of a few Pa)
    plev[e0 + 7, 1:6] = plev[e0 + 7, 0] - np.array([2.0, 5.0, 9.0, 20.0, 60.0])
    play[e0 + 7, :5] = 0.5 * (plev[e0 + 7, :5] + plev[e0 + 7, 1:6])
    # (8) isothermal column at 250 K, zero ozone
    tlay[e0 + 8] = 250.0
    tlev[e0 + 8] = 250.0
    tsfc[e0 + 8] = 250.0
    o3[e0 + 8] = 0.0

    vmr = np.zeros((ncol, NLAY, len(GAS_NAMES)), dtype=F32)
    h2o_scale = F32(0.028964 / 0.018016)
    vmr[:, :, GAS_NAMES.index("h2o")] = qv.astype(F32) * h2o_scale
    vmr[:, :, GAS_NAMES.index("o3")] = o3.astype(F32)
    for gas, value in trace.items():
        if gas in GAS_NAMES:
            vmr[:, :, GAS_NAMES.index(gas)] = F32(value)
    # Vary the well-mixed gases a little across columns.
    vmr[:, :, GAS_NAMES.index("co2")] *= (
        1.0 + 0.2 * rng.random((ncol, 1))).astype(F32)
    vmr[:, :, GAS_NAMES.index("ch4")] *= (
        1.0 + 0.3 * rng.random((ncol, 1))).astype(F32)
    vmr[:, :, GAS_NAMES.index("no2")] = F32(2e-9)

    out = dict(
        play=play.astype(F32), plev=plev.astype(F32), tlay=tlay.astype(F32),
        tlev=tlev.astype(F32), tsfc=tsfc.astype(F32), vmr=vmr,
        clwp=clwp.astype(F32), ciwp=ciwp.astype(F32),
        reliq=reliq.astype(F32), dgice=dgice.astype(F32),
        cldfra=cldfra.astype(F32), emis=emis.astype(F32),
        alb=alb.astype(F32), mu0=mu0.astype(F32),
        regime=np.array([c["regime"] for c in cols]))
    # Play must sit strictly inside its layer for every column.
    assert np.all(out["plev"][:, :-1] > out["play"]), "play above its base"
    assert np.all(out["play"] > out["plev"][:, 1:]), "play below its top"
    return out


def add_device_inputs(cols):
    """McICA masks (WOOF's generator) and the TOA source: shared inputs."""
    import cupy as cp
    from gpuwm.core import rrtmgp as R

    lw = R.load_gas_tables("lw")
    sw = R.load_gas_tables("sw")
    play = cp.asarray(cols["play"])
    cf = cp.asarray(cols["cldfra"])
    cols["mask_lw"] = cp.asnumpy(R.mcica_cloud_masks(
        play, cf, lw.ngpt, R.MCICA_PERMUTESEED_LW))
    day = np.flatnonzero(cols["mu0"] > 0).astype(np.int32)
    cols["sw_cols"] = day
    cols["mask_sw"] = cp.asnumpy(R.mcica_cloud_masks(
        play[cp.asarray(day)], cf[cp.asarray(day)], sw.ngpt,
        R.MCICA_PERMUTESEED_SW))
    solar = R._solar_source_device(sw)
    solcon = 1361.0
    inc = solar * F32(solcon / float(np.sum(np.asarray(
        sw.solar_source, dtype=np.float64))))
    cols["inc_gpt"] = cp.asnumpy(inc).astype(F32)
    return cols


def write_fortran_input(cols, path):
    def f(a):
        return np.asarray(a).tobytes(order="F")
    ncol, nlay = cols["play"].shape
    nsw = len(cols["sw_cols"])
    with open(path, "wb") as fh:
        fh.write(np.array([ncol, nlay, len(GAS_NAMES), nsw],
                          np.int32).tobytes())
        for name in GAS_NAMES:
            fh.write(name.ljust(32).encode("ascii"))
        for key in ("play", "plev", "tlay", "tlev", "tsfc", "vmr",
                    "clwp", "ciwp", "reliq", "dgice"):
            fh.write(f(cols[key].astype(F32)))
        fh.write(np.array([cols["mask_lw"].shape[2],
                           cols["mask_sw"].shape[2]], np.int32).tobytes())
        fh.write(f(cols["mask_lw"].astype(F32)))
        fh.write(f(cols["emis"].astype(F32)))
        fh.write(f((cols["sw_cols"] + 1).astype(np.int32)))
        fh.write(f(cols["mask_sw"].astype(F32)))
        day = cols["sw_cols"]
        fh.write(f(cols["mu0"][day].astype(F32)))
        fh.write(f(cols["alb"][day].astype(F32)))
        fh.write(f(cols["inc_gpt"].astype(F32)))


def read_fortran_output(cols, path, nband_lw=16, nband_sw=14):
    ncol, nlay = cols["play"].shape
    ngl = cols["mask_lw"].shape[2]
    ngs = cols["mask_sw"].shape[2]
    nsw = len(cols["sw_cols"])
    raw = np.fromfile(path, dtype=F32)
    pos = 0
    out = {}

    def take(name, shape):
        nonlocal pos
        n = int(np.prod(shape))
        out[name] = raw[pos:pos + n].reshape(shape, order="F")
        pos += n
    take("lw_gas_tau", (ncol, nlay, ngl))
    take("lw_lay_src", (ncol, nlay, ngl))
    take("lw_lev_src", (ncol, nlay + 1, ngl))
    take("lw_sfc_src", (ncol, ngl))
    take("lw_cld_tau", (ncol, nlay, nband_lw))
    take("lw_cld_ssa", (ncol, nlay, nband_lw))
    take("lw_cld_g", (ncol, nlay, nband_lw))
    take("lw_tau", (ncol, nlay, ngl))
    take("lw_flux_up", (ncol, nlay + 1))
    take("lw_flux_dn", (ncol, nlay + 1))
    take("sw_gas_tau", (nsw, nlay, ngs))
    take("sw_gas_ssa", (nsw, nlay, ngs))
    take("sw_toa_src", (ngs,))
    take("sw_cld_tau", (nsw, nlay, nband_sw))
    take("sw_cld_ssa", (nsw, nlay, nband_sw))
    take("sw_cld_g", (nsw, nlay, nband_sw))
    take("sw_inc_tau", (nsw, nlay, ngs))
    take("sw_inc_ssa", (nsw, nlay, ngs))
    take("sw_inc_g", (nsw, nlay, ngs))
    take("sw_tau", (nsw, nlay, ngs))
    take("sw_ssa", (nsw, nlay, ngs))
    take("sw_g", (nsw, nlay, ngs))
    take("sw_flux_up", (nsw, nlay + 1))
    take("sw_flux_dn", (nsw, nlay + 1))
    take("sw_flux_dir", (nsw, nlay + 1))
    if pos != raw.size:
        raise ValueError(f"oracle output has {raw.size} words, read {pos}")
    return out


# ---------------------------------------------------------------- WOOF
def run_woof(cols):
    import cupy as cp
    from gpuwm.core import rrtmgp as R

    lw = R.load_gas_tables("lw")
    sw = R.load_gas_tables("sw")
    clw = R.load_cloud_tables("lw")
    csw = R.load_cloud_tables("sw")
    out = {}

    def vmr_cube(rows):
        v = np.zeros((len(rows), nlay, len(GAS_NAMES) + 1), F32)
        for g, name in enumerate(GAS_NAMES):
            v[:, :, lw.gas_index[name]] = cols["vmr"][rows, :, g]
        return cp.asarray(v)

    allc = np.arange(cols["play"].shape[0])
    nlay = cols["play"].shape[1]
    day = cols["sw_cols"]
    D = {k: cp.asarray(cols[k]) for k in (
        "play", "plev", "tlay", "tlev", "tsfc", "clwp", "ciwp", "reliq",
        "dgice", "emis", "alb", "mu0")}
    mask_lw = cp.asarray(cols["mask_lw"])
    mask_sw = cp.asarray(cols["mask_sw"])
    inc_gpt = cp.asarray(cols["inc_gpt"])

    # ------------------------------------------------------------ LW
    vmr = vmr_cube(allc)
    md = R._interpolation_metadata(lw, D["play"], D["tlay"])
    gas = R._gas_optics(lw, D["play"], D["plev"], D["tlay"], vmr,
                        metadata=md, validate=True, zero_g_sentinel=True)
    out["lw_gas_tau"] = gas.tau
    src = R._planck_sources(lw, D["play"], D["plev"], D["tlay"], D["tlev"],
                            D["tsfc"], vmr, metadata=md, validate=True)
    out["lw_lay_src"], out["lw_lev_src"], out["lw_sfc_src"] = (
        src.lay_source, src.lev_source, src.sfc_source)
    cld = R.cloud_optics(clw, D["clwp"], D["ciwp"], D["reliq"], D["dgice"])
    out["lw_cld_tau"], out["lw_cld_ssa"], out["lw_cld_g"] = (
        cld.tau, cld.ssa, cld.g)
    fin = R._finalize_cloud_optics(lw, gas, cld, cloud_mask=mask_lw)
    out["lw_tau"] = fin.tau
    emis_b = cp.broadcast_to(D["emis"][:, None], (len(allc), lw.nband))
    emis = R._expand_band_to_gpoint(emis_b, lw, "surface emissivity")
    staged = R.lw_rte(fin.tau, src.lay_source, src.lev_source,
                      src.sfc_source, emis, top_at_1=False)
    out["lw_flux_up@staged"], out["lw_flux_dn@staged"] = (
        staged.flux_up, staged.flux_dn)
    planck = R.PlanckInputs(tables=lw, play=D["play"], plev=D["plev"],
                            tlay=D["tlay"], tlev=D["tlev"], tsfc=D["tsfc"],
                            vmr=vmr, metadata=md)
    fused_cloud = R.FusedCloudOptics(tables=lw, cloud=cld, mask=mask_lw)
    for arm, fold in (("fused", None), ("fused-fold0", "0")):
        if fold is not None:
            os.environ["GPUWM_RTE_FOLD"] = fold
        try:
            flux = R._lw_rte(gas.tau, None, None, None, emis, top_at_1=False,
                             out=None, incident_out=None,
                             fused_cloud=fused_cloud, planck=planck)
        finally:
            os.environ.pop("GPUWM_RTE_FOLD", None)
        out[f"lw_flux_up@{arm}"], out[f"lw_flux_dn@{arm}"] = (
            flux.flux_up, flux.flux_dn)

    # ------------------------------------------------------------ SW
    dd = cp.asarray(day)
    vmr = vmr_cube(day)
    S = {k: D[k][dd] for k in ("play", "plev", "tlay", "clwp", "ciwp",
                               "reliq", "dgice", "alb", "mu0")}
    md = R._interpolation_metadata(sw, S["play"], S["tlay"])
    gas = R._gas_optics(sw, S["play"], S["plev"], S["tlay"], vmr,
                        metadata=md, validate=True, zero_g_sentinel=True)
    out["sw_gas_tau"], out["sw_gas_ssa"] = gas.tau, gas.ssa
    out["sw_toa_src"] = R._solar_source_device(sw)
    cld = R.cloud_optics(csw, S["clwp"], S["ciwp"], S["reliq"], S["dgice"])
    out["sw_cld_tau"], out["sw_cld_ssa"], out["sw_cld_g"] = (
        cld.tau, cld.ssa, cld.g)
    fin = R._finalize_cloud_optics(sw, gas, cld, cloud_mask=mask_sw)
    out["sw_tau"], out["sw_ssa"], out["sw_g"] = fin.tau, fin.ssa, fin.g
    albedo, inc, mu_chunk = R._prepare_sw_inputs(
        S["alb"], inc_gpt, S["mu0"], nlay, xp=cp)
    staged = R.sw_rte(fin.tau, fin.ssa, fin.g, mu_chunk, albedo, albedo,
                      inc, top_at_1=False)
    for name in ("up", "dn", "dir"):
        out[f"sw_flux_{name}@staged"] = getattr(staged, f"flux_{name}")
    fused_cloud = R.FusedCloudOptics(tables=sw, cloud=cld, mask=mask_sw)
    for arm, fold in (("fused", None), ("fused-fold0", "0")):
        if fold is not None:
            os.environ["GPUWM_RTE_FOLD"] = fold
        try:
            flux = R._sw_rte(gas.tau, gas.ssa, None, mu_chunk, albedo,
                             albedo, inc, top_at_1=False, out=None,
                             fused_cloud=fused_cloud)
        finally:
            os.environ.pop("GPUWM_RTE_FOLD", None)
        for name in ("up", "dn", "dir"):
            out[f"sw_flux_{name}@{arm}"] = getattr(flux, f"flux_{name}")
    return {k: cp.asnumpy(v).astype(F32) for k, v in out.items()}


def bench(cols, ncolumns, repeats):
    """Time WOOF's production RRTMGP stages on the oracle columns, tiled.

    Production chunking (DEFAULT_COLUMN_CHUNK columns) and the workspace
    path's composition (gas optics, band clouds, the McICA mask as input,
    the fused solvers with in-kernel Planck); CUDA events around each
    chunk's stages, median over `repeats` passes after one warm-up pass.
    Runs against any tree: where PlanckInputs has no `plev` field it is
    timed without one.
    """
    import cupy as cp
    from gpuwm.config import DEFAULT_COLUMN_CHUNK
    from gpuwm.core import rrtmgp as R

    lw, sw = R.load_gas_tables("lw"), R.load_gas_tables("sw")
    clw, csw = R.load_cloud_tables("lw"), R.load_cloud_tables("sw")
    nlay = cols["play"].shape[1]
    v = np.zeros((cols["play"].shape[0], nlay, len(GAS_NAMES) + 1), F32)
    for g, name in enumerate(GAS_NAMES):
        v[:, :, lw.gas_index[name]] = cols["vmr"][:, :, g]
    cols = dict(cols, vmr_cube=v)

    def tiled(rows, keys):
        reps = -(-ncolumns // len(rows))
        return {k: np.concatenate([cols[k][rows]] * reps)[:ncolumns]
                for k in keys}

    every = np.arange(cols["play"].shape[0])
    L = tiled(every, ("play", "plev", "tlay", "tlev", "tsfc", "vmr_cube",
                      "clwp", "ciwp", "reliq", "dgice", "mask_lw", "emis"))
    day = cols["sw_cols"]
    S = tiled(day, ("play", "plev", "tlay", "vmr_cube", "clwp", "ciwp",
                    "reliq", "dgice", "alb", "mu0"))
    reps = -(-ncolumns // len(day))
    S["mask_sw"] = np.concatenate([cols["mask_sw"]] * reps)[:ncolumns]
    inc_gpt = cp.asarray(cols["inc_gpt"])
    has_plev = "plev" in R.PlanckInputs._fields
    chunk = DEFAULT_COLUMN_CHUNK
    times = {"lw": [], "sw": []}
    for rep in range(repeats + 1):
        for band in ("lw", "sw"):
            start, stop = cp.cuda.Event(), cp.cuda.Event()
            elapsed = 0.0
            for c0 in range(0, ncolumns, chunk):
                sl = slice(c0, min(c0 + chunk, ncolumns))
                src = L if band == "lw" else S
                a = {k: cp.asarray(x[sl]) for k, x in src.items()}
                n = a["play"].shape[0]
                tables = lw if band == "lw" else sw
                ctab = clw if band == "lw" else csw
                if band == "lw":
                    emis_g = cp.ascontiguousarray(cp.broadcast_to(
                        a["emis"][:, None], (n, lw.ngpt)))
                else:
                    albedo, inc, mu_chunk = R._prepare_sw_inputs(
                        a["alb"], inc_gpt, a["mu0"], nlay, xp=cp)
                cp.cuda.Device().synchronize()
                start.record()
                md = R._interpolation_metadata(tables, a["play"], a["tlay"],
                                               validate=False)
                gas = R._gas_optics(tables, a["play"], a["plev"], a["tlay"],
                                    a["vmr_cube"], metadata=md,
                                    validate=False, zero_g_sentinel=True,
                                    compute_col_dry=False)
                cld = R.cloud_optics(ctab, a["clwp"], a["ciwp"], a["reliq"],
                                     a["dgice"])
                if band == "lw":
                    kw = dict(tables=lw, play=a["play"], tlay=a["tlay"],
                              tlev=a["tlev"], tsfc=a["tsfc"],
                              vmr=a["vmr_cube"], metadata=md)
                    if has_plev:
                        kw["plev"] = a["plev"]
                    R._lw_rte(gas.tau, None, None, None, emis_g,
                              top_at_1=False, out=None, incident_out=None,
                              fused_cloud=R.FusedCloudOptics(
                                  tables=lw, cloud=cld, mask=a["mask_lw"]),
                              planck=R.PlanckInputs(**kw))
                else:
                    R._sw_rte(gas.tau, gas.ssa, None, mu_chunk, albedo,
                              albedo, inc, top_at_1=False, out=None,
                              fused_cloud=R.FusedCloudOptics(
                                  tables=sw, cloud=cld, mask=a["mask_sw"]))
                stop.record()
                stop.synchronize()
                elapsed += cp.cuda.get_elapsed_time(start, stop)
            if rep:
                times[band].append(elapsed)
    for band in ("lw", "sw"):
        print(f"{band}: median {np.median(times[band]):.1f} ms over "
              f"{ncolumns} columns x {nlay} layers (passes "
              + ", ".join(f"{t:.1f}" for t in times[band]) + ")")


# ---------------------------------------------------------------- compare
def _ordered(a):
    i = np.asarray(a, F32).view(np.int32).astype(np.int64)
    return np.where(i < 0, -(i & 0x7FFFFFFF), i)


def ulp(a, b):
    a = np.asarray(a, F32)
    b = np.asarray(b, F32)
    d = np.abs(_ordered(a) - _ordered(b))
    both_nan = np.isnan(a) & np.isnan(b)
    return np.where(both_nan, 0, d)


def compare(cols, ref, woof):
    rows = []
    day = cols["sw_cols"]
    for key in sorted(woof):
        base = key.split("@")[0]
        if base not in ref:
            continue
        a, b = woof[key], ref[base]
        if a.shape != b.shape:
            raise ValueError(f"{key}: WOOF {a.shape} vs oracle {b.shape}")
        d = ulp(a, b)
        n_bad = int(np.count_nonzero(d))
        row = {"field": key, "n": int(d.size), "mismatched": n_bad,
               "max_ulp": int(d.max()) if d.size else 0,
               "max_abs": float(np.max(np.abs(a.astype(np.float64)
                                              - b.astype(np.float64))))}
        if n_bad:
            idx = np.unravel_index(int(np.argmax(d)), d.shape)
            row["worst_index"] = [int(x) for x in idx]
            row["worst"] = {"woof": float(a[idx]), "oracle": float(b[idx])}
            if a.ndim >= 2:
                # Global column numbers (SW fields hold daylit columns only).
                glob = day if base.startswith("sw_") \
                    else np.arange(len(cols["regime"]))
                bad = np.flatnonzero(d.reshape(d.shape[0], -1).max(1))
                row["worst_regime"] = str(cols["regime"][glob[idx[0]]])
                row["columns"] = [int(glob[i]) for i in bad]
        rows.append(row)
    return rows


#: Inputs the committed fixture carries (tests/data/oracles/rrtmgp).
FIXTURE_INPUTS = ("play", "plev", "tlay", "tlev", "tsfc", "vmr", "clwp",
                  "ciwp", "reliq", "dgice", "emis", "alb", "mu0", "regime",
                  "sw_cols", "inc_gpt")


def column_digests(a):
    """sha256 of each column's float32 bytes: (ncol, 32) uint8.  A 1-D
    field (the TOA source) is one column."""
    import hashlib
    a = np.ascontiguousarray(np.asarray(a, F32))
    rows = a[None] if a.ndim == 1 else a
    return np.array([np.frombuffer(hashlib.sha256(r.tobytes()).digest(),
                                   np.uint8) for r in rows])


def write_fixture(cols, ref, ref_fixed, path):
    """Inputs, McICA masks (bit-packed) and per-column digests of every
    oracle output, stock and with the boundary fixes."""
    out = {k: cols[k] for k in FIXTURE_INPUTS}
    for k in ("mask_lw", "mask_sw"):
        out[k + "_packed"] = np.packbits(cols[k].astype(bool), axis=-1)
        out[k + "_shape"] = np.array(cols[k].shape)
    for tag, outputs in (("stock", ref), ("fixed", ref_fixed)):
        for name, value in outputs.items():
            out[f"digest_{tag}__{name}"] = column_digests(value)
    np.savez_compressed(path, **out)


def load_fixture(path):
    """(cols, {"stock": {field: digests}, "fixed": {...}})."""
    raw = dict(np.load(path))
    cols = {k: raw[k] for k in FIXTURE_INPUTS}
    for k in ("mask_lw", "mask_sw"):
        shape = tuple(raw[k + "_shape"])
        cols[k] = np.unpackbits(raw[k + "_packed"], axis=-1,
                                count=shape[-1]).astype(bool)
    digests = {"stock": {}, "fixed": {}}
    for key, value in raw.items():
        if key.startswith("digest_"):
            tag, name = key[len("digest_"):].split("__", 1)
            digests[tag][name] = value
    return cols, digests


def digest_mismatches(woof, digests):
    """{woof field: [column numbers whose digest differs]} for every WOOF
    output that has an oracle counterpart; SW fields are mapped to global
    column numbers by the caller's sw_cols."""
    bad = {}
    for key, value in woof.items():
        base = key.split("@")[0]
        if base not in digests:
            continue
        mine = column_digests(value)
        theirs = digests[base]
        if mine.shape != theirs.shape:
            raise ValueError(f"{key}: {mine.shape} vs {theirs.shape}")
        rows = np.flatnonzero(np.any(mine != theirs, axis=1))
        bad[key] = [int(r) for r in rows]
    return bad


def _print(rows):
    width = max(len(r["field"]) for r in rows)
    for r in rows:
        print(f"{r['field']:<{width}}  n={r['n']:>9}  "
              f"bad={r['mismatched']:>9}  max_ulp={r['max_ulp']:>10}  "
              f"max_abs={r['max_abs']:.3e}  "
              f"{r.get('worst_regime', '')} {r.get('columns', '')}")
    print(f"overall max ULP {max(r['max_ulp'] for r in rows)}")


def main(argv=None):
    global NCOL_PER_REGIME, NLAY
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--npz", required=True)
    m.add_argument("--fortran-in", required=True)
    m.add_argument("--seed", type=int, default=20261007)
    m.add_argument("--per-regime", type=int, default=NCOL_PER_REGIME,
                   help="columns per regime (at least 9: the edge cases)")
    m.add_argument("--nlay", type=int, default=NLAY,
                   help="layers per column (at least 60)")
    b = sub.add_parser("bench")
    b.add_argument("--npz", required=True)
    b.add_argument("--columns", type=int, default=82944)
    b.add_argument("--repeats", type=int, default=5)
    w = sub.add_parser("woof")
    w.add_argument("--npz", required=True)
    w.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--npz", required=True)
    c.add_argument("--ref", required=True)
    c.add_argument("--woof", required=True)
    c.add_argument("--json", required=True)
    fx = sub.add_parser("fixture")
    fx.add_argument("--npz", required=True)
    fx.add_argument("--ref", required=True)
    fx.add_argument("--ref-fixed", required=True)
    fx.add_argument("--out", required=True)
    r = sub.add_parser("refdiff")
    r.add_argument("--npz", required=True)
    r.add_argument("--ref", required=True)
    r.add_argument("--other", required=True)
    r.add_argument("--json", required=True)
    args = p.parse_args(argv)
    if args.cmd == "make":
        NCOL_PER_REGIME, NLAY = max(9, args.per_regime), max(60, args.nlay)
        cols = add_device_inputs(make_columns(args.seed))
        np.savez(args.npz, **cols)
        write_fortran_input(cols, args.fortran_in)
        print(f"{cols['play'].shape[0]} columns, {len(cols['sw_cols'])} "
              f"daylit, {NLAY} layers")
    elif args.cmd == "woof":
        cols = dict(np.load(args.npz))
        from gpuwm import wrf_exact
        result = run_woof(cols)
        result["_strict"] = np.array(bool(wrf_exact.ENABLED))
        np.savez(args.out, **result)
    elif args.cmd == "bench":
        bench(dict(np.load(args.npz)), args.columns, args.repeats)
    elif args.cmd == "fixture":
        cols = dict(np.load(args.npz))
        write_fixture(cols, read_fortran_output(cols, args.ref),
                      read_fortran_output(cols, args.ref_fixed), args.out)
    elif args.cmd == "refdiff":
        cols = dict(np.load(args.npz))
        ref = read_fortran_output(cols, args.ref)
        other = read_fortran_output(cols, args.other)
        rows = compare(cols, ref, other)
        Path(args.json).write_text(json.dumps({"rows": rows}, indent=1)
                                   + "\n")
        _print(rows)
    else:
        cols = dict(np.load(args.npz))
        ref = read_fortran_output(cols, args.ref)
        woof = dict(np.load(args.woof))
        strict = bool(woof.pop("_strict"))
        rows = compare(cols, ref, woof)
        doc = {"strict": strict, "rows": rows}
        Path(args.json).write_text(json.dumps(doc, indent=1) + "\n")
        _print(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
