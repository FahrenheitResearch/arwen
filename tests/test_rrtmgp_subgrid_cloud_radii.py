"""MYNN subgrid cloud reaches both RRTMG engines at a droplet size.

The breakage: with MYNN's subgrid cloud on (``icloud_bl = 1``, the default)
a 750 m coastal-basin product nest put 549 W m-2 of shortwave on the ground
over land at solar noon on a hot late-August day under RTE+RRTMGP against
701 under legacy RRTMG (556 against 710 over land with no resolved cloud
water), and ran colder than legacy at all 13 ASOS stations.  The subgrid
cloud water lands where the microphysics has no cloud of its own, so its
radius there is the scheme's no-cloud background (2.49 um liquid, 4.99 um
ice under Thompson).  WRF's RRTMG wrappers size such a layer at 7.5 um over
land, 10.5 um over water and from the ice temperature table; the
RTE+RRTMGP adapter radiated the background, three times the liquid optical
depth for the same water.

WRF finds such a layer by its radius (at or below 2.5 um liquid, 5 um ice),
and that test misses NSSL's background (2.51 um liquid, 10.01 um ice), a
trace of Thompson liquid below the merge's 1e-6 kg/kg, which Thompson sizes
at its 2.51 um floor, and a trace of Thompson ice sized above 5 um.  MYNN's
water in those layers was radiated at the trace's size on both engines.
Both engines now size every layer the merge gives water, and only those,
and keep a scheme's own radius for its resolved cloud, NSSL's at its 2.51
um floor included.

The CPU tests hold the RTE+RRTMGP rule to the legacy engine's WRF
transcription and carry one hot afternoon column through both engines:
clear sky agrees within a few W m-2, and the subgrid layers agree once both
size them alike.  The GPU tests drive both production adapters.
"""
import sys
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.core.mynn_radiation import (
    merge_mynn_bl_clouds,
    mynn_bl_cloud_supplied,
)
from gpuwm.core.rrtmg_legacy import unsized_mynn_radii

F = np.float32
#: NSSL's cloud radius floor and no-cloud value, formed as its writer forms
#: it (gpuwm/core/kernels/nssl2_diagnostics.cu, metres times 1e6 in FP32).
NSSL_FLOOR_UM = np.float32(np.float32(2.51e-6) * np.float32(1.0e6))

#: The product's 59-level eta ladder and model top.
_ETA = np.array([
    1, 0.993814707, 0.985950649, 0.976014256, 0.963557541,
    0.948093116, 0.929123759, 0.90619123, 0.87894237, 0.847207963,
    0.811077714, 0.770949006, 0.727525413, 0.684030771, 0.642961025,
    0.604180932, 0.567562938, 0.532986403, 0.500337601, 0.469508916,
    0.440399021, 0.412912011, 0.386957437, 0.362449884, 0.339308649,
    0.317457527, 0.296824664, 0.277342081, 0.258945674, 0.241574913,
    0.225172549, 0.2096847, 0.195060253, 0.181251153, 0.168211967,
    0.155899644, 0.144273847, 0.133296132, 0.122930467, 0.113142714,
    0.103900604, 0.095173724, 0.0869334266, 0.0791524947, 0.0718053728,
    0.0648678541, 0.0583171472, 0.0521316081, 0.0462909527, 0.0407758839,
    0.0355683193, 0.030651059, 0.026007941, 0.0216237046, 0.0174838807,
    0.0135748768, 0.00988376327, 0.00639845803, 0.00310745789, 0.0])
_P_TOP = 5000.0


def _legacy_radii(effc, effi, cldfra, tlay, xland, supplied_liquid=None,
                  supplied_ice=None):
    """The legacy engine's radii, in microns: its adapter's MYNN step, then
    its WRF transcription."""
    from gpuwm.core import rrtmg_legacy_prep as prep

    zeros = np.zeros_like(effc)
    re_cloud = (effc * F(1.0e-6)).astype(F)
    re_ice = (effi * F(1.0e-6)).astype(F)
    if supplied_liquid is not None:
        re_cloud, re_ice = unsized_mynn_radii(
            re_cloud, re_ice, supplied_liquid, supplied_ice, ice_rule=True)
    _inflg, _iceflg, rec, rei, _res, _qs, _qi = prep._effective_radii_b(
        np, 1, 1, 1, 1, tlay, cldfra, np.asarray(xland, F),
        re_cloud, re_ice, np.full_like(effc, F(9.99e-6)), zeros, zeros,
        zeros)
    return rec, rei


def test_cloudy_background_radii_are_the_wrf_rrtmg_radii(monkeypatch):
    """Every cloudy layer at a background radius, or given MYNN water,
    takes WRF's radius, and both engines give the same one.

    Checked against the legacy engine (its MYNN step, then
    ``rrtmg_legacy_prep._effective_radii_b``, the transcription of
    module_ra_rrtmg_sw.F:10779-10812) over land, water and WRF's
    neither-flag (xland 1.5), clear and cloudy layers, with and without
    MYNN water, the Thompson and NSSL backgrounds and floors, radii either
    side of WRF's 2.5 and 5 um bounds and computed radii, at temperatures
    across the whole ice table.  A replaced radius must equal the legacy
    engine's bit for bit; a kept one is the scheme's own.
    """
    monkeypatch.setitem(sys.modules, "cupy", np)
    from gpuwm.core.rrtmgp import cloudy_background_radii

    rng = np.random.default_rng(20260927)
    xland = np.array([1.0, 2.0, 1.5], F)
    ncol, nlay = xland.size, 600
    tlay = rng.uniform(165.0, 305.0, (ncol, nlay)).astype(F)
    cldfra = np.where(rng.random((ncol, nlay)) < 0.6,
                      rng.uniform(0.002, 1.0, (ncol, nlay)), 0.0).astype(F)
    effc = rng.choice(F([2.49, 2.4, NSSL_FLOOR_UM, 2.6, 4.0, 8.0, 14.0]),
                      (ncol, nlay))
    effi = rng.choice(F([4.99, 2.51, 4.9, 5.2, 10.01, 20.0, 60.0]),
                      (ncol, nlay))
    # MYNN water on land and water only: gpuwm never writes xland 1.5.
    supplied_liquid = rng.random((ncol, nlay)) < 0.3
    supplied_ice = rng.random((ncol, nlay)) < 0.3
    supplied_liquid[2] = supplied_ice[2] = False

    got_c, got_i = cloudy_background_radii(
        effc, effi, cldfra, tlay, xland, scheme="thompson",
        supplied_liquid=supplied_liquid, supplied_ice=supplied_ice)
    want_c, want_i = _legacy_radii(effc, effi, cldfra, tlay, xland,
                                   supplied_liquid, supplied_ice)

    cloudy = cldfra > 0
    replaced_c = got_c != effc
    replaced_i = got_i != effi
    # The layers that change are exactly the cloudy background ones and the
    # cloudy ones MYNN gave water.
    np.testing.assert_array_equal(
        replaced_c, cloudy & ((effc <= F(2.5)) | supplied_liquid)
        & (xland != F(1.5))[:, None])
    np.testing.assert_array_equal(
        replaced_i, cloudy & ((effi <= F(5.0)) | supplied_ice))
    assert (replaced_c & (effc > F(2.5))).any()
    assert (replaced_i & (effi > F(5.0))).any()
    np.testing.assert_array_equal(got_c[0][replaced_c[0]], F(7.5))
    np.testing.assert_array_equal(got_c[1][replaced_c[1]], F(10.5))
    # Bit for bit where replaced, and WRF's own floors on every other
    # cloudy layer.  A clear layer's radius sizes no water.
    np.testing.assert_array_equal(got_c[replaced_c], want_c[replaced_c])
    np.testing.assert_array_equal(got_i[replaced_i], want_i[replaced_i])
    np.testing.assert_allclose(np.maximum(got_c, F(2.5))[cloudy],
                               want_c[cloudy], rtol=1e-6)
    np.testing.assert_allclose(np.maximum(got_i, F(5.0))[cloudy],
                               want_i[cloudy], rtol=1e-6)


def test_resolved_cloud_at_a_schemes_floor_keeps_its_radius(monkeypatch):
    """Without MYNN's water, a cloudy layer keeps every radius above WRF's
    2.5 um and 5 um bounds, NSSL's 2.51 um floor included.

    NSSL writes 2.51 um where it has no cloud and also for every resolved
    droplet population whose computed radius falls below that floor (0.005
    to 0.02 g/kg at 300 to 1000 drops per cc).  Sized at 7.5 um, a
    three-layer 0.02 g/kg deck near 900 hPa let about 815 W m-2 through
    where it passes about 523 at 2.51 um, so more droplets made a thinner
    cloud.
    """
    monkeypatch.setitem(sys.modules, "cupy", np)
    from gpuwm.core.rrtmgp import cloudy_background_radii

    floor = NSSL_FLOOR_UM
    # Layers: resolved at the floor; the Thompson background; resolved at a
    # computed size; a clear layer at the floor.
    effc = np.array([[floor, 2.49, 9.0, floor]], F)
    effi = np.array([[10.01, 4.99, 30.0, 10.01]], F)
    cldfra = np.array([[1.0, 1.0, 1.0, 0.0]], F)
    tlay = np.full((1, 4), F(250.0))
    for scheme in ("nssl", "thompson", "wsm6"):
        for flag, droplet in ((1.0, 7.5), (2.0, 10.5)):
            xland = np.array([flag], F)
            got_c, got_i = cloudy_background_radii(
                effc, effi, cldfra, tlay, xland, scheme=scheme)
            want_c, want_i = _legacy_radii(effc, effi, cldfra, tlay, xland)
            np.testing.assert_array_equal(
                got_c, np.array([[floor, droplet, 9.0, floor]], F),
                err_msg=scheme)
            np.testing.assert_array_equal(
                got_i, np.array([[10.01, want_i[0, 1], 30.0, 10.01]], F),
                err_msg=scheme)
            assert want_i[0, 1] > F(5.0)
            # The legacy engine keeps the same radii (its metre round trip
            # can move a kept one by an ulp).
            assert got_c[0, 1] == want_c[0, 1]
            np.testing.assert_allclose(got_c[:, :3], want_c[:, :3],
                                       rtol=1e-6)


def test_resolved_cloud_keeps_its_radius_and_mynn_water_does_not(
        monkeypatch):
    """A scheme's resolved cloud at its smallest radius keeps it; a layer
    MYNN gave water is sized as WRF sizes an unsized one, whatever radius
    the scheme wrote there.

    NSSL's 2.51 um is both its no-cloud value and its smallest computed
    radius, so only the merge can tell MYNN's water from NSSL's own cloud.
    Thompson sizes a trace of its own liquid (from 1e-12 kg m-3) at its
    2.51 um floor, so MYNN's water beside that trace was radiated at 2.51
    um on both engines.  P3's ice is radiated as snow at P3's own radius
    (WRF moves it onto the snow species), so only its liquid rule applies.
    """
    monkeypatch.setitem(sys.modules, "cupy", np)
    from gpuwm.core.rrtmgp import cloudy_background_radii

    floor = NSSL_FLOOR_UM
    # Layers: resolved at the floor; MYNN water at the floor; MYNN water
    # at the background; resolved at a computed size; a clear layer.
    effc = np.array([[floor, floor, 2.49, 9.0, floor]], F)
    effi = np.array([[10.01, 10.01, 4.99, 30.0, 10.01]], F)
    cldfra = np.array([[1.0, 1.0, 1.0, 1.0, 0.0]], F)
    supplied = np.array([[False, True, True, False, True]])
    tlay = np.full((1, 5), F(250.0))
    land, water = np.array([1.0], F), np.array([2.0], F)
    table = _legacy_radii(np.full((1, 1), F(2.49)), np.full((1, 1), F(0.0)),
                          np.ones((1, 1), F), tlay[:, :1], land)[1][0, 0]

    for scheme in ("nssl", "thompson", "wsm6"):
        for flag, droplet in ((land, 7.5), (water, 10.5)):
            got_c, got_i = cloudy_background_radii(
                effc, effi, cldfra, tlay, flag, scheme=scheme,
                supplied_liquid=supplied, supplied_ice=supplied)
            np.testing.assert_array_equal(
                got_c, np.array([[floor, droplet, droplet, 9.0, floor]], F),
                err_msg=scheme)
            np.testing.assert_array_equal(
                got_i, np.array([[10.01, table, table, 30.0, 10.01]], F),
                err_msg=scheme)
            want_c, want_i = _legacy_radii(effc, effi, cldfra, tlay, flag,
                                           supplied, supplied)
            cloudy = cldfra > 0
            np.testing.assert_array_equal(got_c[cloudy], want_c[cloudy])
            np.testing.assert_array_equal(got_i[cloudy], want_i[cloudy])

    p3_c, p3_i = cloudy_background_radii(
        effc, effi, cldfra, tlay, land, scheme="p3",
        supplied_liquid=supplied, supplied_ice=supplied)
    np.testing.assert_array_equal(
        p3_c, np.array([[floor, 7.5, 7.5, 9.0, floor]], F))
    np.testing.assert_array_equal(p3_i, effi)
    # The legacy engine leaves P3's ice radius alone for the same reason.
    re_c, re_i = unsized_mynn_radii(
        (effc * F(1e-6)).astype(F), (effi * F(1e-6)).astype(F), supplied,
        supplied, ice_rule=False)
    np.testing.assert_array_equal(re_i, (effi * F(1e-6)).astype(F))
    np.testing.assert_array_equal(re_c[supplied], F(0.0))
    with pytest.raises(ValueError, match="computes its radii in the adapter"):
        cloudy_background_radii(effc, effi, cldfra, tlay, land,
                                scheme="morrison")
    with pytest.raises(ValueError, match="supplied_liquid must be a boolean"):
        cloudy_background_radii(effc, effi, cldfra, tlay, land,
                                scheme="thompson",
                                supplied_liquid=supplied.astype(F))


def test_supplied_layers_are_exactly_where_the_merge_adds_water():
    """``mynn_bl_cloud_supplied`` marks the layers ``merge_mynn_bl_clouds``
    gives nonzero MYNN water, at WRF's strict thresholds, and nothing when
    the merge is off."""
    qc = np.array([[0.9e-6, 1.0e-6, 0.0, 2.0e-8, 0.0, 0.0]], F)
    qi = np.array([[0.9e-8, 1.0e-8, 0.0, 0.0, 3.0e-9, 0.0]], F)
    qc_bl = np.array([[4.0e-7, 5.0e-7, 6.0e-7, 7.0e-5, 0.0, 1.0e-5]], F)
    qi_bl = np.array([[4.0e-9, 5.0e-9, 6.0e-9, 0.0, 2.0e-6, 1.0e-6]], F)
    cldfra_bl = np.array([[0.0011, 0.8, 0.001, 0.5, 0.4, 0.0]], F)
    liquid, ice = mynn_bl_cloud_supplied(
        qc, qi, qc_bl=qc_bl, qi_bl=qi_bl, cldfra_bl=cldfra_bl,
        bl_pbl_physics=5, icloud_bl=1)
    merged_qc, merged_qi, _ = merge_mynn_bl_clouds(
        qc.copy(), qi.copy(), None, qc_bl=qc_bl, qi_bl=qi_bl,
        cldfra_bl=cldfra_bl, bl_pbl_physics=5, icloud_bl=1, itimestep=3)
    np.testing.assert_array_equal(liquid, merged_qc != qc)
    np.testing.assert_array_equal(ice, merged_qi != qi)
    np.testing.assert_array_equal(
        liquid, [[True, False, False, True, False, False]])
    np.testing.assert_array_equal(
        ice, [[True, False, False, False, True, False]])
    for pbl, icloud_bl in ((5, 0), (1, 1)):
        assert mynn_bl_cloud_supplied(
            qc, qi, bl_pbl_physics=pbl, icloud_bl=icloud_bl) == (None, None)
    with pytest.raises(ValueError, match="requires QC_BL"):
        mynn_bl_cloud_supplied(qc, qi, bl_pbl_physics=5, icloud_bl=1)


def _hot_afternoon_column():
    """A deep, dry-adiabatic late-August basin afternoon (26 mm of water)."""
    rd, cp_air, g = 287.0, 1004.0, 9.81
    plev = _P_TOP + _ETA * (97800.0 - _P_TOP)
    play = np.sqrt(plev[:-1] * plev[1:])
    kappa = rd / cp_air
    t700 = 311.0 * 0.7 ** kappa
    t = np.where(play >= 70000.0, 311.0 * (play / 1.0e5) ** kappa,
                 np.maximum(213.0, t700 * (play / 70000.0)
                            ** (rd * 0.0065 / g)))
    rh = np.interp(-play, -np.array([97800, 70000, 40000, 25000, 15000,
                                     5000.0]),
                   [0.30, 0.35, 0.35, 0.10, 0.02, 0.02])
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    qv = np.maximum(rh * 0.622 * es / (play - es), 3.0e-6)
    dz = rd * t * (1 + 0.608 * qv) / g * np.log(plev[:-1] / plev[1:])
    return plev, play, t, qv, dz


_COSZEN = 0.9052973     # 34.21 N 118.49 W, 2026-08-27 20:00 UTC
_SOLCON = 1341.9348     # WRF radconst at that julian day
_ALBEDO = 0.17


def _legacy_surface_sw(column, tlev, *, qc, qi, cldfra, effc, effi):
    """Surface SW down from the legacy WRF RRTMG SW transcription."""
    from gpuwm.core import rrtmg_legacy_prep as prep
    from gpuwm.core import rrtmg_sw as legsw

    plev, play, t, qv, dz = column
    nz = play.size
    zeros = np.zeros(nz, F)
    args = prep.swrad_prep(
        p3d=play.astype(F), p8w=plev.astype(F), t3d=t.astype(F),
        t8w=tlev.astype(F), dz8w=dz.astype(F), qv3d=qv.astype(F),
        qc3d=qc, qr3d=zeros, qi3d=qi, qs3d=zeros, qg3d=zeros,
        cldfra3d=cldfra, o33d=None,
        re_cloud=(effc * F(1.0e-6)).astype(F),
        re_ice=(effi * F(1.0e-6)).astype(F),
        re_snow=np.full(nz, 9.99e-6, F),
        tsk=318.0, albedo=_ALBEDO, xland=1.0, xice=0.0, snow=0.0, xlat=34.21,
        xcoszen=_COSZEN, solcon=_SOLCON, obscur=0.0, icloud=1,
        warm_rain=False, cldovrlp=2, idcor=0, o3input=0, has_reqc=1,
        has_reqi=1, has_reqs=1, yr=2026, julian=238.8333, mp_physics=8,
        g=9.81)
    nlay = args["nlay"]
    res = legsw.rrtmg_sw(
        legsw.load_sw_tables(), nlay, args["icld"], args["play"],
        args["plev"], args["tlay"], args["tlev"], args["tsfc"],
        args["h2ovmr"], args["o3vmr"], args["co2vmr"], args["ch4vmr"],
        args["n2ovmr"], args["o2vmr"], args["asdir"], args["asdif"],
        args["aldir"], args["aldif"], args["coszen"], args["adjes"],
        args["dyofyr"], args["scon"], args["inflgsw"], args["iceflgsw"],
        args["liqflgsw"], args["cldfmcl"], args["taucmcl"], args["ssacmcl"],
        args["asmcmcl"], args["fsfcmcl"], args["ciwpmcl"], args["clwpmcl"],
        args["cswpmcl"], args["reicmcl"], args["relqmcl"], args["resnmcl"],
        np.zeros((nlay, legsw.NBNDSW), F), np.ones((nlay, legsw.NBNDSW), F),
        np.zeros((nlay, legsw.NBNDSW), F), aer_opt=0)
    return float(res["swdflx"][0])


def _rte_surface_sw(column, tlev, *, qc, qi, cldfra, effc, effi):
    """Surface SW down through the RTE+RRTMGP adapter's column preparation
    (above-model layer, gas fill, hydrometeor paths) and the float64 mirrors
    of the RRTMGP optics and solver.  Overcast layers only: every g-point of
    a cloudy layer is cloudy, as McICA makes it at cloud fraction 1."""
    from gpuwm.core import rrtmgp
    from gpuwm.verify import npref

    plev, play, t, qv, _dz = column
    nz = play.size
    tables = rrtmgp.load_gas_tables("sw")
    prof = rrtmgp._extend_above_model_profile(
        play[None].astype(F), plev[None].astype(F), t[None].astype(F),
        tlev[None].astype(F), qv[None].astype(F), p_top=_P_TOP, kind="sw",
        xp=np)
    adapter = object.__new__(rrtmgp.RRTMGPRadiation)
    climatology = rrtmgp.load_trace_climatology()
    adapter.trace_vmr = dict(climatology.trace_vmr)
    adapter.trace_vmr.update(rrtmgp.trace_gases(datetime(2026, 8, 27, 20)))
    order = np.argsort(climatology.pressure_layer_pa)
    adapter._ozone_logp = np.log(
        climatology.pressure_layer_pa[order]).astype(F)
    adapter._ozone_vmr = climatology.ozone_vmr[order].astype(F)
    vmr = adapter._gas_vmr(tables, prof.play, prof.qv)
    gas = npref.np_rrtmgp_gas_optics(
        tables, prof.play, prof.plev, prof.tlay, vmr)
    tau, ssa = np.array(gas.tau), np.array(gas.ssa)
    g = np.zeros_like(tau)
    if cldfra.any():
        paths = rrtmgp.hydrometeor_paths(
            plev[None].astype(F), qc[None], None, qi[None], None,
            microphysics="thompson", effc=effc[None], effi=effi[None],
            effs=np.full((1, nz), F(9.99)), cldfra=cldfra[None],
            snow_treatment=rrtmgp.SNOW_TREATMENT_WRF_DISCOUNT)
        up = prof.upper_nlay
        pad = np.zeros((1, up))
        cld = npref.np_rrtmgp_cloud_optics(
            rrtmgp.load_cloud_tables("sw"),
            np.concatenate([np.asarray(paths.clwp), pad], axis=1),
            np.concatenate([np.asarray(paths.ciwp), pad], axis=1),
            np.concatenate([np.asarray(paths.reliq), pad + 10.0], axis=1),
            np.concatenate([np.asarray(paths.dgice), pad + 50.0], axis=1))
        bands = np.asarray(tables.gpoint_bands)
        tc, wc, gc = (np.asarray(x)[:, :, bands]
                      for x in (cld.tau, cld.ssa, cld.g))
        scatter = tau * ssa + tc * wc
        total = tau + tc
        g = np.where(scatter > 0, tc * wc * gc / np.maximum(scatter, 1e-300),
                     0.0)
        ssa = np.where(total > 0, scatter / np.maximum(total, 1e-300), 0.0)
        tau = total
    dtau, dssa, dg = npref.np_rrtmgp_delta_scale(tau, ssa, g)
    solar = np.array(tables.solar_source, np.float64)
    solar *= _SOLCON / solar.sum()
    albedo = np.full((1, tables.ngpt), _ALBEDO)
    flux = npref.np_rrtmgp_sw_rte(
        dtau, dssa, dg, np.array([_COSZEN]), albedo, albedo, solar[None],
        top_at_1=False)
    return float(flux.flux_dn[0, 0])


def test_subgrid_cloud_shades_the_ground_as_the_wrf_coupling_does(
        monkeypatch):
    """One hot afternoon column through both engines, clear and cloudy.

    Measured on this column (surface shortwave down, W m-2): clear sky
    951.2 legacy, 951.9 RTE.  Three overcast MYNN subgrid liquid layers of
    0.05 g/kg near 750 hPa: 479.4 legacy and 477.2 RTE at 7.5 um, 170.0 RTE
    at the 2.49 um background.  The same with a 1e-8 kg/kg trace of
    Thompson liquid in those layers, which Thompson sizes at its 2.51 um
    floor: 479.3 legacy and 477.2 RTE, where both engines had radiated the
    MYNN water at 2.51 um (174.9 legacy, 170.6 RTE).  Three subgrid ice
    layers of 0.02 g/kg near 400 hPa: 931.4 legacy, 928.6 RTE, 617.7 at the
    4.99 um background.  A resolved NSSL deck of three 0.02 g/kg layers
    near 900 hPa at NSSL's 2.51 um floor: 522.7 on both engines, and 815.4
    had RTE sized it at 7.5 um.
    """
    monkeypatch.setitem(sys.modules, "cupy", np)
    from gpuwm.core import rrtmgp

    column = _hot_afternoon_column()
    plev, play, t, qv, _dz = column
    tlev = np.asarray(rrtmgp._interface_temperatures(
        play[None].astype(F), plev[None].astype(F), t[None].astype(F)))[0]
    nz = play.size
    zeros = np.zeros(nz, F)
    background_c = np.full(nz, F(2.49))
    background_i = np.full(nz, F(4.99))

    clear = dict(qc=zeros, qi=zeros, cldfra=zeros, effc=background_c,
                 effi=background_i)
    legacy_clear = _legacy_surface_sw(column, tlev, **clear)
    rte_clear = _rte_surface_sw(column, tlev, **clear)
    assert abs(rte_clear - legacy_clear) < 10.0, (legacy_clear, rte_clear)

    k700 = int(np.argmin(np.abs(play - 75000.0)))
    k_ice = int(np.argmin(np.abs(t - 253.0)))
    liquid = slice(k700 - 1, k700 + 2)
    measured = {}
    for name, layers, qc_bl, qi_bl, trace in (
            ("liquid", liquid, 5.0e-5, 0.0, 0.0),
            ("liquid beside a trace", liquid, 5.0e-5, 0.0, 1.0e-8),
            ("ice", slice(k_ice - 1, k_ice + 2), 0.0, 2.0e-5, 0.0)):
        cldfra_bl = zeros.copy()
        cldfra_bl[layers] = 1.0
        qc_bl_col = np.where(cldfra_bl > 0, F(qc_bl), F(0.0)).astype(F)
        qi_bl_col = np.where(cldfra_bl > 0, F(qi_bl), F(0.0)).astype(F)
        resolved_qc = np.where(cldfra_bl > 0, F(trace), F(0.0)).astype(F)
        # Thompson's radii for that resolved state: its 2.51 um floor on
        # the trace, its no-cloud background elsewhere.
        scheme_c = np.where(resolved_qc > 0, F(2.51), background_c)
        mynn = dict(qc_bl=qc_bl_col[None], qi_bl=qi_bl_col[None],
                    cldfra_bl=cldfra_bl[None], bl_pbl_physics=5,
                    icloud_bl=1)
        supplied_c, supplied_i = mynn_bl_cloud_supplied(
            resolved_qc[None], zeros[None], **mynn)
        # The WRF merge: QC_BL/QI_BL join the resolved water and CLDFRA_BL
        # becomes the fraction (after the first model step).
        qc, qi, cldfra = merge_mynn_bl_clouds(
            resolved_qc[None].copy(), zeros[None].copy(), zeros[None].copy(),
            itimestep=11, **mynn)
        effc, effi = rrtmgp.cloudy_background_radii(
            scheme_c[None], background_i[None], cldfra, t[None].astype(F),
            np.array([1.0], F), scheme="thompson",
            supplied_liquid=supplied_c, supplied_ice=supplied_i)
        legacy_c, legacy_i = unsized_mynn_radii(
            scheme_c[None], background_i[None], supplied_c, supplied_i,
            ice_rule=True)
        cloud = dict(qc=qc[0], qi=qi[0], cldfra=cldfra[0])
        legacy = _legacy_surface_sw(column, tlev, effc=legacy_c[0],
                                    effi=legacy_i[0], **cloud)
        rte = _rte_surface_sw(column, tlev, effc=effc[0], effi=effi[0],
                              **cloud)
        # What both engines did when the scheme's radius decided alone.
        scheme_rte = _rte_surface_sw(column, tlev, effc=scheme_c,
                                     effi=background_i, **cloud)
        scheme_legacy = _legacy_surface_sw(column, tlev, effc=scheme_c,
                                           effi=background_i, **cloud)
        measured[name] = (legacy, rte, scheme_rte, scheme_legacy)
        assert abs(rte - legacy) < 0.05 * legacy, (name, legacy, rte)
        # The size is what mattered: at the scheme's radius the same cloud
        # put a third less light on the ground, or worse.
        assert scheme_rte < 0.7 * legacy, (name, legacy, scheme_rte)
        assert legacy < legacy_clear - 10.0, (name, legacy, legacy_clear)
    # A trace of resolved water below the merge threshold changes nothing
    # MYNN's water is radiated as.
    for engine in (0, 1):
        plain, traced = (measured["liquid"][engine],
                         measured["liquid beside a trace"][engine])
        assert abs(plain - traced) < 1.0, (engine, plain, traced)
    # Before, the legacy engine radiated MYNN's water beside the trace at
    # 2.51 um, and only the background at WRF's size.
    assert measured["liquid beside a trace"][3] < 0.7 * measured[
        "liquid beside a trace"][0]
    assert abs(measured["liquid"][3] - measured["liquid"][0]) < 1e-3

    # Resolved NSSL cloud at NSSL's floor keeps the floor on both engines.
    k900 = int(np.argmin(np.abs(play - 90000.0)))
    deck = zeros.copy()
    deck[k900 - 1:k900 + 2] = 2.0e-5
    overcast = np.where(deck > 0, F(1.0), F(0.0)).astype(F)
    floor_c = np.full(nz, NSSL_FLOOR_UM)
    nssl_c, nssl_i = rrtmgp.cloudy_background_radii(
        floor_c[None], np.full((1, nz), F(10.01)), overcast[None],
        t[None].astype(F), np.array([1.0], F), scheme="nssl")
    np.testing.assert_array_equal(nssl_c[0], floor_c)
    resolved = dict(qc=deck, qi=zeros, cldfra=overcast,
                    effi=np.full(nz, F(10.01)))
    legacy = _legacy_surface_sw(column, tlev, effc=floor_c, **resolved)
    rte = _rte_surface_sw(column, tlev, effc=nssl_c[0], **resolved)
    larger = _rte_surface_sw(column, tlev, effc=np.full(nz, F(7.5)),
                             **resolved)
    assert abs(rte - legacy) < 0.05 * legacy, (legacy, rte)
    assert larger > rte + 200.0, (rte, larger)


@pytest.mark.gpu
@requires_gpu
def test_production_adapter_radiates_subgrid_cloud_as_the_resolved_one():
    """The production adapter, MYNN merge on: a subgrid cloud at the
    background radius radiates exactly as the same cloud resolved at WRF's
    radius, over land and water, liquid and ice, SW and LW.

    Before the rule the subgrid liquid column put 170 W m-2 on the ground in
    the column test above where the resolved one put 477; here any
    difference at all fails.
    """
    import cupy as cp
    from gpuwm.core.rrtmgp import RRTMGPRadiation

    plev_col, play_col, t_col, qv_col, _dz = _hot_afternoon_column()
    nz, ny = play_col.size, 1
    # Columns: subgrid/resolved liquid over land, the same over water, and
    # subgrid/resolved ice over land.
    xland = np.array([1.0, 1.0, 2.0, 2.0, 1.0, 1.0], F)
    nx = xland.size
    shape = (nz, ny, nx)
    k700 = int(np.argmin(np.abs(play_col - 75000.0)))
    k_ice = int(np.argmin(np.abs(t_col - 253.0)))
    liquid, ice = slice(k700 - 1, k700 + 2), slice(k_ice - 1, k_ice + 2)

    def expand(x):
        return np.broadcast_to(
            np.asarray(x, F)[:, None, None], shape).copy()

    qc, qi = np.zeros(shape, F), np.zeros(shape, F)
    qc_bl, qi_bl = np.zeros(shape, F), np.zeros(shape, F)
    cldfra_bl = np.zeros(shape, F)
    effc = np.full(shape, F(2.49))
    effi = np.full(shape, F(4.99))
    for sub, res, radius in ((0, 1, 7.5), (2, 3, 10.5)):
        qc_bl[liquid, 0, sub] = 5.0e-5
        qc[liquid, 0, res] = 5.0e-5
        effc[liquid, 0, res] = radius
        cldfra_bl[liquid, 0, sub] = cldfra_bl[liquid, 0, res] = 1.0
    qi_bl[ice, 0, 4] = 2.0e-5
    qi[ice, 0, 5] = 2.0e-5
    cldfra_bl[ice, 0, 4] = cldfra_bl[ice, 0, 5] = 1.0
    # The resolved ice column carries WRF's tabulated radius itself, taken
    # from the legacy engine's transcription.
    _rec, rei = _legacy_radii(
        effc[:, 0, 5][None], effi[:, 0, 5][None],
        cldfra_bl[:, 0, 5][None], expand(t_col)[:, 0, 5][None],
        np.array([1.0], F))
    effi[ice, 0, 5] = rei[0][ice]
    # And both liquid-free ice columns see the same land liquid radius.
    effc[ice, 0, 5] = F(7.5)

    exner = (play_col / 1.0e5) ** (287.0 / 1004.0)
    D = cp.asarray
    atmosphere = {
        "pressure": D(expand(play_col)),
        "p_interface": D(np.broadcast_to(
            plev_col.astype(F)[:, None, None], (nz + 1, ny, nx)).copy()),
        "temperature": D(expand(t_col)),
        "theta": D(expand(t_col / exner)),
        "exner": D(expand(exner)),
        "qv": D(expand(qv_col)),
        "qc": D(qc),
        "qi": D(qi),
    }
    fields = {
        "tsk": cp.full((ny, nx), 318.0, cp.float32),
        "albedo": cp.full((ny, nx), 0.17, cp.float32),
        "emiss": cp.full((ny, nx), 0.95, cp.float32),
        "xland": D(xland.reshape(ny, nx)),
        "qc_bl": D(qc_bl), "qi_bl": D(qi_bl), "cldfra_bl": D(cldfra_bl),
    }
    state = SimpleNamespace(
        elapsed_seconds=600.0, qc=D(qc), qi=D(qi),
        qs=cp.zeros(shape, cp.float32), qr=cp.zeros(shape, cp.float32),
        effc=D(effc), effi=D(effi),
        effs=cp.full(shape, 9.99, cp.float32))
    radiation = RRTMGPRadiation(
        datetime(2026, 8, 27, 20), cp.full((ny, nx), 34.21, cp.float32),
        cp.full((ny, nx), -118.49, cp.float32))
    result = radiation(
        atmosphere=atmosphere, fields=fields, state=state,
        cfg=SimpleNamespace(mp_physics=8, dt=60.0, radt=12.0,
                            radt_minutes=12.0, bl_pbl_physics=5,
                            icloud_bl=1))
    out = {name: cp.asnumpy(getattr(result, name))
           for name in ("swdown", "glw", "rthratensw", "rthratenlw")}
    for sub, res in ((0, 1), (2, 3), (4, 5)):
        for name, value in out.items():
            np.testing.assert_array_equal(
                value[..., sub], value[..., res], err_msg=(
                    f"{name}: subgrid column {sub} and resolved column "
                    f"{res} hold the same cloud and differ"))
    # The clouds are real: each shades well below the clear-sky 950 W m-2.
    assert (out["swdown"][0, :4] < 700.0).all(), out["swdown"]
    assert (out["swdown"][0, 4:] < 945.0).all(), out["swdown"]


def _adapter_call(engine, columns, *, mp_physics, bl_pbl_physics):
    """One radiation call of a production adapter over land columns of the
    hot afternoon column.  ``columns`` is a list of dicts of per-layer
    ``qv``, ``qc``, ``qc_bl``, ``cldfra_bl`` and ``effc``; returns surface
    SW down per column."""
    import cupy as cp

    plev_col, play_col, t_col, qv_col, dz_col = _hot_afternoon_column()
    nz, ny, nx = play_col.size, 1, len(columns)
    shape = (nz, ny, nx)

    def grid(name, default):
        out = np.empty(shape, F)
        for i, column in enumerate(columns):
            out[:, 0, i] = column.get(name, default)
        return out

    def expand(x, nk=nz):
        return np.broadcast_to(
            np.asarray(x, F)[:, None, None], (nk, ny, nx)).copy()

    qv = grid("qv", qv_col)
    qc = grid("qc", 0.0)
    effc = grid("effc", 2.49)
    exner = (play_col / 1.0e5) ** (287.0 / 1004.0)
    z_at_w = np.concatenate([[0.0], np.cumsum(dz_col)])
    # WRF's half-to-full weights on this column's eta ladder.
    znw = _ETA.astype(np.float64)
    dnw = np.diff(znw)
    fnm, fnp = np.zeros(nz), np.zeros(nz)
    dn = np.zeros(nz)
    dn[1:] = 0.5 * (dnw[1:] + dnw[:-1])
    fnp[1:] = 0.5 * dnw[1:] / dn[1:]
    fnm[1:] = 0.5 * dnw[:-1] / dn[1:]
    D = cp.asarray
    zeros = cp.zeros(shape, cp.float32)
    state = SimpleNamespace(
        elapsed_seconds=600.0, qv=D(qv), qc=D(qc), qi=zeros,
        qr=zeros, qs=zeros, qg=zeros, effc=D(effc),
        effi=cp.full(shape, 10.01 if mp_physics == 18 else 4.99,
                     cp.float32),
        effs=cp.full(shape, 25.0 if mp_physics == 18 else 9.99, cp.float32),
        fnm=D(fnm.astype(F)), fnp=D(fnp.astype(F)),
        p_top=np.float32(_P_TOP), physics=None)
    atmosphere = {
        "pressure": D(expand(play_col)),
        "p_interface": D(expand(plev_col, nz + 1)),
        "temperature": D(expand(t_col)),
        "theta": D(expand(t_col / exner)),
        "exner": D(expand(exner)),
        "dz": D(expand(dz_col)),
        "z_interface": D(expand(z_at_w, nz + 1)),
        "qv": state.qv, "qc": state.qc, "qi": state.qi,
    }
    fields = {
        "tsk": cp.full((ny, nx), 318.0, cp.float32),
        "albedo": cp.full((ny, nx), 0.17, cp.float32),
        "emiss": cp.full((ny, nx), 0.95, cp.float32),
        "xland": cp.ones((ny, nx), cp.float32),
        "xice": cp.zeros((ny, nx), cp.float32),
        "snow": cp.zeros((ny, nx), cp.float32),
        "qc_bl": D(grid("qc_bl", 0.0)), "qi_bl": zeros,
        "cldfra_bl": D(grid("cldfra_bl", 0.0)),
    }
    start = datetime(2026, 8, 27, 20)
    lat = cp.full((ny, nx), 34.21, cp.float32)
    lon = cp.full((ny, nx), -118.49, cp.float32)
    if engine == "rte":
        from gpuwm.core.rrtmgp import RRTMGPRadiation
        adapter = RRTMGPRadiation(start, lat, lon)
    else:
        from gpuwm.core.rrtmg_legacy import RRTMGLegacyRadiation
        adapter = RRTMGLegacyRadiation(start, lat, lon, p_top=_P_TOP)
    result = adapter(
        atmosphere=atmosphere, fields=fields, state=state,
        cfg=SimpleNamespace(mp_physics=mp_physics, dt=60.0, radt=12.0,
                            radt_minutes=12.0, bl_pbl_physics=bl_pbl_physics,
                            icloud_bl=1, icloud=1, sf_surface_physics=2))
    return cp.asnumpy(result.swdown).reshape(-1)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("engine", ("rte", "legacy"))
def test_both_adapters_size_mynn_water_and_keep_resolved_cloud(engine):
    """Both production adapters, MYNN merge on: MYNN's water beside a trace
    of Thompson liquid, or under NSSL's no-cloud radius, shades the ground
    as the same water resolved at WRF's 7.5 um; a resolved NSSL deck at
    NSSL's 2.51 um floor shades it as that deck at 2.52 um, with MYNN on
    and off.

    Before, both engines radiated MYNN's water at 2.51 um in the first two
    cases (under 200 W m-2 where WRF's size gives about 480), and
    RTE+RRTMGP radiated the resolved NSSL deck at 7.5 um (about 815 where
    the deck passes about 520).
    """
    _plev, play, t, qv, _dz = _hot_afternoon_column()
    nz = play.size
    k700 = int(np.argmin(np.abs(play - 75000.0)))
    k900 = int(np.argmin(np.abs(play - 90000.0)))
    mynn, deck = np.zeros(nz, bool), np.zeros(nz, bool)
    mynn[k700 - 1:k700 + 2] = True
    deck[k900 - 1:k900 + 2] = True

    def where(mask, value, other):
        return np.where(mask, F(value), F(other)).astype(F)

    subgrid = dict(qc_bl=where(mynn, 5.0e-5, 0.0),
                   cldfra_bl=where(mynn, 1.0, 0.0))
    resolved = dict(qc=where(mynn, 5.0e-5, 0.0), effc=where(mynn, 7.5, 2.49),
                    cldfra_bl=where(mynn, 1.0, 0.0))

    thompson = _adapter_call(engine, [
        subgrid,
        dict(subgrid, qc=where(mynn, 1.0e-8, 0.0),
             effc=where(mynn, 2.51, 2.49)),
        resolved,
    ], mp_physics=8, bl_pbl_physics=5)
    assert thompson[2] < 600.0, thompson
    assert abs(thompson[0] - thompson[2]) < 0.05, thompson
    assert abs(thompson[1] - thompson[2]) < 0.5, thompson

    floor = NSSL_FLOOR_UM
    # A saturated deck, so WRF's cloud fraction calls it overcast with MYNN
    # off as well (cal_cldfra1: relative humidity at or above 1).
    es = 610.78 * np.exp(17.2693882 * (t - 273.15) / (t - 35.86))
    saturated = np.where(deck, 1.01 * 0.622 * es / (play - es), qv)
    nssl_deck = dict(qv=saturated.astype(F), qc=where(deck, 2.0e-5, 0.0),
                     effc=where(deck, floor, floor),
                     cldfra_bl=where(deck, 1.0, 0.0))
    for bl_pbl_physics in (5, 1):
        nssl = _adapter_call(engine, [
            nssl_deck,
            dict(nssl_deck, effc=where(deck, 2.52, floor)),
            dict(subgrid, effc=where(mynn, floor, floor)),
            dict(resolved, effc=where(mynn, 7.5, floor)),
        ], mp_physics=18, bl_pbl_physics=bl_pbl_physics)
        assert nssl[0] < 700.0, (bl_pbl_physics, nssl)
        assert abs(nssl[0] - nssl[1]) < 3.0, (bl_pbl_physics, nssl)
        if bl_pbl_physics == 5:
            assert abs(nssl[2] - nssl[3]) < 0.05, nssl
