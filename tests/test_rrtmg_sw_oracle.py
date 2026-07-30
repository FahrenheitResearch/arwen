"""Oracle gates for the legacy RRTMG SW NumPy reference port.

Every gate is max_ulp 0 (bitwise equality of FP32 words) against fixtures
recorded from the UNMODIFIED WRF v4.6.1 Fortran by
tools/rrtmg_wrf461_oracle/sw_fixture_driver.F90, which itself refuses to
write a fixture unless its chained run recomposes bit-identically to the
untouched RRTMG_SWRAD / rrtmg_sw on the same column.
"""

from pathlib import Path

import numpy as np
import pytest

from gpuwm.core import rrtmg_sw as sw

FIXDIR = Path(__file__).resolve().parents[1] / "tools" / "rrtmg_wrf461_oracle" / "sw_fixtures"

_FIX_FILES = ("fixtures_real.npz", "fixtures_synth.npz")


def _case_list():
    cases = set()
    for name in _FIX_FILES:
        cases |= {k.split("/")[0] for k in np.load(FIXDIR / name).files}
    return sorted(cases)


ALL_CASES = _case_list()


def _rt_case_list():
    out = []
    for name in _FIX_FILES:
        files = set(np.load(FIXDIR / name).files)
        out += sorted({k.split("/")[0] for k in files
                       if k.endswith("/rt/ztauc")})
    return sorted(out)


RT_CASES = _rt_case_list()

_tables = None
_fix = None


def tables():
    global _tables
    if _tables is None:
        _tables = sw.tables_from_dump(dict(np.load(FIXDIR / "sw_tables.npz")))
    return _tables


def fixtures():
    global _fix
    if _fix is None:
        _fix = {}
        for name in _FIX_FILES:
            f = np.load(FIXDIR / name)
            _fix.update({k: f[k] for k in f.files})
    return _fix


def case_ids():
    d = fixtures()
    return sorted({k.split("/")[0] for k in d})


def day_case_ids():
    d = fixtures()
    return [c for c in case_ids() if int(d[f"{c}/night"]) == 0]


def assert_bits(name, got, want):
    got = np.asarray(got, dtype=np.float32)
    want = np.asarray(want, dtype=np.float32)
    assert got.shape == want.shape, f"{name}: shape {got.shape} vs {want.shape}"
    gb = got.view(np.uint32)
    wb = want.view(np.uint32)
    if np.array_equal(gb, wb):
        return
    diff = gb.astype(np.int64) - wb.astype(np.int64)
    bad = np.nonzero(diff.reshape(-1))[0]
    i = bad[0]
    raise AssertionError(
        f"{name}: {bad.size}/{got.size} words differ; max |ulp| = "
        f"{np.abs(diff).max()}; first at flat index {i}: "
        f"got {got.reshape(-1)[i]!r} want {want.reshape(-1)[i]!r}")


def assert_ints(name, got, want):
    got = np.asarray(got, dtype=np.int64)
    want = np.asarray(want, dtype=np.int64)
    assert np.array_equal(got, want), f"{name}: integer mismatch"


@pytest.mark.parametrize("case", [])
def _placeholder(case):
    pass


@pytest.mark.parametrize("case", ALL_CASES)
def test_setcoef_sw(case):
    d = fixtures()
    if int(d[f"{case}/night"]) == 1:
        pytest.skip("night column: SW not called")
    t = tables()
    nlayers = int(d[f"{case}/inatm/nlayers"])
    out = sw.setcoef_sw(t, nlayers,
                        d[f"{case}/inatm/pavel"], d[f"{case}/inatm/tavel"],
                        d[f"{case}/inatm/coldry"], d[f"{case}/inatm/wkl"])
    assert out["laytrop"] == int(d[f"{case}/setcoef/laytrop"])
    assert out["laylow"] == int(d[f"{case}/setcoef/laylow"])
    for nm in ("jp", "jt", "jt1", "indself", "indfor"):
        assert_ints(f"{case} setcoef/{nm}", out[nm], d[f"{case}/setcoef/{nm}"])
    for nm in ("colh2o", "colco2", "colo3", "coln2o", "colch4", "colo2",
               "colmol", "co2mult", "selffac", "selffrac", "forfac",
               "forfrac", "fac00", "fac01", "fac10", "fac11"):
        assert_bits(f"{case} setcoef/{nm}", out[nm], d[f"{case}/setcoef/{nm}"])


@pytest.mark.parametrize("case", ALL_CASES)
def test_taumol_sw(case):
    d = fixtures()
    if int(d[f"{case}/night"]) == 1:
        pytest.skip("night column: SW not called")
    t = tables()
    nlayers = int(d[f"{case}/inatm/nlayers"])
    sc = {nm: d[f"{case}/setcoef/{nm}"] for nm in
          ("jp", "jt", "jt1", "indself", "indfor", "colh2o", "colco2",
           "colo3", "colch4", "colo2", "colmol", "selffac", "selffrac",
           "forfac", "forfrac", "fac00", "fac01", "fac10", "fac11")}
    sfluxzen, taug, taur = sw.taumol_sw(
        t, nlayers, sc["colh2o"], sc["colco2"], sc["colch4"], sc["colo2"],
        sc["colo3"], sc["colmol"], int(d[f"{case}/setcoef/laytrop"]),
        sc["jp"], sc["jt"], sc["jt1"], sc["fac00"], sc["fac01"],
        sc["fac10"], sc["fac11"], sc["selffac"], sc["selffrac"],
        sc["indself"], sc["forfac"], sc["forfrac"], sc["indfor"])
    want_taug = d[f"{case}/taumol/taug"]
    want_taur = d[f"{case}/taumol/taur"]
    want_sflux = d[f"{case}/taumol/sfluxzen"]
    # gate per band so a failure names the band routine
    for ib in range(14):
        lo = 0 if ib == 0 else sw.NGS[ib - 1]
        hi = sw.NGS[ib]
        assert_bits(f"{case} taumol{16 + ib}/taug", taug[:, lo:hi],
                    want_taug[:, lo:hi])
        assert_bits(f"{case} taumol{16 + ib}/taur", taur[:, lo:hi],
                    want_taur[:, lo:hi])
        assert_bits(f"{case} taumol{16 + ib}/sfluxzen", sfluxzen[lo:hi],
                    want_sflux[lo:hi])


@pytest.mark.parametrize("case", ALL_CASES)
def test_cldprmc_sw(case):
    d = fixtures()
    if int(d[f"{case}/night"]) == 1:
        pytest.skip("night column: SW not called")
    t = tables()
    nlayers = int(d[f"{case}/inatm/nlayers"])
    taucmc = d[f"{case}/inatm/taucmc"].copy()
    ssacmc = d[f"{case}/inatm/ssacmc"].copy()
    asmcmc = d[f"{case}/inatm/asmcmc"].copy()
    taormc = sw.cldprmc_sw(
        t, nlayers, int(d[f"{case}/inatm/inflag"]),
        int(d[f"{case}/inatm/iceflag"]), int(d[f"{case}/inatm/liqflag"]),
        d[f"{case}/inatm/cldfmc"], d[f"{case}/inatm/ciwpmc"],
        d[f"{case}/inatm/clwpmc"], d[f"{case}/inatm/cswpmc"],
        d[f"{case}/inatm/reicmc"], d[f"{case}/inatm/relqmc"],
        d[f"{case}/inatm/resnmc"], taucmc, ssacmc, asmcmc,
        d[f"{case}/inatm/fsfcmc"])
    assert_bits(f"{case} cldprmc/taormc", taormc, d[f"{case}/cldprmc/taormc"])
    assert_bits(f"{case} cldprmc/taucmc", taucmc, d[f"{case}/cldprmc/taucmc"])
    assert_bits(f"{case} cldprmc/ssacmc", ssacmc, d[f"{case}/cldprmc/ssacmc"])
    assert_bits(f"{case} cldprmc/asmcmc", asmcmc, d[f"{case}/cldprmc/asmcmc"])


@pytest.mark.parametrize("case", RT_CASES)
def test_reftra_vrtqdr_sw(case):
    d = fixtures()
    t = tables()
    nlayers = int(d[f"{case}/inatm/nlayers"])
    klev = nlayers
    albdir = d[f"{case}/spcin/albdir"]
    albdif = d[f"{case}/spcin/albdif"]
    ngb = t.ngb
    cossza = np.float32(d[f"{case}/spcin/cossza"])
    lrtcld = d[f"{case}/rt/lrtchkcld"]
    Fn = np.float32
    for iw in range(1, sw.NGPTSW + 1):
        ibm = int(ngb[iw - 1]) - 15
        # clear reftra
        zrefc = np.zeros(klev + 1, dtype=np.float32)
        zrefdc = np.zeros(klev + 1, dtype=np.float32)
        ztrac = np.zeros(klev + 1, dtype=np.float32)
        ztradc = np.zeros(klev + 1, dtype=np.float32)
        sw.reftra_sw(t, klev, np.ones(klev, dtype=bool),
                     d[f"{case}/rt/zgcc"][:, iw - 1], cossza,
                     d[f"{case}/rt/ztauc"][:, iw - 1],
                     d[f"{case}/rt/zomcc"][:, iw - 1],
                     zrefc, zrefdc, ztrac, ztradc)
        assert_bits(f"{case} rt/zrefc iw={iw}", zrefc[:klev],
                    d[f"{case}/rt/zrefc"][:, iw - 1])
        assert_bits(f"{case} rt/zrefdc iw={iw}", zrefdc[:klev],
                    d[f"{case}/rt/zrefdc"][:, iw - 1])
        assert_bits(f"{case} rt/ztrac iw={iw}", ztrac[:klev],
                    d[f"{case}/rt/ztrac"][:, iw - 1])
        assert_bits(f"{case} rt/ztradc iw={iw}", ztradc[:klev],
                    d[f"{case}/rt/ztradc"][:, iw - 1])
        # cloudy reftra
        zrefo = np.zeros(klev + 1, dtype=np.float32)
        zrefdo = np.zeros(klev + 1, dtype=np.float32)
        ztrao = np.zeros(klev + 1, dtype=np.float32)
        ztrado = np.zeros(klev + 1, dtype=np.float32)
        sw.reftra_sw(t, klev, lrtcld[:, iw - 1] != 0,
                     d[f"{case}/rt/zgco"][:, iw - 1], cossza,
                     d[f"{case}/rt/ztauo"][:, iw - 1],
                     d[f"{case}/rt/zomco"][:, iw - 1],
                     zrefo, zrefdo, ztrao, ztrado)
        assert_bits(f"{case} rt/zrefo iw={iw}", zrefo[:klev],
                    d[f"{case}/rt/zrefo"][:, iw - 1])
        assert_bits(f"{case} rt/zrefdo iw={iw}", zrefdo[:klev],
                    d[f"{case}/rt/zrefdo"][:, iw - 1])
        assert_bits(f"{case} rt/ztrao iw={iw}", ztrao[:klev],
                    d[f"{case}/rt/ztrao"][:, iw - 1])
        assert_bits(f"{case} rt/ztrado iw={iw}", ztrado[:klev],
                    d[f"{case}/rt/ztrado"][:, iw - 1])

        # vrtqdr: clear-sky pass, then total-sky pass (ref/tra combined
        # exactly as spcvmc does)
        pref_t = np.zeros(klev + 1, dtype=np.float32)
        prefd_t = np.zeros(klev + 1, dtype=np.float32)
        ptra_t = np.zeros(klev + 1, dtype=np.float32)
        ptrad_t = np.zeros(klev + 1, dtype=np.float32)
        for jk in range(1, klev + 1):
            ikl = klev + 1 - jk
            cf = d[f"{case}/spcin/zcldfmc"][ikl - 1, iw - 1]
            zclear = Fn(Fn(1.0) - cf)
            pref_t[jk - 1] = Fn(Fn(zclear * zrefc[jk - 1]) + Fn(cf * zrefo[jk - 1]))
            prefd_t[jk - 1] = Fn(Fn(zclear * zrefdc[jk - 1]) + Fn(cf * zrefdo[jk - 1]))
            ptra_t[jk - 1] = Fn(Fn(zclear * ztrac[jk - 1]) + Fn(cf * ztrao[jk - 1]))
            ptrad_t[jk - 1] = Fn(Fn(zclear * ztradc[jk - 1]) + Fn(cf * ztrado[jk - 1]))
        clear_ref = np.concatenate([d[f"{case}/rt/zrefc"][:, iw - 1],
                                    [np.float32(0)]])
        clear_refd = np.concatenate([d[f"{case}/rt/zrefdc"][:, iw - 1],
                                     [np.float32(0)]])
        clear_tra = np.concatenate([d[f"{case}/rt/ztrac"][:, iw - 1],
                                    [np.float32(0)]])
        clear_trad = np.concatenate([d[f"{case}/rt/ztradc"][:, iw - 1],
                                     [np.float32(0)]])
        for pref, prefd, ptra, ptrad, dbt_n, tdbt_n, rdnd_n, rup_n, rupd_n, \
                fd_n, fu_n in (
                (clear_ref, clear_refd, clear_tra, clear_trad, "zdbtc",
                 "ztdbtc", "zrdndc", "zrupc", "zrupdc", "zcd", "zcu"),
                (pref_t, prefd_t, ptra_t, ptrad_t, "zdbt", "ztdbt",
                 "zrdnd", "zrup", "zrupd", "zfd", "zfu")):
            pref = pref.copy(); prefd = prefd.copy()
            ptra = ptra.copy(); ptrad = ptrad.copy()
            pref[klev] = albdir[ibm - 1]
            prefd[klev] = albdif[ibm - 1]
            ptra[klev] = np.float32(0.0)
            ptrad[klev] = np.float32(0.0)
            pdbt = d[f"{case}/rt/{dbt_n}"][:, iw - 1].copy()
            ptdbt = d[f"{case}/rt/{tdbt_n}"][:, iw - 1].copy()
            prdnd = np.zeros(klev + 1, dtype=np.float32)
            prup = np.zeros(klev + 1, dtype=np.float32)
            prupd = np.zeros(klev + 1, dtype=np.float32)
            prup[klev] = albdir[ibm - 1]
            prupd[klev] = albdif[ibm - 1]
            pfd = np.zeros((klev + 1, sw.NGPTSW), dtype=np.float32)
            pfu = np.zeros((klev + 1, sw.NGPTSW), dtype=np.float32)
            sw.vrtqdr_sw(klev, iw, pref, prefd, ptra, ptrad, pdbt,
                         prdnd, prup, prupd, ptdbt, pfd, pfu)
            assert_bits(f"{case} {fd_n} iw={iw}", pfd[:, iw - 1],
                        d[f"{case}/rt/{fd_n}"][:, iw - 1])
            assert_bits(f"{case} {fu_n} iw={iw}", pfu[:, iw - 1],
                        d[f"{case}/rt/{fu_n}"][:, iw - 1])
            assert_bits(f"{case} {rdnd_n} iw={iw}", prdnd,
                        d[f"{case}/rt/{rdnd_n}"][:, iw - 1])
            assert_bits(f"{case} {rup_n} iw={iw}", prup,
                        d[f"{case}/rt/{rup_n}"][:, iw - 1])
            assert_bits(f"{case} {rupd_n} iw={iw}", prupd,
                        d[f"{case}/rt/{rupd_n}"][:, iw - 1])


@pytest.mark.parametrize("case", ALL_CASES)
def test_spcvmc_sw(case):
    d = fixtures()
    if int(d[f"{case}/night"]) == 1:
        pytest.skip("night column: SW not called")
    t = tables()
    nlayers = int(d[f"{case}/inatm/nlayers"])

    def sc(nm):
        return d[f"{case}/setcoef/{nm}"]

    out = sw.spcvmc_sw(
        t, nlayers, sw.JPB1, sw.JPB2, 1,
        d[f"{case}/spcin/albdif"], d[f"{case}/spcin/albdir"],
        d[f"{case}/spcin/zcldfmc"], d[f"{case}/spcin/ztaucmc"],
        d[f"{case}/spcin/zasycmc"], d[f"{case}/spcin/zomgcmc"],
        d[f"{case}/spcin/ztaormc"],
        d[f"{case}/spcin/ztaua"], d[f"{case}/spcin/zasya"],
        d[f"{case}/spcin/zomga"], np.float32(d[f"{case}/spcin/cossza"]),
        d[f"{case}/inatm/adjflux"], int(sc("laytrop")),
        sc("jp"), sc("jt"), sc("jt1"),
        sc("colch4"), sc("colco2"), sc("colh2o"), sc("colmol"),
        sc("colo2"), sc("colo3"),
        sc("fac00"), sc("fac01"), sc("fac10"), sc("fac11"),
        sc("selffac"), sc("selffrac"), sc("indself"),
        sc("forfac"), sc("forfrac"), sc("indfor"))
    for nm, fx in (("pbbfd", "zbbfd"), ("pbbfu", "zbbfu"), ("pbbcd", "zbbcd"),
                   ("pbbcu", "zbbcu"), ("puvfd", "zuvfd"), ("puvcd", "zuvcd"),
                   ("pnifd", "znifd"), ("pnicd", "znicd"),
                   ("pbbfddir", "zbbfddir"), ("pbbcddir", "zbbcddir"),
                   ("puvfddir", "zuvfddir"), ("puvcddir", "zuvcddir"),
                   ("pnifddir", "znifddir"), ("pnicddir", "znicddir")):
        assert_bits(f"{case} spcvmc/{nm}", out[nm], d[f"{case}/spcout/{fx}"])


def test_init_builds_match_oracle_dump():
    """exp_tbl/bpade/heatfac/oneminus rebuilt with the glibc expf
    transcription must equal rrtmg_sw_ini's values bitwise."""
    dump = dict(np.load(FIXDIR / "sw_tables.npz"))
    exp_tbl, bpade = sw.build_exp_tbl()
    assert_bits("exp_tbl", exp_tbl, dump["tbl/exp_tbl"])
    assert_bits("bpade", np.float32(bpade), np.float32(dump["tbl/bpade"]))
    assert_bits("heatfac", np.float32(sw.build_heatfac()),
                np.float32(dump["con/heatfac"]))
    assert_bits("oneminus", np.float32(np.float32(1.0) - np.float32(1e-6)),
                np.float32(dump["con/oneminus"]))


def test_embedded_static_tables_match_oracle_dump():
    dump = dict(np.load(FIXDIR / "sw_tables.npz"))
    for name in sw._STATIC_TABLES:
        got = sw._static(name)
        want = dump[name]
        if got.dtype == np.int32:
            assert_ints(name, got, want)
        else:
            assert_bits(name, got, want)


def test_tables_from_coeffs_matches_dump():
    """Post-merge gate (formerly deferred): the ingest loader's reduced band
    tables must agree bitwise with the oracle dump.  The import is
    unconditional -- this gate must never silently skip again."""
    from gpuwm.ingest.rrtmg_coeffs import load_rrtmg_sw_coefficients

    t2 = sw.tables_from_coeffs(load_rrtmg_sw_coefficients())
    t1 = tables()
    for band in range(16, 30):
        for k, v in t1.kg[band].items():
            got = np.asarray(t2.kg[band][k])
            if got.dtype.kind == "f":
                assert_bits(f"kg{band}/{k}", got.reshape(-1),
                            np.asarray(v).reshape(-1))


def test_libm_shared_module_identity():
    """Post-merge contract: sw.logf/sw.expf delegate to the single audited
    transcription in gpuwm.core.noahmp_libm (no second host transcription
    survives the merge), and the np.float32 adapters are pure type fixes."""
    from gpuwm.core import noahmp_libm

    assert sw._libm is noahmp_libm
    rng = np.random.default_rng(7)
    for x in np.exp(rng.uniform(-11, 7, 4096)).astype(np.float32):
        got = sw.logf(x)
        assert isinstance(got, np.float32)
        assert np.float32(noahmp_libm.logf(float(x))).view(np.uint32) == \
            got.view(np.uint32)
    for x in rng.uniform(-90, 12, 4096).astype(np.float32):
        got = sw.expf(x)
        assert isinstance(got, np.float32)
        assert np.float32(noahmp_libm.expf(float(x))).view(np.uint32) == \
            got.view(np.uint32)


@pytest.mark.parametrize("case", ALL_CASES)
def test_inatm_sw(case):
    d = fixtures()
    if int(d[f"{case}/night"]) == 1:
        pytest.skip("night column: SW not called")
    t = tables()
    nlay = int(d[f"{case}/entry/nlay"])
    ia = sw.inatm_sw(
        t, nlay, int(d[f"{case}/entry/icld"]), 10,
        d[f"{case}/entry/play"], d[f"{case}/entry/plev"],
        d[f"{case}/entry/tlay"], d[f"{case}/entry/tlev"],
        np.float32(d[f"{case}/entry/tsfc"]),
        d[f"{case}/entry/h2ovmr"], d[f"{case}/entry/o3vmr"],
        d[f"{case}/entry/co2vmr"], d[f"{case}/entry/ch4vmr"],
        d[f"{case}/entry/n2ovmr"], d[f"{case}/entry/o2vmr"],
        np.float32(d[f"{case}/entry/adjes"]), int(d[f"{case}/entry/dyofyr"]),
        np.float32(d[f"{case}/entry/scon"]),
        int(d[f"{case}/entry/inflgsw"]), int(d[f"{case}/entry/iceflgsw"]),
        int(d[f"{case}/entry/liqflgsw"]),
        d[f"{case}/entry/cldfmcl"], d[f"{case}/entry/taucmcl"],
        d[f"{case}/entry/ssacmcl"], d[f"{case}/entry/asmcmcl"],
        d[f"{case}/entry/fsfcmcl"], d[f"{case}/entry/ciwpmcl"],
        d[f"{case}/entry/clwpmcl"], d[f"{case}/entry/cswpmcl"],
        d[f"{case}/entry/reicmcl"], d[f"{case}/entry/relqmcl"],
        d[f"{case}/entry/resnmcl"],
        np.zeros((nlay, sw.NBNDSW), np.float32),
        np.ones((nlay, sw.NBNDSW), np.float32),
        np.zeros((nlay, sw.NBNDSW), np.float32))
    assert ia["nlayers"] == int(d[f"{case}/inatm/nlayers"])
    assert ia["inflag"] == int(d[f"{case}/inatm/inflag"])
    assert ia["iceflag"] == int(d[f"{case}/inatm/iceflag"])
    assert ia["liqflag"] == int(d[f"{case}/inatm/liqflag"])
    assert_bits(f"{case} inatm/tbound", np.float32(ia["tbound"]),
                np.float32(d[f"{case}/inatm/tbound"]))
    for nm in ("pavel", "tavel", "pz", "tz", "pdp", "coldry", "wkl",
               "adjflux", "solvar", "cldfmc", "taucmc", "ssacmc", "asmcmc",
               "fsfcmc", "ciwpmc", "clwpmc", "cswpmc", "reicmc", "relqmc",
               "resnmc", "taua", "ssaa", "asma"):
        assert_bits(f"{case} inatm/{nm}", ia[nm], d[f"{case}/inatm/{nm}"])


@pytest.mark.parametrize("case", ALL_CASES)
def test_rrtmg_sw_composition(case):
    """End-to-end rrtmg_sw from the entry fixtures, gated against the
    untouched RRTMG_SWRAD's WRF-level outputs (flux profiles, all 2-D
    diagnostics, RTHRATENSW via the option-4 output mapping)."""
    d = fixtures()
    if int(d[f"{case}/night"]) == 1:
        pytest.skip("night column: SW not called")
    t = tables()
    nlay = int(d[f"{case}/entry/nlay"])
    res = sw.rrtmg_sw(
        t, nlay, int(d[f"{case}/entry/icld"]),
        d[f"{case}/entry/play"], d[f"{case}/entry/plev"],
        d[f"{case}/entry/tlay"], d[f"{case}/entry/tlev"],
        np.float32(d[f"{case}/entry/tsfc"]),
        d[f"{case}/entry/h2ovmr"], d[f"{case}/entry/o3vmr"],
        d[f"{case}/entry/co2vmr"], d[f"{case}/entry/ch4vmr"],
        d[f"{case}/entry/n2ovmr"], d[f"{case}/entry/o2vmr"],
        np.float32(d[f"{case}/entry/asdir"]), np.float32(d[f"{case}/entry/asdif"]),
        np.float32(d[f"{case}/entry/aldir"]), np.float32(d[f"{case}/entry/aldif"]),
        np.float32(d[f"{case}/entry/coszen"]), np.float32(d[f"{case}/entry/adjes"]),
        int(d[f"{case}/entry/dyofyr"]), np.float32(d[f"{case}/entry/scon"]),
        int(d[f"{case}/entry/inflgsw"]), int(d[f"{case}/entry/iceflgsw"]),
        int(d[f"{case}/entry/liqflgsw"]),
        d[f"{case}/entry/cldfmcl"], d[f"{case}/entry/taucmcl"],
        d[f"{case}/entry/ssacmcl"], d[f"{case}/entry/asmcmcl"],
        d[f"{case}/entry/fsfcmcl"], d[f"{case}/entry/ciwpmcl"],
        d[f"{case}/entry/clwpmcl"], d[f"{case}/entry/cswpmcl"],
        d[f"{case}/entry/reicmcl"], d[f"{case}/entry/relqmcl"],
        d[f"{case}/entry/resnmcl"],
        np.zeros((nlay, sw.NBNDSW), np.float32),
        np.ones((nlay, sw.NBNDSW), np.float32),
        np.zeros((nlay, sw.NBNDSW), np.float32), aer_opt=0)

    kte = nlay - 1
    o = sw.swrad_option4_outputs(res, d[f"{case}/in/pi3d"],
                                 np.float32(d[f"{case}/in/xcoszen"]), kte)
    assert_bits(f"{case} wrf/swupflx", o["swupflx"], d[f"{case}/wrf/swupflx"])
    assert_bits(f"{case} wrf/swupflxc", o["swupflxc"], d[f"{case}/wrf/swupflxc"])
    assert_bits(f"{case} wrf/swdnflx", o["swdnflx"], d[f"{case}/wrf/swdnflx"])
    assert_bits(f"{case} wrf/swdnflxc", o["swdnflxc"], d[f"{case}/wrf/swdnflxc"])
    assert_bits(f"{case} wrf/rthratensw", o["rthratensw"],
                d[f"{case}/wrf/rthratensw"])
    assert_bits(f"{case} wrf/rthratenswc", o["rthratenswc"],
                d[f"{case}/wrf/rthratenswc"])
    for nm in ("gsw", "swcf", "swupt", "swuptc", "swdnt", "swdntc", "swupb",
               "swupbc", "swdnb", "swdnbc", "swvisdir", "swvisdif",
               "swnirdir", "swnirdif", "swddir", "swddni", "swddif",
               "swdownc", "swddnic", "swddirc"):
        assert_bits(f"{case} wrf/{nm}", np.float32(o[nm]),
                    np.float32(d[f"{case}/wrf/{nm}"]))


def test_option4_trace_gases_vs_fixture_co2():
    """co2vmr in every day fixture must equal the REAL(4)-exp trace-gas
    helper's value for that fixture's year."""
    d = fixtures()
    checked = 0
    for c in sorted({k.split("/")[0] for k in d}):
        if int(d[f"{c}/night"]) == 1:
            continue
        co2, ch4, n2o, o2 = sw.option4_trace_gases(int(d[f"{c}/in/yr"]))
        assert_bits(f"{c} co2vmr", np.float32(co2),
                    np.float32(d[f"{c}/entry/co2vmr"][0]))
        assert_bits(f"{c} ch4vmr", np.float32(ch4),
                    np.float32(d[f"{c}/entry/ch4vmr"][0]))
        assert_bits(f"{c} n2ovmr", np.float32(n2o),
                    np.float32(d[f"{c}/entry/n2ovmr"][0]))
        assert_bits(f"{c} o2vmr", np.float32(o2),
                    np.float32(d[f"{c}/entry/o2vmr"][0]))
        checked += 1
    assert checked > 0


def test_fails_closed():
    t = tables()
    with pytest.raises(NotImplementedError):
        sw.earth_sun(93)
    with pytest.raises(NotImplementedError):
        sw.spcvmc_sw(t, 1, sw.JPB1, sw.JPB2, 0, *([None] * 32))
