"""Per-routine and end-to-end gates for the RRTMG LW NumPy FP32 port.

Every comparison is bitwise (max_ulp 0) against fixtures dumped from the
UNMODIFIED WRF v4.6.1 phys/module_ra_rrtmg_lw.F by
tools/rrtmg_wrf461_oracle/lw_extract.F90 (see its header for provenance
and regeneration; lw_build.sh runs the whole pipeline).  Fixture location
comes from GPUWM_RRTMG_LW_FIXTURES (default: the oracle scratch dir).

Structural notes pinned by these gates:
* laytrop == 50 in every fixture: with the campaign configuration
  (p_top = 100 mb, deltap = 4 mb buffer layers) the 95.6 mb boundary
  always falls between buffer layers 1 and 2 -- a property of the
  configuration, not an accident of the columns.
* setcoef leaves rat_h2oo3/h2on2o/h2och4/n2oco2, indself and selffrac
  undefined above laytrop and rat_o3co2 undefined below; taumol only
  reads them where defined, and the gates mask accordingly.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOOLS = os.path.join(_REPO, "tools", "rrtmg_wrf461_oracle")
sys.path.insert(0, _TOOLS)

from lw_fixtures import read_fixture  # noqa: E402
from lw_gate import (DEFAULT_FIXDIR, band_slice, case_paths,  # noqa: E402
                     load_coeffs_fixture, port_coeffs_from_fixture,
                     raw_from_coeffs_fixture, ulp_report)

from gpuwm.core import rrtmg_lw as port  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.path.isdir(DEFAULT_FIXDIR),
    reason="RRTMG LW oracle fixtures not present "
           "(run tools/rrtmg_wrf461_oracle/lw_build.sh)")


@pytest.fixture(scope="module")
def cfx():
    return load_coeffs_fixture()


@pytest.fixture(scope="module")
def C(cfx):
    return port_coeffs_from_fixture(cfx)


@pytest.fixture(scope="module")
def cases():
    return [read_fixture(p) for p in case_paths()]


def _assert(msg):
    assert msg is None, msg


def test_init_chain_bitwise(cfx):
    """raw loader dict -> init port == post-init Fortran module state."""
    raw = raw_from_coeffs_fixture(cfx)
    built = port.build_lw_coefficients(raw, np.float32(1004.5))
    _assert(ulp_report(built["wvn/rwgt"], cfx["rrlw_wvn/rwgt"], "rwgt"))
    for name in ("tau_tbl", "exp_tbl", "tfn_tbl", "bpade"):
        _assert(ulp_report(built[f"tbl/{name}"], cfx[f"rrlw_tbl/{name}"],
                           name))
    for name in ("oneminus", "pi", "fluxfac", "heatfac"):
        _assert(ulp_report(built[f"con/{name}"], cfx[f"rrlw_con/{name}"],
                           name))
    from lw_gate_cmbgb import reduced_name
    from lw_gate import RAW_VARS
    for band in range(1, 17):
        for raw_name in RAW_VARS[band]:
            red = reduced_name(raw_name)
            key = f"kg{band:02d}/{red}"
            assert key in built, f"init port missing {key}"
            _assert(ulp_report(built[key],
                               cfx[f"rrlw_kg{band:02d}/{red}"], key))


def test_init_chain_from_ingest_loader(cfx):
    """Post-merge gate (formerly deferred): the production coefficient path
    -- gpuwm.ingest.rrtmg_coeffs.load_rrtmg_lw_coefficients() raw arrays ->
    build_lw_coefficients -- must reproduce the post-init Fortran module
    state bitwise, and every array the loader itself carries (raw and its
    own cmbgb-reduced set, including the absa/absb EQUIVALENCE views) must
    agree bitwise with the oracle dump.  Imports are unconditional -- this
    gate must never silently skip again."""
    from gpuwm.ingest.rrtmg_coeffs import load_rrtmg_lw_coefficients
    from lw_gate import RAW_VARS
    from lw_gate_cmbgb import reduced_name

    loaded = load_rrtmg_lw_coefficients()
    raw = {f"rrlw_kg{band:02d}":
           {v: loaded[f"rrlw_kg{band:02d}"][v] for v in RAW_VARS[band]}
           for band in range(1, 17)}
    built = port.build_lw_coefficients(raw, np.float32(1004.5))
    _assert(ulp_report(built["wvn/rwgt"], cfx["rrlw_wvn/rwgt"], "rwgt"))
    for name in ("tau_tbl", "exp_tbl", "tfn_tbl", "bpade"):
        _assert(ulp_report(built[f"tbl/{name}"], cfx[f"rrlw_tbl/{name}"],
                           name))
    for band in range(1, 17):
        mod = f"rrlw_kg{band:02d}"
        for raw_name in RAW_VARS[band]:
            red = reduced_name(raw_name)
            key = f"kg{band:02d}/{red}"
            assert key in built, f"init port missing {key}"
            _assert(ulp_report(built[key], cfx[f"{mod}/{red}"], key))
        for name, arr in loaded[mod].items():
            oracle_key = f"{mod}/{name}"
            if oracle_key in cfx:
                _assert(ulp_report(arr, cfx[oracle_key],
                                   f"loader {oracle_key}"))


def test_libm_shared_module_identity():
    """Post-merge contract: port.logf/expf/powf delegate to the single
    audited transcription in gpuwm.core.noahmp_libm (no second host
    transcription survives the merge), and the np.float32 adapters are
    pure type fixes."""
    from gpuwm.core import noahmp_libm

    assert port._libm is noahmp_libm
    rng = np.random.default_rng(11)
    for x in np.exp(rng.uniform(-16, 16, 4096)).astype(np.float32):
        got = port.logf(x)
        assert isinstance(got, np.float32)
        assert np.float32(noahmp_libm.logf(float(x))).view(np.uint32) == \
            got.view(np.uint32)
    for x in rng.uniform(-100, 60, 4096).astype(np.float32):
        got = port.expf(x)
        assert isinstance(got, np.float32)
        assert np.float32(noahmp_libm.expf(float(x))).view(np.uint32) == \
            got.view(np.uint32)
    for x in np.exp(rng.uniform(-9, 9, 2048)).astype(np.float32):
        for y in (np.float32("0.65"), np.float32("0.77"),
                  np.float32("0.79"), np.float32("0.68")):
            got = port.powf(x, y)
            assert isinstance(got, np.float32)
            assert np.float32(noahmp_libm.powf(float(x),
                                               float(y))).view(np.uint32) == \
                got.view(np.uint32)


def test_structural_constants(cfx):
    for name, got in (("wt", port.WT), ("ngc", port.NGC),
                      ("ngs", port.NGS), ("ngm", port.NGM),
                      ("ngn", port.NGN), ("ngb", port.NGB),
                      ("nspa", port.NSPA), ("nspb", port.NSPB),
                      ("delwave", port.DELWAVE)):
        want = cfx[f"rrlw_wvn/{name}"]
        if got.dtype == np.float32:
            assert (np.asarray(got).view(np.uint32)
                    == want.view(np.uint32)).all(), name
        else:
            assert np.array_equal(got, want), name


def test_inatm(cases, C):
    for fx in cases:
        nl = int(fx["meta/nlayers"])
        out = port.inatm(
            1, nl, int(fx["in/icld"]), 10, fx["in/play"], fx["in/plev"],
            fx["in/tlay"], fx["in/tlev"], fx["in/tsfc"], fx["in/h2ovmr"],
            fx["in/o3vmr"], fx["in/co2vmr"], fx["in/ch4vmr"],
            fx["in/n2ovmr"], fx["in/o2vmr"], fx["in/cfc11vmr"],
            fx["in/cfc12vmr"], fx["in/cfc22vmr"], fx["in/ccl4vmr"],
            fx["in/emis"], int(fx["in/inflglw"]), int(fx["in/iceflglw"]),
            int(fx["in/liqflglw"]), fx["in/cldfmcl"], fx["in/taucmcl"],
            fx["in/ciwpmcl"], fx["in/clwpmcl"], fx["in/cswpmcl"],
            fx["in/reicmcl"], fx["in/relqmcl"], fx["in/resnmcl"],
            fx["in/tauaer"], C)
        for k in ("pavel", "pz", "tavel", "tz", "tbound", "semiss",
                  "coldry", "wkl", "wbrodl", "wx", "pwvcm", "cldfmc",
                  "taucmc", "ciwpmc", "clwpmc", "cswpmc", "reicmc",
                  "relqmc", "resnmc", "taua"):
            _assert(ulp_report(out[k], fx[f"inatm/{k}"], f"inatm/{k}"))
        for k in ("inflag", "iceflag", "liqflag"):
            assert out[k] == int(fx[f"inatm/{k}"])


def test_inatm_fails_closed_for_icld0(cases, C):
    fx = cases[0]
    nl = int(fx["meta/nlayers"])
    with pytest.raises(ValueError, match="icld"):
        port.inatm(
            1, nl, 0, 10, fx["in/play"], fx["in/plev"], fx["in/tlay"],
            fx["in/tlev"], fx["in/tsfc"], fx["in/h2ovmr"],
            fx["in/o3vmr"], fx["in/co2vmr"], fx["in/ch4vmr"],
            fx["in/n2ovmr"], fx["in/o2vmr"], fx["in/cfc11vmr"],
            fx["in/cfc12vmr"], fx["in/cfc22vmr"], fx["in/ccl4vmr"],
            fx["in/emis"], 2, 3, 1, fx["in/cldfmcl"], fx["in/taucmcl"],
            fx["in/ciwpmcl"], fx["in/clwpmcl"], fx["in/cswpmcl"],
            fx["in/reicmcl"], fx["in/relqmcl"], fx["in/resnmcl"],
            fx["in/tauaer"], C)


def test_cldprmc(cases, C):
    for fx in cases:
        nl = int(fx["meta/nlayers"])
        ncb, taucmc = port.cldprmc(
            nl, int(fx["inatm/inflag"]), int(fx["inatm/iceflag"]),
            int(fx["inatm/liqflag"]), fx["inatm/cldfmc"],
            fx["inatm/ciwpmc"], fx["inatm/clwpmc"], fx["inatm/cswpmc"],
            fx["inatm/reicmc"], fx["inatm/relqmc"], fx["inatm/resnmc"],
            fx["inatm/taucmc"], C)
        assert ncb == int(fx["cldprmc/ncbands"])
        _assert(ulp_report(taucmc, fx["cldprmc/taucmc"], "taucmc"))


def test_setcoef(cases, C):
    full = ("planklay", "planklev", "plankbnd", "colh2o", "colco2",
            "colo3", "coln2o", "colco", "colch4", "colo2", "colbrd",
            "fac00", "fac01", "fac10", "fac11", "selffac", "forfac",
            "forfrac", "minorfrac", "scaleminor", "scaleminorn2",
            "rat_h2oco2", "rat_h2oco2_1")
    lower = ("selffrac", "rat_h2oo3", "rat_h2oo3_1", "rat_h2on2o",
             "rat_h2on2o_1", "rat_h2och4", "rat_h2och4_1", "rat_n2oco2",
             "rat_n2oco2_1")
    upper = ("rat_o3co2", "rat_o3co2_1")
    for fx in cases:
        nl = int(fx["meta/nlayers"])
        lt = int(fx["setcoef/laytrop"])
        out = port.setcoef(
            nl, 1, fx["inatm/pavel"], fx["inatm/tavel"], fx["inatm/tz"],
            float(fx["inatm/tbound"]), fx["inatm/semiss"],
            fx["inatm/coldry"], fx["inatm/wkl"], fx["inatm/wbrodl"], C)
        assert out["laytrop"] == lt
        for k in ("jp", "jt", "jt1", "indfor", "indminor"):
            assert np.array_equal(out[k], fx[f"setcoef/{k}"]), k
        assert np.array_equal(out["indself"][:lt],
                              fx["setcoef/indself"][:lt])
        for k in full:
            _assert(ulp_report(out[k], fx[f"setcoef/{k}"], k))
        for k in lower:
            _assert(ulp_report(out[k][:lt], fx[f"setcoef/{k}"][:lt], k))
        for k in upper:
            _assert(ulp_report(out[k][lt:], fx[f"setcoef/{k}"][lt:], k))


@pytest.mark.parametrize("band", range(1, 17))
def test_taugb_band(cases, C, band):
    from lw_gate import state_from_fixture
    s = band_slice(band)
    impl = port.TAUGB_IMPLS[band]
    for fx in cases:
        nl = int(fx["meta/nlayers"])
        st = state_from_fixture(fx)
        taug = np.zeros((nl, port.NGPTLW), dtype=np.float32)
        fracs = np.zeros((nl, port.NGPTLW), dtype=np.float32)
        with np.errstate(all="ignore"):
            impl(st, C, taug, fracs)
        _assert(ulp_report(taug[:, s], fx["taumol/taug"][:, s],
                           f"taugb{band}/taug"))
        _assert(ulp_report(fracs[:, s], fx["taumol/fracs"][:, s],
                           f"taugb{band}/fracs"))


def test_rtrnmc(cases, C):
    for fx in cases:
        nl = int(fx["meta/nlayers"])
        out = port.rtrnmc(
            nl, 1, 16, 0, fx["inatm/pz"], fx["inatm/semiss"],
            int(fx["cldprmc/ncbands"]), fx["inatm/cldfmc"],
            fx["cldprmc/taucmc"], fx["setcoef/planklay"],
            fx["setcoef/planklev"], fx["setcoef/plankbnd"],
            float(fx["inatm/pwvcm"]), fx["taumol/fracs"],
            fx["taut/taut"], C)
        for k in ("totuflux", "totdflux", "fnet", "htr", "totuclfl",
                  "totdclfl", "fnetc", "htrc"):
            _assert(ulp_report(out[k], fx[f"rtrnmc/{k}"], k))


def test_end_to_end_with_init_port_coefficients(cases, cfx, C):
    """Full chain with the coefficient path production will use:
    raw loader dict -> build_lw_coefficients -> compute chain."""
    C2 = dict(C)
    C2.update(port.build_lw_coefficients(raw_from_coeffs_fixture(cfx),
                                         np.float32(1004.5)))
    for fx in cases:
        nl = int(fx["meta/nlayers"])
        out = port.rrtmg_lw(
            1, nl, int(fx["in/icld"]), fx["in/play"], fx["in/plev"],
            fx["in/tlay"], fx["in/tlev"], fx["in/tsfc"],
            fx["in/h2ovmr"], fx["in/o3vmr"], fx["in/co2vmr"],
            fx["in/ch4vmr"], fx["in/n2ovmr"], fx["in/o2vmr"],
            fx["in/cfc11vmr"], fx["in/cfc12vmr"], fx["in/cfc22vmr"],
            fx["in/ccl4vmr"], fx["in/emis"], int(fx["in/inflglw"]),
            int(fx["in/iceflglw"]), int(fx["in/liqflglw"]),
            fx["in/cldfmcl"], fx["in/taucmcl"], fx["in/ciwpmcl"],
            fx["in/clwpmcl"], fx["in/cswpmcl"], fx["in/reicmcl"],
            fx["in/relqmcl"], fx["in/resnmcl"], fx["in/tauaer"], C2)
        for k in ("uflx", "dflx", "hr", "uflxc", "dflxc", "hrc"):
            _assert(ulp_report(out[k], fx[f"out/{k}"], k))


def test_end_to_end_from_packaged_data_file(cases, cfx, C):
    """Audit item: the REAL production coefficient path -- packaged
    RRTMG_LW_DATA -> gpuwm.ingest.rrtmg_coeffs parser -> raw arrays ->
    build_lw_coefficients -> full compute chain -> out/* at max_ulp 0.
    test_init_chain_from_ingest_loader proves the coefficient stage
    bitwise; this runs the whole chain from the packaged file so a
    parser/endianness/wiring error can never hide behind transitivity."""
    from gpuwm.ingest.rrtmg_coeffs import load_rrtmg_lw_coefficients
    from lw_gate import RAW_VARS

    loaded = load_rrtmg_lw_coefficients()
    raw = {f"rrlw_kg{band:02d}":
           {v: loaded[f"rrlw_kg{band:02d}"][v] for v in RAW_VARS[band]}
           for band in range(1, 17)}
    C2 = dict(C)
    C2.update(port.build_lw_coefficients(raw, np.float32(1004.5)))
    for fx in cases:
        nl = int(fx["meta/nlayers"])
        out = port.rrtmg_lw(
            1, nl, int(fx["in/icld"]), fx["in/play"], fx["in/plev"],
            fx["in/tlay"], fx["in/tlev"], fx["in/tsfc"],
            fx["in/h2ovmr"], fx["in/o3vmr"], fx["in/co2vmr"],
            fx["in/ch4vmr"], fx["in/n2ovmr"], fx["in/o2vmr"],
            fx["in/cfc11vmr"], fx["in/cfc12vmr"], fx["in/cfc22vmr"],
            fx["in/ccl4vmr"], fx["in/emis"], int(fx["in/inflglw"]),
            int(fx["in/iceflglw"]), int(fx["in/liqflglw"]),
            fx["in/cldfmcl"], fx["in/taucmcl"], fx["in/ciwpmcl"],
            fx["in/clwpmcl"], fx["in/cswpmcl"], fx["in/reicmcl"],
            fx["in/relqmcl"], fx["in/resnmcl"], fx["in/tauaer"], C2)
        for k in ("uflx", "dflx", "hr", "uflxc", "dflxc", "hrc"):
            _assert(ulp_report(out[k], fx[f"out/{k}"], k))


def test_fixture_provenance(cases):
    """The oracle's own cross-checks, re-asserted: composed chain ==
    direct rrtmg_lw, and wrapper transcription == real RRTMG_LWRAD."""
    f32 = np.float32
    for fx in cases:
        nl = int(fx["meta/nlayers"])
        nz = int(fx["meta/nz"])
        _assert(ulp_report(fx["out/uflx"][0], fx["rtrnmc/totuflux"],
                           "uflx==totuflux"))
        _assert(ulp_report(fx["out/hr"][0], fx["rtrnmc/htr"][:nl],
                           "hr==htr"))
        _assert(ulp_report(fx["wrap/glw"], fx["out/dflx"][0][0], "glw"))
        _assert(ulp_report(fx["wrap/olr"], fx["out/uflx"][0][nl], "olr"))
        hr = fx["out/hr"][0][:nz].astype(f32)
        rth = ((hr / f32(86400.0)).astype(f32)
               / fx["wrap/pi3d"].astype(f32)).astype(f32)
        _assert(ulp_report(fx["wrap/rthratenlw"], rth, "rthratenlw"))
