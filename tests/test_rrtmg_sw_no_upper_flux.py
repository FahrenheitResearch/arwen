"""Legacy RRTMG shortwave on columns with no layer above the troposphere
switch: the NumPy reference held to WRF, and the kernel's store rule pinned.

THE BREAKAGE THIS PREVENTS
--------------------------
setcoef_sw counts a layer as tropospheric while log(pavel) > 4.56 (pavel
above ~95.6 hPa).  The option-4 driver adds one layer above the model top
at half its pressure, so a model top below ~191 hPa leaves laytrop ==
nlayers and the upper-atmosphere loop of every taumol band is empty.  Bands
16, 17, 27, 28 and 29 set sfluxzen only in that loop, so WRF keeps the zero
taumol_sw writes on entry.  The CUDA kernel got this wrong two ways: bands
16 and 27 returned without storing, so the batched path's recycled scratch
slot handed spcvmc leftover GPU memory, and bands 17, 28 and 29 computed a
flux WRF never sets.  None of the real, synthetic or tall oracle decks
reach this path (they all extend to 10 hPa or higher), so nothing caught
either.

tools/rrtmg_wrf461_oracle/sw_fixtures/fixtures_shallow.npz is recorded by
the unmodified WRF Fortran (sw_make_shallow.py): two columns with laytrop ==
nlayers and one control whose upper loop runs.  This file holds the NumPy
port to it at max_ulp 0, which is the reference the card is held to in
tests/test_rrtmg_sw_no_upper_flux_gpu.py, and pins the kernel's
store-on-every-path rule that keeps the batched scratch slot free of words
left by an earlier column.  It needs no device.
"""

from pathlib import Path
import re

import numpy as np
import pytest

from gpuwm.core import rrtmg_sw as sw

ROOT = Path(__file__).resolve().parents[1]
FIXDIR = ROOT / "tools" / "rrtmg_wrf461_oracle" / "sw_fixtures"
SHALLOW = ("c00301", "c00302")      # laytrop == nlayers
CONTROL = "c00303"                  # the upper loop runs
CASES = SHALLOW + (CONTROL,)
#: The bands whose taumol routine sets sfluxzen in the upper loop.
UPPER_LOOP_BANDS = (16, 17, 27, 28, 29)
#: Every WRF-level output RRTMG_SWRAD's day branch writes (the profiles,
#: the two heating tendencies and the 2-D diagnostics).
WRF_OUTPUTS = ("swupflx", "swupflxc", "swdnflx", "swdnflxc", "rthratensw",
               "rthratenswc", "gsw", "swcf", "swupt", "swuptc", "swdnt",
               "swdntc", "swupb", "swupbc", "swdnb", "swdnbc", "swvisdir",
               "swvisdif", "swnirdir", "swnirdif", "swddir", "swddni",
               "swddif", "swdownc", "swddnic", "swddirc")

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
        with np.load(FIXDIR / "fixtures_shallow.npz") as f:
            _fix = {k: f[k] for k in f.files}
    return _fix


def band_slice(band):
    ib = band - 16
    return slice(0 if ib == 0 else sw.NGS[ib - 1], sw.NGS[ib])


def assert_bits(name, got, want):
    got = np.asarray(got, dtype=np.float32)
    want = np.asarray(want, dtype=np.float32)
    assert got.shape == want.shape, f"{name}: shape {got.shape} vs {want.shape}"
    bad = np.nonzero(got.view(np.uint32).reshape(-1)
                     != want.view(np.uint32).reshape(-1))[0]
    assert bad.size == 0, (
        f"{name}: {bad.size}/{got.size} words differ; first at flat index "
        f"{bad[0]}: got {got.reshape(-1)[bad[0]]!r} want "
        f"{want.reshape(-1)[bad[0]]!r}")


def test_the_deck_reaches_the_empty_upper_loop():
    """Without this the rest of the file proves nothing about the path."""
    d = fixtures()
    for case in CASES:
        assert int(d[f"{case}/night"]) == 0, case
        nlayers = int(d[f"{case}/inatm/nlayers"])
        laytrop = int(d[f"{case}/setcoef/laytrop"])
        if case in SHALLOW:
            assert laytrop == nlayers, (case, laytrop, nlayers)
        else:
            assert laytrop < nlayers, (case, laytrop, nlayers)


def test_wrf_leaves_exactly_the_upper_loop_bands_at_zero():
    """WRF's own answer, read from the recorded Fortran: zero in the five
    upper-loop bands when their loop is empty, a positive flux in every
    other g-point, and no zero anywhere in the control."""
    d = fixtures()
    upper = np.zeros(sw.NGPTSW, dtype=bool)
    for band in UPPER_LOOP_BANDS:
        upper[band_slice(band)] = True
    for case in SHALLOW:
        sflux = d[f"{case}/taumol/sfluxzen"]
        assert np.all(sflux[upper] == np.float32(0.0)), case
        assert np.all(sflux[~upper] > 0), case
    assert np.all(d[f"{CONTROL}/taumol/sfluxzen"] > 0)


@pytest.mark.parametrize("case", CASES)
def test_numpy_setcoef_and_taumol_match_wrf(case):
    d = fixtures()
    t = tables()
    nlayers = int(d[f"{case}/inatm/nlayers"])
    sc = sw.setcoef_sw(t, nlayers, d[f"{case}/inatm/pavel"],
                       d[f"{case}/inatm/tavel"], d[f"{case}/inatm/coldry"],
                       d[f"{case}/inatm/wkl"])
    assert int(sc["laytrop"]) == int(d[f"{case}/setcoef/laytrop"])
    sfluxzen, taug, taur = sw.taumol_sw(
        t, nlayers, sc["colh2o"], sc["colco2"], sc["colch4"], sc["colo2"],
        sc["colo3"], sc["colmol"], int(sc["laytrop"]),
        sc["jp"], sc["jt"], sc["jt1"], sc["fac00"], sc["fac01"],
        sc["fac10"], sc["fac11"], sc["selffac"], sc["selffrac"],
        sc["indself"], sc["forfac"], sc["forfrac"], sc["indfor"])
    for band in range(16, 30):
        g = band_slice(band)
        assert_bits(f"{case} taumol{band}/sfluxzen", sfluxzen[g],
                    d[f"{case}/taumol/sfluxzen"][g])
        assert_bits(f"{case} taumol{band}/taug", taug[:, g],
                    d[f"{case}/taumol/taug"][:, g])
        assert_bits(f"{case} taumol{band}/taur", taur[:, g],
                    d[f"{case}/taumol/taur"][:, g])


@pytest.mark.parametrize("case", CASES)
def test_numpy_composition_matches_wrf(case):
    """The whole option-4 call, gated on WRF's own outputs."""
    d = fixtures()
    e = lambda nm: d[f"{case}/entry/{nm}"]
    nlay = int(e("nlay"))
    res = sw.rrtmg_sw(
        tables(), nlay, int(e("icld")), e("play"), e("plev"), e("tlay"),
        e("tlev"), np.float32(e("tsfc")), e("h2ovmr"), e("o3vmr"),
        e("co2vmr"), e("ch4vmr"), e("n2ovmr"), e("o2vmr"),
        np.float32(e("asdir")), np.float32(e("asdif")),
        np.float32(e("aldir")), np.float32(e("aldif")),
        np.float32(e("coszen")), np.float32(e("adjes")), int(e("dyofyr")),
        np.float32(e("scon")), int(e("inflgsw")), int(e("iceflgsw")),
        int(e("liqflgsw")), e("cldfmcl"), e("taucmcl"), e("ssacmcl"),
        e("asmcmcl"), e("fsfcmcl"), e("ciwpmcl"), e("clwpmcl"),
        e("cswpmcl"), e("reicmcl"), e("relqmcl"), e("resnmcl"),
        np.zeros((nlay, sw.NBNDSW), np.float32),
        np.ones((nlay, sw.NBNDSW), np.float32),
        np.zeros((nlay, sw.NBNDSW), np.float32), aer_opt=0)
    o = sw.swrad_option4_outputs(res, d[f"{case}/in/pi3d"],
                                 np.float32(d[f"{case}/in/xcoszen"]), nlay - 1)
    for nm in WRF_OUTPUTS:
        assert_bits(f"{case} wrf/{nm}", np.float32(o[nm]),
                    np.float32(d[f"{case}/wrf/{nm}"]))


def test_every_sfluxzen_thread_stores_on_every_path():
    """The kernel half of SW_TAKE_SLOTS' "sflux" claim, which the batched
    path relies on to skip zeroing that slot: rsw_sfluxzen_body has no
    return, so its one store at the end runs for every (column, g-point).
    A return before it is the defect this file was written for."""
    source = (ROOT / "gpuwm" / "core" / "kernels" / "rrtmg_sw.cu").read_text(
        encoding="ascii")
    start = source.index("__device__ void rsw_sfluxzen_body(")
    start = source.index("{", start)
    end, depth = start + 1, 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    body = re.sub(r"//[^\n]*", "", source[start:end])
    assert not re.search(r"\breturn\b", body), (
        "rsw_sfluxzen_body returns before its store: the batched sflux slot "
        "is recycled scratch and that g-point would read the previous "
        "chunk's bytes")
    stores = re.findall(r"\bsfluxzen\s*\[[^\]]*\]\s*=(?!=)[^;]*", body)
    assert stores == ["sfluxzen[iw] = v"], stores
    assert body.rstrip().rstrip("}").rstrip().endswith("sfluxzen[iw] = v;")
    claim = dict(sw.SW_TAKE_SLOTS)["sflux"]
    assert claim == "rsw_sfluxzen_b, every (column, g-point)"
