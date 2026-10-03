"""Legacy RRTMG shortwave on the card, on columns with no layer above the
troposphere switch, held to WRF at max_ulp 0.

THE BREAKAGE THIS PREVENTS
--------------------------
When laytrop == nlayers (a model top below ~191 hPa: shallow or idealized
domains), WRF's taumol bands 16, 17, 27, 28 and 29 never reach the upper
loop that sets sfluxzen, and the zero taumol_sw writes on entry stands.
rsw_sfluxzen_body returned without a store for bands 16 and 27 and computed
a flux for 17, 28 and 29.  The per-column path pre-zeroes its buffer, so it
got 16 and 27 right by accident; the batched path, the one a forecast runs,
takes that buffer from SWBatchScratch, which is never zeroed (SW_TAKE_SLOTS
says rsw_sfluxzen_b writes every (column, g-point), which was false here).
Those g-points then carried whatever the previous chunk or call left in GPU
memory into spcvmc: impossible surface fluxes, or NaN.

The batched gate below fills that slot with NaN, with 1e30, and with a
normal column's real fluxes from a previous call, then runs the shallow
columns through it in several chunks: the slot must be the one the call
uses, every output must be finite, and every WRF-level output must equal the
recorded Fortran.  The per-column gates hold the single-column kernel to the
same fixture, which is what catches bands 17, 28 and 29.  The fixture and
the NumPy port's agreement with it are in tests/test_rrtmg_sw_no_upper_flux.py.
"""

import numpy as np
import pytest

from gpuwm.core import rrtmg_sw as sw

cp = pytest.importorskip("cupy")
try:
    cp.cuda.runtime.getDeviceCount()
except Exception:                                     # pragma: no cover
    pytest.skip("no CUDA device", allow_module_level=True)

import test_rrtmg_sw_no_upper_flux as deck            # noqa: E402

#: Every output of the per-column CudaSW.rrtmg_sw dict.
OUT_KEYS = ("swuflx", "swdflx", "swhr", "swuflxc", "swdflxc", "swhrc",
            "swuflxcln", "swdflxcln", "sibvisdir", "sibvisdif",
            "sibnirdir", "sibnirdif", "swdkdir", "swdkdif", "swdkdirc")
_IN_COL = ("play", "plev", "tlay", "tlev", "h2ovmr", "o3vmr", "co2vmr",
           "ch4vmr", "n2ovmr", "o2vmr", "reicmcl", "relqmcl", "resnmcl")
_IN_SCAL = ("tsfc", "asdir", "asdif", "aldir", "aldif", "coszen",
            "adjes", "scon")
_IN_MCICA = ("cldfmcl", "taucmcl", "ssacmcl", "asmcmcl", "fsfcmcl",
             "ciwpmcl", "clwpmcl", "cswpmcl")
_SETCOEF = ("jp", "jt", "jt1", "indself", "indfor", "colh2o", "colco2",
            "colo3", "colch4", "colo2", "colmol", "selffac", "selffrac",
            "forfac", "forfrac", "fac00", "fac01", "fac10", "fac11")
#: The batch: four chunks of (shallow, shallow, control).  Slot rows 0 and
#: 1 only ever hold shallow columns, so a leftover there is never
#: overwritten by a column whose upper loop runs, and every chunk after
#: the first also inherits the previous chunk's bytes.
BATCH = deck.SHALLOW + (deck.CONTROL,)
REPEATS = 4
CHUNK = len(BATCH)

_engine = None


def engine():
    global _engine
    if _engine is None:
        _engine = sw.CudaSW(deck.tables())
    return _engine


def entry(case, nm):
    return deck.fixtures()[f"{case}/entry/{nm}"]


def wrf_outputs_match(label, case, res):
    d = deck.fixtures()
    nlay = int(entry(case, "nlay"))
    o = sw.swrad_option4_outputs(res, d[f"{case}/in/pi3d"],
                                 np.float32(d[f"{case}/in/xcoszen"]),
                                 nlay - 1)
    for nm in deck.WRF_OUTPUTS:
        deck.assert_bits(f"{label} {case} wrf/{nm}", np.float32(o[nm]),
                         np.float32(d[f"{case}/wrf/{nm}"]))


@pytest.mark.parametrize("case", deck.CASES)
def test_single_column_sfluxzen_is_wrfs(case):
    d = deck.fixtures()
    nlayers = int(d[f"{case}/inatm/nlayers"])
    sc = {nm: cp.asarray(d[f"{case}/setcoef/{nm}"]) for nm in _SETCOEF}
    sc["laytrop"] = int(d[f"{case}/setcoef/laytrop"])
    sfluxzen, _taug, _taur = engine().taumol(nlayers, sc)
    got = sfluxzen.get()
    for band in range(16, 30):
        g = deck.band_slice(band)
        deck.assert_bits(f"{case} cuda sfluxzen band {band}", got[g],
                         d[f"{case}/taumol/sfluxzen"][g])


@pytest.mark.parametrize("case", deck.CASES)
def test_single_column_composition_is_wrfs(case):
    e = lambda nm: entry(case, nm)
    res = engine().rrtmg_sw(
        int(e("nlay")), int(e("icld")), e("play"), e("plev"), e("tlay"),
        e("tlev"), np.float32(e("tsfc")), e("h2ovmr"), e("o3vmr"),
        e("co2vmr"), e("ch4vmr"), e("n2ovmr"), e("o2vmr"),
        np.float32(e("asdir")), np.float32(e("asdif")),
        np.float32(e("aldir")), np.float32(e("aldif")),
        np.float32(e("coszen")), np.float32(e("adjes")), int(e("dyofyr")),
        np.float32(e("scon")), int(e("inflgsw")), int(e("iceflgsw")),
        int(e("liqflgsw")), e("cldfmcl"), e("taucmcl"), e("ssacmcl"),
        e("asmcmcl"), e("fsfcmcl"), e("ciwpmcl"), e("clwpmcl"),
        e("cswpmcl"), e("reicmcl"), e("relqmcl"), e("resnmcl"), aer_opt=0)
    wrf_outputs_match("per-column", case, res)


def _flags(cases):
    keys = {(int(entry(c, "icld")), int(entry(c, "inflgsw")),
             int(entry(c, "iceflgsw")), int(entry(c, "liqflgsw")),
             int(entry(c, "dyofyr")), int(entry(c, "nlay"))) for c in cases}
    assert len(keys) == 1, f"the batch must share its flags: {keys}"
    return keys.pop()


def _run_batched(cases, chunk):
    icld, inflg, iceflg, liqflg, dyofyr, nlay = _flags(cases)
    ins = {k: np.stack([np.asarray(entry(c, k)) for c in cases], axis=0)
           for k in _IN_COL}
    ins.update({k: np.asarray([entry(c, k) for c in cases], np.float32)
                for k in _IN_SCAL})
    ins.update({k: np.stack([np.asarray(entry(c, k)) for c in cases], axis=1)
                for k in _IN_MCICA})
    return engine().rrtmg_sw_batched(
        len(cases), nlay, icld, ins["play"], ins["plev"], ins["tlay"],
        ins["tlev"], ins["tsfc"], ins["h2ovmr"], ins["o3vmr"],
        ins["co2vmr"], ins["ch4vmr"], ins["n2ovmr"], ins["o2vmr"],
        ins["asdir"], ins["asdif"], ins["aldir"], ins["aldif"],
        ins["coszen"], ins["adjes"], dyofyr, ins["scon"],
        inflg, iceflg, liqflg,
        ins["cldfmcl"], ins["taucmcl"], ins["ssacmcl"], ins["asmcmcl"],
        ins["fsfcmcl"], ins["ciwpmcl"], ins["clwpmcl"], ins["cswpmcl"],
        ins["reicmcl"], ins["relqmcl"], ins["resnmcl"], aer_opt=0,
        column_chunk=chunk)


def _sflux_slot(rows):
    return engine().scratch.take("sflux", (rows, sw.NGPTSW), np.float32)


@pytest.mark.parametrize("stale", ["nan", "1e30", "previous call"])
def test_batched_sflux_leftovers_never_reach_the_answer(stale):
    c = engine()
    c.release_scratch()
    if stale == "previous call":
        # A normal batch leaves real band 16/17/27/28/29 fluxes in every row.
        _run_batched((deck.CONTROL,) * CHUNK, CHUNK)
    else:
        _sflux_slot(CHUNK).fill(np.float32(float(stale)))
    before = _sflux_slot(CHUNK).data.ptr
    cases = BATCH * REPEATS
    out = _run_batched(cases, CHUNK)
    assert _sflux_slot(CHUNK).data.ptr == before, (
        "the call allocated a fresh sflux slot, so the leftovers were never "
        "offered to it and this test proved nothing")

    # Everything is read before anything is asserted, so a failure shows
    # what the slot held, how far it reached and which columns it spoiled.
    d = deck.fixtures()
    slot = _sflux_slot(CHUNK).get()          # as the last chunk left it
    stale_slot = {}
    for row, case in enumerate(BATCH):
        for band in range(16, 30):
            g = deck.band_slice(band)
            want = d[f"{case}/taumol/sfluxzen"][g]
            if not np.array_equal(slot[row, g].view(np.uint32),
                                  want.view(np.uint32)):
                stale_slot[f"row {row} {case} band {band}"] = (
                    f"holds {slot[row, g][0]!r}, WRF {want[0]!r}")
    nonfinite = {k: int((~np.isfinite(np.asarray(out[k]))).sum())
                 for k in OUT_KEYS}
    nonfinite = {k: n for k, n in nonfinite.items() if n}
    spoiled = {}
    for i, case in enumerate(cases):
        try:
            wrf_outputs_match(f"[{stale}] batched column {i}", case,
                              {k: out[k][i] for k in OUT_KEYS})
        except AssertionError as error:
            spoiled[f"column {i} {case}"] = str(error).splitlines()[0]
    c.release_scratch()
    assert not (stale_slot or nonfinite or spoiled), (
        f"[{stale}] leftover sflux scratch reached the answer\n"
        f"  slot after the call, rows that differ from WRF: {stale_slot}\n"
        f"  non-finite outputs (count per key): {nonfinite}\n"
        f"  columns off WRF ({len(spoiled)} of {len(cases)}): {spoiled}")
