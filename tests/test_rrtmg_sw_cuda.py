"""CUDA-twin oracle gates for the legacy RRTMG SW port.

Same fixtures and same max_ulp 0 bar as tests/test_rrtmg_sw_oracle.py, but
through gpuwm/core/kernels/rrtmg_sw.cu on the GPU.  The composition test
runs twice and requires bitwise-identical results (dual-run discipline for
this machine's GPU).
"""

from pathlib import Path

import numpy as np
import pytest

from gpuwm.core import rrtmg_sw as sw

cp = pytest.importorskip("cupy")
try:
    cp.cuda.runtime.getDeviceCount()
except Exception:                                     # pragma: no cover
    pytest.skip("no CUDA device", allow_module_level=True)

FIXDIR = Path(__file__).resolve().parents[1] / "tools" / "rrtmg_wrf461_oracle" / "sw_fixtures"

_tables = None
_fix = None
_cuda = None


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


def cuda():
    global _cuda
    if _cuda is None:
        _cuda = sw.CudaSW(tables())
    return _cuda


_FIX_FILES = ("fixtures_real.npz", "fixtures_synth.npz")


def _day_cases():
    out = []
    for name in _FIX_FILES:
        f = np.load(FIXDIR / name)
        out += [k.split("/")[0] for k in f.files
                if k.endswith("/night") and int(f[k]) == 0]
    return sorted(out)


DAY_CASES = _day_cases()


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


@pytest.mark.parametrize("case", DAY_CASES)
def test_cuda_setcoef(case):
    d = fixtures()
    c = cuda()
    nlayers = int(d[f"{case}/inatm/nlayers"])
    out = c.setcoef(nlayers, d[f"{case}/inatm/pavel"],
                    d[f"{case}/inatm/tavel"], d[f"{case}/inatm/coldry"],
                    d[f"{case}/inatm/wkl"])
    assert out["laytrop"] == int(d[f"{case}/setcoef/laytrop"])
    assert out["laylow"] == int(d[f"{case}/setcoef/laylow"])
    for nm in ("jp", "jt", "jt1", "indself", "indfor"):
        assert np.array_equal(out[nm].get(), d[f"{case}/setcoef/{nm}"]), nm
    for nm in ("colh2o", "colco2", "colo3", "coln2o", "colch4", "colo2",
               "colmol", "co2mult", "selffac", "selffrac", "forfac",
               "forfrac", "fac00", "fac01", "fac10", "fac11"):
        assert_bits(f"{case} cuda setcoef/{nm}", out[nm].get(),
                    d[f"{case}/setcoef/{nm}"])


@pytest.mark.parametrize("case", DAY_CASES)
def test_cuda_taumol(case):
    d = fixtures()
    c = cuda()
    nlayers = int(d[f"{case}/inatm/nlayers"])
    sc = {nm: cp.asarray(d[f"{case}/setcoef/{nm}"]) for nm in
          ("jp", "jt", "jt1", "indself", "indfor", "colh2o", "colco2",
           "colo3", "colch4", "colo2", "colmol", "selffac", "selffrac",
           "forfac", "forfrac", "fac00", "fac01", "fac10", "fac11")}
    sc["laytrop"] = int(d[f"{case}/setcoef/laytrop"])
    sfluxzen, taug, taur = c.taumol(nlayers, sc)
    for ib in range(14):
        lo = 0 if ib == 0 else sw.NGS[ib - 1]
        hi = sw.NGS[ib]
        assert_bits(f"{case} cuda taumol{16 + ib}/taug",
                    taug.get()[:, lo:hi], d[f"{case}/taumol/taug"][:, lo:hi])
        assert_bits(f"{case} cuda taumol{16 + ib}/taur",
                    taur.get()[:, lo:hi], d[f"{case}/taumol/taur"][:, lo:hi])
        assert_bits(f"{case} cuda taumol{16 + ib}/sfluxzen",
                    sfluxzen.get()[lo:hi], d[f"{case}/taumol/sfluxzen"][lo:hi])


@pytest.mark.parametrize("case", DAY_CASES)
def test_cuda_cldprmc(case):
    d = fixtures()
    c = cuda()
    nlayers = int(d[f"{case}/inatm/nlayers"])
    g = lambda nm: cp.asfortranarray(cp.asarray(d[f"{case}/inatm/{nm}"]))
    taucmc, ssacmc, asmcmc = g("taucmc"), g("ssacmc"), g("asmcmc")
    taormc = c.cldprmc(nlayers, int(d[f"{case}/inatm/inflag"]),
                       int(d[f"{case}/inatm/iceflag"]),
                       int(d[f"{case}/inatm/liqflag"]),
                       g("cldfmc"), g("ciwpmc"), g("clwpmc"), g("cswpmc"),
                       cp.asarray(d[f"{case}/inatm/reicmc"]),
                       cp.asarray(d[f"{case}/inatm/relqmc"]),
                       cp.asarray(d[f"{case}/inatm/resnmc"]),
                       taucmc, ssacmc, asmcmc, g("fsfcmc"))
    assert_bits(f"{case} cuda cldprmc/taormc", taormc.get(),
                d[f"{case}/cldprmc/taormc"])
    assert_bits(f"{case} cuda cldprmc/taucmc", taucmc.get(),
                d[f"{case}/cldprmc/taucmc"])
    assert_bits(f"{case} cuda cldprmc/ssacmc", ssacmc.get(),
                d[f"{case}/cldprmc/ssacmc"])
    assert_bits(f"{case} cuda cldprmc/asmcmc", asmcmc.get(),
                d[f"{case}/cldprmc/asmcmc"])


@pytest.mark.parametrize("case", DAY_CASES)
def test_cuda_spcvmc(case):
    d = fixtures()
    c = cuda()
    nlayers = int(d[f"{case}/inatm/nlayers"])
    sc = {nm: cp.asarray(d[f"{case}/setcoef/{nm}"]) for nm in
          ("jp", "jt", "jt1", "indself", "indfor", "colh2o", "colco2",
           "colo3", "colch4", "colo2", "colmol", "selffac", "selffrac",
           "forfac", "forfrac", "fac00", "fac01", "fac10", "fac11")}
    sc["laytrop"] = int(d[f"{case}/setcoef/laytrop"])
    g = lambda nm: cp.asfortranarray(cp.asarray(d[f"{case}/spcin/{nm}"]))
    out = c.spcvmc(nlayers, cp.asarray(d[f"{case}/spcin/albdif"]),
                   cp.asarray(d[f"{case}/spcin/albdir"]),
                   g("zcldfmc"), g("ztaucmc"), g("zasycmc"), g("zomgcmc"),
                   g("ztaormc"), g("ztaua"), g("zasya"), g("zomga"),
                   np.float32(d[f"{case}/spcin/cossza"]),
                   cp.asarray(d[f"{case}/inatm/adjflux"]), sc)
    for nm, fx in (("pbbfd", "zbbfd"), ("pbbfu", "zbbfu"), ("pbbcd", "zbbcd"),
                   ("pbbcu", "zbbcu"), ("puvfd", "zuvfd"), ("puvcd", "zuvcd"),
                   ("pnifd", "znifd"), ("pnicd", "znicd"),
                   ("pbbfddir", "zbbfddir"), ("pbbcddir", "zbbcddir"),
                   ("puvfddir", "zuvfddir"), ("puvcddir", "zuvcddir"),
                   ("pnifddir", "znifddir"), ("pnicddir", "znicddir")):
        assert_bits(f"{case} cuda spcvmc/{nm}", out[nm].get(),
                    d[f"{case}/spcout/{fx}"])


def _run_composition(case):
    d = fixtures()
    c = cuda()
    nlay = int(d[f"{case}/entry/nlay"])
    return c.rrtmg_sw(
        nlay, int(d[f"{case}/entry/icld"]),
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
        d[f"{case}/entry/resnmcl"], aer_opt=0)


@pytest.mark.parametrize("case", DAY_CASES)
def test_cuda_rrtmg_sw_composition_dual_run(case):
    res1 = _run_composition(case)
    res2 = _run_composition(case)
    for k in res1:
        assert_bits(f"{case} dual-run {k}", res2[k], res1[k])

    _check_wrf_outputs(case, res1)


def _check_wrf_outputs(case, res):
    """The WRF-level oracle comparison shared by the per-column
    composition gate and the batched end-to-end gate."""
    d = fixtures()
    nlay = int(d[f"{case}/entry/nlay"])
    kte = nlay - 1
    o = sw.swrad_option4_outputs(res, d[f"{case}/in/pi3d"],
                                 np.float32(d[f"{case}/in/xcoszen"]), kte)
    for nm in ("swupflx", "swupflxc", "swdnflx", "swdnflxc",
               "rthratensw", "rthratenswc"):
        assert_bits(f"{case} cuda wrf/{nm}", o[nm], d[f"{case}/wrf/{nm}"])
    for nm in ("gsw", "swcf", "swupt", "swuptc", "swdnt", "swdntc",
               "swupb", "swupbc", "swdnb", "swdnbc", "swvisdir",
               "swvisdif", "swnirdir", "swnirdif", "swddir", "swddni",
               "swddif", "swdownc", "swddnic", "swddirc"):
        assert_bits(f"{case} cuda wrf/{nm}", np.float32(o[nm]),
                    np.float32(d[f"{case}/wrf/{nm}"]))


# ---------------------------------------------------------------------------
# Section 11 gates -- batched multi-column path.
#
# The batched entry shares icld/inflgsw/iceflgsw/liqflgsw/dyofyr across a
# batch, so the day deck is grouped by that flag tuple (plus nlay) and
# each group runs as one batch; together the groups cover every day case.
# All GPU-measured claims are dual-run (5090 standing rule).
# ---------------------------------------------------------------------------

DUAL_RUNS = 2

#: Every output of the per-column CudaSW.rrtmg_sw dict: the batched path
#: must reproduce each one bitwise.
OUT_KEYS = ("swuflx", "swdflx", "swhr", "swuflxc", "swdflxc", "swhrc",
            "swuflxcln", "swdflxcln", "sibvisdir", "sibvisdif",
            "sibnirdir", "sibnirdif", "swdkdir", "swdkdif", "swdkdirc")

_IN_COL = ("play", "plev", "tlay", "tlev", "h2ovmr", "o3vmr", "co2vmr",
           "ch4vmr", "n2ovmr", "o2vmr", "reicmcl", "relqmcl", "resnmcl")
_IN_SCAL = ("tsfc", "asdir", "asdif", "aldir", "aldif", "coszen",
            "adjes", "scon")
_IN_MCICA = ("cldfmcl", "taucmcl", "ssacmcl", "asmcmcl", "fsfcmcl",
             "ciwpmcl", "clwpmcl", "cswpmcl")

#: Per-kernel CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES upper bounds (KF_KMAX
#: lesson: a frame of F bytes reserves ~F x 1536 x 170 machine-wide at
#: first launch on the RTX 5090).  Measured 2026-07-27, cupy 14.0.1
#: NVRTC path on sm_120, after the spcvmc per-thread arrays moved to the
#: explicit workspace: every kernel reports 0 B.  Bounds are half a KiB
#: of headroom for toolchain drift; a frame regression past them (e.g.
#: the ~9 KiB spcvmc frame coming back) is exactly what this gate must
#: catch.
LOCAL_FRAME_BOUNDS = {}
LOCAL_FRAME_DEFAULT_BOUND = 512
#: Machine-wide reservation implied by a frame at max occupancy.
RESIDENT_THREADS = 1536 * 170


def _flag_key(case):
    d = fixtures()
    return (int(d[f"{case}/entry/icld"]), int(d[f"{case}/entry/inflgsw"]),
            int(d[f"{case}/entry/iceflgsw"]),
            int(d[f"{case}/entry/liqflgsw"]),
            int(d[f"{case}/entry/dyofyr"]), int(d[f"{case}/entry/nlay"]))


def _deck_groups():
    """Day cases grouped by shared batch flags, deck order kept."""
    groups = {}
    for case in DAY_CASES:
        groups.setdefault(_flag_key(case), []).append(case)
    return groups


def _group_inputs(cs, rep=1):
    """Concatenated (optionally column-tiled) batch inputs for a group."""
    d = fixtures()
    ins = {}
    for k in _IN_COL:
        a = np.stack([np.asarray(d[f"{c}/entry/{k}"]) for c in cs], axis=0)
        ins[k] = np.tile(a, (rep, 1)) if rep > 1 else a
    for k in _IN_SCAL:
        a = np.asarray([d[f"{c}/entry/{k}"] for c in cs], dtype=np.float32)
        ins[k] = np.tile(a, rep) if rep > 1 else a
    for k in _IN_MCICA:
        a = np.stack([np.asarray(d[f"{c}/entry/{k}"]) for c in cs], axis=1)
        ins[k] = np.tile(a, (1, rep, 1)) if rep > 1 else a
    return ins


def _run_batched(cs, ins, chunk=None, probe=None, to_device=False,
                 aer_opt=0):
    key = _flag_key(cs[0])
    icld, inflg, iceflg, liqflg, dyofyr, nlay = key
    if to_device:
        ins = {k: cp.asarray(np.ascontiguousarray(
            np.asarray(v, np.float32))) for k, v in ins.items()}
    ncol = int(ins["tsfc"].shape[0])
    return cuda().rrtmg_sw_batched(
        ncol, nlay, icld, ins["play"], ins["plev"], ins["tlay"],
        ins["tlev"], ins["tsfc"], ins["h2ovmr"], ins["o3vmr"],
        ins["co2vmr"], ins["ch4vmr"], ins["n2ovmr"], ins["o2vmr"],
        ins["asdir"], ins["asdif"], ins["aldir"], ins["aldif"],
        ins["coszen"], ins["adjes"], dyofyr, ins["scon"],
        inflg, iceflg, liqflg,
        ins["cldfmcl"], ins["taucmcl"], ins["ssacmcl"], ins["asmcmcl"],
        ins["fsfcmcl"], ins["ciwpmcl"], ins["clwpmcl"], ins["cswpmcl"],
        ins["reicmcl"], ins["relqmcl"], ins["resnmcl"], aer_opt=aer_opt,
        column_chunk=chunk, _stage_probe=probe)


def _night_xcoszens():
    """The deck's REAL night xcoszen values (<= 0 by the WRF night
    gate).  Night cases carry no SW entry records at all -- WRF never
    builds SW inputs for a night column -- so these drive the
    fail-closed gate, not an oracle comparison."""
    out = []
    for name in _FIX_FILES:
        f = np.load(FIXDIR / name)
        for k in f.files:
            if k.endswith("/night") and int(f[k]) == 1:
                case = k.split("/")[0]
                out.append(np.float32(f[f"{case}/in/xcoszen"]))
    return out


_percol_ref = None


def percol_ref():
    """Per-column CudaSW.rrtmg_sw over the day deck: the batched path's
    reference (itself oracle-gated bitwise by the composition test)."""
    global _percol_ref
    if _percol_ref is None:
        _percol_ref = {case: _run_composition(case) for case in DAY_CASES}
    return _percol_ref


def _group_expected(cs, key, rep=1):
    ref = percol_ref()
    a = np.stack([np.asarray(ref[c][key]) for c in cs], axis=0)
    return np.tile(a, (rep, 1)) if rep > 1 else a


def test_gpu_local_frames():
    """Bound every kernel's local frame; report the implied machine-wide
    reservation.  The spcvmc pipeline's per-thread arrays live in an
    explicit workspace precisely so no kernel carries a multi-KiB frame
    (~9 KiB thread-local would imply ~2.3 GiB reserved machine-wide)."""
    c = cuda()
    runs = [c.local_frame_bytes() for _ in range(DUAL_RUNS)]
    assert runs[0] == runs[1], "local frame query not stable"
    frames = runs[0]
    assert set(frames) == set(sw.SW_GPU_KERNEL_NAMES)
    for name, nbytes in sorted(frames.items(), key=lambda kv: -kv[1]):
        bound = LOCAL_FRAME_BOUNDS.get(name, LOCAL_FRAME_DEFAULT_BOUND)
        implied = nbytes * RESIDENT_THREADS
        print(f"local frame {name}: {nbytes} B "
              f"(implied reservation {implied / 2**20:.1f} MiB)")
        assert nbytes <= bound, (
            f"{name}: local frame {nbytes} B exceeds bound {bound} B "
            f"(implied machine-wide {implied / 2**30:.2f} GiB)")
        assert bound * RESIDENT_THREADS < 2 * 2**30


def test_batched_vram_estimate():
    """sw_batched_vram_bytes honesty: estimate >= pool-measured peak >=
    0.5 * estimate, at two chunk sizes (single-chunk and multi-chunk)."""
    groups = _deck_groups()
    cs = max(groups.values(), key=len)
    nlay = _flag_key(cs[0])[5]
    ins = _group_inputs(cs)
    pool = cp.get_default_memory_pool()
    for chunk in (len(cs), 8):
        estimate = sw.sw_batched_vram_bytes(min(chunk, len(cs)), nlay,
                                            ncol_total=len(cs))
        for _ in range(DUAL_RUNS):
            pool.free_all_blocks()
            base = pool.used_bytes()
            peak = [0]

            def probe(stage, peak=peak):
                peak[0] = max(peak[0], pool.used_bytes())

            _run_batched(cs, ins, chunk=chunk, probe=probe)
            measured = peak[0] - base
            print(f"chunk={chunk}: measured {measured / 2**20:.3f} MiB, "
                  f"estimate {estimate / 2**20:.3f} MiB")
            assert estimate >= measured >= 0.5 * estimate, (
                f"chunk={chunk}: measured {measured} vs "
                f"estimate {estimate}")


@pytest.mark.parametrize("chunk", [21, 11, 3, 1])
def test_batched_vs_percolumn(chunk):
    """Full-deck bit equality: batched == per-column CudaSW.rrtmg_sw for
    every output at several chunk sizes (chunking must be invisible)."""
    groups = _deck_groups()
    for _ in range(DUAL_RUNS):
        for cs in groups.values():
            ins = _group_inputs(cs)
            out = _run_batched(cs, ins, chunk=chunk)
            for k in OUT_KEYS:
                assert_bits(f"chunk{chunk}/{k}", out[k],
                            _group_expected(cs, k))


def test_batched_device_inputs():
    """Device-resident (cupy) inputs take the no-host-round-trip branch
    and must produce the same bits."""
    groups = _deck_groups()
    cs = min((g for g in groups.values() if len(g) >= 2), key=len,
             default=max(groups.values(), key=len))
    ins = _group_inputs(cs)
    for _ in range(DUAL_RUNS):
        out = _run_batched(cs, ins, chunk=3, to_device=True)
        for k in OUT_KEYS:
            assert_bits(f"dev/{k}", out[k], _group_expected(cs, k))


def test_batched_oracle():
    """End-to-end oracle gate at batch width: batched outputs vs the
    Fortran fixture records (through the same WRF-level mapping the
    per-column composition gate uses), max_ulp 0, all day cases."""
    groups = _deck_groups()
    for _ in range(DUAL_RUNS):
        for cs in groups.values():
            ins = _group_inputs(cs)
            out = _run_batched(cs, ins)
            for i, case in enumerate(cs):
                res = {k: out[k][i] for k in OUT_KEYS}
                _check_wrf_outputs(case, res)


def test_batched_night_handling():
    """Mixed day/night batches: night handling must survive batching.

    WRF's untouched-vs-zeroed night semantics live in the driver and the
    shared night helper (a night column's SW call never happens, so the
    deck's night cases carry no SW entry records at all).  Batching
    preserves that split iff (i) a batch containing night columns
    (coszen <= 0, the deck's REAL night xcoszen values) fails closed
    before any device work -- a night column can never silently acquire
    SW outputs -- and (ii) the supported composition, i.e. the caller's
    night gate filtering the day columns out of a mixed set, reproduces
    every day column's per-column bits regardless of where the night
    columns sat (the scatter/gather bookkeeping is what single-column
    gates cannot see)."""
    nightcz = _night_xcoszens()
    assert nightcz and all(z <= np.float32(0.0) for z in nightcz)
    groups = _deck_groups()
    cs = max(groups.values(), key=len)
    day = _group_inputs(cs)
    nday = len(cs)
    nn = len(nightcz)
    total = nday + nn
    # night columns cloned from a day column's entry arrays with real
    # night coszen values, inserted at the front / middle / back
    npos = np.unique(np.asarray([0, total // 2, total - 1][:nn]))
    is_night = np.zeros(total, dtype=bool)
    is_night[npos] = True
    day_slot = np.nonzero(~is_night)[0]
    mixed = {}
    for k in _IN_COL:
        a = day[k]
        m = np.zeros((total,) + a.shape[1:], dtype=a.dtype)
        m[day_slot] = a
        m[npos] = a[0]
        mixed[k] = m
    for k in _IN_SCAL:
        a = day[k]
        m = np.zeros(total, dtype=np.float32)
        m[day_slot] = a
        m[npos] = a[0]
        mixed[k] = m
    for k in _IN_MCICA:
        a = day[k]
        m = np.zeros((a.shape[0], total, a.shape[2]), dtype=a.dtype)
        m[:, day_slot] = a
        m[:, npos] = a[:, :1]
        mixed[k] = m
    mixed["coszen"][npos] = np.asarray(nightcz, np.float32)[:len(npos)]

    for _ in range(DUAL_RUNS):
        # (i) fail closed on any night column, at any position
        with pytest.raises(ValueError, match="day-columns-only"):
            _run_batched(cs, mixed)
        # (ii) the caller's night gate (coszen > 0) filters the day
        # columns; batching the filtered set reproduces per-column bits
        keep = mixed["coszen"] > np.float32(0.0)
        assert int(keep.sum()) == nday and not keep[npos].any()
        filt = {}
        for k in _IN_COL:
            filt[k] = mixed[k][keep]
        for k in _IN_SCAL:
            filt[k] = mixed[k][keep]
        for k in _IN_MCICA:
            filt[k] = mixed[k][:, keep]
        out = _run_batched(cs, filt, chunk=7)
        for k in OUT_KEYS:
            assert_bits(f"mixed-day/{k}", out[k], _group_expected(cs, k))


def test_batched_zero_aerosol_precondition():
    """Zero aerosol is a VALIDATED PRECONDITION of the batched entry
    (audit item 1): aer_opt values whose aerosol optics the device
    composition would silently discard are rejected, including the 2/3
    values the per-column signature historically accepted."""
    groups = _deck_groups()
    cs = min(groups.values(), key=len)
    ins = _group_inputs(cs)
    for bad in (1, 2, 3):
        with pytest.raises(NotImplementedError, match="aer_opt"):
            _run_batched(cs, ins, aer_opt=bad)


def test_batched_wide_determinism():
    """Tile the day deck columns past 50,000, run at the default chunk,
    and require every replica bitwise identical to its source column's
    per-column result (no width- or placement-dependence anywhere).
    The SW chain genuinely produces FP32 subnormals (1e-40 transmittance
    products); a replica differing here is a finding, never tolerable."""
    wide_min = 50000
    rep = -(-wide_min // len(DAY_CASES))
    total = rep * len(DAY_CASES)
    assert total >= wide_min
    groups = _deck_groups()
    for cs in groups.values():
        ins = _group_inputs(cs, rep=rep)
        want = {k: _group_expected(cs, k, rep=rep) for k in OUT_KEYS}
        for _ in range(DUAL_RUNS):
            out = _run_batched(cs, ins)
            for k in OUT_KEYS:
                assert_bits(f"wide[{len(cs)}x{rep}]/{k}", out[k],
                            want[k])
        del ins, want, out
