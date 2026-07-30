"""Bit-for-bit oracle gate for gpuwm.core.rrtmg_mcica.

Fixtures under tests/data/rrtmg_mcica/ hold inputs plus the outputs of
WRF v4.6.1's mcica_subcol_lw / mcica_subcol_sw compiled unmodified
(tools/rrtmg_wrf461_oracle/mcica_fixture_lw.F90 / _sw.F90; regenerate
with build.sh + make_mcica_fixtures.py).  Every float32 output must match
the Fortran exactly (max_ulp 0, asserted as integer bit equality).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gpuwm.core import rrtmg_mcica as M


FIXTURE_DIR = Path(__file__).resolve().parent / "data" / "rrtmg_mcica"
FIXTURES = sorted(FIXTURE_DIR.glob("mcica_*.npz"))
LW_OUTPUTS = ("cldfmcl", "ciwpmcl", "clwpmcl", "cswpmcl", "taucmcl",
              "reicmcl", "relqmcl", "resnmcl")
SW_OUTPUTS = LW_OUTPUTS + ("ssacmcl", "asmcmcl", "fsfcmcl")


def _load(path: Path):
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    return data


def _fixture_kwargs(data):
    side = bytes(data["side"]).decode()
    kwargs = dict(
        iplon=1,
        ncol=int(data["ncol"]), nlay=int(data["nlay"]),
        icld=int(data["icld"]), permuteseed=int(data["permuteseed"]),
        irng=int(data["irng"]),
        play=data["in_play"], cldfrac=data["in_cldfrac"],
        ciwp=data["in_ciwp"], clwp=data["in_clwp"], cswp=data["in_cswp"],
        rei=data["in_rei"], rel=data["in_rel"], res=data["in_res"],
        tauc=data["in_tauc"], hgt=data["in_hgt"],
        idcor=int(data["idcor"]), juldat=int(data["juldat"]),
        lat=np.float32(data["lat"]))
    if side == "sw":
        kwargs.update(ssac=data["in_ssac"], asmc=data["in_asmc"],
                      fsfc=data["in_fsfc"])
    return side, kwargs


def _run(data):
    side, kwargs = _fixture_kwargs(data)
    if side == "sw":
        return side, M.generate_sw_subcolumns(**kwargs)
    return side, M.generate_lw_subcolumns(**kwargs)


def test_fixture_inventory_covers_the_contract():
    assert len(FIXTURES) == 16, "expected 16 committed oracle fixtures"
    seen = {"lw": set(), "sw": set()}
    idcors = set()
    juldat_branches = set()
    lats = set()
    seeds = {"lw": set(), "sw": set()}
    tiny = full = False
    for path in FIXTURES:
        data = _load(path)
        side = bytes(data["side"]).decode()
        seen[side].add(int(data["icld"]))
        idcors.add(int(data["idcor"]))
        juldat_branches.add(int(data["juldat"]) > 181)
        lats.add(float(data["lat"]))
        seeds[side].add(int(data["permuteseed"]))
        cf = data["in_cldfrac"]
        tiny |= bool(np.any((cf > 0) & (cf < np.float32(1e-20))))
        full |= bool(np.any(cf == np.float32(1.0)))
    assert seen["lw"] == {1, 2, 3, 4, 5}
    assert seen["sw"] == {1, 2, 3, 4, 5}
    assert idcors == {0, 1}
    assert juldat_branches == {False, True}
    assert any(lat < 0 for lat in lats)
    # WRF's production permute seeds must be exercised on their own side.
    assert M.LW_PERMUTESEED in seeds["lw"]
    assert M.SW_PERMUTESEED in seeds["sw"]
    assert len(seeds["lw"]) > 1 and len(seeds["sw"]) > 1
    assert tiny, "no fixture exercises 0 < cldfrac < cldmin"
    assert full, "no fixture exercises cldfrac == 1"


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_subcolumns_match_wrf_fortran_bit_for_bit(path):
    data = _load(path)
    side, got = _run(data)
    names = SW_OUTPUTS if side == "sw" else LW_OUTPUTS
    assert set(got) == set(names)
    for name in names:
        ref = data[f"out_{name}"]
        value = got[name]
        assert value.dtype == np.float32 and ref.dtype == np.float32
        assert ref.shape == value.shape, name
        same = np.array_equal(ref.view(np.int32), value.view(np.int32))
        if not same:
            bad = int(np.sum(ref.view(np.int32) != value.view(np.int32)))
            raise AssertionError(
                f"{path.stem}/{name}: {bad} of {ref.size} elements differ "
                "from the WRF Fortran")
    # irng is inout in the Fortran and must remain 0 on the kissvec path.
    assert int(data["out_irng_out"]) == 0


def _minimal_lw_kwargs(ncol=2, nlay=6):
    rng = np.random.default_rng(7)
    play = np.linspace(1000.0, 400.0, nlay, dtype=np.float32)[None, :] \
        + rng.uniform(0.0, 0.5, (ncol, nlay)).astype(np.float32)
    play = np.minimum.accumulate(play.astype(np.float32), axis=1)
    return dict(
        iplon=1, ncol=ncol, nlay=nlay, icld=2, permuteseed=150, irng=0,
        play=play,
        cldfrac=np.full((ncol, nlay), 0.4, np.float32),
        ciwp=np.ones((ncol, nlay), np.float32),
        clwp=np.ones((ncol, nlay), np.float32),
        cswp=np.zeros((ncol, nlay), np.float32),
        rei=np.full((ncol, nlay), 25.0, np.float32),
        rel=np.full((ncol, nlay), 10.0, np.float32),
        res=np.full((ncol, nlay), 100.0, np.float32),
        tauc=np.zeros((M.NBNDLW, ncol, nlay), np.float32),
        hgt=np.cumsum(np.full((ncol, nlay), 300.0, np.float32), axis=1),
        idcor=0, juldat=94, lat=np.float32(35.0))


def test_icld_zero_fails_closed_instead_of_returning_undefined_outputs():
    kwargs = _minimal_lw_kwargs()
    kwargs["icld"] = 0
    with pytest.raises(ValueError, match="undefined"):
        M.generate_lw_subcolumns(**kwargs)


@pytest.mark.parametrize("icld", [-1, 6])
def test_icld_out_of_range_fails_closed(icld):
    kwargs = _minimal_lw_kwargs()
    kwargs["icld"] = icld
    with pytest.raises(ValueError, match="INVALID ICLD"):
        M.generate_lw_subcolumns(**kwargs)


def test_mersenne_twister_request_is_rejected_not_substituted():
    kwargs = _minimal_lw_kwargs()
    kwargs["irng"] = 1
    with pytest.raises(NotImplementedError, match="Mersenne"):
        M.generate_lw_subcolumns(**kwargs)


def test_increasing_bottom_pressure_trips_the_fortran_seed_guard():
    kwargs = _minimal_lw_kwargs()
    play = kwargs["play"].copy()
    play[:, 0] = play[:, 1] - np.float32(1.0)     # pmid(:,1) < pmid(:,2)
    kwargs["play"] = play
    with pytest.raises(ValueError, match="KISSVEC SEED"):
        M.generate_lw_subcolumns(**kwargs)


# ---------------------------------------------------------------------------
# CUDA device-twin gates -- icld=2 (maximum-random), the forecast path.
#
# The device generators (gpu_generate_lw_subcolumns /
# gpu_generate_sw_subcolumns, kernels in
# gpuwm/core/kernels/rrtmg_mcica_wrf.cu) are gated bitwise against the
# NumPy port, which the fixture gates above certify against the
# unmodified WRF Fortran -- so device == NumPy here inherits the oracle
# certification transitively.
#
# Every GPU gate is dual-run (5090 standing rule): the device path runs
# twice and both runs must equal the reference bitwise (which implies
# they match each other).  The GPU is SHARED with other lanes, so the
# wide gates size themselves via _need_vram (free-memory skip, ~4 GiB
# lane budget) and the 50,000-column gates run at reduced nlay: the
# frozen (ngpt, ncol, nlay) float32 outputs alone would cost ~9.7 GiB
# at nlay=74 for the LW side.
# ---------------------------------------------------------------------------

try:
    import cupy as cp
    _GPU_OK = cp.cuda.runtime.getDeviceCount() > 0
except Exception:                                  # pragma: no cover
    cp = None
    _GPU_OK = False

gpu_gate = pytest.mark.skipif(not _GPU_OK, reason="no CUDA GPU / cupy")

DUAL_RUNS = 2
_WIDE_NLAY = 74
_DECK_SEED = 20260727


def _gpu_call(side, kwargs):
    fn = (M.gpu_generate_sw_subcolumns if side == "sw"
          else M.gpu_generate_lw_subcolumns)
    return fn(**kwargs)


def _to_host(out):
    return {k: cp.asnumpy(v) for k, v in out.items()}


def _assert_bitwise(got, want, tag):
    assert set(got) == set(want), tag
    for name, ref in want.items():
        val = got[name]
        assert val.dtype == np.float32 and ref.dtype == np.float32
        assert val.shape == ref.shape, (tag, name)
        if not np.array_equal(val.view(np.uint32), ref.view(np.uint32)):
            bad = int(np.sum(val.view(np.uint32) != ref.view(np.uint32)))
            raise AssertionError(
                f"{tag}/{name}: {bad} of {ref.size} device elements "
                "differ from the NumPy port")


def _r512(nbytes):
    return (int(nbytes) + 511) & ~511


def _call_vram_bytes(side, ncol, nlay, chunk):
    """Device bytes of one full gpu_generate_* call with numpy inputs:
    output slabs + passthrough copies + input uploads + the chunk
    transient (mcica_device_vram_bytes) + guard temporaries."""
    ngpt = M.NGPTSW if side == "sw" else M.NGPTLW
    nbnd = M.NBNDSW if side == "sw" else M.NBNDLW
    nout = 8 if side == "sw" else 5
    nband_src = 4 if side == "sw" else 1
    total = nout * _r512(ngpt * ncol * nlay * 4)       # subcolumn outputs
    total += 8 * _r512(ncol * nlay * 4)   # play cldfrac ciwp clwp cswp
    #                                       rei rel res (uploads = the
    #                                       3 passthrough outputs too)
    total += nband_src * _r512(nbnd * ncol * nlay * 4)
    total += 2 * _r512(ncol * 4)                       # pm0/pm1 guard
    total += M.mcica_device_vram_bytes(min(chunk, max(ncol, 1)),
                                       nlay, ngpt)
    return total


def _need_vram(nbytes):
    """Skip when the shared card cannot host this gate right now."""
    free, _total = cp.cuda.runtime.memGetInfo()
    if free < nbytes + 512 * 2**20:
        pytest.skip(f"needs ~{nbytes / 2**30:.2f} GiB free on the "
                    "shared GPU")


def _synthetic_kwargs(side, ncol, nlay, seed):
    """Random-but-valid deck: strictly decreasing play with varied
    fractional parts (plus integer-valued bottom pressures in one
    column -> zero seeds), cldfrac mixing exact 0s, sub-cldmin values,
    near-1 values, exact 1s and fully clear columns."""
    rng = np.random.default_rng(seed)
    nbnd = M.NBNDSW if side == "sw" else M.NBNDLW
    psfc = rng.uniform(935.0, 1035.0, ncol)
    ptop = rng.uniform(30.0, 90.0, ncol)
    w = rng.uniform(0.2, 1.8, (ncol, nlay - 1))
    w = w / w.sum(axis=1, keepdims=True) * (psfc - ptop)[:, None]
    play = np.empty((ncol, nlay), np.float64)
    play[:, 0] = psfc
    play[:, 1:] = psfc[:, None] - np.cumsum(w, axis=1)
    if ncol >= 2:
        play[1, :4] = np.floor(play[1, :4])   # frac 0 -> zero seed words
    play = play.astype(np.float32)

    cf = rng.random((ncol, nlay)).astype(np.float32)      # [0, 1)
    z = rng.random((ncol, nlay))
    cf[z < 0.30] = np.float32(0.0)
    cf[(z >= 0.30) & (z < 0.36)] = np.float32(1.0) - np.float32(1e-7)
    cf[(z >= 0.36) & (z < 0.40)] = np.nextafter(np.float32(1.0),
                                                np.float32(0.0))
    cf[(z >= 0.40) & (z < 0.43)] = np.float32(1.0)
    cf[(z >= 0.43) & (z < 0.46)] = np.float32(1.0e-30)    # < cldmin
    cf[4::7, :] = np.float32(0.0)             # fully clear columns

    def f32(lo, hi, shape):
        return rng.uniform(lo, hi, shape).astype(np.float32)

    kwargs = dict(
        iplon=1, ncol=ncol, nlay=nlay, icld=2,
        permuteseed=(M.SW_PERMUTESEED if side == "sw"
                     else M.LW_PERMUTESEED),
        irng=0, play=play, cldfrac=cf,
        ciwp=f32(0.0, 90.0, (ncol, nlay)),
        clwp=f32(0.0, 220.0, (ncol, nlay)),
        cswp=f32(0.0, 40.0, (ncol, nlay)),
        rei=f32(5.0, 130.0, (ncol, nlay)),
        rel=f32(2.5, 60.0, (ncol, nlay)),
        res=f32(5.0, 130.0, (ncol, nlay)),
        tauc=f32(0.0, 40.0, (nbnd, ncol, nlay)),
        hgt=np.cumsum(f32(50.0, 400.0, (ncol, nlay)),
                      axis=1).astype(np.float32),
        idcor=1, juldat=200, lat=np.float32(-31.5))
    if side == "sw":
        kwargs.update(ssac=f32(0.3, 1.0, (nbnd, ncol, nlay)),
                      asmc=f32(-0.3, 0.95, (nbnd, ncol, nlay)),
                      fsfc=f32(0.0, 1.0, (nbnd, ncol, nlay)))
    return kwargs


def _numpy_run(side, kwargs):
    fn = (M.generate_sw_subcolumns if side == "sw"
          else M.generate_lw_subcolumns)
    return fn(**kwargs)


def _subset_kwargs(kwargs, cols):
    """The same deck restricted to a column subset."""
    out = dict(kwargs)
    out["ncol"] = len(cols)
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray) and v.ndim >= 2:
            out[k] = v[cols] if v.ndim == 2 else v[:, cols]
    return out


def test_numpy_generators_are_column_independent():
    """Seeds and RNG streams are strictly per-column, so the NumPy port
    on a column subset equals the full run's restriction -- this is
    what licenses the sampled comparison at 50,000 columns below."""
    kwargs = _synthetic_kwargs("lw", 37, 12, seed=_DECK_SEED + 37)
    full = _numpy_run("lw", kwargs)
    cols = np.array([0, 3, 4, 11, 20, 36])
    sub = _numpy_run("lw", _subset_kwargs(kwargs, cols))
    want = {k: (v[:, cols, :] if v.ndim == 3 else v[cols]).copy()
            for k, v in full.items()}
    _assert_bitwise(sub, want, "column-subset")


@gpu_gate
def test_gpu_twin_preflight():
    M.mcica_gpu_preflight(force=True)


@gpu_gate
@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_gpu_twin_matches_numpy_on_fixture_cases(path):
    """Gate 1: device == NumPy bitwise on every committed WRF fixture
    whose overlap the device path carries (icld=2); every other fixture
    must fail closed host-side."""
    data = _load(path)
    side, kwargs = _fixture_kwargs(data)
    if int(data["icld"]) != 2:
        with pytest.raises(NotImplementedError, match="icld=2"):
            _gpu_call(side, kwargs)
        return
    _, want = _run(data)
    names = SW_OUTPUTS if side == "sw" else LW_OUTPUTS
    for run in range(DUAL_RUNS):
        got = _to_host(_gpu_call(side, kwargs))
        _assert_bitwise(got, want, f"{path.stem}/run{run}")
        # Direct oracle assertion (independent of the NumPy port): the
        # device outputs must equal the stored WRF Fortran arrays
        # themselves, so this gate does not rest on transitivity through
        # test_subcolumns_match_wrf_fortran_bit_for_bit.
        for name in names:
            ref = data[f"out_{name}"]
            assert np.array_equal(ref.view(np.int32),
                                  got[name].view(np.int32)), \
                f"{path.stem}/{name}/run{run}: device != WRF Fortran"


@gpu_gate
@pytest.mark.parametrize("side", ["lw", "sw"])
@pytest.mark.parametrize("ncol", [1, 7, 333, 4096])
def test_gpu_twin_wide_synthetic_decks(side, ncol):
    """Gate 2 (full-width comparisons): device == NumPy bitwise on
    random-but-valid decks at several widths, dual-run."""
    _need_vram(_call_vram_bytes(side, ncol, _WIDE_NLAY,
                                M.MCICA_DEVICE_COLUMN_CHUNK))
    kwargs = _synthetic_kwargs(side, ncol, _WIDE_NLAY,
                               seed=_DECK_SEED + ncol)
    want = _numpy_run(side, kwargs)
    for run in range(DUAL_RUNS):
        got = _to_host(_gpu_call(side, kwargs))
        _assert_bitwise(got, want, f"{side}/ncol{ncol}/run{run}")
    del got
    cp.get_default_memory_pool().free_all_blocks()


@gpu_gate
@pytest.mark.parametrize("side,nlay", [("lw", 20), ("sw", 16)])
def test_gpu_twin_50000_columns_sampled(side, nlay):
    """Gate 2 (50,000 columns): one device call spanning several
    internal chunks (default chunk 16384 -> 4 chunks with a ragged
    tail), compared bitwise against the NumPy port on a deterministic
    2,500-column sample (licensed by
    test_numpy_generators_are_column_independent).  nlay is reduced so
    the frozen output slabs respect the shared-GPU lane budget."""
    ncol = 50000
    _need_vram(_call_vram_bytes(side, ncol, nlay,
                                M.MCICA_DEVICE_COLUMN_CHUNK))
    kwargs = _synthetic_kwargs(side, ncol, nlay, seed=_DECK_SEED + nlay)
    cols = np.sort(np.random.default_rng(555).choice(
        ncol, size=2500, replace=False))
    want = _numpy_run(side, _subset_kwargs(kwargs, cols))
    cols_d = cp.asarray(cols)
    for run in range(DUAL_RUNS):
        out = _gpu_call(side, kwargs)
        got = {k: cp.asnumpy(v[:, cols_d, :] if v.ndim == 3
                             else v[cols_d]) for k, v in out.items()}
        del out
        _assert_bitwise(got, want, f"{side}/wide50000/run{run}")
    del got
    cp.get_default_memory_pool().free_all_blocks()


@gpu_gate
@pytest.mark.parametrize("side", ["lw", "sw"])
def test_gpu_twin_chunk_invariance(side):
    """Gate 3: the same batch at two chunk sizes (ragged 1000-column
    chunks vs one 4096 chunk) is bitwise identical on the device.  The
    deck is the ncol=4096 deck of the wide gate above (same seed), so
    these outputs are also NumPy-anchored there."""
    ncol = 4096
    _need_vram(2 * _call_vram_bytes(side, ncol, _WIDE_NLAY, ncol))
    kwargs = _synthetic_kwargs(side, ncol, _WIDE_NLAY,
                               seed=_DECK_SEED + ncol)
    for run in range(DUAL_RUNS):
        a = _gpu_call(side, dict(kwargs, ncol_chunk=1000))
        b = _gpu_call(side, dict(kwargs, ncol_chunk=4096))
        for k in a:
            assert bool(cp.array_equal(
                a[k].view(cp.uint32), b[k].view(cp.uint32))), (
                f"{side}/{k}: chunking changed bits (run {run})")
        del a, b
    cp.get_default_memory_pool().free_all_blocks()


@pytest.mark.parametrize("icld", [0, 1, 3, 4, 5, -1, 6])
def test_gpu_device_path_fails_closed_on_other_icld(icld):
    """Gate 4: any icld other than 2 raises host-side BEFORE any cupy
    import, upload or launch (so this runs GPU-less too)."""
    kwargs = _minimal_lw_kwargs()
    kwargs["icld"] = icld
    with pytest.raises(NotImplementedError, match="icld=2"):
        M.gpu_generate_lw_subcolumns(**kwargs)


def test_gpu_device_path_rejects_mersenne_twister():
    kwargs = _minimal_lw_kwargs()
    kwargs["irng"] = 1
    with pytest.raises(NotImplementedError, match="Mersenne"):
        M.gpu_generate_lw_subcolumns(**kwargs)


@gpu_gate
def test_gpu_device_path_seed_guard_and_layer_floor():
    kwargs = _minimal_lw_kwargs()
    play = kwargs["play"].copy()
    play[:, 0] = play[:, 1] - np.float32(1.0)
    with pytest.raises(ValueError, match="KISSVEC SEED"):
        M.gpu_generate_lw_subcolumns(**dict(kwargs, play=play))
    with pytest.raises(ValueError, match="four layers"):
        M.gpu_generate_lw_subcolumns(**_minimal_lw_kwargs(nlay=3))


@gpu_gate
def test_gpu_twin_local_frames():
    """KF_KMAX lesson: bound every kernel's per-thread local frame (a
    frame of F bytes reserves ~F x 1536 x 170 machine-wide at first
    launch on the RTX 5090).  All three kernels keep their state in
    registers; the bound leaves room only for toolchain drift."""
    M.mcica_gpu_preflight(force=True)
    runs = [M.mcica_gpu_local_frame_bytes() for _ in range(DUAL_RUNS)]
    assert runs[0] == runs[1], "local frame query not stable"
    frames = runs[0]
    assert set(frames) == set(M.MCICA_GPU_KERNEL_NAMES)
    for name, nbytes in sorted(frames.items(), key=lambda kv: -kv[1]):
        implied = nbytes * 1536 * 170
        print(f"local frame {name}: {nbytes} B "
              f"(implied reservation {implied / 2**20:.1f} MiB)")
        assert nbytes <= 512, (
            f"{name}: local frame {nbytes} B exceeds bound 512 B")


@gpu_gate
@pytest.mark.parametrize("chunk", [4096, 1024])
def test_gpu_twin_vram_estimate_honesty(chunk):
    """mcica_device_vram_bytes honesty at call level: the analytic
    call estimate (_call_vram_bytes, whose chunk-transient term is
    mcica_device_vram_bytes) bounds the pool-measured peak from above
    and stays within 2x of it, at single- and multi-chunk sizes."""
    ncol, nlay = 4096, _WIDE_NLAY
    _need_vram(_call_vram_bytes("lw", ncol, nlay, chunk))
    kwargs = _synthetic_kwargs("lw", ncol, nlay, seed=_DECK_SEED + ncol)
    kwargs["ncol_chunk"] = chunk
    estimate = _call_vram_bytes("lw", ncol, nlay, chunk)
    pool = cp.get_default_memory_pool()
    M.mcica_gpu_preflight()          # keep probe allocs out of the base
    for run in range(DUAL_RUNS):
        pool.free_all_blocks()
        base = pool.used_bytes()
        peak = [0]

        def probe(col0, nc, peak=peak):
            peak[0] = max(peak[0], pool.used_bytes())

        out = M.gpu_generate_lw_subcolumns(
            **dict(kwargs, _stage_probe=probe))
        measured = peak[0] - base
        print(f"chunk={chunk}: measured {measured / 2**20:.3f} MiB, "
              f"estimate {estimate / 2**20:.3f} MiB")
        assert estimate >= measured >= 0.5 * estimate, (
            f"chunk={chunk} run={run}: measured {measured} vs "
            f"estimate {estimate}")
        del out
    pool.free_all_blocks()
