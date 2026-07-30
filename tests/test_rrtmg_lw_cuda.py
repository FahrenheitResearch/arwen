"""CUDA gates for the RRTMG LW port -- bitwise vs the Fortran fixtures.

Every GPU-measured claim is dual-run (5090 standing rule): each test
executes its kernel path twice over all cases and requires both runs to
match the fixture bitwise (which implies they match each other).

The preflight proves, on the live device, that subnormal float32 results
survive (CuPy's -ftz=true injection is bypassed via direct NVRTC) and
that the device glibc transcriptions match the host ones bitwise.
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
                     state_from_fixture, ulp_report)

from gpuwm.core import rrtmg_lw as port  # noqa: E402

cp = pytest.importorskip("cupy")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(DEFAULT_FIXDIR),
    reason="RRTMG LW oracle fixtures not present "
           "(run tools/rrtmg_wrf461_oracle/lw_build.sh)")

DUAL_RUNS = 2


@pytest.fixture(scope="module")
def C():
    return port_coeffs_from_fixture(load_coeffs_fixture())


@pytest.fixture(scope="module")
def cases():
    return [read_fixture(p) for p in case_paths()]


def _assert(msg):
    assert msg is None, msg


def test_gpu_preflight():
    port.gpu_preflight(force=True)


def test_gpu_setcoef(cases, C):
    for _ in range(DUAL_RUNS):
        for fx in cases:
            nl = int(fx["meta/nlayers"])
            lt = int(fx["setcoef/laytrop"])
            out = port.gpu_setcoef(
                nl, 1, fx["inatm/pavel"], fx["inatm/tavel"],
                fx["inatm/tz"], float(fx["inatm/tbound"]),
                fx["inatm/semiss"], fx["inatm/coldry"], fx["inatm/wkl"],
                fx["inatm/wbrodl"], C)
            assert out["laytrop"] == lt
            for k in ("jp", "jt", "jt1", "indfor", "indminor"):
                assert np.array_equal(out[k], fx[f"setcoef/{k}"]), k
            assert np.array_equal(out["indself"][:lt],
                                  fx["setcoef/indself"][:lt])
            for k in ("planklay", "planklev", "plankbnd", "colh2o",
                      "colco2", "colo3", "coln2o", "colco", "colch4",
                      "colo2", "colbrd", "fac00", "fac01", "fac10",
                      "fac11", "selffac", "forfac", "forfrac",
                      "minorfrac", "scaleminor", "scaleminorn2",
                      "rat_h2oco2", "rat_h2oco2_1"):
                _assert(ulp_report(out[k], fx[f"setcoef/{k}"], k))
            for k in ("selffrac", "rat_h2oo3", "rat_h2oo3_1",
                      "rat_h2on2o", "rat_h2on2o_1", "rat_h2och4",
                      "rat_h2och4_1", "rat_n2oco2", "rat_n2oco2_1"):
                _assert(ulp_report(out[k][:lt],
                                   fx[f"setcoef/{k}"][:lt], k))
            for k in ("rat_o3co2", "rat_o3co2_1"):
                _assert(ulp_report(out[k][lt:],
                                   fx[f"setcoef/{k}"][lt:], k))


def test_gpu_inatm(cases, C):
    for _ in range(DUAL_RUNS):
        for fx in cases:
            nl = int(fx["meta/nlayers"])
            out = port.gpu_inatm(
                1, nl, int(fx["in/icld"]), 10, fx["in/play"],
                fx["in/plev"], fx["in/tlay"], fx["in/tlev"],
                fx["in/tsfc"], fx["in/h2ovmr"], fx["in/o3vmr"],
                fx["in/co2vmr"], fx["in/ch4vmr"], fx["in/n2ovmr"],
                fx["in/o2vmr"], fx["in/cfc11vmr"], fx["in/cfc12vmr"],
                fx["in/cfc22vmr"], fx["in/ccl4vmr"], fx["in/emis"],
                int(fx["in/inflglw"]), int(fx["in/iceflglw"]),
                int(fx["in/liqflglw"]), fx["in/cldfmcl"],
                fx["in/taucmcl"], fx["in/ciwpmcl"], fx["in/clwpmcl"],
                fx["in/cswpmcl"], fx["in/reicmcl"], fx["in/relqmcl"],
                fx["in/resnmcl"], fx["in/tauaer"], C)
            for k in ("coldry", "wbrodl", "wkl", "wx", "pwvcm",
                      "pavel", "pz", "tavel", "tz", "semiss"):
                _assert(ulp_report(out[k], fx[f"inatm/{k}"], k))


def test_gpu_cldprmc(cases, C):
    for _ in range(DUAL_RUNS):
        for fx in cases:
            nl = int(fx["meta/nlayers"])
            ncb, taucmc = port.gpu_cldprmc(
                nl, int(fx["inatm/inflag"]), int(fx["inatm/iceflag"]),
                int(fx["inatm/liqflag"]), fx["inatm/cldfmc"],
                fx["inatm/ciwpmc"], fx["inatm/clwpmc"],
                fx["inatm/cswpmc"], fx["inatm/reicmc"],
                fx["inatm/relqmc"], fx["inatm/resnmc"],
                fx["inatm/taucmc"], C)
            assert ncb == int(fx["cldprmc/ncbands"])
            _assert(ulp_report(taucmc, fx["cldprmc/taucmc"], "taucmc"))


@pytest.mark.parametrize("band", range(1, 17))
def test_gpu_taugb_band(cases, C, band):
    s = band_slice(band)
    for _ in range(DUAL_RUNS):
        for fx in cases:
            st = state_from_fixture(fx)
            taug_d, fracs_d = port.gpu_taugb(band, st, C)
            taug = cp.asnumpy(taug_d)[0]
            fracs = cp.asnumpy(fracs_d)[0]
            _assert(ulp_report(taug[:, s], fx["taumol/taug"][:, s],
                               f"taugb{band}/taug"))
            _assert(ulp_report(fracs[:, s], fx["taumol/fracs"][:, s],
                               f"taugb{band}/fracs"))


def test_gpu_rtrnmc(cases, C):
    for _ in range(DUAL_RUNS):
        for fx in cases:
            nl = int(fx["meta/nlayers"])
            out = port.gpu_rtrnmc(
                nl, 1, 16, 0, fx["inatm/pz"], fx["inatm/semiss"],
                int(fx["cldprmc/ncbands"]), fx["inatm/cldfmc"],
                fx["cldprmc/taucmc"], fx["setcoef/planklay"],
                fx["setcoef/planklev"], fx["setcoef/plankbnd"],
                float(fx["inatm/pwvcm"]), fx["taumol/fracs"],
                fx["taut/taut"], C)
            for k in ("totuflux", "totdflux", "fnet", "htr",
                      "totuclfl", "totdclfl", "fnetc", "htrc"):
                _assert(ulp_report(out[k], fx[f"rtrnmc/{k}"], k))


def test_gpu_end_to_end(cases, C):
    """Full device chain vs the direct Fortran rrtmg_lw outputs."""
    for _ in range(DUAL_RUNS):
        for fx in cases:
            nl = int(fx["meta/nlayers"])
            out = port.gpu_rrtmg_lw(
                1, nl, int(fx["in/icld"]), fx["in/play"], fx["in/plev"],
                fx["in/tlay"], fx["in/tlev"], fx["in/tsfc"],
                fx["in/h2ovmr"], fx["in/o3vmr"], fx["in/co2vmr"],
                fx["in/ch4vmr"], fx["in/n2ovmr"], fx["in/o2vmr"],
                fx["in/cfc11vmr"], fx["in/cfc12vmr"],
                fx["in/cfc22vmr"], fx["in/ccl4vmr"], fx["in/emis"],
                int(fx["in/inflglw"]), int(fx["in/iceflglw"]),
                int(fx["in/liqflglw"]), fx["in/cldfmcl"],
                fx["in/taucmcl"], fx["in/ciwpmcl"], fx["in/clwpmcl"],
                fx["in/cswpmcl"], fx["in/reicmcl"], fx["in/relqmcl"],
                fx["in/resnmcl"], fx["in/tauaer"], C)
            for k in ("uflx", "dflx", "hr", "uflxc", "dflxc", "hrc"):
                _assert(ulp_report(out[k], fx[f"out/{k}"], k))


# ---------------------------------------------------------------------------
# Section 11 gates -- batched multi-column path.
#
# The batched entry shares icld/inflglw/iceflglw/liqflglw across a batch,
# so the deck is grouped by flag tuple and each group is run as one batch;
# together the groups cover the full deck.
# ---------------------------------------------------------------------------

#: All outputs of the batched entry (batched-vs-per-column bit gates).
OUT_KEYS = ("uflx", "dflx", "hr", "uflxc", "dflxc", "hrc",
            "uflxcln", "dflxcln")
#: Outputs with direct Fortran fixture counterparts (oracle gates).
ORACLE_KEYS = ("uflx", "dflx", "hr", "uflxc", "dflxc", "hrc")

_IN_COL = ("play", "plev", "tlay", "tlev", "tsfc", "h2ovmr", "o3vmr",
           "co2vmr", "ch4vmr", "n2ovmr", "o2vmr", "cfc11vmr", "cfc12vmr",
           "cfc22vmr", "ccl4vmr", "emis", "reicmcl", "relqmcl", "resnmcl",
           "tauaer")
_IN_MCICA = ("cldfmcl", "taucmcl", "ciwpmcl", "clwpmcl", "cswpmcl")

#: Per-kernel CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES upper bounds (KF_KMAX
#: lesson: a frame of F bytes reserves ~F x 1536 x 170 machine-wide at
#: first launch on the RTX 5090).  Measured 2026-07-27, cupy 14.0.1
#: NVRTC path on sm_120: rlw_rtrn_march 2048 B (exactly its four
#: 128-float per-thread work arrays atrans/atot/bbugas/bbutot),
#: rlw_cldprmc 64 B (icb[16] table), every other kernel 0 B.  Bounds are
#: those measurements rounded up half a KiB for toolchain drift; a frame
#: regression past them is exactly what this gate must catch.
LOCAL_FRAME_BOUNDS = {"rlw_rtrn_march": 2560, "rlw_cldprmc": 512}
LOCAL_FRAME_DEFAULT_BOUND = 512
#: Machine-wide reservation implied by a frame at max occupancy.
RESIDENT_THREADS = 1536 * 170


def _flag_key(fx):
    return (int(fx["in/icld"]), int(fx["in/inflglw"]),
            int(fx["in/iceflglw"]), int(fx["in/liqflglw"]))


def _deck_groups(cases):
    """Case indices grouped by shared batch flags, deck order kept."""
    groups = {}
    for i, fx in enumerate(cases):
        groups.setdefault(_flag_key(fx), []).append(i)
    return groups


def _group_inputs(cases, idx, rep=1):
    """Concatenated (optionally column-tiled) batch inputs for a group."""
    ins = {}
    for k in _IN_COL:
        a = np.concatenate([np.asarray(cases[i][f"in/{k}"])
                            for i in idx], axis=0)
        ins[k] = np.tile(a, (rep,) + (1,) * (a.ndim - 1)) if rep > 1 else a
    for k in _IN_MCICA:
        a = np.concatenate([np.asarray(cases[i][f"in/{k}"])
                            for i in idx], axis=1)
        ins[k] = np.tile(a, (1, rep, 1)) if rep > 1 else a
    return ins


def _run_batched(cases, idx, C, ins, chunk=None, probe=None,
                 to_device=False):
    fx0 = cases[idx[0]]
    nl = int(fx0["meta/nlayers"])
    key = _flag_key(fx0)
    if to_device:
        ins = {k: cp.asarray(np.ascontiguousarray(
            np.asarray(v, np.float32))) for k, v in ins.items()}
    ncol = ins["tsfc"].shape[0]
    return port.gpu_rrtmg_lw_batched(
        ncol, nl, key[0], ins["play"], ins["plev"], ins["tlay"],
        ins["tlev"], ins["tsfc"], ins["h2ovmr"], ins["o3vmr"],
        ins["co2vmr"], ins["ch4vmr"], ins["n2ovmr"], ins["o2vmr"],
        ins["cfc11vmr"], ins["cfc12vmr"], ins["cfc22vmr"],
        ins["ccl4vmr"], ins["emis"], key[1], key[2], key[3],
        ins["cldfmcl"], ins["taucmcl"], ins["ciwpmcl"], ins["clwpmcl"],
        ins["cswpmcl"], ins["reicmcl"], ins["relqmcl"], ins["resnmcl"],
        ins["tauaer"], C, column_chunk=chunk, _stage_probe=probe)


@pytest.fixture(scope="module")
def percol_ref(cases, C):
    """Per-column gpu_rrtmg_lw over the deck: the batched path's
    reference (itself oracle-gated bitwise by test_gpu_end_to_end)."""
    ref = []
    for fx in cases:
        nl = int(fx["meta/nlayers"])
        ref.append(port.gpu_rrtmg_lw(
            1, nl, int(fx["in/icld"]), fx["in/play"], fx["in/plev"],
            fx["in/tlay"], fx["in/tlev"], fx["in/tsfc"],
            fx["in/h2ovmr"], fx["in/o3vmr"], fx["in/co2vmr"],
            fx["in/ch4vmr"], fx["in/n2ovmr"], fx["in/o2vmr"],
            fx["in/cfc11vmr"], fx["in/cfc12vmr"], fx["in/cfc22vmr"],
            fx["in/ccl4vmr"], fx["in/emis"], int(fx["in/inflglw"]),
            int(fx["in/iceflglw"]), int(fx["in/liqflglw"]),
            fx["in/cldfmcl"], fx["in/taucmcl"], fx["in/ciwpmcl"],
            fx["in/clwpmcl"], fx["in/cswpmcl"], fx["in/reicmcl"],
            fx["in/relqmcl"], fx["in/resnmcl"], fx["in/tauaer"], C))
    return ref


def _group_expected(ref, idx, key):
    return np.concatenate([ref[i][key] for i in idx], axis=0)


def test_gpu_local_frames():
    """Bound every kernel's local frame; report the implied machine-wide
    reservation.  Restructuring is only warranted past ~8 KiB frame AND
    ~2 GiB reservation; the march kernel sits far below both."""
    port.gpu_preflight(force=True)
    runs = [port.gpu_local_frame_bytes() for _ in range(DUAL_RUNS)]
    assert runs[0] == runs[1], "local frame query not stable"
    frames = runs[0]
    assert set(frames) == set(port.LW_GPU_KERNEL_NAMES)
    for name, nbytes in sorted(frames.items(), key=lambda kv: -kv[1]):
        bound = LOCAL_FRAME_BOUNDS.get(name, LOCAL_FRAME_DEFAULT_BOUND)
        implied = nbytes * RESIDENT_THREADS
        print(f"local frame {name}: {nbytes} B "
              f"(implied reservation {implied / 2**20:.1f} MiB)")
        assert nbytes <= bound, (
            f"{name}: local frame {nbytes} B exceeds bound {bound} B "
            f"(implied machine-wide {implied / 2**30:.2f} GiB)")
        assert bound * RESIDENT_THREADS < 2 * 2**30


def test_batched_vram_estimate(cases, C):
    """lw_batched_vram_bytes honesty: estimate >= pool-measured peak >=
    0.5*estimate, at two chunk sizes (single-chunk and multi-chunk)."""
    groups = _deck_groups(cases)
    idx = max(groups.values(), key=len)
    nl = int(cases[idx[0]]["meta/nlayers"])
    ins = _group_inputs(cases, idx)
    pool = cp.get_default_memory_pool()
    for chunk in (len(idx), 32):
        estimate = (port.lw_batched_vram_bytes(
                        min(chunk, len(idx)), nl, ncol_total=len(idx))
                    + port.lw_batched_const_bytes(C))
        for _ in range(DUAL_RUNS):
            pool.free_all_blocks()
            base = pool.used_bytes()
            peak = [0]

            def probe(stage, peak=peak):
                peak[0] = max(peak[0], pool.used_bytes())

            _run_batched(cases, idx, C, ins, chunk=chunk, probe=probe)
            measured = peak[0] - base
            print(f"chunk={chunk}: measured {measured / 2**20:.3f} MiB, "
                  f"estimate {estimate / 2**20:.3f} MiB")
            assert estimate >= measured >= 0.5 * estimate, (
                f"chunk={chunk}: measured {measured} vs "
                f"estimate {estimate}")


@pytest.mark.parametrize("chunk", [179, 64, 7, 1])
def test_batched_vs_percolumn(cases, C, percol_ref, chunk):
    """Full-deck bit equality: batched == per-column gpu_rrtmg_lw for
    every output at several chunk sizes (chunking must be invisible)."""
    groups = _deck_groups(cases)
    for _ in range(DUAL_RUNS):
        for idx in groups.values():
            ins = _group_inputs(cases, idx)
            out = _run_batched(cases, idx, C, ins, chunk=chunk)
            for k in OUT_KEYS:
                _assert(ulp_report(out[k], _group_expected(
                    percol_ref, idx, k), f"chunk{chunk}/{k}"))


def test_batched_device_inputs(cases, C, percol_ref):
    """Device-resident (cupy) inputs take the no-host-round-trip branch
    and must produce the same bits."""
    groups = _deck_groups(cases)
    idx = min((g for g in groups.values() if len(g) >= 2), key=len,
              default=max(groups.values(), key=len))
    ins = _group_inputs(cases, idx)
    for _ in range(DUAL_RUNS):
        out = _run_batched(cases, idx, C, ins, chunk=7, to_device=True)
        for k in OUT_KEYS:
            _assert(ulp_report(out[k], _group_expected(
                percol_ref, idx, k), f"dev/{k}"))


def test_batched_oracle(cases, C):
    """End-to-end oracle gate at batch width: batched outputs vs the
    Fortran fixture outputs, max_ulp 0, all cases."""
    groups = _deck_groups(cases)
    for _ in range(DUAL_RUNS):
        for idx in groups.values():
            ins = _group_inputs(cases, idx)
            out = _run_batched(cases, idx, C, ins)
            for k in ORACLE_KEYS:
                want = np.concatenate(
                    [np.asarray(cases[i][f"out/{k}"]) for i in idx],
                    axis=0)
                _assert(ulp_report(out[k], want, f"batch/{k}"))


def test_batched_wide_determinism(cases, C, percol_ref):
    """Tile the deck columns past 50,000, run at the default chunk, and
    require every replica bitwise identical to its source column's
    per-column result (no width- or placement-dependence anywhere)."""
    wide_min = 50000
    rep = -(-wide_min // len(cases))
    groups = _deck_groups(cases)
    total = rep * len(cases)
    assert total >= wide_min
    for idx in groups.values():
        ins = _group_inputs(cases, idx, rep=rep)
        want = {k: np.tile(_group_expected(percol_ref, idx, k),
                           (rep,) + (1,) * (percol_ref[idx[0]][k].ndim
                                            - 1))
                for k in OUT_KEYS}
        for _ in range(DUAL_RUNS):
            out = _run_batched(cases, idx, C, ins)
            for k in OUT_KEYS:
                _assert(ulp_report(out[k], want[k],
                                   f"wide[{len(idx)}x{rep}]/{k}"))
        del ins, want, out
