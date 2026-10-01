"""Bitwise McICA gates against the preserved base kernel."""
from pathlib import Path
import hashlib
import numpy as np
import pytest


@pytest.mark.parametrize("nlay", [60, 74, 96, 128])
@pytest.mark.parametrize("ngpt,permuteseed", [(256, 150), (224, 1)])
def test_mcica_original_kernel_edges_and_aliases(nlay, ngpt, permuteseed):
    cp = pytest.importorskip("cupy")
    from gpuwm.core.rrtmgp import mcica_cloud_masks, _mcica_jump_tables
    from gpuwm.verify.npref import np_mcica_maxran_masks
    from gpuwm.core.kernels import _preamble, get_kernel
    rng = np.random.default_rng(903)
    play = np.tile(np.linspace(95000, 1000, nlay, dtype=np.float32), (8, 1))
    play[1] = np.floor(play[1])  # Zero MWC states force the alias fallback.
    play[3, 2] = np.floor(play[3, 2])  # Only the third state aliases.
    play[7, 3] = np.floor(play[7, 3])  # Only the fourth state aliases.
    play[2, :4] = np.array([10.0001, 9.0001, 8.0001, 7.0001], np.float32)
    fractions = np.array([0, 1e-21, 1e-20, 1e-19, 1e-8, .5,
                          np.nextafter(np.float32(1), np.float32(0)), 1], np.float32)
    cf = rng.random((8, nlay), dtype=np.float32)
    cf[0] = 0
    cf[1] = np.resize(fractions, nlay)
    cf[2] = np.resize(fractions[::-1], nlay)
    cf[3, ::2] = 0
    cf[4] = 1e-21
    cf[5] = 1e-20
    cf[6] = 1
    oracle_text = Path(__file__).with_name("rrtmgp_mcica_oracle.cu").read_text()
    assert hashlib.sha256(oracle_text.encode()).hexdigest() == (
        "e59fe155d595c74a8c17619758091633652773a509a56030d3dc1a7d90b11039")
    src = _preamble() + oracle_text
    oracle = cp.RawModule(code=src, options=("-std=c++17",)).get_function("rrtmgp_mcica_maxran")
    dp, dc = cp.asarray(play), cp.asarray(cf)
    expected = cp.empty((8, nlay, ngpt), dtype=cp.bool_)
    _, j1, j2, j3, j4 = _mcica_jump_tables(nlay, ngpt)
    oracle_j2 = cp.ascontiguousarray(j2.reshape(32, ngpt).T).ravel()
    oracle((8,), (ngpt,), (dp, dc, expected, j1, oracle_j2, j3, j4,
           np.int32(8), np.int32(nlay), np.int32(ngpt), np.int32(permuteseed)), shared_mem=nlay * 8)
    got = mcica_cloud_masks(dp, dc, ngpt, permuteseed)
    np.testing.assert_array_equal(cp.asnumpy(got), cp.asnumpy(expected))
    np.testing.assert_array_equal(cp.asnumpy(got), np_mcica_maxran_masks(play, cf, ngpt, permuteseed))
    assert get_kernel("rrtmgp_mcica", "rrtmgp_mcica_maxran").local_size_bytes == 0


def test_mcica_barrett_residues_are_exact():
    cp = pytest.importorskip("cupy")
    from gpuwm.core.kernels import module_source
    extra = r"""
    extern "C" __global__ void check_mod(const unsigned long long* x,
                                        unsigned int* y, int n) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n) {
        y[2*i] = mcica_mod<MCICA_P3>(x[i]);
        y[2*i+1] = mcica_mod<MCICA_P4>(x[i]);
      }
    }
    """
    kernel = cp.RawModule(code=module_source("rrtmgp_mcica") + extra,
                          options=("-std=c++17",)).get_function("check_mod")
    rng = np.random.default_rng(903)
    p3, p4 = 1179647999, 2025259007
    # Both reductions are valid throughout the larger product interval.
    x = np.concatenate([rng.integers(0, p4*p4, 50000, dtype=np.uint64),
                        np.array([0, 1, p3-1, p3, p3+1, p3*p3-1,
                                  p4-1, p4, p4+1, p4*p4-1], np.uint64)])
    got = cp.empty((len(x), 2), cp.uint32)
    kernel(((len(x)+255)//256,), (256,), (cp.asarray(x), got, np.int32(len(x))))
    expected = np.stack([x % p3, x % p4], axis=1).astype(np.uint32)
    np.testing.assert_array_equal(cp.asnumpy(got), expected)


@pytest.mark.parametrize("ngpt,permuteseed", [(256, 150), (224, 1)])
def test_mcica_residue_one_aliases_match_original(ngpt, permuteseed):
    cp = pytest.importorskip("cupy")
    from gpuwm.core.kernels import _preamble
    from gpuwm.core.rrtmgp import mcica_cloud_masks, _mcica_jump_tables
    from gpuwm.verify.npref import np_mcica_maxran_masks
    nlay = 74
    play = np.tile(np.linspace(.005, .0001, nlay, dtype=np.float32), (2, 1))
    play[:, :4] = [1.875, 1.5, 1.25, .0625]
    _, j1, j2, j3, j4 = _mcica_jump_tables(nlay, ngpt)
    for col, (a, modulus, component, table) in enumerate([
            (18000, 1179647999, 2, j3), (30903, 2025259007, 3, j4)]):
        found = False
        for g in range(1, ngpt):
            wanted = pow(65536, permuteseed + g * nlay, modulus)
            pressure = np.float32(wanted / 1e9)
            if not (.0625 < float(pressure) < 1):
                continue
            if int(float(pressure) * 1e9) != wanted:
                continue
            play[col, component] = pressure
            state = wanted
            for _ in range(permuteseed):
                state = a * (state & 65535) + (state >> 16)
            assert (state % modulus) * int(table[g].item()) % modulus == 1
            found = True
            break
        assert found, "no representable pressure for a residue-one alias"
    cf = np.random.default_rng(908).random((2, nlay), dtype=np.float32)
    cf[:, ::3] = 0
    dp, dc = cp.asarray(play), cp.asarray(cf)
    expected = cp.empty((2, nlay, ngpt), cp.bool_)
    legacy = cp.ascontiguousarray(j2.reshape(32, ngpt).T).ravel()
    src = _preamble() + Path(__file__).with_name("rrtmgp_mcica_oracle.cu").read_text()
    oracle = cp.RawModule(code=src, options=("-std=c++17",)).get_function("rrtmgp_mcica_maxran")
    oracle((2,), (ngpt,), (dp, dc, expected, j1, legacy, j3, j4,
           np.int32(2), np.int32(nlay), np.int32(ngpt), np.int32(permuteseed)), shared_mem=nlay * 8)
    got = cp.asnumpy(mcica_cloud_masks(dp, dc, ngpt, permuteseed))
    np.testing.assert_array_equal(got, cp.asnumpy(expected))
    np.testing.assert_array_equal(got, np_mcica_maxran_masks(play, cf, ngpt, permuteseed))
