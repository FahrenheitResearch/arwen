"""Batched small matrix products with one summation order on every card.

Why this exists
---------------
The LETKF transform (:func:`gpuwm.da.letkf.analyze`, ``_transform_chunk``)
forms, per gridpoint, a handful of products in ensemble space: ``C Yb``
(R x P by P x R), ``U D U^T`` (R x R by R x R) and the matrix-vector and
vector-vector contractions that build the mean and perturbation updates.  On
the device those went through ``@`` and ``einsum``; CuPy evaluates both with
``cupy.matmul``, i.e. cuBLAS batched GEMM, and cuBLAS picks its kernel and its
reduction split from the card it runs on.  Measured on a storm-scale parent
chunk's shape (5,520 gridpoints, 10 members, 200 local observations,
float64): the two square products came out bit-identical on an RTX 5090
(170 multiprocessors) and an RTX 5070 Ti (70), but the four einsum
contractions, which reach cuBLAS as matrix-vector shapes, differed in 49 to
82 % of their values (median 512 ulp of cancelled sums).

The model legs are the same bytes on both cards (a 5.2 h spin-up compared
array by array, and a one-hour free forecast of one analysis compared file
by file, both ways round), so the analysis algebra made the first analysis
differ from card to card in the last bit of about 0.3 % of its increment
values, and a convective cycle grows that: given one noise field, the 5090
tracked the 5070 Ti's cycle to a few parts per million for eleven analyses,
then drifted (median increment-RMS difference 1e-4 at the twelfth analysis,
1e-2 at the twentieth, 3e-2 at the thirtieth), and the scored hour's
footprint rain read 0.31 against 0.20, which looked like a card effect on
the physics.  With this module the two cards (drivers 610.43.02 and
595.91.07) wrote the same increment bytes at every analysis, and an RTX 4090
(sm_89) forms the same products bit for bit.  The ensemble perturbations on
this line are drawn on the host with Philox (:mod:`gpuwm.da.perturb`), so
this algebra was the one card-dependent step of a cycle.

Here every output element is one thread's sequential fused multiply-add over
the contracted index, in index order, with ``__fma_rn`` so the rounding is the
IEEE fused one on every architecture.  Nothing depends on the launch grid, the
multiprocessor count or the library version, so any card that runs this
source gets the same bytes.  It is the device counterpart of what numpy's
per-matrix loops already were: one answer per input.

Scope
-----
Float32 and float64, CuPy only, arbitrary element strides (the transform hands
it transposed and broadcast views without copies).  Sizes are those of an
ensemble analysis -- R up to a few dozen members, P up to a few thousand local
observations -- where one thread per output element is also fast: the
contracted length is short and the batch is large.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

__all__ = ["bgemm", "einsum_fixed_order"]

_SOURCE = r"""
extern "C" __global__ void fixed_order_bgemm_f64(
        const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ c,
        const long long batch, const int m, const int k, const int n,
        const long long a_sg, const long long a_sm, const long long a_sk,
        const long long b_sg, const long long b_sk, const long long b_sn) {
    const long long total = batch * (long long)m * (long long)n;
    const long long mn = (long long)m * (long long)n;
    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += (long long)gridDim.x * blockDim.x) {
        const long long g = t / mn;
        const long long r = t - g * mn;
        const long long i = r / n;
        const long long j = r - i * n;
        const double* pa = a + g * a_sg + i * a_sm;
        const double* pb = b + g * b_sg + j * b_sn;
        double acc = 0.0;
        for (int p = 0; p < k; ++p) {
            acc = __fma_rn(pa[(long long)p * a_sk], pb[(long long)p * b_sk], acc);
        }
        c[t] = acc;
    }
}

extern "C" __global__ void fixed_order_bgemm_f32(
        const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c,
        const long long batch, const int m, const int k, const int n,
        const long long a_sg, const long long a_sm, const long long a_sk,
        const long long b_sg, const long long b_sk, const long long b_sn) {
    const long long total = batch * (long long)m * (long long)n;
    const long long mn = (long long)m * (long long)n;
    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += (long long)gridDim.x * blockDim.x) {
        const long long g = t / mn;
        const long long r = t - g * mn;
        const long long i = r / n;
        const long long j = r - i * n;
        const float* pa = a + g * a_sg + i * a_sm;
        const float* pb = b + g * b_sg + j * b_sn;
        float acc = 0.0f;
        for (int p = 0; p < k; ++p) {
            acc = __fmaf_rn(pa[(long long)p * a_sk], pb[(long long)p * b_sk], acc);
        }
        c[t] = acc;
    }
}
"""

#: Threads per block.  The grid is a fixed function of the output size alone
#: (never of the card), and each output element is one thread's own loop, so
#: neither number can move a result; they are fixed only so a receipt can
#: state the launch.
_THREADS = 256
_MAX_BLOCKS = 65535


@lru_cache(maxsize=None)
def _kernel(dtype_str: str):
    import cupy as cp

    name = {"<f8": "fixed_order_bgemm_f64", "<f4": "fixed_order_bgemm_f32"}[dtype_str]
    return cp.RawKernel(_SOURCE, name, options=("-std=c++17",))


def _strides(x):
    item = x.dtype.itemsize
    return tuple(int(s) // item for s in x.strides)


def bgemm(a, b):
    """``c[g] = a[g] @ b[g]`` for ``a (G, M, K)`` and ``b (G, K, N)``, one summation order.

    Both operands must be CuPy arrays of one float dtype (float32 or float64);
    any element strides, including zero (broadcast) and transposed views.
    Returns a new C-contiguous ``(G, M, N)`` array.
    """
    import cupy as cp

    if a.ndim != 3 or b.ndim != 3:
        raise ValueError(f"bgemm wants (G, M, K) and (G, K, N), got {a.shape} and {b.shape}")
    g, m, k = (int(v) for v in a.shape)
    gb, kb, n = (int(v) for v in b.shape)
    if gb != g or kb != k:
        raise ValueError(f"bgemm shapes do not chain: {a.shape} @ {b.shape}")
    if a.dtype != b.dtype or a.dtype not in (np.float32, np.float64):
        raise TypeError(f"bgemm wants one float32/float64 dtype, got {a.dtype} and {b.dtype}")
    c = cp.empty((g, m, n), dtype=a.dtype)
    total = g * m * n
    if total == 0:
        return c
    if k == 0:
        c.fill(0)
        return c
    blocks = max(1, min(_MAX_BLOCKS, (total + _THREADS - 1) // _THREADS))
    a_sg, a_sm, a_sk = _strides(a)
    b_sg, b_sk, b_sn = _strides(b)
    _kernel(np.dtype(a.dtype).str)(
        (blocks,), (_THREADS,),
        (a, b, c, np.int64(g), np.int32(m), np.int32(k), np.int32(n),
         np.int64(a_sg), np.int64(a_sm), np.int64(a_sk),
         np.int64(b_sg), np.int64(b_sk), np.int64(b_sn)))
    return c


def einsum_fixed_order(spec: str, x, y):
    """The five contractions the LETKF transform writes, through :func:`bgemm`.

    Only these subscripts are accepted -- each is mapped to a batched product
    by hand, so an unlisted one is a refusal, never a silent library call.
    """
    if spec == "grp,gp->gr":      # (G, R, P) x (G, P) -> (G, R)
        return bgemm(x, y[:, :, None])[:, :, 0]
    if spec == "grs,gs->gr":      # (G, R, S) x (G, S) -> (G, R)
        return bgemm(x, y[:, :, None])[:, :, 0]
    if spec == "mg,gm->g":        # (R, G) x (G, R) -> (G,)
        return bgemm(y[:, None, :], x.T[:, :, None])[:, 0, 0]
    if spec == "mg,gmk->kg":      # (R, G) x (G, R, K) -> (K, G)
        return bgemm(x.T[:, None, :], y)[:, 0, :].T
    raise ValueError(f"einsum_fixed_order has no fixed-order mapping for {spec!r}")
