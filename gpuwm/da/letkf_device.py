"""LETKF on the card, observation-sparse: the whole analysis in three kernels.

Same mathematics as :func:`gpuwm.da.letkf.analyze` (Hunt, Kostelich and
Szunyogh 2007, section 2.3; R-localisation; RTPS/RTPP; the inactive-point
closed form), organised around a different fact about the data.

Why this exists
---------------
``analyze`` gathers a dense ``(points, stencil slots, members)`` block per
chunk and multiplies the slots that hold no observation by zero.  On a
241 x 241 x 49 storm-scale domain with 32 members and five whole-grid
observation batches that block does not fit a sensible budget, so the
cycle took the host-staged route, where every chunk's localisation weights,
neighbour selection and packing run in single-threaded numpy and only the
small transform reaches the card: 23 minutes per analysis at one host core
while the cards sat idle.

Here the observations are compacted ONCE, on the card, to the ones that
exist (their ``Yb``, innovation and squared error), and each batch gets an
index grid mapping a gridpoint to its observation or -1.  One warp per
analysis gridpoint walks the stencil in the host path's own slot order
(batch, then vertical offset, then horizontal offset), evaluates the
Gaspari-Cohn weights only where an observation exists, and accumulates
``C Yb`` and ``C d`` with sequential fused multiply-adds -- the arithmetic
of :mod:`gpuwm.da.fixed_order_gemm` with the zero-weight terms skipped,
which adds nothing to a fused multiply-add chain.  No atomics anywhere: each
output element is owned by one lane, every sum runs in a fixed order, so
the bytes do not depend on chunk size, launch geometry or card.

The R x R factorisation is :mod:`gpuwm.core.jacobi_eigh` (one block per
matrix), ``Pa`` / ``Wa`` / ``wbar`` are :func:`fixed_order_gemm.bgemm`, and a
second warp-per-point kernel applies the transform to every analysis field
of the point, RTPS or RTPP included, and returns the increment together with
the per-point diagnostics the host path's ``_finish`` reports.

The prior is streamed through the card in contiguous gridpoint ranges, so
the card holds the compact observations plus one chunk, never the whole
ensemble: the route runs on a 32 GiB card beside packed members.

Oracle
------
The host path (``analyze`` on numpy) stays the reference.  This route is
deterministic run to run and chunk-invariant; against the host it agrees to
rounding (transcendentals in the weights, LAPACK against Jacobi in the
factorisation, BLAS blocking against sequential sums), which the tests in
``tests/test_letkf_device_gpu.py`` bound per field.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from gpuwm.da.letkf import (LetkfDiagnostics, LetkfError, _batch_reach_box,
                            _geodesic_m, _horizontal_stencil,
                            _vertical_stencil, _validate_obs, gaspari_cohn)

__all__ = ["supported", "analyze_device", "MAX_MEMBERS", "ColumnWithhold",
           "DIAG_BLOCK"]

#: One warp per gridpoint, one lane per member.
MAX_MEMBERS = 32


_SOURCE = r"""
#define FULL 0xffffffffu

__device__ __forceinline__ double gc_scaled(double d, double two_over_cut)
{
    // gpuwm.da.letkf.gaspari_cohn, operation for operation (compiled with
    // fmad off, so no product is contracted into an addition).
    double r = d * two_over_cut;
    r = fmin(r, 2.0);
    double r2 = r * r;
    double r3 = r2 * r;
    double r4 = r2 * r2;
    double r5 = r4 * r;
    double inner = (((-r5) / 4.0 + r4 / 2.0) + r3 * 0.625) - r2 * (5.0 / 3.0);
    inner = inner + 1.0;
    double rs = (r > 0.0) ? r : 1.0;
    double outer = ((((r5 / 12.0 - r4 / 2.0) + r3 * 0.625) + r2 * (5.0 / 3.0))
                    - r * 5.0) + 4.0;
    outer = outer - (2.0 / 3.0) / rs;
    double out = (r <= 1.0) ? inner : ((r < 2.0) ? outer : 0.0);
    return fmax(out, 0.0);
}

// One warp per analysis gridpoint of the chunk; lane r owns row r of
// A = C Yb and element r of C d.
//
// desc (per batch, int64 x 16):
//   0 j0  1 i0  2 nj  3 ni  4 index-grid offset  5 dk offset  6 ndk
//   7 h offset  8 nh  9 wh-table offset
//   10..15 reach box k0 k1 j0 j1 i0 i1 (k0 < 0: the batch observes nothing)
extern "C" __global__ void letkf_accumulate(
    const long long start, const long long G,
    const int nz, const int ny, const int nx,
    const double* __restrict__ z,
    const int nb,
    const long long* __restrict__ desc,
    const double* __restrict__ vscale,
    const int* __restrict__ dks,
    const int* __restrict__ djs,
    const int* __restrict__ dis,
    const double* __restrict__ wh,
    const int* __restrict__ idxg,
    const double* __restrict__ yb,
    const double* __restrict__ dvec,
    const double* __restrict__ err2,
    double* __restrict__ amat,
    double* __restrict__ cvec,
    int* __restrict__ count,
    const long long* __restrict__ pts,
    const int use_pts,
    const int* __restrict__ czone,
    const unsigned long long* __restrict__ zmask,
    const int words)
{
    const int lane = threadIdx.x & 31;
    const long long w = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (w >= G) return;
    // use_pts: the chunk is a list of gridpoints rather than a range.
    const long long p = use_pts ? pts[w] : start + w;
    const long long plane = (long long)ny * nx;
    const int k = (int)(p / plane);
    const long long rem = p - (long long)k * plane;
    const int j = (int)(rem / nx);
    const int i = (int)(rem - (long long)j * nx);
    const long long col = (long long)j * nx + i;
    const double zp = z[p];
    // words > 0: the column's zone names batches this point does not read
    // (a set bit in its row of zmask, batch b at bit b % 64 of word b / 64).
    // A skipped batch adds no term, exactly as if it were not in the list.
    const int zone = (words > 0) ? czone[col] : -1;

    double acc[RR];
#pragma unroll
    for (int c = 0; c < RR; ++c) acc[c] = 0.0;
    double cacc = 0.0;
    int n = 0;

    for (int b = 0; b < nb; ++b) {
        const long long* d = desc + 16 * b;
        if (d[10] < 0) continue;
        if (zone >= 0 && ((zmask[(long long)zone * words + (b >> 6)]
                           >> (b & 63)) & 1ull)) continue;
        if (k < d[10] || k > d[11] || j < d[12] || j > d[13]
                || i < d[14] || i > d[15]) continue;
        const int j0 = (int)d[0], i0 = (int)d[1];
        const int nj = (int)d[2], ni = (int)d[3];
        const long long ioff = d[4];
        const int* dk = dks + d[5];
        const int ndk = (int)d[6];
        const int* dj = djs + d[7];
        const int* di = dis + d[7];
        const int nh = (int)d[8];
        const double* whb = wh + d[9] + col * nh;
        const double vs = vscale[b];
        for (int a = 0; a < ndk; ++a) {
            const int k2 = k + dk[a];
            if (k2 < 0 || k2 >= nz) continue;
            for (int h0 = 0; h0 < nh; h0 += 32) {
                const int h = h0 + lane;
                int o = -1;
                double winv = 0.0;
                if (h < nh) {
                    const int j2 = j + dj[h];
                    const int i2 = i + di[h];
                    if (j2 >= 0 && j2 < ny && i2 >= 0 && i2 < nx) {
                        const int jw = j2 - j0, iw = i2 - i0;
                        if (jw >= 0 && jw < nj && iw >= 0 && iw < ni) {
                            const int oo = idxg[ioff + ((long long)k2 * nj + jw) * ni + iw];
                            if (oo >= 0) {
                                const double whv = whb[h];
                                const double dz = fabs(z[(long long)k2 * plane + (long long)j2 * nx + i2] - zp);
                                const double geo = gc_scaled(dz, vs) * whv;
                                if (geo > 0.0) {
                                    o = oo;
                                    winv = geo / err2[oo];
                                }
                            }
                        }
                    }
                }
                unsigned m = __ballot_sync(FULL, o >= 0);
                while (m) {
                    const int src = __ffs(m) - 1;
                    m &= m - 1u;
                    const int oo = __shfl_sync(FULL, o, src);
                    const double wi = __shfl_sync(FULL, winv, src);
                    const double y = (lane < RR) ? yb[(long long)oo * RR + lane] : 0.0;
                    const double cm = y * wi;
#pragma unroll
                    for (int c = 0; c < RR; ++c)
                        acc[c] = fma(cm, __shfl_sync(FULL, y, c), acc[c]);
                    cacc = fma(cm, dvec[oo], cacc);
                    ++n;
                }
            }
        }
    }
    if (lane < RR) {
        double* row = amat + (w * RR + lane) * RR;
#pragma unroll
        for (int c = 0; c < RR; ++c) row[c] = acc[c];
        cvec[w * RR + lane] = cacc;
    }
    if (lane == 0) count[w] = n;
}

// One warp per gridpoint of the chunk; lane k owns member k.  Applies the
// transform to every field, or the inactive closed form, and writes the
// per-point diagnostics.
//   pri, inc: (F, RR, G); act: (G) row in wa/wbar or -1
//   diag: (F, 4, G) = sigma_b, (mean increment)^2, posterior spread, max|prior|
//   bad:  (F, G) bit 0 prior non-finite, bit 1 increment non-finite
extern "C" __global__ void letkf_apply(
    const long long G, const int F,
    const double* __restrict__ pri,
    const int* __restrict__ act,
    const double* __restrict__ wa,
    const double* __restrict__ wbar,
    const int relax_mode, const double alpha,
    const int identity, const double step,
    double* __restrict__ inc,
    double* __restrict__ diag,
    int* __restrict__ bad)
{
    const int lane = threadIdx.x & 31;
    const long long g = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (g >= G) return;
    const bool live = lane < RR;
    const int a = act[g];
    double wcol[RR];
    double wb = 0.0;
    if (a >= 0) {
#pragma unroll
        for (int m = 0; m < RR; ++m)
            wcol[m] = live ? wa[((long long)a * RR + m) * RR + lane] : 0.0;
        wb = live ? wbar[(long long)a * RR + lane] : 0.0;
    }
    for (int f = 0; f < F; ++f) {
        const long long base = (long long)f * RR * G;
        const double x = live ? pri[base + (long long)lane * G + g] : 0.0;
        double s = 0.0;
        double mx = 0.0;
        int nonfinite = 0;
#pragma unroll
        for (int m = 0; m < RR; ++m) {
            const double t = __shfl_sync(FULL, x, m);
            s = s + t;
            mx = fmax(mx, fabs(t));
            nonfinite |= !isfinite(t);
        }
        const double mean = s / (double)RR;
        const double xb = x - mean;
        double sb2 = 0.0;
#pragma unroll
        for (int m = 0; m < RR; ++m) {
            const double t = __shfl_sync(FULL, xb, m);
            sb2 = sb2 + t * t;
        }
        const double sb = sqrt(sb2 / (double)(RR - 1));
        double out;
        if (a >= 0) {
            double dbar = 0.0;
            double xa = 0.0;
#pragma unroll
            for (int m = 0; m < RR; ++m) {
                const double xm = __shfl_sync(FULL, xb, m);
                dbar = fma(__shfl_sync(FULL, wb, m), xm, dbar);
                xa = fma(xm, wcol[m], xa);
            }
            if (alpha > 0.0) {
                if (relax_mode == 1) {
                    xa = xa * (1.0 - alpha) + xb * alpha;
                } else {
                    double sa2 = 0.0;
#pragma unroll
                    for (int m = 0; m < RR; ++m) {
                        const double t = __shfl_sync(FULL, xa, m);
                        sa2 = sa2 + t * t;
                    }
                    const double sa = sqrt(sa2 / (double)(RR - 1));
                    const double relax = fmax(0.0, (sa > 0.0)
                        ? (alpha * (sb - sa)) / sa + 1.0 : 1.0);
                    xa = xa * relax;
                }
            }
            out = (dbar + xa) - xb;
        } else {
            out = identity ? 0.0 : xb * step;
        }
        if (live) inc[base + (long long)lane * G + g] = out;
        // diagnostics: the host path's _finish, per point
        const double post = x + out;
        double si = 0.0, sp = 0.0;
        int incbad = 0;
#pragma unroll
        for (int m = 0; m < RR; ++m) {
            const double t = __shfl_sync(FULL, out, m);
            si = si + t;
            sp = sp + __shfl_sync(FULL, post, m);
            incbad |= !isfinite(t);
        }
        const double im = si / (double)RR;
        const double pm = sp / (double)RR;
        double sq = 0.0;
#pragma unroll
        for (int m = 0; m < RR; ++m) {
            const double t = __shfl_sync(FULL, post, m) - pm;
            sq = sq + t * t;
        }
        if (lane == 0) {
            const long long db = (long long)f * 4 * G;
            diag[db + g] = sb;
            diag[db + G + g] = im * im;
            diag[db + 2 * G + g] = sqrt(sq / (double)(RR - 1));
            diag[db + 3 * G + g] = mx;
            bad[(long long)f * G + g] = nonfinite | (incbad << 1);
        }
    }
}

// Row r of x (rows x g, row-major): out[r * nblk + k] is the sum of
// elements [k * B, min((k + 1) * B, g)), added one at a time in point
// order.  Sequential on purpose: the sum of a block depends on nothing but
// the block, so diagnostics summed block by block do not depend on how the
// grid was cut into chunks or onto how many cards (chunks are whole
// blocks).
extern "C" __global__ void block_sums(
    const double* __restrict__ x, const long long rows, const long long g,
    const int B, const long long nblk, double* __restrict__ out)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows * nblk) return;
    const long long r = t / nblk;
    const long long k = t - r * nblk;
    const long long lo = k * B;
    const long long hi = (lo + B < g) ? lo + B : g;
    const double* row = x + r * g;
    double s = 0.0;
    for (long long e = lo; e < hi; ++e) s = s + row[e];
    out[t] = s;
}
"""


@lru_cache(maxsize=None)
def _module(members: int, device: int = 0):
    """The kernels for ``members``, loaded on card ``device``.

    Keyed by card as well: a module is loaded into one CUDA context, and a
    multi-card analysis launches the same kernels on every card.
    """
    import cupy as cp

    with cp.cuda.Device(int(device)):
        module = cp.RawModule(code=_SOURCE.replace("RR", str(int(members))),
                              options=("--fmad=false", "-std=c++14"))
        module.compile()
    return module


def supported(members: int, solve_dtype) -> bool:
    """Whether this route takes the problem; the caller falls back otherwise."""
    from gpuwm.core.jacobi_eigh import supported as jacobi_supported

    return (2 <= int(members) <= MAX_MEMBERS
            and np.dtype(solve_dtype) == np.float64
            and jacobi_supported(int(members), np.float64))


def _flat_member_view(array, members: int):
    """``(R, npts)`` view of one prior field without copying a contiguous one."""
    a = array
    if hasattr(a, "reshape"):
        flat = a.reshape(members, -1)
        return flat
    return np.asarray(a).reshape(members, -1)


#: Gridpoints per diagnostics block.  The per-point spreads and increment
#: sizes are summed block by block, each block sequentially, and automatic
#: chunks are whole blocks, so the receipt's mean spreads are the same
#: bytes on one card or eight and for every chunk size the card's free
#: memory picks.
DIAG_BLOCK = 4096


@dataclass(frozen=True)
class ColumnWithhold:
    """Batches some columns analyse ``fields`` without.

    ``column_zone`` is ``(ny * nx,)`` int32, the zone of each column or -1
    where nothing is withheld; ``zones[z]`` names the batches zone ``z``
    does not read.  At a point of a zoned column the filter solves
    ``fields`` again on every batch except those, in the same pass over
    the grid, and those fields' increments there are that solve's: the
    whole-domain solve without the zone's batches, at that point, since a
    batch the point does not read adds no term to anything the point
    computes.  :func:`gpuwm.da.velocity_dispersion.withhold` states the
    rule and is the host route of the same result.
    """

    fields: tuple
    column_zone: object
    zones: tuple
    #: Per zone, the fraction of the joint solve kept there:
    #: ``keep * joint + (1 - keep) * withheld``.  ``None``: 0 everywhere.
    keep: tuple | None = None


class _Shared:
    """Observation-side arrays one card needs, built once and copied."""

    NAMES = ("zflat", "desc", "vscale", "dks", "djs", "dis", "whs", "idxg",
             "yb", "dvec", "err2", "czone", "zmask", "zkeep")

    def __init__(self, **arrays):
        for name in self.NAMES:
            setattr(self, name, arrays.get(name))

    def to_device(self, device):
        import cupy as cp

        out = {}
        with cp.cuda.Device(int(device)):
            for name in self.NAMES:
                value = getattr(self, name)
                if value is None:
                    out[name] = None
                    continue
                host = value.get() if hasattr(value, "get") else value
                out[name] = cp.asarray(host)
            cp.cuda.runtime.deviceSynchronize()
        return _Shared(**out)


def _visible_devices():
    import cupy as cp

    return tuple(range(int(cp.cuda.runtime.getDeviceCount())))


def _transform(cp, amat, cvec, count, *, r, scale, ident):
    """Active points of a chunk -> ``(act, wa, wbar, na, max_local, sweeps)``.

    The Jacobi factorisation runs one block per matrix and the products
    one fixed order per element, so every point's transform is its own
    and does not depend on which other points share its batch.
    """
    from gpuwm.core.jacobi_eigh import JacobiEighError, batched_eigh
    from gpuwm.da.fixed_order_gemm import bgemm, einsum_fixed_order

    g = int(count.size)
    act = cp.full(g, -1, dtype=cp.int32)
    active = cp.nonzero(count > 0)[0]
    na = int(active.size)
    timing = {"eigen": 0.0, "products": 0.0}
    if not na:
        return (act, cp.zeros((1, r, r), dtype=np.float64),
                cp.zeros((1, r), dtype=np.float64), 0, 0, 0, timing)
    max_local = int(count.max())
    a = amat[active] + scale * ident[None]
    a = (a + cp.swapaxes(a, 1, 2)) * 0.5
    cp.cuda.Stream.null.synchronize()
    te = time.perf_counter()
    try:
        evals, evecs, sweeps = batched_eigh(a, return_sweeps=True)
    except JacobiEighError as exc:
        raise LetkfError(
            "this project's batched Jacobi eigensolver refused the"
            f" localised LETKF matrix (R-1)I/rho + C Yb: {exc}"
            "  No analysis was produced; do not treat the prior as"
            " one.") from exc
    del a
    if not bool(cp.all(evals > 0)):
        raise LetkfError(
            "the localised LETKF matrix (R-1)I/rho + C Yb came back"
            " with a non-positive eigenvalue, which it cannot have"
            " analytically.  Suspect non-finite simulated"
            " observations or an observation error small enough to"
            " overflow 1/err^2; smallest eigenvalue"
            f" {float(evals.min())!r}.")
    tp = time.perf_counter()
    timing["eigen"] = tp - te
    inv = 1.0 / evals
    vt = cp.swapaxes(evecs, 1, 2)
    pa = bgemm(evecs * inv[:, None, :], vt)
    rt = cp.sqrt(inv * float(r - 1))
    wa = bgemm(evecs * rt[:, None, :], vt)
    del evecs, vt, evals, inv, rt
    wbar = einsum_fixed_order("grs,gs->gr", pa, cvec[active])
    del pa
    cp.cuda.Stream.null.synchronize()
    timing["products"] = time.perf_counter() - tp
    act[active] = cp.arange(na, dtype=cp.int32)
    return act, wa, wbar, na, max_local, int(sweeps), timing


def _concat_releasing(cp, parts):
    """``cp.concatenate(parts)`` holding at most the result plus one part:
    each part is dropped from ``parts`` once copied (weight tables of a
    wide conventional radius are gigabytes each)."""
    total = sum(int(p.size) for p in parts)
    out = cp.empty(total, dtype=parts[0].dtype if parts else np.float64)
    at = 0
    while parts:
        part = parts.pop(0)
        out[at:at + int(part.size)] = part
        at += int(part.size)
        del part
    return out


#: Elements per row block of a horizontal weight-table build (about
#: 256 MB per float64 temporary).
_WH_BLOCK_ELEMENTS = 32 * 1024 * 1024


def _point_batch_on_card(cp, o, points, members, shape, work_dtype):
    """A point batch's arrays on the card, checked as the dense ones are.

    Returns ``(name, sel, values, errors, simulated, mask, loc, window)``:
    ``sel`` the flat gridpoint indices, ``values``/``errors`` ``(n,)``,
    ``simulated`` ``(R, n)``, ``mask`` the dense boolean grid the reach box
    reads (one byte per gridpoint).
    """
    from gpuwm.da.letkf import Localization

    tag = f"observation batch {o.name!r}"
    if getattr(o, "window", None) is not None:
        raise LetkfError(f"{tag} is a point batch with a window; points are "
                         "whole-grid indices, so a window would misplace "
                         "every one of them")
    sel = cp.asarray(np.asarray(points.flat_index, np.int64))
    n = int(sel.size)
    values = cp.asarray(np.asarray(points.values, np.float64)).astype(work_dtype)
    errors = cp.asarray(np.asarray(points.errors, np.float64)).astype(work_dtype)
    sim = cp.asarray(np.asarray(points.simulated, np.float64)).astype(work_dtype)
    if tuple(sim.shape) != (members, n):
        raise LetkfError(f"{tag} simulated has shape {tuple(sim.shape)}, "
                         f"expected {(members, n)}")
    if n:
        if not bool(cp.all(cp.isfinite(values))):
            raise LetkfError(f"{tag} has non-finite values")
        if not bool(cp.all(cp.isfinite(errors))) or not bool(
                cp.all(errors > 0)):
            raise LetkfError(f"{tag} has observation errors that are not "
                             "finite and positive")
        if not bool(cp.all(cp.isfinite(sim))):
            raise LetkfError(f"{tag} simulated H(x_k) is non-finite")
    loc = o.localization
    if loc is not None and not isinstance(loc, Localization):
        raise LetkfError(f"{tag} localization must be a Localization or None")
    mask = cp.zeros(shape, dtype=bool)
    mask.reshape(-1)[sel] = True
    return o.name, sel, values, errors, sim, mask, loc, None


def analyze_device(prior, obs, grid, config, diagnostics=None, *,
                   progress=None, chunk_points: int | None = None,
                   devices=None, withhold: ColumnWithhold | None = None,
                   chunk_hook=None):
    """Observation-sparse device LETKF; same contract as :func:`letkf.analyze`.

    ``prior`` and the observation arrays may be numpy or cupy.  Returns
    numpy increments ``{field: (R, nz, ny, nx)}`` in the prior's float
    dtype, as the host-staged route does, so a caller cannot tell the routes
    apart by type.

    ``devices`` lists the cards the chunks run on (``"all"`` for every
    visible card); ``None`` keeps the current card alone.  The observations
    are compacted once, on the current card, and copied to the others; the
    grid's chunks are then taken by whichever card is free.  Every point's
    increment is computed by one warp from its own inputs in a fixed order,
    so the bytes do not depend on the cards or on which card took which
    chunk.  A prior that already lives on a card keeps the analysis there.

    ``withhold`` (:class:`ColumnWithhold`) re-solves some fields at some
    columns without some batches inside the same pass.  ``chunk_hook``, when
    given, is called as ``chunk_hook.chunk(fields, pri, inc, start, stop)``
    on each chunk's card, after the transform and the withheld solves and
    before the increments leave the card (``pri`` and ``inc`` are
    ``(F, R, points)`` float64 there; it may rewrite ``inc``), and once as
    ``chunk_hook.finish(results)`` with its returns in grid order; that
    result is ``diagnostics.chunk_hook_result``.
    """
    import threading

    import cupy as cp

    if diagnostics is None:
        diagnostics = LetkfDiagnostics()
    t_enter = time.perf_counter()
    fields = tuple(config.analysis_fields)
    missing = [f for f in fields if f not in prior]
    if missing:
        raise LetkfError(
            f"analysis_fields not present in the prior: {missing!r}."
            f"  Prior has {sorted(prior)!r}.")
    shapes = {f: tuple(prior[f].shape) for f in fields}
    if len(set(shapes.values())) != 1:
        raise LetkfError(
            f"analysis fields disagree on shape: {shapes!r}.  Every field"
            " must be (members, nz, ny, nx) on the same grid.")
    full = shapes[fields[0]]
    if len(full) != 4:
        raise LetkfError(
            f"prior fields must be 4-D (members, nz, ny, nx), got {full}.")
    members = int(full[0])
    if members < 2:
        raise LetkfError(
            f"LETKF needs at least 2 ensemble members, got {members}."
            "  A one-member 'ensemble' has no covariance to update with.")
    solve_dtype = np.dtype(config.solve_dtype)
    if not supported(members, solve_dtype):
        raise LetkfError(
            f"the observation-sparse device route takes 2..{MAX_MEMBERS}"
            f" members in float64; this analysis has R={members} in"
            f" {solve_dtype.name}")
    work_dtype = np.dtype(prior[fields[0]].dtype)
    if work_dtype.kind != "f":
        work_dtype = np.dtype(np.float64)
    nz, ny, nx = (int(v) for v in full[1:])
    if nz != grid.nz:
        raise LetkfError(
            f"prior has {nz} levels but GridGeometry.heights_m has {grid.nz}.")
    if grid.geodesic and tuple(grid.lat_deg.shape) != (ny, nx):
        raise LetkfError(
            f"prior is on a ({ny}, {nx}) horizontal grid but"
            f" GridGeometry.lat_deg is {tuple(grid.lat_deg.shape)}.")
    npts = nz * ny * nx
    plane = ny * nx
    diagnostics.members = members
    diagnostics.grid_shape = (nz, ny, nx)
    diagnostics.total_points = npts
    diagnostics.prior_inflation = float(config.prior_inflation)
    diagnostics.rtps_alpha = float(config.rtps_alpha)
    diagnostics.relaxation = str(config.relaxation)
    diagnostics.eigensolver = "jacobi"
    diagnostics.matmul = "fixed-order"
    diagnostics.host_staging = False

    rho = float(config.prior_inflation)
    inactive_scale = 1.0 if rho == 1.0 else (
        (1.0 - float(config.rtps_alpha)) * math.sqrt(rho)
        + float(config.rtps_alpha))
    if config.relaxation == "rtps":
        inactive_scale = max(0.0, inactive_scale)
    identity = inactive_scale == 1.0
    step = float(work_dtype.type(inactive_scale - 1.0))

    flat_prior = [_flat_member_view(prior[f], members) for f in fields]
    on_device = [type(a).__module__.split(".")[0] == "cupy" for a in flat_prior]
    primary = int(cp.cuda.runtime.getDevice())
    if devices is None or any(on_device):
        cards = (primary,)
    else:
        cards = tuple(int(d) for d in (
            _visible_devices() if devices == "all" else devices))
        if not cards:
            cards = (primary,)

    # ---- compact the observations, batch by batch, on the card ---------
    zflat = cp.asarray(grid.height_field(ny, nx), dtype=np.float64).reshape(-1)
    if grid.geodesic:
        latflat = cp.asarray(np.radians(grid.lat_deg).reshape(-1),
                             dtype=np.float64)
        lonflat = cp.asarray(np.radians(grid.lon_deg).reshape(-1),
                             dtype=np.float64)
    else:
        cols = np.arange(plane, dtype=np.float64)
        xflat = cp.asarray((cols % nx) * float(grid.dx_m))
        yflat = cp.asarray((cols // nx) * float(grid.dy_m))

    desc_rows, vscales = [], []
    dk_all, dj_all, di_all = [], [], []
    wh_tables, wh_keys = [], {}
    yb_parts, d_parts, e_parts, idx_parts = [], [], [], []
    batch_names = []
    dk_off = h_off = wh_off = idx_off = 0
    nobs = 0
    total_slots = 0
    for o in obs:
        points = getattr(o, "points", None)
        if points is not None:
            # A point batch (gpuwm.da.letkf.PointSet): its observations are
            # already compact, so the dense (R, nz, ny, nx) H(x) is never
            # built.  Same refusals as the dense check, on the points.
            (name, sel, sv, se, ss, mask, loc, window) = _point_batch_on_card(
                cp, o, points, members, (nz, ny, nx), work_dtype)
        else:
            (name, values, err, sim, mask, loc, window), = _validate_obs(
                cp, [o], members, (nz, ny, nx), work_dtype)
        batch_names.append(str(name))
        spec = loc if loc is not None else config.localization
        dj, di = _horizontal_stencil(spec, grid, nx, ny)
        dk = _vertical_stencil(spec, grid)
        if int(dj.size) <= 1:
            raise LetkfError(
                f"observation batch {name!r} has a horizontal localisation"
                f" radius of {spec.horizontal_m!r} m, which does not reach"
                f" even one neighbouring column at dx={grid.dx_m!r} m,"
                f" dy={grid.dy_m!r} m.  No observation could influence any"
                " gridpoint but its own.  Check the units -- radii are"
                " metres here -- or widen the radius.")
        total_slots += int(dj.size) * int(dk.size)
        wj0, wi0 = (window[0], window[2]) if window is not None else (0, 0)
        wnj, wni = int(mask.shape[1]), int(mask.shape[2])
        mflat = mask.reshape(-1)
        if points is not None:
            err2_sel = se.astype(solve_dtype) ** 2
        else:
            err2_full = err.reshape(-1).astype(solve_dtype) ** 2
            sel = cp.nonzero(mflat)[0]
            err2_sel = err2_full[sel]
        nb_obs = int(sel.size)
        if nb_obs and not bool(cp.all(err2_sel > 0)):
            raise LetkfError(
                f"observation batch {name!r} has an observation error whose"
                f" SQUARE underflows to zero in the {config.solve_dtype}"
                " solve, although the error itself is finite and positive."
                "  R^-1 = weight/sigma^2 is then a division by zero and the"
                " transform is meaningless.  Use solve_dtype='float64'"
                " (the default), or express the observation in units that"
                " do not need a standard deviation this small.")
        reach = _batch_reach_box(mflat, nz, wnj, wni, dk, dj, di, cp,
                                 j0=wj0, i0=wi0, grid_ny=ny, grid_nx=nx)
        grid_idx = cp.full(nz * wnj * wni, -1, dtype=cp.int32)
        if nb_obs:
            s = (ss if points is not None
                 else sim.reshape(members, -1)[:, sel]).astype(solve_dtype)
            # Member mean in member order, one add at a time: the sum the
            # host path's ``s.mean(axis=0)`` forms, divided once by R.
            sbar = s[0].copy()
            for m in range(1, members):
                sbar = sbar + s[m]
            sbar = sbar / float(members)
            yb_parts.append(cp.ascontiguousarray((s - sbar[None]).T))
            d_parts.append((sv if points is not None
                            else values.reshape(-1)[sel]).astype(solve_dtype)
                           - sbar)
            e_parts.append(err2_sel)
            grid_idx[sel] = (cp.arange(nb_obs, dtype=cp.int32)
                             + np.int32(nobs))
            del s, sbar
        idx_parts.append(grid_idx)
        key = (tuple(dj.tolist()), tuple(di.tolist()), float(spec.horizontal_m))
        if key not in wh_keys:
            # Built in row blocks: the whole-grid temporaries (index,
            # neighbour and distance arrays, each plane x stencil) are five
            # times the table, which for a 200 km conventional radius at
            # 9 km CONUS (about 1,500 stencil columns) is 13 GB beside a
            # 2.5 GB table, and ran a 32 GB card out of memory.  Each row
            # is computed by the same expressions, so the bytes are those
            # of the one-shot build.
            nh_cols = int(dj.size)
            table = cp.empty((plane, nh_cols), dtype=np.float64)
            djd, did = cp.asarray(dj), cp.asarray(di)
            block = max(1, int(_WH_BLOCK_ELEMENTS // max(nh_cols, 1)))
            for r0 in range(0, plane, block):
                r1 = min(plane, r0 + block)
                cols = cp.arange(r0, r1, dtype=cp.int64)
                jj = cols // nx
                ii = cols - jj * nx
                j2 = cp.clip(jj[:, None] + djd[None, :], 0, ny - 1)
                i2 = cp.clip(ii[:, None] + did[None, :], 0, nx - 1)
                ccol = (jj * nx + ii)[:, None]
                col2 = j2 * nx + i2
                if grid.geodesic:
                    dist = _geodesic_m(latflat[ccol], lonflat[ccol],
                                       latflat[col2], lonflat[col2],
                                       float(grid.earth_radius_m))
                else:
                    dist = cp.hypot(xflat[col2] - xflat[ccol],
                                    yflat[col2] - yflat[ccol])
                table[r0:r1] = gaspari_cohn(dist, float(spec.horizontal_m))
                del dist, j2, i2, col2, ccol, jj, ii, cols
            wh_keys[key] = wh_off
            wh_tables.append(table.reshape(-1))
            wh_off += int(table.size)
            del djd, did
        rbox = (-1,) * 6 if reach is None or nb_obs == 0 else reach
        desc_rows.append([wj0, wi0, wnj, wni, idx_off, dk_off, int(dk.size),
                          h_off, int(dj.size), wh_keys[key], *rbox])
        vscales.append(2.0 / float(spec.vertical_m))
        dk_all.append(dk.astype(np.int32))
        dj_all.append(dj.astype(np.int32))
        di_all.append(di.astype(np.int32))
        dk_off += int(dk.size)
        h_off += int(dj.size)
        idx_off += int(grid_idx.size)
        nobs += nb_obs
        if points is not None:
            del sv, se, ss
        else:
            del values, err, sim, err2_full
        del mask, mflat, err2_sel, sel

    diagnostics.stencil_slots = total_slots
    diagnostics.reachable_stencil_slots = total_slots
    nbatch = len(desc_rows)
    shared = {"zflat": zflat}
    if nbatch:
        shared.update(
            desc=cp.asarray(np.asarray(desc_rows, dtype=np.int64)),
            vscale=cp.asarray(np.asarray(vscales, dtype=np.float64)),
            dks=cp.asarray(np.concatenate(dk_all)),
            djs=cp.asarray(np.concatenate(dj_all)),
            dis=cp.asarray(np.concatenate(di_all)),
            whs=_concat_releasing(cp, wh_tables),
            idxg=cp.concatenate(idx_parts))
    del idx_parts, wh_tables
    if nobs:
        shared.update(yb=cp.concatenate(yb_parts, axis=0),
                      dvec=cp.concatenate(d_parts),
                      err2=cp.concatenate(e_parts))
    else:
        shared.update(yb=cp.zeros((1, members), dtype=np.float64),
                      dvec=cp.zeros(1, dtype=np.float64),
                      err2=cp.ones(1, dtype=np.float64))
    del yb_parts, d_parts, e_parts
    diagnostics.observations_compacted = nobs

    # ---- the withheld solves: zones as batch bit masks -----------------
    # ``withhold`` is one ColumnWithhold or several (one per field set, no
    # field in two of them, so each group's increments are its own).  Zone
    # ids are global across groups; ``czone`` holds one row per group.
    groups = []
    words = 0
    if withhold is not None:
        plans = ([withhold] if isinstance(withhold, ColumnWithhold)
                 else list(withhold))
        plans = [plan for plan in plans if len(plan.zones)]
        seen_fields = set()
        for plan in plans:
            unknown = [f for f in plan.fields if f not in fields]
            if unknown:
                raise LetkfError(
                    f"withheld fields {unknown!r} are not analysis fields "
                    f"{fields!r}")
            if seen_fields & set(plan.fields):
                raise LetkfError(
                    "two withheld groups name the same field "
                    f"{sorted(seen_fields & set(plan.fields))!r}; each "
                    "field's withheld increment must come from one group")
            seen_fields |= set(plan.fields)
        if plans:
            position = {name: b for b, name in enumerate(batch_names)}
            if len(position) != len(batch_names):
                raise LetkfError(
                    "two observation batches share a name, so a withheld "
                    "zone could not say which of them it does not read")
            words = max(1, (nbatch + 63) // 64)
            total_zones = sum(len(plan.zones) for plan in plans)
            zmask = np.zeros((total_zones, words), dtype=np.uint64)
            zkeep = np.zeros(total_zones, dtype=np.float64)
            czone = np.full((len(plans), plane), -1, dtype=np.int32)
            base_zone = 0
            for g, plan in enumerate(plans):
                for z, names in enumerate(plan.zones):
                    for name in names:
                        if name not in position:
                            raise LetkfError(
                                f"withheld batch {name!r} is not one of this "
                                "analysis's batches")
                        b = position[name]
                        zmask[base_zone + z, b // 64] |= (
                            np.uint64(1) << np.uint64(b % 64))
                column_zone = np.asarray(plan.column_zone,
                                         dtype=np.int32).reshape(-1)
                if column_zone.size != plane:
                    raise LetkfError(
                        f"withheld column zones cover {column_zone.size} "
                        f"columns; the grid has {plane}")
                czone[g] = np.where(column_zone >= 0,
                                    column_zone + base_zone, -1)
                if plan.keep is not None:
                    if len(plan.keep) != len(plan.zones):
                        raise LetkfError(
                            f"{len(plan.keep)} keeps for {len(plan.zones)} "
                            "withheld zones")
                    zkeep[base_zone:base_zone + len(plan.zones)] = [
                        float(k) for k in plan.keep]
                held = tuple(f for f in fields if f in tuple(plan.fields))
                groups.append({"fields": held,
                               "index": [fields.index(f) for f in held],
                               "zones": len(plan.zones),
                               "blend": bool(plan.keep is not None and any(
                                   float(k) for k in plan.keep))})
                base_zone += len(plan.zones)
            shared.update(czone=cp.asarray(czone), zmask=cp.asarray(zmask),
                          zkeep=cp.asarray(zkeep))
    held_fields = tuple(f for group in groups for f in group["fields"])
    if not held_fields:
        words = 0
    base = _Shared(**shared)

    # ---- chunk sizing --------------------------------------------------
    F = len(fields)
    r = members
    # Per gridpoint of a chunk: A and its symmetrised copy, eigenvectors,
    # Pa, Wa and the bgemm temporaries (8 R x R), the prior and increment
    # chunk (2 F R), the C d / wbar vectors and the diagnostics.
    per_point = 8 * (8 * r * r + 2 * F * r + 4 * r + 5 * F + 4)
    explicit = chunk_points if chunk_points is not None else config.chunk_points
    copies = {primary: base}
    others = sorted({d for d in cards if d != primary})
    if others:
        from concurrent.futures import ThreadPoolExecutor as _Pool
        with _Pool(max_workers=len(others)) as replicate:
            for device, copy in zip(others, replicate.map(
                    base.to_device, others)):
                copies[device] = copy
    if explicit is not None:
        chunk = int(explicit)
    else:
        budget = int(config.memory_budget_mib * (1 << 20))
        for device in cards:
            try:
                with cp.cuda.Device(device):
                    pool = cp.get_default_memory_pool()
                    pool.free_all_blocks()
                    free_b, _total = cp.cuda.runtime.memGetInfo()
                    budget = min(budget, int(0.8 * (int(free_b)
                                                    + int(pool.free_bytes()))))
            except Exception:
                pass
        chunk = max(1, budget // per_point)
        # Whole diagnostics blocks, so the block sums do not depend on it.
        if chunk >= DIAG_BLOCK:
            chunk -= chunk % DIAG_BLOCK
    chunk = max(1, min(npts, chunk))
    diagnostics.chunk_points = chunk
    diagnostics.chunk_points_initial = chunk
    diagnostics.solve_bytes_per_point = per_point

    scale = float(r - 1) / float(config.prior_inflation)
    alpha = float(solve_dtype.type(config.rtps_alpha))
    relax_mode = 1 if config.relaxation == "rtpp" else 0

    increments = {f: np.empty((r, nz, ny, nx), dtype=work_dtype) for f in fields}
    flat_inc = [increments[f].reshape(r, -1) for f in fields]
    spans = [(lo, min(npts, lo + chunk)) for lo in range(0, npts, chunk)]
    host_prior = not all(on_device)
    cap = F * r * chunk
    threads = 128
    cp.cuda.runtime.deviceSynchronize()
    t_solve = time.perf_counter()
    diagnostics.setup_seconds = t_solve - t_enter

    # ---- the chunk pipeline ---------------------------------------------
    # One host thread per card takes the next chunk of the grid as soon as
    # its card is free.  Each chunk is staged into the card's own pinned
    # buffer and its increments land in their own span of the result, so
    # no two threads ever touch the same bytes; numpy and the copies
    # release the GIL.  Per-chunk results are kept by chunk number and
    # combined in grid order afterwards.
    if chunk_hook is not None and hasattr(chunk_hook, "bind_output_dtype"):
        chunk_hook.bind_output_dtype(work_dtype)
    results = [None] * len(spans)
    lock = threading.Lock()
    cursor = [0]
    done = [0]
    failure = []

    def take():
        with lock:
            if failure or cursor[0] >= len(spans):
                return None
            number = cursor[0]
            cursor[0] += 1
            return number

    def work(device):
        from concurrent.futures import ThreadPoolExecutor
        shared_d = copies[device]
        with cp.cuda.Device(device):
            mod = _module(r, device)
            k_acc = mod.get_function("letkf_accumulate")
            k_apply = mod.get_function("letkf_apply")
            k_blocks = mod.get_function("block_sums")
            ident = cp.eye(r, dtype=np.float64)
            dummy_pts = cp.zeros(1, dtype=cp.int64)
            dummy_zone = cp.zeros(1, dtype=cp.int32)
            dummy_mask = cp.zeros(1, dtype=cp.uint64)
            import cupyx
            in_buf = (cupyx.empty_pinned(cap, dtype=np.float64)
                      if host_prior else None)
            out_bufs = [cupyx.empty_pinned(cap, dtype=np.float64)
                        for _ in range(2)]
            drains = [None, None]
            copier = ThreadPoolExecutor(max_workers=1,
                                        thread_name_prefix=f"letkf-drain-{device}")

            def accumulate(start, g, pts, zoned, group=0):
                amat = cp.empty((g, r, r), dtype=np.float64)
                cvec = cp.empty((g, r), dtype=np.float64)
                count = cp.empty(g, dtype=cp.int32)
                blocks = (g * 32 + threads - 1) // threads
                k_acc((blocks,), (threads,),
                      (np.int64(start), np.int64(g), np.int32(nz),
                       np.int32(ny), np.int32(nx), shared_d.zflat,
                       np.int32(nbatch), shared_d.desc, shared_d.vscale,
                       shared_d.dks, shared_d.djs, shared_d.dis,
                       shared_d.whs, shared_d.idxg, shared_d.yb,
                       shared_d.dvec, shared_d.err2, amat, cvec, count,
                       pts if pts is not None else dummy_pts,
                       np.int32(0 if pts is None else 1),
                       shared_d.czone[group] if zoned else dummy_zone,
                       shared_d.zmask if zoned else dummy_mask,
                       np.int32(words if zoned else 0)))
                return amat, cvec, count

            def apply(pri, act, wa, wbar, g, nf):
                inc = cp.empty((nf, r, g), dtype=np.float64)
                diag = cp.empty((nf, 4, g), dtype=np.float64)
                bad = cp.empty((nf, g), dtype=cp.int32)
                blocks = (g * 32 + threads - 1) // threads
                k_apply((blocks,), (threads,),
                        (np.int64(g), np.int32(nf), pri, act, wa, wbar,
                         np.int32(relax_mode), np.float64(alpha),
                         np.int32(1 if identity else 0), np.float64(step),
                         inc, diag, bad))
                return inc, diag, bad

            try:
                while True:
                    number = take()
                    if number is None:
                        break
                    start, stop = spans[number]
                    g = stop - start
                    out = {"timing": {"stage": 0.0, "weights": 0.0,
                                      "transform": 0.0, "eigen": 0.0,
                                      "products": 0.0, "apply": 0.0,
                                      "withheld": 0.0, "hook": 0.0,
                                      "unstage": 0.0},
                           "device": device}
                    t0 = time.perf_counter()
                    if host_prior:
                        view = in_buf[:F * r * g].reshape(F, r, g)
                        for fi, src in enumerate(flat_prior):
                            if not on_device[fi]:
                                np.copyto(view[fi], src[:, start:stop],
                                          casting="unsafe")
                        pri = cp.asarray(view)
                    else:
                        pri = cp.empty((F, r, g), dtype=np.float64)
                    for fi, src in enumerate(flat_prior):
                        if on_device[fi]:
                            pri[fi] = src[:, start:stop].astype(np.float64)
                    cp.cuda.Stream.null.synchronize()
                    t1 = time.perf_counter()
                    out["timing"]["stage"] = t1 - t0
                    act = cp.full(g, -1, dtype=cp.int32)
                    wa = cp.zeros((1, r, r), dtype=np.float64)
                    wbar = cp.zeros((1, r), dtype=np.float64)
                    na = max_local = sweeps = 0
                    t2 = t1
                    if nobs:
                        amat, cvec, count = accumulate(start, g, None, False)
                        cp.cuda.Stream.null.synchronize()
                        t2 = time.perf_counter()
                        out["timing"]["weights"] = t2 - t1
                        act, wa, wbar, na, max_local, sweeps, tt = _transform(
                            cp, amat, cvec, count, r=r, scale=scale,
                            ident=ident)
                        out["timing"]["eigen"] = tt["eigen"]
                        out["timing"]["products"] = tt["products"]
                        del amat, cvec, count
                    ta = time.perf_counter()
                    inc, diag, bad = apply(pri, act, wa, wbar, g, F)
                    nblk = (g + DIAG_BLOCK - 1) // DIAG_BLOCK
                    sums = cp.empty((F * 4, nblk), dtype=np.float64)
                    k_blocks(((F * 4 * nblk + 127) // 128,), (128,),
                             (diag, np.int64(F * 4), np.int64(g),
                              np.int32(DIAG_BLOCK), np.int64(nblk), sums))
                    out["block_sums"] = cp.asnumpy(sums).reshape(F, 4, nblk)
                    out["maxabs"] = cp.asnumpy(diag[:, 3].max(axis=1))
                    out["maxsig"] = cp.asnumpy(diag[:, 0].max(axis=1))
                    out["prior_bad"] = cp.asnumpy(((bad & 1) != 0).any(axis=1))
                    out["inc_bad"] = cp.asnumpy(((bad & 2) != 0).any(axis=1))
                    del act, wa, wbar, diag, bad, sums
                    cp.cuda.Stream.null.synchronize()
                    t3 = time.perf_counter()
                    out["timing"]["apply"] = t3 - ta
                    out["timing"]["transform"] = t3 - t2
                    out["active"] = na
                    out["max_local"] = max_local
                    out["sweeps"] = sweeps
                    out["solved"] = 1 if na else 0
                    # -- the withheld solves at this chunk's zoned points --
                    out["held_points"] = out["held_active"] = 0
                    for gi, group in enumerate(groups):
                        held_index = group["index"]
                        cols = (cp.arange(start, stop, dtype=cp.int64) % plane)
                        local = cp.nonzero(shared_d.czone[gi][cols] >= 0)[0]
                        del cols
                        gw = int(local.size)
                        if gw:
                            pts = local.astype(cp.int64) + np.int64(start)
                            if nobs:
                                amat, cvec, count = accumulate(
                                    0, gw, pts, True, gi)
                                act_w, wa_w, wbar_w, na_w, ml_w, sw_w, _tt = (
                                    _transform(cp, amat, cvec, count, r=r,
                                               scale=scale, ident=ident))
                                del amat, cvec, count
                                out["max_local"] = max(out["max_local"], ml_w)
                                out["sweeps"] = max(out["sweeps"], sw_w)
                            else:
                                act_w = cp.full(gw, -1, dtype=cp.int32)
                                wa_w = cp.zeros((1, r, r), dtype=np.float64)
                                wbar_w = cp.zeros((1, r), dtype=np.float64)
                                na_w = 0
                            pri_w = cp.ascontiguousarray(
                                pri[held_index][:, :, local])
                            inc_w, _diag_w, bad_w = apply(
                                pri_w, act_w, wa_w, wbar_w, gw,
                                len(held_index))
                            if bool(((bad_w & 2) != 0).any()):
                                which = cp.asnumpy(((bad_w & 2) != 0).any(axis=1))
                                raise LetkfError(
                                    "withheld increment for "
                                    f"{[f for f, b in zip(group['fields'], which) if b]!r}"
                                    " is non-finite.  Every specific guard"
                                    " passed, so this is a genuine numerical"
                                    " failure in the transform -- do not"
                                    " apply this analysis.")
                            if group["blend"]:
                                zk = shared_d.zkeep[shared_d.czone[gi][
                                    (local + np.int64(start)) % plane]][None, :]
                                rest = 1.0 - zk
                            for k, fi in enumerate(held_index):
                                if group["blend"]:
                                    # keep * joint + (1 - keep) * withheld
                                    inc[fi][:, local] = (
                                        zk * inc[fi][:, local]
                                        + rest * inc_w[k])
                                else:
                                    inc[fi][:, local] = inc_w[k]
                            out["held_points"] += gw
                            out["held_active"] += na_w
                            del pri_w, inc_w, _diag_w, bad_w, act_w, wa_w
                            del wbar_w, pts
                        del local
                    if groups:
                        cp.cuda.Stream.null.synchronize()
                    t4 = time.perf_counter()
                    out["timing"]["withheld"] = t4 - t3
                    if chunk_hook is not None:
                        out["hook"] = chunk_hook.chunk(fields, pri, inc,
                                                       start, stop)
                        cp.cuda.Stream.null.synchronize()
                    t5 = time.perf_counter()
                    out["timing"]["hook"] = t5 - t4
                    del pri
                    slot = number % 2
                    if drains[slot] is not None:
                        drains[slot].result()
                    host = out_bufs[slot][:F * r * g].reshape(F, r, g)
                    inc.get(out=host)
                    del inc

                    def drain(view=host, lo=start, hi=stop):
                        for fi in range(F):
                            np.copyto(flat_inc[fi][:, lo:hi], view[fi],
                                      casting="unsafe")
                    drains[slot] = copier.submit(drain)
                    out["timing"]["unstage"] = time.perf_counter() - t5
                    results[number] = out
                    if progress is not None:
                        with lock:
                            done[0] += 1
                            progress({"schema": "gpuwm-da.analysis-progress.v1",
                                      "phase": "solve",
                                      "gridpoints_done": int(sum(
                                          spans[n][1] - spans[n][0]
                                          for n, res in enumerate(results)
                                          if res is not None)),
                                      "gridpoints_total": int(npts),
                                      "chunks": done[0],
                                      "active_points": int(sum(
                                          res["active"] for res in results
                                          if res is not None)),
                                      "elapsed_seconds":
                                          time.perf_counter() - t_enter})
                for pending in drains:
                    if pending is not None:
                        pending.result()
            except BaseException as exc:
                with lock:
                    failure.append(exc)
                raise
            finally:
                copier.shutdown(wait=True)
                cp.cuda.runtime.deviceSynchronize()

    if len(cards) == 1:
        work(cards[0])
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(cards),
                                thread_name_prefix="letkf-card") as pool:
            futures = [pool.submit(work, device) for device in cards]
            for future in futures:
                try:
                    future.result()
                except BaseException:
                    pass
        if failure:
            letkf = [exc for exc in failure if isinstance(exc, LetkfError)]
            raise (letkf or failure)[0]
    if failure:
        raise failure[0]

    # ---- the per-chunk results, in grid order ----------------------------
    blocks = np.concatenate([res["block_sums"] for res in results], axis=2)
    sums = blocks.sum(axis=2)
    maxabs = np.max([res["maxabs"] for res in results], axis=0)
    maxsig = np.max([res["maxsig"] for res in results], axis=0)
    prior_bad = np.any([res["prior_bad"] for res in results], axis=0)
    inc_bad = np.any([res["inc_bad"] for res in results], axis=0)
    timing = {key: sum(res["timing"][key] for res in results)
              for key in results[0]["timing"]}
    t_finish = time.perf_counter()
    diagnostics.solve_seconds = t_finish - t_solve
    # Seconds summed over chunks: with several cards these add up to more
    # than the wall clock, which is solve_seconds.
    diagnostics.weights_seconds = timing["weights"]
    diagnostics.transform_seconds = timing["transform"]
    #: transform_seconds split: the batched Jacobi factorisation, the
    #: fixed-order Pa / Wa / wbar products, and the apply kernel with its
    #: diagnostics reductions.  The remainder is the active-point gather
    #: and symmetrisation.
    diagnostics.eigen_seconds = timing["eigen"]
    diagnostics.products_seconds = timing["products"]
    diagnostics.apply_seconds = timing["apply"]
    diagnostics.stage_seconds = timing["stage"]
    diagnostics.unstage_seconds = timing["unstage"]
    diagnostics.device_chunks = len(spans)
    diagnostics.devices = [int(d) for d in cards]
    diagnostics.device_chunk_counts = {
        str(d): sum(1 for res in results if res["device"] == d)
        for d in cards}
    diagnostics.active_points = int(sum(res["active"] for res in results))
    diagnostics.max_local_obs = int(max(res["max_local"] for res in results))
    diagnostics.batches = int(sum(res["solved"] for res in results))
    diagnostics.max_jacobi_sweeps = int(max(res["sweeps"] for res in results))
    if held_fields:
        diagnostics.withheld = {
            "route": "device, in the filter's pass",
            "fields": list(held_fields),
            "groups": [{"fields": list(group["fields"]),
                        "zones": group["zones"]} for group in groups],
            "zones": int(sum(group["zones"] for group in groups)),
            "points": int(sum(res["held_points"] for res in results)),
            "active_points": int(sum(res["held_active"] for res in results)),
            "seconds": round(timing["withheld"], 3)}
    if chunk_hook is not None:
        diagnostics.chunk_hook_seconds = timing["hook"]

    # ---- the host path's refusals and _finish, from the per-point sums --
    for fi, f in enumerate(fields):
        if prior_bad[fi]:
            raise LetkfError(
                f"prior field {f!r} contains non-finite values; the filter"
                " will not launder them into an analysis.")
        if maxsig[fi] <= 1e-12 * maxabs[fi]:
            raise LetkfError(
                f"prior field {f!r} has no usable ensemble spread anywhere"
                f" (largest pointwise spread {maxsig[fi]!r} against a field"
                f" magnitude of {maxabs[fi]!r}): the members are identical to"
                " rounding, so there is no background covariance and no"
                " analysis to compute.  This is an ensemble-generation"
                " failure, not something the filter should paper over with"
                " a zero increment.  If the field is deliberately constant,"
                " drop it from analysis_fields.")
        if inc_bad[fi]:
            raise LetkfError(
                f"analysis increment for {f!r} is non-finite.  Every"
                " specific guard passed, so this is a genuine numerical"
                " failure in the transform -- do not apply this analysis.")
        diagnostics.prior_spread[f] = float(sums[fi, 0] / npts)
        diagnostics.mean_increment_rms[f] = float(math.sqrt(sums[fi, 1] / npts))
        diagnostics.posterior_spread[f] = float(sums[fi, 2] / npts)
    if chunk_hook is not None:
        diagnostics.chunk_hook_result = chunk_hook.finish(
            [res.get("hook") for res in results])
    try:
        used = 0.0
        held = 0.0
        for device in cards:
            with cp.cuda.Device(device):
                pool = cp.get_default_memory_pool()
                free_b, total_b = cp.cuda.runtime.memGetInfo()
                used = max(used, (int(total_b) - int(free_b)) / (1 << 20))
                held = max(held, pool.total_bytes() / (1 << 20))
                pool.free_all_blocks()
        diagnostics.device_used_mib = used
        diagnostics.pool_total_mib = held
    except Exception:
        pass
    diagnostics.finish_seconds = time.perf_counter() - t_finish
    if progress is not None:
        progress({"schema": "gpuwm-da.analysis-progress.v1",
                  "phase": "complete", "gridpoints_done": int(npts),
                  "gridpoints_total": int(npts), "chunks": len(spans),
                  "active_points": int(diagnostics.active_points),
                  "elapsed_seconds": time.perf_counter() - t_enter})
    return increments
