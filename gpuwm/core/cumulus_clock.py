"""The per-step cumulus clock (WRF ``advance_ppt``) as two launches.

``PhysicsDriver._advance_cumulus_clock`` and ``finish_step`` run on every
model-clock step of every cumulus suite.  Written as CuPy array expressions
they were about 27 small launches (13 for the rain and NCA bookkeeping, 12
masked clears of the held rate volumes and one mask reset), each launched from
Python, so the step spent about 0.14 ms on a 250 x 200 x 50 grid on work that
is a few loads and stores per column.

These two kernels do the same writes in one launch each.  They are exact
transcriptions of the CuPy expressions they replace, operation by operation:

* every multiply, add, subtract and divide is spelled with its ``_rn``
  intrinsic, so NVRTC cannot contract a multiply and an add into one FMA
  where the CuPy expressions rounded twice (CuPy runs each operator as its own
  kernel, so no two operators ever fused);
* ``cp.maximum`` is CuPy's own float32 routine, including its NaN rule
  (either operand NaN gives the canonical quiet NaN 0x7fffffff);
* the clears are select-and-store with no arithmetic, like
  :func:`gpuwm.core.health_ledger.masked_clear`, so a column that does not
  expire keeps its bits.

A caller holding NumPy arrays (the CPU driver harness) keeps the array
expressions; :func:`device_arrays` is the test.
"""

from __future__ import annotations

import numpy as np

_ADVANCE_SOURCE = r"""
extern "C" __global__
void cumulus_clock_advance(float* rainc,
                           float* pending_rainbl,
                           float* surface_raincv,
                           float* nca,
                           const float* pratec,
                           float* expiring,
                           const float dt, const int has_surface_raincv,
                           const long long ncol)
{
    const long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    const float pr = pratec[c];
    // rainc += cu_pratec * dt
    rainc[c] = __fadd_rn(rainc[c], __fmul_rn(pr, dt));
    // cp.maximum(cu_pratec, 0) * dt, CuPy's float32 maximum routine.
    const float wet = (isnan(pr) | isnan(0.0f))
        ? __int_as_float(0x7fffffff) : fmaxf(pr, 0.0f);
    const float increment = __fmul_rn(wet, dt);
    pending_rainbl[c] = __fadd_rn(pending_rainbl[c], increment);
    if (has_surface_raincv) {
        surface_raincv[c] = __fadd_rn(surface_raincv[c], increment);
    }
    // active = nca > 0; expiring = active & (floor(nca / dt + 0.5) <= 1)
    const float held = nca[c];
    const bool active = held > 0.0f;
    const bool expires = active
        & (floorf(__fadd_rn(__fdiv_rn(held, dt), 0.5f)) <= 1.0f);
    expiring[c] = expires ? 1.0f : 0.0f;
    // nca = where(active, nca - dt, nca)
    nca[c] = active ? __fsub_rn(held, dt) : held;
}
"""

_CLEAR_SOURCE = r"""
extern "C" __global__
void cumulus_clock_clear(float* expiring,
                         float* v0, float* v1, float* v2, float* v3,
                         float* v4, float* v5, float* v6, float* v7,
                         float* v8, float* v9, float* v10, float* v11,
                         const int nvolume, const long long nz,
                         const long long ncol)
{
    const long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    // mask = cu_expiring != 0, then masked_clear on every held volume.
    if (expiring[c] != 0.0f) {
        float* const volumes[12] = {v0, v1, v2, v3, v4, v5,
                                    v6, v7, v8, v9, v10, v11};
        for (int j = 0; j < nvolume; ++j) {
            for (long long k = 0; k < nz; ++k) {
                volumes[j][k * ncol + c] = 0.0f;
            }
        }
    }
    // cu_expiring[...] = 0
    expiring[c] = 0.0f;
}
"""

#: The clear kernel's volume slots: six held raw rates and at most six held
#: coupled copies.
MAX_VOLUMES = 12
_THREADS = 256

_ADVANCE = None
_CLEAR = None


def _kernels():
    global _ADVANCE, _CLEAR
    if _ADVANCE is None:
        import cupy as cp
        from gpuwm.certify.kernel_manifest import record_module
        from gpuwm.core.kernels import _compile_observed

        # Forecast translation units, so the kernel manifest records them
        # (tests/test_kernel_manifest.py SITE_FILES): RawModules, the
        # compile a RawKernel of the same source and options makes.
        advance = cp.RawModule(code=_ADVANCE_SOURCE, options=())
        _compile_observed(advance, "gpuwm.core.cumulus_clock:advance")
        record_module("gpuwm.core.cumulus_clock:advance",
                      source=_ADVANCE_SOURCE, options=(), module=advance)
        clear = cp.RawModule(code=_CLEAR_SOURCE, options=())
        _compile_observed(clear, "gpuwm.core.cumulus_clock:clear")
        record_module("gpuwm.core.cumulus_clock:clear",
                      source=_CLEAR_SOURCE, options=(), module=clear)
        _ADVANCE = advance.get_function("cumulus_clock_advance")
        _CLEAR = clear.get_function("cumulus_clock_clear")
    return _ADVANCE, _CLEAR


def device_arrays(*arrays) -> bool:
    """True when every array is a C-contiguous float32 CuPy array."""
    try:
        import cupy as cp
    except ImportError:  # pragma: no cover - the driver imports CuPy
        return False
    return all(type(a) is cp.ndarray and a.dtype == np.float32
               and a.flags.c_contiguous for a in arrays)


def advance(*, rainc, pending_rainbl, surface_raincv, nca, pratec, expiring,
            dt) -> None:
    """One launch for the rain accumulation and the NCA countdown.

    ``surface_raincv`` is ``None`` when no land model reads it.  Every array
    is (ny, nx) float32 and C-contiguous (see :func:`device_arrays`).
    """
    advance_kernel, _ = _kernels()
    ncol = int(nca.size)
    has_surface = surface_raincv is not None
    advance_kernel(
        ((ncol + _THREADS - 1) // _THREADS,), (_THREADS,),
        (rainc, pending_rainbl, surface_raincv if has_surface else rainc,
         nca, pratec, expiring, np.float32(dt), np.int32(has_surface),
         np.int64(ncol)))


def clear(expiring, volumes) -> None:
    """One launch for every masked clear and the mask reset.

    ``volumes`` are (nz, ny, nx) float32 C-contiguous arrays over the same
    columns as ``expiring``; at most :data:`MAX_VOLUMES`.
    """
    _, clear_kernel = _kernels()
    volumes = list(volumes)
    if len(volumes) > MAX_VOLUMES:
        raise ValueError(
            f"the cumulus clock clears at most {MAX_VOLUMES} held volumes in "
            f"one launch, got {len(volumes)}")
    ncol = int(expiring.size)
    nz = int(volumes[0].shape[0]) if volumes else 0
    # Unused slots point at a real volume and are never read: nvolume stops
    # the loop before them.
    pad = volumes[0] if volumes else expiring
    slots = tuple(volumes) + (pad,) * (MAX_VOLUMES - len(volumes))
    clear_kernel(
        ((ncol + _THREADS - 1) // _THREADS,), (_THREADS,),
        (expiring,) + slots
        + (np.int32(len(volumes)), np.int64(nz), np.int64(ncol)))
