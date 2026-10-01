"""Noah's SFCDIAGS (T2, Q2, TH2) as one launch.

``PhysicsDriver._refresh_surface_diagnostics`` runs after every Noah call.
Written as CuPy array expressions it was about 20 launches from Python for a
few loads and stores per column, about 0.11 ms of every default-suite step on
a 250 x 200 grid.  This kernel makes the same writes in one launch and is an
exact transcription of those expressions, operation by operation:

* every multiply, add, subtract and divide is spelled with its ``_rn``
  intrinsic, in the expressions' own association (``rho * CP * safe_chs2``
  is ``(rho * CP) * safe_chs2``), so NVRTC cannot contract a multiply and an
  add or subtract into one FMA where CuPy rounded each operator separately;
* ``cp.power`` on float32 is CuPy's ``powf(in0, in1)``, which is what is
  called here;
* each ``cp.where`` is a select of values both of which are computed, so a
  column's branch never changes another column's bits.

TH2 reads the T2 this launch just wrote, as the third statement of the
expressions read ``f["t2"]`` after the second assigned it.  A caller holding
NumPy arrays keeps the expressions; :func:`device_arrays` is the test.
"""

from __future__ import annotations

import numpy as np

_SOURCE = r"""
extern "C" __global__
void noah_sfcdiags(const float* psfc, const float* tsk, const float* cqs2,
                   const float* chs2, const float* qsfc, const float* qfx,
                   const float* hfx, const float* qv1,
                   float* q2, float* t2, float* th2,
                   const float rd, const float cp_air, const float p0,
                   const float rcp, const float active_floor,
                   const long long ncol)
{
    const long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    const float ps = psfc[c];
    const float ts = tsk[c];
    // rho = psfc / (RD * tsk)
    const float rho = __fdiv_rn(ps, __fmul_rn(rd, ts));
    const float cq = cqs2[c];
    const float ch = chs2[c];
    const bool q_active = cq >= active_floor;
    const bool t_active = ch >= active_floor;
    const float safe_cqs2 = q_active ? cq : 1.0f;
    const float safe_chs2 = t_active ? ch : 1.0f;
    // diagnosed = where(q_active, qsfc - qfx / (rho * safe_cqs2), qsfc)
    const float qs = qsfc[c];
    const float inverted = __fsub_rn(
        qs, __fdiv_rn(qfx[c], __fmul_rn(rho, safe_cqs2)));
    const float diagnosed = q_active ? inverted : qs;
    // q2 = where(diagnosed > 0, diagnosed, qv[0])
    q2[c] = (diagnosed > 0.0f) ? diagnosed : qv1[c];
    // t2 = where(t_active, tsk - hfx / (rho * CP * safe_chs2), tsk)
    const float heated = __fsub_rn(
        ts, __fdiv_rn(hfx[c], __fmul_rn(__fmul_rn(rho, cp_air), safe_chs2)));
    const float temperature = t_active ? heated : ts;
    t2[c] = temperature;
    // th2 = t2 * power(P0 / psfc, RCP)
    th2[c] = __fmul_rn(temperature, powf(__fdiv_rn(p0, ps), rcp));
}
"""

_THREADS = 256
_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        import cupy as cp
        from gpuwm.certify.kernel_manifest import record_module
        from gpuwm.core.kernels import _compile_observed

        # A forecast translation unit, so the kernel manifest records it
        # (tests/test_kernel_manifest.py SITE_FILES): a RawModule, the
        # compile a RawKernel of the same source and options makes.
        module = cp.RawModule(code=_SOURCE, options=())
        _compile_observed(module, "gpuwm.core.noah_sfcdiags")
        record_module("gpuwm.core.noah_sfcdiags", source=_SOURCE,
                      options=(), module=module)
        _KERNEL = module.get_function("noah_sfcdiags")
    return _KERNEL


def device_arrays(shape, *arrays) -> bool:
    """True when every array is a C-contiguous float32 CuPy ``shape`` array."""
    try:
        import cupy as cp
    except ImportError:  # pragma: no cover - the driver imports CuPy
        return False
    return all(type(a) is cp.ndarray and a.dtype == np.float32
               and a.flags.c_contiguous and a.shape == shape for a in arrays)


def refresh(fields, qv1, *, rd, cp_air, p0, rcp, active_floor) -> None:
    """Write ``q2``, ``t2`` and ``th2`` in ``fields`` in one launch.

    The constants are the float32 scalars the expressions use, passed in so
    this module holds no second copy of them.
    """
    ncol = int(fields["tsk"].size)
    _kernel()(
        ((ncol + _THREADS - 1) // _THREADS,), (_THREADS,),
        (fields["psfc"], fields["tsk"], fields["cqs2"], fields["chs2"],
         fields["qsfc"], fields["qfx"], fields["hfx"], qv1,
         fields["q2"], fields["t2"], fields["th2"],
         np.float32(rd), np.float32(cp_air), np.float32(p0), np.float32(rcp),
         np.float32(active_floor), np.int64(ncol)))
