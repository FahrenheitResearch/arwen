"""WSM3 simple-ice three-class microphysics (WRF ``mp_physics=3``).

WSM3 carries three water species: vapour, cloud (``qc``: cloud water at or
above 0 C, cloud ice below it) and precipitation (``qr``: rain at or above
0 C, snow below it).  Its Registry package (Registry.EM_COMMON:3017,
``wsm3scheme``) declares ``moist:qv,qc,qr`` and the three radiation radii,
so F_QI, F_QS and F_QG are false, and module_physics_init.F:4482 leaves
``warm_rain`` false: radiation that honours WRF's species flags moves cold
cloud into ice and cold precipitation into snow before it reads them.

The CUDA kernel (``kernels/wsm3.cu``) is a word-for-word port of WRF
v4.6.1's own call chain, ``phys/module_mp_wsm3.F`` (the ``wsm3`` wrapper,
``wsm32D`` with its semi-Lagrangian sedimentation and ``effectRad_wsm3``),
graded bit for bit against those routines compiled by gfortran over 77
columns and five time steps (tools/wsm3_wrf461_column_oracle,
tests/test_wsm3_wrf461_column_exact.py).  The column scheme is the claim;
whole-model trajectories are separate evidence.

Default arithmetic has three declared repairs (kernels/wsm3.cu
header, docs/wsm3_known_deltas.md): cloud ice that falls out of a lowest
level at or above 0 C is counted as rain instead of being lost (D1), and a
melting level at the model top with rising air freezes at the top level
instead of reading past the column (D2), and the shared WSM6/WDM6
upper-cell remap search retains mass on fall-through (D3).
GPUWM_WRF_EXACT=1 retains stock WRF accounting and remap for D1/D3;
no oracle column reaches D2.

REFL_10CM.  WRF computes none for WSM3 (the driver passes no refl_10cm
argument; module_diag_nwp.F:549).  gpuwm's output diagnostic for mp=3 is its
own, declared: WRF's ``refl10cm_wsm6`` evaluated on WSM3's phase split --
``qr`` as rain at or above 0 C and as snow below, no graupel -- whose rain
and snow intercepts, densities and temperature-dependent snow intercept are
WSM3's own (gpuwm/core/refl.py).
"""

from __future__ import annotations

import numpy as np
import cupy as cp

from gpuwm.config import RunConfig
from gpuwm.core.kernels import get_kernel, get_kernel_int_defines
from gpuwm.core.state import DTYPE, DomainState
from gpuwm.core.wsm_family import (check_surface, check_volume, finish_fields,
                                    precipitation_surface)
from gpuwm.core.wsm3_constants import (WSM3_DEEP_KMAX, WSM3_SHALLOW_KMAX,
                                       WSM3_VERTICAL_LEVEL_BOUNDS,
                                       wrf_init_words, wsm3_level_tier)

_COLUMN_TPB = 32
_SHALLOW_KMAX = WSM3_SHALLOW_KMAX
_KMAX = WSM3_DEEP_KMAX
VERTICAL_LEVEL_BOUNDS = WSM3_VERTICAL_LEVEL_BOUNDS
_kernel_capacity = wsm3_level_tier

_INIT_WORDS: dict = {}


def _init_words_on_device():
    """``wsm3init``'s float32 words, uploaded once per device."""
    key = cp.cuda.runtime.getDevice()
    words = _INIT_WORDS.get(key)
    if words is None:
        words = cp.asarray(wrf_init_words())
        _INIT_WORDS[key] = words
    return words


def wsm3_kernel(func: str, nz: int, conservative_surface: bool = False):
    """``func`` from ``kernels/wsm3.cu`` compiled at the WSM3_KMAX tier for ``nz``.

    ``conservative_surface`` is correct-mode finding WB5
    (``RunConfig.wsm3_conservative_surface``): its own unit compiled with
    ``WSM3_PLM_SURFACE``.  False compiles exactly the unit this function
    compiled before the key existed (the matched WRF arithmetic).
    """
    capacity = _kernel_capacity(nz)
    if conservative_surface:
        defines = (() if capacity == _SHALLOW_KMAX
                   else (("WSM3_KMAX", capacity),))
        return get_kernel_int_defines(
            "wsm3", func, defines + (("WSM3_PLM_SURFACE", 1),))
    if capacity == _SHALLOW_KMAX:
        return get_kernel("wsm3", func)
    return get_kernel_int_defines("wsm3", func, (("WSM3_KMAX", capacity),))


def launch_wsm3(theta, qv, qc, qr, w, rho, pii, pressure, dz,
                rainnc, rainncv, snownc, snowncv, sr, dt: float, *,
                effc, effi, effs, conservative_surface: bool = False) -> None:
    """Launch one WSM3 call over contiguous FP32 ``(nz,ny,nx)`` fields.

    ``w`` is the vertical velocity on the BOTTOM face of each mass level
    (WRF's ``w_2(i,k,j)`` for k = kts..kte), the face the scheme's
    freezing step reads at the melting level.  ``conservative_surface``:
    correct-mode finding WB5 (see :func:`wsm3_kernel`).
    """
    if not isinstance(conservative_surface, bool):
        raise TypeError("conservative_surface must be a bool, got "
                        f"{conservative_surface!r}")
    shape = theta.shape
    if len(shape) != 3:
        raise ValueError(f"WSM3 fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    _kernel_capacity(nz)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    volume = {
        "theta": theta, "qv": qv, "qc": qc, "qr": qr, "w": w,
        "rho": rho, "pii": pii, "pressure": pressure, "dz": dz,
        "effc": effc, "effi": effi, "effs": effs,
    }
    check_volume(volume, shape)
    check_surface({
            "rainnc": rainnc, "rainncv": rainncv,
            "snownc": snownc, "snowncv": snowncv, "sr": sr}, (ny, nx))
    ncol = ny * nx
    blocks = (ncol + _COLUMN_TPB - 1) // _COLUMN_TPB
    kernel = wsm3_kernel("wsm3_column", nz, conservative_surface)
    kernel(
        (blocks,), (_COLUMN_TPB,),
        (theta, qv, qc, qr, w, rho, pressure, pii, dz,
         rainnc, rainncv, snownc, snowncv, sr,
         effc, effi, effs, _init_words_on_device(), DTYPE(dt),
         np.int32(nz), np.int32(ny), np.int32(nx)))


def apply(state: DomainState, cfg: RunConfig, dt: float, *,
          refl_10cm_due: bool = False):
    """Prepare WRF fields, run WSM3, and finish retained MP heating."""
    nz, ny, nx = state.p.shape
    required = ("effc", "effi", "effs")
    missing = [name for name in required if getattr(state, name, None) is None]
    if missing:
        raise ValueError("mp_physics=3 state lacks WSM3 fields: "
                         + ", ".join(missing))
    from gpuwm.core.wsm6 import _wsm6_prepare
    theta, rho, pii, dz = _wsm6_prepare(state, "wsm3")
    # WRF solve_em passes the physical full-level w_2 straight through the
    # microphysics driver (module_microphysics_driver.F:2235, W=w); the
    # scheme reads w(i,k) for k = kts..kte, the lower face of each level.
    w = state.scratch((nz, ny, nx), "wsm3_w")
    w[...] = state.w[:nz]

    surface = precipitation_surface(state, ("rain", "snow"))

    from gpuwm.core.microphysics import save_pre_mp_theta
    save_pre_mp_theta(state)
    launch_wsm3(
        theta, state.qv, state.qc, state.qr, w, rho, pii, state.p, dz,
        dt=dt, **surface,
        effc=state.effc, effi=state.effi, effs=state.effs,
        conservative_surface=bool(
            getattr(cfg, "wsm3_conservative_surface", False)))
    return finish_fields(state, cfg, theta, pii, dt, surface,
                         refl_10cm_due=refl_10cm_due)


__all__ = ["apply", "launch_wsm3", "wsm3_kernel"]
