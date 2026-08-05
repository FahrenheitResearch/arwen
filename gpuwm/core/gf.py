"""Grell-Freitas cumulus adapter: the ``cu_physics=3`` engine callable.

The scheme itself is ``gpuwm/core/kernels/gf.cu`` -- the whole of WRF
v4.6.1's GFDRV per column, held at max_ulp 0 against the byte-frozen
oracle by ``tests/test_gf_gfdrv_cuda.py`` with fzu computed on the device.
This module is only the seam: it packs the engine's column state into the
kernel's input layout, launches ``gf_gfdrv_stage`` over every column, and
returns the driver's Task-1 :class:`~gpuwm.core.physics.CumulusResult`
(rates held until the next due call, RAINCV consumed once per due call --
GF has no NCA persistence; WRF recomputes it every STEPCU step, and the
importer pins ``cudt = 0`` for ``cu_physics=3`` so STEPCU is 1, exactly
WRF's usual GF configuration).

The kernel runs the SHIPPED identity: corrected k22 indexing
(``k22_wrf_faithful = 0``).  The WRF-faithful off-by-one is reachable only
through the parity suites, never from here; the measured ledger for the
difference is tests/test_gf_shallow_cuda.py::
test_the_corrected_k22_ledger_entry (three fixture cases move, all three
rejected either way, zero output words differ).

Recorded deviations of THIS seam (the kernel behind it is bitwise; these
are about what the engine can hand it today, and they are the plumb-list
for the label upgrade):

1. **Forcing tendencies.**  GFDRV's forced states (``tn``/``qo``) sum
   advective (RTHFTEN/RQVFTEN), radiative (RTHRATEN) and boundary-layer
   (RTHBLTEN/RQVBLTEN) forcing.  The engine's dycore does not yet export
   an advective theta/qv forcing pair, and the PBL stack couples its
   rates before the driver retains them -- so this adapter feeds the
   radiative term (held by the driver and read through ``bind_driver``)
   and ZEROS for the other two.  The instability closures still see the
   current state and the omega/moisture-convergence closures are fully
   fed (``omeg = -g*rho*w`` is computed from the live fields); what is
   degraded is the rate-of-change closure family's forcing and, with
   ``ishallow=1``, the shallow ``blqe`` member (``dhdt = 0``).
2. **Convective momentum tendencies.**  The kernel computes GF's
   dudt/dvdt; :class:`CumulusResult` carries no momentum slots, so they
   are not yet coupled.  WRF couples them.
3. **w on mass levels** is the KF-precedent average
   ``0.5*(w[k] + w[k+1])`` of the staggered field.

Ice routing follows WRF's own ``F_QI`` gate: with a prognostic ``qi`` the
258 K split routes cold condensate to RQICUTEN; without one, everything
lands in RQCCUTEN (module_cu_gf_wrfdrv.F:810-840 with F_QI false), which
is exact here because the split routes one number to one of two slots.
"""

from __future__ import annotations

import numpy as np

from gpuwm.core.kf import _model_clock_dt

DTYPE = np.float32

#: gf.cu's compiled level bound; nz above this recompiles via the
#: integer-define tier.
_GF_KMAX_DEFAULT = 40

# gf.cu input/output enum orders (gf_gfdrv_stage).  Must match the kernel;
# tools/gf_wrf461_oracle/gf_field_lists.py carries the same lists for the
# parity suites.
_IN_LEV = ("u", "v", "w", "t", "qv", "p", "pi", "rho", "dz8w", "p8w",
           "rthften", "rqvften", "rthraten", "rthblten", "rqvblten")
_OUT_LEV = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten", "dudt", "dvdt",
            "gdc", "gdc2",
            "outt", "outq", "outqc", "outu", "outv",
            "outts", "outqs", "outqcs")
_OUT_SCA = ("raincv", "pratec", "htop", "hbot", "xmb_shallow", "pret",
            "prets", "cuten", "cutens")
_OUT_ISCA = ("ktop_deep", "k22_shallow", "kbcon_shallow", "ktop_shallow",
             "kbcon", "ktop")


def _gf_module(nz: int):
    if nz <= _GF_KMAX_DEFAULT:
        from gpuwm.core.kernels import load_module

        return load_module("gf")
    from gpuwm.core.kernels import load_module_int_defines

    return load_module_int_defines("gf", (("GF_KMAX", int(nz)),))


class GrellFreitas:
    """Stateless ``cu_physics=3`` callable for :class:`PhysicsDriver`.

    The driver remains the sole cadence authority.  Each due call returns
    fresh rates (Task-1 contract: no ``nca_seconds``) and the RAINCV
    increment the kernel computed as ``pratec * dt``.
    """

    def __init__(self):
        self._driver = None

    def bind_driver(self, driver) -> None:
        """Optional driver hook: gives the adapter the held radiation rates.

        Mirrors the ``update_trigger_history`` optional-hook precedent: the
        driver calls it when present; a directly attached adapter without a
        driver simply runs with zero radiative forcing, which is also what
        a radiation-off configuration feeds.
        """
        self._driver = driver

    def __call__(self, *, atmosphere, fields, state, cfg):
        import cupy as cp

        from gpuwm.core.physics import CumulusResult

        nz, ny, nx = state.p.shape
        ncol = ny * nx

        def cols(a):
            # (nz, ny, nx) -> (ncol, nz), C-contiguous
            return cp.ascontiguousarray(
                a.reshape(nz, ncol).T, dtype=DTYPE)

        w_mass = DTYPE(0.5) * (state.w[:-1] + state.w[1:])
        if self._driver is not None:
            rthraten = self._driver.rthratenlw + self._driver.rthratensw
        else:
            rthraten = cp.zeros((nz, ny, nx), dtype=DTYPE)
        zeros_lev = cp.zeros((ncol, nz), dtype=DTYPE)

        lvin = cp.empty((ncol, len(_IN_LEV), nz), dtype=DTYPE)
        lvin[:, _IN_LEV.index("u"), :] = cols(atmosphere["u"])
        lvin[:, _IN_LEV.index("v"), :] = cols(atmosphere["v"])
        lvin[:, _IN_LEV.index("w"), :] = cols(w_mass)
        lvin[:, _IN_LEV.index("t"), :] = cols(atmosphere["temperature"])
        lvin[:, _IN_LEV.index("qv"), :] = cols(atmosphere["qv"])
        lvin[:, _IN_LEV.index("p"), :] = cols(atmosphere["pressure"])
        lvin[:, _IN_LEV.index("pi"), :] = cols(atmosphere["exner"])
        lvin[:, _IN_LEV.index("rho"), :] = cols(atmosphere["rho"])
        lvin[:, _IN_LEV.index("dz8w"), :] = cols(atmosphere["dz"])
        lvin[:, _IN_LEV.index("p8w"), :] = cols(
            atmosphere["p_interface"][:-1])
        lvin[:, _IN_LEV.index("rthften"), :] = zeros_lev   # deviation 1
        lvin[:, _IN_LEV.index("rqvften"), :] = zeros_lev   # deviation 1
        lvin[:, _IN_LEV.index("rthraten"), :] = cols(rthraten)
        lvin[:, _IN_LEV.index("rthblten"), :] = zeros_lev  # deviation 1
        lvin[:, _IN_LEV.index("rqvblten"), :] = zeros_lev  # deviation 1

        # WRF hands the scheme its model DT (module_cumulus_driver.F:1028)
        # -- the outer clock when the case integrates internal substeps,
        # the same idiom as the KF adapter.
        clock_dt = DTYPE(_model_clock_dt(cfg))
        scin = cp.empty((ncol, 6), dtype=DTYPE)
        scin[:, 0] = state.ht.reshape(ncol)
        scin[:, 1] = fields["hfx"].reshape(ncol)
        scin[:, 2] = fields["qfx"].reshape(ncol)
        scin[:, 3] = fields["xland"].reshape(ncol)
        scin[:, 4] = clock_dt
        scin[:, 5] = DTYPE(cfg.dx)

        iin = cp.empty((ncol, 3), dtype=cp.int32)
        iin[:, 0] = fields["kpbl"].reshape(ncol)
        iin[:, 1] = np.int32(int(cfg.ishallow))
        iin[:, 2] = np.int32(int(cfg.clos_choice))

        lev = cp.zeros((ncol, len(_OUT_LEV), nz), dtype=DTYPE)
        sca = cp.zeros((ncol, len(_OUT_SCA)), dtype=DTYPE)
        isc = cp.zeros((ncol, len(_OUT_ISCA)), dtype=cp.int32)

        module = _gf_module(nz)
        fn = module.get_function("gf_gfdrv_stage")
        block = 64
        fn(((ncol + block - 1) // block,), (block,),
           (lvin, scin, iin, lev, sca, isc,
            np.int32(0),          # SHIPPED default: corrected k22 indexing
            np.int32(ncol), np.int32(nz)))

        def grid(j):
            # (ncol, nz) slice -> (nz, ny, nx)
            return cp.ascontiguousarray(
                lev[:, j, :].T.reshape(nz, ny, nx))

        rthcuten = grid(_OUT_LEV.index("rthcuten"))
        rqvcuten = grid(_OUT_LEV.index("rqvcuten"))
        rqc = grid(_OUT_LEV.index("rqccuten"))
        rqi = grid(_OUT_LEV.index("rqicuten"))
        if getattr(state, "qi", None) is None:
            # WRF's F_QI-false arm: the 258 K split routes ONE number to one
            # of two slots, so folding is exact addition against a zero.
            rqc = cp.ascontiguousarray(rqc + rqi)
            rqi = None
        rainc = cp.ascontiguousarray(
            sca[:, _OUT_SCA.index("raincv")].reshape(ny, nx))

        return CumulusResult(
            rthcuten=rthcuten, rqvcuten=rqvcuten, rqccuten=rqc,
            rqicuten=rqi, rainc=rainc)


__all__ = ["GrellFreitas"]
