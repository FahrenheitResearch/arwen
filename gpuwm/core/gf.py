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
are about what the engine can hand it today):

1. **Forcing tendencies** (CLOSED, task #243).  GFDRV's forced states
   (``tn``/``qo``) sum advective (RTHFTEN/RQVFTEN), radiative (RTHRATEN)
   and boundary-layer (RTHBLTEN/RQVBLTEN) forcing, and all four auxiliary
   lanes are read off the bound driver:
   ``gf_rthdynten``/``gf_rqvdynten`` (the integrator's advective rates --
   the ARW dycore exports them at RK stage 1 of every step into
   ``state.rthften``/``state.rqvften``, bound at driver construction; the
   MPAS driver computes its own exactly as native v8.4.1's
   mpas_atm_time_integration.F:6936 + :2789 construction) and
   ``gf_rthblten``/``gf_rqvblten`` (the PBL slot's raw rates, retained by
   ``PhysicsDriver._couple_pbl_slot`` under WRF's between-calls
   persistence contract -- YSU, MYJ, MYNN, Shin-Hong and SASE all fill
   the slot on the same terms, exactly as WRF's cumulus driver reads
   RTHBLTEN/RQVBLTEN without asking which scheme wrote them).  A lane
   whose driver attribute is ``None`` feeds zeros -- what a driverless
   adapter and the parity harnesses get.

   THE ADVECTIVE LANE IS PURE ADVECTION, deliberately.
   ``module_cumulus_driver.F:867`` pre-folds ``RTHRATEN + RTHBLTEN`` into
   ``RTHFTEN`` for G3SCHEME and NTIEDTKESCHEME only; GFSCHEME is not in
   that list because the kernel sums the three lanes itself at
   ``kernels/gf.cu:4428``.  Handing this adapter a pre-folded RTHFTEN
   would integrate the boundary layer and the sky twice.
2. **Convective momentum tendencies.**  The kernel computes GF's
   dudt/dvdt; :class:`CumulusResult` carries no momentum slots, so they
   are not yet coupled.  WRF couples them; MPAS-A v8.4.1 does NOT
   (its cu_grell_freitas call carries no rucuten/rvcuten), so for the
   MPAS seam this is native parity, not a gap.
3. **w on mass levels** is the KF-precedent average
   ``0.5*(w[k] + w[k+1])`` of the staggered field.
4. **dx** is per-column when the driver carries ``gf_dx_column``
   ([ny, nx], metres); the scalar ``cfg.dx`` otherwise.  WRF's own GFDRV
   takes dx(i,j), so the per-column feed is the WRF interface, not an
   extension.

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


def gf_kernel_capacity(nz: int) -> int:
    """The ``GF_KMAX`` the module is compiled at for this ``nz``."""
    return _GF_KMAX_DEFAULT if nz <= _GF_KMAX_DEFAULT else int(nz)


# ---------------------------------------------------------------------------
# The per-thread column workspace
# ---------------------------------------------------------------------------
# gf.cu used to keep GFDRV's column arrays in the per-thread local frame, and
# CUDA prices a local frame at the card's RESIDENT-THREAD CAPACITY -- one
# per-context backing store of (frame - 1024) * SMs * maxThreadsPerSM, taken
# at first launch and never returned.  MEASURED on node-1 (RTX 5070 Ti, 70
# SMs x 1,536, sm_120, NVRTC 13.3): the 22,416 B frame took 2,196.0 MiB while
# the kernel only ever had 384 threads/SM in flight.  The arrays now live in
# a global workspace this adapter allocates, sized to the threads ACTUALLY in
# flight, and the columns are launched in tiles of that size.
#
# These three must match gf.cu's GFWS_SLOT_COUNT_COL / GFWS_SLOT_COUNT_DRV /
# GFWS_SLOTS; tests/test_gf_workspace.py re-derives them from the .cu source
# and fails if either side moves alone.
GFWS_SLOT_COUNT_COL = 90
GFWS_SLOT_COUNT_DRV = 36
GFWS_SLOTS = GFWS_SLOT_COUNT_COL + GFWS_SLOT_COUNT_DRV

#: Launch block, and the tile's granularity.  gf.cu indexes the workspace by
#: the global thread id within a tile, so a tile is always a whole number of
#: blocks.
GF_BLOCK = 64

#: Blocks per SM the tile is sized for.  MEASURED, not assumed: at 100,000
#: columns on node-1 (RTX 5070 Ti, 70 SMs) the kernel's wall is 31.07 ms at
#: 2 blocks/SM, 24.18 at 4, 24.33 at 6, 25.14 at 8 and 26.63 at 12, against
#: 19.33 ms for the pre-cut local-frame kernel.  4 is the plateau and the
#: cheapest point on it: 422 MiB of workspace where 12 would cost 1,266 MiB
#: and run slower.  The kernel's hardware occupancy at block=64 is 12
#: blocks/SM, so this is a throughput choice, not a residency limit, and the
#: query below only ever lowers it.
GF_TILE_BLOCKS_PER_SM = 4


def gf_workspace_floats(nz: int, columns: int) -> int:
    """Workspace floats for ``columns`` columns in flight at this ``nz``.

    Rounded up to whole blocks: gf.cu interleaves the workspace by LANE
    within a block, the way CUDA lays local memory out across a warp, so the
    unit of allocation is one block's region, not one column's.
    """
    blocks = (int(columns) + GF_BLOCK - 1) // GF_BLOCK
    return blocks * GFWS_SLOTS * (gf_kernel_capacity(nz) + 9) * GF_BLOCK


def gf_tile_columns(fn, ncol: int) -> int:
    """Columns to keep in flight: enough to fill the card, and no more.

    The whole point of the workspace is that it is charged per thread IN
    FLIGHT where the local-memory backing store was charged per thread the
    card could ever hold.  Over-sizing the tile hands that 4x back.
    """
    import cupy as cp

    dev = cp.cuda.Device()
    sms = dev.attributes["MultiProcessorCount"]
    per_sm = GF_TILE_BLOCKS_PER_SM
    try:
        from cupy_backends.cuda.api import driver

        resident = driver.occupancyMaxActiveBlocksPerMultiprocessor(
            fn.kernel.ptr, GF_BLOCK, 0)
        # Never launch more blocks than the card can hold resident: those
        # columns would wait while their workspace slots stayed allocated.
        per_sm = min(per_sm, int(resident))
    except Exception:                              # noqa: BLE001
        pass                                       # keep the measured value
    per_sm = max(1, int(per_sm))
    tile = sms * per_sm * GF_BLOCK
    return int(min(int(ncol), tile))


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

        def held_lane(name):
            """Driver-held auxiliary forcing lane, zeros when absent."""
            lane = (None if self._driver is None
                    else getattr(self._driver, name, None))
            if lane is None:
                return zeros_lev
            if lane.shape != (nz, ny, nx):
                raise ValueError(
                    f"driver {name} must be [nz, ny, nx]={nz, ny, nx}, "
                    f"got {lane.shape}")
            return cols(lane)

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
        lvin[:, _IN_LEV.index("rthften"), :] = held_lane("gf_rthdynten")
        lvin[:, _IN_LEV.index("rqvften"), :] = held_lane("gf_rqvdynten")
        lvin[:, _IN_LEV.index("rthraten"), :] = cols(rthraten)
        lvin[:, _IN_LEV.index("rthblten"), :] = held_lane("gf_rthblten")
        lvin[:, _IN_LEV.index("rqvblten"), :] = held_lane("gf_rqvblten")

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
        dx_column = (None if self._driver is None
                     else getattr(self._driver, "gf_dx_column", None))
        if dx_column is None:
            scin[:, 5] = DTYPE(cfg.dx)
        else:
            if dx_column.shape != (ny, nx):
                raise ValueError(
                    f"driver gf_dx_column must be [ny, nx]={ny, nx}, "
                    f"got {dx_column.shape}")
            scin[:, 5] = cp.asarray(
                dx_column, dtype=DTYPE).reshape(ncol)

        iin = cp.empty((ncol, 3), dtype=cp.int32)
        iin[:, 0] = fields["kpbl"].reshape(ncol)
        iin[:, 1] = np.int32(int(cfg.ishallow))
        iin[:, 2] = np.int32(int(cfg.clos_choice))

        lev = cp.zeros((ncol, len(_OUT_LEV), nz), dtype=DTYPE)
        sca = cp.zeros((ncol, len(_OUT_SCA)), dtype=DTYPE)
        isc = cp.zeros((ncol, len(_OUT_ISCA)), dtype=cp.int32)

        module = _gf_module(nz)
        fn = module.get_function("gf_gfdrv_stage")
        block = GF_BLOCK
        # The column arrays live in a global workspace sized to the threads
        # in flight, not in the per-thread local frame (which CUDA prices at
        # the card's whole resident-thread capacity).  Columns therefore go
        # in tiles of `tile`, each tile launched with the SAME one-thread-
        # one-column mapping the kernel has always had -- the slices below
        # are contiguous views, so the kernel indexes them exactly as before.
        tile = gf_tile_columns(fn, ncol)
        ws = cp.empty(gf_workspace_floats(nz, tile), dtype=DTYPE)
        for lo in range(0, ncol, tile):
            hi = min(lo + tile, ncol)
            fn((((hi - lo) + block - 1) // block,), (block,),
               (lvin[lo:hi], scin[lo:hi], iin[lo:hi],
                lev[lo:hi], sca[lo:hi], isc[lo:hi], ws,
                np.int32(0),      # SHIPPED default: corrected k22 indexing
                np.int32(hi - lo), np.int32(nz)))
        del ws

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
