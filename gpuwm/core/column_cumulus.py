"""Table-driven column cumulus: one launcher for every one-kernel scheme.

A scheme whose whole WRF driver call runs as ONE per-column kernel is
admitted here by a row of :data:`COLUMN_CUMULUS_SCHEMES`, never by its own
adapter module: the row names the kernel unit and entry, the workspace it
needs, the namelist knobs it reads and how WRF maps ``mp_physics`` onto its
condensate classes.  :class:`ColumnCumulus` packs the engine's column state
into the kernel's level-major inputs, launches the entry over tiles of
columns with a workspace sized to the threads in flight, and returns the
driver's Task-1 :class:`~gpuwm.core.physics.CumulusResult`.

The first row is New SAS (``cu_physics = 96``, ``gpuwm/core/kernels/
nsas.cu``), graded word for word against WRF v4.6.1's own object by
``tests/test_nsas_column_oracle.py``.

THE KERNEL CONTRACT a row's entry must keep (``nsas_column`` is the
reference):

* level-major inputs ``a[k*ncol + i]``: ``t, qv, qc, qi, p, pi, rho, dz8w,
  u, v`` on mass levels, then ``p8w, w`` on the nz+1 interfaces;
* a per-column float table (:data:`SC_FIELDS`) and int table
  (:data:`IC_FIELDS`);
* six level-major outputs ``rthcuten, rqvcuten, rqccuten, rqicuten,
  rucuten, rvcuten`` (a null ``rqccuten``/``rqicuten`` is WRF's F_QC/F_QI
  false) and ``sfc[j*ncol + i]`` for ``raincv, pratec, hbot, htop``;
* a workspace of ``slots * (nz + 2) * tile`` floats and the launch scalars
  ``col0, ntile, ncol, nz``.

RAIN. WRF holds PRATEC between calls and accumulates PRATEC*DT on each
model-clock step (module_physics_addtendc.F:2141). The launcher returns the
persistent rate so the shared driver advances precipitation on that clock.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from gpuwm.core.state import DTYPE

#: Per-column float table, in the kernel's column order.
SC_FIELDS = ("xland", "hpbl", "hfx", "qfx", "dx", "dt", "pgcon")
#: Per-column int table, in the kernel's column order.
IC_FIELDS = ("stepcu", "ncloud", "dx_factor")
#: Level-major mass and interface inputs, in the kernel's argument order.
MASS_INPUTS = ("t", "qv", "qc", "qi", "p", "pi", "rho", "dz8w", "u", "v")
IFACE_INPUTS = ("p8w", "w")
LEVEL_OUTPUTS = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten",
                 "rucuten", "rvcuten")
SURFACE_OUTPUTS = ("raincv", "pratec", "hbot", "htop")

#: Threads per block, and the tile's granularity.
BLOCK = 64
#: Blocks per SM the tile is sized for (the workspace is charged per thread
#: in flight, so the tile is the card's width and no more).
TILE_BLOCKS_PER_SM = 4


def wrf_sas_ncloud(mp_physics: int) -> int:
    """module_cu_nsas.F:141-147: no cloud (0), cloud only (1), cloud+ice (2)."""
    mp = int(mp_physics)
    if mp == 0:
        return 0
    if mp in (1, 3):
        return 1
    return 2


@dataclass(frozen=True)
class ColumnCumulusScheme:
    """One row: everything the launcher needs to run a column scheme."""

    cu_physics: int
    name: str
    #: gpuwm/core/kernels/<kernel_module>.cu
    kernel_module: str
    entry: str
    #: Workspace slots per level; the unit's own count is checked against
    #: this by tests/test_nsas_column_oracle.py.
    workspace_slots: int
    #: RunConfig field -> kernel per-column float (namelist knobs).
    config_floats: tuple[tuple[str, str], ...]
    #: RunConfig field -> kernel per-column int (namelist knobs).
    config_ints: tuple[tuple[str, str], ...]
    #: mp_physics -> the scheme's condensate classes (0, 1 or 2).
    ncloud: object
    #: The checkpoint/algorithm identity string.
    identity: str
    row_limits_entry: str | None = None


COLUMN_CUMULUS_SCHEMES = MappingProxyType({
    96: ColumnCumulusScheme(
        cu_physics=96, name="New SAS (YSU) with NSCV shallow convection",
        kernel_module="nsas", entry="nsas_column", workspace_slots=50,
        config_floats=(("sas_pgcon", "pgcon"),),
        config_ints=(("nsas_dx_factor", "dx_factor"),),
        ncloud=wrf_sas_ncloud,
        identity="new-sas-nscv-wrf461-cu_nsas-v1",
        row_limits_entry="nsas_row_limits"),
})


def workspace_floats(scheme: ColumnCumulusScheme, nz: int, tile: int) -> int:
    return int(scheme.workspace_slots) * (int(nz) + 2) * int(tile)


def tile_columns(ncol: int, multiprocessor_count: int) -> int:
    """Columns in flight per launch: the card's width, never more than ncol."""
    tile = int(multiprocessor_count) * TILE_BLOCKS_PER_SM * BLOCK
    return max(1, min(int(ncol), tile))


def workspace_bytes(cu_physics: int, nz: int, ncol: int,
                    multiprocessor_count: int) -> int:
    """Device bytes the launcher allocates for one call (preflight pricing)."""
    scheme = COLUMN_CUMULUS_SCHEMES[int(cu_physics)]
    tile = tile_columns(ncol, multiprocessor_count)
    return 4 * workspace_floats(scheme, nz, tile)


def launch_columns(scheme: ColumnCumulusScheme, *, mass, iface, sc, ic,
                   outputs, sfc, nz: int, ncol: int, prep=None,
                   module=None, row_width: int = 1):
    """Launch ``scheme.entry`` over ``ncol`` columns in tiles.

    ``mass``/``iface`` map the kernel's input names to level-major CuPy
    arrays; ``outputs`` maps LEVEL_OUTPUTS to arrays or None; ``sfc`` is
    ``(4, ncol)``.  ``module`` overrides the compiled unit (the planted
    one-bit tests compile a perturbed copy of the source).
    """
    import cupy as cp

    if module is None:
        from gpuwm.core.kernels import load_module
        module = load_module(scheme.kernel_module)
    if row_width < 1 or ncol % row_width:
        raise ValueError("row_width must divide ncol; partial rows change WRF cloud-search limits")
    fn = module.get_function(scheme.entry)
    limits = None
    if scheme.row_limits_entry is not None:
        nrow = ncol // row_width
        limits = cp.empty((nrow, 3), dtype=cp.int32)
        module.get_function(scheme.row_limits_entry)(
            ((nrow + BLOCK - 1) // BLOCK,), (BLOCK,),
            (mass["p"], iface["p8w"], limits, np.int32(ncol),
             np.int32(nz), np.int32(row_width)))
    sms = int(cp.cuda.Device().attributes["MultiProcessorCount"])
    tile = tile_columns(ncol, sms)
    ws = cp.empty(workspace_floats(scheme, nz, tile), dtype=DTYPE)
    null = np.uint64(0)
    args_in = tuple(mass[n] for n in MASS_INPUTS) + tuple(
        iface[n] for n in IFACE_INPUTS)
    outs = tuple(null if outputs.get(n) is None else outputs[n]
                 for n in LEVEL_OUTPUTS)
    for col0 in range(0, int(ncol), tile):
        width = min(tile, int(ncol) - col0)
        fn(((width + BLOCK - 1) // BLOCK,), (BLOCK,),
           args_in + (sc, ic) + outs
           + (sfc, null if prep is None else prep, ws,
              np.int32(col0), np.int32(width), np.int32(ncol),
              np.int32(nz), null if limits is None else limits,
              np.int32(row_width)))
    del ws


class ColumnCumulus:
    """The ``cu_physics`` callable for every row of the table.

    Stateless between calls (no hold: WRF's SAS family recomputes on its
    STEPCU cadence and the driver holds the rates), so ``release`` frees
    nothing but the driver back-reference.
    """

    def __init__(self, cu_physics: int):
        self.scheme = COLUMN_CUMULUS_SCHEMES[int(cu_physics)]
        self._driver = None

    def bind_driver(self, driver) -> None:
        """Optional driver hook: PBL height and the cumulus cadence."""
        self._driver = driver

    def release(self) -> int:
        self._driver = None
        return 0

    def _stepcu(self) -> int:
        stepcu = getattr(self._driver, "stepcu", 1) if self._driver else 1
        return max(int(stepcu), 1)

    def __call__(self, *, atmosphere, fields, state, cfg):
        import cupy as cp

        from gpuwm.core.kf import _model_clock_dt
        from gpuwm.core.physics import CumulusResult

        scheme = self.scheme
        nz, ny, nx = state.p.shape
        ncol = ny * nx

        def lev(a, levels=nz):
            return cp.ascontiguousarray(a, dtype=DTYPE).reshape(levels, ncol)

        mass = {"t": lev(atmosphere["temperature"]), "qv": lev(atmosphere["qv"]),
                "qc": lev(atmosphere["qc"]), "qi": lev(atmosphere["qi"]),
                "p": lev(atmosphere["pressure"]), "pi": lev(atmosphere["exner"]),
                "rho": lev(atmosphere["rho"]), "dz8w": lev(atmosphere["dz"]),
                "u": lev(atmosphere["u"]), "v": lev(atmosphere["v"])}
        iface = {"p8w": lev(atmosphere["p_interface"], nz + 1),
                 "w": lev(state.w, nz + 1)}

        clock_dt = float(_model_clock_dt(cfg))
        stepcu = self._stepcu()
        floats = {"dx": float(cfg.dx), "dt": clock_dt}
        for field_name, slot in scheme.config_floats:
            floats[slot] = float(getattr(cfg, field_name))
        sc = cp.empty((ncol, len(SC_FIELDS)), dtype=DTYPE)
        sc[:, SC_FIELDS.index("xland")] = fields["xland"].reshape(ncol)
        sc[:, SC_FIELDS.index("hpbl")] = fields["pblh"].reshape(ncol)
        sc[:, SC_FIELDS.index("hfx")] = fields["hfx"].reshape(ncol)
        sc[:, SC_FIELDS.index("qfx")] = fields["qfx"].reshape(ncol)
        for slot, value in floats.items():
            sc[:, SC_FIELDS.index(slot)] = DTYPE(value)

        ncloud = int(scheme.ncloud(cfg.mp_physics))
        ints = {"stepcu": stepcu, "ncloud": ncloud}
        for field_name, slot in scheme.config_ints:
            ints[slot] = int(getattr(cfg, field_name))
        ic = cp.empty((ncol, len(IC_FIELDS)), dtype=cp.int32)
        for slot, value in ints.items():
            ic[:, IC_FIELDS.index(slot)] = np.int32(value)

        # WRF forms RQCCUTEN only where the state carries QC (F_QC) and
        # RQICUTEN only where it carries QI (F_QI), module_cu_nsas.F:285-303.
        has_qc = getattr(state, "qc", None) is not None
        has_qi = getattr(state, "qi", None) is not None
        out = {n: cp.empty((nz, ncol), dtype=DTYPE) for n in LEVEL_OUTPUTS}
        if not has_qc:
            out["rqccuten"] = None
        if not has_qi:
            out["rqicuten"] = None
        sfc = cp.empty((len(SURFACE_OUTPUTS), ncol), dtype=DTYPE)
        launch_columns(scheme, mass=mass, iface=iface, sc=sc, ic=ic,
                       outputs=out, sfc=sfc, nz=nz, ncol=ncol, row_width=nx)

        def grid(a):
            return None if a is None else a.reshape(nz, ny, nx)

        rqc = grid(out["rqccuten"])
        rqi = grid(out["rqicuten"])
        pratec = sfc[SURFACE_OUTPUTS.index("pratec")].reshape(ny, nx)
        return CumulusResult(
            rthcuten=grid(out["rthcuten"]), rqvcuten=grid(out["rqvcuten"]),
            rqccuten=rqc, rqicuten=rqi,
            rucuten=grid(out["rucuten"]), rvcuten=grid(out["rvcuten"]),
            pratec=cp.ascontiguousarray(pratec), persistent_pratec=True)


__all__ = ["COLUMN_CUMULUS_SCHEMES", "ColumnCumulus", "ColumnCumulusScheme",
           "launch_columns", "workspace_bytes", "wrf_sas_ncloud"]
