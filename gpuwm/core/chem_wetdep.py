"""Large-scale removal from WRF v4.7.1 chem/module_wetdep_ls.F.

The driver books the before/after mass difference. removed_mass is an
accumulated output only, so this process never books the ledger twice.
"""
from gpuwm.core.chem_context import ChemAllocation

KEY = "wetdep.ls"
LEDGER = "scavenged"
REQUIRES = ("rainncv", "qc", "rho", "dryrho", "dz8w", "w")
ALLOCATES = (
    ChemAllocation("wetdep_removed", "rows_2d", output_name="WETDEP_LS_REMOVED",
                   units="ug m-2", description="Accumulated large-scale scavenged column mass"),
    # Each row's scavenging factor, read by the kernel from the first word of
    # the row's plane; rebuilt every step from the row, never carried.
    ChemAllocation("wetdep_alpha", "rows_2d", restart="rebuild"),
)
KERNEL_MODULES = ("chem_wetdep_ls",)


def rows(table):
    return table.rows_for(KEY)


def init(ctx):
    """Driver owns allocation and zero initialization; preserve restart state."""


def step(ctx, dt, ktau):
    if dt <= 0:
        raise ValueError("wetdep.ls needs positive dt to prevent division by zero in rain rate")
    import numpy as np
    from gpuwm.core.kernels import get_kernel
    active = rows(ctx.table)
    if not active:
        return
    # WRF calls wetdep_ls only under wetscav_onoff < 0 (chem_driver.F:1687):
    # the large-scale washout is WRF's switch, not the row's.
    if int(getattr(ctx.cfg, "wetscav_onoff", 0)) >= 0:
        return
    kernel = get_kernel("chem_wetdep_ls", "chem_wetdep_ls")
    inputs = tuple(ctx.met(name) for name in
                   ("rainncv", "qc", "rho", "dryrho", "dz8w", "w"))
    for r, row in enumerate(active):
        if row.wetdep_ls_alpha == 0:
            continue    # WRF's cycle (module_wetdep_ls.F:40-46): no launch
        field = ctx.field(row)
        nz, ny, nx = field.shape
        alpha = ctx.diag["wetdep_alpha"][r]
        alpha.reshape(-1)[:1].fill(row.wetdep_ls_alpha)
        # One row per launch, in place on the species field: the kernel's
        # row loop runs once (nr = 1) and reads alpha[0].
        kernel(((nx*ny+127)//128,), (128,),
               (field, *inputs, alpha, ctx.diag["wetdep_removed"][r],
                np.float32(dt), np.int32(nz), np.int32(nx*ny), np.int32(1)))
