"""WRF 4.7.1 GOCART aging, with row indices instead of species names.

pack_parameters reads aging.to, aging.rate_s, aging.coproduct.to and
aging.coproduct.factor. References are zero-based indices in the supplied rows.
The packed mw is WRF's REAL float32 12.0. rate_s is REAL*8 float64
(WRF 4.63D-6), and coproduct.factor is REAL float32 (WRF 8.).
Rows without aging can be targets. Unrelated rows are left untouched.
"""
from __future__ import annotations

import numpy as np

from gpuwm.core.chem_column import columns, parameter, timestep


def pack_parameters(rows):
    n = len(rows)
    out = {"active": np.zeros(n, np.int32), "to": np.full(n, -1, np.int32),
           "rate": np.zeros(n, np.float64), "coproduct_to": np.full(n, -1, np.int32),
           "factor": np.zeros(n, np.float32), "mw": np.full(n, 12, np.float32)}
    targets = set()
    for r, row in enumerate(rows):
        aging = row.get("aging")
        if aging is None:
            continue
        dest = aging["to"]
        if isinstance(dest, bool) or not isinstance(dest, int) or not 0 <= dest < n or dest == r:
            raise ValueError("aging.to must be a different row index")
        if dest in targets:
            raise ValueError("each hydrophilic row must have one aging source")
        targets.add(dest)
        rate = float(aging["rate_s"])
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("aging.rate_s must be finite and positive")
        out["to"][r], out["rate"][r] = dest, rate
        out["active"][[r, dest]] = 1
        coproduct = aging.get("coproduct")
        if coproduct is not None:
            c = coproduct["to"]
            if isinstance(c, bool) or not isinstance(c, int) or not 0 <= c < n or c in (r, dest):
                raise ValueError("coproduct.to must be a third row index")
            factor = np.float32(coproduct["factor"])
            if not np.isfinite(factor) or factor < 0:
                raise ValueError("coproduct.factor must be finite and nonnegative")
            out["coproduct_to"][r], out["factor"][r] = c, factor
    sources = set(np.flatnonzero(out["to"] >= 0))
    if sources & targets:
        raise ValueError("aging chains are not the GOCART aging process")
    for c in out["coproduct_to"]:
        if c >= 0 and c not in targets:
            raise ValueError("a coproduct must target a hydrophilic aging row")
    return out


def upload_parameters(packed):
    import cupy as cp
    return {key: cp.asarray(value) for key, value in packed.items()}


def launch_ageing(species, params, dt, *, workspace=None):
    """Update levels 0..nz-2 in place, leaving the top mass level untouched.

    Workspace is float64 (3, nrows, ny, nx), temporary, not restart state.
    The caller may retain and reuse it. Returns the workspace used.
    Device parameter arrays are trusted output from pack/upload_parameters.
    """
    from gpuwm.core.kernels import get_kernel
    import cupy as cp
    shape, pointers = columns(species)
    n, (nz, ny, nx) = len(species), shape
    dt = timestep(dt, allow_zero=True)
    keys = ("active", "to", "rate", "coproduct_to", "factor", "mw")
    dtypes = (np.int32, np.int32, np.float64, np.int32, np.float32, np.float32)
    args = [parameter(params[k], n, dtype, k) for k, dtype in zip(keys, dtypes)]
    wshape = (3, n, ny, nx)
    if workspace is None:
        workspace = cp.empty(wshape, dtype=cp.float64)
    elif (not isinstance(workspace, cp.ndarray) or workspace.shape != wshape
          or workspace.dtype != np.float64 or not workspace.flags.c_contiguous):
        raise ValueError(f"workspace must be contiguous float64 with shape {wshape}")
    nc = ny * nx
    get_kernel("chem_ageing", "chem_ageing_gocart")(
        ((nc+127)//128,), (128,),
        (pointers, *args, workspace, np.int32(n), np.int32(nz), np.int32(nc), dt))
    return workspace


# ---------------------------------------------------------------------------
# The process (``aging.gocart``): the chem driver's view.
# ---------------------------------------------------------------------------

KEY = "aging.gocart"
#: Aging moves mass between rows and, through the OC co-product of BC aging
#: (WRF's stand-in for SOA, module_gocart_aerosols.F:70), adds some: the
#: driver books the net change under chemistry.
LEDGER = "chemistry"
#: The kernel translation units step() launches (priced by the preflight).
KERNEL_MODULES = ("chem_ageing",)
REQUIRES: tuple[str, ...] = ()
#: The kernel's REAL*8 workspace, (3, participants, ny, nx), is allocated
#: once per state on the first step (twelve 2-D planes for gocart_lite, far
#: less than any shape ChemAllocation offers) and reused.
ALLOCATES: tuple = ()


def rows(table):
    return table.rows_for(KEY)


def participants(table):
    """``(rows, dicts)``: every row aging touches -- the hydrophobic sources
    that name aging.gocart, their hydrophilic partners and the co-product
    target -- in table order, with each source's partner and co-product as
    indices into that list, the form :func:`pack_parameters` reads."""
    sources = [r for r in rows(table) if r.aging is not None]
    names = []
    for r in sources:
        names.append(r.name)
        names.append(r.aging["to"])
        co = r.aging.get("coproduct")
        if co is not None:
            names.append(co["to"])
    ordered = [r for r in table.rows if r.name in set(names)]
    # The chem driver books a process's mass change on the rows that name
    # it and nowhere else, so a partner or co-product row that did not name
    # aging.gocart would have its gain filed under whichever process next
    # measured it (seen: oc2's aged mass booked as "coupling", bc2's as the
    # next step's "transport").  The ledger would still close; the budget
    # would be wrong.
    named = {r.name for r in rows(table)}
    unnamed = sorted(r.name for r in ordered if r.name not in named)
    if unnamed:
        raise ValueError(
            f"rows {unnamed} receive aged mass but do not name {KEY} in "
            "their processes: the mass ledger would book their gain under "
            "another process")
    index = {r.name: i for i, r in enumerate(ordered)}
    dicts = []
    for r in ordered:
        if r.aging is None or r not in sources:
            dicts.append({})
            continue
        rate = r.aging.get("rate_per_s")
        if rate is None:
            rate = 1.0 / float(r.aging["efold_s"])
        entry = {"to": index[r.aging["to"]], "rate_s": float(rate)}
        co = r.aging.get("coproduct")
        if co is not None:
            entry["coproduct"] = {"to": index[co["to"]],
                                  "factor": float(co["factor"])}
        dicts.append({"aging": entry})
    return ordered, dicts


def init(ctx):
    ordered, dicts = participants(ctx.table)
    if ordered:
        pack_parameters(dicts)


def step(ctx, dt, ktau):
    from gpuwm.core.chem_sulfur import chem_step_dt

    run, dtstepc = chem_step_dt(ctx.cfg, ktau, dt)
    if not run:
        return
    chem = ctx.state.chem
    cache = getattr(chem, "gocart_ageing_cache", None)
    if cache is None:
        ordered, dicts = participants(ctx.table)
        cache = chem.gocart_ageing_cache = (
            ordered, upload_parameters(pack_parameters(dicts)))
    ordered, params = cache
    if not ordered:
        return
    workspace = getattr(chem, "gocart_ageing_work", None)
    workspace = launch_ageing([ctx.field(r) for r in ordered], params, dtstepc,
                             workspace=workspace)
    chem.gocart_ageing_work = workspace
