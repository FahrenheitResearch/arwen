"""GOCART sea salt, WRF chem_opt=300, surface level only (process
``emission.seasalt``).

The per sub-bin mass factor of ``source_ss`` does not depend on the column:
it is computed in double on the device, in one thread, once per (rows, dt)
by :func:`prepare_seasalt_table`, and every column then accumulates the
sub-bins sequentially in WRF's ``ir`` order.  :func:`launch_seasalt`
builds the table itself when it is not handed one (the oracle replay), and
the process caches it so a forecast step makes no host read.
"""
from __future__ import annotations

import numpy as np

from gpuwm.core.chem_dust import _arrays


def pack_seasalt_rows(rows):
    """Pack ra, rb (um) and den_seas (kg/m3), all WRF REAL*8."""
    rows = tuple(rows)
    if not rows:
        raise ValueError("sea salt requires at least one row")
    packed = {key: np.asarray([r[key] for r in rows], dtype=np.float64)
              for key in ("ra", "rb", "den_seas")}
    if any(not np.all(np.isfinite(a)) for a in packed.values()):
        raise ValueError("sea salt row parameters must be finite")
    if np.any(packed["den_seas"] <= 0):
        raise ValueError("sea salt density must be positive")
    if np.any(packed["ra"] <= 0) or np.any(packed["rb"] <= packed["ra"]):
        raise ValueError("sea salt edges require 0 < ra < rb")
    return packed


def prepare_seasalt_table(params, dt):
    """``(packed edges and densities, offsets, table)`` for one dt.

    One host read of the sub-bin count, at preparation only.
    """
    from gpuwm.core.kernels import get_kernel
    import cupy as cp
    nrows = len(params["ra"])
    packed = [cp.asarray(params[k]) for k in ("ra", "rb", "den_seas")]
    offsets = cp.empty(nrows+1, dtype=cp.int32)
    get_kernel("chem_seasalt", "chem_seasalt_offsets")((1,), (1,),
        (np.int32(nrows), packed[0], packed[1], offsets))
    count = int(offsets[-1:].get()[0])
    if count < 0 or count > 100000:
        raise ValueError("invalid or excessive sea salt sub-bin count")
    table = cp.empty(count, dtype=cp.float64)
    get_kernel("chem_seasalt", "chem_seasalt_table")((1,), (1,),
        (np.int32(nrows), *packed, offsets, table, np.float32(dt)))
    return packed, offsets, table


def launch_seasalt(species, emissions, params, *, xland, z_at_w, p8w,
                   dz8w, u10, v10, u_phy, v_phy, dx, dt, g=9.81,
                   prepared=None):
    """Set emissions (kg/cell/step) and update level 0 mixing ratios.

    W-level fields have (nz+1, ny, nx). Workspace: two row pointer arrays,
    three double parameter arrays, int32 offsets, and one double per sub-bin.
    """
    from gpuwm.core.kernels import get_kernel
    fields = dict(xland=xland, z_at_w=z_at_w, p8w=p8w, dz8w=dz8w,
                  u10=u10, v10=v10, u_phy=u_phy, v_phy=v_phy)
    cp, shape, ptr, eptr = _arrays(species, emissions, fields)
    nz, ny, nx = shape
    for key, a in fields.items():
        wanted = (nz+1, ny, nx) if key in ("z_at_w", "p8w") else (ny, nx) if key in ("xland", "u10", "v10") else shape
        if a.shape != wanted:
            raise ValueError(f"{key} must have shape {wanted}")
    for key in ("ra", "rb", "den_seas"):
        if params[key].dtype != np.float64 or params[key].shape != (len(species),):
            raise ValueError(f"{key} must be float64 with one value per row")
    if prepared is None:
        prepared = prepare_seasalt_table(params, dt)
    _packed, offsets, table = prepared
    nc = nx*ny
    get_kernel("chem_seasalt", "chem_seasalt")(((nc+127)//128,), (128,),
        (np.int32(len(species)), np.int32(nc), ptr, eptr, offsets, table,
         xland, z_at_w, p8w, dz8w, u10, v10, u_phy, v_phy,
         np.float32(dx), np.float32(g)))
    return species, emissions


# ---------------------------------------------------------------------------
# The process (``emission.seasalt``): the chem driver's view.
# ---------------------------------------------------------------------------

from gpuwm.core.chem_context import ChemAllocation  # noqa: E402

KEY = "emission.seasalt"
LEDGER = "emitted"
#: The kernel translation units step() launches (priced by the preflight).
KERNEL_MODULES = ("chem_seasalt",)
REQUIRES = ("xland", "u10", "v10", "z_at_w", "p8w", "dz8w", "u_phy", "v_phy")
ALLOCATES = (
    ChemAllocation("emis_seas", "rows_2d", restart="rebuild",
                   output_name="ESEAS{n}", units="kg per cell per step",
                   description="sea-salt emission this step, WRF EMIS_SEAS "
                               "(chem/module_gocart_seasalt.F:100-103)"),
)
#: WRF's gravity (share/module_model_constants.F:17, REAL 9.81).
G = 9.81


def rows(table):
    return table.rows_for(KEY)


def row_parameters(row) -> dict:
    """One sea-salt row's bin edges (um) and density, REAL*8 in WRF."""
    if row.settling is None or row.seasalt_emission is None:
        raise ValueError(f"sea-salt row {row.name!r} needs settling "
                         "(density) and seasalt_emission (bin edges)")
    e = row.seasalt_emission
    return {"ra": e["r_lo_um"], "rb": e["r_hi_um"],
            "den_seas": row.settling["density_kg_m3"]}


def _cache(ctx, dt):
    chem = ctx.state.chem
    cache = getattr(chem, "gocart_seasalt_cache", None)
    if cache is None:
        rs = rows(ctx.table)
        cache = chem.gocart_seasalt_cache = {
            "rows": rs,
            "params": pack_seasalt_rows([row_parameters(r) for r in rs]),
            "tables": {}}
    key = float(np.float32(dt))
    if key not in cache["tables"]:
        cache["tables"][key] = prepare_seasalt_table(cache["params"], key)
    return cache, cache["tables"][key]


def init(ctx):
    if int(getattr(ctx.cfg, "seas_opt", 0)) != 1:
        return
    missing = [name for name in REQUIRES if not ctx.has(name)]
    if missing:
        raise ValueError(f"seas_opt=1 reads {missing}, which this run does "
                         "not produce")


def step(ctx, dt, ktau):
    if int(getattr(ctx.cfg, "seas_opt", 0)) != 1:
        return
    cache, prepared = _cache(ctx, dt)
    rs = cache["rows"]
    emis = ctx.diag["emis_seas"]
    launch_seasalt([ctx.field(r) for r in rs],
                   [emis[i] for i in range(len(rs))], cache["params"],
                   xland=ctx.met("xland"), z_at_w=ctx.met("z_at_w"),
                   p8w=ctx.met("p8w"), dz8w=ctx.met("dz8w"),
                   u10=ctx.met("u10"), v10=ctx.met("v10"),
                   u_phy=ctx.met("u_phy"), v_phy=ctx.met("v_phy"),
                   dx=float(ctx.cfg.dx), dt=float(dt), g=G,
                   prepared=prepared)
