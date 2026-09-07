"""Classic RRTM LW allocation shapes and its shared column-cap contract.

These are live-array envelopes for the CuPy transcription, not a coefficient
fitted to one grid/card. The profile, band absorption and transfer phases do
not coexist. Each phase carries its caller's retained arrays; the transfer
phase includes all eleven 140-g-point grids actually retained by the current
driver. Allocator retention/headroom is a separate preflight concern.
"""
from __future__ import annotations

import math
from functools import lru_cache
from numbers import Integral

Shape = tuple[tuple[int, ...], int]


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def effective_column_chunk(configured, ncol: int) -> int:
    """Actual largest block; an explicit positive pin is never retuned."""
    columns = _positive(ncol, "ncol")
    return min(columns, _positive(configured, "column_chunk"))


def allocation_bytes(shapes: dict[str, Shape], *, alignment: int = 1) -> int:
    """Sum independent allocations, optionally rounding each pool request."""
    return sum(((math.prod(shape) * size + alignment - 1) // alignment) * alignment
               for shape, size in shapes.values())


def chunk_workspace_phases(ncol: int, nlayers: int) -> dict[str, dict[str, Shape]]:
    """Named conservative live sets of ``rrtm_longwave_columns``.

    Model-depth temporaries are bounded at radiation depth, which includes
    the actual pressure-dependent Cavallo buffer. Small CuPy expressions are
    represented by separate operands rather than a bytes-per-cell fit.
    """
    c, z = _positive(ncol, "ncol"), _positive(nlayers, "nlayers")
    layer, level, column = (c, z), (c, z + 1), (c,)
    profile = {f"profile/{name}": (layer, 4) for name in
               ("pavel", "tavel", "coldry", "cldfrac", "taucloud")}
    profile.update({"profile/wkl": ((c, z, 35), 4),
                    "profile/wx": ((c, z, 4), 4),
                    "profile/pz": (level, 4), "profile/tz": (level, 4),
                    "profile/tbound": (column, 4)})
    coef = {f"coef/{name}": (layer, 4) for name in (
        "colh2o", "colco2", "colo3", "coln2o", "colch4", "colo2", "co2mult",
        "fac00", "fac01", "fac10", "fac11", "forfac", "selffac", "selffrac",
        "jp", "jt", "jt1", "indself")}
    coef.update({f"coef/{name}": (column, 4)
                 for name in ("laytrop", "layswtch", "laylow")})
    # MM5ATM/O3DATA: four pressure overlaps and at most three RHS operands
    # at the 31-level ozone climatology extent. All other profile construction
    # operands have at most one radiation-depth plane per local name.
    atmosphere = dict(profile)
    atmosphere.update({f"ozone/{name}": ((c, z, 31), 4)
                       for name in ("pb1", "pb2", "pt1", "pt2", "rhs0", "rhs1", "rhs2")})
    atmosphere.update({f"profile_work/{i}": (level, 4) for i in range(32)})
    # TAUMOL keeps earlier bands' named locals alive. Its widest band is 16
    # g-points (not 140): 32 band arrays cover retained low/high/interpolation
    # operands; 128 scalar/index planes cover the accumulated band locals.
    # These deliberate upper sets also cover SETCOEF before TAUMOL starts.
    absorption = {**profile, **coef,
                  "absorption/taug": ((c, z, 140), 4),
                  "absorption/pfrac": ((c, z, 140), 4)}
    absorption.update({f"band_work/{i}": ((c, z, 16), 4) for i in range(32)})
    absorption.update({f"band_scalar/{i}": (layer, 4) for i in range(128)})
    absorption.update({f"band_mask/{i}": (layer, 1) for i in range(16)})
    transfer = {**profile, **coef}
    # Five arrays retained by rrtm_longwave_columns and six owned by RTRN.
    for name in ("taug", "pfrac", "odepth", "tff", "itr", "odclr", "tauf",
                 "abss", "bbu", "bbutot", "atot"):
        transfer[f"gpoint/{name}"] = ((c, z, 140), 4)
    for name in ("plankbnd", "plnkemit"):
        transfer[f"planck/{name}"] = ((c, 16), 4)
    transfer["planck/plvl"] = ((c, z + 1, 16), 4)
    transfer["planck/play"] = ((c, z, 16), 4)
    for name in ("indlev", "tlevfrac", "indlay", "tlayfrac", "odcld", "abscld",
                 "efclfrac", "totdflux", "totdclfl", "totuflux", "totuclfl",
                 "fnet", "fnetc", "dp", "htr", "htrc", "rhs0", "rhs1", "rhs2"):
        transfer[f"transfer/{name}"] = (level, 4)
    transfer["transfer/icldlyr"] = (layer, 1)
    # 21 named spectral row arrays survive the downward loop into the upward
    # loop, plus old assignment values and RHS operands. 32 rows bound those
    # simultaneously live arrays without multiplying them by layer count.
    transfer.update({f"spectral_row/{i}": ((c, 140), 4) for i in range(32)})
    transfer.update({f"scalar_row/{i}": (column, 4) for i in range(16)})
    # The adapter can retain the previous chunk's returned fluxes/tendencies
    # while the next chunk call evaluates. These are counted in every phase.
    for phase in (atmosphere, absorption, transfer):
        phase.update({f"previous_result/{name}": (level, 4) for name in
                      ("tten", "ttenc", "htr", "htrc", "totdflux", "totuflux", "pz")})
        phase["pressure_input/model"] = (layer, 4)
        phase["pressure_input/interface"] = (level, 4)
    return {"profile": atmosphere, "absorption": absorption, "transfer": transfer}


def chunk_workspace_shapes(ncol: int, nlayers: int) -> dict[str, Shape]:
    phases = chunk_workspace_phases(ncol, nlayers)
    peak = max(phases, key=lambda name: allocation_bytes(phases[name], alignment=512))
    return {f"{peak}/{name}": item for name, item in phases[peak].items()}


@lru_cache(maxsize=256)
def chunk_workspace_bytes(ncol: int, nlayers: int) -> int:
    """Live allocation envelope, with CuPy's 512-byte pool request rounding."""
    return allocation_bytes(chunk_workspace_shapes(ncol, nlayers), alignment=512)


def auto_column_chunk(ncol: int, nlayers: int, budget_bytes: int) -> int:
    """Largest block whose allocation envelope fits the caller's allowance.

    There is no artificial 512-column floor: that can exceed available memory
    even though a smaller mathematically identical block fits.
    """
    columns = _positive(ncol, "ncol")
    if budget_bytes < chunk_workspace_bytes(1, nlayers):
        raise MemoryError("classic RRTM longwave cannot fit one column in its available "
                          "workspace; release device memory or reduce the forecast tile")
    lo, hi = 1, columns
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if chunk_workspace_bytes(mid, nlayers) <= budget_bytes:
            lo = mid
        else:
            hi = mid - 1
    return lo


def column_packing_shapes(ncol: int, nz: int) -> dict[str, Shape]:
    """Full-window adapter packing, cloud preparation, and returned copies."""
    c, z = _positive(ncol, "ncol"), _positive(nz, "nz")
    shapes = {f"packed/{name}": ((c, z), 4) for name in (
        "zero", "qv", "qc", "qr", "qi", "qs", "qg", "temperature", "pressure",
        "dz", "cldfra", "tten", "heating", "rthratenlw", "rthratensw")}
    # cal_cldfra1 and optional MYNN BL cloud merge execute before the solver;
    # ten layer operands cover their saturation, blend and selection live set.
    shapes.update({f"cloud_work/{i}": ((c, z), 4) for i in range(10)})
    shapes.update({f"interface/{name}": ((c, z + 1), 4)
                   for name in ("pressure", "height", "t8w", "tw", "t8w_work")})
    shapes.update({f"surface/{name}": ((c,), 4) for name in
                   ("glw", "olr", "swdown", "gsw", "coszen", "t8w_work0", "t8w_work1")})
    return shapes


@lru_cache(maxsize=1)
def table_allocation_shapes() -> dict[str, Shape]:
    """The actual packaged coefficient arrays copied by the classic driver.

    TAUMOL's device cache and the driver's constants cache are distinct, so
    the latter's copied statics are intentionally counted again. This reads
    the established table loader and never imports CuPy or probes a device.
    """
    from gpuwm.core.rrtm_tables import load_rrtm_lw_tables
    tables = load_rrtm_lw_tables()
    shapes = {}
    for group in ("absa", "absb", "selfrefc", "combined", "statics"):
        for name, value in getattr(tables, group).items():
            shapes[f"taumol/{group}/{name}"] = (value.shape, value.dtype.itemsize)
    for name in ("corr1", "corr2"):
        value = getattr(tables, name)
        shapes[f"taumol/{name}"] = (value.shape, value.dtype.itemsize)
    for name in ("MM5ATM__PPROF", "MM5ATM__TPROF", "PREFLOG", "TREF",
                 "TOTPLNK", "DELWAVE", "NGB"):
        shapes[f"driver/{name}"] = (tables.statics[name].shape, 4)
    for name in ("tau", "tf", "trans"):
        shapes[f"driver/{name}"] = (getattr(tables, name).shape, 4)
    shapes["driver/o3wrk"] = ((31,), 4)
    shapes["driver/ppwrkh"] = ((32,), 4)
    return shapes


def call_workspace_shapes(ncol: int, nz: int, p_top: float, column_chunk: int
                          ) -> dict[str, Shape]:
    """One actual domain/window call, including its first-use coefficient cache."""
    from gpuwm.core.rrtm_lw import rrtm_layer_count
    chunk = effective_column_chunk(column_chunk, ncol)
    shapes = {**column_packing_shapes(ncol, nz),
              **chunk_workspace_shapes(chunk, rrtm_layer_count(nz, p_top)),
              **{f"tables/{name}": item for name, item in table_allocation_shapes().items()}}
    padding = allocation_bytes(shapes, alignment=512) - allocation_bytes(shapes)
    if padding:
        shapes["pool_request_rounding"] = ((padding,), 1)
    return shapes


@lru_cache(maxsize=256)
def call_workspace_bytes(ncol: int, nz: int, p_top: float, column_chunk: int) -> int:
    return allocation_bytes(call_workspace_shapes(ncol, nz, p_top, column_chunk))
