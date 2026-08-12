"""RRTMGP's fixed VRAM: where it goes, and how much of it is reclaimable.

The measured whole-physics law on a 12 GB Blackwell card is

    peak = 1.90 GiB FIXED + 677 B/cell        (R^2 = 0.99933)

and walking the rung ladder puts ~1.63 GiB of that intercept on RRTMGP: mp10
alone is 0.31 GiB fixed, ``full(real74)+KF`` is 1.94 GiB.  This module answers
*which allocations* that is, in bytes, and separates the three reasons a byte
can land in the intercept rather than the slope:

``chunk``
    Sized by ``column_chunk`` (default 3125), so it stops growing the moment
    the domain exceeds 3125 columns -- i.e. any tile bigger than 56x56.  A
    least-squares fit over domain size therefore books it as intercept.  This
    is the overwhelming majority of the 1.63 GiB and the whole opportunity.
``table``
    Genuinely constant: the k-distribution and cloud-optics lookups, shared
    per process by ``lru_cache``.
``column``
    Sized by ``ncol = ny*nx``.  These are the honest slope; they are listed so
    the split is auditable, not because they can be reclaimed.

``layer`` marks the ``nz`` dependence within each of the above.

Nothing here allocates.  It transcribes the same two shape functions the
preflight estimator already uses -- ``rrtmgp_workspace_phases`` and
``rrtmgp_column_shapes`` -- so an allocation added to RRTMGP without updating
preflight fails preflight's own single-sourcing tests, not just this report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MIB = 1 << 20
GIB = 1 << 30

#: What ``gpuwm.config.DEFAULT_COLUMN_CHUNK`` is, restated so the report can
#: be read without the config module to hand.  Not imported as a default
#: argument anywhere -- callers pass it explicitly.
SHIPPED_COLUMN_CHUNK = 3125


@dataclass(frozen=True)
class Alloc:
    """One named device allocation with its exact byte count."""

    name: str
    shape: tuple[int, ...]
    itemsize: int
    scaling: str          # "chunk" | "column" | "table"
    layers: str           # "lw74" | "sw50" | "model49" | "-"
    lifetime: str         # "persistent" | "per-call" | "per-chunk"

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * self.itemsize


def workspace_allocations(nz: int, column_chunk: int,
                          p_top: float = 10000.0) -> list[Alloc]:
    """Every slot of the shared solver workspace, at its phase-max phase.

    ``SharedRRTMGPChunkWorkspace`` makes exactly ONE device allocation, of
    ``max(total(phase) for phase in phases)`` bytes, and lays each phase's
    live set over it.  So the phase that dominates is the one that is really
    resident; the others are aliases of the same bytes and are reported at
    their own totals for the reader's sake but must not be summed.
    """
    from gpuwm.core.preflight import rrtmgp_workspace_phases

    phases = rrtmgp_workspace_phases(nz, column_chunk, p_top)

    def total(items):
        return sum(math.prod(s) * i for s, i in items.values())

    peak = max(phases, key=lambda name: total(phases[name]))
    band = "lw74" if peak.startswith("lw") else "sw50"
    return [
        Alloc(f"workspace/{peak}/{slot}", tuple(shape), itemsize,
              "chunk", band, "persistent")
        for slot, (shape, itemsize) in phases[peak].items()
    ]


def workspace_phase_totals(nz: int, column_chunk: int,
                           p_top: float = 10000.0) -> dict[str, int]:
    """Bytes each solver phase would need, for the max-over-phases audit."""
    from gpuwm.core.preflight import rrtmgp_workspace_phases

    return {
        phase: sum(math.prod(s) * i for s, i in items.values())
        for phase, items in rrtmgp_workspace_phases(
            nz, column_chunk, p_top).items()
    }


def column_allocations(cfg, column_chunk: int,
                       p_top: float = 10000.0) -> list[Alloc]:
    """The per-call column-packing transients, split chunk vs column.

    ``rrtmgp_column_shapes`` keys the chunk-capped entries with the prefixes
    ``metadata_`` and ``upper_peak_``; everything else is ``ncol``-sized.  The
    split is read off the leading extent rather than the name so a renamed
    entry cannot silently change category.
    """
    from gpuwm.core.preflight import rrtmgp_column_shapes

    ncol = cfg.ny * cfg.nx
    cap = min(ncol, column_chunk)
    out = []
    for key, (shape, itemsize) in rrtmgp_column_shapes(
            cfg, p_top, column_chunk=column_chunk).items():
        lead = shape[0]
        if lead == cap and cap != ncol:
            scaling = "chunk"
        elif lead == ncol:
            scaling = "column"
        else:
            # cap == ncol: the domain is smaller than one chunk, so the two
            # categories are indistinguishable here.  Name it by the key,
            # which is the estimator's own classification.
            scaling = ("chunk" if key.split("/", 1)[1].startswith(
                ("metadata_", "upper_peak_")) else "column")
        layers = ("lw74" if "upper_peak" in key or "metadata" in key
                  else ("model49" if len(shape) > 1 else "-"))
        out.append(Alloc(key.replace("columns/", "call/"), tuple(shape),
                         itemsize, scaling, layers, "per-call"))
    return out


def table_allocations() -> list[Alloc]:
    """The lru_cache-shared k-distribution and cloud-optics device tables."""
    import numpy as np
    from gpuwm.core.rrtmgp import load_cloud_tables, load_gas_tables

    out = []
    for kind, tables in (("gas_lw", load_gas_tables("lw")),
                         ("gas_sw", load_gas_tables("sw")),
                         ("cloud_lw", load_cloud_tables("lw")),
                         ("cloud_sw", load_cloud_tables("sw"))):
        for name, value in vars(tables).items():
            if isinstance(value, np.ndarray):
                itemsize = 1 if value.dtype == bool else 4
                out.append(Alloc(f"tables/{kind}/{name}", value.shape,
                                 itemsize, "table", "-", "persistent"))
    return out


def attribute(cfg, column_chunk: int = SHIPPED_COLUMN_CHUNK,
              p_top: float = 10000.0) -> list[Alloc]:
    """The full allocation list for one configuration."""
    return (workspace_allocations(cfg.nz, column_chunk, p_top)
            + column_allocations(cfg, column_chunk, p_top)
            + table_allocations())


def totals_by(allocs: list[Alloc], key) -> dict[str, int]:
    out: dict[str, int] = {}
    for a in allocs:
        out[key(a)] = out.get(key(a), 0) + a.nbytes
    return out


def report(cfg, column_chunk: int = SHIPPED_COLUMN_CHUNK,
           p_top: float = 10000.0, *, top: int = 24) -> str:
    """A human-readable attribution table, largest allocation first."""
    allocs = attribute(cfg, column_chunk, p_top)
    ncol = cfg.ny * cfg.nx
    lines = [
        f"RRTMGP VRAM attribution   nz={cfg.nz}  ny x nx = {cfg.ny}x{cfg.nx}"
        f"  ncol={ncol}  column_chunk={column_chunk}  p_top={p_top:.0f}",
        "",
        f"{'allocation':<44}{'shape':>22}{'MiB':>10}  scaling",
        "-" * 88,
    ]
    for a in sorted(allocs, key=lambda a: -a.nbytes)[:top]:
        lines.append(f"{a.name:<44}{str(a.shape):>22}"
                     f"{a.nbytes / MIB:>10.2f}  {a.scaling}")
    rest = sorted(allocs, key=lambda a: -a.nbytes)[top:]
    if rest:
        lines.append(f"{f'... {len(rest)} smaller':<44}{'':>22}"
                     f"{sum(a.nbytes for a in rest) / MIB:>10.2f}")
    lines.append("-" * 88)
    by_scaling = totals_by(allocs, lambda a: a.scaling)
    fixed = by_scaling.get("chunk", 0) + by_scaling.get("table", 0)
    for name in ("chunk", "table", "column"):
        n = by_scaling.get(name, 0)
        lines.append(f"{name + ' total':<44}{'':>22}{n / MIB:>10.2f}")
    lines.append(f"{'FIXED (chunk + table)':<44}{'':>22}{fixed / MIB:>10.2f}"
                 f"   = {fixed / GIB:.3f} GiB")
    per_cell = by_scaling.get("column", 0) / (ncol * cfg.nz)
    lines.append(f"{'per-cell (column total / ncol / nz)':<44}{'':>22}"
                 f"{per_cell:>10.1f}  B/cell")
    return "\n".join(lines)
