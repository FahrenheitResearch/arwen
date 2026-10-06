"""The declared working set of the fork's edge-to-edge sixth-order filter.

``diff_6th_form = "noaa_wrf39"`` (the NOAA WRFV3.9 fork, the request default
of every HRRR recipe source and of ``hrrr_wrf.nl`` namelists in
``gpuwm/data/physics_sources/request-defaults.v1.toml``) runs
:func:`gpuwm.core.dycore.launch_diff6_to_edge` on every specified or nested
domain.  That launcher runs the unchanged diff6 kernel on copies of its
inputs padded with three zero-gradient halo cells (one more on the high
side), so it needs a padded field, a padded tendency, the four padded
planes (column mass and the three map factors) and, under the terrain-slope
taper, a padded base geopotential.

The breakage these slots close: until 2.8.6 those buffers were ``cp.pad``
and ``cp.zeros_like`` transients the run preflight never priced, on the
production HRRR route, growing with the nest (about three padded 3-D fields
and five padded planes per field per step).  A HRRR-route nest could pass
its memory check and still allocate them on top.  They are now
``DomainState.scratch`` slots priced by
:func:`gpuwm.core.preflight.scratch_slot_registry` (and so by the ensemble
plans that read it), with a write-before-read row in the lifetime audit.

This module imports no cupy: the preflight prices it on CPU-only installs.
Shapes are flat element counts; the launcher takes a shaped prefix of each,
so one slot holds whichever row (u, v, w, mass or TKE) is being filtered.
"""
from __future__ import annotations

#: Halo cells padded on each low side; the high side takes one more, so no
#: read of a real point reaches the periodic wrap of the padded core (a
#: staggered high face reads index n + 6 of n + 7).
DIFF6_EDGE_HALO = 3

#: Total padding per horizontal axis (low halo + high halo).
DIFF6_EDGE_PAD = 2 * DIFF6_EDGE_HALO + 1

SLOT_FIELD = "diff6_edge_field"
SLOT_TEND = "diff6_edge_tend"
SLOT_PLANES = "diff6_edge_planes"
SLOT_PHB = "diff6_edge_phb"

DIFF6_EDGE_SLOTS = (SLOT_FIELD, SLOT_TEND, SLOT_PLANES, SLOT_PHB)


def diff6_edge_active(cfg) -> bool:
    """True when production runs the edge form on this domain.

    Mirrors :func:`gpuwm.core.dycore.diff6_to_edge` (fork form and both
    horizontal axes forced, specified or nested) and adds the filter's own
    switch, without importing the cupy-backed dycore.
    """
    return bool(getattr(cfg, "diff_6th_opt", 0)
                and getattr(cfg, "diff_6th_form", "wrf_461") == "noaa_wrf39"
                and (getattr(cfg, "specified", False)
                     or getattr(cfg, "nested", False)))


def diff6_edge_slope(cfg) -> bool:
    """True when the terrain-slope taper reads the padded base geopotential."""
    return int(getattr(cfg, "diff_6th_slopeopt", 0) or 0) >= 1


def plane_shapes(ny: int, nx: int) -> tuple[tuple[int, int], ...]:
    """Padded (mut, msfu, msfv, msft) shapes for a mass grid (ny, nx)."""
    py, px = ny + DIFF6_EDGE_PAD, nx + DIFF6_EDGE_PAD
    return ((py, px), (py, px + 1), (py + 1, px), (py, px))


def field_values(nz: int, ny: int, nx: int) -> int:
    """Largest padded row: u (nz, ny, nx+1), v (nz, ny+1, nx), w (nz+1, ...).

    Mass rows (theta, moisture, scalars, TKE) are (nz, ny, nx) and fit in
    any of the three.
    """
    py, px = ny + DIFF6_EDGE_PAD, nx + DIFF6_EDGE_PAD
    return max(nz * py * (px + 1), nz * (py + 1) * px, (nz + 1) * py * px)


def planes_values(ny: int, nx: int) -> int:
    total = 0
    for py, px in plane_shapes(ny, nx):
        total += py * px
    return total


def phb_values(nz: int, ny: int, nx: int) -> int:
    """Padded base geopotential on w levels, (nz + 1, ny + 7, nx + 7)."""
    return (nz + 1) * (ny + DIFF6_EDGE_PAD) * (nx + DIFF6_EDGE_PAD)


def diff6_edge_slot_shapes(cfg) -> dict[str, tuple[int, ...]]:
    """The edge form's slots for one domain, empty when it does not run."""
    if not diff6_edge_active(cfg):
        return {}
    nz, ny, nx = int(cfg.nz), int(cfg.ny), int(cfg.nx)
    shapes = {SLOT_FIELD: (field_values(nz, ny, nx),),
              SLOT_TEND: (field_values(nz, ny, nx),),
              SLOT_PLANES: (planes_values(ny, nx),)}
    if diff6_edge_slope(cfg):
        shapes[SLOT_PHB] = (phb_values(nz, ny, nx),)
    return shapes


__all__ = [
    "DIFF6_EDGE_HALO", "DIFF6_EDGE_PAD", "DIFF6_EDGE_SLOTS", "SLOT_FIELD",
    "SLOT_TEND", "SLOT_PLANES", "SLOT_PHB", "diff6_edge_active",
    "diff6_edge_slope", "diff6_edge_slot_shapes", "field_values",
    "planes_values", "phb_values", "plane_shapes",
]
