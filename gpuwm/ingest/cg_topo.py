"""WRF's ``smooth_cg_topo``: the root terrain blended toward the source's.

WRF v4.7.1 ``&domains smooth_cg_topo`` (Registry.EM_COMMON:2301, logical,
scope 1, default ``.false.``).  With it on, real.exe replaces the outer
rows of domain 1's terrain with the INPUT MODEL's terrain and ramps back to
the high-resolution terrain over the next ``blend_width`` rows
(dyn_em/module_initialize_real.F:716-762 calls ``blend_terrain``,
dyn_em/nest_init_utils.F:712-785, with ``grid%toposoil`` -- the source's
SOILHGT on the d01 grid -- as the coarse field).  Every later forcing time
reuses the first time's blended terrain (``ht_smooth``, :754-761), so the
initial state, the lateral boundaries and the terrain the forecast
integrates all stand on one blended surface, and the specified zone's
boundary values are no longer interpolated onto terrain the input model
never saw.

The comment WRF carries beside the call is the reason for it: "This
smoothing is similar to the coarse/nest interface.  The outer rows and
cols come from the existing large scale topo, and then the next several
rows/cols are a linear ramp of the large scale model and the hi-res topo
from WPS."  Domain 1 only; a nest's terrain is already blended toward its
parent's by nest initialization.

WRF refuses the request when the input carries no SOILHGT (:729-738,
"The field SOILHGT is required when smoothing the CG topography on d01"),
and so does this module.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

def blend_root_terrain(hgt, source_orography, *, spec_bdy_width: int,
                       blend_width: int) -> np.ndarray:
    """``blend_terrain(toposoil, ht)`` on the d01 mass grid, in FP32.

    ``hgt`` and ``source_orography`` are ``(ny, nx)``.  WRF's operands are
    REAL: the blended rows are single-precision
    ``(blend_cell*ht + (blend_width+1-blend_cell)*toposoil) * r_blend_zones``
    with ``r_blend_zones = 1./(blend_width+1)`` (nest_init_utils.F:753-765),
    the specified rows are the source terrain (:766-769), and the interior
    keeps the high-resolution terrain.  The loop runs inside out
    (``blend_cell`` from ``blend_width`` down to 1) and the last match wins,
    so a corner cell takes the ring nearest the boundary.  The returned
    array keeps ``hgt``'s dtype; the interior values are ``hgt``'s own.
    """
    fine64 = np.asarray(hgt)
    if fine64.ndim != 2:
        raise ValueError("root terrain must be a (ny, nx) mass-grid field")
    coarse = np.asarray(source_orography)
    if coarse.shape != fine64.shape:
        raise ValueError(
            f"source terrain {coarse.shape} does not match the root grid "
            f"{fine64.shape}")
    if not np.isfinite(coarse).all():
        raise ValueError("source terrain on the root grid is not finite")
    sbw, width = int(spec_bdy_width), int(blend_width)
    if sbw < 1 or width < 0:
        raise ValueError("spec_bdy_width must be >= 1 and blend_width >= 0")
    F = np.float32
    fine = fine64.astype(F)
    coarse = coarse.astype(F)
    ny, nx = fine.shape
    ide, jde = nx + 1, ny + 1                 # staggered ends, as in WRF
    i1 = np.arange(1, nx + 1)[None, :]
    j1 = np.arange(1, ny + 1)[:, None]
    r_blend_zones = F(1.0) / F(width + 1)
    out = fine64.copy()
    for blend_cell in range(width, 0, -1):
        hit = ((i1 == sbw + blend_cell) | (j1 == sbw + blend_cell)
               | (i1 == ide - sbw - blend_cell)
               | (j1 == jde - sbw - blend_cell))
        blended = ((F(blend_cell) * fine
                    + F(width + 1 - blend_cell) * coarse) * r_blend_zones)
        out = np.where(hit, blended.astype(out.dtype), out)
    specified = ((i1 <= sbw) | (j1 <= sbw)
                 | (i1 >= ide - sbw) | (j1 >= jde - sbw))
    return np.where(specified, coarse.astype(out.dtype), out)


def apply_smooth_cg_topo(exp, static: Mapping, *, source_orography,
                         route: str):
    """``static`` with ``HGT_M`` blended for WRF's smooth_cg_topo, when asked.

    Returned unchanged when the experiment does not set it.
    ``source_orography`` is the source's terrain on the root mass grid (the
    SOILHGT the initialization reads), or ``None`` when the input carries
    none, which is WRF's fatal and is refused here with its words.  What
    was blended is bound by the experiment itself: ``smooth_cg_topo`` sits
    in the restart and prepared-tree identity whenever it is on
    (gpuwm.core.model.restart_identity_payload), so a root prepared with
    it cannot serve a run without it, or the reverse.
    """
    if not getattr(exp, "smooth_cg_topo", False):
        return static
    if source_orography is None:
        raise ValueError(
            "smooth_cg_topo = .true. blends domain 1's terrain toward the "
            "input model's, and this source carries no terrain of its own "
            f"on the {route} route (WRF: 'The field SOILHGT is required "
            "when smoothing the CG topography on d01', "
            "dyn_em/module_initialize_real.F:729-738).  Supply the source's "
            "surface geopotential or orography, or set smooth_cg_topo = "
            ".false.")
    out = dict(static)
    out["HGT_M"] = blend_root_terrain(
        static["HGT_M"], source_orography,
        spec_bdy_width=int(exp.spec_bdy_width),
        blend_width=int(exp.blend_width))
    return out


class RootTerrainBlend:
    """Blend a route's root terrain once, at its first initialization.

    WRF blends at real.exe's first time and reuses that terrain for every
    later one (``ht_smooth``); the source terrain it blends toward is the
    input model's, which does not change between forcing times.  A route
    hands this the SOURCE_OROGRAPHY of whichever forcing time it
    interpolates first (a hierarchy builds the start time last), and every
    initialization, boundary frame, soil adjustment and published static
    after that stands on the blended terrain.  The route's vertical
    adaptation has already read the unblended terrain by then; it is
    gpuwm's own step-size heuristic, not a WRF quantity, and the blend only
    eases the outer rows toward smoother terrain.
    """

    def __init__(self, exp, static, *, route: str):
        self.exp = exp
        self.static = static
        self.route = route
        self.done = not getattr(exp, "smooth_cg_topo", False)

    def before_initialize(self, source_orography) -> None:
        if self.done:
            return
        self.static["HGT_M"] = apply_smooth_cg_topo(
            self.exp, self.static, source_orography=source_orography,
            route=self.route)["HGT_M"]
        self.done = True
