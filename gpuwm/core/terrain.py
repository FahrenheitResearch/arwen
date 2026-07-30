"""Analytic terrain profiles (setup-time, pure NumPy float64).

Phase 2 Task 3: ``bell_hill`` builds the classic bell-shaped ridge (Witch
of Agnesi) used by the mountain-wave benchmarks.  The profile feeds
``grid.make_base_state(..., terrain_z=...)``, which is the terrain
authority for the run — the surface enters the model through the base
state (``phb[0] = g*h``, per-column dry mass), never through ``phi'``.
"""

from __future__ import annotations

import numpy as np

from gpuwm.config import RunConfig


def bell_hill(cfg: RunConfig) -> np.ndarray:
    """Bell-shaped ridge centered in the domain, ``(ny, nx)`` float64.

    ``h(x, y) = hill_height / (1 + ((x - xc) / hill_halfwidth)^2)`` on the
    domain-centered cell-center x grid (so xc is the domain center),
    uniform in y — a 2-D ridge; ``ny == 1`` gives the classic x-z
    mountain-wave setup.
    """
    x = (np.arange(cfg.nx) + 0.5) * cfg.dx - 0.5 * cfg.nx * cfg.dx
    h = cfg.hill_height / (1.0 + (x / cfg.hill_halfwidth) ** 2)
    return np.broadcast_to(h, (cfg.ny, cfg.nx)).copy()
