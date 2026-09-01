"""Physical wavenumber grids and boundary windows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .backend import namespace


@dataclass(frozen=True)
class SpectralGrid:
    ny: int
    nx: int
    dy_m: float
    dx_m: float

    def __post_init__(self) -> None:
        if self.ny < 2 or self.nx < 2:
            raise ValueError("a spectral grid needs at least 2x2 cells")
        if not all(math.isfinite(v) and v > 0.0 for v in (self.dx_m, self.dy_m)):
            raise ValueError("dx_m and dy_m must be positive and finite")

    @property
    def isotropic_nyquist_rad_m(self) -> float:
        return math.pi / max(self.dx_m, self.dy_m)


def angular_wavenumbers(array: Any, *, dy_m: float, dx_m: float):
    """``(ky, kx, k2, |k|)`` for an rfft2 over the last two axes."""
    xp = namespace(array)
    ny, nx = array.shape[-2:]
    ky = 2.0 * math.pi * xp.fft.fftfreq(ny, d=float(dy_m))
    kx = 2.0 * math.pi * xp.fft.rfftfreq(nx, d=float(dx_m))
    ky2 = ky[:, None]
    kx2 = kx[None, :]
    k2 = ky2 * ky2 + kx2 * kx2
    return ky2, kx2, k2, xp.sqrt(k2)


def edge_window(array: Any, width_cells: int):
    """Raised-cosine 2-D window, zero at edges and one in the interior."""
    xp = namespace(array)
    ny, nx = array.shape[-2:]
    width = int(width_cells)
    if width < 0:
        raise ValueError("edge_taper_cells must be non-negative")
    if width == 0:
        return xp.ones((ny, nx), dtype=array.dtype)
    if 2 * width >= min(ny, nx):
        raise ValueError(
            f"edge_taper_cells={width} leaves no untapered interior on {ny}x{nx}")
    y = xp.arange(ny, dtype=xp.float64)
    x = xp.arange(nx, dtype=xp.float64)
    dy = xp.minimum(y, ny - 1 - y)
    dx = xp.minimum(x, nx - 1 - x)
    wy = 0.5 - 0.5 * xp.cos(math.pi * xp.clip(dy / width, 0.0, 1.0))
    wx = 0.5 - 0.5 * xp.cos(math.pi * xp.clip(dx / width, 0.0, 1.0))
    return (wy[:, None] * wx[None, :]).astype(array.dtype, copy=False)


def even_extension(array: Any):
    """Cell-centred even extension over x and y, preserving leading axes."""
    xp = namespace(array)
    x_extended = xp.concatenate((array, xp.flip(array, axis=-1)), axis=-1)
    return xp.concatenate((x_extended, xp.flip(x_extended, axis=-2)), axis=-2)


def crop_even_extension(array: Any, ny: int, nx: int):
    return array[..., :ny, :nx]


__all__ = ["SpectralGrid", "angular_wavenumbers", "crop_even_extension",
           "edge_window", "even_extension"]
