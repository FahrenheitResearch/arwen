"""Regional spectral Poisson and Helmholtz solvers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from .backend import namespace, scalar
from .grid import angular_wavenumbers, crop_even_extension, even_extension

ScalarBoundary = Literal["periodic", "reflect"]


@dataclass(frozen=True)
class EllipticResult:
    values: Any
    residual_rms: float
    rhs_mean_removed: float


def _solve_periodic(rhs: Any, *, dy_m: float, dx_m: float,
                    alpha: float, beta: float, zero_mode: str):
    xp = namespace(rhs)
    coeff = xp.fft.rfft2(rhs, axes=(-2, -1))
    _ky, _kx, k2, _magnitude = angular_wavenumbers(rhs, dy_m=dy_m, dx_m=dx_m)
    denominator = alpha + beta * k2
    singular = denominator == 0.0
    if bool(xp.any(singular)):
        if zero_mode == "raise":
            raise ValueError("elliptic operator has a singular zero mode")
        if zero_mode != "zero":
            raise ValueError("zero_mode must be 'zero' or 'raise'")
        denominator = xp.where(singular, 1.0, denominator)
        solution_coeff = xp.where(singular, 0.0, coeff / denominator)
    else:
        solution_coeff = coeff / denominator
    values = xp.fft.irfft2(solution_coeff, s=rhs.shape[-2:], axes=(-2, -1))
    return values.astype(rhs.dtype, copy=False), solution_coeff, k2


def solve_helmholtz(rhs: Any, *, dy_m: float, dx_m: float,
                    alpha: float, beta: float = 1.0,
                    boundary: ScalarBoundary = "reflect",
                    periodic_domain: bool = False,
                    zero_mode: str = "raise",
                    remove_rhs_mean: bool = False) -> EllipticResult:
    """Solve ``(alpha - beta*laplacian)x = rhs`` over the last two axes."""
    xp = namespace(rhs)
    values = xp.asarray(rhs)
    if values.ndim < 2 or values.size == 0 or not bool(xp.all(xp.isfinite(values))):
        raise ValueError("elliptic right-hand side must be non-empty and finite")
    if not all(math.isfinite(v) for v in (alpha, beta)) or alpha < 0.0 or beta <= 0.0:
        raise ValueError("elliptic coefficients require alpha >= 0 and beta > 0")
    removed = xp.zeros_like(xp.mean(values, axis=(-2, -1), keepdims=True))
    working = values
    if remove_rhs_mean:
        removed = xp.mean(values, axis=(-2, -1), keepdims=True)
        working = values - removed
    if boundary == "periodic":
        if not periodic_domain:
            raise ValueError("periodic elliptic solve requires periodic_domain=true")
        solution, solution_coeff, k2 = _solve_periodic(
            working, dy_m=dy_m, dx_m=dx_m, alpha=alpha, beta=beta,
            zero_mode=zero_mode)
        reconstructed = xp.fft.irfft2(
            (alpha + beta * k2) * solution_coeff,
            s=working.shape[-2:], axes=(-2, -1))
    elif boundary == "reflect":
        ny, nx = working.shape[-2:]
        extended = even_extension(working)
        full, coeff, k2 = _solve_periodic(
            extended, dy_m=dy_m, dx_m=dx_m, alpha=alpha, beta=beta,
            zero_mode=zero_mode)
        solution = crop_even_extension(full, ny, nx)
        reconstructed_full = xp.fft.irfft2(
            (alpha + beta * k2) * coeff,
            s=extended.shape[-2:], axes=(-2, -1))
        reconstructed = crop_even_extension(reconstructed_full, ny, nx)
    else:
        raise ValueError(f"unsupported elliptic boundary mode {boundary!r}")
    residual = reconstructed - working
    return EllipticResult(
        values=solution,
        residual_rms=scalar(xp.sqrt(xp.mean(residual * residual,
                                             dtype=xp.float64))),
        rhs_mean_removed=scalar(xp.mean(removed, dtype=xp.float64)),
    )


def solve_poisson(rhs: Any, *, dy_m: float, dx_m: float,
                  boundary: ScalarBoundary = "reflect",
                  periodic_domain: bool = False,
                  mean_policy: str = "remove") -> EllipticResult:
    """Solve ``-laplacian(x)=rhs`` with a pinned zero-mean solution."""
    if mean_policy not in ("remove", "raise"):
        raise ValueError("Poisson mean_policy must be 'remove' or 'raise'")
    xp = namespace(rhs)
    mean = xp.mean(rhs, axis=(-2, -1), keepdims=True)
    if mean_policy == "raise" and scalar(xp.max(xp.abs(mean))) > 1.0e-12:
        raise ValueError("Poisson right-hand side must have zero horizontal mean")
    return solve_helmholtz(
        rhs, dy_m=dy_m, dx_m=dx_m, alpha=0.0, beta=1.0,
        boundary=boundary, periodic_domain=periodic_domain,
        zero_mode="zero", remove_rhs_mean=(mean_policy == "remove"))


__all__ = ["EllipticResult", "ScalarBoundary", "solve_helmholtz", "solve_poisson"]
