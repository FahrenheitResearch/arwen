"""Fourier Helmholtz decomposition and scale-selective divergence control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .backend import namespace, scalar
from .grid import angular_wavenumbers, edge_window
from .transfer import Hyperdiffusion

VectorBoundary = Literal["periodic", "tapered"]


@dataclass(frozen=True)
class VectorResult:
    u: Any
    v: Any
    divergence_rms_before: float
    divergence_rms_after: float
    kinetic_energy_before: float
    kinetic_energy_after: float
    rms_increment: float
    max_abs_increment: float


def _check(u: Any, v: Any):
    xp = namespace(u)
    if namespace(v) is not xp:
        raise ValueError("u and v must use one array backend")
    if u.shape != v.shape or u.ndim < 2 or u.size == 0:
        raise ValueError("u and v must have one non-empty common mass-grid shape")
    if not bool(xp.all(xp.isfinite(u))) or not bool(xp.all(xp.isfinite(v))):
        raise ValueError("wind spectral operand is non-finite")
    return xp


def helmholtz_coefficients(u: Any, v: Any, *, dy_m: float, dx_m: float):
    """Return rotational and divergent rfft2 coefficient pairs."""
    xp = _check(u, v)
    uhat = xp.fft.rfft2(u, axes=(-2, -1))
    vhat = xp.fft.rfft2(v, axes=(-2, -1))
    ky, kx, k2, magnitude = angular_wavenumbers(u, dy_m=dy_m, dx_m=dx_m)
    safe = xp.where(k2 > 0.0, k2, 1.0)
    projection = (kx * uhat + ky * vhat) / safe
    udiv = kx * projection
    vdiv = ky * projection
    udiv = xp.where(k2 > 0.0, udiv, 0.0)
    vdiv = xp.where(k2 > 0.0, vdiv, 0.0)
    return (uhat - udiv, vhat - vdiv), (udiv, vdiv), magnitude, kx, ky


def helmholtz_decompose(u: Any, v: Any, *, dy_m: float, dx_m: float):
    xp = _check(u, v)
    rotational, divergent, _magnitude, _kx, _ky = helmholtz_coefficients(
        u, v, dy_m=dy_m, dx_m=dx_m)
    shape = u.shape[-2:]
    urot = xp.fft.irfft2(rotational[0], s=shape, axes=(-2, -1))
    vrot = xp.fft.irfft2(rotational[1], s=shape, axes=(-2, -1))
    udiv = xp.fft.irfft2(divergent[0], s=shape, axes=(-2, -1))
    vdiv = xp.fft.irfft2(divergent[1], s=shape, axes=(-2, -1))
    return (urot, vrot), (udiv, vdiv)


def spectral_divergence(u: Any, v: Any, *, dy_m: float, dx_m: float):
    xp = _check(u, v)
    uhat = xp.fft.rfft2(u, axes=(-2, -1))
    vhat = xp.fft.rfft2(v, axes=(-2, -1))
    ky, kx, _k2, _magnitude = angular_wavenumbers(u, dy_m=dy_m, dx_m=dx_m)
    divergence = 1j * (kx * uhat + ky * vhat)
    return xp.fft.irfft2(divergence, s=u.shape[-2:], axes=(-2, -1))


def _periodic_damp(u: Any, v: Any, *, dy_m: float, dx_m: float, dt_s: float,
                   divergent: Hyperdiffusion,
                   rotational: Hyperdiffusion | None = None):
    xp = _check(u, v)
    rot, div, magnitude, _kx, _ky = helmholtz_coefficients(
        u, v, dy_m=dy_m, dx_m=dx_m)
    d_transfer = divergent.transfer(magnitude, dt_s=dt_s)
    r_transfer = (xp.ones_like(d_transfer) if rotational is None
                  else rotational.transfer(magnitude, dt_s=dt_s))
    uhat = rot[0] * r_transfer + div[0] * d_transfer
    vhat = rot[1] * r_transfer + div[1] * d_transfer
    shape = u.shape[-2:]
    return (xp.fft.irfft2(uhat, s=shape, axes=(-2, -1)).astype(u.dtype, copy=False),
            xp.fft.irfft2(vhat, s=shape, axes=(-2, -1)).astype(v.dtype, copy=False))


def damp_divergence(u: Any, v: Any, *, dy_m: float, dx_m: float, dt_s: float,
                    divergent: Hyperdiffusion,
                    rotational: Hyperdiffusion | None = None,
                    boundary: VectorBoundary = "tapered",
                    edge_taper_cells: int = 12,
                    periodic_domain: bool = False) -> VectorResult:
    """Damp divergent modes while retaining the rotational component.

    The exact orthogonal projector is periodic.  For a limited-area model the
    default ``tapered`` mode applies that projector to a mean-removed, tapered
    interior and blends only its correction back, making the operation zero at
    every outer-edge cell.  It is therefore an interior divergence controller,
    not a claim that a non-periodic regional wind has a unique global Helmholtz
    split.
    """
    xp = _check(u, v)
    u0, v0 = xp.asarray(u), xp.asarray(v)
    if boundary == "periodic":
        if not periodic_domain:
            raise ValueError(
                "periodic wind projection requires periodic_domain=true")
        out_u, out_v = _periodic_damp(
            u0, v0, dy_m=dy_m, dx_m=dx_m, dt_s=dt_s,
            divergent=divergent, rotational=rotational)
    elif boundary == "tapered":
        window = edge_window(u0, edge_taper_cells)
        umean = xp.mean(u0, axis=(-2, -1), keepdims=True)
        vmean = xp.mean(v0, axis=(-2, -1), keepdims=True)
        work_u, work_v = (u0 - umean) * window, (v0 - vmean) * window
        filt_u, filt_v = _periodic_damp(
            work_u, work_v, dy_m=dy_m, dx_m=dx_m, dt_s=dt_s,
            divergent=divergent, rotational=rotational)
        du, dv = window * (filt_u - work_u), window * (filt_v - work_v)
        weight_mean = xp.mean(window, axis=(-2, -1), keepdims=True)
        du = du - window * xp.mean(du, axis=(-2, -1), keepdims=True) / weight_mean
        dv = dv - window * xp.mean(dv, axis=(-2, -1), keepdims=True) / weight_mean
        out_u, out_v = u0 + du, v0 + dv
    else:
        raise ValueError(f"unsupported vector spectral boundary mode {boundary!r}")
    before_div = spectral_divergence(u0, v0, dy_m=dy_m, dx_m=dx_m)
    after_div = spectral_divergence(out_u, out_v, dy_m=dy_m, dx_m=dx_m)
    du, dv = out_u - u0, out_v - v0
    before_ke = 0.5 * xp.mean(u0 * u0 + v0 * v0, dtype=xp.float64)
    after_ke = 0.5 * xp.mean(out_u * out_u + out_v * out_v, dtype=xp.float64)
    return VectorResult(
        u=out_u.astype(u0.dtype, copy=False),
        v=out_v.astype(v0.dtype, copy=False),
        divergence_rms_before=scalar(xp.sqrt(xp.mean(
            before_div * before_div, dtype=xp.float64))),
        divergence_rms_after=scalar(xp.sqrt(xp.mean(
            after_div * after_div, dtype=xp.float64))),
        kinetic_energy_before=scalar(before_ke),
        kinetic_energy_after=scalar(after_ke),
        rms_increment=scalar(xp.sqrt(xp.mean(
            0.5 * (du * du + dv * dv), dtype=xp.float64))),
        max_abs_increment=scalar(xp.maximum(xp.max(xp.abs(du)),
                                            xp.max(xp.abs(dv)))),
    )


def destagger_c_grid(u: Any, v: Any):
    """Destagger WRF/Arwen C-grid horizontal wind to mass points."""
    if u.shape[:-2] != v.shape[:-2]:
        raise ValueError("C-grid u/v leading dimensions differ")
    if u.shape[-2] + 1 != v.shape[-2] or u.shape[-1] != v.shape[-1] + 1:
        raise ValueError(
            "C-grid shapes require U(...,ny,nx+1) and V(...,ny+1,nx)")
    return (0.5 * (u[..., :, :-1] + u[..., :, 1:]),
            0.5 * (v[..., :-1, :] + v[..., 1:, :]))


def lift_mass_increment_to_c_grid(du: Any, dv: Any, *, periodic: bool = False):
    """Lift mass-point wind increments to C-grid faces.

    Non-periodic outer faces receive zero increment, preserving externally
    supplied lateral-boundary values.  Periodic faces use the wrapped average.
    """
    xp = _check(du, dv)
    ny, nx = du.shape[-2:]
    out_u = xp.zeros((*du.shape[:-2], ny, nx + 1), dtype=du.dtype)
    out_v = xp.zeros((*dv.shape[:-2], ny + 1, nx), dtype=dv.dtype)
    out_u[..., :, 1:nx] = 0.5 * (du[..., :, :-1] + du[..., :, 1:])
    out_v[..., 1:ny, :] = 0.5 * (dv[..., :-1, :] + dv[..., 1:, :])
    if periodic:
        edge_u = 0.5 * (du[..., :, -1] + du[..., :, 0])
        edge_v = 0.5 * (dv[..., -1, :] + dv[..., 0, :])
        out_u[..., :, 0] = edge_u
        out_u[..., :, -1] = edge_u
        out_v[..., 0, :] = edge_v
        out_v[..., -1, :] = edge_v
    return out_u, out_v


def damp_c_grid_divergence(u: Any, v: Any, **kwargs) -> VectorResult:
    """Control C-grid wind and report the *realized* lifted-face result.

    The mass-grid projector proposes an increment, then the increment is lifted
    to faces under the boundary contract.  That lift is a smoothing operation,
    so its realized divergence and energy are recomputed after destaggering the
    updated faces; a receipt must never report the un-lifted proposal as if it
    were the state that was actually written.
    """
    xp = namespace(u)
    mass_u, mass_v = destagger_c_grid(u, v)
    proposal = damp_divergence(mass_u, mass_v, **kwargs)
    du_mass, dv_mass = proposal.u - mass_u, proposal.v - mass_v
    periodic = kwargs.get("boundary") == "periodic"
    du_face, dv_face = lift_mass_increment_to_c_grid(
        du_mass, dv_mass, periodic=periodic)
    out_u, out_v = u + du_face, v + dv_face
    realized_u, realized_v = destagger_c_grid(out_u, out_v)
    dy_m, dx_m = float(kwargs["dy_m"]), float(kwargs["dx_m"])
    before_div = spectral_divergence(mass_u, mass_v, dy_m=dy_m, dx_m=dx_m)
    after_div = spectral_divergence(
        realized_u, realized_v, dy_m=dy_m, dx_m=dx_m)
    realized_du, realized_dv = realized_u - mass_u, realized_v - mass_v
    return VectorResult(
        u=out_u, v=out_v,
        divergence_rms_before=scalar(xp.sqrt(xp.mean(
            before_div * before_div, dtype=xp.float64))),
        divergence_rms_after=scalar(xp.sqrt(xp.mean(
            after_div * after_div, dtype=xp.float64))),
        kinetic_energy_before=scalar(0.5 * xp.mean(
            mass_u * mass_u + mass_v * mass_v, dtype=xp.float64)),
        kinetic_energy_after=scalar(0.5 * xp.mean(
            realized_u * realized_u + realized_v * realized_v,
            dtype=xp.float64)),
        rms_increment=scalar(xp.sqrt(xp.mean(
            0.5 * (realized_du * realized_du + realized_dv * realized_dv),
            dtype=xp.float64))),
        max_abs_increment=scalar(xp.maximum(xp.max(xp.abs(du_face)),
                                            xp.max(xp.abs(dv_face)))),
    )


__all__ = ["VectorBoundary", "VectorResult", "damp_c_grid_divergence",
           "damp_divergence", "destagger_c_grid", "helmholtz_coefficients",
           "helmholtz_decompose", "lift_mass_increment_to_c_grid",
           "spectral_divergence"]
