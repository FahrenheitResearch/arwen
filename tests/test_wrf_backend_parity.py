"""CPU-only contracts for semantic native-WRF backend parity."""

from __future__ import annotations

import numpy as np
import pytest

from gpuwm.wrf_backend_parity import (
    _array_record,
    _boundary_native,
    _strip,
    _u_face_mass,
    _uncouple_endpoint,
    _v_face_mass,
    EXACT_RULE,
    WIND_RULE,
)


@pytest.mark.parametrize("logical", ("u", "v", "theta", "phi", "mu", "qv"))
@pytest.mark.parametrize("side", ("west", "east", "south", "north"))
def test_boundary_native_inverts_export_layout(logical, side):
    nz, ny, nx, width = 3, 7, 9, 2
    if logical == "mu":
        native = (np.arange(ny * width).reshape(1, ny, width)
                  if side in {"west", "east"}
                  else np.arange(width * nx).reshape(1, width, nx))
        wrf = native[0].T if side in {"west", "east"} else native[0]
    else:
        z = nz + int(logical == "phi")
        native = (np.arange(z * ny * width).reshape(z, ny, width)
                  if side in {"west", "east"}
                  else np.arange(z * width * nx).reshape(z, width, nx))
        wrf = (np.transpose(native, (2, 0, 1))
               if side in {"west", "east"}
               else np.transpose(native, (1, 0, 2)))
    np.testing.assert_array_equal(_boundary_native(wrf, logical, side), native)


def _producer_coupled(logical, side, primitive, mup, *, mub, c1h, c2h,
                      c1f, c2f, mapfac_u, mapfac_v):
    width = mup.shape[-1] if side in {"west", "east"} else mup.shape[-2]
    mass = np.asarray(_strip(mub, side, width) + mup[0], dtype=np.float32)
    if logical == "mu":
        return mup
    if logical == "u":
        weight = np.asarray(c1h[:, None, None] * _u_face_mass(mass, side)[None]
                            + c2h[:, None, None], dtype=np.float32)
        return np.asarray(
            weight * primitive / _strip(mapfac_u, side, width)[None],
            dtype=np.float32)
    if logical == "v":
        weight = np.asarray(c1h[:, None, None] * _v_face_mass(mass, side)[None]
                            + c2h[:, None, None], dtype=np.float32)
        return np.asarray(
            weight * primitive / _strip(mapfac_v, side, width)[None],
            dtype=np.float32)
    coefficients = (c1f, c2f) if logical == "phi" else (c1h, c2h)
    weight = np.asarray(coefficients[0][:, None, None] * mass[None]
                        + coefficients[1][:, None, None], dtype=np.float32)
    return np.asarray(weight * primitive, dtype=np.float32)


@pytest.mark.parametrize("logical", ("u", "v", "theta", "phi", "qv", "mu"))
@pytest.mark.parametrize("side", ("west", "east", "south", "north"))
def test_uncouple_endpoint_recovers_primitive_on_every_stagger(logical, side):
    rng = np.random.default_rng(20260720)
    nz, ny, nx, width = 4, 8, 10, 3
    mub = rng.uniform(72_000.0, 91_000.0, (ny, nx)).astype(np.float32)
    mup_full = rng.uniform(-900.0, 1300.0, (ny, nx)).astype(np.float32)
    mup = _strip(mup_full, side, width)[None]
    c1h = np.linspace(0.98, 0.12, nz, dtype=np.float32)
    c2h = np.linspace(50.0, 4500.0, nz, dtype=np.float32)
    c1f = np.linspace(1.0, 0.0, nz + 1, dtype=np.float32)
    c2f = np.linspace(0.0, 5000.0, nz + 1, dtype=np.float32)
    mapfac_u = rng.uniform(0.92, 1.12, (ny, nx + 1)).astype(np.float32)
    mapfac_v = rng.uniform(0.92, 1.12, (ny + 1, nx)).astype(np.float32)
    horizontal = {
        "u": _strip(mapfac_u, side, width).shape,
        "v": _strip(mapfac_v, side, width).shape,
        "theta": mup.shape[1:], "phi": mup.shape[1:],
        "qv": mup.shape[1:], "mu": mup.shape[1:],
    }[logical]
    levels = nz + int(logical == "phi")
    if logical == "mu":
        coupled = mup
        expected = _strip(mub, side, width) + mup[0]
    else:
        scale = 1.0e-3 if logical == "qv" else 10.0
        primitive = rng.normal(0.0, scale, (levels, *horizontal)).astype(np.float32)
        coupled = _producer_coupled(
            logical, side, primitive, mup, mub=mub, c1h=c1h, c2h=c2h,
            c1f=c1f, c2f=c2f, mapfac_u=mapfac_u, mapfac_v=mapfac_v)
        expected = primitive
    actual = _uncouple_endpoint(
        logical, side, coupled, mup, mub=mub, c1h=c1h, c2h=c2h,
        c1f=c1f, c2f=c2f, mapfac_u=mapfac_u, mapfac_v=mapfac_v)
    np.testing.assert_allclose(actual, expected, rtol=3.0e-7, atol=2.0e-6)


def test_array_record_rejects_nonfinite_and_exact_signed_zero_change():
    nonfinite = _array_record(
        np.asarray([1.0, np.nan]), np.asarray([1.0, np.nan]), WIND_RULE)
    assert nonfinite["status"] == "FAIL"
    assert nonfinite["reason"] == "non_finite"
    signed_zero = _array_record(
        np.asarray([0.0], dtype=np.float32),
        np.asarray([-0.0], dtype=np.float32), EXACT_RULE)
    assert signed_zero["status"] == "FAIL"


def test_array_record_uses_elementwise_bound_not_common_finite_mask():
    reference = np.asarray([1.0, 2.0], dtype=np.float32)
    candidate = np.asarray([1.0, 2.02], dtype=np.float32)
    result = _array_record(reference, candidate, WIND_RULE)
    assert result["status"] == "FAIL"
    assert result["violations"] == 1
