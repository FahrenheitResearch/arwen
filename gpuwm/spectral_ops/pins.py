"""Immutable arithmetic identity for the Level-2 regional operators."""

from __future__ import annotations

import hashlib
import json

SCHEMA = "gpuwm.regional-spectral-operators/v1"

PINS: dict[str, object] = {
    "schema": SCHEMA,
    "horizontal_basis": {
        "periodic": "real two-dimensional Fourier basis",
        "reflect": (
            "cell-centred even extension to 2*ny by 2*nx followed by a real "
            "two-dimensional Fourier transform; scalar fields only"),
        "tapered": (
            "periodic transform of a mean-removed field multiplied by a "
            "separable raised-cosine edge window; the correction is multiplied "
            "by the same window so every outer-edge cell is unchanged"),
    },
    "wavenumber": "angular radians per metre from fftfreq/rfftfreq and dx,dy",
    "hyperdiffusion": (
        "exact exponential transfer exp[-dt/tau*(k/k_ref)^(2p)*activation], "
        "optionally capped by a maximum per-call damping fraction"),
    "protect_transition": (
        "zero activation at and below k_protect; raised cosine from k_protect "
        "to k_ref; one above k_ref"),
    "mean_policy": "constant Fourier mode is one; cropped/blended corrections are re-centred",
    "vector_split": (
        "orthogonal Fourier Helmholtz projector: longitudinal kk^T/k^2 is "
        "divergent, residual is rotational; k=0 is preserved"),
    "helmholtz": "(alpha - beta*laplacian)x=rhs uses 1/(alpha+beta*k^2)",
    "poisson_zero_mode": "zero-mean right-hand side and zero-mean solution",
    "positive_space": {
        "log": "operate on log(max(x,floor)); exponentiate; optional multiplicative mean repair",
        "linear": "operate directly",
    },
    "precision": (
        "transform arithmetic follows the input/backend FFT dtype; receipt "
        "reductions use host float64; no FP16 policy is admitted"),
    "default_mode": "off",
    "unsafe_periodic_policy": "periodic requires periodic_domain=true",
}


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


PINS_SHA256 = canonical_hash(PINS)


def registration() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "pins": json.loads(json.dumps(PINS)),
        "pins_sha256": PINS_SHA256,
    }


__all__ = ["PINS", "PINS_SHA256", "SCHEMA", "canonical_hash", "registration"]
