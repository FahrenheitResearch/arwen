"""WRF seed labels bound to the engine's keyed process streams.

These labels preserve a namelist's controls. They do not reproduce WRF's
compiler-dependent, date-dependent RANDOM_NUMBER realization.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping

WRF_PROCESS_SEED_DERIVATION = "gpuwm-wrf-process-seed-labels.v1"
WRF_SEED_DEFAULTS = {
    "nens": 1, "iseed_sppt": 53, "iseed_skebs": 811,
    "iseed_spp_conv": 171, "iseed_spp_pbl": 217,
    "iseed_spp_lsm": 317, "iseed_rand_pert": 17,
}
_PROCESS_LABELS = {
    "sppt": "iseed_sppt", "skebs_psi": "iseed_skebs",
    "skebs_theta": "iseed_skebs", "spp_conv": "iseed_spp_conv",
    "spp_pbl": "iseed_spp_pbl", "spp_lsm": "iseed_spp_lsm",
}


def normalize_wrf_seed_labels(labels):
    """Retain signed WRF INTEGER labels; omitted labels keep Registry values."""
    if labels is None:
        return None
    if not isinstance(labels, Mapping) or set(labels) - set(WRF_SEED_DEFAULTS):
        raise ValueError("wrf_seed_labels must name nens and registered iseed_* labels")
    result = dict(WRF_SEED_DEFAULTS)
    result.update(labels)
    if any(type(value) is not int or not -(1 << 31) <= value < (1 << 31)
           for value in result.values()):
        raise ValueError("WRF seed labels must be signed 32-bit integers")
    return result


def process_seed(member_seed, kind, labels=None):
    """Keep original member identity and derive only an explicitly labelled key."""
    if labels is None:
        return member_seed
    labels = normalize_wrf_seed_labels(labels)
    if type(member_seed) is not int or not 0 <= member_seed < (1 << 64):
        raise ValueError("member_seed must be an unsigned 64-bit integer")
    if kind not in _PROCESS_LABELS:
        raise ValueError("WRF seed labels require a registered stochastic process")
    payload = (f"{WRF_PROCESS_SEED_DERIVATION}:{member_seed}:{kind}:"
               f"{labels['nens']}:{labels[_PROCESS_LABELS[kind]]}").encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def seed_label_receipt(labels):
    """The complete label authority, separate from the original member recipe."""
    if labels is None:
        return None
    return {"labels": normalize_wrf_seed_labels(labels),
            "derivation": WRF_PROCESS_SEED_DERIVATION,
            "rng_policy": "engine keyed Philox; not WRF date-dependent RNG realization"}
