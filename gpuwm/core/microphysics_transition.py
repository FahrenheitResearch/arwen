"""Explicit one-way nest microphysics transition contracts.

Stock WRF v4.6.1 normalizes every domain to the innermost microphysics
selector.  GPUWM therefore treats a mixed edge as a versioned extension, not
as WRF equivalence.  Only the community-requested Thompson MP8 -> NSSL-2 MP18
edge is admitted here; every other mixed pair fails before state allocation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np


SAME_SCHEME_POLICY = "same-scheme-only"
MP8_TO_MP18_POLICY = "mp8-to-mp18-mass-diagnosed-v1"
TRANSITION_ORDER = "diagnose-parent-then-spatially-interpolate"
NSSL2_BACKGROUND_CCN_PER_KG = 408163264.0

_DYNAMIC_FIELDS = ("u", "v", "w", "t", "ph", "mu")
_MP18_FIELDS = (
    "qv", "qc", "qr", "qi", "qs", "qg", "qh", "qndrop", "qnr",
    "qni", "qns", "qng", "qnh", "qnn", "qvolg", "qvolh",
)
_FIELD_CODES = {name: code for code, name in enumerate(_MP18_FIELDS)}
_SOURCE_MASS_FIELDS = ("qv", "qc", "qr", "qi", "qs", "qg")
_DIAGNOSED_FIELDS = ("qndrop", "qnr", "qni", "qns", "qng", "qnn", "qvolg")
_ZEROED_FIELDS = ("qh", "qnh", "qvolh")
_RESET_FIELDS = (
    "h_diabatic", "held_tendencies", "precipitation_accumulators",
)
_IGNORED_SOURCE_FIELDS = ("nr", "ni")
_CANONICALIZATION_THRESHOLDS = {
    "cxmin_number_per_m3": 1.0e-8,
    "initial_mass_kg_per_kg": 1.0e-8,
    "cloud_ice_snow_mass_kg_per_kg": 1.0e-13,
    "rain_graupel_mass_kg_per_kg": 1.0e-12,
    "subthreshold_mass_action": "return_to_vapor",
}
_THREADS = 256


@dataclass(frozen=True)
class MicrophysicsTransitionContract:
    """Resolved directed parent-to-child microphysics policy."""

    source_mp_physics: int
    target_mp_physics: int
    policy_id: str
    mixed: bool

    def receipt(self) -> Mapping[str, object]:
        if not self.mixed:
            return {
                "policy_id": self.policy_id,
                "source_mp_physics": self.source_mp_physics,
                "target_mp_physics": self.target_mp_physics,
                "mixed": False,
                "stock_wrf_equivalent": True,
            }
        return {
            "policy_id": self.policy_id,
            "source_mp_physics": self.source_mp_physics,
            "target_mp_physics": self.target_mp_physics,
            "mixed": True,
            "stock_wrf_equivalent": False,
            "stock_wrf_reason": (
                "WRF v4.6.1 requires one mp_physics selector across domains"
            ),
            "translation_order": TRANSITION_ORDER,
            "canonicalized_source_mass_fields": list(_SOURCE_MASS_FIELDS),
            "diagnosed_from_mass_fields": list(_DIAGNOSED_FIELDS),
            "zeroed_fields": list(_ZEROED_FIELDS),
            "field_actions": {
                **{
                    name: "canonicalize_from_source_mass_then_interpolate"
                    for name in _SOURCE_MASS_FIELDS
                },
                **{
                    name: "diagnose_from_source_mass_then_interpolate"
                    for name in _DIAGNOSED_FIELDS
                },
                **{
                    name: "zero_then_interpolate"
                    for name in _ZEROED_FIELDS
                },
            },
            "reset_child_local_fields": list(_RESET_FIELDS),
            "ignored_source_fields": {
                name: "ignored_due_to_scheme_closure_mismatch"
                for name in _IGNORED_SOURCE_FIELDS
            },
            "nssl2_background_ccn_per_kg": NSSL2_BACKGROUND_CCN_PER_KG,
            "canonicalization_thresholds": dict(
                _CANONICALIZATION_THRESHOLDS),
            "implementation": transition_implementation_identity(),
        }


def transition_implementation_identity() -> Mapping[str, str]:
    """Content identity for the executable conversion policy."""

    kernel = Path(__file__).with_name("kernels") / "nest_microphysics.cu"
    kernel_sha = _canonical_source_sha256(kernel)
    driver_sha = _canonical_source_sha256(Path(__file__))
    live_coupler_sha = _canonical_source_sha256(
        Path(__file__).with_name("nest.py"))
    parent_init_sha = _canonical_source_sha256(
        Path(__file__).parents[1] / "ingest" / "nest_init.py")
    metadata = {
        "policy_id": MP8_TO_MP18_POLICY,
        "translation_order": TRANSITION_ORDER,
        "field_codes": dict(_FIELD_CODES),
        "source_mass_fields": list(_SOURCE_MASS_FIELDS),
        "diagnosed_fields": list(_DIAGNOSED_FIELDS),
        "zeroed_fields": list(_ZEROED_FIELDS),
        "reset_fields": list(_RESET_FIELDS),
        "ignored_source_fields": list(_IGNORED_SOURCE_FIELDS),
        "background_ccn_per_kg": NSSL2_BACKGROUND_CCN_PER_KG,
        "canonicalization_thresholds": dict(_CANONICALIZATION_THRESHOLDS),
    }
    payload = json.dumps(
        {
            "driver_sha256": driver_sha,
            "kernel_sha256": kernel_sha,
            "live_coupler_sha256": live_coupler_sha,
            "parent_init_sha256": parent_init_sha,
            "metadata": metadata,
        },
        sort_keys=True, separators=(",", ":"),
    ).encode("ascii")
    return {
        "driver_sha256": driver_sha,
        "kernel_sha256": kernel_sha,
        "live_coupler_sha256": live_coupler_sha,
        "parent_init_sha256": parent_init_sha,
        "contract_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _canonical_source_sha256(path: Path) -> str:
    """Hash source text independently of checkout newline policy."""

    canonical = path.read_text(encoding="utf-8").replace(
        "\r\n", "\n").replace("\r", "\n").encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def resolve_microphysics_transition(
        parent_cfg, child_cfg) -> MicrophysicsTransitionContract:
    """Resolve one edge or reject it with a complete fail-closed message."""

    # Lightweight CPU/setup views used by the idealized initialization path
    # predate the RunConfig physics fields; absence means the legacy dry MP0
    # same-scheme edge, never an implicit mixed transition.
    source = int(getattr(parent_cfg, "mp_physics", 0))
    target = int(getattr(child_cfg, "mp_physics", 0))
    policy = str(getattr(
        child_cfg, "nest_microphysics_transition", SAME_SCHEME_POLICY))
    if source == target:
        if policy != SAME_SCHEME_POLICY:
            raise ValueError(
                f"nest microphysics edge MP{source}->MP{target} is already "
                f"same-scheme and must use {SAME_SCHEME_POLICY!r}, got "
                f"{policy!r}")
        return MicrophysicsTransitionContract(
            source_mp_physics=source, target_mp_physics=target,
            policy_id=policy, mixed=False)

    if (source, target) != (8, 18):
        raise ValueError(
            f"unsupported mixed nest microphysics edge MP{source}->MP{target}; "
            f"only MP8->MP18 with policy {MP8_TO_MP18_POLICY!r} is admitted")
    if policy != MP8_TO_MP18_POLICY:
        raise ValueError(
            "MP8->MP18 nest forcing requires explicit "
            f"nest_microphysics_transition={MP8_TO_MP18_POLICY!r}, got "
            f"{policy!r}")
    missing = []
    for role, cfg in (("parent", parent_cfg), ("child", child_cfg)):
        if not bool(cfg.moist):
            missing.append(f"{role}.moist=true")
        if not bool(cfg.moist_cq):
            missing.append(f"{role}.moist_cq=true")
    if missing:
        raise ValueError(
            "MP8->MP18 nest transition requires the validated moist/CQ "
            "contract on both domains; missing " + ", ".join(missing))
    return MicrophysicsTransitionContract(
        source_mp_physics=source, target_mp_physics=target,
        policy_id=policy, mixed=True)


def transition_handles_field(
        contract: MicrophysicsTransitionContract, field_name: str) -> bool:
    return bool(contract.mixed and field_name in _FIELD_CODES)


def transition_parent_field_shape(state, field_name: str) -> tuple[int, ...]:
    if field_name not in _FIELD_CODES:
        raise ValueError(f"unsupported MP8->MP18 transition field {field_name!r}")
    shape = tuple(int(value) for value in state.qv.shape)
    if len(shape) != 3:
        raise ValueError(f"parent moisture field must be 3-D, got {shape}")
    return shape


def launch_mp8_to_mp18_parent_field(
        state, field_name: str, *, out, coupled: bool) -> object:
    """Write one canonical MP18 endpoint derived only from MP8 mass fields.

    The kernel deliberately has no Thompson number-moment arguments.  Thus two
    MP8 states with identical mass but different ``nr``/``ni`` produce bitwise
    identical MP18 endpoints by construction.
    """

    import cupy as cp
    from gpuwm.core.kernels import get_kernel

    if field_name not in _FIELD_CODES:
        raise ValueError(f"unsupported MP8->MP18 transition field {field_name!r}")
    if not isinstance(coupled, bool):
        raise TypeError("coupled must be bool")
    shape = transition_parent_field_shape(state, field_name)
    arrays = {
        "alt": state.alt, "qv": state.qv, "qc": state.qc,
        "qr": state.qr, "qi": state.qi, "qs": state.qs, "qg": state.qg,
    }
    for name, value in arrays.items():
        if value is None or tuple(value.shape) != shape:
            raise ValueError(
                f"MP8 transition source {name} must have shape {shape}")
        if value.dtype != cp.float32:
            raise TypeError(f"MP8 transition source {name} must be float32")
        if not value.flags.c_contiguous:
            raise ValueError(f"MP8 transition source {name} must be contiguous")
    if tuple(out.shape) != shape or out.dtype != cp.float32:
        raise ValueError(
            f"MP8 transition output must be float32 with shape {shape}")
    if not out.flags.c_contiguous:
        raise ValueError("MP8 transition output must be contiguous")
    ny, nx = shape[1:]
    for name, value, expected in (
            ("mub2d", state.mub2d, (ny, nx)),
            ("mup", state.mup, (ny, nx)),
            ("c1h", state.c1h, (shape[0],)),
            ("c2h", state.c2h, (shape[0],))):
        if tuple(value.shape) != expected or value.dtype != cp.float32:
            raise ValueError(
                f"MP8 transition {name} must be float32 with shape {expected}")

    count = int(out.size)
    get_kernel("nest_microphysics", "mp8_to_mp18_mass_diagnosed_field")(
        ((count + _THREADS - 1) // _THREADS,), (_THREADS,), (
            state.alt, state.qv, state.qc, state.qr, state.qi, state.qs,
            state.qg, state.mub2d, state.mup, state.c1h, state.c2h, out,
            np.int32(_FIELD_CODES[field_name]), np.int32(coupled),
            np.int32(shape[0]), np.int32(ny), np.int32(nx),
        ))
    return out


__all__ = [
    "MP8_TO_MP18_POLICY", "MicrophysicsTransitionContract",
    "NSSL2_BACKGROUND_CCN_PER_KG", "SAME_SCHEME_POLICY",
    "TRANSITION_ORDER", "launch_mp8_to_mp18_parent_field",
    "resolve_microphysics_transition", "transition_handles_field",
    "transition_implementation_identity", "transition_parent_field_shape",
]
