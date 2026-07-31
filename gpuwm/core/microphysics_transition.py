"""Explicit one-way nest microphysics transition contracts.

Stock WRF v4.6.1 accepts an ``mp_physics(max_domains)`` namelist vector but
``share/module_check_a_mundo.F:774-790`` replaces every entry with the
innermost-domain selector.  WRF therefore has no executable mixed-scheme nest
edge.  The matrix in this module is an ArWen extension: shared mass species
are mapped, absent target mass species use their scheme cold-start default,
and target number/volume moments are diagnosed from target mass.

The previously ratified Thompson MP8 -> NSSL-2 MP18 path retains its original
policy id and CUDA entry point.  All other mixed edges use one matrix policy.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np


SAME_SCHEME_POLICY = "same-scheme-only"
MP8_TO_MP18_POLICY = "mp8-to-mp18-mass-diagnosed-v1"
EDGE_MATRIX_POLICY = "mp-edge-mass-diagnosed-v1"
TRANSITION_ORDER = "diagnose-parent-then-spatially-interpolate"
NSSL2_BACKGROUND_CCN_PER_KG = 408163264.0
PORTED_MP_PHYSICS = (1, 6, 8, 10, 18)

_DYNAMIC_FIELDS = ("u", "v", "w", "t", "ph", "mu")
_MASS_FIELDS = {
    1: ("qv", "qc", "qr"),
    6: ("qv", "qc", "qr", "qi", "qs", "qg"),
    8: ("qv", "qc", "qr", "qi", "qs", "qg"),
    10: ("qv", "qc", "qr", "qi", "qs", "qg"),
    18: ("qv", "qc", "qr", "qi", "qs", "qg", "qh"),
}
_MOMENT_FIELDS = {
    1: (),
    6: (),
    8: ("nr", "ni"),
    10: ("nr", "ni", "ns", "ng"),
    18: (
        "qndrop", "qnr", "qni", "qns", "qng", "qnh", "qnn",
        "qvolg", "qvolh",
    ),
}
_TARGET_FIELDS = {
    mp: _MASS_FIELDS[mp] + _MOMENT_FIELDS[mp]
    for mp in PORTED_MP_PHYSICS
}
_ALL_EDGE_FIELDS = tuple(dict.fromkeys(
    name
    for mp in PORTED_MP_PHYSICS
    for name in _TARGET_FIELDS[mp]
))
_EDGE_FIELD_CODES = {
    name: code for code, name in enumerate(_ALL_EDGE_FIELDS)
}
_SOURCE_MASS_CODES = {
    name: code
    for code, name in enumerate(("qv", "qc", "qr", "qi", "qs", "qg", "qh"))
}

# Kept as the exact original MP8->MP18 field-code table.  The content identity
# test and the ratified CUDA entry point both intentionally bind this object.
_MP18_FIELDS = _TARGET_FIELDS[18]
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


def _rimed_category(cfg) -> str | None:
    """Return the physical meaning of a scheme's single ``qg`` category."""

    mp = int(getattr(cfg, "mp_physics", 0))
    if mp == 8:
        return "graupel"
    if mp == 6:
        return "hail" if int(getattr(cfg, "wsm6_hail_opt", 0)) else "graupel"
    if mp == 10:
        return "hail" if int(getattr(cfg, "morr_rimed_ice", 1)) else "graupel"
    return None


@dataclass(frozen=True)
class MicrophysicsTransitionContract:
    """Resolved directed parent-to-child microphysics policy."""

    source_mp_physics: int
    target_mp_physics: int
    policy_id: str
    mixed: bool
    source_rimed_category: str | None = None
    target_rimed_category: str | None = None
    target_morrison_rimed_density: float = 900.0

    def mass_source(self, target_field: str) -> str | None:
        """Source mass field for one target mass, or ``None`` for default."""

        if target_field not in _MASS_FIELDS[self.target_mp_physics]:
            raise ValueError(
                f"{target_field!r} is not a target mass field for "
                f"MP{self.target_mp_physics}")
        if target_field in ("qv", "qc", "qr"):
            return target_field
        if target_field in ("qi", "qs"):
            return (target_field
                    if target_field in _MASS_FIELDS[self.source_mp_physics]
                    else None)
        if target_field == "qg":
            if self.target_mp_physics == 18:
                if self.source_mp_physics == 18:
                    return "qg"
                return ("qg"
                        if self.source_rimed_category == "graupel" else None)
            if self.source_mp_physics == 18:
                return ("qh" if self.target_rimed_category == "hail"
                        else "qg")
            return ("qg"
                    if self.source_rimed_category
                    == self.target_rimed_category else None)
        if target_field == "qh":
            if self.source_mp_physics == 18:
                return "qh"
            return ("qg" if self.source_rimed_category == "hail" else None)
        raise AssertionError(target_field)

    def species_actions(self) -> tuple[Mapping[str, object], ...]:
        """One explicit receipt row for every target and dropped source."""

        rows: list[Mapping[str, object]] = []
        consumed: set[str] = set()
        for target in _MASS_FIELDS[self.target_mp_physics]:
            source = self.mass_source(target)
            if source is None:
                rows.append({
                    "action": "defaulted",
                    "source_field": None,
                    "target_field": target,
                    "reason": "source_species_absent",
                    "default": "zero_mass_mixing_ratio",
                })
            else:
                consumed.add(source)
                rows.append({
                    "action": "mapped",
                    "source_field": source,
                    "target_field": target,
                    "reason": "shared_physical_mass_species",
                })
        for target in _MOMENT_FIELDS[self.target_mp_physics]:
            rows.append({
                "action": "diagnosed",
                "source_field": None,
                "target_field": target,
                "reason": "target_scheme_mass_moment_closure",
            })
        for source in _MASS_FIELDS[self.source_mp_physics]:
            if source not in consumed:
                rows.append({
                    "action": "dropped",
                    "source_field": source,
                    "target_field": None,
                    "reason": "target_species_absent_or_rimed_category_mismatch",
                })
        for source in _MOMENT_FIELDS[self.source_mp_physics]:
            rows.append({
                "action": "dropped",
                "source_field": source,
                "target_field": None,
                "reason": "scheme_closure_mismatch",
            })
        return tuple(rows)

    def receipt(self) -> Mapping[str, object]:
        if not self.mixed:
            return {
                "policy_id": self.policy_id,
                "source_mp_physics": self.source_mp_physics,
                "target_mp_physics": self.target_mp_physics,
                "mixed": False,
                "stock_wrf_equivalent": True,
            }
        species = self.species_actions()
        counts = Counter(str(row["action"]) for row in species)
        receipt: dict[str, object] = {
            "policy_id": self.policy_id,
            "source_mp_physics": self.source_mp_physics,
            "target_mp_physics": self.target_mp_physics,
            "mixed": True,
            "stock_wrf_equivalent": False,
            "stock_wrf_reason": (
                "WRF v4.6.1 normalizes all domains to the innermost "
                "mp_physics selector"
            ),
            "translation_order": TRANSITION_ORDER,
            "source_rimed_category": self.source_rimed_category,
            "target_rimed_category": self.target_rimed_category,
            "species_actions": [dict(row) for row in species],
            "species_action_counts": {
                action: int(counts.get(action, 0))
                for action in ("mapped", "defaulted", "diagnosed", "dropped")
            },
            "reset_child_local_fields": list(_RESET_FIELDS),
            "implementation": transition_implementation_identity(),
        }
        # Preserve every established receipt field for the ratified pair.
        if ((self.source_mp_physics, self.target_mp_physics)
                == (8, 18)):
            receipt.update({
                "canonicalized_source_mass_fields": list(_SOURCE_MASS_FIELDS),
                "diagnosed_from_mass_fields": list(_DIAGNOSED_FIELDS),
                "zeroed_fields": list(_ZEROED_FIELDS),
                "field_actions": {
                    **{
                        name:
                        "canonicalize_from_source_mass_then_interpolate"
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
                "ignored_source_fields": {
                    name: "ignored_due_to_scheme_closure_mismatch"
                    for name in _IGNORED_SOURCE_FIELDS
                },
                "nssl2_background_ccn_per_kg":
                    NSSL2_BACKGROUND_CCN_PER_KG,
                "canonicalization_thresholds": dict(
                    _CANONICALIZATION_THRESHOLDS),
            })
        return receipt


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
        "policy_ids": [MP8_TO_MP18_POLICY, EDGE_MATRIX_POLICY],
        "translation_order": TRANSITION_ORDER,
        "ratified_field_codes": dict(_FIELD_CODES),
        "matrix_field_codes": dict(_EDGE_FIELD_CODES),
        "mass_fields": {str(k): list(v) for k, v in _MASS_FIELDS.items()},
        "moment_fields": {str(k): list(v) for k, v in _MOMENT_FIELDS.items()},
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
    """Resolve one of the 25 ported ordered edges or fail closed."""

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

    if source not in PORTED_MP_PHYSICS or target not in PORTED_MP_PHYSICS:
        raise ValueError(
            f"unsupported mixed nest microphysics edge MP{source}->MP{target}; "
            f"ported selectors are {PORTED_MP_PHYSICS}")
    required_policy = (
        MP8_TO_MP18_POLICY if (source, target) == (8, 18)
        else EDGE_MATRIX_POLICY
    )
    if policy != required_policy:
        raise ValueError(
            f"MP{source}->MP{target} nest forcing requires explicit "
            f"nest_microphysics_transition={required_policy!r}, got "
            f"{policy!r}")
    missing = []
    for role, cfg in (("parent", parent_cfg), ("child", child_cfg)):
        if not bool(cfg.moist):
            missing.append(f"{role}.moist=true")
        if not bool(cfg.moist_cq):
            missing.append(f"{role}.moist_cq=true")
    if missing:
        raise ValueError(
            f"MP{source}->MP{target} nest transition requires the validated "
            "moist/CQ contract on both domains; missing " + ", ".join(missing))
    target_density = (
        400.0 if target == 10
        and int(getattr(child_cfg, "morr_rimed_ice", 1)) == 0
        else 900.0
    )
    return MicrophysicsTransitionContract(
        source_mp_physics=source, target_mp_physics=target,
        policy_id=policy, mixed=True,
        source_rimed_category=_rimed_category(parent_cfg),
        target_rimed_category=_rimed_category(child_cfg),
        target_morrison_rimed_density=target_density,
    )


def transition_handles_field(
        contract: MicrophysicsTransitionContract, field_name: str) -> bool:
    return bool(
        contract.mixed
        and field_name in _TARGET_FIELDS[contract.target_mp_physics]
    )


def transition_parent_field_shape(state, field_name: str) -> tuple[int, ...]:
    if field_name not in _ALL_EDGE_FIELDS:
        raise ValueError(f"unsupported microphysics edge field {field_name!r}")
    shape = tuple(int(value) for value in state.qv.shape)
    if len(shape) != 3:
        raise ValueError(f"parent moisture field must be 3-D, got {shape}")
    return shape


def _validate_transition_arrays(contract, state, out, shape) -> None:
    import cupy as cp

    for name in _MASS_FIELDS[contract.source_mp_physics]:
        value = getattr(state, name, None)
        if value is None or tuple(value.shape) != shape:
            raise ValueError(
                f"microphysics transition source {name} must have "
                f"shape {shape}")
        if value.dtype != cp.float32:
            raise TypeError(
                f"microphysics transition source {name} must be float32")
        if not value.flags.c_contiguous:
            raise ValueError(
                f"microphysics transition source {name} must be contiguous")
    if tuple(out.shape) != shape or out.dtype != cp.float32:
        raise ValueError(
            f"microphysics transition output must be float32 with shape "
            f"{shape}")
    if not out.flags.c_contiguous:
        raise ValueError("microphysics transition output must be contiguous")
    ny, nx = shape[1:]
    for name, value, expected in (
            ("alt", state.alt, shape),
            ("mub2d", state.mub2d, (ny, nx)),
            ("mup", state.mup, (ny, nx)),
            ("c1h", state.c1h, (shape[0],)),
            ("c2h", state.c2h, (shape[0],))):
        if tuple(value.shape) != expected or value.dtype != cp.float32:
            raise ValueError(
                f"microphysics transition {name} must be float32 with "
                f"shape {expected}")


def launch_microphysics_edge_parent_field(
        contract: MicrophysicsTransitionContract, state, field_name: str,
        *, out, coupled: bool) -> object:
    """Write one target field diagnosed on the parent before SINT."""

    import cupy as cp
    from gpuwm.core.kernels import get_kernel

    if not transition_handles_field(contract, field_name):
        raise ValueError(
            f"MP{contract.source_mp_physics}->MP"
            f"{contract.target_mp_physics} does not handle {field_name!r}")
    if not isinstance(coupled, bool):
        raise TypeError("coupled must be bool")
    if (contract.source_mp_physics, contract.target_mp_physics) == (8, 18):
        return launch_mp8_to_mp18_parent_field(
            state, field_name, out=out, coupled=coupled)

    shape = transition_parent_field_shape(state, field_name)
    _validate_transition_arrays(contract, state, out, shape)
    placeholder = state.qv
    source_arrays = []
    for name in ("qv", "qc", "qr", "qi", "qs", "qg", "qh"):
        value = getattr(state, name, None)
        source_arrays.append(placeholder if value is None else value)
    mass_sources = [
        _SOURCE_MASS_CODES.get(contract.mass_source(name), -1)
        if name in _MASS_FIELDS[contract.target_mp_physics] else -1
        for name in ("qv", "qc", "qr", "qi", "qs", "qg", "qh")
    ]
    count = int(out.size)
    ny, nx = shape[1:]
    get_kernel("nest_microphysics", "microphysics_edge_field")(
        ((count + _THREADS - 1) // _THREADS,), (_THREADS,), (
            state.alt, *source_arrays, state.mub2d, state.mup,
            state.c1h, state.c2h, out,
            *[np.int32(code) for code in mass_sources],
            np.int32(_EDGE_FIELD_CODES[field_name]),
            np.int32(contract.target_mp_physics),
            np.float32(contract.target_morrison_rimed_density),
            np.int32(coupled), np.int32(shape[0]), np.int32(ny),
            np.int32(nx),
        ))
    return out


def launch_mp8_to_mp18_parent_field(
        state, field_name: str, *, out, coupled: bool) -> object:
    """Original bit-specified MP8 -> MP18 translation entry point."""

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
    "EDGE_MATRIX_POLICY", "MP8_TO_MP18_POLICY",
    "MicrophysicsTransitionContract", "NSSL2_BACKGROUND_CCN_PER_KG",
    "PORTED_MP_PHYSICS", "SAME_SCHEME_POLICY", "TRANSITION_ORDER",
    "launch_microphysics_edge_parent_field",
    "launch_mp8_to_mp18_parent_field", "resolve_microphysics_transition",
    "transition_handles_field", "transition_implementation_identity",
    "transition_parent_field_shape",
]
