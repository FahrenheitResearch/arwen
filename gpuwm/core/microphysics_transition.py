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
#: Selectors with a ported MIXED nest edge.  APPEND ONLY: ``_ALL_EDGE_FIELDS``
#: iterates this tuple in order and ``_EDGE_FIELD_CODES`` is ``enumerate``
#: over the result, so inserting a selector anywhere but the end silently
#: renumbers the stable host field codes ``kernels/nest_microphysics.cu``
#: switches on -- which would re-point the ratified MP8 -> MP18 nest edge at
#: different fields with nothing raising.
#:
#: mp_physics=28 is DELIBERATELY ABSENT.  It is a fully ported scheme, and
#: an mp=28 parent forcing an mp=28 child works: ``resolve_microphysics_
#: transition`` returns the same-scheme contract before this tuple is ever
#: consulted.  What is absent is a *mixed* edge, because none has been
#: validated -- see :data:`UNVALIDATED_MIXED_EDGE_SELECTORS`.  Listing 28
#: here and then refusing all eleven of its mixed pairs would be a
#: self-contradiction: this tuple's whole meaning is "the mixed edges that
#: work".  When a closure is ratified, APPEND 28 here; it will contribute
#: nc/nwfa/nifa at codes 20/21/22 and move nothing.
PORTED_MP_PHYSICS = (1, 6, 8, 10, 18)

#: Schemes ArWen has ported but whose MIXED nest edges are refused rather
#: than approximated.  Consulted by :func:`resolve_microphysics_transition`
#: purely so the refusal names a reason instead of falling through to the
#: generic "ported selectors are ..." message, which would read as "mp=28 is
#: not implemented" when in fact only the edge closure is missing.
#: mp=16 (WDM6) joins for the same reason and needs its own sentence,
#: because its missing closure is a different one: a mixed edge would have
#: to fill or drain the CCN reservoir nn across the boundary, and the
#: reservoir is not a diagnostic of any other scheme's state -- Morrison's
#: nc is a droplet count, not a count of unactivated nuclei.  Same-scheme
#: mp=16 nesting is unaffected: resolve_microphysics_transition returns the
#: same-scheme contract before this tuple is consulted.
UNVALIDATED_MIXED_EDGE_SELECTORS = (16, 28)

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

#: What a mixed edge touching each refused selector WOULD have to move,
#: recorded so the refusal is specific and so a future package does not have
#: to rediscover it.  Every tuple leads with the moments that ALREADY have a
#: stable host field code, so appending the selector to
#: :data:`PORTED_MP_PHYSICS` extends the code table and reorders nothing:
#: mp=28's nr/ni are codes 6/7 and its nc/nwfa/nifa would take 20/21/22;
#: mp=16's nr is code 6 and its nc/nn would take the next two free codes
#: (20/21 on their own, or 20/23 if 28 is ratified first, since the two
#: schemes' ``nc`` is one field name and therefore one code).
UNVALIDATED_MIXED_EDGE_MOMENTS = {
    16: ("nr", "nc", "nn"),
    28: ("nr", "ni", "nc", "nwfa", "nifa"),
}

#: The scheme's own name, and the paragraph that says why ITS closure is
#: missing.  One row per :data:`UNVALIDATED_MIXED_EDGE_SELECTORS` entry: the
#: two refusals are NOT the same refusal, and a WDM6 operator must never be
#: handed Thompson's reason, Thompson's fallback constants or Thompson's
#: Fortran citation.  Each ``reason`` is a sentence fragment that completes
#: "... but it has no validated cross-scheme entry closure for its moments
#: (...) -- ".
_UNVALIDATED_MIXED_EDGE_REASONS = {
    16: (
        "WDM6, double-moment warm rain",
        "a prognostic cloud droplet number, a prognostic rain number and a "
        "CCN reservoir (nn) that no other ported scheme carries in any form. "
        "The reservoir is the hard one: it counts UNACTIVATED nuclei, so it "
        "is not a diagnostic of any other scheme's state -- Morrison's nc "
        "and Thompson-aerosol's nc are activated droplet counts, and NSSL's "
        "qnn is a different scheme's reservoir under the same WRF Registry "
        "name.  WRF's own cold start for the field is a UNIFORM fill, "
        "scalar(:,:,:,p_qnn) = ccn_conc, and only when the whole array is "
        "still below 1.0 (dyn_em/start_em.F:1750-1774; note the WDM5/WDM6 "
        "arm of that IF is explicitly a NO OP, so the value used is the "
        "Registry default ccn_conc=1.0E8 m-3, Registry.EM_COMMON:2664).  "
        "Seeding a child that way is the obvious candidate and it is the "
        "wrong forecast: it would erase the parent's DEPLETED reservoir "
        "under active convection and hand the nest a fresh 1.0E8 m-3 "
        "everywhere, re-arming activation exactly where the parent had "
        "consumed its aerosol.  Nothing would flag it -- WDM6's own clamp "
        "min(max(nn,1.e8),2.e10) (module_mp_wdm6.F:584) admits that value "
        "as its floor, so the seeded field is inside every bound this tree "
        "checks and the trajectory is still different"
    ),
    28: (
        "Thompson aerosol-aware",
        "a prognostic cloud droplet number plus two aerosol number tracers "
        "that no other ported scheme carries.  WRF's own non-aerosol-aware "
        "fallbacks -- nc = Nt_c/rho, nwfa = 11.1E6/rho, "
        "nifa = naIN1*0.01/rho == 5.0E3/rho "
        "(module_mp_thompson.F:1248-1255, the ELSE branch mp_gt_driver "
        "takes when is_aerosol_aware is FALSE) -- are the obvious "
        "candidate, but nothing has measured them across an ArWen nest "
        "edge, and they would seed the child with the scheme's own floor "
        "values instead of the parent's aerosol field.  That is a "
        "different forecast, and it is one no bound, health rule or "
        "conservation check in this tree would flag"
    ),
}

# A selector may not join UNVALIDATED_MIXED_EDGE_SELECTORS without bringing
# its own moments and its own sentence.  Adding 16 to the selector tuple and
# not to the moments table turned the named refusal into a bare KeyError(16),
# with every advertised gate still claiming the refusal worked; this check
# turns that into an import-time failure that no code path can reach past.
_missing_edge_rows = sorted(
    mp for mp in UNVALIDATED_MIXED_EDGE_SELECTORS
    if mp not in UNVALIDATED_MIXED_EDGE_MOMENTS
    or mp not in _UNVALIDATED_MIXED_EDGE_REASONS
)
if _missing_edge_rows:
    raise RuntimeError(
        "UNVALIDATED_MIXED_EDGE_SELECTORS entries without a moments row and "
        f"a scheme-specific reason: {_missing_edge_rows}; the refusal in "
        "resolve_microphysics_transition would raise KeyError instead of "
        "naming the scheme")
_TARGET_FIELDS = {
    mp: _MASS_FIELDS[mp] + _MOMENT_FIELDS[mp]
    for mp in PORTED_MP_PHYSICS
}
_ALL_EDGE_FIELDS = tuple(dict.fromkeys(
    name
    for mp in PORTED_MP_PHYSICS
    for name in _TARGET_FIELDS[mp]
))
#: The stable host field codes ``kernels/nest_microphysics.cu`` switches on:
#:
#:   qv/qc/qr/qi/qs/qg = 0..5, nr/ni/ns/ng = 6..9, qh = 10,
#:   qndrop/qnr/qni/qns/qng/qnh/qnn/qvolg/qvolh = 11..19,
#:   nc = 20, nwfa = 21, nifa = 22   (mp_physics=28).
#:
#: Codes 20..22 are RESERVED for mp_physics=28's nc/nwfa/nifa and are not
#: allocated today: 28 is absent from :data:`PORTED_MP_PHYSICS` because every
#: mixed edge touching it is refused, so ``microphysics_edge_field`` can
#: never be launched with ``target_mp == 28`` nor with ``field >= 20``.  The
#: kernel's own comment block therefore documents and decodes 0..19 only, and
#: that is correct rather than stale.  Ratifying the closure means appending
#: 28 to PORTED_MP_PHYSICS, which extends this table rather than renumbering
#: it, and adding a kernel arm.
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
    if mp in (8, 28):
        # Thompson's single rimed category is graupel in both entries:
        # module_mp_thompson.F declares one rho_g/am_g/bm_g set used by the
        # classic and the aerosol-aware path alike, and mp=28 adds no hail
        # category.
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
    """Resolve one of the 25 ported ordered edges or fail closed.

    A same-scheme edge resolves for ANY ported ``mp_physics``, including
    mp=16 and mp=28, before the mixed-edge matrix is consulted.  A MIXED
    edge with mp=16 or mp=28 on either side is refused by name, each with
    its OWN missing closure (see
    :data:`UNVALIDATED_MIXED_EDGE_SELECTORS` and
    :data:`_UNVALIDATED_MIXED_EDGE_REASONS`).
    """

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

    # NAMED refusal before the generic "not a ported selector" message.
    # mp=16 and mp=28 ARE ported; only their cross-scheme entry closures are
    # missing, and an operator who reads "ported selectors are
    # (1, 6, 8, 10, 18)" would reasonably conclude the scheme itself is
    # unavailable.  The message body is looked up per scheme, never shared:
    # the two closures are missing for different reasons.
    unvalidated = sorted(
        {source, target} & set(UNVALIDATED_MIXED_EDGE_SELECTORS))
    if unvalidated:
        mp = unvalidated[0]
        moments = ", ".join(UNVALIDATED_MIXED_EDGE_MOMENTS[mp])
        scheme, reason = _UNVALIDATED_MIXED_EDGE_REASONS[mp]
        raise ValueError(
            f"mixed nest microphysics edge MP{source}->MP{target} is REFUSED: "
            f"MP{mp} ({scheme}) is ported and runs, but it has "
            f"no validated cross-scheme entry closure for its moments "
            f"({moments}) -- {reason}.  An honest refusal "
            f"beats an unvalidated closure: configure both domains with "
            f"mp_physics={mp}, or keep MP{mp} on a single domain.")

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
    "UNVALIDATED_MIXED_EDGE_MOMENTS", "UNVALIDATED_MIXED_EDGE_SELECTORS",
    "launch_microphysics_edge_parent_field",
    "launch_mp8_to_mp18_parent_field", "resolve_microphysics_transition",
    "transition_handles_field", "transition_implementation_identity",
    "transition_parent_field_shape",
]
