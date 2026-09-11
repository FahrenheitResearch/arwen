"""Strict contracts for native offline parent-to-child downscaling.

This module is the file-facing half of gpuwm's CUDA-native ``ndown``
replacement.  It deliberately separates cheap metadata validation from the
later GPU interpolation/build transaction: an archived parent series must be
proved complete, geometrically identical, regularly ordered, and sufficiently
frequent before any child state is allocated.

Both gpuwm and stock-WRF history files are accepted.  The reader is closed
world for trajectory fields but tolerant of additional diagnostic variables.
Physics conversion is explicit.  In particular, active condensate may never
be paired with a fabricated zero number moment merely because the parent used
a different microphysics scheme.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Sequence
import time

import netCDF4

from gpuwm import netcdf_bridge
import numpy as np

from gpuwm.core import microphysics_transition as _mt
from gpuwm.core.grid import (BaseState, compute_hybrid_coeffs,
                             finalize_vertical_coord, make_vertical_coord)
from gpuwm.core.nest_interp import register_nest, sint
from gpuwm.core.state import mu_at_u_faces, mu_at_v_faces
from gpuwm.core import constants as c
from gpuwm.vertical_remap import (
    dry_mass_edges,
    geopotential_thickness_per_mass,
    rebuild_geopotential,
    remap_interface_values,
    remap_layer_means,
    remap_receipt,
)
from gpuwm.ingest.lateral_bc import (
    LateralBoundaries,
    build_lateral_interval_from_sides,
    extract_lateral_side,
)
# The SINT positive-definite fix-up, imported rather than re-spelled: the
# online nest lane owns the tolerance policy and the moment membership, and
# a second copy here would be free to drift from it.  Membership matters
# per scheme: P3's number moments ni/nr ARE members while its rime pair
# qir/qib -- small-magnitude mixing ratios -- is DELIBERATELY not
# (gpuwm/ingest/nest_init.py::POSITIVE_DEFINITE_MOMENTS and the comment
# above it), and both lanes get that answer from the one tuple.  The RK
# time-t seeding table is shared for the same reason: nest_init's pair
# list is the census the online lane's own gates iterate, and a re-spelled
# copy here is how NSSL's moment seeds and P3's qir0/qib0 would go
# missing on exactly one of the two birth paths.
from gpuwm.ingest.nest_init import (clamp_sint_undershoot_mapping,
                                    seed_rk_time_t_copies)


_TIME_FORMAT = "%Y-%m-%d_%H:%M:%S"
_REQUIRED_DYNAMICS = frozenset({
    "T", "U", "V", "W", "PH", "PHB", "MU", "MUB", "HGT",
    "P", "PB", "P_TOP", "ZNU", "ZNW", "QVAPOR",
})
_GEOMETRY_DIMS = (
    "west_east", "south_north", "bottom_top", "west_east_stag",
    "south_north_stag", "bottom_top_stag",
)
_GEOMETRY_ATTRS = (
    "DX", "DY", "MAP_PROJ", "TRUELAT1", "TRUELAT2", "STAND_LON",
    "MOAD_CEN_LAT", "CEN_LAT", "CEN_LON", "POLE_LAT", "POLE_LON",
    "HYBRID_OPT", "ETAC",
)
_STATIC_GEOMETRY_FIELDS = ("HGT", "XLAT", "XLONG", "MAPFAC_M", "ZNU", "ZNW")

_MASS_FIELDS = ("qv", "qc", "qr", "qi", "qs", "qg")
_NSSL_FIELDS = (
    "qv", "qc", "qr", "qi", "qs", "qg", "qh", "qndrop", "qnr",
    "qni", "qns", "qng", "qnh", "qnn", "qvolg", "qvolh",
)
_WRF_TO_STATE = MappingProxyType({
    "QVAPOR": "qv", "QCLOUD": "qc", "QRAIN": "qr",
    "QICE": "qi", "QSNOW": "qs", "QGRAUP": "qg",
    "QNCLOUD": "nc", "QNRAIN": "nr", "QNICE": "ni",
    "QNSNOW": "ns", "QNGRAUPEL": "ng",
    # mp_physics=28 (Thompson aerosol-aware).  Transported scalars in WRF's
    # own Registry (Registry.EM_COMMON:3036,
    # ``scalar:qni,qnr,qnc,qnwfa,qnifa,qnbca``); QNCLOUD is already above
    # because Morrison declares the same name.  Only the names
    # ``_transported_source_fields`` asks for are ever read, so adding rows
    # here cannot change what any other scheme reads.
    "QNWFA": "nwfa", "QNIFA": "nifa",
    # mp_physics=50 (P3, one-category ice with prognostic riming).  The
    # rime MASS / rime VOLUME pair rides in the same 4-D ``scalar`` array
    # as the two number moments beside it (Registry.EM_COMMON:555/:557,
    # package ``moist:qv,qc,qr,qi;scalar:qni,qnr,qir,qib`` at :3038); its
    # qi/ni/nr rows are already above under the names Morrison and
    # Thompson declared first.  No radius rows: this map carries no
    # re_cloud/re_ice for ANY scheme, because effc/effi are per-call
    # diagnostics every scheme (P3 included, module_mp_p3.F:2280-2282)
    # rebuilds before reading, never inherited state.
    "QIR": "qir", "QIB": "qib",
})

#: mp_physics=28 surface aerosol emission tendencies (# kg-1 s-1).  NOT
#: transported scalars: they are per-domain cross-step CONSTANTS, and the
#: offline child MUST inherit them from the parent rather than re-derive
#: them.  WRF's ``thompson_init`` fills ``nwfa2d`` at
#: module_mp_thompson.F:510 ONLY inside the "no initial CCN" branch
#: (:493); a child that inherits a parent's nonzero ``nwfa`` takes the
#: ``has_CCN = .TRUE.`` branch at :516-522 instead, which fills nothing.
#: So an offline mp=28 child built without these would run its entire
#: forecast with zero surface aerosol emission and nothing would raise.
#: This is the same argument, and the same resolution, that
#: ``gpuwm/ingest/nest_init.py`` already applies to the ONLINE nest lane,
#: which SINTs both fields on the mass stagger.
_AEROSOL_SURFACE_EMISSION_WRF = ("QNWFA2D", "QNIFA2D")
_AEROSOL_SURFACE_EMISSION_STATE = MappingProxyType({
    "QNWFA2D": "nwfa2d", "QNIFA2D": "nifa2d",
})
_NSSL_WRF_TO_STATE = MappingProxyType({
    "QVAPOR": "qv", "QCLOUD": "qc", "QRAIN": "qr",
    "QICE": "qi", "QSNOW": "qs", "QGRAUP": "qg", "QHAIL": "qh",
    "QNDROP": "qndrop", "QNRAIN": "qnr", "QNICE": "qni",
    "QNSNOW": "qns", "QNGRAUPEL": "qng", "QNHAIL": "qnh",
    "QNCCN": "qnn", "QVGRAUPEL": "qvolg", "QVHAIL": "qvolh",
})
#: mp_physics=9 (Milbrandt-Yau).  A THIRD scheme-qualified map, for the
#: same reason the NSSL one above is a second: QHAIL and QNHAIL are
#: declared by both milbrandt2mom (Registry.EM_COMMON:3025,
#: ``scalar:qh,qnc,qnr,qni,qns,qng,qnh``) and nssl_2mom, and the two
#: schemes bind them to DIFFERENT state fields -- MY2's hail number is
#: ``nh``, NSSL's is ``qnh`` -- so one shared map could only be wrong for
#: one of them.  gpuwm/ingest/wrfinput.py already carries exactly these
#: rows for the same reason; this is the offline lane learning what the
#: root door knew (audit R-017).  The six number moments and the five
#: shared masses reuse the names Morrison and Thompson declared first.
_MY2_WRF_TO_STATE = MappingProxyType({
    "QVAPOR": "qv", "QCLOUD": "qc", "QRAIN": "qr",
    "QICE": "qi", "QSNOW": "qs", "QGRAUP": "qg", "QHAIL": "qh",
    "QNCLOUD": "nc", "QNRAIN": "nr", "QNICE": "ni",
    "QNSNOW": "ns", "QNGRAUPEL": "ng", "QNHAIL": "nh",
})



def _scheme_wrf_to_state(source_mp_physics: int) -> Mapping[str, str]:
    """The wrfout-name -> state-field map for one parent scheme.

    Three schemes need their own: QHAIL/QNHAIL/QNCCN are declared by more
    than one WRF package and bind to different state fields in each, so a
    single shared map could only be right for one of them.  Everything
    else reads the generic map, whose rows are the names Morrison,
    Thompson and P3 declared.  Dispatching here rather than at each call
    site is what kept mp=9 out of the lane after the NSSL map landed
    (audit R-017).
    """

    source_mp = int(source_mp_physics)
    if source_mp == 18:
        return _NSSL_WRF_TO_STATE
    if source_mp == 9:
        return _MY2_WRF_TO_STATE
    return _WRF_TO_STATE


#: Parent microphysics schemes this offline-child route can carry.
#:
#: NOT a profile whitelist -- the four checks below are the ordinary
#: fail-closed kind the 2026-07-31 suite ruling keeps: the child's
#: hydrometeor mapping is written against the transported species of
#: WSM6 (6), Thompson (8), Morrison (10), NSSL (18), Thompson
#: aerosol-aware (28) and P3 (50), and a parent outside that set has no
#: mapping to refuse or accept with.  Naming the set once keeps the four
#: enforcement points from ever disagreeing about which parents the
#: mapping actually implements; each refusal still quotes the exact
#: switch and value.
#:
#: 28 is admitted for the SAME-SCHEME case only: a 28 parent forcing a 28
#: child.  Its transported inventory is classic Thompson's plus
#: ``nc``/``nwfa``/``nifa`` (all three declared scalars at
#: Registry.EM_COMMON:3036) and its two per-domain surface-emission
#: constants (:492-493); every one of them rides the generic SINT/couple
#: paths this module already runs for ``nr``/``ni``, with no new numerics.
#: What is DELIBERATELY NOT admitted is any CROSS-scheme edge touching 28 --
#: see :data:`_CROSS_SCHEME_REFUSED_MP_PHYSICS`.
#: 50 (P3) is admitted on exactly 28's terms: SAME-SCHEME only.  Its
#: transported inventory is ``qv,qc,qr,qi`` plus ``ni``/``nr`` and the
#: prognostic rime pair ``qir``/``qib`` (Registry.EM_COMMON:3038) -- no
#: qs, no qg -- and all four scalars ride the generic SINT/couple paths
#: like the moments beside them.  Every CROSS-scheme edge touching 50 is
#: refused at this module's own gates by
#: :data:`_P3_OFFLINE_EDGE_UNBUILT_MP_PHYSICS` below -- no longer through
#: the derived closure mirror, because the online nest lane RATIFIED the
#: rime-pair closure (``microphysics_transition.p3_edge_entry_reference``
#: / ``p3_edge_exit_reference``) and this offline lane has not wired
#: either leg.
#: mp_physics=16 (WDM6) is DELIBERATELY ABSENT, in the same shape mp=28's
#: cross-scheme refusal takes: the scheme ships and runs, but this module's
#: wrfout field map has no row for its CCN reservoir.  ``nn`` and NSSL's
#: ``qnn`` both publish under QNCCN, so admitting 16 here would make the
#: reverse mapping ambiguous exactly where the child state is built, and a
#: WDM6 child forced from a WDM6 parent would silently start with a
#: zero-filled reservoir -- the inert-aerosol failure mode mp=28's
#: microphysics_init hook exists to prevent.  Admitting it means giving the
#: field map a scheme-qualified QNCCN row and measuring the closure, not
#: adding 16 to this set.
#: DERIVED from the physics registry's per-option ``consumers.offline_child``
#: rows (``same_scheme``), which carry the reasons above and the two this
#: module never named: mp=0 and mp=1 (admitted by this module's own
#: transported-field helper while this set refused them) and mp=9 (no
#: scheme-qualified QHAIL/QNHAIL row in the field map).  Each refused row
#: cites its defect, so admitting a scheme is one row in
#: tools/build_registry.py and never a literal here.
def _offline_child_mp_physics() -> frozenset[int]:
    from gpuwm.physics_registry import consumer_rows_by_selector

    return frozenset(
        int(mp) for mp, row in
        consumer_rows_by_selector("microphysics", "offline_child").items()
        if row.get("same_scheme") is True)


def offline_child_refusal(mp_physics: int) -> str | None:
    """Why a same-scheme parent of ``mp_physics`` is refused, or ``None``."""

    from gpuwm.physics_registry import consumer_rows_by_selector

    row = consumer_rows_by_selector("microphysics", "offline_child").get(
        int(mp_physics))
    if row is None:
        return (f"mp_physics={mp_physics} is not an implemented microphysics "
                "option in gpuwm/physics_registry_v2.json")
    return None if row.get("same_scheme") is True else row.get("refusal")


OFFLINE_CHILD_MP_PHYSICS = _offline_child_mp_physics()

#: Schemes that may not participate in an offline CROSS-physics conversion.
#: DERIVED from ``gpuwm.core.microphysics_transition.
#: UNVALIDATED_MIXED_EDGE_SELECTORS`` rather than re-spelled, so the mirror
#: cannot drift again: the online nest lane refuses every mixed edge touching
#: one of these because no cross-scheme entry closure for its moments has
#: been measured, and an offline downscale that performed the same
#: unvalidated closure through a different code path would defeat that
#: refusal rather than respect it.
#:
#: mp=16 is in this set even though ``OFFLINE_CHILD_MP_PHYSICS`` already
#: excludes it, i.e. an mp=16 parent is refused EARLIER, at
#: ``ParentPhysicsBinding.__post_init__``.  That earlier gate is a stronger
#: refusal but it is a DIFFERENT guarantee -- it says "this module cannot
#: read the field", not "this closure is unmeasured" -- and it would silently
#: stop being a refusal the day the QNCCN field-map row lands.  Keeping the
#: mirror exact means the closure question is answered on its own terms, at
#: the site whose job it is.  mp=50 walked that exact path end to end: it
#: sat in mp=16's shape until the field map learned its qir/qib rows and 50
#: joined the admitted set (the "unreadable" gate vanished), this derived
#: set carried the live refusal for a while, and then the online lane
#: ratified the rime-pair closure and 50 left
#: ``UNVALIDATED_MIXED_EDGE_SELECTORS`` too -- retiring it from here BY
#: DERIVATION, with nothing to re-spell.  What still refuses a P3
#: cross-scheme edge offline is the next constant down, on its own feet.
_CROSS_SCHEME_REFUSED_MP_PHYSICS = frozenset(
    _mt.UNVALIDATED_MIXED_EDGE_SELECTORS)

#: Selectors whose online cross-scheme closure IS ratified but whose
#: offline conversion leg is UNBUILT at this module's sites.  mp=50: the
#: online nest lane closes every P3 mixed edge with the measured rime-pair
#: pair of maps -- ``p3_edge_entry_reference`` merges qi/qs/qg and
#: diagnoses qir/qib, ``p3_edge_exit_reference`` splits by rime state --
#: but nothing here runs either map: :func:`map_microphysics_to_nssl18`
#: consumes a five-species qi/qs/qg inventory directly, and the forcing
#: and initial-state paths convert only through it.  Without this named
#: gate a P3 parent would pass the closure mirror above (50 is not in it
#: any more) and then die on an incidental "lacks transported fields
#: ['qg', 'qs']" shape error -- or worse, a caller padding zero qs/qg
#: would silently drop the parent's rime state.  Wiring the ratified maps
#: into this lane is the named follow-up ``offline-p3-edge-closure``;
#: landing it retires this constant (gate law / guard-retirement law).
_P3_OFFLINE_EDGE_UNBUILT_MP_PHYSICS = frozenset({50})

#: The same statement for the three parents that joined the SAME-scheme
#: admission set with audit R-017.  Admitting a parent to be read is not
#: admitting it to be CONVERTED: :func:`map_microphysics_to_nssl18` is the
#: only converting site, and for each of these it would produce a wrong
#: child rather than refuse, so each gets its own named reason and none is
#: shared.  Building the leg retires its row (guard-retirement law); the
#: same-scheme 0->0, 1->1 and 9->9 downscales this lane now supports are
#: unaffected, because a same-scheme edge never reaches this gate.
_OFFLINE_CROSS_LEG_UNBUILT_REASONS = {
    0: ("an mp=0 parent transports vapour and the warm-rain pair and no "
        "frozen species at all, and the NSSL conversion consumes a "
        "six-species qv/qc/qr/qi/qs/qg inventory; diagnosing ice, snow and "
        "graupel from a parent that carries none is a cross-scheme "
        "question with no measured closure at this site, and padding zeros "
        "would hand the child a frozen inventory the parent never had"),
    1: ("an mp=1 (Kessler) parent transports qv/qc/qr and no frozen "
        "species, and the NSSL conversion consumes a six-species "
        "inventory -- the same missing closure mp=0 has, for the same "
        "reason"),
    9: ("this site zeroes the target's hail mass before the per-scheme "
        "arms run (``result['qh'] = zeros``, the Morrison arm being the "
        "only one that fills it), so an mp=9 parent's SEVENTH transported "
        "species would be silently dropped on the way to NSSL, which "
        "carries hail itself.  The online nest lane closes this edge "
        "properly -- mp=9 and mp=18 are both dual-rimed, so qg->qg and "
        "qh->qh map straight across "
        "(gpuwm/core/microphysics_transition._DUAL_RIMED_SELECTORS) -- and "
        "nothing here runs that mapping"),
}
_OFFLINE_CROSS_LEG_UNBUILT_MP_PHYSICS = frozenset(
    _OFFLINE_CROSS_LEG_UNBUILT_REASONS)


def _cross_scheme_refusal_clause(mp: int) -> str:
    """Name the scheme and the moments its missing closure would need.

    One source of truth with the online nest lane
    (:mod:`gpuwm.core.microphysics_transition`), so an offline refusal can
    never tell a WDM6 operator that their scheme is Thompson.
    """

    scheme = _mt._UNVALIDATED_MIXED_EDGE_REASONS[int(mp)][0]
    moments = "/".join(_mt.UNVALIDATED_MIXED_EDGE_MOMENTS[int(mp)])
    return f"mp_physics={int(mp)} ({scheme}), moments {moments}"


def _refuse_unbuilt_p3_offline_edge(source_mp: int, target_mp: int) -> None:
    """Refuse a P3 cross-scheme edge by name, at every converting site.

    One function for the initial-state path, the forcing path and the
    direct NSSL converter, so the three sites cannot drift apart the way
    the three admission literals once did (the mp=28 lesson recorded at
    ``HRRR_ANALYZED_HYDROMETEOR_MP_PHYSICS``).  See
    :data:`_P3_OFFLINE_EDGE_UNBUILT_MP_PHYSICS` for why this is its own
    refusal and not the retired closure mirror.
    """

    source_mp, target_mp = int(source_mp), int(target_mp)
    if source_mp == target_mp:
        return
    unbuilt = sorted({source_mp, target_mp}
                     & _OFFLINE_CROSS_LEG_UNBUILT_MP_PHYSICS)
    if unbuilt:
        mp = unbuilt[0]
        raise OfflineChildContractError(
            f"offline cross-physics conversion across the mp_physics="
            f"{source_mp} -> {target_mp} edge is REFUSED: "
            f"{_OFFLINE_CROSS_LEG_UNBUILT_REASONS[mp]}.  Same-scheme "
            f"{mp} -> {mp} downscaling IS supported (that is what this "
            "lane admits mp_physics=" + str(mp) + " for); to change "
            "microphysics between the parent and the child, run the child "
            "as an ONLINE nest, where the edge closure is ported.")
    if not ({source_mp, target_mp}
            & _P3_OFFLINE_EDGE_UNBUILT_MP_PHYSICS):
        return
    raise OfflineChildContractError(
        f"offline cross-physics conversion across the mp_physics="
        f"{source_mp} -> {target_mp} edge is REFUSED: the online nest "
        "lane's ratified P3 (mp_physics=50) closure "
        "(gpuwm/core/microphysics_transition.py::p3_edge_entry_reference/"
        "p3_edge_exit_reference) has no offline leg at this site -- the "
        "NSSL mapping consumes a five-species qi/qs/qg inventory and "
        "nothing here runs the measured merge/split first -- so "
        "converting would zero-fill or invent the single ice category's "
        "rime pair qir/qib instead of conserving it; same-scheme "
        "50 -> 50 downscaling is supported, and wiring the ratified maps "
        "into this lane is the named follow-up offline-p3-edge-closure")

#: The parents a CROSS-scheme conversion has a measured mapping for: every
#: admitted parent that is neither cross-refused nor waiting on its offline
#: conversion leg.  Derived, never re-spelled, so the cross-scheme site
#: cannot drift from the same-scheme sites above.
PARENT_SCHEME_CONTRACT = (
    OFFLINE_CHILD_MP_PHYSICS - _CROSS_SCHEME_REFUSED_MP_PHYSICS
    - _P3_OFFLINE_EDGE_UNBUILT_MP_PHYSICS
    - _OFFLINE_CROSS_LEG_UNBUILT_MP_PHYSICS)


class OfflineChildContractError(ValueError):
    """The archived parent cannot safely force the requested child."""


class MomentDiagnosisRequired(OfflineChildContractError):
    """A target number moment needs the official target-scheme initializer."""


def reserve_output_root(path, *, flag: str = "--out") -> Path:
    """Claim one child-run output directory, in words when it cannot.

    A downscale never merges into a directory that already holds a run:
    the ``report.json`` it publishes has to describe ONE run, and the
    frame series beside it has to be that run's.  ``mkdir(exist_ok=
    False)`` enforces that, but its ``FileExistsError`` reached the
    reader as a traceback whose last line was a Windows error number --
    at exit 1, from a command that had already printed a refusal moments
    earlier.  The person re-running with the same output directory gets
    one sentence naming the directory, what it holds, and the two ways
    out instead.

    An EMPTY directory is adopted, not refused: it carries no frames to
    merge with and no receipt to overwrite, so refusing it prevents
    nothing.  ``flag`` is the flag the caller actually typed -- the two
    doors onto this route spell it ``--out`` and ``--outdir``.
    """

    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        try:
            held = sorted(child.name for child in path.iterdir())
        except OSError as probe_error:
            # Not the empty-directory case at all: the entry exists and
            # cannot be read as a directory (a plain file under that
            # name, a symlink with no destination).  This function's
            # whole subject is turning an OS-level exception into a
            # sentence, so the probe must not become the one that
            # escapes.
            detail = (getattr(probe_error, "strerror", None)
                      or str(probe_error))
            raise OfflineChildContractError(
                f"{flag} {path} exists but is not a directory this run can "
                f"claim: {detail}.  Pass a {flag} that names a directory "
                f"path, or remove {path} first.") from probe_error
        if held:
            raise OfflineChildContractError(
                f"{flag} {path} already holds a child run's output "
                f"({', '.join(held)[:120]}), and a downscale never writes "
                f"into a directory it did not create -- the report.json it "
                f"publishes has to describe one run, and the frames beside "
                f"it have to be that run's.  Pass a new {flag}, or remove "
                f"{path} first.") from error
    return path.resolve()


def _unsupported_parent_clause(mp_physics: int, *, what: str) -> str:
    """The refusal text for a parent scheme this lane cannot read.

    Names the scheme, the breakage that keeps it out and the way out, from
    the registry's own ``consumers.offline_child`` row -- so the message a
    user reads is the row an editor would change, and a bare
    "unsupported ... mp_physics=N" can never come back (audit R-017).
    """

    admitted = ", ".join(str(value) for value in sorted(
        OFFLINE_CHILD_MP_PHYSICS))
    reason = offline_child_refusal(int(mp_physics))
    if reason is None:
        reason = ("no reason is recorded for it in the physics registry's "
                  "consumers.offline_child row")
    return (
        f"the offline downscale lane cannot read a {what} parent of "
        f"mp_physics={mp_physics}: {reason}. Parent schemes this lane "
        f"carries same-scheme: {admitted}. The admission is one row in "
        "tools/build_registry.py "
        "(components.microphysics.options.<option>.consumers.offline_child), "
        "not a literal here; run the parent's own scheme as the child, or "
        "prepare the child from a parent archive written by an admitted "
        "scheme.")


@dataclass(frozen=True)
class ParentPhysicsBinding:
    """Authoritative parent-scheme identity from a companion setup record."""

    mp_physics: int
    morr_rimed_ice: int | None
    domain_id: int
    evidence_kind: str
    evidence_path: Path
    evidence_sha256: str

    def __post_init__(self) -> None:
        if int(self.mp_physics) not in OFFLINE_CHILD_MP_PHYSICS:
            raise OfflineChildContractError(
                _unsupported_parent_clause(self.mp_physics, what="bound"))
        if int(self.domain_id) < 1:
            raise OfflineChildContractError("bound parent domain_id must be >= 1")
        if int(self.mp_physics) == 10 and self.morr_rimed_ice not in {0, 1}:
            raise OfflineChildContractError(
                "bound Morrison parent requires morr_rimed_ice=0/1")
        if (int(self.mp_physics) != 10
                and self.morr_rimed_ice is not None):
            raise OfflineChildContractError(
                "morr_rimed_ice evidence is only valid for mp_physics=10")
        if self.evidence_kind not in {"gpuwm-restart", "wrf-namelist"}:
            raise OfflineChildContractError(
                f"unsupported parent physics evidence {self.evidence_kind!r}")
        if not Path(self.evidence_path).is_file():
            raise OfflineChildContractError(
                f"parent physics evidence does not exist: {self.evidence_path}")
        digest = str(self.evidence_sha256).lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise OfflineChildContractError(
                "parent physics evidence_sha256 must be one SHA-256 hex digest")

    def receipt(self) -> Mapping[str, object]:
        return MappingProxyType({
            "mp_physics": int(self.mp_physics),
            "morr_rimed_ice": self.morr_rimed_ice,
            "domain_id": int(self.domain_id),
            "evidence_kind": self.evidence_kind,
            "evidence_path": str(Path(self.evidence_path).resolve()),
            "evidence_sha256": str(self.evidence_sha256).lower(),
        })


def _canonical_sha256(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def bind_parent_physics_from_gpuwm_restart(
        path: str | Path) -> ParentPhysicsBinding:
    """Bind parent MP identity to a validated gpuwm restart header."""

    from gpuwm.io.restart import read_restart_header
    path = Path(path).resolve()
    header = read_restart_header(path)
    config = header.get("config")
    setup = header.get("physics_setup")
    fingerprint = header.get("physics_setup_fingerprint")
    if not isinstance(config, dict) or not isinstance(setup, dict):
        raise OfflineChildContractError(
            f"{path} lacks complete gpuwm config/physics setup evidence")
    if fingerprint != _canonical_sha256(setup):
        raise OfflineChildContractError(
            f"{path} physics setup fingerprint is invalid")
    microphysics = setup.get("microphysics")
    if not isinstance(microphysics, dict):
        raise OfflineChildContractError(
            f"{path} lacks resolved microphysics setup evidence")
    mp_physics = int(config.get("mp_physics", -1))
    if int(microphysics.get("scheme_id", -2)) != mp_physics:
        raise OfflineChildContractError(
            f"{path} config and resolved microphysics identities disagree")
    domain_id = int(header.get("grid_id", config.get("grid_id", 0)))
    morr = None
    if mp_physics == 10:
        morr = int(config.get("morr_rimed_ice", -1))
        resolved = microphysics.get("morrison_rimed_ice")
        if (not isinstance(resolved, dict)
                or int(resolved.get("selection", -2)) != morr):
            raise OfflineChildContractError(
                f"{path} Morrison rimed-ice config/setup identities disagree")
    evidence = {
        "format_version": header.get("format_version"),
        "grid_id": domain_id,
        "config": config,
        "physics_setup": setup,
        "physics_setup_fingerprint": fingerprint,
    }
    return ParentPhysicsBinding(
        mp_physics=mp_physics, morr_rimed_ice=morr, domain_id=domain_id,
        evidence_kind="gpuwm-restart", evidence_path=path,
        evidence_sha256=_canonical_sha256(evidence))


def bind_parent_physics_from_wrf_namelist(
        path: str | Path, *, domain_id: int = 1) -> ParentPhysicsBinding:
    """Bind one WRF domain's MP identity to exact namelist bytes."""

    from gpuwm.namelist_import import parse_namelist
    path = Path(path).resolve()
    domain_id = int(domain_id)
    if domain_id < 1:
        raise OfflineChildContractError("WRF binding domain_id must be >= 1")
    parsed = parse_namelist(path)
    physics = parsed.get("physics", {})

    def domain_value(name: str, *, required: bool, default=None):
        values = physics.get(name)
        if not values:
            if required:
                raise OfflineChildContractError(
                    f"{path} &physics lacks authoritative {name}")
            return default
        return values[min(domain_id - 1, len(values) - 1)]

    mp_physics = int(domain_value("mp_physics", required=True))
    morr = None
    if mp_physics == 10:
        morr = int(domain_value(
            "morr_rimed_ice", required=False, default=1))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return ParentPhysicsBinding(
        mp_physics=mp_physics, morr_rimed_ice=morr, domain_id=domain_id,
        evidence_kind="wrf-namelist", evidence_path=path,
        evidence_sha256=digest)


def _decode_time(variable) -> datetime:
    raw = np.asarray(variable[:])
    if raw.shape[0] != 1:
        raise OfflineChildContractError(
            f"one history file must contain exactly one Time record, got {raw.shape}")
    row = raw[0]
    if row.dtype.kind == "S":
        value = b"".join(row.tolist()).decode("ascii")
    else:
        value = "".join(str(item) for item in row.tolist())
    try:
        return datetime.strptime(value, _TIME_FORMAT)
    except ValueError as exc:
        raise OfflineChildContractError(
            f"invalid WRF Times value {value!r}") from exc


def _hash_array(digest, label: str, value) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(label.encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def _infer_mp_physics(
        inventory: frozenset[str], *, source_kind: str) -> int | None:
    """Advisory scheme id from the WRF names one history frame carries.

    The ladder is ordered MOST DISCRIMINATING FIRST, because several
    packages' inventories are strict supersets of others'.  Every arm
    below is a name declared by exactly one WRF package, or a pair whose
    presence-and-absence only one package satisfies; an arm that merely
    matched a subset reported the wrong scheme with no way for a reader to
    tell, which is what this function was doing for mp 9, 16 and 18 -- all
    three were reported as somebody else's scheme in receipts and in the
    cross-frame agreement check (audit R-018).

    ``source_kind`` is the producer, ``"wrf"`` or ``"gpuwm"``, because the
    warm-rain packages are separated by the WRITER and not by the scheme:
    stock WRF's passiveqv (mp=0) transports qv alone, but gpuwm's own mp=0
    allocates and advects the warm-rain pair beside it
    (:func:`_transported_source_fields`) and gpuwm/io/wrfout.py writes all
    three whenever the state is moist.  A gpuwm frame carrying exactly
    QVAPOR/QCLOUD/QRAIN is therefore mp=0 OR mp=1 with nothing in the
    inventory to separate them, and this function says so by returning
    ``None`` rather than naming the one that happens to be second.
    """
    if source_kind not in ("wrf", "gpuwm"):
        raise ValueError(
            f"parent history producer must be 'wrf' or 'gpuwm', got "
            f"{source_kind!r}")
    # NSSL is first: its volume moments are declared by no other package
    # (Registry.EM_COMMON:3033), so QVGRAUPEL/QVHAIL identify it outright.
    # Without this arm an NSSL stream matched the Morrison arm below on
    # {QNSNOW, QNGRAUPEL} -- and NSSL is an ADMITTED offline parent, so
    # the mislabel reached a receipt for a supported configuration.
    if {"QVGRAUPEL", "QVHAIL"} <= inventory:
        return 18
    # Milbrandt-Yau (Registry.EM_COMMON:3025) is the other package that
    # declares QHAIL and QNHAIL; NSSL is already claimed above, so the
    # pair without the volume moments is MY2's discriminant.  It must
    # precede the Morrison arm for the same superset reason: MY2 carries
    # QNSNOW and QNGRAUPEL too.
    if {"QHAIL", "QNHAIL"} <= inventory:
        return 9
    # mp=28 before mp=8 because its inventory is a strict superset of
    # mp=8's: it carries QNRAIN/QNICE too, so the classic-Thompson arm
    # below would claim an aerosol-aware stream.
    if {"QNWFA", "QNIFA"} <= inventory:
        return 28
    # mp=50 next, and before mp=8, for the same superset reason: a P3
    # stream carries QNRAIN/QNICE beside its rime pair, so the classic-
    # Thompson arm below would claim it.  QIR/QIB are declared by no other
    # scheme (Registry.EM_COMMON:555/:557), which makes the pair the
    # discriminant.
    if {"QIR", "QIB"} <= inventory:
        return 50
    # WDM6 (Registry.EM_COMMON:3031, ``scalar:qnn,qnc,qnr``) before the
    # single-moment WSM6 arm it would otherwise fall into: it carries the
    # same six masses as WSM6 and adds a warm-rain number pair plus the
    # CCN reservoir, and it declares NO ice number, which is what
    # separates it from Morrison and Thompson.  QNCCN alone is not the
    # discriminant -- NSSL publishes ``qnn`` under the same name -- but
    # NSSL is claimed two arms above.
    if ({"QNCCN", "QNCLOUD", "QNRAIN"} <= inventory
            and "QNICE" not in inventory):
        return 16
    if {"QNSNOW", "QNGRAUPEL"} <= inventory:
        return 10
    if {"QNRAIN", "QNICE"} <= inventory:
        return 8
    if {"QICE", "QSNOW", "QGRAUP"} <= inventory:
        return 6
    # Kessler (Registry.EM_COMMON:3015) declares moist:qv,qc,qr and no
    # frozen species; passiveqv (:3014) declares qv alone.  Both were
    # reported as "unknown" -- a receipt row that reads as "this stream is
    # unreadable" for two packages this lane now carries.  Both arms are
    # ABSENCE tests as well as presence tests, because every remaining
    # package is a superset of Kessler's and Kessler's is a superset of
    # passiveqv's: a frame carrying any frozen species or any number
    # moment is not one of these two, and stays unknown rather than being
    # labelled with the smallest package that fits.
    _frozen = {"QICE", "QSNOW", "QGRAUP", "QHAIL"}
    _moments = {"QNCLOUD", "QNRAIN", "QNICE", "QNSNOW", "QNGRAUPEL",
                "QNHAIL", "QNCCN", "QNDROP", "QNWFA", "QNIFA",
                "QVGRAUPEL", "QVHAIL", "QIR", "QIB"}
    if not (inventory & (_frozen | _moments)):
        if {"QVAPOR", "QCLOUD", "QRAIN"} <= inventory:
            # Ambiguous on a gpuwm tape and only there: see the docstring.
            # Claiming Kessler would be the same class of mislabel this
            # ladder was rewritten to end, one package later.
            return None if source_kind == "gpuwm" else 1
        if inventory & {"QVAPOR"} and not (inventory & {"QCLOUD", "QRAIN"}):
            return 0
    return None


@dataclass(frozen=True)
class ParentHistoryFrame:
    """Metadata-only proof for one archived parent state."""

    path: Path
    valid_time: datetime
    source_kind: str
    source_mp_physics: int | None
    inferred_mp_physics: int | None
    dimensions: Mapping[str, int]
    variables: frozenset[str]
    geometry_sha256: str


@dataclass(frozen=True)
class ParentHistoryContract:
    """Validated ordered parent series ready for native child preparation."""

    frames: tuple[ParentHistoryFrame, ...]
    interval_seconds: float
    geometry_sha256: str
    source_kind: str
    source_mp_physics: int | None
    physics_binding: ParentPhysicsBinding | None
    max_boundary_interval_seconds: float

    @property
    def start_time(self) -> datetime:
        return self.frames[0].valid_time

    @property
    def end_time(self) -> datetime:
        return self.frames[-1].valid_time


def inspect_parent_history_frame(
        path: str | Path, *, source_mp_physics: int | None = None,
) -> ParentHistoryFrame:
    """Inspect one gpuwm/WRF history file without loading 3-D trajectory data."""

    path = Path(path)
    with netCDF4.Dataset(path) as dataset:
        feedback = (
            dataset.getncattr("GPUWM_FEEDBACK")
            if "GPUWM_FEEDBACK" in dataset.ncattrs() else None)
        if str(feedback).strip().lower() == "experimental":
            raise OfflineChildContractError(
                f"{path} carries experimental two-way feedback provenance; "
                "gpuwm downscale assumes a one-way parent archive and "
                "refuses feedback-modified parent history")
        if "Times" not in dataset.variables:
            raise OfflineChildContractError(f"{path} has no WRF Times variable")
        valid_time = _decode_time(dataset.variables["Times"])
        variables = frozenset(dataset.variables)
        missing = sorted(_REQUIRED_DYNAMICS - variables)
        if missing:
            raise OfflineChildContractError(
                f"{path} is missing offline-child trajectory fields {missing}")
        dimensions = {}
        for name in _GEOMETRY_DIMS:
            if name not in dataset.dimensions:
                raise OfflineChildContractError(
                    f"{path} is missing WRF geometry dimension {name!r}")
            dimensions[name] = len(dataset.dimensions[name])

        title = str(getattr(dataset, "TITLE", ""))
        source_kind = "gpuwm" if (
            "gpuwm" in title.lower() or "GPUWM_WRITE_COMPLETE" in dataset.ncattrs()
        ) else "wrf"
        inferred_mp = _infer_mp_physics(variables, source_kind=source_kind)
        bound_mp = None
        if source_mp_physics is None and inferred_mp is not None and (
                inferred_mp not in OFFLINE_CHILD_MP_PHYSICS):
            # Nothing was declared, so this inference is the only scheme
            # evidence there is -- and the arms are single-package
            # discriminants, not subset matches, so a positive one is not a
            # guess.  Without this the reader fell through to the blind
            # six-species contract, which a WDM6 archive SATISFIES: the run
            # would have been prepared with the parent's warm-rain numbers
            # and its CCN reservoir silently dropped, which is the exact
            # cross-scheme entry-closure breakage the mixed nest edge
            # refuses by name (audit R-018).
            raise OfflineChildContractError(
                f"{path} carries the transported inventory of "
                f"mp_physics={inferred_mp} and no parent scheme was "
                "declared, so that inventory is the only evidence: "
                + _unsupported_parent_clause(inferred_mp, what="inferred"))
        if source_mp_physics is not None:
            requested = int(source_mp_physics)
            if requested not in OFFLINE_CHILD_MP_PHYSICS:
                raise OfflineChildContractError(
                    _unsupported_parent_clause(requested, what="declared"))
            # Inventory-only inference is advisory. WRF streams may retain
            # dormant number variables, and unified NSSL can expose an
            # inventory that looks like another multi-moment scheme. The
            # companion namelist/setup/manifest is authoritative.
            bound_mp = requested

        digest = hashlib.sha256()
        for name in _GEOMETRY_DIMS:
            digest.update(f"dim:{name}={dimensions[name]};".encode("ascii"))
        for name in _GEOMETRY_ATTRS:
            if name in dataset.ncattrs():
                digest.update(f"attr:{name}={dataset.getncattr(name)!r};".encode())
        for name in _STATIC_GEOMETRY_FIELDS:
            if name not in dataset.variables:
                if name in {"XLAT", "XLONG", "MAPFAC_M"}:
                    # Some minimal stock-WRF history streams omit these; the
                    # projection/dimension identity remains enforceable and a
                    # companion native setup archive supplies the arrays.
                    continue
                raise OfflineChildContractError(
                    f"{path} is missing vertical/static geometry field {name}")
            value = dataset.variables[name][:]
            if value.ndim and value.shape[0] == 1:
                value = value[0]
            _hash_array(digest, name, value)

    return ParentHistoryFrame(
        path=path.resolve(), valid_time=valid_time, source_kind=source_kind,
        source_mp_physics=bound_mp, inferred_mp_physics=inferred_mp,
        dimensions=MappingProxyType(dimensions), variables=variables,
        geometry_sha256=digest.hexdigest(),
    )


def validate_parent_history(
        paths: Sequence[str | Path], *, max_boundary_interval_seconds: float,
        source_mp_physics: int | None = None,
        physics_binding: ParentPhysicsBinding | None = None,
) -> ParentHistoryContract:
    """Prove an archived parent series before constructing a child.

    ``max_boundary_interval_seconds`` is intentionally mandatory.  Cadence is
    a scientific choice tied to child resolution and expected advection; the
    tool will not silently bless hourly parent history for a 500-m child.
    """

    if physics_binding is not None:
        if source_mp_physics is not None:
            raise OfflineChildContractError(
                "pass physics_binding or source_mp_physics, not both")
        source_mp_physics = int(physics_binding.mp_physics)
    maximum = float(max_boundary_interval_seconds)
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise OfflineChildContractError(
            "max_boundary_interval_seconds must be finite and positive")
    frames = tuple(inspect_parent_history_frame(
        path, source_mp_physics=source_mp_physics) for path in paths)
    if len(frames) < 2:
        raise OfflineChildContractError(
            "offline child forcing requires at least two parent history frames")
    if len({frame.geometry_sha256 for frame in frames}) != 1:
        raise OfflineChildContractError(
            "parent history geometry/static state changes between frames")
    if len({frame.source_kind for frame in frames}) != 1:
        raise OfflineChildContractError(
            "parent history mixes gpuwm and stock-WRF producers")
    if len({frame.source_mp_physics for frame in frames}) != 1:
        raise OfflineChildContractError(
            "parent history bound microphysics identity changes between frames")
    if len({frame.inferred_mp_physics for frame in frames}) != 1:
        raise OfflineChildContractError(
            "parent history advisory moisture inventory changes between frames")
    seconds = np.asarray([
        (frame.valid_time - frames[0].valid_time).total_seconds()
        for frame in frames
    ], dtype=np.float64)
    differences = np.diff(seconds)
    if not np.all(np.isfinite(differences)) or np.any(differences <= 0.0):
        raise OfflineChildContractError(
            "parent history times must be unique and strictly increasing")
    interval = float(differences[0])
    if not np.all(differences == interval):
        raise OfflineChildContractError(
            f"parent history cadence is irregular: {differences.tolist()} seconds")
    if interval > maximum:
        raise OfflineChildContractError(
            f"parent history cadence {interval:g} s exceeds the declared child "
            f"forcing limit {maximum:g} s; regenerate a denser parent archive")
    return ParentHistoryContract(
        frames=frames, interval_seconds=interval,
        geometry_sha256=frames[0].geometry_sha256,
        source_kind=frames[0].source_kind,
        source_mp_physics=frames[0].source_mp_physics,
        physics_binding=physics_binding,
        max_boundary_interval_seconds=maximum,
    )


def read_parent_microphysics(
        path: str | Path, *, source_mp_physics: int | None = None,
) -> Mapping[str, np.ndarray]:
    """Read only transported moisture fields from one WRF history record.

    With ``source_mp_physics`` bound, the completeness check is the
    scheme's own transported inventory (``_transported_source_fields``),
    read through the same WRF-name mapping ``_raw_parent_state`` uses.
    Without it the historical closed-world contract stands: the six-species
    mass set is required, because with no scheme evidence there is no
    smaller inventory this reader could honestly call complete -- a P3
    archive (no QSNOW/QGRAUP by Registry.EM_COMMON:3038) is readable
    through the evidence-bearing form, not by weakening the blind one.
    """

    fields: dict[str, np.ndarray] = {}
    wrf_mapping = _WRF_TO_STATE
    required = set(_MASS_FIELDS)
    label = "transported parent mass fields"
    if source_mp_physics is not None:
        source_mp = int(source_mp_physics)
        if source_mp not in OFFLINE_CHILD_MP_PHYSICS:
            raise OfflineChildContractError(
                _unsupported_parent_clause(source_mp, what="declared"))
        wrf_mapping = _scheme_wrf_to_state(source_mp)
        required = set(_transported_source_fields(source_mp))
        label = f"mp_physics={source_mp} transported parent fields"
    # Decoded by the Rust bridge: transported moisture is meteorological
    # field data, whoever wrote the tape.
    with netcdf_bridge.open_dataset(path) as dataset:
        for wrf_name, state_name in wrf_mapping.items():
            if wrf_name not in dataset.variables:
                continue
            value = np.asarray(dataset.variables[wrf_name][:])
            if value.shape[0] != 1:
                raise OfflineChildContractError(
                    f"{path}/{wrf_name} must have exactly one Time record")
            fields[state_name] = np.ascontiguousarray(value[0], dtype=np.float32)
    missing = sorted(required - set(fields))
    if missing:
        raise OfflineChildContractError(
            f"{path} lacks {label} {missing}")
    return MappingProxyType(fields)


def _same_shape(fields: Mapping[str, np.ndarray]) -> tuple[int, ...]:
    shapes = {tuple(np.asarray(value).shape) for value in fields.values()}
    if len(shapes) != 1:
        raise OfflineChildContractError(
            f"microphysics fields have inconsistent shapes {sorted(shapes)}")
    return next(iter(shapes))


def map_microphysics_to_nssl18(
        fields: Mapping[str, np.ndarray], *, source_mp_physics: int,
        morr_rimed_ice: int | None = None,
        diagnose_missing: Callable[[Mapping[str, np.ndarray], Sequence[str]],
                                   Mapping[str, np.ndarray]] | None = None,
        active_mass_threshold: float = 1.0e-8,
) -> tuple[Mapping[str, np.ndarray], Mapping[str, object]]:
    """Map WSM6/Thompson/Morrison transported state into NSSL mp18.

    The optional ``diagnose_missing`` callback is the exact target-scheme
    ``calcnfromq`` implementation.  Until that official initializer is
    supplied, any active mass category lacking its number moment fails closed.
    This prevents a superficially runnable but physically invalid child.
    """

    source_mp = int(source_mp_physics)
    if source_mp in _CROSS_SCHEME_REFUSED_MP_PHYSICS:
        # NAMED refusal, in the same words the online nest lane uses.  What
        # is missing is a validated cross-scheme entry closure for the
        # scheme's own moments, which is exactly why
        # gpuwm/core/microphysics_transition.py refuses every one of its
        # MIXED nest edges.  Converting them here through a second,
        # unvalidated path would defeat that refusal.  The scheme name and
        # the moment list come from that module, never re-spelled here.
        raise OfflineChildContractError(
            "offline cross-physics conversion to NSSL mp18 is REFUSED for "
            f"{_cross_scheme_refusal_clause(source_mp)}: no cross-scheme "
            "entry closure for those moments has been measured, and the "
            "online nest lane refuses the same edge "
            "(gpuwm/core/microphysics_transition.py::"
            f"UNVALIDATED_MIXED_EDGE_SELECTORS); same-scheme {source_mp} -> "
            f"{source_mp} downscaling is supported")
    _refuse_unbuilt_p3_offline_edge(source_mp, 18)
    if source_mp not in PARENT_SCHEME_CONTRACT:
        raise OfflineChildContractError(
            "offline cross-physics conversion to NSSL mp18 has no conversion "
            f"leg for source mp_physics={source_mp}; the parents it "
            "converts are "
            + ", ".join(str(value) for value in sorted(PARENT_SCHEME_CONTRACT))
            + ".  Same-scheme downscaling is a different question and is "
            "supported for "
            + ", ".join(str(value)
                        for value in sorted(OFFLINE_CHILD_MP_PHYSICS)))
    normalized = {name: np.asarray(value, dtype=np.float32)
                  for name, value in fields.items()}
    if source_mp == 18:
        missing = sorted(set(_NSSL_FIELDS) - set(normalized))
        if missing:
            raise OfflineChildContractError(
                f"NSSL passthrough lacks transported fields {missing}")
        result = {}
        shape = _same_shape({name: normalized[name] for name in _NSSL_FIELDS})
        for name in _NSSL_FIELDS:
            value = normalized[name]
            if value.shape != shape or not np.isfinite(value).all() or np.any(value < 0):
                raise OfflineChildContractError(
                    f"NSSL passthrough field {name} is non-finite, negative, "
                    "or wrong-shaped")
            result[name] = np.array(value, copy=True, order="C")
        return MappingProxyType(result), MappingProxyType({
            "source_mp_physics": 18,
            "target_mp_physics": 18,
            "category_mapping": "nssl18-passthrough",
            "carried_source_moments": tuple(_NSSL_FIELDS[7:]),
            "diagnosed_target_moments": (),
            "active_mass_threshold": float(active_mass_threshold),
        })
    missing_mass = sorted(set(_MASS_FIELDS) - set(normalized))
    if missing_mass:
        raise OfflineChildContractError(
            f"source microphysics lacks mass fields {missing_mass}")
    shape = _same_shape({name: normalized[name] for name in _MASS_FIELDS})
    for name in _MASS_FIELDS:
        value = normalized[name]
        if not np.isfinite(value).all() or np.any(value < 0.0):
            raise OfflineChildContractError(
                f"source microphysics mass field {name} is non-finite or negative")
    result = {name: np.array(normalized[name], copy=True, order="C")
              for name in _MASS_FIELDS}
    result["qh"] = np.zeros(shape, dtype=np.float32)
    category_mapping = "graupel-to-graupel"
    if source_mp == 10:
        if morr_rimed_ice not in {0, 1}:
            raise OfflineChildContractError(
                "Morrison -> NSSL requires explicit morr_rimed_ice=0/1")
        if int(morr_rimed_ice) == 1:
            result["qh"] = result["qg"]
            result["qg"] = np.zeros(shape, dtype=np.float32)
            category_mapping = "morrison-hail-to-nssl-hail"

    aliases = {
        "qndrop": ("qndrop", "nc"), "qnr": ("qnr", "nr"),
        "qni": ("qni", "ni"), "qns": ("qns", "ns"),
        "qng": ("qng", "ng"), "qnh": ("qnh", "nh"),
        "qvolg": ("qvolg", "volg"), "qvolh": ("qvolh", "volh"),
    }
    carried = []
    for target, choices in aliases.items():
        for choice in choices:
            if choice in normalized:
                value = np.asarray(normalized[choice], dtype=np.float32)
                if value.shape != shape:
                    raise OfflineChildContractError(
                        f"source {choice} shape {value.shape} != {shape}")
                if not np.isfinite(value).all() or np.any(value < 0.0):
                    raise OfflineChildContractError(
                        f"source moment {choice} is non-finite or negative")
                result[target] = np.array(value, copy=True, order="C")
                carried.append(target)
                break

    # Morrison hail reclassification carries its graupel moments to hail.
    if source_mp == 10 and morr_rimed_ice == 1:
        if "qng" in result and "qnh" not in result:
            result["qnh"] = result.pop("qng")
        if "qvolg" in result and "qvolh" not in result:
            result["qvolh"] = result.pop("qvolg")

    required_by_mass = {
        "qc": "qndrop", "qr": "qnr", "qi": "qni", "qs": "qns",
        "qg": "qng", "qh": "qnh",
    }
    needs = []
    threshold = float(active_mass_threshold)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise OfflineChildContractError(
            "active_mass_threshold must be finite and non-negative")
    for mass_name, number_name in required_by_mass.items():
        active = bool(np.any(result[mass_name] > np.float32(threshold)))
        if number_name not in result:
            if active:
                needs.append(number_name)
            else:
                result[number_name] = np.zeros(shape, dtype=np.float32)
    if needs:
        if diagnose_missing is None:
            raise MomentDiagnosisRequired(
                "active NSSL categories require official calcnfromq diagnosis "
                f"for {sorted(needs)}")
        diagnosed = dict(diagnose_missing(MappingProxyType(result), tuple(needs)))
        for name in needs:
            if name not in diagnosed:
                raise MomentDiagnosisRequired(
                    f"calcnfromq did not return required moment {name}")
            value = np.asarray(diagnosed[name], dtype=np.float32)
            if value.shape != shape or not np.isfinite(value).all() or np.any(value < 0):
                raise OfflineChildContractError(
                    f"diagnosed {name} is non-finite, negative, or wrong-shaped")
            result[name] = np.ascontiguousarray(value)

    if "qvolg" not in result:
        result["qvolg"] = np.ascontiguousarray(result["qg"] / np.float32(700.0))
    if "qvolh" not in result:
        result["qvolh"] = np.ascontiguousarray(result["qh"] / np.float32(900.0))
    # WRF NSSL calcnfromq homogeneous background, in # kg-1.
    qnn_background = np.float32(0.5e9 / 1.225)
    result["qnn"] = np.maximum(
        np.float32(0.0), qnn_background - result["qndrop"]).astype(np.float32)

    unknown = set(result) - set(_NSSL_FIELDS)
    missing = set(_NSSL_FIELDS) - set(result)
    if unknown or missing:
        raise OfflineChildContractError(
            f"internal NSSL conversion inventory drift: unknown={sorted(unknown)}, "
            f"missing={sorted(missing)}")
    receipt = MappingProxyType({
        "source_mp_physics": source_mp,
        "target_mp_physics": 18,
        "category_mapping": category_mapping,
        "carried_source_moments": tuple(sorted(carried)),
        "diagnosed_target_moments": tuple(sorted(needs)),
        "qnn_background_number_per_kg": float(qnn_background),
        "graupel_init_density_kg_m3": 700.0,
        "hail_init_density_kg_m3": 900.0,
        "active_mass_threshold": threshold,
    })
    return MappingProxyType({name: result[name] for name in _NSSL_FIELDS}), receipt


#: Land/soil identity attributes a child surface source must declare.
#: Defaults here would silently rebind category semantics (water index,
#: ice index) across landuse tables, so they are required evidence.
_SURFACE_IDENTITY_ATTRS = ("MMINLU", "ISWATER", "ISLAKE", "ISICE")

#: Child-grid surface fields.  ``required`` is the minimum honest warm
#: start for a land-surface + surface-layer child; ``optional`` fields
#: are carried when present and receipted either way.  All arrays are
#: read on the EXACT child grid -- like WRF's ``ndown``, gpuwm's offline
#: child takes its static/soil identity from a child-grid initialization
#: file and replaces only the meteorology from the parent archive.
_SURFACE_REQUIRED_FIELDS = (
    "LU_INDEX", "LANDMASK", "ISLTYP", "TSK", "TMN", "VEGFRA",
    "TSLB", "SMOIS", "SNOW",
)
_SURFACE_OPTIONAL_FIELDS = (
    "SH2O", "SNOWH", "SNOWC", "SEAICE", "XICE", "PBLH", "UST",
    "PSFC", "T2", "Q2", "TH2", "U10", "V10", "XLAT", "XLONG",
)
_SURFACE_CATEGORY_FIELDS = frozenset({"LU_INDEX", "ISLTYP"})
_SURFACE_SOIL_FIELDS = frozenset({"TSLB", "SMOIS", "SH2O"})

#: The remedy half of the child-surface refusal, shared by the front
#: door's early check (`gpuwm downscale`) and the runner's late guard so
#: the two cannot drift.  It names the flag, the contract, AND an
#: in-product way to SATISFY it: the walked 2.4.1 refusal named only the
#: first two, and the walked user's own preparation already held a valid
#: child-grid file at ``wrf-native-input/wrfinput_d0N`` -- rw-wps emits
#: one per nest -- with no sentence anywhere pointing at it.
CHILD_SURFACE_SOURCE_REMEDY = (
    "pass --child-surface-from with a wrfinput or history file on the "
    "EXACT child grid (ndown-equivalent contract: downscaling replaces "
    "the meteorology, never the land identity).  gpuwm's own "
    "preprocessor builds one per nest: prepare a hierarchy whose nest "
    "IS this child grid (`gpuwm domain --ladder ...`, then rw-wps) and "
    "point the flag at <prepared>/wrf-native-input/wrfinput_d0N; an "
    "archived gpuwm or WRF history frame on the exact child grid works "
    "too")


def child_surface_requirement(cfg) -> str | None:
    """Why this child config needs ``--child-surface-from``, or ``None``.

    The predicate is :func:`gpuwm.offline_child_run.
    _initialize_child_physics`'s own: any of the land-surface, surface-
    layer or PBL selections requires child-grid soil state and land
    identity, which are never fabricated on a real-data child.
    """
    if not (getattr(cfg, "sf_surface_physics", 0)
            or getattr(cfg, "sf_sfclay_physics", 0)
            or getattr(cfg, "bl_pbl_physics", 0)):
        return None
    return (
        "child config enables surface physics (sf_surface_physics="
        f"{cfg.sf_surface_physics}, sf_sfclay_physics="
        f"{cfg.sf_sfclay_physics}, bl_pbl_physics={cfg.bl_pbl_physics}) "
        "but no child-grid surface source was given; "
        + CHILD_SURFACE_SOURCE_REMEDY)


@dataclass(frozen=True)
class ChildSurfaceState:
    """Surface/soil warm-start state read from one child-grid file."""

    path: Path
    fields: Mapping[str, np.ndarray]
    identity: Mapping[str, object]
    receipt: Mapping[str, object]


def read_child_surface_state(
        path: str | Path, *, child_ny: int, child_nx: int,
        num_soil_layers: int) -> ChildSurfaceState:
    """Read surface/soil warm-start fields from an exact-child-grid file.

    Accepts a stock-WRF ``wrfinput``/history file or a gpuwm history
    file whose mass grid is exactly the child's.  This is the offline
    analogue of WRF ``ndown``'s requirement that the child's own
    ``wrfinput`` (from ``real.exe``) supplies static and soil state --
    downscaling replaces the meteorology, never the land identity.
    """

    path = Path(path)
    fields: dict[str, np.ndarray] = {}
    # Reads the child surface FIELDS (_SURFACE_REQUIRED_FIELDS), not just
    # the dimensions above them, so it decodes and goes through Rust.
    # The f32 cast below is unaffected: the bridge promotes f32 storage to
    # f64 exactly, and casting back reproduces the stored bits.
    with netcdf_bridge.open_dataset(path) as dataset:
        for name, expected in (("south_north", int(child_ny)),
                               ("west_east", int(child_nx))):
            if name not in dataset.dimensions:
                raise OfflineChildContractError(
                    f"{path} is missing surface-grid dimension {name!r}")
            actual = len(dataset.dimensions[name])
            if actual != expected:
                raise OfflineChildContractError(
                    f"{path} {name}={actual} does not match the child "
                    f"grid {name}={expected}; the surface source must be "
                    "on the EXACT child grid")
        if "soil_layers_stag" in dataset.dimensions:
            soil = len(dataset.dimensions["soil_layers_stag"])
            if soil != int(num_soil_layers):
                raise OfflineChildContractError(
                    f"{path} soil_layers_stag={soil} does not match the "
                    f"child LSM's {num_soil_layers} soil layers")
        elif int(num_soil_layers) > 0:
            raise OfflineChildContractError(
                f"{path} has no soil_layers_stag dimension; the child's "
                "land-surface scheme requires child-grid soil state")
        missing_attrs = [name for name in _SURFACE_IDENTITY_ATTRS
                         if name not in dataset.ncattrs()]
        if missing_attrs:
            raise OfflineChildContractError(
                f"{path} lacks landuse identity attributes "
                f"{missing_attrs}; category semantics cannot be assumed")
        identity = {
            "MMINLU": str(dataset.getncattr("MMINLU")).strip(),
            "ISWATER": int(dataset.getncattr("ISWATER")),
            "ISLAKE": int(dataset.getncattr("ISLAKE")),
            "ISICE": int(dataset.getncattr("ISICE")),
            "ISOILWATER": int(dataset.getncattr("ISOILWATER"))
            if "ISOILWATER" in dataset.ncattrs() else 14,
        }
        missing = [name for name in _SURFACE_REQUIRED_FIELDS
                   if name not in dataset.variables]
        if missing:
            raise OfflineChildContractError(
                f"{path} lacks required child surface fields {missing}")
        for name in _SURFACE_REQUIRED_FIELDS + _SURFACE_OPTIONAL_FIELDS:
            if name not in dataset.variables:
                continue
            value = np.asarray(dataset.variables[name][:])
            if value.ndim and value.shape[0] == 1 and (
                    dataset.variables[name].dimensions[:1] == ("Time",)):
                value = value[0]
            value = np.ascontiguousarray(value, dtype=np.float32)
            if not np.isfinite(value).all():
                raise OfflineChildContractError(
                    f"{path}/{name} contains NaN or infinity")
            expected_shape = (
                (int(num_soil_layers), int(child_ny), int(child_nx))
                if name in _SURFACE_SOIL_FIELDS
                else (int(child_ny), int(child_nx)))
            if tuple(value.shape) != expected_shape:
                raise OfflineChildContractError(
                    f"{path}/{name} shape {tuple(value.shape)} != "
                    f"{expected_shape}")
            if name in _SURFACE_CATEGORY_FIELDS and not np.array_equal(
                    value, np.rint(value)):
                raise OfflineChildContractError(
                    f"{path}/{name} carries non-integer categories; a "
                    "smoothed/interpolated category field is not a valid "
                    "surface identity")
            fields[name] = value
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = MappingProxyType({
        "path": str(path.resolve()),
        "sha256": digest,
        "identity": dict(identity),
        "carried_fields": tuple(sorted(fields)),
        "policy": "child-grid-surface-source; downscaling replaces "
                  "meteorology, land/soil identity comes from the child's "
                  "own initialization (ndown-equivalent contract)",
    })
    return ChildSurfaceState(
        path=path.resolve(), fields=MappingProxyType(fields),
        identity=MappingProxyType(identity), receipt=receipt)


#: Child-grid surface fields whose value IS a category or a mask, so they
#: are copied from the donor parent cell rather than interpolated: a
#: bilinear land-use index is not a land-use index, and
#: :func:`read_child_surface_state` refuses a non-integer category for
#: exactly that reason.  ``LANDMASK`` rides with them so the child's mask
#: and its land-use come from the SAME parent cell and cannot disagree.
_SURFACE_DONOR_COPY_FIELDS = frozenset(
    {"LU_INDEX", "ISLTYP", "LANDMASK", "SNOWC"})

#: Fields the Registry masks against ``ISICE`` rather than ``ISWATER``
#: (``Registry.EM_COMMON:1417``: ``XICE ... interp_mask_field:lu_index,isice``).
_SURFACE_SEAICE_FIELDS = frozenset({"SEAICE", "XICE"})

#: Not derived from the parent: the child's own latitude/longitude are
#: projection geometry, and the runner already has them exactly -- it
#: SINTs the parent's XLAT/XLONG whenever the surface source carries
#: none (``offline_child_run._initialize_child_physics``).  Interpolating
#: them here would substitute a coarser answer for one already in hand.
_SURFACE_NOT_DERIVED_FIELDS = frozenset({"XLAT", "XLONG"})

#: What the derived child surface IS, in one sentence, for the receipt and
#: for the warning the front door prints.  It is WRF's own answer for a
#: nest that has no ``wrfinput`` of its own (``input_from_file = .false.``,
#: ``med_nest_initial``'s unconditional ``med_interp_domain``,
#: share/mediation_integrate.F:670), run through the Registry-named masked
#: interpolator for the surface/soil family.
DERIVED_CHILD_SURFACE_POLICY = (
    "parent-history-interpolated child surface: land identity and soil "
    "warm start come from the parent's own history frame through WRF's "
    "nest-birth operators -- categories and the landmask by donor-cell "
    "copy, the continuous surface/soil family by interp_mask_field "
    "(Registry.EM_COMMON's masked land interpolator, lu_index/iswater), "
    "which is what WRF does for a nest with input_from_file = .false.  "
    "The child's land identity is therefore its PARENT's, resolved at "
    "the parent's spacing")

#: The fidelity sentence, printed once at the front door.  Named as a
#: cost rather than buried: a child whose coastline is its parent's
#: coastline is a real difference from one built by geogrid at the
#: child's own dx, and a reader of the child's charts must know which
#: they are looking at.
DERIVED_CHILD_SURFACE_CAVEAT = (
    "the child's land-use, soil category and landmask are the PARENT's, "
    "carried down from the parent cell each child column sits in -- "
    "coastlines, lakes and islands the child's spacing could resolve are "
    "not resolved.  Pass --child-surface-from with a child-grid "
    "wrfinput/history file to give the child its own geography instead")


def _donor_copy(field, *, ci, cj):
    """Nearest-donor copy on WRF's masked-interpolator donor cell.

    ``cfld`` is ``(..., ny_parent, nx_parent)``; the result takes the
    value of the coarse cell the child column falls in, so a category or
    a 0/1 mask survives the mapping exactly.
    """
    return np.asarray(field)[..., cj[:, None], ci[None, :]]


def derive_child_surface_from_parent(
        path, *, placement, num_soil_layers: int) -> ChildSurfaceState:
    """Build the child-grid surface state out of the parent's own history.

    THE CLOSED LOOP THIS OPENS (defect #275).  A full-physics child needs
    child-grid land identity and soil state.  Until now the only way to
    supply it was ``--child-surface-from`` pointing at a file on the
    exact child grid, and for a config-driven parent -- the route the
    product steers ERA5 users onto -- no command in the product produced
    one: ``gpuwm run`` writes no ``wrf-native-input/``, ``gpuwm go``
    refuses ``[case_data]`` configs by name, and the prepared-tree route
    needs a front-door manifest the fetch door would only author for one
    source.  So the refusal demanded a file the product could not make,
    and said so only after the parent forecast had been paid for.

    The data was never missing.  The parent's history carries all nine of
    :data:`_SURFACE_REQUIRED_FIELDS` and the landuse identity attributes
    (:data:`gpuwm.io.wrf_output_schema.SURFACE_IDENTITY_OUTPUT_FIELDS`,
    plus the LSM-gated soil family) whenever a land-surface scheme is
    routed -- which a full-physics parent by definition has.  Only the
    GRID was wrong, and putting a parent's surface state on a child grid
    is WRF's own operator, not new science: ``input_from_file = .false.``
    interpolates every field the nest needs from the coarse domain
    (Users' Guide chapter 5; ``med_nest_initial``'s unconditional
    ``med_interp_domain``, share/mediation_integrate.F:670), with the
    surface/soil family going through the Registry's landmask-aware
    ``interp_mask_field`` rather than a plain interpolator.

    WHAT IT COSTS, and why it is a default anyway.  The child inherits
    the parent's land identity at the parent's spacing, which is strictly
    less than a geogrid-built child-grid ``wrfinput`` gives.  It is also
    exactly what a live WRF nest with no input file of its own gets, and
    the product already takes this route for trigger-spawned nests
    (:func:`gpuwm.ingest.nest_spawn_init.spawn_land_state_from_parent`).
    A refusal that names no reachable remedy is not a safeguard; this is
    the reachable remedy, ``--child-surface-from`` stays the
    higher-fidelity one, and the difference is warned about at the front
    door and receipted in the child's report.
    """

    from gpuwm.core.nest_interp import interp_mask_field, mask_donor_index

    path = Path(path)
    child_ny = int(placement.child_ny)
    child_nx = int(placement.child_nx)
    ratio = int(placement.parent_grid_ratio)
    with netcdf_bridge.open_dataset(path) as dataset:
        for name, expected in (("south_north", int(placement.parent_ny)),
                               ("west_east", int(placement.parent_nx))):
            actual = len(dataset.dimensions[name]) \
                if name in dataset.dimensions else None
            if actual != expected:
                raise OfflineChildContractError(
                    f"{path} {name}={actual} is not the parent grid "
                    f"{name}={expected} this placement was built against")
        missing_attrs = [name for name in _SURFACE_IDENTITY_ATTRS
                         if name not in dataset.ncattrs()]
        if missing_attrs:
            raise OfflineChildContractError(
                f"{path} lacks landuse identity attributes "
                f"{missing_attrs}, so the child's category semantics "
                "cannot be read off it; " + CHILD_SURFACE_SOURCE_REMEDY)
        identity = {
            "MMINLU": str(dataset.getncattr("MMINLU")).strip(),
            "ISWATER": int(dataset.getncattr("ISWATER")),
            "ISLAKE": int(dataset.getncattr("ISLAKE")),
            "ISICE": int(dataset.getncattr("ISICE")),
            "ISOILWATER": int(dataset.getncattr("ISOILWATER"))
            if "ISOILWATER" in dataset.ncattrs() else 14,
        }
        missing = [name for name in _SURFACE_REQUIRED_FIELDS
                   if name not in dataset.variables]
        if missing:
            # NAMES THE BREAKAGE: without these the child has no land
            # identity or soil state at all, and the remedy is either a
            # parent history that publishes them (the default inventory
            # of any run with a land-surface scheme) or an explicit
            # child-grid file.
            raise OfflineChildContractError(
                f"{path} does not carry the child surface fields "
                f"{missing}, so a child-grid surface state cannot be "
                "derived from it; re-run the parent with a history "
                "selection that keeps the land-surface inventory, or "
                + CHILD_SURFACE_SOURCE_REMEDY)
        if int(num_soil_layers) > 0:
            soil_dim = ("soil_layers_stag" in dataset.dimensions
                        and len(dataset.dimensions["soil_layers_stag"]))
            if soil_dim != int(num_soil_layers):
                raise OfflineChildContractError(
                    f"{path} soil_layers_stag={soil_dim} does not match "
                    f"the child LSM's {num_soil_layers} soil layers")
        parent_fields: dict[str, np.ndarray] = {}
        for name in (_SURFACE_REQUIRED_FIELDS + _SURFACE_OPTIONAL_FIELDS):
            if name in _SURFACE_NOT_DERIVED_FIELDS:
                continue
            if name not in dataset.variables:
                continue
            value = np.asarray(dataset.variables[name][:])
            if value.ndim and value.shape[0] == 1 and (
                    dataset.variables[name].dimensions[:1] == ("Time",)):
                value = value[0]
            value = np.ascontiguousarray(value, dtype=np.float32)
            if not np.isfinite(value).all():
                raise OfflineChildContractError(
                    f"{path}/{name} contains NaN or infinity")
            parent_fields[name] = value

    ci, _ = mask_donor_index(child_nx, ratio, int(placement.i_parent_start))
    cj, _ = mask_donor_index(child_ny, ratio, int(placement.j_parent_start))
    parent_lu = parent_fields["LU_INDEX"]
    child_lu = _donor_copy(parent_lu, ci=ci, cj=cj)

    fields: dict[str, np.ndarray] = {}
    branch_counts: dict[str, dict[str, int]] = {}
    for name, value in parent_fields.items():
        if name in _SURFACE_DONOR_COPY_FIELDS:
            result = _donor_copy(value, ci=ci, cj=cj)
        else:
            flag = (identity["ISICE"] if name in _SURFACE_SEAICE_FIELDS
                    else identity["ISWATER"])
            result, counts = interp_mask_field(
                value, nri=ratio, nrj=ratio,
                i_parent_start=int(placement.i_parent_start),
                j_parent_start=int(placement.j_parent_start),
                child_landuse=child_lu, parent_landuse=parent_lu,
                flag_category=flag)
            branch_counts[name] = dict(counts)
        result = np.ascontiguousarray(result, dtype=np.float32)
        expected_shape = ((int(num_soil_layers), child_ny, child_nx)
                          if name in _SURFACE_SOIL_FIELDS
                          else (child_ny, child_nx))
        if tuple(result.shape) != expected_shape:
            raise OfflineChildContractError(
                f"{path}/{name} derived shape {tuple(result.shape)} != "
                f"{expected_shape}")
        if name in _SURFACE_CATEGORY_FIELDS and not np.array_equal(
                result, np.rint(result)):
            raise OfflineChildContractError(
                f"{path}/{name} derived non-integer categories; a "
                "smoothed category field is not a valid surface identity")
        if not np.isfinite(result).all():
            raise OfflineChildContractError(
                f"{path}/{name} derived a non-finite child value")
        fields[name] = result

    receipt = MappingProxyType({
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source": "parent-history-interpolated",
        "identity": dict(identity),
        "carried_fields": tuple(sorted(fields)),
        "donor_copied_fields": tuple(
            sorted(set(fields) & _SURFACE_DONOR_COPY_FIELDS)),
        "placement": {
            "parent_grid_ratio": ratio,
            "i_parent_start": int(placement.i_parent_start),
            "j_parent_start": int(placement.j_parent_start),
        },
        "mask_interpolation_branches": {
            name: counts for name, counts in sorted(branch_counts.items())},
        "policy": DERIVED_CHILD_SURFACE_POLICY,
        "caveat": DERIVED_CHILD_SURFACE_CAVEAT,
    })
    return ChildSurfaceState(
        path=path.resolve(), fields=MappingProxyType(fields),
        identity=MappingProxyType(identity), receipt=receipt)


@dataclass(frozen=True)
class OfflineChildPlacement:
    """Fixed child placement inside one archived parent grid.

    Starts retain WRF's one-based namelist semantics.  Construction runs the
    exact SINT stencil coverage gate for mass, x-staggered, and y-staggered
    fields, so an invalid footprint fails before any parent data are read.
    """

    parent_nx: int
    parent_ny: int
    child_nx: int
    child_ny: int
    parent_grid_ratio: int
    i_parent_start: int
    j_parent_start: int

    def __post_init__(self) -> None:
        values = (
            self.parent_nx, self.parent_ny, self.child_nx, self.child_ny,
            self.parent_grid_ratio, self.i_parent_start, self.j_parent_start,
        )
        if any(isinstance(value, bool) or int(value) != value for value in values):
            raise OfflineChildContractError(
                "offline-child placement values must be integers")
        if min(self.parent_nx, self.parent_ny, self.child_nx, self.child_ny) < 1:
            raise OfflineChildContractError(
                "offline-child parent/child extents must be positive")
        if self.parent_grid_ratio < 1:
            raise OfflineChildContractError("parent_grid_ratio must be >= 1")
        # Coverage checks include SINT's +-2 donor stencil.
        for stagger in ("", "x", "y"):
            self.registration(stagger, wrapper="bdy")

    def registration(self, stagger: str, *, wrapper: str):
        return register_nest(
            nri=self.parent_grid_ratio, nrj=self.parent_grid_ratio,
            i_parent_start=self.i_parent_start,
            j_parent_start=self.j_parent_start,
            child_nx=self.child_nx, child_ny=self.child_ny,
            parent_nx=self.parent_nx, parent_ny=self.parent_ny,
            stagger=stagger, wrapper=wrapper,
        )


@dataclass(frozen=True)
class InterpolatedBoundarySnapshot:
    valid_time: datetime
    fields: Mapping[str, np.ndarray]
    receipt: Mapping[str, object]


@dataclass(frozen=True)
class OfflineBoundaryResult:
    boundaries: LateralBoundaries
    frame_receipts: tuple[Mapping[str, object], ...]
    preparation_seconds: float


@dataclass(frozen=True)
class InterpolatedInitialState:
    valid_time: datetime
    fields: Mapping[str, np.ndarray]
    microphysics: Mapping[str, np.ndarray]
    receipt: Mapping[str, object]


def _read_record(dataset, name: str, *, required: bool = True):
    if name not in dataset.variables:
        if required:
            raise OfflineChildContractError(
                f"{Path(dataset.filepath())} is missing required field {name}")
        return None
    value = np.asarray(dataset.variables[name][:])
    if value.ndim and dataset.variables[name].dimensions[:1] == ("Time",):
        if value.shape[0] != 1:
            raise OfflineChildContractError(
                f"{Path(dataset.filepath())}/{name} needs exactly one Time record")
        value = value[0]
    if value.dtype.kind not in "fiu":
        raise OfflineChildContractError(
            f"{Path(dataset.filepath())}/{name} is not numeric")
    value = np.ascontiguousarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise OfflineChildContractError(
            f"{Path(dataset.filepath())}/{name} contains NaN or infinity")
    return value


def _backend_array(value, backend: str):
    if backend == "cpu":
        return np.ascontiguousarray(value, dtype=np.float32)
    if backend != "cuda":
        raise OfflineChildContractError(
            f"offline-child backend must be cpu or cuda, got {backend!r}")
    import cupy as cp
    return cp.asarray(value, dtype=cp.float32)


def _to_host(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value, dtype=np.float32)
    import cupy as cp
    return np.ascontiguousarray(cp.asnumpy(value), dtype=np.float32)


def _transported_source_fields(source_mp_physics: int) -> tuple[str, ...]:
    source_mp = int(source_mp_physics)
    # mp=0 (WRF's passiveqv package, Registry.EM_COMMON:3014) transports
    # qv alone in STOCK WRF, but gpuwm's own mp=0 moist state allocates and
    # advects the warm-rain pair beside it and the ONLINE forcing table
    # (gpuwm/core/preflight.py::nest_field_kinds) forces all three -- so the
    # offline mirror reads all three too.  The contract test pins the two
    # lanes equal, and the lanes are what has to agree here.
    names = ["qv", "qc", "qr"]
    if source_mp in {6, 8, 10, 28}:
        names += ["qi", "qs", "qg"]
    if source_mp == 8:
        names += ["nr", "ni"]
    elif source_mp == 28:
        # Thompson aerosol-aware: classic Thompson's two moments plus the
        # prognostic droplet number and the two aerosol tracers.  Order is
        # nr/ni first so the shared prefix with mp=8 stays visible; the
        # tuple is consumed by name everywhere.  nwfa2d/nifa2d are NOT here
        # -- they are per-domain constants, not transported scalars, and are
        # handled by _AEROSOL_SURFACE_EMISSION_WRF.
        names += ["nr", "ni", "nc", "nwfa", "nifa"]
    elif source_mp == 10:
        names += ["nr", "ni", "ns", "ng"]
    elif source_mp == 50:
        # P3 one-category (Registry.EM_COMMON:3038): moist qv,qc,qr,qi
        # with NO qs and NO qg -- 50 is deliberately absent from the
        # six-species branch above -- plus the two number moments and the
        # prognostic rime mass/volume pair, all four in the same 4-D
        # scalar array.  Spelled in the online forcing table's own order
        # (gpuwm/core/preflight.py::nest_field_kinds, mp==50 arm;
        # gpuwm/core/moist.py::P3_SPECIES), and the offline-inventory
        # contract test pins this tuple against that table so the two
        # lanes cannot drift.
        names += ["qi", "ni", "nr", "qir", "qib"]
    elif source_mp == 18:
        names = list(_NSSL_FIELDS)
    elif source_mp == 9:
        # Milbrandt-Yau (Registry.EM_COMMON:3025): the six-species mass set
        # plus hail mass, and a number moment for every one of the six
        # hydrometeors.  Imported from gpuwm.core.milbrandt2_constants
        # rather than re-spelled, so the offline lane and the online forcing
        # table (gpuwm.core.moist re-exports the same tuple) read one tuple.
        # That module is pure numpy: gpuwm.core.moist imports cupy when it
        # is imported, this function answers for `gpuwm --probe`, the
        # sizing wizard and the offline-child inventory on boxes with no
        # working CuPy, and reading the tuple through moist made every one
        # of them die in an import (tests/test_runplan.py::
        # test_probe_works_on_a_box_whose_cupy_will_not_load,
        # tests/test_offline_child.py on a CPU node).
        from gpuwm.core.milbrandt2_constants import MY2_SPECIES

        names += list(MY2_SPECIES)
    elif source_mp not in {0, 1, 6}:
        # mp=0 (passiveqv), mp=1 (Kessler, Registry.EM_COMMON:3015) and
        # mp=6 (WSM6) exit on the branches above: the first two ARE the
        # qv/qc/qr prefix gpuwm advects for them, and WSM6 is that prefix
        # plus the three frozen masses.  None may fall into this refusal --
        # every member of OFFLINE_CHILD_MP_PHYSICS has to exit this chain
        # with a mapping, which the contract test pins.
        raise OfflineChildContractError(
            _unsupported_parent_clause(source_mp, what="resolved"))
    return tuple(names)


def _resolve_source_physics(
        source_mp_physics: int | None,
        physics_binding: ParentPhysicsBinding | None,
        morr_rimed_ice: int | None,
) -> tuple[int, int | None]:
    if physics_binding is not None:
        if (source_mp_physics is not None
                and int(source_mp_physics) != int(physics_binding.mp_physics)):
            raise OfflineChildContractError(
                "declared source mp_physics conflicts with companion binding")
        source_mp_physics = int(physics_binding.mp_physics)
        if physics_binding.morr_rimed_ice is not None:
            if (morr_rimed_ice is not None
                    and int(morr_rimed_ice)
                    != int(physics_binding.morr_rimed_ice)):
                raise OfflineChildContractError(
                    "declared morr_rimed_ice conflicts with companion binding")
            morr_rimed_ice = int(physics_binding.morr_rimed_ice)
    if source_mp_physics is None:
        raise OfflineChildContractError(
            "source physics must be bound from a companion setup record")
    source_mp_physics = int(source_mp_physics)
    if source_mp_physics not in OFFLINE_CHILD_MP_PHYSICS:
        raise OfflineChildContractError(
            _unsupported_parent_clause(source_mp_physics, what="resolved"))
    return source_mp_physics, morr_rimed_ice


def _raw_parent_state(dataset, source_mp_physics: int):
    raw = {
        name: _read_record(dataset, name)
        for name in (
            "T", "U", "V", "W", "PH", "MU", "MUB", "MAPFAC_M",
            "MAPFAC_U", "MAPFAC_V", "ZNU", "ZNW", "P_TOP",
        )
    }
    moisture = {}
    wrf_mapping = _scheme_wrf_to_state(int(source_mp_physics))
    inverse = {state_name: wrf_name for wrf_name, state_name in wrf_mapping.items()}
    missing = []
    for name in _transported_source_fields(source_mp_physics):
        wrf_name = inverse.get(name)
        if wrf_name is None:
            missing.append(name)
            continue
        moisture[name] = _read_record(dataset, wrf_name)
    if missing:
        raise OfflineChildContractError(
            "history reader has no bound WRF variable mapping for parent "
            f"mp_physics={source_mp_physics} fields {missing}")
    return raw, moisture


def _vertical_coefficients(raw, dataset):
    znw = np.asarray(raw["ZNW"], dtype=np.float64).reshape(-1)
    znu = np.asarray(raw["ZNU"], dtype=np.float64).reshape(-1)
    if znw.size != znu.size + 1 or not np.all(np.diff(znw) < 0.0):
        raise OfflineChildContractError("parent ZNU/ZNW are not one valid eta grid")
    p_top_values = np.asarray(raw["P_TOP"], dtype=np.float64).reshape(-1)
    if p_top_values.size != 1:
        raise OfflineChildContractError("parent P_TOP must be scalar")
    hybrid_opt = int(getattr(dataset, "HYBRID_OPT", 2))
    etac = float(getattr(dataset, "ETAC", 0.2))
    from gpuwm.core.constants import P0
    coeffs = compute_hybrid_coeffs(
        znw, hybrid_opt, etac, float(P0), float(p_top_values[0]))
    return coeffs, hybrid_opt, etac, float(p_top_values[0])



#: Fields on the child's own ladder are rebuilt, not interpolated: ``PB`` is
#: an exact function of the ladder and ``MUB``, and both geopotentials come
#: from the discrete hydrostatic recurrence.  Everything else is rebinned.
_LADDER_INDEPENDENT_INITIAL_FIELDS = frozenset({
    "MU", "MUB", "HGT", "PSFC", "MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "F", "E",
    "SINALPHA", "COSALPHA", "XLAT", "XLONG", "XLAT_U", "XLONG_U", "XLAT_V",
    "XLONG_V",
})


def resolve_child_ladder(child_eta_levels, *, nz: int | None = None):
    """Validate one declared child ladder, or ``None`` for 'inherit'.

    ``None`` is the shipped trajectory: the child takes its parent's ladder
    verbatim and nothing in this module remaps anything.
    """

    if child_eta_levels is None:
        return None
    znw = np.asarray(child_eta_levels, dtype=np.float64).reshape(-1)
    if nz is not None and znw.size != int(nz) + 1:
        raise OfflineChildContractError(
            f"child ladder has {znw.size} interfaces but the child config "
            f"declares nz={nz}, which needs {int(nz) + 1}: the ladder and "
            "the level count have to describe one grid")
    if znw.size < 2 or znw[0] != 1.0 or znw[-1] != 0.0 or not np.all(
            np.diff(znw) < 0.0):
        raise OfflineChildContractError(
            "child eta_levels must decrease strictly from 1.0 at the surface "
            f"to 0.0 at the model top, got {znw[0]!r} .. {znw[-1]!r}: a "
            "non-monotone ladder folds the coordinate and the reference dry "
            "pressure stops decreasing with height")
    return znw


def _mass_edges(znw, mu, hybrid_opt, etac, p_top):
    return dry_mass_edges(np.asarray(znw, dtype=np.float64),
                          hybrid_opt=int(hybrid_opt), etac=float(etac),
                          p_top=float(p_top),
                          mu=np.asarray(mu, dtype=np.float64))


def _remap_geopotential(phi, src_edges, dst_edges):
    """Remap ``alt = dphi/dm`` and rebuild, rather than interpolating PHI.

    PHI is a coordinate quantity, not an extensive one.  Rebuilding it from
    the remapped ``alt`` through the same recurrence
    ``gpuwm/core/grid.py`` :: ``make_base_state`` uses means the child's own
    ``update_diagnostics`` recovers exactly the ``alt`` that was remapped, so
    the state satisfies the dycore's discrete hydrostatic relation instead of
    merely coming close to it.  It also conserves the column's geopotential
    DEPTH exactly, so the child's model top sits where the parent's did.
    """

    phi = np.asarray(phi, dtype=np.float64)
    alt = geopotential_thickness_per_mass(phi, src_edges)
    return rebuild_geopotential(
        phi[0], remap_layer_means(src_edges, alt, dst_edges), dst_edges)


def _remap_initial_state_to_child_ladder(
        fields, source_mixing, *, parent_znw, child_znw, hybrid_opt, etac,
        p_top):
    """Move one horizontally-SINTed parent state onto the child's ladder.

    Runs once, on the host, in float64, after the horizontal interpolation
    and before anything is uploaded.  The integration loop never sees it.

    Total fields are remapped and the perturbations re-derived against the
    NEW base state: a perturbation is bookkeeping relative to a base that is
    itself changing here, so remapping one directly would carry the parent's
    base into the child's.
    """

    receipts = []
    host = {name: np.asarray(value, dtype=np.float64)
            for name, value in fields.items()}
    mub = host["MUB"]
    mu_total = mub + host["MU"]
    hyc = compute_hybrid_coeffs(np.asarray(child_znw, dtype=np.float64),
                               int(hybrid_opt), float(etac), float(c.P0),
                               float(p_top))

    base_src = _mass_edges(parent_znw, mub, hybrid_opt, etac, p_top)
    base_dst = _mass_edges(child_znw, mub, hybrid_opt, etac, p_top)
    tot_src = _mass_edges(parent_znw, mu_total, hybrid_opt, etac, p_top)
    tot_dst = _mass_edges(child_znw, mu_total, hybrid_opt, etac, p_top)

    out = {name: value for name, value in fields.items()
           if name in _LADDER_INDEPENDENT_INITIAL_FIELDS}

    # PB is an exact function of the ladder and MUB -- the same expression
    # make_base_state evaluates -- so it is RECOMPUTED, never rebinned.
    out["PB"] = (hyc["c3h"][:, None, None] * mub[None]
                 + hyc["c4h"][:, None, None] + float(p_top))
    out["PHB"] = _remap_geopotential(host["PHB"], base_src, base_dst)

    # Total geopotential, then the perturbation against the NEW base.
    phi_total = _remap_geopotential(host["PHB"] + host["PH"], tot_src, tot_dst)
    out["PH"] = phi_total - out["PHB"]

    # Total potential temperature, then the perturbation against 300 K (the
    # child's own base theta is derived downstream from the new PB/PHB).
    theta = remap_layer_means(tot_src, host["T"] + 300.0, tot_dst)
    receipts.append(remap_receipt("theta", tot_src, host["T"] + 300.0,
                                  tot_dst, theta))
    out["T"] = theta - 300.0

    out["W"] = remap_interface_values(tot_src, host["W"], tot_dst)
    if "P" in host:
        out["P"] = remap_layer_means(tot_src, host["P"], tot_dst)

    # Momentum carries the mass at its own faces, the same convention
    # _couple_parent uses for the boundary route.
    for name, faces in (("U", mu_at_u_faces), ("V", mu_at_v_faces)):
        if name not in host:
            continue
        mu_face = _edge_pinned(np.asarray(faces(mu_total), dtype=np.float64),
                               mu_total, axis=1 if name == "U" else 0)
        src = _mass_edges(parent_znw, mu_face, hybrid_opt, etac, p_top)
        dst = _mass_edges(child_znw, mu_face, hybrid_opt, etac, p_top)
        out[name] = remap_layer_means(src, host[name], dst)

    child_mixing = {}
    for name, value in source_mixing.items():
        array = np.asarray(value, dtype=np.float64)
        remapped = remap_layer_means(tot_src, array, tot_dst)
        receipts.append(remap_receipt(name, tot_src, array, tot_dst, remapped))
        child_mixing[name] = remapped

    znu = 0.5 * (np.asarray(child_znw)[:-1] + np.asarray(child_znw)[1:])
    out["ZNW"] = np.asarray(child_znw, dtype=np.float64)
    out["ZNU"] = znu
    out["P_TOP"] = np.asarray([float(p_top)], dtype=np.float64)

    # Anything not named above rides through UNCHANGED, which is correct only
    # for a field that does not live on the ladder.  A field that does would
    # otherwise reach the child at the PARENT's level count inside a dict
    # whose other members are on the child's -- a mixed-nz state that the
    # shape checks downstream would not all catch.  Refused by name instead:
    # a 3-D field added to _INITIAL_CORE_FIELDS later has to be given a
    # weight here, and the refusal says so.
    parent_levels = {int(np.asarray(parent_znw).size),
                     int(np.asarray(parent_znw).size) - 1}
    for name, value in fields.items():
        if name in out:
            continue
        array = np.asarray(value)
        if array.ndim >= 3 and int(array.shape[0]) in parent_levels:
            raise OfflineChildContractError(
                f"{name} has {array.shape[0]} levels on the parent's ladder "
                "and no remap weight in "
                "_remap_initial_state_to_child_ladder, so a child on its own "
                "ladder would receive it at the parent's level count while "
                "every other field arrived at the child's.  Give it a weight "
                "(mass for a layer field, interface for a staggered one) or "
                "add it to _LADDER_INDEPENDENT_INITIAL_FIELDS if it does not "
                "live on the ladder.")
        out[name] = value
    return out, child_mixing, receipts

_INITIAL_FIELD_STAGGER = MappingProxyType({
    "U": "x", "V": "y", "MAPFAC_U": "x", "MAPFAC_V": "y",
    "XLAT_U": "x", "XLONG_U": "x", "XLAT_V": "y", "XLONG_V": "y",
})
_INITIAL_CORE_FIELDS = (
    "T", "U", "V", "W", "PH", "MU", "PHB", "MUB", "HGT", "P", "PB",
    "PSFC", "MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "F", "E",
    "SINALPHA", "COSALPHA", "XLAT", "XLONG",
)


def interpolate_parent_initial_state(
        path: str | Path, placement: OfflineChildPlacement, *,
        source_mp_physics: int | None = None,
        physics_binding: ParentPhysicsBinding | None = None,
        target_mp_physics: int | None = None,
        morr_rimed_ice: int | None = None, backend: str = "cpu",
        child_eta_levels=None,
        diagnose_missing: Callable[[Mapping[str, np.ndarray], Sequence[str]],
                                   Mapping[str, np.ndarray]] | None = None,
) -> InterpolatedInitialState:
    """SINT one archived parent state into a standalone child cold start.

    This first executable mode deliberately inherits SINT parent terrain and
    base state.  A later static-geography join may replace/blend those fields,
    but it must run the existing ``blend_terrain``/``adjust_tempqv`` contract;
    this function never claims a high-resolution terrain adjustment happened.
    """

    source_mp_physics, morr_rimed_ice = _resolve_source_physics(
        source_mp_physics, physics_binding, morr_rimed_ice)
    backend = str(backend).strip().lower()
    target_mp = int(source_mp_physics if target_mp_physics is None
                    else target_mp_physics)
    info = inspect_parent_history_frame(path, source_mp_physics=source_mp_physics)
    expected = (placement.parent_ny, placement.parent_nx)
    actual = (info.dimensions["south_north"], info.dimensions["west_east"])
    if actual != expected:
        raise OfflineChildContractError(
            f"parent history mass grid {actual} != placement parent grid {expected}")
    started = time.perf_counter()
    initial_fields = _INITIAL_CORE_FIELDS
    if int(source_mp_physics) == 28:
        # Read them as REQUIRED.  A missing QNWFA2D is not a stream a child
        # may be silently built from: nothing downstream re-derives it (see
        # _AEROSOL_SURFACE_EMISSION_WRF), so tolerating the absence would
        # produce a finite, bounded, aerosol-emission-free forecast.
        initial_fields = initial_fields + _AEROSOL_SURFACE_EMISSION_WRF
    with netcdf_bridge.open_dataset(path) as dataset:
        raw, moisture = _raw_parent_state(dataset, int(source_mp_physics))
        for name in initial_fields:
            if name not in raw:
                raw[name] = _read_record(dataset, name)
        coeffs, hybrid_opt, etac, p_top = _vertical_coefficients(raw, dataset)
    registrations = {
        "": placement.registration("", wrapper="interp"),
        "x": placement.registration("x", wrapper="interp"),
        "y": placement.registration("y", wrapper="interp"),
    }
    constants = {"ZNU", "ZNW", "P_TOP"}
    interpolated = {}
    for name in initial_fields:
        value = _backend_array(raw[name], backend)
        interpolated[name] = sint(
            value, registrations[_INITIAL_FIELD_STAGGER.get(name, "")])
    source_mixing = {
        name: sint(_backend_array(value, backend), registrations[""])
        for name, value in moisture.items()
    }
    # The number moments carry magnitudes of 1e3..1e9 per kilogram, so a
    # float32 SINT can round one across zero.  Applied BEFORE any consumer:
    # the mp18 mapping below refuses a negative source moment, and the
    # engine's radiation gate refuses a negative nr at the first radiative
    # call.  Both of those are correct; the artefact is what has to go.
    initial_clamp = clamp_sint_undershoot_mapping(source_mixing)
    conversion_receipt = None
    _refuse_unbuilt_p3_offline_edge(int(source_mp_physics), target_mp)
    if target_mp == 18:
        mapped, conversion_receipt = map_microphysics_to_nssl18(
            {name: _to_host(value) for name, value in source_mixing.items()},
            source_mp_physics=int(source_mp_physics),
            morr_rimed_ice=morr_rimed_ice,
            diagnose_missing=diagnose_missing,
        )
        source_mixing = mapped
    elif (target_mp != int(source_mp_physics)
            and target_mp in _CROSS_SCHEME_REFUSED_MP_PHYSICS):
        raise OfflineChildContractError(
            "offline cross-physics initialization of a child at "
            f"{_cross_scheme_refusal_clause(target_mp)} from an "
            f"mp_physics={int(source_mp_physics)} parent is REFUSED: no "
            "cross-scheme entry closure for those moments has been measured, "
            "and the online nest lane refuses the same edge "
            "(gpuwm/core/microphysics_transition.py::"
            f"UNVALIDATED_MIXED_EDGE_SELECTORS).  Same-scheme {target_mp} -> "
            f"{target_mp} downscaling is supported")
    elif target_mp != int(source_mp_physics):
        raise OfflineChildContractError(
            "cross-physics offline initialization currently targets NSSL mp18")
    fields = {name: _to_host(value) for name, value in interpolated.items()}
    fields.update({name: np.array(raw[name], copy=True, dtype=np.float32)
                   for name in constants})
    # The child's own ladder, when it declares one.  A child that declares
    # nothing never reaches this branch and its state is bitwise what it was
    # before per-domain ladders existed.
    child_znw = resolve_child_ladder(child_eta_levels)
    remap_receipts = ()
    if child_znw is not None:
        parent_znw = np.asarray(raw["ZNW"], dtype=np.float64).reshape(-1)
        fields, source_mixing, remap_receipts = (
            _remap_initial_state_to_child_ladder(
                fields, {name: _to_host(value)
                         for name, value in source_mixing.items()},
                parent_znw=parent_znw, child_znw=child_znw,
                hybrid_opt=hybrid_opt, etac=etac, p_top=p_top))
        # PHB stays float64.  The remap above produced the child's base
        # geopotential in float64 on a ladder the parent never had, and
        # DomainState.set_base_geopotential subtracts the float32 store
        # from it to build dphb_resid -- the correction that stops the
        # diagnosed pressure degrading as 1/dz.  Rounding it here made
        # that subtraction identically zero, so the FP32 EOS remedy was
        # OFF on precisely the deep-column route it exists for, while
        # being on everywhere else.  Every other consumer casts to
        # float32 explicitly at its own use (``assign`` below), so this
        # widens nothing downstream.  On the NO-ladder route this block
        # does not run at all and PHB arrives float32 from the archive,
        # where the information genuinely does not exist.
        fields = {name: value if name == "PHB"
                  else np.asarray(value, dtype=np.float32)
                  for name, value in fields.items()}
        fields["PHB"] = np.ascontiguousarray(fields["PHB"], dtype=np.float64)
    receipt = MappingProxyType({
        "path": str(Path(path).resolve()),
        "valid_time": info.valid_time.isoformat(),
        "geometry_sha256": info.geometry_sha256,
        "source_mp_physics": int(source_mp_physics),
        "advisory_inferred_mp_physics": info.inferred_mp_physics,
        "source_physics_binding": (
            None if physics_binding is None else dict(physics_binding.receipt())),
        "target_mp_physics": target_mp,
        "backend": backend,
        "positive_definite_clamp": initial_clamp,
        "terrain_policy": "sint-parent-inherited",
        "spinup_policy": (
            "new-standalone-child; source held physics tendencies and "
            "scheduler state are not inherited"),
        "hybrid_opt": hybrid_opt,
        "etac": etac,
        "p_top": p_top,
        # None when the child inherited its parent's ladder.  Otherwise the
        # measured conservation of every field that was rebinned: a remap
        # that quietly failed to conserve must not look like one that did.
        "vertical_remap": None if child_znw is None else {
            "source_levels": int(np.asarray(raw["ZNW"]).size - 1),
            "target_levels": int(child_znw.size - 1),
            "child_eta_levels": tuple(float(v) for v in child_znw),
            "fields": tuple(item.summary() for item in remap_receipts),
            "max_relative_drift": max(
                (item.max_relative_drift for item in remap_receipts),
                default=0.0),
        },
        "conversion": None if conversion_receipt is None else dict(conversion_receipt),
        "seconds": time.perf_counter() - started,
    })
    return InterpolatedInitialState(
        info.valid_time, MappingProxyType(fields),
        MappingProxyType({name: _to_host(value)
                          for name, value in source_mixing.items()}), receipt)


def _child_base_geopotential(phb_parent, mub, *, parent_znw, child_znw,
                             hybrid_opt, etac, p_top):
    """The child ladder's base geopotential, from the parent's own PHB.

    Shared by the initial-state and lateral-boundary routes so the boundary
    strips are relative to exactly the base state the domain was built on; a
    second, subtly different reconstruction here would put a step in the
    geopotential at the edge of the relaxation zone.
    """

    return _remap_geopotential(
        np.asarray(phb_parent, dtype=np.float64),
        _mass_edges(parent_znw, mub, hybrid_opt, etac, p_top),
        _mass_edges(child_znw, mub, hybrid_opt, etac, p_top))


def _remap_boundary_snapshot_to_child_ladder(
        interpolated, *, child_mub, child_phb, parent_znw, child_znw,
        hybrid_opt, etac, p_top, moisture_names):
    """Move one SINTed, COUPLED boundary strip onto the child's ladder.

    The strips arrive coupled by the parent ladder's ``chm``/``chf``
    (:func:`_couple_parent`).  Coupling is a per-layer mass weight, so a
    coupled field is not rebinnable as it stands: it is uncoupled on the
    child with the parent ladder's weight, remapped, and recoupled with the
    child ladder's.  Uncoupling against a weight built from the child's own
    ``mu`` is the convention this module already uses for the cross-physics
    boundary edge, not a second one invented here.
    """

    receipts = []
    child_mu = np.asarray(child_mub, dtype=np.float64) + np.asarray(
        interpolated["mu"][0], dtype=np.float64)

    def coeffs_for(znw):
        return compute_hybrid_coeffs(np.asarray(znw, dtype=np.float64),
                                     int(hybrid_opt), float(etac),
                                     float(c.P0), float(p_top))

    src_c, dst_c = coeffs_for(parent_znw), coeffs_for(child_znw)
    faces = {
        "": child_mu,
        "x": _edge_pinned(np.asarray(mu_at_u_faces(child_mu),
                                     dtype=np.float64), child_mu, axis=1),
        "y": _edge_pinned(np.asarray(mu_at_v_faces(child_mu),
                                     dtype=np.float64), child_mu, axis=0),
    }
    stagger = {"u": "x", "v": "y"}
    edges = {key: (_mass_edges(parent_znw, value, hybrid_opt, etac, p_top),
                   _mass_edges(child_znw, value, hybrid_opt, etac, p_top))
             for key, value in faces.items()}

    def half_weight(co, mu_face):
        return (co["c1h"][:, None, None] * mu_face[None]
                + co["c2h"][:, None, None])

    def full_weight(co, mu_face):
        return (co["c1f"][:, None, None] * mu_face[None]
                + co["c2f"][:, None, None])

    out = {"mu": interpolated["mu"]}
    for name, value in interpolated.items():
        if name == "mu":
            continue
        key = stagger.get(name, "")
        mu_face = faces[key]
        src_edges, dst_edges = edges[key]
        array = np.asarray(_to_host(value), dtype=np.float64)
        if name in ("w", "phi"):
            plain = array / full_weight(src_c, mu_face)
            if name == "phi":
                # Total geopotential, remapped through alt = dphi/dm, then
                # made a perturbation against the CHILD's base again.
                total = _remap_geopotential(
                    np.asarray(child_phb, dtype=np.float64) + plain,
                    src_edges, dst_edges)
                child_base = _child_base_geopotential(
                    child_phb, np.asarray(child_mub, dtype=np.float64),
                    parent_znw=parent_znw, child_znw=child_znw,
                    hybrid_opt=hybrid_opt, etac=etac, p_top=p_top)
                remapped = total - child_base
            else:
                remapped = remap_interface_values(src_edges, plain, dst_edges)
            out[name] = remapped * full_weight(dst_c, mu_face)
            continue
        plain = array / half_weight(src_c, mu_face)
        remapped = remap_layer_means(src_edges, plain, dst_edges)
        if name in moisture_names or name == "theta":
            receipts.append(
                remap_receipt(name, src_edges, plain, dst_edges, remapped))
        out[name] = remapped * half_weight(dst_c, mu_face)
    return out, receipts


def _edge_pinned(face, centre, *, axis):
    """``mu`` at a staggered face, with the outer faces pinned to the centre.

    The same convention :func:`_couple_parent` uses; kept in one place so the
    two routes cannot drift apart.
    """

    face = np.array(face, dtype=np.float64, copy=True)
    if axis == 1:
        face[:, 0] = centre[:, 0]
        face[:, -1] = centre[:, -1]
    else:
        face[0, :] = centre[0, :]
        face[-1, :] = centre[-1, :]
    return face

def _base_from_interpolated_initial(initial: InterpolatedInitialState,
                                    cfg):
    fields = initial.fields
    znw = np.asarray(fields["ZNW"], dtype=np.float64).reshape(-1)
    coord = make_vertical_coord(
        cfg.nz, hybrid_opt=int(initial.receipt["hybrid_opt"]),
        etac=float(initial.receipt["etac"]), eta_levels=znw)
    p_top = float(np.asarray(fields["P_TOP"]).reshape(-1)[0])
    finalize_vertical_coord(coord, p_top)
    mub = np.asarray(fields["MUB"], dtype=np.float64)
    pb = np.asarray(fields["PB"], dtype=np.float64)
    phb = np.asarray(fields["PHB"], dtype=np.float64)
    if pb.shape != (cfg.nz, cfg.ny, cfg.nx):
        raise OfflineChildContractError(
            f"interpolated PB shape {pb.shape} does not match child")
    if phb.shape != (cfg.nz + 1, cfg.ny, cfg.nx):
        raise OfflineChildContractError(
            f"interpolated PHB shape {phb.shape} does not match child")
    delta_phi = phb[1:] - phb[:-1]
    if cfg.hypsometric_opt == 1:
        denominator = (-coord.dnw[:, None, None]
                       * (coord.c1h[:, None, None] * mub[None]
                          + coord.c2h[:, None, None]))
    elif cfg.hypsometric_opt == 2:
        pfu = (coord.c3f[1:, None, None] * mub[None]
               + coord.c4f[1:, None, None] + p_top)
        pfd = (coord.c3f[:-1, None, None] * mub[None]
               + coord.c4f[:-1, None, None] + p_top)
        phm = (coord.c3h[:, None, None] * mub[None]
               + coord.c4h[:, None, None] + p_top)
        denominator = phm * np.log(pfd / pfu)
    else:
        raise OfflineChildContractError(
            f"unsupported hypsometric_opt={cfg.hypsometric_opt}")
    if np.any(denominator <= 0.0):
        raise OfflineChildContractError(
            "interpolated parent base has non-positive hydrostatic increments")
    alb = delta_phi / denominator
    if not np.isfinite(alb).all() or np.any(alb <= 0.0):
        raise OfflineChildContractError(
            "interpolated parent PHB/PB imply invalid base inverse density")
    from gpuwm.core import constants as c
    thb = alb * pb / (c.RD * (pb / c.P0) ** c.RCP)
    if not np.isfinite(thb).all() or np.any(thb <= 0.0):
        raise OfflineChildContractError(
            "interpolated parent base implies invalid base potential temperature")
    return coord, BaseState(
        mub=mub, p_top=p_top, pb=pb, alb=alb, thb=thb, phb=phb,
        terrain_z=np.asarray(fields["HGT"], dtype=np.float64))


def _require_prepared_child_ladder(initial, cfg) -> None:
    """The prepared state and the child config must name ONE ladder.

    A prepared state carries the ladder it was remapped onto.  If the config
    handed to this function names a different one, the arrays would be built
    against a coordinate the state was never interpolated to -- the level
    counts might even agree while the interfaces sit elsewhere, so the shape
    checks downstream would pass and the child would integrate a state whose
    layers are not where its coordinate says they are.
    """

    prepared = initial.receipt.get("vertical_remap")
    declared = None if cfg.eta_levels is None else tuple(
        float(value) for value in cfg.eta_levels)
    if prepared is None:
        if declared is not None:
            raise OfflineChildContractError(
                f"child config declares its own {len(declared) - 1}-level "
                "eta ladder but the prepared state was built on the parent's "
                "ladder: pass child_eta_levels to "
                "interpolate_parent_initial_state so the state is remapped "
                "onto the ladder the child will integrate on")
        return
    if declared is None:
        raise OfflineChildContractError(
            f"prepared state was remapped onto a "
            f"{prepared['target_levels']}-level child ladder but the child "
            "config declares no eta_levels: the config has to carry the "
            "ladder the state was built for")
    if declared != tuple(prepared["child_eta_levels"]):
        raise OfflineChildContractError(
            f"child config eta_levels ({len(declared) - 1} levels) is not "
            f"the ladder the state was prepared on "
            f"({prepared['target_levels']} levels): the state would be "
            "loaded against a coordinate it was never remapped to")


def _require_runnable_child_radiation(cfg, p_top: float) -> None:
    """Refuse a child ladder this domain's own radiation cannot run.

    ``RunConfig`` carries no model-top pressure, so
    ``validate_run_config`` reaches
    ``validate_resolved_physics_vertical_levels`` with ``p_top=None`` and the
    radiation cap-layer arithmetic is skipped (gpuwm/physics_compat.py, the
    "RunConfig-only checks" branch).  That was unreachable while a child's nz
    was pinned to its parent's and real parents run ~50 levels; a child that
    may now name its own deeper ladder can walk straight into it, and the run
    would die at the FIRST radiative call -- after the fetch, the SINT, the
    remap and the whole preparation had been paid for.  The parent archive
    knows the model top, so the check runs here with it.
    """

    from gpuwm.physics_compat import (
        PhysicsVerticalPreflightError,
        validate_resolved_physics_vertical_levels,
    )

    try:
        validate_resolved_physics_vertical_levels(cfg, p_top=float(p_top))
    except PhysicsVerticalPreflightError as exc:
        raise OfflineChildContractError(
            f"child nz={cfg.nz} at the parent's p_top={float(p_top):g} Pa "
            f"exceeds a radiation adapter's layer ceiling: {exc}") from exc

def build_offline_child_domain_state(
        initial: InterpolatedInitialState, cfg, *, array_module=None):
    """Upload an interpolated parent-only cold start into ``DomainState``.

    The returned state has complete dynamics, moisture, map factors, and RK
    time-t copies. Physics and lateral tables are attached by their existing
    production APIs so callers can select the new child physics explicitly.
    """

    if not cfg.specified or cfg.nested:
        raise OfflineChildContractError(
            "standalone offline child requires specified=True and nested=False")
    if initial.receipt.get("source_physics_binding") is None:
        raise OfflineChildContractError(
            "standalone offline child requires authoritative parent physics "
            "evidence from a setup/namelist/restart companion")
    if not cfg.moist:
        raise OfflineChildContractError(
            "offline child with microphysics requires moist=True")
    if not cfg.terrain_opt:
        raise OfflineChildContractError(
            "parent-inherited offline child base requires terrain_opt != 0")
    if int(cfg.hybrid_opt) != int(initial.receipt["hybrid_opt"]):
        raise OfflineChildContractError(
            f"child hybrid_opt={cfg.hybrid_opt} differs from archived parent "
            f"{initial.receipt['hybrid_opt']}")
    # History files persist ETAC as FP32, while RunConfig retains the user's
    # decimal as a Python float.  Compare at the precision of the archived
    # evidence so an exact FP32 round-trip (for example 0.2) is accepted.
    if not np.isclose(
            float(cfg.etac), float(initial.receipt["etac"]),
            rtol=0.0,
            atol=abs(float(np.spacing(np.float32(initial.receipt["etac"]))))):
        raise OfflineChildContractError(
            f"child etac={cfg.etac} differs from archived parent "
            f"{initial.receipt['etac']}")
    target_mp = int(initial.receipt["target_mp_physics"])
    if int(cfg.mp_physics) != target_mp:
        raise OfflineChildContractError(
            f"child cfg mp_physics={cfg.mp_physics} != prepared target {target_mp}")
    _require_prepared_child_ladder(initial, cfg)
    _require_runnable_child_radiation(cfg, float(initial.receipt["p_top"]))
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.state import DomainState
    coord, base = _base_from_interpolated_initial(initial, cfg)
    state = DomainState(cfg, array_module=array_module)
    state.load_base(coord, base)
    xp = np if array_module is np else __import__("cupy")

    def assign(name, value):
        target = getattr(state, name)
        host = np.asarray(value, dtype=np.float32)
        if tuple(target.shape) != tuple(host.shape):
            raise OfflineChildContractError(
                f"initial {name} shape {host.shape} != state {target.shape}")
        target[...] = xp.asarray(host, dtype=xp.float32)

    for state_name, wrf_name in (
            ("u", "U"), ("v", "V"), ("w", "W"),
            ("php", "PH"), ("mup", "MU")):
        assign(state_name, initial.fields[wrf_name])
    total_theta = np.asarray(initial.fields["T"], dtype=np.float32) + np.float32(300.0)
    state.thp[...] = xp.asarray(
        total_theta - np.asarray(base.thb, dtype=np.float32), dtype=xp.float32)
    state.set_map_coriolis(
        initial.fields["MAPFAC_M"], initial.fields["MAPFAC_U"],
        initial.fields["MAPFAC_V"], initial.fields["F"], initial.fields["E"],
        sina=initial.fields["SINALPHA"], cosa=initial.fields["COSALPHA"])
    for name, value in initial.microphysics.items():
        target = getattr(state, name, None)
        if target is None:
            raise OfflineChildContractError(
                f"child DomainState has no target microphysics field {name!r}")
        assign(name, value)
    if target_mp == 28:
        # The two per-domain surface aerosol emission constants.  They are
        # NOT in ``initial.microphysics`` because they are not transported
        # scalars; they arrive through ``initial.fields`` alongside the
        # geometry.  See _AEROSOL_SURFACE_EMISSION_WRF for why the child
        # cannot re-derive them.
        for wrf_name, state_name in _AEROSOL_SURFACE_EMISSION_STATE.items():
            if getattr(state, state_name, None) is None:
                raise OfflineChildContractError(
                    "mp_physics=28 child DomainState has no surface aerosol "
                    f"emission field {state_name!r}")
            if wrf_name not in initial.fields:
                raise OfflineChildContractError(
                    "mp_physics=28 offline child requires the parent's "
                    f"{wrf_name}; without it the child would integrate with "
                    "zero surface aerosol emission and nothing would raise")
            assign(state_name, initial.fields[wrf_name])
    # The online nest lane's own seeding table
    # (gpuwm/ingest/nest_init.py::RK_TIME_T_SEED_PAIRS), not a local copy.
    # The local copy this replaced had already drifted: it stopped at the
    # Morrison moments, so an offline NSSL child's ten moment seeds
    # (qh0/qndrop0/...) and an offline P3 child's rime seeds (qir0/qib0)
    # would have started the first substep at zero while the current
    # fields carried the parent -- the exact twelve-of-fourteen defect the
    # shared table's comment records for the online lane.  Every pair is
    # None-guarded, so schemes without a field skip it, as before.
    seed_rk_time_t_copies(state)
    update_diagnostics(state, cfg.hypsometric_opt)
    return state


def _couple_parent(raw, moisture, coeffs, backend: str):
    arrays = {name: _backend_array(value, backend) for name, value in raw.items()}
    scalars = {name: _backend_array(value, backend)
               for name, value in moisture.items()}
    total_mu = arrays["MUB"] + arrays["MU"]
    mux = mu_at_u_faces(total_mu)
    muy = mu_at_v_faces(total_mu)
    mux[:, 0] = total_mu[:, 0]
    mux[:, -1] = total_mu[:, -1]
    muy[0, :] = total_mu[0, :]
    muy[-1, :] = total_mu[-1, :]
    xp = np if backend == "cpu" else __import__("cupy")
    c1h = xp.asarray(coeffs["c1h"], dtype=xp.float32)[:, None, None]
    c2h = xp.asarray(coeffs["c2h"], dtype=xp.float32)[:, None, None]
    c1f = xp.asarray(coeffs["c1f"], dtype=xp.float32)[:, None, None]
    c2f = xp.asarray(coeffs["c2f"], dtype=xp.float32)[:, None, None]
    chm = c1h * total_mu[None] + c2h
    chf = c1f * total_mu[None] + c2f
    result = {
        "u": (c1h * mux[None] + c2h) * arrays["U"] / arrays["MAPFAC_U"][None],
        "v": (c1h * muy[None] + c2h) * arrays["V"] / arrays["MAPFAC_V"][None],
        "w": chf * arrays["W"] / arrays["MAPFAC_M"][None],
        "theta": chm * arrays["T"],
        "phi": chf * arrays["PH"],
        "mu": arrays["MU"][None],
    }
    result.update({name: chm * value for name, value in scalars.items()})
    return result, arrays, chm


def interpolate_parent_boundary_snapshot(
        path: str | Path, placement: OfflineChildPlacement, *,
        source_mp_physics: int | None = None,
        physics_binding: ParentPhysicsBinding | None = None,
        target_mp_physics: int | None = None,
        morr_rimed_ice: int | None = None, backend: str = "cpu",
        child_eta_levels=None,
        diagnose_missing: Callable[[Mapping[str, np.ndarray], Sequence[str]],
                                   Mapping[str, np.ndarray]] | None = None,
) -> InterpolatedBoundarySnapshot:
    """Conservatively SINT one archived parent state onto a child frame.

    Dynamics and source scalars are coupled on the parent before SINT, matching
    online ``bdy_interp1`` spatial semantics.  When target physics differs,
    coupled source scalars are uncoupled on the child, converted there, and
    recoupled in the target inventory.
    """

    source_mp_physics, morr_rimed_ice = _resolve_source_physics(
        source_mp_physics, physics_binding, morr_rimed_ice)
    backend = str(backend).strip().lower()
    info = inspect_parent_history_frame(path, source_mp_physics=source_mp_physics)
    expected = (placement.parent_ny, placement.parent_nx)
    actual = (info.dimensions["south_north"], info.dimensions["west_east"])
    if actual != expected:
        raise OfflineChildContractError(
            f"parent history mass grid {actual} != placement parent grid {expected}")
    target_mp = int(source_mp_physics if target_mp_physics is None
                    else target_mp_physics)
    started = time.perf_counter()
    child_znw = resolve_child_ladder(child_eta_levels)
    with netcdf_bridge.open_dataset(path) as dataset:
        raw, moisture = _raw_parent_state(dataset, int(source_mp_physics))
        coeffs, hybrid_opt, etac, p_top = _vertical_coefficients(raw, dataset)
        if child_znw is not None:
            # Needed only to make the child's geopotential a perturbation
            # against its OWN base; read here rather than in
            # _raw_parent_state so a child that inherits its parent's ladder
            # still requires exactly the variables it always did.
            raw["PHB"] = _read_record(dataset, "PHB")
    coupled, raw_device, parent_chm = _couple_parent(
        raw, moisture, coeffs, backend)
    registrations = {
        "": placement.registration("", wrapper="bdy"),
        "x": placement.registration("x", wrapper="bdy"),
        "y": placement.registration("y", wrapper="bdy"),
    }
    stagger = {"u": "x", "v": "y"}
    interpolated = {
        name: sint(value, registrations[stagger.get(name, "")])
        for name, value in coupled.items()
    }
    # Same fix-up as the initial state, on the COUPLED moments.  Coupling
    # scales the field and its own peak by the same chm, so the relative
    # tolerance carries over untouched, but the ABSOLUTE floor is in the
    # field's units and has to be carried across with it -- hence
    # floor_scale.  The parent's chm is the right scale for it: it differs
    # from the child's by interpolation, and this is an order-of-magnitude
    # floor, not a precision instrument.  The boundary strip feeds the
    # relaxation zone, so an artefact left here walks into the child's
    # interior on the first blend.
    boundary_clamp = clamp_sint_undershoot_mapping(
        interpolated, floor_scale=float(abs(parent_chm).max()))
    conversion_receipt = None
    if target_mp != int(source_mp_physics):
        _refuse_unbuilt_p3_offline_edge(int(source_mp_physics), target_mp)
        if (target_mp in _CROSS_SCHEME_REFUSED_MP_PHYSICS
                or int(source_mp_physics)
                in _CROSS_SCHEME_REFUSED_MP_PHYSICS):
            refused = (target_mp if target_mp
                       in _CROSS_SCHEME_REFUSED_MP_PHYSICS
                       else int(source_mp_physics))
            raise OfflineChildContractError(
                f"offline cross-physics forcing across the mp_physics="
                f"{int(source_mp_physics)} -> {target_mp} edge is REFUSED: "
                f"{_cross_scheme_refusal_clause(refused)} has no measured "
                "cross-scheme entry closure, and the online "
                "nest lane refuses the same edge "
                "(gpuwm/core/microphysics_transition.py::"
                "UNVALIDATED_MIXED_EDGE_SELECTORS)")
        if target_mp != 18:
            raise OfflineChildContractError(
                "cross-physics offline forcing currently targets NSSL mp18")
        xp = np if backend == "cpu" else __import__("cupy")
        child_mub = sint(raw_device["MUB"], registrations[""])
        child_mup = interpolated["mu"][0]
        child_mu = child_mub + child_mup
        c1h = xp.asarray(coeffs["c1h"], dtype=xp.float32)[:, None, None]
        c2h = xp.asarray(coeffs["c2h"], dtype=xp.float32)[:, None, None]
        child_chm = c1h * child_mu[None] + c2h
        source_mixing = {
            name: _to_host(interpolated.pop(name) / child_chm)
            for name in tuple(moisture)
        }
        mapped, conversion_receipt = map_microphysics_to_nssl18(
            source_mixing, source_mp_physics=int(source_mp_physics),
            morr_rimed_ice=morr_rimed_ice,
            diagnose_missing=diagnose_missing,
        )
        child_chm_host = _to_host(child_chm)
        interpolated.update({
            name: child_chm_host * value for name, value in mapped.items()
        })
    remap_receipts = ()
    if child_znw is not None:
        parent_znw = np.asarray(raw["ZNW"], dtype=np.float64).reshape(-1)
        interpolated, remap_receipts = (
            _remap_boundary_snapshot_to_child_ladder(
                interpolated,
                child_mub=_to_host(sint(raw_device["MUB"], registrations[""])),
                child_phb=_to_host(sint(
                    _backend_array(raw["PHB"], backend), registrations[""])),
                parent_znw=parent_znw, child_znw=child_znw,
                hybrid_opt=hybrid_opt, etac=etac, p_top=p_top,
                moisture_names=frozenset(moisture)))
    fields = MappingProxyType({name: _to_host(value)
                               for name, value in interpolated.items()})
    receipt = MappingProxyType({
        "path": str(Path(path).resolve()),
        "valid_time": info.valid_time.isoformat(),
        "geometry_sha256": info.geometry_sha256,
        "source_kind": info.source_kind,
        "source_mp_physics": int(source_mp_physics),
        "advisory_inferred_mp_physics": info.inferred_mp_physics,
        "source_physics_binding": (
            None if physics_binding is None else dict(physics_binding.receipt())),
        "target_mp_physics": target_mp,
        "backend": backend,
        "positive_definite_clamp": boundary_clamp,
        "hybrid_opt": hybrid_opt,
        "etac": etac,
        "p_top": p_top,
        "field_inventory": tuple(sorted(fields)),
        "vertical_remap": None if child_znw is None else {
            "source_levels": int(np.asarray(raw["ZNW"]).size - 1),
            "target_levels": int(child_znw.size - 1),
            "fields": tuple(item.summary() for item in remap_receipts),
            "max_relative_drift": max(
                (item.max_relative_drift for item in remap_receipts),
                default=0.0),
        },
        "conversion": None if conversion_receipt is None else dict(conversion_receipt),
        "seconds": time.perf_counter() - started,
    })
    return InterpolatedBoundarySnapshot(info.valid_time, fields, receipt)


def build_offline_lateral_boundaries(
        contract: ParentHistoryContract, placement: OfflineChildPlacement, *,
        target_mp_physics: int | None = None,
        morr_rimed_ice: int | None = None, backend: str = "cpu",
        child_eta_levels=None,
        diagnose_missing: Callable[[Mapping[str, np.ndarray], Sequence[str]],
                                   Mapping[str, np.ndarray]] | None = None,
        spec_bdy_width: int = 5, spec_zone: int = 1, relax_zone: int = 4,
) -> OfflineBoundaryResult:
    """Stream parent frames into compact child lateral value/tendency strips."""

    if contract.source_mp_physics is None:
        raise OfflineChildContractError(
            "offline boundary generation requires source mp_physics bound "
            "from a companion setup/namelist/manifest; inventory inference is "
            "advisory only")
    if contract.physics_binding is None:
        raise OfflineChildContractError(
            "offline boundary generation requires authoritative companion "
            "setup/namelist/restart evidence, not a bare scheme integer")
    started = time.perf_counter()
    side_names = ("west", "east", "south", "north")
    previous = None
    intervals = []
    receipts = []
    origin = contract.start_time
    for frame in contract.frames:
        snapshot = interpolate_parent_boundary_snapshot(
            frame.path, placement,
            source_mp_physics=contract.source_mp_physics,
            physics_binding=contract.physics_binding,
            target_mp_physics=target_mp_physics,
            morr_rimed_ice=morr_rimed_ice, backend=backend,
            child_eta_levels=child_eta_levels,
            diagnose_missing=diagnose_missing,
        )
        sides = {
            side: extract_lateral_side(snapshot.fields, side, spec_bdy_width)
            for side in side_names
        }
        if previous is not None:
            previous_time, previous_sides = previous
            intervals.append(build_lateral_interval_from_sides(
                previous_sides, sides,
                start_seconds=(previous_time - origin).total_seconds(),
                end_seconds=(snapshot.valid_time - origin).total_seconds(),
            ))
        previous = (snapshot.valid_time, sides)
        receipts.append(snapshot.receipt)
    boundaries = LateralBoundaries(
        tuple(intervals), int(spec_bdy_width), int(spec_zone), int(relax_zone))
    return OfflineBoundaryResult(
        boundaries=boundaries, frame_receipts=tuple(receipts),
        preparation_seconds=time.perf_counter() - started,
    )


__all__ = [
    "ChildSurfaceState",
    "OFFLINE_CHILD_MP_PHYSICS",
    "InterpolatedBoundarySnapshot", "InterpolatedInitialState",
    "OfflineBoundaryResult",
    "OfflineChildPlacement",
    "DERIVED_CHILD_SURFACE_CAVEAT", "DERIVED_CHILD_SURFACE_POLICY",
    "derive_child_surface_from_parent",
    "read_child_surface_state",
    "MomentDiagnosisRequired", "OfflineChildContractError",
    "ParentHistoryContract", "ParentHistoryFrame", "ParentPhysicsBinding",
    "bind_parent_physics_from_gpuwm_restart",
    "bind_parent_physics_from_wrf_namelist",
    "build_offline_child_domain_state", "build_offline_lateral_boundaries",
    "inspect_parent_history_frame", "interpolate_parent_boundary_snapshot",
    "interpolate_parent_initial_state", "map_microphysics_to_nssl18",
    "read_parent_microphysics", "reserve_output_root",
    "validate_parent_history",
]
