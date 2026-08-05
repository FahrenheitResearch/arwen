"""A fine one-way nest over the radar-DA free forecast.

WHY THIS EXISTS.  The nowcast assimilates and forecasts on a single
coarse domain.  Interpolating its final output to a finer mesh produces
an unbalanced field that spends its first 15--30 minutes adjusting --
exactly the window a nowcast exists to serve.  This module runs a real
one-way nest over the free-forecast legs instead, so the fine field is
produced by the model rather than by a resampler, and takes lateral
forcing from the parent at every parent step throughout.

WHAT THE CODE PERMITTED, STATED PLAINLY.  gpuwm has three ways to give a
run a child domain and none of them is "introduce a nest mid-run":

* the whole-run live nest (``build_experiment`` initialises every child
  eagerly at t=0) is the proven path, but it would carry the nest through
  the DA cycling as well;
* the delayed per-domain ``start_time`` hook (``on_domain_start``) is
  implemented, but it activates a child by calling ``initialize_child``,
  which resolves its source snapshot at the child's own start time
  (``gpuwm.ingest.nest_init._initial_snapshot``).  That COLD-STARTS the
  nest from the raw analysis valid at the free-forecast start.  Every
  radar increment the cycling bought would live only in the parent and
  reach the nest through its lateral boundaries alone -- strictly worse
  than the post-hoc interpolation this work replaces, because it also
  discards the assimilation;
* ``gpuwm downscale`` / the offline child inherits the parent's evolved
  state, but forces the child from archived HISTORY frames rather than
  every parent step, which is the specific weakness being avoided.

None of those is used.  The route taken is the one the DA driver's own
shape makes available: ``tools/da_cycle_prepared.py`` rebuilds the entire
model from scratch at every leg and joins legs through host snapshots,
so a leg is already a fresh model whose clock is placed at the leg
boundary.  A nested leg is therefore not a mid-run introduction at all --
it is an ordinary whole-run two-domain model whose run happens to be one
leg long, and whose parent state at t=0 of that leg is the ANALYSED
state.  The child is built from that parent by full SINT
(:func:`gpuwm.ingest.nest_init.parent_only_init`), so it inherits every
increment the cycling produced, and is then forced laterally by the
parent every parent step for the rest of the leg.

WHAT THAT COSTS, ALSO PLAINLY.  ``parent_only_init`` is the idealized
``input_from_file=F`` branch: it returns ``static_fields=None`` and
``soil=None``, and ``gpuwm.runtime.prepare_child_case`` refuses a
full-physics child without them.  The nowcast profile runs Noah, MM5
surface layer and YSU, so a land state is mandatory.  This module supplies
it by replicating the parent's land column onto the child with WRF's own
nest-down nearest-donor mapping (``interp_fcni``/``interp_fcnm``,
interp_fcn.F), which is the same donor arithmetic
:class:`gpuwm.core.nest_interp.NestRegistration` already builds for SINT.

The consequence is a terrain and land-state policy identical to the one
the offline-child contract already registers as
``sint-parent-inherited``: the fine domain runs the PARENT's terrain
(SINT-interpolated) and the PARENT's land categories, soil and snow
(nearest-donor replicated).  **The nest refines the atmosphere, not the
surface.**  It buys finer dynamics, finer microphysics and a shorter
acoustic step over the analysed storm; it does not buy fine orography,
because on this route no fine static data exists to buy it with.  A
caller who wants fine orography needs a per-domain prepared cache and the
real-data child initialiser, which is a different (and much larger) piece
of work -- and which reopens the balance problem, because real fine
terrain under a SINT-inherited atmosphere is precisely the mismatch WRF's
``blend_terrain``/``adjust_tempqv``/``press_adj`` sequence exists to
repair.

Keeping terrain, base state and land all SINT/nearest of the same parent
is what makes the child's initial state BALANCED: the same linear
operator is applied to the base state and to the prognostic perturbations
that are defined against it, so the child starts in the same hydrostatic
and mass balance the parent was in.  That is the whole point of the
exercise.

ONE-WAY ONLY.  ``feedback`` is pinned to 0 here and the assembled tree is
checked for it.  The parent cannot be altered by the presence of the
child, and ``tests/test_da_nested_forecast_gpu.py`` holds that as a
bitwise ratchet in the same spirit as the N3/N4/N5 ancestor-inertness
gates.

Nothing here is on a default route.  EXPERIMENTAL.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, replace
from fractions import Fraction
from types import MappingProxyType, SimpleNamespace

import numpy as np

from gpuwm.config import validate_run_config
from gpuwm.core.nest_interp import register_nest
from gpuwm.experiment import DomainConfig, ExperimentConfig

#: Schema of the receipt block :func:`nested_forecast_receipt` writes.
RECEIPT_SCHEMA = "gpuwm-da.nested-forecast.v1"

#: Terrain and land-state policy this module implements, named with the
#: same string the offline-child contract already registers so the two are
#: greppably the same claim (``gpuwm/offline_child_run.py``).
TERRAIN_POLICY = "sint-parent-inherited"

#: Land/soil/snow policy: the parent's column, replicated to every child
#: cell inside it by WRF's nest-down nearest-donor mapping.  Named
#: separately from :data:`TERRAIN_POLICY` because it is a different
#: operator (nearest, not SINT) chosen for a different reason (categories
#: and the continuous fields keyed to them must not be blended across a
#: coastline).
LAND_POLICY = "parent-donor-replicated"

#: The default number of ensemble members carried on the nest.  The
#: parent carries the full ensemble; the nest is the detailed picture and
#: is deliberately cheap.  1 = the control trajectory only.
DEFAULT_NEST_MEMBERS = 0

#: Nest-down operator for every field of the native static contract
#: (:data:`gpuwm.native_wrf_contract.NATIVE_STATIC_REQUIRED`).
#:
#: ``donor`` is WRF's ``interp_fcni``/``interp_fcnm`` nearest-donor
#: pickup.  Every static field takes it, for one reason: they are all
#: either a category, a mask, or a climatology keyed to a category, and a
#: bilinear blend of any of those manufactures a surface no column ever
#: had -- a fractional land-use index, a soil temperature averaged across
#: a coastline, a greenness that belongs to neither neighbour.
#:
#: ``grid`` marks the map-factor/Coriolis geometry, which is NOT inherited:
#: the child computes its own from its own :class:`LambertGrid` through
#: ``nest_init._set_map_fields``.  Inheriting it would put the parent's
#: metric on the child's mesh.
STATIC_NEST_DOWN = MappingProxyType({
    "HGT_M": "donor",          # unblended fine terrain -- see TERRAIN_POLICY
    "LANDUSEF": "donor",       # (21,ny,nx) land-use fractions
    "LANDMASK": "donor",       # WPS 1/0 land mask -- interp_fcnm
    "LU_INDEX": "donor",       # dominant land-use category -- interp_fcni
    "SOILCTOP": "donor",       # (16,ny,nx) top-layer soil fractions
    "SCT_DOM": "donor",        # dominant top soil category -- interp_fcni
    "SOILCBOT": "donor",       # (16,ny,nx) bottom-layer soil fractions
    "SCB_DOM": "donor",        # dominant bottom soil category
    "GREENFRAC": "donor",      # (12,ny,nx) climatology keyed to LU_INDEX
    "LAI12M": "donor",         # (12,ny,nx) climatology keyed to LU_INDEX
    "ALBEDO12M": "donor",      # (12,ny,nx) climatology keyed to LU_INDEX
    "SNOALB": "donor",         # maximum snow albedo, keyed to LU_INDEX
    "SOILTEMP": "donor",       # deep soil temperature
    "TMN": "donor",            # deep-layer soil temperature
    "MAPFAC_M": "grid", "MAPFAC_U": "grid", "MAPFAC_V": "grid",
    "F": "grid", "E": "grid", "SINALPHA": "grid", "COSALPHA": "grid",
})

#: The canonical Noah surface inventory
#: (``gpuwm.ingest.hrrr_physics._CANONICAL_SURFACE_FIELDS``) and its
#: nest-down operator.  All donor: every one of these is either a mask or
#: a land-column quantity whose value is only meaningful together with the
#: category and mask it was derived under.  Blending TSLB across a
#: coastline would manufacture a soil temperature no column ever had.
SURFACE_NEST_DOWN = MappingProxyType({
    "TSK": "donor", "TSLB": "donor", "SMOIS": "donor", "SH2O": "donor",
    "TMN": "donor", "SEAICE": "donor", "XLAND": "donor",
    "LANDMASK": "donor", "SNOW": "donor", "SNOWH": "donor",
})

#: Near-surface diagnostics carried down from the PARENT'S OWN physics
#: driver rather than re-derived on the child.
#:
#: The alternative -- diagnosing them from the child's lowest model level
#: -- would require this module to reason about which of them the surface
#: layer overwrites on its mandatory ITIMESTEP=1 call and which the PBL
#: reads first.  Taking the parent's values through the same donor
#: operator as every other surface field needs no such reasoning and
#: keeps one rule for the whole surface: the child starts as its parent's
#: column, replicated.
NEAR_SURFACE_NEST_DOWN = MappingProxyType({
    "psfc": "donor", "t2": "donor", "th2": "donor", "q2": "donor",
    "u10": "donor", "v10": "donor",
})


class NestedForecastRefusal(ValueError):
    """A nested free-forecast leg that this module will not assemble."""


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NestGeometry:
    """The user-facing configuration surface for the nest.

    Everything else about the child is DERIVED.  ``dx``/``dy``/``dt`` are
    never typed by a caller -- they come off the parent through the ratio
    chain exactly as ``ExperimentConfig.dx_exact``/``dt_exact`` define
    them, which is the same rule every shipped nested config in this repo
    states in its header.
    """

    #: Parent-to-child refinement ratio, applied to BOTH space and time.
    #: WRF's SINT assumes a square ratio and gpuwm rejects a non-square
    #: one, so one number is the whole story.
    ratio: int = 3
    #: Child mass-grid extent, in CHILD cells.  ``None`` means "derive
    #: from :attr:`half_width_km`".
    nx: int | None = None
    ny: int | None = None
    #: Alternative to nx/ny: the half-width of the nest in kilometres,
    #: centred in the parent.  This is the natural way to say "a 1 km pass
    #: over the middle 120 km of the nowcast".
    half_width_km: float | None = None
    #: Lower-left parent cell of the child, 1-based WRF namelist
    #: semantics.  ``None`` centres the nest in the parent.
    i_parent_start: int | None = None
    j_parent_start: int | None = None
    #: Child history cadence.  ``None`` inherits the parent's.
    history_interval_s: float | None = None
    #: Number of ensemble members to carry on the nest, beside the
    #: control.  The parent always carries the full ensemble.
    members: int = DEFAULT_NEST_MEMBERS

    def __post_init__(self) -> None:
        if int(self.ratio) < 2:
            raise NestedForecastRefusal(
                f"nest ratio must be at least 2, got {self.ratio!r}: a "
                "ratio-1 nest refines nothing and exists only as an "
                "identity fixture in the N2 verification cases")
        if (self.nx is None) != (self.ny is None):
            raise NestedForecastRefusal(
                "nest nx and ny must be given together or not at all")
        if (self.nx is None) == (self.half_width_km is None):
            raise NestedForecastRefusal(
                "state the nest extent exactly once: either nx/ny in child "
                "cells or half_width_km in kilometres")
        if (self.i_parent_start is None) != (self.j_parent_start is None):
            raise NestedForecastRefusal(
                "nest i_parent_start and j_parent_start must be given "
                "together or not at all")
        if int(self.members) < 0:
            raise NestedForecastRefusal(
                f"nest members must be >= 0, got {self.members!r}")


def resolve_nest_extent(geometry: NestGeometry, parent_run) -> tuple[int, int]:
    """Child mass-grid extent in child cells, as an exact multiple of ratio.

    A child extent that is not a whole number of parent cells has a
    ragged edge whose donor column is shared with the interior, so the
    extent is rounded DOWN to a whole parent cell and reported as such.
    """
    ratio = int(geometry.ratio)
    if geometry.nx is not None:
        nx, ny = int(geometry.nx), int(geometry.ny)
    else:
        child_dx_km = (float(parent_run.dx) / ratio) / 1000.0
        span = 2.0 * float(geometry.half_width_km) / child_dx_km
        nx = ny = int(span)
    nx -= nx % ratio
    ny -= ny % ratio
    if nx < 3 * ratio or ny < 3 * ratio:
        raise NestedForecastRefusal(
            f"nest resolves to {nx}x{ny} child cells, which is smaller "
            f"than the {3 * ratio} cells a five-point specified frame plus "
            "a relaxation zone needs; ask for a larger nest")
    return nx, ny


def centred_placement(nx: int, ny: int, ratio: int,
                      parent_run) -> tuple[int, int]:
    """Centre the child in its parent, in 1-based parent cells.

    Mirrors ``gpuwm.domain_wizard._domain_tables``' centring so a nest
    emitted here lands where the wizard would put it.
    """
    i_start = (int(parent_run.nx) - nx // ratio) // 2 + 1
    j_start = (int(parent_run.ny) - ny // ratio) // 2 + 1
    return i_start, j_start


# ---------------------------------------------------------------------------
# admissibility
# ---------------------------------------------------------------------------

def validate_nest_admissibility(child_run, *, parent_run,
                                acknowledgements=()) -> dict:
    """Refuse a child configuration that is inadmissible at its spacing.

    This is the "enforced through the validator, not by convention" half
    of the design.  Two of these checks exist elsewhere in the tree only
    as ADVISORIES on the wizard's sizing path
    (:func:`gpuwm.domain_wizard.cumulus_gray_zone_advisory`,
    :func:`gpuwm.domain_wizard.gray_zone_advisory`); an advisory printed
    during config authoring cannot police a domain this module derives at
    runtime, so on this route they are refusals.  The thresholds are
    IMPORTED from the wizard rather than restated, so the two surfaces
    cannot drift apart.

    Returns the admissibility record for the receipt.
    """
    from gpuwm.domain_wizard import (CUMULUS_CONVECTION_PERMITTING_DX_KM,
                                     GRAY_ZONE_DX_KM)

    dx_km = float(child_run.dx) / 1000.0
    acknowledged = frozenset(acknowledgements)

    if int(child_run.cu_physics) != 0 \
            and dx_km < CUMULUS_CONVECTION_PERMITTING_DX_KM:
        raise NestedForecastRefusal(
            f"cu_physics={child_run.cu_physics} on a {dx_km:g} km nest is "
            f"below the {CUMULUS_CONVECTION_PERMITTING_DX_KM:g} km "
            "convection-permitting threshold: the registry offers only the "
            "classic Kain-Fritsch closure, which is not scale-aware, so it "
            "would double-count the convection the nest is being built to "
            "resolve explicitly. Nests on this route run cu_physics=0")

    pbl_ack = "nested-forecast:sub-gray-zone-pbl"
    if dx_km < GRAY_ZONE_DX_KM and int(child_run.bl_pbl_physics) != 0 \
            and pbl_ack not in acknowledged:
        raise NestedForecastRefusal(
            f"bl_pbl_physics={child_run.bl_pbl_physics} at {dx_km:g} km is "
            f"below the {GRAY_ZONE_DX_KM:g} km PBL gray-zone boundary. A "
            "column scheme parameterises the whole boundary-layer flux "
            "on the premise that none of the eddies carrying it are "
            "resolved; below this spacing some of them are, and the "
            "transport gets counted twice. Either run the nest at or "
            "above the gray-zone spacing, or select bl_pbl_physics=0 "
            f"with km_opt=3, or acknowledge {pbl_ack!r} deliberately")

    if float(child_run.spec_exp) != 0.0:
        raise NestedForecastRefusal(
            f"spec_exp={child_run.spec_exp} on a nested domain: WRF's "
            "nested lbc_fcx_gcx branch carries no exponential sponge term, "
            "so children force with spec_exp=0 exactly "
            "(gpuwm/experiment.py zeroes this at TOML load; a config "
            "derived in code has to state it)")

    if not bool(child_run.nested) or bool(child_run.specified):
        raise NestedForecastRefusal(
            "a nest must carry nested=True and specified=False: the "
            "specified branch reads external lateral boundary tables, and "
            "a child's tables come from its parent's coupler")

    if int(child_run.nz) != int(parent_run.nz):
        raise NestedForecastRefusal(
            f"vertical nesting is not implemented: child nz={child_run.nz} "
            f"differs from parent nz={parent_run.nz}. The experiment's one "
            "shared eta ladder is used by every domain, so a fine nest "
            "inherits the parent's vertical resolution -- which is often "
            "what actually limits the result")

    if int(child_run.mp_physics) != int(parent_run.mp_physics):
        raise NestedForecastRefusal(
            f"child mp_physics={child_run.mp_physics} differs from parent "
            f"mp_physics={parent_run.mp_physics}: a scheme boundary across "
            "a nest edge is a separate, certified capability "
            "(gpuwm.core.microphysics_transition) and is not part of this "
            "route")

    validate_run_config(child_run)
    return {
        "child_dx_km": dx_km,
        "cumulus_convection_permitting_dx_km": float(
            CUMULUS_CONVECTION_PERMITTING_DX_KM),
        "pbl_gray_zone_dx_km": float(GRAY_ZONE_DX_KM),
        "cu_physics": int(child_run.cu_physics),
        "bl_pbl_physics": int(child_run.bl_pbl_physics),
        "km_opt": int(child_run.km_opt),
        "acknowledgements": sorted(acknowledged),
    }


# ---------------------------------------------------------------------------
# config derivation
# ---------------------------------------------------------------------------

#: Sixth-order diffusion factor by nest depth, the wizard's ladder
#: (``gpuwm.domain_wizard._DIFF6_FACTORS``) indexed the same way.
def _diff6_factor(depth: int) -> float:
    from gpuwm.domain_wizard import _DIFF6_FACTORS
    index = min(int(depth), len(_DIFF6_FACTORS) - 1)
    return float(_DIFF6_FACTORS[index])


def nest_domain_config(exp: ExperimentConfig, geometry: NestGeometry,
                       *, acknowledgements=()) -> DomainConfig:
    """Derive the child ``DomainConfig`` from the parent experiment.

    dx and dt are derived, never typed: dx is the parent's divided by the
    ratio exactly, and dt is the CHAINED single-precision division WRF
    performs in ``share/set_timekeeping.F``, because that -- not the exact
    rational -- is the number the physics actually receives.
    """
    parent_dc = exp.root
    parent_run = parent_dc.run
    ratio = int(geometry.ratio)

    nx, ny = resolve_nest_extent(geometry, parent_run)
    if geometry.i_parent_start is None:
        i_start, j_start = centred_placement(nx, ny, ratio, parent_run)
    else:
        i_start = int(geometry.i_parent_start)
        j_start = int(geometry.j_parent_start)

    dx_exact = Fraction(parent_run.dx) / ratio
    dy_exact = Fraction(parent_run.dy) / ratio
    # WRF's REAL chained division (share/set_timekeeping.F:368), not the
    # exact rational: the FP32 image is what the child's physics is fed.
    child_dt = float(np.float32(parent_run.dt) / np.float32(ratio))
    history = (float(parent_dc.history_interval_s)
               if geometry.history_interval_s is None
               else float(geometry.history_interval_s))

    child_run = replace(
        parent_run,
        grid_id=parent_run.grid_id + 1,
        nx=nx, ny=ny,
        dx=float(dx_exact), dy=float(dy_exact),
        dt=child_dt,
        nested=True, specified=False, open_x=False, open_y=False,
        # Nested lbc_fcx_gcx has no exponential sponge term.
        spec_exp=0.0,
        # Explicit convection at the nest's spacing; the wizard pins this
        # on every nest it emits regardless of the parent's selection.
        cu_physics=0, cudt_minutes=0.0,
        diff_6th_factor=_diff6_factor(1),
        output_interval_s=history,
        case=f"{parent_run.case}_nest" if parent_run.case else "nest",
    )
    validate_nest_admissibility(child_run, parent_run=parent_run,
                                acknowledgements=acknowledgements)
    return DomainConfig(
        grid_id=parent_dc.grid_id + 1,
        parent_id=parent_dc.grid_id,
        i_parent_start=i_start, j_parent_start=j_start,
        parent_grid_ratio=ratio, parent_time_step_ratio=ratio,
        history_interval_s=history, run=child_run,
        time_step=None)


def nested_experiment(exp: ExperimentConfig,
                      child_dc: DomainConfig) -> ExperimentConfig:
    """The parent experiment plus one child, one-way.

    ``feedback`` is pinned to 0 rather than inherited: a nested free
    forecast exists to give a detailed view of the parent's forecast, and
    a child that can restrict its parent is a different experiment whose
    parent trajectory is no longer the one the ensemble was scored on.
    """
    if len(exp.domains) != 1:
        raise NestedForecastRefusal(
            f"nested free-forecast legs attach ONE child to a single-domain "
            f"parent; this experiment already carries {len(exp.domains)} "
            "domains")
    nested = dataclasses.replace(
        exp, feedback=0, smooth_option=0,
        domains=(exp.root, child_dc))
    if nested.feedback != 0:
        raise NestedForecastRefusal("one-way nesting is mandatory here")
    # Prove the derivation rule held rather than asserting it in prose.
    expected_dx = Fraction(exp.root.run.dx) / child_dc.parent_grid_ratio
    if Fraction(child_dc.run.dx) != expected_dx:
        raise NestedForecastRefusal(
            f"child dx {child_dc.run.dx} is not the parent's divided by "
            f"{child_dc.parent_grid_ratio} exactly ({float(expected_dx)})")
    if nested.dx_exact(child_dc.grid_id) != expected_dx:
        raise NestedForecastRefusal(
            "child dx disagrees with ExperimentConfig.dx_exact")
    return nested


# ---------------------------------------------------------------------------
# nest-down of the surface: WRF's interp_fcni / interp_fcnm
# ---------------------------------------------------------------------------

def donor_registration(child_dc: DomainConfig, parent_run):
    """The mass-stagger registration whose ``ci``/``cj`` are WRF's pickup.

    ``interp_fcn.F``'s integer/mask nest-down loops read, for child index
    ``n`` (1-based)::

        ci = ipos + (n-1)/nri

    which is exactly the donor map :func:`register_nest` builds for the
    unstaggered ``interp`` wrapper (its ``ioff`` is 0 for mass stagger).
    Reusing that registration is deliberate: the nearest-donor land field
    and the SINT'd atmosphere above it are then guaranteed to be keyed to
    the SAME parent cell, which is the property that makes the child's
    surface and its column consistent.
    """
    return register_nest(
        nri=child_dc.parent_grid_ratio, nrj=child_dc.parent_grid_ratio,
        i_parent_start=child_dc.i_parent_start,
        j_parent_start=child_dc.j_parent_start,
        child_nx=child_dc.run.nx, child_ny=child_dc.run.ny,
        parent_nx=int(parent_run.nx), parent_ny=int(parent_run.ny),
        stagger="", wrapper="interp")


def donor_nest_down(field, registration) -> np.ndarray:
    """Replicate a parent field onto the child by nearest donor.

    Trailing two axes are (ny, nx); any leading axes (soil layers, the
    twelve climatology months) ride along untouched.  This is WRF's
    ``interp_fcni``/``interp_fcnm``: no averaging, no blending, every
    child cell takes the whole value of the parent cell it lies inside.
    """
    array = np.asarray(field)
    if array.ndim < 2:
        raise ValueError(
            f"nest-down needs at least (ny, nx), got shape {array.shape}")
    expected = (registration.nyp, registration.nxp)
    if array.shape[-2:] != expected:
        raise ValueError(
            f"parent field has horizontal shape {array.shape[-2:]}, "
            f"expected {expected} for this registration")
    return np.ascontiguousarray(
        array[..., registration.cj, :][..., :, registration.ci])


def nest_down_mapping(source, policy, registration, *,
                      what: str) -> dict[str, np.ndarray]:
    """Apply a :data:`STATIC_NEST_DOWN`-style policy to a whole inventory.

    A field with no declared operator is a REFUSAL, not a default.  A
    surface inventory that grows a field this module has never reasoned
    about must be reasoned about once, here, rather than silently taking
    whichever operator happened to be convenient.
    """
    out: dict[str, np.ndarray] = {}
    dropped: list[str] = []
    for name, value in source.items():
        operator = policy.get(name)
        if operator is None:
            raise NestedForecastRefusal(
                f"{what} field {name!r} has no declared nest-down "
                "operator. Every field the child's physics reads must say "
                "how it crosses the nest edge; silently interpolating an "
                "unknown field is how a category becomes a fraction")
        if operator == "grid":
            dropped.append(name)
            continue
        if operator != "donor":
            raise NestedForecastRefusal(
                f"{what} field {name!r} declares unsupported nest-down "
                f"operator {operator!r}")
        out[name] = donor_nest_down(value, registration)
    return out, dropped


def child_land_inventory(static, surface, child_dc, parent_run) -> dict:
    """Child-grid static and Noah surface inventories, plus their receipt."""
    registration = donor_registration(child_dc, parent_run)
    static_host = {name: np.asarray(
        value.get() if hasattr(value, "get") else value)
        for name, value in static.items()}
    surface_host = {name: np.asarray(
        value.get() if hasattr(value, "get") else value)
        for name, value in surface.fields.items()}
    child_static, grid_derived = nest_down_mapping(
        static_host, STATIC_NEST_DOWN, registration, what="static")
    child_surface, _ = nest_down_mapping(
        surface_host, SURFACE_NEST_DOWN, registration, what="surface")
    # XLAND is a derived view of LANDMASK and the prepared-surface
    # validator checks their consistency.  Nearest-donor preserves that
    # by construction (both take the same donor cell), but the check is
    # cheap and the failure it catches is silent.
    if not np.array_equal(
            child_surface["XLAND"],
            np.where(child_surface["LANDMASK"] >= 0.5, 1.0, 2.0)):
        raise NestedForecastRefusal(
            "child XLAND diverged from its LANDMASK under nest-down")
    return {
        "static": child_static,
        "surface": SimpleNamespace(
            fields=MappingProxyType(child_surface)),
        "registration": registration,
        "receipt": {
            "terrain_policy": TERRAIN_POLICY,
            "land_policy": LAND_POLICY,
            "static_fields": sorted(child_static),
            "surface_fields": sorted(child_surface),
            "grid_derived_fields": sorted(grid_derived),
            "donor_operator": "wrf-interp-fcni",
        },
    }


# ---------------------------------------------------------------------------
# child construction and attachment
# ---------------------------------------------------------------------------

def build_nested_child(parent_node, child_dc, *, static, surface,
                       landuse_identity, valid_time, clock,
                       parent_driver, center_lat=None):
    """Build the child domain from the parent's LIVE state.

    This is the step that makes the whole design worth doing.  The child's
    prognostic atmosphere is a full SINT of the parent's -- which, on the
    free-forecast legs of a radar-DA cycle, is the ANALYSED atmosphere,
    carrying every increment the cycling produced.  The child does not
    re-read any analysis file and has no cold-start of its own.

    The parent is read and never written: ``parent_only_init`` fills the
    child through out-of-place ``sint`` into child-owned buffers, and the
    base-state capture copies to the host.  ``tests`` hold that as a
    bitwise property rather than an intention.
    """
    from gpuwm.core.model import DomainNode
    from gpuwm.core.nest import NestCoupler
    from gpuwm.ingest.nest_init import parent_only_init

    inventory = child_land_inventory(
        static, surface, child_dc, parent_node.cfg.run)
    initialized = parent_only_init(child_dc, parent_node)
    node = DomainNode(child_dc, initialized.grid, initialized.state,
                      clock, parent_node, [], None)
    node.coupler = NestCoupler(node, feedback=0)
    driver = _initialize_child_physics(
        initialized, child_dc.run, inventory, landuse_identity,
        valid_time, parent_driver=parent_driver,
        registration=inventory["registration"], center_lat=center_lat)
    parent_node.children.append(node)
    # The prepared route rebuilds a child per leg rather than restoring
    # one, and the restart classifier needs to hear that from the builder.
    node.state._nest_restart_classification = "REBUILT"
    return node, driver, inventory["receipt"]


def _initialize_child_physics(initialized, child_run, inventory,
                              landuse_identity, valid_time, *,
                              parent_driver, registration,
                              center_lat=None):
    """Attach Noah/PBL/surface-layer physics to a parent-derived child.

    Mirrors :func:`gpuwm.ingest.hrrr_physics.initialize_prepared_physics`
    field for field -- the same land-use initialisation, the same
    date-interpolated GEOG climatologies, the same snow-albedo seeding --
    but reads the child-grid inventories this module nested down instead
    of a prepared cache, because a child built from its parent has no
    prepared cache of its own.  ``prepare_child_case`` cannot be reused:
    it requires the T12 real-data products (``real``, ``static_fields``,
    ``horizontal``, ``soil``) and ``parent_only_init`` returns none of
    them by construction.
    """
    import cupy as cp

    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.landuse import initialize_landuse
    from gpuwm.core.physics import initialize_physics
    from gpuwm.static.build import monthly_interp_to_date

    static = inventory["static"]
    fields = inventory["surface"].fields
    state = initialized.state
    grid = initialized.grid

    required = {"MMINLU", "ISWATER", "ISLAKE", "ISICE"}
    if set(landuse_identity) != required:
        raise NestedForecastRefusal(
            "nested child land-use identity must contain exactly "
            f"{sorted(required)}")

    update_diagnostics(state, child_run.hypsometric_opt)
    lat, lon = grid.latlon_mass()
    landuse = initialize_landuse(
        static["LU_INDEX"], soil_type=static["SCT_DOM"],
        landmask=static["LANDMASK"], snow=fields["SNOW"],
        xice=fields["SEAICE"], valid_time=valid_time,
        cen_lat=float(getattr(grid, "cen_lat", grid.ref_lat)
                      if center_lat is None else center_lat),
        mminlu=str(landuse_identity["MMINLU"]),
        iswater=int(landuse_identity["ISWATER"]),
        islake=int(landuse_identity["ISLAKE"]),
        isice=int(landuse_identity["ISICE"]), fractional_seaice=True,
        soil_temperature=fields["TSLB"], sst=fields.get("SST"))
    vegfra = 100.0 * monthly_interp_to_date(static["GREENFRAC"], valid_time)
    lai = monthly_interp_to_date(static["LAI12M"], valid_time)
    driver = initialize_physics(
        state, child_run, landuse=landuse, tsk=fields["TSK"],
        soil_temperature=fields["TSLB"],
        soil_moisture=fields["SMOIS"],
        liquid_moisture=fields["SH2O"],
        ivgtyp=static["LU_INDEX"], isltyp=static["SCT_DOM"],
        vegfra=vegfra, tmn=fields["TMN"], xice=fields["SEAICE"],
        snow=fields["SNOW"], snow_depth=fields["SNOWH"],
        sst=fields.get("SST", fields["TSK"]),
        radiation_start_time=valid_time, radiation_latitude=lat,
        radiation_longitude=lon)
    from gpuwm.core.noah import noah_initial_snow_albedo
    driver.fields["snoalb"][...] = cp.asarray(
        noah_initial_snow_albedo(
            static["SNOALB"], static["LU_INDEX"], driver.noah_params,
            rdmaxalb=child_run.rdmaxalb),
        dtype=cp.float32)
    driver.fields["lai"][...] = cp.asarray(lai, dtype=cp.float32)
    driver.fields["shdmin"][...] = cp.asarray(
        100.0 * static["GREENFRAC"].min(axis=0), dtype=cp.float32)
    driver.fields["shdmax"][...] = cp.asarray(
        100.0 * static["GREENFRAC"].max(axis=0), dtype=cp.float32)
    _nest_down_near_surface(driver, parent_driver, registration, cp)
    return driver


def _nest_down_near_surface(driver, parent_driver, registration, cp) -> None:
    """Carry the parent's 2 m / 10 m diagnostics down by nearest donor."""
    for name, operator in NEAR_SURFACE_NEST_DOWN.items():
        if operator != "donor":
            raise NestedForecastRefusal(
                f"near-surface field {name!r} declares unsupported "
                f"nest-down operator {operator!r}")
        source = parent_driver.fields.get(name)
        if source is None:
            raise NestedForecastRefusal(
                f"parent physics driver carries no {name!r} field; the "
                "near-surface inventory this module nests down is not the "
                "one the parent's driver allocates")
        driver.fields[name][...] = cp.asarray(
            donor_nest_down(source.get() if hasattr(source, "get")
                            else source, registration),
            dtype=driver.fields[name].dtype)


def nested_forecast_receipt(*, geometry: NestGeometry, exp, child_dc,
                            admissibility, land_receipt,
                            legs, nest_members) -> dict:
    """The receipt block a caller records beside its run."""
    child_run = child_dc.run
    parent_run = exp.root.run
    return {
        "schema": RECEIPT_SCHEMA,
        "stability": "experimental",
        "feedback": int(exp.feedback),
        "one_way": exp.feedback == 0,
        "parent": {
            "grid_id": int(parent_run.grid_id),
            "nx": int(parent_run.nx), "ny": int(parent_run.ny),
            "nz": int(parent_run.nz),
            "dx_m": float(parent_run.dx), "dt_s": float(parent_run.dt),
        },
        "nest": {
            "grid_id": int(child_dc.grid_id),
            "nx": int(child_run.nx), "ny": int(child_run.ny),
            "nz": int(child_run.nz),
            "dx_m": float(child_run.dx), "dt_s": float(child_run.dt),
            "dx_exact_m": str(exp.dx_exact(child_dc.grid_id)),
            "dt_exact_s": str(exp.dt_exact(child_dc.grid_id)),
            "parent_grid_ratio": int(child_dc.parent_grid_ratio),
            "parent_time_step_ratio": int(child_dc.parent_time_step_ratio),
            "i_parent_start": int(child_dc.i_parent_start),
            "j_parent_start": int(child_dc.j_parent_start),
            "history_interval_s": float(child_dc.history_interval_s),
        },
        "initialization": {
            "source": "parent-live-state-sint",
            "primitive": "gpuwm.ingest.nest_init.parent_only_init",
            "inherits_assimilated_state": True,
            "cold_start_from_analysis_file": False,
            **land_receipt,
        },
        "admissibility": admissibility,
        "ensemble": {
            "nest_members": int(nest_members),
            "note": "the parent carries the full ensemble; the nest is "
                    "deliberately cheap",
        },
        "legs": list(legs),
        "cost": nest_cost_model(exp, child_dc),
    }


def nest_cost_model(exp, child_dc) -> dict:
    """Dycore cost of the nest relative to its parent, per leg.

    Nest work scales as (child columns / parent columns) times the number
    of child substeps per parent step, which for a square ratio applied to
    both space and time is ``ratio**3`` per covered parent cell.  Reported
    as a ratio rather than seconds because the per-point-step constant is
    a property of the card, not of the configuration.
    """
    parent_run = exp.root.run
    child_run = child_dc.run
    ratio = int(child_dc.parent_grid_ratio)
    parent_columns = int(parent_run.nx) * int(parent_run.ny)
    child_columns = int(child_run.nx) * int(child_run.ny)
    covered_parent_columns = (int(child_run.nx) // ratio) * (
        int(child_run.ny) // ratio)
    return {
        "parent_columns": parent_columns,
        "child_columns": child_columns,
        "covered_parent_columns": covered_parent_columns,
        "parent_fraction_covered": covered_parent_columns / parent_columns,
        "child_substeps_per_parent_step": ratio,
        "column_ratio": child_columns / parent_columns,
        # The headline: how many parent-domain-equivalents of dycore work
        # one nested leg costs on top of the parent's own.
        "dycore_cost_vs_parent": (child_columns / parent_columns) * ratio,
        "note": "multiply by the number of nested trajectories; the parent "
                "ensemble cost is unchanged",
    }
