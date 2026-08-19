"""Typed multi-domain experiment schema and TOML loader (Phase 5, lane L1).

Implements section A of the RATIFIED Phase-5 nesting architecture
(docs/superpowers/specs/2026-07-16-phase5-nesting-architecture.md): frozen
:class:`DomainConfig`/:class:`ExperimentConfig` dataclasses over the
existing :class:`gpuwm.config.RunConfig`, the ``[experiment]`` /
``[shared]`` / ``[[domain]]`` / ``[projection]`` TOML shape, and every
load-time validation rule of the ratified design -- all fail-loud.

Clock fields (architecture section C): the ROOT domain carries WRF's
exact integer-rational model step -- ``time_step`` (integer seconds) plus
the optional ``time_step_fract_num``/``time_step_fract_den`` correction,
mirroring WRF v4.6.1 Registry.EM_COMMON:2245-2246::

    rconfig   integer time_step           namelist,domains  1   -1  ih ...
    rconfig   integer time_step_fract_num namelist,domains  1    0  ih ...
    rconfig   integer time_step_fract_den namelist,domains  1    1  ih ...

Child ``dt`` and ``dx`` are NEVER hand-typed: ``dt_child = dt_parent /
parent_time_step_ratio`` and ``dx_child = dx_parent / parent_grid_ratio``
as exact rationals -- WRF divides the parent's rational time interval by
``parent_time_step_ratio`` EXACTLY (share/set_timekeeping.F:366-368,
``stepTime = domain_get_time_step(parents(1)) / parent_time_step_ratio``).
Explicitly supplied child values are cross-checked and a mismatch is a
hard error: the namelist chain is authoritative (the bundle's d04 runs at
exactly 1000/3 m and 5/3 s -- never a hand-typed "500 m", never 1.6667).

The legacy single-domain path is untouched: ``load_config()`` and the
``[grid]``/``[dynamics]``/``[run]`` tables resolve byte-identically
(pinned by tests/test_config_freeze.py); frozen verify cases construct
``RunConfig`` directly and never migrate.
"""

from __future__ import annotations

import difflib
import math
import io
import tomllib
from dataclasses import dataclass, fields, replace as _dc_replace
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import NamedTuple

import numpy as np

from gpuwm import physics_mode as physics_mode_module
from gpuwm.core import streaming as streaming_module
from gpuwm.config import (DEFAULT_COLUMN_CHUNK,
                          EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT,
                          MIX_ISOTROPIC_AUTO, RunConfig,
                          anisotropic_w_mixing_ratio,
                          auto_mix_isotropic_selection, radiation_enabled,
                          validate_run_config, warn_anisotropic_w_mixing)
from gpuwm.explain import layered, warn

#: Relative tolerance for cross-checking hand-typed child dx/dt against the
#: exact chain-derived rational.  Wide enough for a truncated decimal of
#: the true value (the bundle namelist's ``dx = 333.333333`` vs the exact
#: 1000/3 m, relative error ~1e-9); a wrong value (a hand-typed "500 m"
#: d04 against ratio 3 from 1 km) misses by ~0.5 and is a hard error.
_REL_TOL = 1.0e-6

#: WRF moving-nest namelist controls (Registry.EM_COMMON &domains).  Any
#: appearance is still rejected loudly, and the reason is unchanged: every
#: key here drives CONTINUOUS, per-step nest motion, which invalidates the
#: SINT donor index/weight tables inside the integration.  This tree offers
#: DISCRETE relocation instead -- whole parent cells at cycle boundaries,
#: donor tables rebuilt once per placement generation -- which preserves
#: the premise ("tables precomputed once at setup") that this rejection
#: exists to protect.  The discrete surface is :class:`RelocationConfig`
#: under ``[relocation]``.  Unlike WRF's per-step ``move_interval``, its
#: ``cadence_seconds``/``[[relocation.move]]`` schedule names cycle-boundary
#: OPPORTUNITIES for the discrete mechanism (Drew's storm-following order,
#: leg 2, 2026-08-06 -- superseding leg 1's no-schedule stance for the
#: runner while the per-step keys stay rejected).
_MOVING_NEST_KEYS = frozenset({
    "num_moves", "move_id", "move_interval", "move_cd_x", "move_cd_y",
    "vortex_interval", "max_vortex_speed", "corral_dist", "track_level",
    "time_to_move",
})

#: Keys accepted in ``[relocation]``.  ``enabled`` is mandatory and every
#: other key is refused while it is false, so a config cannot acquire a
#: movable nest by inheriting a block someone left behind.  ``follow`` is
#: the nested ``[relocation.follow]`` storm-tracking table
#: (:mod:`gpuwm.core.storm_tracking`); ``move`` is the manual itinerary
#: (``[[relocation.move]]`` rows); ``cadence_seconds`` is how often the
#: runner consults whichever source is configured.  All three admissible
#: only while enabled.
_RELOCATION_KEYS = frozenset({
    "enabled", "grid_id", "mode", "max_move_parent_cells",
    "min_overlap_fraction", "cadence_seconds", "follow", "move",
})

#: Keys accepted in each ``[[relocation.move]]`` row (the manual follow
#: itinerary; the config-driven stand-in for the tracker seam).
_RELOCATION_MOVE_KEYS = frozenset({
    "at_seconds", "di_parent_cells", "dj_parent_cells",
})

#: The only relocation mode implemented.  Named (rather than implied) so a
#: future second mode has to be selected deliberately.
DISCRETE_RELOCATION_MODE = "discrete-cycle-boundary"

#: Nesting guard keys accepted in [shared] with their WRF Registry
#: defaults; any OTHER value is rejected loudly (only the default
#: machinery is implemented).
_GUARD_DEFAULTS = {
    # Registry.EM_COMMON:2301: default 2 = SINT, the only implemented
    # horizontal nest interpolator (bilinear/NN/quadratic rejected).
    "interp_method_type": 2,
    # Registry.EM_COMMON:2300: 0 = standard eta-level interpolation;
    # 1 = isobaric re-interpolation, not implemented.
    "nest_interp_coord": 0,
    # Vertical nest refinement: rejected (identical vertical grid on all
    # domains; WRF only calls init_domain_vert_nesting when e_vert
    # differs, share/mediation_integrate.F:666).
    "vert_refine_method": 0,
    # High-resolution child terrain ingest: rejected (children SINT the
    # parent terrain and blend it, dyn_em/nest_init_utils.F).
    "input_from_hires": False,
}

_EXPERIMENT_KEYS = frozenset({
    "name", "start_time", "run_seconds", "feedback", "smooth_option",
    "blend_width", "spec_bdy_width", "restart_interval_s",
    "column_chunk", "acknowledgements",
    # The physics-fidelity axis (gpuwm/physics_mode.py).  Experiment-scope
    # because a tree whose domains ran different ledger entries could not be
    # compared across its own nest boundary, and because the whole point of
    # the axis is that one run is one arm.
    "physics_mode", "patchset", "patches",
})
_EXPERIMENT_REQUIRED = ("name", "start_time", "run_seconds",
                        "restart_interval_s")

_PROJECTION_KEYS = ("map_proj", "ref_lat", "ref_lon", "truelat1",
                    "truelat2", "stand_lon")

#: [perturbation] carries exactly one key: the [[perturbation.bubbles]]
#: array of tables.  Scalars may join later; today a stray scalar here is
#: most likely a bubble key that escaped its double-bracket table.
_PERTURBATION_KEYS = ("bubbles",)
_BUBBLE_KEYS = frozenset({
    "center_lat", "center_lon", "center_height_m", "radius_km",
    "depth_m", "amplitude_k", "rh_preserve",
})
_BUBBLE_REQUIRED = ("center_lat", "center_lon", "center_height_m",
                    "radius_km", "depth_m", "amplitude_k")
#: The largest admissible peak theta perturbation.  WRF's own idealized
#: warm bubbles run 3 K (em_quarter_ss); an order of magnitude above that
#: is no longer an initiation nudge but a rewrite of the analysis, so it
#: is refused with the value named rather than integrated in silence.
MAX_BUBBLE_AMPLITUDE_K = 10.0

#: RunConfig keys that may NOT appear in [shared]: they are per-domain
#: (derived or [[domain]]-owned), experiment-owned, or retired in the
#: experiment path.
_SHARED_FORBIDDEN = {
    "nx": "[[domain]] (e_we/nx)", "ny": "[[domain]] (e_sn/ny)",
    "dx": "[[domain]] (root only; children derive dx_parent/ratio)",
    "dy": "[[domain]] (root only)",
    "dt": "the root [[domain]] time_step (children derive)",
    "clock_dt": "nowhere -- retired in the experiment path (always 0.0; "
                "it persists solely in a frozen legacy verification "
                "profile)",
    "run_seconds": "[experiment]", "restart_interval_s": "[experiment]",
    "spec_bdy_width": "[experiment]",
    "output_interval_s": "[[domain]] history_interval_s",
    "case": "nowhere -- the experiment path does not use the legacy "
            "case registry",
    "specified": "[[domain]]", "nested": "[[domain]]",
    "grid_id": "[[domain]]",
}

#: Per-domain RunConfig scalars a [[domain]] table may override
#: (architecture section A: cu_physics/cudt on d01 only per the bundle
#: namelist, radt 12/3/1/1, bldt, epssm's Registry-default tail, and
#: diff_6th_factor 0.12/0.10/0.08/0.06).  The turbulence row (km_opt
#: through tke_drag_coefficient) is the per-domain closure selection the
#: design doc reserved ("turbulence treatment stays configurable per
#: domain"): a PBL parent may carry a PBL-off Smagorinsky child.  Every
#: per-domain RunConfig still passes the full validate_run_config battery,
#: so cross-key refusals (PBL requires a surface layer, km_opt=3/4
#: excludes khdif/kvdif, isfflx=0/2 need their consumer) apply per domain
#: with the same messages as a single-domain config.  Per-domain LES
#: selection is a configuration capability, implemented-unverified for
#: nested LES children: no evidence covers a nested, specified-boundary,
#: moist, or terrain-following LES domain.
#: One entry here is a gpuwm capability WRF cannot express, and it is
#: called out rather than left to be rediscovered: ``isfflx`` is
#: ``nentries=1`` in WRF (Registry.EM_COMMON:2644) -- a SCALAR namelist
#: rconfig, where ``km_opt`` (:2993) and ``bl_pbl_physics`` (:2617) are
#: ``max_domains`` columns.  ``isfflx = 1, 1, 0`` is therefore not a WRF
#: namelist at all: a Fortran namelist read of a list into a scalar is an
#: error, not a per-domain selection.  Admitting it per domain HERE is a
#: deliberate gpuwm-over-WRF extension of the TOML schema, and it is
#: reachable only by writing the TOML directly -- ``gpuwm.namelist_import``
#: reads ``isfflx`` as the scalar WRF spells (commit a0ef9d29 reverted the
#: column reader for inventing a spelling WRF does not have), and
#: ``hrrr_hierarchy_direct``'s certified raw runtime contract pins
#: ``("physics", "isfflx")`` to the scalar ``[1]`` and refuses a column.
#: So no namelist-driven route can produce a per-domain isfflx, and none
#: should: a config that uses it has left WRF-expressible territory and
#: cannot be round-tripped back to a namelist.
_DOMAIN_RUN_OVERRIDES = (
    # clos_choice/ishallow ride with cu_physics: per-domain because the
    # scheme they configure is, and inert (validated zero) on any domain
    # that does not select cu_physics = 3.
    "cu_physics", "cudt_minutes", "clos_choice", "ishallow",
    "radt", "radt_minutes", "bldt",
    "diff_6th_factor", "epssm", "spec_exp", "mp_physics", "moist",
    "moist_cq", "nest_microphysics_transition",
    "km_opt", "bl_pbl_physics", "sf_sfclay_physics", "c_s", "c_k",
    # WRF Registry.EM_COMMON:2889 declares moist_mix6_off max_domains, so it
    # is per domain here for the same reason diff_6th_factor is.
    "moist_mix6_off",
    "diff_6th_opt", "mix_isotropic", "mix_upper_bound", "isfflx",
    "tke_heat_flux", "tke_drag_coefficient", "tke_upper_bound",
    # Output-only, and per domain because its cost scales with the grid:
    # four extra (nz+1, ny, nx) planes per frame, so the finest domains
    # of a tree can be left off while the domains whose subgrid fluxes
    # are being read carry it.  The two SASE PHYSICS selectors are
    # deliberately absent: a nest whose domains ran different closures
    # could not be compared across its own boundary.
    "sase_flux_diag",
    # Output-only on the same terms: two extra (nz, ny, nx) planes per
    # frame, so a tree can carry the horizontal viscosity on the domain
    # whose mixing is being read and leave it off the rest.
    "hmix_k_diag",
    # LES-nest inflow seeding (P3): per-domain because the mechanism IS
    # per-edge -- it perturbs one child's rolling nest-boundary tables,
    # validate_run_config refuses it on a non-nested domain, and like
    # per-domain isfflx these are TOML-only gpuwm-over-WRF keys with no
    # namelist spelling (stock 4.6.1's boundary perturbation is the
    # stoch-package perturb_bdy pattern route, not a cell-perturbation
    # column; PROVENANCE.md D10).
    "inflow_perturbation", "inflow_perturbation_seed",
    "inflow_perturbation_amplitude_scale", "inflow_perturbation_faces",
)

#: Per-domain vertical keys are REJECTED outright (F1 amendment: the
#: vertical grid is single-sourced from ExperimentConfig.vertical, so
#: vertical nesting is impossible by construction).
_DOMAIN_VERTICAL_KEYS = ("nz", "e_vert", "eta_levels", "p_top", "ztop",
                         "hybrid_opt", "etac")

_DOMAIN_KEYS = frozenset({
    "grid_id", "parent_id", "i_parent_start", "j_parent_start",
    "parent_grid_ratio", "parent_time_step_ratio", "history_interval_s",
    "start_time",
    "e_we", "e_sn", "nx", "ny", "dx", "dy", "dt",
    "time_step", "time_step_fract_num", "time_step_fract_den",
    "specified", "nested",
    # The dormant-nest declaration (gpuwm/core/nest_spawn.py): a child
    # carrying `spawn = {...}` is declared, reserved in the memory plan,
    # and integrates nothing until its trigger fires mid-run.
    "spawn",
    # The per-domain [tiles] override (gpuwm/core/streaming.py): a domain
    # carrying `tiles = {...}` chooses its OWN road, and the tree-wide
    # [tiles] table is the default for every domain that does not.  This
    # is how "stream the parent, keep the child resident" -- and its
    # inverse -- are said explicitly rather than left to the planner.
    "tiles",
    *_DOMAIN_RUN_OVERRIDES,
})
_DOMAIN_REQUIRED = ("grid_id", "parent_id", "i_parent_start",
                    "j_parent_start", "parent_grid_ratio",
                    "parent_time_step_ratio", "history_interval_s")


@dataclass(frozen=True)
class VerticalConfig:
    """The single shared vertical coordinate (F1 amendment, §A).

    Frozen/hashable; serializes verbatim in the resolved-TOML round trip
    and sits INSIDE the experiment fingerprint (``eta_levels`` as the
    exact FP64 tuple).  Every domain shares this one object -- vertical
    nesting is rejected by construction (per-domain vertical keys are a
    load error; WRF only calls init_domain_vert_nesting when a nest
    refines the vertical grid, share/mediation_integrate.F:666).
    ``eta_levels = ()`` (with ``p_top = 0``) marks an idealized/legacy
    configuration whose vertical grid is built from nz/ztop instead.

    Invariants live HERE so programmatic construction cannot bypass them
    (shadow review S3): finiteness before ordering, 1.0 -> 0.0 strictly
    decreasing eta, finite non-negative p_top, hybrid_opt in (0, 1, 2),
    finite etac in [0, 1].
    """

    eta_levels: tuple[float, ...]
    p_top: float
    hybrid_opt: int
    etac: float

    @property
    def mass_level_count(self) -> int | None:
        """Return the derived mass-level count for an explicit eta grid.

        Idealized legacy configurations deliberately carry no explicit eta
        interfaces, so their count remains owned by ``RunConfig.nz`` and this
        property returns ``None``.  Real-data callers must require a concrete
        count rather than silently substituting a project-default profile.
        """

        return len(self.eta_levels) - 1 if self.eta_levels else None

    @property
    def interface_level_count(self) -> int | None:
        """Return WRF ``e_vert`` for an explicit eta grid, if present."""

        return len(self.eta_levels) if self.eta_levels else None

    def __post_init__(self):
        for i, value in enumerate(self.eta_levels):
            if not math.isfinite(value):
                raise ValueError(
                    f"eta_levels[{i}] = {value!r} is not finite; every "
                    "eta full level must be a finite float.")
        if self.eta_levels:
            if len(self.eta_levels) < 2:
                raise ValueError(
                    f"eta_levels needs at least 2 full levels, got "
                    f"{len(self.eta_levels)}.")
            if self.eta_levels[0] != 1.0 or self.eta_levels[-1] != 0.0:
                raise ValueError(
                    "eta_levels must run from 1.0 (surface) to 0.0 "
                    f"(top), got {self.eta_levels[0]!r} .. "
                    f"{self.eta_levels[-1]!r}.")
            if any(b >= a for a, b in zip(self.eta_levels,
                                          self.eta_levels[1:])):
                raise ValueError("eta_levels must be strictly decreasing.")
        if not math.isfinite(self.p_top) or self.p_top < 0.0:
            raise ValueError(
                f"p_top = {self.p_top!r} must be a finite non-negative "
                "pressure in Pa.")
        if self.hybrid_opt not in (0, 1, 2):
            raise ValueError(
                f"hybrid_opt must be 0/1 (B(eta) = eta) or 2 (WRF v4 "
                f"cubic-B hybrid), got {self.hybrid_opt!r}.")
        if not math.isfinite(self.etac) or not 0.0 <= self.etac <= 1.0:
            raise ValueError(
                f"etac = {self.etac!r} must be a finite eta value in "
                "[0, 1].")


#: WPS map_proj string -> WRF integer convention (1=lambert, 2=polar
#: stereographic, 3=mercator); the [shared] map_proj gate below.
_MAP_PROJ_WRF_CODES = {"lambert": 1, "polar": 2, "mercator": 3}


@dataclass(frozen=True)
class ProjectionConfig:
    """``[projection]`` table: the WPS &geogrid projection parameter set
    ``grids_from_wps_namelist`` already consumes (gpuwm/static/
    projection.py); resolved per the F1 amendment and consumed by Task 5.
    Frozen/hashable, inside the experiment fingerprint (every parameter
    exact).  Finiteness/range invariants here so programmatic
    construction cannot bypass them (shadow review S3).

    ``map_proj`` is the WPS string: ``lambert``, ``mercator``, or
    ``polar``.  All six keys are always present; for Mercator,
    ``truelat2`` and ``stand_lon`` do not enter the projection math
    (module_llxy semantics) but stay in the fingerprint.  Genuine-limit
    refusals: a Lambert cone spanning both hemispheres is
    ill-conditioned (lc_cone is written for same-signed true
    latitudes), and true latitudes at the pole/equator degenerate the
    respective projections."""

    map_proj: str
    ref_lat: float
    ref_lon: float
    truelat1: float
    truelat2: float
    stand_lon: float

    def __post_init__(self):
        if self.map_proj not in _MAP_PROJ_WRF_CODES:
            normalized = self.map_proj.lower().replace("_", "-")
            latlon = normalized in {
                "lat-lon", "latlon", "regular-ll", "rotated-lat-lon",
                "rotated-ll",
            }
            blocker = (
                " Regular/rotated latitude-longitude needs angular dx/dy "
                "rather than metre spacing and WRF's global/pole polar "
                "filter; rotated grids also need pole_lat/pole_lon state "
                "and the map_proj == 6 curvature branch."
                if latlon else "")
            raise ValueError(
                f"map_proj = {self.map_proj!r} is not supported "
                "(implemented: 'lambert', 'mercator', 'polar')."
                f"{blocker}")
        for key, lo, hi in (("ref_lat", -90.0, 90.0),
                            ("truelat1", -90.0, 90.0),
                            ("truelat2", -90.0, 90.0),
                            ("ref_lon", -180.0, 180.0),
                            ("stand_lon", -180.0, 180.0)):
            value = getattr(self, key)
            if not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(
                    f"{key} = {value!r} in [projection] must be a "
                    f"finite value in [{lo}, {hi}] degrees.")
        if self.map_proj == "lambert":
            if self.truelat1 * self.truelat2 < 0.0:
                raise ValueError(
                    f"Lambert true latitudes {self.truelat1!r} and "
                    f"{self.truelat2!r} span both hemispheres; the "
                    "lc_cone secant formula is defined for a cone on "
                    "one side of the equator.")
            if max(abs(self.truelat1), abs(self.truelat2)) >= 90.0 \
                    or min(abs(self.truelat1), abs(self.truelat2)) <= 0.0:
                raise ValueError(
                    "Lambert true latitudes must lie strictly between "
                    "the equator and the pole, got "
                    f"{self.truelat1!r}/{self.truelat2!r}; use map_proj "
                    "= 'mercator' toward the equator or 'polar' toward "
                    "the pole.")
            if max(abs(self.truelat1), abs(self.truelat2)) >= 89.9 \
                    or min(abs(self.truelat1), abs(self.truelat2)) <= 0.1:
                warn("Lambert true latitudes "
                     f"{self.truelat1!r}/{self.truelat2!r} sit inside "
                     "0.1 degrees of the projection's degenerate limits "
                     "(equator/pole); the cone is ill-conditioned there "
                     "-- mercator or polar is the better tool")
        elif self.map_proj == "mercator":
            if abs(self.truelat1) >= 90.0:
                raise ValueError(
                    f"Mercator truelat1 = {self.truelat1!r} is at the "
                    "pole; the projection degenerates (cos(truelat1) "
                    "-> 0).")
            if abs(self.truelat1) >= 89.9:
                warn(f"Mercator truelat1 = {self.truelat1!r} is within "
                     "0.1 degrees of the pole; cos(truelat1) is nearly "
                     "0 and the scale factor is extreme")
        elif self.map_proj == "polar":
            if abs(self.truelat1) <= 0.1:
                raise ValueError(
                    f"polar stereographic truelat1 = {self.truelat1!r} "
                    "does not select a hemisphere; the pole (from the "
                    "sign of truelat1) is the projection centre.")

    @property
    def wrf_map_proj(self) -> int:
        """WRF integer convention (1=lambert, 2=polar, 3=mercator)."""
        return _MAP_PROJ_WRF_CODES[self.map_proj]


@dataclass(frozen=True)
class BubbleConfig:
    """One validated [[perturbation.bubbles]] entry.

    A cosine-squared warm bubble added ONCE to the initial potential
    temperature after the base real-data state is final: peak
    ``amplitude_k`` Kelvin at (``center_lat``, ``center_lon``,
    ``center_height_m`` AGL), falling to exactly zero at the ellipse
    ``sqrt((r_h/radius_km)^2 + ((z-zc)/depth_m)^2) = 1`` -- the WRF
    em_quarter_ss shape (module_initialize_ideal.F; the port's own
    transcription is gpuwm/verify/cases/moist_bubble.py).  ``depth_m``
    is the vertical HALF-depth.  ``rh_preserve`` keeps relative humidity
    constant through the theta change by adjusting qv inside the bubble;
    the default leaves qv byte-untouched.

    Geometry validation (center inside the coarse domain, at least one
    cell touched) needs the projected grid and therefore happens at
    prepare time (:mod:`gpuwm.ingest.init_perturbation`), before any
    integration step.
    """

    center_lat: float
    center_lon: float
    center_height_m: float
    radius_km: float
    depth_m: float
    amplitude_k: float
    rh_preserve: bool = False

    def __post_init__(self):
        for name in ("center_lat", "center_lon", "center_height_m",
                     "radius_km", "depth_m", "amplitude_k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(
                    value, (int, float)) or not math.isfinite(value):
                raise ValueError(
                    f"{name} must be a finite number, got {value!r}")
            object.__setattr__(self, name, float(value))
        if not -90.0 <= self.center_lat <= 90.0:
            raise ValueError(
                f"center_lat = {self.center_lat!r} is outside [-90, 90] "
                "degrees")
        if not -180.0 <= self.center_lon <= 180.0:
            raise ValueError(
                f"center_lon = {self.center_lon!r} is outside [-180, 180] "
                "degrees")
        if self.center_height_m < 0.0:
            raise ValueError(
                f"center_height_m = {self.center_height_m!r} is below the "
                "surface; the bubble center is metres AGL and must be "
                ">= 0")
        for name in ("radius_km", "depth_m", "amplitude_k"):
            if getattr(self, name) <= 0.0:
                raise ValueError(
                    f"{name} = {getattr(self, name)!r} must be positive; "
                    "a nonpositive bubble is a no-op wearing the name of "
                    "a perturbation")
        if self.amplitude_k > MAX_BUBBLE_AMPLITUDE_K:
            raise ValueError(
                f"amplitude_k = {self.amplitude_k!r} exceeds the "
                f"{MAX_BUBBLE_AMPLITUDE_K:g} K sanity bound: that is no "
                "longer an initiation nudge but a rewrite of the "
                "analysis. Lower the amplitude, or lift the bound with "
                "evidence.")
        if not isinstance(self.rh_preserve, bool):
            raise ValueError(
                f"rh_preserve must be a boolean, got {self.rh_preserve!r}")

    def receipt(self) -> dict:
        """Every accepted value, echoed in the shape it resolved to."""
        return {
            "center_lat": float(self.center_lat),
            "center_lon": float(self.center_lon),
            "center_height_m": float(self.center_height_m),
            "radius_km": float(self.radius_km),
            "depth_m": float(self.depth_m),
            "amplitude_k": float(self.amplitude_k),
            "rh_preserve": bool(self.rh_preserve),
        }


@dataclass(frozen=True)
class PerturbationConfig:
    """The validated [perturbation] block: one or more theta bubbles.

    Absent block = ``ExperimentConfig.perturbation is None`` = zero
    behavior change, byte-identical prepared state (the
    inflow_perturbation OFF discipline).  Present, the block is either
    honored on the experiment runtime path or refused by name on routes
    that do not thread it (:func:`refuse_unrouted_perturbation`); it is
    never dropped.
    """

    bubbles: tuple[BubbleConfig, ...]

    def __post_init__(self):
        if not self.bubbles:
            raise ValueError(
                "[perturbation] must carry at least one "
                "[[perturbation.bubbles]] entry; an empty block is "
                "either a stray table or a disabled feature wearing an "
                "enabled name -- delete the block to disable.")

    def receipt(self) -> dict:
        return {
            "schema": "gpuwm-initial-perturbation-v1",
            "bubbles": [bubble.receipt() for bubble in self.bubbles],
        }


def _build_perturbation(table, source: str) -> PerturbationConfig:
    """Validate [perturbation] / [[perturbation.bubbles]] of ``source``."""
    if not isinstance(table, dict):
        raise ValueError(
            f"[perturbation] of {source} must be a table carrying "
            f"[[perturbation.bubbles]] entries, got {table!r}.")
    _reject_unknown_keys("perturbation", table, _PERTURBATION_KEYS, source)
    _require_keys("perturbation", table, _PERTURBATION_KEYS, source)
    entries = table["bubbles"]
    if not isinstance(entries, list) or not entries or any(
            not isinstance(entry, dict) for entry in entries):
        raise ValueError(
            f"bubbles in [perturbation] of {source} must be an array of "
            "tables (one [[perturbation.bubbles]] block per bubble), got "
            f"{entries!r}.")
    bubbles = []
    for index, entry in enumerate(entries):
        label = f"perturbation.bubbles #{index + 1}"
        _reject_unknown_keys(label, entry, _BUBBLE_KEYS, source)
        _require_keys(label, entry, _BUBBLE_REQUIRED, source)
        try:
            bubbles.append(BubbleConfig(**entry))
        except ValueError as err:
            raise ValueError(
                f"[[{label}]] of {source}: {err}") from None
    return PerturbationConfig(bubbles=tuple(bubbles))


def refuse_unrouted_perturbation(exp, route: str) -> None:
    """Fail loud where a [perturbation] block would otherwise be dropped.

    The governance rule for this block is the same as for every key:
    honored or refused, never ignored.  Routes that do not thread the
    perturbation into their initialization call this immediately after
    loading the experiment.
    """
    if getattr(exp, "perturbation", None) is None:
        return
    raise ValueError(layered(
        f"the {route} route does not apply [perturbation] blocks; "
        "refused rather than ignored, because a dropped perturbation "
        "would run an unperturbed state under the name of your bubbles.",
        "Initial-state theta bubbles are applied by the experiment "
        "runtime path (gpuwm run / gpuwm ingest, through "
        "gpuwm.ingest.real.initialize_real) and by the prepared "
        "domain-tree forecast runner (applied to the restored states, "
        "gpuwm.prepared_domain_tree_forecast)."))


@dataclass(frozen=True)
class DomainConfig:
    """One domain of a nested experiment (architecture section A).

    ``run`` is the fully resolved per-domain :class:`RunConfig` carrying
    nx/ny/nz/dx/dy, the per-domain physics and diffusion selections, the
    ``specified``/``nested`` flags, and ``clock_dt = 0.0`` (retired in the
    experiment path); ``run.dt`` is the binary64 image of the CHAINED
    single-precision WRF REAL dt (``np.float32`` division down the ratio
    chain, share/set_timekeeping.F:368 -- for d04 1.6666666269302368,
    NOT ``float(dt_exact)``); the exact rational lives on
    :meth:`ExperimentConfig.dt_exact`.  (Stale wording fixed per the
    p5t9 adversarial review, finding 4.)

    The root domain (``parent_id == 0``) carries the WRF rational clock
    keys ``time_step`` (integer seconds) + ``time_step_fract_num`` /
    ``time_step_fract_den`` (Registry.EM_COMMON:2245-2246); children carry
    ``time_step = None`` and derive their dt from the ratio chain.
    """

    grid_id: int
    parent_id: int
    i_parent_start: int
    j_parent_start: int
    parent_grid_ratio: int
    parent_time_step_ratio: int
    history_interval_s: float
    run: RunConfig
    time_step: int | None = None
    time_step_fract_num: int = 0
    time_step_fract_den: int = 1
    start_time: datetime | None = None
    #: The validated ``spawn`` table (:class:`gpuwm.core.nest_spawn
    #: .SpawnConfig`), or ``None`` for an ordinary live domain.  ``None``
    #: is the OFF contract and is omitted from the restart-identity
    #: payload and tolerated by the prepared-cache identity, so every
    #: pre-feature fingerprint stays byte-identical.  Non-``None`` marks
    #: this domain DORMANT: declared and memory-reserved from startup,
    #: integrating nothing until the trigger fires; ``i_parent_start`` /
    #: ``j_parent_start`` are then the PLACEHOLDER placement (the memory
    #: plan's and the manual time-trigger's), and the fired placement is
    #: chosen at trigger time (:func:`active_experiment`).
    spawn: "object | None" = None
    #: This domain's own ``[tiles]`` road (:class:`gpuwm.core.streaming
    #: .StreamingOptions`), or ``None`` to take the tree-wide table.
    #: ``None`` is the inherit contract and drops out of the restart
    #: identity, so every experiment written before the per-domain
    #: surface existed keeps its exact fingerprint.
    #:
    #: Unlike ``spawn``, a DECLARED value binds nothing either: ``[tiles]``
    #: is an execution choice whose entire claim is that it changes no
    #: bytes, so a domain that streamed must resume resident and a domain
    #: that ran resident must resume streamed.  See
    #: ``streaming.identity_payload_entry``.
    tiles: "object | None" = None


@dataclass(frozen=True)
class ScheduledRelocationMove:
    """One row of the manual follow itinerary: when, and how far.

    The shift is in whole PARENT cells, the same unit and sign convention
    the tracker's plan provider returns, so the manual mode exercises the
    exact interface the tracker must satisfy.
    """

    at_seconds: float
    di_parent_cells: int = 0
    dj_parent_cells: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.at_seconds)) or float(
                self.at_seconds) <= 0.0:
            raise ValueError(
                f"at_seconds = {self.at_seconds!r} must be a finite, "
                "positive model time; t = 0 is initial placement, not a "
                "move")
        for name in ("di_parent_cells", "dj_parent_cells"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(
                    f"{name} = {value!r} must be a whole number of parent "
                    "cells; discrete relocation has no fractional moves")

    def to_json(self) -> dict[str, object]:
        return {"at_seconds": float(self.at_seconds),
                "di_parent_cells": int(self.di_parent_cells),
                "dj_parent_cells": int(self.dj_parent_cells)}


@dataclass(frozen=True)
class RelocationConfig:
    """Bounds and (optionally) a follow source for discrete relocation.

    Three layers, deliberately separable:

    * The MECHANISM bounds -- which child may move (``grid_id``), how far
      in one event (``max_move_parent_cells``), and how much ground it
      must keep (``min_overlap_fraction``).  A caller driving
      :func:`gpuwm.core.nest_relocation.relocate_child` directly uses
      only these.
    * The FOLLOW SOURCE -- at most one of: ``follow``, the validated
      ``[relocation.follow]`` storm-tracking block
      (:class:`gpuwm.core.storm_tracking.FollowConfig`, the plan-provider
      seam), or ``moves``, the manual ``[[relocation.move]]`` itinerary
      (testable without a tracker, through the SAME provider contract).
      Neither present means the runner schedules nothing and relocation
      stays a manual/API mechanism, exactly as leg 1 shipped it.
    * The CADENCE -- ``cadence_seconds`` names the cycle-boundary
      opportunities at which the follow source is consulted.  ``None``
      with a source configured means EVERY complete cycle boundary (the
      tracker's own cooldown/dead-band are then the rate limit).

    ``None`` on either bound means unbounded, which is admissible but is
    a deliberate choice a config has to make.
    """

    enabled: bool = False
    grid_id: int | None = None
    mode: str = DISCRETE_RELOCATION_MODE
    max_move_parent_cells: int | None = None
    min_overlap_fraction: float | None = None
    cadence_seconds: float | None = None
    #: The validated ``[relocation.follow]`` storm-tracking block
    #: (:class:`gpuwm.core.storm_tracking.FollowConfig`), or ``None`` when
    #: the config carries none.
    follow: "object | None" = None
    #: The manual itinerary (``[[relocation.move]]`` rows), or empty.
    moves: tuple[ScheduledRelocationMove, ...] = ()

    def __post_init__(self) -> None:
        if not self.enabled:
            if self.follow is not None:
                raise ValueError(
                    "[relocation.follow] on a disabled [relocation] block "
                    "is refused: a tracker with no mechanism to drive is a "
                    "config that half-opted-in; set enabled = true "
                    "deliberately, or delete the follow table")
            if self.moves:
                raise ValueError(
                    "[[relocation.move]] rows on a disabled [relocation] "
                    "block are refused: an itinerary with no mechanism to "
                    "drive is a config that half-opted-in; set enabled = "
                    "true deliberately, or delete the rows")
            return
        if self.mode != DISCRETE_RELOCATION_MODE:
            raise ValueError(
                f"relocation mode {self.mode!r} is not implemented; the "
                f"only mode is {DISCRETE_RELOCATION_MODE!r} (whole parent "
                "cells at cycle boundaries)")
        if self.grid_id is None:
            raise ValueError(
                "[relocation] with enabled = true must name the grid_id of "
                "the child that may move; a tree-wide 'something moves' is "
                "not a placement")
        if int(self.grid_id) < 2:
            raise ValueError(
                f"relocation grid_id = {self.grid_id!r} names the root "
                "domain or an invalid id; only a child can be relocated, "
                "because a placement is a position inside a parent")
        if (self.max_move_parent_cells is not None
                and int(self.max_move_parent_cells) < 1):
            raise ValueError(
                "max_move_parent_cells must be at least 1 parent cell; a "
                "relocation moves whole parent cells, and 0 is the null "
                "move, which needs no bound")
        if self.min_overlap_fraction is not None and not (
                0.0 <= float(self.min_overlap_fraction) <= 1.0):
            raise ValueError(
                "min_overlap_fraction is a fraction of the child's cells "
                f"and must lie in [0, 1], got {self.min_overlap_fraction!r}")
        if self.follow is not None and self.moves:
            raise ValueError(
                "[relocation.follow] and [[relocation.move]] are two "
                "follow sources; with both present one would be silently "
                "ignored, so both are refused -- keep the tracker or the "
                "manual itinerary, not both")
        cadence = self.cadence_seconds
        if cadence is not None:
            if self.follow is None and not self.moves:
                raise ValueError(
                    "cadence_seconds names how often the follow source is "
                    "consulted, and this [relocation] has no follow source "
                    "([relocation.follow] or [[relocation.move]]); add "
                    "one, or delete the key")
            if not math.isfinite(float(cadence)) or float(cadence) <= 0.0:
                raise ValueError(
                    f"cadence_seconds = {cadence!r} must be a finite, "
                    "positive number of model seconds")
        if self.moves:
            previous = 0.0
            for move in self.moves:
                at = float(move.at_seconds)
                if at <= previous:
                    raise ValueError(
                        "[[relocation.move]] rows must be strictly "
                        f"increasing in at_seconds; {at} follows {previous}")
                if cadence is not None:
                    multiples = at / float(cadence)
                    if abs(multiples - round(multiples)) > _REL_TOL * max(
                            1.0, abs(multiples)):
                        raise ValueError(
                            f"at_seconds = {at} is not a whole number of "
                            f"cadence_seconds = {float(cadence)}; a move "
                            "can only fire at a cadence opportunity")
                previous = at

    def receipt(self) -> dict[str, object]:
        """The accepted configuration, echoed value for value."""
        return {
            "enabled": bool(self.enabled),
            "grid_id": (None if self.grid_id is None else int(self.grid_id)),
            "mode": self.mode,
            "max_move_parent_cells": (
                None if self.max_move_parent_cells is None
                else int(self.max_move_parent_cells)),
            "min_overlap_fraction": (
                None if self.min_overlap_fraction is None
                else float(self.min_overlap_fraction)),
            "cadence_seconds": (
                None if self.cadence_seconds is None
                else float(self.cadence_seconds)),
            "follow": (None if self.follow is None
                       else self.follow.to_json()),
            "moves": [move.to_json() for move in self.moves],
        }


@dataclass(frozen=True, kw_only=True)
class ExperimentConfig:
    """A validated experiment: domains stored parent-before-child.

    ``vertical``/``projection`` are the F1-amendment value objects:
    frozen/hashable, serialized verbatim in the resolved-TOML round trip,
    and inside the section-B experiment fingerprint -- two experiments
    differing in any eta level or projection parameter compare (and
    hash) distinct.  ``projection`` is required for real TOMLs (shared
    ``map_proj`` nonzero, WRF convention 1=lambert/2=polar/3=mercator);
    ``None`` is reserved for wrapped idealized configs
    (:func:`experiment_from_run_config`).

    TIMING AUTHORITY (F14 amendment): ``run_seconds`` /
    ``restart_interval_s`` here and each domain's ``history_interval_s``
    are the ONLY authoritative timing values; the embedded per-domain
    RunConfig copies are derived and asserted equal at load.

    ``blend_width`` defaults to 5 per Registry.EM_COMMON:2324 (``rconfig
    integer blend_width ... 5 "width of cg fg terrain blended zone"``).
    ``feedback`` is a tree-wide switch: 0 is the production one-way path
    and 1 enables the experimental child-to-parent restriction path.
    ``smooth_option`` remains 0 because parent smoothing is not implemented.
    """

    name: str
    start_time: datetime
    run_seconds: float
    vertical: VerticalConfig
    projection: ProjectionConfig | None = None
    feedback: int = 0
    smooth_option: int = 0
    blend_width: int = 5
    spec_bdy_width: int = 5
    restart_interval_s: float
    domains: tuple[DomainConfig, ...]
    column_chunk: int = DEFAULT_COLUMN_CHUNK
    acknowledgements: tuple[str, ...] = ()
    #: Discrete-relocation admissibility bounds.  Default-disabled, so an
    #: experiment that never mentions [relocation] carries a static nest
    #: and is byte-for-byte the experiment it was before this existed.
    relocation: RelocationConfig = RelocationConfig()
    #: The resolved physics-fidelity axis (:mod:`gpuwm.physics_mode`).  The
    #: resolved values are ALSO baked into each domain's RunConfig, so the
    #: fingerprint would bind the physics without this field; it is carried
    #: anyway because a receipt that says "arwen-patched, patchset v1,
    #: patches [L4]" is readable and a pair of selector values is not.
    #: ``physics_mode.UNGOVERNED`` is the state of every configuration that
    #: does not name the axis, and it authors nothing.
    physics_mode: physics_mode_module.PhysicsModeResolution = (
        physics_mode_module.UNGOVERNED)
    #: The validated [perturbation] block, or ``None`` when the config
    #: does not carry one.  ``None`` is the OFF contract: no applier is
    #: built, no initialization branch runs, and the restart-identity
    #: payload omits the key entirely so absent-block fingerprints stay
    #: byte-identical to pre-feature ones.
    perturbation: "PerturbationConfig | None" = None
    #: The resolved [tiles] block (:mod:`gpuwm.core.streaming`).  An
    #: experiment that never mentions it carries ``StreamingOptions.OFF``,
    #: whose stepper IS ``dycore.step`` -- there is no disabled-streaming
    #: code path, only the absence of one.  Excluded from the restart
    #: identity on purpose: see ``streaming.identity_payload_entry``.
    tiles: "streaming_module.StreamingOptions" = streaming_module.OFF
    #: grid_ids whose ``mix_isotropic`` was CHOSEN BY THE MODEL because
    #: the config left it unset or wrote the ``"auto"`` sentinel (Drew's
    #: 2026-08-16 auto-switch ruling; ``resolve_auto_mix_isotropic``).
    #: A provenance LABEL, not a trajectory input: the chosen value sits
    #: on each domain's ``run.mix_isotropic``, which is what the restart
    #: identity binds, so this field leaves ``restart_identity_payload``
    #: and the sealed-extension identity unconditionally -- an
    #: auto-selected 1 and a written 1 are the same run, and each must
    #: resume the other's checkpoints.  The empty default keeps every
    #: code-constructed experiment (``experiment_from_run_config``, the
    #: frozen verify fixtures) on explicit semantics: what its author
    #: set is what runs.
    auto_mix_isotropic: tuple[int, ...] = ()

    def __post_init__(self):
        if self.feedback not in (0, 1):
            raise ValueError(
                "feedback must be 0 (one-way) or 1 (experimental "
                f"two-way), got {self.feedback!r}.")
        if self.smooth_option != 0:
            raise ValueError(
                "smooth_option must remain 0; parent smoothing is not "
                f"implemented, got {self.smooth_option!r}.")
        if (isinstance(self.column_chunk, bool)
                or not isinstance(self.column_chunk, int)
                or self.column_chunk < 1):
            raise ValueError(
                "column_chunk must be a positive integer number of "
                f"radiation columns, got {self.column_chunk!r}.")
        if (
            not isinstance(self.acknowledgements, tuple)
            or any(
                not isinstance(value, str) or not value.strip()
                for value in self.acknowledgements
            )
        ):
            raise ValueError(
                "acknowledgements must be a tuple of non-empty ids, got "
                f"{self.acknowledgements!r}.")

    @property
    def root(self) -> DomainConfig:
        return self.domains[0]

    def domain(self, grid_id: int) -> DomainConfig:
        for dc in self.domains:
            if dc.grid_id == grid_id:
                return dc
        raise KeyError(
            f"no domain with grid_id={grid_id}; experiment {self.name!r} "
            f"has {[dc.grid_id for dc in self.domains]}")

    def children_of(self, grid_id: int) -> tuple[DomainConfig, ...]:
        self.domain(grid_id)  # KeyError on unknown ids
        return tuple(dc for dc in self.domains if dc.parent_id == grid_id)

    def domain_start_time(self, grid_id: int) -> datetime:
        """Absolute valid time at which one configured domain becomes live.

        ``None`` is retained only for programmatic legacy fixtures created
        before per-domain starts entered the schema; it means the experiment
        start.  Every TOML-loaded domain carries an explicit resolved value.
        """
        value = self.domain(grid_id).start_time
        return self.start_time if value is None else value

    def domain_start_offset_exact(self, grid_id: int) -> Fraction:
        """Exact seconds from the experiment start to a domain start."""
        delta = self.domain_start_time(grid_id) - self.start_time
        return (Fraction(delta.days * 86400 + delta.seconds)
                + Fraction(delta.microseconds, 1_000_000))

    def dt_exact(self, grid_id: int) -> Fraction:
        """The domain's model step as an EXACT rational (seconds).

        Root: ``time_step + time_step_fract_num/time_step_fract_den``
        (Registry.EM_COMMON:2245-2246).  Child: the parent's rational dt
        divided by ``parent_time_step_ratio`` exactly, as WRF's
        ``stepTime = parent stepTime / parent_time_step_ratio``
        (share/set_timekeeping.F:366-368).  Bundle chain (1,4,3,3):
        60, 15, 5, 5/3 s.
        """
        dc = self.domain(grid_id)
        if dc.parent_id == 0:
            return (Fraction(dc.time_step)
                    + Fraction(dc.time_step_fract_num,
                               dc.time_step_fract_den))
        return self.dt_exact(dc.parent_id) / dc.parent_time_step_ratio

    def dx_exact(self, grid_id: int) -> Fraction:
        """The domain's grid spacing as an EXACT rational (metres):
        root ``dx``, children ``dx_parent / parent_grid_ratio`` exactly.
        Bundle chain (1,4,3,3): 12000, 3000, 1000, 1000/3 m."""
        dc = self.domain(grid_id)
        if dc.parent_id == 0:
            return Fraction(dc.run.dx)
        return self.dx_exact(dc.parent_id) / dc.parent_grid_ratio


def readable_config_path(path: str | Path) -> Path:
    """``path`` as a readable regular file, or a refusal saying which
    kind of thing it actually is.

    A mistyped config path is the commonest error there is, and in
    v1.4.0 every wrong-KIND of path reached ``open()`` and came back as
    a Python traceback at exit 1: ``FileNotFoundError`` for a typo,
    ``IsADirectoryError`` for a directory, an eight-argument
    ``TypeError: RunConfig.__init__()`` for a zero-byte file, ``OSError
    [Errno 40]`` for a symlink loop -- and, for a FIFO, no return at
    all, because ``tomllib`` reads it forever.  Every other refusal in
    this product is one sentence at exit 2.

    So the kind is decided BEFORE anything opens the path.  ``is_file()``
    is what separates a regular file from a directory, a FIFO, a device
    and a broken symlink in one call, and it is the guard the fleet's
    hostile-input node asked for by name.
    """

    path = Path(path)
    try:
        if path.is_file():
            if path.stat().st_size == 0:
                raise ValueError(layered(
                    f"{path} is empty, so there is no configuration in "
                    "it to run.\n"
                    "  remedy: gpuwm domain ... --out "
                    f"{path}   # re-author it",
                    "A zero-byte TOML parses to an empty table, which "
                    "used to reach the RunConfig constructor and come "
                    "back as its argument list."))
            return path
    except OSError as error:
        # A symlink loop, a permission wall, a dead mount: is_file()
        # itself is what failed, and its errno is the diagnosis.
        raise ValueError(
            f"{path} cannot be read as a configuration file "
            f"({error.strerror or error}).") from None
    if path.is_dir():
        kind = "is a directory"
    elif path.exists():
        kind = "is not a regular file (a device, socket or FIFO)"
    else:
        kind = "does not exist"
    raise ValueError(layered(
        f"{path} {kind}; pass the experiment .toml that `gpuwm domain` "
        "wrote.",
        "Nothing is opened until the path is known to be a readable "
        "regular file: a FIFO would block this process forever, and a "
        "directory or a missing file used to arrive as a Python "
        "traceback rather than as a refusal."))


def is_experiment_toml(path: str | Path) -> bool:
    """True when ``path`` is an experiment TOML (``[experiment]`` or
    ``[[domain]]`` present) rather than a legacy RunConfig TOML."""
    from gpuwm.config_authority import read_config_authority

    raw = tomllib.load(io.BytesIO(read_config_authority(path).payload))
    return "experiment" in raw or "domain" in raw


def is_experiment_toml_bytes(payload: bytes) -> bool:
    """Byte-oriented route detection for one already captured config."""

    raw = tomllib.load(io.BytesIO(payload))
    return "experiment" in raw or "domain" in raw


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment TOML (fail-loud, section A).

    The one-file case schema's companion tables are split off and
    VALIDATED here -- never silently dropped, and never refused as
    unknown.  ``[fetch]`` (advisory hints, schema owned by
    :func:`gpuwm.fetch.validate_fetch_hints`) always was; ``[case_data]``
    and ``[static]`` now are too, against their own owners' schemas
    (:func:`gpuwm.case_data.build_case_data` without the input-existence
    check -- this loader answers "what experiment is this", which must
    not require the declared inputs to be fetched yet -- and
    :func:`gpuwm.static.highres_production.parse_static_table`).  Before
    task #204 this loader refused the wizard's own ERA5 emission -- the
    exact file ``gpuwm domain --source era5`` writes -- as "does not
    have a table 'case_data'", from every front door that loads through
    here.  Callers that CONSUME the case declarations load through
    :func:`gpuwm.case_data.load_experiment_case` instead; the experiment
    schema itself stays strict.
    """
    from gpuwm.config_authority import read_config_authority

    authority = read_config_authority(path)
    raw = tomllib.load(io.BytesIO(authority.payload))
    source = str(authority.source)
    base_dir = Path(authority.source).parent
    fetch_table = raw.pop("fetch", None)
    if fetch_table is not None:
        from gpuwm.fetch import validate_fetch_hints
        validate_fetch_hints(fetch_table, source=source)
    case_table = raw.pop("case_data", None)
    if case_table is not None:
        from gpuwm.case_data import build_case_data
        build_case_data(case_table, source=source, base_dir=base_dir,
                        require_inputs=False)
    static_table = raw.pop("static", None)
    if static_table is not None:
        from gpuwm.static.highres_production import parse_static_table
        parse_static_table(static_table, source=source, base_dir=base_dir)
    ingest_table = raw.pop("ingest", None)
    if ingest_table is not None:
        from gpuwm.ingest.soil_downscale import parse_ingest_table
        parse_ingest_table(ingest_table, source=source)
    return build_experiment(raw, source=source)


def experiment_from_run_config(cfg: RunConfig,
                               start_time: datetime) -> ExperimentConfig:
    """Wrap a scalar :class:`RunConfig` as a one-domain experiment.

    The RunConfig is carried VERBATIM (no flag rewriting: a periodic
    idealized config wraps as-is -- the multi-domain root-flag rule binds
    only at TOML load).  ``cfg.dt`` decomposes exactly into the WRF
    rational clock keys: ``time_step`` integer seconds plus the exact
    binary remainder as ``time_step_fract_num/den``.
    """
    dt = Fraction(cfg.dt)
    whole = dt.numerator // dt.denominator
    rem = dt - whole
    dom = DomainConfig(
        grid_id=cfg.grid_id, parent_id=0, i_parent_start=1,
        j_parent_start=1, parent_grid_ratio=1, parent_time_step_ratio=1,
        history_interval_s=float(cfg.output_interval_s), run=cfg,
        time_step=int(whole), time_step_fract_num=rem.numerator,
        time_step_fract_den=rem.denominator, start_time=start_time)
    return ExperimentConfig(
        name=cfg.case or "run_config", start_time=start_time,
        run_seconds=float(cfg.run_seconds),
        # eta_levels/p_top are unknown to a scalar RunConfig (legacy
        # cases own their vertical grids); the hybrid selectors are the
        # compatibility copies.  projection = None is RESERVED for these
        # wrapped idealized/legacy configs (F1 amendment).
        vertical=VerticalConfig(eta_levels=(), p_top=0.0,
                                hybrid_opt=cfg.hybrid_opt, etac=cfg.etac),
        projection=None,
        restart_interval_s=float(cfg.restart_interval_s),
        domains=(dom,), spec_bdy_width=cfg.spec_bdy_width)


# ---------------------------------------------------------------------------
# Loader internals
# ---------------------------------------------------------------------------

def _reject_moving_nest_keys(table_name: str, entries: dict,
                             source: str) -> None:
    present = sorted(_MOVING_NEST_KEYS & set(entries))
    if present:
        raise ValueError(
            f"continuous moving-nest key(s) {present} in [{table_name}] of "
            f"{source} are rejected: they drive per-step nest motion, which "
            "invalidates the SINT donor index/weight tables inside the "
            "integration. A nest that follows weather is expressed here as "
            "DISCRETE relocation instead -- whole parent cells at cycle "
            "boundaries, donor tables rebuilt once per placement generation "
            "-- through the [relocation] table (enabled = true) and the "
            "gpuwm.core.nest_relocation primitive. Remove the key(s).")


def _build_relocation(raw: dict, source: str, domains,
                      run_seconds: float) -> RelocationConfig:
    """Validate ``[relocation]``, refusing every key while it is off."""
    if "relocation" not in raw:
        return RelocationConfig()
    table = dict(raw["relocation"])
    _reject_moving_nest_keys("relocation", table, source)
    _reject_unknown_keys("relocation", table, _RELOCATION_KEYS, source)
    enabled = table.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(
            f"enabled in [relocation] of {source} must be a boolean, got "
            f"{enabled!r}.")
    if not enabled:
        stray = sorted(set(table) - {"enabled"})
        if stray:
            raise ValueError(
                f"[relocation] of {source} carries {stray} while enabled "
                "is false or absent. A relocation surface that is off must "
                "be empty, so a nest cannot start moving because a block "
                "was inherited or a flag was flipped somewhere else; set "
                "enabled = true deliberately, or delete the key(s).")
        return RelocationConfig()
    domain_ids = [dc.grid_id for dc in domains]
    grid_id = table.get("grid_id")
    if grid_id is not None and int(grid_id) not in set(domain_ids):
        raise ValueError(
            f"grid_id = {grid_id!r} in [relocation] of {source} is not a "
            f"domain of this experiment (have {sorted(domain_ids)}).")
    rows = table.get("move", [])
    if not isinstance(rows, list):
        raise ValueError(
            f"move in [relocation] of {source} must be written as "
            "[[relocation.move]] array-of-tables rows, got "
            f"{type(rows).__name__}.")
    moves = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(
                f"[[relocation.move]] row {index} of {source} is not a "
                "table.")
        row = dict(row)
        _reject_unknown_keys(
            f"relocation.move[{index}]", row, _RELOCATION_MOVE_KEYS, source)
        _require_keys(f"relocation.move[{index}]", row, ("at_seconds",),
                      source)
        try:
            moves.append(ScheduledRelocationMove(
                at_seconds=float(row["at_seconds"]),
                di_parent_cells=row.get("di_parent_cells", 0),
                dj_parent_cells=row.get("dj_parent_cells", 0)))
        except ValueError as err:
            raise ValueError(
                f"[[relocation.move]] row {index} of {source}: {err}"
            ) from None
    follow = None
    if "follow" in table:
        if not isinstance(table["follow"], dict):
            raise ValueError(
                f"follow in [relocation] of {source} must be the "
                "[relocation.follow] TABLE (the storm-tracking block), got "
                f"{table['follow']!r}.")
        from gpuwm.core.storm_tracking import build_follow_config
        follow = build_follow_config(dict(table["follow"]), source)
    try:
        relocation = RelocationConfig(
            enabled=True,
            grid_id=(None if grid_id is None else int(grid_id)),
            mode=str(table.get("mode", DISCRETE_RELOCATION_MODE)),
            max_move_parent_cells=(
                None if table.get("max_move_parent_cells") is None
                else int(table["max_move_parent_cells"])),
            min_overlap_fraction=(
                None if table.get("min_overlap_fraction") is None
                else float(table["min_overlap_fraction"])),
            cadence_seconds=(
                None if table.get("cadence_seconds") is None
                else float(table["cadence_seconds"])),
            follow=follow,
            moves=tuple(moves))
    except ValueError as err:
        raise ValueError(f"[relocation] of {source}: {err}") from None
    # The runner fires at complete cycle boundaries (root steps, where
    # every parent-child pair is synchronized), so a cadence or a
    # scheduled move that does not land on one can never fire.  Refused
    # here BY NAME rather than discovered as an eternally stationary nest.
    root_dc = domains[0]
    root_dt = (Fraction(root_dc.time_step)
               + Fraction(root_dc.time_step_fract_num,
                          root_dc.time_step_fract_den))

    def _whole_root_steps(seconds: float) -> bool:
        steps = float(seconds) / float(root_dt)
        return abs(steps - round(steps)) <= _REL_TOL * max(1.0, abs(steps))

    if (relocation.cadence_seconds is not None
            and not _whole_root_steps(relocation.cadence_seconds)):
        raise ValueError(
            f"cadence_seconds = {relocation.cadence_seconds} in "
            f"[relocation] of {source} is not a whole number of root "
            f"steps (root dt = {float(root_dt)} s); relocations execute "
            "at parent-step boundaries, and this cadence never lands on "
            "one.")
    for move in relocation.moves:
        if not _whole_root_steps(move.at_seconds):
            raise ValueError(
                f"[[relocation.move]] at_seconds = {move.at_seconds} in "
                f"{source} is not a whole number of root steps (root dt "
                f"= {float(root_dt)} s); it can never fire.")
        if float(move.at_seconds) >= float(run_seconds):
            raise ValueError(
                f"[[relocation.move]] at_seconds = {move.at_seconds} "
                f"in {source} is at or past the end of the run "
                f"(run_seconds = {run_seconds}); it can never fire.")
    if relocation.follow is not None:
        _refuse_unservable_follow_cadence(
            relocation, domains, source, root_dt=root_dt)
    return relocation


def _refuse_unservable_follow_cadence(relocation, domains, source,
                                      *, root_dt) -> None:
    """The reflectivity stash must be able to serve every evaluation.

    The tracker's composite-reflectivity plane is not a diagnostic it can
    ask for on demand: ``refl_10cm`` is stashed by the microphysics
    drivers inside their ``refl_10cm_due`` branch, which follows the
    HISTORY cadence.  So an evaluation cadence that is not a whole
    multiple of the watched domain's ``history_interval_s`` asks for a
    plane at instants where it does not exist, and the run discovers
    that mid-flight as a ``TrackerRefusal`` -- at the first cadence where
    UH is under threshold and the echo fallback is consulted, which may
    be hours in and is exactly the moment a storm-following nest is
    supposed to be working.

    This is issue #111, and the contract is not new: the shipped
    ``moving_nest_20110427_follow_2km.toml`` states it in a comment above
    ``cadence_seconds`` and nothing enforced it.  Refused here, at
    admission, naming both knobs and the multiple that would work.

    It applies to ``field = "uh"`` as much as to ``field =
    "reflectivity"``: the echo handoff is automatic, not opt-in, so a
    UH-primary tracker whose cadence the stash cannot serve is a run that
    refuses the first time rotation is absent.

    The watched domain is the PARENT of ``grid_id``.  ``grid_id`` names
    the child that MOVES; ``RelocationRunner`` hands the provider
    ``node.parent.state``, so the stash whose cadence matters belongs to
    the parent.
    """

    by_id = {int(dc.grid_id): dc for dc in domains}
    try:
        child = by_id[int(relocation.grid_id)]
        parent = by_id[int(child.parent_id)]
        stash = float(parent.history_interval_s)
    except (KeyError, TypeError, ValueError, AttributeError):
        # A child with no resolvable parent, or a domain missing the
        # fields this reads.  Tree integrity and per-domain schema are
        # other validators' refusals; preempting them with a cadence
        # message would send the reader to the wrong knob.
        return
    if not math.isfinite(stash) or stash <= 0.0:
        return
    where = (f"history_interval_s = {stash} s on the domain the tracker "
             f"watches ([[domain]] grid_id = {int(parent.grid_id)}, the "
             f"parent of the relocating grid_id = "
             f"{int(relocation.grid_id)})")
    why = ("the tracker's composite-reflectivity signal is stashed by the "
           "microphysics at history cadence (gpuwm.core.refl), so a "
           "consultation off that cadence asks for a refl_10cm plane that "
           "does not exist and the run refuses mid-flight")
    if relocation.cadence_seconds is None:
        raise ValueError(
            f"[relocation] of {source} configures a [relocation.follow] "
            f"tracker but no cadence_seconds, which means EVERY complete "
            f"cycle boundary (root dt = {float(root_dt)} s), and {where} "
            f"cannot serve that: {why}. Set cadence_seconds to a whole "
            f"multiple of {stash} (the history interval itself, {stash}, "
            f"is the usual choice).")
    cadence = float(relocation.cadence_seconds)
    multiples = cadence / stash
    if abs(multiples - round(multiples)) > _REL_TOL * max(
            1.0, abs(multiples)):
        lower = max(1, int(multiples)) * stash
        raise ValueError(
            f"cadence_seconds = {cadence} in [relocation] of {source} is "
            f"not a whole multiple of {where}: {why}. Use a whole multiple "
            f"of {stash} (nearest below/above: {lower} / "
            f"{lower + stash}), or set that domain's history_interval_s "
            f"to a value {cadence} divides into.")


def _require_keys(table_name: str, entries: dict, required, source: str):
    missing = [key for key in required if key not in entries]
    if missing:
        raise ValueError(
            f"[{table_name}] of {source} is missing required key(s) "
            f"{missing}; present: {sorted(entries)}.")


def did_you_mean(key: str, known) -> str:
    """`` (did you mean 'dampcoef'?)`` for a near miss, else ``""``.

    Public because every schema in this package that refuses an unknown
    key owes the reader the same second half of the sentence.  A second
    copy of this three-liner in :mod:`gpuwm.case_data` would be a second
    cutoff to keep in step with this one.
    """

    close = difflib.get_close_matches(key, sorted(known), n=1, cutoff=0.7)
    return f" (did you mean {close[0]!r}?)" if close else ""


def _reject_axis_authored_keys(table_name: str, entries: dict,
                               resolution, source: str) -> None:
    """Refuse a key the physics-fidelity axis is already the author of.

    The axis is sugar over a resolved patch vector, and sugar that MERGES
    with a hand-written key is how a run ends up integrating a value nobody
    chose: two authors, one key, and a receipt that can name only one of
    them.  So the second author is refused by name, in either mode -- there
    is no "they happen to agree" exemption, because agreeing today is a
    property of the value, not of the configuration, and the ledger's
    faithful/patched edge can move under a new patch-set version while the
    hand-written value stays put.

    The remedy is composition, not a wider schema: a battery arm strips the
    governed keys and lets the ``[experiment]`` overlay write them
    (``gpuwm.physics_mode.governed_keys`` is the exact list to strip).
    """

    if not resolution.governed:
        return
    authored = {patch.key: patch for patch in resolution.resolved
                if patch.key}
    present = sorted(set(entries) & set(authored))
    if not present:
        return
    named = ", ".join(
        f"{key!r} (divergence-ledger {authored[key].entry_id}, resolved to "
        f"{authored[key].value!r})" for key in present)
    raise ValueError(layered(
        f"[{table_name}] of {source} sets {named}, but physics_mode = "
        f"{resolution.mode!r} is already the author of "
        f"{'that key' if len(present) == 1 else 'those keys'}; a key with "
        "two authors runs a value neither of them can be shown to have "
        "chosen.",
        "Remove the key(s) and let the axis write them, or remove "
        "physics_mode and configure the keys directly. The axis governs "
        f"{sorted(authored)} under physics_mode = {resolution.mode!r}, "
        f"patchset = {resolution.patchset!r}; "
        "gpuwm.physics_mode.governed_keys() is the same list for a "
        "composer that has to strip them."))


def _reject_unknown_keys(table_name: str, entries: dict, known,
                         source: str) -> None:
    """Refuse keys this schema does not know, naming each one.

    This used to warn and drop, on the reasoning that the required-key
    and value checks around it catch every misspelling that matters.
    They do not, and cannot: every key with a default is optional by
    construction, so ``damp_coef = 0.4`` passes every one of them and
    the run integrates with ``dampcoef`` at its built-in 0.2.  The
    fleet's hostile-input node caught exactly that -- one stderr line
    among a hundred, then three wrfout frames, 159 product images and
    ``forecast validity PASS`` at exit 0, computed from a coefficient
    the user never chose.

    Warn-not-block is for the run that will almost certainly work.  A
    dropped key is the other case: the answer is confident, complete,
    and not the answer that was asked for.  So it refuses, it names the
    key, and it names the key it thinks was meant.  The full known-key
    list is mechanism, so it sits behind ``--explain``.
    """
    unknown = sorted(set(entries) - set(known))
    if not unknown:
        return
    named = ", ".join(
        f"{key!r}{did_you_mean(key, known)}" for key in unknown)
    raise ValueError(layered(
        f"[{table_name}] of {source} does not have a key {named}; "
        "no key is ignored, because a dropped key runs a default "
        "under the name of your value.",
        f"Known [{table_name}] keys: {sorted(known)}."))


def _positive_int(table_name: str, key: str, value, source: str,
                  minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{key} in [{table_name}] of {source} must be an integer, "
            f"got {value!r}.")
    if value < minimum:
        raise ValueError(
            f"{key} in [{table_name}] of {source} must be >= {minimum}, "
            f"got {value}.")
    return value


def _mass_dims(dom: dict, grid_id, source: str) -> tuple[int, int]:
    """nx/ny mass dimensions from e_we/e_sn (staggered) or nx/ny.

    WRF's e_we/e_sn count STAGGERED u/v points; gpuwm dimensions count
    mass points, one fewer (the frozen real74 case represents the
    251 x 201 namelist domain as 250 x 200).
    """
    dims = []
    for stag_key, mass_key in (("e_we", "nx"), ("e_sn", "ny")):
        stag = dom.get(stag_key)
        mass = dom.get(mass_key)
        if stag is None and mass is None:
            raise ValueError(
                f"[[domain]] grid_id={grid_id} of {source} must carry "
                f"{stag_key} (staggered) or {mass_key} (mass points).")
        if stag is not None:
            stag = _positive_int("domain", stag_key, stag, source, 2)
            if mass is not None and mass != stag - 1:
                raise ValueError(
                    f"[[domain]] grid_id={grid_id} of {source} carries "
                    f"inconsistent dimensions: {stag_key}={stag} "
                    f"(staggered) implies {mass_key}={stag - 1} mass "
                    f"points but {mass_key}={mass} was supplied.")
            dims.append(stag - 1)
        else:
            dims.append(_positive_int("domain", mass_key, mass, source, 1))
    return dims[0], dims[1]


def _reject_misplaced_run_keys(dom: dict, grid_id, source: str) -> None:
    """A real run setting written into [[domain]] refuses, never drops.

    ``_reject_unknown_keys`` warns and drops, and for a typo or a
    leftover that is right.  A key that IS a ``RunConfig`` field is
    neither: the user wrote a setting this model really has, in a table
    that does not carry it, and dropping it silently would run the
    OTHER value -- the one in ``[shared]`` -- while the file on disk
    says otherwise.  That is a confidently-delivered wrong answer, and
    the one class of configuration mistake this loader refuses rather
    than warns about.

    Measured case: ``bl_pbl_physics`` on a ``[[domain]]`` table used to
    warn and drop, so a tree that named an experimental PBL closure on
    one nest ran the shared scheme on every nest and reported success.
    Nothing in the run receipt would have contradicted the file.
    """
    known = {f.name for f in fields(RunConfig)}
    misplaced = sorted(key for key in dom
                       if key in known and key not in _DOMAIN_KEYS)
    if misplaced:
        raise ValueError(
            f"key(s) {misplaced} on [[domain]] grid_id={grid_id} of "
            f"{source} are run settings that are NOT per domain: they "
            f"belong in [shared], where they apply to every domain of "
            f"the tree.  Refused rather than ignored -- a physics or "
            f"run key that parsed here and was dropped would run the "
            f"[shared] value while this file said otherwise.  Per-domain "
            f"keys are {sorted(_DOMAIN_RUN_OVERRIDES)}.")


def _reject_domain_vertical_keys(dom: dict, grid_id, source: str) -> None:
    """ANY per-domain vertical key is rejected (F1 amendment, §A).

    The vertical grid is single-sourced from the one
    ``ExperimentConfig.vertical`` -- vertical nesting is rejected by
    construction (WRF only invokes vertical nest machinery when a nest
    refines the vertical grid: ``if (nest%e_vert /= parent%e_vert)``
    guards init_domain_vert_nesting, share/mediation_integrate.F:666).
    """
    present = [key for key in _DOMAIN_VERTICAL_KEYS if key in dom]
    if present:
        raise ValueError(
            f"vertical key(s) {present} on [[domain]] grid_id={grid_id} "
            f"of {source} are rejected: the vertical grid is "
            "single-sourced from [shared] (ExperimentConfig.vertical) -- "
            "vertical nesting is rejected by construction (WRF only "
            "calls init_domain_vert_nesting when a nest refines the "
            "vertical grid, share/mediation_integrate.F:666).")


def _bad_mix_isotropic_sentinel(value, where: str, source: str) -> str:
    """The refusal for a string that is not the ``"auto"`` sentinel."""

    return (
        f"mix_isotropic = {value!r} in {where} of {source} must be 0 "
        f"(anisotropic mixing lengths), 1 (isotropic (dx*dy*dz)^(1/3)) "
        f"or the string \"{MIX_ISOTROPIC_AUTO}\" -- the same meaning as "
        f"leaving the key unset: the model selects isotropic where "
        f"mix_upper_bound*(dz_max/dx)^2 exceeds "
        f"{EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT} and the WRF-default "
        f"anisotropic form otherwise.")


def _cross_check(kind: str, grid_id, supplied: float, derived: Fraction,
                 chain: str, source: str) -> None:
    """Hand-typed child dx/dt cross-check: the namelist chain is
    authoritative; a mismatch is a hard error (kills the 333.33 m vs
    "500 m" prose confusion -- e.g. a hand-typed 500 m d04 against
    ratio 3 from 1 km)."""
    derived_f = float(derived)
    if not math.isfinite(float(supplied)) or abs(
            float(supplied) - derived_f) > _REL_TOL * abs(derived_f):
        raise ValueError(
            f"[[domain]] grid_id={grid_id} of {source} hand-types "
            f"{kind}={supplied!r} but the parent chain derives {kind} = "
            f"{derived} ({chain}): child {kind} is never hand-typed -- "
            "the namelist chain is authoritative; remove the key or fix "
            "the chain.")


def _check_cadence(label: str, seconds: Fraction, dt: Fraction, grid_id,
                   source: str) -> None:
    if seconds <= 0:
        return  # 0 = every step (WRF convention for radt/cudt/bldt)
    steps = seconds / dt
    if steps.denominator != 1:
        raise ValueError(
            f"{label} = {float(seconds):g} s on domain grid_id={grid_id} "
            f"of {source} is not a whole number of that domain's steps: "
            f"dt = {dt} s exactly, {seconds}/({dt}) = {steps}.")


def _check_run_length_on_step_grid(run_seconds: Fraction, dt: Fraction,
                                   grid_id, source: str) -> None:
    """Refuse a forecast length that is not a whole number of d01 steps.

    ``gpuwm.core.clock.resolve_clock`` has always enforced this -- runs
    end on a root-domain boundary -- but it enforces it at clock
    resolution, which every route reaches only after preparation.  So a
    half-step ``run_seconds`` passed ``gpuwm check`` and then died in an
    unhandled ``ValueError`` at forecast, exiting 1 through a traceback,
    while the identical arithmetic error on ``history_interval_s`` or
    ``restart_interval_s`` was refused here in one line at admission,
    exit 2, before anything was fetched or prepared.  Same error class,
    same config file, two different user experiences.

    This admits nothing new and refuses nothing that ran: it is the
    check ``resolve_clock`` already applies, moved to where its siblings
    live so the refusal arrives before the work rather than after it.
    """
    if run_seconds <= 0:
        return
    steps = run_seconds / dt
    if steps.denominator != 1:
        raise ValueError(
            f"run_seconds = {float(run_seconds):g} s in [experiment] of "
            f"{source} is not a whole number of root-domain "
            f"(grid_id={grid_id}) steps: dt = {dt} s exactly, "
            f"{run_seconds}/({dt}) = {steps}.  Runs end on a d01 "
            "boundary, so the forecast length must land on the step "
            "grid.")


def _check_whole_second_cadence(label: str, seconds: Fraction, grid_id,
                                source: str) -> None:
    """Refuse a publication cadence the file names cannot express.

    History and standalone-restart instants become file names and a
    ``Times`` record at whole-second resolution, and both publishers
    replace.  A quarter-second history interval is divisible by a
    quarter-second step, so it used to pass admission -- and then three
    distinct legal frames formatted to one name, each replacing the last,
    with no exception raised anywhere.  Whole seconds are the widest
    cadence the on-disk contract can name, so they are the widest cadence
    admitted; the alternative is a run that quietly loses its own output.
    """
    if seconds <= 0:
        return
    if seconds.denominator != 1:
        raise ValueError(
            f"{label} = {float(seconds):g} s on domain grid_id={grid_id} "
            f"of {source} is not a whole number of seconds. History and "
            "restart files are named, and their Times record written, to "
            "whole-second resolution, so distinct sub-second instants "
            "would alias onto one file name and the later one would "
            "replace the earlier.")


def validate_boundary_timing(
        exp: ExperimentConfig, boundary_interval_seconds: int, *,
        source: str = "boundary forcing") -> None:
    """Validate the structural hierarchy/forcing timing contract.

    There is no whole-hour requirement.  The decoded boundary cadence must
    be a positive integer number of seconds and an exact number of root
    steps, because the root Davies clock resets only at top-of-step interval
    seams.  A delayed child start must additionally be an exact parent-step
    boundary and an exact boundary-forcing seam.  History output cadence is
    independent and is deliberately absent from this contract.
    """
    if (isinstance(boundary_interval_seconds, bool)
            or not isinstance(boundary_interval_seconds, int)
            or boundary_interval_seconds <= 0):
        raise ValueError(
            "boundary_interval_seconds must be a positive integer number "
            f"of seconds for {source}, got {boundary_interval_seconds!r}.")
    interval = Fraction(boundary_interval_seconds)
    root_steps = interval / exp.dt_exact(exp.root.grid_id)
    if root_steps.denominator != 1:
        raise ValueError(
            f"boundary_interval_seconds = {boundary_interval_seconds} s "
            f"for {source} is not a whole number of root-domain steps: "
            f"d{exp.root.grid_id:02d} dt = "
            f"{exp.dt_exact(exp.root.grid_id)} s exactly, cadence/dt = "
            f"{root_steps}.")
    for dc in exp.domains[1:]:
        offset = exp.domain_start_offset_exact(dc.grid_id)
        parent_offset = exp.domain_start_offset_exact(dc.parent_id)
        parent_dt = exp.dt_exact(dc.parent_id)
        parent_steps = (offset - parent_offset) / parent_dt
        if parent_steps.denominator != 1:
            raise ValueError(
                f"delayed start_time for d{dc.grid_id:02d} "
                f"({exp.domain_start_time(dc.grid_id).isoformat()}; offset "
                f"{float(offset):g} s) is not aligned to its parent step "
                f"boundary: d{dc.parent_id:02d} dt = {parent_dt} s "
                f"exactly, (child-parent start offset)/dt = "
                f"{parent_steps}.")
        forcing_seams = offset / interval
        if forcing_seams.denominator != 1:
            raise ValueError(
                f"delayed start_time for d{dc.grid_id:02d} "
                f"({exp.domain_start_time(dc.grid_id).isoformat()}; offset "
                f"{float(offset):g} s) is not aligned to the "
                f"boundary-forcing cadence: boundary_interval_seconds = "
                f"{boundary_interval_seconds} s, offset/cadence = "
                f"{forcing_seams}.")


def _parent_before_child(domain_tables: list, source: str) -> list:
    """Reorder [[domain]] tables so parents precede their children.

    Declaration order carries no information the tree does not: when
    every table has integer grid_id/parent_id, a wave sort (roots, then
    children of placed domains, preserving declaration order inside a
    wave) recovers the only admissible order.  Anything unresolvable --
    a parent id naming no table, a cycle -- is left as declared for the
    loop's own checks to refuse by name.
    """

    try:
        ids = [int(dom["grid_id"]) for dom in domain_tables]
        parents = [int(dom["parent_id"]) for dom in domain_tables]
    except (KeyError, TypeError, ValueError):
        return domain_tables
    if len(set(ids)) != len(ids):
        return domain_tables
    # Already parent-before-child?  Then the declared order stands --
    # it is a valid order the author chose, and downstream receipts
    # key on domain sequence.
    seen: set[int] = set()
    ordered = True
    for index in range(len(domain_tables)):
        if parents[index] != 0 and parents[index] not in seen:
            ordered = False
            break
        seen.add(ids[index])
    if ordered:
        return domain_tables
    # Repair: emit the earliest declarable table each pass, preserving
    # declaration order among the tables that are ready.
    placed: set[int] = set()
    remaining = list(range(len(domain_tables)))
    order: list[int] = []
    while remaining:
        wave = [i for i in remaining
                if parents[i] == 0 or parents[i] in placed]
        if not wave:
            return domain_tables  # unresolvable; let the loop refuse
        order.extend(wave)
        placed.update(ids[i] for i in wave)
        remaining = [i for i in remaining if i not in wave]
    warn(f"[[domain]] tables in {source} were declared out of "
         "parent-before-child order; reordered by grid_id/parent_id")
    return [domain_tables[i] for i in order]


def build_experiment(raw: dict, source: str) -> ExperimentConfig:
    """Validate a parsed experiment TOML dict and build the config."""
    known_tables = ("experiment", "shared", "projection", "domain",
                    "relocation", "perturbation", "tiles")
    # [ingest] is INGEST POLICY, and it is validated-and-dropped HERE
    # rather than added to the companion list above.  The companion
    # tables declare INPUTS: dropping one loses a setting, so the caller
    # must consume it and reaching this seam with one present is a
    # routing defect worth refusing.  [ingest] declares neither an input
    # nor anything this builder consumes -- its value is read from the
    # config by whoever ingests (gpuwm.ingest.soil_downscale.
    # declared_soil_texture_downscale, and gpuwm.case_data's loader,
    # which puts it on CaseDataConfig) -- so requiring every one of the
    # dozen callers of this function to split it would only mean the
    # switch is unreachable from whichever door someone forgot.  It is
    # still VALIDATED on the way past: a typo refuses here, it does not
    # silently run the default under the name of your setting.
    if isinstance(raw, dict) and "ingest" in raw:
        from gpuwm.ingest.soil_downscale import parse_ingest_table
        raw = dict(raw)
        parse_ingest_table(raw.pop("ingest"), source=source)
    # Companion tables of the ONE-FILE case schema: real, documented
    # tables that belong to other owners (gpuwm.case_data, gpuwm.fetch,
    # gpuwm.static.highres_production) and are split off by every file
    # loader before this builder runs.  Reaching this dict-level seam
    # with one still present is a routing defect in the CALLER, and it
    # must never be reported as the table being unknown: task #204 was
    # exactly that -- the ERA5 adapter told a user their wizard-written
    # config "does not have a table 'case_data'" while the table sat in
    # the file, present and valid.
    companion_tables = ("case_data", "fetch", "static")
    present_companions = [name for name in raw if name in companion_tables]
    if present_companions:
        named = ", ".join(f"[{name}]" for name in present_companions)
        raise ValueError(layered(
            f"experiment config {source} carries {named}, which is part "
            "of the one-file case schema but was handed to the "
            "experiment-table builder unsplit.  The table is PRESENT and "
            "its name is valid; this loading path simply does not consume "
            "it, which is a defect in the calling code, not in the "
            "config.",
            "Every file loader splits the companion tables off first: "
            "gpuwm.experiment.load_experiment validates and detaches "
            "[case_data]/[fetch]/[static], and "
            "gpuwm.case_data.load_experiment_case does the same while "
            "also returning the case declarations.  A caller building "
            "from a raw dict must do the same before build_experiment."))
    unknown_tables = [name for name in raw if name not in known_tables]
    if unknown_tables:
        # A whole stray table is the same defect as a stray key, one
        # order of magnitude larger: `[dynamics]` is the LEGACY config's
        # table name, so pasting a familiar block into an experiment
        # config used to drop every setting in it behind one line.
        named = ", ".join(
            f"{name!r}{did_you_mean(name, known_tables)}"
            for name in unknown_tables)
        raise ValueError(layered(
            f"experiment config {source} does not have a table {named}; "
            "no table is ignored, because a dropped table runs defaults "
            "under the name of your settings.",
            f"Known tables: {list(known_tables)}. The [grid]/[dynamics]/"
            "[run] table names belong to the single-domain legacy config "
            "schema; an experiment config carries [experiment], [shared], "
            "[projection], one [[domain]] per nest, and optionally "
            "[perturbation] (initial-state theta bubbles)."))
    if "experiment" not in raw:
        raise ValueError(
            f"experiment config {source} must carry an [experiment] "
            "table.")
    domain_tables = raw.get("domain", [])
    if not isinstance(domain_tables, list) or not domain_tables:
        raise ValueError(
            f"experiment config {source} must carry at least one "
            "[[domain]] table (array of tables).")

    # ---- [experiment] ------------------------------------------------
    exp = raw["experiment"]
    _reject_moving_nest_keys("experiment", exp, source)
    _reject_unknown_keys("experiment", exp, _EXPERIMENT_KEYS, source)
    _require_keys("experiment", exp, _EXPERIMENT_REQUIRED, source)
    name = exp["name"]
    if not isinstance(name, str) or not name:
        raise ValueError(
            f"name in [experiment] of {source} must be a non-empty "
            f"string, got {name!r}.")
    start_time = exp["start_time"]
    if not isinstance(start_time, datetime) or start_time.tzinfo is not None:
        raise ValueError(
            f"start_time in [experiment] of {source} must be an "
            "offset-free TOML datetime (e.g. 1974-04-03T12:00:00), got "
            f"{start_time!r}.")
    run_seconds = float(exp["run_seconds"])
    if not math.isfinite(run_seconds) or run_seconds <= 0.0:
        raise ValueError(
            f"run_seconds in [experiment] of {source} must be a finite "
            f"positive duration in seconds, got {exp['run_seconds']!r}.")
    feedback = exp.get("feedback", 0)
    if feedback not in (0, 1):
        raise ValueError(
            f"feedback = {feedback!r} in [experiment] of {source} is "
            "rejected: feedback must be 0 (one-way) or 1 "
            "(experimental two-way child-to-parent restriction).")
    smooth_option = exp.get("smooth_option", 0)
    if smooth_option != 0:
        raise ValueError(
            f"smooth_option = {smooth_option!r} in [experiment] of "
            f"{source} is rejected: the parent smoother is not "
            "implemented, including for experimental feedback = 1; set "
            "smooth_option = 0.")
    blend_width = _positive_int("experiment", "blend_width",
                                exp.get("blend_width", 5), source, 0)
    spec_bdy_width = _positive_int("experiment", "spec_bdy_width",
                                   exp.get("spec_bdy_width", 5), source, 1)
    restart_interval_s = float(exp["restart_interval_s"])
    if not math.isfinite(restart_interval_s) or restart_interval_s < 0.0:
        raise ValueError(
            f"restart_interval_s in [experiment] of {source} must be a "
            "finite non-negative interval in seconds (0 disables restart "
            f"writing), got {exp['restart_interval_s']!r}.")
    column_chunk = _positive_int(
        "experiment", "column_chunk",
        exp.get("column_chunk", DEFAULT_COLUMN_CHUNK), source)
    raw_acknowledgements = exp.get("acknowledgements", [])
    if not isinstance(raw_acknowledgements, list):
        raise ValueError(
            f"acknowledgements in [experiment] of {source} must be an array "
            f"of non-empty ids, got {raw_acknowledgements!r}.")
    acknowledgements: list[str] = []
    for index, value in enumerate(raw_acknowledgements):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"acknowledgements[{index}] in [experiment] of {source} "
                f"must be a non-empty string, got {value!r}.")
        acknowledgements.append(value)

    # ---- the physics-fidelity axis (gpuwm/physics_mode.py) -------------
    # ABSENT is not the same state as present-and-"wrf-faithful".  Absent
    # authors nothing, which is what every configuration written before the
    # axis existed means and what keeps them byte-identical; present makes
    # the axis the author of every ledger key, in EITHER mode, which is what
    # lets one case file serve every battery arm through a two-line overlay.
    if "physics_mode" in exp or "patchset" in exp or "patches" in exp:
        if "physics_mode" not in exp:
            named = sorted(set(exp) & {"patchset", "patches"})
            raise ValueError(
                f"[experiment] of {source} sets {named} without "
                "physics_mode; a qualifier on the fidelity axis does not "
                "select it, and implying the patched mode from one would "
                "make the most consequential line of a config the one that "
                "is not written. Add physics_mode = "
                f"{physics_mode_module.PHYSICS_MODE_WRF_FAITHFUL!r} or "
                f"{physics_mode_module.PHYSICS_MODE_ARWEN_PATCHED!r}.")
        physics_mode = physics_mode_module.resolve(
            exp["physics_mode"], exp.get("patchset"), exp.get("patches"),
            source=source)
    else:
        physics_mode = physics_mode_module.UNGOVERNED

    # ---- [projection] ------------------------------------------------
    projection = None
    if "projection" in raw:
        proj = raw["projection"]
        _reject_unknown_keys("projection", proj, _PROJECTION_KEYS, source)
        _require_keys("projection", proj, _PROJECTION_KEYS, source)
        try:
            projection = ProjectionConfig(
                map_proj=str(proj["map_proj"]).lower(),
                ref_lat=float(proj["ref_lat"]),
                ref_lon=float(proj["ref_lon"]),
                truelat1=float(proj["truelat1"]),
                truelat2=float(proj["truelat2"]),
                stand_lon=float(proj["stand_lon"]))
        except ValueError as err:
            raise ValueError(f"[projection] of {source}: {err}") from None

    # ---- [perturbation] ----------------------------------------------
    # ABSENT authors nothing: perturbation = None, byte-identical prepared
    # state, and the restart-identity payload omits the key so every
    # pre-feature fingerprint is preserved (gpuwm/core/model.py).
    perturbation = None
    if "perturbation" in raw:
        perturbation = _build_perturbation(raw["perturbation"], source)

    # ---- [tiles] ---------------------------------------------------
    # ABSENT is the OFF contract, and it is the shared StreamingOptions.OFF
    # object rather than a fresh one: the mode is an EXECUTION choice, it
    # authors no computed value, and
    # gpuwm.core.streaming.identity_payload_entry deliberately contributes
    # nothing to the restart identity -- a checkpoint written resident must
    # resume streamed and one written streamed must resume resident, which
    # is the operation the mode exists for.  Proven in four legs across a
    # file, bit-exact in all of them.
    tiles = streaming_module.StreamingOptions.from_mapping(
        raw.get("tiles"), source=source)

    # ---- [shared] ------------------------------------------------------
    shared = dict(raw.get("shared", {}))
    # Captured from the RAW table, before anything defaults it: whether
    # the author CHOSE radiation at all.  ra_physics defaults to 0 and
    # ra_lw/sw_physics to -1, so by RunConfig time "wrote ra_physics = 0"
    # and "wrote nothing" are the same object -- and the radiation-off
    # land-surface refusal has to tell those two readers apart.  Same
    # idiom, and the same reason, as ``declared_map_proj`` below.
    from gpuwm.physics_compat import declared_radiation_selectors
    declared_radiation = declared_radiation_selectors(shared)
    _reject_moving_nest_keys("shared", shared, source)
    run_field_names = {f.name for f in fields(RunConfig)}
    for key, where in _SHARED_FORBIDDEN.items():
        if key in shared:
            raise ValueError(
                f"key {key!r} in [shared] of {source} is not a shared "
                f"key; it belongs in {where}.")
    shared_known = ((run_field_names - set(_SHARED_FORBIDDEN))
                    | {"e_vert", "eta_levels", "p_top"}
                    | set(_GUARD_DEFAULTS))
    _reject_unknown_keys("shared", shared, shared_known, source)
    _reject_axis_authored_keys("shared", shared, physics_mode, source)

    # mix_isotropic's "auto" sentinel (Drew's 2026-08-16 auto-switch
    # ruling): the string means the same as leaving the key unset -- the
    # model chooses -- so it is stripped HERE and its absence is what the
    # domain loop below reads.  Any other string is refused by name;
    # integers flow through to validate_run_config's 0/1 battery
    # untouched, so every explicit config keeps its exact meaning.
    if isinstance(shared.get("mix_isotropic"), str):
        if shared["mix_isotropic"] != MIX_ISOTROPIC_AUTO:
            raise ValueError(_bad_mix_isotropic_sentinel(
                shared["mix_isotropic"], "[shared]", source))
        del shared["mix_isotropic"]

    for key, default in _GUARD_DEFAULTS.items():
        value = shared.pop(key, default)
        if value == default:
            continue
        if key == "interp_method_type":
            raise ValueError(
                f"interp_method_type = {value!r} in [shared] of {source} "
                "is rejected: only SINT (interp_method_type = 2, the WRF "
                "default per Registry.EM_COMMON:2301) is implemented; "
                "bilinear (1), nearest-neighbour (3) and quadratic (4) "
                "are not.")
        if key == "nest_interp_coord":
            raise ValueError(
                f"nest_interp_coord = {value!r} in [shared] of {source} "
                "is rejected: isobaric nest re-interpolation "
                "(nest_interp_coord = 1) is not implemented; only the "
                "standard eta-level interpolation (0).")
        if key == "vert_refine_method":
            raise ValueError(
                f"vert_refine_method = {value!r} in [shared] of {source} "
                "is rejected: vertical nest refinement is not "
                "implemented -- e_vert/eta_levels/p_top must be identical "
                "across domains (WRF only calls init_domain_vert_nesting "
                "when refining, share/mediation_integrate.F:666).")
        raise ValueError(
            f"input_from_hires = {value!r} in [shared] of {source} is "
            "rejected: high-resolution child terrain input is not "
            "implemented (children SINT + blend the parent terrain, "
            "dyn_em/nest_init_utils.F:712-785).")

    # Accept either gpuwm's mass-level count (nz), WRF's full-interface count
    # (e_vert), or derive both from an explicit eta array.  Every combination
    # is cross-checked before a RunConfig is constructed; no fixed vertical
    # profile is injected when the user omits one spelling.
    eta_levels = tuple(float(v) for v in shared.pop("eta_levels", ()))
    supplied_nz = shared.get("nz")
    supplied_e_vert = shared.pop("e_vert", None)
    if supplied_nz is None and supplied_e_vert is None and not eta_levels:
        raise ValueError(
            f"[shared] of {source} must carry nz or e_vert, or provide "
            "explicit eta_levels from which the level count can be derived.")
    nz = (None if supplied_nz is None else
          _positive_int("shared", "nz", supplied_nz, source, 4))
    e_vert = (None if supplied_e_vert is None else
              _positive_int("shared", "e_vert", supplied_e_vert, source, 5))
    if nz is not None and e_vert is not None and e_vert != nz + 1:
        raise ValueError(
            f"inconsistent vertical counts in [shared] of {source}: "
            f"nz={nz} mass levels require e_vert={nz + 1}, got {e_vert}.")
    if nz is None:
        nz = (e_vert - 1 if e_vert is not None else len(eta_levels) - 1)
    if e_vert is None:
        e_vert = nz + 1
    if eta_levels and len(eta_levels) != e_vert:
        raise ValueError(
            f"eta_levels in [shared] of {source} has "
            f"{len(eta_levels)} entries but nz = {nz} mass levels "
            f"require e_vert = nz + 1 = {e_vert} full levels.")
    shared["nz"] = nz
    if "ztop" not in shared:
        raise ValueError(
            f"[shared] of {source} must carry ztop (RunConfig requires "
            "it for the vertical grid scaffold).")

    # F1 amendment: nz/eta_levels/p_top/hybrid selectors resolve into the
    # ONE VerticalConfig (its __post_init__ owns the finiteness/ordering
    # invariants); hybrid_opt/etac additionally flow into each RunConfig
    # as derived compatibility copies, asserted equal below.
    try:
        vertical = VerticalConfig(
            eta_levels=eta_levels,
            p_top=float(shared.pop("p_top", 0.0)),
            hybrid_opt=shared.get("hybrid_opt", RunConfig.hybrid_opt),
            etac=float(shared.get("etac", RunConfig.etac)))
    except ValueError as err:
        raise ValueError(f"[shared] of {source}: {err}") from None

    # [projection] is required for real experiments and only meaningful
    # there: a nonzero [shared] map_proj (WRF convention: 1 = Lambert,
    # 2 = polar stereographic, 3 = Mercator) <=> a [projection] table,
    # and the integer must agree with the table's WPS string.
    map_proj = shared.get("map_proj", RunConfig.map_proj)
    if map_proj != 0 and projection is None:
        raise ValueError(
            f"[shared] of {source} sets map_proj = {map_proj} but no "
            "[projection] table is present; real experiments must carry "
            "the projection parameter set (F1 amendment).")
    # The [projection] table carries the full parameter set, so a config
    # that simply omits the [shared] integer is completed from the table
    # and nothing is contradicted -- the user wrote one projection, once.
    #
    # A config that DECLARES the integer and disagrees with the table is a
    # different thing, and it used to be a one-line warning that then ran
    # the table's projection anyway.  That is a wrong answer, not an
    # inconvenience: writing map_proj = 3 beside a "lambert" table gave a
    # complete Mercator-labelled request integrated on a Lambert grid, at
    # exit 0 with `gpuwm check` PASS.  Nothing downstream can recover the
    # discarded intent, because shared["map_proj"] is overwritten here.
    # Warn-not-block covers reports we are confident about; it does not
    # cover "your configuration says two different things and I picked
    # one".  Its sibling checks already agree -- a nonzero map_proj with
    # no table raises four lines above, and native_wrf_contract.py raises
    # on this same disagreement between a WPS namelist and the table.
    declared_map_proj = "map_proj" in shared
    if projection is not None and not declared_map_proj:
        map_proj = projection.wrf_map_proj
        shared["map_proj"] = map_proj
    elif projection is not None and map_proj != projection.wrf_map_proj:
        raise ValueError(
            f"[shared] map_proj = {map_proj} in {source} contradicts the "
            f"[projection] table, which selects {projection.map_proj!r} "
            f"(WRF code {projection.wrf_map_proj}).  No value is chosen "
            f"for you, because the one that would be dropped is the one "
            f"naming the projection your grid is defined on.  Set "
            f"[shared] map_proj = {projection.wrf_map_proj} to keep "
            f"{projection.map_proj!r}, or change [projection] map_proj to "
            f"the projection you meant; removing the [shared] key "
            f"entirely also works, and lets the table speak alone.")

    spec_zone = shared.get("spec_zone", RunConfig.spec_zone)
    relax_zone = shared.get("relax_zone", RunConfig.relax_zone)
    if spec_bdy_width < spec_zone + relax_zone:
        raise ValueError(
            f"spec_bdy_width = {spec_bdy_width} in [experiment] of "
            f"{source} must cover spec_zone + relax_zone = "
            f"{spec_zone} + {relax_zone}.")

    # ---- [[domain]] ----------------------------------------------------
    # Declaration order used to be a refusal ("parent-before-child").
    # The tree is fully declared by grid_id/parent_id, so when every id
    # parses, the loader orders the tables itself and says so; only a
    # genuinely unresolvable tree (unknown parent, cycle) still refuses,
    # via the existing checks inside the loop.
    domain_tables = _parent_before_child(domain_tables, source)
    domains: list[DomainConfig] = []
    auto_mix_ids: list[int] = []
    by_id: dict[int, DomainConfig] = {}
    dt_by_id: dict[int, Fraction] = {}
    dx_by_id: dict[int, Fraction] = {}
    fp32_by_id: dict[int, np.float32] = {}
    for index, dom in enumerate(domain_tables):
        _reject_moving_nest_keys("domain", dom, source)
        _reject_domain_vertical_keys(dom, dom.get("grid_id", index + 1),
                                     source)
        _reject_misplaced_run_keys(dom, dom.get("grid_id", index + 1),
                                   source)
        _reject_unknown_keys("domain", dom, _DOMAIN_KEYS, source)
        _reject_axis_authored_keys(
            f"domain grid_id={dom.get('grid_id', index + 1)}",
            dom, physics_mode, source)
        _require_keys(f"domain #{index + 1}", dom, _DOMAIN_REQUIRED, source)
        grid_id = _positive_int("domain", "grid_id", dom["grid_id"], source)
        if grid_id in by_id:
            raise ValueError(
                f"duplicate grid_id = {grid_id} in [[domain]] tables of "
                f"{source}.")
        # This domain's mix_isotropic provenance: AUTO when neither this
        # table nor [shared] writes an integer for it (a per-domain
        # "auto" string overrides a [shared] integer, and is stripped
        # here so only resolved integers ever reach RunConfig).
        if isinstance(dom.get("mix_isotropic"), str):
            if dom["mix_isotropic"] != MIX_ISOTROPIC_AUTO:
                raise ValueError(_bad_mix_isotropic_sentinel(
                    dom["mix_isotropic"],
                    f"[[domain]] grid_id={grid_id}", source))
            dom = dict(dom)
            del dom["mix_isotropic"]
            mix_isotropic_auto = True
        else:
            mix_isotropic_auto = ("mix_isotropic" not in dom
                                  and "mix_isotropic" not in shared)
        parent_id = dom["parent_id"]
        is_root = parent_id == 0
        if index == 0 and not is_root:
            raise ValueError(
                f"the first [[domain]] of {source} must be the root "
                f"(parent_id = 0), got parent_id = {parent_id!r} on "
                f"grid_id = {grid_id} (domains are stored "
                "parent-before-child).")
        if index > 0 and is_root:
            raise ValueError(
                f"exactly one root domain (parent_id = 0) is allowed; "
                f"grid_id = {grid_id} of {source} is a second root.")
        if not is_root and parent_id not in by_id:
            raise ValueError(
                f"parent_id = {parent_id!r} of grid_id = {grid_id} in "
                f"{source} does not name a previously declared domain "
                f"(domains must be listed parent-before-child); declared "
                f"so far: {sorted(by_id)}.")

        domain_start = dom.get("start_time", start_time)
        if (not isinstance(domain_start, datetime)
                or domain_start.tzinfo is not None):
            raise ValueError(
                f"start_time on [[domain]] grid_id = {grid_id} of {source} "
                "must be an offset-free TOML datetime, got "
                f"{domain_start!r}.")
        if is_root and domain_start != start_time:
            warn(f"root [[domain]] start_time = "
                 f"{domain_start.isoformat()} in {source} is ignored; "
                 f"[experiment].start_time = {start_time.isoformat()} "
                 "is authoritative")
            domain_start = start_time
        # Construct through integer microseconds so the exact offset check
        # below cannot inherit a floating duration.
        run_microseconds = Fraction(str(run_seconds)) * 1_000_000
        if run_microseconds.denominator != 1:
            raise ValueError(
                f"run_seconds = {run_seconds!r} in {source} cannot be "
                "represented on Python's microsecond datetime lattice.")
        stop_time = start_time + timedelta(
            microseconds=int(run_microseconds))
        if domain_start < start_time or domain_start >= stop_time:
            raise ValueError(
                f"start_time = {domain_start.isoformat()} on d{grid_id:02d} "
                f"of {source} must lie in the experiment window "
                f"[{start_time.isoformat()}, {stop_time.isoformat()}).")
        if not is_root:
            parent_start = by_id[parent_id].start_time
            if parent_start is None:
                parent_start = start_time
            if domain_start < parent_start:
                raise ValueError(
                    f"start_time = {domain_start.isoformat()} on "
                    f"d{grid_id:02d} of {source} precedes parent "
                    f"d{parent_id:02d} start_time = "
                    f"{parent_start.isoformat()}.")
            delta = domain_start - parent_start
            offset = (Fraction(delta.days * 86400 + delta.seconds)
                      + Fraction(delta.microseconds, 1_000_000))
            parent_steps = offset / dt_by_id[parent_id]
            if parent_steps.denominator != 1:
                raise ValueError(
                    f"delayed start_time for d{grid_id:02d} "
                    f"({domain_start.isoformat()}; offset "
                    f"{float(offset):g} s) is not aligned to its parent "
                    f"step boundary: d{parent_id:02d} dt = "
                    f"{dt_by_id[parent_id]} s exactly, "
                    f"(child-parent start offset)/dt = "
                    f"{parent_steps}.")
            # Task #205: delayed nest activation passed every gate here
            # and then deterministically killed the run at the activation
            # epoch, hours of integration in.  Ordering is deliberate --
            # the structural refusals above (window, parent precedence,
            # step alignment) stay first, because those configs remain
            # invalid even once delayed activation gains its
            # activation-epoch stash and this refusal is lifted.  A
            # domain that also declares spawn is left for the spawn
            # block's own refusal below ("activation time belongs to its
            # trigger"), which names that conflict more precisely than
            # this one can.
            if domain_start != start_time and "spawn" not in dom:
                raise ValueError(layered(
                    "delayed nest activation is refused: start_time = "
                    f"{domain_start.isoformat()} on d{grid_id:02d} of "
                    f"{source} is later than [experiment].start_time = "
                    f"{start_time.isoformat()}, and a run that accepts "
                    "it dies at the activation epoch, where the child's "
                    "first history frame is due before any microphysics "
                    "step has stashed its microphysics-time REFL_10CM "
                    "field. Start every [[domain]] at "
                    "[experiment].start_time (remove this start_time "
                    "key).",
                    "The tree history writer consumes a "
                    "microphysics-time REFL_10CM stash for every frame "
                    "after the experiment's own t = 0 "
                    "(gpuwm.runtime._submit_tree_history_frame -> "
                    "gpuwm.core.refl.consume_refl_10cm), and the stash "
                    "is produced only inside a microphysics step of the "
                    "same domain.  A delayed child's first history frame "
                    "is due AT its activation epoch, before any of its "
                    "steps has run, so the consume raises 'REFL_10CM "
                    "output is due but no microphysics-time field is "
                    "stashed' and the run dies there, deterministically. "
                    "Refusing upfront replaces accepting the config and "
                    "burning the run; a config whose domains all start "
                    "together at [experiment].start_time is unaffected."))

        # --- spawn: the dormant-nest declaration ------------------------
        spawn_cfg = None
        if "spawn" in dom:
            if is_root:
                raise ValueError(
                    f"spawn on the root [[domain]] grid_id = {grid_id} of "
                    f"{source} is refused: a spawn materializes a CHILD "
                    "inside its parent at trigger time, and the root has "
                    "no parent to be placed in.")
            if not isinstance(dom["spawn"], dict):
                raise ValueError(
                    f"spawn on [[domain]] grid_id = {grid_id} of {source} "
                    "must be an inline TABLE, e.g. spawn = { trigger = "
                    f"\"uh\", ... }}, got {dom['spawn']!r}.")
            if "start_time" in dom:
                raise ValueError(
                    f"[[domain]] grid_id = {grid_id} of {source} declares "
                    "both spawn and start_time; a dormant nest's "
                    "activation time belongs to its trigger, and a "
                    "delayed start beside it would be a second author of "
                    "the same instant. Remove one.")
            from gpuwm.core.nest_spawn import build_spawn_config
            spawn_cfg = build_spawn_config(
                dict(dom["spawn"]), source, grid_id=grid_id)

        # --- tiles: this domain's own road -------------------------------
        # Validated by the SAME parser the tree-wide table uses, so one
        # vocabulary of keys, one set of value refusals, one place a new
        # knob is added.  A domain that says nothing takes the tree-wide
        # table; a domain that speaks overrides it entirely rather than
        # merging key-by-key, because a half-inherited tiling ("mode from
        # the tree, store from the domain") is a configuration nobody can
        # read off the file.
        domain_tiles = None
        if "tiles" in dom:
            if not isinstance(dom["tiles"], dict):
                raise ValueError(
                    f"tiles on [[domain]] grid_id = {grid_id} of {source} "
                    "must be an inline TABLE, e.g. tiles = { mode = "
                    f"\"auto\" }}, got {dom['tiles']!r}.")
            # The two budget keys name a CARD, not a domain, and the tree
            # decision subtracts every domain's claim from one number.
            # Per-domain copies of it would be several answers to "how
            # much VRAM is there", and the walk would have to pick one --
            # so they are refused here, where the fix is obvious, rather
            # than silently overridden where it is not.
            card_keys = sorted({"vram_budget_bytes", "host_budget_bytes"}
                               & set(dom["tiles"]))
            if card_keys:
                raise ValueError(
                    f"tiles on [[domain]] grid_id = {grid_id} of {source} "
                    f"sets {card_keys}, which name the CARD and not this "
                    "domain: the tree decision prices every domain against "
                    "one budget, so a per-domain copy would be a second "
                    "answer to how much VRAM there is.  Set them on the "
                    "tree-wide [tiles] table instead; the per-domain table "
                    "chooses this domain's ROAD.")
            domain_tiles = streaming_module.StreamingOptions.from_mapping(
                dict(dom["tiles"]),
                source=f"[[domain]] grid_id = {grid_id} of {source}")

        # Flags: root takes external specified LBCs, children are nested
        # -- WRF &bdy_control, bundle namelist.input specified=T,F,F,F /
        # nested=F,T,T,T; mutually exclusive by construction.
        specified = dom.get("specified", is_root)
        nested = dom.get("nested", not is_root)
        if is_root and (specified is not True or nested is not False):
            warn(f"root domain grid_id = {grid_id} of {source}: "
                 f"specified = {specified!r}, nested = {nested!r} "
                 "corrected to specified = true, nested = false",
                 why="WRF &bdy_control: the head grid takes external "
                     "lateral boundaries; the flags derive from "
                     "parent_id and are never a real choice.")
            specified, nested = True, False
        if not is_root and (nested is not True or specified is not False):
            warn(f"child domain grid_id = {grid_id} of {source}: "
                 f"specified = {specified!r}, nested = {nested!r} "
                 "corrected to nested = true, specified = false",
                 why="WRF &bdy_control: children are forced by their "
                     "parent (bundle specified=T,F,F,F / nested=F,T,T,T); "
                     "the flags derive from parent_id and are never a "
                     "real choice.")
            specified, nested = False, True

        ratio = _positive_int("domain", "parent_grid_ratio",
                              dom["parent_grid_ratio"], source)
        tratio = _positive_int("domain", "parent_time_step_ratio",
                               dom["parent_time_step_ratio"], source)
        i_start = _positive_int("domain", "i_parent_start",
                                dom["i_parent_start"], source)
        j_start = _positive_int("domain", "j_parent_start",
                                dom["j_parent_start"], source)
        if is_root:
            corrected = [key for key, value in
                         (("parent_grid_ratio", ratio),
                          ("parent_time_step_ratio", tratio),
                          ("i_parent_start", i_start),
                          ("j_parent_start", j_start)) if value != 1]
            if corrected:
                warn(f"root domain grid_id = {grid_id} of {source}: "
                     f"{', '.join(corrected)} corrected to 1 (a root has "
                     "no parent, so these keys carry no information)")
                ratio = tratio = i_start = j_start = 1
        else:
            for key, value in (("parent_grid_ratio", ratio),
                               ("parent_time_step_ratio", tratio)):
                if value < 2:
                    raise ValueError(
                        f"{key} = {value} on child domain grid_id = "
                        f"{grid_id} of {source} must be >= 2 (a ratio-1 "
                        "child is not a refinement).")

        nx, ny = _mass_dims(dom, grid_id, source)

        # --- exact-rational dt/dx (never hand-typed on children) -------
        if is_root:
            if "time_step" not in dom:
                raise ValueError(
                    f"the root [[domain]] grid_id = {grid_id} of {source} "
                    "must carry time_step as integer seconds (plus "
                    "optional time_step_fract_num/time_step_fract_den, "
                    "Registry.EM_COMMON:2245-2246).")
            time_step = _positive_int("domain", "time_step",
                                      dom["time_step"], source, 0)
            fract_num = _positive_int(
                "domain", "time_step_fract_num",
                dom.get("time_step_fract_num", 0), source, 0)
            fract_den = _positive_int(
                "domain", "time_step_fract_den",
                dom.get("time_step_fract_den", 1), source)
            dt_ex = Fraction(time_step) + Fraction(fract_num, fract_den)
            if dt_ex <= 0:
                raise ValueError(
                    f"the root [[domain]] grid_id = {grid_id} of "
                    f"{source} resolves a non-positive model step: "
                    f"time_step = {time_step} + {fract_num}/{fract_den} "
                    f"s.")
            # A root `dt` used to warn "is ignored" and be discarded with
            # no comparison at all, so `dt = 60.0` beside `time_step = 5`
            # ran a 5 s step at exit 0 -- and the SAME warning was
            # emitted when the two agreed, which made it noise on the
            # harmless case and silent in effect on the harmful one.
            # Both spellings are the user's and they name one quantity,
            # so the disagreement is checked exactly the way a child's
            # hand-typed dt already is, ten lines below.
            if "dt" in dom:
                _cross_check(
                    "dt", grid_id, dom["dt"], dt_ex,
                    f"time_step {time_step} + {fract_num}/{fract_den} s "
                    "on this root domain", source)
            # WRF-REAL head-grid dt: single-precision evaluation of the
            # rational namelist keys (Registry.EM_COMMON:2245-2246).
            dt_fp32 = (np.float32(time_step)
                       + np.float32(fract_num) / np.float32(fract_den))
            if "dx" not in dom:
                raise ValueError(
                    f"the root [[domain]] grid_id = {grid_id} of {source} "
                    "must carry dx (metres); children derive theirs from "
                    "the ratio chain.")
            # Finiteness BEFORE Fraction(), not after.  Fraction(nan)
            # raises ValueError and reaches the front door as rc 2, but
            # Fraction(inf) raises OverflowError -- one branch apart, and
            # only the first is caught -- so `dx = inf` and `dx = nan`
            # produced a sentence and a traceback for the same mistake.
            dx_value = float(dom["dx"])
            if not math.isfinite(dx_value) or dx_value <= 0:
                raise ValueError(
                    f"dx = {dom['dx']!r} on the root domain of {source} "
                    "must be a positive, finite grid spacing in metres.")
            dx_ex = Fraction(dx_value)
            if "dy" in dom and float(dom["dy"]) != float(dom["dx"]):
                raise ValueError(
                    f"dy = {dom['dy']!r} on grid_id = {grid_id} of "
                    f"{source} must equal dx = {dom['dx']!r} (Lambert "
                    "grids are isotropic).")
        else:
            time_step, fract_num, fract_den = None, 0, 1
            dt_ex = dt_by_id[parent_id] / tratio
            # `time_step` on a child was warned-and-dropped while `dt` on
            # the same table refuses through _cross_check ten lines below
            # -- the same quantity, the same number, one spelling running
            # a different step at exit 0.  The old warning's own `why=`
            # advertised the check it declined to perform ("A hand-typed
            # child dt key is still cross-checked against the chain"): it
            # was describing the sibling branch, not itself.  That WRF
            # has no per-child time_step is the reason to refuse, not to
            # warn: the model cannot deliver what was asked.
            if ("time_step" in dom or "time_step_fract_num" in dom
                    or "time_step_fract_den" in dom):
                child_step = (
                    Fraction(_positive_int(
                        "domain", "time_step", dom.get("time_step", 0),
                        source, 0))
                    + Fraction(
                        _positive_int(
                            "domain", "time_step_fract_num",
                            dom.get("time_step_fract_num", 0), source, 0),
                        _positive_int(
                            "domain", "time_step_fract_den",
                            dom.get("time_step_fract_den", 1), source)))
                # float(), not the Fraction: _cross_check reprs what it
                # is given, and "hand-types time_step=Fraction(5, 1)"
                # shows the reader this loader's internals rather than
                # the number they typed.
                _cross_check(
                    "time_step", grid_id, float(child_step), dt_ex,
                    f"parent dt {dt_by_id[parent_id]} s / "
                    f"parent_time_step_ratio {tratio}", source)
            # CHAINED float32 division down the ratio chain -- WRF's REAL
            # grid%dt = parent%dt / parent_time_step_ratio at every tree
            # edge (share/set_timekeeping.F:368), so the kernel dt
            # matches WRF bit-for-bit on ANY ratio chain (§C; the bundle
            # chain lands np.float32(60)/4/3/3 = 1.6666666, 0x3FD55555).
            # Cadence/tick arithmetic keeps using the exact rational.
            dt_fp32 = fp32_by_id[parent_id] / np.float32(tratio)
            dx_ex = dx_by_id[parent_id] / ratio
            if "dt" in dom:
                _cross_check(
                    "dt", grid_id, dom["dt"], dt_ex,
                    f"parent dt {dt_by_id[parent_id]} s / "
                    f"parent_time_step_ratio {tratio}", source)
            if "dx" in dom:
                _cross_check(
                    "dx", grid_id, dom["dx"], dx_ex,
                    f"parent dx {dx_by_id[parent_id]} m / "
                    f"parent_grid_ratio {ratio}", source)
            if "dy" in dom:
                _cross_check(
                    "dy", grid_id, dom["dy"], dx_ex,
                    f"parent dx {dx_by_id[parent_id]} m / "
                    f"parent_grid_ratio {ratio}", source)

        # --- footprint: alignment + parent-row clearance ----------------
        if not is_root:
            parent = by_id[parent_id]
            for axis, size, start, parent_size in (
                    ("west-east", nx, i_start, parent.run.nx),
                    ("south-north", ny, j_start, parent.run.ny)):
                if size % ratio != 0:
                    raise ValueError(
                        f"child domain grid_id = {grid_id} of {source}: "
                        f"{axis} extent {size} mass cells is not an "
                        f"integer multiple of parent_grid_ratio = "
                        f"{ratio} (WPS requires e_we/e_sn = "
                        "n * ratio + 1).")
                span = size // ratio
                near = start - 1
                far = parent_size - (start + span - 1)
                need = spec_bdy_width + blend_width
                for side, clearance in ((f"{axis} low", near),
                                        (f"{axis} high", far)):
                    if clearance < need:
                        raise ValueError(
                            f"child domain grid_id = {grid_id} of "
                            f"{source} violates the parent-row clearance "
                            f"rule: {side} clearance is {clearance} "
                            f"parent rows but spec_bdy_width + "
                            f"blend_width = {spec_bdy_width} + "
                            f"{blend_width} = {need} rows are required "
                            "(the child boundary must clear the parent's "
                            "own Davies and terrain-blend zones).")

        # --- per-domain RunConfig ---------------------------------------
        history_interval_s = float(dom["history_interval_s"])
        if not math.isfinite(history_interval_s) or history_interval_s <= 0:
            raise ValueError(
                f"history_interval_s = {dom['history_interval_s']!r} on "
                f"domain grid_id = {grid_id} of {source} must be a "
                "finite positive interval in seconds.")
        kw = dict(shared)
        kw.update(
            nx=nx, ny=ny, nz=nz, dx=float(dx_ex), dy=float(dx_ex),
            # run.dt carries the chained-FP32 WRF kernel dt exactly (its
            # binary64 image); the exact rational stays on dt_exact().
            dt=float(dt_fp32), clock_dt=0.0, run_seconds=run_seconds,
            output_interval_s=history_interval_s,
            specified=bool(specified), nested=bool(nested),
            grid_id=grid_id, spec_bdy_width=spec_bdy_width,
            # F14 timing authority: derived compatibility copy on every
            # domain (equality asserted below); restart ALARMS evaluate
            # on the d01 clock only (section C).
            restart_interval_s=restart_interval_s,
        )
        for key in _DOMAIN_RUN_OVERRIDES:
            if key in dom:
                kw[key] = dom[key]
        # The axis writes its resolved vector LAST and onto every domain:
        # it is the author of these keys (the [shared]/[[domain]] refusal
        # above guarantees nobody else wrote them), and one experiment is
        # one arm -- a tree whose domains ran different ledger entries
        # could not be compared across its own nest boundary.
        kw.update(physics_mode.settings)
        # bl_pbl_physics is per-domain for the WRF schemes and PBL-off
        # (the measured PBL-parent/PBL-off-LES-child trees), but SASE is
        # run-wide, never per-nest: the closure is measured single-domain
        # and a tree whose domains ran different closures could not be
        # compared across its own boundary.  A per-domain 900 that parsed
        # here would be exactly that tree, so it is refused by name.
        from gpuwm.config import SASE_PBL_SCHEME
        if dom.get("bl_pbl_physics") == SASE_PBL_SCHEME:
            raise ValueError(
                f"bl_pbl_physics = {SASE_PBL_SCHEME} (SASE) on [[domain]] "
                f"grid_id={grid_id} of {source} is refused: the SASE "
                "closure is selected run-wide in [shared], never per "
                "nest.  A nest whose domains ran different closures "
                "could not be compared across its own boundary, and no "
                "nested SASE tree has been run.")
        # Nonzero spec_exp on a NESTED domain is forced to 0 with a
        # warning: WRF's nested lbc_fcx_gcx branch has NO exponential
        # sponge term -- only the specified branch applies spec_exp
        # (dyn_em/module_bc_em.F lbc_fcx_gcx :1297-1341: specified
        # branch spongeweight :1320, nested branch :1325-1337 with the
        # sponge lines commented out).  gpuwm's nested Davies weights
        # DO read cfg.spec_exp, so zeroing here is what keeps them the
        # exact spec_exp = 0 transliteration of WRF's nested branch
        # (the N1 child_spec_exp pin: children force with spec_exp=0).
        if not is_root and float(kw.get("spec_exp", 0.0)) != 0.0:
            warn(f"spec_exp = {kw['spec_exp']!r} on nested domain "
                 f"grid_id = {grid_id} of {source} is forced to 0, "
                 "matching WRF (the sponge applies on specified "
                 "boundaries only); continuing",
                 why="dyn_em/module_bc_em.F:1297-1341: the nested "
                     "lbc_fcx_gcx branch has no sponge term (:1325); "
                     "children force with spec_exp = 0, exactly as WRF "
                     "would run this namelist.")
            kw["spec_exp"] = 0.0
        # An AUTO domain starts on the RunConfig default (0, WRF's
        # Registry value) so the whole validation battery and the
        # exposure computation below see a resolved integer; the
        # criterion-driven switch to 1 happens in ONE place,
        # resolve_auto_mix_isotropic, after the tree is assembled and
        # the layer depths are knowable.  The pop also covers a
        # per-domain "auto" under a [shared] integer.
        if mix_isotropic_auto:
            kw.pop("mix_isotropic", None)
            auto_mix_ids.append(grid_id)
        # The full legacy invariant battery applies to every per-domain
        # RunConfig (p5t1 review F1): same checks, same messages as
        # load_config.
        run = validate_run_config(RunConfig(**kw))

        # --- cadence divisibility (integer domain steps) ----------------
        _check_cadence("history_interval_s", Fraction(history_interval_s),
                       dt_ex, grid_id, source)
        _check_whole_second_cadence(
            "history_interval_s", Fraction(history_interval_s), grid_id,
            source)
        if radiation_enabled(run):
            radt_min = run.radt if run.radt > 0.0 else run.radt_minutes
            _check_cadence("radt", Fraction(radt_min) * 60, dt_ex,
                           grid_id, source)
        if run.cu_physics == 1:
            _check_cadence("cudt_minutes", Fraction(run.cudt_minutes) * 60,
                           dt_ex, grid_id, source)
        if (run.bl_pbl_physics != 0 or run.sf_sfclay_physics != 0
                or run.sf_surface_physics != 0):
            _check_cadence("bldt", Fraction(run.bldt) * 60, dt_ex,
                           grid_id, source)
        if is_root:
            _check_cadence("restart_interval_s",
                           Fraction(restart_interval_s), dt_ex, grid_id,
                           source)
            _check_whole_second_cadence(
                "restart_interval_s", Fraction(restart_interval_s), grid_id,
                source)
            # The forecast length is on the same step grid its output
            # cadences are, and is refused in the same place and the
            # same way.
            _check_run_length_on_step_grid(
                Fraction(str(run_seconds)), dt_ex, grid_id, source)

        if spawn_cfg is not None:
            # Timing sanity against THIS tree's clocks.  The manual
            # trigger's instant must land on the parent's step lattice
            # (spawning is a cycle-boundary operation); a field window
            # opening at or after the end of the run can never fire and
            # is a disabled feature wearing an enabled name.
            parent_dt = dt_by_id[parent_id]
            if spawn_cfg.at_s is not None:
                steps = Fraction(str(float(spawn_cfg.at_s))) / parent_dt
                if steps.denominator != 1:
                    raise ValueError(
                        f"spawn at_s = {spawn_cfg.at_s:g} s on [[domain]] "
                        f"grid_id = {grid_id} of {source} is not a whole "
                        f"number of parent d{parent_id:02d} steps: dt = "
                        f"{parent_dt} s exactly. A spawn is a "
                        "cycle-boundary operation and its manual instant "
                        "must land on the parent step grid.")
                if float(spawn_cfg.at_s) >= run_seconds:
                    raise ValueError(
                        f"spawn at_s = {spawn_cfg.at_s:g} s on [[domain]] "
                        f"grid_id = {grid_id} of {source} is at or past "
                        f"run_seconds = {run_seconds:g}; the nest could "
                        "never spawn, so its declaration (and its VRAM "
                        "reservation) would buy nothing.")
            if (spawn_cfg.earliest_s is not None
                    and float(spawn_cfg.earliest_s) >= run_seconds):
                raise ValueError(
                    f"spawn earliest_s = {spawn_cfg.earliest_s:g} s on "
                    f"[[domain]] grid_id = {grid_id} of {source} is at or "
                    f"past run_seconds = {run_seconds:g}; the window can "
                    "never open, so the nest could never spawn.")

        dc = DomainConfig(
            grid_id=grid_id, parent_id=parent_id, i_parent_start=i_start,
            j_parent_start=j_start, parent_grid_ratio=ratio,
            parent_time_step_ratio=tratio,
            history_interval_s=history_interval_s, run=run,
            time_step=time_step, time_step_fract_num=fract_num,
            time_step_fract_den=fract_den, start_time=domain_start,
            spawn=spawn_cfg, tiles=domain_tiles)
        domains.append(dc)
        by_id[grid_id] = dc
        dt_by_id[grid_id] = dt_ex
        dx_by_id[grid_id] = dx_ex
        fp32_by_id[grid_id] = dt_fp32

    # Resolve every directed physics edge before constructing the experiment.
    # Same-scheme trees retain their original behavior; mixed schemes require
    # an explicit GPUWM transition instead of reaching a missing parent field.
    from gpuwm.core.microphysics_transition import (
        resolve_microphysics_transition,
    )
    for dc in domains[1:]:
        resolve_microphysics_transition(by_id[dc.parent_id].run, dc.run)

    # km_opt=2 on a nest child: refused only where the nest COUPLING of the
    # prognostic TKE carrier is actually exercised, which is when the parent
    # carries one too.
    #
    # WRF gives tke no ``i`` (nest-interpolation) and no ``f`` (feedback)
    # Registry flag (Registry.EM_COMMON:312), so a child cold-starts its own
    # TKE and never returns it.  Where the PARENT runs km_opt != 2 it has no
    # TKE field at all, so there is nothing to interpolate down and nothing
    # to feed back: the coupling question is void, not unverified, and
    # cold-starting is the only behaviour available to WRF or to ArWen.
    # That case has now been RUN -- a 402x402 250 m km_opt=2 PBL-off child
    # under a km_opt=4 750 m parent, 7 h, status PASS, 31 frames, carrying
    # 4.8x-9.9x the parent's resolved w variance over the same ground and
    # developing it FASTER than the km_opt=3 child on the identical tree
    # (8.34x against 4.17x one hour in), so cold-starting the carrier
    # demonstrably does not handicap the child
    # (docs/superpowers/receipts/les/nested-les-km2-2026-08-02.md).
    #
    # Where the parent IS km_opt=2 the parent holds a live TKE field that
    # WRF pointedly does not hand down, and no such tree has been run.  That
    # is the configuration this refusal now names, and it is the only one.
    for dc in domains[1:]:
        parent = by_id[dc.parent_id].run
        if dc.run.km_opt == 2 and parent.km_opt == 2:
            raise NotImplementedError(
                f"km_opt=2 on domain grid_id = {dc.run.grid_id} whose "
                f"parent grid_id = {parent.grid_id} also runs km_opt=2 is "
                "refused: the parent then holds a prognostic TKE field "
                "that WRF's Registry flags do not interpolate to the child "
                "and do not feed back, and no such tree has been run here. "
                "A km_opt=2 child under a parent that carries no TKE "
                "(km_opt 1/3/4) is admitted and measured. Lift with "
                "evidence, not with a code change alone.")

    # SASE is not usable at nest width: the closure is measured
    # single-domain (GABLS1) and no nested SASE tree has been run, so a
    # multi-domain experiment selecting it anywhere -- even uniformly via
    # [shared] -- is refused rather than silently integrating an
    # unmeasured coupling.  (Per-domain 900 is already refused above.)
    if len(domains) > 1:
        from gpuwm.config import SASE_PBL_SCHEME
        for dc in domains:
            if dc.run.bl_pbl_physics == SASE_PBL_SCHEME:
                raise NotImplementedError(
                    f"bl_pbl_physics = {SASE_PBL_SCHEME} (SASE) on domain "
                    f"grid_id = {dc.run.grid_id} of a {len(domains)}-domain "
                    "tree is refused: the SASE closure is selectable "
                    "run-wide on a single domain only; it is not usable at "
                    "nest width, and no nested SASE tree has been run. "
                    "Lift with evidence, not with a code change alone.")

    # A dormant nest must be a LEAF: a child declared under it would need
    # cascading activation (its parent does not exist until a trigger
    # fires), which no runner implements.  Refused rather than deferred,
    # because the child's own reservation would otherwise price a domain
    # that could never legally start.
    dormant_ids = {dc.grid_id for dc in domains if dc.spawn is not None}
    for dc in domains:
        if dc.parent_id in dormant_ids:
            raise ValueError(
                f"[[domain]] grid_id = {dc.grid_id} of {source} declares "
                f"dormant d{dc.parent_id:02d} as its parent; a nest under "
                "a spawn-triggered nest would need cascading activation, "
                "which is not implemented. Declare the spawn on the leaf, "
                "or make the parent an ordinary domain.")

    relocation = _build_relocation(raw, source, domains, run_seconds)
    experiment = ExperimentConfig(
        name=name, start_time=start_time, run_seconds=run_seconds,
        vertical=vertical, projection=projection,
        feedback=feedback, smooth_option=smooth_option,
        blend_width=blend_width, spec_bdy_width=spec_bdy_width,
        restart_interval_s=restart_interval_s, domains=tuple(domains),
        column_chunk=column_chunk,
        acknowledgements=tuple(acknowledgements),
        relocation=relocation,
        physics_mode=physics_mode,
        perturbation=perturbation,
        tiles=tiles)
    # The mixing-length auto-switch runs HERE, at the one load every
    # front door shares, so run/go/check, both prepared runners and the
    # wizard's candidate loop all execute (and announce) the same
    # selection; the advisory pass further down then sees the RESOLVED
    # tree and warns only about written-out exposures.
    experiment = resolve_auto_mix_isotropic(experiment, auto_mix_ids, source)
    # [tiles] against the TREE, and it needs the assembled experiment
    # because neither half of the question is answerable alone: the block
    # is parsed hundreds of lines above, the parent_id edges hundreds of
    # lines below, and only here are both in hand.  mode = 'on' streams
    # every grid, so on a tree every coupling edge would have BOTH ends
    # streamed -- the one concurrent-nesting shape the coupler refuses --
    # and the combination is refused from the config text, at the one load
    # every front door shares, rather than at the first FORCE, which on a
    # prepared tree is a fetch, two preparations and a whole tree
    # downstream of a fact that was legible in the TOML.  mode = 'auto' is
    # deliberately NOT refused: streamed children and streamed parents are
    # both legal roads, the planner prices each domain against the budget
    # its predecessors left, and steppers_for_tree refuses a both-streamed
    # edge at decision time, before anything is built.
    streaming_module.refuse_streamed_nests(experiment, source=source)
    from gpuwm.physics_compat import (
        constant_longwave_refusal,
        nocturnal_radiation_refusal,
        radiation_off_land_surface_refusal,
        validate_resolved_physics_vertical_levels,
    )
    for domain in experiment.domains:
        validate_resolved_physics_vertical_levels(
            domain.run, p_top=experiment.vertical.p_top)
    # The nocturnal-radiation guard (2026-08-06): a real case (it has a
    # [projection], so it has a place and a clock) whose window includes
    # local night may not run shortwave with longwave OFF undeclared --
    # the surface would radiate all night with no downward longwave and
    # the skin temperature and 2 m moisture collapse.  Guarded HERE, at
    # the one load every front door shares (run/go/check, both prepared
    # runners, the DA drivers, the wizard's candidate loop), so no door
    # can miss it.  [experiment].acknowledgements carries the
    # declared-experiment override; the refusal names it.
    if experiment.projection is not None:
        # The provenance guard (task #125) shares this seam and this
        # voice: a real case may not load under an install whose
        # metadata version contradicts the version its own source
        # declares -- that is the "plots say 1.6.2 while 1.8.7
        # executes" shape, and the refusal names both numbers, both
        # locations and the remedy.  A BORROWED number that agrees --
        # every worktree beside an editable install -- warns one line
        # naming the install the number came from, and never refuses;
        # a wheel has one version claim and passes in silence.
        # Idealized loads (no projection, no place, no clock) are not
        # a disagreement, exactly as they are not this nocturnal
        # guard's business.
        from gpuwm.provenance_gate import require_version_identity
        require_version_identity(source)
        refusal = nocturnal_radiation_refusal(
            [dc.run for dc in experiment.domains],
            start_time=experiment.start_time,
            run_seconds=experiment.run_seconds,
            ref_lat=experiment.projection.ref_lat,
            ref_lon=experiment.projection.ref_lon,
            acknowledgements=experiment.acknowledgements)
        if refusal is not None:
            raise ValueError(f"experiment config {source}: {refusal}")
        # THE TWO RADIATION-ABSENCE GUARDS, in order of how much they
        # diagnose.  Both landed in this release, both read the same
        # selectors, and they overlap on exactly one class -- both
        # streams off under a land-surface scheme -- which is the class
        # BOTH of them are about.  Where they overlap, the radiation-off
        # guard speaks first, because "you have no radiation at all" is
        # the larger fact and its message carries the three remedies;
        # being told "your longwave is a declared constant" while
        # shortwave is also off would answer a smaller question than the
        # one the reader has.  A config in that overlap needs BOTH
        # declarations, which is a feature and not an accident: they are
        # two claims -- "nothing computes my sky" and "the number my land
        # surface integrates is one I typed" -- and a file that means
        # both says both.
        #
        # The radiation-OFF land-surface guard (2026-08-09), the other
        # half of the same hole.  The nocturnal guard tests sw > 0 and
        # lw == 0, so a suite with BOTH streams off walked past it while
        # initialize_physics attached no radiation adapter at all -- and
        # Noah/Noah-MP/RUC read fields["glw"] every surface step anyway.
        # Same door, same declaration idiom, no clock: a sky nothing
        # computes is wrong at noon as well as at midnight.  The
        # declared-selector set travels with it because radiation
        # defaults to OFF: a config that simply never named a selector is
        # in this class too, and gets told THAT rather than being quoted
        # two zeros it does not contain.
        refusal = radiation_off_land_surface_refusal(
            [dc.run for dc in experiment.domains],
            acknowledgements=experiment.acknowledgements,
            declared_selectors=declared_radiation)
        if refusal is not None:
            raise ValueError(f"experiment config {source}: {refusal}")
        # The constant-GLW guard (2026-08-09), the nocturnal guard's
        # companion and deliberately a SECOND question rather than a
        # clause of the first: that one asks whether the window is
        # survivable, this one asks whether the downward longwave
        # exists at all.  Kept separate because the nocturnal
        # acknowledgement was checked before any physics was inspected,
        # so a config carrying it never had its GLW source examined --
        # which is how ten shipped configs came to integrate a frozen
        # 300 W m-2.  Same load, same front doors, its own token.  It
        # refuses EXACTLY the set initialize_physics refuses, so a config
        # that passes here runs rather than dying mid-preparation.
        refusal = constant_longwave_refusal(
            [dc.run for dc in experiment.domains],
            acknowledgements=experiment.acknowledgements)
        if refusal is not None:
            raise ValueError(f"experiment config {source}: {refusal}")
    _advise_anisotropic_w_mixing(experiment, source)
    _assert_derived_copies(experiment, source)
    return experiment


class ExposedMixing(NamedTuple):
    """What :func:`anisotropic_w_mixing_exposure` hands its callers.

    ``dz_max`` -- deepest base-state layer of the vertical coordinate, in
    metres, or ``inf`` when the coordinate cannot be resolved at all.
    ``inf`` is a sentinel a caller must route on, not a depth to compute
    with; see :func:`anisotropic_w_mixing_exposure` for what it selects.
    ``domains`` -- the DomainConfigs on the exposed path, possibly empty.
    ``ladder`` -- how ``dz_max`` was obtained, EMPTY when the config
    wrote its own eta interfaces and non-empty when they were resolved
    for it; callers put it in the advisory so a derived number never
    reads like a declared one.
    """

    dz_max: float
    domains: tuple
    ladder: str


def anisotropic_w_mixing_exposure(experiment: ExperimentConfig
                                  ) -> ExposedMixing:
    """Layer depths and the domains that would be judged against them.

    This is the only place both halves of the question are known -- the
    shared vertical coordinate owns the layer depths, each domain owns
    its horizontal spacing and its mixing selectors -- so it is where the
    layer depths meet the selectors.  It computes no verdict: callers
    hand the result to :func:`gpuwm.config.anisotropic_w_mixing_advice`,
    which owns the criterion and the wording.

    A COORDINATE WITH NO EXPLICIT ETA INTERFACES IS RESOLVED, NOT
    SKIPPED.  Skipping it was a hole: ``km_opt = 3`` with
    ``mix_isotropic = 0`` at ``dx = 100`` m and no ``eta_levels`` is the
    exact configuration the criterion exists for, and until 2026-08-09 it
    came back empty and every door went quiet -- silence read as a pass
    by anything downstream, which is the opposite of what the absence of
    a number means.  When the ladder is not written down it is rebuilt
    the way the model rebuilds it: the uniform interfaces
    :func:`gpuwm.core.grid.make_vertical_coord` produces from ``nz``
    (every idealized/legacy route in the tree calls it with no
    ``stretch``), with the model top expressed as a pressure by
    :func:`gpuwm.core.grid.analytic_base_pressure`, whose base state is
    the one :func:`~gpuwm.core.grid.base_layer_depths` already inverts --
    so the resolved column spans exactly ``[0, ztop]``.

    ``nz``/``ztop`` come off the first domain because they cannot differ
    across domains: they are in ``_DOMAIN_VERTICAL_KEYS``, so a
    per-domain spelling is a load error and the vertical grid is shared
    by construction.

    ``dz_max = inf`` when even that fails (a model top above the analytic
    base state's ~24.6 km ceiling).  Infinity is a SENTINEL, not a large
    depth: no ratio is computed from it -- ``anisotropic_w_mixing_ratio``
    returns ``None`` for a non-finite depth, because there is no number
    and inventing one would put a fictitious depth into a stability
    criterion.  What it does instead is select a different sentence:
    :func:`gpuwm.config.anisotropic_w_mixing_advice` recognises the
    non-finite depth on an otherwise-exposed domain and advises that the
    criterion CANNOT BE EVALUATED here, quoting ``ladder`` for the reason.
    That advisory reaches the same three doors the numeric one does --
    the load-time warning, ``gpuwm check``, and the repository screen in
    ``tests/test_shipped_configs_mixing_stability.py``, which reports the
    domain with an infinite ratio exactly as its legacy branch does.

    Until 2026-08-09 that sentinel went nowhere: the ``None`` it produced
    was the same ``None`` a config OFF the exposed path produces, so all
    three doors read it as "not applicable" and went quiet.  Anything
    here that consumes the ratio alone must not read its absence as a
    pass -- ask for the advice.
    """

    exposed = tuple(d for d in experiment.domains
                    if d.run.km_opt in (2, 3) and d.run.mix_isotropic == 0)
    if not exposed:
        return ExposedMixing(0.0, (), "")
    dz_max, ladder = _mixing_layer_depths(experiment)
    return ExposedMixing(dz_max, exposed, ladder)


def _mixing_layer_depths(experiment: ExperimentConfig) -> tuple[float, str]:
    """``(dz_max, ladder)`` -- THE layer-depth resolution, in one place.

    Extracted from :func:`anisotropic_w_mixing_exposure` so the
    auto-switch resolver (:func:`resolve_auto_mix_isotropic`) and the
    switched-domain reporter (:func:`auto_selected_isotropic_mixing`)
    read exactly the depths the advisory reads -- one implementation of
    the criterion's inputs, never a copy.  The ``inf`` sentinel and the
    ladder provenance keep their exposure-era meanings.
    """

    from gpuwm.core.grid import (analytic_base_pressure, base_layer_depths,
                                 make_vertical_coord)

    vertical = experiment.vertical
    if vertical.eta_levels:
        znw = vertical.eta_levels
        p_top = vertical.p_top
        ladder = ""
    else:
        run = experiment.domains[0].run
        p_top = analytic_base_pressure(run.ztop)
        if p_top is None:
            return (
                math.inf,
                f"no eta_levels, and ztop = {float(run.ztop):g} m is above "
                f"the analytic base state's representable ceiling, so no "
                f"layer depth can be resolved for this grid at all")
        znw = make_vertical_coord(int(run.nz)).znw
        ladder = (f"resolved ladder -- this config declares no eta_levels, "
                  f"so the depths are the uniform nz = {int(run.nz)} "
                  f"interfaces the model builds, spanning ztop = "
                  f"{float(run.ztop):g} m")

    dz_max = float(base_layer_depths(
        znw, vertical.hybrid_opt, vertical.etac, p_top).max())
    return dz_max, ladder


def resolve_auto_mix_isotropic(experiment: ExperimentConfig, auto_ids,
                               source: str) -> ExperimentConfig:
    """Apply the mixing-length auto-switch (Drew, 2026-08-16) to one tree.

    ``auto_ids`` are the grid_ids whose config left ``mix_isotropic``
    unset (or wrote ``"auto"``).  Each such domain that selects the
    per-axis path (``km_opt`` 2/3) is judged by the one criterion
    implementation -- :func:`gpuwm.config.anisotropic_w_mixing_ratio`
    over :func:`_mixing_layer_depths` -- and where the ratio exceeds
    :data:`gpuwm.config.EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT` the domain
    runs ``mix_isotropic = 1``, announced by one loud
    :func:`gpuwm.explain.warn` line per switched domain.

    Everything else is untouched, deliberately and observably:

    * a domain that satisfies the criterion keeps the WRF-default 0 and
      is byte-identical (restart identity included) to the same file
      with ``mix_isotropic = 0`` written out;
    * a WRITTEN 0 or 1 is never here (``auto_ids`` excludes it) -- an
      explicit danger-zone 0 keeps the anisotropic form and gets the
      forced-override advisory instead;
    * an unresolvable layer depth switches nothing: the criterion has
      no number there, and the no-number advisory (not this resolver)
      is what speaks.

    The re-validated replacement RunConfig keeps the p5t1 battery
    binding on what actually runs.
    """

    auto_ids = tuple(sorted({int(grid_id) for grid_id in auto_ids}))
    if not auto_ids:
        return experiment
    experiment = _dc_replace(experiment, auto_mix_isotropic=auto_ids)
    candidates = [dc for dc in experiment.domains
                  if dc.grid_id in set(auto_ids)
                  and dc.run.km_opt in (2, 3)
                  and dc.run.mix_isotropic == 0]
    if not candidates:
        return experiment
    dz_max, ladder = _mixing_layer_depths(experiment)
    switched: dict[int, float] = {}
    for dc in candidates:
        ratio = anisotropic_w_mixing_ratio(
            km_opt=dc.run.km_opt, mix_isotropic=0,
            mix_upper_bound=dc.run.mix_upper_bound,
            dx=dc.run.dx, dy=dc.run.dy, dz_max=dz_max)
        if (ratio is not None
                and ratio > EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT):
            switched[dc.grid_id] = float(ratio)
    if not switched:
        return experiment
    experiment = _dc_replace(experiment, domains=tuple(
        _dc_replace(dc, run=validate_run_config(
            _dc_replace(dc.run, mix_isotropic=1)))
        if dc.grid_id in switched else dc
        for dc in experiment.domains))
    for grid_id in sorted(switched):
        warn(auto_mix_isotropic_selection(
            where=f"domain grid_id = {grid_id} of {source}",
            ratio=switched[grid_id], ladder=ladder),
            why="With mix_isotropic = 0 the vertical exchange coefficient "
                "is built and capped on the LAYER DEPTH and then diffuses "
                "w over the HORIZONTAL spacing; past "
                "mix_upper_bound*(dz_max/dx)^2 = 1/4 that operator can "
                "invert a 2dx mode and past 1/2 grow one. The config "
                "declined to choose a length, so the model chooses the "
                "one that is stable on this grid.")
    return experiment


def auto_selected_isotropic_mixing(experiment: ExperimentConfig
                                   ) -> tuple[tuple[tuple[int, float], ...],
                                              str]:
    """``((grid_id, ratio), ...)`` the model switched, plus the ladder.

    The reporting face of :func:`resolve_auto_mix_isotropic`, for doors
    that see only the RESOLVED experiment (``gpuwm check`` first among
    them): a domain is reported here exactly when its ``mix_isotropic =
    1`` was the model's choice, with the ratio the anisotropic path
    would have reached -- recomputed through the same
    :func:`_mixing_layer_depths` /
    :func:`gpuwm.config.anisotropic_w_mixing_ratio` pair the resolver
    used, so the two can never tell different stories.  A WRITTEN 1 is
    never reported: that configuration is legitimate and quiet.
    """

    auto = set(getattr(experiment, "auto_mix_isotropic", ()) or ())
    selected = [dc for dc in experiment.domains
                if dc.grid_id in auto and dc.run.km_opt in (2, 3)
                and dc.run.mix_isotropic == 1]
    if not selected:
        return (), ""
    dz_max, ladder = _mixing_layer_depths(experiment)
    out = []
    for dc in selected:
        ratio = anisotropic_w_mixing_ratio(
            km_opt=dc.run.km_opt, mix_isotropic=0,
            mix_upper_bound=dc.run.mix_upper_bound,
            dx=dc.run.dx, dy=dc.run.dy, dz_max=dz_max)
        if (ratio is not None
                and ratio > EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT):
            out.append((dc.grid_id, float(ratio)))
    return tuple(out), ladder


def _advise_anisotropic_w_mixing(experiment: ExperimentConfig,
                                 source: str) -> None:
    """Advisory pass: per-axis mixing lengths against the explicit limit.

    Warn, never block: see
    :func:`gpuwm.config.warn_anisotropic_w_mixing` for the ruling and
    the numbers behind it.  ``gpuwm check`` repeats the same sentence in
    its advisory list, because this line is emitted at config load and
    the report is where a reader is looking.
    """

    dz_max, exposed, ladder = anisotropic_w_mixing_exposure(experiment)
    auto = set(getattr(experiment, "auto_mix_isotropic", ()) or ())
    for domain in exposed:
        # Post-resolution, an exposed domain either WROTE its 0 (the
        # forced-override state, said as such) or left it unset on a
        # depth the criterion could not evaluate (the no-number
        # advisory, which the forced flag does not touch).
        warn_anisotropic_w_mixing(
            where=f"domain grid_id = {domain.grid_id} of {source}",
            km_opt=domain.run.km_opt,
            mix_isotropic=domain.run.mix_isotropic,
            mix_upper_bound=domain.run.mix_upper_bound,
            dx=domain.run.dx, dy=domain.run.dy, dz_max=dz_max,
            ladder=ladder, forced=domain.grid_id not in auto)


def _assert_derived_copies(experiment: ExperimentConfig,
                           source: str) -> None:
    """F14 timing authority + F1 vertical single-sourcing assertions.

    ``ExperimentConfig.run_seconds``/``restart_interval_s`` and each
    ``DomainConfig.history_interval_s`` are the ONLY authoritative
    timing values; the embedded RunConfig copies are derived and must
    match exactly.  Likewise ``hybrid_opt``/``etac`` on every RunConfig
    are compatibility copies of ``experiment.vertical``.  By
    construction these hold; the assertion guards future drift loudly.
    """
    for dc in experiment.domains:
        for label, copy, authority in (
                ("run_seconds", dc.run.run_seconds,
                 experiment.run_seconds),
                ("output_interval_s", dc.run.output_interval_s,
                 dc.history_interval_s),
                ("restart_interval_s", dc.run.restart_interval_s,
                 experiment.restart_interval_s),
                ("hybrid_opt", dc.run.hybrid_opt,
                 experiment.vertical.hybrid_opt),
                ("etac", dc.run.etac, experiment.vertical.etac)):
            if copy != authority:
                raise ValueError(
                    f"derived RunConfig copy diverged on grid_id = "
                    f"{dc.grid_id} of {source}: run.{label} = {copy!r} "
                    f"but the authoritative experiment value is "
                    f"{authority!r} (F14 timing authority / F1 vertical "
                    "single-sourcing).")


# ---------------------------------------------------------------------------
# Dormant nests: the active-tree views the spawn runner consumes
# ---------------------------------------------------------------------------

def dormant_domain_ids(exp: ExperimentConfig) -> tuple[int, ...]:
    """grid_ids of every declared spawn-triggered (dormant) nest."""
    return tuple(dc.grid_id for dc in exp.domains
                 if getattr(dc, "spawn", None) is not None)


def validate_spawn_placement(exp: ExperimentConfig, grid_id: int,
                             i_parent_start: int,
                             j_parent_start: int) -> None:
    """Re-run the loader's placement admission for a trigger-chosen spot.

    A spawned placement bypasses :func:`build_experiment` (the tree was
    admitted with the placeholder placement), so the SAME parent-row
    clearance rule is applied here, with the same numbers and the same
    sentence shape.  ``register_nest``'s +-2 SINT stencil refusal still
    has the final word at materialization.
    """
    dc = exp.domain(grid_id)
    parent = exp.domain(dc.parent_id)
    need = int(exp.spec_bdy_width) + int(exp.blend_width)
    for axis, size, start, parent_size in (
            ("west-east", dc.run.nx, int(i_parent_start), parent.run.nx),
            ("south-north", dc.run.ny, int(j_parent_start), parent.run.ny)):
        if start < 1:
            raise ValueError(
                f"spawn placement {start} on the {axis} axis of "
                f"d{grid_id:02d} is not 1-based WRF namelist semantics")
        span = size // dc.parent_grid_ratio
        near = start - 1
        far = parent_size - (start + span - 1)
        for side, clearance in ((f"{axis} low", near),
                                (f"{axis} high", far)):
            if clearance < need:
                raise ValueError(
                    f"spawned placement ({i_parent_start}, "
                    f"{j_parent_start}) for d{grid_id:02d} violates the "
                    f"parent-row clearance rule: {side} clearance is "
                    f"{clearance} parent rows but spec_bdy_width + "
                    f"blend_width = {exp.spec_bdy_width} + "
                    f"{exp.blend_width} = {need} rows are required.")


def active_experiment(exp: ExperimentConfig,
                      spawned: dict | None = None) -> ExperimentConfig:
    """The tree the executor integrates: dormant nests out, spawned in.

    ``spawned`` maps grid_id -> ``(i_parent_start, j_parent_start)`` for
    every dormant nest whose trigger has fired; those domains join the
    active tree AT the fired placement (their ``spawn`` declaration is
    KEPT, so the activated experiment's identity binds the fact and the
    terms of the spawn).  Dormant nests not in ``spawned`` are removed
    -- they cost their memory-plan reservation and zero compute, which
    is the declared contract.  With no dormant nests this is the
    identity, byte-for-byte the same object.

    This is the schedule-surgery seam the leg runner consumes: legs
    before a trigger integrate ``active_experiment(exp)``, and the leg
    after a fire integrates ``active_experiment(exp, {gid: (i, j)})``.
    """
    spawned = {} if spawned is None else dict(spawned)
    dormant = set(dormant_domain_ids(exp))
    unknown = sorted(set(spawned) - dormant)
    if unknown:
        raise ValueError(
            f"spawned grid_id(s) {unknown} are not declared dormant "
            f"nests of experiment {exp.name!r} (dormant: "
            f"{sorted(dormant)}); only a declared spawn can activate.")
    if not dormant:
        return exp
    for grid_id, position in spawned.items():
        i_start, j_start = (int(position[0]), int(position[1]))
        validate_spawn_placement(exp, grid_id, i_start, j_start)
    domains = []
    for dc in exp.domains:
        if dc.grid_id in spawned:
            i_start, j_start = spawned[dc.grid_id]
            domains.append(_dc_replace(
                dc, i_parent_start=int(i_start),
                j_parent_start=int(j_start)))
        elif dc.grid_id in dormant:
            continue
        else:
            domains.append(dc)
    return _dc_replace(exp, domains=tuple(domains))


def pre_spawn_experiment(exp: ExperimentConfig) -> ExperimentConfig:
    """The startup tree: every dormant nest removed, nothing spawned."""
    return active_experiment(exp, None)


def refuse_unrouted_spawn(exp: ExperimentConfig, route: str) -> None:
    """Fail loud where a spawn declaration would otherwise be dropped.

    Same governance as [perturbation]: honored or refused, never
    ignored.  A route that neither reserves the dormant nest nor
    evaluates its trigger calls this immediately after loading the
    experiment, so the user learns at admission -- not after a run that
    silently integrated without the nest they declared.
    """
    dormant = dormant_domain_ids(exp)
    if not dormant:
        return
    named = ", ".join(f"d{gid:02d}" for gid in dormant)
    raise ValueError(layered(
        f"the {route} route does not implement spawn-triggered nests; "
        f"{named} declare(s) spawn tables that this route would neither "
        "reserve nor watch, so the run would quietly integrate without "
        "the nest(s) you declared.",
        "Dormant nests are reserved and spawned on the experiment tree "
        "path (gpuwm.core.model.build_experiment reserves; the "
        "leg-boundary spawn runner activates through "
        "gpuwm.experiment.active_experiment and "
        "gpuwm.ingest.nest_spawn_init)."))
