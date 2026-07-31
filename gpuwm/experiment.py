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

import math
import tomllib
from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path

import numpy as np

from gpuwm.config import (DEFAULT_COLUMN_CHUNK, RunConfig,
                          radiation_enabled, validate_run_config)

#: Relative tolerance for cross-checking hand-typed child dx/dt against the
#: exact chain-derived rational.  Wide enough for a truncated decimal of
#: the true value (the bundle namelist's ``dx = 333.333333`` vs the exact
#: 1000/3 m, relative error ~1e-9); a wrong value (a hand-typed "500 m"
#: d04 against ratio 3 from 1 km) misses by ~0.5 and is a hard error.
_REL_TOL = 1.0e-6

#: WRF moving-nest namelist controls (Registry.EM_COMMON &domains).  Any
#: appearance is rejected loudly: moving nests are a design-spec non-goal
#: (architecture "WRF deviations (registered)"), and the SINT donor tables
#: are precomputed once at setup on the premise that nests are static.
_MOVING_NEST_KEYS = frozenset({
    "num_moves", "move_id", "move_interval", "move_cd_x", "move_cd_y",
    "vortex_interval", "max_vortex_speed", "corral_dist", "track_level",
    "time_to_move",
})

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
})
_EXPERIMENT_REQUIRED = ("name", "start_time", "run_seconds",
                        "restart_interval_s")

_PROJECTION_KEYS = ("map_proj", "ref_lat", "ref_lon", "truelat1",
                    "truelat2", "stand_lon")

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
#: diff_6th_factor 0.12/0.10/0.08/0.06).
_DOMAIN_RUN_OVERRIDES = (
    "cu_physics", "cudt_minutes", "radt", "radt_minutes", "bldt",
    "diff_6th_factor", "epssm", "spec_exp", "mp_physics", "moist",
    "moist_cq", "nest_microphysics_transition",
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
            raise ValueError(
                f"map_proj = {self.map_proj!r} is not supported "
                "(implemented: 'lambert', 'mercator', 'polar').")
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
            if max(abs(self.truelat1), abs(self.truelat2)) >= 89.9 \
                    or min(abs(self.truelat1), abs(self.truelat2)) <= 0.1:
                raise ValueError(
                    "Lambert true latitudes must lie strictly between "
                    "the equator and the pole (|truelat| in [0.1, 89.9] "
                    f"degrees), got {self.truelat1!r}/{self.truelat2!r}; "
                    "use map_proj = 'mercator' toward the equator or "
                    "'polar' toward the pole.")
        elif self.map_proj == "mercator":
            if abs(self.truelat1) >= 89.9:
                raise ValueError(
                    f"Mercator truelat1 = {self.truelat1!r} is at the "
                    "pole; the projection degenerates (cos(truelat1) "
                    "-> 0).")
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


def is_experiment_toml(path: str | Path) -> bool:
    """True when ``path`` is an experiment TOML (``[experiment]`` or
    ``[[domain]]`` present) rather than a legacy RunConfig TOML."""
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    return "experiment" in raw or "domain" in raw


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment TOML (fail-loud, section A).

    An advisory ``[fetch]`` hints table (emitted by ``gpuwm domain``,
    schema owned by :func:`gpuwm.fetch.validate_fetch_hints`) is split
    off and validated here, exactly as ``[case_data]`` is split off by
    :func:`gpuwm.case_data.load_experiment_case`; the experiment schema
    itself stays strict.
    """
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    fetch_table = raw.pop("fetch", None)
    if fetch_table is not None:
        from gpuwm.fetch import validate_fetch_hints
        validate_fetch_hints(fetch_table, source=str(path))
    return build_experiment(raw, source=str(path))


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
            f"moving-nest key(s) {present} in [{table_name}] of {source} "
            "are rejected: moving nests are a Phase-5 non-goal (static "
            "nests only -- SINT donor index/weight tables are precomputed "
            "once at setup); remove the key(s).")


def _require_keys(table_name: str, entries: dict, required, source: str):
    missing = [key for key in required if key not in entries]
    if missing:
        raise ValueError(
            f"[{table_name}] of {source} is missing required key(s) "
            f"{missing}; present: {sorted(entries)}.")


def _reject_unknown_keys(table_name: str, entries: dict, known,
                         source: str) -> None:
    unknown = sorted(set(entries) - set(known))
    if unknown:
        raise ValueError(
            f"unknown key(s) {unknown} in [{table_name}] of {source}; "
            f"known keys: {sorted(known)}.")


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


def build_experiment(raw: dict, source: str) -> ExperimentConfig:
    """Validate a parsed experiment TOML dict and build the config."""
    known_tables = ("experiment", "shared", "projection", "domain")
    unknown_tables = [name for name in raw if name not in known_tables]
    if unknown_tables:
        raise ValueError(
            f"unknown table(s)/top-level key(s) {unknown_tables} in "
            f"experiment config {source}; known tables: "
            f"{list(known_tables)}.")
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

    # ---- [shared] ------------------------------------------------------
    shared = dict(raw.get("shared", {}))
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
    if map_proj == 0 and projection is not None:
        raise ValueError(
            f"a [projection] table is present in {source} but [shared] "
            "resolves map_proj = 0 (idealized/none); set map_proj to "
            "the WRF code matching the table (1 = lambert, 2 = polar, "
            "3 = mercator) or remove the table.")
    if projection is not None and map_proj != projection.wrf_map_proj:
        raise ValueError(
            f"[shared] of {source} sets map_proj = {map_proj} but the "
            f"[projection] table declares {projection.map_proj!r} "
            f"(WRF code {projection.wrf_map_proj}); the two must agree.")

    spec_zone = shared.get("spec_zone", RunConfig.spec_zone)
    relax_zone = shared.get("relax_zone", RunConfig.relax_zone)
    if spec_bdy_width < spec_zone + relax_zone:
        raise ValueError(
            f"spec_bdy_width = {spec_bdy_width} in [experiment] of "
            f"{source} must cover spec_zone + relax_zone = "
            f"{spec_zone} + {relax_zone}.")

    # ---- [[domain]] ----------------------------------------------------
    domains: list[DomainConfig] = []
    by_id: dict[int, DomainConfig] = {}
    dt_by_id: dict[int, Fraction] = {}
    dx_by_id: dict[int, Fraction] = {}
    fp32_by_id: dict[int, np.float32] = {}
    for index, dom in enumerate(domain_tables):
        _reject_moving_nest_keys("domain", dom, source)
        _reject_domain_vertical_keys(dom, dom.get("grid_id", index + 1),
                                     source)
        _reject_unknown_keys("domain", dom, _DOMAIN_KEYS, source)
        _require_keys(f"domain #{index + 1}", dom, _DOMAIN_REQUIRED, source)
        grid_id = _positive_int("domain", "grid_id", dom["grid_id"], source)
        if grid_id in by_id:
            raise ValueError(
                f"duplicate grid_id = {grid_id} in [[domain]] tables of "
                f"{source}.")
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
            raise ValueError(
                f"root [[domain]] grid_id = {grid_id} of {source} has "
                f"start_time = {domain_start.isoformat()}, but the root "
                "must start at [experiment].start_time = "
                f"{start_time.isoformat()}.")
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

        # Flags: root takes external specified LBCs, children are nested
        # -- WRF &bdy_control, bundle namelist.input specified=T,F,F,F /
        # nested=F,T,T,T; mutually exclusive by construction.
        specified = dom.get("specified", is_root)
        nested = dom.get("nested", not is_root)
        if is_root and (specified is not True or nested is not False):
            raise ValueError(
                f"the root domain grid_id = {grid_id} of {source} must "
                f"have specified = true and nested = false (WRF "
                "&bdy_control: the head grid takes external lateral "
                "boundaries), got specified = "
                f"{specified!r}, nested = {nested!r}.")
        if not is_root and (nested is not True or specified is not False):
            raise ValueError(
                f"child domain grid_id = {grid_id} of {source} must have "
                "nested = true and specified = false (WRF &bdy_control: "
                "children are forced by their parent, bundle "
                "specified=T,F,F,F / nested=F,T,T,T), got specified = "
                f"{specified!r}, nested = {nested!r}.")

        ratio = _positive_int("domain", "parent_grid_ratio",
                              dom["parent_grid_ratio"], source)
        tratio = _positive_int("domain", "parent_time_step_ratio",
                               dom["parent_time_step_ratio"], source)
        i_start = _positive_int("domain", "i_parent_start",
                                dom["i_parent_start"], source)
        j_start = _positive_int("domain", "j_parent_start",
                                dom["j_parent_start"], source)
        if is_root:
            for key, value in (("parent_grid_ratio", ratio),
                               ("parent_time_step_ratio", tratio),
                               ("i_parent_start", i_start),
                               ("j_parent_start", j_start)):
                if value != 1:
                    raise ValueError(
                        f"the root domain grid_id = {grid_id} of {source} "
                        f"must have {key} = 1, got {value}.")
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
            if "dt" in dom:
                raise ValueError(
                    f"the root [[domain]] grid_id = {grid_id} of {source} "
                    "carries its model step as time_step (integer "
                    "seconds) plus optional time_step_fract_num/den "
                    "(Registry.EM_COMMON:2245-2246), not a dt key; "
                    f"remove dt = {dom['dt']!r}.")
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
            # WRF-REAL head-grid dt: single-precision evaluation of the
            # rational namelist keys (Registry.EM_COMMON:2245-2246).
            dt_fp32 = (np.float32(time_step)
                       + np.float32(fract_num) / np.float32(fract_den))
            if "dx" not in dom:
                raise ValueError(
                    f"the root [[domain]] grid_id = {grid_id} of {source} "
                    "must carry dx (metres); children derive theirs from "
                    "the ratio chain.")
            dx_ex = Fraction(float(dom["dx"]))
            if dx_ex <= 0:
                raise ValueError(
                    f"dx = {dom['dx']!r} on the root domain of {source} "
                    "must be positive.")
            if "dy" in dom and float(dom["dy"]) != float(dom["dx"]):
                raise ValueError(
                    f"dy = {dom['dy']!r} on grid_id = {grid_id} of "
                    f"{source} must equal dx = {dom['dx']!r} (Lambert "
                    "grids are isotropic).")
        else:
            if "time_step" in dom or "time_step_fract_num" in dom \
                    or "time_step_fract_den" in dom:
                raise ValueError(
                    f"time_step keys on child domain grid_id = {grid_id} "
                    f"of {source} are rejected: time_step is a head-grid "
                    "(d01) key (Registry.EM_COMMON:2245 declares it "
                    "scalar); the child dt derives from the parent chain "
                    "exactly (share/set_timekeeping.F:366-368).")
            time_step, fract_num, fract_den = None, 0, 1
            dt_ex = dt_by_id[parent_id] / tratio
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
        # The full legacy invariant battery applies to every per-domain
        # RunConfig (p5t1 review F1): same checks, same messages as
        # load_config.
        run = validate_run_config(RunConfig(**kw))

        # Nonzero spec_exp on a NESTED domain is rejected loudly: WRF's
        # nested lbc_fcx_gcx branch has NO exponential sponge term -- only
        # the specified branch applies spec_exp (dyn_em/module_bc_em.F
        # lbc_fcx_gcx :1297-1341: specified branch spongeweight :1320,
        # nested branch :1325-1337 with the sponge lines commented out).
        # gpuwm reuses the Phase-4 Davies ``_weights`` with spec_exp = 0
        # as a STATED transliteration of that branch, so a nonzero child
        # spec_exp would silently change nothing.
        if run.nested and run.spec_exp != 0.0:
            raise ValueError(
                f"spec_exp = {run.spec_exp!r} on nested domain grid_id = "
                f"{grid_id} of {source} is rejected: WRF's nested "
                "lbc_fcx_gcx branch has no exponential sponge term "
                "(dyn_em/module_bc_em.F:1297-1341, nested branch :1325 -- "
                "sponge only in the specified branch :1320); gpuwm's "
                "nested Davies weights are the spec_exp = 0 "
                "transliteration of that branch, so set spec_exp = 0.")

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

        dc = DomainConfig(
            grid_id=grid_id, parent_id=parent_id, i_parent_start=i_start,
            j_parent_start=j_start, parent_grid_ratio=ratio,
            parent_time_step_ratio=tratio,
            history_interval_s=history_interval_s, run=run,
            time_step=time_step, time_step_fract_num=fract_num,
            time_step_fract_den=fract_den, start_time=domain_start)
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

    experiment = ExperimentConfig(
        name=name, start_time=start_time, run_seconds=run_seconds,
        vertical=vertical, projection=projection,
        feedback=feedback, smooth_option=smooth_option,
        blend_width=blend_width, spec_bdy_width=spec_bdy_width,
        restart_interval_s=restart_interval_s, domains=tuple(domains),
        column_chunk=column_chunk,
        acknowledgements=tuple(acknowledgements))
    from gpuwm.physics_compat import (
        validate_resolved_physics_vertical_levels,
    )
    for domain in experiment.domains:
        validate_resolved_physics_vertical_levels(
            domain.run, p_top=experiment.vertical.p_top)
    _assert_derived_copies(experiment, source)
    return experiment


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
