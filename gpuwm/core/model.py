"""Domain-tree construction and the Phase-5 flat-schedule executor."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import nullcontext
import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import math
import numpy as np
from pathlib import Path
import time
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

from gpuwm.core.clock import DomainClock, Schedule
from gpuwm.config import (SURFACE_RADIATION_POLICY_REQUIRED,
                          radiation_scheme_ids)
from gpuwm.core.state import DomainState
from gpuwm.experiment import DomainConfig
from gpuwm.static.lambert import LambertGrid


RestartPhase = Literal["PERIOD_BEGIN"]
PERIOD_BEGIN: RestartPhase = "PERIOD_BEGIN"
"""The only legal multi-domain checkpoint phase.

Task 14 serializes this marker explicitly; it is never inferred from elapsed
time.  Restore invalidates child boundary tables, and the ordinary parent STEP
then FORCE must rebuild them before any child STEP.
"""


@dataclass
class FeedbackScratch:
    """Placeholder for child-owned restricted feedback data.

    Phase 5 allocates no feedback payload because ``feedback=0`` is enforced
    at experiment load.  Phase 5b replaces ``payload`` with the typed scratch
    views produced by ``feedback_prepare`` without changing the structural
    coupler protocol or schedule table.
    """

    payload: object | None = None


@runtime_checkable
class NestCoupler(Protocol):
    """Node-facing parent/child coupling surface.

    Concrete couplers satisfy this protocol structurally; they do not need to
    import or inherit it.  The parent-read-only rule applies to ``force``
    ONLY.  Phase 5b's feedback transaction makes its parent-mutation boundary
    explicit; all three feedback phases remain dormant while Phase 5 enforces
    one-way coupling at experiment load.

    WRF initialization is part of that activation contract, not an optional
    post-hoc sweep: ``med_nest_initial`` saves ``parent%ht_coarse`` and calls
    ``med_nest_feedback`` (share/mediation_integrate.F:774-777); when
    ``feedback != 0`` the mediation layer mutates the parent
    (:1035-1037), after which initialization re-runs ``start_domain`` on the
    parent (:787-838).  A Phase-5b builder must reproduce that ordered
    initialization transaction.  A bottom-up feedback sweep after all
    domains have been initialized is not equivalent.
    """

    def force(self, node: DomainNode) -> None:
        """Read the parent without mutation and refresh/mutate the child."""
        ...

    def feedback_prepare(self, node: DomainNode,
                         out: FeedbackScratch) -> None:
        """Read the child only and write restricted data into scratch."""
        ...

    def feedback_commit(self, node: DomainNode) -> None:
        """PARENT MUTATED: masked merge plus ``ht_coarse`` bookkeeping."""
        ...

    def feedback_finalize(self, node: DomainNode) -> None:
        """PARENT MUTATED: restore parent halo/diagnostic validity."""
        ...


@dataclass
class DomainNode:
    """One configured domain and its parent-first tree links."""

    cfg: DomainConfig
    grid: LambertGrid
    state: DomainState
    clock: DomainClock
    parent: DomainNode | None
    children: list[DomainNode]
    coupler: NestCoupler | None

    def __post_init__(self) -> None:
        # Lifecycle state is deliberately not a dataclass field: the public
        # structural contract remains the configured tree, while delayed
        # activation and restart restore own this runtime-only marker.
        self._started = True
        cfg_id = self.cfg.grid_id
        if self.cfg.run.grid_id != cfg_id:
            raise ValueError(
                f"DomainNode grid_id mismatch: cfg.grid_id={cfg_id}, "
                f"cfg.run.grid_id={self.cfg.run.grid_id}")
        if self.clock.spec.grid_id != cfg_id:
            raise ValueError(
                f"DomainNode grid_id mismatch: cfg.grid_id={cfg_id}, "
                f"clock.spec.grid_id={self.clock.spec.grid_id}")

        if self.cfg.parent_id == 0:
            if self.parent is not None:
                raise ValueError(
                    f"root DomainNode grid_id={cfg_id} must have parent=None")
            if self.coupler is not None:
                raise ValueError(
                    f"root DomainNode grid_id={cfg_id} must have coupler=None")
            return

        if self.parent is None:
            raise ValueError(
                f"child DomainNode grid_id={cfg_id} must receive its "
                "already-constructed parent (parent-before-child)")
        if self.parent.cfg.grid_id != self.cfg.parent_id:
            raise ValueError(
                f"DomainNode parent_id mismatch for grid_id={cfg_id}: "
                f"cfg.parent_id={self.cfg.parent_id}, "
                f"parent.cfg.grid_id={self.parent.cfg.grid_id}")


@dataclass
class ExperimentState:
    """Loop-free container populated by the full Task-14 builder.

    The future executor may checkpoint this tree only at ``PERIOD_BEGIN``
    with equal domain ticks, no pending FORCE/FEEDBACK/D2H/mutation, and all
    prior feedback committed.  A restored tree starts with invalid child
    boundary tables; eager rebuild from parent(t) is forbidden, and a child
    STEP before the normal parent STEP + FORCE rebuild is an assertion error.
    """

    root: DomainNode
    nodes_by_grid_id: Mapping[int, DomainNode]
    schedule: Schedule
    memory_ledger: object | None
    experiment_fingerprint: str

    #: The route's build-time context (experiment, case data, forcing
    #: times, radiation workspace), set by ``build_experiment``.  Left
    #: DELIBERATELY unannotated so it stays a plain class attribute and
    #: not a dataclass field: hand-assembled trees (the idealized and
    #: verify paths) construct this positionally and never set it, and a
    #: real default is what lets the executor read it as an ordinary
    #: attribute -- the clock-module audit
    #: (tests/test_clock.py::test_no_float_elapsed_accumulation_audit)
    #: bans getattr reflection here.
    _activation_context = None

    #: The experiment this tree was CONFIGURED from, published by
    #: whichever route built the tree (:func:`publish_declared_experiment`).
    #: Deliberately not a key of ``_activation_context`` above: that
    #: bundle is one route's ingest context and a tree assembled from a
    #: prepared cache has none of it, so a checkpoint that reached for
    #: the declaration there could only be written by the route that
    #: happened to carry the ingest.  Same unannotated class attribute
    #: for the same reason.
    _declared_experiment = None

    def node(self, grid_id: int) -> DomainNode:
        """Return one domain node by its configured grid identifier."""
        return self.nodes_by_grid_id[grid_id]

    def walk_parent_first(self) -> Iterator[DomainNode]:
        """Walk the tree depth-first with every parent before its children."""
        stack = [self.root]
        while stack:
            current = stack.pop()
            yield current
            stack.extend(reversed(current.children))


@dataclass
class SharedRRTMGPChunkWorkspace:
    """One real byte allocation shared by all sequential RRTMGP adapters.

    Each solver phase lays its audited live set over the same backing.  Common
    optics entries keep identical offsets into the following RTE phase, while
    dead mask storage is reused by sources/fluxes.  ``phase`` returns only the
    active column prefix; kernels or explicit fills overwrite every returned
    element before its first read (``RRTMGP_WORKSPACE_LIFETIME_AUDIT``).

    The hidden injection fields keep this allocation CPU-testable with NumPy;
    production passes neither and therefore performs exactly one CuPy byte
    allocation of the preflight ledger's phase maximum.
    """

    nz: int
    column_chunk: int
    p_top: float = 5000.0
    _array_module: object | None = field(
        default=None, repr=False, compare=False)
    _phase_layouts_input: object | None = field(
        default=None, repr=False, compare=False)
    _storage: object = field(init=False, repr=False, compare=False)
    _phase_layouts: object = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.nz < 1 or self.column_chunk < 1:
            raise ValueError("RRTMGP workspace dimensions must be positive")
        if not math.isfinite(self.p_top) or self.p_top < 0.0:
            raise ValueError(
                "RRTMGP workspace p_top must be finite and nonnegative")
        xp = self._array_module
        if xp is None:
            import cupy as xp
        layouts = self._phase_layouts_input
        if layouts is None:
            from gpuwm.core.preflight import rrtmgp_workspace_phases
            layouts = rrtmgp_workspace_phases(
                self.nz, self.column_chunk, self.p_top)
        layouts = {str(phase): dict(items)
                   for phase, items in dict(layouts).items()}
        if not layouts:
            raise ValueError("RRTMGP workspace has no solver phases")
        totals = {}
        for phase, items in layouts.items():
            offset = 0
            for name, (shape, itemsize) in items.items():
                shape = tuple(int(extent) for extent in shape)
                itemsize = int(itemsize)
                if (not shape or shape[0] != self.column_chunk
                        or any(extent < 1 for extent in shape)
                        or itemsize not in (1, 4)):
                    raise ValueError(
                        f"invalid RRTMGP workspace slot {phase}/{name}: "
                        f"{shape}, itemsize={itemsize}")
                if offset % itemsize:
                    raise ValueError(
                        f"unaligned RRTMGP workspace slot {phase}/{name}")
                offset = offset + math.prod(shape) * itemsize
            totals[phase] = offset
        storage = xp.empty((max(totals.values()),), dtype=xp.uint8)
        self._storage = storage
        self._phase_layouts = MappingProxyType(layouts)

    @property
    def nbytes(self) -> int:
        return int(self._storage.nbytes)

    @property
    def storage(self):
        """The single backing allocation (identity/debug inspection only)."""
        return self._storage

    def phase(self, name: str, ncol: int) -> Mapping[str, object]:
        """Return active-prefix views for one audited solver live set."""
        if ncol < 1 or ncol > self.column_chunk:
            raise ValueError(
                f"RRTMGP chunk columns {ncol} outside 1..{self.column_chunk}")
        try:
            layout = self._phase_layouts[name]
        except KeyError as exc:
            raise KeyError(f"unknown RRTMGP workspace phase {name!r}") from exc
        views = {}
        offset = 0
        for slot, (raw_shape, itemsize) in layout.items():
            shape = tuple(int(extent) for extent in raw_shape)
            count = math.prod(shape)
            size = count * int(itemsize)
            dtype = np.bool_ if int(itemsize) == 1 else np.float32
            full = self._storage[offset:offset + size].view(dtype).reshape(shape)
            views[slot] = full[:ncol]
            offset = offset + size
        return MappingProxyType(views)


@dataclass(frozen=True)
class ModelMemoryLedger:
    """Builder record tying the runtime arena/workspace to the estimator."""

    estimate: object
    shared_scratch_arena_bytes: int
    shared_dycore_state_workspace_bytes: int
    radiation_workspace: SharedRRTMGPChunkWorkspace | None


@dataclass
class ModelRuntimeStatus:
    """Transaction state used by the tree checkpoint phase contract."""

    schedule_cursor: str = PERIOD_BEGIN
    pending_force: int = 0
    pending_feedback: int = 0
    pending_d2h: int = 0
    mutation_in_progress: bool = False
    prior_feedback_committed: bool = True


def _jsonable(value):
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (datetime, Path)):
        return value.isoformat() if isinstance(value, datetime) else str(value)
    if hasattr(value, "item"):
        return value.item()
    return value


#: The experiment fields a restart is permitted to differ in.  This IS
#: the published ``gpuwm run --restart`` contract ("only the forecast
#: length / output and restart cadence may differ"), held in one place so
#: every route that builds a restart identity excludes the same set.
RESTART_TOLERATED_EXPERIMENT_FIELDS = (
    "run_seconds", "restart_interval_s", "acknowledgements",
    # Relocation ADMISSIBILITY BOUNDS, not a trajectory input.  Nothing in
    # RelocationConfig reaches a computed value: it says which child may
    # move and how far, and a move is an explicit event recorded in its
    # own segment chain, never something these bounds derive.  Excluding
    # it is also the conservative reading of the restart contract rather
    # than a relaxation of it -- the field is new, so binding it would
    # move the fingerprint of every experiment that never mentions
    # relocation and refuse every checkpoint written before it existed.
    # Held byte-inert by tests/test_nest_relocation.py.
    "relocation",
    # [tiles] is an EXECUTION choice and the only thing it claims is
    # that it changes nothing.  A domain integrated resident and the same
    # domain streamed tile-by-tile from pinned host RAM produce the same
    # bytes -- proven carrier by carrier at every physics rung, and proven
    # again across a checkpoint in four legs (streamed -> file -> streamed,
    # streamed -> file -> MONOLITHIC through the unmodified restore path,
    # monolithic -> file -> streamed, all bit-exact).  Binding the mode
    # would therefore refuse exactly the operation it exists for: a
    # forecast that outgrew its card resuming on the card it outgrew.
    # See gpuwm.core.streaming.identity_payload_entry.
    "tiles",
    # [output] selects which variables reach the HISTORY tape
    # (gpuwm.io.history_selection).  It changes no number the model
    # computes and touches no checkpoint: checkpoints are separate files
    # written from model state by gpuwm.io.restart, never from the
    # history frame.  Binding it would refuse the operation the surface
    # exists for -- a run that filled its disk resuming with a trimmed
    # tape -- so a trimmed run resumes a full run's checkpoints and back
    # again.  Same law as "tiles" above.
    "output")
RESTART_TOLERATED_DOMAIN_FIELDS = ("history_interval_s",)
RESTART_TOLERATED_RUN_FIELDS = (
    "run_seconds", "output_interval_s", "restart_interval_s")

#: ``mp_physics`` -> the RunConfig fields ONLY that scheme reads.  A domain
#: running some other scheme drops them from its identity entirely rather
#: than carrying them at their defaults -- the absent-stays-absent
#: convention ``perturbation`` and the per-domain ``spawn`` declaration
#: already use, applied to scheme-scoped knobs.
#:
#: This is not a relaxation of the restart contract, and the test that
#: matters is whether any code path can read the field: under a different
#: ``mp_physics`` no launcher, adapter, allocator or diagnostic touches
#: them (``gpuwm/core/wdm6.py`` is the only reader of ``wdm6_hail_opt``,
#: ``gpuwm/core/state.py``'s CCN fill the only reader of
#: ``wdm6_ccn_conc``, ``gpuwm/core/nssl2_contract.py`` the only reader of
#: the five NSSL selectors, and all of them sit behind their scheme's
#: dispatch row).  ``validate_run_config`` REFUSES a non-default value on
#: any other scheme for every field listed here, so off-scheme these
#: fields can hold only their defaults and dropping them discards no
#: information.  A field no path reads cannot move a trajectory, so
#: binding it would buy no safety and would cost the thing that IS
#: load-bearing: every experiment written before these schemes existed
#: keeps its exact fingerprint, and every checkpoint those runs wrote
#: stays resumable.  Under their own scheme the fields bind, value for
#: value, because there they ARE the trajectory.
#:
#: The NSSL row was added at the 1.9 assembly.  Its five selectors landed
#: unscoped and moved the pre-NSSL fingerprint anchor
#: (``tests/test_water_overlay.py``), which is the exact regression this
#: table exists to prevent; the anchor is back at its lane-base value with
#: the row in place.  Every scheme-scoped knob goes here.
#: The mp=28 row was added at the 2.5.8 integration.  Its two aerosol-source
#: selectors (lane/wif-default) landed unscoped, exactly as the NSSL five
#: had, and moved BOTH frozen anchors in ``tests/test_water_overlay.py`` --
#: measured by payload diff against the pristine base f27fc897a: exactly
#: four added keys, these two per domain, no other change.  Unscoped they
#: would have made every 2.5.7 checkpoint refuse to resume under 2.5.8 for
#: fields only the aerosol-aware Thompson path reads (every reader in
#: gpuwm/ingest/real.py sits behind ``cfg.mp_physics == 28``).
SCHEME_SCOPED_RUN_FIELDS: dict[int, tuple[str, ...]] = {
    16: ("wdm6_hail_opt", "wdm6_ccn_conc"),
    18: ("nssl_2moment_on", "nssl_hail_on", "nssl_ccn_on",
         "nssl_density_on", "nssl_3moment"),
    28: ("wif_climatology_path", "mp28_aerosol_source"),
    # The mp=50 row was added with the P3 CUDA port.  p3_backend selects
    # WHICH implementation integrates the column, and the three arms are
    # byte-identical on every p3-fortref fixture where they have been
    # compared, so it is not even a trajectory choice -- but it is still a
    # RunConfig field, and an unscoped RunConfig field moves EVERY
    # experiment fingerprint whether or not it changes a number.  It is
    # scoped here, and gpuwm/config.py refuses a non-default value under
    # any other scheme, which is what makes the scoping lossless.
    50: ("p3_backend",),
}

#: The adaptive-timestep surface, scoped OFF the restart identity whenever
#: ``use_adaptive_time_step`` is false -- see ``restart_identity_payload``.
#: The flag itself is in the tuple because "off" is the state every
#: checkpoint written before this block existed describes, so it must drop
#: out entirely rather than appear as an added false.
ADAPTIVE_TIMESTEP_RUN_FIELDS: tuple[str, ...] = (
    "use_adaptive_time_step", "step_to_output_time", "adaptation_domain",
    "target_cfl", "target_hcfl", "max_step_increase_pct",
    "starting_time_step", "starting_time_step_den",
    "max_time_step", "max_time_step_den",
    "min_time_step", "min_time_step_den",
)

#: The same surface MINUS the feature flag: controller POLICY.
#:
#: These govern how the clock behaves from the resume point onward; they
#: are not model state, and the checkpoint does not depend on them.  A
#: restart may therefore change them -- see ``gpuwm.io.restart``, which
#: allows it and reports the change rather than refusing, because a
#: checkpoint whose controller setting cannot be altered is useless for
#: recovering the run that setting killed.
#:
#: ``use_adaptive_time_step`` is excluded deliberately.  Flipping it
#: leaves the carried controller state describing a clock that no longer
#: runs, so that stays a refusal.
ADAPTIVE_POLICY_RUN_FIELDS: frozenset[str] = frozenset(
    ADAPTIVE_TIMESTEP_RUN_FIELDS) - {"use_adaptive_time_step"}


def restart_identity_payload(exp) -> dict:
    """The experiment, minus everything a restart may legally change.

    Extracted so the prepared domain-tree runner can bind the same
    identity this function defines instead of hashing the experiment
    FILE.  A file digest moves when any byte of the TOML moves, which
    made the tree route refuse every one of the three changes the
    ``--restart`` contract publishes as permitted -- including extending
    the run, which is the reason people configure checkpoints at all.
    """

    from gpuwm.experiment import experiment_config_document
    experiment = _jsonable(experiment_config_document(exp))
    # The new attribute follower is an explicitly selected trajectory policy.
    # Bind its source, reduction, direction and movement controls while keeping
    # the pre-existing relocation exemption byte-identical for legacy signals.
    attribute_policy = experiment.get("relocation", {})
    if (attribute_policy.get("follow") or {}).get("field") == "attribute":
        attribute_policy = {key: value for key, value in attribute_policy.items()
                            if key != "track"}
    else:
        attribute_policy = None
    for name in RESTART_TOLERATED_EXPERIMENT_FIELDS:
        experiment.pop(name, None)
    if attribute_policy is not None:
        experiment["relocation"] = attribute_policy
    # ABSENT [perturbation] stays absent from the identity payload, not
    # present-as-null: every experiment written before the field existed
    # keeps its exact fingerprint and restart identity (the mixed_edges
    # convention in experiment_fingerprint, applied at the source).  A
    # configured block binds, value for value.
    if experiment.get("perturbation") is None:
        experiment.pop("perturbation", None)
    # Same convention for [spectral_numerics]: ABSENT stays absent, so
    # every pre-feature fingerprint and checkpoint is preserved.  A
    # PRESENT block binds value for value -- the Level-2 contract makes
    # the resolved operator config part of what the run IS, because a
    # resume across a changed operator (an applied filter appearing,
    # disappearing, or retuned mid-trajectory) is not the same
    # experiment and must refuse.
    if experiment.get("spectral_numerics") is None:
        experiment.pop("spectral_numerics", None)
    # The mixing-length provenance label leaves the identity
    # UNCONDITIONALLY, on the [tiles] convention: it records WHO chose
    # each domain's mix_isotropic, while the chosen value itself sits on
    # run.mix_isotropic and binds there.  An auto-selected 1 and a
    # written 1 integrate the same bytes, so each must resume the
    # other's checkpoints -- and every pre-feature fingerprint stays
    # byte-identical because the key is simply absent, as it always was.
    experiment.pop("auto_mix_isotropic", None)
    for domain in experiment.get("domains", ()):
        for name in RESTART_TOLERATED_DOMAIN_FIELDS:
            domain.pop(name, None)
        # Same convention for the per-domain spawn declaration: absent
        # stays absent (pre-feature fingerprints byte-identical), and a
        # declared spawn BINDS, value for value.  It decides trajectory --
        # when a slot fires, where the newborn lands, how many episodes it
        # may serve -- and a resume rebuilds the checkpoint's live tree
        # from the runner's state, so restoring one under a DIFFERENT
        # declaration would re-materialize episodes against a policy that
        # never produced them.  Binding it makes that a refusal.
        if domain.get("spawn") is None:
            domain.pop("spawn", None)
        # The three per-domain lifecycle tables take the SAME convention as
        # the spawn declaration beside them, and for the same two reasons in
        # both directions.
        #
        # Absent stays absent because these fields landed on DomainConfig
        # defaulting to None, and the payload is built by
        # ``dataclasses.asdict``: an unpopped None serializes as
        # ``retire: null, rearm: null, follow: null`` on EVERY domain of
        # EVERY experiment, including every experiment written before the
        # fields existed.  That is pure identity churn -- it moves each of
        # those fingerprints without changing one number any of them
        # integrates, and a moved fingerprint is a checkpoint that refuses
        # to restore.  It cost the pre-lifecycle anchor
        # (tests/test_water_overlay.py) exactly the way the NSSL row above
        # did, which is what that anchor is for.
        #
        # A DECLARED table binds, value for value, because each of the three
        # decides trajectory: [retire] says when a child stops integrating,
        # [rearm] how many episodes the slot may serve, [follow] where the
        # child sits.  A resume across a change to any of them must refuse.
        for name in ("retire", "rearm", "follow"):
            if domain.get(name) is None:
                domain.pop(name, None)
        # The per-domain [tiles] road leaves the identity UNCONDITIONALLY,
        # declared or not -- the one place this file's absent-stays-absent
        # convention is not enough.  A declared spawn binds because it
        # decides the trajectory a resume rebuilds; a declared tiling binds
        # NOTHING because its entire claim is that it changes no bytes, and
        # binding it would refuse the operation it exists for: a domain
        # that streamed resuming resident, or a domain that outgrew its
        # card resuming streamed.  Same law as the tree-wide table above
        # (RESTART_TOLERATED_EXPERIMENT_FIELDS "tiles").
        domain.pop("tiles", None)
        # The per-domain [output] selection leaves UNCONDITIONALLY too,
        # and for the same reason: it decides which variables reach the
        # HISTORY tape and binds no number the model computes.  A domain
        # that wrote the full inventory must resume trimmed, and one that
        # trimmed must resume full.
        domain.pop("output", None)
        run = domain.get("run", {})
        for name in RESTART_TOLERATED_RUN_FIELDS:
            run.pop(name, None)
        # ``eta_levels`` on the absent-stays-absent convention of the
        # blocks above: its default ``None`` means "inherit the ladder the
        # source carries", which is exactly and only what every experiment
        # written before the field existed did, so at its default it
        # carries no trajectory information and binding it was pure
        # identity churn -- the prepared-cache table already tolerated it
        # and this payload did not (ENG-010).  A DECLARED ladder binds,
        # value for value: a child on its own eta grid is a different run.
        if run.get("eta_levels") is None:
            run.pop("eta_levels", None)
        # Scheme-scoped knobs leave the identity of every domain that does
        # not select their scheme (:data:`SCHEME_SCOPED_RUN_FIELDS`).
        selected = run.get("mp_physics")
        for scheme, names in SCHEME_SCOPED_RUN_FIELDS.items():
            if selected == scheme:
                continue
            for name in names:
                run.pop(name, None)
        # THE SURFACE-RADIATION CARRIER POLICY, on the same convention the
        # perturbation and spawn blocks above use: the DEFAULT drops out of
        # the identity payload, so every experiment written before the
        # policy existed keeps its exact fingerprint and restart identity,
        # and the DECLARED ESCAPE binds.  That is the whole requirement --
        # "wrf_compat_zero" has to be part of what a run IS, because a leg
        # that consumed a sky nobody computed is not the same experiment as
        # one that did not, and a resume across the change must refuse.  A
        # label that changed every existing fingerprint while changing no
        # trajectory would be identity churn, which is the opposite of what
        # identity is for.
        if run.get("surface_radiation_policy") == (
                SURFACE_RADIATION_POLICY_REQUIRED):
            run.pop("surface_radiation_policy", None)
        # THE ADAPTIVE-TIMESTEP BLOCK, on that same convention and for the
        # same reason.  Twelve fields joined RunConfig at once, and an
        # unscoped RunConfig field moves EVERY experiment fingerprint
        # whether or not it changes a number -- which would have refused
        # every checkpoint already written, for a controller that is off.
        #
        # OFF is the whole argument, and it is stronger here than
        # "the default value": with use_adaptive_time_step false the
        # controller never runs, so the other eleven are read by nothing
        # at all and cannot have influenced a single byte of the
        # trajectory.  Dropping them discards no information.
        #
        # ON, the flag and the CONFIGURED step still bind value for value.
        #
        # The POLICY does not, and this paragraph used to say it did: "a
        # resume that changed target_cfl, a clamp, or the growth bound
        # would be integrating on a different clock from the leg that
        # wrote the checkpoint, and that is not the same experiment."
        # True, and the conclusion drawn from it was wrong -- integrating
        # on a different clock from here on is precisely what a recovery
        # resume is for, and binding it made restart_interval_s useless
        # for the one case that most wants it: a run that died because of
        # the setting you now need to change.
        #
        # `gpuwm.io.restart` allows such a resume and REPORTS the change,
        # naming each field and its old -> new value.  This fingerprint
        # has to agree with that, or it refuses what the walk beside it
        # just permitted -- which is what happened, and it surfaced as a
        # relocation-lineage refusal because a moved tree picks that
        # message for ANY fingerprint mismatch.
        #
        # `dt` stays bound: the identity carries the CONFIGURED step, not
        # the adapted one, so it is stable across a run and changing it
        # really is a different experiment.
        if not run.get("use_adaptive_time_step", False):
            for name in ADAPTIVE_TIMESTEP_RUN_FIELDS:
                run.pop(name, None)
        else:
            for name in ADAPTIVE_POLICY_RUN_FIELDS:
                run.pop(name, None)
    return experiment


def experiment_fingerprint(exp, catalog) -> str:
    """Hash immutable trajectory setup plus the InputCatalog inventory.

    Forecast duration and output/restart cadence are deliberately excluded:
    they can change on resume without changing any integration before or
    after the checkpoint.  Domain geometry, timestep, physics, nesting,
    forcing provenance, and every other experiment field remain bound.
    """
    experiment = restart_identity_payload(exp)
    payload = {
        "experiment": experiment,
        "input_catalog": _jsonable(catalog.run_provenance),
    }
    # A mixed-scheme restart is trajectory-compatible only with the exact
    # translator implementation, field map, and policy metadata.  Keep the
    # legacy/same-scheme payload byte-inert, but bind every mixed directed
    # edge to its complete executable identity.
    from gpuwm.core.microphysics_transition import (
        resolve_microphysics_transition,
    )
    by_id = {domain.grid_id: domain for domain in exp.domains}
    mixed_edges = []
    for domain in exp.domains:
        if domain.parent_id == 0:
            continue
        contract = resolve_microphysics_transition(
            by_id[domain.parent_id].run, domain.run)
        if contract.mixed:
            mixed_edges.append({
                "source_domain": int(domain.parent_id),
                "target_domain": int(domain.grid_id),
                "contract": _jsonable(contract.receipt()),
            })
    if mixed_edges:
        payload["mixed_microphysics_transitions"] = mixed_edges
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def uses_modern_rrtmgp_workspace(exp) -> bool:
    """Whether the tree build must allocate the shared RRTMGP workspace.

    Variant-aware, mirroring ``estimate_experiment``: the persistent
    :class:`SharedRRTMGPChunkWorkspace` exists only for the modern
    RTE+RRTMGP adapter.  The legacy-RRTMG variant runs one domain at a
    time under its engine-default ``column_chunk=None`` contract with
    LW/SW allocations sequenced and freed between calls, so its VRAM
    price is the estimator's transient call-peak envelope
    (``workspace_bytes`` under ``uses_legacy``), never a held workspace
    -- constructing (and cross-checking) the modern workspace for it
    both wastes the allocation and trips the memory-ledger drift guard.
    Mixed variants retain this modern workspace while legacy engines run.
    """
    from gpuwm.physics_compat import RRTMG_VARIANT_LEGACY, rrtmg_variant

    variants_44 = {rrtmg_variant(dc.run) for dc in exp.domains
                   if 4 in radiation_scheme_ids(dc.run)}
    return bool(variants_44) and variants_44 != {RRTMG_VARIANT_LEGACY}


def _forcing_cadence_seconds(catalog) -> float:
    """Resolve T9's LBC cadence from decoded InputCatalog records."""
    records = tuple(catalog.lbc_records)
    if not records:
        raise ValueError("the forcing catalog has no LBC interval")
    deltas = {float(record.delta_seconds) for record in records}
    if len(deltas) != 1:
        raise ValueError(
            "the integer DomainClock requires one uniform forcing cadence; "
            f"InputCatalog deltas are {sorted(deltas)}")
    return deltas.pop()


def publish_declared_experiment(model, exp) -> None:
    """Publish, on the tree, the experiment the tree was configured from.

    THE ONE CARRIER, for every route.  A checkpoint that persists a live
    follower has to record where each domain was DECLARED alongside where
    it now sits, or a resume cannot tell a nest the follower moved from a
    nest somebody reconfigured -- and the declaration is a fact about the
    configuration, not about whichever ingest built the tree.  Any route
    that assembles an :class:`ExperimentState` calls this, so restart
    stays a capability of the tree rather than of the route.
    """
    if exp is None:
        raise ValueError(
            "a tree is published with the experiment it was configured "
            "from; publishing None leaves the checkpoint writer unable to "
            "say whether a follower's recorded placement is the declared "
            "one or one the follower moved to")
    model._declared_experiment = exp


def build_experiment(exp, case_data) -> ExperimentState:
    """Build the all-resident domain tree parent before child.

    The real root follows the existing Task-2 preparation path.  Each child
    then follows Task 12's WRF-order initialization and receives its own
    physics driver and Task-13 coupler.  Every state is constructed against
    the one shared transient arena before any scratch slot is materialized.
    """
    from gpuwm import runtime
    from gpuwm.core.clock import build_schedule, resolve_clock
    from gpuwm.core.preflight import estimate_experiment
    from gpuwm.core.state import (build_shared_dycore_state_workspace,
                                  build_shared_scratch_arena)
    from gpuwm.ingest.grib import clear_forcing_caches
    from gpuwm.ingest.lateral_bc import bind_lateral_boundary_clock
    from gpuwm.ingest.nest_init import initialize_child
    from gpuwm.ingest.preflight import build_input_catalog
    from gpuwm.core.nest import NestCoupler as ConcreteNestCoupler

    catalog = build_input_catalog(case_data)
    snapshots = runtime.forcing_snapshots(case_data, catalog)
    forcing_times = runtime.forcing_schedule(exp, case_data, snapshots)
    lbc_interval_s = _forcing_cadence_seconds(catalog)
    # Dormant (spawn-declared) nests are RESERVED but not INTEGRATED:
    # the memory plan, the shared arenas and the fingerprint below price
    # and bind the FULL experiment (a declared-but-never-triggered nest
    # costs its reserved VRAM and zero compute -- that is the contract),
    # while the clock, the schedule and the child-init loop see only the
    # active tree.  Mid-run activation is the spawn runner's leg
    # boundary (gpuwm.experiment.active_experiment), not this builder's.
    from gpuwm.experiment import pre_spawn_experiment
    active = pre_spawn_experiment(exp)
    tick_clock = resolve_clock(active, lbc_interval_s=lbc_interval_s)
    schedule = build_schedule(active, tick_clock)
    clocks = tick_clock.clocks()
    estimate = estimate_experiment(
        exp, forcing_interval_seconds=lbc_interval_s)
    dycore_state_workspace = (
        build_shared_dycore_state_workspace(exp.domains)
        if len(exp.domains) > 1 else None)
    if (dycore_state_workspace is not None
            and dycore_state_workspace.nbytes
            != estimate.dycore_state_workspace_bytes):
        raise RuntimeError(
            "runtime shared dycore-state workspace drifted from the memory "
            f"ledger: {dycore_state_workspace.nbytes} != "
            f"{estimate.dycore_state_workspace_bytes} bytes")
    arena = (build_shared_scratch_arena(exp.domains)
             if len(exp.domains) > 1 else None)
    if arena is not None and arena.nbytes != estimate.scratch_arena_bytes:
        raise RuntimeError(
            "runtime scratch arena drifted from the memory ledger: "
            f"{arena.nbytes} != {estimate.scratch_arena_bytes} bytes")
    # Variant-aware (F' contract): the shared chunk workspace is a modern
    # RTE+RRTMGP object.  Under ra_rrtmg_variant='rrtmg_legacy' the
    # estimator's workspace_bytes is the legacy transient call-peak
    # envelope, not a persistent allocation, and the legacy adapter keeps
    # its engine-default column_chunk=None contract -- so no workspace is
    # built (and no chunking attributes are stamped onto the adapter).
    radiation_workspace = (
        SharedRRTMGPChunkWorkspace(
            nz=exp.root.run.nz, column_chunk=exp.column_chunk,
            p_top=exp.vertical.p_top)
        if uses_modern_rrtmgp_workspace(exp) else None)
    if (radiation_workspace is not None
            and radiation_workspace.nbytes != estimate.workspace_bytes):
        raise RuntimeError(
            "runtime RRTMGP workspace drifted from the memory ledger: "
            f"{radiation_workspace.nbytes} != {estimate.workspace_bytes} bytes")
    ledger = ModelMemoryLedger(
        estimate=estimate,
        shared_scratch_arena_bytes=(0 if arena is None else arena.nbytes),
        shared_dycore_state_workspace_bytes=(
            0 if dycore_state_workspace is None
            else dycore_state_workspace.nbytes),
        radiation_workspace=radiation_workspace)

    prepared_root = runtime.prepare_root_experiment_case(
        exp, case_data, input_catalog=catalog,
        forcing_by_time=snapshots, scratch_arena=arena,
        dycore_state_workspace=dycore_state_workspace)
    # The raw decode is spent: the root's initial state and every
    # boundary frame are built from it above, and nothing below reads it.
    # The children about to be initialized -- and the mid-run delayed
    # start, spawn and relocation initializers that reach the same door
    # later -- take their source snapshot from the CATALOG
    # (gpuwm.ingest.nest_init._initial_snapshot), which holds its own
    # frozen copies and is untouched here.  Held to the end of the run
    # this made the ERA5 decode, not the forecast, the host-memory
    # binding phase: the catalog's copies, a SECOND frozen set under the
    # runtime's own cache key, and the partials both were copied from.
    # The mapping is dropped as well as the caches -- a cleared cache
    # frees nothing while a local still names the arrays.  Byte-inert:
    # these caches memoize a pure function of immutable input bytes, so
    # the only thing a later decode of the same key loses is time.
    del snapshots
    clear_forcing_caches()
    root_dc = exp.root
    root = DomainNode(
        cfg=root_dc, grid=prepared_root.grid,
        state=prepared_root.initial_result.state,
        clock=clocks[root_dc.grid_id], parent=None, children=[], coupler=None)
    root._started = True
    # Davies clock bind (seam closure 2026-07-28, retires the F20
    # adjudication): the root's external Davies launches consume WRF's
    # post-increment dtbc.  WRF resets dtbc on every boundary read
    # (share/mediation_integrate.F:1515-1522) and adds one model step at
    # solve entry BEFORE any relax/spec consumer (dyn_em/solve_em.F:
    # 371-372); the executor's on_lbc_reset + prepare_step already runs
    # that exact recurrence on the root DomainClock, so binding here --
    # after attachment, before any solve or restart restoration -- makes
    # every root boundary launch take dtbc_launch_fp32 (dt..T_bdy) in
    # place of the retired one-step-lagged elapsed-based value (0..T-dt).
    # The former frozen-Phase-4 anchor bytes encode the retired
    # semantics; the N-series ratchets regenerate against the seam-
    # closure anchor epoch (PROVENANCE.md "Root external-boundary dtbc").
    bind_lateral_boundary_clock(root.state, root.clock)
    nodes: dict[int, DomainNode] = {root_dc.grid_id: root}
    prepared_by_id = {root_dc.grid_id: prepared_root}

    from gpuwm.core.cam_ozone import configure_cam_ozone
    configure_cam_ozone(root.state, root_dc.run, exp=exp, dc=root_dc, grid=root.grid)
    root_radiation = (root.state.physics.radiation_callable
                      if root.state.physics is not None else None)
    from gpuwm.core.radiation_composition import attach_modern_workspace
    attach_modern_workspace(root_radiation, radiation_workspace)

    initial_perturbation_receipts = []
    if exp.perturbation is not None:
        initial_perturbation_receipts.append(
            dict(prepared_root.initial_result.initial_perturbation))

    for dc in active.domains[1:]:
        parent = nodes[dc.parent_id]
        # A child that starts WITH the experiment applies the configured
        # initial-state bubbles on its own grid (real-data nest init
        # re-ingests the source per domain, so nothing arrives from the
        # parent's theta).  A delayed-start child initializes from the
        # analysis at its activation time -- the parent has evolved the
        # bubble by then -- so it takes no fresh analytic bubble.
        child_starts_at_t0 = clocks[dc.grid_id].spec.start_ticks == 0
        initialized = initialize_child(
            dc, parent, catalog, exp.vertical,
            source_orography=case_data.source_orography,
            scratch_arena=arena,
            dycore_state_workspace=dycore_state_workspace,
            sfcp_to_sfcp=case_data.sfcp_to_sfcp,
            initial_perturbation=(
                exp.perturbation if child_starts_at_t0 else None))
        if exp.perturbation is not None:
            initial_perturbation_receipts.append(
                dict(initialized.real.initial_perturbation)
                if child_starts_at_t0 else {
                    "grid_id": int(dc.grid_id),
                    "applied": False,
                    "reason": "delayed start: this domain initializes "
                              "from the analysis at its activation time, "
                              "after the perturbation instant",
                })
        prepared = runtime.prepare_child_case(
            initialized, dc, exp=exp, data=case_data,
            forcing_times=forcing_times,
            radiation_workspace=radiation_workspace,
            # Legacy-RRTMG ozone routing (WRF computes o33d on the root
            # domain only; children receive parent-interpolated fields):
            # the child's adapter takes the parent's radiation callable
            # as its ozone provider.  Inert for every other variant.
            radiation_parent=parent.state.physics.radiation_callable)
        node = DomainNode(
            cfg=dc, grid=initialized.grid, state=initialized.state,
            clock=clocks[dc.grid_id], parent=parent,
            children=[], coupler=None)
        node.coupler = ConcreteNestCoupler(
            node, feedback=exp.feedback,
            smooth_option=exp.smooth_option)
        node._started = clocks[dc.grid_id].spec.start_ticks == 0
        parent.children.append(node)
        # The marker is present both before and after FORCE.  It makes the
        # restart setup fingerprint independent of rolling table contents.
        node.state._nest_restart_classification = "REBUILT"
        nodes[dc.grid_id] = node
        prepared_by_id[dc.grid_id] = prepared
        if exp.feedback == 1 and node._started:
            initial = FeedbackScratch()
            node.coupler.feedback_prepare(node, initial)
            node.coupler.feedback_commit(node)
            node.coupler.feedback_finalize(node)

    built = ExperimentState(
        root=root, nodes_by_grid_id=MappingProxyType(nodes),
        schedule=schedule, memory_ledger=ledger,
        experiment_fingerprint=experiment_fingerprint(exp, catalog))
    # Keep the wave-1 dataclass contract stable: Task-14 runtime machinery is
    # attached as non-field infrastructure, so T12/T13 consumers remain inert.
    built._scratch_arena = arena
    built._dycore_state_workspace = dycore_state_workspace
    built._prepared_by_grid_id = prepared_by_id
    built._initial_perturbation_receipts = tuple(
        initial_perturbation_receipts)
    built._input_catalog = catalog
    publish_declared_experiment(built, exp)
    built._activation_context = {
        "experiment": exp,
        "case_data": case_data,
        "forcing_times": forcing_times,
        "radiation_workspace": radiation_workspace,
    }
    built._runtime_status = ModelRuntimeStatus()
    built._feedback_provenance = runtime.feedback_provenance(exp)
    built._resumed = False
    built._resume_committed_history_grid_ids = frozenset()
    built._io_manager = None
    built._last_checkpoint = None
    return built


def _trim_default_pool() -> None:
    """Release UNUSED cached CuPy pool blocks back to the driver.

    Live allocations are untouched, so this cannot change any computed
    value (see execute_experiment's pool_trim_per_period).

    THE BREAKAGE THE GUARD PREVENTS.  A trim is an optimisation, never a
    correctness step, but it runs at every step boundary -- so when the
    pool itself cannot be reached, raising here aborts a run that never
    needed a device.  That is reachable with CuPy INSTALLED and every
    device masked (``CUDA_VISIBLE_DEVICES=-1``, which tests/conftest.py
    plants for the whole session): the first pool call answers
    ``cudaErrorNoDevice`` and killed host-only nest runs at their first
    step boundary.  An unreachable pool leaves the cache where it is,
    which is the same state a trim that freed nothing would leave.
    """
    import cupy as cp
    try:
        cp.get_default_memory_pool().free_all_blocks()
    except cp.cuda.runtime.CUDARuntimeError:
        return


def execute_experiment(
        model: ExperimentState, *, history_handler=None,
        restart_handler=None, progress_callback=None,
        validate_state: bool = True, health_debug: bool = False,
        arena_nan_poison: bool = False,
        skip_feedback_path: bool = False,
        pool_trim_per_period: bool = True,
        relocation_runner=None, steppers=None, step_observer=None,
        experiment=None, delayed_child_initializer=None):
    """Wire one :class:`ExperimentState` into ``execute_schedule``.

    STEP calls the domain's existing dycore, FORCE calls its node-facing
    coupler, and FEEDBACK walks the dormant three-phase transaction.  The
    clock module remains the only op-table walker and therefore the only
    source of runtime scheduling comparisons.

    ``relocation_runner`` is a
    :class:`gpuwm.core.relocation_runner.RelocationRunner` constructed by
    the route (the route owns the child-physics preparer and, for
    ``follow = "provider"``, the live tracker).  It is consulted at every
    PERIOD_BEGIN -- the complete cycle boundary where all clocks are
    synchronized, so every relocation lands on a parent-step boundary --
    and a moved child gets a fresh health validator armed immediately.  A
    configuration that schedules relocation on a route that did not wire a
    runner refuses here, at start, rather than integrating a nest that
    silently never follows anything.

    ``pool_trim_per_period`` releases the CuPy default pool's UNUSED
    cached blocks at every period commit.  Measured on the 4-domain
    real74 shape (2026-07-17 5-sim-min A/B): step-time churn re-inflates
    the pool from ~21 to ~30 GiB held against ~20 GiB live, driving WDDM
    page demotion; per-period trimming held the pool at 21.5-23.4 GiB,
    kept the device below the demotion band, and cut wall time 32%.
    Byte-inert by construction -- free_all_blocks releases only unused
    cached blocks, never live allocations, so no computed value can
    change; allocator behavior is not part of any ratified comparator.
    Disable only for allocator forensics.

    ``steppers`` is ``{grid_id: callable}``, the callable a STEP op steps
    that domain with.  ``None`` -- and any grid absent from a supplied
    mapping -- binds ``gpuwm.core.dycore.step`` ITSELF, so an experiment
    that configures nothing runs the identical function through the
    identical call, not a wrapper that forwards to it.  A domain whose
    ``[tiles]`` fired binds a
    :class:`gpuwm.core.streaming.StreamedDomain` instead, which has the
    same signature and advances the same domain by the same one model
    step -- out of a pinned host store, one tile of it on the card at a
    time.  Everything around the call is unchanged BECAUSE the loop is
    unchanged: the same clock refresh either side, the same health
    validators, the same ``refl_10cm_due`` handshake, the same history
    and restart ops on the same cadences.

    ``delayed_child_initializer`` optionally supplies a route's dated
    child analysis at activation. It receives the child node and exact
    clock and returns ``(initialized, prepared)`` with the same grid/state
    and prepared-case attributes as the ordinary case-data initializer.
    The executor still owns the clock refresh, coupler, initial feedback,
    health validation and activation marker. Omission uses the input catalog.

    ``step_observer`` is called once per DOMAIN per model time step,
    immediately after that domain's STEP op commits, with
    ``grid_id``/``step_count``/``model_seconds``/``step_wall_seconds``.
    It is what lets a front door print WRF's ``Timing for main:`` line
    per step (:class:`gpuwm.progress_log.StepLog`); ``progress_callback``
    below cannot, because it fires once per ROOT step and a nest taking
    36 substeps inside one of those is invisible to it.  ``None`` -- the
    default -- adds one ``is not None`` test per STEP op and changes
    nothing else.
    """
    from gpuwm.core.clock import execute_schedule
    from gpuwm.core.dycore import step
    from gpuwm.core.state import refresh_model_time

    status = model._runtime_status
    context = model._activation_context
    if relocation_runner is None and context is not None:
        exp = context.get("experiment")
        relocation = None if exp is None else exp.relocation
        # A follow target that is still DORMANT is not a void arm: the
        # nest does not exist yet, so there is nothing to move and no
        # grid to anchor a runner on.  The leg walk
        # (gpuwm.runtime.walk_spawn_legs) wires the runner at the birth
        # boundary, which is the first instant that nest could legally
        # move -- so the refusal is DEFERRED here, not waived.  A target
        # that is present in the tree still refuses exactly as before.
        target = None if relocation is None else relocation.grid_id
        # Attribute access in a try, never reflection: this module is
        # AST-audited against getattr/setattr
        # (tests/test_clock.py::test_no_float_elapsed_accumulation_audit),
        # and the only callers that lack these attributes are the stub
        # models in the refusal tests, for which no carve-out applies.
        try:
            dormant_target = (
                target is not None
                and int(target) not in model.nodes_by_grid_id
                and exp is not None
                and any(dc.grid_id == int(target) and dc.spawn is not None
                        for dc in exp.domains))
        except AttributeError:
            dormant_target = False
        if (relocation is not None and relocation.enabled
                and not dormant_target
                and (relocation.follow is not None or relocation.moves)):
            source = ("[relocation.follow]" if relocation.follow is not None
                      else "[[relocation.move]]")
            raise RuntimeError(
                f"[relocation] configures a follow source ({source}) but "
                "this route wired no RelocationRunner; a nest that "
                "silently never moves is a void experiment arm, so this "
                "refuses instead.  The case-data route wires the real-data "
                "runner (gpuwm.runtime.build_real_relocation_runner: "
                "footprint-rebuilt statics + the land-surface preparer); "
                "any other route constructs "
                "gpuwm.core.relocation_runner.RelocationRunner and passes "
                "it here, or drives relocate_child directly")
    # Resolved to a plain dict here, keyed by grid, so a delayed-start child
    # that joins the tree later falls back to the dycore's own step rather
    # than to whatever the last domain happened to bind.
    steppers = dict(steppers or {})
    # THE LEVEL-2 SPECTRAL SEAM (gpuwm.spectral_seam).  ``None`` -- every
    # experiment without an active [spectral_numerics] -- costs the STEP
    # op one ``is not None`` test and nothing else.  Active, it refuses a
    # streamed domain and a false periodic declaration HERE, at start,
    # and the STEP op below calls it once per domain immediately after
    # the slow RK state commit.  Attached to the model so leg walks
    # (spawn, restart) that re-enter this function keep one receipt
    # ledger for the whole run.
    # ``experiment`` is the AUTHORITY here and the activation context is
    # only the fallback, because the two prepared routes -- the ones the
    # Level-2 contract names as honoring the config -- build their own
    # ``ExperimentState`` and therefore have no activation context at
    # all.  Reading the context alone made an active [spectral_numerics]
    # invisible on exactly those routes: a shadow tree run on real HRRR
    # bytes wrote zero receipts and still emitted a clean PASS capsule
    # (2026-08-18), which is the silently-absent operator the
    # honored-or-refused governance exists to forbid.
    from gpuwm.spectral_seam import attach_seam
    seam_experiment = experiment
    if seam_experiment is None and context is not None:
        seam_experiment = context.get("experiment")
    spectral_seam = attach_seam(model, seam_experiment, steppers)
    feedback_scratch = {
        node.cfg.grid_id: FeedbackScratch()
        for node in model.walk_parent_first() if node.parent is not None}
    arena = model._scratch_arena
    dycore_state_workspace = model._dycore_state_workspace
    io_manager = model._io_manager
    restart_enabled = model.root.clock.spec.restart_ticks is not None

    validators = {}
    if validate_state:
        # A streamed domain lives in its canonical store, even when preparation
        # began resident. Bind the existing health rules to that store;
        # unstreamed domains keep the existing resident validator.
        from gpuwm.core.health import health_validator_for_domain
        validators = {node.cfg.grid_id: health_validator_for_domain(model, node)
                      for node in model.walk_parent_first()}
        for validator in validators.values():
            validator.require_healthy(phase="initialized-or-restored")

    def poison() -> None:
        if arena_nan_poison and arena is not None:
            arena.poison()

    def domain_turn(owner):
        if dycore_state_workspace is None:
            return nullcontext()
        return dycore_state_workspace.acquire(owner)

    def _streamed(grid_id):
        """The streamed stepper for one grid, or ``None`` if it is resident.

        ``steppers`` holds ``dycore.step`` for nothing -- a resident grid is
        simply absent -- so this is the whole test.
        """
        from gpuwm.core.streaming import is_streaming

        stepper = steppers.get(int(grid_id))
        return stepper if is_streaming(stepper) else None

    def allocation_scope(grid_id):
        stream = _streamed(grid_id)
        return nullcontext() if stream is None else stream.allocation_scope()

    def on_period_begin(period, clocks) -> None:
        status.schedule_cursor = PERIOD_BEGIN
        status.prior_feedback_committed = True
        if relocation_runner is None:
            return
        # A STREAMED parent keeps its arrays in a store, not on its state,
        # and the whole of relocation reads the state: the tracker reduces
        # the parent's UH plane, the runner zeroes that plane, and the
        # rebuild takes a full SINT of the live parent.  With a host store
        # every one of those reads t=0 unless the store is projected onto
        # the state first -- and the zeroing has to be projected BACK, or
        # the window silently stops meaning "since I last looked".
        #
        # Only on a cadence boundary, and only for the target's parent.  The
        # planes are (ny, nx); the SINT source is the whole state, which is
        # why it is done here (once per cadence) and not per step.
        from gpuwm.core.streaming import TRACKER_PLANE_CARRIERS

        parent_streams = {}
        # Bare runner stubs in the refusal controls predate collections.
        try:
            collection = bool(relocation_runner.is_collection)
        except AttributeError:
            collection = False
        due_targets = ()
        if collection:
            due_targets = tuple(
                gid for gid in relocation_runner.target_grid_ids
                if gid in model.nodes_by_grid_id
                and relocation_runner.runners[gid].is_due(model, clocks))
        elif relocation_runner.is_due(model, clocks):
            due_targets = (int(relocation_runner.config.grid_id),)
        for gid in due_targets:
            target = model.node(gid)
            runner = relocation_runner.runners[gid] if collection else relocation_runner
            try:
                reconstruction_factory = runner.streamed_reconstruction_factory
            except AttributeError:
                reconstruction_factory = None
            store_rebuild = (_streamed(gid) is not None
                             and callable(reconstruction_factory))
            if _streamed(gid) is not None and not store_rebuild:
                raise RuntimeError(
                    f"[relocation] targets grid {gid}, which is STREAMED. "
                    "Relocation must rebuild its store, tile plan, geography "
                    "gathers and packed nest-table windows before replacing "
                    "the live stepper; that reconstruction is not implemented. "
                    "Run the moving child resident (the parent may stream).")
            if target.parent is not None and not store_rebuild:
                parent_gid = int(target.parent.cfg.grid_id)
                parent_stream = _streamed(parent_gid)
                if parent_stream is not None and parent_gid not in parent_streams:
                    # The tracker reads its canonical plane directly; an
                    # accepted move also needs the existing full-state donor.
                    parent_stream.sync_to_state()
                    parent_streams[parent_gid] = parent_stream
        outcome = relocation_runner.on_period_begin(
            model, clocks, period=period,
            # The executor's validator holds live references into the
            # outgoing child; dropping it here is what lets the
            # host-staged release actually free the device bytes.
            before_rebuild=lambda gid: validators.pop(gid, None))
        for parent_stream in parent_streams.values():
            # reset_tracker_window already zeroes every live copy, including
            # generated follower carriers. Preserve the legacy fixed-plane
            # synchronization for providers that write those planes directly.
            parent_stream.sync_from_state(TRACKER_PLANE_CARRIERS)
        rows = [] if outcome is None else (
            outcome.get("outcomes", []) if outcome.get("event") == "batch"
            else [outcome])
        for row in rows:
            if row.get("event") != "relocated":
                continue
            gid = int(row["grid_id"])
            if validate_state:
                from gpuwm.core.health import health_validator_for_domain
                validators[gid] = health_validator_for_domain(model, model.node(gid))
                validators[gid].require_healthy(
                    phase=f"post-relocation.d{gid:02d}")

    def on_domain_start(grid_id, clock) -> None:
        """Run the ordinary WRF-order child init at its delayed boundary."""
        from gpuwm.core.nest import NestCoupler as ConcreteNestCoupler
        from gpuwm.ingest.nest_init import initialize_child
        from gpuwm.runtime import prepare_child_case

        node = model.node(grid_id)
        if node.parent is None:
            raise RuntimeError("the root domain cannot have a delayed start")
        if bool(node._started):
            return
        context = model._activation_context
        exp = context["experiment"]
        # Deliberately no initial_perturbation here: a delayed-start
        # child initializes from the analysis at its ACTIVATION time,
        # after the perturbation instant, and the receipts already say
        # so (build_experiment's delayed-start row).
        if delayed_child_initializer is None:
            data = context["case_data"]
            initialized = initialize_child(
                node.cfg, node.parent, model._input_catalog, exp.vertical,
                source_orography=data.source_orography,
                scratch_arena=arena,
                dycore_state_workspace=dycore_state_workspace,
                sfcp_to_sfcp=data.sfcp_to_sfcp)
            prepared = prepare_child_case(
                initialized, node.cfg, exp=exp, data=data,
                forcing_times=context["forcing_times"],
                radiation_workspace=context["radiation_workspace"],
                radiation_parent=(
                    node.parent.state.physics.radiation_callable))
        else:
            initialized, prepared = delayed_child_initializer(node, clock)
        node.grid = initialized.grid
        node.state = initialized.state
        node.coupler = ConcreteNestCoupler(
            node, feedback=exp.feedback,
            smooth_option=exp.smooth_option)
        refresh_model_time(node.state, clock)
        node.state._nest_restart_classification = "REBUILT"
        node._started = True
        model._prepared_by_grid_id[grid_id] = prepared
        if exp.feedback == 1:
            initial = FeedbackScratch()
            node.coupler.feedback_prepare(node, initial)
            node.coupler.feedback_commit(node)
            node.coupler.feedback_finalize(node)
        if validate_state:
            from gpuwm.core.health import health_validator_for_domain
            validators[grid_id] = health_validator_for_domain(model, node)
            validators[grid_id].require_healthy(
                phase=f"delayed-start.d{grid_id:02d}")

    def on_step(grid_id, clock) -> None:
        # WRF prints one `Timing for main:` line per model time step per
        # domain, and THIS is the only place that number exists: the
        # period-commit callback below fires once per ROOT step, so a
        # nest taking 36 substeps inside one of them was invisible to
        # every progress consumer this package had.  One perf_counter
        # pair, host-side, deliberately WITHOUT a device synchronise --
        # syncing per step to get a "true" GPU time would serialise the
        # pipeline the timing exists to observe, and WRF's own number is
        # host wall time across the step's launches too.
        started_wall = time.perf_counter()
        with domain_turn(("STEP", grid_id)):
            node = model.node(grid_id)
            if adaptive_driver is not None:
                # After any relocation rebuild, before the solve.
                adaptive_driver.before_step(grid_id)
            if node.clock is not clock:
                raise RuntimeError(
                    "DomainNode clock identity drifted from executor")
            if node.parent is not None:
                if node.coupler is None or not bool(node.coupler.valid):
                    raise AssertionError(
                        f"child d{grid_id:02d} STEP attempted with INVALID "
                        "nest tables; ordinary parent STEP -> FORCE is "
                        "required")
            if validators and health_debug:
                validators[grid_id].require_healthy(
                    phase=f"pre-step.d{grid_id:02d}")
            status.schedule_cursor = "EXECUTING"
            # Kernel-facing curr_secs mirrors WRF REAL; after the solve the
            # public state clock is refreshed from exact integer ticks.
            refresh_model_time(node.state, clock, kernel_launch=True)
            # The clock is the CALENDAR AUTHORITY, and a streamed domain
            # does not read the state it was just written onto: its tiles
            # take elapsed_seconds from the sweep's scalar carriers.  Impose
            # it here, from the same value the resident kernels get, or the
            # domain runs a second free-running clock and every physics
            # cadence -- and dtbc -- is evaluated against it.
            streamed_here = _streamed(grid_id)
            if streamed_here is not None:
                streamed_here.impose_clock(node.state.elapsed_seconds)
            steppers.get(grid_id, step)(
                node.state, node.cfg.run,
                # REFL_10CM is a one-frame producer/consumer handoff. A
                # headless forecast still advances history alarms in the
                # clock report, but has no output consumer, so it must not
                # stage a field that can never be consumed.
                refl_10cm_due=(
                    history_handler is not None
                    and clock.history_rings_within_step()))
            refresh_model_time(node.state, clock, after_step=True)
            if spectral_seam is not None:
                # THE Level-2 slow-large-step hook seam.  The RK slow-mode
                # state above is committed and the acoustic substeps are
                # over; output, nest feedback and the next large step have
                # not happened.  Fires once per domain per model time
                # step.  The step is numbered the way step_observer below
                # numbers it -- the clock still holds pre-step ticks, so
                # the committed step is step_count + 1 -- and shadow/apply
                # receipts land in the seam's ledger for the run capsule.
                spectral_seam.after_step(
                    node.state, grid_id, node.cfg.run,
                    step_count=clock.step_count + 1,
                    model_seconds=((clock.ticks + clock.step_ticks)
                                   / clock.tick_den),
                    streamed=streamed_here is not None)
            if validate_state:
                # Adaptive single-domain runs use this executor too. Keep the
                # existing per-step NaN and live-layer CFL gate before output,
                # feedback or checkpoint consumers can observe this state.
                # Streamed reports come from the canonical sweep, not its
                # unchanged resident initialization snapshot.
                from gpuwm.core.streaming import step_health

                report = step_health(steppers.get(grid_id), node.state,
                                     node.cfg.run,
                                     boundary_width=node.cfg.run.spec_bdy_width)
                if report["nan"]:
                    raise RuntimeError(
                        f"d{grid_id:02d} integration produced a non-finite state "
                        f"at model step {clock.step_count + 1}")
                cfl = report["cfl"]
                if cfl is not None and not math.isfinite(float(cfl)):
                    raise RuntimeError(
                        f"d{grid_id:02d} integration produced a non-finite "
                        f"vertical Courant number at model step {clock.step_count + 1}: "
                        "a model layer's live thickness is non-positive or non-finite")
            poison()
            if validators and health_debug:
                validators[grid_id].require_healthy(
                    phase=f"post-step.d{grid_id:02d}")
            if pool_trim_per_period:
                # Step-cadence trim: one root period holds up to 36 d04
                # substeps of transients; trimming only at period commit let
                # the pool refill past the WDDM ceiling on the 4-domain
                # shape (measured 32.2 GiB peak, 50 s above 31.5 GiB in a
                # 15-sim-min run). Same byte-inert release, per STEP op.
                _trim_default_pool()
        if step_observer is not None:
            # Outside the turn, after the step committed: a step that
            # raised is not a step that happened.  Telemetry never fails
            # a run, so a sink that raises is dropped for that one line.
            #
            # POST-STEP values, and they have to be computed rather than
            # read: `execute_schedule` calls `dom.advance()` AFTER this
            # hook returns (gpuwm/core/clock.py, the STEP op), so the
            # clock still holds the state the step started from.
            # Reporting it raw would number the first step 0 and stamp
            # it with the previous step's valid time -- off by one, on
            # every line, in the direction a reader cannot detect.  Same
            # integer-tick derivation `refresh_model_time(after_step=True)`
            # uses: one FP64 division of exact integers, never an
            # accumulation.
            try:
                step_observer(
                    grid_id=grid_id, step_count=clock.step_count + 1,
                    model_seconds=((clock.ticks + clock.step_ticks)
                                   / clock.tick_den),
                    step_wall_seconds=time.perf_counter() - started_wall,
                    # The step JUST TAKEN, from the same integer-tick
                    # lattice as model_seconds above -- not the step the
                    # clock is about to adopt.  Under an adaptive dt the
                    # two differ on most steps, and reporting the next
                    # one would label every line with a step that has
                    # not happened yet.  Whether this is emitted at all
                    # is the log's decision, not this seam's.
                    dt=clock.step_ticks / clock.tick_den)
            except Exception:  # noqa: BLE001 - telemetry never fails a run
                pass

    def on_force(child_id, parent_id, child_clock, parent_clock) -> None:
        with domain_turn(("FORCE", child_id, parent_id)), allocation_scope(child_id):
            node = model.node(child_id)
            if node.parent is None or node.parent.cfg.grid_id != parent_id:
                raise RuntimeError(
                    "schedule FORCE edge differs from domain tree")
            status.pending_force = status.pending_force + 1
            status.prior_feedback_committed = False
            # NO store projection here, any more.  The coupler owns its own
            # reads: ``NestCoupler._coupled_parent_field`` pulls exactly the
            # two windows FORCE needs through the store seam
            # (``streaming.refresh_from_store`` with
            # ``nest.parent_footprint_window``), so the executor projecting
            # a window of EVERY carrier on the coupler's behalf was the same
            # bytes moved twice -- and a second copy of the footprint
            # geometry that a relocation could leave behind.  The window
            # arithmetic and its halo bound moved to ``gpuwm.core.nest``
            # with the reads.
            try:
                node.coupler.force(node)
                if validators and health_debug:
                    validators[child_id].require_healthy(
                        phase=(f"post-force.d{child_id:02d}"
                               f"-from-d{parent_id:02d}"))
            finally:
                status.pending_force = status.pending_force - 1
            poison()

    def on_feedback_prepare(child_id, parent_id, child_clock,
                            parent_clock) -> None:
        node = model.node(child_id)
        status.pending_feedback = status.pending_feedback + 1
        node.coupler.feedback_prepare(node, feedback_scratch[child_id])

    def on_feedback_commit(child_id, parent_id, child_clock,
                           parent_clock) -> None:
        node = model.node(child_id)
        # Two-way feedback is the one nest op that WRITES the parent, and a
        # streamed parent's writes have to land in its store or they are
        # thrown away at the next sweep.  The COUPLER owns that transaction:
        # ``NestCoupler.feedback_commit`` starts from the store and ends in
        # it (``_sync_in``/``_sync_out`` -> ``streaming.refresh_from_store``
        # / ``commit_to_store``, both of which drain the in-flight sweep
        # tail first), and ``feedback_finalize`` carries the whole-parent
        # diagnostics back the same way.  MEASURED bit-identical to the
        # all-resident run, with both feedback controls firing:
        # ``tilestream/test_nest.py`` leg 3 (coupler-driven) and
        # ``tilestream/test_nest_executor.py`` (this dispatch, through
        # execute_experiment).
        #
        # A refusal used to stand here, asserting "nothing projects a write
        # back".  It arrived in the same three-way merge (2358b3c06) as the
        # coupler arm it contradicted -- two lanes, one merged truth
        # missing -- and no gate drove this dispatch with a streamed parent,
        # so the contradiction sat unexercised.  The dispatch is now the
        # same for both modes; what an edge genuinely cannot do is refused
        # where the capability lives, by the coupler (cross-scheme edges
        # off a streamed parent in ``_coupled_parent_field``, an edge with
        # BOTH ends streamed in ``force``).
        status.mutation_in_progress = True
        try:
            with allocation_scope(child_id):
                node.coupler.feedback_commit(node)
        finally:
            status.mutation_in_progress = False

    def on_feedback_finalize(child_id, parent_id, child_clock,
                             parent_clock) -> None:
        node = model.node(child_id)
        status.mutation_in_progress = True
        try:
            with allocation_scope(child_id):
                node.coupler.feedback_finalize(node)
        finally:
            status.mutation_in_progress = False
            status.pending_feedback = status.pending_feedback - 1
        poison()

    def on_history(grid_id, ticks) -> None:
        if history_handler is not None:
            if validators:
                validators[grid_id].require_healthy(
                    phase=f"pre-history.d{grid_id:02d}")
            history_handler(model, model.node(grid_id), ticks)
        if io_manager is not None:
            status.pending_d2h = int(io_manager.pending)

    def on_restart(ticks) -> None:
        if io_manager is not None:
            io_manager.drain()
            status.pending_d2h = int(io_manager.pending)
        if restart_handler is not None:
            restart_handler(model, ticks)

    def on_period_end(period, clocks) -> None:
        status.schedule_cursor = PERIOD_BEGIN
        status.prior_feedback_committed = status.pending_feedback == 0
        root_clock = clocks[model.root.cfg.grid_id]
        mandatory = (
            root_clock.at_stop_time
            or any(clock.history_due() for clock in clocks.values())
            or (restart_enabled and root_clock.restart_due()))
        if validators and (health_debug or mandatory
                           or root_clock.step_count % 4 == 0):
            for gid, validator in validators.items():
                validator.require_healthy(
                    phase=f"post-d01-sync.d{gid:02d}")

    #: Wall clock of the outer step that is committing, measured the
    #: same way the single-domain loop measures its own
    #: (runtime.integrate_prepared_case).  It used to be published as a
    #: literal 0.0 from here, which meant the tree route -- the one that
    #: runs the deep nests, where the number matters most -- reported no
    #: cadence at all: every consumer of step_wall_seconds (the
    #: supervisor's stale-step threshold, any progress display) was
    #: reading a constant.  One perf_counter pair per outer step.
    period_wall = [time.perf_counter()]

    def on_period_commit(period, clocks) -> None:
        root_clock = clocks[model.root.cfg.grid_id]
        if pool_trim_per_period:
            _trim_default_pool()
        now = time.perf_counter()
        # WALL time, not model time.  The local is deliberately NOT named
        # step_wall_seconds even though the keyword it feeds is: the
        # clock audit (tests/test_clock.py) bans any assignment in this
        # module whose target name carries "elapsed"/"second", because
        # MODEL elapsed seconds must be tick-derived and never
        # float-accumulated.  A perf_counter difference is a different
        # quantity that happens to share the word, and the audit keeps
        # its full force here: an actual elapsed-seconds assignment
        # still trips it.
        step_wall = now - period_wall[0]
        period_wall[0] = now
        if progress_callback is not None:
            progress_callback(
                model_elapsed_seconds=root_clock.elapsed_seconds,
                outer_step=root_clock.step_count,
                last_durable_wrfout=(None if io_manager is None else
                                     io_manager.last_durable_wrfout),
                last_checkpoint=model._last_checkpoint,
                phase="post-d01-sync",
                step_wall_seconds=step_wall,
                # The whole tree's clocks, not only the root's.  This
                # dict was already in scope and deliberately unforwarded,
                # which left a nested run with no per-domain advancement
                # anywhere on the callback: a consumer that wanted it had
                # to parse child wrfout filenames, the re-derivation the
                # output hook exists to avoid.
                #
                # Additive and safe.  Every consumer of this callback
                # takes **kwargs -- the supervisor heartbeat, both
                # prepared runners, run-plan's observer, the verify
                # cases -- so an extra key is a TypeError for none of
                # them, and one that ignores it behaves exactly as
                # before.
                #
                # Read, never accumulated: each value is that clock's
                # own tick-derived figure, which is what the clock audit
                # in tests/test_clock.py is protecting.
                domain_clocks={
                    grid_id: float(clock.elapsed_seconds)
                    for grid_id, clock in clocks.items()})

    clocks = {node.cfg.grid_id: node.clock
              for node in model.walk_parent_first()}
    root_ticks = model.root.clock.ticks
    if root_ticks % model.schedule.period_ticks != 0:
        raise ValueError("model does not start at a PERIOD_BEGIN tick")
    start_period = root_ticks // model.schedule.period_ticks
    if progress_callback is not None:
        # Stepping begins here.  Without this transition the last published
        # phase is whatever preparation label the runner left behind
        # (initialize-domain-writers on the tree path), so a first-step
        # physics failure was stamped as a preparation failure and sent a
        # real diagnosis down the wrong path.  ``outer_step`` stays the
        # completed-step count run-progress.json already tracks; the phase
        # names the 1-based outer step being attempted, matching the
        # single-domain loop's ``outer-N.substep-M`` convention.
        root_clock = model.root.clock
        progress_callback(
            model_elapsed_seconds=root_clock.elapsed_seconds,
            outer_step=root_clock.step_count,
            last_durable_wrfout=(None if io_manager is None else
                                 io_manager.last_durable_wrfout),
            last_checkpoint=model._last_checkpoint,
            phase=f"stepping:outer-{root_clock.step_count + 1}",
            step_wall_seconds=0.0)
    # Start the cadence clock at the last instant before stepping, so
    # the first outer step's wall time is the step's and not the
    # transition heartbeat's.
    period_wall[0] = time.perf_counter()
    # ADAPTIVE TIME STEP.  Off, this is None and execute_schedule takes
    # its precomputed periodic walk exactly as before -- the whole feature
    # is one argument that stays None.  On, the driver owns every float:
    # clock.py's walk is AST-audited integer-only, so what reaches it is a
    # step_ticks and nothing else.
    adaptive_driver = None
    if bool(model.root.cfg.run.use_adaptive_time_step):
        from gpuwm.core.adaptive_clock import AdaptiveClockDriver
        from gpuwm.core.dycore import (enable_wrf_cfl_recording,
                                       take_wrf_cfl)
        # The controller reads the same reduction the probe does rather
        # than carrying a second copy of the kernel.
        enable_wrf_cfl_recording()
        adaptive_driver = AdaptiveClockDriver(
            model, cfl_source=take_wrf_cfl,
            tick_den=model.schedule.clock.tick_den,
            carrier_source=lambda gid: (
                None if _streamed(gid) is None
                else _streamed(gid).carrier_provenance()),
            map_factor_source=lambda gid: (
                None if _streamed(gid) is None
                else _streamed(gid).maximum_map_factor()))

    try:
        execution = execute_schedule(
            model.schedule, on_step=on_step, on_force=on_force,
            on_period_steps=adaptive_driver,
            on_feedback_prepare=on_feedback_prepare,
            on_feedback_commit=on_feedback_commit,
            on_feedback_finalize=on_feedback_finalize,
            on_history=on_history, on_restart=on_restart,
            on_domain_start=on_domain_start,
            on_period_begin=on_period_begin, on_period_end=on_period_end,
            on_period_commit=on_period_commit,
            skip_feedback_path=skip_feedback_path, clocks=clocks,
            start_period=start_period,
            started_grid_ids=(
                node.cfg.grid_id for node in model.walk_parent_first()
                if bool(node._started)),
            committed_initial_history_grid_ids=(
                model._resume_committed_history_grid_ids
                if bool(model._resumed) else ()))
    finally:
        if adaptive_driver is not None:
            # THE FOLD IS A MODULE GLOBAL, so it outlives this run.  A
            # second experiment in the same process -- a chained `gpuwm
            # go`, a test module -- would otherwise keep launching
            # w_cfl_stat every RK stage for a run that never asked for
            # it, hold 4.72 MB per domain, and read this run's row for a
            # grid_id it reuses.  In a finally so a refused run releases
            # it too.
            from gpuwm.core.dycore import reset_wrf_cfl_recording
            reset_wrf_cfl_recording()
    if relocation_runner is not None:
        relocation_runner.close_receipt(model)
    return execution


__all__ = [
    "PERIOD_BEGIN", "RestartPhase", "DomainNode", "FeedbackScratch",
    "NestCoupler", "ExperimentState", "ModelMemoryLedger",
    "ModelRuntimeStatus", "SharedRRTMGPChunkWorkspace", "build_experiment",
    "execute_experiment", "experiment_fingerprint",
]
