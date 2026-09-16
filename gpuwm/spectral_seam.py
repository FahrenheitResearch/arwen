"""The one seam between Arwen's slow-step loop and the Level-2 operators.

WHERE THE HOOK FIRES.  ``gpuwm.core.model.execute_experiment`` owns the
authoritative slow-large-step commit point: its STEP op calls the domain's
stepper (``dycore.step`` itself, or a ``StreamedDomain`` with the same
signature) and then refreshes the state clock from exact integer ticks
(``refresh_model_time(..., after_step=True)``).  That refresh is the commit;
the seam call sits immediately after it, inside the same domain turn, once
per domain per model time step -- after the RK slow-mode state is final,
never inside an acoustic substep, before health validation, output, nest
feedback or the next large step.  The 2026-08-17 survey of the current tree
(docs/handoffs/CURRENT-CORE-SPECTRAL-SEAM-SURVEY.json) records exactly this
anchor; the delivered survey the package promised was lost with its empty
patch, so the seam was re-derived from the live sources.

CONTRACTS HELD HERE (the operator package holds the arithmetic ones):

- absent/off pays one ``is None`` test in the loop and reads no state;
- a streamed domain refuses an active mode: its ``node.state`` is the t=0
  attach snapshot, the forecast lives in the pinned host store, so the
  complete target planes are NOT resident at the hook seam -- shadow would
  receipt the initial condition under later timestamps and apply would
  mutate planes the sweep never reads back.  The sentence lives in ONE
  place, :func:`streamed_spectral_refusal`.  ONE door reads it today, and
  it is a late one: :meth:`SpectralSeam.validate_domain` raises it at
  attach.  :func:`refuse_streamed_spectral_numerics` answers the same
  question from the configuration alone, out of that same sentence, but
  NO caller invokes it -- not in this module, not anywhere in the tree --
  so plan review carries no refusal for this combination and attach is
  the only door standing.  The three call sites that would change that
  are outside this lane's boundary and are handed back, with their exact
  lines, on the strict xfail in ``tests/test_spectral_seam.py``
  (``test_plan_review_carries_the_streamed_spectral_sentence``):
  ``_streaming_refusal`` in ``gpuwm/runplan.py``, the shared config-load
  door in ``gpuwm/experiment.py``, and ``gpuwm check`` in
  ``gpuwm/core/preflight.py``.  One residue stays genuinely late even
  after those land, and is not hoistable: a spawn-born nest joins after
  the plan is reviewed, and under ``[tiles] mode = "auto"`` its
  streamed-ness is the planner's run-time answer, so that domain is
  validated at its first committed step (:meth:`after_step`) with the
  same sentence;
- a periodic declaration must be true of the domain: ``periodic_domain``
  (and the ``periodic`` boundary) refuse on any domain with open,
  specified or nested lateral boundaries;
- receipts are counted per domain and bound into the run capsule, and a
  completed run whose ``apply`` receipts are missing refuses a clean
  completion capsule (:meth:`SpectralSeam.require_complete`).
"""

from __future__ import annotations

from typing import Any, Mapping

#: The capsule ``receipts`` key this seam owns.
CAPSULE_RECEIPT_KEY = "spectral_numerics"


def _domain_is_periodic(run_cfg) -> tuple[bool, str]:
    """Whether both horizontal axes of one domain actually wrap.

    Mirrors ``gpuwm.core.dycore._boundary_x/_boundary_y``: an axis is
    periodic exactly when it is neither open nor externally forced
    (specified or nested).  Returns ``(answer, reason)`` so a refusal can
    name what broke the wrap.
    """
    reasons = []
    if getattr(run_cfg, "open_x", False):
        reasons.append("open_x")
    if getattr(run_cfg, "open_y", False):
        reasons.append("open_y")
    if getattr(run_cfg, "specified", False):
        reasons.append("specified lateral boundaries")
    if getattr(run_cfg, "nested", False):
        reasons.append("nested forcing")
    return (not reasons, ", ".join(reasons))


def streamed_spectral_refusal(config, grid_id: int) -> str:
    """The one sentence a streamed domain under an active mode gets.

    Written once and read by both doors, so the configuration-only refusal
    and the attach-time backstop can never say different things about one
    configuration.  It names the concrete breakage (the planes the operator
    would touch are not the planes the forecast is in) and the two ways
    out (run the domain resident, or turn the operator off).
    """
    return (
        f"[spectral_numerics] mode = {getattr(config, 'mode', 'off')!r} on "
        f"streamed domain d{int(grid_id):02d} is refused: a "
        "streamed domain's node.state is the t=0 attach snapshot "
        "(the forecast lives in the pinned host store), so the "
        "complete target planes are not resident at the hook "
        "seam.  Shadow would write receipts about the initial "
        "condition under forecast timestamps; apply would mutate "
        "planes the tile sweep never reads back.  Run this "
        "domain resident, or set mode = \"off\".")


def _declared_streamed_grid_ids(exp) -> tuple[int, ...]:
    """Grids the configuration ALONE says will stream, in tree order.

    ``mode = "on"`` is a declaration: that domain streams, and the answer
    is legible in the TOML.  ``mode = "auto"`` is a question the planner
    answers against the machine at run time, so a domain on auto is NOT
    listed here: refusing it from the configuration would refuse a tree
    the planner would have run resident, and a road nobody has priced yet
    is not the same thing as a road that cannot exist.

    ORDER, because a refusal names one domain out of the set.  This is
    the experiment's own domain order, which the loader has already put
    parent before child (``gpuwm/experiment.py`` ``_parent_before_child``,
    the reordering it warns about rather than refusing), so the id this
    answers first is an ancestor of, or a peer declared ahead of, every
    other id it lists -- the direction
    :meth:`gpuwm.core.model.Model.walk_parent_first` takes at attach.
    Sorting by grid id instead would name the lowest id, which on a tree
    whose ids do not ascend in declaration order is a different domain
    from the one the attach door names for that same configuration.
    """
    from gpuwm.core.streaming import options_for_domain

    tree = getattr(exp, "tiles", None)
    return tuple(
        int(dc.grid_id) for dc in (getattr(exp, "domains", ()) or ())
        if options_for_domain(dc, tree).mode == "on")


def refuse_streamed_spectral_numerics(exp, streamed_grid_ids=None) -> None:
    """Refuse [spectral_numerics] x a streamed domain from the config alone.

    NO CALLER INVOKES THIS YET.  Both halves of the combination are
    legible in the experiment TOML, so the question belongs at plan
    review (config load, dry run, ``gpuwm check``) rather than at attach,
    and answering it from the configuration alone is what lets it be
    asked there.  The three doors that would ask it live in files outside
    this lane's boundary; they are named, with their lines, on the strict
    xfail ``test_plan_review_carries_the_streamed_spectral_sentence`` in
    ``tests/test_spectral_seam.py``.  Until one of them calls this, the
    attach-time raise in :meth:`SpectralSeam.validate_domain` is the only
    door this combination meets, and plan review says nothing about it.

    Pass ``streamed_grid_ids`` when the caller already knows which grids
    stream, parent first; otherwise the declared ids are read off the
    resolved ``[tiles]`` tables by :func:`_declared_streamed_grid_ids`,
    which answers in that order.  The refusal names the first of them.

    Raises the same ``RuntimeError`` text as
    :meth:`SpectralSeam.validate_domain`, because they are one refusal seen
    from two doors.  ``mode = "off"`` and an absent block pass untouched.
    """
    config = getattr(exp, "spectral_numerics", None)
    if config is None or getattr(config, "mode", "off") == "off":
        return
    if streamed_grid_ids is None:
        streamed_grid_ids = _declared_streamed_grid_ids(exp)
    streamed = [int(grid_id) for grid_id in streamed_grid_ids]
    if streamed:
        raise RuntimeError(streamed_spectral_refusal(config, streamed[0]))


class SpectralSeam:
    """Per-run hook cache, receipt ledger and capsule binding.

    One instance per model run, attached to the ``ExperimentState`` (so
    spawn/restart leg walks that call ``execute_experiment`` repeatedly
    keep one ledger).  Domain hooks are built lazily on first commit --
    a spawn-born nest gets its hook at its first step -- and validated
    eagerly for every node present at attach.
    """

    def __init__(self, config, experiment_name: str):
        config.validate()
        self.config = config
        self.experiment_name = str(experiment_name)
        self._hooks: dict[int, Any] = {}
        self._steps: dict[int, int] = {}
        self._receipts: dict[int, int] = {}
        self._receipt_hash_chain: dict[int, str] = {}

    # -- wiring-time validation -------------------------------------------

    def validate_domain(self, grid_id: int, run_cfg, *,
                        streamed: bool) -> None:
        if streamed:
            # The one door this combination meets today, and a late one.
            # :func:`refuse_streamed_spectral_numerics` would answer the
            # same question from the configuration, before the run, and
            # it holds no second copy of this sentence -- but nothing
            # calls it yet, so a declared-streamed tree still finds out
            # here, at attach.  Read from the one place that holds it.
            raise RuntimeError(
                streamed_spectral_refusal(self.config, grid_id))
        needs_wrap = (self.config.periodic_domain
                      or self.config.boundary == "periodic")
        if needs_wrap:
            periodic, reason = _domain_is_periodic(run_cfg)
            if not periodic:
                raise RuntimeError(
                    "[spectral_numerics] declares periodic_domain = true "
                    f"but domain d{int(grid_id):02d} does not wrap "
                    f"({reason}).  A periodic transform of a non-periodic "
                    "domain couples its opposite lateral boundaries "
                    "through the FFT; declare the domain accurately "
                    "(boundary = \"tapered\") or drop the periodic "
                    "declaration.")

    # -- the per-step seam call -------------------------------------------

    def hook_for(self, grid_id: int, run_cfg):
        grid_id = int(grid_id)
        hook = self._hooks.get(grid_id)
        if hook is None:
            from gpuwm.spectral_ops import SpectralLargeStepHook
            hook = SpectralLargeStepHook(
                config=self.config, dx_m=float(run_cfg.dx),
                dy_m=float(run_cfg.dy), dt_s=float(run_cfg.dt),
                domain=f"d{grid_id:02d}")
            self._hooks[grid_id] = hook
        return hook

    def after_step(self, state, grid_id: int, run_cfg, *, step_count: int,
                   model_seconds: float, streamed: bool = False):
        """Called once per domain immediately after its slow-step commit."""
        grid_id = int(grid_id)
        if grid_id not in self._hooks:
            # A late-joining domain (spawn birth) validates at its first
            # committed step, which is the first instant it could be wrong.
            self.validate_domain(grid_id, run_cfg, streamed=streamed)
        hook = self.hook_for(grid_id, run_cfg)
        self._steps[grid_id] = self._steps.get(grid_id, 0) + 1
        receipt = hook(state, large_step=int(step_count), source={
            "experiment": self.experiment_name,
            "grid_id": grid_id,
            "model_seconds": float(model_seconds),
        })
        if receipt is not None:
            self._receipts[grid_id] = self._receipts.get(grid_id, 0) + 1
            previous = self._receipt_hash_chain.get(grid_id, "")
            from gpuwm.spectral_ops.pins import canonical_hash
            self._receipt_hash_chain[grid_id] = canonical_hash(
                [previous, receipt["receipt_sha256"]])
        return receipt

    # -- completion -------------------------------------------------------

    def expected_receipts(self, grid_id: int) -> int:
        return self._steps.get(int(grid_id), 0) // int(
            self.config.cadence_steps)

    @property
    def complete(self) -> bool:
        return all(self._receipts.get(gid, 0) == self.expected_receipts(gid)
                   for gid in self._steps)

    def require_complete(self) -> None:
        """Refuse a clean completion capsule over missing apply receipts."""
        if self.config.mode != "apply" or self.complete:
            return
        shortfall = {
            f"d{gid:02d}": {"expected": self.expected_receipts(gid),
                            "written": self._receipts.get(gid, 0)}
            for gid in sorted(self._steps)
            if self._receipts.get(gid, 0) != self.expected_receipts(gid)}
        raise RuntimeError(
            "the run applied spectral numerics but its step receipts are "
            f"incomplete ({shortfall}); a clean completion capsule cannot "
            "be emitted, because an applied correction without its "
            "receipt is a state change with no audit trail.")

    def capsule_record(self) -> dict[str, object]:
        """The ``receipts`` section entry the run capsule binds."""
        from gpuwm.spectral_ops.pins import PINS_SHA256, SCHEMA
        return {
            "schema": "gpuwm.spectral-numerics-run-receipts/v1",
            "operator_schema": SCHEMA,
            "operator_pins_sha256": PINS_SHA256,
            "config_sha256": self.config.config_sha256,
            "mode": self.config.mode,
            "cadence_steps": int(self.config.cadence_steps),
            "receipt_directory": self.config.receipt_directory,
            "complete": self.complete,
            "domains": {
                f"d{gid:02d}": {
                    "steps": self._steps.get(gid, 0),
                    "expected_receipts": self.expected_receipts(gid),
                    "receipts": self._receipts.get(gid, 0),
                    "receipt_hash_chain_sha256":
                        self._receipt_hash_chain.get(gid),
                }
                for gid in sorted(self._steps)
            },
        }


def attach_seam(model, exp, steppers: Mapping[int, Any] | None):
    """Build (or re-use) the run's seam and validate the present tree.

    Returns ``None`` when the experiment carries no active
    [spectral_numerics] -- the OFF contract costs the loop one test.
    Idempotent across the spawn/restart leg walks: the first call stores
    the seam on the model, later calls re-validate the (possibly grown)
    tree against it and hand the same ledger back.
    """
    config = getattr(exp, "spectral_numerics", None) if exp is not None \
        else None
    existing = getattr(model, "_spectral_seam", None)
    if config is None or getattr(config, "mode", "off") == "off":
        return existing
    seam = existing
    if seam is None:
        seam = SpectralSeam(config, getattr(exp, "name", "experiment"))
        model._spectral_seam = seam
    from gpuwm.core.streaming import is_streaming
    steppers = dict(steppers or {})
    for node in model.walk_parent_first():
        grid_id = int(node.cfg.grid_id)
        seam.validate_domain(
            grid_id, node.cfg.run,
            streamed=is_streaming(steppers.get(grid_id)))
    return seam


def seam_capsule_receipts(model) -> dict[str, object]:
    """The capsule ``receipts`` mapping for one finished model, or empty.

    Calls :meth:`SpectralSeam.require_complete` first, so a route cannot
    bind a receipts section that claims an applied run it cannot prove.
    """
    seam = getattr(model, "_spectral_seam", None)
    if seam is None:
        return {}
    seam.require_complete()
    return {CAPSULE_RECEIPT_KEY: seam.capsule_record()}


__all__ = ["CAPSULE_RECEIPT_KEY", "SpectralSeam", "attach_seam",
           "refuse_streamed_spectral_numerics", "seam_capsule_receipts",
           "streamed_spectral_refusal"]
