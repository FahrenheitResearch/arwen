"""Out-of-core streaming as a MODE OF ARWEN, not a parallel universe.

:mod:`tilestream` can integrate a domain that does not fit on the card: the
full domain lives in pinned host RAM and tiles of it are cycled through the
GPU, bit-exact against a resident run.  Every one of those results was
produced by ``tilestream``'s own stepping loop.  ArWen's real driver --
:mod:`gpuwm.core.model` for a domain tree, :func:`gpuwm.runtime
.integrate_prepared_case` for a single domain -- knew nothing about any of
it, which meant a user could not turn it on.  This module is the seam that
makes it a configuration rather than a research branch.

THE SEAM IS ONE CALLABLE
------------------------
Both run loops step a domain by calling ``dycore.step(state, cfg, **kw)``.
:func:`make_stepper` returns the callable they call instead.  With streaming
OFF it returns ``gpuwm.core.dycore.step`` ITSELF -- the same function object,
not a wrapper that forwards to it -- so there is no "streaming disabled" code
path to be wrong: the identity ``make_stepper(...) is dycore.step`` is what
the OFF contract means, and it is asserted rather than described
(``tests/test_streaming.py::test_off_returns_the_dycore_step_itself``).

With streaming ON it returns a :class:`StreamedDomain`, whose ``__call__``
has ``dycore.step``'s signature and advances the whole domain by exactly one
model step per call, sweeping the tiles.  The MODEL still owns the loop: it
decides when output, restart, diagnostics, health validation and nest
coupling happen, on its own cadences, with its own handlers.  That is why
:class:`tilestream.driver.TiledRun` exists -- ``run_tiled`` owned a time loop
of its own, and a mode that owns the time loop can only ever sit beside the
model, never inside it.

WHAT "AUTO" DECIDES, AND WHY IT IS NOT "ALWAYS ON"
-------------------------------------------------
Streaming is never free.  MEASURED tiling tax against the identical resident
run, dry, 4090, 1024^2 x 49, 150 steps: tile 128 -> 1.359x, tile 256 ->
1.217x, tile 512 -> 1.346x, and the compute window has to be at least ~500
cells on a side before the number means anything at all (below that the
measurement is of an idle GPU).  So ``mode = "auto"`` streams ONLY when the
domain does not fit resident, and the fitting question is answered by
:mod:`tilestream.autoplan`, whose VRAM model was fitted against 29 measured
allocations on two cards and is a tight upper bound (measured 1.060x /
1.095x of the real allocation).  ``mode = "on"`` streams a domain that would
have fitted, which is what a benchmark or a bit-exactness proof needs and
what a forecast does not.

THREE THINGS A STREAMED DOMAIN MUST CARRY, EACH OF WHICH HAS BEEN A BUG
-----------------------------------------------------------------------
*Geography is INPUT.*  Map factors, Coriolis, rotation, terrain, the
terrain-following base state and every scheme's latitude/longitude grid are
gathered into a buffer when that buffer starts serving a different tile, and
never scattered back: measured bit-identical across 8 steps at the
229-carrier rung.  Rebuilding them per tile from the tile's own config
displaces a tile by up to 1022 km and 20.6% in Coriolis.

*The clock is a carrier.*  ``dycore.step`` advances ``elapsed_seconds`` once
per CALL and ``PhysicsDriver`` turns it into the ``itimestep`` that every
cadence test reads, and ``lateral_bc`` turns it into ``dtbc``.  A buffer
serving k tiles would advance k*dt while the domain advanced dt, so tiles
inside one sweep would disagree about whether radiation, cumulus or the PBL
was due and would integrate different physics -- with no NaN and no warning.
Dropping it changed 874,076 cells; at the 229-carrier rung it moves 153 of
them.  So the buffer's clock is reset to the domain's before every tile step
and the domain's advances exactly once per sweep.

*Lateral boundaries are per tile.*  A tile's TRUE domain edges take the
domain's own tables sliced along the tangential axis; its interior seams take
inert tables.  MEASURED, and this is the fact the mode rests on: seam tables
of zeros, seam tables holding the domain's own coupled values with zero
tendency, and seam tables of deliberate garbage (1e6 in coupled units, 1e4/s
of tendency) all give the BIT-IDENTICAL answer out to 24 steps at the halo
``harness.halo_radius`` prescribes.  The seam relaxation cannot reach the
interior, because it perturbs the RK TENDENCY rather than the state and a
tendency injected at stage 0 is advected by fewer stages than a state
perturbation present at the start of the step -- its cone is strictly inside
the dycore's own.

WHERE A HISTORY FRAME COMES FROM, AND WHY THE ANSWER MOVED
-----------------------------------------------------------
``attach`` COPIES the domain's carriers into the store and the sweep
advances the store; the resident ``DomainState`` it was built from is never
written again.  ArWen's three history call sites -- ``runtime
.write_case_output``, ``runtime._submit_tree_history_frame`` and
``prepared_single_domain_forecast``'s ``history_handler`` -- all read that
state, so a streamed run published a full forecast's worth of frames holding
the INITIAL CONDITION, each with the right inventory, the right ``Times``
and the right global attributes.  Nothing raised.

So :class:`StreamedDomain` marks the state it took over
(``state._streamed_domain``), :meth:`StreamedDomain.history_fields` serves
the frame off the store through :class:`tilestream.output.StoreFrame`, and
``AsyncDomainWrfoutWriter.submit`` grew a ``frame=`` parameter so a frame
that is already on the host is published with no device traffic at all.
MEASURED, ``tilestream.test_history``: five frames -- the cold-start frame
plus four history frames -- per-variable AND whole-file SHA-256 identical to
the resident run at every rung, REFL_10CM included.

REFL_10CM is the one field that needed more than transport, and it needed
two things: the sweep was DROPPING ``step_kwargs`` (so no tile ever ran
``calc_refl10cm``), and the model's one-frame handoff refuses to be
overwritten, which fires on the second tile a buffer serves.  See
:func:`refl_handoff_hook`.

THE HALO IS NEVER MEASURED
--------------------------
It is ``10 + 3*time_step_sound//2`` (:func:`tilestream.harness.halo_radius`)
and nothing else.  A too-small halo is silent AND faster, which is how it
hides; the smallest halo that happens to pass grows with step count, shrinks
with test-domain size and differs by GPU.

That last clause is not a hedge, it is a measurement, and the two halves of
it were taken on the SAME MACHINE.  On the joined configuration at
256x192x49, tile 32x32, N=8, the join lane found halo 15 passing and halo 14
failing on one 4090; ``tilestream/test_join.py`` finds halo 14 passing and
halo 13 failing on the other 4090 of the same dual-GPU box.  Same code, same
config, same card model, margin off by one.  Anything that took its halo
from a sweep would have shipped a number that is wrong on the neighbouring
socket -- which is why the gate's halo CONTROL is half the dependency
radius, a value the dependency argument says cannot work, and why the
margin is printed as data and never asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


#: The three legal values of ``[tiles] mode``.
STREAMING_MODES = ("off", "on", "auto")

#: Attribute :func:`attach` leaves on the prepared state: ``{slot: array}``
#: over the store's ``scratch/*`` members.  See :func:`live_scratch` for what
#: it is for; ``gpuwm/io/restart.py:STATE_INFRA_ATTRS`` classifies it.
STREAMED_SCRATCH_ATTR = "_streamed_scratch"

StreamingMode = Literal["off", "on", "auto"]

#: Every key ``[tiles]`` accepts.  Unknown keys are refused rather than
#: ignored, on the ``[relocation]`` / ``[perturbation]`` precedent: a
#: misspelled knob that silently does nothing is how a run gets configured
#: for a mode it is not in.
STREAMING_KEYS = frozenset({
    "mode", "tile_nx", "tile_ny", "nbuffers", "halo", "store", "write_mode",
    "pipeline", "vram_budget_bytes", "host_budget_bytes",
})


class StreamingRefused(RuntimeError):
    """A configuration that asks for ``[tiles]`` and cannot legally have it."""


class _Unset:
    """Distinguishes "not supplied" from "supplied as None".

    ``attach(scalars=None)`` has to mean CARRY NOTHING, because that is the
    gate's clock control and a default that helpfully computed the scalars
    instead would disarm it -- silently, and in the direction that passes.
    """

    def __repr__(self) -> str:                    # pragma: no cover
        return "<unset>"


UNSET = _Unset()


@dataclass(frozen=True)
class StreamingOptions:
    """The user-facing surface: ``[tiles]`` in an experiment TOML.

    ``mode``
        ``"off"`` (the default, and the whole-tree default state),
        ``"on"``, or ``"auto"``.  OFF is not a code path -- see the module
        docstring.

    ``tile_nx`` / ``tile_ny`` / ``nbuffers``
        The tiling, when the caller wants to pin it.  ``None`` asks
        :mod:`tilestream.autoplan` for it.  NEVER benchmark below a ~500-cell
        compute window: at 224^2 the tiling tax is 5.37x and what is being
        measured is an idle GPU, not the transport.

    ``halo``
        Present so that a deliberately broken halo can be configured for a
        negative control, and for NO other reason.  ``None`` -- the only
        value a forecast may use -- takes it from
        :func:`tilestream.harness.halo_radius`.  A halo below that radius is
        silently wrong and faster, so this key warns when it is set.

    ``store``
        ``"host"`` (pinned host RAM; the out-of-core mode) or ``"device"``
        (the domain store stays in VRAM and only the STEPPING is tiled).
        The device store exists because it isolates the tiling arithmetic
        from the transport: if a run is bit-exact with a device store and
        wrong with a host store, the defect is in the copies, not the tiles.

    ``write_mode``
        ``"ring"`` keeps ONE domain store plus a small arena of halo rings;
        ``"shadow"`` keeps a whole second store.  Ring is the default because
        the host store is the binding constraint at every capacity limit
        measured: at 32.26 B/cell dry against a 44.14 GiB pinned ceiling a
        single store holds 5476^2 x 49 and a shadow pair only 3872^2.  Rings
        are cheap only when the tile DIVIDES the domain -- a ragged trailing
        tile is read right through, 22.7% of the store instead of 2.4% in one
        measured case -- which is why the planner prefers exact tilings.

    ``vram_budget_bytes`` / ``host_budget_bytes``
        Override what :func:`tilestream.autoplan.Machine.detect` found.
        ``/proc/meminfo`` reports the HOST's RAM inside a container and is
        not a budget.  ``host_budget_bytes`` is read BEFORE the machine is
        probed and supplies the host figure to the probe, so it stands in
        for a host source that cannot be read at all rather than merely
        correcting one that can -- see :func:`decide`.
    """

    mode: StreamingMode = "off"
    tile_nx: int | None = None
    tile_ny: int | None = None
    nbuffers: int | None = None
    halo: int | None = None
    store: str = "host"
    write_mode: str = "ring"
    pipeline: str = "prefetch"
    vram_budget_bytes: int | None = None
    host_budget_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in STREAMING_MODES:
            raise ValueError(
                f"[tiles] mode = {self.mode!r} is not one of "
                f"{list(STREAMING_MODES)}")
        if self.store not in ("host", "device"):
            raise ValueError(
                f"[tiles] store = {self.store!r} must be 'host' (the "
                "out-of-core mode) or 'device'")
        if self.write_mode not in ("ring", "shadow"):
            raise ValueError(
                f"[tiles] write_mode = {self.write_mode!r} must be "
                "'ring' or 'shadow'.  'inplace' is tilestream's deliberate "
                "read-at-time-t bug and is not configurable from here.")
        if self.pipeline not in ("prefetch", "naive"):
            raise ValueError(
                f"[tiles] pipeline = {self.pipeline!r} must be "
                "'prefetch' or 'naive'")
        if (self.tile_nx is None) != (self.tile_ny is None):
            raise ValueError(
                "[tiles] tile_nx and tile_ny must be given together; "
                "half a tiling is not a tiling")
        for name in ("tile_nx", "tile_ny", "nbuffers", "halo"):
            value = getattr(self, name)
            if value is not None and int(value) <= 0:
                raise ValueError(
                    f"[tiles] {name} = {value!r} must be positive")
        if self.mode == "off" and any(
                getattr(self, k) is not None
                for k in ("tile_nx", "tile_ny", "nbuffers", "halo")):
            # The [relocation] discipline: a surface that is off must be
            # empty, so nothing can start streaming because a block was
            # inherited and a mode flipped somewhere else.
            raise ValueError(
                "[tiles] carries a tiling while mode is 'off' or "
                "absent.  A [tiles] surface that is off must be empty; "
                "set mode = 'auto' or 'on' deliberately, or delete the "
                "key(s).")

    @property
    def enabled(self) -> bool:
        """``mode`` is not ``"off"``.  Says nothing about whether it fires."""
        return self.mode != "off"

    @classmethod
    def from_mapping(cls, table: Any, *, source: str = "<config>"
                     ) -> "StreamingOptions":
        """Validate one ``[tiles]`` table.  ``None`` gives :data:`OFF`."""
        if table is None:
            return OFF
        if not isinstance(table, dict):
            raise ValueError(
                f"[tiles] of {source} must be a table, got {table!r}")
        unknown = sorted(set(table) - STREAMING_KEYS)
        if unknown:
            raise ValueError(
                f"unknown key(s) {unknown} in [tiles] of {source}; "
                f"known keys: {sorted(STREAMING_KEYS)}")
        return cls(**table)

    def to_json(self) -> dict[str, object]:
        """The receipt form.  OFF serializes as ``None`` -- see below."""
        return {name: getattr(self, name)
                for name in ("mode", "tile_nx", "tile_ny", "nbuffers",
                             "halo", "store", "write_mode", "pipeline")
                if getattr(self, name) is not None}


#: The OFF contract, as one shared object.  An experiment that never mentions
#: [tiles] carries THIS, and :func:`identity_payload_entry` returns
#: nothing for it, so every fingerprint written before streaming existed is
#: byte-identical afterwards.
OFF = StreamingOptions()


def identity_payload_entry(options: "StreamingOptions | None") -> dict:
    """What ``[tiles]`` contributes to the restart identity: NOTHING.

    Deliberate, and the opposite of ``[perturbation]``, which binds.  A
    perturbation changes the forecast; streaming is a promise that it does
    not.  The entire claim of this module is that a domain integrated as one
    resident block and the same domain streamed from host RAM produce the
    same bytes -- proven carrier by carrier at every physics rung, and
    proven again across a checkpoint in four legs (streamed -> file ->
    streamed, streamed -> file -> MONOLITHIC, monolithic -> file ->
    streamed, all bit-exact).  So a checkpoint written by a resident run MUST
    resume streamed and a checkpoint written streamed MUST resume resident;
    binding the mode into the identity would refuse exactly the operation
    that makes the mode worth having -- a forecast that outgrew its card
    resuming on the machine it outgrew.
    """
    return {}


@dataclass(frozen=True)
class StreamingDecision:
    """What ``auto`` decided, and the numbers it decided on.

    Carried into the run and printed by :meth:`explain`, because a run that
    silently chose a different integration mode from the one the operator
    expected is a run whose timings mean nothing.
    """

    stream: bool
    reason: str
    tile_nx: int | None = None
    tile_ny: int | None = None
    nbuffers: int | None = None
    halo: int | None = None
    store: str = "host"
    #: ``"ring"`` (one store plus a small arena of saved halo rings) or
    #: ``"shadow"`` (a whole second store).  Carried on the DECISION and not
    #: read from the options at attach time, because ``decide`` is what the
    #: capacity model was run against: ``autoplan.plan`` is already given
    #: this value and sizes the host budget with it, so a run that then
    #: attached with a different one would be planned for one footprint and
    #: executed with another.  It used to be exactly that -- ``attach``
    #: hardcoded ``"ring"``, so ``[tiles] write_mode = "shadow"`` was
    #: planned for (2x the host store) and silently ignored.
    write_mode: str = "ring"
    resident_bytes: int | None = None
    budget_bytes: int | None = None
    detail: dict = field(default_factory=dict)

    def explain(self) -> str:
        """The decision in the words of the table that configured it.

        ``[tiles]``, not "streaming".  The product has a second door with
        that name -- ``gpuwm stream``, the observation stream -- and this
        sentence is interpolated straight into ``make_stepper``'s refusal,
        so a misconfigured ``[tiles]`` run used to be refused in the
        vocabulary of a feature it has nothing to do with.  The naming
        ruling that renamed the table covers the text a user of the table
        reads.
        """
        if not self.stream:
            return f"[tiles] OFF: {self.reason}"
        return (f"[tiles] ON ({self.store} store): {self.reason}; "
                f"tile {self.tile_nx}x{self.tile_ny}, "
                f"nbuffers={self.nbuffers}, halo={self.halo}")


def _halo_for(cfg) -> int:
    from tilestream.harness import halo_radius

    return int(halo_radius(cfg))


def decide(cfg, options: StreamingOptions | None = None, *, machine=None
           ) -> StreamingDecision:
    """Resolve ``[tiles]`` against this domain and this machine.

    ``"off"`` short-circuits before importing anything: a tree that never
    configures streaming never touches the planner, cupy or ``tilestream``.
    """
    options = OFF if options is None else options
    if options.mode == "off":
        return StreamingDecision(False, "[tiles] mode = 'off'")

    halo = options.halo if options.halo is not None else _halo_for(cfg)
    need = _halo_for(cfg)
    if int(halo) < need:
        import warnings

        warnings.warn(
            f"[tiles] halo = {halo} is below the per-step dependency "
            f"radius {need} for time_step_sound={cfg.time_step_sound}.  "
            "Tile interiors will be silently wrong, and the run will be "
            "FASTER, which is how this defect hides.  The only value a "
            "forecast may configure is none at all.",
            RuntimeWarning, stacklevel=2)

    if options.tile_nx is not None:
        # A pinned tiling asks no question, so it consults no planner and
        # needs no card: the configuration IS the decision.  This is the
        # path a bit-exactness proof runs on, where the tiling is chosen to
        # make the halo and the seams do work rather than to be fast.
        return StreamingDecision(
            True, "[tiles] pins the tiling",
            int(options.tile_nx), int(options.tile_ny),
            int(options.nbuffers or 2), int(halo), options.store,
            options.write_mode)

    import dataclasses

    from tilestream import autoplan

    # The configured budget is read BEFORE the machine is probed, and handed
    # to the probe, because ``detect`` skips the host-memory read entirely
    # when it is told what the host budget is.  Probing first and overriding
    # afterwards made the override unreachable exactly when it was needed:
    # ``detect`` RAISES where it can find no host source, so on Windows --
    # no procfs, no cgroups -- every ``[tiles]`` configuration was refused
    # before this function ever looked at ``host_budget_bytes``, including
    # the ones that set it to the very number the refusal asked for.  On a
    # box where detection works this changes nothing: ``host_bytes`` is
    # replaced with the same value either way and ``pinned_fraction`` is
    # still forced to 1.0 below, so the resulting budget is identical.
    if machine is None:
        machine = autoplan.Machine.detect(
            host_bytes=(None if options.host_budget_bytes is None
                        else int(options.host_budget_bytes)))
    # Both overrides name a BUDGET, so they must land on the budget, and
    # ``Machine`` reaches its budgets through two multipliers:
    # ``vram_budget_bytes = vram_bytes * (1 - vram_headroom)`` and
    # ``host_budget_bytes = host_bytes * pinned_fraction``.  Replacing only
    # the raw capacity leaves the multiplier applied on top, so the
    # configured number is not the budget.
    #
    # For VRAM that was not an approximation, it was zero.  ``vram_headroom
    # = 1.0`` makes the budget ``vram_bytes * (1 - 1.0)``, so every
    # planner-driven configuration that set the key was refused with
    # "no tile fits in 0.00 GiB of VRAM" no matter how large a number it
    # asked for -- MEASURED at 3.2, 3.4, ... 24 GiB, all zero, all refused.
    # It went unnoticed because until the routes had builders there was no
    # configuration that could reach the planner and then run.
    #
    # For host RAM it was silent instead: ``host_bytes = N`` with
    # ``pinned_fraction`` still 0.47 turned a requested 60 GiB into a
    # 28.2 GiB budget, which looks like a plan that simply chose smaller
    # tiles.
    if options.vram_budget_bytes is not None:
        machine = dataclasses.replace(
            machine, vram_bytes=int(options.vram_budget_bytes),
            vram_headroom=0.0)
    if options.host_budget_bytes is not None:
        machine = dataclasses.replace(
            machine, host_bytes=int(options.host_budget_bytes),
            pinned_fraction=1.0, host_source="explicit")

    plan = autoplan.plan(cfg, machine,
                         prefer_resident=(options.mode == "auto"),
                         write_mode=options.write_mode)
    if plan.mode == "resident":
        return StreamingDecision(
            False,
            "the domain fits resident on this card, and the tiling tax is "
            "1.2x-1.4x even at the best tile size",
            resident_bytes=int(plan.vram_bytes),
            budget_bytes=int(plan.vram_budget_bytes),
            # A resident domain pins NO host store; recorded as zero rather
            # than absent so the tree walk's host ledger never has to guess.
            detail={"plan": plan.explain(), "host_claim_bytes": 0})
    return StreamingDecision(
        True,
        ("the resident domain does not fit on this card"
         if options.mode == "auto" else "[tiles] mode = 'on'"),
        int(plan.tile_nx), int(plan.tile_ny), int(plan.nbuffers), int(halo),
        options.store, options.write_mode,
        resident_bytes=int(plan.vram_bytes),
        budget_bytes=int(plan.vram_budget_bytes),
        # THE OTHER BUDGET.  A streamed domain's pinned host store plus its
        # arena is the binding constraint at every capacity limit measured,
        # and it is a per-domain claim on ONE box -- so the tree walk has to
        # subtract it for the next domain exactly as it subtracts VRAM.  It
        # is carried here, from the planner's own arithmetic, rather than
        # re-derived by the walk from a second model of the same thing.
        detail={"plan": plan.explain(),
                "host_claim_bytes": int(plan.host_bytes)})


@dataclass(frozen=True)
class StreamedEnvelope:
    """What a STREAMED forecast of one domain actually holds, priced.

    The number the admission gate needs and did not have.  Every memory
    term ``gpuwm.core.preflight`` computes itemizes a domain resident in
    VRAM, so with ``[tiles]`` configured the gate was answering a question
    nobody asked: MEASURED at the 2.2.0 cut, a 550x550x49 run with 200x200
    tiles -- the exact shape streaming exists for -- was refused by
    ``gpuwm go`` at "15.36 GiB peak envelope exceeds the 13.09 GiB budget"
    on a card with 15.24 GiB free, and the remedy printed was
    ``--no-memory-gate``.  A feature whose only configuration is refused by
    default is not shipped, so the gate learns the streamed envelope here.

    Built from :class:`tilestream.autoplan.Footprint` -- the same measured
    model ``autoplan.plan`` sizes tiles with and the same one ``decide``
    consults -- so the gate and the run cannot give two answers about one
    card.  ``vram_bytes`` is the whole streamed process: CUDA context, the
    rung's per-process fixed cost, and ``nbuffers`` tile buffers of the
    COMPUTE WINDOW (tile plus halo on both axes), with autoplan's safety
    factor already applied.  ``host_bytes`` is the pinned store plus the
    ring arena, which is where the domain actually lives.
    """

    vram_bytes: int
    store_bytes: int
    arena_bytes: int
    host_bytes: int
    host_budget_bytes: int | None
    tile_nx: int
    tile_ny: int
    nbuffers: int
    halo: int
    window_nx: int
    window_ny: int
    rung: str
    write_mode: str = "ring"

    def summary(self) -> str:
        """One sentence, in the vocabulary of the table that configured it."""
        gib = 1024 ** 3
        host = (f"{self.host_bytes / gib:.2f} GiB of pinned host RAM"
                if self.host_budget_bytes is None else
                f"{self.host_bytes / gib:.2f} GiB of pinned host RAM against "
                f"a {self.host_budget_bytes / gib:.2f} GiB budget")
        return (f"[tiles] streams this domain, so the card holds "
                f"{self.nbuffers} tile buffer(s) of "
                f"{self.window_nx}x{self.window_ny} (tile "
                f"{self.tile_nx}x{self.tile_ny} + halo {self.halo}) at the "
                f"{self.rung} rung = {self.vram_bytes / gib:.2f} GiB, not the "
                f"whole domain; the forecast itself lives in {host}")


def streamed_envelope(cfg, options: "StreamingOptions | None" = None, *,
                      machine=None, decision: StreamingDecision | None = None
                      ) -> StreamedEnvelope | None:
    """Price ``cfg`` as the streamed run ``options`` would actually attach.

    Returns ``None`` when this configuration does NOT stream -- ``[tiles]``
    off, or ``mode = "auto"`` deciding the domain fits resident -- which is
    the caller's signal to keep using the resident estimate.  That is the
    whole contract: an admission gate calls this, and if it gets a number
    it prices the run against THAT number instead.

    ``decision`` is accepted so a caller that has already resolved
    ``[tiles]`` (the run itself, a report that prints the decision) prices
    the decision it will execute rather than re-deriving one that could
    differ.  Without it, :func:`decide` resolves the options -- and a
    ``mode = "auto"`` config with no pinned tiling consults the planner,
    which needs a ``machine``; pass one built from an out-of-process probe
    rather than letting ``Machine.detect`` stand a CUDA context up inside a
    long-lived CLI (the reason ``go_cli.memory_gate`` probes in a
    subprocess at all).
    """
    from tilestream import autoplan

    if decision is None:
        decision = decide(cfg, options, machine=machine)
    if not decision.stream:
        return None

    fp = autoplan.footprint_for(cfg)
    nx, ny, nz = int(cfg.nx), int(cfg.ny), int(cfg.nz)
    halo = int(decision.halo if decision.halo is not None else _halo_for(cfg))
    tile_nx = int(decision.tile_nx or nx)
    tile_ny = int(decision.tile_ny or ny)
    nbuffers = int(decision.nbuffers or 2)
    # autoplan's own window arithmetic (_best_tile), not an approximation of
    # it: the compute window is the tile plus a halo on BOTH sides of BOTH
    # axes, and it is the window -- never the tile -- that a buffer holds.
    window_nx = tile_nx + 2 * halo
    window_ny = tile_ny + 2 * halo
    vram = fp.vram_bytes(window_nx * window_ny * nz, nbuffers)
    store = fp.store_bytes(nx * ny * nz)
    if decision.write_mode == "ring":
        arena = store * autoplan.ring_arena_fraction(
            nx, ny, tile_nx, tile_ny, halo) * 1.05
    else:
        arena = store
    host_total = _host_total_bytes()
    return StreamedEnvelope(
        vram_bytes=int(vram), store_bytes=int(store), arena_bytes=int(arena),
        host_bytes=int(store + arena),
        host_budget_bytes=(None if host_total is None
                           else int(host_total * autoplan.PINNED_FRACTION)),
        tile_nx=tile_nx, tile_ny=tile_ny, nbuffers=nbuffers, halo=halo,
        window_nx=window_nx, window_ny=window_ny, rung=fp.rung,
        write_mode=decision.write_mode)


def _host_total_bytes() -> int | None:
    """This box's RAM, or ``None`` rather than a guess.

    The same two sources :meth:`autoplan.Machine.detect` reads, and
    deliberately the SAME functions: this used to carry its own private
    ``GlobalMemoryStatusEx`` probe because ``autoplan`` knew only
    ``/proc/meminfo``, so the PRICING path could see the Windows box's RAM
    while the PLANNING path could not and refused.  One probe, in the lower
    layer, is what keeps those two answers from disagreeing again.

    Unlike ``detect`` this still answers ``None`` rather than raising when
    nothing can be read: its caller is
    :meth:`StreamedEnvelope.summary`, which then simply omits the budget
    line, and a report that cannot price host RAM should not be fatal.
    """
    from tilestream.autoplan import _cgroup_memory_limit, _host_memtotal

    limit = _cgroup_memory_limit()
    total = _host_memtotal()
    if limit is not None and total is not None:
        return min(limit, total)
    return limit if total is None else total


class StreamedDomain:
    """One domain integrated tile-by-tile, one sweep per ``__call__``.

    Constructed by :func:`make_stepper` and used exactly where the run loops
    used ``dycore.step``::

        stepper = make_stepper(state, cfg, options, ...)
        for outer_step in range(outer_steps):
            stepper(state, cfg, refl_10cm_due=due)     # <- one model step
            ... history, restart, diagnostics, nest coupling, unchanged ...

    ``state`` is passed on every call and CHECKED rather than used: the model
    holds the domain state object, and a caller that quietly swapped it
    (a relocation, a restart that rebuilt it) would otherwise stream a domain
    that no longer exists.  The domain's own arrays live in ``self.store``.

    The tile buffers, their streams, the ring arena and each buffer's
    geography and boundary occupancy live on the wrapped
    :class:`tilestream.driver.TiledRun` and survive every step.  Rebuilding
    them per step would cost 3.3 s each for a 544-cell dry buffer AND would
    re-gather the read-only geography every step -- see ``TiledRun``.
    """

    def __init__(self, run, decision: StreamingDecision, *, state=None,
                 scalars=None, host_store=None, stability=None,
                 geography=None, boundaries=None, template=None,
                 inventory_fn=None):
        self._run = run
        self.decision = decision
        self._state = state
        self.scalars = scalars
        self.host_store = host_store
        #: THE RUN'S OWN INVENTORY RULE -- the one ``attach`` built the store
        #: with and handed ``TiledRun`` for the buffers.  Held so that
        #: :meth:`refresh_state` harvests the state by the same rule the
        #: store was filled by.  Harvesting with the plain
        #: ``carrier_inventory`` instead left the one carrier only the rule
        #: adds -- ``scratch/refl_10cm``, REBUILT scratch, excluded from the
        #: plain manifest by construction -- uncopied: MEASURED, the store
        #: held a 69 dBZ field while the refreshed state's slot stayed
        #: all-zero, and the resident road's final_state_digest attested the
        #: zeros.
        self.inventory_fn = inventory_fn
        #: The folded stability record, or ``None`` when nothing asked for
        #: one.  :func:`stability_observer` returns this to the run loop in
        #: place of ``dycore.stability_report``, which under a host store
        #: would reduce over a t=0 snapshot the sweep never writes.
        self.stability = stability
        #: The DOMAIN geography inventory and the DOMAIN's lateral
        #: boundaries.  Held for one reason: a restart HEADER is fingerprinted
        #: over the domain's setup arrays and its LBC tables, and above the
        #: card's ceiling there is no resident state to take them from.  See
        #: :meth:`restart_setup`.
        self._geography = geography
        self._boundaries = boundaries
        #: The SLAB-HEIGHT state a store-direct domain was built through, or
        #: ``None``.  It is not a spare copy of the domain and must never be
        #: read as one: its carriers are the last row slab's and its
        #: horizontally-varying setup is that slab's window.  What it is for
        #: is the two questions whose answers do not vary with height -- which
        #: schemes publish which frame fields (:meth:`history_fields`), and
        #: what the vertical coordinate is (:meth:`restart_setup`) -- and
        #: every use of it here is one of those.
        self._template = template
        self._setup = None
        self._statics_setup = None
        self.steps = 0
        self.report: dict = {}
        self._frame = None
        if state is not None:
            # The MARKER, and it is the reason a streamed forecast stopped
            # writing t = 0 into every history frame.  ArWen's three history
            # call sites read the resident DomainState, which after attach()
            # holds a perfect copy of the INITIAL condition forever -- right
            # shapes, right inventory, right Times, wrong forecast, and no
            # error anywhere.  Marking the state lets the writers ask "is
            # this domain's truth still here?" instead of assuming it.
            try:
                state._streamed_domain = self
            except AttributeError:
                # Reachable only for an object with no instance dict, which
                # ``DomainState`` is not and no state in this model is; the
                # seam's own CPU-only tests attach to bare ``object()``
                # stand-ins.  Nothing publishes a frame from one of those,
                # so an unmarked stand-in cannot hide the defect above.
                pass

    @property
    def tiled_run(self):
        """The :class:`tilestream.driver.TiledRun` doing the sweeping."""
        return self._run

    @property
    def store(self) -> dict:
        """``{name: array}`` of the whole domain, wherever it lives."""
        return self._run.store

    @property
    def state(self):
        """The prepared state this domain was attached to.

        Its CARRIERS are stale from the first sweep on -- they were copied
        into the store and the object is never stepped again -- but its
        GEOGRAPHY is not, because geography is input and the transport never
        writes it back.  That is the only thing anything should read off it,
        and :mod:`tilestream.receipts` is what reads it.
        """
        return self._state

    @property
    def template(self):
        """The slab-height state a store-direct domain classifies against.

        ``None`` for a domain attached to a resident state, which needs no
        stand-in.  See :attr:`_template` for what may and may not be read off
        it.
        """
        return self._template

    def __call__(self, state, cfg, **step_kwargs) -> None:
        """``dycore.step``'s signature.  One model step of the DOMAIN.

        ``refl_10cm_due=True`` is the one kwarg that means more here than it
        does resident, and both extra things it means are in
        :func:`refl_handoff_hook`'s docstring.  The visible half is this: a
        due step leaves the domain's REFL_10CM in the store, and it is also
        STASHED on the resident ``state`` on the way out, so that
        ``gpuwm.core.refl.consume_refl_10cm(state)`` -- which is what all
        three of ArWen's history call sites call, unchanged -- returns the
        domain field a streamed run computed instead of raising "no
        microphysics-time field is stashed".
        """
        if self._state is not None and state is not self._state:
            raise StreamingRefused(
                "the state object handed to the streamed stepper is not the "
                "one it was attached to.  A streamed domain's arrays live in "
                "its store, not on the state, so stepping a different state "
                "would integrate the wrong domain -- and silently, because "
                "both are DomainStates of the same shape.  Re-attach with "
                "make_stepper after anything that replaces the state.")
        if cfg is not self._run.cfg and (
                int(cfg.nx), int(cfg.ny), int(cfg.nz)) != (
                int(self._run.cfg.nx), int(self._run.cfg.ny),
                int(self._run.cfg.nz)):
            raise StreamingRefused(
                f"the streamed domain was attached at "
                f"{self._run.cfg.nx}x{self._run.cfg.ny}x{self._run.cfg.nz} "
                f"and is being stepped at {cfg.nx}x{cfg.ny}x{cfg.nz}")
        due = bool(step_kwargs.get("refl_10cm_due", False))
        if due and REFL_STORE_KEY not in self._run.store:
            raise StreamingRefused(
                f"refl_10cm_due=True but the store carries no "
                f"{REFL_STORE_KEY!r}, so each tile would compute its own "
                "window of REFL_10CM and the transport would throw all of "
                "them away.  A streamed domain that publishes reflectivity "
                "lists that key in its inventory_fn (it is a REBUILT "
                "scratch slot, not a carrier, so the trajectory is "
                "bit-identical either way) and primes the slot on the "
                "domain state and on every tile buffer before attach, "
                "exactly as Kain-Fritsch's lazily-allocated cumulus/w0avg "
                "has to be primed.  Refused rather than publishing a field "
                "that would be the last tile's over 3 of 4 tiles.  MEASURED "
                "before any of this worked, 96x72x49 tile 24x24 through "
                "execute_experiment: the resident run staged REFL_10CM for "
                "1 of its 2 frames and the streamed run for 0 of 2, with no "
                "error and no NaN -- a missing output field and nothing "
                "else.")
        self.report = {}
        if self.stability is not None:
            self.stability.begin_sweep()
        self._run.sweep(1, step_kwargs=step_kwargs, report=self.report)
        self.steps += 1
        if due:
            self._stash_domain_refl(state)

    def history_fields(self) -> dict:
        """One wrfout history frame, assembled off the store, host arrays.

        Built by :class:`tilestream.output.StoreFrame`, whose classification
        of every frame field is MEASURED against the real
        ``wrfout._device_state_frame`` rather than transcribed, and whose
        field order is the device frame's own -- load-bearing, because the
        frame dict doubles as the writer's schema and HDF5 lays its name
        heap out in variable-creation order (the same numbers in a different
        order give a file 189 bytes larger in which every variable compares
        equal).

        Lazy, because it allocates: the derived ``T`` and ``P`` need their
        own destinations, 2 x 4 B/cell, and a run that never writes a frame
        should not pay for them.  Everything else is a ZERO-COPY VIEW of the
        pinned store -- which is what makes an out-of-core frame 2.2x-2.4x
        cheaper than the resident path -- so the returned dict is only valid
        UNTIL THE NEXT SWEEP.  Take it at a sweep boundary and drain the
        writer before stepping on; that is the whole discipline and
        ``tilestream.test_history.negative_async_no_drain`` asserts what
        breaking it does.

        Raises rather than guesses if the store cannot serve the frame: a
        run whose ``inventory_fn`` carries only the trajectory's carriers
        cannot publish the output-only driver diagnostics (``OLR``, and
        ``XKMH``/``XKHH`` under ``hmix_k_diag``), and a frame missing rows a
        resident run writes is not the same frame.

        THAT RAISE IS REAL AS OF 2.2.0 AND WAS NOT BEFORE.  Until then this
        docstring described an intention: ``StoreFrame.fields`` skipped the
        unavailable rows and the writer took the short dict as its schema,
        so the sentence above was true of the design and false of the
        behaviour -- a streamed run published a frame without ``OLR`` and
        reported validity PASS.  :class:`tilestream.output.ShortFrameRefused`
        is the refusal, :func:`diagnostic_inventory` is the reason it no
        longer fires on the product route, and
        ``tests/test_streamed_frame_parity.py`` is what keeps the two
        honest by comparing the streamed and resident FIELD SETS rather
        than trusting either description.

        TWO ROADS TO THE PLAN, ONE FRAME.  With a prepared resident state the
        plan is measured against that state directly, which is what every
        streamed run has always done.  A STORE-DIRECT domain has no such
        object -- and above the card's ceiling cannot have one -- so the plan
        is measured against the slab-height :attr:`template` and re-sized onto
        the store by :func:`tilestream.output.store_frame_plan`; the setup
        statics come from the DOMAIN geography rather than from the slab's
        window of it (:meth:`output_setup`).  Neither substitution is a
        different frame: the classification is a question about schemes and
        the statics are the domain's own arrays, and the gate is that a domain
        small enough to run both ways publishes byte-identical files.
        """
        if self._frame is None:
            from tilestream import output as _output

            if self._state is not None:
                plan = _output.frame_plan(
                    self._state, extra_available=self._run.store.keys())
            elif self._template is not None:
                plan = _output.store_frame_plan(
                    self._template, self._run.store, self._run.cfg,
                    extra_available=self._run.store.keys())
            else:
                raise StreamingRefused(
                    "this streamed domain was attached with neither a "
                    "resident state nor a template state, so there is "
                    "nothing to measure the frame plan against.  Which "
                    "fields a wrfout frame carries and where each of them "
                    "comes from is read off a LIVE state by "
                    "tilestream.output.frame_plan -- deliberately, so that a "
                    "scheme publishing a new diagnostic cannot drift a "
                    "transcribed table -- and a store on its own says only "
                    "which carriers exist, not which of them the writer "
                    "wants under which name.  Pass attach(template=...) the "
                    "slab the store was built through.")
            self._frame = _output.StoreFrame(plan, self._run.store,
                                             self.output_setup(),
                                             self._run.cfg)
        # The frame's non-derived rows are zero-copy VIEWS of the pinned
        # store, so under the deferred sweep seam the previous step's
        # scatter tail must land before they are read.  Idempotent and a
        # flag test when nothing is pending.
        drain = getattr(self._run, "drain", None)
        if drain is not None:
            drain()
        return self._frame.fields()

    def _stash_domain_refl(self, state) -> None:
        """Hand the assembled domain REFL_10CM to the resident consumer.

        The array is the STORE's -- pinned host memory with ``overlap=False``
        semantics, i.e. the next sweep scatters into it -- and the caller
        that consumes it is the history writer, which runs at a sweep
        boundary and drains before stepping on.  That is the same discipline
        every other field of an out-of-core frame is under; see
        ``AsyncDomainWrfoutWriter.submit``'s ``frame`` parameter.

        A stash that is already occupied is left alone and re-raised through
        the model's own guard rather than silently replaced: on a streamed
        domain the only thing that can have stashed is a previous due step
        whose frame was never written, which is exactly the cadence bug
        ``stash_refl_10cm`` exists to catch.
        """
        import numpy as np

        from gpuwm.core.refl import stash_refl_10cm

        if getattr(state, "physics", None) is None:
            return
        stash_refl_10cm(state, np.asarray(self._run.store[REFL_STORE_KEY]))

    @property
    def elapsed_seconds(self) -> float:
        """The DOMAIN clock, which the sweep advances once per step."""
        return float((self.scalars or {}).get("elapsed_seconds", 0.0))

    def refresh_state(self, state=None) -> int:
        """Copy the streamed domain back onto the ``DomainState``, and why.

        With ``store = "host"`` -- the out-of-core mode, and the only one
        worth having -- the forecast lives in pinned host RAM and the
        ``DomainState`` the route is holding is the SNAPSHOT that filled the
        store.  Nothing writes back to it.  MEASURED, 192x144x49 at the
        155-carrier rung through ``execute_experiment``: after 120 streamed
        steps, 0 of 155 carriers on ``node.state`` had moved, against 118 of
        155 for the identical resident run.

        That matters because ``node.state`` is what everything AROUND the
        stepper reads.  ``execute_experiment`` hands it to the health
        validators; ``prepared_single_domain_forecast`` reads it for every
        history frame, for the stability gate and for the final
        ``canonical_state_digest``.  Left alone, a streamed forecast writes
        the INITIAL CONDITION into every wrfout it publishes and passes
        every health check by inspecting a state that never integrated --
        which is worse than the refusal this mode used to give, because it
        looks like a forecast.

        So a route that reads the state must call this first.  It is a
        D2H/H2D of the whole carrier set, so it belongs on the HISTORY
        cadence and at the end of the run, never per step: at 229 B/cell it
        is one domain-sized copy, which is what one history frame already
        costs.  Returns the number of members copied -- the carrier set
        plus the scattered ``diag/*`` producers below -- so a caller can
        assert it moved the whole manifest rather than a lucky subset.

        The ``diag/*`` members ride along (task #219, second finding).
        Output-only driver diagnostics -- ``OLR`` at every shipped rung,
        the eddy-viscosity pair under ``cfg.hmix_k_diag``, the SASE flux
        rows under SASE -- are correctly absent from the carrier manifest
        (nothing reads them back into the trajectory), but the sweep
        gathers and scatters them into the store precisely so they can be
        published.  A refresh that moved only carriers left the DRIVER's
        copies at the snapshot's zeros, so every route that publishes
        frames off the state (``gpuwm.io.wrfout.state_frame``; the offline
        child) emitted OLR = 0 for the whole forecast while the store held
        the sky the tiles computed -- MEASURED: streamed OLR max 0.0
        against ~292 W/m2 resident, with the production StoreFrame route
        publishing the same store correctly the whole time.
        """
        import cupy as cp

        if not self.host_store:
            # ``store = "device"`` makes the store the state's OWN arrays
            # (attach does not copy), so the state is never stale and a
            # refresh would be a self-copy of the whole manifest.
            return 0
        target = self._state if state is None else state
        if target is None:
            raise StreamingRefused(
                "refresh_state() needs the DomainState to copy into; this "
                "streamed domain was attached without one")
        from tilestream import output as _output
        from tilestream import physics_inventory as _physics

        # THE RUN'S RULE, not the plain carrier manifest.  The store was
        # filled by ``inventory_fn(state, None)`` at attach, so a refresh
        # harvesting the state with anything narrower copies back a strict
        # subset and leaves the remainder AT WHATEVER THE STATE LAST HELD.
        # For ``scratch/refl_10cm`` -- present only under the run's rule --
        # that remainder was the primed zeros, and the final health gate and
        # ``canonical_state_digest`` then attested a zeroed reflectivity
        # field for a run whose store held the real one.  The fallback keeps
        # a hand-constructed StreamedDomain (the CPU tests') refreshing the
        # way it always did.
        take = self.inventory_fn or _physics.carrier_inventory
        live = take(target, None)
        store = self._run.store
        missing = sorted(set(live) - set(store))
        if missing:
            raise StreamingRefused(
                f"the store is missing {len(missing)} carrier(s) the state "
                f"holds ({missing[:6]}); refusing a partial refresh, which "
                "would leave the state half at t=0 and half at t=n")
        # The scattered output-only diagnostics, addressed by the SAME
        # naming their gather used (tilestream.output.diagnostic_members;
        # its inverse is what the frame plan reads).  Keys are taken from
        # the STORE -- what the sweep actually carried -- and a carried
        # member with no destination on this state is refused for the same
        # reason a missing carrier is: the silent version publishes zeros
        # for a field the sweep demonstrably computed.
        diag = {name: array for name, array in
                _output.diagnostic_members(target).items()
                if name in store}
        missing_diag = sorted(
            name for name in store
            if name.startswith("diag/") and name not in diag)
        if missing_diag:
            raise StreamingRefused(
                f"the store carries {len(missing_diag)} scattered driver "
                f"diagnostic(s) ({missing_diag[:6]}) with no destination "
                "on this state; refusing a refresh that would publish "
                "zeros for fields the sweep computed")
        for name, dst in {**live, **diag}.items():
            src = store[name]
            if tuple(dst.shape) != tuple(src.shape):
                raise StreamingRefused(
                    f"member {name!r} is {tuple(dst.shape)} on the state "
                    f"and {tuple(src.shape)} in the store")
            dst[...] = cp.asarray(src, dtype=dst.dtype)
        if self.scalars is not None:
            _physics.set_carrier_scalars(target, self.scalars)
        cp.cuda.Stream.null.synchronize()
        return len(live) + len(diag)

    def carrier_provenance(self) -> dict | None:
        """The surface-radiation carrier ledger AS OF THE LAST SWEEP.

        The third observer of the corpse, after the history frame itself and
        the stability fold, and the last one to be found.  ``gpuwm.io.wrfout
        .carrier_provenance_attrs`` reads ``state.physics.carriers``, and
        under a host store that driver is the snapshot the store was filled
        from: it never ran radiation, so its contract still reads
        ``unwritten`` / ``-1.0`` for every carrier while the STORE holds the
        sky those producers demonstrably computed.  MEASURED at the 2.2.0
        cut, 438x350x49 t+1h: ``GLW`` and ``SWDOWN`` byte-identical to the
        resident run (means 411.14745 and 46.99931), and the four
        ``GPUWM_CARRIER_*`` attributes on the same file denying that
        anything wrote them.  Provenance metadata that contradicts the data
        beside it is worse than no metadata, because the whole point of the
        contract (gpuwm/core/radiation_carriers.py) is that a reader can ask
        a wrfout "did this file integrate a sky nobody computed" without the
        run directory.

        The ledger itself was never lost: it rides ``carrier_scalars`` on
        the domain clock, ``_advance_clock`` republishes it into
        :attr:`scalars` after every sweep, and :meth:`refresh_state` already
        restores it onto a state.  Only the EXPORT path was reading the
        stale object, so this is a read of the live one and not a new
        mechanism.  Returns ``None`` when this domain carries no contract
        (no scalars, or a pre-contract streamed checkpoint), which the
        caller must treat as "ask the state" rather than as "unwritten".
        """
        scalars = self.scalars or {}
        rows = scalars.get("carriers")
        if rows is None:
            return None
        return {
            "policy": scalars.get("surface_radiation_policy"),
            "records": {str(name): dict(row) for name, row in rows.items()},
        }

    def domain_mass_measure(self) -> float:
        """The FP64 dry-mass receipt, taken from the STORE.

        Here, and not left to the caller, because the obvious call is the
        wrong one and it fails silently: ``dycore.domain_mass_measure(state)``
        on a streamed domain reads the state the store was FILLED FROM, which
        is never stepped again, so it returns the t=0 mass for the whole
        forecast -- MEASURED 376832 ulp from the truth after 24 steps at
        256x192x49, and looking for all the world like perfect conservation.

        :mod:`tilestream.receipts` takes it from the store in the MONOLITHIC
        traversal order rather than folding it per tile, which is what makes
        it bit-identical to the resident run's rather than 1 ulp away; the
        measurement behind that choice is in that module's docstring.
        """
        from tilestream import receipts

        return receipts.domain_mass_measure(self)

    @property
    def health(self) -> dict:
        """The last step's stability report, folded out of the STORE.

        Same keys as :func:`gpuwm.core.dycore.stability_report`, and bit-equal
        to what that function would return for the same domain integrated
        resident -- the fold is over maxima, which are exact and associative.
        Raises rather than returning a stale report if the sweep did not
        produce one, because the failure this whole seam exists to prevent is
        a safety gate that silently reports the wrong memory.
        """
        report = self.report.get("health")
        if report is None and self.stability is not None:
            # The other arm: ``attach(health_fold=False)`` (the default) arms
            # StreamedStability through TiledRun's observer hook instead of
            # TileHealthFold through ``health_width``.  Same guarantee, same
            # keys, folded over the same interiors.
            report = self.stability()
        if report is None:
            raise StreamingRefused(
                "this streamed domain produced no health fold, so the run "
                "loop's nan / w_max / CFL gate has nothing to observe.  It "
                "must NOT fall back to reading the resident DomainState: "
                "under a host store the sweep never writes it, so that gate "
                "passes forever and a blown-up forecast checkpoints clean.")
        return report

    # ----------------------------------------------------------------- restart
    #
    # A forecast that cannot be checkpointed cannot be operated, and the
    # resident writer cannot be pointed at a streamed domain: ``gpuwm.io
    # .restart.write_restart`` walks ``vars(state)``, ``state._scratch`` and
    # ``state.physics`` and ``.get()``s every array off the device.  A
    # streamed domain's arrays are NOT on that state -- ``attach`` copied
    # them into the store and the state has been frozen at its preparation
    # values ever since.  Handing it to the resident writer therefore does
    # not fail: it writes a complete, self-consistent, fully validating
    # checkpoint of the INITIAL CONDITION, stamped with the current clock.
    # Every shape check passes, every fingerprint matches, and the file
    # resumes into a forecast that silently threw away everything since
    # t=0.  That is why these two methods exist and why the run loops ask
    # the stepper where the domain is instead of assuming.
    @property
    def template_state(self):
        """One tile buffer.  The restart header's shape/dtype template."""
        return self._run.tiles[0]

    def restart_setup(self):
        """The :class:`tilestream.restart_stream.DomainSetup` for the header.

        Two routes, and which one applies is a property of how big the
        domain is rather than a preference.  With a prepared resident state
        still around (every domain that fits on the card, which is every
        domain a bit-exactness proof can use) it is captured directly from
        that state.  Above the card's ceiling there is no such object, and
        the setup is reassembled from the DOMAIN geography store plus a tile
        buffer for the purely vertical arrays -- which is checked, not
        assumed: a wrong reassembly moves ``setup_fingerprint`` and the file
        is refused at RESTORE rather than resumed onto a different
        projection.
        """
        from tilestream import restart_stream

        if self._setup is None:
            if self._state is not None:
                self._setup = restart_stream.capture_domain_setup(self._state)
            elif self._geography is not None:
                self._setup = restart_stream.domain_setup_from_stream(
                    self._geography, self.template_state,
                    lateral_boundaries=self._boundaries)
            else:
                raise StreamingRefused(
                    "this streamed domain was attached with neither a "
                    "prepared state nor a geography inventory, so the "
                    "restart header's setup fingerprint -- map factors, "
                    "Coriolis, terrain, the base state and the LBC tables -- "
                    "cannot be reconstructed.  A tile buffer's own setup is "
                    "the setup of a domain centred on that TILE and would "
                    "produce a checkpoint no run can ever resume.")
        return self._setup

    def output_setup(self):
        """Where a history FRAME's setup statics come from.

        A :class:`tilestream.checkpoint.DomainSetup`.  ``StoreFrame`` reads
        exactly four things off one:
        ``output_statics(nz, ny, nx)`` -- which is ``MUB``, ``HGT``, ``ZNU``,
        ``ZNW``, ``P_TOP`` and the ``PB``/``PHB`` column broadcasts -- plus
        ``thb``, ``pb`` and ``phb`` for the ``T`` and ``P`` derives.  ``MUB``
        and ``HGT`` are horizontal PLANES, and the base state joins them
        whenever ``terrain_opt`` makes it 3-D; the rest are columns and one
        scalar.  That is the whole reason this cannot be taken off a slab: a
        slab's ``mub2d`` and ``ht`` are its own rows of the domain's, so a
        frame built from them would publish one band of the terrain over the
        entire map -- with every column of that band correct, which is what
        makes it hard to see.

        With a resident state it is captured from that state, unchanged.
        Without one it is assembled from the same two sources
        :meth:`restart_setup` assembles the restart header's from -- the
        DOMAIN geography store for everything horizontal, a tile buffer for
        the purely vertical arrays -- and that assembly is CHECKED rather than
        assumed: ``restart_stream.assert_setup_identical`` fingerprints it
        against a monolithic state's, and a wrong reassembly moves
        ``setup_fingerprint`` and is refused at restore.  Reusing it here
        means a frame and a checkpoint of the same instant cannot disagree
        about the terrain they were taken over.

        No physics identity is attached (``physics_setup=None``): it is
        resolved scheme fingerprints for a restart HEADER, an output frame
        reads none of it, and deriving it would need the domain's radiation
        lat/lon grid for nothing.  A checkpoint takes :meth:`restart_setup`,
        which does carry it.
        """
        from tilestream import checkpoint as _checkpoint

        if self._statics_setup is None:
            if self._state is not None:
                self._statics_setup = _checkpoint.DomainSetup.capture(
                    self._state, self._run.cfg)
            else:
                stream_setup = self.restart_setup()
                # The two DomainSetup classes name the same three
                # passthroughs differently (``restart_stream.DomainSetup`` is
                # a dataclass, ``checkpoint.DomainSetup`` duck-types a
                # DomainState), so the translation is spelled here rather
                # than assumed; ``checkpoint._SETUP_PASSTHROUGH`` is the list
                # being satisfied and a name it gains later resolves to
                # ``None``, which is what a domain without one has.
                self._statics_setup = _checkpoint.DomainSetup(
                    self._run.cfg, stream_setup.arrays, stream_setup.scalars,
                    None,
                    {"lateral_boundaries": stream_setup.lateral_boundaries,
                     "_lateral_boundary_device":
                         stream_setup.lateral_boundary_device,
                     "_nest_restart_classification":
                         stream_setup.nest_classification})
        return self._statics_setup

    def write_restart(self, path, cfg, *, run_trackers=None):
        """Write a gpuwm restart file from the store.  No device state.

        Byte-for-byte a ``gpuwm.io.restart`` v5 archive -- same header keys
        in the same order, same member names, same atomic ``.tmp`` publish
        and ``fsync`` -- so ``gpuwm.io.restart.restore_restart`` reads it
        into an ordinary resident run.  That symmetry is the promise
        :func:`identity_payload_entry` makes by contributing nothing to the
        restart identity: a checkpoint written streamed MUST resume
        resident and one written resident MUST resume streamed.
        """
        from tilestream import restart_stream

        return restart_stream.write_streamed_restart(
            path, self.store, cfg, scalars=self.scalars,
            setup=self.restart_setup(), template_state=self.template_state,
            run_trackers=run_trackers,
            # A device store was never page-locked and never needed to be;
            # the pinned check exists to catch a HOST store that was built
            # with plain numpy and would have streamed at a fraction of the
            # bandwidth.
            check_pinned=bool(self.host_store))

    def restore_restart(self, path, cfg):
        """Restore a restart file INTO the store, in place, and reseed the clock.

        The clock reseed is not bookkeeping.  ``TiledRun`` caches the domain
        clock in the sweep's closure -- it must, because ``dycore.step``
        advances ``elapsed_seconds`` once per CALL while the domain advances
        it once per SWEEP -- so a restore that only updated the caller's
        scalars dict would leave the store holding the checkpoint's
        atmosphere and the sweep holding the pre-restore ``itimestep``.  The
        cadence tests every scheme reads are functions of that itimestep, so
        radiation, cumulus and the PBL would fire on the wrong steps and
        ``dtbc`` would interpolate the wrong point of the forcing interval:
        a different forecast, with no NaN and no warning.
        """
        from tilestream import restart_stream

        info = restart_stream.read_streamed_restart(
            path, self.store, cfg, setup=self.restart_setup(),
            template_state=self.template_state, scalars=self.scalars)
        if self.scalars is not None:
            self._run.reseed_clock(self.scalars)
        return info

    def canonical_digest(self, clock, *, scope: str = "trajectory") -> dict:
        """``canonical_state_digest`` of this domain, taken from the STORE.

        The end-of-run evidence every route records, and the third whole-domain
        reader -- after the history frame and the checkpoint -- that cannot be
        pointed at a ``DomainState`` a streamed run does not step.  Pointed at
        one anyway it does not fail: ``state_digest.canonical_state_digest``
        walks the state's manifest, finds every member allocated and finite,
        and returns a complete, correctly-framed digest OF THE ANALYSIS,
        stamped with the forecast's final clock.  That is the same failure
        ``write_restart`` exists to prevent one artefact over, and it is worse
        here, because a digest is exactly the thing a reader trusts to tell
        two trajectories apart.

        ``gpuwm.state_digest.canonical_store_digest`` is the same document
        assembled off ``self.store`` and ``self.scalars``, member for member
        and byte for byte; see that module's docstring for the three things
        that make the equality hold rather than merely be intended.  A domain
        that carries no scalars refuses rather than digesting with a zero
        clock -- ``attach(scalars=None)`` is the gate's CARRY NOTHING control
        and its final digest would otherwise claim model second zero for a run
        that integrated.
        """
        from gpuwm.state_digest import canonical_store_digest

        if self.scalars is None:
            raise StreamingRefused(
                "this streamed domain carries no scalars, so its canonical "
                "digest would be stamped with an elapsed time and physics "
                "call counts of nothing -- attach(scalars=None) is the "
                "gate's CARRY NOTHING control and must not be digested as a "
                "forecast")
        return canonical_store_digest(self.store, self.scalars, clock,
                                      scope=scope)

    def impose_clock(self, seconds: float) -> None:
        """Set the DOMAIN clock the next sweep imposes on every tile.

        ``gpuwm.core.clock`` is the calendar authority for a resident domain:
        ``execute_experiment`` calls ``refresh_model_time`` before and after
        every STEP, so ``dycore.step``'s own ``elapsed_seconds += dt`` is
        overwritten from integer ticks and cannot drift.  A streamed domain
        never reads that state -- its tiles take the clock from
        :attr:`scalars` -- so without this it runs a SECOND, free-running
        clock, and every consumer of ``elapsed_seconds`` (the itimestep every
        physics cadence tests, and ``dtbc``) is evaluated against it.

        Two clocks that both advance by ``dt`` agree for as long as they
        start equal, which is why this is invisible until they do not: a
        state prepared with a warmup step, a resume at a nonzero time, a
        domain that joins the tree late.  MEASURED on the moving-nest lane
        with a one-step warmup before ``attach``: the streamed domain ran
        one step ahead for the whole run, radiation and cumulus fired on
        different steps from the resident reference, and 119 of 158 carriers
        differed -- with no NaN and no refusal, because a domain one step
        out of phase is a perfectly well-formed domain.  The same
        configuration stepped WITHOUT the model, both sides on their own
        clock, is bit-exact over all 158.
        """
        if self.scalars is None:
            raise StreamingRefused(
                "this streamed domain carries no scalars, so it has no clock "
                "to impose -- attach(scalars=None) is the gate's CARRY "
                "NOTHING control and must not be driven by the model loop")
        self.scalars["elapsed_seconds"] = float(seconds)

    # -- the store, projected onto the state --------------------------------
    #
    # Everything in ArWen that is not the stepper reads ``node.state``: the
    # nest coupler interpolates the child's boundaries out of the parent's
    # arrays, the storm tracker reduces the parent's UH plane, the
    # relocation rebuild takes a full SINT of the live parent, the health
    # validator scans it.  A streamed domain's arrays are in the STORE, and
    # with ``store="host"`` the state's device arrays are frozen at the
    # values they held when ``attach`` copied them out -- so every one of
    # those consumers silently reads t=0.  With ``store="device"`` the store
    # IS the state's arrays and there is nothing to do; these methods are
    # then exact no-ops, which is why the device store is the right control
    # for telling a transport bug from a tiling bug.
    #
    # The projection is deliberately NOT automatic and NOT whole-domain by
    # default.  A domain is streamed because it does not fit on the card;
    # copying all of it back every step would be the allocation the mode
    # exists to avoid, paid twice.  So the caller names WHAT it is about to
    # read (``names``) and OVER WHAT GROUND (``window``), and pays only that:
    # the storm tracker needs two (ny, nx) planes, 8 B/cell, once per
    # relocation cadence; the nest coupler needs every carrier but only over
    # the child's footprint plus its stencil, which is what makes a moving
    # nest on a streamed parent affordable at all.

    def _state_arrays(self, names=None) -> dict:
        from tilestream.physics_inventory import carrier_inventory

        if self._state is None:
            raise StreamingRefused(
                "this streamed domain was attached without a state, so "
                "there is nothing to project the store onto")
        live = carrier_inventory(self._state)
        if names is None:
            return live
        return {k: live[k] for k in names if k in live}

    @staticmethod
    def _window_slices(shape, window) -> tuple:
        """One slice rule for every consumer; see :func:`window_slices`."""
        return window_slices(shape, window)

    def sync_to_state(self, names=None, *, window=None) -> int:
        """Copy the store onto the attached state.  Returns arrays copied.

        A no-op for a device store (the two are the same object), so a
        caller never has to ask which store it has.
        """
        if not self.host_store:
            return 0
        import cupy as cp

        store = self.store
        copied = 0
        for key, dev in self._state_arrays(names).items():
            src = store.get(key)
            if src is None or getattr(dev, "shape", None) is None:
                continue
            sl = self._window_slices(dev.shape, window)
            dev[sl] = cp.asarray(src[sl])
            copied += 1
        return copied

    def sync_from_state(self, names, *, window=None) -> int:
        """Copy named carriers from the attached state back into the store.

        The other half of :meth:`sync_to_state`, and the half that is easy
        to forget: a consumer that RESETS what it read -- the relocation
        runner zeroes ``uh_follow_window`` at every evaluation, accepted or
        held -- writes that zero to the state, and a store that never hears
        about it keeps accumulating.  The window then silently stops meaning
        "since I last looked" and starts meaning "since the run began",
        which makes every later move more likely for no physical reason.
        ``names`` is required rather than defaulted, because a whole-state
        push-back would overwrite the sweep's own result with a stale copy.
        """
        if not self.host_store:
            return 0
        import cupy as cp

        store = self.store
        copied = 0
        for key, dev in self._state_arrays(names).items():
            dst = store.get(key)
            if dst is None:
                continue
            sl = self._window_slices(dev.shape, window)
            dst[sl] = cp.asnumpy(dev[sl])
            copied += 1
        return copied

    # -- the whole-domain read/write seam ----------------------------------
    #
    # A streamed domain's arrays live in its store; the ``DomainState`` the
    # model tree holds is kept only as an identity check and its arrays stop
    # changing at attach.  Every consumer that reads the STORE is already
    # written that way -- output writes frames from the store
    # (``tilestream.output``, measured 2.2x-2.4x cheaper than the resident
    # path and byte-identical), restart writes the store
    # (``tilestream.restart_stream``).
    #
    # WHOLE-DOMAIN MODEL CONSUMERS ARE NOT, and they cannot be, because they
    # take a ``parent_state``: ``gpuwm.core.nest_spawn.SpawnWatch.evaluate``
    # and ``gpuwm.core.storm_tracking.StormTracker`` both reach their plane
    # through ``storm_tracking.signal_plane(state, ...)``, which is
    # ``state.existing_scratch(slot)``.  Handed a streamed domain's state
    # they read the plane as it was at attach -- for the UH windows, the zero
    # plane ``DomainState.__init__`` allocated -- and make a decision on it
    # with no error and no receipt saying so.
    #
    # These two methods are that seam, and they are deliberately EXPLICIT and
    # per-carrier rather than a sync at the end of every sweep: the store is
    # pinned host RAM, a blanket sync would copy the whole domain back every
    # step, and the consumers that need it run on leg/relocation cadences,
    # not per step.  ``publish`` is store -> state (read the plane), ``adopt``
    # is state -> store (the consumer zeroed its window and the DOMAIN must
    # see that, or the next fold accumulates on top of a value the consumer
    # believes it already spent).

    def publish(self, names) -> tuple[str, ...]:
        """Copy named carriers out of the store onto the attached state.

        ``names`` are manifest keys (``scratch/uh_spawn_window``).  A name the
        store does not hold is REFUSED rather than skipped: a consumer that
        asked for a plane and silently got the stale one is the exact defect
        this seam exists to remove.  Returns the names copied.
        """
        return self._exchange(names, "publish")

    def adopt(self, names) -> tuple[str, ...]:
        """Copy named carriers off the attached state into the store."""
        return self._exchange(names, "adopt")

    def _exchange(self, names, direction: str) -> tuple[str, ...]:
        from tilestream import physics_inventory as _physics

        if self._state is None:
            raise StreamingRefused(
                "this streamed domain was attached without a state, so there "
                "is nothing to exchange whole-domain carriers with")
        names = tuple(names)
        store = self.store
        live = _physics.streaming_manifest(self._state)
        missing = [n for n in names if n not in store or n not in live]
        if missing:
            raise StreamingRefused(
                f"cannot {direction} {missing} between a streamed domain's "
                f"store and its state: the store holds {len(store)} carriers "
                f"and the state's streaming manifest {len(live)}, and these "
                "are in neither or only one.  A whole-domain consumer that "
                "asked for a plane and quietly got the attach-time one is "
                "what this refusal exists to prevent -- attach with "
                "tilestream.physics_inventory.streaming_inventory, which "
                "carries the slots restart deliberately does not.")
        for name in names:
            src, dst = ((store[name], live[name]) if direction == "publish"
                        else (live[name], store[name]))
            dst[...] = _to(dst, src)
        return names




#: The whole-domain planes a storm-following consumer reduces, as carrier
#: keys.  UP_HELI_MAX is included even though nothing tracks it directly:
#: it is what the history writer emits, and a streamed run whose wrfout
#: carried the frozen t=0 accumulator would be wrong in a file rather than
#: in a decision, which is worse.
TRACKER_PLANE_CARRIERS: tuple[str, ...] = (
    "scratch/up_heli_max", "scratch/uh_follow_window",
    "scratch/uh_spawn_window")
def _to(dst, src):
    """``src`` as something ``dst[...] =`` accepts, host or device."""
    import numpy as _np

    if isinstance(dst, _np.ndarray):
        get = getattr(src, "get", None)
        return src if isinstance(src, _np.ndarray) else get()
    import cupy as _cp

    return _cp.asarray(src)


def allocated_planes(state, names) -> tuple[str, ...]:
    """Which of ``names`` this state actually ALLOCATED, as carrier keys.

    Here rather than at the call site because ``gpuwm/runtime.py`` is not a
    sanctioned scratch-API site: ``tests/test_uh_lifecycle.py``'s roster
    forbids any module outside the owner and the five sanctioned files from
    binding ``existing_scratch`` while naming the accumulator, and it is
    right to -- an indirect ``getattr(state, "existing_scratch")`` reads the
    plane without tripping the direct pin.  This module IS sanctioned (it is
    on the roster), so the lookup lives here and the run loop asks for a
    list of names.

    Duck-typed tolerant exactly like ``uh_diag._tracker_windows``: a reduced
    state or a test double without a scratch pool simply has no window, and a
    state built with ``nwp_diagnostics = 0`` never allocated one, so both
    answer with an empty tuple rather than raising.
    """
    existing = getattr(state, "existing_scratch", None)
    if existing is None:
        return ()
    return tuple(name for name in names
                 if existing(name[len("scratch/"):]) is not None)


def make_stepper(state, cfg, options: StreamingOptions | None = None, *,
                 decision: StreamingDecision | None = None,
                 machine=None, build=None):
    """The callable a run loop steps a domain with.

    Returns ``gpuwm.core.dycore.step`` ITSELF whenever streaming does not
    fire -- the mode is off, or ``auto`` found the domain fits resident.
    That identity is the OFF contract: there is no disabled-streaming branch
    to be subtly different, because there is no branch at all.

    ``build`` is the domain-specific construction the seam cannot invent: a
    callable ``build(state, cfg, decision) -> StreamedDomain``.  The store
    has to be filled from the prepared state, the tile buffers have to be
    built with the SAME physics selectors the domain was prepared with (a
    buffer missing a scheme owns different carriers and the inventory check
    refuses it), the domain's geography has to be inventoried, and the
    boundary tables have to be windowed per tile.  Routes own that; this
    module owns when it happens and what it must satisfy.
    """
    from gpuwm.core.dycore import step

    decision = decide(cfg, options, machine=machine) if decision is None \
        else decision
    if not decision.stream:
        return step
    if build is None:
        raise StreamingRefused(
            f"{decision.explain()}, but this route wired no streamed-domain "
            "builder.  A streamed domain needs its store filled from the "
            "prepared state, tile buffers built with the domain's own "
            "physics selectors, the domain geography inventoried and the "
            "lateral boundary tables windowed per tile -- none of which the "
            "seam can invent.  Refused rather than silently integrated "
            "resident, because a run that quietly declined to stream is a "
            "run that will die at the allocation the mode existed to avoid.")
    streamed = build(state, cfg, decision)
    if not callable(streamed):
        raise StreamingRefused(
            "the streamed-domain builder returned something that is not "
            f"callable ({streamed!r}); it must return a StreamedDomain")
    publish_store(state, streamed)
    return streamed


# ---------------------------------------------------------------------------
# the run loop's safety observers, folded per tile
# ---------------------------------------------------------------------------

#: Blocks per tile in the folded stability reduction.  ``stability_report``
#: sizes its own grid as ``min(256, ceil(n/256))`` and 256 is what every
#: domain big enough to be streamed reaches, so matching it keeps the two
#: reductions the same shape as well as the same arithmetic.
_STABILITY_BLOCKS = 256


class StreamedStability:
    """``stability_report`` for a domain that is not on the card.

    THE DEFECT THIS EXISTS FOR
    --------------------------
    :func:`gpuwm.runtime.integrate_prepared_case` guards a forecast with
    three whole-domain observers and hands all three the prepared
    ``DomainState``::

        report = stability_report(state, integration_cfg, boundary_width=w)
        health.require_healthy(phase=...)
        step_swdown_peak = float(cp.max(state.physics.fields["swdown"]))

    Under ``store = "host"`` that state is a copy taken at t=0 --
    :func:`attach` fills the store with ``gather.pinned_copy``, which COPIES
    -- and no sweep ever writes it again.  So the NaN guard never fires, the
    w_max monitor freezes at its initial value, and a domain that went
    non-finite in the store completes "successfully" and writes a checkpoint
    whose ``run_trackers`` record ``nan_free: true``.  The observers are pure
    (they carry nothing into the answer), which is exactly why nothing caught
    it, and the reduction still RUNS every substep -- so the mode was paying
    a real GPU tax for an answer that was wrong.

    WHY A FOLD IS THE ANSWER AND A RESIDENT MIRROR IS NOT
    -----------------------------------------------------
    Refreshing the ``DomainState`` from the store would mean holding the
    whole domain on the card, which is the one thing the mode exists to
    avoid.  There is no resident copy to keep current, so every whole-domain
    observer either becomes a fold or stops being armed -- and this one folds
    cleanly, because ``stability_report`` is a max/OR reduction and tile
    interiors PARTITION the domain (``tilestream.spec.validate_plan``).

    One record per tile, emitted by ``health_partial_tile`` after that
    tile's step and before its interior is scattered back, folded by
    ``health.cu``'s own ``health_final`` and decoded by
    :func:`gpuwm.core.dycore.decode_stability_record` -- the same decoder the
    resident path uses, deliberately not a copy of it.  The result is
    bit-identical to the monolithic reduction, not approximately equal: max
    is associative and exact, the NaN classes are a bitwise OR, and the
    argmax carries a DOMAIN flat index so the "lowest index wins ties" rule
    resolves the same way whatever order the tiles were swept in.

    COST
    ----
    The reduction moves from once per substep over the whole domain to once
    per TILE over that tile's interior, so it reads the same number of cells
    per substep -- but on data that is already resident, issued into the
    tile's own stream with no readback, and the single eight-word host copy
    happens once per model step instead of once per substep.
    """

    def __init__(self, run, cfg, *, boundary_width: int | None = None,
                 blocks_per_tile: int = _STABILITY_BLOCKS,
                 window: str = "interior"):
        import cupy as cp

        if window not in ("interior", "buffer"):
            raise ValueError(
                f"window must be 'interior' or 'buffer', got {window!r}")
        self._run = run
        self.cfg = cfg
        #: ``"buffer"`` is WRONG ON PURPOSE and exists only as the gate's
        #: negative control.  It folds each tile's whole gathered window --
        #: halo included -- instead of its interior.  The halo is at least
        #: the per-step dependency radius so the interior is bit-exact, but
        #: the halo itself was stepped with insufficient neighbours, so a
        #: buffer fold reports maxima the domain never had.  It is the
        #: obvious implementation and it is the one that must FAIL.
        self.window = window
        self.boundary_width = boundary_width
        self.blocks_per_tile = int(blocks_per_tile)
        self.ntiles = len(run.specs)
        self.folds = 0
        self.tile_launches = 0
        #: How many sweeps have STARTED.  Zero means the domain has not been
        #: stepped even once, which is a different thing from a sweep that
        #: skipped tiles and must not be answered the same way.  See
        #: :meth:`__call__`.
        self.sweeps_begun = 0
        self._seen: set[int] = set()
        self._partial = cp.zeros(
            (self.ntiles * self.blocks_per_tile, 9), dtype=cp.float32)
        self._result = cp.zeros((8,), dtype=cp.float32)
        self._report: dict | None = None
        self._nz = int(cfg.nz)
        self._dny, self._dnx = int(cfg.ny), int(cfg.nx)

    def begin_sweep(self) -> None:
        """Forget the previous step's records, so a short sweep is caught.

        Not bookkeeping: a transport that silently skipped a tile would leave
        that tile's PREVIOUS record in the partial buffer, and a stale max
        folded in with fifteen current ones looks exactly like a healthy
        domain.  With the set cleared here, :meth:`__call__` refuses instead.
        """
        self.sweeps_begun += 1
        self._seen.clear()
        self._report = None

    # -- the per-tile half, inside the sweep --------------------------------

    def observe(self, tile_state, tspec, itile, stream) -> None:
        """``TiledRun``'s ``observer``: one tile's contribution, issued."""
        import numpy as np

        from gpuwm.core import constants as c
        from gpuwm.core.kernels import get_kernel
        from gpuwm.core.state import DTYPE

        # The interior in BUFFER coordinates.  ``halo_left``/``halo_south``
        # are the true offsets even for the clamped edge tiles, whose window
        # is pushed into the domain and whose halo is therefore lopsided.
        jb, ib = int(tspec.halo_south), int(tspec.halo_left)
        iny, inx = int(tspec.interior_ny), int(tspec.interior_nx)
        bny, bnx = int(tspec.cny), int(tspec.cnx)
        jd, idm = int(tspec.j0), int(tspec.i0)
        if self.window == "buffer":
            jb = ib = 0
            iny, inx = bny, bnx
            jd, idm = int(tspec.cj0), int(tspec.ci0)
        # u's closing face belongs to exactly one tile, and only counting it
        # there makes the tiles' u interiors add up to the domain's nx+1
        # columns exactly once.
        #
        # PERIODIC IS DELIBERATELY EXCLUDED, and this is not the same
        # question ``owns_x_alias`` answers.  Under ``periodic=True``
        # ``TileSpec._axis_gather`` reduces every window mod nx and never
        # reads the alias slot, so no buffer holds domain column nx at all --
        # while the SCATTER writes it, from the owning tile's column 0.  In a
        # tiled periodic run u[..., nx] is therefore a literal copy of
        # u[..., 0], which the fold already covers, and reaching for buffer
        # column cnx here would fold a NEIGHBOUR's column in under the alias's
        # name.  Under ``periodic=False`` column nx is a real east boundary
        # face 1800 km from column 0 and the east tile really does gather it.
        u_extra = 1 if (not tspec.periodic and tspec.i1 == tspec.nx) else 0
        phb = getattr(tile_state, "phb", None)
        php = getattr(tile_state, "php", None)
        have_geo = int(phb is not None and php is not None)
        phb_full = int(have_geo and phb.ndim == 3)
        if not have_geo:
            php = phb = tile_state.w
        kernel = get_kernel("health_tile", "health_partial_tile")
        with stream:
            kernel((self.blocks_per_tile,), (256,),
                   (tile_state.u, tile_state.w, tile_state.thp, php, phb,
                    self._partial,
                    np.int32(int(itile) * self.blocks_per_tile),
                    np.int32(bny), np.int32(bnx),
                    np.int32(jb), np.int32(ib),
                    np.int32(iny), np.int32(inx),
                    np.int32(jd), np.int32(idm),
                    np.int32(self._dny), np.int32(self._dnx),
                    np.int32(self._nz), np.int32(u_extra),
                    np.int32(phb_full), np.int32(have_geo),
                    np.int32(0 if self.boundary_width is None
                             else int(self.boundary_width)),
                    DTYPE(c.G)))
        self.tile_launches += 1
        self._seen.add(int(itile))
        self._report = None

    # -- the whole-domain half, once per model step -------------------------

    def __call__(self, state=None, cfg=None, *, boundary_width=None) -> dict:
        """:func:`gpuwm.core.dycore.stability_report`'s signature, exactly.

        ``state`` is accepted and IGNORED once the domain has been swept,
        because the whole point is that it is not where the domain is.  It is
        read in exactly one case -- before the FIRST sweep, see below -- which
        is the one moment it is still the domain.  ``boundary_width`` must
        match the width the tiles were classified against: the
        boundary/interior split is baked into the per-tile records and cannot
        be re-cut afterwards.
        """
        width = (self.boundary_width if boundary_width is None
                 else boundary_width)
        # Compared as INTEGERS with None folded to 0, because the silent
        # failure is the asymmetric one: a record cut at width 0 and read at
        # width 5 comes back with boundary_w_max = interior_w_max = 0.0 --
        # two plausible numbers that are simply not the domain's, on the
        # exact axis this whole module exists to stop lying about.
        if int(width or 0) != int(self.boundary_width or 0):
            raise StreamingRefused(
                f"the folded stability record was cut at boundary_width="
                f"{self.boundary_width} inside the sweep and is being read "
                f"at {boundary_width}.  The boundary/free-interior split is "
                "decided per tile against the DOMAIN's extents while the "
                "tile is on the card; it cannot be re-cut from the folded "
                "record.")
        if self.sweeps_begun == 0:
            # (imports deliberately below this branch: answering the analysis
            # frame needs no kernel and no device)
            # THE ANALYSIS FRAME.  A route that publishes a t = 0 history
            # frame asks for its health before anything has been stepped, and
            # there is no fold yet because there has been no sweep -- not a
            # short one, none.  The honest answer is the one the resident path
            # would give: attach COPIED this state into the store and no sweep
            # has written either since, so the state still IS the initial
            # condition the frame contains.  This is the only moment that is
            # true, which is why it is keyed on sweeps_begun rather than on an
            # empty record: once a sweep has run, the state is the corpse this
            # class exists to stop reading and a short sweep must REFUSE.
            if state is None:
                raise StreamingRefused(
                    "the stability record was read before the first sweep and "
                    "without the domain state, so there is nothing to report "
                    "the initial condition from")
            from gpuwm.core.dycore import stability_report
            return stability_report(
                state, self.cfg if cfg is None else cfg,
                boundary_width=width)

        if len(self._seen) != self.ntiles:
            missing = sorted(set(range(self.ntiles)) - self._seen)
            raise StreamingRefused(
                f"{len(self._seen)} of {self.ntiles} tiles contributed a "
                f"stability record this sweep (missing {missing[:8]}); "
                "reading it now would report the health of part of a domain "
                "as the health of all of it, which is the exact shape of the "
                "defect this fold exists to remove")

        import cupy as cp
        import numpy as np

        from gpuwm.core.dycore import decode_stability_record
        from gpuwm.core.kernels import get_kernel

        if self._report is None:
            # Under the driver's deferred sweep seam the per-tile records
            # were issued on the COMPUTE streams and nothing has barriered
            # since.  Waiting on the compute streams alone is deliberate:
            # the final fold needs every tile's kernels finished, not its
            # transfers landed, and a full drain here would re-expose
            # exactly the scatter tail the deferred seam hides.
            sync = getattr(self._run, "sync_compute", None)
            if sync is not None:
                sync()
            kernel = get_kernel("health", "health_final")
            kernel((1,), (256,),
                   (self._partial, self._result,
                    np.int32(self.ntiles * self.blocks_per_tile)))
            host = cp.asnumpy(self._result)
            self._report = decode_stability_record(
                host, self.cfg if cfg is None else cfg,
                boundary_width=width)
            self.folds += 1
        return dict(self._report)


def stability_observer(stepper):
    """The callable a run loop takes its per-substep stability record with.

    Returns :func:`gpuwm.core.dycore.stability_report` ITSELF for a resident
    domain -- the same OFF contract :func:`make_stepper` keeps, and for the
    same reason: there is no "not streaming" branch of the observer to be
    subtly different, because there is no branch at all.  A streamed domain
    returns its :class:`StreamedStability`, which has the identical
    signature and reduces over the memory the sweep actually writes.
    """
    from gpuwm.core.dycore import stability_report

    folded = getattr(stepper, "stability", None)
    return stability_report if folded is None else folded


def refresh_streamed_state(stepper, state) -> int:
    """Bring ``state`` up to date from the store, for whoever is about to read it.

    The OFF contract, like :func:`stability_observer`: a resident stepper is
    a ``getattr`` and a zero, so a route calls this unconditionally and a run
    with no ``[tiles]`` is unchanged.

    A streamed domain under ``store = "host"`` leaves the ``DomainState`` as
    the snapshot that filled the store, and three readers on this route are
    not foldable the way the stability record was: ``StateHealthValidator``
    is one block per whole field with no tile-interior form,
    ``canonical_state_digest`` is a whole-trajectory hash, and the
    diagnosis path reads fields directly.  Left alone they answer for the
    INITIAL CONDITION and say so nowhere -- a receipt reporting
    ``nan_free: true`` and a final digest that is the analysis, on a run
    that integrated for an hour.

    :meth:`StreamedDomain.refresh_state` is the copy, and its docstring puts
    it "on the HISTORY cadence and at the end of the run, never per step":
    it is one domain-sized D2H/H2D of the carrier set, which is what one
    history frame already costs.  It allocates nothing -- ``attach`` needed a
    resident state to fill the store from, so the destination already
    exists.  Returns the number of carriers copied, so a caller can record
    that the refresh moved the whole manifest rather than a lucky subset.
    """
    if not is_streaming(stepper):
        return 0
    return int(stepper.refresh_state(state) or 0)


def domain_call_counts(stepper, state) -> dict:
    """The DOMAIN's physics call counters, wherever the domain's clock is.

    ``PhysicsDriver.call_counts`` on the prepared state is another observer
    of the corpse: ``dycore.step`` increments the counters on whichever state
    it stepped, and under streaming that is a tile BUFFER.  The domain's own
    counters are the scalar carriers the sweep advances exactly once per
    model step (``tilestream.physics_inventory.carrier_scalars``), which is
    also what a restart writes, so this is the same number either way.

    A streamed domain that carries no counters REFUSES rather than falling
    back to the state's.  The fallback looks harmless and is not: the state's
    counters stopped moving when the store was filled, so the run summary
    would report the radiation cadence of t = 0 for the whole forecast, which
    is the same class of silent staleness the fold exists to remove.
    """
    if is_streaming(stepper):
        counts = (stepper.scalars or {}).get("call_counts")
        if counts is None:
            raise StreamingRefused(
                "the streamed domain carries no call_counts, so the run "
                "loop cannot report how often the surface forcing updated")
        return dict(counts)
    return dict(state.physics.call_counts)


def domain_field_max(stepper, state, key: str, attr: str) -> float:
    """``max`` over one whole-domain field, read where the domain lives.

    ``runtime.integrate_prepared_case`` samples ``swdown``'s peak once per
    OUTER step off ``state.physics.fields``, which is the same corpse the
    stability reduction was reading.  Once per outer step over one 2-D field
    is small enough that the honest fix is to read the store on the host --
    no kernel, no sweep hook, and correct for a field no tile writes.
    """
    import cupy as cp
    import numpy as np

    if is_streaming(stepper):
        value = stepper.store.get(key)
        if value is not None:
            return float(np.asarray(value).max())
    return float(cp.max(attr))


# ---------------------------------------------------------------------------
# lateral boundaries, windowed per tile
# ---------------------------------------------------------------------------

#: Horizontal staggering of each coupled LBC field, as ``(extra_y, extra_x)``
#: on top of the mass grid.  ``lateral_bc._coupled_device_fields`` emits ``u``
#: at ``(nz, ny, nx+1)`` and ``v`` at ``(nz, ny+1, nx)``; everything else sits
#: at mass points.  The side tables inherit that staggering in the TANGENTIAL
#: direction, which is what makes windowing them a per-field question rather
#: than one slice applied to all four sides.
_LBC_STAGGER: dict[str, tuple[int, int]] = {"u": (0, 1), "v": (1, 0)}


def owned_edges(spec) -> dict[str, bool]:
    """Which of a tile's four sides are TRUE DOMAIN EDGES.

    On a non-periodic axis :func:`tilestream.spec.plan_tiles` CLAMPS the
    compute window into the domain, so a side is a true edge exactly when the
    WINDOW touches the domain's own extent -- not when the interior does.
    Both facts matter and they are different tiles' business: the interior
    decides what is scattered back, the window decides which cells the
    boundary kernel writes.

    A PERIODIC axis has no true edges at all, and the axes are asked
    separately because they can disagree (``open_x`` without ``open_y``).
    Without the axis test a periodic plan whose tile 0 happens to land at
    ``cj0 == 0`` -- which it does whenever ``halo`` divides ``tile_ny`` --
    would be told it owns the south domain edge of a domain that has none.
    """
    return {
        "west": (not spec.periodic_x) and spec.ci0 == 0,
        "east": (not spec.periodic_x) and spec.ci0 + spec.cnx == spec.nx,
        "south": (not spec.periodic_y) and spec.cj0 == 0,
        "north": (not spec.periodic_y) and spec.cj0 + spec.cny == spec.ny,
    }


def window_interval(interval, spec, *, width: int, seam: str = "zeros",
                    snapshot=None):
    """One domain ``BoundaryInterval``, windowed onto one tile.

    A side that is a TRUE DOMAIN EDGE keeps the domain's own value and
    tendency, sliced along the TANGENTIAL axis to the tile's compute window,
    with the field's own staggering (``u``'s south/north tables are ``cnx+1``
    wide, ``v``'s west/east tables are ``cny+1`` tall).  East and north
    tables are stored outermost-first, so the same tangential slice is
    correct for them and no reversal is needed.

    A side that is an INTERIOR SEAM gets tables that are INERT, and the whole
    mode rests on the measurement that says they may be:

    ``"zeros"``
        value and tendency both zero.  Cheap, deterministic, and obviously
        not the domain's data, so a seam that reached the interior would show.

    ``"self"``
        the domain's own coupled values at that seam (from ``snapshot``,
        windowed to the tile) with zero tendency -- the self-consistent case,
        physically the most defensible and numerically a completely different
        set of numbers from zeros.

    ``"poison"``
        deliberate garbage: 1e6 in coupled units (roughly 500x a real coupled
        wind) and 1e4/s of tendency.  This is the sharp form of the control.
        Zeros and self-consistent are both plausible, and a cone that reached
        the interior only weakly might survive both; garbage would not.

    MEASURED at 256x192x49, tile 32x32, real Lambert + real terrain +
    specified BCs, halo 16 from ``harness.halo_radius``: all three give the
    BIT-IDENTICAL answer, out to N=24.  The controls that must fire do --
    tables not windowed at all, and true-edge tables scaled by 1.000001,
    both move all nine dry carriers.

    Why the seam cannot reach the interior, given that the halo is only the
    dycore's own dependency radius: the seam forcing perturbs the RK
    TENDENCY, not the state, and a tendency injected at stage 0 is advected
    by fewer stages than a state perturbation present at the start of the
    step, so its cone is strictly inside the dycore's.  The prediction that
    ``halo >= halo_radius + spec_bdy_width`` would be needed is REFUTED: the
    smallest passing halo is 15, one cell BELOW the dycore radius, not five
    above it.
    """
    import numpy as np

    from gpuwm.ingest.lateral_bc import (BoundaryInterval, FieldBoundary,
                                         SideBoundary)

    if seam not in ("zeros", "self", "poison"):
        raise ValueError(f"unknown seam mode {seam!r}")
    owns = owned_edges(spec)
    width = int(width)
    fields_out = {}
    for name, boundary in interval.fields.items():
        ey, ex = _LBC_STAGGER.get(name, (0, 0))
        y0, y1 = spec.cj0, spec.cj0 + spec.cny + ey
        x0, x1 = spec.ci0, spec.ci0 + spec.cnx + ex
        nzf = boundary.west.value.shape[0]
        wshape = (nzf, spec.cny + ey, width)
        sshape = (nzf, width, spec.cnx + ex)

        seam_tables: dict[str, Any] = {}
        if seam == "self":
            if snapshot is None:
                raise ValueError("seam='self' needs the domain snapshot")
            full = np.asarray(snapshot[name], dtype=np.float64)
            if full.ndim == 2:
                full = full[None]
            win = full[:, y0:y1, x0:x1]
            seam_tables = {
                "west": np.ascontiguousarray(win[..., :width]),
                "east": np.ascontiguousarray(win[..., -width:][..., ::-1]),
                "south": np.ascontiguousarray(win[..., :width, :]),
                "north": np.ascontiguousarray(
                    win[..., -width:, :][..., ::-1, :]),
            }

        sides = {}
        for side_name in ("west", "east", "south", "north"):
            side = getattr(boundary, side_name)
            tangential_y = side_name in ("west", "east")
            want = wshape if tangential_y else sshape
            if owns[side_name]:
                value = (side.value[:, y0:y1, :] if tangential_y
                         else side.value[:, :, x0:x1])
                tend = (side.tendency[:, y0:y1, :] if tangential_y
                        else side.tendency[:, :, x0:x1])
                value = np.ascontiguousarray(value)
                tend = np.ascontiguousarray(tend)
            elif seam == "self":
                value = seam_tables[side_name]
                tend = np.zeros_like(value)
            elif seam == "zeros":
                value = np.zeros(want, dtype=np.float64)
                tend = np.zeros(want, dtype=np.float64)
            else:
                rng = np.random.default_rng(
                    abs(hash((spec.ty, spec.tx, name, side_name)))
                    % (2 ** 32))
                value = 1.0e6 * rng.standard_normal(want)
                tend = 1.0e4 * rng.standard_normal(want)
            if value.shape != want:
                raise ValueError(
                    f"tile {spec.index} {name}/{side_name} windowed to "
                    f"{value.shape}, expected {want}")
            sides[side_name] = SideBoundary(value, tend)
        fields_out[name] = FieldBoundary(**sides)
    return BoundaryInterval(interval.start_seconds, interval.end_seconds,
                            fields_out)


def window_boundaries(bnd, spec, *, seam: str = "zeros", snapshot=None):
    """The domain's ``LateralBoundaries`` windowed onto one tile."""
    from gpuwm.ingest.lateral_bc import LateralBoundaries

    return LateralBoundaries(
        tuple(window_interval(iv, spec, width=bnd.spec_bdy_width, seam=seam,
                              snapshot=snapshot)
              for iv in bnd.intervals),
        bnd.spec_bdy_width, bnd.spec_zone, bnd.relax_zone)


def tile_boundary_tables(bnd, specs, *, seam: str = "zeros", snapshot=None):
    """Every tile's windowed forcing, precomputed once on the host.

    Once, not per step: the tables are a function of the tiling and of the
    domain's forcing, and neither moves during a run.  ``dtbc`` -- the
    interpolation weight inside the interval -- is a function of
    ``elapsed_seconds``, which is a CARRIER and is re-imposed on the buffer
    before every tile step.
    """
    return [window_boundaries(bnd, s, seam=seam, snapshot=snapshot)
            for s in specs]


#: ``external_clock=`` sentinel: "derive the clock from ``domain_state``".
#: Distinct from ``None``, which is now a POLICY -- legacy elapsed-seconds
#: compatibility, stated on purpose -- rather than the absence of one.
DERIVE_CLOCK = object()


def make_tile_hook(per_tile, *, domain_state=None,
                   external_clock=DERIVE_CLOCK):
    """A ``TiledRun`` ``tile_hook`` that binds tile ``itile``'s forcing.

    LAZILY, which is a measured fix and not a preference.  The eager
    ``attach_lateral_boundaries`` re-validates, re-packs and re-uploads
    EVERY forcing interval on every buffer-tile change: 27-63 ms per bind
    on a real tile buffer, host-blocking, growing with interval count, tile
    count and boundary size -- ~0.3 s/step at the attribution run's largest
    arm, all of it serialised into the sweep
    (``tilestream/OVERLAP-ATTRIBUTION.md``).  The streaming attachment
    (``attach_streaming_lateral_boundaries``) allocates ONE packed device
    slot per buffer, sized for a single interval; every later bind swaps
    the host tables and invalidates the resident interval id, so the next
    specified-boundary launch reloads that same slot -- measured at
    ~0.01 ms.  Same numbers on the card either way: the reload runs the
    identical float64 -> float32 conversion the eager upload runs, and
    :func:`tilestream.realcase.tile_boundary_binder` -- this hook's twin,
    from which the swap discipline is taken verbatim -- carries the digest
    proof.

    The FIRST bind per buffer converts that buffer from whatever attachment
    its factory gave it (the production factory attaches tile 0's tables
    eagerly so the warmup step can run) to the streaming attachment;
    conversion releases the eager scratch.  A later bind that finds the
    streaming attachment gone refuses: something re-attached behind the
    hook's back, and swapping tables under an eager attachment would leave
    the device serving the OLD tile's forcing with no error anywhere.

    ``domain_state`` is the DOMAIN's own prepared state, and it is here to
    carry the Davies CLOCK BINDING onto every buffer (task #219).  Every
    production driver-forced root binds a ``DomainClock`` to its external
    LBC mirror (``bind_lateral_boundary_clock``), which switches Davies
    consumers to WRF's post-increment ``dtbc`` recurrence (``dt..T_bdy``).
    The buffers this hook converts start with NO binding -- their factory
    attachment never had one -- so without the rebind below every tile
    step fell back to the retired ``elapsed - interval.start`` path
    (``0..T-dt``) and a streamed domain consumed its lateral boundaries
    ONE TIMESTEP LATE whenever the production clock was bound.  MEASURED
    on the offline child at t+15 min: 41 of 76 wrfout fields differed, W
    by 0.0043 m/s on a 0.174 m/s field, a constant one-step phase error.
    The clock is read LAZILY, at first bind, because binding order is a
    route decision this module must not constrain; a domain with no bound
    clock binds nothing and the buffers keep the compatibility semantics,
    which is what keeps the legacy direct routes bit-for-bit.

    ``external_clock`` IS THE POLICY, and it exists because lazy derivation
    was the wrong abstraction for a domain that never had a state.
    ``DERIVE_CLOCK`` (the default) is the behaviour above: read the binding
    off ``domain_state`` at first bind, which is right for a re-attached
    resident domain and is what keeps the legacy direct routes bit-exact.
    A ``DomainClock`` binds THAT clock unconditionally and needs no state.
    ``None`` selects legacy elapsed-seconds compatibility EXPLICITLY.

    The distinction is not decorative.  ``store_domain_builder`` attaches
    with no state at all, so ``domain_state`` is ``None``, ``domain_clock()``
    returned ``None``, and ``converted[id(tile_state)]`` latched on the first
    bind and was never revisited -- so every buffer of a store-direct
    forecast took ``elapsed - interval.start`` (``0..T-dt``) while its
    resident twin took ``dtbc`` (``dt..T_bdy``).  That is the #219 one-step
    phase error, on the road the LARGEST domains take and the one least
    likely to have a resident control beside it.  Deriving nothing from
    nothing cannot be distinguished from deciding to derive nothing, so the
    decision is made a parameter and the store-direct road passes the node's
    own clock.

    And it is checked PER LAUNCH, not only at conversion: a buffer whose
    bound clock is not the expected object is refused rather than stepped.
    A latch that is set once and trusted forever is exactly how the first
    defect survived a gate.
    """
    from gpuwm.ingest.lateral_bc import (attach_streaming_lateral_boundaries,
                                         bind_lateral_boundary_clock)

    converted: dict[int, bool] = {}

    def domain_clock():
        if external_clock is not DERIVE_CLOCK:
            return external_clock
        resident = getattr(domain_state, "_lateral_boundary_device", None)
        if resident is None or getattr(resident, "rolling", False):
            return None
        return resident.clock

    def hook(tile_state, tspec, itile, stream):
        lb = per_tile[itile]
        if not converted.get(id(tile_state)):
            attach_streaming_lateral_boundaries(tile_state, lb)
            clock = domain_clock()
            if clock is not None:
                bind_lateral_boundary_clock(tile_state, clock)
            converted[id(tile_state)] = True
            return
        resident = getattr(tile_state, "_lateral_boundary_device", None)
        if resident is None or not getattr(resident, "streaming_external",
                                           False):
            raise StreamingRefused(
                "a tile buffer this hook had converted to the streaming "
                "lateral attachment no longer carries it; refusing to swap "
                "tables under an eager attachment, which would serve the "
                "previous tile's forcing silently")
        # THE CLOCK, EVERY LAUNCH.  The conversion latch above records that
        # a buffer was bound; it cannot record that the binding SURVIVED,
        # and a buffer that lost its clock keeps integrating with the
        # legacy elapsed-seconds recurrence and says nothing -- a constant
        # one-timestep phase error in the lateral forcing, which is a
        # plausible forecast rather than a failure.
        expected = domain_clock()
        if getattr(resident, "clock", None) is not expected:
            raise StreamingRefused(
                "a tile buffer's bound Davies clock is not the one this "
                f"attachment was built with (expected {expected!r}, found "
                f"{getattr(resident, 'clock', None)!r}).  Stepping it would "
                "consume the lateral boundaries one timestep out of phase "
                "with the domain, which is a healthy-looking wrong "
                "forecast, so it is refused instead.")
        tile_state.lateral_boundaries = lb
        # Force the next _resident_interval() to refill the packed slot
        # from THIS tile's tables.  Without it a buffer keeps serving the
        # previous tile's boundary whenever the two tiles happen to be in
        # the same forcing interval -- which is almost always.
        resident.active_host_interval_id = None

    return hook


# ---------------------------------------------------------------------------
# the one-frame REFL_10CM handoff, across a sweep
# ---------------------------------------------------------------------------

#: The store key a streamed domain publishes REFL_10CM out of.  It is the
#: ``refl_10cm`` SCRATCH slot's own restart-member name, so a run that wants
#: reflectivity simply includes it in its ``inventory_fn`` and the ordinary
#: gather/scatter does the rest.
REFL_STORE_KEY = "scratch/refl_10cm"

#: The ``refl_10cm`` scratch slot's own name, on the state side of the key.
REFL_SCRATCH_SLOT = "refl_10cm"


def prime_refl_10cm(state, cfg) -> bool:
    """Allocate the REFL_10CM scratch slot so the transport can carry it.

    ``refl_10cm`` is a REBUILT scratch slot: microphysics computes it into
    the slot when a frame is due and nothing carries it between steps, so it
    is correctly absent from the carrier manifest and therefore from the
    store.  Under streaming that absence is fatal rather than harmless --
    each tile computes its own window and the transport, having no home for
    it, throws every one away -- which is why
    :meth:`StreamedDomain.__call__` refuses a due frame outright.

    Priming is unconditional, and deliberately so.  Whether reflectivity will
    ever be due is a question only the model's history cadence can answer,
    and attach happens long before the first frame; the alternative to
    priming always is deciding wrong and stopping a forecast an hour in, at
    the first due frame, which is exactly what the product route did.  The
    cost is one domain-sized float32 among the ~130 carriers a physics-on
    domain already streams, under 1% of the store.

    Returns whether the slot had to be created, so a caller can report it.
    """
    if state is None or not hasattr(state, "scratch"):
        return False
    # The slot name is a LITERAL at both call sites, not REFL_SCRATCH_SLOT.
    # tests/test_preflight.py's scratch-completeness gate reads these with an
    # AST scan and can only check a slot it can see: a variable expression is
    # classified "variable" and refused unless pinned in an allowlist, which
    # would exempt this slot from the registry check rather than satisfy it.
    if state.existing_scratch("refl_10cm") is not None:
        return False
    state.scratch((int(cfg.nz), int(cfg.ny), int(cfg.nx)), "refl_10cm")
    return True


def prime_lazy_carriers(state, cfg) -> tuple:
    """Allocate every carrier that would otherwise appear on first USE.

    THE WARM-UP STEP THIS REPLACES.  A tile buffer used to be handed one
    throwaway ``dycore.step`` before it served any tile, for one reason:
    two carriers are allocated lazily, so an inventory taken at construction
    is SHORTER than one taken after a step and ``TiledRun`` refuses the
    mismatch.  ``KainFritsch.ensure_trigger_history`` says it outright --
    "make_physics_tile_state works around it by throwing a warm-up step
    away; the array is (nz, ny, nx) zeros and there is nothing to warm up".

    The workaround is worse than it looks, because a buffer at construction
    holds NOTHING the domain holds.  Its thermodynamics are an analytic
    sounding, its geography is ``harness.neutral_geography``, and -- on a
    specified-boundary domain -- the lateral tables attached to it are the
    DOMAIN's real forcing.  So the throwaway step blended real GFS boundary
    values into an analytic interior and integrated the result.

    MEASURED on the product front door, 550x550x49: that step handed
    rte-rrtmgp a temperature field spanning 211.0 K to 456.7 K and the
    gas-table validator refused it.  Building the buffer on the domain's own
    eta table (:func:`domain_vertical_coord`) had already moved the low end
    from 120.3 K to 211.0 K -- necessary, and not sufficient, because the
    remaining error is not in the coordinate but in stepping fabricated data
    at all.  Legacy RRTMG has no such validator and integrated it silently.

    Priming is what the warm-up was FOR, done directly: allocate the
    carriers, integrate nothing.  Returns the names primed, so a caller can
    report what it did rather than assume.
    """
    primed = []
    if prime_refl_10cm(state, cfg):
        primed.append(REFL_STORE_KEY)
    primed.extend(prime_hmix_k_diag(state, cfg))
    driver = getattr(state, "physics", None)
    cumulus = getattr(driver, "cumulus_callable", None)
    ensure = getattr(cumulus, "ensure_trigger_history", None)
    if ensure is not None:
        ensure(state)
        primed.append("cumulus/w0avg")
    return tuple(primed)


def prime_hmix_k_diag(state, cfg) -> tuple:
    """Allocate the eddy-viscosity PRODUCER slots so the transport can carry them.

    ``XKMH``/``XKHH`` are the ``OLR`` case one level down.  Their frame rows
    come out of ``PhysicsDriver.hmix_k_diag``, but that bundle is a COPY
    taken when a frame is built (gpuwm/core/physics.py:3727-3745) and carries
    nothing between steps -- so scattering the bundle moves zeros, and
    ``tilestream.output`` measured exactly that: with the bundle alone in the
    inventory both fields came back bit-exact against the monolithic
    reference BECAUSE BOTH WERE EXACTLY ZERO, a control passing for the wrong
    reason.  The producer is the dycore's ``smag_km``/``smag_kh`` scratch,
    and that is what :data:`tilestream.output._DIAGNOSTIC_PRODUCERS` names
    and what has to exist before the store is sized.

    Both slots are allocated by ``dycore.step`` on first use, which is one
    step too late: attach takes the inventory that sizes the store, so a
    slot that appears later has nowhere to land and the first history frame
    is refused an hour into the forecast.  Primed here for the same reason
    and on the same cadence as ``refl_10cm``.

    Only under ``cfg.hmix_k_diag`` -- off by default -- because that flag is
    what puts the rows in the frame at all.  Two full 3-D fields is 8 B/cell,
    100x ``OLR``'s, and a run that does not ask for the diagnostic must not
    pay it.  The slot names are LITERALS at every call site for the reason
    :func:`prime_refl_10cm` spells out: tests/test_preflight.py's
    scratch-completeness gate reads them with an AST scan.
    """
    if state is None or not hasattr(state, "scratch"):
        return ()
    if not bool(getattr(cfg, "hmix_k_diag", False)):
        return ()
    shape = (int(cfg.nz), int(cfg.ny), int(cfg.nx))
    primed = []
    if state.existing_scratch("smag_km") is None:
        state.scratch(shape, "smag_km")
        primed.append("diag/XKMH")
    if state.existing_scratch("smag_kh") is None:
        state.scratch(shape, "smag_kh")
        primed.append("diag/XKHH")
    return tuple(primed)


def refl_inventory(base):
    """``base`` plus the REFL_10CM slot, for every object the run inventories.

    The transport applies one ``inventory_fn`` to three different kinds of
    object -- the domain state at attach, the store mapping, and each tile
    buffer -- so the wrapper adds the slot the way each kind carries it: out
    of ``existing_scratch`` for a state or a buffer, and already present by
    key for the store, which was built from the state's inventory.  Never
    allocates: :func:`prime_refl_10cm` is what creates the slot, and a slot
    that is genuinely absent stays absent so the refusal can still fire.
    """
    def inventory(obj, names=None):
        live = dict(base(obj, names))
        if REFL_STORE_KEY in live:
            return live
        existing = getattr(obj, "existing_scratch", None)
        if existing is None:
            return live
        slot = existing("refl_10cm")
        if slot is not None and (names is None or REFL_STORE_KEY in names):
            live[REFL_STORE_KEY] = slot
        return live

    return inventory


def diagnostic_inventory(base):
    """``base`` plus the output-only driver diagnostics, on all three kinds.

    THE SIBLING OF :func:`refl_inventory`, and the same shape of fix.  A
    frame row that physics writes, output publishes and nothing reads back
    is classified REBUILT by ``gpuwm/io/restart.py``, so it is correctly
    absent from the carrier manifest -- and therefore from the store, and
    therefore from the sweep's gather and scatter.  Each tile computes its
    own window of it correctly and the transport throws every one away, so
    the buffer ends a sweep holding the LAST tile's window and the domain
    has no copy at all.  ``OLR`` is that row at every shipped rung;
    ``XKMH``/``XKHH`` join it under ``cfg.hmix_k_diag`` and the SASE flux
    rows under SASE.

    Listing them here makes the ordinary gather/scatter carry them, which
    :func:`tilestream.output.diagnostic_inventory` measured bit-exact
    against a monolithic run (``tilestream.test_io.
    case_diagnostic_scatter``, which asserts the negative too: without the
    scatter the published field is wrong over ``(tiles-1)/tiles`` of the
    domain).  They do NOT become carriers by being listed -- nothing reads
    them back into the trajectory, so a run that carries them integrates
    the identical forecast.  The price is what :func:`tilestream.output.
    scatter_cost` prints: 0.082 B/cell for ``OLR``, 8 B/cell for the
    eddy-viscosity pair.

    WHY THIS WRAPS RATHER THAN REPLACING ``output.diagnostic_inventory``:
    that function composes over ``carrier_manifest`` and this route's base
    is ``streaming_inventory`` (the RESTART manifest, which is the wider
    set -- see :func:`attach`).  Composing keeps one naming, one table and
    one place where a new output-only attribute has to be declared.

    Never allocates, exactly as :func:`refl_inventory` never does:
    :func:`prime_lazy_carriers` is what creates a producer slot, and a slot
    that is genuinely absent stays absent so
    :class:`tilestream.output.ShortFrameRefused` can still fire at the
    first frame rather than a plausible array of zeros being published for
    the whole run.
    """
    from tilestream import output as _output

    def inventory(obj, names=None):
        live = dict(base(obj, names))
        for key, array in _output.diagnostic_members(obj).items():
            if key in live:
                continue
            if names is None or key in names:
                live[key] = array
        return live

    return inventory


def streamed_store_inventory():
    """THE inventory rule a streamed domain's store and its buffers share.

    One expression, stated once, because the two things it builds are
    compared against each other before the first step:
    :class:`tilestream.driver.TiledRun` refuses to run when a tile buffer's
    inventory and the store's differ by a single key, and both are produced
    by whatever ``inventory_fn`` this seam hands out.  Written out twice it
    was already wrong once -- ``store_from_prepared_cache`` harvested its
    slabs with the plain ``physics_inventory.carrier_inventory`` while the
    builders handed ``attach`` this rule, and the store came out one carrier
    short: ``tile state inventory [143] != store inventory [142] ... in TILE
    not in STORE: ['scratch/refl_10cm']``.

    ``streaming_inventory`` rather than ``carrier_inventory`` because the
    sweep's carrier set is the streamed one (see that function's docstring
    on RESTART_ONLY_DRIVER_SLOTS), and :func:`refl_inventory` on top because
    REFL_10CM is REBUILT scratch that no manifest carries by construction --
    the slot the tiles compute into and the store must join.

    :func:`diagnostic_inventory` outermost, for the same one-expression
    reason: the output-only driver diagnostics (``OLR``, and
    ``XKMH``/``XKHH`` under ``hmix_k_diag``) joined the sweep's gather and
    scatter at 2.2.2, and a store built without them would differ from the
    tile buffers by exactly those keys -- the key-for-key comparison this
    function exists to keep green would refuse the run before the first
    step.
    """
    from tilestream import physics_inventory as _physinv

    return diagnostic_inventory(refl_inventory(_physinv.streaming_inventory))


def refl_handoff_hook():
    """A ``TiledRun`` ``post_step_hook`` that clears each tile's REFL stash.

    ``gpuwm.core.refl.stash_refl_10cm`` parks the microphysics call's
    REFL_10CM on the physics driver and REFUSES to overwrite an unconsumed
    handoff.  That refusal is correct for a resident domain -- a second
    unconsumed field there really is a cadence bug, two microphysics calls
    between two history frames -- and it is wrong for a sweep, where the
    handoff is per TILE and the frame is per DOMAIN.  Without this hook the
    SECOND tile a buffer serves raises "REFL_10CM stash was not consumed
    before reuse", naming a defect that is not present and stopping the
    forecast at the first history step.

    Clearing it loses nothing.  The handoff is a REFERENCE to the
    state-owned ``refl_10cm`` scratch slot; the numbers are in the slot, the
    slot is a domain-decomposable array like any other, and a run that lists
    :data:`REFL_STORE_KEY` in its inventory has the scatter join the tile
    windows into the domain field.  A run that does NOT list it is refused
    by :meth:`StreamedDomain.__call__` before the first due step rather than
    quietly publishing whatever the slot last held.
    """
    from gpuwm.core.refl import consume_refl_10cm, refl_10cm_is_stashed

    def hook(tile_state, tspec, itile, stream):
        if refl_10cm_is_stashed(tile_state):
            consume_refl_10cm(tile_state)

    return hook


# ---------------------------------------------------------------------------
# attaching a prepared domain to the streaming transport
# ---------------------------------------------------------------------------

def tile_specs(cfg, decision: StreamingDecision):
    """The tiling ``decision`` describes, as :class:`tilestream.spec.TileSpec`.

    Exposed because a route has to window the boundary tables BEFORE it can
    build its tile buffers: ``cfg.specified`` makes ``dycore.step`` call
    ``apply_state_lateral_boundaries``, which raises without an attachment,
    so a buffer cannot even take its warmup step until it holds tables of
    the right shape -- and the right shape is a property of the tiling.
    """
    from tilestream import spec as _spec

    px, py = _periodic_axes(cfg)
    return _spec.plan_tiles(int(cfg.nx), int(cfg.ny), int(decision.tile_nx),
                            int(decision.tile_ny), int(decision.halo),
                            periodic_x=px, periodic_y=py)


#: WHICH per-tile safety fold :func:`attach` arms.  There are two, from the
#: two branches that independently fixed the same defect, and they answer the
#: same question by different routes:
#:
#: ``False`` (the default)
#:     :class:`StreamedStability`, attached through ``TiledRun``'s generic
#:     ``observer`` hook.  This is the production arm: it is what
#:     ``runtime.integrate_prepared_case``, ``model.execute_experiment`` and
#:     both ``prepared_*_forecast`` routes read, through
#:     :func:`stability_observer`, and it keeps the OFF contract by returning
#:     ``dycore.stability_report`` ITSELF for a resident domain.
#: ``True``
#:     :class:`tilestream.health_fold.TileHealthFold`, through ``TiledRun``'s
#:     ``health_width``.  Gated by ``tilestream/test_obsfold.py``, which
#:     flips this flag.
#:
#: EXACTLY ONE is armed, never both: they are pure reductions so running both
#: would be correct, and it would also charge every streamed step for two
#: whole-domain folds to answer one question.
HEALTH_FOLD_DEFAULT = False


def attach(state, cfg, decision: StreamingDecision, *, tile_state_factory,
           geography=None, boundaries=None, boundary_tables=None,
           seam: str = "zeros",
           snapshot=None, inventory_fn=None, scalars=UNSET, nz=None,
           store=None, template=None,
           check_geography: bool = True,
           observe_stability: bool = True,
           stability_window: str = "interior",
           health_fold: bool | None = None,
           external_clock=DERIVE_CLOCK,
           tile_hook=None) -> StreamedDomain:
    """Move a prepared domain into a store and wrap it in a sweeper.

    ``state`` is the prepared, resident :class:`~gpuwm.core.state.DomainState`
    -- the one ArWen's preparation built and the one the health validator,
    the output writer and the restart writer already know about.  Its
    carriers are copied into the store; with ``decision.store == "host"``
    that store is pinned host RAM, which is the point.

    ``store`` is the other road, and it exists because the first one caps the
    domain at the size of the card rather than at the size of the machine.
    Pass an already-built full-domain carrier store -- as
    :func:`gpuwm.ingest.prepared_store.store_from_prepared_cache` builds one,
    slab by slab, with no domain-shaped device array anywhere -- and nothing
    is copied off a resident state because there is no resident state to copy
    off.  ``state`` may then be ``None``, and ``scalars`` must be given
    explicitly since there is nothing to read them from.  Everything
    downstream is unchanged: :class:`tilestream.driver.TiledRun` has always
    taken a bare ``{carrier key: array}`` dict and never touched a
    ``DomainState``.

    ``tile_state_factory(tile_cfg)`` must build a state with THE SAME PHYSICS
    SELECTORS the domain was prepared with, already warmed by one step.  Both
    conditions are load-bearing and both are checked rather than trusted: a
    buffer missing a scheme owns a different carrier set and the inventory
    comparison refuses it, and two carriers (Kain-Fritsch's ``cumulus/w0avg``
    above all) are allocated LAZILY on first use, so an unwarmed buffer is
    missing arrays the store holds.

    ``template`` is the state a STORE-DIRECT domain answers height-invariant
    questions with, and it is the second half of the ``store`` road rather
    than an option on it.  Two consumers here take a live state and neither
    can be handed a tile buffer: ``tilestream.output.frame_plan`` MEASURES
    which wrfout fields exist and where each comes from by building the real
    device frame and matching object identity (a transcribed table would drift
    the first time a scheme published a new diagnostic), and the restart
    header's vertical setup arrays are rebuilt from ``nz``/``hybrid_opt``/
    ``etac``/``p_top``.  Both answers are the same at every height and neither
    is the same on a tile whose config is not the domain's.  So a store built
    slab by slab keeps its LAST SLAB (``PreparedStore.template``) and that is
    what belongs here; without it a store-direct domain still integrates,
    checkpoints and folds its health, and refuses at the first history frame.

    ``geography`` is the DOMAIN's :func:`tilestream.driver.geography_inventory`
    -- it is INPUT, gathered per buffer and never scattered.

    The lateral forcing arrives either as ``boundaries`` (the DOMAIN's
    ``LateralBoundaries``, windowed here) or as ``boundary_tables`` (already
    windowed, one per tile, in :func:`tile_specs` order).  A route that has
    to build its tile buffers with an attachment already on them -- which
    every specified-BC route does, because a buffer cannot take its warmup
    step without one -- windows once with :func:`tile_boundary_tables` and
    passes the result both ways, rather than windowing twice.
    """
    from tilestream import driver as _driver
    from tilestream import gather as _gather
    from tilestream import physics_inventory as _physics

    if not decision.stream:
        raise StreamingRefused(
            "attach() called on a decision that does not stream; the run "
            "loop must bind gpuwm.core.dycore.step itself in that case, so "
            "that the resident path has no wrapper at all")
    if tile_hook is not None and (boundary_tables is not None
                                  or boundaries is not None):
        raise StreamingRefused(
            "attach() was handed BOTH lateral forcing tables and an "
            "explicit tile_hook; one domain has one lateral forcing "
            "mechanism, so one of these is wrong")

    # streaming_inventory, not carrier_inventory: restart's manifest is the
    # right answer to "what survives a file" and the WRONG answer to "what
    # survives the gap between two steps".  The consumer-owned UH tracking
    # windows are the difference -- see physics_inventory.STREAMING_ONLY_SLOTS
    # and tilestream/test_spawn_stream.py, which measures what excluding them
    # does to a spawn trigger.
    inventory_fn = inventory_fn or _physics.streaming_inventory
    nz = int(cfg.nz) if nz is None else int(nz)
    if state is None and store is None:
        raise StreamingRefused(
            "attach() was given neither a resident state to take a store "
            "from nor a prebuilt store to run on, so there is no domain "
            "here at all")
    if state is None and isinstance(scalars, _Unset):
        raise StreamingRefused(
            "attach() on a store-direct domain needs its carrier scalars "
            "passed explicitly: there is no resident state to read them "
            "from, and defaulting them to nothing would silently reset the "
            "physics clocks a store-built domain already carries")
    # ``UNSET`` means "take the domain's"; an explicit ``None`` means CARRY
    # NOTHING and is the gate's clock control.
    scalars = (_physics.carrier_scalars(state)
               if isinstance(scalars, _Unset) else scalars)

    if store is None:
        live = inventory_fn(state, None)
        if decision.store == "host":
            store = {name: _gather.pinned_copy(arr)
                     for name, arr in live.items()}
        else:
            store = live

    # ``tile_hook`` may arrive explicitly -- a NESTED domain's forcing is
    # not a windowable series, so its route wires a hook of its own
    # (gpuwm.core.nest_stream.make_nest_tile_hook) -- or be built here from
    # the tabulated tables.  Never both: a hook that swapped specified
    # tables under a nested attachment would serve the wrong subsystem's
    # forcing with no error anywhere.
    if boundary_tables is None and boundaries is not None:
        boundary_tables = tile_boundary_tables(
            boundaries, tile_specs(cfg, decision), seam=seam,
            snapshot=snapshot)
    if boundary_tables is not None:
        # ``domain_state=state``: the hook carries the domain's Davies clock
        # binding onto every buffer it converts (task #219; the docstring on
        # make_tile_hook has the measurement).  ``external_clock`` overrides
        # that derivation, and a STATE-LESS attachment must use it: there is
        # no ``state`` to derive from, so the default would silently select
        # the legacy elapsed-seconds recurrence.
        if state is None and external_clock is DERIVE_CLOCK:
            raise StreamingRefused(
                "attach() was given tabulated lateral boundaries and NO "
                "state, so there is nothing to derive the Davies clock "
                "binding from, and deriving nothing would put every tile "
                "buffer on the retired elapsed-seconds recurrence -- one "
                "timestep out of phase with the domain, for the whole "
                "forecast, with nothing in the log.  Pass external_clock= "
                "explicitly: the domain's DomainClock to bind it, or None "
                "to select the legacy compatibility semantics on purpose.")
        tile_hook = make_tile_hook(boundary_tables, domain_state=state,
                                   external_clock=external_clock)

    # The run loop's per-substep safety gate, folded per tile out of the
    # store.  Without it ``integrate_prepared_case``'s
    # ``stability_report(state, ...)`` reads a DomainState this sweep never
    # writes: healthy at t=0 and healthy forever, so a run that went
    # non-finite in the store completes and checkpoints.  ``spec_bdy_width``
    # is the same width the run loop passes.
    width = int(getattr(cfg, "spec_bdy_width", 0) or 0)
    health_fold = (HEALTH_FOLD_DEFAULT if health_fold is None
                   else bool(health_fold))
    # PER AXIS.  feat-open-lateral-bc measured that one flag for both axes
    # clamps the axis the kernels wrap; ``write_mode`` is the caller's,
    # because feat-route-wire found [tiles] write_mode was planned for
    # and then ignored here in favour of a hard-coded "ring".
    px, py = _periodic_axes(cfg)
    run = _driver.TiledRun(
        store, cfg, int(decision.tile_nx), int(decision.tile_ny),
        int(decision.halo), int(decision.nbuffers),
        periodic_x=px, periodic_y=py, write_mode=decision.write_mode,
        tile_state_factory=tile_state_factory,
        inventory_fn=inventory_fn, nz=nz, scalars=scalars,
        geography=geography, check_geography=check_geography,
        tile_hook=tile_hook,
        health_width=(width or None) if health_fold else None,
        # Installed unconditionally, including on a domain that will never
        # ask for reflectivity: the hook is a no-op unless a tile actually
        # stashed something, and making it conditional would mean deciding
        # at ATTACH time a question -- will refl_10cm_due ever be True? --
        # that only the model's history cadence can answer, and getting it
        # wrong stops the forecast at the first history frame.
        post_step_hook=refl_handoff_hook())
    # The domain's scratch arrays are now the STORE's, not the state's.  An
    # external whole-domain write -- the three nwp_diagnostics running-max
    # resets, and anything that follows them -- has to be able to find them;
    # see :func:`live_scratch` for the diagnostic that lost its meaning when
    # it could not.  Bound here rather than in ``StreamedDomain.__init__``
    # because attach is the moment the state stops owning the domain.
    # Skipped without a resident state, which is not a gap: the marker exists
    # so a whole-domain consumer holding the STATE can still find the
    # scratch arrays, and a store-direct domain has handed no such object to
    # anybody.  Its consumers read StreamedDomain.store directly.
    if state is not None:
        setattr(state, STREAMED_SCRATCH_ATTR,
                {name.split("/", 1)[1]: arr for name, arr in store.items()
                 if name.startswith("scratch/")})
    stability = None
    if observe_stability and not health_fold:
        # Attached AFTER the run exists, because the fold is sized from the
        # tile plan the constructor builds.  ``observe_stability=False`` is
        # the negative control for the whole fix: with it off the run loop
        # falls back to reducing over the prepared DomainState, which under a
        # host store is a t=0 corpse -- so the poison test that MUST raise
        # with the fold on MUST complete silently with it off.
        stability = StreamedStability(
            run, cfg,
            boundary_width=int(getattr(cfg, "spec_bdy_width", 0)) or None,
            window=stability_window)
        run.observer = stability.observe
    return StreamedDomain(run, decision, state=state, scalars=scalars,
                          host_store=(decision.store == "host"),
                          stability=stability,
                          # Carried, not just used: geography and the LBC
                          # tables are the two halves of the restart
                          # header's setup fingerprint that a tile buffer
                          # cannot supply, and above the card's ceiling
                          # there is no resident state to fall back to.
                          # See StreamedDomain.restart_setup.
                          geography=geography, boundaries=boundaries,
                          # Carried for the same reason and by the same rule:
                          # a store-direct domain's frame plan and vertical
                          # setup have to come from SOME live state, and the
                          # slab is the only one that is the domain's in every
                          # respect that does not vary with height.
                          template=template,
                          # The rule the store above was filled by, so
                          # refresh_state copies back the SAME key set --
                          # see that method for the carrier a narrower
                          # harvest silently left at t=0.
                          inventory_fn=inventory_fn)


def _is_periodic(cfg) -> bool:
    from tilestream.autoplan import is_periodic

    return bool(is_periodic(cfg))


# ---------------------------------------------------------------------------
# the builder every production route was missing
# ---------------------------------------------------------------------------
#
# :func:`make_stepper` needs a ``build(state, cfg, decision)``, and until this
# section existed no route had one: ``prepared_single_domain_forecast`` and
# ``prepared_domain_tree_forecast`` both called
# :func:`steppers_for_tree` with ``builders`` unset, so every forecast the CLI
# can launch answered ``[tiles] mode = "on"`` with ``StreamingRefused``.
# The mode was configurable and unreachable at the same time.
#
# The construction below is written ONCE, here, rather than per route,
# because none of it is route-specific: it is a function of the PREPARED
# DOMAIN NODE the route is holding when it reaches the seam.  What differs
# between routes -- where the initial condition came from, which proofs were
# checked, which writers are attached -- is all upstream of the state object,
# and the store is filled from that object.

#: What a tile buffer must reproduce about the domain and cannot derive.
#: ``STATE_SETUP_ARRAYS`` entries with a horizontal extent are GATHERED
#: (:func:`tilestream.driver.geography_inventory`); the purely vertical ones
#: and ``STATE_SETUP_SCALARS`` are not, on the argument that a tile rebuilds
#: them exactly from ``nz``/``hybrid_opt``/``etac``/``p_top`` and the base
#: sounding.  That argument holds for a domain built from an ANALYTIC
#: sounding and it does not hold here: a prepared domain's base state comes
#: out of ``initialize_real`` and a tile buffer has no way to reproduce it.
#: So they are IMPOSED from the domain rather than rebuilt and hoped over --
#: cheap (a few hundred bytes per buffer, once) and it removes the only
#: assumption in this path that a real initial condition would break.
_INHERITED_SETUP_SCALARS = ("mub", "p_top", "cf1", "cf2", "cf3", "cfn",
                            "cfn1")


def _domain_start_time(driver):
    """The UTC start the domain's own scheme adapters were built with.

    Taken from the adapters rather than from the experiment, because it is
    the adapters that consume it (solar zenith) and a buffer whose adapter
    disagreed with the domain's by so much as a second would compute a
    different zenith angle for the same column -- and only on the steps
    radiation is due, which is the hardest kind of difference to localize.
    """
    from datetime import datetime

    for attr in ("radiation_callable", "noahmp_geometry"):
        scheme = getattr(driver, attr, None)
        start = getattr(scheme, "start_time", None)
        if isinstance(start, datetime):
            return start
    return None


def _twin_rrtmg_legacy(scheme, cls, lat, lon):
    """Rebuild :class:`~gpuwm.core.rrtmg_legacy.RRTMGLegacyRadiation`.

    Its policy is not reachable by ``dataclasses.replace`` -- the adapter is
    a plain class, and a plain class whose constructor REQUIRES
    ``start_time``/``latitude_deg``/``longitude_deg`` at that -- but every
    one of its constructor arguments is recoverable from the instance, so
    the twin is exact rather than defaulted.  ``start_time`` is the domain's
    (radiation's solar geometry is a function of it, not of the tile);
    ``p_top``/``column_chunk``/``o3input`` are policy and part of the
    restart identity; ``ozone_parent`` is the child-domain o3rad routing,
    which is ``None`` on every domain that can stream today because a nest
    is refused upstream, and is carried anyway rather than assumed.

    Only the geography is replaced, which is the whole point: the tile's
    own ``latitude_deg``/``longitude_deg`` drive ``interp_ozone_to_latitudes``
    at construction, so a twin built at the domain's latitudes would carry
    the wrong ozone column for every tile but one.
    """
    return cls(scheme.start_time, lat, lon,
               p_top=scheme.p_top,
               column_chunk=scheme.column_chunk,
               o3input=scheme.o3input,
               ozone_parent=scheme._ozone_provider)


@dataclass(frozen=True)
class _TwinRecipe:
    """How to rebuild one non-dataclass adapter that requires arguments.

    ``reproduces`` names the constructor parameters ``build`` passes, and is
    CHECKED against the live signature rather than trusted: an adapter that
    grows a policy argument the recipe does not carry makes the recipe stale,
    and a stale recipe is exactly the silent failure the plain-object branch
    below refuses -- a buffer that allocates the same carriers, passes the
    inventory check, and integrates different physics.  ``volatile`` names
    the scalar attributes that are RUNTIME state rather than policy, so the
    equality audit does not mistake a counter for dropped configuration.
    """

    build: object
    reproduces: frozenset
    volatile: frozenset = frozenset()


#: Explicit constructor recipes, keyed ``"module:QualName"`` so registering
#: one imports nothing.
#:
#: The audit behind this table, over every adapter that reaches
#: :func:`_tile_scheme` (``physics.py`` assigns exactly two slots,
#: ``radiation_callable`` and ``cumulus_callable``):
#:
#: ``RRTMGPRadiation``          dataclass -- ``replace`` handles it
#: ``AnalyticClearSkyRadiation``dataclass -- ``replace`` handles it
#: ``KainFritsch``              plain, constructor asks for nothing --
#:                              the reconstruct-and-check branch handles it
#: ``RRTMGLegacyRadiation``     plain AND requires three arguments -- here
#:
#: So the recipe table has exactly one entry and the refusal was not a
#: category of broken schemes, it was this one.  The mechanism is a table
#: rather than a special case because the next such adapter should cost a
#: recipe, not another rewrite of the dispatch.
_TWIN_RECIPES = {
    "gpuwm.core.rrtmg_legacy:RRTMGLegacyRadiation": _TwinRecipe(
        build=_twin_rrtmg_legacy,
        reproduces=frozenset({"start_time", "latitude_deg", "longitude_deg",
                              "p_top", "column_chunk", "ozone_parent",
                              "o3input"}),
        # WRF's radiation call counter; the domain's adapter has stepped
        # when a buffer is built mid-run, a fresh twin has not, and that
        # difference is not dropped policy.
        volatile=frozenset({"update_count"}),
    ),
}


#: What :func:`twin_support` answers, in the order :func:`_tile_scheme`
#: tries them.
TWIN_BY_REPLACE = "dataclass-replace"
TWIN_BY_RECIPE = "recipe"
TWIN_BY_RECONSTRUCTION = "empty-constructor"


def twin_support(cls) -> str | None:
    """How a per-buffer twin of ``cls`` would be built, or ``None``.

    The CLASS-level half of :func:`_tile_scheme`'s dispatch, factored out so
    that "can this scheme stream?" is answerable without a device, without a
    domain and without constructing the adapter -- which for legacy RRTMG
    means CUDA compilation and for RRTMGP means loading gas tables.  That is
    what lets the scheme audit run in the CPU battery, where a scheme added
    without a tile constructor goes red at development time instead of an
    hour into somebody's forecast.

    ``_tile_scheme`` dispatches on this, so the audit cannot drift from the
    behaviour it audits.  It answers the SHAPE question only; the per-INSTANCE
    checks -- that a reconstruction dropped no policy, that a recipe is not
    stale -- still run at twin time against the domain's actual object.
    """
    import dataclasses
    import inspect

    if dataclasses.is_dataclass(cls):
        return TWIN_BY_REPLACE
    if f"{cls.__module__}:{cls.__qualname__}" in _TWIN_RECIPES:
        return TWIN_BY_RECIPE
    empty = inspect.Parameter.empty
    required = [p.name for p in inspect.signature(cls).parameters.values()
                if p.default is empty
                and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    return None if required else TWIN_BY_RECONSTRUCTION


def _assert_twin_reproduced_policy(scheme, twin, volatile):
    """Every scalar the twin and the domain's adapter disagree on is a bug.

    The same audit the reconstruct-and-check branch applies to plain
    adapters, reused so a recipe is held to the standard the branch it
    bypasses would have enforced.
    """
    for key, value in vars(twin).items():
        if key in volatile or not isinstance(value, (str, bool, int, float)):
            continue
        was = getattr(scheme, key, value)
        if isinstance(was, (str, bool, int, float)) and was != value:
            raise StreamingRefused(
                f"the per-buffer twin recipe for {type(scheme).__name__} "
                f"produced {key}={value!r} where the domain's adapter "
                f"carries {was!r}, so the recipe drops policy; a buffer "
                "built with it would own the same carriers and integrate "
                "different physics")


def _tile_scheme(scheme, lat, lon):
    """A FRESH twin of one scheme adapter, at a tile's extents.

    Fresh is the load-bearing word, and it is a bug this function was
    written to fix rather than a precaution.  Handing every buffer the
    DOMAIN's adapter object looks harmless -- Kain-Fritsch's constructor
    takes no arguments and holds no configuration -- and is not: ``w0avg``
    is a CARRIER (``cumulus/w0avg``), so all the buffers and the domain
    would share one array while the transport gathered and scattered it as
    if each buffer owned its own.  ``KainFritsch`` also caches
    ``_history_state``, a reference to the last state it served, and
    ``tilestream.driver.assert_geography_gathered`` walks it: with the
    object shared, buffer 0's adapter pointed at buffer 1's state and the
    check reported buffer 1's radiation lat/lon as ungatherable geography.
    That is what caught it, and it is why ``check_geography`` is left ON.

    Three shapes of adapter, and the rule is the same for all three --
    reproduce the POLICY exactly, replace only the geography:

    *dataclasses* (RRTMGP, the analytic scheme, Noah-MP's geometry) carry
    their policy as init fields -- ``column_chunk``,
    ``trace_gas_overrides``, ``p_top`` -- and their per-column geography as
    ``latitude_deg``/``longitude_deg``.  ``dataclasses.replace`` gives a new
    instance that differs in exactly the two fields the gather is going to
    overwrite anyway.

    *plain objects with an empty constructor* (Kain-Fritsch) are
    reconstructed by calling their class.  That is only correct if the class
    asks for nothing and configures nothing, so both are CHECKED: a
    constructor with a required parameter gets no default reconstruction,
    and neither does any scalar attribute on which the fresh instance and
    the domain's disagree -- which is what a policy the reconstruction
    dropped would look like.  An adapter reconstructed with default policy
    allocates the same carriers and passes the inventory check, so nothing
    downstream would catch it.

    *plain objects that require arguments* (legacy RRTMG) get an explicit
    recipe from :data:`_TWIN_RECIPES`, audited two ways: the recipe must
    name every constructor parameter, and the twin it returns must agree
    with the domain's adapter on every non-volatile scalar.  This branch was
    once the refusal itself -- legacy RRTMG is the DEFAULT radiation of the
    shipped physics suite, so streaming refused the default suite outright,
    and the docstring here asserted legacy RRTMG was a dataclass, which is
    part of how it went unnoticed.  The refusal below is deliberately kept
    for adapters with no recipe: sharing the domain's object is still not
    the fallback.
    """
    import dataclasses
    import inspect

    if scheme is None:
        return None
    if dataclasses.is_dataclass(scheme):
        names = {f.name for f in dataclasses.fields(scheme) if f.init}
        if {"latitude_deg", "longitude_deg"} <= names:
            return dataclasses.replace(scheme, latitude_deg=lat,
                                       longitude_deg=lon)
        return dataclasses.replace(scheme)

    cls = type(scheme)
    empty = inspect.Parameter.empty
    support = twin_support(cls)
    recipe = (_TWIN_RECIPES.get(f"{cls.__module__}:{cls.__qualname__}")
              if support == TWIN_BY_RECIPE else None)
    if recipe is not None:
        declared = {p.name for p in inspect.signature(cls).parameters.values()
                    if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
        stale = declared - recipe.reproduces
        if stale:
            raise StreamingRefused(
                f"the per-buffer twin recipe for {cls.__name__} does not "
                f"reproduce its constructor parameter(s) {sorted(stale)}, so "
                "it is stale against the adapter; a buffer built from it "
                "would take those at their defaults and integrate different "
                "physics from the domain")
        twin = recipe.build(scheme, cls, lat, lon)
        _assert_twin_reproduced_policy(scheme, twin, recipe.volatile)
        return twin
    required = [p.name for p in inspect.signature(cls).parameters.values()
                if p.default is empty
                and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    if required:
        raise StreamingRefused(
            f"the domain's {cls.__name__} is not a dataclass, its "
            f"constructor requires {required}, and no entry in "
            "gpuwm.core.streaming._TWIN_RECIPES says how to rebuild it, so "
            "a per-buffer twin of it cannot be built.  Sharing the domain's "
            "adapter is not the fallback: its carriers would be shared with "
            "every tile buffer at once.  Give the adapter a _TWIN_RECIPES "
            "entry that reproduces its policy and takes the tile's "
            "geography.")
    fresh = cls()
    for key, value in vars(fresh).items():
        if not isinstance(value, (str, bool, int, float)):
            continue
        was = getattr(scheme, key, value)
        if isinstance(was, (str, bool, int, float)) and was != value:
            raise StreamingRefused(
                f"the domain's {cls.__name__} carries {key}={was!r} where a "
                f"freshly constructed one has {value!r}, so it holds policy "
                "this cannot reproduce; a buffer built with the default "
                "would own the same carriers and integrate different "
                "physics")
    return fresh


def domain_vertical_coord(state, cfg):
    """The DOMAIN's own eta coordinate, for a tile buffer to be built on.

    THE DEFECT THIS EXISTS FOR.  ``harness.make_physics_state`` rebuilds the
    vertical coordinate when its caller does not supply one, and what it
    rebuilds is ``make_vertical_coord(nz, hybrid_opt, etac)`` -- the DEFAULT
    stretch.  A real case runs its own explicit eta table.  Its own docstring
    says a real case must supply ``coord`` and that arrays built from the
    wrong table "are the right shape, the right dtype, and wrong, and nothing
    downstream notices"; the prepared factory then did not supply it.

    :func:`_impose_domain_setup` half-hid the consequence.  It copies the
    domain's PURELY VERTICAL setup (``znu``, ``znw``, ``dnw``, ``c1h..c4f``,
    ``p_top``, ``mub``, ``cf1..cfn1``) onto the buffer, so those became the
    domain's -- but ``thb``/``pb``/``alb``/``phb`` are 3-D on any domain with
    terrain and are gathered per tile, so they stayed the buffer's, built by
    ``make_base_state`` against the DEFAULT stretch.  The buffer's warm-up
    step therefore evaluated a WK82 theta column from one atmosphere against
    a pressure column from another.

    MEASURED, 550x550x49 at 3 km through the product front door: the warm-up
    handed rte-rrtmgp a temperature profile spanning 120.3 K to 407.3 K and
    the gas-table validator refused it -- correctly; ``[160, 355] K`` is the
    tables' own range and stays.  120.314407 K is 300 K, the sounding's
    surface theta, taken at 41 hPa: a surface level's theta at the model
    top's pressure, which is the signature of two mismatched coordinates and
    not of uninitialised memory.  Legacy RRTMG has no such validator, so on
    that suite the same wrong warm-up ran silently instead of stopping.

    Built from ``state.znw`` rather than from the config's ``eta_levels`` so
    it is the table the domain IS on, not the one it was asked for.  A table
    this refuses to rebuild is refused loudly: falling back to the default
    stretch is the bug.
    """
    import numpy as np

    from gpuwm.core.grid import make_vertical_coord

    znw = getattr(state, "znw", None)
    if znw is None:
        raise StreamingRefused(
            "the domain state carries no znw, so a tile buffer cannot be "
            "built on the domain's vertical coordinate; it would silently "
            "get the default stretch and a base state for a different "
            "atmosphere")
    eta = np.asarray(_as_host_array(znw), dtype=np.float64)
    try:
        return make_vertical_coord(
            int(cfg.nz), eta_levels=eta,
            hybrid_opt=int(getattr(cfg, "hybrid_opt", 0)),
            etac=float(getattr(cfg, "etac", 0.2)))
    except ValueError as exc:
        raise StreamingRefused(
            f"the domain's eta table cannot be rebuilt for a tile buffer "
            f"({exc}); a buffer built on the default stretch instead would "
            "carry a base state for a different atmosphere, which is how "
            "this failed before it was refused") from exc


def _impose_domain_setup(tile, state) -> int:
    """Give the buffer the DOMAIN's vertical coordinate and setup scalars.

    See :data:`_INHERITED_SETUP_SCALARS`.  The horizontally-varying setup
    arrays are gathered per tile and are deliberately NOT touched here.

    Returns how many entries this actually CHANGED, which is the number that
    says whether the imposition is doing work or is insurance.  On a domain
    built from an analytic sounding -- every gate in ``tilestream`` -- the
    buffer rebuilds the vertical coordinate exactly and the answer is zero.
    On a domain out of ``initialize_real`` it will not be, and there is no
    test in this checkout that can prove that half, because there is no
    prepared cache in it.  Reported rather than assumed for that reason.
    """
    import numpy as np

    from gpuwm.state_serialization_contract import STATE_SETUP_ARRAYS

    changed = 0
    for name in STATE_SETUP_ARRAYS:
        src = getattr(state, name, None)
        dst = getattr(tile, name, None)
        if src is None or dst is None:
            continue
        # ndim < 2 is exactly the complement of geography_inventory's filter:
        # what the gather does not carry, the buffer inherits here.
        if getattr(src, "ndim", 0) >= 2:
            continue
        if getattr(src, "shape", None) != getattr(dst, "shape", None):
            raise StreamingRefused(
                f"the domain's vertical setup array {name!r} is "
                f"{getattr(src, 'shape', None)} and the tile buffer's is "
                f"{getattr(dst, 'shape', None)}; the buffer cannot inherit it")
        if not bool(np.asarray(_as_host_array(dst) == _as_host_array(src)
                               ).all()):
            changed += 1
        dst[...] = src
    for name in _INHERITED_SETUP_SCALARS:
        if not hasattr(state, name) or not hasattr(tile, name):
            continue
        value = getattr(state, name)
        if isinstance(value, (float, int, np.floating, np.integer)):
            if getattr(tile, name) != value:
                changed += 1
            setattr(tile, name, value)
    return changed


def _as_host_array(value):
    """A host copy of a device or host array.

    cupy is imported only if the value could be a device array: a host-only
    caller (and every CPU test) must not need a CUDA build to read a table
    that is already in host memory.
    """
    import numpy as np

    if type(value).__module__.split(".")[0] == "cupy":
        import cupy as cp

        return cp.asnumpy(value)
    return np.asarray(value)


def prepared_tile_state_factory(state, cfg, *, tables0=None, seed: int = 4242,
                                warmup: int = 0):
    """``tile_state_factory`` for a domain ArWen's preparation already built.

    :func:`attach` states the contract: the factory must build a state with
    THE SAME PHYSICS SELECTORS the domain was prepared with, carrying the
    same inventory.  ``warmup`` defaults to 0 because that inventory is now
    reached by :func:`prime_lazy_carriers` rather than by throwing away a
    step -- see there for the temperature field the thrown-away step used to
    hand the radiation.  Four things go into satisfying the contract from a
    prepared node, and each of them has a specific failure it prevents:

    *the selectors come from the config, so they are free.*  ``tile_cfg`` is
    the domain's own ``RunConfig`` with only ``nx``/``ny`` replaced
    (:func:`tilestream.harness.tile_config`), so every ``*_physics`` switch,
    every cadence and every damping constant is the domain's by construction
    rather than by being copied correctly.

    *the buffer is built on NEUTRAL geography, not on the tile's own.*
    :func:`tilestream.harness.neutral_geography` is a POISON fill -- identity
    map factors, zero Coriolis, flat terrain, a latitude nowhere near the
    domain's -- so an array the gather fails to write is obviously wrong
    rather than plausibly wrong.  It still has to be a full ``(ny, nx)``
    terrain field, because ``cfg.terrain_opt`` decides whether ``thb/pb/alb/
    phb`` are 1-D or 3-D and a buffer built flat against a domain with
    terrain has the wrong SHAPES.

    *the scheme adapters are CLONED, not reconstructed.*  See
    :func:`_clone_scheme_at`.

    *the vertical coordinate is inherited.*  See
    :data:`_INHERITED_SETUP_SCALARS`.

    ``tables0`` is tile 0's windowed lateral forcing.  It is attached before
    the warmup step and replaced by the ``tile_hook`` before the buffer ever
    serves a tile: with ``cfg.specified=True``, ``dycore.step`` calls
    ``apply_state_lateral_boundaries``, which raises without an attachment,
    so a buffer cannot take its warmup step at all until it holds tables of
    the right shape -- and the right shape is a property of the tiling.

    The warmup step is not optional either.  Two carriers are allocated
    LAZILY on first use, Kain-Fritsch's ``cumulus/w0avg`` above all, so a
    buffer that has never stepped is missing arrays the store holds and the
    inventory comparison refuses it.
    """
    from gpuwm.ingest.lateral_bc import attach_lateral_boundaries
    from tilestream import harness as _harness

    driver = getattr(state, "physics", None)
    start_time = None if driver is None else _domain_start_time(driver)
    # ONCE, not per buffer: the domain's eta table does not vary by tile, and
    # rebuilding it is where a buffer stops being a model of the domain.  See
    # :func:`domain_vertical_coord`.
    coord = domain_vertical_coord(state, cfg)

    def make(tile_cfg):
        import numpy as np

        geo = _harness.neutral_geography(tile_cfg)
        extra: dict = {}
        if driver is not None:
            lat = np.asarray(geo.lat, dtype=np.float64)
            lon = np.asarray(geo.lon, dtype=np.float64)
            # A FRESH twin per buffer for each -- never the domain's own
            # object, which would share its carriers with every buffer at
            # once.  See :func:`_tile_scheme`.
            for key, attr in (("radiation", "radiation_callable"),
                              ("cumulus", "cumulus_callable")):
                twin = _tile_scheme(getattr(driver, attr, None), lat, lon)
                if twin is not None:
                    extra[key] = twin
        tile, _drv = _harness.make_physics_state(
            tile_cfg, seed, geography=geo, start_time=start_time,
            coord=coord, **extra)
        make.setup_entries_imposed = _impose_domain_setup(tile, state)
        # The buffer's own lazily-allocated carriers, so its inventory matches
        # the store's WITHOUT integrating anything.  See prime_lazy_carriers:
        # the warm-up step this replaces stepped an analytic sounding against
        # the domain's real lateral forcing.
        make.primed = prime_lazy_carriers(tile, tile_cfg)
        if tables0 is not None:
            attach_lateral_boundaries(tile, tables0)
        if warmup:
            _harness.run_steps(tile, tile_cfg, int(warmup))
        return tile

    #: How many vertical-setup entries the last buffer had to INHERIT rather
    #: than rebuild.  Zero on an analytic-sounding domain, which is every
    #: gate in this checkout; nonzero is what a real prepared cache would
    #: produce, and there is none here to prove it with.
    make.setup_entries_imposed = 0
    return make


def prepared_domain_builder(node, *, seam: str = "zeros",
                            tile_seed: int = 4242, warmup: int = 0,
                            check_geography: bool = True):
    """The ``build`` :func:`make_stepper` needs, for one PREPARED domain node.

    ``node`` is a :class:`gpuwm.core.model.DomainNode` the route has already
    prepared: its ``state`` is the resident :class:`~gpuwm.core.state
    .DomainState` the health validator, the output writer and the restart
    writer already know about, its ``cfg.run`` is the config the executor
    will step it with, and -- for a specified-boundary domain -- its
    ``state.lateral_boundaries`` is the domain's own forcing series.

    Three things are read off the node and nothing is invented:

    ``geography``
        :func:`tilestream.driver.geography_store` of the domain state, in
        pinned host RAM when the store is.  It is INPUT: gathered when a
        buffer starts serving a different tile, never scattered back.

    ``boundary_tables``
        the domain's ``LateralBoundaries`` windowed once per tile.  A tile's
        TRUE domain edges take the domain's own tables sliced along the
        tangential axis; its interior seams take inert ones.

    ``tile_state_factory``
        :func:`prepared_tile_state_factory`.

    A CHILD domain streams through its own forcing door.  A nest does not
    hold a ``LateralBoundaries``: its lateral forcing is rebuilt from the
    parent by ``NestCoupler.force`` every parent step into full-perimeter
    rolling device tables, so instead of windowing a tabulated series once,
    the builder wires :func:`gpuwm.core.nest_stream.make_nest_tile_hook` --
    per-buffer packed table windows, re-filled at kernel-launch time
    whenever the rolling generation moves.  A refusal used to stand here
    asserting the per-tile windowing of nest tables was unposed; the
    streamed-child corridor posed and gated it
    (``tilestream/test_streamed_child.py``).
    """
    def build(state, cfg, decision):
        from gpuwm.ingest.lateral_bc import LateralBoundaries
        from tilestream import driver as _driver

        # Inside build, not beside it: build runs only for a domain whose
        # decision STREAMS, so a tree that configures mode = "auto" and
        # whose nests fit resident pays nothing for having nests.
        nested = getattr(node, "parent", None) is not None

        geography = _driver.geography_store(
            state, host=True if decision.store == "host" else None)

        boundaries = getattr(state, "lateral_boundaries", None)
        tables = None
        nest_hook = None
        if nested:
            if isinstance(boundaries, LateralBoundaries):
                raise StreamingRefused(
                    f"grid {int(node.cfg.grid_id)} is a NEST and also "
                    "carries a tabulated LateralBoundaries series.  A "
                    "nested domain's forcing is rebuilt from its parent "
                    "every parent step (NestCoupler.force), so these two "
                    "attachments contradict each other and one of them "
                    "would be silently ignored.")
            from gpuwm.core.nest_stream import make_nest_tile_hook

            nest_hook = make_nest_tile_hook(node)
        elif boundaries is not None:
            if not isinstance(boundaries, LateralBoundaries):
                raise StreamingRefused(
                    "the domain carries lateral forcing of type "
                    f"{type(boundaries).__name__}, which is not the "
                    "tabulated LateralBoundaries series tile_boundary_tables "
                    "windows")
            tables = tile_boundary_tables(
                boundaries, tile_specs(cfg, decision), seam=seam)
        elif bool(getattr(cfg, "specified", False)):
            raise StreamingRefused(
                "cfg.specified is set but the domain state carries no "
                "attached LateralBoundaries, so no per-tile forcing can be "
                "windowed and no buffer could take its warmup step")

        # REFL_10CM before the store is sized, not when the first frame is
        # due.  The slot is REBUILT scratch, so it is absent from the carrier
        # manifest by construction and the store would have nowhere to join
        # the tiles' windows; the sweep then refuses the first due frame,
        # which on this route is an hour into a forecast that was otherwise
        # healthy.  Primed on the domain here and on every buffer in the
        # factory, and carried by refl_inventory on all three.
        #
        # The eddy-viscosity producers ride the same priming for the same
        # reason (prime_hmix_k_diag, and only under cfg.hmix_k_diag).
        prime_lazy_carriers(state, cfg)
        factory = prepared_tile_state_factory(
            state, cfg, tables0=(None if tables is None else tables[0]),
            seed=tile_seed, warmup=warmup)
        # THE FRAME'S FIELD SET, not just its trajectory.  refl_inventory
        # alone carries the reflectivity slot and leaves every OTHER
        # output-only driver diagnostic behind, so a streamed frame
        # published 73 of the resident run's 74 variables -- OLR silently
        # absent, validity PASS, health status_bits 0 (MEASURED at the
        # 2.2.0 cut, 438x350x49 through `gpuwm go`).  diagnostic_inventory
        # carries the rest; StoreFrame refuses a plan that still cannot
        # publish one, so this can no longer fail quietly.
        #
        # streamed_store_inventory, not the expression spelled out here: the
        # store-direct road builds its store with the same call, and the two
        # inventories are compared key for key before the first step.
        streamed = attach(state, cfg, decision, tile_state_factory=factory,
                          geography=geography, boundary_tables=tables,
                          check_geography=check_geography,
                          tile_hook=nest_hook,
                          inventory_fn=streamed_store_inventory())
        if decision.store == "host":
            import warnings

            # Warned in the same register as the short halo, and for the
            # same reason: what follows is silent, plausible and wrong.
            # MEASURED through execute_experiment at 192x144x49, 120 steps,
            # 155 carriers: 0 of 155 carriers on the DomainState moved,
            # against 118 of 155 for the identical resident run.  The
            # forecast is correct and it is in the STORE; the state object
            # every route reads for history frames, the stability gate, the
            # health validators and canonical_state_digest still holds the
            # initial condition.  A route that publishes output from a
            # streamed domain must call
            # ``StreamedDomain.refresh_state(node.state)`` on the history
            # cadence and once more before its final read -- which no route
            # does yet.
            warnings.warn(
                "[tiles] store='host' puts the forecast in a pinned "
                "host store; the DomainState this route holds is the "
                "snapshot that filled it and will NOT advance.  Output, the "
                "stability gate, the health validators and the final "
                "canonical digest all read that state, so they will report "
                "the INITIAL CONDITION for the whole run unless the route "
                "calls StreamedDomain.refresh_state(node.state) before "
                "reading it.  The integration itself is unaffected and is "
                "bit-exact against a resident run.",
                RuntimeWarning, stacklevel=2)
        return streamed

    return build


@dataclass(frozen=True)
class _StandaloneNodeCfg:
    grid_id: int


@dataclass(frozen=True)
class _StandaloneNode:
    """The two facts :func:`prepared_domain_builder` reads off a tree node.

    A ``DomainNode`` is a position in a tree, and the builder asks it exactly
    two questions: is this domain a NEST (``parent``), and what is its grid id
    (for one refusal message).  A domain that is its own root answers both
    without a tree, and saying so in six lines is honest, whereas building a
    one-node tree to satisfy an attribute lookup would not be.
    """

    cfg: _StandaloneNodeCfg
    parent: None = None


def standalone_domain_builder(*, grid_id: int, **kwargs):
    """The ``build`` :func:`make_stepper` needs for a ROOT domain with no tree.

    :func:`prepared_domain_builder` is the whole builder and this adds nothing
    to it -- same store fill, same tile buffers with the domain's own physics
    selectors, same per-tile windowing of the domain's own
    ``LateralBoundaries``.  What it adds is a way to ASK for it from a route
    that never builds a :class:`gpuwm.core.model.DomainModel`.

    :mod:`gpuwm.offline_child_run` is that route, and it is a specified-
    boundary root by construction: the offline child stamps ``parent_id = 0``
    on its own ``DomainTicks`` precisely because no parent domain is resident,
    and its forcing is the tabulated series ``build_offline_lateral_boundaries``
    prepared out of the parent archive.  So it takes the builder's non-nested
    branch, which is the same branch ``gpuwm go``'s d01 takes -- the arm the
    bit-exactness gate (``tilestream/test_join.py``) and the 2.2.2 planner-driven
    GPU leg both cover.
    """
    return prepared_domain_builder(
        _StandaloneNode(cfg=_StandaloneNodeCfg(grid_id=int(grid_id))),
        **kwargs)


def store_domain_builder(bundle, *, clock=DERIVE_CLOCK, seam: str = "zeros",
                         tile_seed: int = 4242, warmup: int = 0,
                         check_geography: bool = True):
    """The ``build`` :func:`make_stepper` needs, for a STORE-DIRECT domain.

    :func:`prepared_domain_builder`'s counterpart for a domain that was never
    resident.  ``bundle`` is a
    :class:`gpuwm.ingest.prepared_store.PreparedStore`: its carriers and its
    geography are already full-domain pinned host arrays, built slab by slab
    off the prepared cache, and its lateral forcing is the domain's own
    series held on the host.

    Everything the resident builder reads off ``node.state`` is read off the
    bundle instead, and the substitution is exact rather than approximate in
    all three places:

    ``geography``
        the bundle's, which was scattered from the same per-slab
        ``geography_inventory`` the resident road gathers from a whole
        domain.

    ``boundary_tables``
        windowed from the bundle's ``LateralBoundaries`` by the same
        :func:`tile_boundary_tables`.  The tables are never attached to a
        domain-shaped state -- there is none -- so the only lateral forcing
        that reaches the card is one tile's edge.

    ``tile_state_factory``
        :func:`prepared_tile_state_factory` on the bundle's SLAB-HEIGHT
        template.  The factory reads exactly three things from it -- the
        scheme adapters to twin, the vertical coordinate, and the
        ``ndim < 2`` setup arrays -- and none of the three varies with
        height, which is why a slab may stand in for the domain.  The
        horizontally-varying setup is gathered per tile from ``geography``,
        as it always was.

    The same slab is ALSO handed to ``attach(template=...)`` and kept on the
    domain, because two whole-domain readers downstream take a live state for
    the same height-invariant reasons the factory does: the history frame's
    plan is measured against one (``tilestream.output.store_frame_plan``) and
    the restart header's vertical setup arrays are rebuilt from one
    (:meth:`StreamedDomain.restart_setup`).  Passing it is what makes a
    store-direct forecast able to publish wrfout at all -- without it the run
    integrates and checkpoints correctly and refuses at its first history
    frame, which is an hour into an otherwise healthy forecast.

    ``prime_lazy_carriers`` is not called on a domain here because there is
    no domain-shaped object to call it on; it ran on every SLAB as the store
    was built, which is the same set of keys reached the same way.  The
    buffers are primed in the factory as before, and ``TiledRun``'s
    inventory comparison still refuses any disagreement between a buffer and
    the store -- the check that would catch it if this reasoning were wrong.
    """
    def build(state, cfg, decision):
        from gpuwm.ingest.lateral_bc import LateralBoundaries

        if getattr(cfg, "grid_id", 1) and state is not None \
                and getattr(state, "parent", None) is not None:
            raise StreamingRefused(
                "a store-direct domain cannot be a NEST: a nested domain's "
                "lateral forcing is rebuilt from its parent every parent "
                "step rather than tabulated, so there is nothing for "
                "tile_boundary_tables to window")
        boundaries = bundle.boundaries
        tables = None
        if boundaries is not None:
            if not isinstance(boundaries, LateralBoundaries):
                raise StreamingRefused(
                    "the prepared store carries lateral forcing of type "
                    f"{type(boundaries).__name__}, which is not the "
                    "tabulated LateralBoundaries series tile_boundary_tables "
                    "windows")
            tables = tile_boundary_tables(
                boundaries, tile_specs(cfg, decision), seam=seam)
        elif bool(getattr(cfg, "specified", False)):
            raise StreamingRefused(
                "cfg.specified is set but the prepared store carries no "
                "LateralBoundaries, so no per-tile forcing can be windowed "
                "and no buffer could take its warmup step")

        factory = prepared_tile_state_factory(
            bundle.template, cfg,
            tables0=(None if tables is None else tables[0]),
            seed=tile_seed, warmup=warmup)
        # The SAME call ``store_from_prepared_cache`` defaults to when it
        # harvests each slab, so the store handed in here and the buffers
        # built from it carry the same keys BY CONSTRUCTION rather than by
        # two literals happening to agree.
        # THE CLOCK IS PASSED, NEVER DERIVED.  ``attach(None, ...)`` has no
        # state to read a binding off, so the derivation this road used to
        # inherit returned None on every buffer and latched -- see
        # make_tile_hook.  Required here rather than defaulted: a
        # store-direct domain is a specified-boundary root, its route holds
        # the DomainClock, and the failure of guessing is a silently
        # one-step-late forecast.
        if tables is not None and clock is DERIVE_CLOCK:
            raise StreamingRefused(
                "store_domain_builder was given a prepared store with "
                "tabulated lateral boundaries and no clock=.  A store-direct "
                "domain is never resident, so there is no DomainState to "
                "derive the Davies binding from and every tile buffer would "
                "take the retired elapsed-seconds recurrence -- the forecast "
                "would consume its lateral boundaries ONE TIMESTEP LATE, for "
                "its whole length, and look healthy doing it.  Pass the "
                "node's clock, or clock=None to ask for the legacy "
                "semantics deliberately.")
        return attach(None, cfg, decision, tile_state_factory=factory,
                      store=bundle.store, scalars=bundle.scalars,
                      geography=bundle.geography, boundary_tables=tables,
                      boundaries=boundaries, template=bundle.template,
                      check_geography=check_geography,
                      external_clock=(None if clock is DERIVE_CLOCK
                                      else clock),
                      inventory_fn=streamed_store_inventory())

    return build


def options_for_domain(domain_cfg, tree_options: "StreamingOptions | None"
                       ) -> "StreamingOptions":
    """The ``[tiles]`` table that governs ONE domain.

    A domain carrying its own ``tiles = {...}`` table takes it; every other
    domain takes the tree-wide ``[tiles]``.  The override REPLACES rather
    than merges, because a half-inherited tiling -- mode from the tree,
    store from the domain -- is a configuration nobody can read off the
    file.

    This is the whole of the per-domain surface as far as the engine is
    concerned: everything downstream keeps asking for "the options of this
    domain" and gets one :class:`StreamingOptions`, so the roads that
    already work on a tree-wide table work per domain with no second
    vocabulary.
    """
    own = getattr(domain_cfg, "tiles", None)
    if own is not None:
        return own
    return OFF if tree_options is None else tree_options


def tree_streams_anywhere(model, options: "StreamingOptions | None") -> bool:
    """Whether ANY domain of ``model`` has ``[tiles]`` enabled.

    The short-circuit both tree entry points open with.  It cannot be
    ``options.enabled`` any more: a tree whose ``[tiles]`` table is absent
    entirely, but one of whose ``[[domain]]`` rows carries ``tiles = {mode
    = "auto"}``, IS a configured run, and returning early on the tree-wide
    table would run it resident in silence -- the one failure this module
    exists to remove.
    """
    if options is not None and options.enabled:
        return True
    return any(options_for_domain(node.cfg, options).enabled
               for node in model.walk_parent_first())


def builders_for_tree(model, options: StreamingOptions | None = None, **kwargs
                      ) -> dict:
    """``{grid_id: build}`` for a whole tree; the routes' half of the seam.

    Paired with :func:`steppers_for_tree` at every route's single call site::

        steppers = streaming.steppers_for_tree(
            model, exp.tiles,
            builders=streaming.builders_for_tree(model, exp.tiles))

    With ``[tiles]`` absent this returns ``{}`` and imports nothing, so
    the resident path is byte-for-byte the path it was before the mode
    existed -- the same contract :func:`steppers_for_tree` keeps, and for the
    same reason.
    """
    options = OFF if options is None else options
    if not tree_streams_anywhere(model, options):
        return {}
    # Every domain whose OWN options are enabled gets a builder.  A domain
    # the tree-wide table leaves off, and that says nothing itself, gets
    # none -- it can never reach a STREAM decision, and handing it a
    # builder would only make an unreachable road look wired.
    return {int(node.cfg.grid_id): prepared_domain_builder(node, **kwargs)
            for node in model.walk_parent_first()
            if options_for_domain(node.cfg, options).enabled}


def _periodic_axes(cfg) -> tuple[bool, bool]:
    """``(periodic_x, periodic_y)`` -- the dycore's own predicates negated.

    Not ``_is_periodic`` twice.  ``open_x`` without ``open_y`` is a domain
    the model steps non-periodically in x and PERIODICALLY in y, and a plan
    built from one flag clamps the axis the kernels wrap: measured at
    256x192x49, tile 32x32, halo 16, ONE dry step, all nine carriers differ
    and the difference is exactly the two y-boundary tile rows.
    """
    from tilestream.autoplan import is_periodic_x, is_periodic_y

    return bool(is_periodic_x(cfg)), bool(is_periodic_y(cfg))


def refuse_unrouted_streaming(exp, route: str, *,
                              consults_the_seam: bool = True) -> None:
    """Refuse ``[tiles] mode = "on"`` on a route that wires no builder.

    Same governance as :func:`gpuwm.experiment.refuse_unrouted_spawn`, called
    from the same place for the same reason: a declared capability is honored
    or refused, never discovered late.

    :func:`make_stepper` already refuses a streamed domain whose route wired
    no ``build`` -- but it is called at the END of a route, after the fetch,
    after preprocessing, after every prepared cache has been restored and
    every ``DomainState`` has been built on the card.  MEASURED on the
    prepared domain-tree runner: ``load_experiment`` is at line 839 and the
    refusal was at 1637 -- 798 lines and one full resident tree construction
    downstream of the load that could have produced it.  A user who turned
    streaming on because their domain does not fit therefore pays the entire
    preparation and then meets an out-of-memory death at the allocation the
    mode existed to avoid -- and, if it somehow fits, a refusal whose message
    they will read as something the run did rather than something the config
    always was.

    WHO STILL CALLS THIS, and who stopped.  ``gpuwm run``
    (:func:`gpuwm.runtime.run_experiment`) does, with
    ``consults_the_seam=False``, because it reads ``exp.tiles`` at no point.
    The two prepared routes and ``gpuwm go`` do NOT, any more: they wire
    :func:`builders_for_tree` and stream for real.  Their calls outlived the
    wiring by a release and refused ``mode = "on"`` -- the one mode that asks
    for streaming unconditionally -- with a message asserting the route wired
    no builder, while ``mode = "auto"`` went through that same builder and
    streamed.  So a user who wrote the explicit form got a refusal quoting a
    fact that had stopped being true, and ``go``, the front door most users
    type, mirrored it before the download.  Removed there; kept here for the
    route that genuinely cannot honour the mode.

    ``mode = "on"`` is knowable at admission with NO device work at all: it
    streams unconditionally, consults no planner and needs no card.  So it is
    refused here.  ``mode = "auto"`` is by default NOT refused: it asks
    :mod:`tilestream.autoplan` about a specific card and legitimately answers
    "resident" on a machine where the domain fits, and asking that question
    at admission would stand up a CUDA primary context in a process that has
    not decided to use the device yet (the reason ``go``'s memory gate probes
    in a subprocess).  What ``auto`` does on a domain that does NOT fit is a
    separate and unfixed gap, named in the message rather than papered over.

    ``consults_the_seam=False`` refuses ``auto`` as well, and is for a route
    that never calls :func:`make_stepper` or :func:`steppers_for_tree` at
    all.  ``gpuwm run`` (:func:`gpuwm.runtime.run_experiment`) is one:
    neither its single-domain arm nor either of its tree arms reads
    ``exp.tiles``, so ``auto`` there is not "asked and answered
    resident" -- it is never asked, and the run proceeds resident with
    nothing said.  That is precisely the silence this module's docstring
    forbids ("Refused rather than silently integrated resident"), and on a
    route that cannot honour the mode the honest answer is a refusal, not a
    decision it will not act on.
    """
    from gpuwm.explain import layered

    options = getattr(exp, "tiles", None) or OFF
    # Every mode this config asks for ANYWHERE, tree-wide or on a
    # [[domain]] row.  Reading only the tree-wide table would let a config
    # whose streaming lives entirely in per-domain tables through an
    # admission whose whole job is to catch it.
    modes = {options.mode} | {
        options_for_domain(dc, options).mode
        for dc in (getattr(exp, "domains", ()) or ())}
    if modes <= {"off"}:
        return
    if "on" not in modes and consults_the_seam:
        return
    asked = "/".join(sorted(m for m in modes if m != "off"))
    unread = ("" if consults_the_seam else
              "  This route does not read [tiles] at ANY point, so "
              "mode = 'auto' is refused here too: it would not be decided "
              "and found resident, it would never be asked.")
    auto_note = (
        "mode = 'auto' is accepted by this route and will run RESIDENT "
        "wherever tilestream.autoplan says the domain fits on the card in "
        "front of it.  On a domain autoplan says does NOT fit, auto reaches "
        "the same missing builder -- but only after the resident allocation "
        "it was trying to avoid has already been attempted."
        if consults_the_seam else
        "mode = 'auto' is refused by this route rather than accepted, "
        "because this route never consults the seam at all: there is no "
        "point at which it would notice the answer.")
    raise StreamingRefused(layered(
        f"[tiles] mode = '{asked}' asks the {route} route to "
        "integrate out of a pinned host store, and that route wires no "
        "streamed-domain builder, so it cannot.  Refusing here, at "
        "admission, rather than after the preparation has been restored and "
        f"every domain has been built on the card.{unread}",
        "A streamed domain needs its store filled from the prepared state, "
        "tile buffers built with the domain's own physics selectors, the "
        "domain geography inventoried and the lateral boundary tables "
        "windowed per tile.  gpuwm.core.streaming.attach() does all of it "
        "and tilestream/test_join.py drives it end to end, bit-exact "
        "against the resident run; what is missing is the route-side "
        "wiring that hands attach() the prepared state.\n\n"
        "Note that wiring alone is not the whole job.  attach() takes a "
        "PREPARED RESIDENT DomainState and copies its carriers into the "
        "store, so through this seam a domain still has to fit on the card "
        "once before it can be streamed off it.  Starting a domain that "
        "never fits means routing the ingest's output into the pinned store "
        "field by field, which tilestream/REAL-DATA.md names as the one "
        "piece of work out-of-core still needs.\n\n"
        + auto_note))


def refuse_streamed_nests(exp, *, source: str = "<config>") -> None:
    """``[tiles] mode = "on"`` over a config that declares a NEST: refused.

    THE PLACEMENT IS THE POINT, and it is the same argument
    :func:`refuse_unrouted_streaming` makes one function up.  ``mode =
    "on"`` streams EVERY grid unconditionally, so on a tree every
    parent-child coupling edge would have BOTH ends streamed -- and that
    is the one concurrent-nesting shape the coupler refuses
    (``gpuwm.core.nest.NestCoupler.force``): it would compose the
    streamed-parent footprint corridor with the streamed-child frame
    corridor and the per-tile table windows in one FORCE, and no gate has
    driven that composition.  Ungated is refused, not run.  Without this
    check the fact is discovered mid-run, at the first FORCE -- after the
    fetch, after the preparation, after the whole tree is built -- so the
    same fact is asked here, at ``build_experiment``, the one load every
    front door shares (``run``/``go``/``check``, both prepared runners,
    the DA drivers, the wizard's candidate loop).  The coupler's refusal
    stays exactly where it is, as the backstop for a caller that
    assembles an :class:`~gpuwm.experiment.ExperimentConfig` without
    going through the loader.

    WHAT THIS DOES NOT SAY -- because it stopped being true when the
    per-domain roads landed -- is that a nested domain cannot stream.  A
    resident parent can drive a tile-streamed child
    (``tilestream/test_streamed_child.py``), a streamed parent can drive
    a resident child (``tilestream/test_nest_executor.py``), and
    :func:`steppers_for_tree` prices both roads against one budget.  Only
    an EDGE with both ends streamed is unsupported, which is exactly the
    shape ``mode = "on"`` forces on every edge of a tree.

    ``mode = "auto"`` IS NOT REFUSED HERE, and that is load-bearing
    rather than an omission: ``auto`` is a request to stream only what
    does not fit, the planner prices each domain against the budget its
    predecessors left, and any mixed shape it produces is legal.  The
    walk itself refuses the one shape that is not, at decision time,
    before anything is built.
    """
    from gpuwm.explain import layered

    options = getattr(exp, "tiles", None) or OFF
    domains = tuple(getattr(exp, "domains", ()) or ())
    # THE EDGE, NOT THE TABLE.  This used to refuse on the tree-wide mode
    # alone, which was exact while ``[tiles]`` was tree-wide and one
    # ``mode = "on"`` really did put both ends of every edge on the
    # streamed road.  With the per-domain surface it is no longer exact:
    # `mode = "on"` over the tree with `tiles = { mode = "off" }` on the
    # nest is "stream the parent, keep the child resident", which is a
    # legal shape and now a reachable one.  So the question asked here is
    # the one the coupler actually answers -- does any single EDGE have
    # both ends unconditionally streamed -- and a tree that says something
    # per domain is judged on what it said.
    mode_by_gid = {int(dc.grid_id): options_for_domain(dc, options).mode
                   for dc in domains}
    both_on = tuple(
        (int(getattr(dc, "parent_id", 0)), int(dc.grid_id))
        for dc in domains
        if int(getattr(dc, "parent_id", 0)) != 0
        and mode_by_gid.get(int(dc.grid_id)) == "on"
        and mode_by_gid.get(int(getattr(dc, "parent_id", 0))) == "on")
    if not both_on:
        return
    nests = tuple(gid for _parent, gid in both_on)
    roots = tuple(int(dc.grid_id) for dc in domains
                  if int(getattr(dc, "parent_id", 0)) == 0)
    root = roots[0] if roots else 1
    # A domain that MOVES is a nest by construction, and it is worth
    # naming separately: a reader who configured [relocation] is thinking
    # about the follow domain, not about the parent_id that makes it a
    # nest, and would otherwise read this refusal as being about some
    # other grid.
    relocation = getattr(exp, "relocation", None)
    moving = (None if relocation is None
              or not getattr(relocation, "enabled", False)
              else int(relocation.grid_id))
    named = ", ".join(f"d{gid:02d}" for gid in nests)
    edges = ", ".join(f"d{p:02d} -> d{c:02d}" for p, c in both_on)
    first_child = int(both_on[0][1])
    first_parent = int(both_on[0][0])
    moving_note = (
        "" if moving is None or moving not in nests else
        f"  d{moving:02d} is also this config's [relocation] follow "
        "domain, which carries a SECOND reason it cannot be the streamed "
        "end: relocating a streamed child would rebuild its store, its "
        "tile plan, its geography gathers and its packed nest-table "
        "windows on a new footprint mid-run, and none of that is built "
        "or gated -- gpuwm.core.model refuses it at the first move.  Run "
        "the moving child RESIDENT and stream its parent instead.")
    raise StreamingRefused(layered(
        f"[tiles] mode = 'on' leaves {len(both_on)} coupling edge(s) of "
        f"{source} with BOTH ends streamed ({edges}; the streamed nest(s) "
        f"are {named}) -- the one concurrent-nesting shape that is "
        "refused (gpuwm.core.nest.NestCoupler.force), because it composes "
        "the streamed-parent footprint corridor with the streamed-child "
        "frame corridor and the per-tile table windows in one FORCE, and "
        "no gate has driven that composition.  Refused here, at config "
        "validation, rather than at the first FORCE -- which is a fetch, "
        "the preparations and a whole tree construction into a run that "
        f"was never going to finish.{moving_note}\n\n"
        "  remedy, and there are three.  SAY WHICH END STREAMS, which is "
        "the direct answer and the one most configs want: [tiles] is "
        "per-domain, so put `tiles = { mode = \"off\" }` on [[domain]] "
        f"grid_id = {first_child} to run it resident under a streamed "
        f"d{first_parent:02d} -- or the inverse, `tiles = {{ mode = "
        f"\"off\" }}` on the parent and leave the nest streaming.  Or set "
        "mode = 'auto', which prices each domain against one budget, "
        "reserves what the domains below it need before a streamed domain "
        "picks its tile, and streams only what does not fit.  Or delete "
        "the [tiles] table and run RESIDENT, which is what this tree did "
        f"before the mode existed.  If it is d{root:02d}, the root, you "
        "need to stream unconditionally -- a benchmark, or a "
        "bit-exactness proof -- run it alone: one [[domain]] table.",
        "Each single-streamed shape is gated alone "
        "(tilestream/test_nest_executor.py streams the parent, "
        "tilestream/test_streamed_child.py streams the child); an edge "
        "with both ends streamed is ungated, and ungated is refused, not "
        "run.\n\n"
        "A STREAMED PARENT OVER RESIDENT NESTS IS REACHABLE, and it is "
        "reachable two ways: name it per domain as above, or let mode = "
        "'auto' find it -- the tree decision reserves every undecided "
        "domain's claim before a streamed parent chooses its tile, so the "
        "parent takes a smaller tile instead of the largest one that fits "
        "and its children keep the card they need.  What remains refused "
        "is only the EDGE with both ends streamed, never the tree that "
        "contains a streamed domain."))


def _CannotPlan():
    """``tilestream.autoplan.CannotPlan``, imported where it is caught.

    Function-local like every other ``tilestream`` reach in this module:
    an unconfigured run must not import the planner at all.
    """
    from tilestream.autoplan import CannotPlan

    return CannotPlan


def _minimum_claim_bytes(node, total_budget: int) -> int:
    """The LEAST VRAM ``node`` can run in, for a reservation.

    A reservation must be a floor, never a wish: too small and it fails to
    protect the domain it was taken for, too large and it refuses a tree
    that would have run.  So each undecided domain is priced at the
    cheapest road actually open to it.

    A domain whose resident price fits the whole card will be RESIDENT --
    under a streamed ancestor it has no choice, because the both-ends law
    below refuses the alternative -- and a resident domain's floor is its
    resident price exactly: it does not shrink.  A domain too big to sit
    resident on the whole card must stream, and a streamed domain's floor
    is one buffer of the smallest legal compute window, the same number
    ``autoplan.plan`` reports when it says no tile fits.  Reserving that
    instead of an unaffordable resident price is what lets the walk reach
    the both-ends refusal -- which names the shape -- rather than dying at
    the planner's "no tile fits in 0.02 GiB", which names an arithmetic
    nobody wrote.
    """
    from tilestream import autoplan

    cfg = node.cfg.run
    fp = autoplan.footprint_for(cfg)
    nz = int(cfg.nz)
    cells = int(cfg.nx) * int(cfg.ny) * nz
    resident = int(fp.resident_bytes(cells))
    if resident <= int(total_budget):
        return resident
    halo = _halo_for(cfg)
    return int(fp.vram_bytes((2 * halo + 1) ** 2 * nz, 1))


def _decision_claim_bytes(node, decision, total_budget: int) -> int:
    """What the road this domain DECIDED on actually claims on the card.

    ``decision.resident_bytes`` is the planner's own number and is used
    wherever the planner was consulted.  The two roads that never consult
    it still occupy the card and still have to be paid for, or the next
    domain in the walk is handed their bytes a second time:

    * a PINNED tiling is priced at its own compute window -- the tile plus
      halo on both axes, times the buffers it pinned;
    * a domain whose ``[tiles]`` is OFF, which is now reachable inside a
      configured tree because the surface is per-domain, is priced
      resident, which is what it is.
    """
    if decision.resident_bytes is not None:
        return int(decision.resident_bytes)
    from tilestream import autoplan

    cfg = node.cfg.run
    fp = autoplan.footprint_for(cfg)
    nz = int(cfg.nz)
    if decision.stream and decision.tile_nx:
        halo = int(decision.halo or 0)
        window = ((int(decision.tile_nx) + 2 * halo)
                  * (int(decision.tile_ny) + 2 * halo) * nz)
        return int(fp.vram_bytes(window, int(decision.nbuffers or 1)))
    return _minimum_claim_bytes(node, total_budget)


def _tree_reservations(nodes, total_budget: int) -> list:
    """``[(reserved_bytes, [grid_id, ...]), ...]``, aligned with ``nodes``.

    Entry ``i`` is what the domains still UNDECIDED after ``nodes[i]`` need
    between them: the sum of their floors plus, for each of them that is a
    child, the coupling corridor its edge costs.  The corridor is priced in
    its RESIDENT form (``corridor_claim_bytes`` with no decision), which is
    the form a reserved domain will be in.
    """
    from gpuwm.core.nest_stream import corridor_claim_bytes

    floors = []
    for node in nodes:
        claim = _minimum_claim_bytes(node, total_budget)
        if getattr(node, "parent", None) is not None:
            claim += int(corridor_claim_bytes(node))
        floors.append(int(claim))
    out = []
    for index, node in enumerate(nodes):
        rest = range(index + 1, len(nodes))
        out.append((sum(floors[j] for j in rest),
                    [int(nodes[j].cfg.grid_id) for j in rest]))
    return out


def steppers_for_tree(model, options: StreamingOptions | None = None, *,
                      builders=None, machine=None, decisions=None) -> dict:
    """``{grid_id: stepper}`` for a whole domain tree.

    The route-facing entry point, and the reason the mode is configurable at
    all: a route reads ``exp.tiles``, calls this, and hands the result to
    :func:`gpuwm.core.model.execute_experiment`.

    With ``[tiles]`` absent -- the default -- this returns an EMPTY dict
    and touches nothing: no planner, no cupy, no ``tilestream`` import, and
    every grid falls through to ``dycore.step`` inside the executor.  A
    resident forecast therefore pays exactly nothing for the mode's
    existence, which is the only acceptable price for a feature most runs
    will never use.

    ``builders`` is ``{grid_id: build}``, the per-domain construction
    :func:`make_stepper` needs.  A domain whose decision says STREAM and
    whose route wired no builder is REFUSED, loudly, rather than quietly run
    resident: the failure that silence produces is an out-of-memory death at
    the allocation the mode existed to avoid, with nothing in the log to say
    the mode never engaged.

    A NEST streams through its own road (``gpuwm.core.nest_stream``), and
    the walk prices it like any domain -- what a tree may NOT contain is a
    coupling edge with BOTH ends streamed, the composition no gate has
    driven.  The walk refuses that shape at DECISION time, before any
    builder runs, so a tree that cannot run is a refusal with names
    instead of a root store filled and a coupler error at the first
    FORCE.  ``mode = "on"`` over a tree is the guaranteed form of it and
    is refused from the config text itself by
    :func:`refuse_streamed_nests`, long before a model exists to walk.

    ``decisions`` IS NOT OPTIONAL DECORATION -- IT IS THE OTHER FAILURE
    ------------------------------------------------------------------
    The return value cannot answer the question an operator actually asks.
    A grid that streamed appears in it; a grid that did NOT stream is simply
    absent, and absent is exactly what a grid looks like under
    ``[tiles]`` that was never configured at all.  So ``mode = "auto"``
    on a domain that fits returns ``{}`` -- the same empty dict as ``mode =
    "off"``, the same empty dict as a config with no ``[tiles]`` table --
    and the run proceeds resident with nothing anywhere to say that the mode
    was asked for and declined.

    That is not a hypothetical.  It is the shape of a FALSE PASS: point a
    streaming test at ``auto``, watch it go green, and conclude that
    streaming works, when what was actually measured is a resident run
    compared against a resident run.  This project has produced results that
    way, and a control that cannot fail is the recurring cause.

    Passing a mutable mapping as ``decisions`` fills it with ``{grid_id:
    StreamingDecision}`` for EVERY grid walked -- streamed or not -- so the
    caller holds the decision and its stated reason rather than inferring
    the decision from a hole in a dict.  :func:`streaming_receipt` turns
    that mapping into the run's receipt.  Callers that genuinely do not care
    may still omit it; the routes may not, and
    ``tilestream/test_route.py::control_auto_records_that_it_declined``
    fails if a route stops recording.

    ``decisions`` is filled with ``{grid_id: StreamingDecision}`` for EVERY
    grid, streamed or not, and it is the receipt's source.  It has to be an
    out-parameter rather than a second call to :func:`decide`, because under
    ``auto`` the decision is a function of the free VRAM at the instant it
    was taken: a receipt that re-derived it later would be describing a
    different card.  A run receipt that records a memory estimate and an
    observed peak beside each other, with nothing saying which execution mode
    produced them, is a receipt whose two memory numbers cannot be compared
    -- a streamed domain's observed peak is a handful of tile buffers against
    an estimate priced for a whole resident domain.
    """
    options = OFF if options is None else options
    if not tree_streams_anywhere(model, options):
        return {}
    import dataclasses

    builders = dict(builders or {})
    out = {}
    # THE TREE IS PRICED TOGETHER.  A per-domain decision that hands every
    # grid the whole card is how a resident parent and a streamed child
    # both plan against the same bytes and meet as an OOM mid-run instead
    # of a refusal.  Walking parent-first, every domain's decision is made
    # against the budget its predecessors LEFT: a resident domain's claim
    # is autoplan's own resident price, a streamed domain's is its tile
    # working set (both carried on the decision as ``resident_bytes``), and
    # a coupled child adds its corridor claim (rolling tables, coupling
    # slots, packed per-tile table windows).  Each decision's ``detail``
    # records the road and the arithmetic -- the receipt an operator reads
    # to see which claims consumed the card.
    #
    # Only when the planner is consulted: a PINNED tiling asks no question
    # and probes no card (the bit-exactness gates depend on that), so it
    # gets roads recorded and no budget arithmetic.  Per domain now, so a
    # tree pinned end to end still probes nothing, and a tree with one
    # planner-driven domain in it does the arithmetic for all of them --
    # a pinned domain occupies the card whether or not it asked a
    # question, and leaving it out of the budget would hand its bytes to
    # the next domain twice.
    nodes = list(model.walk_parent_first())
    per_domain = [options_for_domain(node.cfg, options) for node in nodes]
    consults_planner = any(o.enabled and o.tile_nx is None
                           for o in per_domain)
    machine0 = machine
    if consults_planner and machine0 is None:
        from tilestream import autoplan

        # Detected ONCE for the whole walk, not once per grid: the free
        # VRAM the budget subtracts from must be one number or the
        # claims below double-count whatever moved between probes.
        #
        # ``host_bytes=`` IS HANDED TO THE PROBE, on the same law
        # :func:`decide` states at length: ``detect`` skips the host-memory
        # read entirely when it is told the budget, and RAISES where it can
        # find no host source -- so a bare ``detect()`` here refused every
        # [tiles] tree on a box with no procfs and no cgroups, including
        # the ones whose ``host_budget_bytes`` supplied the very number the
        # refusal asked for.  ``decide``'s own override could not save it:
        # that arm runs only when no machine is passed, and this wrapper
        # passes one.  This is the PRODUCTION door to the planner, so the
        # fix has to be here as well as there.
        machine0 = autoplan.Machine.detect(
            host_bytes=(None if options.host_budget_bytes is None
                        else int(options.host_budget_bytes)))
    # THE CONFIGURED BUDGETS ARE FOLDED INTO THE MACHINE, ONCE.  They name a
    # CARD and a BOX, not a domain, so they cannot ride on the per-domain
    # tables (the loader refuses them there) and they must not ride on the
    # options handed to ``decide`` either: ``decide`` re-applies them to
    # whatever machine it is given, which would clobber every reduction this
    # walk makes -- both the predecessors' spend and the successors'
    # reservation.  Folding them in here leaves exactly one place each
    # number lives.
    if consults_planner and options.vram_budget_bytes is not None:
        machine0 = dataclasses.replace(
            machine0, vram_bytes=int(options.vram_budget_bytes),
            vram_headroom=0.0)
    if consults_planner and options.host_budget_bytes is not None:
        machine0 = dataclasses.replace(
            machine0, host_bytes=int(options.host_budget_bytes),
            pinned_fraction=1.0)
    if consults_planner:
        per_domain = [dataclasses.replace(o, vram_budget_bytes=None,
                                          host_budget_bytes=None)
                      if (o.vram_budget_bytes is not None
                          or o.host_budget_bytes is not None) else o
                      for o in per_domain]
    # DECIDE THE WHOLE TREE, THEN BUILD IT.  Two passes, not one, and the
    # split is the refusal's timing: parent-first, the ROOT's stepper used
    # to be built -- its whole pinned host store filled -- before the
    # child's decision was even taken, so a tree whose decisions could not
    # run together paid the entire run-up to learn a fact the decisions
    # already contained.  With every decision in hand first, an
    # unbuildable tree refuses before a single byte is pinned.
    #
    # TWO LEDGERS, NOT ONE.  VRAM and pinned HOST RAM are separate finite
    # pools and a streamed domain spends both -- its tile buffers on the
    # card, its whole-domain store and arena on the box.  A single total
    # subtracted only the VRAM, so a resident root with two streamed
    # siblings priced EACH child against the entire host budget: both plans
    # were accepted, both stores were pinned, and the second one met
    # cudaHostAlloc instead of the planner refusal that exists to prevent
    # exactly that.  Host RAM is the binding constraint at every capacity
    # limit measured, so it is the ledger that most needs keeping.
    spent = 0
    host_spent = 0
    decided: list = []
    decided_by_gid: dict[int, object] = {}

    def reduced(by: int, host_by: int = 0):
        """``machine0`` with ``by`` VRAM and ``host_by`` host bytes spent."""
        if not (consults_planner and (by or host_by)):
            return machine0
        reduced_machine = machine0
        if by:
            reduced_machine = dataclasses.replace(
                reduced_machine,
                vram_bytes=max(0, machine0.vram_budget_bytes - by),
                vram_headroom=0.0)
        if host_by:
            # ``pinned_fraction`` is forced to 1.0 for the same reason
            # ``vram_headroom`` is zeroed above: the number handed on IS the
            # budget, so a multiplier applied on top of it would shrink the
            # remainder a second time.
            reduced_machine = dataclasses.replace(
                reduced_machine,
                host_bytes=max(0, machine0.host_budget_bytes - host_by),
                pinned_fraction=1.0)
        return reduced_machine

    # RESERVE BEFORE THE GREEDY CHOOSE.  Walking parent-first prices each
    # domain against what its PREDECESSORS left, which is the right law for
    # a resident domain -- its claim is fixed and it takes what it needs.
    # It is the wrong law for a STREAMED one, because a streamed domain's
    # claim is not fixed: the tile search maximises the compute window
    # against whatever budget it is shown, so a streamed parent took the
    # largest clean tile that fit the whole card -- MEASURED at 3.94 of
    # 4.00 GiB, 98.5% -- and its resident child then met the planner's
    # "no tile fits in 0.06 GiB".  STREAMED PARENT + RESIDENT CHILD is a
    # shape the engine runs bit-identically; only the order of the
    # arithmetic made it unreachable.
    #
    # So the successors are priced FIRST, at the floor each of them can
    # run in, and a streamed domain's tile search sees budget-minus-
    # successors.  A parent on a smaller tile is a slower parent; a
    # starved child is no run at all, and the tiling tax is 1.2x-1.4x
    # against a refusal.
    #
    # The reservation constrains the TILE, never the VERDICT: the
    # stream-or-resident question is still asked against the budget the
    # predecessors left, exactly as before, so a tree that ran all-resident
    # decides all-resident still and no existing road moves.
    total_budget = (int(machine0.vram_budget_bytes) if consults_planner
                    else 0)
    reservations = (_tree_reservations(nodes, total_budget)
                    if consults_planner else [(0, [])] * len(nodes))
    for node, node_options, (reserve, reserved_for) in zip(
            nodes, per_domain, reservations):
        gid = int(node.cfg.grid_id)
        cfg = node.cfg.run
        node_machine = reduced(spent, host_spent)
        # Decided ONCE and handed to make_stepper, rather than letting
        # make_stepper decide again: `auto` consults the planner, and a
        # second consultation on a card whose free VRAM moved in between
        # could answer differently from the one that got recorded.  The
        # receipt would then describe a run that did not happen.
        decision = decide(cfg, node_options, machine=node_machine)
        if decision.stream and reserve:
            budget_before = int(decision.budget_bytes or 0)
            tile_machine = reduced(spent + int(reserve), host_spent)
            try:
                decision = decide(cfg, node_options, machine=tile_machine)
            except _CannotPlan() as exc:
                raise StreamingRefused(
                    f"d{gid:02d} streams, and the domains still undecided "
                    f"below it ({', '.join(f'd{g:02d}' for g in reserved_for)}"
                    f") need {reserve / (1024 ** 3):.2f} GiB between them, "
                    f"which leaves nothing to tile d{gid:02d} with: "
                    f"{budget_before / (1024 ** 3):.2f} GiB was free at this "
                    f"point in the walk and the reservation takes "
                    f"{reserve / (1024 ** 3):.2f} GiB of it.  The "
                    "reservation is not optional -- without it d"
                    f"{gid:02d} would take the largest tile that fits and "
                    "its own children would then have no card left, which "
                    "is a refusal one domain later instead of here.  "
                    "Raise [tiles] vram_budget_bytes or free VRAM, shrink "
                    "the tree, or run it resident by deleting the [tiles] "
                    f"table.  The planner's own words: {exc}") from exc
            decision.detail.update(
                reserved_bytes=int(reserve), reserved_for=list(reserved_for),
                budget_before_reserve_bytes=budget_before)
        claim = _decision_claim_bytes(node, decision, total_budget)
        corridor = 0
        if getattr(node, "parent", None) is not None:
            from gpuwm.core.nest_stream import corridor_claim_bytes

            corridor = corridor_claim_bytes(node, decision=decision)
        host_claim = int(decision.detail.get("host_claim_bytes") or 0)
        decision.detail.update(
            road=("streamed" if decision.stream else "resident"),
            claim_bytes=claim, corridor_claim_bytes=corridor,
            budget_spent_before_bytes=int(spent),
            host_spent_before_bytes=int(host_spent),
            configured_mode=node_options.mode)
        if consults_planner:
            spent += claim + corridor
            host_spent += host_claim
        if decisions is not None:
            decisions[gid] = decision
        # A coupling edge with BOTH ends streamed is refused at DECISION
        # time -- the same law gpuwm.core.nest.NestCoupler.force enforces
        # at every FORCE, asked here where it costs a sentence instead of
        # a run.  Each single-streamed shape is gated alone
        # (tilestream/test_nest_executor.py streams the parent,
        # tilestream/test_streamed_child.py streams the child); the
        # composition is ungated, and ungated is refused, not run.
        parent = getattr(node, "parent", None)
        if parent is not None and decision.stream:
            pgid = int(parent.cfg.grid_id)
            pdecision = decided_by_gid.get(pgid)
            if pdecision is not None and pdecision.stream:
                raise StreamingRefused(
                    f"the coupling edge d{pgid:02d} -> d{gid:02d} has BOTH "
                    "ends streamed: it would compose the streamed-parent "
                    "footprint corridor with the streamed-child frame "
                    "corridor and the per-tile table windows in one FORCE, "
                    "and no gate has driven that composition.  Ungated is "
                    "refused, not run -- and refused here, at decision "
                    "time, rather than by NestCoupler.force after the "
                    "whole tree is built.  Leave one end resident: say so "
                    f"explicitly with a per-domain table (tiles = {{ mode = "
                    f"\"off\" }} on [[domain]] grid_id = {gid}, which runs "
                    f"d{gid:02d} resident under a streamed d{pgid:02d}), or "
                    "raise [tiles] vram_budget_bytes or free VRAM so one of "
                    "the two fits resident, or pin the whole run resident "
                    "by deleting the [tiles] table.")
        decided.append((node, cfg, node_options, node_machine, decision))
        decided_by_gid[gid] = decision
    for node, cfg, node_options, node_machine, decision in decided:
        gid = int(node.cfg.grid_id)
        stepper = make_stepper(node.state, cfg, node_options,
                               decision=decision, machine=node_machine,
                               build=builders.get(gid))
        if is_streaming(stepper):
            out[gid] = stepper
    return out


def streaming_receipt(options: StreamingOptions | None,
                      decisions: dict | None) -> dict:
    """What the run RECORDS about which way ``[tiles]`` went.

    Empty for an unconfigured run, and that emptiness is load-bearing: it is
    what keeps every receipt written before this mode existed byte-identical
    afterwards, the same promise :func:`identity_payload_entry` keeps for the
    restart identity.  A configured run gets a per-grid verdict::

        {"configured_mode": "auto",
         "streamed_any": false,
         "domains": {"1": {"streamed": false,
                           "reason": "the domain fits resident on this card,
                                      and the tiling tax is 1.2x-1.4x ...",
                           "resident_bytes": 1234567890, ...}},
         "summary": "[tiles] mode='auto': NO domain streamed; grid 1 ran
                     resident because the domain fits resident on this card"}

    ``streamed_any`` is the field to assert on.  It is stated as a positive
    boolean rather than left implicit in the size of ``domains`` because the
    thing being guarded against is a reader -- human or test -- treating the
    ABSENCE of evidence as evidence: an ``auto`` run that declined and an
    ``off`` run are indistinguishable in the stepper dict, and were
    indistinguishable in the receipt too until this existed.

    ``summary`` is the one line a route prints.  It always names the mode
    that was CONFIGURED next to what actually happened, because those two
    differing is the entire failure: ``mode='auto'`` is a request, not an
    outcome, and a log that echoes only the request is a log that will be
    read as an outcome.

    KEYED ON THE DECISIONS, NOT ON THE TREE-WIDE TABLE
    --------------------------------------------------
    This function used to open with ``not options.enabled`` and return
    ``{}``, which made the TREE-WIDE ``[tiles]`` table the authority on
    whether anything streamed.  It is not the authority -- it is the
    DEFAULT.  Since the surface went per domain, a tree whose tree-wide
    mode is ``off`` and one of whose ``[[domain]]`` rows carries ``tiles =
    {mode = "on"}`` streams that grid, and every such run produced an
    EMPTY receipt: the operator got no line naming which grid had tiled.
    That is the same class of defect :func:`tree_streams_anywhere` exists
    to close -- a per-domain road the tree-wide table cannot see -- and it
    is closed the same way, by reading what the domains DECIDED.

    ``decisions`` is the right key because it is filled by
    :func:`steppers_for_tree` for EVERY grid walked and only for a tree
    that is configured somewhere: an unconfigured tree returns early on
    ``tree_streams_anywhere`` and never fills it, so the empty receipt
    that keeps pre-``[tiles]`` runs byte-identical still comes out empty.
    """
    if not decisions:
        return {}
    if options is None:
        options = OFF
    domains: dict[str, dict] = {}
    for gid in sorted(decisions):
        d = decisions[gid]
        entry: dict[str, object] = {"streamed": bool(d.stream),
                                    "reason": d.reason}
        if d.stream:
            entry.update(tile_nx=d.tile_nx, tile_ny=d.tile_ny,
                         nbuffers=d.nbuffers, halo=d.halo, store=d.store)
        if d.resident_bytes is not None:
            entry["resident_bytes"] = int(d.resident_bytes)
        if d.budget_bytes is not None:
            entry["budget_bytes"] = int(d.budget_bytes)
        # The reservation, where one was taken.  A domain that held budget
        # back for its successors chose a SMALLER tile than the card could
        # hold, and without this an operator reading that tile has no way
        # to tell a reservation from a bad plan.  Absent, not zero, where
        # nothing was reserved: every receipt written before the joint
        # decision existed stays byte-identical.
        if d.detail.get("reserved_bytes"):
            entry["reserved_bytes"] = int(d.detail["reserved_bytes"])
            entry["reserved_for"] = [int(g) for g in
                                     d.detail.get("reserved_for", ())]
        # The mode THIS domain was configured with, recorded only where it
        # differs from the tree-wide one.  A per-domain table is how a user
        # says "stream the parent, keep the child resident", and a receipt
        # that echoed only the tree-wide mode would describe a different
        # configuration from the one on disk.
        own_mode = d.detail.get("configured_mode")
        if own_mode is not None and own_mode != options.mode:
            entry["configured_mode"] = str(own_mode)
        domains[str(int(gid))] = entry
    streamed = sorted(g for g in decisions if decisions[g].stream)
    resident = sorted(g for g in decisions if not decisions[g].stream)
    if streamed and not resident:
        what = f"grid(s) {streamed} streamed"
    elif streamed:
        what = f"grid(s) {streamed} streamed, {resident} ran resident"
    else:
        first = decisions[resident[0]]
        what = (f"NO domain streamed; grid(s) {resident} ran resident "
                f"because {first.reason}")
    # WHERE THE TREE-WIDE MODE DID NOT ASK FOR THIS, SAY WHO DID.  With the
    # tree-wide table disabled, "[tiles] mode='off': grid(s) [2] streamed"
    # is accurate and reads like a contradiction -- an operator has no way
    # to tell it from a receipt that is simply wrong.  Naming the domains
    # that overrode the default turns it into a sentence.
    #
    # Scoped to `not options.enabled` on purpose: that is exactly the case
    # which printed NOTHING before, so there is no shipped byte here to
    # move.  A tree the tree-wide table governs keeps the line it has
    # always printed, which
    # tests/test_streaming.py::test_the_per_domain_receipt_fix_moves_no
    # _shipped_receipt_byte pins literally.
    prefix = f"[tiles] mode={options.mode!r}"
    if not options.enabled:
        overrides = [f"d{int(gid):02d} mode="
                     f"{str(decisions[gid].detail.get('configured_mode'))!r}"
                     for gid in sorted(decisions)
                     if decisions[gid].detail.get("configured_mode")
                     not in (None, options.mode)]
        if overrides:
            prefix = (f"{prefix} tree-wide, overridden per domain "
                      f"({', '.join(overrides)})")
    return {"configured_mode": options.mode,
            "streamed_any": bool(streamed),
            "domains": domains,
            "summary": f"{prefix}: {what}"}


def _drain_streamed(state) -> None:
    """Land a streamed domain's in-flight sweep before its store is touched.

    The one door every store reader outside :mod:`tilestream` shares: the
    ``_streamed_domain`` marker :class:`StreamedDomain` leaves on the state.
    A resident state, and any stand-in whose run carries no ``drain``, is a
    ``getattr`` and a return.
    """
    streamed = getattr(state, "_streamed_domain", None)
    if streamed is None:
        return
    drain = getattr(getattr(streamed, "_run", None), "drain", None)
    if drain is not None:
        drain()


def live_scratch(state, slot: str) -> tuple:
    """Every array that IS scratch ``slot`` for this domain, right now.

    THE PROBLEM THIS SOLVES, which cost a whole diagnostic its meaning.
    A streamed domain's arrays live in the store; the ``DomainState`` the
    model still holds was copied into that store at :func:`attach` and has
    been a corpse ever since.  Reading it is harmless -- nothing does -- but
    WRITING it is not, and three consumers write it every run.  The
    ``nwp_diagnostics`` running maxima are reset externally, by whoever read
    them: ``uh_diag.reset_up_heli_max`` after each history frame is durable,
    and ``uh_diag.reset_tracker_window`` by the relocation runner at every
    evaluation and by the spawn runner at every leg boundary.  Each of those
    reached ``state.existing_scratch(slot)`` and zeroed it, so under a host
    store the fold landed in the store while the reset landed on the corpse
    and the window never actually reset.  "Max since I last looked" silently
    became "max since the run began" -- monotonically biasing the tracker and
    the spawn trigger toward firing early and never releasing, with no NaN
    and no warning, and invisible in any run too short to cross a reset.

    So a consumer that resets a whole-domain accumulator asks for the arrays
    rather than for the state's, and gets one array resident, two streamed
    (the store's, which is the domain, and the state's, which is kept
    consistent so a later re-attach cannot resurrect the stale one).

    SAFE TO WRITE BETWEEN STEPS AND ONLY BETWEEN STEPS.  The store is written
    by tile scatters on non-blocking streams, but ``TiledRun.sweep``
    synchronises every stream and the device before it returns, so the store
    is quiescent at exactly the instants the model does its output, restart
    and diagnostic work.  Anything that wrote it from inside a sweep would be
    racing the scatters.

    ORDER IS PART OF THE CONTRACT: the state's own buffer first, the store's
    last, so ``live_scratch(...)[-1]`` is the authoritative one.
    :func:`domain_scratch` is that expression named, and a READER must use it
    -- the same corpse that swallowed the resets would otherwise be handed to
    the storm tracker as the field it steers on.

    "Between steps" is enforced here rather than assumed: under the driver's
    deferred sweep seam a sweep returns with its scatters still in flight,
    so this drains the streamed domain before handing out arrays anything
    will write.  A resident state, and every test double without a drain,
    pays a ``getattr`` and nothing else.
    """
    _drain_streamed(state)
    views = []
    existing_scratch = getattr(state, "existing_scratch", None)
    if existing_scratch is not None:
        buf = existing_scratch(slot)
        if buf is not None:
            views.append(buf)
    streamed = getattr(state, STREAMED_SCRATCH_ATTR, None)
    if streamed is not None:
        buf = streamed.get(slot)
        if buf is not None:
            views.append(buf)
    elif getattr(state, "_streamed_domain", None) is not None:
        # The other back-reference, from feat-wrfout-stream.  ``attach`` sets
        # both, but a StreamedDomain constructed directly -- which every test
        # double does -- carries only this one, and a reset that silently did
        # nothing for those is the exact failure this function exists for.
        store = getattr(state._streamed_domain, "store", None) or {}
        buf = store.get("scratch/" + slot)
        if buf is not None:
            views.append(buf)
    return tuple(views)


def domain_scratch(state, slot: str):
    """The ONE array that IS the domain's scratch ``slot``, or ``None``.

    What a READER wants, where :func:`live_scratch` is what a WRITER wants.
    Resident, this is ``state.existing_scratch(slot)`` and nothing has
    changed.  Streamed, it is the store's array -- the one the tiles are
    actually folding into -- rather than the copy left on the state at
    attach time, which stops moving the moment the domain does.

    Returns ``None`` for a state that has no such slot, so a caller keeps
    whatever refusal it already raised for that case.
    """
    views = live_scratch(state, slot)
    return views[-1] if views else None
# ---------------------------------------------------------------------------
# where a streamed domain's TRUTH lives, for the readers that are not the sweep
# ---------------------------------------------------------------------------

#: The attribute a streamed domain publishes its store under.  Private on the
#: state because it is not part of ``DomainState``'s serialized contract --
#: ``gpuwm/io/restart.py``'s attribute walk classifies it as infra, and the
#: restart stream reads the store directly (``tilestream.restart_stream``).
_STORE_ATTR = "_streamed_store"


def publish_store(state, streamed) -> None:
    """Bind a streamed domain's store to the ``DomainState`` the model holds.

    THE BUG THIS EXISTS FOR.  :func:`attach` copies the prepared state's
    carriers into a store and every subsequent :meth:`StreamedDomain.__call__`
    updates THAT STORE.  The ``DomainState`` object stays in the tree -- the
    model holds it, the health validator holds it, ``DomainNode.state`` is it
    -- and its device arrays are frozen at the instant of attach.  For the
    sweep that is fine, because the sweep never reads them.  For anything
    else that reads ``node.state`` it is not fine at all, and the reader that
    matters is :class:`gpuwm.core.nest.NestCoupler`: ``force()`` couples
    ``node.parent.state`` (nest.py:229) and would force a child from the
    parent's air AT ATTACH TIME for the whole forecast -- no NaN, no warning,
    and a nest that looks like it is running.

    Publishing the store here rather than teaching every reader about
    :mod:`tilestream` keeps the knowledge in one place: a reader asks
    :func:`domain_store` whether this state is a window onto something else,
    and gets ``None`` for every resident domain, which is every domain in a
    run that configures no ``[tiles]``.
    """
    setattr(state, _STORE_ATTR, streamed.store)


def domain_store(state):
    """``{carrier key: array}`` when ``state`` is streamed, else ``None``.

    ``None`` is the resident answer and the common one; callers branch on it
    rather than on a mode flag, so a domain that never streamed executes the
    identical code it always did.
    """
    return getattr(state, _STORE_ATTR, None)


def window_slices(shape, window) -> tuple:
    """``window`` = ``(j0, j1, i0, i1)`` in MASS cells, on any staggering.

    Staggered arrays carry one extra face on their own axis, so the
    window is widened by one there and clamped to the array -- a
    superset is always safe (it copies a cell the reader will not use)
    and a subset is a stale face inside the zone the reader WILL use.
    The widening is applied on BOTH axes unconditionally rather than per
    stagger, for the same superset-is-free reason: one rule, no per-field
    geometry to get wrong.

    ``window=None`` -- and any array too flat to window -- is the whole
    array, so a caller that windows nothing executes the path it always
    did.
    """
    if window is None or len(shape) < 2:
        return (Ellipsis,)
    j0, j1, i0, i1 = (int(v) for v in window)
    ny, nx = int(shape[-2]), int(shape[-1])
    return (Ellipsis,
            slice(max(0, j0), min(ny, j1 + 1)),
            slice(max(0, i0), min(nx, i1 + 1)))


def frame_windows(ny: int, nx: int, width: int) -> tuple:
    """The four boundary-frame strips of a domain, as ``window_slices`` windows.

    ``((j0, j1, i0, i1) x 4)`` in MASS cells: west/east strips full-height,
    south/north strips full-width, each ``width`` cells deep.  The corners
    are covered TWICE, deliberately: the superset rule that governs every
    window in this subsystem makes an overlapping copy idempotent and a
    missing one a stale cell inside the zone the reader uses.  Each window
    is then sliced by :func:`window_slices` -- the same +1 staggered-face
    widening and clamp every other consumer applies -- so the frame rule
    adds no second slice arithmetic to get wrong.

    This is the CHILD-side mirror of the FORCE corridor's footprint window:
    ``bdy_interp1`` reads the child only inside its boundary zone (the
    kernel writes ``nz*sz*(nyc|nxc)`` table cells per side and reads the
    child field at exactly those positions), so a streamed child's FORCE
    pull needs four strips, not a domain.
    """
    ny, nx, width = int(ny), int(nx), int(width)
    if width < 1:
        raise ValueError("frame width must be at least one cell")
    return (
        (0, ny, 0, width),           # west
        (0, ny, nx - width, nx),     # east
        (0, width, 0, nx),           # south
        (ny - width, ny, 0, nx),     # north
    )


def _carrier_key(attr: str) -> str:
    """``mup`` -> ``state/mup``: the restart member name a store is keyed by.

    ``tilestream.physics_inventory.carrier_manifest`` keys the store by the
    RESTART member name, so the store and a tile buffer address the same
    field by the same string.  A ``DomainState`` attribute is the ``state/``
    family of that namespace.
    """
    return f"state/{attr}"


def refresh_from_store(state, attrs, *, window=None) -> int:
    """Copy named carriers STORE -> the state's own device arrays, in place.

    Returns the bytes moved, so a caller can report the traffic it is paying
    instead of hiding it.  A resident state moves nothing and returns 0.

    ``window`` is ``(j0, j1, i0, i1)`` in parent MASS cells, sliced by
    :func:`window_slices` -- the same superset rule every other consumer of
    a footprint window applies.  It exists for the FORCE corridor: the nest
    coupler reads the parent only inside the child's footprint plus its
    stencil, and a domain is streamed BECAUSE it does not fit, so pulling
    whole fields per parent step is O(parent) traffic where O(child
    footprint) is what the read needs.  ``None`` is the whole field, which
    is what a whole-domain consumer (feedback's finalize diagnostics) still
    wants.

    This repairs a desync; it does not create a device allocation.  The
    arrays written are the ones ``attach`` copied FROM, so this is only
    available while the route still holds a resident prepared state --
    which every route does today (``attach`` copies out of it and nothing
    frees it).  A future route that initialises straight into the host store
    has no such arrays; the windowed read here is the traffic half of that
    corridor, and the slot half (window-shaped device buffers) is the piece
    still standing in front of store-direct nesting.
    """
    store = domain_store(state)
    if store is None:
        return 0
    # The coupler reads the raw published mapping, not the draining store
    # property, so the deferred sweep seam's in-flight tail is landed here.
    _drain_streamed(state)
    import numpy as _np

    moved = 0
    for attr in attrs:
        src = store.get(_carrier_key(attr))
        if src is None:
            continue
        dst = getattr(state, attr, None)
        if dst is None:
            raise StreamingRefused(
                f"the store carries {attr!r} but the state does not; the "
                "two describe different domains")
        if tuple(dst.shape) != tuple(src.shape):
            raise StreamingRefused(
                f"store {attr!r} has shape {tuple(src.shape)}, state has "
                f"{tuple(dst.shape)}")
        sl = window_slices(dst.shape, window)
        if not isinstance(src, _np.ndarray):
            dst[sl] = src[sl]                      # device-store mode
        elif isinstance(dst, _np.ndarray):
            dst[sl] = src[sl]                      # host stand-in (tests)
        elif window is None:
            # Whole-field host -> device.  ``dst[...] = host`` routes
            # through cupy's fill and raises; ndarray.set is the documented
            # door and needs a contiguous destination, which a whole array
            # is.
            dst.set(_np.ascontiguousarray(src))
        else:
            # Windowed host -> device.  The window view of ``dst`` is not
            # contiguous, so it cannot take ``set``; stage the window on
            # the device and assign device-to-device, exactly as
            # ``StreamedDomain.sync_to_state`` does.
            import cupy as _cp

            dst[sl] = _cp.asarray(_np.ascontiguousarray(src[sl]))
        moved += int(src[sl].nbytes if window is not None else dst.nbytes)
    return moved


def commit_to_store(state, attrs, *, window=None) -> int:
    """Copy named carriers the state's device arrays -> the STORE, in place.

    The write half of :func:`refresh_from_store`, for the one caller that
    legitimately mutates a domain from outside ``dycore.step``: two-way nest
    feedback writes the PARENT's prognostics
    (:meth:`gpuwm.core.nest.NestCoupler.feedback_commit`).  Without this the
    next sweep gathers from the store and the feedback is silently discarded
    -- the parent integrates on as though the child had never run.

    ``window`` narrows the write to the child's footprint window under the
    same rule as the read half.  A windowed commit is complete exactly when
    everything the caller CHANGED lies inside the window -- which is the
    feedback restriction's defining property (``feedback_parent_bounds``
    excludes even the child's own specified zone) -- and a window that did
    not cover a write would silently discard it, so the caller names its
    ground and the seam moves only that.
    """
    store = domain_store(state)
    if store is None:
        return 0
    # A WRITE into the store must not race the previous sweep's scatters.
    _drain_streamed(state)
    import numpy as _np

    moved = 0
    for attr in attrs:
        key = _carrier_key(attr)
        dst = store.get(key)
        if dst is None:
            continue
        src = getattr(state, attr, None)
        if src is None:
            raise StreamingRefused(
                f"the store carries {attr!r} but the state does not")
        sl = window_slices(_np.shape(dst), window)
        if not isinstance(dst, _np.ndarray):
            dst[sl] = src[sl]                      # device-store mode
        elif isinstance(src, _np.ndarray):
            dst[sl] = src[sl]                      # host stand-in (tests)
        elif window is None:
            src.get(out=dst)
        else:
            # ``.get()`` on a device view makes its own contiguous host
            # copy, so the strided store write is host-side numpy.
            dst[sl] = src[sl].get()
        moved += int(dst[sl].nbytes if window is not None else dst.nbytes)
    return moved


def is_streaming(stepper) -> bool:
    """Whether a stepper from :func:`make_stepper` streams.

    Answers a question ABOUT A STEPPER, and the history writers deliberately
    do not use it, because they do not have one: ``PerDomainWrfoutWriters
    .submit`` is handed a NODE, and the tree executor keeps the steppers.  A
    streamed domain is recognised there by the marker
    :class:`StreamedDomain` leaves on the resident state it took over
    (``state._streamed_domain``), which travels with the object the writers
    already hold.

    This docstring used to claim that "the run loops" used this function "to
    route output and restart", and they did not: every history call site read
    ``node.state`` unconditionally, so a streamed run wrote the initial
    condition into every frame with a later timestamp and no error anywhere.
    Recorded here rather than quietly corrected, because a docstring
    asserting a wiring that does not exist is how that survived --
    ``tilestream.test_history`` is the gate that would have caught it.
    """
    return isinstance(stepper, StreamedDomain)


def step_health(stepper, state, cfg, *, boundary_width: int):
    """The health of the domain ``stepper`` just advanced, wherever it lives.

    ONE call for both modes, because the run loop must not have two branches
    that can drift apart.  Resident: the dycore's own whole-domain
    ``stability_report`` on the state, which IS the domain.  Streamed: the
    report the sweep folded per tile out of the store, which is where the
    domain actually is -- reading the state there returns the corpse the store
    was filled from and never raises again.
    """
    if is_streaming(stepper):
        return stepper.health
    from gpuwm.core.dycore import stability_report

    return stability_report(state, cfg, boundary_width=boundary_width)


# ``step_health`` and ``stability_observer`` are two front doors onto the same
# guarantee, one from each of the two branches that fixed this defect.  They
# are both kept: ``stability_observer`` returns ``dycore.stability_report``
# ITSELF for a resident domain, which is what the run loop's pinned source
# asserts and what keeps the resident path branchless; ``step_health`` is the
# single-call form.  Neither is a wrapper round the other, and both end up at
# whichever per-tile fold ``attach`` armed.


# ``domain_call_counts`` is defined ONCE, beside ``stability_observer``.
# defect2-observer-fold added a second, identical-in-behaviour copy here; the
# duplicate is dropped rather than kept, because the later definition silently
# shadowed the earlier one and a shadowed observer is exactly the failure this
# whole seam is about.


def receipt_entry(options: StreamingOptions | None,
                  decisions: dict | None = None) -> dict:
    """What ``[tiles]`` contributes to a RUN RECEIPT: everything.

    The deliberate opposite of :func:`identity_payload_entry`, which
    contributes nothing.  Identity answers "may this checkpoint resume
    here?" and the mode must not bind it, because the whole claim of the
    mode is that it changes no byte of the forecast.  A receipt answers
    "what produced these numbers?", and there the mode is load-bearing:
    the memory block of a run receipt carries
    ``preflight_alloc_estimate_bytes`` beside ``gpu_peak_used_bytes_
    observed``, and those two are only comparable if the reader knows the
    run was resident.  Under streaming the estimate prices a whole domain
    and the peak measures a few tile buffers, and a reader with no third
    field to tell them apart will read the gap as slack in the estimate.

    ``mode = "off"`` -- the default, and what every experiment that never
    mentions ``[tiles]`` carries -- returns an empty dict, so a resident
    receipt is byte-identical to the one written before this existed.
    """
    options = OFF if options is None else options
    if not options.enabled:
        return {}
    entry: dict[str, object] = {"configured": options.to_json()}
    if decisions:
        entry["decisions"] = {
            f"d{int(gid):02d}": {
                "streamed": bool(decision.stream),
                "explain": decision.explain(),
                "resident_estimate_bytes": decision.resident_bytes,
                "planner_budget_bytes": decision.budget_bytes,
            }
            for gid, decision in sorted(decisions.items())
        }
        entry["any_streamed"] = any(d.stream for d in decisions.values())
    return entry
