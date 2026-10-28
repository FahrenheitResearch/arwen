# SPDX-License-Identifier: Apache-2.0
"""One process carries every ensemble member across analysis cycles.

THE COST THIS REMOVES.  A cycled ensemble on the WRF-input door runs one fresh
engine process per member per cycle: the process starts, reads its inputs,
restores the member's checkpoint, applies the analysis, steps five or fifteen
minutes and writes a checkpoint the next cycle restores.  On a 300 x 300 x 49
member and a five-minute cycle most of that leg is not stepping.  A member
session builds each member once and keeps it, so a cycle is the stepping, the
history frame the analysis reads, and the analysis applied to the live model.

WHAT IS THE SAME CODE.  Nothing numerical is re-implemented here:

* a member is built by the door's own construction,
  :func:`gpuwm.prepared_domain_tree_forecast.run_prepared_tree` on the member's
  real.exe directory with its checkpoint, and with the WRF-input door's
  initialization (or the analysis door's, :class:`gpuwm.da.analysis_door.
  AnalysisInitialization`, for the first cycle's analysis), handing the
  restored, analysed model over at the ``ensemble_bootstrap`` seam;
* it is advanced by the ordinary executor's own closures
  (:func:`gpuwm.ensemble.prepared_nested_batch.capture_original_member_schedule`
  around :func:`gpuwm.core.model.execute_experiment`) through
  :func:`gpuwm.core.clock.execute_schedule` with ``stop_ticks``, member by
  member: history at the run's own cadence (the namelist the files legs run
  is the one the member was built from), the reflectivity and helicity
  resets at the history rings, the history handler and the stability sample
  the door's handler takes;
* each later analysis goes in through the door's one function,
  :meth:`AnalysisInitialization.apply_door_plan` with
  ``continuation="in-memory"`` (the door's cycling restart without the
  checkpoint round trip), after the previous window is retired with the door's
  G10 check (:meth:`AnalysisInitialization.retire_window`);
* checkpoints are the door's deferred ones
  (:func:`gpuwm.io.restart.write_tree_restart` with ``defer_publish=True`` on
  a checkpoint writer thread ordered after the history frames), at the run's
  restart instants when the cadence asks for them.

Members a card cannot hold are parked between their turns
(:class:`gpuwm.ensemble.managed_storage.ManagedStorage`): built inside their
owner's allocation scope, resumed for their turn, parked after it.

A history frame each member writes at a cycle instant is also handed to a
:class:`gpuwm.io.frame_sink.FrameSink` (the words the file holds), so a
consumer reads the members' states without reading files.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from time import perf_counter
from typing import Callable, Mapping

SCHEMA = "gpuwm.member-session/1"
#: When a member writes a checkpoint: ``every-instant`` at every restart
#: instant its run schedules (what a files leg writes at its end), ``requested``
#: only at instants asked for (:meth:`MemberSession.request_checkpoint`) and at
#: the session's end, ``none`` never.
CADENCES = ("every-instant", "requested", "none")


class MemberSessionError(RuntimeError):
    """A member session refused or stopped; the message names the breakage."""


@dataclass
class MemberSpec:
    """One member of a session.

    ``run_directory`` is the member's real.exe directory (namelist.input,
    wrfinput_d01, wrfbdy_d01), ``restart`` the checkpoint it starts from,
    ``run_seconds`` its run's length from the member's start as the door takes
    ``--run-seconds`` (the session's last instant), ``plan`` the analysis door
    plan applied at the start (``None`` starts the checkpoint as it is).
    ``write_history_files`` keeps the member's ordinary history files beside
    its in-memory frames.  ``storage`` is ``device`` or ``managed`` (parked
    between its turns).
    """

    member_id: int
    run_directory: Path
    restart: Path
    output_directory: Path
    run_seconds: float
    plan: Mapping | None = None
    write_history_files: bool = True
    storage: str = "device"

    def __post_init__(self):
        if isinstance(self.member_id, bool) or int(self.member_id) < 1:
            raise MemberSessionError("a member is named by its global, 1-based member ID")
        if self.storage not in ("device", "managed"):
            raise MemberSessionError(f"member storage {self.storage!r}: device or managed")
        self.member_id = int(self.member_id)
        self.run_directory = Path(self.run_directory)
        self.restart = Path(self.restart)
        self.output_directory = Path(self.output_directory)


@dataclass
class ResidentMember:
    spec: MemberSpec
    inputs: object
    model: object
    base_initialization: object
    analysis: object | None
    binding: object
    writers: object
    publisher: object
    sink: object
    storage: object | None
    checkpoints: dict = field(default_factory=dict)
    receipts: list = field(default_factory=list)
    clocks: dict = field(default_factory=dict)

    @property
    def root(self):
        return self.model.root


def _add(clock: dict, name: str, seconds: float) -> None:
    clock[name] = clock.get(name, 0.0) + float(seconds)


class MemberSession:
    """Every member of an ensemble in one process, carried across cycles.

    ``cadence`` is one of :data:`CADENCES`.  ``frame_fields`` is the frame
    sink's field set (``None``: every field the writer writes).
    ``synchronize`` waits for the card (``cupy.cuda.Device().synchronize``
    by default); it is called at every member's turn boundary so a turn's
    time is its own.
    """

    def __init__(self, *, cadence: str = "requested", frame_fields=None, frames_kept: int = 2,
                 synchronize: Callable | None = None, resident_storage=None):
        """``resident_storage`` (:class:`gpuwm.ensemble.member_storage.ResidentMemberStorage`)
        builds every member into one card-wide shared workspace under its residency plan:
        a member's storage owner is the storage's adapter (its turn is the workspace turn;
        a pinned member is never parked) and the door's initialization is wrapped by
        :class:`gpuwm.ensemble.workspace_initialization.WorkspaceInitialization`.
        ``None`` keeps each member's own ``MemberSpec.storage``."""
        from gpuwm.io.frame_sink import THIN_FIELDS

        if cadence not in CADENCES:
            raise MemberSessionError(f"checkpoint cadence {cadence!r}: one of {list(CADENCES)}")
        self.cadence = cadence
        self.frame_fields = THIN_FIELDS if frame_fields is None else (
            None if frame_fields == "all" else tuple(frame_fields))
        self.frames_kept = int(frames_kept)
        if synchronize is None:
            import cupy as cp
            synchronize = cp.cuda.Device().synchronize
        self.synchronize = synchronize
        self.members: dict[int, ResidentMember] = {}
        self._requested: dict[int, set] = {}
        self.poisoned = False
        self.closed = False
        self._seeded = False
        self.resident_storage = resident_storage

    # ------------------------------------------------------------------ build
    def add_member(self, spec: MemberSpec) -> dict:
        """Build one member through the door's construction and keep it."""
        if self.closed or self.poisoned:
            raise MemberSessionError("the session is closed or poisoned")
        if spec.member_id in self.members:
            raise MemberSessionError(f"member {spec.member_id} is already in the session")
        started = perf_counter()
        clock: dict = {}
        if not self._seeded:
            from gpuwm import kernel_cache_seed
            kernel_cache_seed.seed()
            self._seeded = True
        from gpuwm.wrfinput_door import resolve_wrfinput_run
        from gpuwm.wrfinput_forecast import WrfInitialization, prepare_wrf_run
        from gpuwm.prepared_domain_tree_forecast import run_prepared_tree
        from gpuwm.da.analysis_door import AnalysisInitialization

        out = spec.output_directory
        out.mkdir(parents=True, exist_ok=True)
        t0 = perf_counter()
        run = resolve_wrfinput_run(spec.run_directory)
        inputs = prepare_wrf_run(run, out / "input", run_seconds=spec.run_seconds)
        _add(clock, "prepare_s", perf_counter() - t0)
        base = WrfInitialization(inputs)
        analysis = (None if spec.plan is None else
                    AnalysisInitialization(base, spec.plan, outdir=out / "analysis" / _stamp_of_plan(spec.plan)))
        if analysis is not None:
            analysis.outdir.mkdir(parents=True, exist_ok=True)
        storage = None
        if self.resident_storage is not None:
            storage = self.resident_storage.owner(spec.member_id)
        elif spec.storage == "managed":
            from gpuwm.ensemble.managed_storage import ManagedStorage
            storage = ManagedStorage()

        def door_initialization():
            chosen = analysis if analysis is not None else base
            if self.resident_storage is None:
                return chosen
            return self.resident_storage.member(spec.member_id).initialization(chosen)
        captured: dict = {}

        def capture(**handed):
            captured.update(handed)
            return {"status": "PASS", "member_session": "captured at the ensemble bootstrap seam"}

        t0 = perf_counter()
        from gpuwm.core.resident_admission import memory_gate_override
        from gpuwm.ingest.memory_refusal import ResidentMemoryRefused
        import cupy as cp

        # The door admits one forecast against the card's free memory with its peak ENVELOPE (state, physics,
        # boundary tables, workspaces, step transients and the CUDA context).  In a session the context and the
        # step transients are shared, so the envelope prices a member that joins resident ones too high.  The
        # first member (and every member while nothing else is resident) is admitted by the door as a files leg
        # is; a later one the envelope refuses is built under the door's own --no-memory-gate (the loaders' exact
        # floors still hold), and the receipt says so.  A managed member's pages are the host's until its turn, so
        # the device envelope does not price it.  Cached, unused pool blocks are released first (byte-inert).
        cp.get_default_memory_pool().free_all_blocks()
        admission = "door envelope"
        override = storage is not None
        if override:
            admission = "managed member: device envelope not applied (pages parked between turns)"
        for attempt in (0, 1):
            try:
                with memory_gate_override(override), \
                        (storage.allocate() if storage is not None else nullcontext()):
                    run_prepared_tree(inputs, output_directory=out, io_mode="history", restart=spec.restart,
                                      initialization=door_initialization(),
                                      ensemble_bootstrap=capture)
                break
            except ResidentMemoryRefused as refused:
                if attempt or override or not self.members:
                    raise
                override = True
                admission = (f"door envelope refused with {len(self.members)} member(s) resident "
                             f"({str(refused).splitlines()[0][:200]}); built under the door's --no-memory-gate")
                if analysis is not None:
                    # the refusal came before anything was allocated; the plan's ledger is rewritten by the retry
                    analysis = AnalysisInitialization(base, spec.plan, outdir=analysis.outdir)
        clock["admission"] = admission
        with (storage.allocate() if storage is not None else nullcontext()):
            model = captured.get("model")
            if model is None:
                raise MemberSessionError(f"member {spec.member_id}: the door handed no model over")
            nodes = tuple(model.walk_parent_first())
            if len(nodes) != 1:
                raise MemberSessionError(f"member {spec.member_id}: a member session carries single-domain "
                                         "members (an analysis into nests is the door's named gap G7)")
            member = self._bind(spec, inputs, model, base, analysis, storage)
        self.synchronize()
        _add(clock, "restore_s", perf_counter() - t0)
        self.members[spec.member_id] = member
        self._requested.setdefault(spec.member_id, set())
        if storage is not None:
            t0 = perf_counter()
            row = storage.park()
            _add(clock, "park_s", perf_counter() - t0)
            member.receipts.append({"event": "park", **row})
        clock["build_s"] = perf_counter() - started
        member.clocks["build"] = clock
        receipt = {"member_id": spec.member_id, "ticks": int(model.root.clock.ticks),
                   "valid_time": _wrf(model, model.root.clock.ticks), "storage": spec.storage,
                   "analysis": None if analysis is None else spec.plan["bundle"]["manifest_sha256"],
                   "clock": clock}
        member.receipts.append({"event": "build", **receipt})
        return receipt

    def _bind(self, spec, inputs, model, base, analysis, storage) -> ResidentMember:
        """The door's history writer, checkpoint writer and executor closures for one member."""
        from gpuwm import runtime
        from gpuwm.core import streaming
        from gpuwm.ensemble.prepared_nested_batch import capture_original_member_schedule
        from gpuwm.io.frame_sink import FrameSink
        from gpuwm.io.restart import write_tree_restart
        from gpuwm.io.wrfout import PerDomainWrfoutWriters
        from gpuwm.prepared_single_domain_forecast import _CheckpointPublisher

        exp = inputs.experiment
        out = spec.output_directory
        writers = PerDomainWrfoutWriters(
            model, out / "wrfout", start_time=exp.start_time,
            title=f"gpuwm prepared {inputs.source.upper()} domain tree {exp.name}",
            initial_condition=inputs.source_identity.get("initial_condition"),
            source=inputs.source, history_selection=exp.output,
            simulated_radar=exp.simulated_radar, radar_output_dir=out)
        sink = FrameSink(spec.member_id, fields=self.frame_fields, keep=self.frames_kept)
        writers.attach_frame_sink(sink, write_files=spec.write_history_files)
        model._io_manager = writers
        publisher = _CheckpointPublisher()
        member_ref: dict = {}

        def history_handler(_tree, node, ticks):
            started = perf_counter()
            # The door's handler: the stability sample, then the frame.
            streaming.stability_observer(None)(node.state, node.cfg.run,
                                               boundary_width=node.cfg.run.spec_bdy_width)
            runtime._submit_tree_history_frame(writers, node, ticks)
            _add(member_ref["m"].clocks.setdefault("turn", {}), "history_submit_s", perf_counter() - started)

        def restart_handler(tree, ticks):
            member = member_ref["m"]
            if not self._checkpoint_wanted(member, ticks):
                return
            started = perf_counter()
            publisher.wait()
            valid = exp.start_time + timedelta(seconds=ticks / tree.schedule.clock.tick_den)
            pending = write_tree_restart(out, tree, valid, sealed_forcing_extension=False, defer_publish=True)
            history_durable = writers.durability_barrier()

            def publish(retire=True):
                if history_durable is None:
                    writers.drain()
                else:
                    history_durable()
                path = pending.publish(retire=retire)
                member.checkpoints[int(ticks)] = Path(tree._last_checkpoint if path is None else path)

            if pending.deferrable and history_durable is not None:
                # Superseded sets are retired on this thread at the next wait
                # (see prepared_single_domain_forecast._CheckpointPublisher).
                publisher.submit(lambda: publish(retire=False),
                                 then=pending.retire_superseded)
            else:
                try:
                    publish()
                except BaseException:
                    pending.abandon()
                    raise
            _add(member.clocks.setdefault("turn", {}), "checkpoint_stall_s", perf_counter() - started)

        restart_handler.orders_history_itself = True
        restart_handler.wait_durable = publisher.wait

        def step_observer(**event):
            analysis_now = member_ref["m"].analysis
            observe = getattr(analysis_now, "step_observer", None)
            if callable(observe):
                observe(**event)

        options = dict(history_handler=history_handler, restart_handler=restart_handler,
                       progress_callback=None, validate_state=True, health_debug=False,
                       skip_feedback_path=(int(exp.feedback) == 0), relocation_runner=None,
                       steppers=None, step_observer=step_observer, experiment=exp,
                       delayed_child_initializer=None)
        binding = capture_original_member_schedule(
            spec.member_id, model, operation_authority=lambda **_: ("original-member-only",),
            execution_options=options)
        member = ResidentMember(spec=spec, inputs=inputs, model=model, base_initialization=base,
                                analysis=analysis, binding=binding, writers=writers, publisher=publisher,
                                sink=sink, storage=storage)
        member_ref["m"] = member
        return member

    # ------------------------------------------------------------------ turns
    @contextmanager
    def turn(self, member_id: int, *, park: bool = True):
        """One member's exclusive use of the card: resumed if parked, its
        allocations routed to its own owner, parked again on exit."""
        member = self._member(member_id)
        clock = member.clocks.setdefault("turn", {})
        owner = member.storage
        if owner is not None and owner.phase == "host":
            t0 = perf_counter()
            row = owner.resume()
            _add(clock, "resume_s", perf_counter() - t0)
            member.receipts.append({"event": "resume", **row})
        try:
            with (owner.allocate() if owner is not None else nullcontext()):
                yield member
            self.synchronize()
        except BaseException:
            self.poisoned = True
            raise
        if owner is not None and park:
            t0 = perf_counter()
            row = owner.park()
            _add(clock, "park_s", perf_counter() - t0)
            member.receipts.append({"event": "park", **row})

    def _member(self, member_id) -> ResidentMember:
        if self.poisoned:
            raise MemberSessionError("the session is poisoned: a member's turn failed part-way, so its state "
                                     "is no longer one the files path could have produced")
        try:
            return self.members[int(member_id)]
        except KeyError:
            raise MemberSessionError(f"member {member_id} is not in the session") from None

    # ------------------------------------------------------------------ the cycle's acts (inside a turn)
    def apply(self, member_id: int, plan: Mapping, *, outdir: Path | None = None) -> dict:
        """The next analysis on the live member at its instant: the previous
        window retired with the door's G10 check, then the door's own act."""
        from gpuwm.da.analysis_door import AnalysisInitialization

        member = self._member(member_id)
        clock = member.clocks.setdefault("turn", {})
        t0 = perf_counter()
        retired = None
        if member.analysis is not None:
            retired = member.analysis.retire_window(member.model)
        exp = member.inputs.experiment
        where = Path(outdir) if outdir is not None else (
            member.spec.output_directory / "analysis" / _stamp_of_plan(plan))
        where.mkdir(parents=True, exist_ok=True)
        analysis = AnalysisInitialization(member.base_initialization, plan, outdir=where)
        analysis.apply_door_plan(model=member.model, start_time=exp.start_time,
                                 run_seconds=float(exp.run_seconds), continuation="in-memory")
        member.analysis = analysis
        self.synchronize()
        _add(clock, "apply_s", perf_counter() - t0)
        row = {"event": "apply", "ticks": int(member.root.clock.ticks),
               "valid_time": _wrf(member.model, member.root.clock.ticks),
               "bundle": plan["bundle"]["manifest_sha256"], "retired": retired}
        member.receipts.append(row)
        return row

    def advance(self, member_id: int, target_ticks: int) -> dict:
        """Step one member to ``target_ticks`` with its run's own schedule."""
        from gpuwm.core.clock import execute_schedule

        member = self._member(member_id)
        if isinstance(target_ticks, bool) or not isinstance(target_ticks, int):
            raise MemberSessionError("a target is an integer clock tick")
        binding = member.binding
        schedule, clocks = binding.model.schedule, binding.clocks
        root = member.root.clock
        if target_ticks < root.ticks or target_ticks > schedule.clock.run_ticks:
            raise MemberSessionError(f"member {member_id}: target {target_ticks} is outside "
                                     f"[{root.ticks}, {schedule.clock.run_ticks}]")
        if root.ticks == target_ticks:
            return {"member_id": member_id, "steps": 0, "seconds": 0.0}
        t0 = perf_counter()
        callbacks = {"on_" + name: fn for name, fn in binding.callbacks.items()}
        report = binding.context.run(
            execute_schedule, schedule, clocks=clocks,
            start_period=root.ticks // schedule.period_ticks,
            started_grid_ids=tuple(node.cfg.grid_id for node in binding.model.walk_parent_first()
                                   if node._started),
            # The frame at the instant the segment starts was written when the
            # previous segment (or the run the checkpoint came from) ended.
            committed_initial_history_grid_ids=tuple(clocks) if root.ticks else (),
            skip_feedback_path=binding.skip_feedback_path,
            on_period_steps=binding.adaptive_driver, stop_ticks=target_ticks, **callbacks)
        self.synchronize()
        seconds = perf_counter() - t0
        clock = member.clocks.setdefault("turn", {})
        _add(clock, "advance_s", seconds)
        return {"member_id": member_id, "steps": int(report.steps), "seconds": seconds,
                "ticks": int(root.ticks)}

    # ------------------------------------------------------------------ outputs
    def frame(self, member_id: int, ticks: int | None = None, *, timeout: float = 600.0):
        """The member's history frame at ``ticks`` (default: now), once landed."""
        member = self._member(member_id)
        ticks = member.root.clock.ticks if ticks is None else int(ticks)
        return member.sink.frame(_wrf(member.model, ticks), timeout=timeout)

    def history_path(self, member_id: int, ticks: int | None = None) -> Path:
        from gpuwm.io.wrfout import wrfout_filename
        member = self._member(member_id)
        ticks = member.root.clock.ticks if ticks is None else int(ticks)
        valid = member.inputs.experiment.start_time + timedelta(
            seconds=ticks / member.model.schedule.clock.tick_den)
        return member.spec.output_directory / "wrfout" / wrfout_filename(valid, 1)

    def views(self, member_id: int, names) -> dict:
        """Read-only device views of the member's live state (native strides, no copies)."""
        import cupy as cp
        from gpuwm.ensemble.device_exchange import DeviceField

        member = self._member(member_id)
        state = member.root.state
        device = int(cp.cuda.Device().id)
        out = {}
        for name in names:
            array = getattr(state, name, None)
            if array is None:
                raise MemberSessionError(f"member {member_id} carries no {name}")
            out[name] = DeviceField(array, writable=False, shared=True, members=1, device=device)
        return out

    def request_checkpoint(self, member_id: int, ticks: int) -> None:
        """Ask for a checkpoint at a restart instant the member will reach."""
        self._member(member_id)
        self._requested[int(member_id)].add(int(ticks))

    def _checkpoint_wanted(self, member, ticks) -> bool:
        if self.cadence == "every-instant":
            return True
        if self.cadence == "none":
            return False
        return int(ticks) in self._requested.get(member.spec.member_id, ())

    def drain(self, member_id: int | None = None) -> None:
        """Every queued history frame landed and every checkpoint durable."""
        ids = list(self.members) if member_id is None else [int(member_id)]
        for i in ids:
            member = self.members[i]
            member.writers.drain()
            member.publisher.wait()

    def checkpoint(self, member_id: int, ticks: int | None = None) -> Path | None:
        member = self._member(member_id)
        member.publisher.wait()
        ticks = member.root.clock.ticks if ticks is None else int(ticks)
        return member.checkpoints.get(int(ticks))

    def storage_receipt(self) -> dict:
        rows = {}
        for i, member in self.members.items():
            owner = member.storage
            rows[i] = ({"storage": "device"} if owner is None else
                       {"storage": "managed", **owner.inventory()})
        return rows

    def receipt(self) -> dict:
        return {"schema": SCHEMA, "cadence": self.cadence,
                "members": {i: {"storage": m.spec.storage, "ticks": int(m.root.clock.ticks),
                                "checkpoints": {str(k): str(v) for k, v in sorted(m.checkpoints.items())},
                                "clocks": m.clocks, "events": m.receipts,
                                "frames_seen": m.sink.frames_seen, "frame_bytes_copied": m.sink.copied_bytes}
                            for i, m in sorted(self.members.items())}}

    def close(self, *, retire: bool = True) -> dict:
        """Every frame and checkpoint durable, every window retired, writers closed."""
        if self.closed:
            return self.receipt()
        failures = []
        for i, member in sorted(self.members.items()):
            try:
                if retire and member.analysis is not None and not self.poisoned:
                    with self.turn(i):
                        member.receipts.append({"event": "retire",
                                                "retired": member.analysis.retire_window(member.model)})
                        member.analysis = None
            except BaseException as error:  # noqa: BLE001 - every member is closed; the first failure is raised
                failures.append((i, error))
            for act in (member.writers.drain, member.publisher.wait, member.writers.close):
                try:
                    act()
                except BaseException as error:  # noqa: BLE001
                    failures.append((i, error))
            member.model._io_manager = None
        self.closed = True
        if failures:
            i, error = failures[0]
            raise MemberSessionError(f"member {i}: closing failed: {type(error).__name__}: {error}") from error
        return self.receipt()


def _wrf(model, ticks) -> str:
    valid = model.schedule.clock.start_time + timedelta(seconds=int(ticks) / model.schedule.clock.tick_den)
    return valid.strftime("%Y-%m-%d_%H:%M:%S")


def _stamp_of_plan(plan) -> str:
    """``YYYYmmddHHMMSS`` of the plan's valid time: one folder per analysis."""
    import re
    return re.sub(r"\D", "", str(plan["valid_time"]))[:14]


__all__ = ["CADENCES", "MemberSession", "MemberSessionError", "MemberSpec", "ResidentMember", "SCHEMA"]
