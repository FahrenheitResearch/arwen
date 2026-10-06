"""Dispatch ready member operations from the unchanged ordinary schedules.

Metadata workers run the stock integer executor and stop at its callbacks.
All controllers and numerical callbacks execute on the calling thread. Each
member retains its complete clock hierarchy, adaptive controller and CFL
owner. Compatible frontiers may use an explicitly admitted packed callback;
every other operation uses its original callback in the original order.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass, field, fields, is_dataclass
from fractions import Fraction
from hashlib import sha256
from queue import Queue
from threading import Event
from typing import Callable, Mapping

import numpy as np

from gpuwm.core.clock import execute_schedule
from gpuwm.core.cfl_member import member_cfl_scope
from gpuwm.ensemble.physics_execution import MemberPhysicsBinding, compatible_piece_groups
from gpuwm.ensemble.execution import name_failing_members
from gpuwm.ensemble.batch_state import _temporal_key

CONTRACT = "gpuwm-member-original-schedule-dispatch-v1"
_PACKABLE = frozenset({"step", "force", "feedback_commit", "feedback_finalize"})
_CLOCK_WORDS = ("ticks", "step_ticks", "tick_den", "run_ticks", "step_count",
                "dt_fp32", "dtbc_fp32")
_PHYSICS_PIECES = ("microphysics", "radiation", "surface_layer", "land_surface", "pbl", "cumulus")


def _exact(value):
    if isinstance(value, np.generic):
        return (value.dtype.str, value.tobytes())
    calendar = _temporal_key(value)
    if calendar is not None:
        return calendar
    if isinstance(value, float):
        return ("float64", np.float64(value).tobytes())
    if isinstance(value, Fraction):
        return ("fraction", value.numerator, value.denominator)
    if value is None or isinstance(value, (bool, int, str, bytes)):
        return (type(value).__name__, value)
    if isinstance(value, Mapping):
        return tuple((str(name), _exact(item)) for name, item in sorted(value.items(), key=lambda row: str(row[0])))
    if isinstance(value, (tuple, list)):
        return tuple(_exact(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return (type(value).__qualname__, tuple((row.name, _exact(getattr(value, row.name)))
            for row in fields(value) if row.init and not row.name.startswith("_")))
    raise TypeError(f"schedule grouping authority has no exact metadata representation for {type(value).__qualname__}")


def _clock_key(clock, *, controller_history=True):
    result = tuple((name, _exact(getattr(clock, name))) for name in _CLOCK_WORDS) + (
        ("calendar", _exact(clock.spec)),)
    # The original controller is never packed or substituted. Its last CFL
    # values select the NEXT interval, not this operation. Preserve every
    # other published dependency: before_step reads baseline dt and producer
    # histories to derive the current physics cadence after STEP admission.
    history = clock.adaptive_state
    if not controller_history and isinstance(history, Mapping):
        history = {name: value for name, value in history.items()
                   if name not in {"last_max_vert_cfl", "last_max_horiz_cfl"}}
    return result + (("adaptive_state", _exact(history)),)


def _array_layouts(state):
    """Describe live field ownership/layout without downloading field words."""
    rows, seen = [], set()
    def walk(value, path):
        if hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "strides"):
            rows.append((path, tuple(value.shape), np.dtype(value.dtype).str, tuple(value.strides)))
            return
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, Mapping):
            for name, item in sorted(value.items(), key=lambda row: str(row[0])):
                walk(item, f"{path}/{name}")
        elif is_dataclass(value) and not isinstance(value, type):
            for row in fields(value):
                walk(getattr(value, row.name), f"{path}/{row.name}")
        elif isinstance(value, (tuple, list)):
            for number, item in enumerate(value):
                walk(item, f"{path}/{number}")
        elif hasattr(value, "__dict__") and type(value).__module__.startswith(("gpuwm.", "types")):
            for name, item in sorted(vars(value).items()):
                if name not in {"cfg", "state", "_host_setup_state"}:
                    walk(item, f"{path}/{name}")
    walk(state, "state")
    return tuple(rows)


def _scalar_controls(owner):
    if owner is None or not hasattr(owner, "__dict__"):
        return ()
    return tuple((name, _exact(value)) for name, value in sorted(vars(owner).items())
        if value is None or isinstance(value, (bool, int, float, str, bytes, Fraction, np.generic)))


def _edge_key(node):
    coupler = getattr(node, "coupler", None)
    if coupler is None:
        return None
    geometry = []
    for stagger, reg in sorted(getattr(coupler, "registrations", {}).items()):
        geometry.append((stagger, tuple((name, _exact(getattr(reg, name))) for name in (
            "nri", "nrj", "i_parent_start", "j_parent_start", "nxc", "nyc", "nxp", "nyp",
            "xstag", "ystag", "wrapper")), tuple((name, sha256(getattr(reg, name).tobytes()).hexdigest())
            for name in ("ci", "ip", "cj", "jp", "xig", "xjg"))))
    return (tuple(geometry), tuple((name, _exact(getattr(coupler, name, None))) for name in (
        "feedback", "smooth_option", "placement_generation", "generation", "force_count", "feedback_count")))


@dataclass
class MemberScheduleBinding:
    """A live ordinary model and its original callback/controller owners.

    ``operation_authority`` is the qualified graph's immutable ownership and
    dependency descriptor for this operation. It must include any source,
    grid, physics-table or edge authority not represented by the live config
    and field layout. Equality permits consideration of a group, not launch
    admission. ``step`` excludes adaptive ``before_step``; that original
    driver hook is dispatched separately before a STEP request is formed.
    """
    member_id: int
    model: object
    callbacks: Mapping[str, Callable]
    operation_authority: Callable
    adaptive_driver: object | None = None
    start_period: int = 0
    started_grid_ids: object = None
    committed_initial_history_grid_ids: tuple = ()
    skip_feedback_path: bool = False
    step_owns_before_step: bool = False
    step_bindings: dict | None = None
    context: object = field(init=False, repr=False)
    cfl_owner: object = field(init=False, repr=False)

    def __post_init__(self):
        if isinstance(self.member_id, bool) or not isinstance(self.member_id, int) or self.member_id < 0:
            raise ValueError("scheduled member ID must be a nonnegative integer")
        if not callable(self.operation_authority):
            raise TypeError("scheduled member needs a qualified full-operation ownership authority")
        self.callbacks = dict(self.callbacks)
        if not callable(self.callbacks.get("step")):
            raise ValueError("scheduled member needs its original STEP callback; clock advancement alone is not a forecast")
        if any(node.parent is not None for node in self.model.walk_parent_first()) and not callable(self.callbacks.get("force")):
            raise ValueError("scheduled nested member needs its original FORCE callback before child stepping")
        with member_cfl_scope() as owner:
            self.context = copy_context()
            self.cfl_owner = owner
        if self.adaptive_driver is not None:
            self.cfl_owner.enabled = True

    @property
    def clocks(self):
        return {int(node.cfg.grid_id): node.clock for node in self.model.walk_parent_first()}


@dataclass
class ReadyMemberOperation:
    binding: MemberScheduleBinding
    kind: str
    args: tuple
    sequence: int
    key: object
    done: Event = field(default_factory=Event, repr=False)
    result: object = None
    error: BaseException | None = field(default=None, repr=False)

    @property
    def member_id(self):
        return self.binding.member_id

    @property
    def grid_id(self):
        return int(self.args[0]) if self.kind in {"step", "before_step", "force", "feedback_prepare",
            "feedback_commit", "feedback_finalize", "history", "domain_start"} else None


@dataclass(frozen=True)
class ReadyOperationGroup:
    requests: tuple[ReadyMemberOperation, ...]

    @property
    def member_ids(self):
        return tuple(request.member_id for request in self.requests)

    @property
    def kind(self):
        return self.requests[0].kind


@dataclass(frozen=True)
class ScheduleBatchAdmission:
    eligible: bool
    reason: str
    cfl_binding: str | None = None


def _operation_key(binding, kind, args):
    clocks = binding.clocks
    if any(clock.tick_den != binding.model.schedule.clock.tick_den for clock in clocks.values()):
        raise ValueError("member domain clock lattice differs from its original schedule authority")
    grid = int(args[0]) if kind in {"step", "before_step", "force", "feedback_prepare",
        "feedback_commit", "feedback_finalize", "history", "domain_start"} else None
    if kind in {"step", "before_step", "domain_start"} and args[1] is not clocks[grid]:
        raise ValueError("member operation clock identity differs from its original model owner")
    if kind in {"force", "feedback_prepare", "feedback_commit", "feedback_finalize"}:
        if args[2] is not clocks[grid] or args[3] is not clocks[int(args[1])]:
            raise ValueError("member edge clocks differ from their original model owners")
    if kind == "history" and int(args[1]) != int(clocks[grid].ticks):
        raise ValueError("member history deadline differs from its original live clock")
    if kind.startswith("period_") and any(args[1][gid] is not clock for gid, clock in clocks.items()):
        raise ValueError("member period clocks differ from their original model owners")
    affected = (() if grid is None else (grid, int(args[1])) if kind in {
        "force", "feedback_prepare", "feedback_commit", "feedback_finalize"} else (grid,))
    if not affected:
        affected = tuple(clocks)
    nodes = tuple(binding.model.node(gid) for gid in affected)
    return (kind, grid, tuple((gid, _clock_key(clocks[gid], controller_history=kind not in _PACKABLE)) for gid in affected),
        tuple((_exact(node.cfg), _array_layouts(node.state), _scalar_controls(node.state),
               _scalar_controls(getattr(node.state, "physics", None)), _edge_key(node)) for node in nodes),
        _exact(binding.operation_authority(kind=kind, args=args, nodes=nodes)))


def ready_operation_groups(requests):
    """Group a blocked frontier using complete operation/time authorities."""
    requests = tuple(requests)
    groups = {}
    for request in requests:
        groups.setdefault(request.key, []).append(request)
    result = []
    for candidates in groups.values():
        if candidates[0].kind != "step":
            result.append(ReadyOperationGroup(tuple(candidates)))
            continue
        # Reuse all existing original physics cadence/selector partitions.
        # Full STEP launches require compatibility across every component.
        bindings = tuple(MemberPhysicsBinding(request.binding.model.node(request.grid_id).state,
            request.binding.model.node(request.grid_id).cfg.run,
            getattr(request.binding.model.node(request.grid_id).state, "physics", None), request.member_id,
            config_provider=lambda request=request: request.binding.model.node(request.grid_id).cfg.run)
            for request in candidates)
        labels = [[] for _ in candidates]
        for component in _PHYSICS_PIECES:
            for number, indices in enumerate(compatible_piece_groups(bindings, component)):
                for index in indices:
                    labels[index].append(number)
        partitions = {}
        for request, label in zip(candidates, labels, strict=True):
            partitions.setdefault(tuple(label), []).append(request)
        result.extend(ReadyOperationGroup(tuple(group)) for group in partitions.values())
    return tuple(result)


def execute_packed_member_schedules(bindings, *, admitted_callbacks=None, batch_admission=None,
                                    completion_wait, record_callback=None, original_group_dispatch=None):
    """Run original schedules with a real admitted packed callback dispatch.

    Workers do no CUDA work and park at one callback each. The central
    thread snapshots that complete frontier, executes every ready group,
    then releases the workers to their next original operation. Controllers
    and output callbacks retain member order. A packed STEP requires an
    explicit per-member CFL-ring binding when recording is active. Callback
    failures wake and join every parked worker; no mutated operation retries.
    ``completion_wait`` must finish all queues owned by the supplied callbacks
    before their private CFL banks are retired, including on failure.
    An optional original group dispatcher may run unadmitted numerical
    callbacks on their existing member workers. It must wake and join every
    submitted callback before returning or raising. Metadata and controllers
    retain the original member order, and this path is always unpacked.
    """
    bindings = tuple(bindings)
    if (not bindings or any(not isinstance(binding, MemberScheduleBinding) for binding in bindings)
            or len({binding.member_id for binding in bindings}) != len(bindings)):
        raise ValueError("schedule execution needs unique live member bindings")
    for label, owners in (("model", [binding.model for binding in bindings]),
            ("adaptive controller", [binding.adaptive_driver for binding in bindings if binding.adaptive_driver is not None]),
            ("domain clock", [clock for binding in bindings for clock in binding.clocks.values()]),
            ("mutable state", [node.state for binding in bindings for node in binding.model.walk_parent_first()
                               if node.state is not None])):
        if len({id(owner) for owner in owners}) != len(owners):
            raise ValueError(f"scheduled members cannot share an original {label} owner")
    if not callable(completion_wait):
        raise TypeError("schedule execution needs its actual callback queue completion owner")
    if original_group_dispatch is not None and not callable(original_group_dispatch):
        raise TypeError("original group dispatch must own callable member callbacks and their joins")
    admitted_callbacks = {} if admitted_callbacks is None else dict(admitted_callbacks)
    if set(admitted_callbacks) - _PACKABLE:
        raise ValueError("only numerical schedule operations can use a packed callback")
    if admitted_callbacks and not callable(batch_admission):
        raise TypeError("packed schedule callbacks need explicit complete-group admission")
    queue, stop = Queue(), Event()
    live, reports, operations, frontier = {binding.member_id for binding in bindings}, {}, [], {}
    terminated = set()
    failure = [None]

    def worker(binding):
        sequence = 0
        def submit(kind, *args):
            nonlocal sequence
            if stop.is_set():
                raise failure[0] or RuntimeError("member schedule dispatch stopped")
            request = ReadyMemberOperation(binding, kind, tuple(args), sequence, None)
            sequence += 1
            queue.put(request)
            request.done.wait()
            if request.error is not None:
                raise request.error
            return request.result
        callbacks = {"on_" + name: lambda *args, name=name: submit(name, *args)
            for name in ("step", "force", "feedback_prepare", "feedback_commit", "feedback_finalize",
                "history", "restart", "lbc_reset", "period_begin", "period_end", "period_commit", "domain_start")}
        def step(grid_id, clock):
            if ((binding.adaptive_driver is not None and not binding.step_owns_before_step)
                    or "before_step" in binding.callbacks):
                submit("before_step", grid_id, clock)
            return submit("step", grid_id, clock)
        callbacks["on_step"] = step
        try:
            report = execute_schedule(binding.model.schedule, clocks=binding.clocks,
                start_period=binding.start_period, started_grid_ids=binding.started_grid_ids,
                committed_initial_history_grid_ids=binding.committed_initial_history_grid_ids,
                skip_feedback_path=binding.skip_feedback_path,
                on_period_steps=(None if binding.adaptive_driver is None else
                    lambda *args: submit("period_steps", *args)), **callbacks)
            queue.put((binding.member_id, report, None))
        except BaseException as error:
            queue.put((binding.member_id, None, error))

    def original(request):
        binding = request.binding
        def call():
            if request.kind == "period_steps":
                return binding.adaptive_driver(*request.args)
            if request.kind == "before_step" and binding.adaptive_driver is not None:
                binding.adaptive_driver.before_step(request.grid_id)
            callback = binding.callbacks.get(request.kind)
            return None if callback is None else callback(*request.args)
        try:
            return binding.context.run(call)
        except BaseException as error:
            name_failing_members(error, (request.member_id,), execution_mode="original_member_schedule")
            raise

    pool = ThreadPoolExecutor(max_workers=len(bindings), thread_name_prefix="ensemble-clock")
    futures = [pool.submit(copy_context().run, worker, binding) for binding in bindings]
    try:
        while live:
            while set(frontier) != live:
                item = queue.get()
                if isinstance(item, ReadyMemberOperation):
                    frontier[item.member_id] = item
                else:
                    member, report, error = item
                    live.remove(member)
                    terminated.add(member)
                    reports[member] = report
                    if error is not None:
                        raise error
            requests = tuple(frontier[binding.member_id] for binding in bindings if binding.member_id in live)
            # Every worker is parked now. Capture live clocks and operation
            # authorities centrally, including any graph admission query.
            for request in requests:
                try:
                    request.key = request.binding.context.run(_operation_key, request.binding, request.kind, request.args)
                except BaseException as error:
                    name_failing_members(error, (request.member_id,), execution_mode="member_schedule_admission")
                    raise
            for group in ready_operation_groups(requests):
                callback = admitted_callbacks.get(group.kind)
                admission = None if callback is None or len(group.requests) == 1 else batch_admission(group)
                if admission is not None and not isinstance(admission, ScheduleBatchAdmission):
                    raise TypeError("schedule group admission must carry its actual binding reason")
                packed = admission is not None and admission.eligible
                if packed and group.kind == "step" and any(request.binding.cfl_owner.enabled for request in group.requests):
                    if admission.cfl_binding != "member_owned_rings":
                        raise ValueError("packed adaptive STEP has no admitted private member CFL-ring read/write binding")
                if packed:
                    try:
                        results = tuple(callback(group))
                    except BaseException as error:
                        name_failing_members(error, group.member_ids, execution_mode="packed_member_operation")
                        raise
                    if len(results) != len(group.requests):
                        raise ValueError("packed schedule operation returned an incomplete member roster")
                else:
                    if original_group_dispatch is not None and group.kind in _PACKABLE:
                        try:
                            results = tuple(original_group_dispatch(group))
                        except BaseException as error:
                            name_failing_members(error, group.member_ids, execution_mode="original_member_group")
                            raise
                        if len(results) != len(group.requests):
                            raise ValueError("original group operation returned an incomplete member roster")
                    else:
                        results = tuple(original(request) for request in group.requests)
                for request, result in zip(group.requests, results, strict=True):
                    request.result = result
                row = {"kind": group.kind, "member_ids": list(group.member_ids), "packed": packed,
                       "admission_reason": None if admission is None else admission.reason}
                if not packed and original_group_dispatch is not None and group.kind in _PACKABLE:
                    row["original_dispatch"] = "group_callback"
                    if row["admission_reason"] is None:
                        row["admission_reason"] = (
                            "no qualified packed callback owns this numerical operation" if callback is None else
                            "one ready member cannot use an all-member packed binding")
                operations.append(row)
                if record_callback is not None:
                    record_callback(group, row)
            for request in frontier.values():
                request.done.set()
            frontier.clear()
    except BaseException as error:
        failure[0] = error
        stop.set()
        for request in frontier.values():
            request.error = error
            request.done.set()
        # A worker can have queued its next callback while another member's
        # original clock walk failed. Drain and release every waiter.
        while len(terminated) != len(bindings):
            item = queue.get()
            if isinstance(item, ReadyMemberOperation):
                item.error = error
                item.done.set()
            else:
                terminated.add(item[0])
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        try:
            try:
                completion_wait()
            except BaseException as cleanup_error:
                if failure[0] is None:
                    raise
                failure[0].add_note(f"schedule callback queue completion also failed: {type(cleanup_error).__name__}: {cleanup_error}")
        finally:
            for binding in bindings:
                for bank in binding.cfl_owner.banks.values():
                    bank.clear()
    return {"contract": CONTRACT, "member_order": [binding.member_id for binding in bindings],
            "reports": reports, "operations": operations,
            "clock_policy": "independent_original_execute_schedule_and_controllers",
            "numerical_callback_thread": "caller", "forecast_graph_admission_changed": False}


__all__ = ["CONTRACT", "MemberScheduleBinding", "ReadyMemberOperation", "ReadyOperationGroup",
           "ScheduleBatchAdmission", "ready_operation_groups", "execute_packed_member_schedules"]
