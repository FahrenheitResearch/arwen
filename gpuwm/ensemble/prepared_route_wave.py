"""Explicitly priced component waves inside genuine ordinary runner lifetimes.

Each runner retains its writers, controllers, CUDA allocator and stream on
its own worker. Original callbacks are RPCs to that parked worker. Native
bank owners may join qualified calls after complete cold admission without
replacing ordinary finalization.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from contextvars import copy_context
from dataclasses import dataclass, field
import inspect
from operator import index
from queue import Queue, Empty
from types import MappingProxyType, SimpleNamespace
import threading

import numpy as np

from gpuwm.core.cfl_member import current_cfl_member
from gpuwm.core.clock import execute_schedule
from gpuwm.ensemble.batch_storage import BatchMemoryPlan
from gpuwm.ensemble.batch_state import BatchStateUnsupported, SHARED_STATE_CANDIDATES
from gpuwm.ensemble.execution import name_failing_members
from gpuwm.ensemble.packed_schedule import MemberScheduleBinding
from gpuwm.ensemble.runtime_context import member_output_scope

CONTRACT = "gpuwm-prepared-route-component-wave-v1"


def _bytes(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    value = index(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


@dataclass(frozen=True)
class RouteMember:
    member_id: int
    inputs: object
    output_directory: object
    capture: object
    operation_authority: object
    runner_options: dict = field(default_factory=dict)

    def __post_init__(self):
        _bytes(self.member_id, "member_id")
        if self.capture.member_id != self.member_id or not callable(self.operation_authority):
            raise ValueError("route member needs its own output capture and operation authority")
        if "schedule_dispatch" in self.runner_options:
            raise ValueError("route wave owns the runner's schedule dispatch")
        object.__setattr__(self, "runner_options", MappingProxyType(dict(self.runner_options)))


@dataclass(frozen=True)
class RouteWaveReservation:
    """Caller-derived complete upper bounds, admitted before runner threads.

    Native plans name every bank/edge/physics allocation. Actual live plans
    must be covered before any component allocation. Explicit fixed reserves
    cover collector, stochastic, additional CUDA owners and allocator margin;
    a zero reserve still needs its concrete inactive/not-allocated evidence.
    """
    ordinary_forecast_bytes: dict
    native_plans: dict
    fixed_bytes: dict
    evidence: dict
    available_bytes: int

    def __post_init__(self):
        ordinary = {_bytes(member, "member_id"): _bytes(value, "ordinary forecast")
                    for member, value in self.ordinary_forecast_bytes.items()}
        if not ordinary or any(value == 0 for value in ordinary.values()):
            raise ValueError("reserve every member's complete ordinary forecast")
        plans = dict(self.native_plans)
        if any(not isinstance(plan, BatchMemoryPlan) for plan in plans.values()):
            raise TypeError("native reservations need allocation plans")
        fixed = dict(self.fixed_bytes)
        required = {"collector", "stochastic", "cuda_owners", "allocator_margin"}
        if set(fixed) != required:
            raise ValueError("explicit collector/stochastic/CUDA-owner/allocator reservations are required")
        fixed = {name: _bytes(value, name) for name, value in fixed.items()}
        evidence = dict(self.evidence)
        if any(not evidence.get(name) for name in required | {"ordinary", "native"}):
            raise ValueError("each reservation needs its concrete source or inactive evidence")
        for name, value in (("ordinary_forecast_bytes", ordinary), ("native_plans", plans),
                            ("fixed_bytes", fixed), ("evidence", evidence)):
            object.__setattr__(self, name, MappingProxyType(value))
        object.__setattr__(self, "available_bytes", _bytes(self.available_bytes, "available_bytes"))

    @property
    def required_bytes(self):
        return (sum(self.ordinary_forecast_bytes.values()) + sum(self.fixed_bytes.values())
                + sum(plan.required_bytes(len(self.ordinary_forecast_bytes)) for plan in self.native_plans.values()))

    def admit(self, member_ids, *, require_native=True):
        if set(member_ids) != set(self.ordinary_forecast_bytes):
            raise ValueError("reservation must name every wave member exactly")
        ordinary = sum(self.ordinary_forecast_bytes.values()) + sum(self.fixed_bytes.values())
        if ordinary > self.available_bytes:
            raise MemoryError(f"ordinary route wave requires {ordinary} complete reserved bytes; {self.available_bytes} are available")
        if require_native and self.required_bytes > self.available_bytes:
            return f"native component wave needs {self.required_bytes} complete reserved bytes; {self.available_bytes} are available; the {ordinary}-byte original wave remains admitted"
        return None

    def validate_live_plans(self, actual):
        members = len(self.ordinary_forecast_bytes)
        for name, plan in actual.items():
            reserved = self.native_plans.get(name)
            if reserved is None:
                raise BatchStateUnsupported(f"unpriced live component plan {name}; original members retain their allocations")
            allowed = {row["name"]: row for row in reserved.inventory(members)}
            for row in plan.inventory(members):
                previous = allowed.get(row["name"])
                if previous is None or any(previous[key] != row[key] for key in ("ownership", "shape", "dtype")):
                    raise BatchStateUnsupported(f"live allocation {name}/{row['name']} differs from its pre-thread reservation")
                if previous["allocated_bytes"] < row["allocated_bytes"]:
                    raise BatchStateUnsupported(f"live allocation {name}/{row['name']} exceeds its reserved backing")
            if reserved.reserved_bytes < plan.reserved_bytes:
                raise BatchStateUnsupported(f"live transient reserve {name} exceeds its pre-thread reservation")

    def receipt(self):
        return {"ordinary_forecast_bytes": dict(self.ordinary_forecast_bytes),
            "native_plans": {name: {"arrays": plan.inventory(len(self.ordinary_forecast_bytes)),
                "reserved_bytes": plan.reserved_bytes} for name, plan in self.native_plans.items()},
            "fixed_bytes": dict(self.fixed_bytes), "evidence": dict(self.evidence),
            "required_bytes": self.required_bytes, "available_bytes": self.available_bytes,
            "admitted_before_runner_threads": True}


class _Endpoint:
    def __init__(self, owner, member):
        self.owner, self.member = owner, member
        self.queue, self.binding = Queue(), None
        self.callbacks, self.adaptive_driver = {}, None
        self.stream, self.ready, self.thread_id = None, None, None
        self._terminated = False
        self.retired_owner_identity = None

    def submit(self, kind, args):
        future = Future()
        ready = self.owner.record_event()
        with self.owner._lock:
            self.owner.check()
            if self._terminated:
                raise RuntimeError("original route worker has already retired")
            self.queue.put((kind, args, ready, future))
        return future

    def rpc(self, kind, *args):
        return self.submit(kind, args).result()

    def abort(self, error):
        self.queue.put(("abort", (error,), None, None))

    def _original(self, kind, args):
        self.owner.check()
        if kind == "period_steps":
            return self.adaptive_driver(*args)
        if kind == "before_step" and self.adaptive_driver is not None:
            self.adaptive_driver.before_step(args[0])
        callback = self.callbacks.get(kind)
        return None if callback is None else callback(*args)

    def ordinary_schedule(self):
        binding = self.binding
        names = ("step", "force", "feedback_prepare", "feedback_commit", "feedback_finalize",
                 "history", "restart", "lbc_reset", "period_begin", "period_end", "period_commit", "domain_start")
        callbacks = {"on_" + name: lambda *args, name=name: self._original(name, args) for name in names}
        return execute_schedule(binding.model.schedule, clocks=binding.clocks,
            start_period=binding.start_period, started_grid_ids=binding.started_grid_ids,
            committed_initial_history_grid_ids=binding.committed_initial_history_grid_ids,
            skip_feedback_path=binding.skip_feedback_path,
            on_period_steps=(None if self.adaptive_driver is None else
                             lambda *args: self._original("period_steps", args)), **callbacks)

    def hook(self, **handoff):
        self.owner.check()
        if self.binding is not None:
            raise RuntimeError("ordinary route dispatched its schedule more than once")
        self.thread_id = threading.get_ident()
        self.callbacks, self.adaptive_driver = dict(handoff["callbacks"]), handoff["adaptive_driver"]
        cfl = current_cfl_member()
        if cfl is None:
            raise RuntimeError("route dispatch needs the member's active original CFL owner")
        self.binding = MemberScheduleBinding(self.member.member_id, handoff["model"],
            {name: lambda *args, name=name: self.rpc(name, *args) for name in self.callbacks},
            operation_authority=self.member.operation_authority, adaptive_driver=self.adaptive_driver,
            start_period=handoff["start_period"], started_grid_ids=handoff["started_grid_ids"],
            committed_initial_history_grid_ids=handoff["committed_initial_history_grid_ids"],
            skip_feedback_path=handoff["skip_feedback_path"], step_owns_before_step=True,
            step_bindings=handoff["step_bindings"])
        self.binding.context, self.binding.cfl_owner = copy_context(), cfl
        self.retired_owner_identity = {"cfl": id(cfl), "adaptive": id(self.adaptive_driver),
            "step_bindings": id(handoff["step_bindings"]), "route_thread": self.thread_id}
        if self.owner.cuda is not None:
            self.stream = self.owner.cuda.get_current_stream()
        self.ready = self.owner.record_event()
        self.owner.notifications.put(("captured", self.member.member_id))
        try:
            return self._park()
        finally:
            pending = []
            with self.owner._lock:
                self._terminated = True
                while True:
                    try:
                        pending.append(self.queue.get_nowait())
                    except Empty:
                        break
            for _, _, _, future in pending:
                if future is not None and not future.done():
                    future.set_exception(self.owner._error or RuntimeError("original route retired before its callback"))

    def _park(self):
        while True:
            kind, args, ready, future = self.queue.get()
            if kind == "abort":
                raise args[0]
            if kind == "finish":
                return args[0]
            error, result = None, None
            try:
                self.owner.check()
                if ready is not None:
                    self.stream.wait_event(ready)
                result = self.ordinary_schedule() if kind == "ordinary_schedule" else self._original(kind, args)
            except BaseException as caught:
                error = caught
                name_failing_members(error, (self.member.member_id,), execution_mode="parked_original_route")
                self.owner.fail(error)
            finally:
                try:
                    if self.stream is not None:
                        self.stream.synchronize()
                except BaseException as caught:
                    if error is None:
                        error = caught
                        self.owner.fail(error)
                    elif caught is not error:
                        error.add_note(f"route stream completion also failed: {type(caught).__name__}: {caught}")
                if error is None:
                    future.set_result(result)
                else:
                    future.set_exception(error)
            if error is not None:
                raise error

    def retire(self):
        """Drop captured model/context cycles only after all route work joins."""
        self.binding = None
        self.callbacks.clear()
        self.adaptive_driver = None
        self.stream = self.ready = None


class PreparedRouteWave:
    """Original prepared lifetimes with caller-priced component admission."""
    def __init__(self, members, *, reservation, runner, member_scope, array_module=None,
                 pack_factory=None, batch_factory=None, control=None,
                 shared_fields=tuple(sorted(SHARED_STATE_CANDIDATES)), device_id=0):
        self.members, self.reservation = tuple(members), reservation
        ids = [member.member_id for member in self.members]
        if not ids or len(set(ids)) != len(ids) or any(not isinstance(member, RouteMember) for member in self.members):
            raise ValueError("route wave needs unique bound members")
        if not callable(member_scope) or not callable(runner):
            raise TypeError("route wave needs original runner and CUDA owner scopes")
        self.runner, self.member_scope, self.xp = runner, member_scope, array_module
        self.cuda = None if array_module is None else getattr(array_module, "cuda", None)
        self.device_id = _bytes(device_id, "device_id")
        self.pack_factory, self.batch_factory, self.control = pack_factory, batch_factory, control
        self.shared_fields = tuple(shared_fields)
        self.notifications, self._lock, self._error = Queue(), threading.Lock(), None
        self.endpoints = {member.member_id: _Endpoint(self, member) for member in self.members}
        self._used = False

    def check(self):
        if self._error is not None:
            raise self._error
        if self.control is not None:
            self.control.check()

    def fail(self, error):
        with self._lock:
            if self._error is None:
                self._error = error
                for endpoint in self.endpoints.values():
                    endpoint.abort(error)

    def record_event(self):
        if self.cuda is None:
            return None
        event = self.cuda.Event()
        event.record(self.cuda.get_current_stream())
        return event

    def completion_wait(self):
        error = None
        for endpoint in self.endpoints.values():
            try:
                if endpoint.stream is not None:
                    endpoint.stream.synchronize()
            except BaseException as caught:
                if error is None:
                    error = caught
                elif caught is not error:
                    error.add_note(f"another route stream completion also failed: {type(caught).__name__}: {caught}")
        if error is not None:
            raise error

    def original_group_dispatch(self, group):
        calls, results, error = [], [], None
        for request in group.requests:
            try:
                calls.append(self.endpoints[request.member_id].submit(request.kind, request.args))
            except BaseException as caught:
                error = caught
                self.fail(error)
                break
        for call in calls:
            try:
                results.append(call.result())
            except BaseException as caught:
                if error is None:
                    error = caught
                    self.fail(error)
        if error is not None:
            raise error
        return tuple(results)

    def _run_member(self, member):
        try:
            with self.member_scope(member_id=member.member_id, device_id=self.device_id), member_output_scope(member.capture):
                return self.runner(member.inputs, output_directory=member.output_directory,
                    schedule_dispatch=self.endpoints[member.member_id].hook, **member.runner_options)
        except BaseException as error:
            name_failing_members(error, (member.member_id,), execution_mode="original_prepared_route")
            self.fail(error)
            raise
        finally:
            self.notifications.put(("completed", member.member_id))

    def _native_batch(self, bindings):
        from gpuwm.ensemble.prepared_nested_batch import PreparedHybridNestedBatch, _domain_bank_plan
        factory = self.batch_factory or PreparedHybridNestedBatch
        if "original_group_dispatch" not in inspect.signature(factory.execute).parameters:
            raise BatchStateUnsupported("component executor lacks concurrent original-group dispatch; parked ordinary members retain their complete schedules")
        topology = tuple(bindings[0].clocks)
        if any(tuple(binding.clocks) != topology for binding in bindings):
            raise BatchStateUnsupported("member domain trees differ; original schedules retain each tree")
        plans = {}
        for gid in topology:
            selected = (self.pack_factory.state_fields_for_grid(bindings, gid)
                if hasattr(self.pack_factory, "state_fields_for_grid") else None)
            plans[f"bank:{gid}"] = _domain_bank_plan(tuple(binding.model.node(gid) for binding in bindings),
                shared_fields=self.shared_fields, array_module=self.xp, field_names=selected)
        descriptors = {gid: SimpleNamespace(cfg=bindings[0].model.node(gid).cfg.run,
            storage=SimpleNamespace(specs={spec.name: spec for spec in plans[f"bank:{gid}"].arrays})) for gid in topology}
        plans.update(self.pack_factory.memory_plans(bindings, descriptors))
        self.reservation.validate_live_plans(plans)
        # Fixed reservations remain unavailable to all bank/component owners.
        return factory(bindings, ordinary_forecast_bytes=self.reservation.ordinary_forecast_bytes,
            available_bytes=self.reservation.available_bytes - sum(self.reservation.fixed_bytes.values()),
            ordinary_memory_evidence=self.reservation.evidence["ordinary"], pack_factory=self.pack_factory,
            array_module=self.xp, shared_fields=self.shared_fields, completion_wait=self.completion_wait)

    def run(self):
        if self._used:
            raise RuntimeError("route wave is single-use")
        self._used = True
        self._cold_fallback = self.reservation.admit(tuple(self.endpoints), require_native=self.pack_factory is not None)
        with nullcontext() if self.cuda is None else self.cuda.Device(self.device_id):
            return self._run()

    def _run(self):
        pool, jobs, batch = ThreadPoolExecutor(max_workers=len(self.members), thread_name_prefix="ensemble-prepared-route"), {}, None
        result, fallback = None, self._cold_fallback
        try:
            for member in self.members:
                context = copy_context()
                jobs[member.member_id] = pool.submit(context.run, self._run_member, member)
            captured = set()
            while len(captured) != len(self.members):
                self.check()
                try:
                    kind, member = self.notifications.get(timeout=0.1)
                except Empty:
                    continue
                if kind == "captured":
                    captured.add(member)
                else:
                    self.check()
                    raise RuntimeError(f"member {member} returned without its route schedule handoff")
            for endpoint in self.endpoints.values():
                if endpoint.ready is not None:
                    self.cuda.get_current_stream().wait_event(endpoint.ready)
            bindings = tuple(endpoint.binding for endpoint in self.endpoints.values())
            if self.pack_factory is not None and fallback is None:
                try:
                    batch = self._native_batch(bindings)
                except BatchStateUnsupported as error:
                    fallback = str(error)
                    close = getattr(self.pack_factory, "close", None)
                    if close is not None:
                        close()
            elif self.pack_factory is None:
                fallback = "no qualified component factory was supplied; original members retain their complete schedules"
            if batch is None:
                print(f"route members {tuple(self.endpoints)}: concurrent ordinary schedules: {fallback}", flush=True)
                calls = {member: endpoint.submit("ordinary_schedule", ()) for member, endpoint in self.endpoints.items()}
                reports = {member: call.result() for member, call in calls.items()}
                result = {"reports": reports, "packed_components": False, "fallback_reason": fallback}
            else:
                result = batch.execute(original_group_dispatch=self.original_group_dispatch)
                batch.close()
                batch = None
            self.completion_wait()
            for member, endpoint in self.endpoints.items():
                endpoint.queue.put(("finish", (result["reports"][member],), None, None))
            records = {}
            for member, future in jobs.items():
                record = future.result()
                if not isinstance(record, dict) or record.get("status") != "PASS":
                    raise RuntimeError(f"original member {member} returned a failing route receipt")
                records[member] = record
            return {"contract": CONTRACT, "status": "PASS", "member_order": list(self.endpoints),
                "member_results": records, "execution": result, "memory": self.reservation.receipt(),
                "native_admitted_before_threads": self.pack_factory is not None and self._cold_fallback is None,
                "default_door_enabled": False, "route_finalization": "original_runner",
                "callback_threads": {member: endpoint.thread_id for member, endpoint in self.endpoints.items()}}
        except BaseException as error:
            self.fail(error)
            if self._error is error:
                raise
            raise self._error from error
        finally:
            try:
                if batch is not None:
                    try:
                        batch.close()
                    except BaseException as error:
                        if self._error is None:
                            self.fail(error)
                        elif error is not self._error:
                            self._error.add_note(f"route component cleanup also failed: {type(error).__name__}: {error}")
            finally:
                try:
                    pool.shutdown(wait=True, cancel_futures=True)
                    try:
                        self.completion_wait()
                    except BaseException as error:
                        if self._error is None:
                            raise
                        self._error.add_note(f"route completion also failed: {type(error).__name__}: {error}")
                finally:
                    for endpoint in self.endpoints.values():
                        endpoint.retire()


__all__ = ["CONTRACT", "RouteMember", "RouteWaveReservation", "PreparedRouteWave"]
