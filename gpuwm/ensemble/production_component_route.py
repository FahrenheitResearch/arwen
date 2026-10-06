"""Original production recipe owners inside explicitly priced component waves.

Selection, allocation metadata and durable ensemble finalization belong to
the caller. This adapter adds no default routing policy or numerical work.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
import threading

from gpuwm.ensemble.batch_state import BatchStateUnsupported
from gpuwm.ensemble.execution import member_run_scope
from gpuwm.ensemble.prepared_route_wave import PreparedRouteWave, RouteMember, RouteWaveReservation
from gpuwm.ensemble.runtime_context import MemberOutputCapture, current_capture


@dataclass(frozen=True)
class ProductionComponentRouteResult:
    """Joined original reports and captures for the existing finalizer."""
    member_results: dict
    captures: dict
    output_directories: dict
    execution: dict


def source_component_route_reason(source_execution):
    """Name source owners whose physical callback contract is not bound."""
    if source_execution is None:
        return None
    from gpuwm.ensemble.ordinary_execution import OrdinaryRecipeExecution
    if type(source_execution) is not OrdinaryRecipeExecution:
        return ("component routes bind unchanged OrdinaryRecipeExecution sources; "
                "this source owner may change prepared fields before forecast and has no priced route handoff")
    return None


def run_production_component_wave(session, runner, *, member_inputs, output_directory,
        reservation, progress_adapter, control, runner_options=None, progress_key="progress_callback",
        device_id=0, operation_authority, source_context_callback=None, member_scope=None,
        array_module=None, qualification_receipt=None, pack_factory=None, shared_fields=(),
        wave_factory=PreparedRouteWave):
    """Run one selected cohort, preserving its original source/runner scopes.

    ``member_inputs`` are the caller's already bound planning inputs, keyed
    by global member id. A source owner's forecast callback applies the same
    session configuration and publishes source/stochastic metadata through
    ``source_context_callback`` before invoking the genuine prepared runner.
    The callback accepts member_id, actual_inputs, authority and source_receipt.
    Original report objects are returned unchanged after every route joins.
    """
    if not isinstance(member_inputs, Mapping) or len(member_inputs) < 2:
        raise BatchStateUnsupported("a component route needs a priced cohort of at least two members; a singleton keeps its ordinary runner")
    if not isinstance(reservation, RouteWaveReservation):
        raise TypeError("component routes require a complete pre-thread RouteWaveReservation")
    ids = tuple(member_inputs)
    if any(type(member) is not int or member < 0 for member in ids):
        raise ValueError("component routes require original nonnegative integer member ids")
    if not callable(runner) or not callable(operation_authority):
        raise TypeError("component routes need the original runner and member operation authority")
    if progress_key not in ("observer", "progress_callback"):
        raise ValueError("component route progress must use the original observer or progress_callback binding")
    options = dict(runner_options or {})
    if "schedule_dispatch" in options:
        raise ValueError("component wave owns the prepared runner schedule dispatch")
    source = session.source_execution
    reason = source_component_route_reason(source)
    if reason is not None:
        raise BatchStateUnsupported(reason)
    if source is not None and not callable(source_context_callback):
        raise BatchStateUnsupported("ordinary source component routes need their durable source/stochastic manifest callback before forecast")
    # Ordinary refusal precedes directories, initialization callbacks and
    # worker scopes. Native refusal remains the wave's concurrent fallback.
    reservation.admit(ids, require_native=False)
    control.check()
    root = Path(output_directory).resolve()
    directories = {member: root / "members" / f"member-{member:04d}" for member in ids}
    if any(path.exists() or path.is_symlink() for path in directories.values()):
        raise FileExistsError("component routes require fresh original member output directories")
    if array_module is None:
        array_module = session.array_module
        if array_module is None:
            import cupy as array_module
    if member_scope is None:
        from gpuwm.ensemble.member_stream import member_cuda_scope
        member_scope = lambda **kwargs: member_cuda_scope(array_module=array_module, **kwargs)
    if pack_factory is None and qualification_receipt:
        from gpuwm.ensemble.prepared_nested_batch import ProductionNestedPackFactory
        pack_factory = ProductionNestedPackFactory(qualification_receipt=qualification_receipt)
    from gpuwm.ensemble.radar_output import member_history_required
    captures, retained, members = {}, {}, []
    for member, inputs in member_inputs.items():
        retained[member] = []
        previous = session._initialization_callback(member)
        def initialize(*, _member=member, _previous=previous, **kwargs):
            owner = kwargs.get("model")
            if owner is None:
                owner = kwargs.get("state")
            if owner is not None:
                retained[_member].append(owner)
            if _previous is not None:
                _previous(**kwargs)
        capture = captures[member] = MemberOutputCapture(session.collector.submit, member,
            member_history_required(inputs, session.request.keep_member_files), initialize_callback=initialize)
        selected = dict(options, first_products=None)
        selected[progress_key] = progress_adapter.callback_for_member(member)
        members.append(RouteMember(member, inputs, directories[member], capture,
            lambda *, _member=member, **kwargs: operation_authority(member_id=_member, **kwargs), selected))
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=False)
    reports, lock = {}, threading.Lock()
    def original_route(inputs, *, output_directory, schedule_dispatch, **member_options):
        member = current_capture().member_id
        control.check()
        def forecast(actual_inputs):
            control.check()
            if source is not None:
                actual_inputs = session._configured_member_inputs(actual_inputs)
                authority = session._stochastic_authorities.get(id(actual_inputs))
                source_context_callback(member_id=member, actual_inputs=actual_inputs,
                    authority=authority, source_receipt=source.receipt())
            return runner(actual_inputs, output_directory=output_directory,
                schedule_dispatch=schedule_dispatch, **member_options)
        report = (forecast(inputs) if source is None else source.run_member(member, forecast=forecast,
                  observer=member_options.get(progress_key)))
        with lock:
            reports[member] = report
        coordinated = asdict(report) if is_dataclass(report) else dict(report)
        if coordinated.get("status") is None:
            coordinated["status"] = "PASS"
        return coordinated
    try:
        with member_run_scope(control):
            result = wave_factory(tuple(members), reservation=reservation, runner=original_route,
                member_scope=member_scope, array_module=array_module, pack_factory=pack_factory,
                control=control, device_id=device_id, shared_fields=shared_fields).run()
        if set(reports) != set(ids):
            raise RuntimeError("component route did not return every original member report")
        return ProductionComponentRouteResult(reports, captures, directories, result)
    finally:
        # RouteWave returns or raises only after all original owners join.
        # Captures retain calendars for finalization without retaining models.
        for owners in retained.values():
            owners.clear()


__all__ = ["ProductionComponentRouteResult", "source_component_route_reason", "run_production_component_wave"]
