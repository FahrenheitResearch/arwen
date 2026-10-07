"""Actual sealed RAM and scheduler continuation contracts, CPU only."""
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.da import spread_vts as vts, spread_vts_ram as ram


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.uint8])
def test_sealed_ram_preserves_native_bytes_and_lifetime(tmp_path, dtype):
    original = np.arange(24, dtype=dtype).reshape(2, 3, 4)
    expected = original.tobytes()
    server = ram.RetainedRam(tmp_path/"s", max_bytes=4096).start()
    try:
        receipt = ram.put(server.path, "actual", {"state/u": original}, {"ticks": 900})
        original[...] = 0
        with ram.get(server.path, "actual") as (values, record):
            assert record == receipt
            assert values["state/u"].dtype == np.dtype(dtype)
            assert values["state/u"].tobytes() == expected
            assert not values["state/u"].flags.writeable
            with pytest.raises(ValueError):
                values["state/u"][...] = 2
        assert ram.drop(server.path, "actual")
        assert not ram.drop(server.path, "actual")
        with pytest.raises(ram.RetentionError, match="missing"):
            with ram.get(server.path, "actual"):
                pass
    finally:
        server.clear()
    assert not server.path.exists()


def test_ram_refuses_duplicate_and_budget_without_replacing_prior(tmp_path):
    server = ram.RetainedRam(tmp_path/"s", max_bytes=32).start()
    try:
        ram.put(server.path, "one", {"p": np.arange(4, dtype=np.float64)}, {})
        with pytest.raises(ram.RetentionError, match="already exists"):
            ram.put(server.path, "one", {"p": np.zeros(1)}, {})
        with pytest.raises(ram.RetentionError, match="admission"):
            ram.put(server.path, "two", {"p": np.zeros(1)}, {})
        with ram.get(server.path, "one") as (values, _):
            np.testing.assert_array_equal(values["p"], np.arange(4))
    finally:
        server.clear()


class _FcntlView:
    """The real ``fcntl``, with the sealing names shown or hidden.

    The uv-managed CPython 3.13 builds a release venv runs on publish no
    ``fcntl.F_SEAL_*``, ``F_ADD_SEALS`` or ``F_GET_SEALS``; a system Python
    built against newer headers does.  Hiding or showing them here runs
    both branches of gpuwm.da.spread_vts_ram._sealing on whichever
    interpreter the suite is on.  Calls go to the real module, so the
    kernel still seals and checks the descriptor.
    """

    def __init__(self, names_shown, refuse_add=False):
        import fcntl
        self._fcntl = fcntl
        self._shown = names_shown
        self._refuse_add = refuse_add

    def __getattr__(self, name):
        if name in ram._LINUX_SEALING_ABI:
            if not self._shown:
                raise AttributeError(name)
            return getattr(self._fcntl, name, ram._LINUX_SEALING_ABI[name])
        return getattr(self._fcntl, name)

    def fcntl(self, fd, command, argument=0):
        if self._refuse_add and command == ram._LINUX_SEALING_ABI["F_ADD_SEALS"]:
            raise OSError(22, "Invalid argument")
        return self._fcntl.fcntl(fd, command, argument)


def test_the_kernel_abi_table_matches_every_name_the_interpreter_publishes():
    import fcntl
    import os
    for name, number in ram._LINUX_SEALING_ABI.items():
        if hasattr(fcntl, name):
            assert getattr(fcntl, name) == number, name
    for name, number in ram._LINUX_MEMFD_ABI.items():
        if hasattr(os, name):
            assert getattr(os, name) == number, name


@pytest.mark.parametrize("names_shown", [True, False],
                         ids=["module-names", "kernel-abi"])
def test_sealed_ram_runs_with_and_without_the_module_sealing_names(
        tmp_path, monkeypatch, names_shown):
    view = _FcntlView(names_shown)
    monkeypatch.setattr(ram, "_fcntl", lambda: view)
    _fcntl, add, get, seals, _flags = ram._sealing()
    assert (add, get, seals) == (1033, 1034, 0xF)
    original = np.arange(12, dtype=np.float32).reshape(3, 4)
    server = ram.RetainedRam(tmp_path/"s", max_bytes=4096).start()
    try:
        receipt = ram.put(server.path, "slot", {"state/t": original}, {})
        with ram.get(server.path, "slot") as (values, record):
            assert record == receipt
            assert values["state/t"].tobytes() == original.tobytes()
            assert not values["state/t"].flags.writeable
    finally:
        server.clear()


def test_a_kernel_that_refuses_the_seals_is_refused_by_name(tmp_path,
                                                            monkeypatch):
    view = _FcntlView(False, refuse_add=True)
    monkeypatch.setattr(ram, "_fcntl", lambda: view)
    server = ram.RetainedRam(tmp_path/"s", max_bytes=4096).start()
    try:
        with pytest.raises(ram.RetentionError, match="refused to seal"):
            ram.put(server.path, "slot", {"p": np.zeros(2)}, {})
        with pytest.raises(ram.RetentionError, match="missing"):
            with ram.get(server.path, "slot"):
                pass
    finally:
        server.clear()


@pytest.mark.parametrize("kwargs,reason", [
    ({"analysis_seconds": 600}, "t-900"),
    ({"analysis_seconds": 3600.5}, "whole-second"),
    ({"origin_seconds": 3000}, "t-900"),
    ({"history_seconds": 3600}, "alarms"),
    ({"analysis_seconds": 3700}, "alarms"),
    ({"observation_heating": True}, "TTEN"),
])
def test_preflight_refuses_unreal_or_leaking_slots(kwargs, reason):
    arguments = dict(analysis_seconds=3600, origin_seconds=0, history_seconds=300)
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=reason):
        vts.validate_request(**arguments)


def test_history_drains_native_handoff_before_capturing_actual_clock():
    capture = vts.ForecastRetention(socket_path="unused", member=0,
        analysis_seconds=3600, origin_seconds=0, grid_identity_sha256="a"*64,
        fields=("u",))
    calls = []
    capture.capture = lambda *a, **k: calls.append(("capture", k["offset_seconds"]))
    hook = capture.history(lambda *a: calls.append(("drain", a[2])), driver=None, cfg=None)
    node = SimpleNamespace(cfg=SimpleNamespace(grid_id=1), clock=SimpleNamespace(tick_den=3))
    hook(None, node, 8100)
    hook(None, node, 9000)
    assert calls == [("drain", 8100), ("capture", -900), ("drain", 9000)]


def test_retention_without_output_writer_still_drains_native_handoff(monkeypatch):
    capture = vts.ForecastRetention(socket_path="unused", member=0,
        analysis_seconds=3600, origin_seconds=0, grid_identity_sha256="a"*64, fields=("u",))
    calls = []
    monkeypatch.setattr(vts, "drain_native_history", lambda *a: calls.append(("drain", a[2])))
    capture.capture = lambda *a, **k: calls.append(("capture", k["offset_seconds"]))
    hook = capture.history(None, driver=None, cfg=None)
    node = SimpleNamespace(cfg=SimpleNamespace(grid_id=1), clock=SimpleNamespace(tick_den=1))
    hook(None, node, 2700)
    hook(None, node, 3000)
    assert calls == [("drain", 2700), ("capture", -900), ("drain", 3000)]


@pytest.mark.parametrize("failure", [None, "execute", "capture"])
def test_future_uses_native_owner_and_restores_complete_central_tree(monkeypatch, failure):
    from gpuwm.core import clock, model as native
    from gpuwm.ingest import lateral_bc
    from tools import da_cycle_prepared as cycle
    @dataclass
    class Experiment:
        run_seconds: float
    original_clock = SimpleNamespace(ticks=3600, tick_den=1, elapsed_seconds=3600)
    node = SimpleNamespace(clock=original_clock, cfg=SimpleNamespace(run=SimpleNamespace(specified=False)))
    model = SimpleNamespace(root=node, nodes_by_grid_id={1: node}, schedule="central",
                            _resume_committed_history_grid_ids=frozenset({1}))
    future_clock = SimpleNamespace(ticks=0, tick_den=1, elapsed_seconds=0)
    resolution = SimpleNamespace(tick_den=1, clocks=lambda: {1: future_clock})
    calls = []
    def resolve(exp, **kwargs):
        assert exp.run_seconds == 4500
        calls.append("resolve")
        return resolution
    monkeypatch.setattr(clock, "resolve_clock", resolve)
    monkeypatch.setattr(clock, "build_schedule", lambda *a: SimpleNamespace(period_ticks=60))
    def restore(model, path, *, expected_seconds):
        assert path == "complete-central" and expected_seconds == 3600
        calls.append(("restore", model.schedule == "central"))
        node.clock.ticks = expected_seconds
        node.clock.elapsed_seconds = expected_seconds
    monkeypatch.setattr(cycle, "restore_leg_restart", restore)
    def execute(model, **kwargs):
        calls.append("execute")
        assert kwargs["history_handler"].__name__ == "drain_native_history"
        assert node.clock.ticks == 3600
        node.clock.ticks = node.clock.elapsed_seconds = 4500
        if failure == "execute":
            raise RuntimeError("execution failed")
    monkeypatch.setattr(native, "execute_experiment", execute)
    def capture():
        calls.append("capture")
        assert node.clock.elapsed_seconds == 4500
        if failure == "capture":
            raise RuntimeError("capture failed")
    arguments = dict(model=model, restart="complete-central", experiment=Experiment(3600),
                     boundary_interval_seconds=10800, analysis_seconds=3600, capture=capture)
    if failure:
        with pytest.raises(RuntimeError, match="failed"):
            vts.continue_and_restore(**arguments)
    else:
        receipt = vts.continue_and_restore(**arguments)
        assert receipt["central_restored_seconds"] == 3600
    assert node.clock is original_clock and node.clock.elapsed_seconds == 3600
    assert model.schedule == "central"
    assert calls[0:3] == ["resolve", ("restore", False), "execute"]
    assert calls[-1] == ("restore", True)


def test_named_flags_leave_default_namespace_identical():
    from tools.da_cycle_prepared import build_parser
    parser = build_parser()
    fields = {action.dest: action.default for action in parser._actions}
    import argparse
    assert fields["spread_repair_vtsm"] == argparse.SUPPRESS
    assert fields["spread_vts_ram_gib"] == argparse.SUPPRESS


def test_setup_identity_binds_values_and_order_but_not_storage_dtype():
    a = np.arange(6, dtype=np.float32)
    assert vts.setup_identity({"a": a, "b": a+1}) == vts.setup_identity({"b": (a+1).astype(np.float64), "a": a})
    assert vts.setup_identity({"a": a}) != vts.setup_identity({"a": a+1})
