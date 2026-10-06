"""An adaptive member cannot read or reset another member's CFL history."""
import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, local
from types import SimpleNamespace

from gpuwm.core.cfl_member import CFL_BANK_NAMES, current_cfl_member, member_cfl_scope


def _recording_functions():
    path = Path(__file__).resolve().parents[1] / "gpuwm/core/dycore.py"
    names = {"_wrf_cfl_bank", "_wrf_cfl_recording_enabled",
             "_wrf_cfl_event", "enable_wrf_cfl_recording", "reset_wrf_cfl_recording"}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(functions) == len(names)
    namespace = dict(_WRF_CFL_PROBE=False, _WRF_CFL_THREAD=local(), _env_flag=lambda name: False)
    namespace["cp"] = SimpleNamespace(cuda=SimpleNamespace(Event=lambda **kw: object()))
    namespace.update({name: {} for name in CFL_BANK_NAMES})
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_concurrent_same_grid_members_cannot_reset_each_others_recording():
    namespace = _recording_functions()
    ready, first_reset = Barrier(2), Barrier(2)

    def worker(member):
        with member_cfl_scope() as owner:
            namespace["enable_wrf_cfl_recording"]()
            bank = namespace["_wrf_cfl_bank"]("_WRF_CFL_CALLS")
            bank[1] = (member + 1) * 3
            ready.wait(timeout=5)
            assert namespace["_wrf_cfl_recording_enabled"]()
            assert bank[1] == (member + 1) * 3
            if member == 0:
                namespace["reset_wrf_cfl_recording"]()
            first_reset.wait(timeout=5)
            if member == 1:
                assert namespace["_wrf_cfl_recording_enabled"]()
                assert bank[1] == 6
            else:
                assert not namespace["_wrf_cfl_recording_enabled"]()
                assert bank == {}
        assert current_cfl_member() is None
        assert all(bank == {} for bank in owner.banks.values())

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(worker, (0, 1)))
    assert namespace["_WRF_CFL_PROBE"] is False
    assert namespace["_WRF_CFL_CALLS"] == {}


def test_reused_cfl_events_are_private_to_each_concurrent_member():
    namespace = _recording_functions()
    event_for = namespace["_wrf_cfl_event"]
    global_event = event_for("ready", 1, 0)
    ready, reset = Barrier(2), Barrier(2)

    def worker(member):
        with member_cfl_scope():
            event = event_for("ready", 1, 0)
            assert event_for("ready", 1, 0) is event
            assert event is not global_event
            ready.wait(timeout=5)
            if member == 0:
                namespace["reset_wrf_cfl_recording"]()
            reset.wait(timeout=5)
            if member == 1:
                assert event_for("ready", 1, 0) is event
            else:
                assert event_for("ready", 1, 0) is not event
            return event

    with ThreadPoolExecutor(max_workers=2) as executor:
        events = list(executor.map(worker, (0, 1)))
    assert events[0] is not events[1]
    assert event_for("ready", 1, 0) is global_event


def test_unscoped_ordinary_recording_keeps_original_global_banks():
    namespace = _recording_functions()
    namespace["enable_wrf_cfl_recording"]()
    assert namespace["_WRF_CFL_PROBE"] is True
    assert namespace["_wrf_cfl_bank"]("_WRF_CFL_CALLS") is namespace["_WRF_CFL_CALLS"]
    namespace["_WRF_CFL_CALLS"][1] = 3
    with member_cfl_scope():
        assert namespace["_wrf_cfl_bank"]("_WRF_CFL_CALLS") == {}
        namespace["enable_wrf_cfl_recording"]()
        namespace["reset_wrf_cfl_recording"]()
    assert namespace["_WRF_CFL_PROBE"] is True
    assert namespace["_WRF_CFL_CALLS"] == {1: 3}
    namespace["reset_wrf_cfl_recording"]()
    assert namespace["_WRF_CFL_PROBE"] is False
    assert namespace["_WRF_CFL_CALLS"] == {}
