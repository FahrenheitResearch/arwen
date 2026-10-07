"""Selected library ownership; native numerical preparation has separate gates."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
import ctypes
import inspect
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from gpuwm.core import portable_math as pm
from gpuwm.ingest import cpu_backend, preparation_workers, preprocess_backend
ORIGINAL_RESOLVE_CPU_BRIDGE = cpu_backend.resolve_cpu_bridge


class _Entry:
    def __init__(self, function):
        self.function = function
    def __call__(self, *args):
        return self.function(*args)


def _library(marker):
    def unary(code, source, target, count, workers, *, single):
        ctype = ctypes.c_float if single else ctypes.c_double
        incoming = np.ctypeslib.as_array((ctype * count).from_address(source))
        output = np.ctypeslib.as_array((ctype * count).from_address(target))
        # A route marker, not a replacement or oracle for native math.
        output[...] = incoming + marker
        return 0
    return SimpleNamespace(gpuwm_portable_math_version=_Entry(lambda: 1),
        gpuwm_portable_unary_f64=_Entry(lambda *args: unary(*args, single=False)),
        gpuwm_portable_unary_f32=_Entry(lambda *args: unary(*args, single=True)),
        gpuwm_portable_binary_f64=_Entry(lambda *args: 0),
        gpuwm_portable_binary_f32=_Entry(lambda *args: 0))


@pytest.fixture
def libraries(tmp_path, monkeypatch):
    paths = (tmp_path / "selected-a.dll", tmp_path / "selected-b.dll")
    for path in paths:
        path.write_bytes(b"binding-only native-library fixture")
    records = {str(paths[0]): _library(73.), str(paths[1]): _library(101.)}
    resolved = []
    def resolve(path=None):
        assert path is not None, "an explicit binding must not use the implicit resolver"
        resolved.append(Path(path))
        return Path(path)
    monkeypatch.setattr(cpu_backend, "resolve_cpu_bridge", resolve)
    monkeypatch.setattr(pm.ctypes, "CDLL", lambda path: records[str(path)])
    monkeypatch.setattr(pm, "_resolved", True)
    monkeypatch.setattr(pm, "_library", None)
    monkeypatch.setattr(pm, "_absent_reason", "cached implicit fallback")
    monkeypatch.setattr(pm, "_warned", False)
    monkeypatch.setattr(pm, "_scoped_warned", set())
    monkeypatch.delenv(cpu_backend.CPU_BRIDGE_ENV, raising=False)
    # The preprocessing receipt's parallelism block carries
    # memory_worker_limit, a live reading of host headroom in 32 MiB steps.
    # Unpinned, a busy host moved it between two receipt() calls and
    # test_explicit_host_math_receipt_is_owned_and_implicit_receipt_words_unchanged
    # failed in full Stage 1 runs while passing alone (AUDIT-287 B6).
    monkeypatch.setattr(preparation_workers, "host_available_bytes",
                        lambda *args, **kwargs: 64 * preparation_workers.WORKER_SCRATCH_BYTES)
    return paths, resolved, records


def _word():
    return pm.exp(np.array([2.], dtype=np.float64)).item()


def test_explicit_selected_library_overrides_cached_implicit_fallback_and_restores_it(libraries):
    paths, resolved, _ = libraries
    assert pm.current_cpu_bridge_binding() is None
    assert pm.implementation() == pm.FALLBACK_IMPLEMENTATION
    with pm.cpu_bridge_scope(paths[0]):
        assert pm.implementation() == pm.IMPLEMENTATION
        assert _word() == 75.
        assert pm._library is None and pm._resolved
        with pm.cpu_bridge_scope(paths[1]):
            assert _word() == 103.
        assert _word() == 75.
    assert pm.current_cpu_bridge_binding() is None
    assert pm.implementation() == pm.FALLBACK_IMPLEMENTATION
    assert pm._absent_reason == "cached implicit fallback"
    assert set(resolved) == set(paths)


def test_scope_exception_and_none_do_not_change_ordinary_resolution(libraries):
    paths, resolved, _ = libraries
    with pm.cpu_bridge_scope(None):
        assert pm.current_cpu_bridge_binding() is None
    assert resolved == []
    with pytest.raises(RuntimeError, match="original preparation failed"):
        with pm.cpu_bridge_scope(paths[0]):
            raise RuntimeError("original preparation failed")
    assert pm.current_cpu_bridge_binding() is None and pm._library is None


def test_concurrent_preparations_keep_their_own_selected_libraries(libraries):
    paths, _, _ = libraries
    fence = threading.Barrier(2)
    def task(path):
        with pm.cpu_bridge_scope(path):
            fence.wait(timeout=10)
            return _word(), pm.current_cpu_bridge_binding().path
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(task, path) for path in paths]
        assert [future.result() for future in futures] == [(75., paths[0]), (103., paths[1])]
    assert pm.current_cpu_bridge_binding() is None


def test_explicit_old_library_warns_about_that_library_and_keeps_implicit_cache(libraries, capsys):
    paths, _, records = libraries
    records[str(paths[0])] = SimpleNamespace()
    with pm.cpu_bridge_scope(paths[0]):
        assert pm.implementation() == pm.FALLBACK_IMPLEMENTATION
        pm.exp(np.array([2.]))
        pm.exp(np.array([3.]))
    error = capsys.readouterr().err
    assert error.count("[portable-math] WORKAROUND") == 1
    assert str(paths[0]) in error and "predates" in error
    assert not pm._warned and pm._absent_reason == "cached implicit fallback"


def test_private_column_workers_carry_only_selected_math_binding(libraries, monkeypatch):
    from gpuwm.ingest import real
    paths, _, _ = libraries
    model_context = ContextVar("unrelated_model_context", default=None)
    token = model_context.set("member-owned-model-context")
    monkeypatch.setattr(real, "_return_freed_host_memory", lambda: None)
    try:
        with pm.cpu_bridge_scope(paths[0]):
            with real._column_pool(2) as executor:
                result = executor.submit(lambda: (_word(), model_context.get(), cpu_backend._selected_cpu_bridge.get())).result()
                assert result == (75., None, paths[0])
                assert list(executor.map(lambda value: (_word(), model_context.get()), (0, 1))) == [(75., None), (75., None)]
        assert model_context.get() == "member-owned-model-context"
    finally:
        model_context.reset(token)
    assert pm.current_cpu_bridge_binding() is None


def test_selected_backend_call_scope_preserves_signature_and_actual_bridge(libraries):
    paths, _, _ = libraries
    backend = object.__new__(preprocess_backend.ParallelCpuPreprocessBackend)
    backend._native = SimpleNamespace(path=paths[1])
    def original(value, *, preprocess_backend="cpu", cpu_bridge=None):
        return value, _word()
    wrapped = preprocess_backend.preprocess_math_call(original)
    assert inspect.signature(wrapped) == inspect.signature(original)
    assert wrapped.__wrapped__ is original
    assert wrapped("original", cpu_bridge=paths[0]) == ("original", 75.)
    assert wrapped("same-original", preprocess_backend=backend) == ("same-original", 103.)
    assert pm.current_cpu_bridge_binding() is None


def test_original_gfs_entry_selects_cli_bridge_before_host_setup_without_env(libraries, monkeypatch):
    from gpuwm import gfs_direct
    paths, _, _ = libraries
    reached = []
    class StopAfterEntry(RuntimeError):
        pass
    class Date:
        @staticmethod
        def strptime(value, pattern):
            reached.append((_word(), pm.current_cpu_bridge_binding().path))
            raise StopAfterEntry("entry scope observed before source setup")
    monkeypatch.setattr(gfs_direct, "datetime", Date)
    placeholder = Path("source-entry-control")
    with pytest.raises(StopAfterEntry):
        gfs_direct.prepare_gfs_wrf(series=placeholder, cycle="2024-01-25_00:00:00",
            bridge=placeholder, wps_namelist=placeholder, static_input=None,
            static_receipt=None, experiment_config=placeholder, input_manifest=placeholder,
            input_manifest_sha256=None, output_root=placeholder,
            preprocess_backend="cpu", cpu_preprocess_bridge=paths[0])
    assert reached == [(75., paths[0])]
    assert pm.current_cpu_bridge_binding() is None


def test_cpu_relative_humidity_uses_its_selected_original_backend(libraries, monkeypatch):
    paths, _, _ = libraries
    backend = object.__new__(preprocess_backend.ParallelCpuPreprocessBackend)
    backend._native = SimpleNamespace(path=paths[0])
    monkeypatch.setattr(preprocess_backend, "_era5_rh_to_water_cpu", lambda *args: (args, _word()))
    first, second = object(), object()
    assert backend.era5_rh_to_water(first, second) == ((first, second), 75.)
    assert pm.current_cpu_bridge_binding() is None


def test_nested_finalization_selection_comes_from_original_prepared_child(libraries):
    paths, _, _ = libraries
    backend = object.__new__(preprocess_backend.ParallelCpuPreprocessBackend)
    backend._native = SimpleNamespace(path=paths[1])
    prepared = SimpleNamespace(preprocess_backend=backend)
    @preprocess_backend.preprocess_math_call(prepared_parameter="prepared")
    def finalize(prepared, parent):
        return prepared, parent, _word()
    parent = object()
    assert finalize(prepared, parent) == (prepared, parent, 103.)
    assert pm.current_cpu_bridge_binding() is None


def test_native_helper_scope_has_explicit_path_precedence_and_restores_implicit_ladder(tmp_path, monkeypatch):
    first, second, implicit = (tmp_path / name for name in ("first.dll", "second.dll", "implicit.dll"))
    for path in (first, second, implicit):
        path.write_bytes(b"original-native-resolver-control")
    from gpuwm import bridges
    monkeypatch.setattr(bridges, "find_artifact", lambda *args: implicit)
    assert ORIGINAL_RESOLVE_CPU_BRIDGE() == implicit
    with cpu_backend.cpu_bridge_scope(first):
        assert ORIGINAL_RESOLVE_CPU_BRIDGE() == first
        assert ORIGINAL_RESOLVE_CPU_BRIDGE(second) == second
        with cpu_backend.cpu_bridge_scope(second):
            assert ORIGINAL_RESOLVE_CPU_BRIDGE() == second
        assert ORIGINAL_RESOLVE_CPU_BRIDGE() == first
    assert ORIGINAL_RESOLVE_CPU_BRIDGE() == implicit


def test_explicit_host_math_receipt_is_owned_and_implicit_receipt_words_unchanged(libraries, monkeypatch):
    paths, _, _ = libraries
    backend = object.__new__(preprocess_backend.ParallelCpuPreprocessBackend)
    backend._native = SimpleNamespace(path=paths[0], abi_version=1)
    backend.workers, backend.selection, backend._vertical_routes = 1, None, []
    # 2.8.5's parallelism receipt reads the worker count the caller asked for.
    backend.requested_workers = 1
    monkeypatch.setattr(preprocess_backend, "_implementation_tree", lambda *_: {"original": True})
    monkeypatch.setattr(preprocess_backend, "_shared_contracts", lambda: {"original": True})
    monkeypatch.setattr(preprocess_backend, "_masked_chain_receipt", lambda *args, **kwargs: {"original": True})
    # The parallelism block reads live host memory headroom, which moved between two receipts
    # on the 2.8.6 gate host while the GPU shards ran beside Stage 1; this test is about the
    # bridge scope, so the host is held still.
    from gpuwm.ingest import preparation_workers
    monkeypatch.setattr(preparation_workers, "host_available_bytes", lambda *args, **kwargs: 64 << 30)
    original = backend.receipt()
    with pm.cpu_bridge_scope(paths[0]):
        assert backend.receipt() == original
    with pm.cpu_bridge_scope(paths[0], publish_selection=True):
        receipt = backend.receipt()
        assert receipt["host_setup_math"]["implementation"] == pm.IMPLEMENTATION
        assert receipt["host_setup_math"]["bridge"]["name"] == paths[0].name
        assert {key: value for key, value in receipt.items() if key != "host_setup_math"} == original
    assert backend.receipt() == original
    with pm.cpu_bridge_scope(paths[1], publish_selection=True):
        with pytest.raises(ValueError, match="active host setup math library"):
            backend.receipt()


def test_worker_backend_inherits_original_explicit_selection_instead_of_adding_one(libraries, monkeypatch):
    paths, _, _ = libraries
    backend = object.__new__(preprocess_backend.ParallelCpuPreprocessBackend)
    backend._native = SimpleNamespace(path=paths[0])
    backend._vertical_routes, backend.selection = [], None
    backend._explicit_math_selection = False
    monkeypatch.setattr(preprocess_backend.ParallelCpuPreprocessBackend, "__init__", lambda self, **kwargs: None)
    slot = backend.at_workers(3)
    assert not slot._explicit_math_selection and slot._vertical_routes is backend._vertical_routes
    backend._explicit_math_selection = True
    assert backend.at_workers(3)._explicit_math_selection


def test_original_hrrr_cli_scopes_selected_bridge_before_native_host_setup(libraries, monkeypatch):
    from tools import prepare_hrrr_wrf
    paths, _, _ = libraries
    args = SimpleNamespace(physical_input_provider=None, physical_member_index=None,
        physical_input_store=None, physical_output_store=None, physical_base_prepared=None,
        as_posted=None, sealed_prepared_cache=False, extend_root_preparation=None,
        supplement=(), preprocess_backend="cpu", cpu_preprocess_bridge=paths[0], preprocess_workers=1)
    monkeypatch.setattr(prepare_hrrr_wrf, "_parser", lambda: SimpleNamespace(parse_args=lambda argv: args))
    monkeypatch.setattr(prepare_hrrr_wrf, "_as_posted_refusal", lambda value: None)
    monkeypatch.setattr(prepare_hrrr_wrf, "_configured_defaults", lambda value: None)
    observed = []
    class StopAfterEntry(RuntimeError):
        pass
    def explain(value):
        observed.append((_word(), pm.current_cpu_bridge_binding().publish_selection))
        raise StopAfterEntry("original CLI host setup scope reached")
    monkeypatch.setattr(prepare_hrrr_wrf.explain, "set_explain", explain)
    with pytest.raises(StopAfterEntry):
        prepare_hrrr_wrf.main([])
    assert observed == [(75., True)]
    assert pm.current_cpu_bridge_binding() is None


def test_selected_math_budget_caps_main_and_private_native_calls(libraries, monkeypatch):
    from gpuwm.ingest import real
    paths, _, _ = libraries
    monkeypatch.setattr(preprocess_backend, "available_cpu_count", lambda: 3)
    monkeypatch.setattr(real, "_return_freed_host_memory", lambda: None)
    with preprocess_backend.preprocessing_math_scope("cpu", cpu_bridge=paths[0], workers=1):
        assert pm._workers(None) == 1 and pm._workers(8) == 1
        assert cpu_backend._workers(8, 20) == 1
    with preprocess_backend.preprocessing_math_scope("cpu", cpu_bridge=paths[0], workers=12):
        assert pm._workers(None) <= 3 and pm._workers(12) == 3
        with pm.worker_limit(12):
            assert pm._workers(None) == 3
        assert cpu_backend._workers(12, 20) == 3
        with real._column_pool(2) as executor:
            assert executor.submit(lambda: (pm._workers(12), cpu_backend._workers(12, 20))).result() == (1, 1)
    assert cpu_backend._selected_cpu_worker_cap.get() is None
