"""Cold proof reuse must not hide changed headers or failed execution."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest

from gpuwm import forecast_readiness as ready


def test_header_bytes_and_names_are_bound(tmp_path):
    (tmp_path / "nested").mkdir()
    header = tmp_path / "nested" / "type_traits"
    header.write_text("one")
    first = ready._files(tmp_path)
    header.write_text("two")
    assert ready._files(tmp_path) != first
    header.unlink()
    with pytest.raises(ValueError, match="empty"):
        ready._files(tmp_path)


def test_linked_header_directory_declines_reuse(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "header").write_text("value")
    (tmp_path / "link").symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="linked"):
        ready._files(tmp_path)


def _setup(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())
    state = {"compiler": {"sha": "first"}, "headers": "first"}
    monkeypatch.setattr(ready, "binding", lambda cp, source: dict(state))
    calls = []
    def execute(source, directory, identity=None):
        calls.append(directory)
        (directory / "cupy").mkdir(exist_ok=True)
        (directory / "cupy" / "image.cubin").write_bytes(b"actual-compiled-image")
        return {"self_contained": "ok", "toolkit_headers": "ok"}
    monkeypatch.setattr(ready, "_execute", execute)
    return state, calls


def test_reuses_only_bound_success_and_still_executes(monkeypatch, tmp_path):
    state, calls = _setup(monkeypatch)
    first = ready.run("probe", root=tmp_path)
    second = ready.run("probe", root=tmp_path)
    assert first["readiness_cache"] == "bound-cold"
    assert second["readiness_cache"] == "bound-warm"
    assert len(calls) == 2 and calls[0] == calls[1]
    state["headers"] = "changed"
    third = ready.run("probe", root=tmp_path)
    assert third["readiness_cache"] == "bound-cold"
    assert third["readiness_key"] != first["readiness_key"]


def test_damaged_image_requires_cold_compile(monkeypatch, tmp_path):
    _, calls = _setup(monkeypatch)
    ready.run("probe", root=tmp_path)
    (calls[0] / "cupy" / "image.cubin").write_bytes(b"damaged")
    with pytest.warns(RuntimeWarning, match="rejected"):
        result = ready.run("probe", root=tmp_path)
    assert result["readiness_cache"] == "bound-cold"
    assert calls[0] != calls[1]


def test_failed_run_does_not_publish_readiness(monkeypatch, tmp_path):
    _setup(monkeypatch)
    monkeypatch.setattr(ready, "_execute", lambda *a: {
        "self_contained": "ok", "toolkit_headers": "wrong sum",
        "toolkit_headers_stage": "validation"})
    result = ready.run("probe", root=tmp_path)
    assert result["toolkit_headers"] == "wrong sum"
    assert not list(tmp_path.rglob("ready.json"))


def test_changed_binding_during_probe_is_not_verified(monkeypatch, tmp_path):
    state, _ = _setup(monkeypatch)
    def execute(*args):
        state["headers"] = "changed-during-run"
        return {"self_contained": "ok", "toolkit_headers": "ok"}
    monkeypatch.setattr(ready, "_execute", execute)
    result = ready.run("probe", root=tmp_path)
    assert "changed during" in result["probe"]
    assert not list(tmp_path.rglob("ready.json"))


def test_unbound_layout_runs_cold_and_reports_reason(monkeypatch, tmp_path):
    _, calls = _setup(monkeypatch)
    def unsupported(*args):
        raise ValueError("unsupported test layout")
    monkeypatch.setattr(ready, "binding", unsupported)
    with pytest.warns(RuntimeWarning, match="cold probe"):
        result = ready.run("probe", root=tmp_path)
    assert result["readiness_cache"] == "cold-unbound"
    assert "unsupported" in result["readiness_fallback"]
    assert len(calls) == 1


def test_warm_execution_failure_is_not_cached_success(monkeypatch, tmp_path):
    _setup(monkeypatch)
    ready.run("probe", root=tmp_path)
    monkeypatch.setattr(ready, "_execute", lambda *a: {
        "self_contained": "wrong value 0", "self_contained_stage": "validation",
        "toolkit_headers": "ok"})
    assert ready.run("probe", root=tmp_path)["self_contained"] == "wrong value 0"


def test_forecast_selects_bound_probe_and_doctor_stays_cold(monkeypatch):
    from gpuwm import doctor, go_cli
    monkeypatch.delenv("GPUWM_NO_LOCAL_GPU", raising=False)
    calls = []
    def probe(**kwargs):
        calls.append(kwargs)
        return {"self_contained": "ok", "toolkit_headers": "ok",
                "readiness_cache": "bound-warm" if kwargs else "cold"}
    monkeypatch.setattr(doctor, "_nvrtc_header_probe", probe)
    assert doctor._cuda_headers_check().status == "verified"
    go_cli._require_forecast_device()
    assert calls == [{}, {"reuse_bound": True}]



def test_prepared_route_warms_in_forecast_not_detached_child(monkeypatch, tmp_path):
    from gpuwm import go_cli, kernel_plan, local_gpu
    events = []
    monkeypatch.setattr(kernel_plan, "enabled", lambda: True)
    monkeypatch.setattr(local_gpu, "no_local_gpu", lambda: False)
    monkeypatch.setattr(go_cli, "explain_enabled", lambda args: False)
    monkeypatch.setattr(go_cli, "_kernel_prewarm_status",
                        lambda *args, **kwargs: events.append((args, kwargs)))
    import subprocess
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw:
                        pytest.fail("redundant prewarm child started"))
    go_cli.SpeedHooks.warm_kernels_spawn(
        SimpleNamespace(prepared_root=tmp_path, dry_run=False),
        {"config": tmp_path / "experiment.toml", "root": tmp_path})
    assert events[0][0][1] == "forecast-owned"


def test_unwritable_cache_runs_real_cold_check(monkeypatch, tmp_path):
    _setup(monkeypatch)
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    with pytest.warns(RuntimeWarning):
        result = ready.run("probe", root=blocked)
    assert result["self_contained"] == "ok"
    assert result["readiness_cache"] == "cold-unbound"


def test_probe_rebinds_import_time_cache_backend(monkeypatch, tmp_path):
    class Backend:
        def __init__(self, directory):
            self.directory = directory
    old = Backend("unrelated-cache")
    compiler = SimpleNamespace(_kernel_cache_backend=old,
                               _DiskKernelCacheBackend=Backend)
    compiler._set_kernel_cache_backend = lambda value: setattr(
        compiler, "_kernel_cache_backend", value)
    monkeypatch.setitem(sys.modules, "cupy.cuda", SimpleNamespace(compiler=compiler))
    source = ('import json; from cupy.cuda import compiler; '
              'print(json.dumps({"directory": compiler._kernel_cache_backend.directory}))')
    result = ready._execute(source, tmp_path)
    assert result["directory"] == str(tmp_path / "cupy")
    assert compiler._kernel_cache_backend is old


def test_metadata_without_kernel_images_is_not_a_warm_proof(monkeypatch, tmp_path):
    _setup(monkeypatch)
    result = ready.run("probe", root=tmp_path)
    directory = tmp_path / "readiness" / result["readiness_key"]
    marker = directory / "ready.json"
    row = json.loads(marker.read_text())
    generation = directory / row["generation"]
    for image in (generation / "cupy").iterdir():
        image.unlink()
    (generation / "metadata.json").write_text("{}")
    row["files"] = ready._inventory(generation)
    marker.write_text(json.dumps(row))
    with pytest.warns(RuntimeWarning, match="rejected"):
        assert ready.run("probe", root=tmp_path)["readiness_cache"] == "bound-cold"


def test_cold_fallback_diagnostic_survives_probe_subprocess(monkeypatch, capsys):
    from gpuwm import doctor
    monkeypatch.setattr(doctor, "find_spec", lambda name: object())
    result = {"self_contained": "ok", "toolkit_headers": "ok",
              "readiness_cache": "bound-cold"}
    monkeypatch.setattr(doctor.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(stdout=json.dumps(result),
                                        stderr="rejected cache; compiling cold\n",
                                        returncode=0))
    assert doctor._nvrtc_header_probe(reuse_bound=True) == result
    stderr = capsys.readouterr().err
    assert "rejected cache" in stderr and "current compiler/header binding" in stderr
