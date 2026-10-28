"""CPU staging and cache ownership must not occupy or serialize a GPU run."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import sys
from threading import Event
from types import ModuleType, SimpleNamespace

import pytest

from gpuwm import earth2_run, earth2_runner


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    # No optional runtime, numerical model, download or device is used.
    modules = {name: ModuleType(name) for name in
               ("earth2studio", "earth2studio.models", "earth2studio.models.auto",
                "earth2studio.data", "numpy", "torch")}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    package = SimpleNamespace(default_cache=lambda name: str(tmp_path / name))
    modules["earth2studio.models.auto"].Package = package
    modules["numpy"].array = lambda value: value
    modules["numpy"].datetime64 = lambda value: value
    torch = modules["torch"]
    torch.set_num_threads = lambda _: None
    torch.device = lambda value: value
    torch.save = lambda value, path: path.write_text(value)
    calls = []

    class Data:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fetch_data(**kwargs):
        assert kwargs["device"] == "cpu"
        calls.append(kwargs)
        return SimpleNamespace(squeeze=lambda: json.dumps({
            "time": kwargs["time"], "variable": kwargs["variable"],
            "kwargs": kwargs["source"].kwargs})), None

    modules["earth2studio.data"].Data = Data
    modules["earth2studio.data"].fetch_data = fetch_data
    return calls


def request():
    return {"pinned_version": "0.19.0", "init": "2024-09-24T12:00:00",
            "ic_class": "Data", "ic_kwargs": {},
            "loader_preparations": [{"cache_subdir": "inputs", "filename": "state.pt",
                                     "format": "torch", "time": "$init",
                                     "variables": ["land"]}]}


def test_empty_preparation_needs_no_runtime_or_lock(monkeypatch):
    monkeypatch.setitem(sys.modules, "earth2studio", None)
    monkeypatch.setattr(earth2_runner, "exclusive_lock", lambda *_: pytest.fail("unused loader acquired a lock"))
    with earth2_runner.loader_cache_lock({}):
        assert earth2_runner.prepare_loader_inputs({}) == []


def test_model_without_loader_inputs_constructs_without_cache_lock(runtime, monkeypatch, tmp_path):
    import importlib.metadata
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "0")
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.19.0")
    monkeypatch.setattr(earth2_runner, "exclusive_lock",
                        lambda *_: pytest.fail("model without inputs acquired cache lock"))
    for name in ("earth2studio.models.px", "earth2studio.run", "earth2studio.perturbation"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["earth2studio.perturbation"].Zero = object
    sys.modules["numpy"].random = SimpleNamespace(seed=lambda _: None)
    torch = sys.modules["torch"]
    torch.manual_seed = lambda _: None
    torch.cuda = SimpleNamespace(is_available=lambda: True,
                                 manual_seed_all=lambda _: None,
                                 get_device_capability=lambda _: (9, 0))
    monkeypatch.setattr(earth2_runner, "PackageAudit", lambda *_: SimpleNamespace(finish=lambda: []))

    class ConstructionReached(Exception):
        pass

    def construct(*_, **__):
        raise ConstructionReached

    sys.modules["earth2studio.models.px"].Example = SimpleNamespace(
        load_default_package=lambda: object(), load_model=construct)
    path = tmp_path / "request.json"
    path.write_text(json.dumps(dict(request(), loader_preparations=[], model_class="Example",
                                    seed=0, output=str(tmp_path / "output.nc"),
                                    weight_cache=str(tmp_path), data_cache=str(tmp_path))))
    with pytest.raises(ConstructionReached):
        earth2_runner.main(["runner", str(path)])


def test_staging_entry_point_is_cpu_only_with_device_opt_out(monkeypatch, tmp_path, capsys):
    import importlib.metadata
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.19.0")
    monkeypatch.setitem(sys.modules, "earth2studio", None)
    path = tmp_path / "request.json"
    path.write_text(json.dumps(dict(request(), loader_preparations=[],
                                    weight_cache=str(tmp_path / "weights"),
                                    data_cache=str(tmp_path / "data"))))
    assert earth2_runner.main(["runner", "--prepare-loader-inputs", str(path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["loader_inputs"] == []
    assert report["model_constructed"] is False and report["gpu_used"] is False
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "-1"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["request.json"]


def test_staging_is_reused_but_cycle_variables_and_bytes_are_bound(runtime):
    selected = request()
    with earth2_runner.loader_cache_lock(selected):
        first, = earth2_runner.prepare_loader_inputs(selected)
        cached, = earth2_runner.prepare_loader_inputs(selected)
        assert not first["cache_hit"] and cached["cache_hit"]
        assert len(runtime) == 1 and first["sha256"] == cached["sha256"]
        selected["init"] = "2024-09-25T12:00:00"
        changed, = earth2_runner.prepare_loader_inputs(selected)
        assert not changed["cache_hit"] and changed["sha256"] != first["sha256"]
        selected["loader_preparations"][0]["variables"] = ["terrain"]
        changed, = earth2_runner.prepare_loader_inputs(selected)
        assert not changed["cache_hit"]
        Path(changed["path"]).write_text("corrupt")
        repaired, = earth2_runner.prepare_loader_inputs(selected)
        assert not repaired["cache_hit"] and repaired["sha256"] == changed["sha256"]
        Path(repaired["path"] + ".gpuwm.json").write_text("null")
        repaired, = earth2_runner.prepare_loader_inputs(selected)
        assert not repaired["cache_hit"]
    assert len(runtime) == 5


def test_cache_locks_protect_same_inputs_without_blocking_other_models(runtime):
    selected = request()
    distinct = request()
    distinct["loader_preparations"][0]["cache_subdir"] = "other"
    attempting, entered = Event(), Event()

    def acquire(value, announce=None):
        attempting.set()
        with earth2_runner.loader_cache_lock(value):
            if announce:
                announce.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        with earth2_runner.loader_cache_lock(selected):
            waiting = pool.submit(acquire, selected, entered)
            assert attempting.wait(2)
            independent = pool.submit(acquire, distinct)
            independent.result(timeout=2)
            assert not entered.is_set()
        waiting.result(timeout=2)
    assert entered.is_set()


@pytest.mark.parametrize("force", [False])
def test_host_preparation_runs_cpu_stage_without_gpu_or_replacing_forecasts(monkeypatch, tmp_path, force):
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    monkeypatch.setattr(earth2_run, "earth2_python", lambda: sys.executable)
    monkeypatch.setenv("GPUWM_EARTH2_CACHE", str(tmp_path / "weights"))
    from gpuwm import hf_fetch
    original = earth2_run.resolve_model
    monkeypatch.setattr(earth2_run, "resolve_model", lambda model:
        (original(model)[0], dict(original(model)[1], weights={"host": "huggingface"})))
    monkeypatch.setattr(hf_fetch, "fetch", lambda *_: {"package_root": str(tmp_path / "package")})
    monkeypatch.setattr(earth2_run, "require_inference_vram", lambda *_: pytest.fail("CPU stage probed GPU"))
    donors = []
    monkeypatch.setattr(earth2_run, "_donor", lambda **kwargs: donors.append(kwargs) or {"parallelism": "4 threads"})
    forecast = tmp_path / "earth2-fcn3-20240924T12-m0000-f0000.nc"
    forecast.write_bytes(b"existing frame")

    def run(command, **kwargs):
        assert command[-2] == "--prepare-loader-inputs"
        assert Path(command[-1]).name == "earth2-loader-request.json"
        assert json.loads(Path(command[-1]).read_text())["package_root"] == str(tmp_path / "package")
        kwargs["stdout"].write(json.dumps({"schema": "gpuwm-earth2-loader-preparation-v2", "stages": [],
                                          "model_constructed": True, "gpu_used": False}) + "\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(earth2_run.subprocess, "run", run)
    report = earth2_run.run_model(model="fcn3", cycle=datetime(2024, 9, 24, 12),
                                  hours=24, area=None, out=tmp_path,
                                  prepare_only=True, force=force)
    assert report["gpu_used"] is False and len(donors) == 1
    assert [item["stage"] for item in report["stages"]] == ["donor"]
    assert forecast.read_bytes() == b"existing frame"
    assert not (tmp_path / earth2_run.RECEIPT).exists()


@pytest.mark.parametrize("hidden", ["", "-1"])
def test_fetch_door_runs_cpu_stage_when_cuda_is_hidden(monkeypatch, tmp_path, capsys, hidden):
    # G12 2026-10-20: `CUDA_VISIBLE_DEVICES= gpuwm fetch --source earth2-fcn3 ...`
    # refused ("inference needs 30 GiB of GPU memory; CUDA devices are hidden").
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", hidden)
    monkeypatch.delenv("GPUWM_NO_LOCAL_GPU", raising=False)
    calls = []
    monkeypatch.setattr(earth2_run, "run_model", lambda **kwargs: calls.append(kwargs) or
                        {"wall_seconds": 1.0, "stages": [{"stage": "model_construct_cpu", "seconds": 1.0}]})
    from gpuwm.cli import build_parser
    args = build_parser().parse_args(["fetch", "--source", "earth2-fcn3", "--cycle", "2024-10-08T00",
                                      "--hours", "24", "--out", str(tmp_path)])
    assert earth2_run.fetch(args) == 0
    assert calls[0]["prepare_only"] is True
    assert "No forecast was generated" in capsys.readouterr().out


@pytest.mark.parametrize("prepare_only", [False, True])
def test_external_sealed_fetch_reuses_before_cpu_or_gpu_admission(monkeypatch, tmp_path, prepare_only):
    import hashlib
    from netCDF4 import Dataset

    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    monkeypatch.setattr(earth2_run, "earth2_python", lambda: pytest.fail("reuse probed model runtime"))
    monkeypatch.setattr(earth2_run, "require_inference_vram", lambda *_: pytest.fail("reuse probed GPU"))
    producer = tmp_path / "producer"
    producer.mkdir()
    model_id, row = earth2_run.resolve_model("fcn3")
    cycle = datetime(2024, 9, 24, 12)
    files = []
    for lead in (0, 6):
        path = producer / f"frame-{lead}.nc"
        with Dataset(path, "w") as frame:
            frame.setncattr("earth2_member", 0)
            frame.setncattr("earth2_generation_seed", 333)
        files.append(dict(path=str(path), member=0, seed=333, lead_hours=lead,
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    analysis = producer / "donor.nc"
    analysis.write_bytes(b"sealed donor")
    receipt = dict(schema="gpuwm-earth2-fetch-v2", source="earth2-fcn3", model=model_id,
                   model_row=row, model_table_sha256=hashlib.sha256(earth2_run.TABLE.read_bytes()).hexdigest(),
                   initial_condition_source="era5", cycle=cycle.isoformat(), hours=6,
                   forecast_start_hour=0, area=None, members=[0], runner=dict(files=files, seed=333),
                   donor=dict(analysis=str(analysis), sha256=hashlib.sha256(analysis.read_bytes()).hexdigest(),
                              supplements={"physical_analysis_surface_data": str(analysis)}))
    parent = producer / earth2_run.RECEIPT
    parent.write_text(json.dumps(receipt))
    before = parent.read_bytes()
    output = tmp_path / "consumer"
    report = earth2_run.run_model(model=model_id, cycle=cycle, hours=6, area=None, out=output,
                                  reuse_fetch=producer, prepare_only=prepare_only)
    assert report["reuse"]["verified_files"] == 2
    assert report["reuse"]["root"] == str(producer)
    assert (output / "inputs-m0000.txt").read_text().splitlines() == [item["path"] for item in files]
    assert json.loads((output / earth2_run.RECEIPT).read_text())["runner"]["files"] == files
    assert parent.read_bytes() == before
    assert not list(output.glob("*.nc"))
    assert not (output / "earth2-loader-request.json").exists()
