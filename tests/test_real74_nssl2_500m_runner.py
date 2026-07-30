"""CPU contracts for the dedicated real74 NSSL-2 500 m launcher."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tomllib

import numpy as np
import pytest

from gpuwm.experiment import build_experiment
from tools import run_real74_nssl2_500m as runner


CONFIG = runner.REPOSITORY_ROOT / "configs" / "real74_nssl2_500m.toml"


def _experiment_from_template():
    with CONFIG.open("rb") as stream:
        raw = tomllib.load(stream)
    raw.pop("case_data")
    return build_experiment(raw, source=str(CONFIG))


def _write_source_inventory(root: Path, manifest_path: Path) -> None:
    files = {}
    for index, relative in enumerate(runner.SOURCE_FILES):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"source-{index}\n".encode())
        identity = runner.file_identity(path)
        files[relative] = {
            "bytes": identity["bytes"], "sha256": identity["sha256"],
        }
    geog = root / runner.SOURCE_TREES[0]
    geog.mkdir(parents=True, exist_ok=True)
    (geog / "index").write_text("geography\n", encoding="utf-8")
    identity = runner.tree_identity(geog)
    manifest_path.write_text(json.dumps({
        "schema": runner.INPUT_MANIFEST_SCHEMA,
        "files": files,
        "trees": {runner.SOURCE_TREES[0]: {
            "file_count": identity["file_count"],
            "bytes": identity["bytes"], "sha256": identity["sha256"],
        }},
    }), encoding="utf-8")


def test_named_config_has_exact_nssl_500m_authority():
    exp = _experiment_from_template()
    observed = runner.validate_topology(exp)
    with CONFIG.open("rb") as stream:
        raw = tomllib.load(stream)

    assert exp.start_time == runner.START_TIME
    assert exp.run_seconds == 43200.0
    assert exp.restart_interval_s == 0.0
    assert [row["mass_shape"] for row in observed] == [
        [200, 250], [400, 500], [501, 501], [400, 400],
    ]
    assert [row["dx_m"] for row in observed] == [12000, 3000, 1000, 500]
    assert [row["dt_s"] for row in observed] == [60, 15, 5, 2.5]
    assert [row["history_interval_s"] for row in observed] == [
        3600, 3600, 3600, 1800,
    ]
    assert {row["mp_physics"] for row in observed} == {18}
    assert [row["epssm"] for row in observed] == [0.5, 0.1, 0.1, 0.1]
    assert all(row["moist_cq"] for row in observed)
    assert "morr_rimed_ice" not in raw["shared"]
    assert raw["case_data"]["forcing"] == [
        f"{runner.TEMPLATE_INPUT_ROOT}/{runner.RUNTIME_FORCING_FILES[0]}"
    ]
    assert set(runner.RUNTIME_FORCING_FILES).isdisjoint(
        runner.WPS_ONLY_FORCING_FILES)
    assert set(runner.RUNTIME_FORCING_FILES) | set(
        runner.WPS_ONLY_FORCING_FILES) <= set(runner.SOURCE_FILES)


def test_topology_rejects_unmatched_epssm_and_moist_cq():
    with CONFIG.open("rb") as stream:
        raw = tomllib.load(stream)
    raw.pop("case_data")
    raw["domain"][1]["epssm"] = 0.5
    with pytest.raises(ValueError, match="d02 epssm"):
        runner.validate_topology(build_experiment(raw, source="bad-epssm"))

    raw["domain"][1]["epssm"] = 0.1
    raw["shared"]["moist_cq"] = False
    with pytest.raises(ValueError, match="moist_cq"):
        runner.validate_topology(build_experiment(raw, source="bad-cq"))


def test_output_calendar_is_13_13_13_25_and_ends_at_00z():
    expected = runner.expected_output_names(_experiment_from_template())

    assert {grid_id: len(names) for grid_id, names in expected.items()} == {
        1: 13, 2: 13, 3: 13, 4: 25,
    }
    assert expected[1][0] == "wrfout_d01_1974-04-03_12_00_00"
    assert expected[4][-1] == "wrfout_d04_1974-04-04_00_00_00"


def test_static_allocation_estimate_retains_single_5090_headroom():
    from gpuwm.core.preflight import estimate_experiment

    estimate = estimate_experiment(_experiment_from_template())
    assert estimate.alloc_estimate_bytes < 24 * 1024**3
    assert estimate.footprint_projection_bytes < 24 * 1024**3


def _write_authority_geometry(root: Path, *, d04_ratio: int = 2) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    wps = root / runner.SOURCE_WPS
    wps.parent.mkdir(parents=True, exist_ok=True)
    wps.write_text(
        "&share\n max_dom = 4,\n/\n"
        "&geogrid\n"
        " parent_id = 1, 1, 2, 3,\n"
        f" parent_grid_ratio = 1, 4, 3, {d04_ratio},\n"
        " i_parent_start = 1, 63, 167, 151,\n"
        " j_parent_start = 1, 51, 117, 151,\n"
        " e_we = 251, 501, 502, 401,\n"
        " e_sn = 201, 401, 502, 401,\n"
        " dx = 12000.0,\n dy = 12000.0,\n/\n",
        encoding="utf-8",
    )
    shapes = {1: (200, 250), 2: (400, 500),
              3: (501, 501), 4: (400, 400)}
    for grid_id, shape in shapes.items():
        path = root / (
            f"real_run/met_em.d{grid_id:02d}.1974-04-03_12_00_00.nc")
        path.parent.mkdir(parents=True, exist_ok=True)
        with netcdf4.Dataset(path, "w") as dataset:
            dataset.createDimension("Time", 1)
            dataset.createDimension("south_north", shape[0])
            dataset.createDimension("west_east", shape[1])
            variable = dataset.createVariable(
                "SOILHGT", "f4", ("Time", "south_north", "west_east"))
            variable[0] = np.zeros(shape, dtype=np.float32)


def test_exact_cpu_authority_is_structurally_500m_and_never_derived(tmp_path):
    _write_authority_geometry(tmp_path)
    report = runner.validate_authority_assets(tmp_path)
    assert report["wps"]["parent_grid_ratio"] == [1, 4, 3, 2]
    assert report["wps"]["e_we"] == [251, 501, 502, 401]
    assert report["orography"]["d04"]["shape"] == [400, 400]
    assert report["derived_from_333m"] is False

    _write_authority_geometry(tmp_path, d04_ratio=3)
    with pytest.raises(ValueError, match="parent_grid_ratio mismatch"):
        runner.validate_authority_assets(tmp_path)


def test_materialized_config_binds_exact_cpu_authority_paths(tmp_path):
    input_root = tmp_path / "input"
    destination = tmp_path / "run" / "effective.toml"
    runner.materialize_config(CONFIG, destination, input_root)
    text = destination.read_text(encoding="utf-8")

    assert runner.TEMPLATE_INPUT_ROOT not in text
    assert (input_root / runner.SOURCE_WPS).resolve().as_posix() in text
    assert (input_root / runner.SOURCE_D04_OROGRAPHY).resolve().as_posix() in text
    assert (input_root / "WPS_run/era5_19740403.grb").resolve(
    ).as_posix() in text
    assert (input_root / runner.WPS_ONLY_FORCING_FILES[0]).resolve(
    ).as_posix() not in text


def test_input_manifest_rejects_missing_and_mismatched_pins(
        tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "KNOWN_AUTHORITY_SHA256", {})
    monkeypatch.setattr(runner, "KNOWN_AUTHORITY_BYTES", {})
    root = tmp_path / "inputs"
    manifest = tmp_path / "manifest.json"
    _write_source_inventory(root, manifest)
    verified = runner.verify_input_manifest(root, manifest)
    assert len(verified["files"]) == len(runner.SOURCE_FILES)
    assert len(verified["trees"]) == 1

    (root / runner.SOURCE_FILES[0]).write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        runner.verify_input_manifest(root, manifest)

    _write_source_inventory(root, manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["files"].pop(runner.SOURCE_FILES[-1])
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="inventory mismatch"):
        runner.verify_input_manifest(root, manifest)


def test_input_manifest_rejects_changed_geography_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "KNOWN_AUTHORITY_SHA256", {})
    monkeypatch.setattr(runner, "KNOWN_AUTHORITY_BYTES", {})
    root = tmp_path / "inputs"
    manifest = tmp_path / "manifest.json"
    _write_source_inventory(root, manifest)
    (root / runner.SOURCE_TREES[0] / "index").write_text(
        "changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="tree identity mismatch"):
        runner.verify_input_manifest(root, manifest)


def test_manifest_must_match_registered_cpu_500m_sha(tmp_path, monkeypatch):
    root = tmp_path / "inputs"
    manifest = tmp_path / "manifest.json"
    _write_source_inventory(root, manifest)
    monkeypatch.setattr(runner, "KNOWN_AUTHORITY_SHA256", {
        runner.SOURCE_WPS: "0" * 64,
    })
    monkeypatch.setattr(runner, "KNOWN_AUTHORITY_BYTES", {})

    with pytest.raises(ValueError, match="registered CPU 500 m authority"):
        runner.verify_input_manifest(root, manifest)


def test_vtable_cpu_authority_pin_is_registered():
    relative = "WPS_run/Vtable.ERA5_CDO"
    assert runner.KNOWN_AUTHORITY_SHA256[relative] == (
        "64282b5b35ac7302e274f764327923080883f164f4e605ef06529d1baef6620e")
    assert runner.KNOWN_AUTHORITY_BYTES[relative] == 4256


def test_disk_and_forcing_preflights_fail_closed(tmp_path, monkeypatch):
    assert runner.MINIMUM_FREE_BYTES == 75 * 1024**3
    assert runner.MINIMUM_FREE_BYTES == (
        runner.EXPECTED_OUTPUT_BYTES + runner.PUBLICATION_TEMP_BYTES
        + runner.POST_RUN_RESERVE_BYTES)
    monkeypatch.setattr(
        runner.shutil, "disk_usage",
        lambda _path: SimpleNamespace(free=runner.MINIMUM_FREE_BYTES - 1),
    )
    with pytest.raises(RuntimeError, match="disk preflight failed"):
        runner.require_disk_headroom(tmp_path)

    short = SimpleNamespace(
        run_ceiling_seconds=runner.RUN_SECONDS - 1,
        raise_for_failures=lambda: None,
    )
    with pytest.raises(ValueError, match="forcing ceiling"):
        runner.require_input_preflight(short)

    class Failed:
        run_ceiling_seconds = runner.RUN_SECONDS

        @staticmethod
        def raise_for_failures():
            raise ValueError("input failure")

    with pytest.raises(ValueError, match="input failure"):
        runner.require_input_preflight(Failed())


def test_measured_allocation_gate_rejects_command_failure(
        tmp_path, monkeypatch):
    from gpuwm import supervisor

    gpu = SimpleNamespace(uuid="GPU-test", driver_version="1", name="test")
    monkeypatch.setattr(supervisor, "select_gpu", lambda _uuid: gpu)
    monkeypatch.setattr(
        supervisor, "preflight_exclusive_gpu", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner.subprocess, "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=3, stdout="{}", stderr="allocation failed"),
    )
    config = tmp_path / "effective.toml"
    config.write_text("config\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="failed closed"):
        runner.gpu_allocation_preflight(
            config, tmp_path / "run", "GPU-test")


def test_measured_allocation_gate_requires_all_true_and_measurement(
        tmp_path, monkeypatch):
    from gpuwm import supervisor

    gpu = SimpleNamespace(uuid="GPU-test", driver_version="1", name="test")
    monkeypatch.setattr(supervisor, "select_gpu", lambda _uuid: gpu)
    monkeypatch.setattr(
        supervisor, "preflight_exclusive_gpu", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner.subprocess, "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"gates": {"fits": False}, "alloc": {}}),
            stderr=""),
    )
    config = tmp_path / "effective.toml"
    config.write_text("config\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="complete passing measurement"):
        runner.gpu_allocation_preflight(
            config, tmp_path / "run", "GPU-test")


def _write_tiny_wrfout(path: Path, *, complete: int = 1,
                       qhail: float = 0.0) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    with netcdf4.Dataset(path, "w") as dataset:
        dataset.createDimension("Time", 1)
        dataset.createDimension("DateStrLen", 19)
        dataset.createDimension("west_east", 2)
        dataset.createDimension("south_north", 2)
        dataset.createDimension("bottom_top", 1)
        dataset.createDimension("west_east_stag", 3)
        dataset.createDimension("south_north_stag", 3)
        dataset.createDimension("bottom_top_stag", 2)
        times = dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[0, :] = np.asarray(
            list("1974-04-03_12:00:00"), dtype="S1")
        for name in runner.REQUIRED_HISTORY_VARIABLES:
            if name == "Times":
                continue
            variable = dataset.createVariable(name, "f4", ("Time",))
            variable[:] = qhail if name == "QHAIL" else 0.0
        dataset.GPUWM_WRITE_COMPLETE = complete
        dataset.TITLE = "NSSL test"


def test_wrfout_health_check_requires_publication_and_finite_nssl_fields(
        tmp_path):
    dc = SimpleNamespace(run=SimpleNamespace(nx=2, ny=2, nz=1))
    path = tmp_path / "wrfout"
    _write_tiny_wrfout(path, complete=0)
    with pytest.raises(ValueError, match="completion marker"):
        runner.verify_one_wrfout(
            path, dc, runner.START_TIME, "NSSL test")

    _write_tiny_wrfout(path, qhail=float("nan"))
    with pytest.raises(ValueError, match="QHAIL contains non-finite"):
        runner.verify_one_wrfout(
            path, dc, runner.START_TIME, "NSSL test")

    _write_tiny_wrfout(path, qhail=1.0e-6)
    identity = runner.verify_one_wrfout(
        path, dc, runner.START_TIME, "NSSL test")
    assert identity["extrema"]["QHAIL"]["maximum"] == pytest.approx(1.0e-6)


def test_fatal_log_and_existing_run_artifacts_are_rejected(tmp_path):
    log = tmp_path / "worker-01.stderr.log"
    log.write_text("CUDA error: illegal memory address\n", encoding="utf-8")
    with pytest.raises(ValueError, match="health marker"):
        runner._reject_fatal_logs(tmp_path)

    log.write_text("", encoding="utf-8")
    assert len(runner._reject_fatal_logs(tmp_path)) == 1
    (tmp_path / "gpuwmrst_d01_bad.npz").write_bytes(b"restart")
    with pytest.raises(ValueError, match="already contains"):
        runner._assert_unstarted(tmp_path)


def test_supervisor_config_is_fixed_argv_nonrestarting_and_durable(
        tmp_path, monkeypatch):
    def portable_supervisor_path(path, *, label):
        resolved = str(path.resolve())
        if any(character.isspace() for character in resolved):
            raise ValueError(f"{label} is not Supervisor-safe: {resolved!r}")
        return resolved

    monkeypatch.setattr(runner, "_supervisor_path", portable_supervisor_path)
    input_root = tmp_path / "inputs"
    manifest = tmp_path / "manifest.json"
    registration = tmp_path / "comparison-registration.json"
    policy = tmp_path / "comparison-policy.json"
    run_dir = tmp_path / "run"
    rendered = runner.render_supervisor_config(
        input_root=input_root, input_manifest=manifest,
        comparison_registration=registration, comparison_policy=policy,
        run_dir=run_dir,
        gpu_uuid="GPU-01234567-abcd")

    assert f"[program:{runner.SUPERVISOR_SERVICE}]" in rendered
    assert " run --run-dir " in rendered
    assert f" --input-root {input_root.resolve()} " in rendered
    assert f" --input-manifest {manifest.resolve()} " in rendered
    assert f" --comparison-registration {registration.resolve()} " in rendered
    assert f" --comparison-policy {policy.resolve()} " in rendered
    assert " --gpu-uuid GPU-01234567-abcd" in rendered
    assert "autostart=false" in rendered
    assert "autorestart=false" in rendered
    assert f"stdout_logfile={run_dir.resolve() / 'controller.stdout.log'}" in rendered
    assert f"stderr_logfile={run_dir.resolve() / 'controller.stderr.log'}" in rendered

    with pytest.raises(ValueError, match="Supervisor-safe"):
        runner.render_supervisor_config(
            input_root=tmp_path / "unsafe path", input_manifest=manifest,
            comparison_registration=registration, comparison_policy=policy,
            run_dir=run_dir, gpu_uuid="GPU-01234567-abcd")


def test_supervisor_launcher_installs_without_shell_and_reports_running(
        tmp_path, monkeypatch):
    from gpuwm import supervisor

    gpu = SimpleNamespace(uuid="GPU-test-123", driver_version="1", name="test")
    monkeypatch.setattr(
        runner, "_supervisor_path",
        lambda path, *, label: str(path.resolve()))
    monkeypatch.setattr(supervisor, "select_gpu", lambda _uuid: gpu)
    status_calls = 0

    def fake_supervisorctl(*arguments, allow_missing=False):
        nonlocal status_calls
        del allow_missing
        if arguments[0] == "status":
            status_calls += 1
            stdout = ("no such process" if status_calls == 1 else
                      f"{runner.SUPERVISOR_SERVICE} RUNNING pid 123")
        else:
            stdout = "ok"
        return runner.subprocess.CompletedProcess(
            ["supervisorctl", *arguments], 0, stdout, "")

    monkeypatch.setattr(runner, "_supervisorctl", fake_supervisorctl)
    config = tmp_path / "supervisor" / "run.conf"
    result = runner.launch_under_supervisor(
        input_root=tmp_path / "inputs", input_manifest=tmp_path / "manifest",
        comparison_registration=tmp_path / "comparison-registration.json",
        comparison_policy=tmp_path / "comparison-policy.json",
        run_dir=tmp_path / "run", gpu_uuid=None, config_path=config)

    assert result["service"] == runner.SUPERVISOR_SERVICE
    assert "RUNNING" in result["status"]
    assert config.is_file()
    assert "autorestart=false" in config.read_text(encoding="utf-8")
