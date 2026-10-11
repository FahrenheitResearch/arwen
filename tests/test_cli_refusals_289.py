"""Regression receipts for the CLI defects retained at final 2.8.8 E."""
from __future__ import annotations

import argparse
import importlib
import json
from importlib.metadata import PathDistribution
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def cli(*args, timeout=120):
    return subprocess.run([sys.executable, "-m", "gpuwm.cli", *map(str, args)],
                          capture_output=True, text=True, timeout=timeout)


@pytest.mark.parametrize("source", ["20crv3", "era5-l137"])
def test_D17_local_source_readiness_is_a_readable_refusal(tmp_path, source):
    config = tmp_path / "experiment.toml"
    emitted = cli("domain", "--point=41.97,-93.6", "--point-extent-km", "450",
                  "--root-dx", "12", "--vram-gib", "96", "--source", source,
                  "--cycle", "2024-05-21T18", "--out", config)
    assert emitted.returncode == 0, emitted.stderr
    answer = cli("go", config, "--readiness", "--no-probe")
    assert answer.returncode == 2, answer.stdout + answer.stderr
    assert "usage:" not in answer.stderr, answer.stderr
    assert "argparse refused" not in answer.stderr
    assert "local input" in (answer.stdout + answer.stderr).lower()
    assert "--data-dir" in answer.stdout + answer.stderr


def test_D16_source_local_metadata_is_editable(tmp_path):
    from gpuwm.version_cli import describe_install, local_report
    package = tmp_path / "gpuwm"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (tmp_path / "pyproject.toml").write_text('[project]\nname="gpuwm"\nversion="2.8.8"\n')
    info = tmp_path / "gpuwm.egg-info"
    info.mkdir()
    (info / "PKG-INFO").write_text("Metadata-Version: 2.1\nName: gpuwm\nVersion: 2.8.8\n")
    (info / "SOURCES.txt").write_text("gpuwm/__init__.py\npyproject.toml\n")
    shape = describe_install(package, PathDistribution(info))
    assert shape["editable"]
    assert "Installed as a wheel" not in "\n".join(local_report(shape))


def test_D17_a_directory_name_does_not_prove_local_input_coverage(tmp_path):
    from gpuwm.go_cli import _local_input_readiness

    document, code = _local_input_readiness({"source": "20crv3", "source_root": str(tmp_path)})
    assert code == 2 and not document["ready"]
    assert "fields and time coverage" in document["refusal"]


def test_D18_catalog_header_describes_the_written_physics(tmp_path):
    from gpuwm.physics_catalog import apply_to_experiment
    import tomllib
    config = tmp_path / "experiment.toml"
    emitted = cli("domain", "--point=38.5,-98", "--point-extent-km", "90",
                  "--root-dx", "3", "--vram-gib", "96", "--source", "gfs",
                  "--cycle", "2026-10-08T18", "--out", config)
    assert emitted.returncode == 0, emitted.stderr
    text = apply_to_experiment(config.read_text(), {"choices": {
        "pbl": "myj", "surface_layer": "eta-similarity"}, "source": "gfs"})
    doc = tomllib.loads(text)
    root = {**doc["shared"], **doc["domain"][0]}
    assert root["bl_pbl_physics"] == 2
    header = next(line for line in text.splitlines() if line.startswith("# PHYSICS:"))
    assert "bl_pbl_physics 2" in header
    assert "bl_pbl_physics 1" not in header


@pytest.mark.parametrize("relative", ["", "wrfout", "run/wrfout", "chain/run/wrfout"])
@pytest.mark.parametrize("separator", ["_", "-", pytest.param(":", marks=pytest.mark.skipif(
    sys.platform == "win32", reason="Windows forbids colon filenames"))])
def test_D15_render_folds_the_go_folder_into_frames(tmp_path, relative, separator):
    from gpuwm.render import _fold_render_inputs
    directory = tmp_path / relative
    directory.mkdir(parents=True, exist_ok=True)
    frames = [directory / f"wrfout_d01_2026-10-08_0{hour}{separator}00{separator}00.nc" for hour in (0, 1)]
    for frame in frames:
        frame.write_bytes(b"fixture name only; decoding is tested by the renderer suite")
    args = argparse.Namespace(wrfout=[tmp_path], inputs_from=None, context_wrfout=[])
    assert _fold_render_inputs(args) is None
    assert args.wrfout == frames


def test_D15_empty_directory_refuses_before_render(tmp_path):
    from gpuwm.render import _fold_render_inputs
    args = argparse.Namespace(wrfout=[tmp_path], inputs_from=None, context_wrfout=[])
    refusal = _fold_render_inputs(args)
    assert refusal and "history frames" in refusal


@pytest.mark.parametrize("relative", ["run", "chain/run"])
def test_E_enprod_accepts_the_go_folder(tmp_path, relative):
    from gpuwm.da import enprod
    from test_enprod_v2_manifest import _v2_run
    outer = tmp_path / "go-output"
    run, manifest = _v2_run(outer if relative == "run" else outer / "chain")
    assert enprod.manifest_path(outer) == manifest
    assert enprod.load_manifest(outer).root == enprod.load_manifest(run).root


def test_E_doctor_reports_actual_probe_timeout(tmp_path, monkeypatch):
    from gpuwm import doctor
    module = "cli_probe_stall_fixture"
    (tmp_path / f"{module}.py").write_text("import time\ntime.sleep(2)\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setattr(doctor, "_PROBE_TIMEOUT_S", 0.025)
    importlib.invalidate_caches()
    ok, evidence = doctor._import_probe(module)
    assert not ok
    assert "import probe timed out after" in evidence
    assert "failed to import" not in evidence
    assert "0.25" in evidence


def test_E_doctor_child_interrupt_is_not_an_import_failure(tmp_path, monkeypatch):
    from gpuwm import doctor
    module = "cli_probe_interrupt_fixture"
    (tmp_path / f"{module}.py").write_text("raise KeyboardInterrupt\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    ok, evidence = doctor._import_probe(module)
    assert not ok
    assert "probe" in evidence and "interrupt" in evidence.lower()
    assert "failed to import" not in evidence
    requirement = doctor._Requirement("numpy", "numpy", False)
    result = doctor._base_dependency_check((requirement,), {"numpy": (ok, evidence)})
    assert result.status == "untested" and not result.blocking


def test_D12_restart_mismatch_is_a_refusal_exit():
    from gpuwm.io.restart import RestartMismatchError
    assert getattr(RestartMismatchError("incompatible scheme"), "exit_code", None) == 2


def test_D12_stale_physics_proof_names_prepare_again(tmp_path):
    from test_prepared_single_domain_forecast import _prepared_fixture
    from gpuwm import prepared_single_domain_forecast as runner
    from gpuwm.experiment import load_experiment

    fixture = _prepared_fixture(tmp_path, "gfs")
    proof = json.loads(fixture.proof.read_text())
    for selection in (proof["physics"], proof["export"]["physics"]):
        parts = selection["registry_physics"]["parts"]
        key = next(key for key in parts if key.startswith("components.microphysics.options."))
        parts[key] = "0" * 64
    with pytest.raises(ValueError, match="Prepare again"):
        runner._validate_front_door_physics_proof(proof, source="gfs",
            profile=runner.PHYSICS_PROFILE, cfg=load_experiment(fixture.experiment).root.run)


@pytest.mark.parametrize("has_fetch", [True, False])
def test_D13_prepared_resume_names_the_correct_door(tmp_path, has_fetch):
    config = tmp_path / "experiment.toml"
    emitted = cli("domain", "--point=38.5,-98", "--point-extent-km", "90",
                  "--root-dx", "3", "--vram-gib", "96", "--source", "gfs",
                  "--cycle", "2026-10-08T18", "--out", config)
    assert emitted.returncode == 0, emitted.stderr
    if not has_fetch:
        config.write_text(config.read_text().split("[fetch]", 1)[0])
    answer = cli("resume", config, "--from", tmp_path / "checkpoint.npz", "--outdir", tmp_path / "out")
    assert answer.returncode == 2
    assert "go CONFIG --restart" in answer.stderr
    assert "--source era5" not in answer.stderr


def test_D19_prepared_readiness_validates_the_supplied_tree(tmp_path):
    config = tmp_path / "experiment.toml"
    emitted = cli("domain", "--point=38.5,-98", "--point-extent-km", "90",
                  "--root-dx", "3", "--vram-gib", "96", "--source", "gfs",
                  "--cycle", "2026-10-08T18", "--out", config)
    assert emitted.returncode == 0, emitted.stderr
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "proof.json").write_text('{"schema":"unreadable-prepared-fixture"}')
    before = (prepared / "proof.json").read_bytes()
    answer = cli("go", config, "--prepared-root", prepared, "--readiness", "--no-probe")
    assert answer.returncode == 2, answer.stdout + answer.stderr
    document = json.loads(answer.stdout)
    assert document["state"] == "refused" and not document["ready"]
    assert "no runner" in document["refusal"]
    assert (prepared / "proof.json").read_bytes() == before
    assert not (tmp_path / "experiment-go").exists()
    from gpuwm.runplan import build_plan, plan_readiness, PLAN_SCHEMA

    plan = build_plan({"schema": PLAN_SCHEMA, "name": "prepared-readiness-proof",
        "route": "prepared", "config": {"path": str(config)},
        "output_root": str(tmp_path / "out"),
        "run_options": {"prepared_root": str(prepared)}}, source="test",
        base_dir=tmp_path, sha256="0" * 64)
    document, code = plan_readiness(plan, no_probe=True)
    assert code == 2 and "no runner" in document["refusal"]
