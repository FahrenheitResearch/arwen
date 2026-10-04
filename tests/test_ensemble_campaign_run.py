from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from fractions import Fraction
import importlib.util
from pathlib import Path

import pytest


_PATH = Path(__file__).resolve().parents[1] / "tools/ensemble_campaign_run.py"
_SPEC = importlib.util.spec_from_file_location("ensemble_campaign_run_tested", _PATH)
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


@dataclass(frozen=True)
class Run:
    run_seconds: float = 43200
    dt: float = 15
    dx: float = 3000
    nz: int = 49
    mp_physics: int = 8


@dataclass(frozen=True)
class Domain:
    run: Run = Run()
    grid_id: int = 1
    history_interval_s: float = 3600


@dataclass(frozen=True)
class Experiment:
    run_seconds: float = 43200
    domains: tuple = (Domain(),)
    restart_interval_s: float = 3600

    @property
    def root(self):
        return self.domains[0]

    def dt_exact(self, grid_id):
        assert grid_id == 1
        return Fraction(15)


@dataclass(frozen=True)
class Inputs:
    experiment: Experiment = Experiment()
    cache_identity: str = "original twelve hour cache"


def test_identity_shortens_only_stop_time_after_full_preparation():
    original = Inputs()
    execution = {"purpose": "identity", "source_window_seconds": 43200,
                 "execution_run_seconds": 3600}
    bounded, receipt = runner.bounded_identity_inputs(original, execution)
    assert original.experiment.run_seconds == 43200
    assert original.experiment.root.run.run_seconds == 43200
    assert bounded.experiment.run_seconds == bounded.experiment.root.run.run_seconds == 3600
    assert bounded.experiment.root.run == Run(run_seconds=3600)
    assert bounded.experiment.restart_interval_s == bounded.experiment.root.history_interval_s == 3600
    assert bounded.cache_identity == original.cache_identity
    assert receipt["changed_fields"] == ["run_seconds", "domains[0].run.run_seconds"]
    assert receipt["execution_model_steps"] == 240
    assert receipt["source_experiment_sha256"] != receipt["execution_experiment_sha256"]


def test_calibration_retains_the_original_admitted_inputs():
    original = Inputs()
    result, receipt = runner.bounded_identity_inputs(original, {
        "purpose": "calibration", "source_window_seconds": 43200, "execution_run_seconds": 43200})
    assert result is original
    assert receipt is None


def test_identity_cannot_shorten_an_unverified_window_or_change_step_lattice():
    with pytest.raises(ValueError, match="complete source window"):
        runner.bounded_identity_inputs(Inputs(), {"purpose": "identity", "source_window_seconds": 7200,
                                                  "execution_run_seconds": 3600})
    with pytest.raises(ValueError, match="step lattice"):
        runner.bounded_identity_inputs(Inputs(), {"purpose": "identity", "source_window_seconds": 43200,
                                                  "execution_run_seconds": 3601})


def test_short_identity_cannot_claim_full_calibration():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    recipe = type("Recipe", (), {"start": start, "end": start + timedelta(hours=12)})()
    with pytest.raises(ValueError, match="identity qualification"):
        runner.execution_window({"identity_run_seconds": 3600}, recipe)
    execution = runner.execution_window({"execution_purpose": "identity", "identity_run_seconds": 3600}, recipe)
    assert execution == {"purpose": "identity", "source_window_seconds": 43200,
                         "execution_run_seconds": 3600, "calibration_complete": False}
    with pytest.raises(ValueError, match="identity qualification"):
        runner.execution_window({"execution_purpose": "identity", "identity_run_seconds": True}, recipe)


def test_identity_inventory_ignores_ready_sidecars_and_joins_checkpoint_sets(tmp_path):
    specification = importlib.util.spec_from_file_location(
        "ensemble_identity_compare_tested", _PATH.with_name("ensemble_identity_compare.py"))
    comparator = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(comparator)
    frame = "wrfout_d01_2024-01-01_01_00_00"
    (tmp_path / frame).write_bytes(b"CDF\x02")
    (tmp_path / (frame + ".json")).write_text("{}")
    assert set(comparator.inventory(tmp_path, "wrfout_d*")) == {frame}
    checkpoint = "gpuwmrst_d01_2024-01-01_01_00_00"
    (tmp_path / (checkpoint + "__first.npz")).touch()
    assert set(comparator.inventory(tmp_path, "gpuwmrst_d*.npz")) == {checkpoint}
    (tmp_path / (checkpoint + "__second.npz")).touch()
    with pytest.raises(ValueError, match="duplicate"):
        comparator.inventory(tmp_path, "gpuwmrst_d*.npz")
