"""Measured memory must constrain an archived child on a busy large GPU."""
import json

import pytest

from gpuwm.cli import main
from gpuwm.downscale import _fit_child_size
from gpuwm.domain_wizard import SizingBudget
from test_downscale_cli import _PARENT_CONFIG, _point_args, _add_parent_surface


def test_free_memory_changes_the_actual_fit_without_changing_card_capacity():
    parent = {"nx": 501, "ny": 501, "dx": 1000.0, "dy": 1000.0}
    config = dict(_PARENT_CONFIG, nx=501, ny=501, nz=49)
    args = dict(j0=250, i0=250, ratio=2, run_seconds=3600.0,
                output_interval_s=3600.0, vram_gib=32.0)
    nominal = _fit_child_size(parent, config, **args)
    occupied = _fit_child_size(parent, config, **args, measured_free_bytes=8 * 1024**3)
    assert 8 <= occupied < nominal
    assert occupied % 4 == 0


@pytest.mark.parametrize("extra", [["--vram-gib", "32"], ["--card", "32gb"],
                                  ["--child-size", "48"]])
def test_ambiguous_automatic_sizing_refuses_before_reading_an_archive(tmp_path, capsys, extra):
    assert main(["downscale", str(tmp_path / "absent"), "--point", "35,-97",
                 "--out", str(tmp_path / "output"), "--auto-vram", *extra]) == 2
    assert "--auto-vram" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()


def test_auto_point_plan_uses_one_budget_and_reports_its_basis(tmp_path, capsys, monkeypatch):
    import gpuwm.downscale as downscale
    import gpuwm.domain_wizard as wizard
    args = _point_args(tmp_path)
    index = args.index("--child-size")
    del args[index:index + 2]
    for index in range(3):
        _add_parent_surface(tmp_path / f"wrfout_d01_1974-04-03_{12 + index:02d}_00_00", ny=18, nx=20)
    observed = []
    budget = SizingBudget(32.0, 8 * 1024**3, None, "Measured test device", True)
    def probe(card, capacity):
        observed.append((card, capacity))
        return budget
    fitted = []
    def fit(parent, config, **kwargs):
        fitted.append(kwargs)
        return 12
    monkeypatch.setattr(wizard, "resolve_sizing_budget", probe)
    monkeypatch.setattr(downscale, "_fit_child_size", fit)
    assert main([*args, "--auto-vram", "--dry-run"]) == 0
    output = capsys.readouterr().out
    plan = json.loads(output[output.index("{\n"):])
    assert observed == [(None, None)]
    assert fitted[0]["vram_gib"] == 32.0
    assert fitted[0]["measured_free_bytes"] == 8 * 1024**3
    assert plan["gpu_sizing"]["basis"] == "measured-local"
    assert plan["gpu_sizing"]["free_bytes"] == budget.free_bytes
