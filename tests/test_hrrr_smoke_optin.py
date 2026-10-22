"""2.9.4 ships HRRR-Smoke opt-in: the bare go door runs the HRRR v4 recipe
smoke off, and ``--smoke`` turns it on from gpuwm.smoke_execution's one
HRRR_SMOKE_SETTINGS set, in a per-run config (the recipe is never edited).

Why: smoke on costs ~11-12x per step on full CONUS (8x RTX 5090, fold
f1a3e8609b: ordinary steps 3.50 s vs 0.295 s, radiation steps 5.0-5.4 s vs
1.85 s).  Breakage this prevents: a default HRRR-recipe run paying that
cost, or an opt-in that does not reach the launch.
"""
from pathlib import Path

import pytest

from gpuwm.experiment import load_experiment
from gpuwm.smoke_execution import HRRR_SMOKE_SETTINGS
from gpuwm.smoke_receipts import DEFAULT_OFF_LINE, receipt

RECIPE = Path(__file__).resolve().parents[1] / "gpuwm/data/recipes/hrrr_v4_gsd41.toml"


@pytest.mark.parametrize("flags,chem_sets", [([], ""), (["--smoke"], "smoke")])
def test_go_plan_resolves_smoke_only_with_the_flag(tmp_path, monkeypatch, flags, chem_sets):
    from gpuwm import go_cli
    from gpuwm.cli import main
    from gpuwm.core import preflight
    captured = {}

    def launch(args, *, config, payload, **kw):
        captured.update(path=config, config=load_experiment(config))
        return 0

    # No GPU: an unknown card never earns a capacity decision.
    monkeypatch.setattr(preflight, "device_memory_probe_subprocess", lambda: None)
    monkeypatch.setattr(go_cli, "_registered_launch", launch)
    original = RECIPE.read_bytes()
    assert main(["go", str(RECIPE), *flags, "--outdir", str(tmp_path / "run"),
                 "--wps-namelist", str(tmp_path / "input.namelist.wps")]) == 0
    run = captured["config"].root.run
    assert run.chem_sets == chem_sets
    # Hourly history either way (the 15-min subset stream is next release).
    assert captured["config"].root.history_interval_s == 3600.0
    assert (run.history_interval2_s, run.history_interval_change_s) == (0.0, 0.0)
    assert RECIPE.read_bytes() == original
    if flags:
        assert captured["path"] != RECIPE
        for key, value in HRRR_SMOKE_SETTINGS.items():
            assert getattr(run, key) == value, key
        assert receipt(run) == {}
    else:
        assert captured["path"] == RECIPE
        assert receipt(run) == {"smoke_off": DEFAULT_OFF_LINE}


def test_domain_recipe_and_race_carry_the_flag(tmp_path):
    import tomllib
    from types import SimpleNamespace
    from gpuwm.region_recipe import emit_recipe
    from gpuwm import race
    common = dict(recipe="hrrr_v4_gsd41", region="US", spacing_km=3.0, source="hrrr",
                  cycle="2026-10-11T12", hours=1)
    off, off_receipt = emit_recipe(**common)
    on, on_receipt = emit_recipe(**common, smoke=True)
    assert tomllib.loads(off)["shared"]["chem_sets"] == ""
    assert off_receipt["hrrr_smoke"] == DEFAULT_OFF_LINE
    assert tomllib.loads(on)["shared"]["chem_sets"] == "smoke"
    assert "smoke_off_reason" not in tomllib.loads(on)["shared"]
    args = SimpleNamespace(hours=1, tiles="off", products="all", hydro="auto", devices=None,
                           initial_inputs=None, data_dir=None, publish_root=None, smoke=True)
    command, _standby, _publish = race.command_for(
        args, {"id": "hrrr_v4_gsd41", "config": "hrrr_v4_gsd41.toml"},
        __import__("datetime").datetime(2026, 10, 2, 21), tmp_path)
    assert "--smoke" in command
