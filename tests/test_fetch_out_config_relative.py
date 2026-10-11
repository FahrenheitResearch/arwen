"""D-11 (2.8.8 acceptance): an ERA5 config works from any working directory.

Row SW-era5 wrote ``gpuwm domain ... --source era5 --out sub/experiment.toml``.
``gpuwm go sub/experiment.toml --dry-run`` ran, but ``cd sub && gpuwm go
experiment.toml --dry-run`` refused with "[fetch].out does not produce the
file named by [case_data].forcing": ``[fetch].out`` was written and read
relative to the working directory, ``[case_data].forcing`` relative to the
config.  Both are relative to the config now; a config written before the
fix (an out relative to where the wizard ran) still runs from that folder.
"""
from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

from gpuwm.cli import main as cli_main


def _emit_into_subfolder(tmp_path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    rc = cli_main(["domain", "--point=41.97,-93.6", "--card", "16gb",
                   "--ladder", "12", "--source", "era5", "--cycle", "2024-05-21T18",
                   "--out", "sub/experiment.toml"])
    assert rc == 0
    return tmp_path / "sub" / "experiment.toml"


def _launch_gate(config: Path):
    """What `gpuwm go` runs before acquiring anything (go_cli)."""
    from gpuwm import runplan
    from gpuwm.case_data import build_case_data

    raw = tomllib.loads(config.read_text(encoding="utf-8"))
    data = build_case_data(raw["case_data"], source=str(config), base_dir=config.parent,
                           require_inputs=False, require_met_inputs=False)
    return runplan.declared_forcing_fetch(raw, data, base_dir=config.resolve().parent), data


@pytest.mark.parametrize("launch_from", ["writer", "config", "elsewhere"])
def test_an_era5_config_launches_from_any_folder(tmp_path, monkeypatch, launch_from):
    config = _emit_into_subfolder(tmp_path, monkeypatch)
    raw = tomllib.loads(config.read_text(encoding="utf-8"))
    # Both keys are written relative to the config itself.
    assert (config.parent / raw["fetch"]["out"]).resolve() == (
        config.parent / raw["case_data"]["forcing"][0]).resolve().parent
    where = {"writer": tmp_path, "config": config.parent,
             "elsewhere": tmp_path / "other"}[launch_from]
    where.mkdir(exist_ok=True)
    monkeypatch.chdir(where)
    arguments, data = _launch_gate(Path("..", "sub", "experiment.toml") if launch_from == "elsewhere"
                                   else Path("experiment.toml") if launch_from == "config"
                                   else Path("sub", "experiment.toml"))
    assert arguments is not None
    out = Path(arguments[arguments.index("--out") + 1])
    assert out.resolve() == data.forcing[0].resolve().parent
    assert out.resolve().is_relative_to((tmp_path / "sub").resolve())


@pytest.mark.parametrize("launch_from", ["writer", "config", "elsewhere"])
def test_a_config_written_before_the_fix_runs_from_any_folder(tmp_path, monkeypatch, launch_from):
    config = _emit_into_subfolder(tmp_path, monkeypatch)
    raw_text = config.read_text(encoding="utf-8")
    raw = tomllib.loads(raw_text)
    # The 2.8.7 emission: out relative to the folder the wizard ran in.
    legacy_out = "sub/" + raw["fetch"]["out"]
    config.write_text(raw_text.replace(f'out = "{raw["fetch"]["out"]}"', f'out = "{legacy_out}"'),
                      encoding="utf-8")
    where = {"writer": tmp_path, "config": config.parent,
             "elsewhere": tmp_path / "other"}[launch_from]
    where.mkdir(exist_ok=True)
    monkeypatch.chdir(where)
    arguments, data = _launch_gate(config)
    out = Path(arguments[arguments.index("--out") + 1])
    assert out.resolve() == data.forcing[0].resolve().parent


def test_conflicting_fetch_out_is_not_replaced_by_the_forcing_directory(tmp_path):
    from gpuwm.case_data import resolve_fetch_out

    base = tmp_path / "sub"
    forcing = (base / "data" / "era5-combined.grib",)
    assert resolve_fetch_out("different/data", base_dir=base, forcing=forcing) == (
        base / "different/data").resolve()


def test_the_run_plan_anchors_a_named_config_at_its_own_folder(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from gpuwm.runplan import _config_for_declared_fetch

    config = _emit_into_subfolder(tmp_path, monkeypatch)
    monkeypatch.chdir(config.parent)
    plan = SimpleNamespace(config_intent=None, config_path=config, config_base_dir=config.parent,
                           config_bytes=lambda: config.read_bytes())
    document = _config_for_declared_fetch(plan, {}, tmp_path / "run")
    raw = tomllib.loads(config.read_text(encoding="utf-8"))
    assert Path(document["fetch"]["out"]) == (
        config.parent / raw["case_data"]["forcing"][0]).resolve().parent
