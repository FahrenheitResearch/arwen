"""`gpuwm go --prepare-only` prepares and stops, on a host with no card.

Two breakages, both measured on CPU boxes preparing an HRRR recipe:

* the door probed the card and asked for CuPy before a run that never
  reaches the forecast ("GPU readiness is info: no device", "this command
  needs cupy"), so no CPU host could prepare;
* on the registered (HRRR and staged) routes --prepare-only was not
  handed to the chain at all, so the run went on into the forecast stage
  after its preparation and died there (cudaErrorNoDevice) or held its
  head for later leads the forecast needed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import gpuwm.capabilities as capabilities
import gpuwm.go_cli as go_cli
from gpuwm import runplan
from gpuwm.cli import main

ROOT = Path(__file__).resolve().parents[1]


def _hrrr_config(tmp_path: Path) -> Path:
    text = (ROOT / "configs/recipes/hrrr_v4_gsd41.toml").read_text(
        encoding="utf-8")
    head, _, _ = text.partition("[fetch]")
    path = tmp_path / "hrrr-prepare.toml"
    path.write_text(head + '[fetch]\nsource = "hrrr"\n'
                    'cycle = "2026-10-02T21"\nhours = 2\ncadence = 1\n',
                    encoding="utf-8")
    return path


def test_prepare_only_runs_the_chain_with_prepare_only_and_no_card(
        tmp_path, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("prepare-only asked for the forecast's card")

    monkeypatch.setattr(go_cli, "_require_forecast_device", refuse)
    monkeypatch.setattr(go_cli, "memory_gate", refuse)
    monkeypatch.setattr(capabilities, "require_for_command", refuse)
    monkeypatch.setattr(go_cli, "_registered_launch", refuse)
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    calls = []

    def chain(plan, *, config_path, exp, observer, run_dir, prepare_only):
        calls.append(prepare_only)
        return {"prepared_root": str(run_dir / "prepared"),
                "forecast_started": False}

    monkeypatch.setattr(runplan, "_hrrr_chain", chain)
    config = _hrrr_config(tmp_path)
    rc = main(["go", str(config), "--prepare-only",
               "--outdir", str(tmp_path / "out")])
    assert rc == 0
    assert calls == [True]


def test_a_forecast_run_still_asks_for_the_card(tmp_path, monkeypatch):
    """Only --prepare-only is exempt: a run that forecasts is refused at
    the door on a host whose GPU runtime is missing, as before."""

    class Refused(Exception):
        pass

    def refuse(*_args, **_kwargs):
        raise Refused

    monkeypatch.setattr(capabilities, "require_for_command", refuse)
    config = _hrrr_config(tmp_path)
    with pytest.raises(Refused):
        main(["go", str(config), "--outdir", str(tmp_path / "out")])


def test_a_failed_preparation_returns_its_code_without_a_traceback(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(go_cli, "_require_forecast_device", lambda: None)

    def chain(*_args, **_kwargs):
        raise go_cli.GoStageFailed(3)

    monkeypatch.setattr(runplan, "_hrrr_chain", chain)
    config = _hrrr_config(tmp_path)
    rc = main(["go", str(config), "--prepare-only",
               "--outdir", str(tmp_path / "out")])
    assert rc == 3
    assert "--prepare-only stopped: stage exited 3" in capsys.readouterr().err


def test_an_experiment_document_carries_the_card_placement():
    """The HRRR root publishes its authority through this renderer; a
    [devices] table refused it after the fetch on every multi-card run."""

    import tomllib

    from gpuwm.experiment_document import render_experiment_document

    raw = tomllib.loads((ROOT / "configs/recipes/hrrr_v4_gsd41.toml").read_text(
        encoding="utf-8"))
    raw.pop("fetch", None)
    raw["devices"] = {"count": 2, "grid": "1x2"}
    text = render_experiment_document(raw)
    assert tomllib.loads(text)["devices"] == {"count": 2, "grid": "1x2"}
