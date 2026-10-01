"""A hosted forecast preserves a clock refusal in its terminal event."""
from types import SimpleNamespace

import pytest

from conftest import requires_cupy
from gpuwm.core.adaptive_clock import NestDivideRefusal


@requires_cupy
def test_hosted_clock_refusal_keeps_its_type_message_and_exit_code(
        tmp_path, monkeypatch):
    from gpuwm import prepared_domain_tree_forecast as runner, runtime
    from gpuwm import provenance_gate
    from gpuwm.runplan import (
        EventStream, EVENTS_FILENAME, _staged_forecast, execute_plan,
        load_plan, read_events)
    from test_case_data import make_case_toml
    from test_runplan import _write_plan

    message = "nest divide for grid_id=2 needs 523 substeps inside one parent step"
    monkeypatch.setattr(provenance_gate, "announce_for_main", lambda *a: None)
    monkeypatch.setattr(runner, "preflight_prepared_tree", lambda **k:
                        SimpleNamespace(experiment=SimpleNamespace(domains=[])))
    monkeypatch.setattr(runner.prepared_single, "_route_owned_first_products",
                        lambda *a, **k: None)

    def refuse(*args, **kwargs):
        raise NestDivideRefusal(message)

    monkeypatch.setattr(runner, "run_prepared_tree", refuse)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    config = make_case_toml(tmp_path)

    def arguments(outdir):
        return ["--prepared-root", str(prepared),
                "--preparation-receipt-sha256", "a" * 64,
                "--experiment-config", str(config),
                "--experiment-config-sha256", "b" * 64,
                "--outdir", str(outdir)]

    def hosted(exp, data, outdir, *, progress_callback=None, **kwargs):
        progress_callback(model_elapsed_seconds=85619.76, outer_step=1000)
        _staged_forecast(arguments(tmp_path / "hosted-tree"), layout="tree",
                         observer=progress_callback)

    monkeypatch.setattr(runtime, "run_experiment", hosted)
    run_dir = tmp_path / "run"
    plan = load_plan(_write_plan(tmp_path, config, run_dir))
    with EventStream(run_dir / EVENTS_FILENAME, mirror=None) as events:
        code = execute_plan(plan, events=events)
    assert code != 0
    failed = read_events(run_dir / EVENTS_FILENAME)[-1]
    assert failed["event"] == "failed"
    assert failed["stage"] == "forecast"
    assert failed["error_class"] == "NestDivideRefusal"
    assert failed["message"] == message
    assert failed["exit_code"] == 2
    assert failed["interrupted"] is False
    assert (tmp_path / "hosted-tree" / "evidence" /
            "failed-run-receipt.json").is_file()
    # The standalone module retains its documented refusal return code.
    assert runner.main(arguments(tmp_path / "standalone-tree")) == 2
