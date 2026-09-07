"""Plain default Check output and detailed output describe the same admission."""
import argparse
import json

import pytest

from gpuwm import cli, domain_wizard
from gpuwm.core import preflight as pf
from test_check_host_memory import _fetch_only_config, _era5_case_config, _stub_the_decoded_inputs
from test_check_nested_mixed_road import _nested_auto_tiles


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    monkeypatch.setattr(pf, "device_physical_total_bytes", lambda **_: None)
    monkeypatch.setattr(pf, "host_available_bytes", lambda: 128 * pf.GIB)
    return _fetch_only_config(tmp_path)


def _run(capsys, config, *flags):
    code = cli.main(["check", str(config), *flags])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.mark.parametrize("budget", ["100", "0.1"])
def test_default_is_concise_and_explain_preserves_verdict_and_full_breakdown(config, capsys, budget):
    flags = ["--budget-gib", budget, "--vram-gib", "32"]
    code, concise, _ = _run(capsys, config, *flags)
    detailed_code, detailed, _ = _run(capsys, config, *flags, "--explain")
    assert code == detailed_code
    assert len(concise) < len(detailed) * 0.6
    assert len(concise.splitlines()) <= 20
    assert "TIER 1" not in concise and "TIER 1" in detailed
    assert "Use --explain" in concise
    assert "Physical GPU capacity: n/a (not measured)" in concise
    assert "Target GPU capacity: 32 GiB (declared)" in concise
    assert f"Configured allocation budget: {float(budget):g} GiB requested" in concise
    assert "inferred from --budget-gib; not measured" in concise
    assert "Effective allocation budget:" in concise and "whole-process budget:" in concise
    assert "Forecast execution: resident ([tiles] mode=off)" in concise
    assert "Streaming host memory:" not in concise
    assert "Host RAM:" in concise
    if code:
        assert "WARNING: observed peak envelope" in concise
        assert "alloc_estimate_le_" in concise and ": FAIL" in concise
    json_code, raw, _ = _run(capsys, config, *flags, "--json")
    detailed_json_code, detailed_raw, _ = _run(capsys, config, *flags, "--json", "--explain")
    assert json_code == detailed_json_code == code
    one, two = json.loads(raw), json.loads(detailed_raw)
    for key in ("gates", "budget_bytes", "envelope_budget_bytes", "memory_verdict", "observed_peak_envelope_exceeds_budget"):
        assert one[key] == two[key]
    assert one["physical_total_bytes"] is None
    assert one["declared_capacity_bytes"] == 32 * pf.GIB


def test_shared_measured_snapshot_keeps_physical_total_and_free_distinct(config, capsys):
    parser = argparse.ArgumentParser()
    pf.register_cli(parser.add_subparsers(required=True))
    args = parser.parse_args(["check", str(config), "--free-gib", "6", "--vram-gib", "10"])
    args._shared_sizing_budget = domain_wizard.SizingBudget(
        10, 6 * pf.GIB, None, "same measured sizing fixture", measured=True)
    pf.check_main(args)
    output = capsys.readouterr().out
    assert "Physical GPU capacity: 10.00 GiB (measured)" in output
    assert "Free VRAM used for sizing: 6.00 GiB (measured; shared sizing sample)" in output
    assert "shared measured sizing sample" in output
    assert "Configured allocation budget:" not in output
    assert "inferred" not in output


def test_default_names_mixed_domains_and_streaming_host_claim(config, tmp_path, capsys):
    tree = _nested_auto_tiles(tmp_path)
    code, output, _ = _run(capsys, tree, "--budget-gib", "13")
    assert code == 0
    assert "Forecast execution: mixed resident/streamed" in output
    assert "d01 resident:" in output and "d02 streams" in output
    assert "Streaming host memory:" in output


def test_default_host_refusal_is_short_and_keeps_exit_five(config, tmp_path, monkeypatch, capsys):
    _stub_the_decoded_inputs(monkeypatch)
    monkeypatch.setattr(pf, "host_available_bytes", lambda: 16 * pf.GIB)
    case = _era5_case_config(tmp_path)
    code, output, error = _run(capsys, case, "--budget-gib", "100")
    assert code == 5
    assert "Host forcing decode:" in output and "16.00 GiB available" in output
    assert "REFUSED (exit 5)" in error and "GPU tiling does not reduce source decode RAM" in error
    assert len(error) < 450
    detailed_code, _, detailed_error = _run(capsys, case, "--budget-gib", "100", "--explain")
    assert detailed_code == code and len(detailed_error) > len(error)
