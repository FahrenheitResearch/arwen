"""Prepared check prices its verified static and retained boundary tables."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import pytest

from gpuwm.config import RunConfig
from gpuwm.core import preflight as pf
from gpuwm.experiment import experiment_from_run_config


def _args(tmp_path, monkeypatch, *flags):
    config = tmp_path / "experiment.toml"
    # Deliberately differs from the prepared cache's retained 3-hour series.
    config.write_text('[fetch]\nsource = "gfs"\ncadence = 6\nhours = 24\n')
    root = tmp_path / "prepared"
    root.mkdir()
    cfg = RunConfig(nx=128, ny=128, nz=59, dx=3000., dy=3000., dt=10.,
                    ztop=20000.,
                    run_seconds=3600., terrain_opt=1, moist=True, mp_physics=8,
                    sf_urban_physics=3, sf_surface_physics=4,
                    num_soil_layers=4, sf_sfclay_physics=1,
                    bl_pbl_physics=1)
    exp = experiment_from_run_config(cfg, datetime(2026, 1, 1))
    parser = argparse.ArgumentParser()
    pf.register_cli(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(["check", str(config), "--prepared-root", str(root),
                              "--json", *flags])
    args._prepared_check_inputs = pf.PreparedCheckInputs(
        experiment=exp, prepared_root=root, source="gfs",
        boundary_source=("qc", "qr"),
        forcing_interval_seconds=10800., forcing_intervals=2,
        urban_columns={1: 37})
    monkeypatch.setattr("gpuwm.data_assets.companion_root", lambda: tmp_path)
    monkeypatch.setattr(pf, "_warn_unstaged_physics_tables", lambda *_: None)
    monkeypatch.setattr(pf, "declares_the_local_card", lambda *_: False)
    return args


def test_prepared_check_prices_every_reported_forecast_number(tmp_path, monkeypatch, capsys):
    args = _args(tmp_path, monkeypatch, "--free-gib", "47", "--vram-gib", "48")
    assert pf.check_main(args) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["prepared_urban_columns"] == {"1": 37}
    assert document["forcing_interval_seconds"] == 10800
    assert document["retained_forcing_intervals"] == 2
    assert document["observed_peak_envelope_bytes"] == document["peak_envelope_bytes"]
    assert document["ingest"] is None
    assert "already completed" in document["ingest_not_priced_reason"]
    assert "verified prepared forecast" in document["phase_verdict"]
    assert "d01 37" in document["bem_column_workspace_basis"]
    assert "land cover does not exist" not in document["bem_column_workspace_basis"]
    bound = pf.estimate_experiment(
        args._prepared_check_inputs.experiment, forcing_interval_seconds=10800.,
        forcing_intervals=2, vram_gib=48,
        profile=pf.card_local_memory_profile(48),
        boundary_species=("qc", "qr"))
    assert document["alloc_estimate_bytes"] < bound.alloc_estimate_bytes
    assert document["peak_envelope_bytes"] < bound.peak_envelope_bytes


def test_prepared_check_rejects_an_override_of_its_retained_cadence(tmp_path, monkeypatch):
    args = _args(tmp_path, monkeypatch, "--free-gib", "47", "--vram-gib", "48",
                 "--forcing-interval-s", "21600")
    with pytest.raises(ValueError, match="supplied forcing cadence 10800"):
        pf.check_main(args)


def test_prepared_split_keeps_bound_forcing_and_skips_completed_ingest(
        tmp_path, monkeypatch, capsys):
    from gpuwm.core.devices_memory import GIB
    args = _args(tmp_path, monkeypatch, "--free-gib", "95", "--vram-gib", "96",
                 "--devices", "2")
    monkeypatch.setattr(pf, "host_available_bytes", lambda: 512 * GIB)
    assert pf.check_main(args) == 0
    document = json.loads(capsys.readouterr().out)
    assert len(document["devices"]["cards"]) == 2
    assert document["forcing_interval_seconds"] == 10800
    assert document["retained_forcing_intervals"] == 2
    assert document["boundary_species"] == ["qc", "qr"]
    assert document["preparation"]["already_completed"]
    assert not document["preparation"]["priced"]
    assert document["prepared_urban_columns"] == {"1": 37}
    assert "every-column" in document["bem_column_workspace_basis"]
    assert document["output_disk"]["total_bytes"] > 0


def test_readiness_fallback_keeps_the_prepared_price(tmp_path, monkeypatch):
    args = _args(tmp_path, monkeypatch, "--free-gib", "47", "--vram-gib", "48")
    monkeypatch.setattr(pf, "device_memory_probe_subprocess", lambda: None)
    required = pf._required_memory_without_kernels(
        args._prepared_check_inputs.experiment, args)
    assert required["prepared_urban_columns"] == {1: 37}
    assert "d01 37" in required["bem_column_workspace_basis"]
    assert required["ingest"] is None
    bound = pf.estimate_experiment(
        args._prepared_check_inputs.experiment, forcing_interval_seconds=10800.,
        forcing_intervals=2, vram_gib=48,
        profile=pf.card_local_memory_profile(48),
        boundary_species=("qc", "qr"))
    assert required["alloc_estimate_bytes"] < bound.alloc_estimate_bytes


def test_prepared_alloc_keeps_the_synthetic_all_urban_bound(tmp_path, monkeypatch, capsys):
    from gpuwm.doctor import Check

    args = _args(tmp_path, monkeypatch, "--alloc")
    monkeypatch.setattr("gpuwm.doctor._cuda_headers_check", lambda: Check(
        "CUDA kernel headers", "verified", "fixture kernels ready"))
    monkeypatch.setattr(pf, "live_device_local_memory_profile", lambda: None)

    def before_allocation(exp, **kwargs):
        assert "urban_columns" not in kwargs
        raise pf.PreflightHeadroomError(
            "synthetic probe stopped before allocation", phase="fixture",
            free_bytes=48 * pf.GIB, total_bytes=48 * pf.GIB,
            reserve_bytes=0, remaining_bytes=48 * pf.GIB)

    monkeypatch.setattr(pf, "run_alloc_preflight", before_allocation)
    assert pf.check_main(args) == 3
    document = json.loads(capsys.readouterr().out)
    assert document["prepared_urban_columns"] is None
    assert "synthetic all-urban" in document["bem_column_workspace_basis"]
    bound = pf.estimate_experiment(
        args._prepared_check_inputs.experiment, forcing_interval_seconds=10800.,
        forcing_intervals=2, profile=pf.card_local_memory_profile(None),
        boundary_species=("qc", "qr"))
    assert document["alloc_estimate_bytes"] == bound.alloc_estimate_bytes


def test_prepared_check_rejects_a_config_different_from_the_sealed_proof(tmp_path):
    from test_prepared_single_domain_forecast import _prepared_fixture

    fixture = _prepared_fixture(tmp_path, "gfs", physics_profile=None)
    fixture.experiment.write_bytes(fixture.experiment.read_bytes() + b"\n# changed\n")
    args = argparse.Namespace(config=fixture.experiment,
                              prepared_root=fixture.prepared,
                              wps_namelist=fixture.wps)
    with pytest.raises(ValueError, match="(?i)(sha|digest|differ|mismatch)"):
        pf._prepared_check_inputs(args)


@pytest.mark.parametrize("live_head", [False, True])
def test_prepared_check_extracts_a_verified_real_bundle(tmp_path, monkeypatch, live_head):
    from test_prepared_single_domain_forecast import (
        _bind_synthetic_preflight_geometry, _prepared_fixture)

    fixture = _prepared_fixture(tmp_path, "gfs", physics_profile=None)
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)
    digest = None
    if live_head:
        from test_prepared_head_binding import _unseal

        digest = _unseal(fixture).head["head_sha256"]
    args = argparse.Namespace(config=fixture.experiment,
                              prepared_root=fixture.prepared,
                              wps_namelist=fixture.wps)
    inputs = pf._prepared_check_inputs(args)
    assert inputs.source == "gfs"
    assert inputs.forcing_interval_seconds == 10800
    assert inputs.forcing_intervals == 1
    assert inputs.urban_columns is None
    assert inputs.prepared_head_sha256 == digest
    assert inputs.experiment.root.run.nx > 0
    assert pf._prepared_check_inputs(args) is inputs


def test_input_check_uses_the_prepared_authority_instead_of_decoding_again(tmp_path, monkeypatch, capsys):
    from gpuwm.ingest.preflight import _check_command

    args = _args(tmp_path, monkeypatch, "--free-gib", "47", "--vram-gib", "48")
    assert _check_command(args) == 0
    output = capsys.readouterr()
    assert output.out == ""
    assert "prepared forecast inputs verified" in output.err
