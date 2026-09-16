"""Deferred acquisition binds through the source registry, not a model name.

CPU-only: nothing is fetched, no credential value is read and no run starts.
"""
import hashlib
import json
import tomllib
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm import remote_plan as rp


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture
def saved(tmp_path):
    from gpuwm import domain_wizard as dw
    config = tmp_path / "saved map.toml"
    config.write_text(dw.render_config(name="remote-map", start_time=datetime(2026, 9, 7, 18),
        hours=24, projection=dw._projection_entries(18, -162.1, "auto"),
        dims=dw._dims_for_scale(1, ()), ratios=(),
        fetch_hints=dict(source="gfs", cycle="2026-09-07T12", forecast_start_hour=6, hours=24,
                         out=str(tmp_path / "forcing"), cadence=3), case_data=None), encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema": "gpuwm.run-plan.v1", "name": "reviewed-map",
        "route": "prepared", "config": {"path": str(config)},
        "output_root": str(tmp_path / "local-output"), "run_options": {"render_products": "none"}}),
        encoding="utf-8")
    return config, plan


def build(saved):
    config, plan = saved
    return rp.build_bundle(plan, workspace="/node/work", outdir="/node/work/new-output",
        geog_root="/node/geography", expected_plan_sha256=sha(plan.read_bytes()),
        expected_config_sha256=sha(config.read_bytes()))


def contents(document, name):
    import base64
    return base64.b64decode(next(f["data"] for f in document["files"] if f["name"] == name))


def _case_data(tmp_path, forcing):
    wps = tmp_path / "saved map.namelist.wps"
    wps.write_text("&share\n max_dom=1,\n/\n&geogrid\n dx=12000.,\n dy=12000.,\n/\n")
    vtable = tmp_path / "selected.Vtable"
    vtable.write_text("selected scientific Vtable")
    return {"forcing": [forcing], "vtable": str(vtable), "wps_namelist": str(wps),
            "geog_root": str(tmp_path / "local-geog"), "forcing_interval_s": 10800,
            "sfcp_to_sfcp": True, "output_domain": 1, "output_title": "saved map"}


def test_managed_forcing_binding_follows_the_registry_runner(saved, tmp_path, monkeypatch):
    """Any row whose declared runner is the case-data preparation binds by name."""
    import dataclasses
    from gpuwm import case_data, source_adapters
    from gpuwm.fetch import ERA5_COMBINED_NAMES
    from gpuwm.toml_document import emit_experiment_toml
    config, _plan = saved
    real = source_adapters.get_source_adapter
    # A registered row that is NOT the model this door used to test for, given
    # the case-data runner. Nothing else about it changes.
    monkeypatch.setattr(source_adapters, "get_source_adapter",
        lambda source: (dataclasses.replace(real(source), runner=case_data.CASE_DATA_RUNNER)
                        if source == "gdas" else real(source)))
    raw = tomllib.loads(config.read_text())
    out = tmp_path / "forcing"
    out.mkdir(exist_ok=True)
    raw["fetch"]["source"] = "gdas"
    raw["fetch"].pop("forecast_start_hour", None)
    raw["fetch"]["hours"] = 9
    raw["case_data"] = _case_data(tmp_path, str(out / ERA5_COMBINED_NAMES["cds"]))
    config.write_text(emit_experiment_toml(raw))
    document = build(saved)
    staged = tomllib.loads(contents(document, "case.toml").decode())
    assert staged["case_data"]["forcing"] == [staged["fetch"]["out"] + "/" + ERA5_COMBINED_NAMES["cds"]]
    assert len(document["expected_downloads"]) == 1
    entry = document["expected_downloads"][0]
    assert entry["role"] == "forcing" and entry["source"] == "gdas"
    assert entry["path"] == staged["case_data"]["forcing"][0]


def test_an_unreproducible_forcing_refusal_names_the_source_not_a_model(saved, tmp_path):
    from gpuwm.toml_document import emit_experiment_toml
    config, _plan = saved
    raw = tomllib.loads(config.read_text())
    raw["case_data"] = _case_data(tmp_path, str(tmp_path / "specifically-selected.grib"))
    config.write_text(emit_experiment_toml(raw))
    with pytest.raises(ValueError) as failure:
        build(saved)
    message = str(failure.value)
    assert "specifically-selected.grib" in message
    assert "'gfs'" in message
    assert "acquisition recipe does not produce it on the node" in message
    assert "Correct the forcing/acquisition binding" in message
    assert "transfer that input to the node" in message
    assert "ERA5" not in message


def test_chain_route_reports_its_node_acquisition_at_review(saved):
    document = build(saved)
    plan = json.loads(contents(document, "plan.json"))
    assert document["expected_downloads"] == [
        {"role": "forcing", "path": plan["run_options"]["data_dir"], "source": "gfs",
         "recipe": tomllib.loads(contents(document, "case.toml").decode())["fetch"]}]


def test_node_credential_gate_is_registry_driven(saved, tmp_path, monkeypatch):
    """A declared credential this box reports absent refuses at review, by name."""
    from gpuwm import source_adapters, source_credentials
    from gpuwm.source_credentials import CredentialLocation, SourceCredential
    credential = SourceCredential(
        credential_id="fixture-key", display_name="Registered family key",
        location_kind=CredentialLocation.ENV_VAR, location="ARWEN_FIXTURE_ABSENT_KEY",
        needed_for="acquisition", breakage="the declared forcing cannot be downloaded",
        obtain_url="https://example.invalid/key")
    monkeypatch.delenv("ARWEN_FIXTURE_ABSENT_KEY", raising=False)
    monkeypatch.setattr(source_adapters, "get_source_adapter",
        lambda source: SimpleNamespace(source_id="registered-family", runner=None,
                                       credentials=(credential,)))
    with pytest.raises(ValueError) as failure:
        rp._node_acquisition_readiness([{"role": "forcing", "path": "/node/cache",
                                         "source": "registered-family", "recipe": {}}])
    message = str(failure.value)
    assert "registered-family" in message
    assert source_credentials.credential_short_note(credential) in message


def test_a_credential_this_box_cannot_answer_for_is_never_a_refusal(monkeypatch):
    from gpuwm import source_adapters, source_credentials
    from gpuwm.source_credentials import CredentialLocation, SourceCredential
    credential = SourceCredential(
        credential_id="fixture-key", display_name="Unknowable key",
        location_kind=CredentialLocation.ENV_VAR, location="ARWEN_FIXTURE_UNKNOWN_KEY",
        needed_for="acquisition", breakage="the declared forcing cannot be downloaded")
    monkeypatch.setattr(source_credentials, "credential_present", lambda _c: None)
    monkeypatch.setattr(source_adapters, "get_source_adapter",
        lambda source: SimpleNamespace(source_id="registered-family", runner=None,
                                       credentials=(credential,)))
    assert rp._node_acquisition_readiness([{"role": "forcing", "path": "/node/cache",
                                            "source": "registered-family", "recipe": {}}]) is None


def test_an_unregistered_source_row_does_not_crash_the_readiness_gate():
    assert rp._node_acquisition_readiness([{"role": "forcing", "path": "/node/cache",
                                            "source": "not-a-registered-source", "recipe": {}}]) is None
