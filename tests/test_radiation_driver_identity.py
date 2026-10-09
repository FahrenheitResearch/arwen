"""New off radiation options must retain historical public and restart bytes."""
from dataclasses import fields, make_dataclass, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path

import pytest

from gpuwm.config import CHEM_RUN_FIELDS, RunConfig
from gpuwm.core.model import restart_identity_payload
from gpuwm.experiment import (
    DomainConfig, ExperimentConfig, VerticalConfig,
    domain_config_document, experiment_config_document)
from gpuwm.ingest.prepared_cache import effective_prepared_domain_config
from gpuwm.io import restart


OPTIONS = {"swint_opt": 1, "aer_opt": 3, "alb_sol": 1}


def _experiment(**overrides):
    cfg = RunConfig(nx=6, ny=4, nz=5, dx=3000., dy=3000., ztop=20000.,
                    dt=20., run_seconds=0., mp_physics=28, moist=True,
                    ra_physics=4, ra_rrtmg_variant="rrtmg_legacy", **overrides)
    domain = DomainConfig(grid_id=1, parent_id=0, i_parent_start=1,
        j_parent_start=1, parent_grid_ratio=1, parent_time_step_ratio=1,
        history_interval_s=60., run=cfg, time_step=20)
    return ExperimentConfig(name="generic", start_time=datetime(2026, 1, 1),
        run_seconds=0., vertical=VerticalConfig(
            eta_levels=(1., .8, .6, .4, .2, 0.), p_top=5000., hybrid_opt=2, etac=.2),
        restart_interval_s=0., domains=(domain,))


def _historical_schema(exp):
    """The public off schema, before the three radiation controls existed."""
    domain = exp.domains[0]
    assert domain.run.chem_sets == ""
    neutral = {"ruc_irrigation": "wrf_461", "ruc_qvg_cold_start": "wrf",
               "ruc_2m_diagnostic": "flux", "ruc_snow": "wrf_461"}
    for name, default in neutral.items():
        assert getattr(domain.run, name) == default
    # A synthetic dataclass does not run RunConfig's public omission rules.
    omitted = set(OPTIONS) | set(CHEM_RUN_FIELDS) | set(neutral)
    old_fields = [field for field in fields(domain.run) if field.name not in omitted]
    old_type = make_dataclass("HistoricalRunConfig", [(field.name, field.type) for field in old_fields])
    old_run = old_type(**{field.name: getattr(domain.run, field.name) for field in old_fields})
    return replace(exp, domains=(replace(domain, run=old_run),))


def _bytes(document):
    return json.dumps(document, default=str, allow_nan=False).encode("utf-8")


def test_off_radiation_options_keep_complete_public_document_bytes():
    # Keep the clock selected by the recorded historical document explicit.
    # The later default local-face clock is a separate identity change.
    exp = _experiment(terrain_clock="measured")
    # These bytes came from the exact staging serializers with their real
    # 220-field RunConfig, before the three neutral radiation fields existed.
    # A synthetic dataclass skips earlier RunConfig-specific omission rules.
    control = json.loads((Path(__file__).parent / "fixtures/source_requests"
                          / "albsol-public-documents-d9.json").read_text())
    assert control["revision"] == "d9e34e20419142e9486c8db5c5b07de7132f4cee"
    assert len(control["runconfig_fields"]) == 220
    assert not set(OPTIONS) & set(control["runconfig_fields"])
    expected_hashes = {
        "domain": "88bee8d778fbd5b818f7c08828fbacb07440c8e0df131689df804b3660a67551",
        "experiment": "e66ba16c87169186cf224955d2b63e6a5e8fe764276fbd1b273561638270b041",
        "restart_identity": "4655c5a602c3551871dee41a18d79bcc4ac9fd7876f02c840258f16257a25e97",
    }
    documents = {"domain": domain_config_document(exp.root),
                 "experiment": experiment_config_document(exp),
                 "restart_identity": restart_identity_payload(exp)}
    assert exp.root.run.cycling is False
    assert "cycling" not in documents["domain"]["run"]
    cycling_exp = _experiment(terrain_clock="measured", cycling=True)
    assert domain_config_document(cycling_exp.root)["run"]["cycling"] is True
    assert experiment_config_document(cycling_exp)["domains"][0]["run"]["cycling"] is True
    assert restart_identity_payload(cycling_exp)["domains"][0]["run"]["cycling"] is True
    neutral_later_fields = {
        "thompson_version": "wrf_461",
        "thompson_fork_snow_fall": "blend",
        "rrtmg_cloud_optics_form": "wrf_461",
        "rrtmg_smoke_manifest": "",
    }
    for field, default in neutral_later_fields.items():
        assert getattr(exp.root.run, field) == default
    for name, document in documents.items():
        # Later neutral controls are omitted by the actual current serializers.
        # Their defaults and absence are explicit; every older byte stays pinned.
        runs = ([document["run"]] if name == "domain" else
                [domain["run"] for domain in document["domains"]])
        for run in runs:
            assert not set(neutral_later_fields) & set(run)
        old = control["documents"][name]
        payload = old["utf8"].encode("utf-8")
        assert len(payload) == old["bytes"]
        assert hashlib.sha256(payload).hexdigest() == old["sha256"] == expected_hashes[name]
        assert _bytes(document) == payload, name


@pytest.mark.parametrize("name", OPTIONS)
def test_serialized_zero_option_keeps_pre_field_restart_bytes(monkeypatch, name):
    from gpuwm import experiment
    old = {"domains": [{"run": {}}]}
    monkeypatch.setattr(experiment, "experiment_config_document", lambda exp: old)
    expected = _bytes(restart_identity_payload(object()))
    document = {"domains": [{"run": {name: 0}}]}
    monkeypatch.setattr(experiment, "experiment_config_document", lambda exp: document)
    assert _bytes(restart_identity_payload(object())) == expected


@pytest.mark.parametrize("name,value", OPTIONS.items())
def test_enabled_radiation_options_bind_public_echo_and_restart_identity(name, value):
    exp = _experiment(**{name: value})
    old = _historical_schema(exp)
    assert domain_config_document(exp.domains[0])["run"][name] == value
    assert experiment_config_document(exp)["domains"][0]["run"][name] == value
    assert restart.configuration_echo(exp.domains[0].run)[name] == value
    assert _bytes(restart_identity_payload(exp)) != _bytes(restart_identity_payload(old))
    assert effective_prepared_domain_config(domain_config_document(exp.domains[0])) \
        == effective_prepared_domain_config(domain_config_document(old.domains[0]))
