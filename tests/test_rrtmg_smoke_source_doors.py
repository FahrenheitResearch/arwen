"""Prescribed smoke binds source bytes while an absent source stays absent.

These metadata checks prevent a restart from accepting changed profiles at
the same path and a radiation-only forcing from invalidating preparation.
No profile processing or device access is performed here.
"""
from __future__ import annotations

from copy import deepcopy
import dataclasses
import hashlib
import json
from types import SimpleNamespace

import pytest

from gpuwm.config import RunConfig, validate_radiation_driver_options
from gpuwm.core.model import experiment_fingerprint, restart_identity_payload
from gpuwm.core.rrtmg_smoke_identity import SMOKE_IDENTITY_KEY
from gpuwm.experiment import experiment_from_run_config, load_experiment
from gpuwm.ingest.prepared_cache import compare_prepared_domain_config
from gpuwm.io import restart
from test_rrtmg_smoke_manifest import START, _fixture


def _cfg(path="", **overrides):
    values = dict(nx=4, ny=3, nz=2, dx=2000., dy=2000., ztop=8000.,
                  dt=10., run_seconds=7200., moist=True, mp_physics=28,
                  ra_physics=4, ra_rrtmg_variant="rrtmg_legacy", aer_opt=3,
                  rrtmg_smoke_manifest=str(path))
    values.update(overrides)
    return RunConfig(**values)


def _replace_member(path, raw):
    """Change actual bytes and their declaration without changing the path."""
    member = raw["frames"][0]["value"]
    binary = path.parent / member["path"]
    changed = bytearray(binary.read_bytes())
    changed[0] ^= 1
    binary.write_bytes(changed)
    member["sha256"] = hashlib.sha256(changed).hexdigest()
    path.write_text(json.dumps(raw), encoding="utf-8")


def test_required_radiation_path_is_admitted_without_reading_the_source():
    validate_radiation_driver_options(_cfg("not-opened-at-config-validation.json"))


@pytest.mark.parametrize("override", [
    {"ra_physics": 0}, {"ra_physics": 0, "ra_lw_physics": 4, "ra_sw_physics": 1},
    {"ra_rrtmg_variant": "rte-rrtmgp"}, {"aer_opt": 0},
    {"mp_physics": 8}, {"rrtmg_smoke_manifest": None},
])
def test_a_path_that_would_discard_smoke_is_refused(override):
    with pytest.raises(ValueError, match="rrtmg_smoke_manifest"):
        validate_radiation_driver_options(_cfg("source.json", **override))


def test_empty_source_reads_nothing_and_retains_pre_field_identity(monkeypatch):
    from gpuwm.core import rrtmg_smoke_manifest as provider
    def refused(*args, **kwargs):
        raise AssertionError("an empty source must not read data")
    monkeypatch.setattr(provider, "describe_smoke_source", refused)
    cfg = _cfg()
    old = dataclasses.asdict(cfg)
    old.pop("rrtmg_smoke_manifest")
    echo = restart.configuration_echo(cfg)
    digest = restart._configuration_digest_values(dataclasses.asdict(cfg))
    prior_echo = deepcopy(old)
    restart._drop_default_off_run_keys(prior_echo)
    assert json.dumps(echo).encode() == json.dumps(prior_echo).encode()
    assert digest == restart._configuration_digest_values(old)
    exp = experiment_from_run_config(cfg, START)
    document = restart_identity_payload(exp)
    assert "rrtmg_smoke_manifest" not in document["domains"][0]["run"]
    assert SMOKE_IDENTITY_KEY not in document["domains"][0]["run"]
    restart._require_config_match(echo, cfg, "older-restart.npz")


def test_echo_and_digest_bind_member_bytes_and_preserve_saved_identity(tmp_path):
    path, raw = _fixture(tmp_path)
    cfg = _cfg(path)
    saved = restart.configuration_echo(cfg)
    original_digest_values = restart._configuration_digest_values(saved)
    assert saved[SMOKE_IDENTITY_KEY]["frames"][0]["value"]["sha256"] == \
        raw["frames"][0]["value"]["sha256"]
    _replace_member(path, raw)
    live = restart.configuration_echo(cfg)
    assert live[SMOKE_IDENTITY_KEY] != saved[SMOKE_IDENTITY_KEY]
    assert restart._configuration_digest_values(saved) == original_digest_values
    assert restart._configuration_digest_values(dataclasses.asdict(cfg)) != \
        original_digest_values
    with pytest.raises(restart.RestartMismatchError, match=SMOKE_IDENTITY_KEY):
        restart._require_config_match(saved, cfg, tmp_path / "state.npz")


def test_old_restart_refuses_enabled_source_and_missing_identity(tmp_path):
    path, _ = _fixture(tmp_path)
    cfg = _cfg(path)
    with pytest.raises(restart.RestartMismatchError, match="rrtmg_smoke_manifest"):
        restart._require_config_match(restart.configuration_echo(_cfg()), cfg,
                                      tmp_path / "old.npz")
    stored = restart.configuration_echo(cfg)
    stored.pop(SMOKE_IDENTITY_KEY)
    with pytest.raises(restart.RestartMismatchError, match=SMOKE_IDENTITY_KEY):
        restart._require_config_match(stored, cfg, tmp_path / "missing.npz")


def test_forecast_fingerprint_changes_when_same_path_source_changes(tmp_path):
    path, raw = _fixture(tmp_path)
    exp = experiment_from_run_config(_cfg(path), START)
    catalog = SimpleNamespace(run_provenance={"source": "same-inventory"})
    before = experiment_fingerprint(exp, catalog)
    _replace_member(path, raw)
    assert experiment_fingerprint(exp, catalog) != before
    mismatched = dataclasses.replace(exp, start_time=START.replace(hour=22),
        domains=tuple(dataclasses.replace(domain, start_time=START.replace(hour=22))
                      for domain in exp.domains))
    with pytest.raises(ValueError, match="actual case start"):
        restart_identity_payload(mismatched)


def test_prescribed_source_is_inert_in_actual_prepared_identity():
    from gpuwm.ingest.prepared_cache import prepared_domain_config_identity
    exp = experiment_from_run_config(_cfg(), START)
    prior = prepared_domain_config_identity(exp.root)
    changed = prepared_domain_config_identity(dataclasses.replace(
        exp.root, run=dataclasses.replace(exp.root.run,
            rrtmg_smoke_manifest="not-opened-during-preparation.json")))
    assert SMOKE_IDENTITY_KEY not in prior["run"]
    assert SMOKE_IDENTITY_KEY not in changed["run"]
    assert compare_prepared_domain_config(prior, changed) == ([], [])
    changed["run"]["nz"] = 3
    assert compare_prepared_domain_config(prior, changed) == ([], ["run.nz"])


@pytest.mark.parametrize("door", ["legacy", "experiment"])
def test_config_relative_source_is_normalized_at_both_doors(tmp_path, monkeypatch, door):
    from gpuwm.config import load_config
    from test_experiment import BASE
    home = tmp_path / "config"
    home.mkdir()
    source = home / "source.json"
    source.write_text("{}", encoding="utf-8")
    path = home / "config.toml"
    monkeypatch.chdir(tmp_path)
    if door == "legacy":
        path.write_text('[grid]\nnx=6\nny=4\nnz=5\ndx=2000.0\ndy=2000.0\n'
                        'ztop=8000.0\ndt=10.0\nrun_seconds=7200.0\n'
                        '[run]\nmoist=true\nmp_physics=28\nra_physics=4\n'
                        'ra_rrtmg_variant="rrtmg_legacy"\naer_opt=3\n'
                        'rrtmg_smoke_manifest="source.json"\n', encoding="utf-8")
        cfg = load_config(path)
    else:
        path.write_text(BASE.format(experiment="restart_interval_s=0.0", d01="", d02="", shared=
            'moist=true\nmp_physics=28\nra_physics=4\n'
            'ra_rrtmg_variant="rrtmg_legacy"\naer_opt=3\n'
            'rrtmg_smoke_manifest="source.json"'), encoding="utf-8")
        cfg = load_experiment(path).root.run
    assert cfg.rrtmg_smoke_manifest == str(source.resolve())


def test_config_path_anchoring_keeps_working_directory_precedence(tmp_path, monkeypatch):
    from gpuwm.config import _anchor_config_file_paths
    (tmp_path / "source.json").write_text("{}", encoding="utf-8")
    home = tmp_path / "config"
    home.mkdir()
    (home / "source.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    values = {"rrtmg_smoke_manifest": "source.json"}
    _anchor_config_file_paths(values, home / "config.toml")
    assert values == {"rrtmg_smoke_manifest": "source.json"}
