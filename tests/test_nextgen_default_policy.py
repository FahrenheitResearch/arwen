"""Exercise launcher/member boundaries without fabric or numerical stand-ins."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import Context
import json
import os
from types import SimpleNamespace

import pytest

from gpuwm.ensemble.optional_routing import (
    activation_receipt, enabled_for, mass_member_settings, member_settings,
    member_settings_scope,
)
from gpuwm.hydro import runtime


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    monkeypatch.delenv(runtime.ENV_ENABLED, raising=False)
    monkeypatch.delenv(runtime.ENV_SETTINGS, raising=False)
    monkeypatch.delenv("GPUWM_HYDRO_DOOR_DECISION_JSON", raising=False)


def config(tmp_path, text, name="experiment"):
    path = tmp_path / (name + ".toml")
    path.write_text(text)
    return SimpleNamespace(config=path, experiment_config=path, hydro=None)


@pytest.mark.parametrize('composition', ['realization="prepared-test-row"'])
def test_prepared_settings_default_on_across_cli_and_member(tmp_path, composition):
    args = config(tmp_path, "[hydro]\n" + composition + "\n")
    @runtime.go_scope
    def launch(args):
        assert runtime.current_settings()["enabled"] is True
        assert enabled_for(args)
        snapshot = member_settings(args)
        def worker():
            with member_settings_scope(args, inherited_settings=snapshot):
                return activation_receipt(runtime.current_settings())
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(Context().run, worker).result()
    receipt = launch(args)
    assert receipt["enabled"] is True
    assert receipt["reason"] == "prepared_configuration_default"
    assert runtime.current_settings() is None
    assert runtime.ENV_ENABLED not in os.environ


def test_cli_off_beats_member_table_and_worker_snapshot_even_after_failure(tmp_path, monkeypatch):
    args = config(tmp_path, '[hydro]\nrealization="prepared-test-row"\n')
    member = config(tmp_path, '[hydro]\nenabled=true\nrealization="member"\n', "member")
    args.hydro = "off"
    monkeypatch.setenv(runtime.ENV_ENABLED, "1")
    monkeypatch.setenv(runtime.ENV_SETTINGS, str(args.config))
    inherited = {"enabled": True, "stages": [{"name": "prepared"}]}
    @runtime.go_scope
    def launch(args):
        assert os.environ[runtime.ENV_ENABLED] == "0"
        assert runtime.ENV_SETTINGS not in os.environ
        assert not enabled_for(member, inherited_settings=inherited)
        snapshots = mass_member_settings(
            [{"experiment_config": str(member.config)}],
            [{"parent_index": 0, "member_id": 3}, {"parent_index": 0, "member_id": 8}])
        assert all(row["activation_reason"] == "cli_off" for row in snapshots.values())
        with pytest.raises(ValueError, match="every member"):
            mass_member_settings([{"experiment_config": str(member.config)}],
                [{"parent_index": 0, "member_id": 3}], products_requested=True)
        def worker():
            with member_settings_scope(member, inherited_settings=inherited):
                assert not runtime.current_settings()["enabled"]
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(Context().run, worker).result()
        raise RuntimeError("forecast failed")
    with pytest.raises(RuntimeError, match="forecast failed"):
        launch(args)
    assert os.environ[runtime.ENV_ENABLED] == "1"
    assert os.environ[runtime.ENV_SETTINGS] == str(args.config)
    # The durable copied opt-out also survives a worker without launcher env.
    monkeypatch.delenv(runtime.ENV_ENABLED)
    monkeypatch.delenv(runtime.ENV_SETTINGS)
    assert not enabled_for(member, inherited_settings={"enabled": False, "activation_reason": "cli_off"})


@pytest.mark.parametrize('text', ['[hydro]\n'])
def test_missing_or_disabled_configuration_has_explicit_receipt(tmp_path, text):
    args = config(tmp_path, text)
    @runtime.go_scope
    def launch(args):
        return activation_receipt(runtime.current_settings())
    receipt = json.loads(json.dumps(launch(args)))
    assert receipt["enabled"] is False
    # With no prepared configuration the default-on door decides; these test
    # files have no parent grid, so its named gap is the missing geometry.
    assert receipt["reason"] == ("no_domain_geometry")
    assert "coverage" in receipt["scope"]
    assert receipt["door"]["gaps"][0]["owner"] == "hydro-world-door"


def test_prepared_forecast_entry_defaults_on_without_go_launcher(tmp_path):
    inputs = config(tmp_path, '[hydro]\nrealization="prepared-test-row"\n')
    @runtime.forecast_scope
    def forecast(inputs):
        return runtime.current_settings()["enabled"]
    assert forecast(inputs) is True
    assert runtime.current_settings() is None


def test_launch_reason_does_not_change_checkpoint_realization_identity(tmp_path):
    args = config(tmp_path, '[hydro]\nrealization="lower-colorado-hyfeatures"\n'
                  '[hydro.settings_bindings]\nfabric_manifest="/prepared/manifest.json"\n')

    def snapshot():
        settings = runtime.current_settings()
        receipt = activation_receipt(settings)
        resolved = runtime.resolve_settings(settings)
        # Preparation must leave the launch decision available for receipts.
        assert activation_receipt(settings) == receipt
        return receipt["reason"], resolved["realization_identity"]

    @runtime.go_scope
    def launch(args):
        rows = [snapshot()]
        with ThreadPoolExecutor(max_workers=1) as pool:
            # A child without an in-process scope inherits the launcher env.
            rows.append(pool.submit(Context().run, snapshot).result())
        with member_settings_scope(args):
            rows.append(snapshot())
        return rows

    rows = launch(args)
    args.hydro = str(args.config)
    rows.extend(launch(args))
    assert {reason for reason, _ in rows} == {
        "prepared_configuration_default", "cli_enabled", "launcher_enabled"}
    assert all(identity == rows[0][1] for _, identity in rows)
    assert runtime.current_settings() is None


def test_explicit_missing_settings_refuses_before_runner(tmp_path):
    args = config(tmp_path, "")
    args.hydro = str(tmp_path / "absent.toml")
    @runtime.go_scope
    def launch(args):
        pytest.fail("missing explicit settings reached the runner")
    with pytest.raises(ValueError, match="settings file does not exist"):
        launch(args)


def test_invalid_enabled_cannot_silently_disable_hydro(tmp_path):
    args = config(tmp_path, '[hydro]\nenabled="true"\n')
    with pytest.raises(ValueError, match="enabled must be a boolean"):
        member_settings(args)


@pytest.mark.parametrize("table", [
    'hydro="unavailable preparation"\n',
    '[hydro]\nenabled="true"\n',
])
def test_cli_off_ignores_invalid_hydro_metadata(tmp_path, table):
    args = config(tmp_path, table)
    args.hydro = "off"

    @runtime.go_scope
    def launch(args):
        assert os.environ[runtime.ENV_ENABLED] == "0"
        assert runtime.ENV_SETTINGS not in os.environ
        assert not enabled_for(args)
        with member_settings_scope(args):
            return activation_receipt(runtime.current_settings())

    receipt = launch(args)
    assert receipt["enabled"] is False
    assert receipt["reason"] == "cli_off"
    assert runtime.current_settings() is None
    assert runtime.ENV_ENABLED not in os.environ
    # The default path still reports invalid preparation; only explicit off
    # bypasses hydro metadata. The atmosphere runner owns its configuration.
    args.hydro = None
    with pytest.raises(ValueError):
        launch(args)


def test_member_table_does_not_open_unrelated_broken_launcher_file(tmp_path, monkeypatch):
    member = config(tmp_path, '[hydro]\nrealization="prepared-test-row"\n')
    monkeypatch.setenv(runtime.ENV_SETTINGS, str(tmp_path / "absent.toml"))
    monkeypatch.setenv(runtime.ENV_ENABLED, "1")
    assert enabled_for(member)
    with member_settings_scope(member):
        assert runtime.current_settings()["realization"] == "prepared-test-row"
    # A scoped explicit opt-out still beats that member table without env.
    monkeypatch.delenv(runtime.ENV_ENABLED)
    monkeypatch.delenv(runtime.ENV_SETTINGS)
    with runtime.settings_scope({"enabled": False, "activation_reason": "cli_off"}):
        assert not enabled_for(member)
