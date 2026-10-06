"""Sun-angle albedo doors and unchanged off/default identities. CPU only."""
from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pytest

from gpuwm.config import RunConfig, validate_radiation_driver_options
from gpuwm.io import restart


def _cfg(**changes):
    values = dict(nx=6, ny=4, nz=5, dx=3000.0, dy=3000.0, ztop=20000.0,
                  dt=20.0, run_seconds=0.0, ra_physics=4,
                  ra_rrtmg_variant="rrtmg_legacy")
    values.update(changes)
    return RunConfig(**values)


@pytest.mark.parametrize("value", [-1, 2, 1.0, True, "1", None])
def test_alb_sol_accepts_only_integer_zero_or_one(value):
    with pytest.raises(ValueError, match="alb_sol.*integer 0"):
        validate_radiation_driver_options(_cfg(alb_sol=value))


def test_alb_sol_needs_radiation_that_updates_it():
    validate_radiation_driver_options(_cfg(alb_sol=0))
    validate_radiation_driver_options(_cfg(alb_sol=1))
    with pytest.raises(ValueError, match="alb_sol=1 needs a shortwave"):
        validate_radiation_driver_options(_cfg(alb_sol=1, ra_physics=0))


def test_disabled_alb_sol_keeps_pre_field_checkpoint_echo_and_digest():
    cfg = _cfg()
    before = dataclasses.asdict(cfg)
    before.pop("alb_sol")
    expected = restart._configuration_digest_values(before)
    assert restart._configuration_digest_values(dataclasses.asdict(cfg)) == expected
    assert "alb_sol" not in restart.configuration_echo(cfg)
    old_echo = restart.configuration_echo(cfg)
    restart._require_config_match(old_echo, cfg, "previous.npz")
    on = _cfg(alb_sol=1)
    assert restart.configuration_echo(on)["alb_sol"] == 1
    with pytest.raises(restart.RestartMismatchError, match="alb_sol"):
        restart._require_config_match(old_echo, on, "previous.npz")
    with pytest.raises(restart.RestartMismatchError, match="alb_sol"):
        restart._require_config_match(restart.configuration_echo(on), cfg, "enabled.npz")


def test_disabled_alb_sol_keeps_pre_field_experiment_identity(monkeypatch):
    from gpuwm import experiment
    from gpuwm.core.model import restart_identity_payload

    document = {"domains": [{"run": {"alb_sol": 0}}]}
    monkeypatch.setattr(experiment, "experiment_config_document", lambda exp: document)
    off = restart_identity_payload(object())
    document = {"domains": [{"run": {}}]}
    assert restart_identity_payload(object()) == off
    document = {"domains": [{"run": {"alb_sol": 1}}]}
    assert restart_identity_payload(object()) != off


def test_alb_sol_reuses_identical_preparation_at_both_values():
    from gpuwm.ingest.prepared_cache import effective_prepared_domain_config
    absent = {"run": {"ra_physics": 4}}
    for value in (0, 1):
        changed = {"run": {"ra_physics": 4, "alb_sol": value}}
        assert effective_prepared_domain_config(changed) == effective_prepared_domain_config(absent)


def test_operational_namelist_value_reaches_config_and_route_round_trip(tmp_path):
    from gpuwm.hrrr_route_inputs import render_namelist_input
    from gpuwm.namelist_import import parse_namelist
    from test_namelist_import import _import_radiation_options, _load

    fixture = Path(__file__).parent / "data" / "albsol_config" / "hrrr_wrf.nl"
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == (
        "50ac01dbeaca863dfc313eae7dd53865458b2bffdfcc1e402d350d860bef5694")
    (value,) = parse_namelist(fixture)["physics"]["alb_sol"]
    assert value == 1
    text, report = _import_radiation_options(tmp_path, [f"alb_sol = {value}"])
    exp = _load(tmp_path, text)
    assert all(domain.run.alb_sol == 1 for domain in exp.domains)
    assert ("physics", "alb_sol") in {(row.section, row.key) for row in report.translated}
    emitted = tmp_path / "emitted.namelist.input"
    emitted.write_text(render_namelist_input(exp), encoding="utf-8")
    assert parse_namelist(emitted)["physics"]["alb_sol"] == [1]


def test_omitted_and_explicit_off_imports_are_byte_identical(tmp_path):
    from test_namelist_import import _import_radiation_options

    absent, _ = _import_radiation_options(tmp_path, [])
    off, _ = _import_radiation_options(tmp_path, ["alb_sol = 0"])
    assert absent.encode("utf-8") == off.encode("utf-8")
    assert "alb_sol" not in absent


@pytest.mark.parametrize("line", ["alb_sol = 2", "alb_sol = .true.",
                                  "alb_sol = 1.0", "alb_sol = 0, 1"])
def test_namelist_alb_sol_cannot_silently_substitute_another_value(tmp_path, line):
    from test_namelist_import import _import_radiation_options

    with pytest.raises(ValueError, match="alb_sol"):
        _import_radiation_options(tmp_path, [line])


@pytest.mark.parametrize("filename", ["hrrr_native_quick_demo.toml",
    "hrrr_native_3km_demo.toml", "hrrr_prs_demo.toml", "hrrr_prs_3km_demo.toml"])
def test_shipped_hrrr_recipe_selects_sun_angle_albedo(filename):
    from gpuwm.experiment import load_experiment

    root = Path(__file__).resolve().parents[1]
    exp = load_experiment(root / "configs" / filename)
    assert all(domain.run.alb_sol == 1 for domain in exp.domains)


def test_solar_ruc_template_selects_albedo_without_changing_prior_template():
    from gpuwm.physics_compat import single_domain_runtime_switches

    settings = single_domain_runtime_switches(
        "thompson-mp8-mynn-mynn-ruc-monthly-solar-rrtmg-legacy-v1")
    assert settings["alb_sol"] == 1
    prior = single_domain_runtime_switches(
        "thompson-mp8-mynn-mynn-ruc-monthly-rrtmg-legacy-v1")
    assert prior.get("alb_sol", 0) == 0


@pytest.mark.parametrize("filename", ["hrrr_native_quick_demo.toml",
                                     "hrrr_native_3km_demo.toml"])
def test_native_albedo_and_stock_omission_have_an_explicit_delta(tmp_path, filename):
    from gpuwm.experiment import load_experiment
    from gpuwm.hrrr_route_inputs import render_namelist_input
    from gpuwm.hrrr_hierarchy_direct import _require_raw_stock_delta
    from gpuwm.namelist_import import parse_namelist

    exp = load_experiment(Path(__file__).resolve().parents[1] / "configs" / filename)
    native = tmp_path / "native.input"
    stock = tmp_path / "stock.input"
    native.write_text(render_namelist_input(exp), encoding="utf-8")
    stock.write_text(render_namelist_input(exp, stock=True), encoding="utf-8")
    assert parse_namelist(native)["physics"]["alb_sol"] == [1]
    assert "alb_sol" not in parse_namelist(stock)["physics"]
    receipt = _require_raw_stock_delta(native, stock)
    delta = receipt["allowed_deltas"]["physics.alb_sol"]
    assert delta["native"] == [1] and delta["stock"] is None
    assert delta["stock_effective"] == 0
    assert delta["physics_equivalent"] is False
    assert "Fortran namelist read fail" in delta["reason"]
    assert "identical land shortwave forcing" in receipt["comparison_limitations"][0]


@pytest.mark.parametrize("native_value,stock_value,match", [
    ("2", None, "native.*alb_sol"),
    (".true.", None, "native.*alb_sol"),
    ("1.0", None, "native.*alb_sol"),
    ("1, 1", None, "native.*alb_sol"),
    ("1", "1", "stock-WRF namelist must omit alb_sol"),
    ("1", "0", "stock-WRF namelist must omit alb_sol"),
    (None, "0", "stock-WRF namelist must omit alb_sol"),
])
def test_fork_only_albedo_delta_is_typed_and_stock_always_omits_it(
        tmp_path, native_value, stock_value, match):
    from gpuwm.hrrr_hierarchy_direct import _require_raw_stock_delta
    from test_hrrr_hierarchy_direct import _raw_runtime_namelist

    native = tmp_path / "native.input"
    stock = tmp_path / "stock.input"
    native_text = _raw_runtime_namelist(1, longwave=0, theta_m=0)
    stock_text = _raw_runtime_namelist(1, longwave=1, theta_m=0,
                                       ghg_input=0, do_radar_ref=1)
    if native_value is not None:
        native_text = native_text.replace("&physics\n", f"&physics\n alb_sol = {native_value},\n")
    if stock_value is not None:
        stock_text = stock_text.replace("&physics\n", f"&physics\n alb_sol = {stock_value},\n")
    native.write_text(native_text, encoding="utf-8")
    stock.write_text(stock_text, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        _require_raw_stock_delta(native, stock)


def test_disabled_raw_albedo_omission_is_equivalent_and_other_drift_refuses(tmp_path):
    from gpuwm.hrrr_hierarchy_direct import _require_raw_stock_delta
    from test_hrrr_hierarchy_direct import _raw_runtime_namelist

    native = tmp_path / "native.input"
    stock = tmp_path / "stock.input"
    native_text = _raw_runtime_namelist(1, longwave=0, theta_m=0)
    native_text = native_text.replace("&physics\n", "&physics\n alb_sol = 0,\n")
    stock_text = _raw_runtime_namelist(1, longwave=1, theta_m=0,
                                       ghg_input=0, do_radar_ref=1)
    native.write_text(native_text, encoding="utf-8")
    stock.write_text(stock_text, encoding="utf-8")
    receipt = _require_raw_stock_delta(native, stock)
    assert receipt["allowed_deltas"]["physics.alb_sol"]["physics_equivalent"] is True
    assert "comparison_limitations" not in receipt
    assert "run_hours = 12" in stock_text
    stock.write_text(stock_text.replace("run_hours = 12", "run_hours = 23"), encoding="utf-8")
    with pytest.raises(ValueError, match="time_control/run_hours"):
        _require_raw_stock_delta(native, stock)


def test_stock_experiment_comparison_allows_only_verified_disabled_albedo():
    from dataclasses import replace
    from gpuwm.experiment import load_experiment
    from gpuwm.hrrr_hierarchy_direct import _compare_stock_experiment

    native = load_experiment(Path(__file__).resolve().parents[1] / "configs" /
                             "hrrr_native_quick_demo.toml")
    stock = replace(native, domains=tuple(
        replace(domain, run=replace(domain.run, alb_sol=0))
        for domain in native.domains))
    _compare_stock_experiment(native, stock)
    with pytest.raises(ValueError, match="must carry alb_sol=0"):
        _compare_stock_experiment(native, native)
    drift = replace(stock, run_seconds=stock.run_seconds + 3600)
    with pytest.raises(ValueError, match="beyond the allowed"):
        _compare_stock_experiment(native, drift)


def test_hrrr_frontdoor_writes_both_halves_without_stock_fork_key(tmp_path):
    from gpuwm.experiment import load_experiment
    from gpuwm.hrrr_route_inputs import write_hrrr_route_inputs, route_input_paths
    from gpuwm.hrrr_hierarchy_direct import _require_raw_stock_delta
    from gpuwm.namelist_import import parse_namelist

    root = Path(__file__).resolve().parents[1]
    exp = load_experiment(root / "configs" / "hrrr_native_quick_demo.toml")
    config = tmp_path / "frontdoor.toml"
    written = write_hrrr_route_inputs(config, exp,
        wps_text=(root / "configs" / "hrrr_native_quick_demo.namelist.wps").read_text(),
        writer=lambda path, content: path.write_text(content, encoding="utf-8"))
    paths = route_input_paths(config)
    assert set(written) == set(paths.values())
    assert parse_namelist(paths["namelist_input"])["physics"]["alb_sol"] == [1]
    assert "alb_sol" not in parse_namelist(paths["stock_namelist_input"])["physics"]
    receipt = _require_raw_stock_delta(paths["namelist_input"], paths["stock_namelist_input"])
    assert receipt["allowed_deltas"]["physics.alb_sol"]["physics_equivalent"] is False
