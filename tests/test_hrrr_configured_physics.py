"""Native source preparation must carry actual physics, independent of presets."""
from dataclasses import asdict, replace
from datetime import datetime
from types import SimpleNamespace
import copy

import numpy as np
import pytest

from gpuwm.experiment import VerticalConfig, build_experiment
from gpuwm.experiment_document import render_experiment_document, publish_experiment_document
from gpuwm.hrrr_configuration import resolve_root_experiment
from gpuwm.hrrr_route_inputs import render_namelist_input, target_domain
from gpuwm.hrrr_prepared_bundle import render_wps_namelist
from gpuwm.ingest.hrrr_target import HrrrTargetDomain
from gpuwm.ingest.microphysics_cold_start import cold_start_contract, source_absent_microphysics
from gpuwm.physics_compat import WSM6_PROFILE_ID, MYNN_RUC_PROFILE_ID, single_domain_runtime_switches
from tools import hrrr_single_domain_benchmark as benchmark


def _case(tmp_path, changes=None):
    vertical = VerticalConfig(eta_levels=tuple(float(x) for x in np.linspace(1, 0, 13)),
                              p_top=5000., hybrid_opt=2, etac=.2)
    target = replace(HrrrTargetDomain.legacy_500x500(), nx=50, ny=50, nz=12)
    raw, _ = benchmark._experiment_tables(vertical, run_seconds=3600, target=target,
                                          physics_profile=WSM6_PROFILE_ID)
    if changes:
        raw["shared"].update(changes)
    exp = build_experiment(copy.deepcopy(raw), source="configured native control")
    config = tmp_path / "experiment.toml"
    config.write_text(render_experiment_document(raw), encoding="utf-8")
    namelist = tmp_path / "namelist.input"
    namelist.write_text(render_namelist_input(exp), encoding="utf-8")
    wps = tmp_path / "namelist.wps"
    wps.write_text(render_wps_namelist(exp), encoding="utf-8")
    return exp, target, config, namelist, wps


@pytest.mark.parametrize("changes", [{}, {"sf_sfclay_physics": 1, "epssm": .17},
    {"bl_pbl_physics": 5, "sf_surface_physics": 3, "num_soil_layers": 6, "bldt": 0.},
    {"cu_physics": 1, "cudt_minutes": 7.5}])
def test_actual_configuration_reaches_published_root_without_preset_reconstruction(tmp_path, changes):
    exp, target, config, namelist, wps = _case(tmp_path, changes)
    actual, raw = resolve_root_experiment(target=target, vertical=exp.vertical,
        namelist_input=namelist, start_time=exp.start_time, run_seconds=exp.run_seconds,
        experiment_config=config, wps_namelist=wps)
    assert asdict(actual.root.run) == asdict(exp.root.run)
    publish_experiment_document(tmp_path / "published.toml", raw, actual)
    receipt = benchmark._configured_physics_receipt(actual.root.run)
    assert receipt["profile"] is None
    for key, value in changes.items():
        assert receipt["resolved"][key] == value


@pytest.mark.parametrize("supplied_wps", [False, True])
def test_namelist_only_uses_common_importer_and_keeps_selected_surface(tmp_path, supplied_wps):
    exp, target, config, namelist, wps = _case(tmp_path, {"sf_sfclay_physics": 1})
    actual, raw = resolve_root_experiment(target=target, vertical=exp.vertical,
        namelist_input=namelist, start_time=exp.start_time, run_seconds=exp.run_seconds,
        wps_namelist=wps if supplied_wps else None,
        acknowledgements=tuple(exp.acknowledgements))
    assert actual.root.run.sf_sfclay_physics == 1
    assert actual.root.run.mp_physics == exp.root.run.mp_physics
    assert actual.root.run.cu_physics == exp.root.run.cu_physics


def test_named_profile_remains_an_explicit_equality_assertion(tmp_path):
    exp, target, config, namelist, wps = _case(tmp_path, {"sf_sfclay_physics": 1})
    with pytest.raises(ValueError, match="differs from profile"):
        resolve_root_experiment(target=target, vertical=exp.vertical,
            namelist_input=namelist, start_time=exp.start_time, run_seconds=exp.run_seconds,
            experiment_config=config, physics_profile=WSM6_PROFILE_ID)


def test_configuration_geometry_mismatch_still_refuses_before_source_consumption(tmp_path):
    exp, target, config, namelist, _ = _case(tmp_path)
    with pytest.raises(ValueError, match="configured d01 differs"):
        resolve_root_experiment(target=replace(target, nx=51), vertical=exp.vertical,
            namelist_input=namelist, start_time=exp.start_time, run_seconds=exp.run_seconds,
            experiment_config=config)


@pytest.mark.parametrize("mp", [1, 6, 8, 9, 10, 16, 18, 50])
def test_source_absent_contract_depends_on_microphysics_not_surface_or_profile(mp):
    baseline = SimpleNamespace(mp_physics=mp, wdm6_ccn_conc=1.23e8,
                               sf_surface_physics=2, bl_pbl_physics=1)
    varied = SimpleNamespace(**(vars(baseline) | {"sf_surface_physics": 3, "bl_pbl_physics": 5}))
    assert cold_start_contract(baseline) == cold_start_contract(varied)
    if mp == 16:
        assert cold_start_contract(baseline)[1]["nn"][0] == float(np.float32(1.23e8))


def test_omitted_cli_window_and_history_use_actual_configuration(tmp_path):
    from tools.prepare_hrrr_wrf import _configured_defaults
    exp, _, config, _, _ = _case(tmp_path)
    args = SimpleNamespace(experiment_config=config, run_seconds=None,
                           history_interval_seconds=None)
    _configured_defaults(args)
    assert args.run_seconds == exp.run_seconds
    assert args.history_interval_seconds == exp.root.history_interval_s
    args.run_seconds, args.history_interval_seconds = 1800., 30.
    _configured_defaults(args)
    assert (args.run_seconds, args.history_interval_seconds) == (1800., 30.)


@pytest.mark.parametrize("mp", [1, 6, 8, 9, 10, 16, 18, 50])
def test_cold_start_contract_matches_actual_host_state_allocation(mp):
    from gpuwm.config import RunConfig
    from gpuwm.core.state import DomainState
    cfg = RunConfig(nx=12, ny=12, nz=12, dx=3000., dy=3000., ztop=20000.,
                    dt=5., run_seconds=30., moist=True, mp_physics=mp,
                    wdm6_ccn_conc=1.23e8)
    state = DomainState(cfg, array_module=np)
    _, expected = cold_start_contract(cfg)
    for name, (value, bits) in expected.items():
        field = getattr(state, name)
        assert field.dtype == np.float32
        assert np.all(field.view(np.uint32) == bits), name


def test_namelist_only_defaults_keep_producing_clock_and_cli_override(tmp_path):
    import json
    from tools.prepare_hrrr_wrf import _configured_defaults
    exp, target, _, namelist, wps = _case(tmp_path)
    domain = tmp_path / "target.json"
    domain.write_text(json.dumps(target.to_payload()))
    args = SimpleNamespace(experiment_config=None, namelist_input=namelist,
        domain_spec=domain, wps_namelist=wps, physics_profile=None,
        ack=exp.acknowledgements, run_seconds=None, history_interval_seconds=None)
    _configured_defaults(args)
    assert args.run_seconds == exp.run_seconds == 3600.
    assert args.history_interval_seconds == exp.root.history_interval_s
    args.run_seconds, args.history_interval_seconds = 1800., 30.
    _configured_defaults(args)
    assert (args.run_seconds, args.history_interval_seconds) == (1800., 30.)


def test_prepared_authority_excludes_execution_tiles_but_preserves_physics(tmp_path):
    import tomllib
    exp, target, config, namelist, wps = _case(tmp_path)
    original = config.read_text() + '\n[tiles]\nmode = "auto"\n[fetch]\nsource = "hrrr"\n'
    config.write_text(original)
    actual, raw = resolve_root_experiment(target=target, vertical=exp.vertical,
        namelist_input=namelist, start_time=exp.start_time, run_seconds=exp.run_seconds,
        experiment_config=config, wps_namelist=wps)
    path = publish_experiment_document(tmp_path / "prepared.toml", raw, actual)
    assert "tiles" not in tomllib.loads(path.read_text())
    assert asdict(actual.root.run) == asdict(exp.root.run)
    assert config.read_text() == original



@pytest.mark.parametrize("downscale", [True, False])
def test_native_companion_tables_survive_resolution_and_publication(tmp_path, downscale):
    import tomllib
    from gpuwm.branch import emit_experiment_toml
    from gpuwm.experiment import load_experiment

    exp, target, config, namelist, wps = _case(tmp_path)
    companions = {
        "case_data": {"forcing": "not-fetched.grib", "vtable": "Vtable",
            "wps_namelist": "namelist.wps", "geog_root": "geog",
            "sfcp_to_sfcp": True, "output_title": "declared settings",
            "co2_vmr": 0.000731, "source_orography": {"d01": "terrain.nc"},
            "source_orography_variable": "z"},
        "static": {"highres": {"enabled": False, "cache_root": "static-cache"}},
        "ingest": {"soil_texture_downscale": downscale},
    }
    original = config.read_text() + emit_experiment_toml(companions)
    config.write_text(original)
    actual, raw = resolve_root_experiment(target=target, vertical=exp.vertical,
        namelist_input=namelist, start_time=exp.start_time, run_seconds=exp.run_seconds,
        experiment_config=config, wps_namelist=wps)
    assert asdict(actual.root.run) == asdict(exp.root.run)
    expected_companions = {name: dict(value) for name, value in companions.items()}
    from gpuwm.case_data import resolved_case_data_paths
    expected_companions["case_data"] = resolved_case_data_paths(
        companions["case_data"], base_dir=tmp_path, source=str(config))
    expected_companions["static"] = {"highres": {
        **companions["static"]["highres"], "cache_root": str(tmp_path / "static-cache")}}
    assert {name: raw[name] for name in companions} == expected_companions
    published = publish_experiment_document(tmp_path / "prepared.toml", raw, actual)
    assert {name: tomllib.loads(published.read_text())[name]
            for name in companions} == expected_companions
    assert load_experiment(published) == actual
    assert config.read_text() == original


@pytest.mark.parametrize("extra,match", [
    ('[ingest]\nsoil_texture_downscale = "off"\n', 'must be true or false'),
    ('[static.highres]\nenabled = true\ncache_root = "c"\nunowned = 1\n',
     'does not have a key'),
    ('[case_data]\nco2_vmr = 0.0007\n', 'missing'),
])
def test_native_companion_owner_validation_still_refuses_invalid_data(tmp_path, extra, match):
    exp, target, config, namelist, wps = _case(tmp_path)
    config.write_text(config.read_text() + extra)
    with pytest.raises(ValueError, match=match):
        resolve_root_experiment(target=target, vertical=exp.vertical,
            namelist_input=namelist, start_time=exp.start_time, run_seconds=exp.run_seconds,
            experiment_config=config, wps_namelist=wps)


@pytest.mark.parametrize("changes", [
    {"mp_physics": 9},
    {"mp_physics": 16, "wdm6_ccn_conc": 1.23e8, "wdm6_hail_opt": 1},
    {"bl_pbl_physics": 5, "sf_surface_physics": 3, "num_soil_layers": 6, "bldt": 0.},
])
def test_full_native_emission_and_prepared_guard_keep_configured_physics(tmp_path, changes):
    from gpuwm.hrrr_route_inputs import write_hrrr_route_inputs
    from gpuwm.namelist_import import parse_namelist

    exp, target, config, namelist, wps = _case(tmp_path, changes)
    paths = write_hrrr_route_inputs(config, exp, wps_text=render_wps_namelist(exp),
        writer=lambda path, text: path.write_text(text, encoding="utf-8"))
    if exp.root.run.mp_physics == 16:
        for path in (paths[2], paths[3]):
            physics = parse_namelist(path)["physics"]
            assert physics["hail_opt"] == [1]
            assert physics["ccn_conc"] == [1.23e8]
    actual, raw = resolve_root_experiment(target=target, vertical=exp.vertical,
        namelist_input=namelist, start_time=exp.start_time, run_seconds=exp.run_seconds,
        experiment_config=config, wps_namelist=wps)
    assert asdict(actual.root.run) == asdict(exp.root.run)
    publish_experiment_document(tmp_path / "published.toml", raw, actual)
    receipt = benchmark._configured_physics_receipt(actual.root.run,
        acknowledgements=tuple(actual.acknowledgements))
    assert receipt["profile"] is None
    benchmark._validate_resolved_hrrr_profile(actual, receipt)
    assert all(receipt["resolved"][key] == value for key, value in changes.items())
    receipt["resolved"]["mp_physics"] = 14
    with pytest.raises(RuntimeError, match="physics differs"):
        benchmark._validate_resolved_hrrr_profile(actual, receipt)


@pytest.mark.parametrize("changes,match", [
    ({"mp_physics": 16, "wdm6_ccn_conc": 1.e7}, "outside WDM6"),
    ({"mp_physics": 16, "wdm6_hail_opt": 2}, "wdm6_hail_opt"),
    ({"sf_surface_physics": 3, "num_soil_layers": 5}, "soil"),
    ({"mp_physics": 14}, "not ported"),
    ({"mp_physics": 26}, "not ported"),
    ({"mp_physics": 51}, "not ported"),
    ({"mp_physics": 28}, "aerosol boundary species"),
])
def test_native_emission_still_refuses_invalid_and_unported_selections(tmp_path, changes, match):
    with pytest.raises(ValueError, match=match):
        _case(tmp_path, changes)
