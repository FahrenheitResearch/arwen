"""A191: every checkpoint writer echoes a run's configuration the same way.

The breakage this prevents: the resident writer dropped nine default-off
RunConfig keys from a checkpoint's ``config`` echo and the streamed writer
wrote ``dataclasses.asdict(cfg)`` whole, so two checkpoints of one state,
one per memory road, compared unequal though every array matched
(tests/test_streamed_relocation_rebuild_gpu.py's
test_live_bounded_move_keeps_owner_and_continues_exactly failed on the
2.8.1 bench cards).  The out-of-core store writer had the same raw echo.

Each case writes the same NumPy-backed state through the resident writer
(``gpuwm.io.restart.write_restart``), the streamed writer
(``tilestream.restart_stream.write_streamed_restart``) and the store writer
(``tilestream.checkpoint.write_store_restart``) and requires one echo from
all three, at the defaults and with each default-off field moved.  CPU only.
"""
from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import pytest

from gpuwm.io import restart
from test_restart import (
    _cfg, _fill_setup, _identity_bound_physics_state, _shim_driver_state)


#: Every default-off field restart._drop_default_off_run_keys drops at its
#: default, moved, as (overrides, the keys the echo must then carry).
MOVED = {
    "thompson_version": (dict(mp_physics=28, moist=True, thompson_version="wrf_39_noaa"), {"thompson_version"}),
    "thompson_fork_snow_fall": (dict(mp_physics=28, moist=True, thompson_version="wrf_39_noaa", thompson_fork_snow_fall="wrf_39_noaa"), {"thompson_version", "thompson_fork_snow_fall"}),
    "upper_wind_limiter_form": (dict(upper_wind_limiter_form="noaa_wrf39"),
                                {"upper_wind_limiter_form"}),
    "adaptive_nest_lattice": (dict(adaptive_nest_lattice=True),
                              {"adaptive_nest_lattice"}),
    "zadvect_implicit": (dict(zadvect_implicit=1), {"zadvect_implicit"}),
    "zadvect_implicit_variant": (
        dict(zadvect_implicit=1, zadvect_implicit_variant="wrf_legacy"),
        {"zadvect_implicit", "zadvect_implicit_variant"}),
    "ruc_irrigation": (dict(ruc_irrigation="wrf_45"), {"ruc_irrigation"}),
    "ruc_qvg_cold_start": (dict(ruc_qvg_cold_start="air"),
                           {"ruc_qvg_cold_start"}),
    "ruc_2m_diagnostic": (dict(ruc_2m_diagnostic="log_profile"),
                          {"ruc_2m_diagnostic"}),
    "ruc_snow": (dict(ruc_snow="wrf_45"), {"ruc_snow"}),
    "w_crit_cfl": (dict(zadvect_implicit=1, w_crit_cfl=2.0),
                   {"zadvect_implicit", "w_crit_cfl"}),
    "slope_rad": (dict(slope_rad=1), {"slope_rad"}),
    "topo_shading": (dict(slope_rad=1, topo_shading=1, shadlen=20000.0),
                     {"slope_rad", "topo_shading", "shadlen"}),
    "topo_wind": (dict(bl_pbl_physics=1, sf_sfclay_physics=1, topo_wind=1),
                  {"topo_wind"}),
    "gwd_opt": (dict(gwd_opt=1), {"gwd_opt"}),
    "sf_surface_mosaic": (dict(sf_surface_physics=2, sf_surface_mosaic=1,
                               mosaic_cat=2),
                          {"sf_surface_mosaic", "mosaic_cat"}),
    "mosaic_urban_canopy": (dict(sf_surface_physics=2, sf_urban_physics=1,
                                 sf_surface_mosaic=1,
                                 mosaic_urban_canopy="every_tile"),
                            {"sf_surface_mosaic", "mosaic_cat",
                             "mosaic_urban_canopy"}),
    "diff_opt": (dict(diff_opt=1), {"diff_opt", "mix_full_fields"}),
    "mix_full_fields": (dict(diff_opt=1, mix_full_fields=False),
                        {"diff_opt", "mix_full_fields"}),
    "scalar_pblmix": (dict(scalar_pblmix=1), {"scalar_pblmix"}),
    "use_rap_aero_icbc": (dict(use_rap_aero_icbc=True), {"use_rap_aero_icbc"}),
    "sf_lake_physics": (dict(sf_lake_physics=1),
                        {"sf_lake_physics", "use_lakedepth",
                         "lakedepth_default", "lake_min_elev"}),
    "spp_conv": (dict(spp_conv=1), {"spp_conv"}),
    "spp_pbl": (dict(spp_pbl=1), {"spp_pbl"}),
    "ruc_soilprop": (dict(ruc_soilprop="wrf_461"), {"ruc_soilprop"}),
    # The terrain-clock mode (lane/286-fixed-step-grid): "measured" is the
    # derivation every header before the field ran under and drops out;
    # "pinned" is echoed.
    "terrain_clock": (dict(terrain_clock="pinned"), {"terrain_clock"}),
    "bl_mynn_cloud_tendency_form": (
        dict(bl_mynn_version="gsd_41", bl_mynn_cloud_tendency_form="gsd_41"),
        {"bl_mynn_version", "bl_mynn_cloud_tendency_form"}),
    "bl_mynn_version": (dict(bl_mynn_version="gsd_41"), {"bl_mynn_version"}),
    "bl_mynn_gsd41_unsquared_qtke": (dict(bl_mynn_gsd41_unsquared_qtke=True),
                                     {"bl_mynn_gsd41_unsquared_qtke"}),
    "mynn_sfclay_variant": (dict(mynn_sfclay_variant="gsl_wrf39"),
                            {"mynn_sfclay_variant"}),
    "diff_6th_form": (dict(diff_6th_form="noaa_wrf39", diff_6th_factor2=0.04),
                      {"diff_6th_form", "diff_6th_factor2"}),
    "mp_zero_out": (dict(mp_zero_out=2),
                    {"mp_zero_out", "mp_zero_out_thresh", "mp_zero_out_all"}),
    "v_sca_adv_order": (dict(v_sca_adv_order=5), {"v_sca_adv_order"}),
    "v_mom_adv_order": (dict(v_mom_adv_order=5), {"v_mom_adv_order"}),
    "rrtmg_cloud_optics_form": (
        dict(rrtmg_cloud_optics_form="noaa_wrf39"),
        {"rrtmg_cloud_optics_form"}),
    "swint_opt": (dict(swint_opt=1), {"swint_opt"}),
    "aer_opt": (dict(aer_opt=3, mp_physics=28, moist=True,
                     ra_physics=4, ra_rrtmg_variant="rrtmg_legacy"),
                {"aer_opt"}),
    "alb_sol": (dict(alb_sol=1), {"alb_sol"}),
}

#: The keys an echo at every default leaves out.
DEFAULT_OFF = frozenset({
    "thompson_version", "thompson_fork_snow_fall",
    "upper_wind_limiter_form",
    "adaptive_nest_lattice", "zadvect_implicit", "w_crit_cfl", "slope_rad",
    "topo_shading", "shadlen", "topo_wind", "gwd_opt", "sf_surface_mosaic",
    "mosaic_cat", "mosaic_urban_canopy", "diff_opt", "mix_full_fields",
    "zadvect_implicit_variant",
    # 2.8.5: post-PBL scalar diffusion, the analyzed aerosol start and the
    # CLM lake quartet, each absent at its default as in a 2.8.4 header.
    "scalar_pblmix", "use_rap_aero_icbc", "sf_lake_physics", "use_lakedepth",
    "lakedepth_default", "lake_min_elev",
    "spp_conv", "spp_pbl",
    "ruc_soilprop",
    # "measured", the launch-time derivation of every earlier header.
    "terrain_clock",
    "bl_mynn_version", "bl_mynn_gsd41_unsquared_qtke", "bl_mynn_cloud_tendency_form",
    "mynn_sfclay_variant",
    # The RUC sea-ice threshold switch, absent at 0 as in a 2.8.5 header.
    "fractional_seaice",
    # The WRF v4.6.1 sixth-order filter form and no mp_zero_out pass.
    "diff_6th_form", "diff_6th_factor2",
    "mp_zero_out", "mp_zero_out_thresh", "mp_zero_out_all",
    "ruc_irrigation", "ruc_qvg_cold_start", "ruc_2m_diagnostic",
    "ruc_snow",
    # WRF's advection orders, dropped by the echo at their Registry
    # defaults (3, 3, 5) as in a header written before they existed.
    "v_sca_adv_order", "v_mom_adv_order", "h_mom_adv_order",
    "swint_opt", "aer_opt", "rrtmg_cloud_optics_form", "rrtmg_smoke_manifest",
    "alb_sol",
})


def _echoes(cfg, monkeypatch, tmp_path, *, surface_receipt=None) -> dict[str, dict]:
    """The ``config`` each of the three writers puts in its header."""
    from tilestream import checkpoint, physics_inventory, restart_stream

    state, driver = _shim_driver_state(cfg, monkeypatch)
    if cfg.aer_opt:
        # The active legacy adapter has a real stock identity even though
        # these CPU tests never execute radiation. Bind its required setup
        # just as the restart identity fixture binds the modern adapter.
        from gpuwm.core.rrtmg_legacy import RRTMGLegacyRadiation
        setup = _identity_bound_physics_state(cfg, monkeypatch)[1].radiation_callable
        radiation = object.__new__(RRTMGLegacyRadiation)
        for name in ("start_time", "latitude_deg", "longitude_deg"):
            setattr(radiation, name, getattr(setup, name))
        driver.radiation_callable = radiation
    if surface_receipt is not None:
        state._ensemble_surface_state = surface_receipt
    if cfg.sf_surface_physics == 2:
        driver.noah_params = _identity_bound_physics_state(
            cfg, monkeypatch)[1].noah_params
    if cfg.sf_surface_mosaic:
        # The tile identity a mosaic header binds, without the tiles: this
        # file reads the configuration echo, not the tile carriers
        # (tests/test_noah_mosaic_driver.py checkpoints those on a card).
        from gpuwm.core.noah_mosaic import MosaicCategories
        driver.noah_mosaic = SimpleNamespace(
            mosaic_cat=cfg.mosaic_cat, xice_threshold=0.5,
            categories=MosaicCategories(isurban=13, isice=15, iswater=17,
                                        natural=7, lcz=()))
    _fill_setup(state)
    paths = {"resident": restart.write_restart(
        tmp_path / "resident.npz", state, cfg)}
    # After the resident write, which seeds diff_opt = 1's thermal
    # reference on the state it checkpoints.
    store = {key: value.copy() for key, value in
             physics_inventory.carrier_manifest(state).items()}
    scalars = physics_inventory.carrier_scalars(state)
    paths |= {
        "streamed": restart_stream.write_streamed_restart(
            tmp_path / "streamed.npz", store, cfg, scalars=scalars,
            setup=restart_stream.capture_domain_setup(state),
            template_state=state, check_pinned=False).path,
        "store": checkpoint.write_store_restart(
            tmp_path / "store.npz", store, scalars,
            checkpoint.DomainSetup.capture(state, cfg), cfg),
    }
    return {road: restart.read_restart_header(path)["config"]
            for road, path in paths.items()}


def test_every_writer_echoes_the_defaults_alike(monkeypatch, tmp_path):
    cfg = _cfg()
    echoes = _echoes(cfg, monkeypatch, tmp_path)
    expected = restart.configuration_echo(cfg)
    for road, echo in echoes.items():
        assert echo == expected, road
        # Every writer encodes the header with this serializer. Keep the
        # configuration member's UTF-8 bytes and key order fixed too.
        assert json.dumps(echo, allow_nan=False).encode("utf-8") == \
            json.dumps(expected, allow_nan=False).encode("utf-8"), road
    # At the defaults every default-off key is absent, as in a header
    # written before the field existed, and every other field is echoed.
    assert not DEFAULT_OFF & set(expected)
    # Output export options are not checkpointed forecast settings.
    assert "verify_visuals" not in expected
    assert set(expected) == {f.name for f in dataclasses.fields(cfg)} - DEFAULT_OFF


def test_every_writer_echoes_prescribed_source_identity_alike(monkeypatch, tmp_path):
    """Each memory road binds the same manifest and binary member hashes."""
    from test_rrtmg_smoke_manifest import _fixture
    from gpuwm.core.rrtmg_smoke_identity import SMOKE_IDENTITY_KEY

    source = tmp_path / "source"
    source.mkdir()
    path, raw = _fixture(source)
    # Configuration serialization uses the existing NumPy state shim. The
    # source mode admission is covered by test_rrtmg_smoke_source_doors.
    cfg = _cfg(nz=2, rrtmg_smoke_manifest=str(path))
    echoes = _echoes(cfg, monkeypatch, tmp_path)
    expected = restart.configuration_echo(cfg)
    for road, echo in echoes.items():
        assert json.dumps(echo).encode() == json.dumps(expected).encode(), road
        assert echo[SMOKE_IDENTITY_KEY]["frames"][0]["value"]["sha256"] == \
            raw["frames"][0]["value"]["sha256"]


def test_member_surface_provenance_does_not_change_checkpoint_config_echo_bytes(monkeypatch, tmp_path):
    cfg = _cfg()
    receipt = {"identity": {"member_id": 7, "seed": 907,
               "options": {"kind": "surface-state", "sst_offset_k": 1.0}},
               "receipt": {"realized_fp32_hex": "0000803f0000803f"}}
    expected = json.dumps(restart.configuration_echo(cfg), allow_nan=False).encode("utf-8")
    for road, echo in _echoes(cfg, monkeypatch, tmp_path, surface_receipt=receipt).items():
        assert json.dumps(echo, allow_nan=False).encode("utf-8") == expected, road
        assert not {"ensemble", "perturbation", "seed", "member_id"} & echo.keys()


@pytest.mark.parametrize("field", sorted(MOVED))
def test_every_writer_echoes_a_moved_field_alike(monkeypatch, tmp_path, field):
    overrides, kept = MOVED[field]
    cfg = _cfg(**overrides)
    echoes = _echoes(cfg, monkeypatch, tmp_path)
    expected = restart.configuration_echo(cfg)
    for road, echo in echoes.items():
        assert echo == expected, road
        assert json.dumps(echo, allow_nan=False).encode("utf-8") == \
            json.dumps(expected, allow_nan=False).encode("utf-8"), road
    assert DEFAULT_OFF & set(expected) == kept
    for key in kept:
        assert expected[key] == getattr(cfg, key), key


def test_the_digest_drops_what_the_echo_drops():
    """The configuration digest reads the echo's drops, so a default-off
    field absent from one is absent from the other."""
    for overrides, kept in [({}, set())] + list(MOVED.values()):
        cfg = _cfg(**overrides)
        digest = restart._configuration_digest_values(dataclasses.asdict(cfg))
        assert DEFAULT_OFF & set(digest) == kept, overrides


@pytest.mark.parametrize("field", (
    "bl_mynn_version", "bl_mynn_gsd41_unsquared_qtke",
    "bl_mynn_cloud_tendency_form", "swint_opt", "aer_opt", "alb_sol",
))
def test_inherited_default_off_fields_bind_both_restart_directions(field):
    """An old header may omit the default, never an active selector."""
    default = _cfg()
    legacy = restart.configuration_echo(default)
    assert field not in legacy
    restart._require_config_match(legacy, default, "legacy-default.npz")
    overrides, _kept = MOVED[field]
    changed = _cfg(**overrides)
    active = restart.configuration_echo(changed)
    assert active[field] == getattr(changed, field)
    restart._require_config_match(active, changed, "active-selector.npz")
    with pytest.raises(restart.RestartMismatchError, match=field):
        restart._require_config_match(legacy, changed, "legacy-default.npz")
    with pytest.raises(restart.RestartMismatchError, match=field):
        restart._require_config_match(active, default, "active-selector.npz")


def test_current_echo_resumes_with_omitted_fork_dycore_defaults():
    cfg = _cfg()
    echo = restart.configuration_echo(cfg)
    assert not {"diff_6th_form", "diff_6th_factor2", "upper_wind_limiter_form",
                "mp_zero_out", "mp_zero_out_thresh", "mp_zero_out_all"} & set(echo)
    restart._require_config_match(echo, cfg, "current.npz")


def test_unread_zero_out_knobs_do_not_refuse_continuation():
    stored = _cfg(mp_zero_out=0, mp_zero_out_thresh=2.0e-5, mp_zero_out_all=1)
    live = _cfg(mp_zero_out=0, mp_zero_out_thresh=1.0e-8, mp_zero_out_all=0)
    restart._require_config_match(dataclasses.asdict(stored), live, "off.npz")
    restart._require_config_match(restart.configuration_echo(stored), live, "off-echo.npz")


@pytest.mark.parametrize("name,value", [("mp_zero_out_thresh", 2.0e-5), ("mp_zero_out_all", 1)])
def test_active_zero_out_knobs_still_bind_continuation(name, value):
    stored = _cfg(mp_zero_out=1)
    live = _cfg(mp_zero_out=1, **{name: value})
    with pytest.raises(restart.RestartMismatchError, match=name):
        restart._require_config_match(restart.configuration_echo(stored), live, "on.npz")


@pytest.mark.parametrize("name", ["mp_zero_out_thresh", "mp_zero_out_all"])
def test_active_zero_out_cannot_restore_an_unrecorded_read_knob(name):
    cfg = _cfg(mp_zero_out=1)
    stored = restart.configuration_echo(cfg)
    stored.pop(name)
    with pytest.raises(restart.RestartMismatchError, match=name):
        restart._require_config_match(stored, cfg, "incomplete-on.npz")


@pytest.mark.parametrize("field", ["diff_6th_form", "mp_zero_out", "upper_wind_limiter_form"])
def test_omitted_dycore_default_still_refuses_an_active_selector(field):
    overrides, _kept = MOVED[field]
    with pytest.raises(restart.RestartMismatchError, match=field):
        restart._require_config_match(restart.configuration_echo(_cfg()), _cfg(**overrides), "default.npz")


@pytest.mark.parametrize("overrides", [{}, {"moist": True, "mp_physics": 6}])
@pytest.mark.parametrize("theme", ["paper", "woof-light", "woof-dark", "inherited"])
def test_renderer_environment_preserves_checkpoint_config_bytes(
        monkeypatch, tmp_path, overrides, theme):
    """Renderer settings must not enter any memory road's checkpoint echo."""
    cfg = _cfg(**overrides)
    before_root, after_root = tmp_path / "before", tmp_path / "after"
    before_root.mkdir()
    after_root.mkdir()
    before = _echoes(cfg, monkeypatch, before_root)
    if theme == "inherited":
        theme_file = tmp_path / "site-theme.json"
        theme_file.write_text(json.dumps({"extends": "woof-dark", "name": "site"}),
                              encoding="utf-8")
        theme = str(theme_file)
    for key, value in {
            "RUSTWX_THEME": theme, "RUSTWX_RADAR_COLORS": "nws",
            "RUSTWX_WIND_STREAMLINES": "1"}.items():
        monkeypatch.setenv(key, value)
    after = _echoes(cfg, monkeypatch, after_root)

    def payload(echo):
        return json.dumps(echo, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")

    expected = payload(restart.configuration_echo(cfg))
    assert {payload(echo) for echo in before.values()} == {expected}
    assert {payload(echo) for echo in after.values()} == {expected}


@pytest.mark.parametrize("overrides", [{}, {
    "diff_6th_form": "noaa_wrf39", "diff_6th_factor2": 0.04,
    "upper_wind_limiter_form": "noaa_wrf39", "mp_zero_out": 2,
    "mp_zero_out_all": 1, "mp_zero_out_thresh": 1.0e-8,
}])
def test_every_writer_echo_resumes_its_unchanged_dycore(
        monkeypatch, tmp_path, overrides):
    """A written echo must accept the configuration that produced it."""
    cfg = _cfg(**overrides)
    for road, echo in _echoes(cfg, monkeypatch, tmp_path).items():
        restart._require_config_match(echo, cfg, road)


@pytest.mark.parametrize("key,value", [
    ("diff_6th_form", "noaa_wrf39"),
    ("diff_6th_factor2", 0.04),
    ("upper_wind_limiter_form", "noaa_wrf39"),
    ("mp_zero_out", 2),
])
def test_omitted_dycore_default_refuses_an_enabled_or_changed_value(key, value):
    cfg = _cfg()
    echo = restart.configuration_echo(cfg)
    assert key not in echo
    restart._require_config_match(echo, cfg, "checkpoint")
    with pytest.raises(ValueError, match=key):
        restart._require_config_match(
            echo, dataclasses.replace(cfg, **{key: value}), "checkpoint")


def test_omitted_staged_dycore_defaults_restore_the_same_configuration(tmp_path):
    cfg = _cfg()
    restart._require_config_match(restart.configuration_echo(cfg), cfg, tmp_path / "state.npz")


@pytest.mark.parametrize("key,value", (
    ("diff_6th_form", "noaa_wrf39"), ("diff_6th_factor2", 0.04),
    ("mp_zero_out", 2), ("mp_zero_out_thresh", 1.0e-12),
    ("mp_zero_out_all", 1), ("upper_wind_limiter_form", "noaa_wrf39"),
))
def test_missing_staged_dycore_selector_refuses_a_changed_value(tmp_path, key, value):
    cfg = _cfg()
    header = restart.configuration_echo(cfg)
    overrides = {key: value}
    if key in ("mp_zero_out_thresh", "mp_zero_out_all"):
        overrides["mp_zero_out"] = 2
    changed = dataclasses.replace(cfg, **overrides)
    with pytest.raises(restart.RestartMismatchError, match=key):
        restart._require_config_match(header, changed, tmp_path / "state.npz")


def test_unread_zero_out_parameters_do_not_block_an_off_restart(tmp_path):
    cfg = _cfg(mp_zero_out=0, mp_zero_out_thresh=1.0e-12, mp_zero_out_all=1)
    restart._require_config_match(restart.configuration_echo(cfg), cfg, tmp_path / "state.npz")


@pytest.mark.parametrize("option,default", [("verify_visuals", True)])
def test_output_choices_preserve_every_actual_writer_echo(monkeypatch, tmp_path, option, default):
    """Output options stay outside the configuration in all three headers."""
    from gpuwm.runplan import PLAN_SCHEMA, build_plan, resolve_plan
    from test_case_data import make_case_toml

    config = make_case_toml(tmp_path)
    document = {"schema": PLAN_SCHEMA, "name": "writer-output-choice", "route": "experiment",
                "config": {"path": str(config)}, "output_root": str(tmp_path / "run")}
    ordinary = build_plan(document, source="writer-output-choice", base_dir=tmp_path, sha256="0" * 64)
    assert ordinary.run_options[option] is default
    echoes = []
    for selected in (False, True):
        plan = build_plan({**document, "run_options": {option: selected}},
                          source="writer-output-choice", base_dir=tmp_path, sha256="0" * 64)
        assert plan.run_options[option] is selected
        _, experiment, _ = resolve_plan(plan, require_inputs=False)
        folder = tmp_path / str(selected)
        folder.mkdir()
        echoes.append(_echoes(experiment.domains[0].run, monkeypatch, folder))
    before, after = echoes
    for road in ("resident", "streamed", "store"):
        assert json.dumps(before[road], allow_nan=False).encode("utf-8") == \
            json.dumps(after[road], allow_nan=False).encode("utf-8"), road
        assert option not in before[road] and option not in after[road]
