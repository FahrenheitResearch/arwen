"""New physics controls preserve default restores and bind both restart halves."""
from dataclasses import replace

import numpy as np

import pytest

from gpuwm.io import restart
from test_radiation_driver_identity import OPTIONS, _experiment
from test_checkpoint_config_echo import _checkpoint_state
from test_restart import _cfg, _fill_serialized


NEW_CONTROLS = {
    "thompson_version": ({}, "wrf_39_noaa"),
    "thompson_fork_snow_fall": ({"thompson_version": "wrf_39_noaa"}, "wrf_39_noaa"),
    "bl_mynn_version": ({}, "gsd_41"),
    "bl_mynn_gsd41_unsquared_qtke": ({"bl_mynn_version": "gsd_41"}, True),
    "bl_mynn_cloud_tendency_form": ({"bl_mynn_version": "gsd_41"}, "gsd_41"),
    "swint_opt": ({}, 1),
    "aer_opt": ({}, 3),
    "alb_sol": ({}, 1),
    "rrtmg_cloud_optics_form": ({}, "noaa_wrf39"),
    "rrtmg_smoke_manifest": ({"nx": 4, "ny": 3, "nz": 2, "aer_opt": 3}, None),
}


def _control_config(base):
    return _cfg(moist=True, mp_physics=28, bl_pbl_physics=5,
                sf_sfclay_physics=5, ra_physics=4,
                ra_rrtmg_variant="rrtmg_legacy", **base)


def _words(state):
    result = {name: value.tobytes() for name, value in vars(state).items()
              if isinstance(value, np.ndarray)}
    for name, value in getattr(state, "_scratch", {}).items():
        if isinstance(value, np.ndarray):
            result["scratch/" + name] = value.tobytes()
    driver = getattr(state, "physics", None)
    for name, value in getattr(driver, "fields", {}).items():
        if isinstance(value, np.ndarray):
            result["physics/" + name] = value.tobytes()
    radiation = getattr(driver, "radiation_callable", None)
    if radiation is not None:
        for name, value in vars(radiation).items():
            if isinstance(value, np.ndarray):
                result["radiation/" + name] = value.tobytes()
    return result


def _chosen_control(name, tmp_path):
    base, chosen = NEW_CONTROLS[name]
    if name == "rrtmg_smoke_manifest":
        from test_rrtmg_smoke_manifest import _fixture
        directory = tmp_path / "prescribed-source"
        directory.mkdir()
        chosen = str(_fixture(directory)[0])
    return base, chosen


def _bound_control_state(cfg, monkeypatch):
    """Bind the stock driver and real checked source metadata, without stepping."""
    state, driver = _checkpoint_state(cfg, monkeypatch)
    if cfg.rrtmg_smoke_manifest:
        from gpuwm.core.rrtmg_smoke_manifest import BoundSmokeManifest, validate_smoke_manifest
        from test_rrtmg_smoke_manifest import START
        latitude = np.ones((cfg.ny, cfg.nx), dtype=np.float32)
        longitude = np.ones_like(latitude)
        # The checkpoint requires checked source/vertical identity. Local
        # CUDA geometry and numerical forcing belong to the separate GPU
        # manifest suite, so this CPU owner does not bind device geometry.
        source = object.__new__(BoundSmokeManifest)
        source._manifest = validate_smoke_manifest(cfg.rrtmg_smoke_manifest, START, cfg.nz)
        source._vertical_verified = False
        source._geometry_bound = False
        source.require_vertical((1.0, .5, 0.0), 2, .2, 1500.0)
        radiation = driver.radiation_callable
        radiation.start_time = START
        radiation.latitude_deg = latitude
        radiation.longitude_deg = longitude
        radiation._smoke_provider = source
        assert source.identity() == restart.configuration_echo(cfg)["rrtmg_smoke_manifest_identity"]
    return state, driver


@pytest.mark.parametrize("name", NEW_CONTROLS)
@pytest.mark.parametrize("active", [False, True], ids=["default", "active"])
def test_new_control_matching_checkpoint_restores_actual_words(
        name, active, monkeypatch, tmp_path):
    base, chosen = _chosen_control(name, tmp_path)
    cfg = _control_config(base)
    if active:
        cfg = replace(cfg, **{name: chosen})
    source, _ = _bound_control_state(cfg, monkeypatch)
    _fill_serialized(source, seed=20261004)
    expected = {name: getattr(source, name).tobytes()
                for name in restart.serialized_state_attrs(source)
                if getattr(source, name, None) is not None}
    path = restart.write_restart(tmp_path / "matching.npz", source, cfg)
    header = restart.read_restart_header(path)
    if active:
        assert header["config"][name] == chosen
        if name == "rrtmg_smoke_manifest":
            assert header["config"]["rrtmg_smoke_manifest_identity"] == \
                restart.configuration_echo(cfg)["rrtmg_smoke_manifest_identity"]
    else:
        assert name not in header["config"]
    fresh, _ = _bound_control_state(cfg, monkeypatch)
    _fill_serialized(fresh, seed=20261005)
    restart.restore_restart(path, fresh, cfg)
    for field, words in expected.items():
        assert getattr(fresh, field).tobytes() == words, field


@pytest.mark.parametrize("name", NEW_CONTROLS)
@pytest.mark.parametrize("stored_active", [False, True],
                         ids=["default-to-active", "active-to-default"])
def test_new_control_change_refuses_before_destination_mutation(
        name, stored_active, monkeypatch, tmp_path):
    base, chosen = _chosen_control(name, tmp_path)
    default = _control_config(base)
    active = replace(default, **{name: chosen})
    stored, live = (active, default) if stored_active else (default, active)
    source, _ = _bound_control_state(stored, monkeypatch)
    _fill_serialized(source, seed=20261004)
    path = restart.write_restart(tmp_path / "changed.npz", source, stored)
    destination, _ = _bound_control_state(live, monkeypatch)
    _fill_serialized(destination, seed=20261005)
    before = _words(destination)
    with pytest.raises(restart.RestartMismatchError, match=name):
        restart.restore_restart(path, destination, live)
    assert _words(destination) == before


@pytest.mark.parametrize("stored_changed", [False, True],
                         ids=["original-to-changed-source", "changed-to-original-source"])
def test_same_path_smoke_source_change_refuses_actual_restore_before_mutation(
        stored_changed, monkeypatch, tmp_path):
    from pathlib import Path
    import json
    from test_rrtmg_smoke_source_doors import _replace_member
    base, chosen = _chosen_control("rrtmg_smoke_manifest", tmp_path)
    path = Path(chosen)
    originals = {member: member.read_bytes() for member in path.parent.iterdir()}
    if stored_changed:
        _replace_member(path, json.loads(path.read_text()))
    cfg = replace(_control_config(base), rrtmg_smoke_manifest=chosen)
    source, _ = _bound_control_state(cfg, monkeypatch)
    _fill_serialized(source, seed=20261004)
    checkpoint = restart.write_restart(tmp_path / "same-path.npz", source, cfg)
    stored_identity = restart.read_restart_header(checkpoint)["config"]["rrtmg_smoke_manifest_identity"]
    if stored_changed:
        for member, payload in originals.items():
            member.write_bytes(payload)
    else:
        _replace_member(path, json.loads(path.read_text()))
    assert restart.configuration_echo(cfg)["rrtmg_smoke_manifest_identity"] != stored_identity
    destination, _ = _bound_control_state(cfg, monkeypatch)
    _fill_serialized(destination, seed=20261005)
    before = _words(destination)
    with pytest.raises(restart.RestartMismatchError, match="rrtmg_smoke_manifest_identity"):
        restart.restore_restart(checkpoint, destination, cfg)
    assert _words(destination) == before


@pytest.mark.parametrize("name,value", OPTIONS.items())
def test_default_radiation_echo_restores_and_old_header_refuses_active_option(name, value):
    default = _experiment(**{name: 0}).domains[0].run
    written = restart.configuration_echo(default)
    assert name not in written
    restart._require_config_match(written, default, "default-checkpoint.npz")
    active = replace(default, **{name: value})
    with pytest.raises(restart.RestartMismatchError, match=name):
        restart._require_config_match(written, active, "default-checkpoint.npz")
