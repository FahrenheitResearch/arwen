"""[radar_heating] in the deterministic door: the parts that run on a CPU.

The table and its rows, the execution argument, the refusals (nest, route,
resume inside the forced period, divisibility, missing window, window on
another grid), the restart classification, the strict window reader, the
grid identity from a prepared root and its agreement with the windows
tool, the slab rules and the memory pricing.  Everything that builds or
applies a tendency needs a CUDA device and lives in
``tests/test_forecast_heating_gpu.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpuwm import experiment as E  # noqa: E402
from gpuwm.config import (RADAR_HEATING_KEY_ROWS, RADAR_HEATING_OFF,  # noqa: E402
                          RadarHeatingConfig, declared_key_rows)
from gpuwm.da import forecast_heating as fh  # noqa: E402
from gpuwm.da import radar_tten  # noqa: E402

NX, NY, NZ = 12, 9, 6
START = datetime(2026, 10, 1, 18, 0, 0)


def _experiment_text(tail=""):
    return (
        "[experiment]\n"
        'name = "heat"\n'
        "start_time = 2026-10-01T18:00:00\n"
        "run_seconds = 3600.0\n"
        "restart_interval_s = 0.0\n"
        "[shared]\n"
        "nz = 30\nztop = 20000.0\n"
        "[[domain]]\n"
        "grid_id = 1\nparent_id = 0\ni_parent_start = 1\nj_parent_start = 1\n"
        "parent_grid_ratio = 1\nparent_time_step_ratio = 1\n"
        "history_interval_s = 3600.0\n"
        "nx = 40\nny = 40\ndx = 12000.0\ndy = 12000.0\ntime_step = 60\n"
        f"{tail}")


# --------------------------------------------------------------- the table

def test_the_table_parses_with_its_declared_defaults():
    config = RadarHeatingConfig.from_mapping({"windows": "/w"}, source="t")
    assert config.enabled
    assert (config.window_minutes, config.active_minutes) == (15.0, 60.0)
    assert config.latent_heat_period_min == 20.0
    assert config.mp_tend_lim == 0.07
    assert not config.strict_suppression and not config.pbl_extension
    assert config.window_count == 4
    assert config.window_end_minutes() == (15.0, 30.0, 45.0, 60.0)
    nowcast = RadarHeatingConfig.from_mapping(
        {"windows": "/w", "window_minutes": 10, "active_minutes": 120}, source="t")
    assert nowcast.window_count == 12
    assert nowcast.window_end_minutes()[-1] == 120.0
    assert not RADAR_HEATING_OFF.enabled
    assert RadarHeatingConfig.from_mapping(None, source="t") is RADAR_HEATING_OFF
    rows = declared_key_rows()["radar_heating"]
    assert set(rows) == set(RADAR_HEATING_KEY_ROWS) == {
        "windows", "window_minutes", "active_minutes", "latent_heat_period_min",
        "strict_suppression", "pbl_extension", "mp_tend_lim"}
    assert rows["windows"]["required"] is True
    assert rows["mp_tend_lim"]["default"] == 0.07


@pytest.mark.parametrize("table,phrase", [
    ({"window_minutes": 10}, "names no windows"),
    ({"windows": "/w", "windows_root": "/x"}, "unknown key"),
    ({"windows": "/w", "window_minutes": 15, "active_minutes": 50}, "does not divide"),
    ({"windows": "/w", "window_minutes": 0}, "finite positive"),
    ({"windows": "/w", "mp_tend_lim": -1}, "finite positive"),
    ({"windows": "/w", "strict_suppression": "yes"}, "true or false"),
    ({"windows": ""}, "empty"),
])
def test_the_table_refuses_in_its_own_words(table, phrase):
    with pytest.raises(ValueError) as refusal:
        RadarHeatingConfig.from_mapping(table, source="t")
    assert phrase in str(refusal.value)


def test_the_experiment_carries_the_table_and_an_absent_one_changes_nothing(tmp_path):
    from gpuwm.core.model import restart_identity_payload

    plain = E.build_experiment(tomllib.loads(_experiment_text()), source="x.toml")
    assert plain.radar_heating is RADAR_HEATING_OFF
    assert "radar_heating" not in E.experiment_config_document(plain)
    assert "radar_heating" not in restart_identity_payload(plain)

    config = tmp_path / "case.toml"
    config.write_text(_experiment_text(
        '[radar_heating]\nwindows = "windows/stormscope"\n'
        "window_minutes = 10\nactive_minutes = 120\n"), encoding="utf-8")
    heated = E.build_experiment(tomllib.loads(config.read_text()), source=str(config))
    assert heated.radar_heating.enabled
    # A relative root is the config file's directory, not the shell's.
    assert Path(heated.radar_heating.windows) == tmp_path / "windows" / "stormscope"
    document = E.experiment_config_document(heated)
    assert document["radar_heating"]["window_minutes"] == 10.0
    # The identity binds the heating but not where the files are.
    payload = restart_identity_payload(heated)
    assert payload["radar_heating"]["active_minutes"] == 120.0
    assert "windows" not in payload["radar_heating"]
    moved = E.build_experiment(tomllib.loads(_experiment_text(
        '[radar_heating]\nwindows = "/elsewhere"\n'
        "window_minutes = 10\nactive_minutes = 120\n")), source="x.toml")
    assert restart_identity_payload(moved) == payload


def test_the_execution_argument_round_trips_and_overlays(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = fh.execution_argument(
        '{"windows": "w", "window_minutes": 10, "active_minutes": 120}')
    assert Path(config.windows) == (tmp_path / "w").resolve()
    flags = fh.execution_flags(config)
    assert flags[0] == "--radar-heating-table"
    assert fh.execution_argument(flags[1]) == config
    assert fh.execution_flags(None) == []
    with pytest.raises(ValueError, match="JSON object"):
        fh.execution_argument("{not json")
    plain = E.build_experiment(tomllib.loads(_experiment_text()), source="x.toml")
    assert fh.apply_execution_options(plain, None) is plain
    heated = fh.apply_execution_options(plain, flags[1])
    assert heated.radar_heating == config
    parser = argparse.ArgumentParser()
    fh.add_execution_argument(parser)
    assert parser.parse_args(flags).radar_heating_table == config
    with pytest.raises(SystemExit):
        parser.parse_args(["--radar-heating-table", '{"windows": "w", "active_minutes": 50}'])


def test_the_door_parser_takes_the_table():
    from gpuwm import prepared_single_domain_forecast as door

    parser = door.build_parser()
    option = {a.dest: a for a in parser._actions}["radar_heating_table"]
    assert option.option_strings == ["--radar-heating-table"]


# ------------------------------------------------------------- refusals

def test_a_tree_run_is_refused():
    fh.refuse_tree(1, "door")
    with pytest.raises(fh.ForecastHeatingRefused, match="2 domains"):
        fh.refuse_tree(2, "door")


def test_a_route_that_does_not_attach_the_table_is_refused():
    heated = SimpleNamespace(radar_heating=RadarHeatingConfig(windows="/w"))
    plain = SimpleNamespace(radar_heating=RADAR_HEATING_OFF)
    model = SimpleNamespace(_declared_experiment=heated)
    with pytest.raises(fh.ForecastHeatingRefused, match="does not attach"):
        fh.require_routed(model, None)
    with pytest.raises(fh.ForecastHeatingRefused):
        fh.require_routed(SimpleNamespace(), heated)
    fh.require_routed(SimpleNamespace(), plain)
    fh.require_routed(SimpleNamespace(), None)
    setattr(model, fh.ROUTE_ATTRIBUTE, object())
    fh.require_routed(model, heated)


def test_a_resume_inside_the_forced_period_is_refused():
    heating = RadarHeatingConfig(windows="/w", window_minutes=10.0, active_minutes=120.0)
    assert fh.resume_plan(heating, None) == {"attach": True, "status": "FORCED"}
    with pytest.raises(fh.ForecastHeatingRefused, match="inside the forced period"):
        fh.resume_plan(heating, 70 * 60.0)
    after = fh.resume_plan(heating, 120 * 60.0)
    assert after["attach"] is False and after["status"] == "RESUMED_AFTER_FORCED_PERIOD"


def test_the_forcing_is_restart_infrastructure():
    from gpuwm.io.restart import classify_state_attr

    assert classify_state_attr("radar_tten_forcing") == "infra"
    assert classify_state_attr(radar_tten.STATE_ATTRIBUTE) == "infra"
    assert classify_state_attr(radar_tten.SLAB_ATTRIBUTE) == "infra"


def test_slab_rules_hold_on_the_host():
    assert radar_tten.slab_extent(None, (4, 10, 20)) is None
    assert radar_tten.slab_extent((5, 0, 30, 20), (4, 10, 20)) == (5, 0, 30, 20)
    with pytest.raises(radar_tten.RadarTtenError, match="does not lie inside"):
        radar_tten.slab_extent((25, 0, 30, 20), (4, 10, 20))
    assert radar_tten.slab_refusal(None, None) is None
    assert "whole-domain" in radar_tten.slab_refusal(None, (0, 0))
    assert "not a resident rank slab" in radar_tten.slab_refusal((5, 0, 30, 20), None)
    assert "another" in radar_tten.slab_refusal((5, 0, 30, 20), (0, 0))
    assert radar_tten.slab_refusal((5, 0, 30, 20), (5, 0)) is None
    # The apply kernel tests the ring in the domain's indices.
    assert "const int j0, const int i0, const int gny, const int gnx" in radar_tten._SOURCE
    assert "i >= gnx - ring" in radar_tten._SOURCE


def test_memory_is_priced_per_card_and_the_build_card_pays_the_build():
    heating = RadarHeatingConfig(windows="/w", window_minutes=10.0, active_minutes=120.0)
    cfg = SimpleNamespace(nz=50, ny=100, nx=200)
    specs = [SimpleNamespace(cny=100, cnx=120), SimpleNamespace(cny=100, cnx=120)]
    priced = fh.slab_bytes_per_card(heating, cfg, specs, [3, 5])
    slab = 13 * 4 * 50 * 100 * 120
    build = fh.BUILD_TRANSIENT_VOLUMES * 4 * 50 * 100 * 200
    assert priced == {3: slab + build, 5: slab}
    assert fh.resident_bytes(heating, cfg) == (13 + fh.BUILD_TRANSIENT_VOLUMES) * 4 * 50 * 100 * 200


# ------------------------------------------------- the prepared grid identity

def _array_entry(directory, key, array):
    from gpuwm.ingest.prepared_cache import _array_sha256

    name = key.replace("/", "__") + ".npy"
    np.save(directory / name, array)
    return {"file": name, "shape": list(array.shape), "dtype": str(array.dtype),
            "nbytes": int(array.nbytes), "sha256": _array_sha256(array)}


def _prepared_root(tmp_path, *, nz=NZ, ny=NY, nx=NX, native=True):
    root = tmp_path / "prepared"
    cache = root / ("native/prepared-cache" if native else "prepared-cache")
    cache.mkdir(parents=True)
    terrain = 300.0 + 5.0 * np.arange(nx, dtype=np.float64)[None, :] + np.zeros((ny, nx))
    eta = np.linspace(0.0, 15000.0, nz + 1)
    phb = ((terrain[None] + eta[:, None, None]) * fh.STANDARD_GRAVITY).astype(np.float32)
    php = np.full((nz + 1, ny, nx), 3.0, dtype=np.float32)
    arrays = {key: _array_entry(cache, key, value) for key, value in (
        ("state/php", php), ("base/phb", phb), ("base/terrain_z", terrain))}
    (cache / "header.json").write_text(json.dumps({
        "schema": "x", "status": "READY", "arrays": arrays}), encoding="utf-8")
    geometry = {"mass_shape": [ny, nx], "nz": nz, "dx_m": 3000.0, "dy_m": 3000.0,
                "map_proj": "lambert", "ref_lat": 35.3, "ref_lon": -97.3,
                "known_x": (nx + 1) / 2.0, "known_y": (ny + 1) / 2.0,
                "truelat1": 33.0, "truelat2": 37.0, "stand_lon": -97.3,
                "moad_cen_lat": 35.3, "moad_cen_lon": -97.3}
    name = "native-geometry-receipt.json" if native else "geometry-receipt.json"
    (root / name).write_text(json.dumps({"geometry": geometry}), encoding="utf-8")
    return root


def test_the_grid_identity_comes_from_the_prepared_root(tmp_path):
    from gpuwm.obs.target_grid import TargetGrid
    from gpuwm.static.projection import projection_class

    root = _prepared_root(tmp_path)
    grid = fh.grid_for_prepared(root)
    assert (grid.nz, grid.ny, grid.nx) == (NZ, NY, NX)
    surface = 300.0 + 5.0 * np.arange(NX)[None, :] + 3.0 / fh.STANDARD_GRAVITY
    np.testing.assert_allclose(grid.z_w[0], np.broadcast_to(surface, (NY, NX)),
                               rtol=1e-5)
    projection = projection_class("lambert")(
        ref_lat=35.3, ref_lon=-97.3, truelat1=33.0, truelat2=37.0,
        stand_lon=-97.3, dx=3000.0, dy=3000.0, e_we=NX + 1, e_sn=NY + 1)
    direct = TargetGrid.from_projection(projection, z_w=grid.z_w,
                                        terrain_m=grid.terrain_m, name="prepared-d01")
    assert grid.identity_sha256() == direct.identity_sha256()
    assert fh.prepared_grid_identity(root) == grid.identity_sha256()
    # The portable layout reads the same way.
    other = _prepared_root(tmp_path / "portable", native=False)
    assert fh.grid_for_prepared(other).identity_sha256() == grid.identity_sha256()
    with pytest.raises(fh.ForecastHeatingRefused, match="no geometry receipt"):
        fh.grid_for_prepared(tmp_path / "nothing")
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "geometry-receipt.json").write_text(
        (root / "native-geometry-receipt.json").read_text(), encoding="utf-8")
    with pytest.raises(fh.ForecastHeatingRefused, match="no prepared cache"):
        fh.grid_for_prepared(bare)


def test_the_windows_tool_and_the_door_agree_on_the_grid(tmp_path):
    from gpuwm.obs.radar_tten_grid import write_grid_descriptor
    from tools import radar_tten_windows

    root = _prepared_root(tmp_path)
    args = radar_tten_windows.build_parser().parse_args(
        ["--grid-prepared", str(root), "--out", str(tmp_path / "w"),
         "--windows", "2026-10-01T18:00:00Z"])
    grid, source = radar_tten_windows.target_grid(args)
    assert source == {"prepared_root": str(root.resolve())}
    descriptor = write_grid_descriptor(grid, tmp_path / "w" / "grid" / "grid.json")
    assert descriptor["identity_sha256"] == fh.prepared_grid_identity(root)
    with pytest.raises(SystemExit):
        radar_tten_windows.build_parser().parse_args(["--out", "x"])


# ------------------------------------------------------- the strict reader

def _window(root, end, identity, *, shape=(NZ, NY, NX), lead=None, stated_end=None):
    directory = root / end.strftime("%Y%m%dT%H%MZ")
    directory.mkdir(parents=True, exist_ok=True)
    field = np.full(shape, -99999.0, dtype="<f4")
    field[2, 4, 5] = 35.0
    field[0, 1, 1] = -99.0
    field.tofile(directory / "ref.f32")
    raw = (directory / "ref.f32").read_bytes()
    receipt = {
        "schema": "gpuwm-obs.radar-tten-ref.v1", "status": "READY",
        "shape": list(shape),
        "data": {"file": "ref.f32", "bytes": len(raw),
                 "sha256": hashlib.sha256(raw).hexdigest()},
        "grid": {"identity_sha256": identity},
        "window": {"end": (stated_end or end).strftime("%Y-%m-%dT%H:%M:%SZ")},
        "counts": {"echo_cells": 1},
    }
    if lead is not None:
        receipt["source"] = {"lead_class": lead, "id": "synthetic"}
    (directory / "ref.json").write_text(json.dumps(receipt), encoding="utf-8")
    return directory / "ref.f32"


def test_the_strict_reader_refuses_another_grid_and_a_missing_identity(tmp_path):
    path = _window(tmp_path, START, "a" * 64)
    host, receipt = radar_tten.read_window_host(path, identity_sha256="a" * 64)
    assert host.shape == (NZ, NY, NX) and host.dtype == np.float32
    assert host[2, 4, 5] == 35.0
    with pytest.raises(radar_tten.RadarTtenError, match="not the required grid"):
        radar_tten.read_window_host(path, identity_sha256="b" * 64)
    with pytest.raises(TypeError):
        radar_tten.read_window_host(path)
    with pytest.raises(radar_tten.RadarTtenError, match="grid identity"):
        radar_tten.read_window_host(path, identity_sha256="")
    with pytest.raises(TypeError):
        radar_tten.read_window_reflectivity(path)
    # Altered bytes are refused by digest.
    data = bytearray(path.read_bytes())
    data[0] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(radar_tten.RadarTtenError, match="sha256"):
        radar_tten.read_window_host(path, identity_sha256="a" * 64)


def test_windows_are_found_held_to_their_end_and_labelled(tmp_path):
    from datetime import timedelta

    heating = RadarHeatingConfig(windows=str(tmp_path / "w"), window_minutes=10.0,
                                 active_minutes=30.0)
    identity = "c" * 64
    for n, lead in ((1, "forecast"), (2, "forecast"), (3, "oracle")):
        _window(tmp_path / "w", START + timedelta(minutes=10 * n), identity, lead=lead)
    rows = fh.read_windows(heating, START, identity)
    assert [row["end"] for row in rows] == [
        "2026-10-01T18:10:00Z", "2026-10-01T18:20:00Z", "2026-10-01T18:30:00Z"]
    assert [row["lead_class"] for row in rows] == ["forecast", "forecast", "oracle"]
    # A Level II window states no lead class: it is observed.
    assert fh.lead_class({"window": {"end": "x"}}) == "observed"
    # Another grid: refused.
    with pytest.raises(radar_tten.RadarTtenError, match="not the required grid"):
        fh.read_windows(heating, START, "d" * 64)
    # A missing window: refused, naming it.
    (tmp_path / "w" / "20261001T1830Z" / "ref.f32").unlink()
    with pytest.raises(radar_tten.RadarTtenError, match="missing"):
        fh.read_windows(heating, START, identity)
    # A window whose receipt states another end than its directory.
    _window(tmp_path / "w", START + timedelta(minutes=30), identity,
            stated_end=START + timedelta(minutes=45))
    with pytest.raises(fh.ForecastHeatingRefused, match="receipt says it ends"):
        fh.read_windows(heating, START, identity)
    # A window whose receipt states no end time at all.
    path = _window(tmp_path / "w", START + timedelta(minutes=30), identity)
    receipt = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    del receipt["window"]
    path.with_suffix(".json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(fh.ForecastHeatingRefused, match="states no end time"):
        fh.read_windows(heating, START, identity)


def test_the_hook_validates_on_the_host_before_any_card(tmp_path):
    from datetime import timedelta

    root = _prepared_root(tmp_path)
    identity = fh.prepared_grid_identity(root)
    heating = RadarHeatingConfig(windows=str(tmp_path / "w"), window_minutes=15.0,
                                 active_minutes=30.0)
    for n in (1, 2):
        _window(tmp_path / "w", START + timedelta(minutes=15 * n), identity)
    hook = fh.ForecastHeating(heating, start_time=START, prepared_root=root)
    hook.validate(SimpleNamespace(nz=NZ, ny=NY, nx=NX))
    assert hook.identity == identity and len(hook.windows) == 2
    receipt = hook.receipt()
    assert receipt["schema"] == fh.SCHEMA and receipt["status"] == "FORCED"
    assert receipt["lead_classes"] == ["observed"] and receipt["oracle"] is False
    with pytest.raises(fh.ForecastHeatingRefused, match="the run is"):
        fh.ForecastHeating(heating, start_time=START, prepared_root=root).validate(
            SimpleNamespace(nz=NZ, ny=NY + 1, nx=NX))
    # A resume after the forced period marks the route and attaches nothing.
    model = SimpleNamespace()
    after = fh.ForecastHeating(heating, start_time=START, prepared_root=root,
                               restored_seconds=1800.0)
    after.attach(model, node=None, steppers=None)
    assert getattr(model, fh.ROUTE_ATTRIBUTE) is after
    assert after.receipt()["status"] == "RESUMED_AFTER_FORCED_PERIOD"
    after.require_applied()
    # Off: the door's call returns nothing and marks nothing.
    plain = SimpleNamespace(radar_heating=RADAR_HEATING_OFF)
    assert fh.attach_forecast_heating(SimpleNamespace(), None, plain,
                                      prepared_root=root) is None
    assert fh.prevalidate(plain, prepared_root=root) is None
    # The door's host-only check before any card: the same identity and
    # windows, and the same refusals, as the attach's own validation.
    exp = SimpleNamespace(radar_heating=heating, start_time=START,
                          root=SimpleNamespace(run=SimpleNamespace(nz=NZ, ny=NY, nx=NX)))
    checked = fh.prevalidate(exp, prepared_root=root)
    assert checked[0] == identity and checked[1] == hook.windows
    with pytest.raises(fh.ForecastHeatingRefused, match="the run is"):
        fh.prevalidate(SimpleNamespace(
            radar_heating=heating, start_time=START,
            root=SimpleNamespace(run=SimpleNamespace(nz=NZ + 1, ny=NY, nx=NX))),
            prepared_root=root)
    (tmp_path / "w" / "20261001T1830Z" / "ref.f32").unlink()
    with pytest.raises(radar_tten.RadarTtenError, match="missing"):
        fh.prevalidate(exp, prepared_root=root)


# ------------------------------------------- the table inside a bound config

def test_a_config_carrying_the_table_splits_back_into_the_prepared_bytes(tmp_path):
    prepared = _experiment_text().rstrip("\n").replace("\n", "\r\n") + "\r\n"
    (tmp_path / "prep.toml").write_bytes(prepared.encode("utf-8"))
    arm = (prepared.rstrip("\r\n") + "\r\n"
           + '\n[radar_heating]\nwindows = "stormscope"\nwindow_minutes = 10\n'
             'active_minutes = 120\n')
    (tmp_path / "arm.toml").write_bytes(arm.encode("utf-8"))
    bound, heating = fh.detach_table_from_config(tmp_path / "arm.toml")
    assert bound.read_bytes() == prepared.encode("utf-8")
    # Beside the given config, so the paths it states relative to its own
    # directory resolve to the same files.
    assert bound.parent == tmp_path and bound.name.startswith("arm.bound-")
    assert heating.window_count == 12
    assert Path(heating.windows) == tmp_path / "stormscope"
    # Splitting again reuses the same copy.
    again, _ = fh.detach_table_from_config(tmp_path / "arm.toml")
    assert again == bound
    # A preparation made from the config WITH the table keeps it whole:
    # cutting it would break the binding (gpuwm go's staged route).
    prep_root = tmp_path / "prepared"
    prep_root.mkdir()
    arm_sha = hashlib.sha256(arm.encode("utf-8")).hexdigest()
    (prep_root / "proof.json").write_text(json.dumps(
        {"execution_inputs": {"experiment_config": {"sha256": arm_sha}}}),
        encoding="utf-8")
    whole, kept = fh.detach_table_from_config(tmp_path / "arm.toml",
                                              prepared_root=prep_root)
    assert whole == tmp_path / "arm.toml" and kept == heating
    # A preparation that bound other bytes gets the split copy.
    other_root = tmp_path / "other"
    other_root.mkdir()
    (other_root / "source-input-manifest.json").write_text("{}", encoding="utf-8")
    split, _ = fh.detach_table_from_config(tmp_path / "arm.toml",
                                           prepared_root=other_root)
    assert split == bound
    # A config without the table passes through untouched.
    same, none = fh.detach_table_from_config(tmp_path / "prep.toml")
    assert same == tmp_path / "prep.toml" and none is None
    assert not list(tmp_path.glob("prep.bound-*"))
    # A sub-table the cut would leave behind is refused, not half-split.
    (tmp_path / "odd.toml").write_text(
        _experiment_text() + '[radar_heating]\nwindows = "w"\n[radar_heating.extra]\nx = 1\n',
        encoding="utf-8")
    with pytest.raises(fh.ForecastHeatingRefused, match="changed another part"):
        fh.detach_table_from_config(tmp_path / "odd.toml")
    # A table in the middle of the file is cut cleanly too.
    middle = _experiment_text().replace(
        "[shared]", '[radar_heating]\nwindows = "w"\n\n[shared]', 1)
    base, table = fh.split_experiment_text(middle)
    assert table == '[radar_heating]\nwindows = "w"\n\n'
    assert tomllib.loads(base) == tomllib.loads(_experiment_text())
