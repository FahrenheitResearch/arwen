"""Offline children use the acoustic rule on the terrain they integrate."""

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import json
import shutil
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest

from conftest import requires_netcdf_bridge
from gpuwm.acoustic_adaptation import (
    acoustic_receipt,
    adapt_experiment_to_terrain,
    readings_from_static,
    steepest_slope,
)
from gpuwm.config import RunConfig, load_config
from gpuwm.cli import main as cli_main
from gpuwm.downscale import (
    _render_child_toml, _with_parent_epssm_label, downscale_plan_path,
)
from gpuwm.io import restart
from gpuwm.offline_child import (
    OfflineChildContractError,
    OfflineChildPlacement,
    adapt_child_acoustics,
    child_epssm_is_auto,
    child_terrain_reading,
    interpolate_parent_initial_state,
    resolve_child_run_config,
)
from test_offline_child import _history
from test_restart import _sealed_tree_fixture


@dataclass(frozen=True)
class _Domain:
    grid_id: int
    run: RunConfig


@dataclass(frozen=True)
class _Experiment:
    domains: tuple
    auto_epssm: tuple = ()


def _config(tmp_path, epssm="unset"):
    cfg = RunConfig(
        nx=18, ny=18, nz=4, dx=1000.0, dy=1000.0, ztop=9000.0,
        grid_id=2, dt=5.0, run_seconds=600.0, output_interval_s=300.0,
        time_step_sound=4, epssm=0.1, hybrid_opt=2, etac=0.2, terrain_opt=1,
        moist=True, mp_physics=8, specified=True,
        spec_bdy_width=5, spec_zone=1, relax_zone=4)
    values = asdict(cfg)
    if epssm == "unset":
        del values["epssm"]
    else:
        values["epssm"] = epssm
    path = tmp_path / "child.toml"
    path.write_text(_render_child_toml(values), encoding="utf-8")
    return path, resolve_child_run_config(path)


def _parent(tmp_path):
    """One archive with a flat footprint and a separate steep footprint."""
    path = tmp_path / "parent.nc"
    _history(path, datetime(2026, 1, 1), nx=36, ny=36)
    x = np.arange(36, dtype=np.float32)
    terrain = np.broadcast_to(
        np.maximum(x - np.float32(17.0), 0.0) * np.float32(2370.0),
        (36, 36)).copy()
    with netCDF4.Dataset(path, "a") as dataset:
        dataset.DX = 3000.0
        dataset.DY = 3000.0
        dataset.variables["HGT"][0] = terrain
        dataset.variables["MAPFAC_U"][0] = np.float32(1.1)
        dataset.variables["MAPFAC_V"][0] = np.float32(1.1)
    return path, terrain


def _placement(i_start):
    return OfflineChildPlacement(
        parent_nx=36, parent_ny=36, child_nx=18, child_ny=18,
        parent_grid_ratio=3, i_parent_start=i_start, j_parent_start=10)


@pytest.mark.parametrize("epssm,auto,value", [
    ("unset", True, 0.1),
    ("auto", True, 0.1),
    ({"auto": 0.1}, True, 0.1),
    ({"auto": 0.5}, True, 0.5),
    (0.1, False, 0.1),
    (0.5, False, 0.5),
])
def test_child_loader_keeps_the_epssm_label_and_numeric_baseline(
        tmp_path, epssm, auto, value):
    path, cfg = _config(tmp_path, epssm)
    assert child_epssm_is_auto(path) is auto
    assert cfg.epssm == value


@pytest.mark.parametrize("epssm", [
    '{ auto = 0.5, extra = 1 }',
    '{ value = 0.5 }',
    '{ auto = "auto" }',
    '{ auto = "0.1" }',
    '{ auto = true }',
    '{ auto = -0.1 }',
    '{ auto = nan }',
])
def test_malformed_inherited_auto_values_are_refused(tmp_path, epssm):
    path, _ = _config(tmp_path, "unset")
    path.write_text(path.read_text(encoding="utf-8") +
                    f"epssm = {epssm}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="epssm"):
        resolve_child_run_config(path)


@pytest.mark.parametrize("epssm", ["auto", {"auto": 0.1}, {"auto": 0.5}])
def test_legacy_loader_cannot_silently_lose_the_child_auto_label(tmp_path, epssm):
    path, _ = _config(tmp_path, epssm)
    with pytest.raises(ValueError, match="epssm"):
        load_config(path)


@requires_netcdf_bridge
def test_child_reads_the_exact_terrain_and_map_factors_in_its_initial_state(
        tmp_path):
    path, cfg = _config(tmp_path)
    parent, terrain = _parent(tmp_path)
    placement = _placement(22)
    initial = interpolate_parent_initial_state(
        parent, placement, source_mp_physics=8, child_cfg=cfg)
    reading = child_terrain_reading(parent, placement, child_cfg=cfg)
    expected = steepest_slope(
        initial.fields["HGT"], cfg.dx, cfg.dy,
        msfu=initial.fields["MAPFAC_U"],
        msfv=initial.fields["MAPFAC_V"], label="d02")
    assert reading == expected
    assert reading.slope == pytest.approx(0.869, abs=0.002)
    # The separate flat footprint cannot inherit the whole parent's maximum.
    flat = child_terrain_reading(parent, _placement(6), child_cfg=cfg)
    assert flat.slope == 0.0
    assert steepest_slope(terrain, 3000.0, 3000.0).slope > 0.75


@requires_netcdf_bridge
@pytest.mark.parametrize("epssm", ["unset", "auto"])
def test_offline_child_takes_the_prepared_tree_floor_and_substeps(
        tmp_path, epssm):
    path, cfg = _config(tmp_path, epssm)
    parent, _ = _parent(tmp_path)
    placement = _placement(22)
    initial = interpolate_parent_initial_state(
        parent, placement, source_mp_physics=8, child_cfg=cfg)
    exp = _Experiment((_Domain(2, cfg),), auto_epssm=(2,))
    static = {2: {
        "HGT_M": initial.fields["HGT"],
        "MAPFAC_U": initial.fields["MAPFAC_U"],
        "MAPFAC_V": initial.fields["MAPFAC_V"],
    }}
    prepared, prepared_acoustics = adapt_experiment_to_terrain(
        exp, readings_from_static(exp, static))
    child, child_acoustic = adapt_child_acoustics(
        cfg, child_config_path=path, frame_path=parent, placement=placement)
    assert child == prepared.domains[0].run
    assert (child.epssm, child.time_step_sound) == (0.5, 6)
    assert acoustic_receipt([child_acoustic]) == acoustic_receipt(
        prepared_acoustics)


@requires_netcdf_bridge
def test_explicit_child_epssm_below_its_floor_is_refused_with_the_remedy(
        tmp_path):
    path, cfg = _config(tmp_path, 0.1)
    original = path.read_bytes()
    parent, _ = _parent(tmp_path)
    with pytest.raises(OfflineChildContractError) as caught:
        adapt_child_acoustics(
            cfg, child_config_path=path, frame_path=parent,
            placement=_placement(22))
    message = str(caught.value)
    assert message.startswith("d02's epssm 0.1 is set explicitly")
    assert "set d02's epssm to at least 0.5" in message
    assert "stopped within minutes" in message
    assert path.read_bytes() == original and cfg.epssm == 0.1


@requires_netcdf_bridge
@pytest.mark.parametrize("epssm,code", [("auto", 0), (0.1, 2)])
def test_downscale_review_applies_the_child_rule_before_publishing_a_plan(
        tmp_path, capsys, epssm, code):
    child_path, cfg = _config(tmp_path, epssm)
    parent, _ = _parent(tmp_path)
    first = tmp_path / "wrfout_d01_2026-01-01_00_00_00"
    parent.rename(first)
    second = tmp_path / "wrfout_d01_2026-01-01_00_10_00"
    shutil.copyfile(first, second)
    with netCDF4.Dataset(second, "a") as dataset:
        dataset.variables["Times"][0] = np.frombuffer(
            b"2026-01-01_00:10:00", dtype="S1")
    namelist = tmp_path / "namelist.input"
    namelist.write_text("&physics\n mp_physics = 8,\n/\n", encoding="utf-8")
    outdir = tmp_path / "child-run"
    assert cli_main([
        "downscale", str(tmp_path), "--parent-domain", "1",
        "--parent-namelist", str(namelist),
        "--child-config", str(child_path), "--ratio", "3",
        "--i-parent-start", "22", "--j-parent-start", "10",
        "--out", str(outdir), "--dry-run", "--render-products", "none",
    ]) == code
    captured = capsys.readouterr()
    plan_path = downscale_plan_path(outdir, dry_run=True)
    if code:
        assert not plan_path.exists()
        assert "set d02's epssm to at least 0.5" in captured.err
    else:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        child, acoustic = adapt_child_acoustics(
            cfg, child_config_path=child_path, frame_path=first,
            placement=_placement(22))
        assert (child.epssm, child.time_step_sound) == (0.5, 6)
        assert plan["acoustic_substeps"] == acoustic_receipt([acoustic])
    assert not outdir.exists()


@requires_netcdf_bridge
@pytest.mark.parametrize("epssm", [
    "unset", "auto", 0.1, {"auto": 0.1}, {"auto": 0.5},
])
def test_flat_child_config_and_input_bytes_are_unchanged(tmp_path, epssm):
    path, cfg = _config(tmp_path, epssm)
    parent, _ = _parent(tmp_path)
    original = path.read_bytes()
    before = json.dumps(asdict(cfg), sort_keys=True).encode("utf-8")
    child, adaptation = adapt_child_acoustics(
        cfg, child_config_path=path, frame_path=parent, placement=_placement(6))
    assert child is cfg
    assert json.dumps(asdict(child), sort_keys=True).encode("utf-8") == before
    assert path.read_bytes() == original
    assert not adaptation.adapted and not adaptation.offcentering_raised


@pytest.mark.parametrize("labels", [(), (1,), (2,)])
@pytest.mark.parametrize("parent_epssm", [0.1, 0.5])
def test_tree_checkpoint_carries_each_domains_auto_label_to_its_child(
        monkeypatch, tmp_path, labels, parent_epssm):
    model, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0, payload_seed=31,
        run_overrides={"epssm": parent_epssm})
    model._declared_experiment = SimpleNamespace(auto_epssm=labels)
    root = restart.write_tree_restart(
        tmp_path / "checkpoints", model, start + timedelta(seconds=3600))
    members = {1: root, 2: next(root.parent.glob("gpuwmrst_d02_*.npz"))}
    for gid, member in members.items():
        header = restart.read_restart_header(member)
        assert header.get(restart.AUTO_EPSSM_HEADER_KEY, False) is (
            gid in labels)
        inherited = _with_parent_epssm_label(dict(header["config"]), header)
        child_path = tmp_path / f"child-of-{gid}.toml"
        child_path.write_text(_render_child_toml(inherited), encoding="utf-8")
        cfg = resolve_child_run_config(child_path)
        assert cfg.epssm == header["config"]["epssm"] == parent_epssm
        assert child_epssm_is_auto(child_path) is (gid in labels)
        if gid not in labels:
            assert restart.AUTO_EPSSM_HEADER_KEY not in header


def test_checkpoint_writer_can_receive_the_label_from_a_single_domain_door(
        monkeypatch, tmp_path):
    model, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0, payload_seed=31)
    model._declared_experiment = SimpleNamespace(auto_epssm=(2,))
    root = restart.write_tree_restart(
        tmp_path, model, start + timedelta(seconds=3600), auto_epssm=(1,))
    assert restart.read_restart_header(root)[restart.AUTO_EPSSM_HEADER_KEY] is True
    child = next(tmp_path.glob("gpuwmrst_d02_*.npz"))
    assert restart.AUTO_EPSSM_HEADER_KEY not in restart.read_restart_header(child)


def test_legacy_parent_without_a_label_keeps_its_explicit_epssm():
    inherited = {"epssm": 0.1, "time_step_sound": 4}
    assert _with_parent_epssm_label(inherited, {"config": inherited}) is inherited
