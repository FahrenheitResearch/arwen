"""CPU-side contracts for ``gpuwm downscale`` and its runner plumbing.

The GPU forecast itself is exercised by the acceptance run; these tests
pin the file-facing half: parent-series discovery, point placement, the
derived child config, the child-grid surface source contract, and the
Davies-bind boundary clock the standalone child constructs.
"""

from argparse import Namespace
from datetime import datetime, timedelta
import math

import netCDF4
import numpy as np
import pytest

from gpuwm.cli import main as cli_main
from gpuwm.config import load_config
from gpuwm.downscale import (
    _centered_placement,
    _derive_child_run_config,
    _discover_parent_series,
    _fit_child_size,
    _nearest_parent_index,
    _parse_point,
    _render_child_toml,
)
from gpuwm.offline_child import (
    OfflineChildContractError,
    read_child_surface_state,
)
from gpuwm.offline_child_run import _child_boundary_clock
from test_offline_child import _history


#: A parent RunConfig dict in restart-evidence shape (physics inherited
#: verbatim by the derivation; geometry keys rescaled).  Values follow a
#: two-moment mp8 + YSU + Noah + MM5 surface selection purely as data.
_PARENT_CONFIG = {
    "nx": 20, "ny": 18, "nz": 4, "dx": 1000.0, "dy": 1000.0,
    "ztop": 9000.0, "dt": 5.0, "run_seconds": 21600.0,
    "output_interval_s": 3600.0, "hybrid_opt": 2, "etac": 0.2,
    "hypsometric_opt": 2, "moist": True, "mp_physics": 8,
    "specified": False, "nested": True, "terrain_opt": 1, "map_proj": 1,
    "grid_id": 3, "time_step_sound": 4, "spec_bdy_width": 5,
    "spec_zone": 1, "relax_zone": 4, "not_a_runconfig_key": "dropped",
}


def test_parse_point_bounds():
    assert _parse_point("39.5,-84.0") == (39.5, -84.0)
    with pytest.raises(ValueError, match="LAT,LON"):
        _parse_point("39.5")
    with pytest.raises(ValueError, match="bounds"):
        _parse_point("95.0,-84.0")


def test_discover_parent_series_directory_domains(tmp_path):
    for stamp in ("13_00_00", "12_00_00"):
        for dom in ("01", "03"):
            (tmp_path / f"wrfout_d{dom}_1974-04-03_{stamp}").write_bytes(b"x")
    (tmp_path / "gpuwmrst_d03_x.npz").write_bytes(b"x")
    with pytest.raises(OfflineChildContractError, match="multiple domains"):
        _discover_parent_series([tmp_path], None)
    frames = _discover_parent_series([tmp_path], 3)
    assert [p.name for p in frames] == [
        "wrfout_d03_1974-04-03_12_00_00", "wrfout_d03_1974-04-03_13_00_00"]
    with pytest.raises(OfflineChildContractError, match="no domain-04"):
        _discover_parent_series([tmp_path], 4)


def test_nearest_parent_index_and_centered_placement():
    lat = np.linspace(38.0, 41.0, 31)[:, None] * np.ones((1, 41))
    lon = np.ones((31, 1)) * np.linspace(-86.0, -82.0, 41)[None, :]
    j0, i0 = _nearest_parent_index(lat, lon, 39.5, -84.0)
    assert (j0, i0) == (15, 20)
    parent = {"nx": 41, "ny": 31}
    placement = _centered_placement(
        parent, j0=j0, i0=i0, ratio=3, child_nx=36, child_ny=24)
    # span 12x8 centered on the 1-based point (21, 16).
    assert placement.i_parent_start == 16
    assert placement.j_parent_start == 13
    with pytest.raises(OfflineChildContractError, match="multiple"):
        _centered_placement(parent, j0=j0, i0=i0, ratio=3,
                            child_nx=35, child_ny=24)


def test_derive_child_config_inherits_physics_and_rescales(tmp_path):
    parent = {"nx": 20, "ny": 18, "dx": 1000.0, "dy": 1000.0}
    merged = _derive_child_run_config(
        _PARENT_CONFIG, parent=parent, ratio=2, child_nx=12, child_ny=10,
        run_seconds=600.0, output_interval_s=300.0)
    assert merged["dx"] == 500.0 and merged["dy"] == 500.0
    assert merged["dt"] == 2.5
    assert merged["grid_id"] == 4
    assert merged["specified"] is True and merged["nested"] is False
    assert merged["mp_physics"] == 8          # inherited verbatim
    assert "not_a_runconfig_key" not in merged
    text = _render_child_toml(merged)
    path = tmp_path / "child.toml"
    path.write_text(text, encoding="utf-8")
    cfg = load_config(path)
    assert (cfg.nx, cfg.ny, cfg.dx, cfg.dt) == (12, 10, 500.0, 2.5)
    assert cfg.specified and not cfg.nested


def test_fit_child_size_returns_units_that_fit():
    parent = {"nx": 501, "ny": 501, "dx": 1000.0, "dy": 1000.0}
    config = dict(_PARENT_CONFIG, nx=501, ny=501, nz=49)
    size = _fit_child_size(
        parent, config, j0=250, i0=250, ratio=2, run_seconds=3600.0,
        output_interval_s=3600.0, vram_gib=24.0)
    assert size % 4 == 0 and size >= 8
    _centered_placement(parent, j0=250, i0=250, ratio=2,
                        child_nx=size, child_ny=size)


def test_downscale_cli_dry_run_child_config_mode(tmp_path, capsys):
    start = datetime(1974, 4, 3, 12)
    frames = []
    for index in range(3):
        path = tmp_path / f"wrfout_d03_1974-04-03_{12 + index:02d}_00_00"
        _history(path, start + timedelta(hours=index), ny=18, nx=20)
        frames.append(path)
    namelist = tmp_path / "namelist.input"
    namelist.write_text("&physics\n mp_physics = 8,\n/\n", encoding="utf-8")
    child_toml = tmp_path / "child.toml"
    parent = {"nx": 20, "ny": 18, "dx": 1000.0, "dy": 1000.0}
    merged = _derive_child_run_config(
        _PARENT_CONFIG, parent=parent, ratio=1, child_nx=12, child_ny=10,
        run_seconds=600.0, output_interval_s=300.0)
    child_toml.write_text(_render_child_toml(merged), encoding="utf-8")

    # Cadence is a contract: no flag, no run.  The CLI dispatch boundary
    # turns the OfflineChildContractError refusal into exit 2 + a stderr
    # message (no traceback), matching the wizard/fetch refusal contract.
    rc_args = [
        "downscale", str(tmp_path), "--parent-domain", "3",
        "--parent-namelist", str(namelist),
        "--child-config", str(child_toml), "--ratio", "1",
        "--i-parent-start", "4", "--j-parent-start", "4",
        "--out", str(tmp_path / "child-run"), "--dry-run"]
    assert cli_main(rc_args) == 2
    err = capsys.readouterr().err
    assert "scientific" in err
    assert "Traceback" not in err

    assert cli_main(rc_args + ["--accept-parent-cadence"]) == 0
    out = capsys.readouterr().out
    assert "downscale_plan" in out
    assert "coarser" in out  # 3600 s > the 900 s guidance prints the caveat


def test_cadence_flags_are_mutually_exclusive(tmp_path, capsys):
    """Supplying both cadence options is an argument error naming the
    pair -- --max-boundary-interval-seconds must never silently win
    (audit finding 5)."""
    start = datetime(1974, 4, 3, 12)
    for index in range(2):
        _history(tmp_path / f"wrfout_d03_1974-04-03_{12 + index:02d}_00_00",
                 start + timedelta(hours=index), ny=18, nx=20)
    with pytest.raises(SystemExit) as excinfo:
        cli_main([
            "downscale", str(tmp_path),
            "--parent-namelist", str(tmp_path / "namelist.input"),
            "--child-config", str(tmp_path / "child.toml"),
            "--ratio", "1", "--i-parent-start", "4",
            "--j-parent-start", "4",
            "--max-boundary-interval-seconds", "900",
            "--accept-parent-cadence",
            "--out", str(tmp_path / "child-run"), "--dry-run"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "--accept-parent-cadence" in err
    assert "--max-boundary-interval-seconds" in err
    assert "not allowed with" in err


def _dry_run_plan(tmp_path, capsys, cadence_args):
    import json

    tmp_path.mkdir(parents=True, exist_ok=True)
    start = datetime(1974, 4, 3, 12)
    for index in range(3):
        _history(tmp_path / f"wrfout_d03_1974-04-03_{12 + index:02d}_00_00",
                 start + timedelta(hours=index), ny=18, nx=20)
    namelist = tmp_path / "namelist.input"
    namelist.write_text("&physics\n mp_physics = 8,\n/\n", encoding="utf-8")
    child_toml = tmp_path / "child.toml"
    parent = {"nx": 20, "ny": 18, "dx": 1000.0, "dy": 1000.0}
    merged = _derive_child_run_config(
        _PARENT_CONFIG, parent=parent, ratio=1, child_nx=12, child_ny=10,
        run_seconds=600.0, output_interval_s=300.0)
    child_toml.write_text(_render_child_toml(merged), encoding="utf-8")
    assert cli_main([
        "downscale", str(tmp_path), "--parent-domain", "3",
        "--parent-namelist", str(namelist),
        "--child-config", str(child_toml), "--ratio", "1",
        "--i-parent-start", "4", "--j-parent-start", "4",
        "--out", str(tmp_path / "child-run"), "--dry-run",
        *cadence_args]) == 0
    printed = capsys.readouterr().out
    return json.loads(printed[printed.index("{"):])


def test_plan_records_which_cadence_flag_was_given(tmp_path, capsys):
    plan = _dry_run_plan(tmp_path, capsys, ["--accept-parent-cadence"])
    assert plan["accepted_parent_cadence"] is True
    assert plan["max_boundary_interval_seconds"] == 3600.0

    explicit = _dry_run_plan(
        tmp_path / "explicit", capsys,
        ["--max-boundary-interval-seconds", "3600"])
    assert explicit["accepted_parent_cadence"] is False
    assert explicit["max_boundary_interval_seconds"] == 3600.0


def test_runner_namespace_threads_the_acceptance_provenance(
        tmp_path, capsys, monkeypatch):
    """downscale_main must hand the runner the flag it was given, so
    report.json can record the acknowledgment (audit finding 5)."""
    import gpuwm.offline_child_run as offline_child_run

    captured = {}

    def fake_run(namespace):
        captured["namespace"] = namespace
        return {"result": "PASS"}

    monkeypatch.setattr(offline_child_run, "run", fake_run)
    start = datetime(1974, 4, 3, 12)
    for index in range(3):
        _history(tmp_path / f"wrfout_d03_1974-04-03_{12 + index:02d}_00_00",
                 start + timedelta(hours=index), ny=18, nx=20)
    namelist = tmp_path / "namelist.input"
    namelist.write_text("&physics\n mp_physics = 8,\n/\n", encoding="utf-8")
    child_toml = tmp_path / "child.toml"
    parent = {"nx": 20, "ny": 18, "dx": 1000.0, "dy": 1000.0}
    merged = _derive_child_run_config(
        _PARENT_CONFIG, parent=parent, ratio=1, child_nx=12, child_ny=10,
        run_seconds=600.0, output_interval_s=300.0)
    child_toml.write_text(_render_child_toml(merged), encoding="utf-8")
    assert cli_main([
        "downscale", str(tmp_path), "--parent-domain", "3",
        "--parent-namelist", str(namelist),
        "--child-config", str(child_toml), "--ratio", "1",
        "--i-parent-start", "4", "--j-parent-start", "4",
        "--accept-parent-cadence",
        "--out", str(tmp_path / "child-run")]) == 0
    capsys.readouterr()
    namespace = captured["namespace"]
    assert namespace.accepted_parent_cadence is True
    assert namespace.max_boundary_interval_seconds == 3600.0

    # The standalone runner accepts the same provenance flag, so the
    # two entry points stay isomorphic; it defaults to False.
    from gpuwm.offline_child_run import _parser
    parsed = _parser().parse_args([
        "--parent-history", "x", "--parent-restart", "r",
        "--child-config", "c", "--parent-grid-ratio", "3",
        "--i-parent-start", "1", "--j-parent-start", "1",
        "--max-boundary-interval-seconds", "900", "--outdir", "o"])
    assert parsed.accepted_parent_cadence is False


def test_downscale_cli_rejects_hours_with_child_config(tmp_path, capsys):
    start = datetime(1974, 4, 3, 12)
    for index in range(2):
        _history(tmp_path / f"wrfout_d03_1974-04-03_{12 + index:02d}_00_00",
                 start + timedelta(hours=index), ny=18, nx=20)
    namelist = tmp_path / "namelist.input"
    namelist.write_text("&physics\n mp_physics = 8,\n/\n", encoding="utf-8")
    rc = cli_main([
        "downscale", str(tmp_path),
        "--parent-namelist", str(namelist),
        "--child-config", str(tmp_path / "missing.toml"),
        "--ratio", "1", "--i-parent-start", "4", "--j-parent-start", "4",
        "--hours", "2", "--accept-parent-cadence",
        "--out", str(tmp_path / "child-run"), "--dry-run"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--point" in err
    assert "Traceback" not in err


def _surface_file(path, *, ny=10, nx=12, soil=4, with_identity=True,
                  lu_value=7.0):
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("Time", 1)
        dataset.createDimension("south_north", ny)
        dataset.createDimension("west_east", nx)
        dataset.createDimension("soil_layers_stag", soil)
        if with_identity:
            dataset.MMINLU = "MODIFIED_IGBP_MODIS_NOAH"
            dataset.ISWATER = 17
            dataset.ISLAKE = 21
            dataset.ISICE = 15
            dataset.ISOILWATER = 14
        mass2 = ("Time", "south_north", "west_east")
        soil3 = ("Time", "soil_layers_stag", "south_north", "west_east")
        for name, value in (
                ("LU_INDEX", lu_value), ("LANDMASK", 1.0), ("ISLTYP", 6.0),
                ("TSK", 288.0), ("TMN", 285.0), ("VEGFRA", 50.0),
                ("SNOW", 0.0), ("SNOWH", 0.0), ("PSFC", 96000.0),
                ("T2", 287.0), ("Q2", 0.008), ("U10", 2.0), ("V10", -1.0),
                ("XLAT", 39.5), ("XLONG", -84.0)):
            variable = dataset.createVariable(name, "f4", mass2)
            variable[:] = np.full((1, ny, nx), value, dtype=np.float32)
        for name, value in (("TSLB", 285.0), ("SMOIS", 0.3),
                            ("SH2O", 0.3)):
            variable = dataset.createVariable(name, "f4", soil3)
            variable[:] = np.full((1, soil, ny, nx), value, dtype=np.float32)


def test_child_surface_state_reads_exact_grid(tmp_path):
    path = tmp_path / "wrfinput_child"
    _surface_file(path)
    surface = read_child_surface_state(
        path, child_ny=10, child_nx=12, num_soil_layers=4)
    assert surface.identity["MMINLU"] == "MODIFIED_IGBP_MODIS_NOAH"
    assert surface.identity["ISWATER"] == 17
    assert surface.fields["TSLB"].shape == (4, 10, 12)
    assert surface.fields["LU_INDEX"].dtype == np.float32
    assert "ndown-equivalent" in surface.receipt["policy"]

    with pytest.raises(OfflineChildContractError, match="EXACT child grid"):
        read_child_surface_state(
            path, child_ny=11, child_nx=12, num_soil_layers=4)
    with pytest.raises(OfflineChildContractError, match="soil"):
        read_child_surface_state(
            path, child_ny=10, child_nx=12, num_soil_layers=5)


def test_child_surface_state_requires_identity_and_integer_categories(
        tmp_path):
    anonymous = tmp_path / "no-identity"
    _surface_file(anonymous, with_identity=False)
    with pytest.raises(OfflineChildContractError, match="identity"):
        read_child_surface_state(
            anonymous, child_ny=10, child_nx=12, num_soil_layers=4)
    smoothed = tmp_path / "smoothed-categories"
    _surface_file(smoothed, lu_value=7.4)
    with pytest.raises(OfflineChildContractError, match="non-integer"):
        read_child_surface_state(
            smoothed, child_ny=10, child_nx=12, num_soil_layers=4)


def test_child_boundary_clock_reproduces_wrf_dtbc_recurrence():
    cfg = Namespace(dt=2.5, grid_id=4)
    clock = _child_boundary_clock(
        cfg, lbc_interval_seconds=30.0, steps=24, output_steps=12)
    assert clock.tick_den == 2
    assert clock.spec.step_ticks == 5
    assert clock.spec.lbc_interval_ticks == 60
    assert clock.run_ticks == 120
    observed = []
    for _ in range(24):
        if clock.lbc_reset_due():
            clock.mark_force()
        clock.prepare_step()
        observed.append(float(clock.dtbc_launch_fp32))
        clock.advance()
    # WRF's post-increment recurrence: dtbc restarts at dt after every
    # interval seam (including t=0) and reaches T_bdy on the pre-seam step.
    expected = [2.5 * (1 + index % 12) for index in range(24)]
    assert observed == expected
    assert clock.elapsed_seconds == 60.0
    assert math.isclose(float(clock.elapsed_seconds_fp32), 60.0)

    with pytest.raises(OfflineChildContractError, match="whole"):
        _child_boundary_clock(
            cfg, lbc_interval_seconds=31.0, steps=24, output_steps=12)
