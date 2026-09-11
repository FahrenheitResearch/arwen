"""CPU-side contracts for ``gpuwm downscale`` and its runner plumbing.

The GPU forecast itself is exercised by the acceptance run; these tests
pin the file-facing half: parent-series discovery, point placement, the
derived child config, the child-grid surface source contract, and the
Davies-bind boundary clock the standalone child constructs.
"""

from argparse import Namespace
import json
from datetime import datetime, timedelta
import math
from pathlib import Path
import threading

import netCDF4
import numpy as np
import pytest

from gpuwm import netcdf_bridge
from gpuwm.cli import main as cli_main
from gpuwm.config import load_config
from gpuwm.downscale import (
    DERIVED_CHILD_CONFIG_NAME,
    DOWNSCALE_PLAN_SCHEMA,
    _budget_bytes,
    _centered_placement,
    _derive_child_run_config,
    _discover_parent_series,
    _fit_child_size,
    _nearest_parent_index,
    _parse_point,
    _render_child_toml,
    derived_child_config_path,
    downscale_plan_path,
)
from gpuwm.offline_child import (
    OfflineChildContractError,
    read_child_surface_state,
)
from gpuwm.offline_child_run import _child_boundary_clock
from test_offline_child import _history


#: What every test in this deck runs on: a computer that CAN draw.
#:
#: `gpuwm downscale` draws by default, so a box with no staged renderer
#: is turned away at admission -- before the request's own contracts are
#: read, because the remedy for that refusal must not have to undo an
#: --out this command created.  A test box has no renderer, so without
#: this every refusal this deck pins (the surface-source contract, the
#: --out collisions, the cadence sentences) would arrive as the render
#: refusal instead.  The refusal itself is pinned by
#: ``test_a_computer_that_cannot_draw_is_refused_before_anything_is_opened``,
#: which overrides this, and the catalog is a fixed one so no test in
#: this deck depends on a renderer being installed to be admitted.
_CATALOG_FIXTURE = {
    "engine": "rust",
    "products": [{"name": "composite_reflectivity"}, {"name": "mslp_10m_winds"},
                 {"name": "2m_temperature"}, {"name": "total_qpf"},
                 {"name": "10m_wind_speed_and_direction"}],
    "group_keywords": ["severe", "surface"],
}


@pytest.fixture(autouse=True)
def _a_box_that_can_draw(monkeypatch):
    import gpuwm.go_cli as go_cli
    import gpuwm.runplan as runplan

    monkeypatch.setattr(go_cli, "render_extra_missing", lambda: None)
    monkeypatch.setattr(runplan, "render_catalog", lambda: dict(_CATALOG_FIXTURE))


#: The three tests marked with this hand the door a real wrfout or restart
#: set and let it READ the file, which gpuwm decodes through the Rust
#: rw_netcdf binary and nothing else (gpuwm.netcdf_bridge.NetcdfBridgeMissing
#: otherwise).  The rest of this deck writes its fixtures with netCDF4 and
#: never asks gpuwm to decode one, so the gate is per test rather than a
#: module-level pytestmark that would retire the whole deck.
needs_netcdf_bridge = pytest.mark.skipif(
    netcdf_bridge.find_netcdf_bin() is None,
    reason="rw_netcdf is not built; build tools/rustwx to run this")


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
    size, estimate = _fit_child_size(
        parent, config, j0=250, i0=250, ratio=2, run_seconds=3600.0,
        output_interval_s=3600.0, vram_gib=24.0)
    # The winning size comes back WITH the estimator's own answer for it,
    # so the plan document quotes the number the fit was decided on.
    assert estimate is not None
    assert estimate.peak_envelope_bytes <= _budget_bytes(24.0)[1]
    assert size % 4 == 0 and size >= 8
    _centered_placement(parent, j0=250, i0=250, ratio=2,
                        child_nx=size, child_ny=size)


#: The measured parent of the finding-4 walk: a real 386x308 12 km GFS
#: parent (morrison rte-rrtmgp suite), its restart-evidence config keys
#: verbatim.  Physics inherited as data, exactly as a restart carries it.
_MEASURED_PARENT_CONFIG = {
    "nx": 386, "ny": 308, "nz": 49, "dx": 12000.0, "dy": 12000.0,
    "dt": 60.0, "run_seconds": 7200.0, "output_interval_s": 900.0,
    "ztop": 20000.0, "time_step_sound": 4, "epssm": 0.5,
    "hybrid_opt": 2, "etac": 0.2, "hypsometric_opt": 2,
    "moist": True, "moist_cq": True, "mp_physics": 10,
    "morr_rimed_ice": 1, "ra_physics": 0, "ra_lw_physics": 4,
    "ra_sw_physics": 4, "radt": 12.0, "ra_rrtmg_variant": "rte-rrtmgp",
    "sf_sfclay_physics": 91, "sf_surface_physics": 2,
    "bl_pbl_physics": 1, "cu_physics": 1, "cudt_minutes": 5.0,
    "num_soil_layers": 4, "km_opt": 4, "diff_6th_opt": 2,
    "diff_6th_factor": 0.12, "diff_6th_slopeopt": 1,
    "specified": True, "nested": False, "grid_id": 1,
    "spec_bdy_width": 5, "spec_zone": 1, "relax_zone": 4,
    "map_proj": 1, "terrain_opt": 1, "nwp_diagnostics": 1,
}


def test_the_fit_admits_the_child_the_card_measuredly_ran():
    """The standalone-child fit prices with the affine envelope, not the
    retired reserve + 1.75x multiplicative path.

    MEASURED 2026-08-26 on the RTX 3080 10 GiB (stale-guard audit
    2026-08-25, finding 4): from this exact parent at --vram-gib 10,
    ratio 3, the retired path admitted 282x282 while the affine fit
    admits 342x342 -- and the 342x342 child RAN WHOLE through the real
    downscale door (360 steps, 7,200 s simulated, PASS, machine-wide
    peak 9.24 of 10.24 GB with the desktop compositing beside it).  The
    retired path refused 47% more child area than the card measuredly
    holds; a fit that refuses a run the card completes is the defect.
    """

    parent = {"nx": 386, "ny": 308, "dx": 12000.0, "dy": 12000.0}
    size, _ = _fit_child_size(
        parent, dict(_MEASURED_PARENT_CONFIG), j0=154, i0=193, ratio=3,
        run_seconds=7200.0, output_interval_s=900.0, vram_gib=10.0)
    assert size >= 342, (
        f"the fit admits {size}x{size} where the card measuredly ran "
        "342x342 whole")


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

    # Warn-not-block: with no cadence flag the archive's own cadence is
    # the default -- one warning line says so and the run proceeds.
    rc_args = [
        "downscale", str(tmp_path), "--parent-domain", "3",
        "--parent-namelist", str(namelist),
        "--child-config", str(child_toml), "--ratio", "1",
        "--i-parent-start", "4", "--j-parent-start", "4",
        "--out", str(tmp_path / "child-run"), "--dry-run"]
    # A child's 300-second output interval is not a ceiling on its hourly
    # parent's boundary data. The old research guide conflated these.
    assert cli_main(rc_args + ["--max-boundary-interval-seconds", "300"]) == 2
    refused = capsys.readouterr()
    assert "300" in refused.err
    assert cli_main(rc_args) == 0
    captured = capsys.readouterr()
    assert "downscale_plan" in captured.out
    assert "warning:" in captured.err
    assert "cadence" in captured.err
    # 3600 s > the 900 s guidance prints the caveat, as a warning.
    assert "coarser" in captured.err
    assert "Traceback" not in captured.err

    # --accept-parent-cadence still works (now as the explicit,
    # warning-free spelling of the same default).
    assert cli_main(rc_args + ["--accept-parent-cadence"]) == 0
    captured = capsys.readouterr()
    assert "downscale_plan" in captured.out
    assert "using the parent archive's own" not in captured.err


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


def test_downscale_cli_warns_and_ignores_hours_with_child_config(
        tmp_path, capsys):
    """Warn-not-block: an inert flag is named and ignored, never fatal."""

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
    rc = cli_main([
        "downscale", str(tmp_path),
        "--parent-namelist", str(namelist),
        "--child-config", str(child_toml),
        "--ratio", "1", "--i-parent-start", "4", "--j-parent-start", "4",
        "--hours", "2", "--accept-parent-cadence",
        "--out", str(tmp_path / "child-run"), "--dry-run"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "downscale_plan" in captured.out
    assert "warning:" in captured.err
    assert "--hours" in captured.err
    assert "ignored" in captured.err
    assert "Traceback" not in captured.err


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


#: The reference-shaped surface selection every real parent carries
#: (Noah + MM5 surface layer + YSU), purely as data: a child inheriting
#: it needs a child-grid surface source before anything runs.
_SURFACE_PARENT_CONFIG = dict(
    _PARENT_CONFIG, sf_surface_physics=2, sf_sfclay_physics=91,
    bl_pbl_physics=1, num_soil_layers=4)


def _surface_child_args(tmp_path):
    """One valid --child-config invocation whose child needs a surface."""
    start = datetime(1974, 4, 3, 12)
    for index in range(3):
        path = tmp_path / f"wrfout_d03_1974-04-03_{12 + index:02d}_00_00"
        _history(path, start + timedelta(hours=index), ny=18, nx=20)
    namelist = tmp_path / "namelist.input"
    namelist.write_text("&physics\n mp_physics = 8,\n/\n", encoding="utf-8")
    child_toml = tmp_path / "child.toml"
    parent = {"nx": 20, "ny": 18, "dx": 1000.0, "dy": 1000.0}
    merged = _derive_child_run_config(
        _SURFACE_PARENT_CONFIG, parent=parent, ratio=1, child_nx=12,
        child_ny=10, run_seconds=600.0, output_interval_s=300.0)
    child_toml.write_text(_render_child_toml(merged), encoding="utf-8")
    return [
        "downscale", str(tmp_path), "--parent-domain", "3",
        "--parent-namelist", str(namelist),
        "--child-config", str(child_toml), "--ratio", "1",
        "--i-parent-start", "4", "--j-parent-start", "4",
        "--out", str(tmp_path / "child-run")]


def test_derived_child_config_lands_inside_the_run_it_describes(tmp_path):
    """A run's own config belongs in the run directory, not beside it.

    The walked defect: `--point` derivation wrote `<out>.child.toml`, a
    SIBLING of `--out`, so a reader who opened the run directory looking
    for the config it used found no config at all.
    """
    outdir = tmp_path / "child-run"
    assert derived_child_config_path(outdir, dry_run=False) == (
        outdir / DERIVED_CHILD_CONFIG_NAME)
    # A dry run must NOT create --out: the run that follows reserves it
    # with exist_ok=False and would refuse.  It writes beside it instead.
    planned = derived_child_config_path(outdir, dry_run=True)
    assert planned.parent == tmp_path and planned != outdir


def test_run_accepts_the_output_root_its_caller_reserved(tmp_path):
    """The never-adopt reservation happens exactly once, not zero times.

    "Never adopt" is about a directory that HOLDS an earlier run: its
    frames would be merged with this run's and the report.json beside
    them would then describe two.  An empty directory holds neither, so
    the reservation takes it -- see
    ``test_the_runner_door_refuses_a_used_outdir_in_words`` for the half
    that refuses, in words rather than as a Windows error number.
    """
    from gpuwm.offline_child_run import _create_output_root

    outdir = tmp_path / "child-run"
    assert _create_output_root(outdir) == outdir.resolve()
    (outdir / "gpuwmrst_d02_1974-04-03_12_15_00.npz").write_bytes(b"x")
    with pytest.raises(OfflineChildContractError):
        _create_output_root(outdir)


def test_fit_refusal_names_the_real_error_not_the_budget():
    """A config-invalid derivation must not read as a VRAM verdict.

    ``_fit_child_size`` probed sizes, caught every ValueError, and
    reported "no child fits the N GiB budget inside this parent" -- so a
    size-INDEPENDENT config invalidity (walked live: a restart-evidence
    key the tip refused) was blamed on the card.  The refusal must carry
    the validation error's own sentence.
    """
    parent = {"nx": 501, "ny": 501, "dx": 1000.0, "dy": 1000.0}
    config = dict(_PARENT_CONFIG, nx=501, ny=501, nz=49,
                  sase_moist_n2=False, bl_pbl_physics=1)
    with pytest.raises(OfflineChildContractError) as caught:
        _fit_child_size(
            parent, config, j0=250, i0=250, ratio=2, run_seconds=3600.0,
            output_interval_s=3600.0, vram_gib=24.0)
    assert "sase_moist_n2" in str(caught.value)
    assert "budget" not in str(caught.value)


def test_dry_run_names_the_missing_surface_source(tmp_path, capsys):
    """--dry-run stays runnable and says the run itself will not be.

    The walked defect (2026-08-17, 2.4.1 wheel): a --point derivation's
    dry run printed a green plan with ``"child_surface_from": null`` and
    zero mention that the run would refuse for exactly that null -- the
    plan said GO, the run said no.  The dry run must keep printing the
    plan (deriving the geometry is HOW a user learns which child grid to
    build a surface file for), warn with the remedy, and record the
    requirement in the plan document.
    """
    import json as json_module

    assert cli_main(_surface_child_args(tmp_path) + ["--dry-run"]) == 0
    captured = capsys.readouterr()
    plan = json_module.loads(
        captured.out[captured.out.index("{"):])
    assert plan["child_surface_required"] is True
    assert plan["child_surface_from"] is None
    assert "child-grid surface source" in captured.err
    assert "wrf-native-input/wrfinput_d0N" in captured.err


def test_run_refuses_missing_surface_source_before_any_work(
        tmp_path, capsys):
    """The real run refuses at the front door, naming the in-product remedy.

    Walked on the 2.4.1 wheel: the refusal named the flag and the
    contract but no way to SATISFY them, while the user's own
    preparation already held a valid child-grid ``wrfinput_d0N`` under
    ``wrf-native-input/`` -- rw-wps writes one per nest.  The refusal
    must name that recipe, and it must fire before preprocessing or the
    output directory exist.
    """
    rc = cli_main(_surface_child_args(tmp_path))
    captured = capsys.readouterr()
    assert rc != 0
    assert "child-grid surface source" in captured.err
    assert "--child-surface-from" in captured.err
    assert "wrf-native-input/wrfinput_d0N" in captured.err
    assert "rw-wps" in captured.err
    assert not (tmp_path / "child-run").exists()


def test_surface_satisfied_plan_records_no_requirement(tmp_path, capsys):
    """A surface-free (microphysics-only) child plans as before."""
    import json as json_module

    args = _surface_child_args(tmp_path)
    # Overwrite the child with _PARENT_CONFIG's surface-free selection:
    # the requirement must read False and nothing may warn.
    child_toml = tmp_path / "child.toml"
    parent = {"nx": 20, "ny": 18, "dx": 1000.0, "dy": 1000.0}
    merged = _derive_child_run_config(
        _PARENT_CONFIG, parent=parent, ratio=1, child_nx=12,
        child_ny=10, run_seconds=600.0, output_interval_s=300.0)
    child_toml.write_text(_render_child_toml(merged), encoding="utf-8")
    assert cli_main(args + ["--dry-run"]) == 0
    captured = capsys.readouterr()
    plan = json_module.loads(captured.out[captured.out.index("{"):])
    assert plan["child_surface_required"] is False
    assert "child-grid surface source" not in captured.err


def _add_parent_surface(path, *, ny, nx, soil=4, lu_water_column=None,
                        omit=()):
    """Give an existing parent history frame its land-surface inventory.

    gpuwm's own writer publishes all nine of
    ``offline_child._SURFACE_REQUIRED_FIELDS`` whenever a land-surface
    scheme is routed (``io.wrf_output_schema.SURFACE_IDENTITY_OUTPUT_FIELDS``
    plus the LSM-gated soil family), and stamps the landuse identity
    attributes.  The ``_history`` fixture predates that inventory, so the
    surface half is appended here rather than widening a fixture eleven
    other tests share.
    """
    with netCDF4.Dataset(path, "a") as dataset:
        dataset.MMINLU = "MODIFIED_IGBP_MODIS_NOAH"
        dataset.ISWATER = 17
        dataset.ISLAKE = 21
        dataset.ISICE = 15
        dataset.ISOILWATER = 14
        if "soil_layers_stag" not in dataset.dimensions:
            dataset.createDimension("soil_layers_stag", soil)
        mass2 = ("Time", "south_north", "west_east")
        soil3 = ("Time", "soil_layers_stag", "south_north", "west_east")
        lu = np.full((ny, nx), 7.0, dtype=np.float32)
        landmask = np.ones((ny, nx), dtype=np.float32)
        if lu_water_column is not None:
            lu[:, lu_water_column] = 17.0
            landmask[:, lu_water_column] = 0.0
        for name, value in (
                ("LU_INDEX", lu), ("LANDMASK", landmask),
                ("ISLTYP", np.full((ny, nx), 6.0, dtype=np.float32)),
                ("TSK", np.full((ny, nx), 288.0, dtype=np.float32)),
                ("TMN", np.full((ny, nx), 285.0, dtype=np.float32)),
                ("VEGFRA", np.full((ny, nx), 50.0, dtype=np.float32)),
                ("SNOW", np.zeros((ny, nx), dtype=np.float32)),
                ("SNOWH", np.zeros((ny, nx), dtype=np.float32)),
                ("T2", np.full((ny, nx), 287.0, dtype=np.float32)),
                ("Q2", np.full((ny, nx), 0.008, dtype=np.float32))):
            if name in omit or name in dataset.variables:
                continue
            variable = dataset.createVariable(name, "f4", mass2)
            variable[:] = value[None, ...]
        for name, value in (("TSLB", 285.0), ("SMOIS", 0.3), ("SH2O", 0.3)):
            if name in omit:
                continue
            variable = dataset.createVariable(name, "f4", soil3)
            variable[:] = np.full((1, soil, ny, nx), value, dtype=np.float32)


def test_child_surface_derived_from_parent_history(tmp_path):
    """The parent's own history seeds a full-physics child, no extra file.

    Defect #275: an ERA5 parent reached through `gpuwm run` produced no
    child-grid file anywhere, and every route the product offered for
    making one was closed, so full-physics downscaling of an ERA5 parent
    was unreachable.  The parent history already carries all nine
    required surface fields and the landuse identity attributes; putting
    them on the child grid is WRF's own nest-birth operator
    (``interp_mask_field``), not new science.
    """
    from gpuwm.offline_child import (
        OfflineChildPlacement,
        derive_child_surface_from_parent,
    )

    frame = tmp_path / "wrfout_d01_1974-04-03_12_00_00"
    _history(frame, datetime(1974, 4, 3, 12), ny=18, nx=20)
    _add_parent_surface(frame, ny=18, nx=20, lu_water_column=8)
    placement = OfflineChildPlacement(
        parent_nx=20, parent_ny=18, child_nx=12, child_ny=9,
        parent_grid_ratio=3, i_parent_start=6, j_parent_start=6)
    surface = derive_child_surface_from_parent(
        frame, placement=placement, num_soil_layers=4)
    assert surface.fields["TSLB"].shape == (4, 9, 12)
    assert surface.fields["LU_INDEX"].shape == (9, 12)
    # Categories stay exact integers -- a smoothed category field is not
    # a valid land identity, and read_child_surface_state refuses one.
    for name in ("LU_INDEX", "ISLTYP"):
        value = surface.fields[name]
        assert np.array_equal(value, np.rint(value))
    assert set(np.unique(surface.fields["LU_INDEX"]).tolist()) <= {7.0, 17.0}
    # The child resolves the parent's water column at child spacing.
    assert 17.0 in np.unique(surface.fields["LU_INDEX"]).tolist()
    assert surface.identity["MMINLU"] == "MODIFIED_IGBP_MODIS_NOAH"
    assert surface.identity["ISWATER"] == 17
    assert surface.receipt["source"] == "parent-history-interpolated"
    assert "interp_mask_field" in surface.receipt["policy"]


def test_derivation_refuses_a_parent_without_the_surface_inventory(tmp_path):
    """Naming the missing fields, not just "pass --child-surface-from"."""
    from gpuwm.offline_child import (
        OfflineChildPlacement,
        derive_child_surface_from_parent,
    )

    frame = tmp_path / "wrfout_d01_1974-04-03_12_00_00"
    _history(frame, datetime(1974, 4, 3, 12), ny=18, nx=20)
    _add_parent_surface(frame, ny=18, nx=20, omit=("TMN", "VEGFRA"))
    placement = OfflineChildPlacement(
        parent_nx=20, parent_ny=18, child_nx=12, child_ny=9,
        parent_grid_ratio=3, i_parent_start=6, j_parent_start=6)
    with pytest.raises(OfflineChildContractError) as caught:
        derive_child_surface_from_parent(
            frame, placement=placement, num_soil_layers=4)
    message = str(caught.value)
    assert "TMN" in message and "VEGFRA" in message
    assert "--child-surface-from" in message


def test_full_physics_child_needs_no_surface_flag(tmp_path, capsys):
    """FIXED MEANS DEFAULT: the bare invocation stops refusing.

    The same arguments that produced the walked "child config enables
    surface physics ... but no child-grid surface source was given"
    refusal now plan, with the derivation named in the plan and its
    fidelity cost warned about.
    """
    import json as json_module

    args = _surface_child_args(tmp_path)
    for index in range(3):
        _add_parent_surface(
            tmp_path / f"wrfout_d03_1974-04-03_{12 + index:02d}_00_00",
            ny=18, nx=20, lu_water_column=8)
    assert cli_main(args + ["--dry-run"]) == 0
    captured = capsys.readouterr()
    plan = json_module.loads(captured.out[captured.out.index("{"):])
    assert plan["child_surface_required"] is True
    assert plan["child_surface_source"] == "parent-history-interpolated"
    # The flag is still reported as the higher-fidelity route.
    assert "--child-surface-from" in captured.err
    assert "parent" in captured.err


# ---------------------------------------------------------------------
# The --out reservation: a refusal must not poison the directory it
# reserved, and a directory that already holds a run is a sentence.
# ---------------------------------------------------------------------

def _restart_evidence(path, config):
    """One gpuwm restart header, written as the reader expects to find it.

    ``--point`` derivation inherits the child's physics from the parent's
    restart evidence, so a CPU-side walk of that route needs a restart
    file.  Only the header is read on this route
    (``read_restart_header``), so the archive carries the header and no
    arrays.
    """
    import hashlib
    import json as json_module

    def canonical(value):
        return hashlib.sha256(json_module.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()

    setup = {"microphysics": {"scheme_id": int(config["mp_physics"]),
                              "name": "parent-fixture"}}
    header = {
        "format_version": 6,
        "grid_id": int(config.get("grid_id", 1)),
        "config": dict(config),
        "physics_setup": setup,
        "physics_setup_fingerprint": canonical(setup),
        "setup_fingerprint": canonical(dict(config)),
        "array_manifest": {},
        "elapsed_seconds": 0.0,
    }
    payload = np.frombuffer(
        json_module.dumps(header).encode("utf-8"), dtype=np.uint8)
    np.savez(path, **{"__gpuwm_restart_header__": payload})
    return path if path.suffix == ".npz" else path.with_suffix(".npz")


def _give_the_parent_a_real_projection(path, *, ny, nx):
    """Vary XLAT/XLONG so ``--point`` resolves to an interior parent cell."""
    lat = np.linspace(38.0, 41.0, ny)[:, None] * np.ones((1, nx))
    lon = np.ones((ny, 1)) * np.linspace(-86.0, -82.0, nx)[None, :]
    with netCDF4.Dataset(path, "a") as dataset:
        dataset.variables["XLAT"][0] = lat.astype(np.float32)
        dataset.variables["XLONG"][0] = lon.astype(np.float32)


def _point_args(tmp_path, *, ny=18, nx=20):
    """A ``--point`` invocation whose child inherits surface physics.

    The parent frames carry no land-surface inventory, so the run refuses
    -- AFTER ``--point`` derivation has reserved ``--out`` to hold the
    config it just wrote.  That ordering is the whole subject below.
    """
    start = datetime(1974, 4, 3, 12)
    for index in range(3):
        frame = tmp_path / f"wrfout_d01_1974-04-03_{12 + index:02d}_00_00"
        _history(frame, start + timedelta(hours=index), ny=ny, nx=nx)
        _give_the_parent_a_real_projection(frame, ny=ny, nx=nx)
    restart = _restart_evidence(
        tmp_path / "gpuwmrst_d01_1974-04-03_12_00_00.npz",
        dict(_SURFACE_PARENT_CONFIG, nx=nx, ny=ny, nz=2, grid_id=1,
             dt=3.0, run_seconds=7200.0, nested=False, specified=False))
    return [
        "downscale", str(tmp_path),
        "--parent-restart", str(restart),
        "--point", "39.5,-84.0",
        "--ratio", "1", "--child-size", "12,10",
        # _history's compact archive has two levels; the runnable child
        # requires at least four and declares its own remapping ladder.
        "--child-levels", "4,2.5",
        "--hours", "0.25", "--output-interval-seconds", "900",
        "--out", str(tmp_path / "child-run")]


def test_a_refused_downscale_releases_the_out_it_reserved(tmp_path, capsys):
    """A refusal must not leave the directory it created behind.

    ``--point`` creates ``--out`` create-only so the config it derives can
    live inside the run it describes.  Every refusal raised after that --
    a parent that cannot seed the child's surface is the walked one --
    used to leave ``--out`` holding ``child.toml``, which is a run
    directory describing a run that never happened AND the thing the
    corrected retry then collides with.
    """
    rc = cli_main(_point_args(tmp_path))
    captured = capsys.readouterr()
    assert rc == 2
    assert "child-grid surface source" in captured.err
    assert not (tmp_path / "child-run").exists()


def test_the_corrected_retry_runs_instead_of_colliding(
        tmp_path, capsys, monkeypatch):
    """The whole defect, end to end: refuse, correct, retry, run.

    The retry used to die with an uncaught ``FileExistsError`` from the
    reservation ``mkdir`` -- a Windows error number as the last line, at
    exit 1, for a command that was now correct.
    """
    import gpuwm.offline_child_run as offline_child_run

    args = _point_args(tmp_path)
    assert cli_main(args) == 2
    capsys.readouterr()

    # THE CORRECTION: the parent gains the land-surface inventory the
    # refusal named, so the child can be seeded from the parent's own
    # history.  Nothing else about the command changes.
    for index in range(3):
        _add_parent_surface(
            tmp_path / f"wrfout_d01_1974-04-03_{12 + index:02d}_00_00",
            ny=18, nx=20, lu_water_column=8)
    monkeypatch.setattr(offline_child_run, "run",
                        lambda namespace: {"result": "PASS"})
    assert cli_main(args) == 0
    capsys.readouterr()
    assert (tmp_path / "child-run" / DERIVED_CHILD_CONFIG_NAME).is_file()


def test_an_out_holding_an_earlier_run_is_a_sentence_not_a_traceback(
        tmp_path, capsys):
    """The collision names the directory, what it holds, and the way out."""
    outdir = tmp_path / "child-run"
    outdir.mkdir(parents=True)
    (outdir / "report.json").write_text("{}", encoding="utf-8")

    rc = cli_main(_point_args(tmp_path))
    captured = capsys.readouterr()
    assert rc == 2
    assert "report.json" in captured.err
    assert str(outdir) in captured.err
    assert "--out" in captured.err


def test_child_config_route_refuses_a_used_out_at_the_front_door(
        tmp_path, capsys):
    """The other child mode reaches the same sentence, and reaches it early.

    ``--child-config`` reserved ``--out`` inside the runner, after the
    CUDA import and after the whole parent archive had been validated, so
    a collision was discovered as late as it possibly could be.
    """
    args = _surface_child_args(tmp_path)
    for index in range(3):
        _add_parent_surface(
            tmp_path / f"wrfout_d03_1974-04-03_{12 + index:02d}_00_00",
            ny=18, nx=20, lu_water_column=8)
    outdir = tmp_path / "child-run"
    outdir.mkdir(parents=True)
    (outdir / "wrfout_d02_1974-04-03_12_00_00").write_bytes(b"x")

    rc = cli_main(args)
    captured = capsys.readouterr()
    assert rc == 2
    assert "wrfout_d02_1974-04-03_12_00_00" in captured.err
    assert "--out" in captured.err


def test_an_empty_out_is_adopted_because_refusing_it_prevents_nothing(
        tmp_path, capsys, monkeypatch):
    """An empty directory holds no run to merge with and no receipt to lose.

    Refusing one names no breakage, and it is the state a partially
    cleaned retry can legitimately arrive in.
    """
    import gpuwm.offline_child_run as offline_child_run

    args = _point_args(tmp_path)
    for index in range(3):
        _add_parent_surface(
            tmp_path / f"wrfout_d01_1974-04-03_{12 + index:02d}_00_00",
            ny=18, nx=20, lu_water_column=8)
    (tmp_path / "child-run").mkdir(parents=True)
    monkeypatch.setattr(offline_child_run, "run",
                        lambda namespace: {"result": "PASS"})
    assert cli_main(args) == 0
    capsys.readouterr()
    assert (tmp_path / "child-run" / DERIVED_CHILD_CONFIG_NAME).is_file()


def test_the_runner_door_refuses_a_used_outdir_in_words(tmp_path):
    """``python -m gpuwm.offline_child_run`` gets the same sentence."""
    from gpuwm.offline_child_run import _create_output_root

    outdir = tmp_path / "child-run"
    _create_output_root(outdir)
    (outdir / "report.json").write_text("{}", encoding="utf-8")
    with pytest.raises(OfflineChildContractError) as caught:
        _create_output_root(outdir)
    message = str(caught.value)
    assert "report.json" in message and "--outdir" in message


# ---------------------------------------------------------------------
# The door a controller drives: a parent named by its run directory
# alone, and one machine-readable plan carrying the grid, the price and
# the cadence.
# ---------------------------------------------------------------------

def _named_checkpoint(tmp_path, config):
    """A restart under the name discovery actually looks for."""
    return _restart_evidence(
        tmp_path / "gpuwmrst_d01_1974-04-03_12_00_00.npz", config)


def _controller_point_args(tmp_path, *, restart, extra):
    start = datetime(1974, 4, 3, 12)
    for index in range(3):
        frame = tmp_path / f"wrfout_d01_1974-04-03_{12 + index:02d}_00_00"
        _history(frame, start + timedelta(hours=index), ny=18, nx=20)
        _give_the_parent_a_real_projection(frame, ny=18, nx=20)
    return [
        "downscale", str(tmp_path),
        "--parent-restart", restart,
        "--point", "39.5,-84.0",
        "--ratio", "1",
        "--child-levels", "4,2.5",
        "--hours", "0.25", "--output-interval-seconds", "900",
        *extra,
        "--out", str(tmp_path / "child-run"), "--dry-run"]


def _plan_document(tmp_path):
    import json as json_module

    path = downscale_plan_path(tmp_path / "child-run", dry_run=True)
    return json_module.loads(path.read_text(encoding="utf-8"))


@needs_netcdf_bridge
def test_parent_restart_latest_resolves_the_parents_own_newest_set(
        tmp_path, capsys):
    """A caller holding only the parent's run directory can name it.

    Every front door that lists finished runs holds the run directory and
    nothing else; demanding an exact checkpoint path made the door
    unreachable from there.
    """
    _named_checkpoint(
        tmp_path,
        dict(_SURFACE_PARENT_CONFIG, nx=20, ny=18, nz=2, grid_id=1,
             dt=3.0, run_seconds=7200.0, nested=False, specified=False))
    assert cli_main(_controller_point_args(
        tmp_path, restart="latest",
        extra=["--child-size", "12,10"])) == 0
    capsys.readouterr()
    plan = _plan_document(tmp_path)
    assert plan["parent"]["restart"] == str(
        tmp_path / "gpuwmrst_d01_1974-04-03_12_00_00.npz")
    assert plan["parent"]["run_dir"] == str(tmp_path)
    assert plan["parent"]["frames"] == 3
    assert plan["parent"]["domain"] == 1


@needs_netcdf_bridge
def test_parent_restart_latest_looks_above_a_wrfout_folder(tmp_path, capsys):
    """The prepared routes' own layout: frames under ``wrfout/``, sets above.

    Every ``gpuwm run-plan`` and ``gpuwm go`` forecast writes
    ``<run>/wrfout/wrfout_d01_*`` and ``<run>/gpuwmrst_d01_*``, so a door
    handed the frames folder found no checkpoint beside them and refused
    the one parent every desktop forecast produces.
    """
    _named_checkpoint(
        tmp_path,
        dict(_SURFACE_PARENT_CONFIG, nx=20, ny=18, nz=2, grid_id=1,
             dt=3.0, run_seconds=7200.0, nested=False, specified=False))
    frames = tmp_path / "wrfout"
    frames.mkdir()
    assert cli_main(_controller_point_args(
        frames, restart="latest", extra=["--child-size", "12,10"])) == 0
    capsys.readouterr()
    plan = _plan_document(frames)
    assert plan["parent"]["restart"] == str(
        tmp_path / "gpuwmrst_d01_1974-04-03_12_00_00.npz")
    assert plan["parent"]["run_dir"] == str(frames)

    # With no set in either place the refusal names both directories.
    bare = tmp_path / "bare"
    (bare / "wrfout").mkdir(parents=True)
    assert cli_main(_controller_point_args(
        bare / "wrfout", restart="latest",
        extra=["--child-size", "12,10"])) != 0
    message = capsys.readouterr().err
    assert str(bare / "wrfout") in message and f" or {bare};" in message
    assert "restart_interval_s" in message


def test_parent_restart_latest_names_the_setting_that_makes_one(
        tmp_path, capsys):
    """A parent with no checkpoint is refused with its remedy, at the door.

    The remedy is the PARENT's own setting: this parent has to be re-run
    to be downscalable, and a caller who learns that after paying for a
    forecast has learned it too late.
    """
    assert cli_main(_controller_point_args(
        tmp_path, restart="latest", extra=["--child-size", "12,10"])) != 0
    message = capsys.readouterr().err
    assert "no complete gpuwmrst checkpoint set" in message
    assert "restart_interval_s" in message
    # Not resume's sentence about --outdir: this caller is not resuming.
    assert "--outdir" not in message
    assert not (tmp_path / "child-run").exists()


def test_the_explicit_extent_price_is_taken_on_the_card_the_door_holds(
        tmp_path, capsys, monkeypatch):
    """``--child-size`` with ``--auto-vram`` prices the drawn extent on the
    measured card.

    The refusal that stood here ("--auto-vram requires --point sizing
    without an explicit --child-size") prevented nothing: an explicit
    extent priced against the card in front of the user is exactly what a
    drawn box needs.  The door measures once, prices the given child on
    that card with the probe's own profile, reports the fit under basis
    ``measured-local`` and fills ``gpu_sizing`` as the fitted route does.
    """
    import json as json_module

    import gpuwm.domain_wizard as wizard
    from gpuwm.core import preflight as pf
    from gpuwm.domain_wizard import SizingBudget

    given = tmp_path / "given"
    given.mkdir()
    _named_checkpoint(
        given,
        dict(_SURFACE_PARENT_CONFIG, nx=20, ny=18, nz=2, grid_id=1,
             dt=3.0, run_seconds=7200.0, nested=False, specified=False))
    observed = []
    device_profile = pf.card_local_memory_profile(None)
    budget = SizingBudget(10.0, 7 * 1024 ** 3, device_profile,
                          "measured fixture card", True)

    def probe(card, capacity):
        observed.append((card, capacity))
        return budget

    priced = []
    real_estimate = pf.estimate_experiment

    def estimate(exp, **kwargs):
        priced.append(kwargs)
        return real_estimate(exp, **kwargs)

    monkeypatch.setattr(wizard, "resolve_sizing_budget", probe)
    monkeypatch.setattr(pf, "estimate_experiment", estimate)
    assert cli_main(_controller_point_args(
        given, restart="latest",
        extra=["--child-size", "12,10", "--auto-vram"])) == 0
    out = capsys.readouterr()
    assert "--auto-vram" not in out.err
    plan = json_module.loads(out.out[out.out.index("{\n"):])
    assert observed == [(None, None)]
    assert plan["child_grid"]["nx"] == 12 and plan["child_grid"]["ny"] == 10
    memory = plan["memory"]
    assert memory["basis"] == "measured-local"
    assert memory["vram_gib"] == 10.0
    assert memory["free_bytes"] == budget.free_bytes
    assert memory["peak_envelope_bytes"] > 0
    assert memory["fits"] is (
        memory["peak_envelope_bytes"] <= memory["budget_bytes"])
    assert plan["gpu_sizing"]["basis"] == "measured-local"
    assert plan["gpu_sizing"]["free_bytes"] == budget.free_bytes
    # Priced ONCE, on the measured card's own profile and capacity.
    assert len(priced) == 1
    assert priced[0]["profile"] is device_profile
    assert priced[0]["vram_gib"] == 10.0
    streaming = plan["streaming"]
    assert streaming["mode"] == "resident"
    assert streaming["basis"] == "measured-local"
    assert streaming["machine_free_bytes"] == budget.free_bytes
    assert streaming["peak_envelope_bytes"] == memory["peak_envelope_bytes"]


def test_a_requested_extent_past_the_interior_shrinks_and_says_so():
    """A drawn box that reaches past the parent is placed, not refused."""
    from gpuwm.downscale import _fit_requested_extent
    from gpuwm.runplan import collect_warnings

    lat = np.linspace(38.0, 41.0, 31)[:, None] * np.ones((1, 41))
    lon = np.ones((31, 1)) * np.linspace(-86.0, -82.0, 41)[None, :]
    j0, i0 = _nearest_parent_index(lat, lon, 39.5, -84.0)
    parent = {"nx": 41, "ny": 31}
    records = []
    with collect_warnings(records):
        nx, ny = _fit_requested_extent(
            parent, j0=j0, i0=i0, ratio=3, child_nx=300, child_ny=240,
            lat=39.5, lon=-84.0)
    assert nx % 3 == 0 and ny % 3 == 0
    assert nx < 300 and ny < 240
    # The largest: one more refinement cell on either axis no longer fits.
    _centered_placement(parent, j0=j0, i0=i0, ratio=3, child_nx=nx, child_ny=ny)
    for wider in ((nx + 3, ny), (nx, ny + 3)):
        # The stencil-coverage gate refuses with the placement's own
        # ValueError; the shrink is what keeps a caller from meeting it.
        with pytest.raises((OfflineChildContractError, ValueError)):
            _centered_placement(parent, j0=j0, i0=i0, ratio=3,
                                child_nx=wider[0], child_ny=wider[1])
    assert len(records) == 1
    sentence = records[0]["action"]
    assert "300x240" in sentence and f"{nx}x{ny}" in sentence
    assert "does not fit inside the parent's interior" in sentence
    assert "41x31" in sentence

    # A size the parent holds is placed exactly, silently.
    records.clear()
    with collect_warnings(records):
        assert _fit_requested_extent(
            parent, j0=j0, i0=i0, ratio=3, child_nx=36, child_ny=24,
            lat=39.5, lon=-84.0) == (36, 24)
    assert records == []

    # An extent that is not a whole number of refinement cells rounds down.
    records.clear()
    with collect_warnings(records):
        assert _fit_requested_extent(
            parent, j0=j0, i0=i0, ratio=3, child_nx=35, child_ny=24,
            lat=39.5, lon=-84.0) == (33, 24)
    assert len(records) == 1 and "adjusted to 33x24" in records[0]["action"]

    # A point where even the smallest child cannot exist is refused.
    with pytest.raises(OfflineChildContractError, match="no child can be "
                       "centered"):
        _fit_requested_extent(
            {"nx": 41, "ny": 31}, j0=0, i0=0, ratio=3, child_nx=6,
            child_ny=6, lat=38.0, lon=-86.0)


@needs_netcdf_bridge
def test_the_dry_run_shrinks_a_drawn_extent_and_outlines_the_child(
        tmp_path, capsys):
    """The plan says what was asked, what runs, and where its corners are."""
    _named_checkpoint(
        tmp_path,
        dict(_SURFACE_PARENT_CONFIG, nx=20, ny=18, nz=2, grid_id=1,
             dt=3.0, run_seconds=7200.0, nested=False, specified=False))
    assert cli_main(_controller_point_args(
        tmp_path, restart="latest",
        extra=["--child-size", "60,60", "--vram-gib", "24"])) == 0
    capsys.readouterr()
    plan = _plan_document(tmp_path)
    grid = plan["child_grid"]
    assert grid["nx"] < 60 and grid["ny"] < 60
    shrink = [record for record in plan["warnings"]
              if "does not fit inside the parent's interior" in record["action"]]
    assert len(shrink) == 1
    assert "60x60" in shrink[0]["action"]
    assert f"{grid['nx']}x{grid['ny']}" in shrink[0]["action"]
    assert plan["parent_domain"] == 1 and plan["child_grid_id"] == 2
    outline = plan["child_outline"]
    cells = outline["parent_cells"]
    assert cells["i_first"] == grid["i_parent_start"]
    assert cells["j_first"] == grid["j_parent_start"]
    assert cells["i_last"] == cells["i_first"] + grid["nx"] // grid["ratio"] - 1
    assert cells["j_last"] == cells["j_first"] + grid["ny"] // grid["ratio"] - 1
    # Read off the parent's own XLAT/XLONG (the fixture's projection is a
    # regular lat/lon ladder, so the corners are its rows and columns).
    lat = np.linspace(38.0, 41.0, 18)
    lon = np.linspace(-86.0, -82.0, 20)
    sw, ne = outline["sw"], outline["ne"]
    assert sw == pytest.approx(
        [float(np.float32(lat[cells["j_first"] - 1])),
         float(np.float32(lon[cells["i_first"] - 1]))])
    assert ne == pytest.approx(
        [float(np.float32(lat[cells["j_last"] - 1])),
         float(np.float32(lon[cells["i_last"] - 1]))])
    assert outline["se"][0] == sw[0] and outline["se"][1] == ne[1]
    assert outline["nw"][0] == ne[0] and outline["nw"][1] == sw[1]
    assert "parent mass points" in outline["basis"]


@needs_netcdf_bridge
def test_a_downscaled_run_is_accepted_as_a_parent(tmp_path, capsys):
    """The chain: a child's own frames and sets seed a grid 3 grandchild.

    A downscaled run writes ``wrfout_d02_*`` at its root and
    ``gpuwmrst_d02_<instant>.npz`` beside them.  ``--parent-domain 2``
    selects those frames, ``--parent-restart latest`` discovers those
    sets, the physics evidence's grid id is 2 and the derived grandchild
    is grid 3.
    """
    from gpuwm.config import load_config as read_config

    child_run = tmp_path / "child-run"
    child_run.mkdir()
    start = datetime(1974, 4, 3, 12)
    for index in range(3):
        frame = child_run / f"wrfout_d02_1974-04-03_{12 + index:02d}_00_00"
        _history(frame, start + timedelta(hours=index), ny=18, nx=20)
        _give_the_parent_a_real_projection(frame, ny=18, nx=20)
        with netCDF4.Dataset(frame, "a") as dataset:
            dataset.GRID_ID = np.int32(2)
    _restart_evidence(
        child_run / "gpuwmrst_d02_1974-04-03_14_00_00.npz",
        dict(_SURFACE_PARENT_CONFIG, nx=20, ny=18, nz=2, grid_id=2,
             dt=3.0, run_seconds=7200.0, nested=False, specified=True))
    assert cli_main([
        "downscale", str(child_run), "--parent-domain", "2",
        "--parent-restart", "latest", "--point", "39.5,-84.0",
        "--ratio", "1", "--child-size", "12,10", "--child-levels", "4,2.5",
        "--hours", "0.25", "--output-interval-seconds", "900",
        "--out", str(tmp_path / "grandchild" / "child-run"), "--dry-run"]) == 0
    capsys.readouterr()
    plan = _plan_document(tmp_path / "grandchild")
    assert plan["parent_domain"] == 2
    assert plan["parent"]["domain"] == 2
    assert plan["parent"]["run_dir"] == str(child_run)
    assert plan["parent"]["restart"] == str(
        child_run / "gpuwmrst_d02_1974-04-03_14_00_00.npz")
    assert plan["physics_binding"]["domain_id"] == 2
    assert plan["child_grid_id"] == 3
    grandchild = read_config(plan["child_config"])
    assert grandchild.grid_id == 3
    assert [Path(frame).name[:10] for frame in plan["parent_frames"]] == [
        "wrfout_d02"] * 3


@needs_netcdf_bridge
def test_plan_document_prices_the_child_on_both_sizing_routes(
        tmp_path, capsys):
    """One document, both routes, and the memory number is the engine's.

    An explicit extent used to be the one route that produced no price at
    all, so a caller offering "explicit size" had nothing to show and no
    way to warn before the allocation refused.
    """
    given = tmp_path / "given"
    given.mkdir()
    _named_checkpoint(
        given,
        dict(_SURFACE_PARENT_CONFIG, nx=20, ny=18, nz=2, grid_id=1,
             dt=3.0, run_seconds=7200.0, nested=False, specified=False))
    assert cli_main(_controller_point_args(
        given, restart="latest",
        extra=["--child-size", "12,10", "--vram-gib", "24"])) == 0
    capsys.readouterr()
    plan = _plan_document(given)
    assert plan["schema"] == DOWNSCALE_PLAN_SCHEMA
    assert plan["child_grid"]["nx"] == 12 and plan["child_grid"]["ny"] == 10
    assert plan["child_grid"]["ratio"] == 1
    assert plan["child_grid"]["run_seconds"] == 900.0
    assert plan["child_grid"]["output_interval_s"] == 900.0
    memory = plan["memory"]
    assert memory["basis"] == "explicit-size"
    assert memory["vram_gib"] == 24.0
    assert memory["peak_envelope_bytes"] > 0
    assert memory["budget_bytes"] < memory["free_bytes"]
    assert memory["fits"] is (
        memory["peak_envelope_bytes"] <= memory["budget_bytes"])
    assert plan["cadence"]["seconds"] == 3600.0
    assert plan["cadence"]["accepted_parent_cadence"] is True
    assert plan["cadence"]["max_boundary_interval_seconds"] == 3600.0
    # 3600 s is coarser than the 900 s guidance, so the sentence is here
    # rather than only on a stderr line a GUI never sees.
    assert "coarser than" in plan["cadence"]["warning"]
    assert any("coarser than" in record["action"]
               for record in plan["warnings"])

    fitted = tmp_path / "fitted"
    fitted.mkdir()
    _named_checkpoint(
        fitted,
        dict(_SURFACE_PARENT_CONFIG, nx=20, ny=18, nz=2, grid_id=1,
             dt=3.0, run_seconds=7200.0, nested=False, specified=False))
    assert cli_main(_controller_point_args(
        fitted, restart="latest", extra=["--vram-gib", "24"])) == 0
    capsys.readouterr()
    memory = _plan_document(fitted)["memory"]
    assert memory["basis"] == "capacity"
    assert memory["fits"] is True
    assert memory["peak_envelope_bytes"] <= memory["budget_bytes"]


def test_the_review_refuses_a_child_clock_that_is_not_whole_steps(
        tmp_path, capsys):
    """A hand-written child config whose output or restart interval is not
    a whole number of dt steps is refused at plan review, in the runner's
    own sentence, instead of after the run has started and reserved
    --out.  One function (offline_child_run.child_cadence) answers for
    both doors."""
    start = datetime(1974, 4, 3, 12)
    for index in range(3):
        _history(tmp_path / f"wrfout_d03_1974-04-03_{12 + index:02d}_00_00",
                 start + timedelta(hours=index), ny=18, nx=20)
    namelist = tmp_path / "namelist.input"
    namelist.write_text("&physics\n mp_physics = 8,\n/\n", encoding="utf-8")
    parent = {"nx": 20, "ny": 18, "dx": 1000.0, "dy": 1000.0}

    def review(child: dict, out: str) -> int:
        child_toml = tmp_path / f"{out}.toml"
        child_toml.write_text(_render_child_toml(child), encoding="utf-8")
        return cli_main([
            "downscale", str(tmp_path), "--parent-domain", "3",
            "--parent-namelist", str(namelist),
            "--child-config", str(child_toml), "--ratio", "1",
            "--i-parent-start", "4", "--j-parent-start", "4",
            "--accept-parent-cadence",
            "--out", str(tmp_path / out), "--dry-run"])

    # dt is 5 s: 302 s of output interval is 60.4 steps.
    uneven_output = _derive_child_run_config(
        _PARENT_CONFIG, parent=parent, ratio=1, child_nx=12, child_ny=10,
        run_seconds=600.0, output_interval_s=302.0)
    assert review(uneven_output, "uneven-output") == 2
    captured = capsys.readouterr()
    assert "output_interval_s/dt must be a positive integer" in captured.err
    # The way out rides in the same sentence: the multiple to choose.
    assert ("set output_interval_s to a value that is a whole multiple of dt"
            in captured.err)
    assert "downscale_plan" not in captured.out
    assert "Traceback" not in captured.err

    # 7.5 s of restart interval is 1.5 steps.
    uneven_restart = dict(_derive_child_run_config(
        _PARENT_CONFIG, parent=parent, ratio=1, child_nx=12, child_ny=10,
        run_seconds=600.0, output_interval_s=300.0),
        restart_interval_s=7.5)
    assert review(uneven_restart, "uneven-restart") == 2
    captured = capsys.readouterr()
    assert "restart_interval_s/dt must be a positive integer" in captured.err
    assert "Traceback" not in captured.err

    # The same child on a whole-step clock reviews.
    whole = dict(uneven_restart, restart_interval_s=300.0)
    assert review(whole, "whole") == 0
    assert "downscale_plan" in capsys.readouterr().out


@needs_netcdf_bridge
def test_the_memory_block_is_judged_on_the_budget_the_decision_used(
        tmp_path, capsys, monkeypatch):
    """``memory.fits`` and ``streaming.mode`` are two readings of one comparison.

    The fit ceiling withholds headroom the fit loop needs (0.5 GiB, then
    the larger of 0.25 GiB and five percent); the ``[tiles]`` admission
    judges the configured envelope against free minus 0.5 GiB.  On the
    user's figures the two differed by 0.32 GiB, a window in which the plan
    said ``fits false`` about a child its own decision ran resident.  Once
    a decision exists the memory block reports the decision's budget, so a
    front end reading either block reads the engine's one answer.
    """
    import json as json_module

    import gpuwm.domain_wizard as wizard
    from gpuwm.core import preflight as pf
    from gpuwm.domain_wizard import SizingBudget

    device_profile = pf.card_local_memory_profile(None)
    parent_config = dict(_SURFACE_PARENT_CONFIG, nx=20, ny=18, nz=2,
                         grid_id=1, dt=3.0, run_seconds=7200.0,
                         nested=False, specified=False)

    def review(name: str, free_bytes: int, *tiles: str) -> dict:
        given = tmp_path / name
        given.mkdir()
        _named_checkpoint(given, parent_config)
        budget = SizingBudget(10.0, int(free_bytes), device_profile,
                              "measured fixture card", True)
        monkeypatch.setattr(wizard, "resolve_sizing_budget",
                            lambda card, capacity: budget)
        assert cli_main(_controller_point_args(
            given, restart="latest",
            extra=["--child-size", "12,10", "--auto-vram", *tiles])) == 0
        out = capsys.readouterr()
        return json_module.loads(out.out[out.out.index("{\n"):])

    # [tiles] auto, the block the user's child inherited from its parent:
    # the decision goes through the resident admission and the memory
    # block reports the admission's budget.
    roomy = review("roomy", 7 * 1024 ** 3, "--tiles", "auto")
    memory, streaming = roomy["memory"], roomy["streaming"]
    assert streaming["mode"] == "resident"
    assert "envelope fits" in streaming["why"]
    assert memory["budget_bytes"] == streaming["budget_bytes"]
    assert memory["budget_bytes"] == 7 * 1024 ** 3 - pf.EXTERNAL_MARGIN_BYTES
    assert memory["fits"] is True
    assert memory["fits"] is (
        memory["peak_envelope_bytes"] <= memory["budget_bytes"])
    envelope = int(memory["peak_envelope_bytes"])

    # A card whose free figure puts the envelope INSIDE the window: the
    # admission (free minus 0.5 GiB) says resident, the fit ceiling (a
    # further 0.25 GiB down) would have said no.  One answer.
    free = envelope + pf.EXTERNAL_MARGIN_BYTES + 100 * 1024 ** 2
    narrow = review("narrow", free, "--tiles", "auto")
    memory, streaming = narrow["memory"], narrow["streaming"]
    assert memory["peak_envelope_bytes"] == envelope
    assert streaming["mode"] == "resident"
    assert memory["budget_bytes"] == streaming["budget_bytes"]
    assert memory["fits"] is True
    assert memory["free_bytes"] == free

    # No [tiles] block: the child is resident by configuration, no
    # admission is taken, and the memory block keeps the fit ceiling it
    # always reported (free minus the margin minus the fit headroom).
    plain = review("plain", 7 * 1024 ** 3)
    memory, streaming = plain["memory"], plain["streaming"]
    assert streaming["mode"] == "resident"
    assert streaming["budget_bytes"] is None
    assert memory["budget_bytes"] == _budget_bytes(10.0, 7 * 1024 ** 3)[1]
    assert memory["fits"] is (
        memory["peak_envelope_bytes"] <= memory["budget_bytes"])


@needs_netcdf_bridge
def test_the_parents_restart_is_the_member_of_the_domain_the_frames_come_from(
        tmp_path, capsys):
    """A checkpoint set is one instant of every domain written together.

    ``--parent-restart latest`` with ``--parent-domain 2`` binds the d02
    frames' physics from the set's d02 member.  The set's root member (the
    lowest grid id) describes another grid, and handing it on made the
    door refuse its own nest with "evidence domain 1 does not match
    history GRID_ID=2".  A set with no member for the frames' domain is
    refused naming the members it has and the way out.
    """
    from gpuwm.config import load_config as read_config

    def nest_frames(run: Path, grid_id: int) -> None:
        run.mkdir()
        start = datetime(1974, 4, 3, 12)
        for index in range(3):
            frame = run / f"wrfout_d{grid_id:02d}_1974-04-03_{12 + index:02d}_00_00"
            _history(frame, start + timedelta(hours=index), ny=18, nx=20)
            _give_the_parent_a_real_projection(frame, ny=18, nx=20)
            with netCDF4.Dataset(frame, "a") as dataset:
                dataset.GRID_ID = np.int32(grid_id)

    root_config = dict(_SURFACE_PARENT_CONFIG, nx=40, ny=36, nz=2, grid_id=1,
                       dt=9.0, run_seconds=7200.0, nested=False, specified=True)
    nest_config = dict(_SURFACE_PARENT_CONFIG, nx=20, ny=18, nz=2, grid_id=2,
                       dt=3.0, run_seconds=7200.0, nested=False, specified=True)

    def door(run: Path, domain: int, out: str) -> int:
        return cli_main([
            "downscale", str(run), "--parent-domain", str(domain),
            "--parent-restart", "latest", "--point", "39.5,-84.0",
            "--ratio", "1", "--child-size", "12,10", "--child-levels", "4,2.5",
            "--hours", "0.25", "--output-interval-seconds", "900",
            "--out", str(tmp_path / out / "child-run"), "--dry-run"])

    tree = tmp_path / "tree-run"
    nest_frames(tree, 2)
    _restart_evidence(tree / "gpuwmrst_d01_1974-04-03_14_00_00__tree.npz",
                      root_config)
    nest_member = _restart_evidence(
        tree / "gpuwmrst_d02_1974-04-03_14_00_00__tree.npz", nest_config)
    assert door(tree, 2, "grandchild") == 0
    capsys.readouterr()
    plan = _plan_document(tmp_path / "grandchild")
    assert plan["parent"]["restart"] == str(nest_member)
    assert plan["parent_domain"] == 2
    assert plan["physics_binding"]["domain_id"] == 2
    assert plan["child_grid_id"] == 3
    assert read_config(plan["child_config"]).grid_id == 3

    # Frames of a domain the set never wrote: refused at the door, with
    # the members present and both ways out.
    orphan = tmp_path / "orphan-run"
    nest_frames(orphan, 3)
    _restart_evidence(orphan / "gpuwmrst_d01_1974-04-03_14_00_00__tree.npz",
                      root_config)
    _restart_evidence(orphan / "gpuwmrst_d02_1974-04-03_14_00_00__tree.npz",
                      nest_config)
    assert door(orphan, 3, "orphan-child") == 2
    captured = capsys.readouterr()
    assert "has no d03 member" in captured.err
    assert "d01, d02" in captured.err
    assert "--parent-domain" in captured.err and "--parent-restart" in captured.err
    assert "Traceback" not in captured.err


# ---------------------------------------------------------------------------
# Products: a downscaled forecast is drawn the way a forecast is
# ---------------------------------------------------------------------------

def test_the_plan_says_which_products_the_child_will_be_drawn_as(
        tmp_path, capsys):
    """The reviewed plan carries the product spec, defaulted off the
    forecast door rather than off a second literal kept here."""
    from gpuwm.go_cli import DEFAULT_RENDER_PRODUCTS

    plan = _dry_run_plan(tmp_path, capsys, ["--accept-parent-cadence"])
    assert plan["render_products"] == DEFAULT_RENDER_PRODUCTS == "all"

    off = _dry_run_plan(tmp_path / "off", capsys,
                        ["--accept-parent-cadence",
                         "--render-products", "none"])
    assert off["render_products"] == "none"

    chosen = _dry_run_plan(tmp_path / "chosen", capsys,
                           ["--accept-parent-cadence",
                            "--render-products", "composite_reflectivity,mslp_10m_winds"])
    assert chosen["render_products"] == "composite_reflectivity,mslp_10m_winds"


def _downscale_with_receipts(tmp_path, monkeypatch, *, extra=(),
                             rendered=True):
    """``gpuwm downscale`` through its real runner, integration replaced.

    The child's receipts -- manifest, event stream, the armed render --
    are published exactly as a real child publishes them; only the GPU
    forecast between them is the stand-in, so what this exercises is the
    door, the runner's own finalize, and the shared render stage.
    """

    import json as _json

    import gpuwm.go_cli as go_cli
    import gpuwm.offline_child_run as offline_child_run

    calls = []

    def fake_render_stage(plan, *, explain, observer=None, door="go"):
        calls.append(dict(plan))
        return rendered

    def fake_child(args, progress):
        outdir = Path(args.outdir)
        progress.start(
            outdir=outdir, child_config=Path(args.child_config),
            ratio=int(args.parent_grid_ratio),
            start_time=datetime(1974, 4, 3, 12),
            parent={"run_dir": str(tmp_path), "frames": 3},
            name="Downscale of parent")
        progress.arm_render(outdir=outdir,
                            render_products=args.render_products)
        progress.emit("stage_started", stage="forecast", phase="integrate")
        return {"result": "PASS", "outputs": []}

    monkeypatch.setattr(go_cli, "_render_stage", fake_render_stage)
    monkeypatch.setattr(offline_child_run, "_run", fake_child)

    start = datetime(1974, 4, 3, 12)
    for index in range(3):
        _history(tmp_path / f"wrfout_d03_1974-04-03_{12 + index:02d}_00_00",
                 start + timedelta(hours=index), ny=18, nx=20)
    namelist = tmp_path / "namelist.input"
    namelist.write_text("&physics\n mp_physics = 8,\n/\n", encoding="utf-8")
    child_toml = tmp_path / "child.toml"
    merged = _derive_child_run_config(
        _PARENT_CONFIG,
        parent={"nx": 20, "ny": 18, "dx": 1000.0, "dy": 1000.0},
        ratio=1, child_nx=12, child_ny=10,
        run_seconds=600.0, output_interval_s=300.0)
    child_toml.write_text(_render_child_toml(merged), encoding="utf-8")
    out = tmp_path / "child-run"
    code = cli_main([
        "downscale", str(tmp_path), "--parent-domain", "3",
        "--parent-namelist", str(namelist),
        "--child-config", str(child_toml), "--ratio", "1",
        "--i-parent-start", "4", "--j-parent-start", "4",
        "--accept-parent-cadence", "--out", str(out), *extra])
    events = [
        _json.loads(line) for line
        in (out / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()]
    return code, out, calls, events


def test_a_finished_child_is_rendered_once_into_png_beside_its_frames(
        tmp_path, monkeypatch, capsys):
    code, out, calls, events = _downscale_with_receipts(tmp_path, monkeypatch)
    capsys.readouterr()
    assert code == 0

    # One render, on the child's own frames, into the child's own folder.
    assert len(calls) == 1
    plan = calls[0]
    assert plan["run"] == out
    assert plan["wrfout_dir"] == out
    assert plan["render"] == out / "png"
    assert plan["render_products"] == "all"

    # And a reader of this run sees the sequence a forecast writes.
    staged = [(event["event"], event.get("stage"), event.get("phase"))
              for event in events]
    assert ("stage_started", "finalize", "render") in staged
    finished = [event for event in events
                if event["event"] == "stage_finished"
                and event["stage"] == "finalize"]
    assert len(finished) == 1
    assert events[-1]["event"] == "completed"


def test_render_products_none_draws_nothing_and_says_nothing(
        tmp_path, monkeypatch, capsys):
    code, _out, calls, events = _downscale_with_receipts(
        tmp_path, monkeypatch, extra=["--render-products", "none"])
    capsys.readouterr()
    assert code == 0
    assert calls == []
    assert not [event for event in events
                if event.get("stage") == "finalize"]


def test_a_requested_product_set_that_drew_nothing_is_a_refusal(
        tmp_path, monkeypatch, capsys):
    """The chain's own refusal, on this route: the finished forecast is
    kept and the command that draws it by hand is named."""
    code, out, calls, _events = _downscale_with_receipts(
        tmp_path, monkeypatch, rendered=False)
    err = capsys.readouterr().err
    assert code == 2
    assert len(calls) == 1
    assert "were not produced" in err
    assert "render" in err
    assert "child-run/png" in err.replace(chr(92), "/")
    # The finished child is evidence; a render that drew nothing does not
    # take its frames away.
    assert out.is_dir()


def test_the_door_hands_the_runner_the_spec_it_resolved(
        tmp_path, monkeypatch, capsys):
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
    merged = _derive_child_run_config(
        _PARENT_CONFIG,
        parent={"nx": 20, "ny": 18, "dx": 1000.0, "dy": 1000.0},
        ratio=1, child_nx=12, child_ny=10,
        run_seconds=600.0, output_interval_s=300.0)
    child_toml.write_text(_render_child_toml(merged), encoding="utf-8")
    door = [
        "downscale", str(tmp_path), "--parent-domain", "3",
        "--parent-namelist", str(namelist),
        "--child-config", str(child_toml), "--ratio", "1",
        "--i-parent-start", "4", "--j-parent-start", "4",
        "--accept-parent-cadence"]
    assert cli_main(door + ["--out", str(tmp_path / "a")]) == 0
    assert captured["namespace"].render_products == "all"
    assert cli_main(door + ["--out", str(tmp_path / "b"),
                            "--render-products", "none"]) == 0
    assert captured["namespace"].render_products == "none"
    capsys.readouterr()

    # The engine runner speaks the same flag, and draws nothing without it.
    from gpuwm.offline_child_run import _parser
    base = ["--parent-history", "x", "--parent-restart", "r",
            "--child-config", "c", "--parent-grid-ratio", "3",
            "--i-parent-start", "1", "--j-parent-start", "1",
            "--max-boundary-interval-seconds", "900", "--outdir", "o"]
    assert _parser().parse_args(base).render_products is None
    assert _parser().parse_args(
        base + ["--render-products", "all"]).render_products == "all"


def test_the_childs_first_frame_is_drawn_while_it_is_still_running(
        tmp_path, monkeypatch):
    """The early render is armed off the SAME plan the finalize render
    runs, and the first committed frame dispatches it."""
    from gpuwm.first_products import FirstProducts
    from gpuwm.offline_child_run import _ChildProgress

    dispatched = []
    monkeypatch.setattr(
        FirstProducts, "frame_committed",
        lambda self, **fields: bool(dispatched.append(fields)) or True)

    out = tmp_path / "child-run"
    out.mkdir()
    child_toml = tmp_path / "child.toml"
    child_toml.write_text("# child\n", encoding="utf-8")
    progress = _ChildProgress()
    progress.start(outdir=out, child_config=child_toml, ratio=3,
                   start_time=datetime(1974, 4, 3, 12),
                   parent={"run_dir": str(tmp_path), "frames": 1},
                   name="Downscale of parent")
    plan = progress.arm_render(outdir=out, render_products="all")
    assert plan == {"run": out, "wrfout_dir": out, "render": out / "png",
                    "render_products": "all"}
    assert progress.first_products.render_dir == out / "png"
    frame = str(out / "wrfout_d02_1974-04-03_12_00_00")
    progress.output_committed(domain=2, valid_time="1974-04-03T12:00:00Z",
                              path=frame, bytes=1)
    progress.close()
    assert dispatched == [{"domain": 2,
                           "valid_time": "1974-04-03T12:00:00Z",
                           "path": frame}]

    # No products asked for is no early render and no plan, which is the
    # one place that answer lives.
    quiet = _ChildProgress()
    assert quiet.arm_render(outdir=out, render_products="none") is None
    assert quiet.render_plan is None and quiet.first_products is None


# ---------------------------------------------------------------------------
# Products: what is admitted, what is refused, and what is never drawn
# ---------------------------------------------------------------------------

def _child_door(tmp_path):
    """The argv of one real ``gpuwm downscale`` run, fixtures written."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    start = datetime(1974, 4, 3, 12)
    for index in range(3):
        _history(tmp_path / f"wrfout_d03_1974-04-03_{12 + index:02d}_00_00",
                 start + timedelta(hours=index), ny=18, nx=20)
    namelist = tmp_path / "namelist.input"
    namelist.write_text("&physics\n mp_physics = 8,\n/\n", encoding="utf-8")
    child_toml = tmp_path / "child.toml"
    merged = _derive_child_run_config(
        _PARENT_CONFIG,
        parent={"nx": 20, "ny": 18, "dx": 1000.0, "dy": 1000.0},
        ratio=1, child_nx=12, child_ny=10,
        run_seconds=600.0, output_interval_s=300.0)
    child_toml.write_text(_render_child_toml(merged), encoding="utf-8")
    return [
        "downscale", str(tmp_path), "--parent-domain", "3",
        "--parent-namelist", str(namelist),
        "--child-config", str(child_toml), "--ratio", "1",
        "--i-parent-start", "4", "--j-parent-start", "4",
        "--accept-parent-cadence", "--out", str(tmp_path / "child-run")]


def _count_frame_opens(monkeypatch):
    """Every archive frame this command opens, counted at netCDF4."""

    opened = []
    real = netCDF4.Dataset

    def counting(path, *args, **kwargs):
        opened.append(str(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(netCDF4, "Dataset", counting)
    return opened


def test_a_computer_that_cannot_draw_is_refused_before_anything_is_opened(
        tmp_path, monkeypatch, capsys):
    """The renderer-missing refusal is an ADMISSION fact.

    It used to fire inside the engine runner: nine archived frames had
    been opened, ``--out`` created and the plan document written into it
    by then, so the remedy the refusal names -- run gpuwm setup, then
    repeat this command -- met "--out already holds a child run's
    output" on the repeat.
    """

    import gpuwm.go_cli as go_cli

    monkeypatch.setattr(go_cli, "render_extra_missing",
                        lambda: "no rw_wrfbatch is staged on this computer")
    door = _child_door(tmp_path)
    opened = _count_frame_opens(monkeypatch)
    assert cli_main(door) == 2
    err = capsys.readouterr().err
    assert "gpuwm setup" in err and "--render-products none" in err
    assert "Traceback" not in err
    # Nothing was read and nothing was created, so the remedy works.
    assert opened == []
    assert not (tmp_path / "child-run").exists()


def test_a_product_the_catalog_does_not_carry_is_refused_at_plan_review(
        tmp_path, monkeypatch, capsys):
    """An unknown slug is a renderer that exits nonzero, which is a
    traceback after a whole forecast.  It is a sentence before one."""

    import gpuwm.go_cli as go_cli
    import gpuwm.runplan as runplan

    monkeypatch.setattr(go_cli, "render_extra_missing", lambda: None)
    monkeypatch.setattr(runplan, "render_catalog", lambda: {
        "engine": "rust",
        "products": [{"name": "composite_reflectivity"},
                     {"name": "mslp_10m_winds"}],
        "group_keywords": ["severe"]})
    door = _child_door(tmp_path)
    opened = _count_frame_opens(monkeypatch)
    assert cli_main(door + ["--render-products", "mslp,composite_reflectivity"]) == 2
    err = capsys.readouterr().err
    assert "'mslp'" in err and "--list-products" in err
    assert "Traceback" not in err
    assert opened == []
    assert not (tmp_path / "child-run").exists()

    # A catalog name, a group keyword, an alias and the whole catalog are
    # all admitted, so the check refuses spellings rather than vocabulary.
    for spec in ("composite_reflectivity", "severe", "refl", "all", "none",
                 "var:T2", "ALL", "SEVERE", "composite_reflectivity, mslp_10m_winds",
                 "xsec:wa=1,2,5,10@5", "composite_reflectivity,xsec:QCLOUD/wa=1,2,3",
                 "xsec:wa=1,2,5,mslp_10m_winds"):
        assert go_cli.unknown_render_products(spec) == [], spec
    # A level list is the renderer's own comma grammar; a slug after it is
    # still a slug, and an unknown one is still refused.
    assert go_cli.unknown_render_products("xsec:wa=1,2,bogus") == ["bogus"]


def test_early_pictures_are_counted_from_the_receipt_the_early_render_writes():
    """The receipt lists what was published under ``written``; the count
    the failure sentence carries reads that list, bounded by the same
    wait the finalize stage uses."""

    from gpuwm.offline_child_run import _ChildProgress

    class _Early:
        def __init__(self, receipt):
            self.receipt = receipt
            self.timeouts = []

        def wait(self, timeout="default"):
            self.timeouts.append(timeout)
            return self.receipt

    progress = _ChildProgress()
    assert progress.early_pictures() == 0
    progress._first_products = _Early({"written": ["a.png", "b.png"], "frame": "f"})
    assert progress.early_pictures() == 2
    assert progress._first_products.timeouts == ["default"]
    progress._first_products = _Early(None)
    assert progress.early_pictures() == 0


def test_the_admission_check_says_nothing_when_it_cannot_ask(monkeypatch):
    """A box with no renderer has a different refusal, with a different
    remedy; this one must not answer for it."""

    import gpuwm.go_cli as go_cli
    import gpuwm.runplan as runplan

    monkeypatch.setattr(runplan, "render_catalog", lambda: {
        "engine": None, "products": None, "error": "no renderer is staged"})
    assert go_cli.unknown_render_products("whatever,it,says") == []


def _fake_render_runner(monkeypatch, *, pictures=1):
    """The early render's subprocess, replaced by one that draws.

    It writes into the scratch directory the command names and leaves the
    renderer's OWN scratch sibling behind, exactly as ``rw_wrfbatch``
    does (``gpuwm.render.scratch_root_for``), so the cleanup of that
    sibling is measured rather than assumed.
    """

    import subprocess as _subprocess

    import gpuwm.first_products as first_products

    def runner(command):
        out = Path(command[command.index("--out") + 1])
        (out / "d02" / "composite_reflectivity" / "1974-04-03").mkdir(
            parents=True, exist_ok=True)
        for index in range(pictures):
            (out / "d02" / "composite_reflectivity" / "1974-04-03"
             / f"picture-{index}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        sibling = out.with_name(out.name + ".render-scratch")
        sibling.mkdir(parents=True, exist_ok=True)
        (sibling / "working.bin").write_bytes(b"0" * 16)
        return _subprocess.CompletedProcess(list(command), 0, "", "")

    monkeypatch.setattr(first_products, "_run_render", runner)


def _downscale_drawing(tmp_path, monkeypatch, *, result="PASS",
                       stage_failure=None, extra=()):
    """``gpuwm downscale`` with a real early render and a stub forecast.

    The child's receipts, its armed early render and its report are
    published exactly as a real child publishes them; only the GPU
    integration between them is the stand-in.
    """

    import json as _json

    import gpuwm.go_cli as go_cli
    import gpuwm.offline_child_run as offline_child_run

    calls = []

    def fake_render_stage(plan, *, explain, observer=None, door="go"):
        calls.append({**plan, "door": door})
        if stage_failure is not None:
            raise go_cli.GoStageFailed(stage_failure)
        return True

    def fake_child(args, progress):
        outdir = Path(args.outdir)
        progress.start(
            outdir=outdir, child_config=Path(args.child_config),
            ratio=int(args.parent_grid_ratio),
            start_time=datetime(1974, 4, 3, 12),
            parent={"run_dir": str(tmp_path), "frames": 3},
            name="Downscale of parent")
        progress.arm_render(outdir=outdir,
                            render_products=args.render_products)
        progress.emit("stage_started", stage="forecast", phase="integrate")
        frame = outdir / "wrfout_d02_1974-04-03_12_00_00"
        frame.write_bytes(b"CDF frame")
        progress.output_committed(
            domain=2, valid_time="1974-04-03T12:00:00Z",
            path=str(frame), bytes=frame.stat().st_size)
        report = {"result": result, "outputs": [str(frame)]}
        offline_child_run._publish_report(report, outdir)
        return report

    monkeypatch.setattr(go_cli, "render_extra_missing", lambda: None)
    monkeypatch.setattr(go_cli, "_render_stage", fake_render_stage)
    monkeypatch.setattr(offline_child_run, "_run", fake_child)
    _fake_render_runner(monkeypatch)

    out = tmp_path / "child-run"
    code = cli_main(_child_door(tmp_path) + list(extra))
    events = [
        _json.loads(line) for line
        in (out / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()]
    report = _json.loads((out / "report.json").read_text(encoding="utf-8"))
    return code, out, calls, events, report


def test_the_early_render_leaves_the_picture_tree_and_nothing_else(
        tmp_path, monkeypatch, capsys):
    """Every early render used to leave the renderer's own scratch
    directory inside the picture tree it had just published into."""

    code, out, _calls, _events, report = _downscale_drawing(
        tmp_path, monkeypatch)
    capsys.readouterr()
    assert code == 0
    published = sorted(path.name for path in (out / "png").rglob("*.png"))
    assert published == ["picture-0.png"]
    leftovers = [path for path in (out / "png").rglob("*")
                 if ".render-scratch" in path.name
                 or ".first-products-scratch" in path.name]
    assert leftovers == []
    assert report["products"]["status"] == "DRAWN"


def test_a_child_that_did_not_pass_publishes_no_picture(
        tmp_path, monkeypatch, capsys):
    """THE DECISION: a child that does not pass draws nothing.

    The early render draws the analysis frame while the run still looks
    healthy; a run that then refuses itself must not leave pictures as
    its only artifact that does not carry the verdict.  So what the
    early render published is withdrawn -- and the withdrawal waits for
    it, which is also what stops the render subprocess outliving the
    process that armed it.
    """

    from gpuwm.first_products import FIRST_PRODUCTS_RECEIPT

    code, out, calls, events, report = _downscale_drawing(
        tmp_path, monkeypatch, result="FAIL")
    capsys.readouterr()
    assert code == 1
    # The finalize render never ran; the early one was undone.
    assert calls == []
    assert not (out / "png").exists()
    assert not (out / "png" / FIRST_PRODUCTS_RECEIPT).exists()
    # The frames and the report are evidence and stay.
    assert (out / "wrfout_d02_1974-04-03_12_00_00").is_file()
    assert report["result"] == "FAIL"
    assert report["products"]["status"] == "WITHDRAWN"
    assert "did not pass" in report["products"]["reason"]
    withdrawn = [event for event in events
                 if event.get("code") == "early_render_withdrawn"]
    assert len(withdrawn) == 1 and withdrawn[0]["pictures"] == 1
    assert "publishes no picture" in withdrawn[0]["message"]
    assert events[-1]["event"] == "completed"
    # Nothing is still drawing when the door returns.
    assert [thread for thread in threading.enumerate()
            if thread.name == "gpuwm-first-products"
            and thread.is_alive()] == []


def test_a_render_stage_that_exited_nonzero_is_this_doors_refusal(
        tmp_path, monkeypatch, capsys):
    """A nonzero render stage raises GoStageFailed, which no CLI handler
    knows: the door died at exit 1 with a traceback over a report.json
    that said PASS."""

    code, out, calls, events, report = _downscale_drawing(
        tmp_path, monkeypatch, stage_failure=3)
    captured = capsys.readouterr()
    assert code == 2
    assert "Traceback" not in captured.err
    assert "render stage exited 3" in captured.err
    assert "gpuwm.cli render" in captured.err
    # The forecast passed and its frames are on disk; the pictures did
    # not.  One run, two verdicts, in the one document that says so.
    assert report["result"] == "PASS"
    assert report["products"]["status"] == "FAILED"
    assert "exited 3" in report["products"]["reason"]
    assert "--products" in report["products"]["render_command"]
    assert events[-1]["event"] == "failed"
    assert events[-1]["stage"] == "finalize"
    # And no record of this run names a command its reader did not type.
    assert not [event for event in events
                if "gpuwm go" in json.dumps(event)]
    # And the stage was told which door is running it.
    assert calls[0]["door"] == "downscale"


def test_the_stage_failure_line_names_the_door_that_is_running(capsys):
    """`go: stopped at render` under `gpuwm downscale` sends its reader
    to another command's documentation."""

    import sys

    from gpuwm.go_cli import GoStageFailed, _run_stage

    failing = [sys.executable, "-c", "raise SystemExit(4)"]
    with pytest.raises(GoStageFailed):
        _run_stage("render", failing, explain=False, door="downscale")
    printed = capsys.readouterr().out
    assert "downscale: stopped at render" in printed
    assert "go: stopped at" not in printed


def test_the_remedy_command_renders_the_frames_and_nothing_else(tmp_path):
    """The printed remedy used to glob the whole run directory.

    A child's ``wrfout_dir`` IS its run root, so that command handed
    events.jsonl, report.json, the checkpoints and the picture tree to
    the renderer, which stopped on "NetCDF: Unknown file format".
    """

    import glob

    from gpuwm.go_cli import render_command, wrfout_frames

    run = tmp_path / "child-run"
    (run / "png" / "d02").mkdir(parents=True)
    (run / "png" / "d02" / "picture.png").write_bytes(b"\x89PNG")
    (run / "events.jsonl").write_text("{}\n", encoding="utf-8")
    (run / "report.json").write_text("{}\n", encoding="utf-8")
    (run / "gpuwmrst_d02_1974-04-03_12_10_00.npz").write_bytes(b"NPZ")
    (run / "downscale-plan.json").write_text("{}\n", encoding="utf-8")
    frames = []
    for hour in (12, 13):
        frame = run / f"wrfout_d02_1974-04-03_{hour}_00_00"
        frame.write_bytes(b"CDF")
        frames.append(frame)

    plan = {"run": run, "wrfout_dir": run, "render": run / "png",
            "render_products": "all"}
    assert sorted(wrfout_frames(plan)) == sorted(frames)
    # The command a reader pastes names the frames, and only the frames:
    # nothing in it is a pattern a shell has to expand, because the
    # quoting that makes it pasteable is exactly what stops a shell
    # expanding one.
    printed = render_command(plan)
    targets = printed[printed.index("render") + 1:printed.index("--series")]
    assert sorted(Path(path) for path in targets) == sorted(frames)

    # Before the run has published anything -- the dry-run case -- the
    # pattern is printed instead, and it still selects frames alone.
    empty = tmp_path / "not-yet"
    (empty / "png").mkdir(parents=True)
    (empty / "events.jsonl").write_text("{}" + chr(10), encoding="utf-8")
    pattern = render_command({"run": empty, "wrfout_dir": empty,
                              "render": empty / "png"})[4]
    assert pattern.endswith("wrfout_d*")
    assert glob.glob(pattern) == []


def test_the_child_progress_object_is_the_observer_surface_it_is_handed(
        tmp_path):
    """`_GoObserver` forwards five calls; two of them used to raise.

    ``failed()`` takes no argument on that surface and ``stage_progress``
    has to exist -- the heartbeat calls it whenever a running stage's
    progress file carries a model clock.
    """

    import json as _json

    from gpuwm.offline_child_run import _ChildProgress
    from gpuwm.runplan import _GoObserver

    out = tmp_path / "child-run"
    out.mkdir()
    config = tmp_path / "child.toml"
    config.write_text("# child\n", encoding="utf-8")
    progress = _ChildProgress()
    progress.start(outdir=out, child_config=config, ratio=3,
                   start_time=datetime(1974, 4, 3, 12),
                   parent={"run_dir": str(tmp_path), "frames": 1},
                   name="Downscale of parent")
    observer = _GoObserver(progress)
    observer.stage_heartbeat(
        label="render", elapsed_seconds=1.5,
        progress={"model_elapsed_seconds": 600.0, "status": "running"})
    observer.failed()
    progress.close()
    events = [_json.loads(line) for line
              in (out / "events.jsonl").read_text(encoding="utf-8").splitlines()
              if line.strip()]
    coarse = [event for event in events
              if event["event"] == "model_progress"]
    assert len(coarse) == 1
    assert coarse[0]["source"] == "stage_progress_file"
    assert events[-1]["event"] == "failed" and events[-1]["stage"] == "forecast"
