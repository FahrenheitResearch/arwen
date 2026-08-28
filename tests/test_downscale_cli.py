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
    DERIVED_CHILD_CONFIG_NAME,
    _centered_placement,
    _derive_child_run_config,
    _discover_parent_series,
    _fit_child_size,
    _nearest_parent_index,
    _parse_point,
    _render_child_toml,
    derived_child_config_path,
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
    size = _fit_child_size(
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
    (outdir / "gpuwmrst_d02_final.npz").write_bytes(b"x")
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
        tmp_path / "gpuwmrst_d01_final.npz",
        dict(_SURFACE_PARENT_CONFIG, nx=nx, ny=ny, grid_id=1,
             dt=3.0, run_seconds=7200.0, nested=False, specified=False))
    return [
        "downscale", str(tmp_path),
        "--parent-restart", str(restart),
        "--point", "39.5,-84.0",
        "--ratio", "1", "--child-size", "12,10",
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
