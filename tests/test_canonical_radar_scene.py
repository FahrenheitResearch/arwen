"""Exercise the compiled native atmosphere converter and independent NC reader."""
import json
from pathlib import Path
import subprocess

import netCDF4 as nc
import numpy as np
import pytest


@pytest.fixture
def binary():
    from gpuwm.rustwx import simulated_radar_binary

    value = simulated_radar_binary()
    assert value is not None, (
        "rw_simradar is required by the native CPU partition; skipping it "
        "would leave column temperature, wind and writer contracts untested"
    )
    return str(value)


def capsule(path, valid_time="2026-10-02_00:00:30"):
    shape = (3, 5, 5)
    fields = {
        "pressure_pa": np.broadcast_to(np.array([40000., 70000., 95000.])[:, None, None], shape),
        "potential_temperature_k": np.broadcast_to(np.array([330., 315., 300.])[:, None, None], shape),
        "eastward_wind_m_s": np.arange(75, dtype=float).reshape(shape) * .1,
        "northward_wind_m_s": np.full(shape, -3.),
        "qv_kg_kg": np.broadcast_to(np.array([.001, .006, .01])[:, None, None], shape),
        "qc_kg_kg": np.full(shape, .0001), "qr_kg_kg": np.full(shape, .0002),
        "qi_kg_kg": np.zeros(shape), "qs_kg_kg": np.zeros(shape), "qg_kg_kg": np.zeros(shape),
        "nr_kg1": np.full(shape, 10000.),
    }
    with nc.Dataset(path, "w", format="NETCDF3_64BIT_OFFSET") as f:
        for name, n in [("level", 3), ("interface", 4), ("latitude", 5), ("longitude", 5)]:
            f.createDimension(name, n)
        f.setncatts(dict(schema="native-atmosphere.columns/v1", valid_time=valid_time,
            simulation_start="2026-10-02_00:00:00", source_model="Analytic columns",
            source_checkpoint="state.nc", config_sha256="1"*64, microphysics_scheme="wsm6",
            mp_physics=np.int32(6), vertical_velocity_method="material-height-derivative/one-model-step",
            derivative_interval_s=30., derivative_stencil="backward", gravity_m_s2=9.80616))
        f.createVariable("latitude_deg", "f8", ("latitude",))[:] = np.linspace(34, 36, 5)
        f.createVariable("longitude_deg", "f8", ("longitude",))[:] = np.linspace(261, 263, 5)
        f.createVariable("terrain_height_m", "f8", ("latitude", "longitude"))[:] = 100.
        for name, values in fields.items():
            f.createVariable(name, "f8", ("level", "latitude", "longitude"))[:] = values
        f.createVariable("height_half_m", "f8", ("interface", "latitude", "longitude"))[:] = np.broadcast_to(
            np.array([14000., 8000., 1500., 100.])[:, None, None], (4,5,5))
        f.createVariable("vertical_velocity_half_m_s", "f8", ("interface", "latitude", "longitude"))[:] = np.broadcast_to(
            np.array([0., 2., 1., 0.])[:, None, None], (4,5,5))
    return fields


def convert(binary, source, target):
    return subprocess.run([binary, "--canonical-atmosphere", str(source), "--out", str(target)],
                          capture_output=True, text=True)


def test_python_helper_scene_is_discoverable_for_replay(binary, tmp_path):
    from gpuwm.io.wrfout import iter_wrfout_files
    from gpuwm.rustwx import canonical_radar_scene

    source = tmp_path/"columns.nc"
    capsule(source)
    target = canonical_radar_scene(source, outdir=tmp_path/"canonical")
    assert target.name == "wrfout_d01_columns.nc"
    assert iter_wrfout_files(tmp_path, include_temporaries=False) == [target]


def test_native_units_levels_winds_and_provenance(binary, tmp_path):
    source, target = tmp_path/"columns.nc", tmp_path/"scene.nc"
    fields = capsule(source)
    result = convert(binary, source, target)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["event"] == "canonical_atmosphere_committed"
    with nc.Dataset(target) as f:
        np.testing.assert_array_equal(f["P"][0], fields["pressure_pa"][::-1])
        np.testing.assert_array_equal(f["T"][0]+300., fields["potential_temperature_k"][::-1])
        np.testing.assert_array_equal(f["RADAR_U_EARTH"][0], fields["eastward_wind_m_s"][::-1])
        np.testing.assert_array_equal(f["RADAR_V_EARTH"][0], fields["northward_wind_m_s"][::-1])
        np.testing.assert_allclose(f["QVAPOR"][0], (fields["qv_kg_kg"]/(1-fields["qv_kg_kg"]))[::-1], rtol=1e-14)
        np.testing.assert_allclose(f["QNRAIN"][0], (fields["nr_kg1"]/(1-fields["qv_kg_kg"]))[::-1], rtol=1e-14)
        np.testing.assert_allclose(f["PHB"][0,:,2,2]/9.80665, [100,1500,8000,14000], atol=1e-11)
        np.testing.assert_array_equal(f["W"][0,:,2,2], [0,1,2,0])
        np.testing.assert_array_equal(f["XLONG"][0,2], np.linspace(-99,-97,5))
        assert f.RADAR_SOURCE_CHECKPOINT == "state.nc"
        assert f.RADAR_SOURCE_GRAVITY_M_S2 == 9.80616
        assert f.MP_PHYSICS == 6
        assert f.RADAR_NATIVE_WINDS == "earth-relative-mass-grid/v1"
        assert f.RADAR_VERTICAL_VELOCITY_STENCIL == "backward"
        assert "QNICE" not in f.variables
        assert "MORR_RIMED_ICE" not in f.ncattrs()


def test_actual_temperature_survives_a_different_native_poisson_exponent(binary, tmp_path):
    source, target = tmp_path/"columns.nc", tmp_path/"scene.nc"
    fields = capsule(source)
    native_temperature = fields["potential_temperature_k"] * (fields["pressure_pa"] / 100000.) ** (287./1004.)
    with nc.Dataset(source, "a") as f:
        f.createVariable("temperature_k", "f8", ("level", "latitude", "longitude"))[:] = native_temperature
    result = convert(binary, source, target)
    assert result.returncode == 0, result.stderr
    with nc.Dataset(target) as f:
        reader_temperature = (f["T"][0] + 300.) * (f["P"][0] / 100000.) ** .2857142857
        np.testing.assert_allclose(reader_temperature, native_temperature[::-1], rtol=1e-14)
        assert np.max(np.abs(f["T"][0] + 300. - fields["potential_temperature_k"][::-1])) > .02
        assert f.RADAR_TEMPERATURE_BASIS.startswith("native temperature_k")


@pytest.mark.parametrize("bad", ["height", "vapor", "shape", "missing_w"])
def test_invalid_columns_never_replace_durable_scene(binary, tmp_path, bad):
    source, target = tmp_path/"columns.nc", tmp_path/"scene.nc"
    capsule(source)
    target.write_bytes(b"previous durable scene")
    with nc.Dataset(source, "a") as f:
        if bad == "height": f["height_half_m"][1,2,2] = 15000
        elif bad == "vapor": f["qv_kg_kg"][1,2,2] = 1.0
        elif bad == "shape": f.renameVariable("pressure_pa", "absent_pressure")
        else: f.renameVariable("vertical_velocity_half_m_s", "absent_w")
    result = convert(binary, source, target)
    assert result.returncode != 0
    assert target.read_bytes() == b"previous durable scene"
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize("masked_wind", [False, True])
def test_canonical_scene_reaches_real_bowecho_and_writers(binary, tmp_path, masked_wind):
    source, target = tmp_path/"columns.nc", tmp_path/"scene.nc"
    capsule(source)
    result = convert(binary, source, target)
    assert result.returncode == 0, result.stderr
    if masked_wind:
        # A limited-area mesh may leave uncovered corners in its rectangular
        # frame. Missing winds remain missing instead of rejecting the run.
        with nc.Dataset(target, "a") as f:
            f["RADAR_U_EARTH"][0, :, 0, 0] = np.nan
            f["RADAR_V_EARTH"][0, :, 0, 0] = np.nan
    request = {"schema":"simulated-radar.request/v1", "history_paths":[str(target)],
        "outdir":str(tmp_path/"output"), "config":{"sites":[{"id":"TEST","lat":35.,"lon":-98.,"height_m":110.}],
        "scan_strategy":"custom", "elevations_deg":[.5], "azimuth_step_deg":10.,
        "gate_spacing_m":1000., "range_km":20., "fields":["reflectivity","velocity"],
        "formats":["cfradial1"], "timing":"history"}}
    req = tmp_path/"request.json"
    req.write_text(json.dumps(request))
    result = subprocess.run([binary,"--request",str(req)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path/"output/radar/manifest.json").read_text())
    assert manifest["simulated"] is True
    assert len(manifest["volumes"]) == 1
    volume = manifest["volumes"][0]
    assert volume["fields"] == ["reflectivity","velocity"]
    assert volume["valid_time"].startswith("2026-10-02T00:00:30")


def test_live_scan_timing_writes_each_volume_once_and_leaves_only_named_files(binary, tmp_path):
    """The real binary behind the live queue, scan timing, three histories.

    Scan timing used to write every volume twice (a held anchor, then a
    linear-adjacent scan under a second generation) and every history
    re-encoded and kept every loop, so disk use grew with the square of the
    run while the manifest listed one copy. Now each volume is published
    once, superseded loops are deleted, every file under radar/ is one the
    manifest names, and every file fits the native output bound the disk
    admission prices.
    """
    from gpuwm.rustwx import estimate_simulated_radar
    from gpuwm.simulated_radar import LiveSimulatedRadar, SimulatedRadarOptions

    scenes = []
    for minute in (0, 10, 20):
        source = tmp_path / f"columns-{minute:02d}.nc"
        capsule(source, valid_time=f"2026-10-02_00:{minute:02d}:00")
        target = tmp_path / f"scene-{minute:02d}.nc"
        result = convert(binary, source, target)
        assert result.returncode == 0, result.stderr
        scenes.append(target)
    options = SimulatedRadarOptions.from_mapping({
        "sites": [{"id": "SIM1", "lat": 35., "lon": -98., "height_m": 110.}],
        "scan_strategy": "custom", "elevations_deg": [.5, 1.5], "azimuth_step_deg": 10.,
        "gate_spacing_m": 1000., "range_km": 20., "fields": ["reflectivity", "velocity"],
        "formats": ["level2", "cfradial1"], "timing": "scan"})
    run = tmp_path / "run"
    with LiveSimulatedRadar(options, run) as live:
        for index, scene in enumerate(scenes):
            live.output_committed(domain=1, valid_time=index, path=scene)
    manifest = json.loads((run / "radar" / "manifest.json").read_text())
    volumes = manifest["volumes"]
    assert [v["timing_used"] for v in volumes] == ["linear_adjacent", "linear_adjacent", "held_anchor"]
    generations = [p for p in (run / "radar" / "d01" / "SIM1").iterdir() if p.name != "loops"]
    assert len(generations) == 3, "a superseded generation stayed on disk"
    assert len(manifest["loops"]) == 4
    assert all(len(loop["frames"]) == 3 for loop in manifest["loops"])
    named = {a["path"] for v in volumes for a in v["files"] + v["images"]}
    named |= {loop["path"] for loop in manifest["loops"]}
    on_disk = {p.relative_to(run).as_posix() for p in (run / "radar").rglob("*") if p.is_file()}
    assert on_disk - {"radar/manifest.json", "radar/.manifest.lock"} == named

    bound = estimate_simulated_radar((), outdir=run, config=options.to_mapping())["output"]
    for volume in volumes:
        for record in volume["files"]:
            assert record["bytes"] <= bound["format_bytes_upper_bound"][record["format"]]
        assert len(volume["images"]) == bound["ppi_images_per_site_volume"]
        for image in volume["images"]:
            assert image["bytes"] <= bound["png_bytes_upper_bound"]
    for loop in manifest["loops"]:
        assert loop["bytes"] <= len(loop["frames"]) * bound["gif_frame_bytes_upper_bound"] + 1024
    used = sum(p.stat().st_size for p in (run / "radar" / "d01").rglob("*") if p.is_file())
    assert used <= len(volumes) * bound["output_bytes_per_site_volume_upper_bound"]
