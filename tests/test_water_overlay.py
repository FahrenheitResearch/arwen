"""Water-temperature overlay (task #71): byte-identity first.

CPU-only, synthetic, no downloads.  The absent-configuration path is
tested before the feature: no overlay must mean the exact same objects,
fingerprints, manifests, and catalogs as a tree without the feature.
"""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest

from gpuwm.ingest.grib import Era5Snapshot
from gpuwm.ingest.water_overlay import (
    WaterOverlayError,
    WaterTemperatureOverlay,
    apply_water_temperature_overlay,
    load_water_temperature_overlay,
    masked_bilinear_sample,
    overlay_snapshots,
    overlay_snapshots_by_time,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _analytic(lat, lon):
    """Linear in lat/lon: bilinear interpolation reproduces it exactly."""
    return 290.0 + 1.25 * (lat - 40.0) + 0.75 * (lon + 84.0)


def make_snapshot(valid_time=datetime(1985, 5, 31, 12)):
    """Coarse source crop with an inland water block (a synthetic lake)."""
    latitude = np.arange(40.0, 44.01, 0.5)
    longitude = np.arange(-84.0, -77.99, 0.5)
    ny, nx = latitude.size, longitude.size
    landsea = np.ones((ny, nx))
    landsea[2:7, 3:10] = 0.0
    rows = np.arange(ny)[:, None] * np.ones((1, nx))
    cols = np.ones((ny, 1)) * np.arange(nx)[None, :]
    skintemp = 285.0 + 0.7 * rows + 0.3 * cols
    sst = np.where(landsea < 0.5, 284.0 + 0.5 * rows + 0.2 * cols, np.nan)
    psfc = 101300.0 + 10.0 * rows
    tt = np.stack([280.0 + rows + cols, 250.0 + rows - cols])
    return Era5Snapshot(
        valid_time=valid_time,
        levels_hpa=np.array([1000.0, 500.0]),
        latitude=latitude, longitude=longitude,
        fields={"LANDSEA": landsea, "SKINTEMP": skintemp, "SST": sst,
                "PSFC": psfc, "TT": tt})


def write_overlay(
        path, latitude, longitude, values, *, units="kelvin",
        variable="analysed_sst", standard_name="sea_water_temperature",
        mask=None, time_size=1):
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("time", time_size)
        dataset.createDimension("lat", latitude.size)
        dataset.createDimension("lon", longitude.size)
        dataset.createVariable("lat", "f8", ("lat",))[:] = latitude
        dataset.createVariable("lon", "f8", ("lon",))[:] = longitude
        var = dataset.createVariable(
            variable, "f8", ("time", "lat", "lon"), fill_value=-1.0e30)
        if units is not None:
            var.units = units
        if standard_name is not None:
            var.standard_name = standard_name
        data = np.broadcast_to(values, (time_size, *values.shape)).copy()
        if mask is not None:
            data = np.ma.array(
                data, mask=np.broadcast_to(mask, data.shape))
        var[:] = data
    return Path(path)


def write_fine_overlay(path, **kwargs):
    latitude = np.arange(39.9, 44.11, 0.05)
    longitude = np.arange(-84.1, -77.89, 0.05)
    values = _analytic(latitude[:, None], longitude[None, :])
    return write_overlay(path, latitude, longitude, values, **kwargs)


# ---------------------------------------------------------------------------
# 1. Absent configuration is the identity
# ---------------------------------------------------------------------------

def test_absent_overlay_is_the_identity():
    snapshots = (make_snapshot(),)
    result, receipt = overlay_snapshots(snapshots, None)
    assert result is snapshots
    assert receipt is None
    by_time = {snapshots[0].valid_time: snapshots[0]}
    mapped, receipt = overlay_snapshots_by_time(by_time, None)
    assert mapped is by_time
    assert receipt is None


def test_no_overlay_keeps_the_experiment_fingerprint():
    """The option must not move any existing fingerprint when off.

    The anchor hash was computed at lane base 4211b9e8, before this
    feature existed, from this exact scaffold + catalog stub.  The
    overlay lives on [case_data]/prepare kwargs, never ExperimentConfig,
    so the restart identity payload and this hash are untouched.
    """
    from gpuwm.core.model import experiment_fingerprint
    from gpuwm.verify.cases.nest_ideal_r1_moist import load_scaffold

    exp = load_scaffold()
    catalog = SimpleNamespace(run_provenance={
        "product_id": "stability-anchor",
        "files": ({"role": "forcing", "path": "a.grib", "sha256": "00",
                   "size": 1, "product_id": "era5", "provenance": ""},)})
    assert experiment_fingerprint(exp, catalog) == (
        "5ca17be315a646bf4d45ab39450efad46b8d04497c2eaad54166db51b7d1d2d8")


def test_catalog_identity_moves_exactly_when_the_overlay_does():
    from gpuwm.ingest.preflight import CatalogFile, InputCatalog

    def catalog(files):
        return InputCatalog(
            files=files, product_id="era5", provenance={},
            raw_valid_times=(), valid_times=(), excluded_valid_times=(),
            levels_hpa=(), inventory=(), units={}, spatial_coverage=None,
            masks={}, snapshots=(), field_sources={})

    base_files = (CatalogFile(
        role="forcing", path=Path("a.grib"), sha256="00", size=1),)
    overlay_file = CatalogFile(
        role="water_temperature_overlay", path=Path("sst.nc"),
        sha256="11", size=2)
    without = catalog(base_files)
    unchanged = catalog(base_files)
    with_overlay = catalog((*base_files, overlay_file))
    assert without.fingerprint == unchanged.fingerprint
    assert without.fingerprint != with_overlay.fingerprint
    roles = [item["role"] for item in with_overlay.run_provenance["files"]]
    assert "water_temperature_overlay" in roles
    assert "water_temperature_overlay" not in [
        item["role"] for item in without.run_provenance["files"]]


def test_case_data_key_is_optional_and_off_by_default(tmp_path):
    from gpuwm.case_data import build_case_data

    for name in ("era5.grib", "Vtable.ERA5", "namelist.wps"):
        (tmp_path / name).write_text("x")
    (tmp_path / "geog").mkdir()
    raw = {
        "forcing": ["era5.grib"], "vtable": "Vtable.ERA5",
        "wps_namelist": "namelist.wps", "geog_root": "geog",
        "sfcp_to_sfcp": True, "output_title": "t",
    }
    data = build_case_data(dict(raw), source="test", base_dir=tmp_path)
    assert data.water_temperature_overlay is None
    assert data.water_temperature_overlay_identity() is None
    assert not [record for record in data.resolved_inputs()
                if record.role == "water_temperature_overlay"]

    with pytest.raises(ValueError, match="does not exist"):
        build_case_data(
            {**raw, "water_temperature_overlay": "absent.nc"},
            source="test", base_dir=tmp_path)

    (tmp_path / "sst.nc").write_text("x")
    declared = build_case_data(
        {**raw, "water_temperature_overlay": "sst.nc"},
        source="test", base_dir=tmp_path)
    assert declared.water_temperature_overlay == tmp_path / "sst.nc"
    overlay_records = [record for record in declared.resolved_inputs()
                       if record.role == "water_temperature_overlay"]
    assert [record.path for record in overlay_records] == [
        tmp_path / "sst.nc"]


def test_supervisor_remap_carries_the_overlay(tmp_path):
    from gpuwm.case_data import build_case_data, remap_case_data_files

    for name in ("era5.grib", "Vtable.ERA5", "namelist.wps", "sst.nc"):
        (tmp_path / name).write_text("x")
    (tmp_path / "geog").mkdir()
    data = build_case_data({
        "forcing": ["era5.grib"], "vtable": "Vtable.ERA5",
        "wps_namelist": "namelist.wps", "geog_root": "geog",
        "sfcp_to_sfcp": True, "output_title": "t",
        "water_temperature_overlay": "sst.nc",
    }, source="test", base_dir=tmp_path)
    snapshot_dir = tmp_path / "cas"
    snapshot_dir.mkdir()
    replacements = {}
    for record in data.resolved_inputs():
        if record.role == "geog_root":
            continue
        target = snapshot_dir / record.path.name
        target.write_text("y")
        replacements[record.path] = target
    remapped = remap_case_data_files(data, replacements)
    assert remapped.water_temperature_overlay == snapshot_dir / "sst.nc"
    assert remapped.water_temperature_overlay_identity() == (
        tmp_path / "sst.nc")


# ---------------------------------------------------------------------------
# 2/3. Replacement semantics
# ---------------------------------------------------------------------------

def test_covered_water_replaced_everything_else_bit_identical(tmp_path):
    snapshot = make_snapshot()
    overlay = load_water_temperature_overlay(
        write_fine_overlay(tmp_path / "sst.nc"))
    (result,), receipt = overlay_snapshots((snapshot,), overlay)

    water = np.asarray(snapshot.fields["LANDSEA"]) < 0.5
    lat2d = snapshot.latitude[:, None] * np.ones(snapshot.longitude.size)
    lon2d = np.ones((snapshot.latitude.size, 1)) * snapshot.longitude
    expected = _analytic(lat2d, lon2d)
    for name in ("SKINTEMP", "SST"):
        np.testing.assert_allclose(
            result.fields[name][water], expected[water],
            rtol=0.0, atol=1.0e-9)
        assert np.array_equal(
            result.fields[name][~water], snapshot.fields[name][~water],
            equal_nan=True)
    for name in ("LANDSEA", "PSFC", "TT"):
        assert np.array_equal(
            result.fields[name], snapshot.fields[name], equal_nan=True)
    assert np.array_equal(result.latitude, snapshot.latitude)
    assert result.valid_time == snapshot.valid_time
    assert receipt["replaced_cells"] == int(water.sum())
    assert receipt["fallback_cells"] == 0
    assert receipt["fields"] == ["SST", "SKINTEMP"]
    assert receipt["snapshots"] == 1


def test_uncovered_water_keeps_era5(tmp_path):
    snapshot = make_snapshot()
    latitude = np.arange(39.9, 44.11, 0.05)
    longitude = np.arange(-84.1, -81.31, 0.05)   # stops short of the lake's east
    values = _analytic(latitude[:, None], longitude[None, :])
    overlay = load_water_temperature_overlay(
        write_overlay(tmp_path / "west.nc", latitude, longitude, values))
    result, receipt = apply_water_temperature_overlay(snapshot, overlay)
    water = np.asarray(snapshot.fields["LANDSEA"]) < 0.5
    covered = water & (snapshot.longitude[None, :] <= longitude[-1])
    fallback = water & ~covered
    assert fallback.any() and covered.any()
    assert np.array_equal(
        result.fields["SKINTEMP"][fallback],
        snapshot.fields["SKINTEMP"][fallback])
    assert not np.array_equal(
        result.fields["SKINTEMP"][covered],
        snapshot.fields["SKINTEMP"][covered])
    assert receipt["replaced_cells"] == int(covered.sum())
    assert receipt["fallback_cells"] == int(fallback.sum())


def test_zero_360_longitude_overlay_is_unwrapped(tmp_path):
    snapshot = make_snapshot()
    latitude = np.arange(39.9, 44.11, 0.05)
    longitude = np.arange(275.9, 282.11, 0.05)   # -84.1..-77.89 in 0..360
    values = _analytic(latitude[:, None], longitude[None, :] - 360.0)
    overlay = load_water_temperature_overlay(
        write_overlay(tmp_path / "wrapped.nc", latitude, longitude, values))
    result, receipt = apply_water_temperature_overlay(snapshot, overlay)
    water = np.asarray(snapshot.fields["LANDSEA"]) < 0.5
    assert receipt["replaced_cells"] == int(water.sum())
    lat2d = snapshot.latitude[:, None] * np.ones(snapshot.longitude.size)
    lon2d = np.ones((snapshot.latitude.size, 1)) * snapshot.longitude
    np.testing.assert_allclose(
        result.fields["SKINTEMP"][water], _analytic(lat2d, lon2d)[water],
        rtol=0.0, atol=1.0e-9)


# ---------------------------------------------------------------------------
# 4. Masked bilinear corners
# ---------------------------------------------------------------------------

def _tiny_overlay(valid):
    return WaterTemperatureOverlay(
        path=Path("synthetic"), source_format="netcdf", variable="sst",
        declared_units="K",
        latitude=np.array([0.0, 1.0]), longitude=np.array([0.0, 1.0]),
        temperature_k=np.array([[280.0, 290.0], [300.0, 310.0]]),
        valid=np.asarray(valid))


def test_masked_bilinear_renormalizes_over_valid_corners():
    overlay = _tiny_overlay([[True, True], [True, False]])
    values, covered = masked_bilinear_sample(
        overlay, np.array([0.5]), np.array([0.5]))
    assert covered.all()
    np.testing.assert_allclose(values, [(280.0 + 290.0 + 300.0) / 3.0])


def test_masked_bilinear_all_invalid_is_not_covered():
    overlay = _tiny_overlay([[False, False], [False, False]])
    values, covered = masked_bilinear_sample(
        overlay, np.array([0.5]), np.array([0.5]))
    assert not covered.any()
    assert np.isnan(values).all()


def test_masked_bilinear_outside_bbox_is_not_covered():
    overlay = _tiny_overlay([[True, True], [True, True]])
    values, covered = masked_bilinear_sample(
        overlay, np.array([2.0, 0.5]), np.array([0.5, 0.5]))
    assert list(covered) == [False, True]


# ---------------------------------------------------------------------------
# 5. Refusal catalog
# ---------------------------------------------------------------------------

def test_refuses_unknown_container(tmp_path):
    path = tmp_path / "junk.bin"
    path.write_bytes(b"\x00\x01\x02\x03\x04\x05\x06\x07")
    with pytest.raises(WaterOverlayError, match="neither netCDF"):
        load_water_temperature_overlay(path)


def test_refuses_grib_edition_1(tmp_path):
    path = tmp_path / "one.grib"
    path.write_bytes(b"GRIB\x00\x00\x00\x01" + b"\x00" * 16)
    with pytest.raises(WaterOverlayError, match="edition 1"):
        load_water_temperature_overlay(path)


def test_refuses_zero_and_ambiguous_candidates(tmp_path):
    latitude = np.arange(0.0, 1.01, 0.5)
    longitude = np.arange(0.0, 1.01, 0.5)
    values = np.full((3, 3), 290.0)
    nothing = write_overlay(
        tmp_path / "none.nc", latitude, longitude, values,
        variable="mystery", standard_name=None)
    with pytest.raises(WaterOverlayError, match="exactly one"):
        load_water_temperature_overlay(nothing)
    ambiguous = tmp_path / "two.nc"
    write_overlay(ambiguous, latitude, longitude, values, variable="sst",
                  standard_name=None)
    with netCDF4.Dataset(ambiguous, "a") as dataset:
        second = dataset.createVariable("wtmp", "f8", ("time", "lat", "lon"))
        second.units = "kelvin"
        second[:] = 290.0
    with pytest.raises(WaterOverlayError, match="exactly one"):
        load_water_temperature_overlay(ambiguous)
    # An explicit variable resolves the ambiguity.
    picked = load_water_temperature_overlay(ambiguous, variable="wtmp")
    assert picked.variable == "wtmp"


def test_refuses_missing_and_unknown_units(tmp_path):
    latitude = np.arange(0.0, 1.01, 0.5)
    longitude = np.arange(0.0, 1.01, 0.5)
    values = np.full((3, 3), 290.0)
    unitless = write_overlay(
        tmp_path / "unitless.nc", latitude, longitude, values, units=None)
    with pytest.raises(WaterOverlayError, match="no units"):
        load_water_temperature_overlay(unitless)
    strange = write_overlay(
        tmp_path / "strange.nc", latitude, longitude, values,
        units="furlongs")
    with pytest.raises(WaterOverlayError, match="kelvin and Celsius"):
        load_water_temperature_overlay(strange)


def test_refuses_multi_time_and_scrambled_axis(tmp_path):
    latitude = np.arange(0.0, 1.01, 0.5)
    longitude = np.arange(0.0, 1.01, 0.5)
    values = np.full((3, 3), 290.0)
    stacked = write_overlay(
        tmp_path / "stack.nc", latitude, longitude, values, time_size=2)
    with pytest.raises(WaterOverlayError, match="leading dimension"):
        load_water_temperature_overlay(stacked)
    scrambled = write_overlay(
        tmp_path / "scrambled.nc", np.array([0.0, 1.0, 0.5]), longitude,
        values)
    with pytest.raises(WaterOverlayError, match="not strictly monotonic"):
        load_water_temperature_overlay(scrambled)


def test_refuses_undeclared_sentinel(tmp_path):
    latitude = np.arange(0.0, 2.01, 0.5)
    longitude = np.arange(0.0, 2.01, 0.5)
    values = np.full((5, 5), 290.0)
    values[0, 0] = 9999.0
    path = write_overlay(tmp_path / "sentinel.nc", latitude, longitude,
                         values)
    with pytest.raises(WaterOverlayError, match=r"1 valid cell"):
        load_water_temperature_overlay(path)


def test_refuses_disjoint_overlay_and_missing_fields(tmp_path):
    snapshot = make_snapshot()
    latitude = np.arange(-10.0, -8.99, 0.05)
    longitude = np.arange(10.0, 11.01, 0.05)
    faraway = load_water_temperature_overlay(write_overlay(
        tmp_path / "faraway.nc", latitude, longitude,
        np.full((latitude.size, longitude.size), 290.0)))
    with pytest.raises(WaterOverlayError, match="does not intersect"):
        overlay_snapshots((snapshot,), faraway)

    naked = Era5Snapshot(
        valid_time=snapshot.valid_time, levels_hpa=snapshot.levels_hpa,
        latitude=snapshot.latitude, longitude=snapshot.longitude,
        fields={"SKINTEMP": np.asarray(snapshot.fields["SKINTEMP"])})
    with pytest.raises(WaterOverlayError, match="requires LANDSEA"):
        apply_water_temperature_overlay(naked, faraway)
    landsea_only = Era5Snapshot(
        valid_time=snapshot.valid_time, levels_hpa=snapshot.levels_hpa,
        latitude=snapshot.latitude, longitude=snapshot.longitude,
        fields={"LANDSEA": np.asarray(snapshot.fields["LANDSEA"])})
    with pytest.raises(WaterOverlayError, match="neither SST nor SKINTEMP"):
        apply_water_temperature_overlay(landsea_only, faraway)


# ---------------------------------------------------------------------------
# 6. Units guard, both directions
# ---------------------------------------------------------------------------

def test_celsius_shaped_values_under_kelvin_declaration(tmp_path):
    latitude = np.arange(0.0, 1.01, 0.5)
    longitude = np.arange(0.0, 1.01, 0.5)
    path = write_overlay(tmp_path / "c_as_k.nc", latitude, longitude,
                         np.full((3, 3), 15.0), units="K")
    with pytest.raises(WaterOverlayError, match="Celsius-shaped"):
        load_water_temperature_overlay(path)


def test_kelvin_shaped_values_under_celsius_declaration(tmp_path):
    latitude = np.arange(0.0, 1.01, 0.5)
    longitude = np.arange(0.0, 1.01, 0.5)
    path = write_overlay(tmp_path / "k_as_c.nc", latitude, longitude,
                         np.full((3, 3), 288.0), units="degC")
    with pytest.raises(WaterOverlayError, match="kelvin-shaped"):
        load_water_temperature_overlay(path)


def test_celsius_declaration_converts(tmp_path):
    latitude = np.arange(0.0, 1.01, 0.5)
    longitude = np.arange(0.0, 1.01, 0.5)
    path = write_overlay(tmp_path / "celsius.nc", latitude, longitude,
                         np.full((3, 3), 16.5), units="degrees_Celsius")
    overlay = load_water_temperature_overlay(path)
    np.testing.assert_allclose(overlay.temperature_k, 289.65)
    assert overlay.declared_units == "degrees_Celsius"


def test_masked_cells_fall_back_not_replace(tmp_path):
    """Declared-missing overlay cells never paint water cells."""
    snapshot = make_snapshot()
    latitude = np.arange(39.9, 44.11, 0.05)
    longitude = np.arange(-84.1, -77.89, 0.05)
    values = _analytic(latitude[:, None], longitude[None, :])
    mask = np.zeros(values.shape, dtype=bool)
    mask[:, longitude > -81.0] = True
    overlay = load_water_temperature_overlay(write_overlay(
        tmp_path / "masked.nc", latitude, longitude, values, mask=mask))
    result, receipt = apply_water_temperature_overlay(snapshot, overlay)
    water = np.asarray(snapshot.fields["LANDSEA"]) < 0.5
    east = water & (snapshot.longitude[None, :] > -81.0 + 0.05)
    assert east.any()
    assert np.array_equal(
        result.fields["SKINTEMP"][east], snapshot.fields["SKINTEMP"][east])
    assert receipt["fallback_cells"] > 0


# ---------------------------------------------------------------------------
# 7. Route wiring
# ---------------------------------------------------------------------------

def test_runtime_forcing_snapshots_applies_the_declared_overlay(
        tmp_path, monkeypatch, capsys):
    from gpuwm import runtime

    snapshot = make_snapshot()
    decoded = SimpleNamespace(snapshots=(snapshot,))
    monkeypatch.setattr(
        runtime, "cached_era5_forcing", lambda *args, **kwargs: decoded)
    catalog = SimpleNamespace(
        files=(), valid_times=(snapshot.valid_time,),
        excluded_valid_times=())
    overlay_path = write_fine_overlay(tmp_path / "sst.nc")

    plain = SimpleNamespace(
        forcing=(tmp_path / "era5.grib",), vtable=tmp_path / "vt",
        water_temperature_overlay=None)
    untouched = runtime.forcing_snapshots(plain, catalog)
    assert untouched[snapshot.valid_time] is snapshot

    declared = SimpleNamespace(
        forcing=(tmp_path / "era5.grib",), vtable=tmp_path / "vt",
        water_temperature_overlay=overlay_path)
    overlaid = runtime.forcing_snapshots(declared, catalog)
    water = np.asarray(snapshot.fields["LANDSEA"]) < 0.5
    assert not np.array_equal(
        overlaid[snapshot.valid_time].fields["SKINTEMP"][water],
        snapshot.fields["SKINTEMP"][water])
    assert "water-temperature overlay: replaced" in capsys.readouterr().out


def test_era5_direct_binds_the_overlay_into_the_manifest(
        tmp_path, monkeypatch):
    import hashlib
    import json

    from gpuwm import era5_direct

    roles = {}
    for name in ("era5.grib", "Vtable.ERA5", "bridge.exe", "namelist.wps",
                 "exp.toml", "sst.nc"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        path.chmod(0o755)
        roles[name] = path

    def manifest_for(role_paths):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({
            "schema": era5_direct.INPUT_MANIFEST_SCHEMA,
            "files": {
                role: {"name": path.name,
                       "sha256": hashlib.sha256(
                           path.read_bytes()).hexdigest()}
                for role, path in role_paths.items()
            },
        }), encoding="utf-8")
        return manifest, hashlib.sha256(manifest.read_bytes()).hexdigest()

    base_roles = {
        "grib": roles["era5.grib"], "vtable": roles["Vtable.ERA5"],
        "bridge": roles["bridge.exe"],
        "wps_namelist": roles["namelist.wps"],
        "experiment_config": roles["exp.toml"],
    }
    kwargs = dict(
        grib=roles["era5.grib"], vtable=roles["Vtable.ERA5"],
        bridge=roles["bridge.exe"], wps_namelist=roles["namelist.wps"],
        static_input=None, static_receipt=None, source_orography=None,
        source_orography_variable="SOILHGT",
        experiment_config=roles["exp.toml"],
        output_root=tmp_path / "out", geog_root=tmp_path)

    # Configured overlay against a manifest without the role: refused.
    manifest, digest = manifest_for(base_roles)
    with pytest.raises(ValueError, match="role inventory differs"):
        era5_direct.prepare_era5_wrf(
            **kwargs, input_manifest=manifest,
            input_manifest_sha256=digest,
            water_temperature_overlay=roles["sst.nc"])

    # The overlay role present and hash-bound: the manifest stage passes
    # (a sentinel replaces the heavy preparation that follows it).
    class Sentinel(Exception):
        pass

    def stop(*args, **kwargs):
        raise Sentinel()

    monkeypatch.setattr(era5_direct, "resolve_preprocess_backend", stop)
    manifest, digest = manifest_for(
        {**base_roles, "water_temperature_overlay": roles["sst.nc"]})
    with pytest.raises(Sentinel):
        era5_direct.prepare_era5_wrf(
            **kwargs, input_manifest=manifest,
            input_manifest_sha256=digest,
            water_temperature_overlay=roles["sst.nc"])

    # Absent kwarg: the historical five-role manifest still verifies.
    manifest, digest = manifest_for(base_roles)
    with pytest.raises(Sentinel):
        era5_direct.prepare_era5_wrf(
            **kwargs, input_manifest=manifest,
            input_manifest_sha256=digest)


# ---------------------------------------------------------------------------
# 8. GRIB2 container (skips where the Rust tools are absent)
# ---------------------------------------------------------------------------

def _grib2_tools():
    import os
    root = Path(__file__).resolve().parents[1]
    release = root / "tools" / "grib1_bridge" / "target" / "release"
    tools = []
    for name in ("grib2_inventory", "grib2_dump"):
        explicit = os.environ.get(f"GPUWM_{name.upper()}")
        for candidate in ([Path(explicit)] if explicit else [
                release / name, release / f"{name}.exe"]):
            if candidate.is_file():
                tools.append(candidate)
                break
    return tools if len(tools) == 2 else None


def test_grib2_water_temperature_round_trip(tmp_path):
    eccodes = pytest.importorskip("eccodes")
    tools = _grib2_tools()
    if tools is None:
        pytest.skip("Rust grib2_inventory/grib2_dump are not built")
    sample = eccodes.codes_grib_new_from_samples("GRIB2")
    latitude = np.arange(40.0, 44.01, 0.5)
    longitude = np.arange(-84.0, -77.99, 0.5)
    values = _analytic(latitude[:, None],
                       longitude[None, :]).ravel()
    eccodes.codes_set(sample, "discipline", 10)
    eccodes.codes_set(sample, "parameterCategory", 3)
    eccodes.codes_set(sample, "parameterNumber", 0)
    eccodes.codes_set(sample, "Ni", longitude.size)
    eccodes.codes_set(sample, "Nj", latitude.size)
    eccodes.codes_set(sample, "latitudeOfFirstGridPointInDegrees",
                      float(latitude[0]))
    eccodes.codes_set(sample, "latitudeOfLastGridPointInDegrees",
                      float(latitude[-1]))
    eccodes.codes_set(sample, "longitudeOfFirstGridPointInDegrees",
                      float(longitude[0] % 360.0))
    eccodes.codes_set(sample, "longitudeOfLastGridPointInDegrees",
                      float(longitude[-1] % 360.0))
    eccodes.codes_set(sample, "iDirectionIncrementInDegrees", 0.5)
    eccodes.codes_set(sample, "jDirectionIncrementInDegrees", 0.5)
    eccodes.codes_set(sample, "jScansPositively", 1)
    eccodes.codes_set_values(sample, values)
    path = tmp_path / "water.grib2"
    with path.open("wb") as stream:
        eccodes.codes_write(sample, stream)
    eccodes.codes_release(sample)

    overlay = load_water_temperature_overlay(
        path, grib2_inventory=tools[0], grib2_dump=tools[1])
    assert overlay.source_format == "grib2"
    assert overlay.declared_units == "K"
    np.testing.assert_allclose(overlay.latitude, latitude)
    np.testing.assert_allclose(
        overlay.longitude, longitude, atol=1.0e-9)
    np.testing.assert_allclose(
        overlay.temperature_k,
        _analytic(latitude[:, None], longitude[None, :]), atol=2.0e-2)
