from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest
import gpuwm.wrf_direct as wrf_direct

from gpuwm.wrf_direct import (
    _global_updates,
    _domain_global_attributes,
    _dimensions,
    _array_sha256,
    _canonical,
    _interp_nodes,
    _lbc_to_wrf,
    _load_contract,
    _load_static_geometry_receipt,
    _prepare_output_target,
    _publish_staging,
    _contract_payload_sha256,
    _physics_contract_bundle,
    _prepared_vertical_contract,
    _resolved_prototype_value,
    _date_text,
    _forcing_interval_indices,
    _hierarchy_global_updates,
    _validated_hierarchy,
    _surface_fields,
    _validate_file,
    _write_timestamp_at,
    _write_wrfinput,
    PreparedCache,
    PreparedDomainArtifacts,
    export_prepared_wrf_hierarchy,
    load_domain_artifacts_manifest,
    write_domain_artifacts_manifest,
    _wrf_noah_landuse,
)
from gpuwm.vertical_contract import expected_coordinate_shapes
from gpuwm.config import RunConfig
from gpuwm.core.microphysics_transition import MP8_TO_MP18_POLICY


class _Cache:
    def __init__(self, value):
        self.value = value

    def array(self, name):
        return self.value


def test_stock_wrf_export_rejects_gpuwm_mixed_microphysics_before_output(
        tmp_path):
    parent = RunConfig(
        nx=16, ny=16, nz=4, dx=3000.0, dy=3000.0, ztop=10000.0,
        dt=6.0, run_seconds=60.0, moist=True, moist_cq=True,
        mp_physics=8)
    child = RunConfig(
        nx=24, ny=24, nz=4, dx=1000.0, dy=1000.0, ztop=10000.0,
        dt=2.0, run_seconds=60.0, moist=True, moist_cq=True,
        mp_physics=18, nested=True,
        nest_microphysics_transition=MP8_TO_MP18_POLICY)
    exp = SimpleNamespace(domains=(
        SimpleNamespace(grid_id=1, parent_id=0, run=parent),
        SimpleNamespace(grid_id=2, parent_id=1, run=child),
    ))
    output = tmp_path / "stock-wrf"
    with pytest.raises(ValueError, match="GPUWM extension.*not stock-WRF"):
        export_prepared_wrf_hierarchy(exp, (), output)
    assert not output.exists()
    assert not output.with_name(output.name + f".tmp-{os.getpid()}").exists()


def test_contract_inventory_is_frozen():
    contract = _load_contract()
    assert len(contract["wrfinput"]["variables"]) == 183
    assert len(contract["wrfbdy"]["variables"]) == 115


@pytest.mark.parametrize(
    ("mp_physics", "number_names"),
    [
        (6, ()),
        (8, ("QNICE", "QNRAIN")),
        (10, ("QNICE", "QNSNOW", "QNRAIN", "QNGRAUPEL")),
        (
            18,
            (
                "QHAIL", "QNDROP", "QNRAIN", "QNICE", "QNSNOW",
                "QNGRAUPEL", "QNHAIL", "QNCCN", "QVGRAUPEL", "QVHAIL",
            ),
        ),
    ],
)
def test_registry_physics_contract_adds_exact_input_and_boundary_scalars(
        mp_physics, number_names):
    frozen = _load_contract()
    resolved = _physics_contract_bundle(frozen, mp_physics)
    input_names = {item["name"] for item in resolved["wrfinput"]["variables"]}
    bdy_names = {item["name"] for item in resolved["wrfbdy"]["variables"]}
    assert set(number_names) <= input_names
    for name in number_names:
        for suffix in (
                "_BXS", "_BXE", "_BYS", "_BYE",
                "_BTXS", "_BTXE", "_BTYS", "_BTYE"):
            assert name + suffix in bdy_names
    # Resolution is non-mutating: the provenance-frozen WSM6 template stays
    # byte/variable-count stable.
    assert len(frozen["wrfinput"]["variables"]) == 183
    assert len(frozen["wrfbdy"]["variables"]) == 115


def test_wsm6_resolved_contract_digest_remains_byte_semantically_stable():
    assert _contract_payload_sha256(
        _physics_contract_bundle(_load_contract(), 6)
    ) == "f486395516ca9746d8b259f06ccb5913a23210515f3cd0328348f09695cb1366"


def test_nssl2_resolved_contract_uses_exact_registry_metadata():
    resolved = _physics_contract_bundle(_load_contract(), 18)
    specs = {
        item["name"]: item for item in resolved["wrfinput"]["variables"]
    }

    assert specs["QHAIL"]["attributes"] == {
        "FieldType": 104,
        "MemoryOrder": "XYZ",
        "description": "Hail mixing ratio",
        "units": "kg kg-1",
        "stagger": "",
        "coordinates": "XLONG XLAT XTIME",
    }
    assert specs["QNDROP"]["attributes"]["units"] == "# kg-1"
    assert specs["QNCCN"]["attributes"]["description"] == (
        "CCN Number concentration"
    )
    assert specs["QVGRAUPEL"]["attributes"]["units"] == "m(3) kg(-1)"
    assert specs["QVHAIL"]["attributes"]["units"] == "m(3) kg(-1)"


@pytest.mark.parametrize("mass_levels", [35, 49, 80])
@pytest.mark.parametrize("mp_physics", [8, 10, 18])
def test_number_moment_wrfinput_contract_writes_arbitrary_vertical_shape(
        tmp_path, mass_levels, mp_physics):
    contract = _physics_contract_bundle(
        _load_contract(), mp_physics)["wrfinput"]
    dimensions = _dimensions(contract, nx=3, ny=2, nz=mass_levels)
    geometry = {
        "center_lat": 35.5, "center_lon": -98.0,
        "ref_lat": 35.5, "ref_lon": -98.0,
        "truelat1": 30.0, "truelat2": 60.0,
        "stand_lon": -97.0,
    }
    updates = _global_updates(
        valid_time=datetime(2026, 7, 18), nx=3, ny=2, nz=mass_levels,
        dx=1000.0, dy=1000.0, dt=5.0, geometry=geometry)
    path = tmp_path / f"wrfinput-mp{mp_physics}-{mass_levels}"
    # Live export supplies all coordinate arrays from the prepared cache. Use
    # shape-correct stand-ins here so this unit gate isolates contract sizing
    # instead of asking the frozen 49-level prototype to invent a new grid.
    fields = {}
    for spec in contract["variables"]:
        if "prototype_value" not in spec:
            continue
        if spec["dimensions"] == ["Time", "bottom_top"]:
            fields[spec["name"]] = np.zeros(mass_levels, dtype=np.float32)
        elif spec["dimensions"] == ["Time", "bottom_top_stag"]:
            fields[spec["name"]] = np.zeros(mass_levels + 1, dtype=np.float32)
    _write_wrfinput(
        path, contract, dimensions, updates, fields,
        "2026-07-18_00:00:00")
    expected = {
        8: ("QNICE", "QNRAIN"),
        10: ("QNICE", "QNSNOW", "QNRAIN", "QNGRAUPEL"),
        18: (
            "QHAIL", "QNDROP", "QNRAIN", "QNICE", "QNSNOW",
            "QNGRAUPEL", "QNHAIL", "QNCCN", "QVGRAUPEL", "QVHAIL",
        ),
    }[mp_physics]
    with netCDF4.Dataset(path) as dataset:
        for name in expected:
            variable = dataset.variables[name]
            assert variable.dtype == np.dtype("float32")
            assert variable.shape == (1, mass_levels, 2, 3)
            assert not np.any(variable[:])


def test_global_updates_keep_stock_wrf_v4_gate_and_geometry():
    geometry = {
        "center_lat": 35.5,
        "center_lon": -98.0,
        "ref_lat": 35.5,
        "truelat1": 38.5,
        "truelat2": 38.5,
        "stand_lon": -97.5,
    }
    updates = _global_updates(
        valid_time=datetime(2026, 7, 18), nx=500, ny=400, nz=49,
        dx=1000.0, dy=1000.0, dt=5.0, geometry=geometry)
    assert "GPUWM" in updates["TITLE"]
    assert "V4.6.1" in updates["TITLE"]
    assert updates["WEST-EAST_GRID_DIMENSION"] == 501
    assert updates["SOUTH-NORTH_GRID_DIMENSION"] == 401
    assert updates["GHG_INPUT"] == 0
    assert updates["GRID_ID"] == 1
    assert updates["PARENT_ID"] == 0


@pytest.mark.parametrize("nz", [4, 17, 49, 80, 113])
def test_direct_export_dimensions_are_derived_for_any_level_count(nz):
    contract = _load_contract()
    for role in ("wrfinput", "wrfbdy"):
        dimensions = _dimensions(contract[role], nx=31, ny=23, nz=nz)
        assert dimensions["bottom_top"] == nz
        assert dimensions["bottom_top_stag"] == nz + 1


def _synthetic_vertical_cache(nz, *, nx=3, ny=2, p_top=12_345.0):
    coord_shapes = expected_coordinate_shapes(nz)
    shapes = {f"coord/{name}": shape
              for name, shape in coord_shapes.items()}
    shapes.update({
        "state/u": (nz, ny, nx + 1),
        "state/v": (nz, ny + 1, nx),
        "state/w": (nz + 1, ny, nx),
        "state/php": (nz + 1, ny, nx),
        "state/thp": (nz, ny, nx),
        "state/qv": (nz, ny, nx),
        "state/qc": (nz, ny, nx),
        "state/qr": (nz, ny, nx),
        "state/qi": (nz, ny, nx),
        "state/qs": (nz, ny, nx),
        "state/qg": (nz, ny, nx),
        "state/mup": (ny, nx),
        "base/mub": (ny, nx),
        "base/pb": (nz, ny, nx),
        "base/alb": (nz, ny, nx),
        "base/thb": (nz, ny, nx),
        "base/phb": (nz + 1, ny, nx),
        "base/terrain_z": (ny, nx),
    })

    class Cache:
        _arrays = {name: {"shape": list(shape)}
                   for name, shape in shapes.items()}
        header = {"metadata": {
            "base_scalars": {"p_top": p_top},
            "coord_scalars": {"p_top": p_top},
        }}

        @staticmethod
        def array(name):
            assert name == "coord/znw"
            return np.linspace(1.0, 0.0, nz + 1)

    return Cache()


@pytest.mark.parametrize("nz", [4, 17, 49, 80, 113])
def test_prepared_export_vertical_contract_is_count_agnostic(nz):
    cache = _synthetic_vertical_cache(nz)
    assert _prepared_vertical_contract(
        cache, nx=3, ny=2, nz=nz) == 12_345.0


def test_prepared_export_vertical_contract_rejects_dimension_drift():
    cache = _synthetic_vertical_cache(80)
    cache._arrays["state/w"] = {"shape": [80, 2, 3]}
    with pytest.raises(ValueError, match="shape drift"):
        _prepared_vertical_contract(cache, nx=3, ny=2, nz=80)


def test_zero_prototype_scaffold_derives_live_vertical_shape(tmp_path):
    path = tmp_path / "prototype.nc"
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("Time", None)
        dataset.createDimension("bottom_top", 80)
        variable = dataset.createVariable(
            "T_BASE", "f4", ("Time", "bottom_top"))
        resolved = _resolved_prototype_value(variable, [[0.0] * 49])
        assert resolved.shape == (80,)
        assert resolved.dtype == np.float32
        np.testing.assert_array_equal(resolved, 0.0)


def test_nonzero_mismatched_prototype_remains_fail_closed(tmp_path):
    path = tmp_path / "prototype.nc"
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("Time", None)
        dataset.createDimension("bottom_top", 80)
        variable = dataset.createVariable(
            "PROFILE", "f4", ("Time", "bottom_top"))
        with pytest.raises(ValueError, match="nonzero frozen prototype"):
            _resolved_prototype_value(variable, [[1.0] * 49])


def test_noah_landuse_remaps_lakes_but_preserves_mask():
    raw = np.array([[1, 17, 21], [13, 21, 20]], dtype=np.int32)
    mapped, lakes = _wrf_noah_landuse(raw)
    np.testing.assert_array_equal(
        mapped, np.array([[1, 17, 17], [13, 17, 20]], dtype=np.int32))
    np.testing.assert_array_equal(
        lakes, np.array([[False, False, True], [False, True, False]]))


def test_lbc_orientation_matches_wrf_side_layout():
    west = np.arange(2 * 3 * 5).reshape(2, 3, 5)
    actual_west = _lbc_to_wrf(
        _Cache(west), 0, "u", "west", "value")
    np.testing.assert_array_equal(actual_west, np.transpose(west, (2, 0, 1)))

    south = np.arange(2 * 5 * 4).reshape(2, 5, 4)
    actual_south = _lbc_to_wrf(
        _Cache(south), 0, "u", "south", "value")
    np.testing.assert_array_equal(actual_south, np.transpose(south, (1, 0, 2)))

    mu_west = np.arange(1 * 3 * 5).reshape(1, 3, 5)
    actual_mu = _lbc_to_wrf(
        _Cache(mu_west), 0, "mu", "west", "value")
    np.testing.assert_array_equal(actual_mu, mu_west[0].T)


def test_hourly_forcing_inventory_supports_arbitrary_horizon():
    class Cache:
        header = {
            "identity": {"forcing_hours": list(range(25))},
            "arrays": {f"lbc/{index}/u/west/value": {}
                       for index in range(24)},
        }

    assert _forcing_interval_indices(Cache(), 3600) == list(range(24))


def test_forcing_inventory_supports_six_hour_cadence():
    class Cache:
        header = {
            "identity": {"forcing_hours": [0, 6, 12]},
            "arrays": {f"lbc/{index}/u/west/value": {}
                       for index in range(2)},
        }

    assert _forcing_interval_indices(Cache(), 21_600) == [0, 1]


def test_forcing_inventory_fails_closed_on_cadence_mismatch():
    class Cache:
        header = {
            "identity": {"forcing_hours": [0, 1, 3]},
            "arrays": {
                "lbc/0/u/west/value": {},
                "lbc/1/u/west/value": {},
            },
        }

    try:
        _forcing_interval_indices(Cache(), 3600)
    except ValueError as exc:
        assert "cadence" in str(exc)
    else:
        raise AssertionError("gapped forcing hours were accepted")


def test_canonical_surface_bypasses_source_specific_soil_nodes():
    names = {
        "TSK", "TSLB", "SMOIS", "SH2O", "TMN", "SEAICE", "XLAND",
        "LANDMASK", "SNOW", "SNOWH",
    }
    values = {name: np.ones((2, 2), dtype=np.float64) for name in names}
    for name in ("TSLB", "SMOIS", "SH2O"):
        values[name] = np.ones((4, 2, 2), dtype=np.float64)
    values["TSK"] *= 280.0
    values["TMN"] *= 279.0

    class Cache:
        _arrays = {f"surface/{name}": {} for name in names}

        def array(self, name):
            return values[name.split("/", 1)[1]]

    static = {
        "GREENFRAC": np.full((12, 2, 2), 0.5),
        "ALBEDO12M": np.full((12, 2, 2), 20.0),
        "LAI12M": np.full((12, 2, 2), 2.0),
        "SNOALB": np.full((2, 2), 60.0),
    }
    actual = _surface_fields(Cache(), static, 3)
    np.testing.assert_array_equal(actual["TSLB"], values["TSLB"])
    np.testing.assert_array_equal(actual["SH2O"], values["SH2O"])
    np.testing.assert_allclose(actual["VEGFRA"], 50.0)


def test_timestamp_writer_appends_multiple_boundary_records(tmp_path):
    path = tmp_path / "times.nc"
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("Time", None)
        dataset.createDimension("DateStrLen", 19)
        times = dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
        _write_timestamp_at(times, "2026-07-18_00:00:00", 0)
        _write_timestamp_at(times, "2026-07-18_01:00:00", 1)
    with netCDF4.Dataset(path) as dataset:
        decoded = [b"".join(row).decode("ascii")
                   for row in dataset.variables["Times"][:]]
    assert decoded == ["2026-07-18_00:00:00", "2026-07-18_01:00:00"]


def test_hrrr_nodes_interpolate_to_noah_midpoints():
    depths = np.array([0.0, 0.01, 0.04, 0.10, 0.30, 0.60, 1.0, 1.6, 3.0])
    nodes = depths[:, None, None] * 10.0 + 280.0
    output = _interp_nodes(nodes)
    np.testing.assert_allclose(
        output[:, 0, 0], 280.0 + 10.0 * np.array([0.05, 0.25, 0.70, 1.50]))


def test_valid_time_is_normalized_to_utc():
    value = datetime(2026, 7, 17, 17, tzinfo=timezone(-timedelta(hours=7)))
    assert _date_text(value) == "2026-07-18_00:00:00"


def test_prepared_array_digest_is_verified(tmp_path):
    value = np.arange(6, dtype=np.float32).reshape(2, 3)
    path = tmp_path / "array.npy"
    np.save(path, value)
    digest = _array_sha256(value)
    header = {
        "schema": "gpuwm-prepared-real-cache-v1",
        "status": "READY",
        "identity": {},
        "metadata": {},
        "payload_bytes": int(value.nbytes),
        "arrays": {
            "state/test": {
                "file": path.name,
                "shape": [2, 3],
                "dtype": "float32",
                "sha256": digest,
            },
        },
    }
    basis = {name: header[name] for name in (
        "schema", "identity", "metadata", "arrays", "payload_bytes")}
    header["content_sha256"] = hashlib.sha256(
        _canonical(basis).encode("utf-8")).hexdigest()
    (tmp_path / "header.json").write_text(json.dumps(header), encoding="utf-8")
    np.testing.assert_array_equal(PreparedCache(tmp_path).array("state/test"), value)

    mutated = np.load(path, mmap_mode="r+")
    mutated[0, 0] = 99.0
    mutated.flush()
    del mutated
    try:
        PreparedCache(tmp_path).array("state/test")
    except ValueError as exc:
        assert "digest drift" in str(exc)
    else:
        raise AssertionError("modified cache payload was accepted")


def _domain(grid_id, parent_id, *, specified, nested):
    return SimpleNamespace(
        grid_id=grid_id,
        parent_id=parent_id,
        i_parent_start=1 if grid_id == 1 else 20,
        j_parent_start=1 if grid_id == 1 else 15,
        parent_grid_ratio=1 if grid_id == 1 else 3,
        parent_time_step_ratio=1 if grid_id == 1 else 3,
        run=SimpleNamespace(specified=specified, nested=nested),
    )


def _artifact(grid_id):
    return PreparedDomainArtifacts(
        grid_id, Path(f"prepared-d{grid_id:02d}"),
        Path(f"static-d{grid_id:02d}.npz"),
        Path(f"geometry-d{grid_id:02d}.json"))


def test_hierarchy_pairs_artifacts_by_contiguous_namelist_grid_id():
    d01 = _domain(1, 0, specified=True, nested=False)
    d02 = _domain(2, 1, specified=False, nested=True)
    exp = SimpleNamespace(
        domains=(d01, d02),
        projection=SimpleNamespace(map_proj="lambert"))
    pairs = _validated_hierarchy(exp, (_artifact(2), _artifact(1)))
    assert [(domain.grid_id, artifact.grid_id)
            for domain, artifact in pairs] == [(1, 1), (2, 2)]


def test_hierarchy_rejects_missing_domain_artifact():
    d01 = _domain(1, 0, specified=True, nested=False)
    d02 = _domain(2, 1, specified=False, nested=True)
    exp = SimpleNamespace(
        domains=(d01, d02),
        projection=SimpleNamespace(map_proj="lambert"))
    with pytest.raises(ValueError, match="exactly cover"):
        _validated_hierarchy(exp, (_artifact(1),))


def test_hierarchy_rejects_artifact_misbinding_and_parent_cycle():
    d01 = _domain(1, 0, specified=True, nested=False)
    d02 = _domain(2, 1, specified=False, nested=True)
    exp = SimpleNamespace(
        domains=(d01, d02),
        projection=SimpleNamespace(map_proj="lambert"))

    with pytest.raises(ValueError, match="exactly cover"):
        _validated_hierarchy(exp, (_artifact(1), _artifact(3)))
    with pytest.raises(ValueError, match="duplicate artifacts"):
        _validated_hierarchy(exp, (_artifact(1), _artifact(2), _artifact(2)))

    d02.parent_id = 2
    with pytest.raises(ValueError, match="earlier parent"):
        _validated_hierarchy(exp, (_artifact(1), _artifact(2)))


def test_child_global_updates_emit_wrf_nest_identity():
    domain = _domain(2, 1, specified=False, nested=True)
    cfg = {"nx": 60, "ny": 45, "nz": 8,
           "dx": 4000.0, "dy": 4000.0, "dt": 20.0}
    geometry = {
        "center_lat": 36.0, "center_lon": -97.0,
        "ref_lat": 39.7, "ref_lon": -83.9,
        "truelat1": 30.0, "truelat2": 60.0,
        "stand_lon": -83.9,
    }
    updates = _hierarchy_global_updates(
        valid_time=datetime(1999, 5, 3, 12), cfg=cfg,
        geometry=geometry, domain=domain)
    assert updates["GRID_ID"] == 2
    assert updates["PARENT_ID"] == 1
    assert updates["I_PARENT_START"] == 20
    assert updates["J_PARENT_START"] == 15
    assert updates["PARENT_GRID_RATIO"] == 3
    assert updates["DT"] == 20.0


def test_domain_artifact_manifest_is_relocatable_and_strict(tmp_path):
    manifest = tmp_path / "artifacts.json"
    manifest.write_text(json.dumps({
        "schema": "gpuwm-native-domain-artifacts-v1",
        "domains": [{
            "grid_id": 1,
            "prepared_cache": "cache/d01",
            "static_cache": "static/d01.npz",
            "geometry_receipt": "geometry/d01.json",
        }],
    }), encoding="utf-8")
    (artifact,) = load_domain_artifacts_manifest(manifest)
    assert artifact.prepared_cache == tmp_path / "cache" / "d01"
    assert artifact.static_cache == tmp_path / "static" / "d01.npz"

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["domains"][0]["unbound"] = "forbidden"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="keys must be exactly"):
        load_domain_artifacts_manifest(manifest)


def test_domain_artifact_manifest_rejects_absolute_and_parent_escape(tmp_path):
    manifest = tmp_path / "artifacts.json"
    base = {
        "schema": "gpuwm-native-domain-artifacts-v1",
        "domains": [{
            "grid_id": 1,
            "prepared_cache": "cache/d01",
            "static_cache": "static/d01.npz",
            "geometry_receipt": "geometry/d01.json",
        }],
    }
    base["domains"][0]["prepared_cache"] = str(tmp_path.resolve())
    manifest.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(ValueError, match="relative path"):
        load_domain_artifacts_manifest(manifest)

    base["domains"][0]["prepared_cache"] = "../escape"
    manifest.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        load_domain_artifacts_manifest(manifest)


def test_domain_artifact_manifest_writer_round_trips_relative_inventory(
        tmp_path):
    artifacts = []
    for grid_id in (1, 2):
        root = tmp_path / f"d{grid_id:02d}"
        prepared = root / "prepared-cache"
        prepared.mkdir(parents=True)
        (prepared / "header.json").write_text("{}", encoding="utf-8")
        static = root / "native-static.npz"
        static.write_bytes(b"static")
        geometry = root / "geometry.json"
        geometry.write_text("{}", encoding="utf-8")
        artifacts.append(PreparedDomainArtifacts(
            grid_id, prepared, static, geometry))

    manifest = tmp_path / "domain-artifacts.json"
    payload = write_domain_artifacts_manifest(
        manifest, tuple(reversed(artifacts)))
    loaded = load_domain_artifacts_manifest(manifest)

    assert [item["grid_id"] for item in payload["domains"]] == [1, 2]
    assert payload["domains"][0]["prepared_cache"] \
        == "d01/prepared-cache"
    assert [item.grid_id for item in loaded] == [1, 2]
    assert loaded[1].static_cache == artifacts[1].static_cache.resolve()
    assert not tuple(tmp_path.glob(".*.tmp-*"))


def test_file_validation_checks_real_netcdf_domain_global_attributes(tmp_path):
    contract = _load_contract()["wrfinput"]
    dimensions = _dimensions(contract, nx=3, ny=2, nz=49)
    geometry = {
        "center_lat": 35.5, "center_lon": -98.0,
        "ref_lat": 35.5, "ref_lon": -98.0,
        "truelat1": 30.0, "truelat2": 60.0,
        "stand_lon": -97.0,
    }
    updates = _global_updates(
        valid_time=datetime(2026, 7, 18), nx=3, ny=2, nz=49,
        dx=1000.0, dy=1000.0, dt=5.0, geometry=geometry)
    path = tmp_path / "wrfinput_d01"
    _write_wrfinput(
        path, contract, dimensions, updates, {}, "2026-07-18_00:00:00")
    expected = _domain_global_attributes(updates)
    result = _validate_file(
        path, contract, nx=3, ny=2, nz=49,
        expected_global_attributes=expected)
    assert result["bytes"] > 0

    with netCDF4.Dataset(path, "r+") as dataset:
        dataset.setncattr("GRID_ID", np.int32(2))
    with pytest.raises(ValueError, match="global attribute drift"):
        _validate_file(
            path, contract, nx=3, ny=2, nz=49,
            expected_global_attributes=expected)

    child_cfg = {
        "nx": 3, "ny": 2, "nz": 49,
        "dx": 1000.0 / 3.0, "dy": 1000.0 / 3.0, "dt": 5.0 / 3.0,
    }
    child = _domain(2, 1, specified=False, nested=True)
    child_updates = _hierarchy_global_updates(
        valid_time=datetime(2026, 7, 18), cfg=child_cfg,
        geometry=geometry, domain=child)
    child_path = tmp_path / "wrfinput_d02"
    _write_wrfinput(
        child_path, contract, dimensions, child_updates, {},
        "2026-07-18_00:00:00")
    _validate_file(
        child_path, contract, nx=3, ny=2, nz=49,
        expected_global_attributes=_domain_global_attributes(child_updates))
    with netCDF4.Dataset(child_path) as dataset:
        assert dataset.getncattr("GRID_ID") == 2
        assert dataset.getncattr("PARENT_ID") == 1
        assert dataset.getncattr("I_PARENT_START") == 20
        assert dataset.getncattr("J_PARENT_START") == 15
        assert dataset.getncattr("PARENT_GRID_RATIO") == 3
        assert dataset.getncattr("DX") == np.float32(1000.0 / 3.0)
        assert dataset.getncattr("DT") == np.float32(5.0 / 3.0)


def test_static_geometry_receipt_is_cache_and_namelist_bound(tmp_path):
    static = tmp_path / "native-static.npz"
    static.write_bytes(b"content-addressed-static")
    geometry = {
        "mass_shape": [2, 3], "nz": 4,
        "dx_m": 1000.0, "dy_m": 1000.0,
        "ref_lat": 35.5, "ref_lon": -98.0,
        "truelat1": 30.0, "truelat2": 60.0,
        "stand_lon": -97.0, "center_lat": 35.5,
        "center_lon": -98.0, "lat_range": [35.0, 36.0],
        "lon_range": [-99.0, -97.0],
    }
    receipt = tmp_path / "geometry.json"
    payload = {
        "schema": "gpuwm-native-static-direct-v1",
        "status": "PASS",
        "cache": {
            "path": static.name, "bytes": static.stat().st_size,
            "sha256": hashlib.sha256(static.read_bytes()).hexdigest(),
        },
        "geometry": geometry,
    }
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    observed, _digest = _load_static_geometry_receipt(
        receipt, static, expected_geometry=geometry)
    assert observed == geometry

    payload["geometry"] = {**geometry, "center_lon": -90.0}
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from namelist"):
        _load_static_geometry_receipt(
            receipt, static, expected_geometry=geometry)


def test_failed_overwrite_restores_previous_valid_tree(tmp_path, monkeypatch):
    output = tmp_path / "wrf-ready"
    output.mkdir()
    (output / "manifest.json").write_text("previous", encoding="utf-8")
    staging = tmp_path / "wrf-ready.tmp-123"
    staging.mkdir()
    (staging / "manifest.json").write_text("candidate", encoding="utf-8")
    backup = output.with_name(output.name + ".previous-valid")
    real_replace = wrf_direct.os.replace

    def fail_candidate_publish(source, destination):
        if Path(source) == staging and Path(destination) == output:
            raise OSError("injected publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(wrf_direct.os, "replace", fail_candidate_publish)
    with pytest.raises(OSError, match="injected"):
        _publish_staging(staging, output, backup)
    assert (output / "manifest.json").read_text(encoding="utf-8") == "previous"
    assert staging.exists()
    assert not backup.exists()


def test_interrupted_backup_is_recovered_before_next_overwrite(tmp_path):
    output = tmp_path / "wrf-ready"
    backup = output.with_name(output.name + ".previous-valid")
    backup.mkdir()
    (backup / "manifest.json").write_text("previous", encoding="utf-8")
    assert _prepare_output_target(output, overwrite=True) == backup
    assert (output / "manifest.json").read_text(encoding="utf-8") == "previous"
    assert not backup.exists()


@pytest.mark.parametrize("max_dom", range(1, 22))
def test_hierarchy_export_publishes_every_input_and_root_boundary_atomically(
        tmp_path, monkeypatch, max_dom):
    def run(nx, ny, dx, dt, *, specified, nested):
        return SimpleNamespace(
            nx=nx, ny=ny, nz=8, dx=dx, dy=dx, dt=dt,
            specified=specified, nested=nested, mp_physics=6)

    d01 = _domain(1, 0, specified=True, nested=False)
    d01.run = run(100, 80, 12000.0, 60.0,
                  specified=True, nested=False)
    domains = [d01]
    for grid_id in range(2, max_dom + 1):
        # d01-d04 is a deep chain; d05 is its root sibling and d06 starts
        # a second branch.  Larger synthetic cases remain root siblings so
        # the 1..21 exporter cardinality test does not require huge grids.
        parent_id = ({2: 1, 3: 2, 4: 3, 5: 1, 6: 2}.get(grid_id, 1))
        domain = _domain(
            grid_id, parent_id, specified=False, nested=True)
        ratio_depth = 1 if parent_id == 1 else grid_id - 1
        domain.run = run(
            60, 60, 12000.0 / (3 ** ratio_depth),
            60.0 / (3 ** ratio_depth), specified=False, nested=True)
        domains.append(domain)
    exp = SimpleNamespace(
        domains=tuple(domains),
        projection=SimpleNamespace(map_proj="lambert"),
        start_time=datetime(1999, 5, 3, 12),
        vertical=SimpleNamespace(
            eta_levels=(1.0, 0.9, 0.8, 0.7, 0.6,
                        0.5, 0.4, 0.2, 0.0),
            p_top=5000.0, hybrid_opt=2),
    )
    artifacts = tuple(_artifact(grid_id) for grid_id in range(1, max_dom + 1))

    def fake_root_export(_prepared, _static, _geometry, output, **_kwargs):
        output.mkdir()
        (output / "wrfinput_d01").write_bytes(b"root-input")
        (output / "wrfbdy_d01").write_bytes(b"root-boundary")
        (output / "manifest.json").write_text("{}", encoding="utf-8")
        return {
            "files": {
                "wrfinput_d01": {"bytes": 10, "sha256": "root-input"},
                "wrfbdy_d01": {"bytes": 13, "sha256": "root-boundary"},
            },
            "boundary_record_count": 1,
            "boundary_times": ["1999-05-03_12:00:00"],
            "next_boundary_times": ["1999-05-03_18:00:00"],
            "forcing_hours": [0, 6],
        }

    class Cache:
        def __init__(self, grid_id):
            self.header_path = Path(f"header-d{grid_id:02d}.json")
            self.header = {"content_sha256": f"content-d{grid_id:02d}"}

    def fake_context(artifact, domain, _exp, _expected_grid, _valid_time):
        cfg = {
            "nx": domain.run.nx, "ny": domain.run.ny,
            "nz": domain.run.nz, "dx": domain.run.dx,
            "dy": domain.run.dy, "dt": domain.run.dt,
            "mp_physics": domain.run.mp_physics,
        }
        geometry = {
            "center_lat": 36.0, "center_lon": -97.0,
            "ref_lat": 39.7, "ref_lon": -83.9,
            "truelat1": 30.0, "truelat2": 60.0,
            "stand_lon": -83.9,
        }
        return Cache(domain.grid_id), cfg, geometry, 5000.0, \
            f"static-d{domain.grid_id:02d}"

    class LoadedStatic:
        def __enter__(self):
            return {}

        def __exit__(self, *_args):
            return False

    written_updates = {}

    def fake_write(path, _contract, _dimensions, updates, _fields, _stamp):
        path.write_bytes(b"child-input")
        written_updates[path.name] = updates

    monkeypatch.setattr(wrf_direct, "export_prepared_wrf", fake_root_export)
    monkeypatch.setattr(
        wrf_direct, "grids_from_projection_config",
        lambda _exp: tuple(object() for _ in range(max_dom)))
    monkeypatch.setattr(wrf_direct, "_prepared_domain_context", fake_context)
    monkeypatch.setattr(wrf_direct.np, "load", lambda *_a, **_k: LoadedStatic())
    monkeypatch.setattr(wrf_direct, "_wrfinput_fields", lambda *_a, **_k: {})
    monkeypatch.setattr(wrf_direct, "_write_wrfinput", fake_write)
    monkeypatch.setattr(
        wrf_direct, "_validate_file",
        lambda path, *_a, **_k: {"bytes": path.stat().st_size,
                                 "sha256": path.name})
    monkeypatch.setattr(
        wrf_direct, "_sha256", lambda path: f"sha-{Path(path).name}")

    output = tmp_path / "wrf-ready"
    manifest = export_prepared_wrf_hierarchy(
        exp, artifacts, output, boundary_interval_seconds=21600)
    expected_files = [
        "manifest.json", "wrfbdy_d01",
        *(f"wrfinput_d{grid_id:02d}"
          for grid_id in range(1, max_dom + 1)),
    ]
    assert sorted(path.name for path in output.iterdir()) == sorted(
        expected_files)
    assert manifest["status"] == "READY"
    assert [item["grid_id"] for item in manifest["hierarchy"]] == list(
        range(1, max_dom + 1))
    assert set(name for name in manifest["files"] if name.startswith(
        "wrfbdy")) == {"wrfbdy_d01"}
    if max_dom > 1:
        assert written_updates["wrfinput_d02"]["GRID_ID"] == 2
        assert written_updates["wrfinput_d02"]["PARENT_GRID_RATIO"] == 3
    if max_dom >= 6:
        assert written_updates["wrfinput_d04"]["PARENT_ID"] == 3
        assert written_updates["wrfinput_d05"]["PARENT_ID"] == 1
        assert written_updates["wrfinput_d06"]["PARENT_ID"] == 2
    assert not list(tmp_path.glob("wrf-ready.tmp-*"))
