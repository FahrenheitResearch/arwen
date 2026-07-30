"""CPU contracts for common native stock-WRF artifact writers."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.ingest.hrrr_target import HrrrTargetDomain
from gpuwm.native_wrf_contract import (
    NATIVE_LANDUSE_IDENTITY,
    canonical_noah_surface,
    native_geometry_contract,
    native_static_export_fields,
    validate_native_static_fields,
    verify_native_static_receipt,
    write_native_geometry_receipt,
    write_native_static_cache,
    validate_native_lambert_contracts,
)
from gpuwm.experiment import load_experiment
from gpuwm.static.lambert import LambertGrid
from gpuwm.wrf_direct import _load_static_geometry_receipt
from tools.hrrr_build_native_static import native_static_geometry
from tools.write_hrrr_native_geometry_receipt import (
    write_hrrr_native_geometry_receipt,
)


def _grid():
    return LambertGrid(
        ref_lat=35.0, ref_lon=-97.0, truelat1=30.0, truelat2=60.0,
        stand_lon=-97.0, dx=12_000.0, dy=12_000.0,
        e_we=5, e_sn=4)


def test_common_static_and_geometry_writers_are_hash_bound_and_atomic(tmp_path):
    grid = _grid()
    cfg = SimpleNamespace(nx=4, ny=3, nz=2, dx=12_000.0, dy=12_000.0)
    fields = {
        "HGT_M": np.arange(12, dtype=np.float32).reshape(3, 4),
        "LANDMASK": np.ones((3, 4), dtype=np.float32),
    }
    static_path = tmp_path / "native-static.npz"
    receipt_path = tmp_path / "geometry.json"

    static = write_native_static_cache(static_path, fields)
    receipt = write_native_geometry_receipt(
        receipt_path, grid, cfg, static_path)
    verified = verify_native_static_receipt(
        receipt_path, static_path, grid, cfg)

    assert static["fields"] == ["HGT_M", "LANDMASK"]
    assert static["sha256"] == receipt["cache"]["sha256"]
    assert verified == receipt
    with np.load(static_path, allow_pickle=False) as stored:
        assert stored.files == ["HGT_M", "LANDMASK"]
        assert all(stored[name].dtype == np.float64 for name in stored.files)
    assert not tuple(tmp_path.glob(".*.tmp-*"))


def test_common_static_writer_rejects_nonfinite_and_overwrite(tmp_path):
    path = tmp_path / "native-static.npz"
    with pytest.raises(ValueError, match="not finite"):
        write_native_static_cache(path, {"HGT_M": np.array([np.nan])})
    assert not path.exists()

    write_native_static_cache(path, {"HGT_M": np.array([1.0])})
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_native_static_cache(path, {"HGT_M": np.array([2.0])})


def test_hrrr_static_receipt_is_converted_to_common_wrf_contract(tmp_path):
    target = HrrrTargetDomain(
        name="tiny-contract-fixture", map_proj="lambert",
        nx=14, ny=13, nz=9, dx_m=3000.0, dy_m=3000.0,
        ref_lat=35.0, ref_lon=-97.0, truelat1=30.0,
        truelat2=60.0, stand_lon=-97.0, time_step_seconds=15)
    grid = target.grid()
    cfg = SimpleNamespace(
        nx=target.nx, ny=target.ny, nz=target.nz,
        dx=target.dx_m, dy=target.dy_m)
    static_path = tmp_path / "native-static.npz"
    np.savez(static_path, HGT_M=np.zeros((target.ny, target.nx)))
    digest = hashlib.sha256(static_path.read_bytes()).hexdigest()
    rich_path = tmp_path / "hrrr-static.json"
    rich_path.write_text(json.dumps({
        "schema": "gpuwm-native-hrrr-static-v2",
        "status": "PASS",
        "target_domain": target.to_payload(),
        "target_domain_sha256": target.identity_sha256(),
        "geometry": native_geometry_contract(grid, cfg),
        "cache": {
            "path": static_path.name,
            "bytes": static_path.stat().st_size,
            "sha256": digest,
        },
    }))
    common_path = tmp_path / "native-geometry.json"

    with pytest.raises(ValueError, match="target identity mismatch"):
        write_hrrr_native_geometry_receipt(
            static_cache=static_path,
            hrrr_static_receipt=rich_path,
            output=common_path,
            domain_spec=None,
        )

    # Omission means the legacy target, so a non-legacy rich receipt fails
    # before it can publish a misleading common receipt.
    assert not common_path.exists()


def test_hrrr_static_conversion_round_trips_with_explicit_target(tmp_path):
    target = HrrrTargetDomain(
        name="tiny-contract-fixture", map_proj="lambert",
        nx=14, ny=13, nz=9, dx_m=3000.0, dy_m=3000.0,
        ref_lat=35.0, ref_lon=-97.0, truelat1=30.0,
        truelat2=60.0, stand_lon=-97.0, time_step_seconds=15)
    grid = target.grid()
    cfg = SimpleNamespace(
        nx=target.nx, ny=target.ny, nz=target.nz,
        dx=target.dx_m, dy=target.dy_m)
    static_path = tmp_path / "native-static.npz"
    np.savez(static_path, HGT_M=np.zeros((target.ny, target.nx)))
    digest = hashlib.sha256(static_path.read_bytes()).hexdigest()
    rich_path = tmp_path / "hrrr-static.json"
    rich_path.write_text(json.dumps({
        "schema": "gpuwm-native-hrrr-static-v2",
        "status": "PASS",
        "target_domain": target.to_payload(),
        "target_domain_sha256": target.identity_sha256(),
        "geometry": native_geometry_contract(grid, cfg),
        "cache": {
            "path": static_path.name,
            "bytes": static_path.stat().st_size,
            "sha256": digest,
        },
    }))
    spec = tmp_path / "domain.json"
    spec.write_text(json.dumps(target.to_payload()))
    common_path = tmp_path / "native-geometry.json"

    written = write_hrrr_native_geometry_receipt(
        static_cache=static_path,
        hrrr_static_receipt=rich_path,
        output=common_path,
        domain_spec=spec,
    )
    observed, observed_digest = _load_static_geometry_receipt(
        common_path, static_path,
        expected_geometry=native_geometry_contract(grid, cfg))

    assert written["geometry"] == observed
    assert observed_digest == digest


def _sealed_hrrr_static(tmp_path, target, geometry):
    """A minimal PASS static receipt carrying one geometry document."""

    static_path = tmp_path / "native-static.npz"
    np.savez(static_path, HGT_M=np.zeros((target.ny, target.nx)))
    rich_path = tmp_path / "hrrr-static.json"
    rich_path.write_text(json.dumps({
        "schema": "gpuwm-native-hrrr-static-v2",
        "status": "PASS",
        "target_domain": target.to_payload(),
        "target_domain_sha256": target.identity_sha256(),
        "geometry": geometry,
        "cache": {
            "path": static_path.name,
            "bytes": static_path.stat().st_size,
            "sha256": hashlib.sha256(static_path.read_bytes()).hexdigest(),
        },
    }))
    spec_path = tmp_path / "domain.json"
    spec_path.write_text(json.dumps(target.to_payload()))
    return static_path, rich_path, spec_path


def test_hrrr_static_builder_geometry_is_accepted_by_its_own_verifier(tmp_path):
    """The producer's sealed document must satisfy the consumer's contract.

    v1.0.0 wrote the two independently; the builder omitted ``map_proj`` and
    the receipt equality check therefore failed for every HRRR area a user
    could build, with no user-side workaround.
    """

    target = HrrrTargetDomain(
        name="tiny-contract-fixture", map_proj="lambert",
        nx=14, ny=13, nz=9, dx_m=3000.0, dy_m=3000.0,
        ref_lat=35.0, ref_lon=-97.0, truelat1=30.0,
        truelat2=60.0, stand_lon=-97.0, time_step_seconds=15)
    geometry = native_static_geometry(target)
    assert geometry["map_proj"] == "lambert"

    static_path, rich_path, spec_path = _sealed_hrrr_static(
        tmp_path, target, geometry)
    written = write_hrrr_native_geometry_receipt(
        static_cache=static_path,
        hrrr_static_receipt=rich_path,
        output=tmp_path / "native-geometry.json",
        domain_spec=spec_path,
    )

    assert written["geometry"] == geometry
    # json round-trip too: the sealed document is read back from disk.
    assert json.loads(rich_path.read_text())["geometry"] == geometry


def test_hrrr_geometry_receipt_still_refuses_a_key_short_document(tmp_path):
    """The equality guard stays strict; the fix was on the producer's side."""

    target = HrrrTargetDomain(
        name="tiny-contract-fixture", map_proj="lambert",
        nx=14, ny=13, nz=9, dx_m=3000.0, dy_m=3000.0,
        ref_lat=35.0, ref_lon=-97.0, truelat1=30.0,
        truelat2=60.0, stand_lon=-97.0, time_step_seconds=15)
    geometry = native_static_geometry(target)
    v100_shape = {k: v for k, v in geometry.items() if k != "map_proj"}

    static_path, rich_path, spec_path = _sealed_hrrr_static(
        tmp_path, target, v100_shape)
    output = tmp_path / "native-geometry.json"
    with pytest.raises(ValueError, match="geometry differs from its target"):
        write_hrrr_native_geometry_receipt(
            static_cache=static_path,
            hrrr_static_receipt=rich_path,
            output=output,
            domain_spec=spec_path,
        )
    assert not output.exists()


def _complete_native_static(grid):
    ny, nx = grid.e_sn - 1, grid.e_we - 1
    mass = (ny, nx)
    fields = {
        "HGT_M": np.ones(mass),
        "LANDMASK": np.ones(mass),
        "LU_INDEX": np.ones(mass),
        "SCT_DOM": np.ones(mass),
        "SCB_DOM": np.ones(mass),
        "SNOALB": np.zeros(mass),
        "SOILTEMP": np.full(mass, 285.0),
        "TMN": np.full(mass, 285.0),
        "GREENFRAC": np.full((12, *mass), 0.5),
        "LAI12M": np.full((12, *mass), 2.0),
        "ALBEDO12M": np.full((12, *mass), 20.0),
        "LANDUSEF": np.zeros((21, *mass)),
        "SOILCTOP": np.zeros((16, *mass)),
        "SOILCBOT": np.zeros((16, *mass)),
    }
    for name in ("LANDUSEF", "SOILCTOP", "SOILCBOT"):
        fields[name][0] = 1.0
    return fields


def test_named_sources_accept_branched_arbitrary_d06_lambert_geometry(tmp_path):
    root = Path(__file__).resolve().parents[1]
    template = (
        root / "configs" / "gfs_wrf_hierarchy_proof.toml"
    ).read_text(encoding="utf-8")
    shared = template.split("[[domain]]", 1)[0]
    parents = (0, 1, 1, 2, 2, 3)
    starts_i = (1, 50, 90, 20, 60, 30)
    starts_j = (1, 40, 50, 20, 60, 30)
    sizes = ((250, 200), (120, 120), (120, 120),
             (60, 60), (60, 60), (60, 60))
    blocks = []
    for index, (parent, i_start, j_start, (nx, ny)) in enumerate(zip(
            parents, starts_i, starts_j, sizes), start=1):
        root_domain = index == 1
        block = f"""
[[domain]]
grid_id = {index}
parent_id = {parent}
i_parent_start = {i_start}
j_parent_start = {j_start}
parent_grid_ratio = {1 if root_domain else 3}
parent_time_step_ratio = {1 if root_domain else 3}
nx = {nx}
ny = {ny}
specified = {str(root_domain).lower()}
nested = {str(not root_domain).lower()}
history_interval_s = 3600.0
radt = 12.0
cu_physics = 0
cudt_minutes = 0.0
diff_6th_factor = 0.12
"""
        if root_domain:
            block += "time_step = 5\ndx = 12000.0\n"
        blocks.append(block)
    experiment_path = tmp_path / "d06.toml"
    experiment_path.write_text(shared + "".join(blocks), encoding="utf-8")
    exp = load_experiment(experiment_path)

    wps_path = tmp_path / "namelist.wps"
    wps_path.write_text(f"""
&share
 max_dom = 6,
/
&geogrid
 parent_id = {', '.join(map(str, (1, *parents[1:])))},
 parent_grid_ratio = 1, 3, 3, 3, 3, 3,
 i_parent_start = {', '.join(map(str, starts_i))},
 j_parent_start = {', '.join(map(str, starts_j))},
 e_we = {', '.join(str(nx + 1) for nx, _ in sizes)},
 e_sn = {', '.join(str(ny + 1) for _, ny in sizes)},
 dx = 12000,
 dy = 12000,
 map_proj = 'lambert',
 ref_lat = 39.6848,
 ref_lon = -83.9297,
 truelat1 = 30.0,
 truelat2 = 60.0,
 stand_lon = -83.9297,
/
""", encoding="utf-8")

    for source_name in ("ERA5", "GFS"):
        grids = validate_native_lambert_contracts(
            exp,
            wps_path,
            source_name=source_name,
            source_top_pressure_pa=10_000.0,
        )
        assert len(grids) == 6
        assert [(grid.e_we, grid.e_sn) for grid in grids] == [
            (nx + 1, ny + 1) for nx, ny in sizes
        ]
    assert [domain.parent_id for domain in exp.domains] == list(parents)


@pytest.mark.parametrize(("mutation", "message"), (
    ("schema", "unrecognized or non-PASS"),
    ("status", "unrecognized or non-PASS"),
    ("identity", "target identity mismatch"),
    ("document", "target document mismatch"),
    ("cache-bytes", "does not bind the static cache"),
    ("cache-sha256", "does not bind the static cache"),
    ("cache-path", "cache file name mismatch"),
    ("geometry", "geometry differs from its target"),
))
def test_hrrr_static_conversion_rejects_mutated_evidence(
        tmp_path, mutation, message):
    target = HrrrTargetDomain(
        name="tiny-contract-fixture", map_proj="lambert",
        nx=14, ny=13, nz=9, dx_m=3000.0, dy_m=3000.0,
        ref_lat=35.0, ref_lon=-97.0, truelat1=30.0,
        truelat2=60.0, stand_lon=-97.0, time_step_seconds=15)
    grid = target.grid()
    cfg = SimpleNamespace(
        nx=target.nx, ny=target.ny, nz=target.nz,
        dx=target.dx_m, dy=target.dy_m)
    static_path = tmp_path / "native-static.npz"
    np.savez(static_path, HGT_M=np.zeros((target.ny, target.nx)))
    payload = {
        "schema": "gpuwm-native-hrrr-static-v2",
        "status": "PASS",
        "target_domain": target.to_payload(),
        "target_domain_sha256": target.identity_sha256(),
        "geometry": native_geometry_contract(grid, cfg),
        "cache": {
            "path": static_path.name,
            "bytes": static_path.stat().st_size,
            "sha256": hashlib.sha256(static_path.read_bytes()).hexdigest(),
        },
    }
    mutated = deepcopy(payload)
    if mutation == "schema":
        mutated["schema"] = "gpuwm-native-hrrr-static-v999"
    elif mutation == "status":
        mutated["status"] = "FAIL"
    elif mutation == "identity":
        mutated["target_domain_sha256"] = "0" * 64
    elif mutation == "document":
        mutated["target_domain"]["name"] = "another-domain"
    elif mutation == "cache-bytes":
        mutated["cache"]["bytes"] += 1
    elif mutation == "cache-sha256":
        mutated["cache"]["sha256"] = "0" * 64
    elif mutation == "cache-path":
        mutated["cache"]["path"] = "another-static.npz"
    elif mutation == "geometry":
        mutated["geometry"]["mass_shape"][1] += 1
    else:  # pragma: no cover - parameter list and handler must stay paired.
        raise AssertionError(mutation)
    rich_path = tmp_path / "hrrr-static.json"
    rich_path.write_text(json.dumps(mutated))
    spec = tmp_path / "domain.json"
    spec.write_text(json.dumps(target.to_payload()))
    output = tmp_path / "native-geometry.json"

    with pytest.raises(ValueError, match=message):
        write_hrrr_native_geometry_receipt(
            static_cache=static_path,
            hrrr_static_receipt=rich_path,
            output=output,
            domain_spec=spec,
        )
    assert not output.exists()


def test_hrrr_static_conversion_rejects_overwrite(tmp_path):
    target = HrrrTargetDomain.legacy_500x500()
    grid = target.grid()
    cfg = SimpleNamespace(
        nx=target.nx, ny=target.ny, nz=target.nz,
        dx=target.dx_m, dy=target.dy_m)
    static_path = tmp_path / "native-static.npz"
    np.savez(static_path, HGT_M=np.zeros((1, 1)))
    rich_path = tmp_path / "hrrr-static.json"
    rich_path.write_text(json.dumps({
        "schema": "gpuwm-native-hrrr-static-500x500-v1",
        "status": "PASS",
        "geometry": native_geometry_contract(grid, cfg),
        "cache": {
            "path": static_path.name,
            "bytes": static_path.stat().st_size,
            "sha256": hashlib.sha256(static_path.read_bytes()).hexdigest(),
        },
    }))
    output = tmp_path / "native-geometry.json"

    write_hrrr_native_geometry_receipt(
        static_cache=static_path,
        hrrr_static_receipt=rich_path,
        output=output,
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_hrrr_native_geometry_receipt(
            static_cache=static_path,
            hrrr_static_receipt=rich_path,
            output=output,
        )


def test_canonical_noah_surface_has_exact_wrf_export_inventory():
    values = {
        "tsk": object(),
        "soil_temperature": object(),
        "soil_moisture": object(),
        "liquid_moisture": object(),
        "deep_soil_temperature": object(),
        "xice": object(),
        "xland": object(),
        "landmask": object(),
        "snow_water": object(),
        "snow_depth": object(),
    }
    surface = canonical_noah_surface(SimpleNamespace(**values))
    assert tuple(surface) == (
        "TSK", "TSLB", "SMOIS", "SH2O", "TMN", "SEAICE", "XLAND",
        "LANDMASK", "SNOW", "SNOWH")
    assert surface["TSK"] is values["tsk"]
    assert surface["TSLB"] is values["soil_temperature"]


def test_native_static_export_fields_regenerates_geometry_and_rejects_drift():
    grid = _grid()
    fields = native_static_export_fields(
        {"HGT_M": np.zeros((3, 4))}, grid)
    assert {"MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "F", "E",
            "SINALPHA", "COSALPHA"} <= set(fields)
    np.testing.assert_array_equal(fields["MAPFAC_M"], grid.mapfac_m())

    with pytest.raises(ValueError, match="MAPFAC_M"):
        native_static_export_fields(
            {"MAPFAC_M": np.zeros_like(grid.mapfac_m())}, grid)


def test_native_static_contract_is_exact_modis_noah_and_preserves_geometry():
    grid = _grid()
    fields = _complete_native_static(grid)
    fields["MAPFAC_M"] = grid.mapfac_m()

    actual = validate_native_static_fields(fields, grid, 3, 4)

    assert actual["LANDUSEF"].shape == (21, 3, 4)
    assert actual["SOILCTOP"].shape == (16, 3, 4)
    np.testing.assert_array_equal(actual["MAPFAC_M"], grid.mapfac_m())
    assert NATIVE_LANDUSE_IDENTITY == {
        "MMINLU": "MODIFIED_IGBP_MODIS_NOAH",
        "ISWATER": 17,
        "ISLAKE": 21,
        "ISICE": 15,
    }


@pytest.mark.parametrize(
    "field,replacement,error",
    [
        ("SOILTEMP", np.ones((7, 3, 4)), "SOILTEMP has shape"),
        ("GREENFRAC", np.ones((13, 3, 4)), "GREENFRAC has shape"),
        ("LANDMASK", np.full((3, 4), 0.5), "LANDMASK must be exactly binary"),
    ],
)
def test_native_static_contract_rejects_malformed_shapes_and_categories(
        field, replacement, error):
    grid = _grid()
    fields = _complete_native_static(grid)
    fields[field] = replacement

    with pytest.raises(ValueError, match=error):
        validate_native_static_fields(fields, grid, 3, 4)


def test_native_static_contract_rejects_persisted_geometry_drift():
    grid = _grid()
    fields = _complete_native_static(grid)
    fields["MAPFAC_M"] = np.zeros((3, 4))

    with pytest.raises(ValueError, match="MAPFAC_M"):
        validate_native_static_fields(fields, grid, 3, 4)
