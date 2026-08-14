from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

import gpuwm.era5_direct as era5_direct
from gpuwm.era5_direct import (
    INPUT_MANIFEST_SCHEMA,
    _STATIC_REQUIRED,
    _domain_source_orography,
    _load_static,
    _verify_input_manifest,
    _write_geometry_receipt,
)


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _complete_static_fields(ny: int = 2, nx: int = 3):
    mass = (ny, nx)
    fields = {
        "HGT_M": np.ones(mass, dtype=np.float64),
        "LANDMASK": np.ones(mass, dtype=np.float64),
        "LU_INDEX": np.ones(mass, dtype=np.float64),
        "SCT_DOM": np.ones(mass, dtype=np.float64),
        "SCB_DOM": np.ones(mass, dtype=np.float64),
        "SNOALB": np.zeros(mass, dtype=np.float64),
        "SOILTEMP": np.full(mass, 285.0, dtype=np.float64),
        "TMN": np.full(mass, 285.0, dtype=np.float64),
        "GREENFRAC": np.full((12, *mass), 0.5, dtype=np.float64),
        "LAI12M": np.full((12, *mass), 2.0, dtype=np.float64),
        "ALBEDO12M": np.full((12, *mass), 20.0, dtype=np.float64),
        "LANDUSEF": np.zeros((21, *mass), dtype=np.float64),
        "SOILCTOP": np.zeros((16, *mass), dtype=np.float64),
        "SOILCBOT": np.zeros((16, *mass), dtype=np.float64),
    }
    for name in ("LANDUSEF", "SOILCTOP", "SOILCBOT"):
        fields[name][0] = 1.0
    assert set(fields) == set(_STATIC_REQUIRED)
    return fields


def test_era5_domain_orography_bindings_are_exact_and_ordered(tmp_path):
    declaration = _domain_source_orography([
        f"d06={tmp_path / 'd06.nc'}",
        f"d01={tmp_path / 'd01.nc'}",
        f"d03={tmp_path / 'd03.nc'}",
        f"d02={tmp_path / 'd02.nc'}",
        f"d05={tmp_path / 'd05.nc'}",
        f"d04={tmp_path / 'd04.nc'}",
    ], "SOILHGT")
    assert tuple(domain_id for domain_id, _ in declaration.by_domain) == (
        1, 2, 3, 4, 5, 6)
    with pytest.raises(ValueError, match="duplicate"):
        _domain_source_orography([
            f"d01={tmp_path / 'first.nc'}",
            f"d01={tmp_path / 'second.nc'}",
        ], "SOILHGT")
    with pytest.raises(ValueError, match="dNN=PATH"):
        _domain_source_orography(["one=/tmp/d01.nc"], "SOILHGT")


def test_root_static_can_be_built_directly_from_wps_geog(
        tmp_path, monkeypatch):
    catalog = object()
    receipt = {"schema": "verified-static", "status": "PASS"}
    observed = {}
    monkeypatch.setattr(
        era5_direct,
        "verified_static_catalog",
        lambda wps, geog, ids: (
            observed.update(wps=wps, geog=geog, ids=tuple(ids)) or catalog,
            receipt,
        ),
    )
    fields = _complete_static_fields()
    monkeypatch.setattr(
        era5_direct,
        "build_static_for_domain",
        lambda grid, actual_catalog, domain_id: (
            observed.update(
                grid=grid, catalog=actual_catalog, domain_id=domain_id)
            or fields
        ),
    )
    # The land-use attributes come from the SAME catalog the statics were
    # built from, so the water-temperature assembly and the statics cannot
    # disagree about which category is a lake.
    attrs = {"MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "ISWATER": 17,
             "ISLAKE": 21, "ISICE": 15, "ISURBAN": 13}
    monkeypatch.setattr(
        era5_direct,
        "geog_selection_from_catalog",
        lambda actual_catalog, domain_id: (
            observed.update(
                selection_catalog=actual_catalog,
                selection_domain_id=domain_id)
            or SimpleNamespace(landuse_global_attrs=lambda: attrs)
        ),
    )
    plane = np.ones((2, 3), dtype=np.float64)
    grid = SimpleNamespace(
        mapfac_m=lambda: plane,
        mapfac_u=lambda: np.ones((2, 4)),
        mapfac_v=lambda: np.ones((3, 3)),
        coriolis_m=lambda: (plane * 2, plane * 3),
        rotation_m=lambda: (plane * 4, plane * 5),
    )
    cfg = SimpleNamespace(ny=2, nx=3)

    static, actual_receipt, landuse_attrs = era5_direct._static_from_geog(
        tmp_path / "namelist.wps", tmp_path / "WPS_GEOG", grid, cfg)
    assert actual_receipt is receipt
    assert observed["ids"] == (1,)
    assert observed["catalog"] is catalog
    assert observed["domain_id"] == 1
    assert static["MAPFAC_U"].shape == (2, 4)
    assert landuse_attrs is attrs
    assert observed["selection_catalog"] is catalog
    assert observed["selection_domain_id"] == 1


def test_static_validation_rejects_zero_terrain_over_land_before_export():
    fields = _complete_static_fields()
    fields["HGT_M"][:] = 0.0
    fields["LANDMASK"][:] = 1.0
    plane = np.ones((2, 3), dtype=np.float64)
    grid = SimpleNamespace(
        mapfac_m=lambda: plane,
        mapfac_u=lambda: np.ones((2, 4)),
        mapfac_v=lambda: np.ones((3, 3)),
        coriolis_m=lambda: (plane, plane),
        rotation_m=lambda: (plane, plane),
    )

    with pytest.raises(ValueError, match="identically zero over every land"):
        era5_direct._validated_static(fields, grid, 2, 3)


def test_era5_input_manifest_binds_every_role(tmp_path):
    role_paths = {}
    files = {}
    for role in ("grib", "vtable", "static_input"):
        path = tmp_path / f"{role}.bin"
        path.write_bytes(role.encode("ascii"))
        role_paths[role] = path
        files[role] = {"name": path.name, "sha256": _sha256(path)}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "schema": INPUT_MANIFEST_SCHEMA,
        "files": files,
    }))

    actual = _verify_input_manifest(
        manifest_path, _sha256(manifest_path), role_paths)
    assert actual["schema"] == INPUT_MANIFEST_SCHEMA

    role_paths["grib"].write_bytes(b"mutated")
    with pytest.raises(ValueError, match="digest mismatch for grib"):
        _verify_input_manifest(
            manifest_path, _sha256(manifest_path), role_paths)


def test_era5_static_loader_is_shape_checked_and_source_neutral(tmp_path):
    values = _complete_static_fields()
    static_path = tmp_path / "static.npz"
    np.savez(static_path, **values)
    plane = np.ones((2, 3), dtype=np.float64)

    class Grid:
        def mapfac_m(self):
            return plane

        def mapfac_u(self):
            return np.ones((2, 4), dtype=np.float64)

        def mapfac_v(self):
            return np.ones((3, 3), dtype=np.float64)

        def coriolis_m(self):
            return plane, plane

        def rotation_m(self):
            return np.zeros_like(plane), plane

    actual = _load_static(static_path, Grid(), 2, 3)
    assert set(_STATIC_REQUIRED) < set(actual)
    assert actual["MAPFAC_U"].shape == (2, 4)

    values["TMN"] = np.ones((3, 3), dtype=np.float32)
    np.savez(static_path, **values)
    with pytest.raises(ValueError, match="static field TMN"):
        _load_static(static_path, Grid(), 2, 3)


def test_era5_geometry_receipt_has_portable_cache_reference(tmp_path):
    cache = tmp_path / "native-static.npz"
    cache.write_bytes(b"static")
    plane = np.ones((2, 3), dtype=np.float64)
    grid = SimpleNamespace(
        ref_lat=39.0,
        ref_lon=-84.0,
        truelat1=30.0,
        truelat2=60.0,
        stand_lon=-84.0,
        e_we=4,
        e_sn=3,
        dx=12000.0,
        dy=12000.0,
        cen_lat=39.0,
        cen_lon=-84.0,
        latlon_mass=lambda: (plane * 39.0, plane * -84.0),
    )
    cfg = SimpleNamespace(ny=2, nx=3, nz=49, dx=12000.0, dy=12000.0)
    receipt_path = tmp_path / "receipt.json"
    _write_geometry_receipt(receipt_path, grid, cfg, cache)
    receipt = json.loads(receipt_path.read_text())
    assert receipt["cache"]["path"] == "native-static.npz"
    assert receipt["cache"]["sha256"] == _sha256(cache)


# ---------------------------------------------------------------------------
# the soil hand-off on the route whose orography rides inside the GRIB
#
# The wizard emits a case with no source-orography artifact, because `gpuwm
# fetch --source era5` writes the invariant geopotential INTO
# era5-combined.grib.  On that route `source_terrain` is None by
# construction, and the soil call used to forward that None beside a real
# HGT_M -- so preprocess_noah_soil's all-or-none guard refused the whole
# preparation.  initialize_real had resolved the same field for itself one
# screen earlier; only the soil hand-off had not.

_ERA5_TEMP_NAMES = ("ST000007", "ST007028", "ST028100", "ST100289")
_ERA5_MOIST_NAMES = ("SM000007", "SM007028", "SM028100", "SM100289")


def _era5_land_fields(shape=(3, 4), *, skin=290.0, orography=200.0):
    """What interpolate_era5_to_lambert emits on the SOILGEO route.

    SOURCE_OROGRAPHY is the renamed, remapped SOILGEO record: ERA5's own
    terrain on the target mass grid, in metres.
    """
    fields = {
        "LANDSEA": np.ones(shape),
        "SKINTEMP": np.full(shape, skin),
    }
    fields.update({name: np.full(shape, 288.0) for name in _ERA5_TEMP_NAMES})
    fields.update({name: np.full(shape, 0.30) for name in _ERA5_MOIST_NAMES})
    fields["SOURCE_OROGRAPHY"] = np.full(shape, orography)
    return fields


def test_the_gribs_own_orography_is_what_the_soil_hand_off_lapses_from():
    fields = _era5_land_fields(orography=200.0)
    resolved = era5_direct._soil_source_orography(None, fields)
    # The SOURCE terrain, not the target HGT_M and not None.
    assert resolved is fields["SOURCE_OROGRAPHY"]
    assert float(np.asarray(resolved).flat[0]) == 200.0


def test_a_declared_artifact_still_outranks_the_embedded_record():
    fields = _era5_land_fields(orography=200.0)
    declared = np.full((3, 4), 640.0)
    assert era5_direct._soil_source_orography(declared, fields) is declared


def test_a_route_carrying_neither_source_orography_is_refused_by_name():
    fields = _era5_land_fields()
    del fields["SOURCE_OROGRAPHY"]
    with pytest.raises(ValueError, match="no source orography"):
        era5_direct._soil_source_orography(None, fields)


def test_the_soilgeo_route_reaches_past_the_all_or_none_soil_guard():
    """The wall itself: this raised ValueError before the resolver existed.

    Reverting `_soil_source_orography(source_terrain, initial_met.fields)`
    to a bare `source_terrain` puts this back to
    "terrain and source_orography must be provided together".
    """
    from gpuwm.ingest.soil import preprocess_noah_soil

    shape = (3, 4)
    fields = _era5_land_fields(shape, skin=290.0, orography=200.0)
    terrain = np.full(shape, 700.0)

    state = preprocess_noah_soil(
        fields,
        soil_type=np.full(shape, 6),
        deep_soil_temperature=np.full(shape, 285.0),
        landmask=np.ones(shape),
        terrain=terrain,
        source_orography=era5_direct._soil_source_orography(None, fields),
    )

    # And the adjustment is real, not merely tolerated: WRF's
    # adjust_soil_temp_new lapse over the 500 m the model grid climbs
    # above ERA5's own terrain is -0.0065 * 500 = -3.25 K on land skin.
    assert np.allclose(np.asarray(state.tsk), 290.0 - 3.25)
    assert np.allclose(np.asarray(state.soil_temperature), 288.0 - 3.25)
