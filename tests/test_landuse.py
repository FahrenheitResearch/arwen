"""WRF v4.6.1 cold-start land-use and inland-lake semantics."""

import os
from datetime import datetime
import hashlib
from pathlib import Path
from types import SimpleNamespace

from netCDF4 import Dataset
import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]
BUNDLE = Path(os.environ.get("GPUWM_TEST_WRF74_BUNDLE",
                    "gpuwm-fixture-unset/wrf74-bundle"))
WRF_RUN = BUNDLE / "WRF_source_v4.6.1_group" / "run"
PACKAGED = REPO / "gpuwm" / "data" / "noah_tables" / "LANDUSE.TBL"
requires_real74 = pytest.mark.skipif(
    not all((BUNDLE / "geo_em" / f"geo_em.d0{domain}.nc").is_file()
            for domain in range(1, 5))
    or not all(any((BUNDLE / "wrfout_reference").glob(
        f"wrfout_d0{domain}_*")) for domain in range(1, 5)),
    reason="WRF v4.6.1 real74 geo_em/wrfout bundle is absent")


def test_packaged_landuse_table_is_byte_identical_wrf_v461_asset():
    expected_sha = (
        "cafdb5f4982b88c93f2cb321f18d9559e7c03b6d213388d5ae3d07b7280caa08")
    assert hashlib.sha256(PACKAGED.read_bytes()).hexdigest() == expected_sha
    if (WRF_RUN / "LANDUSE.TBL").is_file():
        assert PACKAGED.read_bytes() == (WRF_RUN / "LANDUSE.TBL").read_bytes()


def test_ingest_preflight_catalogs_and_hash_pins_landuse_table():
    from types import SimpleNamespace

    from gpuwm.ingest.preflight import _TABLE_SHA256, _table_files

    case_data = SimpleNamespace(table_root=REPO / "gpuwm" / "data")
    assert PACKAGED in _table_files(case_data)
    assert _TABLE_SHA256["noah_tables/LANDUSE.TBL"] == (
        "cafdb5f4982b88c93f2cb321f18d9559e7c03b6d213388d5ae3d07b7280caa08")


def test_modified_igbp_landuse_rows_and_noah_water_category_are_exact():
    from gpuwm.core.landuse import load_landuse_table
    from gpuwm.core.noah import load_tables

    table = load_landuse_table()
    assert (table.lutype, table.lucats, table.luseas) == (
        "MODIFIED_IGBP_MODIS_NOAH", 61, 2)
    # LANDUSE.TBL list-directed rows: (ALBD, SLMO, SFEM, SFZ0, THERIN,
    # SCFX, SFHC).  Array order is summer=0, winter=1 and category-1.
    np.testing.assert_array_equal(
        table.values[1, 9],
        np.array([23.0, 0.30, 0.92, 10.0, 4.0, 2.0, 20.8e5]))
    np.testing.assert_array_equal(
        table.values[1, 16],
        np.array([8.0, 1.0, 0.98, 0.01, 6.0, 0.0, 9.0e25]))

    # Noah's paired VEGPARM table has only 20 categories.  WRF therefore
    # routes raw MODIS-with-lakes category 21 through water category 17
    # before Noah; category 21 must never index this table directly.
    veg = load_tables()
    assert (veg.lutype, veg.lucats) == (
        "MODIFIED_IGBP_MODIS_NOAH", 20)
    assert veg.z0mintbl[16] == pytest.approx(1.0e-4)
    assert veg.emissmintbl[16] == pytest.approx(0.98)
    assert veg.albedomintbl[16] == pytest.approx(0.08)


def test_wrf_cold_start_landuse_oracle_includes_freshwater_lake_routing():
    from gpuwm.core.landuse import initialize_landuse

    raw_lu = np.array([[1, 10, 17, 21]], np.int32)
    result = initialize_landuse(
        raw_lu, soil_type=np.array([[6, 6, 14, 14]], np.int32),
        landmask=np.array([[1.0, 1.0, 0.0, 0.0]]),
        snow=np.array([[10.0, 0.0, 0.0, 0.0]]), xice=0.0,
        valid_time=datetime(1974, 4, 3, 12), cen_lat=39.6848,
        mminlu="MODIFIED_IGBP_MODIS_NOAH", iswater=17, islake=21,
        isice=15)

    np.testing.assert_array_equal(result.ivgtyp, [[1, 10, 17, 17]])
    np.testing.assert_array_equal(result.isltyp, [[6, 6, 14, 14]])
    np.testing.assert_array_equal(result.landmask, [[1.0, 1.0, 0.0, 0.0]])
    np.testing.assert_array_equal(result.xland, [[1.0, 1.0, 2.0, 2.0]])
    np.testing.assert_array_equal(result.lakemask, [[0.0, 0.0, 0.0, 1.0]])
    np.testing.assert_array_equal(result.snowc, [[1.0, 0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(result.pblh, 0.0)
    np.testing.assert_array_equal(result.ust, np.float32(1.0e-4))
    np.testing.assert_array_equal(
        result.mavail,
        np.array([[0.60, 0.30, 1.0, 1.0]], dtype=np.float32))
    np.testing.assert_allclose(
        result.z0, [[0.50, 0.10, 1.0e-4, 1.0e-4]], rtol=0.0,
        atol=1.0e-8)
    np.testing.assert_array_equal(result.znt, result.z0)
    np.testing.assert_allclose(
        result.albbck, [[0.12, 0.23, 0.08, 0.08]], rtol=0.0,
        atol=1.0e-8)
    # WRF uses ALBEDO=ALBBCK*(1+SCFX) for SNOWC>0.5 when usemonalb=false.
    np.testing.assert_allclose(
        result.albedo, [[0.48, 0.23, 0.08, 0.08]], rtol=0.0,
        atol=3.0e-8)
    np.testing.assert_array_equal(
        result.embck,
        np.array([[0.95, 0.92, 0.98, 0.98]], dtype=np.float32))
    np.testing.assert_array_equal(result.emiss, result.embck)

    # Source/output parity: lake is freshwater open water, not Noah land.
    assert result.xland[0, 3] >= 1.5
    assert result.lakemask[0, 3] == 1.0
    assert result.lakemask[0, 2] == 0.0  # ocean gets salinity correction


def test_landuse_season_flips_by_central_latitude_like_wrf():
    from gpuwm.core.landuse import initialize_landuse

    inputs = dict(
        lu_index=np.array([[10]], np.int32),
        soil_type=np.array([[6]], np.int32), landmask=np.array([[1.0]]),
        snow=0.0, xice=0.0, valid_time=datetime(1974, 4, 3, 12),
        mminlu="MODIFIED_IGBP_MODIS_NOAH", iswater=17, islake=21,
        isice=15)
    north = initialize_landuse(cen_lat=1.0, **inputs)
    south = initialize_landuse(cen_lat=-1.0, **inputs)
    assert north.season == 2 and south.season == 1
    assert north.albbck[0, 0] == pytest.approx(0.23)
    assert south.albbck[0, 0] == pytest.approx(0.19)
    assert north.mavail[0, 0] == pytest.approx(0.30)
    assert south.mavail[0, 0] == pytest.approx(0.15)

    # SCFX is the one non-seasonal Fortran array: reading the winter block
    # last overwrites its summer records (module_physics_init.F:1907-1918).
    summer_snow = initialize_landuse(
        np.array([[1]], np.int32), soil_type=np.array([[6]], np.int32),
        landmask=np.array([[1.0]]), snow=10.0, xice=0.0,
        valid_time=datetime(1974, 7, 1), cen_lat=1.0,
        mminlu="MODIFIED_IGBP_MODIS_NOAH", iswater=17, islake=21,
        isice=15)
    assert summer_snow.season == 1
    assert summer_snow.albedo[0, 0] == pytest.approx(0.48)


def test_seaice_category_routing_precedes_table_lookup_but_soil_state_waits():
    from gpuwm.core.landuse import initialize_landuse

    result = initialize_landuse(
        np.array([[17]], np.int32), soil_type=np.array([[14]], np.int32),
        landmask=np.array([[0.0]]), snow=0.0, xice=1.0,
        valid_time=datetime(1974, 4, 3, 12), cen_lat=39.6848,
        mminlu="MODIFIED_IGBP_MODIS_NOAH", iswater=17, islake=21,
        isice=15)
    # adjust_for_seaice_post sets the category/masks before physics_init.
    np.testing.assert_array_equal(result.ivgtyp, [[15]])
    np.testing.assert_array_equal(result.isltyp, [[16]])
    np.testing.assert_array_equal(result.landmask, [[1.0]])
    np.testing.assert_array_equal(result.xland, [[1.0]])
    np.testing.assert_array_equal(result.lakemask, [[0.0]])
    np.testing.assert_array_equal(result.albbck, np.float32(0.70))
    np.testing.assert_array_equal(result.embck, np.float32(0.95))
    np.testing.assert_array_equal(result.z0, np.float32(0.001))
    # SMOIS/SH2O/TSLB are deliberately absent: their timestep-one
    # water/sea-ice initialization is a separate porting batch.
    assert not any(hasattr(result, name) for name in ("smois", "sh2o", "tslb"))


def test_physics_driver_consumes_one_coherent_landuse_initialization(
        monkeypatch):
    from gpuwm.config import RunConfig
    from gpuwm.core import physics
    from gpuwm.core.landuse import initialize_landuse

    # Exercise allocation/wiring on NumPy only; no CUDA context is created.
    monkeypatch.setattr(physics, "cp", np)
    cfg = RunConfig(
        nx=4, ny=1, nz=16, dx=2000.0, dy=2000.0, ztop=8000.0,
        dt=10.0, run_seconds=0.0, time_step_sound=4,
        sf_sfclay_physics=1)
    state = SimpleNamespace(
        mup=np.ones((1, 4), dtype=np.float32),
        p=np.ones((16, 1, 4), dtype=np.float32), physics=None)
    landuse = initialize_landuse(
        np.array([[1, 10, 17, 21]], np.int32),
        soil_type=np.array([[6, 6, 14, 14]], np.int32),
        landmask=np.array([[1.0, 1.0, 0.0, 0.0]]), snow=0.0, xice=0.0,
        valid_time=datetime(1974, 4, 3, 12), cen_lat=39.6848,
        mminlu="MODIFIED_IGBP_MODIS_NOAH", iswater=17, islake=21,
        isice=15)

    driver = physics.initialize_physics(state, cfg, landuse=landuse)
    for name in (
            "landmask", "xland", "lakemask", "ivgtyp", "isltyp",
            "snowc", "pblh", "ust", "mavail", "z0", "znt", "albbck",
            "albedo", "embck", "emiss"):
        np.testing.assert_array_equal(
            driver.fields[name], getattr(landuse, name), err_msg=name)


@requires_real74
def test_real74_lake_counts_and_routing_match_wrf_reference_outputs():
    from gpuwm.core.landuse import initialize_landuse

    expected_counts = (1849, 14932, 10758, 1958)
    total = 0
    for domain, expected_count in enumerate(expected_counts, start=1):
        with Dataset(BUNDLE / "geo_em" / f"geo_em.d0{domain}.nc") as geo:
            lu_index = np.asarray(geo["LU_INDEX"][0])
            landmask = np.asarray(geo["LANDMASK"][0])
            soil_type = np.asarray(geo["SCT_DOM"][0])
        lake = lu_index == 21
        assert int(lake.sum()) == expected_count
        total += int(lake.sum())
        initialized = initialize_landuse(
            lu_index, soil_type=soil_type, landmask=landmask,
            snow=np.zeros_like(landmask), xice=np.zeros_like(landmask),
            valid_time=datetime(1974, 4, 3, 12), cen_lat=39.6848,
            mminlu="MODIFIED_IGBP_MODIS_NOAH", iswater=17, islake=21,
            isice=15)
        np.testing.assert_array_equal(initialized.ivgtyp[lake], 17)
        np.testing.assert_array_equal(initialized.isltyp[lake], 14)
        np.testing.assert_array_equal(initialized.landmask[lake], 0.0)
        np.testing.assert_array_equal(initialized.xland[lake], 2.0)
        np.testing.assert_array_equal(initialized.lakemask[lake], 1.0)

        wrf_path = next((BUNDLE / "wrfout_reference").glob(
            f"wrfout_d0{domain}_*"))
        with Dataset(wrf_path) as wrf:
            for name, expected in {
                    "IVGTYP": 17, "ISLTYP": 14, "LANDMASK": 0.0,
                    "XLAND": 2.0, "LAKEMASK": 1.0}.items():
                np.testing.assert_array_equal(
                    np.asarray(wrf[name][0])[lake], expected)
    assert total == 29497
