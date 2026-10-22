"""NOAA's operational HRRR geog_data_res builds: modis_15s, modis_fpar.

The WPS partner of the operational namelist (NOAA-EMC/HRRR v4.1.21
parm/conus/hrrr_namelist.wps, vendored in tests/data/hrrr_v4121_namelists)
names ``geog_data_res = 'modis_15s+modis_fpar+modis_lai+30s'``.  Until
2.9.1 the importer refused ``modis_15s`` and ``modis_fpar`` as datasets
gpuwm does not build, so NOAA's own pair could not import even with its
smoke and HAILCAST settings changed.  WPS 4.6 GEOGRID.TBL.ARW maps
``modis_15s`` to the 15 arc-second MODIS land use (``modis_landuse_
20class_15s``, no lake class) and ``modis_fpar`` to the default greenness
fraction with the default interpolation; stock geogrid.exe writes
``NUM_LAND_CAT = 20`` for the first (measured 2026-10-15) and WRF's real.exe
stops on a namelist ``num_land_cat`` that differs.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from gpuwm import geog_assets
from gpuwm.namelist_import import NamelistRefusal, import_namelists
from gpuwm.static.build import (
    _DEFAULT_GEOG_DIRS, _GEOG_TOKEN_DIRECTORIES, _LANDUSE_CATEGORY_COUNTS,
    GeogSelection, geog_landuse_category_count,
)

ROOT = Path(__file__).resolve().parent
OPERATIONAL = ROOT / "data" / "hrrr_v4121_namelists"
HRRR_TOKENS = "modis_15s+modis_fpar+modis_lai+30s"

from operational_namelist_edits import importable


def test_hrrr_tokens_select_the_wps_table_datasets(tmp_path):
    selection = GeogSelection.from_tokens(tmp_path, HRRR_TOKENS)
    assert selection.landuse == "modis_landuse_20class_15s"
    assert selection.greenfrac == "greenfrac_fpar_modis"
    assert selection.lai == "lai_modis_30s"
    assert (selection.soil_top, selection.soil_bottom) == (
        "soiltype_top_30s", "soiltype_bot_30s")
    for field in ("terrain", "albedo", "snow_albedo", "soil_temperature"):
        assert getattr(selection, field) == _DEFAULT_GEOG_DIRS[field]


def test_modis_fpar_is_exactly_the_default_selection(tmp_path):
    fpar = GeogSelection.from_tokens(tmp_path, "modis_fpar")
    default = GeogSelection.from_tokens(tmp_path, "default")
    assert fpar == replace(default, resolution_tokens=("modis_fpar",))


def test_every_selectable_land_use_has_its_geogrid_category_count():
    landuse = {table["landuse"] for table in _GEOG_TOKEN_DIRECTORIES.values()
               if "landuse" in table} | {_DEFAULT_GEOG_DIRS["landuse"]}
    assert landuse <= set(_LANDUSE_CATEGORY_COUNTS)
    assert geog_landuse_category_count("default") == 21
    assert geog_landuse_category_count("5m") == 21
    assert geog_landuse_category_count(HRRR_TOKENS) == 20
    assert geog_landuse_category_count("default+modis_15s") == 21


def test_hrrr_datasets_are_fetchable_for_the_wrf_door():
    for name in ("modis_landuse_20class_15s", "lai_modis_30s"):
        row = geog_assets.archive_for(name)
        assert geog_assets.GEOG_CONSUMER_WRF in row.optional_for, name
        assert not row.fetch_by_default and row.required_by == ()
        assert row.available_sources == ("ncar",)
    lai = geog_assets.archive_for("lai_modis_30s")
    assert (lai.archive_bytes, lai.archive_sha256) == (
        749633975,
        "b8188f06717767d5ed8898e5afb3a9b8b5ad46b888a530177f161ef3623acf75")


def _pair(tmp_path, nl_text, *, num_land_cat=None):
    wps = (OPERATIONAL / "hrrr_namelist.wps").read_text(encoding="utf-8")
    # NOAA's script rewrites the WPS dates; align them with the namelist's.
    wps = wps.replace("2009-08-21_00:00:00", "2018-08-26_12:00:00")
    wps = wps.replace("2009-08-21_12:00:00", "2018-08-26_18:00:00")
    assert f"geog_data_res = '{HRRR_TOKENS}'" in wps
    if num_land_cat is not None:
        old = "num_land_cat                        = 21,"
        assert old in nl_text
        nl_text = nl_text.replace(
            old, f"num_land_cat                        = {num_land_cat},")
    (tmp_path / "namelist.wps").write_text(wps, encoding="utf-8")
    (tmp_path / "namelist.input").write_text(nl_text, encoding="utf-8")
    return tmp_path / "namelist.wps", tmp_path / "namelist.input"


def test_noaa_pair_no_longer_refuses_its_geog_tokens(tmp_path):
    source = (OPERATIONAL / "hrrr_wrf.nl").read_text(encoding="utf-8")
    with pytest.raises(NamelistRefusal) as caught:
        import_namelists(*_pair(tmp_path, source))
    message = str(caught.value)
    assert "geog_data_res" not in message
    # fc1e2df360 carries the HRRR v4.1.21 fork's &chem keys
    # (simple_dir_fdb, enh_vermix, flam_part, aer_drydep_opt = 111), so the
    # one setting the pair still refuses is num_land_cat.
    for key in ("aer_drydep_opt", "enh_vermix", "flam_part",
                "simple_dir_fdb"):
        assert f"&chem {key} = " not in message, key
    assert "&physics hailcast_opt = " not in message
    # WRF's own stop: geogrid stamps 20 categories for modis_15s.
    assert "&physics num_land_cat = [21]" in message
    assert "Set num_land_cat = 20" in message
    assert "real.exe refuses" in message


def test_noaa_pair_imports_with_its_hrrr_geography(tmp_path):
    source = importable(
        (OPERATIONAL / "hrrr_wrf.nl").read_text(encoding="utf-8"))
    text, report = import_namelists(*_pair(tmp_path, source,
                                           num_land_cat=20))
    fixed = {(f.section, f.key): f for f in report.fixed}
    assert fixed[("physics", "num_land_cat")].fixed_value == 20
    assert "[static" not in text


def test_a_go_config_twin_naming_hrrr_tokens_fetches_their_datasets(tmp_path):
    """``gpuwm go`` builds from the ``<stem>.namelist.wps`` beside the
    configuration; fetch-geog --config and go's geography preflight read the
    same twin, so they name the datasets the run will open."""
    from gpuwm.cli import main
    from gpuwm.geog_assets import geography_for_run

    config = tmp_path / "area.toml"
    assert main(["domain", "--point=43.8,-86.6", "--card", "16gb",
                 "--ladder", "12", "--source", "gfs",
                 "--cycle", "2026-08-18T06", "--hours", "3",
                 "--out", str(config)]) == 0
    twin = config.with_name("area.namelist.wps")
    assert twin.is_file()
    default = geography_for_run(config)["datasets"]
    assert "modis_landuse_20class_30s_with_lakes" in default
    assert "lai_modis_10m" in default
    text = twin.read_text(encoding="utf-8")
    assert "'default'" in text
    twin.write_text(text.replace("'default'", f"'{HRRR_TOKENS}'"),
                    encoding="utf-8")
    hrrr = geography_for_run(config)["datasets"]
    assert "modis_landuse_20class_15s" in hrrr
    assert "lai_modis_30s" in hrrr
    assert "modis_landuse_20class_30s_with_lakes" not in hrrr
    assert "lai_modis_10m" not in hrrr


_INDEX_15S = """type=categorical
category_min=1
category_max=20
projection=regular_ll
dx=0.00416667
dy=0.00416667
known_x=1.0
known_y=1.0
known_lat=-89.9979167
known_lon=-179.9979167
wordsize=1
tile_x=2400
tile_y=2400
tile_z=1
units="category"
description="MODIS modified-IGBP landuse - 500 meter"
mminlu="MODIFIED_IGBP_MODIS_NOAH"
iswater=17
isice=15
isurban=13
"""


def test_a_land_use_without_a_lake_class_stamps_islake_minus_one(tmp_path):
    """The 15 arc-second index (NCAR's, verbatim) has no islake key; geogrid
    writes ISLAKE = -1 and NUM_LAND_CAT = 20.  The preparation refused it
    ("lacks required WRF metadata ['ISLAKE']") until 2.9.1."""
    lu = tmp_path / "modis_landuse_20class_15s"
    lu.mkdir()
    (lu / "index").write_text(_INDEX_15S, encoding="utf-8")
    attrs = GeogSelection.from_tokens(tmp_path, "modis_15s") \
        .landuse_global_attrs()
    assert (attrs["ISLAKE"], attrs["ISWATER"], attrs["NUM_LAND_CAT"]) == (
        -1, 17, 20)


def test_the_forecast_contract_admits_the_twenty_category_static():
    """The prepared static's LANDUSEF was refused at (20, ny, nx) ("expected
    (21, ...)") although geogrid builds exactly that for modis_15s."""
    import numpy as np
    from gpuwm.native_wrf_contract import _landuse_category_count

    assert _landuse_category_count(np.zeros((20, 2, 3))) == 20
    assert _landuse_category_count(np.zeros((21, 2, 3))) == 21
    # Any other count still fails the shape check against 21.
    assert _landuse_category_count(np.zeros((19, 2, 3))) == 21


def test_cg_topo_blend_takes_a_device_array():
    """NOAA's namelist sets smooth_cg_topo = .true.; prepared on the card,
    the source terrain arrives as a device array and the blend stopped at
    "Implicit conversion to a NumPy array is not allowed"."""
    import numpy as np
    from gpuwm.ingest.cg_topo import blend_root_terrain

    class Device:
        __array_priority__ = 0

        def __init__(self, values):
            self._values = values

        def get(self):
            return self._values.copy()

        def __array__(self, *args, **kwargs):
            raise TypeError("Implicit conversion to a NumPy array is not "
                            "allowed")

    rng = np.random.default_rng(3)
    fine = rng.uniform(0, 500, (12, 14))
    coarse = rng.uniform(0, 500, (12, 14))
    host = blend_root_terrain(fine, coarse, spec_bdy_width=1, blend_width=3)
    device = blend_root_terrain(fine, Device(coarse), spec_bdy_width=1,
                                blend_width=3)
    assert device.tobytes() == host.tobytes()


def test_the_native_identity_of_a_lake_free_static_has_no_lake_class():
    """The single-domain forecast handed every static ISLAKE = 21, so RUC's
    mosaic land use indexed LANDUSEF[20] of the 20-plane static ("index 20
    is out of bounds for axis 0 with size 20")."""
    import numpy as np
    from gpuwm.native_wrf_contract import (NATIVE_LANDUSE_IDENTITY,
                                           native_landuse_identity)

    assert native_landuse_identity({"LANDUSEF": np.zeros((21, 2, 2))}) \
        == dict(NATIVE_LANDUSE_IDENTITY)
    assert native_landuse_identity(None) == dict(NATIVE_LANDUSE_IDENTITY)
    lake_free = native_landuse_identity({"LANDUSEF": np.zeros((20, 2, 2))})
    assert lake_free == {**NATIVE_LANDUSE_IDENTITY, "ISLAKE": -1}



def test_history_attributes_of_a_lake_free_static_say_twenty_categories():
    """The wrfout global attributes follow the static, not the default table.

    Breakage (2.9.1 verify of the HRRR geography door): a 1 h forecast on
    modis_15s geography wrote NUM_LAND_CAT = 21 and ISLAKE = 21 into every
    history file although its static has 20 categories and no lake class
    (geogrid and wrf.exe write 20 and -1), because the prepared-forecast
    writer had no GEOG selection and fell back to the default table.
    """
    from datetime import datetime
    from types import SimpleNamespace

    import numpy as np

    from gpuwm.runtime import _global_wrf_attrs

    grid = SimpleNamespace(truelat1=30.0, truelat2=60.0, stand_lon=-97.0,
                           ref_lat=35.0, ref_lon=-97.0)
    start = datetime(2026, 10, 14, 12)
    twenty = {"LANDUSEF": np.zeros((20, 2, 2), np.float32)}
    attrs = _global_wrf_attrs(grid, start, static_fields=twenty)
    assert attrs["NUM_LAND_CAT"] == 20 and attrs["ISLAKE"] == -1
    # Every other static keeps the attributes it always had.
    default = _global_wrf_attrs(grid, start)
    twenty_one = {"LANDUSEF": np.zeros((21, 2, 2), np.float32)}
    assert _global_wrf_attrs(grid, start, static_fields=twenty_one) == default
    assert default["NUM_LAND_CAT"] == 21 and default["ISLAKE"] == 21
