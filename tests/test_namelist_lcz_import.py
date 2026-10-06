"""A WUDAPT Local Climate Zone namelist through the WRF namelist door.

2026-09-30 (the Central Europe proof, "What is left", item 1): a WUDAPT
user's namelist carries ``num_land_cat = 61`` and ``geog_data_res =
'cglc_modis_lcz+default'``.  The importer refused the first ("gpuwm builds
one land-use identity, the 21-category set"), untrue once the urban canopy
runs with ``use_wudapt_lcz = 1``, and dropped the second as "static-build
configuration, not imported", so the imported TOML said ``use_wudapt_lcz =
1`` and built MODIS land cover: every urban cell became UTYPE 5 and no
Local Climate Zone map was used.  A silent wrong answer.

These tests hold the door to: the user's own keys importing as an LCZ land
cover build (``[static.highres]``, ``landcover_source = "cglc-modis-lcz"``),
num_land_cat checked against the count that build stamps, every
geog_data_res token either built or refused by name, the static builder
admitting the token where the block builds it, and the contract table the
site reads carrying both rules.  All CPU.
"""
from __future__ import annotations

import tomllib
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from gpuwm.namelist_import import (NamelistRefusal, import_namelists,
                                   namelist_refusals, parse_namelist_text)

#: The user's pair from the Central Europe proof, as uploaded to its
#: import door: a 3 km root over the Alps and a 1 km nest over Munich and
#: the Inn valley.
USER_WPS = """\
&share
 wrf_core = 'ARW',
 max_dom = 2,
 start_date = '2026-09-22_12:00:00', '2026-09-22_12:00:00',
 end_date   = '2026-09-23_08:00:00', '2026-09-23_08:00:00',
 interval_seconds = 3600,
 io_form_geogrid = 2,
/
&geogrid
 parent_id         = 1, 1,
 parent_grid_ratio = 1, 3,
 i_parent_start    = 1, 38,
 j_parent_start    = 1, 31,
 e_we              = 151, 229,
 e_sn              = 121, 181,
 geog_data_res     = 'cglc_modis_lcz+default', 'cglc_modis_lcz+default',
 dx = 3000,
 dy = 3000,
 map_proj = 'lambert',
 ref_lat   = 47.75,
 ref_lon   = 12.0,
 truelat1  = 37.75,
 truelat2  = 57.75,
 stand_lon = 12.0,
 geog_data_path = '/geog',
 opt_geogrid_tbl_path = './geogrid/',
/
&ungrib
 out_format = 'WPS',
 prefix = 'ERA5',
/
&metgrid
 fg_name = 'ERA5',
/
"""

USER_INPUT = """\
&time_control
 run_hours = 20,
 start_year = 2026, 2026,
 start_month = 09, 09,
 start_day = 22, 22,
 start_hour = 12, 12,
 end_year = 2026, 2026,
 end_month = 09, 09,
 end_day = 23, 23,
 end_hour = 08, 08,
 interval_seconds = 3600,
 input_from_file = .true., .true.,
 history_interval = 60, 60,
 restart = .false.,
 restart_interval = 1440,
/
&domains
 time_step = 15,
 max_dom = 2,
 e_we = 151, 229,
 e_sn = 121, 181,
 e_vert = 50, 50,
 p_top_requested = 5000,
 dx = 3000.0, 1000.0,
 dy = 3000.0, 1000.0,
 grid_id = 1, 2,
 parent_id = 0, 1,
 i_parent_start = 1, 38,
 j_parent_start = 1, 31,
 parent_grid_ratio = 1, 3,
 parent_time_step_ratio = 1, 3,
 feedback = 0,
 smooth_option = 0,
/
&physics
 mp_physics = 8, 8,
 ra_lw_physics = 4, 4,
 ra_sw_physics = 4, 4,
 radt = 12, 12,
 sf_sfclay_physics = 1, 1,
 sf_surface_physics = 2, 2,
 bl_pbl_physics = 9, 9,
 bldt = 0, 0,
 cu_physics = 0, 0,
 num_soil_layers = 4,
 num_land_cat = 61,
 sf_urban_physics = 1, 1,
 use_wudapt_lcz = 1,
 sf_surface_mosaic = 1,
 mosaic_cat = 3,
/
&dynamics
 hybrid_opt = 2,
 etac = 0.2,
 w_damping = 1,
 epssm = 0.5,
 diff_opt = 2, 2,
 km_opt = 4, 4,
 mix_full_fields = .true., .true.,
 diff_6th_opt = 2, 2,
 diff_6th_factor = 0.12, 0.10,
 diff_6th_slopeopt = 1, 1,
 base_temp = 290.,
 damp_opt = 3,
 zdamp = 5000., 5000.,
 dampcoef = 0.2, 0.2,
 khdif = 0, 0,
 kvdif = 0, 0,
 non_hydrostatic = .true., .true.,
 moist_adv_opt = 1, 1,
/
&bdy_control
 spec_bdy_width = 5,
 spec_zone = 1,
 relax_zone = 4,
 specified = .true., .false.,
 nested = .false., .true.,
/
"""

#: Noah mosaic (sf_surface_mosaic, mosaic_cat) is the mosaic lane's door,
#: not this one's; these tests read the land-use keys with it set aside.
_MOSAIC = " sf_surface_mosaic = 1,\n mosaic_cat = 3,\n"
CACHE = "/srv/highres-cache"


def _input(text=USER_INPUT, *, drop=(), replace=()):
    text = text.replace(_MOSAIC, "")
    for line in drop:
        assert line in text, line
        text = text.replace(line, "")
    for old, new in replace:
        assert old in text, old
        text = text.replace(old, new)
    return text


def _wps(geog="'cglc_modis_lcz+default', 'cglc_modis_lcz+default'"):
    old = "'cglc_modis_lcz+default', 'cglc_modis_lcz+default'"
    return USER_WPS.replace(old, geog)


def _pair(tmp_path, wps, inp):
    folder = tmp_path / f"pair{len(list(tmp_path.glob('pair*')))}"
    folder.mkdir()
    (folder / "namelist.wps").write_text(wps, encoding="utf-8")
    (folder / "namelist.input").write_text(inp, encoding="utf-8")
    return folder / "namelist.wps", folder / "namelist.input"


def _import(tmp_path, wps=None, inp=None, **kwargs):
    kwargs.setdefault("static_cache_root", CACHE)
    return import_namelists(*_pair(tmp_path, _wps() if wps is None else wps,
                                   _input() if inp is None else inp),
                            **kwargs)


def _case_bytes(text: str, wps_path: Path) -> bytes:
    return (text + "\n[case_data]\n"
            'forcing = ["era5.grib"]\n'
            'vtable = "Vtable.ERA5"\n'
            f'wps_namelist = "{wps_path.as_posix()}"\n'
            f'geog_root = "{wps_path.parent.as_posix()}"\n'
            "sfcp_to_sfcp = true\n"
            'output_title = "lcz import"\n').encode("utf-8")


# ---------------------------------------------------------------------------
# The user's own keys
# ---------------------------------------------------------------------------

def test_the_wudapt_users_keys_import_as_an_lcz_land_cover_build(tmp_path):
    text, report = _import(tmp_path)
    toml = tomllib.loads(text)
    assert toml["static"] == {"highres": {
        "enabled": True, "cache_root": CACHE, "fields": "all",
        "landcover_source": "cglc-modis-lcz"}}
    assert toml["shared"]["sf_urban_physics"] == 1
    assert toml["shared"]["use_wudapt_lcz"] == 1
    fixed = {f.key: f for f in report.fixed}
    assert fixed["num_land_cat"].values == (61,)
    assert fixed["num_land_cat"].fixed_value == 61
    assert "categories 51-61" in fixed["num_land_cat"].reason
    # Not dropped any more: it produced the block, and the report says
    # what else that block replaces.
    assert "geog_data_res" not in {d.key for d in report.dropped}
    notice = next(n for n in report.notices if "geog_data_res" in n)
    for words in ('landcover_source = "cglc-modis-lcz"', "every domain",
                  "also replaces terrain", "Copernicus DEM GLO-30",
                  "soil texture (SoilGrids", CACHE):
        assert words in notice, words
    assert "also replaces terrain" in report.format()
    assert "# &geogrid geog_data_res 'cglc_modis_lcz+default'" in text


def test_the_users_whole_pair_is_not_refused_for_its_land_use_keys(tmp_path):
    """The exact upload, Noah mosaic keys and all: whatever else this
    engine says about it, num_land_cat and geog_data_res are not refused."""
    problems = namelist_refusals(
        parse_namelist_text(USER_WPS), parse_namelist_text(USER_INPUT),
        static_cache_root=CACHE)
    for problem in problems:
        keys = {key for _, key in getattr(problem, "keys", ())}
        assert not keys & {"num_land_cat", "geog_data_res"}, str(problem)
        assert "num_land_cat" not in str(problem)
        assert "geog_data_res" not in str(problem)


def test_the_imported_toml_builds_the_lcz_legend(tmp_path, monkeypatch):
    """Loaded as a case, the import selects CGLC-MODIS-LCZ with the urban
    legend, the static builder admits the user's own namelist.wps token,
    and the overlay stamps 61 categories with the LCZ classes kept."""
    from test_static_highres_landcover_sources import (
        MODIS21_ATTRS, N, _grid, _noah_baseline, _stub_categories,
        _stub_fetch)

    from gpuwm.case_data import load_experiment_case_bytes
    from gpuwm.static import highres
    from gpuwm.static.build import GeogSelection
    from gpuwm.static.highres_production import apply_highres_statics

    wps_path, inp_path = _pair(tmp_path, _wps(), _input())
    text, _ = import_namelists(wps_path, inp_path,
                               static_cache_root=tmp_path / "cache")
    _, data = load_experiment_case_bytes(
        _case_bytes(text, wps_path), source="imported.toml",
        base_dir=tmp_path, require_inputs=False)
    config = data.static_highres
    assert (config.enabled, config.fields, config.landcover_source) == (
        True, "all", "cglc-modis-lcz")
    assert (config.sf_urban_physics, config.use_wudapt_lcz) == (1, 1)
    assert config.cache_root == tmp_path / "cache"
    for grid_id in (1, 2):
        selection = GeogSelection.from_case_data(data, domain_id=grid_id)
        assert selection.resolution_tokens == ("cglc_modis_lcz", "default")
        assert selection.landuse == "modis_landuse_20class_30s_with_lakes"

    _stub_categories(monkeypatch)
    _stub_fetch(monkeypatch, {}, pixels={str(lcz): 12 for lcz in
                                         range(51, 62)})

    def categories(source, grid, mapping, *, category_count):
        fractions = np.zeros((category_count, grid.e_sn - 1, grid.e_we - 1))
        for column in range(grid.e_we - 1):
            fractions[mapping[51 + column % 11] - 1, :, column] = 1.0
        return fractions

    monkeypatch.setattr(highres, "resample_mapped_categories", categories)
    fields, receipt = apply_highres_statics(
        _noah_baseline(), _grid(47.75, 12.0, dx=1000.0, n=N + 1),
        config=config, domain_id=2, case_date=date(2026, 9, 22),
        landuse_attrs=MODIS21_ATTRS)
    assert fields["LANDUSEF"].shape == (61, N, N)
    built = set(np.unique(fields["LU_INDEX"][fields["LANDMASK"] > 0.5]))
    assert built == set(float(lcz) for lcz in range(51, 62))
    assert receipt["config"]["urban_legend"] == "urban"


# ---------------------------------------------------------------------------
# num_land_cat: the count the static build stamps
# ---------------------------------------------------------------------------

def test_61_is_refused_where_no_lcz_land_cover_is_built(tmp_path):
    with pytest.raises(NamelistRefusal) as caught:
        _import(tmp_path, wps=_wps("'default', 'default'"))
    message = str(caught.value)
    assert caught.value.keys == (("physics", "num_land_cat"),)
    assert "builds 21 categories on d01, d02" in message
    assert "the GEOG tree's MODIS land use" in message
    assert "Set num_land_cat = 21" in message
    assert "cglc_modis_lcz" in message
    # An urban run's wrfout says 61 whatever its geography
    # (gpuwm.io.wrfout), so the reason may not claim the wrfout says 21.
    assert "wrfout" not in message


def _refusals(wps, inp):
    return namelist_refusals(parse_namelist_text(wps),
                             parse_namelist_text(inp),
                             static_cache_root=CACHE)


def test_a_refused_lcz_token_does_not_also_refuse_its_61(tmp_path):
    """One import lists every fix, and the fixes must agree: with the LCZ
    token refused, num_land_cat = 61 is checked against the MODIS land use
    the set-aside key leaves, and "Set num_land_cat = 21" beside "Set
    use_wudapt_lcz = 1" is advice that refuses again once followed."""
    inp = _input(replace=((" use_wudapt_lcz = 1,\n",
                           " use_wudapt_lcz = 0,\n"),))
    problems = _refusals(_wps(), inp)
    assert len(problems) == 1, [str(p) for p in problems]
    assert problems[0].keys == (("geogrid", "geog_data_res"),)
    message = str(problems[0])
    assert "Set use_wudapt_lcz = 1" in message
    assert "num_land_cat = 21" in message
    # Either fix then imports.
    _import(tmp_path, inp=inp.replace(" use_wudapt_lcz = 0,\n",
                                      " use_wudapt_lcz = 1,\n"))
    _import(tmp_path, wps=_wps("'default', 'default'"),
            inp=inp.replace(" num_land_cat = 61,\n", " num_land_cat = 21,\n"))


def test_an_unbuilt_token_beside_the_lcz_token_is_the_one_refusal(tmp_path):
    wps = _wps("'cglc_modis_lcz+modis_30s', 'cglc_modis_lcz+default'")
    problems = _refusals(wps, _input())
    assert len(problems) == 1, [str(p) for p in problems]
    assert problems[0].keys == (("geogrid", "geog_data_res"),)
    assert "['modis_30s']" in str(problems[0])
    text, _ = _import(tmp_path)
    assert tomllib.loads(text)["static"]["highres"]["landcover_source"] == \
        "cglc-modis-lcz"


def test_a_count_no_token_makes_right_is_reported_beside_it():
    """The stand-in hides only the count a land-cover token makes right."""
    problems = _refusals(_wps("'modis_30s', 'default'"),
                         _input(replace=((" num_land_cat = 61,\n",
                                          " num_land_cat = 24,\n"),)))
    assert [p.keys for p in problems] == [
        (("geogrid", "geog_data_res"),), (("physics", "num_land_cat"),)]
    assert "geog_data_res" not in str(problems[1])


def test_61_is_refused_where_no_urban_canopy_keeps_the_lcz_classes(tmp_path):
    inp = _input(replace=((" sf_urban_physics = 1, 1,\n",
                           " sf_urban_physics = 0, 0,\n"),
                          (" use_wudapt_lcz = 1,\n",
                           " use_wudapt_lcz = 0,\n")))
    with pytest.raises(NamelistRefusal) as caught:
        _import(tmp_path, inp=inp)
    message = str(caught.value)
    assert "builds 21 categories on d01, d02" in message
    assert "collapsed to MODIS urban (13)" in message
    # 21 is what it builds, and 21 imports with the block still written.
    text, report = _import(tmp_path, inp=inp.replace(
        " num_land_cat = 61,\n", " num_land_cat = 21,\n"))
    assert tomllib.loads(text)["static"]["highres"]["landcover_source"] == \
        "cglc-modis-lcz"
    assert {f.key: f.fixed_value for f in report.fixed}["num_land_cat"] == 21


def test_21_is_refused_where_the_lcz_legend_builds_61(tmp_path):
    with pytest.raises(NamelistRefusal, match="builds 61 categories"):
        _import(tmp_path, inp=_input(replace=((" num_land_cat = 61,\n",
                                               " num_land_cat = 21,\n"),)))


def test_num_land_cat_left_out_takes_the_built_count(tmp_path):
    text, report = _import(tmp_path,
                           inp=_input(drop=(" num_land_cat = 61,\n",)))
    assert "num_land_cat" not in {f.key for f in report.fixed}
    assert tomllib.loads(text)["static"]["highres"]["landcover_source"] == \
        "cglc-modis-lcz"


def test_lcz_land_cover_without_the_lcz_table_is_wrfs_own_stop(tmp_path):
    inp = _input(drop=(" num_land_cat = 61,\n",),
                 replace=((" use_wudapt_lcz = 1,\n",
                           " use_wudapt_lcz = 0,\n"),))
    with pytest.raises(NamelistRefusal) as caught:
        _import(tmp_path, inp=inp)
    message = str(caught.value)
    assert caught.value.keys == (("geogrid", "geog_data_res"),)
    assert "USING 10 WUDAPT LCZ WITHOUT URBPARM_LCZ.TBL" in message
    assert "Set use_wudapt_lcz = 1" in message


# ---------------------------------------------------------------------------
# geog_data_res: built or refused by name, never dropped
# ---------------------------------------------------------------------------

def test_a_token_nothing_builds_is_refused_by_name(tmp_path):
    with pytest.raises(NamelistRefusal) as caught:
        _import(tmp_path, wps=_wps("'modis_30s+default', 'nlcd2011_9s'"),
                inp=_input(drop=(" num_land_cat = 61,\n",)))
    message = str(caught.value)
    assert caught.value.keys == (("geogrid", "geog_data_res"),)
    assert "['modis_30s', 'nlcd2011_9s']" in message
    assert "silent substitution" in message
    for token in ("default", "5m", "modis_lai", "cglc_modis_lcz"):
        assert token in message


@pytest.mark.parametrize("geog", ["'default', 'default'", "'5m'",
                                  "'modis_lai+default', '5m'"])
def test_geog_tree_tokens_write_no_block_and_change_nothing(tmp_path, geog):
    inp = _input(drop=(" num_land_cat = 61,\n",))
    text, report = _import(tmp_path, wps=_wps(geog), inp=inp)
    reference, _ = _import(tmp_path, wps=_wps("'default', 'default'"),
                           inp=inp)
    assert "[static" not in text
    assert text == reference
    dropped = {d.key: d.reason for d in report.dropped}
    assert dropped["geog_data_res"] == (
        "GEOG dataset/resolution selection is static-build configuration, "
        "not imported")


def test_one_token_on_the_nest_scopes_the_block_by_spacing(tmp_path):
    wps = _wps("'default', 'cglc_modis_lcz+default'")
    text, report = _import(tmp_path, wps=wps,
                           inp=_input(drop=(" num_land_cat = 61,\n",)))
    block = tomllib.loads(text)["static"]["highres"]
    assert block["max_dx_m"] == 1000.0
    notice = next(n for n in report.notices if "geog_data_res" in n)
    assert "d02 (max_dx_m = 1000" in notice
    # WRF reads one num_land_cat for the run, and this tree builds 21 on
    # d01 and 61 on d02, so no value the namelist gives describes both.
    with pytest.raises(NamelistRefusal) as caught:
        _import(tmp_path, wps=wps)
    message = str(caught.value)
    assert "21 categories on d01 and 61 categories on d02" in message
    assert "no single value" in message


def test_a_spacing_bound_that_cannot_be_drawn_is_refused(tmp_path):
    with pytest.raises(NamelistRefusal) as caught:
        _import(tmp_path, wps=_wps("'cglc_modis_lcz', 'default'"),
                inp=_input(drop=(" num_land_cat = 61,\n",)))
    message = str(caught.value)
    assert "named on d01 but not on d02" in message
    assert "max_dx_m" in message


def test_the_route_doors_split_the_static_companion(tmp_path, monkeypatch):
    """The native HRRR hierarchy and the native WRF export import a pair and
    build the experiment from the text.  Handed to the experiment-table
    builder whole, the LCZ import's [static] table was refused as "a defect
    in the calling code" on a pair those doors took before; it is split
    off as every file door does.  (Each route's own vertical-ladder check
    is not this test's subject and is set aside.)"""
    from gpuwm import hrrr_hierarchy_direct, native_wrf_contract
    from gpuwm.wrf_direct import export_prepared_wrf_namelists

    monkeypatch.setattr(hrrr_hierarchy_direct,
                        "validate_native_lambert_contracts",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(native_wrf_contract,
                        "validate_native_lambert_contracts",
                        lambda *args, **kwargs: None)
    wps_path, inp_path = _pair(tmp_path, _wps(), _input())
    exp, text, _ = hrrr_hierarchy_direct._native_experiment(wps_path,
                                                           inp_path)
    assert len(exp.domains) == 2
    assert "[static.highres]" in text
    missing = tmp_path / "no-domain-artifacts.json"
    with pytest.raises(FileNotFoundError) as caught:
        export_prepared_wrf_namelists(wps_path, inp_path, missing,
                                      tmp_path / "export")
    assert "no-domain-artifacts.json" in str(caught.value)


def test_an_input_file_route_still_drops_it(tmp_path):
    """On the wrfinput and met_em doors the statics come from the files."""
    text, report = import_namelists(
        *_pair(tmp_path, _wps("'modis_30s', 'cglc_modis_lcz'"),
               _input(drop=(" num_land_cat = 61,\n",))),
        landuse_identity={"MMINLU": "MODIFIED_IGBP_MODIS_NOAH",
                          "NUM_LAND_CAT": 61})
    assert "[static" not in text
    assert "geog_data_res" in {d.key for d in report.dropped}


def test_the_cache_root_defaults_to_the_engines_own(tmp_path, monkeypatch):
    from gpuwm.static import highres_production

    engine_cache = tmp_path / "per-user" / "highres-cache"
    monkeypatch.setattr(highres_production, "default_highres_cache_root",
                        lambda: engine_cache)
    text, _ = _import(tmp_path, static_cache_root=None)
    assert tomllib.loads(text)["static"]["highres"]["cache_root"] == \
        engine_cache.as_posix()


def test_the_cli_writes_the_block_and_says_what_it_replaces(tmp_path, capsys):
    import gpuwm.cli as cli

    wps_path, inp_path = _pair(tmp_path, _wps(), _input())
    out = tmp_path / "imported.toml"
    assert cli.main(["import-namelist", str(wps_path), str(inp_path),
                     "--output", str(out), "--static-cache-root",
                     str(tmp_path / "hc")]) == 0
    block = tomllib.loads(out.read_text(encoding="utf-8"))["static"]
    assert block["highres"]["cache_root"] == (tmp_path / "hc").resolve(
        ).as_posix()
    assert "also replaces terrain" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The static builder reads the same token from the user's namelist.wps
# ---------------------------------------------------------------------------

def test_the_static_builder_refuses_the_token_without_the_block():
    from gpuwm.static.build import GeogSelection

    with pytest.raises(ValueError) as caught:
        GeogSelection.from_tokens("/geog", "cglc_modis_lcz+default")
    message = str(caught.value)
    assert "[static.highres]" in message
    assert "silent substitution" in message
    selection = GeogSelection.from_tokens(
        "/geog", "cglc_modis_lcz+5m", highres_landcover="cglc-modis-lcz")
    assert selection.resolution_tokens == ("cglc_modis_lcz", "5m")
    assert selection.landuse == "modis_landuse_20class_5m_with_lakes"
    assert GeogSelection.from_tokens(
        "/geog", "cglc_modis_lcz", highres_landcover="cglc-modis-lcz"
    ).terrain == "topo_gmted2010_30s"
    with pytest.raises(ValueError, match="unrecognized token"):
        GeogSelection.from_tokens("/geog", "modis_30s")


def test_the_static_builder_reads_the_blocks_spacing_scope(tmp_path):
    from types import SimpleNamespace

    from gpuwm.static.build import GeogSelection
    from gpuwm.static.highres_production import HighresStaticConfig

    wps = tmp_path / "namelist.wps"
    wps.write_text(_wps(), encoding="utf-8")
    scoped = HighresStaticConfig(True, tmp_path, fields="all",
                                 max_dx_m=1000.0)
    data = SimpleNamespace(wps_namelist=wps, geog_root=tmp_path,
                           static_highres=scoped)
    assert GeogSelection.from_case_data(data, domain_id=2).landuse == \
        "modis_landuse_20class_30s_with_lakes"
    with pytest.raises(ValueError, match="builds none on this domain"):
        GeogSelection.from_case_data(data, domain_id=1)
    terrain_only = SimpleNamespace(
        wps_namelist=wps, geog_root=tmp_path,
        static_highres=HighresStaticConfig(True, tmp_path, fields="terrain"))
    with pytest.raises(ValueError, match="builds none on this domain"):
        GeogSelection.from_case_data(terrain_only, domain_id=2)


# ---------------------------------------------------------------------------
# The catalog gpuwm prep and gpuwm go resolve every domain through (A170)
# ---------------------------------------------------------------------------
#
# hrrr_native_static.verified_static_catalog binds each domain's GEOG
# selection for the ERA5, GFS and mapped-source doors, their terrain
# surveys and both hierarchy routes.  It handed the block to the selection
# only when the block carried terrain smoothing, so a pair imported with
# no GEOGRID.TBL smoothing was refused for the very token its block builds.

def _geog_root(tmp_path):
    """A GEOG root whose WPS default directories each hold an index."""
    from gpuwm.static import build

    root = tmp_path / "GEOG"
    selection = build.GeogSelection.fallback(root)
    for field in build._DEFAULT_GEOG_DIRS:
        directory = selection.path(field)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "index").write_text(f"# {field}\n", encoding="utf-8")
    return root


def _lcz_block(**changes):
    from gpuwm.static.highres_production import HighresStaticConfig

    settings = {"fields": "all", "landcover_source": "cglc-modis-lcz",
                **changes}
    return HighresStaticConfig(True, Path(CACHE), **settings)


def _two_domains():
    from types import SimpleNamespace

    return SimpleNamespace(domains=(SimpleNamespace(grid_id=1),
                                    SimpleNamespace(grid_id=2)))


def test_the_catalog_carries_a_land_cover_block_with_no_smoothing(tmp_path):
    from gpuwm.hrrr_native_static import verified_static_catalog
    from gpuwm.static.build import geog_selection_from_catalog

    root = _geog_root(tmp_path)
    wps = tmp_path / "namelist.wps"
    wps.write_text(_wps(), encoding="utf-8")
    block = _lcz_block()
    assert not block.terrain_smoothing
    with pytest.raises(ValueError, match="builds none on this domain"):
        verified_static_catalog(wps, root, (1, 2))
    catalog, receipt = verified_static_catalog(wps, root, (1, 2),
                                               static_highres=block)
    assert catalog.static_highres is block
    assert "terrain_smoothing" not in receipt
    for grid_id in (1, 2):
        assert geog_selection_from_catalog(
            catalog, grid_id).resolution_tokens == ("cglc_modis_lcz",
                                                    "default")
    # The token names no GEOG directory: the GEOG tree beneath the block
    # is the one the namelist's default token selects.
    default = tmp_path / "default.wps"
    default.write_text(_wps("'default', 'default'"), encoding="utf-8")
    _, plain = verified_static_catalog(default, root, (1, 2))
    assert receipt["selections"] == plain["selections"]


@pytest.mark.parametrize("carrier", ["none", "disabled", "terrain",
                                     "engine-default"])
def test_a_carrier_the_selection_does_not_read_stays_off_the_catalog(
        tmp_path, carrier):
    from gpuwm.hrrr_native_static import verified_static_catalog
    from gpuwm.static.highres_production import (HighresStaticConfig,
                                                 default_static_highres)
    from gpuwm.static.terrain_smoothing import selection_carrier_kwargs

    config = {"none": None,
              "disabled": HighresStaticConfig(False, Path(CACHE)),
              "terrain": _lcz_block(fields="terrain"),
              "engine-default": default_static_highres([1000.0])}[carrier]
    root = _geog_root(tmp_path)
    wps = tmp_path / "namelist.wps"
    wps.write_text(_wps("'default', 'default'"), encoding="utf-8")
    plain_catalog, plain_receipt = verified_static_catalog(wps, root, (1, 2))
    catalog, receipt = verified_static_catalog(wps, root, (1, 2),
                                               static_highres=config)
    assert receipt == plain_receipt
    assert vars(catalog) == vars(plain_catalog)
    assert not hasattr(catalog, "static_highres")
    assert selection_carrier_kwargs(config) == {}


def test_a_trees_children_see_the_land_cover_block(tmp_path):
    from types import SimpleNamespace

    from gpuwm.ingest.nest_init import _static_catalog
    from gpuwm.static.build import geog_selection_from_catalog
    from gpuwm.static.terrain_smoothing import catalog_with_smoothing

    root = _geog_root(tmp_path)
    wps = tmp_path / "namelist.wps"
    wps.write_text(_wps(), encoding="utf-8")
    inner = SimpleNamespace(files=(
        SimpleNamespace(role="wps_namelist", path=wps),
        SimpleNamespace(role="geog_index",
                        path=root / "topo_gmted2010_30s" / "index")))
    with pytest.raises(ValueError, match="builds none on this domain"):
        geog_selection_from_catalog(
            _static_catalog(SimpleNamespace(static_catalog=inner)), 2)
    block = _lcz_block()
    view = _static_catalog(SimpleNamespace(static_catalog=inner,
                                           static_highres=block))
    assert view is not inner and view.static_highres is block
    assert geog_selection_from_catalog(view, 2).resolution_tokens == (
        "cglc_modis_lcz", "default")
    assert catalog_with_smoothing(view, block) is view
    assert catalog_with_smoothing(inner, _lcz_block(fields="terrain")) is inner


@pytest.mark.parametrize("door", ["era5_direct", "gfs_direct",
                                  "mapped_direct"])
def test_each_doors_terrain_survey_carries_the_block(tmp_path, door):
    import importlib

    module = importlib.import_module(f"gpuwm.{door}")
    root = _geog_root(tmp_path)
    wps = tmp_path / "namelist.wps"
    wps.write_text(_wps(), encoding="utf-8")
    block = _lcz_block()
    catalog = module._survey_static_catalog(_two_domains(), wps, root, block)
    assert catalog.static_highres is block
    with pytest.raises(ValueError, match="builds none on this domain"):
        module._survey_static_catalog(_two_domains(), wps, root)


def test_the_root_static_build_carries_the_block(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from gpuwm import era5_direct

    built = []
    monkeypatch.setattr(era5_direct, "build_static_for_domain",
                        lambda grid, catalog, grid_id: built.append(catalog)
                        or {})
    monkeypatch.setattr(era5_direct, "_validated_static",
                        lambda fields, *args, **kwargs: fields)
    monkeypatch.setattr(
        era5_direct, "geog_selection_from_catalog",
        lambda catalog, grid_id: SimpleNamespace(
            landuse_global_attrs=lambda: {}))
    root = _geog_root(tmp_path)
    wps = tmp_path / "namelist.wps"
    wps.write_text(_wps(), encoding="utf-8")
    block = _lcz_block()
    cfg = SimpleNamespace(ny=1, nx=1)
    _, receipt, _ = era5_direct._static_from_geog(
        wps, root, None, cfg, static_highres=block)
    (catalog,) = built
    assert catalog.static_highres is block
    assert "terrain_smoothing" not in receipt


def test_the_source_hierarchy_catalog_carries_the_block(tmp_path,
                                                        monkeypatch):
    from types import SimpleNamespace

    from gpuwm import source_hierarchy

    monkeypatch.setattr(source_hierarchy, "NestedInputCatalog",
                        lambda **kwargs: SimpleNamespace(**kwargs))
    root = _geog_root(tmp_path)
    wps = tmp_path / "namelist.wps"
    wps.write_text(_wps(), encoding="utf-8")
    block = _lcz_block()
    static_catalog, receipt, catalog = source_hierarchy._hierarchy_catalog(
        _two_domains(), (), wps_namelist=wps, geog_root=root,
        source_name="GFS", source_manifest_sha256="0" * 64,
        orography_receipt=None, source_coverage_receipt=None,
        preprocess_selection=None, inventory=None, units=None,
        soil_texture_downscale=False, water_temperature_policy=None,
        static_highres=block)
    assert static_catalog.static_highres is block
    assert catalog.static_highres is block
    assert "terrain_smoothing" not in receipt


# ---------------------------------------------------------------------------
# The contract table the site reads
# ---------------------------------------------------------------------------

def test_the_contract_carries_both_rules():
    from gpuwm.namelist_contract import build_namelist_contract

    contract = build_namelist_contract()
    sections = contract["sections"]
    num_land_cat = sections["physics"]["keys"]["num_land_cat"]
    assert num_land_cat["values"] == [21, 61]
    assert "use_wudapt_lcz = 1" in num_land_cat["why"]
    geog = sections["geogrid"]["keys"]["geog_data_res"]
    assert geog["values"] is None
    # 3ada3ce436, lane/2km-landusef, admits BNU top/bottom soil tiles.
    # The contract sorts admitted tokens; this is not dataset precedence.
    # Retain the LCZ token and its independent urban/61-category rule.
    assert geog["tokens"] == ["30s", "5m", "bnu_soil_30s",
                              "cglc_modis_lcz", "default", "modis_lai"]
    assert geog["token_separator"] == "+"
    assert geog["reach"] == "max_dom"
    assert "also replaces terrain and soil" in geog["why"]
    assert all(row["used"] for row in contract["baselines"])
