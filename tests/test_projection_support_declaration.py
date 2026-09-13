"""One declaration of the implemented projection set, read by every door.

The lat-lon refusal itself is unchanged and is NOT relaxed here: what
changes is that the set it enumerates and the sentence it gives are
published once, by the module that would have to grow the projection
(:mod:`gpuwm.static.projection`), instead of being typed out again in
every door that answers for it.
"""

from __future__ import annotations

import pytest

from gpuwm.namelist_compat import analyze_namelists
from gpuwm.static import projection as projection_module
from gpuwm.static.projection import projection_class

_WPS = """\
&share
 wrf_core = 'ARW',
 max_dom = 1,
 start_date = '2020-05-01_00:00:00',
 end_date = '2020-05-01_12:00:00',
 interval_seconds = 3600,
/
&geogrid
 parent_id = 1,
 parent_grid_ratio = 1,
 i_parent_start = 1,
 j_parent_start = 1,
 e_we = 121,
 e_sn = 101,
 geog_data_res = 'default',
 dx = 12000,
 dy = 12000,
 map_proj = '{map_proj}',
 ref_lat = 35.0,
 ref_lon = -97.0,
 truelat1 = 30.0,
 truelat2 = 60.0,
 stand_lon = -97.0,
 geog_data_path = '/geog',
/
"""

_INPUT = """\
&time_control
 run_hours = 12,
 start_year = 2020,
 start_month = 5,
 start_day = 1,
 start_hour = 0,
 end_year = 2020,
 end_month = 5,
 end_day = 1,
 end_hour = 12,
 input_from_file = .true.,
/
&domains
 time_step = 60,
 max_dom = 1,
 e_we = 121,
 e_sn = 101,
 e_vert = 50,
 p_top_requested = 5000,
 dx = 12000.0,
 dy = 12000.0,
 grid_id = 1,
 parent_id = 0,
 i_parent_start = 1,
 j_parent_start = 1,
 parent_grid_ratio = 1,
 parent_time_step_ratio = 1,
 feedback = 0,
 smooth_option = 0,
/
&physics
 mp_physics = 8,
 ra_lw_physics = 4,
 ra_sw_physics = 4,
 sf_sfclay_physics = 91,
 sf_surface_physics = 2,
 bl_pbl_physics = 1,
 cu_physics = 0,
 num_soil_layers = 4,
 sf_urban_physics = 0,
/
&dynamics
 hybrid_opt = 2,
 etac = 0.2,
 use_theta_m = 0,
 diff_opt = 2,
 km_opt = 4,
 mix_full_fields = .true.,
/
&bdy_control
 spec_bdy_width = 5,
 specified = .true.,
 nested = .false.,
/
"""


def _pair(tmp_path, map_proj: str):
    wps = tmp_path / "namelist.wps"
    wps.write_text(_WPS.format(map_proj=map_proj), encoding="utf-8")
    inp = tmp_path / "namelist.input"
    inp.write_text(_INPUT, encoding="utf-8")
    return wps, inp


def test_a_projection_added_to_the_declaration_is_admitted_by_the_report(
        tmp_path, monkeypatch):
    """The support report reads the projection module's declaration.

    With a fourth entry in the code table, the door admits a namelist
    naming it -- proof that the set is read rather than re-typed.  The
    door used to hold its own literal tuple and refuse regardless.
    """
    patched = dict(projection_module.WRF_MAP_PROJ_CODES)
    patched["rotated-lat-lon"] = 6
    monkeypatch.setattr(projection_module, "WRF_MAP_PROJ_CODES", patched)

    report = analyze_namelists(*_pair(tmp_path, "rotated-lat-lon"))
    assert not [item for item in report["issues"]
                if item["code"] == "UNSUPPORTED_PROJECTION"]
    assert report["geometry"]["projection"]["map_proj"] == "rotated-lat-lon"


def test_both_surfaces_give_the_same_blocker_sentence(tmp_path):
    """The refusal stands, word for word, on both surfaces -- because both
    read the one published sentence."""
    blocker = projection_module.latlon_blocker()
    action = next(item for item in analyze_namelists(
        *_pair(tmp_path, "lat-lon"))["issues"]
        if item["code"] == "UNSUPPORTED_PROJECTION")["action"]
    assert blocker in action

    with pytest.raises(NotImplementedError) as excinfo:
        projection_class("lat-lon")
    assert blocker in str(excinfo.value)

    # The concrete reasons the refusal names are all still there.
    for named in ("angular dx/dy", "polar filter", "pole_lat/pole_lon",
                  "map_proj == 6"):
        assert named in blocker


def test_the_declaration_is_the_code_table_and_nothing_else():
    implemented = projection_module.implemented_projections()
    assert implemented == tuple(sorted(projection_module.WRF_MAP_PROJ_CODES))
    assert implemented == ("lambert", "mercator", "polar")
