"""Public RW-WPS namelist support-report gates."""

from __future__ import annotations

import json

import pytest

from gpuwm.namelist_compat import analyze_namelists, require_supported_namelists
from gpuwm.source_cli import EXIT_CONFIG, main as source_cli_main


def _write_pair(
    tmp_path, *, max_dom=6, mass_levels=49, mp=8, ra_lw=4, ra_sw=4,
    extra_physics="",
):
    def values(root, child=None):
        child = root if child is None else child
        return ", ".join(str(value) for value in [root] + [child] * (max_dom - 1))

    eta = ", ".join(
        f"{1.0 - index / mass_levels:.17g}"
        for index in range(mass_levels + 1)
    )
    wps = f"""&share
 wrf_core = 'ARW',
 max_dom = {max_dom},
 start_date = {values("'2020-05-01_00:00:00'")},
 end_date = {values("'2020-05-01_12:00:00'")},
 interval_seconds = 3600,
/
&geogrid
 parent_id = {values(1)},
 parent_grid_ratio = {values(1, 3)},
 i_parent_start = {values(1, 20)},
 j_parent_start = {values(1, 20)},
 e_we = {values(121, 61)},
 e_sn = {values(101, 61)},
 geog_data_res = {values("'default'")},
 dx = 12000,
 dy = 12000,
 map_proj = 'lambert',
 ref_lat = 35.0,
 ref_lon = -97.0,
 truelat1 = 30.0,
 truelat2 = 60.0,
 stand_lon = -97.0,
 geog_data_path = '/geog',
/
&ungrib
 out_format = 'WPS',
 prefix = 'SOURCE',
/
&metgrid
 fg_name = 'SOURCE',
/
"""
    inp = f"""&time_control
 run_hours = 12,
 start_year = {values(2020)},
 start_month = {values(5)},
 start_day = {values(1)},
 start_hour = {values(0)},
 end_year = {values(2020)},
 end_month = {values(5)},
 end_day = {values(1)},
 end_hour = {values(12)},
 input_from_file = {values('.true.')},
 history_interval = {values(60, 15)},
/
&domains
 time_step = 60,
 max_dom = {max_dom},
 e_we = {values(121, 61)},
 e_sn = {values(101, 61)},
 e_vert = {values(mass_levels + 1)},
 eta_levels = {eta},
 p_top_requested = {values(5000)},
 dx = {values(12000.0, 4000.0)},
 dy = {values(12000.0, 4000.0)},
 grid_id = {', '.join(str(i) for i in range(1, max_dom + 1))},
 parent_id = {values(0, 1)},
 i_parent_start = {values(1, 20)},
 j_parent_start = {values(1, 20)},
 parent_grid_ratio = {values(1, 3)},
 parent_time_step_ratio = {values(1, 3)},
 feedback = 0,
 smooth_option = 0,
/
&physics
 mp_physics = {values(mp)},
 ra_lw_physics = {values(ra_lw)},
 ra_sw_physics = {values(ra_sw)},
 sf_sfclay_physics = {values(91)},
 sf_surface_physics = {values(2)},
 bl_pbl_physics = {values(1)},
 cu_physics = {values(0)},
 num_soil_layers = {values(4)},
 sf_urban_physics = {values(0)},
 radt = {values(10)},
 {extra_physics}
/
&dynamics
 hybrid_opt = 2,
 etac = 0.2,
 use_theta_m = 0,
 diff_opt = {values(2)},
 km_opt = {values(4)},
 mix_full_fields = {values('.true.')},
/
&bdy_control
 spec_bdy_width = 5,
 specified = {values('.true.', '.false.')},
 nested = {values('.false.', '.true.')},
/
"""
    wps_path = tmp_path / "namelist.wps"
    input_path = tmp_path / "namelist.input"
    wps_path.write_text(wps, encoding="utf-8")
    input_path.write_text(inp, encoding="utf-8")
    return wps_path, input_path


def test_six_domain_thompson_stock_export_passes_gpuwm_runtime_separate(tmp_path):
    report = analyze_namelists(*_write_pair(tmp_path, mp=8))
    assert report["verdict"] == "PASS"
    assert report["max_dom"] == 6
    assert report["geometry"]["domain_count"] == 6
    assert [row["grid_id"] for row in report["geometry"]["domains"]] == list(range(1, 7))
    assert report["geometry"]["projection"]["map_proj"] == "lambert"
    assert report["required_state"]["stock_wrf_export"]["verdict"] == "PASS"
    # Pinned FAIL while mp8 was env-gated; PASS since the promotion to a
    # first-class scheme with packaged tables (product/v1 packaging lane
    # 2026-07-28).  The two verdicts remain independently computed, which
    # is what this test is for.
    assert report["required_state"]["gpuwm_runtime"] == {
        "verdict": "PASS",
        "reasons": [],
    }
    fields = report["required_state"]["stock_wrf_export"]["domains"][5]["wrfinput_fields"]
    assert [field["netcdf_name"] for field in fields][-2:] == ["QNICE", "QNRAIN"]
    # The report itself is strict JSON, suitable for automation and receipts.
    json.dumps(report, allow_nan=False)


def _stagger_pair(tmp_path, *, child_minute: int, interval: int):
    wps, inp = _write_pair(tmp_path, max_dom=2, mp=6)
    wps_text = wps.read_text(encoding="utf-8").replace(
        "start_date = '2020-05-01_00:00:00', "
        "'2020-05-01_00:00:00',",
        "start_date = '2020-05-01_00:00:00', "
        f"'2020-05-01_00:{child_minute:02d}:00',").replace(
            "interval_seconds = 3600",
            f"interval_seconds = {interval}")
    input_text = inp.read_text(encoding="utf-8").replace(
        " start_hour = 0, 0,",
        " start_hour = 0, 0,\n"
        f" start_minute = 0, {child_minute},\n"
        " start_second = 0, 0,\n"
        f" interval_seconds = {interval},")
    wps.write_text(wps_text, encoding="utf-8")
    inp.write_text(input_text, encoding="utf-8")
    return wps, inp


def test_support_report_claims_staggered_five_minute_timing_only_when_real(
        tmp_path):
    report = analyze_namelists(
        *_stagger_pair(tmp_path, child_minute=5, interval=300))

    assert report["verdict"] == "PASS"
    assert report["required_state"]["gpuwm_runtime"]["verdict"] == "PASS"
    assert report["timing"]["boundary_interval_seconds"] == 300
    assert report["timing"]["root_cadence_steps_exact"] == "5"
    assert report["timing"]["domains"][1] == {
        "grid_id": 2,
        "start_time": "2020-05-01T00:05:00",
        "end_time": "2020-05-01T12:00:00",
        "offset_seconds": 300,
        "dt_seconds_exact": "20",
        "parent_step_alignment": "PASS",
        "forcing_seam_alignment": "PASS",
    }


def test_support_report_keeps_precise_timing_refusals(tmp_path):
    staggered = analyze_namelists(
        *_stagger_pair(tmp_path, child_minute=4, interval=300))
    assert staggered["verdict"] == "PASS"
    runtime = staggered["required_state"]["gpuwm_runtime"]
    assert runtime["verdict"] == "FAIL"
    assert any(
        "d02" in reason and "offset/cadence = 4/5" in reason
        for reason in runtime["reasons"])

    cadence = analyze_namelists(
        *_stagger_pair(tmp_path, child_minute=0, interval=310))
    assert cadence["verdict"] == "PASS"
    assert cadence["required_state"]["gpuwm_runtime"]["verdict"] == "FAIL"
    assert any(
        "whole number of root-domain steps" in reason
        and "cadence/dt = 31/6" in reason
        for reason in cadence["required_state"]["gpuwm_runtime"]["reasons"])


def test_support_report_rejects_unimplemented_projection(tmp_path):
    wps, inp = _write_pair(tmp_path, max_dom=2, mp=6)
    wps.write_text(
        wps.read_text(encoding="utf-8").replace(
            "map_proj = 'lambert'", "map_proj = 'lat-lon'"
        ),
        encoding="utf-8",
    )
    report = analyze_namelists(wps, inp)
    assert report["verdict"] == "FAIL"
    issue = next(
        value for value in report["issues"]
        if value["code"] == "UNSUPPORTED_PROJECTION"
    )
    assert "Lambert conformal, Mercator, and polar stereographic" \
        in issue["message"]
    assert "rejected rather than approximated" in issue["action"]
    assert "angular dx/dy" in issue["action"]
    assert "polar filter" in issue["action"]
    assert "map_proj == 6" in issue["action"]


def test_support_report_accepts_mercator_geometry(tmp_path):
    """Worldwide lane: mercator geometry is reported, not refused, and
    the WPS-optional truelat2/stand_lon default per module_llxy
    semantics (truelat1 / ref_lon)."""
    wps, inp = _write_pair(tmp_path, max_dom=2, mp=6)
    wps.write_text(
        wps.read_text(encoding="utf-8").replace(
            "map_proj = 'lambert'", "map_proj = 'mercator'"
        ),
        encoding="utf-8",
    )
    report = analyze_namelists(wps, inp)
    assert not [issue for issue in report["issues"]
                if issue["code"] in ("UNSUPPORTED_PROJECTION",
                                     "INVALID_PROJECTION")]
    assert report["geometry"]["projection"]["map_proj"] == "mercator"


@pytest.mark.parametrize(
    ("wps_before", "wps_after", "input_before", "input_after", "message"),
    [
        (
            "e_we = 121, 61,", "e_we = 121, 62,",
            "e_we = 121, 61,", "e_we = 121, 62,",
            "minus one must be divisible",
        ),
        (
            "i_parent_start = 1, 20,", "i_parent_start = 1, 115,",
            "i_parent_start = 1, 20,", "i_parent_start = 1, 115,",
            "do not fit",
        ),
        (
            None, None,
            "dx = 12000.0, 4000.0,", "dx = 12000.0, 5000.0,",
            "expected 4000.0",
        ),
    ],
)
def test_support_report_rejects_illegal_nest_geometry(
    tmp_path, wps_before, wps_after, input_before, input_after, message,
):
    wps, inp = _write_pair(tmp_path, max_dom=2, mp=6)
    if wps_before is not None:
        wps.write_text(
            wps.read_text(encoding="utf-8").replace(wps_before, wps_after),
            encoding="utf-8",
        )
    inp.write_text(
        inp.read_text(encoding="utf-8").replace(input_before, input_after),
        encoding="utf-8",
    )
    report = analyze_namelists(wps, inp)
    assert report["verdict"] == "FAIL"
    issue = next(
        value for value in report["issues"]
        if value["code"] == "INVALID_DOMAIN_HIERARCHY"
    )
    assert message in issue["message"]


def test_support_report_rejects_inconsistent_boundary_width(tmp_path):
    wps, inp = _write_pair(tmp_path, max_dom=2, mp=6)
    inp.write_text(
        inp.read_text(encoding="utf-8").replace(
            " spec_bdy_width = 5,",
            " spec_bdy_width = 5,\n spec_zone = 2,\n relax_zone = 4,",
        ),
        encoding="utf-8",
    )
    report = analyze_namelists(wps, inp)
    assert report["verdict"] == "FAIL"
    issue = next(
        value for value in report["issues"]
        if value["code"] == "INVALID_BOUNDARY_TOPOLOGY"
    )
    assert "must equal" in issue["message"]


def test_thompson_runtime_is_reported_runnable_without_env(
        monkeypatch, tmp_path):
    """mp8 is runtime-supported with NO Thompson environment set: the
    enable guard is retired and the table root defaults to the packaged
    assets (mp8 promotion, product/v1 packaging lane 2026-07-28)."""
    monkeypatch.delenv("GPUWM_EXPERIMENTAL_THOMPSON_MP8", raising=False)
    monkeypatch.delenv("GPUWM_THOMPSON_TABLE_ROOT", raising=False)
    report = analyze_namelists(*_write_pair(tmp_path, mp=8))
    assert report["verdict"] == "PASS"
    assert report["required_state"]["stock_wrf_export"]["verdict"] == "PASS"
    assert report["required_state"]["gpuwm_runtime"] == {
        "verdict": "PASS",
        "reasons": [],
    }


def test_omitted_use_theta_m_keeps_stock_export_but_fails_gpuwm_runtime(
        tmp_path):
    wps, inp = _write_pair(tmp_path, mp=6)
    inp.write_text(
        inp.read_text(encoding="utf-8").replace(" use_theta_m = 0,\n", ""),
        encoding="utf-8",
    )
    report = analyze_namelists(wps, inp)
    assert report["verdict"] == "PASS"
    assert report["required_state"]["stock_wrf_export"]["verdict"] == "PASS"
    runtime = report["required_state"]["gpuwm_runtime"]
    assert runtime["verdict"] == "FAIL"
    assert "WRF Registry default 1" in runtime["reasons"][0]
    assert "use_theta_m = 0" in runtime["reasons"][0]


@pytest.mark.parametrize(
    ("line", "default_text", "action_text"),
    [
        (
            " mix_full_fields = .true., .true., .true., .true., .true., .true.,\n",
            "WRF Registry default false",
            "mix_full_fields = .true.",
        ),
        (
            " smooth_option = 0,\n",
            "WRF Registry default 2",
            "smooth_option = 0",
        ),
    ],
)
def test_gpuwm_runtime_reports_trajectory_changing_omitted_defaults(
        tmp_path, line, default_text, action_text):
    wps, inp = _write_pair(tmp_path, mp=6)
    text = inp.read_text(encoding="utf-8")
    assert line in text
    inp.write_text(text.replace(line, ""), encoding="utf-8")
    report = analyze_namelists(wps, inp)
    assert report["verdict"] == "PASS"
    assert report["required_state"]["stock_wrf_export"]["verdict"] == "PASS"
    runtime = report["required_state"]["gpuwm_runtime"]
    assert runtime["verdict"] == "FAIL"
    assert any(
        default_text in reason and action_text in reason
        for reason in runtime["reasons"]
    )


def test_morrison_stock_inventory_keeps_all_number_moments(tmp_path):
    report = require_supported_namelists(*_write_pair(tmp_path, mp=10))
    assert report["required_state"]["gpuwm_runtime"]["verdict"] == "PASS"
    names = [
        field["netcdf_name"]
        for field in report["required_state"]["stock_wrf_export"]["domains"][0]["wrfinput_fields"]
    ]
    assert names[-4:] == ["QNICE", "QNSNOW", "QNRAIN", "QNGRAUPEL"]


def test_nssl2_runtime_is_reported_runnable_with_full_moment_inventory(
        tmp_path):
    """mp18 is runtime-supported since the certified NSSL merge
    (product/v1 NSSL lane 2026-07-29).  The runtime verdict and the stock
    export inventory stay independently computed, exactly as the mp8
    promotion pinned; NSSL's registry maturity remains
    "wrf-matched-run-candidate" and is not this report's claim."""
    report = require_supported_namelists(*_write_pair(tmp_path, mp=18))
    assert report["verdict"] == "PASS"
    assert report["required_state"]["stock_wrf_export"]["verdict"] == "PASS"
    assert report["required_state"]["gpuwm_runtime"] == {
        "verdict": "PASS",
        "reasons": [],
    }
    names = [
        field["netcdf_name"]
        for field in report["required_state"]["stock_wrf_export"]["domains"][0]["wrfinput_fields"]
    ]
    # WRF's resolved option-18 default packages: hail mass plus every
    # second moment and both predicted volumes (Registry.EM_COMMON:3033,
    # 3049-3056 after module_check_a_mundo resolves the -1 selectors).
    assert names[-10:] == [
        "QHAIL", "QNDROP", "QNRAIN", "QNICE", "QNSNOW",
        "QNGRAUPEL", "QNHAIL", "QNCCN", "QVGRAUPEL", "QVHAIL",
    ]


def test_p3_runtime_is_reported_runnable_and_not_denied_as_unimplemented(
        tmp_path):
    """mp50 is runtime-supported since the P3 port landed.

    The report used to print, once per domain, "gpuwm runtime on paired
    head does not implement mp_physics=50" for a selector
    gpuwm.config.MP_PHYSICS_ACCEPTED admits, gpuwm/core/microphysics.py
    dispatches to gpuwm/core/p3.py, and this same report writes a full
    stock-export row for.  A user reading the support report for an mp=50
    experiment was told to abandon a scheme that runs.  Radiation is the
    Dudhia pair here, because the 4/4 RTE+RRTMGP pairing is separately
    refused (the test below).
    """
    report = require_supported_namelists(
        *_write_pair(tmp_path, mp=50, ra_lw=0, ra_sw=1))
    assert report["verdict"] == "PASS"
    assert report["required_state"]["stock_wrf_export"]["verdict"] == "PASS"
    assert report["required_state"]["gpuwm_runtime"] == {
        "verdict": "PASS",
        "reasons": [],
    }
    row = report["required_state"]["stock_wrf_export"]["domains"][0]
    assert row["microphysics"] == "P3 one-category two-moment ice"
    names = [field["netcdf_name"] for field in row["wrfinput_fields"]]
    # P3's Registry package p3_1category (Registry.EM_COMMON:3038): one ice
    # category with the rime pair, and NO QSNOW/QGRAUP to carry.
    assert names[-4:] == ["QNICE", "QNRAIN", "QIR", "QIB"]
    assert "QSNOW" not in names and "QGRAUP" not in names


def test_p3_rte_rrtmgp_pairing_is_refused_by_name_not_as_unimplemented(
        tmp_path):
    """Admitting the SCHEME does not admit the 4/4 RTE+RRTMGP PAIRING.

    P3 supplies cloud and ice radii but no snow radius -- WRF sets
    has_reqs=0 for the P3 family (phys/module_physics_init.F:1027-1033) --
    so gpuwm.config.validate_p3_radiation refuses ra_lw_physics=4 /
    ra_sw_physics=4 at the head's config door, by default
    (ra_rrtmg_variant defaults to "rte-rrtmgp").  The concrete breakage
    this reason prevents is a green support report followed by a full WPS
    export and then a NotImplementedError at config load.  The reason must
    say WHY, and must not say the scheme is unimplemented.
    """
    report = analyze_namelists(*_write_pair(tmp_path, mp=50, ra_lw=4, ra_sw=4))
    assert report["required_state"]["stock_wrf_export"]["verdict"] == "PASS"
    runtime = report["required_state"]["gpuwm_runtime"]
    assert runtime["verdict"] == "FAIL"
    assert len(runtime["reasons"]) == 6
    for reason in runtime["reasons"]:
        assert "does not implement mp_physics=50" not in reason
        assert "implements mp_physics=50" in reason
        assert "ra_lw_physics=4/ra_sw_physics=4" in reason
        assert "has_reqs=0" in reason
        assert "module_physics_init.F:1027-1033" in reason
        assert "validate_p3_radiation" in reason


@pytest.mark.parametrize("mass_levels", [35, 49, 80])
def test_namelist_report_accepts_arbitrary_structural_vertical_counts(tmp_path, mass_levels):
    report = require_supported_namelists(
        *_write_pair(tmp_path, mass_levels=mass_levels, mp=6),
        source_top_pressure_pa=5000.0,
    )
    assert report["vertical"]["mass_levels"] == mass_levels
    assert report["vertical"]["e_vert"] == mass_levels + 1
    assert len(report["vertical"]["eta_levels"]) == mass_levels + 1
    assert report["vertical"]["coverage"] == "verified"


def test_unclassified_physics_fails_closed_and_names_action(tmp_path):
    report = analyze_namelists(
        *_write_pair(tmp_path, extra_physics="mystery_cloud_state = 1,"),
    )
    assert report["verdict"] == "FAIL"
    issue = next(
        item for item in report["issues"]
        if item["code"] == "UNCLASSIFIED_NAMELIST_SETTING"
    )
    assert issue["location"] == "&physics/mystery_cloud_state"
    assert "will not be ignored or substituted" in issue["action"]


def test_gfl_era_spreading_key_fails_closed_on_the_support_surface(tmp_path):
    """The v4.8.0 Grell-Freitas-Li generation hazard, on the RW-WPS side.

    A v4.8.0 namelist selecting GFL (cu_physics = 3) is
    byte-indistinguishable from a v4.6.1 GF one at the option level
    (GF-UPSTREAM-AND-NOCTURNAL-BIAS.md, ranked action 0); the one
    spelling that reveals spreading-generation intent is cugd_avedx.
    This report's stock target is pinned as unchanged WRF v4.6.1, and
    cugd_avedx must stay UNCLASSIFIED -- fail-closed -- rather than ever
    being silently blessed into a classification, because blessing it
    would green-light a namelist written for a scheme generation the
    v4.6.1 target does not contain.
    """
    report = analyze_namelists(
        *_write_pair(tmp_path, extra_physics="cu_physics = 3,\n"
                                             " cugd_avedx = 3,"),
    )
    assert report["verdict"] == "FAIL"
    issue = next(
        item for item in report["issues"]
        if item["code"] == "UNCLASSIFIED_NAMELIST_SETTING"
        and item["location"] == "&physics/cugd_avedx"
    )
    assert "will not be ignored or substituted" in issue["action"]
    assert (report["required_state"]["stock_wrf_export"]["target"]
            == "unchanged WRF v4.6.1")


def test_wrf_runner_runtime_io_keys_classify_without_unclassified(tmp_path):
    """The CPU-WRF runtime I/O keys a WRF-Runner-generated namelist always
    carries (2026-07-30 interop verification) classify as runtime-only
    instead of spraying UNCLASSIFIED_NAMELIST_SETTING noise."""
    wps, inp = _write_pair(tmp_path)
    text = inp.read_text(encoding="utf-8").replace(
        " run_hours = 12,",
        " run_hours = 12,\n"
        " io_form_auxinput2 = 2,\n"
        " override_restart_timers = .true.,\n"
        " iofields_filename = 'iofields.txt',\n"
        " ignore_iofields_warning = .true.,\n"
        " fine_input_stream = 0, 0, 0, 0, 0, 0,",
    )
    inp.write_text(text, encoding="utf-8")
    report = analyze_namelists(wps, inp, source_top_pressure_pa=5000.0)
    assert not [item for item in report["issues"]
                if item["code"] == "UNCLASSIFIED_NAMELIST_SETTING"]
    runtime_keys = {
        entry["key"] for entry in report["classifications"]
        ["runtime_output_only"] if entry["section"] == "time_control"
    }
    assert {"io_form_auxinput2", "override_restart_timers",
            "iofields_filename", "ignore_iofields_warning"} <= runtime_keys
    # all-zero fine_input_stream is the prepared path: classified, no issue
    assert not [item for item in report["issues"]
                if item["code"] == "NEST_INPUT_STREAM_UNSUPPORTED"]
    assert report["verdict"] == "PASS"


def test_delayed_nest_input_stream_fails_precisely(tmp_path):
    """fine_input_stream = 0, 2 (WRF's delayed-nest-start pattern, the
    newest WRF-Runner feature) names the missing per-nest input file
    rather than an unclassified-setting shrug."""
    wps, inp = _write_pair(tmp_path)
    text = inp.read_text(encoding="utf-8").replace(
        " run_hours = 12,",
        " run_hours = 12,\n fine_input_stream = 0, 2, 2, 2, 2, 2,",
    )
    inp.write_text(text, encoding="utf-8")
    report = analyze_namelists(wps, inp, source_top_pressure_pa=5000.0)
    assert report["verdict"] == "FAIL"
    issue = next(item for item in report["issues"]
                 if item["code"] == "NEST_INPUT_STREAM_UNSUPPORTED")
    assert "met_em-class input" in issue["message"]
    assert not [item for item in report["issues"]
                if item["code"] == "UNCLASSIFIED_NAMELIST_SETTING"]


def test_active_grid_fdda_fails_as_missing_wrffdda(tmp_path):
    """grid_fdda=1 was silently classified runtime-only while real.exe is
    the only producer of the wrffdda_d0N file it reads; the report must
    refuse what the export cannot feed (found live against a
    WRF-Runner-generated nudging namelist, 2026-07-30)."""
    wps, inp = _write_pair(tmp_path)
    text = inp.read_text(encoding="utf-8").replace(
        "&dynamics",
        "&fdda\n grid_fdda = 1, 0, 0, 0, 0, 0,\n/\n&dynamics",
        1,
    )
    inp.write_text(text, encoding="utf-8")
    report = analyze_namelists(wps, inp, source_top_pressure_pa=5000.0)
    assert report["verdict"] == "FAIL"
    issue = next(item for item in report["issues"]
                 if item["code"] == "FDDA_INPUT_NOT_PRODUCED")
    assert "wrffdda_d0N" in issue["message"]


def test_mosaic_monalb_rdlai2d_gate_on_nondefault_values(tmp_path):
    """sf_surface_mosaic/usemonalb/rdlai2d are initialized-state selectors:
    the WRF-default values pass cleanly, anything else is refused inside
    UNSUPPORTED_PHYSICS_STATE with the exact selector named."""
    report = analyze_namelists(
        *_write_pair(tmp_path, extra_physics=(
            "sf_surface_mosaic = 0, 0, 0, 0, 0, 0,\n"
            " usemonalb = .false.,\n rdlai2d = .false.,")),
        source_top_pressure_pa=5000.0,
    )
    assert report["verdict"] == "PASS"

    report = analyze_namelists(
        *_write_pair(tmp_path, extra_physics=(
            "sf_surface_mosaic = 1, 1, 1, 1, 1, 1,\n"
            " usemonalb = .true.,\n rdlai2d = .true.,")),
        source_top_pressure_pa=5000.0,
    )
    assert report["verdict"] == "FAIL"
    message = next(item["message"] for item in report["issues"]
                   if item["code"] == "UNSUPPORTED_PHYSICS_STATE")
    assert "sf_surface_mosaic=1" in message
    assert "usemonalb=.true." in message
    assert "rdlai2d=.true." in message
    assert not [item for item in report["issues"]
                if item["code"] == "UNCLASSIFIED_NAMELIST_SETTING"]


def test_unsupported_land_model_fails_precisely(tmp_path):
    wps, inp = _write_pair(tmp_path)
    text = inp.read_text(encoding="utf-8").replace(
        "sf_surface_physics = 2, 2, 2, 2, 2, 2,",
        "sf_surface_physics = 3, 3, 3, 3, 3, 3,",
    )
    inp.write_text(text, encoding="utf-8")
    report = analyze_namelists(wps, inp)
    assert report["verdict"] == "FAIL"
    assert any(
        item["code"] == "UNSUPPORTED_PHYSICS_STATE"
        and "Noah=2/4 layers" in item["message"]
        for item in report["issues"]
    )


def test_source_vertical_coverage_is_not_extrapolated(tmp_path):
    report = analyze_namelists(
        *_write_pair(tmp_path, mass_levels=80),
        source_top_pressure_pa=10000.0,
    )
    assert report["verdict"] == "FAIL"
    assert any(
        item["code"] == "INVALID_VERTICAL_GRID"
        and "source atmosphere stops" in item["message"]
        for item in report["issues"]
    )


def test_public_engine_cli_emits_machine_report_and_exit_status(tmp_path, capsys):
    wps, inp = _write_pair(tmp_path, max_dom=6, mp=8)
    assert source_cli_main([
        "--namelist-support-report",
        "--wps-namelist", str(wps),
        "--namelist-input", str(inp),
        "--source-top-pressure-pa", "5000",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == "rw-wps.namelist-support.v1"
    assert report["required_state"]["stock_wrf_export"]["verdict"] == "PASS"

    bad_wps, bad_inp = _write_pair(
        tmp_path, max_dom=6, mp=8, extra_physics="unknown_state_switch = 1,"
    )
    assert source_cli_main([
        "--namelist-support-report",
        "--wps-namelist", str(bad_wps),
        "--namelist-input", str(bad_inp),
    ]) == EXIT_CONFIG
    assert json.loads(capsys.readouterr().out)["verdict"] == "FAIL"


# ---------------------------------------------------------------------------
# Step one of migrating-from-wps.md, when the file is not there yet
# ---------------------------------------------------------------------------

def test_the_support_report_refuses_a_missing_namelist_in_one_sentence(
        tmp_path, capsys):
    """`FileNotFoundError` out of pathlib, five frames deep.

    docs/migrating-from-wps.md makes this the FIRST command a person
    migrating an existing WRF setup runs, so pointing it at a file that
    is not there yet is the commonest way to meet it -- and `gpuwm
    import-namelist` answers the identical condition with one sentence.
    Two surfaces, one condition, one answer.
    """

    wps, inp = _write_pair(tmp_path, max_dom=1, mp=8)
    missing = tmp_path / "not-written-yet.namelist.wps"

    assert source_cli_main([
        "--namelist-support-report",
        "--wps-namelist", str(missing),
        "--namelist-input", str(inp),
    ]) == EXIT_CONFIG
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert "cannot read namelist.wps" in captured.err
    assert str(missing) in captured.err

    # The namelist.input half is named as itself, not as the other one.
    assert source_cli_main([
        "--namelist-support-report",
        "--wps-namelist", str(wps),
        "--namelist-input", str(tmp_path / "absent.namelist.input"),
    ]) == EXIT_CONFIG
    assert "cannot read namelist.input" in capsys.readouterr().err

    # Negative control: with both files present the report still runs,
    # so the guard above is a guard and not a blanket refusal.
    assert source_cli_main([
        "--namelist-support-report",
        "--wps-namelist", str(wps),
        "--namelist-input", str(inp),
        "--source-top-pressure-pa", "5000",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["schema"] \
        == "rw-wps.namelist-support.v1"


def test_the_two_migration_doors_give_the_same_sentence(tmp_path, capsys):
    """`gpuwm import-namelist` and `rw-wps --namelist-support-report`
    now share one refusal, through one function."""

    from gpuwm.cli import main as gpuwm_main

    missing = tmp_path / "nope.namelist.wps"
    _wps, inp = _write_pair(tmp_path, max_dom=1, mp=8)

    assert gpuwm_main(["import-namelist", str(missing), str(inp)]) == 2
    importer = capsys.readouterr().err
    assert source_cli_main([
        "--namelist-support-report",
        "--wps-namelist", str(missing), "--namelist-input", str(inp),
    ]) == EXIT_CONFIG
    reporter = capsys.readouterr().err

    core = f"cannot read namelist.wps {missing}"
    assert core in importer
    assert core in reporter
