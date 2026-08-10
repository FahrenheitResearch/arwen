"""Namelist importer tests (Phase-5 Task 1, recon G10).

The bundle gate retains the original namelist as a fail-loud fixture for
unsupported ``nwp_diagnostics=1`` / default ``use_theta_m=1``, then imports
an explicit effective copy (both set to 0) and reproduces the committed
``configs/real74_4dom.toml`` byte-for-byte modulo the enumerated
importer-behaviour changes (the rrtmg -v2 token; ``bl_pbl_physics = 11``
importing natively since the Shin-Hong port).  Its SubstitutionReport
names mp 55 -> 10 (Morrison) and ra -> RTE+RRTMGP; bl_pbl 11 stopped
being a substitution when the scheme was admitted.  Synthetic fixtures
cover the importer's other rejection rules without needing the bundle.
All CPU.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import pytest

import gpuwm.cli as cli
from gpuwm.physics_compat import CONSTANT_DOWNWARD_LONGWAVE_ACK
from gpuwm.experiment import DEFAULT_COLUMN_CHUNK, load_experiment
from gpuwm.namelist_import import (SubstitutionReport, import_namelists,
                                   parse_namelist)
from gpuwm.native_wrf_contract import validate_native_lambert_contracts
from gpuwm.static.lambert import (
    grids_from_projection_config,
    grids_from_wps_namelist,
)
from gpuwm.verify.cases.real74_d01 import BUNDLE

REPO = Path(__file__).resolve().parents[1]
BUNDLE_WPS = BUNDLE / "namelists" / "namelist.wps"
BUNDLE_INPUT = BUNDLE / "namelists" / "namelist.input"
requires_bundle = pytest.mark.skipif(
    not BUNDLE_INPUT.exists(),
    reason="WRF 1974 reference bundle not present")

#: Synthetic 2-domain namelist pair mirroring the bundle's shape (child
#: ratio 3 at (40, 30) inside a 100 x 80 parent; ISHMAEL/Shin-Hong/RRTMG
#: selections -- ISHMAEL and RRTMG exercise both ratified substitutions
#: without the bundle, and Shin-Hong exercises the native bl_pbl=11
#: import that replaced its former substitution).
WPS_TEXT = """\
&share
 wrf_core = 'ARW',
 max_dom = 2,
 start_date = '1999-05-03_12:00:00', '1999-05-03_12:00:00',
 end_date   = '1999-05-03_18:00:00', '1999-05-03_18:00:00',
 interval_seconds = 21600,
 io_form_geogrid = 2,
/
&geogrid
 parent_id         = 1, 1,
 parent_grid_ratio = 1, 3,
 i_parent_start    = 1, 40,
 j_parent_start    = 1, 30,
 e_we              = 101, 61,
 e_sn              = 81, 61,
 geog_data_res     = 'default', 'default',
 dx = 12000,
 dy = 12000,
 map_proj = 'lambert',
 ref_lat   = 39.7,
 ref_lon   = -83.9,
 truelat1  = 30.0,
 truelat2  = 60.0,
 stand_lon = -83.9,
 geog_data_path = '/geog',
/
&ungrib
 out_format = 'WPS',
 prefix = 'ERA5',
/
&metgrid
 fg_name = 'ERA5',
/
"""

INPUT_TEXT = """\
&time_control
 run_hours = 6,
 start_year = 1999, 1999,
 start_month = 05, 05,
 start_day = 03, 03,
 start_hour = 12, 12,
 end_year = 1999, 1999,
 end_month = 05, 05,
 end_day = 03, 03,
 end_hour = 18, 18,
 interval_seconds = 21600,
 input_from_file = .true., .true.,
 history_interval = 60, 15,
 restart = .false.,
 restart_interval = 60,
/
&domains
 time_step = 60,
 max_dom = 2,
 e_we = 101, 61,
 e_sn = 81, 61,
 e_vert = 9, 9,
 eta_levels = 1.0, 0.9, 0.8, 0.7, 0.6,
              0.5, 0.4, 0.2, 0.0,
 p_top_requested = 5000,
 dx = 12000.0, 4000.0,
 dy = 12000.0, 4000.0,
 grid_id = 1, 2,
 parent_id = 0, 1,
 i_parent_start = 1, 40,
 j_parent_start = 1, 30,
 parent_grid_ratio = 1, 3,
 parent_time_step_ratio = 1, 3,
 feedback = 0,
 smooth_option = 0,
/
&physics
 mp_physics = 55, 55,
 ra_lw_physics = 4, 4,
 ra_sw_physics = 4, 4,
 radt = 12, 3,
 sf_sfclay_physics = 91, 91,
 sf_surface_physics = 2, 2,
 bl_pbl_physics = 11, 11,
 bldt = 0, 0,
 cu_physics = 1, 0,
 cudt = 5, 0,
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
 base_temp = 2.90D2,
 damp_opt = 3,
 zdamp = 2*5000.,
 dampcoef = 0.2, 0.2,
 khdif = 0, 0,
 kvdif = 0, 0,
 non_hydrostatic = 2*.true.,
 use_theta_m = 0,
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


def _pair(tmp_path, wps=WPS_TEXT, inp=INPUT_TEXT):
    wps_path = tmp_path / "namelist.wps"
    inp_path = tmp_path / "namelist.input"
    wps_path.write_text(wps)
    inp_path.write_text(inp)
    return wps_path, inp_path


def _generic_hierarchy_pair(tmp_path, max_dom: int):
    """Valid all-sibling hierarchy used to exercise every compiled count."""

    def col(root, child=None):
        if child is None:
            child = root
        return ", ".join(str(value) for value in (
            [root] + [child] * (max_dom - 1)))

    start = col("'1999-05-03_12:00:00'")
    end = col("'1999-05-03_18:00:00'")
    wps = f"""\
&share
 wrf_core = 'ARW',
 max_dom = {max_dom},
 start_date = {start},
 end_date = {end},
 interval_seconds = 21600,
/
&geogrid
 parent_id = {col(1)},
 parent_grid_ratio = {col(1, 3)},
 i_parent_start = {col(1, 40)},
 j_parent_start = {col(1, 30)},
 e_we = {col(101, 61)},
 e_sn = {col(81, 61)},
 geog_data_res = {col("'default'")},
 dx = 12000,
 dy = 12000,
 map_proj = 'lambert',
 ref_lat = 39.7,
 ref_lon = -83.9,
 truelat1 = 30.0,
 truelat2 = 60.0,
 stand_lon = -83.9,
 geog_data_path = '/geog',
/
"""
    inp = f"""\
&time_control
 run_hours = 6,
 start_year = {col(1999)},
 start_month = {col(5)},
 start_day = {col(3)},
 start_hour = {col(12)},
 end_year = {col(1999)},
 end_month = {col(5)},
 end_day = {col(3)},
 end_hour = {col(18)},
 input_from_file = {col('.true.')},
 history_interval = {col(60, 15)},
 restart_interval = 60,
/
&domains
 time_step = 60,
 max_dom = {max_dom},
 e_we = {col(101, 61)},
 e_sn = {col(81, 61)},
 e_vert = {col(9)},
 eta_levels = 1.0, 0.9, 0.8, 0.7, 0.6,
              0.5, 0.4, 0.2, 0.0,
 p_top_requested = 5000,
 dx = {col(12000.0, 4000.0)},
 dy = {col(12000.0, 4000.0)},
 grid_id = {', '.join(str(i) for i in range(1, max_dom + 1))},
 parent_id = {col(0, 1)},
 i_parent_start = {col(1, 40)},
 j_parent_start = {col(1, 30)},
 parent_grid_ratio = {col(1, 3)},
 parent_time_step_ratio = {col(1, 3)},
 feedback = 0,
 smooth_option = 0,
/
&physics
 mp_physics = {col(6)},
 ra_lw_physics = {col(4)},
 ra_sw_physics = {col(4)},
 radt = {col(12, 3)},
 sf_sfclay_physics = {col(91)},
 sf_surface_physics = {col(2)},
 bl_pbl_physics = {col(1)},
 bldt = {col(0)},
 cu_physics = {col(1, 0)},
 cudt = {col(5, 0)},
/
&dynamics
 hybrid_opt = 2,
 etac = 0.2,
 w_damping = 1,
 epssm = {col(0.5)},
 diff_opt = {col(2)},
 km_opt = {col(4)},
 mix_full_fields = {col('.true.')},
 diff_6th_opt = {col(2)},
 diff_6th_factor = {col(0.12, 0.10)},
 diff_6th_slopeopt = {col(1)},
 base_temp = 290.0,
 damp_opt = 3,
 zdamp = {col(5000.0)},
 dampcoef = {col(0.2)},
 khdif = {col(0)},
 kvdif = {col(0)},
 non_hydrostatic = {col('.true.')},
 use_theta_m = 0,
 moist_adv_opt = {col(1)},
/
&bdy_control
 spec_bdy_width = 5,
 spec_zone = 1,
 relax_zone = 4,
 specified = {col('.true.', '.false.')},
 nested = {col('.false.', '.true.')},
/
"""
    return _pair(tmp_path, wps=wps, inp=inp)


def _effective_bundle_input(tmp_path):
    """Scope-adjusted publication namelist accepted by gpuwm.

    The original is retained read-only and tested as rejected below (on
    use_theta_m).  This copy makes the adjudicated dry-theta choice
    explicit, and DROPS the campaign's ``nwp_diagnostics = 1``: the knob
    is supported now, but the committed flagship TOML predates the
    UP_HELI_MAX diagnostic and its byte identity is the round-trip gate --
    omission resolves to the same Registry default 0 the old scope
    adjustment pinned.  test_nwp_diagnostics_maps_and_reaches_every_run_
    config proves the = 1 spelling translates.
    """
    text = BUNDLE_INPUT.read_text(encoding="utf-8")
    nwp = " nwp_diagnostics                     = 1,\n"
    assert text.count(nwp) == 1
    text = text.replace(nwp, "")
    assert text.count("&dynamics\n") == 1
    text = text.replace("&dynamics\n", "&dynamics\n use_theta_m = 0,\n")
    # Keep the publication filename stable; the emitted TOML header records
    # input_path.name and the committed effective config pins that identity.
    path = tmp_path / "namelist.input"
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def test_parser_handles_continuation_lines_and_types(tmp_path):
    """eta_levels spans multiple '='-free continuation lines; booleans,
    ints, floats and quoted strings decode to Python types; Fortran
    repetition constants (2*5000.) expand and D-exponent reals (2.90D2)
    parse (shadow S2)."""
    _, inp_path = _pair(tmp_path)
    parsed = parse_namelist(inp_path)
    assert len(parsed["domains"]["eta_levels"]) == 9
    assert parsed["domains"]["eta_levels"][0] == 1.0
    assert parsed["time_control"]["input_from_file"] == [True, True]
    assert parsed["time_control"]["restart"] == [False]
    assert parsed["domains"]["time_step"] == [60]
    # D-exponent real and repetition constants
    assert parsed["dynamics"]["base_temp"] == [290.0]
    assert parsed["dynamics"]["zdamp"] == [5000.0, 5000.0]
    assert parsed["dynamics"]["non_hydrostatic"] == [True, True]
    assert parsed["dynamics"]["use_theta_m"] == [0]
    wps = parse_namelist(tmp_path / "namelist.wps")
    assert wps["share"]["start_date"][0] == "1999-05-03_12:00:00"
    assert wps["geogrid"]["map_proj"] == ["lambert"]


def test_parser_repetition_edge_cases(tmp_path):
    f = tmp_path / "edge.nml"
    f.write_text("&s\n a = 3*1.5,\n b = 4*.true.,\n c = 2*'x',\n"
                 " d = 1.0D-2, -2.5d0,\n/\n")
    parsed = parse_namelist(f)
    assert parsed["s"]["a"] == [1.5, 1.5, 1.5]
    assert parsed["s"]["b"] == [True, True, True, True]
    assert parsed["s"]["c"] == ["x", "x"]
    assert parsed["s"]["d"] == [0.01, -2.5]


# ---------------------------------------------------------------------------
# Synthetic import: staggered conversion, substitutions, derived chain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("max_dom", range(1, 22))
def test_importer_builds_every_supported_hierarchy_cardinality(
        tmp_path, max_dom):
    toml_text, _report = import_namelists(
        *_generic_hierarchy_pair(tmp_path, max_dom),
        name=f"generic-{max_dom}")
    resolved = tmp_path / "resolved.toml"
    resolved.write_text(toml_text, encoding="utf-8")
    exp = load_experiment(resolved)

    assert [domain.grid_id for domain in exp.domains] == list(
        range(1, max_dom + 1))
    assert exp.root.parent_id == 0
    assert all(domain.parent_id == 1 for domain in exp.domains[1:])
    assert all(domain.run.nested for domain in exp.domains[1:])
    assert all(not domain.run.specified for domain in exp.domains[1:])


def _replace_namelist_column(text: str, key: str, values) -> str:
    replacement = f" {key} = {', '.join(str(value) for value in values)},"
    lines = text.splitlines()
    matches = [index for index, line in enumerate(lines)
               if line.strip().startswith(f"{key} =")]
    assert len(matches) == 1, (key, matches)
    lines[matches[0]] = replacement
    return "\n".join(lines) + "\n"


def test_six_domain_deep_chain_and_siblings_use_varied_valid_geometry(
        tmp_path):
    wps, inp = _generic_hierarchy_pair(tmp_path, 6)
    wps_text = wps.read_text(encoding="utf-8")
    input_text = inp.read_text(encoding="utf-8")
    wps_columns = {
        "parent_id": (1, 1, 2, 3, 1, 2),
        "parent_grid_ratio": (1, 3, 2, 3, 2, 4),
        "i_parent_start": (1, 40, 30, 20, 150, 100),
        "j_parent_start": (1, 50, 25, 25, 140, 100),
        "e_we": (301, 181, 121, 151, 101, 161),
        "e_sn": (301, 181, 121, 151, 101, 161),
    }
    input_columns = {
        **wps_columns,
        "parent_id": (0, 1, 2, 3, 1, 2),
        "parent_time_step_ratio": (1, 3, 2, 3, 2, 4),
        "dx": (12000.0, 4000.0, 2000.0, 2000.0 / 3.0,
               6000.0, 1000.0),
        "dy": (12000.0, 4000.0, 2000.0, 2000.0 / 3.0,
               6000.0, 1000.0),
    }
    for key, values in wps_columns.items():
        wps_text = _replace_namelist_column(wps_text, key, values)
    for key, values in input_columns.items():
        input_text = _replace_namelist_column(input_text, key, values)
    wps.write_text(wps_text, encoding="utf-8")
    inp.write_text(input_text, encoding="utf-8")

    toml_text, _report = import_namelists(wps, inp, name="six-domain-tree")
    resolved = tmp_path / "six-domain-tree.toml"
    resolved.write_text(toml_text, encoding="utf-8")
    exp = load_experiment(resolved)

    assert [(domain.grid_id, domain.parent_id)
            for domain in exp.domains] == [
                (1, 0), (2, 1), (3, 2), (4, 3), (5, 1), (6, 2)]
    assert [domain.parent_grid_ratio for domain in exp.domains] == [
        1, 3, 2, 3, 2, 4]
    assert [domain.run.nx for domain in exp.domains] == [
        300, 180, 120, 150, 100, 160]


@pytest.mark.parametrize(
    "parents, label",
    [((0, 9, 1), "orphan"), ((0, 3, 2), "cycle")],
)
def test_importer_rejects_orphan_and_cycle_before_initialization(
        tmp_path, parents, label):
    wps, inp = _generic_hierarchy_pair(tmp_path, 3)
    wps_text = _replace_namelist_column(
        wps.read_text(encoding="utf-8"), "parent_id",
        (1, parents[1], parents[2]))
    input_text = _replace_namelist_column(
        inp.read_text(encoding="utf-8"), "parent_id", parents)
    wps.write_text(wps_text, encoding="utf-8")
    inp.write_text(input_text, encoding="utf-8")

    with pytest.raises(ValueError, match="previously declared domain"):
        import_namelists(wps, inp, name=f"rejected-{label}")


@pytest.mark.parametrize(
    "key, values",
    [
        ("parent_id", (0, 1.9)),
        ("parent_grid_ratio", (1, ".true.")),
        ("e_we", (101, 61.5)),
    ],
)
def test_importer_does_not_coerce_noninteger_hierarchy_tokens(
        tmp_path, key, values):
    wps, inp = _generic_hierarchy_pair(tmp_path, 2)
    wps_values = (1, values[1]) if key == "parent_id" else values
    wps_text = _replace_namelist_column(
        wps.read_text(encoding="utf-8"), key, wps_values)
    input_text = _replace_namelist_column(
        inp.read_text(encoding="utf-8"), key, values)
    wps.write_text(wps_text, encoding="utf-8")
    inp.write_text(input_text, encoding="utf-8")

    with pytest.raises(ValueError, match="Fortran integer tokens"):
        import_namelists(wps, inp, name="reject-coercion")


@pytest.mark.parametrize(
    "key, values",
    [
        ("parent_id", (0, 1)),
        ("parent_grid_ratio", (9, 3)),
        ("i_parent_start", (2, 40)),
        ("j_parent_start", (2, 30)),
    ],
)
def test_importer_binds_wps_root_topology(tmp_path, key, values):
    wps, inp = _generic_hierarchy_pair(tmp_path, 2)
    wps.write_text(_replace_namelist_column(
        wps.read_text(encoding="utf-8"), key, values), encoding="utf-8")
    with pytest.raises(ValueError, match="WPS d01 root topology"):
        import_namelists(wps, inp, name="reject-wps-root")


def test_nonintegral_root_spacing_uses_one_exact_ratio_chain(tmp_path):
    wps, inp = _generic_hierarchy_pair(tmp_path, 4)
    wps_text = wps.read_text(encoding="utf-8")
    input_text = inp.read_text(encoding="utf-8")
    topology = {
        "parent_id": (1, 1, 2, 3),
        "parent_grid_ratio": (1, 3, 3, 3),
        "i_parent_start": (1, 50, 100, 100),
        "j_parent_start": (1, 50, 100, 100),
        "e_we": (200, 301, 301, 301),
        "e_sn": (200, 301, 301, 301),
    }
    for key, values in topology.items():
        wps_text = _replace_namelist_column(wps_text, key, values)
    wps_text = _replace_namelist_column(
        wps_text, "dx", (2999.4213047435587,))
    wps_text = _replace_namelist_column(
        wps_text, "dy", (2999.4213047435587,))

    input_topology = {
        **topology,
        "parent_id": (0, 1, 2, 3),
        "parent_time_step_ratio": (1, 3, 3, 3),
        "dx": (
            2999.4213047435587, 999.8071015811862,
            333.26903386039544, 111.08967795346514),
        "dy": (
            2999.4213047435587, 999.8071015811862,
            333.26903386039544, 111.08967795346514),
    }
    for key, values in input_topology.items():
        input_text = _replace_namelist_column(input_text, key, values)
    wps.write_text(wps_text, encoding="utf-8")
    inp.write_text(input_text, encoding="utf-8")

    toml_text, _report = import_namelists(
        wps, inp, name="nonintegral-root-chain")
    resolved = tmp_path / "nonintegral-root-chain.toml"
    resolved.write_text(toml_text, encoding="utf-8")
    exp = load_experiment(resolved)
    wps_grids = grids_from_wps_namelist(wps)
    exp_grids = grids_from_projection_config(exp)
    expected = [domain.run.dx for domain in exp.domains]

    assert [grid.dx for grid in wps_grids] == expected
    assert [grid.dx for grid in exp_grids] == expected

def test_synthetic_import_resolves_and_reports(tmp_path):
    toml_text, report = import_namelists(*_pair(tmp_path), name="synth99")
    out = tmp_path / "synth.toml"
    out.write_text(toml_text)
    exp = load_experiment(out)
    assert exp.name == "synth99"
    assert exp.start_time == datetime(1999, 5, 3, 12)
    assert exp.run_seconds == 21600.0
    assert exp.restart_interval_s == 3600.0
    # staggered e_we/e_sn/e_vert -> mass dims, explicitly
    assert (exp.root.run.nx, exp.root.run.ny) == (100, 80)
    assert (exp.domain(2).run.nx, exp.domain(2).run.ny) == (60, 60)
    assert exp.root.run.nz == 8 and len(exp.vertical.eta_levels) == 9
    # -v2: the importer emits the current WRF-matching snow-coupling token;
    # configurations carrying the older -v1 token keep the -v1 behavior.
    assert exp.root.run.wrf_rrtmg_compatibility == \
        "wrf-rrtmg-4-4-to-rte-rrtmgp-v2"
    # A scalar Fortran namelist assignment changes only the first Registry
    # array element; d02 retains WRF's initialized epssm=0.1 default.
    assert [dc.run.epssm for dc in exp.domains] == [0.5, 0.1]
    # RunConfig-default moist_cq: a Morrison + Shin-Hong suite matches no
    # shipped profile since bl_pbl = 11 imports natively, so the implicit
    # switch takes implicit_runtime_switches' documented fallback (it was
    # [True, True] from the Morrison profile while 11 substituted to YSU).
    assert [dc.run.moist_cq for dc in exp.domains] == [False, False]
    assert toml_text.count("epssm = 0.1") == 1
    assert 'wrf_rrtmg_compatibility = "wrf-rrtmg-4-4-to-rte-rrtmgp-v2"' \
        in toml_text
    # derived chain: the emitted TOML carries NO child dx/dt keys
    assert exp.dt_exact(2) == Fraction(20)
    assert exp.dx_exact(2) == Fraction(4000)
    assert "dx = 4000" not in toml_text
    # ratified substitutions, structured and named -- never silent
    assert isinstance(report, SubstitutionReport)
    subs = {(s.key, s.wrf_value): (s.gpuwm_key, s.gpuwm_value,
                                   s.gpuwm_name)
            for s in report.substitutions}
    assert subs[("mp_physics", 55)] == ("mp_physics", 10,
                                        "Morrison 2-moment")
    # bl_pbl 11 imports natively since the Shin-Hong port: no substitution
    # row, and the selector reaches every domain as itself (the mp8/mp18
    # promotion contract shape).
    assert not any(s.key == "bl_pbl_physics" for s in report.substitutions)
    assert all(dc.run.bl_pbl_physics == 11 for dc in exp.domains)
    assert subs[("ra_lw_physics/ra_sw_physics", 4)] == (
        "ra_physics", 4, "RTE+RRTMGP")
    formatted = report.format()
    for token in ("mp_physics 55", "Morrison 2-moment",
                  "RRTMG", "RTE+RRTMGP"):
        assert token in formatted, token
    assert "bl_pbl_physics 11" not in formatted
    # every consumed-without-counterpart key carries a reason
    assert all(d.reason for d in report.dropped)
    dropped_keys = {(d.section, d.key) for d in report.dropped}
    assert ("time_control", "restart") in dropped_keys
    assert ("geogrid", "geog_data_path") in dropped_keys


def test_import_accepts_real_shaped_staggered_start_columns_and_five_minute_forcing(
        tmp_path):
    wps = WPS_TEXT.replace(
        "start_date = '1999-05-03_12:00:00', "
        "'1999-05-03_12:00:00',",
        "start_date = '1999-05-03_12:00:00', "
        "'1999-05-03_12:05:00',").replace(
            "interval_seconds = 21600", "interval_seconds = 300")
    inp = INPUT_TEXT.replace(
        " start_hour = 12, 12,",
        " start_hour = 12, 12,\n"
        " start_minute = 00, 05,\n"
        " start_second = 00, 00,").replace(
            "interval_seconds = 21600", "interval_seconds = 300")

    toml_text, _report = import_namelists(
        *_pair(tmp_path, wps=wps, inp=inp), name="staggered-five-minute")
    resolved = tmp_path / "staggered-five-minute.toml"
    resolved.write_text(toml_text, encoding="utf-8")
    exp = load_experiment(resolved)

    assert exp.domain_start_time(1) == datetime(1999, 5, 3, 12, 0)
    assert exp.domain_start_time(2) == datetime(1999, 5, 3, 12, 5)
    assert exp.domain_start_offset_exact(2) == Fraction(300)
    assert "start_time = 1999-05-03T12:05:00" in toml_text


def test_imported_two_domain_geometry_passes_native_hierarchy_contract(
        tmp_path):
    wps_path, input_path = _pair(tmp_path)
    toml_text, _report = import_namelists(wps_path, input_path, name="geometry")
    resolved = tmp_path / "resolved.toml"
    resolved.write_text(toml_text, encoding="utf-8")
    exp = load_experiment(resolved)
    grids = validate_native_lambert_contracts(
        exp, wps_path, source_name="synthetic")
    assert len(grids) == 2
    assert (grids[0].e_we, grids[0].e_sn, grids[0].dx) == (101, 81, 12000.0)
    assert (grids[1].e_we, grids[1].e_sn, grids[1].dx) == (61, 61, 4000.0)


def test_explicit_epssm_column_preserves_child_value(tmp_path):
    inp = INPUT_TEXT.replace(" epssm = 0.5,", " epssm = 0.5, 0.5,")
    toml_text, _ = import_namelists(*_pair(tmp_path, inp=inp))
    out = tmp_path / "epssm-explicit.toml"
    out.write_text(toml_text)
    assert [dc.run.epssm for dc in load_experiment(out).domains] == [0.5, 0.5]
    assert "epssm = 0.1" not in toml_text


def test_omitted_epssm_uses_registry_default_on_every_domain(tmp_path):
    inp = INPUT_TEXT.replace(" epssm = 0.5,\n", "")
    toml_text, _ = import_namelists(*_pair(tmp_path, inp=inp))
    out = tmp_path / "epssm-default.toml"
    out.write_text(toml_text)
    assert [dc.run.epssm for dc in load_experiment(out).domains] == [0.1, 0.1]
    assert toml_text.count("epssm = 0.1") == 1


def test_direct_scheme_ids_map_without_substitution(tmp_path):
    inp = INPUT_TEXT.replace("mp_physics = 55, 55",
                             "mp_physics = 10, 10").replace(
        "bl_pbl_physics = 11, 11", "bl_pbl_physics = 1, 1")
    _, report = import_namelists(*_pair(tmp_path, inp=inp))
    keys = {s.key for s in report.substitutions}
    assert "mp_physics" not in keys
    assert "bl_pbl_physics" not in keys
    assert "ra_lw_physics/ra_sw_physics" in keys  # RRTMG -> RTE+RRTMGP


def test_thompson_maps_without_substitution_or_environment(
        monkeypatch, tmp_path):
    """mp8 imports first-class: no enable guard, no table-root environment.

    This pinned the guarded-Thompson-pending contract (both env vars
    required) until the canonical classic tables became package data and
    mp8 was promoted (product decision, product/v1 packaging lane
    2026-07-28).  The byte validation of the resolved table root still
    fails closed at load -- it moved, it did not disappear.
    """
    monkeypatch.delenv("GPUWM_EXPERIMENTAL_THOMPSON_MP8", raising=False)
    monkeypatch.delenv("GPUWM_THOMPSON_TABLE_ROOT", raising=False)
    inp = INPUT_TEXT.replace("mp_physics = 55, 55", "mp_physics = 8, 8")
    toml_text, report = import_namelists(*_pair(tmp_path, inp=inp))
    output = tmp_path / "thompson.toml"
    output.write_text(toml_text, encoding="utf-8")
    assert load_experiment(output).root.run.mp_physics == 8
    assert not any(item.key == "mp_physics" for item in report.substitutions)


def test_nssl2_maps_first_class_without_substitution(tmp_path):
    """mp18 imports selector-for-selector since the certified NSSL merge
    (product/v1 NSSL lane 2026-07-29): the mapping row is native, no
    substitution is recorded, and the emitted TOML loads with the
    selector intact.  The scheme's maturity ("wrf-matched-run-candidate" in
    physics_registry_v2.json) is a registry statement, not an importer
    gate -- exactly the mp8 promotion contract shape."""
    inp = INPUT_TEXT.replace("mp_physics = 55, 55", "mp_physics = 18, 18")
    toml_text, report = import_namelists(*_pair(tmp_path, inp=inp))
    output = tmp_path / "nssl2.toml"
    output.write_text(toml_text, encoding="utf-8")
    assert load_experiment(output).root.run.mp_physics == 18
    assert not any(item.key == "mp_physics" for item in report.substitutions)


def test_target_thompson_mynn_ruc_suite_imports_without_substitution(tmp_path):
    inp = INPUT_TEXT.replace("mp_physics = 55, 55",
                             "mp_physics = 8, 8").replace(
        "sf_sfclay_physics = 91, 91",
        "sf_sfclay_physics = 5, 5").replace(
        "sf_surface_physics = 2, 2",
        "sf_surface_physics = 3, 3").replace(
        "bl_pbl_physics = 11, 11",
        "bl_pbl_physics = 5, 5").replace(
        " bldt = 0, 0,", " num_soil_layers = 9,\n bldt = 0, 0,")
    toml_text, report = import_namelists(
        *_pair(tmp_path, inp=inp), name="friend_suite")
    output = tmp_path / "friend-suite.toml"
    output.write_text(toml_text, encoding="utf-8")
    cfg = load_experiment(output).root.run
    assert (cfg.mp_physics, cfg.sf_sfclay_physics,
            cfg.sf_surface_physics, cfg.bl_pbl_physics,
            cfg.num_soil_layers) == (8, 5, 3, 5, 9)
    assert [item.key for item in report.substitutions] == [
        "ra_lw_physics/ra_sw_physics"
    ]


def test_split_radiation_imports_natively_and_refuses_the_unported_half(
        tmp_path):
    """The native split pair, and the loud refusal that replaced a quiet one.

    This began as ``test_rrtm_dudhia_pair_emits_native_split_radiation_config``
    and asserted that a 1/1 WRF RRTM+Dudhia namelist imported to a loadable
    TOML.  It did -- and every such TOML then raised NotImplementedError from
    ``gpuwm/core/physics.py`` ``initialize_physics`` at driver construction,
    because the RRTM longwave is not ported.  The importer refuses every OTHER
    unported selector outright (``bl_pbl_physics = 2`` is "no ratified gpuwm
    mapping"), so accepting this one was the anomaly, not the rule; it survived
    only because ``validate_run_config`` had ``ra_lw_physics`` 1 inside its
    accepted set with no readiness blocker behind it.  Now it is refused where
    the rest are.  What must NOT happen is a silent remap onto RTE+RRTMGP, and
    that is still asserted: the refusal names the selector and no ``ra_``
    substitution is offered anywhere.
    """
    def _radiation(lw, sw):
        return INPUT_TEXT.replace(
            " ra_lw_physics = 4, 4,", f" ra_lw_physics = {lw}, {lw},").replace(
            " ra_sw_physics = 4, 4,", f" ra_sw_physics = {sw}, {sw},").replace(
            "&physics\n", "&physics\n icloud = 1,\n swrad_scat = 0.8,\n")

    # The unported half: refused by name, and never substituted.
    with pytest.raises(NotImplementedError, match="ra_lw_physics=1") as raised:
        import_namelists(*_pair(tmp_path, inp=_radiation(1, 1)),
                         name="rrtm_dudhia")
    assert "16-band" in str(raised.value)

    # The implemented half of the same split representation still imports, and
    # this is where the native-split coverage lives: ra_physics is zeroed, the
    # two component selectors are emitted separately, icloud and swrad_scat
    # ride with them, and nothing on a radiation key is substituted.
    # Dudhia-only under Noah is a declared constant downward longwave
    # since the constant-GLW guard, and a WRF namelist cannot spell a
    # gpuwm declaration -- so it arrives through the importer's channel,
    # which exists for exactly this.  WRF v4.6.1 refuses the same
    # namelist outright (phys/module_radiation_driver.F:2245).
    toml_text, report = import_namelists(
        *_pair(tmp_path, inp=_radiation(0, 1)), name="dudhia_split",
        acknowledgements=(CONSTANT_DOWNWARD_LONGWAVE_ACK,))
    output = tmp_path / "dudhia_split.toml"
    output.write_text(toml_text)
    exp = load_experiment(output)
    for dc in exp.domains:
        assert dc.run.ra_physics == 0
        assert dc.run.ra_lw_physics == 0
        assert dc.run.ra_sw_physics == 1
        assert dc.run.icloud == 1
        assert dc.run.swrad_scat == pytest.approx(0.8)
    assert "ra_lw_physics = 0" in toml_text
    assert "ra_sw_physics = 1" in toml_text
    assert "swrad_scat = 0.8" in toml_text
    assert all("ra_" not in substitution.key
               for substitution in report.substitutions)


def test_morr_rimed_ice_defaults_hail_and_preserves_explicit_graupel(
        tmp_path):
    """Registry.EM_COMMON:2663-2666 defaults the scalar option to hail."""
    toml_text, _ = import_namelists(*_pair(tmp_path))
    out = tmp_path / "hail.toml"
    out.write_text(toml_text)
    assert "morr_rimed_ice = 1" in toml_text
    assert load_experiment(out).root.run.morr_rimed_ice == 1

    inp = INPUT_TEXT.replace("&physics\n",
                             "&physics\n morr_rimed_ice = 0,\n")
    toml_text, _ = import_namelists(*_pair(tmp_path, inp=inp))
    out.write_text(toml_text)
    assert "morr_rimed_ice = 0" in toml_text
    assert load_experiment(out).root.run.morr_rimed_ice == 0


def test_morr_rimed_ice_import_rejects_invalid_value(tmp_path):
    inp = INPUT_TEXT.replace("&physics\n",
                             "&physics\n morr_rimed_ice = 2,\n")
    with pytest.raises(ValueError, match="morr_rimed_ice"):
        import_namelists(*_pair(tmp_path, inp=inp))


def test_wsm6_and_hail_opt_import_without_substitution(tmp_path):
    inp = INPUT_TEXT.replace("mp_physics = 55, 55",
                             "mp_physics = 6, 6").replace(
        "&physics\n", "&physics\n hail_opt = 1,\n")
    toml_text, report = import_namelists(*_pair(tmp_path, inp=inp))
    out = tmp_path / "wsm6.toml"
    out.write_text(toml_text, encoding="utf-8")
    cfg = load_experiment(out).root.run
    assert cfg.mp_physics == 6
    assert cfg.wsm6_hail_opt == 1
    assert "wsm6_hail_opt = 1" in toml_text
    assert not any(item.key == "mp_physics" for item in report.substitutions)


def test_wsm6_hail_opt_import_rejects_invalid_value(tmp_path):
    inp = INPUT_TEXT.replace("mp_physics = 55, 55",
                             "mp_physics = 6, 6").replace(
        "&physics\n", "&physics\n hail_opt = 2,\n")
    with pytest.raises(ValueError, match="hail_opt"):
        import_namelists(*_pair(tmp_path, inp=inp))


# ---------------------------------------------------------------------------
# Importer rejection fixtures
# ---------------------------------------------------------------------------

def test_rejects_unmapped_scheme(tmp_path):
    # mp_physics = 8 stood here while Thompson was env-gated; it imports
    # first-class now (see test_thompson_maps_without_substitution_or_
    # environment).  Lin (mp 2) has no gpuwm implementation and keeps this
    # rejection path honest.
    inp = INPUT_TEXT.replace("mp_physics = 55, 55", "mp_physics = 2, 2")
    with pytest.raises(ValueError, match="no ratified gpuwm mapping"):
        import_namelists(*_pair(tmp_path, inp=inp))
    inp = INPUT_TEXT.replace("bl_pbl_physics = 11, 11",
                             "bl_pbl_physics = 5, 5")
    # WRF v4.6.1 explicitly accepts MYNN PBL with the MM5 surface layer.
    toml_text, _ = import_namelists(*_pair(tmp_path, inp=inp))
    assert "bl_pbl_physics = 5" in toml_text
    assert "sf_sfclay_physics = 91" in toml_text


def test_rejects_moving_nest_keys(tmp_path):
    inp = INPUT_TEXT.replace(" time_step = 60,",
                             " time_step = 60,\n num_moves = 2,")
    with pytest.raises(ValueError, match="moving-nest"):
        import_namelists(*_pair(tmp_path, inp=inp))


def test_rejects_wps_input_layout_mismatch(tmp_path):
    inp = INPUT_TEXT.replace("e_we = 101, 61", "e_we = 101, 64")
    with pytest.raises(ValueError, match="nest layout mismatch"):
        import_namelists(*_pair(tmp_path, inp=inp))
    inp = INPUT_TEXT.replace("i_parent_start = 1, 40",
                             "i_parent_start = 1, 42")
    with pytest.raises(ValueError, match="nest layout mismatch"):
        import_namelists(*_pair(tmp_path, inp=inp))


def test_rejects_hand_typed_child_dx(tmp_path):
    """The namelist-side 500-m fixture: &domains hand-types a child dx
    that contradicts the parent/ratio chain."""
    inp = INPUT_TEXT.replace("dx = 12000.0, 4000.0",
                             "dx = 12000.0, 500.0")
    with pytest.raises(ValueError, match="never hand-typed"):
        import_namelists(*_pair(tmp_path, inp=inp))
    # a truncated decimal of the exact value is accepted (the bundle's
    # 333.333333-style entry), and the emitted TOML drops it
    inp = INPUT_TEXT.replace("dx = 12000.0, 4000.0",
                             "dx = 12000.0, 3999.999999")
    toml_text, _ = import_namelists(*_pair(tmp_path, inp=inp))
    assert "3999" not in toml_text


def test_rejects_vertical_nesting(tmp_path):
    inp = INPUT_TEXT.replace("e_vert = 9, 9", "e_vert = 9, 8")
    with pytest.raises(ValueError, match="must be identical"):
        import_namelists(*_pair(tmp_path, inp=inp))
    inp = INPUT_TEXT.replace(" time_step = 60,",
                             " time_step = 60,\n vert_refine_method = 1,")
    with pytest.raises(ValueError, match="vert_refine_method"):
        import_namelists(*_pair(tmp_path, inp=inp))


def test_rejects_idealized_input_from_file(tmp_path):
    inp = INPUT_TEXT.replace("input_from_file = .true., .true.,",
                             "input_from_file = .true., .false.,")
    with pytest.raises(ValueError, match="input_from_file"):
        import_namelists(*_pair(tmp_path, inp=inp))


def test_rejects_unmapped_namelist_key(tmp_path):
    # tracer_opt has no gpuwm counterpart of any class (rk_ord, the old
    # example here, is now a validated fixed-by-ArWen key).
    inp = INPUT_TEXT.replace(" hybrid_opt = 2,",
                             " hybrid_opt = 2,\n tracer_opt = 2,")
    with pytest.raises(ValueError, match="unmapped key"):
        import_namelists(*_pair(tmp_path, inp=inp))


def test_ordinary_land_use_keys_import_with_receipts(tmp_path):
    """``num_land_cat`` and ``fractional_seaice``, from the field report.

    Both are ordinary keys in a WPS/WRF-Runner-generated namelist -- gpuwm
    writes both itself in ``tools/write_hrrr_stock_wrf_namelist.py`` -- and
    both used to reach the unmapped-key refusal, which stopped a public
    HRRR hierarchy import on the importer's own map rather than on
    anything about the run.  Neither is silently dropped: one is validated
    against the land-use identity gpuwm actually builds, the other is
    reported with the branch each initialization route runs.
    """

    inp = INPUT_TEXT.replace(
        " cudt = 5, 0,",
        " cudt = 5, 0,\n num_land_cat = 21,\n fractional_seaice = 1,")
    toml_text, report = import_namelists(*_pair(tmp_path, inp=inp))

    # Neither key invents a TOML value.
    assert "num_land_cat" not in toml_text
    assert "fractional_seaice" not in toml_text
    fixed = {(f.section, f.key): f for f in report.fixed}
    assert fixed[("physics", "num_land_cat")].fixed_value == 21
    assert "LANDUSE.TBL" in fixed[("physics", "num_land_cat")].reason
    dropped = {(d.section, d.key): d for d in report.dropped}
    seaice = dropped[("physics", "fractional_seaice")]
    assert seaice.values == (1,)
    assert "FRACTIONAL" in seaice.reason.upper()
    rendered = report.format()
    assert "num_land_cat" in rendered and "fractional_seaice" in rendered

    # A category count describing geography gpuwm does not build refuses.
    usgs = INPUT_TEXT.replace(
        " cudt = 5, 0,", " cudt = 5, 0,\n num_land_cat = 24,")
    with pytest.raises(ValueError, match="num_land_cat"):
        import_namelists(*_pair(tmp_path, inp=usgs))
    nonsense = INPUT_TEXT.replace(
        " cudt = 5, 0,", " cudt = 5, 0,\n fractional_seaice = 2,")
    with pytest.raises(ValueError, match="fractional_seaice"):
        import_namelists(*_pair(tmp_path, inp=nonsense))

    # Negative control: a key that genuinely has no gpuwm counterpart is
    # still refused by name, so this widened nothing else.
    garbage = INPUT_TEXT.replace(
        " cudt = 5, 0,", " cudt = 5, 0,\n num_land_cat = 21,\n"
        " definitely_not_a_wrf_key = 3,")
    with pytest.raises(ValueError,
                       match="unmapped key.*definitely_not_a_wrf_key"):
        import_namelists(*_pair(tmp_path, inp=garbage))


def test_implicit_switches_come_from_the_shipped_profile_not_the_importer(
        tmp_path):
    """``moist_cq`` and ``top_lid`` have ONE authority now.

    WRF's namelist has no ``moist_cq`` key, and gpuwm's ``top_lid``
    default is deliberately not WRF's Registry default.  The importer used
    to answer both itself -- ``moist_cq = mp_physics > 0`` and WRF's open
    top -- while the HRRR root preparer and the domain wizard read the
    shipped physics profiles.  For every WSM6-family suite the two
    answers were opposite, so a public root prepared from a profile could
    never bind a public hierarchy imported from the same namelist.
    """

    from gpuwm.physics_compat import (
        MORRISON_PROFILE_ID, WSM6_PROFILE_ID, single_domain_runtime_switches,
    )

    wsm6 = INPUT_TEXT.replace(
        " mp_physics = 55, 55,", " mp_physics = 6, 6,").replace(
        " ra_lw_physics = 4, 4,", " ra_lw_physics = 0, 0,").replace(
        " ra_sw_physics = 4, 4,", " ra_sw_physics = 1, 1,").replace(
        " bl_pbl_physics = 11, 11,", " bl_pbl_physics = 1, 1,").replace(
        " cu_physics = 1, 0,", " cu_physics = 0, 0,").replace(
        " radt = 12, 3,", " radt = 1, 1,")
    toml_text, report = import_namelists(
        *_pair(tmp_path, inp=wsm6),
        # WSM6 no-radiation is a declared constant downward longwave;
        # the claim under test is implicit switches, not radiation.
        acknowledgements=(CONSTANT_DOWNWARD_LONGWAVE_ACK,))

    profile = single_domain_runtime_switches(WSM6_PROFILE_ID)
    assert profile["moist_cq"] is False and profile["top_lid"] is True
    assert "moist_cq = false" in toml_text
    assert "top_lid = true" in toml_text
    applied = {entry.key: entry for entry in report.defaults_applied}
    assert WSM6_PROFILE_ID in applied["top_lid"].reason

    # The Morrison-family profile states the other answer, and the importer
    # agrees with it for the profile's own suite.  The fixture selects
    # bl_pbl_physics = 1 explicitly: since the Shin-Hong port the base
    # pair's bl = 11 imports natively, and Morrison + Shin-Hong is NOT a
    # shipped profile -- so the base pair now exercises the RunConfig
    # fallback below instead of the Morrison row.
    morrison = single_domain_runtime_switches(MORRISON_PROFILE_ID)
    assert morrison["moist_cq"] is True and morrison["top_lid"] is False
    morrison_inp = INPUT_TEXT.replace(" bl_pbl_physics = 11, 11,",
                                      " bl_pbl_physics = 1, 1,")
    profile_toml, _ = import_namelists(*_pair(tmp_path, inp=morrison_inp))
    assert "moist_cq = true" in profile_toml
    assert "top_lid = false" in profile_toml

    # And the native Shin-Hong variant of the same suite matches no
    # shipped profile, so both implicit switches take gpuwm's RunConfig
    # defaults -- implicit_runtime_switches' documented fallback.
    shinhong_toml, _ = import_namelists(*_pair(tmp_path))
    assert "moist_cq = false" in shinhong_toml
    assert "top_lid = true" in shinhong_toml

    # An explicit namelist top_lid still wins over the profile: this is a
    # default, not an override.
    explicit = wsm6.replace(
        " use_theta_m = 0,",
        " use_theta_m = 0,\n top_lid = .false., .false.,")
    explicit_toml, _ = import_namelists(
        *_pair(tmp_path, inp=explicit),
        # Derived from the WSM6 no-radiation pair above, so it carries the
        # same constant-longwave declaration.
        acknowledgements=(CONSTANT_DOWNWARD_LONGWAVE_ACK,))
    assert "top_lid = false" in explicit_toml


def test_rejects_unsupported_theta_m(tmp_path):
    inp = INPUT_TEXT.replace(" use_theta_m = 0,", " use_theta_m = 1,")
    with pytest.raises(ValueError, match="use_theta_m"):
        import_namelists(*_pair(tmp_path, inp=inp))

    # Omission takes WRF's Registry default 1; it must not silently become
    # gpuwm's implemented dry-theta branch.
    inp = INPUT_TEXT.replace(" use_theta_m = 0,\n", "")
    with pytest.raises(ValueError, match="Registry default when omitted"):
        import_namelists(*_pair(tmp_path, inp=inp))


def test_nwp_diagnostics_maps_and_reaches_every_run_config(tmp_path):
    """The formerly pinned &time_control diagnostic knob now translates:
    = 1 lands on every per-domain RunConfig (the gate gpuwm/core/dycore.py
    reads for the UP_HELI_MAX kernel), is reported as Translated, and is
    emitted only when supplied."""
    inp = INPUT_TEXT.replace(
        " restart = .false.,",
        " restart = .false.,\n nwp_diagnostics = 1,")
    toml_text, report = import_namelists(*_pair(tmp_path, inp=inp))
    assert "nwp_diagnostics = 1" in toml_text
    exp = _load(tmp_path, toml_text)
    assert [dc.run.nwp_diagnostics for dc in exp.domains] == [1, 1]
    translated = {(t.section, t.key) for t in report.translated}
    assert ("time_control", "nwp_diagnostics") in translated
    fixed = {(f.section, f.key) for f in report.fixed}
    assert ("time_control", "nwp_diagnostics") not in fixed

    # Absent emits nothing (WRF Registry default 0 == RunConfig default),
    # keeping established imports byte-identical.
    toml_text, _ = import_namelists(*_pair(tmp_path))
    assert "nwp_diagnostics" not in toml_text
    assert _load(tmp_path, toml_text).root.run.nwp_diagnostics == 0

    # Values outside 0/1 refuse loudly.
    inp = INPUT_TEXT.replace(
        " restart = .false.,",
        " restart = .false.,\n nwp_diagnostics = 2,")
    with pytest.raises(ValueError, match="nwp_diagnostics"):
        import_namelists(*_pair(tmp_path, inp=inp))


def test_imports_feedback_as_tree_wide_experimental_switch(tmp_path):
    """Validation runs THROUGH the schema loader at import time."""
    inp = INPUT_TEXT.replace("feedback = 0", "feedback = 1")
    toml_text, _ = import_namelists(*_pair(tmp_path, inp=inp))
    assert "feedback = 1" in toml_text
    assert _load(tmp_path, toml_text).feedback == 1


def test_rejects_two_simultaneous_horizontal_mixing_schemes(tmp_path):
    inp = INPUT_TEXT.replace(" khdif = 0, 0,", " khdif = 75, 75,")
    with pytest.raises(ValueError, match="Smagorinsky.*khdif/kvdif"):
        import_namelists(*_pair(tmp_path, inp=inp))


def test_km_opt4_without_pbl_imports_complete_vertical_diffusion(tmp_path):
    inp = INPUT_TEXT.replace("bl_pbl_physics = 11, 11",
                             "bl_pbl_physics = 0, 0")
    toml_text, _report = import_namelists(*_pair(tmp_path, inp=inp))
    imported = _load(tmp_path, toml_text)
    assert all(domain.run.bl_pbl_physics == 0 for domain in imported.domains)
    assert all(domain.run.km_opt == 4 for domain in imported.domains)


def test_mix_full_fields_must_be_explicitly_true(tmp_path):
    omitted = INPUT_TEXT.replace(
        " mix_full_fields = .true., .true.,\n", "")
    # A missing must-set key is reported by the one-sweep missing-key
    # census (single report, not one traceback per key).
    with pytest.raises(ValueError,
                       match=r"missing required key\(s\)[\s\S]*"
                             r"mix_full_fields"):
        import_namelists(*_pair(tmp_path, inp=omitted))

    false = INPUT_TEXT.replace(
        " mix_full_fields = .true., .true.,",
        " mix_full_fields = .true., .false.,")
    with pytest.raises(ValueError, match="mix_full_fields"):
        import_namelists(*_pair(tmp_path, inp=false))

    toml_text, _ = import_namelists(*_pair(tmp_path, inp=INPUT_TEXT))
    assert "km_opt = 4" in toml_text


def test_omitted_keys_take_wrf_registry_defaults(tmp_path):
    """Review F2: omitted namelist keys resolve to their WRF v4.6.1
    Registry defaults, never gpuwm-convenient values."""
    # feedback omitted => Registry default 1 => an implicitly TWO-WAY
    # namelist selects the experimental path instead of silently
    # importing as one-way.  Keep smooth_option explicit here because its
    # independent Registry default 2 remains unsupported.
    inp = INPUT_TEXT.replace(" feedback = 0,\n", "")
    toml_text, _ = import_namelists(*_pair(tmp_path, inp=inp))
    assert _load(tmp_path, toml_text).feedback == 1
    # smooth_option omitted => Registry default 2 => rejected loudly
    inp = INPUT_TEXT.replace(" smooth_option = 0,\n", "")
    with pytest.raises(ValueError, match="smooth_option"):
        import_namelists(*_pair(tmp_path, inp=inp))
    # km_opt omitted => Registry default -1 = must-set => hard error
    # (surfaced by the one-sweep missing-key census)
    inp = INPUT_TEXT.replace(" km_opt = 4, 4,\n", "")
    with pytest.raises(ValueError, match=r"missing required key\(s\)"
                                         r"[\s\S]*km_opt"):
        import_namelists(*_pair(tmp_path, inp=inp))
    # p_top_requested omitted => Registry default 5000 Pa, visible in
    # the emitted TOML (not 0)
    inp = INPUT_TEXT.replace(" p_top_requested = 5000,\n", "")
    toml_text, _ = import_namelists(*_pair(tmp_path, inp=inp))
    assert "p_top = 5000.0" in toml_text
    # hybrid_opt/damp_opt omitted => Registry defaults 2/3
    inp = INPUT_TEXT.replace(" hybrid_opt = 2,\n", "").replace(
        " damp_opt = 3,\n", "")
    toml_text, _ = import_namelists(*_pair(tmp_path, inp=inp))
    assert "hybrid_opt = 2" in toml_text
    assert "damp_opt = 3" in toml_text


def test_grell_freitas_imports_natively_per_domain(tmp_path):
    """cu_physics = 3 is a scheme ArWen runs, not a substitution.

    The shinhong round-trip contract, on the cumulus axis: the selector
    reaches its domain as itself, the Grell-family keys ride along with
    their Registry defaults, cudt is dropped WITH the GF-specific reason
    (GF runs on the model step), and the emitted TOML pins
    cudt_minutes = 0 so the RunConfig validator's STEPCU=1 law holds on
    load.
    """
    # GF's vertical preflight wants nz >= 12 (the inversion-layer search
    # window); the base fixture's 8 mass levels serve KF but not GF.
    gf_inp = INPUT_TEXT.replace(
        " cu_physics = 1, 0,", " cu_physics = 3, 0,").replace(
        " cudt = 5, 0,",
        " cudt = 5, 0,\n ishallow = 1,\n clos_choice = 0,").replace(
        " e_vert = 9, 9,", " e_vert = 14, 14,").replace(
        " eta_levels = 1.0, 0.9, 0.8, 0.7, 0.6,\n"
        "              0.5, 0.4, 0.2, 0.0,",
        " eta_levels = 1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7,\n"
        "              0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0,")
    toml_text, report = import_namelists(*_pair(tmp_path, inp=gf_inp),
                                         name="gf3")
    out = tmp_path / "gf3.toml"
    out.write_text(toml_text)
    exp = load_experiment(out)
    assert [dc.run.cu_physics for dc in exp.domains] == [3, 0]
    assert not any(s.key == "cu_physics" for s in report.substitutions)
    assert exp.root.run.ishallow == 1
    assert exp.root.run.clos_choice == 0
    assert exp.domain(2).run.ishallow == 0
    assert exp.root.run.cudt_minutes == 0.0
    dropped = {entry.key: entry for entry in report.dropped}
    assert "cudt[1]" in dropped
    assert "no cudt cadence" in dropped["cudt[1]"].reason
    assert "cudt_minutes = 0.0" in toml_text
    assert "ishallow = 1" in toml_text

    # The Grell-family keys without a Grell scheme are dropped with the
    # driver's own reason, never silently honoured.
    stray = INPUT_TEXT.replace(
        " cudt = 5, 0,", " cudt = 5, 0,\n ishallow = 1,")
    stray_toml, stray_report = import_namelists(
        *_pair(tmp_path, inp=stray), name="stray")
    sdropped = {entry.key: entry for entry in stray_report.dropped}
    assert "clos_choice/ishallow" in sdropped
    assert "no Grell scheme selected" in \
        sdropped["clos_choice/ishallow"].reason
    assert "ishallow = 1" not in stray_toml


@pytest.mark.parametrize(
    ("line", "key"),
    [(" sf_sfclay_physics = 91, 91,\n", "sf_sfclay_physics"),
     (" cu_physics = 1, 0,\n", "cu_physics")])
def test_omitted_required_surface_and_cumulus_schemes_are_rejected(
        tmp_path, line, key):
    """WRF Registry -1 sentinels fatal; omission must not mean disabled."""
    assert line in INPUT_TEXT
    inp = INPUT_TEXT.replace(line, "")
    with pytest.raises(ValueError, match=rf"{key}.*must-set"):
        import_namelists(*_pair(tmp_path, inp=inp))


@pytest.mark.parametrize(
    "replacement",
    [" moist_adv_opt = 2, 2,",
     " moist_adv_opt = 1, 1,\n scalar_adv_opt = 2, 2,"])
def test_rejects_unimplemented_or_divergent_scalar_advection(
        tmp_path, replacement):
    """Only WRF's matched PD option is implemented for both scalar classes."""
    inp = INPUT_TEXT.replace(" moist_adv_opt = 1, 1,", replacement)
    with pytest.raises(ValueError, match="moist_adv_opt/scalar_adv_opt"):
        import_namelists(*_pair(tmp_path, inp=inp))


def test_gpuwm_supplied_values_are_recorded(tmp_path):
    """F2 last mile: values with no WRF Registry source are recorded as
    AppliedDefault entries -- never silent."""
    _, report = import_namelists(*_pair(tmp_path))
    applied = {a.key: a for a in report.defaults_applied}
    assert applied["ztop"].value == 20000.0
    assert "scaffold" in applied["ztop"].reason
    # the synthetic namelist omits time_step_sound (Registry 0 = auto)
    assert applied["time_step_sound"].value == 4
    assert "gpuwm-supplied values" in report.format()


def test_wrf_radt_zero_means_every_step(tmp_path):
    """Review F3: radt = 0 (WRF: radiation every step) must emit
    radt_minutes = 0.0 -- RunConfig's compat `radt` key would silently
    fall back to the 12-minute radt_minutes default."""
    inp = INPUT_TEXT.replace(" radt = 12, 3,", " radt = 0, 0,")
    toml_text, _ = import_namelists(*_pair(tmp_path, inp=inp))
    assert "radt_minutes = 0.0" in toml_text
    assert "radt = 0.0" not in toml_text
    out = tmp_path / "radt0.toml"
    out.write_text(toml_text)
    exp = load_experiment(out)
    for dc in exp.domains:
        assert dc.run.radt == 0.0 and dc.run.radt_minutes == 0.0


def test_root_sponge_spec_exp_imports_legally(tmp_path):
    """Review F4: a nonzero spec_exp lands on the ROOT [[domain]] (the
    sponge lives in the specified branch, module_bc_em.F:1320); children
    keep spec_exp = 0 and the import must not over-reject."""
    inp = INPUT_TEXT.replace(" spec_bdy_width = 5,",
                             " spec_bdy_width = 5,\n spec_exp = 0.33,")
    toml_text, _ = import_namelists(*_pair(tmp_path, inp=inp))
    out = tmp_path / "sponge.toml"
    out.write_text(toml_text)
    exp = load_experiment(out)
    assert exp.root.run.spec_exp == 0.33
    assert exp.domain(2).run.spec_exp == 0.0


def test_rejects_repetition_false_input_from_file(tmp_path):
    """Shadow S2's silent false->true hazard: `2*.false.` now expands to
    real booleans and trips the input_from_file rejection instead of
    truth-testing a residual string."""
    inp = INPUT_TEXT.replace("input_from_file = .true., .true.,",
                             "input_from_file = 2*.false.,")
    with pytest.raises(ValueError, match="input_from_file"):
        import_namelists(*_pair(tmp_path, inp=inp))


def test_rejects_non_logical_boolean_columns(tmp_path):
    inp = INPUT_TEXT.replace("specified = .true., .false.,",
                             "specified = 1, 0,")
    with pytest.raises(ValueError, match="Fortran logical"):
        import_namelists(*_pair(tmp_path, inp=inp))


def test_rejects_implausible_max_dom(tmp_path):
    """Shadow S4: max_dom is range-checked before any per-domain array
    expansion."""
    inp = INPUT_TEXT.replace(" max_dom = 2,", " max_dom = 0,")
    with pytest.raises(ValueError, match="max_dom"):
        import_namelists(*_pair(tmp_path, inp=inp))
    inp = INPUT_TEXT.replace(" max_dom = 2,", " max_dom = 5000,")
    with pytest.raises(ValueError, match=r"\[1, 21\]"):
        import_namelists(*_pair(tmp_path, inp=inp))


# ---------------------------------------------------------------------------
# Knob-parity lane: RunConfig-honored keys translate, pinned keys fix,
# and the report carries the three explicit sections.
# ---------------------------------------------------------------------------

def _import_with(tmp_path, extra_dynamics="", extra_physics="",
                 extra_input="", inp_filter=None):
    """INPUT_TEXT with lines appended inside &dynamics / &physics.

    ``inp_filter`` rewrites existing columns first (e.g. turning the
    second domain into an LES child) so the appended lines land in a
    namelist that actually selects the scheme under test.
    """
    inp = INPUT_TEXT
    if inp_filter is not None:
        inp = inp_filter(inp)
    if extra_dynamics:
        inp = inp.replace(" hybrid_opt = 2,",
                          " hybrid_opt = 2,\n" + extra_dynamics)
    if extra_physics:
        inp = inp.replace(" mp_physics = 55, 55,",
                          " mp_physics = 55, 55,\n" + extra_physics)
    if extra_input:
        inp += extra_input
    return import_namelists(*_pair(tmp_path, inp=inp))


def _load(tmp_path, toml_text, name="knob.toml"):
    out = tmp_path / name
    out.write_text(toml_text)
    return load_experiment(out)


def test_dynamics_runconfig_knobs_reach_the_run_config(tmp_path):
    """Booby-trap reach test: distinctive c_s/diff_6th_thresh values must
    land on every per-domain RunConfig (the constants gpuwm/core/dycore.py
    reads at :632 and :802)."""
    toml_text, report = _import_with(
        tmp_path,
        extra_dynamics=" c_s = 0.18, 0.18,\n diff_6th_thresh = 0.25,\n")
    exp = _load(tmp_path, toml_text)
    assert [dc.run.c_s for dc in exp.domains] == [0.18, 0.18]
    assert [dc.run.diff_6th_thresh for dc in exp.domains] == [0.25, 0.25]
    translated = {(t.section, t.key) for t in report.translated}
    assert ("dynamics", "c_s") in translated
    assert ("dynamics", "diff_6th_thresh") in translated


def test_dynamics_runconfig_knobs_absent_emit_nothing(tmp_path):
    """Absent keys emit nothing (Registry default == RunConfig default),
    keeping established imports byte-identical."""
    toml_text, _ = import_namelists(*_pair(tmp_path))
    assert "c_s" not in toml_text.replace("diff_6th", "")
    assert "diff_6th_thresh" not in toml_text
    exp = _load(tmp_path, toml_text)
    assert exp.root.run.c_s == 0.25
    assert exp.root.run.diff_6th_thresh == 0.10


def test_moist_mix6_off_imports_per_domain_and_absent_emits_nothing(
        tmp_path):
    """WRF's own &dynamics moist_mix6_off (Registry.EM_COMMON:2889,
    max_domains, default .false.; divergence-ledger entry L4) maps 1:1
    onto the RunConfig field of the same spelling.  A scalar assignment
    changes d01 only -- epssm's Fortran-assignment rule, the unassigned
    tail keeps the Registry default -- and an absent key emits nothing,
    keeping established imports byte-identical."""
    toml_text, report = _import_with(
        tmp_path, extra_dynamics=" moist_mix6_off = .true.,\n")
    exp = _load(tmp_path, toml_text)
    assert [dc.run.moist_mix6_off for dc in exp.domains] == [True, False]
    translated = {(t.section, t.key) for t in report.translated}
    assert ("dynamics", "moist_mix6_off") in translated
    plain, _ = import_namelists(*_pair(tmp_path))
    assert "moist_mix6_off" not in plain
    exp_plain = _load(tmp_path, plain, name="plain.toml")
    assert [dc.run.moist_mix6_off for dc in exp_plain.domains] == [
        False, False]
    with pytest.raises(ValueError, match="Fortran logical"):
        _import_with(tmp_path, extra_dynamics=" moist_mix6_off = 1, 1,\n")


def test_physics_runconfig_knobs_reach_the_run_config(tmp_path):
    """Every newly mapped &physics knob lands on RunConfig with its
    distinctive (non-default) value -- no dead knobs."""
    toml_text, report = _import_with(
        tmp_path,
        extra_physics=(" no_mp_heating = 1,\n"
                       " mp_tend_lim = 7.5,\n"
                       " ysu_topdown_pblmix = 0,\n"
                       " isftcflx = 1,\n"
                       " iz0tlnd = 2,\n"
                       " usemonalb = .true.,\n"
                       " rdlai2d = .true.,\n"
                       " opt_thcnd = 2,\n"))
    exp = _load(tmp_path, toml_text)
    for dc in exp.domains:
        assert dc.run.no_mp_heating == 1
        assert dc.run.mp_tend_lim == 7.5
        assert dc.run.ysu_topdown_pblmix == 0
        assert dc.run.isftcflx == 1
        assert dc.run.iz0tlnd == 2
        assert dc.run.usemonalb is True
        assert dc.run.rdlai2d is True
        assert dc.run.opt_thcnd == 2
    translated = {(t.section, t.key) for t in report.translated}
    for key in ("no_mp_heating", "mp_tend_lim", "ysu_topdown_pblmix",
                "isftcflx", "iz0tlnd", "usemonalb", "rdlai2d",
                "opt_thcnd"):
        assert ("physics", key) in translated, key


@pytest.mark.parametrize(("line", "match"), [
    (" no_mp_heating = 2,\n", "no_mp_heating"),
    (" mp_tend_lim = -1.0,\n", "mp_tend_lim"),
    (" ysu_topdown_pblmix = 2,\n", "ysu_topdown_pblmix"),
    (" isftcflx = 5,\n", "isftcflx"),
    (" iz0tlnd = 3,\n", "iz0tlnd"),
    (" opt_thcnd = 3,\n", "opt_thcnd"),
    (" usemonalb = 1,\n", "Fortran logical"),
])
def test_new_physics_knob_invalid_values_are_rejected(tmp_path, line,
                                                      match):
    with pytest.raises(ValueError, match=match):
        _import_with(tmp_path, extra_physics=line)


def test_pinned_dynamics_orders_fix_and_refuse(tmp_path):
    """rk_ord / advection orders / momentum_adv_opt: the WRF defaults the
    dycore hardcodes import as fixed-by-ArWen; any other value refuses."""
    _, report = _import_with(
        tmp_path,
        extra_dynamics=(" rk_ord = 3,\n h_mom_adv_order = 5, 5,\n"
                        " v_mom_adv_order = 3, 3,\n"
                        " v_sca_adv_order = 3, 3,\n"
                        " momentum_adv_opt = 1, 1,\n"))
    fixed = {(f.section, f.key): f for f in report.fixed}
    for key, pin in (("rk_ord", 3), ("h_mom_adv_order", 5),
                     ("v_mom_adv_order", 3), ("v_sca_adv_order", 3),
                     ("momentum_adv_opt", 1)):
        entry = fixed[("dynamics", key)]
        assert entry.fixed_value == pin
        assert entry.reason
    for line, match in ((" rk_ord = 2,\n", "rk_ord"),
                        (" h_mom_adv_order = 3, 3,\n", "h_mom_adv_order"),
                        (" v_sca_adv_order = 5, 5,\n", "v_sca_adv_order"),
                        (" momentum_adv_opt = 3, 3,\n",
                         "momentum_adv_opt")):
        with pytest.raises(ValueError, match=match):
            _import_with(tmp_path, extra_dynamics=line)


def test_h_sca_adv_order_nondefault_is_refused(tmp_path):
    """gpuwm's h_sca_adv_order feeds only the geopotential advection, so
    a non-Registry value would not mean what WRF means -- refuse."""
    with pytest.raises(ValueError, match="h_sca_adv_order"):
        _import_with(tmp_path,
                     extra_dynamics=" h_sca_adv_order = 2,\n")


def test_hypsometric_opt_imports_from_domains_where_wrf_declares_it(
        tmp_path):
    """The mirror of the emitter defect, on the reading side.

    WRF v4.6.1 declares hypsometric_opt in &domains as a scalar
    (Registry.EM_COMMON:2283, ``namelist,domains``, nentries 1;
    run/README.namelist documents it inside the &domains block).  The
    importer read it from &dynamics, so a namelist shaped the way a real
    WRF namelist must be shaped -- the only kind wrf.exe can read --
    imported at the Registry default and quietly lost the setting.
    """
    inp = INPUT_TEXT.replace(" p_top_requested = 5000,",
                             " p_top_requested = 5000,\n"
                             " hypsometric_opt = 1,")
    assert "hypsometric_opt" in inp
    toml_text, _report = import_namelists(*_pair(tmp_path, inp=inp))
    out = tmp_path / "hypsometric.toml"
    out.write_text(toml_text)
    assert load_experiment(out).root.run.hypsometric_opt == 1

    # Omitted, it still binds to the ratified WRF Registry default.
    default_text, _ = import_namelists(*_pair(tmp_path))
    default = tmp_path / "default.toml"
    default.write_text(default_text)
    assert load_experiment(default).root.run.hypsometric_opt == 2


def test_hypsometric_opt_in_dynamics_is_refused_by_name(tmp_path):
    """The old gpuwm-only spelling is refused, not tolerated.

    Every other unmapped key gets the generic "extend the ratified map"
    refusal, which is true here but does not say where the key belongs.
    A namelist carrying hypsometric_opt in &dynamics is one wrf.exe
    cannot read at all, so this one names the section and the repair.
    """
    inp = INPUT_TEXT.replace(" hybrid_opt = 2,",
                             " hybrid_opt = 2,\n hypsometric_opt = 1,")
    with pytest.raises(ValueError, match="hypsometric_opt") as refusal:
        import_namelists(*_pair(tmp_path, inp=inp))
    message = str(refusal.value)
    assert "&domains" in message
    assert "Move the key to &domains" in message


def test_tke_adv_opt_drops_as_inert(tmp_path):
    _, report = _import_with(tmp_path,
                             extra_dynamics=" tke_adv_opt = 1, 1,\n")
    dropped = {(d.section, d.key): d for d in report.dropped}
    assert "prognostic-TKE" in dropped[("dynamics", "tke_adv_opt")].reason


@pytest.mark.parametrize("key", [
    "swint_opt", "gwd_opt", "sf_lake_physics", "shcu_physics",
    "topo_shading", "slope_rad", "kf_edrates", "flag_sm_adj",
    "sst_update", "sst_skin", "tmn_update",
])
def test_pinned_physics_neutrals_fix_at_zero_and_refuse_nonzero(
        tmp_path, key):
    _, report = _import_with(tmp_path, extra_physics=f" {key} = 0,\n")
    fixed = {(f.section, f.key): f for f in report.fixed}
    assert fixed[("physics", key)].fixed_value == 0
    assert fixed[("physics", key)].reason
    with pytest.raises(ValueError, match=key):
        _import_with(tmp_path, extra_physics=f" {key} = 1,\n")


def test_use_mp_re_fixes_at_one_and_refuses_zero(tmp_path):
    _, report = _import_with(tmp_path, extra_physics=" use_mp_re = 1,\n")
    fixed = {(f.section, f.key): f for f in report.fixed}
    assert fixed[("physics", "use_mp_re")].fixed_value == 1
    with pytest.raises(ValueError, match="use_mp_re"):
        _import_with(tmp_path, extra_physics=" use_mp_re = 0,\n")


def test_isfflx_zero_is_emitted_and_reaches_every_domain(tmp_path):
    text, _ = _import_with(tmp_path, extra_physics=" isfflx = 0,\n")
    assert "isfflx = 0" in text
    exp = _load(tmp_path, text, "isfflx-zero.toml")
    assert {domain.run.isfflx for domain in exp.domains} == {0}


def test_cu_rad_feedback_false_fixes_true_refuses(tmp_path):
    _, report = _import_with(
        tmp_path, extra_physics=" cu_rad_feedback = .false., .false.,\n")
    fixed = {(f.section, f.key) for f in report.fixed}
    assert ("physics", "cu_rad_feedback") in fixed
    with pytest.raises(ValueError, match="cu_rad_feedback"):
        _import_with(tmp_path,
                     extra_physics=" cu_rad_feedback = .true., .true.,\n")


def test_mynn_identity_keys_fix_at_identity_and_refuse_others(tmp_path):
    _, report = _import_with(
        tmp_path,
        extra_physics=(" bl_mynn_mixlength = 1,\n icloud_bl = 1,\n"
                       " bl_mynn_tkeadvect = .false., .false.,\n"
                       " bl_mynn_closure = 2.6,\n"))
    fixed = {(f.section, f.key): f for f in report.fixed}
    assert fixed[("physics", "bl_mynn_mixlength")].fixed_value == 1
    assert fixed[("physics", "bl_mynn_closure")].fixed_value == 2.6
    for line, match in ((" bl_mynn_mixlength = 2,\n", "bl_mynn_mixlength"),
                        (" icloud_bl = 0,\n", "icloud_bl"),
                        (" bl_mynn_tkeadvect = .true., .true.,\n",
                         "bl_mynn_tkeadvect")):
        with pytest.raises(ValueError, match=match):
            _import_with(tmp_path, extra_physics=line)


def test_noah_mp_section_identity_values_fix_others_refuse(tmp_path):
    section = ("&noah_mp\n dveg = 4,\n opt_run = 3,\n opt_sfc = 1,\n"
               " soiltstep = 0,\n/\n")
    _, report = _import_with(tmp_path, extra_input=section)
    fixed = {(f.section, f.key): f for f in report.fixed}
    assert fixed[("noah_mp", "dveg")].fixed_value == 4
    assert fixed[("noah_mp", "opt_run")].fixed_value == 3
    assert "identity" in fixed[("noah_mp", "dveg")].reason
    with pytest.raises(ValueError, match="dveg"):
        _import_with(tmp_path, extra_input="&noah_mp\n dveg = 5,\n/\n")
    with pytest.raises(ValueError, match="unmapped key"):
        _import_with(tmp_path,
                     extra_input="&noah_mp\n not_a_noahmp_key = 1,\n/\n")


def test_stoch_section_off_drops_seeds_and_refuses_active_schemes(
        tmp_path):
    section = ("&stoch\n spp = 0,\n spp_pbl = 0,\n iseed_spp_lsm = 123,\n"
               " nens = 1,\n/\n")
    _, report = _import_with(tmp_path, extra_input=section)
    fixed = {(f.section, f.key) for f in report.fixed}
    dropped = {(d.section, d.key) for d in report.dropped}
    assert ("stoch", "spp") in fixed
    assert ("stoch", "spp_pbl") in fixed
    assert ("stoch", "iseed_spp_lsm") in dropped
    assert ("stoch", "nens") in dropped
    with pytest.raises(ValueError, match="stochastic"):
        _import_with(tmp_path, extra_input="&stoch\n spp_lsm = 1,\n/\n")


def test_fdda_active_nudging_refuses_disabled_drops(tmp_path):
    with pytest.raises(ValueError, match="nudging"):
        _import_with(tmp_path,
                     extra_input="&fdda\n grid_fdda = 1, 1,\n/\n")
    _, report = _import_with(
        tmp_path,
        extra_input="&fdda\n grid_fdda = 0, 0,\n gfdda_inname = 'x',\n/\n")
    dropped = {(d.section, d.key) for d in report.dropped}
    assert ("fdda", "grid_fdda") in dropped
    assert ("fdda", "gfdda_inname") in dropped


def test_time_control_stream_and_logging_keys_drop(tmp_path):
    inp = INPUT_TEXT.replace(
        " restart = .false.,",
        " restart = .false.,\n debug_level = 100,\n"
        " adjust_output_times = .true.,\n"
        " auxhist2_interval = 10, 10,\n"
        " auxinput4_inname = 'wrflowinp_d<domain>',\n"
        " iofields_filename = 'fields.txt', 'fields.txt',\n")
    _, report = import_namelists(*_pair(tmp_path, inp=inp))
    dropped = {(d.section, d.key) for d in report.dropped}
    for key in ("debug_level", "adjust_output_times", "auxhist2_interval",
                "auxinput4_inname", "iofields_filename"):
        assert ("time_control", key) in dropped, key


def test_domains_decomposition_keys_drop(tmp_path):
    inp = INPUT_TEXT.replace(" time_step = 60,",
                             " time_step = 60,\n numtiles = 4,\n"
                             " nproc_x = 2,\n nproc_y = 3,")
    _, report = import_namelists(*_pair(tmp_path, inp=inp))
    dropped = {(d.section, d.key) for d in report.dropped}
    for key in ("numtiles", "nproc_x", "nproc_y"):
        assert ("domains", key) in dropped, key


def test_auto_level_keys_inert_with_eta_refused_without(tmp_path):
    withkeys = INPUT_TEXT.replace(
        " p_top_requested = 5000,",
        " p_top_requested = 5000,\n auto_levels_opt = 2,\n"
        " dzstretch_s = 1.3,")
    _, report = import_namelists(*_pair(tmp_path, inp=withkeys))
    dropped = {(d.section, d.key): d for d in report.dropped}
    assert "inert" in dropped[("domains", "auto_levels_opt")].reason
    # without explicit eta_levels the generator keys would select level
    # generation gpuwm cannot perform
    noeta = withkeys.replace(
        """ eta_levels = 1.0, 0.9, 0.8, 0.7, 0.6,
              0.5, 0.4, 0.2, 0.0,\n""", "")
    with pytest.raises(ValueError, match="automatic eta-level"):
        import_namelists(*_pair(tmp_path, inp=noeta))


def test_nssl_parameters_pin_registry_defaults_under_mp18(tmp_path):
    base = INPUT_TEXT.replace("mp_physics = 55, 55", "mp_physics = 18, 18")
    ok = base.replace(" mp_physics = 18, 18,",
                      " mp_physics = 18, 18,\n nssl_alphah = 0.0,\n"
                      " nssl_rho_qhl = 900.0,")
    _, report = import_namelists(*_pair(tmp_path, inp=ok))
    fixed = {(f.section, f.key): f for f in report.fixed}
    assert fixed[("physics", "nssl_alphah")].fixed_value == 0.0
    bad = base.replace(" mp_physics = 18, 18,",
                       " mp_physics = 18, 18,\n nssl_cccn = 1.0e9,")
    with pytest.raises(ValueError, match="nssl_cccn"):
        import_namelists(*_pair(tmp_path, inp=bad))
    # under any other scheme the parameters are inert in WRF too -> drop
    inert = INPUT_TEXT.replace(" mp_physics = 55, 55,",
                               " mp_physics = 55, 55,\n nssl_cccn = 1.0e9,")
    _, report = import_namelists(*_pair(tmp_path, inp=inert))
    dropped = {(d.section, d.key): d for d in report.dropped}
    assert "inert" in dropped[("physics", "nssl_cccn")].reason


def test_report_carries_three_explicit_sections(tmp_path):
    _, report = _import_with(tmp_path,
                             extra_physics=" swint_opt = 0,\n")
    formatted = report.format()
    assert "Translated (namelist -> experiment TOML):" in formatted
    assert "Fixed by ArWen (validated against the only implemented " \
           "value):" in formatted
    assert "Not implemented (namelist keys consumed without a gpuwm " \
           "counterpart):" in formatted
    # a fixed entry shows the pinned value and the why
    assert "[fixed: 0]" in formatted
    # translated keys are grouped per namelist section
    assert "&domains:" in formatted
    translated = {(t.section, t.key) for t in report.translated}
    assert ("domains", "time_step") in translated
    assert ("physics", "mp_physics") in translated
    # no key appears in two classes
    fixed_keys = {(f.section, f.key) for f in report.fixed}
    dropped_keys = {(d.section, d.key) for d in report.dropped}
    assert not translated & fixed_keys
    assert not translated & dropped_keys
    assert not fixed_keys & dropped_keys


# ---------------------------------------------------------------------------
# Bundle round-trip gate
# ---------------------------------------------------------------------------

@requires_bundle
def test_original_bundle_namelist_rejects_unsupported_modes():
    """The published original is not silently rewritten to gpuwm scope.

    Since the nwp_diagnostics unpin (STEP17) the original's remaining
    unsupported mode is use_theta_m: the campaign namelist omits it, WRF's
    Registry default is 1, and gpuwm implements the dry-theta branch only.
    """
    with pytest.raises(ValueError, match="use_theta_m"):
        import_namelists(BUNDLE_WPS, BUNDLE_INPUT, name="real74_4dom")


#: ``[experiment]`` settings the committed flagship config carries that the
#: importer cannot derive, because they have no WRF namelist counterpart at
#: all.  ``column_chunk`` is a gpuwm RRTMGP throughput knob; the case sets
#: 6250 against the library default 3125 and
#: ``tests/test_preflight.py::test_estimate_4dom_golden_pins`` is what pins
#: the footprint that buys.  The point of naming them HERE is that the
#: round-trip below stays byte-for-byte on everything else: a hand edit to
#: the committed file that is not one of these still fails.
HAND_DECLARED_EXPERIMENT_SETTINGS = ("column_chunk",)


def _strip_hand_declared_settings(text: str) -> tuple[str, dict[str, str]]:
    """Remove each declared key and the comment block written above it."""
    kept: list[str] = []
    pending: list[str] = []
    found: dict[str, str] = {}
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            pending.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in HAND_DECLARED_EXPERIMENT_SETTINGS:
            assert key not in found, f"{key} declared twice"
            found[key] = stripped.split("=", 1)[1].strip()
            pending.clear()          # the block above it goes with it
            continue
        kept.extend(pending)
        pending.clear()
        kept.append(line)
    kept.extend(pending)
    return "".join(kept), found


@requires_bundle
def test_effective_bundle_round_trip_reproduces_committed_toml(tmp_path):
    """The explicit effective namelist reproduces the committed config."""
    effective = _effective_bundle_input(tmp_path)
    toml_text, report = import_namelists(BUNDLE_WPS, effective,
                                         name="real74_4dom")
    committed = (REPO / "configs" / "real74_4dom.toml").read_text()
    # The committed flagship config is a -v1 receipt of the campaign as it
    # ran; the importer now emits the -v2 WRF-matching snow-coupling token.
    # The receipt is never relabeled, so the round trip must match byte for
    # byte EXCEPT that single token line.
    committed_line = 'wrf_rrtmg_compatibility = "wrf-rrtmg-4-4-to-rte-rrtmgp-v1"'
    emitted_line = 'wrf_rrtmg_compatibility = "wrf-rrtmg-4-4-to-rte-rrtmgp-v2"'
    assert committed_line in committed
    assert emitted_line in toml_text
    harmonized = committed.replace(committed_line, emitted_line)
    # Second enumerated receipt-vs-importer divergence, same shape as the
    # token above: the campaign ran the ratified Shin-Hong -> YSU
    # substitution and its receipt records bl_pbl_physics = 1; the importer
    # imports 11 natively since the Shin-Hong port.  The receipt is never
    # rewritten, so the comparison harmonizes exactly the lines that one
    # admission moves: the selector, its substitution-header comment, and
    # the two profile-implicit switches -- a Morrison + Shin-Hong suite is
    # not a shipped profile, so implicit_runtime_switches answers with
    # RunConfig defaults (its documented fallback) instead of the Morrison
    # profile's moist_cq/top_lid.
    committed_bl = "\nbl_pbl_physics = 1\n"
    emitted_bl = "\nbl_pbl_physics = 11\n"
    assert committed_bl in harmonized
    assert emitted_bl in "\n" + toml_text
    harmonized = harmonized.replace(committed_bl, emitted_bl)
    header_bl = ("\n#   bl_pbl_physics 11 (Shin-Hong) -> "
                 "bl_pbl_physics 1 (YSU)")
    assert header_bl in harmonized
    assert "Shin-Hong" not in toml_text
    harmonized = harmonized.replace(header_bl, "")
    for receipt_line, imported_line in (("\ntop_lid = false\n",
                                         "\ntop_lid = true\n"),
                                        ("\nmoist_cq = true\n",
                                         "\nmoist_cq = false\n")):
        assert receipt_line in harmonized
        assert imported_line in toml_text
        harmonized = harmonized.replace(receipt_line, imported_line)
    # The committed flagship config = importer output + the hand-declared
    # [case_data] table (declared inputs are not derivable from namelists)
    # + the hand-declared [experiment] settings named above (gpuwm-only knobs
    # with no namelist counterpart).  Both sets are enumerated, so this stays
    # a byte-for-byte reproduction test of everything the importer owns.
    generated, hand = _strip_hand_declared_settings(harmonized)
    assert set(hand) == set(HAND_DECLARED_EXPERIMENT_SETTINGS), hand
    assert hand["column_chunk"] == "6250"
    assert generated.startswith(toml_text)
    appended = generated[len(toml_text):]
    assert appended.lstrip("\r\n").startswith("[case_data]")
    # and the importer, which knows nothing about the knob, emitted none of it
    assert "column_chunk" not in toml_text
    subs = {(s.key, s.wrf_value, s.gpuwm_key, s.gpuwm_value)
            for s in report.substitutions}
    # bl_pbl 11 left this set when the Shin-Hong port admitted the scheme;
    # the two remaining rows are the complete ratified-substitution
    # inventory.
    assert subs == {
        ("mp_physics", 55, "mp_physics", 10),
        ("ra_lw_physics/ra_sw_physics", 4, "ra_physics", 4),
    }
    # the bundle's hand-typed 333.333333 d04 dx cross-checks against the
    # exact 1000/3 m chain and is dropped, not copied
    assert "333.33" not in toml_text
    dropped = {(d.section, d.key) for d in report.dropped}
    assert ("domains", "dx (children)") in dropped


@requires_bundle
def test_effective_bundle_import_matches_loaded_committed_experiment(tmp_path):
    effective = _effective_bundle_input(tmp_path)
    toml_text, _ = import_namelists(BUNDLE_WPS, effective,
                                    name="real74_4dom")
    regenerated = tmp_path / "regen.toml"
    regenerated.write_text(toml_text)
    from gpuwm.case_data import load_experiment_case
    # The committed config carries the -v1 receipt token (campaign as run);
    # a fresh import emits -v2.
    committed_text = (REPO / "configs" / "real74_4dom.toml").read_text()
    harmonized = committed_text.replace(
        'wrf_rrtmg_compatibility = "wrf-rrtmg-4-4-to-rte-rrtmgp-v1"',
        'wrf_rrtmg_compatibility = "wrf-rrtmg-4-4-to-rte-rrtmgp-v2"')
    assert harmonized != committed_text
    # Same enumerated importer-behaviour changes as the byte round-trip
    # test above: the campaign receipt records the Shin-Hong -> YSU
    # substitution; a fresh import is native bl_pbl_physics = 11, and a
    # Morrison + Shin-Hong suite matches no shipped profile, so the
    # implicit moist_cq/top_lid answers fall back to RunConfig defaults.
    with_bl = (harmonized
               .replace("\nbl_pbl_physics = 1\n", "\nbl_pbl_physics = 11\n")
               .replace("\ntop_lid = false\n", "\ntop_lid = true\n")
               .replace("\nmoist_cq = true\n", "\nmoist_cq = false\n"))
    assert with_bl != harmonized
    harmonized = with_bl
    harmonized_path = tmp_path / "committed_harmonized.toml"
    harmonized_path.write_text(harmonized)
    # The committed config declares its source-orography pins as
    # repo-relative [case_data] assets (configs/real74/*.nc); mirror that
    # directory beside the relocated copy so the declarations resolve from
    # tmp_path exactly as they do from configs/.  Bytes are the committed
    # assets themselves; no assertion below changes.
    import shutil
    shutil.copytree(REPO / "configs" / "real74", tmp_path / "real74")
    imported = load_experiment(regenerated)
    committed = load_experiment_case(harmonized_path)[0]
    # The hand-declared knob is the ONLY remaining difference, which is what
    # makes it a declaration rather than a divergence: the importer emits the
    # library default and the case overrides it by hand.
    assert imported.column_chunk == DEFAULT_COLUMN_CHUNK
    assert committed.column_chunk == 6250
    assert dataclasses.replace(
        imported, column_chunk=committed.column_chunk) == committed


# ---------------------------------------------------------------------------
# CLI subcommand
# ---------------------------------------------------------------------------

def test_cli_import_namelist_writes_output_and_report(tmp_path, capsys):
    wps_path, inp_path = _pair(tmp_path)
    out = tmp_path / "resolved.toml"
    assert cli.main(["import-namelist", str(wps_path), str(inp_path),
                     "--output", str(out), "--name", "synth99"]) == 0
    captured = capsys.readouterr().out
    assert "Physics substitutions" in captured
    assert "Morrison 2-moment" in captured
    assert out.exists()
    assert not out.with_suffix(out.suffix + ".tmp").exists()  # atomic
    exp = load_experiment(out)
    assert exp.name == "synth99"
    assert exp.dt_exact(2) == Fraction(20)


def test_cli_import_namelist_refuses_output_aliasing(tmp_path):
    """Shadow S4: --output resolving to an input namelist must refuse
    rather than destroy the source after reading it."""
    wps_path, inp_path = _pair(tmp_path)
    rc = cli.main(["import-namelist", str(wps_path), str(inp_path),
                   "--output", str(inp_path)])
    assert rc == 2  # uniform CLI refusal boundary: message, no traceback
    assert inp_path.read_text() == INPUT_TEXT  # source intact


def test_cli_import_namelist_unported_selector_refuses_without_traceback(
        tmp_path, capsys):
    """A validate_run_config NotImplementedError (ra_lw_physics=1, the
    unported WRF RRTM longwave) must reach the operator as the uniform CLI
    refusal -- message on stderr, exit 2 -- not as a leaked traceback.

    Observed live during the 2026-07-30 WRF-Runner interop verification: a
    runner-generated namelist pair carrying ra_lw_physics=1 crashed
    ``gpuwm import-namelist`` with a stack trace where every neighbouring
    refusal (unmapped keys, FDDA, mosaic) printed one actionable line."""
    unported = INPUT_TEXT.replace(
        " ra_lw_physics = 4, 4,", " ra_lw_physics = 1, 1,").replace(
        " ra_sw_physics = 4, 4,", " ra_sw_physics = 1, 1,")
    wps_path, inp_path = _pair(tmp_path, inp=unported)
    out = tmp_path / "resolved.toml"
    rc = cli.main(["import-namelist", str(wps_path), str(inp_path),
                   "--output", str(out)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "gpuwm import-namelist: ra_lw_physics=1" in err
    assert "Traceback" not in err
    assert not out.exists()  # refusal precedes any output publication


_ONE_DOMAIN_WPS = """\
&share
 wrf_core = "ARW",
 max_dom = {max_dom},
 interval_seconds = 10800,
/
&geogrid
 parent_id         = 1,
 parent_grid_ratio = 1,
 i_parent_start    = 1,
 j_parent_start    = 1,
 e_we              = 100,
 e_sn              = 80,
 geog_data_res     = "default",
 dx = 12000,
 dy = 12000,
 map_proj = "lambert",
 ref_lat   = 40.0,
 ref_lon   = -96.0,
 truelat1  = 30.0,
 truelat2  = 50.0,
 stand_lon = -96.0,
/
"""


def test_max_dom_beyond_the_declared_arrays_is_refused_by_name(tmp_path):
    """&share/max_dom and the &geogrid arrays are declared independently,
    so nothing but a length check stops them disagreeing.

    On 1.4.0 this was `IndexError: list index out of range` -- and since
    gfs_direct.main catches only (ValueError, OSError), the IndexError
    escaped the door that turns refusals into sentences and reached the
    reader as a traceback at rc 1.
    """
    wps = tmp_path / "short.namelist.wps"
    wps.write_text(_ONE_DOMAIN_WPS.format(max_dom=2), encoding="utf-8")
    with pytest.raises(ValueError) as raised:
        grids_from_wps_namelist(wps)
    message = str(raised.value)
    assert "max_dom = 2" in message
    # every short array named at once, not one traceback per edit
    for key in ("e_we", "e_sn", "parent_id", "parent_grid_ratio",
                "i_parent_start", "j_parent_start"):
        assert key in message
    assert "set max_dom to the number of domains" in message


def test_max_dom_matching_the_declared_arrays_still_loads(tmp_path):
    """The negative control: the check refuses a disagreement, not a
    single-domain namelist."""
    wps = tmp_path / "ok.namelist.wps"
    wps.write_text(_ONE_DOMAIN_WPS.format(max_dom=1), encoding="utf-8")
    grids = grids_from_wps_namelist(wps)
    assert len(grids) == 1
    assert grids[0].e_we == 100 and grids[0].e_sn == 80


# --------------------------------------------------------------------
# The km_opt=2/3 turbulence parameter row.
#
# Every key below is `max_domains` in Registry.EM_COMMON and sits in
# gpuwm.experiment._DOMAIN_RUN_OVERRIDES.  Before the row was read, an
# LES child had NO expressible pair of inputs: omitting c_k resolved
# gpuwm's RunConfig default 0.15 against an LES TOML's em_les 0.10 and
# the prepared cache refused the forecast on `run.c_k`, while supplying
# c_k in the namelist hit the unmapped-key refusal.  Both directions
# were closed, and with them the whole km_opt=2 hierarchy gate.
# --------------------------------------------------------------------

def _les_child(inp: str) -> str:
    """The base pair with an LES (km_opt=3, PBL-off) nested second domain.

    km_opt=3 and not 2 because validate_run_config refuses km_opt=2 on a
    NEST child outright (no nested LES domain has been run), and that
    refusal stands -- this fixture exercises the namelist bridge, not the
    gate.  km_opt=3 is the closure the nested LES probe actually ran.
    """
    return (inp.replace(" km_opt = 4, 4,", " km_opt = 4, 3,")
               .replace(" bl_pbl_physics = 11, 11,",
                        " bl_pbl_physics = 11, 0,"))


def _les_root(inp: str) -> str:
    """The base pair whose ROOT selects km_opt=2 (prognostic TKE).

    The root is the only domain km_opt=2 is admitted on today: the gate
    requires bl_pbl_physics=0 and refuses ``nested``.
    """
    return (inp.replace(" km_opt = 4, 4,", " km_opt = 2, 4,")
               .replace(" bl_pbl_physics = 11, 11,",
                        " bl_pbl_physics = 0, 11,"))


def test_turbulence_row_reaches_every_per_domain_run_config(tmp_path):
    """Booby-trap reach test: a NON-UNIFORM turbulence row must land per
    domain, which is the whole point of `a PBL parent may carry a PBL-off
    LES child`.  c_k is the key that closed the km_opt=2 gate."""
    toml_text, report = _import_with(
        tmp_path,
        inp_filter=_les_child,
        extra_dynamics=(" c_s = 0.25, 0.18,\n"
                        " c_k = 0.15, 0.10,\n"
                        " mix_isotropic = 0, 1,\n"
                        " mix_upper_bound = 0.1, 0.25,\n"
                        " tke_upper_bound = 1000., 200.,\n"
                        " tke_heat_flux = 0., 0.24,\n"
                        " tke_drag_coefficient = 0., 0.0013,\n"))
    exp = _load(tmp_path, toml_text)
    assert [dc.run.km_opt for dc in exp.domains] == [4, 3]
    assert [dc.run.c_s for dc in exp.domains] == [0.25, 0.18]
    assert [dc.run.c_k for dc in exp.domains] == [0.15, 0.10]
    assert [dc.run.mix_isotropic for dc in exp.domains] == [0, 1]
    assert [dc.run.mix_upper_bound for dc in exp.domains] == [0.1, 0.25]
    assert [dc.run.tke_upper_bound for dc in exp.domains] == [1000.0, 200.0]
    assert [dc.run.tke_heat_flux for dc in exp.domains] == [0.0, 0.24]
    assert [dc.run.tke_drag_coefficient
            for dc in exp.domains] == [0.0, 0.0013]
    translated = {(t.section, t.key) for t in report.translated}
    for key in ("c_s", "c_k", "mix_isotropic", "mix_upper_bound",
                "tke_upper_bound", "tke_heat_flux",
                "tke_drag_coefficient"):
        assert ("dynamics", key) in translated, key
    # mix_isotropic must stay an INT through the TOML round trip.
    assert "mix_isotropic = 1" in toml_text
    assert all(isinstance(dc.run.mix_isotropic, int) for dc in exp.domains)


def test_turbulence_row_absent_emits_nothing(tmp_path):
    """Every Registry default equals gpuwm's frozen RunConfig default, so
    omission resolves identically and established imports stay
    byte-identical."""
    toml_text, _ = import_namelists(*_pair(tmp_path))
    for key in ("c_s", "c_k", "mix_isotropic", "mix_upper_bound",
                "tke_upper_bound", "tke_heat_flux",
                "tke_drag_coefficient"):
        assert key not in toml_text, key
    root = _load(tmp_path, toml_text).root.run
    assert (root.c_s, root.c_k) == (0.25, 0.15)
    assert (root.mix_isotropic, root.mix_upper_bound) == (0, 0.1)
    assert root.tke_upper_bound == 1000.0
    assert (root.tke_heat_flux, root.tke_drag_coefficient) == (0.0, 0.0)


def test_uniform_turbulence_row_emits_shared_only(tmp_path):
    """A uniform column is a [shared] value with no per-domain override --
    the same rule epssm follows."""
    toml_text, _ = _import_with(
        tmp_path, extra_dynamics=" c_k = 0.12, 0.12,\n")
    assert toml_text.count("c_k") == 1
    exp = _load(tmp_path, toml_text)
    assert [dc.run.c_k for dc in exp.domains] == [0.12, 0.12]


@pytest.mark.parametrize("key,bad", [
    ("c_s", "0.25, -1.0,"),
    ("c_k", "0.15, 0.0,"),
    ("mix_upper_bound", "0.1, 0.0,"),
    ("tke_upper_bound", "1000., -5.,"),
    ("tke_drag_coefficient", "0., -0.1,"),
    ("mix_isotropic", "0, 2,"),
])
def test_turbulence_row_refuses_out_of_range_values(tmp_path, key, bad):
    """Namelist-anchored refusal, not a downstream RunConfig traceback."""
    with pytest.raises(ValueError, match=key):
        _import_with(tmp_path, extra_dynamics=f" {key} = {bad}\n")


def test_tke_adv_opt_is_pinned_when_a_domain_selects_km_opt_2(tmp_path):
    """gpuwm advects the km_opt=2 TKE carrier through WRF's
    positive-definite RK3 rows (gpuwm/core/moist.py advance_tke_stage)
    and implements no other transport, so tke_adv_opt = 1 is PINNED --
    not dropped as inert, which is what it was until km_opt=2 falsified
    that rationale."""
    _, report = _import_with(tmp_path, inp_filter=_les_root,
                             extra_dynamics=" tke_adv_opt = 1, 1,\n")
    fixed = {(f.section, f.key): f for f in report.fixed}
    assert ("dynamics", "tke_adv_opt") in fixed
    assert "positive-definite" in fixed[("dynamics", "tke_adv_opt")].reason
    assert ("dynamics", "tke_adv_opt") not in {
        (d.section, d.key) for d in report.dropped}


def test_tke_adv_opt_refuses_a_transport_gpuwm_does_not_implement(tmp_path):
    """A key that changes the answer must be refused, never dropped."""
    with pytest.raises(ValueError, match="tke_adv_opt"):
        _import_with(tmp_path, inp_filter=_les_root,
                     extra_dynamics=" tke_adv_opt = 0, 0,\n")


def test_tke_adv_opt_stays_inert_without_a_prognostic_tke_domain(tmp_path):
    """With no km_opt=2 domain there is no TKE carrier to transport, so
    WRF would not consume it either."""
    _, report = _import_with(tmp_path,
                             extra_dynamics=" tke_adv_opt = 0, 0,\n")
    dropped = {(d.section, d.key): d for d in report.dropped}
    assert ("dynamics", "tke_adv_opt") in dropped
    assert "km_opt = 2" in dropped[("dynamics", "tke_adv_opt")].reason


# --------------------------------------------------------------------
# The gray-zone parent chain (P4).  The 11 -> YSU remap died when the
# Shin-Hong port admitted the scheme (_BL_MAP row 11 is native, the
# physics_compat matrix scores its four (11, sfclay) cells, and
# PHYSICS_SLOT_DISPATCH row 11 runs it -- the three legs that widen
# together per _BL_MAP's own policy).  This test is the acceptance
# instrument for that claim on the release line: a THREE-domain chain
# whose parents select 11 and whose child is PBL-off must round-trip
# per domain with no substitution row, or the remap is not dead.
# --------------------------------------------------------------------

def test_shinhong_parent_chain_round_trips_per_domain_without_remap(
        tmp_path):
    """bl_pbl_physics = 11, 11, 0 -> per-domain [11, 11, 0], no
    substitution row, through the emitted TOML and back out of
    load_experiment (the P4 gray-zone parent-chain shape: SH parents
    with km_opt=4, PBL-off child on the LES closure)."""
    wps, inp = _generic_hierarchy_pair(tmp_path, 3)
    wps_text = wps.read_text(encoding="utf-8")
    input_text = inp.read_text(encoding="utf-8")
    wps_columns = {
        "parent_id": (1, 1, 2),
        "parent_grid_ratio": (1, 3, 3),
        "i_parent_start": (1, 40, 15),
        "j_parent_start": (1, 30, 15),
        "e_we": (101, 61, 31),
        "e_sn": (81, 61, 31),
    }
    for key, values in wps_columns.items():
        wps_text = _replace_namelist_column(wps_text, key, values)
    input_columns = {
        **wps_columns,
        "parent_id": (0, 1, 2),
        "parent_time_step_ratio": (1, 3, 3),
        "dx": (12000.0, 4000.0, 1333.333333),
        "dy": (12000.0, 4000.0, 1333.333333),
        "bl_pbl_physics": (11, 11, 0),
        "km_opt": (4, 4, 3),
    }
    for key, values in input_columns.items():
        input_text = _replace_namelist_column(input_text, key, values)
    wps.write_text(wps_text, encoding="utf-8")
    inp.write_text(input_text, encoding="utf-8")

    toml_text, report = import_namelists(wps, inp, name="sh-parent-chain")
    # No substitution leg may survive on any domain: the row is native.
    assert not any(s.key == "bl_pbl_physics" for s in report.substitutions)
    assert "bl_pbl_physics 11" not in report.format()
    # Round trip: the emitted TOML reloads to the same per-domain column.
    exp = _load(tmp_path, toml_text, name="sh-parent-chain.toml")
    assert [dc.run.bl_pbl_physics for dc in exp.domains] == [11, 11, 0]
    assert [dc.run.km_opt for dc in exp.domains] == [4, 4, 3]
