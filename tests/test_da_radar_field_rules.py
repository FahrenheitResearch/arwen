"""Radar field rules and number rediagnosis (gpuwm.da.field_rules, lane 3).

Breakage pinned, named: on the 2026-10-01 19Z CONUS analysis every
observation updated all 14 fields; radar added three times more vapour than
condensate (36 % of it under 0-15 dBZ echo), theta moved up to 16.8 K, and
numbers and aerosols were analysed from radar.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.da import field_rules as fr
from gpuwm.da.letkf import GriddedObs


def _z_batch(values):
    v = np.asarray(values, dtype=np.float64)
    return GriddedObs(name="z", values=v, errors=np.full(v.shape, 3.0),
                      simulated=np.zeros((2,) + v.shape), mask=v > -90)


def test_echo_columns_and_dilation():
    v = np.full((2, 5, 5), -99.0)
    v[1, 2, 2] = 30.0
    v[0, 0, 0] = 10.0          # weak: not precipitation
    cols = fr.echo_columns([_z_batch(v)], echo_floor=15.0, dilate=1)
    assert cols[2, 2] and cols[1, 2] and cols[2, 3]
    assert not cols[0, 0] and cols.sum() == 5


def test_combine_applies_the_design_rules():
    shape = (2, 1, 2, 2)
    joint = {n: np.full(shape, 1.0) for n in
             ("u", "thp", "qv", "qr", "nr", "nwfa")}
    joint["qv"][:] = 0.005
    no_radar = {"u": np.full(shape, 0.25), "thp": np.full(shape, 0.5),
                "qv": np.full(shape, 0.001)}
    radar_only = {"qr": np.full(shape, 0.002)}
    cols = np.array([[True, False], [False, False]])
    out, r = fr.combine(joint, no_radar=no_radar, radar_only=radar_only,
                        analysis_fields=tuple(joint),
                        z_thermo_weight=0.1, z_hydrometeors=True,
                        qv_cap=1e-4, columns=cols)
    assert np.all(out["u"] == 0.25)                       # no radar in winds
    assert out["thp"][0, 0, 0, 0] == pytest.approx(0.5 + 0.1 * 0.5)
    assert out["thp"][0, 0, 1, 1] == pytest.approx(0.5)   # outside echo
    assert out["qv"][0, 0, 0, 0] == pytest.approx(0.001 + 1e-4)  # capped
    assert np.all(out["qr"] == 0.002)                     # radar-only mass
    assert np.all(out["nr"] == 0) and np.all(out["nwfa"] == 0)
    out, _ = fr.combine(joint, no_radar=no_radar, radar_only=None,
                        analysis_fields=tuple(joint), z_thermo_weight=0.0,
                        z_hydrometeors=False, qv_cap=None, columns=cols)
    assert np.all(out["qr"] == 0) and np.all(out["thp"] == 0.5)


def test_rain_number_follows_mass_by_the_scheme_relation():
    from gpuwm.core.thompson_entry import make_rain_number

    nz, ny, nx = 1, 1, 3
    prior = {"qr": np.array([[[[1e-3, 0.0, 1e-3]]]]),
             "nr": np.array([[[[5e3, 0.0, 5e3]]]]),
             "qv": np.full((1, nz, ny, nx), 0.01)}
    inc = {"qr": np.array([[[[1e-3, 2e-3, 0.0]]]]),
           "nr": np.zeros((1, nz, ny, nx))}
    states = {0: {"p": np.full((nz, ny, nx), 90000.0),
                  "alt": np.full((nz, ny, nx), 1.0 / 1.1)}}
    out, r = fr.rediagnose_numbers(prior, inc, states, [0], mode="scheme",
                                   mp_physics=28)
    na = prior["nr"][0, 0, 0] + out["nr"][0, 0, 0]
    temp = fr._temperature(90000.0, 1.0 / 1.1, 0.01)
    want = make_rain_number(np.array([2e-3 * 1.1]), np.array([temp]))[0] / 1.1
    assert na[0] == pytest.approx(want, rel=1e-6)
    assert na[1] == pytest.approx(want, rel=1e-6)
    assert na[2] == 5e3                                    # untouched
    assert r["species"]["nr"]["points_rediagnosed"] == 2


def test_preserve_size_scales_number_with_mass():
    prior = {"qs": np.array([[[[1e-3]]]]), "ns": np.array([[[[100.0]]]])}
    inc = {"qs": np.array([[[[1e-3]]]])}
    out, r = fr.rediagnose_numbers(prior, inc, {0: {}}, [0], mode="scheme",
                                   mp_physics=10)
    assert prior["ns"][0, 0, 0, 0] + out["ns"][0, 0, 0, 0] == pytest.approx(200.0)
    assert r["species"]["ns"]["relation"] == "background size"


def test_settings_are_checked():
    with pytest.raises(fr.FieldRuleError):
        fr.check_settings("loose", 0.1, None, "scheme")
    with pytest.raises(fr.FieldRuleError):
        fr.check_settings("design", 1.5, None, "scheme")
    from gpuwm.da.letkf import Localization
    from gpuwm.da.radar_assimilation import RadarAssimilationConfig

    cfg = RadarAssimilationConfig(
        localization=Localization(horizontal_m=36e3, vertical_m=6e3),
        rtps_alpha=0.9)
    assert cfg.field_rules == "design" and cfg.z_thermo_weight == 0.1
    assert cfg.z_qv_cap is None


def test_cloud_water_path_is_a_condensate_observation():
    assert "cwp" in fr.CONDENSATE_TOKENS
    assert fr.batch_token("cwp:goes16@slot0") == "cwp"


def _named(name, shape=(2, 3, 3)):
    v = np.full(shape, 30.0)
    return GriddedObs(name=name, values=v, errors=np.full(shape, 3.0),
                      simulated=np.zeros((2,) + shape),
                      mask=np.ones(shape, bool))


def test_the_rules_are_in_pass_gates_in_lane_0_groups():
    batches = [_named("z"), _named("z0"), _named("vr:KTLX"),
               _named("temperature_2m"), _named("cwp:goes")]
    fields = ("u", "v", "thp", "qv", "qr", "nr", "nwfa")
    echo = np.zeros((3, 3), bool)
    echo[1, 1] = True
    gates, echo_gates, r = fr.radar_rule_gates(
        batches, fields, shape=(2, 3, 3), echo=echo, z_thermo_weight=0.1,
        z_hydrometeors=True)
    by = {(g.batch, frozenset(g.fields), float(g.keep)): np.asarray(g.columns)
          for g in gates}
    assert by[("z", frozenset({"u", "v"}), 0.0)].all()
    thermo = by[("z", frozenset({"thp", "qv"}), 0.0)]
    assert not thermo[1, 1] and thermo.sum() == 8      # outside echo only
    kept = by[("z", frozenset({"thp", "qv"}), 0.1)]
    assert kept[1, 1] and kept.sum() == 1              # keep w inside echo
    hydro = frozenset(fr.withheld_fields({"withheld": "hydrometeors"}))
    assert ("vr:KTLX", hydro, 0.0) in by and ("temperature_2m", hydro, 0.0) in by
    assert ("z", hydro, 0.0) not in by and ("cwp:goes", hydro, 0.0) not in by
    assert echo_gates is None
    fr.group_by_fields(gates)      # field sets identical or disjoint


def test_keep0_gates_take_precedence_inside_echo():
    echo = np.ones((3, 3), bool)
    blocked = np.zeros((3, 3), bool)
    blocked[0, 0] = True                     # a dispersion-gated column
    gates, _, _ = fr.radar_rule_gates(
        [_named("z")], ("thp", "qv"), shape=(2, 3, 3), echo=echo,
        z_thermo_weight=0.2, z_hydrometeors=True, keep0_thermo=blocked)
    zero = [np.asarray(g.columns) for g in gates if g.keep == 0.0]
    kept = [np.asarray(g.columns) for g in gates if g.keep == 0.2]
    assert zero[0][0, 0] and not kept[0][0, 0]
    assert not (zero[0] & kept[0]).any()     # one keep per column


def test_a_qv_cap_routes_the_echo_blend_to_the_host():
    gates, echo_gates, _ = fr.radar_rule_gates(
        [_named("z")], ("thp", "qv"), shape=(2, 3, 3),
        echo=np.ones((3, 3), bool), z_thermo_weight=0.1,
        z_hydrometeors=True, in_pass_keep=False)
    assert all(g.keep == 0.0 for g in gates)
    assert [g.batch for g in echo_gates] == ["z"]


def test_weight_zero_withholds_radar_thermo_everywhere_and_kenda_masses():
    gates, echo_gates, _ = fr.radar_rule_gates(
        [_named("z")], ("thp", "qv", "qr"), shape=(2, 3, 3),
        echo=np.ones((3, 3), bool), z_thermo_weight=0.0, z_hydrometeors=False)
    cols = {frozenset(g.fields): np.asarray(g.columns) for g in gates}
    assert cols[frozenset({"thp", "qv"})].all() and echo_gates is None
    assert any("qr" in g.fields for g in gates)          # KENDA: z off masses


def test_blend_inside_echo_only():
    shape = (1, 1, 2, 2)
    joint = {"thp": np.full(shape, 1.0), "qv": np.full(shape, 0.004)}
    withheld = {"thp": np.full(shape, 0.5), "qv": np.full(shape, 0.001)}
    cols = np.array([[True, False], [False, False]])
    out = fr.blend_echo(joint, withheld, columns=cols, weight=0.1,
                        qv_cap=1e-4)
    assert out["thp"][0, 0, 0, 0] == pytest.approx(0.55)
    assert out["thp"][0, 0, 1, 1] == 1.0
    assert out["qv"][0, 0, 0, 0] == pytest.approx(0.001 + 1e-4)


def test_numbers_and_aerosols_take_no_filter_increment():
    inc = {n: np.ones((1, 1, 1, 1)) for n in ("qr", "nr", "nwfa", "thp")}
    out = fr.zero_unanalysed(inc, tuple(inc))
    assert out["nr"].sum() == 0 and out["nwfa"].sum() == 0
    assert out["qr"].sum() == 1 and out["thp"].sum() == 1
