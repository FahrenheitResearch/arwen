"""Which fields each kind of observation may move (audit S9).

Every observation used to update every analysed field it reached: cloud
water path dried the inflow through sampled covariances, and surface
reports moved storm hydrometeors aloft.  gpuwm.da.field_rules withholds
them, through the dispersion gate's exact local re-solve.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.da import field_rules
from gpuwm.da.letkf import (GridGeometry, GriddedObs, LetkfConfig,
                            Localization, analyze)
from gpuwm.da.velocity_dispersion import DispersionGate, withhold

NZ, NY, NX, R = 4, 12, 12, 8
LOC = Localization(horizontal_m=6000.0, vertical_m=4000.0)


def _prior():
    rng = np.random.default_rng(7)
    common = rng.normal(size=(R, NZ, NY, NX))
    return {"qc": 1e-4 + 5e-5 * common,
            "qv": 8e-3 + 1e-3 * (0.8 * common
                                 + 0.6 * rng.normal(size=(R, NZ, NY, NX))),
            "u": 3.0 + rng.normal(size=(R, NZ, NY, NX))}


def _grid():
    return GridGeometry(dx_m=1000.0, dy_m=1000.0,
                        heights_m=(np.arange(NZ) + 0.5) * 500.0)


def _batch(name, prior, field="qc", offset=5e-5):
    mask = np.zeros((NZ, NY, NX), bool)
    mask[1, 6, 6] = True
    values = np.where(mask, prior[field].mean(axis=0) + offset, 0.0)
    errors = np.full((NZ, NY, NX), 1e-5)
    return GriddedObs(name=name, values=values, errors=errors,
                      simulated=prior[field].copy(), mask=mask)


def _solve(prior, batches, fields, geometry):
    cfg = LetkfConfig(localization=LOC, analysis_fields=tuple(fields),
                      rtps_alpha=0.0)
    return analyze(prior, batches, geometry, cfg)


def test_the_rules_are_a_table_read_by_batch_name():
    assert field_rules.rule_for_batch("cwp")[0] == "cloud-water-path"
    assert field_rules.rule_for_batch("cwp@slot2")[0] == "cloud-water-path"
    assert field_rules.rule_for_batch("temperature_2m:asos")[0] == "surface"
    assert field_rules.rule_for_batch("vr:KDMX") == (None, None)
    row = field_rules.KIND_FIELD_RULES["surface"]
    withheld = field_rules.withheld_fields(row)
    assert "qr" in withheld and "qv" not in withheld and "u" not in withheld


def test_cloud_water_path_no_longer_moves_vapour():
    prior, grid = _prior(), _grid()
    fields = ("qc", "qv", "u")
    batches = [_batch("cwp", prior)]
    joint = _solve(prior, batches, fields, grid)
    assert np.abs(joint["qv"]).max() > 0.0, "the case must couple qc and qv"
    gates, receipt = field_rules.kind_gates(
        batches, fields, shape=(NZ, NY, NX), dx_m=1000.0, dy_m=1000.0,
        localization=LOC)
    assert [g["kind"] for g in receipt["gates"]] == ["cloud-water-path"]
    ruled, _ = withhold(_solve, prior, batches, joint, gates, fields,
                        geometry=grid, localization=LOC)
    # Only batch withheld: the re-solve has no observation, so vapour (and
    # heat, not analysed here) take the unobserved transform: zero.
    assert np.abs(ruled["qv"]).max() == 0.0
    np.testing.assert_array_equal(ruled["qc"], joint["qc"])
    np.testing.assert_array_equal(ruled["u"], joint["u"])


def test_surface_reports_no_longer_move_hydrometeors():
    prior, grid = _prior(), _grid()
    fields = ("qc", "qv")
    batches = [_batch("temperature_2m:asos", prior, field="qv", offset=1e-3)]
    joint = _solve(prior, batches, fields, grid)
    assert np.abs(joint["qc"]).max() > 0.0
    gates, _ = field_rules.kind_gates(
        batches, fields, shape=(NZ, NY, NX), dx_m=1000.0, dy_m=1000.0,
        localization=LOC)
    ruled, _ = withhold(_solve, prior, batches, joint, gates, fields,
                        geometry=grid, localization=LOC)
    assert np.abs(ruled["qc"]).max() == 0.0
    np.testing.assert_array_equal(ruled["qv"], joint["qv"])


def test_gates_reach_only_the_batch_columns_plus_the_cutoff():
    prior = _prior()
    gates, _ = field_rules.kind_gates(
        [_batch("cwp", prior)], ("qc", "qv"), shape=(NZ, NY, NX),
        dx_m=1000.0, dy_m=1000.0, localization=LOC)
    columns = np.asarray(gates[0].columns)
    assert columns[6, 6] and not columns[0, 0]
    assert columns.sum() < NY * NX


def test_groups_share_no_field():
    a = DispersionGate("vr:A", ("thp", "qv"), np.ones((2, 2), bool))
    b = DispersionGate("cwp", ("qv", "thp"), np.ones((2, 2), bool))
    c = DispersionGate("t2:x", ("qr", "qs"), np.ones((2, 2), bool))
    groups = field_rules.group_by_fields([a, b, c])
    assert [len(g) for g in groups] == [2, 1]
    bad = DispersionGate("x", ("qv", "qr"), np.ones((2, 2), bool))
    with pytest.raises(ValueError, match="identical or disjoint"):
        field_rules.group_by_fields([a, bad])


def test_the_config_default_is_on():
    from gpuwm.da.radar_assimilation import RadarAssimilationConfig

    cfg = RadarAssimilationConfig(localization=LOC, rtps_alpha=0.9)
    assert cfg.kind_field_rules is True
