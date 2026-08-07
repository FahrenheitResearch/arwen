"""[perturbation] initial-state theta bubbles: schema, OFF path, application.

One focused node per behavior (the project's iterating discipline):
schema refusals, the absence no-op (byte-identical state), the applied
bubble's location/magnitude, RH preservation, and the two geometry
refusals (center outside the coarse domain; enabled bubble touching
zero cells).  The application tests run ``initialize_real`` end to end
on the CPU backend so the seam is exercised where it lives, not in a
mock.
"""

from datetime import datetime

import numpy as np
import pytest

from gpuwm.config import RunConfig
from gpuwm.core.grid import make_vertical_coord
from gpuwm.experiment import (
    MAX_BUBBLE_AMPLITUDE_K,
    BubbleConfig,
    PerturbationConfig,
    build_experiment,
    refuse_unrouted_perturbation,
)
from gpuwm.ingest.horiz import HorizontalSnapshot
from gpuwm.ingest.init_perturbation import build_initial_state_perturbation
from gpuwm.ingest.real import (
    _mixing_ratio_to_relative_humidity,
    _temperature_from_potential_temperature,
    hydrostatic_residual,
    initialize_real,
)
from gpuwm.static.lambert import LambertGrid


# ---------------------------------------------------------------------------
# Schema fixtures
# ---------------------------------------------------------------------------

def _experiment_raw(perturbation=None):
    raw = {
        "experiment": {
            "name": "schema_probe",
            "start_time": datetime(2026, 8, 1, 12),
            "run_seconds": 3600.0,
            "restart_interval_s": 0.0,
        },
        "projection": {
            "map_proj": "lambert", "ref_lat": 38.5, "ref_lon": -99.5,
            "truelat1": 30.0, "truelat2": 50.0, "stand_lon": -99.5,
        },
        "shared": {
            "nz": 6, "ztop": 16000.0, "p_top": 10000.0,
            "eta_levels": [1.0, 0.9, 0.74, 0.56, 0.38, 0.19, 0.0],
            "hybrid_opt": 2, "etac": 0.2, "moist": True,
            "terrain_opt": 1, "base_temp": 290.0,
        },
        "domain": [{
            "grid_id": 1, "parent_id": 0, "i_parent_start": 1,
            "j_parent_start": 1, "parent_grid_ratio": 1,
            "parent_time_step_ratio": 1, "nx": 24, "ny": 20,
            "dx": 3000.0, "time_step": 15, "specified": True,
            "nested": False, "history_interval_s": 3600.0,
        }],
    }
    if perturbation is not None:
        raw["perturbation"] = perturbation
    return raw


def _bubble_entry(**overrides):
    entry = {
        "center_lat": 38.5, "center_lon": -99.5,
        "center_height_m": 1500.0, "radius_km": 10.0,
        "depth_m": 1500.0, "amplitude_k": 2.5,
    }
    entry.update(overrides)
    return entry


# ---------------------------------------------------------------------------
# Schema: refusals and acceptance
# ---------------------------------------------------------------------------

def test_unknown_key_in_perturbation_block_is_refused():
    raw = _experiment_raw({"bubbles": [_bubble_entry()], "bubles": 1})
    with pytest.raises(ValueError, match=r"does not have a key"):
        build_experiment(raw, source="probe.toml")


def test_unknown_key_in_bubble_is_refused_with_suggestion():
    raw = _experiment_raw({"bubbles": [_bubble_entry(radius_kms=5.0)]})
    with pytest.raises(ValueError, match=r"radius_kms.*radius_km"):
        build_experiment(raw, source="probe.toml")


def test_missing_required_bubble_key_is_refused():
    entry = _bubble_entry()
    del entry["amplitude_k"]
    raw = _experiment_raw({"bubbles": [entry]})
    with pytest.raises(ValueError, match=r"missing required key.*amplitude_k"):
        build_experiment(raw, source="probe.toml")


@pytest.mark.parametrize("key", ["radius_km", "depth_m", "amplitude_k"])
def test_nonpositive_bubble_geometry_is_refused(key):
    raw = _experiment_raw({"bubbles": [_bubble_entry(**{key: 0.0})]})
    with pytest.raises(ValueError, match=rf"{key} = 0\.0 must be positive"):
        build_experiment(raw, source="probe.toml")


def test_amplitude_beyond_sanity_bound_is_refused_with_value_named():
    raw = _experiment_raw({"bubbles": [_bubble_entry(amplitude_k=12.5)]})
    with pytest.raises(
            ValueError,
            match=rf"amplitude_k = 12\.5 exceeds the "
                  rf"{MAX_BUBBLE_AMPLITUDE_K:g} K sanity bound"):
        build_experiment(raw, source="probe.toml")


def test_empty_bubbles_array_is_refused():
    raw = _experiment_raw({"bubbles": []})
    with pytest.raises(ValueError, match=r"array of tables"):
        build_experiment(raw, source="probe.toml")


def test_accepted_block_echoes_every_value_into_the_receipt():
    raw = _experiment_raw({"bubbles": [_bubble_entry(rh_preserve=True)]})
    exp = build_experiment(raw, source="probe.toml")
    assert exp.perturbation is not None
    receipt = exp.perturbation.receipt()
    assert receipt["schema"] == "gpuwm-initial-perturbation-v1"
    assert receipt["bubbles"] == [{
        "center_lat": 38.5, "center_lon": -99.5,
        "center_height_m": 1500.0, "radius_km": 10.0,
        "depth_m": 1500.0, "amplitude_k": 2.5, "rh_preserve": True,
    }]


def test_unrouted_route_refuses_a_configured_block_by_name():
    exp = build_experiment(
        _experiment_raw({"bubbles": [_bubble_entry()]}),
        source="probe.toml")
    with pytest.raises(ValueError, match=r"probe-route.*does not apply"):
        refuse_unrouted_perturbation(exp, "probe-route")
    refuse_unrouted_perturbation(  # absent block: no-op
        build_experiment(_experiment_raw(), source="probe.toml"),
        "probe-route")


# ---------------------------------------------------------------------------
# Absence: zero behavior change, byte-identical prepared state
# ---------------------------------------------------------------------------

def _application_fixture(ny=20, nx=24, nz=6):
    """A CPU-only initialize_real setup with a real Lambert grid."""
    eta = np.array([1.0, 0.9, 0.74, 0.56, 0.38, 0.19, 0.0])
    assert eta.size == nz + 1
    coord = make_vertical_coord(nz, hybrid_opt=2, etac=0.2, eta_levels=eta)
    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=3000.0, dy=3000.0,
                    ztop=16000.0, dt=15.0, run_seconds=900.0,
                    hybrid_opt=2, etac=0.2, moist=True, terrain_opt=1,
                    base_temp=290.0)
    grid = LambertGrid(ref_lat=38.5, ref_lon=-99.5, truelat1=30.0,
                       truelat2=50.0, stand_lon=-99.5, dx=3000.0,
                       dy=3000.0, e_we=nx + 1, e_sn=ny + 1)
    levels = np.array([100.0, 200.0, 300.0, 500.0, 700.0, 850.0, 1000.0])
    pressure = levels[:, None, None] * 100.0
    shape = (levels.size, ny, nx)
    temperature = np.broadcast_to(
        215.0 + 72.0 * (pressure / 100000.0) ** 0.20, shape).copy()
    height = np.broadcast_to(
        -7800.0 * np.log(pressure / 100000.0), shape).copy()
    rh = np.broadcast_to(35.0 + 45.0 * (pressure / 100000.0), shape).copy()
    fields = {
        "TT": temperature.astype(np.float32),
        "GHT": height.astype(np.float32),
        "RH": rh.astype(np.float32),
        "UU": np.full((levels.size, ny, nx + 1), 8.0, np.float32),
        "VV": np.full((levels.size, ny + 1, nx), -2.0, np.float32),
        "PSFC": np.full((ny, nx), 96500.0, np.float32),
        "T2": np.full((ny, nx), 288.0, np.float32),
        "D2": np.full((ny, nx), 281.0, np.float32),
        "U10": np.full((ny, nx + 1), 7.0, np.float32),
        "V10": np.full((ny + 1, nx), -1.5, np.float32),
    }
    snapshot = HorizontalSnapshot(
        valid_time=datetime(2026, 8, 1, 12), levels_hpa=levels,
        fields=fields)
    terrain = np.full((ny, nx), 350.0)
    source_orography = np.full((ny, nx), 340.0)
    return cfg, coord, grid, snapshot, terrain, source_orography


def _initialize(perturbation_cfg=None, require_containment=True, **kwargs):
    cfg, coord, grid, snapshot, terrain, source_orography = (
        _application_fixture())
    applier = build_initial_state_perturbation(
        perturbation_cfg, grid, grid_id=1,
        require_containment=require_containment)
    return initialize_real(
        snapshot, cfg, coord, terrain,
        source_orography=source_orography, p_top=10000.0,
        sfcp_to_sfcp=True, preprocess_backend="cpu",
        state_backend="preprocess", initial_perturbation=applier,
        **kwargs), grid


def test_absent_block_leaves_the_prepared_state_byte_identical():
    """OFF contract: no applier, no branch, byte-identical fields."""
    baseline, _ = _initialize()          # kwarg carries applier=None
    off_result, _ = _initialize(perturbation_cfg=None)
    for name in ("thp", "php", "mup", "qv", "u", "v"):
        np.testing.assert_array_equal(
            np.asarray(getattr(baseline.state, name)),
            np.asarray(getattr(off_result.state, name)))
    assert off_result.initial_perturbation == {}


def test_applied_bubble_peaks_at_the_declared_center():
    bubble = BubbleConfig(center_lat=38.5, center_lon=-99.5,
                          center_height_m=1500.0, radius_km=10.0,
                          depth_m=1500.0, amplitude_k=2.5)
    spec = PerturbationConfig(bubbles=(bubble,))
    baseline, _ = _initialize()
    perturbed, grid = _initialize(perturbation_cfg=spec)
    receipt = perturbed.initial_perturbation
    assert receipt["schema"] == "gpuwm-initial-perturbation-apply-v1"
    assert receipt["grid_id"] == 1
    (row,) = receipt["bubbles"]
    assert row["applied"] is True
    assert row["cells_touched"] > 0
    assert 0.0 < row["max_theta_added_k"] <= 2.5
    # The written delta is where the receipt says it is: the theta
    # difference between the runs peaks at the cell nearest the center
    # and is exactly zero outside the ellipse.
    delta = (np.asarray(perturbed.state.thp, dtype=np.float64)
             - np.asarray(baseline.state.thp, dtype=np.float64))
    k, j, i = np.unravel_index(np.abs(delta).argmax(), delta.shape)
    x, y = grid.latlon_to_ij(38.5, -99.5)
    assert abs((i + 1) - float(x)) <= 0.51
    assert abs((j + 1) - float(y)) <= 0.51
    assert delta.max() == pytest.approx(row["max_theta_added_k"],
                                        abs=2.0e-3)
    assert delta.max() > 2.0        # near-center cell close to the peak
    far = np.ones_like(delta, dtype=bool)
    j0, i0 = int(round(float(y))) - 1, int(round(float(x))) - 1
    lo_j, hi_j = max(0, j0 - 5), min(delta.shape[1], j0 + 6)
    lo_i, hi_i = max(0, i0 - 5), min(delta.shape[2], i0 + 6)
    far[:, lo_j:hi_j, lo_i:hi_i] = False
    assert np.all(delta[far] == 0.0)
    # qv untouched without rh_preserve
    np.testing.assert_array_equal(np.asarray(perturbed.state.qv),
                                  np.asarray(baseline.state.qv))
    # and the perturbed state still passes the hydrostatic gate.
    residual = float(np.max(hydrostatic_residual(perturbed)))
    baseline_residual = float(np.max(hydrostatic_residual(baseline)))
    assert residual <= 4.0 * max(baseline_residual, 1.0e-3)


def test_rh_preserve_holds_relative_humidity_through_the_theta_change():
    bubble = BubbleConfig(center_lat=38.5, center_lon=-99.5,
                          center_height_m=1500.0, radius_km=10.0,
                          depth_m=1500.0, amplitude_k=2.5,
                          rh_preserve=True)
    spec = PerturbationConfig(bubbles=(bubble,))
    baseline, _ = _initialize()
    perturbed, _ = _initialize(perturbation_cfg=spec)
    (row,) = perturbed.initial_perturbation["bubbles"]
    assert row["rh_preserve"] is True
    assert row["max_qv_delta_kg_kg"] > 0.0
    mask = (np.asarray(perturbed.state.thp, dtype=np.float64)
            != np.asarray(baseline.state.thp, dtype=np.float64))
    assert mask.any()

    def relative_humidity(result):
        theta = (np.asarray(result.state.thp, dtype=np.float64)
                 + np.asarray(result.base.thb, dtype=np.float64))
        pressure = np.asarray(result.total_pressure, dtype=np.float64)
        temperature = _temperature_from_potential_temperature(
            theta, pressure)
        return _mixing_ratio_to_relative_humidity(
            temperature, pressure,
            np.asarray(result.state.qv, dtype=np.float64))

    rh_before = relative_humidity(baseline)[mask]
    rh_after = relative_humidity(perturbed)[mask]
    # FP32 state storage bounds the round trip; RH must survive to well
    # under one percent where a 2.5 K bubble would otherwise move it by
    # tens of percent.
    assert float(np.max(np.abs(rh_after - rh_before))) < 0.5
    unchanged_rh = float(np.max(np.abs(
        relative_humidity(perturbed)[~mask]
        - relative_humidity(baseline)[~mask])))
    assert unchanged_rh == 0.0


def test_center_outside_the_coarse_domain_is_refused():
    bubble = BubbleConfig(center_lat=45.0, center_lon=-80.0,
                          center_height_m=1500.0, radius_km=10.0,
                          depth_m=1500.0, amplitude_k=2.5)
    spec = PerturbationConfig(bubbles=(bubble,))
    with pytest.raises(ValueError, match=r"outside domain d01"):
        _initialize(perturbation_cfg=spec)


def test_center_outside_a_child_domain_is_recorded_not_refused():
    bubble = BubbleConfig(center_lat=45.0, center_lon=-80.0,
                          center_height_m=1500.0, radius_km=10.0,
                          depth_m=1500.0, amplitude_k=2.5)
    spec = PerturbationConfig(bubbles=(bubble,))
    result, _ = _initialize(perturbation_cfg=spec,
                            require_containment=False)
    (row,) = result.initial_perturbation["bubbles"]
    assert row["applied"] is False
    assert row["reason"] == "center outside this domain"
    assert row["cells_touched"] == 0


def test_enabled_bubble_touching_zero_cells_is_refused():
    bubble = BubbleConfig(center_lat=38.5, center_lon=-99.5,
                          center_height_m=15000.0, radius_km=0.4,
                          depth_m=10.0, amplitude_k=2.5)
    spec = PerturbationConfig(bubbles=(bubble,))
    with pytest.raises(ValueError, match=r"touches zero cells"):
        _initialize(perturbation_cfg=spec)


def test_apply_to_state_matches_the_initialize_real_seam():
    """The prepared-tree application point writes the same bubble.

    Route A applies inside ``initialize_real`` (FP64 columns, rebalanced
    geopotential); route B applies to the already-initialized state (the
    restored-prepared-cache seam).  Same cells, same receipt stats, and
    the written theta agrees to FP32 storage precision.
    """
    bubble = BubbleConfig(center_lat=38.5, center_lon=-99.5,
                          center_height_m=1500.0, radius_km=10.0,
                          depth_m=1500.0, amplitude_k=2.5)
    spec = PerturbationConfig(bubbles=(bubble,))
    seam, grid = _initialize(perturbation_cfg=spec)
    baseline, _ = _initialize()
    applier = build_initial_state_perturbation(
        spec, grid, grid_id=1, require_containment=True)
    state_receipt = applier.apply_to_state(baseline.state)
    assert state_receipt["application_point"] == "restored-prepared-state"
    (seam_row,) = seam.initial_perturbation["bubbles"]
    (state_row,) = state_receipt["bubbles"]
    assert state_row["cells_touched"] == seam_row["cells_touched"]
    # The state route reads FP32-stored thp/php, so the peak agrees to
    # storage precision, not to FP64 identity.
    assert state_row["max_theta_added_k"] == pytest.approx(
        seam_row["max_theta_added_k"], abs=1.0e-5)
    thp_seam = np.asarray(seam.state.thp, dtype=np.float64)
    thp_state = np.asarray(baseline.state.thp, dtype=np.float64)
    assert float(np.max(np.abs(thp_seam - thp_state))) < 5.0e-3
    # qv untouched on both routes without rh_preserve
    np.testing.assert_array_equal(np.asarray(baseline.state.qv),
                                  np.asarray(seam.state.qv))


def test_restart_identity_omits_an_absent_block_and_binds_a_present_one():
    from gpuwm.core.model import restart_identity_payload

    absent = build_experiment(_experiment_raw(), source="probe.toml")
    assert "perturbation" not in restart_identity_payload(absent)
    present = build_experiment(
        _experiment_raw({"bubbles": [_bubble_entry()]}),
        source="probe.toml")
    payload = restart_identity_payload(present)
    assert payload["perturbation"]["bubbles"][0]["amplitude_k"] == 2.5
