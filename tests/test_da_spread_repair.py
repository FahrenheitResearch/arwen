"""Lane 8 contracts, including the real CPU LETKF orchestration path."""
from dataclasses import replace

import numpy as np
import pytest

from gpuwm.da.spread_repair import (
    AdaptiveInflationState, SpreadRepairConfig, SpreadRepairController,
    additive_repair, echo_gates, scale_deviations, _smooth_noise,
    TimeShiftSnapshot, time_shifted_covariance, central_time_posterior,
    TimeShiftBuffer, inflate_observation_anomalies,
)
from gpuwm.da.letkf import GridGeometry, GriddedObs, LetkfConfig, Localization


def fields(shape=(10, 12, 12), members=6):
    return {"u": np.zeros((members,)+shape),
            "v": np.zeros((members,)+shape),
            "thp": np.full((members,)+shape, 5.),
            "qv": np.full((members,)+shape, .01),
            "qr": np.zeros((members,)+shape)}


def test_gates_positive_innovation_and_missing_coverage():
    cfg = SpreadRepairConfig()
    obs = np.array([[[35., 35., 30., 20., 0., np.nan]]])
    hx = np.broadcast_to(np.array([[[[15., 40., 5., 0., -35., 0.]]]]), (4,1,1,6))
    coverage = np.array([[[True, True, False, True, True, False]]])
    gate = echo_gates(obs, hx, coverage, cfg)
    np.testing.assert_array_equal(gate["additive"], [[[True, False, False, False, False, False]]])
    np.testing.assert_array_equal(gate["targeted"], [[[False, False, False, True, False, False]]])


def test_noise_clear_air_identity_and_mean_preservation():
    prior = fields()
    z = np.zeros((10,12,12)); z[:, 4:8, 4:8] = 35
    hx = np.zeros((6,10,12,12))
    gates = echo_gates(z, hx, np.ones_like(z, bool), SpreadRepairConfig())
    post = additive_repair(prior, gates, SpreadRepairConfig(), seed=52, cycle=0,
                           pressure=np.full((6,10,12,12), 80000.))
    for field in prior:
        np.testing.assert_array_equal(post[field][:, :, :4], prior[field][:, :, :4])
        np.testing.assert_allclose(post[field].mean(0), prior[field].mean(0), atol=2e-15, rtol=0)
    assert np.std(post["qv"][:, 3:7, 4:8, 4:8]) > 0
    assert np.std(post["u"][:, 3:7, 4:8, 4:8]) > 0
    np.testing.assert_array_equal(post["qr"], prior["qr"])
    # The top taper remains zero after normalization and mean removal.
    np.testing.assert_array_equal(post["u"][:, -1], prior["u"][:, -1])


def test_staggered_gate_never_leaks_to_face_adjacent_to_clear_air():
    prior = {"u": np.zeros((4,8,8,9))}
    z = np.zeros((8,8,8)); z[:, :, 4:] = 35
    gates = echo_gates(z, np.zeros((4,8,8,8)), np.ones_like(z,bool), SpreadRepairConfig())
    post = additive_repair(prior, gates, SpreadRepairConfig(), seed=3, cycle=1)
    np.testing.assert_array_equal(post["u"][..., :5], prior["u"][..., :5])
    assert np.std(post["u"][..., 5:]) > 0


def test_noise_reproducibility_and_cycle_stream_independence():
    cfg = SpreadRepairConfig()
    a = _smooth_noise((12,16,16), 22, cfg, np)
    b = _smooth_noise((12,16,16), 22, cfg, np)
    c = _smooth_noise((12,16,16), 23, cfg, np)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_vapour_bounds_preserve_mean_instead_of_clipping():
    prior = fields(shape=(8,8,8))
    gates = {"additive": np.ones((8,8,8), bool), "targeted": np.zeros((8,8,8),bool)}
    cfg = replace(SpreadRepairConfig(), vapour_fraction_std=100.)
    post = additive_repair(prior, gates, cfg, seed=1, cycle=1,
                           pressure=np.full((6,8,8,8), 80000.), vapour_upper=.011)
    assert post["qv"].min() >= 0
    assert post["qv"].max() <= .011 + 1e-15
    np.testing.assert_allclose(post["qv"].mean(0), prior["qv"].mean(0), atol=1e-16)


def test_rescale_zero_anomalies_is_not_claimed_as_spread_creation():
    prior = np.full((6,2,2,2), .01)
    post = scale_deviations(prior, 10., gate=np.ones((2,2,2),bool), lower=0.)
    np.testing.assert_allclose(post, prior, atol=1e-17)


def test_bound_rescale_preserves_positive_mean():
    prior = np.array([0., .01, .02, .03])[:,None,None,None]
    post = scale_deviations(prior, 9., gate=np.ones((1,1,1),bool), lower=0.,upper=.03)
    assert post.min() >= 0 and post.max() <= .03
    np.testing.assert_allclose(post.mean(0), prior.mean(0), rtol=0, atol=1e-17)


def test_controller_runs_actual_letkf_and_commits_restart_after_success():
    shape = (2,2,2)
    x = np.array([-2., -1., 1., 2.])[:,None,None,None]*np.ones(shape)
    prior = {"u": x}
    radar = GriddedObs("reflectivity", np.full(shape,35.), 10., x+10., np.ones(shape,bool))
    cfg = SpreadRepairConfig(wind_std_ms=0., adaptive=True, top_taper_levels=0)
    controller = SpreadRepairController(cfg)
    grid = GridGeometry(1000.,1000.,np.array([0.,1000.]))
    filter_cfg = LetkfConfig(Localization(2000.,2000.), ("u",), rtps_alpha=0.)
    seen = []
    def rebuild(states):
        seen.append(states["u"].copy())
        return [replace(radar, simulated=states["u"]+10.)]
    increments = controller.analyze(prior, [radar], grid, filter_cfg, radar=radar,
                                   rebuild_observations=rebuild)
    assert len(seen) == 1
    assert np.isfinite(increments["u"]).all()
    assert np.mean(increments["u"]) > 0
    assert controller.state.cycles == 1
    before = controller.state.checkpoint()
    def failed(states):
        raise RuntimeError("intentional operator failure")
    with pytest.raises(RuntimeError, match="intentional"):
        controller.analyze(prior, [radar], grid, filter_cfg, radar=radar,
                           rebuild_observations=failed)
    np.testing.assert_array_equal(controller.state.shape, before["shape"])
    np.testing.assert_array_equal(controller.state.scale, before["scale"])
    assert controller.state.cycles == before["cycles"]


def test_time_shifting_returns_central_roster_recentred_on_posterior_mean():
    snaps = [TimeShiftSnapshot(offset, {"u": np.array([1.,3.])[:,None,None,None]+offset/900},
                              -3600, -3600) for offset in (-900,0,900)]
    pooled, receipt = time_shifted_covariance(snaps, analysis_seconds=0)
    inc = {"u": np.arange(6.)[:,None,None,None]}
    central = central_time_posterior(pooled, inc, receipt)["u"]
    assert central.shape[0] == 2
    np.testing.assert_allclose(central.mean(0), (pooled["u"]+inc["u"]).mean(0))


def test_retained_slots_own_arrays_and_restart_without_changing_samples():
    buffer = TimeShiftBuffer()
    original = np.arange(4.)[:,None,None,None]
    for valid in (-900,0,900):
        buffer.retain(valid, {"u":original}, forecast_origin_seconds=-1800,
                      observation_cutoff_seconds=-1800)
    original[:] = 100
    restored = TimeShiftBuffer.restore(buffer.checkpoint())
    a,_ = time_shifted_covariance(buffer.snapshots(0), analysis_seconds=0)
    b,_ = time_shifted_covariance(restored.snapshots(0), analysis_seconds=0)
    np.testing.assert_array_equal(a["u"], b["u"])
    assert np.max(a["u"]) == 3
    with pytest.raises(ValueError, match="missing retained"):
        restored.snapshots(900)


def test_retained_slots_expire_without_overwriting_other_origins():
    buffer = TimeShiftBuffer(retention_seconds=1800)
    states = {"u":np.ones((2,1,1,1))}
    for valid in (0,900,1800,2700):
        buffer.retain(valid, states, forecast_origin_seconds=0, observation_cutoff_seconds=0)
    assert sorted(buffer._slots) == [900,1800,2700]
    with pytest.raises(ValueError, match="replacing"):
        buffer.retain(1800,states,forecast_origin_seconds=0,observation_cutoff_seconds=0)


def test_sparse_observation_inflation_preserves_point_roster_and_mean():
    from gpuwm.da.letkf import PointSet, PointSimulated
    shape = (2,2,2)
    hx = np.array([[1.,2.], [3.,4.], [5.,6.]])
    points = PointSet(np.array([0,7]),np.array([10.,20.]),np.array([1.,1.]),hx)
    batch = GriddedObs("z",np.zeros(shape),np.ones(shape),PointSimulated(points,shape),
                       np.ones(shape,bool),points=points)
    factor = np.ones(shape); factor.flat[0] = 4
    changed = inflate_observation_anomalies([batch],factor,np)[0]
    np.testing.assert_array_equal(changed.points.flat_index,points.flat_index)
    np.testing.assert_allclose(changed.points.simulated.mean(0),hx.mean(0))
    np.testing.assert_allclose(changed.points.simulated[:,0]-hx[:,0].mean(),2*(hx[:,0]-hx[:,0].mean()))
    np.testing.assert_array_equal(changed.points.simulated[:,1],hx[:,1])


def test_sparse_window_inflation_uses_local_indices_and_storage_shape():
    from gpuwm.da.letkf import PointSet, PointSimulated
    shape = (1,2,2)
    hx = np.array([[1.],[3.]])
    points = PointSet(np.array([0]),np.array([10.]),np.array([1.]),hx)
    batch = GriddedObs("vr",np.zeros(shape),np.ones(shape),PointSimulated(points,shape),
                       np.ones(shape,bool),points=points,window=(1,2,1,2))
    factor = np.ones((1,4,4)); factor[0,1,1] = 9
    changed = inflate_observation_anomalies([batch],factor,np)[0]
    assert changed.simulated.shape == (2,)+shape
    np.testing.assert_allclose(changed.points.simulated[:,0],[-1.,5.])


def test_cycle_runner_seam_uses_actual_filter_and_reports_its_policy():
    shape = (2,2,2)
    prior = {"u":np.array([-2.,-1.,1.,2.])[:,None,None,None]*np.ones(shape)}
    radar = GriddedObs("z",np.full(shape,35.),10.,prior["u"]+10,np.ones(shape,bool))
    controller = SpreadRepairController(SpreadRepairConfig(adaptive=False,wind_std_ms=0.))
    grid = GridGeometry(1000.,1000.,np.array([0.,1000.]))
    cfg = LetkfConfig(Localization(3000.,2000.),("u",),rtps_alpha=0.)
    increment = controller.analysis_runner()(prior,[radar],grid,cfg)
    assert np.mean(increment["u"]) > 0
    assert controller.last_receipt["operator_update"] == "linearized-covariance"
    assert controller.last_receipt["clear_air_noise"] is False


def test_auxiliary_fields_reach_operator_unchanged_and_receive_no_increment():
    shape = (2,2,2)
    x = np.array([-2.,-1.,1.,2.])[:,None,None,None]*np.ones(shape)
    prior = {"u":x,"auxiliary":np.full_like(x,42.)}
    radar = GriddedObs("z",np.full(shape,35.),10.,x+10,np.ones(shape,bool))
    controller = SpreadRepairController(SpreadRepairConfig(adaptive=False,wind_std_ms=0.))
    def rebuild(states):
        np.testing.assert_array_equal(states["auxiliary"],prior["auxiliary"])
        return [replace(radar,simulated=states["u"]+10)]
    result = controller.analyze(prior,[radar],GridGeometry(1000.,1000.,np.array([0.,1000.])),
        LetkfConfig(Localization(3000.,2000.),("u",),rtps_alpha=0.),radar=radar,
        rebuild_observations=rebuild)
    assert set(result) == {"u"}


def test_bounded_state_and_linearized_operator_share_effective_inflation():
    shape = (2,2,2)
    x = np.array([0.,.001,.002])[:,None,None,None]*np.ones(shape)
    prior = {"qr":x}
    radar = GriddedObs("z",np.full(shape,35.),10.,x*1000,np.ones(shape,bool))
    controller = SpreadRepairController(SpreadRepairConfig(adaptive=True))
    seen = []
    def rebuild(states):
        seen.append(states["qr"].copy())
        return [replace(radar,simulated=states["qr"]*1000)]
    controller.analyze(prior,[radar],GridGeometry(1000.,1000.,np.array([0.,1000.])),
        LetkfConfig(Localization(3000.,2000.),("qr",),rtps_alpha=0.),radar=radar,
        rebuild_observations=rebuild)
    np.testing.assert_allclose(seen[0],x,rtol=0,atol=1e-18)
    assert controller.last_receipt["inflation_max"] == pytest.approx(1.)


def test_vts_analysis_keeps_k_forecast_trajectories():
    shape = (2,2,2)
    snapshots = [TimeShiftSnapshot(offset,
        {"u":(np.array([-1.,1.])+offset/900)[:,None,None,None]*np.ones(shape)},
        -1800,-1800) for offset in (-900,0,900)]
    def observations(states):
        return [GriddedObs("z",np.full(shape,35.),10.,states["u"]+10,np.ones(shape,bool))]
    controller = SpreadRepairController(SpreadRepairConfig(adaptive=False,wind_std_ms=0.))
    increment = controller.analyze_time_shifted(snapshots,
        GridGeometry(1000.,1000.,np.array([0.,1000.])),
        LetkfConfig(Localization(3000.,2000.),("u",),rtps_alpha=0.),
        analysis_seconds=0,observation_builder=observations)
    assert increment["u"].shape == (2,)+shape
    assert controller.last_receipt["valid_time_shifting"]["covariance_samples"] == 6


def test_vts_recentering_preserves_water_mean_and_nonnegative_members():
    pooled = {"qr":np.array([0.,0.,0.,2.,0.,0.])[:,None,None,None],
              "auxiliary":np.arange(6.)[:,None,None,None]}
    out = central_time_posterior(pooled,{"qr":np.zeros_like(pooled["qr"])},
                               {"trajectory_members":2})
    assert out["qr"].min() >= 0
    np.testing.assert_allclose(out["qr"].mean(0),pooled["qr"].mean(0))
    np.testing.assert_array_equal(out["auxiliary"],pooled["auxiliary"][2:4])


def test_retained_snapshot_views_cannot_change_future_covariance():
    buffer = TimeShiftBuffer()
    for valid in (-900,0,900):
        buffer.retain(valid,{"u":np.ones((2,1,1,1))},
                      forecast_origin_seconds=-1800,observation_cutoff_seconds=-1800)
    slots = buffer.snapshots(0)
    slots[0].states["u"][:] = 100
    assert buffer.snapshots(0)[0].states["u"].max() == 1


def test_shifted_slots_must_share_forecast_origin_and_observation_cutoff():
    snapshots = [TimeShiftSnapshot(offset,{"u":np.ones((2,1,1,1))},
                                  -1800 if offset < 900 else -900,-1800)
                 for offset in (-900,0,900)]
    with pytest.raises(ValueError,match="common forecast origin"):
        time_shifted_covariance(snapshots,analysis_seconds=0)


def test_cycle_parser_exposes_selectable_spread_policy():
    from tools.da_cycle_prepared import build_parser
    args = build_parser().parse_args(["--prepared-root","fixture","--spread-repair","innovation",
        "--proof-sha256","fixture","--source-manifest-sha256","fixture",
        "--prepared-content-sha256","fixture","--physics-profile","fixture",
        "--run-seconds","3600","--history-interval-seconds","900","--out","fixture"])
    assert args.spread_repair == "innovation"
    assert args.spread_repair_z_threshold == 25.
    from tools.da_cycle_prepared import plan_radar_assimilation
    args.hydrometeors = True
    args.reflectivity_analysis = True
    cfg = plan_radar_assimilation(args,28,analysis_fields=("u","v","thp"),cwp=False)
    assert cfg.spread_repair == "innovation"
    assert replace(cfg,rtps_alpha=1.2).spread_repair == "innovation"
    assert cfg.spread_repair_z_threshold == 25.


def test_non_radar_filter_pass_keeps_the_baseline_solver():
    shape = (2,2,2)
    prior = {"u":np.array([-2.,-1.,1.,2.])[:,None,None,None]*np.ones(shape)}
    batch = GriddedObs("environment",np.full(shape,1.),1.,prior["u"],np.ones(shape,bool))
    cfg = LetkfConfig(Localization(3000.,2000.),("u",),rtps_alpha=0.)
    grid = GridGeometry(1000.,1000.,np.array([0.,1000.]))
    controller = SpreadRepairController(SpreadRepairConfig(adaptive=False))
    from gpuwm.da.letkf import analyze
    expected = analyze(prior,[batch],grid,cfg)
    actual = controller.analysis_runner()(prior,[batch],grid,cfg)
    np.testing.assert_array_equal(actual["u"],expected["u"])
    assert controller.last_receipt["applied"] is False


@pytest.mark.parametrize("kwargs", [{"policy":"unknown"}, {"horizontal_scale_cells":1.},
    {"temperature_std_k":-1.}, {"quadrature_points":2}, {"inflation_initial_sd":0.}])
def test_config_rejects_named_breakage(kwargs):
    with pytest.raises(ValueError):
        SpreadRepairConfig(**kwargs)
