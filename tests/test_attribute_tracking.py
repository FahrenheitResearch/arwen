"""Native attribute following: scientific operators and live-store contracts."""
from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest

from gpuwm.core import storm_tracking as st
from gpuwm.core.attribute_tracking import (ATTRIBUTE_KEYS, ATTRIBUTE_UNITS,
    attribute_plane, validate_attribute_domains)
from gpuwm.core.nest_lifecycle import build_domain_follow_config
from gpuwm.core.streamed_state import CanonicalStoreState
from gpuwm.core.storm_track_writer import CSV_POSITION_COLUMNS, csv_columns


def config(**overrides):
    values = dict(field="attribute", attribute="theta", extremum="max",
                  reduction="column_max", threshold=301., search_margin_cells=10,
                  min_shift_cells=1, max_shift_cells=6, cooldown_seconds=20.)
    values.update(overrides)
    return st.FollowConfig(**values)


def state(volume=None):
    thp = np.zeros((3, 40, 50)) if volume is None else volume.copy()
    return SimpleNamespace(thp=thp, thb=np.full(thp.shape[0], 300.),
        nz=thp.shape[0], ny=thp.shape[1], nx=thp.shape[2])


def table(cfg):
    return {key: value for key, value in cfg.to_json().items() if key in st.FOLLOW_KEYS}


def test_registry_and_per_domain_roundtrip_preserve_exact_config():
    for name, units in ATTRIBUTE_UNITS.items():
        cfg = config(attribute=name, extremum="min", reduction="model_level", model_level=2)
        assert st.build_follow_config(table(cfg), "test") == cfg
        follower = build_domain_follow_config({**table(cfg), "cadence_seconds": 60.}, "test", grid_id=2)
        assert follower.tracker == cfg
        assert follower.to_json()["threshold_units"] == units
    legacy = st.FollowConfig(field="reflectivity", threshold=30., search_margin_cells=10,
        min_shift_cells=1, max_shift_cells=6, cooldown_seconds=20.)
    assert not set(legacy.to_json()) & ATTRIBUTE_KEYS
    assert st.build_follow_config(table(legacy), "test") == legacy


@pytest.mark.parametrize("change", [
    {"attribute": "rh"}, {"attribute": "thp"}, {"attribute": None},
    {"extremum": None}, {"extremum": "maximum"},
    {"reduction": None}, {"reduction": "surface"},
    {"model_level": 0}, {"reduction": "model_level"},
    {"reduction": "model_level", "model_level": -1},
    {"reduction": "model_level", "model_level": True},
    {"reduction": "model_level", "model_level": 1.5},
    {"field": "reflectivity"}, {"level_hpa": 0.}, {"fallback_threshold": 20.},
    {"threshold": float("nan")},
])
def test_config_refuses_unimplemented_or_ambiguous_requests(change):
    with pytest.raises(ValueError):
        config(**change)


@pytest.mark.parametrize("reduction", ["column_max", "column_min", "column_mean", "model_level"])
@pytest.mark.parametrize("attribute", list(ATTRIBUTE_UNITS))
def test_native_reduction_matches_independent_volume_reference(attribute, reduction):
    rng = np.random.default_rng(31)
    volume = rng.uniform(-2, 4, (3, 7, 9))
    live = state(volume)
    setattr(live, attribute, volume.copy())
    if attribute == "theta":
        live.thb = rng.uniform(290, 310, volume.shape)
        expected = live.thb + live.thp
    elif attribute == "w":
        live.w = rng.uniform(-3, 6, (4, 7, 9))
        expected = .5 * (live.w[:-1] + live.w[1:])
    else:
        expected = volume
    cfg = config(attribute=attribute, reduction=reduction,
                 model_level=1 if reduction == "model_level" else None)
    references = {"column_max": expected.max(axis=0), "column_min": expected.min(axis=0),
                  "column_mean": expected.mean(axis=0), "model_level": expected[1]}
    actual = attribute_plane(live, cfg)
    np.testing.assert_allclose(actual, references[reduction], rtol=1e-14, atol=1e-14)
    window = attribute_plane(live, cfg, window=(1, 5, 2, 7))
    np.testing.assert_array_equal(window[1:5, 2:7], actual[1:5, 2:7])
    assert np.isnan(window[:1]).all() and np.isnan(window[:, :2]).all()
    np.testing.assert_array_equal(live.thp, volume)


def test_scalar_theta_base_is_native_potential_temperature():
    live = state(np.arange(18.).reshape(3, 2, 3))
    live.thb = np.array([290., 305., 320.])
    np.testing.assert_array_equal(attribute_plane(live, config(reduction="column_mean")),
        (live.thp + live.thb[:, None, None]).mean(axis=0))


@pytest.mark.parametrize("reduction", ["column_max", "column_min", "column_mean"])
def test_nonfinite_levels_invalidate_column_and_empty_feature_holds(reduction):
    live = state()
    live.thp[0, 20, 25] = np.inf
    live.thp[1, 20, 26] = np.nan
    cfg = config(reduction=reduction)
    plane = attribute_plane(live, cfg)
    assert np.isnan(plane[20, 25:27]).all()
    assert st.StormTracker(cfg).locate(live, footprint(), 0.).found is None


def footprint():
    return st.NestFootprint(grid_id=2, i_parent_start=15, j_parent_start=10,
        child_nx=31, child_ny=31, parent_grid_ratio=3, parent_dx_m=1000.)


@pytest.mark.parametrize("direction,peak,threshold", [("max", 306., 301.), ("min", 294., 299.)])
def test_centroid_shift_signed_extremum_and_restart_cooldown(direction, peak, threshold):
    live = state()
    live.thp[:, 19, 25] = peak - 300.
    cfg = config(extremum=direction, threshold=threshold)
    tracker = st.StormTracker(cfg)
    fp = footprint()
    fix = tracker.locate(live, fp, 0.)
    assert fix.center_parent_ij == (25., 19.)
    assert fix.extremum == peak
    assert fix.evidence["extremum_kind"] == ("minimum" if direction == "min" else "maximum")
    assert fix.evidence["extremum_units"] == "K"
    assert fix.evidence["threshold_units"] == "K"
    assert tracker.desired_shift(live, fp, 0.) == (6, 5)
    resumed = st.StormTracker(st.build_follow_config(table(cfg), "restart"))
    resumed.restore_state(tracker.state_json())
    assert resumed.desired_shift(live, fp, 10.) is None
    assert resumed.desired_shift(live, fp, 21.) == (6, 5)


def test_minimum_refinement_preserves_direction_units_and_small_values():
    coarse = state()
    coarse.qv = np.full_like(coarse.thp, .01)
    coarse.qv[:, 19, 25] = .00321
    fine = state()
    fine.qv = np.full_like(fine.thp, .01)
    fine.qv[:, 20, 30] = .0012345
    cfg = config(attribute="qv", extremum="min", threshold=.005, refine_grid_id=3)
    source = st.RefinementSource(grid_id=3, state=fine, origin_i=0., origin_j=0.,
        scale_i=1., scale_j=1., edge_margin_cells=2, dx_m=1000.)
    fix = st.StormTracker(cfg).locate(coarse, footprint(), 0., refinement=source)
    assert fix.center_parent_ij == (30., 20.)
    assert fix.extremum == .00321
    assert fix.evidence["max_value"] == .00321
    assert fix.evidence["refinement"]["refine_extremum"] == .0012345
    assert fix.evidence["refinement"]["threshold_units"] == "kg/kg"


def test_live_store_drain_replaces_poisoned_attachment_mirror():
    live = state()
    live.thp.fill(9999.)
    store = {"state/thp": np.zeros_like(live.thp)}
    geography = {"setup/thb": np.full_like(live.thp, 300.)}
    live.thb = np.full_like(live.thp, -9999.)
    calls = []
    def drain():
        calls.append("drained")
        if "state/thp" in store:
            store["state/thp"][:, 19, 25] = 5.
    live._streamed_domain = SimpleNamespace(store=store, _geography=geography,
        _run=SimpleNamespace(drain=drain))
    live._streamed_store = store
    fix = st.StormTracker(config()).locate(live, footprint(), 0.)
    assert calls and fix.center_parent_ij == (25., 19.) and fix.extremum == 305.
    assert np.all(live.thp == 9999.)
    del store["state/thp"]
    with pytest.raises(st.TrackerRefusal, match="absent"):
        st.planes_for(live, config())


def test_canonical_host_uses_full_arrays_and_never_resident_methods():
    slab = np.full((3, 2, 50), 9999.)
    template = SimpleNamespace(thp=slab, thb=slab.copy(),
        total_theta=lambda: pytest.fail("resident method used"))
    volume = np.zeros((3, 40, 50))
    volume[:, 19, 25] = 5.
    live = CanonicalStoreState(template, SimpleNamespace(nx=50, ny=40, nz=3),
        store={"state/thp": volume}, geography={"setup/thb": np.full_like(volume, 300.)},
        scalars={}, inventory={"state/thp": template.thp},
        geography_inventory={"setup/thb": template.thb})
    fix = st.StormTracker(config()).locate(live, footprint(), 0.)
    assert fix.center_parent_ij == (25., 19.) and fix.extremum == 305.


@pytest.mark.parametrize("bad", ["missing", "shape", "level", "base"])
def test_invalid_live_state_refuses_before_tracking(bad):
    live, cfg = state(), config()
    if bad == "missing":
        cfg = config(attribute="qr")
    elif bad == "shape":
        live.thp = live.thp[0]
    elif bad == "level":
        cfg = config(reduction="model_level", model_level=3)
    else:
        live.thb = np.zeros(2)
    with pytest.raises(st.TrackerRefusal):
        st.planes_for(live, cfg)


def test_native_domain_admission_and_position_only_writer():
    parent = SimpleNamespace(grid_id=1, parent_id=0, follow=None,
        run=SimpleNamespace(nz=3, moist=False))
    mover = SimpleNamespace(grid_id=2, parent_id=1, follow=None,
        run=SimpleNamespace(nz=3, moist=True))
    relocation = SimpleNamespace(grid_id=2, follow=config(attribute="qv"))
    with pytest.raises(ValueError, match="moist = true"):
        validate_attribute_domains([parent, mover], relocation)
    relocation.follow = config(reduction="model_level", model_level=3)
    with pytest.raises(ValueError, match="outside source"):
        validate_attribute_domains([parent, mover], relocation)
    relocation.follow = config()
    validate_attribute_domains([parent, mover], relocation)
    assert csv_columns(tracked_field="attribute") == CSV_POSITION_COLUMNS


@pytest.mark.parametrize("location", ["relocation", "domain"])
def test_actual_loader_admits_attribute_and_refuses_unavailable_levels(tmp_path, location):
    import tomllib
    from gpuwm.branch import emit_experiment_toml
    from gpuwm.experiment import load_experiment, experiment_config_document
    from gpuwm.core.model import restart_identity_payload
    from test_prepared_domain_tree_forecast import _write_two_domain_config
    path = _write_two_domain_config(tmp_path)
    raw = tomllib.loads(path.read_text())
    follow = table(config(attribute="qv", threshold=.01, reduction="model_level", model_level=7))
    if location == "relocation":
        raw["relocation"] = dict(enabled=True, grid_id=2, cadence_seconds=60., follow=follow)
    else:
        raw["domain"][1]["follow"] = dict(follow, cadence_seconds=60.)
    path.write_text(emit_experiment_toml(raw), encoding="utf-8")
    accepted = load_experiment(path)
    document = experiment_config_document(accepted)
    identity = restart_identity_payload(accepted)
    assert "qv" in str(document) and "model_level" in str(identity)
    follow["model_level"] = 8
    if location == "domain":
        raw["domain"][1]["follow"]["model_level"] = 8
    path.write_text(emit_experiment_toml(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="outside source"):
        load_experiment(path)
    # Changing the attribute selector also changes the actual restart identity.
    if location == "relocation":
        other = replace(accepted, relocation=replace(accepted.relocation,
            follow=replace(accepted.relocation.follow, attribute="qr")))
    else:
        child = accepted.domains[1]
        other = replace(accepted, domains=(accepted.root, replace(child,
            follow=replace(child.follow, tracker=replace(child.follow.tracker, attribute="qr")))))
    assert restart_identity_payload(other) != identity


def test_legacy_public_config_identity_has_no_new_null_attribute_keys():
    from gpuwm.experiment import _public_config_value
    cfg = st.FollowConfig(field="reflectivity", threshold=30., search_margin_cells=10,
        min_shift_cells=1, max_shift_cells=6, cooldown_seconds=20.)
    assert _public_config_value(cfg) == dict(field="reflectivity", threshold=30.,
        search_margin_cells=10, min_shift_cells=1, max_shift_cells=6, cooldown_seconds=20.,
        fallback_threshold=None, level_hpa=None, refine_grid_id=None, radius_km=50., report_level_hpa=())


def test_attribute_track_admission_has_position_only_semantics():
    from test_relocation_track_config import _build, _raw, _domains
    raw = _raw(follow=table(config(attribute="w", threshold=1.)),
               track={"path": "attribute.csv", "interval_seconds": 60.})
    # No wind diagnostic or history-cadence dependency exists for a position row.
    accepted = _build(raw, domains=_domains(sfclay=0))
    assert accepted.track.interval_seconds == 60.
    raw["relocation"]["track"]["output_level"] = [850.]
    with pytest.raises(ValueError, match="refuses output_level"):
        _build(raw, domains=_domains(sfclay=0))
