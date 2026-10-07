"""Actual Rust sidecar writes, accepted-generation joins and adaptive retries."""
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pytest

from gpuwm.da.letkf import GridGeometry, GriddedObs, LetkfConfig, Localization
from gpuwm.da.obs_radar import REFLECTIVITY_NAME
from gpuwm.da.spread_repair import SpreadRepairConfig, SpreadRepairController
from gpuwm.da.spread_restart import (
    SpreadRestartError, capture, clock_valid_time, fork_controller, read_sidecar)
from tools import da_ensemble_state as generations
from test_da_ensemble_state import identity, sets_for


def setup_controller(*, adaptive=True):
    config = SpreadRepairConfig(adaptive=adaptive, wind_std_ms=0., top_taper_levels=0)
    return SpreadRepairController(config, seed=7)


def solve(controller, *, finish=True, raw=None, used=None):
    shape = (4, 6, 8)
    x = np.array([-2., -1., 1., 2.])[:, None, None, None] * np.ones(shape)
    prior = {"u": x}
    radar = GriddedObs(REFLECTIVITY_NAME, np.full(shape, 35.), 10.,
                      x + 16., np.ones(shape, bool))
    controller.set_analysis_time("2026-01-01T00:15:00Z")
    runner = controller.analysis_runner(gate_radar=radar if raw is None else raw)
    grid = GridGeometry(1000., 1000., np.arange(4.) * 1000.)
    cfg = LetkfConfig(Localization(2000., 2000.), ("u",), rtps_alpha=0.)
    result = runner(prior, [radar if used is None else used], grid, cfg)
    if finish:
        result = runner.finish(prior, result)
    return result, runner, prior, radar


def save(tmp_path, controller, *, name="generation"):
    ident = identity(4)
    record = capture(controller, identity=ident.to_payload(),
        grid_identity_sha256="a" * 64, valid_time="2026-01-01T00:15:00Z",
        elapsed_seconds=900., leg_number=0)
    folder = tmp_path / name
    manifest = generations.write_generation(folder, identity=ident,
        elapsed_seconds=900., leg_number=0, restarts=sets_for(tmp_path, 4),
        pending={}, spread_repair=record)
    return folder, manifest, ident


def test_named_flag_reaches_real_config_without_off_metadata():
    from tools.da_cycle_prepared import build_parser, plan_radar_assimilation
    argv = ["--prepared-root", "fixture", "--spread-repair", "innovation",
        "--proof-sha256", "fixture", "--source-manifest-sha256", "fixture",
        "--prepared-content-sha256", "fixture", "--physics-profile", "fixture",
        "--run-seconds", "3600", "--history-interval-seconds", "900", "--out", "fixture"]
    baseline = build_parser().parse_args(argv)
    assert "spread_repair_adaptive" not in vars(baseline)
    selected = build_parser().parse_args(argv + ["--spread-repair-adaptive"])
    for args in (baseline, selected):
        args.hydrometeors = True
        args.reflectivity_analysis = True
    assert not plan_radar_assimilation(baseline, 28,
        analysis_fields=("u", "v", "thp"), cwp=False).spread_repair_adaptive
    assert plan_radar_assimilation(selected, 28,
        analysis_fields=("u", "v", "thp"), cwp=False).spread_repair_adaptive


def test_actual_native_generation_roundtrip_reproduces_next_analysis(tmp_path):
    from gpuwm.ingest.prepared_writer import native_writer
    assert native_writer() is not None, "this integration test requires the current Rust writer"
    continuous = setup_controller()
    solve(continuous)
    folder, manifest, ident = save(tmp_path, continuous)
    _, _, accepted = generations.read_generation(folder, ident)
    state, metadata = read_sidecar(folder, accepted, identity=ident.to_payload(),
        config=continuous.config, seed=7, grid_identity_sha256="a" * 64,
        valid_time="2026-01-01T00:15:00Z")
    resumed = SpreadRepairController(continuous.config, seed=7, state=state)
    np.testing.assert_array_equal(resumed.state.shape, continuous.state.shape)
    np.testing.assert_array_equal(resumed.state.scale, continuous.state.scale)
    assert metadata["cycles"] == 1
    expected, *_ = solve(continuous)
    actual, *_ = solve(resumed)
    np.testing.assert_array_equal(expected["u"], actual["u"])
    np.testing.assert_array_equal(resumed.state.shape, continuous.state.shape)
    np.testing.assert_array_equal(resumed.state.scale, continuous.state.scale)
    assert resumed.state.cycles == continuous.state.cycles == 2


def test_accepted_manifest_does_not_commit_over_failed_native_sidecar(tmp_path, monkeypatch):
    from gpuwm.ingest import prepared_writer
    controller = setup_controller()
    solve(controller)
    def fail(*args, **kwargs):
        raise OSError("intentional native array write failure")
    monkeypatch.setattr(prepared_writer, "write_arrays", fail)
    with pytest.raises(OSError, match="intentional"):
        save(tmp_path, controller)
    assert not generations.manifest_path(tmp_path / "generation").exists()


@pytest.mark.parametrize("change,match", [
    ({"seed": 8}, "configuration/seed changed"),
    ({"grid_identity_sha256": "b" * 64}, "target grid changed"),
    ({"valid_time": "2026-01-01T00:30:00Z"}, "valid UTC differs"),
    ({"identity": identity(4, prepared_content_sha256="d" * 64).to_payload()}, "grid/prepared identity"),
])
def test_resume_refuses_changed_meaning_without_reset(tmp_path, change, match):
    controller = setup_controller(); solve(controller)
    folder, manifest, ident = save(tmp_path, controller)
    args = dict(identity=ident.to_payload(), config=controller.config, seed=7,
        grid_identity_sha256="a" * 64, valid_time="2026-01-01T00:15:00Z")
    args.update(change)
    with pytest.raises(SpreadRestartError, match=match):
        read_sidecar(folder, manifest, **args)


def test_missing_or_corrupted_posterior_is_a_named_refusal(tmp_path):
    controller = setup_controller(); solve(controller)
    folder, manifest, ident = save(tmp_path, controller)
    legacy = dict(manifest); legacy.pop("spread_repair")
    with pytest.raises(SpreadRestartError, match="requires its saved inflation state"):
        read_sidecar(folder, legacy, identity=ident.to_payload())
    field = folder / "spread-repair/scale.npy"
    payload = bytearray(field.read_bytes()); payload[-1] ^= 1; field.write_bytes(payload)
    with pytest.raises(SpreadRestartError, match="scale hash changed"):
        generations.read_generation(folder, ident)


@pytest.mark.parametrize("reference", [[], "legacy", {"directory": "../other", "metadata": "state.json"}])
def test_changed_sidecar_reference_is_a_named_refusal(tmp_path, reference):
    with pytest.raises(SpreadRestartError, match="owned generation sidecar"):
        read_sidecar(tmp_path, {"spread_repair": reference}, identity=identity(4).to_payload())


def test_unselected_generation_manifest_bytes_are_unchanged(tmp_path, monkeypatch):
    class Clock:
        @staticmethod
        def now(tz):
            return datetime(2026, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(generations, "datetime", Clock)
    ident = identity(1); restarts = sets_for(tmp_path, 1)
    args = dict(identity=ident, elapsed_seconds=900., leg_number=0,
        restarts=restarts, pending={})
    a = tmp_path / "default"; b = tmp_path / "explicit-none"
    generations.write_generation(a, **args)
    generations.write_generation(b, **args, spread_repair=None)
    assert generations.manifest_path(a).read_bytes() == generations.manifest_path(b).read_bytes()
    assert "spread_repair" not in generations.read_manifest(a)
    assert not (a / "spread-repair").exists()


def test_adaptive_likelihood_uses_final_solver_batch_with_raw_echo_gates(monkeypatch):
    from gpuwm.da import spread_repair as module
    shape = (4, 6, 8)
    x = np.array([-2., -1., 1., 2.])[:, None, None, None] * np.ones(shape)
    raw = GriddedObs(REFLECTIVITY_NAME, np.full(shape, 35.), 2., x + 16., np.ones(shape, bool))
    mask = raw.mask.copy(); mask[..., 0] = False
    used = replace(raw, errors=np.full(shape, 40.),
                   simulated=np.maximum(raw.simulated, 15.), mask=mask)
    observed = []
    original = module.update_adaptive_inflation
    def watch(state, z, hx, errors, gate, config, **kwargs):
        observed.append((z.copy(), hx.copy(), np.asarray(errors).copy(), gate.copy()))
        return original(state, z, hx, errors, gate, config, **kwargs)
    monkeypatch.setattr(module, "update_adaptive_inflation", watch)
    controller = setup_controller()
    solve(controller, raw=raw, used=used)
    z, hx, errors, gate = observed[0]
    np.testing.assert_array_equal(z, used.values)
    np.testing.assert_array_equal(hx, used.simulated)
    np.testing.assert_array_equal(errors, used.errors)
    np.testing.assert_array_equal(gate, used.mask)
    assert controller.last_receipt["observed_echo_cells"] == int(raw.mask.sum())


def test_adaptive_candidate_waits_for_finish_and_capacity_retry_is_identity():
    controller = setup_controller()
    first, _, _, _ = solve(controller, finish=False)
    assert controller.state is None
    repeated, runner, prior, _ = solve(controller, finish=False)
    np.testing.assert_array_equal(first["u"], repeated["u"])
    runner.finish(prior, repeated)
    assert controller.state.cycles == 1
    saved = controller.state.checkpoint()
    runner.finish(prior, repeated)
    assert controller.state.cycles == 1
    np.testing.assert_array_equal(controller.state.shape, saved["shape"])


def test_verification_fork_does_not_advance_carried_posterior():
    controller = setup_controller(); solve(controller)
    before = controller.state.checkpoint()
    verification = fork_controller(controller)
    solve(verification)
    assert verification.state.cycles == 2 and controller.state.cycles == 1
    np.testing.assert_array_equal(controller.state.shape, before["shape"])
    np.testing.assert_array_equal(controller.state.scale, before["scale"])


def test_engine_utc_clock_preserves_explicit_zone_and_rejects_ambiguous_sidecar_time(tmp_path):
    assert clock_valid_time(datetime(2026, 1, 1), 900.) == "2026-01-01T00:15:00.000000+00:00"
    controller = setup_controller()
    with pytest.raises(SpreadRestartError, match="explicit UTC"):
        capture(controller, identity=identity(4).to_payload(), grid_identity_sha256="a" * 64,
            valid_time="2026-01-01T00:15:00", elapsed_seconds=900., leg_number=0)
