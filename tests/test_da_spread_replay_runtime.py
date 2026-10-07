"""Replay runner contracts exercised through the real execution dispatcher.

CUDA capacity failure is injected before computation. The retry uses the
real CPU LETKF, so these tests require no card and belong on a box CPU.
They verify recovery routing and progress delivery, not CUDA qualification.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.da.letkf import (
    GridGeometry, GriddedObs, LetkfConfig, LetkfDiagnostics, Localization,
)
from gpuwm.da.obs_radar import REFLECTIVITY_NAME
from gpuwm.da.radar_assimilation import MemberStack, _execute_analysis
from gpuwm.da.spread_repair import SpreadRepairConfig, SpreadRepairController

# Register the existing CPU-only checkpoint/real-radar-file fixtures here.
# Their implementation stays with the radar adapter tests; this lane owns
# only the regression that exercises the spread runner's integration.
from test_radar_assimilation import hydro_grid, hydro_world  # noqa: F401


def _inputs():
    shape = (2, 2, 2)
    states = {
        member: {"qv": np.full(shape, value, dtype=np.float32)}
        for member, value in enumerate((0.009, 0.01, 0.011))
    }
    prior = {"qv": MemberStack("qv", states, tuple(states), shape)}
    batch = GriddedObs(
        name=REFLECTIVITY_NAME,
        values=np.zeros(shape),
        errors=3.0,
        simulated=np.zeros((3,) + shape),
        mask=np.zeros(shape, dtype=bool),
    )
    geometry = GridGeometry(
        dx_m=3000.0, dy_m=3000.0, heights_m=np.array([100.0, 300.0])
    )
    config = LetkfConfig(
        localization=Localization(horizontal_m=6000.0, vertical_m=500.0),
        analysis_fields=("qv",), rtps_alpha=0.0, host_workers=1,
    )
    return prior, [batch], geometry, config


@pytest.mark.parametrize("policy", ["off", "innovation"])
def test_spread_runner_retains_cuda_capacity_retry_and_progress(monkeypatch, policy):
    from gpuwm.da import radar_assimilation

    controller = SpreadRepairController(SpreadRepairConfig(
        policy=policy, adaptive=False, temperature_std_k=0.0,
        wind_std_ms=0.0, vapour_fraction_std=0.0,
    ))
    runner = controller.analysis_runner()
    prior, batches, geometry, config = _inputs()
    attempts = []
    released = []
    progress = []
    namespace = SimpleNamespace(
        get_default_memory_pool=lambda: SimpleNamespace(
            free_all_blocks=lambda: released.append(True)
        )
    )

    def attempt(solver, candidate_prior, candidate_batches, candidate_geometry,
                candidate_config, *, namespace, storage, supports_staging,
                progress, diagnostics=None, device_options=None):
        attempts.append(storage)
        if storage == "cuda-resident":
            raise MemoryError("injected resident allocation failure")
        assert storage == "host-staged-cuda"
        assert supports_staging
        diagnostics = diagnostics or LetkfDiagnostics()
        increments = solver(
            candidate_prior, candidate_batches, candidate_geometry,
            candidate_config, diagnostics, solve_namespace=np, progress=progress,
        )
        return increments, diagnostics, 0.0, 0.0

    monkeypatch.setattr(radar_assimilation, "_analysis_attempt", attempt)
    increments, _, _, _, storage, receipt = _execute_analysis(
        runner, prior, batches, geometry, config, namespace=namespace,
        device="cuda", progress=progress.append,
    )
    assert attempts == ["cuda-resident", "host-staged-cuda"]
    assert storage == "host-staged-cuda"
    assert released == [True]
    assert [row["status"] for row in receipt] == ["memory-failed", "computed"]
    np.testing.assert_array_equal(increments["qv"], np.zeros(prior["qv"].shape))
    assert any(row.get("phase") == "retry" for row in progress)
    assert any(
        row.get("phase") == "complete" and row.get("storage") == "host-staged-cuda"
        for row in progress
    )


@pytest.mark.parametrize("policy", ["off", "innovation"])
def test_precipitation_runner_delivers_real_letkf_progress(policy):
    controller = SpreadRepairController(SpreadRepairConfig(
        policy=policy, adaptive=False, temperature_std_k=0.0,
        wind_std_ms=0.0, vapour_fraction_std=0.0,
    ))
    prior, batches, geometry, config = _inputs()
    progress = []
    increments = controller.analysis_runner()(
        prior, batches, geometry, config, solve_namespace=np,
        progress=progress.append,
    )
    np.testing.assert_array_equal(increments["qv"], np.zeros(prior["qv"].shape))
    assert progress
    assert progress[-1]["phase"] == "complete"


def _moisture_inputs(values, simulated, observed, shape=(2, 2, 2)):
    prior = {"qv":np.stack([
        np.full(shape,value,dtype=np.float64) for value in values
    ])}
    batch = GriddedObs(
        name=REFLECTIVITY_NAME, values=np.full(shape,observed), errors=3.0,
        simulated=np.stack([np.full(shape,value) for value in simulated]),
        mask=np.ones(shape,dtype=bool),
    )
    geometry = GridGeometry(
        dx_m=3000.0, dy_m=3000.0,
        heights_m=np.arange(shape[0],dtype=np.float64)*200.0+100.0,
    )
    config = LetkfConfig(
        localization=Localization(horizontal_m=6000.0,vertical_m=500.0),
        analysis_fields=("qv",),rtps_alpha=0.0,host_workers=1,
    )
    return prior,batch,geometry,config


def test_deferred_maintenance_allows_caller_to_bound_negative_posterior():
    from gpuwm.da.positivity import apply_positivity

    prior,batch,geometry,config = _moisture_inputs(
        (0.0001,0.001,0.0019),(65.0,75.0,85.0),65.0,
    )
    controller = SpreadRepairController(SpreadRepairConfig(
        adaptive=False,temperature_std_k=0.0,wind_std_ms=0.0,
    ))
    increments = controller.analyze(
        prior,[batch],geometry,config,radar=batch,
        rebuild_observations="linearized",maintain_spread=False,
        solve_namespace=np,
    )
    assert (prior["qv"]+increments["qv"] < 0.0).any()
    assert controller.last_receipt["maintenance_applied"] is False

    # Exercise the native protocol too: its solve must also defer the
    # noise guard until the existing positivity policy has run.
    runner = controller.analysis_runner()
    increments = runner(prior,[batch],geometry,config,solve_namespace=np)
    bounded,_ = apply_positivity(
        prior,increments,policy="mean-preserving",fields=("qv",),
    )
    before = prior["qv"]+bounded["qv"]
    assert before.min() >= -1e-15
    finished = runner.finish(prior,bounded,solve_namespace=np)
    after = prior["qv"]+finished["qv"]
    assert after.min() >= -1e-15
    np.testing.assert_allclose(after.mean(axis=0),before.mean(axis=0),atol=1e-14,rtol=0.0)
    assert controller.last_receipt["maintenance_stage"] == "after-posterior-positivity"


def test_raw_dry_echo_gate_survives_conditioning_and_maintenance_draws_once(monkeypatch):
    from dataclasses import replace
    from gpuwm.da import spread_repair

    prior,raw,geometry,config = _moisture_inputs(
        (0.0025,0.003,0.0035),(0.0,0.0,0.0),20.0,shape=(6,3,3),
    )
    conditioned = replace(raw,simulated=np.full(raw.simulated.shape,10.0))
    controller = SpreadRepairController(SpreadRepairConfig(
        adaptive=False,temperature_std_k=0.0,wind_std_ms=0.0,
    ),seed=19)
    runner = controller.analysis_runner(gate_radar=raw)
    calls = []
    repair = spread_repair.additive_repair

    def counted(*args,**kwargs):
        calls.append(kwargs["cycle"])
        return repair(*args,**kwargs)

    monkeypatch.setattr(spread_repair,"additive_repair",counted)
    increments = runner(prior,[conditioned],geometry,config,solve_namespace=np)
    assert calls == []
    assert controller.last_receipt["missed_echo_cells"] == raw.values.size
    before = prior["qv"]+increments["qv"]
    finished = runner.finish(prior,increments,solve_namespace=np)
    after = prior["qv"]+finished["qv"]
    assert calls == [0]
    assert np.max(np.abs(after-before)) > 1e-9
    assert after.min() >= 0.0
    np.testing.assert_allclose(after.mean(axis=0),before.mean(axis=0),atol=1e-14,rtol=0.0)
    assert controller.last_receipt["gate_reflectivity"] == "raw"
    repeated = runner.finish(prior,finished,solve_namespace=np)
    assert repeated is finished
    assert calls == [0]
    assert controller.state.cycles == 1


def test_cropped_subsolve_does_not_advance_controller_or_maintain_spread():
    from gpuwm.da.letkf import analyze

    controller = SpreadRepairController(SpreadRepairConfig(
        adaptive=False,temperature_std_k=0.0,wind_std_ms=0.0,
    ))
    runner = controller.analysis_runner()
    prior,batches,geometry,config = _inputs()
    runner(prior,batches,geometry,config,solve_namespace=np)
    checkpoint = controller.state.checkpoint()
    assert runner.subsolve is analyze
    result = runner.subsolve(prior,batches,geometry,config,solve_namespace=np)
    np.testing.assert_array_equal(result["qv"],np.zeros(prior["qv"].shape))
    np.testing.assert_array_equal(controller.state.shape,checkpoint["shape"])
    np.testing.assert_array_equal(controller.state.scale,checkpoint["scale"])
    assert controller.state.cycles == checkpoint["cycles"]


def test_reconstructed_and_persistent_runners_share_time_based_noise():
    prior,raw,geometry,config = _moisture_inputs(
        (0.0029,0.003,0.0031),(0.0,0.0,0.0),20.0,shape=(6,3,3),
    )
    repair_config = SpreadRepairConfig(
        adaptive=False,temperature_std_k=0.0,wind_std_ms=0.0,
    )
    persistent = SpreadRepairController(repair_config,seed=37)
    runner = persistent.analysis_runner(gate_radar=raw)

    def slot(controller,run,valid_time):
        controller.set_analysis_time(valid_time)
        increments = run(prior,[raw],geometry,config,solve_namespace=np)
        return run.finish(prior,increments,solve_namespace=np)["qv"].copy()

    first = slot(persistent,runner,"2000-01-01T01:00:00Z")
    second = slot(persistent,runner,"2000-01-01T02:00:00Z")
    assert persistent.state.cycles == 2
    assert not np.array_equal(first,second)
    replay = SpreadRepairController(repair_config,seed=37)
    replay_result = slot(replay,replay.analysis_runner(gate_radar=raw),
                         "2000-01-01T02:00:00+00:00")
    np.testing.assert_array_equal(replay_result,second)
    assert replay.state.cycles == 1
    assert replay.last_receipt["noise_phase"] == persistent.last_receipt["noise_phase"]
    assert replay.last_receipt["noise_phase_source"] == "analysis-valid-time"

    # A different explicit timezone spelling of the same instant is the
    # same observation slot, independent of factory reconstruction.
    equivalent = SpreadRepairController(repair_config,seed=37)
    same = slot(equivalent,equivalent.analysis_runner(gate_radar=raw),
                "2000-01-01T03:00:00+01:00")
    np.testing.assert_array_equal(same,second)


@pytest.mark.parametrize("value", ["not-a-time","2000-01-01T01:00:00"])
def test_noise_time_rejects_invalid_or_ambiguous_slot_metadata(value):
    controller = SpreadRepairController()
    with pytest.raises(ValueError,match="valid time"):
        controller.set_analysis_time(value)


def test_absent_time_metadata_preserves_counter_based_noise_phase():
    prior,raw,geometry,config = _moisture_inputs(
        (0.0029,0.003,0.0031),(0.0,0.0,0.0),20.0,shape=(6,3,3),
    )
    controller = SpreadRepairController(SpreadRepairConfig(
        adaptive=False,temperature_std_k=0.0,wind_std_ms=0.0,
    ),seed=37)
    run = controller.analysis_runner(gate_radar=raw)
    first = run.finish(prior,run(prior,[raw],geometry,config),solve_namespace=np)
    assert controller.last_receipt["noise_phase"] == 0
    assert controller.last_receipt["noise_phase_source"] == "analysis-counter"
    controller.set_analysis_time(None)
    second = run.finish(prior,run(prior,[raw],geometry,config),solve_namespace=np)
    assert controller.last_receipt["noise_phase"] == 1
    assert not np.array_equal(first["qv"],second["qv"])


def test_native_factory_and_persistent_cycle_use_the_same_observation_slot(
        hydro_grid, hydro_world):
    """The native caller supplies time to both construction paths.

    A deterministic dry forward operator isolates the missed-echo repair:
    changing only the file's analysis time must change the draw, while
    reconstruction and a persistent cycle must agree for the same instant.
    """
    from gpuwm.da.obs_radar import read_document
    from gpuwm.da.radar_assimilation import (
        assimilate_radar_grid, member_background_checkpoint,
        read_checkpoint_state,
    )
    from test_radar_assimilation import _hydro_config

    checkpoints = {
        index: member_background_checkpoint(info["member_dir"])
        for index, info in hydro_world.member_states.items()
    }
    document = read_document(hydro_world.obs_path,expected_grid=hydro_grid)
    variables = dict(document["variables"])
    shape = variables["z_obs"].shape
    covered = hydro_world.interior
    for name in ("z_obs","z_max","z_mean"):
        variables[name] = np.where(covered,20.0,0.0).astype(np.float32)
    variables["z_mask"] = covered.astype(np.int8)
    variables["z_err"] = np.where(covered,3.0,0.0).astype(np.float32)
    variables["z_count"] = covered.astype(np.int32)
    document = dict(document,variables=variables)
    config = _hydro_config(
        solve_device="host",velocity=False,rtps_alpha=0.0,
        spread_repair="innovation",spread_repair_seed=37,
    )

    def dry_provider(member,state):
        return np.zeros(shape,dtype=np.float64)

    def slot(valid_time,runner=None):
        observed = dict(document,valid_time=valid_time)
        return assimilate_radar_grid(
            checkpoints,observed,hydro_grid,config,
            reflectivity_provider=dry_provider,analysis_runner=runner,
        )

    first,first_receipt = slot("2000-01-01T01:00:00Z")
    reconstructed,reconstructed_receipt = slot("2000-01-01T02:00:00Z")
    equivalent,equivalent_receipt = slot("2000-01-01T03:00:00+01:00")
    assert first_receipt["spread_repair"]["noise_phase_source"] == "analysis-valid-time"
    assert first_receipt["spread_repair"]["noise_phase"] != reconstructed_receipt["spread_repair"]["noise_phase"]
    assert equivalent_receipt["spread_repair"]["noise_phase"] == reconstructed_receipt["spread_repair"]["noise_phase"]
    assert first_receipt["spread_repair"]["missed_echo_cells"] == int(covered.sum())
    assert any(
        not np.array_equal(first[index]["qv"],reconstructed[index]["qv"])
        for index in checkpoints
    )
    for index in checkpoints:
        for name in reconstructed[index]:
            np.testing.assert_array_equal(
                equivalent[index][name],reconstructed[index][name],
            )

    pressure = tuple(
        read_checkpoint_state(checkpoints[index])["p"]
        for index in sorted(checkpoints)
    )
    persistent = SpreadRepairController(
        SpreadRepairConfig(adaptive=False),seed=37,
    )
    runner = persistent.analysis_runner(pressure=pressure)
    slot("2000-01-01T01:00:00Z",runner)
    cycled,cycled_receipt = slot("2000-01-01T02:00:00+00:00",runner)
    assert persistent.state.cycles == 2
    assert cycled_receipt["spread_repair"]["noise_phase"] == reconstructed_receipt["spread_repair"]["noise_phase"]
    for index in checkpoints:
        for name in reconstructed[index]:
            np.testing.assert_array_equal(
                cycled[index][name],reconstructed[index][name],
            )
