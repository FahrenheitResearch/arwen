"""warm-kernels and the go chain compile a configuration's static kernel plan on every core.

Breakage these tests prevent: on a fresh box (no recorded kernel plan) each
`gpuwm warm-kernels` profile worker compiled its kernels one after another at
100 % of one core for 60 to 120 s while the other cores sat idle (173 s for
161 kernels, 3 workers, 48 cores, RTX PRO 6000 box, 2026-10-17); and a first
`gpuwm go` left its ~30 s fetch-and-prepare window unused while step 1 still
carried the compiles.  CPU only: the plan pool, loaders and processes are
stood in for.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
from types import SimpleNamespace

from gpuwm import kernel_plan, kernel_prewarm, warm_kernels


def _plan_with_pool(monkeypatch, tag="t"):
    submitted = []
    pool = SimpleNamespace(submit=lambda job, *args: submitted.append((job, args)))
    monkeypatch.setattr(kernel_plan, "_state", dict(kernel_plan._state, installed=True, tag=tag, pool=pool))
    monkeypatch.setattr(kernel_plan, "submit", lambda job, *args: submitted.append((job, args)))
    monkeypatch.setattr(kernel_prewarm, "_static_started", set())
    return submitted


def test_submit_static_submits_one_capture_per_module_once_per_tag(monkeypatch):
    submitted = _plan_with_pool(monkeypatch, tag="one")
    jobs = [("noah", "factory-a", ("noah",)), ("smag2d[GPUWM_SMAG_INDEX32=1]", "factory-b", ("smag2d", ()))]
    monkeypatch.setattr(kernel_prewarm, "static_jobs", lambda experiment: jobs)
    assert kernel_prewarm.submit_static(object()) == 2
    assert [(job, args) for job, args in submitted] == [
        (kernel_plan.ahead, ("factory-a", "noah")), (kernel_plan.ahead, ("factory-b", "smag2d", ()))]
    assert kernel_prewarm.submit_static(object()) == 0  # the same tag: already submitted
    kernel_plan._state["tag"] = "two"  # a worker rebinds a second profile
    assert kernel_prewarm.submit_static(object()) == 2


def test_submit_static_needs_an_open_pool(monkeypatch):
    _plan_with_pool(monkeypatch)
    kernel_plan._state["pool"] = None
    monkeypatch.setattr(kernel_prewarm, "static_jobs", lambda experiment: [("x", "f", ("x",))])
    assert kernel_prewarm.submit_static(object()) == 0


def test_single_domain_experiment_prices_the_profile_selectors():
    run = SimpleNamespace(mp_physics=10, cu_physics=1, bl_pbl_physics=1, sf_sfclay_physics=1,
                          sf_surface_physics=2, ra_physics=4, ra_lw_physics=4, ra_sw_physics=4,
                          ysu_version="wrf_480", run_seconds=24.0)
    experiment = kernel_prewarm.single_domain_experiment(run)
    assert experiment.domains[0].run is run and experiment.domains[0].grid_id == 1
    assert experiment.domains[0].history_interval_s == 0.0 and experiment.run_seconds == 24.0


def test_each_profile_binds_its_own_plan_and_submits_its_static_jobs(monkeypatch):
    calls = []
    state = {"installed": False, "tag": None}

    def install(tag_text=None, **_):
        state.update(installed=True, tag=tag_text); calls.append(("install", tag_text)); return True

    def rebind(tag_text):
        state["tag"] = tag_text; calls.append(("rebind", tag_text)); return True

    monkeypatch.setattr(kernel_plan, "_state", state)
    monkeypatch.setattr(kernel_plan, "install", install)
    monkeypatch.setattr(kernel_plan, "rebind", rebind)
    monkeypatch.setattr(kernel_plan, "start", lambda: calls.append(("start",)))
    monkeypatch.setattr(kernel_plan, "close", lambda: calls.append(("close",)))
    monkeypatch.setattr(warm_kernels, "_profile_config", lambda profile, levels: SimpleNamespace(profile=profile))
    monkeypatch.setattr(kernel_prewarm, "single_domain_experiment", lambda run: ("exp", run.profile))
    monkeypatch.setattr(kernel_prewarm, "submit_static", lambda experiment: calls.append(("static", experiment)) or 3)
    assert warm_kernels._bind_profile_plan("p1", 40)
    assert warm_kernels._bind_profile_plan("p2", 40)
    assert calls == [("install", "warm-kernels p1 levels=40"), ("start",), ("static", ("exp", "p1")), ("close",),
                     ("rebind", "warm-kernels p2 levels=40"), ("start",), ("static", ("exp", "p2")), ("close",)]


def test_the_in_process_pass_and_the_worker_both_bind_the_plan():
    from pathlib import Path

    text = Path(warm_kernels.__file__).read_text(encoding="utf-8")
    worker = text.index("def _profile_worker(")
    serial = text.index("for profile in (() if workers > 1 else profiles):")
    assert text.index("_bind_profile_plan(profile, levels)", worker) < text.index("_run_profile(profile, levels)", worker)
    assert text.index("_bind_profile_plan(profile, levels)", serial) < text.index("_run_profile(profile, levels)", serial)


def _go_args(**overrides):
    args = argparse.Namespace(dry_run=False, explain=False)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_go_spawns_the_kernel_prewarm_and_joins_it_before_the_forecast(monkeypatch, tmp_path):
    from gpuwm import go_cli

    started = []

    class _Process:
        def __init__(self, command, **kwargs):
            started.append((command, kwargs)); self.waited = None

        def wait(self, timeout=None):
            self.waited = timeout

    monkeypatch.setattr(go_cli, "_KERNEL_PREWARM", {})
    monkeypatch.setattr(go_cli, "explain_enabled", lambda args: False)
    monkeypatch.setattr(subprocess, "Popen", _Process)
    monkeypatch.setattr("gpuwm.local_gpu.no_local_gpu", lambda: False)
    monkeypatch.delenv(kernel_plan.ENVIRONMENT, raising=False)
    plan = {"config": tmp_path / "d.toml", "root": tmp_path, "profile": "morrison-mp10"}
    before = dict(plan)
    assert go_cli.SpeedHooks.warm_kernels_spawn(_go_args(), plan) is None
    assert plan == before
    (command, kwargs), = started
    assert command == [sys.executable, "-m", "gpuwm.kernel_prewarm", str(tmp_path / "d.toml"),
                       "--physics-profile", "morrison-mp10"]
    assert kwargs["stdin"] is subprocess.DEVNULL and (tmp_path / "kernel-prewarm.log").exists()
    process = go_cli._KERNEL_PREWARM[str(tmp_path)][0]
    go_cli._join_kernel_prewarm(plan)
    assert process.waited == go_cli._KERNEL_PREWARM_WAIT_S and go_cli._KERNEL_PREWARM == {}
    go_cli._join_kernel_prewarm(plan)  # nothing to join a second time


def test_go_does_not_spawn_for_a_dry_run_or_without_a_card(monkeypatch, tmp_path):
    from gpuwm import go_cli

    monkeypatch.setattr(go_cli, "_KERNEL_PREWARM", {})
    monkeypatch.setattr(go_cli, "explain_enabled", lambda args: False)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("spawned")))
    plan = {"config": tmp_path / "d.toml", "root": tmp_path}
    assert go_cli.SpeedHooks.warm_kernels_spawn(_go_args(dry_run=True), plan) is None
    monkeypatch.setattr("gpuwm.local_gpu.no_local_gpu", lambda: True)
    assert go_cli.SpeedHooks.warm_kernels_spawn(_go_args(), plan) is None
    monkeypatch.setattr("gpuwm.local_gpu.no_local_gpu", lambda: False)
    monkeypatch.setenv(kernel_plan.ENVIRONMENT, "off")
    assert go_cli.SpeedHooks.warm_kernels_spawn(_go_args(), plan) is None
    assert go_cli._KERNEL_PREWARM == {}


def test_warm_static_compiles_ahead_then_loads_each_module_for_its_disk_entry(monkeypatch):
    order = []
    monkeypatch.setattr(kernel_plan, "install", lambda tag_text=None: order.append("install") or True)
    monkeypatch.setattr(kernel_plan, "start", lambda: order.append("start") or {})
    monkeypatch.setattr(kernel_plan, "close", lambda: order.append("close"))
    monkeypatch.setattr(kernel_plan, "receipt", lambda: {"r": 1})
    monkeypatch.setattr(kernel_prewarm, "plan_tag", lambda experiment: "tag")
    monkeypatch.setattr(kernel_prewarm, "submit_static", lambda experiment: order.append("static") or 2)
    jobs = [("a", lambda name: order.append(("load", name)), ("a",)),
            ("b", lambda name: (_ for _ in ()).throw(ValueError("refused")), ("b",))]
    monkeypatch.setattr(kernel_prewarm, "static_jobs", lambda experiment: jobs)
    receipt = kernel_prewarm.warm_static(object())
    assert order == ["install", "start", "static", "close", ("load", "a")]
    assert receipt["modules"] == 2 and receipt["loaded"] == 1
    assert receipt["errors"] == {"b": "ValueError: refused"} and receipt["plan"] == {"r": 1}
    assert threading.current_thread() is threading.main_thread()


def test_registered_moving_tree_go_reaches_prewarm_before_fetch(monkeypatch, tmp_path):
    """R11's registered-source route bypassed the prepared-go hook entirely."""
    from gpuwm import capabilities, go_cli, runplan
    from gpuwm.cli import main
    from gpuwm.stage_timing import bind_ledger

    config = tmp_path / "moving.toml"
    assert main(["domain", "--point=50,8", "--card", "16gb", "--ladder", "12-3",
                 "--source", "icon-eu", "--cycle", "2026-10-17T15", "--hours", "1",
                 "--out", str(config)]) == 0
    with config.open("a") as stream:
        stream.write('\n[relocation]\nenabled = true\ngrid_id = 2\n'
                     'move = [{at_seconds = 60.0, di_parent_cells = 1, dj_parent_cells = 0}]\n')
    for name in ("require_for_command", "require"):
        monkeypatch.setattr(capabilities, name, lambda *a, **k: None)
    monkeypatch.setattr(go_cli, "_require_forecast_device", lambda: None)
    monkeypatch.setattr(go_cli, "geography_refusal", lambda *a, **k: None)
    monkeypatch.setattr(go_cli, "memory_gate", lambda *a, **k: {
        "verdict": "fits", "refuse": False, "warn": False})
    monkeypatch.setattr(go_cli, "_KERNEL_PREWARM", {})
    monkeypatch.setattr("gpuwm.local_gpu.no_local_gpu", lambda: False)
    monkeypatch.delenv(kernel_plan.ENVIRONMENT, raising=False)
    order = []

    class Process:
        pid = 123

        def __init__(self, command, **kwargs):
            assert command[3] == str(config)
            assert (tmp_path / "go").is_dir()
            order.append("spawn")

        def wait(self, timeout=None):
            order.append("join")
            return 0

    monkeypatch.setattr(subprocess, "Popen", Process)

    def execute(plan, *, events):
        # The real executor requires an empty launch before accepting it.
        assert not (plan.run_dir / "kernel-prewarm.log").exists()
        bind_ledger(plan.run_dir)
        events.emit("plan_accepted")
        events.emit("stage_started", stage="fetch")
        order.append("fetch")
        events.emit("stage_started", stage="prepare")
        events.emit("stage_started", stage="forecast")
        order.append("forecast")
        return 0

    monkeypatch.setattr(runplan, "execute_plan", execute)
    assert main(["go", str(config), "--outdir", str(tmp_path / "go"),
                 "--run-stamp", "off", "--products", "none"]) == 0
    assert order == ["spawn", "fetch", "join", "forecast"]
    assert go_cli._KERNEL_PREWARM == {}


def test_go_failed_spawn_names_reason_in_terminal_and_timeline(monkeypatch, tmp_path, capsys):
    import json
    from gpuwm import go_cli

    monkeypatch.setattr(go_cli, "explain_enabled", lambda args: False)
    monkeypatch.setattr("gpuwm.local_gpu.no_local_gpu", lambda: False)
    monkeypatch.delenv(kernel_plan.ENVIRONMENT, raising=False)

    def fail(*a, **k):
        raise OSError("process unavailable")

    monkeypatch.setattr(subprocess, "Popen", fail)
    go_cli.SpeedHooks.warm_kernels_spawn(_go_args(), {"config": "x.toml", "root": tmp_path})
    assert "note: kernel pre-warm not started: OSError: process unavailable" in capsys.readouterr().out
    row, = [json.loads(line) for line in (tmp_path / "timeline.jsonl").read_text().splitlines()]
    assert row["stage"] == "prep.kernel_prewarm"
    assert row["attrs"]["status"] == "not started"
    assert row["attrs"]["reason"] == "OSError: process unavailable"


def test_static_jobs_cover_the_moving_tree_forecast_compile_events():
    """The cold R11 step-1 event labels, including the inline units and RTE tiers."""
    from dataclasses import replace
    from gpuwm.config import RunConfig

    run = RunConfig(nx=50, ny=50, dx=12000, dy=12000, ztop=20000, dt=60,
                    run_seconds=120, nz=49, mp_physics=10, cu_physics=1, bl_pbl_physics=1,
                    sf_sfclay_physics=91, sf_surface_physics=2, ra_physics=0,
                    ra_lw_physics=4, ra_sw_physics=4, ra_rrtmg_variant="rte-rrtmgp",
                    diff_6th_opt=2, km_opt=4, nwp_diagnostics=1, ysu_version="wrf_480")
    experiment = SimpleNamespace(
        domains=(SimpleNamespace(run=run, grid_id=1, history_interval_s=60),
                 SimpleNamespace(run=replace(run, cu_physics=0), grid_id=2, history_interval_s=60)),
        vertical=SimpleNamespace(p_top=5000.0), run_seconds=120.0)
    jobs = kernel_prewarm.static_jobs(experiment)
    labels = {label for label, _, _ in jobs}
    assert jobs[0][0] == "sfclay"  # an 8 s serial tail when queued behind short units
    assert next(factory for label, factory, _ in jobs if label == "kf") is kernel_prewarm._warm_kf
    assert {"real_init", "real_init_math"} <= labels  # root initialization's cold compile
    # Loader unit labels from events.jsonl, normalized by removing its
    # gpuwm.core.kernels prefix. Inline factories keep their own identities.
    observed = {
        "health", "rk_bookkeeping", "rrtmgp_validation", "rrtmgp_cloud", "rrtmgp_gas",
        "rrtmgp_mcica", "rrtmgp_rte", "sfclay", "noah", "ysu_480", "ysu_validation",
        "kf", "kf_validation", "smag2d", "smag2d[GPUWM_SMAG_INDEX32=1]", "diff6",
        "diff6_seam", "dycore", "acoustic", "bandwidth_glue", "advection", "coriolis_map",
        "held_heating", "openbc", "lbc_state", "lbc_flow", "pd_advection", "surface_w",
        "uh_diag", "microphysics_validation", "morrison", "rrtmg_legacy_solar",
        "gpuwm.core.tendency_coupling", "gpuwm.core.noah_forcing", "gpuwm.core.noah_sfcdiags",
        "gpuwm.core.cumulus_clock:advance", "gpuwm.core.cumulus_clock:clear",
        "gpuwm.core.noahmp_slab_libm:noahmp_libm_slab",
    }
    assert observed <= labels, observed - labels
    tiers = {args[1] for label, _, args in jobs if label.startswith("rrtmgp_rte[")}
    assert tiers == {(("RRTMGP_MAX_LAYERS", 62),), (("RRTMGP_MAX_LAYERS", 50),)}
