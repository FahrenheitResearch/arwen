"""Targeted tests for the sequential single-GPU ensemble engine.

The orchestration properties -- seed derivation, manifest atomicity,
refusal of a completed member, resume, and byte-identity given a seed --
are exercised on the engine's real code with a member kernel that is a
state, not a model: the perturbation hook, the increment applier, and
the state hash under test are the shipped ones.

The end-to-end tiny-domain run lives in
``test_ensemble_tiny_domain_gpu.py`` and needs a device and staged data.
"""

from __future__ import annotations

import json
import types

import numpy as np
import pytest

from gpuwm.ensemble.bench import bench_rows, format_table
from gpuwm.ensemble.config import load_ensemble_config
from gpuwm.ensemble.cycle import run_cycles
from gpuwm.ensemble.engine import (
    MemberAlreadyCompleteError, member_seeds, prepare_ensemble, run_ensemble,
)
from gpuwm.ensemble.increments import (
    apply_increments, apply_increments_to_checkpoint,
)
from gpuwm.ensemble.manifest import (
    CYCLE_MANIFEST_SCHEMA, ENSEMBLE_MANIFEST_NAME, ENSEMBLE_MANIFEST_SCHEMA,
    member_directory_name, read_manifest, write_manifest_atomically,
)
from gpuwm.ensemble.member import MemberOutcome
from gpuwm.ensemble.perturbation import resolve_perturbation
from gpuwm.ensemble.seeds import member_seed
from gpuwm.ensemble.state_sha import live_state_sha256

BASE_TOML = """
[experiment]
name = "ensemble_engine_unit"
start_time = 1999-05-03T12:00:00
run_seconds = 120.0
restart_interval_s = 60.0

[projection]
map_proj = "lambert"
ref_lat = 39.6848
ref_lon = -83.9297
truelat1 = 30.0
truelat2 = 60.0
stand_lon = -83.9297

[shared]
nz = 4
ztop = 20000.0
p_top = 10000.0

[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = 8
ny = 8
time_step = 60
dx = 12000.0
history_interval_s = 60.0
"""


def _write_base(tmp_path):
    path = tmp_path / "base.toml"
    path.write_text(BASE_TOML, encoding="utf-8")
    return path


def _write_overlay(tmp_path, *, n_members=2, base_seed=20260730,
                   perturbation="experimental-stub", options=None,
                   base_config=None):
    base = base_config or _write_base(tmp_path)
    body = [
        "[ensemble]",
        f'base_config = "{base.name}"',
        f"n_members = {n_members}",
        f"base_seed = {base_seed}",
        f'perturbation = "{perturbation}"',
    ]
    if options:
        body.append("")
        body.append("[ensemble.perturbation_options]")
        for key, value in options.items():
            body.append(f"{key} = {json.dumps(value)}")
    path = tmp_path / "ensemble.toml"
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------- seeds


def test_member_seed_is_deterministic_and_index_dependent():
    first = [member_seed(7, index) for index in range(4)]
    assert first == [member_seed(7, index) for index in range(4)]
    assert len(set(first)) == 4
    # A neighbouring base seed must not produce a shifted copy of the
    # same stream, which base_seed + index would.
    assert not set(first) & {member_seed(8, index) for index in range(4)}


@pytest.mark.parametrize("bad", [-1, True, 1.0, "3"])
def test_member_seed_refuses_nonsense(bad):
    with pytest.raises((TypeError, ValueError)):
        member_seed(bad, 0)


# -------------------------------------------------------------- config


def test_load_ensemble_config_binds_the_base_config_sha(tmp_path):
    overlay = _write_overlay(tmp_path)
    cfg = load_ensemble_config(overlay)
    assert cfg.n_members == 2
    assert cfg.base_config.name == "base.toml"
    assert len(cfg.base_config_sha256) == 64
    # Editing the base config changes the binding.
    before = cfg.base_config_sha256
    cfg.base_config.write_text(BASE_TOML + "\n# touched\n", encoding="utf-8")
    assert load_ensemble_config(overlay).base_config_sha256 != before


def test_ensemble_config_refuses_unknown_keys(tmp_path):
    _write_base(tmp_path)
    overlay = tmp_path / "ensemble.toml"
    overlay.write_text(
        '[ensemble]\nbase_config = "base.toml"\nn_members = 2\n'
        'base_seed = 1\nperturbation = "none"\nmembers = 3\n',
        encoding="utf-8")
    with pytest.raises(ValueError, match="unknown key"):
        load_ensemble_config(overlay)


def test_ensemble_config_refuses_missing_base_config(tmp_path):
    overlay = tmp_path / "ensemble.toml"
    overlay.write_text(
        '[ensemble]\nbase_config = "absent.toml"\nn_members = 2\n'
        'base_seed = 1\nperturbation = "none"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="does not exist"):
        load_ensemble_config(overlay)


@pytest.mark.parametrize("n_members", [0, -3, 100000])
def test_ensemble_config_refuses_bad_member_counts(tmp_path, n_members):
    _write_base(tmp_path)
    overlay = tmp_path / "ensemble.toml"
    overlay.write_text(
        f'[ensemble]\nbase_config = "base.toml"\nn_members = {n_members}\n'
        'base_seed = 1\nperturbation = "none"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="n_members"):
        load_ensemble_config(overlay)


# ------------------------------------------------------------ manifest


def test_manifest_write_is_atomic_and_schema_checked(tmp_path):
    target = tmp_path / "ensemble-manifest.json"
    write_manifest_atomically(target, {"schema": ENSEMBLE_MANIFEST_SCHEMA,
                                       "members": []})
    assert not list(tmp_path.glob("*.tmp"))
    assert read_manifest(target, schema=ENSEMBLE_MANIFEST_SCHEMA)["schema"] \
        == ENSEMBLE_MANIFEST_SCHEMA
    with pytest.raises(ValueError, match="expected"):
        read_manifest(target, schema=CYCLE_MANIFEST_SCHEMA)
    with pytest.raises(ValueError, match="refusing to write"):
        write_manifest_atomically(target, {"schema": "something-else"})


def test_prepare_ensemble_lays_out_members_and_seeds(tmp_path):
    cfg = load_ensemble_config(_write_overlay(tmp_path, n_members=3))
    root = tmp_path / "ens"
    path = prepare_ensemble(cfg, root)
    manifest = read_manifest(path, schema=ENSEMBLE_MANIFEST_SCHEMA)
    assert manifest["n_members"] == 3
    assert [record["seed"] for record in manifest["members"]] \
        == member_seeds(cfg)
    assert all(record["status"] == "PENDING"
               for record in manifest["members"])
    for index in range(3):
        assert (root / member_directory_name(index)).is_dir()


def test_prepare_ensemble_refuses_a_changed_config(tmp_path):
    overlay = _write_overlay(tmp_path, n_members=2)
    cfg = load_ensemble_config(overlay)
    root = tmp_path / "ens"
    prepare_ensemble(cfg, root)
    other = load_ensemble_config(_write_overlay(
        tmp_path, n_members=2, base_seed=1))
    with pytest.raises(ValueError, match="different ensemble"):
        prepare_ensemble(other, root)


# ------------------------------------------- member kernel for the engine
#
# A state carrying the restart contract's attribute names, so the
# shipped perturbation hook and the shipped state hash both operate on
# it exactly as they do on a device state.


def _contract_names(count=3):
    from gpuwm.ensemble.state_sha import serialized_state_attrs

    return serialized_state_attrs()[:count]


def _make_state():
    names = _contract_names()
    state = types.SimpleNamespace()
    for position, name in enumerate(names):
        setattr(state, name, np.full((2, 3, 4), float(position),
                                     dtype=np.float64))
    return state, names


def _kernel_runner(*, base_config, member_dir, index, seed, perturbation,
                   perturbation_options, run_seconds=None, **_):
    """A member whose only physics is 'the seed decides the state'."""
    state, names = _make_state()
    hook = resolve_perturbation(perturbation)
    initial = live_state_sha256(state)
    report = hook(state, seed, dict(perturbation_options or {}))
    # A deterministic advance so the final hash differs from the
    # perturbed one exactly as a real integration's does.
    for name in names:
        getattr(state, name)[...] += 1.0
    member_dir.mkdir(parents=True, exist_ok=True)
    (member_dir / "member.json").write_text(
        json.dumps({"seed": seed, "index": index}, sort_keys=True),
        encoding="utf-8")
    return MemberOutcome(
        index=index, seed=seed, member_dir=member_dir,
        initial_state_sha256=initial,
        final_state_sha256=live_state_sha256(state),
        wall_seconds=0.25 * (index + 1), sim_seconds=float(run_seconds or 120),
        wrfout_count=2, last_checkpoint=None,
        perturbation={"report": report})


# -------------------------------------------------------------- engine


def test_run_ensemble_records_every_member(tmp_path):
    cfg = load_ensemble_config(_write_overlay(
        tmp_path, n_members=3, perturbation="experimental-stub",
        options={"field": _contract_names()[0], "amplitude": 0.5}))
    root = tmp_path / "ens"
    result = run_ensemble(cfg, root, runner=_kernel_runner)
    assert result.status == "COMPLETE"
    assert result.ran == (0, 1, 2)
    manifest = read_manifest(root / ENSEMBLE_MANIFEST_NAME,
                             schema=ENSEMBLE_MANIFEST_SCHEMA)
    shas = [record["final_state_sha256"] for record in manifest["members"]]
    assert all(len(sha) == 64 for sha in shas)
    assert len(set(shas)) == 3, "distinct seeds must give distinct members"
    assert manifest["seed_derivation"] == "gpuwm-ensemble-seed.v1"
    assert manifest["experimental"] is True


def test_completed_member_is_refused_not_rerun(tmp_path):
    cfg = load_ensemble_config(_write_overlay(tmp_path, n_members=2,
                                              perturbation="none"))
    root = tmp_path / "ens"
    run_ensemble(cfg, root, runner=_kernel_runner)
    with pytest.raises(MemberAlreadyCompleteError, match="already DONE"):
        run_ensemble(cfg, root, members=[0], runner=_kernel_runner)
    with pytest.raises(MemberAlreadyCompleteError):
        run_ensemble(cfg, root, resume=False, runner=_kernel_runner)


def test_resume_picks_up_at_the_first_incomplete_member(tmp_path):
    cfg = load_ensemble_config(_write_overlay(tmp_path, n_members=4,
                                              perturbation="none"))
    root = tmp_path / "ens"
    boom = RuntimeError("device fell over")

    def flaky(**kwargs):
        if kwargs["index"] == 2:
            raise boom
        return _kernel_runner(**kwargs)

    with pytest.raises(RuntimeError, match="device fell over"):
        run_ensemble(cfg, root, runner=flaky)
    manifest = read_manifest(root / ENSEMBLE_MANIFEST_NAME,
                             schema=ENSEMBLE_MANIFEST_SCHEMA)
    assert [record["status"] for record in manifest["members"]] \
        == ["DONE", "DONE", "FAILED", "PENDING"]
    assert manifest["status"] == "FAILED"
    assert manifest["members"][2]["error"]["type"] == "RuntimeError"

    first_two = [record["final_state_sha256"]
                 for record in manifest["members"][:2]]
    result = run_ensemble(cfg, root, runner=_kernel_runner)
    assert result.ran == (2, 3)
    assert result.skipped == ()
    resumed = read_manifest(root / ENSEMBLE_MANIFEST_NAME,
                            schema=ENSEMBLE_MANIFEST_SCHEMA)
    assert resumed["status"] == "COMPLETE"
    assert [record["final_state_sha256"]
            for record in resumed["members"][:2]] == first_two


def test_a_member_is_byte_identical_given_its_seed(tmp_path):
    options = {"field": _contract_names()[0], "amplitude": 0.75}
    cfg_a = load_ensemble_config(_write_overlay(
        tmp_path / "a", n_members=2, perturbation="experimental-stub",
        options=options, base_config=_write_base(_mkdir(tmp_path / "a"))))
    cfg_b = load_ensemble_config(_write_overlay(
        tmp_path / "b", n_members=2, perturbation="experimental-stub",
        options=options, base_config=_write_base(_mkdir(tmp_path / "b"))))
    first = run_ensemble(cfg_a, tmp_path / "a" / "ens", runner=_kernel_runner)
    second = run_ensemble(cfg_b, tmp_path / "b" / "ens", runner=_kernel_runner)
    left = read_manifest(first.manifest_path,
                         schema=ENSEMBLE_MANIFEST_SCHEMA)["members"]
    right = read_manifest(second.manifest_path,
                          schema=ENSEMBLE_MANIFEST_SCHEMA)["members"]
    assert [record["seed"] for record in left] \
        == [record["seed"] for record in right]
    assert [record["final_state_sha256"] for record in left] \
        == [record["final_state_sha256"] for record in right]
    assert left[0]["final_state_sha256"] != left[1]["final_state_sha256"]


def _mkdir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


# -------------------------------------------------------- perturbation


def test_perturbation_none_leaves_the_state_alone():
    state, _ = _make_state()
    before = live_state_sha256(state)
    hook = resolve_perturbation("none")
    report = hook(state, 5, {})
    assert report["applied"] is False
    assert live_state_sha256(state) == before


def test_perturbation_stub_is_seed_deterministic():
    names = _contract_names()
    options = {"field": names[0], "amplitude": 0.5}
    hook = resolve_perturbation("experimental-stub")
    left, _ = _make_state()
    right, _ = _make_state()
    other, _ = _make_state()
    hook(left, 11, options)
    hook(right, 11, options)
    hook(other, 12, options)
    assert live_state_sha256(left) == live_state_sha256(right)
    assert live_state_sha256(left) != live_state_sha256(other)


def test_perturbation_reference_to_the_da_lane_fails_closed():
    pytest.importorskip("pytest")
    try:
        import gpuwm.da.perturb  # noqa: F401
    except ImportError:
        with pytest.raises(ValueError, match="not importable"):
            resolve_perturbation("gpuwm.da.perturb")
    else:  # the lane landed: the contract entry point must exist
        assert callable(resolve_perturbation("gpuwm.da.perturb").apply)


def test_unknown_perturbation_reference_is_refused():
    with pytest.raises(ValueError, match="unknown perturbation reference"):
        resolve_perturbation("hand-wavy")


# ---------------------------------------------------------- increments


def test_apply_increments_adds_and_receipts():
    state, names = _make_state()
    increments = {names[0]: np.full((2, 3, 4), 2.0),
                  names[1]: np.full((2, 3, 4), -1.0)}
    before = live_state_sha256(state)
    receipt = apply_increments(state, increments)
    assert receipt["contract"] == "gpuwm-da-increments.v1"
    assert receipt["state_sha256_before"] == before
    assert receipt["state_sha256_after"] == live_state_sha256(state)
    assert np.allclose(getattr(state, names[0]), 2.0)
    assert np.allclose(getattr(state, names[1]), 0.0)
    assert [entry["field"] for entry in receipt["fields"]] \
        == sorted([names[0], names[1]])


@pytest.mark.parametrize("increments, match", [
    ({"not_a_field": np.zeros((2, 3, 4))}, "does not carry"),
    ({}, "empty"),
])
def test_apply_increments_refuses_bad_input(increments, match):
    state, _ = _make_state()
    with pytest.raises((ValueError, TypeError), match=match):
        apply_increments(state, increments)


def test_apply_increments_refuses_shape_and_nan():
    state, names = _make_state()
    before = live_state_sha256(state)
    with pytest.raises(ValueError, match="shape"):
        apply_increments(state, {names[0]: np.zeros((2, 3))})
    with pytest.raises(ValueError, match="non-finite"):
        apply_increments(state, {names[0]: np.full((2, 3, 4), np.nan)})
    assert live_state_sha256(state) == before, "no partial application"


def test_apply_increments_to_checkpoint_writes_a_new_file(tmp_path):
    names = _contract_names()
    background = tmp_path / "gpuwmrst_000.npz"
    payload = {f"state/{name}": np.zeros((2, 3), dtype=np.float32)
               for name in names}
    payload["meta/elapsed_seconds"] = np.asarray(60.0)
    np.savez(background, **payload)
    analysis = tmp_path / "analysis.npz"
    receipt = apply_increments_to_checkpoint(
        background, {names[0]: np.full((2, 3), 1.5)}, analysis)
    assert receipt["state_sha256_before"] != receipt["state_sha256_after"]
    with np.load(analysis, allow_pickle=False) as data:
        assert np.allclose(data[f"state/{names[0]}"], 1.5)
        assert np.allclose(data[f"state/{names[1]}"], 0.0)
        assert float(data["meta/elapsed_seconds"]) == 60.0
    with np.load(background, allow_pickle=False) as data:
        assert np.allclose(data[f"state/{names[0]}"], 0.0), \
            "the background must survive its own analysis"
    with pytest.raises(ValueError, match="over its own background"):
        apply_increments_to_checkpoint(
            background, {names[0]: np.zeros((2, 3))}, background)


def test_apply_increments_to_checkpoint_refuses_a_non_prognostic(tmp_path):
    names = _contract_names()
    background = tmp_path / "gpuwmrst_000.npz"
    np.savez(background, **{f"state/{names[0]}": np.zeros((2, 3))})
    with pytest.raises(ValueError, match="restart prognostic contract"):
        apply_increments_to_checkpoint(
            background, {"nwp_diagnostics": np.zeros((2, 3))},
            tmp_path / "analysis.npz")


# --------------------------------------------------------------- cycle


def _cycling_runner(**kwargs):
    """A member kernel that also leaves a checkpoint for the DA seam."""
    outcome = _kernel_runner(**kwargs)
    names = _contract_names()
    payload = {f"state/{name}": np.full((2, 3), float(kwargs["index"]),
                                        dtype=np.float32)
               for name in names}
    np.savez(kwargs["member_dir"] / "gpuwmrst_d01_000.npz", **payload)
    return outcome


def test_cycle_driver_stops_at_the_seam_and_applies_increments(tmp_path):
    cfg = load_ensemble_config(_write_overlay(tmp_path, n_members=2,
                                              perturbation="none"))
    root = tmp_path / "ens"
    names = _contract_names()
    seen = []

    def assimilate(cycle_index, member_states):
        seen.append((cycle_index, sorted(member_states)))
        return {index: {names[0]: np.full((2, 3), 0.25)}
                for index in member_states}

    result = run_cycles(cfg, root, n_cycles=2, cycle_seconds=60.0,
                        assimilate=assimilate, runner=_cycling_runner)
    assert result.cycles_run == (0, 1)
    manifest = read_manifest(result.manifest_path,
                             schema=CYCLE_MANIFEST_SCHEMA)
    assert manifest["schema"] == "gpuwm-da-cycle-manifest.v1"
    assert len(manifest["cycles"]) == 2
    assert seen == [(0, [0, 1]), (1, [0, 1])]
    first = manifest["cycles"][0]
    assert first["start_offset_seconds"] == 0.0
    assert first["end_offset_seconds"] == 60.0
    assert [entry["state_sha256"] for entry in first["members"]] \
        != [None, None]
    assimilation = first["assimilation"]
    assert assimilation["status"] == "APPLIED"
    # The engine cannot know the METHOD -- that is what the seam is for --
    # but it always knows which callable it invoked, and a receipt that
    # said only "method: null" could not tell two analyses from different
    # filters and different observations apart.
    method = assimilation["method"]
    assert method["callable"].endswith("assimilate")
    assert method["declared_by"] is None, \
        "this callable declared no provenance, and the receipt says so"
    assert method["provenance"] is None
    assert assimilation["member_count"] == 2
    assert assimilation["attempt"] == 1
    for index in range(2):
        analysis = root / "cycle_000" / member_directory_name(index) \
            / "analysis.npz"
        assert analysis.is_file()


def test_cycle_without_an_assimilation_step_records_the_empty_slot(tmp_path):
    cfg = load_ensemble_config(_write_overlay(tmp_path, n_members=1,
                                              perturbation="none"))
    root = tmp_path / "ens"
    run_cycles(cfg, root, n_cycles=1, cycle_seconds=30.0,
               runner=_cycling_runner)
    manifest = read_manifest(root / "da-cycle-manifest.json",
                             schema=CYCLE_MANIFEST_SCHEMA)
    assert manifest["cycles"][0]["assimilation"] is None
    assert manifest["cycles"][0]["status"] == "DONE"


def test_cycle_refuses_a_member_with_no_checkpoint(tmp_path):
    cfg = load_ensemble_config(_write_overlay(tmp_path, n_members=1,
                                              perturbation="none"))
    names = _contract_names()
    with pytest.raises(ValueError, match="no gpuwmrst"):
        run_cycles(cfg, tmp_path / "ens", n_cycles=1, cycle_seconds=30.0,
                   assimilate=lambda _i, states: {
                       index: {names[0]: np.zeros((2, 3))}
                       for index in states},
                   runner=_kernel_runner)


# --------------------------------------------------------------- bench


def test_bench_table_reports_rates_and_makes_no_claim(tmp_path):
    cfg = load_ensemble_config(_write_overlay(tmp_path, n_members=2,
                                              perturbation="none"))
    root = tmp_path / "ens"
    run_ensemble(cfg, root, run_seconds=120.0, runner=_kernel_runner)
    manifest = read_manifest(root / ENSEMBLE_MANIFEST_NAME,
                             schema=ENSEMBLE_MANIFEST_SCHEMA)
    rows = bench_rows(manifest)
    assert [row["member"] for row in rows] == [0, 1]
    # 0.25 s wall for 120 s of simulation is 0.125 s per simulated minute.
    assert rows[0]["wall_s_per_sim_min"] == pytest.approx(0.125)
    assert rows[1]["wall_s_per_sim_min"] == pytest.approx(0.25)
    table = format_table(rows, title="unit")
    assert "wall_s_per_sim_min" in table
    assert "no realtime or member-count claim" in table
