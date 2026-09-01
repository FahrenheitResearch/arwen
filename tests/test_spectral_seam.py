"""[spectral_numerics] through the resolved experiment and the slow-step seam.

CPU-only coverage of the Level-2 wiring: the config rides the experiment
TOML into ``ExperimentConfig``, binds the restart identity when present
(and only then), refuses the loops that cannot honor it, refuses streamed
domains and false periodic declarations at attach, ledgers receipts per
committed step, and blocks a clean completion capsule when apply receipts
are missing.  The one GPU-shaped line -- the ``execute_experiment`` STEP op
calling the seam -- is held in place by a source-order gate, because a hook
that drifts out of the commit point (into an acoustic substep, after
output, or out of the file entirely) is precisely the breakage the
delivered survey existed to prevent.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core.model import restart_identity_payload
from gpuwm.experiment import (build_experiment,
                              refuse_unrouted_spectral_numerics)
from gpuwm.spectral_ops import SpectralNumericsConfig
from gpuwm.spectral_ops.config import from_mapping
from gpuwm.spectral_seam import (SpectralSeam, attach_seam,
                                 seam_capsule_receipts)


def _experiment_raw(spectral=None):
    raw = {
        "experiment": {
            "name": "seam_probe",
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
    if spectral is not None:
        raw["spectral_numerics"] = spectral
    return raw


SHADOW_TABLE = {
    "mode": "shadow", "boundary": "tapered", "edge_taper_cells": 4,
    "scalar": [{"field": "thp",
                "diffusion": {"order": 3, "reference_wavelength_m": 18000.0,
                              "e_fold_time_s": 450.0}}],
}


def shadow_config(**overrides):
    table = dict(SHADOW_TABLE)
    table.update(overrides)
    return from_mapping(table)


# ---------------------------------------------------------------------------
# config through the resolved experiment


def test_absent_table_resolves_to_none():
    exp = build_experiment(_experiment_raw(), source="probe.toml")
    assert exp.spectral_numerics is None


def test_present_table_resolves_to_the_owners_config():
    exp = build_experiment(_experiment_raw(SHADOW_TABLE),
                           source="probe.toml")
    assert isinstance(exp.spectral_numerics, SpectralNumericsConfig)
    assert exp.spectral_numerics.mode == "shadow"
    assert exp.spectral_numerics.scalar_targets[0].field == "thp"


def test_unknown_key_refusal_names_the_table_and_source():
    raw = _experiment_raw({"mode": "shadow", "cadence": 2})
    with pytest.raises(ValueError, match=r"\[spectral_numerics\] of "
                                         r"probe.toml.*cadence"):
        build_experiment(raw, source="probe.toml")


def test_whole_table_survives_the_unknown_table_sweep():
    # Guards the known_tables registration: without it the table would be
    # refused as unknown and every setting in it dropped behind one line.
    exp = build_experiment(_experiment_raw(SHADOW_TABLE),
                           source="probe.toml")
    assert exp.spectral_numerics is not None


# ---------------------------------------------------------------------------
# restart / config identity


def test_absent_config_stays_absent_from_the_restart_identity():
    exp = build_experiment(_experiment_raw(), source="probe.toml")
    payload = restart_identity_payload(exp)
    assert "spectral_numerics" not in payload


def test_present_config_binds_the_restart_identity_value_for_value():
    exp = build_experiment(_experiment_raw(SHADOW_TABLE),
                           source="probe.toml")
    payload = restart_identity_payload(exp)
    bound = payload["spectral_numerics"]
    assert bound["mode"] == "shadow"
    assert bound["scalar_targets"][0]["field"] == "thp"
    retuned = dict(SHADOW_TABLE)
    retuned["scalar"] = [{
        "field": "thp",
        "diffusion": {"order": 3, "reference_wavelength_m": 24000.0,
                      "e_fold_time_s": 450.0}}]
    other = build_experiment(_experiment_raw(retuned), source="probe.toml")
    assert restart_identity_payload(other) != payload


# ---------------------------------------------------------------------------
# unrouted refusal


def test_off_and_absent_pass_the_unrouted_refusal():
    absent = build_experiment(_experiment_raw(), source="probe.toml")
    refuse_unrouted_spectral_numerics(absent, "probe-route")
    off = build_experiment(
        _experiment_raw({"mode": "off"}), source="probe.toml")
    refuse_unrouted_spectral_numerics(off, "probe-route")


def test_active_mode_refuses_on_an_unwired_route_naming_the_route():
    exp = build_experiment(_experiment_raw(SHADOW_TABLE),
                           source="probe.toml")
    with pytest.raises(RuntimeError, match="probe-route"):
        refuse_unrouted_spectral_numerics(exp, "probe-route")


# ---------------------------------------------------------------------------
# attach-time refusals


def _run_cfg(**overrides):
    values = dict(dx=3000.0, dy=3000.0, dt=15.0, open_x=False, open_y=False,
                  specified=False, nested=False)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_streamed_domain_refuses_an_active_mode():
    seam = SpectralSeam(shadow_config(), "probe")
    with pytest.raises(RuntimeError, match="t=0 attach snapshot"):
        seam.validate_domain(2, _run_cfg(), streamed=True)


def test_false_periodic_declaration_refuses_naming_what_broke_the_wrap():
    seam = SpectralSeam(
        shadow_config(boundary="periodic", periodic_domain=True), "probe")
    with pytest.raises(RuntimeError, match="specified lateral boundaries"):
        seam.validate_domain(1, _run_cfg(specified=True), streamed=False)
    with pytest.raises(RuntimeError, match="nested"):
        seam.validate_domain(2, _run_cfg(nested=True), streamed=False)
    with pytest.raises(RuntimeError, match="open_x"):
        seam.validate_domain(1, _run_cfg(open_x=True), streamed=False)


def test_a_truly_periodic_domain_may_declare_the_wrap():
    seam = SpectralSeam(
        shadow_config(boundary="periodic", periodic_domain=True), "probe")
    seam.validate_domain(1, _run_cfg(), streamed=False)


def test_tapered_mode_on_a_specified_domain_is_admitted():
    seam = SpectralSeam(shadow_config(), "probe")
    seam.validate_domain(1, _run_cfg(specified=True), streamed=False)


# ---------------------------------------------------------------------------
# the per-step ledger


def _stepped_seam(config, steps, ny=20, nx=24):
    seam = SpectralSeam(config, "probe")
    rng = np.random.default_rng(1)
    state = {"thp": rng.normal(size=(3, ny, nx))}
    run_cfg = _run_cfg(specified=True)
    receipts = []
    for step in range(1, steps + 1):
        receipts.append(seam.after_step(
            state, 1, run_cfg, step_count=step,
            model_seconds=15.0 * step))
    return seam, state, receipts


def test_shadow_steps_ledger_receipts_and_leave_state_bitwise_alone():
    rng = np.random.default_rng(1)
    before = rng.normal(size=(3, 20, 24))
    seam, state, receipts = _stepped_seam(shadow_config(), steps=3)
    np.testing.assert_array_equal(state["thp"], before)
    assert all(r is not None for r in receipts)
    record = seam.capsule_record()
    assert record["mode"] == "shadow"
    assert record["complete"] is True
    assert record["domains"]["d01"] == {
        "steps": 3, "expected_receipts": 3, "receipts": 3,
        "receipt_hash_chain_sha256":
            record["domains"]["d01"]["receipt_hash_chain_sha256"]}
    assert record["domains"]["d01"]["receipt_hash_chain_sha256"] is not None
    assert record["operator_pins_sha256"] == (
        "549502b5f1b66fff4dda949ba5a16cfb9ed71bb52877c2ef2f395d36c031c2ad")


def test_cadence_steps_skip_without_reading_state():
    class ReadsNothing:
        def __getattr__(self, name):
            raise AssertionError(f"non-cadence step read {name!r}")

    seam = SpectralSeam(shadow_config(cadence_steps=2), "probe")
    run_cfg = _run_cfg(specified=True)
    # Step 1 is off-cadence (1 % 2 != 0): the hook must not read state.
    seam.validate_domain(1, run_cfg, streamed=False)
    assert seam.after_step(ReadsNothing(), 1, run_cfg, step_count=1,
                           model_seconds=15.0) is None
    state = {"thp": np.random.default_rng(2).normal(size=(3, 20, 24))}
    assert seam.after_step(state, 1, run_cfg, step_count=2,
                           model_seconds=30.0) is not None
    assert seam.expected_receipts(1) == 1
    assert seam.complete


def test_apply_steps_mutate_and_ledger_applied_receipts():
    seam, state, receipts = _stepped_seam(shadow_config(mode="apply"),
                                          steps=2)
    assert all(r["applied"] for r in receipts)
    assert seam.capsule_record()["complete"] is True


def test_missing_apply_receipts_block_a_clean_completion_capsule():
    seam = SpectralSeam(shadow_config(mode="apply"), "probe")
    # Simulate a run that stepped but whose hook never produced receipts
    # (the mis-wiring this contract exists to catch).
    seam._steps[1] = 5
    with pytest.raises(RuntimeError, match="incomplete"):
        seam.require_complete()
    model = SimpleNamespace(_spectral_seam=seam)
    with pytest.raises(RuntimeError, match="incomplete"):
        seam_capsule_receipts(model)


def test_shadow_shortfall_is_recorded_not_fatal():
    seam = SpectralSeam(shadow_config(), "probe")
    seam._steps[1] = 5
    seam.require_complete()          # shadow never blocks completion
    record = seam.capsule_record()
    assert record["complete"] is False


def test_a_streamed_late_joiner_refuses_at_its_first_commit():
    seam = SpectralSeam(shadow_config(), "probe")
    with pytest.raises(RuntimeError, match="t=0 attach snapshot"):
        seam.after_step({}, 4, _run_cfg(nested=True), step_count=1,
                        model_seconds=15.0, streamed=True)


# ---------------------------------------------------------------------------
# attach_seam against a model-shaped object


class _Node:
    def __init__(self, grid_id, run_cfg):
        self.cfg = SimpleNamespace(grid_id=grid_id, run=run_cfg)


class _Model:
    def __init__(self, nodes):
        self._nodes = nodes

    def walk_parent_first(self):
        return list(self._nodes)


def _exp(config):
    return SimpleNamespace(spectral_numerics=config, name="probe")


def test_attach_returns_none_for_absent_and_off():
    model = _Model([_Node(1, _run_cfg(specified=True))])
    assert attach_seam(model, _exp(None), {}) is None
    assert attach_seam(model, _exp(SpectralNumericsConfig()), {}) is None
    assert attach_seam(model, None, {}) is None
    assert seam_capsule_receipts(model) == {}


def test_attach_builds_once_and_reuses_across_leg_walks():
    model = _Model([_Node(1, _run_cfg(specified=True))])
    seam = attach_seam(model, _exp(shadow_config()), {})
    assert seam is not None
    assert attach_seam(model, _exp(shadow_config()), {}) is seam
    assert model._spectral_seam is seam


def test_attach_refuses_a_streamed_grid_up_front():
    class _Streamed:
        pass

    from gpuwm.core import streaming

    stepper = streaming.StreamedDomain.__new__(streaming.StreamedDomain)
    model = _Model([_Node(1, _run_cfg(specified=True))])
    with pytest.raises(RuntimeError, match="t=0 attach snapshot"):
        attach_seam(model, _exp(shadow_config()), {1: stepper})


def test_capsule_receipts_bind_the_seam_record():
    model = _Model([_Node(1, _run_cfg(specified=True))])
    seam = attach_seam(model, _exp(shadow_config()), {})
    state = {"thp": np.random.default_rng(5).normal(size=(3, 20, 24))}
    seam.after_step(state, 1, _run_cfg(specified=True), step_count=1,
                    model_seconds=15.0)
    receipts = seam_capsule_receipts(model)
    assert receipts["spectral_numerics"]["domains"]["d01"]["receipts"] == 1


# ---------------------------------------------------------------------------
# the seam's place in execute_experiment


def test_the_hook_sits_at_the_slow_step_commit_point_in_source_order():
    """The STEP op calls the seam AFTER the post-step clock refresh and
    BEFORE poison/health/observer -- i.e. at the slow RK commit, never in
    an acoustic substep (those live inside dycore.step, below this call)
    and never after output or feedback (those are later ops).  A refactor
    that moves or drops the call must move this gate with it consciously.
    """
    import inspect

    from gpuwm.core.model import execute_experiment

    source = inspect.getsource(execute_experiment)
    on_step = source[source.index("def on_step"):
                     source.index("def on_force")]
    stepper_call = on_step.index("steppers.get(grid_id, step)(")
    commit = on_step.index("after_step=True")
    seam_call = on_step.index("spectral_seam.after_step(")
    poison = on_step.index("poison()")
    assert stepper_call < commit < seam_call < poison, (
        "the Level-2 hook must fire once per domain immediately after the "
        "slow RK state commit (refresh_model_time(after_step=True)) and "
        "before anything else observes the step")
    assert "step_count=clock.step_count + 1" in on_step


def test_the_committed_seam_survey_matches_the_live_tree():
    """docs/handoffs/CURRENT-CORE-SPECTRAL-SEAM-SURVEY.json is a record of
    the live seam, and this holds it against the tree both ways -- a
    seam refactor must regenerate the record, and a stale record must not
    describe a seam that no longer exists."""
    from tools.spectral_seam_survey import main as survey_main

    assert survey_main(["--check"]) == 0


def test_execute_experiment_attaches_the_seam_before_stepping():
    import inspect

    from gpuwm.core.model import execute_experiment

    source = inspect.getsource(execute_experiment)
    attach = source.index("attach_seam(")
    on_step = source.index("def on_step")
    assert attach < on_step, (
        "the seam (and its streamed/periodic refusals) must attach at "
        "start, before any step commits")


def test_every_integrating_route_hands_the_seam_its_experiment():
    """A route that constructs its own ``ExperimentState`` never sets
    ``_activation_context``, so an experiment that reaches the seam only
    through that attribute is INVISIBLE on exactly the two prepared
    routes the Level-2 contract names as honoring the config.

    Measured on real HRRR bytes 2026-08-18: a shadow run through
    ``gpuwm.prepared_domain_tree_forecast`` (2 domains, 60 + 240
    committed steps) wrote ZERO step receipts, bound no
    ``receipts.spectral_numerics`` section, and still emitted a clean
    PASS capsule -- the operator was silently absent on a route that
    neither honored nor refused it, which is the single state the
    honored-or-refused governance exists to make impossible.  Every call
    site therefore hands ``execute_experiment`` the resolved experiment
    explicitly, and this gate fails closed for a route added later.
    """
    import ast
    import pathlib

    import gpuwm

    root = pathlib.Path(gpuwm.__file__).parent
    missing = []
    seen = 0
    for path in sorted(root.rglob("*.py")):
        # gpuwm/verify/cases/* are the frozen Level-1 numerics cases.
        # They build their own idealized experiments in process and no
        # user config reaches them, so there is no active
        # [spectral_numerics] for them to drop -- and their pins must
        # not move to satisfy a gate about forecast routes.
        if "verify" in path.relative_to(root).parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (func.id if isinstance(func, ast.Name)
                    else func.attr if isinstance(func, ast.Attribute)
                    else None)
            if name != "execute_experiment":
                continue
            seen += 1
            if not any(keyword.arg == "experiment"
                       for keyword in node.keywords):
                missing.append(
                    f"{path.relative_to(root)}:{node.lineno}")
    assert seen >= 3, "the gate found no execute_experiment call sites"
    assert not missing, (
        "these routes call execute_experiment without handing it the "
        "resolved experiment, so an active [spectral_numerics] would run "
        f"as a silently absent operator: {missing}")


def test_execute_experiment_takes_the_experiment_explicitly():
    """The explicit argument is the seam's authority; the activation
    context is only the fallback for the front-door builder that sets
    it.  A route must be able to hand its experiment over without
    forging an activation context it does not own."""
    import inspect

    from gpuwm.core.model import execute_experiment

    parameters = inspect.signature(execute_experiment).parameters
    assert "experiment" in parameters, (
        "execute_experiment must accept the resolved experiment directly")
    assert parameters["experiment"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["experiment"].default is None, (
        "omitted, the seam falls back to the activation context, which "
        "keeps every pre-feature caller byte-identical")

    source = inspect.getsource(execute_experiment)
    attach = source[source.index("attach_seam("):]
    assert "seam_experiment" in attach, (
        "the attach must read the resolved experiment, not the "
        "activation context alone")
