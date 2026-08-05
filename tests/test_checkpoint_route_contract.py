# tests/test_checkpoint_route_contract.py
"""Checkpointing and restart keep the promises the product publishes.

Three findings from the v1.4.0 fleet test, each of which is either a
feature that must work or a truth that must be told before the run
rather than after it:

* **C-04** -- checkpointing is unreachable on the single-domain route
  with no ``[case_data]`` table, the route the wizard steers everyone
  to.  The runner's warning is honest but arrives at forecast time;
  ``gpuwm check`` said nothing; ``gpuwm resume`` then pointed back at
  the knob just declared inert.  Disposition: TELL THE TRUTH EARLY.
* **C-11** -- on the one route that can checkpoint, restart refused
  every config change ``gpuwm run --restart`` documents as permitted,
  as an unhandled traceback.  Disposition: MAKE IT WORK, and refuse the
  genuine mismatches by name.
* **C-10** -- a ``run_seconds`` off the step grid passed ``gpuwm check``
  and died in a traceback at forecast, while the identical arithmetic
  error on the two cadence keys was refused cleanly at admission.
  Disposition: REFUSE IT WHERE ITS SIBLINGS ARE REFUSED.
"""

import json
import tomllib
from pathlib import Path

import pytest

from gpuwm.checkpoint_routes import (
    CHECKPOINTLESS_ROUTE_ADVISORY, checkpoint_route_advisory,
    route_writes_checkpoints,
)
from gpuwm.experiment import build_experiment


def _wizard_config(tmp_path, *, ladder="12", extra=()):
    """A real wizard-emitted TOML -- the config a new user actually has."""

    from gpuwm.cli import main as cli_main

    out = tmp_path / "area.toml"
    rc = cli_main(["domain", "--point=25.76,-80.19", "--card", "24gb",
                   "--ladder", ladder, "--source", "gfs",
                   "--cycle", "2026-07-29T18", "--hours", "6",
                   "--out", str(out), *extra])
    assert rc == 0
    return out


def _raw(config):
    """The wizard's TOML as ``build_experiment`` receives it.

    ``build_experiment`` is the INNER loader and its table schema is
    deliberately strict.  The advisory ``[fetch]`` hints table the
    wizard emits is split off and validated one layer up -- by
    ``experiment.load_experiment``, ``case_data.
    load_experiment_case_bytes`` and ``core.preflight``, all three
    identically -- and never reaches it.  These tests call the inner
    function directly, so they perform the same split; without it
    a4417efb's restored unknown-TABLE refusal fires on ``[fetch]``
    here while every shipped entry point loads the same file cleanly
    (measured: ``gpuwm check`` on this exact wizard output reaches the
    memory preflight with no table refusal).
    """

    with open(config, "rb") as stream:
        raw = tomllib.load(stream)
    fetch_table = raw.pop("fetch", None)
    if fetch_table is not None:
        from gpuwm.fetch import validate_fetch_hints
        validate_fetch_hints(fetch_table, source=str(config))
    return raw


# ---------------------------------------------------------------------------
# C-04: the route that cannot checkpoint says so at check time
# ---------------------------------------------------------------------------

def test_the_route_matrix_matches_the_runners_that_exist():
    """One place decides, and it decides what the runners actually do."""

    assert not route_writes_checkpoints(domain_count=1, has_case_data=False)
    assert route_writes_checkpoints(domain_count=2, has_case_data=False)
    assert route_writes_checkpoints(domain_count=1, has_case_data=True)


def test_the_advisory_fires_only_when_something_was_actually_asked_for():
    """CONTROL: it must be silent on both sides, or it is noise."""

    inert = dict(domain_count=1, has_case_data=False)
    assert checkpoint_route_advisory(
        **inert, restart_interval_s=3600.0) == CHECKPOINTLESS_ROUTE_ADVISORY
    # Asked for nothing: nothing was lost, so nothing is said.
    assert checkpoint_route_advisory(**inert, restart_interval_s=0.0) is None
    # A route that honours the knob is never warned about it.
    assert checkpoint_route_advisory(
        domain_count=2, has_case_data=False,
        restart_interval_s=3600.0) is None
    assert checkpoint_route_advisory(
        domain_count=1, has_case_data=True,
        restart_interval_s=3600.0) is None


def test_check_names_the_route_limitation_before_the_forecast(
        tmp_path, capsys, monkeypatch):
    """C-04: `gpuwm check` is where this must be learned, not at forecast."""

    from gpuwm.core import preflight

    config = _wizard_config(tmp_path)
    raw = _raw(config)
    assert float(raw["experiment"]["restart_interval_s"]) == 0.0, (
        "the wizard emits an inert 0 here; this test asks for checkpoints")
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "restart_interval_s = 0.0", "restart_interval_s = 3600.0"),
        encoding="utf-8", newline="\n")
    capsys.readouterr()

    exp = preflight._load_experiment_any(config)
    advisories = preflight.check_advisories(exp, config)
    assert CHECKPOINTLESS_ROUTE_ADVISORY in advisories

    # NEGATIVE CONTROL: the same config with the knob left alone earns
    # no advisory, so a green check stays green for everybody else.
    plain = _wizard_config(tmp_path / "plain")
    assert preflight.check_advisories(
        preflight._load_experiment_any(plain), plain) == []


def test_resume_stops_pointing_at_the_knob_it_just_called_inert(tmp_path):
    """C-04: the circular advice becomes a route statement."""

    from gpuwm.resume import resolve_resume_checkpoint, route_note

    config = _wizard_config(tmp_path)
    outdir = tmp_path / "run"
    outdir.mkdir()

    with pytest.raises(ValueError) as caught:
        resolve_resume_checkpoint(outdir, config=config)
    message = str(caught.value)
    assert "writes no checkpoints" in message
    assert "[case_data]" in message and "multi-domain" in message

    # CONTROL: without the config the message is exactly what it was --
    # the route sentence is added, never substituted for the old advice.
    with pytest.raises(ValueError) as bare:
        resolve_resume_checkpoint(outdir)
    assert "writes no checkpoints" not in str(bare.value)
    assert str(bare.value) in message

    # CONTROL: a config whose route CAN checkpoint gets no route note,
    # because for it the original advice is the true advice.
    tree = _wizard_config(tmp_path / "tree", ladder="12-3")
    assert len(_raw(tree)["domain"]) > 1
    assert route_note(tree) == ""


# ---------------------------------------------------------------------------
# C-10: an off-grid run_seconds is refused where its siblings are
# ---------------------------------------------------------------------------

def _single_domain_raw(tmp_path):
    return _raw(_wizard_config(tmp_path))


def _root_dt(raw) -> float:
    """The root domain's exact timestep, as the wizard writes it."""

    root = raw["domain"][0]
    dt = float(root["time_step"])
    num = float(root.get("time_step_fract_num", 0))
    den = float(root.get("time_step_fract_den", 1))
    return dt + num / den


def _mutated(raw):
    """A deep copy that keeps the TOML datetime build_experiment demands."""

    import copy

    return copy.deepcopy(raw)


def test_run_seconds_off_the_step_grid_is_refused_at_admission(tmp_path):
    """C-10: the same error class as the cadences, in the same place."""

    raw = _single_domain_raw(tmp_path)
    dt = _root_dt(raw)
    assert dt > 1.0

    raw["experiment"]["run_seconds"] = 2.0 * dt
    build_experiment(raw, source="admission-test")  # on the grid: admitted

    raw["experiment"]["run_seconds"] = 2.5 * dt
    with pytest.raises(ValueError, match="run_seconds"):
        build_experiment(raw, source="admission-test")


def test_the_refusal_states_the_same_fact_the_clock_would_have(tmp_path):
    """The message a user gets early says what the late one said."""

    raw = _single_domain_raw(tmp_path)
    dt = _root_dt(raw)
    raw["experiment"]["run_seconds"] = 2.5 * dt
    with pytest.raises(ValueError) as caught:
        build_experiment(raw, source="admission-test")
    message = str(caught.value)
    assert "whole number of root-domain" in message
    assert "d01 boundary" in message


def test_the_three_timing_keys_are_refused_the_same_way(tmp_path):
    """CONTROL: run_seconds now behaves like the two that already did.

    Before this change `history_interval_s` and `restart_interval_s`
    were refused here and `run_seconds` was not -- the whole finding.
    """

    raw = _single_domain_raw(tmp_path)
    dt = _root_dt(raw)
    refusals = {}
    for key, table in (("run_seconds", "experiment"),
                       ("restart_interval_s", "experiment"),
                       ("history_interval_s", "domain")):
        mutated = _mutated(raw)
        holder = (mutated["experiment"] if table == "experiment"
                  else mutated["domain"][0])
        holder[key] = 2.5 * dt
        with pytest.raises(ValueError) as caught:
            build_experiment(mutated, source="admission-test")
        refusals[key] = str(caught.value)
    assert len(refusals) == 3
    for key, message in refusals.items():
        assert key in message, message
        assert "whole number" in message, message


# ---------------------------------------------------------------------------
# C-11: the permitted restart changes are permitted
# ---------------------------------------------------------------------------

def _identity(raw, tmp_path, name):
    """The tree route's restart identity for one experiment TOML."""

    from gpuwm.core.model import restart_identity_payload

    exp = build_experiment(raw, source=name)
    return json.dumps(restart_identity_payload(exp), sort_keys=True)


def test_the_documented_permitted_changes_do_not_move_the_identity(tmp_path):
    """C-11: extending a run from a checkpoint is what checkpoints are for.

    `gpuwm run --restart` publishes exactly this tolerance and
    FIRST-LIGHT section 7 builds its worked example on it.  The tree
    route used to hash the experiment FILE, so all three of these moved
    the fingerprint and all three were refused.
    """

    raw = _raw(_wizard_config(tmp_path))
    base = _identity(raw, tmp_path, "base")

    extended = _mutated(raw)
    extended["experiment"]["run_seconds"] = (
        float(raw["experiment"]["run_seconds"]) + 3600.0)
    assert _identity(extended, tmp_path, "extended") == base

    recadenced = _mutated(raw)
    recadenced["domain"][0]["history_interval_s"] = 1800.0
    assert _identity(recadenced, tmp_path, "recadenced") == base

    rerestarted = _mutated(raw)
    rerestarted["experiment"]["restart_interval_s"] = 7200.0
    assert _identity(rerestarted, tmp_path, "rerestarted") == base


def test_a_change_outside_the_contract_still_moves_the_identity(tmp_path):
    """CONTROL: the tolerance is a tolerance, not a hole.

    Without this the fix could have excluded everything and every
    checkpoint would restore into every run.
    """

    raw = _raw(_wizard_config(tmp_path))
    base = _identity(raw, tmp_path, "base")

    # Geometry, timestep and physics are trajectory identity and must
    # every one of them move the fingerprint.
    grid = _mutated(raw)
    grid["domain"][0]["nx"] = int(grid["domain"][0]["nx"]) - 2
    assert _identity(grid, tmp_path, "nx") != base

    step = _mutated(raw)
    step["domain"][0]["time_step"] = int(step["domain"][0]["time_step"]) // 2
    step["experiment"]["run_seconds"] = float(
        step["experiment"]["run_seconds"])
    assert _identity(step, tmp_path, "time_step") != base

    physics = _mutated(raw)
    current = int(physics["shared"]["mp_physics"])
    physics["shared"]["mp_physics"] = 6 if current != 6 else 8
    assert _identity(physics, tmp_path, "mp_physics") != base

    spacing = _mutated(raw)
    spacing["experiment"]["spec_bdy_width"] = int(
        spacing["experiment"]["spec_bdy_width"]) + 1
    assert _identity(spacing, tmp_path, "spec_bdy_width") != base


def test_a_genuine_mismatch_is_refused_by_name(tmp_path):
    """C-11 second half: 'fingerprint mismatch' names nothing actionable."""

    from gpuwm.io.restart import tree_fingerprint_mismatch_reason
    from types import SimpleNamespace

    live = {"schema": "s", "experiment_identity": {"nx": 100},
            "preparation_receipt_sha256": "aaaa",
            "execution_plan": {"plan_id": "p"}}
    stored = {**live, "preparation_receipt_sha256": "bbbb"}
    model = SimpleNamespace(
        experiment_fingerprint="deadbeef",
        _experiment_fingerprint_components=live)
    reason = tree_fingerprint_mismatch_reason(
        1, {"experiment_fingerprint_components": stored}, model)
    assert "preparation_receipt_sha256" in reason
    assert "forecast length" in reason
    assert "experiment_identity" not in reason

    # CONTROL: a checkpoint from before the named-component format still
    # refuses -- it just cannot name the component, and says so.
    legacy = tree_fingerprint_mismatch_reason(1, {}, model)
    assert "different run" in legacy
    assert "preparation_receipt_sha256" not in legacy


def test_the_identity_components_are_json_serializable(tmp_path):
    """CONTROL: these components are WRITTEN, not only hashed.

    They go into the checkpoint header, so a MappingProxyType or a Path
    surviving into them fails the checkpoint write itself -- after the
    forecast has already integrated and published frames.  Caught
    exactly that way on a real 2-domain run; this is its guard.
    """

    from types import MappingProxyType, SimpleNamespace

    from gpuwm import prepared_domain_tree_forecast as tree

    exp = build_experiment(_raw(_wizard_config(tmp_path, ladder="12-3")),
                           source="components-test")
    inputs = SimpleNamespace(
        experiment=exp,
        authority_sha256=MappingProxyType({"preparation_receipt": "abcd"}),
        domains=(SimpleNamespace(
            grid_id=1,
            cache_reader=SimpleNamespace(content_sha256="beef")),),
        execution_plan=MappingProxyType(
            {"plan_id": "p", "edges": (MappingProxyType({"a": 1}),)}),
    )
    components = tree.tree_restart_identity_components(
        inputs, MappingProxyType({"runtime": "x"}))
    json.dumps(components)  # must not raise
    assert set(components) == set(tree.TREE_RESTART_IDENTITY_COMPONENTS)
    assert isinstance(components["execution_plan"], dict)


def test_the_plan_identity_drops_the_cadence_the_receipt_describes():
    """CONTROL: the plan is a receipt; only its trajectory half is identity.

    `_domain_rows` records `history_interval_s` for the report, so
    hashing the plan verbatim re-bound the output cadence one layer
    above the prepared-cache identity -- and a restart with a changed
    history cadence was refused naming `execution_plan`.
    """

    from gpuwm.prepared_domain_tree_forecast import _plan_restart_identity

    plan = {"plan_id": "p", "domains": [
        {"grid_id": 1, "nx": 100, "history_interval_s": 3600.0},
        {"grid_id": 2, "nx": 200, "history_interval_s": 900.0}]}
    recadenced = {"plan_id": "p", "domains": [
        {"grid_id": 1, "nx": 100, "history_interval_s": 1800.0},
        {"grid_id": 2, "nx": 200, "history_interval_s": 300.0}]}
    assert _plan_restart_identity(plan) == _plan_restart_identity(recadenced)

    # CONTROL: a geometry change in the same rows still moves it.
    regridded = {"plan_id": "p", "domains": [
        {"grid_id": 1, "nx": 98, "history_interval_s": 3600.0},
        {"grid_id": 2, "nx": 200, "history_interval_s": 900.0}]}
    assert _plan_restart_identity(plan) != _plan_restart_identity(regridded)


def _tree_fingerprint(raw, tmp_path, name):
    """The digest the domain-tree runner compares on restore."""

    import hashlib
    from types import MappingProxyType, SimpleNamespace

    from gpuwm import prepared_domain_tree_forecast as tree

    exp = build_experiment(raw, source=name)
    inputs = SimpleNamespace(
        experiment=exp,
        authority_sha256=MappingProxyType({"preparation_receipt": "abcd"}),
        domains=tuple(
            SimpleNamespace(grid_id=d.grid_id,
                            cache_reader=SimpleNamespace(
                                content_sha256=f"cache{d.grid_id}"))
            for d in exp.domains),
        execution_plan=tree.resolve_execution_plan(exp),
    )
    components = tree.tree_restart_identity_components(
        inputs, MappingProxyType({"runtime": "pinned"}))
    return hashlib.sha256(
        tree._canonical(components).encode("utf-8")).hexdigest()


def test_the_tree_fingerprint_survives_all_three_permitted_changes(tmp_path):
    """C-11 end to end at the digest the restore actually compares.

    This is the value `restore_tree_restart` refuses on.  Every one of
    the three changes `--restart` publishes as permitted used to move it,
    because the whole experiment TOML's SHA-256 was one of its
    components -- and the output cadence moved it a second time through
    the execution plan's per-domain rows.
    """

    raw = _raw(_wizard_config(tmp_path, ladder="12-3"))
    assert len(raw["domain"]) == 2
    base = _tree_fingerprint(raw, tmp_path, "base")

    extended = _mutated(raw)
    extended["experiment"]["run_seconds"] = (
        float(raw["experiment"]["run_seconds"]) + 3600.0)
    assert _tree_fingerprint(extended, tmp_path, "extended") == base

    recadenced = _mutated(raw)
    for domain in recadenced["domain"]:
        domain["history_interval_s"] = float(
            domain["history_interval_s"]) / 2.0
    assert _tree_fingerprint(recadenced, tmp_path, "recadenced") == base

    rerestarted = _mutated(raw)
    rerestarted["experiment"]["restart_interval_s"] = 7200.0
    assert _tree_fingerprint(rerestarted, tmp_path, "rerestarted") == base

    # CONTROL: trajectory changes still move it, or the fingerprint
    # would have stopped binding anything.
    physics = _mutated(raw)
    current = int(physics["shared"]["mp_physics"])
    physics["shared"]["mp_physics"] = 6 if current != 6 else 8
    assert _tree_fingerprint(physics, tmp_path, "physics") != base

    stepped = _mutated(raw)
    stepped["domain"][0]["time_step"] = int(
        stepped["domain"][0]["time_step"]) // 2
    assert _tree_fingerprint(stepped, tmp_path, "stepped") != base


def test_the_permitted_keys_are_the_ones_the_help_text_publishes():
    """The tolerance in code IS the tolerance in the published contract.

    Three places answer "which fields may differ across a restart", and
    they must answer it identically: the single-domain restart
    allow-list, the experiment-level restart identity, and the prepared
    cache's non-trajectory partition.  The last of those was missing
    `run_seconds` -- which is why extending a run refused on both
    prepared routes even before the fingerprint was consulted.
    """

    from gpuwm.core.model import (
        RESTART_TOLERATED_DOMAIN_FIELDS, RESTART_TOLERATED_RUN_FIELDS)
    from gpuwm.ingest.prepared_cache import NON_TRAJECTORY_IDENTITY_FIELDS
    from gpuwm.io.restart import CONFIG_RUN_LENGTH_FIELDS

    # The single-domain route's allow-list and the experiment-level one
    # describe the same three user-facing knobs under their two spellings
    # (`output_interval_s` is `history_interval_s` on a RunConfig).
    assert set(RESTART_TOLERATED_RUN_FIELDS) == set(CONFIG_RUN_LENGTH_FIELDS)
    assert RESTART_TOLERATED_DOMAIN_FIELDS == ("history_interval_s",)
    # The prepared-cache document carries the output cadence under BOTH
    # spellings (`run.output_interval_s` and the domain-level
    # `history_interval_s`), so the partition must name both or the
    # cadence stays bound after all.
    assert NON_TRAJECTORY_IDENTITY_FIELDS == frozenset(
        f"run.{name}" for name in CONFIG_RUN_LENGTH_FIELDS
    ) | {"history_interval_s"}


def test_extending_a_run_does_not_move_the_prepared_cache_identity(tmp_path):
    """C-11: the gate a restart hits BEFORE the fingerprint is consulted."""

    from gpuwm.ingest.prepared_cache import (
        compare_prepared_domain_config, effective_prepared_domain_config)

    cached = {"history_interval_s": 3600.0,
              "run": {"nx": 100, "dt": 60.0, "run_seconds": 7200.0,
                      "output_interval_s": 3600.0,
                      "restart_interval_s": 3600.0}}
    extended = {"history_interval_s": 1800.0,
                "run": {**cached["run"], "run_seconds": 10800.0,
                        "output_interval_s": 1800.0,
                        "restart_interval_s": 7200.0}}
    _tolerated, differing = compare_prepared_domain_config(
        effective_prepared_domain_config(cached),
        effective_prepared_domain_config(extended))
    assert differing == []

    # CONTROL: geometry and timestep still move it.
    for key, value in (("nx", 98), ("dt", 30.0)):
        moved = {"run": {**cached["run"], key: value}}
        _tolerated, differing = compare_prepared_domain_config(
            effective_prepared_domain_config(cached),
            effective_prepared_domain_config(moved))
        assert differing == [f"run.{key}"], (key, differing)


def test_radiation_aggregate_and_explicit_spellings_share_one_identity():
    """The historical ra_physics aggregate is a SPELLING, not a selection.

    gpuwm/config.py documents ``ra_lw_physics = ra_sw_physics = -1`` as
    preserving the aggregate exactly, and explicit pairs require
    ``ra_physics = 0``.  A shipped 4/4 profile writes the explicit
    (0, 4, 4) triple; the WRF namelist importer emits the coupled
    aggregate (4, -1, -1); the two describe one radiation selection and
    must bind one prepared root.  A genuinely different pair still
    refuses.
    """

    from gpuwm.ingest.prepared_cache import (
        compare_prepared_domain_config, effective_prepared_domain_config)

    base = {"nx": 100, "dt": 60.0}
    aggregate = {"run": {**base, "ra_physics": 4,
                         "ra_lw_physics": -1, "ra_sw_physics": -1}}
    explicit = {"run": {**base, "ra_physics": 0,
                        "ra_lw_physics": 4, "ra_sw_physics": 4}}
    _tolerated, differing = compare_prepared_domain_config(
        effective_prepared_domain_config(aggregate),
        effective_prepared_domain_config(explicit))
    assert differing == []

    # CONTROL: the aggregate is not equivalent to a DIFFERENT explicit
    # pair -- Dudhia-shortwave (0, 1) still moves the identity.
    dudhia = {"run": {**base, "ra_physics": 0,
                      "ra_lw_physics": 0, "ra_sw_physics": 1}}
    _tolerated, differing = compare_prepared_domain_config(
        effective_prepared_domain_config(aggregate),
        effective_prepared_domain_config(dudhia))
    assert sorted(differing) == ["run.ra_lw_physics", "run.ra_sw_physics"]


def test_the_route_advisory_carries_no_case_or_source_token():
    """Generic code names no case and no source (project standing rule)."""

    from gpuwm import checkpoint_routes

    source = Path(checkpoint_routes.__file__).read_text(encoding="utf-8")
    for token in ("real74", "ohio", "hrrr", "gfs", "era5", "1974"):
        assert token not in source.lower(), token
    assert "restart_interval_s" in CHECKPOINTLESS_ROUTE_ADVISORY
