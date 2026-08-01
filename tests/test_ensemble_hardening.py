"""The ensemble engine's failure modes, each pinned by the case that broke it.

Every test here corresponds to a defect that was reachable in the shipped
engine and is now a refusal, a correction, or a recorded fact.  They are
written against the behaviour rather than the patch: each one fails on the
code as it stood, and none of them widens an existing gate.

The cycling tests use a member runner that implements the REAL integrator's
clock contract -- ``run_seconds`` is the total forecast length from the
experiment start, a restart restores its own elapsed time, and a restart
that has already reached the total is a refusal.  That is the whole reason
the second leg was broken and the reason a runner that ignores the argument
could not see it.
"""

from __future__ import annotations

import ast
import json
import types

import numpy as np
import pytest

from gpuwm.ensemble import cycle as cycle_module
from gpuwm.ensemble.config import load_ensemble_config
from gpuwm.ensemble.cycle import (
    ANALYSIS_NAME, PUBLICATION_MARKER_NAME, _leg_horizon, cycle_root,
    publication_marker_path, recover_analysis_publication, run_cycles,
)
from gpuwm.ensemble.engine import prepare_ensemble, run_ensemble
from gpuwm.ensemble.increments import (
    STAGED_SUFFIX, apply_increments, apply_increments_to_checkpoint,
)
from gpuwm.ensemble.manifest import (
    CYCLE_MANIFEST_SCHEMA, ENSEMBLE_MANIFEST_NAME, ENSEMBLE_MANIFEST_SCHEMA,
    member_directory_name, read_manifest, seed_hex,
)
from gpuwm.ensemble.member import MemberOutcome
from gpuwm.ensemble.seeds import member_seed
from gpuwm.ensemble.state_sha import (
    checkpoint_state_sha_receipt, live_state_sha256, live_state_sha_receipt,
    serialized_state_attrs,
)

BASE_TOML = """
[experiment]
name = "ensemble_hardening_unit"
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


def _write_overlay(directory, *, n_members=2, base_seed=20260730,
                   perturbation="none", options=None, extra_comment=""):
    directory.mkdir(parents=True, exist_ok=True)
    base = directory / "base.toml"
    base.write_text(BASE_TOML, encoding="utf-8")
    body = [
        "[ensemble]",
        'base_config = "base.toml"',
        f"n_members = {n_members}",
        f"base_seed = {base_seed}",
        f'perturbation = "{perturbation}"',
    ]
    if options:
        body.append("")
        body.append("[ensemble.perturbation_options]")
        for key, value in options.items():
            body.append(f"{key} = {json.dumps(value)}")
    if extra_comment:
        body.append(f"# {extra_comment}")
    path = directory / "ensemble.toml"
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    return path


def _contract_names(count=3):
    return serialized_state_attrs()[:count]


CHECKPOINT_NAME = "gpuwmrst_leg.npz"


def _write_leg_checkpoint(member_dir, value, elapsed_seconds):
    names = _contract_names()
    payload = {f"state/{name}": np.full((2, 3), float(value), np.float32)
               for name in names}
    payload["meta/elapsed_seconds"] = np.asarray(float(elapsed_seconds))
    np.savez(member_dir / CHECKPOINT_NAME, **payload)


def _checkpoint_elapsed(path):
    with np.load(path, allow_pickle=False) as data:
        return float(data["meta/elapsed_seconds"])


def clock_honest_runner(*, base_config, member_dir, index, seed,
                        perturbation, perturbation_options,
                        run_seconds=None, restart=None, **_):
    """A member that obeys gpuwm.runtime's own ``run_seconds`` contract.

    ``run_seconds`` is the TOTAL forecast length from the experiment start
    time; a restart resumes at its stored elapsed time and integrates to
    that total; a restart already at the total is the refusal the real
    integrator raises.  A runner that treats ``run_seconds`` as a leg
    length cannot see the defect this file exists to pin.
    """
    member_dir.mkdir(parents=True, exist_ok=True)
    total = float(run_seconds if run_seconds is not None else 120.0)
    if restart is None:
        start = 0.0
        value = float(index)
    else:
        start = _checkpoint_elapsed(restart)
        with np.load(restart, allow_pickle=False) as data:
            value = float(data[f"state/{_contract_names()[0]}"].ravel()[0])
    if start >= total:
        raise ValueError(
            f"restart file is already at {start} s; nothing to integrate "
            f"before run_seconds={total}")
    _write_leg_checkpoint(member_dir, value + 1.0, total)
    return MemberOutcome(
        index=index, seed=seed, member_dir=member_dir,
        initial_state_sha256=f"{index:064d}",
        final_state_sha256=f"{index:063d}{int(total) % 10}",
        wall_seconds=0.1, sim_seconds=total - start, wrfout_count=1,
        last_checkpoint=str(member_dir / CHECKPOINT_NAME),
        perturbation={"restart_from": None if restart is None
                      else str(restart)})


# ------------------------------------------------------- F-01 second leg


def test_the_second_leg_is_given_the_cumulative_horizon(tmp_path):
    """The defect: leg N+1 was handed the leg length, not the total.

    ``integrate_prepared_case`` measures ``run_seconds`` from the
    experiment's start time and refuses a restart already standing on it,
    so cycle 1 died with "restart file is already at 60.0 s; nothing to
    integrate before run_seconds=60.0" on a real GPU run.
    """

    cfg = load_ensemble_config(_write_overlay(tmp_path / "a", n_members=2))
    root = tmp_path / "a" / "ens"

    def assimilate(cycle_index, member_states):
        names = _contract_names()
        return {index: {names[0]: np.full((2, 3), 0.25, np.float32)}
                for index in member_states}

    result = run_cycles(cfg, root, n_cycles=3, cycle_seconds=60.0,
                        assimilate=assimilate, runner=clock_honest_runner)
    assert result.cycles_run == (0, 1, 2)
    manifest = read_manifest(result.manifest_path,
                             schema=CYCLE_MANIFEST_SCHEMA)
    horizons = [entry["run_seconds_total"] for entry in manifest["cycles"]]
    assert horizons == [60.0, 120.0, 180.0], (
        "leg N restarting from an analysis must be given the cumulative "
        "horizon; the leg duration is what the integrator refuses")
    # Each leg still advanced exactly one leg length.
    for cycle_index in range(3):
        leg = read_manifest(
            cycle_root(root, cycle_index) / ENSEMBLE_MANIFEST_NAME,
            schema=ENSEMBLE_MANIFEST_SCHEMA)
        assert [record["sim_seconds"] for record in leg["members"]] \
            == [60.0, 60.0]


def test_a_forecast_only_cycle_still_runs_leg_length_horizons(tmp_path):
    """Without restart-from-analysis every leg starts at zero again."""

    cfg = load_ensemble_config(_write_overlay(tmp_path / "b", n_members=1))
    result = run_cycles(cfg, tmp_path / "b" / "ens", n_cycles=2,
                        cycle_seconds=60.0, runner=clock_honest_runner,
                        restart_from_analysis=False)
    manifest = read_manifest(result.manifest_path,
                             schema=CYCLE_MANIFEST_SCHEMA)
    assert [entry["run_seconds_total"] for entry in manifest["cycles"]] \
        == [60.0, 60.0]
    assert not any(entry["restart_clocks"]["restarted"]
                   for entry in manifest["cycles"])


def test_a_restart_whose_clock_disagrees_with_the_timeline_is_refused(
        tmp_path):
    """A leg cannot integrate members standing at different times."""

    previous = tmp_path / "cycle_000"
    for index in range(2):
        member = previous / member_directory_name(index)
        member.mkdir(parents=True)
        _write_leg_checkpoint(member, index, 60.0 if index == 0 else 45.0)
        (member / CHECKPOINT_NAME).replace(member / ANALYSIS_NAME)
    restarts = {index: previous / member_directory_name(index) / ANALYSIS_NAME
                for index in range(2)}
    with pytest.raises(ValueError, match="restart from a different clock"):
        _leg_horizon(restarts, 1, 60.0)
    # The agreeing case is the cumulative horizon and says so.
    _write_leg_checkpoint(previous / member_directory_name(1), 1, 60.0)
    (previous / member_directory_name(1) / CHECKPOINT_NAME).replace(
        restarts[1])
    seconds, clocks = _leg_horizon(restarts, 1, 60.0)
    assert seconds == 120.0
    assert clocks["stated_start_seconds"] == {0: 60.0, 1: 60.0}


# --------------------------------------------- F-02 atomicity and resume


def test_a_refusal_on_a_later_member_publishes_no_analysis_at_all(tmp_path):
    """The defect: member 0's analysis was live when member 1 refused."""

    cfg = load_ensemble_config(_write_overlay(tmp_path / "c", n_members=2))
    root = tmp_path / "c" / "ens"
    names = _contract_names()

    def assimilate(cycle_index, member_states):
        return {
            0: {names[0]: np.full((2, 3), 0.25, np.float32)},
            1: {names[0]: np.zeros((7, 7), np.float32)},  # wrong shape
        }

    with pytest.raises(ValueError, match="shape"):
        run_cycles(cfg, root, n_cycles=1, cycle_seconds=60.0,
                   assimilate=assimilate, runner=clock_honest_runner)
    for index in range(2):
        member = cycle_root(root, 0) / member_directory_name(index)
        assert not (member / ANALYSIS_NAME).exists(), (
            f"member {index} was published although the ensemble's "
            "assimilation refused")
        assert not (member / (ANALYSIS_NAME + STAGED_SUFFIX)).exists(), \
            "a failed phase one must not leave staged files behind"
        assert (member / CHECKPOINT_NAME).is_file(), \
            "the background must survive a refused analysis"


def test_an_incomplete_roster_is_refused_before_anything_is_written(tmp_path):
    cfg = load_ensemble_config(_write_overlay(tmp_path / "d", n_members=3))
    root = tmp_path / "d" / "ens"
    names = _contract_names()

    with pytest.raises(ValueError, match=r"member\(s\) \[2\] got no"):
        run_cycles(cfg, root, n_cycles=1, cycle_seconds=60.0,
                   assimilate=lambda _i, states: {
                       index: {names[0]: np.zeros((2, 3), np.float32)}
                       for index in (0, 1)},
                   runner=clock_honest_runner)
    assert not list(cycle_root(root, 0).rglob(ANALYSIS_NAME))


def test_a_crash_during_assimilation_leaves_one_resumable_cycle_record(
        tmp_path):
    """The defect: the retry appended a second entry for the same cycle."""

    cfg = load_ensemble_config(_write_overlay(tmp_path / "e", n_members=2))
    root = tmp_path / "e" / "ens"
    names = _contract_names()
    calls = {"n": 0}

    def flaky(cycle_index, member_states):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("the filter fell over")
        return {index: {names[0]: np.full((2, 3), 0.25, np.float32)}
                for index in member_states}

    with pytest.raises(RuntimeError, match="fell over"):
        run_cycles(cfg, root, n_cycles=1, cycle_seconds=60.0,
                   assimilate=flaky, runner=clock_honest_runner)
    crashed = read_manifest(root / "da-cycle-manifest.json",
                            schema=CYCLE_MANIFEST_SCHEMA)
    assert [(entry["cycle"], entry["status"]) for entry in crashed["cycles"]] \
        == [(0, "FORECAST_COMPLETE")]

    result = run_cycles(cfg, root, n_cycles=1, cycle_seconds=60.0,
                        assimilate=flaky, runner=clock_honest_runner)
    resumed = read_manifest(result.manifest_path,
                            schema=CYCLE_MANIFEST_SCHEMA)
    assert [(entry["cycle"], entry["status"]) for entry in resumed["cycles"]] \
        == [(0, "DONE")], "one cycle is one record, however many attempts"
    assert resumed["cycles"][0]["attempt"] == 2
    assert resumed["cycles"][0]["assimilation"]["attempt"] == 2


def test_a_crash_on_the_second_rename_never_leaves_a_silent_mixed_roster(
        tmp_path, monkeypatch):
    """The probe that reopened F-02, promoted to a permanent test.

    Publication was a bare rename loop.  Forcing a failure on rename 2 of
    2 left member 0's analysis LIVE and member 1's merely staged, with
    nothing on disk saying which of the two states the leg was in -- and
    the next start could not tell an interrupted publication from an
    experiment that had genuinely analysed one member.

    A rename is atomic for one file and there is no call that makes two
    of them one operation, so what the fix guarantees is the recoverable
    form: the transaction is declared before the first rename, so the
    mixed roster is DETECTED, and recovery rolls it forward.
    """

    cfg = load_ensemble_config(_write_overlay(tmp_path / "p", n_members=2))
    root = tmp_path / "p" / "ens"
    names = _contract_names()
    calls = {"n": 0}
    real_publish = cycle_module.publish_staged_analysis

    def flaky_publish(staged, destination):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("the disk went away between renames")
        return real_publish(staged, destination)

    monkeypatch.setattr(cycle_module, "publish_staged_analysis",
                        flaky_publish)
    with pytest.raises(OSError, match="between renames"):
        run_cycles(cfg, root, n_cycles=1, cycle_seconds=60.0,
                   assimilate=lambda _i, states: {
                       index: {names[0]: np.full((2, 3), 0.25, np.float32)}
                       for index in states},
                   runner=clock_honest_runner)

    leg = cycle_root(root, 0)
    # On disk the roster IS mixed; that is what a partial rename loop
    # leaves and no filesystem call prevents it.
    assert (leg / member_directory_name(0) / ANALYSIS_NAME).is_file()
    assert not (leg / member_directory_name(1) / ANALYSIS_NAME).exists()
    assert (leg / member_directory_name(1)
            / (ANALYSIS_NAME + STAGED_SUFFIX)).is_file()
    # And the marker says so, naming every member of the transaction.
    marker = publication_marker_path(leg)
    assert marker.is_file()
    declared = json.loads(marker.read_text(encoding="utf-8"))
    assert declared["cycle"] == 0
    assert [entry["member"] for entry in declared["members"]] == [0, 1]

    report = recover_analysis_publication(leg)
    assert report["rolled_forward"] == [1]
    assert report["already_live"] == [0]
    for index in range(2):
        assert (leg / member_directory_name(index) / ANALYSIS_NAME).is_file()
        assert not (leg / member_directory_name(index)
                    / (ANALYSIS_NAME + STAGED_SUFFIX)).exists()
    assert not marker.exists(), "a settled transaction leaves no marker"
    # Idempotent: recovering a settled leg is a no-op, not a second pass.
    assert recover_analysis_publication(leg) is None


def test_a_reader_of_an_interrupted_publication_gets_the_whole_roster(
        tmp_path, monkeypatch):
    """The next start, not a hand call to the recovery function.

    ``run_cycles`` settles every leg before it reads one, so the retry
    that follows a crashed publication sees ten analyses or a refusal --
    never the six-of-ten roster that made the ensemble scientifically
    heterogeneous while every number in the manifest stayed true.
    """

    cfg = load_ensemble_config(_write_overlay(tmp_path / "q", n_members=2))
    root = tmp_path / "q" / "ens"
    names = _contract_names()
    calls = {"n": 0}
    real_publish = cycle_module.publish_staged_analysis

    def flaky_publish(staged, destination):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("the disk went away between renames")
        return real_publish(staged, destination)

    def assimilate(_index, states):
        return {index: {names[0]: np.full((2, 3), 0.25, np.float32)}
                for index in states}

    monkeypatch.setattr(cycle_module, "publish_staged_analysis",
                        flaky_publish)
    with pytest.raises(OSError):
        run_cycles(cfg, root, n_cycles=2, cycle_seconds=60.0,
                   assimilate=assimilate, runner=clock_honest_runner)
    assert publication_marker_path(cycle_root(root, 0)).is_file()

    monkeypatch.setattr(cycle_module, "publish_staged_analysis",
                        real_publish)
    result = run_cycles(cfg, root, n_cycles=2, cycle_seconds=60.0,
                        assimilate=assimilate, runner=clock_honest_runner)
    assert result.status == "COMPLETE"
    assert not publication_marker_path(cycle_root(root, 0)).exists()
    for index in range(2):
        assert (cycle_root(root, 0) / member_directory_name(index)
                / ANALYSIS_NAME).is_file()
    # Cycle 1 restarted from a complete analysis roster.
    manifest = read_manifest(result.manifest_path,
                             schema=CYCLE_MANIFEST_SCHEMA)
    assert manifest["cycles"][1]["restart_clocks"]["restarted"] is True


def test_an_unrecoverable_publication_refuses_loudly_rather_than_guessing(
        tmp_path, monkeypatch):
    """Neither published nor staged: the transaction cannot be completed.

    Rolling forward is only safe because the staged bytes are the same
    analysis the published members carry.  With the staged file gone
    there is nothing to roll forward FROM, and inventing one -- or
    quietly proceeding with the members that did land -- is the
    half-analysed ensemble this contract exists to refuse.
    """

    cfg = load_ensemble_config(_write_overlay(tmp_path / "r", n_members=2))
    root = tmp_path / "r" / "ens"
    names = _contract_names()
    calls = {"n": 0}
    real_publish = cycle_module.publish_staged_analysis

    def flaky_publish(staged, destination):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("the disk went away between renames")
        return real_publish(staged, destination)

    monkeypatch.setattr(cycle_module, "publish_staged_analysis",
                        flaky_publish)
    with pytest.raises(OSError):
        run_cycles(cfg, root, n_cycles=1, cycle_seconds=60.0,
                   assimilate=lambda _i, states: {
                       index: {names[0]: np.full((2, 3), 0.25, np.float32)}
                       for index in states},
                   runner=clock_honest_runner)

    leg = cycle_root(root, 0)
    (leg / member_directory_name(1)
     / (ANALYSIS_NAME + STAGED_SUFFIX)).unlink()
    with pytest.raises(ValueError) as caught:
        recover_analysis_publication(leg)
    message = str(caught.value)
    assert "[1]" in message
    assert "roster is mixed" in message
    assert publication_marker_path(leg).is_file(), (
        "an unrecoverable transaction keeps its marker; clearing it would "
        "make the next start read the mixed roster as a whole one")


# ------------------------------------------- F-02 the consumer contract
#
# Re-verification's scope qualification: recovery is correct at every
# simulated failure point for the in-tree reader, but a raw filesystem
# observer can see the sequential-renaming middle.  A rename is atomic for
# one file and no call makes N of them one operation, so "no mixed live
# roster is ever OBSERVABLE" is not a property a rename loop can have.
#
# The contract is therefore stated instead of implied: read_analysis_roster
# is the only supported way to observe a leg's analyses, it settles the
# transaction before it reports anything, and a raw directory scan is out
# of contract.  These tests are that contract.


_PY_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _scope_header_nodes(node):
    """The expressions a definition evaluates in its ENCLOSING scope.

    Everything a ``def`` or ``class`` statement carries except its body:
    decorators, default arguments, keyword-only defaults, parameter and
    return annotations, base classes and class keywords.  Python evaluates
    every one of them where the definition is written, not inside it.

    Round 4 walked scope *bodies* and nothing else, so each of these was
    reached by neither the enclosing scope -- which skipped the definition
    node outright -- nor the definition's own walk, which saw only
    ``body``.  Re-verification #6 measured it at 8 of 8 probes missed,
    including ``def read(leg, name="analysis.npz")``: a complete, working,
    out-of-contract roster consumer needing no import at all.
    """

    header = list(getattr(node, "decorator_list", ()))
    header.extend(getattr(node, "type_params", ()) or ())
    header.extend(getattr(node, "bases", ()))
    header.extend(getattr(node, "keywords", ()))
    arguments = getattr(node, "args", None)
    if arguments is not None:
        header.append(arguments)
    returns = getattr(node, "returns", None)
    if returns is not None:
        header.append(returns)
    return header


def _scope_attribution(tree):
    """``[(label, nodes)]`` -- every node in ``tree``, in exactly one scope.

    Outermost first, in source order.  The partition is the property that
    matters, and it is asserted directly by
    ``test_consumer_guard_attributes_every_node_to_exactly_one_scope``: a
    guard that skips a position cannot see a consumer written there, and
    "the walk is over scopes" is only true if nothing falls between them.

    Three rules, and they are Python's own:

    * a scope's ``body`` belongs to that scope;
    * a definition's *header* (:func:`_scope_header_nodes`) belongs to the
      scope the definition is written in, because that is where Python
      evaluates it -- so a decorator on a nested function is the enclosing
      function's code, not the nested one's;
    * a lambda is an expression, not a scope, so it belongs whole to
      whichever scope carries it.

    Docstrings are dropped -- several functions and modules describe the
    contract in prose without touching the roster -- and so are comments,
    by virtue of walking syntax rather than text.
    """

    def own_statements(node):
        statements = list(node.body)
        if statements and isinstance(statements[0], ast.Expr) \
                and isinstance(statements[0].value, ast.Constant) \
                and isinstance(statements[0].value.value, str):
            statements = statements[1:]
        return statements

    out = []

    def visit(label, node):
        mine = []
        nested = []
        pending = list(own_statements(node))
        while pending:
            current = pending.pop(0)
            mine.append(current)
            if isinstance(current, _PY_SCOPES):
                # Walked as its own scope below -- but what is written ON
                # the definition is evaluated here, so it is this scope's.
                nested.append(current)
                pending.extend(_scope_header_nodes(current))
                continue
            pending.extend(ast.iter_child_nodes(current))
        out.append((label, mine))
        for child in nested:
            visit(child.name, child)

    visit("<module>", tree)
    return out


def _analysis_roster_consumer_offenders(package):
    """Scopes that name the analysis file outside its supported doors.

    Every scope, not every function, and every position in each scope --
    see :func:`_scope_attribution`.  Re-verification #5 walked only
    ``FunctionDef``/``AsyncFunctionDef``, so a module-level
    ``ROSTER = cycle.ANALYSIS_NAME`` was invisible; re-verification #6
    walked only scope bodies, so ``def read(leg, name="analysis.npz")``
    was.  A guard whose whole job is that there is one door has to reach
    every place a second one can be written.
    """

    # The scopes that legitimately touch the file name: the module that
    # DEFINES it, the writer, the recovery pass that settles a transaction,
    # and the supported reader.
    allowed = {
        "gpuwm/ensemble/cycle.py": {
            "<module>",                      # ANALYSIS_NAME is defined here
            "_assimilate_cycle",             # the writer
            "recover_analysis_publication",  # settles the transaction
            "read_analysis_roster",          # the supported reader
        },
    }

    def names_the_analysis_file(nodes, imported_aliases):
        """Does this scope's own code name the analysis file?"""

        for node in nodes:
            if isinstance(node, ast.Name) and node.id in imported_aliases:
                return True
            if isinstance(node, ast.Attribute) \
                    and node.attr == "ANALYSIS_NAME":
                return True
            if isinstance(node, ast.Constant) \
                    and node.value == "analysis.npz":
                return True
        return False

    offenders = []
    for path in sorted(package.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "ANALYSIS_NAME" not in text and "analysis.npz" not in text:
            continue
        relative = path.relative_to(package.parent).as_posix()
        tree = ast.parse(text)
        imported_aliases = {"ANALYSIS_NAME"}
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            for alias in node.names:
                if alias.name == "ANALYSIS_NAME":
                    imported_aliases.add(alias.asname or alias.name)
        for label, nodes in _scope_attribution(tree):
            if not names_the_analysis_file(nodes, imported_aliases):
                continue
            if label in allowed.get(relative, set()):
                continue
            offenders.append(f"{relative}:{label}")
    return offenders


def _publish_failing_at(monkeypatch, nth):
    """Make the ``nth`` rename of the publication loop fail."""
    calls = {"n": 0}
    real_publish = cycle_module.publish_staged_analysis

    def flaky_publish(staged, destination):
        calls["n"] += 1
        if calls["n"] == nth:
            raise OSError("the disk went away between renames")
        return real_publish(staged, destination)

    monkeypatch.setattr(cycle_module, "publish_staged_analysis",
                        flaky_publish)
    return real_publish


def _interrupted_leg(tmp_path, monkeypatch, *, nth, n_members=3,
                     directory="cc"):
    """A leg whose publication was interrupted at rename ``nth``."""
    cfg = load_ensemble_config(
        _write_overlay(tmp_path / directory, n_members=n_members))
    root = tmp_path / directory / "ens"
    names = _contract_names()
    _publish_failing_at(monkeypatch, nth)
    with pytest.raises(OSError, match="between renames"):
        run_cycles(cfg, root, n_cycles=1, cycle_seconds=60.0,
                   assimilate=lambda _i, states: {
                       index: {names[0]: np.full((2, 3), 0.25, np.float32)}
                       for index in states},
                   runner=clock_honest_runner)
    return root


@pytest.mark.parametrize("nth, raw_live", [(1, 0), (2, 1), (3, 2)])
def test_the_supported_reader_returns_the_whole_roster_at_every_cut(
        tmp_path, monkeypatch, nth, raw_live):
    """Three members, publication interrupted at each rename in turn.

    ``raw_live`` is what a raw directory scan would have reported at that
    cut -- 0, 1, and 2 of 3 -- and it is asserted, because the point is
    not that the middle is invisible.  It is that the middle is not what
    the supported reader reports: the marker names all three members, and
    ``read_analysis_roster`` settles the transaction before it looks, so
    the roster it returns is three at every cut.
    """

    root = _interrupted_leg(tmp_path, monkeypatch, nth=nth,
                            directory=f"cut{nth}")
    leg = cycle_root(root, 0)

    scanned = [index for index in range(3)
               if (leg / member_directory_name(index)
                   / ANALYSIS_NAME).is_file()]
    assert len(scanned) == raw_live, (
        "the raw middle is real; the contract is that nothing reads it")
    marker = json.loads(
        publication_marker_path(leg).read_text(encoding="utf-8"))
    assert [entry["member"] for entry in marker["members"]] == [0, 1, 2]

    roster = cycle_module.read_analysis_roster(leg, n_members=3)
    assert sorted(roster) == [0, 1, 2]
    assert all(path.is_file() for path in roster.values())
    assert not publication_marker_path(leg).exists()
    # Idempotent: a settled leg reads the same way twice.
    assert sorted(cycle_module.read_analysis_roster(leg, n_members=3)) \
        == [0, 1, 2]


def test_the_supported_reader_refuses_an_unrecoverable_roster(
        tmp_path, monkeypatch):
    """Neither published nor staged is a refusal, through the reader too.

    The reader is not a softer door onto the same directory: everything
    ``recover_analysis_publication`` refuses, it refuses, because it is
    the first thing the reader does.
    """

    root = _interrupted_leg(tmp_path, monkeypatch, nth=2, n_members=2,
                            directory="dd")
    leg = cycle_root(root, 0)
    (leg / member_directory_name(1)
     / (ANALYSIS_NAME + STAGED_SUFFIX)).unlink()
    with pytest.raises(ValueError, match="roster is mixed"):
        cycle_module.read_analysis_roster(leg, n_members=2)


def test_the_supported_reader_reports_a_forecast_only_leg_as_empty(tmp_path):
    """Zero analyses is an answer, and it is not the same as a refusal."""

    cfg = load_ensemble_config(_write_overlay(tmp_path / "ee", n_members=2))
    root = tmp_path / "ee" / "ens"
    run_cycles(cfg, root, n_cycles=1, cycle_seconds=60.0,
               runner=clock_honest_runner)
    leg = cycle_root(root, 0)
    assert cycle_module.read_analysis_roster(leg, n_members=2) == {}
    with pytest.raises(TypeError, match="n_members"):
        cycle_module.read_analysis_roster(leg)


def test_the_supported_reader_refuses_a_partial_roster_with_no_transaction(
        tmp_path):
    """Some analysed, some not, and no marker to explain it.

    This is the state an operator can produce by hand, and the one the
    marker cannot speak for because the transaction completed.  Reporting
    the members that are there would be reporting half an ensemble.
    """

    cfg = load_ensemble_config(_write_overlay(tmp_path / "ff", n_members=2))
    root = tmp_path / "ff" / "ens"
    names = _contract_names()
    run_cycles(cfg, root, n_cycles=1, cycle_seconds=60.0,
               assimilate=lambda _i, states: {
                   index: {names[0]: np.full((2, 3), 0.25, np.float32)}
                   for index in states},
               runner=clock_honest_runner)
    leg = cycle_root(root, 0)
    assert not publication_marker_path(leg).exists()
    (leg / member_directory_name(1) / ANALYSIS_NAME).unlink()
    with pytest.raises(TypeError, match="n_members"):
        cycle_module.read_analysis_roster(leg)
    with pytest.raises(ValueError, match="not one ensemble"):
        cycle_module.read_analysis_roster(leg, n_members=2)


def _fully_analysed_leg(tmp_path, directory, n_members):
    """Leg 0 of a run in which every member was analysed and published."""

    cfg = load_ensemble_config(
        _write_overlay(tmp_path / directory, n_members=n_members))
    root = tmp_path / directory / "ens"
    names = _contract_names()
    run_cycles(cfg, root, n_cycles=1, cycle_seconds=60.0,
               assimilate=lambda _i, states: {
                   index: {names[0]: np.full((2, 3), 0.25, np.float32)}
                   for index in states},
               runner=clock_honest_runner)
    leg = cycle_root(root, 0)
    for index in range(n_members):
        assert (leg / member_directory_name(index) / ANALYSIS_NAME).is_file()
    return leg


def test_the_supported_reader_refuses_a_count_that_is_not_a_positive_int(
        tmp_path):
    """RV5-03: requiring a count is not the same as believing one.

    ``n_members`` became required and was then used unchecked.  ``0`` and
    ``-1`` made ``range()`` empty, so a fully analysed leg read as ``{}``
    -- and ``{}`` means "forecast-only" to ``_analysis_restarts``, which
    restarts from the base config.  Three analyses computed, receipted,
    written, and discarded, with nothing raised and every number in the
    manifest still true.
    """

    leg = _fully_analysed_leg(tmp_path, "rv5count", 3)
    # The control: the count the leg was written with reads the whole leg.
    assert sorted(cycle_module.read_analysis_roster(leg, n_members=3)) \
        == [0, 1, 2]

    for count in (0, -1, -3):
        with pytest.raises(ValueError, match="at least one member") as caught:
            cycle_module.read_analysis_roster(leg, n_members=count)
        assert "forecast-only" in str(caught.value)

    # True is an int in Python, so a flag passed where a count belongs
    # bought a one-member ensemble; 2.9 truncated; "3" converted.  The
    # repository refuses this exact class in SuperobParams and in the
    # [ensemble] config reader, and now here.
    for count in (True, False, 2.9, 3.0, "3", None, 3 + 0j):
        with pytest.raises(TypeError, match="positive int"):
            cycle_module.read_analysis_roster(leg, n_members=count)


def test_the_supported_reader_falsifies_the_count_against_the_leg(tmp_path):
    """RV5-03: the leg's own directories can contradict the count.

    The reader is already walking the leg, so it can see ``member_002``.
    Two against three analysed members used to return two of them and call
    it one ensemble; the third was never mentioned.  This is the case
    checkpoint tree-support is made of, where a branch's member count comes
    from a tree node rather than from the config that wrote the leg.
    """

    leg = _fully_analysed_leg(tmp_path, "rv5leg", 3)

    with pytest.raises(ValueError) as caught:
        cycle_module.read_analysis_roster(leg, n_members=2)
    message = str(caught.value)
    assert "n_members=2" in message, message
    assert "[0, 1, 2]" in message, "the refusal names both numbers"
    assert "[2]" in message and "beyond the declared count" in message

    # And a count larger than the leg is the same contradiction the other
    # way round: the directories are evidence even where no analysis is.
    with pytest.raises(ValueError) as caught:
        cycle_module.read_analysis_roster(leg, n_members=5)
    assert "n_members=5" in str(caught.value)
    assert "[0, 1, 2]" in str(caught.value)

    # The control, again, after both refusals: nothing was consumed.
    assert sorted(cycle_module.read_analysis_roster(leg, n_members=3)) \
        == [0, 1, 2]


def test_a_member_directory_spelled_in_unicode_digits_is_not_canonical(
        tmp_path):
    """RV6-05: ``str.isdigit`` is true for characters ``int`` refuses.

    ``"²".isdigit()`` is ``True`` and ``int("²")`` raises, so a
    directory named ``member_²`` beside a real leg used to leave the
    falsification scan through a bare ``invalid literal for int() with
    base 10`` -- a message about a conversion, in a function whose whole
    output is a statement about a roster.  It fails closed either way; the
    fix is that the leg still reads, because that spelling is not one
    ``member_directory_name`` composes and so is not a member of anything.
    """

    leg = _fully_analysed_leg(tmp_path, "rv6nit", 3)
    for name in ("member_²", "member_٣", "member_0003"):
        (leg / name).mkdir()

    assert cycle_module._member_directory_indices(leg) == [0, 1, 2]
    assert sorted(cycle_module.read_analysis_roster(leg, n_members=3)) \
        == [0, 1, 2]


def test_a_forecast_only_leg_is_still_falsified_against_its_count(tmp_path):
    """``{}`` means forecast-only, and only when the count was honoured."""

    cfg = load_ensemble_config(_write_overlay(tmp_path / "rv5fo",
                                              n_members=3))
    root = tmp_path / "rv5fo" / "ens"
    run_cycles(cfg, root, n_cycles=1, cycle_seconds=60.0,
               runner=clock_honest_runner)
    leg = cycle_root(root, 0)
    assert cycle_module.read_analysis_roster(leg, n_members=3) == {}
    with pytest.raises(ValueError, match="n_members=2"):
        cycle_module.read_analysis_roster(leg, n_members=2)


def test_every_analysis_roster_consumer_goes_through_the_reader():
    """The contract is only a contract if a new consumer cannot skip it.

    ``ANALYSIS_NAME`` names the file a leg's analyses land in.  Anything
    that mentions it is either publishing one, recovering one, or reading
    the roster -- and only the last is a consumer.  Every consumer in this
    tree must be ``read_analysis_roster``, so this walks the package and
    fails on a mention anywhere else.

    A source-level check because that is the shape of the requirement: it
    is not about what one call does, it is about there being one door.
    """

    import pathlib

    package = pathlib.Path(cycle_module.__file__).resolve().parent.parent
    offenders = _analysis_roster_consumer_offenders(package)
    assert not offenders, (
        "these functions observe a leg's analyses without going through "
        f"gpuwm.ensemble.cycle.read_analysis_roster: {offenders}. A raw "
        "directory scan can see the middle of a publication; the reader "
        "settles the transaction first, which is why it is the only "
        "supported way to observe a roster")


def test_consumer_guard_flags_qualified_and_import_aliased_constant_uses(
        tmp_path):
    """RV4-04: the source tripwire covers the audit's qualified bypass."""

    package = tmp_path / "gpuwm"
    (package / "ensemble").mkdir(parents=True)
    (package / "ensemble" / "cycle.py").write_text(
        "# synthetic scan root\n", encoding="utf-8")
    (package / "bad_consumer.py").write_text(
        "import gpuwm.ensemble.cycle as cycle\n"
        "from gpuwm.ensemble.cycle import ANALYSIS_NAME as analysis_file\n"
        "def illicit_reader():\n"
        "    return cycle.ANALYSIS_NAME\n"
        "def aliased_reader():\n"
        "    return analysis_file\n",
        encoding="utf-8")

    assert _analysis_roster_consumer_offenders(package) == [
        "gpuwm/bad_consumer.py:illicit_reader",
        "gpuwm/bad_consumer.py:aliased_reader",
    ]


def _synthetic_scan_root(tmp_path):
    """A package tree whose ``cycle.py`` exists but names nothing."""

    package = tmp_path / "gpuwm"
    (package / "ensemble").mkdir(parents=True)
    (package / "ensemble" / "cycle.py").write_text(
        "# synthetic scan root\n", encoding="utf-8")
    return package


def test_consumer_guard_flags_module_class_and_lambda_scope_uses(tmp_path):
    """RV5-04: the tripwire walked only function definitions.

    A5's ``read = lambda ...``, A3's module-level assignment and A4's
    class-body assignment are the PLAINEST spellings there are -- not
    computed, not dynamic -- and a guard that only entered function
    definitions could not see any of them.  Each is written here the way
    the audit wrote it.
    """

    package = _synthetic_scan_root(tmp_path)
    (package / "class_scope.py").write_text(
        "import gpuwm.ensemble.cycle as cycle\n"
        "class Reader:\n"
        "    name = cycle.ANALYSIS_NAME\n",
        encoding="utf-8")
    (package / "lambda_scope.py").write_text(
        "import gpuwm.ensemble.cycle as cycle\n"
        "read = lambda leg: leg / cycle.ANALYSIS_NAME\n",
        encoding="utf-8")
    (package / "literal_scope.py").write_text(
        'NAME = "analysis.npz"\n', encoding="utf-8")
    (package / "module_scope.py").write_text(
        "import gpuwm.ensemble.cycle as cycle\n"
        "ROSTER = cycle.ANALYSIS_NAME\n",
        encoding="utf-8")

    assert _analysis_roster_consumer_offenders(package) == [
        "gpuwm/class_scope.py:Reader",
        "gpuwm/lambda_scope.py:<module>",
        "gpuwm/literal_scope.py:<module>",
        "gpuwm/module_scope.py:<module>",
    ]


def test_consumer_guard_attributes_a_mention_to_its_innermost_scope(tmp_path):
    """One scope per mention, and prose is still not a mention.

    A nested definition is walked as its own scope rather than as part of
    its parent's, so a method is reported once under its own name and not
    also under the class that holds it -- and a docstring or a comment
    describing the contract is not a consumer of it.
    """

    package = _synthetic_scan_root(tmp_path)
    (package / "nested_scope.py").write_text(
        "import gpuwm.ensemble.cycle as cycle\n"
        "class Reader:\n"
        '    """Reads analysis.npz, it says here."""\n'
        "    def read(self, leg):\n"
        "        return leg / cycle.ANALYSIS_NAME\n"
        "    def outer(self):\n"
        "        def inner():\n"
        "            return cycle.ANALYSIS_NAME\n"
        "        return inner\n",
        encoding="utf-8")
    (package / "prose_only.py").write_text(
        '"""A module that only talks about analysis.npz."""\n'
        "# and a comment naming ANALYSIS_NAME too\n"
        "def describe():\n"
        '    """Returns nothing; mentions analysis.npz in prose."""\n'
        "    return None\n",
        encoding="utf-8")

    assert _analysis_roster_consumer_offenders(package) == [
        "gpuwm/nested_scope.py:read",
        "gpuwm/nested_scope.py:inner",
    ]


# Re-verification #6's eight probes, verbatim in shape: every position a
# ``def`` or ``class`` statement carries OUTSIDE its body.  All eight were
# missed by the body-only walk, and the plainest of them --
# ``def read(leg, name="analysis.npz")`` -- is a complete, working,
# out-of-contract roster consumer that needs no import at all.
#
# The expected label is the scope Python evaluates the position in, which
# is the scope the definition is written in and not the definition itself:
# a decorator on a nested function is the enclosing function's code.
_RV6_04_PROBES = {
    "p1_decorator.py": (
        "import gpuwm.ensemble.cycle as cycle\n"
        "def register(name):\n"
        "    return lambda fn: fn\n"
        "@register(cycle.ANALYSIS_NAME)\n"
        "def read(leg):\n"
        "    return leg\n",
        ["<module>"],
    ),
    "p2_default_arg.py": (
        "import gpuwm.ensemble.cycle as cycle\n"
        "def read(leg, name=cycle.ANALYSIS_NAME):\n"
        "    return leg / name\n",
        ["<module>"],
    ),
    "p2b_default_arg_literal.py": (
        "def read(leg, name=\"analysis.npz\"):\n"
        "    return leg / name\n",
        ["<module>"],
    ),
    "p2c_kwonly_default.py": (
        "import gpuwm.ensemble.cycle as cycle\n"
        "def read(leg, *, name=cycle.ANALYSIS_NAME):\n"
        "    return leg / name\n",
        ["<module>"],
    ),
    "p3_annotation.py": (
        "from typing import Literal\n"
        "def read(leg: Literal[\"analysis.npz\"]) -> Literal[\"analysis.npz\"]:\n"
        "    return leg\n",
        ["<module>"],
    ),
    "p4_class_base.py": (
        "import gpuwm.ensemble.cycle as cycle\n"
        "def base(name):\n"
        "    return object\n"
        "def meta(name):\n"
        "    return type\n"
        "class Reader(base(cycle.ANALYSIS_NAME)):\n"
        "    pass\n"
        "class Keyworded(metaclass=meta(cycle.ANALYSIS_NAME)):\n"
        "    pass\n",
        ["<module>"],
    ),
    "p5_class_decorator.py": (
        "import gpuwm.ensemble.cycle as cycle\n"
        "def register(name):\n"
        "    return lambda cls: cls\n"
        "@register(cycle.ANALYSIS_NAME)\n"
        "class Reader:\n"
        "    pass\n",
        ["<module>"],
    ),
    "p6_nested_fn_decorator.py": (
        "import gpuwm.ensemble.cycle as cycle\n"
        "def register(name):\n"
        "    return lambda fn: fn\n"
        "def outer():\n"
        "    @register(cycle.ANALYSIS_NAME)\n"
        "    def inner(leg):\n"
        "        return leg\n"
        "    return inner\n",
        ["outer"],
    ),
}


def test_consumer_guard_reaches_every_position_a_definition_carries(tmp_path):
    """RV6-04: the walk reached scope BODIES and nothing else.

    ``own_statements(node)`` returned ``node.body`` and the mention walk
    did ``continue`` on any node that is a scope, so a definition's
    decorators, defaults, keyword-only defaults, annotations, bases and
    class keywords were reached by neither the enclosing scope nor the
    definition itself.  Eight of eight probed positions were invisible.
    """

    package = _synthetic_scan_root(tmp_path)
    expected = []
    for name, (source, labels) in _RV6_04_PROBES.items():
        (package / name).write_text(source, encoding="utf-8")
        expected.extend(f"gpuwm/{name}:{label}" for label in labels)
    assert _analysis_roster_consumer_offenders(package) == expected

    # And one at a time, so a shape that only fires beside another is not
    # mistaken for one that fires.
    for name, (source, labels) in _RV6_04_PROBES.items():
        alone = _synthetic_scan_root(tmp_path / name.replace(".", "_"))
        (alone / name).write_text(source, encoding="utf-8")
        assert _analysis_roster_consumer_offenders(alone) == [
            f"gpuwm/{name}:{label}" for label in labels], name


def test_consumer_guard_attributes_every_node_to_exactly_one_scope():
    """The class behind RV6-04, stated as the property rather than probed.

    Eight fixtures close eight shapes; this closes the reason there were
    eight.  Every node the module's syntax tree carries -- bar the module
    node itself and the docstrings deliberately dropped -- belongs to
    exactly one scope: none is walked twice, and none is walked by nobody.
    A position no scope owns is a position a consumer can be written in
    and not be seen, whatever it happens to be called this round.

    Checked against the file the contract lives in, this test file (which
    is a large sample of real decorators, defaults and annotations), and
    all eight probes.
    """

    import pathlib

    # CPython caches one instance of each context and operator node -- every
    # ``Load`` in a module is the same object -- so identity cannot count
    # them.  None of them carries a name or a constant, so none of them can
    # be a mention; they are excluded from both sides of the comparison
    # rather than from the walk.
    shared = (ast.expr_context, ast.operator, ast.unaryop, ast.boolop,
              ast.cmpop)

    sources = {
        "gpuwm/ensemble/cycle.py":
            pathlib.Path(cycle_module.__file__).read_text(encoding="utf-8"),
        "tests/test_ensemble_hardening.py":
            pathlib.Path(__file__).read_text(encoding="utf-8"),
    }
    sources.update({name: source
                    for name, (source, _) in _RV6_04_PROBES.items()})

    for label, source in sources.items():
        tree = ast.parse(source)
        identities = [id(node)
                      for _scope, nodes in _scope_attribution(tree)
                      for node in nodes
                      if not isinstance(node, shared)]
        assert len(identities) == len(set(identities)), \
            f"{label}: a node is attributed to two scopes"

        dropped = set()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Module,) + _PY_SCOPES):
                continue
            if not node.body:
                continue
            first = node.body[0]
            if isinstance(first, ast.Expr) \
                    and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                dropped.update({id(first), id(first.value)})

        every = {id(node) for node in ast.walk(tree)
                 if not isinstance(node, shared)} - {id(tree)} - dropped
        assert set(identities) == every, (
            f"{label}: {len(every - set(identities))} node(s) belong to no "
            "scope and are walked by nothing")


def test_the_restart_reader_is_the_supported_reader(tmp_path, monkeypatch):
    """The one production consumer, proved to go through the door.

    Not by reading the source -- the test above does that -- but by
    replacing the reader and requiring the cycling driver to have used it.
    """

    cfg = load_ensemble_config(_write_overlay(tmp_path / "gg", n_members=2))
    root = tmp_path / "gg" / "ens"
    names = _contract_names()
    seen = []
    real_reader = cycle_module.read_analysis_roster

    def watched(leg_root, **kwargs):
        seen.append(str(leg_root))
        return real_reader(leg_root, **kwargs)

    monkeypatch.setattr(cycle_module, "read_analysis_roster", watched)
    result = run_cycles(cfg, root, n_cycles=2, cycle_seconds=60.0,
                        assimilate=lambda _i, states: {
                            index: {names[0]: np.full((2, 3), 0.25,
                                                      np.float32)}
                            for index in states},
                        runner=clock_honest_runner)
    assert result.status == "COMPLETE"
    assert seen == [str(cycle_root(root, 0))], (
        "leg 1's restarts must come from the supported reader, once, "
        "against leg 0")


def test_a_completed_publication_leaves_no_marker(tmp_path):
    cfg = load_ensemble_config(_write_overlay(tmp_path / "s", n_members=2))
    root = tmp_path / "s" / "ens"
    names = _contract_names()
    result = run_cycles(cfg, root, n_cycles=1, cycle_seconds=60.0,
                        assimilate=lambda _i, states: {
                            index: {names[0]: np.full((2, 3), 0.25,
                                                      np.float32)}
                            for index in states},
                        runner=clock_honest_runner)
    assert not publication_marker_path(cycle_root(root, 0)).exists()
    receipt = read_manifest(result.manifest_path,
                            schema=CYCLE_MANIFEST_SCHEMA)[
        "cycles"][0]["assimilation"]
    assert receipt["publication_marker"] == PUBLICATION_MARKER_NAME
    assert "three-phase" in receipt["commit"]


# ---------------------------------------------------- F-03 overflow after cast


def _overflow_state():
    names = _contract_names(1)
    state = types.SimpleNamespace()
    setattr(state, names[0], np.zeros((2, 3), dtype=np.float32))
    return state, names[0]


def test_a_finite_increment_that_overflows_the_target_dtype_is_refused():
    """float64 1e300 is finite; in float32 it is inf, and it was written."""

    state, name = _overflow_state()
    before = live_state_sha256(state)
    with pytest.raises(ValueError, match="cast to the float32"):
        apply_increments(state, {name: np.full((2, 3), 1e300, np.float64)})
    assert live_state_sha256(state) == before
    assert np.all(np.isfinite(getattr(state, name)))


def test_an_increment_that_overflows_against_the_background_is_refused():
    """Representable in float32, still inf once added to the background."""

    state, name = _overflow_state()
    getattr(state, name)[...] = np.float32(3.0e38)
    before = live_state_sha256(state)
    with pytest.raises(ValueError, match="produces 6 non-finite"):
        apply_increments(state, {name: np.full((2, 3), 3.0e38, np.float32)})
    assert live_state_sha256(state) == before


def test_the_checkpoint_path_refuses_the_same_overflow(tmp_path):
    names = _contract_names(1)
    background = tmp_path / "gpuwmrst_000.npz"
    np.savez(background,
             **{f"state/{names[0]}": np.zeros((2, 3), np.float32)})
    analysis = tmp_path / ANALYSIS_NAME
    with pytest.raises(ValueError, match="cast to the float32"):
        apply_increments_to_checkpoint(
            background, {names[0]: np.full((2, 3), 1e300, np.float64)},
            analysis)
    assert not analysis.exists()
    assert not analysis.with_name(analysis.name + STAGED_SUFFIX).exists()


# ------------------------------------------------ F-04 resume compatibility


def test_resume_refuses_changed_perturbation_options(tmp_path):
    options = {"field": _contract_names()[0], "amplitude": 0.5}
    overlay = _write_overlay(tmp_path / "f", n_members=2,
                             perturbation="experimental-stub",
                             options=options)
    cfg = load_ensemble_config(overlay)
    root = tmp_path / "f" / "ens"
    prepare_ensemble(cfg, root)
    changed = load_ensemble_config(_write_overlay(
        tmp_path / "f", n_members=2, perturbation="experimental-stub",
        options={**options, "amplitude": 0.9}))
    with pytest.raises(ValueError, match="perturbation_options_sha256"):
        prepare_ensemble(changed, root)


def test_resume_refuses_a_changed_overlay_even_when_the_knobs_match(tmp_path):
    overlay = _write_overlay(tmp_path / "g", n_members=2)
    cfg = load_ensemble_config(overlay)
    root = tmp_path / "g" / "ens"
    prepare_ensemble(cfg, root)
    same_knobs = load_ensemble_config(_write_overlay(
        tmp_path / "g", n_members=2, extra_comment="edited"))
    with pytest.raises(ValueError, match="ensemble overlay sha256"):
        prepare_ensemble(same_knobs, root)


def test_resume_refuses_a_different_forecast_length(tmp_path):
    """The defect: a resumed ensemble mixed 60 s and 120 s members."""

    cfg = load_ensemble_config(_write_overlay(tmp_path / "h", n_members=2))
    root = tmp_path / "h" / "ens"

    def flaky(**kwargs):
        if kwargs["index"] == 1:
            raise RuntimeError("device fell over")
        return clock_honest_runner(**kwargs)

    with pytest.raises(RuntimeError):
        run_ensemble(cfg, root, run_seconds=60.0, runner=flaky)
    with pytest.raises(ValueError, match="run_seconds"):
        run_ensemble(cfg, root, run_seconds=120.0,
                     runner=clock_honest_runner)
    result = run_ensemble(cfg, root, run_seconds=60.0,
                          runner=clock_honest_runner)
    assert result.status == "COMPLETE"
    manifest = read_manifest(result.manifest_path,
                             schema=ENSEMBLE_MANIFEST_SCHEMA)
    assert {record["sim_seconds"] for record in manifest["members"]} == {60.0}


@pytest.mark.parametrize("kwargs, needle", [
    ({"n_cycles": 3}, "n_cycles"),
    ({"cycle_seconds": 30.0}, "cycle_seconds"),
    ({"positivity": "none"}, "positivity"),
    ({"restart_from_analysis": False}, "restart_from_analysis"),
])
def test_cycle_resume_refuses_a_reinterpreted_timeline(tmp_path, kwargs,
                                                       needle):
    cfg = load_ensemble_config(_write_overlay(tmp_path / "i", n_members=1))
    root = tmp_path / "i" / "ens"
    base = dict(n_cycles=2, cycle_seconds=60.0, positivity="clip",
                restart_from_analysis=True, runner=clock_honest_runner)
    run_cycles(cfg, root, **base)
    with pytest.raises(ValueError, match=needle):
        run_cycles(cfg, root, **{**base, **kwargs})


# ------------------------------------------------------ F-08 state inventory


def test_the_state_sha_reports_the_inventory_it_covered(tmp_path):
    """A hash over one array must be legible AS a hash over one array."""

    names = _contract_names(1)
    partial = tmp_path / "gpuwmrst_partial.npz"
    np.savez(partial, **{f"state/{names[0]}": np.zeros((2, 3), np.float32)})
    receipt = checkpoint_state_sha_receipt(partial)
    assert len(receipt["sha256"]) == 64
    assert receipt["inventory"]["complete"] is False
    assert receipt["inventory"]["present"] == [names[0]]
    assert len(receipt["inventory"]["missing"]) \
        == len(serialized_state_attrs()) - 1
    with pytest.raises(ValueError, match="truncated inventory"):
        checkpoint_state_sha_receipt(partial, require_complete=True)


def test_a_whole_state_reports_a_complete_inventory():
    state = types.SimpleNamespace()
    for name in serialized_state_attrs():
        setattr(state, name, np.zeros((2, 2), np.float32))
    receipt = live_state_sha_receipt(state, require_complete=True)
    assert receipt["inventory"]["complete"] is True
    assert receipt["inventory"]["missing"] == []
    assert "STATE_SERIALIZED_ATTRS" in receipt["inventory"]["scope"]


def test_the_manifest_records_each_member_state_inventory(tmp_path):
    cfg = load_ensemble_config(_write_overlay(tmp_path / "j", n_members=1))
    result = run_ensemble(cfg, tmp_path / "j" / "ens", run_seconds=60.0,
                          runner=clock_honest_runner)
    manifest = read_manifest(result.manifest_path,
                             schema=ENSEMBLE_MANIFEST_SCHEMA)
    assert "final_state_inventory" in manifest["members"][0]


# ------------------------------------------------------- F-09 method receipt


def test_the_receipt_names_the_method_the_callable_declared(tmp_path):
    cfg = load_ensemble_config(_write_overlay(tmp_path / "k", n_members=1))
    names = _contract_names()

    def assimilate(cycle_index, member_states):
        increments = {index: {names[0]: np.full((2, 3), 0.25, np.float32)}
                      for index in member_states}
        return increments, {"method": "a named filter", "version": "1.2"}

    result = run_cycles(cfg, tmp_path / "k" / "ens", n_cycles=1,
                        cycle_seconds=60.0, assimilate=assimilate,
                        runner=clock_honest_runner)
    manifest = read_manifest(result.manifest_path,
                             schema=CYCLE_MANIFEST_SCHEMA)
    method = manifest["cycles"][0]["assimilation"]["method"]
    assert method["declared_by"] == "assimilation-callable"
    assert method["provenance"]["method"] == "a named filter"
    assert method["callable"].endswith("assimilate")


def test_the_caller_can_declare_the_method_when_the_callable_does_not(
        tmp_path):
    cfg = load_ensemble_config(_write_overlay(tmp_path / "l", n_members=1))
    names = _contract_names()
    result = run_cycles(
        cfg, tmp_path / "l" / "ens", n_cycles=1, cycle_seconds=60.0,
        assimilate=lambda _i, states: {
            index: {names[0]: np.zeros((2, 3), np.float32)}
            for index in states},
        assimilation_method={"resolved_from": "some.module:analyse"},
        runner=clock_honest_runner)
    method = read_manifest(result.manifest_path,
                           schema=CYCLE_MANIFEST_SCHEMA)[
        "cycles"][0]["assimilation"]["method"]
    assert method["declared_by"] == "caller"
    assert method["provenance"]["resolved_from"] == "some.module:analyse"


# --------------------------------------------------------- F-13 seed encoding


def test_the_manifest_carries_an_interoperable_seed_spelling(tmp_path):
    cfg = load_ensemble_config(_write_overlay(tmp_path / "m", n_members=4,
                                              base_seed=20260730))
    manifest = read_manifest(prepare_ensemble(cfg, tmp_path / "m" / "ens"),
                             schema=ENSEMBLE_MANIFEST_SCHEMA)
    assert "2^53" in manifest["seed_encoding"]
    for index, record in enumerate(manifest["members"]):
        seed = member_seed(20260730, index)
        assert record["seed"] == seed
        assert record["seed_hex"] == seed_hex(seed)
        assert int(record["seed_hex"], 16) == seed
    # The finding's own evidence: these seeds do not survive binary64, so
    # the hex spelling is not decoration.
    assert any(int(float(record["seed"])) != record["seed"]
               for record in manifest["members"])


# ------------------------------------------------------- F-14 durable publish


def test_the_manifest_write_leaves_no_fixed_tmp_name_behind(tmp_path):
    cfg = load_ensemble_config(_write_overlay(tmp_path / "n", n_members=1))
    root = tmp_path / "n" / "ens"
    prepare_ensemble(cfg, root)
    assert (root / ENSEMBLE_MANIFEST_NAME).is_file()
    assert not list(root.glob("*.tmp")), \
        "a fixed .tmp name is a collision between two writers"
