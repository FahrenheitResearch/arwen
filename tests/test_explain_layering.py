"""The one output-layering convention, and the ``gpuwm setup`` wrapper.

Two complaints produced this module.  The CLI printed everything it
knew at once, so the next command was never findable; and a wheel
install still needed a list of staging chores nobody could infer the
order of.  The answers are one flag with one meaning everywhere, and
one command that runs the chores.

What these tests defend is the *contract*, not the wording: that both
halves of a layered message survive inside the exception, that the
default layer never hides a remedy, that every subcommand really does
take the flag the pointers name, and that setup never swallows a
refusal.
"""

from __future__ import annotations

import argparse

import pytest

from gpuwm import doctor, explain, setup_cli
from gpuwm.cli import main as cli_main


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------

def test_layered_keeps_both_halves_in_one_string():
    """Content contract: nothing a layered message says is thrown away.

    The refusals travel as ValueError through call chains this package
    does not own, and tests across the tree read ``str(error)``.
    Keeping both halves inside the one string is what lets the layer be
    chosen at the print boundary without changing what any of those
    readers see.
    """

    message = explain.layered("what happened\n  remedy: do this",
                              "because of the mechanism")
    assert "what happened" in message
    assert "remedy: do this" in message
    assert "because of the mechanism" in message

    action, why = explain.split(message)
    assert action == "what happened\n  remedy: do this"
    assert why == "because of the mechanism"


def test_an_unlayered_message_passes_through_untouched():
    """Most refusals are one sentence; layering must be a no-op on them."""

    assert explain.layered("just this", "") == "just this"
    assert explain.split("just this") == ("just this", "")
    for flag in (False, True):
        assert explain.render("just this", explain=flag,
                              command="gpuwm x") == "just this"


def test_the_default_layer_keeps_the_remedy_and_names_the_flag():
    """The rule the whole convention rests on: a remedy never moves."""

    message = explain.layered("refused\n  remedy: pass --force",
                              "the mechanism paragraph")
    terse = explain.render(message, explain=False, command="gpuwm fetch")
    assert "remedy: pass --force" in terse
    assert "the mechanism paragraph" not in terse
    assert "gpuwm fetch --explain" in terse

    full = explain.render(message, explain=True, command="gpuwm fetch")
    assert "remedy: pass --force" in full
    assert "the mechanism paragraph" in full


def test_the_sentinel_never_reaches_a_terminal_in_either_layer():
    """A marker a reader can see is a bug, whichever layer was asked for."""

    message = explain.layered("action", "why")
    for flag in (False, True):
        rendered = explain.render(message, explain=flag, command="gpuwm x")
        assert explain.EXPLAIN_MARK.strip() not in rendered


def test_the_pointer_is_omitted_rather_than_guessed():
    """No command name, no pointer: a wrong one is worse than none."""

    message = explain.layered("action", "why")
    assert explain.render(message, explain=False, command=None) == "action"


# ---------------------------------------------------------------------------
# The flag really is everywhere the pointers claim
# ---------------------------------------------------------------------------

def _registered_subcommands() -> list[str]:
    """Every subcommand the real parser carries, read off the parser.

    Read rather than transcribed on purpose.  A hand-kept list is a
    sweep that shrinks silently: the day someone registers a
    subcommand and does not think about this file, the transcribed
    version keeps passing while the flag it is supposed to guarantee is
    missing from the new command -- which is exactly when the pointer
    starts lying.
    """

    from gpuwm.cli import build_parser

    for action in build_parser()._actions:  # noqa: SLF001 - argparse only API
        if isinstance(action, argparse._SubParsersAction):
            return sorted(action.choices)
    raise AssertionError("gpuwm's parser registers no subcommands")


def test_the_subcommand_sweep_covers_the_whole_surface():
    """Guard the guard: the enumeration must not come back empty."""

    names = _registered_subcommands()
    assert len(names) >= 15
    # A few anchors, so a parser that silently lost its registrars
    # cannot pass by returning some other non-empty list.
    for anchor in ("doctor", "setup", "domain", "fetch", "run"):
        assert anchor in names


@pytest.mark.parametrize("name", _registered_subcommands())
def test_every_subcommand_accepts_explain(capsys, name):
    """The pointer names ``gpuwm <command> --explain`` for ANY command.

    That sentence is only true if the flag is on all of them, and this
    is what keeps a newly registered subcommand from quietly making it
    false.
    """

    with pytest.raises(SystemExit) as exit_info:
        cli_main([name, "--help"])
    assert exit_info.value.code == 0
    assert "--explain" in capsys.readouterr().out


def test_add_explain_flag_is_idempotent():
    """Two registrars share a parser; a second add must not be a crash."""

    parser = argparse.ArgumentParser()
    explain.add_explain_flag(parser)
    explain.add_explain_flag(parser)
    assert parser.parse_args([]).explain is False
    assert parser.parse_args(["--explain"]).explain is True


# ---------------------------------------------------------------------------
# Doctor: the same estate at two widths
# ---------------------------------------------------------------------------

def _fake(name, status, **kwargs):
    return doctor.Check(name, status, "detail text", **kwargs)


def test_the_terse_report_folds_a_shared_remedy_into_one_line():
    """Six identical remedies read as six problems; they are one."""

    checks = [
        _fake("bridge a", "missing", remedy="r",
              action="gpuwm fetch-bridges", group="bridges"),
        _fake("bridge b", "missing", remedy="r",
              action="gpuwm fetch-bridges", group="bridges"),
        _fake("bridge c", "missing", remedy="r",
              action="gpuwm fetch-bridges", group="bridges"),
    ]
    brief = doctor.format_brief(checks)
    assert brief.count("gpuwm fetch-bridges") == 1
    assert "bridges (3)" in brief
    # The full report still names every one of them.
    full = doctor.format_report(checks)
    for name in ("bridge a", "bridge b", "bridge c"):
        assert name in full


def test_folding_never_merges_across_status_or_remedy():
    """A fold is a presentation of sameness, never an assertion of it."""

    checks = [
        _fake("bridge a", "missing", action="gpuwm fetch-bridges",
              group="bridges"),
        _fake("bridge b", "verified", group="bridges"),
        _fake("bridge c", "missing", action="cargo build", group="bridges"),
    ]
    brief = doctor.format_brief(checks)
    assert "gpuwm fetch-bridges" in brief and "cargo build" in brief
    assert "(3)" not in brief


def test_the_terse_summary_points_at_setup_only_when_it_is_shorter():
    """One gap already printed its one command; a wrapper adds a step."""

    one = doctor.format_brief(
        [_fake("thompson tables", "missing", action="gpuwm fetch-tables")])
    assert "gpuwm setup" not in one

    both = doctor.format_brief([
        _fake("bridge a", "missing", action="gpuwm fetch-bridges"),
        _fake("thompson tables", "missing", action="gpuwm fetch-tables"),
    ])
    assert "gpuwm setup" in both
    assert "gpuwm fetch-bridges" in both and "gpuwm fetch-tables" in both


def test_a_published_bundle_makes_the_fresh_install_summary_name_setup(
        monkeypatch):
    """The moment a PyPI user actually meets, end to end.

    On a release that published a bundle for the reader's platform,
    every Rust gap resolves to one download and the table gap to
    another -- which is exactly the pair ``gpuwm setup`` runs.  This
    pins that the terse report reaches that conclusion rather than
    leaving the reader to notice it, and that the per-line commands are
    still there for anyone who would rather run them separately.
    """

    from gpuwm import bridges

    monkeypatch.setattr(bridges, "prebuilt_bundle_offer",
                        lambda: ("  gpuwm fetch-bridges",))
    monkeypatch.setattr(bridges, "sources_present", lambda *a, **k: False)

    action = doctor._build_action()
    assert action == "gpuwm fetch-bridges"

    brief = doctor.format_brief([
        _fake("bridge grib1_bridge", "missing", action=action,
              group=doctor._GROUP_BRIDGES),
        _fake("bridge gfs_grib2_bridge", "missing", action=action,
              group=doctor._GROUP_BRIDGES),
        _fake("thompson tables", "missing", action="gpuwm fetch-tables"),
        _fake("WPS_GEOG", "missing", action="gpuwm fetch-geog"),
    ])
    assert "gpuwm setup runs gpuwm fetch-bridges then gpuwm fetch-tables" \
        in brief
    # Geog is NOT claimed by setup: it is opt-in, and the summary must
    # not imply the wrapper closes a gap it deliberately leaves open.
    assert "-> gpuwm fetch-geog" in brief
    assert "fetch-geog" not in brief.splitlines()[-2]


def test_every_actionable_gap_carries_a_next_command():
    """A MISSING line with no action is a dead end on the default layer.

    The full report can afford a remedy that is six lines of comments;
    the one-line form cannot, so every gap has to have named the single
    thing to do -- even when that thing is a sentence rather than a
    command, as it is for a path only the reader knows.
    """

    for check in doctor.collect_checks():
        if check.status == "missing":
            assert check.action, check.name
            assert check.action.strip() == check.action


#: One check of every status, with a multi-line remedy, rendered by
#: :func:`gpuwm.doctor.format_report`.  Verified byte-for-byte equal to
#: what v1.2.0's format_report produced for the same input (compared
#: against `git show 39984b0e:gpuwm/doctor.py`), which is what "the
#: --explain layer is preserved verbatim" has to mean if it means
#: anything.
_GOLDEN_FULL_REPORT = """\
gpuwm doctor: runtime estate
  ok      python: 3.13 on this machine
  present manifest: schema only, so presence-only
  info    root: not set
  MISSING bridge x: not staged
          remedy: gpuwm fetch-bridges
                  # one download, verified against the packaged pins
                  # before anything is staged.
gpuwm doctor: 1 gap(s).  Every remedy line above is either a command \
to run as printed, in the order printed, or a '#' comment."""


def test_the_explain_layer_still_renders_exactly_what_it_always_did():
    """The verbatim promise, pinned as output rather than as intent.

    ``--explain`` exists to hand back the long form unchanged.  A
    docstring saying so is not a guarantee; this is.  If a future edit
    reflows the label column, the ten-space remedy gutter, or the
    closing sentence, this fails and the promise gets renegotiated
    deliberately instead of quietly.
    """

    checks = [
        doctor.Check("python", "verified", "3.13 on this machine"),
        doctor.Check("manifest", "present", "schema only, so presence-only"),
        doctor.Check("root", "info", "not set"),
        doctor.Check(
            "bridge x", "missing", "not staged",
            "gpuwm fetch-bridges\n"
            "  # one download, verified against the packaged pins\n"
            "  # before anything is staged.",
            action="gpuwm fetch-bridges", brief="not staged",
            group="bridges"),
    ]
    assert doctor.format_report(checks) == _GOLDEN_FULL_REPORT


def test_the_terse_layer_reports_the_same_findings_as_the_full_one():
    """Two widths, one estate: neither layer may invent or lose a gap."""

    checks = doctor.collect_checks()
    brief = doctor.format_brief(checks)
    full = doctor.format_report(checks)

    gaps = sum(1 for check in checks if check.status == "missing")
    if gaps:
        assert f"{gaps} gap(s)" in brief
        assert f"{gaps} gap(s)" in full
    else:
        assert "no gaps" in brief and "no gaps" in full


def test_the_json_layer_carries_both_widths():
    """A front end reading --json gets the terse fields and the full ones."""

    payload = doctor.collect_checks()[0].__dict__
    for field in ("name", "status", "detail", "remedy", "action", "brief",
                  "group"):
        assert field in payload


# ---------------------------------------------------------------------------
# gpuwm setup
# ---------------------------------------------------------------------------

def test_setup_takes_each_steps_own_defaults_not_a_transcription():
    """The wrapper must not become a second declaration of the defaults."""

    from gpuwm import geog_assets

    geog = setup_cli.step_namespace("gpuwm.geog_assets")
    assert geog.datasets == "all"
    assert geog.bundle is False
    assert geog.root is None
    # The handler travels with the namespace, so there is no name to drift.
    assert geog.func is geog_assets.fetch_geog_main

    bridges_ns = setup_cli.step_namespace("gpuwm.bridge_assets")
    assert bridges_ns.from_dir is None and bridges_ns.list is False


def test_setup_runs_the_steps_in_order_and_prints_one_line_each(
        monkeypatch, capsys):
    calls = []

    def fake(module_name, overrides):
        calls.append(module_name)
        return 0, f"{module_name}: chatty\nsecond line\n"

    monkeypatch.setattr(setup_cli, "_run_step", fake)
    monkeypatch.setattr("gpuwm.doctor.collect_checks", lambda: [])

    rc = setup_cli.setup_main(argparse.Namespace(
        with_geog=False, from_dir=None, explain=False))
    printed = capsys.readouterr().out
    assert rc == 0
    assert calls == ["gpuwm.bridge_assets", "gpuwm.table_assets"]
    assert "  ok      bridges" in printed
    assert "  ok      tables" in printed
    # A succeeding step's chatter stays in its own command's output.
    assert "chatty" not in printed


def test_setup_explain_replays_every_steps_own_output(monkeypatch, capsys):
    monkeypatch.setattr(
        setup_cli, "_run_step",
        lambda module_name, overrides: (0, "the full receipt\n"))
    monkeypatch.setattr("gpuwm.doctor.collect_checks", lambda: [])

    setup_cli.setup_main(argparse.Namespace(
        with_geog=False, from_dir=None, explain=True))
    assert "the full receipt" in capsys.readouterr().out


def test_setup_never_swallows_a_refusal(monkeypatch, capsys):
    """A wrapper that hides why a step refused is worse than no wrapper."""

    def fake(module_name, overrides):
        if module_name.endswith("bridge_assets"):
            return 2, "gpuwm fetch-bridges: REFUSED: no bundle for this\n"
        return 0, ""

    monkeypatch.setattr(setup_cli, "_run_step", fake)
    monkeypatch.setattr("gpuwm.doctor.collect_checks", lambda: [])

    rc = setup_cli.setup_main(argparse.Namespace(
        with_geog=False, from_dir=None, explain=False))
    printed = capsys.readouterr().out
    assert rc == 2
    assert "FAILED  bridges" in printed
    # Verbatim, and without --explain, because this is the reason.
    assert "REFUSED: no bundle for this" in printed
    # And the independent step still ran.
    assert "  ok      tables" in printed


def test_a_step_that_raises_keeps_what_it_had_already_printed(
        monkeypatch, capsys):
    """A download that died most of the way through IS the diagnosis.

    Letting the exception escape _run_step would have discarded the
    step's captured output along with it, leaving the reader a bare
    traceback where the useful part was the four lines above it.
    """

    def exploding(namespace):
        print("gpuwm fetch-tables: downloading freezeH2O.dat (243 MiB)")
        raise OSError("connection reset by peer")

    monkeypatch.setattr(
        setup_cli, "step_namespace",
        lambda module_name: argparse.Namespace(func=exploding))
    monkeypatch.setattr("gpuwm.doctor.collect_checks", lambda: [])

    rc = setup_cli.setup_main(argparse.Namespace(
        with_geog=False, from_dir=None, explain=False))
    printed = capsys.readouterr().out
    assert rc == 2
    assert "downloading freezeH2O.dat" in printed
    assert "OSError: connection reset by peer" in printed
    # The estate report still prints: after a partial setup, "where do
    # I stand" is exactly the question.
    assert "gpuwm doctor" in printed


def test_setup_does_not_fetch_geog_unless_asked_and_prints_the_size(
        monkeypatch, capsys):
    """16 GB is a decision, not a side effect of typing ``setup``."""

    seen = []

    def fake(module_name, overrides):
        seen.append(module_name)
        return 0, ""

    monkeypatch.setattr(setup_cli, "_run_step", fake)
    monkeypatch.setattr("gpuwm.doctor.collect_checks", lambda: [])

    setup_cli.setup_main(argparse.Namespace(
        with_geog=False, from_dir=None, explain=False))
    assert "gpuwm.geog_assets" not in seen
    assert "16 GB" not in capsys.readouterr().out

    seen.clear()
    setup_cli.setup_main(argparse.Namespace(
        with_geog=True, from_dir=None, explain=False))
    printed = capsys.readouterr().out
    assert "gpuwm.geog_assets" in seen
    assert "16 GB" in printed
    # The size is announced BEFORE the first byte moves.
    assert printed.index("16 GB") < printed.index("  ok      bridges")


def test_setup_is_registered_and_dispatches_through_the_real_cli(capsys):
    """cli._dispatch carries a hardcoded name list; this is its gate."""

    with pytest.raises(SystemExit) as exit_info:
        cli_main(["setup", "--help"])
    assert exit_info.value.code == 0
    printed = capsys.readouterr().out
    assert "--with-geog" in printed and "--explain" in printed


# ---------------------------------------------------------------------------
# The physics refusal, at both widths
# ---------------------------------------------------------------------------

def test_a_physics_refusal_keeps_its_rule_and_defers_its_mechanism():
    from gpuwm.physics_compat import require_ready_wrf_physics

    with pytest.raises(ValueError) as error_info:
        require_ready_wrf_physics(
            mp_physics=8, sf_sfclay_physics=5, bl_pbl_physics=1,
            sf_surface_physics=2, num_soil_layers=4)
    message = str(error_info.value)

    terse = explain.render(message, explain=False, command="gpuwm run")
    assert "WRF v4.6.1 PBL/surface-layer compatibility" in terse
    assert "WRF v4.6.1 refuses this pairing" in terse
    assert "no substitutions were applied" in terse
    # The mechanism waits to be asked for.
    assert "phys/module_physics_init.F:" not in terse
    assert "gpuwm run --explain" in terse

    full = explain.render(message, explain=True, command="gpuwm run")
    assert "phys/module_physics_init.F:3213-3219,3699-3701" in full


def test_the_noahmp_budget_remedy_is_never_behind_the_flag():
    """The one blocker whose second element is a remedy, not a reason.

    A reader refused for an unmeasured grid width, and not told about
    the variable that accepts it, has been refused with no way forward.
    """

    from gpuwm.physics_compat import (
        NOAHMP_EXPERT_COLUMN_BUDGET_ENV, pending_wrf_physics_components)

    blockers = pending_wrf_physics_components(
        mp_physics=8, sf_sfclay_physics=1, bl_pbl_physics=1,
        sf_surface_physics=4, num_soil_layers=4, columns=10_000_000)
    budget = [b for b in blockers if b.component == "Noah-MP column budget"]
    assert budget, [b.component for b in blockers]
    assert NOAHMP_EXPERT_COLUMN_BUDGET_ENV in budget[0].action()
    assert NOAHMP_EXPERT_COLUMN_BUDGET_ENV not in budget[0].why()
    # format() is unchanged: it still says everything, on one line.
    assert budget[0].action() in budget[0].format()
    assert budget[0].why() in budget[0].format()
