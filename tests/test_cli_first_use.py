"""The first terminal screen leads to real commands without starting work."""
import argparse

import pytest

from gpuwm import cli


def test_empty_invocation_is_a_short_action_guide(capsys, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("opening help started work")
    monkeypatch.setattr(cli, "_dispatch", unexpected)
    monkeypatch.setattr(cli.domain_interactive, "collect", unexpected)
    assert cli.main([]) == 0
    said = capsys.readouterr()
    assert not said.err
    assert len(said.out.splitlines()) <= 30
    assert "usage: gpuwm COMMAND [OPTIONS]" in said.out
    for command in ("gpuwm domain", "gpuwm go CONFIG", "gpuwm doctor",
                    "gpuwm version", "gpuwm sources", "gpuwm run --wrfinput DIR",
                    "gpuwm run --met-em DIR", "gpuwm --help-all"):
        assert command in said.out


def test_top_help_is_short_and_exhaustive_help_remains_available(capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["--help"])
    assert stop.value.code == 0
    short = capsys.readouterr().out
    assert len(short.splitlines()) <= 30
    with pytest.raises(SystemExit) as stop:
        cli.main(["--help-all"])
    assert stop.value.code == 0
    full = capsys.readouterr().out
    parser = cli.build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    for command in sub.choices:
        assert command in full
    assert len(full) > len(short)


def test_command_help_retains_every_existing_option(capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["go", "--help"])
    assert stop.value.code == 0
    said = capsys.readouterr().out
    for option in ("--prepared-root", "--restart", "--dry-run", "--data-dir"):
        assert option in said


def test_bad_command_still_fails_without_the_full_inventory(capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["not-a-command"])
    assert stop.value.code == 2
    said = capsys.readouterr().err
    assert "invalid choice" in said
    assert "usage: gpuwm COMMAND [OPTIONS]" in said


def test_guided_help_does_not_replace_explicit_domain_settings():
    parsed = cli.build_parser().parse_args([
        "domain", "--point=35.3,-97.5", "--card", "24gb", "--root-dx", "6",
        "--chain", "3,2", "--hours", "48", "--source", "era5",
        "--cycle", "1969-08-17T00", "--out", "configs/run63.toml"])
    assert parsed.root_dx == 6
    assert parsed.chain == "3,2"
    assert parsed.hours == 48
    assert parsed.source == "era5"


@pytest.mark.parametrize("source", ["gfs", "hrrr", "icon-eu", "new-mapped-source"])
def test_automatic_route_ends_with_one_launch_action(source, capsys, monkeypatch):
    from gpuwm import domain_wizard as wizard
    monkeypatch.setattr(wizard, "source_credential_notes", lambda source: [])
    command = 'gpuwm go "configs/my forecast.toml" --data-dir "data/my forecast"'
    wizard._print_next_steps(f"gpuwm fetch --source {source}", "gpuwm check CONFIG",
        command, source=source, deferred=True, explain=False)
    said = capsys.readouterr().out
    assert command + " --dry-run" in said
    next_step = said.split("next:", 1)[1]
    assert command in next_step
    assert "gpuwm fetch" not in said and "gpuwm check" not in said


def test_manual_acquisition_and_detailed_steps_remain_available(capsys, monkeypatch):
    from gpuwm import domain_wizard as wizard
    monkeypatch.setattr(wizard, "source_credential_notes", lambda source: ["Configure your key."])
    for acquisition, explain in (("Download your declared files", False),
                                 ("gpuwm fetch --source gfs", True)):
        wizard._print_next_steps(acquisition, "gpuwm check CONFIG", "gpuwm go CONFIG",
            source="test-source", deferred=True, explain=explain)
        said = capsys.readouterr().out
        assert acquisition in said and "gpuwm check CONFIG" in said
        assert "Configure your key." in said and "gpuwm go CONFIG" in said
