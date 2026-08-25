"""Bare ``gpuwm domain`` at a terminal, and the same command in a script.

The owner typed ``gpuwm domain``, got argparse's usage dump, and asked
"so we don't have an easier way of doing it?".  There is one now, and
these tests hold the two halves of it apart:

* on a terminal, four questions and a run;
* anywhere else -- a pipe, a redirect, CI -- byte-for-byte today's
  usage error and exit 2, because a prompt nobody can see is a hang,
  and a hang is a worse answer than an error.

The load-bearing test is the equivalence gate: a session that answers X
must produce the same TOML as the flag invocation spelling X, because
the prompts are supposed to collect values and nothing else.
"""

from __future__ import annotations

import argparse
import io
import re
from pathlib import Path

import pytest

from gpuwm import domain_interactive as interactive
from gpuwm.cli import main as cli_main


class _Tty(io.StringIO):
    """A stream that claims to be a terminal."""

    def isatty(self) -> bool:
        return True


class _Pipe(io.StringIO):
    """A stream that does not."""

    def isatty(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# The gate: who gets prompts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stdin_tty, stdout_tty, expected", [
    (True, True, True),      # a person at a prompt
    (True, False, False),    # output piped to a file or a log
    (False, True, False),    # input redirected from a file
    (False, False, False),   # CI
])
def test_prompts_require_both_streams_to_be_terminals(
        stdin_tty, stdout_tty, expected):
    """Tested at every value, because three of the four must NOT prompt.

    stdin alone is not enough.  A run whose stdout is a pipe is a run
    nobody is watching, and asking a question into that pipe blocks the
    job on something no one will ever read.
    """

    stdin = _Tty() if stdin_tty else _Pipe()
    stdout = _Tty() if stdout_tty else _Pipe()
    assert interactive.is_interactive(
        ["domain"], stdin=stdin, stdout=stdout) is expected


@pytest.mark.parametrize("argv", [
    ["domain", "--explain"],
    ["domain", "--point=35.3,-97.5"],
    ["domain", "--help"],
    ["doctor"],
    ["setup"],
    [],
])
def test_any_flag_or_other_command_is_never_interactive(argv):
    """A caller who started stating what they want is not interrupted."""

    assert interactive.is_interactive(
        argv, stdin=_Tty(), stdout=_Tty()) is False


def test_a_stream_that_cannot_answer_isatty_is_not_a_terminal():
    """Fail toward the path that cannot hang."""

    class Closed:
        def isatty(self):
            raise ValueError("I/O operation on closed file")

    assert interactive.is_interactive(
        ["domain"], stdin=Closed(), stdout=_Tty()) is False
    assert interactive.is_interactive(
        ["domain"], stdin=object(), stdout=_Tty()) is False


def test_bare_domain_off_a_terminal_keeps_the_usage_error(capsys,
                                                          monkeypatch):
    """Today's behaviour, to the letter, for every script that has one."""

    monkeypatch.setattr("sys.stdin", _Pipe())
    with pytest.raises(SystemExit) as exit_info:
        cli_main(["domain"])
    assert exit_info.value.code == 2
    error = capsys.readouterr().err
    assert "the following arguments are required" in error
    assert "--point" in error and "--cycle" in error and "--out" in error


# ---------------------------------------------------------------------------
# The prompts themselves
# ---------------------------------------------------------------------------

def _answers(monkeypatch, replies):
    """Feed ``replies`` to input(); record the prompt labels shown."""

    shown: list[str] = []
    queue = list(replies)

    def fake_input(label=""):
        shown.append(label)
        if not queue:
            raise EOFError
        return queue.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)
    return shown


def _no_gpu(monkeypatch):
    monkeypatch.setattr(interactive, "detected_vram_gib", lambda: None)


BASE_REPLIES = ["35.3,-97.5", "", "2026-07-29T18", "", ""]


def test_every_prompt_shows_its_default_and_enter_accepts_it(monkeypatch):
    """Enter is only a safe key if the value it accepts is on screen."""

    shown = _answers(monkeypatch, BASE_REPLIES)
    _no_gpu(monkeypatch)
    argv = interactive.collect(printer=lambda *a, **k: None)

    labels = " ".join(shown)
    assert "[gfs]" in labels
    assert "[latest]" in labels
    assert "[6]" in labels
    assert str(Path("configs") / "area_35p30n_97p50w.toml") in labels
    # The point has no default -- there is nothing honest to guess.
    assert re.search(r"center point, lat,lon: $", shown[0])

    # --ladder and --physics-profile are not prompted for: they are the
    # two answers the session supplies so that what it emits is a file
    # `gpuwm go` runs end to end.  They are in the argv, and the session
    # says both out loud (asserted below in
    # test_the_session_states_the_two_defaults_it_supplied).
    assert argv == ["domain", "--point=35.3,-97.5", "--source", "gfs",
                    "--cycle", "2026-07-29T18", "--hours", "6",
                    "--ladder", interactive.DEFAULT_LADDER,
                    "--physics-profile",
                    interactive.default_physics_profile("gfs"),
                    "--out", str(Path("configs") /
                                 "area_35p30n_97p50w.toml")]


@pytest.mark.parametrize("bad, good, complaint", [
    ("not a point", "35.3,-97.5", "decimal degrees"),
    ("95,-97.5", "35.3,-97.5", "[-90, 90]"),
    ("90,-97.5", "35.3,-97.5", "pole itself"),
    # -200 now wraps with a warning (warn-not-block); -400 is beyond
    # the wrappable window and still re-prompts.
    ("35.3,-400", "35.3,-97.5", "[-180, 180]"),
])
def test_a_bad_point_re_prompts_and_says_why(monkeypatch, capsys, bad,
                                             good, complaint):
    """A typo in one answer must not discard the others.

    Four values at four values of the failure: the wizard's own point
    validator is reused, so the message a reader sees at the prompt is
    the message the flag path would have printed.
    """

    _answers(monkeypatch, [bad, good, "", "2026-07-29T18", "", ""])
    _no_gpu(monkeypatch)
    argv = interactive.collect(printer=lambda *a, **k: None)
    assert complaint in capsys.readouterr().out
    assert f"--point={good}" in argv


@pytest.mark.parametrize("bad", ["GFS2", "era-5", "not-a-model"])
def test_an_unknown_source_re_prompts(monkeypatch, capsys, bad):
    _answers(monkeypatch, ["35.3,-97.5", bad, "hrrr", "latest", "", ""])
    _no_gpu(monkeypatch)
    argv = interactive.collect(printer=lambda *a, **k: None)
    assert "is not a registered source" in capsys.readouterr().out
    assert argv[argv.index("--source") + 1] == "hrrr"


@pytest.mark.parametrize("typed", ["gefs", "rap", "gem"])
def test_a_registered_source_the_prompt_does_not_offer_is_accepted(
        monkeypatch, typed):
    """The prompt SUGGESTS three; it must not REFUSE the other thirteen.

    ``gefs`` was one of this suite's own examples of an unknown source,
    which is the measurement: the prompt's shortlist was also its accepted
    set, so every source shipped after those three was unreachable here.
    """

    from gpuwm.source_adapters import get_source_adapter

    _answers(monkeypatch, ["35.3,-97.5", typed, "2026-08-17T00", "", ""])
    _no_gpu(monkeypatch)
    argv = interactive.collect(printer=lambda *a, **k: None)
    assert (argv[argv.index("--source") + 1]
            == get_source_adapter(typed).source_id)


def test_an_empty_answer_takes_the_default_rather_than_complaining(
        monkeypatch, capsys):
    """Enter is the whole point of showing a default; it is not an error."""

    _answers(monkeypatch, BASE_REPLIES)
    _no_gpu(monkeypatch)
    argv = interactive.collect(printer=lambda *a, **k: None)
    assert "must be one of" not in capsys.readouterr().out
    assert argv[argv.index("--source") + 1] == "gfs"
    assert argv[argv.index("--hours") + 1] == "6"


def test_source_answers_are_case_folded(monkeypatch):
    """`GFS` is the same choice as `gfs`; the flag value is canonical."""

    _answers(monkeypatch,
             ["35.3,-97.5", "HRRR", "2026-07-29T18", "", ""])
    _no_gpu(monkeypatch)
    argv = interactive.collect(printer=lambda *a, **k: None)
    assert argv[argv.index("--source") + 1] == "hrrr"


@pytest.mark.parametrize("bad, complaint", [
    ("0", "at least 1"),
    ("-3", "at least 1"),
    ("six", "whole number"),
    ("6.5", "whole number"),
])
def test_bad_forecast_hours_re_prompt(monkeypatch, capsys, bad, complaint):
    _answers(monkeypatch,
             ["35.3,-97.5", "", "2026-07-29T18", bad, "12", ""])
    _no_gpu(monkeypatch)
    argv = interactive.collect(printer=lambda *a, **k: None)
    assert complaint in capsys.readouterr().out
    assert argv[argv.index("--hours") + 1] == "12"


def test_a_reanalysis_is_offered_latest_and_the_prompt_accepts_it(
        monkeypatch):
    """ERA5's `latest` is a time, so this door offers it like any other.

    It used to be refused here by name -- "a reanalysis with weeks of
    latency" -- while the flags door carried a second copy of the same
    rule and a third rule about sources with no fetch route.  Three
    hand-written gates for one question the resolver answers.
    """

    _answers(monkeypatch, ["35.3,-97.5", "era5", "latest", "", ""])
    monkeypatch.setattr("gpuwm.fetch.cds_credentials_present", lambda: True)
    _no_gpu(monkeypatch)
    argv = interactive.collect(printer=lambda *a, **k: None)

    assert argv[argv.index("--cycle") + 1] == "latest"
    assert interactive.default_cycle_answer("era5") == "latest"


def test_the_cycle_prompt_never_offers_a_latest_that_cannot_resolve(
        monkeypatch):
    """The default and the validator are the SAME predicate.

    Offering `latest` for a source nothing can resolve one for defaults
    the reader straight into a refusal two prompts later, which throws
    away answers they had already given.  This is the invariant that
    replaces the two hand-written rules.
    """

    from gpuwm.source_cycles import cycle_grid_for

    validator = interactive._cycle_validator("nam")
    if cycle_grid_for("nam") is None:
        assert interactive.default_cycle_answer("nam") is None
        with pytest.raises(ValueError) as refusal:
            validator("latest")
        assert "declares when that source initializes" in str(refusal.value)


@pytest.mark.parametrize("present", [True, False])
def test_choosing_era5_names_the_cds_key_only_when_it_is_absent(
        monkeypatch, tmp_path, present):
    """Same rule as the NEXT block: a pointer, and only when it is true.

    The pointer is the registry row's credential column now, so the
    fixture moves HOME rather than stubbing a presence function: what
    is measured is the file the declared location names.
    """

    from gpuwm import fetch

    home = tmp_path / "home"
    home.mkdir()
    if present:
        (home / fetch.CDSAPIRC_NAME).write_text("url: x\nkey: y\n")
    monkeypatch.setattr(fetch.Path, "home", staticmethod(lambda: home))

    lines: list[str] = []
    _answers(monkeypatch, ["35.3,-97.5", "era5", "2026-07-29T18", "", ""])
    _no_gpu(monkeypatch)
    interactive.collect(printer=lambda *a, **k: lines.append(" ".join(
        str(x) for x in a)))
    named = any("Copernicus CDS key" in line for line in lines)
    assert named is (not present)


def test_a_closed_stdin_aborts_instead_of_looping(monkeypatch):
    """EOF means nobody is there; looping on it would spin forever."""

    _answers(monkeypatch, [])
    _no_gpu(monkeypatch)
    with pytest.raises(interactive.PromptAborted):
        interactive.collect(printer=lambda *a, **k: None)


def test_an_abort_writes_nothing_and_exits_two(monkeypatch, capsys):
    monkeypatch.setattr(interactive, "is_interactive",
                        lambda argv, **k: argv == ["domain"])
    monkeypatch.setattr(
        interactive, "collect",
        lambda **k: (_ for _ in ()).throw(interactive.PromptAborted("eof")))
    assert cli_main(["domain"]) == 2
    assert "nothing was written" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Card autodetect
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("detected, expect_flag", [
    (31.84, True),
    (None, False),
])
def test_the_card_is_detected_when_it_can_be_and_stated_when_it_cannot(
        monkeypatch, detected, expect_flag):
    """Both arms print which number is about to size the domain.

    The reader was not asked, so they have to be told -- otherwise the
    first they learn of the budget is the sizing line, and on the
    no-GPU arm that number is an assumption rather than a measurement.
    """

    lines: list[str] = []
    _answers(monkeypatch, BASE_REPLIES)
    monkeypatch.setattr(interactive, "detected_vram_gib", lambda: detected)
    argv = interactive.collect(
        printer=lambda *a, **k: lines.append(" ".join(str(x) for x in a)))
    said = " ".join(lines)
    if expect_flag:
        assert argv[argv.index("--vram-gib") + 1] == "31.84"
        assert "31.84 GiB detected" in said
    else:
        assert "--vram-gib" not in argv
        assert "no GPU readable" in said


def test_an_unreadable_card_is_none_rather_than_a_guess(monkeypatch):
    """The probe's own failure mode, not an invented number."""

    monkeypatch.setattr("gpuwm.core.preflight.device_physical_total_bytes",
                        lambda: (_ for _ in ()).throw(OSError("no nvidia-smi")))
    assert interactive.detected_vram_gib() is None
    monkeypatch.setattr("gpuwm.core.preflight.device_physical_total_bytes",
                        lambda: 0)
    assert interactive.detected_vram_gib() is None
    # 32 GiB card as NVML reports it, in bytes.
    monkeypatch.setattr("gpuwm.core.preflight.device_physical_total_bytes",
                        lambda: 34179497472)
    assert interactive.detected_vram_gib() == pytest.approx(31.83, abs=0.02)


# ---------------------------------------------------------------------------
# THE equivalence gate
# ---------------------------------------------------------------------------

def _run_flags(tmp_path, argv_tail):
    out = tmp_path / "flags.toml"
    rc = cli_main(["domain", *argv_tail, "--out", str(out)])
    return rc, out


def test_an_interactive_session_emits_the_same_file_as_the_flags(
        tmp_path, monkeypatch, capsys):
    """The gate the whole design exists to satisfy.

    The prompts collect values and hand back an argv; the same parser
    and the same ``domain_main`` do everything after that.  If these two
    files ever differ, a second config-building path has appeared.

    Compared byte for byte with the provenance line excluded, because
    that line is the ONE thing the two doors are supposed to disagree
    about -- and it is asserted on directly below.
    """

    session_out = tmp_path / "session.toml"
    _answers(monkeypatch, ["35.3,-97.5", "gfs", "2026-07-29T18", "6",
                           str(session_out)])
    monkeypatch.setattr(interactive, "detected_vram_gib", lambda: 24.0)
    monkeypatch.setattr(interactive, "is_interactive",
                        lambda argv, **k: argv == ["domain"])
    assert cli_main(["domain"]) == 0
    capsys.readouterr()

    flags_out = tmp_path / "flags.toml"
    # The same flags the session assembled, including the two it
    # supplies.  Comparing against a DIFFERENT flag set would not test
    # the equivalence this file exists for -- it would test that two
    # different requests produce two different files, which they should.
    assert cli_main([
        "domain", "--point=35.3,-97.5", "--source", "gfs",
        "--cycle", "2026-07-29T18", "--hours", "6",
        "--ladder", interactive.DEFAULT_LADDER,
        "--physics-profile", interactive.default_physics_profile("gfs"),
        "--vram-gib", "24.00", "--out", str(flags_out)]) == 0
    capsys.readouterr()

    session = session_out.read_bytes()
    flags = flags_out.read_bytes()
    assert session[:3] != b"\xef\xbb\xbf" and flags[:3] != b"\xef\xbb\xbf"

    def without_provenance(raw: bytes) -> list[bytes]:
        return [line for line in raw.split(b"\n")
                if not line.startswith(b"# Emitted by")]

    assert without_provenance(session) == without_provenance(flags)

    # ...and the provenance line is the difference, on purpose.
    assert b"(interactive session)" in session.split(b"\n")[0]
    assert b"(interactive session)" not in flags.split(b"\n")[0]
    assert flags.split(b"\n")[0].startswith(b"# Emitted by `gpuwm domain`")


def test_the_session_prints_the_command_it_assembled(tmp_path, monkeypatch,
                                                     capsys):
    """The short path teaches the long one: next time, paste this."""

    out = tmp_path / "taught.toml"
    _answers(monkeypatch,
             ["35.3,-97.5", "gfs", "2026-07-29T18", "6", str(out)])
    monkeypatch.setattr(interactive, "detected_vram_gib", lambda: None)
    monkeypatch.setattr(interactive, "is_interactive",
                        lambda argv, **k: argv == ["domain"])
    assert cli_main(["domain"]) == 0
    printed = capsys.readouterr().out
    assert "gpuwm domain --point=35.3,-97.5 --source gfs" in printed
    assert "--cycle 2026-07-29T18 --hours 6" in printed


def test_the_printed_command_round_trips_through_the_parser(monkeypatch):
    """What it prints must be what it ran -- and be re-runnable."""

    import shlex

    from gpuwm.cli import build_parser

    _answers(monkeypatch, BASE_REPLIES)
    monkeypatch.setattr(interactive, "detected_vram_gib", lambda: 24.0)
    argv = interactive.collect(printer=lambda *a, **k: None)

    printed = interactive.printable_command(argv)
    assert printed.startswith("gpuwm domain ")
    reparsed = shlex.split(printed)[1:]
    assert reparsed == argv
    # And the real parser accepts it.
    args = build_parser().parse_args(reparsed)
    assert args.command == "domain"
    assert args.source == "gfs"
    assert args.hours == 6
    assert args.vram_gib == pytest.approx(24.0)


# ---------------------------------------------------------------------------
# The seam: what a bare session emits has to be something `gpuwm go` runs
# ---------------------------------------------------------------------------

def test_the_session_states_the_defaults_it_supplied(monkeypatch):
    """A default that changes what runs is announced, not discovered.

    Converted from ``..._the_two_defaults_...``: there are three now.
    The forecast lead a run starts from is a default of exactly that
    kind -- a session that never mentions it is a session in which a
    user cannot reach a window deep in a forecast without integrating
    to it -- so it is named where the other two are named.  Both
    original defaults keep every assertion they had.
    """

    _answers(monkeypatch, BASE_REPLIES)
    _no_gpu(monkeypatch)
    said: list[str] = []
    interactive.collect(printer=lambda *parts: said.append(" ".join(
        str(part) for part in parts)))
    text = "\n".join(said)
    assert f"ladder: {interactive.DEFAULT_LADDER} km" in text
    assert interactive.default_physics_profile("gfs") in text
    assert "--physics-profile" in text
    assert "--ladder" in text
    assert "f000 analysis" in text
    assert "--forecast-start-hour" in text


@pytest.mark.parametrize("source", sorted(interactive.SOURCES))
def test_every_default_profile_is_offered_and_model_validated(source):
    """The table is quoting the registry; this re-derives what it quotes.

    Three facts, from the generated registry and the shipped switch
    tables rather than from a second copy of the answer: the profile is
    one the source's own runner route declares, its route ADMITS the
    physics it selects, and no stronger-maturity profile satisfies both.

    The middle fact is new in 1.8 and it is the one whose absence let a
    bug ship.  This test asserted registry reachability and
    ``wrf-matched-run`` maturity, both of which were true of
    ``morrison-mp10-...`` on hrrr -- while ``validate_route_physics``
    refused that emission on ``cu_physics=1``, because HRRR's 3 km grid
    is convection permitting.  A bare hrrr session emitted a config its
    own route would not take, and this test watched it happen.  Maturity
    is now derived as the strongest that survives admissibility rather
    than pinned to the ladder's top rung, so a route whose physics
    contract excludes every ``wrf-matched-run`` template gets the
    strongest suite it can actually run instead of a green test and a
    broken emission.
    """

    from gpuwm.physics_compat import single_domain_runtime_switches
    from gpuwm.physics_registry import physics_registry

    profile = interactive.default_physics_profile(source)
    payload = physics_registry()
    offering = [
        route_id for route_id, route in payload["runner_routes"].items()
        if profile in route.get("source_template_ids", {}).get(source, ())]
    assert offering, (
        f"no runner route declares {profile} for {source}; the "
        "interactive default would emit a config its own runner refuses")

    def admissible(candidate):
        """Does this source's route accept the suite, switch for switch?"""
        if source != "hrrr":
            return True
        from gpuwm.hrrr_route_inputs import (ADMITTED_PBL_PHYSICS,
                                             ADMITTED_RADIATION_PAIRS,
                                             REQUIRED_PHYSICS,
                                             SUPPORTED_MICROPHYSICS)
        try:
            switches = single_domain_runtime_switches(candidate)
        except ValueError:
            return False
        return (
            all(int(switches[key]) == value
                for key, value in REQUIRED_PHYSICS.items())
            and int(switches["bl_pbl_physics"]) in ADMITTED_PBL_PHYSICS
            and (int(switches["ra_lw_physics"]),
                 int(switches["ra_sw_physics"])) in ADMITTED_RADIATION_PAIRS
            and int(switches["mp_physics"]) in SUPPORTED_MICROPHYSICS)

    assert admissible(profile), (
        f"{source} default {profile} is declared by its route but its "
        "physics gate refuses the suite; this is exactly the emission "
        "the registry check alone cannot see")

    ranks = {name: index for index, name in enumerate(
        payload["maturity_ladder"]["order"])}
    reachable = {
        candidate
        for route in payload["runner_routes"].values()
        for candidate in route.get("source_template_ids", {}).get(source, ())
        if admissible(candidate)
    }
    best = max(ranks[payload["templates"][candidate]["maturity"]]
               for candidate in reachable)
    assert ranks[payload["templates"][profile]["maturity"]] == best, (
        f"{source} default {profile} is not the strongest ADMISSIBLE "
        f"maturity available; stronger admissible suites exist")


def test_a_bare_session_emits_a_config_gpuwm_go_accepts(tmp_path,
                                                        monkeypatch,
                                                        capsys):
    """The seam this wave closed, checked against the real emission.

    Not a mock and not a re-implementation of the wizard's output: the
    prompt session runs, ``domain_main`` writes an actual TOML, and
    ``gpuwm go``'s own plan reader is handed that file.  Before this
    change it refused twice over -- the auto ladder emitted four
    domains, and the default suite matched no shipped profile -- so the
    two new front doors did not compose.
    """

    from gpuwm.go_cli import plan_from_config

    out = tmp_path / "bare.toml"
    _answers(monkeypatch, ["35.3,-97.5", "gfs", "2026-07-29T18", "6",
                           str(out)])
    monkeypatch.setattr(interactive, "detected_vram_gib", lambda: 12.0)
    monkeypatch.setattr(interactive, "is_interactive",
                        lambda argv, **k: argv == ["domain"])
    assert cli_main(["domain"]) == 0
    printed = capsys.readouterr().out

    plan = plan_from_config(out)
    assert plan["source"] == "gfs"
    assert plan["profile"] == interactive.default_physics_profile("gfs")
    assert plan["hours"] == 6

    # ...and the file says so on the way out, with no `gpuwm run` in
    # sight: that command refuses a GFS config by design.
    assert "gpuwm go " in printed
    assert "gpuwm run " not in printed
