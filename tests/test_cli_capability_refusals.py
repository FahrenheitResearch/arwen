"""The front door refuses before expensive work, and every remedy is real.

These tests are written to run against an INSTALLED WHEEL as well as a
checkout: nothing here imports a repository fixture, reads a repository
path, or needs a card.  A source checkout has everything on the path and
hides exactly the defects this file is about, so the same file is run
against the wheel in a clean venv as the proof of record.

Every assertion is one of three shapes:

* the refusal FIRES when the capability is missing;
* the refusal is SILENT when it is present (an instrument that always
  says "missing" measures nothing);
* the remedy RESOLVES -- the extra exists and is non-empty, the command
  is a real subcommand, the flag is a real flag.
"""

from __future__ import annotations

import json
import re
import sys
from importlib.metadata import metadata, requires
from pathlib import Path

import pytest

from gpuwm import capabilities


# ---------------------------------------------------------------------------
# The registry itself: derived remedies, and remedies that resolve
# ---------------------------------------------------------------------------


def _declared_extras() -> dict[str, list[str]]:
    """extra -> the requirement lines it installs, from THIS install."""

    found: dict[str, list[str]] = {
        name: [] for name in
        (metadata("gpuwm").get_all("Provides-Extra") or [])}
    for line in requires("gpuwm") or []:
        match = re.search(r'extra\s*==\s*[\'"]([^\'"]+)[\'"]', line)
        if match and match.group(1) in found:
            found[match.group(1)].append(line.split(";")[0].strip())
    return found


def _extras_named(text: str) -> set[str]:
    return set(re.findall(r"gpuwm\[([a-z0-9\-,]+)\]", text))


#: Every message this lane can print, as text, for the remedy sweeps.
def _all_remedy_text() -> str:
    from gpuwm import go_cli, render, runplan

    parts = [item.remedy for item in capabilities.REQUIREMENTS]
    parts += [capabilities.refusal("gpuwm run", item,
                                   before="Refusing here, before anything.")
              for item in capabilities.REQUIREMENTS]
    parts += [value for value in capabilities._BEFORE.values()]
    parts += [value for value in runplan._REMEDIES.values()]
    parts.append(render.engine_refusal("matplotlib", "matplotlib",
                                       "not built") or "")
    parts.append(render.engine_refusal("matplotlib", "auto",
                                       "not built") or "")
    parts.append(go_cli.__doc__ or "")
    return "\n".join(parts)


def test_every_extra_a_remedy_names_exists_and_installs_something():
    """A remedy naming an empty or absent extra is a remedy that fails.

    Both halves matter.  ``gpuwm[geog]`` still RESOLVES in 2.3.3 and is
    now EMPTY, so `pip install 'gpuwm[geog]'` succeeds and installs
    nothing -- a remedy that reports success and fixes the problem not
    at all is worse than one that errors.
    """

    declared = _declared_extras()
    named = set()
    for spelling in _extras_named(_all_remedy_text()):
        named.update(part.strip() for part in spelling.split(","))
    assert named, "the sweep found no extras at all; it is not measuring"
    for extra in sorted(named):
        assert extra in declared, (
            f"a remedy names gpuwm[{extra}], which this distribution does "
            f"not declare; declared: {sorted(declared)}")
        assert declared[extra], (
            f"a remedy names gpuwm[{extra}], which is declared but EMPTY, "
            "so running it would install nothing")


#: ``gpuwm <command>`` where it is being OFFERED, not merely discussed:
#: after ``remedy:``, inside backticks, or on a ``#`` continuation line.
#: A looser sweep matches English ("the same extra as ...") and would
#: fail on prose while missing nothing real.
_OFFERED_COMMAND = re.compile(
    r"(?:remedy:\s*|`|^\s*#\s+(?:or\s+)?)gpuwm ([a-z][a-z-]+)", re.MULTILINE)


def test_every_gpuwm_command_a_remedy_names_is_a_real_subcommand():
    from gpuwm.cli import build_parser

    real = set(build_parser()._subparsers._group_actions[0].choices)
    offered = set(_OFFERED_COMMAND.findall(_all_remedy_text()))
    assert offered, "the sweep found no commands at all; it is not measuring"
    for command in sorted(offered):
        assert command in real, (
            f"a remedy names `gpuwm {command}`, which is not a subcommand; "
            f"real: {sorted(real)}")


def test_every_flag_a_remedy_names_is_a_real_flag():
    from gpuwm.cli import build_parser

    parser = build_parser()
    choices = parser._subparsers._group_actions[0].choices
    known = {"--explain"}
    for sub in choices.values():
        for action in sub._actions:
            known.update(action.option_strings)
    for flag in sorted(set(re.findall(r"(--[a-z][a-z0-9-]+)",
                                      _all_remedy_text()))):
        if flag in ("--upgrade", "--force-reinstall", "--no-deps"):
            continue            # pip's flags, not ours
        assert flag in known, f"a remedy names {flag}, which no parser has"


def test_a_missing_non_cupy_module_does_not_get_the_cupy_remedy():
    """THE defect: a remedy keyed on the exception CLASS.

    Every import failure is a ``ModuleNotFoundError``, so a table keyed
    on that name answered a missing wrf-rust, scipy or pyshp with a GPU
    install line.
    """

    from gpuwm import runplan

    # scipy's expectation is `scipy`, not an extra.  It named gpuwm[obs]
    # while that extra carried scipy; scipy is a base dependency now and
    # [obs] is an empty compatibility alias, so naming it here would
    # assert the exact defect
    # test_every_extra_a_remedy_names_exists_and_installs_something
    # forbids.  What this test is about is unchanged: a remedy keyed on
    # the missing MODULE, never on the exception class.
    for module, expected in (("scipy", "force-reinstall 'scipy"),
                             ("wrf", "gpuwm[render]"),
                             ("shapefile", "gpuwm[render]")):
        error = ModuleNotFoundError(f"No module named '{module}'",
                                    name=module)
        remedy = runplan._remedy(error)
        assert remedy is not None, f"{module} got no remedy at all"
        assert expected in remedy, f"{module} -> {remedy!r}"
        assert "gpu-cu12" not in remedy and "gpu-cu13" not in remedy, (
            f"a missing {module} was answered with the CuPy remedy: "
            f"{remedy!r}")


def test_the_cupy_remedy_is_still_reached_for_cupy():
    """The other direction: the fix must not have removed the real one."""

    from gpuwm import runplan

    remedy = runplan._remedy(
        ModuleNotFoundError("No module named 'cupy'", name="cupy"))
    assert remedy is not None and "gpu-cu12" in remedy


def test_a_relayed_worker_message_still_derives_its_remedy():
    """A worker's failure reaches the parent as TEXT, not as an exception."""

    relayed = RuntimeError(
        "worker exited with status 1: ModuleNotFoundError: No module named "
        "'cupy'; no durable manifest-valid checkpoint is available")
    remedy = capabilities.remedy_for_error(relayed)
    assert remedy is not None and "gpu-cu12" in remedy


def test_an_unknown_module_gets_no_remedy_rather_than_the_nearest_one():
    error = ModuleNotFoundError("No module named 'frobnicate'",
                                name="frobnicate")
    assert capabilities.remedy_for_error(error) is None
    assert capabilities.remedy_for_module("frobnicate") is None


def test_presence_probe_answers_both_directions():
    """An instrument that always says 'missing' measures nothing."""

    assert capabilities.is_installed("json") is True
    assert capabilities.is_installed("numpy") is True
    assert capabilities.is_installed(
        "a_module_that_is_not_installed_anywhere") is False


def test_require_is_silent_when_the_requirement_resolves():
    present = capabilities.Requirement(
        module="json", distribution="python", extras=(),
        unlocks="nothing", remedy="  remedy: none")
    capabilities.require("gpuwm run", present)      # must not raise


def test_require_names_the_door_verbatim():
    absent = capabilities.Requirement(
        module="a_module_that_is_not_installed_anywhere",
        distribution="nothing", extras=(), unlocks="nothing",
        remedy="  remedy: none")
    with pytest.raises(capabilities.CapabilityMissing) as caught:
        capabilities.require("python -m gpuwm.prepared_domain_tree_forecast",
                             absent)
    assert str(caught.value).startswith(
        "python -m gpuwm.prepared_domain_tree_forecast:")
    assert caught.value.requirement is absent


def test_every_command_in_the_table_is_a_real_subcommand():
    from gpuwm.cli import build_parser

    real = set(build_parser()._subparsers._group_actions[0].choices)
    for command in capabilities.COMMAND_REQUIREMENTS:
        assert command in real


# ---------------------------------------------------------------------------
# H21 / H19 -- run-plan
# ---------------------------------------------------------------------------


class _Check:
    def __init__(self, name, status, blocking):
        self.name = name
        self.status = status
        self.blocking = blocking
        self.detail = ""


def test_probe_is_not_ready_when_the_run_front_door_would_refuse(monkeypatch):
    """A green light over a hole.

    doctor carries the CuPy check as NON-blocking on purpose, so an
    install whose only gap is the GPU runtime reported
    ``"ready": true, "blocking_gaps": 0`` -- to a front end whose very
    next call is the run that then refuses.
    """

    from gpuwm import doctor, runplan

    monkeypatch.setattr(doctor, "collect_checks",
                        lambda *a, **k: [_Check("cupy", "missing", False)])
    monkeypatch.setattr(doctor, "blocking_gaps", lambda checks: [])
    monkeypatch.setattr(capabilities, "is_installed",
                        lambda module: module != "cupy")

    document = runplan.probe_environment()
    readiness = document["readiness"]
    assert readiness["blocking_gaps"] == 0
    assert readiness["ready"] is False, (
        "the probe reported ready on an install the run front door "
        "refuses")
    unmet = readiness["unmet_run_requirements"]
    assert [item["module"] for item in unmet] == ["cupy"]
    assert "gpu-cu12" in unmet[0]["remedy"]


def test_probe_is_ready_when_nothing_is_missing(monkeypatch):
    """The other direction: the gate must not be permanently red."""

    from gpuwm import doctor, runplan

    monkeypatch.setattr(doctor, "collect_checks",
                        lambda *a, **k: [_Check("cupy", "ok", False)])
    monkeypatch.setattr(doctor, "blocking_gaps", lambda checks: [])
    monkeypatch.setattr(capabilities, "is_installed", lambda module: True)

    readiness = runplan.probe_environment()["readiness"]
    assert readiness["ready"] is True
    assert readiness["unmet_run_requirements"] == []


def test_probe_reports_unknown_rather_than_ready_when_it_cannot_check(
        monkeypatch):
    from gpuwm import doctor, runplan

    def explode(*a, **k):
        raise RuntimeError("nvml is wedged")

    monkeypatch.setattr(doctor, "collect_checks", explode)
    readiness = runplan.probe_environment()["readiness"]
    assert readiness["collected"] is False
    assert readiness["ready"] is None, "unknown must never read as ready"


def test_probe_without_readiness_reports_unknown_not_absent():
    from gpuwm import runplan

    readiness = runplan.probe_environment(readiness=False)["readiness"]
    assert readiness["collected"] is False
    assert "ready" in readiness and readiness["ready"] is None


# ---------------------------------------------------------------------------
# H9 / H10 / H20 / H22 -- render
# ---------------------------------------------------------------------------


def test_render_refusal_leads_with_the_remedy_for_what_is_broken(monkeypatch):
    """The wrong remedy first is the whole defect.

    A bare install printed ``Build it with: ...`` -- an advisory about
    the engine that was NOT selected -- and then died in a traceback
    whose tail carried the install line that would have worked.
    """

    from gpuwm import render

    monkeypatch.setattr(render, "matplotlib_engine_gap",
                        lambda: capabilities.SCIENCE_CORE)
    text = render.engine_refusal("matplotlib", "auto", "not built")
    assert text is not None
    first = text.splitlines()[0]
    assert "fetch-bridges" not in text.split("[[explain]]")[0]
    assert "neither render engine can draw" in first
    action = text.split("[[explain]]")[0]
    assert "gpuwm setup" in action and "gpuwm[render]" in action


def test_matplotlib_engine_refuses_naming_its_own_extra(monkeypatch):
    """H10: the documented fallback needs the extra it falls back FROM."""

    from gpuwm import render

    monkeypatch.setattr(render, "matplotlib_engine_gap",
                        lambda: capabilities.SCIENCE_CORE)
    text = render.engine_refusal("matplotlib", "matplotlib", "requested")
    assert text is not None
    assert "gpuwm[render]" in text
    assert "wrf-rust" in text


def test_engine_refusal_is_silent_when_the_engine_can_draw(monkeypatch):
    from gpuwm import render

    monkeypatch.setattr(render, "matplotlib_engine_gap", lambda: None)
    assert render.engine_refusal("matplotlib", "auto", "not built") is None
    assert render.engine_refusal("rust", "auto", "ok") is None


def test_pair_remedy_names_a_distribution_the_extra_really_ships():
    """H22: `[render]` is wrf-rust + pyshp and has never held Pillow."""

    declared = _declared_extras()
    render_extra = " ".join(declared.get("render", [])).lower()
    assert "pillow" not in render_extra, (
        "the premise moved: [render] now ships Pillow, so this remedy "
        "should be revisited")
    remedy = capabilities.PILLOW.remedy
    assert "pip install Pillow" in remedy
    assert "gpuwm[render]" not in remedy


def test_the_chain_render_gate_agrees_with_the_render_command(monkeypatch):
    """H20: `go` skipped rendering on installs where render draws 161 PNGs."""

    from gpuwm import go_cli, render

    monkeypatch.setattr(render, "drawable_engine",
                        lambda: ("rust", "C:/staged/rw_wrfbatch.exe"))
    assert go_cli.render_extra_missing() is None

    monkeypatch.setattr(render, "drawable_engine",
                        lambda: (None, "no engine here"))
    assert go_cli.render_extra_missing() == "no engine here"


# ---------------------------------------------------------------------------
# H2 -- the memory gate stops swallowing the reason
# ---------------------------------------------------------------------------


class _Completed:
    def __init__(self, returncode, stdout=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = ""


def test_memory_probe_distinguishes_no_runtime_from_no_card(monkeypatch):
    from gpuwm.core import preflight

    # The never-touch-the-local-device switch short-circuits the probe
    # before the run seam (tests/test_no_local_gpu_contract.py owns that
    # contract); this test is about the exit-code discrimination behind it.
    monkeypatch.delenv("GPUWM_NO_LOCAL_GPU", raising=False)

    reason = preflight.device_memory_probe_reason(
        run=lambda *a, **k: _Completed(preflight.PROBE_EXIT_NO_RUNTIME))
    assert reason is not None and "CuPy" in reason

    reason = preflight.device_memory_probe_reason(
        run=lambda *a, **k: _Completed(3))
    assert reason is not None and "device" in reason

    payload = json.dumps({"free_bytes": 1024, "total_bytes": 2048})
    assert preflight.device_memory_probe_reason(
        run=lambda *a, **k: _Completed(0, payload)) is None


def test_the_probe_source_reports_a_missing_runtime_distinctly(tmp_path):
    """Run the probe's own source in a real interpreter with no cupy.

    The shadow module is put on ``PYTHONPATH`` ahead of site-packages so
    the real CuPy is never imported.  That is not only for determinism:
    importing it would stand a CUDA primary context up on whatever card
    this test runs beside, which is exactly what the probe exists to
    keep out of the parent process.
    """

    import os
    import subprocess

    from gpuwm.core import preflight

    script = tmp_path / "probe.py"
    script.write_text(preflight._DEVICE_MEMORY_PROBE_SOURCE, encoding="utf-8")
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "cupy.py").write_text(
        "raise ImportError('shadowed for this test', name='cupy')",
        encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(shadow)
    env.pop("PYTHONSAFEPATH", None)
    done = subprocess.run([sys.executable, str(script)],
                          capture_output=True, text=True, env=env,
                          cwd=str(tmp_path), timeout=300)
    assert done.returncode == preflight.PROBE_EXIT_NO_RUNTIME, done.stderr
    assert "no-runtime" in done.stderr
    assert done.stdout.strip() == ""


# ---------------------------------------------------------------------------
# H4 -- the prepared runners refuse before they claim anything
# ---------------------------------------------------------------------------


def _prepared_argv(outdir: Path, config: Path) -> list[str]:
    return ["--source", "gfs", "--prepared-root", str(outdir.parent / "prep"),
            "--proof-sha256", "0" * 64, "--source-manifest-sha256", "0" * 64,
            "--prepared-content-sha256", "0" * 64,
            "--experiment-config", str(config),
            "--wps-namelist", str(outdir.parent / "namelist.wps"),
            "--io-mode", "history", "--outdir", str(outdir),
            "--run-seconds", "60"]


def test_prepared_forecast_refuses_before_claiming_the_output_directory(
        monkeypatch, tmp_path, capsys):
    """H4: it used to create the run dir and write progress.json first."""

    from gpuwm import prepared_single_domain_forecast as runner

    monkeypatch.setattr(capabilities, "is_installed",
                        lambda module: module != "cupy")
    config = tmp_path / "config.toml"
    config.write_text("[experiment]\nname = \"x\"\n", encoding="utf-8")
    outdir = tmp_path / "run"

    code = runner.main(_prepared_argv(outdir, config))

    assert code == 2
    assert not outdir.exists(), (
        "the runner claimed its output directory before checking whether "
        "it could run at all")
    text = capsys.readouterr().err
    assert "cupy" in text and "gpu-cu12" in text
    assert "Traceback" not in text
    # No `--explain` pointer: that flag is on this module's OTHER parser
    # (`--materialize-authorities`), and the forecast parser rejects it
    # with a usage dump.  A refusal that points at a flag the door does
    # not have is the same defect as a remedy naming a missing extra.
    assert "--explain" not in text


def test_neither_prepared_door_declares_the_flag_its_refusal_omits():
    """The premise behind omitting the ``--explain`` pointer, asserted.

    ``prepared_single_domain_forecast`` registers ``--explain`` on its
    ``--materialize-authorities`` parser only; the forecast parser -- the
    one these refusals are printed by -- does not have it.  If that ever
    changes, this test fails and the pointer can come back.
    """

    from gpuwm import prepared_domain_tree_forecast as tree
    from gpuwm import prepared_single_domain_forecast as single

    tree_flags = {option for action in tree.build_parser()._actions
                  for option in action.option_strings}
    assert "--explain" not in tree_flags

    forecast_source = single.__loader__.get_source(single.__name__)
    body = forecast_source.split("def _parse_args(", 1)[1].split(
        "\ndef ", 1)[0]
    assert "add_explain_flag" not in body, (
        "the forecast parser now registers --explain; the refusal may "
        "carry its pointer again")


def test_prepared_tree_forecast_refuses_before_claiming_the_directory(
        monkeypatch, tmp_path, capsys):
    from gpuwm import prepared_domain_tree_forecast as runner

    monkeypatch.setattr(capabilities, "is_installed",
                        lambda module: module != "cupy")
    config = tmp_path / "config.toml"
    config.write_text("[experiment]\nname = \"x\"\n", encoding="utf-8")
    outdir = tmp_path / "run"

    code = runner.main([
        "--prepared-root", str(tmp_path / "prep"),
        "--preparation-receipt-sha256", "0" * 64,
        "--experiment-config", str(config),
        "--experiment-config-sha256", "0" * 64,
        "--outdir", str(outdir)])

    assert code == 2
    assert not outdir.exists()
    text = capsys.readouterr().err
    assert "cupy" in text and "gpu-cu12" in text
    assert "Traceback" not in text
    assert "--explain" not in text, (
        "this parser has no --explain; a pointer at it names a flag the "
        "door does not have")


def test_prepared_forecast_is_silent_when_the_runtime_is_present(
        monkeypatch, tmp_path, capsys):
    """The other direction: it must reach its own refusals, not ours."""

    from gpuwm import prepared_single_domain_forecast as runner

    monkeypatch.setattr(capabilities, "is_installed", lambda module: True)
    config = tmp_path / "config.toml"
    config.write_text("[experiment]\nname = \"x\"\n", encoding="utf-8")
    outdir = tmp_path / "run"
    try:
        runner.main(_prepared_argv(outdir, config))
    except BaseException:                                   # noqa: BLE001
        pass
    text = capsys.readouterr().err
    assert "this command needs cupy" not in text


# ---------------------------------------------------------------------------
# H3 / H2 -- the CLI front door
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("command", sorted(capabilities.COMMAND_REQUIREMENTS))
def test_the_front_door_refuses_every_gpu_command(monkeypatch, command):
    monkeypatch.setattr(capabilities, "is_installed",
                        lambda module: module != "cupy")
    with pytest.raises(capabilities.CapabilityMissing) as caught:
        capabilities.require_for_command(command)
    assert str(caught.value).startswith(f"gpuwm {command}:")


def test_the_front_door_is_silent_for_commands_that_need_no_card(monkeypatch):
    monkeypatch.setattr(capabilities, "is_installed",
                        lambda module: module != "cupy")
    for command in ("fetch", "render", "doctor", "version", "cases",
                    "report", "run-plan", "setup", "stream", "multi-run",
                    "import-namelist", "enprod"):
        capabilities.require_for_command(command)      # must not raise


def test_run_refuses_before_the_config_is_even_read(monkeypatch, capsys,
                                                    tmp_path):
    """H3: it used to select a GPU, spawn a worker and prepare the case."""

    from gpuwm import cli

    monkeypatch.setattr(capabilities, "is_installed",
                        lambda module: module != "cupy")
    missing_config = tmp_path / "not-here.toml"
    code = cli.main(["run", str(missing_config), "--outdir",
                     str(tmp_path / "out")])
    assert code == 2
    text = capsys.readouterr().err
    assert "this command needs cupy" in text
    assert "Traceback" not in text
    # The refusal tail is the reader's own invocation (UX finding N8), so
    # the config path legitimately appears there -- as their argv echoed
    # back, never as a report about the file.  The config was never
    # opened: outside that echo, its absence is not what was reported.
    assert "--explain for the reason" in text
    body = "\n".join(line for line in text.splitlines()
                     if "--explain for the reason" not in line)
    assert "not-here.toml" not in body


def test_a_supervisor_error_naming_a_missing_module_is_a_refusal(
        monkeypatch, capsys, tmp_path):
    """H3's residue: the worker's gap must not escape as a traceback.

    The front-door gate above makes the CuPy case unreachable, but a
    worker can fail on any import, and ``SupervisorError`` is a
    ``RuntimeError`` that ``run`` was not on the boundary's list for --
    the same mechanism that let a bare ``RuntimeError`` escape the
    terrain path at exit 1.
    """

    from gpuwm import capabilities as caps
    from gpuwm import cli

    monkeypatch.setattr(caps, "is_installed", lambda module: True)

    def explode(args):
        raise RuntimeError(
            "worker exited with status 1: ModuleNotFoundError: No module "
            "named 'cupy'; no durable manifest-valid checkpoint is available")

    monkeypatch.setattr(cli, "_dispatch", explode)
    code = cli.main(["run", str(tmp_path / "any.toml"), "--outdir",
                     str(tmp_path / "out")])
    text = capsys.readouterr().err
    assert code == 2
    assert "Traceback" not in text
    assert "gpu-cu12" in text


def test_a_supervisor_error_from_a_real_crash_keeps_its_traceback(
        monkeypatch, tmp_path):
    """The other direction: a crash must not be dressed as a usage error."""

    from gpuwm import capabilities as caps
    from gpuwm import cli

    monkeypatch.setattr(caps, "is_installed", lambda module: True)

    def explode(args):
        raise RuntimeError("worker exited with status 1: CUDA_ERROR_UNKNOWN")

    monkeypatch.setattr(cli, "_dispatch", explode)
    with pytest.raises(RuntimeError):
        cli.main(["run", str(tmp_path / "any.toml"), "--outdir",
                  str(tmp_path / "out")])


def test_go_dry_run_is_not_refused_but_warns(monkeypatch):
    """A dry run spends nothing, so it prints instead of refusing."""

    from gpuwm import capabilities as caps

    monkeypatch.setattr(caps, "is_installed", lambda module: module != "cupy")
    unmet = caps.missing(caps.COMMAND_REQUIREMENTS["go"])
    assert [item.module for item in unmet] == ["cupy"]
