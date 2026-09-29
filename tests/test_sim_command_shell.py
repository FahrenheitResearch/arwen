"""``gpuwm sim --print-command`` prints a line the reader's shell can run.

The line was always ``shlex.join``, which is POSIX quoting.  On Windows
it is pasted into PowerShell, where a quoted interpreter path followed
by ``-m`` is "Unexpected token '-m'", and a folder with an apostrophe
came out as POSIX quote concatenation.  The line is now quoted for the
shell it is printed in, and the Windows spelling is checked by
PowerShell itself, running it where one is installed.
"""

from __future__ import annotations

import json
import sys

import pytest

from gpuwm import prep_output, stage_cli
from gpuwm.cli import main as cli_main
from host_shell_words import powershell, powershell_words, run_powershell
from test_stage_seams import _authority, _single_domain_bundle, _tree_bundle


needs_powershell = pytest.mark.skipif(
    powershell() is None, reason="PowerShell is not installed on this machine")

# Words a PowerShell line has to carry through unchanged: each one is
# something PowerShell would otherwise read as syntax, a number, or a
# quote that ends the string.  The one word left out is a lone "--%":
# PowerShell takes it as its stop-parsing marker even quoted, and no
# spelling passes it to a native program, so no command gpuwm prints
# carries it.
_AWKWARD = [
    "plain", "with space", "owner's case", "Drew\u2019s case",
    "\u2018single\u2019 \u201adouble\u201b", "a,b", "$HOME", "`tick",
    "semi;colon", "x--%", "--", "@splat", "#hash", "(paren)", "{brace}",
    "100%", "0x10", "1.10", "5kb", "1e5", "-5", "-.5", "-name:value",
    "a|b", "a&b", "<in", "~", "\u00fcn\u00efcode", "C:\\cases\\x y\\case.toml",
    "--outdir", "-m",
]


@needs_powershell
def test_powershell_runs_the_line_with_every_word_intact(tmp_path):
    """Executed, not only parsed: the words the program receives."""

    echo = tmp_path / "echo_words.py"
    echo.write_text("import json, sys\n"
                    "print(json.dumps(sys.argv[1:], ensure_ascii=True))\n",
                    encoding="utf-8")
    line = prep_output.shell_command([sys.executable, str(echo), *_AWKWARD],
                                     shell="powershell")
    result = run_powershell(line)
    assert result.returncode == 0, (line, result.stderr)
    assert json.loads(result.stdout.strip().splitlines()[-1]) == _AWKWARD


def test_a_quoted_program_is_called_with_the_call_operator():
    line = prep_output.shell_command(
        ["C:\\Program Files\\Python\\python.exe", "-m", "gpuwm.cli"],
        shell="powershell")
    assert line == "& 'C:\\Program Files\\Python\\python.exe' -m gpuwm.cli"
    assert prep_output.shell_command(
        ["C:\\Python\\python.exe", "-m", "gpuwm.cli"], shell="powershell"
    ) == "C:\\Python\\python.exe -m gpuwm.cli"


def test_the_posix_line_is_unchanged():
    words = ["/opt/my python/bin/python", "-m", "gpuwm.cli", "owner's case"]
    import shlex

    assert prep_output.shell_command(words, shell="posix") == shlex.join(words)


def _printed(monkeypatch, capsys, argv):
    assert cli_main(argv) == 0
    return capsys.readouterr().out.strip().splitlines()[-1]


@needs_powershell
@pytest.mark.parametrize("layout", ["single", "tree"])
def test_the_windows_sim_line_parses_to_the_runner_argv(tmp_path, monkeypatch,
                                                        capsys, layout):
    """The finding's own case: an interpreter path that needs quoting,
    and folders with an apostrophe and a curly apostrophe."""

    monkeypatch.setattr(prep_output, "host_shell", lambda: "powershell")
    interpreter = str(tmp_path / "Python 3.12" / "python.exe")
    monkeypatch.setattr(sys, "executable", interpreter)
    if layout == "single":
        root = _single_domain_bundle(tmp_path / "bundle's folder")
    else:
        root = _tree_bundle(tmp_path / "bundle's folder")
    config, wps = _authority(tmp_path / "Drew\u2019s authority")
    output = tmp_path / "run folder"
    argv = ["sim", str(root), "--experiment-config", str(config),
            "--outdir", str(output), "--print-command"]
    if layout == "single":
        argv[4:4] = ["--wps-namelist", str(wps)]
    line = _printed(monkeypatch, capsys, argv)
    assert line.startswith("& ")
    words = powershell_words(line)
    runner = (stage_cli.SINGLE_DOMAIN_RUNNER if layout == "single"
              else stage_cli.TREE_RUNNER)
    assert words[:3] == [interpreter, "-m", runner]
    assert words[words.index("--prepared-root") + 1] == str(root)
    assert words[words.index("--experiment-config") + 1] == str(config)
    if layout == "single":
        assert words[words.index("--wps-namelist") + 1] == str(wps)
    assert words[words.index("--outdir") + 1].startswith(str(output))
    assert not output.exists()
