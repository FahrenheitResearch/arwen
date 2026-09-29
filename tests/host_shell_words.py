"""Read a printed command line the way the shell it was printed for reads it.

``gpuwm sim --print-command`` prints a POSIX line on Linux and macOS and a
PowerShell line on Windows (:func:`gpuwm.prep_output.host_shell`).  A test
that split the Windows line with ``shlex`` would be checking a spelling
the reader's shell never sees, so the PowerShell side is parsed by
PowerShell's own parser.
"""

from __future__ import annotations

import base64
import json
import os
import shlex
import shutil
import subprocess


def powershell() -> str | None:
    """A PowerShell executable on this machine: pwsh, else Windows PowerShell."""

    return shutil.which("pwsh") or shutil.which("powershell")


# Every element of the one command the line holds, as the program being
# called receives it.  A bare word is a string constant or, for a dash
# word, a parameter token whose text is passed through; anything else
# (a variable, a subexpression, an expandable string) is reported as
# what it is, so a line that would expand something fails the caller's
# comparison instead of passing silently.  Non-ASCII output is escaped so
# the console encoding cannot change it.
_PARSE = r"""
$text = $env:GPUWM_TEST_COMMAND_LINE
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseInput($text, [ref]$tokens, [ref]$errors)
$calls = @($ast.FindAll({ param($node) $node -is [System.Management.Automation.Language.CommandAst] }, $true))
$words = New-Object System.Collections.Generic.List[string]
if ($calls.Count -eq 1) {
  foreach ($element in $calls[0].CommandElements) {
    if ($element -is [System.Management.Automation.Language.StringConstantExpressionAst]) {
      $words.Add($element.Value)
    } elseif ($element -is [System.Management.Automation.Language.CommandParameterAst]) {
      $words.Add($element.Extent.Text)
    } else {
      $words.Add('<' + $element.GetType().Name + ' ' + $element.Extent.Text + '>')
    }
  }
}
$document = @{
  errors = @($errors | ForEach-Object { $_.Message })
  commands = $calls.Count
  words = @($words)
}
$json = ConvertTo-Json -Compress -InputObject $document
[regex]::Replace($json, '[^\x00-\x7f]', { param($m) '\u{0:x4}' -f [int][char]$m.Value })
"""


def _encoded(script: str) -> str:
    return base64.b64encode(script.encode("utf-16-le")).decode("ascii")


def run_powershell(script: str, *, environment: dict | None = None,
                   timeout: float = 120) -> subprocess.CompletedProcess:
    """Run ``script`` in PowerShell exactly as typed at its prompt."""

    executable = powershell()
    if executable is None:
        raise RuntimeError("no PowerShell on this machine")
    return subprocess.run(
        [executable, "-NoLogo", "-NoProfile", "-NonInteractive",
         "-EncodedCommand", _encoded(script)],
        capture_output=True, text=True, timeout=timeout,
        env={**os.environ, **(environment or {})})


def powershell_words(line: str) -> list[str]:
    """The arguments PowerShell's parser finds in ``line``, which must be
    exactly one command and parse without an error."""

    result = run_powershell(
        _PARSE, environment={"GPUWM_TEST_COMMAND_LINE": line.strip()})
    assert result.returncode == 0, result.stderr
    document = json.loads(result.stdout)
    assert not document["errors"], (line, document["errors"])
    assert document["commands"] == 1, (line, document)
    return list(document["words"])


def host_shell_words(line: str) -> list[str]:
    """``line`` split the way the shell it was printed for splits it."""

    from gpuwm.prep_output import host_shell

    if host_shell() == "powershell":
        return powershell_words(line)
    return shlex.split(line)
