"""Stage subprocesses import from the install, not from the caller.

Every stage of the prepared chain -- both GFS routes and both HRRR
routes -- is spawned as ``python -m MODULE``, and ``-m`` prepends the
CURRENT DIRECTORY to ``sys.path``, ahead of the installed package.
:func:`gpuwm.go_cli._stage_cwd` is deliberately the caller's directory
so a relative ``--out`` means what the person typing it meant, which is
what made this reachable: a chain started from inside a source checkout
imported that checkout instead of the install, and a live run was
hijacked exactly that way.

Moving the cwd would have fixed the imports and broken the relative
paths.  These pin the fix that separates the two.
"""

from __future__ import annotations

import gpuwm.go_cli as go_cli


def test_stage_subprocesses_run_with_a_safe_import_path():
    env = go_cli._stage_env()
    assert env["PYTHONSAFEPATH"] == "1"
    # It is the inherited environment plus that, not a replacement.
    import os

    assert env.get("PATH") == os.environ.get("PATH")


def test_a_poisoned_working_directory_cannot_hijack_a_stage_import(
        tmp_path):
    """`python -m` prepends CWD to sys.path; a live run was hijacked.

    The stage cwd is deliberately the caller's directory so a relative
    --out means what they typed, so the fix cannot be to move it.  This
    proves the env var does the job the cwd change would have.
    """

    import os
    import subprocess
    import sys

    (tmp_path / "json.py").write_text(
        "raise SystemExit('HIJACKED')\n", encoding="utf-8")
    probe = ["-c", "import json; print(json.__name__)"]

    poisoned = subprocess.run(
        [sys.executable, *probe], cwd=str(tmp_path), text=True,
        capture_output=True, env={**os.environ})
    protected = subprocess.run(
        [sys.executable, *probe], cwd=str(tmp_path), text=True,
        capture_output=True, env=go_cli._stage_env())

    assert "HIJACKED" in (poisoned.stdout + poisoned.stderr)
    assert protected.returncode == 0
    assert protected.stdout.strip() == "json"


def test_every_stage_spawn_goes_through_the_hardened_environment():
    """One spawn point, so one place had to be got right."""

    import inspect

    source = inspect.getsource(go_cli._run_stage)
    assert "env=_stage_env()" in source
    assert source.count("subprocess.Popen(") == 1
