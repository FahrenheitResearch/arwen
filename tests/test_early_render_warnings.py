"""Early and every-frame render subprocesses relay explicit warnings."""

import os
import sys

import pytest

from gpuwm import first_products, go_cli, live_products


WARNING = 'warning: the left subtitle does not fit the plot width and was drawn as "Init..."'
GEOREF = "WARNING georef fixture is unreadable; starting a fresh record"
PROGRESS = "TIMING total=5"


@pytest.mark.parametrize("entry", [first_products._run_render, live_products._run_render],
                         ids=["early", "every-frame"])
@pytest.mark.parametrize("exit_code", [0, 1])
@pytest.mark.parametrize("warnings", [True, False], ids=["warnings", "quiet"])
def test_render_worker_relays_warnings(entry, exit_code, warnings,
                                      monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(go_cli, "_stage_cwd", lambda: tmp_path)
    monkeypatch.setattr(go_cli, "_stage_env", lambda: dict(os.environ))
    lines = [PROGRESS, WARNING, GEOREF] if warnings else [PROGRESS]
    stderr = "\n".join(lines) + "\n"
    script = ("import sys; print('progress fixture'); "
              f"sys.stderr.write({stderr!r}); sys.exit({exit_code})")

    completed = entry([sys.executable, "-S", "-c", script])

    captured = capsys.readouterr()
    assert completed.returncode == exit_code
    assert completed.stdout == "progress fixture\n"
    assert completed.stderr == stderr
    assert not first_products._RUNNING
    assert captured.err.splitlines() == ([WARNING, GEOREF] if warnings else [])
    assert captured.out == ""
