"""The isolated model runtime cannot evade the engine's local GPU opt-out."""
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys

import pytest

from gpuwm import earth2_run, earth2_runner


def test_generation_refuses_before_runtime_probe_or_output(monkeypatch, tmp_path):
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    def unexpected():
        pytest.fail("runtime interpreter was probed despite the GPU opt-out")
    monkeypatch.setattr(earth2_run, "earth2_python", unexpected)
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match="GPUWM_NO_LOCAL_GPU"):
        earth2_run.run_model(model="fcn3", cycle=datetime(2024, 9, 24, 12),
                             hours=72, area=None, out=output, member="0,1,2")
    assert not output.exists()


def test_standalone_runner_refuses_before_request_or_optional_imports(tmp_path):
    # -S hides site packages. An absent request proves the guard precedes even
    # request loading, so no Earth2Studio/Torch install or download is needed.
    result = subprocess.run([sys.executable, "-S", str(Path(earth2_runner.__file__)),
                             str(tmp_path / "absent.json")],
                            env=dict(os.environ, GPUWM_NO_LOCAL_GPU="1"),
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "GPUWM_NO_LOCAL_GPU is set" in result.stderr
    assert "FileNotFoundError" not in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
