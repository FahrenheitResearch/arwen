"""Full serialized RK continuation, with fresh processes for every phase."""
import os
from pathlib import Path
import subprocess
import sys

import pytest
from conftest import requires_gpu


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("version,mixlength,cycling", (
    ("wrf_461", 1, False), ("wrf_461", 2, False),
    ("gsd_41", 1, False), ("gsd_41", 2, False),
    ("wrf_461", 1, True), ("wrf_461", 2, True),
    ("gsd_41", 2, True)))
def test_every_serialized_field_survives_a_mid_run_restart(tmp_path, version, mixlength, cycling):
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / "tools" / "mynn_review_restart.py"),
               "--folder", str(tmp_path), "--version", version,
               "--mixlength", str(mixlength), "--out", str(tmp_path / "receipt.json")]
    if cycling:
        command.append("--cycling")
    for phase in ("reference", "split", "resume", "check"):
        result = subprocess.run(command + ["--phase", phase], cwd=root,
                                env=os.environ.copy(), text=True,
                                capture_output=True)
        assert result.returncode == 0, result.stdout + result.stderr
