"""The native terminal names a snapshot size it cannot take.

`arwen-tui --snapshot-width -1` printed "Error: ParseIntError { kind:
InvalidDigit }" and 65536 "PosOverflow", naming neither the option nor
the sizes it takes.  Runs against the binary GPUWM_TUI_BIN names; the
same rule is held by the crate's own unit test
(snapshot_dimensions_are_refused_by_option_and_range).
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


def _binary() -> str:
    executable = os.environ.get("GPUWM_TUI_BIN")
    if not executable or not Path(executable).is_file():
        pytest.skip("GPUWM_TUI_BIN must name the native terminal built from this tree")
    return executable


@pytest.mark.parametrize("option,largest", [("--snapshot-width", 400),
                                            ("--snapshot-height", 160)])
@pytest.mark.parametrize("value", ["-1", "0", "65536", "nan"])
def test_native_snapshot_dimension_names_the_option_and_range(tmp_path, option, largest, value):
    result = subprocess.run(
        [_binary(), "--snapshot", str(tmp_path / "bad.html"), option, value],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "APPDATA": str(tmp_path),
             "HOME": str(tmp_path)})
    assert result.returncode != 0
    assert option in result.stderr and f"from 1 to {largest}" in result.stderr
    assert "ParseIntError" not in result.stderr
    assert not (tmp_path / "bad.html").exists()
