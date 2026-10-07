"""CPU-only public driver admission and live surface capture checks."""
import ast
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools.da_cycle_prepared import build_parser
from tools.da_member_leg import capture_surface_diagnostics


def _driver(dewpoint=False):
    fields = {name: np.full((2, 3), value, dtype=np.float32)
              for name, value in (("t2", 291), ("u10", 3), ("v10", 4))}
    if dewpoint:
        fields.update(q2=np.full((2, 3), .01, dtype=np.float32),
                      psfc=np.full((2, 3), 93_000, dtype=np.float32))
    # A distinct bottom-level pressure would expose a wrong source selection.
    return SimpleNamespace(fields=fields, state=SimpleNamespace(
        p=np.full((4, 2, 3), 85_000, dtype=np.float32)))


def test_default_surface_capture_does_not_require_new_diagnostics():
    captured = capture_surface_diagnostics(_driver())
    assert set(captured) == {"t2", "u10", "v10"}
    assert all(array.dtype == np.float64 for array in captured.values())


def test_opted_in_capture_uses_live_q2_and_driver_psfc_in_pa():
    driver = _driver(dewpoint=True)
    captured = capture_surface_diagnostics(driver, dewpoint=True)
    assert set(captured) == {"t2", "u10", "v10", "q2", "psfc"}
    np.testing.assert_array_equal(captured["q2"], driver.fields["q2"].astype(np.float64))
    np.testing.assert_array_equal(captured["psfc"], np.full((2, 3), 93_000.0))
    assert not np.any(captured["psfc"] == driver.state.p[0])
    driver.fields["q2"][...] = 0
    driver.fields["psfc"][...] = 0
    assert np.all(captured["q2"] > 0)
    assert np.all(captured["psfc"] == 93_000)


def test_missing_live_psfc_fails_instead_of_using_bottom_pressure():
    driver = _driver(dewpoint=True)
    driver.fields.pop("psfc")
    with pytest.raises(RuntimeError, match="live driver diagnostics.*psfc"):
        capture_surface_diagnostics(driver, dewpoint=True)


def test_cli_admits_explicit_dewpoint_flag_without_a_default_sigma():
    base = ["--prepared-root", "synthetic-test-prepared", "--physics-profile", "synthetic-test-profile",
            "--proof-sha256", "synthetic-proof", "--source-manifest-sha256", "synthetic-source",
            "--prepared-content-sha256", "synthetic-prepared", "--run-seconds", "60",
            "--history-interval-seconds", "120",
            "--out", "synthetic-test-output"]
    assert build_parser().parse_args(base).sfc_td_sigma_k is None
    args = build_parser().parse_args(base + ["--surface-obs", "synthetic-surface.json",
                                            "--sfc-td-sigma-k", "2", "--hydrometeors"])
    assert args.sfc_td_sigma_k == 2.0 and args.hydrometeors


def test_native_dewpoint_admission_precedes_any_cuda_import():
    source = Path(__file__).resolve().parents[1] / "tools/da_cycle_prepared.py"
    tree = ast.parse(source.read_text())
    cycle = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "cycle")
    cuda_imports = [node.lineno for node in ast.walk(cycle) if isinstance(node, ast.Import)
                    and any(alias.name == "cupy" for alias in node.names)]
    require_calls = [node.lineno for node in ast.walk(cycle) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == "require_native_operator"]
    assert cuda_imports and require_calls and min(require_calls) < min(cuda_imports)
    text = ast.get_source_segment(source.read_text(), cycle)
    assert "--sfc-td-sigma-k requires --hydrometeors so Q2 can constrain vapor" in text
    assert "--sfc-td-sigma-k must be a finite positive standard deviation in K" in text


@pytest.mark.parametrize("extra,message", [
    (["--sfc-td-sigma-k", "2"], "requires --hydrometeors"),
    (["--sfc-td-sigma-k", "0", "--hydrometeors"], "finite positive standard deviation"),
    (["--sfc-td-sigma-k", "-1", "--hydrometeors"], "finite positive standard deviation"),
    (["--sfc-td-sigma-k", "nan", "--hydrometeors"], "finite positive standard deviation"),
])
def test_actual_cli_dewpoint_invalid_options_stop_before_model_work(extra, message):
    root = Path(__file__).resolve().parents[1]
    args = [sys.executable, "-m", "tools.da_cycle_prepared",
            "--prepared-root", "synthetic-test-prepared", "--physics-profile", "synthetic-test-profile",
            "--proof-sha256", "synthetic-proof", "--source-manifest-sha256", "synthetic-source",
            "--prepared-content-sha256", "synthetic-prepared", "--run-seconds", "60",
            "--history-interval-seconds", "120", "--out", "synthetic-test-output",
            "--surface-obs", "synthetic-surface.json", "--obs", "synthetic-obs.nc"] + extra
    env = {**os.environ, "GPUWM_NO_LOCAL_GPU": "1", "CUDA_VISIBLE_DEVICES": "-1"}
    result = subprocess.run(args, cwd=root, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 2 and message in result.stderr


def test_actual_cli_missing_native_library_fails_before_cuda(tmp_path):
    root = Path(__file__).resolve().parents[1]
    args = [sys.executable, "-m", "tools.da_cycle_prepared",
            "--prepared-root", "synthetic-test-prepared", "--physics-profile", "synthetic-test-profile",
            "--proof-sha256", "synthetic-proof", "--source-manifest-sha256", "synthetic-source",
            "--prepared-content-sha256", "synthetic-prepared", "--run-seconds", "60",
            "--history-interval-seconds", "120", "--out", "synthetic-test-output",
            "--surface-obs", "synthetic-surface.json", "--obs", "synthetic-obs.nc",
            "--sfc-td-sigma-k", "2", "--hydrometeors"]
    missing = tmp_path / "missing-native-library.dll"
    env = {**os.environ, "GPUWM_NO_LOCAL_GPU": "1", "CUDA_VISIBLE_DEVICES": "-1",
           "GPUWM_OBSSCORE_BRIDGE": str(missing)}
    result = subprocess.run(args, cwd=root, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert "GPUWM_OBSSCORE_BRIDGE names a missing file" in result.stderr
    assert "cupy" not in result.stderr.lower()
