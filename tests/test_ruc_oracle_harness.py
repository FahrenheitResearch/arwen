"""The actual comparison commands must propagate a planted HFX bit flip."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
TOOL = ROOT / "tools/ruc_lsm_gpu_oracle"


def test_comparator_keeps_signed_zero_words_and_rejects_nonfinite():
    from importlib.util import module_from_spec, spec_from_file_location
    sys.path.insert(0, str(TOOL))
    spec = spec_from_file_location("ruc_column_compare", TOOL / "compare.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    positive = np.asarray([0., 1., np.nan, np.inf], np.float32)
    negative = np.asarray([-0., np.nextafter(np.float32(1), np.float32(2)), np.nan, np.inf], np.float32)
    distance, absolute = module.grade_field(positive, negative)
    np.testing.assert_array_equal(distance, [1, 1, 1, 1])
    assert np.all(np.isinf(absolute[2:]))


def _posix_bash():
    """A bash that runs the harness script, or None.

    On Windows the first bash on PATH can be System32's WSL launcher, which
    is no shell without an installed distribution (public CI 38002026864,
    cpu windows-2025: "Windows Subsystem for Linux has no installed
    distributions"). Git for Windows ships the bash the harness is run with.
    """
    if os.name != "nt":
        return shutil.which("bash")
    git = shutil.which("git")
    if git:
        # git.exe sits in Git\cmd, Git\bin or Git\mingw64\bin; bash.exe in Git\bin.
        for root in list(Path(git).resolve().parents)[:3]:
            for candidate in (root / "bin" / "bash.exe", root / "usr" / "bin" / "bash.exe"):
                if candidate.is_file():
                    return str(candidate)
    found = shutil.which("bash")
    if found and Path(found).resolve().parent.name.lower() != "system32":
        return found
    return None


@pytest.mark.parametrize("arm", ["free", "replay"])
def test_shell_comparison_failure_exits_nonzero(tmp_path, arm):
    bash = _posix_bash()
    if bash is None:
        pytest.skip("requires bash")
    sys.path.insert(0, str(TOOL))
    import layout
    from columns import PROFILES
    with np.load(ROOT / "tests/data/oracles/ruc_lsm/wrf461-nzs9.npz") as data:
        receipt = json.loads(bytes(data["receipt"]).decode())
        meta = receipt["options"] | {"regime": receipt["regime"]}
        (tmp_path / "case.json").write_text(json.dumps(meta))
        parts = [data[f"init__{n}"] for n in layout.INIT_2D + layout.INIT_PROFILES]
        for k in range(1, meta["nsteps"] + 1):
            parts.extend(data[f"step{k}__{n}"] for n in layout.OUT_2D + PROFILES +
                         (layout.SFCEVP_ONCE_STEP, layout.SFCEVP_ONCE_ACC))
        raw = b"".join(a.astype("<f4").tobytes() for a in parts)
        for reference in ("defined", "o0", "stock"):
            (tmp_path / f"outputs-{reference}.bin").write_bytes(raw)
        for replay in (False, True):
            arrays = {n: data[n].copy() for n in data.files if n != "receipt"}
            for k in range(1, meta["nsteps"] + 1):
                arrays[f"step{k}__sfcevp"] = data[f"step{k}__" + (
                    layout.SFCEVP_ONCE_STEP if replay else layout.SFCEVP_ONCE_ACC)].copy()
            if replay == (arm == "replay"):
                arrays["step1__hfx"].view("u4")[0] ^= np.uint32(1)
            np.savez(tmp_path / ("woof-replay.npz" if replay else "woof.npz"), **arrays)
    script = (TOOL / "run_case.sh").read_text()
    comparisons = script[script.rindex('cd "$out"'):]
    # The harness calls bare ``python``, the oracle box's own.  Hand the
    # comparison this suite's interpreter (which has numpy) whatever the
    # launching shell's PATH holds: a Stage 1 started as venv/bin/python on
    # a box whose only system interpreter is python3 failed here with
    # "python: command not found" before any comparison ran (2.9.0 lane A1,
    # test host A).  A shell shim, not a symlink, so a venv interpreter keeps its
    # own site-packages.
    shim = tmp_path / "interpreter"
    shim.mkdir()
    if os.name == "nt":
        interpreter_dir = str(Path(sys.executable).parent)
    else:
        (shim / "python").write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
        (shim / "python").chmod(0o755)
        interpreter_dir = str(shim)
    result = subprocess.run([bash, "-c", "set -euo pipefail\n" + comparisons],
                            env=os.environ | {
                                "out": str(tmp_path), "tool": str(TOOL),
                                "PATH": interpreter_dir + os.pathsep
                                + os.environ.get("PATH", "")},
                            text=True, capture_output=True)
    assert "hfx" in result.stdout, result.stdout + result.stderr
    assert result.returncode != 0, result.stdout + result.stderr
