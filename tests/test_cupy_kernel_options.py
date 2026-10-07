"""No CuPy-compiled kernel spells FTZ itself.

CuPy appends ``-ftz=true`` after the caller's options to every kernel it
compiles (cupy/cuda/compiler.py _compile_with_cache_cuda).  NVRTC 12
refuses a program whose FTZ is spelled twice ("--ftz (-ftz) defined more
than once"), so a kernel that also asked for ``--ftz=true`` failed to
compile on every gpu-cu12 install, and on a CUDA 13 CuPy that fell back
to a system NVRTC 12: measured as the MYNN gsd_41 validation scan
stopping a 2-card HRRR forecast at its first step.  The arithmetic is the
same without the second spelling.  A kernel that needs IEEE subnormals
keeps its ``--ftz=false`` route outside RawModule (gpuwm.nvrtc_ptx_cache).
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_CUPY_COMPILED = {"RawKernel", "RawModule", "ElementwiseKernel",
                  "ReductionKernel"}


def _offenders():
    found = []
    for path in sorted((ROOT / "gpuwm").rglob("*.py")):
        if "verify" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", getattr(node.func, "id", None))
            if name not in _CUPY_COMPILED:
                continue
            for keyword in node.keywords:
                if keyword.arg != "options":
                    continue
                for constant in ast.walk(keyword.value):
                    if (isinstance(constant, ast.Constant)
                            and isinstance(constant.value, str)
                            and constant.value.lstrip("-").startswith("ftz=true")):
                        found.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    return found


def test_no_cupy_compiled_kernel_spells_ftz_true_itself():
    assert _offenders() == []
