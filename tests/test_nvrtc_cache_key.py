"""A160: no NVRTC compile can be served an entry built under another -ftz.

NVRTC 13.4's ComputeCache key ignores ``-ftz`` for one program name and one
source (gpuwm/nvrtc_cache_key.py has the measurement).  CuPy's routes name
their programs after the options (``RawModule``) or a fresh directory
(``compile_using_nvrtc``); every direct NVRTC compile in the tree goes
through :func:`gpuwm.nvrtc_cache_key.compile_program`, which appends the
options to the source.  The CPU rows hold the stamp and the source scan;
the card rows compile in fresh processes that share one cache, in both
orders, and read the flush mode back from the kernel: ``FLT_MIN * 0.5f`` is
0 under ``-ftz=true`` and the subnormal 2**-127 under ``--ftz=false`` on
every architecture.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from conftest import requires_gpu

from gpuwm.nvrtc_cache_key import SYMBOL, keyed_source, stamp

ROOT = Path(__file__).resolve().parents[1]

#: Options that change the code NVRTC emits for one source.
CODE_OPTIONS = ("-ftz=true", "--ftz=false", "--fmad=false", "--fmad=true",
                "-prec-div=false", "-prec-sqrt=false", "-use_fast_math",
                "-maxrregcount=32", "-lineinfo", "-G", "-arch=sm_120",
                "-arch=compute_89", "-Xptxas=-O1")


def test_every_code_option_changes_the_stamp():
    base = ("-std=c++17",)
    stamps = {stamp(base)}
    for option in CODE_OPTIONS:
        stamps.add(stamp(base + (option,)))
    assert len(stamps) == len(CODE_OPTIONS) + 1
    assert stamp(base + ("-ftz=true",)) == stamp(["-std=c++17", "-ftz=true"])
    # Order is kept: NVRTC honours the last of two contradictory flags.
    assert stamp(("-ftz=true", "--ftz=false")) != stamp(
        ("--ftz=false", "-ftz=true"))


def test_the_stamp_is_appended_once_and_moves_no_line():
    source = "#define A 1\n__global__ void k(float* y) { y[0] = 1.0f / 3.0f; }"
    options = ("-std=c++17", "-ftz=true")
    keyed = keyed_source(source, options)
    lines = keyed.splitlines()
    assert lines[:2] == source.splitlines()
    assert lines[2] == stamp(options) and len(lines) == 3
    assert keyed_source(keyed, options) == keyed
    assert keyed_source(source + "\n", options) == source + "\n" + stamp(
        options) + "\n"


def test_the_stamp_is_one_c_string_literal():
    line = stamp(('-DNAME="a b"', "-Ic:\\x", "\x1f"))
    literal = line.split("= ", 1)[1].rstrip(";")
    assert literal.startswith('"') and literal.endswith('"')
    body = literal[1:-1]
    # every quote in the body is escaped or closes a hex-escape segment
    assert re.sub(r'\\.|""', "", body).count('"') == 0
    assert SYMBOL in line and line.startswith('extern "C" __device__ const char')


#: Direct NVRTC program constructors, and the files allowed to call them:
#: the keyed helper, and the two build-banner probes that compile an empty
#: source (no code, nothing to be served).
_DIRECT = re.compile(r"\b(?:nvrtc\.createProgram|_NVRTCProgram)\(")
_ALLOWED = {
    "gpuwm/nvrtc_cache_key.py": None,
    "gpuwm/certify/compile_platform.py": '("", ',
    "gpuwm/core/device_probe.py": '("", ',
}


def test_no_direct_nvrtc_compile_bypasses_the_key():
    """A160's guard: a fixed-name NVRTC compile outside the helper can be
    answered with a cubin or PTX built under the other flush mode."""
    listed = subprocess.run(
        ["git", "ls-files", "--", "gpuwm/*.py", "tools/*.py", "tests/*.py"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    assert "gpuwm/nvrtc_cache_key.py" in listed
    offenders = []
    for rel in listed:
        if rel == "tests/test_nvrtc_cache_key.py":
            continue
        text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        for match in _DIRECT.finditer(text):
            line = text[:match.start()].count("\n") + 1
            if rel in _ALLOWED:
                need = _ALLOWED[rel]
                if need is None or text[match.end() - 1:].startswith(need):
                    continue
            offenders.append(f"{rel}:{line}")
    assert not offenders, (
        "direct NVRTC compiles outside gpuwm.nvrtc_cache_key.compile_program:"
        f" {offenders}.  NVRTC's cache ignores -ftz for a fixed program name"
        " (A160); compile through the helper.")


# -- card rows ---------------------------------------------------------------

_SOURCE = r'''
extern "C" __global__ void half(const float* x, float* y) { y[0] = x[0] * 0.5f; }
'''

_CHILD = r'''
import json, sys
import numpy as np
import cupy as cp
from cupy.cuda import compiler, nvrtc
route, flag, source = sys.argv[1], sys.argv[2], sys.argv[3]
cc = cp.cuda.Device().compute_capability
if route == "raw":
    program = nvrtc.createProgram(source, "a160.cu", [], [])
    nvrtc.compileProgram(program, ["-std=c++17", flag, f"-arch=sm_{cc}"])
    blob = nvrtc.getCUBIN(program)
    module = cp.cuda.function.Module(); module.load(blob)
elif route == "keyed":
    from gpuwm.nvrtc_cache_key import compile_program
    blob = compile_program(source, ("-std=c++17", flag, f"-arch=sm_{cc}"),
                           name="a160.cu", target="cubin")
    module = cp.cuda.function.Module(); module.load(blob)
elif route == "rawmodule":
    module = cp.RawModule(code=source, options=("-std=c++17",))
elif route == "direct":
    blob, _ = compiler.compile_using_nvrtc(source, ("-std=c++17", flag), cc)
    module = cp.cuda.function.Module(); module.load(blob)
x = cp.asarray(np.array([np.finfo(np.float32).tiny], np.float32))
y = cp.zeros(1, cp.float32)
module.get_function("half")((1,), (1,), (x, y))
value = float(cp.asnumpy(y)[0])
print(json.dumps({"flushed": value == 0.0, "value": value, "cc": cc,
                  "nvrtc": list(nvrtc.getVersion())}))
'''


def _compile_in_fresh_process(tmp_path, route, flag):
    env = dict(os.environ)
    for name in ("CUDA_CACHE_DISABLE", "CUPY_CACHE_IN_MEMORY"):
        env.pop(name, None)
    env.update(HOME=str(tmp_path), CUDA_CACHE_PATH=str(tmp_path / "nv"),
               CUPY_CACHE_DIR=str(tmp_path / "cupy"),
               PYTHONPATH=str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""))
    done = subprocess.run(
        [sys.executable, "-c", _CHILD, route, flag, _SOURCE], cwd=tmp_path,
        env=env, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def _pair(tmp_path, first, second):
    return (_compile_in_fresh_process(tmp_path, *first),
            _compile_in_fresh_process(tmp_path, *second))


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("order", ["ftz-first", "ieee-first"])
def test_nvrtc_still_ignores_ftz_for_a_fixed_program_name(tmp_path, order):
    """The mechanism: the unkeyed fixed-name compile IS served the other
    mode's cubin.  Measured on sm_120 under NVRTC 13.4 only; when this
    fails there, NVIDIA keyed -ftz and gpuwm/nvrtc_cache_key.py can go."""
    import cupy
    toolchain = (cupy.cuda.Device().compute_capability,
                 tuple(cupy.cuda.nvrtc.getVersion()))
    if toolchain != ("120", (13, 4)):
        pytest.skip(f"the A160 collision was measured on sm_120 under NVRTC"
                    f" 13.4; this is sm_{toolchain[0]} under NVRTC"
                    f" {toolchain[1]}")
    arms = [("raw", "-ftz=true"), ("raw", "--ftz=false")]
    if order == "ieee-first":
        arms.reverse()
    first, second = _pair(tmp_path, *arms)
    assert first["flushed"] == (arms[0][1] == "-ftz=true")
    assert second["flushed"] == first["flushed"], (
        "NVRTC honoured the second process's -ftz: its cache key includes the"
        " flag now.  Retire gpuwm/nvrtc_cache_key.py and this file's guard.")


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("arms", [
    (("keyed", "-ftz=true"), ("keyed", "--ftz=false")),
    (("keyed", "--ftz=false"), ("keyed", "-ftz=true")),
    (("rawmodule", "-ftz=true"), ("direct", "--ftz=false")),
    (("direct", "--ftz=false"), ("rawmodule", "-ftz=true")),
], ids=["keyed-ftz-first", "keyed-ieee-first", "cupy-ftz-first",
        "cupy-ieee-first"])
def test_every_compile_route_gets_the_flush_mode_it_asked_for(tmp_path, arms):
    """The keyed helper and both CuPy routes, each pair in fresh processes
    sharing one cache.  ``RawModule`` compiles with -ftz=true whatever the
    caller passes (CuPy appends it)."""
    import cupy  # noqa: F401  (marks this test for -m "not gpu")
    for got, (route, flag) in zip(_pair(tmp_path, *arms), arms):
        assert got["flushed"] == (flag == "-ftz=true"), (route, flag, got)
