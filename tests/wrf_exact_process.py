"""Run a word-for-word WRF proof with its process-level arithmetic opt-in."""
from functools import wraps
import gc
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def wrf_exact_process(test):
    """Keep strict source selection and compiler hooks out of the parent suite.

    Arguments must be JSON values or paths. These column proofs need no pytest
    fixtures; their parameterized inputs are passed to the same test in a fresh
    interpreter, before gpuwm or CuPy selects any kernels.
    """
    @wraps(test)
    def run(*args, **kwargs):
        if (os.environ.get("GPUWM_WRF_EXACT") == "1"
                and os.environ.get("GPUWM_NUMERICS") == "strict"):
            return test(*args, **kwargs)
        import cupy as cp

        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                pytest.skip("no CUDA device")
        except cp.cuda.runtime.CUDARuntimeError:
            pytest.skip("no CUDA runtime")

        # Earlier GPU tests can leave cycles and cached allocations behind.
        gc.collect()
        cp.cuda.Device().synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        arguments = inspect.signature(test).bind(*args, **kwargs).arguments
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, GPUWM_WRF_EXACT="1", GPUWM_NUMERICS="strict")
        env["PYTHONPATH"] = os.pathsep.join(
            [str(root / "tests"), str(root), env.get("PYTHONPATH", "")])
        command = """
import json, runpy, sys
import pytest
namespace = runpy.run_path(sys.argv[1])
try:
    namespace[sys.argv[2]](**json.loads(sys.argv[3]))
except pytest.skip.Exception as exc:
    print(str(exc))
    sys.exit(77)
"""
        result = subprocess.run(
            [sys.executable, "-c", command, inspect.getfile(test), test.__name__,
             json.dumps(arguments, default=os.fspath)],
            cwd=root, env=env, capture_output=True, text=True, timeout=240)
        if result.returncode == 77:
            pytest.skip(result.stdout.strip())
        assert result.returncode == 0, result.stdout + result.stderr
    return run
