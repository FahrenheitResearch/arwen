"""gpuwm switches off numpy's huge-page advice, however it is imported.

THE BREAKAGE THIS PREVENTS: on box E (THP defrag ``madvise``, most of 723 GB
in page cache) numpy's MADV_HUGEPAGE on every large array sent each
allocation into synchronous compaction: the 9 km CONUS DA controller spent
14,770 s in the kernel against 448 s of its own work in one analysis.
"""
import subprocess
import sys

import pytest


@pytest.mark.parametrize("order", ["import numpy, gpuwm", "import gpuwm, numpy"])
def test_importing_gpuwm_turns_the_advice_off_for_this_process_and_children(order):
    code = (f"{order}; import os, numpy as np; "
            "core = getattr(np, '_core', None) or np.core; "
            "print(os.environ['NUMPY_MADVISE_HUGEPAGE'], "
            "core.multiarray._get_madvise_hugepage())")
    env = {k: v for k, v in __import__("os").environ.items()
           if k != "NUMPY_MADVISE_HUGEPAGE"}
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True, check=True).stdout.split()
    assert out == ["0", "False"]


def test_a_callers_explicit_choice_is_kept():
    import os
    code = ("import gpuwm, os; print(os.environ['NUMPY_MADVISE_HUGEPAGE'])")
    env = dict(os.environ, NUMPY_MADVISE_HUGEPAGE="1")
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True, check=True).stdout.split()
    assert out == ["1"]
