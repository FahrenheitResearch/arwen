"""Importing the Grell-Freitas module must not import CuPy.

The memory pricing in ``gpuwm.core.preflight`` reads GF's tile constants
and workspace size on hosts with no GPU runtime (``gpuwm check`` sizing a
configuration for another machine).  The native-result receipt the driver
admits is built on first use for exactly this reason; this test is what
breaks if it moves back to import time.
"""

from __future__ import annotations

import subprocess
import sys


def test_importing_gf_leaves_cupy_unimported():
    code = (
        "import sys\n"
        "import gpuwm.core.gf as gf\n"
        "gf.gf_workspace_floats(50, 1024)\n"
        "assert 'cupy' not in sys.modules, 'gpuwm.core.gf imported cupy'\n"
    )
    done = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr


def test_gf_workspace_pricing_covers_the_larger_driver_tile_without_cupy():
    code = (
        "import sys\n"
        "from types import SimpleNamespace as NS\n"
        "from gpuwm.core import gf, preflight\n"
        "run = NS(nx=250, ny=200, nz=50, cu_physics=3)\n"
        "exp = NS(domains=[NS(run=run)])\n"
        "profile = NS(multiprocessor_count=128)\n"
        "priced = preflight.gf_column_workspace_bytes(exp, profile=profile)\n"
        "assert priced == gf.gf_workspace_floats(50, 50000) * 4\n"
        "assert priced > gf.gf_workspace_floats(50, 32768) * 4\n"
        "assert 'cupy' not in sys.modules, 'GF workspace pricing imported cupy'\n"
    )
    done = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
