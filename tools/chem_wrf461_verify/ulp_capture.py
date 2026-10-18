"""pytest plugin: log every chem oracle comparison the parity tests make.

Load with ``-p ulp_capture`` (this folder on PYTHONPATH) and set
``CHEM_ULP_LOG`` to a JSONL path.  The plugin wraps
``gpuwm.verify.chem_oracle.ulp_table`` before any test module or helper
imports it by name, so every table a parity test asserts on is also written
out with the test that made it: one line per comparison, holding the test
node id, the array shape, ``max_ulp``, ``n_nonzero`` and ``n``.

It changes nothing a test sees: the wrapped call returns the original table.
A test that fails its 0-ULP assertion still logs the table first, which is how
the default-arithmetic run reports its real maximum instead of only "failed".
"""
from __future__ import annotations

import json
import os

import numpy as np

import gpuwm.verify.chem_oracle as _oracle

_LOG = os.environ.get("CHEM_ULP_LOG")
_CURRENT = {"test": None}
_original = _oracle.ulp_table


def _logged_ulp_table(port, ref):
    table = _original(port, ref)
    if _LOG:
        row = {"test": _CURRENT["test"], "shape": list(np.shape(ref)),
               "dtype": str(np.asarray(ref).dtype), **table}
        with open(_LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
    return table


_oracle.ulp_table = _logged_ulp_table


def pytest_runtest_setup(item):
    _CURRENT["test"] = item.nodeid
