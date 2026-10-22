"""Read the urban WRF v4.7.1 oracle fixtures and measure a port against them.

The fixtures are written by ``tools/urban_wrf471_oracle/oracle_io.F90``: one
directory per case, one raw little-endian ``<name>.bin`` per array and a
``MANIFEST.txt`` naming each array's kind and Fortran extents.  :func:`load`
returns ``{case: {name: ndarray}}`` with Fortran index order preserved
(``order='F'``), so a WRF ``(i, k, j)`` array is indexed ``[i, k, j]`` here
and a parity test transposes it to gpuwm's ``(k, j, i)`` in plain sight.

Fixtures live under ``tests/data/oracles/urban/<lane>/``, each with a
``PROVENANCE.md`` naming the WRF tree, the compiler and the SHA-256 of every
source that built it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from gpuwm.verify.chem_oracle import load as _load, load_case, ulp_table
from gpuwm.verify.wrf471_fixtures import require_fixture_dir

# Test data in a source checkout, not package data (gpuwm.verify.wrf471_fixtures).
ORACLE_ROOT = Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "urban"


def load(directory: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """Every case under ``directory`` (sub-directories with a MANIFEST)."""
    return _load(require_fixture_dir(directory, "urban"))


__all__ = ["ORACLE_ROOT", "load", "load_case", "ulp_table"]
