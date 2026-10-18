"""WOOF's RTE+RRTMGP against the upstream Fortran, word for word.

The referee is earth-system-radiation/rte-rrtmgp fa107a1 (the commit
gpuwm/core/rrtmgp.py transcribes) built single precision (RTE_USE_SP) with
gfortran 13.3.0 -O0 -ffp-contract=off on glibc 2.39
(tools/rrtmgp_rte_upstream_oracle/build.sh), driven by
rte_column_oracle.F90 through the same composition WOOF's production call
makes: gas optics with internal Planck sources, band cloud optics at ice
roughness 2, the McICA subcolumn mask, increment, delta scaling for the
SW, the one-angle LW solver and the two-stream SW solver, broadband fluxes.

72 columns x 60 layers (tropical convective, midlatitude summer, stable
night, polar snow and ice, ocean stratocumulus, high terrain, dry
subtropical, edge cases); the fixture carries the inputs and a SHA-256 of
every column of every oracle output, so a match here is bitwise on every
word.  Three WOOF compositions are graded: the production workspace path
(solvers fold the clouds and derive the Planck sources in registers), the
same solvers with GPUWM_RTE_FOLD=0 (partials plus reduce kernel), and the
workspace-less path (standalone Planck kernel, finalize kernels).

What makes it exact: the three units compile with --fmad=false, call
WOOF's own float32 expf/logf (rrtmgp_libm.cuh) and evaluate every
statement in the Fortran's association (rrtmgp_planck_common.cuh,
rrtmgp_gas.cu, rrtmgp_rte.cu).

Two references are held.  Against upstream with
tools/rrtmgp_rte_upstream_oracle/upstream-boundary-fixes.patch (three
interpolation fractions measured from the table node actually read) every
column of every output is equal.  Against the stock upstream, exactly the
two edge columns that reach those three boundaries differ, and they MUST:
column 63 holds a layer and the surface at 355 K, the top of the
temperature tables, where the stock Fortran reads the absorption 15 K and
the Planck function 1 K too cold; column 71 has zero ozone, so eta = 1 for
the ozone flavours, where the stock Fortran reads the eta = 7/8 node.
WOOF keeps the fixed reading (a bug is documented, not copied).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "rrtmgp_rte_upstream_oracle" / "oracle.py"
FIXTURE = ROOT / "tests" / "data" / "oracles" / "rrtmgp" / \
    "upstream-fa107a1-columns.npz"

#: The two edge columns of the fixture that reach the stock upstream's
#: boundary defects (see the module docstring): 63 (355 K) and 71 (zero
#: ozone).  Which OUTPUTS of those columns move is read off the fixture
#: itself, where the stock and the fixed builds' digests differ.
STOCK_DEFECT_COLUMNS = {63, 71}


def _tool():
    spec = importlib.util.spec_from_file_location("rrtmgp_oracle", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def graded():
    oracle = _tool()
    cols, digests = oracle.load_fixture(FIXTURE)
    woof = oracle.run_woof(cols)
    day = cols["sw_cols"]

    def mismatches(tag):
        raw = oracle.digest_mismatches(woof, digests[tag])
        return {key: ([int(day[c]) for c in cols_]
                      if key.startswith("sw_") and key != "sw_toa_src"
                      else cols_)
                for key, cols_ in raw.items()}
    # Columns where the two Fortran builds themselves disagree, per field.
    refdiff = {}
    for key in woof:
        base = key.split("@")[0]
        rows = np.flatnonzero(np.any(digests["stock"][base]
                                     != digests["fixed"][base], axis=1))
        refdiff[key] = ([int(day[c]) for c in rows]
                        if key.startswith("sw_") and key != "sw_toa_src"
                        else [int(c) for c in rows])
    return woof, mismatches("fixed"), mismatches("stock"), refdiff


@pytest.mark.gpu
@requires_gpu
@pytest.mark.skipif(not __import__("gpuwm.wrf_exact", fromlist=["ENABLED"]).ENABLED,
                    reason="WRF word-for-word oracle of the strict (GPUWM_WRF_EXACT=1) path; a default process runs 2.9.1's arithmetic")
def test_every_output_word_equals_upstream_with_boundary_fixes(graded):
    woof, fixed, _, _ = graded
    # Every WOOF output has an oracle counterpart and was graded: gas optics,
    # Planck sources, band cloud optics, finalized optics, TOA source and
    # the broadband fluxes of all three compositions.
    assert sorted(fixed) == sorted(woof)
    assert {key: cols for key, cols in fixed.items() if cols} == {}


@pytest.mark.gpu
@requires_gpu
@pytest.mark.skipif(not __import__("gpuwm.wrf_exact", fromlist=["ENABLED"]).ENABLED,
                    reason="WRF word-for-word oracle of the strict (GPUWM_WRF_EXACT=1) path; a default process runs 2.9.1's arithmetic")
def test_stock_upstream_differs_only_at_its_boundary_defects(graded):
    _, _, stock, refdiff = graded
    # WOOF differs from stock upstream in exactly the columns where stock and
    # fixed upstream differ from each other ...
    assert stock == refdiff
    # ... those are confined to the two defect columns, and both columns
    # do show the defect (the divergence is pinned, not merely allowed).
    touched = set().union(*stock.values())
    assert touched == STOCK_DEFECT_COLUMNS
    assert stock["lw_gas_tau"] == sorted(STOCK_DEFECT_COLUMNS)
