"""The regime-column WRF oracle, on the device: every word, every column.

See tests/test_gf_regime_column_oracle.py for the fixture.  This file runs
``gf_gfdrv_stage`` -- the kernel gpuwm/core/gf.py launches -- on all 280
columns with fzu COMPUTED (the shipped path; nothing pinned from the
capture) and compares against WRF's own GFDRV linked with the correctly
rounded gamma WOOF ships.  Run under ``GPUWM_WRF_EXACT=1`` (strict) and
without it (default); both must give 0 differing words against their independently compiled WRF controls.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

_ROOT = Path(__file__).resolve().parents[1]
_TOOLS = _ROOT / "tools" / "gf_wrf461_oracle"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import gf_regime_columns as grc                                    # noqa: E402
import gf_regime_gpu as rg                                         # noqa: E402

FIX = Path(os.environ.get("GF_REGIME_ORACLE_DIR",
    _ROOT / "tests" / "data" / "oracles" / "gf" / "regime"))


@pytest.fixture(scope="module")
def oracle():
    return rg.load_oracle(FIX)


# ==========================================================================
# the kernel against WRF (GPU)
# ==========================================================================
@pytest.fixture(scope="module")
def runs(oracle):
    inp = oracle[0]
    return dict(faithful=rg.run_kernel(inp, 1), shipped=rg.run_kernel(inp, 0))


@pytest.mark.gpu
def test_every_output_word_is_wrfs_on_every_column(oracle, runs):
    """The claim: 0 ULP, bitwise, on all eight tendency fields, the five
    surface fields and the four integer diagnostics, on all 280 columns,
    with fzu COMPUTED (nothing pinned from the capture)."""
    res = rg.compare(runs["faithful"], rg.load_reference(FIX, "safe"))
    bad = {k: v for k, v in res["fields"].items() if v["words_differ"]}
    assert res["words"] == oracle[0]["ncol"] * (oracle[0]["nz"] * 8 + 5 + 4)
    assert res["words_differ"] == 0, bad


@pytest.mark.gpu
def test_shipped_every_word_matches_corrected_wrf(
        oracle, runs):
    """Changed k22 is no excuse: grade all fields against corrected WRF."""
    res = rg.compare(runs["shipped"], rg.load_reference(FIX, "corrected"))
    assert res["words_differ"] == 0, res


@pytest.mark.gpu
def test_216_pinned_fzu_shipped_words_match_corrected_wrf():
    from gpuwm.core.gf import _gf_module
    from test_gf_gfdrv_cuda import _launch
    fixture, want = rg.load_branch_reference()
    result = rg.compare(_launch(_gf_module(want["nz"]), fixture, 0), want)
    assert result["words"] == 216 * (40 * 8 + 5 + 4)
    assert result["words_differ"] == 0, result


@pytest.mark.gpu
def test_against_wrfs_own_gamma_the_divergence_is_the_documented_one(
        oracle, runs):
    """Against stock WRF the kernel differs exactly where the two WRF builds
    differ from each other, and nowhere else: the whole residual is gamma."""
    from gpuwm import wrf_exact
    if wrf_exact.ENABLED:
        # Strict uses the official scalar build, whose compiler flags are
        # independently recorded by record_scalar_reference.py.
        result = rg.compare(runs["faithful"], rg.load_reference(FIX, "safe"))
        assert result["words_differ"] == 0, result
        return
    _inp, cr, st, _man = oracle
    res = rg.compare(runs["faithful"], st)
    wrf_vs_wrf = rg.compare(
        dict(lev=_as_kernel_lev(cr), sca=_as_kernel_sca(cr),
             isc=_as_kernel_isc(cr)), st)
    assert res["words_differ"] > 0
    assert res["columns"] == wrf_vs_wrf["columns"]
    assert res["words_differ"] == wrf_vs_wrf["words_differ"]


def _as_kernel_lev(o):
    from gf_field_lists import DRV_LEV_FIELDS

    out = np.zeros((o["ncol"], len(DRV_LEV_FIELDS), o["nz"]), np.float32)
    for j, name in enumerate(grc.OUT_LEV):
        out[:, DRV_LEV_FIELDS.index(name), :] = o["lev"][:, j, :]
    return out


def _as_kernel_sca(o):
    from gf_field_lists import DRV_SCA_FIELDS

    out = np.zeros((o["ncol"], len(DRV_SCA_FIELDS)), np.float32)
    for j, name in enumerate(grc.OUT_SCA):
        out[:, DRV_SCA_FIELDS.index(name)] = o["sca"][:, j]
    return out


def _as_kernel_isc(o):
    from gf_field_lists import DRV_ISCA_FIELDS

    out = np.zeros((o["ncol"], len(DRV_ISCA_FIELDS)), np.int32)
    for j, name in enumerate(grc.OUT_INT):
        out[:, DRV_ISCA_FIELDS.index(name)] = o["isc"][:, j]
    return out
