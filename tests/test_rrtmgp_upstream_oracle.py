"""CPU gates for the RTE+RRTMGP upstream column oracle.

The device grading is tests/test_rrtmgp_upstream_oracle_gpu.py.  These
gates hold the three things that grading depends on and that a source edit
could silently take away, each named by the breakage it prevents:

* the three RRTMGP units compile without multiply-add contraction and call
  WOOF's own expf/logf, never CUDA's: a fused multiply-add or a CUDA libm
  call moves the last bits of the optics and fluxes away from the
  single-precision reference (measured before the fix on ordinary columns:
  relative differences up to 5e-6 on fluxes, 1 to 2 ULP on cloud optics);
* the SW solar source is the reference's float32 set_solar_variability,
  not a float64 evaluation rounded once (64 of the 224 g-points differed
  by 1 ULP);
* the upstream boundary patch the second referee is built with stays the
  three documented one-line changes, so "equal to the patched upstream"
  cannot quietly come to mean something else.
"""

from __future__ import annotations

import pytest

import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "gpuwm" / "core" / "kernels"
PATCH = ROOT / "tools" / "rrtmgp_rte_upstream_oracle" / \
    "upstream-boundary-fixes.patch"
FIXTURE = ROOT / "tests" / "data" / "oracles" / "rrtmgp" / \
    "upstream-fa107a1-columns.npz"

UNITS = ("rrtmgp_gas", "rrtmgp_rte", "rrtmgp_cloud")


def test_rrtmgp_units_compile_without_contraction():
    from gpuwm.core.kernels import _NO_FMAD_MODULES, _STRICT_NO_FMAD_MODULES, module_options
    for unit in UNITS:
        assert unit in _NO_FMAD_MODULES | _STRICT_NO_FMAD_MODULES
        assert "--fmad=false" in module_options(unit)


def test_rrtmgp_units_call_woof_libm_and_no_fused_intrinsic():
    from gpuwm.core.kernels import EXTRA_HEADERS
    for unit in ("rrtmgp_gas", "rrtmgp_rte"):
        assert EXTRA_HEADERS[unit][0] == "rrtmgp_libm.cuh"
    texts = [(KERNELS / f"{u}.cu").read_text(encoding="utf-8")
             for u in UNITS]
    texts.append((KERNELS / "rrtmgp_planck_common.cuh")
                 .read_text(encoding="utf-8"))
    for text in texts:
        code = re.sub(r"//[^\n]*", "", text)
        # A CUDA libm call: expf/logf/powf/exp/log not spelled rrtmgp_*.
        assert not re.search(r"(?<![A-Za-z0-9_])(?:expf|logf|powf|exp|log|"
                             r"__expf|__logf)\s*\(", code)
        # An explicit single-precision fused multiply-add.
        assert "__fmaf_rn" not in code and "fmaf(" not in code


def test_solar_source_is_the_float32_set_solar_variability():
    from netCDF4 import Dataset
    from gpuwm.core.rrtmgp import _table, load_gas_tables
    f32 = np.float32
    with Dataset(_table("rrtmgp-gas-sw-g224.nc"), "r") as nc:
        quiet = np.asarray(nc["solar_source_quiet"][:], f32)
        facular = np.asarray(nc["solar_source_facular"][:], f32)
        sunspot = np.asarray(nc["solar_source_sunspot"][:], f32)
        mg = f32(nc["mg_default"].getValue())
        sb = f32(nc["sb_default"].getValue())
    want = np.empty_like(quiet)
    for g in range(quiet.size):   # the Fortran loop, one g-point at a time
        a = f32(quiet[g] + f32(f32(mg - f32(0.1495954)) * facular[g]))
        want[g] = f32(a + f32(f32(sb - f32(0.00066696)) * sunspot[g]))
    got = load_gas_tables("sw").solar_source.astype(f32)
    assert got.tobytes() == want.tobytes()


def test_solar_source_equals_the_upstream_toa_source_digest():
    import hashlib
    from gpuwm.core.rrtmgp import load_gas_tables
    raw = np.load(FIXTURE)
    got = load_gas_tables("sw").solar_source.astype(np.float32)
    digest = np.frombuffer(hashlib.sha256(got.tobytes()).digest(), np.uint8)
    assert (raw["digest_stock__sw_toa_src"][0] == digest).all()


def test_upstream_boundary_patch_is_the_three_documented_lines():
    text = PATCH.read_text(encoding="utf-8")
    removed = [l[1:].strip() for l in text.splitlines()
               if l.startswith("-") and not l.startswith("---")]
    added = [l[1:].strip() for l in text.splitlines()
             if l.startswith("+") and not l.startswith("+++")]
    assert removed == [
        "ftemp(icol,ilay) = (tlay(icol,ilay) - temp_ref(jtemp_)) * "
        "temp_ref_delta_inv",
        "feta = loceta - aint(loceta)",
        "frac = val0 - aint(val0) ! get fractional part",
    ]
    assert [a.split("!")[0].strip() for a in added] == [
        "ftemp(icol,ilay) = (tlay(icol,ilay) - temp_ref(jtemp(icol,ilay)))"
        " * temp_ref_delta_inv",
        "feta = loceta - real(jeta(itemp,icol,ilay,iflav)-1, wp)",
        "frac = val0 - real(min(size(table,dim=1)-1, max(1, int(val0)+1))"
        "-1, wp)",
    ]


@pytest.fixture(autouse=True)
def _strict_source_contract(monkeypatch):
    import gpuwm.wrf_exact as exact
    monkeypatch.setattr(exact, "ENABLED", True)
