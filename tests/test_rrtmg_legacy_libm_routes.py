"""CPU pins of the two libm routes the legacy RRTMG column oracle fixed.

tools/rrtmg_legacy_column_oracle found WOOF's legacy RRTMG call off WRF
v4.6.1 in two places, both transcendental calls on REAL(4):

* COSZEN, DECLIN and SOLCON came from NumPy's float32 sin/cos/arcsin on the
  host (rrtmg_legacy.radconst / calc_coszen): COSZEN up to 64 ULP off on 16
  of 128 columns, and every daytime shortwave output with it.  The adapter
  now takes them from kernels/rrtmg_legacy_solar.cu (WOOF's own float32
  sinf/cosf/asinf, explicit single-rounded arithmetic).
* cal_cldfra1 used CuPy's cp.exp / cp.power (CUDA builtins): CLDFRA up to
  2,113 ULP off on 123 to 261 values per regime, and the longwave and
  shortwave through it.  It now calls WOOF's own float32 expf / powf.

The device results are graded bitwise by
tests/test_rrtmg_legacy_column_oracle.py (GPU shard); these pins keep the
routes from regressing on a CPU leg, where that file skips.
"""
from __future__ import annotations

import inspect
from pathlib import Path

from gpuwm.core import rrtmg_legacy, rrtmgp
from gpuwm.core.kernels import EXTRA_HEADERS

KERNELS = Path(rrtmg_legacy.__file__).resolve().parent / "kernels"


def test_adapter_takes_its_sun_from_the_device_solar_unit():
    source = inspect.getsource(rrtmg_legacy.RRTMGLegacyRadiation.__call__)
    assert "solar_geometry_device(" in source
    import ast, textwrap
    # The strict wrf_461 arm (`if ENABLED and generation == "wrf_461"`)
    # takes this unit's device sun; the other arms are 2.9.1's solar
    # generations: wrf_480 on gpuwm.core.solar's device unit, and the
    # default-arithmetic wrf_461 on the NumPy radconst/calc_coszen twins.
    branches=[n for n in ast.walk(ast.parse(textwrap.dedent(source)))
              if isinstance(n,ast.If) and "ENABLED" in ast.unparse(n.test)
              and "wrf_461" in ast.unparse(n.test)]
    branches=[n for n in branches if "solar_geometry_device(" in ast.unparse(n)]
    assert len(branches)==1
    strict=ast.unparse(ast.Module(body=branches[0].body,type_ignores=[]))
    default=ast.unparse(ast.Module(body=branches[0].orelse,type_ignores=[]))
    assert "solar_geometry_device(" in strict
    assert "calc_coszen(" not in strict and "radconst(" not in strict
    assert "calc_coszen(" in default and "radconst(" in default


def test_solar_unit_uses_woof_float32_trig_and_wrf_461_clamp():
    assert EXTRA_HEADERS["rrtmg_legacy_solar"] == (
        "glibc_flt32.cuh", "glibc_trig_flt32.cuh")
    text = (KERNELS / "rrtmg_legacy_solar.cu").read_text(encoding="utf-8")
    code = "\n".join(line.split("//")[0] for line in text.splitlines())
    for name in ("glibc_sinf(", "glibc_cosf(", "glibc_asinf("):
        assert name in code
    for builtin in (" sinf(", " cosf(", " asinf(", "__sinf", "__cosf"):
        assert builtin not in code
    # calc_coszen's 4.6.1 clamp, min(max(coszen, -1.0), 1.0)
    assert "fminf(fmaxf(c, -1.0f), 1.0f)" in code


def test_cal_cldfra1_uses_woof_float32_expf_and_powf():
    source = inspect.getsource(rrtmgp.cal_cldfra1)
    code = "\n".join(line.split("#")[0] for line in source.splitlines())
    assert "slab_expf(" in code and "slab_powf(" in code
    assert "cp.exp(" not in code and "cp.power(" not in code
