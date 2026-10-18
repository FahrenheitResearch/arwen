"""The CUDA gsd_41 MYNN driver against the fork's own driver, 84 columns.

``bl_mynn_version = "gsd_41"`` against NOAA-EMC/HRRR v4.1.21
``module_bl_mynn.F`` ``mynn_bl_driver`` (gfortran 13.3.0 -O0, glibc 2.39),
HRRR's namelist identity, over twenty-one regimes x four variants (convective,
desert, stable and very stable nights, marine stratocumulus, shallow
cumulus, tropical ocean, cold-air outbreak, cold pool with fog and ice,
arctic snowpack, mixed-phase cloud, high terrain day and night, high wind,
neutral with zero fluxes, saturated fog, capped mixed layer, deep moist
tropics, a very dry column, cirrus, subnormal trace condensate and snow;
dx 750 m to 25 km), twelve 20 s steps
(tests/_mynn_columns_gsd41.py, tools/mynn_pbl_gsd41_oracle/build_columns.sh).

* With the fork's own forms every output of every step is the fork's float32
  bits, step by step from the recorded inputs and free-running from the cold
  start, with the HRRR door's snow handed in under ``flag_qs = True`` (the
  fork has no snow argument, so snow must change nothing).
* The default ``bl_mynn_cloud_tendency_form = "wrf_461"`` may move only the
  theta and vapour tendencies (the P13/P22 water-conservation fix); its
  cloud-water tendency keeps the fork's values, some exact zeros with the
  other sign.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu

import _mynn_columns_gsd41 as C

DEFECT_FIX_OUTPUTS = {"rthblten", "rqvblten"}


def _device_driver():
    import cupy as cp
    from gpuwm.core.mynn_pbl_gpu import mynn_bl_driver_cuda

    seen = []

    def driver(values, **kwargs):
        out = mynn_bl_driver_cuda(
            {name: cp.asarray(np.ascontiguousarray(value))
             for name, value in values.items()}, **kwargs)
        seen.append(cp.asnumpy(out["rqsblten"]))
        return out
    return driver, cp.asnumpy, seen


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("snow", (True, False), ids=("flag_qs", "no_snow"))
@pytest.mark.parametrize("asis", (False, True), ids=("squared", "as_written"))
@pytest.mark.parametrize("replay", (True, False), ids=("replay", "free"))
def test_fork_forms_are_bitwise_the_fork_driver(asis, replay, snow):
    driver, host, seen = _device_driver()
    results = C.integrate(driver, asis=asis, replay=replay, to_host=host,
                          snow=snow, bl_mynn_cloud_tendency_form="gsd_41")
    table = C.compare(results, C.column_names())
    assert {name: row for name, row in table.items() if row["bits"]} == {}, \
        table
    # the fork returns no snow tendency, and neither does the port
    assert all(not np.any(tend) for tend in seen)


@pytest.mark.gpu
@requires_gpu
def test_default_tendency_form_differs_from_the_fork_only_in_its_fix():
    driver, host, _ = _device_driver()
    table = C.compare(C.integrate(driver, asis=False, replay=True,
                                  to_host=host))
    # By value (ULP): the fix's own cloud-water arithmetic writes some exact
    # zeros of rqcblten with the other sign (0 ULP, different bits).
    moved = {name for name, row in table.items() if row["ulp"]}
    assert moved <= DEFECT_FIX_OUTPUTS, table
    assert moved, table
