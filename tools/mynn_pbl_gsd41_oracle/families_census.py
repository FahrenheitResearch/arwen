"""Max FP32 ULP of the CUDA gsd_41 MYNN driver against the fork's driver.

Reads gpuwm/data/mynn/oracle/driver-families-gsd41{,-asis}.csv.gz through
tests/_mynn_families_gsd41.py, for both cloud-tendency forms and both modes
(replay, free).  PYTHONPATH=<checkout>/tests:<checkout>.
"""
import json
import sys

import numpy as np


def main():
    import cupy as cp
    import _mynn_families_gsd41 as G
    from gpuwm.core.mynn_pbl_gpu import mynn_bl_driver_cuda

    def driver(values, **kwargs):
        return mynn_bl_driver_cuda(
            {n: cp.asarray(np.ascontiguousarray(v)) for n, v in values.items()},
            **kwargs)

    out = {}
    for asis in (False, True):
        for form in ("wrf_461", "gsd_41"):
            for replay in (True, False):
                key = (f"{'asis' if asis else 'sq'}_{form}_"
                       f"{'replay' if replay else 'free'}")
                try:
                    table = G.max_ulp_table(G.integrate(
                        driver, asis=asis, replay=replay, to_host=cp.asnumpy,
                        bl_mynn_cloud_tendency_form=form))
                    out[key] = {n: {"max": max(s), "per_step": s}
                                for n, s in table.items()}
                except Exception as error:  # report, keep measuring
                    out[key] = {"error": repr(error)}
                print(key, {n: v["max"] for n, v in out[key].items()}
                      if "error" not in out[key] else out[key], flush=True)
    if len(sys.argv) > 1:
        json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
