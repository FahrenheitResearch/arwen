"""Measure the CUDA gsd_41 MYNN driver against the 84-column fork oracle.

Runs every configuration of tests/test_mynn_gsd41_columns_exact_gpu.py and
writes one JSON document: per configuration and output, the max ULP, the max
absolute difference, the number of float32 values whose bits differ, and
the worst column and step.  Run it once with GPUWM_WRF_EXACT=1 (the strict
build) and once without (default arithmetic).

usage: python columns_report.py OUT.json [ORACLE_DIR]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import numpy as np  # noqa: E402

import _mynn_columns_gsd41 as C  # noqa: E402


def main(out: str, oracle_dir: str | None = None) -> int:
    import cupy as cp
    from gpuwm.core.mynn_pbl_gpu import mynn_bl_driver_cuda

    if oracle_dir:
        C.ORACLE_DIR = Path(oracle_dir)
    names = C.column_names()

    def driver(values, **kwargs):
        return mynn_bl_driver_cuda(
            {k: cp.asarray(np.ascontiguousarray(v)) for k, v in values.items()},
            **kwargs)

    configs = []
    for asis in (False, True):
        for replay in (True, False):
            for snow in (True, False):
                configs.append(dict(asis=asis, replay=replay, snow=snow,
                                    bl_mynn_cloud_tendency_form="gsd_41"))
    for replay in (True, False):
        configs.append(dict(asis=False, replay=replay, snow=True))
    report = {
        "GPUWM_WRF_EXACT": os.environ.get("GPUWM_WRF_EXACT", ""),
        "device": cp.cuda.runtime.getDeviceProperties(
            cp.cuda.Device().id)["name"].decode(),
        "ncol": len(names), "columns": names, "configs": [],
    }
    oracles = {asis: C.load(asis) for asis in (False, True)}
    for cfg in configs:
        kwargs = dict(cfg)
        asis = kwargs.pop("asis")
        results = C.integrate(driver, asis=asis, to_host=cp.asnumpy,
                              oracle=oracles[asis], **kwargs)
        table = C.compare(results, names)
        report["configs"].append({
            "config": {**cfg, "bl_mynn_cloud_tendency_form":
                       cfg.get("bl_mynn_cloud_tendency_form", "wrf_461")},
            "max_ulp": max(row["ulp"] for row in table.values()),
            "bits_differing": sum(row["bits"] for row in table.values()),
            "values": sum(row["values"] for row in table.values()),
            "outputs": table,
        })
        print(json.dumps({k: v for k, v in report["configs"][-1].items()
                          if k != "outputs"}))
    Path(out).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
