"""Column water budgets of the gsd_41 MYNN driver on the 84-column oracle.

Replays the twelve recorded steps through the CUDA driver under both cloud
tendency forms and writes, per column, the worst-step residuals (kg m-2 s-1)

* mass: sum(rho_d*dz*(RQVBLTEN+RQCBLTEN+RQIBLTEN)) - QFX, with
  rho_d = rho/(1+qv+qc+qi): the water mass the column gains beyond its
  surface flux;
* dz:   sum(dz*(1-sqv)*(RQVBLTEN+RQCBLTEN+RQIBLTEN))*rho(1) - QFX: what a
  density-free solve (the fork's P11 transport) conserves;

and the number of level-steps whose condensate a 20 s step of the tendency
would take negative.  docs/dev/mynn-gsd41.md "Water budget" quotes it.

usage: python columns_budget.py OUT.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import numpy as np  # noqa: E402

import _mynn_columns_gsd41 as C  # noqa: E402


def main(out: str) -> int:
    import cupy as cp
    from gpuwm.core.mynn_pbl_gpu import mynn_bl_driver_cuda

    oracle, names = C.load(False), C.column_names()

    def driver(values, **kwargs):
        return mynn_bl_driver_cuda(
            {k: cp.asarray(np.ascontiguousarray(v)) for k, v in values.items()},
            **kwargs)

    report = {}
    for form in ("gsd_41", "wrf_461"):
        results = C.integrate(driver, asis=False, replay=True,
                              to_host=cp.asnumpy, oracle=oracle,
                              bl_mynn_cloud_tendency_form=form)
        mass, dzw, neg = [], [], []
        for (_, actual, _), (_, inp, _) in zip(results, oracle["steps"]):
            qv, qc, qi = (inp[n].astype(np.float64) for n in ("qv", "qc", "qi"))
            rhod = inp["rho"] / (1.0 + qv + qc + qi)
            dz = inp["dz"].astype(np.float64)
            a = {n: np.asarray(actual[n], np.float64).reshape(qv.shape)
                 for n in ("rqvblten", "rqcblten", "rqiblten")}
            total = a["rqvblten"] + a["rqcblten"] + a["rqiblten"]
            qfx = inp["qfx"].astype(np.float64)
            mass.append((rhod * dz * total).sum(1) - qfx)
            dzw.append((dz * total * (1.0 - qv / (1.0 + qv))).sum(1)
                       * inp["rho"][:, 0] - qfx)
            delt = float(oracle["delt"])
            neg.append(((qc + delt * a["rqcblten"]) < 0).sum(1)
                       + ((qi + delt * a["rqiblten"]) < 0).sum(1))
        mass, dzw, neg = np.array(mass), np.array(dzw), np.array(neg)
        report[form] = {
            name: {"mass": float(mass[np.abs(mass[:, j]).argmax(), j]),
                   "dz": float(dzw[np.abs(dzw[:, j]).argmax(), j]),
                   "negative_condensate_levels": int(neg[:, j].sum()),
                   "qfx": float(oracle["scal"][j, C.SCALARS.index("qfx")])}
            for j, name in enumerate(names)}
    Path(out).write_text(json.dumps(report, indent=1))
    for name in names:
        g, w = report["gsd_41"][name], report["wrf_461"][name]
        print(f"{name:36s} qfx {g['qfx']:.2e} | gsd_41 mass {g['mass']:+.2e} "
              f"dz {g['dz']:+.2e} | wrf_461 mass {w['mass']:+.2e} "
              f"dz {w['dz']:+.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
