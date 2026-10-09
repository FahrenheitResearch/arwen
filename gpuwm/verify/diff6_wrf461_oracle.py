"""Launch the production sixth-order row on WRF v4.6.1 column-oracle cases.

The cases are tools/wrf_diffusion_oracle/diff6_wrf461_cases.py's: compiled,
byte-unmodified WRF v4.6.1 sixth_order_diffusion on real WRF states.  This
module rebuilds each case's RunConfig and runs ``dycore.add_diff6_row``, the
row ``prepare_fixed_tendencies`` runs, on the case's exact float32 inputs.
"""
from __future__ import annotations

import numpy as np


_DIFF6_DRY_SLOT = {"U": "smag_ru", "V": "smag_rv", "W": "smag_rw", "T": "smag_rth"}
_DIFF6_STAGGER = {"u": "x", "v": "y", "w": "z", "m": ""}


def diff6_wrf461_config(case):
    """The RunConfig a WRF 4.6.1 column-oracle case describes.

    dt is formed as gpuwm forms a root domain's dt from the namelist
    rational (float32 time_step + num/den, experiment.py), dx/dy as the
    REAL values WRF holds.
    """
    from gpuwm.config import RunConfig
    nz, ny, nx = case["dims"]
    mode = case["mode_name"]
    dt = float(np.float32(case["time_step"])
               + np.float32(case["num"]) / np.float32(case["den"]))
    return RunConfig(nx=nx, ny=ny, nz=nz, dx=float(np.float32(case["dx"])),
                     dy=float(np.float32(case["dy"])), ztop=20000.0, dt=dt,
                     run_seconds=dt, diff_6th_opt=case["opt"],
                     diff_6th_factor=case["factor"],
                     diff_6th_slopeopt=case["slopeopt"],
                     diff_6th_thresh=case["thresh"],
                     open_x=mode in ("open", "open_x"),
                     open_y=mode in ("open", "open_y"),
                     specified=mode == "specified", nested=mode == "nested",
                     diff_6th_form="wrf_461")


def launch_diff6_wrf461_case(case, values, *, legacy_composition=False,
                             scalar_dt_mutation=False):
    """Run the production diff6 row (``dycore.add_diff6_row``) on one case.

    The carrying slot starts as the case's incoming ``tendency``.
    ``legacy_composition`` reproduces the pre-fix product (increment formed
    on a zeroed temporary, then added); ``scalar_dt_mutation`` hands a
    scalar row the dry rows' dt.  Both exist to prove the oracle
    discriminates them.
    """
    import cupy as cp
    from gpuwm.core import dycore

    cfg = diff6_wrf461_config(case)
    slot = _DIFF6_DRY_SLOT.get(case["var"], "smag_r" + case["var"].lower())
    if scalar_dt_mutation:
        slot = "smag_rth"
    dev = {k: cp.asarray(v) for k, v in values.items()
           if k not in ("reference", "latitude")}
    target = dev["tendency"].copy()
    maps = dict(phb=dev["phb"], msfu=dev["MAPFAC_UX"], msfv=dev["MAPFAC_VY"],
                msft=dev["MAPFAC_MX"])
    args = (dev["field"],)
    rest = (dev["mut"], dev["c1"], dev["c2"], _DIFF6_STAGGER[case["stagger"]])
    if legacy_composition:
        tmp = cp.zeros_like(target)
        dycore.add_diff6_row(cfg, slot, *args, tmp, *rest, **maps)
        target += tmp
    else:
        dycore.add_diff6_row(cfg, slot, *args, target, *rest, **maps)
    return target.get()
