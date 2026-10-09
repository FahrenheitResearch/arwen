"""Max FP32 ULP distance of every CUDA MYNN leaf and driver from WRF v4.6.1.

Run from a checkout with a GPU (``PYTHONPATH=<checkout>/tests:<checkout>``)::

    python tools/mynn_pbl_wrf461_oracle/ulp_census.py [--out census.json]

Every number is measured against the committed oracle CSVs built from the
unmodified WRF v4.6.1 ``module_bl_mynn.F`` (gfortran 13.3.0 -O0, glibc 2.39);
nothing here reads a budget.  The leaf calls mirror the GPU tests that gate
them (tests/test_mynn_pbl_gpu.py, tests/test_mynn_mixlength2.py,
tests/test_mynn_wrf461_exact_gpu.py), so a census row and its gate read the
same arrays.  The families rows integrate the six column families of
``driver-families{,2}.csv.gz`` both step by step from the recorded inputs
(``replay``) and free-running from the cold start (``free``).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def _ulp(got, want) -> int:
    from gpuwm.core.fp32_ulp import fp32_ulp_distance
    got = np.asarray(got, dtype=np.float32)
    want = np.asarray(want, dtype=np.float32)
    return int(fp32_ulp_distance(got.reshape(want.shape), want).max())


def leaves() -> dict[str, dict[str, int]]:
    import cupy as cp
    import test_mynn_pbl as T
    from gpuwm.core import mynn_pbl_gpu as G
    from gpuwm.core.mynn_pbl import MYNN_LEVEL2_INPUTS, MYNN_LEVEL2_OUTPUTS

    host = cp.asnumpy
    out: dict[str, dict[str, int]] = {}

    # mym_level2 ------------------------------------------------------------
    import test_mynn_pbl_gpu as TG
    _, fields = TG._oracle_fields()
    actual = G.mynn_level2_pairs_cuda(
        {name: cp.asarray(fields[name]) for name in MYNN_LEVEL2_INPUTS})
    out["level2"] = {name: _ulp(host(getattr(actual, name)), fields[name])
                     for name in MYNN_LEVEL2_OUTPUTS}

    # get_pblh + scale_aware -------------------------------------------------
    _, fields, kzi = T._pblh_oracle()
    zw = np.concatenate((fields["zw"][:, :1], fields["zw_next"]), axis=1)
    actual = G.mynn_pblh_scale_columns_cuda(
        cp.asarray(fields["thetav"]), cp.asarray(fields["qke"]),
        cp.asarray(zw), cp.asarray(fields["dz"]),
        cp.asarray(fields["landsea"][:, 0]), cp.asarray(fields["dx"][:, 0]))
    out["pblh"] = {name: _ulp(host(getattr(actual, name)), fields[name][:, 0])
                   for name in ("zi", "psig_bl", "psig_shcu")}
    out["pblh"]["kzi_mismatch"] = int(
        np.count_nonzero(host(actual.kzi) != kzi[:, 0]))

    # mym_length, option 1 ---------------------------------------------------
    _, fields = T._mixlength_oracle()
    cols = ("dz", "zw", "u", "v", "qke", "dtv", "theta", "vt", "vq",
            "cldfra", "edmf_w", "edmf_a")
    scal = ("xland", "dx", "rmo", "flt", "fltv", "flq", "zi", "psig_bl")
    inputs = {n: cp.asarray(fields[n]) for n in cols}
    inputs.update({n: cp.asarray(fields[n][:, 0]) for n in scal})
    actual = G.mynn_mixlength_default_cuda(inputs)
    out["mixlength1"] = {n: _ulp(host(getattr(actual, n)), fields[n])
                         for n in ("el", "qkw")}

    # mym_turbulence, option 1 ------------------------------------------------
    _, fields = T._turbulence_oracle()
    arrays = ("dz", "zw", "u", "v", "thl", "thetav", "ql", "qw", "qke",
              "tsq", "qsq", "cov", "vt", "vq", "theta", "cldfra",
              "edmf_w", "edmf_a", "tkeprodtd")
    scal = ("xland", "dx", "rmo", "flt", "fltv", "flq", "zi", "psig_bl",
            "psig_shcu")
    inputs = {n: cp.asarray(fields[n]) for n in arrays}
    inputs.update({n: cp.asarray(fields[n][:, 0]) for n in scal})
    actual = G.mynn_turbulence_default_cuda(inputs)
    out["turbulence1"] = {
        n: _ulp(host(getattr(actual, n)), fields[n])
        for n in ("dfm", "dfh", "dfq", "tcd", "qcd", "pdk", "pdt", "pdq",
                  "pdc", "el", "sm", "sh")}

    # mym_predict ----------------------------------------------------------
    _, fields = T._predict_oracle()
    inputs = {n: cp.asarray(fields[n]) for n in (
        "dz", "rho", "dfq", "pdk", "pdt", "pdq", "pdc", "el", "s_aw",
        "s_awqke")}
    inputs.update({n: cp.asarray(fields[n][:, 0])
                   for n in ("ust", "flt", "flq", "pmz", "phh", "delt")})
    inputs.update({n: cp.asarray(fields[f"{n}_before"])
                   for n in ("qke", "tsq", "qsq", "cov")})
    actual = G.mynn_predict_default_cuda(inputs)
    out["predict"] = {n: _ulp(host(getattr(actual, n)), fields[f"{n}_after"])
                      for n in ("qke", "tsq", "qsq", "cov")}

    # mym_condensation ---------------------------------------------------
    _, fields = T._condensation_oracle()
    actual = G.mynn_condensation_default_cuda(
        {n: cp.asarray(v) for n, v in T._condensation_inputs(fields).items()})
    out["condensation"] = {}
    for n in ("qc_bl", "qi_bl", "cldfra", "vt", "vq", "sgm"):
        rec = f"{n}_after" if n in ("vt", "vq", "sgm") else n
        out["condensation"][n] = _ulp(host(getattr(actual, n)), fields[rec])

    # mynn_tendencies, no mass flux and with mass flux ------------------------
    _, fields = T._tendencies_oracle()
    actual = G.mynn_tendencies_nomf_cuda(
        {n: cp.asarray(v) for n, v in T._tendencies_inputs(fields).items()})
    out["tendencies_nomf"] = {
        n: _ulp(host(getattr(actual, n)), fields[n])
        for n in ("du", "dv", "dth", "dqv", "dqc", "dqi", "dqs")}
    out["tendencies_nomf"]["thl"] = _ulp(host(actual.thl),
                                         fields["thl_after"])
    _, fields = T._tendencies_mf_oracle()
    worst: dict[str, int] = {}
    for index, case in enumerate(T.TENDENCIES_MF_CASES):
        inputs, mom = T._tendencies_mf_inputs(fields, case)
        actual = G.mynn_tendencies_default_cuda(
            {n: cp.asarray(v) for n, v in inputs.items()},
            bl_mynn_edmf_mom=mom)
        for n in ("du", "dv", "dth", "dqv", "dqc", "dqi", "dqs"):
            worst[n] = max(worst.get(n, 0), _ulp(
                host(getattr(actual, n))[0], fields[n][index]))
    out["tendencies_mf"] = worst

    # mym_initialize -------------------------------------------------------
    from gpuwm.core.mynn_pbl import MYNN_INITIALIZE_OUTPUTS
    _, fields = T._initialize_oracle()
    worst = {}
    for case, _name in enumerate(T.INITIALIZE_CASES):
        actual = G.mynn_initialize_default_cuda(
            {k: cp.asarray(v)
             for k, v in T._initialize_inputs(fields, case).items()},
            initialize_qke=T._initialize_flag(fields, case))
        for n in MYNN_INITIALIZE_OUTPUTS:
            worst[n] = max(worst.get(n, 0), _ulp(
                host(getattr(actual, n))[0], fields[n][case]))
    out["initialize1"] = worst

    # DMP_mf ----------------------------------------------------------------
    rows = T._dmp_mf_oracle()
    worst = {}
    for case in T.DMP_MF_CASES:
        values = T._dmp_mf_inputs(T._dmp_mf_case(rows, case))
        actual = G.mynn_dmp_mf_cuda({n: cp.asarray(v)
                                     for n, v in values.items()})
        for n, want in T._dmp_mf_expected(T._dmp_mf_case(rows, case)).items():
            got = host(getattr(actual, n))
            got = got[0] if got.ndim == 2 else got
            want = np.asarray(want)
            if want.dtype.kind in "iu":
                worst[n] = max(worst.get(n, 0), int(np.abs(
                    np.asarray(got).astype(np.int64) - want).max()))
            else:
                worst[n] = max(worst.get(n, 0), _ulp(got, want))
    out["dmp_mf"] = worst
    return out


def drivers() -> dict[str, dict[str, object]]:
    import cupy as cp
    import test_mynn_pbl as T
    import _mynn_families as F
    from gpuwm.core.mynn_pbl_gpu import mynn_bl_driver_cuda

    def driver(values, **kwargs):
        return mynn_bl_driver_cuda(
            {n: cp.asarray(np.ascontiguousarray(v)) for n, v in values.items()},
            **kwargs)

    out: dict[str, dict[str, object]] = {}
    for mixlength, csv_name in ((1, "driver.csv"), (2, "driver2.csv")):
        T.DRIVER_ORACLE = T.ORACLE.with_name(csv_name)
        for step in (1, 2):
            blocks, values, initflag, delt = T._driver_step(step)
            actual = driver(values, initflag=initflag, delt=delt,
                            flag_qs=True, bl_mynn_mixlength=mixlength)
            row = {}
            for n in T.DRIVER_PROFILE_OUTPUTS:
                key = T.DRIVER_OUTPUT_CSV.get(n, n)
                want = np.asarray([[np.float32(r[key]) for r in b]
                                   for b in blocks])
                row[n] = _ulp(cp.asnumpy(actual[n]), want)
            for n in ("pblh", "rmol", "maxwidth", "maxmf", "ztop_plume"):
                want = np.asarray([np.float32(b[0][n]) for b in blocks])
                row[n] = _ulp(cp.asnumpy(actual[n]).reshape(-1), want)
            out[f"driver_ml{mixlength}_step{step}"] = row
    for mixlength in (1, 2):
        for replay in (True, False):
            table = F.max_ulp_table(F.integrate(
                driver, mixlength, replay=replay, to_host=cp.asnumpy))
            out[f"families_ml{mixlength}_{'replay' if replay else 'free'}"] = {
                name: {"max": max(per_step), "per_step": per_step}
                for name, per_step in table.items()}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out")
    parser.add_argument("--only", choices=("leaves", "drivers"))
    args = parser.parse_args(argv)
    import cupy as cp
    census = {
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "compute_capability": cp.cuda.Device().compute_capability,
        "nvrtc": list(cp.cuda.nvrtc.getVersion()),
    }
    import traceback
    for part, run in (("leaves", leaves), ("drivers", drivers)):
        if args.only in (None, part):
            try:
                census[part] = run()
            except Exception:  # a census reports what it could not measure
                census[part] = {"error": traceback.format_exc()}
    text = json.dumps(census, indent=1, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as stream:
            stream.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
