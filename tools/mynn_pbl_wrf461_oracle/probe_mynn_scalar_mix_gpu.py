#!/usr/bin/env python3
"""Gate the GPU mixscalars TU against the CPU reference, every fixture case.

W4 full-admission lane (GPU half).  The CPU reference module
``gpuwm.core.mynn_scalar_mix`` is bit-exact vs the anchored gfortran
fixtures (``w4-oracle-fixtures``, validators in this
directory), so it is the oracle-of-record here: the two kernels of the new
``kernels/mynn_scalar_mix.cu`` are compared against it column-for-column
over EVERY fixture case, all five species each.

Three-way closure per case:
  * flux arm  (dmp-mf-mixscalars.csv, 12 cases): CPU ``dmp_qn_flux_column``
    on the ``_dmp_mf_column`` plume-edge exports, asserted bit-equal to the
    fixture s_awqn* columns, then GPU ``mynn_dmp_qn_flux_columns`` on the
    SAME exports, asserted bit-equal to the CPU reference.
  * solve arm (tendencies-mf-mixscalars.csv, 9 cases): the harness rebuilds
    dtz/rhoinv/khdz/hdz/dzinv with the reference construction, feeds CPU
    ``mix_scalar_column``, asserts its dqn bit-equal to the fixture dqn*
    columns (which proves the rebuild), then GPU ``mynn_mix_scalar_columns``
    on the PRIMITIVE inputs, asserted bit-equal to the CPU qn2 AND dqn.

Emits the per-solve ulp table (max |ulp| GPU vs CPU per case/species) even
though the gate is exact-or-fail, so the report can quote it verbatim.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys

import numpy as np

from gpuwm.core.fp32_ulp import monotone_fp32_key
from gpuwm.core.mynn_pbl import (
    MYNN_DMP_MF_COLUMN_INPUTS,
    MYNN_DMP_MF_QN_COLUMN_INPUTS,
    MYNN_DMP_MF_SCALAR_INPUTS,
    MYNN_TENDENCIES_INTERFACE_INPUTS,
    MYNN_TENDENCIES_LAYER_INPUTS,
    MYNN_TENDENCIES_QN_INTERFACE_INPUTS,
    MYNN_TENDENCIES_QN_LAYER_INPUTS,
    MYNN_TENDENCIES_SCALAR_INPUTS,
    _dmp_mf_column,
)
from gpuwm.core.mynn_scalar_mix import (
    QN_SOLVE_ORDER,
    dmp_qn_flux_column,
    mix_scalar_column,
)
from gpuwm.core.mynn_scalar_mix_gpu import (
    mynn_dmp_qn_flux_columns_cuda,
    mynn_mix_scalar_columns_cuda,
)

F = np.float32
NZ = 30
DMP_CASES = (
    "land_dry", "land_cumulus", "water_cumulus", "stable_off",
    "resolved_w", "flux_limited", "stochastic", "high_wind_thin",
    "deep_pblh", "cloud_base_capped", "fine_grid", "dead_probe",
)
TEND_CASES = (
    "land_cumulus", "water_cumulus", "deep_plume", "fine_grid",
    "momentum_off", "momentum_off_probe", "downdraft_probe",
    "moisture_repair", "subsidence_probe",
)
DMP_RENAMED = {
    "qc_bl": "qc_bl_before", "cldfra_bl": "cldfra_bl_before",
    "vt": "vt_before", "vq": "vq_before",
}
TEND_RENAMED = {
    "thl": "thl_before", "sqv": "sqv_before", "sqc": "sqc_before",
    "sqi": "sqi_before", "sqs": "sqs_before",
}


def _ulp(got: np.ndarray, expected: np.ndarray) -> int:
    return int(np.abs(monotone_fp32_key(got)
                      - monotone_fp32_key(expected)).max(initial=0))


def _rows(path: Path, cases: tuple[str, ...]) -> dict[str, list[dict]]:
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(cases) * NZ:
        raise SystemExit(f"{path}: expected {len(cases) * NZ} rows")
    out = {case: [row for row in rows if row["case"] == case]
           for case in cases}
    for case, selected in out.items():
        if len(selected) != NZ:
            raise SystemExit(f"{case}: {len(selected)} levels")
    return out


def _col(selected, name, rename=None):
    key = (rename or {}).get(name, name)
    return np.asarray([np.float32(row[key]) for row in selected],
                      dtype=np.float32)


def _iface(selected, name):
    return np.asarray(
        [*[np.float32(row[name]) for row in selected],
         np.float32(selected[-1][f"{name}_next"])], dtype=np.float32)


def flux_arm(path: Path) -> dict:
    summary: dict[str, dict[str, int]] = {}
    for case, selected in _rows(path, DMP_CASES).items():
        args = [
            _col(selected, name, DMP_RENAMED)
            for name in MYNN_DMP_MF_COLUMN_INPUTS
        ]
        args.append(_iface(selected, "zw"))
        args.extend(F(selected[0][name])
                    for name in MYNN_DMP_MF_SCALAR_INPUTS)
        result = _dmp_mf_column(*args)
        dz = _col(selected, "dz")
        zw = _iface(selected, "zw")
        nz = dz.size
        nup = result["up_w"].shape[1]
        summary[case] = {}
        for species in MYNN_DMP_MF_QN_COLUMN_INPUTS:
            qn = _col(selected, species)
            cpu = dmp_qn_flux_column(
                qn, dz, zw, result["up_w"], result["up_a"], result["ent"],
                result["rhoz"], result["psig_w"], result["plume_active"],
                result["limiter_adjustment"],
            )
            fixture = _iface(selected, f"s_aw{species}")
            np.testing.assert_array_equal(
                cpu, fixture, err_msg=f"{case}/s_aw{species} CPU vs fixture")
            # up_w/up_a are (nz+1, nup); the kernel reads k-major and never
            # touches the unwritten top row (structural zeros there).
            gpu = mynn_dmp_qn_flux_columns_cuda(
                qn[None, :], dz[None, :], zw[None, :],
                result["up_w"][None, :, :], result["up_a"][None, :, :],
                result["ent"][None, :, :], result["rhoz"][None, :nz],
                np.asarray([result["psig_w"]], dtype=np.float32),
                np.asarray([1 if result["plume_active"] else 0],
                           dtype=np.int32),
                np.asarray([result["limiter_adjustment"]],
                           dtype=np.float32),
            ).get()[0]
            summary[case][f"s_aw{species}"] = _ulp(gpu, cpu)
            np.testing.assert_array_equal(
                gpu, cpu, err_msg=f"{case}/s_aw{species} GPU vs CPU")
        assert nup == 8
    return summary


def _solve_arrays(selected):
    """The reference dtz/rhoinv/khdz/hdz/dzinv construction, replicated.

    Proven against the tree, not trusted: the CPU dqn produced from these
    arrays is asserted bit-equal to the fixture dqn* columns, which came
    through gpuwm.core.mynn_pbl's own construction.
    """
    dz = _col(selected, "dz")
    rho = _col(selected, "rho")
    dfh = _col(selected, "dfh")
    s_aw = _iface(selected, "s_aw")
    delt = F(selected[0]["delt"])
    nz = dz.size
    half = F(0.5)
    dtz = np.asarray([F(delt / dz[k]) for k in range(nz)], dtype=np.float32)
    rhoz = np.empty(nz + 1, dtype=np.float32)
    rhoinv = np.empty(nz, dtype=np.float32)
    khdz = np.empty(nz + 1, dtype=np.float32)
    rhoz[0] = rho[0]
    rhoinv[0] = F(F(1.0) / rho[0])
    khdz[0] = F(rhoz[0] * dfh[0])
    for k in range(1, nz):
        rhoz[k] = F(F(F(rho[k] * dz[k - 1]) + F(rho[k - 1] * dz[k]))
                    / F(dz[k - 1] + dz[k]))
        rhoz[k] = max(rhoz[k], F(1.0e-4))
        rhoinv[k] = F(F(1.0) / max(rho[k], F(1.0e-4)))
        khdz[k] = F(rhoz[k] * dfh[k])
    rhoz[nz] = rhoz[nz - 1]
    khdz[nz] = F(rhoz[nz] * dfh[nz - 1])
    for k in range(1, nz - 1):
        khdz[k] = max(khdz[k], F(half * s_aw[k]))
        khdz[k] = max(khdz[k], F(-F(half * F(s_aw[k] - s_aw[k + 1]))))
    hdz = np.asarray([F(F(half * dtz[k]) * rhoinv[k]) for k in range(nz)],
                     dtype=np.float32)
    dzinv = np.asarray([F(dtz[k] * rhoinv[k]) for k in range(nz)],
                       dtype=np.float32)
    return dz, rho, dfh, s_aw, delt, dtz, rhoinv, khdz, hdz, dzinv


def solve_arm(path: Path) -> dict:
    summary: dict[str, dict[str, int]] = {}
    for case, selected in _rows(path, TEND_CASES).items():
        (dz, rho, dfh, s_aw, delt, dtz, rhoinv, khdz, hdz,
         dzinv) = _solve_arrays(selected)
        summary[case] = {}
        for species in QN_SOLVE_ORDER:
            qn = _col(selected, species, TEND_RENAMED)
            s_awqn = _iface(selected, f"s_aw{species}")
            qn2_cpu, dqn_cpu = mix_scalar_column(
                qn, dtz, rhoinv, khdz, hdz, dzinv, s_aw, s_awqn, delt)
            fixture = _col(selected, f"d{species}")
            np.testing.assert_array_equal(
                dqn_cpu, fixture,
                err_msg=f"{case}/d{species} CPU vs fixture")
            qn2_gpu, dqn_gpu = mynn_mix_scalar_columns_cuda(
                qn[None, :], dz[None, :], rho[None, :], dfh[None, :],
                s_aw[None, :], s_awqn[None, :],
                np.asarray([delt], dtype=np.float32),
            )
            qn2_gpu = qn2_gpu.get()[0]
            dqn_gpu = dqn_gpu.get()[0]
            summary[case][f"{species}2"] = _ulp(qn2_gpu, qn2_cpu)
            summary[case][f"d{species}"] = _ulp(dqn_gpu, dqn_cpu)
            np.testing.assert_array_equal(
                qn2_gpu, qn2_cpu, err_msg=f"{case}/{species}2 GPU vs CPU")
            np.testing.assert_array_equal(
                dqn_gpu, dqn_cpu, err_msg=f"{case}/d{species} GPU vs CPU")
    return summary


def routed_lane_arm(path: Path) -> dict:
    """The RUNTIME route: mynn_tendencies_default_cuda(bl_mynn_mixscalars=1).

    Every fixture case through the actual lifted device lane — base
    outputs AND the five dqn* asserted bit-equal to the fixture columns
    (base via the CPU-equality already proven by the committed validator;
    here directly against the fixture CSV).
    """
    from gpuwm.core.mynn_pbl_gpu import mynn_tendencies_default_cuda

    base_outputs = ("du", "dv", "dth", "dqv", "dqc", "dqi", "dqs", "dozone")
    summary: dict[str, dict[str, int]] = {}
    for case, selected in _rows(path, TEND_CASES).items():
        values: dict[str, np.ndarray] = {}
        for name in (*MYNN_TENDENCIES_LAYER_INPUTS,
                     *MYNN_TENDENCIES_QN_LAYER_INPUTS):
            values[name] = _col(selected, name, TEND_RENAMED)[None, :]
        for name in (*MYNN_TENDENCIES_INTERFACE_INPUTS,
                     *MYNN_TENDENCIES_QN_INTERFACE_INPUTS):
            values[name] = _iface(selected, name)[None, :]
        for name in MYNN_TENDENCIES_SCALAR_INPUTS:
            values[name] = np.asarray([np.float32(selected[0][name])],
                                      dtype=np.float32)
        edmf_mom = int(selected[0]["bl_mynn_edmf_mom"])
        result = mynn_tendencies_default_cuda(
            values, bl_mynn_edmf_mom=edmf_mom, bl_mynn_mixscalars=1,
            flag_qnc=True, flag_qni=True, flag_qnwfa=True,
            flag_qnifa=True, flag_qnbca=True,
        )
        summary[case] = {}
        for name in (*base_outputs,
                     *[f"d{s}" for s in QN_SOLVE_ORDER]):
            expected = _col(selected, name)
            got = getattr(result, name).get()[0]
            summary[case][name] = _ulp(got, expected)
            np.testing.assert_array_equal(
                got, expected, err_msg=f"{case}/{name} routed lane")
        if all(not np.any(getattr(result, f"d{s}").get()[0] != 0.0)
               for s in QN_SOLVE_ORDER):
            raise AssertionError(f"{case}: routed lane mixed nothing")
    return summary


def main(fixture_dir: str) -> None:
    root = Path(fixture_dir)
    report = {
        "flux_arm_max_ulp_gpu_vs_cpu": flux_arm(
            root / "dmp-mf-mixscalars.csv"),
        "solve_arm_max_ulp_gpu_vs_cpu": solve_arm(
            root / "tendencies-mf-mixscalars.csv"),
        "routed_lane_max_ulp_vs_fixture": routed_lane_arm(
            root / "tendencies-mf-mixscalars.csv"),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    worst = max(v for arm in report.values()
                for case in arm.values() for v in case.values())
    print(f"WORST_ULP {worst}")
    print("PASS mynn_scalar_mix GPU vs CPU reference (bit-exact), "
          "CPU vs fixture (bit-exact), every case, all five species")


if __name__ == "__main__":
    # The anchored w4-oracle-fixtures family.  Its location is an
    # environment variable rather than a hard-coded absolute path:
    # the old default named a tree outside this branch, so a probe
    # run without an argument would have silently pointed at
    # nothing and reported "no cases" as a pass.
    main(sys.argv[1] if len(sys.argv) > 1
         else os.environ["GPUWM_W4_ORACLE_FIXTURES"])
