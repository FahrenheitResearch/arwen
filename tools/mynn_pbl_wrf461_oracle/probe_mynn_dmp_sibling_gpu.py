#!/usr/bin/env python3
"""Gate the sibling DMP unit: bit-identity, exports, device flux chain.

W4 full-admission lane (mf-close, Stage A).  Three claims, every DMP
fixture case:

  * IDENTITY: every classic ``MynnDmpMfResult`` output of the sibling
    dispatch (``bl_mynn_mixscalars=1``, ``kernels/mynn_dmp_sibling.cu``)
    is BIT-IDENTICAL to the frozen dispatch (``bl_mynn_mixscalars=0``,
    frozen ``kernels/mynn_pbl.cu``) on the same inputs — the tagged
    export lines changed no classic number.
  * EXPORTS: the sibling's four exports (pre-limiter ``up_a``,
    ``psig_w``, the NUP2 plume-active gate, ``limiter_adjustment``) and
    the scratch-viewed ``up_w``/``ent``/``rhoz`` are BIT-EQUAL to the
    CPU reference ``_dmp_mf_column``'s exports.
  * CHAIN: the five ``s_awqn*`` the wrapper now returns — device DMP
    producer -> device flux kernel, end to end — are BIT-EQUAL to the
    CPU ``dmp_qn_flux_column`` on the CPU exports AND to the anchored
    fixture columns.  The per-case/species ulp table is emitted verbatim.
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
    _dmp_mf_column,
)
from gpuwm.core.mynn_pbl_gpu import MynnDmpMfResult, mynn_dmp_mf_cuda
from gpuwm.core.mynn_scalar_mix import dmp_qn_flux_column

F = np.float32
NZ = 30
DMP_CASES = (
    "land_dry", "land_cumulus", "water_cumulus", "stable_off",
    "resolved_w", "flux_limited", "stochastic", "high_wind_thin",
    "deep_pblh", "cloud_base_capped", "fine_grid", "dead_probe",
)
DMP_RENAMED = {
    "qc_bl": "qc_bl_before", "cldfra_bl": "cldfra_bl_before",
    "vt": "vt_before", "vq": "vq_before",
}
QN_FLUXES = tuple(f"s_aw{name}" for name in MYNN_DMP_MF_QN_COLUMN_INPUTS)
CLASSIC = tuple(
    name for name in MynnDmpMfResult.__dataclass_fields__
    if name not in QN_FLUXES
)


def _ulp(got: np.ndarray, expected: np.ndarray) -> int:
    return int(np.abs(monotone_fp32_key(got)
                      - monotone_fp32_key(expected)).max(initial=0))


def _rows(path: Path) -> dict[str, list[dict]]:
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    out = {case: [row for row in rows if row["case"] == case]
           for case in DMP_CASES}
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


def main(fixture_dir: str) -> None:
    rows = _rows(Path(fixture_dir) / "dmp-mf-mixscalars.csv")
    identity = {}
    exports_tbl = {}
    chain_tbl = {}
    for case, selected in rows.items():
        values = {
            name: _col(selected, name, DMP_RENAMED)[None, :]
            for name in MYNN_DMP_MF_COLUMN_INPUTS
        }
        values["zw"] = _iface(selected, "zw")[None, :]
        for name in MYNN_DMP_MF_SCALAR_INPUTS:
            values[name] = np.asarray([F(selected[0][name])],
                                      dtype=np.float32)
        qn = {name: _col(selected, name)
              for name in MYNN_DMP_MF_QN_COLUMN_INPUTS}

        frozen = mynn_dmp_mf_cuda(dict(values), bl_mynn_mixscalars=0)
        sink: dict = {}
        sibling = mynn_dmp_mf_cuda(
            {**values, **{k: v[None, :] for k, v in qn.items()}},
            bl_mynn_mixscalars=1, export_sink=sink,
        )

        # IDENTITY: classic outputs, frozen vs sibling, bitwise.
        diffs = {}
        for name in CLASSIC:
            a = getattr(frozen, name).get()
            b = getattr(sibling, name).get()
            n = int(np.count_nonzero(
                a.view(np.uint8) != b.view(np.uint8)))
            if n:
                diffs[name] = n
        identity[case] = diffs
        assert not diffs, f"{case}: sibling classic outputs differ {diffs}"

        # EXPORTS vs the CPU reference.
        cpu_args = [
            _col(selected, name, DMP_RENAMED)
            for name in MYNN_DMP_MF_COLUMN_INPUTS
        ]
        cpu_args.append(_iface(selected, "zw"))
        cpu_args.extend(F(selected[0][name])
                        for name in MYNN_DMP_MF_SCALAR_INPUTS)
        cpu = _dmp_mf_column(*cpu_args)
        nz = NZ
        checks = {
            "psig_w": (sink["psig_w"].get()[0],
                       np.float32(cpu["psig_w"])),
            "plume_active": (int(sink["plume_active"].get()[0]),
                             int(bool(cpu["plume_active"]))),
            "limiter_adjustment": (
                sink["limiter_adjustment"].get()[0],
                np.float32(cpu["limiter_adjustment"])),
            "up_a_pre": (sink["up_a_pre"].get()[0], cpu["up_a"]),
            "up_w": (sink["up_w"].get()[0], cpu["up_w"]),
            "ent": (sink["ent"].get()[0], cpu["ent"]),
            "rhoz": (sink["rhoz"].get()[0], cpu["rhoz"][:nz]),
        }
        exports_tbl[case] = {}
        for name, (got, want) in checks.items():
            same = np.array_equal(np.asarray(got), np.asarray(want))
            exports_tbl[case][name] = "bit-equal" if same else "DIFFERS"
            assert same, f"{case}: export {name} differs from CPU"

        # CHAIN: device s_awqn* vs CPU chain and vs fixture.
        chain_tbl[case] = {}
        dz = _col(selected, "dz")
        zw = _iface(selected, "zw")
        for name in MYNN_DMP_MF_QN_COLUMN_INPUTS:
            cpu_flux = dmp_qn_flux_column(
                qn[name], dz, zw, cpu["up_w"], cpu["up_a"], cpu["ent"],
                cpu["rhoz"], cpu["psig_w"], cpu["plume_active"],
                cpu["limiter_adjustment"],
            )
            fixture = _iface(selected, f"s_aw{name}")
            got = getattr(sibling, f"s_aw{name}").get()[0]
            chain_tbl[case][f"s_aw{name}"] = {
                "ulp_vs_cpu": _ulp(got, cpu_flux),
                "ulp_vs_fixture": _ulp(got, fixture),
            }
            np.testing.assert_array_equal(
                got, cpu_flux, err_msg=f"{case}/s_aw{name} device vs CPU")
            np.testing.assert_array_equal(
                got, fixture,
                err_msg=f"{case}/s_aw{name} device vs fixture")

    report = {
        "identity_diffs_frozen_vs_sibling": identity,
        "exports_vs_cpu": exports_tbl,
        "device_chain_ulp": chain_tbl,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    worst = max(v for case in chain_tbl.values()
                for sp in case.values() for v in sp.values())
    print(f"WORST_ULP {worst}")
    print("PASS sibling DMP unit: classic outputs bit-identical to the "
          "frozen kernel, exports bit-equal to the CPU reference, device "
          "flux chain bit-equal to CPU and fixtures, all 12 cases")


if __name__ == "__main__":
    # The anchored w4-oracle-fixtures family.  Its location is an
    # environment variable rather than a hard-coded absolute path:
    # the old default named a tree outside this branch, so a probe
    # run without an argument would have silently pointed at
    # nothing and reported "no cases" as a pass.
    main(sys.argv[1] if len(sys.argv) > 1
         else os.environ["GPUWM_W4_ORACLE_FIXTURES"])
