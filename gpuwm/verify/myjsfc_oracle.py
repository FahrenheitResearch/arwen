"""Grade the Eta similarity surface layer against its WRF v4.6.1 oracle.

The fixture (tests/data/oracles/myjsfc/myjsfc-wrf461.npz) is written by
tools/myjsfc_wrf461_oracle/build.sh: the byte-unmodified
phys/module_sf_myjsfc.F compiled by gfortran, driven column by column by
run_myjsfc.F90.  It carries MYJSFCINIT's PSIM/PSIH tables and, for each
set of columns, the post-call words of every MYJSFC call.

Two comparisons:

* :func:`compare_tables` -- the tables WOOF builds on the host
  (:func:`gpuwm.core.myjsfc_tables.build_psi_tables`) against the tables
  MYJSFCINIT built, word for word.  CPU only.
* :func:`measure` -- the CUDA kernel against MYJSFC, every INOUT and output
  field, bitwise, in two modes: ``replay`` hands the kernel WRF's own
  pre-call state at every call (each call graded alone), ``free_run``
  lets the kernel carry its own state from call to call (what a forecast
  does).

ULP distance is measured on the float32 total order, so +0 and -0 are 0
apart and two NaNs are 0 apart. The mismatch count grades raw words,
including both zero signs and NaN payloads. The default surface compile
preserves subnormals; ``flushed_subnormal`` remains a diagnostic count.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

F = np.float32
FIXTURE = (Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles"
           / "myjsfc" / "myjsfc-wrf461.npz")
TABLE_NAMES = ("psim1", "psih1", "psim2", "psih2")
SCALAR_NAMES = ("dzeta1", "dzeta2", "ztmin1", "ztmax1", "ztmin2", "ztmax2",
                "fh01", "fh02")
_STATE_KEYS = {"ust": "ustar", "znt": "znt", "thz0": "thz0", "qz0": "qz0",
               "uz0": "uz0", "vz0": "vz0", "qsfc": "qsfc", "akhs": "akhs",
               "akms": "akms"}


def load(path: Path | str = FIXTURE) -> dict:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def set_names(fixture: dict) -> list[str]:
    return sorted({key.split("/")[0] for key in fixture
                   if key.endswith("/meta")})


def _ordered(bits: np.ndarray) -> np.ndarray:
    signed = bits.astype(np.int64)
    return np.where(signed & 0x80000000, 0x80000000 - signed, signed)


def ulp_distance(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    got = np.ascontiguousarray(got, dtype=F)
    want = np.ascontiguousarray(want, dtype=F)
    dist = np.abs(_ordered(got.view(np.uint32)) - _ordered(want.view(np.uint32)))
    both_nan = np.isnan(got) & np.isnan(want)
    return np.where(both_nan, 0, dist)


def _stats(got: np.ndarray, want: np.ndarray) -> dict:
    dist = ulp_distance(got, want)
    tiny = np.finfo(F).tiny
    flushed = int(np.sum((got == 0.0) & (want != 0.0) & (np.abs(want) < tiny)))
    with np.errstate(invalid="ignore"):
        absdiff = np.abs(got.astype(np.float64) - want.astype(np.float64))
    absdiff = np.where(np.isnan(absdiff), 0.0, absdiff)
    return {"max_ulp": int(dist.max(initial=0)),
            "mismatch": int(np.count_nonzero(np.ascontiguousarray(got, F).view(np.uint32) != np.ascontiguousarray(want, F).view(np.uint32))),
            "count": int(dist.size),
            "max_abs": float(absdiff.max(initial=0.0)),
            "flushed_subnormal": flushed}


def compare_tables(fixture: dict) -> dict:
    """WOOF's MYJSFCINIT tables against the oracle's, word for word."""
    from gpuwm.core.myjsfc_tables import build_psi_tables
    tables = build_psi_tables()
    report = {}
    for name in TABLE_NAMES:
        report[name] = _stats(np.asarray(tables[name], F),
                              fixture[f"tables/{name}"])
    want = fixture["tables/scalars"]
    got = np.asarray([tables[name] for name in SCALAR_NAMES], F)
    report["scalars"] = _stats(got, want)
    return report


def _device_inputs(cp, case: dict, launcher) -> tuple[dict, dict]:
    def dev(array):
        return cp.ascontiguousarray(cp.asarray(array, dtype=F))
    nz, ncol = case["dz"].shape
    columns = {"dz": dev(case["dz"].reshape(nz, 1, ncol)),
               "tke": dev(case["q2"].reshape(nz, 1, ncol))}
    surface = {
        "u1": case["u"][0], "v1": case["v"][0], "t1": case["t"][0],
        "th1": case["th"][0], "qv1": case["qv"][0], "qc1": case["qc"][0],
        "p1": case["pmid"][0], "psfc": case["pint"][0], "tsk": case["tsk"],
        "xland": case["xland"], "mavail": case["mavail"],
        "z0base": case["z0base"], "ht": case["ht"],
    }
    surface = {name: dev(np.asarray(value).reshape(1, ncol))
               for name, value in surface.items()
               if name in launcher._SURFACE_INPUTS}
    return columns, surface


def measure(fixture: dict) -> dict:
    """Run the CUDA kernel over every set and call; grade every field."""
    import cupy as cp
    from gpuwm.core import myjsfc as launcher

    names = launcher.MYJ_SFCLAY_INOUT + launcher.MYJ_SFCLAY_OUTPUTS
    fields = [str(f) for f in fixture["fields"]]
    if tuple(fields) != tuple(names):
        raise ValueError(f"fixture fields {fields} are not the launcher's "
                         f"{names}")
    report = {"replay": {}, "free_run": {}, "per_set": {}}
    collected = {mode: {name: ([], []) for name in names}
                 for mode in ("replay", "free_run")}
    for set_name in set_names(fixture):
        ncol, nz, nsteps, it0 = (int(v) for v in fixture[f"{set_name}/meta"])
        case = {key.split("/", 2)[2]: value for key, value in fixture.items()
                if key.startswith(f"{set_name}/in/")}
        want_all = fixture[f"{set_name}/out"]          # (nsteps, NF, ncol)
        columns, surface = _device_inputs(cp, case, launcher)
        initial = {name: case[_STATE_KEYS[name]]
                   for name in launcher.MYJ_SFCLAY_INOUT}
        per_set = {}
        for mode in ("replay", "free_run"):
            carried = {name: cp.asarray(initial[name].reshape(1, ncol), F)
                       for name in launcher.MYJ_SFCLAY_INOUT}
            for step in range(nsteps):
                if mode == "replay" and step > 0:
                    pre = want_all[step - 1]
                    carried = {name: cp.asarray(
                        pre[fields.index(name)].reshape(1, ncol), F)
                        for name in launcher.MYJ_SFCLAY_INOUT}
                state = {name: cp.ascontiguousarray(carried[name].copy())
                         for name in launcher.MYJ_SFCLAY_INOUT}
                outputs = {name: cp.zeros((1, ncol), dtype=F)
                           for name in launcher.MYJ_SFCLAY_OUTPUTS}
                launcher.launch_myj_sfclay(columns, surface, state, outputs,
                                           itimestep=it0 + step)
                cp.cuda.Device().synchronize()
                got_all = {**state, **outputs}
                for name in names:
                    got = cp.asnumpy(got_all[name]).reshape(ncol)
                    want = want_all[step, fields.index(name)]
                    collected[mode][name][0].append(got)
                    collected[mode][name][1].append(want)
                    stats = _stats(got, want)
                    if stats["mismatch"]:
                        key = f"{mode}/step{step}"
                        per_set.setdefault(key, {})[name] = stats
                carried = state
        report["per_set"][set_name] = per_set
    for mode in ("replay", "free_run"):
        for name in names:
            got = np.concatenate(collected[mode][name][0])
            want = np.concatenate(collected[mode][name][1])
            report[mode][name] = _stats(got, want)
    return report


def measure_cpu_authority(fixture: dict) -> dict:
    """The float32 CPU authority against MYJSFC, same grading as :func:`measure`.

    ``gpuwm.verify.myj_ref.np_myjsfc_column`` is the host twin of the CUDA
    kernel that tests/test_myj_port.py compares it with.  Graded here column
    by column on CPU, replaying WRF's pre-call state and free-running on
    its own, so the twin's distance from WRF is a measured number rather
    than an assumed tolerance.
    """
    from gpuwm.verify.myj_ref import myjsfc_psi_tables, np_myjsfc_column

    tables = myjsfc_psi_tables()
    fields = [str(f) for f in fixture["fields"]]
    inout = tuple(_STATE_KEYS)
    report = {"replay": {}, "free_run": {}, "per_set": {}}
    collected = {mode: {name: ([], []) for name in fields}
                 for mode in ("replay", "free_run")}
    for set_name in set_names(fixture):
        ncol, nz, nsteps, it0 = (int(v) for v in fixture[f"{set_name}/meta"])
        case = {key.split("/", 2)[2]: value for key, value in fixture.items()
                if key.startswith(f"{set_name}/in/")}
        want_all = fixture[f"{set_name}/out"]
        per_set = {}
        for mode in ("replay", "free_run"):
            carried = {name: case[_STATE_KEYS[name]].astype(F).copy()
                       for name in inout}
            for step in range(nsteps):
                if mode == "replay" and step > 0:
                    pre = want_all[step - 1]
                    carried = {name: pre[fields.index(name)].astype(F).copy()
                               for name in inout}
                got = np.zeros((len(fields), ncol), F)
                for i in range(ncol):
                    out = np_myjsfc_column(
                        dz=case["dz"][:, i], tke=case["q2"][:, i],
                        u1=case["u"][0, i], v1=case["v"][0, i],
                        t1=case["t"][0, i], th1=case["th"][0, i],
                        qv1=case["qv"][0, i], qc1=case["qc"][0, i],
                        p1=case["pmid"][0, i], psfc=case["pint"][0, i],
                        tsk=case["tsk"][i], xland=case["xland"][i],
                        mavail=case["mavail"][i], z0base=case["z0base"][i],
                        ht=case["ht"][i], itimestep=it0 + step,
                        tables=tables,
                        **{name: carried[name][i] for name in inout})
                    for j, name in enumerate(fields):
                        got[j, i] = out[name]
                for j, name in enumerate(fields):
                    want = want_all[step, j]
                    collected[mode][name][0].append(got[j])
                    collected[mode][name][1].append(want)
                    stats = _stats(got[j], want)
                    if stats["mismatch"]:
                        per_set.setdefault(f"{mode}/step{step}", {})[name] = stats
                carried = {name: got[fields.index(name)].copy()
                           for name in inout}
        report["per_set"][set_name] = per_set
    for mode in ("replay", "free_run"):
        for name in fields:
            report[mode][name] = _stats(
                np.concatenate(collected[mode][name][0]),
                np.concatenate(collected[mode][name][1]))
    return report


def format_report(tables: dict, report: dict | None, header: str) -> str:
    lines = [header, "== MYJSFCINIT tables (host build vs WRF)"]
    for name, stats in tables.items():
        lines.append(f"  {name:9s} max_ulp {stats['max_ulp']:>12d}  "
                     f"bit_mismatch {stats['mismatch']:>6d}/{stats['count']:<6d}"
                     f"  max_abs {stats['max_abs']:.3e}")
    if report is None:
        return "\n".join(lines) + "\n"
    for mode in ("replay", "free_run"):
        lines.append(f"== {mode}")
        for name, stats in report[mode].items():
            lines.append(
                f"  {name:8s} max_ulp {stats['max_ulp']:>12d}  bit_mismatch "
                f"{stats['mismatch']:>5d}/{stats['count']:<5d}  max_abs "
                f"{stats['max_abs']:.3e}  flushed_subnormal "
                f"{stats['flushed_subnormal']}")
    lines.append("== per set, calls with any bit mismatch")
    for set_name, per_set in report["per_set"].items():
        lines.append(f"  {set_name}: " + (", ".join(
            f"{key}[{','.join(sorted(v))}]" for key, v in per_set.items())
            or "none"))
    return "\n".join(lines) + "\n"


__all__ = ["FIXTURE", "load", "set_names", "ulp_distance", "compare_tables",
           "measure", "measure_cpu_authority", "format_report"]
