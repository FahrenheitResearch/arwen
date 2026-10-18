"""The GSL drag suite (gwd_opt = 3) against WRF v4.6.1's own Fortran.

``tools/gsl_drag_wrf461_oracle/build.sh`` runs WRF v4.6.1's byte-unmodified
``module_bl_gwdo_gsl.F`` on two grids of physically consistent columns (an
HRRR-style 50-level grid and a 40-level one; convective, stable-night,
windy, blocking, ocean, high-terrain, snow/ice, tropical and critical-level
families plus 40 hand-built edge cases) at grid lengths that reach every
taper branch, with and without the SPP pattern.  ``pack.py`` folds the
fixture tree into one compressed ``.npz`` (``tests/data/oracles/gsl_drag``),
keyed ``"<grid>/<case>/<name>"`` with Fortran index order kept.

:func:`measure` runs :func:`gpuwm.core.terrain_drag.launch_gwdo_gsl` on each
case's inputs and compares every output word for word: RUBLTEN, RVBLTEN,
RTHBLTEN, the eight DTAUX3D/DTAUY3D components and the eight column
stresses, plus RUBLTEN/RVBLTEN again from a launch without diagnostics.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from gpuwm.core.fp32_ulp import fp32_ulp_distance

FIXTURE = (Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles"
           / "gsl_drag" / "gsl_drag_wrf461.npz")
COMPONENTS = ("ls", "bl", "ss", "fd")


def load(path=FIXTURE) -> dict[str, dict[str, np.ndarray]]:
    """``{"A/inputs": {...}, "A/dx3000": {...}, ...}`` from the packed file
    or from a raw fixture directory (``build.sh`` output)."""
    path = Path(path)
    cases: dict[str, dict[str, np.ndarray]] = {}
    if path.is_dir():
        from gpuwm.verify.urban_oracle import load_case
        for manifest in sorted(path.glob("*/*/MANIFEST.txt")):
            case = manifest.parent
            cases[f"{case.parent.name}/{case.name}"] = load_case(case)
        return cases
    with np.load(path) as packed:
        for key in packed.files:
            grid, case, name = key.split("/")
            cases.setdefault(f"{grid}/{case}", {})[name] = packed[key]
    return cases


def _kij(a) -> np.ndarray:
    """Oracle (ncol, nz) -> the port's (nz, 1, ncol)."""
    return np.ascontiguousarray(np.asarray(a, np.float32).T[:, None, :])


def _row(a) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a)[None, :])


def _planes(a):
    return [np.ascontiguousarray(np.asarray(a, np.float32)[:, m][None, :])
            for m in range(4)]


def run_case(inputs, case, *, diagnostics=True, pass_spp=True):
    """Launch the port on one case; host arrays keyed like the fixture.

    ``pass_spp=False`` launches an spp_pbl = 1 case without its pattern,
    which is what the port did before it took one (the measured defect).
    """
    import cupy as cp

    from gpuwm.core.terrain_drag import gsl_kpblmax, launch_gwdo_gsl

    atm = {name: cp.asarray(_kij(inputs[key])) for name, key in (
        ("u", "u3d"), ("v", "v3d"), ("temperature", "t3d"), ("qv", "qv3d"),
        ("pressure", "p3d"), ("exner", "pi3d"), ("p_interface", "p3di"),
        ("z", "z"), ("dz", "dz"))}
    du = cp.asarray(_kij(inputs["rublten0"]))
    dv = cp.asarray(_kij(inputs["rvblten0"]))
    diag = {} if diagnostics else None
    rstoch = None
    if int(case["spp_pbl"]) == 1 and pass_spp:
        # WRF: rstoch1D(i) = pattern_spp_pbl(i, kts, j).
        rstoch = _row(np.asarray(inputs["pattern_spp_pbl"])[:, 0])
    launch_gwdo_gsl(
        atm, du, dv,
        ls={"var": _row(inputs["var2d"]), "con": _row(inputs["oc12d"]),
            "oa": _planes(inputs["oa"]), "ol": _planes(inputs["ol"])},
        ss={"var": _row(inputs["var2dss"]), "con": _row(inputs["oc12dss"]),
            "oa": _planes(inputs["oass"]), "ol": _planes(inputs["olss"])},
        sina=_row(inputs["sina"]), cosa=_row(inputs["cosa"]),
        xland=_row(inputs["xland"]), br=_row(inputs["br"]),
        pblh=_row(inputs["pblh"]), kpbl=_row(inputs["kpbl"]),
        kpblmax=gsl_kpblmax(inputs["znu"]), dx=float(case["dx"]),
        dt=float(case["dt"]), diagnostics=diag,
        **({} if rstoch is None else {"rstoch": rstoch}))
    got = {"rublten": cp.asnumpy(du), "rvblten": cp.asnumpy(dv)}
    if diagnostics:
        for comp in COMPONENTS:
            got[f"dtaux3d_{comp}"] = cp.asnumpy(diag[f"DTAUX3D_{comp}"])
            got[f"dtauy3d_{comp}"] = cp.asnumpy(diag[f"DTAUY3D_{comp}"])
            got[f"dusfcg_{comp}"] = cp.asnumpy(diag[f"DUSFCG_{comp}"])
            got[f"dvsfcg_{comp}"] = cp.asnumpy(diag[f"DVSFCG_{comp}"])
    return got


def _compare(got, want) -> dict:
    got = np.ascontiguousarray(got, np.float32)
    want = np.ascontiguousarray(want, np.float32)
    differ = got.view(np.uint32) != want.view(np.uint32)
    ulp = fp32_ulp_distance(got, want)
    finite = np.isfinite(got) & np.isfinite(want)
    absd = np.abs(got.astype(np.float64) - want.astype(np.float64))
    cols = np.flatnonzero(differ.reshape(-1, differ.shape[-1]).any(axis=0))
    return {"lanes": int(got.size), "differ": int(differ.sum()),
            "max_ulp": int(ulp.max()) if ulp.size else 0,
            "max_abs": float(absd[finite].max()) if finite.any() else 0.0,
            "nonfinite_want": int((~np.isfinite(want)).sum()),
            "columns": [int(c) for c in cols]}


def measure(cases, *, pass_spp=True) -> dict:
    """Every output of every case, word for word.  ``{case: {field: stats}}``."""
    out = {}
    for key in sorted(cases):
        grid, name = key.split("/")
        if name == "inputs":
            continue
        case, inputs = cases[key], cases[f"{grid}/inputs"]
        got = run_case(inputs, case, diagnostics=True, pass_spp=pass_spp)
        bare = run_case(inputs, case, diagnostics=False, pass_spp=pass_spp)
        fields = {}
        for field in ("rublten", "rvblten"):
            fields[field] = _compare(got[field], _kij(case[field]))
            fields[f"{field}[no diagnostics]"] = _compare(bare[field],
                                                         _kij(case[field]))
        # gsl_diss_ht_opt = 0: WRF leaves RTHBLTEN as the PBL left it, and
        # the port never touches it; the oracle's value must be the input's.
        fields["rthblten"] = _compare(_kij(inputs["rthblten0"]),
                                      _kij(case["rthblten"]))
        for comp in COMPONENTS:
            for f in (f"dtaux3d_{comp}", f"dtauy3d_{comp}"):
                fields[f] = _compare(got[f], _kij(case[f]))
            for f in (f"dusfcg_{comp}", f"dvsfcg_{comp}"):
                fields[f] = _compare(got[f], _row(case[f]))
        out[key] = fields
    return out


def summary(result) -> dict:
    worst_ulp = max(f["max_ulp"] for c in result.values() for f in c.values())
    worst_abs = max(f["max_abs"] for c in result.values() for f in c.values())
    differ = sum(f["differ"] for c in result.values() for f in c.values())
    lanes = sum(f["lanes"] for c in result.values() for f in c.values())
    return {"cases": len(result), "lanes": lanes, "differing_lanes": differ,
            "max_ulp": worst_ulp, "max_abs": worst_abs}
