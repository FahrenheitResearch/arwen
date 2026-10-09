"""Compare the engine's km_opt=3 chain with every recorded WRF 4.6.1 word (GPU).

Runs the production launchers (``launch_wrf_smag3d_km``,
``launch_wrf_smag2d_hd``, ``launch_wrf_smag2d_vertical``) on the captured
input states.  The arithmetic mode is whatever the process selected before
import: default, ``GPUWM_WRF_EXACT=1`` (strict), optionally with
``GPUWM_WRF_EXACT_DIFFUSION=1``.  Arms per case and ``mix_isotropic``:

* ``coef``      deformation tensors, BN2 and the four exchange coefficients
* ``h_chain``   horizontal tendencies from the engine's OWN K and tensors
* ``h_iso``     horizontal tendencies from WRF's K and tensors (operator only)
* ``v{f}_*``    vertical driver for isfflx f, engine K, against WRF ``stock``
                (xkmh into vertical_diffusion_w_2) and ``wcontrol`` (xkmv)
* ``vi{f}_*``   the same vertical driver fed WRF's K (operator only)

Every float32 word is compared bitwise (signed zeros included).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "wrf_diffusion_oracle"))
sys.path.insert(0, str(HERE))

TENSORS = ("div", "d11", "d22", "d33", "d12", "d13", "d23")
COEFFICIENTS = ("kmh", "kmv", "khh", "khv", "bn2")
HORIZONTAL = ("u", "v", "w", "theta", "qv", "qc", "qi")
VERTICAL = ("u", "v", "w", "theta", "qv", "qc", "qi", "hfx_after", "qfx_after")


def _columns_bad(got, ref, nx, ny):
    """Set of (j, i) columns holding at least one differing word."""
    a = np.ascontiguousarray(got, np.float32).view(np.uint32)
    b = np.ascontiguousarray(ref, np.float32).view(np.uint32)
    diff = a != b
    while diff.ndim > 2:
        diff = diff.any(axis=0)
    jj, ii = np.nonzero(diff)
    return {(min(int(j), ny - 1), min(int(i), nx - 1)) for j, i in zip(jj, ii)}


def _config(meta, iso, **extra):
    from gpuwm.config import RunConfig
    values = dict(nx=int(meta["nx"]), ny=int(meta["ny"]), nz=int(meta["nz"]),
                  dx=meta["dx"], dy=meta["dy"], dt=meta["dt"], ztop=20000.,
                  run_seconds=0., open_x=bool(meta["bx"]), open_y=bool(meta["by"]),
                  km_opt=3, c_s=meta["c_s"], mix_isotropic=iso,
                  mix_upper_bound=meta["mix_upper_bound"], bl_pbl_physics=0)
    values.update(extra)
    return RunConfig(**values)


def engine_case(arrays, meta, iso, wrf=None):
    """Every engine output word for one case and ``mix_isotropic``.

    ``wrf`` (the captured WRF words) enables the operator-only arms that feed
    WRF's own K and tensors; without it only the engine's own chain runs.
    """
    import cupy as cp
    from gpuwm.core import dycore
    from gpuwm.verify.diffusion_oracle import device_state
    from deformation_compare import port_outputs
    from vertical_gpu import vertical_driver_gpu
    nx, ny, nz = (int(meta[k]) for k in ("nx", "ny", "nz"))
    arrays = dict(arrays)
    arrays.setdefault("mup", np.subtract(arrays["mut"], arrays["mub2d"], dtype=np.float32))
    arrays.setdefault("mup0", arrays["mup"])
    # km_opt=3 reads no TKE; the shared coefficient helper reports it back.
    arrays.setdefault("tke", np.full_like(arrays["alt"], 0.5))
    out = {}
    # 1. Coefficients and the full deformation tensor (production launcher).
    # The inline tensors (d13/d23 at w levels, d33, div) are read through
    # deformation_compare's probe, whose device functions are production
    # smag2d's; it must compile as production smag2d does (DIFFUSION_OPTIONS
    # through the diffusion compile site), not as a default RawModule, which
    # contracts multiply-adds the production words do not.
    from unittest.mock import patch
    from gpuwm.core.kernels import compile_diffusion_source, module_options

    raw_module = cp.RawModule

    def production_probe(*args, **kwargs):
        code = kwargs.get("code", args[0] if args else "")
        if "oracle_expose_deformation" not in code:
            return raw_module(*args, **kwargs)
        return compile_diffusion_source(code, "verify:km3-deformation-probe",
                                        module_options("smag2d"))
    with patch.object(cp, "RawModule", production_probe):
        coef = port_outputs(arrays, meta, km_opt=3, isotropic=iso)
    for key in COEFFICIENTS:
        out[f"coef__{key}"] = coef[key]
    if iso == 0:
        for key in TENSORS:
            out[f"coef__{key}"] = coef[key]
    # 2. Horizontal tendencies, chained (engine K) and isolated (WRF K).
    cf = tuple(meta[k] for k in ("cf1", "cf2", "cf3"))
    for arm in ("h_chain", "h_iso") if wrf is not None else ("h_chain",):
        state = device_state(arrays, cf=cf)
        cfg = _config(meta, iso)
        if arm == "h_chain":
            km = cp.zeros_like(state.alt)
            kh = cp.zeros_like(km)
            d11, d22, d12 = dycore.launch_wrf_smag3d_km(state, cfg, km, kh, time_t=False)
            kmv = state.scratch(km.shape, "smag_kmv").copy()
            deformation = (d11, d22, d12)
            kmh, khh = km, kh
        else:
            kmh = cp.asarray(wrf[f"wrf__iso{iso}__kmh"])
            khh = cp.asarray(wrf[f"wrf__iso{iso}__khh"])
            kmv = cp.asarray(wrf[f"wrf__iso{iso}__kmv"])
            deformation = (cp.asarray(wrf["wrf__d11"]), cp.asarray(wrf["wrf__d22"]),
                           cp.ascontiguousarray(cp.asarray(wrf["wrf__d12"])[:, :ny, :nx]))
        for name, field, stag, xk in (("u", state.u, "x", kmh), ("v", state.v, "y", kmh),
                                      ("w", state.w, "z", kmv), ("theta", state.thp, "", khh),
                                      ("qv", state.qv, "", khh), ("qc", state.qc, "", khh),
                                      ("qi", state.qi, "", khh)):
            tend = cp.zeros_like(field)
            dycore.launch_wrf_smag2d_hd(state, cfg, field, xk, tend, stagger=stag, time_t=False,
                                        full_theta=name == "theta", deformation=deformation)
            dycore._zero_open_strips(tend, cfg, 1)
            out[f"{arm}__{name}"] = cp.asnumpy(tend)
    # 3. Vertical driver with the engine's own coefficients.
    engine_k = {"kmh": coef["kmh"], "kmv": coef["kmv"], "khv": coef["khv"]}
    arms = [("v", engine_k)]
    if wrf is not None:
        arms.append(("vi", {k: wrf[f"wrf__iso{iso}__{k}"] for k in ("kmh", "kmv", "khv")}))
    for arm, coefficients in arms:
        for flux in (0, 1, 2):
            got = vertical_driver_gpu(arrays, {**meta, "nx": nx, "ny": ny, "nz": nz}, coefficients,
                                      km_opt=3, isfflx=flux, initial_tke=0.)
            for key in VERTICAL:
                out[f"{arm}{flux}__{key}"] = got[key]
    return out


def reference_key(arm, field, iso):
    if arm == "coef":
        return f"wrf__{field}" if field in TENSORS else f"wrf__iso{iso}__{field}"
    if arm.startswith("h_"):
        return f"wrf__iso{iso}__h_{field}"
    raise KeyError(arm)


def compare_case(path, dump=None):
    from gpuwm.verify.diffusion_oracle import word_comparison
    with np.load(path) as data:
        wrf = {k: data[k] for k in data.files}
    arrays = {k.removeprefix("input__"): v for k, v in wrf.items() if k.startswith("input__")}
    meta = json.loads(str(wrf["meta_json"]))
    nx, ny = int(meta["nx"]), int(meta["ny"])
    rows = {}
    bad = {}
    for iso in (0, 1):
        got = engine_case(arrays, meta, iso, wrf)
        if dump is not None:
            np.savez_compressed(Path(dump) / f"{Path(path).stem}-iso{iso}.npz", **got)
        for key, value in got.items():
            arm, field = key.split("__")
            refs = []
            if arm.startswith("v"):
                flux = arm[-1]
                for wrf_arm in ("stock", "wcontrol"):
                    refs.append((f"{arm}_{wrf_arm}", wrf[f"wrf__iso{iso}__v{flux}_{wrf_arm}_{field}"]))
            else:
                refs.append((arm, wrf[reference_key(arm, field, iso)]))
            for label, ref in refs:
                metric = word_comparison(value, ref)
                metric["gpu_sha256"] = hashlib.sha256(np.ascontiguousarray(value, np.float32).tobytes()).hexdigest()
                rows[f"iso{iso}/{label}/{field}"] = metric
                bad.setdefault(f"iso{iso}/{label}", set()).update(_columns_bad(value, ref, nx, ny))
    columns = {arm: {"columns": nx * ny, "columns_with_any_difference": len(s)} for arm, s in bad.items()}
    return {"meta": {k: meta[k] for k in ("regime", "family", "nx", "ny", "nz", "dx", "dt", "bx", "by")},
            "fields": rows, "columns": columns}


def summarize(cases):
    groups = {}
    for case in cases.values():
        for key, m in case["fields"].items():
            iso, label, field = key.split("/")
            arm = label
            g = groups.setdefault(arm, {}).setdefault(field, {"words": 0, "different_words": 0,
                                                              "max_ulp": 0, "max_absolute": 0.0})
            g["words"] += m["words"]
            g["different_words"] += m["different_words"]
            g["max_ulp"] = max(g["max_ulp"], m["max_ulp"])
            g["max_absolute"] = max(g["max_absolute"], m["max_absolute"])
    return groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fixtures", type=Path)
    p.add_argument("receipt", type=Path)
    p.add_argument("--dump", type=Path, help="also save every engine array here")
    p.add_argument("--only", nargs="*", help="case names to run (diagnosis)")
    a = p.parse_args()
    if a.dump is not None:
        a.dump.mkdir(parents=True, exist_ok=True)
    import cupy as cp
    from gpuwm.core.kernels import module_source
    from gpuwm import wrf_exact
    manifest = json.loads((a.fixtures / "manifest.json").read_text())
    cases = {}
    for case in manifest["cases"]:
        if a.only and case["name"] not in a.only:
            continue
        path = a.fixtures / case["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != case["sha256"]:
            raise ValueError(f"fixture changed: {case['file']}")
        cases[case["name"]] = compare_case(path, a.dump)
        worst = max(m["max_ulp"] for m in cases[case["name"]]["fields"].values())
        print(case["name"], "max_ulp", worst, flush=True)
    summary = summarize(cases)
    receipt = {"schema": "wrf461-km3-column-oracle-compare-v1",
               "mode": {"GPUWM_WRF_EXACT": os.environ.get("GPUWM_WRF_EXACT", "0"),
                        "GPUWM_WRF_EXACT_DIFFUSION": os.environ.get("GPUWM_WRF_EXACT_DIFFUSION", "0"),
                        "strict_options": list(wrf_exact.STRICT_OPTIONS) if wrf_exact.ENABLED else []},
               "device": str(cp.cuda.runtime.getDeviceProperties(0)["name"]),
               "cupy": cp.__version__,
               "smag2d_source_sha256": hashlib.sha256(module_source("smag2d").encode()).hexdigest(),
               "columns": manifest["columns"], "cases": cases, "summary": summary}
    a.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({arm: {f: [v["different_words"], v["max_ulp"], v["max_absolute"]]
                            for f, v in fields.items()} for arm, fields in summary.items()}, indent=1))


if __name__ == "__main__":
    main()
