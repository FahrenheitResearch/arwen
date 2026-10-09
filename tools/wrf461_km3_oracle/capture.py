"""Record every WRF 4.6.1 km_opt=3 output word for the column-oracle cases (CPU).

For each case and each ``mix_isotropic`` (0 and 1) the compiled WRF chain
runs exactly as first_rk_step_part2 orders it for diff_opt=2, km_opt=3,
bl_pbl_physics=0:

  compute_diff_metrics -> phy_prep -> cal_deform_and_div (+ phy_bc) ->
  calculate_km_kh (calculate_N2, smag_km) (+ phy_bc) ->
  horizontal_diffusion_2 -> vertical_diffusion_2 (isfflx = 0, 1, 2)

The vertical driver runs twice per isfflx: ``stock`` hands
vertical_diffusion_w_2 WRF's own xkmh; ``wcontrol`` hands it xkmv (the
engine's documented choice, see README), so the remaining vertical words
are still compared bitwise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "wrf_diffusion_oracle"))
sys.path.insert(0, str(HERE))
from cases import core, pad2  # noqa: E402
from deformation_reference import reference_for_case  # noqa: E402
import horizontal_driver  # noqa: E402
import vertical_driver  # noqa: E402
import km3_cases  # noqa: E402

COEFFICIENTS = ("kmh", "kmv", "khh", "khv", "bn2")
TENSORS = ("div", "d11", "d22", "d33", "d12", "d13", "d23")
HORIZONTAL = ("u", "v", "w", "theta", "qv", "qc", "qi")
VERTICAL = ("u", "v", "w", "theta", "qv", "qc", "qi", "hfx_after", "qfx_after")


def tensor_core(key, a, nx, ny, nz):
    if key == "d13":
        return np.ascontiguousarray(a[3:3 + nx + 1, :nz + 1, 3:3 + ny].transpose(1, 2, 0))
    if key == "d23":
        return np.ascontiguousarray(a[3:3 + nx, :nz + 1, 3:3 + ny + 1].transpose(1, 2, 0))
    if key == "d12":
        return np.ascontiguousarray(a[3:3 + nx + 1, :nz, 3:3 + ny + 1].transpose(1, 2, 0))
    return core(a, nx, ny, nz)


def run_case(library, arrays, meta):
    nx, ny, nz = (int(meta[k]) for k in ("nx", "ny", "nz"))
    saved = {}
    for iso in (0, 1):
        ref = reference_for_case(library, arrays, meta, km_opt=3, isotropic=iso)
        tag = f"iso{iso}"
        if iso == 0:
            for key in TENSORS:
                saved["wrf__" + key] = tensor_core(key, ref[key], nx, ny, nz)
        for key in COEFFICIENTS:
            saved[f"wrf__{tag}__{key}"] = core(ref[key], nx, ny, nz)
        horizontal = horizontal_driver.reference(library, ref, meta, 3)
        for key in HORIZONTAL:
            saved[f"wrf__{tag}__h_{key}"] = horizontal[key]
        for key in ("ust", "hfx", "qfx"):
            ref[key] = pad2(arrays[key], nx, ny, meta["bx"], meta["by"])
        for arm in ("stock", "wcontrol"):
            prepared = dict(ref)
            if arm == "wcontrol":
                prepared["kmh"] = ref["kmv"]
            for flux in (0, 1, 2):
                vertical = vertical_driver.reference(library, prepared, meta, km_opt=3, isfflx=flux)
                for key in VERTICAL:
                    saved[f"wrf__{tag}__v{flux}_{arm}_{key}"] = vertical[key]
    return saved


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("library", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--fixtures", type=Path, required=True,
                   help="tests/data/wrf471_diffusion (retained real-state inputs)")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "wrf461-km3-column-oracle-v1", "cases": [],
                "build": json.loads((a.library.parent / "build-receipt.json").read_text())}
    for name, arrays, meta in km3_cases.all_cases(a.fixtures):
        saved = {"input__" + k: v for k, v in arrays.items()}
        saved.update(run_case(a.library, arrays, meta))
        saved["meta_json"] = np.array(json.dumps(meta, sort_keys=True))
        target = a.output / f"{name}.npz"
        np.savez_compressed(target, **saved)
        words = sum(v.size for k, v in saved.items() if k.startswith("wrf__"))
        manifest["cases"].append({"name": name, "file": target.name, "columns": km3_cases.columns(meta),
                                  "regime": meta["regime"], "family": meta["family"],
                                  "bx": meta["bx"], "by": meta["by"], "dx": meta["dx"],
                                  "nz": meta["nz"], "wrf_words": int(words),
                                  "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
        print(name, words, flush=True)
    manifest["columns"] = sum(c["columns"] for c in manifest["cases"])
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
