"""Build the WRF v4.6.1 km_opt=1 (isotropic_km) mixing-package oracle.

Compiles byte-unmodified WRF v4.6.1 bodies -- the metric, deformation and
coefficient chain (``compute_diff_metrics``, ``cal_deform_and_div``,
``calculate_km_kh`` -> ``isotropic_km``), the complete
``horizontal_diffusion_2`` and ``vertical_diffusion_2`` drivers with their
leaves, ``set_physical_bc3d`` and ``phy_prep`` -- with the same service ABI
and flags as the v4.7.1 diffusion oracles (build_common.py).  The v4.6.1
``module_diffusion_em.F``, ``module_model_constants.F`` and ``module_bc.F``
are byte-identical to the v4.7.1 files pinned there; only
``module_big_step_utilities_em.F`` (``phy_prep``) differs between the two
releases, so this build pins all four v4.6.1 hashes itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from build_common import CONFIG_SOURCE, FLAGS, extract_routines, sha256

WRF_RELEASE = "4.6.1"
WRF_COMMIT = "d66e442fccc04111067e29274c9f9eaccc3cef28"
PINS = {
    "dyn_em/module_diffusion_em.F": "a7d4570c97e51c635e86a0dbd628c6846457ac5b93d5a7af798b118c7d8d2d54",
    "dyn_em/module_big_step_utilities_em.F": "8f0649b458fceabd5c31c87ec9f266840fb164e2629322904a3f6ee04c72331a",
    "share/module_model_constants.F": "5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062",
    "share/module_bc.F": "61b9235004b2a7799faabaa928276af8a7ef2e4672619c8ad120c857461301ad",
}
DIFFUSION_ROUTINES = (
    # metrics, tensors and coefficients (deformation_build.ROUTINES)
    "cal_deform_and_div", "calculate_km_kh", "calculate_N2", "cal_dampkm",
    "isotropic_km", "smag_km", "smag2d_km", "tke_km", "calc_l_scale",
    "pthl", "pu", "compute_diff_metrics",
    # the two outer drivers and every leaf they call
    "horizontal_diffusion_2", "horizontal_diffusion_u_2",
    "horizontal_diffusion_v_2", "horizontal_diffusion_w_2",
    "horizontal_diffusion_s", "vertical_diffusion_2",
    "vertical_diffusion_u_2", "vertical_diffusion_v_2",
    "vertical_diffusion_w_2", "vertical_diffusion_s",
    "cal_titau_11_22_33", "cal_titau_12_21", "cal_titau_13_31",
    "cal_titau_23_32")
WRAPPERS = ("deformation_wrappers.F90", "km1_wrapper.F90")


def build(source_root: Path, output: Path) -> Path:
    root = Path(source_root).resolve()
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for rel, pin in PINS.items():
        if sha256(root / rel) != pin:
            raise ValueError(f"WRF source changed: {rel}. The oracle must use the pinned v4.6.1 bytes.")
    bodies, spans = extract_routines(root / "dyn_em/module_diffusion_em.F", DIFFUSION_ROUTINES)
    extra = {}
    for rel, names in (("share/module_bc.F", ["set_physical_bc3d"]),
                       ("dyn_em/module_big_step_utilities_em.F", ["phy_prep"])):
        more, more_spans = extract_routines(root / rel, names)
        bodies += b"\n" + more
        extra[rel] = {"sha256": sha256(root / rel), "routines": more_spans}
    generated = out / "wrf_routines.F90"
    generated.write_bytes(b"module module_diffusion_em\nuse module_oracle_config\nuse module_model_constants\n"
                          b"   INTEGER, PARAMETER            :: bdyzone = 4\ncontains\n"
                          + bodies + b"\nend module module_diffusion_em\n")
    (out / "oracle_config.F90").write_text(CONFIG_SOURCE, encoding="utf-8", newline="\n")
    here = Path(__file__).resolve().parent
    wrapper = out / "km1_wrappers.F90"
    wrapper.write_bytes(b"\n".join((here / name).read_bytes() for name in WRAPPERS))
    commands = []
    for src, obj in ((out / "oracle_config.F90", "oracle_config.o"),
                     (root / "share/module_model_constants.F", "constants.o"),
                     (generated, "routines.o"), (wrapper, "wrapper.o")):
        command = ["gfortran", *FLAGS, "-c", str(src), "-o", obj]
        commands.append(command)
        subprocess.run(command, cwd=out, check=True)
    command = ["gfortran", "-shared", "oracle_config.o", "constants.o", "routines.o", "wrapper.o", "-o", "oracle.so"]
    commands.append(command)
    subprocess.run(command, cwd=out, check=True)
    compiler = subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0]
    receipt = {"wrf_release": WRF_RELEASE, "wrf_commit": WRF_COMMIT, "compiler": compiler,
               "flags": FLAGS,
               "commands": [[Path(a).name if a.startswith("/") else a for a in c] for c in commands],
               "routines": spans, "extra_sources": extra,
               "source_sha256": {rel: sha256(root / rel) for rel in PINS},
               "wrapper_sources_sha256": {n: sha256(here / n) for n in WRAPPERS},
               "config_sha256": sha256(out / "oracle_config.F90"),
               "generated_sha256": sha256(generated), "library_sha256": sha256(out / "oracle.so")}
    (out / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
    return out / "oracle.so"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source_root", type=Path, help="pristine WRF v4.6.1 tree")
    p.add_argument("output", type=Path)
    a = p.parse_args()
    print(build(a.source_root, a.output))
