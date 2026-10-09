"""Compile the WRF v4.6.1 km_opt=2 routines into the column-oracle library.

The routine bodies are byte slices of the pinned WRF v4.6.1 sources (commit
d66e442f).  Two builds are made from the same slices:

* ``noopt``: ``-O0 -ffp-contract=off -fno-tree-vectorize``.  Every REAL**REAL,
  EXP and LOG is one scalar call into the host's libm and there is no
  contraction.  This is the 0 ULP referee.
* ``stock``: WRF's own GNU flags (``-O2 -ftree-vectorize -funroll-loops`` and
  the rest of arch/configure.defaults).  Reported for reference only: at -O2
  gfortran may vectorise a loop into libmvec, which rounds differently from
  the scalar call, so a whole-model build is WRF against itself.

Usage: python build.py WRF_SOURCE_ROOT OUT_DIR [--variant noopt|stock|both]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "wrf_diffusion_oracle"))
from build_common import CONFIG_SOURCE, extract_routines  # noqa: E402

WRF_RELEASE = "4.6.1"
WRF_COMMIT = "d66e442fccc04111067e29274c9f9eaccc3cef28"
SOURCE_PINS = {
    "dyn_em/module_diffusion_em.F":
        "a7d4570c97e51c635e86a0dbd628c6846457ac5b93d5a7af798b118c7d8d2d54",
    "dyn_em/module_big_step_utilities_em.F":
        "8f0649b458fceabd5c31c87ec9f266840fb164e2629322904a3f6ee04c72331a",
    "share/module_model_constants.F":
        "5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062",
    "share/module_bc.F":
        "61b9235004b2a7799faabaa928276af8a7ef2e4672619c8ad120c857461301ad",
}
DIFFUSION_ROUTINES = (
    "cal_deform_and_div", "calculate_km_kh", "calculate_N2", "cal_dampkm",
    "isotropic_km", "smag_km", "smag2d_km", "tke_km", "calc_l_scale",
    "pthl", "pu", "compute_diff_metrics",
    "tke_rhs", "tke_shear", "tke_buoyancy", "tke_dissip")
FLAGS = {
    "noopt": ["-O0", "-g", "-fPIC", "-cpp", "-ffree-form",
              "-ffree-line-length-none", "-ffp-contract=off",
              "-fno-tree-vectorize"],
    # arch/configure.defaults, GNU (gfortran/gcc) entry, as the dmpar build
    # on this box used it, plus -fPIC for the shared library.
    "stock": ["-O2", "-ftree-vectorize", "-funroll-loops", "-w", "-fPIC",
              "-cpp", "-ffree-form", "-ffree-line-length-none",
              "-fconvert=big-endian", "-frecord-marker=4",
              "-fallow-argument-mismatch", "-fallow-invalid-boz"],
}
WRAPPERS = (HERE.parent / "wrf_diffusion_oracle" / "deformation_wrappers.F90",
            HERE.parent / "wrf_diffusion_oracle" / "vertical_tke_wrapper.F90",
            HERE / "km2_wrapper.F90")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(root: Path, out: Path, variant: str) -> Path:
    out = out.resolve() / variant
    out.mkdir(parents=True, exist_ok=True)
    for rel, pin in SOURCE_PINS.items():
        got = sha256(root / rel)
        if got != pin:
            raise ValueError(f"{rel} is not the pinned WRF {WRF_RELEASE} file "
                             f"({got})")
    bodies, spans = extract_routines(root / "dyn_em/module_diffusion_em.F",
                                     DIFFUSION_ROUTINES)
    extra = {}
    for rel, names in (("share/module_bc.F", ["set_physical_bc3d"]),
                       ("dyn_em/module_big_step_utilities_em.F", ["phy_prep"])):
        more, more_spans = extract_routines(root / rel, names)
        bodies += b"\n" + more
        extra[rel] = more_spans
    generated = out / "wrf_routines.F90"
    generated.write_bytes(
        b"module module_diffusion_em\nuse module_oracle_config\n"
        b"use module_model_constants\n"
        b"   INTEGER, PARAMETER            :: bdyzone = 4\ncontains\n"
        + bodies + b"\nend module module_diffusion_em\n")
    (out / "oracle_config.F90").write_text(CONFIG_SOURCE, encoding="utf-8",
                                           newline="\n")
    wrapper = out / "wrappers.F90"
    wrapper.write_bytes(b"\n".join(p.read_bytes() for p in WRAPPERS))
    flags = FLAGS[variant]
    commands = []
    for src, obj in ((out / "oracle_config.F90", "config.o"),
                     (root / "share/module_model_constants.F", "constants.o"),
                     (generated, "routines.o"), (wrapper, "wrapper.o")):
        command = ["gfortran", *flags, "-c", str(src), "-o", obj]
        commands.append(command)
        subprocess.run(command, cwd=out, check=True)
    command = ["gfortran", "-shared", "config.o", "constants.o", "routines.o",
               "wrapper.o", "-o", "oracle.so"]
    commands.append(command)
    subprocess.run(command, cwd=out, check=True)
    receipt = {
        "wrf_release": WRF_RELEASE, "wrf_commit": WRF_COMMIT,
        "variant": variant,
        "compiler": subprocess.check_output(["gfortran", "--version"],
                                            text=True).splitlines()[0],
        "libc": subprocess.check_output(["ldd", "--version"],
                                        text=True).splitlines()[0],
        "source_sha256": {rel: sha256(root / rel) for rel in SOURCE_PINS},
        "routines": spans, "extra_routines": extra,
        "wrapper_sha256": {p.name: sha256(p) for p in WRAPPERS},
        "generated_sha256": sha256(generated),
        "library_sha256": sha256(out / "oracle.so"),
        "commands": commands,
    }
    (out / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                            encoding="utf-8", newline="\n")
    return out / "oracle.so"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("wrf_root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--variant", choices=("noopt", "stock", "both"),
                    default="both")
    args = ap.parse_args()
    for variant in (("noopt", "stock") if args.variant == "both"
                    else (args.variant,)):
        print(build(args.wrf_root, args.out, variant))


if __name__ == "__main__":
    main()
