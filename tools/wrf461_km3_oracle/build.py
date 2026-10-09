"""Compile the WRF 4.6.1 km_opt=3 chain as one column-oracle library.

Unmodified routine bodies are extracted byte-for-byte from a pristine WRF
v4.6.1 tree (tag v4.6.1, commit d66e442f) and compiled with the C ABI storage
adapters already used by tools/wrf_diffusion_oracle (deformation, horizontal
outer driver, vertical outer driver).  Those adapters only lay out storage and
call WRF; no arithmetic is stubbed.

Two arithmetic variants:

* ``oracle`` (default): gfortran -O0 -ffp-contract=off -fno-tree-vectorize.
  Scalar libm calls, no contraction.  This is the column-oracle reference.
* ``--stock``: WRF's own GNU flags (-O2 -ftree-vectorize -funroll-loops).
  x86-64 baseline has no FMA; vectorised loops may call libmvec.  Recorded
  so the report can say whether stock WRF arithmetic differs from the oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "wrf_diffusion_oracle"
sys.path.insert(0, str(OLD))
from build_common import CONFIG_SOURCE, extract_routines  # noqa: E402

WRF_RELEASE = "4.6.1"
WRF_COMMIT = "d66e442fccc04111067e29274c9f9eaccc3cef28"
SOURCE_PINS = {
    "dyn_em/module_diffusion_em.F": "a7d4570c97e51c635e86a0dbd628c6846457ac5b93d5a7af798b118c7d8d2d54",
    "dyn_em/module_big_step_utilities_em.F": "8f0649b458fceabd5c31c87ec9f266840fb164e2629322904a3f6ee04c72331a",
    "share/module_model_constants.F": "5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062",
    "share/module_bc.F": "61b9235004b2a7799faabaa928276af8a7ef2e4672619c8ad120c857461301ad",
}
DIFFUSION_ROUTINES = (
    # cal_deform_and_div + calculate_km_kh (km_opt=3 -> calculate_N2, smag_km)
    "cal_deform_and_div", "calculate_km_kh", "calculate_N2", "cal_dampkm",
    "isotropic_km", "smag_km", "smag2d_km", "tke_km", "calc_l_scale",
    "pthl", "pu", "compute_diff_metrics",
    # horizontal_diffusion_2 and its leaves
    "horizontal_diffusion_2", "horizontal_diffusion_u_2", "horizontal_diffusion_v_2",
    "horizontal_diffusion_w_2", "horizontal_diffusion_s", "cal_titau_11_22_33",
    "cal_titau_12_21", "cal_titau_13_31", "cal_titau_23_32",
    # vertical_diffusion_2 and its leaves
    "vertical_diffusion_2", "vertical_diffusion_u_2", "vertical_diffusion_v_2",
    "vertical_diffusion_w_2", "vertical_diffusion_s",
)
WRAPPERS = ("deformation_wrappers.F90", "horizontal_driver_wrapper.F90",
            "vertical_driver_complete.F90")
ORACLE_FLAGS = ["-O0", "-g", "-fPIC", "-cpp", "-ffree-form", "-ffree-line-length-none",
                "-ffp-contract=off", "-fno-tree-vectorize", "-fcheck=bounds"]
STOCK_FLAGS = ["-O2", "-ftree-vectorize", "-funroll-loops", "-fPIC", "-cpp", "-ffree-form",
               "-ffree-line-length-none", "-fconvert=big-endian", "-frecord-marker=4"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(wrf_root, output, *, stock=False):
    wrf_root = Path(wrf_root)
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for rel, pin in SOURCE_PINS.items():
        got = sha256(wrf_root / rel)
        if got != pin:
            raise ValueError(f"{rel} is not the pinned WRF {WRF_RELEASE} file ({got})")
    diffusion = wrf_root / "dyn_em/module_diffusion_em.F"
    bodies, spans = extract_routines(diffusion, DIFFUSION_ROUTINES)
    extra = {}
    for rel, names in (("share/module_bc.F", ["set_physical_bc3d"]),
                       ("dyn_em/module_big_step_utilities_em.F", ["phy_prep"])):
        extra_bodies, extra_spans = extract_routines(wrf_root / rel, names)
        bodies += b"\n" + extra_bodies
        extra[rel] = {"sha256": sha256(wrf_root / rel), "routines": extra_spans}
    generated = out / "wrf_routines.F90"
    generated.write_bytes(b"module module_diffusion_em\nuse module_oracle_config\n"
                          b"use module_model_constants\n"
                          b"   INTEGER, PARAMETER            :: bdyzone = 4\n"
                          b"contains\n" + bodies + b"\nend module module_diffusion_em\n")
    (out / "oracle_config.F90").write_text(CONFIG_SOURCE, encoding="utf-8", newline="\n")
    wrapper = out / "km3_wrappers.F90"
    wrapper.write_bytes(b"\n".join((OLD / name).read_bytes() for name in WRAPPERS))
    flags = STOCK_FLAGS if stock else ORACLE_FLAGS
    commands = []
    objects = []
    for src, obj in ((out / "oracle_config.F90", "oracle_config.o"),
                     (wrf_root / "share/module_model_constants.F", "constants.o"),
                     (generated, "routines.o"), (wrapper, "wrappers.o")):
        command = ["gfortran", *flags, "-c", str(src), "-o", obj]
        subprocess.run(command, cwd=out, check=True)
        commands.append(command)
        objects.append(obj)
    command = ["gfortran", "-shared", *objects, "-o", "oracle.so"]
    subprocess.run(command, cwd=out, check=True)
    commands.append(command)
    compiler = subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0]
    libc = subprocess.run(["ldd", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    linked = subprocess.run(["ldd", str(out / "oracle.so")], capture_output=True, text=True).stdout
    receipt = {"wrf_release": WRF_RELEASE, "wrf_commit": WRF_COMMIT, "variant": "stock" if stock else "oracle",
               "compiler": compiler, "libc": libc, "flags": flags,
               "links_libmvec": "libmvec" in linked,
               "commands": [[Path(a).name if a.startswith("/") else a for a in c] for c in commands],
               "routines": spans, "extra_sources": extra,
               "source_sha256": SOURCE_PINS,
               "wrappers": {name: sha256(OLD / name) for name in WRAPPERS},
               "generated_sha256": sha256(generated), "library_sha256": sha256(out / "oracle.so")}
    (out / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return out / "oracle.so"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("wrf_root", type=Path, help="pristine WRF v4.6.1 source tree")
    p.add_argument("output", type=Path)
    p.add_argument("--stock", action="store_true")
    a = p.parse_args()
    print(build(a.wrf_root, a.output, stock=a.stock))
