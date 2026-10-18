"""Build and run the SPP column oracle against unmodified WRF v4.6.1 physics.

CPU only.  Run on the oracle platform of record (gfortran 13.3.0, glibc
2.39), under nice, with thread counts capped:

    python tools/spp_wrf461_column_oracle/build.py \
        --wrf-source <WRF-4.6.1>/phys --output <new directory> [--steps 8]

Then ``pack.py`` turns the CSV captures into the test fixture and
``compare.py`` grades WOOF's GPU kernels against them.

The WRF modules are byte-pinned (the same pins as tools/spp_wrf461_oracle)
and compiled -O0 -ffp-contract=off.  The receipt records the compiler, the C
library, every command, and the sha256 of every generated source and
capture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from spp_column_sources import write_sources  # noqa: E402


def _pins() -> dict[str, str]:
    """The v4.6.1 byte pins of tools/spp_wrf461_oracle/build.py, not a copy."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "spp_wrf461_oracle_build", ROOT / "tools" / "spp_wrf461_oracle" / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.PINS)


PINS = _pins()

FLAGS = ["gfortran", "-O0", "-ffp-contract=off", "-cpp", "-ffree-form",
         "-ffree-line-length-none"]
#: Pattern-table offsets for the two GF runs: every column gets a different
#: closure pattern in each.
GF_OFFSETS = (0, 9)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrf-source", type=Path, required=True,
                        help="WRF v4.6.1 phys/ directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name, pin in PINS.items():
        source = args.wrf_source / name
        if sha256(source) != pin:
            raise SystemExit(f"WRF source bytes differ from the v4.6.1 pin: {name}")
        shutil.copy2(source, out / name)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", GPUWM_NO_LOCAL_GPU="1")
    commands: list[list[str]] = []

    def run(command):
        commands.append(list(map(str, command)))
        subprocess.run(list(map(str, command)), cwd=out, env=env, check=True)

    shutil.copy2(ROOT / "tools" / "mynn_pbl_wrf461_oracle" / "stub_wrf.F90", out / "stub_wrf.F90")
    shutil.copy2(ROOT / "tools" / "gf_wrf461_oracle" / "gf_cases.F90", out / "gf_cases.F90")
    generated = write_sources(out)
    for name in ("stub_wrf.F90", "module_bl_mynn_common.F", "module_bl_mynn.F",
                 "module_gfs_machine.F", "module_gfs_physcons.F",
                 "module_cu_gf_deep.F", "module_cu_gf_sh.F",
                 "module_cu_gf_wrfdrv.F", "gf_cases.F90",
                 "run_mynn_spp_columns.F90", "run_gf_spp_columns.F90"):
        run(FLAGS + ["-c", name])
    run(["gfortran", "-o", "run_mynn_spp_columns", "stub_wrf.o",
         "module_bl_mynn_common.o", "module_bl_mynn.o", "run_mynn_spp_columns.o"])
    run(["gfortran", "-o", "run_gf_spp_columns", "module_gfs_machine.o",
         "module_gfs_physcons.o", "module_cu_gf_deep.o", "module_cu_gf_sh.o",
         "module_cu_gf_wrfdrv.o", "gf_cases.o", "run_gf_spp_columns.o"])
    # libmvec guard: an -O0 object must not call a SIMD math entry point.
    for obj in sorted(out.glob("*.o")):
        symbols = subprocess.check_output(["nm", "-u", obj.name], cwd=out, text=True)
        if "_ZGV" in symbols:
            raise SystemExit(f"{obj.name} calls libmvec; the -O0 reference must be scalar")
    for mixlength in (1, 2):
        run(["./run_mynn_spp_columns", f"mynn-spp-ml{mixlength}.csv", mixlength, args.steps])
    for offset in GF_OFFSETS:
        run(["./run_gf_spp_columns", f"gf-spp-o{offset}-levels.csv",
             f"gf-spp-o{offset}-surface.csv", f"gf-spp-o{offset}-consistency.csv", offset])
    receipt = {
        "wrf_version": "4.6.1",
        "source_pins": PINS,
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "libc": list(platform.libc_ver()),
        "flags": FLAGS[1:],
        "steps": args.steps,
        "gf_offsets": list(GF_OFFSETS),
        "commands": commands,
        "generated_sources": {name: hashlib.sha256(text.encode()).hexdigest()
                              for name, text in generated.items()},
        "files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                  for path in sorted(out.iterdir()) if path.suffix == ".csv"},
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"SPP column oracle written to {out}")


if __name__ == "__main__":
    main()
