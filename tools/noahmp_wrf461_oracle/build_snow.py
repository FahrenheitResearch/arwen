#!/usr/bin/env python3
"""Build and run the Noah-MP snow-leaf oracle against pinned WRF v4.6.1.

Written in Python rather than shell on purpose: this repository has
``core.autocrlf=true`` and no ``*.sh text eol=lf`` attribute, so a checked-out
shell script arrives with CRLF terminators and dies under WSL bash.  Python 3
reads its own source with universal newlines, and this script normalises the
Fortran sources it copies into the build directory, so the harness is
line-ending agnostic end to end.

Steps, in order, each of which must pass:

1. the pristine ``phys/module_sf_noahmplsm.F`` must hash to the pinned digest;
2. the accessibility lift must be textually nothing but ``private`` ->
   ``public `` (``snow_visibility_patch.py check``);
3. the pristine and patched module must compile to byte-identical object code
   (``snow_visibility_patch.py objects``);
4. the negative controls for both of those checks must fail as designed;
5. the leaves must consult no ``noahmp_parameters`` component beyond ``SSI``
   and ``SNOW_RET_FAC`` -- audited against the pristine source text, because
   the harness supplies only those two;
6. the fixture is generated;
7. a second build with ``-finit-real=snan -finit-integer=-2147483647`` must
   produce a byte-identical fixture, which proves no emitted value depends on
   an uninitialised local (DIVIDE in particular carries DZ/SWICE/SWLIQ/TSNO
   slots that are only conditionally assigned).

Usage::

    python3 build_snow.py WRF_SOURCE_ROOT BUILD_DIR [--install]

``--install`` copies the fixture over ``gpuwm/data/noahmp/oracle/noahmp-snow.csv``.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

sys.path.insert(0, str(HERE))
import snow_visibility_patch as vis  # noqa: E402

FFLAGS = ["-O0", "-cpp", "-ffree-form", "-ffree-line-length-none"]
INIT_FLAGS = ["-finit-real=snan", "-finit-integer=-2147483647", "-finit-logical=false"]

# Line span of the seven snow leaves in the pinned module, inclusive.
LEAF_SPAN = (6398, 7230)
ALLOWED_PARAM_COMPONENTS = {"SSI", "SNOW_RET_FAC"}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(
            f"command failed in {cwd}: {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}"
        )


def copy_lf(src: Path, dst: Path) -> None:
    """Copy normalising CRLF -> LF so a Windows checkout builds identically."""
    dst.write_bytes(src.read_bytes().replace(b"\r\n", b"\n"))


def audit_parameter_use(pristine: Path) -> set[str]:
    lines = pristine.read_bytes().decode("latin-1").splitlines()
    body = lines[LEAF_SPAN[0] - 1 : LEAF_SPAN[1]]
    # Confirm the span really is the seven leaves and nothing else.
    if "SUBROUTINE SNOWWATER" not in body[0]:
        raise SystemExit(f"leaf span start moved: {body[0]!r}")
    if "END SUBROUTINE SNOWH2O" not in body[-1]:
        raise SystemExit(f"leaf span end moved: {body[-1]!r}")
    used = set(re.findall(r"parameters%([A-Za-z_]\w*)", "\n".join(body)))
    extra = used - ALLOWED_PARAM_COMPONENTS
    if extra:
        raise SystemExit(
            "snow leaves consult noahmp_parameters components the harness does "
            f"not supply: {sorted(extra)}"
        )
    return used


def build_variant(build_dir: Path, tag: str, patched: Path, gecros: Path,
                  extra_flags: list[str]) -> Path:
    d = build_dir / tag
    d.mkdir(parents=True, exist_ok=True)
    flags = FFLAGS + extra_flags

    copy_lf(gecros, d / gecros.name)
    copy_lf(patched, d / "module_sf_noahmplsm.F")
    copy_lf(HERE / "stub_wrf_snow.F90", d / "stub_wrf_snow.F90")
    copy_lf(HERE / "run_snow.F90", d / "run_snow.F90")

    run(["gfortran", "-c", *flags, "stub_wrf_snow.F90"], d)
    run(["gfortran", "-c", *flags, gecros.name], d)
    run(["gfortran", "-c", *flags, "-I", ".", "module_sf_noahmplsm.F"], d)
    run(["gfortran", "-c", *flags, "-I", ".", "run_snow.F90"], d)
    run(["gfortran", "-o", "run_snow", "stub_wrf_snow.o",
         gecros.stem + ".o", "module_sf_noahmplsm.o", "run_snow.o"], d)

    csv = d / "noahmp-snow.csv"
    run(["./run_snow", csv.name], d)

    # The float32 EXP sweep the CUDA kernel is gated against.  gfortran lowers
    # REAL(4) EXP to expf@plt, so this is the live glibc symbol's own output.
    copy_lf(HERE / "run_snow_expf.F90", d / "run_snow_expf.F90")
    run(["gfortran", "-c", *flags, "run_snow_expf.F90"], d)
    run(["gfortran", "-o", "run_snow_expf", "run_snow_expf.o"], d)
    run(["./run_snow_expf", "noahmp-snow-expf.csv"], d)
    return csv


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        raise SystemExit("usage: build_snow.py WRF_SOURCE_ROOT BUILD_DIR [--install]")
    source_root = Path(argv[0]).resolve()
    build_dir = Path(argv[1]).resolve()
    install = "--install" in argv[2:]

    pristine = source_root / "phys" / "module_sf_noahmplsm.F"
    gecros = source_root / "phys" / "module_sf_gecros.F"
    for f in (pristine, gecros):
        if not f.is_file():
            raise SystemExit(f"missing pinned source: {f}")

    build_dir.mkdir(parents=True, exist_ok=True)

    # 1. pinned identity
    digest = sha256_file(pristine)
    if digest != vis.PRISTINE_SHA256:
        raise SystemExit(
            f"module_sf_noahmplsm.F is not the pinned file\n"
            f"  expected {vis.PRISTINE_SHA256}\n  got      {digest}"
        )
    print(f"[1] pristine module_sf_noahmplsm.F sha256 {digest}")

    # 2. accessibility lift, textual proof
    patched_path = build_dir / "module_sf_noahmplsm.patched.F"
    patched_bytes, lifted = vis.apply_patch(pristine.read_bytes())
    vis.check_patch(pristine.read_bytes(), patched_bytes)
    patched_path.write_bytes(patched_bytes)
    print(f"[2] lifted {lifted} `private ::` statements; text diff is exactly "
          f"private->public, file length unchanged")

    # 3. accessibility lift, generative proof
    gecros_dir = vis._prepare_gecros(gecros, build_dir / "objcmp")
    obj_digest = vis.compare_objects(
        pristine.read_bytes(), patched_bytes, build_dir / "objcmp", gecros_dir,
        vis.lifted_names(pristine.read_bytes()),
    )
    print(f"[3] object code: identical section sizes, identical non-.text "
          f"sections, identical .text disassembly, no symbol moved, no new "
          f"undefined symbol; {obj_digest}")

    # 4. negative controls
    vis.self_test(pristine, gecros)
    print("[4] negative controls pass")

    # 5. parameter-surface audit
    used = audit_parameter_use(pristine)
    print(f"[5] snow leaves consult noahmp_parameters components: {sorted(used)}")

    # 6. fixture
    csv = build_variant(build_dir, "fixture", patched_path, gecros, [])
    n_rows = len(csv.read_bytes().splitlines()) - 1
    print(f"[6] fixture: {n_rows} data rows, sha256 {sha256_file(csv)}")

    # 7. uninitialised-memory control
    csv_snan = build_variant(build_dir, "snan", patched_path, gecros, INIT_FLAGS)
    if csv.read_bytes() != csv_snan.read_bytes():
        raise SystemExit(
            "fixture changed under -finit-real=snan: an emitted value depends "
            "on an uninitialised local"
        )
    print("[7] identical under -finit-real=snan: no emitted value reads "
          "uninitialised memory")

    # provenance
    sums = build_dir / "snow-oracle-sha256sums.txt"
    lines = []
    for label, path in [
        ("phys/module_sf_noahmplsm.F", pristine),
        ("phys/module_sf_gecros.F", gecros),
        ("tools/noahmp_wrf461_oracle/snow_visibility_patch.py", HERE / "snow_visibility_patch.py"),
        ("tools/noahmp_wrf461_oracle/stub_wrf_snow.F90", HERE / "stub_wrf_snow.F90"),
        ("tools/noahmp_wrf461_oracle/run_snow.F90", HERE / "run_snow.F90"),
        ("tools/noahmp_wrf461_oracle/run_snow_expf.F90", HERE / "run_snow_expf.F90"),
        ("tools/noahmp_wrf461_oracle/build_snow.py", HERE / "build_snow.py"),
        ("gpuwm/data/noahmp/oracle/noahmp-snow.csv", csv),
        ("gpuwm/data/noahmp/oracle/noahmp-snow-expf.csv", csv.parent / "noahmp-snow-expf.csv"),
    ]:
        lines.append(f"{sha256_file(path)}  {label}")
    lines.append(f"{obj_digest}  module_sf_noahmplsm.o (pristine == patched)")
    cc = subprocess.run(["gfortran", "--version"], capture_output=True, text=True)
    lines.append(cc.stdout.splitlines()[0])
    sums.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    if install:
        out_dir = REPO / "gpuwm" / "data" / "noahmp" / "oracle"
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in ("noahmp-snow.csv", "noahmp-snow-expf.csv"):
            dest = out_dir / name
            dest.write_bytes((csv.parent / name).read_bytes())
            print(f"installed -> {dest}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
