"""Build the operational HRRR radiation-driver oracle from the fork's own text.

The reference for swint_opt = 1 and aer_opt = 3 is the WRF fork operational
HRRR v4 runs: NOAA-EMC/HRRR tag v4.1.21, whose ``sorc/hrrr_wrfarw.fd/WRFV3.9``
physics files are byte-identical at the commit this tree already pins for
its vertical-advection oracle (tools/ieva_wrf_oracle/legacy_build.py).  The
routines are extracted here by name from the downloaded files (sha256
pinned below), wrapped in modules with stand-ins for ``module_wrf_error``
(messages and a fatal stop, no numerical work) and ``module_mp_thompson``
(the fork's own ``RSLF``, extracted verbatim), and compiled with gfortran at
-O0 and no floating-point contraction, so every statement is the single
rounded IEEE float32 operation the engine twins perform.

Usage: python build.py SOURCE_DIRECTORY BUILD_DIRECTORY
Missing source files are fetched from the immutable public commit below.
The WRF notice is reproduced in ../../licenses/LICENSE-WRF-public-domain.txt.

Products in BUILD_DIRECTORY: ``oracle_swint`` (tools/.../oracle_swint.f90,
the swint_opt = 1 routines and calc_coszen) and ``oracle_aer3``
(oracle_aer3.f90, gt_aod + calc_aerosol_rrtmg_sw), each driven by its
``run_*_oracle.py`` which writes the fixture under tests/data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.request import urlopen

COMMIT = "40ee6058c2fc6624cbfbbe8cf1c20c59e6a45827"
TAG = "v4.1.21"
BASE = (f"https://raw.githubusercontent.com/NOAA-EMC/HRRR/{COMMIT}/"
        "sorc/hrrr_wrfarw.fd/WRFV3.9/phys/")
PIN = {
    "module_radiation_driver.F":
        "7464639e53f6f40b4810943ee9fa21a40b0f53f7525c08cc13b84ae4c7d82801",
    "module_ra_aerosol.F":
        "9931482abac91768fd00674c23b8c953e69ce49fa026c2c026b123845785dc42",
    "module_mp_thompson.F":
        "4d60011188443eb432294f7693beb64bdbc8f812541a15c7800013060c877283",
}

HERE = Path(__file__).resolve().parent

STUBS = """module module_wrf_error
character(len=512) :: wrf_err_message
contains
logical function wrf_at_debug_level(level)
integer, intent(in) :: level
wrf_at_debug_level = .false.
end function
end module
! WRF's frame routines are external to the physics modules: the aerosol
! module calls them without a USE, so they are provided the same way.
subroutine wrf_debug(level, message)
integer, intent(in) :: level
character(len=*), intent(in) :: message
end subroutine
subroutine wrf_message(message)
character(len=*), intent(in) :: message
end subroutine
subroutine wrf_error_fatal(message)
character(len=*), intent(in) :: message
print *, 'wrf_error_fatal: ', trim(message)
stop 1
end subroutine
"""

DRIVER_ROUTINES = ("radconst", "calc_coszen", "update_swinterp_parameters",
                   "interp_sw_radiation", "gt_aod")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(source_dir: Path) -> dict[str, str]:
    source_dir.mkdir(parents=True, exist_ok=True)
    receipt = {}
    for name, want in PIN.items():
        path = source_dir / name
        if not path.is_file():
            with urlopen(BASE + name) as response:
                path.write_bytes(response.read())
        got = sha256_of(path)
        if got != want:
            raise SystemExit(f"{name}: sha256 {got} != pinned {want}")
        receipt[name] = got
    return receipt


def extract(text: str, kind: str, name: str) -> str:
    """The ``kind name ... end kind name`` block, verbatim, case-insensitive."""
    head = re.compile(rf"^[ \t]*(?:real[ \t]+)?{kind}[ \t]+{name}[ \t]*\(",
                      re.IGNORECASE | re.MULTILINE)
    tail = re.compile(rf"^[ \t]*end[ \t]+{kind}[ \t]+{name}[ \t]*$",
                      re.IGNORECASE | re.MULTILINE)
    starts = [m.start() for m in head.finditer(text)]
    if len(starts) != 1:
        raise SystemExit(f"{kind} {name}: found {len(starts)} definitions")
    end = tail.search(text, starts[0])
    if end is None:
        raise SystemExit(f"{kind} {name}: no end statement")
    return text[starts[0]:end.end()] + "\n"


def assemble(source_dir: Path) -> str:
    driver = (source_dir / "module_radiation_driver.F").read_text(
        encoding="utf-8", errors="replace")
    aerosol = (source_dir / "module_ra_aerosol.F").read_text(
        encoding="utf-8", errors="replace")
    thompson = (source_dir / "module_mp_thompson.F").read_text(
        encoding="utf-8", errors="replace")
    parts = [STUBS, "module module_mp_thompson\ncontains\n",
             extract(thompson, "function", "RSLF"), "end module\n", aerosol,
             "module hrrr_radiation_driver_oracle\nuse module_wrf_error\n"
             "contains\n"]
    parts += [extract(driver, "subroutine", name) for name in DRIVER_ROUTINES]
    parts.append("end module\n")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("build_dir", type=Path)
    args = parser.parse_args()
    receipt = fetch(args.source_dir)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    unit = args.build_dir / "wrf_extract.F90"
    unit.write_text(assemble(args.source_dir), encoding="utf-8")
    flags = ["-O0", "-ffp-contract=off", "-ffree-form", "-ffree-line-length-none",
             "-cpp", "-DEM_CORE=1"]
    version = subprocess.run(["gfortran", "--version"], capture_output=True,
                             text=True, check=True).stdout.splitlines()[0]
    for program in ("oracle_swint", "oracle_aer3"):
        src = HERE / f"{program}.f90"
        if not src.is_file():
            continue
        subprocess.run(["gfortran", *flags, "-J", str(args.build_dir),
                        "-o", str(args.build_dir / program), str(unit),
                        str(src)], check=True)
    (args.build_dir / "receipt.json").write_text(json.dumps({
        "commit": COMMIT, "tag": TAG, "sources": receipt,
        "gfortran": version, "flags": flags,
        "extract_sha256": sha256_of(unit)}, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({"commit": COMMIT, "gfortran": version}, indent=2))


if __name__ == "__main__":
    main()
