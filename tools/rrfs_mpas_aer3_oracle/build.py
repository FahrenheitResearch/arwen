"""Build the NOAA MPAS fork's aer_opt = 3 shortwave aerosol oracle.

The reference for ``rrtmg_aer3_type_source = "landmask"`` is the MPAS fork
NOAA's RRFS v2 runs, RRFSx/MPAS-Model at the commit the rrfs-workflow pins
(1d6e370e7b43a332074809f170532a216c2212fb, 8.3.1-noaa).  Its physics
files are fetched from that immutable commit and checked against the
SHA-256 values below, then compiled verbatim with gfortran at -O0 without
floating-point contraction, so every statement is the single rounded IEEE
float32 operation the engine twins perform:

* physics_wrf/module_mp_thompson_aerosols.F  gt_aod
* physics_wrf/module_ra_rrtmg_sw_aerosols.F  calc_aerosol_rrtmg_sw and its
  helpers, with the rural, urban and maritime Shettle-Fenn tables
* mpas_atmphys_functions.F                    RSLF (the Thompson polynomial)

The only additions are stand-ins for MPAS framework modules the three
files USE (kind types, the log, the fatal-error utility): they hold no
numerical work.  oracle_aer3_mpas.f90 calls the routines the way
mpas_atmphys_driver_radiation_sw.F does; this script checks that the
driver's land/water type lines and the aerosol PARAMETERs of
mpas_atmphys_vars.F are the text the oracle restates.

Usage: python build.py SOURCE_DIRECTORY BUILD_DIRECTORY
Missing source files are fetched from the public commit.  MPAS is
distributed under its BSD-style licence (LICENSE in the MPAS-Model
repository); the WRF-derived physics files carry the WRF public-domain
notice reproduced in ../../licenses/LICENSE-WRF-public-domain.txt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.request import urlopen

COMMIT = "1d6e370e7b43a332074809f170532a216c2212fb"
REPO = "RRFSx/MPAS-Model"
BASE = (f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/"
        "src/core_atmosphere/physics/")
PIN = {
    "mpas_atmphys_functions.F":
        "2cf3763c2781468ed59134acecfeb04a903859356ab30c18d48e595e5688e37e",
    "physics_wrf/module_mp_thompson_aerosols.F":
        "ffdd347c5396d7fdcf2784c6c3fd7c453008afd31ab297414c18e5a402b4d248",
    "physics_wrf/module_ra_rrtmg_sw_aerosols.F":
        "4985e49f8225fdca6a40ae253cfe4ed2dd86c3fc1a46469cb3369fd1fb972cfd",
    "mpas_atmphys_driver_radiation_sw.F":
        "cf27cfa597771c3184c0b46fd4b7d7edcedcedbd966a3ed2dd82bc326ee5aa56",
    "mpas_atmphys_vars.F":
        "2ba656a9b7ad669b6445f9f366dd1072ae48ac5bf072ab84d23ae53d981dc16d",
}

#: Text the oracle driver restates, which must be in the pinned files.
DRIVER_ANCHORS = (
    "if(xland_p(i,j)==1._RKIND) then",
    "taer_type_p(i,j) = 1",
    "elseif(xland_p(i,j)==2._RKIND) then",
    "taer_type_p(i,j) = 3",
    "tauaer_p(i,k,j,n) = 0._RKIND",
    "ssaaer_p(i,k,j,n) = 1._RKIND",
    "asyaer_p(i,k,j,n) = 0._RKIND",
    "taod5502d_p(i,j) = taod5502d_p(i,j) + taod5503d_p(i,k,j)",
)
VARS_ANCHORS = (
    "integer,parameter:: taer_aod550_opt = 2",
    "integer,parameter:: taer_angexp_opt = 3",
    "integer,parameter:: taer_ssa_opt    = 3",
    "integer,parameter:: taer_asy_opt    = 3",
    "real(kind=RKIND),parameter:: aer_aod550_val = 0.12",
    "real(kind=RKIND),parameter:: aer_angexp_val = 1.3",
    "real(kind=RKIND),parameter:: aer_ssa_val    = 0.85",
    "real(kind=RKIND),parameter:: aer_asy_val    = 0.9",
)

HERE = Path(__file__).resolve().parent

STUBS = """module mpas_kind_types
integer, parameter :: RKIND = selected_real_kind(6)
end module
module mpas_derived_types
integer, parameter :: MPAS_LOG_ERR = 3
end module
module mpas_log
contains
subroutine mpas_log_write(message, messageType, intArgs, realArgs)
character(len=*), intent(in) :: message
integer, intent(in), optional :: messageType
integer, intent(in), optional :: intArgs(:)
real, intent(in), optional :: realArgs(:)
end subroutine
end module
module mpas_atmphys_utilities
contains
subroutine physics_error_fatal(message)
character(len=*), intent(in) :: message
print *, 'physics_error_fatal: ', trim(message)
stop 1
end subroutine
subroutine physics_message(message)
character(len=*), intent(in) :: message
end subroutine
end module
"""

ORDER = ("mpas_atmphys_functions.F",
         "physics_wrf/module_mp_thompson_aerosols.F",
         "physics_wrf/module_ra_rrtmg_sw_aerosols.F")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(source_dir: Path) -> dict[str, str]:
    receipt = {}
    for name, want in PIN.items():
        path = source_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file():
            with urlopen(BASE + name) as response:
                path.write_bytes(response.read())
        got = sha256_of(path)
        if got != want:
            raise SystemExit(f"{name}: sha256 {got} != pinned {want}")
        receipt[name] = got
    return receipt


def check_anchors(source_dir: Path) -> None:
    def squash(text):
        return " ".join(text.split())
    driver = squash((source_dir / "mpas_atmphys_driver_radiation_sw.F").read_text(
        encoding="utf-8", errors="replace"))
    for line in DRIVER_ANCHORS:
        if squash(line) not in driver:
            raise SystemExit(f"driver anchor missing: {line}")
    params = squash((source_dir / "mpas_atmphys_vars.F").read_text(
        encoding="utf-8", errors="replace"))
    for line in VARS_ANCHORS:
        if squash(line) not in params:
            raise SystemExit(f"mpas_atmphys_vars anchor missing: {line}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("build_dir", type=Path)
    args = parser.parse_args()
    receipt = fetch(args.source_dir)
    check_anchors(args.source_dir)
    build = args.build_dir
    build.mkdir(parents=True, exist_ok=True)
    (build / "mpas_stubs.F90").write_text(STUBS, encoding="utf-8")
    flags = ["-O0", "-ffp-contract=off", "-ffree-form",
             "-ffree-line-length-none", "-cpp"]
    version = subprocess.run(["gfortran", "--version"], capture_output=True,
                             text=True, check=True).stdout.splitlines()[0]
    units = [build / "mpas_stubs.F90"]
    units += [args.source_dir / name for name in ORDER]
    units.append(HERE / "oracle_aer3_mpas.f90")
    subprocess.run(["gfortran", *flags, "-J", str(build), "-o",
                    str(build / "oracle_aer3_mpas"), *map(str, units)],
                   check=True)
    (build / "receipt.json").write_text(json.dumps({
        "repository": REPO, "commit": COMMIT, "sources": receipt,
        "gfortran": version, "flags": flags,
        "driver_sha256": sha256_of(HERE / "oracle_aer3_mpas.f90"),
        "stubs_sha256": sha256_of(build / "mpas_stubs.F90")},
        indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"commit": COMMIT, "gfortran": version}, indent=2))


if __name__ == "__main__":
    main()
