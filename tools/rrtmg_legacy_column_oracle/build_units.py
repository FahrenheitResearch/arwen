"""Assemble the WRF Fortran units the legacy RRTMG column oracle compiles.

Every physics statement comes from an upstream file, extracted verbatim by
routine name (the ``extract`` helper of tools/hrrr_radiation_driver_oracle):

* ``raddrv_extract.F90``: module ``wrf461_raddrv_extract`` holding WRF
  v4.6.1's ``radconst``, ``calc_coszen``, ``cal_cldfra1``,
  ``ozn_time_int`` and ``ozn_p_int`` (phys/module_radiation_driver.F) and
  ``lin_interpol2`` + ``oznini`` (phys/module_ra_cam_support.F), with the
  module variables oznini fills (module_ra_cam_support.F's
  ozmixin_save / lat_ozone_save / plev_ozone_save / levsiz_ozone_save /
  have_ozone, as tools/rrtmg_wrf461_oracle/ozn_build.sh declares them).
* ``aer_unit.F90`` for the aer_opt = 3 optics, one of two generations:
  - ``stock``: WRF v4.6.1's ``RSLF`` (phys/module_mp_thompson.F) in a
    stand-in ``module_mp_thompson`` and ``gt_aod`` (module_radiation_driver.F)
    in module ``wrf_aer_gt``; ``calc_aerosol_rrtmg_sw`` is compiled from the
    unmodified phys/module_ra_aerosol.F by build.sh.
  - ``fork``: the operational HRRR fork (NOAA-EMC/HRRR v4.1.21, sha256
    pinned in tools/hrrr_radiation_driver_oracle/build.py) ``RSLF``,
    ``module_ra_aerosol`` (whole file) and ``gt_aod`` (module ``wrf_aer_gt``).

usage: python build_units.py ARM WRF461_ROOT FORK_SOURCE_DIR BUILD_DIR
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "hrrr_build", HERE.parent / "hrrr_radiation_driver_oracle" / "build.py")
hrrr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hrrr)
extract = hrrr.extract

OZONE_STATE = """
      real, pointer :: ozmixin_save(:,:,:,:), lat_ozone_save(:), plev_ozone_save(:)
      integer :: levsiz_ozone_save=-1
      logical :: have_ozone=.false.
"""


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    arm, root, fork_dir, build = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    build.mkdir(parents=True, exist_ok=True)
    phys = root / "phys"
    driver = read(phys / "module_radiation_driver.F")
    cam = read(phys / "module_ra_cam_support.F")
    parts = ["module wrf461_raddrv_extract\n  implicit none\n", OZONE_STATE,
             "contains\n",
             extract(cam, "function", "lin_interpol2"),
             extract(cam, "subroutine", "oznini")]
    parts += [extract(driver, "subroutine", name) for name in
              ("radconst", "calc_coszen", "cal_cldfra1", "ozn_time_int",
               "ozn_p_int")]
    parts.append("end module wrf461_raddrv_extract\n")
    (build / "raddrv_extract.F90").write_text("".join(parts), encoding="utf-8")
    receipt = {"arm": arm,
               "wrf461": {name: sha(phys / name) for name in (
                   "module_radiation_driver.F", "module_ra_cam_support.F",
                   "module_ra_rrtmg_lw.F", "module_ra_rrtmg_sw.F",
                   "module_ra_aerosol.F", "module_mp_thompson.F")}}
    if arm in ("stock", "stock-o2"):
        thompson = read(phys / "module_mp_thompson.F")
        unit = ("module module_mp_thompson\n  implicit none\ncontains\n"
                + extract(thompson, "function", "RSLF")
                + "end module module_mp_thompson\n"
                + "module wrf_aer_gt\ncontains\n"
                + extract(driver, "subroutine", "gt_aod")
                + "end module wrf_aer_gt\n")
    elif arm in ("fork", "fork-cloud"):
        receipt["fork"] = hrrr.fetch(fork_dir)
        receipt["fork_commit"] = hrrr.COMMIT
        fdriver = read(fork_dir / "module_radiation_driver.F")
        faer = read(fork_dir / "module_ra_aerosol.F")
        fthompson = read(fork_dir / "module_mp_thompson.F")
        unit = ("module module_mp_thompson\n  implicit none\ncontains\n"
                + extract(fthompson, "function", "RSLF")
                + "end module module_mp_thompson\n"
                + faer + "\n"
                + "module wrf_aer_gt\ncontains\n"
                + extract(fdriver, "subroutine", "gt_aod")
                + "end module wrf_aer_gt\n")
    else:
        raise SystemExit(f"unknown arm {arm!r}")
    (build / "aer_unit.F90").write_text(unit, encoding="utf-8")
    receipt["raddrv_extract_sha256"] = sha(build / "raddrv_extract.F90")
    receipt["aer_unit_sha256"] = sha(build / "aer_unit.F90")
    (build / "units-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                              encoding="utf-8")


if __name__ == "__main__":
    main()
