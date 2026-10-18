"""WRF v4.6.1's RRTMG wrappers with the operational branch's cloud rules.

``rrtmg_cloud_optics_form = "noaa_wrf39"`` is WOOF's name for the two
wrapper cloud rules the operational RAP/HRRR branch (NOAA-EMC/HRRR v4.1.21,
WRFV3.9) carries on top of an otherwise legacy RRTMG
(gpuwm.core.rrtmg_legacy_prep.cloud_form_is_fork):

1. RRTMG_SWRAD, unsized cloudy liquid (radius at or below 2.5 um): 9.6 um
   over water and 5.4 um over land (fork module_ra_rrtmg_sw.F:10464-10474),
   where v4.6.1 has 10.5 and 7.5 (:10781-10789).  RRTMG_LWRAD keeps 10.5
   and 7.5 in both.
2. Both wrappers' snow path: all of the snow mass, ``snow_mass_factor =
   1.0``, reduced to (130/res)**2 above 130 um (fork SW :10744-10747, LW
   :12347-12349), where v4.6.1 keeps 99 percent (SW :11057-11061, LW
   :12517-12521; its 1 percent added to a ``gicewp`` that nothing reads
   afterwards).

The oracle arm for the form is therefore v4.6.1's two files with exactly
those statements replaced by the branch's, both texts checked verbatim
(each replaced span must occur exactly once, and each branch span must
occur in the branch file).  Everything else in the two files is v4.6.1's,
as WOOF's legacy RRTMG is.

    python cloud_form_transplant.py WRF461_ROOT FORK_SRC OUT_DIR

writes OUT_DIR/module_ra_rrtmg_{lw,sw}.F and OUT_DIR/cloud-form-receipt.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

SW_RADII_461 = """                  if (recloud1D(ncol,K).LE.2.5.AND.cldfra3d(i,k,j).gt.0. &
     &                            .AND. (XLAND(I,J)-1.5).GT.0.) then      !--- Ocean
                     recloud1D(ncol,K) = 10.5
                  elseif (recloud1D(ncol,K).LE.2.5.AND.cldfra3d(i,k,j).gt.0. &
     &                            .AND. (XLAND(I,J)-1.5).LT.0.) then      !--- Land
                     recloud1D(ncol,K) = 7.5
                  endif"""
SW_RADII_FORK = """                  if (recloud1D(ncol,K).LE.2.5.AND.cldfra3d(i,k,j).gt.0. &
     &                            .AND. (XLAND(I,J)-1.5).GT.0.) then      !--- Ocean
                     recloud1D(ncol,K) = 9.6
                  elseif (recloud1D(ncol,K).LE.2.5.AND.cldfra3d(i,k,j).gt.0. &
     &                            .AND. (XLAND(I,J)-1.5).LT.0.) then      !--- Land
                     recloud1D(ncol,K) = 5.4
                  endif"""
SNOW_FACTOR_461 = """                 snow_mass_factor = 0.99        ! Assume 1% of snow overlaps the cloud ice category
                 gicewp = gicewp + (qs1d(k)*(1.0-snow_mass_factor) * pdel(ncol,k)*100.0 / gravmks * 1000.0)"""
SNOW_FACTOR_FORK = "                 snow_mass_factor = 1.0"
SNOW_CAP_461 = """                     snow_mass_factor = MIN(snow_mass_factor,                       &
     &                         (130.0/resnow1d(ncol,k))*(130.0/resnow1d(ncol,k)))"""
SNOW_CAP_FORK = """                     snow_mass_factor = (130.0/resnow1d(ncol,k))*(130.0/resnow1d(ncol,k))"""

RULES = {
    "module_ra_rrtmg_sw.F": ((SW_RADII_461, SW_RADII_FORK),
                             (SNOW_FACTOR_461, SNOW_FACTOR_FORK),
                             (SNOW_CAP_461, SNOW_CAP_FORK)),
    "module_ra_rrtmg_lw.F": ((SNOW_FACTOR_461, SNOW_FACTOR_FORK),
                             (SNOW_CAP_461, SNOW_CAP_FORK)),
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")


def transplant(wrf_root: Path, fork_src: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    receipt = {"files": {}}
    for name, rules in RULES.items():
        stock = _read(wrf_root / "phys" / name)
        fork = _read(fork_src / "phys" / name)
        text = stock
        for old, new in rules:
            if stock.count(old) != 1:
                raise ValueError(f"{name}: v4.6.1 span occurs {stock.count(old)} times")
            # The branch text is matched with its trailing whitespace removed
            # per line (the branch file has none on these lines either).
            if new not in fork:
                raise ValueError(f"{name}: branch span not found in the branch file")
            text = text.replace(old, new)
        (out / name).write_text(text, encoding="utf-8")
        receipt["files"][name] = {
            "wrf461_sha256": hashlib.sha256((wrf_root / "phys" / name).read_bytes()).hexdigest(),
            "fork_sha256": hashlib.sha256((fork_src / "phys" / name).read_bytes()).hexdigest(),
            "transplanted_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "spans_replaced": len(rules)}
    (out / "cloud-form-receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    return receipt


if __name__ == "__main__":
    print(json.dumps(transplant(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])),
                     indent=1))
