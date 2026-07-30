#!/usr/bin/env python3
"""Structural checks for the unmodified-WRF RUC ``snowtemp`` contract oracle.

``oracle/snowtemp.csv`` pins six snow regimes and the gpuwm port reproduces
all of them bit for bit, but a mutation study over that port showed the
fixture cannot detect 72 of its 344 argument read sites being wrong, and
leaves six reachable ``if`` arms unexecuted.  ``oracle/snowtemp_contract.csv``
adds the regimes that close those gaps.  This validator checks that each
regime really entered the branch it was built for -- the mutation study
itself lives in
``tests/test_ruc.py::test_snow_temperature_step_pins_every_argument``.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np


CASES = (
    "melt_dense_dry_evap",
    "melt_trace_all_evap",
    "melt_blended_trace",
    "bottom_melt_2layer",
    "bottom_melt_trace",
    "lowdens_new_on_aged",
    "melt_dense_moist",
    "melt_hot_surface_1layer",
)
NCASE = len(CASES)
NZS = 9
XLMELT = 3.35e5


def main(path: str) -> None:
    with Path(path).open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != NCASE * NZS:
        raise SystemExit(f"expected {NCASE * NZS} rows, got {len(rows)}")
    cases = tuple(dict.fromkeys(row["case"] for row in rows))
    if cases != CASES:
        raise SystemExit(f"unexpected case inventory {cases!r}")
    numeric = tuple(key for key in rows[0] if key != "case")
    values = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float32)
        .reshape(NCASE, NZS)
        for name in numeric
    }
    if not all(np.all(np.isfinite(field)) for field in values.values()):
        raise SystemExit("RUC snowtemp contract oracle has non-finite output")
    col = {name: field[:, 0] for name, field in values.items()}

    # --- the same physical bounds the six-regime fixture is held to --------
    for name in ("tso_before", "tso_after", "soilt_after", "soilt1_after"):
        field = values[name]
        if not np.all((field > 180.0) & (field < 330.0)):
            raise SystemExit(f"{name} left the plausible temperature range")
    if not np.all(col["snwe_after"] >= 0.0):
        raise SystemExit("snowtemp produced negative snow water equivalent")
    if not np.all(col["snhei_after"] >= 0.0):
        raise SystemExit("snowtemp produced negative snow depth")
    if not np.all((col["rhosn_after"] >= 58.8) & (col["rhosn_after"] <= 500.0)):
        raise SystemExit("snow density left the Koren 58.8-500 kg/m^3 band")
    if not np.all((col["beta_after"] > 0.0) & (col["beta_after"] <= 1.0)):
        raise SystemExit("sublimation efficiency beta left (0,1]")
    for name in ("smelt", "snoh", "rsm", "qvg_after", "qsg_after", "qcg_after"):
        if not np.all(col[name] >= 0.0):
            raise SystemExit(f"{name} went negative")
    if not np.all((col["tsnav_after"] > -100.0) & (col["tsnav_after"] < 60.0)):
        raise SystemExit("snow-pack mean temperature left the plausible range")
    if not np.allclose(
        col["snheiprint"], col["snweprint"] * 1.0e3 / col["rhosn_after"],
        rtol=1e-5, atol=1e-9,
    ):
        raise SystemExit("reported snow depth is inconsistent with its density")
    if not np.allclose(col["snweprint"], col["snwe_after"], rtol=1e-6, atol=1e-12):
        raise SystemExit("snweprint does not mirror the updated snwe")
    if not all(np.any(values["tso_after"][n] != values["tso_before"][n])
               for n in range(NCASE)):
        raise SystemExit("snowtemp failed to evolve every temperature profile")

    snth = col["snth"]
    deltsn = col["deltsn"]
    snhei = col["snhei_before"]
    melted = col["smelt"] > 0.0

    # --- 1 evaporation half of the melt budget (:5436-5450) ---------------
    # The condensation half at :5427 writes dew = -epot; the evaporation half
    # leaves dew alone and is the only reader of tranf.  epot > 0 there means
    # qvatm < qsg.
    if not (snth[0] <= snhei[0] <= deltsn[0] + snth[0]):
        raise SystemExit("regime 1 is not a one-layer pack")
    if not (melted[0] and col["snoh"][0] > 0.0):
        raise SystemExit("regime 1 did not melt")
    if not (col["qvatm"][0] < col["qsg_after"][0]
            and col["dew_after"][0] == col["dew_before"][0]):
        raise SystemExit("regime 1 did not take the evaporation half")
    # :5499 exempts rhosn >= 350 with no new snow from the Egglston limiter.
    if not (col["rhosn_before"][0] >= 350.0 and col["newsnow"][0] == 0.0):
        raise SystemExit("regime 1 is not the dense-snow exemption")
    if not col["smelt"][0] * XLMELT * 1.0e3 > 18.8:
        raise SystemExit("regime 1 was Egglston limited after all")
    if not (0.0 < col["snowfrac"][0] < 1.0):
        raise SystemExit("regime 1 must carry partial snow cover")
    # :5533 keeps no liquid in dense snow.
    if not col["rsm"][0] == 0.0:
        raise SystemExit("regime 1 retained liquid in a dense pack")

    # --- 2 all the remaining snow can evaporate (:5483-5491) --------------
    # beta enters at 1 and is rewritten to snwepr/(epot*ras*delt) < 1.
    if not col["beta_before"][1] == 1.0:
        raise SystemExit("regime 2 must enter with beta = 1")
    if not col["beta_after"][1] < 1.0:
        raise SystemExit("regime 2 did not reduce beta inside the melt block")
    if not col["snwe_after"][1] == 0.0:
        raise SystemExit("regime 2 did not remove the pack")
    if not col["qvatm"][1] < col["qsg_after"][1]:
        raise SystemExit("regime 2 does not have a positive epot")

    # --- 3 melt on blended snow, rr cap, no liquid retention -------------
    if not snhei[2] < snth[2]:
        raise SystemExit("regime 3 is not below the blending threshold")
    if not (melted[2] and col["snoh"][2] > 0.0):
        raise SystemExit("regime 3 did not melt")
    if not col["dew_after"][2] > 0.0:
        raise SystemExit("regime 3 did not take the condensation half")
    # :5510-5512 caps smelt at the available snow and zeroes snwe.
    if not col["snwe_after"][2] == 0.0:
        raise SystemExit("regime 3 melt was not capped by the available snow")
    if not (snhei[2] <= 0.01 and col["rsm"][2] == 0.0):
        raise SystemExit("regime 3 retained liquid in a sub-centimetre pack")

    # --- 4 two-layer bottom melt, unlimited, partial cover ---------------
    if not snhei[3] > deltsn[3] + snth[3]:
        raise SystemExit("regime 4 is not a two-layer pack")
    if not col["soilt_after"][3] < 273.15:
        raise SystemExit("regime 4 melted from the top after all")
    if not (melted[3] and col["snoh"][3] == 0.0):
        raise SystemExit("regime 4 did not melt from the bottom only")
    if not (col["rhosn_before"][3] >= 350.0 and col["newsnow"][3] == 0.0):
        raise SystemExit("regime 4 is not the dense-snow exemption")
    if not col["smelt"][3] > 5.8e-9:
        raise SystemExit("regime 4 hit the 5.8e-9 bottom-melt cap")
    if not (0.0 < col["snowfrac"][3] < 1.0):
        raise SystemExit("regime 4 must carry partial snow cover")
    if not col["snhei_after"][3] > 0.0:
        raise SystemExit("regime 4 should keep a pack after bottom melt")

    # --- 5 bottom melt limited by the snow that is left (:5663-5664) -----
    # Top melt runs first, so smelt carries both parts; smeltg is what the
    # bottom added.  rr binding means 0 < smeltg < the 5.8e-9 cap.
    if not (0.0 < col["snowfrac"][4] < 1.0):
        raise SystemExit("regime 5 must carry partial snow cover")
    if not (melted[4] and col["snoh"][4] > 0.0):
        raise SystemExit("regime 5 did not melt from the top")
    if not col["soilt_after"][4] > 273.15:
        raise SystemExit(
            "regime 5 skin temperature was clamped, so tso(1) cannot open "
            "the bottom-melt gate"
        )
    smeltg = col["smelt"][4] - col["snoh"][4] / XLMELT * 1.0e-3
    if not 0.0 < smeltg < 5.8e-9:
        raise SystemExit(
            f"regime 5 bottom melt {smeltg!r} was not limited by rr"
        )
    if not (col["snwe_after"][4] == 0.0 and col["snhei_after"][4] == 0.0):
        raise SystemExit("regime 5 should melt the whole pack from below")

    # --- 6 low-density new snow on an aged pack (:5049) ------------------
    if not (col["rhosn_before"][5] >= 156.0 and col["newsnow"][5] > 0.0
            and col["rhonewsn"][5] < 156.0):
        raise SystemExit("regime 6 does not select keff by its second disjunct")
    if not col["smelt"][5] == 0.0:
        raise SystemExit("regime 6 melted unexpectedly")

    # --- 7 condensation-half melt with no limiter in the way -------------
    if not (snth[6] <= snhei[6] <= deltsn[6] + snth[6]):
        raise SystemExit("regime 7 is not a one-layer pack")
    if not (melted[6] and col["snoh"][6] > 0.0):
        raise SystemExit("regime 7 did not melt")
    if not col["dew_after"][6] > 0.0:
        raise SystemExit("regime 7 did not take the condensation half")
    if not (col["rhosn_before"][6] >= 350.0 and col["newsnow"][6] == 0.0):
        raise SystemExit("regime 7 is not the dense-snow exemption")
    if not col["smelt"][6] * XLMELT * 1.0e3 > 18.8:
        raise SystemExit("regime 7 was Egglston limited after all")

    # --- 8 both Egglston guards released by soilt >= 283 -----------------
    if not (snth[7] <= snhei[7] <= deltsn[7] + snth[7]):
        raise SystemExit("regime 8 is not a one-layer pack")
    if not col["rhosn_before"][7] < 350.0:
        raise SystemExit("regime 8 must be below the dense-snow threshold")
    if not col["soilt_after"][7] >= 283.0:
        raise SystemExit(
            "regime 8 skin temperature must clear 283 K to release both "
            "Egglston guards"
        )
    if not (melted[7] and col["snoh"][7] > 0.0 and col["rsm"][7] > 0.0):
        raise SystemExit("regime 8 did not melt and retain liquid")
    if not col["rhosn_after"][7] > col["rhosn_before"][7]:
        raise SystemExit("regime 8 did not densify the pack")
    if not col["smelt"][7] * XLMELT * 1.0e3 > 18.8:
        raise SystemExit("regime 8 was Egglston limited after all")
    if not values["tso_after"][7, 0] > 273.15:
        raise SystemExit("regime 8 did not open the bottom-melt gate")
    bottom = col["smelt"][7] - col["snoh"][7] / XLMELT * 1.0e-3
    if not bottom > 5.8e-9:
        raise SystemExit("regime 8 hit the 5.8e-9 bottom-melt cap")
    if not col["snhei_after"][7] > 0.0:
        raise SystemExit("regime 8 should keep a pack after bottom melt")

    print("RUC snowtemp contract oracle: PASS "
          "(eight regimes: melt under evaporation, melt with full "
          "evaporation of the pack, blended melt capped by available snow, "
          "unlimited two-layer bottom melt, bottom melt capped by available "
          "snow, low-density new snow on an aged pack, unlimited melt under "
          "condensation, and an above-283 K skin that releases both "
          "Egglston guards)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: validate_snowtemp_contract_oracle.py RUC_SNOWTEMP_CONTRACT.csv"
        )
    main(sys.argv[1])
