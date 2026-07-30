#!/usr/bin/env python3
"""Structural checks for the unmodified-WRF RUC ``snowtemp`` snow oracle.

There is no gpuwm snow port yet -- ``gpuwm/core/ruc.py`` reproduces the
snow-free land column only -- so this validator does NOT compare the CSV
against any gpuwm result.  It checks that the pinned fixture is structurally
complete, finite, physically bounded, and that each named regime really
entered the snow branch it was designed to exercise.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np


CASES = (
    "shallow_fresh_1layer",
    "deep_aged_2layer",
    "melting_2layer",
    "thin_blended",
    "warm_ground_bottom_melt",
    "sublimating_all_snow",
)
NCASE = len(CASES)
NZS = 9


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
        raise SystemExit("RUC snowtemp oracle contains non-finite output")

    col = {name: field[:, 0] for name, field in values.items()}

    # --- physical bounds -------------------------------------------------
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

    # --- each regime took its intended branch ----------------------------
    if not np.array_equal(col["ilnb_after"], [1.0, 2.0, 2.0, 1.0, 1.0, 1.0]):
        raise SystemExit(f"unexpected snow layer counts {col['ilnb_after']!r}")
    snth = col["snth"]
    deltsn = col["deltsn"]
    snhei = col["snhei_before"]
    if not (snth[0] <= snhei[0] <= deltsn[0] + snth[0]):
        raise SystemExit("shallow regime is not in the one-layer depth band")
    if not snhei[1] > deltsn[1] + snth[1]:
        raise SystemExit("deep regime is not in the two-layer depth band")
    if not snhei[3] < snth[3]:
        raise SystemExit("thin regime is not below the blending threshold")

    if not (col["smelt"][2] > 0.0 and col["snoh"][2] > 0.0
            and col["rsm"][2] > 0.0):
        raise SystemExit("melting regime produced no melt")
    if not col["rhosn_after"][2] > col["rhosn_before"][2]:
        raise SystemExit("melting regime did not densify the pack")
    if not abs(col["soilt_after"][2] - 273.15) < 1.0e-3:
        raise SystemExit("melting regime skin temperature was not held at 273.15")

    if not (col["smelt"][4] > 0.0 and col["snoh"][4] == 0.0):
        raise SystemExit("warm-ground regime did not melt from the bottom only")
    if not col["soilt_after"][4] < 273.15:
        raise SystemExit("warm-ground regime melted from the top after all")
    if not abs(values["tso_after"][4, 0] - 273.15) < 1.0e-3:
        raise SystemExit("warm-ground regime did not hold tso(1) at 273.15")

    if not col["beta_after"][5] < 1.0:
        raise SystemExit("sublimation regime was not evaporation limited")
    if not (col["snwe_after"][5] == 0.0 and col["snhei_after"][5] == 0.0):
        raise SystemExit("sublimation regime did not remove the pack")

    if not np.all(col["smelt"][[0, 1, 3]] == 0.0):
        raise SystemExit("a cold regime melted unexpectedly")

    print("RUC snowtemp oracle: PASS "
          "(six finite snow regimes: 1-layer, 2-layer, top melt, blended, "
          "bottom melt, full sublimation)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_snowtemp_oracle.py RUC_SNOWTEMP.csv")
    main(sys.argv[1])
