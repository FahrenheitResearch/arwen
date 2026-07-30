#!/usr/bin/env python3
"""Structural checks for the unmodified-WRF RUC ``snowseaice`` oracle.

There is no gpuwm snow port yet -- ``gpuwm/core/ruc.py`` reproduces the
snow-free land column only -- so this validator does NOT compare the CSV
against any gpuwm result.  It checks that the pinned fixture is structurally
complete, finite, physically bounded, internally consistent, and that each
named regime really entered the snow-on-ice branch it was designed to
exercise.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np


CASES = (
    "blended_thin_on_ice",
    "one_layer_on_ice",
    "two_layer_deep_on_ice",
    "melting_on_ice",
    "sublimating_on_ice",
)
NCASE = len(CASES)
NZS = 9
TICE_MAX = 271.4


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
        raise SystemExit("RUC snowseaice oracle contains non-finite output")

    col = {name: field[:, 0] for name, field in values.items()}

    # --- sea-ice column properties ---------------------------------------
    if not np.all((values["rhosice"] > 900.0) & (values["rhosice"] < 930.0)):
        raise SystemExit("sea-ice density left the Zubov range")
    if not np.all(values["capice"] > 0.0):
        raise SystemExit("sea-ice heat capacity went non-positive")
    if not np.allclose(
        values["thdifice"], 2.260872 / values["capice"], rtol=1e-5, atol=0.0
    ):
        raise SystemExit("sea-ice diffusivity is inconsistent with its capacity")
    if not np.allclose(values["tice"], values["tso_before"] - 273.15,
                       rtol=0.0, atol=1e-2):
        raise SystemExit("tice does not mirror the input ice temperature")

    # --- physical bounds and internal identities -------------------------
    for name in ("tso_before", "tso_after", "soilt_after", "soilt1_after"):
        field = values[name]
        if not np.all((field > 180.0) & (field < 330.0)):
            raise SystemExit(f"{name} left the plausible temperature range")
    if not np.all(values["tso_after"] <= TICE_MAX + 1e-3):
        raise SystemExit("snowseaice failed to cap the ice column at 271.4 K")
    if not np.all(col["snwe_after"] >= 0.0):
        raise SystemExit("snowseaice produced negative snow water equivalent")
    if not np.all((col["rhosn_after"] >= 58.8) & (col["rhosn_after"] <= 500.0)):
        raise SystemExit("snow density left the Koren 58.8-500 kg/m^3 band")
    for name in ("smelt", "snoh", "rsm", "snom_after", "qcg_after"):
        if not np.all(col[name] >= 0.0):
            raise SystemExit(f"{name} went negative")
    if not np.allclose(
        col["snheiprint"], col["snweprint"] * 1.0e3 / col["rhosn_after"],
        rtol=1e-5, atol=1e-9,
    ):
        raise SystemExit("reported snow depth is inconsistent with its density")
    if not np.all(col["fltot"] == 0.0):
        raise SystemExit("snowseaice broke its closed energy budget identity")
    if not all(np.any(values["tso_after"][n] != values["tso_before"][n])
               for n in range(NCASE)):
        raise SystemExit("snowseaice failed to evolve every ice profile")

    # --- each regime took its intended branch ----------------------------
    if not np.array_equal(col["ilnb_after"], [1.0, 1.0, 2.0, 1.0, 1.0]):
        raise SystemExit(f"unexpected snow layer counts {col['ilnb_after']!r}")
    snth = 0.01e3 / col["rhosn_before"]
    deltsn = 0.05e3 / col["rhosn_before"]
    snhei = col["snhei_before"]
    if not snhei[0] < snth[0]:
        raise SystemExit("thin regime is not below the blending threshold")
    if not (snth[1] <= snhei[1] <= deltsn[1] + snth[1]):
        raise SystemExit("one-layer regime is not in the one-layer depth band")
    if not snhei[2] > deltsn[2] + snth[2]:
        raise SystemExit("deep regime is not in the two-layer depth band")

    if not (col["smelt"][3] > 0.0 and col["snoh"][3] > 0.0
            and col["rsm"][3] > 0.0):
        raise SystemExit("melting regime produced no melt")
    if not col["snom_after"][3] > col["snom_before"][3]:
        raise SystemExit("melting regime did not accumulate snow melt")
    if not abs(col["soilt_after"][3] - 273.15) < 1.0e-3:
        raise SystemExit("melting regime skin temperature was not held at 273.15")

    if not (col["snwe_after"][4] == 0.0 and col["snhei_after"][4] == 0.0):
        raise SystemExit("sublimation regime did not remove the pack")
    if not (col["alb_after"][4] == np.float32(0.55)
            and col["znt_after"][4] == np.float32(0.011)
            and col["emiss_after"][4] == np.float32(0.98)):
        raise SystemExit("bare sea-ice surface properties were not restored")
    for n in range(4):
        if not (col["alb_after"][n] == col["alb_before"][n]
                and col["znt_after"][n] == col["znt_before"][n]
                and col["emiss_after"][n] == col["emiss_before"][n]):
            raise SystemExit(
                f"snow-covered regime {CASES[n]!r} lost its snow surface state")
    if not np.all(col["smelt"][[0, 1, 2, 4]] == 0.0):
        raise SystemExit("a cold regime melted unexpectedly")

    print("RUC snowseaice oracle: PASS "
          "(five finite snow-on-ice regimes: blended, 1-layer, 2-layer, "
          "melting, sublimated to bare ice)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_snowseaice_oracle.py RUC_SNOWSEAICE.csv")
    main(sys.argv[1])
