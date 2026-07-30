#!/usr/bin/env python3
"""Structural checks for the RUC ``snowseaice`` argument-contract oracle.

``oracle/snowseaice.csv`` pins the arithmetic of five snow-on-ice regimes but
holds ``myj = .false.``, ``snowfrac``/``meltfactor``/``rainf`` at unity,
``ilnb = 1``, ``snwe > 0`` and ``xlv = 2.5e6`` on every row.  Each of those is
either a multiplicative identity or a branch the file never enters, so a
transcription that dropped the argument outright still reproduced the file bit
for bit -- a mutation study over the gpuwm port found that 13 of 14
argument-dropping mutants survived it.

``oracle/snowseaice_contract.csv`` closes that gap.  This validator checks
that each of its ten cases really carries the argument value or reaches the
branch it was built for, and that the file is otherwise structurally sound.
It does NOT compare against any gpuwm result; ``tests/test_ruc.py`` and
``tests/test_ruc_gpu.py`` do the bitwise comparison.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np


CASES = (
    "myj_evaporating",
    "myj_condensing",
    "partial_cover_melt",
    "blended_ilnb_entry",
    "bare_ice_entry",
    "deltsn_halved",
    "thin_pack_melt",
    "melt_evaporating",
    "xlv_offset",
    "snhei_state_drift",
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
        raise SystemExit("RUC snowseaice contract oracle is not finite")

    col = {name: field[:, 0] for name, field in values.items()}

    # --- the header matches run_snowseaice.F90 so one reader serves both ---
    reference = (
        Path(__file__).resolve().parents[2]
        / "gpuwm" / "data" / "ruc" / "oracle" / "snowseaice.csv"
    )
    if reference.is_file():
        with reference.open(newline="", encoding="ascii") as stream:
            if next(csv.reader(stream)) != list(rows[0]):
                raise SystemExit(
                    "contract oracle header diverged from snowseaice.csv"
                )

    # --- sea-ice column properties, same Zubov relations ------------------
    if not np.all((values["rhosice"] > 900.0) & (values["rhosice"] < 930.0)):
        raise SystemExit("sea-ice density left the Zubov range")
    if not np.allclose(
        values["thdifice"], 2.260872 / values["capice"], rtol=1e-5, atol=0.0
    ):
        raise SystemExit("sea-ice diffusivity is inconsistent with its capacity")

    # --- physical bounds and internal identities --------------------------
    for name in ("tso_before", "tso_after", "soilt_after", "soilt1_after"):
        field = values[name]
        if not np.all((field > 180.0) & (field < 330.0)):
            raise SystemExit(f"{name} left the plausible temperature range")
    if not np.all(values["tso_after"] <= TICE_MAX + 1e-3):
        raise SystemExit("contract oracle failed to cap the ice column")
    if not np.all(col["snwe_after"] >= 0.0):
        raise SystemExit("contract oracle produced negative snow water")
    if not np.all((col["rhosn_after"] >= 58.8) & (col["rhosn_after"] <= 500.0)):
        raise SystemExit("snow density left the Koren 58.8-500 kg/m^3 band")
    if not np.all(col["fltot"] == 0.0):
        raise SystemExit("contract oracle broke the closed energy budget")

    # --- each case carries the argument it was built for -------------------
    if not np.array_equal(
        col["myj"], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    ):
        raise SystemExit(f"unexpected myj inventory {col['myj']!r}")
    if not (col["snowfrac"][2] == np.float32(0.6)
            and col["meltfactor"][2] == np.float32(0.4)
            and col["rainf"][2] == np.float32(0.5)
            and col["prcpms"][2] > 0.0):
        raise SystemExit("partial_cover_melt is not off the unity arguments")
    if not np.all(col["snowfrac"][[0, 1, 3, 4, 5, 6, 7, 8, 9]] == 1.0):
        raise SystemExit("only partial_cover_melt should hold snowfrac < 1")
    if col["ilnb_before"][3] != 2.0:
        raise SystemExit("blended_ilnb_entry did not enter with ilnb = 2")
    if not (col["snwe_before"][4] == 0.0 and col["snhei_before"][4] == 0.0):
        raise SystemExit("bare_ice_entry entered with snow present")
    if not (col["s_before"][4] != 0.0
            and col["s_after"][4] == col["s_before"][4]):
        raise SystemExit(
            "bare_ice_entry did not carry its incoming s through untouched"
        )
    if not (col["alb_after"][4] == np.float32(0.55)
            and col["znt_after"][4] == np.float32(0.011)
            and col["emiss_after"][4] == np.float32(0.98)):
        raise SystemExit("bare_ice_entry did not restore the bare sea ice")
    if col["xlv"][8] != np.float32(2.4e6):
        raise SystemExit("xlv_offset is still carrying the default XLV")
    if not np.all(col["xlv"][[0, 1, 2, 3, 4, 5, 6, 7, 9]] == np.float32(2.5e6)):
        raise SystemExit("only xlv_offset should leave the default XLV")

    # deltsn_halved: snhei clears deltsn+snth by less than snth (:3950-3951).
    snth = 0.01e3 / col["rhosn_before"]
    deltsn = 0.05e3 / col["rhosn_before"]
    snhei = col["snhei_before"]
    if not (snhei[5] >= deltsn[5] + snth[5]
            and snhei[5] - deltsn[5] - snth[5] < snth[5]):
        raise SystemExit("deltsn_halved is not in the deltsn-halving band")

    # snhei_state_drift: the incoming snhei is not snwe*1.e3/rhosn, and the
    # two values fall on opposite sides of the halving test.
    rebuilt = np.float32(
        np.float32(col["snwe_before"][9] * np.float32(1.0e3))
        / col["rhosn_before"][9]
    )
    if snhei[9] == rebuilt:
        raise SystemExit("snhei_state_drift carries a consistent snhei")
    halves_rebuilt = (
        rebuilt >= deltsn[9] + snth[9]
        and rebuilt - deltsn[9] - snth[9] < snth[9]
    )
    halves_drift = (
        snhei[9] >= deltsn[9] + snth[9]
        and snhei[9] - deltsn[9] - snth[9] < snth[9]
    )
    if halves_rebuilt == halves_drift:
        raise SystemExit(
            "snhei_state_drift does not change the deltsn halving decision"
        )

    # The melting cases melt; thin_pack_melt keeps no Koren liquid.
    for index in (2, 6, 7):
        if not (col["smelt"][index] > 0.0 and col["snoh"][index] > 0.0):
            raise SystemExit(f"{CASES[index]!r} produced no melt")
        if not col["snom_after"][index] > col["snom_before"][index]:
            raise SystemExit(f"{CASES[index]!r} did not accumulate melt")
    if not (col["snhei_before"][6] <= 0.01 and col["rsm"][6] == 0.0):
        raise SystemExit("thin_pack_melt did not exercise the 1 cm rsm guard")
    if not (col["rsm"][2] > 0.0 and col["rsm"][7] > 0.0):
        raise SystemExit("the deep melting cases retained no Koren liquid")
    # melt_evaporating takes the melt block's evaporating side, and
    # partial_cover_melt its condensing side.
    if not (col["dew_after"][7] == 0.0 and col["eeta"][7] > 0.0):
        raise SystemExit("melt_evaporating did not evaporate")
    if not (col["dew_after"][2] > 0.0 and col["eeta"][2] < 0.0):
        raise SystemExit("partial_cover_melt did not condense")

    # --- one, two and blended snow layers are all represented -------------
    if not np.array_equal(
        col["ilnb_after"], [1, 1, 1, 2, 1, 2, 1, 1, 2, 2]
    ):
        raise SystemExit(f"unexpected snow layer counts {col['ilnb_after']!r}")

    print("RUC snowseaice contract oracle: PASS "
          f"({NCASE} argument-contract regimes: myj both ways, fractional "
          "cover, entering ilnb, bare ice, deltsn halving, the 1 cm rsm "
          "guard, melt evaporation, a non-default xlv and snowh/snow drift)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: validate_snowseaice_contract_oracle.py RUC_CSV"
        )
    main(sys.argv[1])
