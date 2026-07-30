#!/usr/bin/env python3
"""Structural, bound and branch checks for the unmodified-WRF Noah-MP
NOAHMP_TABLES + TRANSFER_MP_PARAMETERS oracle.

No gpuwm comparison happens here: the Noah-MP column port does not exist yet.
This only proves the fixture is well formed, physically bounded, and that each
intended branch of TRANSFER_MP_PARAMETERS actually executed.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

CASES = (
    "evergreen_needleleaf",
    "grassland",
    "cropland",
    "urban",
    "barren",
    "snow_ice",
    "water",
)
# MODIS ("MODIFIED_IGBP_MODIS_NOAH") category identity from MPTABLE.TBL.
MODIS_IDENTITY = {
    "ISWATER": 17,
    "ISBARREN": 16,
    "ISICE": 15,
    "ISCROP": 12,
    "EBLFOREST": 2,
}
# GENPARM.TBL scalars, identical for every non-urban case.
GENPARM = {"CSOIL": 2.0e6, "ZBOT": -8.0, "CZIL": 0.1, "REFDK": 2.0e-6,
           "REFKDT": 3.0}
SLOPE_TABLE = (0.1, 0.6, 1.0, 0.35, 0.55, 0.8, 0.63, 0.0, 0.0)
FRZK = 0.15
NSOIL = 4


def fail(message: str) -> None:
    raise SystemExit(f"Noah-MP parameter oracle: {message}")


def main(path: str) -> None:
    with Path(path).open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))

    order = tuple(dict.fromkeys(row["case"] for row in rows))
    if order != CASES:
        fail(f"unexpected case inventory {order!r}")

    table: dict[str, dict[tuple[str, int], float]] = {c: {} for c in CASES}
    meta: dict[str, dict[str, int]] = {}
    for row in rows:
        case = row["case"]
        key = (row["field"], int(row["index"]))
        if key in table[case]:
            fail(f"{case}: duplicate entry {key}")
        value = float(row["value"])
        if value != value or value in (float("inf"), float("-inf")):
            fail(f"{case}: non-finite {key}")
        table[case][key] = value
        meta.setdefault(case, {
            "vegtype": int(row["vegtype"]),
            "soiltype": int(row["soiltype"]),
            "slopetype": int(row["slopetype"]),
            "soilcolor": int(row["soilcolor"]),
        })

    fields = {key for key in table[CASES[0]]}
    for case in CASES[1:]:
        if set(table[case]) != fields:
            fail(f"{case}: field inventory differs from {CASES[0]}")
    if len(rows) != len(CASES) * len(fields):
        fail(f"expected {len(CASES) * len(fields)} rows, got {len(rows)}")

    for case in CASES:
        v = table[case]
        m = meta[case]

        for name, expected in MODIS_IDENTITY.items():
            if int(v[(name, 0)]) != expected:
                fail(f"{case}: {name} is {v[(name, 0)]}, expected {expected}"
                     " -- MODIS land-use identity was not loaded")

        # Urban branch of TRANSFER_MP_PARAMETERS.
        urban = int(v[("URBAN_FLAG", 0)])
        if urban not in (0, 1):
            fail(f"{case}: URBAN_FLAG is not boolean")
        if (case == "urban") != bool(urban):
            fail(f"{case}: URBAN_FLAG branch did not follow the vegetation type")
        if urban:
            for k in range(1, NSOIL + 1):
                if abs(v[("SMCMAX", k)] - 0.45) > 1e-6:
                    fail("urban SMCMAX override did not execute")
                if abs(v[("SMCREF", k)] - 0.42) > 1e-6:
                    fail("urban SMCREF override did not execute")
                if abs(v[("SMCWLT", k)] - 0.40) > 1e-6:
                    fail("urban SMCWLT override did not execute")
            if abs(v[("CSOIL", 0)] - 3.0e6) > 1.0:
                fail("urban CSOIL override did not execute")
        else:
            for name, expected in GENPARM.items():
                if abs(v[(name, 0)] - expected) > abs(expected) * 1e-6:
                    fail(f"{case}: GENPARM {name} is {v[(name, 0)]},"
                         f" expected {expected}")

        # SOILPARM physical bounds and ordering.
        for k in range(1, NSOIL + 1):
            smcmax = v[("SMCMAX", k)]
            smcref = v[("SMCREF", k)]
            smcwlt = v[("SMCWLT", k)]
            smcdry = v[("SMCDRY", k)]
            if not 0.0 < smcmax <= 1.0:
                fail(f"{case}: SMCMAX({k}) = {smcmax} outside (0, 1]")
            if not 0.0 <= smcdry <= smcwlt + 1e-6 <= smcref + 1e-6 <= smcmax:
                fail(f"{case}: soil moisture thresholds not ordered at layer {k}")
            if v[("BEXP", k)] <= 0.0:
                fail(f"{case}: BEXP({k}) is not positive")
            if v[("DKSAT", k)] <= 0.0 or v[("DWSAT", k)] <= 0.0:
                fail(f"{case}: saturated conductivity/diffusivity not positive")
            if v[("PSISAT", k)] <= 0.0:
                fail(f"{case}: PSISAT({k}) is not positive")
            if not 0.0 <= v[("QUARTZ", k)] <= 1.0:
                fail(f"{case}: QUARTZ({k}) outside [0, 1]")

        # Derived GENPARM quantities WRF forms inside the transfer.
        kdt = v[("REFKDT", 0)] * v[("DKSAT", 1)] / v[("REFDK", 0)]
        if abs(v[("KDT", 0)] - kdt) > max(1e-5, abs(kdt) * 1e-5):
            fail(f"{case}: KDT is {v[('KDT', 0)]}, expected {kdt}")
        frzx = FRZK * (v[("SMCMAX", 1)] / v[("SMCREF", 1)]) * (0.412 / 0.468)
        if abs(v[("FRZX", 0)] - frzx) > 1e-5:
            fail(f"{case}: FRZX is {v[('FRZX', 0)]}, expected {frzx}")
        slope = SLOPE_TABLE[m["slopetype"] - 1]
        if abs(v[("SLOPE", 0)] - slope) > 1e-6:
            fail(f"{case}: SLOPE is {v[('SLOPE', 0)]}, expected {slope}"
                 f" for slopetype {m['slopetype']}")

        # Vegetation and radiation blocks.
        if not -1.0 <= v[("XL", 0)] <= 1.0:
            fail(f"{case}: leaf orientation index outside [-1, 1]")
        for band in (1, 2):
            for name in ("RHOL", "RHOS", "TAUL", "TAUS", "ALBSAT", "ALBDRY",
                         "ALBICE", "ALBLAK", "OMEGAS"):
                x = v[(name, band)]
                if not 0.0 <= x <= 1.0:
                    fail(f"{case}: {name}({band}) = {x} outside [0, 1]")
            if v[("ALBDRY", band)] < v[("ALBSAT", band)]:
                fail(f"{case}: dry soil albedo below saturated soil albedo")
        if not 0.0 <= v[("SNOW_EMIS", 0)] <= 1.0:
            fail(f"{case}: SNOW_EMIS outside [0, 1]")
        if v[("RSMIN", 0)] <= 0.0 or v[("RSMAX", 0)] <= v[("RSMIN", 0)]:
            fail(f"{case}: stomatal resistance limits not ordered")
        nroot = int(v[("NROOT", 0)])
        if not 0 <= nroot <= NSOIL:
            fail(f"{case}: NROOT = {nroot} outside [0, {NSOIL}]")
        for month in range(1, 13):
            if v[("LAIM", month)] < 0.0 or v[("SAIM", month)] < 0.0:
                fail(f"{case}: negative LAIM/SAIM in month {month}")

    # Non-vegetated MODIS categories must carry a zero annual LAI cycle;
    # vegetated ones must not.
    for case in ("barren", "snow_ice", "water"):
        if any(table[case][("LAIM", m)] != 0.0 for m in range(1, 13)):
            fail(f"{case}: expected a zero monthly LAI cycle")
    for case in ("evergreen_needleleaf", "grassland", "cropland"):
        if max(table[case][("LAIM", m)] for m in range(1, 13)) <= 0.0:
            fail(f"{case}: vegetated category has no LAI cycle")

    # Soil-colour indexing must actually vary the ground albedo.
    if table["barren"][("ALBSAT", 1)] == table["grassland"][("ALBSAT", 1)]:
        fail("soil-colour indexing of ALBSAT did not execute")
    if table["snow_ice"][("ALBSAT", 1)] == table["grassland"][("ALBSAT", 1)]:
        fail("soil-colour indexing of ALBSAT did not execute")
    # Soil-type indexing must actually vary the hydraulic parameters.
    if len({table[c][("DKSAT", 1)] for c in CASES}) < 5:
        fail("soil-type indexing of DKSAT did not execute")
    # Vegetation-type indexing must actually vary the canopy parameters.
    if len({table[c][("Z0MVT", 0)] for c in CASES}) < 4:
        fail("vegetation-type indexing of Z0MVT did not execute")

    print(f"Noah-MP parameter oracle: PASS "
          f"({len(CASES)} cases x {len(fields)} transferred fields)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_parameters_oracle.py NOAHMP_PARAMETERS.csv")
    main(sys.argv[1])
