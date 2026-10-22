#!/usr/bin/env python3
"""Structural checks for unmodified-WRF RUC ``soilmoist``."""

from __future__ import annotations

import sys

import numpy as np

if __package__:
    from .validate_soil_oracle import _load_profiles
else:
    from validate_soil_oracle import _load_profiles


CASES = ("rain_wet", "dry_evap", "dew", "frozen_melt")


def main(path: str) -> None:
    values = _load_profiles(path, CASES)
    if not all(np.all(np.isfinite(field)) for field in values.values()):
        raise SystemExit("RUC soilmoist oracle contains non-finite output")
    if not np.array_equal(
        values["soiliqw_before"].view(np.uint32),
        values["soiliqw_after"].view(np.uint32),
    ):
        raise SystemExit("soilmoist unexpectedly changed soiliqw")
    if not np.all((values["mavail"] > 0.0) & (values["mavail"] <= 1.0)):
        raise SystemExit("soilmoist returned invalid moisture availability")
    if not values["infiltrp"][0, 0] < 0.0:
        raise SystemExit("rain infiltration branch did not execute")
    if not values["soilmois_after"][1, 0] < values["soilmois_before"][1, 0]:
        raise SystemExit("dry evaporation branch did not remove top-level water")
    if not values["soilmois_after"][2, 0] > values["soilmois_before"][2, 0]:
        raise SystemExit("dew branch did not add top-level water")
    if not values["infmax"][3, 0] <= values["infmax"][0, 0]:
        raise SystemExit("frozen-ground infiltration reduction did not discriminate")
    print("RUC soilmoist oracle: PASS (four finite evolving regimes)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_soilmoist_oracle.py RUC_SOILMOIST.csv")
    main(sys.argv[1])
