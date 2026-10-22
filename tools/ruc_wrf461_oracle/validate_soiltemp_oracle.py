#!/usr/bin/env python3
"""Structural checks for unmodified-WRF RUC ``soiltemp``."""

from __future__ import annotations

import sys

import numpy as np

if __package__:
    from .validate_soil_oracle import _load_profiles
else:
    from validate_soil_oracle import _load_profiles


CASES = (
    "moist_saturated",
    "dry_second_solve",
    "warm_rain",
    "humid_condensing",
)


def main(path: str) -> None:
    values = _load_profiles(path, CASES)
    if not all(np.all(np.isfinite(field)) for field in values.values()):
        raise SystemExit("RUC soiltemp oracle contains non-finite output")
    if not all(np.any(values["tso_after"][case] != values["tso_before"][case])
               for case in range(4)):
        raise SystemExit("soiltemp failed to evolve every temperature profile")
    if not values["qvg_after"][1, 0] < values["qsg_after"][1, 0]:
        raise SystemExit("dry second-solve regime did not remain unsaturated")
    if not values["qcg_after"][3, 0] > 0.0:
        raise SystemExit("humid regime did not enter the condensing branch")
    if not values["rainf"][2, 0] == 1.0 or not values["prcpms"][2, 0] > 0.0:
        raise SystemExit("rain-heating regime was not pinned")
    print("RUC soiltemp oracle: PASS (four finite nonlinear heat regimes)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_soiltemp_oracle.py RUC_SOILTEMP.csv")
    main(sys.argv[1])
