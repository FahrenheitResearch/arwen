#!/usr/bin/env python3
"""Structural and branch checks for unmodified-WRF RUC ``soilprop``."""

from __future__ import annotations

import sys

import numpy as np

if __package__:
    from .validate_soil_oracle import _load_profiles
else:
    from validate_soil_oracle import _load_profiles


CASES = ("warm_wet", "warm_dry", "frozen", "deep_frozen_keep")


def main(path: str) -> None:
    values = _load_profiles(path, CASES)
    if not all(np.all(np.isfinite(field)) for field in values.values()):
        raise SystemExit("RUC soilprop oracle contains non-finite output")
    if not np.all(values["hydro"][0] > np.float32(0.0)):
        raise SystemExit("warm-wet hydraulic branch did not execute")
    if not np.any(values["hydro"][1] == np.float32(0.0)):
        raise SystemExit("warm-dry conductivity floor did not execute")
    if not (
        np.all(values["diffu"][3, :-1] == np.float32(0.0))
        and np.all(values["hydro"][3] == np.float32(0.0))
    ):
        raise SystemExit("deep-frozen conductivity shutoff did not execute")
    for name in ("thdif", "diffu", "cap"):
        if not np.all(values[name][:, -1] == np.float32(0.0)):
            raise SystemExit(f"WRF last-level {name} initialization drifted")
    print("RUC soilprop oracle: PASS (four finite discriminating regimes)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_soilprop_oracle.py RUC_SOILPROP.csv")
    main(sys.argv[1])
