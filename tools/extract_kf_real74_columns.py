"""Extract the two Phase-4 KF gate columns from the Phase-3 12Z state."""

from __future__ import annotations

from pathlib import Path

import cupy as cp
import numpy as np

from gpuwm.core.physics import _prepare_atmosphere
from gpuwm.verify.cases.real74_d01 import prepare_phase3_case


COLUMNS = {"unstable": (53, 113), "stable": (194, 5)}
FIELDS = ("u", "v", "temperature", "qv", "qc", "pressure", "exner", "dz")


def main() -> None:
    prepared = prepare_phase3_case()
    state = prepared.initial_result.state
    atmosphere = _prepare_atmosphere(state)
    w = cp.asnumpy(0.5 * (state.w[:-1] + state.w[1:]))
    latitude, longitude = prepared.grid.latlon_mass()
    data: dict[str, np.ndarray] = {}
    for label, (j, i) in COLUMNS.items():
        for field in FIELDS:
            data[f"{label}_{field}"] = cp.asnumpy(
                atmosphere[field][:, j, i]).astype(np.float32)
        data[f"{label}_w"] = np.asarray(w[:, j, i], dtype=np.float32)
        data[f"{label}_j"] = np.asarray(j, dtype=np.int32)
        data[f"{label}_i"] = np.asarray(i, dtype=np.int32)
        data[f"{label}_latitude"] = np.asarray(latitude[j, i], np.float64)
        data[f"{label}_longitude"] = np.asarray(longitude[j, i], np.float64)
    destination = (Path(__file__).resolve().parents[1] / "tests" / "data" /
                   "kf_real74_12z_columns.npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **data)


if __name__ == "__main__":
    main()
