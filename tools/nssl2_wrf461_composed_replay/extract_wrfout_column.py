"""Extract one real MP18 column for the WRF/GPU composed-stage replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


REGISTRY_VARIABLES = (
    ("qv", "QVAPOR"),
    ("qc", "QCLOUD"),
    ("qr", "QRAIN"),
    ("qi", "QICE"),
    ("qs", "QSNOW"),
    ("qg", "QGRAUP"),
    ("qh", "QHAIL"),
    ("qndrop", "QNDROP"),
    ("qnr", "QNRAIN"),
    ("qni", "QNICE"),
    ("qns", "QNSNOW"),
    ("qng", "QNGRAUPEL"),
    ("qnh", "QNHAIL"),
    ("qnn", "QNCCN"),
    ("qvolg", "QVGRAUPEL"),
    ("qvolh", "QVHAIL"),
)
TEXT_FIELDS = (
    "theta", *(name for name, _ in REGISTRY_VARIABLES),
    "pressure", "exner", "rho", "dz", "w_lower", "w_upper",
)

# Match gpuwm.core.constants and the live `_prepare_fields` path exactly.
# The wrfout contract does not persist ALT/PII, so reconstruct dry density
# from the same moist EOS that diagnosed pressure and reconstruct Exner from
# the production RD/CP ratio.
RD = np.float32(287.0)
RV_OVER_RD = np.float32(461.6 / 287.0)
CP = np.float32(1004.5)
P0 = np.float32(100000.0)
G = np.float32(9.81)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _field(dataset: Dataset, name: str) -> np.ndarray:
    if name not in dataset.variables:
        raise KeyError(f"wrfout is missing required MP18 variable {name!r}")
    variable = dataset.variables[name]
    value = variable[0] if variable.dimensions[:1] == ("Time",) else variable[:]
    if np.ma.isMaskedArray(value):
        value = value.filled(np.nan)
    return np.asarray(value, dtype=np.float32)


def _column(field: np.ndarray, j: int, i: int) -> np.ndarray:
    if field.ndim != 3:
        raise ValueError(f"expected a 3-D atmospheric field, got {field.shape}")
    return np.ascontiguousarray(field[:, j, i], dtype=np.float32)


def _select_column(dataset: Dataset) -> tuple[int, int, str, float]:
    if "REFL_10CM" in dataset.variables:
        reflectivity = _field(dataset, "REFL_10CM")
        score = np.nanmax(reflectivity, axis=0)
        selection = "column_max_refl_10cm"
    else:
        score = np.zeros_like(_field(dataset, "QCLOUD")[0])
        for name in ("QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QHAIL"):
            score += np.nansum(np.maximum(_field(dataset, name), 0.0), axis=0)
        selection = "column_integrated_hydromass"
    flat = int(np.nanargmax(score))
    j, i = np.unravel_index(flat, score.shape)
    return int(j), int(i), selection, float(score[j, i])


def extract(path: Path, output_prefix: Path, dt_s: float, *,
            selected_i: int | None, selected_j: int | None) -> None:
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be positive and finite")
    with Dataset(path, mode="r") as dataset:
        if (selected_i is None) != (selected_j is None):
            raise ValueError("--i and --j must be supplied together")
        if selected_i is None:
            j, i, selection, score = _select_column(dataset)
        else:
            i, j = int(selected_i), int(selected_j)
            selection, score = "explicit", float("nan")

        registry = {
            name: _column(_field(dataset, variable), j, i)
            for name, variable in REGISTRY_VARIABLES
        }
        theta = np.ascontiguousarray(
            _column(_field(dataset, "T"), j, i) + np.float32(300.0),
            dtype=np.float32,
        )
        pressure = np.ascontiguousarray(
            _column(_field(dataset, "P"), j, i)
            + _column(_field(dataset, "PB"), j, i),
            dtype=np.float32,
        )
        exner = np.ascontiguousarray(
            np.power(pressure / P0, RD / CP, dtype=np.float32),
            dtype=np.float32,
        )
        temperature = np.ascontiguousarray(theta * exner, dtype=np.float32)
        rho = np.ascontiguousarray(
            pressure / (RD * temperature * (1.0 + RV_OVER_RD * registry["qv"])),
            dtype=np.float32,
        )
        geopotential = (
            _column(_field(dataset, "PH"), j, i)
            + _column(_field(dataset, "PHB"), j, i)
        )
        height = np.ascontiguousarray(geopotential / G, dtype=np.float32)
        dz = np.ascontiguousarray(height[1:] - height[:-1], dtype=np.float32)
        w_interface = _column(_field(dataset, "W"), j, i)
        if w_interface.size != theta.size + 1:
            raise ValueError(
                f"W has {w_interface.size} levels for {theta.size} mass levels")

        latitude = None
        longitude = None
        if "XLAT" in dataset.variables:
            latitude = float(_field(dataset, "XLAT")[j, i])
        if "XLONG" in dataset.variables:
            longitude = float(_field(dataset, "XLONG")[j, i])

    arrays = {
        "theta": theta,
        **registry,
        "pressure": pressure,
        "exner": exner,
        "rho": rho,
        "dz": dz,
        "w_interface": w_interface,
        "dt_s": np.asarray(np.float32(dt_s)),
        "i": np.asarray(i, dtype=np.int32),
        "j": np.asarray(j, dtype=np.int32),
    }
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = output_prefix.with_suffix(".npz")
    text_path = output_prefix.with_suffix(".txt")
    metadata_path = output_prefix.with_suffix(".json")
    np.savez(npz_path, **arrays)

    rows = np.column_stack([
        theta,
        *(registry[name] for name, _ in REGISTRY_VARIABLES),
        pressure,
        exner,
        rho,
        dz,
        w_interface[:-1],
        w_interface[1:],
    ]).astype(np.float32, copy=False)
    with text_path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(f"{theta.size} {float(np.float32(dt_s)):.9g}\n")
        np.savetxt(stream, rows, fmt="%.9e")

    metadata = {
        "schema": "gpuwm.nssl2.composed-column-input/v1",
        "source": str(path.resolve()),
        "source_sha256": _sha256(path),
        "selection": selection,
        "selection_score": score,
        "i": i,
        "j": j,
        "latitude": latitude,
        "longitude": longitude,
        "nz": int(theta.size),
        "dt_s": float(np.float32(dt_s)),
        "text_fields": list(TEXT_FIELDS),
        "npz_sha256": _sha256(npz_path),
        "text_sha256": _sha256(text_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wrfout", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--dt-s", type=float, required=True)
    parser.add_argument("--i", type=int)
    parser.add_argument("--j", type=int)
    args = parser.parse_args()
    extract(
        args.wrfout,
        args.output_prefix,
        args.dt_s,
        selected_i=args.i,
        selected_j=args.j,
    )


if __name__ == "__main__":
    main()
