"""Build the orography/land-mask supplement a pressure-level source omits.

Some reanalyses publish a complete pressure-level atmosphere, a complete
surface state and a complete soil column, and then do not publish the two
INVARIANT fields WRF-real needs to put that state on a model grid: the
source model's own orography, and its land mask.  NOAA PSL's NetCDF
distribution of 20CRv3 is exactly that shape (MEASURED 2026-08-16: no
``hgt.sfc``/``land`` file exists anywhere under
``Datasets/20thC_ReanV3/``), and the same hole appears in other archives.

This tool closes it from the archive's OWN published fields, and writes a
provenance document saying exactly how, so the receipt a run carries names
the method rather than implying the producer shipped these fields.

Orography
    The source model's surface geopotential height, recovered by evaluating
    its published pressure-level geopotential height at its published
    surface pressure, linearly in ``ln p``, with linear extrapolation in
    ``ln p`` below the lowest published level.  For a hydrostatic archive
    whose pressure-level heights are computed down from the model column,
    this returns the model's own orography to within the vertical
    interpolation error of the level spacing near the ground.  MEASURED
    against 20CRv3 1974-04-03 18Z: 2275 m at 39N/105W (Colorado Front
    Range), 379 m at 35N/98W (central Oklahoma), 50 m at 30N/92W (Louisiana
    coast).  This is a DIVERGENCE from a published orography field and is
    recorded as one; it is not bit-equal to the archive's own model
    orography and does not claim to be.

Land mask
    The valid/missing footprint of a land-only published field -- a soil
    parameter such as the wilting point, which the producer writes only
    where it has soil.  1.0 where the field is present, 0.0 where it is
    missing.  Nothing is thresholded and nothing is guessed: the mask is the
    producer's own statement about where its land model runs.

Every input is named on the command line, so this is not bound to one
model.  The output carries the primary series' own time axis because the
composition contract binds a supplement at every primary valid time; the
field itself is written once and repeated, and the tool refuses to write a
supplement whose repeated field is not identical across those times.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


SUPPLEMENT_SCHEMA = "gpuwm-pressure-level-invariant-supplement-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path, variable: str):
    """Read one variable plus its coordinates through the Rust bridge."""

    from gpuwm import netcdf_bridge

    with netcdf_bridge.open_dataset(path) as dataset:
        if variable not in dataset.variables:
            raise SystemExit(
                f"{path} has no variable {variable!r}; it offers "
                + ", ".join(sorted(dataset.variables))
            )
        handle = dataset.variables[variable]
        values = np.ma.asarray(handle[:])
        dimensions = tuple(handle.dimensions)
        coordinates = {
            name: np.asarray(dataset.variables[name][:], dtype=np.float64)
            for name in dimensions
            if name in dataset.variables
        }
        units = getattr(handle, "units", None)
    return values, dimensions, coordinates, units


def _axis(dimensions, coordinates, kind: str) -> str:
    for name in dimensions:
        if name.lower().startswith(kind):
            return name
    raise SystemExit(
        f"no {kind} dimension among {list(dimensions)}"
    )


def surface_geopotential_height(
    height: np.ndarray,
    level_pa: np.ndarray,
    surface_pressure: np.ndarray,
) -> np.ndarray:
    """Evaluate a level-ordered height profile at the surface pressure.

    ``height`` is ``(level, y, x)`` and ``level_pa`` is its coordinate in
    pascals; ``surface_pressure`` is ``(y, x)`` in pascals.  The
    interpolation is linear in ``ln p`` and EXTRAPOLATES past both ends with
    the slope of the nearest pair, because a surface pressure above the
    lowest published level is ordinary (sea level under high pressure) and
    clamping there would put every such point at the lowest level's height.
    """

    order = np.argsort(level_pa)
    ordered_pressure = np.log(np.asarray(level_pa, dtype=np.float64)[order])
    ordered_height = np.asarray(height, dtype=np.float64)[order]
    if ordered_pressure.size < 2:
        raise SystemExit("surface height recovery needs at least two levels")
    target = np.log(np.asarray(surface_pressure, dtype=np.float64))
    index = np.searchsorted(ordered_pressure, target)
    index = np.clip(index, 1, ordered_pressure.size - 1)
    lower = index - 1
    lower_pressure = ordered_pressure[lower]
    upper_pressure = ordered_pressure[index]
    weight = (target - lower_pressure) / (upper_pressure - lower_pressure)
    rows, columns = np.indices(target.shape)
    lower_height = ordered_height[lower, rows, columns]
    upper_height = ordered_height[index, rows, columns]
    return lower_height + weight * (upper_height - lower_height)


def _write_supplement(
    output: Path,
    *,
    latitude: np.ndarray,
    longitude: np.ndarray,
    times: np.ndarray,
    time_units: str,
    terrain: np.ndarray,
    land: np.ndarray,
) -> None:
    """Write the supplement as classic NetCDF, atomically.

    The write is netCDF4's, deliberately and narrowly: this repository has
    no Rust NetCDF writer yet (the read path is entirely Rust and this tool
    reads through it), and a staging artifact is the one place the gap is
    survivable.  When the Rust writer lands this function is the only thing
    here that changes.
    """

    import netCDF4

    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp")
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with netCDF4.Dataset(temporary, "w", format="NETCDF3_CLASSIC") as tape:
            tape.createDimension("time", times.size)
            tape.createDimension("lat", latitude.size)
            tape.createDimension("lon", longitude.size)
            time_variable = tape.createVariable("time", "f8", ("time",))
            time_variable.units = time_units
            time_variable.standard_name = "time"
            time_variable.calendar = "gregorian"
            time_variable[:] = times
            latitude_variable = tape.createVariable("lat", "f4", ("lat",))
            latitude_variable.units = "degrees_north"
            latitude_variable.standard_name = "latitude"
            latitude_variable[:] = latitude
            longitude_variable = tape.createVariable("lon", "f4", ("lon",))
            longitude_variable.units = "degrees_east"
            longitude_variable.standard_name = "longitude"
            longitude_variable[:] = longitude
            orography = tape.createVariable(
                "orog", "f4", ("time", "lat", "lon"))
            orography.units = "m"
            orography.standard_name = "surface_altitude"
            orography.long_name = "source model surface geopotential height"
            orography.level_desc = "Surface"
            orography[:] = np.broadcast_to(
                terrain, (times.size, *terrain.shape))
            mask = tape.createVariable("land", "f4", ("time", "lat", "lon"))
            mask.units = "1"
            mask.standard_name = "land_binary_mask"
            mask.long_name = "source model land mask"
            mask.level_desc = "Surface"
            mask[:] = np.broadcast_to(land, (times.size, *land.shape))
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build(arguments: argparse.Namespace) -> dict[str, object]:
    height, height_dimensions, height_coordinates, height_units = _read(
        arguments.height, arguments.height_variable)
    pressure, pressure_dimensions, pressure_coordinates, pressure_units = _read(
        arguments.surface_pressure, arguments.surface_pressure_variable)
    land_field, land_dimensions, _land_coordinates, _land_units = _read(
        arguments.land_source, arguments.land_source_variable)

    if height_units != "m":
        raise SystemExit(
            f"{arguments.height_variable} units {height_units!r} are not 'm'; "
            "this tool recovers a HEIGHT profile, not a geopotential")
    if pressure_units != "Pa":
        raise SystemExit(
            f"{arguments.surface_pressure_variable} units {pressure_units!r} "
            "are not 'Pa'")

    level_name = _axis(height_dimensions, height_coordinates, "lev")
    level = height_coordinates[level_name]
    scale = {"millibar": 100.0, "hPa": 100.0, "mbar": 100.0, "Pa": 1.0}
    _values, _dimensions, _coordinates, level_units = _read(
        arguments.height, level_name)
    if level_units not in scale:
        raise SystemExit(
            f"level units {level_units!r} are not one of {sorted(scale)}")
    level_pa = level * scale[level_units]

    time_name = _axis(height_dimensions, height_coordinates, "time")
    times = height_coordinates[time_name]
    _values, _dimensions, _coordinates, time_units = _read(
        arguments.height, time_name)
    latitude = height_coordinates[_axis(height_dimensions, height_coordinates, "lat")]
    longitude = height_coordinates[_axis(height_dimensions, height_coordinates, "lon")]

    terrain_by_time = []
    for index in range(times.size):
        terrain_by_time.append(surface_geopotential_height(
            np.asarray(height[index].filled(np.nan)
                       if np.ma.isMaskedArray(height) else height[index]),
            level_pa,
            np.asarray(pressure[index].filled(np.nan)
                       if np.ma.isMaskedArray(pressure) else pressure[index]),
        ))
    terrain = terrain_by_time[0]
    if not np.isfinite(terrain).all():
        raise SystemExit("recovered orography is not finite everywhere")
    spread = max(
        float(np.max(np.abs(value - terrain))) for value in terrain_by_time
    ) if len(terrain_by_time) > 1 else 0.0
    # An orography that moves with the weather is not an orography.  The
    # recovery evaluates a profile at a surface pressure that changes between
    # valid times, so a few tenths of a metre of drift is the interpolation
    # error and is expected; tens of metres would mean the profile and the
    # surface pressure do not describe the same column, and the supplement
    # would then be a plausible-looking field bound to the wrong model.
    if spread > arguments.invariance_tolerance_m:
        raise SystemExit(
            f"recovered orography moves {spread:.3f} m across the supplied "
            f"valid times, past the {arguments.invariance_tolerance_m} m "
            "tolerance; the height profile and the surface pressure are not "
            "describing one column")

    land_values = np.ma.asarray(land_field)
    present = ~np.ma.getmaskarray(land_values)
    present &= np.isfinite(np.asarray(land_values.filled(np.nan)))
    land = present[0].astype(np.float64)
    if any(not np.array_equal(present[index].astype(np.float64), land)
           for index in range(1, present.shape[0])):
        raise SystemExit(
            f"{arguments.land_source_variable} changes its valid footprint "
            "across the series, so it does not describe an invariant land mask")
    if land.sum() == 0 or land.sum() == land.size:
        raise SystemExit(
            f"{arguments.land_source_variable} is present everywhere or "
            "nowhere in this window, so it carries no land mask")

    _write_supplement(
        arguments.output,
        latitude=latitude, longitude=longitude,
        times=times, time_units=time_units,
        terrain=terrain, land=land,
    )
    receipt = {
        "schema": SUPPLEMENT_SCHEMA,
        "supplement": {
            "path": str(arguments.output.resolve()),
            "sha256": _sha256(arguments.output),
            "variables": {"orog": "m", "land": "1"},
        },
        "orography": {
            "method": "geopotential_height_evaluated_at_surface_pressure",
            "interpolation": "linear_in_ln_p_with_linear_extrapolation",
            "divergence": (
                "recovered from the archive's own pressure-level heights, "
                "not read from a published orography field; it is not "
                "bit-equal to the source model's orography"
            ),
            "height_source": {
                "path": str(arguments.height.resolve()),
                "sha256": _sha256(arguments.height),
                "variable": arguments.height_variable,
            },
            "surface_pressure_source": {
                "path": str(arguments.surface_pressure.resolve()),
                "sha256": _sha256(arguments.surface_pressure),
                "variable": arguments.surface_pressure_variable,
            },
            "level_count": int(level_pa.size),
            "extrapolated_points": int(
                (np.asarray(
                    pressure[0].filled(np.nan)
                    if np.ma.isMaskedArray(pressure) else pressure[0]
                ) > float(np.max(level_pa))).sum()
            ),
            "time_invariance_metres": spread,
            "minimum_m": float(np.min(terrain)),
            "maximum_m": float(np.max(terrain)),
        },
        "land_mask": {
            "method": "valid_footprint_of_a_land_only_published_field",
            "source": {
                "path": str(arguments.land_source.resolve()),
                "sha256": _sha256(arguments.land_source),
                "variable": arguments.land_source_variable,
            },
            "land_points": int(land.sum()),
            "total_points": int(land.size),
        },
        "time_axis": {
            "units": time_units,
            "count": int(times.size),
            "source": str(arguments.height.resolve()),
        },
    }
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.build_pressure_level_invariant_supplement",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("--height", type=Path, required=True,
                        help="pressure-level geopotential HEIGHT file")
    parser.add_argument("--height-variable", required=True)
    parser.add_argument("--surface-pressure", type=Path, required=True)
    parser.add_argument("--surface-pressure-variable", required=True)
    parser.add_argument("--land-source", type=Path, required=True,
                        help="a file whose variable is published on land only")
    parser.add_argument("--land-source-variable", required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="supplement NetCDF to write")
    parser.add_argument("--receipt", type=Path, default=None,
                        help="write the provenance receipt here as well")
    parser.add_argument("--invariance-tolerance-m", type=float, default=10.0,
                        help="how far the recovered orography may move across "
                             "the supplied valid times before the supplement "
                             "is refused (default 10 m)")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.output.exists():
        raise SystemExit(
            f"{arguments.output} exists; this tool never replaces a "
            "supplement another run may be bound to")
    receipt = build(arguments)
    text = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.receipt is not None:
        arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
        arguments.receipt.write_text(text, encoding="utf-8", newline="\n")
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
