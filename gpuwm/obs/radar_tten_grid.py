"""Level-II volumes onto the model grid for radar latent heating, via Rust.

``rw_nexrad grid-ref`` decodes raw NEXRAD Level-II volumes and grids their
reflectivity, every radar together, onto a model grid in NOAA's
``ref2tten`` convention (float32 ``(nz, ny, nx)``: echo cells in dBZ, -99
where the radar looked and saw no echo, -99999 where no radar sampled).
The whole data path -- decode, beam geometry, projection, binning and the
reduction -- is Rust.  This module only

* writes the grid descriptor the binary reads (:func:`write_grid_descriptor`):
  :meth:`TargetGrid.descriptor`, the projection's resolved parameters, the
  grid identity, and ``z_w`` as a raw little-endian float64 file beside it;
* runs the binary (:func:`grid_reflectivity`);
* reads the product back and holds it to its receipt
  (:func:`read_reflectivity`): schema, shape, grid identity, byte count and
  digest.  A product that does not name the grid in hand is refused, never
  reshaped.

The binary is found exactly as :func:`gpuwm.obs.nexrad.find_nexrad_bin`
finds the NEXRAD front door.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from gpuwm.obs.nexrad import find_nexrad_bin, nexrad_remedy

#: The descriptor the binary reads.
GRID_SCHEMA = "gpuwm-obs.radar-tten-grid.v1"
#: The receipt the binary writes beside the product.
REF_SCHEMA = "gpuwm-obs.radar-tten-ref.v1"

#: NOAA's values (``gpuwm.da.radar_tten.NO_ECHO_DBZ`` / ``NO_COVERAGE_DBZ``).
OBSERVED_NO_ECHO = -99.0
NO_COVERAGE = -99999.0

REDUCE_MODES = ("mean", "max")


class RadarTtenGridError(ValueError):
    """The product does not prove it is what the caller asked for."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def projection_spec(grid) -> dict:
    """The grid's projection with every default resolved.

    The ``static-fields`` crate's ``GridSpec`` document, which is the
    byte-parity Rust port of :mod:`gpuwm.static.projection`.  A translated
    grid is described by its own resolved known point rather than by
    delegation to its reference, which agrees with the delegated transform
    to the last bits of a float, far inside the half cell a gate rounds by.
    """

    projection = grid.projection
    return {
        "kind": str(projection.map_proj),
        "ref_lat": float(projection.ref_lat),
        "ref_lon": float(projection.ref_lon),
        "truelat1": float(projection.truelat1),
        "truelat2": float(projection.truelat2),
        "stand_lon": float(projection.stand_lon),
        "dx": float(projection.dx),
        "dy": float(projection.dy),
        "e_we": int(projection.e_we),
        "e_sn": int(projection.e_sn),
        "known_x": float(projection.known_x),
        "known_y": float(projection.known_y),
        "moad_cen_lat": float(projection.moad_cen_lat),
        "moad_cen_lon": float(projection.moad_cen_lon),
    }


def write_grid_descriptor(grid, path) -> dict:
    """Write ``path`` (the JSON) and ``<stem>.z_w.f64`` beside it.

    Returns the descriptor.  Rewrites both atomically; an unchanged grid
    rewrites identical bytes.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    z_w = np.ascontiguousarray(grid.z_w, dtype="<f8")
    z_path = path.with_name(path.stem + ".z_w.f64")
    tmp = z_path.with_name(z_path.name + ".tmp")
    z_w.tofile(tmp)
    tmp.replace(z_path)
    descriptor = {
        "schema": GRID_SCHEMA,
        "name": grid.name,
        "identity_sha256": grid.identity_sha256(),
        "descriptor": grid.descriptor(),
        "nx": int(grid.nx),
        "ny": int(grid.ny),
        "nz": int(grid.nz),
        "projection": projection_spec(grid),
        "z_w": {
            "file": z_path.name,
            "dtype": "<f8",
            "shape": [int(v) for v in z_w.shape],
            "sha256": _sha256_file(z_path),
        },
    }
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(descriptor, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    tmp.replace(path)
    return descriptor


def read_reflectivity(prefix, *, grid=None, identity_sha256: str | None = None):
    """Read ``<prefix>.json`` and the field it names; return ``(ref, receipt)``.

    The field is the file the receipt's ``data.file`` names (relative to the
    receipt), ``<prefix>.ref.f32`` unless the call chose another.  ``ref`` is
    float32 ``(nz, ny, nx)``.  The receipt must declare
    :data:`REF_SCHEMA`, a shape that accounts for every byte, the digest of
    those bytes, and -- when a ``grid`` or ``identity_sha256`` is given --
    that grid's identity.  Any disagreement is a :class:`RadarTtenGridError`.
    """

    prefix = Path(prefix)
    receipt_path = Path(str(prefix) + ".json")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RadarTtenGridError(
            f"cannot read the grid-ref receipt {receipt_path}: {error}") from error
    if receipt.get("schema") != REF_SCHEMA or receipt.get("status") != "READY":
        raise RadarTtenGridError(
            f"{receipt_path} declares schema {receipt.get('schema')!r} status "
            f"{receipt.get('status')!r}; expected {REF_SCHEMA!r} READY")
    shape = receipt.get("shape")
    if (not isinstance(shape, list) or len(shape) != 3
            or not all(isinstance(v, int) and v > 0 for v in shape)):
        raise RadarTtenGridError(f"{receipt_path} states no (nz, ny, nx) shape: {shape!r}")
    expected = None
    if grid is not None:
        expected = grid.identity_sha256()
        if tuple(shape) != (grid.nz, grid.ny, grid.nx):
            raise RadarTtenGridError(
                f"the product's shape {tuple(shape)} is not the grid's "
                f"{(grid.nz, grid.ny, grid.nx)}")
    if identity_sha256 is not None:
        if expected is not None and identity_sha256 != expected:
            raise RadarTtenGridError(
                "the identity passed and the grid passed disagree")
        expected = identity_sha256
    stated = (receipt.get("grid") or {}).get("identity_sha256")
    if expected is not None and stated != expected:
        raise RadarTtenGridError(
            f"the product was gridded onto {stated!r}, not the required grid "
            f"{expected!r}; refusing to read observations placed in other columns")
    data = receipt.get("data") or {}
    named = data.get("file")
    if not isinstance(named, str) or not named:
        raise RadarTtenGridError(f"{receipt_path} names no data file")
    data_path = Path(named)
    if not data_path.is_absolute():
        data_path = receipt_path.parent / data_path
    count = int(np.prod(shape))
    try:
        size = data_path.stat().st_size
    except OSError as error:
        raise RadarTtenGridError(f"cannot read {data_path}: {error}") from error
    if size != count * 4 or data.get("bytes") != size:
        raise RadarTtenGridError(
            f"{data_path} holds {size} bytes; shape {shape} needs {count * 4} "
            f"and the receipt states {data.get('bytes')}")
    digest = _sha256_file(data_path)
    if digest != data.get("sha256"):
        raise RadarTtenGridError(
            f"{data_path} has sha256 {digest}, the receipt states {data.get('sha256')}")
    ref = np.fromfile(data_path, dtype="<f4").reshape(shape)
    return ref, receipt


def grid_reflectivity(grid, volumes, out_prefix, *, reduce: str = "mean",
                      max_range_km: float = 250.0,
                      max_elevation_deg: float = 20.0,
                      threads: int | None = None,
                      window_end: str | None = None,
                      max_age_s: float | None = None,
                      grid_descriptor=None, out_data=None, binary=None,
                      timeout=None):
    """Grid ``volumes`` onto ``grid`` with ``rw_nexrad grid-ref``.

    ``grid_descriptor`` reuses an already written descriptor (several
    windows on one grid write it once); otherwise one is written next to
    ``out_prefix``.  ``out_data`` places the float32 field somewhere other
    than ``<out_prefix>.ref.f32``.  Returns ``(ref, receipt)`` from
    :func:`read_reflectivity`.
    """

    if reduce not in REDUCE_MODES:
        raise ValueError(f"reduce must be one of {REDUCE_MODES}, got {reduce!r}")
    volumes = [Path(v) for v in volumes]
    if not volumes:
        raise ValueError("no volumes to grid")
    binary = Path(binary) if binary is not None else find_nexrad_bin()
    if binary is None:
        raise FileNotFoundError(nexrad_remedy())
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    if grid_descriptor is None:
        grid_descriptor = Path(str(out_prefix) + ".grid.json")
        write_grid_descriptor(grid, grid_descriptor)
    list_path = Path(str(out_prefix) + ".volumes.txt")
    list_path.write_text("".join(f"{v}\n" for v in volumes), encoding="utf-8")
    command = [str(binary), "grid-ref", "--grid", str(grid_descriptor),
               "--volume-list", str(list_path), "--out", str(out_prefix),
               "--reduce", reduce, "--max-range-km", repr(float(max_range_km)),
               "--max-elevation-deg", repr(float(max_elevation_deg))]
    if out_data is not None:
        command += ["--out-data", str(out_data)]
    if threads is not None:
        command += ["--threads", str(int(threads))]
    if window_end is not None:
        command += ["--window-end", str(window_end)]
    if max_age_s is not None:
        command += ["--max-age-s", repr(float(max_age_s))]
    result = subprocess.run(command, capture_output=True, text=True,
                            errors="replace", timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"rw_nexrad grid-ref exited {result.returncode}: "
            f"{(result.stderr or '').strip()[-4000:]}")
    return read_reflectivity(out_prefix, grid=grid)
