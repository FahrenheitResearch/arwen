"""Validate the geometry and representation already baked into WRF input.

The Rust NetCDF reader owns decoding. This leaf module compares its metadata
and vertical coordinate with the requested experiment before state restoration.
It imports no forecast runtime, so standalone preparation can use the same gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from gpuwm import netcdf_bridge
from gpuwm.vertical_contract import validate_explicit_eta_grid


@dataclass(frozen=True)
class WrfinputIdentity:
    path: Path
    dimensions: Mapping[str, int]
    attributes: Mapping[str, object]
    eta_levels: tuple[float, ...]
    p_top: float
    start_time: datetime


def _integer(value, name: str, path: Path) -> int:
    try:
        number = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"{path}: {name} must be a finite integer") from None
    if not np.isfinite(number) or number != int(number):
        raise ValueError(f"{path}: {name} must be a finite integer")
    return int(number)


def _vector(dataset, name: str, path: Path) -> np.ndarray:
    if name not in dataset.variables:
        raise ValueError(f"{path}: missing {name}; the file cannot establish its vertical grid")
    value = np.asarray(dataset.variables[name][...], dtype=np.float64).reshape(-1)
    if not np.isfinite(value).all():
        raise ValueError(f"{path}: {name} contains a non-finite value")
    return value


def read_wrfinput_identity(path) -> WrfinputIdentity:
    """Read and check dimensions, representation, record time and vertical grid."""
    path = Path(path)
    with netcdf_bridge.open_dataset(path) as dataset:
        dimensions = {name: len(dim) for name, dim in dataset.dimensions.items()}
        attributes = {name: dataset.getncattr(name) for name in dataset.ncattrs()}
        required = ("GRID_ID", "MP_PHYSICS", "SF_SURFACE_PHYSICS",
                    "USE_THETA_M", "HYBRID_OPT", "ETAC", "DX", "DY", "START_DATE")
        missing = [name for name in required if name not in attributes]
        if missing:
            raise ValueError(f"{path}: missing input identity attributes {missing}")
        for name in required[:5]:
            attributes[name] = _integer(attributes[name], name, path)
        for name in ("west_east", "south_north", "bottom_top", "soil_layers_stag"):
            if dimensions.get(name, 0) <= 0:
                raise ValueError(f"{path}: missing or empty {name} dimension")
        for mass, face in (("west_east", "west_east_stag"),
                           ("south_north", "south_north_stag"),
                           ("bottom_top", "bottom_top_stag")):
            if dimensions.get(face) != dimensions[mass] + 1:
                raise ValueError(f"{path}: {face} must have one more point than {mass}")
        if dimensions.get("Time") != 1:
            raise ValueError(f"{path}: an initial-condition file must contain exactly one Time record")
        eta = _vector(dataset, "ZNW", path)
        top = _vector(dataset, "P_TOP", path)
        if top.size != 1:
            raise ValueError(f"{path}: P_TOP must contain one value")
        validate_explicit_eta_grid(eta, nz=dimensions["bottom_top"],
                                  p_top=float(top[0]), context=str(path))
        mid = _vector(dataset, "ZNU", path)
        if mid.shape != (dimensions["bottom_top"],) or not np.allclose(
                mid, (eta[:-1] + eta[1:]) * 0.5, rtol=0.0, atol=1e-7):
            raise ValueError(f"{path}: ZNU disagrees with the ZNW layer midpoints")
        if "Times" not in dataset.variables:
            raise ValueError(f"{path}: missing Times record")
        chars = np.asarray(dataset.variables["Times"][...]).reshape(-1)
        record = b"".join(bytes(item) for item in chars).decode("ascii").strip("\x00 ")
        try:
            start = datetime.strptime(record, "%Y-%m-%d_%H:%M:%S")
        except ValueError:
            raise ValueError(f"{path}: invalid Times record {record!r}") from None
        if record != str(attributes["START_DATE"]):
            raise ValueError(f"{path}: Times {record} differs from START_DATE {attributes['START_DATE']}")
    return WrfinputIdentity(path, MappingProxyType(dimensions),
                            MappingProxyType(attributes), tuple(eta), float(top[0]), start)


def check_wrfinput_identity(identity: WrfinputIdentity, *, domain, vertical,
                            start_time: datetime, soil_layers: int) -> None:
    """A namelist may describe file-defined state; it cannot reinterpret it."""
    attributes = identity.attributes
    # WRF stores dry theta in T regardless of this switch. The switch
    # selects THM and mass-coupled boundary T, validated at the boundary seam.
    if attributes["USE_THETA_M"] not in (0, 1):
        raise ValueError(f"{identity.path}: unknown USE_THETA_M={attributes['USE_THETA_M']}")
    cfg = domain.run
    pairs = (
        ("GRID_ID", attributes["GRID_ID"], domain.grid_id),
        ("nx", identity.dimensions["west_east"], cfg.nx),
        ("ny", identity.dimensions["south_north"], cfg.ny),
        ("nz", identity.dimensions["bottom_top"], cfg.nz),
        ("soil_layers_stag", identity.dimensions["soil_layers_stag"], soil_layers),
        ("MP_PHYSICS", attributes["MP_PHYSICS"], cfg.mp_physics),
        ("SF_SURFACE_PHYSICS", attributes["SF_SURFACE_PHYSICS"], cfg.sf_surface_physics),
        ("HYBRID_OPT", attributes["HYBRID_OPT"], vertical.hybrid_opt),
        ("start_time", identity.start_time, start_time),
    )
    differences = [f"{name}: file={found}, config={wanted}"
                   for name, found, wanted in pairs if found != wanted]
    for name, found, wanted, tolerance in (
            ("P_TOP", identity.p_top, vertical.p_top, 1e-3),
            ("ETAC", attributes["ETAC"], vertical.etac, 1e-7),
            ("DX", attributes["DX"], cfg.dx, 1e-3),
            ("DY", attributes["DY"], cfg.dy, 1e-3)):
        if not np.isfinite(float(found)) or not np.isclose(
                float(found), float(wanted), rtol=0.0, atol=tolerance):
            differences.append(f"{name}: file={found}, config={wanted}")
    requested_eta = np.asarray(vertical.eta_levels, dtype=np.float64)
    if requested_eta.shape != (len(identity.eta_levels),) or not np.allclose(
            identity.eta_levels, requested_eta, rtol=0.0, atol=1e-7):
        differences.append("ZNW: file eta levels differ from the config")
    if differences:
        raise ValueError(
            f"{identity.path}: WRF input and config disagree: "
            + "; ".join(differences)
            + ". Use the producing namelist or generate a new real.exe pair.")
