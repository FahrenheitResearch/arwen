"""Which library writes the gridded observation products.

``gpuwm-obs.radar-grid.v1`` and ``gpuwm-obs.goes-grid.v1`` are the two files
the DA lanes read and nothing upstream of them.  Both are written here, and
both are written by the SAME engine decision, because a cycle whose radar
product is classic and whose satellite product is HDF5 -- because the
environment moved between two calls of one door -- is a file set nobody
declared.

``rust`` is the DEFAULT and needs no flag: the schema goes through
:class:`gpuwm.io.classic_product.ClassicProduct` onto the dependency-free
classic writer at ``tools/rustwx/crates/netcdf-writer``.  ``python`` reaches
the netCDF4/HDF5 writer these products used through 2.4, and is an ANNOUNCED
WORKAROUND: it is selected only by naming :data:`OBS_GRID_WRITER_ENV`, never
on its own and never as a fallback.

The read halves are unaffected and were already Rust: both products decode
their observation arrays through :mod:`gpuwm.netcdf_bridge`.
"""

from __future__ import annotations

import os
from pathlib import Path

#: The environment variable that selects the netCDF4 workaround engine.
OBS_GRID_WRITER_ENV = "GPUWM_OBS_GRID_WRITER"

_ENGINES = ("rust", "python")


class ObsGridWriterUnavailable(RuntimeError):
    """The Rust classic writer is not loadable, and nothing replaces it."""


def resolve_obs_grid_engine(engine: str | None = None) -> str:
    """The writer engine: explicit argument, else the environment, else rust."""

    value = engine if engine is not None else os.environ.get(
        OBS_GRID_WRITER_ENV, "rust")
    value = str(value).strip().lower()
    if value not in _ENGINES:
        raise ValueError(
            f"unknown observation-product writer engine {value!r} (from "
            f"{OBS_GRID_WRITER_ENV!r} or the engine argument); expected one "
            f"of {_ENGINES}.  'rust' is the default classic writer; 'python' "
            f"is the netCDF4 workaround.")
    return value


def open_obs_grid_product(path: Path | str, *, engine: str | None = None):
    """Open ``path`` for writing one gridded observation product.

    Returns a ``netCDF4.Dataset``-shaped writer usable as a context manager.
    Both engines answer the same surface, so the product writers above have
    one shape rather than two.
    """

    selected = resolve_obs_grid_engine(engine)
    if selected == "python":
        import netCDF4                                   # noqa: PLC0415

        return netCDF4.Dataset(path, "w", format="NETCDF4_CLASSIC")

    from gpuwm.io import classic_product                 # noqa: PLC0415
    from gpuwm.io import nc_writer_bridge                # noqa: PLC0415

    reason = nc_writer_bridge.unavailable_reason()
    if reason is not None:
        raise ObsGridWriterUnavailable(
            f"the Rust NetCDF writer is not loadable here ({reason}), so "
            f"{Path(path).name} cannot be written.\n\n"
            "There is deliberately no automatic fallback.  Falling back to "
            "netCDF4 would mean the container of an observation product "
            "depends on which box happened to write it: the same DA cycle "
            "would emit a classic radar file on one node and an HDF5 one on "
            "another, the dual-write parity check would stop comparing the "
            "shipped path against anything, and the 'default' writer would "
            "quietly stop being the one that is tested.\n\n"
            "Build it from a checkout:\n"
            "  cd tools/rustwx && cargo build --release --offline\n"
            f"or point {nc_writer_bridge.NCWRITE_BRIDGE_ENV} at a built "
            "library.\n"
            f"To write this product with netCDF4 instead, as an EXPLICIT "
            f"workaround, set {OBS_GRID_WRITER_ENV}=python.")
    return classic_product.ClassicProduct(path)


__all__ = [
    "OBS_GRID_WRITER_ENV",
    "ObsGridWriterUnavailable",
    "open_obs_grid_product",
    "resolve_obs_grid_engine",
]
