"""Small WRF-form files for the analysis bundle and door suites.

A real.exe-shaped classic file on a tiny Lambert grid: the dycore prognostics on their own
staggering, the base state, the hybrid coordinate coefficients, the geolocation, the static
fields and the WRF global attributes the bundle fingerprints read.  Hydrostatic enough for
:mod:`gpuwm.da.hydrostatic` (positive pressures, a moist column).
"""
from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np

VALID = "2022-06-13_23:00:00"


def _coordinate(nz: int):
    znw = np.linspace(1.0, 0.0, nz + 1)
    znu = 0.5 * (znw[1:] + znw[:-1])
    dnw = np.diff(znw)
    dn = np.zeros(nz)
    dn[1:] = 0.5 * (dnw[1:] + dnw[:-1])
    rdn = np.zeros(nz)
    rdn[1:] = 1.0 / dn[1:]
    return znw, znu, dnw, dn, rdn


def write_wrf_state(path: Path, *, nz: int = 4, ny: int = 6, nx: int = 7, valid: str = VALID,
                    fmt: str = "NETCDF3_64BIT_OFFSET", theta_offset: float = 0.0,
                    qv_scale: float = 1.0, mu_offset: float = 0.0, hgt_offset: float = 0.0,
                    xlat_offset: float = 0.0, znw_top: float = 0.0, extra: dict | None = None) -> Path:
    """Write one WRF-form state; the keyword offsets make related states (an analysis, a
    background, another grid)."""

    path = Path(path)
    znw, znu, dnw, dn, rdn = _coordinate(nz)
    znw = znw.copy()
    znw[-1] = znw_top
    p_top = 1.0e4
    mub = np.full((ny, nx), 9.0e4, np.float64)
    jj, ii = np.mgrid[0:ny, 0:nx]
    mu = 50.0 + 3.0 * ii + 2.0 * jj + mu_offset
    pb = znu[:, None, None] * mub[None] + p_top
    p = znu[:, None, None] * mu[None]
    t = (2.0 + 0.5 * np.arange(nz)[:, None, None] + 0.1 * ii[None] + theta_offset)
    qv = qv_scale * 0.012 * znu[:, None, None] * np.ones((nz, ny, nx))
    qr = np.zeros((nz, ny, nx))
    qr[1, 2, 3] = 1.0e-3
    ph = np.zeros((nz + 1, ny, nx))
    ph[1:] = 10.0 * np.arange(1, nz + 1)[:, None, None]
    phb = 9.81 * (np.linspace(0.0, 15000.0, nz + 1)[:, None, None] + 0.0 * ii[None])
    fields = {
        "U": (("bottom_top", "south_north", "west_east_stag"), 5.0 + np.zeros((nz, ny, nx + 1))),
        "V": (("bottom_top", "south_north_stag", "west_east"), -2.0 + np.zeros((nz, ny + 1, nx))),
        "W": (("bottom_top_stag", "south_north", "west_east"), np.zeros((nz + 1, ny, nx))),
        "T": (("bottom_top", "south_north", "west_east"), t),
        "PH": (("bottom_top_stag", "south_north", "west_east"), ph),
        "PHB": (("bottom_top_stag", "south_north", "west_east"), phb),
        "MU": (("south_north", "west_east"), mu),
        "MUB": (("south_north", "west_east"), mub),
        "P": (("bottom_top", "south_north", "west_east"), p),
        "PB": (("bottom_top", "south_north", "west_east"), pb),
        "QVAPOR": (("bottom_top", "south_north", "west_east"), qv),
        "QRAIN": (("bottom_top", "south_north", "west_east"), qr),
        "XLAT": (("south_north", "west_east"), 35.0 + 0.1 * jj + xlat_offset),
        "XLONG": (("south_north", "west_east"), -97.0 + 0.1 * ii),
        "MAPFAC_M": (("south_north", "west_east"), np.ones((ny, nx))),
        "MAPFAC_U": (("south_north", "west_east_stag"), np.ones((ny, nx + 1))),
        "MAPFAC_V": (("south_north_stag", "west_east"), np.ones((ny + 1, nx))),
        "HGT": (("south_north", "west_east"), 300.0 + 10.0 * ii + hgt_offset),
        "LANDMASK": (("south_north", "west_east"), np.ones((ny, nx))),
        "LU_INDEX": (("south_north", "west_east"), np.full((ny, nx), 10.0)),
        "ZNW": (("bottom_top_stag",), znw),
        "ZNU": (("bottom_top",), znu),
        "DNW": (("bottom_top",), dnw),
        "DN": (("bottom_top",), dn),
        "RDNW": (("bottom_top",), 1.0 / dnw),
        "RDN": (("bottom_top",), rdn),
        "C1H": (("bottom_top",), np.ones(nz)), "C2H": (("bottom_top",), np.zeros(nz)),
        "C1F": (("bottom_top_stag",), np.ones(nz + 1)), "C2F": (("bottom_top_stag",), np.zeros(nz + 1)),
        "C3H": (("bottom_top",), znu), "C4H": (("bottom_top",), np.zeros(nz)),
        "C3F": (("bottom_top_stag",), znw), "C4F": (("bottom_top_stag",), np.zeros(nz + 1)),
        "P_TOP": ((), np.asarray(p_top)),
    }
    for name, value in (extra or {}).items():
        fields[name] = value
    with netCDF4.Dataset(path, "w", format=fmt) as dataset:
        for name, size in (("Time", None), ("DateStrLen", 19), ("west_east", nx), ("west_east_stag", nx + 1),
                           ("south_north", ny), ("south_north_stag", ny + 1), ("bottom_top", nz),
                           ("bottom_top_stag", nz + 1)):
            dataset.createDimension(name, size)
        times = dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[0, :] = np.array(list(valid), dtype="S1")
        for name, (dims, values) in fields.items():
            variable = dataset.createVariable(name, "f4", ("Time",) + tuple(dims))
            variable[0] = np.asarray(values, dtype=np.float32)
        dataset.setncatts({
            "TITLE": " OUTPUT FROM REAL_EM V4.6.1 PREPROCESSOR", "START_DATE": valid,
            "SIMULATION_START_DATE": valid, "DX": np.float32(3000.0), "DY": np.float32(3000.0),
            "MAP_PROJ": np.int32(1), "TRUELAT1": np.float32(30.0), "TRUELAT2": np.float32(60.0),
            "STAND_LON": np.float32(-97.0), "MOAD_CEN_LAT": np.float32(35.0), "CEN_LAT": np.float32(35.25),
            "CEN_LON": np.float32(-96.7), "HYBRID_OPT": np.int32(2), "ETAC": np.float32(0.2),
            "USE_THETA_M": np.int32(0), "GRID_ID": np.int32(1)})
    return path


def bundle_identity() -> dict:
    """The producer identity :func:`gpuwm.io.analysis_bundle.make_bundle` takes."""

    return dict(grid_id="test-3km", static_builder="real.exe 4.6.1", physics_suite="arwen-cp",
                physics_mode="correct", kind="mean", members=36, recipe_id="filter.hybrid-1",
                recipe_sha256="a" * 64, cycles=5, window=900.0, producer="test-producer 1")


#: wrfbdy base name -> (logical key, z dimension or None).
_BDY = {"U": ("u", "bottom_top"), "V": ("v", "bottom_top"), "T": ("theta", "bottom_top"),
        "PH": ("phi", "bottom_top_stag"), "MU": ("mu", None), "QVAPOR": ("qv", "bottom_top")}
_SIDE_DIM = {("U", "X"): "south_north", ("U", "Y"): "west_east_stag",
             ("V", "X"): "south_north_stag", ("V", "Y"): "west_east"}


def raw_fields(path: Path) -> dict:
    with netCDF4.Dataset(path) as dataset:
        return {name: np.asarray(variable[0], dtype=np.float32)
                for name, variable in dataset.variables.items() if name != "Times"}


def write_wrfbdy(path: Path, start: Path, *, width: int = 2, interval: int = 3600, records: int = 2,
                 tendency: float = 1.0e-3) -> Path:
    """A real.exe-shaped wrfbdy whose first record is ``start``'s own coupling (the head the
    WRF-input door checks a pair against), each later record the previous one carried by its
    tendency."""

    from datetime import datetime, timedelta

    from gpuwm.ingest.wrfinput import _boundary_strip, wrf_initial_boundary_head
    from gpuwm.wrf_direct import _side_to_wrf

    raw = raw_fields(start)
    with netCDF4.Dataset(start) as dataset:
        attrs = {name: dataset.getncattr(name) for name in dataset.ncattrs()}
        sizes = {name: len(dim) for name, dim in dataset.dimensions.items()}
    head = wrf_initial_boundary_head(raw, {key: (base,) for base, (key, _z) in _BDY.items()},
                                     moist_theta=int(attrs["USE_THETA_M"]) == 1)
    first = datetime.strptime(attrs["START_DATE"], "%Y-%m-%d_%H:%M:%S")
    stamps = [(first + timedelta(seconds=interval * k)).strftime("%Y-%m-%d_%H:%M:%S")
              for k in range(records)]
    nexts = [(first + timedelta(seconds=interval * (k + 1))).strftime("%Y-%m-%d_%H:%M:%S")
             for k in range(records)]
    sides = {"XS": "west", "XE": "east", "YS": "south", "YE": "north"}
    with netCDF4.Dataset(path, "w", format="NETCDF3_64BIT_OFFSET") as dataset:
        for name, size in (("Time", None), ("DateStrLen", 19), ("bdy_width", width)):
            dataset.createDimension(name, size)
        for name in ("west_east", "west_east_stag", "south_north", "south_north_stag", "bottom_top",
                     "bottom_top_stag"):
            dataset.createDimension(name, sizes[name])
        times = dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
        this = dataset.createVariable("md___thisbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_", "S1",
                                      ("Time", "DateStrLen"))
        nxt = dataset.createVariable("md___nextbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_", "S1",
                                     ("Time", "DateStrLen"))
        for k in range(records):
            times[k, :] = np.array(list(stamps[k]), dtype="S1")
            this[k, :] = np.array(list(stamps[k]), dtype="S1")
            nxt[k, :] = np.array(list(nexts[k]), dtype="S1")
        for base, (key, zdim) in _BDY.items():
            for suffix, side in sides.items():
                axis = suffix[0]
                side_dim = _SIDE_DIM.get((base, axis)) or ("south_north" if axis == "X" else "west_east")
                dims = ("Time", "bdy_width") + ((zdim,) if zdim else ()) + (side_dim,)
                value = np.asarray(_side_to_wrf(_boundary_strip(head[key], side, width), key, side),
                                   np.float64)
                rate = np.full_like(value, tendency * (1.0 if key != "mu" else 100.0))
                b = dataset.createVariable(f"{base}_B{suffix}", "f4", dims)
                bt = dataset.createVariable(f"{base}_BT{suffix}", "f4", dims)
                for k in range(records):
                    b[k] = value.astype(np.float32)
                    bt[k] = rate.astype(np.float32)
                    value = value + interval * rate
        dataset.setncatts({name: attrs[name] for name in (
            "TITLE", "START_DATE", "DX", "DY", "MAP_PROJ", "TRUELAT1", "TRUELAT2", "STAND_LON", "CEN_LAT",
            "CEN_LON", "HYBRID_OPT", "ETAC", "USE_THETA_M", "GRID_ID")})
    return path
