"""Reusable real-weather verification metrics and map primitives.

Phase 5, Task 7 extracts these operations from the frozen
``real74_d01`` case.  The arithmetic, operation order, dtypes, masks, map
layout, labels, and filenames are intentionally unchanged.  Case modules
provide policy through the frozen map-spec records below; this module owns
only reusable readers, scorers, diagnostics, and renderers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def boundary_zone_blowup(boundary_w_max: float,
                         interior_w_max: float) -> bool:
    """Frozen real74 boundary-reflection diagnostic.

    This is the exact predicate historically evaluated by
    :func:`gpuwm.runtime.run_prepared_real_case`: a non-finite boundary
    maximum fires, as does a boundary maximum more than five times the
    larger of the free-interior maximum and 1 m/s.  N2(c) imports this
    function so its ``boundary_zone_blowup`` evidence cannot drift from the
    frozen real74 metric definition.
    """
    return bool(not np.isfinite(boundary_w_max)
                or boundary_w_max > 5.0 * max(interior_w_max, 1.0))


def _read_wrf_field(ds, name: str) -> np.ndarray:
    value = np.asarray(np.ma.filled(ds.variables[name][0], np.nan),
                       dtype=np.float64)
    return value


def _interpolate_to_pressure(field: np.ndarray, pressure_hpa: np.ndarray,
                             target_hpa: float) -> np.ndarray:
    """Log-pressure interpolation on WRF bottom-to-top model columns."""
    if field.shape != pressure_hpa.shape or field.ndim != 3:
        raise ValueError("field and pressure must share a 3-D WRF mass grid")
    valid_column = (np.isfinite(field).all(axis=0)
                    & np.isfinite(pressure_hpa).all(axis=0)
                    & (target_hpa <= pressure_hpa[0])
                    & (target_hpa >= pressure_hpa[-1]))
    upper = np.sum(pressure_hpa >= target_hpa, axis=0)
    upper = np.clip(upper, 1, pressure_hpa.shape[0] - 1)
    lower = upper - 1
    lo = np.take_along_axis(field, lower[None], axis=0)[0]
    hi = np.take_along_axis(field, upper[None], axis=0)[0]
    plo = np.take_along_axis(pressure_hpa, lower[None], axis=0)[0]
    phi = np.take_along_axis(pressure_hpa, upper[None], axis=0)[0]
    weight = ((np.log(target_hpa) - np.log(plo))
              / (np.log(phi) - np.log(plo)))
    out = lo + weight * (hi - lo)
    return np.where(valid_column, out, np.nan)


def _dcomputeseaprs(pressure_pa, temperature, qv, height_msl) -> np.ndarray:
    """Sea-level pressure (hPa): wrf-python DCOMPUTESEAPRS (Shuell 1995).

    Transcribed from the validated wrf-rust oracle
    (wrf-core/src/diag/pressure.rs:60-147, which the metrics audit
    matched to wrf-python's Fortran to 4.6e-10 hPa on this case):
    reference level ~PCONST = 100 hPa above the surface, log-p
    interpolation of Tv/z to it, US-standard-lapse surface and
    sea-level temperatures, the TC = 290.66 K "ridiculous MM5 test"
    capping, and the two-temperature exponential reduction with
    wrf-python's rounded G = 9.81 / Rd = 287.  ``height_msl`` is full
    geopotential / 9.80665 (the oracle's height_msl).  Inputs are
    (nz, ny, nx) bottom-to-top with pressure in Pa; returns (ny, nx) hPa.
    """
    g_slp, rd_slp, ussalr = 9.81, 287.0, 0.0065
    pconst, tc = 10000.0, 273.16 + 17.5
    p_sfc = pressure_pa[0]
    above = (p_sfc[None] - pressure_pa) >= pconst
    if not above.any(axis=0).all():
        raise ValueError(
            "DCOMPUTESEAPRS requires a level 100 hPa above every surface")
    klo = np.argmax(above, axis=0)
    if int(klo.min()) < 1:
        raise ValueError("DCOMPUTESEAPRS reference level fell on k=0")
    khi = klo - 1

    def at(field, k):
        return np.take_along_axis(field, k[None], axis=0)[0]

    plo, phi_p = at(pressure_pa, klo), at(pressure_pa, khi)
    # Virtual temperature with wrf-python's 0.608 (pressure.rs:106-110).
    tlo = at(temperature, klo) * (1.0 + 0.608 * np.maximum(at(qv, klo), 0.0))
    thi = at(temperature, khi) * (1.0 + 0.608 * np.maximum(at(qv, khi), 0.0))
    zlo, zhi = at(height_msl, klo), at(height_msl, khi)
    p_at_pconst = p_sfc - pconst
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = ((np.log(p_at_pconst) - np.log(phi_p))
                / (np.log(plo) - np.log(phi_p)))
    degenerate = np.abs(plo - phi_p) < 1.0
    t_at_pconst = np.where(degenerate, tlo, thi + frac * (tlo - thi))
    z_at_pconst = np.where(degenerate, zlo, zhi + frac * (zlo - zhi))
    t_surf = t_at_pconst * (p_sfc / p_at_pconst) ** (ussalr * rd_slp / g_slp)
    t_sea_level = t_at_pconst + ussalr * z_at_pconst
    t_sea_level = np.where((t_surf <= tc) & (t_sea_level >= tc), tc,
                           tc - 0.005 * (t_surf - tc) ** 2)
    return 0.01 * p_sfc * np.exp(
        2.0 * g_slp * height_msl[0] / (rd_slp * (t_sea_level + t_surf)))


def _wrf_diagnostics(path: Path) -> dict[str, object]:
    """Read comparison and map fields without a wrf-python dependency."""
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        pressure_pa = (_read_wrf_field(ds, "P")
                       + _read_wrf_field(ds, "PB"))
        pressure = pressure_pa / 100.0
        theta = _read_wrf_field(ds, "T") + 300.0
        # WRF rcp = r_d/cp with cp = 7*r_d/2 = 1004.5
        # (share/module_model_constants.F:19-20,31).
        temperature = theta * (pressure / 1000.0) ** (287.0 / 1004.5)
        u_stag = _read_wrf_field(ds, "U")
        v_stag = _read_wrf_field(ds, "V")
        u = 0.5 * (u_stag[..., :-1] + u_stag[..., 1:])
        v = 0.5 * (v_stag[:, :-1, :] + v_stag[:, 1:, :])
        phi = _read_wrf_field(ds, "PH") + _read_wrf_field(ds, "PHB")
        height = 0.5 * (phi[:-1] + phi[1:]) / 9.81
        qv = _read_wrf_field(ds, "QVAPOR")
        # Audit R16: the previous single-level exponential lift left a
        # terrain-correlated deviation vs the standard diagnostic (rms
        # 1.41 hPa, max 9.79 hPa over the Rockies); score the maps and
        # gate on DCOMPUTESEAPRS instead.
        mslp = _dcomputeseaprs(pressure_pa, temperature, qv,
                               0.5 * (phi[:-1] + phi[1:]) / 9.80665)
        levels = {}
        for level in (500, 700, 850):
            levels[level] = {
                "temperature": _interpolate_to_pressure(
                    temperature, pressure, level),
                "u": _interpolate_to_pressure(u, pressure, level),
                "v": _interpolate_to_pressure(v, pressure, level),
                "height": _interpolate_to_pressure(height, pressure, level),
            }
        return {
            "pressure": pressure, "levels": levels, "mslp": mslp,
            "lat": _read_wrf_field(ds, "XLAT"),
            "lon": _read_wrf_field(ds, "XLONG"),
            "t2": _read_wrf_field(ds, "T2"),
            "rainnc": _read_wrf_field(ds, "RAINNC"),
            "rainc": (_read_wrf_field(ds, "RAINC")
                      if "RAINC" in ds.variables
                      else np.zeros_like(_read_wrf_field(ds, "RAINNC"))),
        }


def interior_region(shape) -> tuple[slice, slice]:
    """Exclude a five-cell specified-plus-relaxation Davies frame."""
    return ((slice(5, -5), slice(5, -5))
            if min(shape) > 10 else (slice(None), slice(None)))


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    # Pressure-level references legitimately carry NaNs where the requested
    # surface is outside a column.  Those reference cells define missing
    # support; a non-finite candidate on finite reference support is instead
    # a failed comparison and must not shrink the sample opportunistically.
    reference_support = np.isfinite(b)
    if (np.isinf(a).any()
            or not reference_support.any()
            or not np.isfinite(a[reference_support]).all()):
        return float("nan")
    delta = a[reference_support] - b[reference_support]
    return float(np.sqrt(np.mean(delta * delta, dtype=np.float64)))


def _pattern_correlation(a: np.ndarray, b: np.ndarray) -> float:
    reference_support = np.isfinite(b)
    if (np.isinf(a).any()
            or reference_support.sum() < 2
            or not np.isfinite(a[reference_support]).all()):
        return float("nan")
    aa = a[reference_support] - np.mean(
        a[reference_support], dtype=np.float64)
    bb = b[reference_support] - np.mean(
        b[reference_support], dtype=np.float64)
    scale = float(np.sqrt(np.dot(aa, aa) * np.dot(bb, bb)))
    if scale == 0.0:
        return (1.0 if np.array_equal(
            a[reference_support], b[reference_support]) else float("nan"))
    return float(np.dot(aa, bb) / scale)


def score_pair(model: np.ndarray, source: np.ndarray, *,
               mask: str = "interior") -> tuple[float, float]:
    """Return RMSE and pattern correlation under a named domain mask."""
    if model.shape != source.shape or model.ndim != 2:
        raise ValueError("model and source must share a two-dimensional grid")
    if mask == "interior":
        region = interior_region(model.shape)
    elif mask == "full":
        region = (slice(None), slice(None))
    else:
        raise ValueError(f"unsupported verification mask {mask!r}")
    return (_rmse(model[region], source[region]),
            _pattern_correlation(model[region], source[region]))


@dataclass(frozen=True)
class SynopticMapSpec:
    """Case-owned labels and filenames for the three synoptic maps."""

    mslp_t2_filename: str
    mslp_t2_title: str
    height_wind_filename: str
    height_wind_title: str
    precip_filename: str
    precip_title: str


def make_synoptic_maps(final_path, accumulation_start_path, output_dir,
                       spec: SynopticMapSpec) -> tuple[Path, ...]:
    """Write MSLP/T2, 500-hPa height/wind, and precipitation maps."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final = _wrf_diagnostics(Path(final_path))
    accumulation_start = _wrf_diagnostics(Path(accumulation_start_path))
    lon, lat = final["lon"], final["lat"]
    paths = []

    path = output_dir / spec.mslp_t2_filename
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    shading = ax.pcolormesh(lon, lat, final["t2"] - 273.15,
                            shading="auto", cmap="coolwarm")
    contours = ax.contour(lon, lat, final["mslp"], colors="black",
                          linewidths=0.8)
    ax.clabel(contours, inline=True, fontsize=7, fmt="%.0f")
    fig.colorbar(shading, ax=ax, label="2 m temperature (degC)")
    ax.set(title=spec.mslp_t2_title,
           xlabel="Longitude", ylabel="Latitude")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    level = final["levels"][500]
    path = output_dir / spec.height_wind_filename
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    contours = ax.contour(lon, lat, level["height"], colors="black",
                          linewidths=0.9)
    ax.clabel(contours, inline=True, fontsize=7, fmt="%.0f")
    stride = max(min(lon.shape) // 18, 1)
    ax.quiver(lon[::stride, ::stride], lat[::stride, ::stride],
              level["u"][::stride, ::stride],
              level["v"][::stride, ::stride], scale=500.0)
    ax.set(title=spec.height_wind_title,
           xlabel="Longitude", ylabel="Latitude")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    precip = np.maximum((final["rainnc"] + final["rainc"])
                        - (accumulation_start["rainnc"]
                           + accumulation_start["rainc"]), 0.0)
    path = output_dir / spec.precip_filename
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    shading = ax.pcolormesh(lon, lat, precip, shading="auto", cmap="Blues")
    fig.colorbar(shading, ax=ax,
                 label="6 h accumulated total precipitation (mm)")
    ax.set(title=spec.precip_title,
           xlabel="Longitude", ylabel="Latitude")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    return tuple(paths)


@dataclass(frozen=True)
class ReflectivityMapSpec:
    """Case-owned output policy for a composite-reflectivity map."""

    filename: str
    title: str
    bounds: tuple[float, ...]
    colors: tuple[str, ...]


def make_composite_reflectivity_map(
        wrfout_path, output_dir, spec: ReflectivityMapSpec, *, time_index=0
        ) -> Path:
    """Write a column-maximum simulated-reflectivity PNG."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    import netCDF4

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(wrfout_path) as ds:
        if "REFL_10CM" not in ds.variables:
            raise ValueError(f"{wrfout_path} carries no REFL_10CM variable; "
                             "write frames with the reflectivity opt-in "
                             "before mapping it")
        refl = np.asarray(np.ma.filled(
            ds.variables["REFL_10CM"][time_index], np.nan), dtype=np.float64)
        lat = np.asarray(ds.variables["XLAT"][time_index], dtype=np.float64)
        lon = np.asarray(ds.variables["XLONG"][time_index], dtype=np.float64)
    if not np.isfinite(refl).all():
        raise ValueError("REFL_10CM map input is non-finite")
    composite = np.max(refl, axis=0)

    path = output_dir / spec.filename
    cmap = ListedColormap(spec.colors)
    norm = BoundaryNorm(spec.bounds, cmap.N)
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    ax.set_facecolor("0.92")
    shading = ax.pcolormesh(
        lon, lat, np.ma.masked_less(composite, spec.bounds[0]),
        cmap=cmap, norm=norm, shading="auto")
    fig.colorbar(shading, ax=ax, ticks=spec.bounds,
                 label="composite reflectivity (dBZ)")
    ax.set(title=spec.title,
           xlabel="Longitude", ylabel="Latitude")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# Public spellings for new profiles; underscored aliases remain the exact
# frozen surface imported by real74_d01 and its regression tests.
rmse = _rmse
pattern_correlation = _pattern_correlation
interpolate_to_pressure = _interpolate_to_pressure
wrf_diagnostics = _wrf_diagnostics


__all__ = [
    "ReflectivityMapSpec", "SynopticMapSpec", "boundary_zone_blowup",
    "interior_region", "interpolate_to_pressure",
    "make_composite_reflectivity_map", "make_synoptic_maps",
    "pattern_correlation", "rmse", "score_pair", "wrf_diagnostics",
]
