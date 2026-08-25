"""Reusable real-weather verification metrics and map primitives.

Phase 5, Task 7 extracts these operations from the frozen
``real74_d01`` case.  The arithmetic, operation order, dtypes and masks
are intentionally unchanged.  Case modules provide policy through the
frozen map-spec records below; this module owns only reusable readers,
scorers, diagnostics, and the seam that drives the renderer.

MAP LAYOUT, LABELS AND FILENAMES ARE NO LONGER THIS MODULE'S.  Every
panel here is a weather field, the render law (CLAUDE.md, Drew
2026-08-06) reserves those for the production Rust renderer
``rw_wrfbatch`` driven through :mod:`gpuwm.rustwx`, and until the
2026-08-18 hidden-scope audit they were drawn with
``matplotlib.pcolormesh``/``contour``/``quiver`` right here, on the bare
default of ``gpuwm verify`` (audit F6).  They come from the renderer
now, filed under the same ``<out>/<domain>/<product>/<valid-day>/``
layout as every other render, and a run with no usable renderer emits
NO weather-field imagery and says so rather than substituting an engine
the law does not allow.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gpuwm import science_core as _pin
#: The sea-level reduction and its smoother moved to ``gpuwm.core.mslp``
#: so ``gpuwm.core.storm_tracking`` can reach them without importing this
#: developer verification tree, which the standalone preprocessing
#: distribution omits.  Re-exported rather than restated: there is still
#: one definition of each, and every case module and gate that binds
#: ``metrics._dcomputeseaprs`` binds the same object it always did.
from gpuwm.core.mslp import (  # noqa: F401
    MSLP_SMOOTH_CENTER_WEIGHT,
    MSLP_SMOOTH_PASSES,
    _dcomputeseaprs,
    _nine_point_smooth,
)


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


def _science_core():
    """The mandated science core, or a refusal naming what to install.

    Function-local so importing this module -- which every real-weather
    case does -- never depends on the extra being present; the refusal
    lands on the call that needs it, with the install line.
    """

    try:
        import wrf
    except ImportError:
        raise ImportError(
            f"the mandated science core is not installed, and gpuwm has no "
            f"Python reimplementation of this diagnostic.\n"
            f"  pip install '{_pin.SCIENCE_CORE_REQUIREMENT}'\n"
            f"  # or: pip install 'gpuwm[render]'") from None
    version = _pin.installed_science_core_version()
    if not _pin.version_supported(version):
        raise ImportError(_pin.science_core_refusal(version))
    return wrf


def _interpolate_to_pressure(field: np.ndarray, pressure_hpa: np.ndarray,
                             target_hpa: float) -> np.ndarray:
    """Log-pressure interpolation on WRF bottom-to-top model columns.

    Delegates to the science core's ``interplevel``.  This function used
    to carry its own numpy transcription of that interpolation -- a
    take_along_axis bracket search and a log-p weight -- which is the
    kind of duplicate this project does not keep: the operation exists
    in Rust, in the library gpuwm mandates for exactly these
    diagnostics.  The two were measured against each other on a
    30-level synthetic column set at 925/850/700/500/300 hPa before the
    transcription was removed, and agreed to 0.000e+00 over 2,500
    compared points at every level.

    The shape contract stays here because it is this module's contract,
    not the core's, and because a mismatched pair must be refused with
    the message the cases already expect.
    """

    if field.shape != pressure_hpa.shape or field.ndim != 3:
        raise ValueError("field and pressure must share a 3-D WRF mass grid")
    return np.asarray(
        _science_core().interplevel(field, pressure_hpa, float(target_hpa)),
        dtype=np.float64)


#: Terrain band (metres MSL) over which the MSLP display treatment ramps
#: in.  Below the floor the reduction increment ln(SLP/p_sfc) is small
#: (<= ~0.06 at 500 m) so its noise is visually negligible and the
#: treatment is the exact identity; above the ceiling the increment is
#: large enough that its grid-scale noise dominates the contour field
#: and the smoothed increment is used in full.  Keyed on terrain height
#: only -- never on region or case identity.
MSLP_SMOOTH_TERRAIN_FLOOR_M = 500.0
MSLP_SMOOTH_TERRAIN_CEILING_M = 1500.0

def terrain_smoothed_mslp(mslp_hpa: np.ndarray,
                          surface_pressure_pa: np.ndarray,
                          terrain_m: np.ndarray, *,
                          passes: int = MSLP_SMOOTH_PASSES,
                          center_weight: float = MSLP_SMOOTH_CENTER_WEIGHT,
                          ) -> np.ndarray:
    """Display-grade MSLP: smooth the below-terrain extrapolation only.

    The sea-level reduction ends with ``SLP = p_sfc * exp(r)`` where the
    increment ``r = 2 g z_sfc / (Rd (T_sl + T_sfc))`` is reconstructed
    here as ``ln(SLP / p_sfc)``.  ``r`` is proportional to terrain
    height, so grid-scale variation in the fictitious below-ground
    temperatures that is invisible at sea level (fractions of an hPa)
    scribbles multi-hPa noise across high terrain -- the published
    failure mode of station-style reductions (Benjamin & Miller 1990,
    Mon. Wea. Rev. 118, 2099-2116, who replace or smooth the
    below-ground extrapolation for exactly this reason).  Across the
    whole certified window wrf-rust exposes only the raw reduction --
    re-MEASURED on 0.2.38, whose ``list_variables()`` carries exactly one
    sea-level entry, ``slp`` "Sea-level pressure", and no smoothed or
    display variant -- so the standard display treatment is applied here,
    from the published algorithm:

    1. ``r`` is smoothed with the standard nine-point smoother
       (:data:`MSLP_SMOOTH_PASSES` passes, Shuman 1957 family -- the
       treatment wrf-python's SLP plotting workflow documents).
    2. The smoothed candidate ``p_sfc * exp(smooth(r))`` is blended
       with the raw field on a weight that ramps 0 -> 1 across
       :data:`MSLP_SMOOTH_TERRAIN_FLOOR_M` ..
       :data:`MSLP_SMOOTH_TERRAIN_CEILING_M`, keyed on terrain height
       ONLY.  At or below the floor the result is the input, exactly
       (``np.where`` identity, zero tolerance): over low terrain the
       raw field is already clean and must not move.

    The scored/gated ``mslp`` is deliberately NOT this field: gates
    stay on the frozen raw DCOMPUTESEAPRS transcription (audit R16).

    ``mslp_hpa`` is the raw reduction (hPa), ``surface_pressure_pa``
    the pressure the reduction started from (Pa, the lowest mass
    level, i.e. ``pressure_pa[0]``), ``terrain_m`` the surface height
    (m MSL).  All three share one 2-D grid.
    """
    mslp = np.asarray(mslp_hpa, dtype=np.float64)
    p_sfc = np.asarray(surface_pressure_pa, dtype=np.float64)
    terrain = np.asarray(terrain_m, dtype=np.float64)
    if (mslp.ndim != 2 or mslp.shape != p_sfc.shape
            or mslp.shape != terrain.shape):
        raise ValueError(
            "mslp, surface pressure, and terrain must share one "
            "two-dimensional grid")
    weight = np.clip(
        (terrain - MSLP_SMOOTH_TERRAIN_FLOOR_M)
        / (MSLP_SMOOTH_TERRAIN_CEILING_M - MSLP_SMOOTH_TERRAIN_FLOOR_M),
        0.0, 1.0)
    if not np.any(weight > 0.0):
        return mslp.copy()
    increment = np.log(mslp / (0.01 * p_sfc))
    candidate = 0.01 * p_sfc * np.exp(
        _nine_point_smooth(increment, passes, center_weight))
    return np.where(weight > 0.0, mslp + weight * (candidate - mslp), mslp)


def _wrf_diagnostics(path: Path) -> dict[str, object]:
    """Read comparison and map fields without a wrf-python dependency."""
    from gpuwm import netcdf_bridge

    with netcdf_bridge.open_dataset(path) as ds:
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
        # Display twin of ``mslp``: the below-terrain extrapolation is
        # smoothed, keyed on terrain height (the staggered surface
        # geopotential level).  Scoring and gates keep the raw field.
        mslp_display = terrain_smoothed_mslp(
            mslp, pressure_pa[0], phi[0] / 9.80665)
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
            "mslp_display": mslp_display,
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


#: Panel size for every verification map, in pixels.  The renderer's
#: own default aspect, matching ``gpuwm render``'s ``--size``, so a
#: verification panel and a product panel of the same field are the same
#: picture at the same scale.
SYNOPTIC_MAP_SIZE = (1200, 900)


@dataclass(frozen=True)
class SynopticMapSpec:
    """Case-owned product policy for a case's synoptic panels.

    ``products`` are slugs in ``rw_wrfbatch``'s OWN catalog, and that is
    the whole spec.  It used to carry a filename and a title per figure,
    because this module drew the figures; under the render law (CLAUDE.md,
    Drew 2026-08-06) it does not, and titles, palettes, projections and
    filenames all belong to the renderer that the campaign product sheets
    were proven pixel-identical against.  What stays case-owned is WHICH
    charts a case wants.
    """

    products: tuple[str, ...]
    #: Provenance label stamped on every panel.  ``None`` takes
    #: ``gpuwm.render.default_source_label()`` -- the brand plus the
    #: version that is EXECUTING -- which is what the render door uses.
    source_label: str | None = None


def _renderer_series(paths, output_dir, products, source_label,
                     *, frames=None) -> tuple[Path, ...]:
    """Draw ``products`` for a wrfout series with the production renderer.

    ONE store over the whole series, because the verification set is
    windowed: ``qpf_6h`` is F012 minus F006 and there is no way to
    difference two frames that never met.  Everything else -- the
    contract handshake, the provenance gate, the
    ``<out>/<domain>/<product>/<valid-day>/`` layout -- is the render
    door's, reached through :mod:`gpuwm.render`, so a verification panel
    cannot come from an engine the render door would refuse.
    """

    from gpuwm import render

    written, failures, skipped = render.render_series_rust(
        [Path(item) for item in paths], products=",".join(products),
        timeidx=frames, outdir=Path(output_dir), size=SYNOPTIC_MAP_SIZE,
        source_label=source_label)
    for failure in failures:
        print(f"verify: render FAILED: {failure}", file=sys.stderr)
    for slug, reason in skipped:
        print(f"verify: no {slug} panel: {reason}", file=sys.stderr)
    return tuple(written)


def make_synoptic_maps(final_path, accumulation_start_path, output_dir,
                       spec: SynopticMapSpec) -> tuple[Path, ...]:
    """The case's synoptic panels, drawn by ``rw_wrfbatch``, or ``()``.

    Both frames go into one store: the accumulation start supplies the
    window the 6 h QPF panel is differenced against, and the final frame
    supplies everything else.

    With no usable renderer this door emits NO weather-field imagery and
    says so, at the point the imagery would have appeared.  That is the
    lawful degradation for a verification run (the product of ``gpuwm
    verify`` is its metrics and gates; the panels are evidence), and it
    replaces the defect the 2026-08-18 hidden-scope audit filed as F6:
    four of the exact product classes the render law names -- MSLP, 2 m
    temperature, 500 hPa height with a wind quiver, 6 h accumulated
    precipitation -- drawn with ``matplotlib.pcolormesh``/``contour``/
    ``quiver`` on the BARE DEFAULT of the door, with no renderer probe
    and no announcement anywhere in ``gpuwm/verify/``.  Omitting
    ``--outdir`` did not skip them; it sent them to a temporary
    directory, so there was no way to run the case without the unlawful
    drawing happening.
    """

    from gpuwm import render

    output_dir = Path(output_dir)
    try:
        render.require_renderer()
    except RuntimeError as refusal:
        print("verify: no weather-field imagery for this run -- "
              + str(refusal).split("[[explain]]")[0].strip()
              + "\n  The verification metrics and gates below are "
                "unaffected; only the panels are missing.",
              file=sys.stderr)
        return ()
    return _renderer_series(
        (accumulation_start_path, final_path), output_dir, spec.products,
        spec.source_label)


@dataclass(frozen=True)
class ReflectivityMapSpec:
    """Case-owned product policy for a composite-reflectivity panel.

    The palette and the title were this module's until the render law
    took them back: ``composite_reflectivity`` is a catalog product with
    its own operational dBZ ramp, and a second ramp declared here is the
    duplicate-of-a-shipped-product this project keeps paying for.
    """

    product: str = "composite_reflectivity"
    source_label: str | None = None


def make_composite_reflectivity_map(
        wrfout_path, output_dir, spec: ReflectivityMapSpec, *, time_index=0
        ) -> Path:
    """One column-maximum simulated-reflectivity panel, from the renderer.

    A REFUSAL rather than a degradation when the renderer is not usable,
    and the asymmetry with :func:`make_synoptic_maps` is deliberate: this
    is called by someone who asked for exactly one picture, so returning
    nothing would answer a direct request with silence.  The synoptic set
    is evidence attached to a gate run that still has its metrics.
    """

    written = _renderer_series(
        (wrfout_path,), output_dir, (spec.product,), spec.source_label,
        frames=int(time_index))
    if not written:
        raise RuntimeError(
            f"{Path(wrfout_path)}: the renderer drew no "
            f"{spec.product} panel -- most often the frame carries no "
            "REFL_10CM (write frames with the reflectivity opt-in), and "
            "the SKIPPED reason above names the field it wanted")
    if len(written) != 1:
        raise RuntimeError(
            f"expected one {spec.product} panel, got {len(written)}: "
            f"{[path.name for path in written]}")
    return written[0]


# Public spellings for new profiles; underscored aliases remain the exact
# frozen surface imported by real74_d01 and its regression tests.
rmse = _rmse
pattern_correlation = _pattern_correlation
interpolate_to_pressure = _interpolate_to_pressure
wrf_diagnostics = _wrf_diagnostics


__all__ = [
    "MSLP_SMOOTH_PASSES", "MSLP_SMOOTH_TERRAIN_CEILING_M",
    "MSLP_SMOOTH_TERRAIN_FLOOR_M", "ReflectivityMapSpec",
    "SYNOPTIC_MAP_SIZE", "SynopticMapSpec", "boundary_zone_blowup",
    "interior_region",
    "interpolate_to_pressure", "make_composite_reflectivity_map",
    "make_synoptic_maps", "pattern_correlation", "rmse", "score_pair",
    "terrain_smoothed_mslp", "wrf_diagnostics",
]
