"""``gpuwm render`` -- forecast product PNGs from wrfout NetCDF files.

Two engines render:

* ``rust`` -- the vendored Rusty Weather production renderer
  (``tools/rustwx``, :mod:`gpuwm.rustwx`): the campaign product sheets'
  quality tier, with basemaps and typography.  Its vendored catalog
  carries 324 entries, 151 of which the runtime lister evaluates as
  implicit-render candidates per file -- surface fields, the full
  200-850 mb isobaric chart families, CAPE/CIN/SRH/shear/STP severe
  suite, the heavy ECAPE family (``--heavy``), and multi-hour windowed
  accumulations -- and everything a file's stored fields prove out
  renders (``--list-products`` shows the per-file verdict and every
  reason).
  The default whenever the binary is built and probes as runnable.
* ``matplotlib`` -- the wrf-rust + matplotlib path below: the no-rust
  fallback, always available with the ``[render]`` extra.

``--pair A_DIR B_DIR`` composes two runs' rendered PNGs into labeled
side-by-side comparison sheets (:mod:`gpuwm.pair_compose`) -- either
engine's output pairs.

Every output filename carries a domain + resolution token
(:func:`domain_token`: ``d02-3km``, sub-kilometre nests as ``d05-111m``)
read from the file's own ``GRID_ID`` and ``DX``.  Without it two nests
of one run share every other filename component and the second render
at a lead silently overwrote the first -- a forecast lost with no error
and an exit code of 0.  The same spacing appears in the plot subtitle
as ``Δx 3 km``, and plots are labelled with the model that produced
them (``--source-label``, default ``ArWen``) rather than the GDEX fetch
source the rust engine's ``wrf`` store identity inherits.

Every derived quantity comes from the mandated ``wrf`` package (pip
distribution ``wrf-rust``): destaggering, earth-rotation, and unit
conversion are ``wrf.getvar`` calls, never local formulas.  When a file
carries no earth-rotation fields (idealized/minimal wrfouts without
SINALPHA/COSALPHA), the 10 m wind panel falls back to grid-relative raw
``U10``/``V10`` in the model's native m/s and labels both the barbs and
the colorbar accordingly -- an honest degradation, not a hand-rolled
conversion.  The one composition performed here is WRF's own
accumulation-bucket total (``RAINC + RAINNC``), which is bookkeeping
over Registry accumulators rather than a diagnostic.  Composite reflectivity is the column maximum
of the model-native ``REFL_10CM`` -- a direct field product, matching
``tools/matched_wrfout_refl_figures.py`` -- not a re-derived simulated
reflectivity.

Rendering conventions follow that tool: Agg backend, the standard NWS
5-dBZ reflectivity scale, recessive axes, per-panel colorbars labeled
with units, and valid-time titles.  Geographic extent is the file's own
XLAT/XLONG curvilinear coordinates (``wrf.latlon_coords``), so terrain
and projection shape the frame instead of bare array indices.

The module imports ``wrf`` and matplotlib lazily: ``gpuwm --help`` and
every non-render command work without the render extra installed, and
the failure names the missing dependency and the install command.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from gpuwm import explain

#: The pip distribution/version the render products are certified against.
WRF_PACKAGE_REQUIREMENT = "wrf-rust==0.2.35"

#: Standard NWS 5-dBZ-step reflectivity scale (5..75), as in
#: tools/matched_wrfout_refl_figures.py (domain convention).
_NWS_COLORS = (
    "#04e9e7", "#019ff4", "#0300f4", "#02fd02", "#01c501", "#008e00",
    "#fdf802", "#e5bc00", "#fd9500", "#fd0000", "#d40000", "#bc0000",
    "#f800fd", "#9854c6",
)
_NWS_LEVELS = tuple(np.arange(5.0, 76.0, 5.0))

#: Accumulated-precipitation bounds (mm) on an NWS-style QPF ladder.
_PRECIP_LEVELS = (0.1, 1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 25.0,
                  35.0, 50.0, 75.0, 100.0, 150.0)

_WRFOUT_DOMAIN = re.compile(r"wrfout_(d\d{2})")

#: Provenance label for locally-imported runs.  The rust engine inherits
#: a GDEX fetch source from its ``wrf`` store-model identity, which this
#: lane never fetched from; ``--source-label`` renames it for stock-WRF
#: files, which ArWen did not produce and must not claim.
DEFAULT_SOURCE_LABEL = "ArWen"


def _import_wrf():
    try:
        import wrf
    except ImportError as exc:
        raise RuntimeError(
            "gpuwm render needs the mandated 'wrf' package (pip "
            f"distribution {WRF_PACKAGE_REQUIREMENT}); install it with "
            "pip install 'gpuwm[render]' or pip install "
            f"'{WRF_PACKAGE_REQUIREMENT}'") from exc
    return wrf


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


#: The slug an output carries when the file's domain identity cannot be
#: established.  Spelled exactly as the rust engine spells it
#: (``rw-wrfbatch``'s ``native_grid``): a reader comparing two engines'
#: outputs should not have to learn two words for one fact.
NATIVE_GRID_SLUG = "native_grid"


def _domain_tag(path: Path) -> str | None:
    """``d0X`` when the file PROVES its domain, else ``None``.

    Deliberately the same rule ``rw-wrfbatch::file_grid_identity`` uses,
    in the same precedence: the ``GRID_ID`` global attribute when it is a
    plausible domain number, otherwise a ``wrfout_dNN`` filename,
    otherwise nothing.  Two engines that degrade differently are two
    behaviours to document and one of them will be wrong.

    ``None`` is the whole point of the change.  This used to return
    ``"d01"`` for a file carrying neither piece of evidence, which
    labelled an anonymous input as the parent domain on no evidence at
    all -- and, because the token is what separates one nest's output
    filenames from another's, let two anonymous inputs at one valid time
    overwrite each other in exactly the silence this release exists to
    end.
    """

    try:
        import netCDF4
        with netCDF4.Dataset(path) as ds:
            if "GRID_ID" in ds.ncattrs():
                grid_id = int(ds.getncattr("GRID_ID"))
                # The rust side accepts 1..=99; anything else is not a
                # domain number and is treated as no evidence.
                if 1 <= grid_id <= 99:
                    return f"d{grid_id:02d}"
    except Exception:
        pass
    match = _WRFOUT_DOMAIN.search(path.name)
    return match.group(1) if match else None


def _grid_spacing_m(path: Path) -> float | None:
    """The file's own ``DX`` in metres, or None when it declares none."""

    try:
        import netCDF4
        with netCDF4.Dataset(path) as ds:
            if "DX" not in ds.ncattrs():
                return None
            spacing = float(ds.getncattr("DX"))
    except Exception:
        return None
    if not np.isfinite(spacing) or spacing <= 0.0:
        return None
    return spacing


def _spacing_parts(spacing_m: float | None) -> tuple[str, str] | None:
    """``(value, unit)`` for a grid spacing: ``('3', 'km')``/``('333', 'm')``.

    At and above a kilometre the number is trimmed kilometres (a 3:1 nest
    of a 12 km parent is ``1.333 km``); below it, integer metres, because
    ``0.333 km`` is a worse label for a 333 m nest than ``333 m`` is.
    """

    if spacing_m is None:
        return None
    metres = round(float(spacing_m))
    if metres >= 1_000:
        value = f"{spacing_m / 1_000.0:.3f}".rstrip("0").rstrip(".")
        return (value or "0"), "km"
    return f"{metres:d}", "m"


def resolution_token(spacing_m: float | None) -> str | None:
    """``3km``/``333m`` -- the resolution half of the filename token."""

    parts = _spacing_parts(spacing_m)
    return None if parts is None else f"{parts[0]}{parts[1]}"


def spacing_label(spacing_m: float | None) -> str | None:
    """``Δx 3 km`` -- the same number, spelled for the plot title."""

    parts = _spacing_parts(spacing_m)
    return None if parts is None else f"Δx {parts[0]} {parts[1]}"


def domain_token(domain: str | None, spacing_m: float | None) -> str:
    """``d02-3km`` -- the filename token that separates a run's nests.

    Two nests of one run share every other filename component (product,
    valid time, and on the rust engine model/cycle/lead as well), so
    without this the second domain rendered at a lead silently
    overwrites the first.

    The three degradation steps are ``rw-wrfbatch::native_domain_slug``'s,
    exactly: identity plus a usable ``DX`` gives ``d02-3km``; identity
    without one gives a bare ``d02``, because the domain alone still
    separates the nests, which is the token's whole job; and no identity
    gives :data:`NATIVE_GRID_SLUG` with no resolution appended -- a
    resolution is not an identity, and pretending otherwise is how
    ``d01`` got printed on files that never claimed to be d01.
    """

    if domain is None:
        return NATIVE_GRID_SLUG
    resolution = resolution_token(spacing_m)
    return domain if resolution is None else f"{domain}-{resolution}"


def plot_context(domain: str | None, spacing_m: float | None, stamp: str,
                 source_label: str) -> str:
    """The second title line every matplotlib product carries.

    Mirrors the rust engine's subtitle: grid identity, the spacing the
    file declares, the valid time, and who produced the forecast.
    """

    segments = [domain_token(domain, spacing_m)]
    spacing = spacing_label(spacing_m)
    if spacing is not None:
        segments.append(spacing)
    segments.append(f"valid {stamp}")
    segments.append(source_label)
    return " | ".join(segments)


def _figure(plt, lat, lon):
    """One recessive-axes lat/lon panel sized to the domain aspect."""
    ny, nx = np.asarray(lat).shape
    width = 8.0
    height = max(4.0, width * ny / max(nx, 1) * 0.9)
    fig, axis = plt.subplots(figsize=(width, height))
    axis.set_xlabel("longitude (degrees east)", fontsize=9, color="#444444")
    axis.set_ylabel("latitude (degrees north)", fontsize=9, color="#444444")
    axis.tick_params(labelsize=8, colors="#444444")
    for spine in axis.spines.values():
        spine.set_color("#999999")
    return fig, axis


def _finish(fig, axis, mappable, *, title: str, cbar_label: str,
            out_png: Path, dpi: int, ticks=None) -> None:
    cbar = fig.colorbar(mappable, ax=axis, shrink=0.9, pad=0.02,
                        ticks=ticks)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    axis.set_title(title, fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)


def _render_refl(wf, timeidx, lat, lon, *, plt, wrf, context,
                 out_png: Path, dpi: int) -> None:
    from matplotlib.colors import BoundaryNorm, ListedColormap

    refl = np.asarray(wrf.getvar(wf, "REFL_10CM", timeidx=timeidx))
    composite = refl.max(axis=0)
    cmap = ListedColormap(_NWS_COLORS)
    cmap.set_under("none")
    cmap.set_over(_NWS_COLORS[-1])
    norm = BoundaryNorm(_NWS_LEVELS, cmap.N)
    fig, axis = _figure(plt, lat, lon)
    mesh = axis.pcolormesh(lon, lat, composite, cmap=cmap, norm=norm,
                           shading="auto")
    _finish(fig, axis, mesh,
            title=f"composite reflectivity (column-max REFL_10CM)\n"
                  f"{context}",
            cbar_label="composite dBZ", out_png=out_png, dpi=dpi,
            ticks=_NWS_LEVELS[::2])
    plt.close(fig)


def _render_t2(wf, timeidx, lat, lon, *, plt, wrf, context,
               out_png: Path, dpi: int) -> None:
    t2 = np.asarray(wrf.getvar(wf, "t2", timeidx=timeidx, units="degC"))
    fig, axis = _figure(plt, lat, lon)
    # Spectral temperature scale (meteorological convention), symmetric
    # ticks left to matplotlib; the data range sets the limits.
    mesh = axis.pcolormesh(lon, lat, t2, cmap="RdYlBu_r", shading="auto")
    _finish(fig, axis, mesh,
            title=f"2 m temperature\n{context}",
            cbar_label="deg C", out_png=out_png, dpi=dpi)
    plt.close(fig)


def _render_wind10(wf, timeidx, lat, lon, *, plt, wrf, context,
                   out_png: Path, dpi: int) -> None:
    try:
        # Earth-rotated components (needs SINALPHA/COSALPHA in the file);
        # knots come from wrf.getvar's own unit handling.
        uv = np.asarray(wrf.getvar(wf, "uvmet10", timeidx=timeidx,
                                   units="kt"))
        u_barb, v_barb = uv[0], uv[1]
        speed = np.asarray(wrf.getvar(wf, "wspd10", timeidx=timeidx,
                                      units="kt"))
        rotation, speed_units = "earth-rotated", "kt"
    except Exception:
        # Idealized/minimal files carry no rotation fields; the fallback
        # is grid-relative raw U10/V10 in the model's native m/s, labeled
        # as such -- unit conversion is wrf.getvar's job, never a local
        # formula, so the whole panel degrades to m/s honestly rather
        # than converting by hand.
        u_barb = np.asarray(wrf.getvar(wf, "U10", timeidx=timeidx))
        v_barb = np.asarray(wrf.getvar(wf, "V10", timeidx=timeidx))
        speed = np.asarray(wrf.getvar(wf, "wspd10", timeidx=timeidx))
        rotation, speed_units = "grid-relative", "m s-1"
    fig, axis = _figure(plt, lat, lon)
    mesh = axis.pcolormesh(lon, lat, speed, cmap="viridis", shading="auto")
    ny, nx = speed.shape
    # ~18 barbs along the long axis keeps flags legible at any nest size.
    step = max(1, int(np.ceil(max(ny, nx) / 18)))
    sub = (slice(step // 2, None, step), slice(step // 2, None, step))
    axis.barbs(np.asarray(lon)[sub], np.asarray(lat)[sub],
               u_barb[sub], v_barb[sub], length=5.5,
               linewidth=0.6, color="#222222")
    _finish(fig, axis, mesh,
            title=f"10 m wind ({rotation} barbs)\n{context}",
            cbar_label=f"10 m wind speed ({speed_units})",
            out_png=out_png, dpi=dpi)
    plt.close(fig)


def _render_precip(wf, timeidx, lat, lon, *, plt, wrf, context,
                   out_png: Path, dpi: int) -> None:
    from matplotlib.colors import BoundaryNorm

    buckets = []
    for name in ("RAINC", "RAINNC"):
        try:
            buckets.append(
                np.asarray(wrf.getvar(wf, name, timeidx=timeidx)))
        except Exception:
            continue
    if not buckets:
        raise RuntimeError(
            "neither RAINC nor RAINNC is present; there is no "
            "accumulated-precipitation field to render")
    total = buckets[0]
    for bucket in buckets[1:]:
        total = total + bucket
    import matplotlib

    fig, axis = _figure(plt, lat, lon)
    cmap = matplotlib.colormaps["YlGnBu"].resampled(
        len(_PRECIP_LEVELS) - 1)
    cmap.set_under("none")
    norm = BoundaryNorm(_PRECIP_LEVELS, cmap.N)
    mesh = axis.pcolormesh(lon, lat, total, cmap=cmap, norm=norm,
                           shading="auto")
    _finish(fig, axis, mesh,
            title=f"accumulated precipitation (RAINC + RAINNC)\n"
                  f"{context}",
            cbar_label="mm since simulation start", out_png=out_png,
            dpi=dpi, ticks=_PRECIP_LEVELS)
    plt.close(fig)


def _render_olr(wf, timeidx, lat, lon, *, plt, wrf, context,
                out_png: Path, dpi: int) -> None:
    """Top-of-atmosphere outgoing longwave, drawn as synthetic satellite IR.

    ``OLR`` is what the longwave scheme was already computing and wrfout
    frames have carried since 1.5.1, so this is the one product here whose
    subject is the model as a satellite would see it.

    Inverted grayscale is what makes it read that way: an IR image is dark
    where the emitting surface is warm and bright where it is cold, and OLR
    runs the same direction -- deep cold cloud tops emit least, the clear
    warm surface emits most -- so ``gray_r`` puts storm tops white against a
    dark background exactly as the imagery convention does.

    The panel stays in W m-2 rather than converting to a brightness
    temperature.  That conversion is a derived quantity and derived
    quantities are the ``wrf`` package's job, never a formula written at a
    plotting site (the same rule that keeps ``_render_wind10`` degrading to
    m s-1 instead of converting knots by hand).

    Limits come from the frame, as they do for ``t2`` and ``wind10``: there
    is no standard OLR ladder to hard-code, and inventing one here would put
    unsourced numbers between the reader and the field.  So shade is
    comparable within a panel, not between panels -- read the colorbar.
    """

    olr = np.asarray(wrf.getvar(wf, "OLR", timeidx=timeidx))
    fig, axis = _figure(plt, lat, lon)
    mesh = axis.pcolormesh(lon, lat, olr, cmap="gray_r", shading="auto")
    _finish(fig, axis, mesh,
            title=f"outgoing longwave radiation at TOA "
                  f"(synthetic IR)\n{context}",
            cbar_label="OLR (W m-2)", out_png=out_png, dpi=dpi)
    plt.close(fig)


#: Product registry: CLI name -> renderer.  ``all`` expands to this order.
PRODUCTS = {
    "refl": _render_refl,
    "t2": _render_t2,
    "wind10": _render_wind10,
    "precip": _render_precip,
    "olr": _render_olr,
}

#: The shared product names, as the rust engine's catalog slugs.
#:
#: This maps four of the five in :data:`PRODUCTS`.  ``olr`` is absent
#: because the rust catalog has no OLR chart yet -- that half is upstream
#: work in the renderer's own catalog, and the entry lands here when the
#: slug it would name exists.  Pointing an alias at a plausible-looking
#: slug before then is precisely the ``wind10`` bug described below: a
#: mapping that reads correct and returns the wrong chart.  Until the
#: catalog carries one, ``--products olr --engine rust`` passes ``olr``
#: through as a raw slug and the renderer refuses it by name, which is a
#: legible answer; ``--products all --engine rust`` is unaffected, since
#: ``all`` is the renderer's own catalog and never this table.
#:
#: Each value is the slug of a chart whose SUBJECT is what the key
#: names.  ``wind10`` used to map to ``mslp_10m_winds`` under a comment
#: calling that "the catalog's standalone surface-wind product"; it is
#: not.  ``mslp_10m_winds`` is a mean-sea-level PRESSURE analysis with
#: 10 m barbs drawn over it, so ``--products wind10 --engine rust``
#: returned a pressure chart to anyone who asked for wind, while the
#: matplotlib engine's own ``wind10`` drew a genuine wind map.  During a
#: wildfire that asymmetry is what pushed an operator onto the fallback
#: engine to get a wind map at all.  The rust catalog now carries a real
#: standalone wind chart and this points at it.
#:
#: Any other token is passed through as a raw catalog slug, which the
#: renderer validates strictly (see :func:`parse_products_rust`).
RUST_PRODUCT_ALIASES = {
    # column-max simulated reflectivity
    "refl": "composite_reflectivity",
    # 2 m temperature fill
    "t2": "2m_temperature",
    # 10 m wind speed fill + 10 m barbs, one frame, nothing else on it
    "wind10": "10m_wind_speed_and_direction",
    # run-total accumulated precipitation
    "precip": "total_qpf",
}

#: What each alias must actually DRAW, as the canonical store selector
#: its fill resolves to.  This is the assertion a string-equality test
#: could not make: the ``wind10`` bug was a correct-looking mapping to a
#: real slug that filled with the wrong quantity, and any test comparing
#: ``RUST_PRODUCT_ALIASES["wind10"]`` to a hardcoded string would have
#: agreed with the broken code.  ``tests/test_render_rust.py`` checks
#: these against the renderer's own catalog, from the built binary.
RUST_PRODUCT_ALIAS_FILL_SELECTORS = {
    "refl": "composite_reflectivity_entire_atmosphere",
    "t2": "temperature_2m_agl",
    "wind10": "wind_speed_10m_agl",
    "precip": "total_precipitation_surface",
}


def parse_products_rust(spec: str) -> str:
    """CLI product spec -> the rust renderer's ``--products`` value.

    ``all`` expands to the renderer's own inspected catalog (its "all"),
    the four shared names map through :data:`RUST_PRODUCT_ALIASES`, and
    unknown tokens pass through as catalog slugs for the renderer's own
    strict validation -- so ``--products sbcape,srh_0_1km`` works without
    this module re-declaring the rust catalog.
    """

    slugs: list[str] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token == "all":
            return "all"
        slug = RUST_PRODUCT_ALIASES.get(token, token)
        if slug not in slugs:
            slugs.append(slug)
    if not slugs:
        raise ValueError("no products requested")
    return ",".join(slugs)


def parse_size(spec: str) -> tuple[int, int]:
    """``1200x900`` -> (width, height); rust-engine output pixels."""

    parts = spec.lower().split("x")
    if len(parts) != 2:
        raise ValueError(
            f"--size must be WIDTHxHEIGHT (e.g. 1200x900), got {spec!r}")
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(
            f"--size must be WIDTHxHEIGHT (e.g. 1200x900), got {spec!r}"
        ) from None
    if width < 320 or height < 240:
        raise ValueError("--size must be at least 320x240")
    return width, height


def parse_products(spec: str) -> tuple[str, ...]:
    """``refl,t2`` -> product tuple; ``all`` -> every product, in order."""
    names: list[str] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token == "all":
            names.extend(name for name in PRODUCTS if name not in names)
            continue
        if token not in PRODUCTS:
            raise ValueError(
                f"unknown product {token!r}; choose from "
                f"{', '.join(PRODUCTS)} or 'all'")
        if token not in names:
            names.append(token)
    if not names:
        raise ValueError("no products requested")
    return tuple(names)


def parse_timeidx(spec: str) -> int | None:
    """``N`` -> that frame, ``all`` -> None (every frame in the file)."""
    if spec == "all":
        return None
    try:
        index = int(spec)
    except ValueError:
        raise ValueError(
            f"--timeidx must be an integer or 'all', got {spec!r}"
        ) from None
    if index < 0:
        raise ValueError("--timeidx must be non-negative")
    return index


def _stamp_for_filename(stamp: str) -> str:
    """Valid-time stamp with Windows-illegal colons replaced."""
    return stamp.replace(":", "-")


def render_wrfouts(paths, *, products: tuple[str, ...],
                   timeidx: int | None, outdir: Path,
                   dpi: int = 150,
                   source_label: str = DEFAULT_SOURCE_LABEL,
                   ) -> tuple[list[Path], list[str],
                              list[tuple[str, str]]]:
    """Render every requested product/frame.

    Returns ``(written, failures, skipped)``.  The third list is the one
    this function used to fold into the second, and folding them was a
    wrong answer with an exit code attached: a product whose *declared*
    inputs (:data:`_MPL_PRODUCT_NEEDS`) are simply not in the file has
    not failed at anything.  The cold-start history frame is the standing
    example -- no microphysics call precedes it, so it carries no
    ``REFL_10CM``, a registered deviation -- and reflectivity coming back
    empty from it used to fail the render, and with it the whole ``gpuwm
    go`` chain, on a forecast that had completed.

    The distinction is drawn from the table, not from the exception, so
    it holds for every product and stays true when the catalog grows:

    * declared inputs absent  -> **skipped**, with the names of what is
      absent; the caller reports it and the exit code does not move.
    * declared inputs present -> rendered, or **failed** if it raised.
      That is a real failure and still is one.

    Nothing is silent either way: both lists come back for the CLI to
    print.
    """
    wrf = _import_wrf()
    plt = _pyplot()
    written: list[Path] = []
    failures: list[str] = []
    skipped: list[tuple[str, str]] = []
    #: output path -> the input that claimed it, this invocation.
    claims: dict[Path, Path] = {}
    for path in (Path(p) for p in paths):
        try:
            wrffile = wrf.WrfFile(str(path))
            stamps = list(wrffile.times())
        except Exception as exc:
            failures.append(f"{path}: unreadable wrfout ({exc})")
            continue
        present = _file_variables(path)
        domain = _domain_tag(path)
        spacing_m = _grid_spacing_m(path)
        token = domain_token(domain, spacing_m)
        if timeidx is None:
            indices = range(len(stamps))
        elif timeidx >= len(stamps):
            failures.append(
                f"{path}: --timeidx {timeidx} out of range; file has "
                f"{len(stamps)} frame(s)")
            continue
        else:
            indices = (timeidx,)
        for index in indices:
            stamp = stamps[index]
            try:
                lat, lon = wrf.latlon_coords(wrffile, timeidx=index)
            except Exception as exc:
                failures.append(
                    f"{path}[{index}]: no XLAT/XLONG coordinates ({exc})")
                break
            # Antimeridian-crossing domains: XLONG jumps between +180
            # and -180 mid-array, which pcolormesh renders as a smear
            # across the full axis.  Unwrap onto the branch nearest the
            # domain centre (the axis may then extend past +/-180,
            # which matplotlib labels correctly).
            lon_values = np.asarray(lon)
            if lon_values.size and (lon_values.max()
                                    - lon_values.min()) > 180.0:
                center = float(lon_values[tuple(
                    s // 2 for s in lon_values.shape)])
                lon = center + ((lon - center + 180.0) % 360.0 - 180.0)
            context = plot_context(domain, spacing_m, stamp, source_label)
            for product in products:
                absent = ([] if present is None
                          else missing_declared_inputs(present, product))
                if absent:
                    skipped.append((product, f"{path}[{index}] carries no "
                                             f"{', '.join(absent)}"))
                    continue
                out_png = outdir / (
                    f"{product}_{token}_{_stamp_for_filename(stamp)}.png")
                # The token separates nests, but it cannot separate two
                # inputs it could not identify: both are `native_grid`.
                # Refusing the second is the same call this release made
                # for named nests -- an overwrite that reports success
                # is the failure, not the collision.
                claimed = claims.get(out_png)
                if claimed is not None and claimed != path:
                    failures.append(
                        f"{path}[{index}] {product}: would overwrite "
                        f"{out_png.name}, already rendered from "
                        f"{claimed}.  Both inputs resolve to the same "
                        f"output name because neither declares a "
                        f"distinguishable domain identity (GRID_ID or a "
                        f"wrfout_dNN filename).  Render them into "
                        f"separate --out directories, or give the files "
                        f"their GRID_ID.")
                    continue
                claims[out_png] = path
                try:
                    PRODUCTS[product](
                        wrffile, index, lat, lon, plt=plt, wrf=wrf,
                        context=context, out_png=out_png, dpi=dpi)
                except Exception as exc:
                    failures.append(f"{path}[{index}] {product}: {exc}")
                    continue
                written.append(out_png)
                print(f"render: {out_png}")
    return written, failures, skipped


#: The matplotlib engine's per-product wrfout variable needs.  Two
#: readers now: the --list-products availability report, and the render
#: loop's own pre-check (:func:`missing_declared_inputs`), so the status
#: a listing prints and the decision a render makes come from one table
#: rather than from a table and an exception.
_MPL_PRODUCT_NEEDS = {
    "refl": ("REFL_10CM",),
    "t2": ("T2",),
    "wind10": ("U10", "V10"),
    "precip": ("RAINC|RAINNC",),
    # OLR is present exactly when the attached longwave scheme produced
    # it, so a radiation-off run lists this product as missing-fields
    # rather than failing -- which is the honest report of that run.
    "olr": ("OLR",),
}


def missing_declared_inputs(present, product: str) -> list[str]:
    """This product's declared inputs that ``present`` does not carry.

    ``present`` is the file's variable-name set.  Each entry of
    :data:`_MPL_PRODUCT_NEEDS` is one requirement, and ``A|B`` means
    either satisfies it, so what comes back reads as the requirement did.

    Empty means every declared input is there -- which is the whole
    point of asking: a product that then fails to render failed for a
    reason this table does not know about, and that is a real failure.
    """

    missing = []
    for need in _MPL_PRODUCT_NEEDS.get(product, ()):
        options = need.split("|")
        if not any(option in present for option in options):
            missing.append(" or ".join(options))
    return missing


def _file_variables(path: Path) -> set[str] | None:
    """The wrfout's variable names, or None when they cannot be read.

    None is not a failure here: it means the pre-check abstains and
    rendering decides, which is exactly the behaviour that predates the
    pre-check.  The read is names only -- no field values -- so it is
    not a derived quantity and does not belong to the mandated ``wrf``
    package.
    """

    try:
        import netCDF4

        with netCDF4.Dataset(path) as dataset:
            return set(dataset.variables)
    except Exception:
        return None


def _list_products_matplotlib(path: Path) -> list[tuple[str, str, str, str]]:
    """(product, kind, status, detail) rows for the matplotlib catalog."""

    import netCDF4

    with netCDF4.Dataset(path) as ds:
        present = set(ds.variables)
    rows = []
    for product in _MPL_PRODUCT_NEEDS:
        missing = missing_declared_inputs(present, product)
        if missing:
            rows.append((product, "matplotlib", "missing-fields",
                         f"not in file: {', '.join(missing)}"))
        else:
            rows.append((product, "matplotlib", "renderable",
                         "required variables present"))
    return rows


def list_products_main(args: argparse.Namespace, engine: str) -> int:
    """``gpuwm render --list-products WRFOUT...``: catalog + availability."""

    failures = 0
    for path in args.wrfout:
        print(f"render: product catalog for {path} (engine {engine})")
        try:
            if engine == "rust":
                from gpuwm import rustwx

                renderer = rustwx.find_renderer()
                import tempfile
                with tempfile.TemporaryDirectory(
                        prefix="gpuwm-rwlist-") as store:
                    rows, summary = rustwx.list_products(
                        renderer, path, store_root=Path(store),
                        heavy=args.heavy)
            else:
                rows = _list_products_matplotlib(path)
                counts: dict[str, int] = {}
                for _, _, status, _ in rows:
                    counts[status] = counts.get(status, 0) + 1
                summary = f"total={len(rows)} " + " ".join(
                    f"{status}={count}"
                    for status, count in sorted(counts.items()))
        except (RuntimeError, OSError) as exc:
            print(f"render FAIL: {exc}", file=sys.stderr)
            failures += 1
            continue
        for slug, kind, status, detail in rows:
            print(f"  {status:<14} {kind:<9} {slug:<40} {detail}")
        print(f"render: {path}: {summary}")
    return 1 if failures else 0


def catalog_main(args) -> int:
    """``gpuwm render --list-products`` with no file: the vocabulary.

    Two different questions share the flag.  With a wrfout it means
    "which of these can THIS file render, and why not the rest" --
    availability, which needs the file.  Without one it means "what may
    I put in ``--products``", which needs only the build, and which is
    the question someone has before they have run anything at all.

    Answered by asking the engine, never from a copy kept here: the rust
    catalog is the renderer's, and a second list in this module is the
    enumeration-drift failure this project keeps paying for.  With no
    rust engine installed, the matplotlib engine's own products are the
    honest answer, and the notice above already said which engine spoke.
    """

    from gpuwm import rustwx

    engine, why = _resolve_engine(args.engine)
    notice = fallback_notice(engine, why)
    if notice is not None:
        print(notice, file=sys.stderr)
    print(f"render: product catalog (engine {engine})")
    if engine == "matplotlib":
        for product in PRODUCTS:
            print(f"  {product}")
        print(f"render: {len(PRODUCTS)} products")
        return 0
    renderer = rustwx.find_renderer()
    try:
        result = subprocess.run(
            [str(renderer), "--list-products"], capture_output=True,
            text=True, errors="replace", env=rustwx.renderer_env())
    except OSError as exc:
        print(f"render: renderer failed to launch: {exc}", file=sys.stderr)
        return 2
    if result.returncode != 0:
        tail = [line for line in (result.stderr or "").splitlines()
                if line.strip()]
        print(f"render: {tail[-1] if tail else f'exit {result.returncode}'}",
              file=sys.stderr)
        return 2
    print(result.stdout, end="")
    print("render: `gpuwm render --list-products WRFOUT` adds which of "
          "these that file can actually render, and why not the rest")
    return 0


def render_wrfouts_rust(paths, *, products: str, timeidx: int | None,
                        outdir: Path, size: tuple[int, int],
                        heavy: bool = False,
                        source_label: str = DEFAULT_SOURCE_LABEL,
                        ) -> tuple[list[Path], list[str],
                                   list[tuple[str, str]]]:
    """Rusty Weather engine; ``(written, failures, skipped)``.

    One renderer invocation per wrfout file, each with its own scratch
    store (two files in one store would merge into one run's timeline;
    per-file isolation keeps every input independently comparable, the
    campaign convention).  The scratch store lives under the output
    directory and is removed as soon as the file's render finishes.
    """

    from gpuwm import rustwx

    renderer = rustwx.find_renderer()
    if renderer is None:
        raise RuntimeError(
            "the rust render engine is not built; "
            + rustwx.renderer_remedy())
    outdir.mkdir(parents=True, exist_ok=True)
    frames = "all" if timeidx is None else str(timeidx)
    width, height = size
    written: list[Path] = []
    failures: list[str] = []
    skipped: list[tuple[str, str]] = []
    for path in (Path(p) for p in paths):
        import tempfile
        store = Path(tempfile.mkdtemp(prefix=".rwstore-", dir=outdir))
        try:
            file_written, file_failures, file_skipped = rustwx.run_renderer(
                renderer, path, store_root=store, out_dir=outdir,
                products=products, frames=frames, width=width,
                height=height, heavy=heavy, source_label=source_label)
        finally:
            _remove_scratch_store(store)
        file_written = [_rebrand_engine_output(png) for png in file_written]
        written.extend(file_written)
        failures.extend(file_failures)
        skipped.extend(file_skipped)
        for png in file_written:
            print(f"render: {png}")
    return written, failures, skipped


#: What the shipped product calls its rendered files.
OUTPUT_FILENAME_PREFIX = "arwen_"

#: What the vendored engine calls them: rustwx-products hardcodes
#: ``rustwx_`` into every output filename format string (e.g.
#: ``tools/rustwx/crates/rustwx-products/src/derived.rs``).
_ENGINE_FILENAME_PREFIX = "rustwx_"


def _rebrand_engine_output(png: Path) -> Path:
    """The product's filename for one engine-written PNG.

    The vendored engine stamps its own name into every file it writes;
    the product these files come out of is ArWen, and a PNG named for
    the vendored crate is a first-run head-scratcher (the field run
    asked what "rustwx" was).  The rename happens here -- the one seam
    every wheel-shipped render flows through -- rather than in the
    vendored Rust, which keeps the engine byte-identical to its
    campaign builds.  Filenames only: module names, store identities
    and the pair-compose lead marker (which reads both spellings) are
    untouched.

    A name the engine did not brand passes through unchanged, and so
    does one whose rename fails -- a drawn image under yesterday's name
    beats a failure over branding.
    """

    if not png.name.startswith(_ENGINE_FILENAME_PREFIX):
        return png
    target = png.with_name(
        OUTPUT_FILENAME_PREFIX + png.name[len(_ENGINE_FILENAME_PREFIX):])
    try:
        os.replace(png, target)
    except OSError:
        return png
    return target


def _remove_scratch_store(store: Path) -> None:
    """Delete a per-file scratch store, riding out Windows handle lag.

    The renderer memory-maps its hour files; on Windows the mapping's
    release can trail the process exit by a beat, making an immediate
    rmtree fail with 'directory not empty'.  A few short retries clear
    it; a scratch directory that STILL cannot be removed is worth a
    warning, never a failed render.
    """

    import shutil
    import time

    for delay in (0.0, 0.25, 0.5, 1.0, 2.0, 2.0, 2.0, 2.0):
        if delay:
            time.sleep(delay)
        try:
            shutil.rmtree(store)
        except OSError:
            continue
        return
    shutil.rmtree(store, ignore_errors=True)
    if store.exists():
        print(f"render: warning: scratch store left behind: {store}",
              file=sys.stderr)


def _resolve_engine(requested: str) -> tuple[str, str]:
    """(engine, why) for ``--engine auto|rust|matplotlib``.

    ``auto`` selects rust exactly when the built binary resolves AND
    probe-executes (the doctor-verified condition); anything less falls
    back to matplotlib with the reason in ``why``.
    """

    if requested == "matplotlib":
        return "matplotlib", "requested"
    from gpuwm import rustwx

    renderer = rustwx.find_renderer()
    if requested == "rust":
        if renderer is None:
            raise RuntimeError(
                "--engine rust: the renderer is not built; "
                + rustwx.renderer_remedy())
        return "rust", str(renderer)
    if renderer is None:
        return "matplotlib", "rust renderer not built (gpuwm doctor shows " \
                             "the build one-liner)"
    ok, evidence = rustwx.probe_renderer(renderer)
    if ok:
        return "rust", str(renderer)
    return "matplotlib", f"rust renderer unusable: {evidence}"


#: How many products each engine can draw, for the fallback notice.
#: The rust catalog's 151 is its implicit-render candidate count on a
#: typical wrfout; the matplotlib count is read from :data:`PRODUCTS`,
#: so adding a fallback product cannot leave this notice overstating
#: what the fallback costs.
_RUST_CATALOG_PRODUCTS = 151


def fallback_notice(engine: str, why: str) -> str | None:
    """ONE line saying the matplotlib fallback is in use, and why.

    Silence here cost a pilot a whole session: four products came out,
    they looked right, and nothing said the 151-product rust catalog was
    simply not installed.  A fallback that does not announce itself is
    indistinguishable from the real thing until someone counts.

    One line, and it carries everything needed to act on it: the engine
    actually used, the products available against the rust catalog, and
    a fix that is true on THIS install.  The multi-line remedy belongs
    to ``gpuwm doctor``; a notice that scrolls is a notice that gets
    skimmed.

    That last clause used to be the bare ``cd tools/rustwx; cargo build``
    one-liner, naming a directory a wheel install does not have.  The
    1.0.1 remedy contract fixed that everywhere except here, because
    this call site assembled its own string.  It goes through the same
    install-aware machinery now, in the one-line form: the one-liner
    where the crate exists, a pointer to ``gpuwm doctor`` where it does
    not -- because the honest answer there is a whole bootstrap and this
    notice gets exactly one physical line.
    """

    if engine != "matplotlib" or why == "requested":
        return None
    from gpuwm import bridges, rustwx

    fix = bridges.install_aware_one_line_hint(
        rustwx.CARGO_BUILD_HINT, rustwx.RUSTWX_CRATE_RELATIVE)
    return (f"render: engine matplotlib -- rust render engine not "
            f"available ({why}); {len(PRODUCTS)} of the rust catalog's "
            f"{_RUST_CATALOG_PRODUCTS} products are renderable. Build it "
            f"with: {fix}")


def skip_notice(skipped: list[tuple[str, str]]) -> str | None:
    """ONE line for the products this render did not draw, or ``None``.

    A skip is not a failure and must not print as one -- but it must
    print, and it must print the PRODUCT NAMES.  A bare count is the
    shape of message a reader cannot act on: it says something is
    missing from the directory without saying what, so the only way to
    find out is to diff a listing against the catalog.

    Names in the action half, then; per-item evidence -- which file,
    which frame, which field -- behind ``--explain``, verbatim from the
    engine, because paraphrasing "which field was missing" would leave
    the reader guessing which product lost which input.

    The ``note:`` prefix is load-bearing rather than decorative.  It is
    this tree's word for "true and worth knowing, not a fault", and
    ``gpuwm go``'s stage runner surfaces a passing stage's ``warning:``
    and ``note:`` lines through its output capture -- so without it this
    sentence is printed by ``gpuwm render`` and swallowed by the chain,
    which is the silent success this change exists to avoid.

    The action half also states that the exit code is not about this.  A
    reader who has just watched a chain stop at render needs to know
    which of the two things they are looking at, and a count with no
    verdict attached reads as the bad one.
    """

    if not skipped:
        return None
    names = sorted({product for product, _detail in skipped})
    return explain.layered(
        f"note: render skipped {len(skipped)} product render(s) "
        f"({', '.join(names)}) -- the file(s) do not carry their declared "
        f"input fields.  That is not a failure and does not change the "
        f"exit code",
        "\n".join(f"  skipped {product}: {detail}"
                  for product, detail in skipped))


def missing_basemap_notice(renderer) -> str | None:
    """ONE line when the renderer resolves but its map assets do not.

    The other way to ship a plot believing it is something it is not.
    A renderer with no basemaps does not fail, warn, or exit non-zero:
    it draws the weather over a blank rectangle, and 1.4.0 shipped a
    tropical cyclone with no coastline that way (F4/B-11).  Nothing in
    the transcript said so, because as far as the renderer is concerned
    a map with no geography is a map.

    Delivery is fixed -- the bundle carries the shapefiles beside the
    binary now -- but an install that staged its binaries under 1.4.0
    and has not re-run ``gpuwm fetch-bridges`` still has the eight
    executables and no assets, which is exactly the silent state.  So
    this is checked at render time against the renderer's own
    resolution ladder, and it names the one command that fixes it.
    """

    from gpuwm import bridges, rustwx

    if renderer is None or rustwx.resolve_basemap_dir(renderer) is not None:
        return None
    if rustwx.cartopy_natural_earth_root() is not None:
        # The renderer's own last-resort fallback, which it consults per
        # layer after the candidate roots.  Geography will be drawn, so
        # warning here would be a false alarm on every workstation that
        # has ever run cartopy -- and a notice that cries wolf on
        # developer machines is a notice nobody reads on a real one.
        return None
    fix = ("run `gpuwm fetch-bridges`, which stages them beside the "
           "renderer" if bridges.prebuilt_bundle_offer() is not None
           else "set RUSTWX_BASEMAP_DIR to a checkout's "
                "tools/rustwx/assets/basemap")
    return ("render: WARNING: no basemap assets resolve for this renderer, "
            "so plots will be drawn with no coastlines, borders or state "
            f"lines -- {fix}")


def _pair_main(args: argparse.Namespace) -> int:
    try:
        from gpuwm.pair_compose import compose_pairs
    except ImportError:
        print("render: --pair needs Pillow (installed with the render "
              "extra: pip install 'gpuwm[render]')", file=sys.stderr)
        return 2
    left, right = (Path(p) for p in args.pair)
    labels = args.pair_labels or (None, None)
    try:
        sheets = compose_pairs(
            left, right, args.out, title=args.pair_title,
            subtitle=args.pair_subtitle, left_label=labels[0],
            right_label=labels[1])
    except ValueError as exc:
        print(f"render: {exc}", file=sys.stderr)
        return 2
    for sheet in sheets:
        print(f"render: {sheet}")
    print(f"render: {len(sheets)} pair sheet(s) -> {args.out}")
    return 0


def render_main(args: argparse.Namespace) -> int:
    if args.pair:
        if args.wrfout:
            print("render: --pair composes already-rendered PNG "
                  "directories; wrfout arguments do not combine with it",
                  file=sys.stderr)
            return 2
        return _pair_main(args)
    if not args.wrfout and args.list_products:
        # "What may I put in --products?" is a question about this build,
        # not about a file, and a forecaster who has not run anything yet
        # is exactly who asks it.  Demanding a wrfout first is why the
        # only way to discover a product slug was to read the source.
        # The renderer already answers it (`rw_wrfbatch --list-products`
        # with no inputs prints its vocabulary), so this asks rather than
        # keeping a second copy of the catalog here to go stale.
        return catalog_main(args)
    if not args.wrfout:
        print("render: at least one WRFOUT file is required "
              "(or --list-products alone for the product catalog, or "
              "--pair A_DIR B_DIR)", file=sys.stderr)
        return 2
    try:
        timeidx = parse_timeidx(args.timeidx)
        size = parse_size(args.size)
        engine, why = _resolve_engine(args.engine)
        if engine == "rust":
            rust_products = parse_products_rust(args.products)
        else:
            products = parse_products(args.products)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"render: {exc}", file=sys.stderr)
        return 2
    notice = fallback_notice(engine, why)
    if notice is not None:
        print(notice, file=sys.stderr)
    if engine == "rust":
        from gpuwm import rustwx

        blank = missing_basemap_notice(rustwx.find_renderer())
        if blank is not None:
            print(blank, file=sys.stderr)
    if args.list_products:
        return list_products_main(args, engine)
    print(f"render: engine {engine} ({why})")
    if engine == "rust":
        try:
            written, failures, skipped = render_wrfouts_rust(
                args.wrfout, products=rust_products, timeidx=timeidx,
                outdir=args.out, size=size, heavy=args.heavy,
                source_label=args.source_label)
        except RuntimeError as exc:
            print(f"render: {exc}", file=sys.stderr)
            return 2
    else:
        written, failures, skipped = render_wrfouts(
            args.wrfout, products=products, timeidx=timeidx,
            outdir=args.out, dpi=args.dpi,
            source_label=args.source_label)
    for failure in failures:
        print(f"render FAIL: {failure}", file=sys.stderr)
    notice = skip_notice(skipped)
    if notice is not None:
        print(explain.render(notice, explain=explain.explain_enabled(args),
                             command="gpuwm render"), file=sys.stderr)
    print(f"render: {len(written)} file(s) -> {args.out}")
    # Written-and-no-failures, where a SKIP IS NOT A FAILURE.  The two
    # were one list until 1.4.1 and the exit code could not tell them
    # apart, so a registered absence -- the cold-start frame's missing
    # REFL_10CM -- returned 1 from a render that had drawn every product
    # the file could support, and `gpuwm go` stopped the chain on a
    # forecast that had succeeded.
    #
    # The two arms that still return 1 are the ones that mean something
    # went wrong: any failure at all, and a render that produced no
    # image.  The second is what keeps a skip from becoming a silent
    # success -- ask for one product, have its inputs be absent, and the
    # answer is still a nonzero exit, because nothing was drawn.
    return 0 if written and not failures else 1


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "render",
        help="render forecast product PNGs from wrfout files via the "
             "wrf package (composite reflectivity, 2 m temperature, "
             "10 m wind, accumulated precipitation, TOA outgoing "
             "longwave as synthetic IR)")
    parser.add_argument(
        "wrfout", type=Path, nargs="*", metavar="WRFOUT",
        help="wrfout NetCDF file(s) written by gpuwm run")
    parser.add_argument(
        "--engine", choices=("auto", "rust", "matplotlib"),
        default="auto",
        help="render engine: the vendored Rusty Weather renderer "
             "(campaign plot quality; 151 implicit-render catalog "
             "candidates per file) or the matplotlib fallback; 'auto' "
             "(default) uses rust whenever its binary is built and "
             "probes as runnable")
    parser.add_argument(
        "--products", default="all", metavar="LIST",
        help="comma-separated products: "
             f"{', '.join(PRODUCTS)}, or 'all' (default); with the rust "
             "engine, raw catalog slugs (sbcape, srh_0_1km, ...) also "
             "work and 'all' renders its full catalog")
    parser.add_argument(
        "--timeidx", default="all", metavar="N|all",
        help="frame index within each file, or 'all' (default)")
    parser.add_argument(
        "--out", type=Path, default=Path("out/render"), metavar="DIR",
        help="output directory for the PNGs (default out/render)")
    parser.add_argument(
        "--dpi", type=int, default=150, metavar="N",
        help="PNG resolution, matplotlib engine (default 150)")
    parser.add_argument(
        "--size", default="1200x900", metavar="WxH",
        help="output pixels, rust engine (default 1200x900)")
    parser.add_argument(
        "--source-label", default=DEFAULT_SOURCE_LABEL, metavar="TEXT",
        help="model/provenance label stamped on every plot (default "
             f"{DEFAULT_SOURCE_LABEL}); set it when rendering wrfout "
             "files this model did not produce, so the sheet does not "
             "claim them")
    parser.add_argument(
        "--heavy", action="store_true",
        help="rust engine: also compute the heavy ECAPE product family "
             "at import (SBECAPE/SBNCAPE/SBECIN, ECAPE SCP/EHI/...; "
             "adds substantial per-frame import time)")
    parser.add_argument(
        "--list-products", action="store_true",
        help="list the engine's product catalog with per-file "
             "availability (why each product is or is not renderable "
             "from this wrfout) instead of rendering")
    parser.add_argument(
        "--pair", nargs=2, metavar=("A_DIR", "B_DIR"), type=Path,
        help="compose two runs' rendered PNG directories into labeled "
             "side-by-side comparison sheets (no wrfout arguments)")
    parser.add_argument(
        "--pair-title", default="Paired comparison", metavar="TITLE",
        help="pair-sheet title (default 'Paired comparison')")
    parser.add_argument(
        "--pair-subtitle", default="", metavar="TEXT",
        help="optional pair-sheet subtitle")
    parser.add_argument(
        "--pair-labels", nargs=2, metavar=("LEFT", "RIGHT"),
        help="panel labels (default: the two directory names)")
    parser.set_defaults(func=render_main)
    return parser


__all__ = ["DEFAULT_SOURCE_LABEL", "PRODUCTS", "RUST_PRODUCT_ALIASES",
           "WRF_PACKAGE_REQUIREMENT", "domain_token", "list_products_main",
           "parse_products", "parse_products_rust", "parse_size",
           "parse_timeidx", "plot_context", "register_cli", "render_main",
           "missing_basemap_notice", "missing_declared_inputs",
           "render_wrfouts", "render_wrfouts_rust", "resolution_token",
           "skip_notice", "spacing_label"]
