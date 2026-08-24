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
  The ONLY engine ``--engine auto`` will select, and the only one any
  in-process caller or chain reaches.
* ``matplotlib`` -- the wrf-rust + matplotlib path below, reachable
  only by typing ``--engine matplotlib``, and announcing itself as a
  ``WORKAROUND:`` line on every run.

The render law (CLAUDE.md, Drew 2026-08-06) reserves weather-field
product plots for ``rw_wrfbatch`` and names exactly ONE permitted
fallback, ``tools/da_nowcast_render.py``, which composes multi-panel
sheets from a DA nowcast case directory and serves none of this door's
products.  ``--engine auto`` therefore does not degrade: with no usable
renderer it REFUSES, naming the missing artifact and ``gpuwm
fetch-bridges``, at a nonzero exit.  It used to draw five weather
fields with matplotlib and report success, which is how a box with a
failed bridge staging produced pictures that looked like the
production catalog and were not it.

``--pair A_DIR B_DIR`` composes two runs' rendered PNGs into labeled
side-by-side comparison sheets (:mod:`gpuwm.pair_compose`) -- either
engine's output pairs.

``--streamlines`` / ``--barbs`` choose how the wind is drawn on every
product carrying a wind layer.  Neither given, the engine keeps its
automatic per-grid choice (streamlines on curvilinear and projected
grids, barbs on plain lat/lon) and the invocation is byte-identical to
every earlier release.  ``RUSTWX_WIND_STREAMLINES`` still works and
these two outrank it: a flag a stale shell export could silently
overrule would be a flag that lies.

``--products var:<stored name>`` draws any 2-D field the wrfout carries,
INCLUDING one added to a user's own WRF Registry -- ``MSLP_ANOM`` is
imported as ``wrf_mslp_anom`` and rendered by ``--products
var:wrf_mslp_anom`` with no product definition written anywhere.
``--list-products`` names every ``var:`` row a given file can serve.

Both engines file what they draw through :mod:`gpuwm.render_layout`:
``<--out>/<domain>/<product>/<valid-day>/<file>.png``, by default and
without a flag.  Every picture used to land in one directory, which on
a three-nest run of the rust catalog is five figures of files of every
product and every valid time with nothing but filenames to sort them.
``--layout flat`` restores the v2.4.1 single directory for a consumer
that still needs it.  See ``docs/render-output-layout.md``.

Every output filename carries a domain + resolution token
(:func:`domain_token`: ``d02-3km``, sub-kilometre nests as ``d05-111m``)
read from the file's own ``GRID_ID`` and ``DX``.  Without it two nests
of one run share every other filename component and the second render
at a lead silently overwrote the first -- a forecast lost with no error
and an exit code of 0.  The same spacing appears in the plot subtitle
as ``Δx 3 km``, and plots are labelled with the model that produced
them (``--source-label``, default :func:`default_source_label`) rather
than the GDEX fetch source the rust engine's ``wrf`` store identity
inherits.  That default carries the version of the code that is
EXECUTING (``ArWen 1.8.7``), which is not the same number as
``gpuwm.__version__``: see :func:`default_source_label`.

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
import contextlib
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from gpuwm import explain, render_layout, run_stamp
from gpuwm.io.history_selection import PRODUCT_HISTORY_INPUTS
from gpuwm.science_core import SCIENCE_CORE_REQUIREMENT

#: The pip requirement the render products are certified against, quoted
#: verbatim in the install hint below.  Sourced from gpuwm.science_core so
#: the hint and the pyproject extra cannot drift apart.
WRF_PACKAGE_REQUIREMENT = SCIENCE_CORE_REQUIREMENT

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
#:
#: The BRAND half only.  What actually reaches a plot is
#: :func:`default_source_label`, which appends the executing version.
DEFAULT_SOURCE_LABEL = "ArWen"


def default_source_label() -> str:
    """``ArWen 1.8.7`` -- the brand plus the version that is EXECUTING.

    THE product stamp, and the reason it is a function rather than a
    constant.  Both engines put this string on the plot verbatim: the
    matplotlib engine through :func:`plot_context`, and the rust engine
    through ``--source-label``, which ``rw_wrfbatch`` renders as its
    ``subtitle_right`` without interpreting it.  So whatever this
    returns is literally the label a reader sees on the image, and it
    is the only place on a PNG where the producing build can be named
    at all.

    The number comes from :func:`gpuwm.provenance_gate.executing_version`
    and NOT from ``gpuwm.__version__``, because those two are not the
    same claim.  ``__version__`` asks distribution metadata for the
    version of the distribution NAMED gpuwm, which on a box with a
    stale editable install answers for a tree that is not the one
    running -- exactly the field report this program was opened for,
    where plots were labelled 1.6.2 and the reader believed they had
    installed 1.8, and nothing in the product could say which of them
    was right.  ``executing_version`` prefers the running code's own
    declaration, so the label describes the code that drew the image.

    An install that genuinely cannot name its version contributes
    nothing rather than the ``0+unknown`` sentinel: a plot is not a
    diagnostics channel, and a bare ``ArWen`` is the honest label there.
    ``gpuwm version`` and the render receipt carry the full story.
    """

    from gpuwm.provenance import UNKNOWN_VERSION
    from gpuwm.provenance_gate import executing_version

    try:
        version = executing_version()
    except Exception:                                   # noqa: BLE001
        return DEFAULT_SOURCE_LABEL
    if not version or version == UNKNOWN_VERSION:
        return DEFAULT_SOURCE_LABEL
    return f"{DEFAULT_SOURCE_LABEL} {version}"


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
    # Same ceiling as the rust engine's placement seam, same remedy: the
    # layout's three folders push the longest product names past
    # Windows' MAX_PATH from any real case root, and a savefig that
    # cannot open its target loses the picture outright here.
    spelled = render_layout.fs_path(out_png)
    Path(spelled).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(spelled, dpi=dpi)


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
                   source_label: str | None = None,
                   layout: str = render_layout.DEFAULT_LAYOUT,
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
    # Resolved here rather than in the signature: a default argument is
    # evaluated at import, and this one resolves provenance (a git
    # subprocess).  `gpuwm --help` must not pay for it.
    if source_label is None:
        source_label = default_source_label()
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
                # WHERE the picture goes is one decision, made in one
                # place, for both engines: gpuwm.render_layout.  The
                # filename is unchanged from v2.4.1 -- only the
                # directory beneath `--out` is new -- so a reader who
                # knows the old name still recognises the file.
                out_png = render_layout.place(
                    outdir, domain=token, product=product,
                    day=render_layout.valid_day(stamp), layout=layout,
                    filename=(f"{product}_{token}_"
                              f"{_stamp_for_filename(stamp)}.png"))
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


#: The matplotlib engine's per-product wrfout variable needs, READ OFF
#: gpuwm's own render product catalog rather than declared again here.
#:
#: Three readers now: the --list-products availability report, the render
#: loop's own pre-check (:func:`missing_declared_inputs`), and the
#: ``[output]`` history-selection plan-time dependency warning
#: (:func:`gpuwm.io.history_selection.lost_products`).  They read ONE
#: table, so the status a listing prints, the decision a render makes,
#: and the products a config is warned it is about to lose cannot
#: disagree about which variable a product needs.
#:
#: This view is restricted to :data:`PRODUCTS` because it is the
#: matplotlib engine's own catalog; the shared table additionally
#: carries the rust catalog slugs those names alias onto, which this
#: engine cannot draw.
_MPL_PRODUCT_NEEDS = {
    name: PRODUCT_HISTORY_INPUTS[name] for name in PRODUCTS}


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


#: The global attribute a trimmed run stamps into its own wrfout, naming
#: the variables it deliberately did not write
#: (:meth:`gpuwm.io.history_selection.HistorySelection.wrfout_attrs`).
#: A full-default run stamps nothing, so its absence means "nothing was
#: dropped" and never "this file is old".
HISTORY_DROPPED_ATTR = "GPUWM_HISTORY_DROPPED"


def history_dropped_variables(path) -> frozenset[str]:
    """The variables this wrfout's own run chose not to write.

    Empty when the file does not say -- a full-default run, or any
    artifact written before ``[output]`` existed.  Empty is also the
    answer for a file that cannot be opened at all: this is a
    DIAGNOSTIC, and it must never be the reason a render dies.  A file
    that is really unreadable fails loudly a moment later, in the render
    itself, with a message about the file rather than about a table.
    """

    try:
        import netCDF4

        with netCDF4.Dataset(Path(path)) as dataset:
            raw = getattr(dataset, HISTORY_DROPPED_ATTR, "")
    except Exception:
        return frozenset()
    return frozenset(
        name for name in str(raw).split(",") if name.strip()) or frozenset()


def history_killed_products(paths, product_tokens) -> dict[str, str]:
    """Requested products whose inputs THIS RUN dropped -> why, by name.

    The mapping comes from gpuwm's own render product catalog
    (:data:`gpuwm.io.history_selection.PRODUCT_HISTORY_INPUTS`), which is
    also what the ``[output]`` plan-time warning reads -- so a product
    the config warned about is the same product named here.

    Two deliberate abstentions.  ``all`` asks for whatever the file
    supports and is not a request for a specific product, so it is never
    reported.  A token the table does not carry is not claimed: the
    renderer owns 151 products and answers for their inputs itself
    through ``--list-products``; inventing a dependency here for a recipe
    this table has not verified is the correct-looking-wrong-mapping
    defect ``wind10`` already cost once.
    """

    from gpuwm.io.history_selection import PRODUCT_HISTORY_INPUTS

    dropped: set[str] = set()
    for path in paths:
        dropped |= history_dropped_variables(path)
    if not dropped:
        return {}
    killed: dict[str, str] = {}
    for token in product_tokens:
        token = str(token).strip()
        if token == "all":
            continue
        requirements = PRODUCT_HISTORY_INPUTS.get(token)
        if not requirements:
            continue
        lost = [" or ".join(requirement.split("|"))
                for requirement in requirements
                if all(option in dropped
                       for option in requirement.split("|"))]
        if lost:
            killed[token] = (
                f"{token} is drawn from {', '.join(lost)}, which this "
                "run's [output] history_drop / preset did not write")
    return killed


def _history_selection_action(killed: dict[str, str], *, everything: bool
                              ) -> str:
    lines = "\n".join(f"  {detail}." for detail in killed.values())
    head = ("gpuwm render: every product you asked for needs a variable "
            "this run did not write."
            if everything else
            "note: render cannot draw "
            f"{', '.join(sorted(killed))} from these file(s) -- the "
            "variable(s) they need were not written.")
    return (
        f"{head}\n{lines}\n"
        "  remedy: re-run without dropping the variable(s) -- delete the "
        "name from [output] history_drop, or widen the preset (severe "
        "keeps the storm-scale volumes, full keeps everything).\n"
        "  # to see what THIS file can still draw:\n"
        "  #   gpuwm render --list-products FILE")


_HISTORY_SELECTION_WHY = (
    "The file says so itself: a run that trimmed its history stamps "
    f"{HISTORY_DROPPED_ATTR} into every wrfout it writes, naming the "
    "variables it chose not to write.  Without that stamp the only "
    "available answer would be 'not in file', which reads like a "
    "damaged artifact rather than a configuration choice made before "
    "the run -- and the remedy for the two is not the same.\n\n"
    "The product-to-variable mapping is gpuwm's own render product "
    "catalog (gpuwm.io.history_selection.PRODUCT_HISTORY_INPUTS), the "
    "same table the [output] plan-time warning reads at config "
    "resolution, so the product named here is the product that warning "
    "named hours earlier.")


def history_selection_refusal(paths, product_tokens) -> str | None:
    """Refusal text when EVERY requested product lost an input, else None.

    Nothing can be drawn, so this is a refusal and not a note: it is
    raised at the front door, before a run directory is claimed and
    before a wrfout is opened for rendering, and it names the variable,
    the product and the way back.
    """

    tokens = [str(token).strip() for token in product_tokens
              if str(token).strip()]
    killed = history_killed_products(paths, tokens)
    if not killed or set(killed) != set(tokens):
        return None
    return explain.layered(_history_selection_action(killed, everything=True),
                           _HISTORY_SELECTION_WHY)


def history_selection_notice(paths, product_tokens) -> str | None:
    """One ``note:`` line for the requested products this run cannot draw.

    The survivors are still drawn -- refusing the whole command because
    one product of four lost its input would be worse than the silence
    it replaces.
    """

    killed = history_killed_products(paths, product_tokens)
    if not killed:
        return None
    return explain.layered(_history_selection_action(killed, everything=False),
                           _HISTORY_SELECTION_WHY)


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
    usable rust engine this REFUSES rather than answering from the other
    engine's five: ``--engine auto`` would not draw those products
    either (render law, audit F7), so listing them would be a menu
    nothing serves.  ``--engine matplotlib`` still lists its own.
    """

    from gpuwm import rustwx

    try:
        engine, why = _resolve_engine(args.engine)
    except (RuntimeError, FileNotFoundError) as exc:
        # ``--engine rust`` is a refusal now, not a fallback, and this
        # entry point is reached before the render path's own try.  The
        # other two failures here print one line and exit 2; a renderer
        # that cannot answer for itself is the same shape of answer.
        print(f"render: {exc}", file=sys.stderr)
        return 2
    notice = matplotlib_workaround_notice(engine)
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


def require_renderer() -> Path:
    """The usable ``rw_wrfbatch``, or the refusal that names its absence.

    The SAME gate :func:`_resolve_engine` reads, at the LAST seam before
    the engine is launched.  The CLI already asked, but the in-process
    callers do not go through the CLI (``gpuwm go``'s first-products
    leg, the DA gallery, the verification door's synoptic panels,
    tests), and a gate that only one caller passes through is a gate the
    next caller walks around.  There is no fallback to offer here --
    weather fields belong to this binary -- so the reason is raised.
    """

    from gpuwm import rustwx

    renderer = rustwx.find_renderer()
    if renderer is None:
        raise RuntimeError(_unresolvable_renderer_refusal(
            "it is not built (nothing on the resolution ladder holds a "
            "rw_wrfbatch binary)"))
    reason = renderer_refusal(renderer)
    if reason is not None:
        raise RuntimeError(_unresolvable_renderer_refusal(
            f"the renderer at {renderer} may not be used by this tree: "
            f"{reason}"))
    return renderer


def render_series_rust(paths, *, products: str, timeidx: int | None,
                       outdir: Path, size: tuple[int, int],
                       heavy: bool = False,
                       source_label: str | None = None,
                       layout: str = render_layout.DEFAULT_LAYOUT,
                       overlays: Path | None = None,
                       annotate: Path | None = None,
                       streamlines: bool | None = None,
                       ) -> tuple[list[Path], list[str],
                                  list[tuple[str, str]]]:
    """One store over a whole wrfout SERIES; ``(written, failures, skipped)``.

    :func:`render_wrfouts_rust` renders one file per store, which is the
    campaign convention and the right default for independent inputs.
    It is also the one thing a WINDOWED product cannot be drawn under:
    ``qpf_6h`` is defined as F012 minus F006, so a store holding only
    F012 has nothing to difference and the engine skips it by name.
    This renders the whole series into ONE store, which is what a
    verification door's 6 h accumulation -- two separate hourly wrfout
    files -- actually needs.

    The domain token comes from the LAST file, the frame whose valid
    time the panels carry; every file in a series is one nest by
    construction, because a store that mixed two nests would merge two
    grids into one run.  Placement and rebranding are
    :func:`_place_engine_output`'s, unchanged, so these panels land in
    the same ``<out>/<domain>/<product>/<valid-day>/`` layout as every
    other render.
    """

    from gpuwm import rustwx

    series = [Path(item) for item in paths]
    if not series:
        raise ValueError("a render series needs at least one wrfout file")
    renderer = require_renderer()
    if source_label is None:
        source_label = default_source_label()
    outdir.mkdir(parents=True, exist_ok=True)
    subject = series[-1]
    token = domain_token(_domain_tag(subject), _grid_spacing_m(subject))
    width, height = size
    with scratch_store(outdir) as store:
        written, failures, skipped = rustwx.run_renderer_series(
            renderer, series, store_root=store, out_dir=outdir,
            products=products, frames="all" if timeidx is None
            else str(timeidx), width=width, height=height, heavy=heavy,
            source_label=source_label, overlays=overlays,
            annotate=annotate, streamlines=streamlines)
    written = [_place_engine_output(png, outdir, token, layout)
               for png in written]
    return written, failures, skipped


def render_wrfouts_rust(paths, *, products: str, timeidx: int | None,
                        outdir: Path, size: tuple[int, int],
                        heavy: bool = False,
                        source_label: str | None = None,
                        layout: str = render_layout.DEFAULT_LAYOUT,
                        overlays: Path | None = None,
                        annotate: Path | None = None,
                        streamlines: bool | None = None,
                        ) -> tuple[list[Path], list[str],
                                   list[tuple[str, str]]]:
    """Rusty Weather engine; ``(written, failures, skipped)``.

    One renderer invocation per wrfout file, each with its own scratch
    store (two files in one store would merge into one run's timeline;
    per-file isolation keeps every input independently comparable, the
    campaign convention).  The scratch store lives under the output
    directory and is removed as soon as the file's render finishes.

    ``overlays``/``annotate`` are JSON files in the renderer's own
    geographic-overlay schema (coordinates in degrees; see
    ``tools/rustwx/crates/rustwx-products/src/geographic_overlays.rs``).
    They are what the tile-streamed and DA lanes needed and could not
    have: a boundary-zone box, tile seams, storm-report markers, radar
    sites and range rings, on a panel the production renderer drew.
    Omitted, the renderer runs no overlay code and the PNGs are
    byte-identical to every earlier build.
    """

    from gpuwm import rustwx

    renderer = require_renderer()
    if source_label is None:
        source_label = default_source_label()
    outdir.mkdir(parents=True, exist_ok=True)
    frames = "all" if timeidx is None else str(timeidx)
    width, height = size
    written: list[Path] = []
    failures: list[str] = []
    skipped: list[tuple[str, str]] = []
    for path in (Path(p) for p in paths):
        # The domain this file PROVES, read the same way the matplotlib
        # engine reads it, from the file's own GRID_ID/DX.  The engine
        # spells the same token into its filenames, but evidence from
        # the file beats a transcription, and one file's outputs all
        # belong to one nest whatever any filename says.
        token = domain_token(_domain_tag(path), _grid_spacing_m(path))
        with scratch_store(outdir) as store:
            file_written, file_failures, file_skipped = rustwx.run_renderer(
                renderer, path, store_root=store, out_dir=outdir,
                products=products, frames=frames, width=width,
                height=height, heavy=heavy, source_label=source_label,
                overlays=overlays, annotate=annotate,
                streamlines=streamlines)
        file_written = [_place_engine_output(png, outdir, token, layout)
                        for png in file_written]
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


def _place_engine_output(png: Path, outdir: Path, domain: str,
                         layout: str) -> Path:
    """Rebrand one engine-written PNG and file it under the layout.

    The rust engine writes every picture straight into ``--out-dir``
    with a name of its own, which is what made a real run's output a
    single directory of thousands of files.  It is not asked to change:
    the vendored crate stays byte-identical to its campaign builds, and
    the placement happens here, at the same seam the rebrand already
    happened at -- one move per file, after the renderer has exited and
    the file is complete.

    Every failure degrades to leaving the picture where the engine put
    it, and SAYS SO on stderr, naming the file and why.  A drawn image
    at the old path beats no image over filing -- layout is not
    correctness -- but the silent form of this exact fallback is how a
    parser that could not read the engine's one-digit cycle hours
    shipped every 00Z-09Z run flat under a front door printing ``layout
    nested``.  The caller's returned list names wherever the file
    actually is, so nothing downstream is told about a file that is not
    there.
    """

    png = _rebrand_engine_output(png)
    if layout == render_layout.FLAT:
        return png
    parsed = render_layout.parse_engine_output(png.name, domain=domain)
    if parsed is None:
        # Not a name this engine's grammar produces -- so nothing here
        # knows which product or valid time it is, and inventing a
        # folder for it would file it under a guess.
        print(f"render: warning: left flat, filename does not parse as "
              f"engine output: {png}", file=sys.stderr)
        return png
    filed_domain, product, day = parsed
    # The delivered name drops the domain and product tokens, because
    # the two folders it is about to sit in spell exactly those and a
    # frame carrying them twice cost a measured delivery 310 characters
    # -- past MAX_PATH for every tool the recipient opens it WITH, which
    # `fs_path` does nothing about.  Done here, at the organisation
    # step, rather than in the engine: the vendored crate stays
    # byte-identical to its campaign builds, and `--layout flat` keeps
    # the v2.4.1 name (this branch is nested-only; flat returned above).
    target = render_layout.place(
        outdir, domain=filed_domain, product=product, day=day,
        filename=render_layout.delivered_name(
            png.name, domain=filed_domain, product=product),
        layout=layout)
    if target == png:
        return png
    try:
        # Through render_layout.fs_path: three folders plus the engine's
        # filename put the longest products past Windows' MAX_PATH from
        # any real case root, and without this the warning below fires
        # for exactly those products -- the layout ruling inverted for
        # `composite_reflectivity` while `2m_temperature` beside it
        # files correctly.
        spelled = render_layout.fs_path(target)
        Path(spelled).parent.mkdir(parents=True, exist_ok=True)
        os.replace(render_layout.fs_path(png), spelled)
    except OSError as error:
        print(f"render: warning: left flat, could not move into layout "
              f"({error}): {png}", file=sys.stderr)
        return png
    return target


#: A render's working scratch lives BESIDE the delivered tree, in
#: ``<delivery><SCRATCH_SUFFIX>/``, never inside it.
SCRATCH_SUFFIX = ".render-scratch"


def scratch_root_for(outdir) -> Path:
    """Where working scratch for a delivery to ``outdir`` belongs.

    A SIBLING of the delivered directory.  Scratch used to be created
    inside it (``mkdtemp(dir=outdir)``), which breaks a delivery two
    ways, both seen on 2026-08-20: 25 ``.rwstore-*`` directories were
    left inside a delivered product folder, with paths long enough that
    Windows could not list the folder at all; and a ``tar`` pull of the
    tree raced a live render -- ``tar: .rwstore-p_4f7zzy: File removed
    before we read it``.  The removal is best-effort by nature (a
    memory-mapped hour file can hold its handle past process exit), so
    the guarantee "a delivered tree contains only products" cannot rest
    on cleanup.  It rests on the location.

    The sibling is taken beside the CASE ROOT, not beside the run
    folder: what gets pulled, tarred and browsed is the case folder the
    caller named with ``--out``, so scratch parked one level inside it
    would still be in the archive and still be in the way.  A run folder
    is recognised by its own name (``run_stamp.is_run_folder``), so this
    needs nothing threaded through from the door.
    """

    outdir = Path(os.path.abspath(outdir))
    anchor = outdir.parent if run_stamp.is_run_folder(outdir) else outdir
    return anchor.parent / f"{anchor.name}{SCRATCH_SUFFIX}"


@contextlib.contextmanager
def scratch_store(outdir, *, prefix: str = "rwstore-"):
    """One renderer invocation's working store, outside the delivery.

    Removed on exit; the sibling root goes with the last store in it, so
    concurrent renders into the same delivery each keep their own and
    none removes another's (``rmdir`` refuses a non-empty directory).

    A parent directory that cannot hold the sibling (read-only, or a
    delivery written straight into a mount point) is not a reason to
    fail a render OR to fall back into the delivered tree: the store
    goes to the system temp area instead, named so it is recognisable.
    """

    import tempfile

    root = scratch_root_for(outdir)
    try:
        root.mkdir(parents=True, exist_ok=True)
        store = Path(tempfile.mkdtemp(prefix=prefix, dir=root))
    except OSError as error:
        root = Path(tempfile.mkdtemp(prefix="gpuwm-render-scratch-"))
        store = Path(tempfile.mkdtemp(prefix=prefix, dir=root))
        print(f"render: scratch beside {outdir} was not writable "
              f"({error}); working store is {store}", file=sys.stderr)
    try:
        yield store
    finally:
        _remove_scratch_store(store)
        try:
            root.rmdir()
        except OSError:
            pass


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


def renderer_refusal(renderer) -> str | None:
    """Why this tree may not draw with ``renderer``, or None if it may.

    THE single answer to "is this renderer usable", and the only one.
    Both seams that can launch the engine read it -- :func:`_resolve_engine`
    for the CLI, :func:`render_wrfouts_rust` for every in-process caller
    (``gpuwm go``'s first-products leg, the DA gallery) -- so a renderer
    can never be usable at one seam and foreign at the other.  Two
    independently correct gates landed here in one release and would
    have been exactly that: two definitions of usable, diverging on the
    first case they disagreed about.

    Two questions, in the order a reader wants them answered:

    * THE CONTRACT (task #106).  :func:`gpuwm.rustwx.probe_renderer`
      runs ``--help``, then the ``--abi`` handshake against
      :data:`gpuwm.rustwx.RENDERER_ABI_MARKER`.  This is the decisive
      one, because it is a statement about the BINARY's own answer
      rather than about where it sits: two builds of ``rw_wrfbatch``
      with different md5s both passed the old ``--help``-only probe and
      both reported ``verified``, and a build predating the handshake
      says ``unknown option --abi`` on exit 2.
    * THE PROVENANCE (task #125).
      :func:`gpuwm.provenance_gate.renderer_bridge_refusal` refuses a
      binary that answers the contract correctly and still belongs to
      another tree -- the case the handshake cannot see, because a
      sibling checkout at the same contract version answers it
      perfectly.  ``find_renderer`` returns the first candidate that is
      a file and its last candidate is ``~/.gpuwm/bridges``, a directory
      every gpuwm on the machine writes into and none of them owns, so
      "nobody chose this binary" is a real and silent state.

    The reason is returned rather than raised, because what to DO about
    it belongs to the caller: ``auto`` degrades and says why, an
    explicit ``--engine rust`` refuses, and a direct in-process render
    has no fallback to offer at all.
    """

    from gpuwm import rustwx
    from gpuwm.provenance_gate import renderer_bridge_refusal

    ok, evidence = rustwx.probe_renderer(renderer)
    if not ok:
        return evidence
    # The provenance clause carries its own remedies (build here, or
    # declare the staged binary through the environment), so it is
    # returned verbatim rather than re-worded.
    return renderer_bridge_refusal(renderer, env_var=rustwx.RENDERER_ENV)


def _unresolvable_renderer_refusal(reason: str) -> str:
    """The one refusal both ``auto`` and ``rust`` raise, layered.

    THE render-law fix (hidden-scope audit F7, 2026-08-18).  Weather
    fields belong to ``rw_wrfbatch``; the law (CLAUDE.md, Drew
    2026-08-06) names exactly one fallback, ``da_nowcast_render.py``,
    and that tool draws multi-panel sheets from a DA case directory --
    it has no product for a wrfout's reflectivity, temperature, wind,
    precipitation or OLR panel.  So there is nothing lawful to fall back
    to here, and the answer is a refusal that names the missing artifact
    and stages it.

    ``gpuwm fetch-bridges`` leads because it is the remedy that works on
    a wheel install, which is where the defect was found: staging failed
    silently, ``--engine auto`` drew five matplotlib weather fields, and
    the command exited 0.  The cargo one-liner follows for a checkout,
    and the explicit matplotlib workaround is named last -- naming the
    exit is not the same as taking it (1.8.8 refusal sweep).
    """

    from gpuwm import rustwx

    # The reason may itself be a LAYERED message -- the provenance
    # clause is one -- and nesting two `[[explain]]` marks in one string
    # truncates this remedy at the inner mark for every consumer that
    # splits on the first one (`explain.render`, and the JSON catalog
    # that carries the action half into a document).  So the inner
    # explanation is lifted out and appended to the outer one, which is
    # where it belonged: one message, one action half, one why half.
    reason, nested_why = explain.split(str(reason))
    reason = reason.rstrip()
    # No `gpuwm render:` prefix here.  Every terminal caller adds one
    # (`render_main` and `catalog_main` both print "render: " + this),
    # and a message that carries its own prefix printed "render: gpuwm
    # render: ..." on the first line a refused user sees.  The subject
    # is named in the sentence instead, which also reads correctly
    # where there is no prefix at all -- the run-plan catalog document
    # carries this string verbatim in its `error` field.
    action = (
        f"the rust render engine ({rustwx.RENDERER_NAME}) is "
        f"not usable here: {reason}\n"
        "  Weather-field product plots come from the real Rust renderer, "
        "so this is refused rather than drawn by something else.\n"
        "  remedy: gpuwm fetch-bridges\n"
        f"  # stages {rustwx.RENDERER_NAME} and the basemap assets it "
        "draws from\n"
        f"  #   ... or, from a checkout: {rustwx.CARGO_BUILD_HINT}\n"
        f"  #   ... or set {rustwx.RENDERER_ENV} to a built binary you "
        "mean to use\n"
        "  # `gpuwm render --engine matplotlib` still exists as a NAMED "
        "WORKAROUND;\n"
        "  # it draws five analysis-grade panels, not the production "
        "catalog.")
    why = _RENDER_LAW_WHY
    if nested_why.strip():
        why = f"{why}\n\n{nested_why.strip()}"
    return explain.layered(action, why)


#: How every message here names the one engine allowed to draw a
#: weather field, so the phrase cannot drift between call sites.
_RENDERER_SUBJECT = "the Rust renderer rw_wrfbatch, through gpuwm.rustwx"

_RENDER_LAW_WHY = (
    "The render law (CLAUDE.md, Drew 2026-08-06) permits exactly one "
    "fallback for weather fields, `tools/da_nowcast_render.py`, and that "
    "tool composes multi-panel sheets from a DA nowcast case directory "
    "-- it serves none of this door's products.  `--engine auto` used to "
    "degrade to a SECOND fallback: on a box where bridge staging failed "
    "it drew composite reflectivity, 2 m temperature, 10 m wind, "
    "accumulated precipitation and OLR with matplotlib and exited 0, so "
    "an operator got pictures and no signal that the 151-product "
    "production catalog had never run.  `auto` now has two answers, the "
    "rust engine or this refusal, and the exit code says which.")


def _resolve_engine(requested: str) -> tuple[str, str]:
    """(engine, why) for ``--engine auto|rust|matplotlib``.

    Rust is selected exactly when the binary resolves AND
    :func:`renderer_refusal` has nothing to say about it.  ``auto`` and
    ``rust`` now differ only in wording: NEITHER degrades, because the
    thing they would degrade to is matplotlib drawing weather fields,
    which the render law does not allow (audit F7).  ``auto`` means
    "resolve the engine for me", not "draw with whatever is lying
    around".

    The explicit path used to return the resolved binary WITHOUT the
    probe -- only ``auto`` asked the contract question -- so the one
    caller that pinned ``--engine rust`` to be certain of the real
    renderer was exactly the caller that could still get a foreign one:
    with a stale bridge staged, ``auto`` fell back naming the mismatch
    while ``rust`` ran the stale build silently.  An explicit request is
    a statement about which engine must draw, so failing its contract is
    a refusal, never a silent substitution and never a fallback.

    ``--engine matplotlib`` remains reachable, by that name only, and
    every run of it prints :func:`matplotlib_workaround_notice`.
    """

    if requested == "matplotlib":
        return "matplotlib", "requested"
    from gpuwm import rustwx

    renderer = rustwx.find_renderer()
    if renderer is None:
        if requested == "rust":
            raise RuntimeError(
                "--engine rust: the renderer is not built; "
                + rustwx.renderer_remedy())
        raise RuntimeError(_unresolvable_renderer_refusal(
            "it is not built (nothing on the resolution ladder, including "
            "~/.gpuwm/bridges, holds a rw_wrfbatch binary)"))
    reason = renderer_refusal(renderer)
    if reason is None:
        return "rust", str(renderer)
    if requested == "rust":
        # NO SILENT SUBSTITUTION -- an explicit --engine rust that got a
        # different engine is the defect this refusal exists for -- but
        # the exit is named, because one demonstrably exists: the same
        # command with --engine matplotlib renders the analysis-grade
        # subset on this identical tree.  Refusing to choose for the
        # caller is not the same as refusing to tell them what the
        # choices are (1.8.8 refusal sweep).
        raise RuntimeError(
            f"--engine rust: the renderer at {renderer} failed its "
            f"contract check: {reason}\n"
            f"  Rebuild the renderer ({rustwx.CARGO_BUILD_HINT}), stage "
            f"one with `gpuwm fetch-bridges`, or pass --engine matplotlib "
            f"to take the named workaround outright.")
    raise RuntimeError(_unresolvable_renderer_refusal(
        f"the renderer at {renderer} failed its contract check: {reason}"))


def matplotlib_engine_gap():
    """The requirement that stops the matplotlib engine drawing, or ``None``.

    The fallback engine is not a fallback until this is ``None``.  It
    computes every derived field through ``wrf`` (pip distribution
    ``wrf-rust``), the mandated science core, which lives behind the
    same ``render`` extra the rust engine's documentation offers as the
    thing to fall back FROM.  Asked without importing, so this is safe
    to call on any path including ``--help``.
    """

    from gpuwm import capabilities

    if capabilities.is_installed(capabilities.SCIENCE_CORE.module):
        return None
    return capabilities.SCIENCE_CORE


def drawable_engine() -> tuple[str | None, str]:
    """Which engine ``gpuwm go`` may chain into, and why -- or ``(None, why)``.

    Exactly one engine is chainable: the rust one.  ``(None, why)`` when
    it is not staged or fails its contract, and the chain then SKIPS its
    render stage with that reason printed, which is the lawful shape of
    "no imagery here" -- a forecast that finished plus a named absence.

    The matplotlib engine is deliberately absent from this answer even
    when its science core is installed (audit F7).  It draws weather
    fields, the render law reserves those for ``rw_wrfbatch``, and a
    chain that reaches it automatically is precisely the second fallback
    the law forbids.  It stays reachable by name at the ``gpuwm render``
    front door, where a human typed it.
    """

    try:
        engine, why = _resolve_engine("auto")
    except Exception as error:                              # noqa: BLE001
        # Deliberately broad, and only here.  This function is asked by
        # `gpuwm go` DURING a chain, purely to decide whether to run the
        # render stage; resolving the rust engine means launching it for
        # its contract handshake, and a launch that fails for any reason
        # must degrade this answer, never end a forecast that has
        # already succeeded.  The reason is carried, not swallowed --
        # and the first line is what the chain prints, so the layered
        # explain half is dropped here rather than pasted into a stage
        # log.
        first = str(error).split("[[explain]]")[0].strip()
        return None, first or f"the rust engine could not be probed: {error}"
    if engine == "rust":
        return "rust", why
    return None, (
        f"the rust render engine is not available ({why}); weather-field "
        f"product plots come from {_RENDERER_SUBJECT}, so the chain draws "
        "nothing rather than substituting another engine")


def engine_refusal(engine: str, requested: str, why: str) -> str | None:
    """The refusal for an engine that cannot draw, or ``None``.

    THE fix for the wrong-remedy-first defect.  A bare install used to
    print ``rust render engine not available ... Build it with: gpuwm
    fetch-bridges`` -- an advisory about the OTHER engine -- and then
    die in a traceback whose tail carried the remedy that would actually
    have worked.  The first line a reader sees is now the one that fixes
    the thing that is broken.

    Both routes are named, in the order that costs least: staging the
    rust engine is one command, needs no extra, and draws the whole
    catalog (measured 161 PNGs against 5 renderable matplotlib products
    on the same file).  An explicit ``--engine matplotlib`` is a request
    for the fallback, so its own remedy leads there instead.
    """

    if engine != "matplotlib":
        return None
    gap = matplotlib_engine_gap()
    if gap is None:
        return None
    if requested == "matplotlib":
        action = (
            "gpuwm render --engine matplotlib: the matplotlib engine "
            f"needs {gap.label}, which is not installed.\n"
            "  The fallback engine computes every derived field through "
            "it, so it needs the same extra as the rust engine it falls "
            "back from.\n"
            f"{gap.remedy}")
    else:
        action = (
            f"gpuwm render: neither render engine can draw here.\n"
            f"  The rust engine is not available ({why}), and the "
            f"matplotlib fallback needs {gap.label}, which is not "
            "installed.\n"
            "  remedy: gpuwm setup\n"
            "  # stages the rust render engine (no extra needed); it "
            "draws the\n"
            f"  # full catalog, against {len(PRODUCTS)} products for the "
            "fallback\n"
            "  #   ... or, for the matplotlib fallback engine:\n"
            "  #   pip install 'gpuwm[render]'")
    return explain.layered(action, _ENGINE_WHY)


_ENGINE_WHY = (
    "Refused at the front door, before a single wrfout was opened.  The "
    "version this replaces printed an advisory naming the rust engine's "
    "build command -- a remedy for the engine that was NOT selected -- "
    "and then raised, so the correct install line arrived at the tail of "
    "a traceback, below the wrong one.\n\n"
    "`gpuwm render --list-products FILE` is deliberately not refused "
    "here: the matplotlib catalog reads the file with netCDF4 alone and "
    "answers correctly on an install with no render extra at all.")


#: How many products each engine can draw, for the fallback notice.
#: The rust catalog's 151 is its implicit-render candidate count on a
#: typical wrfout; the matplotlib count is read from :data:`PRODUCTS`,
#: so adding a fallback product cannot leave this notice overstating
#: what the fallback costs.
_RUST_CATALOG_PRODUCTS = 151


def matplotlib_workaround_notice(engine: str) -> str | None:
    """ONE line saying the matplotlib engine is drawing, in the workaround voice.

    Silence here cost a pilot a whole session: four products came out,
    they looked right, and nothing said the 151-product rust catalog was
    simply not installed.  An engine that does not announce itself is
    indistinguishable from the real thing until someone counts.

    It is a WORKAROUND now, not a fallback, and the word is the point.
    Nothing degrades into this engine any more (audit F7): it is
    reachable only by typing ``--engine matplotlib``, and "Fixed means
    default" (Drew 2026-08-10) says an opt-in remedy must be REPORTED as
    a workaround every time it runs.  The prefix is this tree's existing
    spelling for that -- ``gpuwm/static/rust_bridge.py`` prints the same
    ``WORKAROUND:`` line whenever the Python geog reader stands in.

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

    if engine != "matplotlib":
        return None
    from gpuwm import bridges, rustwx

    fix = bridges.install_aware_one_line_hint(
        rustwx.CARGO_BUILD_HINT, rustwx.RUSTWX_CRATE_RELATIVE)
    return (f"WORKAROUND: render engine matplotlib -- the render law "
            f"reserves weather-field product plots for {_RENDERER_SUBJECT}; "
            f"{len(PRODUCTS)} of the rust catalog's "
            f"{_RUST_CATALOG_PRODUCTS} products are renderable this way. "
            f"Stage the real engine with `gpuwm fetch-bridges`, or build "
            f"it: {fix}")


def skip_notice(skipped: list[tuple[str, str]],
                wrote_any: bool = True) -> str | None:
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

    The action half also states what the skips mean for the exit code,
    and it must state it TRUTHFULLY in both arms.  With at least one
    image drawn (``wrote_any``), skips do not change the exit code and
    the notice says so.  With NOTHING drawn, the skips are exactly why
    the command exits nonzero (the deliberate ask-for-one-absent-product
    arm of the main dispatch), and the old reassurance -- "not a failure
    and does not change the exit code" -- was false there: a real user
    who requested ``lifted_index`` from a wrfout got rendered=0, exit 1,
    and a note insisting nothing had failed.  That arm now says nothing
    rendered and points at ``--list-products``, which prints the
    per-product verdict and reason for THIS file.
    """

    if not skipped:
        return None
    names = sorted({product for product, _detail in skipped})
    if wrote_any:
        verdict = ("That is not a failure and does not change the "
                   "exit code")
    else:
        verdict = ("Nothing else was drawn, so nothing rendered and the "
                   "exit code is nonzero; run --list-products to see "
                   "which products this file can support")
    return explain.layered(
        f"note: render skipped {len(skipped)} product render(s) "
        f"({', '.join(names)}) -- the file(s) do not carry their declared "
        f"input fields.  {verdict}",
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
    # Pillow is checked HERE, by resolution, rather than by catching the
    # import of `gpuwm.pair_compose`.  Two defects in one line:
    #
    # * the message named `gpuwm[render]`, which is wrf-rust + pyshp and
    #   has never contained Pillow.  Pillow arrives transitively with
    #   matplotlib, a BASE dependency, so the extra it named would not
    #   have installed it and the reader would have run the command,
    #   waited for a wheel, and met the same refusal.
    # * the guard was inert: `pair_compose` imports PIL inside its
    #   functions, so this module imports cleanly with no Pillow at all
    #   and the ImportError arrived later, from inside `compose_pairs`,
    #   as a traceback.
    from gpuwm import capabilities

    if not capabilities.is_installed(capabilities.PILLOW.module):
        print(explain.render(
            capabilities.refusal("gpuwm render --pair", capabilities.PILLOW),
            explain=explain.explain_enabled(args),
            command="gpuwm render"), file=sys.stderr)
        return 2
    try:
        from gpuwm.pair_compose import compose_pairs
    except ImportError as error:
        # Kept as the floor under the resolution check above: a Pillow
        # that resolves and then fails to import (a broken wheel, a
        # partial uninstall) is still a refusal, not a traceback.
        remedy = capabilities.remedy_for_error(error) or ""
        print(f"render: --pair could not load its image toolkit: {error}\n"
              + remedy, file=sys.stderr)
        return 2
    left, right = (Path(p) for p in args.pair)
    labels = args.pair_labels or (None, None)
    # A pair sheet is an output of this invocation like any picture, so
    # it gets its own run folder too.  No inputs are passed: a
    # comparison of two runs belongs to neither one's cycle, so the
    # stamp carries the launch instant and no init time.
    _claim_run_dir(args)
    try:
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
    finally:
        _publish_run_dir(args)


def _claim_run_dir(args: argparse.Namespace, inputs=()) -> Path:
    """Rebind ``args.out`` to this render's own run folder, and say so.

    One render is one run of the renderer, and two renders into one
    ``--out`` used to drop their pictures among each other -- same
    directory, same filenames for the same product and valid time, the
    second silently overwriting the first.  The stamped level fixes
    that above the layout: ``<domain>/<product>/<valid-day>`` is
    untouched underneath it.

    ``args.out`` is REBOUND rather than threaded through as a second
    value because every path this door writes -- the pictures, the pair
    sheets, the per-file scratch store, the closing summary line -- is
    computed from it.  Two names for the output directory is how one of
    them ends up being the one nobody updated.

    The initialisation time comes from the FIRST INPUT THAT PROVES ONE
    (``SIMULATION_START_DATE``), and is omitted when none does; the
    ``--pair`` route passes no inputs and takes a launch-only stamp,
    because a comparison sheet belongs to no single run's cycle.

    A freshly allocated folder is NOT yet published: ``latest-run.txt``
    stays where it was until :func:`_publish_run_dir` sees at least one
    PNG in the folder, and a render that produced none removes the
    folder again.  A failed render used to leave an empty stamped
    folder with the pointer naming it, so a script following the
    documented pointer could not tell a failed render from a successful
    empty one (UX finding N5).
    """

    out = Path(args.out)
    enabled = run_stamp.run_stamp_enabled(args)
    if enabled and run_stamp.is_run_folder(out):
        # The caller named the run folder.  A second stamp level inside
        # it would bury the path they typed, so it is honoured verbatim
        # -- and said out loud, because "I asked for a stamp and got no
        # new folder" is otherwise a silent surprise.
        print(f"render: run folder {out} (named on the command line; "
              f"drawing into it rather than stamping inside it)")
    run_dir = run_stamp.resolve(
        out, init=run_stamp.first_wrfout_init(inputs), enabled=enabled,
        publish=False)
    if enabled and run_dir != out:
        print(f"render: run folder {run_stamp.relative_to_case(run_dir, out)}"
              f" under {out}")
    args.out = run_dir
    # What _publish_run_dir needs on every exit path: whether THIS
    # invocation allocated a fresh folder, and under which case root.
    args._run_claim_created = run_dir != out
    args._run_claim_root = out
    return run_dir


def _publish_run_dir(args: argparse.Namespace) -> None:
    """Publish or retract this render's run folder by what it produced.

    Called on EVERY exit path after :func:`_claim_run_dir` -- the
    normal return and both exception returns -- and a no-op when this
    invocation allocated no fresh folder (``--run-stamp off``, or an
    ``--out`` already inside a run).

    At least one PNG under the folder publishes it: ``latest-run.txt``
    moves to name it, exactly as before.  No PNG at all retracts it:
    the folder is removed and the pointer is left untouched, so a
    script reading the pointer always lands on the newest run that
    actually drew something, and a failed render leaves no new folder
    to mistake for output.
    """

    if not getattr(args, "_run_claim_created", False):
        return
    args._run_claim_created = False
    run_dir = Path(args.out)
    case_root = Path(args._run_claim_root)
    if any(run_dir.rglob("*.png")):
        run_stamp.record_latest(case_root, run_dir)
        return
    import shutil

    try:
        shutil.rmtree(run_dir)
    except OSError:
        # A folder that cannot be removed is left; it still was not
        # published, which is the half a pointer-following script needs.
        return
    print("render: no images were produced, so run folder "
          f"{run_stamp.relative_to_case(run_dir, case_root)} was not "
          f"published (removed; {run_stamp.LATEST_POINTER} untouched)",
          file=sys.stderr)


def render_main(args: argparse.Namespace) -> int:
    # Which tree is drawing these plots.  Idempotent -- `gpuwm.cli.main`
    # has normally already announced -- and here as well because this
    # handler is reachable without that front door.
    from gpuwm.provenance_gate import announce

    announce("gpuwm render")
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
        # Through the layering boundary, not raw: the bridge refusal
        # `_resolve_engine` can raise is composed with `explain.layered`,
        # and printing it unrendered would put the [[explain]] sentinel
        # on a terminal.
        print("render: " + explain.render(
            str(exc), explain=explain.explain_enabled(args),
            command="gpuwm render"), file=sys.stderr)
        return 2
    # BEFORE the fallback advisory, and before any wrfout is opened: an
    # engine that cannot draw is refused with ITS OWN remedy first.
    # `--list-products` is exempt because the matplotlib catalog needs
    # only netCDF4 and is a working path on a bare install.
    if not args.list_products:
        refusal = engine_refusal(engine, args.engine, why)
        if refusal is not None:
            print(explain.render(
                refusal, explain=explain.explain_enabled(args),
                command="gpuwm render"), file=sys.stderr)
            return 2
    notice = matplotlib_workaround_notice(engine)
    if notice is not None:
        print(notice, file=sys.stderr)
    if engine == "rust":
        from gpuwm import rustwx

        blank = missing_basemap_notice(rustwx.find_renderer())
        if blank is not None:
            print(blank, file=sys.stderr)
    if args.list_products:
        return list_products_main(args, engine)
    # THE [output] PRE-CHECK, before a run directory is claimed and
    # before a wrfout is opened for rendering.  A run that trimmed its
    # history says so in its own files, so a request for a product whose
    # input that run dropped is answerable HERE, by name, with the way
    # back -- instead of the bare "not in file" this used to become,
    # which reads like a damaged artifact rather than the configuration
    # choice it is.  Exempt: --list-products (it exists to answer this),
    # and --products all (a request for whatever the file supports).
    requested = ([token.strip() for token in args.products.split(",")
                  if token.strip()])
    history_refusal = history_selection_refusal(args.wrfout, requested)
    if history_refusal is not None:
        print(explain.render(
            history_refusal, explain=explain.explain_enabled(args),
            command="gpuwm render"), file=sys.stderr)
        return 2
    history_note = history_selection_notice(args.wrfout, requested)
    if history_note is not None:
        print(explain.render(
            history_note, explain=explain.explain_enabled(args),
            command="gpuwm render"), file=sys.stderr)
    print(f"render: engine {engine} ({why})")
    # AFTER every refusal and after --list-products: this creates a
    # directory, and a command that draws nothing must leave none --
    # which is also why every exit path below runs _publish_run_dir.
    _claim_run_dir(args, args.wrfout)
    try:
        # WHERE the pictures will be, printed BEFORE they are drawn, so
        # a script driving this can compute the path it is going to
        # watch rather than glob for it afterwards.
        # One separator on both arms: the reader's own --out arrives
        # with this platform's, and gluing a forward-slash template
        # behind it printed a path in two spellings (UX finding N24).
        print(f"render: layout {args.layout} -- "
              + (render_layout.describe(str(args.out))
                 if args.layout == render_layout.NESTED
                 else f"{args.out}{os.sep}<file>.png (legacy flat)"))
        if engine == "rust":
            # The receipt line for WHICH engine binary drew these
            # products.  Recorded on the passing path too: "the in-tree
            # engine ran" is the fact that was missing when a foreign
            # engine's 153 products had to be explained after the fact.
            from gpuwm import rustwx
            from gpuwm.provenance_gate import bridge_tree_match

            verdict = bridge_tree_match(rustwx.find_renderer(),
                                        env_var=rustwx.RENDERER_ENV)
            print(f"render: engine bridge {verdict.verdict} "
                  f"({verdict.basis})")
            try:
                written, failures, skipped = render_wrfouts_rust(
                    args.wrfout, products=rust_products, timeidx=timeidx,
                    outdir=args.out, size=size, heavy=args.heavy,
                    source_label=args.source_label, layout=args.layout,
                    overlays=args.overlays, annotate=args.annotate,
                    streamlines=args.streamlines)
            except (RuntimeError, ValueError) as exc:
                print("render: " + explain.render(
                    str(exc), explain=explain.explain_enabled(args),
                    command="gpuwm render"), file=sys.stderr)
                return 2
        else:
            written, failures, skipped = render_wrfouts(
                args.wrfout, products=products, timeidx=timeidx,
                outdir=args.out, dpi=args.dpi,
                source_label=args.source_label, layout=args.layout)
        for failure in failures:
            print(f"render FAIL: {failure}", file=sys.stderr)
        notice = skip_notice(skipped, wrote_any=bool(written))
        if notice is not None:
            print(explain.render(
                notice, explain=explain.explain_enabled(args),
                command="gpuwm render"), file=sys.stderr)
        print(f"render: {len(written)} file(s) -> {args.out}")
    finally:
        _publish_run_dir(args)
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
             "candidates per file) or the matplotlib workaround; 'auto' "
             "(default) uses rust whenever its binary is built and "
             "probes as runnable, and REFUSES otherwise rather than "
             "drawing weather fields with matplotlib -- 'matplotlib' "
             "asks for that workaround by name and announces itself")
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
        help="where the PNGs go (default out/render).  Each render "
             "claims its own timestamped run folder under it, so two "
             "renders never overwrite each other; point it at an "
             "existing run-... folder to draw into that one")
    parser.add_argument(
        "--layout", choices=render_layout.LAYOUTS,
        default=render_layout.DEFAULT_LAYOUT,
        help="how the PNGs are arranged inside this render's run "
             "folder: 'nested' (default) files each picture at "
             f"{render_layout.describe('<run folder>', sep='/')}, "
             "so a run's thousands of frames are separated by nest, by "
             "chart and by day and a script can predict a path without "
             "globbing; 'flat' writes every picture directly into the "
             "run folder, which is what releases up to 2.4.1 did and is "
             "kept only for consumers still written against it (with "
             "--run-stamp off it is the v2.4.1 tree exactly)")
    run_stamp.add_argument(parser, artifacts="PNGs")
    parser.add_argument(
        "--dpi", type=int, default=150, metavar="N",
        help="PNG resolution, matplotlib engine (default 150)")
    parser.add_argument(
        "--size", default="1200x900", metavar="WxH",
        help="output pixels, rust engine (default 1200x900)")
    parser.add_argument(
        # None, not the string: the default is resolved at render time
        # by `default_source_label()`, which asks provenance which tree
        # is executing.  Computing it here would put a git subprocess in
        # `gpuwm --help`.
        "--source-label", default=None, metavar="TEXT",
        help="model/provenance label stamped on every plot (default "
             f"'{DEFAULT_SOURCE_LABEL} <the executing version>'); set it "
             "when rendering wrfout files this model did not produce, so "
             "the sheet does not claim them")
    parser.add_argument(
        "--heavy", action="store_true",
        help="rust engine: also compute the heavy ECAPE product family "
             "at import (SBECAPE/SBNCAPE/SBECIN, ECAPE SCP/EHI/...; "
             "adds substantial per-frame import time)")
    wind = parser.add_mutually_exclusive_group()
    wind.add_argument(
        "--streamlines", dest="streamlines", action="store_true",
        default=None,
        help="rust engine: draw the wind as STREAMLINES instead of "
             "barbs on every product that carries a wind layer.  "
             "Without either flag the engine keeps its automatic "
             "choice (streamlines on curvilinear and projected grids, "
             "barbs on plain lat/lon), and the RUSTWX_WIND_STREAMLINES "
             "environment variable still works; this flag and --barbs "
             "outrank it")
    wind.add_argument(
        "--barbs", dest="streamlines", action="store_false",
        default=None,
        help="rust engine: draw the wind as BARBS, overruling both the "
             "automatic choice and any inherited "
             "RUSTWX_WIND_STREAMLINES")
    parser.add_argument(
        "--overlays", type=Path, metavar="FILE.json",
        help="rust engine: draw map overlays given in geographic DEGREES "
             "on every panel -- lines, closed boxes, markers, labels and "
             "range rings.  This is the seam a boundary-zone frame, tile "
             "seams, storm-report markers and radar sites needed; the "
             "schema is documented in "
             "tools/rustwx/crates/rustwx-products/src/"
             "geographic_overlays.rs.  Omitted, the renderer runs no "
             "overlay code and the PNGs are byte-identical")
    parser.add_argument(
        "--annotate", type=Path, metavar="FILE.json",
        help="rust engine: override the panel title and the three "
             "subtitle slots (title, title_suffix, subtitle_left, "
             "subtitle_center, subtitle_right).  A short badge belongs "
             "in the centre slot; anything sentence-length belongs on "
             "the left, which owns the row's width")
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
           "WRF_PACKAGE_REQUIREMENT", "default_source_label",
           "domain_token", "list_products_main",
           "matplotlib_workaround_notice",
           "parse_products", "parse_products_rust", "parse_size",
           "parse_timeidx", "plot_context", "register_cli", "render_main",
           "missing_basemap_notice", "missing_declared_inputs",
           "render_series_rust", "render_wrfouts", "render_wrfouts_rust",
           "require_renderer", "resolution_token", "skip_notice",
           "spacing_label"]
