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
import re
import sys
from pathlib import Path

import numpy as np

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


def _domain_tag(path: Path) -> str:
    """``d0X`` for output names: GRID_ID attribute, else the filename."""
    try:
        import netCDF4
        with netCDF4.Dataset(path) as ds:
            if "GRID_ID" in ds.ncattrs():
                return f"d{int(ds.getncattr('GRID_ID')):02d}"
    except Exception:
        pass
    match = _WRFOUT_DOMAIN.search(path.name)
    return match.group(1) if match else "d01"


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


def _render_refl(wf, timeidx, lat, lon, *, plt, wrf, stamp, domain,
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
                  f"{domain} valid {stamp}",
            cbar_label="composite dBZ", out_png=out_png, dpi=dpi,
            ticks=_NWS_LEVELS[::2])
    plt.close(fig)


def _render_t2(wf, timeidx, lat, lon, *, plt, wrf, stamp, domain,
               out_png: Path, dpi: int) -> None:
    t2 = np.asarray(wrf.getvar(wf, "t2", timeidx=timeidx, units="degC"))
    fig, axis = _figure(plt, lat, lon)
    # Spectral temperature scale (meteorological convention), symmetric
    # ticks left to matplotlib; the data range sets the limits.
    mesh = axis.pcolormesh(lon, lat, t2, cmap="RdYlBu_r", shading="auto")
    _finish(fig, axis, mesh,
            title=f"2 m temperature\n{domain} valid {stamp}",
            cbar_label="deg C", out_png=out_png, dpi=dpi)
    plt.close(fig)


def _render_wind10(wf, timeidx, lat, lon, *, plt, wrf, stamp, domain,
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
            title=f"10 m wind ({rotation} barbs)\n{domain} valid {stamp}",
            cbar_label=f"10 m wind speed ({speed_units})",
            out_png=out_png, dpi=dpi)
    plt.close(fig)


def _render_precip(wf, timeidx, lat, lon, *, plt, wrf, stamp, domain,
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
                  f"{domain} valid {stamp}",
            cbar_label="mm since simulation start", out_png=out_png,
            dpi=dpi, ticks=_PRECIP_LEVELS)
    plt.close(fig)


#: Product registry: CLI name -> renderer.  ``all`` expands to this order.
PRODUCTS = {
    "refl": _render_refl,
    "t2": _render_t2,
    "wind10": _render_wind10,
    "precip": _render_precip,
}

#: The four shared product names, as the rust engine's catalog slugs.
#: ``wind10`` maps to the classic MSLP + 10 m wind-barb chart -- the
#: catalog's standalone surface-wind product.  Any other token is passed
#: through as a raw catalog slug, which the renderer validates strictly.
RUST_PRODUCT_ALIASES = {
    "refl": "composite_reflectivity",
    "t2": "2m_temperature",
    "wind10": "mslp_10m_winds",
    "precip": "total_qpf",
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
                   dpi: int = 150) -> tuple[list[Path], list[str]]:
    """Render every requested product/frame; return (written, failures).

    A missing input variable fails that one product/frame with a recorded
    message and the remaining work continues, exactly like the matched-
    figures tool's per-frame skip -- except nothing is silent: every
    failure is returned for the CLI to print and reflect in its exit code.
    """
    wrf = _import_wrf()
    plt = _pyplot()
    written: list[Path] = []
    failures: list[str] = []
    for path in (Path(p) for p in paths):
        try:
            wrffile = wrf.WrfFile(str(path))
            stamps = list(wrffile.times())
        except Exception as exc:
            failures.append(f"{path}: unreadable wrfout ({exc})")
            continue
        domain = _domain_tag(path)
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
            for product in products:
                out_png = outdir / (
                    f"{product}_{domain}_{_stamp_for_filename(stamp)}.png")
                try:
                    PRODUCTS[product](
                        wrffile, index, lat, lon, plt=plt, wrf=wrf,
                        stamp=stamp, domain=domain, out_png=out_png,
                        dpi=dpi)
                except Exception as exc:
                    failures.append(f"{path}[{index}] {product}: {exc}")
                    continue
                written.append(out_png)
                print(f"render: {out_png}")
    return written, failures


#: The matplotlib engine's per-product wrfout variable needs, for the
#: --list-products availability report (rendering itself goes through
#: wrf.getvar as ever; this is presence, honestly labeled as such).
_MPL_PRODUCT_NEEDS = {
    "refl": ("REFL_10CM",),
    "t2": ("T2",),
    "wind10": ("U10", "V10"),
    "precip": ("RAINC|RAINNC",),
}


def _list_products_matplotlib(path: Path) -> list[tuple[str, str, str, str]]:
    """(product, kind, status, detail) rows for the matplotlib catalog."""

    import netCDF4

    with netCDF4.Dataset(path) as ds:
        present = set(ds.variables)
    rows = []
    for product, needs in _MPL_PRODUCT_NEEDS.items():
        missing = []
        for need in needs:
            options = need.split("|")
            if not any(option in present for option in options):
                missing.append(" or ".join(options))
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


def render_wrfouts_rust(paths, *, products: str, timeidx: int | None,
                        outdir: Path, size: tuple[int, int],
                        heavy: bool = False) -> tuple[list[Path],
                                                      list[str]]:
    """Render via the vendored Rusty Weather engine; (written, failures).

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
    for path in (Path(p) for p in paths):
        import tempfile
        store = Path(tempfile.mkdtemp(prefix=".rwstore-", dir=outdir))
        try:
            file_written, file_failures = rustwx.run_renderer(
                renderer, path, store_root=store, out_dir=outdir,
                products=products, frames=frames, width=width,
                height=height, heavy=heavy)
        finally:
            _remove_scratch_store(store)
        written.extend(file_written)
        failures.extend(file_failures)
        for png in file_written:
            print(f"render: {png}")
    return written, failures


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
    if not args.wrfout:
        print("render: at least one WRFOUT file is required "
              "(or --pair A_DIR B_DIR)", file=sys.stderr)
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
    if args.list_products:
        return list_products_main(args, engine)
    print(f"render: engine {engine} ({why})")
    if engine == "rust":
        try:
            written, failures = render_wrfouts_rust(
                args.wrfout, products=rust_products, timeidx=timeidx,
                outdir=args.out, size=size, heavy=args.heavy)
        except RuntimeError as exc:
            print(f"render: {exc}", file=sys.stderr)
            return 2
    else:
        written, failures = render_wrfouts(
            args.wrfout, products=products, timeidx=timeidx,
            outdir=args.out, dpi=args.dpi)
    for failure in failures:
        print(f"render FAIL: {failure}", file=sys.stderr)
    print(f"render: {len(written)} file(s) -> {args.out}")
    return 0 if written and not failures else 1


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "render",
        help="render forecast product PNGs from wrfout files via the "
             "wrf package (composite reflectivity, 2 m temperature, "
             "10 m wind, accumulated precipitation)")
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


__all__ = ["PRODUCTS", "RUST_PRODUCT_ALIASES", "WRF_PACKAGE_REQUIREMENT",
           "list_products_main", "parse_products", "parse_products_rust",
           "parse_size", "parse_timeidx", "register_cli", "render_main",
           "render_wrfouts", "render_wrfouts_rust"]
