"""Locate and drive the vendored Rusty Weather renderer (``rw_wrfbatch``).

``gpuwm render --engine rust`` renders wrfout files through the vendored
``tools/rustwx`` workspace -- the production Rusty Weather batch
renderer, the same engine (and plot quality) as the campaign's paired
CPU-vs-GPU product sheets.  Like the GRIB bridges, the pip wheel ships
no compiled Rust: the binary is built once from the vendored workspace
(``cargo build --release --locked --offline``) and then *pointed at*,
with the same resolution ladder as :mod:`gpuwm.bridges`:

1. the ``GPUWM_RW_WRFBATCH`` environment variable naming the built file
   (a missing file it names is a hard error, never silently skipped);
2. a source checkout's ``tools/rustwx/target/{release,debug}``;
3. ``<root>/libexec/bridges`` beside the package;
4. ``~/.gpuwm/bridges``.

The renderer draws coastlines/state/county basemaps from the vendored
Natural Earth + US Census assets in ``tools/rustwx/assets/basemap``.
When the binary runs from a checkout it finds them by walking its own
ancestors; for a relocated binary :func:`renderer_env` pins
``RUSTWX_BASEMAP_DIR`` to the checkout assets when they exist, and an
explicit ``RUSTWX_BASEMAP_DIR``/``RUSTWX_ASSETS_DIR`` in the caller's
environment always wins.

Nothing here runs cargo.  Resolution has one side effect and one
only: an artifact found in ``~/.gpuwm/bridges`` that is not the one this
release pinned is re-fetched before it is handed to a door
(:func:`gpuwm.bridges.require_release_pin`).  ``gpuwm doctor`` resolves
inside :func:`gpuwm.bridges.inspection_only`, where there is no side
effect at all, so the report still says what the estate IS.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from gpuwm import bridges
from gpuwm.bridges import (RUSTWX_CRATE_RELATIVE, artifact_remedy,
                           cargo_build_one_liner, default_bridge_dir,
                           executable_name, packaged_bridge_dir)

#: Environment variable naming a prebuilt renderer executable.
RENDERER_ENV = "GPUWM_RW_WRFBATCH"

#: Executable base name of the vendored batch renderer.
RENDERER_NAME = "rw_wrfbatch"

#: The one-liner that builds the renderer, from a checkout root.
#: Shell-correct for this platform: Windows PowerShell 5.1 cannot
#: parse ``&&``.
CARGO_BUILD_HINT = cargo_build_one_liner(RUSTWX_CRATE_RELATIVE)

#: The exact ``rw_wrfbatch --abi`` line this wrapper was written against.
#:
#: The renderer was the only bundled binary with no contract check.  Its
#: two workspace siblings pin one (:data:`gpuwm.rustwx_fetch
#: .FETCH_ABI_MARKER`, :data:`gpuwm.obs.nexrad.NEXRAD_ABI_MARKER`) and
#: the five GRIB decoders pin a static one
#: (:data:`gpuwm.bridges.BRIDGE_ABI_MARKERS`); ``rw_wrfbatch`` was asked
#: only whether it started.  Two builds four megabytes and two days
#: apart -- md5 72c739e8... and 894cc90f... -- both printed the usage
#: line, both were reported ``verified``, and neither was from this
#: tree.  A renderer that verifies on "it launches" is a renderer that
#: can draw a whole product set from a foreign engine (task #106).
#:
#: The literal spells the contract out rather than carrying a version
#: number, exactly as ``BRIDGE_ABI_MARKERS`` requires: the ``PRODUCT``
#: /``CATALOG`` row grammar :func:`list_products` parses, the
#: ``RENDERED``/``SKIPPED``/``FAILED`` events :func:`run_renderer`
#: parses, and the generic ``var:`` vocabulary whose absence from a
#: stale build is what #106 was reported as.  Changing any of those
#: changes this string, and every binary predating the change answers
#: ``unknown option --abi`` instead of the old grammar.
RENDERER_ABI_MARKER = (
    "gpuwm-rw-wrfbatch-catalog-v1\tPRODUCT\tslug\tkind\tstatus\tdetail\t"
    "CATALOG\t"
    "gpuwm-rw-wrfbatch-events-v1\tRENDERED\tSKIPPED\tFAILED\t"
    "gpuwm-rw-wrfbatch-vocabulary-v1\tgeneric\tvar:\tselectable_slugs")

_PROBE_TIMEOUT_S = 20


def crate_dir() -> Path:
    """The vendored Rusty Weather workspace of a source checkout."""

    return Path(__file__).resolve().parent.parent / "tools" / "rustwx"


def basemap_dir() -> Path:
    """The vendored basemap assets (Natural Earth + US counties)."""

    return crate_dir() / "assets" / "basemap"


#: What to tell a caller whose install cannot read the basemap
#: shapefiles.  Spelled out here, once, beside the resolver for the
#: assets it reads, so every entry point says the same thing.
#:
#: Two halves, because a reader who installs only the package still
#: cannot draw: ``pyshp`` reads the geometry and the vendored assets
#: under :func:`basemap_dir` ARE the geometry, and those arrive in the
#: bundle ``gpuwm fetch-bridges`` stages.  Naming only the pip line
#: would send someone to a second failure one step later.
PYSHP_REMEDY = (
    "the map frame needs pyshp (it reads the Natural Earth and US Census "
    "shapefiles the basemap is drawn from); install it with "
    "`pip install pyshp>=2.3` or `pip install gpuwm[render]`, and run "
    "`gpuwm fetch-bridges` if the vendored basemap assets are not staged "
    "yet")


def pyshp_available() -> bool:
    """Whether the shapefile reader every basemap needs can be imported.

    Asked at a front door rather than left to the function-local ``import
    shapefile`` inside the renderers, for exactly the reason
    :func:`gpuwm.obs.dealias.scipy_available` exists: the two failures are
    not the same failure.

    The DA nowcast's render stage runs DEAD LAST -- after the survey, the
    fetch, the preparation, the free forecast and every DA cycle -- and
    reaching that import meant a bare ``ModuleNotFoundError: No module
    named 'shapefile'`` with no message at all, having destroyed the most
    work of any failure in the product.  Answering here costs a
    ``find_spec`` before the run starts.

    ``find_spec`` rather than a real import: this is asked on the hot path
    of a front door that may then not draw anything, and importing a
    module to learn whether it exists is a side effect a capability check
    should not have.
    """

    from importlib.util import find_spec

    try:
        return find_spec("shapefile") is not None
    except (ImportError, ValueError):     # pragma: no cover - broken install
        return False


def require_pyshp() -> None:
    """Import-time gate for a module whose whole job is drawing a map.

    The named refusal the three bare ``import shapefile`` call sites
    lacked.  ``ImportError`` keeps the class a caller would already be
    catching around an import, and the message carries the remedy.
    """

    if not pyshp_available():
        raise ImportError(PYSHP_REMEDY)


#: How many ancestors of the renderer executable's own directory the
#: renderer walks looking for ``assets/basemap``.  Mirrors
#: ``rustwx-render``'s ``basemap_root_candidates``; a build at
#: ``tools/rustwx/target/release/`` reaches the crate's assets at the
#: second ancestor, which is why a renderer built from a clone finds its
#: basemaps whatever directory it is launched from.
_RENDERER_EXE_ANCESTORS = 8

#: And how many ancestors of the working directory it walks.
_RENDERER_CWD_ANCESTORS = 6


def basemap_candidates(renderer: Path | None = None) -> tuple[Path, ...]:
    """Where the RENDERER looks for basemaps, in its own order.

    Not where gpuwm keeps them.  ``gpuwm doctor`` used to probe the
    single checkout path :func:`basemap_dir` and announce "NO basemap
    assets found" whenever it was absent -- which is every pip install,
    including ones where ``rw_wrfbatch`` was resolving the assets
    perfectly well from its own build directory.  A report that
    contradicts the artifact is worse than no report, so this mirrors
    ``rustwx-render``'s ``basemap_root_candidates`` instead:

    1. ``RUSTWX_BASEMAP_DIR``;
    2. ``RUSTWX_ASSETS_DIR/basemap``;
    3. ``assets/basemap`` and ``Resources/assets/basemap`` under each of
       the first eight ancestors of the executable's own directory;
    4. ``assets/basemap`` and ``basemap`` under each of the first six
       ancestors of the working directory;
    5. the crate's own ``assets/basemap`` -- the compile-time workspace
       root, which for this vendored crate is :func:`basemap_dir`.

    Duplicates are dropped, first occurrence winning, exactly as the
    renderer does it.
    """

    candidates: list[Path] = []

    def push(path: Path) -> None:
        if path not in candidates:
            candidates.append(path)

    override = os.environ.get("RUSTWX_BASEMAP_DIR")
    if override:
        push(Path(override))
    assets = os.environ.get("RUSTWX_ASSETS_DIR")
    if assets:
        push(Path(assets) / "basemap")

    if renderer is not None:
        parent = Path(renderer).resolve().parent
        for ancestor in (parent, *parent.parents)[:_RENDERER_EXE_ANCESTORS]:
            push(ancestor / "assets" / "basemap")
            push(ancestor / "Resources" / "assets" / "basemap")

    working = Path.cwd()
    for ancestor in (working, *working.parents)[:_RENDERER_CWD_ANCESTORS]:
        push(ancestor / "assets" / "basemap")
        push(ancestor / "basemap")

    push(basemap_dir())
    return tuple(candidates)


def cartopy_natural_earth_root() -> Path | None:
    """The cartopy shapefile cache, if this machine has one.

    Not part of :func:`basemap_candidates` -- the renderer consults this
    *after* those, per layer, inside its own loaders -- but it is real
    geography and it is why this bug survived a release.  A workstation
    that has ever run cartopy has this directory, so ``rw_wrfbatch``
    draws perfectly good coastlines there while the same binary on a
    clean machine draws none.  Anything that reports on basemap
    availability has to know about it or it reports a state the
    artifact does not have.

    Mirrors ``rustwx-render``'s ``cartopy_natural_earth_root``,
    including its ``USERPROFILE``-before-``HOME`` order.
    """

    home = os.environ.get("USERPROFILE") or os.environ.get("HOME")
    if not home:
        return None
    root = (Path(home) / ".local" / "share" / "cartopy" / "shapefiles"
            / "natural_earth")
    return root if root.is_dir() else None


def resolve_basemap_dir(renderer: Path | None = None) -> Path | None:
    """The first candidate that exists, or None if the renderer has none.

    The renderer resolves each asset SUBDIRECTORY independently, so a
    root that exists is evidence rather than proof; a root that exists
    nowhere is proof, and that is the only case worth warning about.
    """

    for candidate in basemap_candidates(renderer):
        if candidate.is_dir():
            return candidate
    return None


def renderer_candidates() -> tuple[Path, ...]:
    """Deterministic candidate paths for the renderer, best first."""

    filename = executable_name(RENDERER_NAME)
    candidates: list[Path] = []
    override = os.environ.get(RENDERER_ENV)
    if override:
        candidates.append(Path(override))
    root = Path(__file__).resolve().parent.parent
    candidates.extend((
        crate_dir() / "target" / "release" / filename,
        crate_dir() / "target" / "debug" / filename,
        root / "libexec" / "bridges" / filename,
        packaged_bridge_dir() / filename,
        default_bridge_dir() / filename,
    ))
    return tuple(candidates)


def find_renderer() -> Path | None:
    """First existing candidate, or None.

    An environment override that names a missing file is a hard error:
    explicit configuration must fail loudly, not fall through.

    Loudly AND with the exit named (1.8.8 refusal sweep).  The message
    used to end at the path, which leaves a reader who set the variable
    weeks ago -- or inherited it from a shell profile -- with a
    diagnosis and no instruction.  Three ways out, in the order they are
    likely to be wanted: point the variable somewhere real, drop it and
    take the vendored ladder, or ask for the fallback engine outright.
    """

    override = os.environ.get(RENDERER_ENV)
    for candidate in renderer_candidates():
        if candidate.is_file():
            return bridges.accept_resolved(candidate.resolve())
        if override and candidate == Path(override):
            raise FileNotFoundError(
                f"{RENDERER_ENV} names a missing file: {candidate}.  "
                f"Point it at a built rw_wrfbatch binary, unset "
                f"{RENDERER_ENV} to use the vendored resolution ladder "
                f"(build it with: {CARGO_BUILD_HINT}), or pass "
                f"--engine matplotlib to draw with the fallback engine.")
    return None


def renderer_remedy() -> str:
    """The remedy for a missing renderer, true for THIS install.

    Delegates rather than repeating the shape a third time: this copy is
    what kept a ``<clone>`` placeholder and an "exact copy-pasteable"
    claim after the bridge copy stopped making either.
    """

    return artifact_remedy(
        env_var=RENDERER_ENV, filename=executable_name(RENDERER_NAME),
        subject="the rust render engine",
        crate_relative=RUSTWX_CRATE_RELATIVE,
        one_liner=CARGO_BUILD_HINT)


def renderer_env() -> dict[str, str]:
    """Subprocess environment for the renderer.

    An explicit ``RUSTWX_BASEMAP_DIR``/``RUSTWX_ASSETS_DIR`` is the
    user's to keep; otherwise the vendored checkout assets are pinned so
    a binary running from ``libexec`` or ``~/.gpuwm/bridges`` still
    draws its basemaps.
    """

    env = dict(os.environ)
    if "RUSTWX_BASEMAP_DIR" not in env and "RUSTWX_ASSETS_DIR" not in env:
        assets = basemap_dir()
        if assets.is_dir():
            env["RUSTWX_BASEMAP_DIR"] = str(assets)
    return env


def probe_renderer(path: Path) -> tuple[bool, str]:
    """``--help`` then ``--abi``: is this binary runnable, and is it ours?

    ``rw_wrfbatch --help`` prints its usage line and exits 0.  That
    observable separates a runnable executable from an empty, truncated,
    or wrong-platform file, which refuses to launch (OSError) or dies
    with an abnormal status and no usage text.

    Launching is necessary and was never sufficient.  ``--abi`` is the
    stale-build half, and it is the same handshake ``rw_fetch`` and
    ``rw_nexrad`` have always answered -- not a new mechanism, the
    existing one finally applied to the third bundled binary in this
    workspace.  A build whose catalog grammar, event words or generic
    ``var:`` vocabulary differ from :data:`RENDERER_ABI_MARKER` fails
    here, where a report says so, instead of at the product sheet where
    a reader counts plots and wonders.

    Two builds of ``rw_wrfbatch`` with different md5s both passed the
    ``--help``-only probe and both were reported ``verified``; that is
    the defect this closes, and the remedy is REBUILD, never re-point:
    the contract moved, so every binary older than it fails the same
    way.

    The header is read before the launch, as in every probe in this
    package: on Windows a corrupt image header can hang
    ``CreateProcess`` where no timeout reaches.  See
    :func:`gpuwm.bridges.native_executable_format`.
    """

    ok, evidence = bridges.launchable(path)
    if not ok:
        return False, f"{evidence} -- corrupt, stale, or built for " \
                      "another platform"
    try:
        with bridges.quiet_loader_errors():
            probe = subprocess.run(
                [str(path), "--help"], capture_output=True, text=True,
                errors="replace", timeout=_PROBE_TIMEOUT_S)
    except OSError as error:
        return False, f"exists but failed to execute: {error}"
    except subprocess.TimeoutExpired:
        return False, (f"probe invocation did not exit within "
                       f"{_PROBE_TIMEOUT_S} s")
    transcript = f"{probe.stdout or ''}{probe.stderr or ''}"
    if probe.returncode != 0 or "usage: rw_wrfbatch" not in transcript:
        return False, (f"probe --help exited {probe.returncode} without the "
                       "expected usage line -- corrupt, stale, or built for "
                       "another platform")
    try:
        with bridges.quiet_loader_errors():
            abi = subprocess.run(
                [str(path), "--abi"], capture_output=True, text=True,
                errors="replace", timeout=_PROBE_TIMEOUT_S)
    except OSError as error:
        return False, f"--abi did not run: {error}"
    except subprocess.TimeoutExpired:
        return False, (f"--abi did not exit within {_PROBE_TIMEOUT_S} s")
    observed = (abi.stdout or "").strip()
    if abi.returncode != 0 or observed != RENDERER_ABI_MARKER:
        # A build predating the handshake answers `unknown option --abi`
        # on exit 2, so name that case for what it is rather than
        # reporting an empty string against a long expected line.
        seen = (f"exit {abi.returncode} with no --abi line" if not observed
                else f"exit {abi.returncode}: {observed[:120]!r}")
        return False, (
            "launches, but --abi does not match the render contract this "
            f"gpuwm expects ({seen}) -- it is a build from another "
            "checkout, so its product catalog is not this tree's; REBUILD "
            f"it, do not re-point {RENDERER_ENV} at another copy: "
            f"{CARGO_BUILD_HINT}")
    return True, ("probe --help exited 0 with its usage line; --abi matches "
                  "the render contract")


def list_products(renderer: Path, wrfout: Path, *, store_root: Path,
                  heavy: bool = False
                  ) -> tuple[list[tuple[str, str, str, str]], str]:
    """One catalog listing for ``wrfout``: (rows, summary).

    Rows are ``(slug, kind, status, detail)`` exactly as the renderer's
    ``--list-products`` mode emits them (statuses: renderable,
    missing-fields, blocked, excluded -- there is no identity-gated
    status; every non-renderable row names fields or an honest lane
    reason); ``summary`` is its ``CATALOG ...`` tally line.  The import
    into ``store_root`` is the real one -- availability is proven
    against the stored fields, never guessed from filenames.
    """

    command = [
        str(renderer),
        "--store-root", str(store_root),
        # Required by the CLI contract but never written in list mode.
        "--out-dir", str(store_root),
        "--list-products",
    ]
    if heavy:
        command.append("--heavy")
    command.append(str(wrfout))
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, errors="replace",
            env=renderer_env())
    except OSError as error:
        raise RuntimeError(
            f"{wrfout}: renderer failed to launch: {error}") from error
    if result.returncode != 0:
        tail = [line for line in (result.stderr or "").splitlines()
                if line.strip()]
        raise RuntimeError(
            f"{wrfout}: {tail[-1] if tail else f'exit {result.returncode}'}")
    rows: list[tuple[str, str, str, str]] = []
    summary = ""
    for line in (result.stdout or "").splitlines():
        if line.startswith("PRODUCT\t"):
            parts = line.split("\t")
            if len(parts) == 5:
                rows.append((parts[1], parts[2], parts[3], parts[4]))
        elif line.startswith("CATALOG "):
            summary = line[len("CATALOG "):]
    if not rows:
        raise RuntimeError(
            f"{wrfout}: renderer produced no catalog rows")
    return rows, summary


def run_renderer(renderer: Path, wrfout: Path, *, store_root: Path,
                 out_dir: Path, products: str, frames: str,
                 width: int, height: int, heavy: bool = False,
                 source_label: str | None = None,
                 overlays: Path | None = None,
                 annotate: Path | None = None,
                 streamlines: bool | None = None,
                 ) -> tuple[list[Path], list[str],
                            list[tuple[str, str]]]:
    """Render one wrfout file into ``out_dir``; (written, failures, skipped).

    One invocation per file with its own store root, exactly like the
    campaign flow: a shared store would merge two wrfouts into one run.
    The renderer's event lines are relayed to stdout as they arrive is
    not attempted -- the run is short-lived and its transcript is small,
    so it is captured and parsed for RENDERED/SKIPPED/FAILED events
    instead.

    The engine has always drawn the third distinction itself: a product
    whose stored fields are not there is ``SKIPPED <slug> <reason>`` on
    stdout and is *not* counted in ``summary.failed``, so the process
    still exits 0.  This function used to read only two of the three
    lines, which made an honest skip invisible to every caller -- the
    reason a reader saw 53 images and no word about the 54th.  All three
    are read now, and only ``FAILED`` is a failure.
    """

    return run_renderer_series(
        renderer, (wrfout,), store_root=store_root, out_dir=out_dir,
        products=products, frames=frames, width=width, height=height,
        heavy=heavy, source_label=source_label, overlays=overlays,
        annotate=annotate, streamlines=streamlines)


def run_renderer_series(renderer: Path, wrfouts, *, store_root: Path,
                        out_dir: Path, products: str, frames: str,
                        width: int, height: int, heavy: bool = False,
                        source_label: str | None = None,
                        overlays: Path | None = None,
                        annotate: Path | None = None,
                        streamlines: bool | None = None,
                        ) -> tuple[list[Path], list[str],
                                   list[tuple[str, str]]]:
    """One invocation over a whole wrfout SERIES, into ONE store.

    ``rw_wrfbatch``'s CLI has always taken ``wrfout...``; what this adds
    is a Python caller that uses it.  :func:`run_renderer` renders one
    file per store because per-file isolation is the campaign
    convention, and that convention is exactly what a WINDOWED product
    cannot be drawn under: ``qpf_6h`` is a statement about two frames
    six hours apart ("F012 minus F006", in the engine's own catalog
    detail), so a store holding only F012 has nothing to difference
    against and the product is skipped.  The verification door's 6 h
    accumulation is precisely that shape -- two separate hourly wrfout
    files -- and it is why this exists rather than a fourth spelling of
    the same subprocess call.

    The store is the caller's to create and to remove; a series store is
    deliberately NOT per-file, because merging the frames into one run
    timeline is the whole point.

    Event parsing and the failure contract are :func:`run_renderer`'s,
    unchanged: RENDERED/SKIPPED on stdout, FAILED on stderr, and a
    nonzero exit with no FAILED line reported as one failure carrying
    the last stderr line.  Messages name the LAST file in the series --
    the frame whose valid time the panels carry.
    """

    inputs = [Path(item) for item in wrfouts]
    if not inputs:
        raise ValueError(
            "a render series needs at least one wrfout file; an empty "
            "series would launch the renderer with no input and read its "
            "usage line as a failure")
    subject = inputs[-1]
    command = [
        str(renderer),
        "--store-root", str(store_root),
        "--out-dir", str(out_dir),
        "--products", products,
        "--frames", frames,
        "--width", str(width),
        "--height", str(height),
    ]
    if heavy:
        command.append("--heavy")
    if source_label:
        command.extend(("--source-label", source_label))
    # The wind layer, when the caller asked for one.  ``None`` adds no
    # argument at all, so an invocation that never mentions the wind is
    # byte-identical to every earlier release and
    # ``RUSTWX_WIND_STREAMLINES`` keeps its existing meaning.
    if streamlines is not None:
        command.append("--streamlines" if streamlines else "--barbs")
    # 2.5.0: map overlays in geographic degrees and panel annotations.
    # Absent, the renderer runs no overlay code and the PNGs are
    # byte-identical to every earlier build -- which is what
    # ``tools/rustwx_render_regression_gate.py`` gates.
    if overlays is not None:
        command.extend(("--overlays", str(overlays)))
    if annotate is not None:
        command.extend(("--annotate", str(annotate)))
    command.extend(str(path) for path in inputs)
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, errors="replace",
            env=renderer_env())
    except OSError as error:
        return [], [f"{subject}: renderer failed to launch: {error}"], []
    written: list[Path] = []
    failures: list[str] = []
    skipped: list[tuple[str, str]] = []
    for line in (result.stdout or "").splitlines():
        if line.startswith("RENDERED "):
            _, _, rest = line.partition(" ")
            _, _, path = rest.partition(" ")
            if path:
                written.append(Path(path))
        elif line.startswith("SKIPPED "):
            slug, _, reason = line[len("SKIPPED "):].partition(" ")
            skipped.append(
                (slug, f"{subject}: {reason or 'no reason given'}"))
    for line in (result.stderr or "").splitlines():
        if line.startswith("FAILED "):
            failures.append(f"{subject}: {line[len('FAILED '):]}")
    if result.returncode != 0:
        tail = [line for line in (result.stderr or "").splitlines()
                if line.strip()]
        detail = tail[-1] if tail else f"exit {result.returncode}"
        if not failures:
            failures.append(f"{subject}: {detail}")
    return written, failures, skipped


__all__ = [
    "CARGO_BUILD_HINT", "RENDERER_ABI_MARKER", "RENDERER_ENV",
    "RENDERER_NAME", "basemap_dir",
    "basemap_candidates", "crate_dir", "find_renderer", "list_products",
    "probe_renderer", "renderer_candidates", "renderer_env",
    "renderer_remedy", "resolve_basemap_dir", "run_renderer",
    "run_renderer_series",
]
