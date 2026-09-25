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
#: ``RENDERED``/``SKIPPED``/``FAILED`` events and the ``SECTIONFILL``
#: line :func:`run_renderer` parses, and the generic ``var:`` and
#: vertical-section ``xsec:``
#: vocabularies whose absence from a stale build is what #106 was
#: reported as.  Changing any of those changes this string, and every
#: binary predating the change answers ``unknown option --abi`` instead
#: of the old grammar.
RENDERER_ABI_MARKER = (
    "gpuwm-rw-wrfbatch-catalog-v1\tPRODUCT\tslug\tkind\tstatus\tdetail\t"
    "code\tCATALOG\t"
    "gpuwm-rw-wrfbatch-requirements-v1\tNEEDS\tslug\tselector\tPLANNED\t"
    "store_field\t"
    "gpuwm-rw-wrfbatch-events-v1\tRENDERED\tSKIPPED\tFAILED\t"
    "gpuwm-rw-wrfbatch-sections-v1\tSECTIONFILL\tslug\tlo\thi\tabsence\t"
    "rule\t"
    "gpuwm-rw-wrfbatch-vocabulary-v1\tgeneric\tvar:\txsec:\tmesh:\t"
    "meshdiff:\tselectable_slugs")

#: The generic product families, read OUT of the pinned marker rather
#: than listed again beside it.  The marker is the contract a built
#: renderer answers with, so a family added there is advertised here in
#: the same commit and a door cannot end up offering a vocabulary the
#: engine does not have.
GENERIC_FAMILIES: tuple[str, ...] = tuple(
    field for field in RENDERER_ABI_MARKER.split("\t")
    if field.endswith(":"))

#: The vertical-section family's prefix, spelled once.
SECTION_PREFIX = "xsec:"

#: The event word carrying the range one vertical cut's fill was drawn
#: over.  A section's colour bar is fitted per frame at BOTH ends -- to
#: the rung its own signal reaches and, since a fill that would use less
#: than half a zero-anchored bar takes its own minimum instead, to the
#: air the cut holds -- so two pictures of one line an hour apart can be
#: drawn on two different bars and neither picture says so.  This is
#: what puts it in the receipt.
SECTION_FILL_EVENT = "SECTIONFILL"

#: The one generic family a STORE can be enumerated for.
#:
#: The generic catalog is, by construction, every stored 2-D variable
#: that no named product already draws -- so for this prefix, and only
#: this prefix, "there is no row" is PROOF that the store carries no
#: such variable, and the renderer answers a request for one with
#: ``stored 2-D variable "X" does not exist`` and fails the whole
#: invocation.  ``xsec:`` is cut from the wrfout files by the section
#: lane and ``mesh:``/``meshdiff:`` read a mesh, so none of those is a
#: store product and none can be decided from a store listing.
GENERIC_VAR_PREFIX = "var:"

#: The group keywords that name no product and carry no section term.
#:
#: Public because a door has to tell "the whole catalog, whatever it
#: holds" apart from "these named slugs".  The engine expands a group
#: itself and leaves out what it cannot draw, so a group token carries
#: no promise a caller can check and must pass through untouched; a
#: NAMED slug is a promise, and a door that forwards one the catalog
#: has just refused gets the renderer's whole-invocation failure.
GROUP_KEYWORDS = frozenset(
    {"all", "direct", "derived", "heavy", "windowed"})

#: The spelling this module used before the set was public.
_GROUP_KEYWORDS = GROUP_KEYWORDS

#: The two codes a windowed row carries when the store's TIME AXIS is
#: what excluded it: a history that is not on whole hours, or one whose
#: frames sit on an exact-time ordinal axis.  No door branches on them
#: -- :func:`catalog_verdict` drops every non-renderable row alike --
#: and they are here as the engine's own vocabulary, named in
#: ``docs/render-output-layout.md`` as the family a forwarded product
#: FAILS a whole invocation on, and pinned by the catalog contract
#: tests.  A reason may be reworded at any time; these may not.
WINDOW_AXIS_CODES = ("windowed-needs-whole-hour-frames",
                     "windowed-ordinal-axis")

#: How the generic emitter opens the row it prints for a stored
#: variable whose name no request can carry.  It is the one
#: ``GENERIC_EXCLUDED`` form that is NOT a renderable variable, so
#: the listing skips it rather than folding it in.
_GENERIC_NAME_REFUSED = "name is not request-safe"

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
    # A platform wheel resolves its bundled executable before the copy in
    # ~/.gpuwm/bridges. Its ancestors therefore miss the map assets installed
    # by fetch-bridges. renderer_env passes this staged root to that executable;
    # diagnostics must report the same wrapper-level fallback.
    if "RUSTWX_BASEMAP_DIR" not in os.environ and "RUSTWX_ASSETS_DIR" not in os.environ:
        staged = default_bridge_dir() / "assets" / "basemap"
        if staged.is_dir():
            return staged
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
    user's to keep; otherwise the vendored checkout assets, or the assets
    staged by fetch-bridges, are pinned so the platform wheel's preferred
    ``libexec`` binary still draws its basemaps.
    """

    env = dict(os.environ)
    if "RUSTWX_BASEMAP_DIR" not in env and "RUSTWX_ASSETS_DIR" not in env:
        assets = basemap_dir()
        if not assets.is_dir():
            assets = default_bridge_dir() / "assets" / "basemap"
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
    status; every non-renderable row names fields or a stated lane
    reason); ``summary`` is its ``CATALOG ...`` tally line.  The import
    into ``store_root`` is the real one -- availability is proven
    against the stored fields, never guessed from filenames.

    Four fields, deliberately: this is what every consumer in the tree
    already unpacks.  :func:`catalog_rows` is the same listing with the
    machine code beside each row, for a consumer that has to DECIDE
    something rather than print it.
    """

    return list_products_series(renderer, (wrfout,), store_root=store_root, heavy=heavy)


def catalog_rows(renderer: Path, wrfouts, *, store_root: Path,
                 heavy: bool = False
                 ) -> tuple[list[tuple[str, str, str, str, str]], str]:
    """The same listing, with each row's machine code: (rows, summary).

    ``(slug, kind, status, detail, code)``.  The detail is prose and is
    what a reader sees; the code is a stable spelling of WHY the row has
    the status it has, and it is what a consumer matches on.  Matching
    on the prose is a copy of the engine's sentences kept in Python,
    which stops matching the moment the engine rewords one -- at which
    point the excluded slug is forwarded and the whole render fails.

    A renderer built before the code column answers rows of five fields
    and the code comes back empty, which every helper below treats as
    "fall back to the prose".
    """

    return _catalog_listing(renderer, wrfouts, store_root=store_root, heavy=heavy)


def list_products_series(renderer: Path, wrfouts, *, store_root: Path,
                         heavy: bool = False
                         ) -> tuple[list[tuple[str, str, str, str]], str]:
    """Ask native availability after importing the complete selected timeline."""
    rows, summary = _catalog_listing(renderer, wrfouts, store_root=store_root,
                                     heavy=heavy)
    return [row[:4] for row in rows], summary


def _catalog_listing(renderer: Path, wrfouts, *, store_root: Path,
                     heavy: bool = False
                     ) -> tuple[list[tuple[str, str, str, str, str]], str]:
    """One parse of the renderer's catalog mode, for both accessors."""
    inputs = [Path(path) for path in wrfouts]
    if not inputs:
        raise ValueError("a product availability series needs history files")
    wrfout = inputs[-1]
    command = [
        str(renderer),
        "--store-root", str(store_root),
        # Required by the CLI contract but never written in list mode.
        "--out-dir", str(store_root),
        "--list-products",
    ]
    if heavy:
        command.append("--heavy")
    command.extend(str(path) for path in inputs)
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
    rows: list[tuple[str, str, str, str, str]] = []
    summary = ""
    for line in (result.stdout or "").splitlines():
        if line.startswith("PRODUCT\t"):
            # FIVE OR MORE, never exactly five: the row grew a machine
            # code, and a parser that demanded the old width dropped
            # every row of a current build and then reported that the
            # renderer produced no catalog at all.
            parts = line.split("\t")
            if len(parts) >= 5:
                rows.append((parts[1], parts[2], parts[3], parts[4],
                             parts[5] if len(parts) > 5 else ""))
        elif line.startswith("CATALOG "):
            summary = line[len("CATALOG "):]
    if not rows:
        raise RuntimeError(
            f"{wrfout}: renderer produced no catalog rows")
    # The store's DEDUPED generic variables, which the listing reports on
    # stderr: a variable the store carries whose grid a named product
    # already draws, so the catalog drops the duplicate `var:` row.  They
    # are folded in as renderable rows because the question a consumer
    # asks of a `var:` token is "does this store carry the variable?" --
    # and for these the answer is yes.  Without them, a listing's silence
    # about a present-but-deduped variable reads as absence, and a door
    # that drops on absence would refuse a spelling this build accepts.
    #
    # ONLY the deduped form.  The emitter prints this row for two
    # OPPOSITE outcomes (`rusty-weather/src/batch_render.rs`): the
    # deduped one, which is drawable, and a stored variable whose NAME
    # no request can carry, printed escape_debug'd because such a name
    # holds commas, control characters or edge whitespace.  Folding the
    # second in would make `gpuwm render --list-products` print it as
    # renderable, which is the opposite of what the row says.  No real
    # request can match that name, so the skip refuses nothing; the
    # listing simply stops claiming a spelling it cannot accept.
    for line in (result.stderr or "").splitlines():
        if not line.startswith("GENERIC_EXCLUDED\t"):
            continue
        parts = line.split("\t")
        if len(parts) < 2 or not parts[1]:
            continue
        reason = parts[2] if len(parts) > 2 else ""
        if reason.startswith(_GENERIC_NAME_REFUSED):
            continue
        rows.append((f"{GENERIC_VAR_PREFIX}{parts[1]}", "generic",
                     "renderable",
                     reason or "stored variable already drawn by a named "
                     "product",
                     "generic-deduped"))
    return rows, summary


def catalog_code(row) -> str:
    """One row's machine code, or ``""`` for a build that emits none."""

    return row[4] if len(row) >= 5 else ""


def catalog_verdict(rows, requested) -> tuple[str, list[tuple[str, str]]]:
    """``(the spec that can be drawn, the named skips)`` for one request.

    The availability listing has already been paid for by the time a
    door has rows in hand; this reads them.  Every requested slug whose
    row is renderable stays in the spec, in the order it was asked for;
    every requested slug whose row is anything else comes back paired
    with the engine's OWN detail, verbatim, so the reader gets the
    engine's reason rather than a Python paraphrase of it.

    A token with no row at all passes through untouched: the group
    keywords (``all``, ``direct`` ...), the section and mesh families
    and any slug this build knows and this listing did not mention.  The
    engine is the authority on an unknown slug, and guessing here would
    refuse a spelling this build accepts.

    :data:`GENERIC_VAR_PREFIX` is the ONE exception, and it is not a
    guess.  The generic catalog enumerates the store: every stored 2-D
    variable appears, as its own row or as a deduped one folded in by
    the listing.  So a ``var:`` token with no row is PROVEN absent from
    this store, and forwarding it gets ``stored 2-D variable "X" does
    not exist`` and a failed invocation -- which is how a shipped
    preset naming two variables in wrfout spelling rather than store
    spelling discarded a whole 13-frame series.

    That proof needs the enumeration to have RUN, so it is taken from
    the listing rather than assumed: the exception applies only when
    this listing carried a generic row at all.  A listing with none has
    said nothing about the store's variables -- a build without the
    generic enumeration is one way to get one -- and reading its silence
    as absence would drop every ``var:`` request a user made, by guess,
    with the engine never asked.  Those are forwarded unchanged and the
    renderer decides, the same way :func:`catalog_rows` treats a build
    whose rows carry no machine code.

    An empty spec is the caller's signal to refuse before launching
    rather than to run an empty render.
    """

    wanted = ([token.strip() for token in requested.split(",")]
              if isinstance(requested, str)
              else [str(token).strip() for token in requested])
    wanted = [token for token in wanted if token]
    status = {row[0]: (row[2], row[3]) for row in rows}
    # Whether this listing enumerated the store's 2-D variables at all,
    # which is what makes an absent row a reading rather than a silence.
    enumerated = any(row[1] == "generic"
                     or row[0].startswith(GENERIC_VAR_PREFIX)
                     for row in rows)
    available, excluded = [], []
    for token in dict.fromkeys(wanted):
        row = status.get(token)
        if row is None:
            if enumerated and token.startswith(GENERIC_VAR_PREFIX):
                excluded.append((
                    token,
                    "no stored 2-D variable "
                    f"{token[len(GENERIC_VAR_PREFIX):]!r} in this store; "
                    "gpuwm render --list-products names the ones there are"))
                continue
            available.append(token)
            continue
        if row[0] == "renderable":
            available.append(token)
            continue
        excluded.append((token, row[1]))
    return ",".join(available), excluded


def split_section_spec(products: str) -> tuple[str, list[str]]:
    """``(the store spec, the xsec: terms)`` of one --products spelling.

    The engine's own split, mirrored: ``rw-wrfbatch/src/section.rs``
    ``split_product_spec``.  A level list inside a section term is
    comma-separated too (``xsec:wa=1,2,5@5``), so a purely numeric token
    following a section term whose last term opened a level list is that
    list's continuation and not a product -- no product slug is only
    digits, a sign, a point and an ``@``.  A group keyword names no
    section at all.

    It exists so a door can answer "does this request need a section
    line?" without a renderer and without a wrfout, which is what a plan
    review has.
    """

    trimmed = (products or "").strip()
    if trimmed.lower() in _GROUP_KEYWORDS:
        return trimmed, []
    tokens: list[str] = []
    for token in (part.strip() for part in trimmed.split(",")):
        if not token:
            continue
        if (tokens and tokens[-1].startswith(SECTION_PREFIX)
                and _level_list_open(tokens[-1]) and _is_level_token(token)):
            tokens[-1] = f"{tokens[-1]},{token}"
            continue
        tokens.append(token)
    store = [token for token in tokens if not token.startswith(SECTION_PREFIX)]
    sections = [token for token in tokens if token.startswith(SECTION_PREFIX)]
    return ",".join(store), sections


def _level_list_open(token: str) -> bool:
    """True when the token's last term opened an ``=`` level list."""

    last = token.rsplit("/", 1)[-1]
    return "=" in last and "@" not in last.rsplit("=", 1)[-1]


def _is_level_token(token: str) -> bool:
    """True for a bare level (``5``, ``-2.5``, ``10@5``), never a slug."""

    body = token.split("@", 1)[0]
    if not body:
        return False
    return all(char.isdigit() or char in "+-." for char in body)


#: The height range ``rw_wrfbatch`` accepts for ``--section-top-km``
#: (``tools/rustwx/crates/rw-wrfbatch/src/main.rs``), and the ceiling it
#: uses when the caller names none.
SECTION_TOP_KM_RANGE = (1.0, 40.0)
SECTION_TOP_KM_DEFAULT = 14.0


def section_top_problem(top_km) -> str | None:
    """The refusal for a section ceiling outside 1-40 km, or ``None``.

    The engine's own sentence, because the engine is the authority on
    its own grammar, and it fires at the door so a caller reads it
    before a render launches rather than after.

    This is a refusal about a VALUE, not about a request: a spec that
    names a cut with no line is not refused anywhere any more, it is
    dropped per product by :func:`drop_storeless_terms` and the rest of
    the request is drawn.
    """

    if top_km is None:
        return None
    low, high = SECTION_TOP_KM_RANGE
    try:
        value = float(top_km)
    except (TypeError, ValueError):
        return (f"--section-top-km '{top_km}' is not within "
                f"{low:g}-{high:g} km")
    if not (value == value and abs(value) != float("inf")
            and low <= value <= high):
        return (f"--section-top-km '{top_km}' is not within "
                f"{low:g}-{high:g} km")
    return None


#: The generic families drawn from a MESH file rather than from a
#: history frame, read out of the pinned vocabulary marker.
MESH_PREFIXES: tuple[str, ...] = tuple(
    family for family in GENERIC_FAMILIES if family.startswith("mesh"))


#: Why a mesh term cannot be drawn from history frames, per term.
_MESH_TERM_REASON = (
    "mesh products are drawn from a mesh file's cell boundaries "
    "(rw_wrfbatch --mesh-grid FILE.nc); a history frame carries cell "
    "centres and no polygons, and this render door passes no mesh file. "
    "Ask for the instantaneous or windowed product of the same field, "
    "which is drawn from the frames themselves, or draw the mesh panel "
    "with rw_wrfbatch --mesh-grid FILE.nc directly")

#: Why a section term cannot be drawn without a line, per term.
_SECTION_TERM_REASON = (
    "a vertical section is cut along a LINE and this invocation composes "
    "none. Add --section lat,lon,lat,lon or --section FILE.json to draw "
    "it, or drop the term")


def drop_storeless_terms(products: str, *, section=None
                         ) -> tuple[str, list[tuple[str, str]]]:
    """``(the spec a wrfout render can carry, the terms dropped)``.

    Three families are not store products and cannot be decided from a
    store listing, so :func:`catalog_verdict` never sees them: ``mesh:``
    and ``meshdiff:`` read a mesh file's cell boundaries, and ``xsec:``
    is cut along a section line.  A request that names one of them on a
    door with no mesh file and no line is answered here, with nothing
    opened and nothing launched.

    WHAT BREAKAGE THIS PREVENTS (gate law).  The renderer's own answer
    to such a term is PER INVOCATION, taken before a single picture is
    drawn (``rw-wrfbatch/src/main.rs``): a ``mesh:`` term with no
    ``--mesh-grid`` is a usage error at argument validation, a ``mesh:``
    term beside store or ``xsec:`` products is refused at the entry to
    the batch render because the two families read different inputs, and
    an ``xsec:`` term with no ``--section`` is refused before the store
    render starts.  So ONE such term forwarded from a door costs every
    other requested product its pictures: measured on the shipped wheel,
    a two-product series naming ``mesh:cell_area`` exits 1 with no
    pictures at all.

    The drop is therefore per PRODUCT, which the renderer will not do
    for itself: the term goes, the rest of the request is drawn, and the
    caller reports each dropped term with the sentence returned beside
    it.  A caller left with an empty spec has nothing to draw and
    refuses rather than launching an empty render, the same signal
    :func:`catalog_verdict` gives.

    ``section`` is the line the caller will pass to the renderer.  With
    one, an ``xsec:`` term is drawable here and is kept; the mesh
    families have no such door, because no door in this package passes
    ``--mesh-grid``.

    A request that drops nothing comes back byte for byte.  One that
    drops something is rebuilt from the engine's own tokenization
    (:func:`split_section_spec`, so a section term's comma-separated
    level list stays one term), which groups the surviving section
    terms after the store terms; the renderer splits the spec into those
    same two lists before it draws, so the order between families is not
    a fact anything downstream reads.
    """

    store_spec, sections = split_section_spec(products)
    kept: list[str] = []
    dropped: list[tuple[str, str]] = []
    for term in (token.strip() for token in store_spec.split(",")):
        if not term:
            continue
        if term.lower().startswith(MESH_PREFIXES):
            dropped.append((term, _MESH_TERM_REASON))
        else:
            kept.append(term)
    drawable_section = section is not None and str(section).strip() != ""
    for term in sections:
        if drawable_section:
            kept.append(term)
        else:
            dropped.append((term, _SECTION_TERM_REASON))
    if not dropped:
        return products, []
    return ",".join(kept), dropped


def parse_section_fill(line: str) -> dict | None:
    """One ``SECTIONFILL`` line as a receipt row, or ``None``.

    ``SECTIONFILL <slug> lo=<v> hi=<v> absence=<0|1> rule=<token>``.  A
    line whose numbers do not parse is dropped rather than raised on: a
    receipt is metadata beside a picture that was drawn, and a render
    that succeeded is not failed over a field a later engine spells
    differently.
    """

    if not line.startswith(SECTION_FILL_EVENT + " "):
        return None
    slug, _, rest = line[len(SECTION_FILL_EVENT) + 1:].partition(" ")
    fields = {}
    for token in rest.split():
        key, sep, value = token.partition("=")
        if sep:
            fields[key] = value
    try:
        low = float(fields["lo"])
        high = float(fields["hi"])
    except (KeyError, ValueError):
        return None
    if not slug or low != low or high != high:
        return None
    return {"family": slug, "lo": low, "hi": high,
            "absence": fields.get("absence") == "1",
            "rule": fields.get("rule", "")}


def run_renderer(renderer: Path, wrfout: Path, *, store_root: Path,
                 out_dir: Path, products: str, frames: str,
                 width: int, height: int, heavy: bool = False,
                 source_label: str | None = None,
                 overlays: Path | None = None,
                 annotate: Path | None = None,
                 streamlines: bool | None = None,
                 theme: str | None = None,
                 section: str | None = None,
                 isotherms: str | None = None,
                 section_across_km: float | None = None,
                 section_size: tuple[int, int] | None = None,
                 section_top_km: float | None = None,
                 fills: list | None = None,
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
    lines, which made an accurate skip invisible to every caller -- the
    reason a reader saw 53 images and no word about the 54th.  All three
    are read now, and only ``FAILED`` is a failure.

    ``fills``, when a list is given, collects one
    :func:`parse_section_fill` row per vertical cut drawn: the range its
    colour bar spans and the rule that set it.  It is an out-parameter
    rather than a fourth return value because every caller of these two
    functions already unpacks three.
    """

    return run_renderer_series(
        renderer, (wrfout,), store_root=store_root, out_dir=out_dir,
        products=products, frames=frames, width=width, height=height,
        heavy=heavy, source_label=source_label, overlays=overlays,
        annotate=annotate, streamlines=streamlines, theme=theme,
        section=section, isotherms=isotherms,
        section_across_km=section_across_km, section_size=section_size,
        section_top_km=section_top_km, fills=fills)


def run_renderer_series(renderer: Path, wrfouts, *, store_root: Path,
                        out_dir: Path, products: str, frames: str,
                        width: int, height: int, heavy: bool = False,
                        source_label: str | None = None,
                        overlays: Path | None = None,
                        annotate: Path | None = None,
                        streamlines: bool | None = None,
                        theme: str | None = None,
                        section: str | None = None,
                        isotherms: str | None = None,
                        section_across_km: float | None = None,
                        section_size: tuple[int, int] | None = None,
                        section_top_km: float | None = None,
                        fills: list | None = None,
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
    # The render theme: a built-in name (``default``, ``dark``) or a JSON
    # theme file.  ``None`` adds no argument, so an invocation that never
    # names a theme is byte-identical to every earlier release, and the
    # engine's own default look is what it draws.
    if theme is not None:
        command.extend(("--theme", str(theme)))
    # The vertical-section line for ``xsec:`` products, its isotherm set
    # and the optional across-line frame; every one is engine grammar,
    # forwarded verbatim so the engine's own refusals name the mistake.
    if section is not None:
        command.extend(("--section", str(section)))
    if isotherms is not None:
        command.extend(("--isotherms", str(isotherms)))
    if section_across_km is not None:
        command.extend(("--section-across", repr(float(section_across_km))))
    # The size a SECTION is drawn at.  Absent, the engine draws a section
    # landscape 2:1 at the map's width, so a caller that never mentions it
    # is byte-identical to every earlier release for MAP products and gets
    # a cut whose shape is not the map's.
    if section_size is not None:
        width, height = section_size
        command.extend(("--section-size", f"{int(width)}x{int(height)}"))
    # The ceiling of a section's fitted height range.  Absent, the engine
    # keeps its own 14 km, so a caller that never mentions it draws what
    # every earlier release drew.  A published cut was 14 km tall whatever
    # the air in it was doing, which puts a 1 km marine layer in the
    # bottom fourteenth of the frame; the flag existed in the engine from
    # the start and no door forwarded it.
    top_problem = section_top_problem(section_top_km)
    if top_problem is not None:
        raise ValueError(top_problem)
    if section_top_km is not None:
        command.extend(("--section-top-km", repr(float(section_top_km))))
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
        elif fills is not None:
            row = parse_section_fill(line)
            if row is not None:
                fills.append(row)
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
    "run_renderer_series", "list_products_series",
    "GENERIC_FAMILIES", "GENERIC_VAR_PREFIX", "GROUP_KEYWORDS",
    "SECTION_PREFIX", "SECTION_TOP_KM_DEFAULT", "SECTION_TOP_KM_RANGE",
    "WINDOW_AXIS_CODES",
    "catalog_code", "catalog_rows",
    "catalog_verdict", "drop_storeless_terms",
    "MESH_PREFIXES", "split_section_spec", "section_top_problem",
    "SECTION_FILL_EVENT", "parse_section_fill",
]
