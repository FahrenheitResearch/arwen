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
from typing import NamedTuple

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
#: parses, and the generic ``var:`` and vertical-section ``xsec:``
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

#: The group keywords that name no product and carry no section term.
_GROUP_KEYWORDS = frozenset(
    {"all", "direct", "derived", "heavy", "windowed"})

#: The two window-axis exclusions, as the engine's PROSE spelled them
#: before the row carried a code.  Kept only so a renderer built before
#: the code column still gets its window skip honoured; every row that
#: carries a code is matched on the code.
_WINDOW_AXIS_REASONS = frozenset({
    "windowed accumulations need more than one stored whole-hour frame",
    "exact-time ordinal axis; fixed-hour windows are undefined on it",
})

#: The codes those two rows carry.  A reason may be reworded at any
#: time; these may not, which is the whole point of them.
WINDOW_AXIS_CODES = ("windowed-needs-whole-hour-frames",
                     "windowed-ordinal-axis")

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
    return rows, summary


class CatalogRequirements(NamedTuple):
    """What each catalog slug needs, and what this build's import writes.

    Both halves are FILELESS: ``needs`` comes from the shared recipe
    table and ``planned`` from the import's own static plan, so the pair
    answers "can this install draw that product?" before a wrfout
    exists.  That is the question a plan review asks, and the
    store-aware listing cannot be asked it -- it needs an imported
    store, which is the thing that does not exist yet.

    ``needs`` is spelled in the STORE's selector vocabulary, never in
    wrfout variable names.  A wrf-core diagnostic declares no input list
    at all and its compute function reads whatever it needs at run time,
    so a row spelled in netCDF variable names would be an invented
    mapping of the kind this tree has already paid for once.
    """

    needs: dict[str, tuple[str, ...]]
    planned: frozenset[str]
    basis: str


#: How a requirements answer describes itself, so a reader can price a
#: warning built on it.  It is a PLAN, not a store: a run whose wrfout
#: sheds a variable narrows it further, and the store-aware listing is
#: still the authority once a file exists.
REQUIREMENTS_BASIS = (
    "this renderer build's own catalog requirements and wrfout import "
    "plan (rw_wrfbatch --list-products with no file), not a store")


def parse_catalog_requirements(text: str) -> CatalogRequirements:
    """The NEEDS/PLANNED pair out of one fileless catalog listing."""

    needs: dict[str, tuple[str, ...]] = {}
    planned: set[str] = set()
    for line in (text or "").splitlines():
        parts = line.rstrip("\n").split("\t")
        if parts[0] == "NEEDS" and len(parts) >= 3:
            needs[parts[1]] = tuple(part for part in parts[2:] if part)
        elif parts[0] == "PLANNED" and len(parts) >= 2 and parts[1]:
            planned.add(parts[1])
    return CatalogRequirements(needs, frozenset(planned), REQUIREMENTS_BASIS)


_REQUIREMENTS_CACHE: dict[tuple, CatalogRequirements] = {}


def catalog_requirements(renderer: Path | None = None) -> CatalogRequirements | None:
    """Ask the installed renderer for the fileless pair, or ``None``.

    ``None`` means the question could not be ASKED -- no renderer
    resolvable, or a build too old to answer these rows.  It is never a
    refusal and never an empty answer dressed as one: a caller that gets
    ``None`` states its basis and warns once rather than declaring every
    product undrawable, because "unmeasured" is not "impossible".

    Parsed once per renderer build (path, size, mtime), because the
    answer is a property of the binary and a picker asks it per preset.
    """

    path = Path(renderer) if renderer is not None else find_renderer()
    if path is None:
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    cached = _REQUIREMENTS_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        result = subprocess.run(
            [str(path), "--list-products"], capture_output=True, text=True,
            errors="replace", env=renderer_env(), timeout=_PROBE_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    answer = parse_catalog_requirements(result.stdout or "")
    if not answer.needs or not answer.planned:
        # A build predating the requirement rows.  Saying so is the
        # answer; inventing rows for it is how two catalogs start
        # disagreeing about one question.
        return None
    _REQUIREMENTS_CACHE[key] = answer
    return answer


def undrawable(slugs, *, requirements: CatalogRequirements | None) -> dict[str, str]:
    """Which of ``slugs`` this install cannot draw, and why; by slug.

    One function, both doors: the preset picker and the plan-time
    history warning ask exactly this, and two answers to one question is
    how a picker and a render come to disagree.

    A slug the build carries no requirement row for is NOT reported.
    Silence there is the truthful answer -- this says what it can prove
    from the two tables it read, and nothing else.  Each reason names
    the selector that is missing and the basis it was decided on, so a
    reader can price it.
    """

    if requirements is None:
        return {}
    missing: dict[str, str] = {}
    for slug in dict.fromkeys(str(slug) for slug in slugs):
        required = requirements.needs.get(slug)
        if not required:
            continue
        absent = [key for key in required if key not in requirements.planned]
        if not absent:
            continue
        missing[slug] = (
            f"the wrfout import does not write {', '.join(absent)}, which "
            f"{slug} is drawn from (basis: {requirements.basis})")
    return missing


def catalog_code(row) -> str:
    """One row's machine code, or ``""`` for a build that emits none."""

    return row[4] if len(row) >= 5 else ""


def window_axis_unavailable(rows, requested) -> dict[str, str]:
    """Which requested slugs the catalog excluded on its WINDOW AXIS.

    THE selector for that skip, and the reason it is a function rather
    than a set of sentences at each door: the two exclusions are the
    rows a caller must act on -- drop the slug, or the whole render
    fails -- and a caller that recognised them by their English text
    stopped recognising them the first time the engine reworded one.

    The value is the engine's own detail, verbatim, because that is what
    the reader is shown.  A row from a build older than the code column
    is still matched, by its prose, so this is a strict improvement
    rather than a new requirement on the binary.
    """

    wanted = [token.strip() for token in requested] if not isinstance(
        requested, str) else [token.strip() for token in requested.split(",")]
    wanted = [token for token in wanted if token]
    excluded = {}
    for row in rows:
        slug, kind, status, detail = row[0], row[1], row[2], row[3]
        if kind != "windowed" or status != "excluded":
            continue
        code = catalog_code(row)
        if code in WINDOW_AXIS_CODES or (not code and detail in _WINDOW_AXIS_REASONS):
            excluded[slug] = detail
    return {slug: excluded[slug] for slug in dict.fromkeys(wanted)
            if slug in excluded}


def catalog_verdict(rows, requested) -> tuple[str, list[tuple[str, str]]]:
    """``(the spec that can be drawn, the named skips)`` for one request.

    The availability listing has already been paid for by the time a
    door has rows in hand; this reads them.  Every requested slug whose
    row is renderable stays in the spec, in the order it was asked for;
    every requested slug whose row is anything else comes back paired
    with the engine's OWN detail, verbatim, so the reader gets the
    engine's reason rather than a Python paraphrase of it.

    A token with no row at all passes through untouched: the group
    keywords (``all``, ``direct`` ...) and the generic families
    (:data:`GENERIC_FAMILIES`) name no catalog row, and the engine is
    the authority on an unknown slug -- guessing here would refuse a
    spelling this build accepts.

    An empty spec is the caller's signal to refuse before launching
    rather than to run an empty render.
    """

    wanted = ([token.strip() for token in requested.split(",")]
              if isinstance(requested, str)
              else [str(token).strip() for token in requested])
    wanted = [token for token in wanted if token]
    status = {row[0]: (row[2], row[3]) for row in rows}
    available, excluded = [], []
    for token in dict.fromkeys(wanted):
        row = status.get(token)
        if row is None or row[0] == "renderable":
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


def section_required(products: str) -> bool:
    """True when this ``--products`` spelling needs a section line."""

    return bool(split_section_spec(products)[1])


def section_spec_problem(products: str, *, section=None) -> str | None:
    """The refusal for a section request with no line, or ``None``.

    The engine's own sentence, because the engine is the authority on
    its own grammar -- and it fires HERE, at plan review, instead of
    after a forecast: a chain that names an ``xsec:`` product but
    composes no section line is refused by the renderer at render time,
    which is hours of integration paid for before the mistake is read.
    """

    if section is not None and str(section).strip():
        return None
    if not section_required(products):
        return None
    return ("xsec: products need a line: --section lat,lon,lat,lon or "
            "--section FILE.json.  This chain composes no section line, so "
            "the renderer would refuse after the forecast; draw the cut with "
            "`gpuwm render --section lat,lon,lat,lon FILE` instead.")


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
    """

    return run_renderer_series(
        renderer, (wrfout,), store_root=store_root, out_dir=out_dir,
        products=products, frames=frames, width=width, height=height,
        heavy=heavy, source_label=source_label, overlays=overlays,
        annotate=annotate, streamlines=streamlines, theme=theme,
        section=section, isotherms=isotherms,
        section_across_km=section_across_km, section_size=section_size)


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
    "run_renderer_series", "list_products_series",
]
