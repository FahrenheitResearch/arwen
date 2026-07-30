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

Nothing here runs cargo; resolution is read-only so ``gpuwm doctor``
can report the estate without side effects.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from gpuwm.bridges import (RUSTWX_CRATE_RELATIVE, artifact_remedy,
                           cargo_build_one_liner, default_bridge_dir,
                           executable_name)

#: Environment variable naming a prebuilt renderer executable.
RENDERER_ENV = "GPUWM_RW_WRFBATCH"

#: Executable base name of the vendored batch renderer.
RENDERER_NAME = "rw_wrfbatch"

#: The one-liner that builds the renderer, from a checkout root.
#: Shell-correct for this platform: Windows PowerShell 5.1 cannot
#: parse ``&&``.
CARGO_BUILD_HINT = cargo_build_one_liner(RUSTWX_CRATE_RELATIVE)

_PROBE_TIMEOUT_S = 20


def crate_dir() -> Path:
    """The vendored Rusty Weather workspace of a source checkout."""

    return Path(__file__).resolve().parent.parent / "tools" / "rustwx"


def basemap_dir() -> Path:
    """The vendored basemap assets (Natural Earth + US counties)."""

    return crate_dir() / "assets" / "basemap"


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
        default_bridge_dir() / filename,
    ))
    return tuple(candidates)


def find_renderer() -> Path | None:
    """First existing candidate, or None.

    An environment override that names a missing file is a hard error:
    explicit configuration must fail loudly, not fall through.
    """

    override = os.environ.get(RENDERER_ENV)
    for candidate in renderer_candidates():
        if candidate.is_file():
            return candidate.resolve()
        if override and candidate == Path(override):
            raise FileNotFoundError(
                f"{RENDERER_ENV} names a missing file: {candidate}")
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
    """Launch ``path --help`` once; can this binary execute at all?

    ``rw_wrfbatch --help`` prints its usage line and exits 0.  That
    observable separates a runnable executable from an empty, truncated,
    or wrong-platform file, which refuses to launch (OSError) or dies
    with an abnormal status and no usage text.
    """

    try:
        probe = subprocess.run(
            [str(path), "--help"], capture_output=True, text=True,
            errors="replace", timeout=_PROBE_TIMEOUT_S)
    except OSError as error:
        return False, f"exists but failed to execute: {error}"
    except subprocess.TimeoutExpired:
        return False, (f"probe invocation did not exit within "
                       f"{_PROBE_TIMEOUT_S} s")
    transcript = f"{probe.stdout or ''}{probe.stderr or ''}"
    if probe.returncode == 0 and "usage: rw_wrfbatch" in transcript:
        return True, "probe --help exited 0 with its usage line"
    return False, (f"probe --help exited {probe.returncode} without the "
                   "expected usage line -- corrupt, stale, or built for "
                   "another platform")


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
                 ) -> tuple[list[Path], list[str]]:
    """Render one wrfout file into ``out_dir``; (written, failures).

    One invocation per file with its own store root, exactly like the
    campaign flow: a shared store would merge two wrfouts into one run.
    The renderer's event lines are relayed to stdout as they arrive is
    not attempted -- the run is short-lived and its transcript is small,
    so it is captured and parsed for RENDERED/FAILED events instead.
    """

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
    command.append(str(wrfout))
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, errors="replace",
            env=renderer_env())
    except OSError as error:
        return [], [f"{wrfout}: renderer failed to launch: {error}"]
    written: list[Path] = []
    failures: list[str] = []
    for line in (result.stdout or "").splitlines():
        if line.startswith("RENDERED "):
            _, _, rest = line.partition(" ")
            _, _, path = rest.partition(" ")
            if path:
                written.append(Path(path))
    for line in (result.stderr or "").splitlines():
        if line.startswith("FAILED "):
            failures.append(f"{wrfout}: {line[len('FAILED '):]}")
    if result.returncode != 0:
        tail = [line for line in (result.stderr or "").splitlines()
                if line.strip()]
        detail = tail[-1] if tail else f"exit {result.returncode}"
        if not failures:
            failures.append(f"{wrfout}: {detail}")
    return written, failures


__all__ = [
    "CARGO_BUILD_HINT", "RENDERER_ENV", "RENDERER_NAME", "basemap_dir",
    "basemap_candidates", "crate_dir", "find_renderer", "list_products",
    "probe_renderer", "renderer_candidates", "renderer_env",
    "renderer_remedy", "resolve_basemap_dir", "run_renderer",
]
