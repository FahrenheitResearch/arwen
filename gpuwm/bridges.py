"""Locate the built Rust bridge executables outside a source checkout.

The pip wheel deliberately ships no compiled Rust: the fail-closed GRIB
decoders and the CPU preprocessing library are built once from the
vendored ``tools/grib1_bridge`` workspace (``cargo build --release
--locked --offline``) and then *pointed at*.  This module is the single
resolution mechanism shared by ingest (:func:`gpuwm.ingest.grib
.build_rust_bridge`), ``gpuwm doctor``, and documentation:

1. an explicit per-executable environment variable
   (:data:`BRIDGE_ENV`) naming the built file;
2. a source checkout's own ``tools/grib1_bridge/target/{release,debug}``
   build tree (the developer path -- ingest may also *build* there);
3. ``<root>/libexec/bridges`` beside the package (the sealed runtime
   archive layout);
4. the user-level default directory :func:`default_bridge_dir`
   (``~/.gpuwm/bridges``), where a wheel user copies the built
   executables once.

Nothing here runs cargo; resolution is read-only so ``gpuwm doctor``
can report the estate without side effects.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Bridge executable -> the environment variable that names a prebuilt
#: copy.  The same variables drive the sealed-runtime decoder binding in
#: :mod:`gpuwm.source_cli`, so one mechanism serves both install modes.
BRIDGE_ENV = {
    "grib1_bridge": "GPUWM_GRIB1_BRIDGE",
    "gfs_grib2_bridge": "GPUWM_GFS_GRIB2_BRIDGE",
    "hrrr_grib2_bridge": "GPUWM_HRRR_DECODER",
    "grib2_inventory": "GPUWM_GRIB2_INVENTORY",
    "grib2_dump": "GPUWM_GRIB2_DUMP",
}

#: The one-liner that builds every bridge from a source clone.
CARGO_BUILD_HINT = (
    "cd tools/grib1_bridge && cargo build --release --locked --offline")


def default_bridge_dir() -> Path:
    """User-level directory for prebuilt bridges: ``~/.gpuwm/bridges``."""

    return Path.home() / ".gpuwm" / "bridges"


def _package_parent() -> Path:
    """The directory containing the ``gpuwm`` package.

    A source checkout's repository root, or ``site-packages`` for an
    installed wheel (where the crate does not exist).
    """

    return Path(__file__).resolve().parent.parent


def crate_dir() -> Path:
    """The vendored Rust workspace of a source checkout (may not exist)."""

    return _package_parent() / "tools" / "grib1_bridge"


def executable_name(name: str) -> str:
    return f"{name}.exe" if os.name == "nt" else name


def artifact_candidates(env_var: str, filename: str) -> tuple[Path, ...]:
    """Deterministic candidate paths for one built artifact, best first.

    THE resolution order for everything ``tools/grib1_bridge`` builds
    (bridge executables and the CPU preprocessing library alike):
    environment override, checkout release, checkout debug, ``libexec``
    beside the package, user-level default directory.  The environment
    override comes first; a missing file it names is the caller's error
    to raise (never silently skipped).
    """

    candidates: list[Path] = []
    override = os.environ.get(env_var)
    if override:
        candidates.append(Path(override))
    root = _package_parent()
    candidates.extend((
        crate_dir() / "target" / "release" / filename,
        crate_dir() / "target" / "debug" / filename,
        root / "libexec" / "bridges" / filename,
        default_bridge_dir() / filename,
    ))
    return tuple(candidates)


def find_artifact(env_var: str, filename: str) -> Path | None:
    """First existing candidate, or None.

    An environment override that names a missing file is a hard error:
    explicit configuration must fail loudly, not fall through to a
    different executable or library.
    """

    override = os.environ.get(env_var)
    for candidate in artifact_candidates(env_var, filename):
        if candidate.is_file():
            return candidate.resolve()
        if override and candidate == Path(override):
            raise FileNotFoundError(
                f"{env_var} names a missing file: {candidate}")
    return None


def bridge_candidates(name: str) -> tuple[Path, ...]:
    """Deterministic candidate paths for bridge ``name``, best first."""

    if name not in BRIDGE_ENV:
        raise ValueError(f"unknown bridge executable {name!r}; known: "
                         f"{sorted(BRIDGE_ENV)}")
    return artifact_candidates(BRIDGE_ENV[name], executable_name(name))


def find_bridge(name: str) -> Path | None:
    """First existing candidate for bridge ``name``, or None.

    See :func:`find_artifact` for the fail-loud override contract.
    """

    if name not in BRIDGE_ENV:
        raise ValueError(f"unknown bridge executable {name!r}; known: "
                         f"{sorted(BRIDGE_ENV)}")
    return find_artifact(BRIDGE_ENV[name], executable_name(name))


def bridge_remedy(name: str) -> str:
    """The exact copy-pasteable remedy for a missing bridge."""

    env = BRIDGE_ENV[name]
    filename = executable_name(name)
    return (
        "build the bridges once from a source clone of this repository:\n"
        f"    {CARGO_BUILD_HINT}\n"
        f"  then EITHER set {env}=<clone>/tools/grib1_bridge/target/"
        f"release/{filename}\n"
        f"  OR copy the built executables into {default_bridge_dir()}")


__all__ = [
    "BRIDGE_ENV", "CARGO_BUILD_HINT", "artifact_candidates",
    "bridge_candidates", "bridge_remedy", "crate_dir",
    "default_bridge_dir", "executable_name", "find_artifact",
    "find_bridge",
]
