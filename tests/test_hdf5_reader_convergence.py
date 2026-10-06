"""Keep NetCDF metadata and values on the same HDF5 reader.

An external build of ``netcrust`` previously pulled the registry HDF5
reader through ``netcdf-reader``. That parser omitted the managed free
space amount in fractal heap headers, then checked the wrong byte range
and rejected valid NetCDF-4 files with ``ChecksumMismatch``. Workspace
patches hid the defect from the engine's own binaries but did not reach
standalone consumers.

The facade now carries a complete path dependency chain. These checks
prevent that chain from depending on a consumer's workspace patch, and
ensure the engine workspaces lock the same readers. Real file/value
regressions exercise the Rust reader separately.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_FACADE_WORKSPACES = ("tools/rustwx",)
_READER_WORKSPACES = ("tools/rustwx", "tools/rw_wps", "tools/zarr_bridge")
_READERS = ("hdf5-reader", "netcdf-reader")
_PACKAGE_NAME = re.compile(r'^name = "(?P<name>[^"]+)"', re.M)
_PACKAGE_VERSION = re.compile(r'^version = "(?P<version>[^"]+)"', re.M)
_PACKAGE_SOURCE = re.compile(r'^source = "(?P<source>[^"]+)"', re.M)


def _section(text: str, name: str) -> str:
    match = re.search(
        rf'^\[{re.escape(name)}\][ \t]*\n(?P<body>.*?)(?=^\[|\Z)',
        text, re.M | re.S)
    return match.group("body") if match else ""


def _inline_path(text: str, package: str) -> str | None:
    match = re.search(
        rf'^\s*{re.escape(package)}\s*=\s*\{{[^}}]*\bpath\s*=\s*'
        r'"(?P<path>[^"]+)"', text, re.M)
    return match.group("path") if match else None


def _dependency_path(manifest: Path, package: str) -> str | None:
    text = manifest.read_text(encoding="utf-8")
    inline = _inline_path(_section(text, "dependencies"), package)
    if inline is not None:
        return inline
    match = re.search(
        r'^\s*path\s*=\s*"(?P<path>[^"]+)"',
        _section(text, f"dependencies.{package}"), re.M)
    return match.group("path") if match else None


def _patch_path(manifest: Path, package: str) -> str | None:
    text = manifest.read_text(encoding="utf-8")
    return _inline_path(_section(text, "patch.crates-io"), package)


def _locked_readers(workspace: Path, package: str) -> list[str | None]:
    """Sources for version 0.3.0; None means a path dependency.

    Other major/minor versions may serve unrelated callers. The facade
    and its direct registry callers use the 0.3.0 reader pair.
    """

    text = (workspace / "Cargo.lock").read_text(encoding="utf-8")
    sources: list[str | None] = []
    for block in text.split("[[package]]")[1:]:
        name = _PACKAGE_NAME.search(block)
        version = _PACKAGE_VERSION.search(block)
        if (name is None or name.group("name") != package
                or version is None or version.group("version") != "0.3.0"):
            continue
        source = _PACKAGE_SOURCE.search(block)
        sources.append(source.group("source") if source else None)
    return sources


@pytest.mark.parametrize("relative", _FACADE_WORKSPACES)
def test_facade_reader_chain_is_self_contained(relative: str) -> None:
    """Prevents standalone consumers selecting the broken registry parser."""

    facade = _ROOT / relative / "vendor/netcrust"
    for package in _READERS:
        expected = f"vendor/{package}"
        actual = _dependency_path(facade / "Cargo.toml", package)
        assert actual == expected, (
            f"{relative}/vendor/netcrust must depend on {package} by path "
            f"{expected!r}, so external builds do not require a workspace "
            f"patch; found {actual!r}")
        assert (facade / expected / "Cargo.toml").is_file()

    metadata_reader = facade / "vendor/netcdf-reader"
    actual = _dependency_path(metadata_reader / "Cargo.toml", "hdf5-reader")
    assert actual == "../hdf5-reader", (
        "the NetCDF metadata reader must use the facade's HDF5 reader "
        f"through its sibling path, found {actual!r}")
    assert (metadata_reader / actual).resolve() == (
        facade / "vendor/hdf5-reader").resolve()


@pytest.mark.parametrize("package", _READERS)
def test_direct_registry_callers_share_facade_readers(package: str) -> None:
    """Prevents rustwx's direct callers and facade using different decoders."""

    expected = f"vendor/netcrust/vendor/{package}"
    actual = _patch_path(_ROOT / "tools/rustwx/Cargo.toml", package)
    assert actual == expected, (
        f"tools/rustwx direct {package} dependencies must resolve to "
        f"{expected!r}, found {actual!r}")


@pytest.mark.parametrize("relative", _READER_WORKSPACES)
@pytest.mark.parametrize("package", _READERS)
def test_workspaces_lock_one_path_reader(relative: str, package: str) -> None:
    """Prevents stale locks retaining the defective metadata decoder."""

    sources = _locked_readers(_ROOT / relative, package)
    assert sources == [None], (
        f"{relative}/Cargo.lock must resolve one path {package} 0.3.0 "
        f"package; found sources {sources!r}. Regenerate the lock offline "
        "after changing the reader dependency chain.")


@pytest.mark.parametrize("relative", (
    "tools/rw_wps/crates/rw-wps",
    "tools/rw_wps/crates/mapped-engine",
    "tools/zarr_bridge",
))
def test_all_workspaces_depend_on_one_shared_facade(relative: str) -> None:
    """A reader fix reaches preprocessing and rendering through one source."""

    manifest = _ROOT / relative / "Cargo.toml"
    path = _dependency_path(manifest, "netcrust")
    assert path is not None
    assert (manifest.parent / path).resolve() == (
        _ROOT / "tools/rustwx/vendor/netcrust").resolve()


def test_retired_facade_reexports_the_shared_reader() -> None:
    facade = _ROOT / "tools/rw_wps/vendor/netcrust"
    manifest = facade / "Cargo.toml"
    text = manifest.read_text(encoding="utf-8")
    assert _PACKAGE_NAME.search(text).group("name") == "netcrust-compat"
    assert _PACKAGE_NAME.search(_section(text, "lib")).group("name") == "netcrust"
    path = _dependency_path(manifest, "netcrust_shared")
    assert path is not None
    assert (facade / path).resolve() == (
        _ROOT / "tools/rustwx/vendor/netcrust").resolve()
    assert "pub use netcrust_shared::*;" in (
        facade / "src/lib.rs").read_text(encoding="utf-8")
