"""One HDF5 reader per workspace, and the pin that keeps it one.

``netcrust`` is the tree's NetCDF read facade.  It reaches HDF5 two ways:
directly, through a ``hdf5-reader`` PATH dependency on the hardened
vendored copy, and indirectly, through ``netcdf-reader 0.3`` -- which
declares ``hdf5-reader = "0.3"`` and therefore resolves the crates.io
package unless the workspace redirects it.

``tools/rustwx`` carries that redirect.  ``tools/rw_wps`` did not, so its
``Cargo.lock`` locked TWO packages both named ``hdf5-reader 0.3.0``, and
the one that actually built netcrust's variable and dimension tables was
the stock registry crate -- 2782 lines of ``dataset.rs`` against the
vendored copy's 2874.

The concrete breakage that caused, measured on this tree: every file
written by ``netCDF4.Dataset(path, "w")`` refused inside
``gpuwm_mapped_engine`` with

    HDF5 error: checksum mismatch: expected 0x00000000, got 0xa31af89a

while the shipped ``rw_netcdf.exe`` -- same netcrust source, built inside
the patched rustwx workspace -- read the identical bytes and enumerated
every variable.  Four NetCDF suites went from 22 failures to 18 on the
one-line fix, so this is not a tidiness pin: it is the difference between
the mapped engine reading gpuwm's own NetCDF-4 corpus and refusing it.

These are text gates on purpose, in the ``test_grib_core_convergence``
tradition: they read manifests and lockfiles, need no Rust toolchain, and
so they run on every commit rather than at a cut.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]

#: Every workspace whose build closure contains ``netcrust``, with the
#: vendored ``hdf5-reader`` path each one's redirect must name.  The path
#: is relative to the workspace root because that is how cargo resolves a
#: ``[patch]`` entry, and it differs between the two only in that both
#: workspaces vendor their own netcrust snapshot.
_NETCRUST_WORKSPACES = {
    "tools/rw_wps": "vendor/netcrust/vendor/hdf5-reader",
    "tools/rustwx": "vendor/netcrust/vendor/hdf5-reader",
}

_PACKAGE_NAME = re.compile(r'^name = "(?P<name>[^"]+)"', re.M)
_PACKAGE_SOURCE = re.compile(r'^source = "(?P<source>[^"]+)"', re.M)


def _patch_redirect(workspace: Path) -> str | None:
    """The path ``[patch.crates-io] hdf5-reader`` redirects to, if any."""

    text = (workspace / "Cargo.toml").read_text(encoding="utf-8")
    match = re.search(
        r'^\s*hdf5-reader\s*=\s*\{[^}]*\bpath\s*=\s*"(?P<path>[^"]+)"',
        text, re.M)
    if match is None or "[patch.crates-io]" not in text:
        return None
    return match.group("path")


def _locked_hdf5_readers(workspace: Path) -> list[str | None]:
    """Every ``hdf5-reader`` package in the lock, by ``source``.

    ``None`` is a path package -- the vendored copy.  A registry string is
    the crates.io package, and a lock holding both is exactly the split
    this module exists to prevent.
    """

    text = (workspace / "Cargo.lock").read_text(encoding="utf-8")
    sources: list[str | None] = []
    for block in text.split("[[package]]")[1:]:
        name = _PACKAGE_NAME.search(block)
        if name is None or name.group("name") != "hdf5-reader":
            continue
        source = _PACKAGE_SOURCE.search(block)
        sources.append(source.group("source") if source else None)
    return sources


@pytest.mark.parametrize("relative", sorted(_NETCRUST_WORKSPACES))
def test_every_netcrust_workspace_redirects_hdf5_reader(relative: str) -> None:
    """Without the redirect, netcdf-reader silently pulls the stock crate.

    Prevents: a NetCDF-4 file that ``rw_netcdf`` reads refusing inside
    ``gpuwm_mapped_engine`` with an HDF5 checksum mismatch, because the
    two binaries linked different readers.
    """

    expected = _NETCRUST_WORKSPACES[relative]
    redirect = _patch_redirect(_ROOT / relative)
    assert redirect == expected, (
        f"{relative}/Cargo.toml must carry\n"
        f'    [patch.crates-io]\n'
        f'    hdf5-reader = {{ path = "{expected}" }}\n'
        f"so netcrust's `netcdf-reader 0.3` dependency resolves the "
        f"vendored hardened reader instead of the crates.io package; "
        f"found {redirect!r}")


@pytest.mark.parametrize("relative", sorted(_NETCRUST_WORKSPACES))
def test_every_netcrust_workspace_locks_exactly_one_hdf5_reader(
    relative: str,
) -> None:
    """Two locked readers means one of them is decoding the real bytes.

    Prevents: the redirect being added to the manifest while the lock is
    left stale, which builds and tests exactly as the unpatched tree did.
    """

    sources = _locked_hdf5_readers(_ROOT / relative)
    assert sources == [None], (
        f"{relative}/Cargo.lock resolves {len(sources)} hdf5-reader "
        f"package(s) with sources {sources!r}; it must resolve exactly one, "
        f"the path package from "
        f"{_NETCRUST_WORKSPACES[relative]}. Regenerate the lock after adding "
        f"the [patch.crates-io] redirect (`cargo update --offline -w`).")


def test_the_vendored_reader_is_not_the_stock_one() -> None:
    """The redirect only matters because the two trees differ.

    Prevents this whole module going quietly vacuous if a future vendor
    refresh made the vendored copy a byte-for-byte crates.io snapshot: at
    that point the pin would be protecting nothing and should be
    re-argued rather than kept out of habit.
    """

    workspace = _ROOT / "tools" / "rw_wps"
    vendored = workspace / "vendor/netcrust/vendor/hdf5-reader/src/dataset.rs"
    stock = workspace / "vendor/crates-io/hdf5-reader-0.3.0/src/dataset.rs"
    if not stock.is_file():
        pytest.skip("the stock hdf5-reader is no longer vendored")
    assert vendored.is_file(), f"{vendored} is missing"
    assert vendored.read_bytes() != stock.read_bytes(), (
        "the vendored and crates.io hdf5-reader trees are now identical, so "
        "the [patch.crates-io] redirect protects nothing; re-argue the pin "
        "instead of keeping it out of habit")
