"""``gpuwm fetch-tables``: stage externalized physics table assets.

Most packaged lookup tables ship inside the wheel and the repository.
The exceptions are the *externalized* assets named here: files whose
size exceeds distribution-channel limits, published as versioned GitHub
release assets and staged into the resolved table root on demand.
Currently that is the two largest classic Thompson coefficient tables:

- ``freezeH2O.dat`` (243 MiB): over GitHub's 100 MiB blob limit, so it
  is absent from both the repository and the wheel/sdist -- every
  install fetches it once.
- ``qr_acr_qg_V4.dat`` (71 MiB): fits in the repository (git-clone
  installs already have it; fetch-tables verifies and moves on) but
  would push the wheel and sdist over PyPI's 100 MB per-file cap, so
  wheel installs fetch it.

The staging contract is the model's own: every byte that lands in the
table root is verified against the exact size and SHA-256 pins in
:data:`gpuwm.core.thompson_contract.CLASSIC_TABLE_ASSETS` -- the same
pins every ``mp_physics=8`` run enforces at load -- *before* the file is
atomically moved into place.  A mismatched download is deleted and
refused, never installed; an existing file with wrong bytes is refused,
never overwritten (delete it yourself and re-run).  The command is
idempotent: with all assets present and byte-valid it verifies and
exits 0 without touching the network.

``--from DIR`` stages from a local directory (offline/air-gapped
installs) under the same verification.  ``GPUWM_TABLE_ASSET_URL_BASE``
overrides the download base URL (mirrors); the asset filename is
appended to it either way.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request

from gpuwm import fetch_guard
from gpuwm.core.thompson_contract import (
    CLASSIC_TABLE_ASSETS,
    TableAsset,
    validate_table_assets,
)

#: Externalized asset filenames: published as release assets and
#: excluded from the wheel/sdist (freezeH2O.dat is additionally absent
#: from the repository -- GitHub rejects blobs over 100 MiB).
#: Everything else in CLASSIC_TABLE_ASSETS ships in the package, so its
#: absence means a broken install (reinstall), not a pending download.
EXTERNALIZED_TABLE_FILENAMES = frozenset({
    "freezeH2O.dat",
    "qr_acr_qg_V4.dat",
})

#: Release the pinned assets were published with.  The pins themselves
#: live in thompson_contract; this only says where the bytes are hosted.
RELEASE_ASSET_BASE_URL = (
    "https://github.com/FahrenheitResearch/arwen/releases/download/v1.0.0")

ASSET_URL_BASE_ENV = "GPUWM_TABLE_ASSET_URL_BASE"

_BLOCK_BYTES = 8 * 1024 * 1024


def asset_url_base() -> str:
    override = os.environ.get(ASSET_URL_BASE_ENV)
    return override.rstrip("/") if override else RELEASE_ASSET_BASE_URL


class TableAssetError(RuntimeError):
    """A refusal: wrong bytes, unfetchable asset, or unusable source."""


def classify_assets(
        root: Path,
        assets: tuple[TableAsset, ...] = CLASSIC_TABLE_ASSETS,
) -> tuple[list[TableAsset], list[str], list[TableAsset]]:
    """(valid, invalid-detail, absent) partition of ``assets`` at ``root``.

    ``invalid`` entries are human-readable refusals for files that exist
    with the wrong size or SHA-256; ``absent`` entries simply do not
    exist yet.
    """

    valid: list[TableAsset] = []
    invalid: list[str] = []
    absent: list[TableAsset] = []
    for asset in assets:
        path = root / asset.filename
        if not path.is_file():
            absent.append(asset)
            continue
        try:
            validate_table_assets(root, (asset,))
        except ValueError as error:
            invalid.append(str(error))
            continue
        valid.append(asset)
    return valid, invalid, absent


def missing_externalized_assets(
        root: Path,
        assets: tuple[TableAsset, ...] = CLASSIC_TABLE_ASSETS,
) -> list[TableAsset]:
    """Externalized assets absent from ``root`` (fetchable gaps)."""

    _valid, _invalid, absent = classify_assets(root, assets)
    return [asset for asset in absent
            if asset.filename in EXTERNALIZED_TABLE_FILENAMES]


def _staging_path(root: Path, asset: TableAsset) -> Path:
    """A staging name only this process, and only this call, can own.

    The staging file used to be a fixed ``.<filename>.fetch-partial``
    shared by every writer into the table root.  Verification and the
    ``os.replace`` that installs the verified bytes are separate steps,
    so a second writer could rewrite that fixed file in between and get
    its bytes installed under the first writer's successful pin check --
    an install that never matched the pin it was blessed by.  A unique
    name means the bytes hashed are the bytes moved.
    """

    return root / (f".{asset.filename}.fetch-partial-"
                   f"{os.getpid()}-{time.time_ns()}")


def _stage(temp: Path, final: Path, asset: TableAsset) -> None:
    """Verify ``temp`` against ``asset`` pins, then atomically install."""

    actual_bytes = temp.stat().st_size
    if actual_bytes != asset.bytes:
        temp.unlink(missing_ok=True)
        raise TableAssetError(
            f"{asset.filename}: staged {actual_bytes} bytes, expected "
            f"{asset.bytes}; refused and deleted")
    digest = hashlib.sha256()
    with temp.open("rb") as stream:
        while block := stream.read(_BLOCK_BYTES):
            digest.update(block)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != asset.sha256:
        temp.unlink(missing_ok=True)
        raise TableAssetError(
            f"{asset.filename}: staged SHA-256 {actual_sha256}, expected "
            f"{asset.sha256}; refused and deleted")
    os.replace(temp, final)


def fetch_asset_from_url(root: Path, asset: TableAsset, url: str) -> Path:
    """Download ``url`` into ``root`` with pre-install verification.

    Held under the table-root lock: staging, verifying and installing is
    one transaction, so a concurrent fetch into the same root queues
    behind it instead of interleaving with it.
    """

    final = root / asset.filename
    with fetch_guard.hold("fetch-tables", root):
        temp = _staging_path(root, asset)
        try:
            with urllib.request.urlopen(url) as response, \
                    temp.open("wb") as sink:
                shutil.copyfileobj(response, sink, _BLOCK_BYTES)
        except (urllib.error.URLError, OSError) as error:
            temp.unlink(missing_ok=True)
            raise TableAssetError(
                f"{asset.filename}: download failed from {url}: {error}")
        _stage(temp, final, asset)
    return final


def fetch_asset_from_dir(root: Path, asset: TableAsset,
                         source_dir: Path) -> Path:
    """Copy ``asset`` from a local directory with the same verification."""

    source = source_dir / asset.filename
    if not source.is_file():
        raise TableAssetError(
            f"{asset.filename}: not found in --from directory {source_dir}")
    final = root / asset.filename
    with fetch_guard.hold("fetch-tables", root):
        temp = _staging_path(root, asset)
        try:
            shutil.copyfile(source, temp)
        except OSError as error:
            temp.unlink(missing_ok=True)
            raise TableAssetError(
                f"{asset.filename}: copy from {source} failed: {error}")
        _stage(temp, final, asset)
    return final


def fetch_tables_main(args) -> int:
    from gpuwm.physics_compat import thompson_table_root

    root = Path(thompson_table_root())
    if not root.is_dir():
        print(f"gpuwm fetch-tables: table root {root} is not a directory "
              "(broken install, or GPUWM_THOMPSON_TABLE_ROOT points "
              "nowhere)")
        return 2

    valid, invalid, absent = classify_assets(root)
    for line in invalid:
        print(f"gpuwm fetch-tables: REFUSED: {line} -- an existing file "
              "is never overwritten; delete it and re-run")
    if invalid:
        return 2

    packaged_gaps = [asset for asset in absent
                     if asset.filename not in EXTERNALIZED_TABLE_FILENAMES]
    for asset in packaged_gaps:
        print(f"gpuwm fetch-tables: {asset.filename} is missing but ships "
              "inside the package itself; this install is incomplete -- "
              "reinstall gpuwm rather than fetching")
    if packaged_gaps:
        return 2

    fetchable = [asset for asset in absent
                 if asset.filename in EXTERNALIZED_TABLE_FILENAMES]
    if not fetchable:
        print(f"gpuwm fetch-tables: all {len(valid)} table assets at "
              f"{root} verified (exact size + SHA-256); nothing to fetch")
        return 0

    source_dir = getattr(args, "from_dir", None)
    base = asset_url_base()
    for asset in fetchable:
        mib = asset.bytes / (1024 * 1024)
        try:
            if source_dir is not None:
                print(f"gpuwm fetch-tables: staging {asset.filename} "
                      f"({mib:.1f} MiB) from {source_dir}")
                final = fetch_asset_from_dir(root, asset, Path(source_dir))
            else:
                url = f"{base}/{asset.filename}"
                print(f"gpuwm fetch-tables: downloading {asset.filename} "
                      f"({mib:.1f} MiB) from {url}")
                final = fetch_asset_from_url(root, asset, url)
        except TableAssetError as error:
            print(f"gpuwm fetch-tables: REFUSED: {error}")
            return 2
        print(f"gpuwm fetch-tables: verified and installed {final} "
              f"({asset.bytes:,} B, SHA-256 {asset.sha256[:12]}...)")

    # Close the loop with the exact validation every mp8 run performs.
    validate_table_assets(root)
    print(f"gpuwm fetch-tables: table root {root} complete and byte-valid")
    return 0


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "fetch-tables",
        help="download the externalized physics table assets (currently "
             "the 243 MiB Thompson freezeH2O.dat) into the resolved table "
             "root, SHA-256-verified against the packaged pins before "
             "install; idempotent when everything is already present")
    parser.add_argument(
        "--from", dest="from_dir", metavar="DIR", default=None,
        help="stage from a local directory instead of downloading "
             "(offline installs); verification is identical")
    parser.set_defaults(func=fetch_tables_main)
    return parser


__all__ = [
    "ASSET_URL_BASE_ENV",
    "EXTERNALIZED_TABLE_FILENAMES",
    "RELEASE_ASSET_BASE_URL",
    "TableAssetError",
    "asset_url_base",
    "classify_assets",
    "fetch_asset_from_dir",
    "fetch_asset_from_url",
    "fetch_tables_main",
    "missing_externalized_assets",
    "register_cli",
]
