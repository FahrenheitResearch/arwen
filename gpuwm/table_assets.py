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

``--wif`` adds ONE more externalized asset to the same transaction: the
225 MB ``QNWFA_QNIFA_SIGMA_MONTHLY.dat`` global monthly aerosol
climatology the mp=28 WIF ingest reads
(:mod:`gpuwm.ingest.wif_dataset` holds its pin and its acquisition
route).  It is opt-in rather than default because it is an INPUT
DATASET, not a coefficient table: no scheme selection makes it
mandatory, only ``aer_init_opt=1`` with ``wif_input_opt=1`` does, and a
default install must not be asked for a quarter-gigabyte it will never
open.  It stages into ``~/.gpuwm/wif`` -- its own root, because a table
root is validated as a complete set and a WPS intermediate file inside
one would make "complete" unanswerable -- under the identical
verify-then-atomically-install contract, since the reason both files
are externalized is one reason: the release checklist's size gates.

WHERE the bytes land moved once, deliberately.  This command used to
stage into whatever ``thompson_table_root()`` resolved to, which on a
wheel install is *inside site-packages* -- so the 315 MiB a user had
just paid for was deleted by the next ``pip install --upgrade`` or venv
rebuild, silently, and the failure surfaced as a bare
``FileNotFoundError`` traceback in the middle of a forecast that had
already paid for fetch and preprocessing.  Staging now targets
``~/.gpuwm/tables/thompson`` (:func:`gpuwm.physics_compat
.user_thompson_table_root`), beside ``~/.gpuwm/bridges``, and the
staged root is made *complete*: the assets that do ship in the package
are copied in beside the downloaded ones, because a root is validated
as a whole set and a half-staged directory would resolve to nothing.
An install whose packaged root is already complete -- every git clone,
and every wheel that staged into site-packages before this change --
is left exactly where it is.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request

from gpuwm import bridges
from gpuwm import fetch_guard
from gpuwm.progress import ByteCounter
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

#: Transfer block, smaller than the hashing block on purpose: it is the
#: resolution of the byte counter above, and an eight-MiB step over a
#: slow link is a number that stands still for seconds at a time.
_TRANSFER_BLOCK_BYTES = 1024 * 1024


def asset_url_base() -> str:
    override = os.environ.get(ASSET_URL_BASE_ENV)
    return override.rstrip("/") if override else RELEASE_ASSET_BASE_URL


def wif_asset_url_base() -> str:
    """Where ``--wif`` downloads from: the VERSIONED release-assets base.

    This used to return :func:`asset_url_base`, i.e. the fixed ``v1.0.0``
    release the coefficient tables are published under -- where this file
    has never existed.  ``gpuwm fetch-tables --wif`` therefore refused
    with ``HTTP Error 404`` on every install that had no local copy, which
    is every install: there was no reachable route to the dataset at all.

    The coefficient tables can sit at a fixed tag because their bytes are
    frozen -- the same four files answer every release.  This dataset is
    carried as an asset of the release that pins it, alongside the bridge
    bundles, so its base has to MOVE with the release.  That is exactly
    what :func:`gpuwm.bridge_assets.asset_url_base` computes
    (``<REPOSITORY_URL>/releases/download/<pins.release>``), including the
    ``GPUWM_BRIDGE_ASSET_URL_BASE`` override, so it is reused rather than
    restated: one versioned base, not two that can drift apart.

    ``GPUWM_TABLE_ASSET_URL_BASE`` still wins when it is set, because an
    operator who has named a mirror for this command's downloads has named
    it for all of them.
    """

    override = os.environ.get(ASSET_URL_BASE_ENV)
    if override and override.strip():
        return override.strip().rstrip("/")

    from gpuwm import bridge_assets

    try:
        return bridge_assets.asset_url_base(bridge_assets.load_pins())
    except Exception as error:
        from gpuwm.ingest.wif_dataset import WIF_DATASET_FILE

        raise TableAssetError(
            f"{WIF_DATASET_FILE}: no download URL could be resolved.  It "
            "is published as an asset of the release that pins it, at "
            f"<{bridges.REPOSITORY_URL}/releases/download/"
            f"<release>/{WIF_DATASET_FILE}>, and the packaged release pin "
            f"could not be read: {error}\n"
            "  how to supply it: stage from a local copy with `gpuwm "
            "fetch-tables --wif --from DIR`, or set "
            f"{ASSET_URL_BASE_ENV} (or GPUWM_BRIDGE_ASSET_URL_BASE) to a "
            "base URL that carries the file")


class TableAssetError(RuntimeError):
    """A refusal: wrong bytes, unfetchable asset, or unusable source."""


class MissingTableAssets(FileNotFoundError):
    """One-sentence refusal: a selected scheme's tables are not staged.

    A ``FileNotFoundError`` subclass on purpose.  Every stage that can
    meet this already turns ``OSError`` into a sentence rather than a
    traceback, so the sentence below reaches the reader through the
    handlers that exist instead of needing a new one at each site.
    """


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


def unstaged_table_assets(
        root: Path | None = None,
        assets: tuple[TableAsset, ...] = CLASSIC_TABLE_ASSETS,
) -> list[TableAsset]:
    """Assets that are simply not there in ``root``.

    Presence, and nothing else.  Size and SHA-256 stay where they
    belong -- re-validated at load, on every launch, by
    ``validate_table_assets``, which can name the expected and actual
    bytes.  Absence is the question a preflight owns, and answering it
    must not cost 362 MiB of hashing on every run; a file that is
    present but wrong is drift, a different sentence, and this function
    deliberately does not compete with it.
    """

    if root is None:
        from gpuwm.physics_compat import thompson_table_root

        root = Path(thompson_table_root())
    root = Path(root)
    gaps: list[TableAsset] = []
    for asset in assets:
        try:
            present = (root / asset.filename).is_file()
        except OSError:
            present = False
        if not present:
            gaps.append(asset)
    return gaps


def require_thompson_tables(
        root: Path | None = None,
        assets: tuple[TableAsset, ...] = CLASSIC_TABLE_ASSETS,
) -> Path:
    """The resolved mp8 table root, or a one-sentence refusal.

    THE preflight for ``mp_physics=8``.  It exists because the same
    condition used to be discovered by ``validate_table_assets`` at the
    top of the GPU run -- after the fetch, after preprocessing, minutes
    of paid work in -- and reached the reader as a ``FileNotFoundError``
    traceback naming a path inside site-packages.  Called from the
    stages that select physics, it is one sentence naming the table and
    the command that stages it, before anything expensive starts.
    """

    if root is None:
        from gpuwm.physics_compat import thompson_table_root

        root = Path(thompson_table_root())
    root = Path(root)
    # ``assets`` is a parameter because the pins are not universal: a
    # caller that binds its own table authority (and the harnesses that
    # stand in for one) must be checked against the set IT will load,
    # not against a second opinion this module happens to hold.
    gaps = unstaged_table_assets(root, assets)
    if not gaps:
        return root
    mib = sum(asset.bytes for asset in gaps) / (1024 * 1024)
    raise MissingTableAssets(
        f"mp_physics=8 (Thompson) needs "
        f"{', '.join(asset.filename for asset in gaps)}, which "
        f"{'are' if len(gaps) > 1 else 'is'} not staged at {root} -- run "
        f"`gpuwm fetch-tables` ({mib:.0f} MiB, SHA-256 verified against "
        f"the packaged pins) before preparing this forecast")


def staging_is_self_chosen(root: Path) -> bool:
    """True when ``root`` is the user-level location THIS command picked.

    The distinction matters for one behaviour only: a self-chosen root
    is completed from the package (below), because nothing else will
    complete it and an incomplete root resolves to nothing.  A root an
    operator NAMED through ``GPUWM_THOMPSON_TABLE_ROOT`` is theirs --
    this command fetches the externalized assets into it and reports
    anything else missing as a broken install, exactly as it always
    has, rather than quietly filling their mirror from the package.
    """

    from gpuwm.physics_compat import (THOMPSON_TABLE_ROOT_ENV,
                                      user_thompson_table_root)

    if os.environ.get(THOMPSON_TABLE_ROOT_ENV):
        return False
    try:
        return Path(root) == user_thompson_table_root()
    except (RuntimeError, OSError):  # pragma: no cover - no home directory
        return False


def staging_root(root: Path | None = None) -> Path:
    """WHERE ``fetch-tables`` writes: the root the next run will read.

    Deliberately not "wherever ``thompson_table_root()`` points today".
    On a fresh wheel that is site-packages, which the next upgrade
    empties; the durable location is the user-level directory beside
    ``~/.gpuwm/bridges``.  An install whose *packaged* root is already
    complete keeps it -- a clone, or a wheel staged before this change
    -- because that is the root :func:`thompson_table_root` resolves for
    it, and re-staging 362 MiB to change nothing is not a service.

    ``GPUWM_THOMPSON_TABLE_ROOT`` still wins outright: an operator who
    named a root is telling this command where the bytes go.
    """

    if root is not None:
        return Path(root)
    from gpuwm.physics_compat import (
        THOMPSON_TABLE_ROOT_ENV,
        packaged_thompson_table_root,
        user_thompson_table_root,
    )

    override = os.environ.get(THOMPSON_TABLE_ROOT_ENV)
    if override:
        return Path(override)
    packaged = packaged_thompson_table_root()
    if not unstaged_table_assets(packaged):
        return packaged
    try:
        return user_thompson_table_root()
    except (RuntimeError, OSError):  # pragma: no cover - no home directory
        return packaged


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

    The transfer COUNTS ITS BYTES on stderr while it runs.  This is the
    303 MiB that made ``gpuwm setup`` sit silent for 15.5 s of its 16.2
    (UX finding N9), and the pinned size in ``asset`` is the total, so
    the percentage is the contract's own number rather than a header the
    host supplied.  stderr because ``gpuwm setup`` captures each step's
    stdout; see :mod:`gpuwm.progress`.
    """

    final = root / asset.filename
    with fetch_guard.hold("fetch-tables", root):
        temp = _staging_path(root, asset)
        counter = ByteCounter(
            f"gpuwm fetch-tables: {asset.filename}", asset.bytes)
        try:
            with urllib.request.urlopen(url) as response, \
                    temp.open("wb") as sink:
                while True:
                    block = response.read(_TRANSFER_BLOCK_BYTES)
                    if not block:
                        break
                    sink.write(block)
                    counter.advance(len(block))
        except (urllib.error.URLError, OSError) as error:
            temp.unlink(missing_ok=True)
            raise TableAssetError(
                f"{asset.filename}: download failed from {url}: {error}")
        finally:
            counter.close()
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


def stage_classic_tables(args) -> int:
    """Stage the mandatory classic coefficient tables.  0, or 2 on refusal.

    The mandatory leg of ``fetch-tables``: every ``mp_physics=8`` and
    ``mp_physics=28`` run reads this set, and no flag makes it optional.
    Split out of :func:`fetch_tables_main` so the optional ``--wif`` leg
    cannot decide whether it runs.
    """

    from gpuwm.physics_compat import packaged_thompson_table_root

    root = staging_root(getattr(args, "root", None))
    packaged = packaged_thompson_table_root()
    if not root.is_dir():
        if root == packaged:
            print(f"gpuwm fetch-tables: table root {root} is not a "
                  "directory (broken install, or "
                  "GPUWM_THOMPSON_TABLE_ROOT points nowhere)")
            return 2
        # The user-level staging root is ours to create, exactly as
        # `gpuwm fetch-bridges` creates ~/.gpuwm/bridges.
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            print(f"gpuwm fetch-tables: cannot create table root {root}: "
                  f"{error}")
            return 2
        print(f"gpuwm fetch-tables: staging into {root} (outside the "
              "install, so a wheel upgrade cannot delete it)")

    valid, invalid, absent = classify_assets(root)
    for line in invalid:
        print(f"gpuwm fetch-tables: REFUSED: {line} -- an existing file "
              "is never overwritten; delete it and re-run")
    if invalid:
        return 2

    # A staged root has to hold the WHOLE set: resolution asks whether a
    # root is complete, so downloading the two externalized assets into
    # ~/.gpuwm and leaving the other two behind in site-packages would
    # produce two half-roots and no runnable one.  Assets the package
    # does carry are copied across under the same pin verification.
    copied: list[TableAsset] = []
    if root != packaged and staging_is_self_chosen(root):
        available = {asset.filename for asset in classify_assets(packaged)[0]}
        for asset in list(absent):
            if asset.filename not in available:
                continue
            mib = asset.bytes / (1024 * 1024)
            print(f"gpuwm fetch-tables: copying {asset.filename} "
                  f"({mib:.1f} MiB) from the package into {root}")
            try:
                fetch_asset_from_dir(root, asset, packaged)
            except TableAssetError as error:
                print(f"gpuwm fetch-tables: REFUSED: {error}")
                return 2
            absent.remove(asset)
            copied.append(asset)
        # mp=28 reads CCN_ACTIVATE.BIN out of the SAME root as the
        # classic tables -- the aerosol contract shares the root on
        # purpose, so tnccn_act and the process tables cannot come from
        # two different WRF builds.  It is not a member of
        # CLASSIC_TABLE_ASSETS, so the loop above never carried it, and
        # a staged root without it is complete for mp=8 and fatal for
        # mp=28: on a wheel install `fetch-tables` reported the root
        # "complete and byte-valid" and the next mp=28 forecast died at
        # its first microphysics step with MissingAerosolTableAsset,
        # while the packaged copy sat in site-packages the whole time.
        # Copied under the identical pin verification; a copy already
        # present is left alone.
        from gpuwm.core.thompson_aerosol_contract import AEROSOL_TABLE_ASSETS
        for asset in AEROSOL_TABLE_ASSETS:
            if (root / asset.filename).is_file():
                continue
            if not (packaged / asset.filename).is_file():
                continue
            print(f"gpuwm fetch-tables: copying {asset.filename} "
                  f"({asset.bytes:,} B, the mp=28 activation table) from "
                  f"the package into {root}")
            try:
                fetch_asset_from_dir(root, asset, packaged)
            except TableAssetError as error:
                print(f"gpuwm fetch-tables: REFUSED: {error}")
                return 2
            copied.append(asset)

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
    if not fetchable and not copied:
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


def fetch_tables_main(args) -> int:
    """Run the mandatory table leg and the optional ``--wif`` leg.

    The two legs are INDEPENDENT.  ``--wif`` names an extra optional
    dataset; it does not change whether the coefficient tables every run
    needs get staged.  This used to run the WIF leg first and return on
    its exit code, so `gpuwm fetch-tables --wif` against an unreachable
    dataset refused and left the mandatory tables unstaged -- the user
    asked for both and got neither, with nothing in the output saying the
    classic leg had never been attempted.

    The exit code is the worst of the legs that ran, and the closing
    summary names each leg as staged or refused, so "did my tables land"
    is answered by reading the last line rather than by inferring it from
    a single number.
    """

    wif_requested = bool(getattr(args, "wif", False))
    wif_only = bool(getattr(args, "wif_only", False))

    wif_code: int | None = None
    if wif_requested:
        wif_code = stage_wif_dataset(getattr(args, "from_dir", None),
                                     getattr(args, "wif_root", None))

    # --wif-only is the one flag that legitimately skips the mandatory
    # leg, because it is the operator saying so in as many words.
    classic_code: int | None = None
    if not wif_only:
        classic_code = stage_classic_tables(args)

    legs = []
    if classic_code is not None:
        legs.append(("classic coefficient tables", classic_code))
    if wif_code is not None:
        legs.append(("--wif aerosol climatology dataset", wif_code))
    if len(legs) > 1 or (wif_requested and not wif_only):
        for label, code in legs:
            state = "staged" if code == 0 else "REFUSED"
            print(f"gpuwm fetch-tables: {label}: {state}")
    return max((code for _, code in legs), default=0)


def stage_wif_dataset(source_dir=None, root=None) -> int:
    """Stage the mp=28 WIF climatology dataset under the table contract.

    The bytes are verified against
    :data:`gpuwm.ingest.wif_dataset.WIF_DATASET_ASSET` -- exact size then
    SHA-256 -- BEFORE the atomic move, an existing file with the wrong
    bytes is refused rather than overwritten, and a present, valid file
    costs one hash and exits 0.  That is ``_stage``'s contract unchanged;
    only the root and the pin differ, because the reason this file is
    externalized is the same reason ``freezeH2O.dat`` is.
    """

    from gpuwm.ingest.wif_dataset import (
        WIF_DATASET_ASSET, WIF_DATASET_FILE, resolve_wif_data_root)

    wif_root = resolve_wif_data_root(root)
    try:
        wif_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"gpuwm fetch-tables --wif: cannot create WIF data root "
              f"{wif_root}: {error}")
        return 2
    final = wif_root / WIF_DATASET_FILE
    mib = WIF_DATASET_ASSET.bytes / (1024 * 1024)
    if final.is_file():
        try:
            _verify_in_place(final, WIF_DATASET_ASSET)
        except TableAssetError as error:
            print(f"gpuwm fetch-tables --wif: REFUSED: {error} -- an "
                  "existing file is never overwritten; delete it and re-run")
            return 2
        print(f"gpuwm fetch-tables --wif: {final} already staged and "
              f"byte-valid ({WIF_DATASET_ASSET.bytes:,} B, SHA-256 "
              f"{WIF_DATASET_ASSET.sha256[:12]}...); nothing to fetch")
        return 0
    try:
        if source_dir is not None:
            print(f"gpuwm fetch-tables --wif: staging {WIF_DATASET_FILE} "
                  f"({mib:.1f} MiB) from {source_dir}")
            fetch_asset_from_dir(wif_root, WIF_DATASET_ASSET,
                                 Path(source_dir))
        else:
            url = f"{wif_asset_url_base()}/{WIF_DATASET_FILE}"
            print(f"gpuwm fetch-tables --wif: downloading "
                  f"{WIF_DATASET_FILE} ({mib:.1f} MiB) from {url}")
            fetch_asset_from_url(wif_root, WIF_DATASET_ASSET, url)
    except TableAssetError as error:
        print(f"gpuwm fetch-tables --wif: REFUSED: {error}")
        return 2
    print(f"gpuwm fetch-tables --wif: verified and installed {final} "
          f"({WIF_DATASET_ASSET.bytes:,} B, SHA-256 "
          f"{WIF_DATASET_ASSET.sha256[:12]}...)")
    return 0


def _verify_in_place(path: Path, asset: TableAsset) -> None:
    """Size then SHA-256 of a file already in its final location."""

    actual_bytes = path.stat().st_size
    if actual_bytes != asset.bytes:
        raise TableAssetError(
            f"{path}: {actual_bytes} bytes, expected {asset.bytes}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(_BLOCK_BYTES):
            digest.update(block)
    if digest.hexdigest() != asset.sha256:
        raise TableAssetError(
            f"{path}: SHA-256 {digest.hexdigest()}, expected {asset.sha256}")


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "fetch-tables",
        help="download the externalized physics table assets (currently "
             "the 243 MiB Thompson freezeH2O.dat) into ~/.gpuwm/tables/"
             "thompson -- outside the install, so a wheel upgrade cannot "
             "delete them -- SHA-256-verified against the packaged pins "
             "before install; idempotent when everything is already "
             "present")
    parser.add_argument(
        "--from", dest="from_dir", metavar="DIR", default=None,
        help="stage from a local directory instead of downloading "
             "(offline installs); verification is identical")
    parser.add_argument(
        "--wif", action="store_true",
        help="also stage QNWFA_QNIFA_SIGMA_MONTHLY.dat (215 MiB), the "
             "global monthly aerosol climatology the mp_physics=28 WIF "
             "ingest reads (aer_init_opt=1 with wif_input_opt=1), into "
             "~/.gpuwm/wif under the same SHA-256 contract.  Opt-in: it "
             "is an input dataset, not a coefficient table, and no "
             "default install opens it")
    parser.add_argument(
        "--wif-only", dest="wif_only", action="store_true",
        help="with --wif, stage only that dataset and leave the "
             "coefficient tables alone")
    parser.add_argument(
        "--wif-root", dest="wif_root", metavar="DIR", default=None,
        help="stage the WIF dataset into DIR instead of ~/.gpuwm/wif "
             "(same meaning as GPUWM_WIF_DATA_ROOT)")
    parser.set_defaults(func=fetch_tables_main)
    return parser


__all__ = [
    "ASSET_URL_BASE_ENV",
    "EXTERNALIZED_TABLE_FILENAMES",
    "MissingTableAssets",
    "RELEASE_ASSET_BASE_URL",
    "TableAssetError",
    "asset_url_base",
    "classify_assets",
    "fetch_asset_from_dir",
    "fetch_asset_from_url",
    "fetch_tables_main",
    "missing_externalized_assets",
    "register_cli",
    "require_thompson_tables",
    "staging_is_self_chosen",
    "stage_classic_tables",
    "stage_wif_dataset",
    "staging_root",
    "unstaged_table_assets",
    "wif_asset_url_base",
]
