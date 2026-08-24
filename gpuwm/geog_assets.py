"""``gpuwm fetch-geog``: download and stage the WPS_GEOG static tree.

The WRF static builder opens nine fixed WPS_GEOG dataset directories
(:data:`gpuwm.domain_wizard.GEOG_DATASETS`); until this command existed,
staging them was entirely manual.  ``fetch-geog`` downloads the dataset
tarballs, verifies every byte, extracts them into the standard layout,
and validates each dataset the same way ``gpuwm doctor`` does (the
``index`` file must exist -- here it must also parse).

The pin table is wider than those nine: the MPAS static builder
(``rw_mpas_static``) also reads the Noah-MP soil products, which arrive
as one ``soilgrids`` archive holding seven dataset directories under a
common parent.  A pin declares such children in ``index_subdirs`` so
validation looks for the index where it actually is, which keeps a
multi-dataset product table work rather than a second code path.

WHICH DOOR needs a pin is a column of that table too
(:data:`GEOG_CONSUMERS`, and each pin's ``required_by``), not a branch
in the code that reads it.  ``wrf`` is the WRF static builder's nine;
``mesh`` is what ``gpuwm mesh`` must open to write the static half of
its pair, which is seven of those nine plus the soil archive.  Every
reader -- the default fetch set, ``--datasets wrf``, and each of
``gpuwm doctor``'s two completeness verdicts -- is that column read at
a different scope, so adding a dataset for a future door is a row and a
consumer name rather than a new code path.

The soil archive is a REQUIRED download for the mesh door and no part
of a WRF-only install.  It is large (864 MB compressed, ~13 GB
unpacked) and the default fetch stages it anyway, because a tree the
installer called complete and ``gpuwm mesh`` then refuses is worse than
a long download nobody was surprised by.  ``--datasets wrf`` is the
spelling for an operator who knows the mesh door is not theirs.

Two download sources serve byte-identical archives:

* ``hf`` (default) -- the ArWen mirror on Hugging Face
  (:data:`HF_MIRROR_BASE_URL`), a verbatim copy of the NCAR per-dataset
  tarballs republished for CDN bandwidth with attribution.  Every
  archive is verified against the size + SHA-256 pins in
  :data:`GEOG_ARCHIVES` before extraction.
* ``ncar`` -- the upstream NCAR server (:data:`NCAR_BASE_URL`), a single
  host that can be slow.  NCAR publishes no checksums, so the same pins
  apply; they were computed from a TLS download from UCAR on
  2026-07-29, which is the honest limit of first-download integrity for
  this source (see docs/public/DATA.md).  If NCAR republishes an
  archive the pin mismatch refuses loudly; ``--allow-upstream-drift``
  accepts the new bytes when the size stays inside a sanity band, and
  records the archive as unpinned in the local manifest.

``--bundle`` fetches NCAR's single ``geog_high_res_mandatory.tar.gz``
(2.6 GB compressed, ~29 GB unpacked) instead of the per-dataset
tarballs and extracts only the requested datasets from it.  The bundle
exists only on NCAR.  The per-dataset route downloads half as many
bytes (~1.3 GB) and is the default.

Downloads are resumable (HTTP Range against the recorded partial), a
complete verified archive or an already-valid dataset directory is
never re-downloaded, and every fetch appends to a local
``geog-fetch-manifest.json`` recording url/bytes/sha256 per archive for
future re-verification.

No case is named here; WPS_GEOG is a public static dataset, not a case.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import tarfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from gpuwm import fetch_guard

GEOG_FETCH_MANIFEST_SCHEMA = "gpuwm-geog-fetch-manifest-v1"
GEOG_FETCH_MANIFEST_NAME = "geog-fetch-manifest.json"

#: The ArWen WPS_GEOG mirror: a Hugging Face dataset repo holding the
#: nine NCAR per-dataset tarballs byte-for-byte (plus a MANIFEST.sha256
#: and provenance README).  Resolve URLs are plain HTTPS behind a CDN;
#: no account or hub library is needed to download.
HF_MIRROR_REPO = "deepguess/wps-geog-arwen"
HF_MIRROR_BASE_URL = (
    f"https://huggingface.co/datasets/{HF_MIRROR_REPO}/resolve/main")

#: Upstream: the NCAR WPS geographical-data server (single host; the
#: index page is get_sources_wps_geog.html).  No checksums published.
NCAR_BASE_URL = "https://www2.mmm.ucar.edu/wrf/src/wps_files"

#: Override the download base URL (private mirrors, tests).  The archive
#: filename is appended; the override wins over --source's base.
GEOG_URL_BASE_ENV = "GPUWM_GEOG_URL_BASE"

GEOG_SOURCES = ("hf", "ncar")

#: The doors a pin can be required by.  A door a user can type, never a
#: builder's internal name: ``wrf`` is the WRF static path (``gpuwm
#: prepare``/``gpuwm go``, via ``rw-wps``) and ``mesh`` is ``gpuwm
#: mesh``'s static half (``rw_mpas_static``).  A consumer name is also
#: accepted by ``--datasets``, so the scoping is selectable and not just
#: declarative.
GEOG_CONSUMER_WRF = "wrf"
GEOG_CONSUMER_MESH = "mesh"
GEOG_CONSUMERS = (GEOG_CONSUMER_WRF, GEOG_CONSUMER_MESH)

#: Subdirectory of the geog root that holds in-flight and verified
#: archives (kept only under --keep-archives after extraction).
ARCHIVE_SUBDIR = ".fetch-geog"

#: With --allow-upstream-drift, an unpinned NCAR archive is still
#: refused outside [pin/2, pin*2] -- wide enough for a republished
#: dataset, narrow enough to catch an error page or truncation.
DRIFT_SIZE_BAND = (0.5, 2.0)

_BLOCK_BYTES = 8 * 1024 * 1024
_USER_AGENT = "gpuwm-fetch-geog/1"
_TIMEOUT_S = 120


@dataclass(frozen=True)
class GeogArchive:
    """One WPS_GEOG dataset tarball, pinned by size and SHA-256.

    ``dataset`` is the extracted top-level directory name (equal to the
    entry in :data:`gpuwm.domain_wizard.GEOG_DATASETS`); ``filename`` is
    the tarball name, identical on NCAR and the mirror.  The pins were
    computed from the NCAR bytes on 2026-07-29 (sizes independently
    confirmed against the server's Content-Length); the mirror serves
    the same bytes, so one pin covers both sources.
    ``extracted_bytes`` is the measured sum of member sizes -- sizing
    information for humans, not a verification bar.

    ``index_subdirs`` names the child directories that carry the WPS
    ``index`` when one archive ships several datasets under a common
    top-level directory.  Most archives are a single dataset and leave it
    empty, which keeps the index at ``<dataset>/index``.  Declaring the
    children here rather than special-casing an archive name is what lets
    a future multi-dataset product be table work instead of a new code
    path.

    ``in_mandatory_bundle`` records whether NCAR's single mandatory-fields
    bundle carries this dataset.  That is a measured fact about that one
    archive, so a pin whose presence in it nobody has verified says False
    and ``--bundle`` refuses the dataset up front instead of extracting
    nothing and calling it done.

    ``required_by`` names the doors that cannot open without this pin,
    from :data:`GEOG_CONSUMERS`.  It replaced a single ``optional`` flag,
    which could only say "somebody needs this" and "nobody does" -- and
    the archive nine-tenths of installs never open is the one the mesh
    door hard-requires, so that flag made ``gpuwm doctor`` choose between
    demanding 13 GB of a WRF-only box and calling a tree complete that
    ``gpuwm mesh`` refuses.  Scoped by consumer, each verdict reads its
    own column and both are right.  Every pin is required by at least one
    door; a pin no door names would be a download with no reader.
    """

    dataset: str
    filename: str
    archive_bytes: int
    archive_sha256: str
    extracted_bytes: int
    index_subdirs: tuple[str, ...] = ()
    in_mandatory_bundle: bool = True
    required_by: tuple[str, ...] = (GEOG_CONSUMER_WRF,)


#: Both doors need this pin; only the WRF static builder does; only the
#: mesh one does.  Spelled out here so each row's scoping reads as a
#: column of the table rather than as an argument list.
_WRF_AND_MESH = (GEOG_CONSUMER_WRF, GEOG_CONSUMER_MESH)
_WRF_ONLY = (GEOG_CONSUMER_WRF,)
_MESH_ONLY = (GEOG_CONSUMER_MESH,)

#: The ten archives, in GEOG_DATASETS order with the mesh-only soil
#: container last.  A test binds the ``wrf``-scoped names to
#: gpuwm.domain_wizard.GEOG_DATASETS and the ``mesh``-scoped ones to
#: gpuwm.rustwx_static.REQUIRED_GEOG_DATASETS through the child
#: declarations, so neither column can drift from the builder it speaks
#: for.
GEOG_ARCHIVES: tuple[GeogArchive, ...] = (
    GeogArchive(
        "topo_gmted2010_30s", "topo_gmted2010_30s.tar.bz2", 182721518,
        "20e02ef61b16c8d66753db8f1ede0edc0ac74e6e1b87961cdbdc483040ff22d7",
        1884949355, required_by=_WRF_AND_MESH),
    GeogArchive(
        "modis_landuse_20class_30s_with_lakes",
        "modis_landuse_20class_30s_with_lakes.tar.bz2", 32532921,
        "fe052cc1000638c32ac7191c779ad070efae458ff0751bc4aa2a716e38bec578",
        933120358, required_by=_WRF_AND_MESH),
    GeogArchive(
        "soiltype_top_30s", "soiltype_top_30s.tar.bz2", 1639561,
        "6d45d93cc0d14ac4019f1a2857e56fb1fbc860f135e12d89fc0a569278242315",
        933120270, required_by=_WRF_AND_MESH),
    GeogArchive(
        "soiltype_bot_30s", "soiltype_bot_30s.tar.bz2", 1919818,
        "818febf2d141a46152f0990877e3ca15fd23a61961d978f71f5fdd0ea80bb335",
        933120273, required_by=_WRF_ONLY),
    GeogArchive(
        "greenfrac_fpar_modis", "greenfrac_fpar_modis.tar.bz2", 990624380,
        "a31d44c7b0c3dfcbaf933b3c30b4c6e52762a3451cf4e1951d7b49f3751b4d03",
        11197440257, required_by=_WRF_AND_MESH),
    GeogArchive(
        "lai_modis_10m", "lai_modis_10m.tar.bz2", 2604794,
        "7dfe8e2a7c48c085c8d180ffda993ce878b3c054ed7e5c185414c1993b913ebf",
        30863207, required_by=_WRF_ONLY),
    GeogArchive(
        "albedo_modis", "albedo_modis.tar.bz2", 77919797,
        "cf0d3df8e6b07a2dc7222a0cbc023205a991859395292dc6161597faf02f8e6f",
        628316651, required_by=_WRF_AND_MESH),
    GeogArchive(
        "maxsnowalb_modis", "maxsnowalb_modis.tar.bz2", 4446019,
        "eca29b1707e184c942eb8f1bfdbafbb4e6650cb97c15784c293fda39e24c1371",
        52359972, required_by=_WRF_AND_MESH),
    GeogArchive(
        "soiltemp_1deg", "soiltemp_1deg.tar.bz2", 31057,
        "c407c0b1d9d864ff49708fa3b6ba2f17078535ccfc05a944024ad052e5f914d4",
        138645, required_by=_WRF_AND_MESH),
    # The Noah-MP soil products.  One upstream archive carries seven
    # dataset directories, each with its own WPS index, so the pin
    # declares the children rather than assuming an index at the top.
    # rw_mpas_static reads soilcomp for --soilcomp and texture_layer1..4
    # for --soilcl1..4; texture_top and texture_bot are the surface and
    # bottom composites that ship alongside them.  The WRF static builder
    # opens none of these, which is why GEOG_DATASETS stays at nine and
    # why this row is scoped to the mesh door alone: a WRF-only install
    # is not short of anything without it.  For the mesh door it is
    # REQUIRED, not optional -- the one static writer emits the five
    # soil-composition variables and cannot invent them.
    # Pinned from a TLS download from UCAR on 2026-08-23, the same way
    # and from the same host as the nine above.
    GeogArchive(
        "soilgrids", "soilgrids.tar.bz2", 864090244,
        "3d42737a68a52a2be10281f0505c6cb6c05c03707e9382b3a823495afc7937bb",
        13138518107,
        ("soilcomp", "texture_top", "texture_bot",
         "texture_layer1", "texture_layer2",
         "texture_layer3", "texture_layer4"),
        in_mandatory_bundle=False, required_by=_MESH_ONLY),
)

#: NCAR's "highest resolution mandatory fields" bundle: the fallback
#: when the per-dataset tarballs are unavailable.  NCAR-only.
MANDATORY_BUNDLE_FILENAME = "geog_high_res_mandatory.tar.gz"
MANDATORY_BUNDLE_BYTES = 2772782816
MANDATORY_BUNDLE_SHA256 = \
    "89b026b9db0a03c0c995e53b4a1d99663af1f6bda21b3b34c3c2c07386da5493"
#: Datasets of the nine that the bundle actually contains (verified by
#: walking the archive on 2026-07-29: 22,783 files, 30.7 GB unpacked,
#: everything nested under one WPS_GEOG/ directory the extractor
#: strips).  A requested dataset outside this set cannot be served by
#: --bundle and refuses up front.
MANDATORY_BUNDLE_DATASETS: frozenset[str] = frozenset(
    archive.dataset for archive in GEOG_ARCHIVES
    if archive.in_mandatory_bundle)


class GeogFetchError(RuntimeError):
    """An operational fetch failure with its remedy in the message."""


def geog_datasets() -> tuple[str, ...]:
    """Every dataset ``fetch-geog`` can stage, in canonical order."""
    return tuple(archive.dataset for archive in GEOG_ARCHIVES)


def datasets_required_by(consumer: str) -> tuple[str, ...]:
    """The pins ``consumer``'s door cannot open without, in table order.

    The scoped answer replaced a single "required set" that had to be
    true for every door at once.  There is no such set: the WRF static
    builder never opens the soil container and the mesh static builder
    hard-requires it, so one list could only overstate one door's needs
    or understate the other's.  ``gpuwm doctor`` asks per consumer and
    reports two verdicts that do not contradict each other.
    """

    if consumer not in GEOG_CONSUMERS:
        raise ValueError(f"unknown geography consumer {consumer!r}; "
                         f"expected one of {', '.join(GEOG_CONSUMERS)}")
    return tuple(archive.dataset for archive in GEOG_ARCHIVES
                 if consumer in archive.required_by)


def size_phrase(datasets: tuple[str, ...] | None = None) -> str:
    """``'~2.2 GB compressed, ~30 GB unpacked'`` for a set of pins.

    ``None`` means the default fetch set, which is every pin.  Decimal
    GB, matching every other published size in this product.

    Published because three surfaces quote these numbers before the
    first byte moves -- ``gpuwm setup --with-geog``, the notice printed
    when setup skips the tree, and ``gpuwm doctor``'s remedy -- and all
    three carried the figure as a literal.  Adding a pin left every one
    of them understating the download by most of its size, in the exact
    place a reader is deciding whether to start it.
    """

    chosen = (GEOG_ARCHIVES if datasets is None
              else tuple(archive_for(name) for name in datasets))
    compressed = sum(archive.archive_bytes for archive in chosen) / 1e9
    unpacked = sum(archive.extracted_bytes for archive in chosen) / 1e9
    return (f"~{compressed:.1f} GB compressed, "
            f"~{unpacked:.0f} GB unpacked")


def archive_for(dataset: str) -> GeogArchive:
    for archive in GEOG_ARCHIVES:
        if archive.dataset == dataset:
            return archive
    raise ValueError(f"unknown WPS_GEOG dataset {dataset!r}")


def parse_datasets(raw: str) -> tuple[str, ...]:
    """Parse ``--datasets`` into dataset names, in canonical order.

    Three spellings, all read off the same table: ``all`` (the default,
    every pin -- the soil container included, which is the ruling that
    made it a sanctioned default download), a consumer name from
    :data:`GEOG_CONSUMERS` standing for that door's whole set, and an
    explicit comma list.  They mix, and duplicates collapse.

    ``--datasets wrf`` is the subset spelling for an operator who knows
    the mesh door is not theirs and does not want its ~13 GB.  It is a
    selector over the scoping column rather than a ``--skip`` flag,
    because a flag would have to be re-argued for every future door and
    this stays a row in the table.
    """

    known = geog_datasets()
    if raw.strip().lower() == "all":
        return known
    requested = [part.strip() for part in raw.split(",") if part.strip()]
    if not requested:
        raise ValueError("--datasets must be 'all', a consumer name "
                         f"({', '.join(GEOG_CONSUMERS)}), or a "
                         "comma-separated list of dataset names")
    picked: set[str] = set()
    unknown: list[str] = []
    for name in requested:
        if name in GEOG_CONSUMERS:
            picked.update(datasets_required_by(name))
        elif name in known:
            picked.add(name)
        else:
            unknown.append(name)
    if unknown:
        raise ValueError(
            f"unknown dataset(s) {', '.join(sorted(set(unknown)))}; expected "
            f"'all', a consumer name ({', '.join(GEOG_CONSUMERS)}), or names "
            f"from: {', '.join(known)}")
    return tuple(name for name in known if name in picked)


def resolve_source(source: str | None, bundle: bool) -> str:
    """Concrete source for one invocation; ``--bundle`` is NCAR-only."""

    if source is not None and source not in GEOG_SOURCES:
        raise ValueError(f"unknown --source {source!r}; expected one of "
                         f"{GEOG_SOURCES}")
    if bundle:
        if source == "hf":
            raise ValueError(
                "--bundle is served by NCAR only (the mirror republishes "
                "the per-dataset tarballs); drop --bundle or use "
                "--source ncar")
        return "ncar"
    return source if source is not None else "hf"


def archive_url(filename: str, source: str) -> str:
    """Download URL for ``filename`` from ``source`` (env override wins)."""

    override = os.environ.get(GEOG_URL_BASE_ENV)
    if override and override.strip():
        return f"{override.strip().rstrip('/')}/{filename}"
    if source == "hf":
        return f"{HF_MIRROR_BASE_URL}/{filename}"
    if source == "ncar":
        return f"{NCAR_BASE_URL}/{filename}"
    raise ValueError(f"unknown source {source!r}")


def default_geog_root() -> Path:
    """The root fetch-geog stages into when ``--root`` is not given.

    Exactly the tree the rest of gpuwm already reads:
    ``$GPUWM_CASE_DATA_ROOT/WPS_GEOG`` -- the default ``geog_root`` in
    wizard-emitted configs and the path ``gpuwm doctor`` checks -- with
    the same unset-variable fallback as every other case-data path
    (:func:`gpuwm.case_data.case_data_root`).
    """

    from gpuwm.case_data import case_data_root

    return case_data_root() / "WPS_GEOG"


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def dataset_corpus(root: Path, dataset: str) -> tuple[int, int]:
    """``(regular file count, total bytes)`` of one installed dataset.

    A cheap stat walk: the tile trees run to gigabytes, so re-hashing
    them at every use is not a bar anybody would leave switched on.
    """

    files = 0
    total = 0
    for path in (root / dataset).rglob("*"):
        if path.is_file():
            files += 1
            total += path.stat().st_size
    return files, total


def installed_receipt(root: Path, dataset: str) -> dict | None:
    """What the local manifest says was installed for ``dataset``.

    Returns the ``{"files": n, "bytes": n}`` entry recorded when the
    dataset was extracted, or ``None`` when no manifest claims it (an
    install from before this receipt existed, or a directory some other
    tool staged).
    """

    manifest = _load_manifest(root)
    for entry in manifest.get("archives", {}).values():
        if not isinstance(entry, dict):
            continue
        datasets = entry.get("datasets")
        if not isinstance(datasets, dict):
            continue
        recorded = datasets.get(dataset)
        if (isinstance(recorded, dict)
                and isinstance(recorded.get("files"), int)
                and isinstance(recorded.get("bytes"), int)):
            return recorded
    return None


def validate_dataset_dir(root: Path, dataset: str, *,
                         check_receipt: bool = True) -> tuple[bool, str]:
    """(ok, detail) for one staged dataset directory.

    The bar is doctor's own -- the directory exists and carries its WPS
    ``index`` file -- plus the index must actually parse
    (:func:`gpuwm.static.geog.parse_index`), so a zero-byte or truncated
    index cannot pass.

    And, when the local manifest recorded what was installed, the tile
    corpus must still match that receipt.  Index-only validation was the
    whole bar until now, which meant a directory whose *tiles* were
    corrupt, truncated, deleted or added to was classified valid beside
    an intact ``index`` and reused indefinitely -- fetch returned zero
    work and never looked at the manifest.  Comparing the file count and
    total byte count against the extraction receipt catches missing,
    extra and short tiles for the cost of a stat walk.  It does not
    catch a same-size mutation of a tile's contents: that needs a full
    content digest of a multi-gigabyte tree, which is a deliberate
    verification mode rather than something to run on every command.

    ``check_receipt=False`` asks for the historical index-only bar, for
    callers validating a tree they have just staged themselves.
    """

    directory = root / dataset
    if not directory.is_dir():
        return False, f"{directory} does not exist"
    try:
        subdirs = archive_for(dataset).index_subdirs
    except ValueError:
        subdirs = ()
    # A single-dataset archive carries one index at the top; a container
    # archive carries one per declared child.  Every index named here has
    # to pass, so a container missing one child is a finding, not a pass
    # earned by its siblings.
    for index in ([directory / "index"] if not subdirs
                  else [directory / name / "index" for name in subdirs]):
        if not index.is_file():
            return False, f"{index.parent} lacks its WPS `index` file"
        try:
            from gpuwm.static.geog import parse_index
            parsed = parse_index(index)
            # parse_index defaults absent keys to None; a truncated or
            # foreign file "parses" vacuously, so require the fields no
            # real WPS index omits, and a decodable word size.
            absent = [key for key in ("dx", "dy", "known_lat", "known_lon",
                                      "wordsize", "tile_x", "tile_y")
                      if getattr(parsed, key) is None]
            if absent:
                return False, (f"{index} lacks required WPS index key(s): "
                               + ", ".join(absent))
            parsed.dtype  # noqa: B018 -- raises on an undecodable wordsize
        except Exception as error:  # any parse failure is the finding
            return False, f"{index} does not parse as a WPS index: {error}"
    if not check_receipt:
        return True, f"{directory} present, index parses"
    receipt = installed_receipt(root, dataset)
    if receipt is None:
        return True, (f"{directory} present, index parses (no local "
                      f"{GEOG_FETCH_MANIFEST_NAME} entry to check the "
                      "tile corpus against)")
    files, total = dataset_corpus(root, dataset)
    if (files, total) != (receipt["files"], receipt["bytes"]):
        return False, (
            f"{directory} no longer matches what "
            f"{GEOG_FETCH_MANIFEST_NAME} records was installed: "
            f"{files} files / {total:,} B on disk, "
            f"{receipt['files']} files / {receipt['bytes']:,} B recorded "
            "(a tile is missing, truncated, or added)")
    return True, (f"{directory} present, index parses, {files} tiles / "
                  f"{total:,} B match the recorded install")


def _quarantine(path: Path, progress, label: str) -> Path:
    """Move a refused file aside (never deleted, never reused).

    The name is proven free first: ``.rejected-<time_ns>`` on its own
    collides inside a clock tick, and ``os.replace`` onto the collision
    destroys the older evidence.
    """

    aside = fetch_guard.quarantine(path)
    progress(f"fetch-geog {label}: moved rejected file aside to "
             f"{aside.name}")
    return aside


# ---------------------------------------------------------------------------
# Download (resumable)
# ---------------------------------------------------------------------------

def _default_urlopen(request: Request):
    return urlopen(request, timeout=_TIMEOUT_S)


#: Sidecar recording which object a partial archive is a prefix OF.
_RESUME_SIDECAR_SUFFIX = ".fetch-resume.json"


def _resume_sidecar(dest: Path) -> Path:
    return dest.with_name(dest.name + _RESUME_SIDECAR_SUFFIX)


def _resume_identity(response, url: str) -> dict:
    headers = getattr(response, "headers", None)
    getter = headers.get if headers is not None else (lambda _key: None)
    return {
        "url": url,
        "etag": getter("ETag"),
        "last_modified": getter("Last-Modified"),
    }


def _content_range_start(value: str) -> int | None:
    """The first byte a ``206`` says it is sending, or None."""

    text = (value or "").strip()
    if not text.lower().startswith("bytes "):
        return None
    span = text[6:].split("/", 1)[0].strip()
    first = span.split("-", 1)[0].strip()
    try:
        return int(first)
    except ValueError:
        return None


def download_archive(url: str, dest: Path, *, expected_bytes: int,
                     progress, label: str, strict_size: bool = True,
                     urlopen_fn=_default_urlopen) -> None:
    """Download ``url`` to ``dest`` with Range resume and a size bar.

    A ``dest`` left by an interrupted run resumes from its byte count
    (both NCAR and the mirror serve ranges).  A server that ignores the
    Range header restarts cleanly.  With ``strict_size`` (the pinned
    path) the final size must equal ``expected_bytes`` -- a mismatch
    refuses and quarantines rather than handing a short file to the
    hash check with a confusing story.  ``--allow-upstream-drift``
    turns ``strict_size`` off, deferring to the verify step's sanity
    band (a republished archive rarely keeps its byte count).

    Resume is bound to an *object*, not just to a byte count.  A partial
    file used to be resumed from its length alone: nothing checked that
    the bytes about to be appended came from the same URL, the same
    version, or even the requested offset, so a republished or
    reconfigured upstream could produce a chimera stitched out of two
    generations.  The exact pin normally catches that; under
    ``--allow-upstream-drift`` -- where the size band *is* the whole
    integrity policy -- a same-size chimera passed.  Now the first
    response's identity (URL, ``ETag``, ``Last-Modified``) is recorded
    in a sidecar, a resume must match it and is sent with ``If-Range``,
    and a ``206`` whose ``Content-Range`` does not start at the
    requested offset is refused.  A mismatch restarts from zero after
    setting the partial aside; nothing is deleted.
    """

    offset = dest.stat().st_size if dest.exists() else 0
    sidecar = _resume_sidecar(dest)
    recorded: dict = {}
    if offset:
        try:
            loaded = json.loads(sidecar.read_text(encoding="utf-8"))
            recorded = loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError):
            recorded = {}
        bound_to_another = bool(recorded) and recorded.get("url") != url
        # A partial with no record at all is either a pre-1.1.3 resume
        # or something else's file.  On the pinned path the exact
        # size+SHA-256 pin is a complete integrity policy and catches any
        # mixture, so such a partial is still worth resuming.  Under
        # --allow-upstream-drift there is no pin -- the size band IS the
        # policy, and a same-size chimera passes it -- so an unbound
        # partial has to start again.
        unbound_and_unpinned = not recorded and not strict_size
        if bound_to_another or unbound_and_unpinned:
            progress(
                f"fetch-geog {label}: the partial archive on disk is not a "
                f"recorded prefix of {url} "
                + (f"(it was recorded against {recorded.get('url')!r})"
                   if bound_to_another else
                   "(no resume record, and --allow-upstream-drift has no "
                   "pin to catch a mixed archive)")
                + "; setting it aside and downloading from the start")
            _quarantine(dest, progress, label)
            sidecar.unlink(missing_ok=True)
            offset = 0
            recorded = {}
    if strict_size and offset > expected_bytes:
        _quarantine(dest, progress, label)
        sidecar.unlink(missing_ok=True)
        offset = 0
        recorded = {}
    if strict_size and offset == expected_bytes:
        return
    headers = {"User-Agent": _USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        validator = recorded.get("etag") or recorded.get("last_modified")
        if validator:
            # If the object changed, the server answers 200 with the whole
            # new body and the restart below is taken -- which is exactly
            # the outcome wanted, decided by the origin rather than
            # guessed at here.
            headers["If-Range"] = validator
        progress(f"fetch-geog {label}: resuming at "
                 f"{offset / (1024 * 1024):.1f} MiB")
    request = Request(url, headers=headers)
    started = time.perf_counter()
    try:
        with urlopen_fn(request) as response:
            status = getattr(response, "status", 200)
            if offset and status == 206:
                response_headers = getattr(response, "headers", None)
                raw_range = ("" if response_headers is None
                             else response_headers.get("Content-Range") or "")
                start = _content_range_start(raw_range)
                if start != offset:
                    raise GeogFetchError(
                        f"{label}: resume asked for bytes {offset}- and the "
                        f"server answered 206 with Content-Range "
                        f"{raw_range!r}, which does not start there; the "
                        "partial file is kept and untouched -- move it "
                        "aside and re-run to download from the start")
            mode = "ab" if (offset and status == 206) else "wb"
            if offset and status != 206:
                progress(f"fetch-geog {label}: server ignored the resume "
                         "range (or the object changed); restarting the "
                         "download")
                recorded = {}
            if not recorded:
                fetch_guard.atomic_write_text(
                    sidecar,
                    json.dumps(_resume_identity(response, url),
                               indent=2, sort_keys=True) + "\n",
                    tag="geog")
            with dest.open(mode) as sink:
                while block := response.read(_BLOCK_BYTES):
                    sink.write(block)
    except HTTPError as error:
        hint = ""
        # HF answers 401 for a repo that does not exist (or is private
        # to anonymous callers), 404 for a missing file in a live repo.
        if error.code in (401, 403, 404) and "huggingface.co" in url:
            hint = ("  (the mirror archive is absent -- if the mirror "
                    "repo is not published yet, use --source ncar)")
        raise GeogFetchError(
            f"{label}: HTTP {error.code} from {url}{hint}; the partial "
            f"file is kept and a re-run resumes") from error
    except (URLError, OSError, TimeoutError) as error:
        raise GeogFetchError(
            f"{label}: download failed from {url}: {error}; the partial "
            "file is kept and a re-run resumes") from error
    size = dest.stat().st_size
    if strict_size and size != expected_bytes:
        _quarantine(dest, progress, label)
        sidecar.unlink(missing_ok=True)
        raise GeogFetchError(
            f"{label}: downloaded {size:,} B from {url}, expected "
            f"{expected_bytes:,} B; the file was moved aside -- re-run "
            "to fetch fresh (if this repeats, the upstream archive has "
            "changed; see --allow-upstream-drift)")
    # The archive is whole: the resume record has nothing left to bind.
    sidecar.unlink(missing_ok=True)
    seconds = time.perf_counter() - started
    rate = (size - offset) / (1024 * 1024) / seconds if seconds > 0 else 0.0
    progress(f"fetch-geog {label}: {dest.name} {size:,} B in "
             f"{seconds:.1f} s ({rate:.1f} MiB/s)")


def verify_archive(dest: Path, *, expected_sha256: str, expected_bytes: int,
                   source: str, allow_drift: bool, progress,
                   label: str) -> tuple[str, bool]:
    """(observed sha256, pinned?) for a completely downloaded archive.

    Pinned means the bytes match the packaged pin.  A mismatch refuses
    (quarantining the file) unless ``--allow-upstream-drift`` was given
    for an NCAR download whose size stays inside
    :data:`DRIFT_SIZE_BAND`; the mirror is our own publication, so
    drift there is corruption by definition and is never acceptable.
    """

    observed = sha256_file(dest)
    if observed == expected_sha256:
        return observed, True
    if source == "ncar" and allow_drift:
        size = dest.stat().st_size
        low = int(expected_bytes * DRIFT_SIZE_BAND[0])
        high = int(expected_bytes * DRIFT_SIZE_BAND[1])
        if not low <= size <= high:
            _quarantine(dest, progress, label)
            raise GeogFetchError(
                f"{label}: --allow-upstream-drift refused: {size:,} B is "
                f"outside the sanity band [{low:,}, {high:,}] around the "
                "pinned size -- that is an error page or truncation, not "
                "a republished dataset")
        progress(f"fetch-geog {label}: WARNING: accepting UNPINNED bytes "
                 f"under --allow-upstream-drift (sha256 {observed[:12]}..., "
                 f"pin {expected_sha256[:12]}...); recorded as unpinned in "
                 f"{GEOG_FETCH_MANIFEST_NAME}")
        return observed, False
    _quarantine(dest, progress, label)
    if source == "hf":
        raise GeogFetchError(
            f"{label}: sha256 {observed[:12]}... does not match the pin "
            f"{expected_sha256[:12]}... -- the mirror serves exactly the "
            "pinned bytes, so this is a corrupted transfer; the file was "
            "moved aside, re-run to fetch fresh")
    raise GeogFetchError(
        f"{label}: sha256 {observed[:12]}... does not match the pin "
        f"{expected_sha256[:12]}... computed from NCAR's bytes on "
        "2026-07-29.  Either the transfer corrupted (re-run fetches "
        "fresh; the mismatched file was moved aside) or NCAR republished "
        "the archive.  --source hf serves the pinned bytes; "
        "--allow-upstream-drift accepts NCAR's new bytes and records "
        "them as unpinned")


# ---------------------------------------------------------------------------
# Extraction (safe members only, staged then atomically placed)
# ---------------------------------------------------------------------------

def _safe_member(member: tarfile.TarInfo, datasets: tuple[str, ...],
                 archive_name: str) -> tuple[str, str] | None:
    """(dataset, relative path to extract) for a member, or None to skip.

    Only plain files and directories whose paths are relative, ``..``
    free, and rooted in a requested dataset directory are extracted;
    anything else (symlink, device, absolute path, traversal) is
    refused loudly -- these archives contain nothing of the sort, so
    their presence means a corrupted or hostile archive.

    The per-dataset tarballs carry their dataset directory at the top
    level (``soiltemp_1deg/index``); the mandatory bundle nests
    everything under one ``WPS_GEOG/`` directory
    (``WPS_GEOG/soiltemp_1deg/index`` -- verified by walking the
    archive).  Both shapes land identically: the extracted path always
    starts at the dataset directory.
    """

    name = member.name
    parts = [part for part in name.split("/") if part not in ("", ".")]
    if name.startswith(("/", "\\")) or ".." in parts or ":" in name:
        raise GeogFetchError(
            f"{archive_name} contains an unsafe member path {name!r}; "
            "refusing to extract")
    if not parts:
        return None
    if not (member.isfile() or member.isdir()):
        raise GeogFetchError(
            f"{archive_name} contains a non-regular member {name!r} "
            f"(type {member.type!r}); refusing to extract")
    # NCAR packed the mandatory bundle on macOS: it carries AppleDouble
    # ``._*`` sidecars (212 B resource-fork remnants) beside every
    # directory and tile.  They are not data; skipping them makes the
    # bundle and per-dataset routes produce byte-identical trees.
    if parts[-1].startswith("._") or parts[-1] == ".DS_Store":
        return None
    if parts[0] in datasets:
        return parts[0], "/".join(parts)
    if len(parts) > 1 and parts[0] == "WPS_GEOG" and parts[1] in datasets:
        return parts[1], "/".join(parts[1:])
    return None


def extract_datasets(archive: Path, root: Path, datasets: tuple[str, ...],
                     *, progress, label: str) -> dict[str, dict]:
    """Extract ``datasets`` from ``archive`` into ``root``.

    Members stream into a temporary directory beside the target; each
    completed dataset must carry a parsing ``index`` before it is
    renamed into place, so ``root`` never holds a half-extracted or
    index-less dataset directory.  Returns per-dataset
    ``{"files": n, "bytes": n}``.
    """

    tmp = root / f"{ARCHIVE_SUBDIR}-extract-{time.time_ns()}"
    tmp.mkdir(parents=True)
    counts: dict[str, dict] = {name: {"files": 0, "bytes": 0}
                               for name in datasets}
    started = time.perf_counter()
    try:
        with tarfile.open(archive, "r:*") as tar:
            for member in tar:
                placed = _safe_member(member, datasets, archive.name)
                if placed is None:
                    continue
                dataset, relative = placed
                target = tmp / relative
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise GeogFetchError(
                        f"{archive.name}: unreadable member {member.name}")
                with source, target.open("wb") as sink:
                    while block := source.read(_BLOCK_BYTES):
                        sink.write(block)
                counts[dataset]["files"] += 1
                counts[dataset]["bytes"] += member.size
    except tarfile.TarError as error:
        raise GeogFetchError(
            f"{label}: {archive.name} failed to extract: {error} -- "
            "the archive is corrupt; move it aside and re-run") from error

    for dataset in datasets:
        staged = tmp / dataset
        # Index-only here on purpose: this tree was just written by this
        # process and no receipt describes it yet -- the corpus check
        # belongs to the reuse path, against the receipt published below.
        ok, detail = validate_dataset_dir(tmp, dataset, check_receipt=False)
        if not ok:
            raise GeogFetchError(
                f"{label}: extracted dataset failed validation: {detail} "
                f"(staged under {tmp}, left for inspection)")
        final = root / dataset
        if final.exists():
            _quarantine(final, progress, label)
        os.replace(staged, final)
        progress(f"fetch-geog {label}: staged {final} "
                 f"({counts[dataset]['files']} files, "
                 f"{counts[dataset]['bytes']:,} B) in "
                 f"{time.perf_counter() - started:.1f} s")
    try:
        tmp.rmdir()  # only ever empty on success
    except OSError:
        pass
    return counts


# ---------------------------------------------------------------------------
# Local manifest
# ---------------------------------------------------------------------------

def _load_manifest(root: Path) -> dict:
    path = root / GEOG_FETCH_MANIFEST_NAME
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (isinstance(payload, dict)
                    and payload.get("schema") == GEOG_FETCH_MANIFEST_SCHEMA
                    and isinstance(payload.get("archives"), dict)):
                return payload
        except (ValueError, UnicodeDecodeError, OSError):
            pass
    return {"schema": GEOG_FETCH_MANIFEST_SCHEMA, "archives": {}}


def _write_manifest(root: Path, payload: dict) -> Path:
    payload["updated"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    return fetch_guard.atomic_write_text(
        root / GEOG_FETCH_MANIFEST_NAME,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        tag="geog")


def publish_archive_entry(root: Path, filename: str, entry: dict) -> Path:
    """Record one installed archive in the manifest, immediately.

    Two defects closed at once, and both need the read to happen here
    rather than at the start of the run.

    *Provenance survived nothing.*  Entries used to accumulate in memory
    and be published once, after every archive, and the verified archive
    was normally deleted before that.  A kill between the directory
    install and the final write lost the only record of where those
    tiles came from -- and the rerun classified the directory valid and
    returned before it could rebuild the entry.  Publishing per archive
    means the receipt is never further behind the disk than one archive.

    *Concurrent fetches lost each other's updates.*  Two processes each
    loaded the manifest at startup and each replaced it with its own
    version, so whichever finished last dropped the other's datasets
    while both directories sat installed and unrecorded.  Re-reading the
    canonical manifest here, under the geog-root lock, merges instead.
    """

    with fetch_guard.hold("fetch-geog", root):
        manifest = _load_manifest(root)
        manifest["archives"][filename] = entry
        return _write_manifest(root, manifest)


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------

def _print_listing(datasets: tuple[str, ...], source: str,
                   root: Path) -> None:
    total_archive = 0
    total_extracted = 0
    print(f"fetch-geog: root {root}")
    print(f"fetch-geog: source {source} "
          f"({archive_url('<archive>', source)})")
    for name in datasets:
        archive = archive_for(name)
        total_archive += archive.archive_bytes
        total_extracted += archive.extracted_bytes
        ok, _ = validate_dataset_dir(root, name)
        state = "staged" if ok else "needed"
        print(f"  {state}  {name}: {archive.filename} "
              f"{archive.archive_bytes / (1024 * 1024):.1f} MiB "
              f"(~{archive.extracted_bytes / (1024 * 1024):.0f} MiB "
              "unpacked)")
    print(f"fetch-geog: total {total_archive / (1024 ** 3):.2f} GiB "
          f"download, ~{total_extracted / (1024 ** 3):.1f} GiB unpacked; "
          f"bundle alternative {MANDATORY_BUNDLE_BYTES / (1024 ** 3):.2f} "
          "GiB download (--bundle, NCAR only)")


def _human_size(n: int) -> str:
    """``4096`` -> ``'4.0 KiB'``; adaptive binary units, one decimal."""
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")  # pragma: no cover


def _bytes_label(n: int) -> str:
    """Exact bytes first (they match Content-Length sums), human after."""
    exact = f"{n:,} B"
    return exact if n < 1024 else f"{exact} (~{_human_size(n)})"


def fetch_geog(*, root: Path, datasets: tuple[str, ...], source: str,
               bundle: bool = False, keep_archives: bool = False,
               allow_drift: bool = False, progress=print,
               urlopen_fn=_default_urlopen) -> int:
    """Stage ``datasets`` under ``root``; the fetch-geog engine.

    Returns the count of datasets newly staged (0 = everything was
    already present and valid).

    One writer per geog root: the classification, the archive
    downloads, the extractions and every manifest publication run under
    an exclusive OS lock, so two concurrent ``fetch-geog`` runs cannot
    quarantine each other's freshly installed directories or race on the
    same archive path.
    """

    root.mkdir(parents=True, exist_ok=True)
    with fetch_guard.hold("fetch-geog", root, progress=progress):
        return _fetch_geog_locked(
            root=root, datasets=datasets, source=source, bundle=bundle,
            keep_archives=keep_archives, allow_drift=allow_drift,
            progress=progress, urlopen_fn=urlopen_fn)


def _fetch_geog_locked(*, root: Path, datasets: tuple[str, ...], source: str,
                       bundle: bool, keep_archives: bool, allow_drift: bool,
                       progress, urlopen_fn) -> int:
    """The fetch-geog engine, with the geog-root lock already held."""

    needed: list[str] = []
    for name in datasets:
        ok, detail = validate_dataset_dir(root, name)
        if ok:
            progress(f"fetch-geog {name}: {detail} -- skipped")
        else:
            needed.append(name)
    if not needed:
        progress(f"fetch-geog: all {len(datasets)} requested datasets "
                 f"already staged and valid at {root}")
        return 0

    if bundle:
        outside = sorted(set(needed) - MANDATORY_BUNDLE_DATASETS)
        if outside:
            raise ValueError(
                f"--bundle cannot serve {', '.join(outside)}: the "
                "mandatory bundle does not contain them; use the "
                "per-dataset route (drop --bundle)")

    # The whole bill, announced before the first byte moves.  `gpuwm
    # setup --with-geog` promises "the size is printed before anything
    # downloads", and the engine both entry points share is the only
    # place that can keep that promise for `gpuwm fetch-geog` too --
    # the first field run watched 16 GB arrive with sizes printed only
    # as each dataset landed.  Covers exactly the datasets that still
    # need staging (the ones above were skipped as already valid);
    # under --bundle the download is the one bundle archive however few
    # datasets are extracted from it.
    download = (MANDATORY_BUNDLE_BYTES if bundle else
                sum(archive_for(name).archive_bytes for name in needed))
    unpacked = sum(archive_for(name).extracted_bytes for name in needed)
    progress(f"fetch-geog: {len(needed)} of {len(datasets)} requested "
             f"datasets need staging: {_bytes_label(download)} download, "
             f"~{_human_size(unpacked)} unpacked")

    archive_dir = root / ARCHIVE_SUBDIR
    archive_dir.mkdir(parents=True, exist_ok=True)
    staged = 0

    def run_archive(filename: str, expected_bytes: int,
                    expected_sha256: str, targets: tuple[str, ...],
                    label: str) -> None:
        nonlocal staged
        url = archive_url(filename, source)
        dest = archive_dir / filename
        if dest.exists() and dest.stat().st_size == expected_bytes \
                and sha256_file(dest) == expected_sha256:
            progress(f"fetch-geog {label}: {filename} already downloaded "
                     "and pin-verified -- reusing")
            observed, pinned = expected_sha256, True
        else:
            download_archive(url, dest, expected_bytes=expected_bytes,
                             progress=progress, label=label,
                             strict_size=not (allow_drift
                                              and source == "ncar"),
                             urlopen_fn=urlopen_fn)
            observed, pinned = verify_archive(
                dest, expected_sha256=expected_sha256,
                expected_bytes=expected_bytes, source=source,
                allow_drift=allow_drift, progress=progress, label=label)
        counts = extract_datasets(dest, root, targets,
                                  progress=progress, label=label)
        staged += len(targets)
        # Published NOW, before the verified archive is removed: the
        # receipt is the only record of where these tiles came from, and
        # holding it in memory until every archive finished meant a kill
        # in between lost it with nothing able to rebuild it.
        publish_archive_entry(root, filename, {
            "url": url, "source": source,
            "archive_bytes": dest.stat().st_size,
            "archive_sha256": observed, "pinned": pinned,
            "datasets": {target: counts[target] for target in targets},
            "fetched_at": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
        })
        if keep_archives:
            progress(f"fetch-geog {label}: kept archive {dest}")
        else:
            dest.unlink()
            progress(f"fetch-geog {label}: removed verified archive "
                     f"{dest.name} (--keep-archives retains it)")

    if bundle:
        run_archive(MANDATORY_BUNDLE_FILENAME, MANDATORY_BUNDLE_BYTES,
                    MANDATORY_BUNDLE_SHA256, tuple(needed), "bundle")
    else:
        for name in needed:
            archive = archive_for(name)
            run_archive(archive.filename, archive.archive_bytes,
                        archive.archive_sha256, (name,), name)

    failures = []
    for name in datasets:
        ok, detail = validate_dataset_dir(root, name)
        if not ok:
            failures.append(detail)
    if failures:
        raise GeogFetchError(
            "post-stage validation failed:\n  " + "\n  ".join(failures))
    progress(f"fetch-geog: {staged} dataset(s) staged; all "
             f"{len(datasets)} requested datasets valid at {root} "
             "(each carries its parsing WPS `index`); `gpuwm doctor` "
             "re-checks the full estate")
    return staged


def fetch_geog_main(args) -> int:
    datasets = parse_datasets(args.datasets)
    source = resolve_source(args.source, args.bundle)
    if args.root is not None:
        root = Path(args.root)
    else:
        root = default_geog_root()
        print(f"fetch-geog: --root not given; staging into {root} "
              "(the $GPUWM_CASE_DATA_ROOT/WPS_GEOG default every other "
              "command reads)")
    if getattr(args, "list", False):
        _print_listing(datasets, source, root)
        return 0
    fetch_geog(root=root, datasets=datasets, source=source,
               bundle=args.bundle, keep_archives=args.keep_archives,
               allow_drift=args.allow_upstream_drift)
    return 0


def register_cli(subparsers) -> None:
    # Both totals are read off the table.  The unpacked figure used to be
    # the literal "~16 GiB", which stopped being true the moment a pin was
    # added and left the command understating its own download by most of
    # its size.
    total_gib = sum(a.archive_bytes for a in GEOG_ARCHIVES) / (1024 ** 3)
    unpacked_gib = sum(a.extracted_bytes for a in GEOG_ARCHIVES) / (1024 ** 3)
    parser = subparsers.add_parser(
        "fetch-geog",
        help=f"download and stage the {len(GEOG_ARCHIVES)} WPS_GEOG static "
             f"datasets (~{total_gib:.1f} GiB download, "
             f"~{unpacked_gib:.0f} GiB unpacked), SHA-256-verified against "
             "the packaged pins, resumable, idempotent when everything is "
             "already staged")
    parser.add_argument(
        "--root", type=Path, default=None, metavar="DIR",
        help="geog root to stage into (default: "
             "$GPUWM_CASE_DATA_ROOT/WPS_GEOG, exactly what gpuwm doctor "
             "checks and wizard configs reference)")
    wrf_count = len(datasets_required_by(GEOG_CONSUMER_WRF))
    mesh_gib = sum(archive_for(name).extracted_bytes
                   for name in datasets_required_by(GEOG_CONSUMER_MESH)
                   if GEOG_CONSUMER_WRF not in archive_for(name).required_by)
    parser.add_argument(
        "--datasets", default="all", metavar="all|CONSUMER|NAME,NAME",
        help=f"which datasets to stage (default 'all', every pin -- the "
             f"{len(GEOG_ARCHIVES)} above).  A consumer name stands for one "
             f"door's whole set: '{GEOG_CONSUMER_WRF}' is the {wrf_count} "
             f"the WRF static builder opens, '{GEOG_CONSUMER_MESH}' is what "
             "gpuwm mesh needs for the static half of its pair.  Use "
             f"'--datasets {GEOG_CONSUMER_WRF}' to skip the "
             f"~{mesh_gib / (1024 ** 3):.0f} GiB Noah-MP soil archive that "
             "only gpuwm mesh reads")
    parser.add_argument(
        "--source", default=None, choices=GEOG_SOURCES,
        help="download host: 'hf' (default) is the ArWen mirror on "
             "Hugging Face (CDN bandwidth, pinned bytes); 'ncar' is the "
             "upstream NCAR server (single host, can be slow, no "
             "upstream checksums -- the same pins are enforced)")
    parser.add_argument(
        "--bundle", action="store_true",
        help="fetch NCAR's single geog_high_res_mandatory.tar.gz "
             f"({MANDATORY_BUNDLE_BYTES / (1024 ** 3):.1f} GiB) instead "
             "of the per-dataset tarballs and extract the requested "
             "datasets from it (fallback; NCAR only)")
    parser.add_argument(
        "--keep-archives", action="store_true",
        help="keep the verified tarballs under <root>/"
             f"{ARCHIVE_SUBDIR} after extraction (default: remove each "
             "one after its datasets validate)")
    parser.add_argument(
        "--allow-upstream-drift", action="store_true",
        help="accept an NCAR archive whose bytes no longer match the "
             "packaged pin (recorded as unpinned; refused outside a "
             "sanity size band); never applies to the mirror")
    parser.add_argument(
        "--list", action="store_true",
        help="print the dataset/size/source table and per-dataset "
             "staged state, then exit without touching the network")
    parser.set_defaults(func=fetch_geog_main)
    return parser


__all__ = [
    "ARCHIVE_SUBDIR", "DRIFT_SIZE_BAND", "GEOG_ARCHIVES",
    "GEOG_CONSUMERS", "GEOG_CONSUMER_MESH", "GEOG_CONSUMER_WRF",
    "GEOG_FETCH_MANIFEST_NAME", "GEOG_FETCH_MANIFEST_SCHEMA",
    "GEOG_SOURCES", "GEOG_URL_BASE_ENV", "GeogArchive", "GeogFetchError",
    "HF_MIRROR_BASE_URL", "HF_MIRROR_REPO", "MANDATORY_BUNDLE_BYTES",
    "MANDATORY_BUNDLE_DATASETS", "MANDATORY_BUNDLE_FILENAME",
    "MANDATORY_BUNDLE_SHA256", "NCAR_BASE_URL", "archive_for",
    "archive_url", "datasets_required_by", "default_geog_root",
    "download_archive", "extract_datasets", "fetch_geog", "fetch_geog_main",
    "geog_datasets", "parse_datasets", "register_cli", "resolve_source",
    "sha256_file", "size_phrase", "validate_dataset_dir",
]
