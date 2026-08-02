"""``gpuwm fetch-bridges``: stage the prebuilt Rust artifacts.

The pip wheel ships no compiled Rust.  Until this command existed the
only way to get the GRIB decoders, the CPU preprocessing library, the
fetch backbone and the batch renderer onto a wheel install was to clone
the repository and run ``cargo build`` twice -- a Rust toolchain, a
2.5 GB checkout and a few minutes of compiling, for eight files.
``gpuwm fetch-bridges`` is the same trade :mod:`gpuwm.table_assets`
already makes for the externalized physics tables: the artifacts are
published as versioned GitHub release assets, their exact size and
SHA-256 pins are packaged inside the wheel at release time, and every
byte is verified against those pins *before* anything is installed.

What is staged, and where
-------------------------
One bundle per platform, holding the eight artifacts of
:data:`BUNDLED_ARTIFACTS`, staged into :func:`gpuwm.bridges
.default_bridge_dir` (``~/.gpuwm/bridges``) -- the last rung of the
resolution ladder every consumer already searches, so nothing else in
gpuwm needs to know this command exists.  ``--dest DIR`` stages
somewhere else; the per-artifact environment variables keep overriding
everything, and a set override is reported rather than silently
shadowed.

A bundle also carries the renderer's map assets -- the Natural Earth
and US Census shapefiles ``rw_wrfbatch`` draws coastlines, borders,
state lines and counties from -- under :data:`ASSET_ROOT`, staged
beside the binaries as ``<dest>/assets/basemap/...``.  That is not a
new lookup: ``rw_wrfbatch`` already resolves ``assets/basemap`` under
its own ancestors, so a renderer staged at ``~/.gpuwm/bridges`` finds
the shapefiles itself, with no environment variable set and no Python
in the loop.  They travel in the bundle rather than the wheel because
the wheel has no room: it is 74.6 MiB against PyPI's 100 MB per-file
cap and the assets deflate to 20.2 MiB.  Binary and basemaps arriving
by one mechanism is the point -- a renderer that draws a cyclone over
a blank rectangle is what happens when they arrive by two.

Platform support is a capability check on the OS and machine
architecture -- can this box run the bytes in that bundle -- and never
an identity gate.  A platform with no published bundle is told so by
name and handed the build-from-source route, which remains the
universal one.

The staging contract
--------------------
Every artifact is verified three ways before it is installed: the exact
byte count, the SHA-256 pin packaged with this release, and (for the
decoders that declare one) the contract marker
:data:`gpuwm.bridges.BRIDGE_ABI_MARKERS` -- the same static check
``gpuwm doctor`` applies to a binary already on disk.  A staged file
that fails any of the three is deleted and refused, never installed.
Extraction reads members by their exact pinned filename, so an archive
cannot place a byte anywhere the pins did not name.

Where this deliberately differs from ``fetch-tables``: an existing file
whose bytes do not match the pin is *replaced* here rather than
refused.  A physics table is a fixed asset, so a wrong one on disk is
the operator's to explain; a bridge executable is a versioned build,
and yesterday's copy sitting in ``~/.gpuwm/bridges`` after a Python-half
upgrade is exactly the skew that made 1.1.0 preparations fail with a
message blaming the file gpuwm had just written correctly.  Replacing
it is the point of the command.  The replacement is still gated on the
new bytes passing all three checks first.

Offline and mirrors
-------------------
``--from DIR`` stages from a local directory under identical
verification: either the bundle archive itself, or the eight artifacts
loose in that directory (what an air-gapped operator has after building
them on a machine that does have a toolchain).
``GPUWM_BRIDGE_ASSET_URL_BASE`` overrides the download base URL; the
bundle filename is appended to it either way.

Pins are generated at release time
----------------------------------
The bundles are built by the release workflow from the same commit as
the wheel, and ``tools/build_bridge_bundle.py`` computes the pins from
those exact bytes and writes them into :data:`PINS_RESOURCE` before the
wheel is built.  A tree that has not been through that step carries a
pins document declaring no platforms, and this command says so and
refuses rather than inventing a hash to check against.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform as platform_module
import shutil
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile

from gpuwm import bridges
from gpuwm.explain import warn

#: Schema of the packaged pins document.
PINS_SCHEMA = "gpuwm-bridge-pins-v1"

#: Schema of the manifest published beside the bundles as a release
#: asset (the human/machine-readable index of what a release shipped).
BUNDLE_MANIFEST_SCHEMA = "gpuwm-bridge-bundle-manifest-v1"

#: The packaged pins document, relative to the ``gpuwm`` package.
PINS_RESOURCE = "data/bridges/bridge-pins.json"

#: Platform keys a release publishes bundles for.  A key names an OS and
#: a machine architecture and nothing else: it answers "can this box run
#: those bytes", never "who is asking".
SUPPORTED_PLATFORMS = ("win-x86_64", "linux-x86_64")

#: Override the download base URL (private mirrors, tests).  The bundle
#: filename is appended to it.
ASSET_URL_BASE_ENV = "GPUWM_BRIDGE_ASSET_URL_BASE"

#: Subdirectory of the staging destination holding in-flight and
#: verified bundle archives (kept only under ``--keep-bundle``).
ARCHIVE_SUBDIR = ".fetch-bridges"

#: Prefix, inside a bundle and inside the staging destination, of every
#: data file the renderer reads.  ``rw_wrfbatch`` searches
#: ``assets/basemap`` under the first eight ancestors of its own
#: directory, so staging under this prefix puts the shapefiles where the
#: binary already looks -- the reason the asset half needs no new
#: resolution logic, no environment variable, and no cooperation from
#: the Python half at all.
ASSET_ROOT = "assets"

#: The asset subtrees a bundle is required to carry, relative to
#: :data:`ASSET_ROOT`.  Directories, never filenames: the file list is
#: generated from the tree at pack time and pinned by hash, so it cannot
#: drift the way a hand-maintained enumeration does.  This tuple is the
#: declaration a release is checked against.
REQUIRED_ASSET_SUBDIRS = ("basemap",)

_BLOCK_BYTES = 8 * 1024 * 1024
_USER_AGENT = "gpuwm-fetch-bridges/1"
_TIMEOUT_S = 120


class BridgeAssetError(RuntimeError):
    """A refusal: wrong bytes, unfetchable bundle, or unusable source."""


# ---------------------------------------------------------------------------
# What a bundle carries
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BundledArtifact:
    """One built file a bundle carries, and what it is for.

    ``kind`` is ``"executable"`` or ``"library"``: the two spellings a
    platform gives a built artifact, which is the whole reason the
    filename cannot be derived from the logical name alone.
    """

    name: str
    kind: str
    crate: str
    env_var: str
    consumer: str


#: The eight artifacts a bundle carries, in build order: the five GRIB
#: decoders and the CPU preprocessing library from the decoder
#: workspace, then the fetch backbone and the batch renderer from the
#: renderer workspace.  The environment variables are the ones the
#: resolution ladder already honours, so a staged bundle and a
#: hand-built tree are found by exactly the same code.
BUNDLED_ARTIFACTS: tuple[BundledArtifact, ...] = (
    BundledArtifact(
        "grib1_bridge", "executable", bridges.CRATE_RELATIVE,
        bridges.BRIDGE_ENV["grib1_bridge"],
        "ERA5 route (gpuwm check/run, rw-wps --source era5)"),
    BundledArtifact(
        "gfs_grib2_bridge", "executable", bridges.CRATE_RELATIVE,
        bridges.BRIDGE_ENV["gfs_grib2_bridge"],
        "GFS front door (rw-wps --source gfs)"),
    BundledArtifact(
        "hrrr_grib2_bridge", "executable", bridges.CRATE_RELATIVE,
        bridges.BRIDGE_ENV["hrrr_grib2_bridge"],
        "HRRR front door (rw-wps --source hrrr)"),
    BundledArtifact(
        "grib2_inventory", "executable", bridges.CRATE_RELATIVE,
        bridges.BRIDGE_ENV["grib2_inventory"],
        "20CRv3/mapped GRIB2 routes"),
    BundledArtifact(
        "grib2_dump", "executable", bridges.CRATE_RELATIVE,
        bridges.BRIDGE_ENV["grib2_dump"],
        "20CRv3/mapped GRIB2 routes"),
    BundledArtifact(
        "gpuwm_preprocess_cpu", "library", bridges.CRATE_RELATIVE,
        "GPUWM_CPU_PREPROCESS_BRIDGE",
        "--preprocess-backend cpu"),
    BundledArtifact(
        "rw_fetch", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "GPUWM_RW_FETCH", "gpuwm fetch --engine rust"),
    BundledArtifact(
        "rw_wrfbatch", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "GPUWM_RW_WRFBATCH", "gpuwm render --engine rust"),
)


def artifact_filename(artifact: BundledArtifact, platform: str) -> str:
    """The filename ``artifact`` is built under on ``platform``.

    Parametric in the platform rather than in the host, because the
    release tool inspects both bundles from one machine.  On the host it
    agrees with the resolvers gpuwm already uses
    (:func:`gpuwm.bridges.executable_name` and the CPU backend's own
    library naming); a test binds the two.
    """

    if platform not in SUPPORTED_PLATFORMS:
        raise ValueError(f"unknown platform {platform!r}; known: "
                         f"{SUPPORTED_PLATFORMS}")
    windows = platform.startswith("win-")
    if artifact.kind == "library":
        return f"{artifact.name}.dll" if windows else f"lib{artifact.name}.so"
    return f"{artifact.name}.exe" if windows else artifact.name


def host_platform() -> str | None:
    """This box's platform key, or None when no bundle is published.

    A capability check: the operating system and the machine
    architecture, which together decide whether the bytes in a bundle
    can execute here.  None is not a refusal of service -- it routes to
    the build-from-source remedy, which works everywhere.
    """

    machine = platform_module.machine().lower()
    if sys.platform == "win32" and machine in ("amd64", "x86_64"):
        return "win-x86_64"
    if sys.platform.startswith("linux") and machine == "x86_64":
        return "linux-x86_64"
    return None


def host_platform_description() -> str:
    """What :func:`host_platform` looked at, for an honest refusal."""

    return f"{sys.platform}/{platform_module.machine() or 'unknown'}"


# ---------------------------------------------------------------------------
# The pins document
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BinaryPin:
    """One artifact inside a bundle, pinned by exact size and SHA-256."""

    artifact: str
    filename: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class AssetPin:
    """One data file inside a bundle, pinned by exact size and SHA-256.

    ``path`` is the archive member name and, unchanged, the path the
    file is staged to under the destination -- a relative POSIX path
    beginning with :data:`ASSET_ROOT`.  Unlike a :class:`BinaryPin` it
    names no artifact, no environment variable and no ABI marker: a
    shapefile is data, identical on every platform, and the only
    questions worth asking of it are how many bytes it is and whether
    they hash to what the release published.
    """

    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class BundlePin:
    """One platform's bundle: the archive, pinned, and its contents.

    ``assets`` defaults to empty so a pins document written before the
    asset half existed still parses; a *release* with no assets is
    refused by ``tools/build_bridge_bundle.py pin``, which is where that
    staleness would otherwise be introduced.
    """

    platform: str
    filename: str
    bytes: int
    sha256: str
    binaries: tuple[BinaryPin, ...]
    assets: tuple[AssetPin, ...] = ()


@dataclass(frozen=True)
class BridgePins:
    """The packaged pins: which release, and one bundle per platform."""

    release: str | None
    platforms: dict[str, BundlePin]

    def bundle_for(self, platform: str | None) -> BundlePin | None:
        if platform is None:
            return None
        return self.platforms.get(platform)


def packaged_pins_path() -> Path:
    """Path of the pins document inside this installation."""

    return Path(__file__).resolve().parent / Path(PINS_RESOURCE)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BridgeAssetError(f"bridge pins document is malformed: {message}")


def parse_pins(payload: object, *, origin: str = "pins") -> BridgePins:
    """Validate a pins document and return it, or refuse.

    A malformed pins document must fail loudly here rather than produce
    a bundle nobody can verify: every field this staging path relies on
    is required, and a hash is required to look like one.
    """

    _require(isinstance(payload, dict), f"{origin} is not an object")
    assert isinstance(payload, dict)  # narrowed by _require
    _require(payload.get("schema") == PINS_SCHEMA,
             f"{origin} schema is {payload.get('schema')!r}, expected "
             f"{PINS_SCHEMA!r}")
    release = payload.get("release")
    _require(release is None or (isinstance(release, str) and release),
             f"{origin} release must be a non-empty string or null")
    raw_platforms = payload.get("platforms")
    _require(isinstance(raw_platforms, dict),
             f"{origin} platforms must be an object")
    assert isinstance(raw_platforms, dict)
    platforms: dict[str, BundlePin] = {}
    for key, record in sorted(raw_platforms.items()):
        if key not in SUPPORTED_PLATFORMS:
            # A pins document from a newer release may name a platform
            # this build does not know; that entry is irrelevant to
            # this host and skipping it keeps the document usable.
            warn(f"{origin} declares platform {key!r} this build does "
                 "not know; skipping that entry")
            continue
        _require(isinstance(record, dict), f"{key}: record is not an object")
        bundle = record.get("bundle")
        _require(isinstance(bundle, dict), f"{key}: bundle is not an object")
        binaries_raw = record.get("binaries")
        _require(isinstance(binaries_raw, list) and binaries_raw,
                 f"{key}: binaries must be a non-empty list")
        binaries: list[BinaryPin] = []
        for entry in binaries_raw:
            _require(isinstance(entry, dict),
                     f"{key}: a binaries entry is not an object")
            binaries.append(BinaryPin(
                artifact=_string(entry, "artifact", key),
                filename=_string(entry, "filename", key),
                bytes=_size(entry, key),
                sha256=_digest(entry, key)))
        assets_raw = record.get("assets", [])
        _require(isinstance(assets_raw, list),
                 f"{key}: assets must be a list")
        assets: list[AssetPin] = []
        for entry in assets_raw:
            _require(isinstance(entry, dict),
                     f"{key}: an assets entry is not an object")
            assets.append(AssetPin(
                path=_asset_path(entry, key),
                bytes=_size(entry, key),
                sha256=_digest(entry, key)))
        _require(release is not None,
                 f"{key}: a platform is pinned but the release is null")
        platforms[key] = BundlePin(
            platform=key, filename=_string(bundle, "filename", key),
            bytes=_size(bundle, key), sha256=_digest(bundle, key),
            binaries=tuple(binaries), assets=tuple(assets))
    return BridgePins(release=release, platforms=platforms)


def _string(record: dict, field: str, where: str) -> str:
    value = record.get(field)
    _require(isinstance(value, str) and value,
             f"{where}: {field} must be a non-empty string")
    assert isinstance(value, str)
    return value


def _asset_path(record: dict, where: str) -> str:
    """A relative asset path under :data:`ASSET_ROOT`, or a refusal.

    The pins document decides where staging writes, so it is the place
    to refuse a path that could escape the destination.  Absolute paths,
    drive letters, backslashes and ``..`` components are all rejected
    here rather than left for the filesystem to interpret; staging then
    joins a path it already knows is contained.
    """

    value = _string(record, "path", where)
    parts = value.split("/")
    _require(
        not value.startswith("/") and "\\" not in value and ":" not in value
        and all(part not in ("", ".", "..") for part in parts)
        and parts[0] == ASSET_ROOT and len(parts) > 1,
        f"{where}: asset path {value!r} must be a relative path under "
        f"{ASSET_ROOT!r}/ with no '..' component")
    return value


def _size(record: dict, where: str) -> int:
    value = record.get("bytes")
    _require(isinstance(value, int) and not isinstance(value, bool)
             and value > 0, f"{where}: bytes must be a positive integer")
    assert isinstance(value, int)
    return value


def _digest(record: dict, where: str) -> str:
    value = record.get("sha256")
    _require(isinstance(value, str) and len(value) == 64
             and all(c in "0123456789abcdef" for c in value),
             f"{where}: sha256 must be 64 lowercase hex characters")
    assert isinstance(value, str)
    return value


def load_pins(path: Path | None = None) -> BridgePins:
    """Read and validate the packaged pins document."""

    resolved = packaged_pins_path() if path is None else Path(path)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except OSError as error:
        raise BridgeAssetError(
            f"bridge pins document {resolved} is unreadable: {error}")
    except ValueError as error:
        raise BridgeAssetError(
            f"bridge pins document {resolved} is not JSON: {error}")
    return parse_pins(payload, origin=str(resolved))


def staging_available(pins: BridgePins | None = None) -> bool:
    """Can THIS install stage a bundle for THIS platform?

    Both halves are capabilities of artifacts rather than opinions: a
    platform key for this OS/arch, and a bundle pinned for it in the
    document this wheel actually carries.  ``gpuwm doctor`` offers
    ``gpuwm fetch-bridges`` exactly when this is true, so the report
    never advertises a command that would refuse.
    """

    platform = host_platform()
    if platform is None:
        return False
    try:
        resolved = load_pins() if pins is None else pins
    except BridgeAssetError:
        return False
    return resolved.bundle_for(platform) is not None


def asset_url_base(pins: BridgePins) -> str:
    """Base URL the bundle is downloaded from (env override wins)."""

    override = os.environ.get(ASSET_URL_BASE_ENV)
    if override and override.strip():
        return override.strip().rstrip("/")
    if not pins.release:
        raise BridgeAssetError(
            "the packaged pins declare no release, so there is no URL to "
            f"download from; set {ASSET_URL_BASE_ENV} or use --from DIR")
    return f"{bridges.REPOSITORY_URL}/releases/download/{pins.release}"


def bundle_url(pins: BridgePins, bundle: BundlePin) -> str:
    return f"{asset_url_base(pins)}/{bundle.filename}"


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def verify_pinned_file(path: Path, *, expected_bytes: int,
                       expected_sha256: str, label: str) -> None:
    """Exact size and SHA-256, or a refusal naming both observations."""

    actual = path.stat().st_size
    if actual != expected_bytes:
        raise BridgeAssetError(
            f"{label}: {actual:,} bytes, expected {expected_bytes:,}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise BridgeAssetError(
            f"{label}: SHA-256 {observed}, expected {expected_sha256}")


def verify_contract_marker(artifact: str, path: Path) -> None:
    """The static contract check ``gpuwm doctor`` applies, before install.

    A bundle is built from the same commit as the wheel that carries its
    pins, so a marker mismatch here means the release process assembled
    the two from different trees.  Catching it at staging is the whole
    difference between a refusal and a preparation that dies hours later
    blaming a file gpuwm wrote correctly.
    """

    if artifact not in bridges.BRIDGE_ABI_MARKERS:
        return
    ok, evidence = bridges.bridge_abi_matches(artifact, path)
    if not ok:
        raise BridgeAssetError(f"{path.name}: {evidence}")


def matches_pin(path: Path, pin: BinaryPin) -> bool:
    """Is the file already exactly these bytes?  No exceptions raised."""

    try:
        if not path.is_file() or path.stat().st_size != pin.bytes:
            return False
        return sha256_file(path) == pin.sha256
    except OSError:
        return False


def classify_destination(dest: Path, bundle: BundlePin
                         ) -> tuple[list[BinaryPin], list[BinaryPin],
                                    list[BinaryPin]]:
    """(already pinned, present but different, absent) at ``dest``."""

    staged: list[BinaryPin] = []
    stale: list[BinaryPin] = []
    absent: list[BinaryPin] = []
    for pin in bundle.binaries:
        path = dest / pin.filename
        if not path.is_file():
            absent.append(pin)
        elif matches_pin(path, pin):
            staged.append(pin)
        else:
            stale.append(pin)
    return staged, stale, absent


def classify_assets(dest: Path, bundle: BundlePin
                    ) -> tuple[list[AssetPin], list[AssetPin],
                               list[AssetPin]]:
    """(already pinned, present but different, absent) for the assets.

    Separate from :func:`classify_destination` because the two answer
    different questions for the caller: eight binaries are listed
    individually in a report, and several dozen shapefiles are counted.
    """

    staged: list[AssetPin] = []
    stale: list[AssetPin] = []
    absent: list[AssetPin] = []
    for pin in bundle.assets:
        path = dest / pin.path
        if not path.is_file():
            absent.append(pin)
        elif matches_pin(path, pin):
            staged.append(pin)
        else:
            stale.append(pin)
    return staged, stale, absent


# ---------------------------------------------------------------------------
# Download (resumable, restartable)
# ---------------------------------------------------------------------------

def _default_urlopen(request: Request):
    return urlopen(request, timeout=_TIMEOUT_S)


def download_bundle(url: str, dest: Path, *, expected_bytes: int,
                    progress, urlopen_fn=_default_urlopen) -> None:
    """Download ``url`` to ``dest``, resuming or restarting as needed.

    An interrupted run leaves a partial file; the next run asks the
    server to continue from its byte count.  A server that answers
    anything but ``206 Partial Content`` -- including one that has no
    idea what a range is -- restarts the transfer from zero instead of
    appending to bytes it did not extend, which is the failure mode a
    naive resume turns into a corrupt archive.  A partial longer than
    the pin is discarded outright: it cannot be a prefix of the file we
    are pinned to.
    """

    offset = dest.stat().st_size if dest.exists() else 0
    if offset > expected_bytes:
        dest.unlink()
        progress(f"gpuwm fetch-bridges: the partial download was larger "
                 f"than the pinned {expected_bytes:,} B; discarded it and "
                 "restarted")
        offset = 0
    if offset == expected_bytes:
        progress("gpuwm fetch-bridges: the partial download is already "
                 "complete; verifying it instead of re-downloading")
        return
    headers = {"User-Agent": _USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        progress(f"gpuwm fetch-bridges: resuming at {offset:,} B of "
                 f"{expected_bytes:,} B")
    request = Request(url, headers=headers)
    try:
        with urlopen_fn(request) as response:
            status = getattr(response, "status", None)
            mode = "ab" if (offset and status == 206) else "wb"
            if offset and status != 206:
                progress("gpuwm fetch-bridges: the server ignored the "
                         "resume range; restarting the download")
            with dest.open(mode) as sink:
                while block := response.read(_BLOCK_BYTES):
                    sink.write(block)
    except HTTPError as error:
        raise BridgeAssetError(
            f"HTTP {error.code} from {url}; the partial file is kept and "
            "a re-run resumes")
    except (URLError, OSError, TimeoutError) as error:
        raise BridgeAssetError(
            f"download failed from {url}: {error}; the partial file is "
            "kept and a re-run resumes")
    size = dest.stat().st_size
    if size != expected_bytes:
        dest.unlink(missing_ok=True)
        raise BridgeAssetError(
            f"downloaded {size:,} B from {url}, expected "
            f"{expected_bytes:,} B; the incomplete file was removed -- "
            "re-run to fetch it again")


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------

def _install(temp: Path, final: Path, pin: BinaryPin) -> None:
    """Verify ``temp`` three ways, then atomically install it."""

    try:
        verify_pinned_file(temp, expected_bytes=pin.bytes,
                           expected_sha256=pin.sha256, label=pin.filename)
        verify_contract_marker(pin.artifact, temp)
    except BridgeAssetError:
        temp.unlink(missing_ok=True)
        raise
    if os.name != "nt":
        # A zip carries no reliable execute bit, and an executable that
        # cannot be executed is not staged, it is merely present.
        os.chmod(temp, 0o755)
    os.replace(temp, final)


def _install_asset(temp: Path, final: Path, pin: AssetPin) -> None:
    """Verify ``temp`` by size and hash, then atomically install it.

    Two checks, not three, and no execute bit: a shapefile has no ABI
    marker to match and nothing should ever run it.  Everything else --
    verify before install, delete on failure, atomic replace -- is what
    :func:`_install` does for a binary.
    """

    try:
        verify_pinned_file(temp, expected_bytes=pin.bytes,
                           expected_sha256=pin.sha256, label=pin.path)
    except BridgeAssetError:
        temp.unlink(missing_ok=True)
        raise
    if os.name != "nt":
        os.chmod(temp, 0o644)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temp, final)


def _bundle_members(archive: Path) -> list[str]:
    try:
        with zipfile.ZipFile(archive) as zf:
            return zf.namelist()
    except (zipfile.BadZipFile, OSError) as error:
        raise BridgeAssetError(f"{archive.name} is not a readable bundle "
                               f"archive: {error}")


def stage_from_bundle(archive: Path, bundle: BundlePin, dest: Path,
                      *, progress) -> list[Path]:
    """Extract, verify and install every pinned artifact from ``archive``.

    Members are read by their exact pinned filename, never by whatever
    path the archive proposes, so nothing can be written outside
    ``dest``.  A pinned name the archive does not carry is a refusal
    that names what the archive holds instead -- which is how the other
    platform's bundle identifies itself.
    """

    present = set(_bundle_members(archive))
    missing = [pin.filename for pin in bundle.binaries
               if pin.filename not in present]
    if missing:
        listed = ", ".join(sorted(present)[:8]) or "nothing"
        raise BridgeAssetError(
            f"{archive.name} does not carry {', '.join(missing)} -- it "
            f"holds {listed}.  This is not the {bundle.platform} bundle")
    absent_assets = [pin.path for pin in bundle.assets
                     if pin.path not in present]
    if absent_assets:
        raise BridgeAssetError(
            f"{archive.name} is missing {len(absent_assets)} pinned map "
            f"asset(s), starting with {absent_assets[0]}.  A bundle whose "
            "assets do not match its pins would render geography-less "
            "plots; refusing it")
    dest.mkdir(parents=True, exist_ok=True)
    work = dest / f"{ARCHIVE_SUBDIR}-stage"
    work.mkdir(parents=True, exist_ok=True)
    installed: list[Path] = []
    try:
        with zipfile.ZipFile(archive) as zf:
            for pin in bundle.binaries:
                temp = work / pin.filename
                with zf.open(pin.filename) as source, \
                        temp.open("wb") as sink:
                    shutil.copyfileobj(source, sink, _BLOCK_BYTES)
                final = dest / pin.filename
                replaced = final.is_file()
                _install(temp, final, pin)
                installed.append(final)
                progress(
                    f"gpuwm fetch-bridges: {'replaced' if replaced else 'staged'}"
                    f" {final} ({pin.bytes:,} B, SHA-256 {pin.sha256[:12]}...)")
            staged_assets = 0
            asset_bytes = 0
            for pin in bundle.assets:
                # The pinned path is validated relative and contained by
                # parse_pins, so the join cannot leave dest.
                temp = work / "asset.part"
                with zf.open(pin.path) as source, temp.open("wb") as sink:
                    shutil.copyfileobj(source, sink, _BLOCK_BYTES)
                final = dest / pin.path
                _install_asset(temp, final, pin)
                installed.append(final)
                staged_assets += 1
                asset_bytes += pin.bytes
            if staged_assets:
                progress(
                    f"gpuwm fetch-bridges: staged {staged_assets} map asset "
                    f"file(s) ({asset_bytes / (1024 * 1024):.1f} MiB) under "
                    f"{dest / ASSET_ROOT}; the renderer finds them there "
                    "without any environment variable")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return installed


def stage_from_loose_files(source_dir: Path, bundle: BundlePin, dest: Path,
                           *, progress) -> list[Path]:
    """Install the pinned artifacts sitting loose in ``source_dir``.

    What an air-gapped operator has after building on a machine that
    does have a toolchain: eight files, no archive.  Same three checks,
    same atomic install.
    """

    absent = [pin.filename for pin in bundle.binaries
              if not (source_dir / pin.filename).is_file()]
    present = [pin for pin in bundle.binaries
               if (source_dir / pin.filename).is_file()]
    if absent and not present:
        raise BridgeAssetError(
            f"{source_dir} carries neither {bundle.filename} nor the loose "
            f"artifacts; missing {', '.join(absent)}")
    if absent:
        # The eight artifacts are independent; an air-gapped operator
        # with the decoders but not the renderer gets the decoders,
        # verified, and doctor names what is still missing.
        warn(f"{source_dir} is missing {len(absent)} of "
             f"{len(bundle.binaries)} pinned artifact(s) "
             f"({', '.join(absent)}); staging the {len(present)} that "
             "are present -- gpuwm doctor reports the rest")
    dest.mkdir(parents=True, exist_ok=True)
    work = dest / f"{ARCHIVE_SUBDIR}-stage"
    work.mkdir(parents=True, exist_ok=True)
    installed: list[Path] = []
    try:
        for pin in present:
            temp = work / pin.filename
            shutil.copyfile(source_dir / pin.filename, temp)
            final = dest / pin.filename
            replaced = final.is_file()
            _install(temp, final, pin)
            installed.append(final)
            progress(
                f"gpuwm fetch-bridges: {'replaced' if replaced else 'staged'}"
                f" {final} ({pin.bytes:,} B, SHA-256 {pin.sha256[:12]}...)")
        # The map assets keep their layout in a loose directory too, so
        # an operator who unzipped a bundle by hand and an operator who
        # copied a checkout's assets/ both land here.
        staged_assets = 0
        missing_assets: list[str] = []
        for pin in bundle.assets:
            origin = source_dir / pin.path
            if not origin.is_file():
                missing_assets.append(pin.path)
                continue
            temp = work / "asset.part"
            shutil.copyfile(origin, temp)
            final = dest / pin.path
            _install_asset(temp, final, pin)
            installed.append(final)
            staged_assets += 1
        if staged_assets:
            progress(f"gpuwm fetch-bridges: staged {staged_assets} map "
                     f"asset file(s) under {dest / ASSET_ROOT}")
        if missing_assets:
            warn(f"{source_dir} carries no {ASSET_ROOT}/ tree for "
                 f"{len(missing_assets)} pinned map asset file(s); the "
                 "renderer will draw plots without coastlines or borders "
                 "unless it finds them elsewhere")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return installed


def stage_from_dir(source_dir: Path, bundle: BundlePin, dest: Path,
                   *, progress) -> list[Path]:
    """``--from DIR``: the bundle archive if it is there, else the files."""

    archive = source_dir / bundle.filename
    if archive.is_file():
        verify_pinned_file(archive, expected_bytes=bundle.bytes,
                           expected_sha256=bundle.sha256,
                           label=bundle.filename)
        progress(f"gpuwm fetch-bridges: {archive} matches the packaged "
                 "bundle pin; staging from it")
        return stage_from_bundle(archive, bundle, dest, progress=progress)
    return stage_from_loose_files(source_dir, bundle, dest, progress=progress)


def fetch_bundle(pins: BridgePins, bundle: BundlePin, dest: Path, *,
                 keep_bundle: bool = False, progress=print,
                 urlopen_fn=_default_urlopen) -> list[Path]:
    """Download, verify and stage the platform bundle."""

    archive_dir = dest / ARCHIVE_SUBDIR
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / bundle.filename
    url = bundle_url(pins, bundle)
    if archive.is_file() and archive.stat().st_size == bundle.bytes \
            and sha256_file(archive) == bundle.sha256:
        progress(f"gpuwm fetch-bridges: {bundle.filename} is already "
                 "downloaded and pin-verified; reusing it")
    else:
        progress(f"gpuwm fetch-bridges: downloading {bundle.filename} "
                 f"({bundle.bytes / (1024 * 1024):.1f} MiB) from {url}")
        download_bundle(url, archive, expected_bytes=bundle.bytes,
                        progress=progress, urlopen_fn=urlopen_fn)
        try:
            verify_pinned_file(archive, expected_bytes=bundle.bytes,
                               expected_sha256=bundle.sha256,
                               label=bundle.filename)
        except BridgeAssetError:
            archive.unlink(missing_ok=True)
            raise
    installed = stage_from_bundle(archive, bundle, dest, progress=progress)
    if keep_bundle:
        progress(f"gpuwm fetch-bridges: kept the verified bundle {archive}")
    else:
        archive.unlink(missing_ok=True)
        try:
            archive_dir.rmdir()
        except OSError:
            pass
    return installed


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------

def _unsupported_platform_note(platform: str | None) -> str:
    if platform is None:
        return (f"gpuwm fetch-bridges: no bundle is published for "
                f"{host_platform_description()} (bundles exist for "
                f"{', '.join(SUPPORTED_PLATFORMS)}); build the artifacts "
                "from a clone instead -- `gpuwm doctor` prints the exact "
                "steps for this install")
    return (f"gpuwm fetch-bridges: this install carries no bundle pins for "
            f"{platform}; build the artifacts from a clone instead -- "
            "`gpuwm doctor` prints the exact steps for this install")


def _override_warnings(bundle: BundlePin) -> list[str]:
    """Environment overrides that would shadow what we just staged."""

    by_name = {artifact.name: artifact for artifact in BUNDLED_ARTIFACTS}
    notes: list[str] = []
    for pin in bundle.binaries:
        artifact = by_name.get(pin.artifact)
        if artifact is None:
            continue
        override = os.environ.get(artifact.env_var)
        if override:
            notes.append(
                f"gpuwm fetch-bridges: NOTE: {artifact.env_var} is set to "
                f"{override}, which wins over the staged copy; unset it to "
                "use what was just staged")
    return notes


def _print_listing(pins: BridgePins, bundle: BundlePin, dest: Path) -> None:
    print(f"gpuwm fetch-bridges: platform {bundle.platform} "
          f"({host_platform_description()})")
    print(f"gpuwm fetch-bridges: release {pins.release}")
    try:
        print(f"gpuwm fetch-bridges: bundle {bundle_url(pins, bundle)} "
              f"({bundle.bytes / (1024 * 1024):.1f} MiB)")
    except BridgeAssetError as error:
        print(f"gpuwm fetch-bridges: bundle URL unavailable: {error}")
    print(f"gpuwm fetch-bridges: destination {dest}")
    staged, stale, absent = classify_destination(dest, bundle)
    states: dict[str, str] = {}
    for state, group in (("staged", staged), ("differs", stale),
                         ("needed", absent)):
        for pin in group:
            states[pin.filename] = state
    for pin in bundle.binaries:
        print(f"  {states[pin.filename]:<8} {pin.filename} "
              f"({pin.bytes:,} B)")
    print(f"gpuwm fetch-bridges: {len(staged)} of "
          f"{len(bundle.binaries)} artifacts already staged and pin-valid")
    if bundle.assets:
        held, differs, needed = classify_assets(dest, bundle)
        total = sum(pin.bytes for pin in bundle.assets)
        print(f"gpuwm fetch-bridges: {len(held)} of {len(bundle.assets)} map "
              f"asset files staged and pin-valid under {dest / ASSET_ROOT} "
              f"({total / (1024 * 1024):.1f} MiB packed)"
              + (f"; {len(differs)} differ, {len(needed)} needed"
                 if (differs or needed) else ""))


def fetch_bridges_main(args) -> int:
    dest = (Path(args.dest) if getattr(args, "dest", None)
            else bridges.default_bridge_dir())
    platform = host_platform()
    try:
        pins = load_pins()
    except BridgeAssetError as error:
        print(f"gpuwm fetch-bridges: REFUSED: {error}")
        return 2
    bundle = pins.bundle_for(platform)
    if bundle is None:
        print(_unsupported_platform_note(platform))
        return 2

    if getattr(args, "list", False):
        _print_listing(pins, bundle, dest)
        return 0

    staged, stale, absent = classify_destination(dest, bundle)
    held_assets, stale_assets, absent_assets = classify_assets(dest, bundle)
    if not stale and not absent and not stale_assets and not absent_assets:
        print(f"gpuwm fetch-bridges: all {len(staged)} artifacts and "
              f"{len(held_assets)} map asset file(s) at {dest} verified "
              "(exact size + SHA-256); nothing to fetch")
        for note in _override_warnings(bundle):
            print(note)
        return 0
    if (stale_assets or absent_assets) and not stale and not absent:
        # The exact shape of the bug this asset half exists to close: a
        # complete set of binaries staged by an older release, and no
        # map assets beside them.  Say so, rather than reporting a
        # generic re-fetch.
        print(f"gpuwm fetch-bridges: the {len(staged)} artifacts at {dest} "
              f"are current, but {len(stale_assets) + len(absent_assets)} of "
              f"{len(bundle.assets)} map asset file(s) are missing or stale "
              "-- without them the renderer draws plots with no coastlines "
              "or borders")
    if stale:
        print(f"gpuwm fetch-bridges: {len(stale)} artifact(s) at {dest} do "
              "not match this release's pins and will be replaced once the "
              f"new bytes verify: {', '.join(p.filename for p in stale)}")

    source_dir = getattr(args, "from_dir", None)
    try:
        if source_dir is not None:
            installed = stage_from_dir(Path(source_dir), bundle, dest,
                                       progress=print)
        else:
            installed = fetch_bundle(
                pins, bundle, dest,
                keep_bundle=getattr(args, "keep_bundle", False),
                progress=print)
    except BridgeAssetError as error:
        print(f"gpuwm fetch-bridges: REFUSED: {error}")
        return 2

    remaining = [pin.filename for pin in bundle.binaries
                 if not matches_pin(dest / pin.filename, pin)]
    installed_names = {path.name for path in installed}
    broken = [name for name in remaining if name in installed_names]
    if broken:
        # Something this invocation just wrote fails its pin: that is
        # an integrity failure, and it stays a refusal.
        print("gpuwm fetch-bridges: REFUSED: staging finished but "
              f"{', '.join(broken)} still do not match their pins")
        return 2
    if remaining:
        # A partial --from DIR staging: everything written verified;
        # what the directory did not carry is reported, not fatal.
        warn(f"{len(remaining)} pinned artifact(s) are still not "
             f"staged ({', '.join(remaining)}); gpuwm doctor names "
             "the remedy for each")
    verified = len(bundle.binaries) - len(remaining)
    print(f"gpuwm fetch-bridges: {len(installed)} file(s) staged at "
          f"{dest}; {verified} of {len(bundle.binaries)} artifacts verified "
          "against the packaged pins.  `gpuwm doctor` re-checks the "
          "full estate")
    unstaged_assets = [pin.path for pin in bundle.assets
                       if not matches_pin(dest / pin.path, pin)]
    if unstaged_assets:
        warn(f"{len(unstaged_assets)} of {len(bundle.assets)} pinned map "
             "asset file(s) are still not staged; the renderer will draw "
             "plots without coastlines or borders unless it finds them "
             "elsewhere")
    elif bundle.assets:
        print(f"gpuwm fetch-bridges: {len(bundle.assets)} map asset file(s) "
              f"verified under {dest / ASSET_ROOT}")
    for note in _override_warnings(bundle):
        print(note)
    return 0


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "fetch-bridges",
        help="download the prebuilt Rust artifacts for this platform "
             "(GRIB decoders, CPU preprocessing library, fetch backbone, "
             "batch renderer) into ~/.gpuwm/bridges, each one verified "
             "against the packaged SHA-256 pins before it is installed; "
             "idempotent when everything is already staged")
    parser.add_argument(
        "--from", dest="from_dir", metavar="DIR", default=None,
        help="stage from a local directory instead of downloading "
             "(offline installs): either the bundle archive or the "
             "artifacts loose in it; verification is identical")
    parser.add_argument(
        "--dest", metavar="DIR", default=None,
        help="stage into DIR instead of ~/.gpuwm/bridges (gpuwm finds "
             "the default on its own; anywhere else needs the "
             "per-artifact environment variables)")
    parser.add_argument(
        "--keep-bundle", action="store_true",
        help=f"keep the verified archive under <dest>/{ARCHIVE_SUBDIR} "
             "after staging (default: remove it)")
    parser.add_argument(
        "--list", action="store_true",
        help="print the platform, bundle and per-artifact staged state, "
             "then exit without touching the network")
    parser.set_defaults(func=fetch_bridges_main)
    return parser


__all__ = [
    "ARCHIVE_SUBDIR", "ASSET_ROOT", "ASSET_URL_BASE_ENV",
    "BUNDLED_ARTIFACTS", "BUNDLE_MANIFEST_SCHEMA", "REQUIRED_ASSET_SUBDIRS",
    "AssetPin", "BinaryPin", "BridgeAssetError", "BridgePins",
    "BundlePin", "BundledArtifact", "PINS_RESOURCE", "PINS_SCHEMA",
    "SUPPORTED_PLATFORMS", "artifact_filename", "asset_url_base",
    "bundle_url", "classify_assets", "classify_destination",
    "download_bundle", "fetch_bundle",
    "fetch_bridges_main", "host_platform", "host_platform_description",
    "load_pins", "matches_pin", "packaged_pins_path", "parse_pins",
    "register_cli", "sha256_file", "stage_from_bundle", "stage_from_dir",
    "stage_from_loose_files", "staging_available", "verify_contract_marker",
    "verify_pinned_file",
]
