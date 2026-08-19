"""The renderer's map assets must arrive with the renderer that reads them.

``rw_wrfbatch`` draws coastlines, national and state borders, lakes and
counties from the Natural Earth + US Census shapefiles in
``tools/rustwx/assets/basemap``.  Those shapefiles reached a pip install
by no mechanism at all: the wheel declared ``tools =
["prepare_hrrr_*.sh"]`` and nothing else, and the bundle carried eight
binaries and no data.  The result was not an error -- it was a plot.  A
tropical cyclone rendered over a blank white rectangle, with nothing on
the image to say where on Earth it was, produced by a lane running the
published wheel exactly as documented.

The bug survived because the machine it was developed on hides it:
``rustwx-render`` falls back to a **cartopy** Natural Earth cache under
``$HOME/.local/share/cartopy``, and a workstation that has ever run
cartopy has one.  Every test in this file therefore either avoids the
renderer's own fallbacks entirely or redirects ``HOME``/``USERPROFILE``
away from them, and the end-to-end proof runs against an installed
wheel with the repository nowhere on the path.

Why the bundle and not the wheel
--------------------------------
Measured, not assumed: the wheel is 74.6 MiB compressed against PyPI's
100 MB per-file cap, and the asset tree deflates to 20.2 MiB.  Shipping
the shapefiles in the wheel leaves roughly half a megabyte of headroom
before an upload starts being rejected, which is not a margin a release
can be run on -- ``MANIFEST.in`` already externalizes the two largest
Thompson tables for exactly this reason.  The bundle has room, and it is
where the consuming binary already is: staged under
``<dest>/assets/basemap``, the shapefiles sit on the resolution ladder
``rw_wrfbatch`` already walks (``assets/basemap`` under the first eight
ancestors of its own directory), so the binary finds them with no
environment variable set and no cooperation from the Python half.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from gpuwm import bridge_assets, rustwx

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_TOOL = REPO_ROOT / "tools" / "build_bridge_bundle.py"
SOURCE_ASSETS = REPO_ROOT / "tools" / "rustwx" / "assets"

#: The source revision these tests "release".  Pin verifies every
#: binary's embedded GPUWM_BRIDGE_SOURCE_REV stamp against it, so the
#: stubs embed it; these tests are about the asset half and must not
#: fail (or pass) on the staleness half.
SOURCE_REV = "5eed" * 10


def _require_source_assets() -> None:
    if not SOURCE_ASSETS.is_dir():
        pytest.skip("the asset contract needs the source tree, not an install")


def _bundle_filename(release: str, platform: str) -> str:
    """The release tool's own naming, imported rather than repeated."""

    sys.path.insert(0, str(REPO_ROOT))
    try:
        import importlib

        packer = importlib.import_module("tools.build_bridge_bundle")
        return packer.bundle_filename(release, platform)
    finally:
        sys.path.remove(str(REPO_ROOT))


def _stub_payload(artifact) -> bytes:
    """The bytes a placeholder binary must carry to be pinnable.

    Both proofs the release tool asks for, on every stub: the declared
    contract marker (what a VENDORED artifact is held to, since its
    source does not move with this checkout) and the
    ``GPUWM_BRIDGE_SOURCE_REV`` stamp (what every other artifact is held
    to).  Carrying both means these tests reach the asset question with
    the binary question genuinely answered -- and it means a stub that
    goes stale against either check reds here rather than silently
    short-circuiting the check a test downstream is actually about.
    """

    from gpuwm import bridges

    marker = bridges.BRIDGE_ABI_MARKERS.get(artifact.name, b"")
    stamp = bridge_assets.SOURCE_REV_MARKER + SOURCE_REV.encode()
    return f"stub::{artifact.name}::".encode() + marker + b"::" + stamp


def _stub_artifacts(directory: Path, platform: str) -> Path:
    """One placeholder file per bundled artifact, named for ``platform``.

    The placeholders embed each bridge's declared contract marker, so
    staging runs its real three-way check (size, SHA-256, ABI marker)
    instead of a weakened one.  These tests are about the asset half;
    they must not reach it by making the binary half easier.
    """

    directory.mkdir(parents=True, exist_ok=True)
    for artifact in bridge_assets.BUNDLED_ARTIFACTS:
        name = bridge_assets.artifact_filename(artifact, platform)
        (directory / name).write_bytes(_stub_payload(artifact))
    return directory


def _pack(tmp_path: Path, *, platform: str = "linux-x86_64",
          release: str = "v0-assets") -> Path:
    """Pack a real bundle with the real release tool."""

    search = _stub_artifacts(tmp_path / "artifacts", platform)
    out = tmp_path / "bundles"
    result = subprocess.run(
        [sys.executable, str(BUNDLE_TOOL), "pack", "--release", release,
         "--platform", platform, "--search", str(search), "--out", str(out)],
        capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stdout + result.stderr
    return out / _bundle_filename(release, platform)


# ---------------------------------------------------------------------------
# The declaration cannot go stale
# ---------------------------------------------------------------------------

def test_the_bundle_carries_every_map_asset_file_in_the_tree(tmp_path):
    """Walked from the tree, never enumerated.

    This is the gate the wheel's ``tools = ["prepare_hrrr_*.sh"]`` line
    did not have.  A new shapefile -- a new resolution, a new layer, a
    replacement of the whole ``basemap`` directory -- is carried because
    the packer walks what is there, and this test fails the moment a
    file on disk is not in the archive.
    """

    _require_source_assets()
    archive = _pack(tmp_path)
    import zipfile

    with zipfile.ZipFile(archive) as zf:
        members = set(zf.namelist())

    on_disk = set()
    for subdir in bridge_assets.REQUIRED_ASSET_SUBDIRS:
        root = SOURCE_ASSETS / subdir
        for path in root.rglob("*"):
            if path.is_file():
                on_disk.add("/".join((
                    bridge_assets.ASSET_ROOT, subdir,
                    path.relative_to(root).as_posix())))
    assert on_disk, "found no map assets on disk -- the walk is broken"
    missing = sorted(on_disk - members)
    assert not missing, (
        f"{len(missing)} map asset file(s) exist in the tree but would not "
        "reach a bundle, so an installed renderer would draw plots without "
        "them:\n  " + "\n  ".join(missing))


def test_the_shapefile_layers_the_renderer_reads_are_all_carried(tmp_path):
    """Name the layers by hand, once, where a human will read the failure.

    The walk above proves the archive matches the tree; this proves the
    tree still holds what ``rustwx-render`` actually opens.  A layer
    deleted from the repository passes the walk (nothing on disk is
    missing from the archive) and silently stops being drawn, which is
    the same class of silent loss in a different disguise.
    """

    _require_source_assets()
    archive = _pack(tmp_path)
    import zipfile

    with zipfile.ZipFile(archive) as zf:
        members = set(zf.namelist())

    required = [
        "assets/basemap/natural_earth_10m/ne_10m_coastline.shp",
        "assets/basemap/natural_earth_10m/ne_10m_land.shp",
        "assets/basemap/natural_earth_10m/ne_10m_ocean.shp",
        "assets/basemap/natural_earth_10m/ne_10m_lakes.shp",
        "assets/basemap/natural_earth_10m/"
        "ne_10m_admin_0_boundary_lines_land.shp",
        "assets/basemap/natural_earth_10m/"
        "ne_10m_admin_1_states_provinces_lines.shp",
        "assets/basemap/natural_earth_110m/ne_110m_coastline.shp",
        "assets/basemap/us_counties_5m/cb_2023_us_county_5m.shp",
    ]
    absent = [name for name in required if name not in members]
    assert not absent, (
        "the renderer opens these layers and the bundle does not carry "
        "them:\n  " + "\n  ".join(absent))
    # A .shp without its .shx is an unreadable shapefile, not a partial one.
    for name in required:
        index = name[:-len(".shp")] + ".shx"
        assert index in members, f"{name} ships without its index {index}"


def test_pack_refuses_when_a_required_asset_directory_is_absent(tmp_path,
                                                               monkeypatch):
    """A missing asset tree stops the release, it does not thin the bundle."""

    _require_source_assets()
    empty = tmp_path / "empty-assets"
    (empty / "unrelated").mkdir(parents=True)
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import importlib

        packer = importlib.import_module("tools.build_bridge_bundle")
        with pytest.raises(SystemExit) as refusal:
            packer.collect_assets(empty)
    finally:
        sys.path.remove(str(REPO_ROOT))
    assert "coastlines" in str(refusal.value)


def test_pin_refuses_a_bundle_that_carries_no_map_assets(tmp_path):
    """The regression gate: an asset-less bundle cannot be pinned.

    This is precisely the bundle 1.4.0 published -- eight binaries and
    nothing else -- and the release tool must now refuse to write pins
    for it rather than produce a wheel that stages geography-less plots.

    The binaries here are the SAME pinnable stubs the good bundle uses
    (:func:`_stub_payload`: contract marker plus source-rev stamp), so
    the only thing wrong with this archive is the missing ``assets/``
    tree.  That matters because pin checks the binaries first: a stub
    that carried the stamp but not the vendored artifacts' contract
    marker earned the staleness refusal instead, and the asset refusal
    this test names was never reached -- the test passed on the wrong
    sentence.  Both refusals stay live; each is asserted where it is
    the one thing wrong.
    """

    import zipfile

    platform = "linux-x86_64"
    release = "v0-assetless"
    archive = tmp_path / _bundle_filename(release, platform)
    with zipfile.ZipFile(archive, "w") as zf:
        for artifact in bridge_assets.BUNDLED_ARTIFACTS:
            zf.writestr(bridge_assets.artifact_filename(artifact, platform),
                        _stub_payload(artifact))
    result = subprocess.run(
        [sys.executable, str(BUNDLE_TOOL), "pin", "--release", release,
         "--source-rev", SOURCE_REV,
         "--bundle", str(archive), "--out", str(tmp_path / "pins.json")],
        capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode != 0
    assert "no assets/ members" in (result.stdout + result.stderr)
    assert not (tmp_path / "pins.json").exists()


# ---------------------------------------------------------------------------
# Staging puts them where the renderer looks
# ---------------------------------------------------------------------------

def _pin_document(tmp_path: Path, archive: Path, release: str) -> dict:
    out = tmp_path / "pins.json"
    result = subprocess.run(
        [sys.executable, str(BUNDLE_TOOL), "pin", "--release", release,
         "--source-rev", SOURCE_REV,
         "--bundle", str(archive), "--out", str(out)],
        capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(out.read_text(encoding="utf-8"))


def test_staging_writes_the_assets_where_the_renderer_already_looks(
        tmp_path, monkeypatch):
    """The staged path must be on ``rw_wrfbatch``'s own resolution ladder.

    Not "somewhere gpuwm can find and pass along": the binary resolves
    ``assets/basemap`` under its own ancestors, so a bundle staged into
    ``~/.gpuwm/bridges`` is found by a renderer nobody configured.  That
    is what makes the two halves inseparable in practice rather than
    only by intention.
    """

    _require_source_assets()
    release = "v0-assets"
    archive = _pack(tmp_path, release=release)
    document = _pin_document(tmp_path, archive, release)
    pins = bridge_assets.parse_pins(document)
    bundle = pins.platforms["linux-x86_64"]
    assert bundle.assets, "the pinned bundle declares no assets"

    dest = tmp_path / "bridges"
    bridge_assets.stage_from_bundle(archive, bundle, dest,
                                    progress=lambda _msg: None)

    staged_root = dest / bridge_assets.ASSET_ROOT / "basemap"
    assert staged_root.is_dir()
    assert (staged_root / "natural_earth_10m" / "ne_10m_coastline.shp"
            ).is_file()

    # Every pinned asset landed, byte for byte.
    for pin in bundle.assets:
        path = dest / pin.path
        assert path.is_file(), f"{pin.path} was not staged"
        assert path.stat().st_size == pin.bytes
        assert hashlib.sha256(path.read_bytes()).hexdigest() == pin.sha256

    # And the renderer's ladder resolves it, with no environment help.
    monkeypatch.delenv("RUSTWX_BASEMAP_DIR", raising=False)
    monkeypatch.delenv("RUSTWX_ASSETS_DIR", raising=False)
    renderer = dest / "rw_wrfbatch"
    assert staged_root in rustwx.basemap_candidates(renderer)
    assert rustwx.resolve_basemap_dir(renderer) == staged_root


def test_a_corrupt_asset_is_refused_rather_than_staged(tmp_path):
    """Asset bytes are verified exactly as binary bytes are."""

    _require_source_assets()
    release = "v0-assets"
    archive = _pack(tmp_path, release=release)
    document = _pin_document(tmp_path, archive, release)
    pins = bridge_assets.parse_pins(document)
    bundle = pins.platforms["linux-x86_64"]

    victim = bundle.assets[0]
    tampered = bridge_assets.BundlePin(
        platform=bundle.platform, filename=bundle.filename,
        bytes=bundle.bytes, sha256=bundle.sha256, binaries=bundle.binaries,
        assets=(bridge_assets.AssetPin(
            path=victim.path, bytes=victim.bytes,
            sha256="0" * 64),) + bundle.assets[1:])
    dest = tmp_path / "bridges"
    with pytest.raises(bridge_assets.BridgeAssetError, match="SHA-256"):
        bridge_assets.stage_from_bundle(archive, tampered, dest,
                                        progress=lambda _msg: None)
    assert not (dest / victim.path).exists()


@pytest.mark.parametrize("escape", [
    "../outside.shp",
    "assets/../../outside.shp",
    "/etc/passwd",
    "assets/basemap/../../../outside.shp",
    "C:/windows/system32/evil.dll",
    "assets\\basemap\\evil.shp",
    "basemap/no-asset-root.shp",
])
def test_a_pinned_asset_path_cannot_escape_the_destination(escape):
    """The pins document decides where staging writes, so it is checked."""

    payload = {
        "schema": bridge_assets.PINS_SCHEMA, "release": "v0",
        "platforms": {"linux-x86_64": {
            "bundle": {"filename": "b.zip", "bytes": 1, "sha256": "a" * 64},
            "binaries": [{"artifact": "grib1_bridge",
                          "filename": "grib1_bridge",
                          "bytes": 1, "sha256": "b" * 64}],
            "assets": [{"path": escape, "bytes": 1, "sha256": "c" * 64}]}},
    }
    with pytest.raises(bridge_assets.BridgeAssetError, match="asset path"):
        bridge_assets.parse_pins(payload)


def test_a_bundle_missing_a_pinned_asset_is_refused(tmp_path):
    """A half-built bundle must not stage its binaries and shrug."""

    _require_source_assets()
    release = "v0-assets"
    archive = _pack(tmp_path, release=release)
    document = _pin_document(tmp_path, archive, release)
    pins = bridge_assets.parse_pins(document)
    bundle = pins.platforms["linux-x86_64"]
    invented = bridge_assets.BundlePin(
        platform=bundle.platform, filename=bundle.filename,
        bytes=bundle.bytes, sha256=bundle.sha256, binaries=bundle.binaries,
        assets=bundle.assets + (bridge_assets.AssetPin(
            path="assets/basemap/never_packed.shp", bytes=1,
            sha256="d" * 64),))
    with pytest.raises(bridge_assets.BridgeAssetError,
                       match="never_packed.shp"):
        bridge_assets.stage_from_bundle(archive, invented,
                                        tmp_path / "bridges",
                                        progress=lambda _msg: None)


def test_current_binaries_with_missing_assets_are_reported_not_skipped(
        tmp_path, monkeypatch, capsys):
    """The upgrade path out of 1.4.0, in the words a user needs.

    Someone who already ran ``fetch-bridges`` has eight current binaries
    and no assets.  ``nothing to fetch`` would leave them rendering
    blank maps forever, so the asset gap alone must drive a re-stage and
    say why.
    """

    _require_source_assets()
    release = "v0-assets"
    archive = _pack(tmp_path, release=release)
    document = _pin_document(tmp_path, archive, release)
    pins = bridge_assets.parse_pins(document)
    bundle = pins.platforms["linux-x86_64"]

    dest = tmp_path / "bridges"
    bridge_assets.stage_from_bundle(archive, bundle, dest,
                                    progress=lambda _msg: None)
    # Exactly the 1.4.0 end state: binaries staged, assets absent.
    import shutil

    shutil.rmtree(dest / bridge_assets.ASSET_ROOT)
    staged, stale, absent = bridge_assets.classify_destination(dest, bundle)
    assert not stale and not absent, "the binaries should still be current"
    held, _stale_assets, absent_assets = bridge_assets.classify_assets(
        dest, bundle)
    assert not held and absent_assets

    monkeypatch.setattr(bridge_assets, "load_pins", lambda path=None: pins)
    monkeypatch.setattr(bridge_assets, "host_platform",
                        lambda: "linux-x86_64")
    import argparse

    args = argparse.Namespace(from_dir=str(archive.parent), dest=str(dest),
                              keep_bundle=False, list=False)
    assert bridge_assets.fetch_bridges_main(args) == 0
    out = capsys.readouterr().out
    assert "map asset" in out
    assert "no coastlines" in out
    for pin in bundle.assets:
        assert bridge_assets.matches_pin(dest / pin.path, pin)


# ---------------------------------------------------------------------------
# The silent state must stop being silent
# ---------------------------------------------------------------------------

def test_a_renderer_with_no_basemaps_warns_before_it_draws(tmp_path,
                                                           monkeypatch):
    """The other way to ship a plot believing it is something else.

    Delivery is fixed, but an install that staged its binaries under
    1.4.0 and never re-ran ``fetch-bridges`` still has eight executables
    and no assets.  That state renders successfully, exits zero, and
    produces blank geography -- so it must say so.
    """

    from gpuwm import render

    monkeypatch.delenv("RUSTWX_BASEMAP_DIR", raising=False)
    monkeypatch.delenv("RUSTWX_ASSETS_DIR", raising=False)
    # A renderer with nothing on its ladder: no assets/basemap under any
    # ancestor of the binary, and a working directory equally bare.
    bridges = tmp_path / "deep" / "bridges"
    bridges.mkdir(parents=True)
    renderer = bridges / "rw_wrfbatch"
    renderer.write_bytes(b"stub")
    workdir = tmp_path / "deep" / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    # The checkout fallback (basemap_dir) is the last rung; point it away.
    monkeypatch.setattr(rustwx, "basemap_dir", lambda: tmp_path / "nowhere")
    # ... and away from the cartopy cache this workstation may well have,
    # which the renderer would otherwise draw from.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))

    notice = render.missing_basemap_notice(renderer)
    assert notice is not None
    assert "no coastlines" in notice
    # Actionable: one sentence, one remedy, no multi-line bootstrap.
    assert notice.count("\n") == 0
    assert "fetch-bridges" in notice or "RUSTWX_BASEMAP_DIR" in notice

    # And silence once the assets are where the renderer looks.
    staged = bridges / "assets" / "basemap"
    staged.mkdir(parents=True)
    assert render.missing_basemap_notice(renderer) is None
    # A renderer that does not exist at all is the fallback notice's
    # business, not this one's.
    assert render.missing_basemap_notice(None) is None


# ---------------------------------------------------------------------------
# Discoverability: the catalog without a file
# ---------------------------------------------------------------------------

def test_the_product_catalog_is_listable_without_a_wrfout(capsys, monkeypatch):
    """"What may I put in --products?" is a question about the build.

    It used to be answerable only by reading the source or by already
    having a wrfout: ``gpuwm render --list-products`` alone refused with
    "at least one WRFOUT file is required".  A forecaster asking which
    products exist is precisely someone who has not run anything yet.
    """

    from gpuwm import render

    args = argparse.Namespace(
        wrfout=[], list_products=True, pair=None, engine="matplotlib",
        products="all", timeidx="all", out=Path("out"), dpi=150,
        size="1200x900", heavy=False, source_label="ArWen", explain=False)
    assert render.render_main(args) == 0
    out = capsys.readouterr().out
    assert "product catalog" in out
    # The matplotlib engine's own products, from its own declaration.
    for product in render.PRODUCTS:
        assert product in out


def test_the_rust_catalog_comes_from_the_renderer_not_a_copy(monkeypatch):
    """A second copy of the catalog in Python is drift waiting to happen.

    The rust catalog belongs to the renderer, which already answers
    ``--list-products`` with no inputs.  This asserts the Python half
    asks it rather than keeping a list -- the same discipline the
    ``--products`` parser already follows by passing unknown slugs
    straight through for the renderer's strict validation.
    """

    import inspect

    from gpuwm import render

    source = inspect.getsource(render.catalog_main)
    assert "--list-products" in source
    assert "find_renderer" in source
    # No literal rust slug may appear in this module's catalog path.
    for slug in ("sbcape", "srh_0_1km", "composite_reflectivity"):
        assert slug not in source


def test_the_cartopy_cache_counts_as_geography_and_silences_the_warning(
        tmp_path, monkeypatch):
    """The fallback that hid this bug for a release must not be ignored.

    ``rustwx-render`` falls back to $HOME/.local/share/cartopy, so a
    workstation that has ever run cartopy draws perfectly good
    coastlines.  Warning there would be a false alarm on every developer
    machine, and a notice that cries wolf is a notice nobody reads.
    """

    from gpuwm import render

    monkeypatch.delenv("RUSTWX_BASEMAP_DIR", raising=False)
    monkeypatch.delenv("RUSTWX_ASSETS_DIR", raising=False)
    bridges = tmp_path / "deep" / "bridges"
    bridges.mkdir(parents=True)
    renderer = bridges / "rw_wrfbatch"
    renderer.write_bytes(b"stub")
    workdir = tmp_path / "deep" / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(rustwx, "basemap_dir", lambda: tmp_path / "nowhere")

    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    assert rustwx.cartopy_natural_earth_root() is None
    assert render.missing_basemap_notice(renderer) is not None

    (home / ".local" / "share" / "cartopy" / "shapefiles"
     / "natural_earth").mkdir(parents=True)
    assert rustwx.cartopy_natural_earth_root() is not None
    assert render.missing_basemap_notice(renderer) is None
