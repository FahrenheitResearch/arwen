"""``gpuwm fetch-geog``: staging, verification, and refusal contract.

The WPS_GEOG tree is user-staged data the whole static path depends on;
fetch-geog is its installer.  These tests bind the pin table to the
static builder's dataset list, exercise the resumable download and
skip-complete logic against a fake transport, prove the extraction
safety and index-validation bars on tiny synthetic tarballs, and pin
the doctor/wizard remedy text to the command.  A live network smoke is
marked ``network`` and additionally gated on ``GPUWM_NETWORK_TESTS=1``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path

import pytest

from gpuwm import geog_assets
from gpuwm.domain_wizard import GEOG_DATASETS
from gpuwm.geog_assets import (
    GEOG_FETCH_MANIFEST_NAME,
    GEOG_FETCH_MANIFEST_SCHEMA,
    GeogArchive,
    GeogFetchError,
    archive_url,
    default_geog_root,
    fetch_geog,
    parse_datasets,
    resolve_source,
    validate_dataset_dir,
)

_INDEX_TEXT = """\
type = continuous
projection = regular_ll
dx = 0.5
dy = 0.5
known_x = 1.0
known_y = 1.0
known_lat = -89.75
known_lon = -179.75
wordsize = 2
tile_x = 4
tile_y = 4
tile_z = 1
"""


# ---------------------------------------------------------------------------
# Synthetic archive construction + fake transport
# ---------------------------------------------------------------------------

def _build_archive(datasets: tuple[str, ...], *, mode: str = "w:bz2",
                   with_index: bool = True,
                   evil_member: str | None = None,
                   prefix: str = "") -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode=mode) as tar:
        for bare in datasets:
            name = f"{prefix}{bare}"
            info = tarfile.TarInfo(name)
            info.type = tarfile.DIRTYPE
            tar.addfile(info)
            tile = b"\x00\x01" * 16
            data = tarfile.TarInfo(f"{name}/00001-00004.00001-00004")
            data.size = len(tile)
            tar.addfile(data, io.BytesIO(tile))
            if with_index:
                payload = _INDEX_TEXT.encode()
                index = tarfile.TarInfo(f"{name}/index")
                index.size = len(payload)
                tar.addfile(index, io.BytesIO(payload))
        if evil_member is not None:
            info = tarfile.TarInfo(evil_member)
            info.size = 4
            tar.addfile(info, io.BytesIO(b"evil"))
    return buffer.getvalue()


def _add_sidecars(datasets: tuple[str, ...], *, mode: str = "w:bz2",
                  prefix: str = "") -> bytes:
    """An archive with AppleDouble junk beside real members (as NCAR's
    mandatory bundle genuinely carries)."""
    base = _build_archive(datasets, mode=mode, prefix=prefix)
    buffer = io.BytesIO(base)
    out = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="r:*") as src, \
            tarfile.open(fileobj=out, mode=mode) as dst:
        for member in src:
            payload = src.extractfile(member) if member.isfile() else None
            dst.addfile(member, payload)
        for name in datasets:
            junk = tarfile.TarInfo(f"{prefix}._{name}")
            junk.size = 3
            dst.addfile(junk, io.BytesIO(b"AD\x00"))
            inner = tarfile.TarInfo(f"{prefix}{name}/._index")
            inner.size = 3
            dst.addfile(inner, io.BytesIO(b"AD\x00"))
    return out.getvalue()


def _pin(dataset: str, payload: bytes,
         filename: str | None = None) -> GeogArchive:
    return GeogArchive(dataset, filename or f"{dataset}.tar.bz2",
                       len(payload), hashlib.sha256(payload).hexdigest(),
                       4096)


class _FakeResponse:
    """A stub that answers like a real origin, headers included.

    A ``206`` without ``Content-Range`` is not a thing HTTP allows, and
    a stub that omits it lets a resume gate pass vacuously -- the
    transport is supposed to check that the server is sending the bytes
    it asked for, and it cannot check a header the stub never sends.
    """

    def __init__(self, payload: bytes, status: int = 200,
                 headers: dict[str, str] | None = None):
        self._data = io.BytesIO(payload)
        self.status = status
        self.headers = dict(headers or {})

    def read(self, n: int = -1) -> bytes:
        return self._data.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_transport(files: dict[str, bytes], *, honor_range: bool = True,
                    etag: str = '"pinned-object-v1"'):
    """urlopen_fn serving ``files`` by basename, recording every request."""

    calls: list = []

    def urlopen_fn(request):
        calls.append(request)
        name = request.full_url.rsplit("/", 1)[1]
        payload = files[name]
        header = request.headers.get("Range")
        if header and honor_range:
            offset = int(header.split("=", 1)[1].rstrip("-"))
            return _FakeResponse(payload[offset:], status=206, headers={
                "ETag": etag,
                "Content-Range":
                    f"bytes {offset}-{len(payload) - 1}/{len(payload)}",
            })
        return _FakeResponse(payload, headers={"ETag": etag})

    urlopen_fn.calls = calls
    return urlopen_fn


def _refusing_transport():
    def urlopen_fn(request):  # pragma: no cover - the assertion IS the test
        raise AssertionError(f"network touched: {request.full_url}")
    return urlopen_fn


# ---------------------------------------------------------------------------
# Pin table and argument parsing
# ---------------------------------------------------------------------------

def test_pin_table_binds_the_static_builders_dataset_list():
    # The WRF static builder's datasets must every one be fetchable.  The
    # pin table is allowed to be wider than that list -- the MPAS static
    # builder reads Noah-MP soil products the WRF one never opens -- so the
    # bar is containment, not equality.  Equality would make adding a
    # dataset for one builder look like a defect in the other.
    fetchable = geog_assets.geog_datasets()
    missing = [name for name in GEOG_DATASETS if name not in fetchable]
    assert not missing, f"unfetchable static-builder dataset(s): {missing}"
    assert len(set(fetchable)) == len(fetchable), "duplicate pin"
    # Which door needs a pin is a COLUMN of the table, not a branch in the
    # code that reads it.  The WRF-scoped column is exactly the nine the
    # WRF static builder opens: doctor's WRF-side completeness verdict is
    # read straight off it, and a pin drifting into that column would have
    # doctor demand a dataset that builder never opens.
    assert (geog_assets.datasets_required_by(geog_assets.GEOG_CONSUMER_WRF)
            == GEOG_DATASETS)
    for archive in geog_assets.GEOG_ARCHIVES:
        assert archive.required_by, f"{archive.dataset} serves no door"
        unknown = set(archive.required_by) - set(geog_assets.GEOG_CONSUMERS)
        assert not unknown, f"{archive.dataset}: unknown consumer {unknown}"


def test_the_mesh_column_carries_every_dataset_the_mesh_door_opens():
    # rw_mpas_static's required list is dataset DIRECTORIES; the pin table
    # is ARCHIVES, and one archive carries seven directories.  The two are
    # bound through the pin table's own child declarations, so a dataset
    # added to the engine's list without a mesh-scoped pin behind it is a
    # tree doctor calls complete and the door then refuses.
    from gpuwm.rustwx_static import REQUIRED_GEOG_DATASETS, fetchable_for

    mesh_pins = set(geog_assets.datasets_required_by(
        geog_assets.GEOG_CONSUMER_MESH))
    assert set(fetchable_for(REQUIRED_GEOG_DATASETS)) == mesh_pins
    assert "soilgrids" in mesh_pins


def test_the_soil_archive_is_required_by_the_mesh_door_only():
    # The ruling: 13 GB unpacked is a sanctioned requirement for the mesh
    # path, and invisible to a WRF-only install.
    archive = geog_assets.archive_for("soilgrids")
    assert archive.required_by == (geog_assets.GEOG_CONSUMER_MESH,)
    assert "soilgrids" not in geog_assets.datasets_required_by(
        geog_assets.GEOG_CONSUMER_WRF)


def test_the_noahmp_soil_container_is_pinned_and_declares_its_children():
    # rw_mpas_static requires soilcomp and four soil texture layers.  They
    # arrive in one upstream archive as child directories, each with its
    # own WPS index, so the pin has to name the children or a staged tree
    # cannot be validated.
    archive = geog_assets.archive_for("soilgrids")
    assert archive.filename == "soilgrids.tar.bz2"
    assert set(archive.index_subdirs) == {
        "soilcomp", "texture_top", "texture_bot",
        "texture_layer1", "texture_layer2",
        "texture_layer3", "texture_layer4",
    }
    # The five rw_mpas_static reads must each be a declared child.
    for child in ("soilcomp", "texture_layer1", "texture_layer2",
                  "texture_layer3", "texture_layer4"):
        assert child in archive.index_subdirs


def test_the_reader_finds_every_mesh_dataset_where_the_fetcher_stages_it(
        tmp_path):
    """The fetcher's layout and the door's reader, bound through one table.

    The fetcher validates a container archive child by child at
    ``root/<archive>/<child>/index``; the reader behind doctor's mesh row
    and `gpuwm mesh`'s refusal used to look at ``root/<child>/index``
    alone.  Both passed their own tests and every install that followed
    doctor's remedy was told it was still five datasets short.  This
    stages a tree the fetcher's own bar accepts and asks the reader.
    """

    from gpuwm import rustwx_static

    root = tmp_path / "WPS_GEOG"
    for archive in geog_assets.GEOG_ARCHIVES:
        if geog_assets.GEOG_CONSUMER_MESH not in archive.required_by:
            continue
        for child in archive.index_subdirs or ("",):
            directory = root / archive.dataset / child
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "index").write_text(_INDEX_TEXT)
        ok, detail = validate_dataset_dir(root, archive.dataset,
                                          check_receipt=False)
        assert ok, detail
    assert rustwx_static.missing_geog_datasets(root) == []
    for name in rustwx_static.REQUIRED_GEOG_DATASETS:
        assert rustwx_static.geog_dataset_dir(root, name) is not None, name


def test_the_mandatory_bundle_claims_only_what_was_walked():
    # --bundle serves the nine the 2026-07-29 walk found.  A pin added
    # later must not silently widen that claim, or --bundle extracts
    # nothing for it and reports success.
    assert geog_assets.MANDATORY_BUNDLE_DATASETS == frozenset(GEOG_DATASETS)
    assert "soilgrids" not in geog_assets.MANDATORY_BUNDLE_DATASETS


def test_single_dataset_pins_declare_no_children():
    for archive in geog_assets.GEOG_ARCHIVES:
        if archive.dataset in GEOG_DATASETS:
            assert archive.index_subdirs == (), archive.dataset


def test_pins_are_real_digests_and_positive_sizes():
    for archive in geog_assets.GEOG_ARCHIVES:
        assert len(archive.archive_sha256) == 64, archive.dataset
        int(archive.archive_sha256, 16)
        assert archive.archive_bytes > 0
        assert archive.extracted_bytes > archive.archive_bytes
        assert archive.filename == f"{archive.dataset}.tar.bz2"
    assert len(geog_assets.MANDATORY_BUNDLE_SHA256) == 64
    int(geog_assets.MANDATORY_BUNDLE_SHA256, 16)


def test_parse_datasets_all_and_subset_and_unknown():
    assert parse_datasets("all") == geog_assets.geog_datasets()
    # canonical order regardless of request order, duplicates collapsed
    picked = parse_datasets("soiltemp_1deg,topo_gmted2010_30s,soiltemp_1deg")
    assert picked == ("topo_gmted2010_30s", "soiltemp_1deg")
    with pytest.raises(ValueError, match="unknown dataset"):
        parse_datasets("topo_gmted2010_30s,not_a_dataset")
    with pytest.raises(ValueError, match="comma-separated"):
        parse_datasets(" , ")


def test_the_default_fetch_stages_the_soil_archive_too():
    # The ruling made the ~13 GB soil archive a sanctioned default
    # download for the mesh path.  A bare `gpuwm fetch-geog` -- whose
    # --datasets default is what this parses -- must therefore include it,
    # or `gpuwm mesh` refuses a tree the installer just called done.
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    geog_assets.register_cli(sub)
    args = parser.parse_args(["fetch-geog"])
    assert parse_datasets(args.datasets) == geog_assets.geog_datasets()
    assert "soilgrids" in parse_datasets(args.datasets)


def test_a_consumer_name_selects_that_doors_datasets():
    # The subset spelling for an operator who knows they only want WRF:
    # a consumer name off the table's own scoping column, not a per-door
    # flag.  `--datasets wrf` is the WRF-only escape from the 13 GB.
    assert parse_datasets("wrf") == GEOG_DATASETS
    assert "soilgrids" not in parse_datasets("wrf")
    assert parse_datasets("mesh") == geog_assets.datasets_required_by(
        geog_assets.GEOG_CONSUMER_MESH)
    # and it composes with explicit names, in canonical order
    assert parse_datasets("wrf,soilgrids") == geog_assets.geog_datasets()


def test_resolve_source_defaults_and_bundle_rules():
    assert resolve_source(None, False) == "hf"
    assert resolve_source("ncar", False) == "ncar"
    assert resolve_source(None, True) == "ncar"  # bundle implies ncar
    with pytest.raises(ValueError, match="NCAR only"):
        resolve_source("hf", True)
    with pytest.raises(ValueError, match="unknown --source"):
        resolve_source("ftp", False)


def test_archive_url_per_source_and_env_override(monkeypatch):
    monkeypatch.delenv(geog_assets.GEOG_URL_BASE_ENV, raising=False)
    assert archive_url("x.tar.bz2", "hf") == (
        f"{geog_assets.HF_MIRROR_BASE_URL}/x.tar.bz2")
    assert archive_url("x.tar.bz2", "ncar") == (
        f"{geog_assets.NCAR_BASE_URL}/x.tar.bz2")
    monkeypatch.setenv(geog_assets.GEOG_URL_BASE_ENV,
                       "https://mirror.example/geog/")
    assert archive_url("x.tar.bz2", "hf") == (
        "https://mirror.example/geog/x.tar.bz2")
    assert archive_url("x.tar.bz2", "ncar") == (
        "https://mirror.example/geog/x.tar.bz2")


def test_default_geog_root_is_the_case_data_layout(monkeypatch, tmp_path):
    monkeypatch.setenv("GPUWM_CASE_DATA_ROOT", str(tmp_path))
    assert default_geog_root() == tmp_path / "WPS_GEOG"


# ---------------------------------------------------------------------------
# Dataset directory validation (the doctor bar, plus a parsing index)
# ---------------------------------------------------------------------------

def test_validate_dataset_dir_bars(tmp_path):
    ok, detail = validate_dataset_dir(tmp_path, "absent_ds")
    assert not ok and "does not exist" in detail

    (tmp_path / "noindex_ds").mkdir()
    ok, detail = validate_dataset_dir(tmp_path, "noindex_ds")
    assert not ok and "index" in detail

    hollow = tmp_path / "hollow_ds"
    hollow.mkdir()
    (hollow / "index").write_text("")  # vacuous: no keys at all
    ok, detail = validate_dataset_dir(tmp_path, "hollow_ds")
    assert not ok and "required WPS index key" in detail

    good = tmp_path / "good_ds"
    good.mkdir()
    (good / "index").write_text(_INDEX_TEXT)
    ok, detail = validate_dataset_dir(tmp_path, "good_ds")
    assert ok, detail


def test_validate_dataset_dir_checks_every_declared_child(tmp_path, monkeypatch):
    # A container archive's index lives one level down, once per child.
    # Every declared child has to pass: a tree missing one is a finding,
    # not a pass earned by its siblings.
    monkeypatch.setattr(
        geog_assets, "GEOG_ARCHIVES",
        (GeogArchive("box_ds", "box_ds.tar.bz2", 10, "0" * 64, 20,
                     ("first", "second")),))

    box = tmp_path / "box_ds"
    (box / "first").mkdir(parents=True)
    ok, detail = validate_dataset_dir(tmp_path, "box_ds", check_receipt=False)
    assert not ok and "lacks its WPS `index` file" in detail

    (box / "first" / "index").write_text(_INDEX_TEXT)
    (box / "second").mkdir()
    ok, detail = validate_dataset_dir(tmp_path, "box_ds", check_receipt=False)
    assert not ok and "second" in detail, detail

    (box / "second" / "index").write_text(_INDEX_TEXT)
    ok, detail = validate_dataset_dir(tmp_path, "box_ds", check_receipt=False)
    assert ok, detail

    # A top-level index does not substitute for the children's.
    (box / "second" / "index").unlink()
    (box / "index").write_text(_INDEX_TEXT)
    ok, detail = validate_dataset_dir(tmp_path, "box_ds", check_receipt=False)
    assert not ok and "second" in detail, detail


# ---------------------------------------------------------------------------
# The engine: skip, download, verify, extract, manifest
# ---------------------------------------------------------------------------

def test_fetch_stages_verifies_and_writes_manifest(tmp_path, monkeypatch):
    payload = _build_archive(("alpha_ds",))
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES",
                        (_pin("alpha_ds", payload),))
    transport = _fake_transport({"alpha_ds.tar.bz2": payload})
    root = tmp_path / "WPS_GEOG"

    staged = fetch_geog(root=root, datasets=("alpha_ds",), source="hf",
                        progress=lambda *_: None, urlopen_fn=transport)

    assert staged == 1
    ok, detail = validate_dataset_dir(root, "alpha_ds")
    assert ok, detail
    manifest = json.loads(
        (root / GEOG_FETCH_MANIFEST_NAME).read_text())
    assert manifest["schema"] == GEOG_FETCH_MANIFEST_SCHEMA
    entry = manifest["archives"]["alpha_ds.tar.bz2"]
    assert entry["archive_sha256"] == hashlib.sha256(payload).hexdigest()
    assert entry["pinned"] is True
    assert entry["source"] == "hf"
    assert entry["datasets"]["alpha_ds"]["files"] == 2
    # default: the verified archive is removed after extraction
    assert not (root / geog_assets.ARCHIVE_SUBDIR
                / "alpha_ds.tar.bz2").exists()
    # no extraction droppings
    assert not list(root.glob(f"{geog_assets.ARCHIVE_SUBDIR}-extract-*"))


def test_fetch_prints_the_total_before_the_first_byte_moves(
        tmp_path, monkeypatch):
    """The size total is announced up front, not discovered per dataset.

    ``gpuwm setup --with-geog`` promises "the size is printed before
    anything downloads"; the engine both entry points share must keep
    that promise for ``gpuwm fetch-geog`` too -- the first field run
    watched 16 GB arrive with the only sizes printed as each dataset
    landed.  The total covers exactly the datasets that still need
    staging, not ones already valid on disk.
    """
    alpha = _build_archive(("alpha_ds",))
    beta = _build_archive(("beta_ds",))
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES",
                        (_pin("alpha_ds", alpha), _pin("beta_ds", beta)))
    root = tmp_path / "WPS_GEOG"
    staged_dir = root / "alpha_ds"          # alpha is already valid,
    staged_dir.mkdir(parents=True)          # so only beta counts
    (staged_dir / "index").write_text(_INDEX_TEXT)

    events: list[str] = []
    inner = _fake_transport({"beta_ds.tar.bz2": beta})

    def transport(request):
        events.append("NETWORK")
        return inner(request)

    fetch_geog(root=root, datasets=("alpha_ds", "beta_ds"), source="hf",
               progress=events.append, urlopen_fn=transport)

    totals = [line for line in events
              if isinstance(line, str) and "need staging" in line]
    assert len(totals) == 1, events
    total = totals[0]
    # Only beta's pin is in the total, and both numbers are stated.
    assert "1 of 2" in total
    assert f"{len(beta):,} B" in total          # download size
    assert "~4.0 KiB" in total                  # extracted_bytes pin (4096)
    # Printed before the first byte moved.
    assert events.index(total) < events.index("NETWORK")


def test_fetch_skips_already_valid_datasets_without_network(
        tmp_path, monkeypatch):
    payload = _build_archive(("alpha_ds",))
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES",
                        (_pin("alpha_ds", payload),))
    root = tmp_path / "WPS_GEOG"
    staged_dir = root / "alpha_ds"
    staged_dir.mkdir(parents=True)
    (staged_dir / "index").write_text(_INDEX_TEXT)

    staged = fetch_geog(root=root, datasets=("alpha_ds",), source="hf",
                        progress=lambda *_: None,
                        urlopen_fn=_refusing_transport())
    assert staged == 0


def test_fetch_resumes_a_partial_download_with_a_range_request(
        tmp_path, monkeypatch):
    payload = _build_archive(("alpha_ds",))
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES",
                        (_pin("alpha_ds", payload),))
    transport = _fake_transport({"alpha_ds.tar.bz2": payload})
    root = tmp_path / "WPS_GEOG"
    partial = root / geog_assets.ARCHIVE_SUBDIR / "alpha_ds.tar.bz2"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(payload[:100])

    staged = fetch_geog(root=root, datasets=("alpha_ds",), source="hf",
                        progress=lambda *_: None, urlopen_fn=transport)

    assert staged == 1
    (request,) = transport.calls
    assert request.headers.get("Range") == "bytes=100-"
    ok, _ = validate_dataset_dir(root, "alpha_ds")
    assert ok


def test_fetch_restarts_cleanly_when_the_server_ignores_range(
        tmp_path, monkeypatch):
    payload = _build_archive(("alpha_ds",))
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES",
                        (_pin("alpha_ds", payload),))
    transport = _fake_transport({"alpha_ds.tar.bz2": payload},
                                honor_range=False)
    root = tmp_path / "WPS_GEOG"
    partial = root / geog_assets.ARCHIVE_SUBDIR / "alpha_ds.tar.bz2"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"garbage-prefix")

    staged = fetch_geog(root=root, datasets=("alpha_ds",), source="hf",
                        progress=lambda *_: None, urlopen_fn=transport)
    assert staged == 1
    ok, _ = validate_dataset_dir(root, "alpha_ds")
    assert ok


def test_fetch_reuses_a_complete_verified_archive(tmp_path, monkeypatch):
    payload = _build_archive(("alpha_ds",))
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES",
                        (_pin("alpha_ds", payload),))
    root = tmp_path / "WPS_GEOG"
    complete = root / geog_assets.ARCHIVE_SUBDIR / "alpha_ds.tar.bz2"
    complete.parent.mkdir(parents=True)
    complete.write_bytes(payload)

    staged = fetch_geog(root=root, datasets=("alpha_ds",), source="hf",
                        progress=lambda *_: None,
                        urlopen_fn=_refusing_transport())
    assert staged == 1
    ok, _ = validate_dataset_dir(root, "alpha_ds")
    assert ok


def test_sha_mismatch_refuses_and_quarantines(tmp_path, monkeypatch):
    payload = _build_archive(("alpha_ds",))
    wrong = payload + b"tail"
    pin = _pin("alpha_ds", payload)
    pin = GeogArchive(pin.dataset, pin.filename, len(wrong),
                      pin.archive_sha256, pin.extracted_bytes)
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES", (pin,))
    transport = _fake_transport({"alpha_ds.tar.bz2": wrong})
    root = tmp_path / "WPS_GEOG"

    with pytest.raises(GeogFetchError, match="does not match the pin"):
        fetch_geog(root=root, datasets=("alpha_ds",), source="hf",
                   progress=lambda *_: None, urlopen_fn=transport)
    assert not (root / "alpha_ds").exists()
    rejected = list((root / geog_assets.ARCHIVE_SUBDIR).glob("*.rejected-*"))
    assert len(rejected) == 1


def test_upstream_drift_is_ncar_only_and_band_limited(tmp_path, monkeypatch):
    payload = _build_archive(("alpha_ds",))
    pin = GeogArchive("alpha_ds", "alpha_ds.tar.bz2", len(payload),
                      "0" * 64, 4096)  # pin that can never match
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES", (pin,))
    root = tmp_path / "WPS_GEOG"

    # hf: drift is corruption by definition, flag or no flag
    with pytest.raises(GeogFetchError, match="corrupted transfer"):
        fetch_geog(root=root, datasets=("alpha_ds",), source="hf",
                   allow_drift=True, progress=lambda *_: None,
                   urlopen_fn=_fake_transport(
                       {"alpha_ds.tar.bz2": payload}))

    # ncar without the flag: refused with the drift story
    with pytest.raises(GeogFetchError, match="allow-upstream-drift"):
        fetch_geog(root=root, datasets=("alpha_ds",), source="ncar",
                   progress=lambda *_: None,
                   urlopen_fn=_fake_transport(
                       {"alpha_ds.tar.bz2": payload}))

    # ncar with the flag, size inside the band: accepted, marked unpinned
    staged = fetch_geog(root=root, datasets=("alpha_ds",), source="ncar",
                        allow_drift=True, progress=lambda *_: None,
                        urlopen_fn=_fake_transport(
                            {"alpha_ds.tar.bz2": payload}))
    assert staged == 1
    manifest = json.loads((root / GEOG_FETCH_MANIFEST_NAME).read_text())
    assert manifest["archives"]["alpha_ds.tar.bz2"]["pinned"] is False


def test_upstream_drift_outside_the_size_band_refuses(tmp_path, monkeypatch):
    payload = _build_archive(("alpha_ds",))
    # pin claims the archive should be far larger than what arrives:
    # with drift allowed the exact-size bar is off, so the verify
    # step's sanity band is what must catch the runt
    pin = GeogArchive("alpha_ds", "alpha_ds.tar.bz2", len(payload) * 10,
                      "0" * 64, 4096)
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES", (pin,))
    root = tmp_path / "WPS_GEOG"
    with pytest.raises(GeogFetchError, match="outside the sanity band"):
        fetch_geog(root=root, datasets=("alpha_ds",), source="ncar",
                   allow_drift=True, progress=lambda *_: None,
                   urlopen_fn=_fake_transport(
                       {"alpha_ds.tar.bz2": payload}))
    assert not (root / "alpha_ds").exists()


def test_keep_archives_retains_the_verified_tarball(tmp_path, monkeypatch):
    payload = _build_archive(("alpha_ds",))
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES",
                        (_pin("alpha_ds", payload),))
    root = tmp_path / "WPS_GEOG"
    fetch_geog(root=root, datasets=("alpha_ds",), source="hf",
               keep_archives=True, progress=lambda *_: None,
               urlopen_fn=_fake_transport({"alpha_ds.tar.bz2": payload}))
    kept = root / geog_assets.ARCHIVE_SUBDIR / "alpha_ds.tar.bz2"
    assert kept.read_bytes() == payload


# ---------------------------------------------------------------------------
# Extraction safety and validation bars
# ---------------------------------------------------------------------------

def test_archive_without_an_index_is_refused(tmp_path, monkeypatch):
    payload = _build_archive(("alpha_ds",), with_index=False)
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES",
                        (_pin("alpha_ds", payload),))
    root = tmp_path / "WPS_GEOG"
    with pytest.raises(GeogFetchError, match="failed validation"):
        fetch_geog(root=root, datasets=("alpha_ds",), source="hf",
                   progress=lambda *_: None,
                   urlopen_fn=_fake_transport(
                       {"alpha_ds.tar.bz2": payload}))
    assert not (root / "alpha_ds").exists()


def test_traversal_member_is_refused(tmp_path, monkeypatch):
    payload = _build_archive(("alpha_ds",),
                             evil_member="alpha_ds/../../evil")
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES",
                        (_pin("alpha_ds", payload),))
    root = tmp_path / "WPS_GEOG"
    with pytest.raises(GeogFetchError, match="unsafe member"):
        fetch_geog(root=root, datasets=("alpha_ds",), source="hf",
                   progress=lambda *_: None,
                   urlopen_fn=_fake_transport(
                       {"alpha_ds.tar.bz2": payload}))
    assert not (tmp_path / "evil").exists()
    assert not (root / "alpha_ds").exists()


def test_symlink_member_is_refused(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:bz2") as tar:
        link = tarfile.TarInfo("alpha_ds/link")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/hosts"
        tar.addfile(link)
    payload = buffer.getvalue()
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES",
                        (_pin("alpha_ds", payload),))
    root = tmp_path / "WPS_GEOG"
    with pytest.raises(GeogFetchError, match="non-regular member"):
        fetch_geog(root=root, datasets=("alpha_ds",), source="hf",
                   progress=lambda *_: None,
                   urlopen_fn=_fake_transport(
                       {"alpha_ds.tar.bz2": payload}))


def test_invalid_existing_dataset_is_quarantined_not_overwritten(
        tmp_path, monkeypatch):
    payload = _build_archive(("alpha_ds",))
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES",
                        (_pin("alpha_ds", payload),))
    root = tmp_path / "WPS_GEOG"
    broken = root / "alpha_ds"
    broken.mkdir(parents=True)
    (broken / "leftover").write_text("partial extraction debris")

    staged = fetch_geog(root=root, datasets=("alpha_ds",), source="hf",
                        progress=lambda *_: None,
                        urlopen_fn=_fake_transport(
                            {"alpha_ds.tar.bz2": payload}))
    assert staged == 1
    ok, _ = validate_dataset_dir(root, "alpha_ds")
    assert ok
    quarantined = list(root.glob("alpha_ds.rejected-*"))
    assert len(quarantined) == 1
    assert (quarantined[0] / "leftover").is_file()


# ---------------------------------------------------------------------------
# Bundle mode
# ---------------------------------------------------------------------------

def test_bundle_extracts_only_needed_datasets_from_one_archive(
        tmp_path, monkeypatch):
    payload = _build_archive(("alpha_ds", "beta_ds", "extra_ds"),
                             mode="w:gz")
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES",
                        (_pin("alpha_ds", b"unused-a"),
                         _pin("beta_ds", b"unused-b")))
    monkeypatch.setattr(geog_assets, "MANDATORY_BUNDLE_BYTES", len(payload))
    monkeypatch.setattr(geog_assets, "MANDATORY_BUNDLE_SHA256",
                        hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(geog_assets, "MANDATORY_BUNDLE_DATASETS",
                        frozenset({"alpha_ds", "beta_ds"}))
    root = tmp_path / "WPS_GEOG"
    # beta_ds is already staged; only alpha_ds should be extracted
    staged_dir = root / "beta_ds"
    staged_dir.mkdir(parents=True)
    (staged_dir / "index").write_text(_INDEX_TEXT)

    transport = _fake_transport(
        {geog_assets.MANDATORY_BUNDLE_FILENAME: payload})
    staged = fetch_geog(root=root, datasets=("alpha_ds", "beta_ds"),
                        source="ncar", bundle=True,
                        progress=lambda *_: None, urlopen_fn=transport)
    assert staged == 1
    assert (root / "alpha_ds" / "index").is_file()
    assert not (root / "extra_ds").exists()
    (request,) = transport.calls
    assert request.full_url.endswith(geog_assets.MANDATORY_BUNDLE_FILENAME)


def test_bundle_wps_geog_prefix_layout_is_stripped(tmp_path, monkeypatch):
    """NCAR's mandatory bundle nests datasets under one WPS_GEOG/ dir
    (verified by walking the real archive); extraction must land them
    at root/<dataset> exactly like the per-dataset tarballs."""
    payload = _build_archive(("alpha_ds",), mode="w:gz",
                             prefix="WPS_GEOG/")
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES",
                        (_pin("alpha_ds", b"unused"),))
    monkeypatch.setattr(geog_assets, "MANDATORY_BUNDLE_BYTES", len(payload))
    monkeypatch.setattr(geog_assets, "MANDATORY_BUNDLE_SHA256",
                        hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(geog_assets, "MANDATORY_BUNDLE_DATASETS",
                        frozenset({"alpha_ds"}))
    root = tmp_path / "WPS_GEOG"
    staged = fetch_geog(root=root, datasets=("alpha_ds",), source="ncar",
                        bundle=True, progress=lambda *_: None,
                        urlopen_fn=_fake_transport(
                            {geog_assets.MANDATORY_BUNDLE_FILENAME:
                             payload}))
    assert staged == 1
    assert (root / "alpha_ds" / "index").is_file()
    assert not (root / "WPS_GEOG").exists()


def test_appledouble_sidecars_are_never_extracted(tmp_path, monkeypatch):
    """NCAR's bundle carries macOS `._*` sidecars beside every dataset
    (verified by walking the real archive); neither route may land
    them in the staged tree."""
    payload = _add_sidecars(("alpha_ds",), mode="w:gz",
                            prefix="WPS_GEOG/")
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES",
                        (_pin("alpha_ds", b"unused"),))
    monkeypatch.setattr(geog_assets, "MANDATORY_BUNDLE_BYTES", len(payload))
    monkeypatch.setattr(geog_assets, "MANDATORY_BUNDLE_SHA256",
                        hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(geog_assets, "MANDATORY_BUNDLE_DATASETS",
                        frozenset({"alpha_ds"}))
    root = tmp_path / "WPS_GEOG"
    fetch_geog(root=root, datasets=("alpha_ds",), source="ncar",
               bundle=True, progress=lambda *_: None,
               urlopen_fn=_fake_transport(
                   {geog_assets.MANDATORY_BUNDLE_FILENAME: payload}))
    assert (root / "alpha_ds" / "index").is_file()
    assert not (root / "alpha_ds" / "._index").exists()
    assert not list(root.rglob("._*"))


def test_bundle_refuses_datasets_it_does_not_contain(tmp_path, monkeypatch):
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES",
                        (_pin("alpha_ds", b"x"),))
    monkeypatch.setattr(geog_assets, "MANDATORY_BUNDLE_DATASETS",
                        frozenset())
    with pytest.raises(ValueError, match="does not contain"):
        fetch_geog(root=tmp_path / "WPS_GEOG", datasets=("alpha_ds",),
                   source="ncar", bundle=True, progress=lambda *_: None,
                   urlopen_fn=_refusing_transport())


# ---------------------------------------------------------------------------
# Remedy text: doctor and wizard point at this command
# ---------------------------------------------------------------------------

def test_doctor_remedy_is_the_fetch_geog_command():
    from gpuwm.doctor import GEOG_HINT
    assert "gpuwm fetch-geog" in GEOG_HINT
    # the honest size warning survives rewording
    assert "GB" in GEOG_HINT


def test_wizard_geog_help_names_the_downloader(capsys):
    from gpuwm.domain_wizard import _print_geog_help
    _print_geog_help()
    out = capsys.readouterr().out
    assert "gpuwm fetch-geog" in out
    for dataset in GEOG_DATASETS:
        assert dataset in out


# ---------------------------------------------------------------------------
# CLI wiring (through the real dispatcher)
# ---------------------------------------------------------------------------

def test_cli_list_mode_runs_offline_through_dispatch(tmp_path, capsys):
    from gpuwm.cli import main
    code = main(["fetch-geog", "--list", "--root", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "fetch-geog: total" in out
    for dataset in GEOG_DATASETS:
        assert dataset in out


def test_cli_refuses_unknown_dataset_as_a_usage_error(tmp_path, capsys):
    from gpuwm.cli import main
    code = main(["fetch-geog", "--list", "--root", str(tmp_path),
                 "--datasets", "not_a_dataset"])
    assert code == 2
    assert "unknown dataset" in capsys.readouterr().err


def test_cli_refuses_bundle_from_the_mirror(tmp_path, capsys):
    from gpuwm.cli import main
    code = main(["fetch-geog", "--list", "--root", str(tmp_path),
                 "--source", "hf", "--bundle"])
    assert code == 2
    assert "NCAR only" in capsys.readouterr().err


def test_fetch_geog_main_defaults_root_to_case_data_layout(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GPUWM_CASE_DATA_ROOT", str(tmp_path))
    from gpuwm.cli import main
    code = main(["fetch-geog", "--list"])
    out = capsys.readouterr().out
    assert code == 0
    assert str(tmp_path / "WPS_GEOG") in out


# ---------------------------------------------------------------------------
# Live network smoke (opt-in)
# ---------------------------------------------------------------------------

@pytest.mark.network
@pytest.mark.skipif(os.environ.get("GPUWM_NETWORK_TESTS") != "1",
                    reason="live network smoke; set GPUWM_NETWORK_TESTS=1")
def test_live_upstream_archives_match_the_pins():
    """HEAD every pinned archive on NCAR; sizes must equal the pins."""
    from urllib.request import Request, urlopen

    for archive in geog_assets.GEOG_ARCHIVES:
        url = archive_url(archive.filename, "ncar")
        request = Request(url, method="HEAD",
                          headers={"User-Agent": "gpuwm-tests/1"})
        with urlopen(request, timeout=60) as response:
            assert response.status == 200, url
            length = int(response.headers["Content-Length"])
        assert length == archive.archive_bytes, (
            f"{archive.filename}: NCAR now serves {length} B, pin is "
            f"{archive.archive_bytes} B -- upstream drift; refresh the "
            "pins and the mirror together")


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("GPUWM_NETWORK_TESTS") != "1",
                    reason="live network smoke; set GPUWM_NETWORK_TESTS=1")
def test_live_mirror_serves_the_pinned_bytes_sizes():
    """HEAD the mirror; skip (not fail) while the repo is unpublished."""
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    for archive in geog_assets.GEOG_ARCHIVES:
        url = archive_url(archive.filename, "hf")
        request = Request(url, method="HEAD",
                          headers={"User-Agent": "gpuwm-tests/1"})
        try:
            with urlopen(request, timeout=60) as response:
                assert response.status == 200, url
                length = int(response.headers["Content-Length"])
        except HTTPError as error:
            # HF answers 401 (not 404) for a repo that does not exist
            # or is private to anonymous callers
            if error.code in (401, 403, 404):
                pytest.skip(f"mirror not published yet "
                            f"(HTTP {error.code}): {url}")
            raise
        assert length == archive.archive_bytes, url


def _publish_tool():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "tools" / "publish_geog_mirror.py"
    spec = importlib.util.spec_from_file_location("publish_geog_mirror", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_mirror_publisher_knows_every_pinned_archive():
    """Every archive the pin table verifies has a provenance row, and the
    mirror README renders with all of them.

    The breakage this prevents: soilgrids was pinned on 2026-08-23 and the
    publisher, written for nine tarballs, had no row for it -- so the
    mirror lagged the pin table by a whole archive and every mesh-door
    fetch 404'd on the default source until 2026-09-01.  A pinned archive
    the publisher cannot render must fail here, not on the next mirror
    snapshot.
    """
    tool = _publish_tool()
    missing = [a.dataset for a in geog_assets.GEOG_ARCHIVES
               if a.dataset not in tool._PROVENANCE]
    assert missing == [], f"publisher has no provenance row for {missing}"
    stale = sorted(set(tool._PROVENANCE) - {a.dataset for a in geog_assets.GEOG_ARCHIVES})
    assert stale == [], f"publisher carries provenance for unpinned {stale}"
    readme = tool._readme_text()
    for archive in geog_assets.GEOG_ARCHIVES:
        assert f"`{archive.filename}` | {archive.archive_bytes:,} | `{archive.archive_sha256}`" in readme
        assert f"| `{archive.dataset}` |" in readme
