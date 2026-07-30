"""``gpuwm fetch-tables``: staging, verification, and refusal contract.

The externalized Thompson asset (freezeH2O.dat) ships as a release
asset, not in the wheel or repository, so the fetch path IS the install
path for it: every byte must be verified against the thompson_contract
pins before an atomic install, mismatches must be refused and deleted,
and an existing wrong file must never be overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import pytest

from gpuwm import table_assets
from gpuwm.core.thompson_contract import (
    CLASSIC_TABLE_ASSETS,
    TableAsset,
    validate_table_assets,
)
from gpuwm.physics_compat import packaged_thompson_table_root

REPO_ROOT = Path(__file__).resolve().parents[1]


def _asset_for(payload: bytes, name: str = "synthetic.dat") -> TableAsset:
    return TableAsset(name, len(payload),
                      hashlib.sha256(payload).hexdigest())


def _args(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**{"from_dir": None, **kwargs})


# ---------------------------------------------------------------------------
# Unit: staging semantics with synthetic assets
# ---------------------------------------------------------------------------

def test_fetch_from_dir_verifies_then_installs_atomically(tmp_path):
    payload = b"\x00\x01" * 4096
    asset = _asset_for(payload)
    source = tmp_path / "src"
    root = tmp_path / "root"
    source.mkdir(), root.mkdir()
    (source / asset.filename).write_bytes(payload)

    final = table_assets.fetch_asset_from_dir(root, asset, source)
    assert final == root / asset.filename
    assert final.read_bytes() == payload
    # no partial droppings
    assert list(root.glob(".*fetch-partial")) == []


def test_fetch_from_dir_refuses_and_deletes_wrong_bytes(tmp_path):
    payload = b"\x00\x01" * 4096
    asset = _asset_for(payload)
    source = tmp_path / "src"
    root = tmp_path / "root"
    source.mkdir(), root.mkdir()
    (source / asset.filename).write_bytes(b"corrupted" + payload)

    with pytest.raises(table_assets.TableAssetError, match="refused"):
        table_assets.fetch_asset_from_dir(root, asset, source)
    assert not (root / asset.filename).exists()
    assert list(root.glob(".*fetch-partial")) == []


def test_fetch_from_url_verifies_sha256_not_just_size(tmp_path):
    payload = os.urandom(8192)
    asset = _asset_for(payload)
    served = tmp_path / "served"
    root = tmp_path / "root"
    served.mkdir(), root.mkdir()

    # same size, different bytes -> refusal
    evil = bytearray(payload)
    evil[0] ^= 0xFF
    (served / asset.filename).write_bytes(bytes(evil))
    url = (served / asset.filename).as_uri()
    with pytest.raises(table_assets.TableAssetError, match="SHA-256"):
        table_assets.fetch_asset_from_url(root, asset, url)
    assert not (root / asset.filename).exists()

    # correct bytes -> installed
    (served / asset.filename).write_bytes(payload)
    final = table_assets.fetch_asset_from_url(root, asset, url)
    assert final.read_bytes() == payload


def test_classify_assets_partitions_valid_invalid_absent(tmp_path):
    good = b"good" * 1000
    asset_ok = _asset_for(good, "ok.dat")
    asset_bad = _asset_for(b"expected", "bad.dat")
    asset_gone = _asset_for(b"absent", "gone.dat")
    (tmp_path / "ok.dat").write_bytes(good)
    (tmp_path / "bad.dat").write_bytes(b"not-expected")

    valid, invalid, absent = table_assets.classify_assets(
        tmp_path, (asset_ok, asset_bad, asset_gone))
    assert [a.filename for a in valid] == ["ok.dat"]
    assert len(invalid) == 1 and "bad.dat" in invalid[0]
    assert [a.filename for a in absent] == ["gone.dat"]


# ---------------------------------------------------------------------------
# Integration: the real pins, the real packaged root
# ---------------------------------------------------------------------------

def _staged_root_without_externalized(tmp_path, monkeypatch) -> Path:
    """A table root holding the packaged (wheel-shipped) assets only.

    This is exactly what a wheel install looks like before
    ``gpuwm fetch-tables`` runs.  Hardlinks keep the copies free.
    """

    packaged = packaged_thompson_table_root()
    if not packaged.is_dir():
        pytest.skip("packaged table root absent (wheel-only checkout)")
    root = tmp_path / "tables"
    root.mkdir()
    for asset in CLASSIC_TABLE_ASSETS:
        if asset.filename in table_assets.EXTERNALIZED_TABLE_FILENAMES:
            continue
        source = packaged / asset.filename
        if not source.is_file():
            pytest.skip(f"packaged asset absent: {asset.filename}")
        try:
            os.link(source, root / asset.filename)
        except OSError:
            import shutil
            shutil.copyfile(source, root / asset.filename)
    monkeypatch.setenv("GPUWM_THOMPSON_TABLE_ROOT", str(root))
    return root


def test_doctor_names_the_fetch_remedy_for_the_externalized_gap(
        tmp_path, monkeypatch):
    from gpuwm import doctor

    _staged_root_without_externalized(tmp_path, monkeypatch)
    check = doctor._thompson_tables_check()
    assert check.status == "missing"
    assert "freezeH2O.dat" in check.detail
    assert "externalized" in check.detail
    assert check.remedy is not None
    assert "gpuwm fetch-tables" in check.remedy
    # the reinstall hint is the wrong remedy for a fetchable gap
    assert "reinstall" not in check.remedy


def test_fetch_tables_main_stages_offline_and_is_idempotent(
        tmp_path, monkeypatch, capsys):
    packaged = packaged_thompson_table_root()
    externalized = [packaged / name
                    for name in table_assets.EXTERNALIZED_TABLE_FILENAMES]
    if not all(path.is_file() for path in externalized):
        pytest.skip("externalized asset bytes not present in this checkout")

    root = _staged_root_without_externalized(tmp_path, monkeypatch)
    assert table_assets.fetch_tables_main(
        _args(from_dir=str(packaged))) == 0
    validate_table_assets(root)  # the exact load-time validation
    out = capsys.readouterr().out
    assert "verified and installed" in out

    # second run: nothing to fetch, still 0, no network, no rewrite
    before = (root / "freezeH2O.dat").stat().st_mtime_ns
    assert table_assets.fetch_tables_main(_args()) == 0
    assert (root / "freezeH2O.dat").stat().st_mtime_ns == before
    assert "nothing to fetch" in capsys.readouterr().out


def test_fetch_tables_main_refuses_existing_wrong_bytes(
        tmp_path, monkeypatch, capsys):
    root = _staged_root_without_externalized(tmp_path, monkeypatch)
    (root / "freezeH2O.dat").write_bytes(b"tampered")
    assert table_assets.fetch_tables_main(_args()) == 2
    out = capsys.readouterr().out
    assert "REFUSED" in out and "never overwritten" in out
    # refusal must not delete or replace the operator's file
    assert (root / "freezeH2O.dat").read_bytes() == b"tampered"


def test_fetch_tables_main_treats_packaged_gap_as_reinstall(
        tmp_path, monkeypatch, capsys):
    root = _staged_root_without_externalized(tmp_path, monkeypatch)
    (root / "thompson_aux_tables.dat").unlink()
    assert table_assets.fetch_tables_main(_args()) == 2
    assert "reinstall" in capsys.readouterr().out


def test_cli_dispatches_fetch_tables_for_real(monkeypatch, capsys):
    """Through gpuwm.cli.main, not just --help: the dispatch table must
    route fetch-tables to its handler (a --help-only probe once passed
    while the real dispatch raised AttributeError)."""

    from gpuwm.cli import main

    packaged = packaged_thompson_table_root()
    valid, invalid, absent = table_assets.classify_assets(packaged)
    if invalid or absent:
        pytest.skip("packaged table root incomplete in this checkout")
    monkeypatch.setenv("GPUWM_THOMPSON_TABLE_ROOT", str(packaged))
    assert main(["fetch-tables"]) == 0
    assert "nothing to fetch" in capsys.readouterr().out
