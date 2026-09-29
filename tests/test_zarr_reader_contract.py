"""The fetch launches a Zarr reader that speaks this tree's contract, or refuses.

A reader staged in the user bridge directory by an older release still
launched, and refused every ARCO ERA5 request with
``level: source units "Hectopascal(hPa)" disagree with declared units
"hPa"`` although this tree's reader reads that spelling through its
units table.  Its record schema had not changed, so the old contract
marker let it through.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from gpuwm import bridges, zarr_bridge

REPO = Path(__file__).resolve().parents[1]
CURRENT = b"\x7fELF " + bridges.BRIDGE_ABI_MARKERS["rw_zarr"] + b" reader"
OLDER = b"\x7fELF arwen.regular-forcing-record.v1 reader"


def _ladder(tmp_path, monkeypatch):
    monkeypatch.delenv(zarr_bridge.BRIDGE_ENV, raising=False)
    monkeypatch.setattr(zarr_bridge, "__file__", str(tmp_path / "tree/gpuwm/zarr_bridge.py"))
    monkeypatch.setattr(zarr_bridge, "packaged_bridge_dir", lambda: tmp_path / "package")
    monkeypatch.setattr(zarr_bridge, "default_bridge_dir", lambda: tmp_path / "home")
    monkeypatch.setattr(zarr_bridge, "accept_resolved", lambda path: path)
    name = bridges.executable_name("rw_zarr")
    build = tmp_path / "tree" / zarr_bridge.CRATE_RELATIVE / "target/release" / name
    home = tmp_path / "home" / name
    return build, home


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_the_marker_is_compiled_from_the_readers_own_source():
    marker = bridges.BRIDGE_ABI_MARKERS["rw_zarr"].decode("utf-8")
    sources = "".join(path.read_text(encoding="utf-8")
                      for path in (REPO / "tools/zarr_bridge/src").glob("*.rs"))
    assert marker in sources
    assert zarr_bridge.ABI_MARKER == bridges.BRIDGE_ABI_MARKERS["rw_zarr"]


def test_a_reader_older_than_the_contract_is_passed_over(tmp_path, monkeypatch):
    build, home = _ladder(tmp_path, monkeypatch)
    _write(build, OLDER)
    _write(home, CURRENT)
    assert zarr_bridge.resolve_zarr_bin() == home.resolve()


def test_only_an_older_reader_is_refused_in_plain_words(tmp_path, monkeypatch):
    _build, home = _ladder(tmp_path, monkeypatch)
    _write(home, OLDER)
    with pytest.raises(bridges.DecoderContractError) as refused:
        zarr_bridge.resolve_zarr_bin()
    message = str(refused.value)
    assert str(home) in message and "older than this gpuwm" in message
    assert "Hectopascal(hPa)" in message


def test_a_named_override_is_held_to_the_same_contract(tmp_path, monkeypatch):
    _ladder(tmp_path, monkeypatch)
    named = _write(tmp_path / "elsewhere" / "rw_zarr.exe", OLDER)
    monkeypatch.setenv(zarr_bridge.BRIDGE_ENV, str(named))
    with pytest.raises(bridges.DecoderContractError):
        zarr_bridge.resolve_zarr_bin()
    named.write_bytes(CURRENT)
    assert zarr_bridge.resolve_zarr_bin() == named.resolve()
