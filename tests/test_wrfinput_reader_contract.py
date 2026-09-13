"""The default decoder cannot silently retain its old numeric-only ABI."""

import pytest

from gpuwm import bridge_assets, bridges, netcdf_bridge


def test_default_reader_rejects_numeric_only_artifact_before_opening_input(tmp_path, monkeypatch):
    old = tmp_path / "rw_netcdf"
    old.write_bytes(b"gpuwm-rw-netcdf-inventory-v1\tgpuwm-rw-netcdf-dump-v1")
    monkeypatch.setattr(netcdf_bridge, "netcdf_candidates", lambda: (old,))
    monkeypatch.setattr(netcdf_bridge, "accept_resolved", lambda path: path)
    monkeypatch.setattr(netcdf_bridge, "netcdf_remedy", lambda: "gpuwm fetch-bridges")
    with pytest.raises(netcdf_bridge.NetcdfDecodeError, match="WRF Times"):
        netcdf_bridge.resolve_netcdf_bin()


def test_release_staging_rejects_the_same_numeric_only_artifact(tmp_path):
    old = tmp_path / "rw_netcdf"
    old.write_bytes(b"gpuwm-rw-netcdf-inventory-v1\tgpuwm-rw-netcdf-dump-v1")
    with pytest.raises(bridge_assets.BridgeAssetError, match="contract"):
        bridge_assets.verify_contract_marker("rw_netcdf", old)


def test_current_reader_is_accepted_by_resolution_and_staging(tmp_path, monkeypatch):
    current = tmp_path / "rw_netcdf"
    current.write_bytes(b"gpuwm-rw-netcdf-dump-v1\tdtype\t<f8\t|S1")
    monkeypatch.setattr(netcdf_bridge, "netcdf_candidates", lambda: (current,))
    monkeypatch.setattr(netcdf_bridge, "accept_resolved", lambda path: path)
    assert netcdf_bridge.resolve_netcdf_bin() == current
    bridge_assets.verify_contract_marker("rw_netcdf", current)


def test_default_resolution_recovers_from_an_older_higher_priority_build(tmp_path, monkeypatch):
    old, current = tmp_path / "old", tmp_path / "current"
    old.write_bytes(b"gpuwm-rw-netcdf-dump-v1")
    current.write_bytes(b"gpuwm-rw-netcdf-dump-v1\tdtype\t<f8\t|S1")
    monkeypatch.delenv(netcdf_bridge.NETCDF_ENV, raising=False)
    monkeypatch.setattr(netcdf_bridge, "netcdf_candidates", lambda: (old, current))
    monkeypatch.setattr(netcdf_bridge, "accept_resolved", lambda path: path)
    assert netcdf_bridge.resolve_netcdf_bin() == current


def test_an_explicit_old_reader_does_not_silently_select_different_bytes(tmp_path, monkeypatch):
    old, current = tmp_path / "old", tmp_path / "current"
    old.write_bytes(b"gpuwm-rw-netcdf-dump-v1")
    current.write_bytes(b"gpuwm-rw-netcdf-dump-v1\tdtype\t<f8\t|S1")
    monkeypatch.setenv(netcdf_bridge.NETCDF_ENV, str(old))
    monkeypatch.setattr(netcdf_bridge, "netcdf_candidates", lambda: (old, current))
    monkeypatch.setattr(netcdf_bridge, "accept_resolved", lambda path: path)
    monkeypatch.setattr(netcdf_bridge, "netcdf_remedy", lambda: "gpuwm fetch-bridges")
    with pytest.raises(netcdf_bridge.NetcdfDecodeError, match="WRF Times"):
        netcdf_bridge.resolve_netcdf_bin()


def test_doctor_reports_an_old_reader_without_aborting_its_inventory(tmp_path, monkeypatch):
    from gpuwm import doctor

    old = tmp_path / "rw_netcdf"
    old.write_bytes(b"gpuwm-rw-netcdf-dump-v1")
    monkeypatch.setattr(netcdf_bridge, "netcdf_candidates", lambda: (old,))
    monkeypatch.setattr(netcdf_bridge, "accept_resolved", lambda path: path)
    monkeypatch.setattr(netcdf_bridge, "netcdf_remedy", lambda: "gpuwm fetch-bridges")
    check = doctor._netcdf_decoder_check()
    assert check.status == "missing"
    assert "character" in check.detail
