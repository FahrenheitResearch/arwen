"""The default decoder cannot silently retain its old numeric-only ABI."""

import pytest

from gpuwm import bridge_assets, bridges, netcdf_bridge


@pytest.mark.parametrize("has_qv", [False, True])
def test_wrfinput_metadata_detects_vapor_from_headers_without_decoding(
        tmp_path, monkeypatch, has_qv):
    from gpuwm.ingest import wrfinput as wi

    class HeaderVariable:
        def __getitem__(self, key):
            raise AssertionError("metadata must not decode a field")

    class HeaderDataset:
        dimensions = {name: range(size) for name, size in (
            ("west_east", 6), ("west_east_stag", 7),
            ("south_north", 4), ("south_north_stag", 5),
            ("bottom_top", 3), ("bottom_top_stag", 4),
            ("soil_layers_stag", 4))}
        variables = {name: HeaderVariable() for name in
                     (("U", "QVAPOR") if has_qv else ("U",))}
        attributes = {"GRID_ID": 1, "MP_PHYSICS": 0,
                      "SF_SURFACE_PHYSICS": 0,
                      "START_DATE": "2024-01-01_00:00:00"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def ncattrs(self):
            return tuple(self.attributes)

        def getncattr(self, name):
            return self.attributes[name]

    monkeypatch.setattr(netcdf_bridge, "open_dataset",
                        lambda path: HeaderDataset())
    metadata = wi.read_wrfinput_metadata(tmp_path / "wrfinput")
    assert metadata.has_qv is has_qv
    assert metadata.mp_physics == 0
    assert (metadata.nx, metadata.ny, metadata.nz) == (6, 4, 3)


def test_wrfinput_metadata_appends_optional_vapor_presence():
    from pathlib import Path
    from gpuwm.ingest.wrfinput import WrfinputMetadata

    metadata = WrfinputMetadata(Path("wrfinput"), 1, {}, 0, 0, "", {})
    assert metadata.has_qv is False


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
    current.write_bytes(b"gpuwm-rw-netcdf-dump-v1\tdtype\t<f8\t|S1\twater_layer_conversion\tsource_soil_recovery")
    monkeypatch.setattr(netcdf_bridge, "netcdf_candidates", lambda: (current,))
    monkeypatch.setattr(netcdf_bridge, "accept_resolved", lambda path: path)
    assert netcdf_bridge.resolve_netcdf_bin() == current
    bridge_assets.verify_contract_marker("rw_netcdf", current)


def test_default_resolution_recovers_from_an_older_higher_priority_build(tmp_path, monkeypatch):
    old, current = tmp_path / "old", tmp_path / "current"
    old.write_bytes(b"gpuwm-rw-netcdf-dump-v1")
    current.write_bytes(b"gpuwm-rw-netcdf-dump-v1\tdtype\t<f8\t|S1\twater_layer_conversion\tsource_soil_recovery")
    monkeypatch.delenv(netcdf_bridge.NETCDF_ENV, raising=False)
    monkeypatch.setattr(netcdf_bridge, "netcdf_candidates", lambda: (old, current))
    monkeypatch.setattr(netcdf_bridge, "accept_resolved", lambda path: path)
    assert netcdf_bridge.resolve_netcdf_bin() == current


def test_an_explicit_old_reader_does_not_silently_select_different_bytes(tmp_path, monkeypatch):
    old, current = tmp_path / "old", tmp_path / "current"
    old.write_bytes(b"gpuwm-rw-netcdf-dump-v1")
    current.write_bytes(b"gpuwm-rw-netcdf-dump-v1\tdtype\t<f8\t|S1\twater_layer_conversion\tsource_soil_recovery")
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
