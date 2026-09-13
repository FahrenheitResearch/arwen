"""C-048: local capability evidence, never a credential or network check."""
from pathlib import Path
import pytest
from gpuwm import doctor, fetch, bridges, zarr_bridge


@pytest.mark.parametrize("cds,credentials,arco", [
    (False,False,False), (True,False,False), (True,True,False),
    (False,False,True), (True,True,True)])
def test_era5_transport_reports_actual_local_modes(tmp_path, monkeypatch, cds, credentials, arco):
    rc = tmp_path / "credentials"
    if credentials:
        rc.write_text("DO-NOT-READ-THIS-SECRET")
    monkeypatch.setattr(fetch, "cds_credentials_path", lambda: rc)
    monkeypatch.delenv("CDSAPI_KEY", raising=False)
    monkeypatch.delenv("CDSAPI_URL", raising=False)
    monkeypatch.setattr(doctor, "_import_probe", lambda *a, **kw: (cds, "local import fixture"))
    def resolve():
        assert bridges._INSPECTION_ONLY
        if not arco:
            raise FileNotFoundError("fixture absent bridge")
        return Path("/fixture/rw_zarr")
    monkeypatch.setattr(zarr_bridge, "resolve_zarr_bin", resolve)
    monkeypatch.setattr(bridges, "bridge_abi_matches", lambda *a: (True, "local ABI fixture"))
    check = doctor._era5_fetch_path_check()
    assert check.name == "era5 route fetch transport"
    assert "--retrieve" in check.detail and "--era5-provider arco" in check.detail
    assert "request document" in check.detail
    assert "not tested" in check.detail
    assert "DO-NOT-READ-THIS-SECRET" not in str(check)
    assert check.status == ("verified" if cds and credentials else "info")
    if arco:
        assert "keyless ARCO reader is locally available" in check.detail


def test_era5_transport_does_not_count_a_key_without_url(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "cds_credentials_path", lambda: tmp_path / "absent")
    monkeypatch.setattr(doctor, "_import_probe", lambda *a, **kw: (True, "local import fixture"))
    monkeypatch.setenv("CDSAPI_KEY", "PRIVATE-KEY")
    monkeypatch.delenv("CDSAPI_URL", raising=False)
    monkeypatch.setattr(zarr_bridge, "resolve_zarr_bin", lambda: (_ for _ in ()).throw(FileNotFoundError()))
    assert doctor._era5_fetch_path_check().status == "info"
    monkeypatch.setenv("CDSAPI_URL", "https://example.invalid")
    check = doctor._era5_fetch_path_check()
    assert check.status == "verified"
    assert "PRIVATE-KEY" not in str(check)


def test_real_era5_route_uses_transport_probe(monkeypatch):
    sentinel = doctor.Check("era5 route fetch transport", "info", "fixture")
    monkeypatch.setattr(doctor, "_decoder_route_check", lambda *a: doctor.Check("decoder", "info", "fixture"))
    monkeypatch.setattr(doctor, "_era5_fetch_path_check", lambda: sentinel)
    assert doctor._source_route_checks("era5")[-1] is sentinel
