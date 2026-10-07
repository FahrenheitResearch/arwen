"""The source MP28 profile binds its actual generation's process tables."""
from pathlib import Path

import pytest

import tools.hrrr_single_domain_benchmark as native

from gpuwm.core.thompson_contract import FORK_TABLE_ASSETS, FORK_TABLE_SET_ID
from gpuwm.physics_compat import thompson_fork_table_root, thompson_table_root


PROFILE = "thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1"


def test_staged_mp28_native_authority_binds_fork_process_tables_and_classic_activation():
    """Preview declares pins; first use verifies real fork and activation assets."""
    selected = native._native_hrrr_runtime_switches(PROFILE)
    assert selected["mp_physics"] == 28
    assert selected["thompson_version"] == selected["thompson_fork_snow_fall"] == "wrf_39_noaa"
    try:
        authority = native._microphysics_table_authority(PROFILE)
    except FileNotFoundError as missing:
        pytest.skip("real Thompson table-asset integration unavailable: " + str(missing))
    assert authority["table_set"] == FORK_TABLE_SET_ID
    assert authority["asset_validation"] == "deferred_to_runtime_before_first_use"
    assert Path(authority["table_root"]).resolve() == Path(thompson_fork_table_root()).resolve()
    assert authority["assets"] == [
        {"filename": asset.filename, "bytes": asset.bytes, "sha256": asset.sha256}
        for asset in FORK_TABLE_ASSETS
    ]
    activation = authority["classic_aerosol_authority"]
    assert Path(activation["table_root"]).resolve() == Path(thompson_table_root()).resolve()
    assert activation["mp_physics"] == 28
    assert "CCN_ACTIVATE.BIN" in {asset["filename"] for asset in activation["assets"]}
    from gpuwm.core import microphysics_aerosol
    from gpuwm.core.thompson_contract import validate_table_assets

    runtime_root = Path(microphysics_aerosol._wrf39_table_root()).resolve()
    assert runtime_root == Path(authority["table_root"]).resolve()
    validated = validate_table_assets(runtime_root, FORK_TABLE_ASSETS)
    assert [{"filename": asset.filename, "bytes": asset.bytes, "sha256": asset.sha256}
            for asset in validated] == authority["assets"]


def test_host_authority_wiring_selects_exact_fork_and_classic_aerosol_pins(tmp_path, monkeypatch):
    """Observe the table contract calls without claiming real asset validation."""
    from gpuwm import physics_compat, table_assets, thompson_fork_assets
    from gpuwm.core import thompson_contract
    from gpuwm.core.thompson_aerosol_contract import AEROSOL_TABLE_ASSETS, AEROSOL_TABLE_SET_ID

    classic = tmp_path / "classic"
    fork = tmp_path / "fork"
    required_classic = (*thompson_contract.CLASSIC_TABLE_ASSETS, *AEROSOL_TABLE_ASSETS)
    calls = []
    def require(*, assets):
        calls.append(("require", assets))
        return classic
    def validate(root, assets=thompson_contract.CLASSIC_TABLE_ASSETS):
        calls.append(("validate", Path(root), assets))
        return assets
    monkeypatch.setattr(table_assets, "require_thompson_tables", require)
    monkeypatch.setattr(thompson_contract, "validate_table_assets", validate)
    monkeypatch.setattr(physics_compat, "thompson_table_root", lambda: str(classic))
    monkeypatch.setattr(physics_compat, "thompson_fork_table_root", lambda: str(fork))
    monkeypatch.setattr(thompson_fork_assets, "ensure_thompson_fork_tables",
                        lambda *a, **k: pytest.fail("configuration preview acquired fork tables"))
    authority = native._microphysics_table_authority(PROFILE)
    assert calls == [("require", required_classic),
                     ("validate", classic, required_classic)]
    assert authority["table_set"] == FORK_TABLE_SET_ID
    assert authority["table_root"] == str(fork)
    assert authority["asset_validation"] == "deferred_to_runtime_before_first_use"
    assert authority["assets"] == [
        {"filename": asset.filename, "bytes": asset.bytes, "sha256": asset.sha256}
        for asset in FORK_TABLE_ASSETS]
    activation = authority["classic_aerosol_authority"]
    assert activation["table_root"] == str(classic)
    assert activation["table_set"] == AEROSOL_TABLE_SET_ID
    assert activation["assets"] == [
        {"filename": asset.filename, "bytes": asset.bytes, "sha256": asset.sha256}
        for asset in required_classic]


def test_fork_first_use_requires_acquisition_without_a_classic_fallback(tmp_path, monkeypatch):
    """Observe the real runtime resolver, without claiming coefficient validation."""
    from gpuwm import thompson_fork_assets
    from gpuwm.core import microphysics_aerosol

    selected = tmp_path / "canonical-fork"
    calls = []

    def ensure():
        calls.append(FORK_TABLE_ASSETS)
        return selected

    monkeypatch.setattr(thompson_fork_assets, "ensure_thompson_fork_tables", ensure)
    assert microphysics_aerosol._wrf39_table_root() == str(selected)
    assert calls == [FORK_TABLE_ASSETS]
    assert len(FORK_TABLE_ASSETS) == 4

    def absent():
        raise FileNotFoundError("canonical fork process tables unavailable before first use")

    monkeypatch.setattr(thompson_fork_assets, "ensure_thompson_fork_tables", absent)
    with pytest.raises(FileNotFoundError, match="canonical fork process tables unavailable"):
        microphysics_aerosol._wrf39_table_root()
