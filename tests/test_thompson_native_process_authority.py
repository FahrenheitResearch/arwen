"""The source MP28 profile binds its actual generation's process tables."""
from pathlib import Path

import pytest

import tools.hrrr_single_domain_benchmark as native

from gpuwm.core.thompson_contract import FORK_TABLE_ASSETS, FORK_TABLE_SET_ID
from gpuwm.physics_compat import thompson_fork_table_root, thompson_table_root


PROFILE = "thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1"


def test_staged_mp28_native_authority_binds_fork_process_tables_and_classic_activation():
    selected = native._native_hrrr_runtime_switches(PROFILE)
    assert selected["mp_physics"] == 28
    assert selected["thompson_version"] == selected["thompson_fork_snow_fall"] == "wrf_39_noaa"
    try:
        authority = native._microphysics_table_authority(PROFILE)
    except FileNotFoundError as missing:
        pytest.skip("real Thompson table-asset integration unavailable: " + str(missing))
    assert authority["table_set"] == FORK_TABLE_SET_ID
    assert Path(authority["table_root"]).resolve() == Path(thompson_fork_table_root()).resolve()
    assert authority["assets"] == [
        {"filename": asset.filename, "bytes": asset.bytes, "sha256": asset.sha256}
        for asset in FORK_TABLE_ASSETS
    ]
    activation = authority["classic_aerosol_authority"]
    assert Path(activation["table_root"]).resolve() == Path(thompson_table_root()).resolve()
    assert activation["mp_physics"] == 28
    assert "CCN_ACTIVATE.BIN" in {asset["filename"] for asset in activation["assets"]}


def test_host_authority_wiring_selects_exact_fork_and_classic_aerosol_pins(tmp_path, monkeypatch):
    """Observe the table contract calls without claiming real asset validation."""
    from gpuwm import physics_compat, table_assets
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
    authority = native._microphysics_table_authority(PROFILE)
    # The classic activation root is staged and byte-validated here.  The
    # fork process set is only DECLARED at this preview: since 75daa3892
    # (lane/286-staging-repair) it is acquired and byte-validated by
    # gpuwm.thompson_fork_assets.ensure_thompson_fork_tables before first
    # use, so a configuration preview neither downloads nor compiles it.
    assert calls == [("require", required_classic),
                     ("validate", classic, required_classic)]
    assert authority["asset_validation"] == "deferred_to_runtime_before_first_use"
    assert authority["table_set"] == FORK_TABLE_SET_ID
    assert authority["table_root"] == str(fork)
    assert authority["assets"] == [
        {"filename": asset.filename, "bytes": asset.bytes, "sha256": asset.sha256}
        for asset in FORK_TABLE_ASSETS]
    activation = authority["classic_aerosol_authority"]
    assert activation["table_root"] == str(classic)
    assert activation["table_set"] == AEROSOL_TABLE_SET_ID
    assert activation["assets"] == [
        {"filename": asset.filename, "bytes": asset.bytes, "sha256": asset.sha256}
        for asset in required_classic]
