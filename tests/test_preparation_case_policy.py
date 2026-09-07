"""Optional preparation companions keep their captured authority and owner."""
from pathlib import Path
import hashlib
import pytest

from gpuwm.case_data import (optional_case_data_from_config,
    optional_case_data_from_tables, preparation_case_policy, trace_gas_overrides_from_config)


def _table(**changes):
    return {"forcing": "not-fetched.grib", "vtable": "Vtable",
        "wps_namelist": "namelist.wps", "geog_root": "geog",
        "sfcp_to_sfcp": True, "output_title": "declared", **changes}


def test_absent_companions_keep_default_operands():
    assert optional_case_data_from_tables({}, source="empty", base_dir=Path(".")) is None
    assert preparation_case_policy(None) == {"sfcp_to_sfcp": True,
        "water_temperature_policy": "era5_class_coherent", "water_temperature_overlay": None}


def test_optional_companions_share_captured_input_authority(tmp_path, monkeypatch):
    from gpuwm.branch import emit_experiment_toml
    from gpuwm.config_authority import authority_environment
    source = tmp_path / "case.toml"
    payload = emit_experiment_toml({"case_data": _table(co2_vmr=.000701,
        water_temperature_overlay="water.nc", water_temperature_policy="wrf_compat")}).encode()
    source.write_bytes(payload)
    capture = tmp_path / "capture.toml"
    capture.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    for key, value in authority_environment(source=source, payload_path=capture, sha256=digest).items():
        monkeypatch.setenv(key, value)
    source.write_text("changed original", encoding="utf-8")
    data = optional_case_data_from_config(source, expected_sha256=digest)
    assert preparation_case_policy(data) == {"sfcp_to_sfcp": True,
        "water_temperature_policy": "wrf_compat", "water_temperature_overlay": str(tmp_path / "water.nc")}
    assert trace_gas_overrides_from_config(source, expected_sha256=digest) == {"co2": .000701}
    capture.write_bytes(payload + b"# changed")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        optional_case_data_from_config(source)


@pytest.mark.parametrize("bad", [{"sfcp_to_sfcp": "false"},
    {"water_temperature_policy": "unknown"}, {"unused_companion": 1}])
def test_optional_companions_keep_the_full_schema_validation(tmp_path, bad):
    with pytest.raises(ValueError):
        optional_case_data_from_tables({"case_data": _table(**bad)},
                                      source="bad.toml", base_dir=tmp_path)


def test_actual_false_surface_pressure_policy_is_retained(tmp_path):
    data = optional_case_data_from_tables({"case_data": _table(sfcp_to_sfcp=False)},
                                          source="case.toml", base_dir=tmp_path)
    assert preparation_case_policy(data)["sfcp_to_sfcp"] is False


def test_native_namelist_pressure_control_is_not_lost_without_companions(tmp_path):
    assert preparation_case_policy(None, sfcp_to_sfcp=False)["sfcp_to_sfcp"] is False
    with pytest.raises(ValueError, match="boolean"):
        preparation_case_policy(None, sfcp_to_sfcp=0)
    data = optional_case_data_from_tables({"case_data": _table(sfcp_to_sfcp=False)},
                                          source="case.toml", base_dir=tmp_path)
    assert preparation_case_policy(data, sfcp_to_sfcp=False)["sfcp_to_sfcp"] is False
    with pytest.raises(ValueError, match="differs from the explicit namelist"):
        preparation_case_policy(data, sfcp_to_sfcp=True)
