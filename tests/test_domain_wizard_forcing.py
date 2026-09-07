"""Native hourly GRIB1 metadata controls the wizard's actual input schedule."""
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import tomllib

import numpy as np
import pytest

from gpuwm import domain_wizard as wizard
from gpuwm.core import preflight
from gpuwm.ingest.grib import (build_rust_bridge, decode_era5_grib,
                               inspect_era5_forcing_times)


START = datetime(2026, 7, 29, 18)


def grib1_message(when, *, parameter=130, level_type=100):
    # Independent WMO GRIB1 test encoding: real PDS, regular 2x2 GDS,
    # simple-packed field [270,271,272,273], IBM reference270. Native parser
    # and normal value decoding both read these bytes in the positive test.
    pds = bytearray(28)
    pds[:3] = (28).to_bytes(3, "big")
    pds[3:10] = bytes((128, 98, 0, 0, 128, parameter, level_type))
    pds[10:12] = (500 if level_type == 100 else 0).to_bytes(2, "big")
    pds[12:18] = bytes((when.year % 100 or 100, when.month, when.day,
                         when.hour, when.minute, 1))
    pds[24] = (when.year - 1) // 100 + 1
    gds = bytearray(32)
    gds[:3] = (32).to_bytes(3, "big")
    gds[4] = 255
    gds[6:10] = bytes((0, 2, 0, 2))
    gds[10:13] = (45000).to_bytes(3, "big")
    gds[13:16] = (0x800000 | 90000).to_bytes(3, "big")
    gds[16] = 128
    gds[17:20] = (40000).to_bytes(3, "big")
    gds[20:23] = (0x800000 | 85000).to_bytes(3, "big")
    gds[23:27] = (5000).to_bytes(2, "big") * 2
    bds = bytearray(15)
    bds[:3] = (15).to_bytes(3, "big")
    bds[6:10] = bytes.fromhex("4310e000")
    bds[10:] = bytes((8, 0, 1, 2, 3))
    body = bytes(pds + gds + bds) + b"7777"
    return b"GRIB" + (8 + len(body)).to_bytes(3, "big") + b"\x01" + body


@pytest.fixture(scope="module")
def bridge():
    return build_rust_bridge(release=False)


def write_forcing(tmp_path, hours=range(9)):
    path = tmp_path / "hourly.grib"
    path.write_bytes(b"".join(grib1_message(START + timedelta(hours=hour))
                              for hour in hours))
    return path


def test_native_inventory_matches_independent_times_and_real_value_decode(tmp_path, bridge):
    forcing = write_forcing(tmp_path, range(3))
    expected = tuple(START + timedelta(hours=hour) for hour in range(3))
    assert inspect_era5_forcing_times([forcing], wizard._PACKAGED_VTABLE,
                                     bridge=bridge) == expected
    decoded = decode_era5_grib(forcing, wizard._PACKAGED_VTABLE, bridge=bridge)
    assert tuple(snapshot.valid_time for snapshot in decoded) == expected
    for snapshot in decoded:
        np.testing.assert_array_equal(snapshot.fields["T"], [[[270, 271], [272, 273]]])
    # An auxiliary surface record at an unrelated date is not a forcing time.
    with forcing.open("ab") as stream:
        stream.write(grib1_message(START - timedelta(days=10), parameter=129, level_type=1))
    assert inspect_era5_forcing_times([forcing], wizard._PACKAGED_VTABLE,
                                     bridge=bridge) == expected


def test_metadata_inventory_does_not_unpack_field_payloads(tmp_path, bridge):
    message = bytearray(grib1_message(START))
    message[8 + 28 + 32 + 10] = 32  # Header-valid, too few packed bits for4 values.
    forcing = tmp_path / "header-only.grib"
    forcing.write_bytes(message)
    assert inspect_era5_forcing_times([forcing], wizard._PACKAGED_VTABLE,
                                     bridge=bridge) == (START,)
    with pytest.raises(RuntimeError, match="Rust GRIB1 bridge failed"):
        decode_era5_grib(forcing, wizard._PACKAGED_VTABLE, bridge=bridge)
    assert not list(tmp_path.glob("**/values.f64"))


@pytest.mark.parametrize("damage", ["truncated", "bad-section"])
def test_inventory_refuses_a_bad_message_after_a_good_prefix(tmp_path, bridge, damage):
    second = bytearray(grib1_message(START + timedelta(hours=1)))
    if damage == "truncated":
        second = second[:-1]
    else:
        second[8 + 28:8 + 28 + 3] = (31).to_bytes(3, "big")
    forcing = tmp_path / "bad.grib"
    forcing.write_bytes(grib1_message(START) + second)
    native = subprocess.run([str(bridge), "--inventory", str(forcing)], capture_output=True)
    assert native.returncode != 0
    with pytest.raises((ValueError, RuntimeError), match="truncated|metadata inventory failed"):
        inspect_era5_forcing_times([forcing], wizard._PACKAGED_VTABLE, bridge=bridge)


def emit(tmp_path, forcing, *, hours=2, extra=()):
    from gpuwm.cli import main
    out = tmp_path / "area.toml"
    rc = main(["domain", "--point", "35.3,-97.5", "--source", "era5",
               "--cycle", "2026-07-29T18", "--hours", str(hours), "--card", "12gb",
               "--root-dx", "3", "--nz", "76", "--forcing", str(forcing),
               "--out", str(out), *extra])
    return rc, out


def test_supplied_hourly_window_is_priced_before_fit_and_retained_in_config(
        tmp_path, bridge, monkeypatch, capsys):
    monkeypatch.setenv("GPUWM_GRIB1_BRIDGE", str(bridge))
    forcing = write_forcing(tmp_path)
    calls = []
    actual = wizard.estimate_phases
    def record(exp, **kwargs):
        calls.append(kwargs.copy())
        return actual(exp, **kwargs)
    monkeypatch.setattr(wizard, "estimate_phases", record)
    rc, out = emit(tmp_path, forcing)
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    raw = tomllib.loads(out.read_text())
    assert raw["experiment"]["run_seconds"] == 7200
    assert raw["case_data"]["forcing_interval_s"] == 3600
    assert raw["fetch"]["cadence"] == 1
    assert "interval_seconds = 3600," in out.with_suffix(".namelist.wps").read_text()
    assert calls and all(call["forcing_interval_seconds"] == 3600 for call in calls)
    assert all(call["ingest_forcing_interval_seconds"] == 3600 for call in calls)
    assert all(call["forcing_intervals"] == 8 for call in calls)
    exp = wizard.experiment_from_text(out.read_text(), source=str(out))
    cadence, count = preflight.config_forcing_schedule(out, exp)
    assert (cadence, count) == (3600, 8)
    full = preflight.estimate_phases(exp, source="era5", forcing_interval_seconds=cadence,
                                    ingest_forcing_interval_seconds=cadence,
                                    forcing_intervals=count)
    clipped = preflight.estimate_phases(exp, source="era5", forcing_interval_seconds=cadence,
                                       ingest_forcing_interval_seconds=cadence)
    # Six extra retained intervals each own one packed FP32 root boundary table.
    extra = 6 * 4 * preflight.lbc_interval_values(exp.root.run)
    assert (full.forecast.domains[0].category_bytes("lbc")
            - clipped.forecast.domains[0].category_bytes("lbc")) == extra
    assert full.ingest.n_forcing_times == 9
    assert clipped.ingest.n_forcing_times == 3
    assert full.peak_envelope_bytes > clipped.peak_envelope_bytes


@pytest.mark.parametrize("hours, window, message", [
    ([0, 1, 3], 2, "gap or nonuniform cadence"),
    ([1, 2, 3], 2, "missing the requested start"),
    ([0, 1], 2, "before the requested end"),
    ([0], 2, "at least two"),
])
def test_supplied_time_defects_refuse_before_writing(
        tmp_path, bridge, monkeypatch, capsys, hours, window, message):
    monkeypatch.setenv("GPUWM_GRIB1_BRIDGE", str(bridge))
    forcing = write_forcing(tmp_path, hours)
    rc, out = emit(tmp_path, forcing, hours=window)
    assert rc == 2
    assert message in capsys.readouterr().err
    assert not out.exists()


def test_catalog_times_outrank_fetch_hint_and_do_not_require_redecode(tmp_path, monkeypatch):
    from gpuwm.ingest import grib
    def forbidden(*args, **kwargs):
        pytest.fail("an available input catalog must not be decoded again")
    monkeypatch.setattr(grib, "inspect_era5_forcing_times", forbidden)
    text = wizard.render_config(
        name="catalog", start_time=START, hours=2,
        projection=wizard._projection_entries(35.3, -97.5, "auto"),
        dims=[(100, 80)], ratios=(), fetch_hints=wizard._candidate_fetch_hints("era5"),
        case_data={"forcing": ["absent.grib"], "vtable": str(wizard._PACKAGED_VTABLE),
                   "wps_namelist": "absent.wps", "geog_root": "absent-geog",
                   "forcing_interval_s": 3600, "sfcp_to_sfcp": True,
                   "output_domain": 1, "output_title": "test"})
    path = tmp_path / "catalog.toml"
    path.write_text(text)
    exp = wizard.experiment_from_text(text, source=str(path))
    catalog = SimpleNamespace(valid_times=tuple(START + timedelta(hours=h) for h in range(9)))
    assert preflight.config_forcing_schedule(
        path, exp, input_catalog=catalog, fetch_cadence_hours=6) == (3600, 8)


@pytest.mark.parametrize("count", [0, -1, True, 2.5])
def test_retained_interval_override_rejects_invalid_counts(count):
    with pytest.raises(ValueError, match="positive integer"):
        preflight.lbc_intervals(7200, 3600, retained_intervals=count)


def test_fetch_plan_cadence_override_applies_when_no_case_inputs_exist(tmp_path):
    text = wizard.render_config(
        name="planned", start_time=START, hours=6,
        projection=wizard._projection_entries(35.3, -97.5, "auto"),
        dims=[(100, 80)], ratios=(), fetch_hints=wizard._candidate_fetch_hints("gfs"),
        case_data=None)
    path = tmp_path / "planned.toml"
    path.write_text(text)
    exp = wizard.experiment_from_text(text, source=str(path))
    assert preflight.config_forcing_schedule(path, exp, fetch_cadence_hours=1) == (3600, None)


def test_check_and_go_price_actual_window_despite_six_hour_fetch_hint(
        tmp_path, bridge, monkeypatch, capsys):
    import argparse
    import dataclasses
    from gpuwm import doctor, go_cli

    monkeypatch.setenv("GPUWM_GRIB1_BRIDGE", str(bridge))
    forcing = write_forcing(tmp_path)
    geog = tmp_path / "geog"
    geog.mkdir()
    wps = tmp_path / "case.wps"
    wps.write_text("&geogrid\n geog_data_res = 'default',\n/\n")
    text = wizard.render_config(
        name="admission", start_time=START, hours=2,
        projection=wizard._projection_entries(35.3, -97.5, "auto"),
        dims=[(100, 80)], ratios=(),
        fetch_hints={**wizard._candidate_fetch_hints("era5"), "cadence": 6},
        case_data={"forcing": [forcing.as_posix()],
                   "vtable": wizard._PACKAGED_VTABLE.as_posix(),
                   "wps_namelist": wps.as_posix(), "geog_root": geog.as_posix(),
                   "forcing_interval_s": 3600, "sfcp_to_sfcp": True,
                   "output_domain": 1, "output_title": "test"})
    config = tmp_path / "admission.toml"
    config.write_text(text)
    profile = preflight.card_local_memory_profile(12.)
    probe = {"free_bytes": int(11.25 * preflight.GIB),
             "profile": dataclasses.asdict(profile)}
    monkeypatch.setattr(preflight, "device_memory_probe_subprocess", lambda: probe)
    monkeypatch.setattr(preflight, "profile_from_device_probe", lambda _: profile)
    monkeypatch.setattr(preflight, "declares_the_local_card", lambda *_: False)
    monkeypatch.setattr(doctor, "_cuda_headers_check", lambda: pytest.fail("GPU probed"))
    calls = []
    actual = preflight.estimate_phases

    def record(exp, **kwargs):
        calls.append(kwargs.copy())
        return actual(exp, **kwargs)

    monkeypatch.setattr(preflight, "estimate_phases", record)
    parser = argparse.ArgumentParser()
    preflight.register_cli(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(["check", str(config), "--free-gib", "11.25",
                              "--vram-gib", "12", "--json"])
    assert preflight.check_main(args) == 0
    report = json.loads(capsys.readouterr().out)
    gate = go_cli.memory_gate({"config": str(config), "cadence": 6}, vram_gib=12.)
    assert len(calls) == 2
    assert all(call["forcing_interval_seconds"] == 3600 for call in calls)
    assert all(call["ingest_forcing_interval_seconds"] == 3600 for call in calls)
    assert all(call["forcing_intervals"] == 8 for call in calls)
    assert report["ingest"]["forcing_times"] == 9
    assert report["peak_envelope_bytes"] == gate["phases"].peak_envelope_bytes
    assert report["envelope_budget_bytes"] == gate["budget_bytes"]
    args.forcing_interval_s = 21600
    with pytest.raises(ValueError, match="disagrees with the supplied forcing cadence"):
        preflight.check_main(args)
