"""Explicit donor authority and the sealed/live native reader contract."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from gpuwm.ingest.native_supplements import (
    gate_supplement_fields, supplement_bindings, verify_supplement_receipt,
)
from tools.hrrr_pipeline import _parse_series, verify_source_tree


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(root):
    paths = sorted(path for path in root.rglob("*")
                   if path.is_file() and path.name != "SHA256SUMS")
    manifest = root / "SHA256SUMS"
    manifest.write_text("".join(
        f"{_digest(path)}  {path.relative_to(root).as_posix()}\n" for path in paths))
    return _digest(manifest)


def test_declared_donor_uses_source_manifest_and_is_rechecked_at_publication(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("atmos", "soil", "pmsl"):
        (source / name).write_bytes(name.encode())
    manifest_sha = _manifest(source)
    series = tmp_path / "series.tsv"
    series.write_text("".join(
        f"{hour}\tsource/atmos\tsource/soil\tPMSL=source/pmsl\n" for hour in (0, 1)))
    receipt = verify_source_tree(
        source_root=source, manifest=source / "SHA256SUMS",
        expected_manifest_sha256=manifest_sha, series=series, workers=1)
    assert receipt["supplement_bindings"] == [
        {"forecast_hour": hour, "field": "PMSL", "path": str(source / "pmsl"),
         "sha256": _digest(source / "pmsl")} for hour in (0, 1)]
    verify_supplement_receipt(receipt)
    (source / "pmsl").write_bytes(b"changed donor")
    with pytest.raises(ValueError, match="changed before publication"):
        verify_supplement_receipt(receipt)
    with pytest.raises(ValueError, match="payload hash mismatch"):
        verify_source_tree(source_root=source, manifest=source / "SHA256SUMS",
                           expected_manifest_sha256=manifest_sha, series=series, workers=1)


def test_undeclared_donor_is_not_authenticated_by_its_series_path(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "primary").write_bytes(b"primary")
    manifest_sha = _manifest(source)
    (source / "donor").write_bytes(b"unbound")
    series = tmp_path / "series.tsv"
    series.write_text("".join(
        f"{h}\tsource/primary\tsource/primary\tPMSL=source/donor\n" for h in (0, 1)))
    with pytest.raises(ValueError, match="not bound by source manifest"):
        verify_source_tree(source_root=source, manifest=source / "SHA256SUMS",
                           expected_manifest_sha256=manifest_sha, series=series, workers=1)


@pytest.mark.parametrize("values", [("PMSL=",), ("other=file",), ("PMSL=x", "PMSL=x")])
def test_invalid_or_duplicate_declaration_is_named(values):
    with pytest.raises(ValueError, match="supplement|duplicate"):
        supplement_bindings(values)


def test_native_reader_keeps_default_arrays_and_loads_only_declared_supplements(tmp_path):
    from gpuwm.ingest.hrrr import (
        _ATMOSPHERE_2D, _ATMOSPHERE_3D, load_hrrr_native_series,
        load_hrrr_pipeline_ready_window,
    )
    root = tmp_path / "bridge"
    root.mkdir()
    base_gate = ("status\tPASS\ncycle\t2021-12-10 18:00:00\n"
                 "forecast_hours\t0,1\nseries_count\t2\n"
                 "atmosphere_selected_per_time\t561\nhybrid_levels\t50\n"
                 "soil_selected_per_time\t18\nwindow_shape\t2x2\n"
                 "qice_mapping\tPASS discipline=0 category=1 parameter=82 level_type=105\n"
                 "cross_time_inventory\tPASS\n"
                 "window_zero_based_inclusive\ti=5..6 j=7..8\n")
    (root / "gate.txt").write_text(base_gate)
    for hour in (0, 1):
        atmosphere = root / f"atmosphere-f{hour:02d}"
        soil = root / f"soil-f{hour:02d}"
        atmosphere.mkdir(); soil.mkdir()
        for name in _ATMOSPHERE_3D:
            np.full((50, 2, 2), hour + 1, dtype="<f4").tofile(atmosphere / f"{name}.f32le")
        for name in _ATMOSPHERE_2D:
            np.full((2, 2), hour + 2, dtype="<f4").tofile(atmosphere / f"{name}.f32le")
        for name in ("SOILT", "SOILW"):
            np.full((9, 2, 2), hour + 3, dtype="<f4").tofile(soil / f"{name}.f32le")
    default = load_hrrr_native_series(root, (0, 1), expected_manifest_sha256=_manifest(root))
    assert all("PMSL" not in frame.fields for frame in default)
    (root / "gate.txt").write_text(base_gate +
        "supplement_fields\tPMSL\nsupplement_units\tPMSL=Pa\n"
        "supplement_alignment\texact_primary_grid_and_source_time\n")
    for hour in (0, 1):
        np.full((2, 2), 101000 + hour, dtype="<f4").tofile(
            root / f"atmosphere-f{hour:02d}/PMSL.f32le")
    supplemented = load_hrrr_native_series(root, (0, 1), expected_manifest_sha256=_manifest(root))
    from gpuwm.ingest.native_supplements import require_native_pressure_field
    require_native_pressure_field({"sfcp_to_sfcp": False}, bridge_root=root)
    for old, new in zip(default, supplemented):
        for name in old.fields:
            assert old.fields[name].tobytes() == new.fields[name].tobytes()
        live = load_hrrr_pipeline_ready_window(root, new.forecast_hour)
        np.testing.assert_array_equal(live.fields["PMSL"], new.fields["PMSL"])
        assert new.fields["PMSL"][0, 0] == 101000 + new.forecast_hour
    manifest = root / "SHA256SUMS"
    manifest.write_text("\n".join(row for row in manifest.read_text().splitlines()
                                    if "PMSL.f32le" not in row) + "\n")
    with pytest.raises(ValueError, match="supplement payload is not bound"):
        load_hrrr_native_series(root, (0, 1), expected_manifest_sha256=_digest(manifest))


def test_loader_refuses_unestablished_units_or_alignment():
    with pytest.raises(ValueError, match="units or alignment"):
        gate_supplement_fields({"supplement_fields": "PMSL", "supplement_units": "PMSL=hPa"})


def test_public_native_command_preserves_explicit_donor_through_actual_parser(capsys, tmp_path):
    import shlex
    from gpuwm.source_cli import main
    from tools.prepare_hrrr_wrf import _parser
    donor = tmp_path / "pressure donor.grib2"
    result = main([
        "--source", "hrrr", "--source-root", str(tmp_path),
        "--source-sha256s", str(tmp_path / "SHA256SUMS"),
        "--source-sha256s-sha256", "0" * 64,
        "--geog-root", str(tmp_path / "geog"),
        "--domain-spec", str(tmp_path / "domain.json"),
        "--namelist-input", str(tmp_path / "namelist.input"),
        "--valid-time", "2021-12-10_18:00:00", "--output-root", str(tmp_path / "out"),
        "--supplement", f"PMSL={donor}", "--dry-run"])
    assert result == 0
    command = shlex.split(capsys.readouterr().out.strip())
    parsed = _parser().parse_args(command[2:])
    assert parsed.supplement == [f"PMSL={donor}".replace(chr(92), "/")]


def test_bridge_extension_retains_declared_field_and_selection_receipt(tmp_path):
    from test_prepare_hrrr_wrf import _fake_sealed_bridge
    from tools.prepare_hrrr_wrf import _bridge_manifest_extension
    prior = _fake_sealed_bridge(tmp_path / "prior", [0, 1])
    suffix = _fake_sealed_bridge(tmp_path / "suffix", [1, 2])
    for root, hours in ((prior, [0, 1]), (suffix, [1, 2])):
        gate = root / "gate.txt"
        gate.write_text(gate.read_text() +
            "supplement_fields\tPMSL\nsupplement_units\tPMSL=Pa\n"
            "supplement_alignment\texact_primary_grid_and_source_time\n")
        for hour in hours:
            (root / f"atmosphere-f{hour:02d}/PMSL.f32le").write_bytes(f"pmsl:{hour}".encode())
        (root / "supplement-inventory.tsv").write_text(
            "forecast_hour\tfield\n" + "".join(f"{h}\tPMSL\n" for h in hours))
        _manifest(root)
    output = tmp_path / "merged"
    _bridge_manifest_extension(predecessor=prior, suffix=suffix, output=output,
                               old_hours=[0, 1], new_hours=[0, 1, 2])
    for hour in (0, 1, 2):
        assert (output / f"atmosphere-f{hour:02d}/PMSL.f32le").read_bytes() == f"pmsl:{hour}".encode()
    assert (output / "supplement-inventory.tsv").read_text().splitlines() == [
        "forecast_hour\tfield", "0\tPMSL", "1\tPMSL", "2\tPMSL"]
    from tools.prepare_hrrr_wrf import _manifest_entries, _verify_manifest_payloads
    _verify_manifest_payloads(output, _manifest_entries(output / "SHA256SUMS"))
    authority = prior / "SHA256SUMS"
    authority.write_text("\n".join(row for row in authority.read_text().splitlines()
                                  if "supplement-inventory.tsv" not in row) + "\n")
    with pytest.raises(ValueError, match="selection receipt is not bound"):
        _bridge_manifest_extension(predecessor=prior, suffix=suffix,
                                   output=tmp_path / "unbound-merge",
                                   old_hours=[0, 1], new_hours=[0, 1, 2])
    assert not (tmp_path / "unbound-merge").exists()


def test_source_extension_accepts_only_explicit_new_donor_and_keeps_prefix(tmp_path):
    from datetime import datetime
    from tools.prepare_hrrr_wrf import _source_manifest_extension
    root = tmp_path / "source"
    root.mkdir()
    for hour in (0, 1):
        for kind in ("wrfnat", "soil"):
            (root / f"hrrr.t18z.{kind}f{hour:02d}.grib2").write_bytes(f"{kind}:{hour}".encode())
    _manifest(root)
    prior = tmp_path / "prior-SHA256SUMS"
    prior.write_bytes((root / "SHA256SUMS").read_bytes())
    for kind in ("wrfnat", "soil"):
        (root / f"hrrr.t18z.{kind}f02.grib2").write_bytes(f"{kind}:2".encode())
    donor = root / "pressure-donor.grib2"
    donor.write_bytes(b"declared donor")
    _manifest(root)
    args = dict(predecessor=prior, extended=root / "SHA256SUMS", source_root=root,
                old_hours=[0, 1], new_hours=[0, 1, 2], cycle=datetime(2021, 12, 10, 18))
    with pytest.raises(ValueError, match="explicitly declared"):
        _source_manifest_extension(**args)
    receipt = _source_manifest_extension(**args, supplemental_paths=[donor])
    assert [row["path"] for row in receipt["added_entries"]] == [
        "hrrr.t18z.soilf02.grib2", "hrrr.t18z.wrfnatf02.grib2", "pressure-donor.grib2"]


def test_missing_pressure_is_named_before_static_or_native_work(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tools import prepare_hrrr_wrf as prepare
    import gpuwm.hrrr_configuration as configuration
    import gpuwm.ingest.hrrr_target as target
    import gpuwm.vertical_contract as vertical
    namelist = tmp_path / "namelist.input"
    namelist.write_text("&domains\n sfcp_to_sfcp = .false.,\n/\n")
    monkeypatch.setattr(target, "load_hrrr_target_domain", lambda path: SimpleNamespace(nz=49))
    monkeypatch.setattr(vertical, "explicit_vertical_from_wrf_namelist", lambda *a, **kw: object())
    monkeypatch.setattr(configuration, "resolve_root_experiment", lambda **kw: (
        SimpleNamespace(root=SimpleNamespace(run=object())), {}))
    monkeypatch.setattr(prepare, "_require_microphysics_tables", lambda *a: pytest.fail("native work preceded missing-field refusal"))
    monkeypatch.setattr(prepare, "_decoder", lambda *a: pytest.fail("decoder resolution preceded missing-field refusal"))
    with pytest.raises(ValueError, match="sfcp_to_sfcp=false requires analyzed PMSL"):
        prepare._prepare_from_argv([
            "--source-root", str(tmp_path), "--source-manifest", str(tmp_path / "SHA256SUMS"),
            "--source-manifest-sha256", "0" * 64, "--namelist-input", str(namelist),
            "--geog-root", str(tmp_path / "geog"), "--domain-spec", str(tmp_path / "domain.json"),
            "--cycle", "2021-12-10_18:00:00", "--run-seconds", "3600",
            "--history-interval-seconds", "60", "--output-root", str(tmp_path / "output")])
    assert not (tmp_path / "output").exists()


def test_pressure_choice_is_never_changed_to_fill_missing_donor():
    from gpuwm.ingest.native_supplements import require_native_pressure_field
    requested = {"sfcp_to_sfcp": False}
    with pytest.raises(ValueError, match="--supplement PMSL=GRIB"):
        require_native_pressure_field(requested)
    assert requested == {"sfcp_to_sfcp": False}
    require_native_pressure_field({"sfcp_to_sfcp": True})
    require_native_pressure_field(requested, bindings=[("PMSL", Path("declared"))])
