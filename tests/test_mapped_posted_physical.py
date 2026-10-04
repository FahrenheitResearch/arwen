"""Mapped posted physical fields retain ordinary lead and trajectory authority."""
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm import mapped_direct
from gpuwm.ingest.boundary_stream import posted_lead_marker_sha256


def _source():
    source = object.__new__(mapped_direct._PostedMappedSource)
    start = datetime(2024, 5, 21, 12)
    marker = {"source": "gefs", "cycle": "2024-05-21T12:00:00Z", "member": "p01",
              "lead": 0, "valid_time": start.isoformat(), "objects": [
                  {"name": "p01.f000.grib2", "sha256": "a" * 64, "bytes": 123}]}
    source.__dict__.update(
        schedule={"source": "gefs", "cycle": marker["cycle"], "member": "p01"},
        leads=(0, 3), valid_times=(start, start + timedelta(hours=3)),
        markers={0: marker}, through=lambda index: None,
        frames=SimpleNamespace(header=lambda index: SimpleNamespace(source_cycle=start.isoformat())),
        _physical_evidence={0: {"input_manifest_sha256": "b" * 64,
            "bundle": SimpleNamespace(), "primary_rows": (),
            "posted_leads": {"0": posted_lead_marker_sha256(marker)}}})
    return source


def test_first_physical_knot_needs_no_future_marker_or_full_manifest(monkeypatch):
    source = _source()
    monkeypatch.setattr(mapped_direct, "mapped_composition_receipt", lambda _: {})
    monkeypatch.setattr(mapped_direct, "composition_receipt_identity_sha256", lambda _: "c" * 64)
    trajectory = source.physical_trajectory()
    assert trajectory.member == "p01"
    assert len(source.markers) == 1
    evidence = source.physical_evidence(0)
    assert evidence["posted_leads"] == {"0": posted_lead_marker_sha256(source.markers[0])}
    assert set(evidence["decoded_leads"]) == {"0"}
    assert len(evidence["decoded_leads"]["0"]) == 64


def test_physical_trajectory_accepts_the_ordinary_naive_utc_cycle_spelling():
    source = _source()
    source.schedule["cycle"] = "2024-05-21T12"
    source.markers[0]["cycle"] = "2024-05-21T12"
    assert source.physical_trajectory().cycle.isoformat() == "2024-05-21T12:00:00+00:00"


@pytest.mark.parametrize("change", [
    {"member": "p02"}, {"source": "geps"}, {"cycle": "2024-05-21T06:00:00Z"},
    {"valid_time": "2024-05-21T15:00:00"},
])
def test_physical_capture_refuses_foreign_member_cycle_source_or_time(change):
    source = _source()
    source.markers[0].update(change)
    with pytest.raises(ValueError):
        source.physical_trajectory()


def test_physical_capture_refuses_raw_marker_changes_after_decode(monkeypatch):
    source = _source()
    source.markers[0]["objects"][0]["sha256"] = "d" * 64
    with pytest.raises(ValueError, match="raw marker changed"):
        source.physical_evidence(0)


def test_physical_capture_refuses_native_cycle_even_when_markers_agree():
    source = _source()
    source.frames = SimpleNamespace(header=lambda index: SimpleNamespace(
        source_cycle="2024-05-21T06:00:00"))
    with pytest.raises(ValueError, match="decoded source cycle"):
        source.physical_evidence(0)


def test_physical_capture_checks_actual_member_bytes_once_per_batch(monkeypatch, tmp_path):
    from gpuwm import forcing_member, member_prep
    source = _source()
    path = tmp_path / "member.grib2"
    path.write_bytes(b"analytical member fixture")
    source._physical_evidence[0]["primary_rows"] = ((path, mapped_direct._sha256(path)),)
    monkeypatch.setattr(mapped_direct, "mapped_composition_receipt", lambda _: {})
    monkeypatch.setattr(mapped_direct, "composition_receipt_identity_sha256", lambda _: "c" * 64)
    monkeypatch.setattr(forcing_member, "member_contract", lambda source, member: (None, "grammar", member))
    calls = []
    monkeypatch.setattr(member_prep, "verify_member_file", lambda *args: calls.append(args))
    source.physical_evidence(0)
    source.physical_evidence(0)
    assert calls == [("grammar", "p01", path)]


def test_physical_capture_refuses_changed_native_member_input(tmp_path):
    source = _source()
    path = tmp_path / "member.grib2"
    path.write_bytes(b"original")
    source._physical_evidence[0]["primary_rows"] = ((path, mapped_direct._sha256(path)),)
    path.write_bytes(b"different")
    with pytest.raises(ValueError, match="primary bytes changed"):
        source.physical_evidence(0)


def test_physical_capture_propagates_the_ordinary_future_lead_failure():
    source = _source()
    class SourceFailure(RuntimeError):
        pass
    def through(index):
        if index == 1:
            raise SourceFailure("ordinary source did not post its next lead")
    source.through = through
    with pytest.raises(SourceFailure, match="ordinary source"):
        source.physical_evidence(1)


def test_mapped_cli_keeps_the_original_physical_member_index(monkeypatch):
    from test_mapped_direct import _mapped_cli_args
    captured = {}
    monkeypatch.setattr(mapped_direct, "load_mapping", lambda _: {"format": "grib2"})
    monkeypatch.setattr(mapped_direct, "prepare_mapped_wrf", lambda **kwargs:
                        captured.update(kwargs) or {"schema": "proof", "status": "PASS"})
    argv = _mapped_cli_args("grib2", "--grib2-inventory", "/bin/inventory",
                            "--grib2-dump", "/bin/dump")
    argv.extend(("--physical-input-provider", "/case/provider", "--physical-member-index", "19"))
    assert mapped_direct.main(argv) == 0
    assert captured["physical_input_provider"] == Path("/case/provider")
    assert captured["physical_member_index"] == 19


def test_shared_context_posted_capture_enters_the_ordinary_native_producer(monkeypatch, tmp_path):
    from test_ensemble_mapped_physical import _context
    from gpuwm import prep_handoff, source_adapters, source_cli
    from gpuwm.ensemble import posted_physical
    context, _, _, _ = _context(monkeypatch, tmp_path)
    captured = []
    monkeypatch.setattr(prep_handoff, "posted_preparation_arguments_from_directory", lambda root: [
        "--source", "gefs", "--input-list", str(tmp_path / "future-inputs.txt"),
        "--author-input-manifest", str(tmp_path / "inputs.json"),
        "--as-posted", str(tmp_path / "posting")])
    monkeypatch.setattr(source_adapters, "get_source_adapter", lambda source:
                        SimpleNamespace(runner="mapped_composition_v1"))
    monkeypatch.setattr(source_cli, "main", lambda argv: captured.extend(argv) or 0)
    monkeypatch.setattr(posted_physical, "PostedPhysicalStream", lambda root: SimpleNamespace(head_sha256="a" * 64))
    monkeypatch.setattr(posted_physical, "capture_source_seal", lambda root, stream: {"verified": True})
    try:
        result = context.capture_posted(tmp_path / "fetched", tmp_path / "physical",
                                        prepared_root=tmp_path / "prepared", geog_root=tmp_path / "geog")
        assert result["source_seal"] == {"verified": True}
        assert captured[captured.index("--experiment-config") + 1] == str(context.config_path)
        assert captured[captured.index("--static-input") + 1] == str(context.static_input_path)
        assert captured[captured.index("--physical-output-store") + 1] == str(tmp_path / "physical")
        assert "--as-posted" in captured and "--no-stock-wrf-export" in captured
        assert not (tmp_path / "future-inputs.txt").exists()
    finally:
        context.close()
