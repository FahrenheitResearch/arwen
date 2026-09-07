"""Actual preparation stages are observable without consuming JSON stdout."""
import json

import pytest

from gpuwm.progress import PREP_EVENT_PREFIX, PREP_EVENT_SCHEMA, prep_stage


def _events(text):
    return [json.loads(line[len(PREP_EVENT_PREFIX):])
            for line in text.splitlines() if line.startswith(PREP_EVENT_PREFIX)]


def test_stage_pairs_preserve_raw_receipt_stdout(capsys):
    with prep_stage("root_initialize", label="Initialize root forcing states",
                    backend="cpu", count=6):
        print('{"status":"READY"}')
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"status": "READY"}
    started, finished = _events(captured.err)
    assert [started["event"], finished["event"]] == ["started", "finished"]
    for event in (started, finished):
        assert event["schema"] == PREP_EVENT_SCHEMA
        assert event["stage"] == "root_initialize"
        assert event["backend"] == "cpu" and event["count"] == 6
    assert finished["elapsed_seconds"] >= 0


def test_failed_stage_reports_and_preserves_the_original_exception(capsys):
    error = OSError("cannot write prepared cache")
    with pytest.raises(OSError) as caught:
        with prep_stage("prepared_cache"):
            raise error
    assert caught.value is error
    records = _events(capsys.readouterr().err)
    assert [row["event"] for row in records] == ["started", "failed"]
    assert records[-1]["error_type"] == "OSError"
    assert records[-1]["error"] == str(error)


def test_mapped_preparation_reports_real_completed_stages(monkeypatch, tmp_path, capsys):
    import gpuwm.mapped_direct as mapped
    from test_mapped_direct import _install_prepare_fakes
    args, calls, _ = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cpu")
    mapped.prepare_mapped_wrf(**args, stock_wrf_export="off")
    records = _events(capsys.readouterr().err)
    stages = [row["stage"] for row in records if row["event"] == "finished"]
    assert stages == ["source_decode", "root_static", "root_initialize",
                      "prepared_cache", "wrf_export"]
    init = next(row for row in records if row["stage"] == "root_initialize")
    assert init["backend"] == "cpu" and init["count"] == calls["initialize"]
    assert records[-1]["outcome"] == "not_requested"


def test_mapped_decoder_failure_is_not_reported_as_complete(monkeypatch, tmp_path, capsys):
    import gpuwm.mapped_direct as mapped
    from test_mapped_direct import _install_prepare_fakes
    args, _, _ = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cpu")
    def failed(*args, **kwargs):
        raise ValueError("bad source inventory")
    monkeypatch.setattr(mapped, "decode_composed_source", failed)
    with pytest.raises(ValueError, match="bad source inventory"):
        mapped.prepare_mapped_wrf(**args)
    records = _events(capsys.readouterr().err)
    assert [(row["stage"], row["event"]) for row in records] == [
        ("source_decode", "started"), ("source_decode", "failed")]


def test_optional_export_refusal_reports_no_files_produced(monkeypatch, tmp_path, capsys):
    import gpuwm.mapped_direct as mapped
    from gpuwm.wrf_direct import StockWrfExportUnsupported
    from test_mapped_direct import _install_prepare_fakes
    args, _, _ = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=1, backend="cpu")
    def unsupported(*args, **kwargs):
        raise StockWrfExportUnsupported("unrepresentable export")
    monkeypatch.setattr(mapped, "export_prepared_wrf", unsupported)
    proof = mapped.prepare_mapped_wrf(**args)
    assert proof["export"]["status"] == "REFUSED"
    records = _events(capsys.readouterr().err)
    assert records[-1]["event"] == "finished"
    assert records[-1]["outcome"] == "refused"
    assert records[-1]["reason"] == "unrepresentable export"
