"""Renderer output metadata: counts, exact reasons and durable image bindings."""
import json
from pathlib import Path
import pytest
from gpuwm import render_receipts as receipts


def _png(root, family, name):
    path=root/"d01-12km"/family/"2013-05-31"/(name+".png")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(b"owned renderer-result metadata fixture "+name.encode())
    return path


def _publish(root, written, skipped=(), failures=(), spec="temperature,wind_gust"):
    return receipts.publish_invocation(root=root,engine="rust",requested_spec=spec,
        written=written,failures=failures,skipped=skipped,layout="nested")


def test_early_and_final_render_counts_include_early_skips_and_exact_reasons(tmp_path):
    early=_png(tmp_path,"temperature","early")
    first=_publish(tmp_path,[early],[("wind_gust","first frame: not stored: wind_gust_10m_agl")])
    assert first["rendered_png_count"]==1
    later=_png(tmp_path,"temperature","later")
    final=_publish(tmp_path,[later],[("wind_gust","second frame: not stored: wind_gust_10m_agl")])
    assert final["schema"]=="gpuwm.render-summary.v1"
    assert final["requested_family_count"]==2
    assert final["rendered_png_count"]==2 and final["rendered_family_count"]==1
    assert final["skipped_count"]==2 and final["skipped_family_count"]==1
    assert final["skipped_families"]==[{"name":"wind_gust","count":2,
        "reasons":["first frame: not stored: wind_gust_10m_agl","second frame: not stored: wind_gust_10m_agl"],"additional_reasons":0}]
    assert final["invocation_count"]==2
    assert receipts.read_summary(tmp_path)==final
    assert len(list((tmp_path/".render-receipts").glob("*.json")))==2


def test_repeated_image_path_counts_once_and_changed_file_cannot_claim_success(tmp_path):
    image=_png(tmp_path,"temperature","one")
    _publish(tmp_path,[image])
    summary=_publish(tmp_path,[image])
    assert summary["rendered_png_count"]==1 and summary["invocation_count"]==2
    image.write_bytes(b"changed after receipt")
    with pytest.raises(ValueError,match="changed after its render receipt"):
        receipts.summarize(tmp_path)


def test_full_exact_reasons_survive_bounded_status_summary(tmp_path):
    skipped=[(f"family_{index}",f"input {number}: "+"雨"*500) for index in range(64) for number in range(4)]
    summary=_publish(tmp_path,[],skipped,failures=["native decode failed"],spec="all")
    assert summary["requested_families"] is None and summary["requested_family_count"] is None
    assert summary["skipped_count"]==256 and summary["failure_count"]==1
    assert len(json.dumps(summary).encode())<64*1024
    assert Path(summary["summary_path"]).stat().st_size<=receipts._MAX_STATUS_BYTES
    exact=json.loads(Path(summary["receipt_paths"][0]).read_text())["skipped"]
    assert [(row["family"],row["reason"]) for row in exact]==skipped
    for row in summary["skipped_families"]:
        assert row["additional_reasons"]+len(row["reasons"])==4
        assert all((row["name"],reason) in skipped for reason in row["reasons"])


def test_receipt_refuses_png_outside_owned_output_directory(tmp_path):
    output=tmp_path/"output"
    other=_png(tmp_path/"unrelated","temperature","one")
    with pytest.raises(ValueError,match="inside its output directory"):
        _publish(output,[other])


def test_render_stage_end_adds_the_actual_optional_summary_to_stage_events(tmp_path):
    from gpuwm.runplan import EventStream, RunObserver, _GoObserver, read_events
    summary=_publish(tmp_path/"png",[_png(tmp_path/"png","temperature","one")],
                     [("wind_gust","not stored: wind_gust_10m_agl")])
    with EventStream(tmp_path/"events.jsonl",mirror=None) as events:
        observer=RunObserver(events)
        observer.enter_stage("finalize",phase="render")
        _GoObserver(observer).stage_end(label="render",exit_code=0,ok=True,elapsed_seconds=1.,progress=summary)
        observer.finish_stage()
    assert read_events(tmp_path/"events.jsonl")[-1]["render_summary"]==summary


def test_first_products_preserves_native_receipts_before_scratch_cleanup(tmp_path):
    from datetime import datetime
    import hashlib
    import subprocess
    from gpuwm.first_products import FirstProducts
    from gpuwm.render_layout import fs_path
    root = tmp_path / "png"
    frame = tmp_path / "wrfout_d01_2013-05-31_12_00_00"
    frame.write_bytes(b"owned committed frame fixture")
    original_receipts = []
    warnings = []

    def renderer(command):
        scratch = Path(command[command.index("--out") + 1])
        image = _png(scratch, "temperature", "early")
        summary = _publish(scratch, [image], [("wind_gust", "not stored on initial frame")])
        original_receipts.append(Path(summary["receipt_paths"][0]).read_bytes())
        return subprocess.CompletedProcess(command, 0, "", "")

    trigger = FirstProducts({"run": tmp_path, "render": root, "render_products": "temperature,wind_gust"},
        report=lambda receipt: None, warn=lambda *args, **kwargs: warnings.append((args, kwargs)), runner=renderer)
    assert trigger.frame_committed(domain=1, valid_time=datetime(2013, 5, 31, 12), path=frame)
    assert trigger.wait(timeout=10) is not None
    assert not warnings and not (root / ".first-products-scratch").exists()
    early = receipts.read_summary(root)
    assert early["rendered_png_count"] == 1 and early["first_products_included"] is True
    invocation = json.loads(Path(early["receipt_paths"][0]).read_text())
    original = Path(invocation["publication"]["preserved_original_path"])
    assert original.read_bytes() == original_receipts[0]
    assert invocation["publication"]["source_receipt_sha256"] == hashlib.sha256(original.read_bytes()).hexdigest()
    assert all(Path(row["path"]).is_relative_to(Path(fs_path(root, descend=True)).resolve()) for row in invocation["rendered"])
    final = _publish(root, [_png(root, "temperature", "later")], [("wind_gust", "not stored on later frame")])
    assert final["rendered_png_count"] == 2 and final["invocation_count"] == 2
    assert final["skipped_count"] == 2 and final["first_products_included"] is True


def _legacy_first_products(root, frame, image, *, context=True, repeated=False):
    import hashlib
    final_image = image if repeated else _png(root, "temperature", "later")
    summary = receipts.publish_invocation(root=root, engine="rust", requested_spec="temperature",
        written=[final_image], failures=[], skipped=[], layout="nested", context_inputs=[frame] if context else [])
    first = {"schema": "gpuwm.first-products.v1", "published_unix_ms": 1369994400000,
        "frame": str(frame), "frame_sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
        "render_products": "temperature", "written": [{"name": image.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(image.read_bytes()).hexdigest()}]}
    path = root / "first-products.json"
    path.write_text(json.dumps(first))
    return summary, path


@pytest.mark.parametrize("repeated", [False, True])
def test_legacy_summary_counts_unique_receipted_paths_without_image_or_weather_reads(tmp_path, monkeypatch, repeated):
    root = tmp_path / "png"
    frame = tmp_path / "wrfout_d01_2013-05-31_12_00_00"
    frame.write_bytes(b"owned committed frame fixture")
    image = _png(root, "temperature", "early")
    summary, first = _legacy_first_products(root, frame, image, repeated=repeated)
    originals = {path: path.read_bytes() for path in root.rglob("*.json")}
    real_open = Path.open

    def metadata_only(path, *args, **kwargs):
        assert path != frame and path.suffix != ".png", "status must read metadata only"
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", metadata_only)
    monkeypatch.setattr(receipts, "_hash", lambda path: pytest.fail("status must not hash images or WRFs"))
    merged = receipts.merge_recorded_summary(root, summary)
    assert merged["rendered_png_count"] == (1 if repeated else 2)
    assert merged["invocation_count"] == 2 and merged["first_products_included"] is True
    assert Path(merged["first_products_receipt"]["path"]).samefile(first)
    assert merged["first_products_receipt"]["skips_available"] is False
    assert {path: path.read_bytes() for path in originals} == originals
    assert len(json.dumps(merged).encode()) < 60 * 1024


def test_legacy_receipt_must_be_the_final_invocation_context(tmp_path):
    root = tmp_path / "png"
    frame = tmp_path / "frame"
    frame.write_bytes(b"unrelated frame")
    summary, _ = _legacy_first_products(root, frame, _png(root, "temperature", "early"), context=False)
    assert receipts.merge_recorded_summary(root, summary) is summary


def test_legacy_receipt_cannot_claim_an_image_outside_the_render_tree(tmp_path):
    root = tmp_path / "png"
    frame = tmp_path / "frame"
    frame.write_bytes(b"owned frame")
    summary, path = _legacy_first_products(root, frame, _png(root, "temperature", "early"))
    first = json.loads(path.read_text())
    first["written"][0]["name"] = "../outside.png"
    path.write_text(json.dumps(first))
    with pytest.raises(ValueError, match="inside its output directory"):
        receipts.merge_recorded_summary(root, summary)
