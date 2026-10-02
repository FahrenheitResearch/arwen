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


def test_a_delivery_outside_the_nested_layout_is_reported_as_broken(tmp_path):
    """A picture flat at the root did not reach the layout it claims."""
    flat = tmp_path / "arwen_wrf_19740403_18z_f000_d01-1km_composite_reflectivity.png"
    flat.write_bytes(b"owned renderer-result metadata fixture flat")
    summary = receipts.deliver(root=tmp_path, engine="rust",
                               requested_spec="composite_reflectivity",
                               written=[flat], failures=[], skipped=[],
                               layout="nested")
    assert summary["rendered_png_count"] == 0
    assert summary["failure_count"] == 1
    assert flat.name in summary["failures"][0]


def test_the_same_delivery_is_one_clean_row_under_the_flat_layout(tmp_path):
    flat = tmp_path / "arwen_wrf_19740403_18z_f000_d01-1km_composite_reflectivity.png"
    flat.write_bytes(b"owned renderer-result metadata fixture flat")
    summary = receipts.deliver(root=tmp_path, engine="rust",
                               requested_spec="composite_reflectivity",
                               written=[flat], failures=[], skipped=[],
                               layout="flat")
    assert summary["rendered_png_count"] == 1 and summary["failure_count"] == 0
    assert summary["rendered_families"] == [{"name": "composite_reflectivity", "count": 1}]


def test_a_lane_names_its_own_product_instead_of_unclassified(tmp_path):
    """An ensemble panel's filename is not the wrfout engine's grammar."""
    panel = tmp_path / "d02-3km" / "refl-ens-mean" / "1974-04-03" / "refl-ens-mean_d02-3km_1974-04-03_18-00-00.png"
    panel.parent.mkdir(parents=True, exist_ok=True)
    panel.write_bytes(b"owned renderer-result metadata fixture ensemble")
    summary = receipts.deliver(root=tmp_path, engine="rust", requested_spec="refl:mean",
                               written=[panel], failures=[], skipped=[], layout="nested",
                               families={str(panel): "refl-ens-mean"})
    assert summary["rendered_png_count"] == 1
    assert summary["rendered_families"] == [{"name": "refl-ens-mean", "count": 1}]


def test_absolute_product_keys_follow_the_same_filesystem_spelling_as_images(
        tmp_path, monkeypatch):
    """A path alias must preserve a named product in the actual receipt.

    Windows adds a long-path prefix to receipt paths.  A hard-linked alias
    provides the same identity with a different canonical spelling on every
    platform, so this regression also exercises that lookup on Linux.
    """
    from gpuwm import render_layout

    image = tmp_path / "panel.png"
    image.write_bytes(b"owned renderer-result metadata fixture panel")
    canonical = tmp_path / "filesystem-panel.png"
    canonical.hardlink_to(image)
    assert canonical.samefile(image)
    original_fs_path = render_layout.fs_path
    image_fs = Path(original_fs_path(image, descend=True)).resolve()

    def filesystem_path(value, *, descend=False):
        result = original_fs_path(value, descend=descend)
        if Path(result).resolve() == image_fs:
            return original_fs_path(canonical, descend=descend)
        return result

    monkeypatch.setattr(render_layout, "fs_path", filesystem_path)
    names = {str(image.resolve()): "diagnostic_panel"}
    summary = receipts.publish_invocation(
        root=tmp_path, engine="rust", requested_spec="diagnostic_panel",
        written=[image], failures=[], skipped=[], layout="flat", families=names)
    assert summary["rendered_png_count"] == 1
    assert summary["rendered_families"] == [{"name": "diagnostic_panel", "count": 1}]
    assert receipts.read_summary(tmp_path) == summary
    assert names == {str(image.resolve()): "diagnostic_panel"}


def test_a_frame_that_reached_the_reader_by_a_lesser_route_is_recorded(tmp_path):
    """The layout degradation used to exist only on stderr."""
    image = _png(tmp_path, "temperature", "one")
    summary = receipts.publish_invocation(
        root=tmp_path, engine="rust", requested_spec="2m_temperature",
        written=[image], failures=[], skipped=[], layout="nested",
        degraded=[(tmp_path / "left.png", "left flat, could not move into layout")])
    assert summary["degraded_count"] == 1
    assert summary["degraded"][0]["reason"].startswith("left flat")
    assert receipts.read_summary(tmp_path)["degraded_count"] == 1


def test_the_node_side_gallery_publishes_a_receipt(tmp_path):
    """Every PNG-writing lane leaves a record, not only gpuwm render."""
    from gpuwm import remote_native_plots

    panels = []
    for slug in ("composite_reflectivity", "2m_temperature"):
        path = tmp_path / f"{slug}.png"
        path.write_bytes(b"owned renderer-result metadata fixture " + slug.encode())
        panels.append({"slug": slug, "path": str(path),
                       "bytes": path.stat().st_size})
    remote_native_plots._publish_receipt(
        tmp_path, panels, [row["slug"] for row in panels])
    summary = receipts.read_summary(tmp_path)
    assert summary is not None
    assert summary["rendered_png_count"] == 2
    assert sorted(row["name"] for row in summary["rendered_families"]) == [
        "2m_temperature", "composite_reflectivity"]


def _prior_receipts(root, count, failures=()):
    """``count`` small, valid receipts already filed in ``root``."""
    directory = root / ".render-receipts"
    directory.mkdir(parents=True, exist_ok=True)
    total = 0
    for index in range(count):
        ident = f"{index:032x}"
        payload = json.dumps({
            "schema": receipts.INVOCATION_SCHEMA, "id": ident,
            "created_utc": "2026-09-27T00:00:00+00:00",
            "output_root": str(root), "engine": "rust",
            "requested_spec": "temperature", "layout": "nested",
            "rendered": [], "skipped": [], "failures": list(failures)}).encode()
        (directory / f"{ident}.json").write_bytes(payload)
        total += len(payload)
    return total


def test_a_folder_drawn_through_a_long_run_keeps_publishing(tmp_path):
    """One receipt per pass: a long run drawn while it runs files thousands.

    They are small and fit the aggregate byte bound many times over, so the
    next pass publishes and the summary counts every one of them.
    """
    from gpuwm.render_layout import fs_path
    root = Path(fs_path(tmp_path, descend=True)).resolve()
    prior = _prior_receipts(root, 4096)
    assert prior < receipts._MAX_RECEIPT_BYTES // 4
    summary = _publish(root, [_png(root, "temperature", "next")], spec="temperature")
    assert summary["invocation_count"] == 4097
    assert summary["rendered_png_count"] == 1
    assert len(summary["receipt_paths"]) + summary["additional_receipts"] == 4097
    assert receipts.read_summary(root)["invocation_count"] == 4097


def test_receipts_past_the_aggregate_byte_bound_are_refused(tmp_path, monkeypatch):
    from gpuwm.render_layout import fs_path
    root = Path(fs_path(tmp_path, descend=True)).resolve()
    prior = _prior_receipts(root, 3, failures=["x" * 2048])
    monkeypatch.setattr(receipts, "_MAX_RECEIPT_BYTES", prior - 1)
    with pytest.raises(ValueError, match="aggregate byte bound"):
        receipts.summarize(root)


def test_long_valid_selections_publish_a_bounded_summary(tmp_path):
    """A long selection must not stop publication.

    Four invocations, each asking for 128 stored fields under 128-character
    names (the viewer's own bounds), filled the status envelope with the
    preview lists themselves, and the summary refused with "exceeds its
    status bound" once no reason text was left to drop.
    """
    from gpuwm.remote_processed_v2 import _products
    selections = []
    for batch in range(4):
        names = [f"var:field_{batch}_{index:03d}_" + "x" * 100 for index in range(128)]
        assert _products(names) == sorted(names)
        selections.append(",".join(names))
        summary = _publish(tmp_path, [], [(name, "not stored in this frame") for name in names],
                           spec=selections[-1])
    assert Path(summary["summary_path"]).stat().st_size <= receipts._MAX_STATUS_BYTES
    assert receipts.read_summary(tmp_path) == summary
    assert summary["requested_family_count"] == 512
    assert summary["skipped_count"] == 512 and summary["undrawn_family_count"] == 512
    for rows, count, total in (("requested_specs", "additional_requested_specs", 4),
                               ("requested_families", "additional_requested_families", 512),
                               ("skipped_families", "additional_skipped_families", 512),
                               ("undrawn_families", "additional_undrawn_families", 512)):
        assert len(summary[rows]) + summary[count] == total, rows
    assert summary["additional_requested_specs"] > 0
    exact = [json.loads(path.read_text(encoding="utf-8"))
             for path in (tmp_path / ".render-receipts").glob("*.json")]
    assert sorted(row["requested_spec"] for row in exact) == sorted(selections)
    assert all(len(row["skipped"]) == 128 for row in exact)


def test_a_short_selection_keeps_every_preview_row(tmp_path):
    summary = _publish(tmp_path, [], [("var:temperature_2m", "not stored")],
                       spec="var:temperature_2m")
    assert summary["requested_specs"] == ["var:temperature_2m"]
    assert summary["requested_families"] == ["var:temperature_2m"]
    assert summary["additional_requested_specs"] == 0
    assert summary["additional_requested_families"] == 0
    assert summary["additional_undrawn_families"] == 0
