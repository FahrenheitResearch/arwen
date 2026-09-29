"""A picture removed after it was published does not block the next render.

Breakage this prevents: every render publication re-verifies the pictures of
every earlier invocation in the folder, and one earlier PNG removed by hand
made each later publication fail with "Render receipt names no regular PNG
inside its output directory", after the new pictures were already drawn.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from gpuwm import render_receipts

from test_render_receipts import _png, _publish


def test_a_removed_earlier_png_does_not_refuse_a_new_publication(tmp_path):
    old = _png(tmp_path, "temperature", "old")
    first = _publish(tmp_path, [old])
    receipt = Path(first["receipt_paths"][0])
    recorded = receipt.read_bytes()
    old.unlink()
    new = _png(tmp_path, "wind", "new")
    result = _publish(tmp_path, [new], spec="wind")
    assert result["rendered_png_count"] == 1
    assert result["rendered_family_count"] == 1
    assert result["rendered_families"] == [{"name": "wind", "count": 1}]
    assert result["degraded_count"] == 1
    # The receipt's own spelling of the path (extended-length on Windows).
    assert Path(result["degraded"][0]["path"]).parts[-4:] == old.parts[-4:]
    assert "removed" in result["degraded"][0]["reason"]
    # The earlier invocation's receipt is the record of what it drew, and
    # is kept byte for byte.
    assert receipt.read_bytes() == recorded
    assert render_receipts.summarize(tmp_path)["rendered_png_count"] == 1


def test_a_whole_removed_product_folder_does_not_refuse_either(tmp_path):
    old = _png(tmp_path, "temperature", "old")
    _publish(tmp_path, [old])
    old.unlink()
    old.parent.rmdir()
    new = _png(tmp_path, "wind", "new")
    assert _publish(tmp_path, [new], spec="wind")["rendered_png_count"] == 1


def test_a_file_standing_where_a_picture_folder_was_counts_as_removed(tmp_path):
    # The day folder and the product folder, each replaced by a file.
    for replaced in (1, 2):
        root = tmp_path / str(replaced)
        old = _png(root, "temperature", "old")
        _publish(root, [old])
        folder = old.parents[replaced - 1]
        for path in sorted(folder.rglob("*"), reverse=True):
            path.unlink() if path.is_file() else path.rmdir()
        folder.rmdir()
        folder.write_bytes(b"not a folder")
        summary = render_receipts.summarize(root)
        assert summary["rendered_png_count"] == 0
        assert summary["degraded_count"] == 1
        assert "removed" in summary["degraded"][0]["reason"]
        new = _png(root, "wind", "new")
        assert _publish(root, [new], spec="wind")["rendered_png_count"] == 1


def test_a_new_output_that_was_never_written_still_refuses(tmp_path):
    with pytest.raises(ValueError, match="regular PNG"):
        _publish(tmp_path, [tmp_path / "never-produced.png"])


def test_an_earlier_png_changed_in_place_still_refuses(tmp_path):
    old = _png(tmp_path, "temperature", "old")
    _publish(tmp_path, [old])
    old.write_bytes(b"changed after publication")
    with pytest.raises(ValueError, match="changed after its render receipt"):
        render_receipts.summarize(tmp_path)


def test_an_earlier_png_replaced_by_a_folder_still_refuses(tmp_path):
    old = _png(tmp_path, "temperature", "old")
    _publish(tmp_path, [old])
    old.unlink()
    old.mkdir()
    with pytest.raises(ValueError, match="regular PNG"):
        render_receipts.summarize(tmp_path)
