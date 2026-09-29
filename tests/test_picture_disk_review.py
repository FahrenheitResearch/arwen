"""Measured picture prices remain traceable and bounded after review."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from gpuwm import disk_budget, runplan, rustwx

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "tools/wiki_seed/runs/bytes-per-product/sizes.json"


@pytest.fixture(autouse=True)
def no_native_catalog(monkeypatch):
    monkeypatch.setattr(runplan, "render_catalog", lambda: {"products": None})


def test_unlisted_groups_cost_one_fallback_picture(monkeypatch):
    table = copy.deepcopy(disk_budget._picture_table())
    assert "unmeasured_groups" not in table
    # An obsolete configuration cannot restore a special group multiplier.
    table["unmeasured_groups"] = {"future_group": 16}
    monkeypatch.setattr(disk_budget, "_picture_table", lambda: table)
    monkeypatch.setattr(rustwx, "GROUP_KEYWORDS", rustwx.GROUP_KEYWORDS | {"future_group"})
    assert disk_budget._picture_products("future_group", measured=True)["future_group"][2] == 1


def test_every_product_price_rebuilds_from_recorded_bracket_maxima():
    record = json.loads(RECORD.read_text())
    table = disk_budget._picture_table()
    assert table["headroom"] == record["headroom"]
    assert table["columns"] == record["columns"]
    assert set(table["products"]) == set(record["catalog"])
    for name, (kind, first) in record["catalog"].items():
        brackets = [[], []]
        for _, product, columns, _, size, _, _ in record["pictures"]:
            if product == name and columns is not None:
                brackets[int(math.prod(columns) > record["columns"][0])].append(size)
        expected = [max(values) if values else default
                    for values, default in zip(brackets, record["unmeasured_bytes"])]
        expected[1] = max(expected)
        assert table["products"][name] == [kind, first, *expected], name
        # The stored maxima are raw bytes; headroom is applied once at pricing.
        if kind == "direct":
            for columns, raw in zip(table["columns"], expected):
                assert disk_budget.projected_picture_bytes(columns, 1, 0, 3600, name) == math.ceil(
                    raw * table["headroom"]), name


def test_collector_rebuild_needs_only_the_committed_record(tmp_path):
    table = tmp_path / "table.json"
    result = subprocess.run([sys.executable, str(ROOT / "tools/wiki_seed/collect_picture_sizes.py"),
                             "--rebuild", "--record", str(RECORD), "--table", str(table)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert table.read_bytes() == disk_budget.picture_table_path().read_bytes()


def test_later_sampled_rain_and_reflectivity_sizes_are_included():
    record = json.loads(RECORD.read_text())
    table = disk_budget._picture_table()
    sampled = {name for name, run in record["runs"].items()
               if not run["complete_inventory"] and run["history_frames"] > 2}
    assert sampled
    for name in ("qpf_total", "composite_reflectivity", "composite_reflectivity_uh"):
        values = [[], []]
        for run, product, columns, _, size, _, _ in record["pictures"]:
            if run in sampled and product == name:
                values[int(math.prod(columns) > table["columns"][0])].append(size)
        assert any(values), name
        for i, sizes in enumerate(values):
            if sizes:
                assert table["products"][name][i + 2] >= max(sizes), name


def test_collector_keeps_partial_inventories_and_each_pictures_grid(tmp_path):
    from tools.wiki_seed.collect_picture_sizes import build_table, collect

    def frame(gid, size):
        return {"attributes": {"GRID_ID": str(gid), "DX": "1000", "DY": "1000"},
                "dimensions": {"west_east": size, "south_north": size}}
    audit = {"checked_utc": "2026-01-01T00:00:00Z", "png_count": 99,
             "frames": [frame(1, 36), frame(2, 76), frame(2, 76)],
             "pictures": [{"product": "future_field", "bytes": size,
                 "path": f"png/d02-grid/future_field/2026-01-01/f{hour:03}.png"}
                 for hour, size in [(0, 100), (12, 2_000_000)]]}
    (tmp_path / "artifact-audit.json").write_text(json.dumps(audit))
    record = collect(tmp_path, {"future_field": ["direct", 0]})
    assert len(record["pictures"]) == 2
    assert all(row[2] == [76, 76] for row in record["pictures"])
    assert not next(iter(record["runs"].values()))["complete_inventory"]
    assert build_table(record)["products"]["future_field"][3] == 2_000_000


def test_fallback_is_the_largest_recorded_picture_with_headroom_once():
    record = json.loads(RECORD.read_text())
    table = disk_budget._picture_table()
    largest = max(row[4] for row in record["pictures"])
    assert table["fallback_bytes"] == largest
    assert disk_budget.projected_picture_bytes(36, 36, 0, 3600, "future_product") == math.ceil(
        largest * table["headroom"])


def test_inflated_prices_fail_the_independent_run_ceiling(monkeypatch):
    from test_downscale_checkpoint_disk import test_the_picture_price_covers_every_frame_it_was_read_from

    original = disk_budget.projected_picture_bytes
    monkeypatch.setattr(disk_budget, "projected_picture_bytes", lambda *a, **kw: 2 * original(*a, **kw))
    with pytest.raises(AssertionError):
        test_the_picture_price_covers_every_frame_it_was_read_from()


def test_no_catalog_refusal_still_offers_fewer_products():
    from gpuwm.offline_child_run import child_disk_remedy

    message = child_disk_remedy({"picture_bytes": 1, "pictures_per_frame": None})
    assert "draw fewer products (--render-products; none draws nothing)" in message
    assert "render-products" not in child_disk_remedy({"picture_bytes": 0})


def test_complete_inventory_fixture_has_matching_per_picture_records():
    record = json.loads(RECORD.read_text())
    fixtures = json.loads((ROOT / "tests/fixtures/picture-disk-measurements.json").read_text())
    for row in fixtures:
        source = row["source_sha256"][:16]
        assert record["runs"][source]["complete_inventory"]
        pictures = [p for p in record["pictures"] if p[0] == source]
        assert len(pictures) == row["pictures"]
        assert sum(p[4] for p in pictures) == row["bytes"]
