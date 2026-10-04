"""Exact native diagnostic replay and real Rust ensemble map artifacts."""
import hashlib
from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.ensemble.batch_products import FieldProducts, prepare_product_frame, write_product_frame
from gpuwm.ensemble.batch_product_output import NativeDiagnosticSpool

pytestmark = [pytest.mark.gpu, requires_gpu]


def _renderer():
    from gpuwm import rustwx
    renderer = Path(os.environ["GPUWM_ENSEMBLE_RENDERER"]) if "GPUWM_ENSEMBLE_RENDERER" in os.environ else rustwx.find_renderer()
    if renderer is None:
        pytest.fail("the product identity gate requires the built native ensemble renderer")
    return renderer


@pytest.mark.parametrize("members", [1, 4, 10, 65])
def test_replayed_packs_match_all_resident_file_words_and_preserve_members(tmp_path, members):
    import cupy as cp
    renderer = _renderer()
    coords = np.zeros((7, 9), np.float32)
    coords[:] = np.linspace(35, 35.1, 7, dtype=np.float32)[:, None]
    longitude = np.broadcast_to(np.linspace(-99, -98.9, 9, dtype=np.float32), coords.shape).copy()
    words = np.random.default_rng(7281).uniform(0, 40, (members, 7, 9)).astype(np.float32)
    words[0, 0, 0] = np.float32(-0.0)
    if members > 1:
        words[1, 2, 3] = np.asarray([0x7fc01234], np.uint32).view(np.float32)[0]
    values = cp.asarray(words)
    before = values.get().tobytes()
    requests = (FieldProducts("wind10", "m s-1", (10, 25), paintball=True,
                              spaghetti=True, postage_stamp=True),)
    all_resident = prepare_product_frame({"wind10": values}, requests, available_bytes=1 << 28)
    all_resident()
    control = tmp_path / "control.nc"
    all_resident.requests = tuple(replace(r, postage_stamp=False) for r in requests)
    write_product_frame(control, all_resident, valid_time="2026-10-02_00:00:00",
                        latitude=coords, longitude=longitude)
    spool = NativeDiagnosticSpool(tmp_path / "packed", members=members, requests=requests,
        latitude=coords, longitude=longitude, renderer=renderer, tile_rows=3)
    # Packs may arrive from any card in any order. Each diagnostic is placed
    # into its stable member slot before the canonical two-pass reduction.
    groups = [list(range(start, min(start + 3, members))) for start in range(0, members, 3)]
    for group in reversed(groups):
        group.reverse()
        packed = cp.asarray(words[group])
        done = spool.submit({"wind10": packed}, member_ids=group, valid_time="2026-10-02_00:00:00")
        assert done == (group == groups[0])
    row = spool.finish("2026-10-02_00:00:00", available_bytes=1 << 28, render_products=("prob",))
    product = spool.root / row["products"][0]
    assert hashlib.sha256(control.read_bytes()).digest() == hashlib.sha256(product.read_bytes()).digest()
    assert values.get().tobytes() == before
    assert len(row["maps"]) == 2 and all((spool.root / path).is_file() for path in row["maps"])
    manifest = json.loads(spool.manifest_path.read_text())
    assert manifest["frames"][0]["status"] == "complete"
    assert manifest["frames"][0]["members_received"] == list(range(members))
    assert manifest["member_files"] == []
    assert not list(spool.root.rglob("wrfout*"))
    assert not list((spool.root / ".ensemble-diagnostics").rglob("*.nc"))
    assert sum(item["bytes"] for item in manifest["deleted_scratch"]) > 0


def test_rust_statistics_paintball_and_postage_are_real_finished_pngs(tmp_path):
    import cupy as cp
    from PIL import Image
    renderer = _renderer()
    latitude = np.broadcast_to(np.linspace(35, 35.1, 5, dtype=np.float32)[:, None], (5, 7)).copy()
    longitude = np.broadcast_to(np.linspace(-99, -98.9, 7, dtype=np.float32), (5, 7)).copy()
    values = cp.asarray(np.arange(4 * 5 * 7, dtype=np.float32).reshape(4, 5, 7) * np.float32(0.2) + np.float32(280))
    requests = (FieldProducts("temperature2", "K", (293.15,), paintball=True, postage_stamp=True),)
    spool = NativeDiagnosticSpool(tmp_path, members=4, requests=requests,
        latitude=latitude, longitude=longitude, renderer=renderer, tile_rows=3)
    assert spool.submit({"temperature2": values}, member_ids=(0, 1, 2, 3), valid_time="2026-10-02_01:00:00")
    row = spool.finish("2026-10-02_01:00:00", available_bytes=1 << 26)
    assert len(row["maps"]) == 7
    products = [path.split("/")[2] for path in row["maps"]]
    assert any("postage" in product for product in products)
    for relative in row["maps"]:
        with Image.open(spool.root / relative) as picture:
            picture.verify()
    stamps = [spool.root / path for path in row["maps"] if "postage" in path]
    with Image.open(stamps[0]) as picture:
        assert picture.size == (1920, 360)


def test_ordinary_collector_uses_live_streamed_frame_and_exact_clock_windows(tmp_path):
    import cupy as cp
    from datetime import datetime, timedelta
    from types import SimpleNamespace
    from gpuwm.ensemble.batch_product_output import HeadlineDiagnosticCollector
    from gpuwm import netcdf_bridge
    renderer = _renderer()
    start = datetime(2026, 10, 2)
    coords = np.broadcast_to(np.linspace(35, 35.1, 3, dtype=np.float32)[:, None], (3, 5)).copy()
    longitude = np.broadcast_to(np.linspace(-99, -98.9, 5, dtype=np.float32), (3, 5)).copy()
    requests = tuple(FieldProducts(name, units, (threshold,)) for name, units, threshold in (
        ("wind10", "m s-1", 5), ("temperature2", "K", 293.15), ("dewpoint2", "K", 290),
        ("humidity2", "%", 50), ("rain_total", "mm", 1), ("qpf_1h", "mm", 1),
        ("qpf_3h", "mm", 3), ("qpf_6h", "mm", 6), ("refl", "dBZ", 40), ("gust", "m s-1", 10)))
    collector = HeadlineDiagnosticCollector(tmp_path, members=2, renderer=renderer, start_time=start,
        requests=requests, tile_rows=3, available_bytes=1 << 27, render_products=("prob",))
    class StaleState:
        @property
        def physics(self):
            raise AssertionError("a streamed collector must never read the stale resident state")
    for member in range(2):
        for hour in (0, 1, 3, 6):
            data = {"U10": np.full((3, 5), 3, np.float32), "V10": np.full((3, 5), 4, np.float32),
                "T2": np.full((3, 5), 300 + member, np.float32), "Q2": np.full((3, 5), 0.012, np.float32),
                "PSFC": np.full((3, 5), 100000, np.float32),
                "RAINNC": np.full((3, 5), 50 + hour * (member + 1), np.float32),
                "RAINC": np.full((3, 5), 1, np.float32), "RAINSH": np.zeros((3, 5), np.float32)}
            if member == 0:
                device = {name: cp.asarray(value) for name, value in data.items()}
                state = SimpleNamespace(physics=SimpleNamespace(output_fields=lambda: device))
                streamed = None
            else:
                state = StaleState()
                streamed = SimpleNamespace(history_fields=lambda: data)
            row = collector(state=state, streamed=streamed, metadata={"XLAT": coords, "XLONG": longitude},
                refl_field=cp.full((2, 3, 5), 45, cp.float32), valid_time=start + timedelta(hours=hour),
                grid_id=1, episode=0, member_id=member)
            assert (row is not None) == (member == 1)
    spool = next(iter(collector.spools.values()))
    first = spool.frames["2026-10-02_00:00:00"]
    assert set(first["unavailable_fields"]) == {"qpf_1h", "qpf_3h", "qpf_6h", "gust"}
    final = spool.frames["2026-10-02_06:00:00"]
    # The 6 h window begins at a retained exact output. There is no 5 h
    # snapshot, so a 6 h timestamp cannot claim a complete 1 h window.
    assert set(final["unavailable_fields"]) == {"qpf_1h", "gust"}
    path = spool.root / final["products"][0]
    with netcdf_bridge.Dataset(path) as reader:
        for name, expected in (("wind10_mean", 5), ("rain_total_mean", 9),
                               ("qpf_3h_mean", 4.5), ("qpf_6h_mean", 9), ("refl_mean", 45)):
            decoded = np.asarray(reader.variables[name][:]).astype(np.float32)
            np.testing.assert_array_equal(decoded, np.float32(expected))
        assert "gust_mean" not in reader.variables
        assert "qpf_1h_mean" not in reader.variables


def test_packed_reflectivity_stride_has_identical_words_without_volume_copy(tmp_path, monkeypatch):
    import cupy as cp
    from datetime import datetime
    from types import SimpleNamespace
    from gpuwm.ensemble.batch_product_output import HeadlineDiagnosticCollector
    members, levels, ny, nx = 4, 5, 3, 7
    raw = np.random.default_rng(482).uniform(-20, 60, (levels, members, ny, nx)).astype(np.float32)
    raw[:, :, 0, 0] = np.nan
    raw[:, :, 0, 1] = -0.0
    packed = cp.asarray(raw)
    before = packed.get().tobytes()
    coords = np.full((ny, nx), 35, np.float32)
    metadata = {"XLAT": coords, "XLONG": np.full_like(coords, -99)}
    requests = (FieldProducts("refl", "dBZ", (40,)),)
    ordinary = HeadlineDiagnosticCollector(tmp_path / "ordinary", members=members,
        renderer=_renderer(), start_time=datetime(2026, 10, 2), requests=requests,
        render_products=("prob",), available_bytes=1 << 27)
    strided = HeadlineDiagnosticCollector(tmp_path / "strided", members=members,
        renderer=_renderer(), start_time=datetime(2026, 10, 2), requests=requests,
        render_products=("prob",), available_bytes=1 << 27)
    state = SimpleNamespace(physics=SimpleNamespace(output_fields=lambda: {}))
    contiguous = [cp.ascontiguousarray(packed[:, member]) for member in range(members)]
    for member, volume in enumerate(contiguous):
        ordinary(state=state, streamed=None, metadata=metadata, refl_field=volume,
                 valid_time=datetime(2026, 10, 2), grid_id=1, episode=0, member_id=member)
    copy = cp.ascontiguousarray
    def forbid_volume_copy(array, *args, **kw):
        assert array.shape != (levels, ny, nx), "packed member reflectivity must be borrowed"
        return copy(array, *args, **kw)
    monkeypatch.setattr(cp, "ascontiguousarray", forbid_volume_copy)
    for member in range(members):
        strided(state=state, streamed=None, metadata=metadata, refl_field=packed[:, member],
                valid_time=datetime(2026, 10, 2), grid_id=1, episode=0, member_id=member)
    left = next(iter(ordinary.spools.values()))
    right = next(iter(strided.spools.values()))
    for relative in left.frames["2026-10-02_00:00:00"]["products"]:
        assert (left.root / relative).read_bytes() == (right.root / relative).read_bytes()
    assert packed.get().tobytes() == before


def test_counter_baseline_before_delayed_history_and_same_tick_output(tmp_path):
    import cupy as cp
    from datetime import datetime, timedelta
    from types import SimpleNamespace
    from gpuwm import netcdf_bridge
    from gpuwm.ensemble.batch_product_output import HeadlineDiagnosticCollector
    start = datetime(2026, 10, 2)
    latitude = np.broadcast_to(np.linspace(35, 35.1, 3, dtype=np.float32)[:, None], (3, 5)).copy()
    longitude = np.broadcast_to(np.linspace(-99, -98.9, 5, dtype=np.float32), (3, 5)).copy()
    requests = (FieldProducts("rain_total", "mm", (1,)), FieldProducts("qpf_1h", "mm", (1,)))
    collector = HeadlineDiagnosticCollector(tmp_path, members=1, renderer=_renderer(),
        start_time=start, requests=requests, available_bytes=1 << 27, render_products=("prob",))
    rain = cp.full((3, 5), 50, cp.float32)
    before = rain.get().tobytes()
    arguments = dict(grid_id=1, episode=0, member_id=0, latitude=latitude, longitude=longitude)
    assert collector.capture_rain_counters({"RAINNC": rain}, valid_time=start, **arguments)
    assert collector.manifest_paths == ()
    assert rain.get().tobytes() == before
    rain.fill(54)
    valid = start + timedelta(hours=1)
    assert collector.capture_rain_counters({"RAINNC": rain}, valid_time=valid, **arguments)
    state = SimpleNamespace(physics=SimpleNamespace(output_fields=lambda: {"RAINNC": rain}))
    row = collector(state=state, streamed=None, metadata={"XLAT": latitude, "XLONG": longitude},
        refl_field=None, valid_time=valid, grid_id=1, episode=0, member_id=0)
    with netcdf_bridge.Dataset(tmp_path / row["products"][0]) as reader:
        for name in ("rain_total_mean", "qpf_1h_mean"):
            np.testing.assert_array_equal(np.asarray(reader.variables[name][:]).astype(np.float32), np.float32(4))
    assert rain.get().tobytes() == np.full((3, 5), 54, np.float32).tobytes()
    with pytest.raises(ValueError, match="duplicated"):
        collector(state=state, streamed=None, metadata={"XLAT": latitude, "XLONG": longitude},
            refl_field=None, valid_time=valid, grid_id=1, episode=0, member_id=0)


def test_off_scheme_zero_counter_has_no_surface_driver_or_extra_output(tmp_path):
    import cupy as cp
    from datetime import datetime, timedelta
    from types import SimpleNamespace
    from gpuwm import netcdf_bridge
    from gpuwm.ensemble.batch_product_output import HeadlineDiagnosticCollector
    start = datetime(2026, 10, 2)
    latitude = np.broadcast_to(np.linspace(35, 35.1, 3, dtype=np.float32)[:, None], (3, 5)).copy()
    longitude = np.broadcast_to(np.linspace(-99, -98.9, 5, dtype=np.float32), (3, 5)).copy()
    requests = (FieldProducts("rain_total", "mm", (1,)), FieldProducts("qpf_1h", "mm", (1,)))
    collector = HeadlineDiagnosticCollector(tmp_path, members=1, renderer=_renderer(),
        start_time=start, requests=requests, available_bytes=1 << 27, render_products=("prob",))
    arguments = dict(grid_id=1, episode=0, member_id=0, latitude=latitude, longitude=longitude,
                     absent_zero_fields=("RAINNC", "RAINC", "RAINSH"))
    assert collector.capture_rain_counters({}, valid_time=start, **arguments)
    valid = start + timedelta(hours=1)
    assert collector.capture_rain_counters({}, valid_time=valid, **arguments)
    assert collector.manifest_paths == ()
    row = collector(state=SimpleNamespace(physics=None), streamed=None,
        metadata={"XLAT": latitude, "XLONG": longitude}, refl_field=None, valid_time=valid,
        grid_id=1, episode=0, member_id=0)
    with netcdf_bridge.Dataset(tmp_path / row["products"][0]) as reader:
        np.testing.assert_array_equal(np.asarray(reader.variables["qpf_1h_mean"][:]).astype(np.float32), np.float32(0))
    assert next(iter(collector.spools.values())).frames[valid.strftime("%Y-%m-%d_%H:%M:%S")]["unavailable_fields"] == []
