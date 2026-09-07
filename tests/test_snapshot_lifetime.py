"""An owned preparation snapshot need not retain a second decoded atmosphere."""
from dataclasses import replace
import gc
import weakref

import numpy as np
import pytest

from gpuwm.ingest.water_overlay import apply_water_temperature_overlay, load_bound_water_overlay
import test_mapped_frameset_streaming as frames_fixture
from test_water_overlay import make_snapshot, write_fine_overlay


def test_surface_replacement_copies_only_updates_and_preserves_projection():
    original = make_snapshot()
    original = replace(original, projection={"family": "lambert", "parameters": {"axis_unit_m": 1000.}})
    update = np.full_like(original.fields["SKINTEMP"], 299.)
    result = original.with_fields({"SKINTEMP": update})
    assert result.projection == original.projection
    assert result.valid_time == original.valid_time
    assert result.fields["TT"] is original.fields["TT"]
    assert result.fields["LANDSEA"] is original.fields["LANDSEA"]
    assert not np.shares_memory(result.fields["SKINTEMP"], update)
    update[:] = -1.
    np.testing.assert_array_equal(result.fields["SKINTEMP"], 299.)
    assert all(not array.flags.writeable for array in result.fields.values())
    with pytest.raises(ValueError):
        result.fields["TT"][0, 0, 0] = 0.
    with pytest.raises(TypeError):
        result.fields["NEW"] = update
    assert original.with_fields({}) is original


@pytest.mark.parametrize("bad,error", [
    (np.ones((2, 2), dtype=np.float64), ValueError),
    (np.ones((6, 7), dtype=np.float32), TypeError),
])
def test_surface_replacement_keeps_constructor_validation(bad, error):
    original = make_snapshot()
    before = original.fields["SKINTEMP"].tobytes()
    with pytest.raises(error):
        original.with_fields({"SKINTEMP": bad})
    assert original.fields["SKINTEMP"].tobytes() == before


def test_real_water_overlay_retains_unmodified_atmosphere(tmp_path):
    original = make_snapshot()
    overlay, _ = load_bound_water_overlay(write_fine_overlay(tmp_path / "water.nc"))
    result, receipt = apply_water_temperature_overlay(original, overlay)
    assert receipt["replaced_cells"] > 0
    assert result.fields["TT"] is original.fields["TT"]
    assert result.fields["LANDSEA"] is original.fields["LANDSEA"]
    assert not np.shares_memory(result.fields["SKINTEMP"], original.fields["SKINTEMP"])


def test_packing_releases_decoded_frame_but_keeps_snapshot_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(frames_fixture, "_NY", 4)
    monkeypatch.setattr(frames_fixture, "_NX", 5)
    directory = frames_fixture._write_fixture_frameset(tmp_path / "frames", 2)
    authority = tmp_path / "authority"
    authority.write_text("fixture")
    bundle = frames_fixture._bundle(directory, authority)
    snapshots = bundle.regular_snapshots()
    held = bundle.frames[0]
    reference = weakref.ref(held)
    wanted = held.fields["air_temperature"].values.tobytes()
    result = snapshots[0]
    # External consumers retain their own frame; dropping the reader's
    # cache does not revoke their arrays or scratch files.
    assert held.fields["air_temperature"].values.tobytes() == wanted
    assert result.fields["T"].tobytes() == wanted
    assert snapshots[0] is result
    del held
    gc.collect()
    assert reference() is None
    assert (directory / "frames.f64").is_file()
    assert snapshots[1].valid_time > result.valid_time
    reread = bundle.frames[0]
    assert reread.fields["air_temperature"].values.tobytes() == wanted


def test_frame_release_is_by_identity_and_preserves_other_cached_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(frames_fixture, "_NY", 3)
    monkeypatch.setattr(frames_fixture, "_NX", 4)
    directory = frames_fixture._write_fixture_frameset(tmp_path / "frames", 2)
    frames = frames_fixture.engine_bridge.open_frameset(directory)
    first, second = frames[0], frames[1]
    frames.release_frame(first)
    assert frames[1] is second
    window = frames[1:]
    window.release_frame(second)
    assert frames[1] is not second
    np.testing.assert_array_equal(frames[1].fields["air_temperature"].values,
                                  second.fields["air_temperature"].values)
