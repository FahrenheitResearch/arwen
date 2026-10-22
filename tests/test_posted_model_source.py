"""Exercise the native reader against a real posted forecast, never a mock.

The optional artifact paths select a fetched public inventory and a built
library. Normal CPU runs without those artifacts skip this integration proof.
"""
import gc
import json
import os
from pathlib import Path

import numpy as np
import pytest

from gpuwm.verify.obs.posted_source import PostedGrib2Source

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "configs/rrfs_v1_like/posted_surface_fields.json"
INPUT = os.environ.get("GPUWM_POSTED_TEST_INPUTS")
LIBRARY = os.environ.get("GPUWM_POSTED_TEST_LIBRARY")
pytestmark = pytest.mark.skipif(not INPUT or not LIBRARY,
                               reason="real posted GRIB2 and a built native reader are required")


def source(table=TABLE, inputs=INPUT):
    return PostedGrib2Source(inputs, field_table=table, reader_library=LIBRARY)


def test_fields_and_projection_match_the_posted_artifact():
    reader = source()
    grid = reader.grid()
    times = reader.valid_times()
    assert times == tuple(sorted(times))
    assert len(times) >= 1
    locate = reader.station_locator()
    for y, x in [(0, 0), (grid.shape[0] // 2, grid.shape[1] // 2),
                 (grid.shape[0] - 1, grid.shape[1] - 1)]:
        p = locate("point", grid.latitude[y, x], grid.longitude[y, x])
        assert p.x == pytest.approx(x, abs=1e-7)
        assert p.y == pytest.approx(y, abs=1e-7)
    for time in times:
        for name in ("temperature_2m", "dewpoint_2m", "wind_speed_10m", "mslp"):
            field = reader.surface_field(time, name)
            assert field.shape == grid.shape
            assert np.all(np.isfinite(field))
            assert not field.flags.writeable
        x = reader.surface_field(time, "wind_component_x_10m")
        y = reader.surface_field(time, "wind_component_y_10m")
        assert np.allclose(reader.surface_field(time, "wind_speed_10m"),
                           np.hypot(x, y), rtol=2e-15, atol=0)
    assert reader.land_mask().dtype == np.bool_
    assert reader.record()["selected_records"]
    assert all(len(r["sha256"]) == 64 for r in reader.record()["selected_records"])


def test_a_returned_array_keeps_its_native_owner_alive():
    reader = source()
    field = reader.surface_field(reader.valid_times()[0], "temperature_2m")
    before = field.copy()
    del reader
    gc.collect()
    assert np.array_equal(field, before)


def test_a_wrong_local_parameter_authority_is_refused(tmp_path):
    table = json.loads(TABLE.read_text())
    table["fields"]["composite_reflectivity"]["selector"]["center"] = 255
    path = tmp_path / "wrong-authority.json"
    path.write_text(json.dumps(table))
    with pytest.raises(ValueError, match="no selected record"):
        source(path)


def test_ambiguous_forecast_inputs_are_refused(tmp_path):
    original = sorted(Path(INPUT).glob("*.grib2"))[0]
    os.link(original, tmp_path / "one.grib2")
    os.link(original, tmp_path / "two.grib2")
    with pytest.raises(ValueError, match="duplicate selected record"):
        source(inputs=tmp_path)


def test_no_interval_precipitation_is_silently_scored():
    reader = source()
    with pytest.raises(ValueError, match="accumulation window"):
        reader.precipitation_accumulation(reader.valid_times()[0])


def test_a_truncated_posted_artifact_is_refused(tmp_path):
    original = sorted(Path(INPUT).glob("*.grib2"))[0]
    (tmp_path / "truncated.grib2").write_bytes(original.read_bytes()[:-4])
    with pytest.raises(ValueError, match="truncated GRIB2 envelope"):
        source(inputs=tmp_path)
