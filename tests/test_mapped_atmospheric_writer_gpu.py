"""Native retained payloads reach the unchanged CUDA interpolation plans."""
from dataclasses import replace

from conftest import requires_gpu
from test_mapped_atmospheric_writer import engine, pair, source, target
import test_mapped_frameset_streaming as bundle_fixture
from gpuwm.ingest.horiz import interpolate_era5_to_lambert

pytestmark = requires_gpu


def test_actual_native_writer_to_cuda_preserves_every_horizontal_field(pair, source):
    import cupy as cp
    full, small, _ = pair
    authority = source[0]
    full_bundle = replace(bundle_fixture._bundle(full.directory, authority), frames=full)
    small_bundle = replace(bundle_fixture._bundle(small.directory, authority), frames=small)
    expected = interpolate_era5_to_lambert(full_bundle.regular_snapshots()[0],
                                         target(), backend="cuda")
    actual = interpolate_era5_to_lambert(
        small_bundle.regular_snapshots().for_grids((target(),))[0], target(), backend="cuda")
    assert small._full_frames is None
    assert expected.fields.keys() == actual.fields.keys()
    assert expected.levels_hpa.tobytes() == actual.levels_hpa.tobytes()
    for name in expected.fields:
        assert cp.asnumpy(actual.fields[name]).tobytes() == cp.asnumpy(expected.fields[name]).tobytes(), name
