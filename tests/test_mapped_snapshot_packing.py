"""Policy-owned zeros preserve values without a second immutable snapshot."""
from dataclasses import replace

import numpy as np
import pytest

from gpuwm.ingest.grib import Era5Snapshot
from gpuwm.mapped_source import mapped_frames_to_regular_snapshots
import test_mapped_frameset_streaming as fixture


@pytest.fixture
def frame(monkeypatch):
    monkeypatch.setattr(fixture, "_NY", 3)
    monkeypatch.setattr(fixture, "_NX", 4)
    return fixture._one_frame()


def _historical_pack(frame):
    ordinary = mapped_frames_to_regular_snapshots((frame,))[0]
    fields = dict(ordinary.fields)
    for name in ("QC", "QR", "QI", "QS", "QG"):
        fields[name] = np.zeros_like(fields["PRES"])
    return replace(ordinary, fields=fields)


def _assert_same_snapshot(actual, expected):
    assert actual.valid_time == expected.valid_time
    assert actual.projection == expected.projection
    for axis in ("latitude", "longitude", "levels_hpa"):
        assert getattr(actual, axis).tobytes() == getattr(expected, axis).tobytes()
    assert tuple(actual.fields) == tuple(expected.fields)
    for name, expected_values in expected.fields.items():
        values = actual.fields[name]
        assert values.shape == expected_values.shape
        assert values.tobytes() == expected_values.tobytes()
        assert not values.flags.writeable
        assert values.flags.owndata
        assert values.flags.c_contiguous


def test_composed_pack_matches_historical_owned_bytes_once(frame, tmp_path, monkeypatch):
    expected = _historical_pack(frame)
    constructors = []
    original = Era5Snapshot.__post_init__
    def recorded(self):
        constructors.append(self.valid_time)
        return original(self)
    monkeypatch.setattr(Era5Snapshot, "__post_init__", recorded)
    authority = tmp_path / "authority"
    authority.write_text("fixture")
    actual = fixture._bundle_from_frames((frame,), authority).regular_snapshots()[0]
    assert constructors == [frame.valid_time]
    _assert_same_snapshot(actual, expected)
    for a, b in (("QC", "QR"), ("QI", "QS"), ("QG", "PRES")):
        assert not np.shares_memory(actual.fields[a], actual.fields[b])
    source = frame.fields["air_temperature"].values
    assert not np.shares_memory(actual.fields["T"], source)
    source.setflags(write=True)
    source[0, 0, 0] += 1.0
    assert actual.fields["T"][0, 0, 0] == expected.fields["T"][0, 0, 0]


@pytest.mark.parametrize("canonical", (
    "cloud_water_mixing_ratio", "rain_water_mixing_ratio",
    "cloud_ice_mixing_ratio", "snow_mixing_ratio", "graupel_or_hail_mixing_ratio",
))
@pytest.mark.parametrize("policy", (None, "unsupported"))
def test_zero_pack_requires_each_explicit_policy(frame, canonical, policy):
    policies = dict(frame.header.initialization_policies)
    if policy is None:
        del policies[canonical]
    else:
        policies[canonical] = policy
    # A missing declaration fails at canonical-frame admission already;
    # an unsupported nonempty declaration reaches the adapter's policy gate.
    with pytest.raises(ValueError, match="explicit.*policy|explicit-zero policy"):
        changed = replace(frame, header=replace(
            frame.header, initialization_policies=policies))
        mapped_frames_to_regular_snapshots(
            (changed,), initialize_absent_hydrometeors=True)
    # Ordinary conversion still has no implicit hydrometeor initialization.
    assert "QC" not in mapped_frames_to_regular_snapshots((frame,))[0].fields


def test_zero_policy_never_overwrites_a_present_prognostic(frame):
    fields = dict(frame.fields)
    fields["cloud_water_mixing_ratio"] = replace(
        fields["specific_humidity"], name="cloud_water_mixing_ratio")
    frame = replace(frame, fields=fields)
    with pytest.raises(ValueError, match="cannot inject mapped prognostic fields"):
        mapped_frames_to_regular_snapshots((frame,), initialize_absent_hydrometeors=True)


def test_real_netcdf_decoding_reaches_identical_regular_pack(tmp_path):
    import test_mapped_source as nc
    from gpuwm.mapped_source import decode_mapped_source
    mapping = tmp_path / "mapping.json"
    source = tmp_path / "source.nc"
    nc._write_mapping(mapping, nc._mapping())
    nc._write_source(source)
    for frame in decode_mapped_source(mapping, [source]):
        actual = mapped_frames_to_regular_snapshots(
            (frame,), initialize_absent_hydrometeors=True)[0]
        _assert_same_snapshot(actual, _historical_pack(frame))
