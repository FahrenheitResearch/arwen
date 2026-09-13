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


def _with_hydrometeors(frame, values_by_name):
    """Attach real planes to the fixture frame, header descriptors and all.

    A field the frame CARRIES is not an absent field, so its explicit-zero
    initialization policy is dropped along with it: the point of these
    cases is that a carried plane needs no policy at all.
    """

    fields = dict(frame.fields)
    descriptors = list(frame.header.fields)
    template = next(d for d in descriptors
                    if d.canonical_name == "specific_humidity")
    policies = dict(frame.header.initialization_policies)
    for name, values in values_by_name.items():
        fields[name] = replace(
            fields["specific_humidity"], name=name, values=values)
        descriptors.append(replace(
            template, canonical_name=name, source_field=name,
            data_reference=f"fixture:{name}",
            shape=tuple(int(size) for size in values.shape)))
        policies.pop(name, None)
    return replace(frame, fields=fields, header=replace(
        frame.header, fields=tuple(descriptors),
        initialization_policies=policies))


def test_a_present_hydrometeor_is_packed_not_refused(frame):
    """Rewritten from test_zero_policy_never_overwrites_a_present_prognostic.

    That test pinned the ABI-gap refusal ("cannot inject mapped prognostic
    fields") that this change retires.  What it was really protecting is
    that a declared zero policy never overwrites a decoded plane, so the
    same frame now asserts the stronger thing: the decoded plane is PACKED
    bit for bit, and the four the frame does not carry are still the
    policy's exact zeros.
    """

    shape = frame.fields["specific_humidity"].values.shape
    qc = (np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
          + 1.0) * 1e-4
    changed = _with_hydrometeors(frame, {"cloud_water_mixing_ratio": qc})
    snapshot = mapped_frames_to_regular_snapshots(
        (changed,), initialize_absent_hydrometeors=True)[0]
    assert snapshot.fields["QC"].tobytes() == qc.tobytes()
    for absent in ("QR", "QI", "QS", "QG"):
        assert np.array_equal(snapshot.fields[absent],
                              np.zeros_like(snapshot.fields["PRES"]))


def test_a_frame_carrying_all_five_needs_no_initialization_policy(frame):
    shape = frame.fields["specific_humidity"].values.shape
    planes = {
        name: np.full(shape, value, dtype=np.float64)
        for name, value in (
            ("cloud_water_mixing_ratio", 1e-4),
            ("rain_water_mixing_ratio", 2e-4),
            ("cloud_ice_mixing_ratio", 3e-4),
            ("snow_mixing_ratio", 4e-4),
            ("graupel_or_hail_mixing_ratio", 5e-4))}
    changed = _with_hydrometeors(frame, planes)
    # None of the five carries an initialization policy any more, and the
    # pack runs with the flag off as well as on.
    assert not (set(planes)
                & set(changed.header.initialization_policies))
    for snapshot in (
            mapped_frames_to_regular_snapshots((changed,))[0],
            mapped_frames_to_regular_snapshots(
                (changed,), initialize_absent_hydrometeors=True)[0]):
        for legacy, canonical in (("QC", "cloud_water_mixing_ratio"),
                                  ("QR", "rain_water_mixing_ratio"),
                                  ("QI", "cloud_ice_mixing_ratio"),
                                  ("QS", "snow_mixing_ratio"),
                                  ("QG", "graupel_or_hail_mixing_ratio")):
            assert snapshot.fields[legacy].tobytes() \
                == planes[canonical].tobytes()
            assert np.all(snapshot.fields[legacy] > 0.0)


def test_a_carried_vertical_velocity_is_dropped_with_one_notice(
        frame, monkeypatch):
    """A source carrying more than the join consumes is a drop, not a refusal.

    ONE notice per run, not one per door: a single preparation reaches
    warn_regular_join_drops from both plan-review call sites in
    gpuwm.mapped_direct.prepare_mapped_wrf and again from the frame join,
    and every one of them still gets the dropped list back for its
    receipt.
    """

    from gpuwm import explain, mapped_source

    monkeypatch.setattr(mapped_source, "_JOIN_DROPS_WARNED", set())
    shape = frame.fields["specific_humidity"].values.shape
    changed = _with_hydrometeors(
        frame, {"vertical_velocity": np.full(shape, -0.1, dtype=np.float64)})
    seen = []
    explain.add_warning_observer(seen.append)
    try:
        review = mapped_source.warn_regular_join_drops(
            ("vertical_velocity",), subject="this mapping")
        again = mapped_source.warn_regular_join_drops(
            ("vertical_velocity",), subject="this mapping")
        snapshot = mapped_frames_to_regular_snapshots((changed,))[0]
    finally:
        explain.remove_warning_observer(seen.append)
    assert "W" not in snapshot.fields
    assert snapshot.fields["T"].shape == shape
    # Every door still learns what was dropped; only the line is once.
    assert review == again == ("vertical_velocity",)
    named = [record for record in seen
             if "vertical_velocity" in record["action"]]
    assert len(named) == 1
    assert "no consumer" in named[0]["action"]


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
