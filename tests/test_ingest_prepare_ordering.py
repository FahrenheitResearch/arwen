"""The prepare loop builds its START time last, and that changes no number.

THE DEFECT.  Every prepared-cache adapter walked its forcing times in
time order.  The start time is the first one built and the last one used
-- the prepared cache, the wrfinput export and the surface analysis are
all written from it once the boundaries are complete -- so it sat on the
device for the whole loop while each later time was interpolated and
initialized underneath it.  Two complete full-domain analyses and two
complete states therefore coexisted at the peak with only one of them
being worked on.  Priced by ``estimate_ingest`` at 800x800x49, mp=10,
three GFS times, that second resident time is 14.67 GiB of device
residency against 7.66, and a peak envelope of 23.92 GiB against 15.86:
the difference between preparing that domain on a 16 GiB card and dying
in preprocessing after the whole forcing chain had already been fetched.

THE FIX is an order and nothing else.  ``start_last_forcing_order`` puts
the start time last; each earlier time hands
:class:`StateBoundaryFrames` its four perimeter strips (host memory,
O(perimeter)) against its own POSITION and is released before the next is
built.  Which makes the acceptance test for this change a bit-identity
test, not a memory test: a pure reordering that moves one boundary
number is not a reordering, and a prepared cache is not allowed to notice
that its inputs were built in a different sequence.

Every test here is CPU-only and needs no device.
"""

from datetime import datetime, timedelta
import json

import numpy as np
import pytest

from gpuwm.ingest.lateral_bc import (
    StateBoundaryFrames,
    build_lateral_boundaries,
    start_last_forcing_order,
)


class _FakeState:
    """A state whose coupled snapshot is already the host arrays.

    ``add_state`` is the call the adapters make, and it reaches the
    device through ``domain_boundary_snapshot``.  Patching that one seam
    (``_host_snapshot`` below) keeps these tests on the CPU while still
    exercising the entry point the adapters actually use.
    """

    def __init__(self, snapshot):
        self.snapshot = snapshot


@pytest.fixture(autouse=True)
def _host_snapshot(monkeypatch):
    import gpuwm.ingest.lateral_bc as lateral_bc

    monkeypatch.setattr(
        lateral_bc, "domain_boundary_snapshot",
        lambda state: state.snapshot)


def _snapshots(count=5, seed=20260812, ny=40, nx=50):
    """``count`` distinct coupled full-domain snapshots.

    Distinct per time and asymmetric in every direction, so a frame
    written to the wrong position, a side taken from the wrong end, or a
    tendency differenced across the wrong pair cannot cancel out.
    """
    rng = np.random.default_rng(seed)
    shapes = {
        "u": (6, ny, nx + 1), "v": (6, ny + 1, nx),
        "theta": (6, ny, nx), "phi": (7, ny, nx),
        "mu": (1, ny, nx), "qv": (6, ny, nx),
    }
    return [
        {name: rng.standard_normal(shape).astype(np.float32)
         for name, shape in shapes.items()}
        for _ in range(count)
    ]


def _assert_same_bytes(actual, expected):
    assert len(actual.intervals) == len(expected.intervals)
    assert actual.spec_bdy_width == expected.spec_bdy_width
    assert actual.spec_zone == expected.spec_zone
    assert actual.relax_zone == expected.relax_zone
    for index, (want, got) in enumerate(
            zip(expected.intervals, actual.intervals)):
        assert got.start_seconds == want.start_seconds, index
        assert got.end_seconds == want.end_seconds, index
        assert set(got.fields) == set(want.fields), index
        for name in want.fields:
            for side in ("west", "east", "south", "north"):
                a = getattr(got.fields[name], side)
                b = getattr(want.fields[name], side)
                assert a.value.dtype == b.value.dtype
                assert a.value.shape == b.value.shape
                assert a.value.tobytes() == b.value.tobytes(), (
                    f"interval {index} {name}/{side} value")
                assert a.tendency.tobytes() == b.tendency.tobytes(), (
                    f"interval {index} {name}/{side} tendency")


# ---------------------------------------------------------------------------
# The order itself
# ---------------------------------------------------------------------------


def test_start_last_order_visits_every_time_once_with_the_start_last():
    """A permutation, with position 0 at the end.  Nothing subtler."""
    for count in range(2, 12):
        order = start_last_forcing_order(count)
        assert sorted(order) == list(range(count)), count
        assert order[-1] == 0, count
        # The later times keep their own order, which is what lets a
        # caller reason about the sequence its receipts are printed in.
        assert list(order[:-1]) == list(range(1, count)), count


def test_start_last_order_refuses_a_sequence_it_cannot_difference():
    """One forcing time makes no interval, so it is refused by name and
    not silently turned into a loop that leaves ``initial_result`` None
    and tracebacks on an attribute two hundred lines later."""
    for count in (1, 0, -3):
        with pytest.raises(ValueError, match="at least two forcing times"):
            start_last_forcing_order(count)


# ---------------------------------------------------------------------------
# The acceptance test: the reordering moves no byte
# ---------------------------------------------------------------------------


def test_start_last_frames_are_bit_identical_to_the_in_order_build():
    """THE ACCEPTANCE TEST for the reordering.

    Same snapshots, same times; one accumulator fed in time order and one
    fed in ``start_last_forcing_order``, each interval compared value and
    tendency, byte for byte.  The all-at-once builder is the third
    reference, so this also pins that neither ordering drifted from the
    original whole-state builder.
    """
    snapshots = _snapshots()
    start = datetime(2026, 8, 12, 0, 0, 0)
    times = [start + timedelta(hours=3 * n) for n in range(len(snapshots))]

    reference = build_lateral_boundaries(
        snapshots, times, spec_bdy_width=5, spec_zone=1, relax_zone=4)

    in_order = StateBoundaryFrames(spec_bdy_width=5, spec_zone=1,
                                   relax_zone=4)
    for snapshot in snapshots:
        in_order.add_snapshot(snapshot)

    start_last = StateBoundaryFrames(spec_bdy_width=5, spec_zone=1,
                                     relax_zone=4)
    order = start_last_forcing_order(len(snapshots))
    assert order != tuple(range(len(snapshots))), (
        "the orders must actually differ or this test proves nothing")
    for index in order:
        start_last.add_snapshot(snapshots[index], index=index)

    _assert_same_bytes(in_order.build(times), reference)
    _assert_same_bytes(start_last.build(times), reference)


def test_any_arrival_order_at_all_lands_on_the_same_bytes():
    """Position, not arrival, is what the intervals are built from.

    Reversed and shuffled arrivals are included deliberately: the
    start-last order is one permutation, and a test that only exercises
    that one cannot tell "sorted by position" from "happened to work".
    """
    snapshots = _snapshots(count=4)
    times = [0.0, 10800.0, 21600.0, 32400.0]
    reference = build_lateral_boundaries(
        snapshots, times, spec_bdy_width=5, spec_zone=1, relax_zone=4)

    for arrival in ((0, 1, 2, 3), (3, 2, 1, 0), (2, 0, 3, 1),
                    start_last_forcing_order(4)):
        frames = StateBoundaryFrames(spec_bdy_width=5, spec_zone=1,
                                     relax_zone=4)
        for index in arrival:
            frames.add_state(_FakeState(snapshots[index]), index=index)
        _assert_same_bytes(frames.build(times), reference)


def test_the_position_keying_is_what_makes_the_reorder_safe():
    """The counter-case, so the ``index=`` is not decorative.

    Reordering the loop is only free BECAUSE the frames are addressed by
    position.  Feed the same start-last sequence to the accumulator
    without saying where each frame belongs and it takes them in arrival
    order, which pairs the last forcing time with the first and produces
    boundary tendencies across intervals that never existed.  This test
    exists so that dropping the ``index=`` from an adapter cannot look
    like a harmless simplification.
    """
    snapshots = _snapshots(count=4)
    times = [0.0, 10800.0, 21600.0, 32400.0]
    reference = build_lateral_boundaries(
        snapshots, times, spec_bdy_width=5, spec_zone=1, relax_zone=4)

    unkeyed = StateBoundaryFrames(spec_bdy_width=5, spec_zone=1,
                                  relax_zone=4)
    for index in start_last_forcing_order(len(snapshots)):
        unkeyed.add_snapshot(snapshots[index])
    wrong = unkeyed.build(times)

    first_reference = reference.intervals[0].fields["theta"].west
    first_wrong = wrong.intervals[0].fields["theta"].west
    assert first_wrong.value.tobytes() != first_reference.value.tobytes()
    assert first_wrong.tendency.tobytes() != first_reference.tendency.tobytes()
    # And it is wrong in the specific way arrival order predicts: the
    # first interval starts from forcing time 1, not from the start time.
    expected_first = build_lateral_boundaries(
        [snapshots[1], snapshots[2]], times[:2],
        spec_bdy_width=5, spec_zone=1, relax_zone=4)
    assert (first_wrong.value.tobytes()
            == expected_first.intervals[0].fields["theta"].west.value.tobytes())


def test_the_written_prepared_cache_does_not_notice_the_reorder(tmp_path):
    """The deliverable, not just the intermediate.

    Comparing boundary arrays proves the accumulator; what a user gets is
    a prepared cache directory, and that is what must not move.  Two
    caches are written from one state and one met, differing only in
    whether their boundary tables were accumulated in time order or
    start-last, and compared on the container's OWN content hash -- the
    same digest a restore checks itself against.
    """
    from gpuwm.ingest.prepared_cache import write_prepared_cache
    from test_prepared_cache import _fixture

    initial, met, _fixture_boundaries = _fixture()
    snapshots = _snapshots(count=4)
    times = [0.0, 10800.0, 21600.0, 32400.0]

    def accumulate(order):
        frames = StateBoundaryFrames(spec_bdy_width=5, spec_zone=1,
                                     relax_zone=4)
        for index in order:
            frames.add_snapshot(snapshots[index], index=index)
        return frames.build(times)

    identity = {"source": "prepare-ordering-fixture"}
    receipts = {}
    for label, order in (("in-order", tuple(range(len(snapshots)))),
                         ("start-last",
                          start_last_forcing_order(len(snapshots)))):
        receipts[label] = write_prepared_cache(
            tmp_path / label, identity=identity, initial_result=initial,
            met=met, boundaries=accumulate(order))

    assert receipts["in-order"]["status"] == "BUILT"
    assert (receipts["start-last"]["content_sha256"]
            == receipts["in-order"]["content_sha256"])
    assert (receipts["start-last"]["payload_bytes"]
            == receipts["in-order"]["payload_bytes"])

    # And the bytes themselves, not only the digest of them: a hash that
    # agreed while the files differed would be a defect in the hash, and
    # this test would then be measuring the wrong instrument.  Every
    # payload file compares byte for byte.  ``header.json`` carries one
    # field that is not content -- ``created_utc``, the wall clock at
    # publication -- so it is compared as parsed JSON with that key
    # lifted out, and the set of differing keys is asserted to be exactly
    # {created_utc} so that a second non-content field cannot arrive here
    # unnoticed.
    def tree(root):
        return {path.relative_to(root).as_posix(): path.read_bytes()
                for path in sorted(root.rglob("*")) if path.is_file()}

    first = tree(tmp_path / "in-order")
    second = tree(tmp_path / "start-last")
    assert sorted(first) == sorted(second)
    for name in first:
        if name == "header.json":
            continue
        assert first[name] == second[name], name

    headers = [json.loads(tree_bytes["header.json"].decode("utf-8"))
               for tree_bytes in (first, second)]
    assert {key for key in set(headers[0]) | set(headers[1])
            if headers[0].get(key) != headers[1].get(key)} == {"created_utc"}


# ---------------------------------------------------------------------------
# The guards the position-keyed accumulator needs
# ---------------------------------------------------------------------------


def test_a_hole_in_the_positions_is_named_not_differenced_across():
    """The count can match while the sequence has a hole.

    An out-of-order caller that repeats one position and skips another
    arrives at ``build`` with the right number of frames.  Differencing
    across the hole would produce a tendency between the wrong pair of
    times -- a wrong boundary table, silently -- so the missing positions
    are named instead.
    """
    snapshots = _snapshots(count=4)
    times = [0.0, 10800.0, 21600.0, 32400.0]
    frames = StateBoundaryFrames(spec_bdy_width=5)
    for index in (0, 1, 3):
        frames.add_snapshot(snapshots[index], index=index)
    frames.add_snapshot(snapshots[2], index=7)
    assert len(frames) == len(times)
    with pytest.raises(ValueError, match=r"indices \[2\] were never added"):
        frames.build(times)


def test_a_repeated_position_is_refused_at_the_add():
    snapshots = _snapshots(count=3)
    frames = StateBoundaryFrames(spec_bdy_width=5)
    frames.add_snapshot(snapshots[0], index=1)
    with pytest.raises(ValueError, match="index 1 was added twice"):
        frames.add_snapshot(snapshots[1], index=1)
    with pytest.raises(ValueError, match="index -1 is negative"):
        frames.add_snapshot(snapshots[2], index=-1)


def test_arrival_order_and_explicit_positions_cannot_be_mixed():
    """Half a sequence counted by arrival and half addressed by position
    would overwrite one frame or leave a hole, and the first symptom
    would be a boundary tendency across the wrong pair of times."""
    snapshots = _snapshots(count=3)

    appended = StateBoundaryFrames(spec_bdy_width=5)
    appended.add_snapshot(snapshots[0])
    with pytest.raises(ValueError, match="all be added with index="):
        appended.add_snapshot(snapshots[1], index=1)

    indexed = StateBoundaryFrames(spec_bdy_width=5)
    indexed.add_snapshot(snapshots[0], index=0)
    with pytest.raises(ValueError, match="all be added with index="):
        indexed.add_snapshot(snapshots[1])


def test_the_position_keyed_path_keeps_every_geometry_guard():
    """The reordering must not have bought its memory with a guard.

    ``add_snapshot`` is where the too-small-domain and inconsistent-
    inventory refusals live; going through it with ``index=`` has to fire
    exactly the same ones.
    """
    snapshots = _snapshots(count=2)

    frames = StateBoundaryFrames(spec_bdy_width=5)
    frames.add_snapshot(snapshots[0], index=0)
    with pytest.raises(ValueError, match="inventories differ"):
        frames.add_snapshot({"u": snapshots[1]["u"]}, index=1)

    narrow = {"theta": np.zeros((3, 8, 6), dtype=np.float32)}
    with pytest.raises(ValueError, match="domain is too small"):
        StateBoundaryFrames(spec_bdy_width=5).add_snapshot(narrow, index=0)

    with pytest.raises(ValueError, match="field inventory is empty"):
        StateBoundaryFrames(spec_bdy_width=5).add_snapshot({}, index=0)


def test_retained_bytes_still_counts_only_the_perimeter():
    """The residency claim, on the position-keyed path.

    A domain wide enough for the claim to mean something: the perimeter
    fraction is O(width/ny + width/nx), so on the 40x50 grid the rest of
    this file uses it would be 45% and would prove nothing about a real
    root.
    """
    width = 5
    snapshots = _snapshots(count=4, ny=120, nx=150)
    frames = StateBoundaryFrames(spec_bdy_width=width)
    for index in start_last_forcing_order(len(snapshots)):
        frames.add_snapshot(snapshots[index], index=index)

    def perimeter_elements(shape):
        nz, ny, nx = shape
        return 2 * nz * ny * width + 2 * nz * nx * width

    expected = len(snapshots) * 8 * sum(
        perimeter_elements(np.asarray(value).shape)
        for value in snapshots[0].values())
    one_time_float64 = sum(
        np.asarray(value).size * 8 for value in snapshots[0].values())

    assert len(frames) == len(snapshots)
    assert frames.retained_bytes == expected
    assert frames.retained_bytes / len(frames) < 0.2 * one_time_float64


# ---------------------------------------------------------------------------
# Every adapter that owns a copy of the loop
# ---------------------------------------------------------------------------


def test_every_prepare_adapter_builds_its_start_time_last():
    """The loop is duplicated three times, so the fix has to be.

    Source-level because reaching these loops for real needs a decoded
    GRIB chain, a geog root and a device; what is pinned is that each of
    the three adapters drives its forcing loop from
    ``start_last_forcing_order`` and hands the accumulator an explicit
    ``index=``.  A fourth adapter growing an in-order copy of the loop is
    exactly the event this fails on -- ``enumerate(snapshots)`` is the
    shape that had the defect.
    """
    import inspect

    from gpuwm import era5_direct, gfs_direct, mapped_direct

    for module in (gfs_direct, era5_direct, mapped_direct):
        source = inspect.getsource(module)
        assert "for index in start_last_forcing_order(len(snapshots)):" in \
            source, f"{module.__name__} does not build the start time last"
        assert "forcing.add_state(initialized.state, index=index)" in source, (
            f"{module.__name__} adds frames by arrival order, so the "
            "reordered loop would write the intervals out of sequence")
        assert "for index, source in enumerate(snapshots)" not in source, (
            f"{module.__name__} still has an in-order forcing loop")
