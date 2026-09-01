"""The nested-domain path must stay one valid time deep.

A mapped preparation streams its forcing series: the frameset is opened
rather than read, and the regular-source snapshots pack one valid time
when that valid time is asked for.  The hierarchy path -- every nested
preparation, which is every run with a child domain -- joins that series
to a static catalog through :class:`NestedInputCatalog` and then selects
ONE snapshot per child at its start time.

Named breakage: a catalog that copies the series into a tuple holds
every valid time's arrays for the whole preparation, which is exactly
the residency the streamed decode exists to avoid.  On a 3 km CONUS
source with a 45-level ladder one valid time is gigabytes, so a flat
catalog puts a seven-time nested run back where the OOM reaper found it.

These tests measure residency directly: the double below creates a fresh
snapshot per index and counts how many are alive at once, so a consumer
that retains the series shows a peak equal to the number of valid times
and a consumer that streams shows one.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
import gc
import weakref

import pytest

from gpuwm.ingest.nest_init import NestedInputCatalog, _initial_snapshot


class _Snapshot:
    """One forcing time, as heavy as its liveness counter says it is."""

    def __init__(self, valid_time: datetime) -> None:
        self.valid_time = valid_time


class _StreamedSeries(Sequence):
    """A forcing series that packs one valid time at a time.

    The stand-in for ``MappedSourceBundle.regular_snapshots()``: it
    publishes its valid times from a document (so a caller never has to
    pack a snapshot to read a clock), and every index builds a NEW
    snapshot object.  Nothing here keeps a snapshot alive, so `peak` is
    the number the CALLER held at once.
    """

    def __init__(self, times: Sequence[datetime]) -> None:
        self._times = tuple(times)
        self.live = 0
        self.peak = 0
        self.packed = 0
        # The weakrefs themselves must outlive the objects they watch,
        # or their release callbacks never run and every snapshot looks
        # resident forever.
        self._watchers: list[weakref.ref] = []

    @property
    def valid_times(self) -> tuple[datetime, ...]:
        return self._times

    def __len__(self) -> int:
        return len(self._times)

    def __getitem__(self, position):
        if isinstance(position, slice):
            raise AssertionError(
                "the nested path must not slice the forcing series")
        index = int(position)
        if index < 0:
            index += len(self._times)
        if not 0 <= index < len(self._times):
            raise IndexError(position)
        snapshot = _Snapshot(self._times[index])
        self.packed += 1
        self.live += 1
        self.peak = max(self.peak, self.live)

        def _released(_reference, series=self) -> None:
            series.live -= 1

        self._watchers.append(weakref.ref(snapshot, _released))
        return snapshot


def _series(count: int) -> _StreamedSeries:
    start = datetime(2026, 8, 17, 0, 0, 0)
    return _StreamedSeries(
        [start + timedelta(hours=hour) for hour in range(count)])


def test_nested_catalog_holds_one_valid_time_not_the_series():
    series = _series(7)
    catalog = NestedInputCatalog(snapshots=series, static_catalog=object())
    gc.collect()
    assert series.peak <= 1, (
        f"the nested catalog held {series.peak} valid times at once; a "
        "streamed forcing series must stay one deep")
    assert catalog.valid_times == series.valid_times


def test_nested_catalog_reads_the_clock_without_packing_a_snapshot():
    series = _series(7)
    catalog = NestedInputCatalog(snapshots=series, static_catalog=object())
    packed_after_build = series.packed
    assert catalog.valid_times == series.valid_times
    assert series.packed == packed_after_build, (
        "reading the forcing clock packed a valid time; the series "
        "publishes its own valid times")


def test_child_selection_packs_only_the_requested_valid_time():
    series = _series(7)
    catalog = NestedInputCatalog(snapshots=series, static_catalog=object())
    gc.collect()
    before = series.packed
    chosen = _initial_snapshot(catalog, series.valid_times[3])
    assert chosen.valid_time == series.valid_times[3]
    assert series.packed - before == 1, (
        f"selecting one child start time packed {series.packed - before} "
        "valid times")


def test_a_plain_tuple_series_still_works():
    # The hierarchy path also carries eagerly decoded adapters; the
    # streaming rework must not make a tuple illegal.
    start = datetime(2026, 8, 17, 0, 0, 0)
    snapshots = tuple(
        _Snapshot(start + timedelta(hours=hour)) for hour in range(3))
    catalog = NestedInputCatalog(
        snapshots=snapshots, static_catalog=object())
    assert catalog.valid_times == tuple(
        snapshot.valid_time for snapshot in snapshots)
    assert _initial_snapshot(catalog) is snapshots[0]


def test_non_increasing_valid_times_still_refuse_on_a_streamed_series():
    start = datetime(2026, 8, 17, 0, 0, 0)
    series = _StreamedSeries([start, start - timedelta(hours=1)])
    with pytest.raises(ValueError, match="unique increasing valid times"):
        NestedInputCatalog(snapshots=series, static_catalog=object())


def test_a_mixed_adapter_series_still_refuses():
    class _Other(_Snapshot):
        pass

    start = datetime(2026, 8, 17, 0, 0, 0)
    snapshots = (
        _Snapshot(start), _Other(start + timedelta(hours=1)))
    with pytest.raises(TypeError, match="one adapter type"):
        NestedInputCatalog(snapshots=snapshots, static_catalog=object())
