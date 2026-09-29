"""What a forcing time costs in initialize_real, and that the cost never shows in the state.

Two things made the "Initialize root forcing states" stage of a 1792 x 1024
x 55 preparation run for 21 minutes with its GPU idle and one CPU core busy:

* The float64 setup columns ran serially.  Every column helper already
  splits its arrays into contiguous slabs whose results are the serial
  bytes, but only an explicit ``column_workers`` turned that on and no
  preparation route passed one.  The default is now every CPU the process
  may run on.
* Every forcing time built the HRRR hydrometeor vertical disposition
  receipt, one operator replay per source level over the whole domain,
  and every route then dropped it for all but the start time.
  ``boundary_only`` skips it for a time whose result only feeds the
  lateral boundaries.

Both must leave the state byte-identical, which is what these tests hold.
The threads of a column helper also free their slab temporaries into their
own glibc arenas, which keep them resident; every column pool hands them
back to the operating system when it closes, so the worker count stops
raising the preparation's host memory.
"""

import platform
import sys

import numpy as np
import pytest

from gpuwm.ingest import real

from test_real_init import (
    _ReferencePreprocessBackend,
    _analyzed_hrrr_real_init,
    _pressure_level_real_init,
)


def _state_arrays(result):
    state = result.state
    arrays = {}
    for name, value in sorted(vars(state).items()):
        if isinstance(value, np.ndarray) and value.size:
            arrays[name] = value
    assert arrays, "the state carried no arrays to compare"
    return arrays


def _assert_same_state(first, second):
    left, right = _state_arrays(first), _state_arrays(second)
    assert sorted(left) == sorted(right)
    for name in left:
        assert left[name].dtype == right[name].dtype, name
        assert left[name].tobytes() == right[name].tobytes(), name


class _CountingBackend(_ReferencePreprocessBackend):
    """The reference backend, counting every vertical-plan application."""

    def __init__(self):
        self.applications = 0

    def prepare_wrf_vertical(self, source, surface, target):
        plan = super().prepare_wrf_vertical(source, surface, target)
        backend = self

        class Counted:
            def apply(self, field, surface_value, **options):
                backend.applications += 1
                return plan.apply(field, surface_value, **options)

        return Counted()


def test_a_boundary_only_time_keeps_its_state_and_skips_the_disposition_replays():
    kept_backend, boundary_backend = _CountingBackend(), _CountingBackend()
    kept, _ = _analyzed_hrrr_real_init(
        8, preprocess_backend=kept_backend)
    boundary, _ = _analyzed_hrrr_real_init(
        8, preprocess_backend=boundary_backend,
        init_kwargs={"boundary_only": True})

    _assert_same_state(kept, boundary)
    disposition = kept.hydrometeor_initialization["vertical_disposition"]
    assert disposition["geometry"]["production_target_support"]["level_replays"]
    assert set(disposition["species"]) == {"QC", "QR", "QI", "QS", "QG"}
    assert boundary.hydrometeor_initialization["vertical_disposition"] == {}
    # Six source levels replayed one at a time, then three replays for each
    # of the five retained species: 21 applications a boundary time no
    # longer pays for.
    assert kept_backend.applications - boundary_backend.applications == 6 + 3 * 5


def test_boundary_only_must_be_a_boolean():
    try:
        _analyzed_hrrr_real_init(8, init_kwargs={"boundary_only": "yes"})
    except TypeError as error:
        assert "boundary_only" in str(error)
    else:
        raise AssertionError("a non-boolean boundary_only was accepted")


def test_the_default_column_workers_give_the_serial_bytes_on_the_native_lane():
    serial, _ = _analyzed_hrrr_real_init(
        8, shape=(7, 5), init_kwargs={"column_workers": 1})
    default, _ = _analyzed_hrrr_real_init(8, shape=(7, 5))
    threaded, _ = _analyzed_hrrr_real_init(
        8, shape=(7, 5), init_kwargs={"column_workers": 3})
    _assert_same_state(serial, default)
    _assert_same_state(serial, threaded)


def test_the_default_column_workers_give_the_serial_bytes_on_the_rh_lane():
    serial, _ = _pressure_level_real_init(
        8, shape=(7, 5), init_kwargs={"column_workers": 1})
    default, _ = _pressure_level_real_init(8, shape=(7, 5))
    threaded, _ = _pressure_level_real_init(
        8, shape=(7, 5), init_kwargs={"column_workers": 4})
    _assert_same_state(serial, default)
    _assert_same_state(serial, threaded)


def test_every_column_pool_returns_its_threads_freed_memory(monkeypatch):
    pools, trims = [], []

    class CountingPool(real.ThreadPoolExecutor):
        def __init__(self, *args, **kwargs):
            pools.append(1)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(real, "ThreadPoolExecutor", CountingPool)
    monkeypatch.setattr(real, "_return_freed_host_memory",
                        lambda: trims.append(1))
    serial, _ = _analyzed_hrrr_real_init(
        8, shape=(7, 5), init_kwargs={"column_workers": 1})
    assert pools == [] and len(trims) == 1
    trims.clear()
    threaded, _ = _analyzed_hrrr_real_init(
        8, shape=(7, 5), init_kwargs={"column_workers": 3})
    # One return per pool as it closes, and one as the forcing time ends.
    assert pools and len(trims) == len(pools) + 1
    _assert_same_state(serial, threaded)


def test_a_column_pool_returns_freed_memory_when_a_worker_fails(monkeypatch):
    trims = []
    monkeypatch.setattr(real, "_return_freed_host_memory",
                        lambda: trims.append(1))
    with pytest.raises(ZeroDivisionError):
        with real._column_pool(2) as executor:
            executor.submit(lambda: 1 / 0).result()
    assert trims == [1]


def test_the_allocator_trim_resolves_once_and_does_nothing_without_glibc(
        monkeypatch):
    monkeypatch.setattr(real, "_MALLOC_TRIM", None)
    real._return_freed_host_memory()
    resolved = real._MALLOC_TRIM
    if sys.platform.startswith("linux") and platform.libc_ver()[0] == "glibc":
        assert callable(resolved)
    else:
        assert resolved is False
    real._return_freed_host_memory()
    assert real._MALLOC_TRIM is resolved
