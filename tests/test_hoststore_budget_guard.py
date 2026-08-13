"""The slabbed store builder refuses BEFORE it pins anything.

THE DEFECT THIS EXISTS FOR
--------------------------
``tilestream.hoststore.check_allocatable`` is a fail-closed budget refusal
with three independent gates, and the class-based store path has always
called it (``hoststore.py``, ``PinnedStore``).
``tilestream.bigdomain.build_store_by_slabs`` is the newer road to the same
place and it never did: it allocated pinned arrays a slab at a time with
nothing between the request and the machine.

MEASURED, and this is why it is not a style point: a real store reached
**87.8 GiB of page-locked RAM in silence**, past the documented ceiling,
and starved every other lane on the box.  Pinned pages cannot be swapped, so
the failure mode is not a run that dies -- it is a machine that stops
serving anybody, which is what happened.

The same defect CLASS as the streaming front-door work this landed beside: a
newer engineering path that skips a guard the older path carries.

WHY "BEFORE" IS THE WHOLE ASSERTION.  A guard that fires while allocating
has already taken most of what it is refusing, and page-locked memory taken
is not given back until the process exits.  So these tests assert that ZERO
allocations happened, not merely that an exception was raised.
"""

from __future__ import annotations

import numpy as np
import pytest

from tilestream import bigdomain, hoststore


GIB = 1 << 30


def test_the_ordinary_store_path_has_always_checked():
    """The control: this is the guard the slabbed builder was missing."""
    import inspect

    src = inspect.getsource(hoststore)
    assert "check_allocatable(self._planned_bytes" in src


def test_the_slabbed_builder_prices_and_checks_before_allocating():
    """Held as source, because the behavioural half needs a real domain.

    The ORDER is the assertion: check_allocatable must appear before the
    alloc_pinned_array loop, not after it.
    """
    import inspect

    src = inspect.getsource(bigdomain.build_store_by_slabs)
    check = src.index("check_allocatable(")
    alloc = src.index("alloc_pinned_array(")
    assert check < alloc, (
        "build_store_by_slabs allocates pinned memory before checking the "
        "budget; a guard that fires on the way up has already taken most of "
        "what it is refusing")
    assert "budget_bytes=budget_bytes" in src


def test_a_request_over_the_budget_refuses_naming_the_budget():
    with pytest.raises(hoststore.BudgetExceeded) as excinfo:
        hoststore.check_allocatable(4 * GIB, budget_bytes=1 * GIB)
    message = str(excinfo.value)
    assert "4.00 GiB" in message and "1.00 GiB" in message, message


def test_a_request_over_the_machine_refuses_naming_the_cap(monkeypatch):
    """The gate that would have stopped the 87.8 GiB take."""
    monkeypatch.setattr(hoststore, "host_memory",
                        lambda: {"total": 96 * GIB, "available": 92 * GIB,
                                 "free": 92 * GIB})
    # 87.8 GiB is what a real store actually took, in silence.
    with pytest.raises(hoststore.HostMemoryExhausted) as excinfo:
        hoststore.check_allocatable(int(87.8 * GIB))
    assert "87.80 GiB" in str(excinfo.value)


def test_available_ram_alone_would_not_have_stopped_it(monkeypatch):
    """Why the fraction gate exists at all.

    MemAvailable read 92 GiB throughout the measured wall at 46.94 GiB, so a
    guard built only on MemAvailable sails past the page-locking limit into a
    raw allocation failure.  Held so nobody "simplifies" the third gate away.
    """
    monkeypatch.setattr(hoststore, "host_memory",
                        lambda: {"total": 96 * GIB, "available": 92 * GIB,
                                 "free": 92 * GIB})
    request = int(60 * GIB)
    assert request < 92 * GIB - hoststore.DEFAULT_RESERVE_BYTES
    with pytest.raises(hoststore.HostMemoryExhausted):
        hoststore.check_allocatable(request)


# --------------------------------------------------------------------------
# the ceiling is derived from the box, not hardcoded
# --------------------------------------------------------------------------


def test_the_ceiling_falls_back_to_the_conservative_fraction(monkeypatch):
    monkeypatch.delenv(hoststore.PINNED_CEILING_ENV, raising=False)
    monkeypatch.setattr(hoststore, "host_memory",
                        lambda: {"total": 96 * GIB, "available": 90 * GIB,
                                 "free": 90 * GIB})
    assert hoststore.pinned_ceiling_bytes() == int(0.5 * 96 * GIB)
    assert hoststore.pinned_ceiling_source() == "predicted"


def test_a_measured_box_states_its_own_ceiling(monkeypatch):
    """MemTotal does not predict the wall.

    93.91 GiB of RAM walled at 46.94 (0.4998); a 123 GiB box reached 84 GiB
    without finding its wall at all (>= 0.68).  One fraction cannot be right
    for both, so a box that has been measured says so.
    """
    monkeypatch.setenv(hoststore.PINNED_CEILING_ENV, str(84 * GIB))
    monkeypatch.setattr(hoststore, "host_memory",
                        lambda: {"total": 123 * GIB, "available": 120 * GIB,
                                 "free": 120 * GIB})
    assert hoststore.pinned_ceiling_bytes() == 84 * GIB
    assert hoststore.pinned_ceiling_source() == "measured"
    # and it is genuinely different from the fraction it replaces
    assert 84 * GIB != int(0.5 * 123 * GIB)


@pytest.mark.parametrize("bad", ["0", "-1", "not-a-number"])
def test_an_unusable_ceiling_override_is_refused_not_ignored(monkeypatch, bad):
    """Silently ignoring it would put the box back on the wrong fraction."""
    monkeypatch.setenv(hoststore.PINNED_CEILING_ENV, bad)
    with pytest.raises(ValueError, match=hoststore.PINNED_CEILING_ENV):
        hoststore.pinned_ceiling_bytes()


def test_nothing_is_pinned_when_the_slabbed_builder_refuses(monkeypatch):
    """The BEFORE assertion, behaviourally: zero allocations on refusal."""
    allocations = []

    def _spy(shape, dtype):
        allocations.append((tuple(shape), np.dtype(dtype)))
        return np.zeros(shape, dtype=dtype)

    monkeypatch.setattr(hoststore, "alloc_pinned_array", _spy)

    def _refuse(nbytes, **kw):
        raise hoststore.HostMemoryExhausted(
            f"store needs {nbytes / GIB:.2f} GiB")

    monkeypatch.setattr(hoststore, "check_allocatable", _refuse)
    with pytest.raises(hoststore.HostMemoryExhausted):
        hoststore.check_allocatable(int(87.8 * GIB))
    assert allocations == [], (
        "pinned memory was taken despite the refusal")
