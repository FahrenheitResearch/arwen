"""The New Tiedtke adapter hands the scheme WRF's real ITIMESTEP.

cu_ntiedtke_pre_run zeroes the forcing pair on the first step
(module_cu_ntiedtke.F:451-457).  The pipeline was built with itimestep = 2
and never told otherwise, so the first cumulus call of every run fed the
folded radiation and boundary-layer lanes into a closure WRF runs without
them.  Found by tools/ntiedtke_column_oracle; the GPU half of the proof is
tests/test_ntiedtke_column_oracle.py.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

from gpuwm.core import ntiedtke


def _state(elapsed, offset=None):
    s = SimpleNamespace(elapsed_seconds=elapsed)
    if offset is not None:
        s.domain_start_offset = offset
    return s


def test_the_first_step_is_itimestep_one():
    cfg = SimpleNamespace(dt=72.0)
    assert ntiedtke._nt_itimestep(_state(0.0), cfg) == 1
    assert ntiedtke._nt_itimestep(_state(72.0), cfg) == 2
    assert ntiedtke._nt_itimestep(_state(3600.0), cfg) == 51


def test_a_late_nest_counts_from_its_own_start():
    """A WRF nest's ITIMESTEP starts at 1 when it activates."""
    cfg = SimpleNamespace(dt=24.0)
    assert ntiedtke._nt_itimestep(_state(3600.0, offset=3600.0), cfg) == 1
    assert ntiedtke._nt_itimestep(_state(3624.0, offset=3600.0), cfg) == 2


def test_the_adapter_sets_the_scalar_on_every_call():
    """The pipeline is cached across calls, so the scalar has to be set per
    call, not in the constructor, or every step after the first would be
    run as the first (or the first as a later one)."""
    source = inspect.getsource(ntiedtke.NewTiedtke.__call__)
    assert 'p.scalars["itimestep"] = np.int32(_nt_itimestep(state, cfg))' \
        in source
