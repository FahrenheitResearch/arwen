"""The two SASE constants the config and state layers need.

They live here, in a module with no imports at all, because of where
they have to be readable from.  The closure's authority module is
``gpuwm.verify.sase_ref``, and ``gpuwm/verify`` is a DEVELOPER
verification tree that the standalone CPU preprocessing distribution
deliberately omits -- so a config- or state-layer import of it breaks
that wheel (``tests/test_native_wrf_distribution.py`` is the gate that
says so, and it caught exactly that).

There is still ONE definition of each value: ``sase_ref`` imports them
from here rather than restating them, so the authority module's
constant registry and its configuration hash keep binding the same
numbers they always did.
"""

from __future__ import annotations

#: Deepest column the closure's implicit vertical solve accepts.
#:
#: An IMPLEMENTATION bound, not a physical one: the tridiagonal sweeps
#: keep three FP64 columns of this depth in per-thread local memory, so
#: the device ``SASE_KMAX`` define, the launcher's rejection and
#: ``gpuwm.config.SASE_MAX_NZ`` must be one number.  Must stay a power of
#: two -- it is emitted as a compile-time define alongside the block size
#: that sizes shared-memory tree reductions.
MAX_COLUMN_LEVELS = 128

#: Realizability floor on the prognostic subgrid energy [m2 s-2].
#:
#: The value the fused step clips to and the value a cold start fills, so
#: the state allocator and the closure agree on what "no turbulence yet"
#: is.  Registered in the closure's configuration hash.
E_MIN = 1.0e-6
