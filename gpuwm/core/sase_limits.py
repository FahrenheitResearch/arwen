"""SASE constants shared by configuration, state and CPU memory accounting.

They live here, in a module with no imports at all, because of where
they have to be readable from.  The closure's authority module is
``gpuwm.verify.sase_ref``, and ``gpuwm/verify`` is a DEVELOPER
verification tree that the standalone CPU preprocessing distribution
deliberately omits -- so a config- or state-layer import of it breaks
that wheel (``tests/test_native_wrf_distribution.py`` is the gate that
says so, and it caught exactly that).

There is ONE definition of each value. ``sase_ref`` imports the column
limit and energy floor from here, preserving its constant registry and
configuration hash. The CUDA launch tier and CPU workspace estimator
share the reduction block size through this same import-free module.
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

#: Threads in each SASE reduction block. The CUDA launch tier and the CPU
#: workspace estimator share this value so accounting never imports CuPy.
THREADS_PER_BLOCK = 128

#: Realizability floor on the prognostic subgrid energy [m2 s-2].
#:
#: The value the fused step clips to and the value a cold start fills, so
#: the state allocator and the closure agree on what "no turbulence yet"
#: is.  Registered in the closure's configuration hash.
E_MIN = 1.0e-6
