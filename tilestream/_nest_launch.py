"""Run the nest gate on a CONTENDED card without dying at someone else's peak.

Not part of the gate and not imported by it.  These boxes are shared by a
dozen agents; three runs of ``tilestream.test_nest`` were killed by
``OutOfMemoryError`` at allocations of 200 MB and 7 MB -- not because the
gate is large but because another process took the card between the gate's
first allocation and its tenth.  A measurement interrupted that way says
nothing about the code under test, so this wrapper does two things and
nothing else:

1. WAITS until the device has ``--reserve`` GiB free (polling
   ``cupy.cuda.runtime.memGetInfo``, never ``nvidia-smi``, whose used/free is
   unreliable under WSL2 and reports the whole host on a shared box);
2. RESERVES that memory into CuPy's own pool and keeps it there, by
   allocating it once and neutering ``free_all_blocks`` for the run.  The
   pool then hands the same physical bytes back to the gate instead of
   returning them to the driver for a neighbour to take.

Neither changes a computed value: ``free_all_blocks`` releases only UNUSED
cached blocks and allocator behaviour is not part of any comparator (the
same argument ``gpuwm.core.model._trim_default_pool`` makes for calling it).
"""

from __future__ import annotations

import argparse
import sys
import time


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reserve", type=float, default=14.0,
                    help="GiB of device memory to wait for and then hold")
    ap.add_argument("--headroom", type=float, default=5.0,
                    help="GiB left outside the reserve for CUDA's own"
                         " module loading and library scratch")
    ap.add_argument("--wait-minutes", type=float, default=240.0)
    ap.add_argument("--module", default="tilestream.test_nest",
                    help="the gate module to run (it must expose main(argv))")
    ap.add_argument("gate", nargs=argparse.REMAINDER,
                    help="arguments forwarded to the gate module")
    args = ap.parse_args(argv)

    import cupy as cp

    need = float(args.reserve) * 2 ** 30
    # Headroom the reserve must NOT eat.  CUDA module loading, the JIT's
    # own workspaces and every library's scratch live OUTSIDE CuPy's pool:
    # a reserve of 13.0 GiB taken out of 13.6 GiB free left 0.6 GiB for
    # them and the run died in ``launchKernel`` with
    # CUDA_ERROR_OUT_OF_MEMORY, which reads like a model that is too big
    # and is not.
    headroom = float(args.headroom) * 2 ** 30
    deadline = time.time() + float(args.wait_minutes) * 60.0
    while True:
        try:
            free, total = cp.cuda.runtime.memGetInfo()
        except cp.cuda.runtime.CUDARuntimeError as exc:
            # A card with no room left to create a CONTEXT fails here rather
            # than reporting 0 free.  That is still "busy", not "broken".
            if time.time() > deadline:
                print(f"gave up: {exc}", flush=True)
                return 2
            print(f"waiting: device refuses a context ({exc})", flush=True)
            time.sleep(20.0)
            continue
        if free >= need + headroom:
            print(f"reserving {need / 2**30:.1f} GiB of "
                  f"{free / 2**30:.1f} GiB free ({total / 2**30:.1f} total), "
                  f"leaving {headroom / 2**30:.1f} GiB of headroom",
                  flush=True)
            break
        if time.time() > deadline:
            print(f"gave up: only {free / 2**30:.1f} GiB free, need "
                  f"{(need + headroom) / 2**30:.1f}", flush=True)
            return 2
        print(f"waiting: {free / 2**30:.1f} GiB free, need "
              f"{(need + headroom) / 2**30:.1f}", flush=True)
        time.sleep(20.0)

    pool = cp.get_default_memory_pool()
    if need <= 0:
        # WAIT ONLY.  A gate that relies on ``free_all_blocks`` between its
        # own cases -- ``tilestream.test_gate`` does, at every rung -- must
        # not have it disarmed: holding every block the run ever touched
        # pushed its footprint past 13.4 GiB and OOMed a card that had room
        # for the gate itself.  So the reserve is opt-in.
        import importlib as _il

        rc = _il.import_module(args.module).main(
            [a for a in args.gate if a != "--"])
        return rc

    block = cp.empty(int(need) // 4, dtype=cp.float32)
    del block                      # -> the POOL keeps it, the driver does not

    class _HoldingPool:
        """The real pool with ``free_all_blocks`` disarmed.

        ``MemoryPool`` is a Cython extension type and its methods are
        read-only, so the interception has to be at
        ``get_default_memory_pool`` rather than on the object.
        """

        def __init__(self, real):
            self._real = real

        def free_all_blocks(self, *a, **k):
            return None

        def __getattr__(self, name):
            return getattr(self._real, name)

    holding = _HoldingPool(pool)
    cp.get_default_memory_pool = lambda: holding

    import importlib

    gate = importlib.import_module(args.module)

    gate_argv = list(args.gate)
    if gate_argv and gate_argv[0] == "--":
        gate_argv = gate_argv[1:]
    rc = gate.main(gate_argv)
    print(f"pool held {pool.total_bytes() / 2**30:.1f} GiB, "
          f"used {pool.used_bytes() / 2**30:.1f} GiB at exit", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
