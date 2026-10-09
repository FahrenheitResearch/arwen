"""The one lock every netCDF4/HDF5 session in this process takes.

netCDF4 releases the GIL around HDF5 calls and the shipped HDF5 library
is not thread-safe, so two Python threads inside it at once is a crash,
not a race that merely produces wrong numbers.  :mod:`gpuwm.io.wrfout`
has serialized its own writer sessions on this basis since the async
writer landed; the lock lived there, which meant it protected the writer
against itself and against nothing else.

THE BREAKAGE THIS PREVENTS, measured: a two-domain run with a relocating
1 km child died with SIGSEGV or SIGBUS immediately after
``wrfout_d01_..._12_40_00`` was written, in 3 of 8 full runs, always at
the same point -- the 2400 s relocation.  ``faulthandler`` on the SIGBUS
put the main thread at ``gpuwm/core/rrtmgp.py`` opening
``rfmip-clear-sky-inputs.nc`` inside the RRTMGP state being rebuilt for
the moved child, while a ``gpuwm-wrfout-*`` writer thread sat in
``validate_wrfout_file`` reopening the tape it had just published.  Two
threads inside HDF5, one of them holding a lock the other had never
heard of.  Nine frames on disk out of fourteen, on both the release line
and the lane, so it is neither adaptive- nor relocation-specific: any
netCDF4 read that lands beside a publish can do it.

The rule this module exists to make enforceable: EVERY netCDF4 session
in this process is opened under :data:`NETCDF4_IO_LOCK`.  It lives here
rather than in ``gpuwm.io.wrfout`` so that a reader in
``gpuwm.core`` can take it without importing the writer.
"""
from __future__ import annotations

import contextlib
import threading

#: The process-wide netCDF4/HDF5 session lock.  Not reentrant: a holder
#: must not open a second session, and no call site does -- the writer's
#: publish-time re-validation happens inside the session it already
#: holds, which is why it is not wrapped a second time.
NETCDF4_IO_LOCK = threading.Lock()


class _DeferredNetCDF4:
    """The ``netCDF4`` module, imported the first time it is used.

    THE BREAKAGE THIS PREVENTS (D-10, 2.8.8 acceptance): netCDF4 has no
    free-threaded build, and importing it switches a free-threaded
    interpreter's lock back on for the rest of the process ("The global
    interpreter lock (GIL) has been enabled to load module
    'netCDF4._netCDF4'").  Six modules imported it at module scope, so a
    preparation hosted in-process by the site's ``engine_prepare.py`` --
    which reads every byte through the Rust route and never opens a
    netCDF4 dataset -- turned the lock on just by importing the parser
    (``gpuwm.downscale``), the domain-artifact reader
    (``gpuwm.wrf_direct``) and the memory estimate (``gpuwm.core.rrtmgp``).

    ``from gpuwm.io.netcdf_serialization import netCDF4`` is the spelling
    for a module that uses netCDF4 on some path: its call sites read
    ``netCDF4.Dataset(...)`` unchanged, and the import happens on the
    first attribute asked for, which is the first real use.
    """

    __slots__ = ()

    def __getattr__(self, name: str):
        import netCDF4 as library

        return getattr(library, name)

    def __repr__(self) -> str:
        return "<netCDF4, imported on first use>"


#: ``netCDF4``, deferred to its first use (:class:`_DeferredNetCDF4`).
netCDF4 = _DeferredNetCDF4()


@contextlib.contextmanager
def netcdf4_session():
    """Hold the process-wide netCDF4 lock for one open/read/close.

    Wrap the whole ``with Dataset(...)`` block, not just the open: HDF5
    is entered again on every variable read, so releasing at the open
    would serialize nothing that matters.
    """
    with NETCDF4_IO_LOCK:
        yield
