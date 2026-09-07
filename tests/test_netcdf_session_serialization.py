"""One netCDF4/HDF5 session at a time, across the WHOLE process.

THE BUG THIS PINS.  netCDF4 releases the GIL around HDF5 calls and the
shipped HDF5 library is not thread-safe, so two Python threads inside it
at once is a crash rather than a race that merely produces wrong numbers.
``gpuwm.io.wrfout`` had known this since its async writer landed and held
a lock for it -- a lock private to that module, which therefore protected
the writer against itself and against nothing else.

MEASURED: a two-domain run with a relocating 1 km child died with SIGSEGV
or SIGBUS immediately after ``wrfout_d01_..._12_40_00`` was written, in 3
of 8 full runs, always at the 2400 s relocation, leaving nine frames of
fourteen on disk.  ``faulthandler`` on the SIGBUS put the main thread in
``gpuwm/core/rrtmgp.py`` opening ``rfmip-clear-sky-inputs.nc`` for the
RRTMGP state being rebuilt for the moved child, while a
``gpuwm-wrfout-*`` writer thread sat in ``validate_wrfout_file``
reopening the tape it had just published.  It happened on the release
line and on the lane, so it is neither adaptive- nor relocation-specific:
any netCDF4 read that lands beside a publish can do it.
"""

from __future__ import annotations

import ast
import pathlib
import threading

import pytest


# ------------------------------------------------------------ one lock

def test_the_writer_and_the_table_reader_share_one_lock():
    """Two locks is the same as none: each guards a different door."""
    from gpuwm.io import netcdf_serialization, wrfout

    assert wrfout._NETCDF4_IO_LOCK is netcdf_serialization.NETCDF4_IO_LOCK


def test_the_session_actually_serializes():
    """The positive control: it is a lock, not a decorative context."""
    from gpuwm.io.netcdf_serialization import netcdf4_session

    entered = threading.Event()
    second_got_in = threading.Event()

    def other():
        with netcdf4_session():
            second_got_in.set()

    with netcdf4_session():
        thread = threading.Thread(target=other, daemon=True)
        thread.start()
        entered.set()
        assert not second_got_in.wait(0.3), (
            "a second session entered while the first was held")
    thread.join(timeout=5.0)
    assert second_got_in.is_set(), "the second session never ran"
    assert entered.is_set()


def test_the_lock_is_not_reentrant_by_construction():
    """A holder must not open a second session, and the type says so.

    An RLock here would let one call site nest two sessions and look
    correct while the contract it documents ('every session takes this')
    quietly stopped meaning anything.
    """
    from gpuwm.io.netcdf_serialization import NETCDF4_IO_LOCK

    assert isinstance(NETCDF4_IO_LOCK, type(threading.Lock()))


# ------------------------------------ every table read is under the lock

def _dataset_opens(path):
    """(lineno, guarded) for every ``Dataset(...)`` opened in a with."""
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        names = []
        opens = []
        for item in node.items:
            call = item.context_expr
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            name = (func.id if isinstance(func, ast.Name)
                    else func.attr if isinstance(func, ast.Attribute)
                    else "")
            names.append(name)
            if name == "Dataset":
                opens.append(call.lineno)
        for lineno in opens:
            found.append((lineno, "netcdf4_session" in names))
    return found


def test_every_rrtmgp_table_read_holds_the_session():
    """The reader that was measured crashing, pinned at the source.

    A grep rather than a run because the crash is intermittent -- 3 in 8
    -- so a test that tried to reproduce it would be green most of the
    time for the wrong reason.  What is checkable without luck is that no
    ``Dataset(...)`` in this module is opened outside the session, which
    is the property the crash violated.
    """
    import gpuwm

    # The SOURCE, not the imported module: importing rrtmgp pulls the
    # packaged radiation tables in, and this audit is about what the file
    # says, so it must hold on a box that has no companion data.
    path = pathlib.Path(gpuwm.__file__).parent / "core" / "rrtmgp.py"
    opens = _dataset_opens(path)
    assert opens, "no Dataset opens found -- the audit is measuring nothing"
    unguarded = [lineno for lineno, guarded in opens if not guarded]
    assert unguarded == [], (
        f"gpuwm/core/rrtmgp.py opens netCDF4 outside netcdf4_session() at "
        f"lines {unguarded}; the writer thread can be inside HDF5 at the "
        f"same instant")


def test_the_audit_catches_an_unguarded_open(tmp_path):
    """Test the tester, in the direction that matters."""
    source = tmp_path / "sample.py"
    source.write_text(
        "with Dataset('a.nc', 'r') as nc:\n    pass\n"
        "with netcdf4_session(), Dataset('b.nc', 'r') as nc:\n    pass\n",
        encoding="utf-8")
    opens = _dataset_opens(source)
    assert [guarded for _lineno, guarded in opens] == [False, True]


@pytest.mark.parametrize("module", ["gpuwm.io.wrfout"])
def test_the_writer_still_serializes_its_own_session(module):
    """The lock moved modules; it must not have moved out of the writer."""
    import importlib

    mod = importlib.import_module(module)
    text = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
    assert "with _NETCDF4_IO_LOCK:" in text
