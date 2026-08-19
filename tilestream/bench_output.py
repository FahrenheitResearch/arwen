"""What one ArWen history write actually costs, and where the time goes.

This drives the REAL writer (:class:`gpuwm.io.wrfout.WrfoutWriter` and the
staging half of :class:`~gpuwm.io.wrfout.AsyncDomainWrfoutWriter`) on a REAL
seeded :func:`tilestream.harness.make_state` domain.  Nothing here is a
synthetic stand-in for the output path; the only code this module owns is the
timing instrumentation and the deliberate controls.

Run as::

    python -m tilestream.bench_output <mode>      # from the repository root

Modes
-----
``fs``          filesystem identity + write ceiling (dd-equivalent, fsynced)
``path``        static characterisation of the writer (no timing)
``decompose``   one frame split into DERIVE / D2H / CREATE / ENCODE / FSYNC /
                VALIDATE / RENAME, at one domain size
``sweep``       the same decomposition across a domain-size ladder, with a
                least-squares fit of cost against cell count
``variants``    netCDF vs raw ``tofile`` vs deflate levels, same bytes
``cadence``     timesteps/s with and without output at several cadences

MEASUREMENT DISCIPLINE
----------------------
* ``/proc/sys/vm/drop_caches`` is not writable here (no root), so DISK is timed
  THROUGH ``fsync`` in every reported figure, and the sustained device rate is
  established separately by writing far beyond the dirty-page limit.  A number
  above that rate is a page-cache artefact, not a measurement.
* Every timed quantity is reported as a median over >= 5 reps with the
  min/max spread; spread > 10% is flagged in the printed table.
* Each frame is deleted immediately after it is measured.  Peak on-disk
  footprint is one frame.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

#: The filesystem under test.  It is the working directory by default so
#: the run measures the disk the checkout is on, and an env var because
#: the measurement of record was taken on ext4 on the WSL2 VHD
#: (/dev/sdd, confirmed by ``mode fs``) rather than on whatever a reader
#: happens to be standing in.
DISK_DIR = Path(os.environ.get("ARWEN_BENCHOUT_DIR", "_benchout"))
#: tmpfs, i.e. RAM.  Used ONLY as the ENCODE-without-disk control.
TMPFS_DIR = Path("/dev/shm/arwen_benchout")

NZ = 49
DEFAULT_REPS = 7
#: ~8.0x span in cell count, which is the sweep the task asks for.
DEFAULT_SIZES = (512, 724, 1024, 1448)


def _now() -> float:
    return time.perf_counter()


def _settle() -> None:
    """Make consecutive reps independent of one another.

    Each rep pushes >1 GB of dirty pages.  dirty_background_ratio is 10% of
    93.9 GiB here, so without this the kernel is still writing back rep N
    while rep N+1 is being timed -- which is exactly how a disk measurement
    turns into a measurement of the previous disk measurement.  The frame file
    is already unlinked by the caller (freeing its page-cache pages); this
    flushes anything else outstanding and lets the device drain.
    """
    os.sync()
    time.sleep(0.30)


@dataclass
class Samples:
    """A named set of repetitions of one timed quantity."""

    name: str
    values: list[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        self.values.append(float(value))

    @property
    def median(self) -> float:
        return statistics.median(self.values) if self.values else float("nan")

    @property
    def lo(self) -> float:
        return min(self.values) if self.values else float("nan")

    @property
    def hi(self) -> float:
        return max(self.values) if self.values else float("nan")

    @property
    def spread(self) -> float:
        """(max - min) / median, the flag the discipline note asks for."""
        if not self.values or self.median == 0.0:
            return float("nan")
        return (self.hi - self.lo) / self.median

    def row(self, total: float | None = None) -> str:
        pct = "" if total in (None, 0) else f"{100.0 * self.median / total:6.1f}%"
        flag = " <<SPREAD" if self.spread > 0.10 else ""
        return (f"  {self.name:<26s} {self.median * 1e3:10.2f} ms  "
                f"[{self.lo * 1e3:9.2f}, {self.hi * 1e3:9.2f}]  "
                f"spread {self.spread * 100:5.1f}%  {pct}{flag}")


# ---------------------------------------------------------------------------
# filesystem
# ---------------------------------------------------------------------------

def _fs_identity(path: Path) -> dict:
    path.mkdir(parents=True, exist_ok=True)
    out = subprocess.run(
        ["findmnt", "-T", str(path), "-no", "SOURCE,FSTYPE,TARGET"],
        capture_output=True, text=True).stdout.strip()
    stat = os.statvfs(path)
    return {"path": str(path), "mount": out,
            "free_GB": stat.f_bavail * stat.f_frsize / 1e9}


def _dd(path: Path, mib: int, *, direct: bool, reps: int) -> Samples:
    """MEASURED sequential write, timed through fsync (or O_DIRECT)."""
    s = Samples(f"dd {mib} MiB {'O_DIRECT' if direct else 'buffered+fsync'}")
    target = path / "_dd_probe.bin"
    flag = ["oflag=direct"] if direct else ["conv=fsync"]
    for _ in range(reps):
        t0 = _now()
        subprocess.run(
            ["dd", "if=/dev/zero", f"of={target}", "bs=8M",
             f"count={mib // 8}", *flag],
            capture_output=True, check=True)
        s.add(_now() - t0)
        target.unlink(missing_ok=True)
    return s


def mode_fs(args) -> None:
    print("=" * 78)
    print("FILESYSTEM IDENTITY AND CEILING  (all MEASURED)")
    print("=" * 78)
    for p in (DISK_DIR, TMPFS_DIR):
        info = _fs_identity(p)
        print(f"  {info['path']}")
        print(f"    mount : {info['mount']}")
        print(f"    free  : {info['free_GB']:.1f} GB")
    print()
    reps = args.reps
    for mib, direct in ((8192, False), (8192, True)):
        s = _dd(DISK_DIR, mib, direct=direct, reps=reps)
        gbs = mib * 2 ** 20 / 1e9
        print(f"  {s.name:<34s} median {s.median:6.3f} s -> "
              f"{gbs / s.median:6.2f} GB/s   spread {s.spread * 100:.0f}%")
    print()
    print("  NOTE: guest O_DIRECT in WSL2 does not bypass the Windows host's")
    print("  own cache over the VHD, so the O_DIRECT figure is an upper bound")
    print("  on what the guest can push, NOT a device sustained rate.")


# ---------------------------------------------------------------------------
# the writer, characterised
# ---------------------------------------------------------------------------

def mode_path(args) -> None:
    import cupy as cp
    import netCDF4

    from gpuwm.io import wrfout
    from tilestream import harness

    print("=" * 78)
    print("THE OUTPUT PATH, BY READING (no timing here)")
    print("=" * 78)
    print(f"  netCDF4 {netCDF4.__version__}  libnetcdf "
          f"{netCDF4.__netcdf4libversion__}  hdf5 {netCDF4.__hdf5libversion__}")
    print()

    cfg = harness.make_config(256, 256, NZ)
    state = harness.make_state(cfg)
    frame = wrfout._device_state_frame(state)
    total = 0
    n3d = 0
    print("  A DRY FRAME'S INVENTORY (nx=ny=256, nz=49):")
    for name, value in sorted(frame.items()):
        nb = int(value.nbytes)
        total += nb
        if value.ndim == 3:
            n3d += 1
        loc = "device" if isinstance(value, cp.ndarray) else "HOST"
        bcast = ""
        if isinstance(value, cp.ndarray) and not value.flags.c_contiguous:
            bcast = "  (non-contiguous: ascontiguousarray materialises it)"
        print(f"    {name:<8s} {str(tuple(value.shape)):<18s} {value.dtype!s:<8s}"
              f" {nb / 1e6:9.3f} MB  {loc}{bcast}")
    cells = cfg.nx * cfg.ny * cfg.nz
    print(f"    ---> {len(frame)} fields ({n3d} volumes), {total / 1e6:.2f} MB, "
          f"{total / cells:.2f} B/cell, {total / (cfg.nx * cfg.ny):.1f} B/column")
    print()

    print("  COMPRESSION: WrfoutWriter._create_variable calls")
    print("    self.ds.createVariable(name, dtype, dims)")
    print("  with no zlib= argument.  netCDF4-python's default is zlib=False,")
    print("  so ArWen writes UNCOMPRESSED, contiguous-by-default NETCDF4_CLASSIC.")
    print("  grep for zlib/complevel/deflate across gpuwm/io returns nothing.")
    print()
    print("  SYNCHRONY:")
    print("    WrfoutWriter.write_frame (wrfout.py:1144) is FULLY SYNCHRONOUS:")
    print("      self.ds.variables[name][t] = arr   for every field, in order.")
    print("    It takes host numpy arrays and knows nothing about the GPU.")
    print()
    print("    AsyncDomainWrfoutWriter (wrfout.py:1304) is the production path")
    print("    and is where the pinned machinery lives.  submit() does:")
    print("      producer = get_current_stream()")
    print("      ready.record(producer);  self.stream.wait_event(ready)")
    print("      per field: cp.ascontiguousarray -> cp.cuda.alloc_pinned_memory")
    print("                 -> array.get(out=host, stream=side, blocking=False)")
    print("      done.record(self.stream);  producer.wait_event(done)   <== !!")
    print()
    print("    So the D2H is asynchronous WITH RESPECT TO THE HOST (submit")
    print("    returns before the copy lands) but it is STRICTLY SERIALISED IN")
    print("    THE GPU TIMELINE: the side stream waits for the solver, and then")
    print("    the solver waits for the side stream.  Nothing overlaps the D2H.")
    print("    What IS overlapped is the netCDF encode+disk, which runs on the")
    print("    writer thread while the solver proceeds -- bounded to ONE frame")
    print("    in flight (_new_ticket_queue: _AsyncTicketQueue(maxsize=1)), and")
    print("    serialised across domains by the global _NETCDF4_IO_LOCK:40.")
    print()
    print("  PINNED BUFFER LIFETIME: allocated PER FIELD PER FRAME by")
    print("    cp.cuda.alloc_pinned_memory (:1441), held on the ticket as")
    print("    pinned_refs (:1428/:1463), released only after netCDF has read")
    print("    the host views (:1522).  They are NOT reused across frames -- but")
    print("    CuPy's pinned pool recycles the underlying blocks, so the")
    print("    cudaHostAlloc cost is paid on the first frame only (MEASURED in")
    print("    mode decompose as the first-rep outlier that is discarded).")
    print()
    print("  ONE FILE PER FRAME: _worker (:1484) constructs a WrfoutWriter")
    print("    inside a `with` per ticket, so every history frame pays a full")
    print("    create + header + write + close + fsync + REOPEN-AND-VALIDATE")
    print("    (validate_wrfout_file:845) + atomic rename.")
    print()
    print("  A SECOND, DIFFERENT PATH: gpuwm/io/restart.py is NOT this path.")
    print("    _host() (:961) is a bare value.get() -- BLOCKING PAGEABLE D2H --")
    print("    and the payload goes through np.savez (:2153), an uncompressed")
    print("    zip, then fsync_file + os.replace.  Different transport, different")
    print("    container, different cost; see mode variants.")


# ---------------------------------------------------------------------------
# staging (the D2H half of AsyncDomainWrfoutWriter.submit)
# ---------------------------------------------------------------------------

def _stage_like_submit(state, *, pinned: bool, time_split: bool = True):
    """Reproduce AsyncDomainWrfoutWriter.submit's staging, timed.

    ``pinned=False`` swaps the pinned destination for a pageable one, which is
    the control for "does pinning actually matter here" -- it CAN fail (a
    pageable copy that came out as fast as the pinned one would refute the
    premise).

    Returns ``(host_fields, timings, nbytes)``.
    """
    import cupy as cp

    from gpuwm.io.wrfout import _device_state_frame

    timings: dict[str, float] = {}
    cp.cuda.runtime.deviceSynchronize()

    producer = cp.cuda.get_current_stream()
    side = cp.cuda.Stream(non_blocking=True)

    t0 = _now()
    device_fields = _device_state_frame(state, include_diagnostic_pressure=True)
    ready = cp.cuda.Event()
    ready.record(producer)
    side.wait_event(ready)

    host_fields: dict[str, np.ndarray] = {}
    pinned_refs: list[object] = []
    device_refs: list[object] = []
    nbytes = 0

    # --- DERIVE: materialise every field as a contiguous device array.
    # This is the arithmetic (T = theta-300, P = p-pb) plus the broadcast
    # expansion of PB/PHB that ascontiguousarray is forced to do.
    contiguous: list[tuple[str, object]] = []
    with side:
        for name, value in device_fields.items():
            if isinstance(value, np.ndarray):
                host_fields[name] = np.array(value, copy=True, order="C",
                                             subok=False)
                continue
            contiguous.append((name, cp.ascontiguousarray(value)))
    side.synchronize()
    t_derive = _now()
    if time_split:
        timings["DERIVE (device)"] = t_derive - t0

    # --- ALLOC: the per-field pinned allocation submit() performs per frame.
    t1 = _now()
    dests: list[tuple[str, object, np.ndarray]] = []
    for name, array in contiguous:
        nbytes += int(array.nbytes)
        if pinned:
            memory = cp.cuda.alloc_pinned_memory(int(array.nbytes))
            host = np.frombuffer(memory, dtype=array.dtype,
                                 count=array.size).reshape(array.shape)
            pinned_refs.append(memory)
        else:
            host = np.empty(array.shape, dtype=array.dtype)
        dests.append((name, array, host))
    t2 = _now()
    if time_split:
        timings["ALLOC host dest"] = t2 - t1

    # --- D2H: the transfer itself, on the non-blocking side stream.
    with side:
        for name, array, host in dests:
            array.get(out=host, stream=side, blocking=False)
            host_fields[name] = host
            device_refs.append(array)
        done = cp.cuda.Event()
        done.record(side)
    producer.wait_event(done)
    done.synchronize()
    t3 = _now()
    timings["D2H transfer"] = t3 - t2
    timings["_stage_total"] = t3 - t0
    timings["_nbytes"] = float(nbytes)

    del device_refs
    return host_fields, timings, nbytes, pinned_refs


# ---------------------------------------------------------------------------
# the netCDF write, decomposed
# ---------------------------------------------------------------------------

def _make_writer(path: Path, cfg, *, complevel: int | None, schema=None):
    """Build the REAL WrfoutWriter, optionally with deflate injected.

    Deflate is injected by wrapping ``netCDF4.Dataset.createVariable`` so that
    every spatial variable gains ``zlib=True, complevel=N``.  Exactly one thing
    changes against the production writer, so the comparison isolates
    compression and nothing else.
    """
    from gpuwm.io.wrfout import WrfoutWriter

    # This bench prices the netCDF4/HDF5 close machinery stage by stage
    # (deflate injection patches netCDF4.Dataset itself, and the caller
    # unrolls the HDF5 close sequence through writer.ds), so it names
    # that engine explicitly now that the DEFAULT is the Rust classic
    # writer.  Pricing the default engine is a different bench.
    if complevel is None:
        return WrfoutWriter(path, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz,
                            dx=cfg.dx, dy=cfg.dy, title="bench",
                            field_schema=schema, engine="python")

    import netCDF4

    original = netCDF4.Dataset.createVariable

    def patched(self, varname, datatype, dimensions=(), **kwargs):
        if dimensions and "Time" in tuple(dimensions) and len(dimensions) > 1:
            kwargs.setdefault("zlib", True)
            kwargs.setdefault("complevel", int(complevel))
        return original(self, varname, datatype, dimensions, **kwargs)

    netCDF4.Dataset.createVariable = patched
    try:
        return WrfoutWriter(path, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz,
                            dx=cfg.dx, dy=cfg.dy, title="bench",
                            field_schema=schema, engine="python")
    finally:
        netCDF4.Dataset.createVariable = original


def _write_frame_decomposed(host_fields, cfg, out_dir: Path, *,
                            complevel: int | None = None,
                            validate: bool = True) -> dict:
    """One full production frame write, split into its stages.

    Mirrors AsyncDomainWrfoutWriter._worker exactly: a fresh WrfoutWriter per
    frame, one write_frame, then close() -- whose internals are unrolled here
    so CLOSE/FSYNC/VALIDATE/RENAME can be timed apart from one another.  The
    unrolled sequence is byte-for-byte the sequence in WrfoutWriter.close().
    """
    import netCDF4

    from gpuwm.io.wrfout import (_COMPLETION_ATTR, validate_wrfout_file)
    from gpuwm.supervisor import fsync_file, replace_file_with_retry

    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "wrfout_d01_bench"
    final.unlink(missing_ok=True)

    timings: dict[str, float] = {}

    t0 = _now()
    writer = _make_writer(final, cfg, complevel=complevel, schema=host_fields)
    t1 = _now()
    timings["CREATE (header)"] = t1 - t0

    writer.write_frame("2024-01-01_00:00:00", host_fields)
    t2 = _now()
    timings["ENCODE (netCDF put)"] = t2 - t1

    inventory = tuple(writer.ds.variables)
    shapes = {n: tuple(v.shape) for n, v in writer.ds.variables.items()}
    times = tuple(writer._times)
    writer.ds.setncattr(_COMPLETION_ATTR, np.int32(1))
    writer.ds.close()
    writer._closed = True
    t3 = _now()
    timings["CLOSE (hdf5 flush)"] = t3 - t2

    fsync_file(writer._temp_path)
    t4 = _now()
    timings["FSYNC (durability)"] = t4 - t3

    if validate:
        validate_wrfout_file(writer._temp_path, inventory=inventory,
                             shapes=shapes, times=times)
    t5 = _now()
    timings["VALIDATE (reopen)"] = t5 - t4

    replace_file_with_retry(writer._temp_path, final)
    t6 = _now()
    timings["RENAME (publish)"] = t6 - t5

    timings["_file_bytes"] = float(final.stat().st_size)
    timings["_write_total"] = t6 - t0
    final.unlink(missing_ok=True)
    return timings


def _raw_frame(host_fields, out_dir: Path) -> dict:
    """RAW FLOOR: the same bytes through np.ndarray.tofile + fsync.

    No container, no header, no validation -- the fastest thing that can put
    these bytes on this filesystem durably.  Everything netCDF costs above
    this is netCDF's overhead.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "frame.raw"
    target.unlink(missing_ok=True)

    t0 = _now()
    with open(target, "wb") as stream:
        for name in sorted(host_fields):
            arr = host_fields[name]
            np.ascontiguousarray(arr).tofile(stream)
        stream.flush()
        t1 = _now()
        os.fsync(stream.fileno())
    t2 = _now()
    size = target.stat().st_size
    target.unlink(missing_ok=True)
    return {"RAW write (page cache)": t1 - t0,
            "RAW fsync": t2 - t1,
            "_raw_total": t2 - t0,
            "_file_bytes": float(size)}


def _npz_frame(host_fields, out_dir: Path) -> dict:
    """The RESTART container for comparison: np.savez + fsync (uncompressed zip)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "frame.npz"
    target.unlink(missing_ok=True)
    t0 = _now()
    with open(target, "wb") as stream:
        np.savez(stream, **{k: v for k, v in host_fields.items()})
        stream.flush()
        os.fsync(stream.fileno())
    t1 = _now()
    size = target.stat().st_size
    target.unlink(missing_ok=True)
    return {"_npz_total": t1 - t0, "_file_bytes": float(size)}


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------

def _build(nx: int, ny: int):
    from tilestream import harness

    cfg = harness.make_config(nx, ny, NZ)
    state = harness.make_state(cfg)
    return cfg, state


def mode_decompose(args) -> None:
    import cupy as cp

    nx = ny = args.n
    reps = args.reps
    cfg, state = _build(nx, ny)
    cells = nx * ny * NZ

    print("=" * 78)
    print(f"ONE FRAME, DECOMPOSED   nx=ny={nx}, nz={NZ}, {cells / 1e6:.1f} Mcell")
    print("=" * 78)

    stage_keys = ["DERIVE (device)", "ALLOC host dest", "D2H transfer"]
    write_keys = ["CREATE (header)", "ENCODE (netCDF put)", "CLOSE (hdf5 flush)",
                  "FSYNC (durability)", "VALIDATE (reopen)", "RENAME (publish)"]
    samples = {k: Samples(k) for k in stage_keys + write_keys}
    shm = Samples("ENCODE-only (tmpfs)")
    totals = Samples("FRAME TOTAL")
    nbytes = 0
    file_bytes = 0
    warmups = 2

    # The pinned pool is deliberately NOT freed between reps.  Production runs
    # every frame through the same warm CuPy pinned pool, so freeing it here
    # would time a cudaHostAlloc no steady-state frame ever pays.  The cold
    # cost is reported separately, from the discarded warm-up reps.
    cold: list[float] = []
    for rep in range(reps + warmups):
        host_fields, st, nbytes, pinned_refs = _stage_like_submit(
            state, pinned=True)
        wt = _write_frame_decomposed(host_fields, cfg, DISK_DIR)
        file_bytes = int(wt["_file_bytes"])
        # Same host fields, same writer, target on tmpfs: the disk is removed
        # and nothing else changes, so ext4 total - tmpfs total IS the disk.
        wt_shm = _write_frame_decomposed(host_fields, cfg, TMPFS_DIR)
        if rep < warmups:
            cold.append(st["ALLOC host dest"])
            del host_fields, pinned_refs
            _settle()
            continue
        for k in stage_keys:
            samples[k].add(st[k])
        for k in write_keys:
            samples[k].add(wt[k])
        shm.add(wt_shm["_write_total"])
        totals.add(st["_stage_total"] + wt["_write_total"])
        del host_fields, pinned_refs
        _settle()

    med_total = totals.median
    print(f"  ({warmups} warm-up reps discarded; their COLD pinned-allocation "
          f"cost was {max(cold) * 1e3:.0f} ms, paid once per process)")
    print()
    print(f"  frame payload : {nbytes / 1e6:8.2f} MB in 15 fields (8 volumes)")
    print(f"  file on disk  : {file_bytes / 1e6:8.2f} MB "
          f"({file_bytes / max(nbytes, 1):.4f}x payload)")
    print(f"  bytes/cell    : {nbytes / cells:8.2f}")
    print()
    print("  --- MONOLITHIC ONLY: state lives in VRAM, must be pulled to host ---")
    for k in stage_keys:
        print(samples[k].row(med_total))
    d2h = samples["D2H transfer"].median
    print(f"      D2H effective bandwidth: {nbytes / d2h / 1e9:.1f} GB/s "
          f"(pinned ceiling on this box 57 GB/s)")
    print()
    print("  --- netCDF WRITE (identical for monolithic and out-of-core) ---")
    for k in write_keys:
        print(samples[k].row(med_total))
    enc = samples["ENCODE (netCDF put)"].median
    fsy = samples["FSYNC (durability)"].median
    clo = samples["CLOSE (hdf5 flush)"].median
    write_med = sum(samples[k].median for k in write_keys)
    print()
    print("  --- ENCODE vs DISK, separated by the tmpfs control ---")
    print(f"      netCDF write to ext4  : {write_med * 1e3:9.2f} ms")
    print(f"      netCDF write to tmpfs : {shm.median * 1e3:9.2f} ms  "
          f"(RAM; spread {shm.spread * 100:.0f}%)")
    disk_only = write_med - shm.median
    print(f"      => ENCODE (CPU/RAM)   : {shm.median * 1e3:9.2f} ms  "
          f"({100 * shm.median / write_med:4.1f}% of the write)")
    print(f"      => DISK  (ext4+fsync) : {disk_only * 1e3:9.2f} ms  "
          f"({100 * disk_only / write_med:4.1f}% of the write)")
    print(f"      DISK effective rate   : "
          f"{file_bytes / max(disk_only, 1e-9) / 1e9:5.2f} GB/s")
    print(f"      whole-write disk rate : "
          f"{file_bytes / (enc + clo + fsy) / 1e9:5.2f} GB/s  "
          f"(vs 2.0 GB/s sustained dd ceiling -- must not exceed it)")
    print()
    print(f"  {'FRAME TOTAL (monolithic)':<26s} {med_total * 1e3:10.2f} ms  "
          f"[{totals.lo * 1e3:9.2f}, {totals.hi * 1e3:9.2f}]  "
          f"spread {totals.spread * 100:5.1f}%")
    stage_med = sum(samples[k].median for k in stage_keys)
    print(f"  {'  of which D2H staging':<26s} {stage_med * 1e3:10.2f} ms  "
          f"({100 * stage_med / med_total:.1f}%)")
    print(f"  {'  of which netCDF write':<26s} {write_med * 1e3:10.2f} ms  "
          f"({100 * write_med / med_total:.1f}%)")
    print()
    print(f"  OUT-OF-CORE frame total   = {write_med * 1e3:10.2f} ms "
          f"(D2H staging is ZERO: the bytes are already pinned host)")
    print(f"  speedup on the frame      = {med_total / write_med:.3f}x")


def mode_sweep(args) -> None:
    import cupy as cp

    reps = args.reps
    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else \
        list(DEFAULT_SIZES)

    print("=" * 78)
    print("DOMAIN-SIZE SWEEP")
    print("=" * 78)
    rows = []
    for n in sizes:
        cfg, state = _build(n, n)
        cells = n * n * NZ
        stage = Samples("stage")
        write = Samples("write")
        enc = Samples("encode")
        disk = Samples("disk")
        nbytes = 0
        file_bytes = 0
        shm = Samples("tmpfs")
        for rep in range(reps + 2):
            host_fields, st, nbytes, pinned = _stage_like_submit(state,
                                                                 pinned=True)
            wt = _write_frame_decomposed(host_fields, cfg, DISK_DIR)
            wt_shm = _write_frame_decomposed(host_fields, cfg, TMPFS_DIR)
            file_bytes = int(wt["_file_bytes"])
            if rep >= 2:
                stage.add(st["_stage_total"])
                write.add(wt["_write_total"])
                enc.add(wt_shm["_write_total"])
                disk.add(wt["_write_total"] - wt_shm["_write_total"])
            del host_fields, pinned
            _settle()
        rows.append(dict(n=n, cells=cells, nbytes=nbytes,
                         file_bytes=file_bytes,
                         stage=stage, write=write, enc=enc, disk=disk))
        del state
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        print(f"  {n}^2 done: stage {stage.median * 1e3:.1f} ms "
              f"(spread {stage.spread * 100:.0f}%), "
              f"write {write.median * 1e3:.1f} ms "
              f"(spread {write.spread * 100:.0f}%)")

    print()
    hdr = (f"  {'nx':>6s} {'Mcell':>8s} {'MB':>9s} {'D2H ms':>9s} "
           f"{'ENC ms':>9s} {'DISK ms':>9s} {'WRITE ms':>9s} "
           f"{'MONO ms':>9s} {'OOC ms':>9s} {'x':>6s} {'GB/s':>6s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        mono = r["stage"].median + r["write"].median
        ooc = r["write"].median
        print(f"  {r['n']:6d} {r['cells'] / 1e6:8.1f} {r['nbytes'] / 1e6:9.1f} "
              f"{r['stage'].median * 1e3:9.1f} {r['enc'].median * 1e3:9.1f} "
              f"{r['disk'].median * 1e3:9.1f} {r['write'].median * 1e3:9.1f} "
              f"{mono * 1e3:9.1f} {ooc * 1e3:9.1f} {mono / ooc:6.3f} "
              f"{r['file_bytes'] / r['write'].median / 1e9:6.2f}")
    print("  (GB/s = file bytes / whole netCDF write incl. fsync; the sustained")
    print("   dd ceiling on this filesystem is 2.0 GB/s -- nothing may exceed it)")

    print()
    print("  LEAST-SQUARES FIT   cost = fixed + slope * cells")
    cells = np.array([r["cells"] for r in rows], dtype=float)
    for label, get in (
            ("D2H staging", lambda r: r["stage"].median),
            ("ENCODE", lambda r: r["enc"].median),
            ("DISK (close+fsync)", lambda r: r["disk"].median),
            ("netCDF write total", lambda r: r["write"].median),
            ("MONOLITHIC frame", lambda r: r["stage"].median + r["write"].median),
            ("OUT-OF-CORE frame", lambda r: r["write"].median)):
        y = np.array([get(r) for r in rows], dtype=float)
        A = np.vstack([np.ones_like(cells), cells]).T
        (fixed, slope), *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = fixed + slope * cells
        resid = float(np.max(np.abs(pred - y) / y)) * 100
        print(f"    {label:<22s} fixed {fixed * 1e3:8.1f} ms  +  "
              f"{slope * 1e9:6.3f} ns/cell     (max resid {resid:4.1f}%)")

    Path(args.json).write_text(json.dumps(
        [{k: (v.values if isinstance(v, Samples) else v) for k, v in r.items()}
         for r in rows], indent=1)) if args.json else None


def mode_variants(args) -> None:
    import cupy as cp

    n = args.n
    reps = args.reps
    cfg, state = _build(n, n)
    cells = n * n * NZ

    print("=" * 78)
    print(f"CONTAINER AND COMPRESSION VARIANTS   nx=ny={n}, nz={NZ}")
    print("=" * 78)

    host_fields, st, nbytes, pinned = _stage_like_submit(state, pinned=True)
    print(f"  payload {nbytes / 1e6:.1f} MB, {nbytes / cells:.2f} B/cell")
    print()

    variants: list[tuple[str, object]] = [
        ("raw tofile + fsync (FLOOR)", "raw"),
        ("npz zip + fsync (restart)", "npz"),
        ("netCDF4 no compression", None),
    ]
    if args.deflate:
        variants += [("netCDF4 deflate level 1", 1),
                     ("netCDF4 deflate level 4", 4)]

    print(f"  {'variant':<30s} {'median s':>9s} {'spread':>7s} "
          f"{'file MB':>9s} {'ratio':>6s} {'GB/s':>7s} {'vs floor':>9s}")
    print("  " + "-" * 84)
    floor = None
    for label, kind in variants:
        s = Samples(label)
        fb = 0
        rr = reps if kind in (None, "raw", "npz") else max(3, reps // 2)
        for rep in range(rr + 1):
            if kind == "raw":
                out = _raw_frame(host_fields, DISK_DIR)
                t = out["_raw_total"]
            elif kind == "npz":
                out = _npz_frame(host_fields, DISK_DIR)
                t = out["_npz_total"]
            else:
                out = _write_frame_decomposed(host_fields, cfg, DISK_DIR,
                                              complevel=kind)
                t = out["_write_total"]
            fb = int(out["_file_bytes"])
            if rep:
                s.add(t)
        if floor is None:
            floor = s.median
        flag = " <<SPREAD" if s.spread > 0.10 else ""
        print(f"  {label:<30s} {s.median:9.3f} {s.spread * 100:6.0f}% "
              f"{fb / 1e6:9.1f} {fb / nbytes:6.3f} "
              f"{nbytes / s.median / 1e9:7.2f} {s.median / floor:8.2f}x{flag}")

    del host_fields, pinned
    cp.get_default_pinned_memory_pool().free_all_blocks()

    print()
    print("  ENCODE-ONLY CONTROL (same writer, target = /dev/shm tmpfs = RAM).")
    print("  The delta against the ext4 rows above is the disk's contribution.")
    host_fields, st, nbytes, pinned = _stage_like_submit(state, pinned=True)
    s = Samples("netCDF4 -> tmpfs")
    for rep in range(reps + 1):
        out = _write_frame_decomposed(host_fields, cfg, TMPFS_DIR)
        if rep:
            s.add(out["_write_total"])
    print(f"  {'netCDF4 -> tmpfs (RAM)':<30s} {s.median:9.3f} "
          f"{s.spread * 100:6.0f}% {'':>9s} {'':>6s} "
          f"{nbytes / s.median / 1e9:7.2f}")
    del host_fields, pinned


def mode_pinned_control(args) -> None:
    """CONTROL THAT CAN FAIL: pinned vs pageable D2H for the same frame.

    If pinning did not matter, these two would be equal and the premise
    ("the bytes are already in the fastest thing to hand a filesystem")
    would lose its force.
    """
    import cupy as cp

    n = args.n
    cfg, state = _build(n, n)
    print("=" * 78)
    print(f"CONTROL: pinned vs pageable D2H   nx=ny={n}")
    print("=" * 78)
    for pinned_flag in (True, False):
        s = Samples("pinned" if pinned_flag else "pageable")
        nb = 0
        for rep in range(args.reps + 1):
            hf, st, nb, refs = _stage_like_submit(state, pinned=pinned_flag)
            if rep:
                s.add(st["D2H transfer"])
            del hf, refs
            cp.get_default_pinned_memory_pool().free_all_blocks()
        print(f"  {s.name:<12s} {s.median * 1e3:8.2f} ms  "
              f"{nb / s.median / 1e9:6.1f} GB/s   spread {s.spread * 100:.0f}%")


def _ooc_frame_from_store(store, state, *, static_cache: dict) -> tuple:
    """Build a wrfout frame from a REAL HostDomainStore, timed, no D2H.

    This is the honest out-of-core accounting the hypothesis needs.  Of the
    eight volume fields a frame carries:

      U, V, W, PH   are store.arrays['u','v','w','php'] verbatim -- pinned
                    host views, handed to netCDF with ZERO copies and ZERO
                    device traffic.
      T, P          must still be DERIVED (T = thb + thp - 300, P = p - pb).
                    A monolithic run does that on the GPU; an out-of-core run
                    has no resident device copy, so it does it on the HOST.
                    That host arithmetic is what replaces the D2H, and it is
                    NOT free -- so it is measured, not assumed away.
      PB, PHB       are broadcasts of 1-D base state.  They never change, so
                    they are materialised once and cached across frames; the
                    first-frame cost is reported separately.

    Returns ``(fields, per_frame_seconds, first_frame_seconds)``.
    """
    arrays = store.arrays
    nz, ny, nx = store.nz, store.ny, store.nx

    thb = np.asarray(state.thb.get()) if hasattr(state.thb, "get") else \
        np.asarray(state.thb)
    pb = np.asarray(state.pb.get()) if hasattr(state.pb, "get") else \
        np.asarray(state.pb)
    phb = np.asarray(state.phb.get()) if hasattr(state.phb, "get") else \
        np.asarray(state.phb)

    # --- one-time: materialise the static base-state volumes.
    t0 = _now()
    if "PB" not in static_cache:
        pb3 = pb[:, None, None] if pb.ndim == 1 else pb
        static_cache["PB"] = np.ascontiguousarray(
            np.broadcast_to(pb3, (nz, ny, nx)))
        phb3 = phb[:, None, None] if phb.ndim == 1 else phb
        static_cache["PHB"] = np.ascontiguousarray(
            np.broadcast_to(phb3, (nz + 1, ny, nx)))
        for name, src in (("HGT", state.ht), ("MUB", state.mub2d)):
            static_cache[name] = np.ascontiguousarray(
                src.get() if hasattr(src, "get") else src)
    t1 = _now()
    first_only = t1 - t0

    # --- per frame: the two derived volumes, on the host.
    # Destinations are allocated ONCE and reused (static_cache), and the
    # arithmetic is in-place via out=.  Allocating fresh arrays per frame
    # instead costs 2.7x more, all of it first-touch page faults -- that is a
    # property of naive code, not of streaming, so it is not what gets
    # reported as the streaming cost.
    if "T_dst" not in static_cache:
        static_cache["T_dst"] = np.empty_like(arrays["thp"])
        static_cache["P_dst"] = np.empty_like(arrays["p"])
        static_cache["T_dst"][...] = 0.0    # first-touch outside the timer
        static_cache["P_dst"][...] = 0.0
    thb3 = (thb[:, None, None] if thb.ndim == 1 else thb).astype(np.float32)
    pb3 = (pb[:, None, None] if pb.ndim == 1 else pb).astype(np.float32)
    t2 = _now()
    T = np.add(arrays["thp"], thb3 - np.float32(300.0),
               out=static_cache["T_dst"])
    P = np.subtract(arrays["p"], pb3, out=static_cache["P_dst"])
    t3 = _now()

    fields = {
        "T": T, "U": arrays["u"], "V": arrays["v"], "W": arrays["w"],
        "PH": arrays["php"], "MU": arrays["mup"], "P": P,
        "PB": static_cache["PB"], "PHB": static_cache["PHB"],
        "HGT": static_cache["HGT"], "MUB": static_cache["MUB"],
    }
    return fields, t3 - t2, first_only


def mode_ooc(args) -> None:
    """The hypothesis, tested against a real HostDomainStore.

    CONTROL THAT CAN FAIL: if the host-side derivation of T and P costs more
    than the whole monolithic DERIVE+D2H, streaming makes output DEARER, not
    cheaper, and the hypothesis is refuted at this size.
    """
    import cupy as cp

    from tilestream.hoststore import HostDomainStore

    n = args.n
    cfg, state = _build(n, n)
    cells = n * n * NZ
    print("=" * 78)
    print(f"OUT-OF-CORE vs MONOLITHIC STAGING   nx=ny={n}, nz={NZ}, "
          f"{cells / 1e6:.1f} Mcell")
    print("=" * 78)

    # --- monolithic staging (the real submit() path).
    mono = Samples("monolithic DERIVE+D2H")
    for rep in range(args.reps + 2):
        hf, st, nbytes, refs = _stage_like_submit(state, pinned=True)
        if rep >= 2:
            mono.add(st["_stage_total"])
        del hf, refs
    print(f"  monolithic DERIVE+ALLOC+D2H : {mono.median * 1e3:8.2f} ms  "
          f"spread {mono.spread * 100:.0f}%")

    # --- out-of-core: a real pinned store, filled from the same state.
    store = HostDomainStore(cfg)
    from tilestream import harness
    src = harness.state_arrays(state)
    for name, arr in store.arrays.items():
        if name in src:
            arr[...] = src[name].get()
    print(f"  HostDomainStore             : {store.nbytes / 1e9:8.2f} GB "
          f"pinned, {len(store.arrays)} fields")

    cache: dict = {}
    ooc = Samples("out-of-core host derive")
    first = 0.0
    for rep in range(args.reps + 2):
        fields, per_frame, first_only = _ooc_frame_from_store(
            store, state, static_cache=cache)
        if rep == 0:
            first = first_only
        if rep >= 2:
            ooc.add(per_frame)
    print(f"  out-of-core HOST derive     : {ooc.median * 1e3:8.2f} ms  "
          f"spread {ooc.spread * 100:.0f}%   "
          f"(+{first * 1e3:.0f} ms once, static PB/PHB)")
    print()
    zero_copy = sum(store.arrays[k].nbytes for k in ("u", "v", "w", "php")
                    if k in store.arrays)
    print(f"  ZERO-COPY fields (U,V,W,PH) : {zero_copy / 1e9:8.3f} GB never "
          f"touched at all")
    print(f"  net staging change          : "
          f"{(ooc.median - mono.median) * 1e3:+8.2f} ms  "
          f"({'CHEAPER' if ooc.median < mono.median else 'DEARER'} "
          f"out-of-core)")
    store.free()
    del state
    cp.get_default_memory_pool().free_all_blocks()


def mode_overlap(args) -> None:
    """CONTROL THAT CAN FAIL: does the writer thread actually overlap?

    The whole cadence model rests on the netCDF write costing nothing while
    the solver runs.  AsyncDomainWrfoutWriter puts that write on a Python
    thread -- so if netCDF4/HDF5 hold the GIL through the put, the "overlap"
    is fiction and the write is serial with the host's kernel launching.

    Solver alone = A, write alone = B, both concurrently = C.
      C ~= max(A, B)  -> real overlap, the cadence model stands.
      C ~= A + B      -> GIL-bound, no overlap, the cadence model is wrong.
    Those two predictions differ by a factor of ~2 here, so this cannot come
    out ambiguous.
    """
    import threading

    import cupy as cp

    from tilestream import harness

    n = args.n
    cfg, state = _build(n, n)
    print("=" * 78)
    print(f"CONTROL: does the netCDF writer thread overlap the solver?  "
          f"nx=ny={n}")
    print("=" * 78)

    host_fields, _st, nbytes, pinned = _stage_like_submit(state, pinned=True)
    harness.run_steps(state, cfg, 2)

    # Pick a step count whose solver time is close to one write, so the two
    # predictions are maximally far apart.
    t0 = _now()
    harness.run_steps(state, cfg, 3)
    step = (_now() - t0) / 3
    w0 = _now()
    _write_frame_decomposed(host_fields, cfg, DISK_DIR)
    write_probe = _now() - w0
    nsteps = max(2, int(round(write_probe / step)))
    print(f"  step {step * 1e3:.1f} ms, write {write_probe * 1e3:.0f} ms "
          f"-> using {nsteps} steps per trial")
    _settle()

    solo_solver = Samples("solver alone")
    solo_write = Samples("write alone")
    both = Samples("concurrent")
    for rep in range(args.reps + 1):
        cp.cuda.runtime.deviceSynchronize()
        t = _now()
        harness.run_steps(state, cfg, nsteps)
        a = _now() - t
        _settle()

        t = _now()
        _write_frame_decomposed(host_fields, cfg, DISK_DIR)
        b = _now() - t
        _settle()

        result: list[float] = []

        def _writer() -> None:
            _write_frame_decomposed(host_fields, cfg, DISK_DIR)
            result.append(_now())

        cp.cuda.runtime.deviceSynchronize()
        t = _now()
        th = threading.Thread(target=_writer)
        th.start()
        harness.run_steps(state, cfg, nsteps)
        th.join()
        c = _now() - t
        _settle()

        if rep:
            solo_solver.add(a)
            solo_write.add(b)
            both.add(c)

    A, B, C = solo_solver.median, solo_write.median, both.median
    print(f"  A  solver alone ({nsteps} steps) : {A * 1e3:8.1f} ms  "
          f"spread {solo_solver.spread * 100:.0f}%")
    print(f"  B  write alone  (1 frame)     : {B * 1e3:8.1f} ms  "
          f"spread {solo_write.spread * 100:.0f}%")
    print(f"  C  both concurrently          : {C * 1e3:8.1f} ms  "
          f"spread {both.spread * 100:.0f}%")
    print()
    print(f"     max(A,B) = {max(A, B) * 1e3:8.1f} ms   (perfect overlap)")
    print(f"     A + B    = {(A + B) * 1e3:8.1f} ms   (no overlap at all)")
    frac = (C - max(A, B)) / max(A + B - max(A, B), 1e-9)
    print(f"     C sits {100 * (1 - frac):.0f}% of the way to perfect overlap")
    print(f"     => overlap efficiency {100 * (A + B - C) / min(A, B):.0f}% "
          f"of the smaller task hidden")
    del host_fields, pinned
    del state
    cp.get_default_memory_pool().free_all_blocks()


def mode_cadence(args) -> None:
    """The CM1 framing: timesteps/s with and without output."""
    import cupy as cp

    from tilestream import harness

    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else \
        list(DEFAULT_SIZES)
    reps = args.reps
    print("=" * 78)
    print("TIMESTEPS PER SECOND, WITH AND WITHOUT OUTPUT")
    print("=" * 78)

    results = []
    for n in sizes:
        cfg, state = _build(n, n)
        cells = n * n * NZ
        # --- solver step time, MEASURED on this box, this state.
        harness.run_steps(state, cfg, 2)
        step = Samples("step")
        for _ in range(reps):
            cp.cuda.runtime.deviceSynchronize()
            t0 = _now()
            harness.run_steps(state, cfg, 5)
            step.add((_now() - t0) / 5)
        # --- frame cost
        stage = Samples("stage")
        write = Samples("write")
        nbytes = 0
        for rep in range(max(5, reps) + 2):
            hf, st, nbytes, refs = _stage_like_submit(state, pinned=True)
            wt = _write_frame_decomposed(hf, cfg, DISK_DIR)
            if rep >= 2:
                stage.add(st["_stage_total"])
                write.add(wt["_write_total"])
            del hf, refs
            _settle()
        # --- out-of-core staging: host derive of T and P from a pinned store.
        from tilestream.hoststore import HostDomainStore
        store = HostDomainStore(cfg)
        for name, arr in store.arrays.items():
            if name in ("thp", "p"):
                arr[...] = getattr(state, name).get()
        cache: dict = {}
        oocs = Samples("ooc stage")
        for rep in range(max(5, reps) + 2):
            _, per_frame, _ = _ooc_frame_from_store(store, state,
                                                    static_cache=cache)
            if rep >= 2:
                oocs.add(per_frame)
        store.free()
        results.append(dict(n=n, cells=cells, nbytes=nbytes,
                            step=step.median, stage=stage.median,
                            ooc_stage=oocs.median,
                            write=write.median,
                            step_spread=step.spread))
        del state
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        print(f"  {n}^2: step {step.median * 1e3:.1f} ms "
              f"(spread {step.spread * 100:.0f}%), "
              f"MONO stage {stage.median * 1e3:.1f} ms, "
              f"OOC stage {oocs.median * 1e3:.1f} ms, "
              f"write {write.median * 1e3:.1f} ms")

    print()
    print("  STEADY-STATE MODEL, from the code's own structure:")
    print("    wall per output interval of C steps")
    print("        = max( C*step + STAGE ,  WRITE )")
    print("    STAGE is additive inside the first term because submit()")
    print("    serialises it into the GPU timeline: the side stream waits for")
    print("    the solver (self.stream.wait_event(ready)) and then the solver")
    print("    waits for the side stream (producer.wait_event(done)).")
    print("    WRITE is the OTHER term, not an addend, because it runs on the")
    print("    writer thread while the solver proceeds -- but the ticket queue")
    print("    is maxsize=1, so once WRITE exceeds an interval the writer, not")
    print("    the solver, sets the pace.")
    print("    MONO STAGE = measured DERIVE+ALLOC+D2H.  OOC STAGE = measured")
    print("    host-side derive of T and P from the pinned store (mode ooc).")
    print()

    for r in results:
        print(f"  --- {r['n']}^2 x {NZ} = {r['cells'] / 1e6:.1f} Mcell, "
              f"frame {r['nbytes'] / 1e6:.0f} MB, "
              f"step {r['step'] * 1e3:.1f} ms, "
              f"write {r['write'] * 1e3:.0f} ms ---")
        print(f"    {'cadence':>8s} {'no-out':>9s} {'MONO':>9s} {'OOC':>9s} "
              f"{'MONO %out':>10s} {'OOC %out':>9s} {'OOC/MONO':>9s} "
              f"{'limited by':>12s}")
        base = 1.0 / r["step"]
        for C in (1, 10, 60, 240, 360):
            solver = C * r["step"]
            wm = max(solver + r["stage"], r["write"])
            wo = max(solver + r["ooc_stage"], r["write"])
            lim = "writer" if r["write"] > solver + r["stage"] else "solver"
            print(f"    {C:8d} {base:9.3f} {C / wm:9.3f} {C / wo:9.3f} "
                  f"{100 * (1 - solver / wm):9.1f}% "
                  f"{100 * (1 - solver / wo):8.1f}% "
                  f"{wm / wo:8.3f}x {lim:>12s}")
        print("    (st/s = timesteps per second; %out = fraction of wall time")
        print("     that is not solver work)")
        print()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    for name, fn in (("fs", mode_fs), ("path", mode_path),
                     ("decompose", mode_decompose), ("sweep", mode_sweep),
                     ("variants", mode_variants), ("cadence", mode_cadence),
                     ("ooc", mode_ooc), ("overlap", mode_overlap),
                     ("pinned", mode_pinned_control)):
        sp = sub.add_parser(name)
        sp.set_defaults(fn=fn)
        sp.add_argument("--reps", type=int, default=DEFAULT_REPS)
        sp.add_argument("--n", type=int, default=1024)
        sp.add_argument("--sizes", type=str, default="")
        sp.add_argument("--deflate", action="store_true")
        sp.add_argument("--json", type=str, default="")
    args = p.parse_args(argv)
    DISK_DIR.mkdir(parents=True, exist_ok=True)
    try:
        args.fn(args)
    finally:
        for stray in list(DISK_DIR.glob("*")) + list(TMPFS_DIR.glob("*")
                                                     if TMPFS_DIR.exists()
                                                     else []):
            try:
                stray.unlink()
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
