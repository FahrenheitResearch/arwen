"""Does out-of-core streaming make OUTPUT cheaper?  Measured end to end.

Phase 1 (``tilestream.bench_output``) decomposed one monolithic frame and
established the ceilings: the filesystem, the D2H, the netCDF encode, the
overlap control.  It stopped one step short of the hypothesis, in two ways
that this module exists to close:

1. It timed the out-of-core *staging* (a host derive of T and P) against the
   monolithic *staging* (DERIVE + pinned alloc + D2H) and then ASSUMED the
   netCDF write that follows is identical for both.  It never actually ran a
   frame through ``WrfoutWriter`` out of a :class:`~tilestream.hoststore.
   HostDomainStore`.  Handing netCDF a long-lived pinned view is not obviously
   the same as handing it a freshly-pooled pinned buffer, so that is measured
   here rather than assumed.

2. It never checked that the two paths write the SAME BYTES.  They do not.
   See :func:`mode_verify`.

The third thing this module adds is the question Phase 1 could not ask,
because it never ran the driver: in a tiled run the store is host-resident
continuously, so can a frame be written WHILE the GPU steps?  That depends
entirely on ``run_tiled``'s read/write discipline, and the answer differs
between the two write modes -- see :func:`mode_tear`.

    python -m tilestream.bench_ooc_output <mode> [--n N] [--reps R]

Modes: verify, endtoend, tear, snapshot, concurrent, project, all.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from tilestream.bench_output import (DISK_DIR, NZ, Samples, _build, _now,
                                     _settle, _stage_like_submit)

G = 9.81
#: Running total of bytes this process has put on the filesystem, so the
#: 100 GB cap in the brief is an observed number and not a hope.
WRITTEN = [0]


def _note_written(nbytes: float) -> None:
    WRITTEN[0] += int(nbytes)


def gpu_busy() -> tuple[int, int]:
    """(MiB used on the device, % utilisation) -- other lanes share this card.

    Recorded next to every GPU-side figure.  A DERIVE timed while another
    process is holding the SMs is a measurement of that process, and this lane
    has already seen a 439 ms 'DERIVE' that is 7 ms on an idle card.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        mem, util = out.splitlines()[0].split(",")
        return int(mem), int(util)
    except Exception:
        return -1, -1


def _free_gb(path: Path = DISK_DIR) -> float:
    path.mkdir(parents=True, exist_ok=True)
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def _digest(arrays: dict) -> str:
    """SHA-256 over a {name: ndarray} inventory, in sorted name order."""
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(name.encode())
        h.update(a.dtype.str.encode())
        h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
        h.update(a.tobytes(order="C"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# the out-of-core frame, built two ways
# ---------------------------------------------------------------------------

def _statics(state, nz, ny, nx, cache: dict) -> float:
    """Materialise everything that never changes.  Returns the one-off cost.

    PB/PHB are broadcasts of a 1-D base state and MUB/HGT/ZNU/ZNW/P_TOP are
    setup arrays.  None of them is in the streaming carrier set, none changes
    between frames, so a real streaming writer holds them once.  Timed and
    reported separately so it cannot be smuggled into the per-frame figure.
    """
    t0 = _now()
    if "PB" not in cache:
        def host(x):
            return np.asarray(x.get() if hasattr(x, "get") else x)

        pb, phb, thb = host(state.pb), host(state.phb), host(state.thb)
        cache["_pb1"] = pb
        cache["_phb1"] = phb
        cache["_thb1"] = thb
        pb3 = pb[:, None, None] if pb.ndim == 1 else pb
        phb3 = phb[:, None, None] if phb.ndim == 1 else phb
        cache["PB"] = np.ascontiguousarray(np.broadcast_to(pb3, (nz, ny, nx)))
        cache["PHB"] = np.ascontiguousarray(
            np.broadcast_to(phb3, (nz + 1, ny, nx)))
        cache["MUB"] = np.ascontiguousarray(host(state.mub2d))
        cache["HGT"] = np.ascontiguousarray(host(state.ht))
        for name, attr in (("P_TOP", "p_top"), ("ZNU", "znu"), ("ZNW", "znw")):
            v = getattr(state, attr, None)
            if v is not None:
                cache[name] = np.asarray(host(v) if hasattr(v, "get")
                                         else np.float32(v))
        # Per-frame destinations, first-touched OUTSIDE any timer.  Phase 1
        # measured a 2.7x penalty for allocating these per frame; it is pure
        # first-touch page faulting and is a property of naive code, not of
        # streaming, so it is paid here and stated rather than hidden.
        cache["T_dst"] = np.zeros((nz, ny, nx), np.float32)
        cache["P_dst"] = np.zeros((nz, ny, nx), np.float32)
        cache["Z_dst"] = np.zeros((nz + 1, ny, nx), np.float32)
    return _now() - t0


def ooc_frame(store, state, cache: dict, *, exact: bool):
    """A full wrfout frame out of the pinned store, with NO device traffic.

    ``exact=True`` reproduces ``_device_state_frame``'s arithmetic operation
    for operation, so the bytes match a monolithic frame bit for bit.
    ``exact=False`` is Phase 1's reassociated form, kept because it is the
    obvious way to write this and it is WRONG -- see :func:`mode_verify`.

    Returns ``(fields, per_frame_seconds)``.
    """
    a = store.arrays
    nz, ny, nx = store.nz, store.ny, store.nx
    _statics(state, nz, ny, nx, cache)
    thb1, pb1, phb1 = cache["_thb1"], cache["_pb1"], cache["_phb1"]
    thb3 = thb1[:, None, None] if thb1.ndim == 1 else thb1
    pb3 = pb1[:, None, None] if pb1.ndim == 1 else pb1
    phb3 = phb1[:, None, None] if phb1.ndim == 1 else phb1

    t0 = _now()
    if exact:
        # T: the GPU does (thb + thp) then -300, in that order.  Float
        # addition is not associative, so the order is load-bearing.
        T = np.add(thb3, a["thp"], out=cache["T_dst"])
        np.subtract(T, np.float32(300.0), out=T)
    else:
        T = np.add(a["thp"], thb3 - np.float32(300.0), out=cache["T_dst"])
    P = np.subtract(a["p"], pb3, out=cache["P_dst"])

    # PSFC.  The device expression materialises the whole (nz+1, ny, nx)
    # geopotential before slicing three levels out of it; only levels 0..2
    # can reach the answer, so the host form slices FIRST.  That is a change
    # in work, not in arithmetic -- the surviving operations are identical
    # and so are the bytes (asserted in mode_verify).
    zif = np.add(phb3[:3], a["php"][:3], out=cache["Z_dst"][:3])
    np.divide(zif, np.float32(G), out=zif)
    zm0 = 0.5 * (zif[0] + zif[1])
    zm1 = 0.5 * (zif[1] + zif[2])
    w1 = (zif[0] - zm1) / (zm0 - zm1)
    psfc = w1 * a["p"][0] + (np.float32(1.0) - w1) * a["p"][1]
    t1 = _now()

    built = {
        "T": T, "U": a["u"], "V": a["v"], "W": a["w"], "PH": a["php"],
        "MU": a["mup"], "P": P, "PSFC": psfc,
        "PB": cache["PB"], "PHB": cache["PHB"],
        "MUB": cache["MUB"], "HGT": cache["HGT"],
    }
    for name in ("P_TOP", "ZNU", "ZNW"):
        if name in cache:
            built[name] = cache[name]

    # PRESENTATION ORDER IS LOAD-BEARING.  The dict doubles as the writer's
    # field schema, so it decides the order netCDF creates the variables in,
    # and HDF5's name heap is laid out in creation order.  Same data in a
    # different order gives a file that is 189 bytes larger and hashes
    # differently while every variable in it is identical -- MEASURED, and
    # constant at 189 bytes from 192^2 to 256^2.  Matching the device frame's
    # order is what turns "identical fields" into "identical file".
    order = tuple(ORDER) if ORDER else tuple(built)
    fields = {k: built[k] for k in order if k in built}
    for k in built:                                   # never silently drop
        fields.setdefault(k, built[k])
    return fields, t1 - t0


#: The order ``_device_state_frame`` yields its fields in.  Populated from the
#: real thing by :func:`adopt_order` so it cannot drift from the source.
ORDER: tuple[str, ...] = ()


def adopt_order(monolithic_fields) -> None:
    global ORDER
    ORDER = tuple(monolithic_fields)


def _fill_store(store, state):
    from tilestream import harness

    src = harness.state_arrays(state)
    for name, arr in store.arrays.items():
        if name in src:
            arr[...] = src[name].get()


# ---------------------------------------------------------------------------
# writing a frame and KEEPING it, so it can be checksummed
# ---------------------------------------------------------------------------

def write_frame_kept(host_fields, cfg, out_dir: Path, stem: str) -> dict:
    """One production frame write, left on disk, timed through fsync.

    Same unrolled sequence as Phase 1's ``_write_frame_decomposed`` (which is
    ``WrfoutWriter.close()`` opened up), except the file is not deleted --
    this module has to read it back and compare it.
    """
    import netCDF4  # noqa: F401  (imported for the side effect of a clear error)

    from gpuwm.io.wrfout import _COMPLETION_ATTR, validate_wrfout_file
    from gpuwm.supervisor import fsync_file, replace_file_with_retry
    from tilestream.bench_output import _make_writer

    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / stem
    final.unlink(missing_ok=True)

    t0 = _now()
    writer = _make_writer(final, cfg, complevel=None, schema=host_fields)
    t1 = _now()
    writer.write_frame("2024-01-01_00:00:00", host_fields)
    t2 = _now()
    inventory = tuple(writer.ds.variables)
    shapes = {n: tuple(v.shape) for n, v in writer.ds.variables.items()}
    times = tuple(writer._times)
    writer.ds.setncattr(_COMPLETION_ATTR, np.int32(1))
    writer.ds.close()
    writer._closed = True
    t3 = _now()
    fsync_file(writer._temp_path)
    t4 = _now()
    validate_wrfout_file(writer._temp_path, inventory=inventory,
                         shapes=shapes, times=times)
    t5 = _now()
    replace_file_with_retry(writer._temp_path, final)
    t6 = _now()
    size = float(final.stat().st_size)
    _note_written(size)
    return {"CREATE": t1 - t0, "ENCODE": t2 - t1, "CLOSE": t3 - t2,
            "FSYNC": t4 - t3, "VALIDATE": t5 - t4, "RENAME": t6 - t5,
            "_write_total": t6 - t0, "_file_bytes": size, "_path": final}


def file_var_digests(path: Path) -> dict[str, str]:
    """Per-variable SHA-256 of a wrfout, so timestamps cannot mask a diff."""
    import netCDF4

    out: dict[str, str] = {}
    with netCDF4.Dataset(path, "r") as ds:
        for name, var in ds.variables.items():
            data = var[:]
            arr = np.ascontiguousarray(
                np.ma.getdata(data) if np.ma.isMaskedArray(data) else data)
            out[name] = hashlib.sha256(arr.tobytes(order="C")).hexdigest()
    return out


# ---------------------------------------------------------------------------
# 1. CORRECTNESS.  Mandatory, and it fails.
# ---------------------------------------------------------------------------

def mode_verify(args) -> None:
    """Do the two paths write the same bytes?

    A CONTROL THAT CAN FAIL, and the point of the whole exercise: an output
    "optimisation" that writes different data is a bug, not a speedup.  Three
    comparisons, in increasing strength:

      field level   monolithic host_fields vs out-of-core fields, bitwise
      file level    per-variable SHA-256 read back out of the two netCDFs
      exactness     the same, with the reassociated (Phase 1) arithmetic,
                    which is what the comparison has to be able to catch
    """
    import cupy as cp

    from tilestream.hoststore import HostDomainStore

    n = args.n
    cfg, state = _build(n, n)
    print("=" * 78)
    print(f"CORRECTNESS: monolithic bytes vs out-of-core bytes   nx=ny={n}, "
          f"nz={NZ}")
    print("=" * 78)

    host_fields, _st, _nb, pinned = _stage_like_submit(state, pinned=True)
    adopt_order(host_fields)
    store = HostDomainStore(cfg)
    _fill_store(store, state)

    for label, exact in (("EXACT (same op order)", True),
                         ("REASSOCIATED (Phase 1 form)", False)):
        cache: dict = {}
        fields, _dt = ooc_frame(store, state, cache, exact=exact)
        print(f"\n  --- {label} ---")
        common = sorted(set(fields) & set(host_fields))
        missing = sorted(set(host_fields) - set(fields))
        extra = sorted(set(fields) - set(host_fields))
        bad = []
        for name in common:
            a = np.ascontiguousarray(host_fields[name])
            b = np.ascontiguousarray(fields[name])
            if a.shape != b.shape or a.dtype != b.dtype:
                bad.append((name, "shape/dtype", np.nan, 0))
                continue
            same = np.array_equal(a.view(np.uint8), b.view(np.uint8))
            if not same:
                d = np.abs(a.astype(np.float64) - b.astype(np.float64))
                nd = int(np.count_nonzero(a.view(np.uint8) != b.view(np.uint8)))
                bad.append((name, "bytes differ", float(d.max()), nd))
        print(f"    fields compared : {len(common)}  "
              f"(missing {missing or 'none'}, extra {extra or 'none'})")
        if not bad:
            print("    BITWISE IDENTICAL on every field")
        else:
            for name, why, mx, nd in bad:
                print(f"    {name:6s} {why:14s} max|diff| = {mx:.3e}  "
                      f"({nd} bytes differ)")
        if exact and not bad:
            digests = {"mono": _digest({k: host_fields[k] for k in common}),
                       "ooc": _digest({k: fields[k] for k in common})}
            print(f"    SHA-256 mono = {digests['mono'][:32]}")
            print(f"    SHA-256 ooc  = {digests['ooc'][:32]}")
            print(f"    -> {'MATCH' if digests['mono'] == digests['ooc'] else 'DIFFER'}")

    # --- file level, on the exact form.
    cache = {}
    fields, _dt = ooc_frame(store, state, cache, exact=True)
    print("\n  --- FILE LEVEL (write both, read both back) ---")
    a = write_frame_kept(host_fields, cfg, DISK_DIR, "verify_mono.nc")
    b = write_frame_kept(fields, cfg, DISK_DIR, "verify_ooc.nc")
    da, db = file_var_digests(a["_path"]), file_var_digests(b["_path"])
    print(f"    variables in file : {len(da)} vs {len(db)}")
    diff = [k for k in sorted(set(da) | set(db)) if da.get(k) != db.get(k)]
    for k in sorted(da):
        flag = "DIFFER" if da.get(k) != db.get(k) else "same"
        if flag == "DIFFER":
            print(f"      {k:10s} {flag}")
    print(f"    {len(da) - len(diff)}/{len(da)} variables identical; "
          f"differing: {diff or 'NONE'}")
    ra, rb = a["_path"].stat().st_size, b["_path"].stat().st_size
    print(f"    file bytes        : {ra} vs {rb}  "
          f"({'equal' if ra == rb else 'DIFFERENT'})")
    ha = hashlib.sha256(a["_path"].read_bytes()).hexdigest()
    hb = hashlib.sha256(b["_path"].read_bytes()).hexdigest()
    print(f"    whole-file SHA    : {'IDENTICAL' if ha == hb else 'DIFFER'}")
    print(f"      mono {ha[:48]}")
    print(f"      ooc  {hb[:48]}")

    # A control for the control: the container has to be reproducible against
    # ITSELF, or "identical" would mean nothing.  Same fields, written twice.
    c = write_frame_kept(host_fields, cfg, DISK_DIR, "verify_mono2.nc")
    hc = hashlib.sha256(c["_path"].read_bytes()).hexdigest()
    print(f"    container determinism (same input twice): "
          f"{'reproducible' if hc == ha else 'NOT REPRODUCIBLE - the whole '
             'file comparison above is void'}")

    # And the negative control: shuffle the presentation order only.  If this
    # did NOT change the file, ordering would not be load-bearing and the
    # claim above would be unfalsifiable.
    shuffled = {k: fields[k] for k in sorted(fields)}
    d = write_frame_kept(shuffled, cfg, DISK_DIR, "verify_shuf.nc")
    hd = hashlib.sha256(d["_path"].read_bytes()).hexdigest()
    print(f"    NEGATIVE CONTROL, field order shuffled  : "
          f"{'still identical (ordering NOT load-bearing)' if hd == ha else f'differs, +{int(d['_file_bytes'] - ra)} bytes'}"
          f"  [per-variable digests still "
          f"{'equal' if file_var_digests(d['_path']) == da else 'DIFFERENT'}]")

    for p in (a, b, c, d):
        p["_path"].unlink(missing_ok=True)

    store.free()
    del host_fields, pinned, state
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


# ---------------------------------------------------------------------------
# 2. THE HYPOTHESIS, end to end
# ---------------------------------------------------------------------------

def mode_endtoend(args) -> None:
    """(a) monolithic D2H+encode+write vs (b) streamed encode+write.

    Both paths run the REAL ``WrfoutWriter`` over the SAME 15-field frame on
    the SAME filesystem, timed through fsync.

    WHY THE TERMS ARE TIMED SEPARATELY AND THEN COMPOSED, rather than as one
    paired A/B loop.  The quantity under test (the D2H) is ~30 ms; the disk
    write it sits in front of is ~1800 ms.  MEASURED at 1024^2, interleaving
    the write with the transfer leaves the transfer's MEDIAN unmoved (36.5 ms
    alone vs 36.3 ms interleaved) but takes its spread from 12% to 549% -- so
    a paired loop answers a 30 ms question with a 1800 ms ruler.  Each term is
    therefore measured in its own tight loop and the totals are COMPOSED from
    the medians, which is stated on every derived row.

    The write is measured from BOTH sources anyway, because "the write is the
    same for both paths" is an assumption Phase 1 made and this checks it.
    """
    import cupy as cp

    from tilestream.hoststore import HostDomainStore

    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else [
        512, 724, 1024]
    reps = args.reps
    print("=" * 78)
    print("END TO END: monolithic (D2H + write) vs streamed (write only)")
    print(f"  target {DISK_DIR}   free {_free_gb():.0f} GB   reps {reps}")
    print(f"  GPU at start: {gpu_busy()[0]} MiB used, {gpu_busy()[1]}% util "
          f"(this lane shares the card)")
    print("=" * 78)
    rows = []
    for n in sizes:
        cfg, state = _build(n, n)
        cells = n * n * NZ
        store = HostDomainStore(cfg)
        _fill_store(store, state)
        cache: dict = {}
        # Adopt the device frame's field order so both paths present netCDF
        # the same schema -- otherwise the two files differ (mode_verify) and
        # the ENCODE comparison is not over the same container layout.
        probe, _s, _n, _p = _stage_like_submit(state, pinned=True)
        adopt_order(probe)
        del probe, _p
        # Warm the statics and the destinations outside every timer.
        ooc_frame(store, state, cache, exact=True)

        stage = Samples("stage total")
        dev_d = Samples("DERIVE device")
        alloc = Samples("ALLOC pinned")
        d2h = Samples("D2H")
        derive = Samples("host derive")
        mono_w = Samples("mono write")
        ooc_w = Samples("ooc write")
        enc_m = Samples("mono ENCODE")
        enc_o = Samples("ooc ENCODE")
        payload = 0.0
        fbytes = 0.0

        # --- PHASE 1: the GPU-side staging, in its own loop, no disk.
        for rep in range(reps + 2):
            hf, st, nb, pin = _stage_like_submit(state, pinned=True)
            if rep >= 2:
                stage.add(st["_stage_total"])
                dev_d.add(st["DERIVE (device)"])
                alloc.add(st["ALLOC host dest"])
                d2h.add(st["D2H transfer"])
                payload = float(nb)
            mono_fields = hf
            del pin

        # --- PHASE 2: the host-side derive, in its own loop, no disk.
        for rep in range(reps + 2):
            fo, dt = ooc_frame(store, state, cache, exact=True)
            if rep >= 2:
                derive.add(dt)
            ooc_fields = fo

        # --- PHASE 3: the write, from both sources, alternating.
        for rep in range(reps + 2):
            if rep % 2 == 0:
                wa = write_frame_kept(mono_fields, cfg, DISK_DIR, "e2e_m.nc")
                wa["_path"].unlink(missing_ok=True)
                _settle()
                wb = write_frame_kept(ooc_fields, cfg, DISK_DIR, "e2e_o.nc")
                wb["_path"].unlink(missing_ok=True)
                _settle()
            else:
                wb = write_frame_kept(ooc_fields, cfg, DISK_DIR, "e2e_o.nc")
                wb["_path"].unlink(missing_ok=True)
                _settle()
                wa = write_frame_kept(mono_fields, cfg, DISK_DIR, "e2e_m.nc")
                wa["_path"].unlink(missing_ok=True)
                _settle()
            if rep >= 2:
                mono_w.add(wa["_write_total"])
                ooc_w.add(wb["_write_total"])
                enc_m.add(wa["ENCODE"])
                enc_o.add(wb["ENCODE"])
                fbytes = wa["_file_bytes"]
        del mono_fields, ooc_fields

        # The WRITE the two paths share.  Any difference between mono_w and
        # ooc_w is noise unless it exceeds the write's own spread, so the
        # common write used in the totals is the pooled median -- otherwise
        # disk noise would be booked as a property of streaming.
        common_w = statistics.median(mono_w.values + ooc_w.values)
        mono_total = stage.median + common_w
        ooc_total = derive.median + common_w
        save = mono_total - ooc_total
        rows.append(dict(
            nx=n, cells=cells, payload_MB=payload / 1e6, file_MB=fbytes / 1e6,
            stage_ms=stage.median * 1e3, derive_device_ms=dev_d.median * 1e3,
            alloc_ms=alloc.median * 1e3, d2h_ms=d2h.median * 1e3,
            host_derive_ms=derive.median * 1e3,
            mono_write_ms=mono_w.median * 1e3, ooc_write_ms=ooc_w.median * 1e3,
            common_write_ms=common_w * 1e3,
            mono_encode_ms=enc_m.median * 1e3, ooc_encode_ms=enc_o.median * 1e3,
            mono_total_ms=mono_total * 1e3, ooc_total_ms=ooc_total * 1e3,
            saving_ms=save * 1e3, ratio=mono_total / ooc_total,
            stage_share=stage.median / mono_total,
            spread_stage=stage.spread, spread_d2h=d2h.spread,
            spread_derive=derive.spread, spread_write_m=mono_w.spread,
            spread_write_o=ooc_w.spread,
            write_delta_frac=abs(mono_w.median - ooc_w.median) / common_w))
        r = rows[-1]
        print(f"\n  nx={n}  {cells/1e6:.1f} Mcell  payload {r['payload_MB']:.0f} MB"
              f"  file {r['file_MB']:.0f} MB")
        print(f"    MEASURED, GPU side : DERIVE {r['derive_device_ms']:7.2f} "
              f"+ ALLOC {r['alloc_ms']:5.2f} + D2H {r['d2h_ms']:7.2f} = "
              f"{r['stage_ms']:7.2f} ms   spread {r['spread_stage']*100:3.0f}% "
              f"(D2H alone {r['spread_d2h']*100:3.0f}%)")
        print(f"    MEASURED, host side: host derive T,P,PSFC "
              f"{r['host_derive_ms']:7.2f} ms   spread "
              f"{r['spread_derive']*100:3.0f}%")
        print(f"    MEASURED, write    : mono {r['mono_write_ms']:8.1f} vs ooc "
              f"{r['ooc_write_ms']:8.1f} ms  (differ by "
              f"{r['write_delta_frac']*100:.1f}%, spreads "
              f"{r['spread_write_m']*100:.0f}%/{r['spread_write_o']*100:.0f}%)"
              f" -> pooled {r['common_write_ms']:8.1f}")
        print(f"    MEASURED, ENCODE   : mono {r['mono_encode_ms']:8.1f} vs ooc "
              f"{r['ooc_encode_ms']:8.1f} ms  -> encoding from pinned-store "
              f"views is {r['mono_encode_ms']/r['ooc_encode_ms']:.3f}x")
        print(f"    DERIVED total (a)  : {r['mono_total_ms']:8.1f} ms  "
              f"(stage is {r['stage_share']*100:.1f}% of it)")
        print(f"    DERIVED total (b)  : {r['ooc_total_ms']:8.1f} ms")
        print(f"    DERIVED saving     : {r['saving_ms']:+8.1f} ms  "
              f"({r['ratio']:.4f}x)")
        store.free()
        del state, cache
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    print(f"\n  cumulative bytes written by this process: "
          f"{WRITTEN[0]/1e9:.1f} GB   free now {_free_gb():.0f} GB")
    _dump(args, "endtoend", rows)


# ---------------------------------------------------------------------------
# 3. CAN THE WRITER RUN WHILE THE GPU STEPS?
# ---------------------------------------------------------------------------

def mode_tear(args) -> None:
    """THE ATTRACTIVE CLAIM, tested against the real driver.

    The claim: the store is host-resident continuously, so a frame can be
    written while the GPU computes, touching memory the solver is not using.

    Whether that is true is decided by ``run_tiled``'s read/write discipline,
    and the two write modes differ completely:

      shadow  tiles READ ``src`` and WRITE ``other``; the two swap at the end
              of every sweep.  ``src`` is untouched for the whole sweep, so a
              reader gets a clean time-t snapshot -- for ONE sweep.
      ring    (the DEFAULT, and the mode that buys 1.90x the domain) keeps
              ONE store.  ``src is dst is home``.  Every tile reads and writes
              it in the same sweep.

    CONTROL THAT CAN FAIL: the concurrent snapshot's digest is compared with
    the digest at time t and at time t+dt.  Clean => it equals one of them.
    Torn => it equals neither.  Shadow is expected clean and ring torn; if
    ring came out clean the claim would stand as written.
    """
    import threading

    import cupy as cp

    from tilestream import driver, harness
    from tilestream.hoststore import HostDomainStore

    n = args.n
    tile = args.tile or max(64, n // 2)
    cfg, state = _build(n, n)
    print("=" * 78)
    print(f"CONCURRENCY: is the store readable while run_tiled steps?  "
          f"nx=ny={n}, tile={tile}")
    print("=" * 78)

    store = HostDomainStore(cfg)
    _fill_store(store, state)
    base = {k: np.array(v, copy=True) for k, v in store.arrays.items()}
    d_t = _digest(base)

    # The reference generations: what the store holds at t, and at t+1 sweep.
    def reset():
        for k, v in base.items():
            store.arrays[k][...] = v

    for mode in ("shadow", "ring"):
        reset()
        driver.run_tiled(store, cfg, tile, tile, halo=16, nsteps=1,
                         write_mode=mode)
        d_t1 = _digest(store.arrays)
        reset()

        print(f"\n  --- write_mode={mode!r} ---")
        print(f"    digest(t)    {d_t[:24]}")
        print(f"    digest(t+dt) {d_t1[:24]}   "
              f"{'(sweep changed the store)' if d_t1 != d_t else 'NO CHANGE - test is void'}")
        if d_t1 == d_t:
            print("    the sweep did not change the store; this control cannot"
                  " discriminate and the result below is meaningless.")

        # (i) a genuinely concurrent reader: no synchronisation whatever,
        #     which is exactly the proposed design.
        snap: dict = {}
        stop = threading.Event()

        def reader():
            time.sleep(args.delay)
            for k, v in store.arrays.items():
                snap[k] = np.array(v, copy=True)
            stop.set()

        th = threading.Thread(target=reader)
        th.start()
        driver.run_tiled(store, cfg, tile, tile, halo=16, nsteps=args.sweeps,
                         write_mode=mode)
        th.join()
        d_snap = _digest(snap)
        verdict = ("CLEAN t" if d_snap == d_t else
                   "CLEAN t+dt" if d_snap == d_t1 else "TORN")
        print(f"    concurrent reader over {args.sweeps} sweep(s), no sync:")
        print(f"      digest(snapshot) {d_snap[:24]}  ->  {verdict}")
        if verdict == "TORN":
            mixed = []
            for k in sorted(snap):
                same_t = np.array_equal(snap[k], base[k])
                if not same_t:
                    mixed.append(k)
            print(f"      fields already advanced in the snapshot: "
                  f"{len(mixed)}/{len(snap)}  {mixed[:6]}")

        # (ii) the deterministic form: snapshot from the progress callback at
        #      the midpoint tile, after a device sync, so the result is
        #      reproducible rather than a race outcome.
        reset()
        det: dict = {}
        ntiles_seen = [0]

        def progress(istep, itile, tspec):
            ntiles_seen[0] = max(ntiles_seen[0], itile + 1)
            if istep == 0 and itile == 0 and not det:
                cp.cuda.runtime.deviceSynchronize()
                for k, v in store.arrays.items():
                    det[k] = np.array(v, copy=True)

        driver.run_tiled(store, cfg, tile, tile, halo=16, nsteps=1,
                         write_mode=mode, progress=progress)
        d_det = _digest(det)
        dverdict = ("CLEAN t" if d_det == d_t else
                    "CLEAN t+dt" if d_det == d_t1 else "TORN")
        print(f"    deterministic snapshot after tile 0 of "
              f"{ntiles_seen[0]} (device-synced):")
        print(f"      digest(snapshot) {d_det[:24]}  ->  {dverdict}")

    reset()
    store.free()
    del state
    cp.get_default_memory_pool().free_all_blocks()


def mode_transfer(args) -> None:
    """The two staging costs across domain size, robust to a shared GPU.

    Another lane is on this card and its occupancy swings from 4.7 to 31.8 GiB
    and 15% to 99% util within a minute, so a MEDIAN over a handful of reps is
    a measurement of that lane.  Contention can only ADD time to a transfer,
    never remove it, so the MINIMUM over many reps is the uncontended floor
    and that is what is reported as the figure.  The median is printed beside
    it: where the two agree the card was quiet, where they diverge the gap is
    the contention and the minimum is still the hardware.
    """
    import cupy as cp

    from tilestream.hoststore import HostDomainStore

    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else [
        512, 724, 1024, 1448]
    reps = max(15, args.reps)
    print("=" * 78)
    print("STAGING COST vs DOMAIN SIZE   (min over %d reps = uncontended "
          "floor)" % reps)
    print("=" * 78)
    print(f"  {'nx':>5s} {'Mcell':>7s} {'MB':>7s} | {'DERIVEdev':>10s} "
          f"{'D2H':>8s} {'GB/s':>6s} | {'hostderive':>10s} "
          f"{'snapshot':>9s} {'GB/s':>6s} | GPU")
    rows = []
    for n in sizes:
        cfg, state = _build(n, n)
        store = HostDomainStore(cfg)
        _fill_store(store, state)
        probe, _s, _nn, _p = _stage_like_submit(state, pinned=True)
        adopt_order(probe)
        del probe, _p
        cache: dict = {}
        ooc_frame(store, state, cache, exact=True)

        dv, tr, hd = Samples("dev"), Samples("d2h"), Samples("hostderive")
        nb = 0.0
        for rep in range(reps):
            hf, st, nbb, pin = _stage_like_submit(state, pinned=True)
            dv.add(st["DERIVE (device)"])
            tr.add(st["D2H transfer"])
            nb = float(nbb)
            del hf, pin
            _fo, dt = ooc_frame(store, state, cache, exact=True)
            hd.add(dt)

        # The snapshot a streamed writer needs (mode_window shows it must).
        names = [k for k in ("u", "v", "w", "thp", "php", "p", "mup")
                 if k in store.arrays]
        vol = sum(store.arrays[k].nbytes for k in names)
        dst = {k: np.zeros_like(store.arrays[k]) for k in names}
        sn = Samples("snapshot")
        for rep in range(reps):
            t0 = _now()
            for k in names:
                np.copyto(dst[k], store.arrays[k])
            sn.add(_now() - t0)

        mem, util = gpu_busy()
        r = dict(nx=n, cells=n * n * NZ, payload_MB=nb / 1e6,
                 volatile_MB=vol / 1e6,
                 derive_dev_ms=dv.lo * 1e3, derive_dev_med_ms=dv.median * 1e3,
                 d2h_ms=tr.lo * 1e3, d2h_med_ms=tr.median * 1e3,
                 d2h_GBs=nb / tr.lo / 1e9,
                 host_derive_ms=hd.lo * 1e3, host_derive_med_ms=hd.median * 1e3,
                 snapshot_ms=sn.lo * 1e3, snapshot_med_ms=sn.median * 1e3,
                 snapshot_GBs=vol / sn.lo / 1e9,
                 gpu_mib=mem, gpu_util=util, reps=reps)
        rows.append(r)
        print(f"  {n:5d} {r['cells']/1e6:7.1f} {r['payload_MB']:7.0f} | "
              f"{r['derive_dev_ms']:10.2f} {r['d2h_ms']:8.2f} "
              f"{r['d2h_GBs']:6.1f} | {r['host_derive_ms']:10.2f} "
              f"{r['snapshot_ms']:9.2f} {r['snapshot_GBs']:6.1f} | "
              f"{mem}MiB/{util}%")
        store.free()
        del state, cache, dst
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    _dump(args, "transfer", rows)


def mode_steps(args) -> None:
    """Solver step time per domain size -- the denominator of every cadence.

    Same min-over-many discipline as :func:`mode_transfer`: the card is
    shared, contention only adds, so the minimum is the hardware.
    """
    import cupy as cp

    from tilestream import harness

    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else [
        512, 724, 1024, 1448]
    reps = max(9, args.reps)
    print("=" * 78)
    print(f"SOLVER STEP TIME  (min over {reps} reps = uncontended floor)")
    print("=" * 78)
    rows = []
    for n in sizes:
        cfg, state = _build(n, n)
        harness.run_steps(state, cfg, 2)
        s = Samples("step")
        for _ in range(reps):
            cp.cuda.runtime.deviceSynchronize()
            t = _now()
            harness.run_steps(state, cfg, 2)
            s.add((_now() - t) / 2)
        cells = n * n * NZ
        mem, util = gpu_busy()
        rows.append(dict(nx=n, cells=cells, step_ms=s.lo * 1e3,
                         step_med_ms=s.median * 1e3,
                         ns_per_cell=s.lo / cells * 1e9,
                         spread=s.spread, gpu_mib=mem, gpu_util=util))
        r = rows[-1]
        print(f"  nx={n:5d} {cells/1e6:7.1f} Mcell  step {r['step_ms']:8.2f} ms "
              f"(median {r['step_med_ms']:8.2f}, spread {r['spread']*100:4.0f}%)"
              f"  {r['ns_per_cell']:.3f} ns/cell   GPU {mem}MiB/{util}%")
        del state
        cp.get_default_memory_pool().free_all_blocks()
    _dump(args, "steps", rows)


def mode_window(args) -> None:
    """WHEN, exactly, is the caller's store safe to read?

    :func:`mode_tear` answers "is it safe during a sweep" (shadow yes, ring
    no).  This maps the whole window: a deterministic snapshot is taken from
    the ``progress`` callback at every (sweep, tile) point of a multi-sweep
    run, and each is classified against the generation digests.  That is what
    decides whether a writer can be handed the store and left to run, or has
    to be given a copy.
    """
    import cupy as cp

    from tilestream import driver
    from tilestream.hoststore import HostDomainStore

    n, tile = args.n, (args.tile or max(64, args.n // 2))
    nsw = max(2, args.sweeps)
    cfg, state = _build(n, n)
    store = HostDomainStore(cfg)
    _fill_store(store, state)
    base = {k: np.array(v, copy=True) for k, v in store.arrays.items()}

    print("=" * 78)
    print(f"SAFE WINDOW: when is the caller's store readable?  nx=ny={n}, "
          f"tile={tile}, {nsw} sweeps")
    print("=" * 78)

    def reset():
        for k, v in base.items():
            store.arrays[k][...] = v

    # Generation digests: what the store holds after each whole sweep.
    gens = [_digest(base)]
    for s in range(nsw):
        driver.run_tiled(store, cfg, tile, tile, halo=16, nsteps=1,
                         write_mode=args.mode2)
        gens.append(_digest(store.arrays))
    label = {d: f"t+{i}" for i, d in enumerate(gens)}

    reset()
    seen: list[tuple[int, int, str]] = []

    def progress(istep, itile, tspec):
        cp.cuda.runtime.deviceSynchronize()
        d = _digest(store.arrays)
        seen.append((istep, itile, label.get(d, "TORN")))

    driver.run_tiled(store, cfg, tile, tile, halo=16, nsteps=nsw,
                     write_mode=args.mode2, progress=progress)
    final = label.get(_digest(store.arrays), "TORN")

    print(f"  write_mode={args.mode2!r}; snapshot taken after every tile, "
          f"device-synced (so this is the BEST case for a reader)")
    ntiles = max(t for _s, t, _v in seen) + 1
    for s in range(nsw):
        row = [v for (ss, _t, v) in seen if ss == s]
        clean = sum(1 for v in row if v != "TORN")
        print(f"    sweep {s}: " + " ".join(f"{v:>5s}" for v in row) +
              f"   ({clean}/{len(row)} tile boundaries clean)")
    print(f"    after the call returns: {final}")
    print(f"  => a reader of the caller's store sees a consistent frame at "
          f"{sum(1 for _s,_t,v in seen if v != 'TORN')}/{len(seen)} of the "
          f"{ntiles} x {nsw} in-sweep observation points")
    store.free()
    del state
    cp.get_default_memory_pool().free_all_blocks()


def mode_snapshot(args) -> None:
    """What a consistent snapshot actually costs, against the D2H it replaces.

    If the store is not quiescent (ring mode, always), a writer needs its own
    copy of the frame's fields taken at a sweep boundary.  That copy is a HOST
    memcpy.  The monolithic path's equivalent is a PCIe D2H.  Which is faster
    is not obvious and is the whole hypothesis in one line.

    CONTROL THAT CAN FAIL: if the host memcpy beats the D2H, the hypothesis
    survives even in ring mode.
    """
    import cupy as cp

    from tilestream.hoststore import HostDomainStore

    n = args.n
    cfg, state = _build(n, n)
    store = HostDomainStore(cfg)
    _fill_store(store, state)
    print("=" * 78)
    print(f"SNAPSHOT vs D2H   nx=ny={n}, nz={NZ}")
    print("=" * 78)

    # The frame's volatile volume fields -- what a snapshot must copy.
    names = [k for k in ("u", "v", "w", "thp", "php", "p", "mup")
             if k in store.arrays]
    vol = sum(store.arrays[k].nbytes for k in names)
    dst = {k: np.zeros_like(store.arrays[k]) for k in names}   # first-touched
    pin = {}
    for k in names:
        a = store.arrays[k]
        mem = cp.cuda.alloc_pinned_memory(int(a.nbytes))
        h = np.frombuffer(mem, dtype=a.dtype, count=a.size).reshape(a.shape)
        h[...] = 0
        pin[k] = (mem, h)

    page = Samples("host memcpy 1 thread")
    pind = Samples("host memcpy 1 thread -> pinned")
    for rep in range(args.reps + 2):
        t0 = _now()
        for k in names:
            np.copyto(dst[k], store.arrays[k])
        t1 = _now()
        for k in names:
            np.copyto(pin[k][1], store.arrays[k])
        t2 = _now()
        if rep >= 2:
            page.add(t1 - t0)
            pind.add(t2 - t1)

    # A THREADED snapshot.  A single np.copyto is one core against DDR5; the
    # honest comparison for "what would a real streaming writer pay" is the
    # best host copy available, not the first one written.  numpy releases the
    # GIL inside copyto, so threads genuinely run.
    from concurrent.futures import ThreadPoolExecutor
    threaded: dict[int, Samples] = {}
    chunks: list[tuple] = []
    for k in names:
        a, b = store.arrays[k], dst[k]
        nrow = a.shape[0]
        for i in range(0, nrow, max(1, nrow // 8)):
            chunks.append((b, a, slice(i, min(nrow, i + max(1, nrow // 8)))))
    for nthread in (2, 4, 8, 16):
        s = Samples(f"host memcpy {nthread} threads")
        with ThreadPoolExecutor(max_workers=nthread) as ex:
            for rep in range(args.reps + 2):
                t0 = _now()
                list(ex.map(lambda c: np.copyto(c[0][c[2]], c[1][c[2]]),
                            chunks))
                t1 = _now()
                if rep >= 2:
                    s.add(t1 - t0)
        threaded[nthread] = s

    d2h = Samples("PCIe D2H (pinned)")
    for rep in range(args.reps + 2):
        hf, st, nb, refs = _stage_like_submit(state, pinned=True)
        if rep >= 2:
            d2h.add(st["D2H transfer"])
        del hf, refs
        cp.get_default_pinned_memory_pool().free_all_blocks()

    mem, util = gpu_busy()
    print(f"  volatile volume bytes            : {vol/1e9:.3f} GB "
          f"({len(names)} fields: {' '.join(names)})")
    print(f"  D2H payload (full 15-field frame): {nb/1e9:.3f} GB   "
          f"(includes the broadcast PB/PHB the store does not hold)")
    print(f"  GPU during this run              : {mem} MiB, {util}% util")
    best = min([page, pind] + list(threaded.values()),
               key=lambda s: s.median)
    for s in [page, pind] + [threaded[k] for k in sorted(threaded)]:
        flag = "  <- best host copy" if s is best else ""
        print(f"  {s.name:<34s} {s.median*1e3:8.2f} ms  "
              f"{vol/s.median/1e9:6.1f} GB/s   spread {s.spread*100:4.0f}%{flag}")
    print(f"  {d2h.name:<34s} {d2h.median*1e3:8.2f} ms  "
          f"{nb/d2h.median/1e9:6.1f} GB/s   spread {d2h.spread*100:4.0f}%"
          f"{'   <<CONTENDED, not usable' if d2h.spread > 0.5 else ''}")

    r_host = vol / best.median / 1e9
    r_d2h = nb / d2h.median / 1e9
    print(f"\n  best host copy {r_host:.1f} GB/s vs PCIe D2H {r_d2h:.1f} GB/s")
    print(f"  -> a streamed writer that needs a consistent snapshot pays "
          f"{best.median*1e3:.1f} ms of HOST copy in place of "
          f"{d2h.median*1e3:.1f} ms of PCIe")
    _dump(args, "snapshot", [dict(
        nx=n, volatile_GB=vol / 1e9, frame_GB=nb / 1e9,
        host_1thread_ms=page.median * 1e3, host_pinned_ms=pind.median * 1e3,
        host_best_ms=best.median * 1e3, host_best_label=best.name,
        threaded_ms={k: v.median * 1e3 for k, v in threaded.items()},
        d2h_ms=d2h.median * 1e3, d2h_spread=d2h.spread,
        host_GBs=r_host, d2h_GBs=r_d2h, gpu_mib=mem, gpu_util=util)])
    store.free()
    del state
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def mode_concurrent(args) -> None:
    """The real thing: run_tiled stepping while a writer thread writes a frame.

    A = sweeps alone, B = one frame alone, C = both.  C ~= max(A,B) means the
    output really is free at this cadence; C ~= A+B means it is not.  Unlike
    Phase 1's overlap control this drives the TILED driver, so the writer is
    competing with the driver's own host-side gather/scatter work and its
    pinned host store -- which is the configuration the claim is about.
    """
    import threading

    import cupy as cp

    from tilestream import driver
    from tilestream.hoststore import HostDomainStore

    n = args.n
    tile = args.tile or max(64, n // 2)
    cfg, state = _build(n, n)
    store = HostDomainStore(cfg)
    _fill_store(store, state)
    cache: dict = {}
    fields, _dt = ooc_frame(store, state, cache, exact=True)
    # Snapshot the frame ONCE; the writer writes from this, which is the
    # sweep-boundary-snapshot design mode_tear shows is required.
    frozen = {k: np.array(v, copy=True) for k, v in fields.items()}

    print("=" * 78)
    print(f"CONCURRENT: tiled solver + writer thread   nx=ny={n}, tile={tile}")
    print("=" * 78)

    driver.run_tiled(store, cfg, tile, tile, halo=16, nsteps=1)
    t0 = _now()
    driver.run_tiled(store, cfg, tile, tile, halo=16, nsteps=2)
    sweep = (_now() - t0) / 2
    w0 = _now()
    wr = write_frame_kept(frozen, cfg, DISK_DIR, "conc_probe.nc")
    wr["_path"].unlink(missing_ok=True)
    probe = _now() - w0
    nsweeps = max(2, int(round(probe / sweep)))
    print(f"  sweep {sweep*1e3:.0f} ms, frame {probe*1e3:.0f} ms -> "
          f"{nsweeps} sweeps per trial")
    _settle()

    A, B, C = Samples("solver alone"), Samples("write alone"), Samples("both")
    for rep in range(args.reps + 1):
        cp.cuda.runtime.deviceSynchronize()
        t = _now()
        driver.run_tiled(store, cfg, tile, tile, halo=16, nsteps=nsweeps)
        a = _now() - t
        _settle()
        t = _now()
        w = write_frame_kept(frozen, cfg, DISK_DIR, "conc_b.nc")
        w["_path"].unlink(missing_ok=True)
        b = _now() - t
        _settle()

        def _writer():
            ww = write_frame_kept(frozen, cfg, DISK_DIR, "conc_c.nc")
            ww["_path"].unlink(missing_ok=True)

        cp.cuda.runtime.deviceSynchronize()
        t = _now()
        th = threading.Thread(target=_writer)
        th.start()
        driver.run_tiled(store, cfg, tile, tile, halo=16, nsteps=nsweeps)
        th.join()
        c = _now() - t
        _settle()
        if rep:
            A.add(a), B.add(b), C.add(c)

    a, b, c = A.median, B.median, C.median
    print(f"  A  {nsweeps} sweeps alone : {a*1e3:8.1f} ms  spread {A.spread*100:3.0f}%")
    print(f"  B  1 frame alone      : {b*1e3:8.1f} ms  spread {B.spread*100:3.0f}%")
    print(f"  C  both               : {c*1e3:8.1f} ms  spread {C.spread*100:3.0f}%")
    print(f"     max(A,B) {max(a,b)*1e3:8.1f}   A+B {(a+b)*1e3:8.1f}")
    print(f"     overlap efficiency {(a + b - c)/min(a, b)*100:.0f}% of the "
          f"smaller task hidden")
    print(f"     tiled-solver slowdown while writing: {c/a:.3f}x")
    _dump(args, "concurrent", [dict(
        nx=n, tile=tile, sweeps=nsweeps, solver_ms=a * 1e3, write_ms=b * 1e3,
        both_ms=c * 1e3, overlap_eff=(a + b - c) / min(a, b),
        slowdown=c / a)])
    store.free()
    del state
    cp.get_default_memory_pool().free_all_blocks()


def _dump(args, key: str, rows: list) -> None:
    """Append a block to the plot-data JSON without losing the other blocks."""
    import json

    path = Path(args.json)
    blob = {}
    if path.exists():
        blob = json.loads(path.read_text())
    blob.setdefault("blocks", {})[key] = rows
    path.write_text(json.dumps(blob, indent=1))
    print(f"  [json] {key}: {len(rows)} row(s) -> {path}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["verify", "endtoend", "tear", "window",
                                    "transfer", "steps", "snapshot",
                                    "concurrent", "all"])
    p.add_argument("--mode2", default="ring",
                   help="write_mode for the window map: ring or shadow")
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--tile", type=int, default=0)
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--sweeps", type=int, default=1)
    p.add_argument("--delay", type=float, default=0.0)
    p.add_argument("--sizes", type=str, default="")
    p.add_argument("--json", type=str,
                   default=str(Path(__file__).with_name(
                       "output-scaling.json")))
    args = p.parse_args(argv)
    modes = ({"verify": mode_verify, "endtoend": mode_endtoend,
              "tear": mode_tear, "window": mode_window,
              "transfer": mode_transfer, "steps": mode_steps,
              "snapshot": mode_snapshot,
              "concurrent": mode_concurrent})
    if args.mode == "all":
        for fn in modes.values():
            fn(args)
    else:
        modes[args.mode](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
