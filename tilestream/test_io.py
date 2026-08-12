"""THE I/O GATE: a streamed forecast must WRITE and RESUME like a resident one.

Run it::

    python -m tilestream.test_io              # from the repository root

:mod:`tilestream.test_gate` proves the streamed trajectory equals the
monolithic one.  That is necessary and it is not sufficient: a forecast that
cannot write a wrfout is not a forecast, and a forecast that cannot be
resumed is not a forecast either.  This file proves both, and it proves them
the only way that means anything -- by comparing BYTES against the resident
path that already works, and by asserting that each comparison FAILS when the
thing it tests is broken.

WHAT IS UNDER TEST
------------------
``tilestream.output``
    A wrfout frame built with no device traffic at all, out of the pinned
    host store plus the domain's setup.  Compared field by field and then
    file by file against ``gpuwm.io.wrfout._device_state_frame`` on the
    resident state, at four physics rungs.

``tilestream.checkpoint``
    A v5 gpuwm restart written out of the same store, and read back into a
    fresh one.  Compared header-key by header-key and member by member
    against ``restart.write_restart``, then round-tripped four ways.

THE FOUR LEGS OF THE ROUND TRIP
-------------------------------
One reference -- ``N + M`` uninterrupted monolithic steps -- and four ways to
reach it.  Nothing here would be proved by streamed->streamed alone: two
copies of the same mistake agree with each other.

======================  ==============================================
leg                     what it would catch that the others would not
======================  ==============================================
streamed, N + M         the streaming loop itself (already gated, run
                        here so a checkpoint failure cannot hide behind
                        a streaming failure)
streamed N, restart,    the checkpoint round trip inside one lane
streamed M
streamed N, restart,    the file really is a gpuwm restart: an
MONOLITHIC M            unmodified ``restore_restart`` reads it into a
                        resident state and continues the trajectory
monolithic N, restart,  the reader really validates: a checkpoint the
streamed M              resident path wrote resumes a streamed run
======================  ==============================================

THE NEGATIVE CONTROLS, AND WHY EACH ONE EXISTS
-----------------------------------------------
* ``negative_reassociated_t`` -- ``thp + (thb - 300)`` instead of
  ``(thb + thp) - 300``.  Float addition is not associative, and this is the
  obvious way to write the host derive.  Running it turned up something
  §12.1 did not say: the difference is 26.2% of T's bytes on a freshly
  seeded state and EXACTLY ZERO after one ``dycore.step``, because a stepped
  ``thp`` is 100.00% on the ulp(300) representable grid.  The control has to
  run unwarmed, and it reports the whole matrix so that is visible.
* ``negative_field_order`` -- the same fields in a different order.  Every
  variable compares equal and the FILE is different, which is the trap the
  "compare checksums, not bytes" advice exists for.
* ``negative_tile_fingerprint`` -- a checkpoint whose ``setup_fingerprint``
  was computed from a TILE state.  This is the dangerous one: a monolithic
  resume refuses it (fail closed), but a streamed resume that made the same
  mistake ACCEPTS it -- so the fingerprint silently stops protecting against
  resuming onto different terrain, a different projection or a different base
  state.  The control asserts both halves.
* ``negative_tile_physics_identity`` -- the same, one level down: the
  resolved physics identity taken from a tile driver without substituting the
  DOMAIN's radiation lat/lon grid.
* ``negative_partial_inventory`` -- a checkpoint carrying only ``state/*``,
  which is milestone one's inventory.  ``read_store_restart`` must refuse it
  rather than resume with dirty ``scratch/``, ``driver/`` and ``fields/``
  carriers.
* ``negative_dropped_scalars`` -- resume without ``elapsed_seconds``, and
  resume without the driver call counts and ``microphysics_updates``.  Run at
  ``full fast cadence`` ON PURPOSE: at the long-cadence rungs eight steps of
  3 s never reach a cadence boundary, so the same defect is invisible there.
  The control runs at both and prints both, so the reader can see the cheap
  configuration certifying the bug.
* ``case_diagnostic_scatter`` -- the positive half proves each tile computes
  its own ``OLR`` window correctly and the scatter assembles it bit-exactly;
  the negative half publishes the tile BUFFER after the sweep, which is what
  a naive implementation does, and it is the last tile's values over the
  whole domain.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

from tilestream import checkpoint as tsck
from tilestream import driver, harness, hoststore
from tilestream import output as tsout
from tilestream import physics_inventory as physinv
from tilestream import spec as tspec
from tilestream import test_gate


NZ = test_gate.NZ
SEED = test_gate.SEED
NX, NY = test_gate.PHYS_NX, test_gate.PHYS_NY

#: Frame comparisons run at these four rungs.  Chosen for the frame's own
#: structure rather than for the trajectory's: dry is 15 fields with no
#: physics driver at all (so ``PSFC`` is the z-extrapolated derive rather
#: than a carrier), mp10 adds the Morrison moments and the precipitation
#: accumulators, full+KF adds radiation (and with it the one unavailable
#: diagnostic), and full+MYNN+Noah-MP is 122 fields including the layered
#: soil and snow arrays whose leading axis is not vertical.
FRAME_RUNGS = ("dry (control)", "mp10 Morrison", "full(real74) +KF",
               "full+MYNN+Noah-MP")

#: Round-trip rungs.  ``full fast cadence`` is not decoration: it is the only
#: rung where eight steps cross a radiation and a cumulus boundary, so it is
#: the only rung where a dropped clock or a dropped call count is visible.
ROUNDTRIP_RUNGS = ("mp10 Morrison", "full(real74) +KF", "full+MYNN+Noah-MP",
                   "full fast cadence")

#: Everything this file writes lands here and is removed on the way out.
#: One frame at 96x80x49 with 122 fields is ~55 MB, so the peak is well
#: under a gigabyte even with both paths' files on disk at once.
WORK_DIR = Path("/tmp/tilestream-io-gate")


# --------------------------------------------------------------------------
# shared machinery
# --------------------------------------------------------------------------

def _as_numpy(array) -> np.ndarray:
    import cupy as cp
    return cp.asnumpy(array) if isinstance(array, cp.ndarray) \
        else np.asarray(array)


def _free_device() -> None:
    import cupy as cp
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def build_state(rung: str, nx=NX, ny=NY, nz=NZ, warmup: int = 2,
                **overrides):
    """``(cfg, state)`` at ``rung``, warmed up so the carrier set is final.

    ``warmup >= 1`` for the reason ``test_gate.physics_reference`` states:
    Kain-Fritsch allocates ``cumulus/w0avg`` on its FIRST call, so a state
    that has never stepped has a shorter carrier manifest than one that has,
    and a store sized from it would be missing a field that later appears --
    and so would a checkpoint.
    """
    cfg = harness.make_config(nx, ny, nz,
                              **{**test_gate.PHYSICS_RUNGS[rung],
                                 **overrides})
    # ``default_builder`` serves the dry rung too -- it returns ``(state,
    # None)`` when ``physics_driver_required(cfg)`` is false -- so every rung
    # here gets the WK82 sounding rather than the harness's constant 300 K
    # profile, and the dry case is a genuine control on the same base state
    # rather than a different problem.
    state, _drv = physinv.default_builder(cfg, SEED)
    harness.run_steps(state, cfg, warmup)
    return cfg, state


def make_store(cfg, state, *, inventory_fn=physinv.carrier_inventory):
    """A pinned full-domain store sized from ``state``'s live inventory."""
    inv = inventory_fn(state)
    manifest = hoststore.manifest_from_arrays(inv, cfg.nz, cfg.ny, cfg.nx)
    store = hoststore.HostDomainStore(cfg, manifest=manifest,
                                      inventory_fn=inventory_fn)
    store.assert_pinned()
    return store


def stream(store, cfg, scalars, nsteps, *, tile_nx, tile_ny, halo=None,
           inventory_fn=physinv.carrier_inventory, nbuffers=2,
           write_mode="shadow") -> dict:
    """One tiled integration of ``store``, in place, with the domain clock.

    ``write_mode`` is passed explicitly rather than defaulted so this gate
    does not move when the driver's default does.  Nothing here depends on
    the choice: every case in this file was re-run with ``"ring"`` against
    the halo-ring implementation and passed identically, which is what the
    sweep-boundary rule buys -- the modes differ in when a READER can see a
    consistent store, and a frame or a checkpoint taken between sweeps is
    after the last scatter under either.
    """
    import cupy as cp

    if halo is None:
        halo = harness.halo_radius(cfg)
    report: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        driver.run_tiled(store, cfg, tile_nx, tile_ny, halo=halo,
                         nsteps=nsteps, nbuffers=nbuffers,
                         write_mode=write_mode, report=report,
                         inventory_fn=inventory_fn, nz=int(cfg.nz),
                         tile_state_factory=driver.make_physics_tile_state,
                         scalars=scalars)
    cp.cuda.runtime.deviceSynchronize()
    return report


def frame_diff(want: dict, got: dict) -> list[tuple]:
    """Bytewise field comparison; ``[]`` means the two frames are identical."""
    bad = []
    for name in sorted(set(want) & set(got)):
        a = np.ascontiguousarray(_as_numpy(want[name]))
        b = np.ascontiguousarray(_as_numpy(got[name]))
        if a.shape != b.shape or a.dtype != b.dtype:
            bad.append((name, "shape/dtype", str(a.shape), str(b.shape)))
            continue
        if np.array_equal(a.view(np.uint8), b.view(np.uint8)):
            continue
        delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
        bad.append((name, "bytes",
                    int(np.count_nonzero(a.view(np.uint8)
                                         != b.view(np.uint8))),
                    float(delta.max())))
    return bad


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream_:
        for chunk in iter(lambda: stream_.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def variable_digests(path: Path) -> dict[str, str]:
    """Per-variable SHA-256 out of a written wrfout."""
    import netCDF4

    out: dict[str, str] = {}
    with netCDF4.Dataset(path, "r") as ds:
        for name, var in ds.variables.items():
            data = var[:]
            arr = np.ascontiguousarray(
                np.ma.getdata(data) if np.ma.isMaskedArray(data) else data)
            out[name] = hashlib.sha256(arr.tobytes(order="C")).hexdigest()
    return out


# --------------------------------------------------------------------------
# A.  the frame
# --------------------------------------------------------------------------

def case_frame_bytes(rung: str) -> dict:
    """A frame off the pinned store must equal the device frame, byte for byte.

    Both halves are compared: the field dict (so a difference is localised to
    a variable) and then the written netCDF (so the WRITER's view of the two
    frames is compared too, including the variable creation order that HDF5's
    name heap is laid out in).
    """
    from gpuwm.io import wrfout

    cfg, state = build_state(rung)
    plan = tsout.frame_plan(state)
    store = make_store(cfg, state)
    store.fill_from(state)
    setup = tsck.DomainSetup.capture(state, cfg)
    frame = tsout.StoreFrame(plan, store, setup, cfg)

    device = {name: _as_numpy(value) for name, value
              in wrfout._device_state_frame(state).items()}
    streamed = frame.fields()
    bad = frame_diff(device, streamed)
    order_ok = tuple(streamed) == tuple(
        n for n in plan.order
        if plan.source[n] != tsout.SOURCE_DRIVER_DIAGNOSTIC)

    work = WORK_DIR / f"frame-{abs(hash(rung)):x}"
    work.mkdir(parents=True, exist_ok=True)
    mono_path = tsout.write_frame(work / "mono.nc", device, cfg,
                                  "0001-01-01_00:00:00", title="gate")
    # The device frame carries fields the store cannot serve, so the two
    # files are only comparable over the frame the streamed path can write.
    # Restricting the MONOLITHIC one to that inventory (rather than padding
    # the streamed one) keeps the comparison honest: the missing rows are
    # reported by :func:`case_diagnostic_sources`, not papered over here.
    trimmed = {k: device[k] for k in streamed}
    trim_path = tsout.write_frame(work / "mono-trimmed.nc", trimmed, cfg,
                                  "0001-01-01_00:00:00", title="gate")
    ooc_path = tsout.write_frame(work / "ooc.nc", streamed, cfg,
                                 "0001-01-01_00:00:00", title="gate")
    var_ok = variable_digests(trim_path) == variable_digests(ooc_path)
    file_ok = file_digest(trim_path) == file_digest(ooc_path)

    record = {
        "rung": rung,
        "fields": len(plan.order),
        "written": len(streamed),
        "bitexact": not bad and order_ok,
        "bad": bad,
        "order_ok": order_ok,
        "variables_identical": var_ok,
        "file_identical": file_ok,
        "file_sha256": file_digest(ooc_path)[:16],
        "unavailable": plan.unavailable,
        "mono_bytes": mono_path.stat().st_size,
        "summary": plan.summary(),
    }
    store.free()
    del state
    _free_device()
    shutil.rmtree(work, ignore_errors=True)
    return record


#: ``ulp(300)`` in FP32.  300 lies in [2^8, 2^9), so the spacing of
#: representable neighbours there is ``2^(8-23) = 2^-15``.
ULP_300 = 2.0 ** -15


def _ulp300_quantised(array) -> float:
    """Fraction of ``array`` that is an exact multiple of ``ulp(300)``."""
    scaled = np.asarray(array, dtype=np.float64) / ULP_300
    return float(np.count_nonzero(scaled == np.round(scaled))) / scaled.size


def negative_reassociated_t(rung: str = "mp10 Morrison",
                            steps=(0, 1, 2)) -> dict:
    """``thp + (thb - 300)`` must be caught -- ON A STATE THAT CAN SHOW IT.

    A CONTROL THAT MUST FAIL, and the run that made it fail found something
    the project's own §12.1 result did not say out loud: **the reassociation
    is invisible after one dycore step.**

    ``thb - 300`` is EXACT for any ``thb`` in [150, 600] (Sterbenz), so the
    reassociated form is ``thp + (thb - 300)`` computed without error and the
    device's is ``fl(fl(thb + thp) - 300)``.  Those disagree only when ``thp``
    is NOT on the representable grid at magnitude 300.  A freshly seeded
    ``thp`` is not: MEASURED 0.11% of values land on that grid, and the two
    orders differ in 43.9% of T's bytes with a maximum error of 1.526e-05 K
    -- which reproduces §12.1's 5,641,326 of 12,845,056 exactly.  After ONE
    ``dycore.step``, **100.00% of thp is an exact multiple of ulp(300)** and
    the two orders agree on every byte, at BOTH base states (constant 300 K
    and the WK82 sounding, whose thb runs 300-473 K here).

    So a version of this control that warmed the state up first -- which is
    what every other case in this file does, for the carrier-manifest reason
    -- would PASS while the reassociated derive was in the code.  That is
    FACT 1's shape again: the more realistic configuration is the weaker
    test.  The matrix is reported in full so that is visible rather than
    implied.
    """
    outcomes: dict[str, dict] = {}
    cfg = test_gate.physics_cfg(rung, NX, NY, NZ)
    state, _drv = physinv.default_builder(cfg, SEED)
    import cupy as cp
    thb = np.ascontiguousarray(cp.asnumpy(state.thb), dtype=np.float32)
    thb3 = thb if thb.ndim == 3 else thb[:, None, None]
    taken = 0
    for target in steps:
        harness.run_steps(state, cfg, target - taken)
        taken = target
        thp = cp.asnumpy(state.thp)
        exact = np.empty_like(thp)
        tsout.derive_t(thb, thp, exact)
        reassociated = thp + (thb3 - np.float32(300.0))
        differing = int(np.count_nonzero(
            np.ascontiguousarray(exact).view(np.uint8)
            != np.ascontiguousarray(reassociated).view(np.uint8)))
        outcomes[f"{target} step(s)"] = {
            "differing_bytes": differing,
            "total_bytes": int(exact.nbytes),
            "fraction": differing / float(exact.nbytes),
            "max_abs": float(np.abs(exact.astype(np.float64)
                                    - reassociated.astype(np.float64)).max()),
            "thp_on_ulp300_grid": _ulp300_quantised(thp),
        }
    record = {
        "rung": rung,
        "thb_range": (float(thb.min()), float(thb.max())),
        "outcomes": outcomes,
        "caught": outcomes["0 step(s)"]["differing_bytes"] > 0,
    }
    del state
    _free_device()
    return record


def negative_field_order(rung: str = "mp10 Morrison") -> dict:
    """Same data, different order: every variable equal, the FILE different.

    A CONTROL THAT MUST FAIL, and the reason :class:`~tilestream.output.
    FramePlan` carries the device frame's own iteration order rather than
    ``sorted()``.
    """
    cfg, state = build_state(rung)
    plan = tsout.frame_plan(state)
    store = make_store(cfg, state)
    store.fill_from(state)
    setup = tsck.DomainSetup.capture(state, cfg)
    fields = tsout.StoreFrame(plan, store, setup, cfg).fields()
    shuffled = {k: fields[k] for k in reversed(list(fields))}

    work = WORK_DIR / "order"
    work.mkdir(parents=True, exist_ok=True)
    a = tsout.write_frame(work / "a.nc", fields, cfg, "0001-01-01_00:00:00")
    b = tsout.write_frame(work / "b.nc", shuffled, cfg, "0001-01-01_00:00:00")
    record = {
        "variables_identical": variable_digests(a) == variable_digests(b),
        "file_identical": file_digest(a) == file_digest(b),
        "size_delta": b.stat().st_size - a.stat().st_size,
    }
    record["caught"] = (record["variables_identical"]
                        and not record["file_identical"])
    store.free()
    del state
    _free_device()
    shutil.rmtree(work, ignore_errors=True)
    return record


# --------------------------------------------------------------------------
# B.  the diagnostics: which ones survive being computed per tile
# --------------------------------------------------------------------------

def case_derives_per_tile(rung: str = "full(real74) +KF", *,
                          tile_nx: int = 32, tile_ny: int = 24) -> dict:
    """Are the derived fields separable?  Compute them TILE BY TILE and compare.

    ``T``, ``P`` and ``PSFC`` are the three fields a streamed frame computes
    rather than reads, and the question the brief asks about them is whether
    they can be computed during the sweep instead of in one whole-domain pass
    at the end.  ``T`` and ``P`` are pointwise and ``PSFC`` is column-local,
    so the answer should be yes -- but "should be" is how a broadcast bug at a
    tile edge survives.  This assembles each of them from disjoint tile
    windows (no halo: a pointwise operator needs none) and compares the
    result byte for byte with the whole-domain derive.

    A whole-domain pass over the finished store is the SIMPLER thing and this
    file's default, so the value of the result is not a speedup -- it is that
    the 29.6 ms host derive at 1024^2 is available to be hidden under the
    sweep by a caller who wants it, and that per-tile derives at real tile
    sizes do not change a single byte.
    """
    cfg, state = build_state(rung)
    plan = tsout.frame_plan(state)
    store = make_store(cfg, state)
    store.fill_from(state)
    setup = tsck.DomainSetup.capture(state, cfg)
    whole = tsout.StoreFrame(plan, store, setup, cfg).fields()

    thb = np.asarray(setup.thb)
    thb3 = thb if thb.ndim == 3 else thb[:, None, None]
    pb = np.asarray(setup.pb)
    pb3 = pb if pb.ndim == 3 else pb[:, None, None]
    phb = np.asarray(setup.phb)
    phb3 = phb if phb.ndim == 3 else phb[:, None, None]
    thp, p, php = (store.arrays["state/thp"], store.arrays["state/p"],
                   store.arrays["state/php"])

    tiled: dict[str, np.ndarray] = {
        "T": np.zeros_like(whole["T"]),
        "P": np.zeros_like(whole["P"]),
    }
    windows = 0
    for j0 in range(0, cfg.ny, tile_ny):
        j1 = min(j0 + tile_ny, cfg.ny)
        for i0 in range(0, cfg.nx, tile_nx):
            i1 = min(i0 + tile_nx, cfg.nx)
            windows += 1
            view = (slice(None), slice(j0, j1), slice(i0, i1))
            tsout.derive_t(thb3, thp[view], tiled["T"][view])
            np.subtract(p[view], pb3, out=tiled["P"][view])
    bad = frame_diff({k: whole[k] for k in tiled}, tiled)

    # PSFC only when the state derives it; with a physics driver it is a
    # carrier (fields/psfc) and there is nothing to separate.
    psfc_note = "carrier (fields/psfc)"
    if plan.source.get("PSFC") == tsout.SOURCE_DERIVED:
        work = np.zeros((3, cfg.ny, cfg.nx), np.float32)
        assembled = np.zeros_like(whole["PSFC"])
        for j0 in range(0, cfg.ny, tile_ny):
            j1 = min(j0 + tile_ny, cfg.ny)
            for i0 in range(0, cfg.nx, tile_nx):
                i1 = min(i0 + tile_nx, cfg.nx)
                view = (slice(None), slice(j0, j1), slice(i0, i1))
                assembled[j0:j1, i0:i1] = tsout.derive_psfc(
                    phb3, php[view], p[view], work[:, j0:j1, i0:i1])
        bad += frame_diff({"PSFC": whole["PSFC"]}, {"PSFC": assembled})
        psfc_note = "derived, separable"

    record = {"rung": rung, "windows": windows, "bitexact": not bad,
              "bad": bad, "psfc": psfc_note,
              "tile": (tile_ny, tile_nx)}
    store.free()
    del state
    _free_device()
    return record


def case_history_writer(rung: str = "mp10 Morrison", *, tile_nx: int = 48,
                        tile_ny: int = 40, nsteps: int = 2) -> dict:
    """The writer THREAD, driven from a streamed run at sweep boundaries.

    Both modes, end to end: a frame before the sweeps and one after each,
    written on the background thread that ``netCDF4``'s GIL release is the
    reason for, then reopened and compared against the frame the store held
    at the moment of submission.

    The overlap mode is where a mistake would live, and the mistake it would
    be is subtle: the writer thread holds the frame dict while the solver
    keeps stepping, so if the carriers were served as views instead of
    snapshots the file would contain a later generation than the timestamp
    claims.  The comparison is against a copy taken at submission time, which
    is exactly the discrepancy that would show.
    """
    cfg, state = build_state(rung)
    plan = tsout.frame_plan(state)
    store = make_store(cfg, state)
    store.fill_from(state)
    setup = tsck.DomainSetup.capture(state, cfg)
    scalars = physinv.carrier_scalars(state)
    del state
    _free_device()

    outcomes = {}
    for overlap in (False, True):
        work = WORK_DIR / f"writer-{int(overlap)}"
        work.mkdir(parents=True, exist_ok=True)
        frame = tsout.StoreFrame(plan, store, setup, cfg, overlap=overlap)
        expected: list[dict] = []
        paths: list[Path] = []
        clock = dict(scalars)
        with tsout.StoreHistoryWriter(frame, cfg, title="gate") as writer:
            for index in range(nsteps + 1):
                if index:
                    stream(store, cfg, clock, 1, tile_nx=tile_nx,
                           tile_ny=tile_ny)
                path = work / f"wrfout_{index:02d}.nc"
                # Snapshot for the comparison BEFORE submitting, so a writer
                # that published a later generation is visible.
                expected.append({name: np.array(value, copy=True)
                                 for name, value in frame.fields().items()})
                writer.submit(path, f"0001-01-01_00:00:{index * 3:02d}")
                paths.append(path)
            written = writer.close()
        import netCDF4

        bad = []
        for path, want in zip(paths, expected):
            with netCDF4.Dataset(path, "r") as ds:
                for name, value in want.items():
                    if name not in ds.variables:
                        bad.append((path.name, name, "missing"))
                        continue
                    stored = np.ascontiguousarray(
                        np.ma.getdata(ds.variables[name][:]))
                    got = stored[0] if stored.ndim == np.ndim(value) + 1 \
                        else stored
                    if not np.array_equal(np.ascontiguousarray(got),
                                          np.ascontiguousarray(value)):
                        bad.append((path.name, name, "differs"))
        outcomes["overlap" if overlap else "synchronous"] = {
            "frames": len(written),
            "bad": bad[:6],
            "ok": not bad and len(written) == nsteps + 1,
            "snapshot_MB": frame.snapshot_bytes / 1e6,
        }
        shutil.rmtree(work, ignore_errors=True)

    record = {"rung": rung, "outcomes": outcomes,
              "ok": all(o["ok"] for o in outcomes.values())}
    store.free()
    _free_device()
    return record


def negative_overlap_aliasing(rung: str = "mp10 Morrison", *,
                              tile_nx: int = 48, tile_ny: int = 40) -> dict:
    """``overlap=False`` hands out VIEWS.  Prove it, so ``overlap=True`` means
    something.

    A CONTROL THAT MUST FAIL, and stated as a property rather than as a race:
    take a frame, step the store, and ask whether the frame's carrier arrays
    changed underneath it.

    * ``overlap=False`` -- they DO.  The carriers are zero-copy views of the
      pinned store, which is what makes the frame free, and it is why the
      synchronous mode must finish its write before the solver resumes.
    * ``overlap=True`` -- they do NOT.  The carriers were copied into the
      double buffer, so the writer thread may hold them across any number of
      sweeps.

    Timing this as a race against a background writer would give a control
    that passes by luck on a fast disk.  This one cannot.
    """
    cfg, state = build_state(rung)
    plan = tsout.frame_plan(state)
    store = make_store(cfg, state)
    store.fill_from(state)
    setup = tsck.DomainSetup.capture(state, cfg)
    scalars = physinv.carrier_scalars(state)
    del state
    _free_device()

    outcomes = {}
    for overlap in (False, True):
        frame = tsout.StoreFrame(plan, store, setup, cfg, overlap=overlap)
        fields = frame.fields()
        carriers = plan.names(tsout.SOURCE_CARRIER)
        before = {name: np.array(fields[name], copy=True)
                  for name in carriers}
        stream(store, cfg, dict(scalars), 1, tile_nx=tile_nx,
               tile_ny=tile_ny)
        moved = sorted(name for name in carriers
                       if not np.array_equal(
                           np.ascontiguousarray(fields[name]),
                           np.ascontiguousarray(before[name])))
        outcomes["overlap" if overlap else "zero-copy"] = {
            "carriers": len(carriers),
            "changed_under_the_frame": len(moved),
            "examples": moved[:4],
        }
    record = {
        "rung": rung,
        "outcomes": outcomes,
        "caught": (outcomes["zero-copy"]["changed_under_the_frame"] > 0
                   and outcomes["overlap"]["changed_under_the_frame"] == 0),
    }
    store.free()
    _free_device()
    return record


def case_reductions_per_tile(rung: str = "full(real74) +KF", *,
                             tile_nx: int = 32, tile_ny: int = 24) -> dict:
    """The one class of diagnostic that is NOT a per-cell function: reductions.

    ``gpuwm/runtime.py:2262`` puts five WHOLE-DOMAIN reductions in every
    checkpoint's ``run_trackers`` -- ``w_max_ms``, its boundary/interior
    split, ``swdown_peak_wm2`` and the ``nan_free`` flag -- and every one of
    them is a ``cp.max``/any over the full field.  A tile's answer is not the
    domain's answer, so they must be REDUCED across tiles rather than read off
    whichever tile was last in the buffer.

    Two ways to do that and only one of them works, which is what this
    measures:

    ``over tile INTERIORS``
        BIT-EXACT for MAX (and for ANY): the interiors partition the domain
        and max is associative in the exact sense.  NOT bit-exact for a SUM,
        and this is the part worth knowing -- folding per-tile partial sums
        changes the summation ORDER, and float addition is not associative,
        so the answer agrees to rounding and never to the byte.  MEASURED
        below as a relative error.
    ``over tile COMPUTE WINDOWS`` (interior + halo)
        still exact for MAX and ANY, because they are idempotent and a halo
        cell counted twice is a cell counted once; WRONG for a sum, a mean or
        a count by exactly the halo redundancy -- MEASURED here as a factor
        of 5.6 at halo 16 with 24x32 tiles, which is the same
        ``(T + 2*halo)^2 / T^2`` the tiling tax is quoted in.

    So the rule for a streamed run: idempotent reductions may be folded from
    the tile BUFFER and are bit-exact; additive ones must be folded from the
    INTERIOR window and are still only exact to rounding.  Nothing in a
    wrfout FRAME is a reduction -- every field there is pointwise,
    column-local, or a short horizontal stencil -- so this constrains the run
    summary and the checkpoint header, not history output.
    """
    cfg, state = build_state(rung)
    store = make_store(cfg, state)
    store.fill_from(state)
    field = np.asarray(store.arrays["state/w"])
    halo = harness.halo_radius(cfg)
    specs = tspec.plan_tiles(cfg.nx, cfg.ny, tile_nx, tile_ny, halo, True)

    def window(spec, gathered: bool):
        if not gathered:
            return field[:, spec.j0:spec.j1, spec.i0:spec.i1]
        rows = np.arange(spec.cj0, spec.cj0 + spec.cny) % cfg.ny
        cols = np.arange(spec.ci0, spec.ci0 + spec.cnx) % cfg.nx
        return field[:, rows[:, None], cols[None, :]]

    truth_max = float(field.max())
    truth_sum = float(field.astype(np.float64).sum())
    results = {}
    for label, gathered in (("interiors", False), ("compute windows", True)):
        pieces = [window(spec, gathered) for spec in specs]
        folded_max = float(max(float(piece.max()) for piece in pieces))
        folded_sum = float(sum(piece.astype(np.float64).sum()
                               for piece in pieces))
        results[label] = {
            "max_exact": folded_max == truth_max,
            "sum_exact": folded_sum == truth_sum,
            "sum_ratio": folded_sum / truth_sum if truth_sum else float("nan"),
            "sum_rel_error": abs(folded_sum - truth_sum) / abs(truth_sum)
            if truth_sum else float("nan"),
        }
    redundancy = sum(spec.cnx * spec.cny for spec in specs) / float(
        cfg.nx * cfg.ny)
    record = {
        "rung": rung,
        "tiles": len(specs),
        "halo": halo,
        "redundancy": redundancy,
        "results": results,
        # The verdict is the whole finding: MAX survives both folds
        # bit-exactly, an interior SUM is right to rounding, and a windowed
        # SUM is wrong by the redundancy factor -- three different outcomes
        # that a single "close enough" check would have collapsed into one.
        # The windowed ratio is only APPROXIMATELY the redundancy, because
        # the halo cells it double-counts are not a uniform sample of the
        # field; 5% is loose enough not to depend on the seed and tight
        # enough that nothing but double-counting produces it.
        "caught": (results["interiors"]["max_exact"]
                   and results["compute windows"]["max_exact"]
                   and results["interiors"]["sum_rel_error"] < 1e-12
                   and abs(results["compute windows"]["sum_ratio"]
                           / redundancy - 1.0) < 0.05),
    }
    store.free()
    del state
    _free_device()
    return record


def case_diagnostic_sources(rung: str = "full(real74) +KF",
                            **overrides) -> dict:
    """Which frame fields the carrier store CANNOT serve, and what they cost.

    The brief's question, answered by measurement rather than by reading the
    code: build the real device frame, classify every field against the
    carrier manifest and the setup arrays, and report what is left over.
    """
    cfg, state = build_state(rung, **overrides)
    plan = tsout.frame_plan(state)
    extended = tsout.frame_plan(
        state, extra_available=tsout.diagnostic_inventory(state))
    cost = tsout.scatter_cost(state, cfg)
    record = {
        "rung": rung + ("".join(f" + {k}={v}" for k, v in overrides.items())),
        "summary": plan.summary(),
        "unavailable": {name: plan.origin[name] for name in plan.unavailable},
        "after_scatter": extended.unavailable,
        "carrier_bytes_per_cell": round(cost["carriers"], 3),
        "diagnostic_bytes_per_cell": round(cost["diagnostics"], 4),
        "diagnostic_keys": cost["names"],
    }
    del state
    _free_device()
    return record


def case_diagnostic_scatter(rung: str = "full fast cadence", *,
                            tile_nx: int = 48, tile_ny: int = 40,
                            nsteps: int = 1, **overrides) -> dict:
    """Scatter ``OLR`` like a carrier and it is bit-exact; do not and it is not.

    Positive: a streamed run whose ``inventory_fn`` is
    :func:`tilestream.output.diagnostic_inventory` gathers and scatters the
    output-only driver diagnostics along with the carriers, and the assembled
    domain field equals the monolithic one exactly.  Each tile's own
    radiation call computes its own window correctly -- radiation is
    column-local -- so nothing but the transport was ever missing.

    NEGATIVE: the same run without the scatter.  The naive implementation
    reads the diagnostic off the tile buffer after the sweep, and that buffer
    holds the LAST tile's window, so the published field is wrong over
    ``(tiles-1)/tiles`` of the domain.  Both halves are asserted.
    """
    import cupy as cp

    cfg, state = build_state(rung, warmup=1, **overrides)
    inv = tsout.diagnostic_inventory(state)
    diag_keys = tuple(k for k in inv if k.startswith("diag/"))
    if not diag_keys:
        return {"rung": rung, "skipped": "no output-only driver diagnostics"}

    start = {k: _as_numpy(v).copy() for k, v in inv.items()}
    start_scalars = physinv.carrier_scalars(state)
    harness.run_steps(state, cfg, nsteps)
    reference = {k: _as_numpy(v).copy()
                 for k, v in tsout.diagnostic_inventory(state).items()}
    radiation_calls = (physinv.carrier_scalars(state)["call_counts"]
                       .get("radiation", 0)
                       - start_scalars["call_counts"].get("radiation", 0))
    del state
    _free_device()

    manifest = hoststore.manifest_from_arrays(start, cfg.nz, cfg.ny, cfg.nx)
    store = hoststore.HostDomainStore(
        cfg, manifest=manifest, inventory_fn=tsout.diagnostic_inventory)
    for key, value in start.items():
        store.arrays[key][...] = value
    scalars = dict(start_scalars)
    stream(store, cfg, scalars, nsteps, tile_nx=tile_nx, tile_ny=tile_ny,
           inventory_fn=tsout.diagnostic_inventory)
    scattered_ok = {
        key: np.array_equal(np.ascontiguousarray(store.arrays[key]),
                            np.ascontiguousarray(reference[key]))
        for key in diag_keys}

    # The naive path: publish the tile BUFFER after the sweep.  A buffer is
    # one tile wide and the last tile through it is the one that wrote it, so
    # every tile's slot in the published field carries the LAST tile's
    # values.  Reconstructed here from the reference field rather than read
    # out of the driver, so the comparison is against the same numbers the
    # scattered run produced -- what is under test is the transport, not the
    # physics.  Requires equal tiles, which 96x80 with 48x40 gives exactly.
    specs = tspec.plan_tiles(cfg.nx, cfg.ny, tile_nx, tile_ny,
                             harness.halo_radius(cfg), True)
    shapes = {(s.j1 - s.j0, s.i1 - s.i0) for s in specs}
    if len(shapes) != 1:
        raise ValueError(
            f"this control needs a tile that divides the domain; {tile_ny}x"
            f"{tile_nx} on {cfg.ny}x{cfg.nx} gives windows {sorted(shapes)}")
    naive_wrong = {}
    varies = {}
    for key in diag_keys:
        truth = reference[key]
        last = specs[-1]
        window = truth[..., last.j0:last.j1, last.i0:last.i1]
        naive = np.empty_like(truth)
        for spec in specs:
            naive[..., spec.j0:spec.j1, spec.i0:spec.i1] = window
        naive_wrong[key] = int(np.count_nonzero(naive != truth))
        # A diagnostic that is the same in every tile window cannot tell the
        # scattered path from the naive one, so the NEGATIVE half of this
        # control has nothing to say about it and must not be counted as
        # having fired.  Reported rather than silently skipped.
        varies[key] = bool(naive_wrong[key])

    record = {
        "rung": rung,
        "tiles": len(specs),
        "radiation_calls_in_window": radiation_calls,
        "diagnostics": diag_keys,
        "scattered_bitexact": all(scattered_ok.values()),
        "unscattered_wrong_cells": naive_wrong,
        "domain_cells": int(cfg.ny * cfg.nx),
        "per_diagnostic_shape": {k: tuple(int(s) for s in
                                          np.shape(reference[k]))
                                 for k in diag_keys},
        "varies_between_tiles": varies,
        # The negative half only counts where the field actually varies
        # between tile windows; the positive half must hold everywhere.
        "caught": (all(scattered_ok.values())
                   and any(varies.values())),
    }
    store.free()
    _free_device()
    return record


# --------------------------------------------------------------------------
# C.  the restart
# --------------------------------------------------------------------------

def case_header_equivalence(rung: str) -> dict:
    """A streamed checkpoint must BE a monolithic checkpoint, not resemble one.

    Header keys, header values, member set, member ORDER and member bytes,
    all compared against a real ``restart.write_restart`` from the resident
    state the store was filled from.  ``created`` is a wall-clock timestamp
    and is the only permitted difference.

    This comparison is also the drift guard for
    :func:`tilestream.checkpoint.store_restart_header`: if restart.py grows a
    header key or bumps ``RESTART_FORMAT_VERSION``, this fails.
    """
    from gpuwm.io import restart

    cfg, state = build_state(rung)
    work = WORK_DIR / f"hdr-{abs(hash(rung)):x}"
    work.mkdir(parents=True, exist_ok=True)
    mono = restart.write_restart(work / "mono.gpuwmrst", state, cfg)
    store = make_store(cfg, state)
    store.fill_from(state)
    setup = tsck.DomainSetup.capture(state, cfg)
    streamed = tsck.write_store_restart(
        work / "streamed.gpuwmrst", store,
        physinv.carrier_scalars(state), setup, cfg)

    mono_header, mono_arrays = restart._load_restart(mono, with_arrays=True)
    st_header, st_arrays = restart._load_restart(streamed, with_arrays=True)
    value_diff = sorted(
        key for key in set(mono_header) | set(st_header)
        if key != "created" and mono_header.get(key) != st_header.get(key))
    member_bytes_diff = sorted(
        key for key in set(mono_arrays) & set(st_arrays)
        if not np.array_equal(
            np.ascontiguousarray(mono_arrays[key]).view(np.uint8),
            np.ascontiguousarray(st_arrays[key]).view(np.uint8)))
    record = {
        "rung": rung,
        "members": len(st_arrays),
        "keys_equal": set(mono_header) == set(st_header),
        "values_equal": not value_diff,
        "value_diff": value_diff,
        "member_set_equal": set(mono_arrays) == set(st_arrays),
        "member_order_equal": list(mono_arrays) == list(st_arrays),
        "member_bytes_equal": not member_bytes_diff,
        "member_bytes_diff": member_bytes_diff,
        "mono_bytes": mono.stat().st_size,
        "streamed_bytes": streamed.stat().st_size,
    }
    record["equivalent"] = (record["keys_equal"] and record["values_equal"]
                            and record["member_set_equal"]
                            and record["member_order_equal"]
                            and record["member_bytes_equal"])
    store.free()
    del state
    _free_device()
    shutil.rmtree(work, ignore_errors=True)
    return record


def case_roundtrip(rung: str, *, tile_nx: int = 48, tile_ny: int = 40,
                   nsteps: int = 4, resume_at: int = 2) -> dict:
    """The whole point: N steps, checkpoint, resume, continue -- bit-exact.

    Four legs against one uninterrupted monolithic reference; see the module
    docstring's table for what each one is for.
    """
    import cupy as cp

    from gpuwm.io import restart

    work = WORK_DIR / f"rt-{abs(hash(rung)):x}"
    work.mkdir(parents=True, exist_ok=True)
    cfg, state = build_state(rung)
    inv = physinv.carrier_inventory(state)
    manifest = hoststore.manifest_from_arrays(inv, cfg.nz, cfg.ny, cfg.nx)
    start = {k: _as_numpy(v).copy() for k, v in inv.items()}
    start_scalars = physinv.carrier_scalars(state)
    setup = tsck.DomainSetup.capture(state, cfg)
    restart.write_restart(work / "start.gpuwmrst", state, cfg)

    harness.run_steps(state, cfg, nsteps)
    reference = physinv.field_digests(physinv.carrier_manifest(state))
    ref_scalars = physinv.carrier_scalars(state)
    del state
    _free_device()

    def fresh_store():
        store = hoststore.HostDomainStore(
            cfg, manifest=manifest, inventory_fn=physinv.carrier_inventory)
        return store

    def loaded_store():
        store = fresh_store()
        for key, value in start.items():
            store.arrays[key][...] = value
        return store

    legs: dict[str, dict] = {}

    # Leg 1: streamed straight through.
    store = loaded_store()
    scalars = dict(start_scalars)
    stream(store, cfg, scalars, nsteps, tile_nx=tile_nx, tile_ny=tile_ny)
    legs["streamed"] = {
        "bitexact": physinv.field_digests(store.arrays) == reference,
        "scalars": scalars == ref_scalars}
    store.free()
    _free_device()

    # Leg 2: streamed -> checkpoint -> streamed.
    store = loaded_store()
    scalars = dict(start_scalars)
    stream(store, cfg, scalars, resume_at, tile_nx=tile_nx, tile_ny=tile_ny)
    mid = tsck.write_store_restart(work / "mid.gpuwmrst", store, scalars,
                                   setup, cfg)
    store.free()
    _free_device()
    resumed = fresh_store()
    resumed_scalars = tsck.read_store_restart(mid, resumed, setup, cfg)
    stream(resumed, cfg, resumed_scalars, nsteps - resume_at,
           tile_nx=tile_nx, tile_ny=tile_ny)
    legs["streamed+restart+streamed"] = {
        "bitexact": physinv.field_digests(resumed.arrays) == reference,
        "scalars": resumed_scalars == ref_scalars}
    resumed.free()
    _free_device()

    # Leg 3: streamed -> checkpoint -> MONOLITHIC.  The file has to be a real
    # gpuwm restart for this to work at all: ``restore_restart`` is unmodified.
    _cfg, resident = build_state(rung)
    info = restart.restore_restart(mid, resident, cfg)
    harness.run_steps(resident, cfg, nsteps - resume_at)
    legs["streamed+restart+monolithic"] = {
        "bitexact": physinv.field_digests(
            physinv.carrier_manifest(resident)) == reference,
        "scalars": physinv.carrier_scalars(resident) == ref_scalars,
        "elapsed": info.elapsed_seconds}
    del resident
    _free_device()

    # Leg 4: MONOLITHIC -> checkpoint -> streamed.
    _cfg, resident = build_state(rung)
    harness.run_steps(resident, cfg, resume_at)
    mono_mid = restart.write_restart(work / "mono-mid.gpuwmrst", resident, cfg)
    del resident
    _free_device()
    store = fresh_store()
    mono_scalars = tsck.read_store_restart(mono_mid, store, setup, cfg)
    stream(store, cfg, mono_scalars, nsteps - resume_at,
           tile_nx=tile_nx, tile_ny=tile_ny)
    legs["monolithic+restart+streamed"] = {
        "bitexact": physinv.field_digests(store.arrays) == reference,
        "scalars": mono_scalars == ref_scalars}
    store.free()
    _free_device()

    record = {
        "rung": rung,
        "nsteps": nsteps,
        "resume_at": resume_at,
        "carriers": len(start),
        "legs": legs,
        "bitexact": all(leg["bitexact"] and leg["scalars"]
                        for leg in legs.values()),
        "checkpoint_bytes": mid.stat().st_size,
        "setup_bytes_per_cell": round(setup.bytes_per_cell(cfg), 4),
    }
    shutil.rmtree(work, ignore_errors=True)
    return record


# --------------------------------------------------------------------------
# C negatives
# --------------------------------------------------------------------------

def negative_tile_fingerprint(rung: str = "full(real74) +KF") -> dict:
    """A tile-derived setup fingerprint: refused by one reader, not the other.

    A CONTROL THAT MUST FAIL, and the sharpest one in this file.  Computing
    the header's ``setup_fingerprint`` from the tile that happens to be in
    the buffer is the obvious shortcut, and its consequence is not that the
    checkpoint stops working -- it is that a second streamed run making the
    same shortcut accepts it, so the fingerprint stops protecting anything.
    Both halves are asserted: the monolithic reader must REFUSE, and the
    streamed reader carrying the same wrong setup must ACCEPT, which is the
    silence being measured.
    """
    from gpuwm.io import restart

    work = WORK_DIR / "neg-fingerprint"
    work.mkdir(parents=True, exist_ok=True)
    cfg, state = build_state(rung)
    store = make_store(cfg, state)
    store.fill_from(state)
    scalars = physinv.carrier_scalars(state)
    good = tsck.DomainSetup.capture(state, cfg)

    tile_cfg = harness.tile_config(cfg, 48, 40)
    tile_state, _drv = physinv.default_builder(tile_cfg, SEED)
    harness.run_steps(tile_state, tile_cfg, 1)
    bad = tsck.DomainSetup.capture(
        tile_state, cfg,
        physics_setup=tsck.domain_physics_setup(
            tile_state, cfg,
            latitude=state.physics.radiation_callable.latitude_deg,
            longitude=state.physics.radiation_callable.longitude_deg))
    path = tsck.write_store_restart(work / "tile-setup.gpuwmrst", store,
                                    scalars, bad, cfg)

    monolithic_refused = None
    _cfg, resident = build_state(rung)
    try:
        restart.restore_restart(path, resident, cfg)
    except restart.RestartMismatchError as exc:
        monolithic_refused = str(exc).splitlines()[0][:110]
    del resident
    _free_device()

    streamed_refused = None
    victim = make_store(cfg, state)
    try:
        tsck.read_store_restart(path, victim, bad, cfg)
    except restart.RestartMismatchError as exc:
        streamed_refused = str(exc).splitlines()[0][:110]

    record = {
        "domain_fingerprint": good.setup_fingerprint[:16],
        "tile_fingerprint": bad.setup_fingerprint[:16],
        "monolithic_refused": monolithic_refused,
        "streamed_accepted_the_same_file": streamed_refused is None,
        "caught": (monolithic_refused is not None
                   and good.setup_fingerprint != bad.setup_fingerprint),
    }
    victim.free()
    store.free()
    del state, tile_state
    _free_device()
    shutil.rmtree(work, ignore_errors=True)
    return record


def negative_tile_physics_identity(rung: str = "full+MYNN+Noah-MP") -> dict:
    """The physics identity taken from a tile, without the domain's lat/lon.

    A CONTROL THAT MUST FAIL.  The identity is mostly config- and
    asset-derived, so a tile gets nearly all of it right; what it gets wrong
    is exactly the domain-shaped geography -- the radiation grid at every
    radiation rung, and Noah-MP's solar geometry under
    ``sf_surface_physics=4``.  A checkpoint carrying it must be refused.
    """
    from gpuwm.io import restart

    work = WORK_DIR / "neg-identity"
    work.mkdir(parents=True, exist_ok=True)
    cfg, state = build_state(rung)
    store = make_store(cfg, state)
    store.fill_from(state)
    scalars = physinv.carrier_scalars(state)

    tile_cfg = harness.tile_config(cfg, 48, 40)
    tile_state, _drv = physinv.default_builder(tile_cfg, SEED)
    harness.run_steps(tile_state, tile_cfg, 1)
    # Setup arrays from the DOMAIN so the setup fingerprint matches and this
    # control isolates the physics identity and nothing else.
    unsubstituted = tsck.domain_physics_setup(tile_state, cfg)
    substituted = tsck.domain_physics_setup(
        tile_state, cfg,
        latitude=state.physics.radiation_callable.latitude_deg,
        longitude=state.physics.radiation_callable.longitude_deg)
    domain = tsck.DomainSetup.capture(state, cfg)
    bad = tsck.DomainSetup.capture(state, cfg,
                                   physics_setup=unsubstituted)
    path = tsck.write_store_restart(work / "tile-physics.gpuwmrst", store,
                                    scalars, bad, cfg)

    refused = None
    _cfg, resident = build_state(rung)
    try:
        restart.restore_restart(path, resident, cfg)
    except restart.RestartMismatchError as exc:
        refused = str(exc).splitlines()[0][:110]
    del resident
    _free_device()

    record = {
        "rung": rung,
        "substitution_reproduces_domain":
            substituted == domain.physics_setup,
        "unsubstituted_differs": unsubstituted != domain.physics_setup,
        "monolithic_refused": refused,
        "caught": refused is not None
        and substituted == domain.physics_setup,
    }
    store.free()
    del state, tile_state
    _free_device()
    shutil.rmtree(work, ignore_errors=True)
    return record


def negative_partial_inventory(rung: str = "full(real74) +KF") -> dict:
    """A checkpoint carrying only ``state/*`` must be refused, not resumed.

    A CONTROL THAT MUST FAIL.  ``state/*`` alone is EXACTLY milestone one's
    streaming inventory, and at a physics rung it is 46 of 155 carriers: a
    resume that accepted it would continue with dirty precipitation
    accumulators, dirty held tendencies and a dirty surface.
    """
    from gpuwm.io import restart

    work = WORK_DIR / "neg-partial"
    work.mkdir(parents=True, exist_ok=True)
    cfg, state = build_state(rung)
    store = make_store(cfg, state)
    store.fill_from(state)
    setup = tsck.DomainSetup.capture(state, cfg)
    scalars = physinv.carrier_scalars(state)
    partial = {k: v for k, v in store.arrays.items()
               if k.startswith("state/")}
    path = tsck.write_store_restart(work / "partial.gpuwmrst", partial,
                                    scalars, setup, cfg)

    streamed_refused = None
    try:
        tsck.read_store_restart(path, store, setup, cfg)
    except restart.RestartMismatchError as exc:
        streamed_refused = str(exc).splitlines()[0][:110]
    monolithic_refused = None
    _cfg, resident = build_state(rung)
    try:
        restart.restore_restart(path, resident, cfg)
    except restart.RestartMismatchError as exc:
        monolithic_refused = str(exc).splitlines()[0][:110]
    del resident
    _free_device()

    record = {
        "carriers": len(store.arrays),
        "written": len(partial),
        "streamed_refused": streamed_refused,
        "monolithic_refused": monolithic_refused,
        "caught": streamed_refused is not None
        and monolithic_refused is not None,
    }
    store.free()
    del state
    _free_device()
    shutil.rmtree(work, ignore_errors=True)
    return record


def negative_dropped_scalars(rung: str, *, tile_nx: int = 48,
                             tile_ny: int = 40, nsteps: int = 8,
                             resume_at: int = 4) -> dict:
    """Resume without the clock, and without the driver counters.

    A CONTROL THAT MUST FAIL -- AT THE RIGHT RUNG.  The clock drives every
    physics cadence test, so dropping it changes the forecast only if the
    resumed window actually crosses a cadence boundary.  Run this at
    ``full(real74) +KF`` (radt 12 min, cudt 5 min) and eight 3-second steps
    never reach one, so a dropped clock is INVISIBLE and the control passes
    while the defect is present.  That is FACT 1 in a new place, and it is
    why the caller runs this at ``full fast cadence`` as well and the report
    prints both.
    """
    work = WORK_DIR / f"neg-scalars-{abs(hash(rung)):x}"
    work.mkdir(parents=True, exist_ok=True)
    cfg, state = build_state(rung)
    inv = physinv.carrier_inventory(state)
    manifest = hoststore.manifest_from_arrays(inv, cfg.nz, cfg.ny, cfg.nx)
    start = {k: _as_numpy(v).copy() for k, v in inv.items()}
    start_scalars = physinv.carrier_scalars(state)
    setup = tsck.DomainSetup.capture(state, cfg)
    harness.run_steps(state, cfg, nsteps)
    reference = physinv.field_digests(physinv.carrier_manifest(state))
    del state
    _free_device()

    def fresh():
        store = hoststore.HostDomainStore(
            cfg, manifest=manifest, inventory_fn=physinv.carrier_inventory)
        return store

    store = fresh()
    for key, value in start.items():
        store.arrays[key][...] = value
    scalars = dict(start_scalars)
    stream(store, cfg, scalars, resume_at, tile_nx=tile_nx, tile_ny=tile_ny)
    mid = tsck.write_store_restart(work / "mid.gpuwmrst", store, scalars,
                                   setup, cfg)
    store.free()
    _free_device()

    outcomes = {}
    for label, mutate in (
            ("clock dropped", lambda s: {**s, "elapsed_seconds": 0.0}),
            # Zeroed, not deleted: the driver reads its own keys by name,
            # so an EMPTY dict raises instead of integrating.  Zeros are what
            # a cold-started driver carries, i.e. exactly what a resume that
            # forgot to restore the counters would run with.
            ("counters dropped", lambda s: {
                **s, "call_counts": {k: 0 for k in s["call_counts"]},
                "microphysics_updates": 0, "ysu_nan_guard_fires": 0}),
            ("restored (control)", lambda s: s)):
        store = fresh()
        loaded = mutate(tsck.read_store_restart(mid, store, setup, cfg))
        stream(store, cfg, loaded, nsteps - resume_at,
               tile_nx=tile_nx, tile_ny=tile_ny)
        digests = physinv.field_digests(store.arrays)
        differing = sorted(k for k in reference if reference[k] != digests[k])
        outcomes[label] = {"bitexact": not differing,
                           "differing": len(differing),
                           "carriers": len(reference)}
        store.free()
        _free_device()

    record = {
        "rung": rung,
        "radt_minutes": float(cfg.radt_minutes),
        "cudt_minutes": float(cfg.cudt_minutes),
        "window_seconds": (nsteps - resume_at) * float(cfg.dt),
        "outcomes": outcomes,
        "caught": (not outcomes["clock dropped"]["bitexact"]
                   and not outcomes["counters dropped"]["bitexact"]
                   and outcomes["restored (control)"]["bitexact"]),
    }
    shutil.rmtree(work, ignore_errors=True)
    return record


# --------------------------------------------------------------------------
# cost
# --------------------------------------------------------------------------

#: Cost shapes.  Deliberately NON-SQUARE, and not by preference:
#: ``hoststore.manifest_from_arrays`` refuses ``ny == nx`` outright so a y/x
#: transposition cannot pass unnoticed while the manifest is being derived.
#: They stop at 640x624 because the RESIDENT half of the comparison has to
#: fit on the card next to whatever else is running on it: a moist state at
#: this rung costs ~0.38 KB/cell of VRAM, so 640x624x49 is 7.4 GiB and
#: 1024x1008x49 would be 19.  The quantity reported is ns per cell and both
#: paths are linear in bytes, so the small shapes extrapolate.
COST_SHAPES = ((384, 368), (512, 496), (640, 624))


def case_frame_cost(rung: str = "mp10 Morrison", nx: int = 512,
                    ny: int = 496, reps: int = 5) -> dict:
    """Frame cost from a host store versus from a resident device state.

    Both paths are timed to the SAME endpoint -- a complete host frame dict
    ready to hand to ``WrfoutWriter`` -- because that is where the two paths
    converge and the netCDF write that follows is measurably identical for
    both (§12.3: the two write medians differ by 2.3-6.2% while their own
    spreads are 11-51%).

    Timed in its own loop, not interleaved with the write: the quantity is
    ~30 ms and the write it sits in front of is ~1800 ms, and interleaving
    them leaves the median unmoved while taking the spread from 12% to 549%.
    Minimum over ``reps`` is reported alongside the median because another
    lane may hold this GPU; contention can only ADD time, so the minimum is
    the uncontended floor.
    """
    import cupy as cp

    from gpuwm.io import wrfout

    cfg = test_gate.physics_cfg(rung, nx, ny, NZ)
    state, _drv = physinv.default_builder(cfg, SEED)
    harness.run_steps(state, cfg, 1)
    plan = tsout.frame_plan(state)
    store = make_store(cfg, state)
    store.fill_from(state)
    setup = tsck.DomainSetup.capture(state, cfg)
    frame = tsout.StoreFrame(plan, store, setup, cfg)
    snapshot_frame = tsout.StoreFrame(plan, store, setup, cfg, overlap=True)

    def timed(fn, n):
        samples = []
        for _ in range(n):
            cp.cuda.runtime.deviceSynchronize()
            t0 = time.perf_counter()
            fn()
            cp.cuda.runtime.deviceSynchronize()
            samples.append(time.perf_counter() - t0)
        return samples

    # The resident path is ``AsyncDomainWrfoutWriter.submit``'s staging:
    # device DERIVE (the T/P arithmetic and the PB/PHB broadcast expansion
    # ``ascontiguousarray`` is forced to materialise), then a D2H into PINNED
    # destinations on a non-blocking side stream.  The destinations are
    # allocated once and REUSED, because production recycles through CuPy's
    # pinned pool -- a cold pinned allocation is 100.48 ms against 0.04 ms
    # recycled, and timing the cold one is the artefact that produced this
    # project's seventh false result.  Pageable destinations would be a
    # different measurement entirely: 15 GB/s against 50-57.
    side = cp.cuda.Stream(non_blocking=True)
    pinned_dest: dict[str, np.ndarray] = {}
    pinned_refs: list[object] = []

    def resident():
        producer = cp.cuda.get_current_stream()
        ready = cp.cuda.Event()
        fields = wrfout._device_state_frame(state)
        host: dict[str, np.ndarray] = {}
        contiguous = []
        with side:
            for name, value in fields.items():
                if isinstance(value, np.ndarray):
                    host[name] = np.array(value, copy=True, order="C")
                    continue
                contiguous.append((name, cp.ascontiguousarray(value)))
        ready.record(producer)
        side.wait_event(ready)
        with side:
            for name, array in contiguous:
                dest = pinned_dest.get(name)
                if dest is None or dest.shape != array.shape:
                    memory = cp.cuda.alloc_pinned_memory(int(array.nbytes))
                    dest = np.frombuffer(memory, dtype=array.dtype,
                                         count=array.size).reshape(array.shape)
                    pinned_dest[name] = dest
                    pinned_refs.append(memory)
                array.get(out=dest, stream=side, blocking=False)
                host[name] = dest
            done = cp.cuda.Event()
            done.record(side)
        producer.wait_event(done)
        done.synchronize()
        return host

    resident()                     # allocate the pinned destinations and warm
    frame.fields()                 # every first-touch page OUTSIDE the timer
    snapshot_frame.fields()

    res = timed(resident, reps)
    ooc = timed(frame.fields, reps)
    snap = timed(snapshot_frame.fields, reps)
    cells = float(nx * ny * NZ)
    payload = sum(int(np.asarray(v).nbytes) for v in frame.fields().values())

    record = {
        "rung": rung, "nx": nx, "ny": ny, "nz": NZ,
        "fields": len(plan.order),
        "frame_MB": payload / 1e6,
        "resident_ms": (min(res) * 1e3, float(np.median(res)) * 1e3),
        "store_ms": (min(ooc) * 1e3, float(np.median(ooc)) * 1e3),
        "store_overlap_ms": (min(snap) * 1e3, float(np.median(snap)) * 1e3),
        "resident_ns_per_cell": min(res) / cells * 1e9,
        "store_ns_per_cell": min(ooc) / cells * 1e9,
        "store_overlap_ns_per_cell": min(snap) / cells * 1e9,
        "snapshot_MB": snapshot_frame.snapshot_bytes / 1e6,
        "setup_bytes_per_cell": setup.bytes_per_cell(cfg),
    }
    store.free()
    del state
    _free_device()
    return record


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

def _ok(flag: bool) -> str:
    return "PASS" if flag else "FAIL"


def main(argv=None) -> int:
    import cupy as cp

    argv = list(sys.argv[1:] if argv is None else argv)
    quick = "--quick" in argv
    with_cost = "--cost" in argv

    free, total = cp.cuda.runtime.memGetInfo()
    print(f"cupy {cp.__version__}  "
          f"{cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}  "
          f"{free / 2**30:.1f} GiB free of {total / 2**30:.1f}")
    print(f"domain {NX}x{NY}x{NZ}, tiles 48x40, halo from "
          f"harness.halo_radius\n")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    def run(label, fn, verdict, *args, **kwargs):
        try:
            record = fn(*args, **kwargs)
        except Exception as exc:                       # noqa: BLE001
            failures.append(f"{label}: raised {exc!r}")
            print(f"  ERROR {label}: {exc!r}")
            traceback.print_exc()
            return None
        good = verdict(record)
        if not good:
            failures.append(f"{label}: {record}")
        print(f"  {_ok(good)}  {label}")
        return record

    print("=" * 78)
    print("A.  A WRFOUT FRAME OUT OF THE PINNED STORE")
    print("=" * 78)
    rungs = FRAME_RUNGS[:2] if quick else FRAME_RUNGS
    for rung in rungs:
        rec = run(f"frame bytes == device frame  |  {rung}",
                  case_frame_bytes, lambda r: (r["bitexact"]
                                               and r["variables_identical"]
                                               and r["file_identical"]),
                  rung)
        if rec:
            print(f"        {rec['summary']}")
            print(f"        wrote {rec['written']} of {rec['fields']} "
                  f"fields, sha256 {rec['file_sha256']}, "
                  f"variables identical={rec['variables_identical']}, "
                  f"file identical={rec['file_identical']}")
            if rec["unavailable"]:
                print(f"        NOT available from the carrier store: "
                      f"{list(rec['unavailable'])}")
            if rec["bad"]:
                print(f"        differing: {rec['bad'][:4]}")

    rec = run("NEGATIVE reassociated T is caught", negative_reassociated_t,
              lambda r: r["caught"])
    if rec:
        print(f"        thb runs {rec['thb_range'][0]:.1f} to "
              f"{rec['thb_range'][1]:.1f} K")
        for label, out in rec["outcomes"].items():
            print(f"        {label:12s} {out['differing_bytes']:>9d} of "
                  f"{out['total_bytes']} bytes differ "
                  f"({out['fraction'] * 100:5.1f}%), max |delta| = "
                  f"{out['max_abs']:.3e} K, thp on the ulp(300) grid: "
                  f"{out['thp_on_ulp300_grid'] * 100:6.2f}%")
        print("        ^ ONE dycore step puts every thp on the ulp(300) grid "
              "and the reassociation goes silent:")
        print("          a warmed-up control certifies the wrong "
              "arithmetic.")
    rec = run("NEGATIVE shuffled field order changes the FILE",
              negative_field_order, lambda r: r["caught"])
    if rec:
        print(f"        variables identical={rec['variables_identical']}, "
              f"file identical={rec['file_identical']}, "
              f"size delta {rec['size_delta']:+d} bytes")
    rec = run("history writer thread, both modes, driven by a streamed run",
              case_history_writer, lambda r: r["ok"])
    if rec:
        for label, out in rec["outcomes"].items():
            print(f"        {label:12s} {out['frames']} frames published and "
                  f"reread, mismatches {out['bad'] or 'none'}, snapshot "
                  f"{out['snapshot_MB']:.1f} MB")
    rec = run("NEGATIVE a zero-copy frame moves under the writer",
              negative_overlap_aliasing, lambda r: r["caught"])
    if rec:
        for label, out in rec["outcomes"].items():
            print(f"        {label:10s} {out['changed_under_the_frame']} of "
                  f"{out['carriers']} carriers changed under the frame after "
                  f"one sweep {out['examples']}")
    print()

    print("=" * 78)
    print("B.  DIAGNOSTICS: WHAT CAN BE COMPUTED PER TILE, AND WHAT CANNOT")
    print("=" * 78)
    rec = run("derived fields are separable per tile", case_derives_per_tile,
              lambda r: r["bitexact"])
    if rec:
        print(f"        T and P assembled from {rec['windows']} disjoint "
              f"{rec['tile'][0]}x{rec['tile'][1]} windows, byte-identical to "
              f"the whole-domain pass; PSFC: {rec['psfc']}")
    rec = run("reductions: interiors exact, windows exact only for MAX",
              case_reductions_per_tile, lambda r: r["caught"])
    if rec:
        for label, out in rec["results"].items():
            print(f"        over tile {label:16s} MAX bit-exact="
                  f"{out['max_exact']}, SUM bit-exact={out['sum_exact']}, "
                  f"sum/domain = {out['sum_ratio']:.4f} "
                  f"(rel err {out['sum_rel_error']:.2e})")
        print(f"        ^ {rec['tiles']} tiles, halo {rec['halo']}, "
              f"redundancy {rec['redundancy']:.4f}x.  MAX folds exactly from "
              f"either window;")
        print("          an interior SUM agrees only to rounding "
              "(reassociation), and a windowed SUM is wrong by the "
              "redundancy.")
    census = [(rung, {}) for rung in
              (FRAME_RUNGS if not quick else FRAME_RUNGS[2:])]
    # One row with the dict-valued diagnostic bundle turned on, so the
    # ``driver.hmix_k_diag['XKMH']`` -> ``diag/hmix_k_diag/XKMH`` naming is
    # exercised rather than merely implemented: OLR is the only unavailable
    # field at every shipped rung, and a census that only ever saw a scalar
    # attribute would not notice the bundle case at all.
    census.append(("full(real74) +KF", {"hmix_k_diag": True}))
    for rung, overrides in census:
        label = rung + "".join(f" + {k}" for k in overrides)
        rec = run(f"frame source census  |  {label}",
                  case_diagnostic_sources, lambda r: True, rung, **overrides)
        if rec:
            print(f"        {rec['summary']}")
            print(f"        unavailable: {rec['unavailable'] or '{}'}  "
                  f"-> after scatter: {list(rec['after_scatter']) or '[]'}  "
                  f"({rec['diagnostic_bytes_per_cell']} B/cell on top of "
                  f"{rec['carrier_bytes_per_cell']})")
    for overrides in ({}, {"hmix_k_diag": True}):
        label = "NEGATIVE unscattered diagnostic is the last tile's" + \
            "".join(f"  |  + {k}" for k in overrides)
        rec = run(label, case_diagnostic_scatter,
                  lambda r: r.get("skipped") or r["caught"], **overrides)
        if rec and not rec.get("skipped"):
            print(f"        {rec['tiles']} tiles, radiation fired "
                  f"{rec['radiation_calls_in_window']}x in the window")
            print(f"        scattered {rec['diagnostics']} bit-exact="
                  f"{rec['scattered_bitexact']}; unscattered wrong in "
                  f"{rec['unscattered_wrong_cells']} of "
                  f"{rec['domain_cells']} columns")
    print()

    print("=" * 78)
    print("C.  RESTART: WRITE FROM THE STORE, READ IT BACK, CONTINUE")
    print("=" * 78)
    for rung in (ROUNDTRIP_RUNGS[:1] if quick else ROUNDTRIP_RUNGS):
        rec = run(f"checkpoint == monolithic checkpoint  |  {rung}",
                  case_header_equivalence, lambda r: r["equivalent"], rung)
        if rec:
            print(f"        {rec['members']} members, "
                  f"{rec['streamed_bytes'] / 1e6:.1f} MB; header keys "
                  f"equal={rec['keys_equal']}, values equal (modulo "
                  f"created)={rec['values_equal']}, member order "
                  f"equal={rec['member_order_equal']}, bytes "
                  f"equal={rec['member_bytes_equal']}")
            if rec["value_diff"]:
                print(f"        header differences: {rec['value_diff']}")
    for rung in (ROUNDTRIP_RUNGS[:1] if quick else ROUNDTRIP_RUNGS):
        rec = run(f"round trip, 4 legs  |  {rung}", case_roundtrip,
                  lambda r: r["bitexact"], rung)
        if rec:
            for leg, result in rec["legs"].items():
                print(f"        {_ok(result['bitexact'] and result['scalars'])}"
                      f"  {leg}")
            print(f"        {rec['carriers']} carriers, checkpoint "
                  f"{rec['checkpoint_bytes'] / 1e6:.1f} MB, setup "
                  f"{rec['setup_bytes_per_cell']} B/cell")
    print()

    print("=" * 78)
    print("C NEGATIVES")
    print("=" * 78)
    rec = run("NEGATIVE tile-derived setup fingerprint",
              negative_tile_fingerprint, lambda r: r["caught"])
    if rec:
        print(f"        domain {rec['domain_fingerprint']} vs tile "
              f"{rec['tile_fingerprint']}")
        print(f"        monolithic reader refused: "
              f"{rec['monolithic_refused']}")
        print(f"        a streamed reader carrying the SAME wrong setup "
              f"accepted it: {rec['streamed_accepted_the_same_file']}"
              f"  <- this is the silence")
    rec = run("NEGATIVE tile-derived physics identity",
              negative_tile_physics_identity, lambda r: r["caught"])
    if rec:
        print(f"        substitution reproduces the domain identity: "
              f"{rec['substitution_reproduces_domain']}; unsubstituted "
              f"differs: {rec['unsubstituted_differs']}")
        print(f"        monolithic reader refused: "
              f"{rec['monolithic_refused']}")
    rec = run("NEGATIVE state/* only is refused", negative_partial_inventory,
              lambda r: r["caught"])
    if rec:
        print(f"        wrote {rec['written']} of {rec['carriers']} carriers")
        print(f"        streamed reader: {rec['streamed_refused']}")
        print(f"        monolithic reader: {rec['monolithic_refused']}")
    for rung, expect in (("full fast cadence", True),
                         ("full(real74) +KF", False)):
        if quick and not expect:
            continue
        rec = run(f"NEGATIVE dropped scalars  |  {rung}",
                  negative_dropped_scalars,
                  (lambda r: r["caught"]) if expect
                  else (lambda r: r["outcomes"]["restored (control)"]
                        ["bitexact"]),
                  rung)
        if rec:
            print(f"        radt {rec['radt_minutes']} min, cudt "
                  f"{rec['cudt_minutes']} min, resumed window "
                  f"{rec['window_seconds']} s")
            for label, out in rec["outcomes"].items():
                print(f"        {label:22s} bit-exact={out['bitexact']}  "
                      f"({out['differing']} of {out['carriers']} carriers "
                      f"differ)")
            if not expect:
                print("        ^ the SAME two defects at a long cadence.  "
                      "The clock is still caught -- it does more than gate "
                      "cadences --")
                print("          but DROPPED COUNTERS are invisible here: "
                      "the cheap configuration certifies that bug.")
    print()

    if with_cost:
        print("=" * 78)
        print("COST: A FRAME FROM THE HOST STORE VS FROM A RESIDENT STATE")
        print("=" * 78)
        for nx, ny in COST_SHAPES:
            rec = run(f"frame cost {nx}x{ny}", case_frame_cost,
                      lambda r: True, nx=nx, ny=ny)
            if rec:
                print(f"        {rec['fields']} fields, "
                      f"{rec['frame_MB']:.0f} MB")
                print(f"        resident (device derive + D2H): "
                      f"{rec['resident_ms'][0]:8.2f} ms min "
                      f"({rec['resident_ns_per_cell']:.3f} ns/cell)")
                print(f"        host store, zero-copy:          "
                      f"{rec['store_ms'][0]:8.2f} ms min "
                      f"({rec['store_ns_per_cell']:.3f} ns/cell)")
                print(f"        host store, overlap snapshot:   "
                      f"{rec['store_overlap_ms'][0]:8.2f} ms min "
                      f"({rec['store_overlap_ns_per_cell']:.3f} ns/cell, "
                      f"+{rec['snapshot_MB']:.0f} MB host RAM)")
        print()

    shutil.rmtree(WORK_DIR, ignore_errors=True)
    print("=" * 78)
    if failures:
        print(f"I/O GATE FAILED -- {len(failures)} problem(s)")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("I/O GATE PASSED -- a streamed run writes a wrfout and resumes "
          "from its own checkpoint, bit-exact, and every negative control "
          "failed as specified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
