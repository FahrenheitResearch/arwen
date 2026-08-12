"""The restart gate: stream N, checkpoint, reload, continue, same digest.

Nothing in this project had ever written a restart from a streamed run.  This
is the proof that one can be written and read back, and it is built the way
every other proof here is built -- against a monolithic reference, with the
controls that make the comparison capable of failing.

THE SHAPE OF THE PROOF
----------------------
For one configuration (a real Lambert projection, real terrain, the
229-carrier ``full+MYNN+Noah-MP`` rung and the fast-cadence rung that is the
only one calling radiation during the compared steps)::

    A   monolithic 2N steps                        -> reference digests
    B   streamed 2N steps, uninterrupted           -> must equal A
    C   streamed N, WRITE restart, fresh store,
        fresh tile buffers, READ restart,
        streamed N more                            -> must equal A and B

C is the claim.  A is what keeps B and C honest: a restart round trip that
reproduced a streamed run which was itself wrong would still pass B == C.

WHAT "FRESH" MEANS, AND WHY IT IS THE WHOLE POINT
-------------------------------------------------
The second half of C never touches the first half's memory.  Its store is a
NEW pinned allocation, its geography store is a NEW pinned allocation, and its
tile buffers are built from scratch by ``run_tiled``.  The only thing that
crosses is the file.  ``preset`` decides what the new store holds BEFORE the
restart is read:

``"start"`` (default)
    the preparation state, i.e. exactly what a real resumed run has after it
    rebuilds its initial condition and before it restores.  A carrier the
    restart fails to deliver silently reverts to t=0 -- plausible values, no
    NaN, no warning.  This is the failure mode the three-inventory mismatch
    produces and it is what the drop control exercises.

``"poison"``
    NaN everywhere.  Bit-exact anyway, which is the statement that nothing in
    the answer came from what the store held before the restore.

THE CONTROLS
------------
``drop=("fields/ust",)  allow_missing=True``   MUST differ.  One carrier left
    out of the file and the reader told not to mind.  ``ust`` is the friction
    velocity: carried by the surface layer, read by the PBL, and in the
    ``fields/*`` half of the manifest that ``STATE_SERIALIZED_ATTRS`` cannot
    see -- so it is exactly the kind of state a contract-only restart drops.

``drop=("fields/ust",)  allow_missing=False``  MUST REFUSE.  The same file
    read normally.  Without this row the first control proves only that a
    broken file breaks; with it, the breakage is unshippable.

``store built from STATE_SERIALIZED_ATTRS``    MUST REFUSE at WRITE.  25
    members offered where the manifest has 229.

``a restart from the wrong setup``             MUST REFUSE at READ.  Same
    config, same carriers, geography rebuilt per tile instead of the
    domain's: the setup fingerprint moves and the file is rejected rather
    than resumed onto a different projection.

Run me: ``python tilestream/test_restart_gate.py [--quick]``.
"""

from __future__ import annotations

import hashlib
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

from tilestream import driver, gather, harness, restart_stream
from tilestream import physics_inventory as physinv
from tilestream import spec as tspec
from tilestream import test_gate as gate


NX, NY, NZ = 96, 80, 49
SEED = 11
WARMUP = 1

#: Where the checkpoints go.  Overridden by ``--dir``; a restart write is an
#: I/O measurement and the file system it lands on is part of the number.
SCRATCH = Path("/tmp/arwen-restart-gate")

_REF_CACHE: dict = {}


def _as_numpy(array) -> np.ndarray:
    import cupy as cp

    return cp.asnumpy(array) if isinstance(array, cp.ndarray) \
        else np.asarray(array)


def digest_of(arrays) -> str:
    """One SHA-256 over every carrier, in sorted key order."""
    digest = hashlib.sha256()
    for key in sorted(arrays):
        host = np.ascontiguousarray(_as_numpy(arrays[key]))
        digest.update(key.encode("utf-8"))
        digest.update(host.dtype.str.encode("ascii"))
        digest.update(np.asarray(host.shape, dtype=np.int64).tobytes())
        digest.update(host.tobytes(order="C"))
    return digest.hexdigest()


def reference(rung, nsteps, *, nx=NX, ny=NY, nz=NZ, seed=SEED):
    """``(cfg, start, start_scalars, geo, ref_digest, ref_fields, setup)``.

    One monolithic run of ``nsteps`` on a real projection, plus everything a
    streamed run needs to reproduce it: the start carriers, the domain clock,
    the geography inventory, and a :class:`restart_stream.DomainSetup`
    captured from the DOMAIN state (which is what
    :func:`restart_stream.domain_setup_from_stream` is later checked against).

    The warmup step is not cosmetic -- Kain-Fritsch allocates
    ``cumulus/w0avg`` on its first call, so a state that has never stepped has
    a SHORTER manifest than the store must hold.
    """
    import cupy as cp

    key = (rung, nx, ny, nz, nsteps, seed)
    if key in _REF_CACHE:
        return _REF_CACHE[key]
    cfg = gate.geography_cfg(rung, nx, ny, nz)
    geo = harness.make_geography(cfg)
    state, _drv = harness.make_physics_state(cfg, seed, geography=geo)
    harness.run_steps(state, cfg, WARMUP)
    start = {k: _as_numpy(v).copy()
             for k, v in physinv.carrier_inventory(state).items()}
    start_scalars = physinv.carrier_scalars(state)
    geo_start = {k: _as_numpy(v).copy()
                 for k, v in driver.geography_inventory(state).items()}
    setup = restart_stream.capture_domain_setup(state)
    fingerprints = _fingerprints(state, cfg)
    harness.run_steps(state, cfg, nsteps)
    ref_fields = {k: _as_numpy(v).copy()
                  for k, v in physinv.carrier_inventory(state).items()}
    ref_scalars = physinv.carrier_scalars(state)
    del state, _drv
    cp.get_default_memory_pool().free_all_blocks()
    _REF_CACHE[key] = (cfg, start, start_scalars, geo_start, ref_fields,
                       ref_scalars, setup, fingerprints)
    return _REF_CACHE[key]


def _fingerprints(state, cfg) -> tuple[str, str]:
    from gpuwm.io import restart

    return (restart.setup_fingerprint(state),
            restart._json_sha256(restart.physics_setup_identity(state, cfg)))


def _pinned(arrays) -> dict:
    return {k: gather.pinned_copy(np.ascontiguousarray(v))
            for k, v in arrays.items()}


def _poisoned_like(arrays) -> dict:
    """A pinned store full of NaN (integers get their sentinel maximum).

    Anything the restore fails to write is then unmistakable rather than
    plausible.  Integer carriers cannot hold NaN, so they get ``iinfo.max``,
    which is equally not a physical value.
    """
    out = {}
    for key, value in arrays.items():
        blank = np.empty_like(np.ascontiguousarray(value))
        if blank.dtype.kind == "f":
            blank[...] = np.nan
        else:
            blank[...] = np.iinfo(blank.dtype).max
        out[key] = gather.pinned_copy(blank)
    return out


def _template(cfg, tile_nx, tile_ny, halo, *, rebuild=False):
    """One tile buffer, built exactly as ``run_tiled`` builds its own.

    The restart header needs a ``PhysicsDriver`` whose scheme objects it can
    interrogate.  A streamed run has nothing else -- there is no domain state
    -- so the header is built from a TILE-sized buffer with the domain's
    geography temporarily bound onto it (``restart_stream.
    domain_header_view``).  This costs one extra buffer beside the
    ``nbuffers`` the run already holds; at 96x80 with 3x3 tiling that is
    ~1/9 of a domain, and it is the only device allocation the restart path
    makes.

    ``rebuild=True`` builds it on the PER-TILE geography rebuild instead of
    the neutral poison, which is the "wrong setup" control: the fingerprint
    it produces belongs to a domain centred on the tile.
    """
    specs = tspec.plan_tiles(cfg.nx, cfg.ny, tile_nx, tile_ny, halo, True)
    tile_cfg = harness.tile_config(cfg, specs[0].cnx, specs[0].cny)
    build = harness.make_geography if rebuild else harness.neutral_geography
    return driver.make_physics_tile_state(
        tile_cfg, builder=harness.geography_builder(build))


def _stream(store, cfg, geo_store, scalars, tile_nx, tile_ny, halo, nsteps,
            *, nbuffers=2, write_mode="ring") -> dict:
    """``nsteps`` of streamed integration over ``store``; returns the report."""
    import cupy as cp

    kwargs = driver.geography_run_kwargs(cfg, None, geography=geo_store,
                                         geography_fn=harness.neutral_geography)
    kwargs["scalars"] = scalars
    report: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        driver.run_tiled(store, cfg, tile_nx, tile_ny, halo=halo,
                         nsteps=nsteps, nbuffers=nbuffers,
                         write_mode=write_mode, report=report, **kwargs)
    cp.cuda.runtime.deviceSynchronize()
    return report


# --------------------------------------------------------------------------
# the inventory reconciliation, as a gate row
# --------------------------------------------------------------------------

def precondition_inventories(rung="full+MYNN+Noah-MP", nx=40, ny=32,
                             nz=NZ) -> list[str]:
    """Measure the three inventories and assert what the module claims.

    Fails if ``restart``'s enforced manifest and the streamed carrier
    inventory ever stop being the same set of the same objects -- which is
    the only reason a streamed restart is possible at all.
    """
    import cupy as cp

    cfg = gate.geography_cfg(rung, nx, ny, nz)
    geo = harness.make_geography(cfg)
    state, _drv = harness.make_physics_state(cfg, SEED, geography=geo)
    harness.run_steps(state, cfg, WARMUP)
    rec = restart_stream.reconcile_inventories(state, cfg)
    del state, _drv
    cp.get_default_memory_pool().free_all_blocks()

    if rec["only_enforced"] or rec["only_streamed"]:
        raise AssertionError(
            "restart's enforced manifest and the streamed carrier inventory "
            f"have diverged: only in restart {rec['only_enforced']}, only "
            f"streamed {rec['only_streamed']}")
    if not rec["same_objects"]:
        raise AssertionError(
            "the streamed inventory is no longer restart's own manifest but "
            "a copy of it, which can drift; see physics_inventory."
            "carrier_manifest")
    if rec["contract_unlisted"]:
        raise AssertionError(
            f"restart serialises state/{rec['contract_unlisted']} which "
            "STATE_SERIALIZED_ATTRS does not list")
    n_state = len(rec["contract_allocated"])
    lines = [
        f"restart manifest == streamed carrier inventory: "
        f"{len(rec['enforced'])} members, SAME OBJECTS "
        f"({', '.join(f'{k} {v}' for k, v in rec['prefixes'].items())})",
        f"STATE_SERIALIZED_ATTRS is a DIFFERENT list: {len(rec['contract'])} "
        f"names, {n_state} allocated here = {n_state}/"
        f"{len(rec['enforced'])} members and "
        f"{100 * rec['state_fraction']:.1f}% of the carrier bytes; the other "
        f"{len(rec['enforced']) - n_state} live in state._scratch, "
        "state.physics and on the scheme objects",
        f"contract names with no member at this rung (absent or None): "
        f"{rec['contract_absent']}",
    ]
    return lines


def precondition_setup(rung="full+MYNN+Noah-MP", tile=40, nsteps=1) -> str:
    """A DomainSetup assembled from STREAM DATA fingerprints as the domain.

    :func:`restart_stream.domain_setup_from_stream` takes everything
    horizontal from the geography store a tiled run already gathers from and
    everything vertical from a tile buffer.  If that assembly were even one
    array off, the restart it writes would be refused by gpuwm's own reader
    -- or, worse, accepted by a reader that had been handed the same wrong
    setup.  So it is checked against a MONOLITHIC state's fingerprints.
    """
    cfg, _start, _sc, geo_start, _ref, _rsc, setup, fps = reference(
        rung, nsteps)
    halo = harness.halo_radius(cfg)
    template = _template(cfg, tile, tile, halo)
    geo_store = _pinned(geo_start)
    rebuilt = restart_stream.domain_setup_from_stream(geo_store, template)

    from gpuwm.io import restart as _restart

    with restart_stream.domain_header_view(rebuilt, template) as view:
        got = (_restart.setup_fingerprint(view),
               _restart._json_sha256(
                   _restart.physics_setup_identity(view, cfg)))
    if got != fps:
        raise AssertionError(
            f"stream-assembled setup fingerprints {got[0][:16]}/{got[1][:16]} "
            f"!= monolithic {fps[0][:16]}/{fps[1][:16]}")
    # and the buffer must be handed back exactly as it was lent
    after = driver.geography_inventory(template)
    for key, value in after.items():
        if key.startswith("setup/"):
            continue
        if value is rebuilt.scheme_geography.get(key):
            raise AssertionError(
                f"domain_header_view left the DOMAIN's {key} bound to the "
                "tile buffer; a step would then read domain-shaped geography")
    return (f"setup fingerprint {got[0][:16]} and physics setup "
            f"{got[1][:16]} rebuilt from {rebuilt.nbytes / 1e6:.2f} MB of "
            f"geography + tile verticals, identical to the monolithic "
            f"domain's; header view restored the buffer")


def precondition_format(rung="full+MYNN+Noah-MP", tile=40, nsteps=1) -> str:
    """gpuwm's own reader must accept a file this module wrote, and vice versa.

    The strongest available statement that this is a gpuwm restart and not a
    lookalike: ``restart.restore_restart`` -- with every refusal the resident
    path applies, including the ENFORCED driver payload validation this
    module does not reimplement -- is pointed at a streamed file and must
    reproduce the source state bit-for-bit.  Then the reverse: a file written
    by ``restart.write_restart`` is read back into a pinned store.
    """
    import cupy as cp
    from gpuwm.io import restart as _restart

    cfg, start, start_scalars, geo_start, _ref, _rsc, setup, _fps = reference(
        rung, nsteps)
    halo = harness.halo_radius(cfg)
    template = _template(cfg, tile, tile, halo)
    store = _pinned(start)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    path = SCRATCH / "format_check.npz"

    # COUNT the device->host copies the streamed writer makes, by watching
    # restart._host -- the bare ``.get()`` the resident writer runs on all 229
    # carriers.  The claim is that this path runs it on NONE of them; whatever
    # it does run it on is header setup that a scheme happens to keep in VRAM
    # (RRTMGP's ozone profile), is 1-D, and does not scale with the domain.
    # Measured rather than asserted, because "no device state" is the whole
    # reason this function exists.
    import cupy as _cp

    original_host = _restart._host
    pulled: list[tuple[int, tuple]] = []

    def counting_host(value):
        if isinstance(value, _cp.ndarray):
            pulled.append((int(value.nbytes), tuple(int(s)
                                                    for s in value.shape)))
        return original_host(value)

    try:
        _restart._host = counting_host
        info = restart_stream.write_streamed_restart(
            path, store, cfg, scalars=dict(start_scalars), setup=setup,
            template_state=template)
    finally:
        _restart._host = original_host
    carrier_shapes = {tuple(int(s) for s in np.asarray(v).shape)
                      for v in store.values()}
    big = [rec for rec in pulled if rec[1] in carrier_shapes]
    if big:
        raise AssertionError(
            f"the streamed writer pulled {len(big)} DOMAIN-shaped arrays off "
            f"the device: {big[:4]}")
    d2h_bytes = sum(rec[0] for rec in pulled)

    # 1. gpuwm's resident reader, into a freshly prepared DOMAIN state.
    geo = harness.make_geography(cfg)
    fresh, _drv = harness.make_physics_state(cfg, SEED + 1, geography=geo)
    harness.run_steps(fresh, cfg, WARMUP)
    _restart.restore_restart(path, fresh, cfg)
    got = physinv.field_digests(physinv.carrier_inventory(fresh))
    want = physinv.field_digests(store)
    bad = sorted(k for k in want if want[k] != got.get(k))
    if bad:
        raise AssertionError(
            f"gpuwm restore_restart read a streamed file and got {len(bad)} "
            f"carriers wrong, first {bad[:5]}")
    if float(fresh.elapsed_seconds) != float(start_scalars["elapsed_seconds"]):
        raise AssertionError("restored clock differs")

    # 2. the other direction: gpuwm writes, the streamed reader reads.
    mono_path = SCRATCH / "format_check_resident.npz"
    _restart.write_restart(mono_path, fresh, cfg)
    blank = _poisoned_like(start)
    scalars = dict(start_scalars)
    restart_stream.read_streamed_restart(
        mono_path, blank, cfg, setup=setup, template_state=template,
        scalars=scalars)
    got2 = physinv.field_digests(blank)
    bad2 = sorted(k for k in want if want[k] != got2.get(k))
    if bad2:
        raise AssertionError(
            f"the streamed reader read a resident file and got {len(bad2)} "
            f"carriers wrong, first {bad2[:5]}")
    if scalars != start_scalars:
        raise AssertionError(f"scalars {scalars} != {start_scalars}")

    del fresh, _drv, store, blank
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    for p in (path, mono_path):
        p.unlink(missing_ok=True)
    return (f"{info.members} members, {info.bytes / 1e6:.1f} MB: gpuwm "
            f"restore_restart accepts a streamed file bit-exactly, and the "
            f"streamed reader accepts a resident file bit-exactly; the "
            f"streamed write pulled {len(pulled)} arrays / "
            f"{d2h_bytes / 1e3:.1f} kB off the device, NONE of them a "
            f"carrier (the resident writer pulls all {info.members} = "
            f"{info.bytes / 1e6:.1f} MB)")


# --------------------------------------------------------------------------
# the round trip
# --------------------------------------------------------------------------

def round_trip_case(rung, tile_nx, tile_ny, nsteps, *, halo=None, nx=NX,
                    ny=NY, nz=NZ, nbuffers=2, write_mode="ring",
                    drop=(), allow_missing=False, preset="start",
                    wrong_setup=False, restore_scalars=True,
                    seed=SEED) -> dict:
    """Stream N, checkpoint, reload into a FRESH store, stream N, compare.

    ``nsteps`` is N; the comparison is at 2N.  Returns the three digests and
    everything that was measured on the way, including how many times
    radiation and cumulus fired in each half -- because a restart comparison
    over a window where the expensive, cadence-driven physics never runs is
    not a comparison of anything (see PHYSICS CADENCE below).
    """
    import cupy as cp

    total = 2 * int(nsteps)
    cfg, start, start_scalars, geo_start, ref_fields, ref_scalars, \
        _mono_setup, _fps = reference(rung, total, nx=nx, ny=ny, nz=nz,
                                      seed=seed)
    if halo is None:
        halo = harness.halo_radius(cfg)
    ref_digest = digest_of(ref_fields)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    # A DETERMINISTIC name.  Python's hash() is salted per process, so a
    # crashed run would leave a file nobody can identify afterwards.
    tag = hashlib.sha256(
        f"{rung}|{tile_nx}x{tile_ny}|{nsteps}|{write_mode}|{preset}|"
        f"{drop}|{allow_missing}|{restore_scalars}".encode()).hexdigest()[:12]
    path = SCRATCH / f"rt_{tag}.npz"

    # --- B: uninterrupted streamed 2N ---------------------------------
    store_b = _pinned(start)
    geo_b = _pinned(geo_start)
    scalars_b = dict(start_scalars)
    t0 = time.perf_counter()
    _stream(store_b, cfg, geo_b, scalars_b, tile_nx, tile_ny, halo, total,
            nbuffers=nbuffers, write_mode=write_mode)
    uninterrupted_seconds = time.perf_counter() - t0
    digest_b = digest_of(store_b)
    fired_b = _fired(start_scalars, scalars_b)
    del store_b, geo_b

    # --- C: streamed N, checkpoint, fresh everything, streamed N ------
    store_c = _pinned(start)
    geo_c = _pinned(geo_start)
    scalars_c = dict(start_scalars)
    _stream(store_c, cfg, geo_c, scalars_c, tile_nx, tile_ny, halo,
            int(nsteps), nbuffers=nbuffers, write_mode=write_mode)
    fired_first = _fired(start_scalars, scalars_c)
    mid_digest = digest_of(store_c)

    template = _template(cfg, tile_nx, tile_ny, halo, rebuild=wrong_setup)
    setup = restart_stream.domain_setup_from_stream(geo_c, template)
    write = restart_stream.write_streamed_restart(
        path, store_c, cfg, scalars=scalars_c, setup=setup,
        template_state=template, drop=drop)
    del store_c, geo_c

    # Everything below this line is a NEW process's worth of memory: a new
    # store, a new geography store, new tile buffers inside run_tiled.  The
    # only thing that crosses is the file.
    store_d = (_poisoned_like(start) if preset == "poison" else _pinned(start))
    geo_d = _pinned(geo_start)
    template_r = _template(cfg, tile_nx, tile_ny, halo, rebuild=False)
    setup_r = restart_stream.domain_setup_from_stream(geo_d, template_r)
    scalars_d = dict(start_scalars)
    refused = None
    try:
        # ``scalars=None`` restores the ARRAYS and leaves the clock and the
        # driver call counters at this run's preparation values.  That is the
        # scalar half of the same silent drop: elapsed_seconds is the
        # argument to every physics cadence test (itimestep = floor(elapsed/dt
        # + 0.5) + 1), so a resumed run whose clock did not come back
        # evaluates radiation, cumulus and the PBL as due at the wrong steps
        # and integrates a different, entirely plausible forecast.
        read = restart_stream.read_streamed_restart(
            path, store_d, cfg, setup=setup_r, template_state=template_r,
            allow_missing=allow_missing,
            scalars=scalars_d if restore_scalars else None)
    except restart_stream.RestartRefused as exc:
        refused, read = str(exc), None
    if refused is None:
        restored_digest = digest_of(store_d)
        # Snapshot the counters the SECOND half actually starts from, not the
        # ones the file carries: under restore_scalars=False those differ,
        # and that difference is the whole point of that control.
        before_second = {k: dict(v) if isinstance(v, dict) else v
                         for k, v in scalars_d.items()}
        _stream(store_d, cfg, geo_d, scalars_d, tile_nx, tile_ny, halo,
                int(nsteps), nbuffers=nbuffers, write_mode=write_mode)
        digest_c = digest_of(store_d)
        fired_second = _fired(before_second, scalars_d)
        # Both digest maps computed ONCE.  Putting field_digests(ref_fields)
        # inside the comprehension's condition re-hashes the whole 105 MB
        # reference for every one of the 229 keys.
        got_digests = physinv.field_digests(store_d)
        want_digests = physinv.field_digests(ref_fields)
        differing = sorted(k for k, v in got_digests.items()
                           if v != want_digests.get(k))
    else:
        restored_digest = digest_c = ""
        fired_second, differing = {}, []

    record = {
        "refused": refused,
        "bitexact": refused is None and digest_c == ref_digest
        and digest_b == ref_digest,
        "roundtrip_matches_uninterrupted": digest_c == digest_b,
        "uninterrupted_matches_monolithic": digest_b == ref_digest,
        "restored_matches_checkpoint": restored_digest == mid_digest,
        "monolithic": ref_digest,
        "uninterrupted": digest_b,
        "roundtrip": digest_c,
        "differing": differing,
        "carriers": len(start),
        "nsteps": int(nsteps),
        "total": total,
        "halo": int(halo),
        "tiles": len(tspec.plan_tiles(nx, ny, tile_nx, tile_ny, halo, True)),
        "clock_ok": (refused is not None
                     or abs(scalars_d["elapsed_seconds"]
                            - ref_scalars["elapsed_seconds"]) < 1e-12),
        "write": write,
        "read": read,
        "fired_uninterrupted": fired_b,
        "fired_first_half": fired_first,
        "fired_second_half": fired_second,
        "uninterrupted_seconds": uninterrupted_seconds,
        "dropped": tuple(drop),
    }
    if differing and refused is None:
        worst, max_abs = differing[0], 0.0
        for key in differing:
            a = np.asarray(_as_numpy(store_d[key]), dtype=np.float64)
            b = np.asarray(ref_fields[key], dtype=np.float64)
            this = float(np.abs(a - b).max()) if a.size else 0.0
            if this > max_abs:
                worst, max_abs = key, this
        record.update(worst_field=worst, max_abs=max_abs)
    del store_d, geo_d, template, template_r
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    path.unlink(missing_ok=True)
    return record


def _fired(before: dict, after: dict) -> dict:
    """How many times each cadence-driven scheme ran between two clocks.

    PHYSICS CADENCE.  ``PhysicsDriver.call_counts`` is a carrier, so it is in
    the restart header and it is restored; the difference across a window is
    therefore exactly how many times radiation, cumulus, the surface layer
    and the PBL fired inside it.  Printed on BOTH sides of every comparison,
    because this project has already published timings and comparisons taken
    over windows where radiation and cumulus fired zero times.
    """
    b = before.get("call_counts", {})
    a = after.get("call_counts", {})
    return {k: int(a.get(k, 0)) - int(b.get(k, 0)) for k in sorted(a)}


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------

def refusal_contract_store(rung="full+MYNN+Noah-MP", tile=40,
                           nsteps=1) -> str:
    """A store built from STATE_SERIALIZED_ATTRS must be REFUSED at write.

    The trap in one row.  ``hoststore.HostDomainStore`` defaults to
    ``attrs=STATE_SERIALIZED_ATTRS``; a restart written from such a store
    would hold 25 of 229 members and would pass every self-consistency check
    a reader can make against a resuming state that has all 229 slots
    allocated by preparation.
    """
    cfg, start, start_scalars, geo_start, _r, _rs, setup, _f = reference(
        rung, nsteps)
    halo = harness.halo_radius(cfg)
    template = _template(cfg, tile, tile, halo)
    partial = _pinned({k: v for k, v in start.items()
                       if k.startswith("state/")})
    SCRATCH.mkdir(parents=True, exist_ok=True)
    path = SCRATCH / "refusal_contract.npz"
    try:
        restart_stream.write_streamed_restart(
            path, partial, cfg, scalars=dict(start_scalars), setup=setup,
            template_state=template)
    except restart_stream.RestartRefused as exc:
        if path.exists():
            path.unlink()
            raise AssertionError("refused but still wrote a file")
        return (f"write REFUSED a {len(partial)}-member "
                f"STATE_SERIALIZED_ATTRS store against a "
                f"{len(start)}-member manifest: {str(exc)[:96]}...")
    path.unlink(missing_ok=True)
    raise AssertionError(
        "a STATE_SERIALIZED_ATTRS-only store was accepted; a restart written "
        "from it would silently drop 204 of 229 carriers")


def refusal_wrong_setup(rung="full+MYNN+Noah-MP", tile=40, nsteps=1) -> str:
    """A restart whose header came from the per-tile geography REBUILD.

    Same config, same carriers, same file format -- and a setup fingerprint
    belonging to a domain centred on the tile rather than on the domain
    (MEASURED elsewhere in this gate at up to 5.83 deg of latitude and 827 km
    of great circle).  The reader must reject it.  This is the row that says
    the fingerprint in the header is doing work.
    """
    cfg, start, start_scalars, geo_start, _r, _rs, _setup, _f = reference(
        rung, nsteps)
    halo = harness.halo_radius(cfg)
    geo_store = _pinned(geo_start)
    bad_template = _template(cfg, tile, tile, halo, rebuild=True)
    # the tile's OWN rebuilt geography, i.e. no gather at all
    bad_setup = restart_stream.capture_domain_setup(bad_template)
    good_template = _template(cfg, tile, tile, halo)
    good_setup = restart_stream.domain_setup_from_stream(geo_store,
                                                         good_template)
    store = _pinned(start)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    path = SCRATCH / "refusal_setup.npz"
    # A store whose arrays are domain-shaped but whose header describes the
    # tile's geography: write it by force, then try to read it back.
    written = restart_stream.write_streamed_restart(
        path, store, cfg, scalars=dict(start_scalars), setup=bad_setup,
        template_state=bad_template)
    try:
        restart_stream.read_streamed_restart(
            path, store, cfg, setup=good_setup, template_state=good_template)
    except restart_stream.RestartRefused as exc:
        path.unlink(missing_ok=True)
        return (f"read REFUSED a {written.members}-member file whose header "
                f"was fingerprinted on the per-tile geography rebuild: "
                f"{str(exc)[:104]}...")
    path.unlink(missing_ok=True)
    raise AssertionError(
        "a restart written against the per-tile geography rebuild was "
        "accepted; the setup fingerprint is not gating anything")


# --------------------------------------------------------------------------
# cost
# --------------------------------------------------------------------------

def write_cost(rung="full+MYNN+Noah-MP", nx=250, ny=200, nz=NZ,
               resident: bool = True, repeats: int = 2,
               probe_nx=40, probe_ny=32, step_samples: int = 8) -> dict:
    """What a restart costs at a realistic domain, three ways.

    The store is built at ``(nz, ny, nx)`` from the SHAPE RULES read off a
    small probe state (``hoststore.manifest_from_arrays``), so the cost can
    be measured at sizes where a resident device state does not exist -- and
    above the card's ceiling that is the only measurement available, which is
    the point.  ``resident=True`` additionally builds the device state and
    times ``gpuwm.io.restart.write_restart`` beside it, pageable and pinned.
    """
    import cupy as cp

    from tilestream import hoststore

    cfg = gate.geography_cfg(rung, probe_nx, probe_ny, nz)
    geo = harness.make_geography(cfg)
    probe, _drv = harness.make_physics_state(cfg, SEED, geography=geo)
    harness.run_steps(probe, cfg, WARMUP)
    manifest = hoststore.manifest_from_arrays(
        physinv.carrier_inventory(probe), nz, probe_ny, probe_nx)
    del probe, _drv
    cp.get_default_memory_pool().free_all_blocks()

    import dataclasses as _dc
    big_cfg = _dc.replace(cfg, nx=int(nx), ny=int(ny), nz=int(nz))
    store = hoststore.HostDomainStore(big_cfg, manifest=manifest)
    for array in store.arrays.values():
        array[...] = 0.0 if array.dtype.kind == "f" else 0

    geo_big = harness.make_geography(big_cfg)
    if not resident:
        raise NotImplementedError(
            "resident=False would need the domain geography store and clock "
            "from a live streamed run; measure_write_cost takes those "
            "directly, so call it rather than this convenience wrapper")
    # The device state is built ONLY so the resident writer has something to
    # be timed on.  It is what bounds the sizes this function can visit --
    # which is the finding, not a limitation of the measurement: above the
    # card's ceiling there IS no resident path to compare against.
    dev, _d2 = harness.make_physics_state(big_cfg, SEED, geography=geo_big)
    harness.run_steps(dev, big_cfg, WARMUP)
    setup = restart_stream.capture_domain_setup(dev)
    # Give the host store the real field values, so the file that gets
    # written is a real checkpoint rather than a compressible block of zeros.
    for key, value in physinv.carrier_inventory(dev).items():
        np.copyto(store.arrays[key], _as_numpy(value))
    scalars = physinv.carrier_scalars(dev)
    template = dev

    SCRATCH.mkdir(parents=True, exist_ok=True)
    out = restart_stream.measure_write_cost(
        store.arrays, big_cfg, scalars=scalars, setup=setup,
        template_state=template, path=SCRATCH / "cost.npz",
        repeats=repeats, device_state=dev)

    # A checkpoint is only expensive relative to something.  Time real steps
    # at the SAME size on the SAME state and report how many times each
    # cadence-driven scheme fired inside that window -- a step count taken
    # where radiation and cumulus never ran is not a step cost, and this
    # project has already published numbers from exactly such a window.
    before = physinv.carrier_scalars(dev)
    t0 = time.perf_counter()
    harness.run_steps(dev, big_cfg, step_samples)
    step_seconds = (time.perf_counter() - t0) / float(step_samples)
    after = physinv.carrier_scalars(dev)
    out["step_seconds"] = step_seconds
    out["step_samples"] = int(step_samples)
    out["step_fired"] = {k: int(after["call_counts"].get(k, 0))
                         - int(before["call_counts"].get(k, 0))
                         for k in sorted(after["call_counts"])}
    out["write_in_steps"] = out["streamed"] / step_seconds

    out.update(nx=int(nx), ny=int(ny), nz=int(nz),
               cells=int(nx) * int(ny) * int(nz),
               store_bytes=store.nbytes,
               bytes_per_cell=store.bytes_per_cell)
    store.free()
    del dev
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return out


# --------------------------------------------------------------------------
# the matrix
# --------------------------------------------------------------------------

#: (label, kwargs, expected-bit-exact).  ``full fast cadence`` is here for
#: one reason: it is the only rung whose radiation fires during the compared
#: steps, so it is the only one where the restart has to carry the radiation
#: tendencies and the cadence counters correctly to reproduce anything.
CASES: list[tuple[str, dict, bool]] = [
    ("full+MYNN+Noah-MP  |  3x3 ragged (40x30), N=4+4",
     dict(rung="full+MYNN+Noah-MP", tile_nx=40, tile_ny=30, nsteps=4), True),
    ("full+MYNN+Noah-MP  |  2x2 corners, N=2+2",
     dict(rung="full+MYNN+Noah-MP", tile_nx=48, tile_ny=40, nsteps=2), True),
    ("full+MYNN+Noah-MP  |  3x3 ragged, N=4+4, shadow",
     dict(rung="full+MYNN+Noah-MP", tile_nx=40, tile_ny=30, nsteps=4,
          write_mode="shadow"), True),
    ("full+MYNN+Noah-MP  |  3x3 ragged, N=4+4, NaN-POISONED fresh store",
     dict(rung="full+MYNN+Noah-MP", tile_nx=40, tile_ny=30, nsteps=4,
          preset="poison"), True),
    ("full fast cadence  |  3x3 ragged, N=4+4 (radiation every step)",
     dict(rung="full fast cadence", tile_nx=40, tile_ny=30, nsteps=4), True),
    ("full fast cadence  |  2x1 seam, N=1+1",
     dict(rung="full fast cadence", tile_nx=48, tile_ny=80, nsteps=1), True),
    ("mp10 Morrison      |  3x3 ragged, N=4+4",
     dict(rung="mp10 Morrison", tile_nx=40, tile_ny=30, nsteps=4), True),
    # mp18 is the rung with the strictest restart contract in the tree:
    # gpuwm refuses a write whose persistent MP18 set is not canonical
    # (_validate_nssl2_live_restart_state) and a FILE whose is not
    # (_validate_nssl2_stored_restart_state).  This module delegates both
    # rather than reimplementing them, so this row is what exercises the
    # delegation.
    ("full+NSSL mp18    |  3x3 ragged, N=2+2",
     dict(rung="full+NSSL mp18", tile_nx=40, tile_ny=30, nsteps=2), True),
    # No PhysicsDriver at all: the header's "driver" block is None, the
    # carrier scalars are the clock alone, and the reader must not invent a
    # driver payload.  9 carriers instead of 229.
    ("dry (control)     |  3x3 ragged, N=4+4",
     dict(rung="dry (control)", tile_nx=40, tile_ny=30, nsteps=4), True),
    # --- negative controls: these MUST NOT be bit-exact ------------------
    ("NEGATIVE one carrier (fields/ust) DROPPED from the restart, reader "
     "told to allow it",
     dict(rung="full+MYNN+Noah-MP", tile_nx=40, tile_ny=30, nsteps=4,
          drop=("fields/ust",), allow_missing=True), False),
    ("NEGATIVE one carrier (state/thp) DROPPED, fast cadence, reader told "
     "to allow it",
     dict(rung="full fast cadence", tile_nx=40, tile_ny=30, nsteps=4,
          drop=("state/thp",), allow_missing=True), False),
    ("NEGATIVE the checkpoint's CLOCK and call counters not restored "
     "(arrays only), fast cadence",
     dict(rung="full fast cadence", tile_nx=40, tile_ny=30, nsteps=4,
          restore_scalars=False), False),
]

#: Rows whose expected outcome is a REFUSAL rather than a digest.
REFUSALS: list[tuple[str, dict]] = [
    ("the same dropped-carrier file, read NORMALLY (allow_missing=False)",
     dict(rung="full+MYNN+Noah-MP", tile_nx=40, tile_ny=30, nsteps=2,
          drop=("fields/ust",), allow_missing=False)),
]


def _fmt_fired(fired: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in fired.items()) or "(none)"


def main(argv=None) -> int:
    import cupy as cp

    argv = list(sys.argv[1:] if argv is None else argv)
    quick = "--quick" in argv
    with_cost = "--no-cost" not in argv
    global SCRATCH
    if "--dir" in argv:
        SCRATCH = Path(argv[argv.index("--dir") + 1])

    free, total = cp.cuda.runtime.memGetInfo()
    print(f"cupy {cp.__version__}  "
          f"{cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}  "
          f"{free / 2**30:.1f} GiB free of {total / 2**30:.1f}")
    print(f"checkpoints -> {SCRATCH}")
    print()
    failures: list[str] = []

    print("=" * 78)
    print("PRECONDITIONS")
    print("=" * 78)
    try:
        for line in precondition_inventories():
            print(f"  PASS  {line}")
    except Exception as exc:                           # noqa: BLE001
        failures.append(f"inventories: {exc}")
        print(f"  FAIL  inventories: {exc}")
        traceback.print_exc()
    for name, fn in (("stream-assembled setup == domain", precondition_setup),
                     ("format is gpuwm's, both directions",
                      precondition_format)):
        try:
            print(f"  PASS  {name}\n        {fn()}")
        except Exception as exc:                       # noqa: BLE001
            failures.append(f"precondition {name}: {exc}")
            print(f"  FAIL  {name}: {exc}")
            traceback.print_exc()
    print()

    print("=" * 78)
    print("ROUND TRIP  (stream N, checkpoint, FRESH store, reload, stream N; "
          "compare at 2N)")
    print("=" * 78)
    # --quick keeps the first four positive rows and ALL THREE negative
    # controls.  A quick mode that drops a control is a quick mode that
    # cannot fail, which is the failure this project keeps having.
    negatives = [row for row in CASES if not row[2]]
    cases = ([row for row in CASES if row[2]][:4] + negatives) if quick \
        else CASES
    for label, kwargs, expect in cases:
        try:
            rec = round_trip_case(**kwargs)
        except Exception as exc:                       # noqa: BLE001
            failures.append(f"{label}: raised {exc!r}")
            print(f"  ERROR {label}: {exc!r}")
            traceback.print_exc()
            continue
        ok = rec["bitexact"] == expect and rec["refused"] is None
        print(f"  {'PASS' if ok else 'FAIL':4s}  round trip "
              f"{'==' if rec['roundtrip'] == rec['monolithic'] else '!='} "
              f"monolithic, uninterrupted "
              f"{'==' if rec['uninterrupted'] == rec['monolithic'] else '!='}"
              f" monolithic, round trip "
              f"{'==' if rec['roundtrip_matches_uninterrupted'] else '!='} "
              f"uninterrupted")
        print(f"        {label}")
        print(f"        {rec['carriers']} carriers, {rec['tiles']} tiles, "
              f"halo {rec['halo']}, {rec['nsteps']}+{rec['nsteps']} steps, "
              f"clock {'ok' if rec['clock_ok'] else 'WRONG'}")
        w = rec["write"]
        print(f"        checkpoint {w.members} members {w.bytes / 1e6:.1f} MB "
              f"in {w.seconds * 1e3:.0f} ms "
              f"({w.gigabytes_per_second:.2f} GB/s, header "
              f"{w.header_seconds * 1e3:.0f} ms, 0 device copies); "
              f"restored store == checkpointed store: "
              f"{rec['restored_matches_checkpoint']}")
        print(f"        physics fired  first half: "
              f"{_fmt_fired(rec['fired_first_half'])}")
        print(f"                       second half: "
              f"{_fmt_fired(rec['fired_second_half'])}")
        print(f"                       uninterrupted 2N: "
              f"{_fmt_fired(rec['fired_uninterrupted'])}")
        if rec["bitexact"]:
            print(f"        sha256 {rec['roundtrip'][:48]}")
        elif rec["differing"]:
            print(f"        {len(rec['differing'])} carriers differ, worst "
                  f"{rec.get('worst_field')} maxabs "
                  f"{rec.get('max_abs', 0.0):.3e}")
            print(f"        first differing: {rec['differing'][:6]}")
        if not ok:
            failures.append(
                f"{label}: bitexact={rec['bitexact']} (expected {expect})"
                f"{'' if rec['refused'] is None else ' REFUSED: ' + rec['refused'][:80]}")
    print()

    print("=" * 78)
    print("REFUSALS  (a checkpoint that cannot be trusted must not be read)")
    print("=" * 78)
    for label, kwargs in REFUSALS:
        try:
            rec = round_trip_case(**kwargs)
        except Exception as exc:                       # noqa: BLE001
            failures.append(f"refusal {label}: raised {exc!r}")
            print(f"  ERROR {label}: {exc!r}")
            traceback.print_exc()
            continue
        ok = rec["refused"] is not None
        print(f"  {'PASS' if ok else 'FAIL':4s}  "
              f"{'REFUSED' if ok else 'ACCEPTED (should have refused)'}")
        print(f"        {label}")
        if ok:
            print(f"        {rec['refused'][:200]}")
        else:
            failures.append(f"refusal {label}: the reader accepted it")
    for name, fn in (("STATE_SERIALIZED_ATTRS-only store",
                      refusal_contract_store),
                     ("header fingerprinted on the per-tile rebuild",
                      refusal_wrong_setup)):
        try:
            print(f"  PASS  {name}\n        {fn()}")
        except Exception as exc:                       # noqa: BLE001
            failures.append(f"refusal {name}: {exc}")
            print(f"  FAIL  {name}: {exc}")
            traceback.print_exc()
    print()

    if with_cost:
        print("=" * 78)
        print("COST  (what a checkpoint costs, and whether restart._host's "
              "PAGEABLE copy is worth fixing)")
        print("=" * 78)
        # The RESIDENT comparison needs a device state, and that is what caps
        # this sweep: at 400x320x49 the resident footprint is ~2.5 GiB fixed
        # plus 656 B/cell = 6.6 GiB, which is as much of the user's own card
        # as this gate is allowed to take.  Larger points come from the
        # streamed writer alone, where there is no device state at all --
        # which is exactly the regime this whole lane exists for.
        sizes = [(250, 200)] if quick else [(250, 200), (400, 320)]
        for nx, ny in sizes:
            try:
                rec = write_cost(nx=nx, ny=ny, repeats=2)
            except Exception as exc:                   # noqa: BLE001
                print(f"  ERROR {nx}x{ny}: {exc!r}")
                traceback.print_exc()
                continue
            print(f"  {nx}x{ny}x{rec['nz']}  {rec['cells'] / 1e6:.2f} M cells"
                  f"  {rec['store_bytes'] / 2**30:.2f} GiB store "
                  f"({rec['bytes_per_cell']:.1f} B/cell), "
                  f"{rec['members']} members -> {rec['filesystem']}")
            print(f"        streamed (pinned host store, 0 device copies) "
                  f"{rec['streamed']:.2f} s  "
                  f"{rec['streamed_gbps']:.2f} GB/s  "
                  f"(header {rec['streamed_header_seconds'] * 1e3:.0f} ms)")
            if "resident" in rec:
                print(f"        resident write_restart (PAGEABLE .get())   "
                      f"{rec['resident']:.2f} s  "
                      f"{rec['resident_gbps']:.2f} GB/s")
                print(f"        resident write_restart (pinned staging)    "
                      f"{rec['resident_pinned']:.2f} s  "
                      f"{rec['resident_pinned_gbps']:.2f} GB/s")
                print(f"        D2H alone: pageable {rec['d2h_pageable']:.2f} "
                      f"s ({rec['d2h_pageable_gbps']:.2f} GB/s), pinned "
                      f"{rec['d2h_pinned']:.2f} s "
                      f"({rec['d2h_pinned_gbps']:.2f} GB/s) = "
                      f"{rec['d2h_speedup']:.1f}x")
                share = (rec["d2h_pageable"] / rec["resident"]) * 100.0
                print(f"        the pageable copy is {share:.0f}% of the "
                      f"resident write; fixing it moves the total by "
                      f"{100 * (1 - rec['resident_pinned'] / rec['resident']):.0f}%")
            print(f"        one step here costs {rec['step_seconds']:.3f} s "
                  f"(mean of {rec['step_samples']}, physics fired "
                  f"{_fmt_fired(rec['step_fired'])}), so a checkpoint is "
                  f"{rec['write_in_steps']:.1f} steps")
        print()

    print("=" * 78)
    if failures:
        print(f"RESTART GATE FAILED -- {len(failures)}")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("RESTART GATE PASSED -- a streamed run can be checkpointed and "
          "resumed bit-exactly, and every control fired.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
