"""THE DA/ENSEMBLE GATE: does the perturbation and the LETKF survive streaming?

Run it::

    python -m tilestream.test_da_stream

Three execution modes, two legs, and a control for each claim.

  RESIDENT   ``gpuwm.da.perturb.apply_perturbations`` on a CuPy DomainState,
             ``gpuwm.da.letkf.analyze`` on a CuPy prior.  If this fails the
             feature is broken independently of anything the streaming lane
             did, which is itself the finding.
  STREAMED   the same domain in a pinned :class:`tilestream.hoststore
             .HostDomainStore`, addressed through
             :class:`tilestream.da_stream.StoreStateView`.
  MULTI-GPU  static row decomposition.  There is no distributed FFT in this
             tree, so the question is answered by measurement rather than by
             running one: :func:`tilestream.da_stream.slab_filter_control`.

WHY THE DOMAIN IS SMALL ON PURPOSE
----------------------------------
192x160x49, 8 members.  This gate answers EXACTNESS, not throughput, and it
quotes no timing at all: a 192-cell compute window measures an idle GPU, not
the code (see ``tilestream/RESULTS.md``).  Nothing here fires a physics
cadence either, because nothing here steps the model -- the perturbation is
an initialisation and the analysis is an increment, so the radiation/cumulus
cadence trap that has produced three false results in this project does not
apply to this lane.  It DOES apply the moment somebody times a streamed
ensemble forecast, and that measurement is not in this file.

THE FOUR CONTROLS, AND WHAT EACH ONE WOULD CATCH
------------------------------------------------
1.  ``tile_filter_control`` -- the same white noise filtered per 48x40 tile
    with the streaming lane's own halo.  MUST FIRE.  If a per-tile FFT
    reproduced the domain FFT then the whole-domain claim would be empty.
2.  ``fft_host = False`` resident-versus-store -- MUST FIRE.  It is the
    control on the fix: with the filter left on whichever backend the state
    happens to live on, the resident and streamed members of one seed differ,
    and a gate that only ran with the fix on could not tell whether the fix
    did anything.
3.  the store view must be a VIEW.  A copy would perturb a throwaway and
    hand the streamed run its untouched background -- a member that believes
    it was perturbed and was not.  Checked by digesting the store's own
    pinned buffers after the call, not the view.
4.  LETKF with an all-False observation mask -- the closed-form inactive
    transform, which at ``prior_inflation = 1`` is BITWISE zero.  A filter
    that quietly did something on a no-observation cycle would show up here
    and nowhere else.
"""

from __future__ import annotations

import json
import math
import sys
import time

import numpy as np

from tilestream import da_stream as ds
from tilestream import harness

NZ = 49
NY = 160
NX = 192
TILE_NY = 40
TILE_NX = 48
MEMBERS = 8
SEED = 20260809

#: 6 km on a 500 m grid = 12 cells, comfortably inside the module's own
#: resolvability window (floor 2 cells, ceiling span/(2 pi) = 12.7 km) and
#: long enough that the tile control fires by 9% rather than by rounding.
LENGTH_SCALE_KM = 6.0
VERTICAL_SCALE_LEVELS = 6.0


def _cfg():
    return harness.make_config(NX, NY, NZ, moist=True, mp_physics=0)


def _perturbation_config(fft_host: bool, compute_dtype: str = "float64"):
    from gpuwm.da.perturb import FieldPerturbation, PerturbationConfig

    def spec(name, amplitude, mode="additive", **kw):
        return FieldPerturbation(
            name=name, amplitude=amplitude, length_scale_km=LENGTH_SCALE_KM,
            vertical_scale_levels=VERTICAL_SCALE_LEVELS, mode=mode, **kw)

    return PerturbationConfig(
        dx_km=0.5, dy_km=0.5,
        fields=(spec("theta", 1.0), spec("u", 1.5), spec("v", 1.5)),
        rim_width=5, rh_cap=None, fft_host=fft_host,
        compute_dtype=compute_dtype)


# --------------------------------------------------------------------------
# Leg 1
# --------------------------------------------------------------------------

def leg1_perturbation(halo: int) -> list[str]:
    """Draw one member three ways and compare byte for byte."""
    import cupy as cp

    from gpuwm.da.perturb import apply_perturbations
    from tilestream import hoststore

    lines: list[str] = []
    cfg = _cfg()

    # ONE device state alive at a time.  Every rented card on this project is
    # shared with another lane, and a gate that holds three domains at once
    # to compare three arms is a gate that dies on a busy card for reasons
    # that have nothing to do with what it is testing -- which is exactly
    # what happened on the first run of this file.
    base = harness.make_state(cfg, seed=SEED)
    base_digests = {n: ds._digest(getattr(base, n))
                    for n in ds.STORE_READABLE
                    if getattr(base, n, None) is not None}
    base_hash = harness.hash_state(base)
    dtypes = f"thp={base.thp.dtype} u={base.u.dtype}"
    lines.append(f"  base state fields: {sorted(base_digests)}")

    #: The pinned store, sized from the persisted contract and filled from
    #: the SAME base state, so any difference downstream is the
    #: perturbation's and not the initial condition's.
    store = hoststore.HostDomainStore(cfg)
    store.fill_from(base)
    store.assert_pinned()
    del base
    cp.get_default_memory_pool().free_all_blocks()

    lines.append(f"  pinned store: {ds.fmt_bytes(store.nbytes)} over "
                 f"{len(store.names)} fields, "
                 f"{ds.fmt_bytes(store.pinned_pool_bytes)} of pool")
    if store.hash() != base_hash:
        lines.append("  FAIL round trip: store.hash() != hash_state(base)")
        return lines
    lines.append("  round trip OK: store.hash() == hash_state(base)")
    lines.append(f"  state dtype: {dtypes} (the draw is float64 and the "
                 f"STATE is not, which is the whole reason the next table "
                 f"needs reading carefully)")

    # The unperturbed background, on the host, once.  Every arm restores
    # from this instead of rebuilding a device state to copy down.
    background = {n: np.array(store.arrays[n], copy=True)
                  for n in store.names}

    results: dict[str, dict] = {}
    arms = (("float64", False), ("float32", False),
            ("float64", True), ("float32", True))
    for compute_dtype, fft_host in arms:
        pcfg = _perturbation_config(fft_host, compute_dtype)
        tag = f"draw={compute_dtype} fft_host={str(fft_host):<5}"

        # --- RESIDENT: a real CuPy DomainState, exactly as ArWen prepares
        # it, perturbed in place.  This is the control the whole exercise
        # rests on: if this arm fails the feature is broken independently of
        # anything the streaming lane did.
        dev = harness.make_state(cfg, seed=SEED)
        prov_dev = apply_perturbations(dev, 4242, pcfg)
        cp.cuda.runtime.deviceSynchronize()
        dig_dev = {n: ds._digest(getattr(dev, n))
                   for n in ("thp", "u", "v")}
        dev_host = {n: np.asarray(getattr(dev, n).get())
                    for n in ("thp", "u", "v")}
        del dev
        cp.get_default_memory_pool().free_all_blocks()

        # --- HOST MIRROR: same numbers, NumPy namespace, no store --------
        mirror = ds.StoreStateView(
            {n: np.array(background[n], copy=True) for n in background})
        prov_mir = apply_perturbations(mirror, 4242, pcfg)
        dig_mir = mirror.field_digests(("thp", "u", "v"))

        # --- STREAMED: the pinned store itself, through the view ---------
        for n in store.names:
            store.arrays[n][...] = background[n]
        view = ds.StoreStateView(store.arrays)
        prov_sto = apply_perturbations(view, 4242, pcfg)
        # Digest the STORE's buffers, not the view's attributes: control 3.
        dig_sto = {n: ds._digest(store.arrays[n]) for n in ("thp", "u", "v")}

        same_noise = (prov_dev["noise_sha256"] == prov_mir["noise_sha256"]
                      == prov_sto["noise_sha256"])
        store_eq_mirror = dig_sto == dig_mir
        dev_eq_store = dig_dev == dig_sto
        untouched = all(dig_sto[n] != base_digests[n] for n in dig_sto)

        # How far apart the two arms are, in the field's units, in bits, and
        # -- the number that matters on a domain bigger than this one -- in
        # EXPECTED differing cells per million.  The state is float32 and
        # the draw is float64, so a float64 backend gap of ~1e-15 usually
        # rounds to the same float32 word: whether resident and streamed
        # agree is then a coin toss weighted by gap/ulp, and a gate this
        # size wins the toss.  Summing gap/ulp over the domain gives the
        # expected number of flips WITHOUT waiting for one, which is the
        # only honest way to extrapolate from 1.5 M cells to 54 M.
        gaps = {}
        flips = 0
        expected = 0.0
        for name in ("thp", "u", "v"):
            a = dev_host[name].astype(np.float64)
            b = np.asarray(store.arrays[name], dtype=np.float64)
            scale = float(np.abs(a).max())
            differ = int(np.count_nonzero(a != b))
            flips += differ
            ulp = np.spacing(np.abs(a).astype(np.float32)).astype(np.float64)
            gapf = np.abs(a - b)
            # Only the sub-ULP part is a coin toss; a gap of a whole ULP or
            # more is a certain difference and is already counted in
            # ``differ``.
            expected += float(np.minimum(gapf / np.maximum(ulp, 1e-300),
                                         1.0).sum())
            gaps[name] = {
                "max_abs": float(np.abs(a - b).max()),
                "max_rel": (float(np.abs(a - b).max() / scale)
                            if scale else 0.0),
                "differing_cells": differ,
                "cells": int(a.size),
            }

        results[tag] = {
            "noise_sha_agrees": same_noise,
            "store_eq_mirror": store_eq_mirror,
            "resident_eq_streamed": dev_eq_store,
            "store_actually_written": untouched,
            "fft_backend": {"resident": prov_dev["fft_backend"],
                            "mirror": prov_mir["fft_backend"],
                            "store": prov_sto["fft_backend"]},
            "resident_vs_streamed": gaps,
            "observed_flips": flips,
            "expected_flips": expected,
            "total_cells": sum(g["cells"] for g in gaps.values()),
        }

    for tag, r in results.items():
        lines.append(f"  [{tag}]")
        lines.append(f"    noise_sha256 identical across all three arms: "
                     f"{r['noise_sha_agrees']}")
        lines.append(f"    fft backends: {r['fft_backend']}")
        lines.append(f"    STREAMED store == host mirror (bitwise): "
                     f"{r['store_eq_mirror']}")
        lines.append(f"    store buffers changed by the call (view not a "
                     f"copy): {r['store_actually_written']}")
        lines.append(f"    RESIDENT == STREAMED (bitwise): "
                     f"{r['resident_eq_streamed']}")
        for name, g in r["resident_vs_streamed"].items():
            lines.append(f"      {name}: max_abs={g['max_abs']:.3e} "
                         f"max_rel={g['max_rel']:.3e} "
                         f"cells differing {g['differing_cells']}/"
                         f"{g['cells']}")
        lines.append(f"    observed differing cells {r['observed_flips']} of "
                     f"{r['total_cells']}")

    off64 = results["draw=float64 fft_host=False"]
    off32 = results["draw=float32 fft_host=False"]
    on64 = results["draw=float64 fft_host=True "]
    on32 = results["draw=float32 fft_host=True "]
    lines.append("  --- verdicts ---")
    lines.append(_verdict("streamed store reproduces the host mirror in "
                          "every arm",
                          all(r["store_eq_mirror"] for r in results.values())))
    lines.append(_verdict("NEGATIVE CONTROL fires: resident != streamed with "
                          "fft_host=False (float32 draw)",
                          not off32["resident_eq_streamed"]))
    lines.append(_verdict("FIX works: resident == streamed with "
                          "fft_host=True, both draw precisions",
                          on64["resident_eq_streamed"]
                          and on32["resident_eq_streamed"]))
    lines.append(_verdict("view writes the pinned buffers, not a copy",
                          all(r["store_actually_written"]
                              for r in results.values())))
    lines.append(f"    NOTE: the float64-draw arm happened to agree here "
                 f"({off64['observed_flips']} cells differ) and that is NOT "
                 f"a guarantee -- see the forecast below.")
    lines.extend(float64_flip_forecast(background))

    # --- control 3: a COPY-backed view leaves the domain unperturbed -----
    # The failure this catches is the quiet one: the member's manifest says
    # it was perturbed, its provenance carries a real noise_sha256, and the
    # domain the streamed run integrates is the untouched background.
    for n in store.names:
        store.arrays[n][...] = background[n]
    copies = {n: np.array(store.arrays[n], copy=True) for n in
              ds.STORE_READABLE if n in store.arrays}
    before = {n: ds._digest(store.arrays[n]) for n in copies}
    apply_perturbations(ds.StoreStateView(copies), 4242,
                        _perturbation_config(True))
    after = {n: ds._digest(store.arrays[n]) for n in copies}
    copies_changed = any(ds._digest(copies[n]) != before[n] for n in copies)
    lines.append("  --- NEGATIVE CONTROL: a copy-backed view ---")
    lines.append(_verdict("the copy WAS perturbed (so the call really ran)",
                          copies_changed))
    lines.append(_verdict("and the pinned store is BYTE-IDENTICAL -- the "
                          "silently-unperturbed member this view must not "
                          "produce", before == after))

    # --- control 1: the tile FFT ----------------------------------------
    lines.append("  --- NEGATIVE CONTROL: the same noise filtered per tile "
                 f"({TILE_NY}x{TILE_NX}, halo {halo}) ---")
    tc = ds.tile_filter_control(
        (NZ, NY, NX), seed=4242, name="theta", dx_km=0.5, dy_km=0.5,
        length_scale_km=LENGTH_SCALE_KM,
        vertical_scale_levels=VERTICAL_SCALE_LEVELS,
        tile_ny=TILE_NY, tile_nx=TILE_NX, halo=halo)
    lines.append(f"    max_rel={tc['max_rel']:.3e}  rms_rel="
                 f"{tc['rms_rel']:.3e}  bitwise_identical="
                 f"{tc['bitwise_identical']}")
    lines.append(f"    seam mean {tc['seam_mean_rel']:.3e} vs interior mean "
                 f"{tc['interior_mean_rel']:.3e} -> "
                 f"{tc['seam_over_interior_mean']:.1f}x concentrated on the "
                 f"tile seams")
    lines.append(_verdict("NEGATIVE CONTROL fires: a per-tile FFT is not the "
                          "domain FFT", not tc["bitwise_identical"]
                          and tc["max_rel"] > 1e-3))
    return lines


def _verdict(label: str, ok: bool) -> str:
    return f"    [{'PASS' if ok else 'FAIL'}] {label}"


def float64_flip_forecast(background) -> list[str]:
    """Why the float64 draw agreeing at THIS size proves nothing.

    A float64 draw filtered by cuFFT and by pocketfft differs by about one
    float64 ULP (``tilestream.fft_backend_probe`` measures 8e-16 relative in
    >90% of cells at every shape tried).  The state it is added to is
    FLOAT32, whose ULP is eight orders of magnitude wider, so the two
    increments almost always round to the same float32 word and the resident
    and streamed members come out byte-identical.  Almost.

    Whether a given cell flips is a coin toss weighted by ``gap / ulp32``,
    so the expected number of flips over a domain is the SUM of that ratio
    -- computable exactly, from the two increments before the cast, without
    waiting for a flip to happen.  This function computes it at the gate's
    size and extrapolates by cell count, which is the only honest way to get
    from 1.5 M cells to the 54 M of an operational domain.

    Read it as: the float64 arm agreeing above is a property of the gate's
    SIZE, not of the code, and ``fft_host`` is what turns it into a property
    of the code.
    """
    import cupy as cp

    from gpuwm.da.perturb import boundary_taper, gaussian_random_field

    lines = ["  --- FORECAST: does the float64 arm stay lucky? ---"]
    thp = np.asarray(background["thp"], dtype=np.float64)
    nz, ny, nx = thp.shape
    common = dict(seed=4242, name="theta", dx_km=0.5, dy_km=0.5,
                  length_scale_km=LENGTH_SCALE_KM,
                  vertical_scale_levels=VERTICAL_SCALE_LEVELS,
                  dtype="float64")
    host_draw, _ = gaussian_random_field((nz, ny, nx), xp=np, **common)
    dev_draw, _ = gaussian_random_field((nz, ny, nx), xp=cp, **common)
    dev_draw = np.asarray(dev_draw.get())
    cp.get_default_memory_pool().free_all_blocks()

    taper = boundary_taper(ny, nx, 5, kind="cosine", xp=np)[None, :, :]
    sum_h = thp + host_draw * 1.0 * taper
    sum_d = thp + dev_draw * 1.0 * taper
    gap = np.abs(sum_h - sum_d)
    ulp = np.spacing(np.abs(sum_h).astype(np.float32)).astype(np.float64)
    flip_p = np.minimum(gap / np.maximum(ulp, 1e-300), 1.0)
    observed = int(np.count_nonzero(
        sum_h.astype(np.float32) != sum_d.astype(np.float32)))
    expected = float(flip_p.sum())
    cells = thp.size
    lines.append(f"    float64 increment gap between backends: max "
                 f"{float(gap.max()):.3e}, mean {float(gap.mean()):.3e}; "
                 f"float32 ULP of the sum: mean {float(ulp.mean()):.3e}")
    lines.append(f"    cells whose float32 result differs: observed "
                 f"{observed}, expected {expected:.3f} of {cells}")
    per_cell = expected / cells
    lines.append(f"    per-cell flip probability {per_cell:.3e} -- one field, "
                 f"one member:")
    for label, dnz, dny, dnx in (("this gate", nz, ny, nx),
                                 ("1200x900x50", 50, 900, 1200),
                                 ("2048x2048x50", 50, 2048, 2048),
                                 ("4000x4000x50", 50, 4000, 4000)):
        n = dnz * dny * dnx
        lines.append(f"      {label:>14}: {per_cell * n:8.2f} cells differ "
                     f"per field per member "
                     f"({per_cell * n * 3 * MEMBERS:8.2f} over 3 fields x "
                     f"{MEMBERS} members)")
    lines.append("    A resident/streamed pair that differs in a handful of "
                 "cells per member is the worst possible failure: it is "
                 "reproducible only statistically, it never trips a digest "
                 "check that was written at gate size, and it grows with the "
                 "domain -- i.e. it appears exactly when streaming is the "
                 "reason you are running at all.")
    return lines


# --------------------------------------------------------------------------
# Leg 2
# --------------------------------------------------------------------------

def leg2_letkf() -> list[str]:
    """One LETKF analysis from a resident prior and from a pinned one."""
    import cupy as cp

    from gpuwm.da import letkf
    from gpuwm.da.perturb import apply_perturbations
    from tilestream import hoststore

    lines: list[str] = []
    cfg = _cfg()
    base = harness.make_state(cfg, seed=SEED)
    pcfg = _perturbation_config(fft_host=True)

    fields = ("thp",)
    stores = []
    peak_pinned = 0
    t0 = time.perf_counter()
    for k in range(MEMBERS):
        member = harness.make_state(cfg, seed=SEED)
        apply_perturbations(member, 1000 + k, pcfg)
        cp.cuda.runtime.deviceSynchronize()
        st = hoststore.HostDomainStore(cfg)
        st.fill_from(member)
        stores.append(st)
        peak_pinned += st.nbytes
        del member
        cp.get_default_memory_pool().free_all_blocks()
    build_s = time.perf_counter() - t0
    lines.append(f"  {MEMBERS} members into pinned stores in {build_s:.1f} s")
    lines.append(f"  PEAK PINNED for the ensemble: "
                 f"{ds.fmt_bytes(peak_pinned)} "
                 f"({ds.fmt_bytes(stores[0].nbytes)} per member x "
                 f"{MEMBERS}), {stores[0].bytes_per_cell:.1f} B/cell over "
                 f"{len(stores[0].names)} persisted fields")

    # --- the priors ------------------------------------------------------
    # STREAMED: assembled member by member out of the pinned buffers.  This
    # is a COPY into pageable host memory on purpose -- analyze() will call
    # xp.asarray(..., float64) on it anyway, and a pinned analysis buffer
    # would double the unswappable footprint for no gain.
    prior_host = {f: np.stack([np.asarray(s.arrays[f], dtype=np.float64)
                               for s in stores]) for f in fields}
    prior_dev = {f: cp.asarray(prior_host[f]) for f in fields}

    ny, nx = NY, NX
    heights = np.linspace(50.0, 15000.0, NZ)[:, None, None] * np.ones(
        (NZ, ny, nx))
    grid = letkf.GridGeometry(dx_m=500.0, dy_m=500.0, heights_m=heights)
    # 1.5 km on a 500 m grid = a 3-cell horizontal stencil half-width.  That
    # number IS the LETKF's halo, and it is the structural fact that
    # separates this leg from the perturbation: Gaspari-Cohn has COMPACT
    # support, so ``_horizontal_stencil`` returns a bounded index box and a
    # rank or a tile that carries ceil(horizontal_m / dx) cells of neighbour
    # sees everything its analysis can read.  The FFT has no such number.
    loc = letkf.Localization(horizontal_m=1500.0, vertical_m=1000.0)
    # chunk_points left to the budget: an explicit value bypasses the
    # free-memory reading, and every card on this project is shared.
    lcfg = letkf.LetkfConfig(localization=loc, analysis_fields=fields,
                             rtps_alpha=0.9, prior_inflation=1.0,
                             memory_budget_mib=256.0)
    dj, di = letkf._horizontal_stencil(loc, grid, nx, ny)
    lines.append(f"  localisation halo: horizontal +/-"
                 f"{int(np.abs(di).max())} cells, "
                 f"{len(di)} pruned offsets; vertical stencil "
                 f"{len(letkf._vertical_stencil(loc, grid))} levels -- a "
                 f"FINITE cone, which is what makes the analysis tileable "
                 f"at all")

    # --- synthetic observations, on purpose sparse and localised ---------
    # Sparse in z and x to keep the analysis cheap, DENSE IN Y on a 4-row
    # pitch, which is the only property that matters for the slab control
    # below: the seams sit at rows 40/80/120, the stencil half-width is 2
    # rows, so an observation two rows outside a slab is exactly what a
    # one-cell-short halo drops.  The first version of this gate had
    # observations 20 rows apart, none of them landed in that band, and the
    # too-narrow-halo control PASSED when it should have failed -- a control
    # that cannot reach the thing it is controlling for is not a control.
    rng = np.random.default_rng(7)
    mask = np.zeros((NZ, ny, nx), dtype=bool)
    mask[15:40:18, 2:159:4, 40:160:60] = True
    nobs = int(mask.sum())
    truth = np.asarray(prior_host[fields[0]]).mean(axis=0)
    values = truth + 0.5 * rng.standard_normal((NZ, ny, nx))
    simulated = np.asarray(prior_host[fields[0]])
    errors = np.full((NZ, ny, nx), 0.5)
    lines.append(f"  observations: {nobs} active gridpoints of "
                 f"{NZ * ny * nx} ({100.0 * nobs / (NZ * ny * nx):.3f}%)")

    obs_host = [letkf.GriddedObs(name="synthetic-theta", values=values,
                                 errors=errors, simulated=simulated,
                                 mask=mask)]
    obs_dev = [letkf.GriddedObs(name="synthetic-theta",
                                values=cp.asarray(values),
                                errors=cp.asarray(errors),
                                simulated=cp.asarray(simulated),
                                mask=cp.asarray(mask))]

    diag_h = letkf.LetkfDiagnostics()
    inc_host = letkf.analyze(prior_host, obs_host, grid, lcfg, diag_h)
    diag_d = letkf.LetkfDiagnostics()
    inc_dev = letkf.analyze(prior_dev, obs_dev, grid, lcfg, diag_d)
    cp.cuda.runtime.deviceSynchronize()

    # --- and the same host prior built WITHOUT the stores ----------------
    # The store leg's own control: if the two host priors disagree the fault
    # is the store, and if they agree the fault (if any) is the namespace.
    prior_plain = {f: np.stack(
        [np.asarray(cp.asnumpy(cp.asarray(prior_dev[f][k])),
                    dtype=np.float64) for k in range(MEMBERS)])
        for f in fields}
    inc_plain = letkf.analyze(prior_plain, obs_host, grid, lcfg,
                              letkf.LetkfDiagnostics())

    for f in fields:
        a = np.asarray(inc_host[f], dtype=np.float64)
        b = np.asarray(cp.asnumpy(inc_dev[f]), dtype=np.float64)
        c = np.asarray(inc_plain[f], dtype=np.float64)
        scale = float(np.abs(a).max())
        lines.append(f"  field {f!r}: increment |max| = {scale:.6e}")
        lines.append(f"    store-assembled host prior == plain host prior "
                     f"(bitwise): {bool(np.array_equal(a, c))}")
        lines.append(f"    RESIDENT (cupy) == STREAMED (numpy) bitwise: "
                     f"{bool(np.array_equal(a, b))}")
        lines.append(f"      max_abs={float(np.abs(a - b).max()):.3e} "
                     f"max_rel="
                     f"{float(np.abs(a - b).max() / scale):.3e}")
        lines.append(f"    nonzero increments: host "
                     f"{int(np.count_nonzero(a))}, device "
                     f"{int(np.count_nonzero(b))} of {a.size}")
        lines.append(f"    prior spread host {diag_h.prior_spread[f]:.6e} "
                     f"device {diag_d.prior_spread[f]:.6e}")
        lines.append(f"    posterior spread host "
                     f"{diag_h.posterior_spread[f]:.6e} device "
                     f"{diag_d.posterior_spread[f]:.6e}")
    lines.append(f"  active gridpoints: host {diag_h.active_points} / "
                 f"{diag_h.total_points}, device {diag_d.active_points}; "
                 f"stencil slots {diag_h.stencil_slots}; eigensolver "
                 f"host {diag_h.eigensolver!r} device "
                 f"{diag_d.eigensolver!r}; chunks host {diag_h.batches} "
                 f"device {diag_d.batches} at {diag_d.chunk_points} points")
    lines.append(_verdict("CONTROL: the solve actually ran (active points "
                          "> 0 on both sides)",
                          diag_h.active_points > 0
                          and diag_d.active_points > 0))
    lines.append(_verdict("CONTROL: both sides selected the same active set",
                          diag_h.active_points == diag_d.active_points))

    # --- control 4: the no-observation cycle -----------------------------
    empty = [letkf.GriddedObs(name="none", values=values, errors=errors,
                              simulated=simulated,
                              mask=np.zeros_like(mask))]
    inc_none = letkf.analyze(prior_host, empty, grid, lcfg,
                             letkf.LetkfDiagnostics())
    all_zero = all(not np.any(np.asarray(inc_none[f])) for f in fields)
    lines.append(_verdict("CONTROL: a no-observation cycle at "
                          "prior_inflation=1 returns bitwise-zero increments",
                          all_zero))
    lines.append(_verdict("CONTROL: the observed cycle is NOT zero (the "
                          "solve actually fired)",
                          bool(np.any(np.asarray(inc_host[fields[0]])))))

    # --- the leg's real question: does the analysis TILE? ----------------
    # Run in one namespace on purpose.  The resident-vs-streamed comparison
    # above conflates two things -- the transport and the linear algebra --
    # and the transport is the only one streaming is responsible for.  Here
    # both arms are NumPy, so a difference can only be geometry.
    hy = int(np.abs(dj).max())
    lines.append("  --- DOES THE ANALYSIS TILE?  4 row slabs, one "
                 "namespace ---")
    for halo in (hy, hy - 1):
        r = ds.letkf_slab_control(prior_host, obs_host, grid, lcfg,
                                  nslabs=4, halo=halo)
        detail = r["fields"][fields[0]]
        lines.append(f"    halo {halo} (stencil half-width {hy}): "
                     f"bitwise_identical={r['bitwise_identical']} "
                     f"max_abs={detail['max_abs']:.3e} "
                     f"max_rel={detail['max_rel']:.3e} "
                     f"cells differing {detail['differing_cells']}/"
                     f"{detail['cells']}")
        if halo == hy:
            lines.append(_verdict("the LETKF analysis is BIT-EXACT in slabs "
                                  "at the stencil half-width -- the operator "
                                  "the FFT is not",
                                  r["bitwise_identical"]))
        else:
            lines.append(_verdict("NEGATIVE CONTROL fires: one cell short of "
                                  "the stencil half-width is NOT bit-exact",
                                  not r["bitwise_identical"]))

    ledger = ds.letkf_ledger(NZ, NY, NX, MEMBERS, len(fields),
                             store_bytes_per_member=stores[0].nbytes)
    lines.append("  --- LEDGER ---")
    lines.append(f"    pinned ensemble store: "
                 f"{ds.fmt_bytes(ledger['pinned_store_bytes'])}")
    lines.append(f"    analyze() peak (pri+xb+increments, float64): "
                 f"{ds.fmt_bytes(ledger['analysis_peak_bytes'])}")
    lines.append(f"    per observation batch: "
                 f"{ds.fmt_bytes(ledger['per_obs_batch_bytes'])}")
    # Two per-cell figures, because they differ by 5.8x and the wrong one
    # makes an ensemble look affordable.  48.3 B/cell is THIS gate's dry
    # persisted set (13 fields); 279.8 B/cell is the FULL PHYSICS carrier
    # manifest, 229 fields, measured by test_gate's own
    # precondition_carrier_hoststore.  An operational ensemble streams the
    # second one.
    dry_per_cell = stores[0].bytes_per_cell
    carrier_per_cell = 279.8
    limit = ds.container_memory_limit()
    cap = limit.get("pinned_ceiling_from_cgroup")
    lines.append(f"    this container: cgroup limit "
                 f"{ds.fmt_bytes(limit['cgroup_bytes'])} "
                 f"({limit.get('cgroup_path')}), /proc/meminfo says "
                 f"{ds.fmt_bytes(limit['meminfo_total'])} "
                 f"(overstates by {limit.get('overstatement', 1.0):.2f}x); "
                 f"pinned ceiling from the cgroup "
                 f"{ds.fmt_bytes(cap)} vs from meminfo "
                 f"{ds.fmt_bytes(limit['pinned_ceiling_from_meminfo'])}")
    for label, (dnx, dny, dnz) in (
            ("1200x900x50 (the HRRR real-case lane's domain)", (1200, 900, 50)),
            ("2048x2048x50", (2048, 2048, 50)),
            ("4000x4000x50", (4000, 4000, 50))):
        cells = dnx * dny * dnz
        dry = dry_per_cell * cells * MEMBERS
        carrier = carrier_per_cell * cells * MEMBERS
        analysis = 3 * cells * MEMBERS * 8 * len(fields)
        frac = (f", {100.0 * carrier / cap:.0f}% of this container's pinned "
                f"ceiling" if cap else "")
        lines.append(f"    {label}: pinned {ds.fmt_bytes(dry)} dry / "
                     f"{ds.fmt_bytes(carrier)} full physics for "
                     f"{MEMBERS} members{frac}; + analyze() "
                     f"{ds.fmt_bytes(analysis)} per analysis field")

    for st in stores:
        st.free()
    return lines


def leg1b_end_to_end(halo: int, nsteps: int = 4) -> list[str]:
    """PERTURB, THEN INTEGRATE.  The claim the other legs only set up.

    Everything above compares perturbed STATES.  A perturbed state that the
    streamed run then integrates differently is still a broken feature, and
    the perturbation is exactly the kind of input that could expose one: it
    writes a smooth, full-amplitude, correlated field over the whole domain
    including the halo bands, where the seeded background was white and the
    tile edges were quiet.  So this arm runs the real thing:

      RESIDENT  make_state -> perturb (fft_host) -> nsteps of dycore.step
      STREAMED  the SAME start, in a pinned HostDomainStore -> perturb the
                STORE through StoreStateView -> nsteps of driver.run_tiled

    and hashes the whole persisted inventory both ways.  The control that
    makes it mean something is the halo: the same comparison at
    ``halo_radius - 3``, which MUST fail, because a gate whose only
    configuration passes has not been shown to be able to fail.
    """
    import cupy as cp

    from tilestream import driver
    from tilestream import hoststore
    from tilestream import spec as tspec

    lines: list[str] = []
    cfg = _cfg()
    pcfg = _perturbation_config(fft_host=True)

    # --- resident: perturb, then step monolithically ---------------------
    state = harness.make_state(cfg, seed=SEED)
    prov = ds.perturb_member(state, 4242, pcfg)
    cp.cuda.runtime.deviceSynchronize()
    start = {n: np.asarray(a.get()).copy()
             for n, a in harness.state_arrays(state).items()}
    start_hash = harness.hash_state(state)
    harness.run_steps(state, cfg, nsteps)
    cp.cuda.runtime.deviceSynchronize()
    ref = {n: np.asarray(a.get()).copy()
           for n, a in harness.state_arrays(state).items()}
    ref_hash = harness.hash_state(state)
    del state
    cp.get_default_memory_pool().free_all_blocks()
    lines.append(f"  resident: perturbed (noise "
                 f"{prov['noise_sha256'][:16]}..., fft_backend "
                 f"{prov['fft_backend']}), start sha {start_hash[:16]}..., "
                 f"after {nsteps} steps {ref_hash[:16]}...")

    # --- streamed: perturb the STORE, then sweep tiles -------------------
    for h in (halo, halo - 3):
        store = hoststore.HostDomainStore(cfg)
        base = harness.make_state(cfg, seed=SEED)
        store.fill_from(base)
        del base
        cp.get_default_memory_pool().free_all_blocks()

        prov_s = ds.perturb_member(ds.StoreStateView(store.arrays), 4242,
                                   pcfg)
        same_start = (store.hash() == start_hash)

        specs = tspec.plan_tiles(NX, NY, TILE_NX, TILE_NY, h, True)
        tspec.validate_plan(specs, NY, NX)
        report: dict = {}
        driver.run_tiled(store, cfg, TILE_NX, TILE_NY, halo=h,
                         nsteps=nsteps, nbuffers=1, write_mode="ring",
                         report=report)
        cp.cuda.runtime.deviceSynchronize()
        final = driver._arrays_of(store)
        tiled_hash = ds._digest_inventory(final)
        agree = (tiled_hash == ds._digest_inventory(ref))
        differing = sum(1 for n in ref
                        if not np.array_equal(np.asarray(final[n]), ref[n]))
        lines.append(f"  streamed halo {h}: start identical to the resident "
                     f"perturbed state: {same_start}; noise "
                     f"{prov_s['noise_sha256'] == prov['noise_sha256']}; "
                     f"after {nsteps} steps {len(specs)} tiles -> "
                     f"{'BIT-EXACT' if agree else 'DIFFERS'} "
                     f"({differing} of {len(ref)} arrays differ)")
        if h == halo:
            lines.append(_verdict("STREAMED integrates the perturbed domain "
                                  "bit-exactly against RESIDENT", agree))
        else:
            lines.append(_verdict(f"NEGATIVE CONTROL fires: halo {h} is NOT "
                                  f"bit-exact", not agree))
        store.free()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    return lines


def _synthetic_ensemble(members: int, fields=("thp",)):
    """An ``R``-member prior built entirely on the host.

    The LETKF does not inspect what its numbers mean, so a seeded background
    plus the module's own perturbation machinery is a real ensemble as far
    as the filter is concerned -- and it is the STREAMED representation by
    construction, because nothing here ever touches a device.  That is what
    lets the slab control run on a machine whose cards are all busy, which
    on this project is most of the time.
    """
    from gpuwm.da.perturb import boundary_taper, gaussian_random_field

    rng = np.random.default_rng(SEED)
    background = rng.standard_normal((NZ, NY, NX)) * 2.0
    taper = boundary_taper(NY, NX, 5, kind="cosine", xp=np)[None, :, :]
    prior = {}
    for f in fields:
        stack = np.empty((members, NZ, NY, NX), dtype=np.float64)
        for k in range(members):
            draw, _ = gaussian_random_field(
                (NZ, NY, NX), seed=1000 + k, name=f, dx_km=0.5, dy_km=0.5,
                length_scale_km=LENGTH_SCALE_KM,
                vertical_scale_levels=VERTICAL_SCALE_LEVELS,
                xp=np, dtype="float64", fft_host=True)
            stack[k] = background + draw * taper
        prior[f] = stack
    return prior


def leg2_slabs() -> list[str]:
    """Does the analysis tile?  One namespace, so only geometry can differ.

    Split out from :func:`leg2_letkf` because it needs no GPU at all, and
    because it asks a different question.  ``leg2_letkf`` compares a device
    analysis with a host one and therefore conflates the TRANSPORT (which
    streaming owns) with the LINEAR ALGEBRA (which it does not).  This
    compares a whole-domain host analysis with a slabbed host analysis:
    same arithmetic, same library, same order within a point, so a
    difference can only be that a slab could not see something.
    """
    from gpuwm.da import letkf

    lines: list[str] = []
    fields = ("thp",)
    prior = _synthetic_ensemble(MEMBERS, fields)
    lines.append(f"  synthetic host ensemble: {MEMBERS} members, "
                 f"{fields} at {NZ}x{NY}x{NX}, "
                 f"{ds.fmt_bytes(prior[fields[0]].nbytes)} per field")

    heights = np.linspace(50.0, 15000.0, NZ)[:, None, None] * np.ones(
        (NZ, NY, NX))
    grid = letkf.GridGeometry(dx_m=500.0, dy_m=500.0, heights_m=heights)
    loc = letkf.Localization(horizontal_m=1500.0, vertical_m=1000.0)
    lcfg = letkf.LetkfConfig(localization=loc, analysis_fields=fields,
                             rtps_alpha=0.9, prior_inflation=1.0,
                             memory_budget_mib=256.0)
    dj, di = letkf._horizontal_stencil(loc, grid, NX, NY)
    hy = int(np.abs(dj).max())
    lines.append(f"  localisation stencil: +/-{hy} rows, +/-"
                 f"{int(np.abs(di).max())} cols, {len(di)} pruned offsets; "
                 f"vertical {len(letkf._vertical_stencil(loc, grid))} levels")
    lines.append(f"  THIS NUMBER IS THE HALO. Gaspari-Cohn has compact "
                 f"support, so the analysis has a finite dependency cone and "
                 f"the FFT does not; that is the entire difference between "
                 f"the two halves of this lane.")

    rng = np.random.default_rng(7)
    mask = np.zeros((NZ, NY, NX), dtype=bool)
    # Dense in y on a 4-row pitch so that an observation lands two rows
    # outside every slab seam -- exactly what a one-cell-short halo drops.
    mask[15:40:18, 2:159:4, 40:160:60] = True
    truth = prior[fields[0]].mean(axis=0)
    values = truth + 0.5 * rng.standard_normal((NZ, NY, NX))
    errors = np.full((NZ, NY, NX), 0.5)
    obs = [letkf.GriddedObs(name="synthetic-theta", values=values,
                            errors=errors, simulated=prior[fields[0]],
                            mask=mask)]
    diag = letkf.LetkfDiagnostics()
    whole = letkf.analyze(prior, obs, grid, lcfg, diag)
    lines.append(f"  observations {int(mask.sum())}; active gridpoints "
                 f"{diag.active_points} of {diag.total_points}; stencil "
                 f"slots {diag.stencil_slots}; eigensolver "
                 f"{diag.eigensolver!r}; {diag.batches} batches at "
                 f"{diag.chunk_points} points")
    lines.append(_verdict("CONTROL: the solve fired (nonzero increments)",
                          bool(np.any(np.asarray(whole[fields[0]])))))

    lines.append("  --- 4 row slabs, stitched, against the monolithic "
                 "analysis ---")
    ok_at, fired_below = None, None
    for halo in (hy + 1, hy, hy - 1, 0):
        r = ds.letkf_slab_control(prior, obs, grid, lcfg, nslabs=4,
                                  halo=halo)
        d = r["fields"][fields[0]]
        lines.append(f"    halo {halo}: bitwise_identical="
                     f"{str(r['bitwise_identical']):>5} "
                     f"max_abs={d['max_abs']:.3e} "
                     f"max_rel={d['max_rel']:.3e} cells differing "
                     f"{d['differing_cells']}/{d['cells']}")
        if halo == hy:
            ok_at = r["bitwise_identical"]
        if halo == hy - 1:
            fired_below = not r["bitwise_identical"]
    lines.append(_verdict(f"the analysis is BIT-EXACT in slabs at halo {hy} "
                          f"= the stencil half-width", bool(ok_at)))
    lines.append(_verdict(f"NEGATIVE CONTROL fires: halo {hy - 1} is NOT "
                          f"bit-exact", bool(fired_below)))
    return lines


# --------------------------------------------------------------------------
# Mode 3
# --------------------------------------------------------------------------

def leg3_multigpu(halo: int) -> list[str]:
    """The decomposition multi-GPU actually uses, against the domain FFT."""
    lines: list[str] = []
    lines.append("  static row decomposition, halo "
                 f"{halo}, same white noise:")
    for nranks in (2, 4):
        r = ds.slab_filter_control(
            (NZ, NY, NX), seed=4242, name="theta", dx_km=0.5, dy_km=0.5,
            length_scale_km=LENGTH_SCALE_KM,
            vertical_scale_levels=VERTICAL_SCALE_LEVELS,
            nranks=nranks, halo=halo)
        lines.append(f"    {nranks} ranks: max_rel={r['max_rel']:.3e} "
                     f"rms_rel={r['rms_rel']:.3e} "
                     f"bitwise_identical={r['bitwise_identical']} "
                     f"seam/interior={r['seam_over_interior_mean']:.0f}x")
        lines.append(_verdict(f"NEGATIVE CONTROL fires at {nranks} ranks",
                              not r["bitwise_identical"]
                              and r["max_rel"] > 1e-3))
    return lines


def halo_sweep(halos=(16, 24, 32, 48, 60, 64, 80)) -> list[str]:
    """How wide a halo would a tiled filter need?  The answer is 'all of it'.

    Widening a halo is the reflex fix for a tiled operator that disagrees
    with its monolithic reference, and on the dycore it is the right one --
    the influence cone is finite.  A Gaussian spectral filter has no finite
    cone, so this sweep exists to show what widening buys: the error falls
    like the filter itself and only reaches float64 rounding once the
    gathered window is the WHOLE DOMAIN, at which point the tile is not a
    tile.  Read the ``window`` column, not just the error.
    """
    lines = [f"  {'halo':>5} {'window (y,x)':>14} {'max_rel':>11} "
             f"{'rms_rel':>11}  covers domain?"]
    for h in halos:
        r = ds.tile_filter_control(
            (NZ, NY, NX), seed=4242, name="theta", dx_km=0.5, dy_km=0.5,
            length_scale_km=LENGTH_SCALE_KM,
            vertical_scale_levels=VERTICAL_SCALE_LEVELS,
            tile_ny=TILE_NY, tile_nx=TILE_NX, halo=h)
        wy, wx = TILE_NY + 2 * h, TILE_NX + 2 * h
        covers = "YES" if (wy >= NY and wx >= NX) else "no"
        lines.append(f"  {h:>5} {f'{wy}x{wx}':>14} {r['max_rel']:>11.3e} "
                     f"{r['rms_rel']:>11.3e}  {covers}")
    lines.append(f"  (domain is {NY}x{NX}; a window that covers it is not a "
                 f"tile, and even then the differently-shaped transform is "
                 f"not bit-exact)")
    return lines


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    cfg = _cfg()
    halo = harness.halo_radius(cfg)
    print("=" * 74)
    print(f"DA / ENSEMBLE vs STREAMING   {NX}x{NY}x{NZ}, {MEMBERS} members, "
          f"halo {halo} (harness.halo_radius)")
    print(f"length_scale {LENGTH_SCALE_KM} km = "
          f"{LENGTH_SCALE_KM / 0.5:.0f} cells, vertical "
          f"{VERTICAL_SCALE_LEVELS} levels")
    print("no timing is quoted: a 192-cell window measures an idle GPU")
    print("=" * 74)

    want = set(argv) or {"leg1", "leg1b", "leg2", "leg3", "slabs", "sweep"}
    if "leg1" in want:
        print("\nLEG 1 -- PERTURBATION (resident vs pinned host store)")
        for line in leg1_perturbation(halo):
            print(line)
    if "leg1b" in want:
        print("\nLEG 1b -- PERTURB THEN INTEGRATE (end to end, both modes)")
        for line in leg1b_end_to_end(halo):
            print(line)
    if "sweep" in want:
        print("\nHALO SWEEP -- what would a tiled filter need?")
        for line in halo_sweep():
            print(line)
    if "leg3" in want:
        print("\nMODE 3 -- MULTI-GPU (static row decomposition)")
        for line in leg3_multigpu(halo):
            print(line)
    if "slabs" in want:
        print("\nLEG 2a -- DOES THE LETKF TILE?  (host only, no GPU)")
        for line in leg2_slabs():
            print(line)
    if "leg2" in want:
        print("\nLEG 2b -- LETKF (resident prior vs prior from pinned "
              "stores)")
        for line in leg2_letkf():
            print(line)
    print("\ndone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
