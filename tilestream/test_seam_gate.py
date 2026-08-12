"""The hardened seam gate: a gate that COULD have caught the ``mup`` bug.

The 154 Mcell imagery shipped with a real seam in surface pressure and a
bit-exactness gate that was green the whole time.  Neither statement was a
lie; they were about different exchange sets, and the green one was also taken
on a horizontally uniform state over three steps.
:doc:`tilestream/SEAM-IN-SURFACE-PRESSURE.md` has the full mechanism.  This
module is the instrument that fixes the gate, and it is built so that the
question "does this gate actually discriminate?" is answered by the gate
itself rather than asserted in a docstring.

THE THREE THINGS THE OLD GATE LACKED, each an independent axis here
-------------------------------------------------------------------

``ic``
    ``uniform``   horizontally uniform, vertically structured -- what the old
                  gate ran.  Every column identical, so a stale halo holds the
                  same bits as a fresh one.  **Such a gate cannot detect an
                  exchange omission in principle**, whatever carriers it
                  covers, and this module proves that rather than repeating it.

    ``perturbed`` a SMOOTH perturbation that is a function of GLOBAL ``(j, i)``
                  and is then SLICED per rank by ``load_from_host``.  Global
                  indexing is not a detail: all ranks have identical local
                  shapes, so a rank-local draw gives every rank the same tile
                  and builds hard seams into the initial condition.  Smooth,
                  not white: white noise puts all its variance at 2*dx*, which
                  is both physically wrong (it drove psfc to ~1250 hPa and
                  killed a run inside RRTMGP) and the one spectrum a seam can
                  hide in.

``steps``
    The old digest ran **three** steps.  A stale halo has to propagate inward
    before it reaches a cell the scatter keeps, so duration is a real axis and
    is swept, not assumed.

``field``
    A seam check on a SMOOTH field.  ``mup`` is the dycore's prognostic column
    dry mass -- the surface pressure -- and ``p[0]`` is the diagnosed pressure
    at the lowest model level.  Reflectivity was clean at 0.63-0.93 through the
    entire seam episode because a seam hides in a noisy field and cannot hide
    in a smooth one.

WHAT IS ON TRIAL
----------------

``exchange``
    ``ndim>=3``   the set the 154 Mcell runner used: three-dimensional
                  carriers only.  In this inventory that drops exactly one
                  array, ``mup`` -- the same "one 12.6 MB plane, 73 -> 74
                  arrays" the fix costs at production size.
    ``namespace`` every 3-D carrier PLUS the whole dynamics ``state/``
                  namespace whatever its rank.  The fix.  Rank conflates
                  column-local physics with horizontally-coupled dynamics
                  prognostics; namespace does not.

``tile_mup``
    The second, stacked defect: ``mup`` was also RANK-TILED -- the initialiser
    regenerated only ``thp`` and ``w`` from global indices, so every rank held
    an identical copy at t=0 *and* never exchanged it.  Reproduced here by
    overwriting each rank's ``mup`` with rank 0's window after the slice.

Run::

    python -m tilestream.test_seam_gate                  # the discrimination matrix
    NX=192 NY=128 STEPS=24 python -m tilestream.test_seam_gate
    GRID=2x2 python -m tilestream.test_seam_gate         # (gy)x(gx)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from tilestream import decomp, driver as _driver, harness, multigpu
from tilestream.test_decomp_gate import (context_verdict,
                                         cuda_contexts)

#: Non-square on purpose: a y/x transposition cannot pass a square domain.
NX = int(os.environ.get("NX", "192"))
NY = int(os.environ.get("NY", "128"))
NZ = int(os.environ.get("NZ", "49"))

#: More than the three the old digest ran.  Swept, so the duration axis is
#: measured rather than asserted.
STEPS = int(os.environ.get("STEPS", "24"))

#: The step counts reported for every cell of the matrix.  ``3`` is in the
#: list because it is what the old gate ran and the comparison is the point.
LADDER = tuple(int(v) for v in os.environ.get(
    "LADDER", "1,3,%d" % STEPS).split(","))

#: A seam ratio above this is a discontinuity.  Same threshold the slice gate
#: uses, so the two instruments are readable against each other.
SEAM_MAX = float(os.environ.get("SEAM_MAX", "3.0"))


def _grid() -> tuple[int, int]:
    raw = os.environ.get("GRID", "1x2")
    gy, gx = (int(v) for v in raw.split("x"))
    return gy, gx


def _devices(nsub: int) -> list[int]:
    """Which physical card each sub-domain lives on.

    Defaults to card 0 for every sub-domain.  That is not a compromise: the
    decomposition's correctness is a property of the geometry and the seam
    copies, and running the ranks on one card removes the peer transport as a
    variable while leaving every index, every pitch and every step identical.
    ``DEVICES=0,1`` spreads them when more than one card is free.
    """
    raw = os.environ.get("DEVICES")
    if not raw:
        return [0] * nsub
    got = [int(v) for v in raw.split(",")]
    if len(got) != nsub:
        raise SystemExit(f"DEVICES lists {len(got)} cards, need {nsub}")
    return got


# --------------------------------------------------------------------------
# initial conditions
# --------------------------------------------------------------------------

def _at_rest(cfg):
    """The same zeroed base state every sub-domain buffer is built from.

    Using ``driver.make_tile_state`` for the monolithic reference as well is
    what keeps the comparison honest: the arrays OUTSIDE the persisted
    inventory -- the acoustic and tendency bundles -- are then at rest on both
    sides, so a digest difference can only come from the carriers the seam is
    responsible for.
    """
    return _driver.make_tile_state(cfg)


def _sounding(nz: int) -> np.ndarray:
    """A vertically structured, horizontally constant theta perturbation."""
    k = np.arange(nz, dtype=np.float64)
    return (2.5 * np.exp(-((k - 0.30 * nz) / (0.22 * nz)) ** 2)
            - 1.1 * np.exp(-((k - 0.72 * nz) / (0.16 * nz)) ** 2))


def _smooth_global(ny: int, nx: int, *, kj: int, ki: int,
                   phase: float) -> np.ndarray:
    """A periodic, low-wavenumber field of GLOBAL ``(j, i)``.

    Periodic in both axes by construction (integer wavenumbers over the full
    extent), so the domain's own wrap is continuous and the only candidate
    discontinuities in the assembled field are the rank boundaries.  That is
    what makes the seam ratio interpretable.
    """
    j = np.arange(ny, dtype=np.float64)[:, None] / ny
    i = np.arange(nx, dtype=np.float64)[None, :] / nx
    tau = 2.0 * np.pi
    return (np.sin(tau * (ki * i + phase))
            * np.cos(tau * (kj * j - 0.37 * phase))
            + 0.45 * np.sin(tau * (2 * ki * i + 3 * kj * j + 0.11)))


def build_ic(cfg, *, kind: str):
    """A monolithic ``DomainState`` carrying the requested initial condition.

    ``uniform``   vertical structure only.  Every column is the same column.
    ``perturbed`` the same sounding plus a smooth global-index perturbation on
                  ``thp``, ``w``, ``u``, ``v`` and -- the field the whole
                  episode is about -- ``mup``.
    """
    import cupy as cp

    from gpuwm.core.diagnostics import update_diagnostics

    state = _at_rest(cfg)
    nz, ny, nx = int(cfg.nz), int(cfg.ny), int(cfg.nx)
    prof = _sounding(nz)

    thp = cp.asnumpy(state.thp).astype(np.float64)
    for k in range(min(nz, thp.shape[0])):
        thp[k] += prof[k]

    if kind == "perturbed":
        # Amplitudes are the seeded harness's own (`harness._SEED_FIELDS`), so
        # this state is no rougher than the one every existing gate row was
        # measured on -- only smoother, and horizontally structured.
        for k in range(thp.shape[0]):
            thp[k] += 0.5 * _smooth_global(ny, nx, kj=2, ki=3,
                                           phase=0.05 * k)
        w = cp.asnumpy(state.w).astype(np.float64)
        for k in range(1, w.shape[0] - 1):
            w[k] += 0.05 * _smooth_global(ny, nx, kj=3, ki=2,
                                          phase=0.07 * k)
        state.w[...] = cp.asarray(w, dtype=state.w.dtype)

        u = cp.asnumpy(state.u).astype(np.float64)
        for k in range(u.shape[0]):
            g = _smooth_global(ny, u.shape[-1] - 1, kj=1, ki=2, phase=0.03 * k)
            u[k, :, :-1] += 0.05 * g
            u[k, :, -1] = u[k, :, 0]          # the staggered alias column
        state.u[...] = cp.asarray(u, dtype=state.u.dtype)

        v = cp.asnumpy(state.v).astype(np.float64)
        for k in range(v.shape[0]):
            g = _smooth_global(v.shape[-2] - 1, nx, kj=2, ki=1, phase=0.04 * k)
            v[k, :-1, :] += 0.05 * g
            v[k, -1, :] = v[k, 0, :]
        state.v[...] = cp.asarray(v, dtype=state.v.dtype)

        # state/mup: the prognostic surface pressure, and the sole 2-D carrier
        # in this namespace.  20.0 is the harness's own amplitude for it.
        mup = cp.asnumpy(state.mup).astype(np.float64)
        mup += 20.0 * _smooth_global(ny, nx, kj=1, ki=1, phase=0.0)
        state.mup[...] = cp.asarray(mup, dtype=state.mup.dtype)
    elif kind != "uniform":
        raise SystemExit(f"unknown ic {kind!r}")

    state.thp[...] = cp.asarray(thp, dtype=state.thp.dtype)
    state.w[0] = 0.0
    state.w[-1] = 0.0
    update_diagnostics(state)                 # consistent p, al, alt
    return state


# --------------------------------------------------------------------------
# the two exchange sets
# --------------------------------------------------------------------------

def horizontal_spread(state) -> dict:
    """Max over the domain of each carrier's horizontal range.

    The instrument that decides whether the word ``uniform`` in the
    matrix is true.  A gate row taken on a state this reports as flat
    cannot detect an exchange omission, and the reader is entitled to
    see the number rather than the adjective.
    """
    import cupy as cp

    out = {}
    for name, arr in harness.state_arrays(state).items():
        a = cp.asnumpy(arr).astype(np.float64)
        if a.ndim == 2:
            out[name] = float(a.max() - a.min())
        else:
            flat = a.reshape(a.shape[0], -1)
            out[name] = float((flat.max(axis=1) - flat.min(axis=1)).max())
    return out


def exchange_set(arrays: dict, rule: str) -> list[str]:
    """The carrier names a given selection rule would exchange.

    ``ndim>=3``   what the 154 Mcell runner shipped.
    ``namespace`` 3-D, plus every carrier in the dynamics ``state/`` namespace
                  whatever its rank.  ``harness.state_arrays`` IS that
                  namespace -- its keys are the ``STATE_SERIALIZED_ATTRS``
                  attribute names, which the physics inventory spells
                  ``state/<name>`` -- so the rule admits the whole dict.
    """
    names = list(arrays)
    if rule == "namespace":
        return names
    if rule == "ndim>=3":
        return [n for n in names if int(arrays[n].ndim) >= 3]
    raise SystemExit(f"unknown exchange rule {rule!r}")


# --------------------------------------------------------------------------
# fields the seam is measured on
# --------------------------------------------------------------------------

def seam_fields(host: dict) -> dict:
    """Smooth 2-D horizontal fields to measure continuity on.

    ``mup``  the prognostic column dry mass, i.e. the surface pressure, and
             the carrier the ``ndim>=3`` rule drops.
    ``p[0]`` the diagnosed pressure at the lowest model level.  It is a 3-D
             carrier, so it is exchanged under BOTH rules -- a seam in it
             cannot be explained by the field itself being skipped, only by
             something stale that it is computed from.  That is the "drags
             p, al, alt, php with it" claim, measured.
    """
    out = {}
    if "mup" in host:
        out["mup (surface pressure prognostic)"] = np.asarray(host["mup"])
    if "p" in host and np.asarray(host["p"]).ndim == 3:
        out["p[0] (lowest-level pressure)"] = np.asarray(host["p"])[0]
    return out


# --------------------------------------------------------------------------
# one cell of the matrix
# --------------------------------------------------------------------------

def run_cell(cfg, *, ic: str, rule: str, ladder, grid, devices,
             tile_mup: bool = False, transport: str = "host",
             step_mode: str = "sequential",
             exchange_mode: str = "blocking") -> dict:
    """One (ic, exchange rule) pair, digested at every step count in ``ladder``.

    The monolithic reference is stepped in the SAME process and the SAME
    ladder, restarted from the identical host inventory at every rung, so the
    two sides differ in nothing but the decomposition.
    """
    import cupy as cp

    nsub = grid[0] * grid[1]
    dev0 = devices[0]
    with cp.cuda.Device(dev0):
        ref = build_ic(cfg, kind=ic)
        start = multigpu.download_state(ref)
        del ref
        cp.get_default_memory_pool().free_all_blocks()

    out = {"ic": ic, "rule": rule, "tile_mup": tile_mup, "rungs": {},
           "monolithic_ladder": {}}

    for nsteps in ladder:
        # -- monolithic truth, from the same t=0 host inventory -----------
        with cp.cuda.Device(dev0):
            ref = build_ic(cfg, kind=ic)
            harness.run_steps(ref, cfg, nsteps)
            truth = multigpu.download_state(ref)
            h_mono = multigpu.hash_host(truth)
            del ref
            cp.get_default_memory_pool().free_all_blocks()

        # -- the decomposition --------------------------------------------
        dom = multigpu.MultiGPUDomain(
            cfg, ngpu=nsub, grid=grid, devices=devices, transport=transport,
            exchange_names=None if rule == "namespace"
            else exchange_set(harness.state_arrays(_at_rest(cfg)), rule))
        dom.load_from_host(start)
        if tile_mup:
            # DEFECT 2, reproduced: every rank holds rank 0's window of mup
            # instead of its own.  The slice is discarded; the tile is copied.
            src = cp.asnumpy(dom.arrays[0]["mup"])
            for g, dev in enumerate(dom.devices):
                with cp.cuda.Device(dev):
                    dom.arrays[g]["mup"][...] = cp.asarray(src)
                    cp.cuda.runtime.deviceSynchronize()
        dom.run(nsteps, step_mode=step_mode, exchange_mode=exchange_mode)
        host = dom.assemble_host()
        h_dec = multigpu.hash_host(host)
        specs = list(dom.specs)
        dom.close()
        cp.get_default_memory_pool().free_all_blocks()

        out["monolithic_ladder"][nsteps] = h_mono
        rung = {"hash_monolithic": h_mono, "hash_decomposed": h_dec,
                "bitexact": h_dec == h_mono, "seams": {}}
        if not rung["bitexact"]:
            diff = multigpu.compare_hosts(truth, host, int(cfg.nx))
            rung["differing"] = sorted(diff)
            rung["diff"] = {k: diff[k] for k in sorted(diff)[:8]}
        for label, field in seam_fields(host).items():
            st = decomp.seam_statistics(field, specs, nx=int(cfg.nx),
                                        ny=int(cfg.ny))
            rung["seams"][label] = st
        out["rungs"][nsteps] = rung
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def _evolved(cell) -> bool:
    """Did the monolithic reference actually integrate?

    'A negative control that cannot fire makes a PASS meaningless.'  A
    horizontally uniform state that is also TEMPORALLY static would
    report green for every configuration and prove nothing at all, so
    the row is disqualified rather than believed.
    """
    lad = cell["monolithic_ladder"]
    if len(lad) < 2:
        return True
    lo, hi = min(lad), max(lad)
    return lad[lo] != lad[hi]


def _verdict(rung) -> str:
    if not rung["bitexact"]:
        return "RED"
    for st in rung["seams"].values():
        for axis in ("x", "y"):
            s = st.get(axis)
            if s and not s["degenerate"] and s["ratio"] >= SEAM_MAX:
                return "RED"
    return "green"


def _print_cell(cell, ladder) -> None:
    tag = f"ic={cell['ic']:<9s} exchange={cell['rule']:<9s}"
    if cell["tile_mup"]:
        tag += "  +rank-tiled mup"
    print(f"\n--- {tag} ---")
    for n in ladder:
        r = cell["rungs"][n]
        print(f"  {n:3d} steps  digest {r['hash_decomposed'][:16]}  "
              f"monolithic {r['hash_monolithic'][:16]}  "
              f"BITEXACT {'PASS' if r['bitexact'] else 'FAIL'}  "
              f"-> {_verdict(r)}")
        if not r["bitexact"]:
            print(f"      {len(r['differing'])} carriers differ: "
                  f"{', '.join(r['differing'])}")
            for k, d in r["diff"].items():
                if "max_abs" in d:
                    print(f"        {k}: max|d|={d['max_abs']:.6g} over "
                          f"{d['bad_cols']} x-columns, span {d['col_span']}")
        for label, st in r["seams"].items():
            for axis in ("x", "y"):
                s = st.get(axis)
                if s is None:
                    continue
                if s["degenerate"]:
                    print(f"      SEAM {label} {axis}: horizontally uniform, "
                          f"carries no seam signal, NOT GATED")
                    continue
                print(f"      SEAM {label} {axis}: across {s['seam_mean']:.6g}"
                      f"  elsewhere {s['bulk_mean']:.6g}  ratio "
                      f"{s['ratio']:.4f}  "
                      f"{'PASS' if s['ratio'] < SEAM_MAX else 'FAIL'}")


def main(argv=None) -> int:
    import cupy as cp

    grid = _grid()
    nsub = grid[0] * grid[1]
    devices = _devices(nsub)
    cfg = harness.make_config(NX, NY, NZ)
    halo = harness.halo_radius(cfg)
    ctx0 = cuda_contexts()

    import tilestream as _ts
    print(f"TREE: {_ts.__file__}")
    print(f"DOMAIN {NX} x {NY} x {NZ} = {NX * NY * NZ / 1e6:.2f} Mcell   "
          f"grid {grid[0]}x{grid[1]} = {nsub} sub-domains on devices "
          f"{devices}   halo={halo}")
    print(f"LADDER {LADDER} steps   seam threshold {SEAM_MAX}")
    print(f"CUDA CONTEXTS ON THE BOX AT START: {ctx0}"
          + ("   <- CONTENDED; a bit-exactness verdict taken here is NOT a "
             "pass" if ctx0 > 0
             else "   (nothing on the card yet, this gate included)"))
    inv = harness.state_arrays(_at_rest(cfg))
    for rule in ("ndim>=3", "namespace"):
        got = exchange_set(inv, rule)
        print(f"  exchange rule {rule:<9s}: {len(got)} of {len(inv)} carriers"
              + (f"  (drops {sorted(set(inv) - set(got))})"
                 if len(got) != len(inv) else "  (drops nothing)"))
    for kind in ("uniform", "perturbed"):
        st = build_ic(cfg, kind=kind)
        spread = horizontal_spread(st)
        del st
        cp.get_default_memory_pool().free_all_blocks()
        worst = max(spread.values())
        print(f"  IC {kind:<9s}: horizontal range per carrier "
              + ", ".join(f"{k}={v:.6g}" for k, v in
                           sorted(spread.items()))
              + f"   -> {'FLAT' if worst == 0.0 else 'STRUCTURED'}")
    cp.get_default_memory_pool().free_all_blocks()

    cells = [
        # The old gate's own conditions, with the shipped exchange set.
        dict(ic="uniform", rule="ndim>=3"),
        dict(ic="uniform", rule="namespace"),
        # The hardened conditions.
        dict(ic="perturbed", rule="ndim>=3"),
        dict(ic="perturbed", rule="namespace"),
        # Defect 2, on top of the fixed exchange set.
        dict(ic="perturbed", rule="namespace", tile_mup=True),
    ]

    results = []
    t0 = time.perf_counter()
    for spec in cells:
        cell = run_cell(cfg, ladder=LADDER, grid=grid, devices=devices, **spec)
        _print_cell(cell, LADDER)
        results.append(cell)

    # ---------------------------------------------------------- the matrix
    print("\n" + "=" * 78)
    print("DISCRIMINATION MATRIX  (green = a gate in this configuration "
          "reports PASS)")
    print("=" * 78)
    head = "  ".join(f"{n:>9d}" for n in LADDER)
    print(f"{'configuration':<44s}{head}")
    for cell in results:
        name = f"ic={cell['ic']}, exchange={cell['rule']}"
        if cell["tile_mup"]:
            name += ", rank-tiled mup"
        row = "  ".join(f"{_verdict(cell['rungs'][n]):>9s}" for n in LADDER)
        mark = "" if _evolved(cell) else "   <- VACUOUS: the reference " \
                                        "never integrated"
        print(f"{name:<44s}{row}{mark}")

    def cell_of(ic, rule, tile=False):
        for c in results:
            if c["ic"] == ic and c["rule"] == rule and c["tile_mup"] == tile:
                return c
        raise KeyError((ic, rule, tile))

    old_gate = cell_of("uniform", "ndim>=3")["rungs"][3]
    new_gate = cell_of("perturbed", "ndim>=3")["rungs"][max(LADDER)]
    fixed = cell_of("perturbed", "namespace")["rungs"][max(LADDER)]
    tiled = cell_of("perturbed", "namespace", True)["rungs"][max(LADDER)]

    print()
    print("THE PROOF THIS GATE DISCRIMINATES, in four cells:")
    print(f"  1. old conditions (uniform IC, 3 steps) + the SHIPPED exchange "
          f"set  -> {_verdict(old_gate)}")
    print(f"  2. hardened conditions (perturbed global-sliced IC, "
          f"{max(LADDER)} steps) + the SAME shipped set -> "
          f"{_verdict(new_gate)}")
    print(f"  3. hardened conditions + the namespace fix                      "
          f"          -> {_verdict(fixed)}")
    print(f"  4. hardened conditions + fix + defect 2 (rank-tiled mup)        "
          f"          -> {_verdict(tiled)}")

    vacuous = [f"ic={c['ic']}/{c['rule']}" for c in results
               if not _evolved(c)]
    print("\nEVOLUTION CHECK (a row that never moved cannot gate "
          "anything): "
          + ("every row integrated" if not vacuous
             else f"VACUOUS ROWS {vacuous}"))
    discriminates = (not vacuous
                     and _verdict(old_gate) == "green"
                     and _verdict(new_gate) == "RED"
                     and _verdict(fixed) == "green"
                     and _verdict(tiled) == "RED")
    ctx1, verdict = context_verdict(ctx0)
    print(f"\nCUDA CONTEXTS AT END: {ctx1} (start {ctx0}).  " + verdict)
    print(f"elapsed {time.perf_counter() - t0:.1f} s")
    print("HARDENED SEAM GATE DISCRIMINATES: "
          + ("YES" if discriminates else "NO"))
    return 0 if discriminates else 1


if __name__ == "__main__":
    sys.exit(main())
