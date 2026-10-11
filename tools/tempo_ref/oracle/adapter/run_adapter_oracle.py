"""Real DomainState/apply referee, strict word comparisons and field receipts.

All 26 TEMPO outputs are observed. State, radii and reflectivity are read
after the real adapter; the eight raw surface diagnostics are observed at
the real runtime boundary without changing its result. No mock physics.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_woof_oracle import (STATE_IN, UPDATED, SURFACE, COLUMN, OUTPUTS,
                             read_reference, undefined_frozen_fraction)


def word_misses(got, want):
    got, want = np.asarray(got), np.asarray(want)
    assert got.shape == want.shape
    want = want.astype(got.dtype)
    uint = np.uint32 if got.dtype.itemsize == 4 else np.uint64
    return got.view(uint) != want.view(uint)


def read_inputs(path):
    sets = []
    with path.open("rb") as f:
        nsets = int(np.fromfile(f, "<i8", 1)[0])
        for _ in range(nsets):
            nc, nz, steps = map(int, np.fromfile(f, "<i8", 3))
            dt = float(np.fromfile(f, "<f8", 1)[0])
            fields = {n: np.fromfile(f, "<f8", nc*nz).reshape(nc,nz).astype(np.float32)
                      for n in STATE_IN}
            phi = np.fromfile(f, "<f8", nc*(nz+1)).reshape(nc,nz+1).astype(np.float32)
            emission = np.fromfile(f, "<f8", nc).astype(np.float32)
            sets.append(dict(ncol=nc, nz=nz, nsteps=steps, dt=dt,
                             fields=fields, phi=phi, emission=emission))
        assert not f.read(1), "trailing input"
    return sets


def run(root, receipt, *, plant_bit=False, native_cloud=False):
    import cupy as cp
    from gpuwm.config import RunConfig
    from gpuwm.core.state import DomainState
    from gpuwm.core.physics import initialize_physics
    from gpuwm.core.refl import consume_refl_10cm
    from gpuwm.core import tempo
    sets = read_inputs(root / "adapter_in.bin")
    read_reference(root / ("adapter_cloud_out.bin" if native_cloud else "adapter_out.bin"), sets)
    os.environ["GPUWM_TEMPO_BLOB"] = str(root / "tempo_blob.bin")
    tempo._RUNTIME.clear()
    per_field = {n: dict(values=0, mismatched=0, first_miss=None) for n in OUTPUTS}
    hashes = {n: [hashlib.sha256(), hashlib.sha256()] for n in OUTPUTS}
    ff_count = 0
    exceptions = []
    first = None
    native = []
    for si, s in enumerate(sets, 1):
        nc, nz = s["ncol"], s["nz"]
        cfg = RunConfig(nx=nc, ny=1, nz=nz, dx=1000., dy=1000., ztop=40000.,
                        dt=s["dt"], run_seconds=s["dt"]*s["nsteps"], moist=True,
                        mp_physics="tempo", tempo_ml_for_bl_nc=native_cloud,
                        tempo_surface_emission_before=native_cloud)
        state = DomainState(cfg)
        state.thb[...] = 300.0
        def volume(a):
            return cp.ascontiguousarray(cp.asarray(a.T[:,None,:]))
        for n, a in s["fields"].items():
            target = {"t": "thp", "qb": "volg"}.get(n,n)
            if n == "dz":
                continue
            if n == "w":
                state.w[:-1] = volume(a)
            else:
                getattr(state,target)[...] = volume(a)
        state.php[...] = volume(s["phi"])
        state.nwfa2d[...] = cp.asarray(s["emission"][None,:])
        initialize_physics(state, cfg)
        if native_cloud:
            i = np.arange(1, nc+1, dtype=np.int32)[None,:]
            k = np.arange(1, nz+1, dtype=np.int32)[:,None]
            state.physics.fields["qc_bl"] = cp.ascontiguousarray(cp.asarray(
                (np.float32(1.e-5) * (1+(i+k)%4).astype(np.float32))[:,None,:]))
            state.physics.fields["cldfra_bl"] = cp.ascontiguousarray(cp.asarray(
                (np.float32(.25)+np.float32(.125)*((i+2*k)%5).astype(np.float32))[:,None,:]))
        rt = tempo._runtime(cfg,nz)
        observed = []
        original = rt.step
        def observe(*args, **kwargs):
            out = original(*args, **kwargs)
            observed.append(out)
            return out
        rt.step = observe
        try:
            for it, ref in enumerate(s["ref"], 1):
                diag = tempo.apply(state,cfg,s["dt"],refl_10cm_due=True)
                out = observed.pop()
                # Plant in the real state after apply, then restore it before
                # the next free-running step. It must be detected in each call.
                if plant_bit:
                    state.thp.view(cp.uint32)[0,0,0] ^= cp.uint32(1)
                got = {n: getattr(state,{"t":"thp","qb":"volg"}.get(n,n))
                       [:,0,:].get().T for n in UPDATED}
                got.update({n: out["surface"][n][0,:].get()[:,None] for n in SURFACE})
                got.update({"refl10cm": state.existing_scratch("refl_10cm")[:,0,:].get().T,
                            "re_cloud": state.effc[:,0,:].get().T,
                            "re_ice": state.effi[:,0,:].get().T,
                            "re_snow": state.effs[:,0,:].get().T})
                consume_refl_10cm(state)
                if plant_bit:
                    state.thp.view(cp.uint32)[0,0,0] ^= cp.uint32(1)
                for n in OUTPUTS:
                    g, w = got[n], ref[n].astype(got[n].dtype)
                    excused = np.zeros(g.shape, dtype=bool)
                    if n == "frozen_fraction":
                        # Reuse the documented predecessor exception exactly.
                        # Main:747-750's only assignment is +0 when all four
                        # precipitation amounts are zero. Any other reference
                        # word at that combination comes from main:386's early
                        # return. Tiny inactive mass need not be literal zero.
                        excused = undefined_frozen_fraction(ref, g)
                        ff_count += int(excused.sum())
                        for c,k in np.argwhere(excused):
                            exceptions.append(dict(set=si,step=it,column=int(c)+1,
                                field=n, reason="tempo-frozen-fraction-early-return",
                                source="NCAR/TEMPO 9adf9ef module_mp_tempo_main.F90:386,747-750"))
                    misses = word_misses(g,w) & ~excused
                    row = per_field[n]
                    row["values"] += g.size
                    row["mismatched"] += int(misses.sum())
                    if misses.any() and row["first_miss"] is None:
                        c,k = map(int,np.argwhere(misses)[0])
                        detail = dict(set=si,step=it,column=c+1,level=k+1,
                                      got=float(g[c,k]),want=float(w[c,k]))
                        row["first_miss"] = detail
                        if first is None:
                            first = dict(field=n,**detail)
                    hashes[n][0].update(np.where(excused,w,g).tobytes())
                    hashes[n][1].update(w.tobytes())
                native.append(dict(set=si, step=it,
                    theta_sha256=hashlib.sha256(state.thp.get().tobytes()).hexdigest(),
                    rain_sha256=hashlib.sha256(diag.rainncv.get().tobytes()).hexdigest()))
        finally:
            rt.step = original
    result = dict(strict=os.environ.get("GPUWM_WRF_EXACT")=="1", plant_bit=plant_bit,
                  native_cloud=native_cloud,
                  columns=sum(s["ncol"] for s in sets), fields=per_field,
                  values=sum(r["values"] for r in per_field.values()),
                  expected_values=sum(s["nsteps"]*s["ncol"]*(18*s["nz"]+8) for s in sets),
                  mismatched=sum(r["mismatched"] for r in per_field.values()),
                  first_miss=first, undefined_frozen_fraction_cells=ff_count,
                  exceptions=exceptions,
                  hashes={n:dict(got=h[0].hexdigest(),want=h[1].hexdigest()) for n,h in hashes.items()},
                  native_steps=native, seed_exception=None)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k in
                      ("strict","columns","values","mismatched","first_miss")}))
    return result


if __name__ == "__main__":
    result = run(Path(sys.argv[1]), Path(sys.argv[2]), plant_bit="--plant-bit" in sys.argv,
                 native_cloud="--native-cloud" in sys.argv)
    raise SystemExit(1 if result["mismatched"] else 0)
