"""Feed real WRF states to compiled WRF v4.6.1 sixth_order_diffusion.

NetCDF decoding is done by the Rust ``rw_netcdf dump --raw``.  This script
only assembles standalone cases from those exact binary exports, runs the two
compiled builds of the unmodified routine (``strict`` and ``stock``, see
diff6_wrf461_build.py), refuses any case where the two builds disagree by a
single word, and stores the reference beside the exact float32 inputs.

States (each a real WRF 4.6.1 state, not a synthetic field):

* ``conus13z``  CONUS 12 km (425 x 300 x 50), Thompson microphysics, one hour
  into a forecast from 2019-11-26 12Z.  Night over the continent, Pacific,
  Atlantic and Gulf ocean, the Rockies, and a snow and ice storm.
* ``conus12z``  the same domain's initial state (real.exe output).
* ``iowa30``    Iowa 3 km (41 x 41 x 51), 30 minutes into a stock WRF 4.6.1
  serial run from a convective HRRR-initialised state on 2024-05-21 21Z,
  Thompson microphysics: updrafts to 28 m/s, graupel to 10 g/kg.

The full-domain cases use the real case's boundary (specified).  Patch
cases cut from the same states exercise every boundary mode (periodic,
specified, nested, four-sided open, open in x only, open in y only), a
nonzero incoming tendency, and edge cases: zero and constant fields,
subnormal checkerboards, tiny hydrometeor tails whose limiter products
underflow, map factors from 0.25 to 2.75, a ridge steep enough to clamp the
slope taper to zero, dx != dy, and a fractional time step.

Halos that the routine must not read (non-periodic axes, and the unused
mass column/row at ide/jde) are filled with NaN, so any read of them would
show up as a non-finite reference word.
"""
from pathlib import Path
import argparse
import hashlib
import json
import struct
import subprocess
from concurrent.futures import ProcessPoolExecutor
import numpy as np

FIELDS_DRY = {"U": "u", "V": "v", "W": "w", "T": "m"}
SCALARS = ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QNICE", "QNRAIN", "QKE")
MAPS = ("MAPFAC_MX", "MAPFAC_MY", "MAPFAC_UX", "MAPFAC_UY", "MAPFAC_VX", "MAPFAC_VY")
OPTS = ((1, 0), (2, 0), (1, 1), (2, 1))
# boundary modes: name -> (open_xs, open_xe, open_ys, open_ye, specified, nested)
MODES = {"periodic": (0, 0, 0, 0, 0, 0), "specified": (0, 0, 0, 0, 1, 0), "nested": (0, 0, 0, 0, 0, 1),
         "open": (1, 1, 1, 1, 0, 0), "open_x": (1, 1, 0, 0, 0, 0), "open_y": (0, 0, 1, 1, 0, 0)}


def read_dump(directory, frame=-1):
    meta = json.loads((directory / "metadata.json").read_text())
    return {v["name"]: np.fromfile(directory / v["filename"], dtype="<f8").reshape(v["shape"])[frame].astype(np.float32)
            for v in meta["variables"]}


def state_from(raw, *, k0=0, nz=None, y0=0, ny=None, x0=0, nx=None):
    """Cut a (possibly full) window; keeps WRF's staggered extents."""
    NZ = raw["T"].shape[0] if nz is None else nz
    NY = raw["T"].shape[1] - y0 if ny is None else ny
    NX = raw["T"].shape[2] - x0 if nx is None else nx
    s = {}
    for name, a in raw.items():
        if a.ndim == 3:
            sz = NZ + (a.shape[0] == raw["T"].shape[0] + 1)
            sy = NY + (a.shape[1] == raw["T"].shape[1] + 1)
            sx = NX + (a.shape[2] == raw["T"].shape[2] + 1)
            s[name] = a[k0:k0 + sz, y0:y0 + sy, x0:x0 + sx].copy()
        elif a.ndim == 2:
            sy = NY + (a.shape[0] == raw["T"].shape[1] + 1)
            sx = NX + (a.shape[1] == raw["T"].shape[2] + 1)
            s[name] = a[y0:y0 + sy, x0:x0 + sx].copy()
        elif a.ndim == 1:
            s[name] = a[k0:k0 + NZ + (a.shape[0] == raw["T"].shape[0] + 1)].copy()
    s["MUT"] = (s["MU"] + s["MUB"]).astype(np.float32)
    s["_dims"] = (NZ, NY, NX)
    return s


def halo(a, nx, ny, periodic_x, periodic_y):
    """Embed a core array in WRF memory (-3:n+3); NaN where never to be read."""
    lead = a.shape[:-2]
    out = np.full(lead + (ny + 7, nx + 7), np.nan, dtype=np.float32)
    cy, cx = a.shape[-2], a.shape[-1]
    if periodic_x and periodic_y:
        ys = np.arange(-3, ny + 4) % ny; xs = np.arange(-3, nx + 4) % nx
        out[...] = a[..., ys, :][..., xs]
    elif periodic_x:
        xs = np.arange(-3, nx + 4) % nx
        out[..., 3:3 + cy, :] = a[..., :, xs]
    elif periodic_y:
        ys = np.arange(-3, ny + 4) % ny
        out[..., :, 3:3 + cx] = a[..., ys, :]
    else:
        out[..., 3:3 + cy, 3:3 + cx] = a
    return out


def levels(a, nz1):
    """Pad the vertical to the driver's kme = nz+1 with NaN (never read)."""
    if a.shape[0] == nz1:
        return a
    pad = np.full((nz1 - a.shape[0],) + a.shape[1:], np.nan, dtype=np.float32)
    return np.concatenate([a, pad])


def fbytes(a):
    if a.ndim == 3:
        a = a.transpose(1, 0, 2)          # (k, j, i) -> Fortran (i, k, j) memory
    return np.ascontiguousarray(a, dtype="<f4").tobytes()


def run_case(job):
    """Build one input stream, run both builds, return the reference."""
    build, work, case, inputs = job
    s = case
    nz, ny, nx = s["dims"]
    px = not (s["mode"][0] or s["mode"][4] or s["mode"][5])
    py = not (s["mode"][2] or s["mode"][4] or s["mode"][5])
    head = struct.pack("<15i", nx, ny, nz, s["opt"], s["slopeopt"], *s["mode"], s["time_step"], s["num"], s["den"], s["scalar_row"])
    head += s["stagger"].encode() + struct.pack("<4f", s["factor"], s["dx"], s["dy"], s["thresh"])
    arrays = [levels(inputs["field"], nz + 1), levels(inputs["tendency"], nz + 1), inputs["mut"],
              levels(inputs["c1"], nz + 1), levels(inputs["c2"], nz + 1), levels(inputs["phb"], nz + 1),
              *(inputs[m] for m in MAPS)]
    blob = head
    for a in arrays:
        blob += fbytes(a if a.ndim == 1 else halo(a, nx, ny, px, py))
    infile = work / f"case-{s['case']:04d}.in"
    infile.write_bytes(blob)
    outs = {}
    for flavour in ("strict", "stock"):
        outfile = work / f"case-{s['case']:04d}.{flavour}.out"
        subprocess.run([str(build / flavour / "diff6_driver"), str(infile), str(outfile)], check=True)
        raw = outfile.read_bytes()
        outfile.unlink()
        outs[flavour] = raw
    infile.unlink()
    if outs["strict"] != outs["stock"]:
        raise SystemExit(f"case {s['case']}: strict and stock WRF builds disagree")
    dt_arg, rdx, rdy = struct.unpack("<3f", outs["strict"][:12])
    ref = np.frombuffer(outs["strict"][12:], dtype="<f4").reshape(ny + 7, nz + 1, nx + 7).transpose(1, 0, 2)
    f = inputs["field"]
    ref = ref[:f.shape[0], 3:3 + f.shape[1], 3:3 + f.shape[2]].copy()
    return s["case"], ref, (dt_arg, rdx, rdy)


def stagger_of(var):
    return FIELDS_DRY.get(var, "m")


def case_inputs(state, var, mode, tendency=None):
    """Exact float32 inputs.  On a periodic axis the redundant staggered
    column/row is set equal to column/row 0 (the definition of periodic), so
    both sides hold identical stored values there."""
    f = state[var].copy()
    full = var == "W"
    maps = {m: state[m].copy() for m in MAPS}
    px = not (mode[0] or mode[4] or mode[5])
    py = not (mode[2] or mode[4] or mode[5])
    if px:
        if var == "U":
            f[..., -1] = f[..., 0]
        for m in ("MAPFAC_UX", "MAPFAC_UY"):
            maps[m][..., -1] = maps[m][..., 0]
    if py:
        if var == "V":
            f[..., -1, :] = f[..., 0, :]
        for m in ("MAPFAC_VX", "MAPFAC_VY"):
            maps[m][..., -1, :] = maps[m][..., 0, :]
    t0 = np.zeros_like(f) if tendency is None else tendency.astype(np.float32)
    for arr, periodic, axis_stag in ((t0, px, var == "U"), (t0, py, var == "V")):
        if periodic and axis_stag:
            if var == "U":
                arr[..., -1] = arr[..., 0]
            else:
                arr[..., -1, :] = arr[..., 0, :]
    return {"field": f, "tendency": t0,
            "mut": state["MUT"], "c1": state["C1F" if full else "C1H"], "c2": state["C2F" if full else "C2H"],
            "phb": state["PHB"], **maps}


def edge_states(raw13, rawio):
    """Patch states (24 x 20 columns) cut from the real states plus edge probes."""
    NX, NY, NZ = 24, 20, 12
    out = []
    land13 = raw13["XLAND"] < 1.5
    theta = raw13["T"]
    scores = {
        "rockies": (raw13, 0, raw13["HGT"]),
        "pacific": (raw13, 0, np.where(raw13["XLONG"] < -125, (~land13) * 1.0, 0.0)),
        "snowband": (raw13, 4, (raw13["QSNOW"] + raw13["QICE"]).sum(0)),
        "gulf": (raw13, 0, np.where((~land13) & (raw13["XLAT"] < 29) & (raw13["XLONG"] > -97), raw13["QVAPOR"][0], 0.0)),
        "plains_night": (raw13, 0, np.where(land13 & (raw13["HGT"] < 800), theta[4] - theta[0], -1e9)),
        "iowa_storm": (rawio, 0, np.abs(rawio["W"]).max(0)),
        "upper_hybrid": (raw13, 30, raw13["HGT"]),
    }
    for label, (raw, k0, score) in scores.items():
        my, mx = raw["T"].shape[1], raw["T"].shape[2]
        score = np.asarray(score, dtype=np.float64)
        # a window centred on the regime's strongest column, kept inside the domain
        from numpy.lib.stride_tricks import sliding_window_view
        sums = sliding_window_view(score, (NY, NX)).sum(axis=(-1, -2))
        y0, x0 = np.unravel_index(int(np.argmax(sums)), sums.shape)
        out.append((label, state_from(raw, k0=k0, nz=NZ, y0=int(y0), ny=NY, x0=int(x0), nx=NX)))
    base = out[0][1]
    zero = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
    for v in ("U", "V", "W", "T", "QVAPOR"):
        zero[v][:] = 0
    out.append(("zero", zero))
    const = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
    for v in ("U", "V", "W", "T", "QVAPOR"):
        const[v][:] = np.float32(7.25)
    out.append(("constant", const))
    sub = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
    for v in ("U", "V", "W", "T", "QVAPOR"):
        kk, jj, ii = np.indices(sub[v].shape)
        sub[v][:] = (((-1.0) ** (ii + jj)) * (1 + (kk % 5)) * 2.0 ** -140).astype(np.float32)
    out.append(("subnormal_checkerboard", sub))
    tiny = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
    rng = np.random.default_rng(461)
    for v in ("U", "V", "W", "T", "QVAPOR"):
        tiny[v][:] = (np.exp(rng.uniform(np.log(1e-24), np.log(1e-16), tiny[v].shape))).astype(np.float32)
    out.append(("tiny_tails", tiny))
    ext = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in out[1][1].items()}
    for m in MAPS:
        ext[m][:] = np.linspace(.25, 2.75, ext[m].shape[1], dtype=np.float32)[None]
    out.append(("map_extremes", ext))
    steep = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
    ridge = np.float32(9.81) * (np.maximum(0, 2600 - 650 * np.abs(np.arange(NX) - 11))[None, :] + 300 * np.arange(NY)[:, None])
    steep["PHB"] = (steep["PHB"] + ridge[None].astype(np.float32)).astype(np.float32)
    out.append(("steep_ridge", steep))
    return out


def plan(dumps):
    raw13 = read_dump(dumps / "conus13z")
    raw12 = read_dump(dumps / "conus12z")
    rawio = read_dump(dumps / "iowa30", frame=-1)
    for raw in (raw13, raw12, rawio):
        for a, b in (("MAPFAC_MX", "MAPFAC_MY"), ("MAPFAC_UX", "MAPFAC_UY"), ("MAPFAC_VX", "MAPFAC_VY")):
            assert np.array_equal(raw[a], raw[b]), "projection is not isotropic"
    groups = []
    full = [("conus13z", state_from(raw13), 72, ["U", "V", "W", "T", *SCALARS], 12000.0),
            ("conus12z", state_from(raw12), 72, ["U", "V", "T", "QVAPOR"], 12000.0),
            ("iowa30", state_from(rawio), 15, ["U", "V", "W", "T", *SCALARS[:-1]], 3000.0)]
    for label, state, ts, variables, dx in full:
        groups.append((label, state, [dict(var=v, opt=o, slopeopt=sl, mode="specified", time_step=ts, num=0, den=1,
                                           dx=dx, dy=dx, factor=0.12, thresh=0.1, accumulate=False)
                                      for v in variables for o, sl in OPTS]))
    for i, (label, state) in enumerate(edge_states(raw13, rawio)):
        rows = []
        variables = ["U", "V", "W", "T", "QVAPOR"] + (["QSNOW", "QICE", "QGRAUP"] if label in ("snowband", "iowa_storm") else [])
        for j, v in enumerate(variables):
            for m, mode in enumerate(MODES):
                o, sl = OPTS[(i + j + m) % 4]
                frac = (i + j + m) % 5 == 0
                dx, dy = ((1333.3334, 1333.3334) if (i + m) % 4 == 1 else (3000.0, 1750.0) if (i + m) % 4 == 2 else (12000.0, 12000.0))
                if label == "steep_ridge":
                    sl = 1
                rows.append(dict(var=v, opt=o, slopeopt=sl, mode=mode, time_step=13 if frac else 20,
                                 num=1 if frac else 0, den=3 if frac else 1, dx=dx, dy=dy,
                                 factor=[0.12, 0.25, 0.06][(i + m) % 3], thresh=0.1,
                                 accumulate=mode == "periodic" or (j + m) % 3 == 0))
        groups.append((label, state, rows))
    return groups


def generate(dumps, build, output, workers, work=None):
    output.mkdir(parents=True, exist_ok=True)
    work = output / "work" if work is None else work
    work.mkdir(parents=True, exist_ok=True)
    records, n = [], 0
    for label, state, rows in plan(dumps):
        nz, ny, nx = state["_dims"]
        gdir = output / label
        gdir.mkdir(exist_ok=True)
        keep = ["MUT", "PHB", "C1H", "C2H", "C1F", "C2F", "XLAND", "HGT", "XLAT", "XLONG", *MAPS]
        keep += [v for v in ("U", "V", "W", "T", *SCALARS) if v in state]
        np.savez(gdir / "state.npz", **{k: state[k] for k in keep if k in state})
        jobs = []
        for r in rows:
            var = r["var"]
            tendency = None
            if r["accumulate"]:
                rng = np.random.default_rng(n)
                scale = np.float32(np.max(np.abs(state["MUT"])) * 1e-6 * (1 + np.max(np.abs(state[var]))))
                tendency = (rng.standard_normal(state[var].shape) * scale).astype(np.float32)
            inputs = case_inputs(state, var, MODES[r["mode"]], tendency)
            case = {"case": n, "group": label, **r, "stagger": stagger_of(var), "dims": (nz, ny, nx),
                    "mode": MODES[r["mode"]], "mode_name": r["mode"], "scalar_row": int(var in SCALARS)}
            jobs.append((build, work, case, inputs))
            n += 1
        with ProcessPoolExecutor(workers) as pool:
            for (case_id, ref, chain), job in zip(pool.map(run_case, jobs), jobs):
                c = job[2]
                np.save(gdir / f"ref-{case_id:04d}.npy", ref)
                if nx * ny < 10000:      # patch: keep its exact per-case inputs
                    np.savez(gdir / f"inputs-{case_id:04d}.npz", **job[3])
                else:                    # full domain: inputs are state.npz, unmodified
                    assert not c["accumulate"] and c["mode_name"] == "specified"
                rec = {k: v for k, v in c.items() if k not in ("dims", "mode")}
                rec.update(dims=list(c["dims"]), wrf_dt_arg=chain[0], wrf_rdx=chain[1], wrf_rdy=chain[2],
                           reference_sha256=hashlib.sha256(ref.tobytes()).hexdigest(),
                           reference_finite=bool(np.isfinite(ref).all()), words=int(ref.size))
                records.append(rec)
        print(f"{label}: cases={len(rows)} words={sum(r['words'] for r in records if r['group'] == label)}", flush=True)
    work.rmdir()
    meta = json.loads((build / "diff6-wrf461-build.json").read_text())
    meta.update(schema="wrf461-diff6-oracle-v1", cases=records,
                sources=(dumps / "source-sha256.txt").read_text().splitlines(),
                strict_equals_stock="every case: the two builds' output streams are byte-identical",
                tools_sha256={p: hashlib.sha256((Path(__file__).parent / p).read_bytes()).hexdigest()
                              for p in ("diff6_wrf461_driver.f90", "diff6_wrf461_build.py", "diff6_wrf461_cases.py")})
    (output / "oracle.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"compiled WRF 4.6.1 cases={len(records)} words={sum(r['words'] for r in records)} "
          f"nonfinite_cases={sum(not r['reference_finite'] for r in records)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dumps", type=Path); p.add_argument("build", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--work", type=Path, help="scratch for case streams (a RAM disk helps)")
    a = p.parse_args()
    generate(a.dumps, a.build, a.output, a.workers, a.work)
