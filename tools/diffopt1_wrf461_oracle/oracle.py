"""WRF v4.6.1 diff_opt=1 column oracle: cases, WRF words, engine words.

  python oracle.py cases   --source DIR --out DIR
  python oracle.py wrf     --library BUILD/oracle.so --cases DIR --prefix wrf
  python oracle.py compare --cases DIR --receipt RECEIPT.json   (GPU, under the mutex)

``cases`` takes real WRF windows (km_opt=4 column-oracle case files: 12 km
CONUS 2019-11-26 13 UTC windows, a 3 km convective window and named edge
transformations) and adds the two inputs diff_opt=1 needs that a
diff_opt=2/km_opt=4 history does not carry: prognostic TKE and WRF's
``t_init``.  ``wrf`` runs the compiled WRF pipeline (pipeline.F90) for every
case and every ARM.  ``compare`` runs the engine's production launchers --
``launch_wrf_smag2d_km`` / ``launch_wrf_tke_km`` and the same
``_compute_wrf_smag_tendencies`` the forecast calls, which routes
diff_opt=1 to ``_compute_coordinate_tendencies`` -- on identical input words
and compares every output word, signed zeros included.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path

import numpy as np

F32 = np.float32
H = 5          # Fortran halo: memory -4..n+5, interior 1..n at python offset 5
BOUNDARY = {"periodic": 0, "open": 1, "specified": 2}
SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg")
NUMBERS = ("ni", "nr")
T0 = F32(300.0)

# Configurations a diff_opt=1 run can select with km_opt 2 or 4.
# isfflx=0 under diff_opt=1 seeds TKE at 1e-6 whatever the surface constants.
# bl_pbl_physics=0 adds WRF's constant-kvdif vertical routines with kvdif=0.
ARMS = {
    "km4": dict(km_opt=4, isotropic=0, isfflx=1, pbl=1),
    "km4_pbl0": dict(km_opt=4, isotropic=0, isfflx=1, pbl=0),
    "km2": dict(km_opt=2, isotropic=0, isfflx=1, pbl=1),
    "km2_isotropic": dict(km_opt=2, isotropic=1, isfflx=1, pbl=1),
    "km2_seed_pbl0": dict(km_opt=2, isotropic=0, isfflx=0, pbl=0),
}
C_S, C_K, MIX_UPPER_BOUND = 0.25, 0.15, 0.1


def _t0_round(x):
    """fl(fl(300 + x) - 300): exact under every arithmetic (Sterbenz), so the
    engine's ``thp + thb - 300`` with ``thb = 300`` returns the same word."""
    x = np.asarray(x, F32)
    return (np.asarray(T0 + x, F32) - T0).astype(F32)


# ---------------------------------------------------------------- cases

def _tke(a, name):
    """Deterministic prognostic TKE with a boundary-layer profile.

    Every case also carries columns at exactly 0, 1e-12, the 1e-6 WRF seed
    and 1e-40 (below the smallest normal float32), so both tke_seed branches
    and the MAX floor are hit.
    """
    nz, ny, nx = a["t2"].shape
    zf = (a["php"].astype(np.float64) + a["phb"]) / 9.81
    zh = 0.5 * (zf[:-1] + zf[1:])
    agl = np.maximum(zh - zf[0][None], 0.0)
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    wm = 0.5 * (np.abs(a["w"][:-1]) + np.abs(a["w"][1:]))
    tke = (0.04 + 2.2 * np.exp(-agl / 1100.0) * (1.0 + 0.6 * np.sin(0.7 * ii + 1.3 * jj))[None]
           + 0.8 * np.minimum(wm, 5.0) + 0.3 * 1e3 * (a["qc"] + a["qi"]))
    tke = tke.astype(F32)
    if name.startswith("zero_flow"):
        return np.zeros_like(tke)
    if name.startswith("signed_zero_flow"):
        return np.full_like(tke, F32(-0.0))
    if name.startswith("near_zero_flow"):
        return (tke * F32(1e-30)).astype(F32)
    tke[:, 1, 3] = 0.0
    tke[:, 2, 5] = F32(1e-12)
    tke[:, 4, 7] = F32(1e-6)
    tke[:, 6, 9] = F32(1e-40)          # below the smallest normal float32
    tke[:, 8, 2] = F32(-0.0)
    return tke


def _t_init(a):
    """WRF t_init (initial theta - 300 K): the case's theta minus a smooth,
    level-dependent evolution, rounded so 300 + t_init is representable."""
    nz, ny, nx = a["t2"].shape
    kk, jj, ii = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    delta = 0.35 * np.sin(0.45 * ii - 0.3 * jj + 0.2 * kk) + 0.05 * kk / nz
    return _t0_round(a["t2"].astype(np.float64) - delta)


def make_cases(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(args.source.glob("case-*.npz")):
        with np.load(path) as data:
            arrays = {k.removeprefix("input__"): data[k] for k in data.files if k.startswith("input__")}
            meta = json.loads(str(data["meta_json"]))
        name = path.stem.removeprefix("case-")
        arrays["t2"] = _t0_round(arrays["t2"])
        arrays["tke"] = _tke(arrays, name)
        arrays["t_init"] = _t_init(arrays)
        meta = {**meta, "source_case": path.name,
                "source_case_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "dt": float(6.0 * meta["dx"] / 1000.0)}
        payload = {"input__" + k: np.ascontiguousarray(v) for k, v in arrays.items()}
        payload["meta_json"] = np.array(json.dumps(meta, sort_keys=True))
        target = out / path.name
        np.savez_compressed(target, **payload)
        rows.append({"case": name, "file": target.name, "regime": meta["regime"],
                     "boundary": meta["boundary"], "columns": meta["nx"] * meta["ny"],
                     "dx": meta["dx"], "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
        print(name, flush=True)
    manifest = {"schema": "diffopt1-wrf461-cases-v1", "cases": rows,
                "columns": sum(r["columns"] for r in rows), "arms": ARMS}
    (out / "cases-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(len(rows), "cases,", manifest["columns"], "columns")


# ---------------------------------------------------------------- WRF side

def _pad(field, nx, ny, nz):
    """Engine (k, j, i) interior into WRF (i, k, j) memory with zero halos."""
    f = np.asarray(field, F32)
    out = np.zeros((nx + 2 * H, nz + 1, ny + 2 * H), F32, order="F")
    kk, jj, ii = f.shape
    out[H:H + ii, :kk, H:H + jj] = f.transpose(2, 0, 1)
    return out


def _pad2(field, nx, ny):
    f = np.asarray(field, F32)
    out = np.zeros((nx + 2 * H, ny + 2 * H), F32, order="F")
    out[H:H + f.shape[1], H:H + f.shape[0]] = f.T
    return out


def _core(a, nx, ny, nz, sx=0, sy=0, sz=0):
    return np.ascontiguousarray(a[H:H + nx + sx, :nz + sz, H:H + ny + sy].transpose(1, 2, 0))


def _col(v, nz):
    out = np.zeros(nz + 1, F32)
    v = np.asarray(v, F32)
    out[:v.size] = v
    return out


OUT3 = ("z", "rdz", "rdzw", "zx", "zy", "div", "d11", "d22", "d33", "d12", "d13", "d23",
        "kmh", "kmv", "khh", "khv", "bn2", "tu", "tv", "tw", "tth", "ttke")
# (name, (sx, sy, sz)) WRF output words kept per case
KEEP = {"d11": (0, 0, 0), "d22": (0, 0, 0), "d12": (1, 1, 0), "kmh": (0, 0, 0), "khh": (0, 0, 0),
        "kmv": (0, 0, 0), "khv": (0, 0, 0), "bn2": (0, 0, 0),
        "tu": (1, 0, 0), "tv": (0, 1, 0), "tw": (0, 0, 1), "tth": (0, 0, 0), "ttke": (0, 0, 0)}


def wrf_pipeline(lib, arrays, meta, arm):
    nx, ny, nz = meta["nx"], meta["ny"], meta["nz"]
    fn = lib.diffopt1_pipeline
    fn.restype = None
    n_moist, n_scalar = 1 + len(SPECIES), 1 + len(NUMBERS)
    ins3 = {k: _pad(arrays[k], nx, ny, nz)
            for k in ("u", "v", "w", "php", "phb", "t2", "t_init", "alt", "p", "pb", "tke")}
    moist = np.zeros((*ins3["t2"].shape, n_moist), F32, order="F")
    for slot, name in enumerate(SPECIES, start=1):
        moist[..., slot] = _pad(arrays[name], nx, ny, nz)
    scalar = np.zeros((*ins3["t2"].shape, n_scalar), F32, order="F")
    for slot, name in enumerate(NUMBERS, start=1):
        scalar[..., slot] = _pad(arrays[name], nx, ny, nz)
    ins2 = {k: _pad2(arrays[k], nx, ny) for k in ("msfu", "msfv", "msft", "mut")}
    cols = {k: _col(arrays[k], nz) for k in ("c1h", "c2h", "c1f", "c2f", "dn", "dnw", "fnm", "fnp", "znw")}
    dx, dy = F32(meta["dx"]), F32(meta["dy"])
    rdx, rdy = F32(1) / dx, F32(1) / dy
    outs = {k: np.zeros_like(ins3["t2"]) for k in OUT3}
    tmoist = np.zeros_like(moist)
    tscalar = np.zeros_like(scalar)
    ptr = lambda a: ctypes.c_void_p(a.ctypes.data)
    cfg = ARMS[arm]
    args = [ctypes.c_int(v) for v in (nx, ny, nz, BOUNDARY[meta["boundary"]], cfg["km_opt"],
                                      cfg["isotropic"], cfg["isfflx"], cfg["pbl"], n_moist, n_scalar)]
    args += [ptr(ins3[k]) for k in ("u", "v", "w", "php", "phb", "t2", "t_init", "alt", "p", "pb")]
    args += [ptr(moist), ptr(scalar), ptr(ins3["tke"])]
    args += [ptr(ins2[k]) for k in ("msfu", "msfv", "msft", "mut")]
    args += [ptr(cols[k]) for k in ("c1h", "c2h", "c1f", "c2f", "dn", "dnw", "fnm", "fnp", "znw")]
    args += [ctypes.c_float(v) for v in (dx, dy, rdx, rdy, meta["cf1"], meta["cf2"], meta["cf3"],
                                         meta["p_top"], C_S, C_K, meta["dt"], MIX_UPPER_BOUND)]
    args += [ptr(outs[k]) for k in OUT3] + [ptr(tmoist), ptr(tscalar)]
    fn(*args)
    result = {k: _core(outs[k], nx, ny, nz, *KEEP[k]) for k in KEEP}
    for slot, name in enumerate(SPECIES, start=1):
        result["t" + name] = _core(tmoist[..., slot], nx, ny, nz)
    for slot, name in enumerate(NUMBERS, start=1):
        result["t" + name] = _core(tscalar[..., slot], nx, ny, nz)
    if cfg["km_opt"] != 2:
        for key in ("kmv", "khv", "bn2", "ttke"):
            result.pop(key)
    return result


def run_wrf(args):
    lib = ctypes.CDLL(str(args.library.resolve()))
    manifest = json.loads((args.cases / "cases-manifest.json").read_text())
    build = json.loads((args.library.parent / "build-receipt.json").read_text())
    store = args.cases / f"{args.prefix}-words.npz"
    words_all = {}
    for row in manifest["cases"]:
        with np.load(args.cases / row["file"]) as data:
            arrays = {k.removeprefix("input__"): data[k] for k in data.files if k.startswith("input__")}
            meta = json.loads(str(data["meta_json"]))
        for arm in ARMS:
            words = wrf_pipeline(lib, arrays, meta, arm)
            bad = sorted(k for k, v in words.items() if not np.isfinite(v).all())
            words_all.update({f"{row['case']}__{arm}__{k}": v for k, v in words.items()})
            print(row["case"], arm, "nonfinite:" if bad else "finite", bad, flush=True)
    np.savez_compressed(store, **words_all)
    manifest[args.prefix + "_build"] = build
    manifest[args.prefix + "_words_sha256"] = hashlib.sha256(store.read_bytes()).hexdigest()
    (args.cases / "cases-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def compare_words(args):
    """WRF-vs-WRF: e.g. the stock-flag build against the strict reference."""
    from gpuwm.verify.diffusion_oracle import word_comparison
    with np.load(args.cases / f"{args.a}-words.npz") as a, np.load(args.cases / f"{args.b}-words.npz") as b:
        assert set(a.files) == set(b.files)
        total = {"words": 0, "different_words": 0, "max_ulp": 0}
        for key in a.files:
            m = word_comparison(a[key], b[key])
            total["words"] += m["words"]
            total["different_words"] += m["different_words"]
            total["max_ulp"] = max(total["max_ulp"], m["max_ulp"])
    print(json.dumps({"a": args.a, "b": args.b, **total}))


# ---------------------------------------------------------------- engine side

def engine_state(arrays, meta, arm):
    import cupy as cp
    from types import SimpleNamespace
    from gpuwm.config import RunConfig
    nx, ny, nz = meta["nx"], meta["ny"], meta["nz"]
    dev = {k: cp.asarray(np.asarray(v, F32)) for k, v in arrays.items()
           if k not in ("hgt", "lat", "t2", "pb", "p", "mut", "t_init", "tke")}
    dev["thp"] = cp.asarray(arrays["t_init"])       # initial state, for t_init capture
    dev["thb"] = cp.full_like(dev["thp"], 300.0)
    dev["p"] = cp.asarray((np.asarray(arrays["p"], F32) + np.asarray(arrays["pb"], F32)).astype(F32))
    dev["mub2d"] = cp.asarray(arrays["mut"])
    dev["mup"] = cp.zeros_like(dev["mub2d"])
    dev["mup0"] = dev["mup"]
    dev["tke"] = cp.asarray(arrays["tke"])
    for name in ("u", "v", "w", "php", *SPECIES, *NUMBERS, "tke"):
        dev[name + "0"] = dev[name]
    for name in ("ru_t", "rv_t", "rw_t", "rth_t"):
        dev[name] = None
    store = {}

    def scratch(shape, slot, dtype=None):
        shape = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
        buf = store.get(slot)
        if buf is None:
            buf = store[slot] = cp.zeros(shape, dtype=cp.float32 if dtype is None else dtype)
        elif buf.shape != shape:
            raise ValueError(f"scratch slot {slot!r} has shape {buf.shape}, requested {shape}")
        return buf
    state = SimpleNamespace(**dev, scratch=scratch, _scratch=store, physics=None, chem=None,
                            elapsed_seconds=0.0,
                            cf1=F32(meta["cf1"]), cf2=F32(meta["cf2"]), cf3=F32(meta["cf3"]))
    boundary = meta["boundary"]
    c = ARMS[arm]
    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=meta["dx"], dy=meta["dy"], dt=meta["dt"],
                    ztop=20000.0, run_seconds=0.0,
                    open_x=boundary == "open", open_y=boundary == "open",
                    specified=boundary == "specified", km_opt=c["km_opt"], diff_opt=1,
                    c_s=C_S, c_k=C_K, mix_isotropic=c["isotropic"], isfflx=c["isfflx"],
                    bl_pbl_physics=c["pbl"], mix_upper_bound=MIX_UPPER_BOUND,
                    tke_drag_coefficient=0.0013, tke_heat_flux=0.24)
    return state, cfg


def engine_outputs(arrays, meta, arm):
    """Every output word of the production diff_opt=1 path for one case and arm."""
    import cupy as cp
    from gpuwm.core import dycore
    nx, ny, nz = meta["nx"], meta["ny"], meta["nz"]
    km_opt = ARMS[arm]["km_opt"]
    state, cfg = engine_state(arrays, meta, arm)
    # WRF t_init: captured from the initial thermal field exactly as a
    # forecast's first step does, then the state moves to the case's theta.
    dycore.initialize_coordinate_reference(state, cfg)
    state.thp = cp.asarray(arrays["t2"])
    state.thp0 = state.thp
    out = {}
    # The coefficient launchers alone, to keep the deformation tensors that
    # the tendency pass later overwrites in their borrowed buffers.
    km = cp.zeros((nz, ny, nx), cp.float32)
    kh = cp.zeros_like(km)
    if km_opt == 2:
        d11, d22, d12 = dycore.launch_wrf_tke_km(state, cfg, km, kh, time_t=True)
        out["kmv"] = cp.asnumpy(state._scratch["smag_kmv"])
        out["khv"] = cp.asnumpy(state._scratch["smag_khv"])
        out["bn2"] = cp.asnumpy(state._scratch["diff6_x"].reshape(-1)[:nz * ny * nx].reshape(nz, ny, nx))
    else:
        d11, d22, d12 = dycore.launch_wrf_smag2d_km(state, cfg, km, kh, time_t=True)
    out["d11"], out["d22"] = cp.asnumpy(d11), cp.asnumpy(d22)
    bx = meta["boundary"] != "periodic"
    ii = np.minimum(np.arange(nx + 1), nx - 1) if bx else np.arange(nx + 1) % nx
    jj = np.minimum(np.arange(ny + 1), ny - 1) if bx else np.arange(ny + 1) % ny
    out["d12"] = np.ascontiguousarray(cp.asnumpy(d12)[:, jj][:, :, ii])
    first_km, first_kh = cp.asnumpy(km), cp.asnumpy(kh)
    # The forecast's own once-per-step builder (prepare_fixed_tendencies'
    # call), on the production K buffers and carrying slots.
    km2 = state.scratch((nz, ny, nx), "smag_km")
    kh2 = state.scratch((nz, ny, nx), "smag_kh")
    specs = dycore._smag2d_specs(state, km2, kh2, time_t=True,
                                 kmv=dycore._horizontal_w_km(state, cfg))
    for f0, _t, _k, _c1, _c2, slot, _s in specs:
        state.scratch(f0.shape, slot)[...] = 0
    if km_opt == 2:
        state.scratch((nz, ny, nx), "smag_rtke")[...] = 0
    dycore._compute_wrf_smag_tendencies(state, cfg, km2, kh2, specs, time_t=True)
    cp.cuda.Device().synchronize()
    out["kmh"], out["khh"] = cp.asnumpy(km2), cp.asnumpy(kh2)
    out["k_first_launch_equal"] = bool(
        np.array_equal(first_km.view(np.uint32), out["kmh"].view(np.uint32))
        and np.array_equal(first_kh.view(np.uint32), out["khh"].view(np.uint32)))
    rows = {slot: f0 for f0, _t, _k, _c1, _c2, slot, _s in specs}
    slots = {"tu": "smag_ru", "tv": "smag_rv", "tw": "smag_rw", "tth": "smag_rth"}
    slots.update({"t" + n: "smag_r" + n for n in (*SPECIES, *NUMBERS)})
    for key, slot in slots.items():
        if slot not in rows:
            raise ValueError(f"production specs carry no {slot} row")
        out[key] = cp.asnumpy(state.scratch(rows[slot].shape, slot))
    if km_opt == 2:
        out["ttke"] = cp.asnumpy(state._scratch["smag_rtke"])
    return out


# Consumed by the model under diff_opt=1: K_h pair and every tendency.
# kmv/khv/bn2 are written by WRF (history XKMV/XKHV) but nothing reads them
# under diff_opt=1; d11/d22/d12 are the staged tensors feeding K.
CONSUMED = ("kmh", "khh", "tu", "tv", "tw", "tth", *("t" + n for n in SPECIES),
            *("t" + n for n in NUMBERS), "ttke")
DIAGNOSTIC = ("d11", "d22", "d12", "kmv", "khv", "bn2")


def compare(args):
    import cupy as cp
    from gpuwm.verify.diffusion_oracle import word_comparison
    from gpuwm.core.kernels import module_source
    from gpuwm import wrf_exact
    manifest = json.loads((args.cases / "cases-manifest.json").read_text())
    cases, fields, saved = {}, {}, {}
    k_equal = True
    with np.load(args.cases / f"{args.reference}-words.npz") as ref:
        refwords = {k: ref[k] for k in ref.files}
    for row in manifest["cases"]:
        with np.load(args.cases / row["file"]) as data:
            arrays = {k.removeprefix("input__"): data[k] for k in data.files if k.startswith("input__")}
            meta = json.loads(str(data["meta_json"]))
        for arm in ARMS:
            got = engine_outputs(arrays, meta, arm)
            k_equal &= got.pop("k_first_launch_equal")
            result = {}
            for key, value in got.items():
                wrf = refwords[f"{row['case']}__{arm}__{key}"]
                if args.save:
                    saved[f"{row['case']}__{arm}__{key}"] = np.asarray(value, F32)
                m = word_comparison(value, wrf)
                a, b = np.asarray(value, F32), np.asarray(wrf, F32)
                unequal = a.view(np.uint32) != b.view(np.uint32)
                if unequal.any():
                    idx = np.argwhere(unequal)
                    m["first_mismatches"] = [
                        {"kji": [int(x) for x in t], "engine": float(a[tuple(t)]),
                         "engine_hex": f"{int(a.view(np.uint32)[tuple(t)]):08x}",
                         "wrf": float(b[tuple(t)]),
                         "wrf_hex": f"{int(b.view(np.uint32)[tuple(t)]):08x}"} for t in idx[:5]]
                    m["columns_with_difference"] = int(np.unique(idx[:, 1:], axis=0).shape[0])
                result[key] = m
                agg = fields.setdefault(key, {"words": 0, "different_words": 0, "max_ulp": 0,
                                              "max_absolute": 0.0, "case_arms_with_difference": 0})
                agg["words"] += m["words"]
                agg["different_words"] += m["different_words"]
                agg["max_ulp"] = max(agg["max_ulp"], m["max_ulp"])
                agg["max_absolute"] = max(agg["max_absolute"], m["max_absolute"])
                agg["case_arms_with_difference"] += int(m["different_words"] > 0)
            cases[f"{row['case']}__{arm}"] = result
            print(f"{row['case']:34s} {arm:14s} diff={sum(v['different_words'] for v in result.values()):7d} "
                  f"max_ulp={max(v['max_ulp'] for v in result.values())}", flush=True)
    group = lambda names: {
        "words": sum(fields[n]["words"] for n in names if n in fields),
        "different_words": sum(fields[n]["different_words"] for n in names if n in fields),
        "max_ulp": max((fields[n]["max_ulp"] for n in names if n in fields), default=0),
        "max_absolute": max((fields[n]["max_absolute"] for n in names if n in fields), default=0.0)}
    receipt = {
        "schema": "diffopt1-wrf461-column-oracle-receipt-v1",
        "reference": args.reference,
        "reference_build": manifest.get(args.reference + "_build"),
        "arithmetic": {"GPUWM_WRF_EXACT": os.environ.get("GPUWM_WRF_EXACT", "0"),
                       "GPUWM_WRF_EXACT_DIFFUSION": os.environ.get("GPUWM_WRF_EXACT_DIFFUSION", "0"),
                       "strict_options": list(wrf_exact.STRICT_OPTIONS) if wrf_exact.ENABLED else []},
        "device": str(cp.cuda.runtime.getDeviceProperties(0)["name"]),
        "cupy": cp.__version__,
        "module_source_sha256": {n: hashlib.sha256(module_source(n).encode()).hexdigest()
                                 for n in ("smag2d", "diff_opt1")},
        "cases_manifest_sha256": hashlib.sha256((args.cases / "cases-manifest.json").read_bytes()).hexdigest(),
        "case_count": len(manifest["cases"]), "arms": ARMS, "columns": manifest["columns"],
        "k_first_launch_equal": k_equal,
        "consumed": group(CONSUMED), "diagnostic": group(DIAGNOSTIC),
        "fields": fields, "cases": cases}
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    if args.save:
        np.savez_compressed(args.save, **saved)
    print(json.dumps({k: receipt[k] for k in ("arithmetic", "device", "consumed", "diagnostic",
                                              "k_first_launch_equal")}, indent=2))
    for key, agg in fields.items():
        print(f"  {key:6s} words={agg['words']:9d} diff={agg['different_words']:8d} max_ulp={agg['max_ulp']:10d} "
              f"max_abs={agg['max_absolute']:.3e} case_arms={agg['case_arms_with_difference']}")


# ---------------------------------------------------------------- sealed subset

# (corpus case, x0, y0): 8 x 7 column crops kept in the repository.  One
# per regime: convective, stable night, snow/ice over ocean, high terrain
# with snow, warm cloud over ocean on a periodic domain (WRF's seam defect
# lives there), and the signed-zero edge case.
SEAL_CASES = (
    ("convective_plains_3km-specified", 3, 2),
    ("plains_stable_night-specified", 2, 1),
    ("pacific_snow_storm-specified", 3, 3),
    ("rockies_snow_wave-specified", 2, 2),
    ("gulf_cloud_ocean-periodic", 4, 3),
    ("signed_zero_flow-specified", 2, 1),
)
SEAL_NX, SEAL_NY = 8, 7


def crop(arrays, meta, x0, y0, nx=SEAL_NX, ny=SEAL_NY):
    """A smaller window of a corpus case, every field on its own staggering."""
    out = {}
    for key, a in arrays.items():
        a = np.asarray(a)
        if a.ndim == 1:
            out[key] = a.copy()
            continue
        sy = 1 if key in ("v", "msfv") else 0
        sx = 1 if key in ("u", "msfu") else 0
        out[key] = np.ascontiguousarray(a[..., y0:y0 + ny + sy, x0:x0 + nx + sx])
    if meta["boundary"] == "periodic":
        out["u"][:, :, -1] = out["u"][:, :, 0]
        out["msfu"][:, -1] = out["msfu"][:, 0]
        out["v"][:, -1, :] = out["v"][:, 0, :]
        out["msfv"][-1, :] = out["msfv"][0, :]
    return out, {**meta, "nx": nx, "ny": ny, "crop_x0": x0, "crop_y0": y0}


def seal(args):
    """Crop SEAL_CASES, run unmodified WRF and the seam control, keep both."""
    lib = ctypes.CDLL(str(args.library.resolve()))
    ctl = ctypes.CDLL(str(args.control.resolve()))
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, x0, y0 in SEAL_CASES:
        with np.load(args.cases / f"case-{name}.npz") as data:
            arrays = {k.removeprefix("input__"): data[k] for k in data.files if k.startswith("input__")}
            meta = json.loads(str(data["meta_json"]))
        arrays, meta = crop(arrays, meta, x0, y0)
        payload = {"input__" + k: v for k, v in arrays.items()}
        payload["meta_json"] = np.array(json.dumps(meta, sort_keys=True))
        moved = 0
        for arm in ARMS:
            words = wrf_pipeline(lib, arrays, meta, arm)
            control = wrf_pipeline(ctl, arrays, meta, arm)
            for key, value in control.items():
                payload[f"ctl__{arm}__{key}"] = value
                if value.tobytes() != words[key].tobytes():
                    payload[f"wrf__{arm}__{key}"] = words[key]
                    moved += int(np.count_nonzero(value.view(np.uint32) != words[key].view(np.uint32)))
        if meta["boundary"] != "periodic" and moved:
            raise ValueError(f"{name}: the seam control moved a non-periodic word")
        path = args.out / f"case-{name}.npz"
        np.savez_compressed(path, **payload)
        rows.append({"case": name, "file": path.name, "regime": meta["regime"],
                     "boundary": meta["boundary"], "columns": SEAL_NX * SEAL_NY,
                     "words_moved_by_seam_control": moved,
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        print(name, "seam-control moved", moved, flush=True)
    builds = {}
    for tag, library in (("wrf", args.library), ("ctl", args.control)):
        receipt = json.loads((library.parent / "build-receipt.json").read_text())
        receipt.pop("library_sha256", None)
        builds[tag] = receipt
    manifest = {"schema": "diffopt1-wrf461-sealed-v1", "cases": rows, "arms": ARMS,
                "columns": sum(r["columns"] for r in rows), "builds": builds,
                "crop": [SEAL_NX, SEAL_NY]}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                            encoding="utf-8", newline="\n")
    print(len(rows), "sealed cases,", manifest["columns"], "columns")


def replay(args):
    """Engine words for every sealed case and arm against the sealed WRF words.

    Writes {case__arm: {field: {"ctl": different, "wrf": different}}}; "wrf"
    is present only where unmodified WRF differs from the seam control.
    """
    from gpuwm import wrf_exact
    manifest = json.loads((args.sealed / "manifest.json").read_text())
    result = {"strict_options": list(wrf_exact.STRICT_OPTIONS) if wrf_exact.ENABLED else [],
              "cases": {}}
    for row in manifest["cases"]:
        with np.load(args.sealed / row["file"]) as data:
            arrays = {k.removeprefix("input__"): data[k] for k in data.files if k.startswith("input__")}
            meta = json.loads(str(data["meta_json"]))
            sealed = {k: data[k] for k in data.files if k.startswith(("ctl__", "wrf__"))}
        for arm in ARMS:
            got = engine_outputs(arrays, meta, arm)
            got.pop("k_first_launch_equal")
            fields = {}
            for key, value in got.items():
                ctl = sealed[f"ctl__{arm}__{key}"]
                wrf = sealed.get(f"wrf__{arm}__{key}", ctl)
                a = np.asarray(value, F32).view(np.uint32)
                fields[key] = {"ctl": int(np.count_nonzero(a != ctl.view(np.uint32))),
                               "wrf": int(np.count_nonzero(a != wrf.view(np.uint32))),
                               "words": int(a.size)}
            result["cases"][f"{row['case']}__{arm}"] = fields
    args.json.write_text(json.dumps(result, indent=1) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    c = sub.add_parser("cases")
    c.add_argument("--source", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    w = sub.add_parser("wrf")
    w.add_argument("--library", type=Path, required=True)
    w.add_argument("--cases", type=Path, required=True)
    w.add_argument("--prefix", default="wrf")
    x = sub.add_parser("wrf-vs-wrf")
    x.add_argument("--cases", type=Path, required=True)
    x.add_argument("--a", default="wrf")
    x.add_argument("--b", default="wrfstock")
    m = sub.add_parser("compare")
    m.add_argument("--cases", type=Path, required=True)
    m.add_argument("--receipt", type=Path, required=True)
    m.add_argument("--reference", default="wrf")
    m.add_argument("--save", type=Path)
    z = sub.add_parser("seal")
    z.add_argument("--cases", type=Path, required=True)
    z.add_argument("--library", type=Path, required=True)
    z.add_argument("--control", type=Path, required=True)
    z.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("replay")
    r.add_argument("--sealed", type=Path, required=True)
    r.add_argument("--json", type=Path, required=True)
    a = p.parse_args()
    {"cases": make_cases, "wrf": run_wrf, "wrf-vs-wrf": compare_words, "compare": compare,
     "seal": seal, "replay": replay}[a.action](a)


if __name__ == "__main__":
    main()
