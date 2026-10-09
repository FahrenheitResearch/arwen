"""WRF v4.6.1 km_opt=4 / diff_opt=2 column oracle: cases, WRF words, engine words.

  python oracle.py cases   --reader RW_NETCDF --conus WRFOUT --convective WRFOUT --out DIR
  python oracle.py wrf     --library oracle.so --cases DIR
  python oracle.py compare --cases DIR --receipt RECEIPT.json     (GPU, under the mutex)

``cases`` cuts real windows and named edge transformations (see smag2d_cases.py).
``wrf`` runs the compiled WRF pipeline (pipeline.F90) on every case and stores
every WRF output word next to the inputs.  ``compare`` runs the engine's
production launchers -- the same ``_compute_wrf_smag_tendencies`` the forecast
calls -- on the identical input words and compares every output word.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smag2d_cases import (F32, NUMBERS, SPECIES, decode, edge_cases, periodic, save_case,  # noqa: E402
                   scalars, window)

H = 5          # Fortran halo: memory -4..n+5, interior 1..n at python offset 5
NX, NY = 14, 12
BOUNDARY = {"periodic": 0, "open": 1, "specified": 2}

# (name, regime, x, y) windows of the 12 km CONUS history (2019-11-26 13 UTC)
CONUS_WINDOWS = (
    ("rockies_snow_wave", "high terrain, snow, mountain-wave w, stable night", 130, 80),
    ("high_terrain_dry", "high terrain, dry, night", 105, 105),
    ("plains_stable_night", "flat land, nocturnal inversion", 305, 105),
    ("pacific_marine_heating", "ocean, cold air over warm water (surface heating)", 5, 100),
    ("pacific_snow_storm", "ocean, snow and cloud, ascent", 5, 228),
    ("great_lakes_snow", "lake/land snow, night", 330, 230),
    ("gulf_cloud_ocean", "warm ocean, liquid cloud", 230, 5),
    ("atlantic_open_ocean", "open ocean, clear", 405, 30),
)


def _pick(raw, score, nx=NX, ny=NY):
    """Window origin maximising a 2-D score's maximum, away from the edges."""
    best, at = -np.inf, None
    ny0, nx0 = score.shape
    for y in range(4, ny0 - ny - 5, 3):
        for x in range(4, nx0 - nx - 5, 3):
            s = score[y:y + ny, x:x + nx].max()
            if s > best:
                best, at = s, (x, y)
    return at


def make_cases(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    raw, info = decode(args.reader, args.conus, out / "decoded-conus")
    sc = scalars(raw)
    hgt = raw["HGT"]
    slope = np.hypot(np.gradient(hgt, axis=0), np.gradient(hgt, axis=1)) / info["dx"]
    windows = list(CONUS_WINDOWS)
    for name, regime, score in (
            ("steepest_terrain", "steepest real terrain slope in the domain", slope),
            ("snowiest_column", "largest column snow mixing ratio", np.abs(raw["QSNOW"]).max(axis=0)),
            ("strongest_w", "largest |w| in the domain", np.abs(raw["W"]).max(axis=0)),
            ("iciest_column", "largest column cloud-ice mixing ratio", np.abs(raw["QICE"]).max(axis=0))):
        x, y = _pick(raw, score)
        windows.append((name, regime, x, y))
    rows = []

    def add(name, arrays, meta, boundary):
        arrays = {k: v.copy() for k, v in arrays.items()}
        if boundary == "periodic":
            arrays = periodic(arrays)
        meta = {**meta, "boundary": boundary, "nx": NX, "ny": NY,
                "nz": int(arrays["u"].shape[0])}
        case = f"{name}-{boundary}"
        path = save_case(out, case, arrays, meta)
        rows.append({"case": case, "file": path.name, "regime": meta["regime"],
                     "boundary": boundary, "columns": NX * NY,
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        print(case, flush=True)

    base_meta = {**info, **sc}
    for name, regime, x, y in windows:
        a = window(raw, x=x, y=y, nx=NX, ny=NY)
        add(name, a, {**base_meta, "regime": regime, "window_x": x, "window_y": y,
                      "max_hgt_m": float(a["hgt"].max()), "max_abs_w": float(np.abs(a["w"]).max()),
                      "max_qs": float(a["qs"].max()), "max_qi": float(a["qi"].max()),
                      "max_qc": float(a["qc"].max())}, "specified")
        if name in ("rockies_snow_wave", "gulf_cloud_ocean"):
            add(name, a, {**base_meta, "regime": regime + "; periodic lateral boundaries",
                          "window_x": x, "window_y": y}, "periodic")
        if name in ("plains_stable_night",):
            add(name, a, {**base_meta, "regime": regime + "; open lateral boundaries",
                          "window_x": x, "window_y": y}, "open")
    base = window(raw, x=305, y=105, nx=NX, ny=NY)
    for edge, (arrays, note) in edge_cases(base, NX, NY).items():
        add(edge, arrays, {**base_meta, "regime": "edge case: " + note,
                           "window_x": 305, "window_y": 105}, "specified")
        if edge in ("map_extremes", "updraft_core"):
            add(edge, arrays, {**base_meta, "regime": "edge case: " + note + "; periodic",
                               "window_x": 305, "window_y": 105}, "periodic")
        if edge in ("updraft_core",):
            add(edge, arrays, {**base_meta, "regime": "edge case: " + note + "; open",
                               "window_x": 305, "window_y": 105}, "open")
    craw, cinfo = decode(args.reader, args.convective, out / "decoded-convective",
                         time_index=args.convective_time)
    csc = scalars(craw)
    a = window(craw, x=13, y=14, nx=NX, ny=NY)
    add("convective_plains_3km", a, {**cinfo, **csc, "regime":
        "3 km warm-season convective environment (2024-05-21, Iowa), deep moist unstable, strong shear",
        "window_x": 13, "window_y": 14, "max_abs_w": float(np.abs(a["w"]).max())}, "specified")
    manifest = {"schema": "smag2d-wrf461-cases-v1", "cases": rows,
                "columns": sum(r["columns"] for r in rows)}
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


OUT3 = ("z", "rdz", "rdzw", "zx", "zy", "rho", "div", "d11", "d22", "d33", "d12", "d13", "d23",
        "kmh", "kmv", "khh", "khv", "bn2", "tu", "tv", "tw", "tth", "ttke")
# (name, (sx, sy, sz)) WRF output words kept per case
KEEP = {"d11": (0, 0, 0), "d22": (0, 0, 0), "d12": (1, 1, 0), "d13": (1, 0, 1), "d23": (0, 1, 1),
        "d33": (0, 0, 0), "div": (0, 0, 0), "kmh": (0, 0, 0), "kmv": (0, 0, 0), "khh": (0, 0, 0),
        "khv": (0, 0, 0), "tu": (1, 0, 0), "tv": (0, 1, 0), "tw": (0, 0, 1), "tth": (0, 0, 0),
        "rho": (0, 0, 0), "rdzw": (0, 0, 0), "zx": (1, 0, 1), "zy": (0, 1, 1)}


def wrf_pipeline(lib, arrays, meta, *, mix_full=1):
    nx, ny, nz = meta["nx"], meta["ny"], meta["nz"]
    fn = lib.smag2d_km4_pipeline
    fn.restype = None
    n_moist, n_scalar = 1 + len(SPECIES), 1 + len(NUMBERS)
    ins3 = {k: _pad(arrays[k], nx, ny, nz) for k in ("u", "v", "w", "php", "phb", "t2", "alt", "p", "pb")}
    ins3["tke"] = np.zeros_like(ins3["t2"])
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
    args = [ctypes.c_int(v) for v in (nx, ny, nz, BOUNDARY[meta["boundary"]], mix_full, n_moist, n_scalar)]
    args += [ptr(ins3[k]) for k in ("u", "v", "w", "php", "phb", "t2", "alt", "p", "pb")]
    args += [ptr(moist), ptr(scalar), ptr(ins3["tke"])]
    args += [ptr(ins2[k]) for k in ("msfu", "msfv", "msft", "mut")]
    args += [ptr(cols[k]) for k in ("c1h", "c2h", "c1f", "c2f", "dn", "dnw", "fnm", "fnp", "znw")]
    args += [ctypes.c_float(v) for v in (dx, dy, rdx, rdy, meta["cf1"], meta["cf2"], meta["cf3"],
                                         meta["p_top"], 0.25, 1.0)]
    args += [ptr(outs[k]) for k in OUT3] + [ptr(tmoist), ptr(tscalar)]
    fn(*args)
    result = {k: _core(outs[k], nx, ny, nz, *KEEP[k]) for k in KEEP}
    for slot, name in enumerate(SPECIES, start=1):
        result["t" + name] = _core(tmoist[..., slot], nx, ny, nz)
    for slot, name in enumerate(NUMBERS, start=1):
        result["t" + name] = _core(tscalar[..., slot], nx, ny, nz)
    return result


def run_wrf(args):
    lib = ctypes.CDLL(str(args.library.resolve()))
    prefix = args.prefix
    manifest = json.loads((args.cases / "cases-manifest.json").read_text())
    build = json.loads((args.library.parent / "build-receipt.json").read_text())
    mix_note = {}
    for row in manifest["cases"]:
        path = args.cases / row["file"]
        with np.load(path) as data:
            payload = {k: data[k] for k in data.files}
        arrays = {k.removeprefix("input__"): v for k, v in payload.items() if k.startswith("input__")}
        meta = json.loads(str(payload["meta_json"]))
        words = wrf_pipeline(lib, arrays, meta, mix_full=1)
        other = wrf_pipeline(lib, arrays, meta, mix_full=0)
        differs = sorted(k for k in words if words[k].tobytes() != other[k].tobytes())
        mix_note[row["case"]] = differs
        payload = {k: v for k, v in payload.items() if not k.startswith(prefix + "__")}
        payload.update({prefix + "__" + k: v for k, v in words.items()})
        np.savez_compressed(path, **payload)
        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        bad = [k for k, v in words.items() if not np.isfinite(v).all()]
        print(row["case"], "nonfinite:" if bad else "finite", bad, "mix_full_fields-differs:", differs, flush=True)
    manifest[prefix + "_build"] = build
    manifest[prefix + "_mix_full_fields_false_differs"] = mix_note
    (args.cases / "cases-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


# ---------------------------------------------------------------- engine side

def engine_outputs(arrays, meta):
    """Every output word of the production km_opt=4 path for one case."""
    import cupy as cp
    from types import SimpleNamespace
    from gpuwm.config import RunConfig
    from gpuwm.core import dycore
    from gpuwm.core.kernels import module_options, module_source, compile_diffusion_source
    nx, ny, nz = meta["nx"], meta["ny"], meta["nz"]
    dev = {k: cp.asarray(np.asarray(v, F32)) for k, v in arrays.items()
           if k not in ("hgt", "lat", "t2", "pb", "p", "mut")}
    dev["thp"] = cp.asarray(arrays["t2"])
    dev["thb"] = cp.full_like(dev["thp"], 300.0)
    dev["p"] = cp.asarray(np.asarray(arrays["p"], F32) + np.asarray(arrays["pb"], F32))
    dev["mub2d"] = cp.asarray(arrays["mut"])
    dev["mup"] = cp.zeros_like(dev["mub2d"])
    dev["mup0"] = dev["mup"]
    for name in ("u", "v", "w", "php", "thp", *SPECIES, *NUMBERS):
        dev[name + "0"] = dev[name]
    for name in ("ru_t", "rv_t", "rw_t", "rth_t"):
        dev[name] = None
    cache = {}

    def scratch(shape, slot):
        key = (tuple(shape), slot)
        if key not in cache:
            cache[key] = cp.zeros(shape, dtype=cp.float32)
        return cache[key]
    state = SimpleNamespace(**dev, scratch=scratch, physics=None, chem=None,
                            cf1=F32(meta["cf1"]), cf2=F32(meta["cf2"]), cf3=F32(meta["cf3"]))
    boundary = meta["boundary"]
    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=meta["dx"], dy=meta["dy"], dt=1.0, ztop=20000.0,
                    run_seconds=0.0, open_x=boundary == "open", open_y=boundary == "open",
                    specified=boundary == "specified", km_opt=4, diff_opt=2, c_s=0.25,
                    bl_pbl_physics=1)
    out = {}
    km = cp.zeros((nz, ny, nx), cp.float32)
    kh = cp.zeros_like(km)
    d11, d22, d12 = dycore.launch_wrf_smag2d_km(state, cfg, km, kh, time_t=True)
    out["d11"], out["d22"] = cp.asnumpy(d11), cp.asnumpy(d22)
    bx = boundary != "periodic"
    ii = np.minimum(np.arange(nx + 1), nx - 1) if bx else np.arange(nx + 1) % nx
    jj = np.minimum(np.arange(ny + 1), ny - 1) if bx else np.arange(ny + 1) % ny
    out["d12"] = np.ascontiguousarray(cp.asnumpy(d12)[:, jj][:, :, ii])
    # The production w operator evaluates D13/D23 inline (wrf_defor13 and
    # wrf_defor23); the probe launches those same device functions.
    probe_src = (module_source("smag2d").replace("#undef WRF_SMAG_GRID_ARGS", "")
                 .replace("#undef WRF_SMAG_MAKE_GRID", "")
                 + (Path(__file__).with_name("tensor_probe.cu")).read_text())
    probe = compile_diffusion_source(probe_src, "verify:km4-tensor-probe", module_options("smag2d"))
    d13 = cp.zeros((nz + 1, ny, nx + 1), cp.float32)
    d23 = cp.zeros((nz + 1, ny + 1, nx), cp.float32)
    probe.get_function("oracle_expose_d13_d23")(
        ((nx + 128) // 128, ny + 1, nz + 1), (128, 1, 1),
        tuple(dycore._wrf_smag_grid_args(state, cfg, time_t=True) + [d13, d23] +
              [np.int32(nz), np.int32(ny), np.int32(nx), np.int32(1),
               np.int32(bx), np.int32(bx)]))
    out["d13"], out["d23"] = cp.asnumpy(d13), cp.asnumpy(d23)
    # The forecast's own once-per-step builder, on fresh K buffers.
    km2 = state.scratch((nz, ny, nx), "smag_km")
    kh2 = state.scratch((nz, ny, nx), "smag_kh")
    specs = dycore._smag2d_specs(state, km2, kh2, time_t=True,
                                 kmv=dycore._horizontal_w_km(state, cfg))
    for f0, _t, _k, _c1, _c2, slot, _s in specs:
        state.scratch(f0.shape, slot)[...] = 0
    dycore._compute_wrf_smag_tendencies(state, cfg, km2, kh2, specs, time_t=True)
    cp.cuda.Device().synchronize()
    out["kmh"], out["khh"] = cp.asnumpy(km2), cp.asnumpy(kh2)
    out["kmv"] = out["kmh"]          # the w row's K: _horizontal_w_km is None for km_opt=4
    out["k_first_launch_equal"] = bool(np.array_equal(cp.asnumpy(km).view(np.uint32),
                                                      out["kmh"].view(np.uint32)))
    slots = {"tu": "smag_ru", "tv": "smag_rv", "tw": "smag_rw", "tth": "smag_rth"}
    slots.update({"t" + n: "smag_r" + n for n in (*SPECIES, *NUMBERS)})
    rows = {slot: f0 for f0, _t, _k, _c1, _c2, slot, _s in specs}
    for key, slot in slots.items():
        if slot not in rows:
            raise ValueError(f"production specs carry no {slot} row")
        out[key] = cp.asnumpy(state.scratch(rows[slot].shape, slot))
    return out


COMPARED = ("d11", "d22", "d12", "d13", "d23", "kmh", "kmv", "khh", "tu", "tv", "tw", "tth",
            *("t" + n for n in SPECIES), *("t" + n for n in NUMBERS))


def compare(args):
    import cupy as cp
    from gpuwm.verify.diffusion_oracle import word_comparison
    from gpuwm.core.kernels import module_source
    from gpuwm import wrf_exact
    manifest_path = args.cases / "cases-manifest.json"
    if not manifest_path.exists():
        manifest_path = args.cases / "manifest.json"      # the sealed repository fixtures
    manifest = json.loads(manifest_path.read_text())
    cases, fields, saved, fallback = {}, {}, {}, []
    for row in manifest["cases"]:
        with np.load(args.cases / row["file"]) as data:
            arrays = {k.removeprefix("input__"): data[k] for k in data.files if k.startswith("input__")}
            wrf = {k.removeprefix(args.reference + "__"): data[k] for k in data.files
                   if k.startswith(args.reference + "__")}
            if not wrf and args.reference == "wrfctl" and row["boundary"] != "periodic":
                # The control edits only periodic-seam loops; for a specified
                # or open case its words are the reference's (checked when the
                # corpus was built), and the sealed set keeps one copy.
                wrf = {k.removeprefix("wrf__"): data[k] for k in data.files if k.startswith("wrf__")}
                fallback.append(row["case"])
            meta = json.loads(str(data["meta_json"]))
        got = engine_outputs(arrays, meta)
        if args.save:
            saved.update({f"{row['case']}__{k}": np.asarray(got[k], F32) for k in COMPARED})
        result = {}
        for key in COMPARED:
            m = word_comparison(got[key], wrf[key])
            a, b = np.asarray(got[key], F32), np.asarray(wrf[key], F32)
            scale = float(np.max(np.abs(b.astype(np.float64)), initial=0.0))
            m["max_absolute_over_field_max"] = m["max_absolute"] / scale if scale else 0.0
            unequal = a.view(np.uint32) != b.view(np.uint32)
            if unequal.any():
                idx = np.argwhere(unequal)
                m["first_mismatches"] = [
                    {"kji": [int(x) for x in t], "engine": float(a[tuple(t)]), "wrf": float(b[tuple(t)])}
                    for t in idx[:5]]
                # Columns (j, i) holding any differing word.
                m["columns_with_difference"] = int(np.unique(idx[:, 1:], axis=0).shape[0])
            result[key] = m
            agg = fields.setdefault(key, {"words": 0, "different_words": 0, "max_ulp": 0,
                                          "max_absolute": 0.0, "cases_with_difference": 0})
            agg["words"] += m["words"]
            agg["different_words"] += m["different_words"]
            agg["max_ulp"] = max(agg["max_ulp"], m["max_ulp"])
            agg["max_absolute"] = max(agg["max_absolute"], m["max_absolute"])
            agg["cases_with_difference"] += int(m["different_words"] > 0)
        result["_k_first_launch_equal"] = got["k_first_launch_equal"]
        cases[row["case"]] = result
        worst = max(v["max_ulp"] for k, v in result.items() if not k.startswith("_"))
        print(f"{row['case']:40s} different_words={sum(v['different_words'] for k, v in result.items() if not k.startswith('_')):8d} max_ulp={worst}", flush=True)
    receipt = {
        "schema": "smag2d-wrf461-column-oracle-receipt-v1",
        "reference": args.reference,
        "reference_build": manifest.get(args.reference + "_build"),
        "reference_words_shared_with_wrf": fallback,
        "arithmetic": {"GPUWM_WRF_EXACT": os.environ.get("GPUWM_WRF_EXACT"),
                       "GPUWM_WRF_EXACT_DIFFUSION": os.environ.get("GPUWM_WRF_EXACT_DIFFUSION"),
                       "strict_build": wrf_exact.ENABLED,
                       "diffusion_order_selected": wrf_exact.DIFFUSION_ENABLED,
                       "strict_options": list(wrf_exact.STRICT_OPTIONS) if wrf_exact.ENABLED else []},
        "device": str(cp.cuda.runtime.getDeviceProperties(0)["name"]),
        "cupy": cp.__version__,
        "smag2d_source_sha256": hashlib.sha256(module_source("smag2d").encode()).hexdigest(),
        "cases_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "case_count": len(cases), "columns": manifest["columns"],
        "fields": fields, "cases": cases,
        "total_words": sum(v["words"] for v in fields.values()),
        "total_different_words": sum(v["different_words"] for v in fields.values()),
        "max_ulp": max(v["max_ulp"] for v in fields.values()),
        "max_absolute": max(v["max_absolute"] for v in fields.values())}
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    if args.save:
        np.savez_compressed(args.save, **saved)
    print(json.dumps({k: receipt[k] for k in ("arithmetic", "device", "total_words",
                                              "total_different_words", "max_ulp", "max_absolute")}, indent=2))
    for key, agg in fields.items():
        print(f"  {key:6s} words={agg['words']:9d} diff={agg['different_words']:8d} max_ulp={agg['max_ulp']:10d} "
              f"max_abs={agg['max_absolute']:.3e} cases={agg['cases_with_difference']}")


def seal(args):
    """Copy the corpus into the repository's compact, hash-sealed fixture set.

    Kept per case: every input word, the unmodified-WRF words of every
    compared output plus WRF's zx/zy, and for periodic cases the attribution
    control's words of every compared output.
    """
    manifest = json.loads((args.cases / "cases-manifest.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in manifest["cases"]:
        with np.load(args.cases / row["file"]) as data:
            keep = {k: data[k] for k in data.files
                    if k.startswith("input__") or k == "meta_json"
                    or k in {"wrf__" + c for c in (*COMPARED, "zx", "zy")}
                    or (row["boundary"] == "periodic" and k in {"wrfctl__" + c for c in COMPARED})}
            if row["boundary"] != "periodic":
                same = all(data["wrf__" + c].tobytes() == data["wrfctl__" + c].tobytes() for c in COMPARED)
                if not same:
                    raise ValueError(f"{row['case']}: the control moved a non-periodic word")
        path = args.out / row["file"]
        np.savez_compressed(path, **keep)
        rows.append({**row, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    sealed = {"schema": "smag2d-wrf461-fixtures-v1", "cases": rows, "columns": manifest["columns"],
              "compared_fields": list(COMPARED),
              "wrf_build": manifest["wrf_build"], "wrfctl_build": manifest["wrfctl_build"],
              "mix_full_fields_false_differs": manifest["wrf_mix_full_fields_false_differs"]}
    for build in ("wrf_build", "wrfctl_build"):
        sealed[build].pop("library_sha256", None)
    (args.out / "manifest.json").write_text(json.dumps(sealed, indent=2) + "\n", encoding="utf-8",
                                            newline="\n")
    print(len(rows), "cases sealed into", args.out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    c = sub.add_parser("cases")
    c.add_argument("--reader", type=Path, required=True)
    c.add_argument("--conus", type=Path, required=True)
    c.add_argument("--convective", type=Path, required=True)
    c.add_argument("--convective-time", type=int, default=1,
                   help="history frame of --convective (the 21:00:15 UTC frame of the source run)")
    c.add_argument("--out", type=Path, required=True)
    w = sub.add_parser("wrf")
    w.add_argument("--library", type=Path, required=True)
    w.add_argument("--cases", type=Path, required=True)
    w.add_argument("--prefix", default="wrf",
                   help="key prefix: wrf for the unmodified reference, wrfctl for an attribution control")
    m = sub.add_parser("compare")
    m.add_argument("--cases", type=Path, required=True)
    m.add_argument("--receipt", type=Path, required=True)
    m.add_argument("--reference", default="wrf", help="wrf (unmodified WRF) or wrfctl (control)")
    m.add_argument("--save", type=Path, help="also keep every engine output word in this npz")
    z = sub.add_parser("seal")
    z.add_argument("--cases", type=Path, required=True)
    z.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    {"cases": make_cases, "wrf": run_wrf, "compare": compare, "seal": seal}[a.action](a)


if __name__ == "__main__":
    main()
