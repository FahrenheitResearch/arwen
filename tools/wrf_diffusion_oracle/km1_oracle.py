"""km_opt=1 column oracle: compiled WRF v4.6.1 versus the production launchers.

Three steps, each its own subcommand:

``cases``    crop real columns from a real.exe wrfinput (Rust ``rw_netcdf``
             raw decode) into regime patches and edge transformations.
``capture``  run the compiled WRF v4.6.1 package (km1_build.py) on every
             patch and arm: compute_diff_metrics, phy_prep,
             cal_deform_and_div, phy_bc, calculate_km_kh(km_opt=1) ->
             isotropic_km, phy_bc, vertical_diffusion_2 (PBL off) and
             horizontal_diffusion_2, in module_first_rk_step_part2.F order.
``compare``  run gpuwm's production ``dycore._compute_wrf_smag_tendencies``
             (the function ``prepare_fixed_tendencies`` calls) on the same
             inputs and measure every output word.  Select the arithmetic
             before the process imports gpuwm (GPUWM_WRF_EXACT=1 for the
             strict build).

A patch is a 12 x 10 block of real columns with all 50 levels.  The
horizontal operator needs neighbours, so a "column" here is one interior
column of a patch and its outputs are every level of every tendency in it.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

from cases import core, pad2, pad3
from deformation_reference import _invoke, reference_for_case

NX, NY = 12, 10
READ_VARIABLES = ("U V W T P PB PH PHB QVAPOR AL ALB MU MUB MAPFAC_M MAPFAC_U "
                  "MAPFAC_V FNM FNP ZNW P_TOP DN DNW C1H C2H C1F C2F CF1 CF2 CF3 "
                  "XLAT XLONG HGT LANDMASK SNOWH TSK").split()
MOIST = ("qv", "qc", "qi", "qr", "qs", "qg")       # WRF moist slots 2..7
# Arms: (name, khdif, kvdif, bl_pbl_physics, isfflx, cd0, heat).
ARMS = (
    ("pbl_off_k75", 75.0, 75.0, 0, 0, 0.0013, 0.24),
    ("pbl_off_meso", 500.0, 2.0, 0, 1, 0.0, 0.0),
    ("pbl_on_horizontal", 300.0, 1.0, 1, 1, 0.0, 0.0),
    ("pbl_off_surface_only", 0.0, 0.0, 0, 2, 0.0, 0.1),
    ("pbl_off_odd_words", float(np.float32(1.0) / np.float32(3.0)), 7.0e-3, 0, 0, 0.0021, -0.05),
)
OUTPUTS = ("u", "v", "w", "theta") + MOIST + ("hfx_after",)


# ---------------------------------------------------------------- cases ----
def raw_dump(source: Path, reader: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(reader), "dump", "--raw", str(source), str(out), *READ_VARIABLES], check=True)
    meta = json.loads((out / "metadata.json").read_text())
    data = {v["name"]: np.fromfile(out / v["filename"], dtype="<f8").reshape(v["shape"])[0].astype(np.float32)
            for v in meta["variables"]}
    inventory = json.loads(subprocess.check_output([str(reader), "inventory", str(source)]))
    attrs = inventory["global_attributes"]
    if isinstance(attrs, list):
        attrs = {a["name"]: a["value"] for a in attrs}
    info = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "reader_sha256": hashlib.sha256(reader.read_bytes()).hexdigest(),
            "dx": float(attrs["DX"]), "dy": float(attrs["DY"]),
            "start_date": str(attrs.get("START_DATE", ""))}
    return data, info


def _windows(field2d, nx=NX, ny=NY, stride=4):
    H, W = field2d.shape
    for y in range(4, H - ny - 4, stride):
        for x in range(4, W - nx - 4, stride):
            yield y, x, field2d[y:y + ny, x:x + nx]


def choose_crops(raw):
    """Regime patches chosen from the analysis itself (indices recorded)."""
    hgt, land, lat, lon = raw["HGT"], raw["LANDMASK"], raw["XLAT"], raw["XLONG"]
    snow, qv0 = raw["SNOWH"], raw["QVAPOR"][0]
    th = raw["T"] + np.float32(300.0)
    stab = th[4] - th[0]                     # lower-layer stability (K)
    def best(score):
        top = None
        for y, x, _ in _windows(hgt):
            s = score(y, x, (slice(y, y + NY), slice(x, x + NX)))
            if s is not None and (top is None or s > top[0]):
                top = (s, y, x)
        if top is None:
            raise ValueError("no window satisfies the regime rule")
        return top[1], top[2]
    rules = {
        "high_terrain": lambda y, x, w: float(hgt[w].mean()),
        "ocean_pacific": lambda y, x, w: (-float(lon[w].mean()) if land[w].max() == 0 and lon[w].max() < -122 else None),
        "ocean_gulf": lambda y, x, w: (float(qv0[w].mean()) if land[w].max() == 0 and lat[w].max() < 29.5 and -97 < lon[w].min() and lon[w].max() < -82 else None),
        "stable_night_plains": lambda y, x, w: (float(stab[w].mean()) if land[w].min() == 1 and hgt[w].std() < 60 and -102 < lon[w].min() and lon[w].max() < -94 and 34 < lat[w].min() and lat[w].max() < 44 else None),
        "snow_cover": lambda y, x, w: (float(snow[w].mean()) if land[w].min() == 1 else None),
        "moist_convective": lambda y, x, w: (float(qv0[w].mean()) if land[w].min() == 1 else None),
        "coast_mixed": lambda y, x, w: (-abs(float(land[w].mean()) - 0.5) if lon[w].min() > -90 and lat[w].min() > 40 else None),
        "atlantic_ocean": lambda y, x, w: (float(lon[w].mean()) if land[w].max() == 0 and lon[w].min() > -74 and lat[w].min() > 30 else None),
    }
    return {name: best(rule) for name, rule in rules.items()}


def _synthetic_hydrometeors(a):
    """Deterministic condensate from the real column state (all six species).

    The analysis carries vapour only; mixing is linear in each species, so
    condensate placed where the real air is near saturation exercises every
    moist row with realistic magnitudes and vertical structure.
    """
    p = a["p"].astype(np.float64)
    th = (a["thp"] + np.float32(300.0)).astype(np.float64)
    t = th * (p / 1.0e5) ** (287.0 / 1004.0)
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    qs = 0.622 * es / np.maximum(p - es, 1.0)
    rh = a["qv"].astype(np.float64) / qs
    wet = np.clip((rh - 0.6) / 0.4, 0.0, 1.0)
    warm = np.clip((t - 258.0) / 15.0, 0.0, 1.0)
    out = {"qc": 4e-4 * wet * warm, "qr": 2e-4 * wet ** 2 * warm,
           "qi": 1.5e-4 * wet * (1 - warm), "qs": 3e-4 * wet * (1 - warm) ** 0.5,
           "qg": 1e-4 * wet ** 3 * (1 - warm) * (t > 240)}
    return {k: v.astype(np.float32) for k, v in out.items()}


def crop(raw, y, x, *, bx, by):
    pairs = {"u": "U", "v": "V", "w": "W", "php": "PH", "phb": "PHB", "qv": "QVAPOR",
             "msft": "MAPFAC_M", "msfu": "MAPFAC_U", "msfv": "MAPFAC_V"}
    a = {}
    for name, var in pairs.items():
        sy, sx = NY + (name in ("v", "msfv")), NX + (name in ("u", "msfu"))
        a[name] = raw[var][..., y:y + sy, x:x + sx].copy()
    a["alt"] = np.add(raw["AL"], raw["ALB"], dtype=np.float32)[:, y:y + NY, x:x + NX].copy()
    a["p"] = np.add(raw["P"], raw["PB"], dtype=np.float32)[:, y:y + NY, x:x + NX].copy()
    a["thp"] = raw["T"][:, y:y + NY, x:x + NX].copy()          # strict ingest: thb = 300
    a["mut"] = np.add(raw["MU"], raw["MUB"], dtype=np.float32)[y:y + NY, x:x + NX].copy()
    a["mub2d"] = raw["MUB"][y:y + NY, x:x + NX].copy()
    for name in ("fnm", "fnp", "znw", "dn", "dnw", "c1h", "c2h", "c1f", "c2f"):
        a[name] = raw[name.upper()].copy()
    a.update(_synthetic_hydrometeors(a))
    iy, ix = np.meshgrid(np.arange(NY), np.arange(NX), indexing="ij")
    phase = (np.sin(2 * np.pi * ix / NX + 0.3) * np.cos(2 * np.pi * iy / NY)).astype(np.float32)
    a["ust"] = (0.35 + 0.15 * phase).astype(np.float32)
    a["hfx"] = (60.0 + 45.0 * phase).astype(np.float32)
    a["qfx"] = (5.0e-5 + 3.0e-5 * phase).astype(np.float32)
    if not bx:
        a["u"][:, :, -1] = a["u"][:, :, 0]
        a["msfu"][:, -1] = a["msfu"][:, 0]
    if not by:
        a["v"][:, -1, :] = a["v"][:, 0, :]
        a["msfv"][-1, :] = a["msfv"][0, :]
    return a


def edge_variant(a, name):
    a = {k: v.copy() for k, v in a.items()}
    iy, ix = np.meshgrid(np.arange(NY), np.arange(NX), indexing="ij")
    if name == "steep":
        terrain = (1500 * np.sin(2 * np.pi * ix / NX) * np.cos(2 * np.pi * iy / NY)).astype(np.float32)
        taper = np.linspace(1., 0., a["phb"].shape[0], dtype=np.float32)
        a["phb"] += np.float32(9.81) * taper[:, None, None] * terrain[None]
    elif name == "map_extremes":
        for key in ("msft", "msfu", "msfv"):
            jy, jx = np.meshgrid(np.arange(a[key].shape[0]), np.arange(a[key].shape[1]), indexing="ij")
            a[key] = (.25 + 3.75 * (.5 + .5 * np.sin(2 * np.pi * jx / NX) * np.cos(2 * np.pi * jy / NY))).astype(np.float32)
    elif name in ("zero_flow", "near_zero_flow"):
        scale = np.float32(0. if name == "zero_flow" else 1.e-20)
        for key in ("u", "v", "w"):
            a[key] = a[key] * scale
        for key in MOIST[1:]:
            a[key].fill(0.)
    else:
        raise ValueError(name)
    return a


def make_cases(args):
    raw, info = raw_dump(args.wrfinput, args.reader, args.work / "raw")
    chosen = choose_crops(raw)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "km1-wrf461-cases-v1", "source": info, "patch": [NX, NY], "cases": []}
    common = {"dx": info["dx"], "dy": info["dy"], "nx": NX, "ny": NY, "nz": int(raw["T"].shape[0]),
              "p_top": float(raw["P_TOP"]), "cf1": float(raw["CF1"]), "cf2": float(raw["CF2"]),
              "cf3": float(raw["CF3"]), "dt": 1.0}
    entries = []
    for index, (name, (y, x)) in enumerate(chosen.items()):
        bx = by = int(index % 2 == 0)                 # alternate open / periodic
        entries.append((name, crop(raw, y, x, bx=bx, by=by),
                        {**common, "bx": bx, "by": by, "y0": int(y), "x0": int(x),
                         "lat": float(raw["XLAT"][y:y + NY, x:x + NX].mean()),
                         "lon": float(raw["XLONG"][y:y + NY, x:x + NX].mean()),
                         "hgt_mean": float(raw["HGT"][y:y + NY, x:x + NX].mean())}))
    plains = next(e for e in entries if e[0] == "stable_night_plains")
    for variant, (bx, by) in (("steep", (1, 1)), ("map_extremes", (0, 0)),
                              ("zero_flow", (0, 0)), ("near_zero_flow", (1, 1))):
        y, x = plains[2]["y0"], plains[2]["x0"]
        base = crop(raw, y, x, bx=bx, by=by)
        entries.append(("edge_" + variant, edge_variant(base, variant),
                        {**plains[2], "bx": bx, "by": by, "edge": variant}))
    for name, arrays, meta in entries:
        target = args.output / f"km1-input-{name}.npz"
        np.savez_compressed(target, meta_json=np.array(json.dumps(meta, sort_keys=True)),
                            **{"in_" + k: v for k, v in arrays.items()})
        manifest["cases"].append({"name": name, "file": target.name, "meta": meta,
                                  "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
        print(name, meta.get("lat"), meta.get("lon"), "open" if meta["bx"] else "periodic", flush=True)
    (args.output / "km1-cases.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")


# -------------------------------------------------------------- capture ----
def wrf_reference(lib, arrays, meta, arm):
    _name, khdif, kvdif, pbl, isfflx, cd0, heat = arm
    nx, ny, nz, bx, by = (int(meta[k]) for k in ("nx", "ny", "nz", "bx", "by"))
    head = [nx, ny, nz, bx, by]
    shape = (nx + 6, nz + 1, ny + 6)
    prep = reference_for_case(lib, {**arrays, "thb": np.full(nz, 300., np.float32)}, meta, km_opt=1)
    k = {key: np.zeros(shape, np.float32, order="F") for key in ("kmh", "kmv", "khh", "khv", "bn2")}
    _invoke(lib, "oracle_km1_coefficients", head + [float(np.float32(khdif)), float(np.float32(kvdif))] + [
        prep[key] for key in ("theta", "temp", "p", "p8w", "t8w", "moist", "tke", "msft", "rdz", "rdzw",
                              "zx", "zy", "dn", "dnw", "div", "d11", "d22", "d33", "d12", "d13", "d23")]
        + [float(meta["dx"]), float(meta["dy"]), float(meta["dt"]), float(meta["cf1"]), float(meta["cf2"]),
           float(meta["cf3"])] + [k[key] for key in ("kmh", "kmv", "khh", "khv", "bn2")])
    for key in ("kmh", "kmv", "khh", "khv"):
        _invoke(lib, "oracle_bc", head + [0, k[key]])
    moist = np.zeros((*shape, 7), np.float32, order="F")
    for slot, key in enumerate(MOIST, start=1):
        moist[:, :, :, slot] = pad3(arrays[key], nx, ny, nz, bx, by)
    t2 = pad3(arrays["thp"], nx, ny, nz, bx, by)
    sfc = {key: pad2(arrays[key], nx, ny, bx, by) for key in ("hfx", "qfx", "ust")}
    out = {key: np.zeros(shape, np.float32, order="F") for key in ("tu", "tv", "tw", "tth")}
    tmoist = np.zeros((*shape, 7), np.float32, order="F")
    fn = lib.oracle_km1_package
    fn.restype = None
    args = head + [pbl, isfflx]
    ptr = lambda a: ctypes.c_void_p(a.ctypes.data)
    seq = ([ctypes.c_int(v) for v in args] + [ctypes.c_float(cd0), ctypes.c_float(heat)]
           + [ptr(a) for a in (prep["u"], prep["v"], t2, prep["theta"], prep["tke"], moist,
                               prep["d11"], prep["d22"], prep["d33"], prep["d12"], prep["d13"], prep["d23"],
                               prep["div"], k["kmh"], k["kmv"], k["khh"], k["khv"], prep["rho"], prep["rdz"],
                               prep["rdzw"], prep["zx"], prep["zy"], prep["msfu"], prep["msfv"], prep["msft"],
                               prep["dn"], prep["dnw"], prep["fnm"], prep["fnp"])]
           + [ctypes.c_float(np.float32(1.0 / meta["dx"])), ctypes.c_float(np.float32(1.0 / meta["dy"])),
              ctypes.c_float(meta["cf1"]), ctypes.c_float(meta["cf2"]), ctypes.c_float(meta["cf3"])]
           + [ptr(sfc["hfx"]), ptr(sfc["qfx"]), ptr(sfc["ust"])]
           + [ptr(out[key]) for key in ("tu", "tv", "tw", "tth")] + [ptr(tmoist)])
    fn(*seq)
    result = {"u": core(out["tu"], nx, ny, nz, "x"), "v": core(out["tv"], nx, ny, nz, "y"),
              "w": core(out["tw"], nx, ny, nz, "z"), "theta": core(out["tth"], nx, ny, nz)}
    for slot, key in enumerate(MOIST, start=1):
        result[key] = core(tmoist[:, :, :, slot], nx, ny, nz)
    result["hfx_after"] = np.ascontiguousarray(sfc["hfx"][3:3 + nx, 3:3 + ny].T)
    for key in ("kmh", "kmv", "khh", "khv"):
        result["coef_" + key] = core(k[key], nx, ny, nz)
    return result


def capture(args):
    lib = ctypes.CDLL(str(args.library))
    manifest = json.loads((args.folder / "km1-cases.json").read_text())
    receipt = json.loads((args.library.parent / "build-receipt.json").read_text())
    manifest["fortran_build"] = receipt
    manifest["arms"] = [list(a) for a in ARMS]
    for case in manifest["cases"]:
        path = args.folder / case["file"]
        with np.load(path) as data:
            arrays = {k[3:]: data[k] for k in data.files if k.startswith("in_")}
            meta = json.loads(str(data["meta_json"]))
        payload = {}
        for arm in ARMS:
            for key, value in wrf_reference(lib, arrays, meta, arm).items():
                payload[f"{arm[0]}__{key}"] = value
        target = args.folder / f"km1-wrf461-{case['name']}.npz"
        np.savez_compressed(target, **payload)
        case["reference_file"] = target.name
        case["reference_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        print("captured", case["name"], flush=True)
    (args.folder / "km1-cases.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")


# -------------------------------------------------------------- compare ----
def woof_outputs(arrays, meta, arm):
    """The production mixing package on the fixture state; every output word."""
    import cupy as cp
    from gpuwm.config import RunConfig
    import gpuwm.core.dycore as dycore
    _name, khdif, kvdif, pbl, isfflx, cd0, heat = arm
    nz = int(meta["nz"])
    dev = {k: cp.asarray(v, dtype=cp.float32) for k, v in arrays.items()}
    cache = {}

    def scratch(shape, slot):
        key = (tuple(shape), slot)
        if key not in cache:
            cache[key] = cp.zeros(shape, dtype=cp.float32)
        return cache[key]
    fields = {"ustm": dev["ust"], "hfx": dev["hfx"], "qfx": dev["qfx"]}
    state = SimpleNamespace(
        **{k: dev[k] for k in ("u", "v", "w", "php", "phb", "alt", "p", "thp", "msft", "msfu", "msfv",
                               "fnm", "fnp", "dn", "dnw", "c1h", "c2h", "c1f", "c2f", "mub2d") + MOIST},
        thb=cp.full(nz, 300.0, dtype=cp.float32), mup=dev["mut"] - dev["mub2d"],
        mup0=dev["mut"] - dev["mub2d"], cf1=meta["cf1"], cf2=meta["cf2"], cf3=meta["cf3"],
        ru_t=cp.zeros_like(dev["u"]), rv_t=cp.zeros_like(dev["v"]), rw_t=cp.zeros_like(dev["w"]),
        rth_t=cp.zeros_like(dev["thp"]), tke=None, chem=None, has_msf=True,
        physics=SimpleNamespace(fields=fields), scratch=scratch)
    cfg = RunConfig(nx=int(meta["nx"]), ny=int(meta["ny"]), nz=nz, dx=meta["dx"], dy=meta["dy"],
                    dt=meta["dt"], ztop=20000., run_seconds=0., km_opt=1, diff_opt=2,
                    khdif=khdif, kvdif=kvdif, bl_pbl_physics=pbl, isfflx=isfflx,
                    sf_sfclay_physics=1 if isfflx in (1, 2) else 0,
                    tke_drag_coefficient=cd0, tke_heat_flux=heat,
                    open_x=bool(meta["bx"]), open_y=bool(meta["by"]), moist=True)
    assert dycore.wrf_mixing_package_active(cfg)
    km = scratch((nz, int(meta["ny"]), int(meta["nx"])), "smag_km")
    kh = scratch((nz, int(meta["ny"]), int(meta["nx"])), "smag_kh")
    specs = dycore._smag2d_specs(state, km, kh, kmv=dycore._horizontal_w_km(state, cfg))
    for f0, _t, _x, _c1, _c2, slot, _s in specs:
        scratch(f0.shape, slot)[...] = 0
    dycore._compute_wrf_smag_tendencies(state, cfg, km, kh, specs, time_t=False)
    got = {}
    for f0, _t, _x, _c1, _c2, slot, _s in specs:
        name = {"smag_ru": "u", "smag_rv": "v", "smag_rw": "w", "smag_rth": "theta"}.get(slot, slot[6:])
        got[name] = cp.asnumpy(scratch(f0.shape, slot))
    got["hfx_after"] = cp.asnumpy(fields["hfx"])
    got["coef_kmh"] = cp.asnumpy(km)
    got["coef_khh"] = cp.asnumpy(kh)
    got["coef_kmv"] = cp.asnumpy(scratch(km.shape, "smag_kmv"))
    got["coef_khv"] = cp.asnumpy(scratch(km.shape, "smag_khv"))
    return got


def interior_mask(shape, key, meta):
    """Words outside the two places the engine's storage conventions apply.

    Open boundaries: the mixing package clears a width-1 strip of every
    dry and moist row after the vertical pass (dycore._zero_open_strips),
    while WRF's vertical_diffusion_2 surface-flux loops run i = its ..
    min(ite, ide-1) with no open-boundary trim.  Periodic: the redundant
    u column ide / v row jde is a halo copy in WRF and row/column 0 here.
    Both sets are measured and reported separately, never dropped.
    """
    mask = np.ones(shape, bool)
    stag = {"u": "x", "v": "y", "w": "z"}.get(key, "")
    if meta["bx"]:
        mask[..., 0] = False
        mask[..., -1] = False
    elif stag == "x":
        mask[..., -1] = False
    if meta["by"]:
        mask[..., 0, :] = False
        mask[..., -1, :] = False
    elif stag == "y":
        mask[..., -1, :] = False
    return mask


def _accumulate(summary, key, m):
    agg = summary.setdefault(key, {"words": 0, "different_words": 0, "max_ulp": 0, "max_absolute": 0.0})
    agg["words"] += m["words"]
    agg["different_words"] += m["different_words"]
    agg["max_ulp"] = max(agg["max_ulp"], m["max_ulp"])
    agg["max_absolute"] = max(agg["max_absolute"], m["max_absolute"])


def compare(args):
    import cupy as cp
    from gpuwm.core.kernels import module_source
    from gpuwm.verify.diffusion_oracle import word_comparison
    import gpuwm.wrf_exact as wrf_exact
    manifest = json.loads((args.folder / "km1-cases.json").read_text())
    rows, summary, interior = {}, {}, {}
    for case in manifest["cases"]:
        with np.load(args.folder / case["file"]) as data:
            arrays = {k[3:]: data[k] for k in data.files if k.startswith("in_")}
            meta = json.loads(str(data["meta_json"]))
        with np.load(args.folder / case["reference_file"]) as ref:
            reference = {k: ref[k] for k in ref.files}
        for arm in ARMS:
            got = woof_outputs(arrays, meta, arm)
            row = {}
            for key in OUTPUTS + ("coef_kmh", "coef_kmv", "coef_khh", "coef_khv"):
                want = reference[f"{arm[0]}__{key}"]
                m = word_comparison(got[key], want)
                bad = np.argwhere(got[key].view(np.uint32) != want.view(np.uint32))
                m["first_differences"] = [
                    {"index": [int(i) for i in idx], "gpu": f"0x{int(got[key][tuple(idx)].view(np.uint32)):08x}",
                     "wrf": f"0x{int(want[tuple(idx)].view(np.uint32)):08x}"} for idx in bad[:4]]
                m["gpu_sha256"] = hashlib.sha256(np.ascontiguousarray(got[key]).tobytes()).hexdigest()
                mask = interior_mask(want.shape, key, meta)
                m["interior"] = word_comparison(got[key][mask], want[mask])
                m["convention_words"] = word_comparison(got[key][~mask], want[~mask]) if (~mask).any() else None
                row[key] = m
                _accumulate(summary, key, m)
                _accumulate(interior, key, m["interior"])
            rows[f"{case['name']}::{arm[0]}"] = row
    columns = sum(int(c["meta"]["nx"]) * int(c["meta"]["ny"]) for c in manifest["cases"])
    receipt = {"schema": "km1-wrf461-compare-v1",
               "strict": wrf_exact.ENABLED, "strict_options": list(wrf_exact.STRICT_OPTIONS) if wrf_exact.ENABLED else [],
               "exact_diffusion_substage": wrf_exact.DIFFUSION_ENABLED,
               "device": str(cp.cuda.runtime.getDeviceProperties(0)["name"]),
               "cupy": cp.__version__,
               "smag2d_source_sha256": hashlib.sha256(module_source("smag2d").encode()).hexdigest(),
               "patches": len(manifest["cases"]), "columns": columns, "arms": [a[0] for a in ARMS],
               "case_arms": len(rows),
               "total_words": sum(v["words"] for v in summary.values()),
               "total_different_words": sum(v["different_words"] for v in summary.values()),
               "max_ulp": max(v["max_ulp"] for v in summary.values()),
               "interior_total_words": sum(v["words"] for v in interior.values()),
               "interior_different_words": sum(v["different_words"] for v in interior.values()),
               "interior_max_ulp": max(v["max_ulp"] for v in interior.values()),
               "summary": summary, "interior_summary": interior, "rows": rows}
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: receipt[k] for k in ("strict", "exact_diffusion_substage", "device", "columns",
                                               "case_arms", "total_words", "total_different_words", "max_ulp",
                                               "interior_total_words", "interior_different_words",
                                               "interior_max_ulp")}))
    for key in summary:
        print(key, "all", summary[key], "interior", interior[key])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    c = sub.add_parser("cases")
    c.add_argument("wrfinput", type=Path); c.add_argument("reader", type=Path)
    c.add_argument("work", type=Path); c.add_argument("output", type=Path)
    cap = sub.add_parser("capture")
    cap.add_argument("folder", type=Path); cap.add_argument("library", type=Path)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("folder", type=Path); cmp_.add_argument("receipt", type=Path)
    a = p.parse_args()
    {"cases": make_cases, "capture": capture, "compare": compare}[a.action](a)
