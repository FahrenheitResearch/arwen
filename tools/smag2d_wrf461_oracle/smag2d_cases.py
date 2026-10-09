"""Column-oracle input cases for WRF km_opt=4 / diff_opt=2.

Real cases are windows of WRF v4.6.1 history files decoded by the engine's
Rust reader (rw_netcdf dump --raw).  NumPy only selects windows, derives the
inverse density from WRF's own equation of state and applies the named edge
transformations below.  Every array is float32 in the engine's (k, j, i)
layout with the real C-grid staggering.

Inputs WRF and the engine both receive, word for word:
  u v w php phb t2 alt p pb qv qc qr qi qs qg ni nr msft msfu msfv mut
  fnm fnp dn dnw znw c1h c2h c1f c2f cf1 cf2 cf3 p_top dx dy
t2 is WRF's grid%t_2 = theta - 300 K.  It is first rounded through
fl(fl(300 + T) - 300), which is exact (Sterbenz), so 300 + t2 is
representable and the engine's thp + thb - 300 reconstruction with
thb = 300 returns the same word under every arithmetic mode.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

F32 = np.float32
VARIABLES = ("U V W PH PHB T THM P PB MU MUB QVAPOR QCLOUD QRAIN QICE QSNOW QGRAUP "
             "QNICE QNRAIN MAPFAC_M MAPFAC_U MAPFAC_V FNM FNP DN DNW ZNW C1H C2H "
             "C1F C2F CF1 CF2 CF3 P_TOP HGT XLAT").split()
SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg")
NUMBERS = ("ni", "nr")
WRF_NAME = {"qv": "QVAPOR", "qc": "QCLOUD", "qr": "QRAIN", "qi": "QICE", "qs": "QSNOW",
            "qg": "QGRAUP", "ni": "QNICE", "nr": "QNRAIN"}
# WRF share/module_model_constants.F
R_D, P1000, CP = 287.0, 1.0e5, 7.0 * 287.0 / 2.0
CVPM = -(CP - R_D) / CP


def decode(reader: Path, source: Path, out: Path, time_index: int = -1):
    """All VARIABLES at one time, through the Rust reader."""
    out.mkdir(parents=True, exist_ok=True)
    inventory = json.loads(subprocess.check_output([str(reader), "inventory", str(source)]))
    present = {v["name"] for v in inventory["variables"]}
    names = [v for v in VARIABLES if v in present]
    subprocess.run([str(reader), "dump", "--raw", str(source), str(out), *names], check=True,
                   stdout=subprocess.DEVNULL)
    meta = json.loads((out / "metadata.json").read_text())
    data = {v["name"]: np.fromfile(out / v["filename"], "<f8").reshape(v["shape"])[time_index]
            for v in meta["variables"]}
    attrs = inventory["global_attributes"]
    if isinstance(attrs, list):
        attrs = {a["name"]: a["value"] for a in attrs}
    info = {"source": source.name, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "reader_sha256": hashlib.sha256(reader.read_bytes()).hexdigest(),
            "dx": float(attrs["DX"]), "dy": float(attrs["DY"]), "time_index": time_index,
            "absent_in_source": sorted(set(VARIABLES) - present)}
    return data, info


def window(raw, *, x, y, nx, ny):
    """One window of every field with its own staggering, all levels kept."""
    zeros3 = np.zeros_like(raw["T"])
    def mass(name):
        return F32(raw.get(name, zeros3)[:, y:y + ny, x:x + nx])
    a = {"u": F32(raw["U"][:, y:y + ny, x:x + nx + 1]),
         "v": F32(raw["V"][:, y:y + ny + 1, x:x + nx]),
         "w": F32(raw["W"][:, y:y + ny, x:x + nx]),
         "php": F32(raw["PH"][:, y:y + ny, x:x + nx]),
         "phb": F32(raw["PHB"][:, y:y + ny, x:x + nx]),
         "p": mass("P"), "pb": mass("PB"),
         "msft": F32(raw["MAPFAC_M"][y:y + ny, x:x + nx]),
         "msfu": F32(raw["MAPFAC_U"][y:y + ny, x:x + nx + 1]),
         "msfv": F32(raw["MAPFAC_V"][y:y + ny + 1, x:x + nx]),
         "mut": F32(raw["MU"][y:y + ny, x:x + nx] + raw["MUB"][y:y + ny, x:x + nx]),
         "hgt": F32(raw["HGT"][y:y + ny, x:x + nx]),
         "lat": F32(raw["XLAT"][y:y + ny, x:x + nx])}
    for name in SPECIES + NUMBERS:
        a[name] = mass(WRF_NAME[name])
    a["qv"] = np.maximum(a["qv"], F32(0))
    theta = F32(F32(300) + mass("T"))
    a["t2"] = F32(theta - F32(300))
    thm = raw.get("THM")
    theta_m = (np.float64(theta) if thm is None else thm[:, y:y + ny, x:x + nx] + 300.0)
    pressure = np.float64(a["p"]) + np.float64(a["pb"])
    # WRF's dry inverse density from its moist-theta equation of state.
    a["alt"] = F32((R_D / P1000) * theta_m * (pressure / P1000) ** CVPM)
    for name in ("fnm", "fnp", "dn", "dnw", "znw", "c1h", "c2h", "c1f", "c2f"):
        a[name] = F32(raw[name.upper()])
    return a


def scalars(raw):
    return {k.lower(): float(F32(raw[k])) for k in ("CF1", "CF2", "CF3", "P_TOP")}


def periodic(a):
    """Make the redundant periodic faces equal, as WRF's periodic halo does."""
    a["u"][:, :, -1] = a["u"][:, :, 0]
    a["msfu"][:, -1] = a["msfu"][:, 0]
    a["v"][:, -1, :] = a["v"][:, 0, :]
    a["msfv"][-1, :] = a["msfv"][0, :]
    return a


def edge_cases(base, nx, ny):
    """Named deterministic transformations of one real window."""
    iy, ix = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    nzw = base["phb"].shape[0]
    taper = np.linspace(1.0, 0.0, nzw, dtype=F32)[:, None, None]
    out = {}
    a = {k: v.copy() for k, v in base.items()}
    for key in ("u", "v", "w"):
        a[key] = np.zeros_like(a[key])
    for key in SPECIES + NUMBERS:
        a[key] = np.zeros_like(a[key])
    out["zero_flow"] = (a, "u=v=w=0 and every water species 0 (positive zeros)")
    a = {k: v.copy() for k, v in base.items()}
    for key in ("u", "v", "w"):
        a[key] = F32(a[key] * F32(0))
    out["signed_zero_flow"] = (a, "winds multiplied by 0, so westerly/southerly negatives become -0")
    a = {k: v.copy() for k, v in base.items()}
    for key in ("u", "v", "w"):
        a[key] = F32(a[key] * F32(1e-20))
    out["near_zero_flow"] = (a, "winds times 1e-20: strain products underflow into subnormals")
    a = {k: v.copy() for k, v in base.items()}
    for key in ("msft", "msfu", "msfv"):
        jy, jx = np.meshgrid(np.arange(a[key].shape[0]), np.arange(a[key].shape[1]), indexing="ij")
        a[key] = F32(0.25 + 3.75 * (0.5 + 0.5 * np.sin(2 * np.pi * jx / nx) * np.cos(2 * np.pi * jy / ny)))
    out["map_extremes"] = (a, "map factors swept 0.25..4.0")
    a = {k: v.copy() for k, v in base.items()}
    terrain = F32(1500.0 * np.sin(2 * np.pi * ix / nx * 1.5) * np.cos(2 * np.pi * iy / ny))
    a["phb"] = F32(a["phb"] + F32(9.81) * taper * terrain[None])
    out["steep_synthetic"] = (a, "1500 m sinusoidal terrain added to PHB, tapered to the top: slope limiter alpha >> 1")
    a = {k: v.copy() for k, v in base.items()}
    for key in ("u", "v"):
        a[key] = F32(a[key] * F32(25))
    a["w"] = F32(a["w"] * F32(25))
    out["strong_shear_cap"] = (a, "winds times 25: strain above def_limit and K at the 10*mlen cap")
    a = {k: v.copy() for k, v in base.items()}
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    r2 = ((ix - cx) ** 2 + (iy - cy) ** 2) / 4.0
    nzm = a["u"].shape[0]
    height = np.sin(np.pi * np.arange(nzw) / (nzw - 1))[:, None, None]
    a["w"] = F32(a["w"] + 28.0 * height * np.exp(-r2)[None])
    level = np.arange(nzm)[:, None, None]
    shape = np.where(level < nzm // 3, -1.0, np.where(level > 2 * nzm // 3, 1.0, 0.0))
    ux = np.arange(nx + 1)[None, None, :] - (cx + 0.5)
    vy = np.arange(ny + 1)[None, :, None] - (cy + 0.5)
    a["u"] = F32(a["u"] + 6.0 * shape * np.tanh(ux / 2.0) * np.exp(-((np.arange(ny)[None, :, None] - cy) ** 2) / 8.0))
    a["v"] = F32(a["v"] + 6.0 * shape * np.tanh(vy / 2.0) * np.exp(-((np.arange(nx)[None, None, :] - cx) ** 2) / 8.0))
    core = np.exp(-r2)[None]
    for key, peak in (("qc", 2.5e-3), ("qr", 4e-3), ("qi", 8e-4), ("qs", 2e-3), ("qg", 6e-3)):
        a[key] = F32(a[key] + peak * core * height[:-1])
    a["ni"] = F32(a["ni"] + 2.0e5 * core * height[:-1])
    a["nr"] = F32(a["nr"] + 5.0e4 * core * height[:-1])
    out["updraft_core"] = (a, "synthetic 28 m/s updraft core with low-level convergence, upper divergence and heavy hydrometeors")
    a = {k: v.copy() for k, v in base.items()}
    a = {k: (v[..., ::-1, :].copy() if v.ndim >= 2 else v.copy()) for k, v in a.items()}
    a["v"] = F32(-a["v"])
    a["lat"] = F32(-a["lat"])
    out["southern_mirror"] = (a, "window mirrored north-south with v negated: a southern-hemisphere image")
    return out


def save_case(folder: Path, name: str, arrays, meta):
    payload = {"input__" + k: np.ascontiguousarray(v) for k, v in arrays.items()}
    payload["meta_json"] = np.array(json.dumps(meta, sort_keys=True))
    path = folder / f"case-{name}.npz"
    np.savez_compressed(path, **payload)
    return path
