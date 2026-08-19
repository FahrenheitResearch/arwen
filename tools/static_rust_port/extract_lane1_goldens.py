"""Extract lane-1 golden cases by RUNNING the real Python implementation.

Writes bit-exact binary arrays + a JSON manifest under
``tools/rustwx/crates/static-fields/tests/goldens/lane1/`` for the Rust
port's unit tests (byte parity is the contract on the WPS path; see
docs/dev/static-rust-port.md section 3).

Everything here calls the actual product code -- ``LambertGrid``,
``MercatorGrid``, ``PolarStereoGrid``, ``_wps32_for``/``_DomainSampler``
(the float32 sampling twins and their ULP band logic), the corridor
geometry/probe/crop functions and ``_write_deterministic_npz`` -- never a
re-implementation.

Run from the repo root:  python tools/static_rust_port/extract_lane1_goldens.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from gpuwm.static.build import _DomainSampler, _wps32_for  # noqa: E402
from gpuwm.static.corridor import (  # noqa: E402
    ChildStaticsCorridor, _write_deterministic_npz, corridor_cost,
    corridor_geometry, corridor_grid, grid_identity_probes)
from gpuwm.static.lambert import LambertGrid  # noqa: E402
from gpuwm.static.projection import (  # noqa: E402
    MercatorGrid, PolarStereoGrid)

OUT = ROOT / "tools" / "rustwx" / "crates" / "static-fields" / "tests" \
    / "goldens" / "lane1"
OUT.mkdir(parents=True, exist_ok=True)

MANIFEST: dict = {
    "generator": "tools/static_rust_port/extract_lane1_goldens.py",
    "numpy": np.__version__,
    "cases": {},
}

DTYPES = {
    np.dtype(np.float64): "f64",
    np.dtype(np.float32): "f32",
    np.dtype(np.uint8): "u8",
    np.dtype(np.bool_): "u8",
}


def case(name: str) -> dict:
    return MANIFEST["cases"].setdefault(name, {"arrays": {}, "scalars": {}})


def put_array(case_name: str, key: str, value: np.ndarray) -> None:
    value = np.asarray(value)
    if value.dtype == np.bool_:
        value = value.astype(np.uint8)
    dtype = DTYPES[value.dtype]
    filename = f"{case_name}.{key}.{dtype}"
    raw = np.ascontiguousarray(value).tobytes()
    (OUT / filename).write_bytes(raw)
    case(case_name)["arrays"][key] = {
        "file": filename,
        "dtype": dtype,
        "shape": list(value.shape),
    }


def put_scalar(case_name: str, key: str, value) -> None:
    """Store a scalar's exact bit pattern (f64 -> u64 hex, f32 -> u32)."""
    if isinstance(value, (np.float32,)):
        bits = int(np.float32(value).view(np.uint32))
        case(case_name)["scalars"][key] = {"f32_bits": f"{bits:08x}"}
    else:
        bits = int(np.float64(value).view(np.uint64))
        case(case_name)["scalars"][key] = {"f64_bits": f"{bits:016x}"}


def put_json(case_name: str, key: str, value) -> None:
    case(case_name)[key] = value


# ---------------------------------------------------------------------------
# Grids (the parity harness's Lambert pair + smoke domains + SH branches)
# ---------------------------------------------------------------------------

PARENT_SPEC = dict(
    kind="lambert", ref_lat=39.5, ref_lon=-84.0,
    truelat1=38.0, truelat2=41.0, stand_lon=-84.0,
    dx=9000.0, dy=9000.0, e_we=52, e_sn=45,
    known_x=26.0, known_y=22.5,
    moad_cen_lat=39.5, moad_cen_lon=-84.0)

lam_parent = LambertGrid(
    PARENT_SPEC["ref_lat"], PARENT_SPEC["ref_lon"], PARENT_SPEC["truelat1"],
    PARENT_SPEC["truelat2"], PARENT_SPEC["stand_lon"], PARENT_SPEC["dx"],
    PARENT_SPEC["dy"], PARENT_SPEC["e_we"], PARENT_SPEC["e_sn"],
    known_x=PARENT_SPEC["known_x"], known_y=PARENT_SPEC["known_y"])

lam_d02 = lam_parent.nest(18, 15, 3, 46, 40)
lam_d03 = lam_d02.nest(10, 8, 5, 26, 23)   # 600 m: sub-km twin paths

lam_sh = LambertGrid(-36.5, 145.0, -34.0, -38.0, 145.0,
                     12000.0, 12000.0, 40, 34)

merc = MercatorGrid(21.5, 121.0, 20.0, 20.0, 121.0, 12000.0, 12000.0, 42, 36)
merc_subkm = MercatorGrid(21.5, 121.0, 20.0, 20.0, 121.0, 800.0, 800.0,
                          30, 26)
polar = PolarStereoGrid(64.0, -152.0, 71.0, 71.0, -150.0, 15000.0, 15000.0,
                        40, 34)
polar_sh = PolarStereoGrid(-64.0, 160.0, -71.0, -71.0, 165.0,
                           15000.0, 15000.0, 36, 30)
polar_subkm = PolarStereoGrid(64.0, -152.0, 71.0, 71.0, -150.0, 900.0, 900.0,
                              28, 24)

GRIDS = {
    "lam_parent": lam_parent,
    "lam_d02": lam_d02,
    "lam_d03": lam_d03,
    "lam_sh": lam_sh,
    "merc": merc,
    "merc_subkm": merc_subkm,
    "polar": polar,
    "polar_sh": polar_sh,
    "polar_subkm": polar_subkm,
}

SPECS = {
    "lam_parent": PARENT_SPEC,
    "lam_sh": dict(kind="lambert", ref_lat=-36.5, ref_lon=145.0,
                   truelat1=-34.0, truelat2=-38.0, stand_lon=145.0,
                   dx=12000.0, dy=12000.0, e_we=40, e_sn=34,
                   known_x=20.0, known_y=17.0,
                   moad_cen_lat=-36.5, moad_cen_lon=145.0),
    "merc": dict(kind="mercator", ref_lat=21.5, ref_lon=121.0,
                 truelat1=20.0, truelat2=20.0, stand_lon=121.0,
                 dx=12000.0, dy=12000.0, e_we=42, e_sn=36,
                 known_x=21.0, known_y=18.0,
                 moad_cen_lat=21.5, moad_cen_lon=121.0),
    "merc_subkm": dict(kind="mercator", ref_lat=21.5, ref_lon=121.0,
                       truelat1=20.0, truelat2=20.0, stand_lon=121.0,
                       dx=800.0, dy=800.0, e_we=30, e_sn=26,
                       known_x=15.0, known_y=13.0,
                       moad_cen_lat=21.5, moad_cen_lon=121.0),
    "polar": dict(kind="polar", ref_lat=64.0, ref_lon=-152.0,
                  truelat1=71.0, truelat2=71.0, stand_lon=-150.0,
                  dx=15000.0, dy=15000.0, e_we=40, e_sn=34,
                  known_x=20.0, known_y=17.0,
                  moad_cen_lat=64.0, moad_cen_lon=-152.0),
    "polar_sh": dict(kind="polar", ref_lat=-64.0, ref_lon=160.0,
                     truelat1=-71.0, truelat2=-71.0, stand_lon=165.0,
                     dx=15000.0, dy=15000.0, e_we=36, e_sn=30,
                     known_x=18.0, known_y=15.0,
                     moad_cen_lat=-64.0, moad_cen_lon=160.0),
    "polar_subkm": dict(kind="polar", ref_lat=64.0, ref_lon=-152.0,
                        truelat1=71.0, truelat2=71.0, stand_lon=-150.0,
                        dx=900.0, dy=900.0, e_we=28, e_sn=24,
                        known_x=14.0, known_y=12.0,
                        moad_cen_lat=64.0, moad_cen_lon=-152.0),
}

NEST_CHAIN = {
    "lam_d02": {"parent": "lam_parent", "i_parent_start": 18,
                "j_parent_start": 15, "parent_grid_ratio": 3,
                "e_we": 46, "e_sn": 40},
    "lam_d03": {"parent": "lam_d02", "i_parent_start": 10,
                "j_parent_start": 8, "parent_grid_ratio": 5,
                "e_we": 26, "e_sn": 23},
}


def dump_state(name: str, grid) -> None:
    put_scalar(name, "hemi", grid.hemi)
    put_scalar(name, "cen_lat", grid.cen_lat)
    put_scalar(name, "cen_lon", grid.cen_lon)
    put_scalar(name, "known_x", grid.known_x)
    put_scalar(name, "known_y", grid.known_y)
    put_scalar(name, "ref_lat", grid.ref_lat)
    put_scalar(name, "ref_lon", grid.ref_lon)
    put_scalar(name, "dx", grid.dx)
    if isinstance(grid, LambertGrid):
        for key in ("cone", "rebydx", "rsw", "polei", "polej"):
            put_scalar(name, key, getattr(grid, key))
    elif isinstance(grid, MercatorGrid):
        for key in ("dlon", "rsw"):
            put_scalar(name, key, getattr(grid, key))
    elif isinstance(grid, PolarStereoGrid):
        for key in ("rebydx", "rsw", "polei", "polej"):
            put_scalar(name, key, getattr(grid, key))


def dump_grid(name: str, grid, *, staggers=("mass", "u", "v", "c"),
              derived=True) -> None:
    dump_state(name, grid)
    latlon = {"mass": grid.latlon_mass, "u": grid.latlon_u,
              "v": grid.latlon_v, "c": grid.latlon_c}
    for stagger in staggers:
        lat, lon = latlon[stagger]()
        put_array(name, f"lat_{stagger}", lat)
        put_array(name, f"lon_{stagger}", lon)
    if derived:
        put_array(name, "mapfac_m", grid.mapfac_m())
        put_array(name, "mapfac_u", grid.mapfac_u())
        put_array(name, "mapfac_v", grid.mapfac_v())
        f, e = grid.coriolis_m()
        put_array(name, "coriolis_f", f)
        put_array(name, "coriolis_e", e)
        sin_m, cos_m = grid.rotation_m()
        put_array(name, "sinalpha_m", sin_m)
        put_array(name, "cosalpha_m", cos_m)
        sin_u, cos_u = grid.rotation_u()
        put_array(name, "sinalpha_u", sin_u)
        put_array(name, "cosalpha_u", cos_u)
        sin_v, cos_v = grid.rotation_v()
        put_array(name, "sinalpha_v", sin_v)
        put_array(name, "cosalpha_v", cos_v)
    # float64 inverse transform, driven with the grid's own mass latlon
    lat, lon = grid.latlon_mass()
    x, y = grid.latlon_to_ij(lat, lon)
    put_array(name, "llij_x", x)
    put_array(name, "llij_y", y)


for grid_name, grid_obj in GRIDS.items():
    if grid_name in NEST_CHAIN:
        dump_grid(grid_name, grid_obj,
                  staggers=("mass", "u"), derived=(grid_name == "lam_d03"))
        put_json(grid_name, "nest", NEST_CHAIN[grid_name])
    else:
        dump_grid(grid_name, grid_obj)
    if grid_name in SPECS:
        put_json(grid_name, "spec", SPECS[grid_name])

# ---------------------------------------------------------------------------
# Translated grids: delegation, re-extent, composition
# ---------------------------------------------------------------------------

tr = lam_parent.translated(3, -2)
lat, lon = tr.latlon_mass()
put_array("translated", "lat_mass", lat)
put_array("translated", "lon_mass", lon)
put_scalar("translated", "cen_lat", tr.cen_lat)
put_scalar("translated", "cen_lon", tr.cen_lon)
x, y = tr.latlon_to_ij(lat, lon)
put_array("translated", "llij_x", x)
put_array("translated", "llij_y", y)
put_json("translated", "offset", [3, -2])

tr2 = tr.translated(-1, 4, e_we=20, e_sn=18)
lat2, lon2 = tr2.latlon_mass()
put_array("translated", "compose_lat_mass", lat2)
put_array("translated", "compose_lon_mass", lon2)
put_json("translated", "compose_offset", [-1, 4])
put_json("translated", "compose_extent", [20, 18])

# ---------------------------------------------------------------------------
# Corridor geometry / identity probes / cost / crop
# ---------------------------------------------------------------------------


class _Run:
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny


class _ChildDc:
    def __init__(self):
        self.grid_id = 2
        self.parent_id = 1
        self.parent_grid_ratio = 3
        self.i_parent_start = 18
        self.j_parent_start = 15
        self.run = _Run(45, 39)


child_dc = _ChildDc()
parent_run = _Run(51, 44)
geometry = corridor_geometry(child_dc, parent_run)
put_json("corridor", "geometry", geometry)
put_json("corridor", "cost", corridor_cost(child_dc, parent_run))

cgrid = corridor_grid(lam_d02, geometry)
probes = grid_identity_probes(cgrid)
put_json("corridor", "probe_order", list(probes))
for point, (plat, plon) in probes.items():
    put_scalar("corridor", f"probe_{point}_lat", plat)
    put_scalar("corridor", f"probe_{point}_lon", plon)


# A second, small geometry keeps the crop fixture arrays lean; the slice
# arithmetic is identical at any size.
class _SmallChildDc(_ChildDc):
    def __init__(self):
        super().__init__()
        self.parent_grid_ratio = 3
        self.i_parent_start = 4
        self.j_parent_start = 3
        self.run = _Run(12, 10)


small_dc = _SmallChildDc()
small_parent = _Run(9, 8)
small_geometry = corridor_geometry(small_dc, small_parent)
put_json("corridor", "crop_geometry", small_geometry)

ny = int(small_geometry["corridor_ny"])
nx = int(small_geometry["corridor_nx"])
rng = np.random.default_rng(20260817)
fields = {
    "PLANE_A": rng.normal(300.0, 40.0, (ny, nx)),
    "PLANE_B": rng.uniform(-1.0, 1.0, (ny, nx)),
    "STACK_C": rng.uniform(0.0, 1.0, (5, ny, nx)),
}
corridor_obj = ChildStaticsCorridor(geometry=small_geometry, fields=fields,
                                    cache_sha256="0" * 64)
crop = corridor_obj.crop(5, 4)
for fname, cropped in crop.items():
    put_array("corridor", f"crop_{fname}", cropped)
put_json("corridor", "crop_placement", [5, 4])
for fname, full in fields.items():
    put_array("corridor", f"full_{fname}", full)

# ---------------------------------------------------------------------------
# Deterministic NPZ seal (the file bytes ARE the contract)
# ---------------------------------------------------------------------------

npz_fields = {
    "ALPHA": np.array([[1.0, -2.5, 0.0, -0.0, 1e-308],
                       [3.25, 300.125, -1e30, 2.0, 0.5]]),
    "BETA": np.arange(24, dtype=np.float64).reshape(2, 3, 4) * 0.125 - 1.0,
    "GAMMA": np.array([[1e16, -1e-16], [123456.789, -0.001]]),
}
npz_path = OUT / "npz_seal.golden.npz"
if npz_path.exists():
    npz_path.unlink()
_write_deterministic_npz(npz_path, npz_fields)
digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
put_json("npz", "file", npz_path.name)
put_json("npz", "sha256", digest)
put_json("npz", "bytes", npz_path.stat().st_size)
put_json("npz", "fields", {
    name: list(np.asarray(value).shape) for name, value in npz_fields.items()
})
for fname, value in npz_fields.items():
    put_array("npz", f"field_{fname}", np.asarray(value, dtype=np.float64))

# ---------------------------------------------------------------------------
# float32 WPS twins + sampling surfaces (the ULP logic, via _DomainSampler)
# ---------------------------------------------------------------------------


def twin_state(twin) -> dict:
    out = {}
    for key, value in vars(twin).items():
        if isinstance(value, np.float32):
            out[key] = f"{int(value.view(np.uint32)):08x}"
    return out


def dump_twin(name: str, grid, adopt: bool) -> None:
    twin = _wps32_for(grid)
    put_json(name, "twin_state", twin_state(twin))
    if adopt:
        twin.adopt_public_pole(grid)
        put_json(name, "twin_state_adopted", twin_state(twin))
    nx = grid.e_we - 1
    ny = grid.e_sn - 1
    halo = 3
    xs = np.arange(1 - halo, nx + halo + 1, dtype=np.float64)
    ys = np.arange(1 - halo, ny + halo + 1, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    lat32, lon32 = twin.ij_to_latlon(X, Y)
    put_array(name, "twin_lat", lat32)
    put_array(name, "twin_lon", lon32)
    x32, y32 = twin.latlon_to_ij(lat32, lon32)
    put_array(name, "twin_llij_x", x32)
    put_array(name, "twin_llij_y", y32)


dump_twin("lam_parent", lam_parent, adopt=False)
dump_twin("lam_d03", lam_d03, adopt=True)
dump_twin("lam_sh", lam_sh, adopt=False)
dump_twin("merc", merc, adopt=False)
dump_twin("merc_subkm", merc_subkm, adopt=True)
dump_twin("polar", polar, adopt=False)
dump_twin("polar_sh", polar_sh, adopt=False)
dump_twin("polar_subkm", polar_subkm, adopt=True)

# translated twin: delegation through the reference twin
ttwin = _wps32_for(tr)
nx = tr.e_we - 1
ny = tr.e_sn - 1
xs = np.arange(1 - 3, nx + 3 + 1, dtype=np.float64)
ys = np.arange(1 - 3, ny + 3 + 1, dtype=np.float64)
X, Y = np.meshgrid(xs, ys)
tlat, tlon = ttwin.ij_to_latlon(X, Y)
put_array("translated", "twin_lat", tlat)
put_array("translated", "twin_lon", tlon)
tx, ty = ttwin.latlon_to_ij(tlat, tlon)
put_array("translated", "twin_llij_x", tx)
put_array("translated", "twin_llij_y", ty)


def dump_surface(name: str, grid) -> None:
    sampler = _DomainSampler(grid, halo=3)
    put_array(name, "surface_lat_e", sampler.lat_e)
    put_array(name, "surface_lat_lower_e", sampler._lat_lower_e)
    lon_e = np.asarray(sampler.lon_e)
    put_array(name, "surface_lon_e", lon_e)
    put_json(name, "surface_lon_e_dtype", str(lon_e.dtype))
    put_array(name, "surface_lon_boundary_band", sampler._lon_boundary_band)
    put_array(name, "surface_lat_integer_band", sampler._lat_integer_band)
    put_array(name, "surface_lat_c", sampler.lat_c)
    put_array(name, "surface_lon_c", sampler.lon_c)
    print(f"{name}: lon_boundary_band "
          f"{int(np.count_nonzero(sampler._lon_boundary_band))}, "
          f"lat_integer_band "
          f"{int(np.count_nonzero(sampler._lat_integer_band))} "
          f"of {sampler._lon_boundary_band.size}")


dump_surface("lam_parent", lam_parent)
dump_surface("lam_d03", lam_d03)
dump_surface("merc", merc)
dump_surface("merc_subkm", merc_subkm)
dump_surface("polar_subkm", polar_subkm)

# ---------------------------------------------------------------------------
# numpy f32 SIMD kernel micro-goldens (sin/cos/exp/log ports)
# ---------------------------------------------------------------------------

samples = np.concatenate([
    rng.uniform(-4.0, 4.0, 3000),
    rng.uniform(-0.1, 0.1, 500),
    rng.uniform(-720.0, 720.0, 500),
    np.array([0.0, -0.0, 1.0, -1.0, np.pi / 4, -np.pi / 4,
              117435.99, 117436.0, 71476.0, 71477.0, 1e-40, -1e-40]),
]).astype(np.float32)
put_array("npmath", "trig_in", samples)
put_array("npmath", "sin_out", np.sin(samples))
put_array("npmath", "cos_out", np.cos(samples))

exp_in = np.concatenate([
    rng.uniform(-10.0, 10.0, 3000),
    rng.uniform(-0.01, 0.01, 500),
    np.array([0.0, -0.0, 88.72283, 88.7229, -103.9720, -103.973,
              -87.0, 60.0, 1e-40]),
]).astype(np.float32)
put_array("npmath", "exp_in", exp_in)
put_array("npmath", "exp_out", np.exp(exp_in))

log_in = np.concatenate([
    rng.uniform(1e-6, 4.0, 3000),
    rng.uniform(0.5, 1.5, 500),
    np.array([1.0, 2.0, 0.5, np.float64(np.finfo(np.float32).tiny),
              1e-42, 1e30]),
]).astype(np.float32)
put_array("npmath", "log_in", log_in)
put_array("npmath", "log_out", np.log(log_in))

manifest_path = OUT / "manifest.json"
manifest_path.write_text(json.dumps(MANIFEST, indent=1, sort_keys=True)
                         + "\n", encoding="utf-8")
total = sum(p.stat().st_size for p in OUT.iterdir())
print(f"wrote {sum(1 for _ in OUT.iterdir())} files, {total/1024:.0f} KiB")
