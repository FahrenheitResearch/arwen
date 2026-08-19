"""Generate lane-2 parity goldens by RUNNING the real Python static
implementation (gpuwm.static.geog / gpuwm.static.build) on real and
synthetic source data.

The Rust unit tests in crates/static-fields consume the outputs under
golden/lane2/.  Regenerate with:

    python tools/rustwx/crates/static-fields/golden/gen_lane2_goldens.py

Array container ("GWARR1"): magic 8 bytes b"GWARR1\\x00\\x00", dtype u8
(0=f64, 1=f32, 2=i64, 3=u8/bool, 4=i64-RLE), ndim u8, dims as u64 LE,
payload LE.  RLE payload: (i64 value, u64 run) pairs.  f64/f32 values in
JSON are hex bit patterns for exactness.
"""
from __future__ import annotations

import json
import os
import shutil
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO))

from gpuwm.static.build import (  # noqa: E402
    _DomainSampler, GeogSelection, build_static, average_16pt, average_4pt,
    dominant_category, four_pt, landmask_from_landusef,
    lu_index_from_landusef, monthly_interp_to_date, one_two_one,
    search_nearest, sixteen_pt, smth_desmth, smth_desmth_special,
    _interp_seq,
)
from gpuwm.static.geog import GeogDataset, parse_index  # noqa: E402
from gpuwm.static.lambert import LambertGrid  # noqa: E402

OUT = HERE / "lane2"
#: Same env-var contract and same default as
#: tests/test_static_rust_parity.py and src/testsupport.rs, composed
#: from this account's home rather than written out: a spelled literal
#: is one developer's absolute path, and the release snapshot refuses to
#: ship a file carrying one.
GEOG_ROOT = Path(os.environ.get("GPUWM_STATIC_PARITY_GEOG")
                 or (Path.home() / "Downloads"
                     / "WRF_1974_MP55_reference_bundle" / "static"
                     / "WPS_GEOG"))


# --------------------------------------------------------------------------
# container helpers
# --------------------------------------------------------------------------

def write_arr(path: Path, a) -> str:
    a = np.asarray(a)
    if a.dtype == np.bool_:
        a = a.astype(np.uint8)
    codes = {np.dtype("f8"): 0, np.dtype("f4"): 1, np.dtype("i8"): 2,
             np.dtype("u1"): 3}
    code = codes[np.dtype(a.dtype.str.lstrip("<>=|"))]
    header = b"GWARR1\x00\x00" + bytes([code, a.ndim])
    header += b"".join(int(d).to_bytes(8, "little") for d in a.shape)
    payload = np.ascontiguousarray(a).astype(
        a.dtype.newbyteorder("<"), copy=False).tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + payload)
    return path.name


def write_rle_i64(path: Path, a) -> str:
    a = np.asarray(a, dtype=np.int64).ravel()
    header = b"GWARR1\x00\x00" + bytes([4, 1])
    header += int(a.size).to_bytes(8, "little")
    chunks = []
    if a.size:
        edges = np.flatnonzero(np.diff(a)) + 1
        starts = np.concatenate(([0], edges))
        ends = np.concatenate((edges, [a.size]))
        for s, e in zip(starts, ends):
            chunks.append(struct.pack("<qQ", int(a[s]), int(e - s)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + b"".join(chunks))
    return path.name


def hex64(v) -> str:
    return f"{np.float64(v).view(np.uint64):016x}"


def hex32(v) -> str:
    return f"{np.float32(v).view(np.uint32):08x}"


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=False))


def index_json(idx) -> dict:
    return {
        "type": idx.type,
        "projection": idx.projection,
        "dx": hex64(idx.dx), "dy": hex64(idx.dy),
        "known_x": hex64(idx.known_x), "known_y": hex64(idx.known_y),
        "known_lat": hex64(idx.known_lat),
        "known_lon": hex64(idx.known_lon),
        "wordsize": idx.wordsize, "tile_x": idx.tile_x,
        "tile_y": idx.tile_y, "tile_z_start": idx.tile_z_start,
        "tile_z_end": idx.tile_z_end, "tile_bdr": idx.tile_bdr,
        "signed": idx.signed, "endian_big": idx.endian == "big",
        "scale_factor": hex64(idx.scale_factor),
        "missing_value": None if idx.missing_value is None
        else hex64(idx.missing_value),
        "category_min": idx.category_min, "category_max": idx.category_max,
        "mminlu": idx.mminlu, "iswater": idx.iswater,
        "islake": idx.islake, "isice": idx.isice, "isurban": idx.isurban,
        "row_order_top_bottom": idx.row_order == "top_bottom",
        "interp_option": idx.interp_option,
    }


def inventory_json(ds: GeogDataset) -> dict:
    return {
        "declared_sparse": ds.declared_sparse,
        "tile_count": len(ds.tiles),
        "tile_inventory_bounds": list(ds.tile_inventory_bounds),
        "nx_global": ds.nx_global, "ny_global": ds.ny_global,
        "wraps_x": ds.wraps_x, "extent_basis": ds.extent_basis,
    }


# --------------------------------------------------------------------------
# synthetic datasets
# --------------------------------------------------------------------------

def make_ds(name: str, index_text: str, tiles: dict[str, np.ndarray]):
    root = OUT / "synthetic" / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    (root / "index").write_text(index_text)
    for fname, arr in tiles.items():
        (root / fname).write_bytes(arr.tobytes())
    return root


def gen_synthetic():
    rng = np.random.default_rng(42)
    exp_dir = OUT / "synthetic_expected"
    manifest = {}

    def record(name, ds, entry):
        entry["index"] = index_json(ds.index)
        entry["inventory"] = inventory_json(ds)
        manifest[name] = entry

    # -- syn_i2_big_bdr: signed big-endian i2, border 2, 2 z planes,
    #    scale + missing, regional (non-wrap) 16x12 world of 4 tiles.
    tx, ty, b, nz = 8, 6, 2, 2
    full = rng.integers(-3000, 3000, size=(nz, 12 + 2 * b, 16 + 2 * b))
    full[full % 17 == 0] = -999   # sprinkle missing
    tiles = {}
    for oy in (1, 7):
        for ox in (1, 9):
            sub = full[:, oy - 1:oy - 1 + ty + 2 * b,
                       ox - 1:ox - 1 + tx + 2 * b]
            tiles[f"{ox:05d}-{ox + tx - 1:05d}.{oy:05d}-{oy + ty - 1:05d}"] \
                = sub.astype(">i2")
    idx_text = (
        "type = continuous\nsigned = yes\nprojection = regular_ll\n"
        "dx = 0.5\ndy = 0.5\nknown_x = 1.0\nknown_y = 1.0\n"
        "known_lat = -89.75\nknown_lon = -179.75\nwordsize = 2\n"
        f"tile_x = {tx}\ntile_y = {ty}\ntile_z = {nz}\ntile_bdr={b}\n"
        "scale_factor = 0.5\nmissing_value = -999.\n"
        "units=\"meters\"\ndescription=\"synthetic i2\"\n")
    root = make_ds("syn_i2_big_bdr", idx_text, tiles)
    ds = GeogDataset(root)
    win = ds.read_window(3, 14, 2, 11)
    tilewin = ds.read_tile_window(9, 1)
    entry = {
        "window": {
            "args": [3, 14, 2, 11], "x0": win.x0, "y0": win.y0,
            "shape": list(win.raw.shape),
            "raw": write_arr(exp_dir / "syn_i2_big_bdr_w0.bin",
                             win.raw.astype(np.int64)),
            "coverage": write_arr(exp_dir / "syn_i2_big_bdr_w0_cov.bin",
                                  win.coverage),
            "values_z": 1,
            "values": write_arr(exp_dir / "syn_i2_big_bdr_w0_vals.bin",
                                win.values(1)),
        },
        "tile_window": {
            "args": [9, 1], "x0": tilewin.x0, "y0": tilewin.y0,
            "shape": list(tilewin.raw.shape),
            "raw": write_arr(exp_dir / "syn_i2_big_bdr_tile.bin",
                             tilewin.raw.astype(np.int64)),
        },
        "required_origins": [list(o)
                             for o in ds.required_tile_origins(3, 14, 2, 11)],
    }
    record("syn_i2_big_bdr", ds, entry)

    # -- syn_u1_topbot_negdy: u1, top_bottom rows, negative dy.
    tx, ty = 6, 4
    tiles = {}
    for oy in (1, 5):
        for ox in (1, 7):
            arr = rng.integers(0, 200, size=(1, ty, tx))
            arr[arr % 23 == 0] = 255
            tiles[f"{ox:05d}-{ox + tx - 1:05d}.{oy:05d}-{oy + ty - 1:05d}"] \
                = arr.astype("u1")
    idx_text = (
        "type=continuous\nprojection=regular_ll\ndx=0.05\ndy=-0.05\n"
        "known_x=1.0\nknown_y=1.0\nknown_lat=89.975\nknown_lon=-179.975\n"
        "wordsize=1\ntile_x=6\ntile_y=4\ntile_z=1\nmissing_value=255.\n"
        "row_order=top_bottom\nscale_factor=0.01\n")
    root = make_ds("syn_u1_topbot_negdy", idx_text, tiles)
    ds = GeogDataset(root)
    win = ds.read_window(1, 12, 1, 8)
    pts = [(89.975, -179.975), (89.7, -179.6), (89.9, -179.699)]
    entry = {
        "window": {
            "args": [1, 12, 1, 8], "x0": win.x0, "y0": win.y0,
            "shape": list(win.raw.shape),
            "raw": write_arr(exp_dir / "syn_u1_tb_w0.bin",
                             win.raw.astype(np.int64)),
            "coverage": write_arr(exp_dir / "syn_u1_tb_w0_cov.bin",
                                  win.coverage),
            "values_z": 0,
            "values": write_arr(exp_dir / "syn_u1_tb_w0_vals.bin",
                                win.values(0)),
        },
        "latlon_to_xy": [
            {"lat": hex64(lat), "lon": hex64(lon),
             "x": hex64(ds.latlon_to_xy(lat, lon)[0]),
             "y": hex64(ds.latlon_to_xy(lat, lon)[1])}
            for lat, lon in pts],
        "xy_to_latlon": [
            {"x": hex64(x), "y": hex64(y),
             "lat": hex64(ds.xy_to_latlon(x, y)[0]),
             "lon": hex64(ds.xy_to_latlon(x, y)[1])}
            for x, y in ((1.0, 1.0), (5.25, 7.75), (12.0, 3.5))],
    }
    record("syn_u1_topbot_negdy", ds, entry)

    # -- syn_wrap: global 12x6 world of 4x3 tiles, u1, wraps in x.
    tx, ty = 4, 3
    world = rng.integers(1, 90, size=(1, 6, 12)).astype("u1")
    tiles = {}
    for oy in (1, 4):
        for ox in (1, 5, 9):
            tiles[f"{ox:05d}-{ox + tx - 1:05d}.{oy:05d}-{oy + ty - 1:05d}"] \
                = np.ascontiguousarray(
                    world[:, oy - 1:oy - 1 + ty, ox - 1:ox - 1 + tx])
    idx_text = (
        "type=continuous\nprojection=regular_ll\ndx=30.0\ndy=30.0\n"
        "known_x=1.0\nknown_y=1.0\nknown_lat=-75.0\nknown_lon=-165.0\n"
        "wordsize=1\ntile_x=4\ntile_y=3\ntile_z=1\n")
    root = make_ds("syn_wrap", idx_text, tiles)
    ds = GeogDataset(root)
    win = ds.read_window(-2, 8, 0, 7)
    entry = {
        "window": {
            "args": [-2, 8, 0, 7], "x0": win.x0, "y0": win.y0,
            "shape": list(win.raw.shape),
            "raw": write_arr(exp_dir / "syn_wrap_w0.bin",
                             win.raw.astype(np.int64)),
            "coverage": write_arr(exp_dir / "syn_wrap_w0_cov.bin",
                                  win.coverage),
            "values_z": 0,
            "values": write_arr(exp_dir / "syn_wrap_w0_vals.bin",
                                win.values(0)),
        },
        "latlon_to_xy": [
            {"lat": hex64(lat), "lon": hex64(lon),
             "x": hex64(ds.latlon_to_xy(lat, lon)[0]),
             "y": hex64(ds.latlon_to_xy(lat, lon)[1])}
            for lat, lon in ((-75.0, -165.0), (10.0, 179.9),
                             (60.0, -179.99), (0.0, 15.0))],
        "required_origins": [list(o)
                             for o in ds.required_tile_origins(-2, 8, 0, 7)],
        "too_wide": {
            "args": [-2, 15, 1, 6],
            "contains": "window wider than the global grid",
        },
    }
    record("syn_wrap", ds, entry)

    # -- syn_sparse: same world, tile (5,1) absent, declared sparse.
    sparse_tiles = {k: v for k, v in tiles.items()
                    if not k.startswith("00005-")}
    root = make_ds("syn_sparse", idx_text + "sparse=yes\n", sparse_tiles)
    ds = GeogDataset(root)
    win = ds.read_window(1, 12, 1, 6)
    entry = {
        "window": {
            "args": [1, 12, 1, 6], "x0": win.x0, "y0": win.y0,
            "shape": list(win.raw.shape),
            "raw": write_arr(exp_dir / "syn_sparse_w0.bin",
                             win.raw.astype(np.int64)),
            "coverage": write_arr(exp_dir / "syn_sparse_w0_cov.bin",
                                  win.coverage),
            "values_z": 0,
            "values": write_arr(exp_dir / "syn_sparse_w0_vals.bin",
                                win.values(0)),
        },
        "missing_tiles": [list(o) for o in ds.missing_tiles(1, 12, 1, 6)],
    }
    record("syn_sparse", ds, entry)

    # -- syn_missing_nonsparse: absent tile without the declaration.
    root = make_ds("syn_missing_nonsparse", idx_text, sparse_tiles)
    ds = GeogDataset(root)
    try:
        ds.read_window(1, 12, 1, 6)
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError as err:
        message = str(err)
    entry = {
        "read_window_error": {
            "args": [1, 12, 1, 6],
            "contains": "expected tile origin (5, 1) is absent",
        },
        "python_message": message,
    }
    record("syn_missing_nonsparse", ds, entry)

    # -- syn_zpad: index declares 2 z planes, file carries 3 (third all
    #    zero) -> accepted; syn_zpad_bad: third plane nonzero -> refused.
    tx, ty = 5, 4
    good = rng.integers(0, 1000, size=(3, ty, tx)).astype(">u2")
    good[2] = 0
    bad = good.copy()
    bad[2, 1, 3] = 7
    zpad_index = (
        "type=continuous\nprojection=regular_ll\ndx=1.0\ndy=1.0\n"
        "known_x=1.0\nknown_y=1.0\nknown_lat=-89.5\nknown_lon=-179.5\n"
        "wordsize=2\ntile_x=5\ntile_y=4\ntile_z=2\n")
    root = make_ds("syn_zpad", zpad_index, {"00001-00005.00001-00004": good})
    ds = GeogDataset(root)
    win = ds.read_window(1, 5, 1, 4)
    entry = {
        "window": {
            "args": [1, 5, 1, 4], "x0": win.x0, "y0": win.y0,
            "shape": list(win.raw.shape),
            "raw": write_arr(exp_dir / "syn_zpad_w0.bin",
                             win.raw.astype(np.int64)),
            "coverage": write_arr(exp_dir / "syn_zpad_w0_cov.bin",
                                  win.coverage),
            "values_z": 1,
            "values": write_arr(exp_dir / "syn_zpad_w0_vals.bin",
                                win.values(1)),
        },
    }
    record("syn_zpad", ds, entry)

    root = make_ds("syn_zpad_bad", zpad_index,
                   {"00001-00005.00001-00004": bad})
    ds = GeogDataset(root)
    try:
        ds.read_window(1, 5, 1, 4)
        raise AssertionError("expected ValueError")
    except ValueError as err:
        message = str(err)
    entry = {
        "read_window_error": {
            "args": [1, 5, 1, 4],
            "contains": ("undeclared trailing planes contain nonzero data "
                         "(first at z=3, y=2, x=4)"),
        },
        "python_message": message,
    }
    record("syn_zpad_bad", ds, entry)

    # -- syn_staged: 1-degree global geometry, only two far-from-origin
    #    tiles staged -> global extent preserved, wraps.
    tx, ty = 30, 30
    tiles = {}
    for ox, oy in ((181, 61), (211, 61)):
        tiles[f"{ox:05d}-{ox + tx - 1:05d}.{oy:05d}-{oy + ty - 1:05d}"] = \
            rng.integers(0, 250, size=(1, ty, tx)).astype("u1")
    idx_text = (
        "type=continuous\nprojection=regular_ll\ndx=1.0\ndy=1.0\n"
        "known_x=1.0\nknown_y=1.0\nknown_lat=-89.5\nknown_lon=-179.5\n"
        "wordsize=1\ntile_x=30\ntile_y=30\ntile_z=1\n")
    root = make_ds("syn_staged", idx_text, tiles)
    ds = GeogDataset(root)
    mask = ds.tile_coverage_mask(175, 245, 55, 95)
    entry = {
        "coverage_mask": {
            "args": [175, 245, 55, 95],
            "mask": write_arr(exp_dir / "syn_staged_cov.bin", mask),
        },
        "missing_tiles": [list(o)
                          for o in ds.missing_tiles(175, 245, 55, 95)],
        "required_origins": [
            list(o) for o in ds.required_tile_origins(175, 245, 55, 95)],
    }
    record("syn_staged", ds, entry)

    # -- syn_global_nx: declared global dims beat inference.
    idx_text = (
        "type=continuous\nprojection=regular_ll\ndx=0.5\ndy=0.5\n"
        "known_x=1.0\nknown_y=1.0\nknown_lat=-89.75\nknown_lon=-179.75\n"
        "wordsize=1\ntile_x=10\ntile_y=10\ntile_z=1\n"
        "global_nx=20\nglobal_ny=10\n")
    tiles = {"00001-00010.00001-00010":
             rng.integers(0, 9, size=(1, 10, 10)).astype("u1"),
             "00011-00020.00001-00010":
             rng.integers(0, 9, size=(1, 10, 10)).astype("u1")}
    root = make_ds("syn_global_nx", idx_text, tiles)
    ds = GeogDataset(root)
    record("syn_global_nx", ds, {})

    # -- syn_le_u2 / syn_i4_big: word decoding variants.
    idx_text = (
        "type=continuous\nprojection=regular_ll\ndx=1.0\ndy=1.0\n"
        "known_x=1.0\nknown_y=1.0\nknown_lat=-89.5\nknown_lon=-179.5\n"
        "wordsize=2\nendian=little\ntile_x=4\ntile_y=3\ntile_z=1\n")
    arr = rng.integers(0, 65535, size=(1, 3, 4)).astype("<u2")
    root = make_ds("syn_le_u2", idx_text, {"00001-00004.00001-00003": arr})
    ds = GeogDataset(root)
    win = ds.read_window(1, 4, 1, 3)
    record("syn_le_u2", ds, {
        "window": {
            "args": [1, 4, 1, 3], "x0": win.x0, "y0": win.y0,
            "shape": list(win.raw.shape),
            "raw": write_arr(exp_dir / "syn_le_u2_w0.bin",
                             win.raw.astype(np.int64)),
            "coverage": write_arr(exp_dir / "syn_le_u2_w0_cov.bin",
                                  win.coverage),
            "values_z": 0,
            "values": write_arr(exp_dir / "syn_le_u2_w0_vals.bin",
                                win.values(0)),
        },
    })

    idx_text = (
        "type=continuous\nprojection=regular_ll\ndx=1.0\ndy=1.0\n"
        "known_x=1.0\nknown_y=1.0\nknown_lat=-89.5\nknown_lon=-179.5\n"
        "wordsize=4\nsigned=yes\ntile_x=4\ntile_y=3\ntile_z=1\n")
    arr = rng.integers(-2**30, 2**30, size=(1, 3, 4)).astype(">i4")
    root = make_ds("syn_i4_big", idx_text, {"00001-00004.00001-00003": arr})
    ds = GeogDataset(root)
    win = ds.read_window(1, 4, 1, 3)
    record("syn_i4_big", ds, {
        "window": {
            "args": [1, 4, 1, 3], "x0": win.x0, "y0": win.y0,
            "shape": list(win.raw.shape),
            "raw": write_arr(exp_dir / "syn_i4_big_w0.bin",
                             win.raw.astype(np.int64)),
            "coverage": write_arr(exp_dir / "syn_i4_big_w0_cov.bin",
                                  win.coverage),
            "values_z": 0,
            "values": write_arr(exp_dir / "syn_i4_big_w0_vals.bin",
                                win.values(0)),
        },
    })

    dump(OUT / "synthetic_expected" / "manifest.json", manifest)


def gen_contcat():
    """The WPS continuous-category source path (per-plane four_pt then
    f32 normalization across categories), driven through the REAL
    `_DomainSampler.categorical` with a stub coarse mesh."""
    rng = np.random.default_rng(23)
    tx, ty, ncat = 6, 5, 3
    tiles = {}
    for oy in (1, 6):
        for ox in (1, 7):
            arr = rng.integers(0, 100, size=(ncat, ty, tx))
            arr[rng.random((ncat, ty, tx)) < 0.15] = 255
            tiles[f"{ox:05d}-{ox + tx - 1:05d}.{oy:05d}-{oy + ty - 1:05d}"] \
                = arr.astype("u1")
    idx_text = (
        "type=continuous\nprojection=regular_ll\ndx=0.25\ndy=0.25\n"
        "known_x=1.0\nknown_y=1.0\nknown_lat=10.125\nknown_lon=20.125\n"
        f"wordsize=1\ntile_x={tx}\ntile_y={ty}\ntile_z={ncat}\n"
        "missing_value=255.\nscale_factor=0.01\n"
        f"category_min=1\ncategory_max={ncat}\n")
    root = make_ds("syn_contcat", idx_text, tiles)
    ds = GeogDataset(root)
    win = ds.read_window(2, 11, 1, 9)

    nx, ny, halo = 3, 2, 3
    nxe, nye = nx + 2 * halo, ny + 2 * halo
    sampler = object.__new__(_DomainSampler)
    sampler.halo = halo
    sampler.nx, sampler.ny = nx, ny
    sampler.nxe, sampler.nye = nxe, nye
    sampler._cells_cache = {}
    sampler.grid = SimpleNamespace(dx=5000.0)
    lat = (10.3 + 0.045 * np.arange(nxe * nye)).reshape(nye, nxe)
    lon = (20.3 + 0.052 * np.arange(nxe * nye)).reshape(nye, nxe)
    sampler.lat_e = lat.astype(np.float32)
    sampler.lon_e = lon.astype(np.float64)

    frac_interp = _DomainSampler.categorical(sampler, ds, win)

    # gcell variant: a grid coarse enough for the 4x ratio, with the
    # cells cache primed by a crafted binning (the fixture-cells seam on
    # the Rust side), leaving two cells empty.
    sampler.grid = SimpleNamespace(dx=200000.0)
    key = (ds.index.dx, ds.index.dy, ds.index.known_x, ds.index.known_y,
           ds.index.known_lat, ds.index.known_lon, win.x0, win.y0,
           win.raw.shape)
    npx = win.raw.shape[1] * win.raw.shape[2]
    flat = (np.arange(npx, dtype=np.int64) % (nxe * nye - 2))
    flat[::7] = -1
    sampler._cells_cache = {key: flat}
    frac_gcell = _DomainSampler.categorical(sampler, ds, win,
                                            fractional_gcell=True)

    entry = {
        "window_args": [2, 11, 1, 9],
        "nx": nx, "ny": ny, "halo": halo,
        "dx_interp": hex64(5000.0),
        "dx_gcell": hex64(200000.0),
        "lat_e": write_arr(OUT / "contcat" / "lat_e.bin",
                           sampler.lat_e),
        "lon_e": write_arr(OUT / "contcat" / "lon_e.bin",
                           sampler.lon_e),
        "key": {
            "dx": hex64(ds.index.dx), "dy": hex64(ds.index.dy),
            "known_x": hex64(ds.index.known_x),
            "known_y": hex64(ds.index.known_y),
            "known_lat": hex64(ds.index.known_lat),
            "known_lon": hex64(ds.index.known_lon),
            "x0": win.x0, "y0": win.y0,
            "shape": list(win.raw.shape),
        },
        "cells": write_arr(OUT / "contcat" / "cells.bin", flat),
        "frac_interp": write_arr(OUT / "contcat" / "frac_interp.bin",
                                 frac_interp),
        "frac_gcell": write_arr(OUT / "contcat" / "frac_gcell.bin",
                                frac_gcell),
    }
    dump(OUT / "contcat" / "goldens.json", entry)


# --------------------------------------------------------------------------
# interp + smooth goldens
# --------------------------------------------------------------------------

def gen_interp():
    rng = np.random.default_rng(7)
    vals = rng.normal(200.0, 150.0, size=(7, 9))
    vals[rng.random((7, 9)) < 0.22] = np.nan
    x0, y0 = 11, 21
    pts_x = [11.0, 12.4, 14.5, 18.9999, 19.0, 10.5001, 10.4999, 13.0,
             16.75, 19.5, 21.0, 11.5, 12.5]
    pts_y = [21.0, 22.7, 23.5, 26.9999, 27.0, 20.5001, 23.0, 20.4999,
             24.25, 27.5, 21.5, 29.0, 23.5]
    xi = np.array(pts_x)
    yi = np.array(pts_y)
    entry = {
        "vals": write_arr(OUT / "interp" / "vals.bin", vals),
        "x0": x0, "y0": y0,
        "xi": [hex64(v) for v in xi],
        "yi": [hex64(v) for v in yi],
        "ops": {},
    }
    for name, fn in (("four_pt", four_pt), ("average_4pt", average_4pt),
                     ("average_16pt", average_16pt),
                     ("sixteen_pt", sixteen_pt),
                     ("search", search_nearest)):
        got = fn(vals, xi, yi, x0, y0)
        entry["ops"][name] = [hex64(v) for v in np.atleast_1d(got)]
    seq = ("four_pt", "average_4pt", "average_16pt", "search")
    got = _interp_seq(vals, xi, yi, x0, y0, seq)
    entry["seq"] = {"ops": list(seq), "out": [hex64(v) for v in got]}

    # search frontier tie-break: a big hole with two nearly equidistant
    # valid points; the finite-frontier rule differs from an
    # unrestricted nearest search.
    hole = np.full((15, 15), np.nan)
    hole[1, 7] = 5.0
    hole[13, 7] = 9.0
    hole[7, 1] = 13.0
    hole[7, 13] = 17.0
    hole[2, 2] = 21.0
    sx = np.array([8.0, 8.2, 7.6, 8.0, 3.4])
    sy = np.array([8.0, 7.9, 8.4, 3.0, 3.2])
    got = search_nearest(hole, sx, sy, 1, 1)
    entry["search_hole"] = {
        "vals": write_arr(OUT / "interp" / "hole.bin", hole),
        "x0": 1, "y0": 1,
        "xi": [hex64(v) for v in sx], "yi": [hex64(v) for v in sy],
        "out": [hex64(v) for v in got],
    }
    dump(OUT / "interp" / "goldens.json", entry)


def gen_smooth():
    rng = np.random.default_rng(11)
    a = rng.normal(50.0, 400.0, size=(12, 11))
    entry = {
        "input": write_arr(OUT / "smooth" / "input.bin", a),
        "one_two_one_1": write_arr(OUT / "smooth" / "oto1.bin",
                                   one_two_one(a, 1)),
        "one_two_one_2": write_arr(OUT / "smooth" / "oto2.bin",
                                   one_two_one(a, 2)),
        "smth_desmth_1": write_arr(OUT / "smooth" / "sd1.bin",
                                   smth_desmth(a, 1)),
        "smth_desmth_special_1": write_arr(
            OUT / "smooth" / "sds1.bin", smth_desmth_special(a, 1)),
    }
    dump(OUT / "smooth" / "goldens.json", entry)


def gen_field_rules():
    rng = np.random.default_rng(13)
    luf = rng.random((5, 4, 6))
    luf[2, 0, 0] = luf[4, 0, 0] = 0.5      # lake == ocean (not strictly >)
    luf[:, 1, 1] = 0.2                      # all-tie -> lowest category
    luf /= luf.sum(axis=0)
    iswater, islake = 3, 5
    lm = landmask_from_landusef(luf, iswater, islake)
    lu = lu_index_from_landusef(luf, lm, iswater, islake)
    lm_nolake = landmask_from_landusef(luf, iswater, None)
    lu_nolake = lu_index_from_landusef(luf, lm_nolake, iswater, None)
    dom = dominant_category(luf)
    entry = {
        "luf": write_arr(OUT / "fields" / "luf.bin", luf),
        "iswater": iswater, "islake": islake,
        "landmask": write_arr(OUT / "fields" / "landmask.bin", lm),
        "lu_index": write_arr(OUT / "fields" / "lu_index.bin", lu),
        "landmask_nolake": write_arr(OUT / "fields" / "landmask_nl.bin",
                                     lm_nolake),
        "lu_index_nolake": write_arr(OUT / "fields" / "lu_index_nl.bin",
                                     lu_nolake),
        "dominant": write_arr(OUT / "fields" / "dominant.bin", dom),
    }

    from datetime import datetime
    monthly = rng.normal(280.0, 12.0, size=(12, 2, 3))
    entry["monthly"] = write_arr(OUT / "fields" / "monthly.bin", monthly)
    cases = []
    for when in (datetime(1974, 2, 20, 6), datetime(1974, 1, 5),
                 datetime(1974, 12, 25, 23), datetime(2024, 2, 15),
                 datetime(2024, 7, 15), datetime(2024, 1, 16)):
        got = monthly_interp_to_date(monthly, when)
        mids = [datetime(when.year, m, 15).timetuple().tm_yday
                for m in range(1, 13)]
        cases.append({
            "year": when.year,
            "julian": when.timetuple().tm_yday,
            "mid_month_julian": mids,
            "out": write_arr(
                OUT / "fields"
                / f"monthly_{when.year}_{when.timetuple().tm_yday}.bin",
                got),
        })
    entry["monthly_cases"] = cases
    dump(OUT / "fields" / "goldens.json", entry)


# --------------------------------------------------------------------------
# pixel binning golden (crafted knife-edge coordinates through the real
# pixel_cells)
# --------------------------------------------------------------------------

def gen_binning():
    nxe, nye, halo = 9, 8, 3
    nyw, nxw = 6, 7
    rng = np.random.default_rng(17)
    gx = rng.uniform(-4.0, 10.0, size=(nyw, nxw))
    gy = rng.uniform(-4.0, 9.0, size=(nyw, nxw))
    # knife edges: exactly half, just inside/outside the 5e-5 snap band
    gx[0, 0], gy[0, 0] = 2.5, 3.5
    gx[0, 1], gy[0, 1] = 2.5 + 4.9e-5, 3.5 - 4.9e-5
    gx[0, 2], gy[0, 2] = 2.5 + 5.0e-5, 3.5 - 5.0e-5
    gx[0, 3], gy[0, 3] = 2.5 - 6.0e-5, 3.5 + 6.0e-5
    gx[0, 4], gy[0, 4] = np.nan, 2.0
    gx[0, 5], gy[0, 5] = 55.0, -55.0
    gx[1, 0], gy[1, 0] = -2.4999, -2.5001

    sampler = object.__new__(_DomainSampler)
    sampler.halo = halo
    sampler.nx, sampler.ny = nxe - 2 * halo, nye - 2 * halo
    sampler.nxe, sampler.nye = nxe, nye
    sampler._cells_cache = {}
    sampler.grid = SimpleNamespace(
        latlon_to_ij=lambda lat, lon: (gx, gy), dx=6000.0)
    ds = SimpleNamespace(
        index=SimpleNamespace(dx=1.0, dy=1.0, known_x=1.0, known_y=1.0,
                              known_lat=0.0, known_lon=0.0),
        xy_to_latlon=lambda x, y: (y * np.ones_like(x),
                                   x * np.ones_like(y)),
    )
    win = SimpleNamespace(x0=1, y0=1, raw=np.zeros((1, nyw, nxw)))
    flat = _DomainSampler.pixel_cells(sampler, ds, win)
    entry = {
        "nxe": nxe, "nye": nye, "halo": halo,
        "shape": [nyw, nxw],
        "gx": write_arr(OUT / "binning" / "gx.bin", gx),
        "gy": write_arr(OUT / "binning" / "gy.bin", gy),
        "flat": write_arr(OUT / "binning" / "flat.bin", flat),
    }
    dump(OUT / "binning" / "goldens.json", entry)


# --------------------------------------------------------------------------
# real-tree goldens: index parses, real windows, two domain packages
# --------------------------------------------------------------------------

def cells_key_json(ds, win):
    idx = ds.index
    return {
        "dx": hex64(idx.dx), "dy": hex64(idx.dy),
        "known_x": hex64(idx.known_x), "known_y": hex64(idx.known_y),
        "known_lat": hex64(idx.known_lat), "known_lon": hex64(idx.known_lon),
        "x0": win.x0, "y0": win.y0,
        "shape": list(win.raw.shape),
    }


def domain_package(tag: str, grid, selection):
    pkg = OUT / f"real_domain_{tag}"
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir(parents=True)
    halo = 3
    dom = _DomainSampler(grid, halo)
    wps = dom._wps_grid
    nx, ny = dom.nx, dom.ny
    nxe, nye = dom.nxe, dom.nye
    xs = np.arange(1 - halo, nx + halo + 1, dtype=np.float64)
    ys = np.arange(1 - halo, ny + halo + 1, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    lat32, lon32 = wps.ij_to_latlon(X, Y)
    lat64, lon64 = grid.ij_to_latlon(X, Y)
    Xc, Yc = np.meshgrid(np.arange(0.5 - halo, nx + halo + 1.0),
                         np.arange(0.5 - halo, ny + halo + 1.0))
    latc32, _ = wps.ij_to_latlon(Xc, Yc)
    _, lonc64 = grid.ij_to_latlon(Xc, Yc)

    meta = {
        "dx": hex64(grid.dx), "halo": halo, "nx": nx, "ny": ny,
        "is_lambert": True,
        "lon_e_dtype": "f32" if dom.lon_e.dtype == np.float32 else "f64",
        "mesh_in": {
            "lat32": write_arr(pkg / "in_lat32.bin", lat32),
            "lon32": write_arr(pkg / "in_lon32.bin", lon32),
            "lat64": write_arr(pkg / "in_lat64.bin", lat64),
            "lon64": write_arr(pkg / "in_lon64.bin", lon64),
            "latc32": write_arr(pkg / "in_latc32.bin", latc32),
            "lonc64": write_arr(pkg / "in_lonc64.bin", lonc64),
        },
        "mesh_out": {
            "lat_e": write_arr(pkg / "out_lat_e.bin", dom.lat_e),
            "lat_lower": write_arr(pkg / "out_lat_lower.bin",
                                   dom._lat_lower_e),
            "lon_e": write_arr(pkg / "out_lon_e.bin", dom.lon_e),
            "lon_band": write_arr(pkg / "out_lon_band.bin",
                                  dom._lon_boundary_band),
            "lat_band": write_arr(pkg / "out_lat_band.bin",
                                  dom._lat_integer_band),
        },
    }

    fields = ("terrain", "landuse", "soil_top", "soil_bottom",
              "greenfrac", "lai", "albedo", "snow_albedo",
              "soil_temperature")
    windows = {}
    gcell_fields = []
    for field in fields:
        ds = GeogDataset(selection.path(field))
        win = dom.window(ds)
        needs_cells = (
            ds.index.type == "categorical"
            or dom.res_ratio(ds) >= 4.0)
        entry = {
            "dir": getattr(selection, field),
            "x0": win.x0, "y0": win.y0,
            "shape": list(win.raw.shape),
            "cells": None,
            "key": cells_key_json(ds, win),
        }
        if needs_cells:
            flat = dom.pixel_cells(ds, win)
            entry["cells"] = write_rle_i64(
                pkg / f"cells_{field}.bin", flat)
            gcell_fields.append(field)
        windows[field] = entry
    meta["windows"] = windows

    # cell_coords golden for one dataset (the f32 path on the sub-km
    # domain, the f64 path on the coarse one).
    probe = "landuse" if grid.dx < 1000.0 else "terrain"
    ds = GeogDataset(selection.path(probe))
    win = dom.window(ds)
    ci, cj = dom.cell_coords(ds, win)
    meta["cell_coords"] = {
        "field": probe,
        "xi": write_arr(pkg / "coords_xi.bin", np.asarray(ci)),
        "yi": write_arr(pkg / "coords_yi.bin", np.asarray(cj)),
    }

    if grid.dx >= 1000.0:
        # raw grid coordinates for the terrain window: the pure-binning
        # golden (gx/gy in, the committed cells out).
        ds = GeogDataset(selection.path("terrain"))
        win = dom.window(ds)
        nyw, nxw = win.raw.shape[1:]
        xs_abs = win.x0 + np.arange(nxw, dtype=np.float64)
        yy = win.y0 + np.arange(nyw, dtype=np.float64)
        lat, lon = ds.xy_to_latlon(xs_abs[None, :], yy[:, None])
        gx, gy = grid.latlon_to_ij(lat, lon)
        meta["binning"] = {
            "field": "terrain",
            "gx": write_arr(pkg / "gx.bin", gx),
            "gy": write_arr(pkg / "gy.bin", gy),
        }

    report = {}
    out = build_static(grid, selection.root, selection=selection,
                       source_coverage_report=report)
    build_files = {}
    for name, arr in out.items():
        build_files[name] = {
            "shape": list(arr.shape),
            "file": write_arr(pkg / f"build_{name}.bin", arr),
        }
    meta["build"] = build_files
    dump(pkg / "receipt_terrain.json", report["terrain"])
    dump(pkg / "package.json", meta)


def gen_real():
    if not (GEOG_ROOT / "topo_gmted2010_30s" / "index").is_file():
        print("real tree absent; skipping real goldens")
        return
    selection = GeogSelection.fallback(GEOG_ROOT)
    manifest = {"root_note": "paths resolved against "
                "GPUWM_STATIC_PARITY_GEOG at test time",
                "indexes": {}}
    for field in ("terrain", "landuse", "soil_top", "soil_bottom",
                  "greenfrac", "lai", "albedo", "snow_albedo",
                  "soil_temperature"):
        rel = getattr(selection, field)
        idx = parse_index(GEOG_ROOT / rel / "index")
        ds = GeogDataset(GEOG_ROOT / rel)
        manifest["indexes"][field] = {
            "dir": rel,
            "index": index_json(idx),
            "inventory": inventory_json(ds),
        }
    # two real window reads: topo (wraps, i2 big, bdr 3) and soiltemp
    # (1-degree, u2, 2 tiles), including out-of-extent rows.
    topo = GeogDataset(GEOG_ROOT / selection.terrain)
    win = topo.read_window(33110, 33190, 15500, 15560)
    manifest["topo_window"] = {
        "args": [33110, 33190, 15500, 15560],
        "shape": list(win.raw.shape),
        "raw": write_arr(OUT / "real" / "topo_w0.bin",
                         win.raw.astype(np.int64)),
        "values": write_arr(OUT / "real" / "topo_w0_vals.bin",
                            win.values(0)),
    }
    st = GeogDataset(GEOG_ROOT / selection.soil_temperature)
    win = st.read_window(170, 220, -3, 12)
    manifest["soiltemp_window"] = {
        "args": [170, 220, -3, 12],
        "shape": list(win.raw.shape),
        "raw": write_arr(OUT / "real" / "soiltemp_w0.bin",
                         win.raw.astype(np.int64)),
        "values": write_arr(OUT / "real" / "soiltemp_w0_vals.bin",
                            win.values(0)),
        "coverage": write_arr(OUT / "real" / "soiltemp_w0_cov.bin",
                              win.coverage),
    }
    dump(OUT / "real" / "manifest.json", manifest)

    coarse = LambertGrid(39.5, -84.0, 38.0, 41.0, -84.0, 6000.0, 6000.0,
                         13, 12)
    domain_package("coarse", coarse, selection)
    subkm = LambertGrid(39.5, -84.0, 38.0, 41.0, -84.0, 250.0, 250.0,
                        8, 7)
    domain_package("subkm", subkm, selection)


def write_provenance():
    import platform
    import subprocess
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
            text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance only
        rev = "unknown"
    dump(OUT / "PROVENANCE.json", {
        "generator": "golden/gen_lane2_goldens.py",
        "oracle": "gpuwm.static (Python reference implementation)",
        "repo_rev": rev,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "geog_root_present": (GEOG_ROOT / "topo_gmted2010_30s"
                              / "index").is_file(),
    })


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gen_synthetic()
    gen_contcat()
    gen_interp()
    gen_smooth()
    gen_field_rules()
    gen_binning()
    gen_real()
    write_provenance()
    print("goldens written to", OUT)


if __name__ == "__main__":
    main()
