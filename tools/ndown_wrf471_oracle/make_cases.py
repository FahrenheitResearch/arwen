"""make_cases.py: the WRF v4.7.1 ndown oracle's cases, and the fixture they become.

    python make_cases.py write DIR      write DIR/<mode>-<name>.case (run them with build.sh)
    python make_cases.py collect DIR NPZ  read every case and the oracle's <case>.out into one NPZ fixture

Two modes, one per WRF routine (see run_ndown.F90 for the byte layout):

* ``rebalance``: dyn_em/module_initialize_real.F:4982-5266, the state moved from the parent's interpolated terrain
  onto the child's.  Four vertical set-ups (the 59-level hybrid ladder the backbone parent runs, with
  hypsometric_opt 2 and 1, and a 40-level terrain-following ladder, tanh-stretched toward the ground, with both)
  times three terrain families: a coast (sea on both terrains, the parent's sea under the child's cliff, coastal
  hills), ridges and canyons (the child's own terrain up to 450 m above or below the parent's), and high terrain
  (crests above 4 km).  Each case is a 4 x 3 slab of columns with a stable profile, a warm and a cold layer, moist and
  dry columns and a dry-mass perturbation of up to 600 Pa.
* ``blend``: dyn_em/nest_init_utils.F:712-785, the terrain blend across the child's edge rows, at the widths a
  derived child and a live nest use.

Every number is written as float32 (WRF's REAL) and recorded in the fixture exactly as written, so the port is
graded on the same inputs the oracle read.  The vertical coefficients come from the engine's own VerticalCoord,
rounded to float32.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REBALANCE_INPUT_2D = ("ht", "ht_fine", "mu_2")
REBALANCE_INPUT_3D = ("t_2", "qv")
COORD_FULL = ("c1f", "c2f", "c3f", "c4f")
COORD_HALF = ("c1h", "c2h", "c3h", "c4h", "dnw", "rdnw", "rdn")
REBALANCE_OUTPUTS = (  # (name, kind): kind h = (ny, nz, nx), f = (ny, nz+1, nx), s = (ny, nx)
    ("pb", "h"), ("t_init", "h"), ("alb", "h"), ("phb", "f"), ("mub", "s"), ("ht", "s"),
    ("t_2", "h"), ("p", "h"), ("alt", "h"), ("al", "h"), ("p_hyd", "h"), ("ph_2", "f"),
    ("ph0", "f"), ("psfc", "s"))
BASE_CONSTANTS = ("t00", "p00", "tlp", "tiso", "p_strat", "tlp_strat", "p_top")

LADDER_59 = (
    1, 0.993814707, 0.985950649, 0.976014256, 0.963557541, 0.948093116, 0.929123759, 0.90619123, 0.87894237,
    0.847207963, 0.811077714, 0.770949006, 0.727525413, 0.684030771, 0.642961025, 0.604180932, 0.567562938,
    0.532986403, 0.500337601, 0.469508916, 0.440399021, 0.412912011, 0.386957437, 0.362449884, 0.339308649,
    0.317457527, 0.296824664, 0.277342081, 0.258945674, 0.241574913, 0.225172549, 0.2096847, 0.195060253,
    0.181251153, 0.168211967, 0.155899644, 0.144273847, 0.133296132, 0.122930467, 0.113142714, 0.103900604,
    0.095173724, 0.0869334266, 0.0791524947, 0.0718053728, 0.0648678541, 0.0583171472, 0.0521316081, 0.0462909527,
    0.0407758839, 0.0355683193, 0.030651059, 0.026007941, 0.0216237046, 0.0174838807, 0.0135748768, 0.00988376327,
    0.00639845803, 0.00310745789, 0)

#: (name, eta ladder or level count, hybrid_opt, etac, p_top, base_temp, hypsometric_opt)
VERTICAL = (
    ("h59-hyps2", LADDER_59, 2, 0.2, 5000.0, 290.0, 2),
    ("h59-hyps1", LADDER_59, 2, 0.2, 5000.0, 290.0, 1),
    ("s40-hyps1", 40, 0, 0.2, 10000.0, 300.0, 1),
    ("s40-hyps2", 40, 0, 0.2, 10000.0, 300.0, 2),
)
TERRAIN = ("coast", "canyon", "high")
NX, NY = 4, 3
#: The oracle's outputs the fixture keeps; al, p_hyd, ph0 and ht are sums or copies of these.
FIXTURE_OUTPUTS = ("pb", "t_init", "alb", "phb", "mub", "t_2", "p", "alt", "ph_2", "psfc")


def vertical_coord(ladder, hybrid_opt, etac, p_top):
    """The engine's VerticalCoord for one set-up, finalized at ``p_top``."""
    from gpuwm.core.grid import finalize_vertical_coord, make_vertical_coord

    if isinstance(ladder, int):
        coord = make_vertical_coord(ladder, stretch=2.0, hybrid_opt=int(hybrid_opt), etac=float(etac))
    else:
        znw = np.asarray(ladder, dtype=np.float64)
        coord = make_vertical_coord(znw.size - 1, hybrid_opt=int(hybrid_opt), etac=float(etac), eta_levels=znw)
    finalize_vertical_coord(coord, float(p_top))
    return coord


def _terrain(kind, rng):
    """(parent terrain interpolated to the child, child terrain), (NY, NX) metres."""
    if kind == "coast":
        # Sea on both terrains, the parent's sea under the child's coastal cliff, and rising coastal hills.
        coarse = np.array([[0.0, 0.0, 40.0, 120.0], [15.0, 60.0, 200.0, 350.0], [80.0, 250.0, 420.0, 600.0]])
        detail = rng.uniform(-150.0, 180.0, coarse.shape)
        fine = np.clip(coarse + detail, 1.0, None)
        fine[0, 0] = 0.0
        fine[0, 1] = 85.0
    elif kind == "canyon":
        # The child's own ridges and canyons up to 450 m above or below the parent's smoothed terrain.
        coarse = rng.uniform(600.0, 1400.0, (NY, NX))
        fine = coarse + rng.choice([-1.0, 1.0], (NY, NX)) * rng.uniform(150.0, 450.0, (NY, NX))
    else:
        coarse = rng.uniform(2600.0, 4100.0, (NY, NX))
        fine = coarse + rng.uniform(-350.0, 350.0, (NY, NX))
    return coarse.astype(np.float32), fine.astype(np.float32)


def rebalance_case(vertical, terrain_kind, seed):
    """One rebalance case: float32 inputs in numpy (j, k, i) order."""
    from gpuwm.core import constants as c
    from gpuwm.ingest.real import _make_real_base_serial

    name, ladder, hybrid_opt, etac, p_top, base_temp, hyps = vertical
    rng = np.random.default_rng(seed)
    coord = vertical_coord(ladder, hybrid_opt, etac, p_top)
    nz = int(coord.dnw.size)
    ht, ht_fine = _terrain(terrain_kind, rng)
    base = _make_real_base_serial(coord, ht.astype(np.float64), p_top, base_temp, hyps)
    # A stable column on the parent's terrain: the base-state theta, a warm layer near 700 hPa, a cold pool in the
    # lowest levels over high ground, and a column-varying perturbation.
    z = (base.phb[:-1] + base.phb[1:]) * 0.5 / c.G - ht[None].astype(np.float64)
    pb = base.pb
    warm = 3.0 * np.exp(-((pb - 70000.0) / 6000.0) ** 2)
    cold = -2.5 * np.exp(-z / 400.0) * (ht[None] > 500.0)
    wave = 0.8 * np.sin(np.linspace(0.0, 3.0, NX))[None, None, :] * np.cos(np.linspace(0.0, 2.0, NY))[None, :, None]
    theta = base.thb + warm + cold + wave + rng.normal(0.0, 0.15, base.thb.shape)
    t_2 = (theta - c.T0).astype(np.float32)
    moist = np.where(np.arange(NX)[None, None, :] % 5 == 0, 0.25, 1.0)
    qv = (0.011 * moist * np.exp(-z / 2300.0) + 2.0e-6).astype(np.float32)
    mu_2 = (600.0 * np.sin(np.linspace(0.0, 4.0, NX))[None, :] * np.cos(np.linspace(0.0, 3.0, NY))[:, None]
            + rng.normal(0.0, 60.0, (NY, NX))).astype(np.float32)
    case = {
        "nx": NX, "ny": NY, "nz": nz, "hypsometric_opt": hyps,
        "t00": base_temp, "p00": 100000.0, "tlp": 50.0, "tiso": 200.0, "p_strat": 0.0, "tlp_strat": -11.0,
        "p_top": p_top, "hybrid_opt": hybrid_opt, "etac": etac,
        "znw": np.asarray(coord.znw, dtype=np.float64),
        "ht": ht, "ht_fine": ht_fine, "mu_2": mu_2,
        "t_2": np.ascontiguousarray(np.transpose(t_2, (1, 0, 2))),
        "qv": np.ascontiguousarray(np.transpose(qv, (1, 0, 2))),
    }
    for key in COORD_FULL + COORD_HALF:
        case[key] = np.asarray(getattr(coord, key), dtype=np.float32)
    return f"rebalance-{name}-{terrain_kind}", case


def write_rebalance(path, case):
    nz = case["nz"]
    with open(path, "wb") as f:
        np.array([case["nx"], case["ny"], nz, case["hypsometric_opt"]], dtype="<i4").tofile(f)
        np.array([case[k] for k in BASE_CONSTANTS], dtype="<f4").tofile(f)
        for key in COORD_FULL:
            assert case[key].shape == (nz + 1,), key
            case[key].astype("<f4").tofile(f)
        for key in COORD_HALF:
            assert case[key].shape == (nz,), key
            case[key].astype("<f4").tofile(f)
        for key in REBALANCE_INPUT_2D + REBALANCE_INPUT_3D:
            case[key].astype("<f4").tofile(f)


def read_rebalance_output(path, nx, ny, nz):
    raw = np.fromfile(path, dtype="<f4")
    shapes = {"h": (ny, nz, nx), "f": (ny, nz + 1, nx), "s": (ny, nx)}
    out, at = {}, 0
    for name, kind in REBALANCE_OUTPUTS:
        size = int(np.prod(shapes[kind]))
        out[name] = raw[at:at + size].reshape(shapes[kind])
        at += size
    if at != raw.size:
        raise ValueError(f"{path}: {raw.size} words, the layout reads {at}")
    return out


BLEND = (  # (name, nx, ny, spec_bdy_width, blend_width)
    ("child-sbw7-bw5", 60, 52, 7, 5),
    ("nest-sbw5-bw5", 40, 36, 5, 5),
    ("narrow-sbw1-bw3", 24, 30, 1, 3),
)


def blend_case(spec, seed):
    name, nx, ny, sbw, bw = spec
    rng = np.random.default_rng(seed)
    coarse = rng.uniform(0.0, 2500.0, (ny, nx)).astype(np.float32)
    fine = (coarse + rng.normal(0.0, 300.0, (ny, nx))).astype(np.float32)
    return f"blend-{name}", {"nx": nx, "ny": ny, "spec_bdy_width": sbw, "blend_width": bw,
                             "ter_interpolated": coarse, "ter_input": fine}


def write_blend(path, case):
    with open(path, "wb") as f:
        np.array([case["nx"], case["ny"], case["spec_bdy_width"], case["blend_width"]], dtype="<i4").tofile(f)
        case["ter_interpolated"].astype("<f4").tofile(f)
        case["ter_input"].astype("<f4").tofile(f)


def all_cases():
    seed = 20261001
    for vertical in VERTICAL:
        for terrain in TERRAIN:
            seed += 1
            yield rebalance_case(vertical, terrain, seed)
    for spec in BLEND:
        seed += 1
        yield blend_case(spec, seed)


def write(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for stem, case in all_cases():
        (write_rebalance if stem.startswith("rebalance-") else write_blend)(directory / f"{stem}.case", case)
        print(stem)


def collect(directory, npz):
    directory = Path(directory)
    arrays = {}
    for stem, case in all_cases():
        out = directory / f"{stem}.out"
        if not out.is_file():
            raise SystemExit(f"{out} is missing: run build.sh on {directory} first")
        for key, value in case.items():
            arrays[f"{stem}/in/{key}"] = np.asarray(value)
        if stem.startswith("rebalance-"):
            full = read_rebalance_output(out, case["nx"], case["ny"], case["nz"])
            result = {key: full[key] for key in FIXTURE_OUTPUTS}
        else:
            result = {"ter_input": np.fromfile(out, dtype="<f4").reshape(case["ny"], case["nx"])}
        for key, value in result.items():
            arrays[f"{stem}/wrf/{key}"] = value
    np.savez_compressed(npz, **arrays)
    print(f"{npz}: {len(arrays)} arrays")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "write":
        write(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "collect":
        collect(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
