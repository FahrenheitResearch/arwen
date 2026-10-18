"""Hash aligned held-diffusion and RK state boundaries on each box."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "wrf_exact_localize"))
sys.path.insert(0, str(Path(__file__).parent))
from coupled_trace_receipt import digest


def points(shape):
    nz, ny, nx = shape
    shapes = {"u": (nz, ny, nx+1), "v": (nz, ny+1, nx),
              "w": (nz+1, ny, nx), "theta": (nz, ny, nx),
              "phi": (nz+1, ny, nx), "mu": (ny, nx)}
    for field, wn, on in (("u", "ru_tendf", "x_hv_ru"),
                           ("v", "rv_tendf", "x_hv_rv"),
                           ("w", "rw_tendf", "x_hv_rw"),
                           ("theta", "t_tendf", "x_hv_rth")):
        yield f"held_diffusion/{field}", 1, 0, "c2_hdiff", wn, 0, 0, "c02_vsmag", on, shapes[field]
    state = (("u", "u_2", "u"), ("v", "v_2", "v"), ("w", "w_2", "w"),
             ("theta", "t_2", "thp"), ("phi", "ph_2", "php"), ("mu", "mu_2", "mup"))
    tends = (("u", "ru_tend", "ru_t"), ("v", "rv_tend", "rv_t"),
             ("w", "rw_tend", "rw_t"), ("theta", "t_tend", "rth_t"),
             ("phi", "ph_tend", "rph_t"), ("mu", "mu_tend", "rmu_t"))
    for rk in (1, 2, 3):
        for field, wn, on in state:
            yield f"rk{rk}/prep/{field}", rk, 0, "b_prep", wn, rk, 0, "b_prep", on, shapes[field]
        for wt, ot in (("d_rktend", "d_slow"), ("e_addtend", "e_lateral")):
            for field, wn, on in tends:
                yield f"rk{rk}/{wt}/{field}", rk, 0, wt, wn, rk, 0, ot, on, shapes[field]


def read_wrf(directory, rk, it, tag, name, shape, *, step=1):
    ids, _, jds, _, kds, *_ = map(int, (directory / "dims.txt").read_text().split())
    raw = (directory / f"s{step:03d}_rk{rk}_it{it:02d}_{tag}__{name}.bin").read_bytes()
    i0, i1, k0, k1, j0, j1 = np.frombuffer(raw[:24], "<i4")
    value = np.frombuffer(raw[24:], "<f4").reshape((i1-i0+1, k1-k0+1, j1-j0+1), order="F")
    nk, nj, ni = (1, *shape) if len(shape) == 2 else shape
    value = value[ids-i0:ids-i0+ni, kds-k0:kds-k0+nk, jds-j0:jds-j0+nj].transpose(1,2,0)
    if len(shape) == 2:
        value = value[0]
    if value.shape != shape:
        raise ValueError(f"WRF crop {value.shape} differs from required {shape}")
    return value


def collect(directory, arm, shape, keys=(), block_words=2048, *, step=1):
    rows, missing = {}, []
    for key, rk, it, wt, wn, ork, oit, ot, on, expected in points(shape):
        if keys and key not in keys:
            continue
        try:
            if arm == "wrf":
                value = read_wrf(directory, rk, it, wt, wn, expected, step=step)
            else:
                paths = list(directory.glob(f"*_s{step:03d}_rk{ork}_it{oit:02d}_{ot}.npz"))
                if len(paths) != 1:
                    raise ValueError(f"{ot}: {len(paths)} matching dumps")
                with np.load(paths[0], allow_pickle=False) as z:
                    value = z[on]
                if value.shape != expected:
                    raise ValueError(f"WOOF shape {value.shape} differs from required {expected}")
            rows[key] = digest(value, block_words)
        except (ValueError, OSError, KeyError) as error:
            missing.append(dict(key=key, error=str(error)))
    known = {p[0] for p in points(shape)}
    if set(keys) - known:
        raise ValueError("unknown dycore trace key")
    return dict(arm=arm, rows=rows, missing=missing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--arm", choices=("wrf", "woof"), required=True)
    parser.add_argument("--shape", type=int, nargs=3, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--key", action="append", default=[])
    parser.add_argument("--block-words", type=int, default=2048)
    parser.add_argument("--step", type=int, default=1)
    args = parser.parse_args()
    if args.block_words < 1:
        parser.error("--block-words must be positive")
    result = collect(args.directory, args.arm, tuple(args.shape), args.key,
                     args.block_words, step=args.step)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(rows=len(result["rows"]), missing=result["missing"])))
    return int(bool(result["missing"]))


if __name__ == "__main__":
    raise SystemExit(main())
