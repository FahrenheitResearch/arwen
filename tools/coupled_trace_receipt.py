"""Small stage hash receipts from the delivered diagnostic trace readers.

Each box reads its own arrays. Only hashes and block boundaries leave the box.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "wrf_exact_localize"))
from cmpphys import chain_pairs, wrf_array, supported_scalar


def digest(value, block_words=2048):
    value = np.ascontiguousarray(value, dtype=np.float32)
    raw = value.tobytes()
    return dict(shape=list(value.shape), words=value.size,
                sha256=hashlib.sha256(raw).hexdigest(), block_words=block_words,
                blocks=[hashlib.sha256(raw[i:i+4*block_words]).hexdigest()
                        for i in range(0, len(raw), 4*block_words)])


def collect(directory, arm, shape, soil_layers, specified, *, keys=(), block_words=2048,
            step=1, cu_physics=None):
    known = {f"{s}/{d}/{n}" for s, d, _, n, _, _ in chain_pairs(components=True)}
    if set(keys) - known:
        raise ValueError(f"unknown trace keys: {sorted(set(keys) - known)}")
    nz, ny, nx = shape
    rows, missing = {}, []
    for stage, direction, wt, wn, ot, on in chain_pairs(components=True):
        key = f"{stage}/{direction}/{wn}"
        if stage == "coupled_cumulus" and cu_physics == 0:
            if key in keys:
                raise ValueError("cu=0 has no computed cumulus component")
            continue
        if keys and key not in keys:
            continue
        try:
            if arm == "woof":
                paths = list(directory.glob(f"*_s{step:03d}_*_{ot}.npz"))
                if len(paths) != 1:
                    raise ValueError(f"{ot}: {len(paths)} dumps")
                with np.load(paths[0], allow_pickle=False) as z:
                    value = z[on]
            else:
                field_shape = (nz, ny, nx)
                if stage in ("sfclay", "noah", "ysu") and wn not in (
                        "rublten", "rvblten", "rthblten", "rqvblten", "rqcblten"):
                    field_shape = ((soil_layers, ny, nx) if wn in ("smois", "tslb", "sh2o") else (ny, nx))
                if wn in ("p_interface", "z_interface"):
                    field_shape = (nz+1, ny, nx)
                if wn == "ru_tendf": field_shape = (nz, ny, nx+1)
                if wn == "rv_tendf": field_shape = (nz, ny+1, nx)
                value = wrf_array(directory, wt, wn, field_shape, step=step)
                if stage in ("coupled_pbl", "coupled_radiation", "coupled_cumulus"):
                    value = supported_scalar(value, specified=specified)
            rows[key] = dict(wrf_tag=wt, wrf_field=wn, woof_tag=ot, woof_field=on,
                             **digest(value, block_words))
        except (ValueError, OSError, KeyError) as error:
            missing.append(dict(key=key, error=str(error)))
    return dict(arm=arm, directory=str(directory), step=step, cu_physics=cu_physics,
                rows=rows, missing=missing)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--arm", choices=("wrf", "woof"), required=True)
    p.add_argument("--shape", type=int, nargs=3, required=True)
    p.add_argument("--soil-layers", type=int, required=True)
    p.add_argument("--specified", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--key", action="append", default=[])
    p.add_argument("--block-words", type=int, default=2048)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--cu-physics", type=int)
    a = p.parse_args()
    if a.block_words < 1:
        p.error("--block-words must be positive")
    receipt = collect(a.directory, a.arm, a.shape, a.soil_layers, a.specified,
                      keys=set(a.key), block_words=a.block_words,
                      step=a.step, cu_physics=a.cu_physics)
    a.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(dict(arm=a.arm, rows=len(receipt["rows"]), missing=receipt["missing"])))
    return int(bool(receipt["missing"]))


if __name__ == "__main__":
    raise SystemExit(main())
