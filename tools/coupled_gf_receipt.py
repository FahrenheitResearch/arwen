"""Hash actual GF launch inputs on each box; export no model arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "wrf_exact_localize"))
from cmpphys import wrf_array
from gpuwm.core.gf import _IN_LEV, _IN_SCA


def collect(directory, arm, shape):
    nz, ny, nx = shape
    if arm == "woof":
        parts = sorted(directory.glob("gf_launch_*.npz"))
        if not parts:
            raise ValueError("no captured GF launches")
        packed = {}
        for key in ("lvin", "scin", "iin"):
            slabs = []
            for path in parts:
                with np.load(path, allow_pickle=False) as z:
                    slabs.append(z[key])
            packed[key] = np.concatenate(slabs)
        if packed["lvin"].shape != (ny * nx, len(_IN_LEV), nz):
            raise ValueError("captured GF launches do not cover exactly one domain")
        values = {name: packed["lvin"][:, j, :].T.reshape(nz, ny, nx)
                  for j, name in enumerate(_IN_LEV)}
        values.update({name: packed["scin"][:, j].reshape(ny, nx)
                       for j, name in enumerate(_IN_SCA[:6])})
        values.update({name: packed["iin"][:, j].reshape(ny, nx)
                       for j, name in enumerate(("kpbl", "ishallow", "clos_choice"))})
    else:
        values = {name: wrf_array(directory, "cu_gf_input",
                                 "qv_curr" if name == "qv" else name, shape)
                  for name in _IN_LEV}
        for name in ("ht", "hfx", "qfx", "xland", "kpbl"):
            value = wrf_array(directory, "cu_gf_input", name, (ny, nx))
            values[name] = value.view(np.int32) if name == "kpbl" else value
        for name in ("dt", "dx", "ishallow", "clos_choice"):
            raw = (directory / f"s001_rk1_it00_cu_gf_input__{name}.bin").read_bytes()
            dtype = "<i4" if name in ("ishallow", "clos_choice") else "<f4"
            scalar = np.frombuffer(raw[24:], dtype=dtype)
            if scalar.size != 1:
                raise ValueError(f"{name} is not a scalar dump")
            values[name] = np.full((ny, nx), scalar[0], dtype=dtype)
    rows = {}
    for name, value in values.items():
        value = np.ascontiguousarray(value)
        inner = np.ascontiguousarray(value[..., 1:-1, 1:-1])
        selected = value[:, 1, 4] if value.ndim == 3 else value[1:2, 4]
        rows[name] = dict(shape=list(value.shape), dtype=str(value.dtype),
                          sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                          interior_sha256=hashlib.sha256(inner.tobytes()).hexdigest(),
                          selected_words=[hex(int(x)) for x in selected.view(np.uint32).flat])
    return dict(arm=arm, shape=list(shape), rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--arm", choices=("wrf", "woof"), required=True)
    parser.add_argument("--shape", nargs=3, type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.directory, args.arm, tuple(args.shape))
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(arm=args.arm, fields=len(result["rows"]))))


if __name__ == "__main__":
    main()
