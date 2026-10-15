"""Pack the Grell 3D oracle's binary runs into the committed npz fixtures.

usage: pack_g3_fixture.py CASES_DIR FIXTURE_DIR

Each configuration becomes FIXTURE_DIR/g3-<cfg>.npz holding every input
array (``in_<name>``), every output array (``out_<name>``), the header
scalars and the column-family map.  Words are stored as written, so the
parity tests compare raw float32 bits.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g3_fixture_io import read_input, read_output  # noqa: E402


def main() -> None:
    cases, fixtures = Path(sys.argv[1]), Path(sys.argv[2])
    fixtures.mkdir(parents=True, exist_ok=True)
    for fin in sorted(cases.glob("g3-in-*.bin")):
        cfg = fin.stem.removeprefix("g3-in-")
        inp = read_input(fin)
        out = read_output(cases / f"g3-out-{cfg}.bin", inp["nx"], inp["ny"],
                          inp["nz"], ensdim=inp["ensdim"])
        arrays = {}
        for k, v in inp.items():
            arrays[f"in_{k}"] = np.asarray(v)
        for k, v in out.items():
            arrays[f"out_{k}"] = v
        fam = cases / f"g3-family-{cfg}.npy"
        if fam.exists():
            arrays["family"] = np.load(fam)
        np.savez_compressed(fixtures / f"g3-{cfg}.npz", **arrays)
        print(f"{cfg}: packed {len(arrays)} arrays")


if __name__ == "__main__":
    main()
