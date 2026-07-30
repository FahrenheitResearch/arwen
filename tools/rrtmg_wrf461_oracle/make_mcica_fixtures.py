"""Generate McICA oracle fixtures for tests/test_rrtmg_mcica.py.

Two phases:

  python make_mcica_fixtures.py emit-inputs WORKDIR
      Writes one big-endian stream input file per case
      (mcica-<case>.in) for the mcica_fixture_lw / mcica_fixture_sw
      drivers built by build.sh.  Inputs are deterministic
      (seeded numpy Generator per case) and cover every cloud-overlap
      option 1..5, both idcor settings, both juldat branches, negative
      latitude, WRF's production permuteseeds (LW 150 / SW 1) plus
      off-nominal seeds, cldfrac values at 0, below cldmin (1e-21),
      mid-range, and exactly 1.

  (then, in WSL, for each case:  ./mcica_fixture_lw|sw mcica-<case>.in
      mcica-<case>.out )

  python make_mcica_fixtures.py package WORKDIR OUTDIR
      Reads each input + Fortran output dump and writes
      OUTDIR/mcica_<case>.npz containing the exact float32/int32 inputs
      and the oracle outputs (in_/out_ key prefixes).

The committed npz files under tests/data/rrtmg_mcica/ are the oracle
gate; this script plus build.sh regenerate them from the unmodified WRF
Fortran.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

NBNDLW = 16
NBNDSW = 14

# case name -> (side, ncol, nlay, icld, permuteseed, irng, idcor, juldat,
#               lat, rng_seed)
CASES = {
    "lw01": ("lw", 8, 52, 2, 150, 0, 0, 94, 35.19, 20260701),
    "lw02": ("lw", 3, 30, 1, 150, 0, 0, 94, 35.19, 20260702),
    "lw03": ("lw", 1, 12, 3, 150, 0, 0, 94, 35.19, 20260703),
    "lw04": ("lw", 4, 26, 4, 150, 0, 0, 94, 35.19, 20260704),
    "lw05": ("lw", 4, 26, 4, 150, 0, 1, 94, 35.19, 20260705),
    "lw06": ("lw", 5, 20, 5, 150, 0, 1, 200, -12.5, 20260706),
    "lw07": ("lw", 2, 16, 2, 97, 0, 0, 94, 35.19, 20260707),
    "lw08": ("lw", 2, 18, 5, 150, 0, 0, 94, 35.19, 20260708),
    "sw01": ("sw", 8, 52, 2, 1, 0, 0, 94, 35.19, 20260711),
    "sw02": ("sw", 3, 30, 1, 1, 0, 0, 94, 35.19, 20260712),
    "sw03": ("sw", 1, 12, 3, 1, 0, 0, 94, 35.19, 20260713),
    "sw04": ("sw", 4, 26, 4, 1, 0, 0, 94, 35.19, 20260714),
    "sw05": ("sw", 4, 26, 4, 1, 0, 1, 94, 35.19, 20260715),
    "sw06": ("sw", 5, 20, 5, 1, 0, 1, 200, -12.5, 20260716),
    "sw07": ("sw", 2, 16, 2, 41, 0, 0, 94, 35.19, 20260717),
    "sw08": ("sw", 2, 18, 5, 1, 0, 0, 94, 35.19, 20260718),
}


def make_inputs(case: str) -> dict[str, np.ndarray]:
    side, ncol, nlay, icld, permuteseed, irng, idcor, juldat, lat, seed = \
        CASES[case]
    rng = np.random.default_rng(seed)
    # Bottom-up columns: play (mb) strictly decreasing, hgt (m) increasing.
    p_surf = rng.uniform(96.0, 103.5, size=ncol) * 10.0
    p_top = rng.uniform(2.5, 6.0, size=ncol) * 10.0
    frac = np.linspace(0.0, 1.0, nlay)[None, :]
    play = p_surf[:, None] * (p_top[:, None] / p_surf[:, None]) ** frac
    play = play + rng.uniform(-0.03, 0.03, size=(ncol, nlay))
    play = np.minimum.accumulate(play - 1e-3 * np.arange(nlay)[None, :],
                                 axis=1)
    hgt = 40.0 + np.cumsum(rng.uniform(90.0, 750.0, size=(ncol, nlay)),
                           axis=1)
    cldfrac = rng.uniform(0.0, 1.0, size=(ncol, nlay))
    zero_mask = rng.uniform(size=(ncol, nlay)) < 0.35
    cldfrac[zero_mask] = 0.0
    tiny_mask = (~zero_mask) & (rng.uniform(size=(ncol, nlay)) < 0.08)
    cldfrac[tiny_mask] = 1.0e-21          # below cldmin -> treated as clear
    full_mask = (~zero_mask) & (rng.uniform(size=(ncol, nlay)) < 0.06)
    cldfrac[full_mask] = 1.0
    ciwp = np.where(cldfrac > 0, rng.uniform(0.0, 60.0, (ncol, nlay)), 0.0)
    clwp = np.where(cldfrac > 0, rng.uniform(0.0, 90.0, (ncol, nlay)), 0.0)
    cswp = np.where(rng.uniform(size=(ncol, nlay)) < 0.3,
                    rng.uniform(0.0, 40.0, (ncol, nlay)), 0.0)
    rei = rng.uniform(5.0, 140.0, (ncol, nlay))
    rel = rng.uniform(2.5, 60.0, (ncol, nlay))
    res = rng.uniform(10.0, 300.0, (ncol, nlay))
    nbnd = NBNDLW if side == "lw" else NBNDSW
    tauc = rng.uniform(0.0, 25.0, (nbnd, ncol, nlay))
    fields = {
        "play": play, "hgt": hgt, "cldfrac": cldfrac, "ciwp": ciwp,
        "clwp": clwp, "cswp": cswp, "rei": rei, "rel": rel, "res": res,
        "tauc": tauc,
    }
    if side == "sw":
        fields["ssac"] = rng.uniform(0.05, 1.0, (nbnd, ncol, nlay))
        fields["asmc"] = rng.uniform(0.0, 0.95, (nbnd, ncol, nlay))
        fields["fsfc"] = rng.uniform(0.0, 0.9, (nbnd, ncol, nlay))
    fields = {k: np.asarray(v, dtype=np.float32) for k, v in fields.items()}
    header = {
        "side": side, "ncol": ncol, "nlay": nlay, "icld": icld,
        "permuteseed": permuteseed, "irng": irng, "idcor": idcor,
        "juldat": juldat, "lat": np.float32(lat),
    }
    return header, fields


def write_input(path: Path, header: dict, fields: dict) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack(
            ">7i", header["ncol"], header["nlay"], header["icld"],
            header["permuteseed"], header["irng"], header["idcor"],
            header["juldat"]))
        # NOTE: numpy *scalars* silently drop the byte-order request in
        # astype(">f4"); struct.pack is the reliable big-endian spelling.
        f.write(struct.pack(">f", float(np.float32(header["lat"]))))
        order = ["play", "hgt", "cldfrac", "ciwp", "clwp", "cswp",
                 "rei", "rel", "res", "tauc"]
        if header["side"] == "sw":
            order += ["ssac", "asmc", "fsfc"]
        for name in order:
            f.write(np.asfortranarray(fields[name]).astype(">f4").tobytes(
                order="F"))


def read_dump(path: Path):
    entries = {}
    with open(path, "rb") as f:
        while True:
            raw = f.read(4)
            if not raw:
                break
            (nlen,) = struct.unpack(">i", raw)
            name = f.read(nlen).decode("ascii")
            (dtype_code,) = struct.unpack(">i", f.read(4))
            (rank,) = struct.unpack(">i", f.read(4))
            dims = struct.unpack(f">{rank}i", f.read(4 * rank)) if rank \
                else ()
            count = int(np.prod(dims)) if rank else 1
            kind = "f4" if dtype_code == 4 else "i4"
            data = np.frombuffer(f.read(4 * count), dtype=">" + kind)
            arr = data.astype(kind)
            entries[name] = arr.reshape(dims, order="F") if rank else \
                arr.reshape(())
    return entries


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    mode = sys.argv[1]
    workdir = Path(sys.argv[2])
    if mode == "emit-inputs":
        workdir.mkdir(parents=True, exist_ok=True)
        for case in CASES:
            header, fields = make_inputs(case)
            write_input(workdir / f"mcica-{case}.in", header, fields)
        (workdir / "mcica-cases.json").write_text(
            json.dumps({k: list(v) for k, v in CASES.items()}, indent=1))
        print(f"wrote {len(CASES)} inputs under {workdir}")
        print("now run, per case, inside WSL from the build dir:")
        print("  ./mcica_fixture_lw mcica-lwNN.in mcica-lwNN.out")
        print("  ./mcica_fixture_sw mcica-swNN.in mcica-swNN.out")
    elif mode == "package":
        if len(sys.argv) != 4:
            raise SystemExit(__doc__)
        outdir = Path(sys.argv[3])
        outdir.mkdir(parents=True, exist_ok=True)
        for case, spec in CASES.items():
            header, fields = make_inputs(case)
            dump = read_dump(workdir / f"mcica-{case}.out")
            payload = {
                "side": np.bytes_(header["side"]),
                "ncol": np.int32(header["ncol"]),
                "nlay": np.int32(header["nlay"]),
                "icld": np.int32(header["icld"]),
                "permuteseed": np.int32(header["permuteseed"]),
                "irng": np.int32(header["irng"]),
                "idcor": np.int32(header["idcor"]),
                "juldat": np.int32(header["juldat"]),
                "lat": np.float32(header["lat"]),
            }
            for name, value in fields.items():
                payload[f"in_{name}"] = value
            for name, value in dump.items():
                payload[f"out_{name.split('/')[-1]}"] = value
            np.savez_compressed(outdir / f"mcica_{case}.npz", **payload)
        print(f"packaged {len(CASES)} fixtures under {outdir}")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
