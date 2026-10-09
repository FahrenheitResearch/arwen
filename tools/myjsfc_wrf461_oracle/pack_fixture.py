"""Pack the oracle's case and output streams into one .npz fixture.

    python pack_fixture.py BUILD_DIR OUT.npz

Keys, per set ``<s>`` named in BUILD_DIR/cases.txt:

* ``<s>/meta``           int32 [ncol, nz, nsteps, it0]
* ``<s>/in/<name>``      float32 inputs as the Fortran read them; column
  arrays are ``(nz, ncol)`` (``pint`` is ``(nz + 1, ncol)``), bottom-up
  exactly as WRF's DZ/PMID/... are (K = 1 is the lowest layer).
* ``<s>/out``            float32 ``(nsteps, NFIELD, ncol)``, the post-call
  words of every MYJSFC call, fields in ``FIELDS`` order.

and once, from the first set (the script refuses if another set's dump
differs): ``tables/<psim1|psih1|psim2|psih2>`` (KZTM words each) and
``tables/scalars`` (DZETA1, DZETA2, ZTMIN1, ZTMAX1, ZTMIN2, ZTMAX2, FH01,
FH02).  ``fields`` names the NFIELD records.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

INOUT = ("ust", "znt", "thz0", "qz0", "uz0", "vz0", "qsfc", "akhs", "akms")
OUTPUTS = ("rmol", "ct", "pblh", "rib", "chs", "chs2", "cqs2", "hfx", "qfx",
           "lh", "flhc", "flqc", "qgh", "cpm", "u10", "v10", "t2", "th2",
           "tshltr", "th10", "q2", "qshltr", "q10", "pshltr", "u10e", "v10e")
FIELDS = INOUT + OUTPUTS
COLUMN_INPUTS = ("dz", "pmid", "th", "t", "qv", "qc", "u", "v", "q2")
SURFACE_INPUTS = ("tsk", "xland", "mavail", "z0base")
STATE_INPUTS = ("qsfc", "thz0", "qz0", "uz0", "vz0", "ustar", "znt", "akhs",
                "akms")


def read_case(path: Path) -> dict:
    raw = path.read_bytes()
    ncol, nz, nsteps, it0 = np.frombuffer(raw, "<i4", 4)
    words = np.frombuffer(raw, "<f4", offset=16)
    at = 0
    case = {"meta": np.asarray([ncol, nz, nsteps, it0], np.int32)}

    def take(count):
        nonlocal at
        block = words[at:at + count].copy()
        at += count
        return block

    case["ht"] = take(ncol)
    for name in COLUMN_INPUTS:
        case[name] = take(nz * ncol).reshape(nz, ncol)
    case["pint"] = take((nz + 1) * ncol).reshape(nz + 1, ncol)
    for name in SURFACE_INPUTS + STATE_INPUTS:
        case[name] = take(ncol)
    if at != words.size:
        raise ValueError(f"{path}: {words.size - at} trailing words")
    return case


def read_out(path: Path) -> dict:
    raw = path.read_bytes()
    ncol, nz, nsteps, it0, kztm, nfield = np.frombuffer(raw, "<i4", 6)
    if nfield != len(FIELDS):
        raise ValueError(f"{path}: {nfield} fields, expected {len(FIELDS)}")
    words = np.frombuffer(raw, "<f4", offset=24)
    scalars = words[:8].copy()
    tables = words[8:8 + 4 * kztm].reshape(4, kztm).copy()
    body = words[8 + 4 * kztm:]
    if body.size != nsteps * nfield * ncol:
        raise ValueError(f"{path}: body has {body.size} words")
    return {"meta": np.asarray([ncol, nz, nsteps, it0], np.int32),
            "scalars": scalars, "tables": tables,
            "out": body.reshape(nsteps, nfield, ncol).copy()}


def pack(build_dir: Path, out_npz: Path) -> None:
    names = (build_dir / "cases.txt").read_text().split()
    arrays = {"fields": np.asarray(FIELDS)}
    first = None
    for name in names:
        case = read_case(build_dir / f"case_{name}.bin")
        out = read_out(build_dir / f"out_{name}.bin")
        if not np.array_equal(case["meta"], out["meta"]):
            raise ValueError(f"{name}: case/out headers disagree")
        if first is None:
            first = out
            arrays["tables/scalars"] = out["scalars"]
            for i, key in enumerate(("psim1", "psih1", "psim2", "psih2")):
                arrays[f"tables/{key}"] = out["tables"][i]
        elif (out["scalars"].tobytes() != first["scalars"].tobytes()
              or out["tables"].tobytes() != first["tables"].tobytes()):
            raise ValueError(f"{name}: MYJSFCINIT tables differ between runs")
        arrays[f"{name}/meta"] = case["meta"]
        for key, value in case.items():
            if key != "meta":
                arrays[f"{name}/in/{key}"] = value
        arrays[f"{name}/out"] = out["out"]
    np.savez_compressed(out_npz, **arrays)


if __name__ == "__main__":
    pack(Path(sys.argv[1]), Path(sys.argv[2]))
