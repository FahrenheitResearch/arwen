"""Loader for the pinned New Tiedtke WRF v4.6.1 oracle.

The CSVs under ``gpuwm/data/ntiedtke/oracle/`` are recorded from
byte-unmodified WRF v4.6.1 by ``tools/ntiedtke_wrf461_oracle/build.sh``;
that directory's README pins the three source digests and explains why
``-DRWORDSIZE=4`` is load-bearing.

Every float in those files is a raw IEEE-754 word in hex, not a decimal
rendering, because the bar is ``max_ulp == 0`` and a decimal rendering is a
lossy view of the thing being pinned.  :func:`word` is therefore the only
way values enter: it reconstructs the exact float32 the reference held.
"""
from __future__ import annotations

import csv
import struct
from functools import lru_cache
from pathlib import Path

import numpy as np

__all__ = [
    "ORACLE_DIR", "word", "words", "load_csv",
    "prep_inputs", "prep_expected", "prep_surface", "stage_a_surface",
    "NT_NZ", "NT_CASES", "NT_DXSWEEP", "NT_DT", "NT_STEPCU",
    "NT_ITIMESTEP", "NT_GRAV",
]

ORACLE_DIR = Path(__file__).resolve().parents[1] / "data" / "ntiedtke" / "oracle"

#: Fixture geometry, mirroring tools/ntiedtke_wrf461_oracle/nt_cases.F90.
NT_NZ = 49
NT_CASES = 18
NT_DXSWEEP = (1500.0, 4500.0, 9000.0, 13500.0, 15000.0, 27000.0)

#: The scalars run_nt_prep.F90 drives the fixture with.
NT_DT = np.float32(60.0)
NT_STEPCU = 1
NT_ITIMESTEP = 2                 # > 1, so qvften/thften are READ not zeroed
NT_GRAV = np.float32(9.81)


def word(hex8: str) -> np.float32:
    """The float32 whose IEEE-754 bits are ``hex8``."""
    return np.float32(struct.unpack("<f", struct.pack("<I", int(hex8, 16)))[0])


def words(hex_list) -> np.ndarray:
    return np.array([word(h) for h in hex_list], dtype=np.float32)


@lru_cache(maxsize=None)
def load_csv(name: str) -> tuple[dict, ...]:
    with open(ORACLE_DIR / name, newline="", encoding="utf-8") as fh:
        return tuple(dict(r) for r in csv.DictReader(fh))


def _by_col(rows, fields, nk):
    """Group hex rows into ``{(case, dx): {field: float32[nk]}}``."""
    out: dict[tuple[int, float], dict[str, np.ndarray]] = {}
    for r in rows:
        key = (int(r["case"]), float(r["dx"]))
        k = int(r["k"]) - 1
        if k >= nk:
            continue
        slot = out.setdefault(
            key, {f: np.zeros(nk, dtype=np.float32) for f in fields})
        for f in fields:
            slot[f][k] = word(r[f])
    return out


#: WRF-order inputs the prep was driven with.  p8w and w carry nz+1 entries;
#: everything else is a half-level array and its (nz+1)th row is padding that
#: run_nt_prep.F90 writes as zero and nothing may read.
_INPUT_HALF = ("t3d", "qv3d", "qc3d", "qi3d", "u3d", "v3d",
               "pcps", "dz8w", "rho3d", "pi3d", "qvften", "thften")
_INPUT_FULL = ("p8w", "w")

#: Scheme-order (TOP-DOWN) outputs.  prsi/ghti are nz+1 arrays whose last
#: entry -- the SURFACE interface after the flip -- is not captured here;
#: nt-prep-consistency.csv proves the whole prep exact end to end, so that
#: entry is validated transitively even though it is not graded directly.
_OUTPUT = ("prsl", "ghtl", "omg", "tf", "qvf", "qcf", "qif", "uf", "vf",
           "qvftenz", "thftenz", "prsi", "ghti")


def prep_inputs():
    rows = load_csv("nt-prep-input.csv")
    half = _by_col(rows, _INPUT_HALF, NT_NZ)
    full = _by_col(rows, _INPUT_FULL, NT_NZ + 1)
    for key, slot in half.items():
        slot.update(full[key])
    return half


def prep_expected():
    return _by_col(load_csv("nt-prep-levels.csv"), _OUTPUT, NT_NZ)


def prep_surface():
    out = {}
    for r in load_csv("nt-prep-surface.csv"):
        out[(int(r["case"]), float(r["dx"]))] = {
            "slimsk": int(r["slimsk"]),
            "dx_hv": word(r["dx_hv"]),
            "hfx_hv": word(r["hfx_hv"]),
            "qfx_hv": word(r["qfx_hv"]),
            "delt": word(r["delt"]),
        }
    return out


def stage_a_surface():
    """``scale_fac``/``scale_fac2`` and ``xland``, from the Stage A fixture."""
    out = {}
    for r in load_csv("nt-surface.csv"):
        out[(int(r["case"]), float(r["dx"]))] = {
            "scale_fac": word(r["scale_fac"]),
            "scale_fac2": word(r["scale_fac2"]),
            "xland": word(r["xland"]),
            "hfx": word(r["hfx"]),
            "qfx": word(r["qfx"]),
            # p8w(1), the SURFACE interface.  After the flip it is
            # prsi(klev+1) -- the one entry of prsi the level fixture does
            # not carry, and cumastrn:509 reads it every step through
            # paph(jk+1) at jk = klev.
            "psfc": word(r["psfc"]),
        }
    return out


# ---------------------------------------------------------------------------
# Slice 2 fixtures: the conversion block and cuinin
# ---------------------------------------------------------------------------
# Recorded by run_nt_cuinin.F90, which reaches the module-PRIVATE cuinin
# through the symbol objcopy --globalize-symbol exposed.  That operation
# leaves .text byte-identical (build.sh asserts it every run and writes
# nt-globalize-receipt.txt), so these words are the pinned source's own.

#: cu_ntiedtke_run's conversion outputs -- cuinin's inputs, since cumastrn
#: passes its dummies straight through (:474-481).
_CONV = ("ztp1", "zqp1", "zqsat", "pgeo", "pgeoh", "pverv", "ptte", "pqte")

#: cuinin's per-level outputs.  pqsenh is captured but its index 0 is
#: UNDEFINED in the reference -- cuinin's jk loop starts at 2 (1-based) and
#: the tail block writes only ptenh(1)/pqenh(1) -- so callers must not grade
#: it there.  See :func:`cuinin_expected` and the test that skips it.
_CUININ = ("ptenh", "pqenh", "pqsenh", "ptu", "pqu", "ptd", "pqd",
           "puu", "pvu", "pud", "pvd", "plu")


def conv_expected():
    return _by_col(load_csv("nt-conv-levels.csv"), _CONV, NT_NZ)


def cuinin_expected():
    rows = load_csv("nt-cuinin-levels.csv")
    out = _by_col(rows, _CUININ, NT_NZ)
    for r in rows:
        key = (int(r["case"]), float(r["dx"]))
        k = int(r["k"]) - 1
        if k >= NT_NZ:
            continue
        out[key].setdefault("klab", np.zeros(NT_NZ, dtype=np.int32))
        out[key]["klab"][k] = int(r["klab"])
    return out


def cuinin_surface():
    return {(int(r["case"]), float(r["dx"])): int(r["klwmin"])
            for r in load_csv("nt-cuinin-surface.csv")}


#: cutypen's per-level outputs.  cutu/cuqu/culu/culab are intent(out) in
#: cutypen but READ before assignment (:1334-1337), and cumastrn passes
#: cuinin's own ptu/pqu/plu/ilab -- so the aliasing is load-bearing and the
#: harness reproduces it.  Getting that wrong is what the NumPy mirror
#: caught: a fixture built from fresh arrays disagreed with the mirror on
#: every shallow column.
_CUTYPEN = ("cutu", "cuqu", "culu")


def cutypen_expected():
    rows = load_csv("nt-cutypen-levels.csv")
    out = _by_col(rows, _CUTYPEN, NT_NZ)
    for r in rows:
        key = (int(r["case"]), float(r["dx"]))
        k = int(r["k"]) - 1
        if k >= NT_NZ:
            continue
        out[key].setdefault("culab", np.zeros(NT_NZ, dtype=np.int32))
        out[key]["culab"][k] = int(r["culab"])
    return out


def cutypen_surface():
    out = {}
    for r in load_csv("nt-cutypen-surface.csv"):
        out[(int(r["case"]), float(r["dx"]))] = {
            "ldcum": int(r["ldcum"]), "ktype": int(r["ktype"]),
            "cubot": int(r["cubot"]), "cutop": int(r["cutop"]),
            "kdpl": int(r["kdpl"]), "wbase": word(r["wbase"]),
        }
    return out


#: cubasmcn's per-level outputs, captured after cuascn's prologue.  ALL
#: THIRTEEN of cubasmcn's outputs are conditionally written, so the harness
#: passes the live pre-state arrays -- fresh ones would be wrong on every
#: non-triggering column.
_MIDLEVEL = ("ptu", "pqu", "plu", "pmfu", "pmfus", "pmfuq", "pmful",
             "pdmfup", "plrain")


def midlevel_expected():
    rows = load_csv("nt-midlevel-levels.csv")
    out = _by_col(rows, _MIDLEVEL, NT_NZ)
    for r in rows:
        key = (int(r["case"]), float(r["dx"]))
        k = int(r["k"]) - 1
        if k >= NT_NZ:
            continue
        out[key].setdefault("klab", np.zeros(NT_NZ, dtype=np.int32))
        out[key]["klab"][k] = int(r["klab"])
    return out


def midlevel_surface():
    return {(int(r["case"]), float(r["dx"])): {
                "ktype": int(r["ktype"]), "kcbot": int(r["kcbot"]),
                "pmfub": word(r["pmfub"])}
            for r in load_csv("nt-midlevel-surface.csv")}


def cuentrn_expected():
    """{(case, dx): {kk: (pdmfen, pdmfde)}}, kk 1-based as the Fortran."""
    out = {}
    for r in load_csv("nt-cuentrn.csv"):
        key = (int(r["case"]), float(r["dx"]))
        out.setdefault(key, {})[int(r["kk"])] = (word(r["pdmfen"]),
                                                word(r["pdmfde"]))
    return out


def mfub_surface():
    """cumastrn:500-541 -- zdhpbl, upbl, zmfub, and the possibly-cleared
    ldcum.  A prerequisite for cuascn, cudlfsn and the closure."""
    return {(int(r["case"]), float(r["dx"])): {
                "ldcum": int(r["ldcum"]),
                "zdhpbl": word(r["zdhpbl"]),
                "upbl": word(r["upbl"]),
                "zmfub": word(r["zmfub"])}
            for r in load_csv("nt-mfub-surface.csv")}
