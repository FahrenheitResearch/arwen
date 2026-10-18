"""Pack the SPP column-oracle captures into fixtures, and read them back.

The CSV captures from ``build.py`` are tens of megabytes of decimal text.
The fixture keeps every value as the exact float32 (or int32) word the CSV
spells, column for column, in compressed ``.npz`` files:

* ``mynn-spp-ml{1,2}.npz``: every column of ``run_mynn_spp_columns``'s CSV,
  one array per CSV column over all rows, plus the ``case`` labels.
* ``gf-spp-o{offset}.npz``: the per-level fields of ``run_gf_spp_columns``
  folded to ``(ncol, nz)`` and its per-column fields, both in the order of
  the GF fixture's column key, plus the raw closure patterns.

``mynn_steps`` rebuilds the row-dictionary blocks that
``tests/_mynn_families.py`` reads, so the families helpers (``step_inputs``,
``step_outputs``, ``advance``) serve this oracle unchanged.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "spp" / "columns"
MYNN_NZ = 30
GF_NZ = 40
#: Fields the WOOF GF kernel returns at the CUP_gf seam (after neg_check).
GF_SEAM_LEVELS = ("outt", "outq", "outqc", "outu", "outv", "outts", "outqs", "outqcs")
#: The prepared column the harness echoes; it must equal the GF fixture's.
GF_ECHO_LEVELS = ("zo", "po", "t2d", "q2d", "tn", "qo", "tshall", "qshall",
                  "dhdt", "us", "vs", "rhoi", "omeg_in")
GF_SURFACE = ("pret", "prets", "xmb_out", "xmbs", "edt")
GF_SURFACE_INT = ("ierr", "kbcon", "ktop", "ierrs", "kbcons", "ktops")
_MYNN_INT = {"step", "k", "initflag", "kpbl_in", "kpbl", "ktop_plume"}


def _word(text: str) -> np.float32:
    return np.float32(float(text))


def pack_mynn(csv_path: Path, npz_path: Path) -> None:
    with Path(csv_path).open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    arrays = {"case": np.asarray([row["case"] for row in rows])}
    for name in rows[0]:
        if name == "case":
            continue
        if name in _MYNN_INT:
            arrays[name] = np.asarray([int(row[name]) for row in rows], np.int32)
        else:
            arrays[name] = np.asarray([_word(row[name]) for row in rows], np.float32)
    np.savez_compressed(npz_path, **arrays)


def mynn_steps(npz_path: Path) -> tuple[list[str], dict[int, list[list[dict]]]]:
    """``(case labels, {step: [rows of column 0, ..., rows of column N-1]})``."""
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    cases = list(dict.fromkeys(arrays["case"].tolist()))
    names = [name for name in arrays if name != "case"]
    nrow = arrays["case"].shape[0]
    rows = [{name: arrays[name][r] for name in names} for r in range(nrow)]
    for r, row in enumerate(rows):
        row["case"] = arrays["case"][r]
    steps: dict[int, list[list[dict]]] = {}
    position = {case: i for i, case in enumerate(cases)}
    for row in rows:
        block = steps.setdefault(int(row["step"]), [[] for _ in cases])
        block[position[str(row["case"])]].append(row)
    for blocks in steps.values():
        assert all(len(block) == MYNN_NZ for block in blocks)
        for block in blocks:
            assert [int(row["k"]) for row in block] == list(range(1, MYNN_NZ + 1))
    return cases, dict(sorted(steps.items()))


def _read(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="ascii") as stream:
        return list(csv.DictReader(stream))


def pack_gf(levels_csv: Path, surface_csv: Path, key: np.ndarray, npz_path: Path) -> None:
    """Fold to the GF fixture's column order ``key`` (case, idx, arm)."""
    order = {tuple(int(v) for v in row): i for i, row in enumerate(key)}
    ncol = key.shape[0]
    levels, surface = _read(levels_csv), _read(surface_csv)
    out: dict[str, np.ndarray] = {"key": np.asarray(key, np.int32)}
    for name in GF_SEAM_LEVELS + GF_ECHO_LEVELS:
        out[name] = np.zeros((ncol, GF_NZ), np.float32)
    for row in levels:
        c = order[(int(row["case"]), int(row["idx"]), int(row["arm"]))]
        k = int(row["k"]) - 1
        for name in GF_SEAM_LEVELS + GF_ECHO_LEVELS:
            out[name][c, k] = _word(row[name])
    for name in GF_SURFACE:
        out[name] = np.zeros(ncol, np.float32)
    for name in GF_SURFACE_INT:
        out[name] = np.zeros(ncol, np.int32)
    out["spp"] = np.zeros((ncol, 4), np.float32)
    seen = set()
    for row in surface:
        c = order[(int(row["case"]), int(row["idx"]), int(row["arm"]))]
        seen.add(c)
        for name in GF_SURFACE:
            out[name][c] = _word(row[name])
        for name in GF_SURFACE_INT:
            out[name][c] = int(row[name])
        out["spp"][c] = [_word(row[f"spp{n}"]) for n in (1, 2, 3, 4)]
    if len(seen) != ncol:
        raise ValueError("GF SPP capture does not cover every fixture column")
    np.savez_compressed(npz_path, **out)


def load_gf(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def regime(case: str) -> dict[str, str]:
    """Family, elevation and pattern shape of a MYNN column label."""
    from spp_column_sources import FAMILIES, PATTERN, TERRAIN
    family, terrain, pattern = case.rsplit("_", 2)
    assert family in FAMILIES
    return {"family": family, "terrain": TERRAIN[int(terrain[1:])],
            "pattern": PATTERN[int(pattern[1:])]}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
