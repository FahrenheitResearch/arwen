"""Read the chem WRF v4.7.1 oracle fixtures and measure a port against them.

The fixtures are written by ``tools/chem_wrf471_oracle/oracle_io.F90``: one
directory per case, one raw little-endian ``<name>.bin`` per array and a
``MANIFEST.txt`` naming each array's kind and Fortran extents.  :func:`load`
returns ``{case: {name: ndarray}}`` with Fortran index order preserved
(``order='F'``), so a WRF ``(i, k, j)`` array is indexed ``[i, k, j]`` here
and a parity test transposes it to gpuwm's ``(k, j, i)`` in plain sight.

Fixtures live under ``tests/data/oracles/chem/<lane>/``, each with a
``PROVENANCE.md`` naming the WRF tree, the compiler and the SHA-256 of every
source that built it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from gpuwm.core.fp32_ulp import fp32_ulp_distance

ORACLE_ROOT = Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "chem"

_KINDS = {"f4": np.dtype("<f4"), "i4": np.dtype("<i4")}


def load_case(case_dir: str | Path) -> dict[str, np.ndarray]:
    """One case directory -> ``{name: ndarray}`` in Fortran index order."""
    case_dir = Path(case_dir)
    out: dict[str, np.ndarray] = {}
    for line in (case_dir / "MANIFEST.txt").read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        name, kind, rank = parts[0], parts[1], int(parts[2])
        shape = tuple(int(v) for v in parts[3:3 + rank])
        dtype = _KINDS[kind]
        raw = np.fromfile(case_dir / f"{name}.bin", dtype=dtype)
        expected = int(np.prod(shape)) if shape else 1
        if raw.size != expected:
            raise ValueError(
                f"{case_dir.name}/{name}: {raw.size} values on disk, the "
                f"manifest says {shape}")
        out[name] = (raw.reshape(shape, order="F").astype(dtype.newbyteorder("="))
                     if shape else raw.astype(dtype.newbyteorder("="))[0])
    return out


def load(directory: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """Every case under ``directory`` (sub-directories with a MANIFEST)."""
    directory = Path(directory)
    cases = {}
    for child in sorted(directory.iterdir()):
        if (child / "MANIFEST.txt").is_file():
            cases[child.name] = load_case(child)
    if not cases:
        raise FileNotFoundError(f"no oracle cases under {directory}")
    return cases


def ulp_table(port, ref) -> dict[str, int]:
    """``{"max_ulp", "n_nonzero", "n"}`` of a float32 port against the oracle.

    NaN against NaN is 0 ULP; NaN against a number is the largest distance
    :func:`gpuwm.core.fp32_ulp.fp32_ulp_distance` reports, so it can never
    hide.  Integer arrays compare exactly (distance 0 or 1 per element).
    """
    port = np.asarray(port)
    ref = np.asarray(ref)
    if port.shape != ref.shape:
        raise ValueError(f"shape {port.shape} against oracle {ref.shape}")
    if np.issubdtype(ref.dtype, np.integer):
        diff = (port.astype(np.int64) != ref.astype(np.int64)).astype(np.int64)
    else:
        diff = np.asarray(fp32_ulp_distance(port.astype(np.float32),
                                            ref.astype(np.float32)),
                          dtype=np.int64)
    return {"max_ulp": int(diff.max()) if diff.size else 0,
            "n_nonzero": int(np.count_nonzero(diff)), "n": int(diff.size)}


__all__ = ["ORACLE_ROOT", "load", "load_case", "ulp_table"]
