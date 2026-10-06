"""Read the Noah mosaic WRF v4.7.1 oracle fixtures and measure a port against them.

The Fortran harness (``tools/noah_mosaic_wrf471_oracle/oracle_io.F90``) writes
one directory per case, one raw little-endian ``<name>.bin`` per array and a
``MANIFEST.txt`` naming each array's kind and Fortran extents.  The committed
corpus is those directories packed one file per case
(``tools/noah_mosaic_wrf471_oracle/pack_fixtures.py``): ``<case>.npz`` holds the
manifest text and every array's words unchanged.  :func:`load` reads either
form and returns ``{case: {name: ndarray}}`` with Fortran index order preserved
(``order='F'``), so a WRF ``(i, k, j)`` array is indexed ``[i, k, j]`` here
and a parity test transposes it to gpuwm's ``(k, j, i)`` in plain sight.

Fixtures live under ``tests/data/oracles/noah_mosaic/<family>/``, with a
``PROVENANCE.md`` naming the WRF tree, the compiler and the SHA-256 of every
source that built them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from gpuwm.core.fp32_ulp import fp32_ulp_distance
from gpuwm.verify.wrf471_fixtures import require_fixture_dir

# Test data in a source checkout, not package data (gpuwm.verify.wrf471_fixtures).
ORACLE_ROOT = Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "noah_mosaic"

_KINDS = {"f4": np.dtype("<f4"), "i4": np.dtype("<i4")}


def load_case(case_dir: str | Path) -> dict[str, np.ndarray]:
    """One case (packed ``.npz`` or raw directory) -> ``{name: ndarray}``.

    Arrays keep Fortran index order.
    """
    case_dir = Path(case_dir)
    packed = None
    if case_dir.suffix == ".npz":
        packed = np.load(case_dir, allow_pickle=False)
        manifest = bytes(packed["__manifest__"]).decode("ascii")
    else:
        manifest = (case_dir / "MANIFEST.txt").read_text(encoding="ascii")
    out: dict[str, np.ndarray] = {}
    for line in manifest.splitlines():
        parts = line.split()
        if not parts:
            continue
        name, kind, rank = parts[0], parts[1], int(parts[2])
        shape = tuple(int(v) for v in parts[3:3 + rank])
        dtype = _KINDS[kind]
        if packed is not None:
            raw = np.asarray(packed[name])
            if raw.dtype != dtype:
                raise ValueError(
                    f"{case_dir.name}/{name}: packed as {raw.dtype}, the "
                    f"manifest says {dtype}")
        else:
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
    """Every case under ``directory``: ``<case>.npz`` packs or raw directories."""
    directory = require_fixture_dir(directory, "Noah mosaic")
    cases = {}
    for child in sorted(directory.iterdir()):
        if child.suffix == ".npz" and child.is_file():
            cases[child.stem] = load_case(child)
        elif (child / "MANIFEST.txt").is_file():
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


def wrf_to_gpuwm(a):
    """WRF (i,k,j) -> gpuwm (k,j,i), and (i,j) -> (j,i)."""
    a = np.asarray(a)
    if a.ndim == 3:
        return np.ascontiguousarray(a.transpose(1, 2, 0))
    if a.ndim == 2:
        return np.ascontiguousarray(a.T)
    return a.copy()


def gpuwm_to_wrf(a):
    """The inverse layout conversion, including device-to-host transfer."""
    a = a.get() if hasattr(a, "get") else np.asarray(a)
    if a.ndim == 3:
        return np.asfortranarray(a.transpose(2, 0, 1))
    if a.ndim == 2:
        return np.asfortranarray(a.T)
    return a.copy()


def soil_reduction(tiles, fractions):
    """D1 corrected reduction in WRF layout, float32, reverse tile order."""
    tiles = np.asarray(tiles, dtype=np.float32)
    fractions = np.asarray(fractions, dtype=np.float32)
    mc = fractions.shape[1]
    out = np.zeros((tiles.shape[0], 4, tiles.shape[2]), dtype=np.float32)
    for t in range(mc - 1, -1, -1):
        for ns in range(4):
            out[:, ns, :] = out[:, ns, :] + tiles[:, 4*t + ns, :] * fractions[:, t, :]
    return out


def wrf_defective_soil_reduction(before, after, fractions):
    """Reproduce D1 with the state present at each WRF accumulation.

    module_sf_noahdrv.F:4151-4153 executes INSIDE the reverse tile loop.
    Its defective index can read an earlier tile's still-unmodified state.
    Using only final tile outputs is not a valid control for this defect.
    """
    mc = fractions.shape[1]
    out = np.zeros((after.shape[0], 4, after.shape[2]), dtype=np.float32)
    for t in range(mc - 1, -1, -1):
        for ns in range(4):
            k = (ns + 1)*(t + 1) - 1
            source = after if k//4 >= t else before
            out[:, ns, :] = out[:, ns, :] + source[:, k, :] * fractions[:, t, :]
    return out


def weighted_twin_increments(twins, fractions, field):
    """D2 corrected addition using single-tile oracle increments."""
    out = np.zeros_like(twins[0][field + "_out"], dtype=np.float32)
    for t in range(len(twins) - 1, -1, -1):
        value = twins[t][field + "_out"]
        # SNOPCX's twin result is negative. WRF forms the positive sx then
        # subtracts its weighted sum only after the loop.
        increment = -value if field == "snopcx" else value
        out = out + increment * fractions[:, t, :]
    return -out if field == "snopcx" else out


def fixture_device_fields(fixture, columns=None):
    """Build device fields from WRF input words with conversions in one place."""
    import cupy as cp
    from gpuwm.core.noah import _F2D, _F3D
    from gpuwm.core.noah_mosaic import MOSAIC_TILE_FIELDS, MOSAIC_SOIL_FIELDS
    if columns is None:
        columns = np.arange(fixture["tsk_in"].shape[0])
    def convert(a):
        return cp.asarray(wrf_to_gpuwm(np.asarray(a)[columns]))
    out = {}
    pressure = fixture["p8w3d_in"]
    derived = dict(psfc=pressure[:, 0, :],
                   sfcprs=(pressure[:, 1, :] + pressure[:, 0, :]) * np.float32(.5),
                   sfctmp=fixture["t3d_in"][:, 0, :], qv1=fixture["qv3d_in"][:, 0, :],
                   dz8w1=fixture["dz8w_in"][:, 0, :])
    for n in ("ivgtyp", "isltyp", *[n for n in _F2D if n != "reslin"], *_F3D,
              *MOSAIC_TILE_FIELDS, *MOSAIC_SOIL_FIELDS, "mosaic_cat_index", "landusef2"):
        a = derived[n] if n in derived else fixture[n + "_in"]
        # WRF category arrays retain NLCAT; gpuwm retains only the tile prefix.
        if n in ("mosaic_cat_index", "landusef2"):
            a = a[:, :int(fixture["mosaic_cat"]), :]
        out[n] = convert(a)
    return out
