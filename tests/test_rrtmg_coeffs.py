"""Bit-for-bit oracle gate for gpuwm.ingest.rrtmg_coeffs.

The authority is a dump written by a Fortran program that USEs the
unmodified WRF v4.6.1 RRTMG modules after running the real init path
(tools/rrtmg_wrf461_oracle/coeffs_dump_lw.F90 / _sw.F90, built by
build.sh).  The committed form of that authority is the per-array SHA-256
manifest gpuwm/data/wrf_radiation/rrtmg_coeffs_oracle_manifest.json,
generated from the dumps by tools/rrtmg_wrf461_oracle/
make_coeffs_manifest.py.  Every loader output must match its manifest
entry exactly (dtype, extents, byte image), which is bit-for-bit
equivalence with the Fortran state.  When the raw dump files themselves
are present (RRTMG_ORACLE_BUILD_DIR or the default oracle build path),
the direct byte comparison runs as well.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np
import pytest

from gpuwm.ingest import rrtmg_coeffs as rc


REPO = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO / "gpuwm" / "data" / "wrf_radiation" / \
    "rrtmg_coeffs_oracle_manifest.json"


def _manifest():
    return json.loads(MANIFEST_PATH.read_text())


def _array_sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _loaded(side: str):
    if side == "lw":
        return rc.load_rrtmg_lw_coefficients()
    return rc.load_rrtmg_sw_coefficients()


def test_packaged_data_files_match_wrf_authority_sha256():
    for name, sha in (("RRTMG_LW_DATA", rc.RRTMG_LW_DATA_SHA256),
                      ("RRTMG_SW_DATA", rc.RRTMG_SW_DATA_SHA256)):
        digest = hashlib.sha256((rc.DATA_DIR / name).read_bytes()).hexdigest()
        assert digest == sha, f"{name} does not match the WRF authority"


def test_contract_module_and_variable_inventory_is_frozen():
    manifest = _manifest()
    expected = set(manifest["arrays"])
    got = set()
    for side in ("lw", "sw"):
        for module, arrays in _loaded(side).items():
            for var in arrays:
                got.add(f"{module}/{var}")
    assert got == expected, (
        "loader inventory diverged from the oracle manifest: "
        f"missing={sorted(expected - got)[:10]} "
        f"extra={sorted(got - expected)[:10]}")
    # The frozen contract: 16 LW + 14 SW band modules.
    lw = rc.load_rrtmg_lw_coefficients()
    sw = rc.load_rrtmg_sw_coefficients()
    assert sorted(lw) == [f"rrlw_kg{i:02d}" for i in range(1, 17)]
    assert sorted(sw) == [f"rrsw_kg{i}" for i in range(16, 30)]


@pytest.mark.parametrize("side", ["lw", "sw"])
def test_every_array_matches_the_fortran_oracle_bit_for_bit(side):
    manifest = _manifest()["arrays"]
    mismatches = []
    checked = 0
    for module, arrays in _loaded(side).items():
        for var, value in arrays.items():
            entry = manifest[f"{module}/{var}"]
            if str(value.dtype) != entry["dtype"] or \
                    list(value.shape) != entry["shape"]:
                mismatches.append(
                    f"{module}/{var}: shape/dtype {value.shape} "
                    f"{value.dtype} != {entry['shape']} {entry['dtype']}")
                continue
            if _array_sha256(value) != entry["sha256"]:
                mismatches.append(f"{module}/{var}: byte image differs")
                continue
            checked += 1
    assert not mismatches, mismatches[:20]
    assert checked == sum(1 for k in manifest if
                          k.startswith("rrlw" if side == "lw" else "rrsw"))


def _read_dump(path: Path):
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


def _oracle_build_dir() -> Path | None:
    env = os.environ.get("RRTMG_ORACLE_BUILD_DIR")
    candidates = [Path(env)] if env else []
    candidates.append(Path.home() / ".claude" / "jobs" / "fea7a141" /
                      "tmp" / "rrtmg_foundation" / "build")
    for candidate in candidates:
        if (candidate / "rrtmg-coeffs-lw.dump").exists():
            return candidate
    return None


@pytest.mark.parametrize("side", ["lw", "sw"])
def test_direct_dump_comparison_when_oracle_dumps_present(side):
    build = _oracle_build_dir()
    if build is None:
        pytest.skip("oracle dump files not on this machine; the committed "
                    "SHA manifest gate above is the portable equivalent")
    dump = _read_dump(build / f"rrtmg-coeffs-{side}.dump")
    manifest = _manifest()
    digest = hashlib.sha256(
        (build / f"rrtmg-coeffs-{side}.dump").read_bytes()).hexdigest()
    assert digest == manifest["dumps"][f"rrtmg-coeffs-{side}.dump"], (
        "local oracle dump is not the manifest's dump; regenerate the "
        "manifest or the dump")
    for module, arrays in _loaded(side).items():
        for var, value in arrays.items():
            ref = dump[f"{module}/{var}"]
            assert ref.dtype == value.dtype and ref.shape == value.shape
            assert np.array_equal(
                ref.view(np.int32) if ref.dtype == np.float32 else ref,
                value.view(np.int32) if value.dtype == np.float32
                else value), f"{module}/{var} differs from the dump"


def test_absa_absb_are_exact_equivalence_views():
    lw = rc.load_rrtmg_lw_coefficients()
    ka = lw["rrlw_kg03"]["ka"]           # (9, 5, 13, ng3)
    absa = lw["rrlw_kg03"]["absa"]       # (585, ng3)
    assert absa.shape == (585, ka.shape[-1])
    assert np.array_equal(
        absa.view(np.int32),
        ka.reshape((-1, ka.shape[-1]), order="F").view(np.int32))


def test_lower_bound_metadata_names_the_kbo_pressure_axis():
    assert rc.fortran_lower_bounds("rrlw_kg01", "kbo") == (1, 13, 1)
    assert rc.fortran_lower_bounds("rrsw_kg17", "kb") == (1, 1, 13, 1)
    assert rc.fortran_lower_bounds("rrsw_kg16", "rayl") == ()
    assert rc.fortran_lower_bounds("rrlw_kg01", "absa") == (1, 1)
