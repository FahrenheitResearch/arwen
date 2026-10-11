"""Execute a registered Thompson statement generation as full GPU columns."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import os
from pathlib import Path

import numpy as np

from gpuwm.core.thompson_contract import (AUXILIARY_TABLE_RECORDS, GENERATED_TABLE_FILES, TableRecord, read_sequential_records)

GENERATION_ASSETS = {
    "wrf_44": (
        ("qr_acr_qgV3.dat",89959776,"cc474317b05b7ac35daa81a88556cfa1fe2150a8aa915efcff441649717ea866"),
        ("qr_acr_qsV2.dat",43764288,"04da562439bf1e0e11af3d30d4c3eb9ddcc599730fa0034f4e4268535062e4d0"),
        ("freezeH2O.dat",254944848,"d80d4d1d270535831a3b2c1161a135ace5ca3e26b2d852d8177efdf3a93a59cf"),
    ),
}
AUXILIARY_ASSET = ("wrf_44_auxiliary.dat",244520,"10b86addf627d81c7cd4a76efcf9d548d210a2004330f285db7cf28b3d39b4f0")


def table_root(cfg):
    value = getattr(cfg,"thompson_table_dir",None) or os.environ.get("GPUWM_THOMPSON_GENERATION_TABLE_ROOT")
    if value is None:
        raise ValueError("the selected Thompson generation needs its matching coefficient directory; set thompson_table_dir to the WRF run folder containing qr_acr_qgV3.dat, qr_acr_qsV2.dat and freezeH2O.dat")
    return Path(value).resolve()


def auxiliary_path():
    return Path(__file__).resolve().parents[1] / "data/thompson/generations" / AUXILIARY_ASSET[0]


def _verify(path, expected):
    name, size, digest = expected
    if not path.is_file() or path.stat().st_size != size:
        raise ValueError(f"Thompson generation table {path} must contain exactly {size} bytes; a different record layout changes the lookup indices")
    actual = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8*1024*1024):
            actual.update(block)
    if actual.hexdigest() != digest:
        raise ValueError(f"Thompson generation table {path} has SHA-256 {actual.hexdigest()}, expected {digest}; mixing generations changes the microphysics rates")
    return {"filename":name,"bytes":size,"sha256":digest}


@lru_cache(None)
def _identity(version, root, signatures):
    from gpuwm.thompson_generations import CLASSIC_GENERATIONS
    assets = [_verify(Path(root)/row[0],row) for row in GENERATION_ASSETS[version]]
    assets.append(_verify(auxiliary_path(),AUXILIARY_ASSET))
    return {"admission":"generated-classic-thompson-column-v1","generation":version,
            "source_sha256":CLASSIC_GENERATIONS[version].source_sha256,"assets":assets}


def table_identity(cfg):
    root = table_root(cfg)
    paths = [root/row[0] for row in GENERATION_ASSETS[cfg.thompson_version]] + [auxiliary_path()]
    signatures = tuple((str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in paths)
    return _identity(cfg.thompson_version,str(root),signatures)

TABLE_ARGUMENTS = (
    "tcg_racg", "tmr_racg", "tcr_gacr", "tnr_racg", "tnr_gacr",
    "tcs_racs1", "tmr_racs1", "tcs_racs2", "tmr_racs2",
    "tcr_sacr1", "tms_sacr1", "tcr_sacr2", "tms_sacr2",
    "tnr_racs1", "tnr_racs2", "tnr_sacr1", "tnr_sacr2",
    "tpi_qrfz", "tni_qrfz", "tpg_qrfz", "tnr_qrfz", "tpi_qcfz", "tni_qcfz",
    "tps_iaus", "tni_iaus", "tpi_ide", "t_Efrw", "t_Efsw",
)


@lru_cache(None)
def tables(root, device):
    import cupy as cp
    records = tuple(TableRecord(n, (37,37,37,37)) for n in ("tcg_racg", "tmr_racg", "tcr_gacr", "tmg_gacr", "tnr_racg", "tnr_gacr"))
    data = read_sequential_records(Path(root) / "qr_acr_qgV3.dat", records, byteorder="big")
    for name in ("qr_acr_qsV2.dat", "freezeH2O.dat"):
        data.update(read_sequential_records(Path(root) / name, GENERATED_TABLE_FILES[name], byteorder="big"))
    data.update(read_sequential_records(auxiliary_path(), AUXILIARY_TABLE_RECORDS[:5]))
    owner = {name: cp.asarray(data[name], dtype=cp.float64, order="F") for name in TABLE_ARGUMENTS}
    cp.cuda.get_current_stream().synchronize()
    return owner


def apply_generated(state, cfg, dt, *, refl_10cm_due=False):
    import cupy as cp
    from gpuwm.core import constants as C
    from gpuwm.core.kernels import get_kernel, get_kernel_int_defines
    from gpuwm.core.microphysics import MicrophysicsDiagnostics
    from gpuwm.core.thompson_aerosol_state import launch_aerosol_exner

    nz, ny, nx = state.p.shape
    if nz > 256:
        raise ValueError("generated Thompson column capacity is 256 levels; a larger column would overwrite private working arrays")
    table_identity(cfg)
    root = table_root(cfg)
    owner = tables(str(root), cp.cuda.Device().id)
    shape = state.p.shape
    th, pii, temperature, dz = [state.scratch(shape, name) for name in ("mp_th", "mp_pii", "mp_thompson_temperature", "mp_dz8w")]
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]
    th[...] = thb + state.thp
    state.h_diabatic[...] = th
    launch_aerosol_exner(state.p, pii)
    temperature[...] = th * pii
    z = state.scratch((nz+1,ny,nx), "mp_z8w")
    z[...] = (phb + state.php) / np.float32(C.G)
    dz[...] = z[1:] - z[:-1]
    # Literal slot names: the restart manifest classifies every scratch slot
    # (tests/test_restart.py harvests the literals), and a built "mp_" + name
    # would hide these serialized accumulators from that check.
    surface = {
        "rainnc": state.scratch((ny,nx), "mp_rainnc"),
        "rainncv": state.scratch((ny,nx), "mp_rainncv"),
        "snownc": state.scratch((ny,nx), "mp_snownc"),
        "snowncv": state.scratch((ny,nx), "mp_snowncv"),
        "graupelnc": state.scratch((ny,nx), "mp_graupelnc"),
        "graupelncv": state.scratch((ny,nx), "mp_graupelncv"),
        "sr": state.scratch((ny,nx), "mp_sr"),
    }
    refl = state.scratch(shape, "refl_10cm")
    fields = [getattr(state, s) for s in ("qv", "qc", "qi", "qr", "qs", "qg", "ni", "nr")]
    args = (*fields, temperature, state.p, state.w, dz,
            *(surface[n] for n in ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc", "graupelncv", "sr")),
            state.effc, state.effi, state.effs, refl, np.int32(refl_10cm_due), np.int32(nz), np.int32(ny*nx), np.float32(dt),
            *(owner[n] for n in TABLE_ARGUMENTS))
    capacity = next(k for k in (64,128,256) if nz <= k)
    kernel = (get_kernel("thompson_column", "thompson_column_step") if capacity == 64
              else get_kernel_int_defines("thompson_column","thompson_column_step",(("THOMPSON_COLUMN_KMAX",capacity),)))
    kernel(((ny*nx+31)//32,), (32,), args)
    th[...] = temperature / pii
    if not cfg.no_mp_heating:
        limit = np.float32(cfg.mp_tend_lim) * np.float32(dt)
        increment = cp.clip(th - state.h_diabatic, -limit, limit)
        state.thp[...] += increment
        state.h_diabatic[...] = increment / np.float32(dt)
    else:
        state.h_diabatic.fill(0)
    if refl_10cm_due:
        state.physics.refl_10cm = refl
    return MicrophysicsDiagnostics(**surface)
