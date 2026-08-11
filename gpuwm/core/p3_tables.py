"""P3 (``mp_physics=50``) lookup-table packaging, loading, and generation.

Transcription authority: byte-frozen WRF v4.6.1
``phys/module_mp_p3.F`` (P3 v4.5.2), subroutine ``p3_init`` lines 112-685.
``p3_init`` does three table jobs for the 1-category / 2-moment-ice
configuration this port ships:

1. READ the main ice lookup table file
   ``p3_lookupTable_1.dat-v5.4_2momI`` (:337-410): a version-checked header
   (:351-366) followed by, for each (density jj=1..5, rime-fraction
   ii=1..4) block, 50 lines of 3 index tokens + 14 ``itab`` process
   quantities (:371-384) and then 50x30 lines of 3 index tokens + 2
   double-precision ice-rain collection quantities stored into the
   single-precision ``itabcoll`` (:387-396).  The byte-identical file from
   the reference bundle's ``run/`` directory is committed under
   ``gpuwm/data/p3/tables`` and SHA-256-validated before parsing, the same
   packaging pattern as Thompson's classic tables.

2. GENERATE the rain fallspeed / ventilation tables ``vn_table``,
   ``vm_table``, ``revap_table`` (300 x 10) by numerically integrating the
   rain PSD over 10 000 size bins (:599-670), and fill ``mu_r_table`` with
   the constant ``mu_r_constant = 0`` (:596; the variable-``mu_r``
   iteration is commented out in the authority).  The generation here is
   float32 throughout with the Fortran's per-bin sequential accumulation
   order preserved (a NumPy ``sum`` would use pairwise reduction and
   change the roundoff); the ``mu_r`` loop columns are all computed with
   ``mu_r = mu_r_constant`` in the authority (:610), so the 10 columns are
   identical by construction and are computed once and broadcast.

3. The multi-category table 2 and the 3-moment table-1 variant are NOT
   packaged: they belong to the unported ``mp_physics=52`` / ``53``
   options, which gpuwm refuses by name (gpuwm/config.py).

Precision note (documented, not a divergence by design): the Fortran
list-directed READ converts the decimal text directly to REAL(4);
this loader parses to float64 and casts, which can in principle
double-round the last ULP of a value whose decimal representation falls
within half a float32 ULP of a rounding boundary.  The committed table
carries 5 significant digits, far from any such boundary, and the oracle
campaign (declared next stage) measures the port end to end.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

#: Environment override for the table root (byte-identical mirrors only:
#: a root with different bytes fails closed in :func:`load_lookup_table_1`).
P3_TABLE_ROOT_ENV = "GPUWM_P3_TABLE_ROOT"

#: Lookup-table array dimensions (module_mp_p3.F:48-57).
ISIZE = 50        # normalized-q index
DENSIZE = 5       # bulk rime density index
RIMSIZE = 4       # rime mass fraction index
RCOLLSIZE = 30    # rain-size index (ice-rain collection)
TABSIZE = 14      # quantities per itab entry
COLLTABSIZE = 2   # quantities per itabcoll entry

#: p3_init's intended table-1 version for the 2-moment-ice configuration
#: (module_mp_p3.F:138); the file header is checked against it (:351-366)
#: and a mismatch is a hard stop under model='WRF' (:362-365).
TABLE_1_2MOM_VERSION = "5.4_2momI"
TABLE_1_2MOM_FILENAME = f"p3_lookupTable_1.dat-v{TABLE_1_2MOM_VERSION}"


@dataclass(frozen=True)
class P3TableAsset:
    filename: str
    sha256: str
    size: int


#: The one packaged asset, pinned to the byte-frozen reference bundle copy
#: (WRF_source_v4.6.1_group/run/p3_lookupTable_1.dat-v5.4_2momI).
TABLE_1_2MOM_ASSET = P3TableAsset(
    filename=TABLE_1_2MOM_FILENAME,
    sha256="5cb47f2ad15b726abeacac21b5877c737178526051661473a465315348867c2f",
    size=1637040,
)


def packaged_p3_table_root() -> Path:
    """The in-package canonical P3 table directory."""
    return Path(__file__).resolve().parent.parent / "data" / "p3" / "tables"


def p3_table_root() -> str:
    """Resolved mp=50 table root: env override, then the packaged directory.

    Unlike Thompson there is no user-staged fallback yet: the single asset
    is 1.6 MiB and ships inside the wheel, so the site-packages copy is
    always present on a healthy install.
    """
    override = os.environ.get(P3_TABLE_ROOT_ENV)
    if override:
        return override
    return str(packaged_p3_table_root())


def _validate_asset_bytes(path: Path, asset: P3TableAsset) -> bytes:
    """Return the file bytes after exact size + SHA-256 validation.

    A missing or wrong-byte table is a hard data dependency failure and
    must never fall back silently: P3's process rates ARE these tables.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"P3 lookup table missing: {path}. mp_physics=50 cannot run "
            f"without {asset.filename} (P3 reads it at init, "
            "module_mp_p3.F:337-410). The canonical copy ships inside the "
            "gpuwm package; set " + P3_TABLE_ROOT_ENV + " only to point at "
            "a byte-identical mirror.")
    blob = path.read_bytes()
    if len(blob) != asset.size:
        raise ValueError(
            f"P3 lookup table {path} has size {len(blob)}, expected "
            f"{asset.size}. Refusing to parse a modified table.")
    digest = hashlib.sha256(blob).hexdigest()
    if digest != asset.sha256:
        raise ValueError(
            f"P3 lookup table {path} SHA-256 {digest} does not match the "
            f"pinned {asset.sha256}. Refusing to parse a modified table.")
    return blob


@lru_cache(maxsize=2)
def load_lookup_table_1(root: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse table 1 (2-moment ice): ``(itab, itabcoll)`` float32 arrays.

    Shapes follow the Fortran declarations (module_mp_p3.F:62/:66):
    ``itab[densize, rimsize, isize, tabsize]`` and
    ``itabcoll[densize, rimsize, isize, rcollsize, colltabsize]``, 0-based.
    The read order matches p3_init exactly (:371-398): per (jj, ii) block,
    50 itab records of 3 skipped index tokens + 14 values, then 50 x 30
    itabcoll records of 3 skipped index tokens + 2 values.  itabcoll is
    parsed as float64 and stored to float32, matching the Fortran's
    ``dp_dum1`` -> REAL(4) assignment (:392-394).
    """
    path = Path(root) / TABLE_1_2MOM_ASSET.filename
    blob = _validate_asset_bytes(path, TABLE_1_2MOM_ASSET)
    text = blob.decode("ascii")
    first_newline = text.index("\n")
    header = text[:first_newline].split()
    # read(10,*) dumstr,version_header_table_1_2mom  (:351)
    if len(header) < 2 or header[1] != TABLE_1_2MOM_VERSION:
        raise ValueError(
            f"P3 lookupTable_1 header names version "
            f"{header[1] if len(header) > 1 else '<missing>'}; P3 v4.5.2 "
            f"requires {TABLE_1_2MOM_VERSION} (module_mp_p3.F:351-366, a "
            "hard stop under model='WRF').")
    tokens = text[first_newline:].split()
    per_block = ISIZE * (3 + TABSIZE) + ISIZE * RCOLLSIZE * (3 + COLLTABSIZE)
    expected = DENSIZE * RIMSIZE * per_block
    if len(tokens) != expected:
        raise ValueError(
            f"P3 lookupTable_1 has {len(tokens)} data tokens, expected "
            f"{expected}; refusing a truncated or reshaped table.")
    values = np.asarray(tokens, dtype=np.float64)
    itab = np.empty((DENSIZE, RIMSIZE, ISIZE, TABSIZE), dtype=np.float32)
    itabcoll = np.empty((DENSIZE, RIMSIZE, ISIZE, RCOLLSIZE, COLLTABSIZE),
                        dtype=np.float32)
    pos = 0
    itab_block = ISIZE * (3 + TABSIZE)
    coll_block = ISIZE * RCOLLSIZE * (3 + COLLTABSIZE)
    for jj in range(DENSIZE):
        for ii in range(RIMSIZE):
            rows = values[pos:pos + itab_block].reshape(ISIZE, 3 + TABSIZE)
            itab[jj, ii, :, :] = rows[:, 3:].astype(np.float32)
            pos += itab_block
            rows = values[pos:pos + coll_block].reshape(
                ISIZE, RCOLLSIZE, 3 + COLLTABSIZE)
            itabcoll[jj, ii, :, :, :] = rows[:, :, 3:].astype(np.float32)
            pos += coll_block
    return itab, itabcoll


@lru_cache(maxsize=1)
def generate_rain_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rain fallspeed/ventilation tables ``(vn, vm, revap)``, each (300, 10).

    Transcription of module_mp_p3.F:599-670 in float32.  ``mu_r`` is the
    constant 0 for every column of the ``mu_r_loop`` (:610 with
    ``mu_r_constant = 0``, :219), so the 10 columns are identical; the
    single column is computed with the Fortran's sequential per-bin
    accumulation order (each bin's contribution added in kk order, exactly
    the scalar loop's roundoff) and broadcast.  With ``mu_r = 0`` the
    Fortran's overflow-guard factors ``10.**(mu_r*alog10(dia)+4.*mu_r)``
    collapse to exactly 1.0 (0*x == 0 in FP32 for finite x, 10**0 == 1),
    while the mu_r+1/mu_r+3 exponent terms remain genuine float32
    log10/pow round trips and are transcribed as written (:652-656).
    """
    f32 = np.float32
    thrd = f32(1.0) / f32(3.0)
    sxth = f32(1.0) / f32(6.0)
    pi = f32(3.14159265)
    piov6 = pi * sxth
    mu_r = f32(0.0)  # mu_r_constant (:219)

    jj = np.arange(1, 301, dtype=np.float32)
    # mean size [m] (:615-619)
    dm = np.where(jj <= 20.0,
                  (jj * f32(10.0) - f32(5.0)) * f32(1.0e-6),
                  ((jj - f32(20.0)) * f32(30.0) + f32(195.0)) * f32(1.0e-6))
    dm = dm.astype(np.float32)
    lamr = (mu_r + f32(1.0)) / dm

    dum1 = np.zeros(300, dtype=np.float32)
    dum2 = np.zeros(300, dtype=np.float32)
    dum3 = np.zeros(300, dtype=np.float32)
    dum4 = np.zeros(300, dtype=np.float32)
    dum5 = np.zeros(300, dtype=np.float32)
    dd = f32(2.0)

    for kk in range(1, 10001):
        dia = (f32(kk) * dd - dd * f32(0.5)) * f32(1.0e-6)  # size bin [m]
        # amg = piov6*997.*dia**3 (:636): gfortran expands the integer power
        # as (dia*dia)*dia, then multiplies by the (piov6*997.) product.
        amg = (piov6 * f32(997.0)) * ((dia * dia) * dia)
        amg = amg * f32(1000.0)                             # kg -> g
        d_um = dia * f32(1.0e6)
        if d_um <= f32(134.43):
            vt = f32(4.5795e3) * amg ** (f32(2.0) * thrd)
        elif d_um < f32(1511.64):
            vt = f32(4.962e1) * amg ** thrd
        elif d_um < f32(3477.84):
            vt = f32(1.732e1) * amg ** sxth
        else:
            vt = f32(9.17)
        log_dia = np.log10(dia)
        pow0 = f32(10.0) ** (mu_r * log_dia + f32(4.0) * mu_r)
        pow3 = f32(10.0) ** ((mu_r + f32(3.0)) * log_dia + f32(4.0) * mu_r)
        pow1 = f32(10.0) ** ((mu_r + f32(1.0)) * log_dia + f32(3.0) * mu_r)
        decay = np.exp(-lamr * dia)
        # Association preserved: ...*exp(-lamr*dia)*dd*1.e-6 groups as
        # ((term*decay)*dd)*1e-6 in the Fortran (:652-656).
        dum1 += vt * pow0 * decay * dd * f32(1.0e-6)
        dum2 += pow0 * decay * dd * f32(1.0e-6)
        dum3 += vt * pow3 * decay * dd * f32(1.0e-6)
        dum4 += pow3 * decay * dd * f32(1.0e-6)
        dum5 += (vt * dia) ** f32(0.5) * pow1 * decay * dd * f32(1.0e-6)

    dum2 = np.maximum(dum2, f32(1.0e-30))
    dum4 = np.maximum(dum4, f32(1.0e-30))
    dum5 = np.maximum(dum5, f32(1.0e-30))

    vn_col = (dum1 / dum2).astype(np.float32)
    vm_col = (dum3 / dum4).astype(np.float32)
    revap_col = (f32(10.0) ** (np.log10(dum5)
                               + (mu_r + f32(1.0)) * np.log10(lamr)
                               - f32(3.0) * mu_r)).astype(np.float32)

    vn = np.repeat(vn_col[:, None], 10, axis=1)
    vm = np.repeat(vm_col[:, None], 10, axis=1)
    revap = np.repeat(revap_col[:, None], 10, axis=1)
    return np.ascontiguousarray(vn), np.ascontiguousarray(vm), \
        np.ascontiguousarray(revap)
