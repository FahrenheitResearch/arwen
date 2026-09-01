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

MEASURED against running Fortran, 2026-08-28 (node-1, gfortran 15.2.0,
WRF's own ``module_mp_p3.F`` compiled unmodified except for an appended
dump routine, ``p3_init(nCat=1, trplMomI=.false., model='WRF')``):

* ``itab`` (14,000 float32) and ``itabcoll`` (60,000 float32) parsed by
  :func:`load_lookup_table_1` are BYTE-IDENTICAL to the module variables
  ``p3_init`` fills -- ``tobytes()`` equality, zero differing words, at
  both -O0 and -O2.  The double-rounding risk the paragraph below names
  is therefore not realised on this table; the digests are pinned as
  :data:`FORTRAN_ITAB_SHA256` / :data:`FORTRAN_ITABCOLL_SHA256`.
* :func:`generate_rain_tables` is NOT byte-identical: ``vn_table`` differs
  in 1,150 of 3,000 entries (max 5 ULP, max relative 3.3e-07),
  ``vm_table`` in 940 (max 4 ULP, 4.2e-07) and ``revap_table`` in 110
  (max 21 ULP, 2.2e-06).  Root cause MEASURED, not inferred: the
  transcription's structure and accumulation order are right (``lamr`` is
  byte-identical), and the divergence is host libm.  Over the loop's own
  10,000 arguments, numpy's float32 ``exp`` differs from glibc ``expf``
  on 5,824 of them by up to 3 ULP, and ``10.**(3*log10(dia))`` differs on
  705 by up to 36 ULP -- the latter being a 1-ULP ``log10`` difference
  amplified by ``ln(10)`` times an exponent of magnitude ~18.  A
  byte-identical rain table needs a correctly-rounded libm on this path
  (gpuwm.core.correctly_rounded_libm) or the generated tables shipped as
  data; neither is decided here, and the divergence is DECLARED rather
  than tolerated by a loosened comparison.

Precision note (documented, not a divergence by design): the Fortran
list-directed READ converts the decimal text directly to REAL(4);
this loader parses to float64 and casts, which can in principle
double-round the last ULP of a value whose decimal representation falls
within half a float32 ULP of a rounding boundary.  The committed table
carries 5 significant digits, far from any such boundary -- and the
measurement above confirms it: every one of the 74,000 parsed values
matches the Fortran bit for bit.
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

#: The record separator a Fortran formatted sequential READ advances on.
#: The packaged table carries CRLF, so the CR stays on the record and is
#: trailing blank to both the Fortran read and ``bytes.split``.
_RECORD_SEPARATOR = b"\n"

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


class P3TableVersionError(ValueError):
    """The table's own header names a version P3 v4.5.2 does not intend.

    Its own class because it is a DIFFERENT operator action from a corrupt
    file: upstream checks the header and nothing else, and under
    ``model='WRF'`` a mismatch is an unconditional ``stop``
    (module_mp_p3.F:351-366).  A ValueError subclass so every existing
    caller that already refuses on ValueError keeps refusing.
    """


@dataclass(frozen=True)
class P3TableAsset:
    filename: str
    sha256: str
    size: int
    #: The version token p3_init requires in this file's own first record
    #: (``read(10,*) dumstr,version_header_...``, module_mp_p3.F:351).
    #: Carried as DATA so a second table (the 3-moment table-1 variant, or
    #: table 2 for nCat>1) is a new row here and not a new code path.
    version: str


#: The one packaged asset, pinned to the byte-frozen reference bundle copy
#: (WRF_source_v4.6.1_group/run/p3_lookupTable_1.dat-v5.4_2momI).
TABLE_1_2MOM_ASSET = P3TableAsset(
    filename=TABLE_1_2MOM_FILENAME,
    sha256="5cb47f2ad15b726abeacac21b5877c737178526051661473a465315348867c2f",
    size=1637040,
    version=TABLE_1_2MOM_VERSION,
)

#: The SAME table as WRF v4.6.1 ships in ``run/``, byte for byte EXCEPT that
#: the packaged copy carries CRLF line endings: 1,606,038 bytes / sha-256
#: ``be1ab6fb...`` upstream against 1,637,040 / ``5cb47f2a...`` here, a
#: difference of exactly 31,002 bytes for 31,002 records.  MEASURED
#: 2026-08-28: WRF's own ``p3_init`` reading either file produces
#: byte-identical ``itab`` and ``itabcoll``, because a Fortran list-directed
#: READ treats the CR as trailing blank.  Recorded here, and named in the
#: digest refusal below, because ``GPUWM_P3_TABLE_ROOT`` invites an operator
#: to point at a mirror and the most obvious mirror on any machine is WRF's
#: own ``run/`` directory -- which this pin refuses.  Whether to ACCEPT the
#: upstream bytes as well is a restart-identity decision, not a loader one:
#: the digest is written into every mp=50 checkpoint's setup identity
#: (gpuwm/io/restart.py::_p3_setup_identity), so repinning to the upstream
#: bytes would make every existing mp=50 checkpoint refuse to resume.
UPSTREAM_WRF_TABLE_1_2MOM_SHA256 = (
    "be1ab6fb03481e376e47c6c79d808af5d8ab069f2b242931e9c54801bad4ae84")
UPSTREAM_WRF_TABLE_1_2MOM_SIZE = 1606038

#: SHA-256 of the PARSED arrays -- C-contiguous little-endian float32
#: ``itab[5,4,50,14]`` and ``itabcoll[5,4,50,30,2]`` -- as produced by WRF's
#: own ``p3_init`` compiled with gfortran 15.2.0 and dumped from the module
#: variables.  This is the parser's correctness anchor: it pins the numbers
#: the Fortran actually loads, not the text this loader happens to read.
#: Measured 2026-08-28 on node-1, identical at -O0 and -O2 and identical
#: from the upstream LF file and the packaged CRLF file.
FORTRAN_ITAB_SHA256 = (
    "45390820659a05f80c50482dacb8c886c93c379fe7c10f468fc1d9216ababaa5")
FORTRAN_ITABCOLL_SHA256 = (
    "7fe622d9eb2a6e495959f2bd87542c05948ffef92b51648e104ca1e7f27da3ee")


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


def _header_version(blob: bytes, path: Path, asset: P3TableAsset) -> str:
    """The version token from the table's first record, or a refusal.

    ``p3_init`` reads exactly this: ``read(10,*) dumstr,version_header_...``
    at module_mp_p3.F:351, i.e. the SECOND whitespace-delimited token of the
    first record.  Transcribed at that granularity so a file whose header
    cannot supply one is named as such instead of failing later as a shape
    error two hundred thousand tokens in.
    """
    head = blob.split(_RECORD_SEPARATOR, 1)[0]
    try:
        tokens = head.decode("ascii").split()
    except UnicodeDecodeError:
        raise P3TableVersionError(
            f"P3 lookup table {path} does not begin with an ASCII header "
            f"record. p3_init reads the version token from the first line "
            "(module_mp_p3.F:351); this file is not a P3 lookup table."
        ) from None
    if len(tokens) < 2:
        raise P3TableVersionError(
            f"P3 lookup table {path} has no version token in its first "
            f"record (read {tokens!r}). p3_init reads "
            "``dumstr, version_header`` there (module_mp_p3.F:351), so a "
            "file without it is not a P3 lookup table.")
    return tokens[1]


def _validate_asset_bytes(path: Path, asset: P3TableAsset) -> bytes:
    """Return the file bytes after version + exact size + SHA-256 validation.

    A missing, wrong-version or wrong-byte table is a hard data dependency
    failure and must never fall back silently: P3's process rates ARE these
    tables.

    ORDER MATTERS, and it is upstream's order.  ``p3_init`` checks the
    header version and nothing else, so the version check runs FIRST here
    too.  Behind the digest it was unreachable: every non-pinned file --
    including a correctly formed table of a DIFFERENT version -- failed on
    the digest and was reported as "modified", which sends an operator who
    staged v5.3, or the 3-moment table under this name, to look for file
    corruption that is not there.  Measured 2026-08-28: a copy of the
    packaged table with only its header version rewritten to ``5.3_2momI``
    -- same length, same token count, a legitimately different table as far
    as any operator is concerned -- was refused with "Refusing to parse a
    modified table" and the version refusal below never ran.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"P3 lookup table missing: {path}. mp_physics=50 cannot run "
            f"without {asset.filename} (P3 reads it at init, "
            "module_mp_p3.F:337-410). The canonical copy ships inside the "
            "gpuwm package; set " + P3_TABLE_ROOT_ENV + " only to point at "
            "a byte-identical mirror.")
    blob = path.read_bytes()
    found = _header_version(blob, path, asset)
    if found != asset.version:
        raise P3TableVersionError(
            f"P3 lookup table {path} names version {found!r}; P3 v4.5.2 "
            f"requires {asset.version!r} (module_mp_p3.F:138, checked at "
            ":351-366 where a mismatch under model='WRF' is an "
            "unconditional stop, not a warning). The table encodes the "
            "ice process rates the scheme integrates: a different version "
            "is different physics, and the 3-moment variant is a "
            "different SHAPE (15 quantities on an 11-deep mu_i axis) that "
            "would reshape into garbage. gpuwm refuses rather than run "
            "P3 v4.5.2 against a table it was not built for.")
    digest = hashlib.sha256(blob).hexdigest()
    # Recognise WRF's OWN copy before the size check, not after it: the two
    # files differ in size (by exactly their CR bytes), so a size-first
    # order would report "wrong size" for the most likely mirror anybody
    # points GPUWM_P3_TABLE_ROOT at and never reach this sentence.
    if (asset is TABLE_1_2MOM_ASSET
            and digest == UPSTREAM_WRF_TABLE_1_2MOM_SHA256):
        raise ValueError(
            f"P3 lookup table {path} is WRF's own run/ copy (LF line "
            f"endings, {UPSTREAM_WRF_TABLE_1_2MOM_SIZE} B). It is the SAME "
            "table: gpuwm packages it with CRLF, and p3_init loads "
            "byte-identical itab/itabcoll from either. gpuwm still refuses "
            "it, because the packaged digest is written into every mp=50 "
            "checkpoint's setup identity (gpuwm/io/restart.py::"
            "_p3_setup_identity) and accepting a second digest silently "
            "would make two different byte sets share one identity. Use "
            "the packaged copy (unset " + P3_TABLE_ROOT_ENV + "); see "
            "gpuwm/data/p3/PROVENANCE.md.")
    if len(blob) != asset.size:
        raise ValueError(
            f"P3 lookup table {path} has size {len(blob)}, expected "
            f"{asset.size}. Refusing to parse a modified table.")
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
    # Version FIRST, then size, then digest -- upstream's order, and the
    # only order in which a wrong-VERSION table is reported as one
    # (:func:`_validate_asset_bytes`).
    blob = _validate_asset_bytes(path, TABLE_1_2MOM_ASSET)
    text = blob.decode("ascii")
    first_newline = text.index("\n")
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
