"""The fast SBM lookup tables: SHA-256 validated, parsed as WRF's READ does.

Transcribes the table reads of ``FAST_HUCMINIT`` (:5384-6236) and
``BREAKINIT_KS`` (:8644-8780) in WRF v4.8.0 ``phys/module_mp_fast_sbm.F``
(LF SHA-256
``fd72158202a98d4ad618ebb608eeadffd50852a9875f8bfd9bb8ad63c15f1a2e``,
identical to v4.6.1).  The READ that consumes each file is at the line
``READ_LINES`` records (:5445, :5500, :5548, :5598, :5648, :5694,
:5799-5912, :5988, :6030, :8725, :8755); the arrays they fill are declared
at :3935-3964 and sized by the ALLOCATE statements beside each READ; the
``YWLL`` zeroing is :5698-5706.  WRF is NCAR/UCAR public domain; the notice
is in ``NOTICE`` and ``gpuwm/data/fast_sbm/LICENSE-WRF.txt``.  The tables come
from NCAR's ``SBM_input_33`` distribution; ``PROVENANCE.md`` beside them
records the archive and per-file pins.

What the loader promises
------------------------
* Every one of the 18 files is checked against its pinned size and SHA-256
  before any of them is parsed, and a difference of any byte refuses the
  load and names the file.  A copy that differs only by CRLF line endings
  is named as such (the P3 table picked up CR bytes through a Windows copy
  tool; this refusal is what keeps that from recurring silently).
* ``GPUWM_FSBM_TABLE_ROOT`` may point at another directory, but only at a
  byte-identical mirror: the same pins apply.
* Arrays are zero-based and keep Fortran index order: ``xi[i-1, j-1]`` is
  ``XI(I,J)``.  A READ list that names several arrays fills them in turn,
  each in column-major order, from one stream of values.
* ``precision="float64"`` gives each decimal correctly rounded to double,
  which is what the promoted oracle build (``-freal-4-real-8``) reads.
  ``precision="float32"`` gives what WRF's native build holds: each decimal
  correctly rounded to float32 in one step, as gfortran's REAL*4 READ does
  (computed here with exact integer arithmetic, never by rounding to double
  first), except for the four arrays WRF itself declares
  ``REAL(KIND=R8SIZE)`` (``bin_mass``, ``tab_colum``, ``tab_dendr`` and
  ``tab_snow``, :3936-3937), which stay float64 in both views because the
  native build reads them as double too.

READ formats
------------
* ``FORMAT(6E13.5)`` (:5499): six 13-column fields per record; the value
  stream continues across the arrays of one READ, so a record can hold the
  end of one array and the start of the next.  The last record of each
  file is short.  Every field here carries an explicit decimal point and
  exponent, so the descriptor's implied scale never applies; a field
  without a decimal point is refused rather than guessed.
* List-directed (``READ(unit,*)``): ``bulkradii33.asc`` is one READ of the
  231 values of ``RADXXO(33,7)`` (:6030).  ``BLKD_SDC.dat`` is 33 READs of
  four values each, one record per READ (:5444-5447); the file has 43 data
  records and a blank one, and the last ten are never read.  Its exponents
  are written with ``D``.
* ``(3I6,1E16.8)`` and ``(2I6,1E16.8)`` (:8725, :8755): each record states
  its own indices and WRF stores the value at the stated position, not at
  the loop position.  Positions are checked to lie inside the array and,
  for ``PKIJ``, to satisfy j <= i; each position is filled once, and every
  position no record names stays at the zero WRF initialises it to
  (:6131-6132).

Tables read but never used by the scheme
----------------------------------------
* ``SLIC``, ``TLIC`` and ``COEFIN`` (``constants33.asc``, :5648): no
  statement in the module reads them after the READ.
* ``RADXXO`` (``bulkradii33.asc``, :6030): likewise.
* ``bin_mass``, ``tab_colum``, ``tab_dendr`` and ``tab_snow``
  (``BLKD_SDC.dat``): only the polarimetric operator reads them (:5197).
* ``YWLG`` is declared at :3947 but never allocated or read; the drop-graupel
  kernels are ``YWLG_300MB``, ``YWLG_500MB`` and ``YWLG_750MB``, exposed here
  as ``ywlg``.

The ``YWLL`` zeroing at :5698-5706 sets entries with an index above 33 to
zero; with 33 bins no such entry exists, and the statement below is the
same no-op.

This module is the verification authority for the table values, not a
production data path: the production reader moves to a byte-verified binary
pack in milestone 2 of the fast-sbm design.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import (
    FsbmRefusal,
    ICEMAX,
    JBREAK,
    NKR,
)

#: Where NCAR serves the archive the packaged files were extracted from.
ARCHIVE_URL = "https://www2.mmm.ucar.edu/wrf/src/wrf_files/SBM_input_33.tgz"
#: SHA-256 of that archive, and its size in bytes.
ARCHIVE_SHA256 = (
    "25ed63ab3550aed110ae24df229faece20eafedfdf21cde01d8973826a7294bf")
ARCHIVE_BYTES = 1_339_105

#: Environment variable naming a byte-identical mirror of the table directory.
TABLE_ROOT_ENV = "GPUWM_FSBM_TABLE_ROOT"

#: The packaged table directory.
PACKAGED_ROOT = (Path(__file__).resolve().parents[2]
                 / "gpuwm" / "data" / "fast_sbm" / "SBM_input_33")

#: Size in bytes and SHA-256 of each file the scheme opens (fast-sbm design,
#: appendix A).  These pins, not ``MANIFEST.sha256``, are the authority.
TABLE_PINS: dict[str, tuple[int, str]] = {
    "BLKD_SDC.dat": (
        2280,
        "6992497ee5b95f0fd215b762ad4af8611079d1e13dbe3db78687fe04a128abf2"),
    "capacity33.asc": (
        3042,
        "33759dbb0e238b37403ce2ad62759cca57c03538d72d34c2cc04bb7b2547c951"),
    "masses33.asc": (
        3042,
        "f53d3acdaa0a3e2a265272fa22a2a0bd54292f51815149370bbf315e3fa1bee4"),
    "termvels33_corrected.asc": (
        3042,
        "bf4209175492a3019563671676787afc3befd948fddc6435db58f7a8d7f702e3"),
    "constants33.asc": (
        3911,
        "d3d7cf345ebf850fa12376a7f173ea12b169f6413910939682494bcc5874a4fa"),
    "kernLL_z33.asc": (
        43016,
        "ce4e140595ff6db2434ecd5376a16c50ea410d03ddc1700bfa95a3490c15106f"),
    "ckli_33_300mb_500mb_750mb.asc": (
        129047,
        "59ce82054f29df7d9f982249b915201f3f0385ae559c5ef2b7cdb1907a46f81a"),
    "cklg_33_300mb_500mb_750mb.asc": (
        43016,
        "f3e8149a7e1bac947706f945a1d2ed0fb81ef2737404a0ef8c8c0c682fb4290f"),
    "cklh_33_300mb_500mb_750mb.asc": (
        43016,
        "a2ffe0b96e4acfcd06dc1a0b10cb28f60601ebe255f8e00a70ce1797705fb50e"),
    "ckls_33_300mb_500mb_750mb.asc": (
        43016,
        "ae282329a306d384bd6de958f1e0bf745c496d286237e8d22c85926081e0456a"),
    "ckii_33_300mb_500mb_750mb.asc": (
        387140,
        "546777c064e7687bbaaaf30ab417dfc146e2decaabc41b270439e33328a75c7f"),
    "ckis_33_300mb_500mb_750mb.asc": (
        129047,
        "657428fd7bbc0dcf5e7810bb0e25af42645f6122a4b24313404b805cfcdec44d"),
    "cksg_33_300mb_500mb_750mb.asc": (
        43016,
        "85b5fd667242ad216dd1b06f0f1c5ddfe15d797ae6c14dcc45e1097743bb37b9"),
    "ckss_33_300mb_500mb_750mb.asc": (
        43016,
        "1946554a0e1341cd12c9ae6be2cd2c786bfc3e4622b8fe87f2e85fcc64373847"),
    "bulkdens33.asc": (
        3042,
        "a5607b3d0b9eb1fcdba30e50b714e348a81aabb64d963b0a00ff4f7b2a24caa6"),
    "bulkradii33.asc": (
        3512,
        "d432d8eed8760f5000c6c86ac1b602687feb0783f558adbf10811fe7040a9de0"),
    "coeff_p_new_33.dat": (
        107730,
        "0fc78f82cfed4033eab9bcaf3f5d6bf07a4f1bbbbf3aff708ede6a0b2e921e34"),
    "coeff_q_new_33.dat": (
        9396,
        "cbda1e0fb2c604d6c8d61ad0abc22be7b1552be93bf5daed10b5186c4ba07d9e"),
}

#: The READ statement (line in ``module_mp_fast_sbm.F``) that consumes each
#: file.  For the two coefficient files it is the READ inside the loop.
READ_LINES: dict[str, int] = {
    "BLKD_SDC.dat": 5445,
    "capacity33.asc": 5500,
    "masses33.asc": 5548,
    "termvels33_corrected.asc": 5598,
    "constants33.asc": 5648,
    "kernLL_z33.asc": 5694,
    "ckli_33_300mb_500mb_750mb.asc": 5799,
    "cklg_33_300mb_500mb_750mb.asc": 5815,
    "cklh_33_300mb_500mb_750mb.asc": 5831,
    "ckls_33_300mb_500mb_750mb.asc": 5847,
    "ckii_33_300mb_500mb_750mb.asc": 5864,
    "ckis_33_300mb_500mb_750mb.asc": 5880,
    "cksg_33_300mb_500mb_750mb.asc": 5896,
    "ckss_33_300mb_500mb_750mb.asc": 5912,
    "bulkdens33.asc": 5988,
    "bulkradii33.asc": 6030,
    "coeff_p_new_33.dat": 8725,
    "coeff_q_new_33.dat": 8755,
}

#: Pressures of the three ``YWLL`` slices, in READ order (:5694).
YWLL_PRESSURES_HPA = (1000.0, 750.0, 500.0)
#: Pressures of the three slices of every ice-related kernel (:5799-5912).
ICE_KERNEL_PRESSURES_HPA = (300.0, 500.0, 750.0)

_PRECISIONS = ("float64", "float32")

# The arrays each FORMAT(6E13.5) READ fills, in the READ's variable order,
# with their Fortran shapes (:3935-3958 declare them, the ALLOCATE statements
# beside each READ size them).
_FIXED_LAYOUT: dict[str, tuple[tuple[str, tuple[int, ...]], ...]] = {
    "capacity33.asc": (
        ("rlec", (NKR,)), ("riec", (NKR, ICEMAX)), ("rsec", (NKR,)),
        ("rgec", (NKR,)), ("rhec", (NKR,))),
    "masses33.asc": (
        ("xl", (NKR,)), ("xi", (NKR, ICEMAX)), ("xs", (NKR,)),
        ("xg", (NKR,)), ("xh", (NKR,))),
    "termvels33_corrected.asc": (
        ("vr1", (NKR,)), ("vr2", (NKR, ICEMAX)), ("vr3", (NKR,)),
        ("vr4", (NKR,)), ("vr5", (NKR,))),
    "constants33.asc": (
        ("slic", (NKR, 6)), ("tlic", (NKR, 2)), ("coefin", (NKR,))),
    "kernLL_z33.asc": (
        ("ywll_1000mb", (NKR, NKR)), ("ywll_750mb", (NKR, NKR)),
        ("ywll_500mb", (NKR, NKR))),
    "ckli_33_300mb_500mb_750mb.asc": (
        ("ywli_300mb", (NKR, NKR, ICEMAX)), ("ywli_500mb", (NKR, NKR, ICEMAX)),
        ("ywli_750mb", (NKR, NKR, ICEMAX))),
    "cklg_33_300mb_500mb_750mb.asc": (
        ("ywlg_300mb", (NKR, NKR)), ("ywlg_500mb", (NKR, NKR)),
        ("ywlg_750mb", (NKR, NKR))),
    "cklh_33_300mb_500mb_750mb.asc": (
        ("ywlh_300mb", (NKR, NKR)), ("ywlh_500mb", (NKR, NKR)),
        ("ywlh_750mb", (NKR, NKR))),
    "ckls_33_300mb_500mb_750mb.asc": (
        ("ywls_300mb", (NKR, NKR)), ("ywls_500mb", (NKR, NKR)),
        ("ywls_750mb", (NKR, NKR))),
    "ckii_33_300mb_500mb_750mb.asc": (
        ("ywii_300mb", (NKR, NKR, ICEMAX, ICEMAX)),
        ("ywii_500mb", (NKR, NKR, ICEMAX, ICEMAX)),
        ("ywii_750mb", (NKR, NKR, ICEMAX, ICEMAX))),
    "ckis_33_300mb_500mb_750mb.asc": (
        ("ywis_300mb", (NKR, NKR, ICEMAX)), ("ywis_500mb", (NKR, NKR, ICEMAX)),
        ("ywis_750mb", (NKR, NKR, ICEMAX))),
    "cksg_33_300mb_500mb_750mb.asc": (
        ("ywsg_300mb", (NKR, NKR)), ("ywsg_500mb", (NKR, NKR)),
        ("ywsg_750mb", (NKR, NKR))),
    "ckss_33_300mb_500mb_750mb.asc": (
        ("ywss_300mb", (NKR, NKR)), ("ywss_500mb", (NKR, NKR)),
        ("ywss_750mb", (NKR, NKR))),
    "bulkdens33.asc": (
        ("ro1bl", (NKR,)), ("ro2bl", (NKR, ICEMAX)), ("ro3bl", (NKR,)),
        ("ro4bl", (NKR,)), ("ro5bl", (NKR,))),
}

#: Records of ``BLKD_SDC.dat`` the scheme reads (``DO kr=1,NKR``, :5444).
_BLKD_RECORDS_READ = NKR
#: Number of values RADXXO(NKR, NHYDRO) takes (ALLOCATE at :6006,
#: NHYDRO = 7 at :3894).
_NHYDRO = 7

# A Fortran real constant as the tables write it: optional sign, digits with
# a decimal point, and an optional exponent written with E or D.
_REAL_TOKEN = re.compile(
    r"^([+-]?)(\d*)\.(\d*)(?:[EeDd]([+-]?\d+))?$")
_INT_FIELD = re.compile(r"^ *[+-]?\d+$")

#: Largest finite float32, as an exact double.
_FLT_MAX = float(np.finfo(np.float32).max)


@dataclass(frozen=True, eq=False)
class FsbmTables:
    """The fast SBM lookup tables in one precision view.

    Array shapes follow Fortran index order with zero-based indices.  The
    stacked kernels put the pressure slice first: ``ywll[0]`` is
    ``YWLL_1000MB``, ``ywll[1]`` ``YWLL_750MB``, ``ywll[2]`` ``YWLL_500MB``
    (``YWLL_PRESSURES_HPA``); every ice-related stack is ordered 300, 500,
    750 hPa (``ICE_KERNEL_PRESSURES_HPA``).  All arrays are read-only views
    of one parse per precision that every instance shares; numpy refuses to
    make such a view writeable, so no caller can change the tables another
    caller holds.

    Units (CGS): masses in g, fall speeds in cm/s at 1000 hPa, capacities
    and radii in cm, bulk densities in g/cm3, drop-drop kernels in cm3/s.
    ``pkij`` is in g3 cm3/s per the comment at :8725.
    """

    precision: str
    root: str
    # (33,)
    xl: np.ndarray
    xs: np.ndarray
    xg: np.ndarray
    xh: np.ndarray
    vr1: np.ndarray
    vr3: np.ndarray
    vr4: np.ndarray
    vr5: np.ndarray
    rlec: np.ndarray
    rsec: np.ndarray
    rgec: np.ndarray
    rhec: np.ndarray
    ro1bl: np.ndarray
    ro3bl: np.ndarray
    ro4bl: np.ndarray
    ro5bl: np.ndarray
    # (33, 3)
    xi: np.ndarray
    vr2: np.ndarray
    riec: np.ndarray
    ro2bl: np.ndarray
    # (3, 33, 33), pressures 1000, 750, 500 hPa
    ywll: np.ndarray
    # (3, 33, 33, 3), pressures 300, 500, 750 hPa
    ywli: np.ndarray
    ywis: np.ndarray
    # (3, 33, 33), pressures 300, 500, 750 hPa
    ywlg: np.ndarray
    ywlh: np.ndarray
    ywls: np.ndarray
    ywsg: np.ndarray
    ywss: np.ndarray
    # (3, 33, 33, 3, 3), pressures 300, 500, 750 hPa
    ywii: np.ndarray
    # (18, 18, 18), zero where j > i
    pkij: np.ndarray
    # (18, 18)
    qkj: np.ndarray
    # (33,); the first four are float64 in both views (REAL*8 in WRF)
    bin_mass: np.ndarray
    tab_colum: np.ndarray
    tab_dendr: np.ndarray
    tab_snow: np.ndarray
    coefin: np.ndarray
    # read, never used by the scheme
    radxxo: np.ndarray  # (33, 7)
    slic: np.ndarray  # (33, 6)
    tlic: np.ndarray  # (33, 2)


# ---------------------------------------------------------------------------
# Byte validation
# ---------------------------------------------------------------------------

def _resolve_root(root: str | os.PathLike | None) -> Path:
    env = os.environ.get(TABLE_ROOT_ENV, "")
    if root is not None:
        chosen, source = Path(root), "the root argument"
    elif env:
        chosen, source = Path(env), TABLE_ROOT_ENV
    else:
        chosen, source = PACKAGED_ROOT, "the package"
    if not chosen.is_dir():
        raise FsbmRefusal("load_tables", f"table directory {chosen} (from "
                          f"{source}) does not exist")
    return chosen


def _validate(root: Path) -> dict[str, bytes]:
    """Read every pinned file and refuse unless all 18 match byte for byte."""
    blobs: dict[str, bytes] = {}
    problems: list[str] = []
    for name, (size, sha) in TABLE_PINS.items():
        path = root / name
        if not path.is_file():
            problems.append(f"{name}: missing from {root}")
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if len(data) == size and digest == sha:
            blobs[name] = data
            continue
        lf = data.replace(b"\r\n", b"\n")
        carriage_returns = data.count(b"\r")
        if (carriage_returns and len(lf) == size
                and hashlib.sha256(lf).hexdigest() == sha):
            problems.append(
                f"{name}: carries CRLF line endings ({carriage_returns} CR "
                "bytes); its LF content matches the pin, so it was copied "
                "through a tool that rewrote line endings; restore the LF "
                "bytes from NCAR's archive with tarfile")
        else:
            problems.append(
                f"{name}: {len(data)} bytes, sha256 {digest}; the pin is "
                f"{size} bytes, sha256 {sha}")
    if problems:
        raise FsbmRefusal(
            "load_tables",
            f"{len(problems)} table file(s) under {root} are not the pinned "
            "NCAR SBM_input_33 bytes: " + "; ".join(problems))
    return blobs


# ---------------------------------------------------------------------------
# Decimal to binary, exactly
# ---------------------------------------------------------------------------

def _decimal_parts(token: str, where: str) -> tuple[bool, int, int]:
    """(negative, integer significand M, decimal exponent E), value M 10^E."""
    match = _REAL_TOKEN.match(token)
    if match is None or not (match.group(2) or match.group(3)):
        raise FsbmRefusal(where, f"{token!r} is not a Fortran real constant "
                          "with a decimal point")
    sign, whole, frac, exp = match.groups()
    significand = int((whole + frac) or "0")
    exponent = (int(exp) if exp is not None else 0) - len(frac)
    return sign == "-", significand, exponent


def nearest_float32(token: str) -> np.float32:
    """The float32 nearest the decimal ``token``, ties to even, in one step.

    Exact integer arithmetic on the decimal's significand and exponent; no
    intermediate double.  This is the rounding gfortran's REAL*4 READ
    performs (``strtof``).  Refuses a value beyond the float32 range.
    """
    negative, sig, exp10 = _decimal_parts(token, "nearest_float32")
    if sig == 0:
        return np.float32(-0.0 if negative else 0.0)
    if exp10 >= 0:
        num, den = sig * 10 ** exp10, 1
    else:
        num, den = sig, 10 ** (-exp10)
    # floor(log2(num / den))
    e2 = num.bit_length() - den.bit_length()
    if e2 >= 0:
        if num < (den << e2):
            e2 -= 1
    elif (num << -e2) < den:
        e2 -= 1
    # 24-bit significand; below the normal range the quantum is 2**-149.
    q = max(e2 - 23, -149)
    if q >= 0:
        n, d = num, den << q
    else:
        n, d = num << -q, den
    mant, rem = divmod(n, d)
    if 2 * rem > d or (2 * rem == d and mant & 1):
        mant += 1
    value = math.ldexp(mant, q)  # exact: mant < 2**25
    if value > _FLT_MAX:
        raise FsbmRefusal("nearest_float32",
                          f"{token!r} is beyond the float32 range")
    return np.float32(-value if negative else value)


def nearest_float64(token: str) -> float:
    """The double nearest the decimal ``token`` (Python's correctly rounded
    conversion, after mapping a Fortran ``D`` exponent to ``E``)."""
    _decimal_parts(token, "nearest_float64")
    return float(token.replace("D", "E").replace("d", "e"))


def _convert(tokens: list[str], precision: str) -> np.ndarray:
    if precision == "float64":
        return np.array([nearest_float64(t) for t in tokens],
                        dtype=np.float64)
    return np.array([nearest_float32(t) for t in tokens], dtype=np.float32)


# ---------------------------------------------------------------------------
# The three READ shapes
# ---------------------------------------------------------------------------

def _records(name: str, data: bytes) -> list[str]:
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as error:
        raise FsbmRefusal(name, f"is not ASCII text: {error}") from None
    if "\r" in text:
        raise FsbmRefusal(name, "carries CR bytes")
    records = text.split("\n")
    if records and records[-1] == "":
        records.pop()
    return records


def _fixed_stream(name: str, records: list[str], count: int,
                  width: int = 13, per_record: int = 6) -> list[str]:
    """Consume ``count`` values of ``FORMAT(per_record E width.d)``.

    Format reversion starts a new record after every ``per_record`` values;
    the final record holds the remainder.  Records are read by column, not
    by whitespace, exactly as the edit descriptor does.
    """
    needed = -(-count // per_record)
    if len(records) != needed:
        raise FsbmRefusal(name, f"holds {len(records)} records; the READ "
                          f"consumes {needed} for {count} values")
    tokens: list[str] = []
    for index, record in enumerate(records):
        take = min(per_record, count - len(tokens))
        if len(record) != take * width:
            raise FsbmRefusal(
                name, f"record {index + 1} is {len(record)} columns; the "
                f"READ takes {take} fields of {width}")
        for field in range(take):
            chunk = record[field * width:(field + 1) * width]
            token = chunk.strip()
            if not token or " " in token:
                raise FsbmRefusal(
                    name, f"record {index + 1} field {field + 1} is "
                    f"{chunk!r}, not one real constant")
            tokens.append(token)
    return tokens


def _list_directed(name: str, record_tokens: list[str], count: int,
                   where: str) -> list[str]:
    """Values of one list-directed READ: blank-separated real constants.

    Repeat counts, commas, slashes and null values are list-directed
    features the pinned files never use; any of them is refused rather
    than interpreted.
    """
    if len(record_tokens) < count:
        raise FsbmRefusal(name, f"{where}: {len(record_tokens)} values where "
                          f"the READ takes {count}")
    for token in record_tokens[:count]:
        if _REAL_TOKEN.match(token) is None:
            raise FsbmRefusal(name, f"{where}: {token!r} is not a plain real "
                              "constant")
    return record_tokens[:count]


def _indexed(name: str, records: list[str], shape: tuple[int, ...],
             precision: str, lower_triangle: bool) -> np.ndarray:
    """``(nI6,1E16.8)`` records, each stating its position (:8725, :8755)."""
    nidx = len(shape)
    width = 6 * nidx + 16
    expected = (shape[0] * shape[1] * (shape[1] + 1) // 2 if lower_triangle
                else math.prod(shape))
    if len(records) != expected:
        raise FsbmRefusal(name, f"holds {len(records)} records; the loops "
                          f"read {expected}")
    positions: list[tuple[int, ...]] = []
    tokens: list[str] = []
    seen: set[tuple[int, ...]] = set()
    for number, record in enumerate(records, start=1):
        if len(record) != width:
            raise FsbmRefusal(name, f"record {number} is {len(record)} "
                              f"columns, not {width}")
        index = []
        for slot in range(nidx):
            field = record[6 * slot:6 * slot + 6]
            if _INT_FIELD.match(field) is None:
                raise FsbmRefusal(name, f"record {number} index field "
                                  f"{field!r} is not an I6 integer")
            index.append(int(field))
        if any(not 1 <= value <= extent
               for value, extent in zip(index, shape)):
            raise FsbmRefusal(name, f"record {number} names {tuple(index)}, "
                              f"outside the array {shape}")
        if lower_triangle and index[2] > index[1]:
            raise FsbmRefusal(name, f"record {number} names {tuple(index)} "
                              "with j > i")
        key = tuple(index)
        if key in seen:
            raise FsbmRefusal(name, f"record {number} fills {key} a second "
                              "time")
        seen.add(key)
        token = record[6 * nidx:].strip()
        if not token or " " in token:
            raise FsbmRefusal(name, f"record {number} value field is "
                              f"{record[6 * nidx:]!r}")
        positions.append(key)
        tokens.append(token)
    values = _convert(tokens, precision)
    out = np.zeros(shape, dtype=values.dtype)
    for key, value in zip(positions, values):
        out[tuple(v - 1 for v in key)] = value
    return out


# ---------------------------------------------------------------------------
# Streams, exposed so tests and the oracle can check every decimal
# ---------------------------------------------------------------------------

def _fixed_count(name: str) -> int:
    return sum(math.prod(shape) for _, shape in _FIXED_LAYOUT[name])


def decimal_streams(blobs: dict[str, bytes]) -> dict[str, list[str]]:
    """Every decimal token each READ consumes, in READ order, per file.

    For the two coefficient files the tokens are in record order (the
    positions are in the records themselves).  The caller supplies bytes
    that already passed ``_validate``.
    """
    streams: dict[str, list[str]] = {}
    for name in _FIXED_LAYOUT:
        streams[name] = _fixed_stream(name, _records(name, blobs[name]),
                                      _fixed_count(name))
    records = _records("BLKD_SDC.dat", blobs["BLKD_SDC.dat"])
    blkd: list[str] = []
    for kr in range(_BLKD_RECORDS_READ):
        blkd.extend(_list_directed("BLKD_SDC.dat", records[kr].split(), 4,
                                   f"record {kr + 1}"))
    streams["BLKD_SDC.dat"] = blkd
    radii = _records("bulkradii33.asc", blobs["bulkradii33.asc"])
    radii_tokens = " ".join(radii).split()
    if len(radii_tokens) != NKR * _NHYDRO:
        raise FsbmRefusal("bulkradii33.asc", f"holds {len(radii_tokens)} "
                          f"values; RADXXO takes {NKR * _NHYDRO}")
    streams["bulkradii33.asc"] = _list_directed(
        "bulkradii33.asc", radii_tokens, NKR * _NHYDRO, "READ at :6030")
    for name, nidx in (("coeff_p_new_33.dat", 3), ("coeff_q_new_33.dat", 2)):
        streams[name] = [record[6 * nidx:].strip()
                         for record in _records(name, blobs[name])]
    return streams


# ---------------------------------------------------------------------------
# Parsing into arrays
# ---------------------------------------------------------------------------

_CACHE: dict[str, dict[str, np.ndarray]] = {}


def _parse(blobs: dict[str, bytes], precision: str) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    streams = decimal_streams(blobs)
    for name, layout in _FIXED_LAYOUT.items():
        values = _convert(streams[name], precision)
        offset = 0
        for label, shape in layout:
            size = math.prod(shape)
            arrays[label] = np.reshape(values[offset:offset + size], shape,
                                       order="F").copy()
            offset += size

    # :5698-5706: entries with an index above 33 are zeroed; with NKR = 33
    # there are none, so this is WRF's own no-op.
    for label in ("ywll_1000mb", "ywll_750mb", "ywll_500mb"):
        arrays[label][33:, :] = 0.0
        arrays[label][:, 33:] = 0.0

    # BLKD_SDC.dat is REAL(KIND=R8SIZE) in WRF (:3936-3937): double in both
    # builds, so float64 in both views.
    blkd = np.array([nearest_float64(t) for t in streams["BLKD_SDC.dat"]],
                    dtype=np.float64).reshape(_BLKD_RECORDS_READ, 4)
    for column, label in enumerate(("bin_mass", "tab_colum", "tab_dendr",
                                    "tab_snow")):
        arrays[label] = blkd[:, column].copy()

    arrays["radxxo"] = np.reshape(
        _convert(streams["bulkradii33.asc"], precision), (NKR, _NHYDRO),
        order="F").copy()

    arrays["pkij"] = _indexed(
        "coeff_p_new_33.dat", _records("coeff_p_new_33.dat",
                                       blobs["coeff_p_new_33.dat"]),
        (JBREAK, JBREAK, JBREAK), precision, lower_triangle=True)
    arrays["qkj"] = _indexed(
        "coeff_q_new_33.dat", _records("coeff_q_new_33.dat",
                                       blobs["coeff_q_new_33.dat"]),
        (JBREAK, JBREAK), precision, lower_triangle=False)

    def stack(prefix: str, pressures: tuple[float, ...]) -> np.ndarray:
        return np.stack([arrays.pop(f"{prefix}_{int(p)}mb")
                         for p in pressures])

    arrays["ywll"] = stack("ywll", YWLL_PRESSURES_HPA)
    for prefix in ("ywli", "ywlg", "ywlh", "ywls", "ywii", "ywis", "ywsg",
                   "ywss"):
        arrays[prefix] = stack(prefix, ICE_KERNEL_PRESSURES_HPA)

    for array in arrays.values():
        array.setflags(write=False)
    return arrays


def load_tables(precision: str = "float64",
                root: str | os.PathLike | None = None) -> FsbmTables:
    """Validate the 18 files byte for byte, then parse them as WRF does.

    ``root`` defaults to ``GPUWM_FSBM_TABLE_ROOT`` when that is set and to
    the packaged directory otherwise; either way every file must match its
    pin.  Refuses with ``FsbmRefusal`` naming each file that differs.
    """
    if precision not in _PRECISIONS:
        raise FsbmRefusal("load_tables", f"precision {precision!r} is not one "
                          f"of {_PRECISIONS}")
    chosen = _resolve_root(root)
    blobs = _validate(chosen)
    # The bytes equal the pins, so one parse per precision serves every root.
    if precision not in _CACHE:
        _CACHE[precision] = _parse(blobs, precision)
    # Hand out views, never the cached arrays themselves: a cached array owns
    # its data, so a caller could setflags(write=True) on it and change the
    # tables every other instance shares.  A view over a read-only owner
    # cannot be made writeable.
    views = {name: array.view() for name, array in _CACHE[precision].items()}
    return FsbmTables(precision=precision, root=str(chosen), **views)


def double_rounding_census(root: str | os.PathLike | None = None
                           ) -> dict[str, int]:
    """Per file, how many decimals double-then-cast would round wrongly.

    Counts the tokens whose float32 by way of the nearest double differs
    from the float32 nearest the decimal.  Only the arrays WRF holds as
    REAL*4 are counted (``BLKD_SDC.dat`` is REAL*8 in WRF).
    """
    streams = decimal_streams(_validate(_resolve_root(root)))
    census: dict[str, int] = {}
    for name, tokens in streams.items():
        if name == "BLKD_SDC.dat":
            continue
        census[name] = sum(
            1 for t in tokens
            if np.float32(nearest_float64(t)) != nearest_float32(t))
    return census
