"""The packaged fast SBM tables and their loader.

THE BREAKAGE THESE GATES PREVENT
--------------------------------
* A table that drifts from NCAR's bytes, most likely by picking up CR line
  endings through a Windows copy tool (the P3 table did exactly that), and
  is then read as if nothing happened.
* A loader that fills the arrays in the wrong order: a READ list naming
  several arrays is one column-major stream, and a row-major fill or a
  per-array record restart gives wrong kernels with the right shapes.
* A float32 view that is rounded by way of double, which differs from
  WRF's REAL*4 READ on any decimal whose nearest double sits on a float32
  midpoint.
* A loader that agrees with this file's reading of the Fortran but not with
  the Fortran itself.  The last section compares every array with the
  arrays WRF's own READ filled in the O64 and O32 oracle builds.

Every other expectation here is derived independently of the loader: the
pins are appendix A of the fast-sbm design, the files are re-parsed by
column or by whitespace in this file, and the float32 rounding is
re-derived with ``fractions.Fraction`` by a different method from the
loader's.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import math
import subprocess
import sys
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from gpuwm.core.fast_sbm import tables as T
from gpuwm.core.fast_sbm.constants import FsbmRefusal

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "gpuwm" / "data" / "fast_sbm"
TABLE_DIR = DATA / "SBM_input_33"

#: Fast-sbm design, appendix A: bytes and SHA-256 of each member FSBM opens.
APPENDIX_A = {
    "BLKD_SDC.dat": (2280, "6992497ee5b95f0fd215b762ad4af8611079d1e13dbe3db78687fe04a128abf2"),
    "capacity33.asc": (3042, "33759dbb0e238b37403ce2ad62759cca57c03538d72d34c2cc04bb7b2547c951"),
    "masses33.asc": (3042, "f53d3acdaa0a3e2a265272fa22a2a0bd54292f51815149370bbf315e3fa1bee4"),
    "termvels33_corrected.asc": (3042, "bf4209175492a3019563671676787afc3befd948fddc6435db58f7a8d7f702e3"),
    "constants33.asc": (3911, "d3d7cf345ebf850fa12376a7f173ea12b169f6413910939682494bcc5874a4fa"),
    "kernLL_z33.asc": (43016, "ce4e140595ff6db2434ecd5376a16c50ea410d03ddc1700bfa95a3490c15106f"),
    "ckli_33_300mb_500mb_750mb.asc": (129047, "59ce82054f29df7d9f982249b915201f3f0385ae559c5ef2b7cdb1907a46f81a"),
    "cklg_33_300mb_500mb_750mb.asc": (43016, "f3e8149a7e1bac947706f945a1d2ed0fb81ef2737404a0ef8c8c0c682fb4290f"),
    "cklh_33_300mb_500mb_750mb.asc": (43016, "a2ffe0b96e4acfcd06dc1a0b10cb28f60601ebe255f8e00a70ce1797705fb50e"),
    "ckls_33_300mb_500mb_750mb.asc": (43016, "ae282329a306d384bd6de958f1e0bf745c496d286237e8d22c85926081e0456a"),
    "ckii_33_300mb_500mb_750mb.asc": (387140, "546777c064e7687bbaaaf30ab417dfc146e2decaabc41b270439e33328a75c7f"),
    "ckis_33_300mb_500mb_750mb.asc": (129047, "657428fd7bbc0dcf5e7810bb0e25af42645f6122a4b24313404b805cfcdec44d"),
    "cksg_33_300mb_500mb_750mb.asc": (43016, "85b5fd667242ad216dd1b06f0f1c5ddfe15d797ae6c14dcc45e1097743bb37b9"),
    "ckss_33_300mb_500mb_750mb.asc": (43016, "1946554a0e1341cd12c9ae6be2cd2c786bfc3e4622b8fe87f2e85fcc64373847"),
    "bulkdens33.asc": (3042, "a5607b3d0b9eb1fcdba30e50b714e348a81aabb64d963b0a00ff4f7b2a24caa6"),
    "bulkradii33.asc": (3512, "d432d8eed8760f5000c6c86ac1b602687feb0783f558adbf10811fe7040a9de0"),
    "coeff_p_new_33.dat": (107730, "0fc78f82cfed4033eab9bcaf3f5d6bf07a4f1bbbbf3aff708ede6a0b2e921e34"),
    "coeff_q_new_33.dat": (9396, "cbda1e0fb2c604d6c8d61ad0abc22be7b1552be93bf5daed10b5186c4ba07d9e"),
}
ARCHIVE_SHA256 = "25ed63ab3550aed110ae24df229faece20eafedfdf21cde01d8973826a7294bf"

#: Each FORMAT(6E13.5) file's READ list as the design's section 3.5 and the
#: FsbmTables contract name it: (loader field, Fortran shape, slice index).
#: A slice index of None is a plain field; an int picks the pressure slice of
#: a stacked kernel field.
LAYOUT = {
    "capacity33.asc": [("rlec", (33,), None), ("riec", (33, 3), None),
                       ("rsec", (33,), None), ("rgec", (33,), None),
                       ("rhec", (33,), None)],
    "masses33.asc": [("xl", (33,), None), ("xi", (33, 3), None),
                     ("xs", (33,), None), ("xg", (33,), None),
                     ("xh", (33,), None)],
    "termvels33_corrected.asc": [("vr1", (33,), None), ("vr2", (33, 3), None),
                                 ("vr3", (33,), None), ("vr4", (33,), None),
                                 ("vr5", (33,), None)],
    "constants33.asc": [("slic", (33, 6), None), ("tlic", (33, 2), None),
                        ("coefin", (33,), None)],
    "kernLL_z33.asc": [("ywll", (33, 33), p) for p in range(3)],
    "ckli_33_300mb_500mb_750mb.asc": [("ywli", (33, 33, 3), p) for p in range(3)],
    "cklg_33_300mb_500mb_750mb.asc": [("ywlg", (33, 33), p) for p in range(3)],
    "cklh_33_300mb_500mb_750mb.asc": [("ywlh", (33, 33), p) for p in range(3)],
    "ckls_33_300mb_500mb_750mb.asc": [("ywls", (33, 33), p) for p in range(3)],
    "ckii_33_300mb_500mb_750mb.asc": [("ywii", (33, 33, 3, 3), p) for p in range(3)],
    "ckis_33_300mb_500mb_750mb.asc": [("ywis", (33, 33, 3), p) for p in range(3)],
    "cksg_33_300mb_500mb_750mb.asc": [("ywsg", (33, 33), p) for p in range(3)],
    "ckss_33_300mb_500mb_750mb.asc": [("ywss", (33, 33), p) for p in range(3)],
    "bulkdens33.asc": [("ro1bl", (33,), None), ("ro2bl", (33, 3), None),
                       ("ro3bl", (33,), None), ("ro4bl", (33,), None),
                       ("ro5bl", (33,), None)],
}

#: Tokens where rounding by way of double would give a different float32
#: from rounding directly, across all REAL*4 tables, measured 2026-09-22.
DOUBLE_ROUNDING_MISMATCHES = 0


def _fields_by_column(name: str) -> list[str]:
    """This test's own FORMAT(6E13.5) reader: 13-column fields, in order."""
    out = []
    for line in (TABLE_DIR / name).read_bytes().decode("ascii").split("\n"):
        for start in range(0, len(line), 13):
            field = line[start:start + 13].strip()
            if field:
                out.append(field)
    return out


def _flat(array: np.ndarray) -> np.ndarray:
    return np.asarray(array).reshape(-1, order="F")


def _array_stream(tables, name: str) -> np.ndarray:
    """The loader's arrays for one READ list, flattened back to the stream."""
    parts = []
    for field, _shape, slice_index in LAYOUT[name]:
        array = getattr(tables, field)
        parts.append(_flat(array if slice_index is None
                           else array[slice_index]))
    return np.concatenate(parts)


@pytest.fixture(scope="module")
def t64():
    return T.load_tables("float64")


@pytest.fixture(scope="module")
def t32():
    return T.load_tables("float32")


def _mirror(tmp_path: Path) -> Path:
    mirror = tmp_path / "SBM_input_33"
    mirror.mkdir()
    for name in APPENDIX_A:
        (mirror / name).write_bytes((TABLE_DIR / name).read_bytes())
    return mirror


# ---------------------------------------------------------------------------
# The bytes
# ---------------------------------------------------------------------------

def test_every_packaged_file_matches_appendix_a_and_is_lf_only():
    present = sorted(p.name for p in TABLE_DIR.iterdir())
    assert present == sorted([*APPENDIX_A, "MANIFEST.sha256"])
    for name, (size, sha) in APPENDIX_A.items():
        data = (TABLE_DIR / name).read_bytes()
        assert len(data) == size, name
        assert hashlib.sha256(data).hexdigest() == sha, name
        assert b"\r" not in data, f"{name} carries CR bytes"
        assert data.endswith(b"\n"), name


def test_the_loader_pins_are_appendix_a():
    assert T.TABLE_PINS == APPENDIX_A
    assert T.ARCHIVE_SHA256 == ARCHIVE_SHA256
    assert T.ARCHIVE_BYTES == 1_339_105


def test_the_manifest_lists_every_pin_in_sha256sum_form():
    data = (TABLE_DIR / "MANIFEST.sha256").read_bytes()
    assert b"\r" not in data
    expected = "".join(f"{APPENDIX_A[n][1]} *{n}\n" for n in sorted(APPENDIX_A))
    assert data.decode("ascii") == expected


def test_the_licence_is_a_byte_copy_of_the_p3_one():
    copy = (DATA / "LICENSE-WRF.txt").read_bytes()
    assert copy == (ROOT / "gpuwm" / "data" / "p3" / "LICENSE-WRF.txt").read_bytes()
    assert b"public domain" in copy


def test_provenance_names_every_member_pin_and_read_line():
    text = (DATA / "PROVENANCE.md").read_text(encoding="utf-8")
    assert ARCHIVE_SHA256 in text
    for name, (size, sha) in APPENDIX_A.items():
        assert f"`{name}` | {size} | `{sha}` | {T.READ_LINES[name]} |" in text, name
    assert "35 regular files" in text
    assert chr(0x2014) not in text, "no em-dashes in shipped prose"


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_a_byte_identical_mirror_loads_and_equals_the_package(tmp_path, t64):
    mirror = _mirror(tmp_path)
    loaded = T.load_tables("float64", root=mirror)
    assert loaded.root == str(mirror)
    assert np.array_equal(loaded.ywii, t64.ywii)
    assert np.array_equal(loaded.pkij, t64.pkij)


def test_one_flipped_byte_is_refused_and_named(tmp_path):
    mirror = _mirror(tmp_path)
    target = mirror / "kernLL_z33.asc"
    data = bytearray(target.read_bytes())
    data[1000] = ord("7") if data[1000] != ord("7") else ord("8")
    target.write_bytes(bytes(data))
    with pytest.raises(FsbmRefusal) as caught:
        T.load_tables("float64", root=mirror)
    message = str(caught.value)
    assert "kernLL_z33.asc" in message
    assert "masses33.asc" not in message  # only the damaged file is named


def test_a_crlf_copy_is_refused_and_named_as_crlf(tmp_path):
    mirror = _mirror(tmp_path)
    target = mirror / "masses33.asc"
    target.write_bytes(target.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(FsbmRefusal) as caught:
        T.load_tables("float32", root=mirror)
    message = str(caught.value)
    assert "masses33.asc" in message and "CRLF" in message


def test_a_missing_file_is_refused(tmp_path):
    mirror = _mirror(tmp_path)
    (mirror / "coeff_q_new_33.dat").unlink()
    with pytest.raises(FsbmRefusal, match="coeff_q_new_33.dat"):
        T.load_tables(root=mirror)


def test_the_environment_root_is_accepted_only_for_identical_bytes(
        tmp_path, monkeypatch):
    mirror = _mirror(tmp_path)
    monkeypatch.setenv(T.TABLE_ROOT_ENV, str(mirror))
    assert T.load_tables().root == str(mirror)
    target = mirror / "bulkdens33.asc"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(FsbmRefusal, match="bulkdens33.asc"):
        T.load_tables()


def test_an_unknown_precision_is_refused():
    with pytest.raises(FsbmRefusal, match="precision"):
        T.load_tables("float16")


# ---------------------------------------------------------------------------
# Shapes and fill order
# ---------------------------------------------------------------------------

EXPECTED_SHAPES = {
    **{n: (33,) for n in ("xl", "xs", "xg", "xh", "vr1", "vr3", "vr4", "vr5",
                          "rlec", "rsec", "rgec", "rhec", "ro1bl", "ro3bl",
                          "ro4bl", "ro5bl", "bin_mass", "tab_colum",
                          "tab_dendr", "tab_snow", "coefin")},
    **{n: (33, 3) for n in ("xi", "vr2", "riec", "ro2bl")},
    "ywll": (3, 33, 33),
    "ywli": (3, 33, 33, 3), "ywis": (3, 33, 33, 3),
    **{n: (3, 33, 33) for n in ("ywlg", "ywlh", "ywls", "ywsg", "ywss")},
    "ywii": (3, 33, 33, 3, 3),
    "pkij": (18, 18, 18), "qkj": (18, 18),
    "radxxo": (33, 7), "slic": (33, 6), "tlic": (33, 2),
}
REAL8 = ("bin_mass", "tab_colum", "tab_dendr", "tab_snow")


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_shapes_dtypes_and_read_only(precision):
    tables = T.load_tables(precision)
    for name, shape in EXPECTED_SHAPES.items():
        array = getattr(tables, name)
        assert array.shape == shape, name
        expected = np.float64 if (precision == "float64" or name in REAL8) \
            else np.float32
        assert array.dtype == expected, name
        assert not array.flags.writeable, name


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_no_caller_can_make_the_shared_tables_writeable(precision):
    """Every FsbmTables of one precision shares one parse.  If an instance
    held an array that owned its data, ``setflags(write=True)`` would succeed
    and a write through it would change the tables every other caller holds.
    The loader hands out views over read-only owners, which numpy refuses to
    make writeable."""
    first = T.load_tables(precision)
    for name in EXPECTED_SHAPES:
        array = getattr(first, name)
        assert not array.flags.owndata, name
        with pytest.raises(ValueError):
            array.setflags(write=True)
        with pytest.raises(ValueError):
            array[(0,) * array.ndim] = 1.0
    second = T.load_tables(precision)
    for name in EXPECTED_SHAPES:
        assert getattr(second, name) is not getattr(first, name), name
        assert np.shares_memory(getattr(second, name), getattr(first, name))


def test_fortran_fill_order_of_masses(t64):
    stream = [float(v) for v in _fields_by_column("masses33.asc")]
    assert len(stream) == 231
    # XI(:,1) is values 34-66 of the file, XI(:,2) 67-99, XI(:,3) 100-132.
    assert np.array_equal(t64.xl, stream[0:33])
    assert np.array_equal(t64.xi[:, 0], stream[33:66])
    assert np.array_equal(t64.xi[:, 1], stream[66:99])
    assert np.array_equal(t64.xi[:, 2], stream[99:132])
    assert np.array_equal(t64.xs, stream[132:165])
    assert np.array_equal(t64.xg, stream[165:198])
    assert np.array_equal(t64.xh, stream[198:231])


def test_every_fixed_format_stream_lands_column_major(t64):
    for name in LAYOUT:
        stream = np.array([float(v) for v in _fields_by_column(name)])
        assert np.array_equal(_array_stream(t64, name), stream), name
    # One element spelled out: YWII_500MB(2,5,3,1) is stream position
    # 9801 + (2-1) + 33*(5-1) + 1089*(3-1) + 3267*(1-1) of ckii.
    ckii = [float(v) for v in _fields_by_column("ckii_33_300mb_500mb_750mb.asc")]
    assert t64.ywii[1, 1, 4, 2, 0] == ckii[9801 + 1 + 33 * 4 + 1089 * 2]
    kern = [float(v) for v in _fields_by_column("kernLL_z33.asc")]
    assert t64.ywll[2, 6, 3] == kern[2 * 1089 + 6 + 33 * 3]


def test_blkd_sdc_reads_the_first_33_records(t64, t32):
    lines = (TABLE_DIR / "BLKD_SDC.dat").read_text(encoding="ascii").split("\n")
    rows = [[float(v.replace("D", "E")) for v in line.split()]
            for line in lines[:33]]
    table = np.array(rows)
    for column, name in enumerate(REAL8):
        assert np.array_equal(getattr(t64, name), table[:, column]), name
        assert np.array_equal(getattr(t32, name), table[:, column]), name
    assert len([line for line in lines if line.strip()]) == 43


def test_radxxo_is_one_list_directed_read(t64):
    tokens = (TABLE_DIR / "bulkradii33.asc").read_text(encoding="ascii").split()
    assert len(tokens) == 231
    assert np.array_equal(_flat(t64.radxxo), [float(v) for v in tokens])


def _indexed_records(name: str, nidx: int):
    out = []
    for line in (TABLE_DIR / name).read_text(encoding="ascii").splitlines():
        parts = line.split()
        out.append((tuple(int(v) for v in parts[:nidx]), parts[nidx]))
    return out


def test_every_pkij_entry_lands_at_its_stated_position(t64, t32):
    records = _indexed_records("coeff_p_new_33.dat", 3)
    assert len(records) == 3078
    positions = {index for index, _ in records}
    assert positions == {(k, i, j) for k in range(1, 19)
                         for i in range(1, 19) for j in range(1, i + 1)}
    for (k, i, j), value in records:
        assert t64.pkij[k - 1, i - 1, j - 1] == float(value)
        assert t32.pkij[k - 1, i - 1, j - 1] == T.nearest_float32(value)
    upper = np.triu(np.ones((18, 18), dtype=bool), 1)
    for k in range(18):
        assert not np.any(t64.pkij[k][upper]), "j > i must stay zero"
    assert np.count_nonzero(t64.pkij) > 0


def test_every_qkj_entry_lands_at_its_stated_position(t64):
    records = _indexed_records("coeff_q_new_33.dat", 2)
    assert {index for index, _ in records} == {
        (k, j) for k in range(1, 19) for j in range(1, 19)}
    for (k, j), value in records:
        assert t64.qkj[k - 1, j - 1] == float(value)


def test_the_indexed_reader_refuses_a_duplicate_or_upper_position():
    good = (TABLE_DIR / "coeff_p_new_33.dat").read_text(encoding="ascii")
    records = good.splitlines()
    duplicate = [records[0]] + records[:-1]
    with pytest.raises(FsbmRefusal, match="second"):
        T._indexed("coeff_p_new_33.dat", duplicate, (18, 18, 18), "float64",
                   lower_triangle=True)
    upper = records.copy()
    upper[1] = "     1     1     2" + upper[1][18:]
    with pytest.raises(FsbmRefusal, match="j > i"):
        T._indexed("coeff_p_new_33.dat", upper, (18, 18, 18), "float64",
                   lower_triangle=True)


def test_the_fixed_reader_refuses_a_field_without_a_decimal_point():
    """E13.5 would read "33510" with an implied scale; the loader refuses
    to guess rather than silently reading it one way or the other."""
    fields = ["0.33510E-10", "0.67021E-10", "33510", "0.26808E-09",
              "0.53617E-09", "0.10723E-08"]
    records = ["".join(f.rjust(13) for f in fields)]
    tokens = T._fixed_stream("masses33.asc", records, 6)
    assert tokens[2] == "33510"
    with pytest.raises(FsbmRefusal, match="decimal point"):
        T.nearest_float64(tokens[2])
    with pytest.raises(FsbmRefusal, match="decimal point"):
        T.nearest_float32(tokens[2])
    short = [records[0][:-1]]
    with pytest.raises(FsbmRefusal, match="columns"):
        T._fixed_stream("masses33.asc", short, 6)


# ---------------------------------------------------------------------------
# Known values
# ---------------------------------------------------------------------------

def test_known_values(t64):
    assert t64.xl[0] == 3.3510e-11
    assert t64.rlec[32] == 0.32510
    assert t64.rlec[14] == 0.50797e-2  # bin 15, the 50.8 um cloud/rain split
    assert t64.vr1[32] == 900.0
    assert np.all(t64.ro1bl == 1.0)
    for name in ("xl", "xs", "xg", "xh"):
        grid = getattr(t64, name)
        ratio = grid[1:] / grid[:-1]
        assert np.all(np.abs(ratio / 2.0 - 1.0) < 5e-5), name
    for column in range(3):
        assert np.array_equal(t64.xi[:, column], t64.xl)


def test_every_drop_mass_is_a_2_um_drop_doubled_to_five_digits():
    """XL(k) is the five-significant-digit rounding of (4/3) pi (2 um)^3
    2^(k-1) g, so the grid is an exact doubling carried to five digits."""
    getcontext().prec = 50
    pi = Decimal("3.14159265358979323846264338327950288")
    first = Decimal(4) / 3 * pi * Decimal("2e-4") ** 3
    tokens = _fields_by_column("masses33.asc")[:33]
    for k, token in enumerate(tokens):
        assert Decimal(format(first * 2 ** k, ".4E")) == Decimal(token), k


# ---------------------------------------------------------------------------
# Decimal to binary
# ---------------------------------------------------------------------------

def _independent_float32(token: str) -> tuple[np.float32, bool]:
    """Nearest float32 by comparing exact distances to the neighbours of the
    double-then-cast value; also says whether double-then-cast was wrong."""
    exact = Fraction(token.replace("D", "E"))
    cast = np.float32(float(exact))
    best = None
    for candidate in (np.nextafter(cast, np.float32(-np.inf)), cast,
                      np.nextafter(cast, np.float32(np.inf))):
        distance = abs(Fraction(float(candidate)) - exact)
        key = (distance, int(np.frombuffer(candidate.tobytes(),
                                           dtype=np.uint32)[0]) & 1)
        if best is None or key < best[0]:
            best = (key, candidate)
    return best[1], bool(best[1] != cast)


def test_the_float64_view_is_every_decimal_correctly_rounded(t64):
    for name in LAYOUT:
        stream = [float(Fraction(v)) for v in _fields_by_column(name)]
        assert np.array_equal(_array_stream(t64, name), stream), name


def test_the_float32_view_is_every_decimal_rounded_once(t32, capsys):
    mismatches = {}
    for name in LAYOUT:
        tokens = _fields_by_column(name)
        expected = []
        wrong = 0
        for token in tokens:
            value, cast_was_wrong = _independent_float32(token)
            expected.append(value)
            wrong += cast_was_wrong
        got = _array_stream(t32, name)
        assert got.dtype == np.float32
        assert np.array_equal(got.view(np.uint32),
                              np.array(expected, dtype=np.float32).view(np.uint32)), name
        mismatches[name] = wrong
    for name, nidx in (("coeff_p_new_33.dat", 3), ("coeff_q_new_33.dat", 2)):
        wrong = 0
        for _index, token in _indexed_records(name, nidx):
            value, cast_was_wrong = _independent_float32(token)
            assert T.nearest_float32(token) == value
            wrong += cast_was_wrong
        mismatches[name] = wrong
    wrong = 0
    radii = (TABLE_DIR / "bulkradii33.asc").read_text(encoding="ascii").split()
    for token, got in zip(radii, _flat(t32.radxxo)):
        value, cast_was_wrong = _independent_float32(token)
        assert got == value
        wrong += cast_was_wrong
    mismatches["bulkradii33.asc"] = wrong
    total = sum(mismatches.values())
    offenders = {k: v for k, v in mismatches.items() if v} or "none"
    with capsys.disabled():
        print(f"\nfast SBM float32 tables: {total} of the REAL*4 decimals "
              f"would round differently by way of double; per file "
              f"{offenders}")
    assert T.double_rounding_census() == mismatches
    assert total == DOUBLE_ROUNDING_MISMATCHES


def test_nearest_float32_rounds_once_where_double_then_cast_does_not():
    """Negative control for the census: a decimal just above a float32
    midpoint whose nearest double IS the midpoint."""
    getcontext().prec = 200
    midpoint = Decimal(1) + Decimal(2) ** -24
    token = format(midpoint + Decimal(2) ** -80, "f")
    assert np.float32(float(token)) == np.float32(1.0)  # double-then-cast
    assert T.nearest_float32(token) == np.float32(1.0 + 2.0 ** -23)
    value, cast_was_wrong = _independent_float32(token)
    assert cast_was_wrong and value == T.nearest_float32(token)
    # An exact midpoint ties to even, both ways.
    assert T.nearest_float32(format(midpoint, "f")) == np.float32(1.0)
    upper = Decimal(1) + 3 * Decimal(2) ** -24
    assert T.nearest_float32(format(upper, "f")) == np.float32(1.0 + 2.0 ** -22)
    # Sign, D exponents, the subnormal quantum and the range edge.
    assert T.nearest_float32("-0.89361E-01") == -np.float32(0.089361)
    assert T.nearest_float32("0.33510D-10") == np.float32(3.3510e-11)
    assert T.nearest_float32("1.0E-45") == np.float32(2.0 ** -149)
    with pytest.raises(FsbmRefusal, match="range"):
        T.nearest_float32("1.0E39")


# ---------------------------------------------------------------------------
# Package hygiene
# ---------------------------------------------------------------------------

def test_no_module_of_the_package_imports_cupy():
    package = ROOT / "gpuwm" / "core" / "fast_sbm"
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            assert not any(n.split(".")[0] == "cupy" for n in names), path.name


def test_importing_the_package_reads_no_table():
    code = ("import sys, gpuwm.core.fast_sbm; "
            "print('gpuwm.core.fast_sbm.tables' in sys.modules)")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"


def test_the_ywll_zeroing_is_a_no_op_at_33_bins(t64):
    """:5698-5706 zeroes indices above 33; the last row and column of every
    slice keep the file's values, so nothing was zeroed."""
    kern = np.array([float(v) for v in _fields_by_column("kernLL_z33.asc")])
    assert math.prod(t64.ywll.shape) == kern.size
    for p in range(3):
        stream = kern[p * 1089:(p + 1) * 1089].reshape(33, 33, order="F")
        assert np.array_equal(t64.ywll[p][32, :], stream[32, :])
        assert np.array_equal(t64.ywll[p][:, 32], stream[:, 32])
        assert np.any(stream[32, :] != 0.0) and np.any(stream[:, 32] != 0.0)


# ---------------------------------------------------------------------------
# Table values: the loader against WRF's own READ (fast-sbm design 5.5)
# ---------------------------------------------------------------------------

#: Fields of the oracle's ``read`` record that are not READ tables: the step
#: FAST_HUCMINIT was called with, and ``BIN_LOG``, which WRF computes as
#: ``log10(bin_mass)`` beside the READ (:5446) for the polarimetric operator.
NOT_READ_TABLES = ("dt", "bin_log")

#: The FsbmTables contract for the stacked kernels, written out here rather
#: than read from the loader, so a loader that relabels a slice is caught:
#: ``ywll`` is 1000, 750, 500 hPa and every ice-related stack 300, 500,
#: 750 hPa.
YWLL_SLICE_ORDER_HPA = (1000, 750, 500)
ICE_SLICE_ORDER_HPA = (300, 500, 750)


def _oracle_read_record(build: str) -> dict[str, np.ndarray]:
    # The fsbm-fortran-oracle item merged in wave 1, so its reader is always
    # here; the skip that waited on it is retired and a missing reader fails.
    from tools.fast_sbm_wrf480_oracle import fixtures

    dump = fixtures.load_fixture(f"tables_{build}")
    return {key.split(".", 1)[1]: value[0] for key, value in dump.items()
            if key.startswith("read.")}


def _loader_array(tables, field: str) -> tuple[str, np.ndarray]:
    """The loader field and array holding the oracle's ``field``.

    The oracle names each pressure slice as WRF declares it
    (``ywll_750mb``); the loader stacks the three slices of a kernel.
    """
    prefix, _, suffix = field.rpartition("_")
    if prefix and suffix.endswith("mb"):
        pressure = int(suffix[:-2])
        order = (YWLL_SLICE_ORDER_HPA if prefix == "ywll"
                 else ICE_SLICE_ORDER_HPA)
        return prefix, getattr(tables, prefix)[order.index(pressure)]
    return field, getattr(tables, field)


@pytest.mark.parametrize("build, precision",
                         [("o64", "float64"), ("o32", "float32")])
def test_every_table_is_the_array_wrfs_own_read_filled(build, precision):
    """The Table values rung: each loader view is byte-identical to the
    arrays WRF's READ statements filled in the matching oracle build (O64
    for float64, O32 for float32).

    Breakage prevented: a view WRF never ran with.  A wrong fill order, a
    decimal rounded unlike gfortran's READ, a REAL*8 table held as float32
    (or a REAL*4 one as float64), a pressure slice out of order or a
    coefficient at the wrong position all pass every check above that
    shares the loader's reading of the Fortran; every process gate that
    follows would then compare the host against tables the Fortran never
    held.
    """
    record = _oracle_read_record(build)
    tables = T.load_tables(precision)
    covered = set()
    for field, expected in record.items():
        if field in NOT_READ_TABLES:
            continue
        name, mine = _loader_array(tables, field)
        covered.add(name)
        assert mine.dtype == expected.dtype, field
        assert mine.shape == expected.shape, field
        assert mine.tobytes() == expected.tobytes(), field
    loader_fields = {f.name for f in dataclasses.fields(T.FsbmTables)}
    assert covered == loader_fields - {"precision", "root"}
