"""The shared Fortran namelist reader (gpuwm.fortran_namelist), form by form.

A142: ``start_year = 2026, start_month = 08, start_day = 25,`` on one line
read as start_year = [2026, 'start_month = 08', 'start_day = 25'], so a
legal namelist packed several keys per line was refused as missing keys.
Every form of Fortran namelist input WRF's own READ(NML=) accepts is
covered here, then whole real namelists are read one key per line and
packed several per line and must parse identically.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from gpuwm.fortran_namelist import (
    parse_namelist,
    parse_namelist_text,
    scan_namelist_text,
)

REPO = Path(__file__).resolve().parents[1]


def _one(body: str, group: str = "g", **kwargs) -> dict:
    return parse_namelist_text(f"&{group}\n{body}\n/\n", **kwargs)[group]


# ---------------------------------------------------------------------------
# Forms that are new with this reader
# ---------------------------------------------------------------------------

def test_a142_several_keys_on_one_line():
    parsed = _one(" start_year = 2026, start_month = 08, start_day = 25,")
    assert parsed == {"start_year": [2026], "start_month": [8],
                      "start_day": [25]}


def test_blanks_separate_values_and_a_name_can_follow_a_blank():
    parsed = _one(" e_we = 100 200 300 e_sn = 80\t90 dx = 3000.")
    assert parsed == {"e_we": [100, 200, 300], "e_sn": [80, 90],
                      "dx": [3000.0]}


def test_mixed_commas_blanks_and_names_mid_line_across_lines():
    parsed = _one(" a = 1, 2 b = 3,\n 4, c = 5 ,6,\n d=7")
    assert parsed == {"a": [1, 2], "b": [3, 4], "c": [5, 6], "d": [7]}


def test_a_whole_group_on_one_line_and_group_after_terminator():
    parsed = parse_namelist_text(
        "&time_control run_hours = 6, history_interval = 60 / &domains "
        "max_dom = 1 /\n")
    assert parsed == {"time_control": {"run_hours": [6],
                                       "history_interval": [60]},
                      "domains": {"max_dom": [1]}}


def test_array_element_assignment():
    parsed = _one(" e_we = 100, 200, 300,\n e_we(2) = 250,\n e_sn(1) = 80,"
                  " e_sn(2) = 90,")
    assert parsed["e_we"] == [100, 250, 300]
    assert parsed["e_sn"] == [80, 90]


def test_array_element_with_several_values_fills_the_following_elements():
    # gfortran's expanded read (a GNU extension, on in WRF's builds) reads
    # x(2) = a, b as x(2) = a, x(3) = b.
    assert _one(" x = 1, 2, 3, 4,\n x(2) = 20, 30,")["x"] == [1, 20, 30, 4]


def test_array_section_assignments():
    parsed = _one(" a = 1, 2, 3, 4, 5, 6,\n a(2:4) = 20, 30, 40,\n"
                  " b = 6*0,\n b(1:5:2) = 1, 3, 5,\n"
                  " c = 1, 2, 3,\n c(:2) = 9, 8,\n"
                  " d = 1, 2,\n d(2:) = 7, 8, 9,")
    assert parsed["a"] == [1, 20, 30, 40, 5, 6]
    assert parsed["b"] == [1, 0, 3, 0, 5, 0]
    assert parsed["c"] == [9, 8, 3]
    assert parsed["d"] == [1, 7, 8, 9]


def test_a_section_given_fewer_values_keeps_the_rest():
    assert _one(" a = 1, 2, 3,\n a(1:3) = 9,")["a"] == [9, 2, 3]


def test_a_repeated_name_overwrites_only_the_elements_it_gives():
    # Each name-value subsequence is its own assignment; WRF's read does
    # not reset the array between them.
    assert _one(" a = 1, 2, 3,\n a = 7,")["a"] == [7, 2, 3]


def test_null_values_keep_what_an_earlier_assignment_set():
    assert _one(" a = 1, 2, 3,\n a = , 20,")["a"] == [1, 20, 3]
    assert _one(" a = 1, 2, 3,\n a = 2*, 30,")["a"] == [1, 2, 30]
    assert _one(" a = 1, 2, 3,\n a = 10 , , 30")["a"] == [10, 2, 30]


def test_trailing_nulls_are_simply_absent():
    assert _one(" a = 1, , ,")["a"] == [1]
    assert _one(" a = 3*")["a"] == []
    assert _one(" a =\n b = 2")["a"] == []


def test_quoted_values_carry_separators_comments_and_slashes():
    parsed = _one(
        " p = '/data/geog, v2!', q = \"a = b\", r = 'it''s', "
        "s = \"say \"\"hi\"\"\", t = '', u = 2*'x, y'")
    assert parsed == {"p": ["/data/geog, v2!"], "q": ["a = b"],
                      "r": ["it's"], "s": ['say "hi"'], "t": [""],
                      "u": ["x, y", "x, y"]}


def test_comments_anywhere_outside_strings():
    parsed = parse_namelist_text(
        "! leading comment & not a group\n"
        "&share ! the group line may carry one\n"
        " max_dom = 2, ! trailing\n"
        " wrf_core = 'ARW' ! a = 3\n"
        "! a whole-line comment inside the group\n"
        " dx = 1, 2 ! values continue\n"
        "      3,\n"
        "/ ! after the terminator\n")
    assert parsed == {"share": {"max_dom": [2], "wrf_core": ["ARW"],
                                "dx": [1, 2, 3]}}


def test_a_comment_straight_after_a_comma_is_an_empty_value_as_gfortran_reads_it():
    # Measured with gfortran 15.2 (the oracle test below re-measures it
    # wherever a gfortran is installed): the comment counts as a value
    # separator when a value follows it, not when it follows a value or
    # sits on a line of its own.
    parsed = _one(" a = 1, ! x\n 2\n b = ! y\n 5\n c = 1 ! z\n , 2\n"
                  " d = 1,\n ! w\n , 2, ! v\n , 3\n e = 1, ! trailing\n f = 2",
                  allow_unset=True)
    assert parsed == {"a": [1, None, 2], "b": [None, 5], "c": [1, 2],
                      "d": [1, None, 2, None, 3], "e": [1], "f": [2]}
    with pytest.raises(ValueError, match="comment straight after a comma"):
        _one(" a = 1, ! x\n 2")


def test_logical_spellings():
    parsed = _one(" a = .true., .false., T, F, .t., .f., true, False, "
                  ".TRUE., t,")
    assert parsed["a"] == [True, False, True, False, True, False, True,
                           False, True, True]


def test_numbers_repeat_counts_and_d_exponents():
    parsed = _one(" a = 3*1.0, 2*-4, b = 2.90D2, 1.0d-2, -2.5E+1,"
                  " c = 08, +5, 1., .5, 1_000")
    assert parsed["a"] == [1.0, 1.0, 1.0, -4, -4]
    assert parsed["b"] == [290.0, 0.01, -25.0]
    assert parsed["c"] == [8, 5, 1.0, 0.5, "1_000"]


def test_end_style_terminators_and_dollar_groups():
    parsed = parse_namelist_text(
        "&a\n x = 1,\n&end\n$b\n y = 2,\n$end\n&c z = 3 &end\n")
    assert parsed == {"a": {"x": [1]}, "b": {"y": [2]}, "c": {"z": [3]}}


def test_names_and_groups_are_case_blind():
    assert parse_namelist_text("&Time_Control\n Run_Hours = 6\n/\n") == \
        {"time_control": {"run_hours": [6]}}


def test_a_byte_order_mark_does_not_hide_the_first_group(tmp_path):
    path = tmp_path / "namelist.wps"
    path.write_bytes(b"\xef\xbb\xbf&share\n max_dom = 1,\n/\n")
    assert parse_namelist(path) == {"share": {"max_dom": [1]}}


def test_scan_spans_cover_exactly_one_assignment():
    text = "&g\n a = 1, b = 2, 3 c = 'x'\n/\n"
    (group,) = scan_namelist_text(text)
    spans = [text[a.start:a.end] for a in group.assignments]
    assert spans == ["a = 1,", "b = 2, 3", "c = 'x'"]
    assert group.terminator == (text.index("/"), text.index("/") + 1)


# ---------------------------------------------------------------------------
# Refusals: each names what reading on would break
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("body, match", [
    (" e_we = 100, , 300,", r"e_we leaves element\(s\) 2 unset"),
    (" e_we = 2*, 300,", r"e_we leaves element\(s\) 1, 2 unset"),
    (" e_we(3) = 300,", r"e_we leaves element\(s\) 1, 2 unset"),
])
def test_an_unset_element_before_a_set_one_is_refused(body, match):
    # The line reader dropped the empty slot and shifted 300 into d02.
    with pytest.raises(ValueError, match=match):
        _one(body)
    assert _one(body, allow_unset=True)["e_we"][-1] == 300
    assert None in _one(body, allow_unset=True)["e_we"]


def test_an_unquoted_slash_ends_the_group_and_is_refused():
    with pytest.raises(ValueError, match="follows the '/' that ends &geogrid"):
        parse_namelist_text(
            "&geogrid\n geog_data_path = /data/geog\n dx = 3000,\n/\n")


def test_a_second_group_of_the_same_name_is_refused():
    with pytest.raises(ValueError, match="a second &physics group"):
        parse_namelist_text("&physics\n mp_physics = 8,\n/\n"
                            "&physics\n cu_physics = 1,\n/\n")


@pytest.mark.parametrize("body, match", [
    (" a(1,2) = 3", "multi-dimensional subscript"),
    (" a(1:4:0) = 3", "zero stride"),
    (" a(1:2) = 1, 2, 3", "3 values for the 2-element section"),
    (" a(0) = 1", "below 1"),
    (" 5, a = 1", "neither a 'name =' nor a value"),
    (" , a = 1", "value separator before any 'name ='"),
    (" a = 'open", "never closes"),
    (" a = 0*1", "positive repeat count"),
    (" a = 'x'y", "directly after a value"),
])
def test_malformed_input_is_refused_by_name(body, match):
    with pytest.raises(ValueError, match=match):
        _one(body)


def test_refusals_name_the_file_and_line(tmp_path):
    path = tmp_path / "namelist.input"
    path.write_text("&domains\n max_dom = 1,\n e_we(1,1) = 5,\n/\n")
    with pytest.raises(ValueError, match=re.escape(f"{path}: namelist line 3")):
        parse_namelist(path)


# ---------------------------------------------------------------------------
# Whole namelists: one key per line and packed several per line agree
# ---------------------------------------------------------------------------

#: WRF 4.7.1's shipped test/em_real/namelist.input, verbatim (trailing
#: blanks included).
WRF_EM_REAL_NAMELIST_INPUT = """\
 &time_control
 run_days                            = 0,
 run_hours                           = 36,
 run_minutes                         = 0,
 run_seconds                         = 0,
 start_year                          = 2019, 2019,
 start_month                         = 09,   09,
 start_day                           = 04,   04,
 start_hour                          = 12,   12,
 end_year                            = 2019, 2019,
 end_month                           = 09,   09,
 end_day                             = 06,   06,
 end_hour                            = 00,   00,
 interval_seconds                    = 10800
 input_from_file                     = .true.,.true.,
 history_interval                    = 60,  60,
 frames_per_outfile                  = 1, 1,
 restart                             = .false.,
 restart_interval                    = 7200,
 io_form_history                     = 2
 io_form_restart                     = 2
 io_form_input                       = 2
 io_form_boundary                    = 2
 /

 &domains
 time_step                           = 90,
 time_step_fract_num                 = 0,
 time_step_fract_den                 = 1,
 max_dom                             = 2,
 e_we                                = 150,    220,
 e_sn                                = 130,    214,
 e_vert                              = 48,     48,
 dzbot                               = 30.
 dzstretch_s                         = 1.11
 dzstretch_u                         = 1.10
 p_top_requested                     = 5000,
 num_metgrid_levels                  = 34,
 num_metgrid_soil_levels             = 4,
 dx                                  = 15000,
 dy                                  = 15000,
 grid_id                             = 1,     2,
 parent_id                           = 0,     1,
 i_parent_start                      = 1,     53,
 j_parent_start                      = 1,     25,
 parent_grid_ratio                   = 1,     3,
 parent_time_step_ratio              = 1,     3,
 feedback                            = 1,
 smooth_option                       = 0
 /

 &physics
 physics_suite                       = 'CONUS'
 mp_physics                          = -1,    -1,
 cu_physics                          = -1,    -1,
 ra_lw_physics                       = -1,    -1,
 ra_sw_physics                       = -1,    -1,
 bl_pbl_physics                      = -1,    -1,
 sf_sfclay_physics                   = -1,    -1,
 sf_surface_physics                  = -1,    -1,
 radt                                = 15,    15,
 bldt                                = 0,     0,
 cudt                                = 0,     0,
 icloud                              = 1,
 num_land_cat                        = 21,
 sf_urban_physics                    = 0,     0,
 fractional_seaice                   = 1,
 /

 &fdda
 /

 &dynamics
 hybrid_opt                          = 2,
 w_damping                           = 0,
 diff_opt                            = 2,      2,
 km_opt                              = 4,      4,
 diff_6th_opt                        = 0,      0,
 diff_6th_factor                     = 0.12,   0.12,
 base_temp                           = 290.
 damp_opt                            = 3,
 zdamp                               = 5000.,  5000.,
 dampcoef                            = 0.2,    0.2,
 khdif                               = 0,      0,
 kvdif                               = 0,      0,
 non_hydrostatic                     = .true., .true.,
 moist_adv_opt                       = 1,      1,
 scalar_adv_opt                      = 1,      1,
 gwd_opt                             = 1,      0,
 /

 &bdy_control
 spec_bdy_width                      = 5,
 specified                           = .true.
 /

 &grib2
 /

 &namelist_quilt
 nio_tasks_per_group = 0,
 nio_groups = 1,
 /
"""

_STARTS_ASSIGNMENT = re.compile(r"[A-Za-z][A-Za-z0-9_%]*\s*(\([^)]*\))?\s*=")


def _strip_comment(line: str) -> str:
    quote = None
    for index, char in enumerate(line):
        if quote is not None:
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char == "!":
            return line[:index]
    return line


def pack(text: str, *, per_line: int = 3, joiner: str = " ") -> str:
    """Re-lay a one-key-per-line namelist several assignments per line.

    Written independently of the reader under test: a line inside a group
    that starts with ``name =`` opens an assignment, any other non-blank
    line continues the previous one.  ``joiner=", "`` also drops each
    assignment's trailing comma, giving exactly the A142 shape
    ``a = 1, b = 2, c = 3,``.
    """

    out: list[str] = []
    items: list[str] = []
    current: str | None = None
    in_group = False

    def flush():
        nonlocal current
        if current is not None:
            items.append(current)
            current = None
        for start in range(0, len(items), per_line):
            chunk = items[start:start + per_line]
            if joiner == ", ":
                chunk = [item.rstrip().rstrip(",") for item in chunk]
                out.append(" " + ", ".join(chunk) + ",")
            else:
                out.append(" " + joiner.join(chunk))
        items.clear()

    for raw in text.splitlines():
        line = _strip_comment(raw).strip()
        if not in_group:
            if line.startswith(("&", "$")):
                in_group = True
                out.append(line)
            continue
        if line == "/" or line.lower() in ("&end", "$end"):
            flush()
            out.append("/")
            in_group = False
        elif _STARTS_ASSIGNMENT.match(line):
            if current is not None:
                items.append(current)
            current = line
        elif line:
            current = (current or "") + " " + line
    return "\n".join(out) + "\n"


def _assert_packing_invariant(text: str) -> dict:
    one_per_line = parse_namelist_text(text)
    for joiner in (" ", ", "):
        for per_line in (2, 3, 1000):
            packed = pack(text, per_line=per_line, joiner=joiner)
            assert parse_namelist_text(packed) == one_per_line, (
                joiner, per_line, packed)
    return one_per_line


def test_wrf_em_real_namelist_input_one_per_line_and_packed_agree():
    parsed = _assert_packing_invariant(WRF_EM_REAL_NAMELIST_INPUT)
    packed = pack(WRF_EM_REAL_NAMELIST_INPUT, joiner=", ")
    assert re.search(r"\n run_days +?= 0, run_hours +?= 36, run_minutes +?= 0,",
                     packed)
    assert parsed["time_control"]["start_month"] == [9, 9]
    assert parsed["time_control"]["interval_seconds"] == [10800]
    assert parsed["time_control"]["input_from_file"] == [True, True]
    assert parsed["domains"]["dzbot"] == [30.0]
    assert parsed["domains"]["i_parent_start"] == [1, 53]
    assert parsed["physics"]["physics_suite"] == ["CONUS"]
    assert parsed["dynamics"]["gwd_opt"] == [1, 0]
    assert parsed["fdda"] == {} and parsed["grib2"] == {}
    assert parsed["namelist_quilt"] == {"nio_tasks_per_group": [0],
                                        "nio_groups": [1]}
    assert len(parsed) == 8


def _repo_namelists() -> list[Path]:
    return sorted(path for pattern in ("*namelist.input*", "*namelist.wps*")
                  for path in (REPO / "configs").rglob(pattern)
                  if path.is_file())


def test_the_repo_carries_namelists_to_check():
    assert len(_repo_namelists()) >= 50


@pytest.mark.parametrize(
    "path", _repo_namelists(),
    ids=lambda path: str(path.relative_to(REPO)).replace(os.sep, "/"))
def test_every_shipped_namelist_parses_the_same_packed(path):
    _assert_packing_invariant(path.read_text(encoding="utf-8-sig"))


def _wrf_distribution_namelists() -> list[Path]:
    """WRF/WPS distribution namelists under GPUWM_WRF_NAMELIST_ROOTS.

    os.pathsep-separated directories (a WRF source tree's test/ and a WPS
    tree), searched for namelist.input* and namelist.wps*; unset skips.
    """

    roots = [Path(root) for root in
             os.environ.get("GPUWM_WRF_NAMELIST_ROOTS", "").split(os.pathsep)
             if root]
    return sorted(path for root in roots
                  for pattern in ("namelist.input*", "namelist.wps*")
                  for path in root.rglob(pattern) if path.is_file())


@pytest.mark.skipif(not _wrf_distribution_namelists(),
                    reason="GPUWM_WRF_NAMELIST_ROOTS names no WRF/WPS tree")
def test_every_wrf_distribution_namelist_parses_the_same_packed():
    paths = _wrf_distribution_namelists()
    for path in paths:
        parsed = _assert_packing_invariant(
            path.read_text(encoding="utf-8-sig", errors="strict"))
        assert parsed, path


# ---------------------------------------------------------------------------
# The Fortran runtime itself is the referee where a compiler is present
# ---------------------------------------------------------------------------

_ORACLE_SOURCE = """\
program nmlcheck
  implicit none
  integer :: a(8), b(8), c(8), d(8), start_year(4), start_month(4), &
             start_day(4), i, ios
  real(8) :: r(8)
  logical :: l(10)
  character(len=40) :: s(4)
  character(len=512) :: path
  namelist /g/ a, b, c, d, start_year, start_month, start_day, r, l, s
  a = -999; b = -999; c = -999; d = -999
  start_year = -999; start_month = -999; start_day = -999
  r = -999.0d0; l = .false.; s = 'UNSET'
  call get_command_argument(1, path)
  open(10, file=trim(path), status='old')
  read(10, nml=g, iostat=ios)
  write(*,'(A,1X,I0)') 'ios', ios
  do i = 1, 8
    write(*,'(A,1X,I0,1X,I0)') 'a', i, a(i)
    write(*,'(A,1X,I0,1X,I0)') 'b', i, b(i)
    write(*,'(A,1X,I0,1X,I0)') 'c', i, c(i)
    write(*,'(A,1X,I0,1X,I0)') 'd', i, d(i)
    write(*,'(A,1X,I0,1X,ES24.16)') 'r', i, r(i)
  end do
  do i = 1, 4
    write(*,'(A,1X,I0,1X,I0)') 'start_year', i, start_year(i)
    write(*,'(A,1X,I0,1X,I0)') 'start_month', i, start_month(i)
    write(*,'(A,1X,I0,1X,I0)') 'start_day', i, start_day(i)
    write(*,'(A,1X,I0,1X,A)') 's', i, '[' // trim(s(i)) // ']'
  end do
  do i = 1, 10
    write(*,'(A,1X,I0,1X,L1)') 'l', i, l(i)
  end do
end program nmlcheck
"""

#: Namelists the runtime and this reader must read the same way.
_ORACLE_CASES = {
    "packed": "&g\n start_year = 2026, 2026, start_month = 08, 08,"
              " start_day = 25, 25,\n/\n",
    "blanks": "&g\n a = 1, 2 b = 3 4 c = 5,\n 6, d=7\n/\n",
    "elements": "&g\n a = 1, 2, 3, 4,\n a(2) = 20,\n b(3) = 30, 40,\n"
                " c = 6*0,\n c(1:5:2) = 1, 3, 5,\n d = 1, 2, 3, d(:2) = 9, 8\n/\n",
    "open-section": "&g\n a = 1, 2,\n a(2:) = 7, 8, 9\n/\n",
    "overlay-and-nulls": "&g\n a = 1, 2, 3,\n a = 7,\n b = 1, 2, 3,\n"
                         " b = , 20,\n c = 10 , , 30\n d = 2*5, 2*, 6\n/\n",
    "strings": "&g\n s = 'x, y!', \"a = b\", 'it''s', '/p/q'\n/\n",
    "logicals": "&g\n l = .true., .false., T, F, .t., .f., true, False,"
                " .TRUE., t\n/\n",
    "reals": "&g\n r = 2.90D2, 1.0d-2, -2.5E+1, 3*1.5, 08\n/\n",
    "one-line": "junk before the group\n&g a = 1 b = 2 /\n",
    "end": "&g\n a = 1,\n&end\n",
    "dollar": "$g\n a = 5\n$end\n",
    "comments": "&g ! the group line\n a = 1, ! x\n 2 ! y\n/\n",
    "comment-positions": "&g\n a = 1, ! x\n 2\n b = ! y\n 5\n c = 1 ! z\n , 2\n"
                         " d = 1,\n ! w\n , 2, ! v\n , 3\n/\n",
}


def _oracle_values(output: str) -> tuple[int, dict]:
    sentinels = {"r": -999.0, "s": "UNSET", "l": None}
    by_name: dict[str, dict[int, object]] = {}
    ios = None
    for line in output.splitlines():
        name, *rest = line.split(None, 2)
        if name == "ios":
            ios = int(rest[0])
            continue
        index, raw = int(rest[0]), rest[1]
        if name == "r":
            value = float(raw)
        elif name == "s":
            value = raw[1:-1]
        elif name == "l":
            value = raw == "T"
        else:
            value = int(raw)
        if value != sentinels.get(name, -999):
            by_name.setdefault(name, {})[index] = value
    table = {}
    for name, elements in by_name.items():
        top = max(elements)
        table[name] = [elements.get(i) for i in range(1, top + 1)]
    return ios, table


@pytest.mark.skipif(
    __import__("shutil").which("gfortran") is None,
    reason="no gfortran to read the namelists with")
@pytest.mark.parametrize("case", sorted(_ORACLE_CASES))
def test_gfortran_reads_each_form_the_same_way(tmp_path, case):
    import subprocess

    source = tmp_path / "nmlcheck.f90"
    source.write_text(_ORACLE_SOURCE)
    binary = tmp_path / "nmlcheck"
    subprocess.run(["gfortran", "-O0", "-o", str(binary), str(source)],
                   check=True, capture_output=True)
    text = _ORACLE_CASES[case]
    namelist = tmp_path / "input.nml"
    namelist.write_text(text)
    ran = subprocess.run([str(binary), str(namelist)], check=True,
                         capture_output=True, text=True)
    ios, runtime = _oracle_values(ran.stdout)
    assert ios == 0, ran.stdout
    parsed = parse_namelist_text(text, allow_unset=True)["g"]
    # l starts .false., so the runtime cannot tell an unset logical from a
    # false one: it is compared only where the case sets all ten.
    if "l" in parsed:
        assert len(parsed["l"]) == 10
    else:
        runtime.pop("l")
    assert {key: values for key, values in parsed.items() if values} == runtime
