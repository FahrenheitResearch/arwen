"""The mapped prep door answers users in refusals, never tracebacks.

The 2.5.0 persona walks (UX finding N6) drove ``gpuwm prep --source
hrrr-prs`` with a config written by the REAL ``gpuwm import-namelist``
from an ordinary WRF 4.6 namelist pair -- ``e_vert = 45`` and no
explicit ``eta_levels``, which is what every stock WRF case looks like
because real.exe generates the ladder itself.  The door answered with
three raw Python tracebacks, and the second and third formed a circle:

1. a missing ``--experiment-config`` file raised a bare
   ``FileNotFoundError`` with no flag name and no remedy;
2. the imported 44 mass levels met the mapping's fixed reference count
   of 49 as ``ValueError: mapped target vertical levels differ``;
3. matching the count (nz = 49) then raised ``explicit eta_levels has
   shape (0,)`` -- demanding a ladder ``import-namelist`` never writes
   and no WRF user types, so no edit the previous message suggested
   could ever terminate.

Every test here subprocesses the real CLI door, exactly as the walk
did.  The contract: a user-reachable refusal exits with the shared
preparation-refusal status, prints the sentence and ITS remedy on
stderr, and never a traceback; and the circle is broken because a
config that CARRIES an explicit ladder is adopted at its own level
count, while a config without one gets both counts and both
reconciling doors named in one message.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from gpuwm.ingest.source_coverage import PREPARATION_REFUSAL_EXIT_CODE

REPO = Path(__file__).resolve().parents[1]

#: The subprocesses run from the walk's own directory (that is the
#: point: pasted relative paths resolve exactly as the persona's did),
#: so THIS tree's gpuwm must win over any installed one.
_ENV = {
    **os.environ,
    "PYTHONPATH": os.pathsep.join(
        entry for entry in (str(REPO), os.environ.get("PYTHONPATH"))
        if entry),
}

# The WRF namelist pair of the persona walk: the import suite's
# fixture pair with the explicit eta ladder REMOVED and e_vert raised
# to 45, which is the shape of an ordinary stock-WRF case (real.exe
# derives the ladder, so nobody types one).
_WPS_TEXT = """\
&share
 wrf_core = 'ARW',
 max_dom = 2,
 start_date = '1999-05-03_12:00:00', '1999-05-03_12:00:00',
 end_date   = '1999-05-03_18:00:00', '1999-05-03_18:00:00',
 interval_seconds = 21600,
 io_form_geogrid = 2,
/
&geogrid
 parent_id         = 1, 1,
 parent_grid_ratio = 1, 3,
 i_parent_start    = 1, 40,
 j_parent_start    = 1, 30,
 e_we              = 101, 61,
 e_sn              = 81, 61,
 geog_data_res     = 'default', 'default',
 dx = 12000,
 dy = 12000,
 map_proj = 'lambert',
 ref_lat   = 39.7,
 ref_lon   = -83.9,
 truelat1  = 30.0,
 truelat2  = 60.0,
 stand_lon = -83.9,
 geog_data_path = '/geog',
/
&ungrib
 out_format = 'WPS',
 prefix = 'ERA5',
/
&metgrid
 fg_name = 'ERA5',
/
"""

_INPUT_TEXT = """\
&time_control
 run_hours = 6,
 start_year = 1999, 1999,
 start_month = 05, 05,
 start_day = 03, 03,
 start_hour = 12, 12,
 end_year = 1999, 1999,
 end_month = 05, 05,
 end_day = 03, 03,
 end_hour = 18, 18,
 interval_seconds = 21600,
 input_from_file = .true., .true.,
 history_interval = 60, 15,
 restart = .false.,
 restart_interval = 60,
/
&domains
 time_step = 60,
 max_dom = 2,
 e_we = 101, 61,
 e_sn = 81, 61,
 e_vert = 45, 45,
 p_top_requested = 5000,
 dx = 12000.0, 4000.0,
 dy = 12000.0, 4000.0,
 grid_id = 1, 2,
 parent_id = 0, 1,
 i_parent_start = 1, 40,
 j_parent_start = 1, 30,
 parent_grid_ratio = 1, 3,
 parent_time_step_ratio = 1, 3,
 feedback = 0,
 smooth_option = 0,
/
&physics
 mp_physics = 55, 55,
 ra_lw_physics = 4, 4,
 ra_sw_physics = 4, 4,
 radt = 12, 3,
 sf_sfclay_physics = 91, 91,
 sf_surface_physics = 2, 2,
 bl_pbl_physics = 11, 11,
 bldt = 0, 0,
 cu_physics = 1, 0,
 cudt = 5, 0,
/
&dynamics
 hybrid_opt = 2,
 etac = 0.2,
 w_damping = 1,
 epssm = 0.5,
 diff_opt = 2, 2,
 km_opt = 4, 4,
 mix_full_fields = .true., .true.,
 diff_6th_opt = 2, 2,
 diff_6th_factor = 0.12, 0.10,
 diff_6th_slopeopt = 1, 1,
 base_temp = 2.90D2,
 damp_opt = 3,
 zdamp = 2*5000.,
 dampcoef = 0.2, 0.2,
 khdif = 0, 0,
 kvdif = 0, 0,
 non_hydrostatic = 2*.true.,
 use_theta_m = 0,
 moist_adv_opt = 1, 1,
/
&bdy_control
 spec_bdy_width = 5,
 spec_zone = 1,
 relax_zone = 4,
 specified = .true., .false.,
 nested = .false., .true.,
/
"""


def _run(argv: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "gpuwm.cli", *argv],
        capture_output=True, text=True, cwd=cwd, timeout=600, env=_ENV,
    )


@pytest.fixture(scope="module")
def walk(tmp_path_factory):
    """The persona walk's estate: imported config, staged bytes, geog."""

    root = tmp_path_factory.mktemp("prep-door-walk")
    (root / "namelist.wps").write_text(_WPS_TEXT, encoding="utf-8")
    (root / "namelist.input").write_text(_INPUT_TEXT, encoding="utf-8")

    # The REAL importer writes the config, exactly as the walk did.
    imported = _run(
        ["import-namelist", "namelist.wps", "namelist.input",
         "--output", "mycase.toml"],
        cwd=root,
    )
    assert imported.returncode == 0, imported.stderr
    config = (root / "mycase.toml").read_text(encoding="utf-8")
    assert "nz = 44" in config, config
    # No LADDER -- the importer mentions eta_levels in a comment, but a
    # stock WRF namelist carries none to translate, so no key lands.
    assert "eta_levels =" not in config, config

    data = root / "data"
    data.mkdir()
    for name in ("hrrr.t21z.wrfprsf00.grib2", "hrrr.t21z.wrfprsf01.grib2"):
        (data / name).write_bytes(b"GRIB-fixture-bytes-" + name.encode())
    (data / "inputs.txt").write_text(
        "".join(f"{data / name}\n" for name in (
            "hrrr.t21z.wrfprsf00.grib2", "hrrr.t21z.wrfprsf01.grib2")),
        encoding="utf-8",
    )
    (root / "geog").mkdir()
    return root


def _prep(walk: Path, *, config: str, out: str) -> subprocess.CompletedProcess:
    data = walk / "data"
    manifest = data / f"inputs-{out}.json"
    return _run(
        [
            "prep", "--source", "hrrr-prs",
            "--input-list", str(data / "inputs.txt"),
            "--supplement",
            f"hrrr_prs_in_band_surface={data / 'hrrr.t21z.wrfprsf00.grib2'}",
            "--supplement",
            f"hrrr_prs_in_band_surface={data / 'hrrr.t21z.wrfprsf01.grib2'}",
            "--author-input-manifest", str(manifest),
            "--wps-namelist", str(walk / "namelist.wps"),
            "--experiment-config", str(walk / config),
            "--geog-root", str(walk / "geog"),
            "--output-root", str(walk / "out" / out),
        ],
        cwd=walk,
    )


def test_missing_experiment_config_is_a_named_refusal(walk):
    """A missing config file names its flag, its path, and its writers."""

    result = _prep(walk, config="configs/missing.toml", out="missing-config")

    assert "Traceback" not in result.stderr, result.stderr
    assert result.returncode == PREPARATION_REFUSAL_EXIT_CODE, (
        result.returncode, result.stderr)
    assert "--experiment-config" in result.stderr
    # The RESOLVED path, which is what tells a user their pasted
    # relative path resolved against the wrong working directory.
    assert str((walk / "configs" / "missing.toml").resolve()) \
        in result.stderr, result.stderr
    assert "remedy" in result.stderr
    assert "import-namelist" in result.stderr


def test_vertical_mismatch_without_ladder_names_both_values_and_doors(walk):
    """44 vs 49 without a ladder: one message, both counts, both doors."""

    result = _prep(walk, config="mycase.toml", out="mismatch")

    assert "Traceback" not in result.stderr, result.stderr
    assert result.returncode == PREPARATION_REFUSAL_EXIT_CODE, (
        result.returncode, result.stderr)
    stderr = result.stderr
    # BOTH values, in the user's own vocabulary as well as gpuwm's.
    assert "44" in stderr and "49" in stderr, stderr
    assert "e_vert=45" in stderr, stderr
    # BOTH doors that reconcile them, so no edit leads back here.
    assert "eta_levels" in stderr, stderr
    assert "gpuwm domain" in stderr, stderr
    assert "remedy" in stderr, stderr


def test_count_matched_config_without_ladder_gets_the_same_door(walk):
    """nz=49 without a ladder must not resurrect the shape-(0,) wall.

    This is the second arc of the walk's circle: the user obeys the
    mismatch refusal by matching the count, and the old door answered
    with ``explicit eta_levels has shape (0,)`` -- a traceback
    demanding a ladder nothing they ran had ever written.  The answer
    must be the same named ladder refusal with the same two doors.
    """

    source = (walk / "mycase.toml").read_text(encoding="utf-8")
    assert "\nnz = 44\n" in source, source
    (walk / "mycase49.toml").write_text(
        source.replace("\nnz = 44\n", "\nnz = 49\n"), encoding="utf-8")

    result = _prep(walk, config="mycase49.toml", out="count-matched")

    assert "Traceback" not in result.stderr, result.stderr
    assert result.returncode == PREPARATION_REFUSAL_EXIT_CODE, (
        result.returncode, result.stderr)
    assert "shape (0,)" not in result.stderr, result.stderr
    assert "eta_levels" in result.stderr
    assert "gpuwm domain" in result.stderr
    assert "remedy" in result.stderr


def test_config_carrying_an_explicit_ladder_is_adopted(walk):
    """An explicit 45-interface ladder walks PAST the vertical contract.

    The circle-breaker: prep adopts the imported vertical ladder when
    the config carries one, at the config's own level count, instead of
    refusing it against the mapping's fixed reference count.  The run
    still fails later on this estate (the staged bytes are fixture
    bytes, the geog tree is empty), but it must get past the vertical
    wall without either sentence the old circle was built from.
    """

    ladder = [1.0 - (index / 45.0) ** 1.3 for index in range(44)] + [0.0]
    lines = ",\n    ".join(repr(value) for value in ladder)
    source = (walk / "mycase.toml").read_text(encoding="utf-8")
    assert "\nnz = 44\n" in source, source
    (walk / "mycase-ladder.toml").write_text(
        source.replace(
            "\nnz = 44\n",
            "\nnz = 44\neta_levels = [\n    " + lines + ",\n]\n"),
        encoding="utf-8")

    result = _prep(walk, config="mycase-ladder.toml", out="adopted")

    assert "vertical levels differ" not in result.stderr, result.stderr
    assert "shape (0,)" not in result.stderr, result.stderr
    assert "vertical ladder is missing" not in result.stderr, result.stderr
