"""The IO boundary's three cheap promises: naming, clocks, and provenance.

Each of these was a place where the layer said something it could not
deliver, and each failure is silent by construction -- no exception, no log
line, just a file that is not there or a number that is not true:

* a legal sub-second history/restart cadence formatted to a whole-second
  name, so distinct frames replaced each other and the run lost its own
  output (K-01);
* a wrfout and a checkpoint that could not name the build that wrote them,
  once separated from the run's logs (K-03);
* a restart header whose ``elapsed_seconds`` could be ``NaN`` or negative
  and be restored, poisoning cadence arithmetic downstream of every
  identity check that had just passed (K-06).

The idealized ``Times`` helper's calendar rollover (K-07) is gated beside
the rest of the helper in ``tests/test_wrfout.py``.
"""

from __future__ import annotations

import json
import textwrap
from datetime import datetime
from fractions import Fraction

import netCDF4
import numpy as np
import pytest

import gpuwm
from gpuwm.experiment import _check_whole_second_cadence, load_experiment
from gpuwm.io.restart import producer_identity, restart_filename
from gpuwm.io.wrfout import WrfoutWriter, wrfout_filename

_WHOLE = datetime(2026, 7, 1, 18, 0, 0)


# ---------------------------------------------------------------------------
# K-01 -- distinct legal instants must not alias onto one name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("microsecond", [250000, 500000, 750000, 1])
def test_sub_second_instants_are_refused_by_both_publishers(microsecond):
    """Four sub-second offsets, both publishers.

    One offset would not prove the gate is on the fraction rather than on
    one value of it, and one publisher would not prove the other's name is
    guarded -- the history frame and the standalone checkpoint aliased
    independently.
    """
    instant = _WHOLE.replace(microsecond=microsecond)
    with pytest.raises(ValueError, match="whole second"):
        wrfout_filename(instant)
    with pytest.raises(ValueError, match="whole second"):
        restart_filename(instant)


def test_whole_second_instants_still_name_files():
    """The negative control: the gate refuses fractions, not everything."""
    assert wrfout_filename(_WHOLE) == "wrfout_d01_2026-07-01_18_00_00"
    assert wrfout_filename(_WHOLE, 3) == "wrfout_d03_2026-07-01_18_00_00"
    assert restart_filename(_WHOLE) == (
        "gpuwmrst_d01_2026-07-01_18_00_00.npz")


def test_the_times_record_refuses_a_value_it_would_have_to_truncate(tmp_path):
    """19 characters is the format, so a 22-character value is an error.

    Truncation is what made two instants describe one second even when the
    caller had the fraction in hand.
    """
    path = tmp_path / "wrfout_d01_2026-07-01_18_00_00"
    writer = WrfoutWriter(path, nx=3, ny=2, nz=2, dx=1000.0, dy=1000.0)
    try:
        with pytest.raises(ValueError, match="19-character"):
            writer.write_frame("2026-07-01_18:00:00.25",
                               {"T": np.zeros((2, 2, 3), np.float32)})
    finally:
        writer.abort()


@pytest.mark.parametrize("seconds", ["1/4", "1/2", "3/4", "5/2"])
def test_fractional_publication_cadences_are_refused_at_admission(seconds):
    """Four fractional cadences, on the shared admission check."""
    with pytest.raises(ValueError, match="whole number of seconds"):
        _check_whole_second_cadence(
            "history_interval_s", Fraction(seconds), 1, "unit test")


@pytest.mark.parametrize("seconds", ["0", "1", "900", "3600"])
def test_whole_second_cadences_pass_admission(seconds):
    """The negative control, including WRF's 0 = "every step" convention."""
    _check_whole_second_cadence(
        "history_interval_s", Fraction(seconds), 1, "unit test")


_SUB_SECOND_EXPERIMENT = """\
[experiment]
name = "synth"
start_time = 2026-07-01T18:00:00
run_seconds = 60.0
restart_interval_s = {restart_interval_s}

[shared]
nz = 8
ztop = 12000.0

[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = 40
ny = 30
time_step = 0
time_step_fract_num = 1
time_step_fract_den = 4
dx = 12000.0
history_interval_s = {history_interval_s}
"""


def _experiment(tmp_path, *, history_interval_s, restart_interval_s="0.0"):
    path = tmp_path / "exp.toml"
    path.write_text(textwrap.dedent(_SUB_SECOND_EXPERIMENT.format(
        history_interval_s=history_interval_s,
        restart_interval_s=restart_interval_s)))
    return path


def test_a_quarter_second_history_cadence_is_refused_by_the_loader(tmp_path):
    """End to end, on the cadence the audit's probe actually used.

    A quarter-second interval on a quarter-second step is a whole number of
    steps, so the step-divisibility check passes it.  It was the only check
    there was, and three legal frames then formatted to one file name.
    """
    with pytest.raises(ValueError, match="whole number of seconds"):
        load_experiment(_experiment(tmp_path, history_interval_s="0.25"))


def test_a_sub_second_restart_cadence_is_refused_by_the_loader(tmp_path):
    """The second cadence field, for the same reason and by the same rule."""
    with pytest.raises(ValueError, match="whole number of seconds"):
        load_experiment(_experiment(
            tmp_path, history_interval_s="1.0", restart_interval_s="0.5"))


def test_whole_second_cadences_on_a_sub_second_step_still_load(tmp_path):
    """The negative control: a fractional *step* is still legal."""
    exp = load_experiment(_experiment(
        tmp_path, history_interval_s="1.0", restart_interval_s="2.0"))
    assert exp.domain(1).history_interval_s == 1.0


# ---------------------------------------------------------------------------
# K-03 -- the file can name the build that wrote it
# ---------------------------------------------------------------------------

def test_a_wrfout_names_the_build_that_wrote_it(tmp_path):
    path = tmp_path / "wrfout_d01_2026-07-01_18_00_00"
    with WrfoutWriter(path, nx=3, ny=2, nz=2, dx=1000.0, dy=1000.0,
                      title="a user-configured title") as writer:
        writer.write_frame("2026-07-01_18:00:00",
                           {"T": np.zeros((2, 2, 3), np.float32)})
    with netCDF4.Dataset(path) as ds:
        assert ds.GPUWM_VERSION == gpuwm.__version__
        # TITLE remains the caller's, which is precisely why it could never
        # have been the provenance seal.
        assert ds.TITLE == "a user-configured title"


def test_producer_identity_is_the_installed_distribution():
    identity = producer_identity()
    assert identity["distribution"] == gpuwm.DISTRIBUTION_NAME
    assert identity["version"] == gpuwm.__version__
    assert identity["restart_format_version"] == "5"


# ---------------------------------------------------------------------------
# K-06 -- the model clock a restore will accept
# ---------------------------------------------------------------------------

def _admissible(value):
    from gpuwm.io.restart import _admissible_elapsed_seconds

    return _admissible_elapsed_seconds(value, "unit test")


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf"), -1.0, -1.0e-9,
              True, "600", None])
def test_inadmissible_clocks_are_refused(value):
    """Non-finite, negative, and non-numeric, at more than one of each."""
    from gpuwm.io.restart import RestartManifestError

    with pytest.raises(RestartManifestError):
        _admissible(value)


@pytest.mark.parametrize("value", [0.0, 1.0e-9, 600.0, 86400.0, 21600])
def test_admissible_clocks_pass(value):
    assert _admissible(value) == float(value)


def test_a_written_header_cannot_express_a_non_finite_clock():
    """``allow_nan=False`` is the writer half of the same refusal.

    Python's json emits bare ``NaN``/``Infinity`` tokens by default, which
    are not JSON at all; a reader that accepts them back is downstream of a
    writer that should never have produced them.
    """
    with pytest.raises(ValueError):
        json.dumps({"elapsed_seconds": float("nan")}, allow_nan=False)
    # And the value the writer would have been handed is refused before it
    # reaches json at all.
    from gpuwm.io.restart import RestartManifestError

    with pytest.raises(RestartManifestError):
        _admissible(float("nan"))
