"""The tile I/O gate's checkpoint header comparison, without a card.

``python -m tilestream.test_io`` compares a checkpoint written out of the
streamed store against one written by ``restart.write_restart`` from the
same resident state.  The resident writer stamps ``written_mode`` (which
memory road wrote the file) and the streamed writer does not, so the gate
reported four equivalent checkpoints as different on the header alone.

``written_mode`` is provenance, never identity
(``gpuwm.io.restart.written_mode_note``), so the comparison exempts it
beside ``created`` and instead checks the streamed file does not claim the
resident road.  This suite pins that verdict on plain dicts: it is the same
function the gate calls, and it runs anywhere.
"""

from __future__ import annotations

from types import SimpleNamespace

from gpuwm.io import restart
from tilestream import test_io

_CFG = SimpleNamespace(ny=40, nx=48)


def _mono_header() -> dict:
    return {
        "format_version": restart.RESTART_FORMAT_VERSION,
        "case": "demo",
        "created": "2026-09-29T00:00:00+00:00",
        "elapsed_seconds": 36.0,
        "config": {"nx": 48, "ny": 40},
        restart.WRITTEN_MODE_HEADER_KEY: restart.written_mode_note(
            restart.RESIDENT_WRITTEN_MODE, _CFG),
    }


def _streamed_header() -> dict:
    header = _mono_header()
    header["created"] = "2026-09-29T00:00:07+00:00"
    del header[restart.WRITTEN_MODE_HEADER_KEY]
    return header


def test_the_written_mode_stamp_is_provenance_not_a_header_difference():
    record = test_io.header_equivalence(_mono_header(), _streamed_header())
    assert record["keys_equal"] is True
    assert record["values_equal"] is True, record["value_diff"]
    assert record["written_mode_ok"] is True
    assert record["streamed_written_mode"] is None


def test_a_streamed_stamp_naming_the_streamed_road_is_accepted():
    streamed = _streamed_header()
    streamed[restart.WRITTEN_MODE_HEADER_KEY] = restart.written_mode_note(
        restart.STREAMED_WRITTEN_MODE, _CFG, store="host")
    record = test_io.header_equivalence(_mono_header(), streamed)
    assert record["values_equal"] is True, record["value_diff"]
    assert record["streamed_written_mode"] == restart.STREAMED_WRITTEN_MODE
    assert record["written_mode_ok"] is True


def test_a_streamed_file_claiming_the_resident_road_is_caught():
    streamed = _streamed_header()
    streamed[restart.WRITTEN_MODE_HEADER_KEY] = restart.written_mode_note(
        restart.RESIDENT_WRITTEN_MODE, _CFG)
    record = test_io.header_equivalence(_mono_header(), streamed)
    assert record["written_mode_ok"] is False


def test_a_real_header_value_difference_is_still_caught():
    streamed = _streamed_header()
    streamed["elapsed_seconds"] = 72.0
    record = test_io.header_equivalence(_mono_header(), streamed)
    assert record["values_equal"] is False
    assert record["value_diff"] == ["elapsed_seconds"]


def test_a_header_key_on_one_side_only_is_still_caught():
    """The drift guard for ``store_restart_header`` survives the exemption."""
    mono = _mono_header()
    mono["root_external_lbc_clock"] = {"start": 0}
    record = test_io.header_equivalence(mono, _streamed_header())
    assert record["keys_equal"] is False
    assert record["value_diff"] == ["root_external_lbc_clock"]
