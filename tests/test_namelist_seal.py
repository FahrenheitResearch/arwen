"""Monotonic stream-window namelist identity tests."""

from datetime import datetime
import hashlib
from pathlib import Path

import pytest

from gpuwm.namelist_seal import namelist_extension_invariant
from gpuwm.stream import _materialize_input_namelist


def _template(path: Path) -> Path:
    path.write_text(
        """&time_control
 run_hours = 1,
 interval_seconds = 3600,
/
&domains
 max_dom = 2,
 time_step = 30,
/
&physics
 mp_physics = 6, 6,
/
""",
        encoding="utf-8",
    )
    return path


def _materialize(tmp_path: Path, *, cycle: datetime, lead: int) -> Path:
    destination = tmp_path / f"f{lead:03d}.input"
    _materialize_input_namelist(
        _template(tmp_path / f"template-{lead}.input"),
        destination,
        cycle=cycle,
        lead=lead,
        domain_starts=[cycle, cycle],
    )
    return destination


@pytest.mark.parametrize("cycle", (
    datetime(2026, 8, 1, 19),
    datetime(2026, 8, 1, 23),
))
def test_monotonic_horizon_changes_keep_one_byte_strict_invariant(
        tmp_path, cycle):
    first = _materialize(tmp_path, cycle=cycle, lead=1)
    second = _materialize(tmp_path, cycle=cycle, lead=2)

    first_bytes = first.read_bytes()
    second_bytes = second.read_bytes()
    assert hashlib.sha256(first_bytes).digest() != \
        hashlib.sha256(second_bytes).digest()
    assert namelist_extension_invariant(
        first, cycle=cycle, run_seconds=3600) == \
        namelist_extension_invariant(
            second, cycle=cycle, run_seconds=7200)


def test_real_immutable_namelist_mutation_changes_the_invariant(tmp_path):
    cycle = datetime(2026, 8, 1, 19)
    first = _materialize(tmp_path, cycle=cycle, lead=1)
    second = _materialize(tmp_path, cycle=cycle, lead=2)
    changed = tmp_path / "changed.input"
    payload = second.read_text(encoding="utf-8")
    assert " time_step = 30," in payload
    changed.write_text(
        payload.replace(" time_step = 30,", " time_step = 31,"),
        encoding="utf-8",
    )

    predecessor = namelist_extension_invariant(
        first, cycle=cycle, run_seconds=3600)
    assert namelist_extension_invariant(
        changed, cycle=cycle, run_seconds=7200) != predecessor


def test_illicit_end_time_mutation_is_refused_before_normalization(tmp_path):
    cycle = datetime(2026, 8, 1, 19)
    second = _materialize(tmp_path, cycle=cycle, lead=2)
    payload = second.read_text(encoding="utf-8")
    assert " end_hour = 21, 21," in payload
    second.write_text(
        payload.replace(" end_hour = 21, 21,", " end_hour = 22, 22,"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="end time differs"):
        namelist_extension_invariant(
            second, cycle=cycle, run_seconds=7200)


def test_materialized_namelist_bytes_are_unchanged_for_one_key_per_line(
        tmp_path):
    cycle = datetime(2026, 8, 1, 19)
    assert _materialize(tmp_path, cycle=cycle, lead=2).read_text(
        encoding="utf-8") == (
        "&time_control\n run_hours = 2,\n interval_seconds = 3600,\n"
        " run_days = 0,\n run_minutes = 0,\n run_seconds = 0,\n"
        " start_year = 2026, 2026,\n start_month = 8, 8,\n"
        " start_day = 1, 1,\n start_hour = 19, 19,\n"
        " start_minute = 0, 0,\n start_second = 0, 0,\n"
        " end_year = 2026, 2026,\n end_month = 8, 8,\n end_day = 1, 1,\n"
        " end_hour = 21, 21,\n end_minute = 0, 0,\n end_second = 0, 0,\n"
        "/\n&domains\n max_dom = 2,\n time_step = 30,\n/\n"
        "&physics\n mp_physics = 6, 6,\n/\n")


_PACKED_TEMPLATE = """&time_control
 run_hours = 1, history_interval = 60, interval_seconds = 3600,
 start_year = 2025,
   2025, start_month = 7, 7, ! packed
 start_year(2) = 2024,
/
&domains
 max_dom = 2, time_step = 30,
/
"""


def _materialize_packed(tmp_path: Path, template: str, *, lead: int) -> Path:
    source = tmp_path / f"packed-{lead}.input"
    source.write_text(template, encoding="utf-8")
    destination = tmp_path / f"packed-f{lead:03d}.input"
    cycle = datetime(2026, 8, 1, 19)
    _materialize_input_namelist(source, destination, cycle=cycle, lead=lead,
                                domain_starts=[cycle, cycle])
    return destination


def test_stream_rewrite_keeps_packed_neighbours_and_replaces_whole_values(
        tmp_path):
    """A142 at the stream door: the line editor replaced everything after
    run_hours on its line (history_interval and interval_seconds were
    lost), kept the continued start_year's second line as a third value,
    and let the later start_year(2) overwrite the new column."""
    from gpuwm.namelist_import import parse_namelist

    written = _materialize_packed(tmp_path, _PACKED_TEMPLATE, lead=2)
    text = written.read_text(encoding="utf-8")
    assert "! packed" in text and "start_year(2)" not in text
    time_control = parse_namelist(written)["time_control"]
    assert time_control["history_interval"] == [60]
    assert time_control["interval_seconds"] == [3600]
    assert time_control["run_hours"] == [2]
    assert time_control["start_year"] == [2026, 2026]
    assert time_control["start_month"] == [8, 8]
    assert time_control["end_hour"] == [21, 21]


def test_stream_rewrite_refuses_a_template_without_the_group(tmp_path):
    # The check the line editor carried compared the update groups with
    # themselves and never fired.
    with pytest.raises(ValueError, match=r"lacks section\(s\) \['time_control'\]"):
        _materialize_packed(tmp_path, "&domains\n max_dom = 2,\n/\n", lead=1)


def test_packed_window_keys_seal_and_their_neighbours_stay_in_identity(
        tmp_path):
    cycle = datetime(2026, 8, 1, 19)
    first = _materialize_packed(tmp_path, _PACKED_TEMPLATE, lead=1)
    second = _materialize_packed(tmp_path, _PACKED_TEMPLATE, lead=2)
    # Re-pack the materialized window keys several per line, as a user's
    # sealed namelist may carry them.
    for path in (first, second):
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace(",\n end_", ", end_"), encoding="utf-8")
        assert " end_year = 2026, 2026, end_month = 8, 8," in \
            path.read_text(encoding="utf-8")
    invariant = namelist_extension_invariant(
        first, cycle=cycle, run_seconds=3600)
    assert namelist_extension_invariant(
        second, cycle=cycle, run_seconds=7200) == invariant
    changed = tmp_path / "changed.input"
    changed.write_text(first.read_text(encoding="utf-8").replace(
        "history_interval = 60", "history_interval = 30"), encoding="utf-8")
    assert namelist_extension_invariant(
        changed, cycle=cycle, run_seconds=3600) != invariant
