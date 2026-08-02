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
