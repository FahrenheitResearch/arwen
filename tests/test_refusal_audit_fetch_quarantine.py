"""Repeated force-refetch must preserve each previous download generation."""
from datetime import datetime

import pytest

from gpuwm import fetch_routes


class _FrozenClock:
    @staticmethod
    def now():
        return datetime(2026, 9, 27, 12, 0, 0)


@pytest.mark.parametrize('relative', ['field.grib2', 'upstream/source/field.grib2'])
def test_two_refetches_in_one_clock_tick_keep_both_previous_downloads(tmp_path, monkeypatch, relative):
    monkeypatch.setattr(fetch_routes, 'datetime', _FrozenClock)
    source = tmp_path / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b'first download')
    first = fetch_routes._quarantine(tmp_path, lambda _: None)
    assert (first / relative).read_bytes() == b'first download'
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b'second download')
    second = fetch_routes._quarantine(tmp_path, lambda _: None)
    assert (first / relative).read_bytes() == b'first download'
    assert (second / relative).read_bytes() == b'second download'
    assert first != second
    assert not source.exists()


def test_old_quarantine_only_does_not_create_an_empty_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_routes, 'datetime', _FrozenClock)
    old = tmp_path/'quarantine-earlier'
    old.mkdir()
    (old/'field').write_bytes(b'prior')
    assert fetch_routes._quarantine(tmp_path, lambda _: None) is None
    assert list(tmp_path.iterdir()) == [old]


def test_preexisting_timestamp_file_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_routes, 'datetime', _FrozenClock)
    occupied = tmp_path/'quarantine-20260927T120000'
    occupied.write_bytes(b'prior receipt')
    (tmp_path/'current').write_bytes(b'current')
    saved = fetch_routes._quarantine(tmp_path, lambda _: None)
    assert saved != occupied
    assert occupied.read_bytes() == b'prior receipt'
    assert (saved/'current').read_bytes() == b'current'


def test_repeated_refetches_preserve_all_generations_and_report_moves(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_routes, 'datetime', _FrozenClock)
    saved = []
    messages = []
    for payload in (b'first', b'second', b'third'):
        (tmp_path / 'current').write_bytes(payload)
        aside = fetch_routes._quarantine(tmp_path, messages.append)
        saved.append((aside, payload))
        assert messages[-1] == f'fetch: --force-refetch moved 1 entry to {aside}'
        for prior, expected in saved:
            assert (prior / 'current').read_bytes() == expected
    assert len({aside for aside, _ in saved}) == 3


def test_empty_directory_does_not_create_a_generation(tmp_path):
    messages = []
    assert fetch_routes._quarantine(tmp_path, messages.append) is None
    assert list(tmp_path.iterdir()) == []
    assert messages == []
