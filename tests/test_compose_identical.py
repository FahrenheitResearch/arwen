"""A composed route's fetch leaves an identical composed file alone (fetch_routes._run_compose).

The defect: members of one ensemble share one download folder and each member's chain fetches it again; the in-place
rewrite of a composed GRIB moved the file an earlier member had hashed, and that member's deferred re-read refused
("authority changed while it was re-read", gem-gdps, 2026-10-25).
"""
from types import SimpleNamespace

from gpuwm import fetch_routes


def _grib(body: bytes) -> bytes:
    return b"GRIB" + body + b"7777"


def _plan(tmp_path):
    (tmp_path / "parts").mkdir()
    (tmp_path / "parts" / "a.grib2").write_bytes(_grib(b"a" * 40))
    (tmp_path / "parts" / "b.grib2").write_bytes(_grib(b"b" * 40))
    step = SimpleNamespace(kind="concat_per_lead", name="composed/x_PT012H.grib2", lead=12,
                           parts=[SimpleNamespace(relpath="parts/a.grib2"), SimpleNamespace(relpath="parts/b.grib2")])
    return SimpleNamespace(compose=(step,), source_id="gem-gdps")


def test_second_fetch_leaves_the_composed_file_untouched(tmp_path):
    plan = _plan(tmp_path)
    first = fetch_routes._run_compose(plan, tmp_path, progress=lambda *_: None)
    dest = tmp_path / "composed" / "x_PT012H.grib2"
    before = dest.stat()
    second = fetch_routes._run_compose(plan, tmp_path, progress=lambda *_: None)
    after = dest.stat()
    assert first == second
    assert (before.st_ino, before.st_mtime_ns, before.st_ctime_ns) == (after.st_ino, after.st_mtime_ns, after.st_ctime_ns)


def test_a_changed_part_rewrites_the_composed_file(tmp_path):
    plan = _plan(tmp_path)
    first = fetch_routes._run_compose(plan, tmp_path, progress=lambda *_: None)
    (tmp_path / "parts" / "b.grib2").write_bytes(_grib(b"c" * 40))
    second = fetch_routes._run_compose(plan, tmp_path, progress=lambda *_: None)
    assert first[0]["sha256"] != second[0]["sha256"]
    dest = tmp_path / "composed" / "x_PT012H.grib2"
    assert dest.read_bytes() == _grib(b"a" * 40) + _grib(b"c" * 40)
    assert not (tmp_path / "composed" / "x_PT012H.grib2.composing").exists()
