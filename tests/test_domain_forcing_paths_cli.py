"""`gpuwm domain --forcing` names an unusable input path before native inventory.

A missing file left as a FileNotFoundError traceback from the inventory's
stat(), and a folder was opened as a zero-length GRIB and refused as
"empty", which describes a file the reader never gave.
"""

from __future__ import annotations

import pytest

from gpuwm.cli import main
from gpuwm.ingest import grib


@pytest.mark.parametrize("kind", ["missing", "folder", "folder-by-glob"])
def test_supplied_forcing_must_be_files(tmp_path, capsys, monkeypatch, kind):
    def never(*_args, **_kwargs):
        pytest.fail("the native inventory was reached with an unusable path")

    monkeypatch.setattr(grib, "build_rust_bridge", never)
    monkeypatch.setattr(grib, "inspect_grib1_envelopes", never)
    supplied = tmp_path / "input.grib"
    spelled = str(supplied)
    if kind != "missing":
        supplied.mkdir()
    if kind == "folder-by-glob":
        spelled = str(tmp_path / "input.gri?")
    out = tmp_path / "case.toml"
    code = main(["domain", "--source", "era5", "--point", "40,-100", "--card", "12gb",
                 "--cycle", "2026-09-27T00", "--hours", "1", "--out", str(out),
                 "--forcing", spelled])
    err = capsys.readouterr().err
    assert code == 2
    assert "--forcing" in err and str(supplied) in err
    assert ("is a folder" if kind != "missing" else "does not exist") in err
    assert "Traceback" not in err
    assert not out.exists()
