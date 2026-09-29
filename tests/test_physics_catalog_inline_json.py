"""The physics check reads a long inline request as JSON, not as a file name.

The defect: ``gpuwm physics-catalog --check TEXT`` asked the file system
whether TEXT was a file before it parsed TEXT as JSON.  The GUI sends its
request inline, and a request that names several families is longer than a
file name may be, so on Linux the check failed with "OSError: [Errno 36]
File name too long" and Next stayed disabled on the Physics step.
"""

from __future__ import annotations

from argparse import Namespace
import errno
import io
import json
import pathlib

import pytest

from gpuwm.physics_catalog import main

#: The Physics step's request for cumulus off, RUC land, WSM6, YSU, RRTM plus Dudhia, revised MM5 and 2-D
#: Smagorinsky: longer than the 255 bytes a file name may take.
REQUEST = {
    "choices": {"cumulus": "off", "land_surface": "ruc-lsm", "microphysics": "wsm6-mp6",
                "pbl": "ysu", "radiation": "wrf-rrtm-dudhia", "surface_layer": "revised-mm5",
                "turbulence": "smagorinsky-2d"},
    "cycle": "2026-09-27T06", "dx_km": 3, "hours": 6, "lat": 39, "lon": -98, "source": "gfs",
}


def _run(argument: str, capsys) -> tuple[int, dict]:
    code = main(Namespace(check=argument, json=True, into=None, out=None, emit=False, preset=None, source=None))
    return code, json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("transport", ["inline", "file", "stdin"])
def test_a_long_request_gets_the_same_verdict_inline_from_a_file_and_on_stdin(transport, tmp_path, monkeypatch,
                                                                                 capsys):
    text = json.dumps(REQUEST, separators=(",", ":"), sort_keys=True)
    assert len(text.encode()) > 255
    if transport == "file":
        path = tmp_path / "physics.json"
        path.write_text(text, encoding="utf-8")
        argument = str(path)
    elif transport == "stdin":
        monkeypatch.setattr("sys.stdin", io.StringIO(text))
        argument = "-"
    else:
        argument = text
    code, reply = _run(argument, capsys)
    assert code == 0, reply
    assert "error" not in reply, reply
    assert reply["valid"] is True, reply
    assert reply["words"] == "Runs. No named set matches these choices."


def test_an_inline_request_is_never_asked_of_the_file_system(monkeypatch, capsys):
    """Python 3.11 to 3.13 on Linux raise ENAMETOOLONG from Path.is_file for a long name; every platform here
    answers the way Linux did, so the check must not ask."""

    real = pathlib.Path.is_file

    def is_file(self, *args, **kwargs):
        if str(self).startswith("{"):
            raise OSError(errno.ENAMETOOLONG, "File name too long", str(self))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "is_file", is_file)
    code, reply = _run(json.dumps(REQUEST, separators=(",", ":"), sort_keys=True), capsys)
    assert code == 0, reply
    assert reply["valid"] is True, reply


def test_text_that_is_neither_json_nor_a_file_answers_with_the_error_document(tmp_path, capsys):
    code, reply = _run(str(tmp_path / "missing.json"), capsys)
    assert code == 2
    assert "names no file" in reply["error"], reply


def test_a_request_file_that_cannot_be_read_answers_with_the_error_document(tmp_path, monkeypatch, capsys):
    path = tmp_path / "physics.json"
    path.write_text(json.dumps(REQUEST), encoding="utf-8")
    real = pathlib.Path.read_text

    def read_text(self, *args, **kwargs):
        if self == path:
            raise PermissionError(errno.EACCES, "Permission denied", str(self))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", read_text)
    code, reply = _run(str(path), capsys)
    assert code == 2
    assert "Permission denied" in reply["error"], reply
