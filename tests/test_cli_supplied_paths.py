"""A path the reader typed that is not there is one sentence, whichever door opened it.

Most doors open their input paths where the work needs them, not at the
parser, and a missing one left as a FileNotFoundError (or, for a folder
where a file belongs, IsADirectoryError / PermissionError) traceback:
`gpuwm spectral cross-box`, `gpuwm spectral check`, `gpuwm spectral-op
check` and `calibrate --input`, `gpuwm domain --vtable`, among others.
The refusal boundary now names the option and the path at exit 2 --
and only for a path the reader supplied: a file the program chose for
itself keeps its traceback, because that is a defect, not a typo.
"""

from __future__ import annotations

import json

import pytest

from gpuwm.cli import main


def _commands(tmp_path, target):
    return {
        "spectral cross-box": (["spectral", "cross-box", target, target], "RECEIPT.json"),
        "spectral check": (["spectral", "check", target], "RECEIPT.json"),
        "spectral-op check": (["spectral-op", "check", target], "RECEIPT"),
        "spectral-op calibrate": (["spectral-op", "calibrate", "--input", target,
                                   "--output", str(tmp_path / "p.json"), "--dt-s", "60"],
                                  "--input"),
    }


@pytest.mark.parametrize("door", ["spectral cross-box", "spectral check",
                                  "spectral-op check", "spectral-op calibrate"])
@pytest.mark.parametrize("kind", ["missing", "folder"])
def test_a_supplied_path_that_is_not_a_file_is_named(tmp_path, capsys, door, kind):
    target = tmp_path / "input.json"
    if kind == "folder":
        target.mkdir()
    argv, named = _commands(tmp_path, str(target))[door]
    assert main(argv) == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert named in err and str(target) in err
    assert ("is a folder" if kind == "folder" else "no such file or folder") in err


def test_a_relative_path_says_where_it_was_looked_for(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["spectral-op", "check", "absent.json"]) == 2
    err = capsys.readouterr().err
    assert "absent.json (the RECEIPT argument): no such file or folder" in err
    assert str(tmp_path) in err


def test_a_path_the_program_chose_keeps_its_traceback(tmp_path, monkeypatch):
    """The negative control: an internal missing file is a defect, not a typo."""
    from gpuwm.spectral_ops import cli as spectral_op_cli

    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({}), encoding="utf-8")

    def broken(_args):
        raise FileNotFoundError(2, "No such file or directory",
                                str(tmp_path / "internal" / "state.npz"))

    monkeypatch.setattr(spectral_op_cli, "_check", broken)
    with pytest.raises(FileNotFoundError):
        main(["spectral-op", "check", str(receipt)])


def test_the_refusal_matches_an_extended_length_spelling_and_names_the_typed_option(tmp_path):
    """deep_io_path hands Windows an extended-length spelling of a long path;
    it is still the path the reader typed."""
    import argparse

    from gpuwm.cli_paths import supplied_path_refusal

    # The four characters backslash, backslash, question mark, backslash.
    EXTENDED = chr(92) * 2 + "?" + chr(92)
    typed = str(tmp_path / "forcing.grib")
    parser = argparse.ArgumentParser(prog="gpuwm")
    sub = parser.add_subparsers(dest="command")
    door = sub.add_parser("door")
    door.add_argument("--forcing", nargs="+")
    args = parser.parse_args(["door", "--forcing", typed])
    error = FileNotFoundError(2, "No such file or directory", EXTENDED + typed)
    refusal = supplied_path_refusal(error, args, parser=parser,
                                    tokens=["door", "--forcing", typed])
    assert refusal == f"--forcing {typed}: no such file or folder"
    internal = FileNotFoundError(2, "No such file or directory", str(tmp_path / "other"))
    assert supplied_path_refusal(internal, args, parser=parser, tokens=[]) is None
    assert supplied_path_refusal(ValueError("x"), args, parser=parser) is None


def test_a_path_whose_folder_is_missing_names_the_folder(tmp_path):
    import argparse

    from gpuwm.cli_paths import supplied_path_refusal

    typed = str(tmp_path / "no-such-folder" / "out.json")
    parser = argparse.ArgumentParser(prog="gpuwm")
    parser.add_argument("--output")
    args = parser.parse_args(["--output", typed])
    error = FileNotFoundError(2, "No such file or directory", typed)
    refusal = supplied_path_refusal(error, args, parser=parser, tokens=["--output", typed])
    assert refusal == (f"--output {typed}: the folder "
                       f"{tmp_path / 'no-such-folder'} does not exist")
