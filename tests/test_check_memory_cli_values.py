"""`gpuwm check` refuses a declared memory figure that is not a quantity.

--budget-gib, --reserve-gib and --vram-gib reach byte arithmetic
(``int(value * GIB)``): infinity left as an OverflowError traceback, NaN
as an unnamed "cannot convert float NaN to integer", and a negative or
zero card total or a negative reserve printed a verdict for a card that
cannot exist.  --rail-mib below one priced against a ceiling of nothing.
"""

from __future__ import annotations

import argparse

import pytest

from gpuwm.cli import main
from gpuwm.core import preflight as pf


def _args(tmp_path, *flags):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pf.register_cli(sub)
    return parser.parse_args(["check", str(tmp_path / "absent.toml"), *flags])


@pytest.mark.parametrize("option,values,partner", [
    ("--vram-gib", ["nan", "inf", "-inf", "-1", "0"], ["--free-gib", "10"]),
    ("--budget-gib", ["nan", "inf", "-inf", "-1"], ["--vram-gib", "12"]),
    ("--reserve-gib", ["nan", "inf", "-inf", "-1"], ["--free-gib", "10", "--vram-gib", "12"]),
    ("--rail-mib", ["0", "-1"], []),
])
def test_bad_declarations_name_the_option_before_the_config_is_read(
        tmp_path, option, values, partner):
    for value in values:
        args = _args(tmp_path, *partner, f"{option}={value}")
        with pytest.raises(ValueError, match=option):
            pf.check_main(args)


@pytest.fixture
def tiny_config(tmp_path, capsys):
    out = tmp_path / "tiny.toml"
    assert main(["domain", "--point", "40,-100", "--card", "12gb", "--source", "gfs",
                 "--cycle", "2026-09-27T00", "--hours", "1", "--point-extent-km", "100",
                 "--out", str(out)]) == 0
    capsys.readouterr()
    return out


def test_the_front_door_refuses_by_name_and_keeps_zero_reserve_and_budget(
        tiny_config, capsys, monkeypatch):
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    for flags, option in (
            (["--free-gib", "10", "--vram-gib=inf"], "--vram-gib"),
            (["--free-gib", "10", "--vram-gib=0"], "--vram-gib"),
            (["--budget-gib=inf", "--vram-gib", "12"], "--budget-gib"),
            (["--free-gib", "10", "--vram-gib", "12", "--reserve-gib=-inf"], "--reserve-gib")):
        assert main(["check", str(tiny_config), "--json", *flags]) == 2
        err = capsys.readouterr().err
        assert option in err and "Traceback" not in err and "OverflowError" not in err
    # A zero reserve is a real statement: the check prices and passes.
    assert main(["check", str(tiny_config), "--json", "--free-gib", "10",
                 "--vram-gib", "12", "--reserve-gib", "0"]) == 0
    capsys.readouterr()
    # A zero budget is a real statement too: the check prices and fails the fit.
    assert main(["check", str(tiny_config), "--json", "--budget-gib", "0",
                 "--vram-gib", "12"]) == 1
    capsys.readouterr()
