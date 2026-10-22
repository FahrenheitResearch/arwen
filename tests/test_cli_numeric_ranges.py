"""Numeric options refuse values outside their meaning at the parser, by name.

``type=float`` reads ``nan``, ``inf`` and ``-inf`` and ``type=int`` reads
any sign, so these options used to carry a value their handler could not
use into the handler, where it ended as an OverflowError traceback
(``int(inf)``, ``Fraction(inf)``, ``socket.bind`` on port 65536), a
sentence that named no option ("cannot convert NaN to integer ratio"),
an AssertionError, or no error at all (a NaN seed radius, a negative
member count, a zero PyPI timeout).  Each row below is one option, the
values it must refuse, and one value it must keep taking.
"""

from __future__ import annotations

import argparse

import pytest

from gpuwm import cli_numbers
from gpuwm.cli import build_parser

NONFINITE = ["nan", "inf", "-inf"]

#: (argv before the option, option, refused values, an accepted value)




@pytest.mark.parametrize('argv,option,bad,good', [
    pytest.param(["obs", "radar", "sites"], "--bbox", ["nan,40,10,50", "0,40,inf,50", "-400,40,10,50", "0,40,10,95", "10,40,5,50"], "0,40,10,50", id='obs radar --bbox'),
    pytest.param(["render", "wrfout"], "--section-across", NONFINITE + ["0", "-1", "1.5"], "2", id='render wrfout --section-across'),
    pytest.param(["downscale", "parent", "--out", "o"], "--child-size", ["0", "-1", "10,-2", "nan", "1,2,3", ""], "90,60", id='downscale parent --child-size'),
])
def test_out_of_range_values_are_refused_by_option_name(capsys, argv, option, bad, good):
    parser = build_parser()
    for value in bad:
        with pytest.raises(SystemExit) as stopped:
            # The =VALUE spelling, so a leading minus is never read as an option.
            parser.parse_args([*argv, f"{option}={value}"])
        assert stopped.value.code == 2, (option, value)
        err = capsys.readouterr().err
        assert f"argument {option}" in err, (option, value, err)
        assert "Traceback" not in err
    parsed = parser.parse_args([*argv, f"{option}={good}"])
    assert parsed is not None


@pytest.mark.parametrize("parse,accepted,refused", [
    (cli_numbers.finite_float, ["0", "-2.5", "1e300"], ["nan", "inf", "-inf", "x", ""]),
    (cli_numbers.positive_float, ["1e-9", "3"], ["0", "-1", "nan", "inf", "x"]),
    (cli_numbers.nonnegative_float, ["0", "2"], ["-1e-9", "nan", "-inf", "x"]),
    (cli_numbers.positive_int, ["1", "400"], ["0", "-3", "1.5", "nan", "inf"]),
    (cli_numbers.nonnegative_int, ["0", "7"], ["-1", "0.5", "inf"]),
    (cli_numbers.int_at_least(4), ["4", "90"], ["3", "-4", "nan"]),
    (cli_numbers.int_between(0, 65535), ["0", "65535"], ["-1", "65536", "80.0"]),
    (cli_numbers.float_between(-90.0, 90.0), ["-90", "0", "90"], ["-90.1", "nan", "inf"]),
])
def test_each_range_type_takes_its_range_and_names_what_it_needs(parse, accepted, refused):
    for text in accepted:
        parse(text)
    for text in refused:
        with pytest.raises(argparse.ArgumentTypeError, match=r"^must be .*, not "):
            parse(text)


@pytest.mark.parametrize('value', [
    pytest.param("nan", id='nan'),
    pytest.param("-1", id='-1'),
])
def test_fetch_top_pressure_and_wait_timeout_refuse_non_numbers_before_the_network(
        tmp_path, capsys, value):
    """Checked in the handler, whose `<= 0` test let NaN and infinity through."""
    from gpuwm.cli import main

    assert main(["fetch", "--source", "gfs", "--cycle", "2026-07-28T06", "--hours", "6",
                 "--area", "30,-100,40,-90", "--out", str(tmp_path / "g"),
                 f"--p-top-pa={value}"]) == 2
    assert "--p-top-pa must be a positive, finite pressure" in capsys.readouterr().err
    assert main(["fetch", "--source", "hrrr", "--cycle", "2026-07-28T05", "--hours", "2",
                 "--out", str(tmp_path / "h"), "--wait-for",
                 f"--wait-timeout-minutes={value}"]) == 2
    assert "--wait-timeout-minutes must be positive and finite" in capsys.readouterr().err
