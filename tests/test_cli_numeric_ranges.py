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
CASES = [
    (["spectral-op", "response", "--reference-wavelength-m", "6000",
      "--e-fold-time-s", "300", "--dt-s", "60"], "--samples",
     ["0", "-1"], "8"),
    (["spectral-op", "response", "--e-fold-time-s", "300", "--dt-s", "60"],
     "--reference-wavelength-m", NONFINITE + ["0", "-1"], "6000"),
    (["spectral-op", "response", "--reference-wavelength-m", "6000", "--dt-s", "60"],
     "--e-fold-time-s", NONFINITE + ["0", "-1"], "300"),
    (["spectral-op", "response", "--reference-wavelength-m", "6000",
      "--e-fold-time-s", "300"], "--dt-s", NONFINITE + ["-1"], "0"),
    (["spectral-op", "response", "--reference-wavelength-m", "6000",
      "--e-fold-time-s", "300", "--dt-s", "60"], "--wavelength-m",
     NONFINITE + ["0", "-1"], "6000"),
    (["spectral-op", "response", "--reference-wavelength-m", "6000",
      "--e-fold-time-s", "300", "--dt-s", "60"], "--minimum-wavelength-m",
     NONFINITE + ["0", "-1"], "3000"),
    (["spectral-op", "response", "--reference-wavelength-m", "6000",
      "--e-fold-time-s", "300", "--dt-s", "60"], "--maximum-wavelength-m",
     NONFINITE + ["0", "-1"], "3000000"),
    (["spectral-op", "response", "--reference-wavelength-m", "6000",
      "--e-fold-time-s", "300", "--dt-s", "60"], "--maximum-damping-fraction",
     NONFINITE + ["-0.5", "1.5"], "0"),
    (["spectral-op", "response", "--reference-wavelength-m", "6000",
      "--e-fold-time-s", "300", "--dt-s", "60"], "--order", ["0", "9"], "3"),
    (["spectral-op", "benchmark"], "--dx-m", NONFINITE + ["0", "-1"], "3000"),
    (["spectral-op", "benchmark"], "--dy-m", NONFINITE + ["0", "-1"], "3000"),
    (["spectral-op", "benchmark"], "--nx", ["0", "-1"], "16"),
    (["spectral-op", "benchmark"], "--repeats", ["0", "-1"], "1"),
    (["spectral-op", "calibrate", "--input", "b.json", "--output", "o.json"],
     "--dt-s", NONFINITE + ["0", "-1"], "60"),
    (["spectral-op", "calibrate", "--input", "b.json", "--output", "o.json",
      "--dt-s", "60"], "--protect-wavelength-m", NONFINITE + ["0", "-1"], "12000"),
    (["domain", "--point", "40,-100", "--card", "12gb", "--cycle", "2026-09-27T00", "--out", "x.toml"],
     "--history-interval", NONFINITE + ["0", "-1"], "1800"),
    (["domain", "--point", "40,-100", "--card", "12gb", "--cycle", "2026-09-27T00", "--out", "x.toml"],
     "--nest-history-interval", NONFINITE + ["0", "-1"], "600"),
    (["domain", "--point", "40,-100", "--card", "12gb", "--cycle", "2026-09-27T00", "--out", "x.toml"],
     "--cadence", ["0", "-1"], "3"),
    (["cyclone-setup", "--point", "25,-80", "--card", "12gb"],
     "--history-interval", NONFINITE + ["0", "-1"], "3600"),
    (["cyclone-setup", "--point", "25,-80", "--card", "12gb"],
     "--nest-history-interval", NONFINITE + ["0", "-1"], "900"),
    (["cyclone-setup", "--point", "25,-80", "--card", "12gb"],
     "--seed-radius-km", NONFINITE + ["0", "-1"], "500"),
    (["mesh", "--background-km", "120", "--card", "rtx-5070-ti"], "--vram-gib",
     NONFINITE + ["0", "-1"], "12"),
    (["mesh", "--background-km", "120"], "--cells", ["0", "-1"], "100"),
    (["mesh", "--background-km", "120"], "--sweeps", ["0", "-1"], "300"),
    (["mesh", "--background-km", "120"], "--tolerance", NONFINITE + ["0", "-1"], "0.001"),
    (["mesh", "--background-km", "120"], "--nominal-dx-m", NONFINITE + ["0", "-1"], "120000"),
    (["version", "--check-pypi"], "--pypi-timeout", NONFINITE + ["0", "-1"], "2"),
    (["obs", "radar", "grid", "--pack", "p", "--grid-wrfout", "w", "--out", "o",
      "--max-elevation-deg", "20"], "--max-range-km", NONFINITE + ["0", "-1"], "200"),
    (["obs", "radar", "grid", "--pack", "p", "--grid-wrfout", "w", "--out", "o",
      "--max-range-km", "200"], "--max-elevation-deg", NONFINITE + ["91", "-91"], "20"),
    (["obs", "radar", "pack", "--file", "f", "--out", "o"], "--max-range-km",
     NONFINITE + ["0", "-1"], "200"),
    (["obs", "radar", "sites"], "--bbox",
     ["nan,40,10,50", "0,40,inf,50", "-400,40,10,50", "0,40,10,95", "10,40,5,50"],
     "0,40,10,50"),
    (["gui"], "--port", ["-1", "65536"], "0"),
    (["enprod", "--make-fixture", "ens"], "--members", ["0", "-1"], "1"),
    (["enprod", "ens"], "--dpi", ["0", "-1"], "150"),
    (["render", "wrfout"], "--dpi", ["0", "-1"], "150"),
    (["render", "wrfout"], "--section-across", NONFINITE + ["0", "-1", "1.5"], "2"),
    (["run", "config.toml"], "--prep-timeout", NONFINITE + ["0", "-1"], "600"),
    (["run", "config.toml"], "--supervisor-max-restarts", ["-1"], "0"),
    (["resume", "config.toml"], "--supervisor-max-restarts", ["-1"], "0"),
    (["branch", "config.toml", "--outdir", "o"], "--prep-timeout",
     NONFINITE + ["0"], "600"),
    (["prep", "--namelist-support-report"], "--source-top-pressure-pa",
     NONFINITE + ["0", "-1"], "5000"),
    (["downscale", "parent", "--out", "o"], "--hours", NONFINITE + ["0", "-1"], "0.25"),
    (["downscale", "parent", "--out", "o"], "--output-interval-seconds", NONFINITE + ["0", "-1"], "900"),
    (["downscale", "parent", "--out", "o"], "--max-boundary-interval-seconds",
     NONFINITE + ["0", "-1"], "3600"),
    (["downscale", "parent", "--out", "o"], "--health-interval-seconds", NONFINITE + ["0", "-1"], "60"),
    (["downscale", "parent", "--out", "o"], "--parent-domain", ["0", "-1"], "2"),
    (["downscale", "parent", "--out", "o"], "--parent-namelist-domain", ["0", "-1"], "1"),
    (["downscale", "parent", "--out", "o"], "--i-parent-start", ["0", "-1"], "10"),
    (["downscale", "parent", "--out", "o"], "--j-parent-start", ["0", "-1"], "10"),
    (["downscale", "parent", "--out", "o"], "--child-size", ["0", "-1", "10,-2", "nan", "1,2,3", ""], "90,60"),
    (["downscale-parent", "run"], "--parent-domain", ["0", "-1"], "1"),
    (["warm-kernels"], "--levels", ["-1", "0", "3"], "4"),
    (["spectral", "cross-box", "a.json", "b.json"], "--tolerance", NONFINITE + ["-1"], "0"),
]


def _ids():
    return [f"{' '.join(argv[:2])} {option}" for argv, option, _bad, _good in CASES]


@pytest.mark.parametrize("argv,option,bad,good", CASES, ids=_ids())
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


@pytest.mark.parametrize("value", NONFINITE + ["0", "-1"])
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
