"""A wavelength the operator leaves untouched is a valid JSON response row.

``gpuwm spectral-op response`` writes its table with ``allow_nan=False``
and the table used to carry ``calls_to_e_fold = inf`` for every row with
unit gain, so any request whose sampled range reached an undamped
wavelength exited 2 with "Out of range float values are not JSON
compliant: inf" and wrote nothing: a 6 km reference over the default
sampled range, any protected scale, a zero damping ceiling, a zero step.
"""

from __future__ import annotations

import json
import math

import pytest

from gpuwm.cli import main
from gpuwm.spectral_ops.response import hyperdiffusion_response
from gpuwm.spectral_ops.transfer import Hyperdiffusion


def _strict(text: str):
    def refuse(name):
        raise AssertionError(f"{name} is not JSON")
    return json.loads(text, parse_constant=refuse)


@pytest.mark.parametrize("extra", [
    [],
    ["--protect-wavelength-m", "12000"],
    ["--maximum-damping-fraction", "0"],
    ["--dt-s", "0"],
])
def test_unit_gain_rows_are_null_in_stdout_and_file(tmp_path, capsys, extra):
    output = tmp_path / "response.json"
    code = main(["spectral-op", "response", "--reference-wavelength-m", "6000",
                 "--e-fold-time-s", "300", "--dt-s", "60",
                 "--output", str(output), *extra])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    document = _strict(captured.out)
    assert _strict(output.read_text(encoding="utf-8")) == document
    undamped = [row for row in document["rows"]
                if row["amplitude_gain_per_call"] == 1.0]
    assert undamped
    assert all(row["calls_to_e_fold"] is None for row in undamped)
    assert all(row["amplitude_damping_percent_per_call"] == 0.0 for row in undamped)
    damped = [row for row in document["rows"]
              if row["amplitude_gain_per_call"] < 1.0]
    assert all(math.isfinite(row["calls_to_e_fold"]) and row["calls_to_e_fold"] > 0
               for row in damped)


def test_the_documented_response_command_runs(capsys):
    """The line docs/public/LEVEL2_SPECTRAL_NUMERICS.md shows stays strict JSON."""
    code = main(["spectral-op", "response", "--reference-wavelength-m", "18000",
                 "--e-fold-time-s", "450", "--dt-s", "60"])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    rows = _strict(captured.out)["rows"]
    assert len(rows) == 64
    assert all(row["calls_to_e_fold"] is None
               or (math.isfinite(row["calls_to_e_fold"]) and row["calls_to_e_fold"] > 0)
               for row in rows)


def test_the_table_itself_carries_no_infinity():
    spec = Hyperdiffusion(order=3, reference_wavelength_m=6000.0, e_fold_time_s=300.0)
    rows = hyperdiffusion_response(spec, dt_s=60.0, wavelengths_m=[6000.0, 3.0e6])
    by_wavelength = {row["wavelength_m"]: row for row in rows}
    assert by_wavelength[3.0e6]["amplitude_gain_per_call"] == 1.0
    assert by_wavelength[3.0e6]["calls_to_e_fold"] is None
    assert by_wavelength[6000.0]["calls_to_e_fold"] == pytest.approx(5.0, rel=1e-12)


def test_the_help_says_what_null_means(capsys):
    with pytest.raises(SystemExit):
        main(["spectral-op", "response", "--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "calls_to_e_fold is null" in help_text


def test_a_protect_wavelength_below_every_candidate_is_a_sentence(tmp_path, capsys):
    """The calibrator's search skipped every reference and asserted."""
    bands = tmp_path / "bands.json"
    bands.write_text(json.dumps({"bands": [
        {"wavelength_m": 24000.0, "power_ratio": 1.1},
        {"wavelength_m": 6000.0, "power_ratio": 4.0}]}), encoding="utf-8")
    code = main(["spectral-op", "calibrate", "--input", str(bands),
                 "--output", str(tmp_path / "p.json"), "--dt-s", "60",
                 "--protect-wavelength-m", "1000"])
    err = capsys.readouterr().err
    assert code == 2
    assert "protect wavelength 1000 m leaves no reference wavelength" in err
    assert "Traceback" not in err and "AssertionError" not in err
    assert not (tmp_path / "p.json").exists()
