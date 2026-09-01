"""``gpuwm spectral-op``: the Level-2 operator front door, CPU-hermetic.

The delivered package's own container never collected this suite (its
combined patch was empty and ``gpuwm.cli`` was not importable there), so
these are the first real runs of the front door.  Everything here goes
through ``gpuwm.cli.main`` -- the argv a user types -- and none of it may
touch CuPy: the pins/check/calibrate/response legs are required to stay
reachable on a CPU-only install.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from gpuwm.cli import main
from gpuwm.spectral_ops import PINS_SHA256
from gpuwm.spectral_ops.pins import canonical_hash

FIXTURES = Path(__file__).parent / "data"


def run_cli(capsys, *argv):
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out


def test_pins_prints_the_registration_and_exits_zero(capsys):
    code, out = run_cli(capsys, "spectral-op", "pins")
    assert code == 0
    payload = json.loads(out)
    assert payload["schema"] == "gpuwm.regional-spectral-operators/v1"
    assert payload["pins_sha256"] == PINS_SHA256
    assert canonical_hash(payload["pins"]) == PINS_SHA256


def test_check_validates_the_delivered_construction_receipt(capsys):
    """The construction checkout's own shadow-step receipt validates.

    This is the cross-implementation anchor: the fixture was written by
    the package author's build, and our vendored code must recompute the
    identical canonical hash chain or the receipts diverge silently.
    """
    receipt = FIXTURES / "spectral_shadow_step_receipt.json"
    expected = json.loads(receipt.read_text(encoding="utf-8"))
    code, out = run_cli(capsys, "spectral-op", "check", str(receipt))
    assert code == 0
    assert out.strip() == expected["receipt_sha256"]


def test_check_refuses_a_tampered_receipt(tmp_path, capsys):
    body = json.loads(
        (FIXTURES / "spectral_shadow_step_receipt.json").read_text(
            encoding="utf-8"))
    body["applied"] = True
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(body), encoding="utf-8")
    # The CLI's refusal convention: one sentence on stderr, exit 2.
    code = main(["spectral-op", "check", str(tampered)])
    captured = capsys.readouterr()
    assert code == 2
    assert "hash" in captured.err


def test_response_prints_the_exact_wavelength_response(capsys):
    code, out = run_cli(
        capsys, "spectral-op", "response",
        "--reference-wavelength-m", "6000", "--e-fold-time-s", "300",
        "--dt-s", "300", "--wavelength-m", "6000", "--wavelength-m", "60000")
    assert code == 0
    payload = json.loads(out)
    rows = {row["wavelength_m"]: row for row in payload["rows"]}
    assert rows[6000.0]["amplitude_gain_per_call"] == pytest.approx(
        math.exp(-1.0), rel=1e-9)
    assert rows[6000.0]["calls_to_e_fold"] == pytest.approx(1.0, rel=1e-9)
    assert (rows[60000.0]["amplitude_gain_per_call"]
            > rows[6000.0]["amplitude_gain_per_call"])


def test_calibrate_writes_a_damping_only_proposal(tmp_path, capsys):
    bands = tmp_path / "bands.json"
    bands.write_text(json.dumps({"bands": [
        {"wavelength_m": 24000.0, "power_ratio": 1.1},
        {"wavelength_m": 12000.0, "power_ratio": 2.0},
        {"wavelength_m": 6000.0, "power_ratio": 4.0},
    ]}), encoding="utf-8")
    output = tmp_path / "proposal.json"
    code, out = run_cli(
        capsys, "spectral-op", "calibrate", "--input", str(bands),
        "--output", str(output), "--dt-s", "60")
    assert code == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written == json.loads(out)
    assert written["status"] == "proposal-only"
    assert all(value <= 1.0 + 1e-12
               for value in written["predicted_amplitude_transfer"])
    body = dict(written)
    assert canonical_hash(
        {k: v for k, v in body.items() if k != "recommendation_sha256"}
    ) == body["recommendation_sha256"]


def test_benchmark_runs_the_cpu_controls(tmp_path, capsys):
    output = tmp_path / "bench.json"
    code, out = run_cli(
        capsys, "spectral-op", "benchmark", "--backend", "numpy",
        "--nx", "48", "--ny", "32", "--levels", "2", "--repeats", "1",
        "--output", str(output))
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == json.loads(out)
    assert payload["backend"] == "numpy"
    assert payload["controls"]["divergence_ratio"] < 1.0


def test_the_door_never_imports_cupy(capsys):
    """CPU reachability is a contract, not a coincidence.

    A CuPy import inside the pins/check legs would make the evidence door
    unreachable exactly where the evidence is read (CI boxes, laptops).
    """
    import sys
    before = sys.modules.get("cupy")
    run_cli(capsys, "spectral-op", "pins")
    run_cli(capsys, "spectral-op", "check",
            str(FIXTURES / "spectral_shadow_step_receipt.json"))
    assert sys.modules.get("cupy") is before
