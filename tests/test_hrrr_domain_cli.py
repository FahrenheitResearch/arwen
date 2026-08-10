from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from gpuwm.source_cli import EXIT_CONFIG, main


ROOT = Path(__file__).parents[1]
OHIO = ROOT / "configs" / "hrrr_target_ohio_192x160_3km.json"
SCHEMA = "gpuwm-hrrr-domain-validation-v1"
FAILED_DOMAIN_SHA256 = (
    "24cde8e888be9c404b3a2afd56c8af6263ae23470825545aa12a245529a04835"
)
FAILED_COVERAGE_ERROR = (
    "target domain plus required interpolation halo leaves HRRR coverage: "
    "required zero-based inclusive window i=642..1387, j=-10..504; native "
    "limits are i=0..1798, j=0..1058"
)


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _write_shifted_failed_domain(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "gpuwm-hrrr-target-domain-v1",
                "name": "hrrr_custom_39p96_m83p10_1000m",
                "map_proj": "lambert",
                "nx": 1991,
                "ny": 1161,
                "nz": 49,
                "dx_m": 1000.0,
                "dy_m": 1000.0,
                "ref_lat": 30.87876088232676,
                "ref_lon": -93.82975840568542,
                "truelat1": 30.0,
                "truelat2": 60.0,
                "stand_lon": -84.0,
                "time_step_seconds": 5,
                "spec_bdy_width": 5,
                "spec_zone": 1,
                "relax_zone": 4,
                "surface_fallback_radius_cells": 10,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_validate_hrrr_domain_accepts_packaged_ohio_target(capsys):
    assert main(["--validate-hrrr-domain", str(OHIO)]) == 0
    rendered = capsys.readouterr().out
    receipt = json.loads(rendered)

    assert set(receipt) == {
        "schema",
        "status",
        "domain_sha256",
        "window",
        "error",
    }
    assert {key: receipt[key] for key in receipt if key != "window"} == {
        "schema": SCHEMA,
        "status": "PASS",
        "domain_sha256": (
            "42a685bd8031ad492f8426c6e0b5ec070717256502536420762fa613cf282abb"
        ),
        "error": None,
    }
    window = receipt["window"]
    assert window["zero_based_inclusive"] == {
        "i": [1190, 1428],
        "j": [512, 722],
    }
    assert window["shape"] == [211, 239]
    assert window["parabolic_halo_cells"] == {
        "below_floor": 1,
        "above_floor": 2,
    }
    assert window["surface_fallback_radius_cells"] == 10
    assert window["target_source_i_range"] == pytest.approx(
        [1199.0772575200124, 1418.6930088547429]
    )
    assert window["target_source_j_range"] == pytest.approx(
        [521.2074577905555, 712.9177907585818]
    )
    assert rendered == _canonical(receipt) + "\n"


def test_validate_hrrr_domain_refuses_exact_shifted_studio_target(
    tmp_path: Path,
    capsys,
):
    target = tmp_path / "shifted-domain.json"
    _write_shifted_failed_domain(target)

    assert main(["--validate-hrrr-domain", str(target)]) == EXIT_CONFIG
    rendered = capsys.readouterr().out
    receipt = json.loads(rendered)

    assert receipt == {
        "schema": SCHEMA,
        "status": "REFUSED",
        "domain_sha256": FAILED_DOMAIN_SHA256,
        "window": None,
        "error": FAILED_COVERAGE_ERROR,
    }
    assert rendered == _canonical(receipt) + "\n"


def test_validate_hrrr_domain_strict_load_refusal_has_no_domain_hash(
    tmp_path: Path,
    capsys,
):
    target = tmp_path / "wrong-schema.json"
    target.write_text('{"schema":"not-a-target"}', encoding="utf-8")

    assert main(["--validate-hrrr-domain", str(target)]) == EXIT_CONFIG
    receipt = json.loads(capsys.readouterr().out)
    assert receipt == {
        "schema": SCHEMA,
        "status": "REFUSED",
        "domain_sha256": None,
        "window": None,
        "error": "unsupported HRRR target-domain schema",
    }


def test_validate_hrrr_domain_is_an_exclusive_inventory_action(capsys):
    with pytest.raises(SystemExit) as inventory_conflict:
        main(["--validate-hrrr-domain", str(OHIO), "--list-sources"])
    assert inventory_conflict.value.code == 2
    assert "choose exactly one inventory option" in capsys.readouterr().err

    with pytest.raises(SystemExit) as unrelated:
        main(["--validate-hrrr-domain", str(OHIO), "--source", "hrrr"])
    assert unrelated.value.code == 2
    assert (
        "--validate-hrrr-domain cannot be combined with other action arguments: "
        "--source"
    ) in capsys.readouterr().err


def test_validate_hrrr_domain_subprocess_exit_and_lf_are_stable():
    # The provenance banner (one line on stderr at every front door) is
    # silenced through its own documented switch: this test pins the
    # RECEIPT's byte stability, and "stderr carries nothing but what
    # this door was asked for" is only true of a caller who asked for
    # quiet.  On a dev worktree the banner would otherwise be here by
    # design.
    environment = dict(os.environ)
    environment["GPUWM_PROVENANCE_BANNER"] = "0"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "gpuwm.source_cli",
            "--validate-hrrr-domain",
            str(OHIO),
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr.decode()
    assert completed.stderr == b""
    assert completed.stdout.endswith(b"\n")
    assert not completed.stdout.endswith(b"\r\n")
    receipt = json.loads(completed.stdout)
    assert receipt["schema"] == SCHEMA
    assert receipt["status"] == "PASS"
    assert completed.stdout == (_canonical(receipt) + "\n").encode("utf-8")
