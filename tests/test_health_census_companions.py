"""Census the same validated companion-table config the forecast loads."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from gpuwm.experiment import load_experiment
from tools.health_field_census import load_any_experiment


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/hrrr_prs_3km_demo.toml"


def test_companion_config_uses_production_resolution():
    expected = load_experiment(CONFIG)
    measured = load_any_experiment(CONFIG)
    assert measured == expected


@pytest.mark.parametrize("table", ["fetch", "static"])
def test_invalid_companion_still_refused_by_its_owner(tmp_path, table):
    payload = CONFIG.read_text() + f"\n[{table}.unknown_census_key]\nvalue = 1\n"
    path = tmp_path / "invalid.toml"
    path.write_text(payload)
    with pytest.raises(ValueError) as expected:
        load_experiment(path)
    with pytest.raises(ValueError) as measured:
        load_any_experiment(path)
    assert str(measured.value) == str(expected.value)


def test_companion_config_measured_by_default_cli(tmp_path):
    # Keep the shipped physics and companion declarations, reducing only
    # the horizontal allocation for this descriptor-inventory check.
    payload = re.sub(r"(?m)^(nx|ny) = \d+$", r"\1 = 12", CONFIG.read_text())
    path = tmp_path / "companion.toml"
    path.write_text(payload)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", GPUWM_NO_LOCAL_GPU="1")
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/health_field_census.py"), str(path),
         "--around", "mp8-lsm2-pbl1-sfclay1-cu0-km4", "--json"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["rows"]
    assert receipt["worst_count"] > 0
    assert not receipt["over_ceiling"]
    assert receipt["domains"][0]["nx"] == 12
