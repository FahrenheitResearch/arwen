"""Radiation-port metadata must survive regeneration from the release registry.

A rebuild from the published 2.8.9 registry document (cd336e62d6) must still
carry every aer_opt = 2 control and publish aer_opt 1 and 2 as implemented
values: the aer_opt = 2 lane once hand-edited the generated document, and a
rebuild silently dropped its namelist rows (port289-final 9c72179fd).
"""
import json
import subprocess
from pathlib import Path

import pytest

RELEASE_E = "cd336e62d6e6b6312c9636e0fdfff84801909b6a"


def _release_registry(root):
    try:
        text = subprocess.check_output(
            ["git", "-C", str(root), "show",
             f"{RELEASE_E}:gpuwm/physics_registry_v2.json"],
            stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("the 2.8.9 release commit is not in this checkout")
    return json.loads(text)


def test_aerosol_controls_are_owned_when_rebuilding_from_release_E(monkeypatch):
    from gpuwm.physics_registry import REGISTRY_REBUILD_ENV
    from gpuwm.core.aer_opt2 import OPTION_KEYS, VALUE_KEYS
    from tools.build_registry import build
    root = Path(__file__).resolve().parents[1]
    release = _release_registry(root)
    monkeypatch.setenv(REGISTRY_REBUILD_ENV, "1")
    rebuilt = build(release)
    names = {*OPTION_KEYS, *VALUE_KEYS, "aer_type", "aer_auxinput15",
             "aer_auxinput15_interval"}
    assert names <= rebuilt["parameters"].keys()
    assert rebuilt["parameters"]["aer_type"]["enum"] == [1, 2, 3]
    assert rebuilt["parameters"]["aer_opt"]["enum"] == [0, 1, 2, 3]
