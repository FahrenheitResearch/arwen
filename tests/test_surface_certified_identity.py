"""Certified Noah profiles must remain byte-identical to v1.1.2."""

from __future__ import annotations

import json
from pathlib import Path

from conftest import requires_gpu


@requires_gpu
def test_four_certified_noah_profiles_match_the_v112_trajectory_bytes():
    from tools.certified_surface_identity import run_profiles

    fixture = Path(__file__).with_name("fixtures") / \
        "certified_surface_identity_v112.json"
    expected = json.loads(fixture.read_text(encoding="utf-8"))
    assert run_profiles() == expected
