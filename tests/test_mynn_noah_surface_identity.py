"""The admitted MYNN/MYNN/Noah surface trajectory stays byte-identical."""

from __future__ import annotations

import json
from pathlib import Path

from conftest import requires_gpu


@requires_gpu
def test_mynn_noah_surface_trajectory_matches_the_pre_pairing_baseline():
    from tools.mynn_noah_surface_identity import run_identity

    fixture = Path(__file__).with_name("fixtures") / \
        "mynn_noah_surface_identity_bf45e88a.json"
    expected = json.loads(fixture.read_text(encoding="utf-8"))
    assert run_identity() == expected
