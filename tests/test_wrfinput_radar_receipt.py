"""The radar-tendency receipt stays strict JSON when a slot covers no point.

Defect: a heating slot with no covered point (the satellite-rain release slot)
reported its maximum as -inf, and the receipt's strict JSON (allow_nan=False)
refused it, stopping the run (no-radar box test, 2026-10-25).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gpuwm.ingest.wrfinput_radar import RadarTendency


def test_receipt_with_an_uncovered_slot_serialises_as_strict_json():
    nz, ny, nx = 4, 3, 5
    covered = np.full((nz, ny, nx), -20.0, np.float32)
    covered[1, 1, 2] = 0.004
    covered[2, 0, 0] = -0.002
    uncovered = np.full((nz, ny, nx), -20.0, np.float32)  # outside [-1, 1] everywhere: no coverage
    tendency = RadarTendency(path=Path("wrfinput_d01"), slots=(covered, uncovered), slot_minutes=(15.0, 30.0))

    receipt = tendency.receipt()
    text = json.dumps(receipt, allow_nan=False)

    back = json.loads(text)
    assert back["covered_points_by_slot"] == [2, 0]
    assert back["max_heating_k_per_s_by_slot"][0] == float(np.float32(0.004))
    assert back["max_heating_k_per_s_by_slot"][1] is None
