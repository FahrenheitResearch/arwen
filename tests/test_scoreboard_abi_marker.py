"""The scoreboard door's contract marker must be a substring of the scorer's own ``--abi`` line.

Breakage it prevents: the obs-match-v2 commit added ``interpolate_to_reports`` to the Rust ABI
constant while ``gpuwm.scoreboard.ABI_MARKER`` and ``bridges.BRIDGE_ABI_MARKERS`` kept the old
action list, so ``gpuwm scoreboard --request`` refused the very binary built beside it (exit 3).
"""
from __future__ import annotations

import re
from pathlib import Path

from gpuwm import bridges, scoreboard

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "tools" / "rustwx" / "crates" / "rw-scoreboard" / "src" / "main.rs"


def _rust_abi() -> str:
    match = re.search(r'pub const ABI: &str = "([^"]+)";', MAIN.read_text(encoding="utf-8"))
    assert match, f"{MAIN} no longer declares its ABI constant"
    return match.group(1)


def test_door_marker_is_in_the_scorer_abi_line():
    assert scoreboard.ABI_MARKER in _rust_abi()


def test_bundle_marker_matches_the_door_marker():
    assert bridges.BRIDGE_ABI_MARKERS["rw_scoreboard"] == scoreboard.ABI_MARKER.encode()


def test_the_door_marker_names_interpolate_to_reports():
    assert "interpolate_to_reports" in scoreboard.ABI_MARKER
