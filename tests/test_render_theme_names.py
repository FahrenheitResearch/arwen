"""The theme names gpuwm.rustwx.RENDER_THEMES promises are the renderer's own.

A caller (the WOOF worker) reads RENDER_THEMES to decide whether it may name
``woof-light-tc`` for a tropical cyclone run.  THE BREAKAGE THIS PREVENTS: a
name listed in Python but missing from the compiled renderer is a THEME_ERROR
at render time, and every map of that run is drawn in the plain default look.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from gpuwm import rustwx

ROOT = Path(__file__).resolve().parents[1]
THEME_RS = ROOT / "tools/rustwx/crates/rustwx-render/src/theme.rs"
TC_JSON = ROOT / "tools/rustwx/crates/rustwx-render/themes/tc.json"


def test_render_themes_match_the_renderer_built_ins():
    if not THEME_RS.is_file():
        pytest.skip("no vendored renderer source in this tree")
    text = THEME_RS.read_text(encoding="utf-8")
    body = re.search(r"fn builtin_names\(\)[^{]*\{\s*&\[(.*?)\]", text, re.S)
    assert body, "builtin_names() not found in theme.rs"
    names = tuple(re.findall(r'"([^"]+)"', body.group(1)))
    assert names == rustwx.RENDER_THEMES


def test_the_tc_sheet_reaches_a_hurricane_core():
    if not TC_JSON.is_file():
        pytest.skip("no vendored renderer source in this tree")
    tc = json.loads(TC_JSON.read_text(encoding="utf-8"))
    edges = tc["colormaps"]["levels"]["mslp_10m_winds_wind_speed"]
    assert edges[-1] >= 180
    assert {34, 50, 64, 83, 96, 113, 137} <= set(edges)
    assert len(tc["colormaps"]["products"]["mslp_10m_winds_wind_speed"]) == len(edges) - 1
    isobars = tc["contours"]["levels"]["mslp"]
    assert isobars[0] == 880 and all(b - a == 4 for a, b in zip(isobars, isobars[1:]))
