"""Keep the humidity floor and EPS citations tied to tagged WRF lines."""

import ast
import json
from pathlib import Path
import re

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_LINES = json.loads(
    (Path(__file__).parent / "fixtures" / "wrf_rh_to_mxrat1_lines.json")
    .read_text(encoding="utf-8")
)
_CITATIONS = [
    ("gpuwm/ingest/real.py", "_saturation_mixing_ratio_serial", "v4.6.1",
     r"EPS = 0\.622.*?module_initialize_real\.F:(\d+)", "eps"),
    ("gpuwm/ingest/real.py", "_saturation_mixing_ratio_serial", "v4.6.1",
     r"module_initialize_real\.F:(\d+), q = MAX", "floor"),
    ("gpuwm/ingest/real.py", "_WRF_QV_MIN_P_SAFE", "v4.6.1",
     r"RH lane \(:(\d+)", "floor"),
    ("gpuwm/ingest/real.py", "_SURFACE_QV_FLOOR_WRF_REFERENCE", "v4.6.1",
     r"rh_to_mxrat1 \(:(\d+) unconditional", "floor"),
    ("gpuwm/ingest/real.py", "_PROGNOSTIC_QV_FLOOR_WRF_REFERENCE", "v4.7.1",
     r"rh_to_mxrat1's unconditional 1e-6 floor \(:(\d+)\)", "floor"),
    ("tests/test_real_init.py", "test_qv_construction_uses_wrf_floor_and_invalid_guard",
     "v4.6.1", r"EPS = 0\.622.*?module_initialize_real\.F:(\d+)", "eps"),
    ("tests/test_real_init.py", "test_rh_lane_surface_qv_already_carries_the_same_floor",
     "v4.6.1", r"rh_to_mxrat1:(\d+)", "floor"),
]
_FACT_TEXT = {
    "eps": "EPS=0.622",
    "floor": "Q(I,K,J)=MAX(EPS*ES/(P(I,K,J)/100.-ES),1.E-6)",
}


@pytest.mark.parametrize(
    "path,symbol,tag,pattern,fact", _CITATIONS,
    ids=[f"{row[1]}-{row[4]}" for row in _CITATIONS],
)
def test_real_init_rh_to_mxrat1_citations(path, symbol, tag, pattern, fact):
    source = (_ROOT / path).read_text(encoding="utf-8")
    start = 0
    for node in ast.parse(source).body:
        names = ([target.id for target in node.targets
                  if isinstance(target, ast.Name)]
                 if isinstance(node, ast.Assign) else [getattr(node, "name", None)])
        if symbol in names:
            # Include the comment block immediately before a constant.
            block = "\n".join(source.splitlines()[start:node.end_lineno])
            break
        start = node.end_lineno
    else:
        pytest.fail(f"Missing citation carrier: {path}:{symbol}")

    cited = re.findall(pattern, block, flags=re.DOTALL)
    assert len(cited) == 1, f"Expected one {fact} citation in {path}:{symbol}: {cited}"
    upstream = next(row for row in _SOURCE_LINES if row["tag"] == tag)
    text = upstream["lines"].get(cited[0], "<line absent from tagged fixture>")
    assert _FACT_TEXT[fact] in "".join(text.upper().split()), (
        f"{path}:{symbol} cites {tag} {upstream['path']}:{cited[0]} "
        f"for {fact}, but that line says {text!r}"
    )
    assert set(re.findall(r"\bv\d+\.\d+\.\d+\b", block)) == {tag}, (
        f"{path}:{symbol} must name the checked WRF tag {tag}"
    )
