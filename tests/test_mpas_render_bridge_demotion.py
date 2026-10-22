"""Keep retired MPAS data-path Python off the product render route.

The Rust ``rw_mpas_convert`` owns this route. The Python reference tools
have been retired; their parity receipt remains in evidence/. This test
path stays because it is part of the source distribution's file listing.
"""

from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BRIDGE_MODULE_NAMES = (
    "mpas_history_to_wrfout",
    "mpas_resample_weights",
    "mpas_render_bridge",
)


def _bridge_references(root: Path) -> list[str]:
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for name in BRIDGE_MODULE_NAMES:
            if name in text:
                offenders.append(f"{path.name} mentions {name}")
    return offenders


def test_reference_tools_are_retired() -> None:
    assert not (REPO / "tools" / "mpas_render_bridge").exists()


def test_no_shipped_module_reaches_the_render_bridge() -> None:
    offenders = _bridge_references(REPO / "gpuwm")
    assert offenders == [], "\n".join(offenders)


def test_the_scan_fires_on_a_planted_product_path(tmp_path: Path) -> None:
    (tmp_path / "clean.py").write_text("x = 1\n", encoding="utf-8")
    assert _bridge_references(tmp_path) == []
    (tmp_path / "planted.py").write_text(
        "import mpas_history_to_wrfout\n", encoding="utf-8"
    )
    assert _bridge_references(tmp_path) == ["planted.py mentions mpas_history_to_wrfout"]
