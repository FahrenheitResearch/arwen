"""#234: the MPAS render bridge is a reference generator, not a product path.

``tools/mpas_render_bridge/`` is 1,722 lines of data-path Python: it reads an
MPAS history file, regrids every field onto a structured window and writes the
NetCDF frame the renderer draws.  Read, regrid, write -- three of the four
things the 2.5 Python boundary names as Rust-only.

The Rust crate ``rw-mpas`` (binary ``rw_mpas_convert``) now does all of it on
the release line, proven bit for bit against this exact pair in
``evidence/rw-mpas-converter-parity.json``.  So the pair stays as the
reference that receipt is about -- deleting it would make the parity claim
unreproducible -- but a bare run of either script must refuse and hand the
user the Rust command, or the demotion is a comment rather than a fact.

These tests drive the real scripts as subprocesses, never an import of their
internals.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
BRIDGE = REPO / "tools" / "mpas_render_bridge"
CONVERTER = BRIDGE / "mpas_history_to_wrfout.py"
WEIGHTS = BRIDGE / "mpas_resample_weights.py"

DEMOTION_EXIT_CODE = 78
ACKNOWLEDGEMENT = "--regenerate-reference"


def _run(script: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *argv],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )


def _refusal_text(result: subprocess.CompletedProcess[str]) -> str:
    return result.stdout + result.stderr


@pytest.mark.parametrize("script", [CONVERTER, WEIGHTS], ids=lambda p: p.name)
def test_bare_run_refuses_naming_the_breakage_and_the_rust_remedy(script: Path) -> None:
    """A bare default run is the case that must stop showing the defect."""

    result = _run(script)
    text = _refusal_text(result)
    assert result.returncode == DEMOTION_EXIT_CODE, text
    # names the breakage
    assert "not a product path" in text
    assert "Python boundary" in text
    # names the remedy, as a command the user can run
    assert "rw_mpas_convert" in text
    assert "--history" in text and "--mesh" in text and "--out-dir" in text
    # names the door for the one legitimate use
    assert ACKNOWLEDGEMENT in text


def test_a_fully_specified_conversion_still_refuses(tmp_path: Path) -> None:
    """Not an argparse artifact: a plausible real invocation refuses too."""

    result = _run(
        CONVERTER,
        "--history", str(tmp_path / "history.nc"),
        "--mesh", str(tmp_path / "mesh.nc"),
        "--cache-dir", str(tmp_path / "cache"),
        "--out-dir", str(tmp_path / "out"),
        "--window", "focus",
        "--field-set", "full",
    )
    assert result.returncode == DEMOTION_EXIT_CODE, _refusal_text(result)
    assert "rw_mpas_convert" in _refusal_text(result)
    assert not (tmp_path / "out").exists(), "the refusal must precede any output"


def test_a_fully_specified_weights_build_still_refuses(tmp_path: Path) -> None:
    result = _run(
        WEIGHTS,
        "--mesh", str(tmp_path / "mesh.nc"),
        "--cache-dir", str(tmp_path / "cache"),
        "--windows", "focus",
    )
    assert result.returncode == DEMOTION_EXIT_CODE, _refusal_text(result)
    assert not (tmp_path / "cache").exists(), "the refusal must precede any cache"


@pytest.mark.parametrize("script", [CONVERTER, WEIGHTS], ids=lambda p: p.name)
def test_the_acknowledgement_admits_the_reference_generator(
    script: Path, tmp_path: Path
) -> None:
    """The fixture door still opens, and it is the ONLY thing that opens it.

    With the acknowledgement the run proceeds and dies on the absent input
    file -- a different failure entirely, which is what proves the gate let
    it through rather than the gate being absent.
    """

    if script is CONVERTER:
        argv = [
            ACKNOWLEDGEMENT,
            "--history", str(tmp_path / "history.nc"),
            "--mesh", str(tmp_path / "mesh.nc"),
            "--cache-dir", str(tmp_path / "cache"),
            "--out-dir", str(tmp_path / "out"),
        ]
    else:
        argv = [
            ACKNOWLEDGEMENT,
            "--mesh", str(tmp_path / "mesh.nc"),
            "--cache-dir", str(tmp_path / "cache"),
            "--windows", "focus",
        ]
    result = _run(script, *argv)
    text = _refusal_text(result)
    assert result.returncode != DEMOTION_EXIT_CODE, text
    assert "not a product path" not in text
    assert "mesh.nc" in text, text


def test_the_documentation_surface_survives_the_demotion() -> None:
    """--print-field-map reads no data, so it stays reachable without the ack."""

    result = _run(CONVERTER, "--print-field-map")
    assert result.returncode == 0, _refusal_text(result)
    assert result.stdout.strip(), "the field map printed nothing"


@pytest.mark.parametrize("script", [CONVERTER, WEIGHTS], ids=lambda p: p.name)
def test_the_module_banner_names_what_supersedes_it(script: Path) -> None:
    """A reader who opens the file learns its status before its code."""

    head = script.read_text(encoding="utf-8")[:4000]
    assert "rw-mpas" in head
    assert "rw_mpas_convert" in head
    assert "not a product path" in head.lower()
    assert "rw-mpas-converter-parity.json" in head


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


def test_no_shipped_module_reaches_the_render_bridge() -> None:
    """The demotion is only real while nothing in the product imports it."""

    offenders = _bridge_references(REPO / "gpuwm")
    assert offenders == [], "\n".join(offenders)


def test_the_scan_fires_on_a_planted_product_path(tmp_path: Path) -> None:
    """Validate the instrument: a scan that found nothing everywhere is not a gate."""

    (tmp_path / "clean.py").write_text("x = 1\n", encoding="utf-8")
    assert _bridge_references(tmp_path) == []
    (tmp_path / "planted.py").write_text(
        "import mpas_history_to_wrfout\n", encoding="utf-8"
    )
    assert _bridge_references(tmp_path) == ["planted.py mentions mpas_history_to_wrfout"]
