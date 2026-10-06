"""Actual native section pixels for inherited WOOF model/source labels."""
from pathlib import Path
import hashlib
import json
import os
import subprocess

import pytest

from gpuwm import cli, render_layout, rustwx
from test_render_rust import RENDERER, needs_renderer


def _native_section_fixture(tmp_path):
    """Reuse the existing Rust writer's PH-backed section-compatible fixture."""
    explicit = os.environ.get("GPUWM_SECTION_TEST_FIXTURE")
    if explicit:
        source = Path(explicit)
        assert source.is_file(), source
        return source
    profile = Path(RENDERER).resolve().parent
    root = Path(__file__).resolve().parents[1]
    directories = [profile / "deps", root / "tools/rustwx/target/release/deps",
                   root / "tools/rustwx/target/debug/deps"]
    writers = [path for directory in directories for path in directory.glob("stored_planes-*")
               if path.is_file() and path.suffix in ("", ".exe")]
    assert writers, ("native section fixture writer not built; run cargo test -p rw-wrfbatch "
                     "--release --locked --test stored_planes --no-run, or set "
                     "GPUWM_SECTION_TEST_FIXTURE to that writer's wrfout")
    writer = max(writers, key=lambda path: path.stat().st_mtime_ns)
    directory = tmp_path / "native-section-fixture"
    env = rustwx.renderer_env()
    env["RW_STORED_PLANE_FIXTURE"] = str(directory)
    done = subprocess.run([str(writer), "--exact", "write_durable_fixture", "--ignored", "--nocapture"],
                          env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    source = directory / "wrfout_d01_2026-08-19_00_00_00"
    assert source.is_file(), done.stdout + done.stderr
    return source


@needs_renderer
@pytest.mark.parametrize("layout", ["fixed", "auto"])
def test_inherited_section_version_labels_match_resolved_literals_and_keep_generic_pixels(
        tmp_path, layout):
    source = _native_section_fixture(tmp_path)
    version_theme = tmp_path / "version.json"
    literal_theme = tmp_path / "literal.json"
    version_theme.write_text(json.dumps({"extends": "woof-light", "text": {
        "source_label": "Recast WOOF {version}", "model_label": "WOOF {version}"}}), encoding="utf-8")
    literal_theme.write_text(json.dumps({"extends": "woof-light", "text": {
        "source_label": "Recast WOOF 2.8.6", "model_label": "WOOF 2.8.6"}}), encoding="utf-8")

    def render(name, *extra):
        out = tmp_path / name
        args = ["render", str(source), "--engine", "rust", "--products", "xsec:tk",
                "--timeidx", "0", "--out", str(out), "--source-label", "ArWen 2.8.6",
                "--section", "36.2,-97.8,36.6,-97.2"]
        if layout == "fixed":
            args += ["--size", "640x480"]
        else:
            args += ["--size", "auto"]
        assert cli.main([*args, *extra]) == 0
        pictures = list(out.rglob("*.png"))
        assert len(pictures) == 1, pictures
        return Path(render_layout.fs_path(pictures[0])).read_bytes()

    generic = render("generic")
    default = render("default", "--theme", "default")
    versioned = render("versioned", "--theme", str(version_theme))
    literal = render("literal", "--theme", str(literal_theme))
    assert generic == default
    assert versioned == literal
    assert versioned != generic
    evidence = os.environ.get("GPUWM_RENDERER_EXTRAS_EVIDENCE")
    if evidence:
        root = Path(evidence)
        root.mkdir(parents=True, exist_ok=True)
        (root / f"section-{layout}-generic.png").write_bytes(generic)
        (root / f"section-{layout}-woof-version.png").write_bytes(versioned)
        (root / f"section-{layout}-receipt.json").write_text(json.dumps({
            "schema": "renderer-extras.section-theme-proof.v1", "fixture": "existing native Rust stored_plane_fixture d01 3km with PH/PHB",
            "layout": layout, "generic_default_byte_identical": generic == default,
            "versioned_literal_byte_identical": versioned == literal,
            "generic_sha256": hashlib.sha256(generic).hexdigest(),
            "woof_sha256": hashlib.sha256(versioned).hexdigest(),
        }, indent=2) + "\n", encoding="utf-8")
