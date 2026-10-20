"""Exercise the real demo door's duration check before Rust acquisition."""
import os
from pathlib import Path
import subprocess
import sys
import tomllib


def test_demo_settings_round_trip_resource_bindings_and_ordered_stages(tmp_path):
    from gpuwm.hydro.config import configure, resolve_configuration
    from gpuwm.hydro.runtime import read_settings, resolve_settings
    from gpuwm.hydro_demo import settings_toml
    from gpuwm import numerics

    row = resolve_configuration("ngen-shipped-noah-pet-cfe-48h").as_dict()
    settings = {
        "enabled": True,
        "realization": row["id"],
        "settings_bindings": {name: f"/prepared/{name}" for name in row["settings_parameters"]},
        "stages": row["settings"]["stages"],
    }
    document = settings_toml(settings)
    assert tomllib.loads(document) == {"hydro": settings}
    path = tmp_path / "hydro.toml"
    path.write_text(document, encoding="utf-8")
    parsed = read_settings(path)
    parsed.pop("settings_path")
    # The runtime inherits an already selected proof mode when the document
    # omits numerics; the metadata-only helper needs that mode made explicit.
    assert resolve_settings(parsed) == configure(dict(settings, numerics=numerics.mode()))


def test_invalid_simulation_duration_does_not_acquire_inputs(tmp_path):
    cache = tmp_path / "data"
    output = tmp_path / "runs"
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "gpuwm.cli", "hydro", "demo", "lower-colorado-tx",
         "--seconds", "1", "--data-dir", str(cache), "--outdir", str(output)],
        cwd=root, env=env, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "whole number of hydro intervals (300 seconds" in result.stderr
    assert not cache.exists()
    assert not output.exists()


def test_unverified_demo_data_is_refused_before_acquisition(tmp_path):
    cache = tmp_path / "data"
    output = tmp_path / "runs"
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "gpuwm.cli", "hydro", "demo", "lower-colorado-tx",
         "--prepare-only", "--data-dir", str(cache), "--outdir", str(output)],
        cwd=root, env=env, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "v201 GeoPackage data licence is unverified" in result.stderr
    assert not cache.exists()
    assert not output.exists()
