"""A live frame must not import every preparation and forecast command."""
from __future__ import annotations

import os
import subprocess
import sys

from gpuwm import cli


def test_render_command_leaves_unrelated_forecast_modules_unloaded():
    script = '''
import sys
from gpuwm.cli import main
try:
    main(["render", "--help"])
except SystemExit as error:
    assert error.code == 0
for name in ("gpuwm.adapt", "gpuwm.domain_wizard", "gpuwm.downscale",
             "gpuwm.prepared_single_domain_forecast"):
    assert name not in sys.modules, name
'''
    result = subprocess.run([sys.executable, "-c", script],
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--series" in result.stdout
    assert "--products" in result.stdout


def test_render_parser_retains_the_full_cli_arguments():
    words = ["render", "frame.nc", "--series", "--products", "all",
             "--context-wrfout", "earlier.nc", "--out", "pictures",
             "--run-stamp", "off", "--size", "1200x900", "--explain"]
    fast = vars(cli.build_parser(render_only=True).parse_args(words))
    full = vars(cli.build_parser().parse_args(words))
    # The combined input checker installs a default on the root parser.
    # The render handler never reads this unrelated callback.
    full.pop("ingest_preflight_handler", None)
    assert fast == full
