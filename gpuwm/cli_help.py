"""A short first-use guide over the complete existing command parser."""
from __future__ import annotations

import argparse
import sys


_START = """ArWen — GPU weather forecasts
usage: gpuwm COMMAND [OPTIONS]

Create and launch
  gpuwm tui                       Open Research (W) or Scenario (I)
  gpuwm research --help           Browse recipes, select hardware, create a study
  gpuwm domain                    Create a configuration with guided questions
  gpuwm domain-fit --help          Fit an editable starter to your GPU
  gpuwm go CONFIG --dry-run       Review the launch plan
  gpuwm go CONFIG                 Prepare and run your forecast
  gpuwm sources                   List input models and acquisition routes

Use existing WRF files
  gpuwm run --wrfinput DIR         Start from wrfinput/wrfbdy and namelist.input
  gpuwm run --met-em DIR           Start from WPS met_em and namelist.input

Continue a forecast
  gpuwm resume CONFIG --outdir RUN_DIR
                                  Find a checkpoint in an existing run
  gpuwm go --help                  Prepared-bundle and explicit restart options

Set up or troubleshoot
  gpuwm setup --with-geog          Install native tools, tables and geography
  gpuwm doctor                    Check this installation and show remedies
  gpuwm version                   Show the version and code actually running

CONFIG is your .toml file; DIR and RUN_DIR are your folders.
Presets are starting points. Your configuration keeps every setting editable.
  gpuwm COMMAND --help             All options for a command
  gpuwm --help-all                 List every command
"""


class AllCommandsHelp(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        parser._print_message(argparse.ArgumentParser.format_help(parser), sys.stdout)
        parser.exit()


class ForecastParser(argparse.ArgumentParser):
    """Only the top level is abbreviated; command parsers remain exhaustive."""

    def format_help(self):
        if self.prog == "gpuwm":
            return _START
        return super().format_help()

    def _check_value(self, action, value):
        if (self.prog == "gpuwm"
                and isinstance(action, argparse._SubParsersAction)
                and value not in action.choices):
            raise argparse.ArgumentError(
                action, f"invalid choice: {value!r}; run gpuwm --help for common "
                "actions or gpuwm --help-all for every command")
        return super()._check_value(action, value)
