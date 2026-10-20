"""Render existing Rust diagnostic frames through the permanent product door."""
import argparse
from pathlib import Path

from gpuwm.storm_diagnostics import render

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('source', type=Path)
parser.add_argument('--label', required=True)
args = parser.parse_args()
print('Rendered', len(render(args.source, [500, 1000, 3000], args.label)), 'pictures')
