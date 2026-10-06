"""Compatibility spelling for the packaged native CF fetch door."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from gpuwm.cf_archive_fetch import acquire
from gpuwm.fetch import Area


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--frames", type=int, default=2)
    for name in ("north", "south", "west", "east"):
        parser.add_argument("--" + name, type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refetch", action="store_true")
    return parser


def fetch(arguments):
    if arguments.frames < 2:
        raise ValueError("--frames must be at least 2 for lateral boundaries")
    return acquire(source="20crv3-cf", cycle=datetime.fromisoformat(arguments.start),
        hours=3 * (arguments.frames - 1), cadence=3,
        area=Area(arguments.south, arguments.west, arguments.north, arguments.east),
        out=arguments.output, force=arguments.refetch)


def main(argv=None):
    print(json.dumps(fetch(_parser().parse_args(argv)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
