"""Read-only plot metadata for the terminal picker; no forecast defaults change."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def presets() -> dict:
    """Small curated requests, separate from the renderer-owned catalog."""
    return json.loads((Path(__file__).parent / "data" / "tui" /
                       "plot-presets.json").read_text(encoding="utf-8"))


def catalog_document() -> dict:
    """Ask the installed renderer without building, staging, or probing a GPU."""
    from gpuwm import bridges
    from gpuwm.runplan import render_catalog

    with bridges.inspection_only():
        catalog = render_catalog()
    return {**catalog, "presets": presets()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", action="store_true",
                        help="return the installed renderer catalog as JSON")
    args = parser.parse_args(argv)
    print(json.dumps(catalog_document() if args.catalog else presets()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
