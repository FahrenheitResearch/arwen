"""Optional NextGen preprocessing CLI; numeric preparation is owned by Rust."""
from __future__ import annotations

from pathlib import Path


def register_cli(hydro_subparsers):
    """Register ``gpuwm hydro prepare CONFIG.json`` on the shared hydro door."""
    parser = hydro_subparsers.add_parser(
        "prepare", help="prepare a NextGen hydrofabric and optional parent-grid weights in Rust"
    )
    parser.add_argument(
        "config", type=Path,
        help="JSON with geopackage, output, data_license and optional grid/subset",
    )
    parser.add_argument(
        "--cell-overlap", action="store_true",
        help="transpose sealed catchment intersections onto an exact terrain grid",
    )
    parser.set_defaults(func=_command)


def _command(args):
    # Keep registration/help independent of native artifacts and CUDA imports.
    if args.cell_overlap:
        import json
        from gpuwm.hydrofabric import prepare_cell_overlap

        print(json.dumps(prepare_cell_overlap(args.config), indent=2))
        return 0
    from gpuwm.hydrofabric import _command as prepare_command

    return prepare_command(args)
