# SPDX-License-Identifier: Apache-2.0
"""Hydro CLI entry point for the existing Rust archive analysis launcher."""
from __future__ import annotations

from gpuwm.hydro_archive import main, register_cli as _register_archive_cli

__all__ = ["main", "register_cli"]


def register_cli(hydro_subparsers):
    """Register ``retrospective`` and its ``func(args)`` callback once."""
    return _register_archive_cli(hydro_subparsers)


if __name__ == "__main__":
    raise SystemExit(main())
