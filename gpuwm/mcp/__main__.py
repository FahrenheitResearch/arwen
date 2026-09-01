"""``python -m gpuwm.mcp``: the stdio server, same door as ``arwen-mcp``."""

from __future__ import annotations

import sys

from gpuwm.mcp.server import main

if __name__ == "__main__":
    sys.exit(main())
