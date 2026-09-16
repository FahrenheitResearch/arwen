"""``python -m gpuwm`` is the ``gpuwm`` console script by another name.

THE BREAKAGE THIS PREVENTS: the terminal's Copernicus CDS key panel reaches
the engine as ``python -m gpuwm cds-credentials --json``, the one caller in
the product that spells the door this way.  Without this module that
command exits with "No module named gpuwm.__main__", the panel reads
"Could not read CDS settings", and a key typed into the product is never
written anywhere, so the ERA5 fetch that follows refuses with a credential
sentence no matter how carefully the key was entered.
"""
from __future__ import annotations

import sys

from gpuwm.cli import main

if __name__ == "__main__":
    sys.exit(main())
