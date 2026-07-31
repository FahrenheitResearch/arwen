#!/usr/bin/env python3
"""Checkout entry point for the prepared domain-tree forecast runner.

The substance now lives in :mod:`gpuwm.prepared_domain_tree_forecast`,
inside the installed package, for the same reason its single-domain
sibling moved: a ``pip install gpuwm`` must be able to run a forecast
without cloning the repository.  It is reachable as
``python -m gpuwm.prepared_domain_tree_forecast`` and as the
``gpuwm-prepared-tree-forecast`` console script.

This file still works, unchanged, as
``python tools/prepared_domain_tree_forecast.py`` -- the spelling in
``docs/public/FIRST-LIGHT.md`` and in retained receipts -- and is a
delegation, not a copy: the module object below IS the package module,
aliased into ``sys.modules``.

The registry route id ``tools.prepared_domain_tree_forecast`` is
unchanged: it names a route in the physics registry, not an import path.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import gpuwm.prepared_domain_tree_forecast as _impl  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(_impl.main())

sys.modules[__name__] = _impl
