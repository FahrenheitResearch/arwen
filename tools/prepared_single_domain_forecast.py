#!/usr/bin/env python3
"""Checkout entry point for the prepared single-domain forecast runner.

The runner's substance now lives in
:mod:`gpuwm.prepared_single_domain_forecast`, inside the installed
package, and this file is a thin delegation to it.

**Why it moved.**  A ``pip install gpuwm`` gives a reader the CLI, the
front doors and the model -- and, until this change, no way to run a
GFS forecast, because the last two stages of the documented chain were
a script path under ``tools/`` that only a git checkout has.  "Install
the product, then clone the repository to use it" is not an install.
The implementation is therefore part of the package, reachable as
``python -m gpuwm.prepared_single_domain_forecast`` (and as the
``gpuwm-prepared-forecast`` console script) from any environment that
has the wheel.

**Why this file stays.**  ``docs/public/FIRST-LIGHT.md``, retained
campaign receipts and years of pasted transcripts name
``python tools/prepared_single_domain_forecast.py``.  That command still
works, unchanged, from a checkout.

**One implementation, two entries.**  There is no copy here: the module
object this file exposes IS the package module, aliased into
``sys.modules`` so that ``import tools.prepared_single_domain_forecast``
and ``import gpuwm.prepared_single_domain_forecast`` are the same object
with the same private helpers.  A twin runner that could drift is
exactly what a receipt must never have to disambiguate.

The registry route id ``tools.prepared_single_domain_forecast`` is
unchanged: it names a route in the physics registry, not an import path.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import gpuwm.prepared_single_domain_forecast as _impl  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(_impl.main())

# Imported, not executed: BE the package module rather than re-export a
# hand-picked subset of it.  Tests and tools reach for private helpers
# (``_runtime_source_identity``, ``_sha256``, ...), and a re-export list
# would go stale the first time one of those moved.
sys.modules[__name__] = _impl
