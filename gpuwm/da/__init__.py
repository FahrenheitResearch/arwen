"""EXPERIMENTAL (ArWen v1.2): data-assimilation and ensemble support.

Nothing in this package is on a certified forecast path, and nothing here is
wired into a default route.  The forecast executable, the physics driver, the
product side, and every certified verification case run exactly as they did
before this package existed; these modules are libraries a caller must reach
for explicitly.  Every public entry point stamps ``status="experimental"``
into its provenance so a downstream product can refuse to ship a field that
passed through it.

Contents (each landed by its own v1.2 lane):

- :mod:`gpuwm.da.perturb` -- ensemble initial-condition perturbations with a
  measured horizontal length scale; ``apply_perturbations(state, seed, cfg)``.
- :mod:`gpuwm.da.obsop` -- observation operators H(x) on the model grid
  (``simulated_reflectivity``, ``radial_velocity``).
- :mod:`gpuwm.da.hotstart` -- rung-1 reflectivity nudging / latent-heat
  insertion increments.
- :mod:`gpuwm.da.letkf` -- the local ensemble transform Kalman filter,
  batched; returns per-member increments, never a state.
- :mod:`gpuwm.da.osse` -- the twin experiment that gates the filter.
- :mod:`gpuwm.da.obs_radar` -- adapter from ``gpuwm-obs.radar-grid.v1`` files
  to the filter's observation structures.
- :mod:`gpuwm.da.positivity` -- the caller-side hydrometeor positivity policy
  the filter deliberately refuses to own.
- :mod:`gpuwm.da.moments` -- what a multi-moment scheme's analysis state
  vector must contain, and the guard that keeps an analysis from leaving a
  species holding mass with a zero number concentration.
- :mod:`gpuwm.da.enprod` -- ensemble products (mean, spread, probability,
  paintball, PMM) over a ``gpuwm-ensemble-manifest.v1`` roster.
- :mod:`gpuwm.da.radar_assimilation` -- the real-radar ``assimilate()`` for
  the cycle driver: member checkpoints -> H(x) -> LETKF -> increments in the
  checkpoint's own (staggered) field vocabulary, plus observation-space
  innovation statistics.

The deliberate absence of any submodule import here is a design statement, and
it is load-bearing twice over.  A ``from gpuwm.da import *`` that pulled in
cupy at import time would make every DA symbol a GPU dependency for callers
that only wanted a dataclass, and would break the test suite's ``-m "not
gpu"`` guarantee (see ``tests/conftest.py``).  It also keeps several lanes
landing modules in this package from all conflicting on the same lines.
Import the module you mean.

EXPERIMENTAL.  Interfaces here are not covered by the v1 stability promise and
may change without a deprecation cycle.
"""

from __future__ import annotations

#: Stamped into every provenance dict this package emits.
STATUS = "experimental"

#: Consumers that refuse experimental code can gate on this.
EXPERIMENTAL = True

__all__ = ["STATUS", "EXPERIMENTAL"]
