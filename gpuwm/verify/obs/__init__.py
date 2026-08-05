"""Observation-verification battery: the referee and its controls.

This package is the referee half of the observation battery: the statistics,
the harness that runs them, the controls that qualify the instrument, and the
evaluator that turns a set of score files into a verdict.  Fetching and
decoding observations is a separate concern and lives elsewhere; everything
here consumes the seam types in :mod:`gpuwm.verify.obs.contracts` and nothing
else.

Verification *statistics and orchestration* live here.  Diagnostics do not:
the science core for reading and diagnosing history tapes is the ``wrf``
package (distribution ``wrf-rust``), and nothing under this package
reimplements one of its diagnostics.  The one place this package computes a
field itself is :mod:`gpuwm.verify.obs.cross_reader`, whose whole job is to
be an *independent control* on that core -- an instrument that agreed with
the thing it is checking because it called it would measure nothing.

The modules, by mechanism:

:mod:`~gpuwm.verify.obs.contracts`
    the seam: observed fields, station reports, provenance, and the protocols
    a model run and an observation archive satisfy.
:mod:`~gpuwm.verify.obs.fss`
    neighborhood fractions skill score with a validity mask, on the existing
    :mod:`gpuwm.verify.field_metrics` engine.
:mod:`~gpuwm.verify.obs.contingency`
    the 2x2 table and the scores derived from it.
:mod:`~gpuwm.verify.obs.regrid`
    the two registered remap operators, as reusable integer plans.
:mod:`~gpuwm.verify.obs.stations`
    point-observation matching, admission and surface statistics.
:mod:`~gpuwm.verify.obs.model_source`
    one reader for both models' history files, science through the mandated
    core.
:mod:`~gpuwm.verify.obs.registration`
    the pins, hashed before any score is looked at.
:mod:`~gpuwm.verify.obs.battery`
    the scoring pass and the score file it writes.
:mod:`~gpuwm.verify.obs.controls`
    the instrument-qualification controls.
:mod:`~gpuwm.verify.obs.promotion`
    the four-clause promotion evaluator.
:mod:`~gpuwm.verify.obs.stubs`
    loud stand-ins for observations that do not exist yet, which can never
    become a verdict.
:mod:`~gpuwm.verify.obs.cross_reader`
    the independent cross-reader control on the science core (imported
    lazily, like ``model_source``, so the referee imports without it).
:mod:`~gpuwm.verify.obs.t0_parity`
    the t=0 parity check between the two models' initial states.

Nothing in this package names a case.  ``model_source``, ``cross_reader``
and ``t0_parity`` are imported lazily by callers that open history files, so
importing the scorer does not require the science core to be installed.
"""

from gpuwm.verify.obs import (
    battery, contingency, contracts, controls, fss, promotion, regrid,
    registration, stations, stubs,
)

__all__ = [
    "battery", "contingency", "contracts", "controls", "fss", "promotion",
    "regrid", "registration", "stations", "stubs",
]
