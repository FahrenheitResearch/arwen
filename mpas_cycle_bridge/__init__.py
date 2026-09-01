"""The bridge between the cycling spine and the MPAS GPU port.

This package exists because of a hard structural fact, not a preference.
The port pins gpuwm by commit and loads it from a frozen Arwen checkout,
and it refuses to do so if ``gpuwm`` is already live from another tree::

    gpuwm was already imported from a different tree; exact frozen
    Arwen v2 Arwen cannot replace a live package

``gpuwm.cycle`` -- the cycling spine -- IS gpuwm.  So a spine that
imports the port into its own process has already claimed the package
name and the port refuses by law.  The two ways out were to rotate the
port's gpuwm pin, or to put a process boundary between the spine and the
forecast.  Rotating the pin would re-base every MPAS receipt that cites
it and would couple cycle development to a frozen commit forever.  The
process boundary costs orchestration and nothing else -- and the port's
own restart proof (15+15 steps across a fresh process, bitwise identical
to 30 continuous) is exactly the evidence that a process boundary here
is lossless.

So: **nothing in this package may import gpuwm.**  Everything the
forecast worker needs lives here, gpuwm-free, and the spine imports
*from* this package rather than the other way round.  The dependency
arrow points away from gpuwm, which is what makes the constraint
structural instead of a comment somebody will one day not read.

The worker enforces it at runtime as well -- see
:mod:`mpas_cycle_bridge.spine_guard`.
"""

from __future__ import annotations

__all__ = ["BRIDGE_SCHEMA"]

#: Version of the segment format the worker writes and the spine reads.
BRIDGE_SCHEMA = "gpuwm.cycle.bridge/v1"
