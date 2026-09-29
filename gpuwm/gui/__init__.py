"""``gpuwm gui``: the local web page that drives ArWen.

A standard-library HTTP server bound to this computer and a page of plain
JavaScript modules with no build step.  The server never does model
work: every button is one ``gpuwm`` command, launched detached, and the
run folder on disk is the only state, so closing the page or the server
never stops a run and a reopened page rebuilds itself from the folders.

Pictures are only what the Rust renderer wrote into a run's
``<domain>/<product>/<valid-day>/`` folders; the server serves those
files as they are and never draws or resizes one.
"""

from __future__ import annotations

DEFAULT_PORT = 8766
PORT_TRIES = 10

__all__ = ["DEFAULT_PORT", "PORT_TRIES"]
