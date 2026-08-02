"""Which forecast routes write checkpoints -- said once, said early.

``restart_interval_s`` is a legal key in every experiment config, and on
one route it does nothing.  A single-domain config with no ``[case_data]``
table is executed by the prepared single-domain runner, which writes no
checkpoints at all; the runner says so, honestly, but it says so at
forecast time, which on the multi-hour run people actually checkpoint is
after the run they wanted to be able to resume has already finished.
``gpuwm check`` -- the stage whose whole job is to refuse or warn before
the time is spent -- said nothing, and ``gpuwm resume`` then pointed the
user back at the very knob that had just been declared inert.

So the route fact lives here, in one place with no heavy imports, and
both ``gpuwm check`` and ``gpuwm resume`` read it from here.  It is an
advisory, not a gate: it changes no exit code and blocks nothing
(warn-not-block).  Nothing in this module names a source, a case or a
runner script -- the discriminator is structural, which is what makes it
true for every source that reaches the prepared route.

Route matrix this encodes:

===============================  =================  ==================
config shape                     runner             checkpoints
===============================  =================  ==================
``[case_data]`` present          ``gpuwm run``      yes
no ``[case_data]``, 1 domain     prepared single    **no**
no ``[case_data]``, 2+ domains   prepared tree      yes
===============================  =================  ==================
"""

from __future__ import annotations

from pathlib import Path

#: The one-sentence advisory.  Detail belongs behind ``--explain``.
CHECKPOINTLESS_ROUTE_ADVISORY = (
    "restart_interval_s is inert on this route: a single-domain config "
    "with no [case_data] table runs on the prepared single-domain "
    "forecaster, which writes no checkpoints, so this run cannot be "
    "resumed."
)

#: The remedies, for ``--explain`` and for the resume refusal.
CHECKPOINTLESS_ROUTE_REMEDY = (
    "Checkpointing is reachable two ways: a multi-domain config, which "
    "runs on the prepared domain-tree forecaster, or a [case_data] "
    "experiment, which runs on `gpuwm run`.  Both write gpuwmrst_d*.npz "
    "sets and both resume from them."
)


def config_has_case_data(path) -> bool:
    """True when the config declares the ``[case_data]`` input table."""

    import tomllib

    try:
        with open(Path(path), "rb") as stream:
            raw = tomllib.load(stream)
    except (OSError, ValueError):
        # An unreadable config is somebody else's refusal to make; this
        # module only ever adds a sentence, so it declines to guess.
        return False
    return isinstance(raw.get("case_data"), dict)


def route_writes_checkpoints(*, domain_count: int,
                             has_case_data: bool) -> bool:
    """Whether the route this config is steered to writes checkpoints."""

    return bool(has_case_data) or int(domain_count) > 1


def checkpoint_route_advisory(*, domain_count: int, has_case_data: bool,
                              restart_interval_s: float) -> str | None:
    """The advisory when a config asks for checkpoints it will not get.

    Silent when the route can checkpoint, and silent when the config did
    not ask: a run with ``restart_interval_s = 0`` has lost nothing and
    does not need to be told about a knob it left alone.
    """

    if route_writes_checkpoints(domain_count=domain_count,
                                has_case_data=has_case_data):
        return None
    try:
        requested = float(restart_interval_s)
    except (TypeError, ValueError):
        return None
    if not requested > 0.0:
        return None
    return CHECKPOINTLESS_ROUTE_ADVISORY
