"""Checkpoint capability shared by configuration checks and resume guidance.

Declared-input, prepared single-domain and prepared tree routes all use
the canonical checkpoint transport. A positive ``restart_interval_s``
enables writing; zero deliberately disables it. Capability is independent
of whether a particular run has already written a valid checkpoint.
"""

from __future__ import annotations

from pathlib import Path

#: The one-sentence advisory.  Detail belongs behind ``--explain``.
CHECKPOINTLESS_ROUTE_ADVISORY = (
    "restart_interval_s requires a valid forecast domain; this "
    "configuration does not declare one."
)

#: The remedies, for ``--explain`` and for the resume refusal.
CHECKPOINTLESS_ROUTE_REMEDY = (
    "Validate the configuration with gpuwm check. Single-domain and "
    "multi-domain runs can write gpuwmrst_d*.npz checkpoints when "
    "restart_interval_s is positive; resume requires a complete valid set."
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

    return bool(has_case_data) or int(domain_count) >= 1


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
