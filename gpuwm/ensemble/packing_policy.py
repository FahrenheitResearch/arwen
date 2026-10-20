"""Measured ordinary-member concurrency, below the live memory ceiling."""
from __future__ import annotations


# Selected from the regional throughput sweep documented in
# docs/dev/ensemble-packing-benchmark.md. This bounds ordinary models and
# component cohorts; qualified native member batches keep their own policy.
DEFAULT_ORDINARY_MEMBER_CAP = 2
POLICY = "gpuwm-ordinary-member-packing-policy-v1"


def ordinary_member_policy(request, *, concurrent):
    """Resolve an explicit cap or the default without changing the request.

    Admission still prices the complete selected physics on each sampled
    card. A small card can execute fewer members, including a streamed
    member. Sequential checkpoint/radar/binding routes keep their policy.
    """
    explicit = request.max_ordinary_members_per_device
    selected = explicit
    source = "explicit_request" if explicit is not None else "existing_memory_admission"
    if explicit is None and concurrent and request.members > 1 and request.recipe != "wofs":
        selected = DEFAULT_ORDINARY_MEMBER_CAP
        source = "measured_default"
    elif explicit is None and concurrent and request.recipe == "wofs":
        source = "live_memory_admission"
    return selected, {"schema": POLICY, "source": source,
        "max_ordinary_members_per_device": selected,
        "physical_memory_ceiling": "unchanged_per_card_admission",
        "numerics": "original_member_physics_and_clocks"}
