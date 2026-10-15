"""Recovery policy is deterministic, bounded and engine independent."""
from __future__ import annotations


def next_action(row, result, history, *, now, cutoff):
    code = result.get("exit_code", 1)
    if code == 0:
        return {"action": "complete"}
    if result.get("error_class") == "admission_wait":
        if now >= cutoff:
            return {"action": "halt", "reason": "card_admission_deadline"}
        return {"action": "wait", "reason": "card_admission", "retry_at": min(cutoff, now + 30), "attempt": history.get("attempt", 0) + 1}
    if row.get("kind") == "da":
        return {"action": "skip", "reason": "da_failed", "initialization": "public_analysis"}
    if code == 75:
        source = history.get("source_index", 0)
        if now < cutoff:
            return {"action": "wait", "retry_at": min(cutoff, now + max(1, result.get("retry_after_seconds", 30)))}
        if source + 1 < len(row.get("sources", [])):
            return {"action": "source", "source_index": source + 1}
        return {"action": "skip", "reason": "source_behind"}
    if code == 130:
        return {"action": "halt", "reason": "interrupted"}
    if code == 2:
        # Repeating a configuration refusal consumes cards and changes nothing.
        return {"action": "halt", "reason": "door_refused", "remedy": result.get("remedy")}
    retries = history.get("retries", 0)
    if retries < 3:
        return {"action": "retry", "retries": retries + 1}
    if row.get("resume_argv") and not history.get("checkpoint_tried"):
        return {"action": "checkpoint", "checkpoint_tried": True}
    degrade = history.get("degrade_index", -1) + 1
    if degrade < len(row.get("degrade", [])):
        return {"action": "degrade", "degrade_index": degrade}
    return {"action": "halt", "reason": "recovery_exhausted", "remedy": result.get("remedy")}


def replan(rows, available_cards):
    """Priority one survives; inadequate capacity produces a named halt."""
    keep, drop = [], []
    mandatory = max((int(r.get("cards", 0)) for r in rows if r.get("priority", 2) == 1), default=0)
    if mandatory > available_cards:
        return {"halt": "priority_one_capacity", "required_cards": mandatory, "available_cards": available_cards}
    # Mandatory rows may serialize on the survivor. Never discard one because
    # their combined requests exceed simultaneous capacity.
    used = mandatory
    for row in sorted(rows, key=lambda r: (r.get("priority", 2), r["id"])):
        cards = int(row.get("cards", 0))
        if row.get("priority", 2) == 1:
            keep.append(row["id"])
            continue
        (keep if used + cards <= available_cards else drop).append(row["id"])
        if row["id"] in keep:
            used += cards
    return {"keep": keep, "drop": drop}


def lost_box(plan, survivor):
    profile = plan.get("single_box_profile")
    if not isinstance(profile, dict):
        return {"halt": "single_box_profile_missing", "survivor": survivor}
    return {"action": "profile", "plan": profile, "survivor": survivor}


def disk_victim(rows):
    choices = [r for r in rows if r.get("priority", 2) != 1]
    return max(choices, key=lambda r: (r.get("priority", 2), r["id"]))["id"] if choices else None
