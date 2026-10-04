"""Bind native member inputs to the exact prepared boundary intervals."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math

HEAD_SCHEMA = "gpuwm-ensemble-physical-boundary-head.v1"
CATALOG_SCHEMA = "gpuwm-ensemble-physical-boundary-catalog.v1"
KEY = "ensemble_physical"


def _copy(value):
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _utc(value):
    value = datetime.fromisoformat(value) if isinstance(value,str) else value
    if not isinstance(value,datetime):
        raise ValueError("physical boundary time must be a datetime")
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def make_head(specification, schedule):
    """Freeze the provider plan, original member and actual native clock."""
    from gpuwm.ensemble.posted_physical import validate_posted_physical_input
    if (not isinstance(specification,dict)
            or set(specification) != {"provider_plan","member_index","initial_receipt"}
            or type(specification["member_index"]) is not int):
        raise ValueError("physical boundary head needs a provider plan, original member and full initial binding")
    plan = _copy(specification["provider_plan"])
    start, end = (_utc(plan["recipe"][key]) for key in ("start","end"))
    if not schedule:
        raise ValueError("physical boundary head requires its native forcing schedule")
    previous = 0.
    times = [start]
    for interval in schedule:
        if (len(interval)!=2 or any(not math.isfinite(float(value)) for value in interval)
                or float(interval[0]) != previous or float(interval[1]) <= previous):
            raise ValueError("physical boundary intervals must form the native contiguous clock from zero")
        previous=float(interval[1])
        times.append(start+timedelta(seconds=previous))
    if times[-1] != end:
        raise ValueError("physical boundary clock differs from the complete recipe window")
    initial = _copy(specification["initial_receipt"])
    validate_posted_physical_input(initial, provider_plan=plan,
                                  member_index=specification["member_index"], valid_time=start)
    return {"schema":HEAD_SCHEMA,"provider_plan":plan,"member_index":specification["member_index"],
            "forcing_valid_times":[instant.isoformat() for instant in times],
            "initial_receipt":{"sha256":digest(initial),"binding":initial}}


def validate_head(head, schedule):
    if not isinstance(head,dict) or set(head)!={"schema","provider_plan","member_index",
                                               "forcing_valid_times","initial_receipt"}:
        raise ValueError("physical boundary head has an unsupported shape")
    expected=make_head({"provider_plan":head["provider_plan"],"member_index":head["member_index"],
                        "initial_receipt":head["initial_receipt"]["binding"]},schedule)
    if head != expected:
        raise ValueError("physical boundary head changed its plan, initial binding or native clock")
    return head


def record(head, index, binding):
    from gpuwm.ensemble.posted_physical import validate_posted_physical_input
    if type(index) is not int or not 0<=index<len(head["forcing_valid_times"]):
        raise ValueError("physical binding index lies outside the native forcing schedule")
    binding=_copy(binding)
    validate_posted_physical_input(binding,provider_plan=head["provider_plan"],
        member_index=head["member_index"],valid_time=_utc(head["forcing_valid_times"][index]))
    result={"sha256":digest(binding),"binding":binding}
    if index==0 and result!=head["initial_receipt"]:
        raise ValueError("initial physical binding changed after the prepared head")
    return result


def validate_record(head,index,value):
    if (not isinstance(value,dict) or set(value)!={"sha256","binding"}
            or record(head,index,value["binding"])!=value):
        raise ValueError("physical boundary record differs from its full input binding")
    return value


def head_catalog(head):
    # Flat knot keys preserve PreparedCacheStream's append-only user metadata.
    return {"schema":CATALOG_SCHEMA,"provider_plan_sha256":digest(head["provider_plan"]),
            "member_index":head["member_index"],"knot/0":head["initial_receipt"]}


def segment_records(head,index,bindings,seen):
    result={}
    for knot in (index,index+1):
        if knot not in bindings:
            raise ValueError(f"physical boundary interval {index} lacks native input knot {knot}")
        value=record(head,knot,bindings[knot])
        if knot in seen and seen[knot]!=value:
            raise ValueError("physical input changed between adjacent native boundary intervals")
        result[str(knot)]=value
    seen.update({int(key):_copy(value) for key,value in result.items()})
    return result


def validate_segment(head,index,records,seen):
    if not isinstance(records,dict) or set(records)!={str(index),str(index+1)}:
        raise ValueError("physical boundary segment must bind exactly its two native endpoints")
    checked={}
    for key,value in records.items():
        knot=int(key)
        checked[knot]=validate_record(head,knot,value)
        if knot in seen and seen[knot]!=value:
            raise ValueError("physical boundary endpoint changed after it was consumed")
    seen.update({key:_copy(value) for key,value in checked.items()})
    return records


def complete_catalog(head,records,provider_seal):
    from gpuwm.ensemble.posted_physical import validate_provider_seal
    expected=set(range(len(head["forcing_valid_times"])))
    if set(records)!=expected:
        raise ValueError("physical input seal is missing native boundary endpoints")
    receipts=[]
    for index,value in records.items():
        validate_record(head,index,value)
        receipts.append(value["binding"]["provider_receipt"])
    validate_provider_seal(provider_seal,provider_plan=head["provider_plan"],receipts=receipts)
    return {**head_catalog(head),**{f"knot/{index}":value for index,value in records.items()},
            "provider_seal":_copy(provider_seal)}


def validate_catalog(head,catalog,records):
    if not isinstance(catalog,dict) or "provider_seal" not in catalog:
        raise ValueError("physical input seal lacks complete source verification")
    expected=complete_catalog(head,records,catalog["provider_seal"])
    if catalog != expected:
        raise ValueError("physical input seal differs from the exact boundary endpoints consumed")
    return catalog
