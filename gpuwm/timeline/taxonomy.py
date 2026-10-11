"""Frozen C1 v1 vocabulary. Families permit dotted, generic descendants.

Intervals can nest and overlap. forecast.construct includes runner startup,
input validation, restore, model construction and initial output, ending at
the start of the first measured model step. forecast.integrate includes the
first step and ends at the end of the last measured step, before final drain.
Never add interval durations to compute rental wall time.
"""

import math
import re
from datetime import datetime

VERSION = 1
PRODUCTS = ("regional", "hex", "da", "ensemble", "multicard", "global")
PHASES = ("begin", "end", "mark")
STAGES = (
    "rental.requested", "rental.running", "rental.terminated",
    "worker.claim", "forecast.construct", "forecast.first_step",
    "forecast.integrate", "finish",
)
FAMILIES = ("boot", "linger", "store", "fetch", "prep",
            "forecast.output", "post", "upload")
GO_STAGES = {
    "authority": "prep.authority", "fetch": "fetch.inputs",
    "chem fetch": "fetch.chemistry", "manifest": "prep.manifest",
    "prepare": "prep.inputs", "forecast": "post.forecast_process",
    "render": "post.render",
}
RUN_STAGES = {"fetch": "fetch.inputs", "prepare": "prep.inputs",
              "initialize": "prep.initialize", "forecast": "post.forecast_process",
              "finalize": "post.finalize"}
KEYS = frozenset(("v", "t_utc", "t_mono", "host", "rental_id", "run_id",
                  "product", "stage", "phase", "card", "pid", "attrs"))
_DOTTED = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")


def valid_stage(stage):
    return isinstance(stage, str) and bool(_DOTTED.fullmatch(stage)) and (
        stage in STAGES or any(stage.startswith(family + ".") for family in FAMILIES))


def validate(event):
    """Strict consumer validation; emitters catch validation failures."""
    if set(event) != KEYS or event["v"] != VERSION or type(event["v"]) is not int:
        raise ValueError("C1 requires exactly the v1 envelope keys")
    if event["product"] not in PRODUCTS or event["phase"] not in PHASES:
        raise ValueError("unknown C1 product or phase")
    if not valid_stage(event["stage"]):
        raise ValueError("unknown C1 stage or malformed dotted name")
    if not isinstance(event["t_utc"], str) or not event["t_utc"].endswith("Z"):
        raise ValueError("C1 t_utc must end in Z")
    datetime.fromisoformat(event["t_utc"].replace("Z", "+00:00"))
    if type(event["t_mono"]) not in (int, float) or not math.isfinite(event["t_mono"]):
        raise ValueError("C1 t_mono must be finite")
    if not isinstance(event["host"], str) or not event["host"]:
        raise ValueError("C1 host must be a nonempty string")
    for key in ("rental_id", "run_id"):
        if event[key] is not None and not isinstance(event[key], str):
            raise ValueError("C1 ids must be strings or null")
    if event["card"] is not None and (type(event["card"]) is not int or event["card"] < 0):
        raise ValueError("C1 card must be a nonnegative integer or null")
    if type(event["pid"]) is not int or event["pid"] <= 0 or not isinstance(event["attrs"], dict):
        raise ValueError("C1 pid must be positive and attrs must be an object")
    return event
