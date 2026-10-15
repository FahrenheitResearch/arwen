"""Bind the shipped uniform-wind initializer to ordinary member start states."""
from __future__ import annotations

import math
from numbers import Real
import struct


def is_initial_perturbation(value):
    return isinstance(value, dict) and value.get("kind") == "uniform-wind"


def validate_initial_perturbation(value):
    if not is_initial_perturbation(value):
        raise ValueError(f"perturbation {value!r} has no initial-state consumer: "
            "every member would start unperturbed under its name. "
            "Next: use kind='uniform-wind' with amplitude in m/s, or remove perturbation.")
    if set(value) - {"kind", "amplitude"}:
        raise ValueError("uniform-wind takes only kind and amplitude in m/s; other controls have no consumer")
    amplitude = value.get("amplitude", 0.5)
    try:
        valid = (not isinstance(amplitude, bool) and isinstance(amplitude, Real)
                 and math.isfinite(amplitude) and amplitude > 0
                 and 0 < struct.unpack("f", struct.pack("f", amplitude))[0] < math.inf)
    except (OverflowError, struct.error):
        valid = False
    if not valid:
        raise ValueError("uniform-wind amplitude must be positive finite float32 m/s")
    return {"kind": "uniform-wind", "amplitude": float(amplitude)}


def initial_perturbation_callback(value, *, member_id, seed, previous=None,
                                  record=None, array_module=None):
    value = validate_initial_perturbation(value)
    receipts = {}
    def initialize(*, model=None, state=None, cfg=None, **kwargs):
        if previous is not None:
            previous(model=model, state=state, cfg=cfg, **kwargs)
        nodes = ([(getattr(node.cfg, "grid_id", None), node.state, node.cfg.run)
                  for node in model.walk_parent_first()
                  if node.state is not None and getattr(node, "_started", True)]
                 if model is not None else [(getattr(cfg, "grid_id", None), state, cfg)])
        from gpuwm.ensemble.batch_perturbation import initialize_member_winds
        from gpuwm.ensemble.surface_recipe import _advanced
        for domain_id, selected, run in nodes:
            if selected is None:
                raise ValueError("uniform-wind initialization supplied no initialized state")
            identity = {"member_id": member_id, "seed": seed, "options": value}
            existing = getattr(selected, "_ensemble_initial_wind", None)
            if existing is not None:
                if existing["identity"] != identity:
                    raise ValueError("uniform-wind state belongs to another member, seed or amplitude")
                receipt = existing["receipt"]
            elif _advanced(selected):
                receipt = {**identity, "domain_id": domain_id, "applied": False,
                           "reason": "continued domains retain their advanced initial state"}
            else:
                import sys
                print(f"WARNING ensemble member {member_id}: uncalibrated spread, WRF default amplitudes; "
                      "uniform-wind uses its recorded illustrative engine amplitude",
                      file=sys.stderr, flush=True)
                receipt = initialize_member_winds(state=selected, cfg=run,
                    member_indices=(member_id,), seeds=(seed,),
                    phase="after_physics_before_step", amplitude=value["amplitude"],
                    array_module=array_module)
                receipt = dict(receipt, **identity, domain_id=domain_id, applied=True)
                selected._ensemble_initial_wind = {"identity": identity, "receipt": receipt}
            receipts[id(selected)] = receipt
        if record is not None:
            record({"member_id": member_id, "seed": seed,
                    "options": value, "domains": list(receipts.values()),
                    "warning": "uncalibrated spread, WRF default amplitudes",
                    "amplitude_origin": "illustrative engine uniform wind amplitude, not WRF rand_perturb"})
    return initialize
