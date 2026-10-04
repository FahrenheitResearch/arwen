"""Member-local first-RK stochastic forcing and lossless checkpoint payloads.

Only the nonmicrophysics physics sum passes through this seam. Pattern updates
are separate from acoustic/RK substeps; returned rates are held by the ordinary
dycore. An off binding performs no device operation or pattern allocation.
"""
from __future__ import annotations

from dataclasses import replace
from collections.abc import Mapping
import numpy as np


class StochasticPhysicsBinding:
    def __init__(self, hook, *, member_id, clock=None, mass_factor_provider=None,
                 recipe_sha256=None):
        self.hook, self.member_id, self.clock = hook, int(member_id), clock
        self.mass_factor_provider = mass_factor_provider
        self.recipe_sha256 = recipe_sha256
        self.applied_steps = 0

    @property
    def enabled(self):
        return bool(getattr(self.hook, "enabled", False))

    def next_update_index(self):
        """Continue the exact restored spectral history across reminted clocks."""
        return int(self.hook.completed_step) + 1

    def before_physics(self, state, cfg, *, step=None):
        if not self.enabled:
            return
        if step is None:
            step = self.next_update_index()
        accepted_step = getattr(self.hook, "set_time_step", None)
        if accepted_step is not None:
            accepted_step(float(cfg.dt))
        self.hook.before_timestep(int(step))
        patterns = getattr(self.hook, "parameter_patterns", None)
        if patterns:
            bind = getattr(getattr(state, "physics", None), "bind_spp_patterns", None)
            if bind is None:
                raise RuntimeError("SPP patterns require the corresponding physics parameter consumers")
            bind(patterns)

    def after_nonmicrophysics(self, state, cfg, tendencies):
        if not self.enabled:
            return tendencies
        if tendencies is None:
            from gpuwm.core.physics import PhysicsTendencies
            tendencies = PhysicsTendencies.zeros(state)
        rates = {"u": tendencies.ru, "v": tendencies.rv,
                 "theta": tendencies.rtheta, "qv": tendencies.rqv}
        if any(value is None for value in rates.values()):
            raise RuntimeError("stochastic forcing requires the complete nonmicrophysics tendency representation")
        factors = None
        if getattr(self.hook, "skebs", None) is not None:
            factors = (coupled_mass_factors(state, cfg) if self.mass_factor_provider is None
                       else self.mass_factor_provider(state, cfg))
        result = self.hook.after_nonmicrophysics(rates, tendency_scope="nonmicrophysics",
                                                mass_factors=factors)
        if getattr(self.hook, "sppt", None) is not None or getattr(self.hook, "skebs", None) is not None:
            close_stochastic_periodic_faces(result, cfg)
        self.applied_steps += 1
        return replace(tendencies, ru=result["u"], rv=result["v"],
                       rtheta=result["theta"], rqv=result["qv"])

    def snapshot(self):
        return {"contract": "gpuwm-ensemble-stochastic-binding.v1",
                "member_id": self.member_id, "recipe_sha256": self.recipe_sha256,
                "applied_steps": self.applied_steps, "hook": self.hook.snapshot()}

    def restore(self, snapshot):
        self.validate_identity(snapshot)
        self.hook.restore(snapshot["hook"])
        self.applied_steps = int(snapshot["applied_steps"])

    def validate_identity(self, snapshot):
        if (snapshot.get("contract") != "gpuwm-ensemble-stochastic-binding.v1"
                or snapshot.get("member_id") != self.member_id
                or snapshot.get("recipe_sha256") != self.recipe_sha256):
            raise ValueError("stochastic checkpoint belongs to another member or source recipe")
        if type(snapshot.get("applied_steps")) is not int or snapshot["applied_steps"] < 0:
            raise ValueError("stochastic checkpoint has an invalid applied step count")


def close_stochastic_periodic_faces(rates, cfg):
    """The periodic closing momentum face is the original face zero word.

    Spectral patterns include an extra edge that can differ from face zero.
    Letting that independently force the duplicate face changes acoustic mass
    fluxes before the ordinary epilogue closes state. Establish the original
    periodic face contract on the held forced rates, using copies only.
    """
    from gpuwm.core.dycore import _boundary_x, _boundary_y
    if not _boundary_x(cfg):
        rates["u"][..., -1] = rates["u"][..., 0]
    if not _boundary_y(cfg):
        rates["v"][..., -1, :] = rates["v"][..., 0, :]


def coupled_mass_factors(state, cfg):
    """The original physics coupling representation for unit physical rates.

    Momentum uses the established A-to-C interpolation, ring masks, map-factor
    divisions and periodic closure. Theta uses its distinct map scaling.
    Moisture remains unforced by SKEBS and is never assigned theta's factor.
    """
    from gpuwm.core.physics import _couple_momentum_to_faces
    mass = state.c1h[:, None, None] * state.total_mu()[None] + state.c2h[:, None, None]
    u, v = _couple_momentum_to_faces(state, cfg, mass, mass)
    theta = mass / state.msft[None] if state.has_msf else mass
    return {"u": u, "v": v, "theta": theta}


def checkpoint_payload(binding, *, array_module=None):
    """Separate JSON metadata and exact FP32 words of complex64 spectra.

    This encoder does no host transfer and performs no floating arithmetic.
    Native restart writers can persist each returned float32 view directly.
    """
    if array_module is None:
        import cupy as array_module
    arrays = {}
    def encode(value, path):
        if isinstance(value, array_module.ndarray):
            if value.dtype != np.dtype("complex64") or not value.flags.c_contiguous:
                raise ValueError("stochastic checkpoint requires contiguous complex64 spectra")
            key = "stochastic/" + "/".join(path)
            words = value.view(np.float32).reshape(value.shape + (2,))
            arrays[key] = words
            return {"array": key, "dtype": "complex64", "shape": list(value.shape)}
        if isinstance(value, Mapping):
            return {name: encode(item, path + (str(name),)) for name, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [encode(item, path + (str(i),)) for i, item in enumerate(value)]
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        raise TypeError("stochastic checkpoint metadata contains an unsupported owner")
    return encode(binding.snapshot(), ("binding",)), arrays


def restore_checkpoint_payload(binding, metadata, arrays, *, array_module=None):
    """Rebuild borrowed complex views and validate the complete provider state."""
    restored = decode_checkpoint_payload(metadata, arrays, array_module=array_module)
    binding.restore(restored)


def decode_checkpoint_payload(metadata, arrays, *, array_module=None):
    """Validate spectrum word inventory before restoring any model state."""
    if array_module is None:
        import cupy as array_module
    used = set()
    def decode(value):
        if isinstance(value, dict) and "array" in value:
            if set(value) != {"array", "dtype", "shape"} or value["dtype"] != "complex64":
                raise ValueError("stochastic checkpoint has an unknown array descriptor")
            key, shape = value["array"], tuple(value["shape"])
            if key in used or key not in arrays:
                raise ValueError("stochastic checkpoint array inventory is incomplete or duplicated")
            words = array_module.asarray(arrays[key])
            if (words.dtype != np.dtype("float32") or words.shape != shape + (2,)
                    or not words.flags.c_contiguous):
                raise ValueError("stochastic spectrum words have the wrong shape or dtype")
            used.add(key)
            return words.view(np.complex64).reshape(shape)
        if isinstance(value, dict):
            return {name: decode(item) for name, item in value.items()}
        if isinstance(value, list):
            return [decode(item) for item in value]
        return value
    restored = decode(metadata)
    if set(arrays) != used:
        raise ValueError("stochastic checkpoint contains unbound spectrum arrays")
    return restored
