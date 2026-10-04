"""Whole-domain stochastic updates with bounded tile and rank windows.

The original spectral owner advances once per domain step. Only two-dimensional
patterns cross the source-card fence. Each window transforms its actual local
nonmicrophysics rates, and the full owner commits after every window succeeds.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from operator import index
import threading
import numpy as np

from gpuwm.ensemble.batch_storage import BatchArraySpec, BatchMemoryPlan

_VARIANTS = {"u": "u", "v": "v", "theta": "mass", "qv": "mass"}


def stochastic_window_indices(spec, variant):
    """Exact full-pattern coordinates under the original periodic face law.

    Periodic closing momentum faces are aliases of face zero, including their
    held stochastic rates. A nonperiodic closing face keeps its physical extra
    edge. Mass coordinates wrap only on their own mass-domain extent.
    """
    if variant not in ("mass", "u", "v"):
        raise ValueError("stochastic windows need a mass/u/v component stagger")
    extra_y, extra_x = int(variant == "v"), int(variant == "u")
    def axis(origin, count, size, extra, periodic):
        logical = np.arange(index(origin), index(origin) + index(count) + extra, dtype=np.int64)
        if periodic:
            return logical % size
        if logical.size and (logical.min() < 0 or logical.max() >= size + extra):
            raise ValueError("nonperiodic stochastic window leaves its original component grid")
        return logical
    return (axis(spec.cj0, spec.cny, spec.ny, extra_y, spec.periodic_y),
            axis(spec.ci0, spec.cnx, spec.nx, extra_x, spec.periodic_x))


def window_memory_plan(shape, *, nz, sppt=False, skebs=False, spp_levels=()):
    """Named window buffers and the original rate-transformation call peak."""
    ny, nx = map(index, shape)
    shapes = {"u": (ny, nx + 1), "v": (ny + 1, nx), "mass": (ny, nx)}
    arrays = []
    if sppt:
        arrays += [BatchArraySpec("stochastic:sppt:" + variant, horizontal, "shared")
                   for variant, horizontal in shapes.items()]
    if skebs:
        arrays += [BatchArraySpec("stochastic:skebs:" + name, shapes[_VARIANTS[name]], "shared")
                   for name in ("u", "v", "theta")]
    for name in spp_levels:
        arrays.append(BatchArraySpec("stochastic:spp:" + name, shapes["mass"], "shared"))
    if sppt or skebs:
        for name in ("u", "v", "theta", "qv"):
            arrays.append(BatchArraySpec("stochastic:result:" + name, (nz,) + shapes[_VARIANTS[name]], "shared"))
    if skebs:
        for name in ("u", "v", "theta"):
            arrays.append(BatchArraySpec("stochastic:mass_factor:" + name, (nz,) + shapes[_VARIANTS[name]], "shared"))
        widest = max(shapes.values(), key=lambda value: value[0] * value[1])
        arrays.append(BatchArraySpec("stochastic:forcing_product_peak", (nz,) + widest, "shared"))
        if sppt:
            arrays += [BatchArraySpec("stochastic:skebs_result_before_sppt:" + name,
                                      (nz,) + shapes[_VARIANTS[name]], "shared") for name in ("u", "v", "theta")]
    if not arrays:
        return None
    return BatchMemoryPlan(tuple(arrays), reserved_bytes=0)


def _host_surface(value):
    if isinstance(value, np.ndarray):
        result = value.copy()
    else:
        result = value.get()
    if result.ndim != 2 or result.dtype != np.dtype("float32"):
        raise ValueError("stochastic transport carries original float32 two-dimensional patterns")
    result.flags.writeable = False
    return result


class StochasticSweepLease:
    def __init__(self, binding, *, array_module=None):
        self.binding, self.hook = binding, binding.hook
        self.xp = array_module
        self.pending = None
        self.expected = 0
        self.completed = set()
        self.host = {}
        self.buffers = {}
        self._lock = threading.RLock()

    @property
    def enabled(self):
        return self.binding.enabled

    def begin(self, cfg, *, windows):
        if not self.enabled:
            return
        if self.pending is not None:
            raise RuntimeError("previous stochastic domain sweep has not completed")
        if self.xp is None:
            import cupy as cp
            self.xp = cp
        processes = ([self.hook.sppt] if self.hook.sppt is not None else [])
        if self.hook.skebs is not None:
            processes += [self.hook.skebs.psi, self.hook.skebs.theta]
        processes += list(self.hook.spp.values())
        devices = {int(process.device) for process in processes}
        if len(devices) > 1:
            raise ValueError("one stochastic spectral owner cannot span different CUDA source cards")
        cuda = getattr(self.xp, "cuda", None)
        scope = nullcontext() if cuda is None or not devices else cuda.Device(next(iter(devices)))
        self.expected = index(windows)
        if self.expected < 1:
            raise ValueError("stochastic sweep must own at least one actual compute window")
        with scope:
            step = self.binding.next_update_index()
            self.hook.set_time_step(float(cfg.dt))
            self.hook.before_timestep(step)
            self.pending, self.completed, self.host = step, set(), {}
            if self.hook._pattern is not None:
                self.host["sppt"] = _host_surface(self.hook._pattern)
            if self.hook._forcing is not None:
                self.host.update({"skebs:" + name: _host_surface(value)
                                  for name, value in self.hook._forcing.items()})
            self.host.update({"spp:" + name: _host_surface(value[0])
                              for name, value in self.hook.parameter_patterns.items()})

    def _window(self, name, host, spec, variant, buffer_id, stream):
        ys, xs = stochastic_window_indices(spec, variant)
        selected = np.ascontiguousarray(host[np.ix_(ys, xs)])
        cuda = getattr(self.xp, "cuda", None)
        device = 0 if cuda is None else int(cuda.runtime.getDevice())
        key = (device, buffer_id, name, variant, selected.shape)
        with self._lock:
            if key not in self.buffers:
                self.buffers[key] = self.xp.empty(selected.shape, dtype=np.float32)
            result = self.buffers[key]
        if isinstance(result, np.ndarray):
            np.copyto(result, selected, casting="no")
        else:
            # The pageable two-dimensional upload finishes before the local
            # rate transform. It never leaves a borrowed temporary in flight.
            result.set(selected, stream=stream)
            if stream is not None:
                stream.synchronize()
        return result

    def bind_window(self, state, cfg, spec, window_id, *, stream=None):
        if not self.enabled:
            return
        with self._lock:
            if self.pending is None or not 0 <= index(window_id) < self.expected:
                raise ValueError("stochastic window is outside this pending full-domain sweep")
        buffer_id = id(state)
        patterns = None
        if "sppt" in self.host:
            by_variant = {variant: self._window("sppt", self.host["sppt"], spec, variant, buffer_id, stream)
                          for variant in ("mass", "u", "v")}
            patterns = {name: by_variant[variant] for name, variant in _VARIANTS.items()}
        forcing = ({name: self._window("skebs:" + name, self.host["skebs:" + name], spec,
                    _VARIANTS[name], buffer_id, stream) for name in ("u", "v", "theta")}
                   if "skebs:u" in self.host else None)
        spp = {name: self.xp.broadcast_to(self._window("spp:" + name, self.host["spp:" + name],
            spec, "mass", buffer_id, stream), (levels, cfg.ny, cfg.nx))
            for name, levels in self.hook.spp_levels.items()}
        state._ensemble_stochastic = StochasticWindowBinding(self, window_id, patterns, forcing, spp)

    def finish(self):
        if not self.enabled:
            return
        with self._lock:
            if self.pending is None or self.completed != set(range(self.expected)):
                raise RuntimeError("stochastic domain sweep lacks complete window application; no pattern commit")
            self.hook.complete_timestep()
            self.binding.applied_steps += 1
            self.pending, self.host = None, {}

    def receipt(self):
        return {"contract": "gpuwm-ensemble-stochastic-sweep-lease.v1",
                "member_id": self.binding.member_id, "pending_step": self.pending,
                "expected_windows": self.expected, "completed_windows": sorted(self.completed),
                "pattern_host_bytes": sum(value.nbytes for value in self.host.values()),
                "window_device_bytes": sum(value.nbytes for value in self.buffers.values()),
                "full_domain_3d_rate_banks": 0,
                "checkpoint_owner": "original full-domain stochastic binding"}


class StochasticWindowBinding:
    def __init__(self, lease, window_id, patterns, forcing, spp):
        self.lease, self.window_id = lease, index(window_id)
        self.patterns, self.forcing, self.spp = patterns, forcing, spp
        self._begun = False

    @property
    def enabled(self):
        return self.lease.enabled

    def before_physics(self, state, cfg):
        if not self.enabled:
            return
        if self._begun:
            raise RuntimeError("one stochastic compute window reached first-RK physics twice")
        self._begun = True
        if self.spp:
            bind = getattr(getattr(state, "physics", None), "bind_spp_patterns", None)
            if bind is None:
                raise RuntimeError("stochastic SPP window needs the actual selected parameter consumers")
            bind(self.spp)

    def after_nonmicrophysics(self, state, cfg, tendencies):
        if not self.enabled:
            return tendencies
        if not self._begun:
            raise RuntimeError("stochastic compute window has not reached first-RK physics")
        if tendencies is None:
            from gpuwm.core.physics import PhysicsTendencies
            tendencies = PhysicsTendencies.zeros(state)
        rates = {"u": tendencies.ru, "v": tendencies.rv, "theta": tendencies.rtheta, "qv": tendencies.rqv}
        if any(value is None for value in rates.values()):
            raise RuntimeError("stochastic window needs every original nonmicrophysics component")
        factors = None
        if self.forcing is not None:
            from gpuwm.ensemble.stochastic_execution import coupled_mass_factors
            provider = self.lease.binding.mass_factor_provider or coupled_mass_factors
            factors = provider(state, cfg)
        with self.lease._lock:
            if self.window_id in self.lease.completed:
                raise RuntimeError("stochastic compute window was already applied")
        result = self.lease.hook.transform_nonmicrophysics(rates, tendency_scope="nonmicrophysics",
            mass_factors=factors, sppt_patterns=self.patterns, skebs_forcing=self.forcing)
        with self.lease._lock:
            self.lease.completed.add(self.window_id)
        return replace(tendencies, ru=result["u"], rv=result["v"], rtheta=result["theta"], rqv=result["qv"])


def attach_stochastic_sweep_lease(run, binding):
    """Late attachment keeps the disabled streaming path completely inert."""
    if binding is None or not binding.enabled:
        return
    lease = getattr(run, "_ensemble_stochastic_lease", None)
    if lease is None:
        run._ensemble_stochastic_lease = StochasticSweepLease(binding)
    elif lease.binding is not binding:
        raise ValueError("streamed stochastic lease belongs to another original member binding")


__all__ = ["StochasticSweepLease", "StochasticWindowBinding", "stochastic_window_indices",
           "window_memory_plan", "attach_stochastic_sweep_lease"]
