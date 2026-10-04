"""Member-local original physics and admitted packed tendency handoff.

One process owns the GPU context. Original operations execute on resident
member views, in roster order. No reduction, timestep minimum or CPU weather
field conversion occurs here. Native replacement is an explicit admitted
callback, so a missing packed binding keeps the ordinary operation intact.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Mapping

from gpuwm.ensemble.suite_capabilities import PHYSICS_EXECUTION_CONTRACT, plan_suite

TENDENCY_FIELDS = ("ru", "rv", "rtheta", "rqv", "rqc", "rqr", "rqi", "rqs", "rw")
_PIECE_SELECTORS = {
    "microphysics": ("mp_physics",),
    "radiation": ("ra_physics", "ra_lw_physics", "ra_sw_physics", "ra_rrtmg_variant"),
    "surface_layer": ("sf_sfclay_physics",),
    "land_surface": ("sf_surface_physics", "sf_surface_mosaic", "sf_urban_physics"),
    "pbl": ("bl_pbl_physics", "topo_wind", "gwd_opt"),
    "cumulus": ("cu_physics",),
}


@dataclass
class MemberPhysicsBinding:
    """An original member state, driver and live configuration authority."""
    state: object
    cfg: object
    driver: object | None
    member_id: int
    config_provider: Callable[[], object] | None = None

    @property
    def current_config(self):
        return self.cfg if self.config_provider is None else self.config_provider()


def _clock_key(binding):
    cfg, state, driver = binding.current_config, binding.state, binding.driver
    # Preserve the ordinary Python/Fraction clock values. FP32 rounding or a
    # minimum across members would change the operation's cadence and dt.
    return tuple((type(value), value) for value in (
        getattr(cfg, "dt", None), getattr(state, "elapsed_seconds", None),
        getattr(state, "domain_start_offset", 0.0),
        getattr(driver, "radiation_due_override", None),
        getattr(driver, "surface_pbl_due_override", None),
        getattr(driver, "cumulus_due_override", None),
        getattr(driver, "radt_minutes", None), getattr(driver, "cudt_minutes", None),
        getattr(driver, "bldt_seconds", None)))


def compatible_piece_groups(bindings, component):
    """Return roster indices with identical entry clocks and selector values.

    These groups are candidates, not a proof that a packed launch is valid.
    A qualified callback must additionally verify its parameters, ownership,
    shapes and dependencies. Diverging adaptive clocks remain separate.
    """
    if component not in _PIECE_SELECTORS:
        raise ValueError(f"unknown physics component {component!r}")
    groups = {}
    for index, binding in enumerate(bindings):
        cfg = binding.current_config
        selectors = tuple((name, getattr(cfg, name, None)) for name in _PIECE_SELECTORS[component])
        key = (_clock_key(binding), selectors)
        groups.setdefault(key, []).append(index)
    return tuple(tuple(indices) for indices in groups.values())


class AdmittedTendencyTarget:
    """Word-copy member tendencies into preallocated outer or column banks.

    The caller owns and prices every target. Optional categories and extra
    scalars must agree across this roster; differing suites can use separate
    targets or retain tuple delivery. Copies have no arithmetic and cannot
    mix members. N=1 requires no target and uses the original object.
    """
    def __init__(self, arrays: Mapping[str, object], *, members: int,
                 layout="outermost", extra_scalars=None, array_module=None):
        if isinstance(members, bool) or not isinstance(members, int) or members < 1:
            raise ValueError("members must be a positive integer")
        if layout not in ("outermost", "column"):
            raise ValueError("tendency layout must be outermost or column")
        if set(arrays) - set(TENDENCY_FIELDS):
            raise ValueError("admitted tendency target contains an unknown component")
        if array_module is None:
            import cupy as array_module
        self.arrays, self.members, self.layout = dict(arrays), members, layout
        self.extra_scalars = {} if extra_scalars is None else dict(extra_scalars)
        self.xp = array_module
        self.result = SimpleNamespace(**{name: self.arrays.get(name) for name in TENDENCY_FIELDS},
                                      extra_scalars=self.extra_scalars)
        for name, array in tuple(self.arrays.items()) + tuple(self.extra_scalars.items()):
            if not isinstance(array, self.xp.ndarray) or not array.flags.c_contiguous:
                raise ValueError(f"tendency target {name!r} must be a contiguous device array")
            if array.ndim != (4 if layout == "outermost" else 3):
                raise ValueError(f"tendency target {name!r} has the wrong layout rank")
            if layout == "outermost" and array.shape[0] != members:
                raise ValueError(f"tendency target {name!r} has the wrong roster size")
            if layout == "column" and array.shape[1] % members:
                raise ValueError(f"tendency target {name!r} cannot be split into member slabs")

    def _view(self, target, member):
        if self.layout == "outermost":
            return target[member]
        nz, rows, nx = target.shape
        return target.reshape(nz, self.members, rows // self.members, nx)[:, member]

    def __call__(self, tendencies):
        values = tuple(tendencies)
        if len(values) != self.members:
            raise ValueError("tendency source roster differs from its admitted target")
        copies = []
        # Validate the complete handoff before writing any target. A missing
        # category must not leave stale forcing that looks like a fresh step.
        for member, tendency in enumerate(values):
            for name in TENDENCY_FIELDS:
                source = None if tendency is None else getattr(tendency, name, None)
                target = self.arrays.get(name)
                if (source is None) != (target is None):
                    raise ValueError(f"member {member} tendency {name!r} differs from its admitted categories")
                if source is not None:
                    copies.append((f"member {member} {name}", self._view(target, member), source))
            extra = {} if tendency is None else (getattr(tendency, "extra_scalars", None) or {})
            if extra.keys() != self.extra_scalars.keys():
                raise ValueError(f"member {member} extra scalar tendencies differ from the admitted categories")
            for name, source in extra.items():
                copies.append((f"member {member} scalar {name}", self._view(self.extra_scalars[name], member), source))
        for name, target, source in copies:
            if not isinstance(source, self.xp.ndarray) or source.shape != target.shape or source.dtype != target.dtype:
                raise ValueError(f"{name} must have the admitted device shape and dtype")
            source_device, target_device = getattr(source, "device", None), getattr(target, "device", None)
            if source_device != target_device:
                raise ValueError(f"{name} crosses GPU devices without an admitted transfer")
        for _name, target, source in copies:
            self.xp.copyto(target, source, casting="no")
        return self.result


class MemberPhysicsExecutor:
    """Exact original member drivers, with optional admitted native pieces.

    A full native adapter is selected by the caller only after its suite,
    state and memory admission. All other operations call the original
    functions directly. In particular, no exception after mutation triggers
    a retry on another execution route.
    """
    def __init__(self, bindings, *, microphysics_apply=None, tendency_target=None):
        self.bindings = tuple(bindings)
        if not self.bindings:
            raise ValueError("physics executor needs at least one member")
        if any(not isinstance(binding, MemberPhysicsBinding) for binding in self.bindings):
            raise TypeError("physics executor needs MemberPhysicsBinding entries")
        ids = tuple(binding.member_id for binding in self.bindings)
        if len(set(ids)) != len(ids):
            raise ValueError("physics executor member ids must be unique")
        for binding in self.bindings:
            driver = binding.driver
            if driver is not None and getattr(driver, "state", binding.state) is not binding.state:
                raise ValueError("an original physics driver must own its member state")
        if len({id(binding.state) for binding in self.bindings}) != len(self.bindings):
            raise ValueError("mutable physics states cannot be shared between members")
        drivers = [binding.driver for binding in self.bindings if binding.driver is not None]
        if len({id(driver) for driver in drivers}) != len(drivers):
            raise ValueError("mutable physics drivers cannot be shared between members")
        self.microphysics_apply = microphysics_apply
        self.tendency_target = tendency_target
        self.call_counts = {"compute": [0] * len(self.bindings), "microphysics": [0] * len(self.bindings),
                            "finish_step": [0] * len(self.bindings)}

    def compute_member(self, index):
        from gpuwm.core.physics_inventory import physics_enabled
        binding = self.bindings[index]
        cfg = binding.current_config
        enabled = physics_enabled(cfg)
        if not enabled and getattr(binding.driver, "cam_ozone", None) is None:
            return None
        if binding.driver is None:
            raise ValueError("active original member physics has no initialized driver")
        result = binding.driver.compute(binding.state, cfg)
        self.call_counts["compute"][index] += 1
        # The ordinary dycore calls an ozone-only driver's cadence but does
        # not attach its otherwise zero held tendencies to the RK stage.
        return result if enabled else None

    def compute(self):
        """Return original held tendencies in member roster order."""
        return tuple(self.compute_member(index) for index in range(len(self.bindings)))

    def compute_packed(self):
        """Use the admitted GPU word-copy target after original calls."""
        if self.tendency_target is None:
            if len(self.bindings) == 1:
                return self.compute_member(0)
            raise ValueError("packed physics needs an admitted tendency target")
        return self.tendency_target(self.compute())

    def apply_microphysics_member(self, index, *, refl_10cm_due=False):
        binding = self.bindings[index]
        cfg = binding.current_config
        # The original dycore has this predicate before calling apply. It
        # never requests a microphysics diagnostic for an off scheme here.
        if not cfg.mp_physics:
            return None
        entry = self.microphysics_apply
        if entry is None:
            from gpuwm.core.microphysics import apply
            entry = apply
        result = entry(binding.state, cfg, cfg.dt, refl_10cm_due=refl_10cm_due)
        if binding.driver is not None:
            binding.driver.accept_microphysics(result, dt=cfg.dt)
        self.call_counts["microphysics"][index] += 1
        return result

    def apply_microphysics(self, *, refl_10cm_due=False):
        due = (refl_10cm_due,) * len(self.bindings) if isinstance(refl_10cm_due, bool) else tuple(refl_10cm_due)
        if len(due) != len(self.bindings):
            raise ValueError("microphysics diagnostic schedule differs from the member roster")
        return tuple(self.apply_microphysics_member(index, refl_10cm_due=value)
                     for index, value in enumerate(due))

    def finish_step_member(self, index):
        driver = self.bindings[index].driver
        if driver is not None:
            driver.finish_step()
            self.call_counts["finish_step"][index] += 1

    def finish_step(self):
        for index in range(len(self.bindings)):
            self.finish_step_member(index)

    def run_piece(self, component, original, *, admitted_batch=None, batch_admission=None):
        """Run an original piece or an explicit qualified group callback.

        ``original(binding)`` invokes the ordinary operation. An admitted
        ``batch(indices, bindings)`` returns one result per index, after
        validating this group's complete leaf contract. Passing no batch
        callback is the automatic fallback, including groups of one. Optional
        ``batch_admission(indices, bindings)`` decides eligibility before any
        mutation; false runs the original piece. Runtime failures propagate.
        """
        if admitted_batch is None:
            # Preserve the ordinary roster launch order even if entry clocks
            # happen to form interleaved compatibility groups.
            if component not in _PIECE_SELECTORS:
                raise ValueError(f"unknown physics component {component!r}")
            return tuple(original(binding) for binding in self.bindings)
        results = [None] * len(self.bindings)
        groups = compatible_piece_groups(self.bindings, component)
        ready = {}
        for indices in groups:
            bindings = tuple(self.bindings[index] for index in indices)
            if len(indices) > 1 and (batch_admission is None or batch_admission(indices, bindings)):
                ready[indices[0]] = (indices, bindings)
        batched_indices = {index for indices, _bindings in ready.values() for index in indices}
        for index, binding in enumerate(self.bindings):
            if index in ready:
                indices, bindings = ready[index]
                group_results = tuple(admitted_batch(indices, bindings))
                if len(group_results) != len(indices):
                    raise ValueError("admitted physics piece returned a different member roster")
                for member_index, result in zip(indices, group_results):
                    results[member_index] = result
            elif index not in batched_indices:
                results[index] = original(binding)
        return tuple(results)

    def receipt(self):
        return {
            "contract": PHYSICS_EXECUTION_CONTRACT,
            "execution": "original_member_operations_one_context",
            "member_ids": [binding.member_id for binding in self.bindings],
            "clock_policy": "independent_original_member_clocks",
            "suite_plans": [plan_suite(binding.current_config, members=1).receipt() for binding in self.bindings],
            "call_counts": {name: list(counts) for name, counts in self.call_counts.items()},
            "packed_tendency_target": self.tendency_target is not None,
        }


__all__ = ["TENDENCY_FIELDS", "MemberPhysicsBinding", "MemberPhysicsExecutor",
           "AdmittedTendencyTarget", "compatible_piece_groups"]
