"""Member pattern providers bound to original initialized domain models."""
from __future__ import annotations

from dataclasses import replace

from gpuwm.ensemble.stochastic import StochasticConfig, StochasticTimestepHook
from gpuwm.ensemble.stochastic_execution import StochasticPhysicsBinding


class StochasticModelProvider:
    """Reference patterns with explicit controls, independent of packing."""
    def __init__(self, *, sppt=None, skebs_psi=None, skebs_theta=None, spp=False, spp_configs=None,
                 wrf_seed_labels=None):
        from gpuwm.ensemble.stochastic_seeds import normalize_wrf_seed_labels
        self.wrf_seed_labels = normalize_wrf_seed_labels(wrf_seed_labels)
        if (skebs_psi is None) != (skebs_theta is None):
            raise ValueError("SKEBS requires both streamfunction and temperature configuration")
        self.sppt, self.skebs_psi, self.skebs_theta = sppt, skebs_psi, skebs_theta
        self.spp = spp
        if spp_configs is not None:
            if (not isinstance(spp_configs, dict) or set(spp_configs) - {"conv", "pbl", "lsm"}
                    or any(not isinstance(config, StochasticConfig) or config.kind != "spp_" + name
                           for name, config in spp_configs.items())):
                raise ValueError("spp_configs must map conv/pbl/lsm to their matching StochasticConfig")
            spp_configs = dict(spp_configs)
        self.spp_configs = spp_configs

    @classmethod
    def from_mapping(cls, controls):
        if not isinstance(controls, dict) or set(controls) - {"sppt", "skebs", "spp", "spp_configs", "wrf_seed_labels"}:
            raise ValueError("stochastic controls must declare only sppt, skebs, spp, spp_configs and wrf_seed_labels")
        def config(kind, value):
            if value is False or value is None:
                return None
            reference = StochasticConfig.wrf_reference(kind)
            if value is True:
                return reference
            if not isinstance(value, dict) or "kind" in value:
                raise ValueError(f"{kind} requires true, false or explicit reference parameter overrides")
            return replace(reference, **value)
        skebs = controls.get("skebs")
        if isinstance(skebs, dict):
            if set(skebs) - {"psi", "theta"}:
                raise ValueError("SKEBS controls must name psi and theta parameters")
            psi, theta = config("skebs_psi", skebs.get("psi", True)), config("skebs_theta", skebs.get("theta", True))
        else:
            psi, theta = config("skebs_psi", skebs), config("skebs_theta", skebs)
        spp = controls.get("spp", False)
        if isinstance(spp, dict):
            if (set(spp) - {"conv", "pbl", "lsm"}
                    or any(type(value) is not int or value not in (0, 1) for value in spp.values())):
                raise ValueError("SPP must name integer 0/1 conv, pbl and lsm consumers")
        elif type(spp) is not bool:
            raise ValueError("SPP must be true, false or selected consumer switches")
        explicit = controls.get("spp_configs")
        if explicit is not None:
            if not isinstance(explicit, dict) or set(explicit) - {"conv", "pbl", "lsm"}:
                raise ValueError("spp_configs must name conv, pbl and lsm parameter overrides")
            normalized = {}
            for name, value in explicit.items():
                if value is True or value is None:
                    value = {}
                if not isinstance(value, dict) or "kind" in value:
                    raise ValueError("spp_configs values require reference parameter overrides")
                normalized[name] = replace(StochasticConfig.wrf_reference("spp_" + name), **value)
            explicit = normalized
        return cls(sppt=config("sppt", controls.get("sppt")), skebs_psi=psi,
                   skebs_theta=theta, spp=spp, spp_configs=explicit,
                   wrf_seed_labels=controls.get("wrf_seed_labels"))

    def configure_experiment(self, experiment):
        """Bind requested SPP switches before original driver construction."""
        if self.spp is True:
            from gpuwm.config import validate_spp_config
            selected = False
            for domain in experiment.domains:
                validate_spp_config(domain.run)
                selected |= any(getattr(domain.run, f"spp_{name}", 0)
                                for name in ("conv", "pbl", "lsm"))
            if not selected:
                raise ValueError("SPP true requires selected spp_conv, spp_pbl or spp_lsm consumers; "
                                 "declare the consumer switches before model initialization")
            return self._validated_spp_experiment(experiment)
        if not isinstance(self.spp, dict) or not self.spp:
            return self._validated_spp_experiment(experiment)
        from gpuwm.config import validate_spp_config
        domains = []
        for domain in experiment.domains:
            cfg = replace(domain.run, **{f"spp_{name}": enabled for name, enabled in self.spp.items()})
            validate_spp_config(cfg)
            domains.append(replace(domain, run=cfg))
        return self._validated_spp_experiment(replace(experiment, domains=tuple(domains)))

    def _validated_spp_experiment(self, experiment):
        if self.spp_configs is not None:
            from gpuwm.config import validate_spp_config
            selected = {name for domain in experiment.domains for name in ("conv", "pbl", "lsm")
                        if getattr(domain.run, f"spp_{name}", 0)}
            if set(self.spp_configs) != selected:
                raise ValueError("spp_configs must exactly match the enabled SPP scheme union; "
                                 "partial and nonselected parameter maps would silently change amplitudes")
            for domain in experiment.domains:
                validate_spp_config(domain.run)
        return experiment

    def enabled_for_experiment(self, experiment):
        """All-off controls are a pure ordinary configuration and allocate nothing."""
        return bool(self.sppt is not None or self.skebs_psi is not None or any(
            getattr(domain.run, f"spp_{name}", 0) for domain in experiment.domains
            for name in ("conv", "pbl", "lsm")))

    def pattern_shape(self, cfg):
        return (int(cfg.ny) + 1, int(cfg.nx) + 1)

    def memory_components(self, experiment, *, fft_workspace_bytes, include_rates=True,
                          window_shapes=None):
        """Declared source banks and conservative original-operation call peaks.

        SPP repeats a two-dimensional pattern through a broadcast view. No
        three-dimensional parameter-pattern bank is claimed. Full-domain
        rate peaks also bound every legal tile/rank window, independently of
        the ordinary planner's later terrain and adaptive-halo refinement.
        The FFT workspace is supplied by the actual selected card's plan.
        ``window_shapes`` supplies original tile specs and buffer counts after
        the ordinary planner has selected a streamed road.
        """
        if not self.enabled_for_experiment(experiment):
            return ()
        import numpy as np
        from gpuwm.ensemble.admission import MemoryComponent
        from gpuwm.ensemble.batch_storage import BatchArraySpec, BatchMemoryPlan
        from gpuwm.ensemble.stochastic_streaming import window_memory_plan
        components = []
        for domain in experiment.domains:
            cfg = domain.run
            spp = tuple(name for name in ("conv", "pbl", "lsm")
                        if getattr(cfg, f"spp_{name}", 0))
            patterns = (("sppt",) if self.sppt is not None else ()) + (
                ("skebs_psi", "skebs_theta") if self.skebs_psi is not None else ()) + spp
            if not patterns:
                continue
            grid = int(domain.grid_id)
            shape = self.pattern_shape(cfg)
            specs = []
            for name in patterns:
                specs.extend((BatchArraySpec(name + ":spectrum", shape, "shared", np.complex64),
                              BatchArraySpec(name + ":amplitude", shape, "shared")))
            # The normalization expression's possible live temporaries are
            # named individually. Adaptive coefficient replacement may hold
            # the old amplitude beside the new one.
            specs.extend(BatchArraySpec("normalization:" + name, shape, "shared", np.float64)
                         for name in ("log_weights", "gamma_terms", "subtract", "scaled", "exp"))
            specs.extend((BatchArraySpec("amplitude_replacement", shape, "shared"),
                          BatchArraySpec("fft_output", shape, "shared", np.complex64),
                          BatchArraySpec("gridpoint_copy", shape, "shared")))
            if self.skebs_psi is not None:
                specs.append(BatchArraySpec("derivative_spectrum", shape, "shared", np.complex64))
            if self.sppt is not None:
                specs.append(BatchArraySpec("retained_sppt", shape, "shared"))
            if self.skebs_psi is not None:
                specs.extend(BatchArraySpec("retained_skebs:" + name, shape, "shared")
                             for name in ("u", "v", "theta"))
            for name in spp:
                # Previous parameter_patterns survives until the new dict is
                # assigned, so both two-dimensional owners can coexist.
                specs.extend(BatchArraySpec("spp:" + name + ":" + age,
                    (cfg.ny, cfg.nx), "shared") for age in ("previous", "current"))
            plan = BatchMemoryPlan(tuple(specs), reserved_bytes=0)
            components.append(MemoryComponent(f"stochastic_source_d{grid:02d}", "stochastic", plan=plan,
                basis="envelope", evidence="WrfStochasticPattern constructor/gridpoint/adaptive replacement and StochasticTimestepHook retained 2-D owners"))
            def rate_plan(horizontal):
                ny, nx = horizontal
                window = window_memory_plan(horizontal, nz=cfg.nz,
                    sppt=self.sppt is not None, skebs=self.skebs_psi is not None, spp_levels=spp)
                if window is None:
                    return None
                extra = []
                if self.skebs_psi is not None:
                    extra.extend((BatchArraySpec("coupled_mass_input", (cfg.nz, ny, nx), "shared"),
                                  BatchArraySpec("total_mu_input", (ny, nx), "shared")))
                if self.sppt is not None:
                    horizontal_edge = (ny + 1, nx + 1)
                    extra.extend((BatchArraySpec("sppt_factor_peak", horizontal_edge, "shared"),
                                  BatchArraySpec("sppt_absolute_peak", horizontal_edge, "shared"),
                                  BatchArraySpec("sppt_finite_peak", horizontal_edge, "shared", np.bool_),
                                  BatchArraySpec("sppt_bound_peak", horizontal_edge, "shared", np.bool_)))
                return BatchMemoryPlan(window.arrays + tuple(extra), reserved_bytes=0)
            if include_rates:
                windows, buffers = (((cfg.ny, cfg.nx),), 1) if window_shapes is None else window_shapes[grid]
                plans = tuple(rate_plan(horizontal) for horizontal in sorted(set(windows)))
                plans = tuple(plan for plan in plans if plan is not None)
                cached = lambda spec: spec.name.startswith(("stochastic:sppt:", "stochastic:skebs:", "stochastic:spp:"))
                specs = [BatchArraySpec(f"buffer{buffer}:shape{number}:" + spec.name,
                    spec.shape, "shared", spec.dtype) for buffer in range(buffers)
                    for number, plan in enumerate(plans) for spec in plan.arrays if cached(spec)]
                calls = tuple(BatchMemoryPlan(tuple(spec for spec in plan.arrays if not cached(spec)), reserved_bytes=0)
                              for plan in plans if any(not cached(spec) for spec in plan.arrays))
                if calls:
                    largest = max(calls, key=lambda plan: plan.required_bytes(1))
                    specs.extend(BatchArraySpec(f"buffer{buffer}:call_peak:" + spec.name,
                        spec.shape, "shared", spec.dtype) for buffer in range(buffers) for spec in largest.arrays)
                window = BatchMemoryPlan(tuple(specs), reserved_bytes=0)
                components.append(MemoryComponent(f"stochastic_rate_peak_d{grid:02d}", "stochastic", plan=window,
                    basis="envelope", evidence=("original coupled_mass_factors and window_memory_plan; "
                        "retained 2-D shape caches and largest concurrent local call for every original tile buffer" if window_shapes is not None else
                        "original coupled_mass_factors and window_memory_plan, full-domain bound for resident calls")))
        components.append(MemoryComponent("stochastic_fft_workspace", "stochastic",
            fixed_bytes=int(fft_workspace_bytes), basis="measured",
            evidence="sum of exact selected-card cuFFT work areas for distinct domain pattern shapes; no transform or weather-field allocation"))
        return tuple(components)

    def sample_fft_workspace_bytes(self, experiment, *, array_module=None, other_experiments=()):
        """Plan the original C2C transforms in a private, disposable pool.

        This runs only inside an authorized forecast device scope. No model
        array, random stream or transform is touched. Runtime/module backing
        remains visible in the subsequent card sample; owned probe blocks
        are released before that sample. Fail closed if the installed plan
        does not expose its actual work area.
        """
        experiments = (experiment,) + tuple(other_experiments)
        if not any(self.enabled_for_experiment(exp) for exp in experiments):
            return 0
        if array_module is None:
            import cupy as array_module
        from cupy.cuda import cufft
        from cupy.fft._fft import _get_cufft_plan_nd
        shapes = {self.pattern_shape(domain.run) for exp in experiments for domain in exp.domains
                  if self.sppt is not None or self.skebs_psi is not None or any(
                      getattr(domain.run, f"spp_{name}", 0) for name in ("conv", "pbl", "lsm"))}
        pool = array_module.cuda.MemoryPool()
        plans = []
        try:
            with array_module.cuda.using_allocator(pool.malloc):
                if array_module.fft.config.enable_nd_planning:
                    plans.extend(_get_cufft_plan_nd(shape, cufft.CUFFT_C2C,
                                 axes=(-2, -1), order="C", to_cache=False) for shape in sorted(shapes))
                else:
                    keys = {(nx, ny) for ny, nx in shapes} | {(ny, nx) for ny, nx in shapes}
                    plans.extend(cufft.Plan1d(length, cufft.CUFFT_C2C, batch)
                                 for length, batch in sorted(keys))
                return sum(int(plan.work_area.mem.size) for plan in plans)
        finally:
            array_module.cuda.get_current_stream().synchronize()
            plans.clear()
            import gc
            gc.collect()
            pool.free_all_blocks()

    def bind_model(self, *, model, member_id, seed, prepared_member=None):
        """Attach full-domain owners before validating a member restart."""
        for node in model.walk_parent_first():
            if node.state is None or not getattr(node, "_started", True):
                continue
            self.bind_state(state=node.state, cfg=node.cfg.run, clock=node.clock,
                            member_id=member_id, seed=seed, prepared_member=prepared_member)

    def bind_state(self, *, state, cfg, member_id, seed, prepared_member=None, clock=None):
        """The fixed single-domain door supplies its actual state and clock."""
        from gpuwm.config import soil_layer_count
        levels = {name: depth for name, depth in (
            ("conv", 4), ("pbl", cfg.nz), ("lsm", soil_layer_count(cfg)))
            if getattr(cfg, f"spp_{name}", 0)}
        selected_configs = ({name: StochasticConfig.wrf_reference("spp_" + name) for name in levels}
                            if self.spp_configs is None else
                            {name: self.spp_configs[name] for name in levels})
        existing = getattr(state, "_ensemble_stochastic", None)
        if existing is not None:
            recipe = None if prepared_member is None else prepared_member.recipe_sha256
            if existing.member_id != member_id or existing.recipe_sha256 != recipe:
                raise ValueError("domain stochastic owner belongs to another source member or recipe")
            hook = existing.hook
            if getattr(hook, "wrf_seed_labels", None) != self.wrf_seed_labels:
                raise ValueError("reconstructed stochastic WRF seed labels differ from the original owner")
            if hook.pending_step is not None:
                raise ValueError("cannot reconstruct a domain during its pending stochastic timestep")
            if (hook.spp_levels != levels or set(hook.spp) != set(levels)
                    or (hook.sppt is None) != (self.sppt is None)
                    or (hook.skebs is None) != (self.skebs_psi is None)):
                raise ValueError("reconstructed stochastic consumers differ from the original member")
            processes = [(hook.spp[name], selected_configs[name]) for name in levels]
            if hook.sppt is not None:
                processes.append((hook.sppt, self.sppt))
            if hook.skebs is not None:
                processes.extend(((hook.skebs.psi, self.skebs_psi), (hook.skebs.theta, self.skebs_theta)))
            for process, config in processes:
                from gpuwm.ensemble.stochastic_seeds import process_seed
                if (process.seed != process_seed(seed, config.kind, self.wrf_seed_labels) or process.shape != self.pattern_shape(cfg)
                        or process.config != config or process.dx != cfg.dx or process.dy != cfg.dy):
                    raise ValueError("reconstructed stochastic seed, shape, spacing or configuration differs from its spectral owner")
            # Adaptive dt is rebound by the original before_physics call.
            # Reconstruction only attaches the actual new clock object.
            if clock is not None:
                existing.clock = clock
            self._attach_state_owner(state, existing)
            return existing
        if self.sppt is None and self.skebs_psi is None and not levels:
            return None
        explicit = {} if self.spp_configs is None else {"spp_configs": {
            name: self.spp_configs[name] for name in levels}}
        if self.wrf_seed_labels is not None:
            explicit["wrf_seed_labels"] = self.wrf_seed_labels
        hook = StochasticTimestepHook((cfg.ny + 1, cfg.nx + 1), dx=cfg.dx, dy=cfg.dy,
            dt=cfg.dt, member_seed=seed, sppt=self.sppt, skebs_psi=self.skebs_psi,
            skebs_theta=self.skebs_theta, spp=bool(levels), spp_levels=levels, **explicit)
        binding = StochasticPhysicsBinding(hook, member_id=member_id, clock=clock,
            recipe_sha256=None if prepared_member is None else prepared_member.recipe_sha256)
        state._ensemble_stochastic = binding
        self._attach_state_owner(state, binding)
        return binding

    @staticmethod
    def _attach_state_owner(state, binding):
        """Bind a rebuilt resident driver or the new actual streamed run."""
        streamed = getattr(state, "_streamed_domain", None)
        if streamed is not None:
            from gpuwm.ensemble.stochastic_streaming import attach_stochastic_sweep_lease
            attach_stochastic_sweep_lease(streamed._run, binding)
        elif getattr(binding.hook, "parameter_patterns", None):
            driver = getattr(state, "physics", None)
            bind = getattr(driver, "bind_spp_patterns", None)
            if bind is None:
                raise ValueError("rebuilt SPP physics requires its original parameter consumers")
            bind(binding.hook.parameter_patterns)
