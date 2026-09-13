"""A deterministic EnOI mean update using the existing local transform.

Index zero is the actual background, not a covariance sample. Remaining
indices are independently perturbed states evaluated only at analysis time.
Their covariance is not advertised as a cycled ensemble covariance.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from gpuwm.da.letkf import GriddedObs, analyze


def perturbation_options(*, length_scale_km=30., vertical_scale_levels=3.,
                         rim_width=5, mp_physics=None, **selectors) -> dict:
    """One policy table, consumed by forecast and static covariance draws."""
    common = dict(length_scale_km=float(length_scale_km),
                  vertical_scale_levels=float(vertical_scale_levels))
    result = dict(fields={
        'u': dict(amplitude=1.5, **common),
        'v': dict(amplitude=1.5, **common),
        'theta': dict(amplitude=.5, **common),
        'qv': dict(amplitude=.05, mode='lognormal', **common),
    }, rim_width=int(rim_width), qv_floor=0., rh_cap=1., fft_host=True)
    if mp_physics is not None:
        from gpuwm.da.moments import scheme_moments
        from gpuwm.da.perturb import SUPPORTED_SPECIES
        scheme = scheme_moments(mp_physics, **selectors)
        result['species'] = {name: dict(amplitude=.4, **common,
                                        threshold_kg_kg=scheme.q_threshold)
                             for name in scheme.mass_fields if name in SUPPORTED_SPECIES}
    return result


def static_analysis(prior, observations, grid, config, diagnostics=None, *,
                    solve_namespace=None, progress=None):
    """Return a mean increment at index 0 and zeros for the static states.

    Cross covariance uses centred sample anomalies. Innovations use the
    observation minus H of the actual background, including for nonlinear
    H. Relaxation of a disposable posterior sample is unnecessary and is
    disabled; no posterior sample is carried into the next forecast.
    """
    first = next(iter(prior.values()))
    if int(first.shape[0]) < 3:
        raise ValueError('Static analysis needs a background and at least two covariance samples; supply additional analysis states.')
    if isinstance(first, np.ndarray):
        xp = np
    else:
        import cupy as xp
    count = int(first.shape[0])
    sample_prior = {}
    for name, values in prior.items():
        if values.shape[0] != count or not bool(xp.all(xp.isfinite(values))):
            raise ValueError(f'Static prior {name} has inconsistent members or nonfinite values; repair the input states.')
        sample_prior[name] = values[1:]
    batches = []
    any_innovation = False
    for batch in observations:
        sim = xp.asarray(batch.simulated)
        if sim.shape[0] != count:
            raise ValueError(f'Observation {batch.name} does not match the static state count; reevaluate its operator.')
        mask = xp.asarray(batch.mask, dtype=bool)
        values = xp.asarray(batch.values)
        if bool(xp.any(mask & (~xp.isfinite(sim[0]) | ~xp.isfinite(values)))):
            raise ValueError(f'Observation {batch.name} has a nonfinite background innovation; correct or mask that observation.')
        any_innovation = any_innovation or bool(xp.any(mask & (values != sim[0])))
        samples = sim[1:]
        centred_h = samples - xp.mean(samples, axis=0, keepdims=True) + sim[0]
        batches.append(GriddedObs(name=batch.name, values=values, errors=batch.errors,
                                  simulated=centred_h, mask=mask,
                                  localization=batch.localization, window=batch.window))
    # Run the owner even for a null innovation: shapes and observation
    # errors still need validation, and its diagnostics count the data.
    increments = analyze(sample_prior, batches, grid,
                         replace(config, rtps_alpha=0., prior_inflation=1.), diagnostics,
                         solve_namespace=solve_namespace, progress=progress)
    answer = {}
    for name, values in prior.items():
        delta = xp.zeros_like(values, dtype=xp.float64)
        if any_innovation:
            delta[0] = xp.mean(increments[name], axis=0)
        answer[name] = delta
    return answer


static_analysis.supports_host_staging = True


def covariance_states(background: dict, setup: dict, run_cfg, *, samples: int,
                      seed: int, options: dict) -> tuple[list[dict], dict]:
    """Build CPU analysis samples and refresh their EOS through its owner.

    Forecast stepping is deliberately absent. Setup arrays are shared and
    immutable; prognostics and diagnosed arrays are independent copies.
    """
    from types import SimpleNamespace
    from gpuwm.da.perturb import PerturbationConfig, apply_perturbations
    from gpuwm.core.diagnostics import update_diagnostics
    if type(samples) is not int or samples < 2:
        raise ValueError('Static covariance needs at least two samples; increase the analysis sample count.')
    cfg = PerturbationConfig.from_mapping(dict(dx_km=run_cfg.dx / 1000.,
                                               dy_km=run_cfg.dy / 1000., **options))
    states = [background]
    receipts = []
    for sample in range(samples):
        fields = {name: np.array(value, copy=True) for name, value in background.items()}
        state = SimpleNamespace(**{**setup, **fields})
        report = apply_perturbations(state, seed + sample, cfg)
        update_diagnostics(state, run_cfg.hypsometric_opt)
        for name in ('p', 'al', 'alt'):
            if not np.all(np.isfinite(fields[name])):
                raise ValueError(f'Static sample {sample} produced nonfinite {name}; reduce covariance amplitudes or correct the background.')
        states.append(fields)
        receipts.append(report)
    return states, dict(method='static-covariance-oi', forecast_trajectories=1,
                        covariance_samples=samples, seed=seed,
                        eos='gpuwm.core.diagnostics.update_diagnostics', draws=receipts,
                        posterior_samples_carried=False,
                        limitation='prescribed perturbations are not flow-dependent forecast covariance')
