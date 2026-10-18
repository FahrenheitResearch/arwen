"""Read-only final-RK scalar accounting for the non-strict PD route.

SPDX-License-Identifier: Apache-2.0
The caller provides a GPU-only reduction of stacked fields. Hooks are process
local, not thread safe. No production module imports this research observer.
"""
import contextlib
import hashlib
from pathlib import Path

FIELDS = ('qv', 'qc', 'qi', 'qr', 'qs', 'qg')
TERMS = ('fixed_mixing', 'physics', 'lateral_source', 'fold_roundoff',
         'mass_change', 'transport', 'update_roundoff', 'positive_clamp',
         'final_scalar_delta')
MIXING_TERMS = ('mixing_package', 'sixth_order_realized', 'unattributed_fixed')
TRANSPORT_TERMS = ('x_flux_divergence', 'y_flux_divergence',
                   'eta_flux_divergence', 'post_pd_remainder')
MASS_TERMS = ('fixed_mixing', 'physics', 'lateral_source', 'fold_roundoff',
              'transport', 'update_roundoff', 'positive_clamp',
              'coupled_scalar_delta')


def transport_kernel():
    """Shadow the installed limiter using its inputs; never write model state.

    Directional terms retain the producer's FP32 face arithmetic. Their
    recombination rounding and subsequent implicit/boundary work are separate.
    This split does not measure truncation error or isolate numerical diffusion.
    """
    import cupy as cp
    from gpuwm.core.kernels import module_source, module_options
    from gpuwm import wrf_exact
    if wrf_exact.ENABLED or wrf_exact.ADVECTION_ENABLED:
        raise ValueError('transport observer does not instrument strict WRF')
    source = module_source('pd_advection')
    digest = hashlib.sha256(source.encode()).hexdigest()
    anchor = '#undef PD_SCALE\n#if GPUWM_WRF_EXACT_C_ADVECTION'
    signature = 'real* __restrict__ tend_out,      // (nz, ny, nx) +='
    if source.count(anchor) != 1 or source.count(signature) != 1:
        raise ValueError('PD transport source anchor changed')
    # The audited apply kernel is the final function in this source.
    if not source.rstrip().endswith('tend_out[IDX3(k, j, i)] += t;\n#endif\n}'):
        raise ValueError('PD transport source tail changed')
    source = source.split(anchor)[0] + r'''
#undef PD_SCALE
    size_t id=IDX3(k,j,i), n=(size_t)nz*ny*nx;
    real m=has_msf?msft[(size_t)j*nx+i]:1.0f;
    real tx=0.0f, ty=0.0f;
    if (!open_x || (i>=1 && i<=nx-2))
        tx=-m*dx_inv*((sx_r*fxc_r+fxl[I3(k,j,i+1,ny,nx+1)])
                     -(sx_l*fxc_l+fxl[I3(k,j,i,ny,nx+1)]));
    if (!open_y || (j>=1 && j<=ny-2))
        ty=-m*dy_inv*((sy_r*fyc_r+fyl[I3(k,j+1,i,ny+1,nx)])
                     -(sy_l*fyc_l+fyl[I3(k,j,i,ny+1,nx)]));
    real tz=-rdnw[k]*((sz_t*fzc_t+fzl[IDX3(k+1,j,i)])
                    -(sz_b*fzc_b+fzl[IDX3(k,j,i)]));
    parts[id]=(double)tx; parts[n+id]=(double)ty; parts[2*n+id]=(double)tz;
}
'''
    source = source.replace(signature, 'double* __restrict__ parts,')
    options = module_options('pd_advection')
    module = cp.RawModule(code=source, options=options)
    # The manifest records every image this process compiles
    # (tests/test_kernel_manifest.py audits each RawModule site); this
    # audit unit is the shipped PD transport source with its apply
    # kernel rewritten to emit the three flux parts, under its own key.
    from gpuwm.certify.kernel_manifest import record_module
    record_module('pd_advection_process_budget', source=source,
                  options=options, module=module)
    return module.get_function('pd_renorm_apply'), digest


class ScalarProcessLedger:
    """Algebraic accounting, not an additive causal counterfactual.

    For old/new coupled layer masses C0/C and source-folded scalar qf,
    qnew-qold = dt*(F+P+B)/C + fold_error*C0/C
                 + qold*(C0/C-1) + dt*A/C + update_error + clamp.
    A already includes the source-dependent PD limiter and any implicit solve.
    Only the final RK stage advances the full-step budget. No RK-stage sum.
    """
    def __init__(self, state, reduce, *, transport=False, mass_reduce=None):
        from gpuwm.core import moist
        from gpuwm.wrf_exact import ENABLED
        if ENABLED:
            raise ValueError('research scalar ledger does not instrument strict WRF')
        self.state, self.reduce = state, reduce
        # Optional caller reduction accepts pressure-coupled scalar changes.
        # Weight with -dnw/G and physical cell areas; never rho*dz again.
        self.mass_reduce = mass_reduce
        self.mass_values = {}
        self.mass_transport_values = {}
        self.mass_mixing_values = {}
        self.values = {}
        self.calls = {name: 0 for name in FIELDS}
        self.active = False
        self.pending = None
        self.mixing_values = {}
        self.mixing_calls = {'prepare': 0, 'mixing_package': 0,
                             'sixth_order': {name: 0 for name in FIELDS}}
        self.mixing_config = None
        self.mixing_raw = {}
        self.preparing = False
        self.transport_values = {}
        self.transport_calls = {name: 0 for name in FIELDS}
        self.transport_raw = None
        self.transport_kernel = None
        self.transport_source_sha256 = None
        if transport:
            self.transport_kernel, self.transport_source_sha256 = transport_kernel()
        self.source_sha256 = hashlib.sha256(Path(moist.__file__).read_bytes()).hexdigest()

    @contextlib.contextmanager
    def observe(self):
        import cupy as cp
        from gpuwm.core import dycore, moist
        old_stage = dycore.advance_scalars_stage
        old_fold = moist._pd_fold_sources
        old_update = moist._update_scalar_in_place
        old_prepare = dycore.prepare_fixed_tendencies
        old_mix = dycore._compute_wrf_smag_tendencies
        old_diff6 = dycore.add_diff6_row
        old_kernel = moist.get_kernel

        def select_kernel(module, name, *args, **kw):
            original = old_kernel(module, name, *args, **kw)
            if (module, name) != ('pd_advection', 'pd_renorm_apply'):
                return original
            def launch(grid, block, arguments):
                original(grid, block, arguments)
                if self.active and self.pending is not None:
                    if self.transport_raw is None:
                        self.transport_raw = cp.empty((3,)+arguments[0].shape, cp.float64)
                    shadow_args = arguments[:15]+(self.transport_raw,)+arguments[16:]
                    self.transport_kernel(grid, block, shadow_args)
                    self.transport_calls[self.pending[0]] += 1
            return launch

        def prepare(state, cfg, *args, **kw):
            if state is not self.state:
                return old_prepare(state, cfg, *args, **kw)
            self.mixing_raw = {}
            self.mixing_config = {key: getattr(cfg, key) for key in
                                  ('km_opt', 'diff_opt', 'diff_6th_opt',
                                   'diff_6th_factor', 'diff_6th_slopeopt')}
            self.preparing = True
            try:
                result = old_prepare(state, cfg, *args, **kw)
                self.mixing_calls['prepare'] += 1
                return result
            finally:
                self.preparing = False

        def mixing(state, cfg, km, kh, specs, **kw):
            result = old_mix(state, cfg, km, kh, specs, **kw)
            if self.preparing and state is self.state:
                for name in FIELDS:
                    slot = 'smag_r' + name
                    if any(row[5] == slot for row in specs):
                        self.mixing_raw.setdefault(name, {})['mixing_package'] = state._scratch[slot].copy()
                self.mixing_calls['mixing_package'] += 1
            return result

        def diff6(cfg, slot, f0, target, *args, **kw):
            name = slot[len('smag_r'):] if slot.startswith('smag_r') else ''
            relevant = self.preparing and name in FIELDS
            before = target.copy() if relevant else None
            result = old_diff6(cfg, slot, f0, target, *args, **kw)
            if relevant:
                # Record the realized FP32 addition, including its rounding;
                # recomputing a filter on zero changes WRF's addition order.
                change = target.astype(cp.float64)-before.astype(cp.float64)
                values = self.mixing_raw.setdefault(name, {})
                if 'sixth_order_realized' not in values:
                    values['sixth_order_realized'] = cp.zeros_like(change)
                values['sixth_order_realized'] += change
                self.mixing_calls['sixth_order'][name] += 1
            return result

        def stage(state, cfg, *args, **kw):
            relevant = state is self.state and kw.get('final', False)
            if relevant and (cfg.moist_adv_opt != 1 or cfg.open_x or cfg.open_y):
                raise ValueError('scalar ledger requires the final positive-definite route')
            self.active = relevant
            try:
                return old_stage(state, cfg, *args, **kw)
            finally:
                self.active = False
                if self.pending is not None:
                    self.pending = None
                    raise RuntimeError('source fold was not followed by its scalar update')

        def fold(state, cfg, name, q0, chm0, dt_eff, physics_tendencies,
                 fixed_tendency=None, lbc_held=None):
            result = old_fold(state, cfg, name, q0, chm0, dt_eff,
                              physics_tendencies, fixed_tendency, lbc_held)
            if self.active and name in FIELDS:
                if self.pending is not None:
                    raise RuntimeError('overlapping scalar source folds')
                # A nested boundary source can alias the transport workspace,
                # which the caller zeros immediately after returning.
                self.pending = (name, q0, result, chm0,
                                fixed_tendency,
                                None if physics_tendencies is None else
                                physics_tendencies.scalar_for(name),
                                None if lbc_held is None else lbc_held.copy())
            return result

        def update(q, q0, tend, c1h, c2h, mu0, mu, dt_eff, **kw):
            if not self.active or self.pending is None:
                return old_update(q, q0, tend, c1h, c2h, mu0, mu, dt_eff, **kw)
            name, original, folded, fold_mass, fixed, physics, lateral = self.pending
            self.pending = None
            if q is not getattr(self.state, name) or q0 is not folded:
                raise RuntimeError('scalar update does not match its source fold')
            if not kw.get('clamp') or kw.get('fixed') is not None or kw.get('physics') is not None:
                raise RuntimeError('unexpected sources after positive-definite fold')
            shape = q.shape
            c0 = (c1h[:, None, None]*mu0.reshape(shape[1:]) + c2h[:, None, None])
            c = (c1h[:, None, None]*mu.reshape(shape[1:]) + c2h[:, None, None])
            c64 = c.astype(cp.float64)
            c064 = c0.astype(cp.float64)
            old64 = original.astype(cp.float64)
            folded64 = folded.astype(cp.float64)
            dt = float(cp.float32(dt_eff))
            sources = [cp.zeros_like(old64) if a is None else dt*a.astype(cp.float64)/c64
                       for a in (fixed, physics, lateral)]
            # The fold's actual denominator is separately supplied by the caller.
            # Include any difference between fold_mass and the update C0 here.
            fold_error = ((folded64-old64)*c064/c64 - sum(sources))
            mass = old64*(c064/c64-1.)
            mapped = tend if kw.get('msft') is None else tend*kw['msft'].reshape(shape[1:])
            transport = dt*mapped.astype(cp.float64)/c64
            # Recreate the unfloored FP32 expression solely to measure clipping.
            raw = (c0*folded + cp.float32(dt_eff)*mapped)/c
            clip = cp.maximum(raw, cp.float32(0)).astype(cp.float64)-raw.astype(cp.float64)
            old_update(q, q0, tend, c1h, c2h, mu0, mu, dt_eff, **kw)
            delta = q.astype(cp.float64)-old64
            error = delta-sum(sources)-fold_error-mass-transport-clip
            compact = self.reduce(cp.stack(sources+[fold_error, mass, transport, error, clip, delta]))
            if name not in self.values:
                self.values[name] = cp.zeros_like(compact)
            self.values[name] += compact
            self.calls[name] += 1
            if self.mass_reduce is not None:
                # Actual coupled mass change, independently formed from both
                # endpoints. Column-mass dilution is absent in this norm.
                coupled_delta = c64*q.astype(cp.float64)-c064*old64
                terms = [v*c64 for v in sources+[fold_error, transport, error, clip]]
                mass_compact = self.mass_reduce(cp.stack(terms+[coupled_delta]))
                if name not in self.mass_values:
                    self.mass_values[name] = cp.zeros_like(mass_compact)
                self.mass_values[name] += mass_compact
            if self.transport_kernel is not None:
                if self.transport_raw is None or self.transport_calls[name] != self.calls[name]:
                    raise RuntimeError('missing or duplicate PD transport capture')
                parts = self.transport_raw
                if kw.get('msft') is not None:
                    parts = parts*kw['msft'].reshape((1,)+shape[1:])
                parts = dt*parts/c64
                remainder = transport-cp.sum(parts, axis=0)
                compact_transport = self.reduce(cp.concatenate((parts, remainder[None])))
                if name not in self.transport_values:
                    self.transport_values[name] = cp.zeros_like(compact_transport)
                self.transport_values[name] += compact_transport
                if self.mass_reduce is not None:
                    mass_transport = self.mass_reduce(cp.concatenate((parts, remainder[None]))*c64)
                    if name not in self.mass_transport_values:
                        self.mass_transport_values[name] = cp.zeros_like(mass_transport)
                    self.mass_transport_values[name] += mass_transport
            if self.mixing_config is not None:
                parts = self.mixing_raw.get(name, {})
                mixing_sources = [cp.zeros_like(old64) if key not in parts else
                                  dt*parts[key].astype(cp.float64)/c64
                                  for key in MIXING_TERMS[:2]]
                mixing_sources.append(sources[0]-sum(mixing_sources))
                compact_mix = self.reduce(cp.stack(mixing_sources))
                if name not in self.mixing_values:
                    self.mixing_values[name] = cp.zeros_like(compact_mix)
                self.mixing_values[name] += compact_mix
                if self.mass_reduce is not None:
                    mass_mix = self.mass_reduce(cp.stack(mixing_sources)*c64)
                    if name not in self.mass_mixing_values:
                        self.mass_mixing_values[name] = cp.zeros_like(mass_mix)
                    self.mass_mixing_values[name] += mass_mix

        dycore.advance_scalars_stage = stage
        moist._pd_fold_sources = fold
        moist._update_scalar_in_place = update
        dycore.prepare_fixed_tendencies = prepare
        dycore._compute_wrf_smag_tendencies = mixing
        dycore.add_diff6_row = diff6
        if self.transport_kernel is not None:
            moist.get_kernel = select_kernel
        try:
            yield self
        finally:
            dycore.advance_scalars_stage = old_stage
            moist._pd_fold_sources = old_fold
            moist._update_scalar_in_place = old_update
            dycore.prepare_fixed_tendencies = old_prepare
            dycore._compute_wrf_smag_tendencies = old_mix
            dycore.add_diff6_row = old_diff6
            moist.get_kernel = old_kernel
