"""Fixed-clock real-input member forecast with aggregate-only Rust output.

The input authority and terrain clock are the ordinary prepared WRF door's.
One common scalar restore supplies the initialization words; no independent
member forecast driver is retained or advanced by this executor.
"""
from __future__ import annotations

from datetime import timedelta
from dataclasses import asdict
import math
import time
import numpy as np


class PreparedRealEnsemble:
    def __init__(self, inputs, *, members, member_seeds, member_initializer=None):
        import cupy as cp
        from gpuwm.core.clock import resolve_clock
        from gpuwm.core.device_inventory import state_array_shapes
        from gpuwm.ingest.wrfinput import restore_domain_state
        from gpuwm.ensemble.batch_state import (BatchedDomainState, PreparedHostMember,
                                                 SHARED_STATE_CANDIDATES)
        from gpuwm.ensemble.batch_moist_dycore import workspace_specs, required_scratch_slots, prepare_moist_step
        from gpuwm.ensemble.batch_boundaries import MemberBoundaryTables
        from gpuwm.ensemble.batch_physics_init import initialize_wrfinput_member_physics
        from gpuwm.core.cam_ozone import cam_ozone_setup
        from gpuwm.case_data import trace_gas_overrides_from_config
        from gpuwm.runtime import declared_constant_glw
        if len(inputs.domains) != 1 or members < 2:
            raise ValueError('this retained real executor owns one root domain and at least two members')
        exp = inputs.experiment
        domain, bundle, grid = exp.root, inputs.domains[0], inputs.grids[0]
        cfg = domain.run
        authority = resolve_clock(exp,lbc_interval_s=float(inputs.boundary_interval_seconds))
        clock = authority.domain_clock(domain.grid_id)
        scalar = restore_domain_state(bundle.restored,cfg)
        names = state_array_shapes(cfg)
        arrays = {name: getattr(scalar,name).get() for name in names}
        extras = workspace_specs(cfg)
        arrays.update({spec.name:np.zeros(spec.shape,spec.dtype) for spec in extras})
        controls = {'physics','lateral_boundaries','_scratch','_scratch_arena','_host_setup_state','_phb_host'}
        scalars = {name:value for name,value in vars(scalar).items() if name not in names and name not in controls}
        snapshot = dict(ticks=clock.ticks,step_ticks=clock.step_ticks,tick_den=clock.tick_den,
                        run_ticks=clock.run_ticks,step_count=clock.step_count,
                        dt_fp32=clock.dt_fp32,dtbc_fp32=clock.dtbc_fp32)
        prepared = PreparedHostMember(cfg,arrays,scalars,snapshot,
            scratch={name:value.get() for name,value in scalar._scratch.items()},phb_host=scalar._phb_host)
        del scalar
        cp.cuda.get_current_stream().synchronize()
        free = lambda: int(cp.cuda.runtime.memGetInfo()[0]) - (512 << 20)
        batch = BatchedDomainState.from_prepared((prepared,)*members,array_module=cp,
            available_bytes=free(),shared_fields=tuple(sorted(SHARED_STATE_CANDIDATES & names.keys())),
            extra_specs=extras,scratch_slots=required_scratch_slots(cfg))
        del arrays,prepared
        tables = MemberBoundaryTables.from_prepared((inputs.boundaries,)*members,cfg,
            array_module=cp,available_bytes=free())
        cam = cam_ozone_setup(exp=exp,dc=domain,grid=grid)
        trace = trace_gas_overrides_from_config(inputs.experiment_config,
                            expected_sha256=inputs.authority_sha256['experiment_config'])
        physics = initialize_wrfinput_member_physics(batch,bundle.restored,
            start_time=exp.start_time,landuse=bundle.landuse,cam_ozone=cam,
            column_chunk=exp.column_chunk,trace_gas_overrides=trace,
            constant_glw_wm2=declared_constant_glw(exp),fractional_seaice=bundle.fractional_seaice,
            available_bytes=free())
        initializer_receipt = None
        if member_initializer is not None:
            initializer_receipt = member_initializer(state=batch,cfg=cfg,
                member_indices=tuple(range(members)),seeds=member_seeds,phase='after_physics_before_step')
        self.inputs,self.cfg,self.batch,self.physics,self.tables = inputs,cfg,batch,physics,tables
        self.clock,self.authority = clock,authority
        self.advance = prepare_moist_step(batch,physics_adapter=physics,tables=tables,boundary_clock=clock)
        self.initializer_receipt = initializer_receipt
        from gpuwm.ensemble.batch_health import PreparedBatchStability,PreparedBatchStateHealth
        self.stability = PreparedBatchStability(batch,cfg,boundary_width=cfg.spec_bdy_width,available_bytes=free())
        self.validator = PreparedBatchStateHealth(batch,physics_driver=physics.driver,tables=tables,available_bytes=free())
        self.initial_health = self.validator.require_healthy(phase='initialized')
        self.final_health = None
        self.final_stability = None

    def step(self):
        clock = self.clock
        if clock.at_stop_time:
            raise ValueError('ensemble clock reached its declared stop time')
        if clock.lbc_reset_due():
            clock.mark_force()
        clock.prepare_step()
        self.batch.elapsed_seconds = float(clock.elapsed_seconds_fp32)
        self.advance(refl_10cm_due=clock.history_rings_within_step())
        for member,report in enumerate(self.stability()):
            if report['nan']:
                raise RuntimeError(f'member {member} integration produced a non-finite state at step {clock.step_count+1}')
            if report['cfl'] is not None and not math.isfinite(float(report['cfl'])):
                raise RuntimeError(f'member {member} integration produced a non-finite vertical Courant number at step {clock.step_count+1}')
        clock.advance()
        self.batch.elapsed_seconds = clock.elapsed_seconds
        self.batch.clock['dtbc_fp32'] = clock.dtbc_fp32

    def receipt(self):
        import cupy as cp
        return {'members':self.batch.members,'dt_seconds':self.cfg.dt,
                'time_step_sound':self.cfg.time_step_sound,'steps':self.clock.step_count,
                'state_plan_bytes':self.batch.plan.required_bytes(self.batch.members),
                'boundary_plan_bytes':self.tables.plan.required_bytes(self.batch.members),
                'physics':self.physics.receipt,'member_initializer':self.initializer_receipt,
                'health':{'stability':self.stability.receipt,'full_state':self.validator.receipt,
                          'initial':[asdict(report) for report in self.initial_health],
                          'final':None if self.final_health is None else [asdict(report) for report in self.final_health],
                          'final_stability':self.final_stability},
                'pool_live_bytes':cp.get_default_memory_pool().used_bytes(),
                'pool_reserved_bytes':cp.get_default_memory_pool().total_bytes(),
                'scalar_forecast_members_advanced':0}

    def consume_output_diagnostics(self):
        """Complete the original one-frame reflectivity handoff on output."""
        if self.clock.ticks == self.clock.spec.start_ticks:
            return None
        from gpuwm.core.refl import consume_refl_10cm
        return consume_refl_10cm(self.physics.state)


def run_real_ensemble(*,inputs,members,output_directory,run_seconds,member_initializer,
                      member_seeds,capture_initial_words,renderer,history_interval_seconds):
    import cupy as cp
    from pathlib import Path
    from gpuwm.ensemble.batch_product_output import PreparedEnsembleOutput
    started = time.perf_counter()
    if inputs.experiment.run_seconds != run_seconds or inputs.experiment.root.history_interval_s != history_interval_seconds:
        raise ValueError('benchmark window or output cadence differs from the prepared clock authority')
    runtime = PreparedRealEnsemble(inputs,members=members,member_seeds=member_seeds,
                                   member_initializer=member_initializer)
    initialized = time.perf_counter()
    out = Path(output_directory)
    out.mkdir(parents=True,exist_ok=False)
    cfg,physics,clock = runtime.cfg,runtime.physics,runtime.clock
    fields = physics.driver.fields
    shape = (members,cfg.ny,cfg.nx)
    latitude,longitude=inputs.grids[0].latlon_mass()
    diagnostics = PreparedEnsembleOutput(u10=fields['u10'].reshape(shape),v10=fields['v10'].reshape(shape),
        temperature2=fields['t2'].reshape(shape),rainnc=physics.driver.microphysics.rainnc.reshape(shape),
        rainc=physics.driver.rainc.reshape(shape) if physics.driver.rainc is not None else None,
        rainsh=fields.get('rainsh').reshape(shape) if fields.get('rainsh') is not None else None,
        latitude=latitude,longitude=longitude,
        members=members,available_bytes=int(cp.cuda.runtime.memGetInfo()[0])-(512<<20))
    diagnostics.capture_initial_rain()
    initial_words=[]
    validation_seconds=0.
    if capture_initial_words:
        import hashlib
        from gpuwm.core.device_inventory import state_array_shapes
        begin=time.perf_counter()
        for member in range(members):
            digest=hashlib.sha256()
            for name in sorted(state_array_shapes(cfg)):
                a=runtime.batch.member_view(name,member).get()
                digest.update(name.encode());digest.update(a.dtype.str.encode());digest.update(str(a.shape).encode());digest.update(a.tobytes())
            initial_words.append({'sha256':digest.hexdigest()})
        validation_seconds=time.perf_counter()-begin
    def frame():
        valid=inputs.experiment.start_time+timedelta(seconds=clock.elapsed_seconds)
        stamp=valid.strftime('%Y-%m-%d_%H-%M-%S')
        runtime.consume_output_diagnostics()
        diagnostics(path=out/f'products_{stamp}.nc',valid_time=valid.isoformat()+'Z',
                    maps_directory=out/'maps',renderer=renderer)
    while not clock.at_stop_time:
        if clock.history_due():
            frame()
        runtime.step()
    runtime.final_health = runtime.validator.require_healthy(phase='final')
    runtime.final_stability = runtime.stability()
    if clock.history_due():
        frame()
    cp.cuda.get_current_stream().synchronize()
    elapsed=time.perf_counter()-started
    if list(out.rglob('wrfout*')):
        raise RuntimeError('aggregate-only ensemble unexpectedly wrote member histories')
    return {'execution_mode':'true_member_batched','wall_seconds':elapsed,
            'initialization_seconds':initialized-started,'validation_transfer_seconds':validation_seconds,
            'initial_words':initial_words,'runtime':runtime.receipt(),
            'output':diagnostics.receipt(hash_artifacts=False),'member_history_count':0,
            'timing_scope':'backend entry through every finished Rust probability map; shared source preparation belongs to caller'}
