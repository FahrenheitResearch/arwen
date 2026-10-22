"""Explicit metadata publication for admitted fixed-layout non-radiation intervals."""
import time
from gpuwm.core.state import refresh_model_time
from gpuwm.ingest.lateral_bc import attach_nest_boundaries
from gpuwm.core.cam_ozone import ROUTING_PARENT_INTERPOLATED

class Metadata:
 def __init__(self,model,closure):
  self.model=model;self.closure=closure;self.status=closure['status'];self.ledger=closure['step_ledger'];self.step_deltas={}
 def capture_before(self,gid):
  p=self.model.node(gid).state.physics
  return dict(p.call_counts),p.microphysics_updates,self.ledger.records,self.ledger.drains
 def capture_after(self,gid,before):
  p=self.model.node(gid).state.physics;counts,micro,records,drains=before
  value=({k:v-counts.get(k,0) for k,v in p.call_counts.items() if v!=counts.get(k,0)},p.microphysics_updates-micro,self.ledger.records-records,self.ledger.drains-drains)
  if gid in self.step_deltas:assert value==self.step_deltas[gid]
  self.step_deltas[gid]=value
 def step(self,gid,clock):
  started=time.perf_counter()
  node=self.model.node(gid);assert node.clock is clock
  if node.parent is not None:assert node.coupler.valid
  self.status.schedule_cursor='EXECUTING'
  refresh_model_time(node.state,clock,kernel_launch=True)
  counts,micro,records,drains=self.step_deltas[gid]
  for name,delta in counts.items():node.state.physics.call_counts[name]+=delta
  node.state.physics.microphysics_updates+=micro
  self.ledger.records+=records;self.ledger.drains+=drains
  refresh_model_time(node.state,clock,after_step=True)
  observer=self.closure['step_observer']
  if observer is not None:
   try:observer(grid_id=gid,step_count=clock.step_count+1,model_seconds=(clock.ticks+clock.step_ticks)/clock.tick_den,step_wall_seconds=time.perf_counter()-started,dt=clock.step_ticks/clock.tick_den)
   except Exception:pass
  walls=self.closure['step_wall_by_domain'];walls[gid]=walls.get(gid,0)+(time.perf_counter()-started)
 def force(self,child_id,parent_id,child_clock,parent_clock):
  node=self.model.node(child_id);coupler=node.coupler;run=node.cfg.run
  assert node.parent.cfg.grid_id==parent_id and parent_clock.ticks-child_clock.ticks==parent_clock.step_ticks
  assert coupler.inflow_perturbation is None
  self.status.prior_feedback_committed=False
  owner=node.state.physics.cam_ozone
  if owner is not None and owner.mode==ROUTING_PARENT_INTERPOLATED:node.state.physics.call_counts['cam_ozone']+=1
  attach_nest_boundaries(node.state,coupler._last_tables,clock=child_clock,spec_bdy_width=run.spec_bdy_width,spec_zone=run.spec_zone,relax_zone=run.relax_zone)
  child_clock.mark_force()
  coupler.last_parent_ticks=int(parent_clock.ticks);coupler.last_parent_step=int(parent_clock.step_count);coupler.force_count+=1
 def feedback_commit(self,child_id,parent_id,child_clock,parent_clock):
  coupler=self.model.node(child_id).coupler
  assert coupler._prepared_feedback['ticks']==int(child_clock.ticks)
  coupler.feedback_count+=1;coupler.last_feedback_ticks=int(child_clock.ticks)
 def feedback_finalize(self,child_id,parent_id,child_clock,parent_clock):
  self.model.node(child_id).coupler._prepared_feedback=None
  self.status.pending_feedback-=1
 def end(self):
  self.status.schedule_cursor='PERIOD_BEGIN';self.status.prior_feedback_committed=self.status.pending_feedback==0

import copy
import cupy as cp
import numpy as np
from gpuwm.core import physics,rrtmg_legacy
from gpuwm.core.physics import _physics_interval_steps,_radiation_step_due,_prepare_atmosphere
from gpuwm.core.radiation_composition import legacy_radiation_adapter
from gpuwm.core.physics_inventory import RADIATION_BOUNDARY_FLUXES

def sync():cp.cuda.get_current_stream().synchronize()

def due(state,cfg):
 epoch=float(getattr(state,'domain_start_offset',0.) or 0.)
 n=int(np.floor((state.elapsed_seconds-epoch)/cfg.dt+0.5))+1
 return _radiation_step_due(n,_physics_interval_steps(state.physics.radt_minutes,cfg.dt),state.physics.radt_minutes)

def warm_day(node):
 adapter=legacy_radiation_adapter(node.state.physics.radiation_callable);new=copy.copy(adapter)
 if new.__dict__.get('_ozmixt_buf') is not None:new._ozmixt_buf=new._ozmixt_buf.copy()
 state=copy.copy(node.state);state.elapsed_seconds+=18*3600;fields=dict(state.physics.fields)
 for key,value in fields.items():
  if key in RADIATION_BOUNDARY_FLUXES:fields[key]=value.copy()
 rrtmg_legacy.RRTMGLegacyRadiation.__call__(new,atmosphere=_prepare_atmosphere(node.state),fields=fields,state=state,cfg=node.cfg.run);sync()

class RadiationMetadata(Metadata):
 def __init__(self,model,closure,recordings):super().__init__(model,closure);self.recordings=recordings;self.radiation_records={}
 def step(self,gid,clock):
  node=self.model.node(gid);p=node.state.physics
  state=copy.copy(node.state);refresh_model_time(state,clock,kernel_launch=True);is_due=due(state,node.cfg.run);now=float(state.elapsed_seconds)
  saved=self.step_deltas[gid];counts,micro,records,drains=saved;counts=dict(counts)
  if not is_due:
   assert counts.pop('radiation')==1
   records-=self.radiation_records[gid];assert records>=0
   if node.parent is None:assert counts.pop('cam_ozone')==1
  self.step_deltas[gid]=(counts,micro,records,drains)
  try:super().step(gid,clock)
  finally:self.step_deltas[gid]=saved
  if is_due:
   recorded=self.recordings[gid];adapter=recorded.adapter;adapter.update_count+=1;adapter._o33d_host=None;recorded.publish_metadata()
   for name in ('glw','swdown','gsw','coszen'):
    if name in p.fields:p.carriers.declare(name,source=physics.CARRIER_SOURCE_RADIATION_SCHEME,model_time=now)
   if node.parent is None:p.carriers.declare('o3rad',source='cam_ozone',model_time=now)
