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
