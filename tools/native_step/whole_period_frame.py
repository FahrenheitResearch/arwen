"""Four explicit domain clocks for a fixed-layout AC diagnostic interval."""
import copy,ctypes as C
from time_slots import Frame,frame_for
from gpuwm.core.state import refresh_model_time
class PeriodFrame(C.Structure):
    _fields_=[(f'd{index}_'+name,kind) for index in range(4) for name,kind in Frame._fields_]

def frame_for_period(model):
    nodes=list(model.nodes_by_grid_id.values())
    assert len(nodes)==2 and nodes[0].parent is None and nodes[1].parent is nodes[0]
    root,child=nodes
    assert root.clock.step_ticks==3*child.clock.step_ticks
    result=PeriodFrame()
    for node,indices in ((root,(0,)),(child,(1,2,3))):
        clock=copy.copy(node.clock);state=copy.copy(node.state)
        resident=copy.copy(state._lateral_boundary_device);resident.clock=clock;state._lateral_boundary_device=resident
        if node.parent is not None:clock.mark_force()
        for index in indices:
            assert not clock.lbc_reset_due(),'external boundary seam needs a new pointer frame'
            clock.prepare_step();refresh_model_time(state,clock,kernel_launch=True)
            frame=frame_for(state,node.cfg.run)
            for name,_ in Frame._fields_:setattr(result,f'd{index}_'+name,getattr(frame,name))
            clock.advance()
    return result
