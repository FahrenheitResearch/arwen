"""Pinned complete-period health consumers, with original refusal semantics.

This diagnostic adapter admits only the resident AC stability record and the
current HealthLedger site map. General model dispatch remains unimplemented.
"""
from contextlib import contextmanager
from pathlib import Path
import sys


def ledger_error(ledger, sites, descriptions):
    def error(index, value):
        site=sites[index];describe=descriptions.get(site)
        if describe is None:
            return FloatingPointError(f'{site} reported health flags {value:#x} and no description was recorded for it')
        try:describe(value)
        except Exception as actual:return actual
        return FloatingPointError(f'{site} reported health flags {value:#x} but its description did not raise; a health site must refuse')
    return error


@contextmanager
def annotate_health(plan, receipt, *, step_label=None):
    import cupy as cp
    from gpuwm.core.health_ledger import HealthLedger
    original_drain=HealthLedger.drain;original_numpy=cp.asnumpy
    def drain(ledger):
        if not ledger._index:return original_drain(ledger)
        sites={i:site for site,i in ledger._index.items()}
        assert ledger._slots.dtype==cp.uint32 and ledger._slots.size<=64
        mask=sum(1<<i for i in sites)
        with plan.zero_guard(ledger_error(ledger,sites,dict(ledger._describe)),word_bytes=4,mask=mask):
            original_drain(ledger)
        receipt.append({'kind':'health_ledger','sites':sites,'mask':mask,'clear_before_raise':True})
    def asnumpy(array,*args,**kwargs):
        caller=sys._getframe(1)
        if (Path(caller.f_code.co_filename).name,caller.f_lineno)==('health.py',1337):
            from validator_annotations import readback
            return readback(plan,original_numpy,array,caller,args,kwargs,receipt)
        if (Path(caller.f_code.co_filename).name,caller.f_lineno)!=('dycore.py',4928):
            return original_numpy(array,*args,**kwargs)
        cfg=caller.f_locals['cfg']
        assert cfg is not None and array.dtype==cp.float32 and array.shape==(8,)
        # Under this explicit bounded admission all finite float32 inputs keep
        # the original Python float64 dt*u/dx and dt*rate arithmetic finite.
        assert 0 < cfg.dt < 1e100 and 1e-100 < cfg.dx < 1e100
        frame=caller.f_back
        while frame is not None and frame.f_code.co_name!='on_step':frame=frame.f_back
        assert frame is not None,'stability consumer needs the model on_step clock'
        clock=frame.f_locals['clock'];grid_id=frame.f_locals['grid_id']
        # This adapter proves a fixed captured interval. An advancing adapter
        # must supply the operation's live step number through a frame slot.
        step_number=clock.step_count+1
        def error(index,bits):
            number=step_number if step_label is None else step_label(grid_id,step_number)
            if index<3:
                return RuntimeError(f'd{grid_id:02d} integration produced a non-finite state at model step {number}')
            return RuntimeError(f'd{grid_id:02d} integration produced a non-finite vertical Courant number at model step {number}: a model layer\'s live thickness is non-positive or non-finite')
        with plan.finite_guard(error,word_bytes=4,mask=0x27):
            result=original_numpy(array,*args,**kwargs)
        receipt.append({'kind':'stability_finite','mask':0x27,'grid_id':grid_id,'word_bytes':4})
        return result
    HealthLedger.drain=drain;cp.asnumpy=asnumpy
    try:yield
    finally:HealthLedger.drain=original_drain;cp.asnumpy=original_numpy
