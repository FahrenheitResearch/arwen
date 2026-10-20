"""Fixed-layout diagnostic full-state validator; no production dispatch change."""
import ast,hashlib
from contextlib import contextmanager
from pathlib import Path

# Re-pinned to lane/speed2-integrate e18bd09a9ca: health.py differs from the audited f19bde24.. copy only in require_healthy (drops descriptor refs after a healthy scan); the decoded validate() body is unchanged.
HEALTH_SHA='4b74fe1c6bedb2129a5dbc81abe62689d29b2e8b40238424b671bf780de50158'

def original_decoder():
    import cupy as cp
    from gpuwm.core import health
    path=Path(health.__file__)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==HEALTH_SHA
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='StateHealthValidator')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='validate')
    start=next(i for i,n in enumerate(method.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple))
    wrapper=ast.parse('def decode(self,words,phase):\n nfield=len(self.fields)\n').body[0]
    wrapper.body.extend(method.body[start:]);module=ast.Module(body=[wrapper],type_ignores=[]);ast.fix_missing_locations(module)
    namespace=dict(vars(health));namespace['cp']=cp
    exec(compile(module,str(path),'exec'),namespace)
    return namespace['decode']


def error_factory(validator,phase):
    import cupy as cp
    import numpy as np
    from gpuwm.core.health import HealthCheckError
    decode=original_decoder()
    def error(index,value):
        words=cp.asnumpy(validator._result).view(np.uint64)
        assert index==1 and int(words[1])==value
        try:report=decode(validator,words,phase)
        except Exception as actual:return actual
        assert not report.ok
        return HealthCheckError(report)
    return error


def readback(plan,original,array,caller,args,kwargs,receipt):
    validator=caller.f_locals['self'];phase=caller.f_locals['phase']
    assert array is validator._result
    with plan.value_guard(error_factory(validator,phase),word_bytes=8,mask=2,expected=(1<<64)-1):
        result=original(array,*args,**kwargs)
    receipt.append({'kind':'full_state_validator','fields':len(validator.fields),'phase':phase,'predicate':'packed == UINT64_MAX'})
    return result


@contextmanager
def annotate_descriptors(plan,receipt):
    from gpuwm.core.health import StateHealthValidator
    original_decoder() # Refuse a different original consumer before recording.
    original=StateHealthValidator._refresh
    def refresh(validator):
        with plan.constant_uploads(private_events=True):original(validator)
        # All arrays and descriptors remain resident for this fixed-time proof.
        plan.resources.extend((validator,validator.fields))
        receipt.append({'kind':'validator_descriptors','count':len(validator.fields),'scope':'fixed layout only'})
    StateHealthValidator._refresh=refresh
    try:yield
    finally:StateHealthValidator._refresh=original
