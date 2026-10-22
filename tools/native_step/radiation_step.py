"""Pinned radiation-due branch annotations for one reusable AC domain step."""
import ast,copy,hashlib
from contextlib import contextmanager
from pathlib import Path
# Re-pinned to integrate/2.9.3 b5b4603f5d8: optional hydro fields/hooks and
# independent ocean reservoirs changed physics.py after speed2-integrate.
# The radiation_due branch moved 5734->5738 with an identical AST.
PHYSICS_SHA256='7beb982b78f171bca7c2aaa389a64382c266d60603a6231febbc7c9d93cda84b'
RADIATION_FLAG=8

def transformed_compute(source):
 if hashlib.sha256(source).hexdigest()!=PHYSICS_SHA256:raise RuntimeError('changed original physics radiation cadence source')
 tree=ast.parse(source);cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='PhysicsDriver')
 method=copy.deepcopy(next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='compute'))
 selected=[(i,x) for i,x in enumerate(method.body) if isinstance(x,ast.If) and x.lineno==5738]
 if len(selected)!=1 or ast.unparse(selected[0][1].test)!='radiation_due' or selected[0][1].orelse:raise RuntimeError('original radiation branch inventory differs')
 i,node=selected[0]
 method.body[i]=ast.copy_location(ast.With(items=[ast.withitem(context_expr=ast.parse('_radiation_plan.when(require=8)',mode='eval').body)],body=node.body),node)
 return ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]))

@contextmanager
def conditional_step(plan,module):
 cls=module.PhysicsDriver;old=cls.compute
 namespace=dict(vars(module),_radiation_plan=plan)
 code=transformed_compute(Path(module.__file__).read_bytes())
 exec(compile(code,module.__file__,'exec'),namespace)
 cls.compute=namespace['compute']
 try:yield
 finally:cls.compute=old

class CadenceScope:
 def __init__(self,plan,metadata,gid):self.plan=plan;self.metadata=metadata;self.gid=gid
 @contextmanager
 def when(self,**flags):
  assert flags=={'require':RADIATION_FLAG}
  before_records=self.metadata.ledger.records
  with self.plan.when(**flags):yield
  self.metadata.radiation_records[self.gid]=self.metadata.ledger.records-before_records
