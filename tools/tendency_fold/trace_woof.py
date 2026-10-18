"""Record actual composer line boundaries without changing its arithmetic."""
import inspect
import json
import os
from pathlib import Path
import sys

import cupy as cp
import numpy as np
from gpuwm.core import physics as P

OUT = Path(os.environ['TENDENCY_FOLD_DUMP'])
OUT.mkdir(parents=True, exist_ok=True)
original = P.PhysicsDriver._compose_tendencies
lines, first = inspect.getsourcelines(original)
assign = next(first + i for i, s in enumerate(lines)
              if 'value[...] = getattr(pbl, name)' in s)
add_rad = next(first + i for i, s in enumerate(lines)
               if 'value += getattr(self.radiation_tendencies, name)' in s)
add_cu = next(first + i for i, s in enumerate(lines)
              if 'value += getattr(self.cumulus_tendencies, name)' in s)


def save(tag, **arrays):
    np.savez(OUT / (tag + '.npz'), **{k: cp.asnumpy(v) for k, v in arrays.items()
                                  if v is not None})


def compose(self, cfg):
    if self.state.elapsed_seconds >= cfg.dt:
        return original(self, cfg)
    save('components', **{label + '_' + name: getattr(component, name, None)
         for label, component in [('pbl', self.pbl_tendencies),
                                   ('radiation', self.radiation_tendencies),
                                   ('cumulus', self.cumulus_tendencies)]
         for name in ('ru', 'rv', 'rtheta')})
    trace = []
    previous = sys.gettrace()
    last = [None]

    def observer(frame, event, arg):
        if frame.f_code is original.__code__ and event == 'line':
            if frame.f_locals.get('name') == 'rtheta':
                # A line event occurs before executing that line.
                tag = {assign: 'woof_after_pbl', add_rad: 'woof_after_radiation',
                       add_cu: 'woof_after_cumulus'}.get(last[0])
                if tag:
                    target = self.tendencies
                    save(tag, rt_tendf=target.rtheta, ru_tendf=target.ru,
                         rv_tendf=target.rv)
                    trace.append(dict(tag=tag, line=frame.f_lineno))
            last[0] = frame.f_lineno
        return observer
    sys.settrace(observer)
    try:
        return original(self, cfg)
    finally:
        sys.settrace(previous)
        (OUT / 'line-boundaries.json').write_text(json.dumps(trace, indent=2))


P.PhysicsDriver._compose_tendencies = compose
couple = P.couple_column_tendencies


def column(state, cfg, **rates):
    if state.elapsed_seconds < cfg.dt:
        # radiation has theta only; cumulus also supplies water vapour.
        if rates.get('rqv') is not None:
            save('raw_cumulus', **rates)
        save('mass_inputs', c1h=state.c1h, c2h=state.c2h,
             mut=state.total_mu())
    return couple(state, cfg, **rates)


P.couple_column_tendencies = column

if os.environ.get('TENDENCY_FOLD_GF_INPUTS') == '1':
    # The production kernel arguments, including its complete output slabs.
    # The wrapper delegates unchanged and keeps every dump on this box.
    import gpuwm.core.gf as G
    gf_call, gf_module = G.GrellFreitas.__call__, G._gf_module
    context = [None]

    def gf_adapter(self, *, state, cfg, **kwargs):
        context[0] = {'part': 0} if state.elapsed_seconds < cfg.dt else None
        try:
            return gf_call(self, state=state, cfg=cfg, **kwargs)
        finally:
            context[0] = None

    class Function:
        def __init__(self, fn):
            self.fn = fn

        def __getattr__(self, name):
            return getattr(self.fn, name)

        def __call__(self, grid, block, args):
            part = context[0]['part']
            context[0]['part'] += 1
            result = self.fn(grid, block, args)
            save(f'gf_launch_{part:04d}', **dict(zip(
                ('lvin', 'scin', 'iin', 'lev', 'sca', 'isc'), args[:6])))
            return result

    class Module:
        def __init__(self, mod):
            self.mod = mod

        def get_function(self, name):
            fn = self.mod.get_function(name)
            return Function(fn) if name == 'gf_gfdrv_stage' else fn

    def module(*args, **kwargs):
        mod = gf_module(*args, **kwargs)
        return Module(mod) if context[0] is not None else mod

    G.GrellFreitas.__call__, G._gf_module = gf_adapter, module

from gpuwm.cli import main

raise SystemExit(main(['verify-exact', *sys.argv[1:]]))
