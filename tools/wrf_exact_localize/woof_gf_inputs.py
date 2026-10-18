"""Copy the packed GF launch arguments and outputs without changing the call."""
import os
from pathlib import Path

import cupy as cp
import numpy as np

import gpuwm.core.gf as G
import woof_hook as H

_context = None
_call = G.GrellFreitas.__call__
_module = G._gf_module


def call(self, *, state, cfg, **kwargs):
    global _context
    previous = _context
    _context = {"part": 0} if H.active(state, cfg) else None
    try:
        return _call(self, state=state, cfg=cfg, **kwargs)
    finally:
        _context = previous


class Function:
    def __init__(self, function):
        self.function = function

    def __getattr__(self, name):
        return getattr(self.function, name)

    def __call__(self, grid, block, args, **kwargs):
        context = _context
        if context is None:
            return self.function(grid, block, args, **kwargs)
        part = context["part"]
        context["part"] += 1
        inputs = {name: cp.asnumpy(args[index]) for index, name in
                  enumerate(("lvin", "scin", "iin"))}
        result = self.function(grid, block, args, **kwargs)
        outputs = {name: cp.asnumpy(args[index + 3]) for index, name in
                   enumerate(("lev", "sca", "isc"))}
        path = Path(os.environ["WOOF_LOCDUMP"]) / f"gf_launch_{part:04d}.npz"
        np.savez(path, **inputs, **outputs)
        return result


class Module:
    def __init__(self, module):
        self.module = module

    def __getattr__(self, name):
        return getattr(self.module, name)

    def get_function(self, name):
        function = self.module.get_function(name)
        return Function(function) if name == "gf_gfdrv_stage" else function


def module(*args, **kwargs):
    result = _module(*args, **kwargs)
    return Module(result) if _context is not None else result


G.GrellFreitas.__call__ = call
G._gf_module = module
