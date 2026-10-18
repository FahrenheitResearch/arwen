"""Copy actual WSM6 arguments, outputs and retained heating at step one."""
import inspect
import gpuwm.core.wsm6 as M
import woof_hook as H

_apply = M.apply
_launch = M.launch_wsm6
_state = None


def apply(state, cfg, *args, **kwargs):
    global _state
    previous = _state
    _state = state if H.active(state, cfg) else None
    try:
        result = _apply(state, cfg, *args, **kwargs)
        if _state is not None:
            H.dump(state, "mp_wsm6_finish")
        return result
    finally:
        _state = previous


def launch(*args, **kwargs):
    if _state is None:
        return _launch(*args, **kwargs)
    inputs = dict(inspect.signature(_launch).bind(*args, **kwargs).arguments)
    H.dump(_state, "mp_wsm6_in", **inputs)
    result = _launch(*args, **kwargs)
    H.dump(_state, "mp_wsm6_out", **inputs)
    return result


M.apply = apply
M.launch_wsm6 = launch
