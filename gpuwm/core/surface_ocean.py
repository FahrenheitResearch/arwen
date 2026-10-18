"""CUDA launchers for the metadata-selected surface-ocean column model."""
import numpy as np
from gpuwm.surface_ocean import ocean_option, validate_ocean_inputs
from gpuwm.core.kernels import get_kernel
from gpuwm.core import constants

STATE_FIELDS = tuple(ocean_option()["state_fields"])
FORCING_FIELDS = ("hfx", "lh", "gsw", "glw", "uair", "vair", "ust", "f", "emiss", "xland")

def _check(fields, names):
    shape = fields["tsk"].shape
    for name in names:
        a = fields[name]
        if a.dtype != np.dtype("float32") or a.shape != shape or not a.flags.c_contiguous:
            raise ValueError(f"surface ocean {name} requires contiguous float32 columns with shape {shape}")
    return fields["tsk"].size

def launch_ocean_init(fields, hml0, *, start=True):
    n = _check(fields, (*STATE_FIELDS, "tsk"))
    get_kernel("oml", "oml_init")(((n+127)//128,), (128,),
        tuple(fields[name] for name in STATE_FIELDS) + (fields["tsk"], np.float32(hml0), np.int32(start), np.int32(n)))

def launch_ocean_step(fields, *, dt, gamma, relaxation_time, constants_override=None):
    n = _check(fields, (*STATE_FIELDS, "tsk", *FORCING_FIELDS))
    c = dict(ocean_option()["constants"])
    if constants_override is not None:
        c.update(constants_override)
    get_kernel("oml", "oml_step")(((n+127)//128,), (128,),
        tuple(fields[name] for name in (*STATE_FIELDS, "tsk", *FORCING_FIELDS)) +
        tuple(np.float32(v) for v in (dt, gamma, relaxation_time, c["rhowater"], c["cwater"], c["stbolt"], constants.G)) + (np.int32(n),))

def initialize_ocean(fields, cfg, initial=None, *, start=True):
    import cupy as cp
    if start:
        validate_ocean_inputs(cfg, initial)
    initial = initial or {}
    shape = fields["tsk"].shape
    for name in STATE_FIELDS:
        value = initial.get(name, 0.0)
        if start and name == "h0ml" and cfg.oml_hml0 <= 0 and getattr(value, "ndim", None) != 2:
            raise ValueError("H0ML must be a 2D input: a scalar does not initialize mixed-layer depth per column")
        a = cp.asarray(value, dtype=cp.float32)
        if a.ndim and a.shape != shape:
            raise ValueError(f"surface ocean {name} requires a 2D input with shape {shape}")
        fields[name] = cp.full(shape, value, dtype=cp.float32) if a.ndim == 0 else cp.ascontiguousarray(a)
    if start and cfg.oml_hml0 <= 0 and bool(cp.any(~cp.isfinite(fields["h0ml"]) | (fields["h0ml"] < 0))):
        raise ValueError("H0ML must be finite and nonnegative: invalid input would leave mixed-layer depth uninitialised")
    launch_ocean_init(fields, cfg.oml_hml0, start=start)
