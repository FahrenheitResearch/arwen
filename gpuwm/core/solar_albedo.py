"""Sun-angle land albedo, refreshed at radiation calls.

The operation order and MODIS class map are those in
``module_radiation_driver.F:781-789,1038-1063`` of NOAA-EMC/HRRR
v4.1.21.  The source-pinned Fortran oracle is built by
``tools/solar_albedo_oracle/build.py``.  ALBSOL is the surface albedo the
shortwave and land surface consume; ALBBCKSOL is the corresponding
snow-free background passed to the land surface.  ALBEDO and ALBBCK stay
the original, overhead-sun fields.

Only snow-free, ice-free land with positive COSZEN is normalized.  The
other cells retain the land surface's last values.  The first call copies
the original albedos before applying that mask.  Only ALBSOL is capped at
0.9, including cells outside the normalization mask.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

F = np.float32

STATE_FIELDS = ("albsol", "albbcksol")
# DATA istwe /5*1,2,2,1,1,5*2,1,2,2,1,2,2,2/ ! MODIS 21 classes
MODIS_ISTWE = (1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 1,
               2, 2, 2)
_D = np.array([F(0.1), F(0.25)], dtype=F)
_INPUTS = ("albedo", "albbck", "xland", "snow", "xice", "ivgtyp",
           *STATE_FIELDS)


def _arrays(fields: Mapping, coszen, *, xp):
    shape = fields["albsol"].shape
    out = []
    for name in (*_INPUTS, "coszen"):
        a = coszen if name == "coszen" else fields[name]
        dtype = np.int32 if name == "ivgtyp" else np.float32
        if not isinstance(a, xp.ndarray):
            raise TypeError(f"solar albedo {name} must be a {xp.__name__} array")
        # COSZEN may be the driver's flat radiation column vector.
        if a.dtype != dtype or (a.shape != shape and not (
                name == "coszen" and a.ndim == 1 and a.size == np.prod(shape))):
            raise ValueError(
                f"solar albedo {name} must have dtype {np.dtype(dtype)} "
                f"and shape {shape}, got {a.dtype} {a.shape}")
        if not a.flags.c_contiguous:
            raise ValueError(f"solar albedo {name} must be C-contiguous")
        out.append(a.reshape(shape))
    return out


def update_solar_albedo_host(fields: Mapping, coszen, *, initialize=False):
    """Update ``albsol`` and ``albbcksol`` in place, float32 host twin.

    The caller selects ``alb_sol=1`` and calls only on radiation steps.
    ``initialize`` corresponds to the fork's ``itimestep == 1``.  IVGTYP
    uses the 21-category MODIS table; no dataset remapping is performed.
    """
    (albedo, albbck, xland, snow, xice, ivgtyp, albsol, albbcksol,
     coszen) = _arrays(fields, coszen, xp=np)
    if initialize:
        albsol[...] = albedo
        albbcksol[...] = albbck
    active = ((coszen > F(0.0)) & (xland < F(1.5))
              & (snow == F(0.0)) & (xice == F(0.0)))
    categories = ivgtyp[active]
    if np.any((categories < 1) | (categories > 21)):
        raise ValueError("solar albedo needs MODIS IVGTYP classes 1 through 21")
    istwe = np.array(MODIS_ISTWE, dtype=np.int32)[categories - 1]
    d = _D[istwe - 1]
    twice_d = F(2.0) * d
    numerator = F(1.0) + twice_d
    denominator = F(1.0) + twice_d * coszen[active]
    dm = numerator / denominator
    normalized = albbck[active] * dm
    albsol[active] = normalized
    albbcksol[active] = normalized
    np.minimum(albsol, F(0.9), out=albsol)


def update_solar_albedo_cuda(fields: Mapping, coszen, *, initialize=False):
    """Device update with the same contract as the host twin.

    Each floating-point operation in the kernel is an explicit rounded
    intrinsic, including the division.  The caller owns the device arrays
    and the radiation schedule; this function carries no geometry state.
    """
    import cupy as cp
    from gpuwm.core.kernels import load_module
    arrays = _arrays(fields, coszen, xp=cp)
    n = fields["albsol"].size
    if not n:
        return
    _, _, xland, snow, xice, ivgtyp, _, _, cosine = arrays
    active = ((cosine > F(0.0)) & (xland < F(1.5))
              & (snow == F(0.0)) & (xice == F(0.0)))
    if bool(cp.any(active & ((ivgtyp < 1) | (ivgtyp > 21)))):
        raise ValueError("solar albedo needs MODIS IVGTYP classes 1 through 21")
    kernel = load_module("solar_albedo").get_function("solar_albedo_update")
    kernel(((n + 255) // 256,), (256,),
           (np.int64(n), np.int32(bool(initialize)),
            *(a.reshape(-1) for a in arrays)))
