"""Correct mode, finding WB1: book the cloud water Thompson settles out.

THE DEFECT.  Thompson's cloud-water sedimentation (``mp_thompson`` in WRF
v4.6.1, the same routine MPAS-A v8.4.2 runs; ``kernels/thompson.cu``
``thompson_cloud_sediment*``) moves droplets down and out of the lowest
level, and the water that leaves the column is booked nowhere: not in
RAINNC, not in the land surface, not in the atmosphere.  The routine
computes the flux and then drops it, while every other flux through the
surface is booked, so the water vanishes from the model's budget.

THE CORRECTION (``physics.mode = "correct"``, the 2.9.0 default;
``RunConfig.thompson_cloud_surface_booking``).  The settled water is booked
in its own accumulated total, ``CLOUDNC`` (mm), with the per-call amount in
``CLOUDNCV``.  Rain is NOT changed: WRF, UFS and MPAS all keep settled cloud
out of their precipitation totals, and adding it to rain was refuted when
the MPAS port of this fix was reviewed (the finding is class B for "add it
to rain" and class A for "recorded nowhere").  The land surface is not
wetted either (fog drip needs its own observation A/B).  So the atmosphere,
the rain and every prognostic word are bit-identical between the two modes;
only the booked total exists in one of them.

HOW.  The frozen kernels are not touched.  The cloud fallout kernel changes
nothing but ``qc``, and converts its flux divergence back to a mixing-ratio
tendency with its own density, ``0.622 p / (287.04 T (max(qv, 1e-10) +
0.622))``, formed from the temperature, pressure and vapour it is handed and
does not change.  The column water on that density, summed over the column
before and after the launch, therefore differs by exactly the flux out of
the lowest level times the step, which is the amount WB1 books.  Columns in
the specified-zone ring, which WRF's clipped microphysics tiles never touch
(the ring guard restores them after the call), book nothing.
"""

from __future__ import annotations

try:  # pragma: no cover - import guard mirrors the other core modules
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None

#: The scratch slots the correction keeps: the accumulated total and the
#: per-call amount (mm = kg m-2), surface-shaped.
CLOUDNC_SLOT = "mp_cloudnc"
CLOUDNCV_SLOT = "mp_cloudncv"


def cloud_column_water(qc, temperature, pressure, qv, dz, *, xp=None):
    """Column cloud water (kg m-2, float64) on the fallout kernel's density.

    ``qc``, ``temperature``, ``pressure``, ``qv`` and ``dz`` are the
    ``(nz, ny, nx)`` arrays the fallout launch reads.
    """
    xp = cp if xp is None else xp
    f32 = xp.float32
    qvk = xp.maximum(f32(1.0e-10), qv)
    density = (f32(0.622) * pressure
               / (f32(287.04) * temperature * (qvk + f32(0.622))))
    return (qc.astype(xp.float64) * density.astype(xp.float64)
            * dz.astype(xp.float64)).sum(axis=0)


def book_settled_cloud(state, cfg, before, qc, temperature, pressure, qv,
                       dz, *, xp=None):
    """Accumulate the water the fallout launch took out of each column.

    ``before`` is :func:`cloud_column_water` taken immediately before the
    launch on the same arrays.  Returns ``(cloudnc, cloudncv)``.
    """
    xp = cp if xp is None else xp
    after = cloud_column_water(qc, temperature, pressure, qv, dz, xp=xp)
    settled = (before - after).astype(xp.float32)
    from gpuwm.core.microphysics import _ring_guard_slices
    slices = _ring_guard_slices(state, cfg)
    if slices:
        for slc in slices:
            settled[slc] = 0
    shape = settled.shape
    cloudncv = state.scratch(shape, CLOUDNCV_SLOT)
    cloudnc = state.scratch(shape, CLOUDNC_SLOT)
    cloudncv[...] = settled
    cloudnc += settled
    return cloudnc, cloudncv


__all__ = ["CLOUDNCV_SLOT", "CLOUDNC_SLOT", "book_settled_cloud",
           "cloud_column_water"]
