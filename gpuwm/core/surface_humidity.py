"""WRF's final Q2 bound, after all surface diagnostic writers."""


def cap_surface_q2(q2, qv1, xland, *, xp, include_water=False) -> None:
    """Apply the final surface-driver mixing-ratio bound in place.

    The public WRF land form is module_surface_driver.F:4443-4457.
    The operational fork at :4117-4127 applies the same bound to every
    column after the land and lake diagnostic writers. The producer supplies
    its array backend; production fields are FP32.
    """
    factor = q2.dtype.type(1.05)
    if include_water:
        q2[...] = xp.minimum(q2, factor * qv1)
    else:
        land_threshold = q2.dtype.type(1.5)
        q2[...] = xp.where(
            xland < land_threshold, xp.minimum(q2, factor * qv1), q2)


def cap_land_q2(q2, qv1, xland, *, xp) -> None:
    """Apply the public WRF land-only surface-driver bound in place."""
    cap_surface_q2(q2, qv1, xland, xp=xp)
