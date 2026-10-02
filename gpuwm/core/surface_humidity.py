"""WRF's final land Q2 bound, after all surface diagnostic writers."""


def cap_land_q2(q2, qv1, xland, *, xp) -> None:
    """Apply module_surface_driver.F:4443-4457 in place.

    WRF limits the diagnostic flux inversion where surface fluxes can include
    vegetation or nonlocal sources. Water columns keep the diagnosed value.
    The producer supplies its array backend; production fields are FP32.
    """
    factor = q2.dtype.type(1.05)
    land_threshold = q2.dtype.type(1.5)
    q2[...] = xp.where(
        xland < land_threshold, xp.minimum(q2, factor * qv1), q2)
