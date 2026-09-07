"""Active boundary-field inventory shared by initialization and memory pricing."""
from __future__ import annotations

from gpuwm.config import RunConfig


def nest_field_kinds(cfg: RunConfig) -> tuple[str, ...]:
    """Child forcing field kinds: u/v/w/t(thm)/ph/mu + ALL active
    moist/scalar species including Thompson/Morrison numbers (architecture
    section D;
    module_bc_em.F:320-345 w coupling; Registry scalar set qnr/qni/qns/
    qng at Registry.EM_COMMON:3026).

    ``nc`` is scheme-dependent, and the reason is the advection copy.  For
    mp_physics=10 it is EXCLUDED: Morrison allocates ``nc`` but has no
    ``nc0``, does not transport it, and diagnoses a fixed 250 cm-3 every
    call, so forcing it across a nest edge would carry a field the child
    immediately overwrites.  For mp_physics=28 it is INCLUDED: aerosol-aware
    Thompson makes droplet number prognostic, allocates ``nc0`` and advects
    it alongside nwfa/nifa (gpuwm/core/moist.py::THOMPSON_AERO_NUMBER_SPECIES),
    so it is a real forced boundary field like ``nr``/``ni``.  The two facts
    are not in tension -- the inventory follows what is transported, not what
    is allocated.

    The three ice MASSES are scheme-dependent for the same reason, and
    ``mp_physics=50`` is EXCLUDED from that block deliberately.  What this
    inventory decides is not "does the scheme have ice" but "which of WRF's
    boundary-forced Registry members does the scheme activate": every kind
    named here is declared ``ikjftb`` with ``i0rhusdf=(bdy_interp:dt)`` --
    the ``moist`` masses at Registry.EM_COMMON:452-469 and the ``scalar``
    moments at :520-558.  P3's package is
    ``moist:qv,qc,qr,qi;scalar:qni,qnr,qir,qib`` (Registry.EM_COMMON:3038):
    ONE ice mass, no ``qs`` and no ``qg``, which is why the mp=50 arm below
    brings its own ``qi`` instead of joining the tuple.  WRF's driver says
    the same thing -- the ``P3_1CATEGORY`` arm passes no snow and no
    graupel array at all (module_microphysics_driver.F:1557-1602, against
    ``mp_p3_wrapper_wrf``'s signature at module_mp_p3.F:690-699).  Folding
    50 in would name ``qi`` twice and price sixteen rolling boundary tables
    for two species an mp=50 child never allocates (gpuwm/core/state.py's
    mp=50 arm), and the first ``force`` would then raise ``state has no
    active nest field 'qs'`` (gpuwm/core/nest.py:186-193).

    P3's remaining package members -- ``state:re_cloud,re_ice,vmi3d,
    rhopo3d,di3d,refl_10cm,th_old,qv_old`` -- are absent for a DIFFERENT
    reason, recorded here so it is not read as a second omission: they are
    ``misc`` state with no ``b`` in the io string (th_old/qv_old at
    Registry.EM_COMMON:1598-1599), so WRF builds no boundary arrays for
    them and ``bdy_interp`` never touches them.  ``p3_main`` rewrites
    th_old/qv_old at the end of every call (module_mp_p3.F:5018-5021);
    they are cross-step carriers the child regenerates, not forced
    boundary fields.

    This inventory contains only REAL prognostic fields handled by WRF
    ``copy_fcn`` (mass-cell or U/V face averaging).  It contains no
    masked/surface ``copy_fcnm`` fields and no integer ``copy_fcni`` fields;
    that is an explicit two-way-feedback scope divergence from stock WRF,
    not a request to apply the mass operator to masked state.
    """
    kinds = ["u", "v", "w", "t", "ph", "mu"]
    if cfg.moist:
        kinds += ["qv", "qc", "qr"]
        if cfg.mp_physics in (6, 8, 9, 10, 16, 18, 28):
            kinds += ["qi", "qs", "qg"]
        if cfg.mp_physics == 9:
            # The inventory follows what is transported
            # (gpuwm/core/moist.py::MY2_SPECIES): hail mass plus all six
            # number moments cross a nest edge with the masses they
            # describe (1.9.1 D1's route: mp=9 had no arm here, so a
            # nested Milbrandt-Yau child would have been forced with
            # WSM6's field set).
            kinds += ["qh", "nc", "nr", "ni", "ns", "ng", "nh"]
        if cfg.mp_physics == 16:
            kinds += ["nn", "nc", "nr"]
        if cfg.mp_physics == 8:
            kinds += ["nr", "ni"]
        if cfg.mp_physics == 28:
            kinds += ["nr", "ni", "nc", "nwfa", "nifa"]
        if cfg.mp_physics == 10:
            kinds += ["nr", "ni", "ns", "ng"]
        if cfg.mp_physics == 18:
            kinds += ["qh", "qndrop", "qnr", "qni", "qns", "qng",
                      "qnh", "qnn", "qvolg", "qvolh"]
        if cfg.mp_physics == 50:
            # P3 one-category: the inventory follows what is transported
            # (gpuwm/core/moist.py::P3_SPECIES), so the rime mass/volume
            # pair is forced across a nest edge exactly like the number
            # moments.  Forcing qi without them would hand the child ice
            # whose rime fraction and rime density came from whatever the
            # child's own last step left behind.  DELIBERATELY not folded
            # into the qi/qs/qg tuple above: this arm carries P3's single
            # ice mass itself, because Registry.EM_COMMON:3038 gives the
            # scheme no qs and no qg (the docstring carries the
            # consequence of widening that tuple).
            kinds += ["qi", "ni", "nr", "qir", "qib"]
    return tuple(kinds)
