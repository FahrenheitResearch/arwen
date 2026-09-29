"""External scalar forcing selected by the implemented WRF operations.

WRF v4.6.1 solve_em.F:2803-2839,2904-2930: water vapour is specified;
supplied aerosols are specified when aer_init_opt > 0. Other scalar
moments retain flow-dependent boundaries unless have_bcs_scalar is set
(that general input operation is not yet represented by RunConfig).
"""
from __future__ import annotations


AEROSOL_BOUNDARY_FIELDS = ("nwfa", "nifa")

#: The supplied boundary scalars WRF carries in its ``scalar`` array rather
#: than ``moist``.  WRF's end-of-step ``spec_bdy_final`` forces a
#: scalar-array species back onto its boundary value only on a NESTED domain
#: (solve_em.F ``scalar_species_bdy_loop_3``).  On a SPECIFIED domain its
#: ring moves by the boundary tendency alone (``spec_bdy_scalar``), which the
#: RK scalar update integrates onto the table.  Water vapour is a moist-array
#: species and is forced back on both kinds of domain.  A further supplied
#: scalar joins this tuple when WRF keeps it in the scalar array.
SCALAR_ARRAY_BOUNDARY_FIELDS = AEROSOL_BOUNDARY_FIELDS


def external_scalar_fields(cfg, *, aerosol_from_input: bool | None = None):
    """Ordered scalar inventory, with an optional resolved ingest decision.

    The decision records whether initialization actually read aerosols. It
    must not be inferred from their values: a supplied zero is still data.
    """
    if not cfg.moist:
        return ()
    if aerosol_from_input is None:
        aerosol_from_input = (
            int(getattr(cfg, "aer_init_opt", 0)) > 0
            or getattr(cfg, "mp28_aerosol_source", "auto") == "climatology")
    if int(cfg.mp_physics) == 28 and aerosol_from_input:
        return ("qv", *AEROSOL_BOUNDARY_FIELDS)
    return ("qv",)


def potential_external_scalar_fields(cfg):
    """Cold pricing includes an auto-resolved aerosol dataset, if available.

    No filesystem probe or dataset read is performed by the estimator. An
    auto selection which falls back to synthetic profiles conservatively
    retains the two possible aerosol rows in its allowance.
    """
    return external_scalar_fields(
        cfg, aerosol_from_input=(
            int(getattr(cfg, "aer_init_opt", 0)) > 0
            or getattr(cfg, "mp28_aerosol_source", "auto") != "synthetic"))
