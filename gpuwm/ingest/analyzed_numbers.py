"""WRF metgrid number moments and their exact active Registry targets."""

METGRID_NUMBER_FIELDS = ("QNI", "QNC", "QNR", "QNS", "QNG", "QNH")
_WRF_NAMES = dict(zip(METGRID_NUMBER_FIELDS,
                     ("QNICE", "QNCLOUD", "QNRAIN", "QNSNOW", "QNGRAUPEL", "QNHAIL")))


def metgrid_number_targets(cfg):
    """Resolve P_QN* membership using the shared selected WRF package.

    module_initialize_real.F:1999-2120 interpolates each flagged source
    only when the corresponding P_QN* belongs to num_3d_s. In particular,
    QNC targets P_QNC, never NSSL's distinct P_QNDROP scalar.
    """
    from gpuwm.ingest.wrfinput import _active_moisture_map
    from gpuwm.core.nest_fields import nest_field_kinds
    active = _active_moisture_map(cfg)
    transported = set(nest_field_kinds(cfg))
    return {name: active[wrf] for name, wrf in _WRF_NAMES.items()
            if wrf in active and active[wrf] in transported}
