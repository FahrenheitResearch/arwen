"""Packed WRF lake state, retaining WRF single-precision restart storage."""

LAKE_STATE_LAYOUT = (
    ("savedtke12d", 1), ("snowdp2d", 1), ("h2osno2d", 1),
    ("snl2d", 1), ("t_grnd2d", 1), ("t_lake3d", 10),
    ("lake_icefrac3d", 10), ("t_soisno3d", 15),
    ("h2osoi_ice3d", 15), ("h2osoi_liq3d", 15),
    ("h2osoi_vol3d", 15), ("z3d", 15), ("dz3d", 15), ("zi3d", 16),
)
LAKE_STATIC_LAYOUT = (
    ("lakedepth2d", 1), ("z_lake3d", 10), ("dz_lake3d", 10),
    ("watsat3d", 10), ("csol3d", 10), ("tkmg3d", 10),
    ("tkdry3d", 10), ("tksatu3d", 10),
)
LAKE_FORCING_LAYOUT = (
    ("t_phy", 1), ("p8w", 2), ("dz8w", 1), ("qvcurr", 1),
    ("u_phy", 1), ("v_phy", 1), ("glw", 1), ("emiss", 1),
    ("rainbl", 1), ("swdown", 1), ("albedo", 1), ("xlat_urb2d", 1),
)
LAKE_OUTPUT_LAYOUT = tuple((name, 1) for name in
                          ("hfx", "lh", "grdflx", "tsk", "qfx", "t2", "th2", "q2", "albedo"))
LAKE_SEED_LAYOUT = tuple((name, 1) for name in
                        ("isltyp", "lake_depth", "tsk", "snow", "xice"))
LAKE_STATE_WORDS = sum(size for _, size in LAKE_STATE_LAYOUT)
LAKE_STATIC_WORDS = sum(size for _, size in LAKE_STATIC_LAYOUT)
LAKE_FORCING_WORDS = sum(size for _, size in LAKE_FORCING_LAYOUT)
LAKE_OUTPUT_WORDS = sum(size for _, size in LAKE_OUTPUT_LAYOUT)
LAKE_DEFAULT_DEPTH = 50.0
LAKE_DEFAULT_USE_DEPTH = 1
LAKE_DEFAULT_MIN_ELEV = 5.0


def layout_offsets(layout):
    """Return stable field slices in a packed field-major array."""
    result = {}
    start = 0
    for name, size in layout:
        result[name] = slice(start, start + size)
        start += size
    return result
