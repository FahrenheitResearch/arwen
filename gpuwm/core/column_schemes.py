"""Device-free metadata shared by column allocation, transport and launch."""
from dataclasses import dataclass

@dataclass(frozen=True)
class ColumnSchemeRow:
    """A single-moment column scheme admitted by one row of metadata.

    Its CUDA unit follows ONE calling convention, which is what lets a row
    stand in for an adapter file: the column kernel takes
    ``theta, <species...>, den, p, pii, dz, <surface...>, effc, effi, effs,
    coef, dt, nz, ny, nx`` and the scheme-native reflectivity kernel takes
    ``<refl species...>, t, p, f32 words, f64 words, refl, nz, ny, nx``.
    Everything scheme-specific is a field here: the state species in kernel
    order, the ``mp_*`` surface accumulators, the level tier, and the
    scheme's init and radar words.  The unit compiles through
    :func:`gpuwm.core.kernels.load_module_unflushed` (no FTZ, no FMA) in
    the default build.
    """

    mp_physics: int
    label: str
    unit: str
    column_entry: str
    refl_entry: str
    #: State mass species after theta, in the column kernel's order.
    species: tuple[str, ...]
    #: State species the reflectivity kernel reads, in its order.
    refl_species: tuple[str, ...]
    #: ``mp_*`` scratch slots of the surface accumulators, kernel order:
    #: total, its per-call increment, the frozen total, its per-call
    #: increment, then the frozen fraction.
    surface: tuple[str, ...]
    tier_define: str
    shallow_tier: int
    level_tier: object
    init_words: object
    radar_words: object
    background_radii: tuple[float, float, float] = (2.49, 4.99, 9.99)
    #: Fixed-template routes with measured whole-composition replay evidence.
    qualified_fixed_routes: tuple[str, ...] = ()


def _wsm5_row() -> ColumnSchemeRow:
    from gpuwm.core.wsm6_constants import (WSM5_SHALLOW_KMAX,
                                           wsm5_init_words, wsm5_level_tier,
                                           wsm5_radar_words)
    return ColumnSchemeRow(
        mp_physics=4, label="WSM5", unit="wsm5",
        column_entry="wsm5_column", refl_entry="wsm5_refl10cm_column",
        species=("qv", "qc", "qi", "qr", "qs"),
        refl_species=("qv", "qr", "qs"),
        surface=("mp_rainnc", "mp_rainncv", "mp_snownc", "mp_snowncv",
                 "mp_sr"),
        tier_define="WSM5_KMAX", shallow_tier=WSM5_SHALLOW_KMAX,
        level_tier=wsm5_level_tier, init_words=wsm5_init_words,
        radar_words=wsm5_radar_words, qualified_fixed_routes=(
            "tools.prepared_single_domain_forecast",
            "tools.prepared_domain_tree_forecast"))


#: ``mp_physics -> row`` for every scheme admitted this way.  WSM5
#: (mp_physics = 4, phys/module_mp_wsm5.F, Registry wsm5scheme: moist
#: qv,qc,qr,qi,qs and state re_cloud,re_ice,re_snow).
COLUMN_SCHEMES: dict[int, ColumnSchemeRow] = {4: _wsm5_row()}
