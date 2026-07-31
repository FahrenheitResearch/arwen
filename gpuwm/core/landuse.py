"""WRF v4.6.1 cold-start land-use table initialization.

``landuse_init`` in ``phys/module_physics_init.F`` supplies the surface
properties consumed by the first radiation, surface-layer, and Noah calls.
This module parses the authoritative packaged ``LANDUSE.TBL`` and performs
that initialization on host arrays before they are copied to the GPU.

For MODIS-with-lakes input, WRF first records ``LAKEMASK`` from raw category
21, then merges that category into water category 17 before deriving
``IVGTYP``/``XLAND``.  Lakes consequently remain freshwater open-water
columns (``XLAND=2`` and Noah is skipped), while ``LAKEMASK=1`` suppresses
SFCLAY's ocean salinity correction.  This is pinned against the real74 WRF
outputs; treating those cells as Noah land would not be source-faithful.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from gpuwm.core.noah import TBL_DIR, _tokens, load_tables


LANDUSE_COLUMNS = (
    "albedo_percent", "mavail", "emiss", "z0_cm", "thermal_inertia",
    "snow_effect", "soil_heat_capacity")


@dataclass(frozen=True)
class LanduseTable:
    """One WRF LANDUSE.TBL section in summer/winter/category order."""

    lutype: str
    lucats: int
    luseas: int
    values: np.ndarray


@dataclass(frozen=True)
class LanduseInitialization:
    """Host-side fields produced by WRF's real-data cold-start path."""

    season: int
    landmask: np.ndarray
    xland: np.ndarray
    lakemask: np.ndarray
    ivgtyp: np.ndarray
    isltyp: np.ndarray
    snowc: np.ndarray
    pblh: np.ndarray
    ust: np.ndarray
    mavail: np.ndarray
    z0: np.ndarray
    znt: np.ndarray
    albbck: np.ndarray
    albedo: np.ndarray
    embck: np.ndarray
    emiss: np.ndarray


def _nonempty_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


@lru_cache(maxsize=None)
def _load_landuse_table(path: str, mminlu: str) -> LanduseTable:
    lines = _nonempty_lines(Path(path).read_text(encoding="ascii"))
    cursor = 0
    while cursor < len(lines):
        header = _tokens(lines[cursor])
        cursor += 1
        if not header:
            continue
        lutype = header[0].strip("'")
        if cursor >= len(lines):
            break
        shape = _tokens(lines[cursor])
        cursor += 1
        if len(shape) < 2:
            raise ValueError(f"malformed LANDUSE.TBL shape for {lutype!r}")
        lucats, luseas = int(shape[0]), int(shape[1])
        values = np.empty(
            (luseas, lucats, len(LANDUSE_COLUMNS)), dtype=np.float64)
        for season in range(luseas):
            if cursor >= len(lines):
                raise ValueError(f"truncated LANDUSE.TBL section {lutype!r}")
            cursor += 1  # SUMMER/WINTER (or the section's season label)
            for expected_category in range(1, lucats + 1):
                if cursor >= len(lines):
                    raise ValueError(
                        f"truncated LANDUSE.TBL category {expected_category} "
                        f"for {lutype!r}")
                tokens = _tokens(lines[cursor])
                cursor += 1
                if len(tokens) < 1 + len(LANDUSE_COLUMNS):
                    raise ValueError(
                        f"malformed LANDUSE.TBL category record for "
                        f"{lutype!r}: {tokens}")
                category = int(float(tokens[0]))
                if category != expected_category:
                    raise ValueError(
                        f"LANDUSE.TBL category {category} where "
                        f"{expected_category} was required for {lutype!r}")
                values[season, category - 1] = [
                    float(value)
                    for value in tokens[1:1 + len(LANDUSE_COLUMNS)]]
        if lutype == mminlu:
            values.setflags(write=False)
            return LanduseTable(lutype, lucats, luseas, values)
    raise ValueError(f"land-use dataset {mminlu!r} not found in LANDUSE.TBL")


def load_landuse_table(
        mminlu: str = "MODIFIED_IGBP_MODIS_NOAH",
        tbl_dir: Path | None = None) -> LanduseTable:
    """Parse one section using WRF ``landuse_init`` list-directed order."""
    directory = TBL_DIR if tbl_dir is None else Path(tbl_dir)
    return _load_landuse_table(
        str((directory / "LANDUSE.TBL").resolve()), str(mminlu))


def _surface_field(value, shape: tuple[int, int], name: str,
                   dtype) -> np.ndarray:
    try:
        return np.ascontiguousarray(
            np.broadcast_to(np.asarray(value, dtype=dtype), shape))
    except ValueError as exc:
        raise ValueError(f"{name} is not broadcastable to {shape}") from exc


def _integer_categories(value, shape: tuple[int, int], name: str) -> np.ndarray:
    raw = _surface_field(value, shape, name, np.float64)
    rounded = np.rint(raw)
    if not np.all(np.isfinite(raw)) or not np.array_equal(raw, rounded):
        raise ValueError(f"{name} must contain finite integer categories")
    return np.ascontiguousarray(rounded, dtype=np.int32)


def _top_soil_level(value, shape: tuple[int, int], name: str):
    """The first soil level of a ``(nsoil, ny, nx)`` profile, or ``None``."""

    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 3:
        if array.shape[0] < 1:
            raise ValueError(f"{name} carries no soil levels")
        array = array[0]
    return _surface_field(array, shape, name, np.float64)


def _reconcile_landmask_soil_category(
        ivgtyp: np.ndarray, soil: np.ndarray, land: np.ndarray, *,
        iswater: int, isoilwater: int,
        soil_temperature, sst, shape: tuple[int, int]) -> np.ndarray:
    """Transcribe ``dyn_em/module_initialize_real.F:3608-3650``.

    real.exe's final "let us make sure (again) that the landmask and the
    veg/soil categories match" pass, which every ``surface_input_source``
    reaches.  A column whose land/water sense disagrees with its soil
    category is never run as it stands: with a real soil temperature it
    becomes land (``IVGTYP 5``, ``ISLTYP 8`` -- the source's "forcing
    artificial silty clay loam"), with a real SST it becomes water, and
    with neither WRF calls ``wrf_error_fatal('mismatch_landmask_ivgtyp')``.

    RUC LSM relies on this and never re-verifies it.  ``soilvegin``
    (``phys/module_sf_ruclsm.F:6973``) has no ``else`` for ``isltyp == 14``,
    so a land column carrying water soil leaves it with ``ref == qmin == 0``
    and ``:913`` evaluates ``0./0.`` into ``MAVAIL``.  Reconciling here --
    where real.exe reconciles -- is what keeps that unreachable, rather
    than clamping a NaN downstream where WRF has no such clamp.

    Only the land-carrying-water-soil arm is reachable from this function:
    ``land`` is derived from ``IVGTYP`` by the caller and water columns have
    already had ``ISLTYP`` forced to ``isoilwater``.  The water arm is
    transcribed anyway, because the source states both.

    Returns the recomputed land mask; ``ivgtyp`` and ``soil`` are updated
    in place.
    """

    mismatch = ((land & (soil == int(isoilwater)))
                | (~land & (soil != int(isoilwater))))
    if not np.any(mismatch):
        return land
    top_soil = _top_soil_level(soil_temperature, shape, "soil_temperature")
    surface_sea = (None if sst is None
                   else _surface_field(sst, shape, "sst", np.float64))
    warm_soil = (np.zeros(shape, dtype=bool) if top_soil is None
                 else np.asarray(top_soil > 1.0))
    warm_sea = (np.zeros(shape, dtype=bool) if surface_sea is None
                else np.asarray(surface_sea > 1.0))
    to_land = mismatch & warm_soil
    to_water = mismatch & ~warm_soil & warm_sea
    unresolved = mismatch & ~to_land & ~to_water
    if np.any(unresolved):
        rows, columns = np.nonzero(unresolved)
        j, i = int(rows[0]), int(columns[0])
        raise ValueError(
            "mismatch_landmask_ivgtyp: "
            f"{int(unresolved.sum())} column(s) disagree between land mask "
            f"and soil category with no soil temperature above 1 K and no "
            f"SST above 1 K to resolve them (first at j={j}, i={i}: "
            f"land={bool(land[j, i])}, IVGTYP={int(ivgtyp[j, i])}, "
            f"ISLTYP={int(soil[j, i])}, ISWATER={int(iswater)}); "
            "dyn_em/module_initialize_real.F:3631-3641 aborts here")
    ivgtyp[to_land] = 5
    soil[to_land] = 8
    ivgtyp[to_water] = int(iswater)
    soil[to_water] = int(isoilwater)
    return ivgtyp != int(iswater)


def initialize_landuse(
        lu_index, *, soil_type, landmask, snow, xice,
        valid_time, cen_lat: float, mminlu: str, iswater: int,
        islake: int, isice: int, isoilwater: int = 14,
        isoilice: int = 16,
        fractional_seaice: bool = False,
        soil_temperature=None, sst=None,
        tbl_dir: Path | None = None) -> LanduseInitialization:
    """Transcribe the WRF real-data/``landuse_init`` cold-start contract.

    ``valid_time`` supplies WRF's one-based integer ``JULDAY``.  The table
    always has a fixed seasonal state for a run: winter is day <105 or >288
    in the Northern Hemisphere, with the selection reversed south of the
    equator.  ``usemonalb`` is deliberately false, matching real74.

    ``soil_temperature`` (a ``(nsoil, ny, nx)`` profile; its top level is
    the one WRF reads) and ``sst`` are the evidence real.exe's final
    landmask/category reconciliation uses -- see
    :func:`_reconcile_landmask_soil_category`.  They are optional because
    not every caller has them; a caller that omits them and then presents a
    mismatched column gets WRF's third arm, the refusal, because there is
    nothing left to decide the column with.

    Timestep-one SMOIS/TSLB/SMCREL initialization over water and sea ice is
    intentionally outside this function and remains a separate porting
    batch.
    """
    raw_lu = np.asarray(lu_index)
    if raw_lu.ndim != 2:
        raise ValueError(f"lu_index must be 2-D, got shape {raw_lu.shape}")
    shape = raw_lu.shape
    raw_lu = _integer_categories(raw_lu, shape, "lu_index")
    soil = _integer_categories(soil_type, shape, "soil_type")
    _surface_field(landmask, shape, "landmask", np.float32)
    snow = _surface_field(snow, shape, "snow", np.float32)
    xice = _surface_field(xice, shape, "xice", np.float32)
    if not np.isfinite(cen_lat):
        raise ValueError("cen_lat must be finite")

    table = load_landuse_table(mminlu, tbl_dir)
    veg = load_tables(mminlu=mminlu, tbl_dir=tbl_dir)
    lake = raw_lu == int(islake)
    ivgtyp = np.where(raw_lu == 0, int(iswater), raw_lu)
    # module_initialize_real.F:2859-2872 merges lake LANDUSEF into water
    # before process_percent_cat_new determines IVGTYP.  LAKEMASK was saved
    # earlier from the unmodified LU_INDEX (:295-306).
    ivgtyp = np.where(lake, int(iswater), ivgtyp).astype(np.int32)
    threshold = np.float32(0.02 if fractional_seaice else 0.5)
    seaice = xice >= threshold
    ivgtyp = np.where(seaice, int(isice), ivgtyp).astype(np.int32)
    land = ivgtyp != int(iswater)
    soil = np.where(~land, int(isoilwater), soil).astype(np.int32)
    # share/module_soil_pre.F:adjust_for_seaice_post assigns category 16
    # before physics_init.  The associated SMOIS/SH2O/TSLB rewrite remains
    # intentionally out of scope here.
    soil = np.where(seaice, int(isoilice), soil).astype(np.int32)
    land = _reconcile_landmask_soil_category(
        ivgtyp, soil, land, iswater=iswater, isoilwater=isoilwater,
        soil_temperature=soil_temperature, sst=sst, shape=shape)

    if np.any(ivgtyp < 1) or np.any(ivgtyp > table.lucats):
        bad = int(ivgtyp[(ivgtyp < 1) | (ivgtyp > table.lucats)][0])
        raise ValueError(
            f"IVGTYP category {bad} is outside LANDUSE.TBL 1..{table.lucats}")
    # Noah VEGPARM has 20 categories for MODIFIED_IGBP_MODIS_NOAH even
    # though LANDUSE.TBL retains LCZ/unassigned slots.  Fail before a device
    # kernel can index an unsupported active land category; raw lake 21 has
    # already been source-faithfully normalized to supported water 17.
    unsupported_noah = land & (ivgtyp > veg.lucats)
    if np.any(unsupported_noah):
        bad = int(ivgtyp[unsupported_noah][0])
        raise ValueError(
            f"active land IVGTYP {bad} exceeds VEGPARM.TBL category count "
            f"{veg.lucats}")

    julday = int(valid_time.timetuple().tm_yday)
    season = 2 if julday < 105 or julday > 288 else 1
    if float(cen_lat) < 0.0:
        season = 3 - season
    if table.luseas == 1:
        season = 1
    if season > table.luseas:
        raise ValueError(
            f"LANDUSE.TBL {mminlu!r} has {table.luseas} seasons, "
            f"cannot select {season}")

    row = table.values[season - 1, ivgtyp - 1]
    albbck = np.asarray(row[..., 0] / 100.0, np.float32)
    mavail = np.asarray(row[..., 1], np.float32)
    embck = np.asarray(row[..., 2], np.float32)
    z0 = np.asarray(row[..., 3] / 100.0, np.float32)
    # SCFX is declared without a season dimension in WRF.  Each seasonal
    # block overwrites it, so the last block read is used in every season.
    snow_effect = np.asarray(
        table.values[table.luseas - 1, ivgtyp - 1, 5], np.float32)
    snowc = np.asarray(snow >= np.float32(10.0), np.float32)
    albedo = np.where(
        snowc > np.float32(0.5),
        albbck * (np.float32(1.0) + snow_effect), albbck).astype(np.float32)
    emiss = embck.copy()
    if np.any(seaice):
        if fractional_seaice:
            albedo = np.where(
                seaice, xice * albbck + (np.float32(1.0) - xice)
                * np.float32(0.08), albedo).astype(np.float32)
            emiss = np.where(
                seaice, xice * embck + (np.float32(1.0) - xice)
                * np.float32(0.98), emiss).astype(np.float32)
        else:
            albedo = np.where(seaice, albbck, albedo).astype(np.float32)
            emiss = np.where(seaice, embck, emiss).astype(np.float32)

    def real(value) -> np.ndarray:
        return np.ascontiguousarray(value, dtype=np.float32)

    return LanduseInitialization(
        season=season,
        landmask=real(land),
        xland=real(np.where(land, 1.0, 2.0)),
        lakemask=real(lake),
        ivgtyp=np.ascontiguousarray(ivgtyp, dtype=np.int32),
        isltyp=np.ascontiguousarray(soil, dtype=np.int32),
        snowc=real(snowc),
        pblh=np.zeros(shape, dtype=np.float32),
        ust=np.full(shape, np.float32(1.0e-4), dtype=np.float32),
        mavail=real(mavail), z0=real(z0), znt=real(z0),
        albbck=real(albbck), albedo=real(albedo),
        embck=real(embck), emiss=real(emiss))


__all__ = ["LANDUSE_COLUMNS", "LanduseInitialization", "LanduseTable",
           "initialize_landuse", "load_landuse_table"]
