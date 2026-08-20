//! The field map: every WRF variable a catalogue product actually reads,
//! paired with the MPAS history (or pinned init) variable carrying the same
//! quantity.
//!
//! What this does not do is invent a field the MPAS history does not carry. A
//! WRF variable with no MPAS source is simply absent, its products report
//! missing fields, and the absence is recorded in the emitted file's
//! `MPAS_ABSENT_WRF_FIELDS` attribute. A missing field is an output-stream gap
//! to fix upstream, never something to synthesize here.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rank {
    Surface,
    Mass3d,
    W3d,
    U3d,
    V3d,
    Soil,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldSource {
    History,
    Init,
    /// A value the WRF schema requires that MPAS has no equivalent for, and
    /// whose correct value is a constant, not an interpolation.
    Constant,
}

#[derive(Debug, Clone, Copy)]
pub struct FieldMapping {
    pub wrf_name: &'static str,
    pub mpas_name: Option<&'static str>,
    pub rank: Rank,
    pub units: &'static str,
    pub description: &'static str,
    pub source: FieldSource,
    pub note: &'static str,
}

const fn f(
    wrf_name: &'static str,
    mpas_name: Option<&'static str>,
    rank: Rank,
    units: &'static str,
    description: &'static str,
    source: FieldSource,
    note: &'static str,
) -> FieldMapping {
    FieldMapping {
        wrf_name,
        mpas_name,
        rank,
        units,
        description,
        source,
        note,
    }
}

pub static FIELD_MAP: &[FieldMapping] = &[
    // --- three-dimensional mass fields --------------------------------------
    f("T", Some("theta"), Rank::Mass3d, "K",
      "perturbation potential temperature theta-t0", FieldSource::History,
      "theta - 300 K; MPAS theta is the dry potential temperature"),
    f("PB", Some("pressure"), Rank::Mass3d, "Pa", "BASE STATE PRESSURE", FieldSource::History,
      "MPAS has no base/perturbation split; PB carries the full model pressure so P+PB is exactly it"),
    f("P", None, Rank::Mass3d, "Pa", "perturbation pressure", FieldSource::Constant,
      "identically zero; see PB"),
    f("QVAPOR", Some("qv"), Rank::Mass3d, "kg kg-1", "Water vapor mixing ratio", FieldSource::History, ""),
    f("QCLOUD", Some("qc"), Rank::Mass3d, "kg kg-1", "Cloud water mixing ratio", FieldSource::History, ""),
    f("QRAIN", Some("qr"), Rank::Mass3d, "kg kg-1", "Rain water mixing ratio", FieldSource::History, ""),
    f("QICE", Some("qi"), Rank::Mass3d, "kg kg-1", "Ice mixing ratio", FieldSource::History, ""),
    f("QSNOW", Some("qs"), Rank::Mass3d, "kg kg-1", "Snow mixing ratio", FieldSource::History, ""),
    f("QGRAUP", Some("qg"), Rank::Mass3d, "kg kg-1", "Graupel mixing ratio", FieldSource::History, ""),
    f("U", Some("u_zonal"), Rank::U3d, "m s-1", "x-wind component", FieldSource::History,
      "earth-relative zonal wind, restaggered in x"),
    f("V", Some("v_meridional"), Rank::V3d, "m s-1", "y-wind component", FieldSource::History,
      "earth-relative meridional wind, restaggered in y"),
    f("W", Some("w"), Rank::W3d, "m s-1", "z-wind component", FieldSource::History, ""),
    f("PHB", Some("zgrid"), Rank::W3d, "m2 s-2", "base-state geopotential", FieldSource::Init,
      "g * zgrid from the pinned init file; MPAS is height based"),
    f("PH", None, Rank::W3d, "m2 s-2", "perturbation geopotential", FieldSource::Constant,
      "identically zero; see PHB"),
    // --- surface diagnostics ------------------------------------------------
    f("T2", Some("t2"), Rank::Surface, "K", "TEMP at 2 M", FieldSource::History, ""),
    f("PSFC", Some("surface_pressure"), Rank::Surface, "Pa", "SFC PRESSURE", FieldSource::History, ""),
    f("U10", Some("u10"), Rank::Surface, "m s-1", "U at 10 M", FieldSource::History, ""),
    f("V10", Some("v10"), Rank::Surface, "m s-1", "V at 10 M", FieldSource::History, ""),
    f("TSK", Some("tsk"), Rank::Surface, "K", "SURFACE SKIN TEMPERATURE", FieldSource::History, ""),
    f("HGT", Some("ter"), Rank::Surface, "m", "Terrain Height", FieldSource::History, ""),
    f("PBLH", Some("pblh"), Rank::Surface, "m", "PBL HEIGHT", FieldSource::History, ""),
    f("HFX", Some("hfx"), Rank::Surface, "W m-2", "UPWARD HEAT FLUX AT THE SURFACE", FieldSource::History, ""),
    f("LH", Some("lh"), Rank::Surface, "W m-2", "LATENT HEAT FLUX AT THE SURFACE", FieldSource::History, ""),
    f("QFX", Some("qfx"), Rank::Surface, "kg m-2 s-1", "UPWARD MOISTURE FLUX AT THE SURFACE", FieldSource::History, ""),
    f("SWDOWN", Some("swdown"), Rank::Surface, "W m-2", "DOWNWARD SHORT WAVE FLUX AT GROUND SURFACE", FieldSource::History, ""),
    f("GLW", Some("glw"), Rank::Surface, "W m-2", "DOWNWARD LONG WAVE FLUX AT GROUND SURFACE", FieldSource::History, ""),
    // --- precipitation accumulations ----------------------------------------
    f("RAINC", Some("rainc"), Rank::Surface, "mm", "ACCUMULATED TOTAL CUMULUS PRECIPITATION", FieldSource::History, ""),
    f("RAINNC", Some("rainnc"), Rank::Surface, "mm", "ACCUMULATED TOTAL GRID SCALE PRECIPITATION", FieldSource::History, ""),
    f("SNOWNC", Some("snownc"), Rank::Surface, "mm", "ACCUMULATED TOTAL GRID SCALE SNOW AND ICE", FieldSource::History, ""),
    f("GRAUPELNC", Some("graupelnc"), Rank::Surface, "mm", "ACCUMULATED TOTAL GRID SCALE GRAUPEL", FieldSource::History, ""),
    // --- land surface --------------------------------------------------------
    f("TSLB", Some("tslb"), Rank::Soil, "K", "SOIL TEMPERATURE", FieldSource::History, ""),
    f("SMOIS", Some("smois"), Rank::Soil, "m3 m-3", "SOIL MOISTURE", FieldSource::History, ""),
    f("LANDMASK", Some("landmask"), Rank::Surface, "", "LAND MASK (1 FOR LAND, 0 FOR WATER)", FieldSource::Init, ""),
    f("LU_INDEX", Some("ivgtyp"), Rank::Surface, "", "LAND USE CATEGORY", FieldSource::Init, ""),
    // --- grid geometry the renderer reads ------------------------------------
    f("SINALPHA", None, Rank::Surface, "", "Local sine of map rotation", FieldSource::Constant,
      "0: the emitted U/V/U10/V10 are already earth-relative, so the renderer's grid-to-earth rotation must be the identity"),
    f("COSALPHA", None, Rank::Surface, "", "Local cosine of map rotation", FieldSource::Constant,
      "1; see SINALPHA"),
];

/// WRF variables some catalogue product wants that the port's history stream
/// does not carry. Gaps to file against the output stream, not things to
/// synthesize.
pub static ABSENT_WRF_FIELDS: &[(&str, &str, &str)] = &[
    ("Q2", "2m water-vapour mixing ratio",
     "the port history writes no q2 and the proof receipt sets q2_products_allowed=false; blocks 2m_dewpoint, 2m_relative_humidity and both of their 10m-wind variants"),
    ("REFL_10CM", "native 10 cm radar reflectivity",
     "not in the port history stream; the renderer falls back to its own dbz diagnostic from QRAIN/QSNOW/QGRAUP, so reflectivity products still render but they are the renderer's diagnostic, not the model's"),
    ("UP_HELI_MAX", "maximum updraught helicity",
     "not in the port history stream; blocks composite_reflectivity_uh"),
    ("WSPD10MAX", "maximum 10 m wind speed",
     "not in the port history stream; the gust product has no store selector in any WRF lane anyway"),
    ("OLR", "top-of-atmosphere outgoing longwave",
     "not in the port history stream; blocks the OLR raw-extra plane"),
    ("SST", "sea surface temperature",
     "not in the port history stream (it is in the pinned init file, but a static init value is not a forecast field)"),
    ("RAINSH", "shallow-cumulus precipitation accumulation",
     "the pinned suite runs Grell-Freitas with no separate shallow bucket; total precipitation is RAINC + RAINNC and is complete"),
];

/// The named field sets. A global 0.25 degree window carries four million
/// points per level, so the surface set keeps only what the surface,
/// precipitation, MSLP and precipitable-water products read -- which is the
/// honest content of a coarse overview anyway.
pub fn field_set_allows(field_set: &str, mapping: &FieldMapping) -> bool {
    match field_set {
        "full" => true,
        "surface" => matches!(
            mapping.wrf_name,
            "T" | "P" | "PB" | "QVAPOR" | "PH" | "PHB"
        ) || matches!(mapping.rank, Rank::Surface | Rank::Soil),
        _ => true,
    }
}

pub fn field_set_is_known(field_set: &str) -> bool {
    matches!(field_set, "full" | "surface")
}
