//! The WOOF post catalog (`woof-post/v1`): every GRIB2 message the exporter
//! can write, with its WMO FM 92 identity (discipline, category, parameter,
//! fixed surfaces) and the method string the manifest records.
//!
//! Identities come from the WOOF clean-room post specification (sections 4
//! to 8) and WMO-No. 306 code tables 4.1, 4.2 and 4.5; the local parameters
//! and level types (192+, 200, 204, 214, 215, 224, 234) are the NCEP local
//! entries the specification names [R10].  Method strings cite the
//! specification section and its public source.

use rw_post::surface::SurfaceField;

/// Which file of a frame a row lands in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileKind {
    /// `wrfsfc_dNN_<valid>.grib2`: surface, column and single-level fields.
    Surface,
    /// `wrfprs_dNN_<valid>.grib2`: the pressure-level block.
    Pressure,
}

/// Where a row's plane comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// Group A surface plane.
    A(SurfaceField),
    /// Group B column planes.
    Pwat,
    Mslp,
    MapsMslp,
    /// Group B pressure-level field, index into `rw_post::plev::PLEV_FIELDS`.
    Plev(usize),
    /// Group C plane, by name in `rw_post::severe::FIELDS`.
    C(&'static str),
    /// Group E plane, by name in `rw_post::group_e::OUT_NAMES`.
    E(&'static str),
    /// Group D plane (reflectivity, cloud, visibility), by name.  Filled by
    /// [`crate::group_d`] from `rw_post::group_d`.
    D(&'static str),
    /// History copies outside the held-out set (spec section 0).
    U10,
    V10,
    /// Total precipitation since the simulation start (RAINNC + RAINC with
    /// the history's bucket counters), product template 4.8.
    Apcp,
    /// An extrema row, index into [`EXTREMA`]: a maximum or minimum over
    /// the request's window (product template 4.8).
    Extreme(usize),
}

/// One catalog row.
#[derive(Clone, Copy, Debug)]
pub struct Row {
    pub id: &'static str,
    pub file: FileKind,
    pub source: Source,
    pub discipline: u8,
    pub category: u8,
    pub number: u8,
    /// First fixed surface (code table 4.5 type, value in the type's unit).
    /// For pressure-level rows the value is filled per level.
    pub level: (u8, f64),
    /// Second fixed surface, for layers.
    pub second: Option<(u8, f64)>,
    pub units: &'static str,
    /// Grid-relative wind component (u = Some(true), v = Some(false)).
    pub wind: Option<bool>,
    pub method: &'static str,
}

const fn row(
    id: &'static str,
    source: Source,
    dcn: (u8, u8, u8),
    level: (u8, f64),
    second: Option<(u8, f64)>,
    units: &'static str,
    method: &'static str,
) -> Row {
    Row { id, file: FileKind::Surface, source, discipline: dcn.0, category: dcn.1, number: dcn.2, level, second, units, wind: None, method }
}

const fn wind_row(mut r: Row, u: bool) -> Row {
    r.wind = Some(u);
    r
}

/// Rows after Group A's 24 (whose identities live with their kernels in
/// `rw_post::surface::SurfaceField::PRODUCTS`).
pub const ROWS: &[Row] = &[
    // ---- history copies (out of the held-out scope, spec section 0)
    wind_row(row("u10", Source::U10, (0, 2, 2), (103, 10.0), None, "m s-1", "U10, the surface layer scheme's 10 m wind (history copy)"), true),
    wind_row(row("v10", Source::V10, (0, 2, 3), (103, 10.0), None, "m s-1", "V10, the surface layer scheme's 10 m wind (history copy)"), false),
    row("total_precipitation", Source::Apcp, (0, 1, 8), (1, 0.0), None, "kg m-2", "RAINNC + RAINC plus the bucket counters times BUCKET_MM, accumulated from the simulation start (product template 4.8)"),
    // ---- Group B (spec 5.2, 5.3)
    row("mslp", Source::Mslp, (0, 3, 1), (101, 0.0), None, "Pa", "SPEC 5.2 D6: NMC reduction of Wea. Forecasting 13 (1998) 833 and Mon. Wea. Rev. 123 (1995) 59 from ground pressure and height"),
    row("maps_mslp", Source::MapsMslp, (0, 3, 198), (101, 0.0), None, "Pa", "SPEC 5.2: MAPS reduction of Mon. Wea. Rev. 118 (1990) 2099 from the 700 hPa temperature, 1-2-1 smoothing of Rev. Geophys. 8 (1970) 359 (D9)"),
    row("pwat", Source::Pwat, (0, 1, 3), (200, 0.0), None, "kg m-2", "SPEC 5.3: (1/g) sum of q dp over model layers (AMS Glossary, precipitable water)"),
    // ---- Group C (spec 6)
    row("sbcape", Source::C("sbcape"), (0, 7, 6), (1, 0.0), None, "J kg-1", "SPEC 6.1-6.2: surface parcel at shelter pressure (D10), Bolton (1980) eq. 43 pseudoadiabat by Newton iteration (MWR 136 (2008) 2764), virtual temperature buoyancy (WAF 9 (1994) 625)"),
    row("sbcin", Source::C("sbcin"), (0, 7, 7), (1, 0.0), None, "J kg-1", "SPEC 6.1-6.2: surface parcel CIN, negative, 0 with no LFC (D12)"),
    row("mlcape", Source::C("mlcape"), (0, 7, 6), (108, 9000.0), Some((108, 0.0)), "J kg-1", "SPEC 6.2: 90 hPa mixed-layer parcel (WAF 17 (2002) 885, D11)"),
    row("mlcin", Source::C("mlcin"), (0, 7, 7), (108, 9000.0), Some((108, 0.0)), "J kg-1", "SPEC 6.2: 90 hPa mixed-layer parcel CIN (D11, D12)"),
    row("mucape", Source::C("mucape"), (0, 7, 6), (108, 30000.0), Some((108, 0.0)), "J kg-1", "SPEC 6.2: most-unstable parcel, maximum theta-e in the lowest 300 hPa (SPC)"),
    row("mucin", Source::C("mucin"), (0, 7, 7), (108, 30000.0), Some((108, 0.0)), "J kg-1", "SPEC 6.2: most-unstable parcel CIN (D12)"),
    row("cape_best180", Source::C("cape_best180"), (0, 7, 6), (108, 18000.0), Some((108, 0.0)), "J kg-1", "SPEC 6.2 D13: six 30 hPa layer parcels in the lowest 180 hPa, maximum theta-e lifted"),
    row("cin_best180", Source::C("cin_best180"), (0, 7, 7), (108, 18000.0), Some((108, 0.0)), "J kg-1", "SPEC 6.2 D13: best-180 parcel CIN"),
    row("lcl_height", Source::C("lcl_height"), (0, 3, 5), (5, 0.0), None, "gpm", "SPEC 6.2: lowest 30 hPa layer parcel, Bolton (1980) eq. 21 LCL, height above ground"),
    wind_row(row("storm_motion_u", Source::C("storm_motion_u"), (0, 2, 27), (103, 0.0), Some((103, 6000.0)), "m s-1", "SPEC 6.3, RULINGS 2: Bunkers internal dynamics (WAF 15 (2000) 61), 0-6 km height-weighted mean wind, 7.5 m s-1 deviation"), true),
    wind_row(row("storm_motion_v", Source::C("storm_motion_v"), (0, 2, 28), (103, 0.0), Some((103, 6000.0)), "m s-1", "SPEC 6.3, RULINGS 2: Bunkers internal dynamics right mover"), false),
    row("srh_0_1km", Source::C("srh_0_1km"), (0, 7, 8), (103, 1000.0), Some((103, 0.0)), "m2 s-2", "SPEC 6.3: storm-relative helicity (16th Conf. Severe Local Storms (1990) 588) from 10 m to 1 km, right mover"),
    row("srh_0_3km", Source::C("srh_0_3km"), (0, 7, 8), (103, 3000.0), Some((103, 0.0)), "m2 s-2", "SPEC 6.3: storm-relative helicity from 10 m to 3 km, right mover"),
    wind_row(row("shear_u_0_1km", Source::C("shear_u_0_1km"), (0, 192, 3), (103, 1000.0), Some((103, 0.0)), "m s-1", "SPEC 6.3 D14: 10 m to 1 km vector wind difference"), true),
    wind_row(row("shear_v_0_1km", Source::C("shear_v_0_1km"), (0, 192, 4), (103, 1000.0), Some((103, 0.0)), "m s-1", "SPEC 6.3 D14: 10 m to 1 km vector wind difference"), false),
    wind_row(row("shear_u_0_6km", Source::C("shear_u_0_6km"), (0, 192, 3), (103, 6000.0), Some((103, 0.0)), "m s-1", "SPEC 6.3, RULINGS 3: 10 m to 6 km AGL vector wind difference (SPC)"), true),
    wind_row(row("shear_v_0_6km", Source::C("shear_v_0_6km"), (0, 192, 4), (103, 6000.0), Some((103, 0.0)), "m s-1", "SPEC 6.3, RULINGS 3: 10 m to 6 km AGL vector wind difference (SPC)"), false),
    row("bulk_shear_0_1km", Source::C("bulk_shear_0_1km"), (0, 192, 1), (103, 1000.0), Some((103, 0.0)), "m s-1", "SPEC 6.3: magnitude of the 0-1 km shear vector"),
    row("bulk_shear_0_6km", Source::C("bulk_shear_0_6km"), (0, 192, 2), (103, 6000.0), Some((103, 0.0)), "m s-1", "SPEC 6.3: magnitude of the 0-6 km shear vector"),
    row("stp", Source::C("stp"), (0, 7, 211), (14, 0.0), None, "1", "SPEC 6.3, RULINGS 4: fixed-layer STP (WAF 18 (2003) 1243, WAF 27 (2012) 1136, SPC), floored at 0"),
    row("ehi_0_1km", Source::C("ehi_0_1km"), (0, 7, 9), (103, 1000.0), Some((103, 0.0)), "1", "SPEC 6.3 D16: sbCAPE x SRH 0-1 km / 160000 (SHARP v1.50, NWS 1991)"),
    // ---- Group D (spec 7): rows reserved, planes arrive with the Rust port
    row("composite_reflectivity", Source::D("composite_reflectivity"), (0, 16, 196), (200, 0.0), None, "dBZ", "SPEC 7.1: column maximum of the reflectivity volume (AMS Glossary)"),
    row("reflectivity_1km", Source::D("reflectivity_1km"), (0, 16, 195), (103, 1000.0), None, "dBZ", "SPEC 7.1, RULINGS 5: linear Z interpolated to 1000 m AGL, then dBZ"),
    row("low_cloud", Source::D("low_cloud"), (0, 6, 3), (214, 0.0), None, "%", "SPEC 7.2, RULINGS 1: maximum cloud fraction in the ISCCP low layer (BAMS 80 (1999) 2261)"),
    row("mid_cloud", Source::D("mid_cloud"), (0, 6, 4), (224, 0.0), None, "%", "SPEC 7.2, RULINGS 1: maximum cloud fraction in the ISCCP middle layer"),
    row("high_cloud", Source::D("high_cloud"), (0, 6, 5), (234, 0.0), None, "%", "SPEC 7.2, RULINGS 1: maximum cloud fraction in the ISCCP high layer"),
    row("cloud_base", Source::D("cloud_base"), (0, 3, 5), (2, 0.0), None, "gpm", "SPEC 7.3 D20: lowest level with cloud fraction > 0.01 and condensate > 1e-6 kg kg-1 (ECMWF 228023)"),
    row("cloud_top", Source::D("cloud_top"), (0, 3, 5), (3, 0.0), None, "gpm", "SPEC 7.3 D20: highest level with the same thresholds"),
    row("cloud_ceiling", Source::D("cloud_ceiling"), (0, 3, 5), (215, 0.0), None, "gpm", "SPEC 7.3 D19: lowest height where cloud fraction reaches 0.5 (FMH-1, ECMWF ceiling)"),
    row("visibility", Source::D("visibility"), (0, 19, 0), (1, 0.0), None, "m", "SPEC 7.4, RULINGS 6 and 10: hydrometeor extinction (JAM 38 (1999) 385), FRAM-L RH haze clear-air baseline (BAMS 90 (2009) 341), 550 nm aerosol extinction where the history carries it (the frame's group_d_sources.aerosol says which), 5 percent contrast meteorological optical range (WMO-No. 8)"),
    row("simulated_ir", Source::D("simulated_ir"), (0, 192, 0), (8, 0.0), None, "K", rw_post::group_d::SIMULATED_IR_METHOD),
    // ---- Group E (spec 8)
    row("pbl_height", Source::E("pbl_height"), (0, 3, 18), (1, 0.0), None, "m", "SPEC 8: bulk Richardson number 0.25 (BLM 81 (1996) 245) with b u*^2, b = 100 (D23)"),
    row("gust", Source::E("gust"), (0, 2, 22), (1, 0.0), None, "m s-1", "SPEC 8 D22, RULINGS 7 (ASOS-scored): mixed-layer TKE gust (MWR 129 (2001) 5) where the history carries TKE, else surface-layer similarity gust (ECMWF IFS Part IV, BLM 11 (1977) 355); the frame's gust record says which"),
    wind_row(row("u80", Source::E("u80"), (0, 2, 2), (103, 80.0), None, "m s-1", "SPEC 8 D25: linear in height between bracketing mass levels, 10 m wind below the lowest level"), true),
    wind_row(row("v80", Source::E("v80"), (0, 2, 3), (103, 80.0), None, "m s-1", "SPEC 8 D25: linear in height between bracketing mass levels"), false),
    row("freezing_height", Source::E("freezing_height"), (0, 3, 5), (4, 0.0), None, "gpm", "SPEC 8: lowest warm-to-cold 273.15 K crossing, linear in height (AMS Glossary, freezing level)"),
    row("freezing_pressure", Source::E("freezing_pressure"), (0, 3, 0), (4, 0.0), None, "Pa", "SPEC 8: pressure of that crossing, linear in ln p"),
    row("highest_freezing_height", Source::E("highest_freezing_height"), (0, 3, 5), (204, 0.0), None, "gpm", "SPEC 8 D24, RULINGS 8: highest crossing below the WMO lapse-rate tropopause (500-50 hPa search)"),
    row("highest_freezing_pressure", Source::E("highest_freezing_pressure"), (0, 3, 0), (204, 0.0), None, "Pa", "SPEC 8 D24, RULINGS 8: pressure of that crossing"),
    row("minus10_height", Source::E("minus10_height"), (0, 3, 5), (20, 263.0), None, "gpm", "SPEC 8 D24: highest 263.15 K crossing below the tropopause"),
    row("minus20_height", Source::E("minus20_height"), (0, 3, 5), (20, 253.0), None, "gpm", "SPEC 8 D24: highest 253.15 K crossing below the tropopause"),
];

/// The pressure-level rows (spec 5.1), level value filled per level (Pa).
pub const PLEV_ROWS: [Row; 6] = [
    Row { id: "height", file: FileKind::Pressure, source: Source::Plev(0), discipline: 0, category: 3, number: 5, level: (100, 0.0), second: None, units: "gpm", wind: None, method: "SPEC 5.1: hypsometric from the bracketing interface with the layer-mean virtual temperature (NCAR/TN-396); below ground per [R12] and the membrane reduction (D8)" },
    Row { id: "temperature", file: FileKind::Pressure, source: Source::Plev(1), discipline: 0, category: 0, number: 0, level: (100, 0.0), second: None, units: "K", wind: None, method: "SPEC 5.1: linear in ln p between bracketing mass levels; below ground 6.5 K km-1 (ICAO, NCAR/TN-396)" },
    Row { id: "dewpoint", file: FileKind::Pressure, source: Source::Plev(2), discipline: 0, category: 0, number: 6, level: (100, 0.0), second: None, units: "K", wind: None, method: "SPEC 5.1: AERK Magnus dewpoint of the interpolated q (JAM 35 (1996) 601)" },
    Row { id: "rh", file: FileKind::Pressure, source: Source::Plev(3), discipline: 0, category: 1, number: 1, level: (100, 0.0), second: None, units: "%", wind: None, method: "SPEC 5.1: RH over water from the interpolated q and T (WMO-No. 8); lowest level held below ground (D7)" },
    Row { id: "u", file: FileKind::Pressure, source: Source::Plev(4), discipline: 0, category: 2, number: 2, level: (100, 0.0), second: None, units: "m s-1", wind: Some(true), method: "SPEC 5.1: linear in ln p; lowest level held below ground (D7)" },
    Row { id: "v", file: FileKind::Pressure, source: Source::Plev(5), discipline: 0, category: 2, number: 3, level: (100, 0.0), second: None, units: "m s-1", wind: Some(false), method: "SPEC 5.1: linear in ln p; lowest level held below ground (D7)" },
];

/// Group A rows, built from the identities rw-post carries beside its kernels.
pub fn group_a_rows() -> Vec<Row> {
    SurfaceField::PRODUCTS
        .iter()
        .map(|s| Row {
            id: s.id,
            file: FileKind::Surface,
            source: Source::A(s.field),
            discipline: s.discipline,
            category: s.category,
            number: s.number,
            level: (s.level_type, s.level_value),
            second: None,
            units: s.units,
            wind: None,
            method: s.method,
        })
        .collect()
}

/// Every surface-file row in write order: Group A, then [`ROWS`].
pub fn surface_rows() -> Vec<Row> {
    let mut rows = group_a_rows();
    rows.extend_from_slice(ROWS);
    rows
}

/// `d.c.n` text of a row.
pub fn grib_code(r: &Row) -> String {
    format!("{}.{}.{}", r.discipline, r.category, r.number)
}

/// Level text, for example `103/1000-103/0`.
pub fn level_text(r: &Row) -> String {
    let one = |(t, v): (u8, f64)| format!("{t}/{v}");
    match r.second {
        Some(s) => format!("{}-{}", one(r.level), one(s)),
        None => one(r.level),
    }
}

/// The default pressure levels, hPa: 100 through 1000 by 25 (spec 5.1).
pub fn default_levels_hpa() -> Vec<u32> {
    (100..=1000).step_by(25).collect()
}

/// One extrema row: a running extreme the history carries under its WRF
/// Registry name (reset at every history write, so each frame holds the
/// extreme since the previous one), written as a maximum or minimum over
/// the request's `extrema_interval_seconds` window (product template 4.8,
/// statistical process from WMO code table 4.10).  Identities are the NCEP
/// local "hourly maximum/minimum" parameters of [R10] (0.7.199, 0.7.200)
/// and WMO wind speed 0.2.1; the template 4.8 time range carries the
/// window, so a 15 min or 3 h window is labelled as such.
#[derive(Clone, Copy, Debug)]
pub struct Extreme {
    pub row: Row,
    /// History variable (WRF Registry name).
    pub carrier: &'static str,
    /// Statistical process (code table 4.10): 2 maximum, 3 minimum.
    pub process: u8,
}

/// Statistical process: maximum (code table 4.10).
pub const PROCESS_MAX: u8 = 2;
/// Statistical process: minimum (code table 4.10).
pub const PROCESS_MIN: u8 = 3;

const fn extreme(id: &'static str, index: usize, carrier: &'static str, dcn: (u8, u8, u8), top: f64, bottom: f64, units: &'static str, process: u8, method: &'static str) -> Extreme {
    let second = if bottom < 0.0 { None } else { Some((103, bottom)) };
    Extreme { row: row(id, Source::Extreme(index), dcn, (103, top), second, units, method), carrier, process }
}

/// The extrema rows, written only when the request sets
/// `extrema_interval_seconds` and only for carriers the history has.
pub const EXTREMA: [Extreme; 9] = [
    extreme("max_updraft_helicity_2_5km", 0, "UP_HELI_MAX", (0, 7, 199), 5000.0, 2000.0, "m2 s-2", PROCESS_MAX,
        "maximum over the window of the history's UP_HELI_MAX (WRF nwp_diagnostics: w times vertical vorticity integrated 2-5 km AGL, running maximum between history writes)"),
    extreme("min_updraft_helicity_2_5km", 1, "UP_HELI_MIN", (0, 7, 200), 5000.0, 2000.0, "m2 s-2", PROCESS_MIN,
        "minimum over the window of the history's UP_HELI_MIN (2-5 km AGL running minimum)"),
    extreme("max_updraft_helicity_0_3km", 2, "UP_HELI_MAX03", (0, 7, 199), 3000.0, 0.0, "m2 s-2", PROCESS_MAX,
        "maximum over the window of the history's UP_HELI_MAX03 (0-3 km AGL running maximum)"),
    extreme("min_updraft_helicity_0_3km", 3, "UP_HELI_MIN03", (0, 7, 200), 3000.0, 0.0, "m2 s-2", PROCESS_MIN,
        "minimum over the window of the history's UP_HELI_MIN03 (0-3 km AGL running minimum)"),
    extreme("max_updraft_helicity_0_2km", 4, "UP_HELI_MAX02", (0, 7, 199), 2000.0, 0.0, "m2 s-2", PROCESS_MAX,
        "maximum over the window of the history's UP_HELI_MAX02 (0-2 km AGL running maximum)"),
    extreme("min_updraft_helicity_0_2km", 5, "UP_HELI_MIN02", (0, 7, 200), 2000.0, 0.0, "m2 s-2", PROCESS_MIN,
        "minimum over the window of the history's UP_HELI_MIN02 (0-2 km AGL running minimum)"),
    extreme("max_updraft_helicity_1_6km", 6, "UP_HELI_MAX16", (0, 7, 199), 6000.0, 1000.0, "m2 s-2", PROCESS_MAX,
        "maximum over the window of the history's UP_HELI_MAX16 (1-6 km AGL running maximum)"),
    extreme("min_updraft_helicity_1_6km", 7, "UP_HELI_MIN16", (0, 7, 200), 6000.0, 1000.0, "m2 s-2", PROCESS_MIN,
        "minimum over the window of the history's UP_HELI_MIN16 (1-6 km AGL running minimum)"),
    extreme("max_wind_speed_10m", 8, "WSPD10MAX", (0, 2, 1), 10.0, -1.0, "m s-1", PROCESS_MAX,
        "maximum over the window of the history's WSPD10MAX (10 m wind speed, running maximum between history writes)"),
];
