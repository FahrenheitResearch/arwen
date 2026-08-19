//! GRIB Edition 1 parameter tables.
//!
//! Provides lookups for parameter indicator codes (PDS byte 9) to
//! human-readable names, abbreviations, and SI units.
//!
//! Two lookup surfaces exist:
//!
//! - [`parameter_entry`] is the authoritative one: it consults the
//!   parameter table version the message CITES (PDS byte 4) together
//!   with the originating center, resolves against the correct table
//!   where one is vendored, and fails closed naming version and
//!   parameter where none is.  A GRIB1 message citing ECMWF table 128
//!   used to be decoded against the WMO/NCEP table below silently --
//!   parameter 134 ("Surface pressure" in ECMWF 128) came back as
//!   NCEP's "Sweat index".
//! - `parameter_name` / `parameter_units` / `parameter_abbrev` are the
//!   raw WMO table 2 rows (with NCEP's center-7 extensions above 127),
//!   consulted by `parameter_entry` and kept public for callers that
//!   have already established the message cites that table.

/// One resolved parameter-table row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParameterTableEntry {
    /// Full parameter name.
    pub name: &'static str,
    /// Units string, when the table records one.
    pub units: Option<&'static str>,
    /// Table-native abbreviation (NCEP mnemonic or ECMWF short name),
    /// when the table records one.
    pub abbrev: Option<&'static str>,
}

/// Resolve a parameter against the table the message cites.
///
/// - Table versions 1-3 are the WMO table 2 lineage.  Indicators 1-127
///   are internationally assigned and resolve for any center; 128-254
///   belong to the originating center, and the extension rows this
///   crate carries there are NCEP's (center 7).  Another center's
///   message in that region is refused rather than decoded against
///   NCEP's rows.
/// - Table version 128 from ECMWF (center 98) resolves against the
///   vendored ECMWF table 128 rows; a parameter without a vendored row
///   is refused naming version and parameter.
/// - Any other (center, version) pair is refused: no table for it is
///   vendored, and answering from WMO table 2 would be a wrong-table
///   decode.  Supporting one needs its rows added to this module.
///
/// `Ok(None)` means the cited table is known and genuinely has no row
/// for the indicator (a reserved code), which is different from not
/// knowing the table at all.
pub fn parameter_entry(
    table_version: u8,
    center_id: u8,
    indicator: u8,
) -> crate::Result<Option<ParameterTableEntry>> {
    match table_version {
        1..=3 => {
            if (128..=254).contains(&indicator) && center_id != 7 {
                return Err(crate::GribError::Parse(format!(
                    "GRIB1 parameter {indicator} from center {center_id} in table \
                     version {table_version} sits in the center-defined region \
                     (128-254); the extension rows vendored here are NCEP's \
                     (center 7), and decoding against them would assign a \
                     scientifically false name.  Supporting it needs center \
                     {center_id}'s table rows added to grib1/tables.rs"
                )));
            }
            Ok(wmo_table_2_entry(indicator))
        }
        128 if center_id == 98 => match ecmwf_table_128_entry(indicator) {
            Some(entry) => Ok(Some(entry)),
            None => Err(crate::GribError::Parse(format!(
                "ECMWF table version 128 parameter {indicator} has no vendored \
                 row; decoding it against WMO table 2 would be a wrong-table \
                 decode.  Supporting it needs its ECMWF table 128 row added \
                 to grib1/tables.rs"
            ))),
        },
        _ => Err(crate::GribError::Parse(format!(
            "GRIB1 message cites parameter table version {table_version} from \
             center {center_id}, for which no table is vendored; decoding \
             parameter {indicator} against WMO table 2 would be a wrong-table \
             decode.  Supporting it needs that center's table added to \
             grib1/tables.rs"
        ))),
    }
}

fn wmo_table_2_entry(indicator: u8) -> Option<ParameterTableEntry> {
    parameter_name(indicator).map(|name| ParameterTableEntry {
        name,
        units: parameter_units(indicator),
        abbrev: parameter_abbrev(indicator),
    })
}

/// ECMWF local parameter table 128 (GRIB1), the table ERA5/ERA-20C
/// GRIB1 archives cite.  Rows carried are the ones the campaign data
/// and the renderer's ERA import read; the ECMWF short name is the
/// abbreviation.  An indicator without a row is a fail-closed refusal
/// in [`parameter_entry`], never a fallback to another table.
fn ecmwf_table_128_entry(indicator: u8) -> Option<ParameterTableEntry> {
    let entry = |name, units, abbrev| {
        Some(ParameterTableEntry {
            name,
            units: Some(units),
            abbrev: Some(abbrev),
        })
    };
    match indicator {
        31 => entry("Sea-ice cover", "(0-1)", "ci"),
        34 => entry("Sea surface temperature", "K", "sst"),
        39 => entry("Volumetric soil water layer 1", "m3/m3", "swvl1"),
        40 => entry("Volumetric soil water layer 2", "m3/m3", "swvl2"),
        41 => entry("Volumetric soil water layer 3", "m3/m3", "swvl3"),
        42 => entry("Volumetric soil water layer 4", "m3/m3", "swvl4"),
        59 => entry("Convective available potential energy", "J/kg", "cape"),
        129 => entry("Geopotential", "m2/s2", "z"),
        130 => entry("Temperature", "K", "t"),
        131 => entry("U component of wind", "m/s", "u"),
        132 => entry("V component of wind", "m/s", "v"),
        133 => entry("Specific humidity", "kg/kg", "q"),
        134 => entry("Surface pressure", "Pa", "sp"),
        135 => entry("Vertical velocity (pressure)", "Pa/s", "w"),
        136 => entry("Total column water", "kg/m2", "tcw"),
        137 => entry("Total column water vapour", "kg/m2", "tcwv"),
        138 => entry("Relative vorticity", "1/s", "vo"),
        139 => entry("Soil temperature level 1", "K", "stl1"),
        141 => entry("Snow depth (water equivalent)", "m", "sd"),
        142 => entry("Large-scale precipitation", "m", "lsp"),
        143 => entry("Convective precipitation", "m", "cp"),
        144 => entry("Snowfall (water equivalent)", "m", "sf"),
        151 => entry("Mean sea level pressure", "Pa", "msl"),
        152 => entry("Logarithm of surface pressure", "~", "lnsp"),
        155 => entry("Divergence", "1/s", "d"),
        156 => entry("Geopotential height", "gpm", "gh"),
        157 => entry("Relative humidity", "%", "r"),
        159 => entry("Boundary layer height", "m", "blh"),
        164 => entry("Total cloud cover", "(0-1)", "tcc"),
        165 => entry("10 m U wind component", "m/s", "10u"),
        166 => entry("10 m V wind component", "m/s", "10v"),
        167 => entry("2 m temperature", "K", "2t"),
        168 => entry("2 m dewpoint temperature", "K", "2d"),
        170 => entry("Soil temperature level 2", "K", "stl2"),
        172 => entry("Land-sea mask", "(0-1)", "lsm"),
        173 => entry("Surface roughness", "m", "sr"),
        182 => entry("Evaporation (water equivalent)", "m", "e"),
        183 => entry("Soil temperature level 3", "K", "stl3"),
        186 => entry("Low cloud cover", "(0-1)", "lcc"),
        187 => entry("Medium cloud cover", "(0-1)", "mcc"),
        188 => entry("High cloud cover", "(0-1)", "hcc"),
        201 => entry("Maximum 2 m temperature", "K", "mx2t"),
        202 => entry("Minimum 2 m temperature", "K", "mn2t"),
        205 => entry("Runoff", "m", "ro"),
        228 => entry("Total precipitation", "m", "tp"),
        235 => entry("Skin temperature", "K", "skt"),
        236 => entry("Soil temperature level 4", "K", "stl4"),
        238 => entry("Temperature of snow layer", "K", "tsn"),
        243 => entry("Forecast albedo", "(0-1)", "fal"),
        244 => entry("Forecast surface roughness", "m", "fsr"),
        245 => entry("Forecast log of surface roughness for heat", "~", "flsr"),
        _ => None,
    }
}

/// Returns the full name for a WMO table 2 parameter indicator.
///
/// Raw table rows only: indicators 128-254 are NCEP's center-7
/// extensions.  Callers that have not already established the message
/// cites this table must go through [`parameter_entry`].
///
/// Returns `None` for reserved or unrecognized codes.
pub fn parameter_name(indicator: u8) -> Option<&'static str> {
    match indicator {
        1 => Some("Pressure"),
        2 => Some("Pressure reduced to MSL"),
        3 => Some("Pressure tendency"),
        4 => Some("Potential vorticity"),
        5 => Some("ICAO Standard Atmosphere reference height"),
        6 => Some("Geopotential"),
        7 => Some("Geopotential height"),
        8 => Some("Geometric height"),
        9 => Some("Standard deviation of height"),
        10 => Some("Total ozone"),
        11 => Some("Temperature"),
        12 => Some("Virtual temperature"),
        13 => Some("Potential temperature"),
        14 => Some("Pseudo-adiabatic potential temperature"),
        15 => Some("Maximum temperature"),
        16 => Some("Minimum temperature"),
        17 => Some("Dew point temperature"),
        18 => Some("Dew point depression"),
        19 => Some("Lapse rate"),
        20 => Some("Visibility"),
        21 => Some("Radar spectra (1)"),
        22 => Some("Radar spectra (2)"),
        23 => Some("Radar spectra (3)"),
        24 => Some("Parcel lifted index (to 500 hPa)"),
        25 => Some("Temperature anomaly"),
        26 => Some("Pressure anomaly"),
        27 => Some("Geopotential height anomaly"),
        28 => Some("Wave spectra (1)"),
        29 => Some("Wave spectra (2)"),
        30 => Some("Wave spectra (3)"),
        31 => Some("Wind direction"),
        32 => Some("Wind speed"),
        33 => Some("u-component of wind"),
        34 => Some("v-component of wind"),
        35 => Some("Stream function"),
        36 => Some("Velocity potential"),
        37 => Some("Montgomery stream function"),
        38 => Some("Sigma coordinate vertical velocity"),
        39 => Some("Vertical velocity (pressure)"),
        40 => Some("Vertical velocity (geometric)"),
        41 => Some("Absolute vorticity"),
        42 => Some("Absolute divergence"),
        43 => Some("Relative vorticity"),
        44 => Some("Relative divergence"),
        45 => Some("Vertical u-component shear"),
        46 => Some("Vertical v-component shear"),
        47 => Some("Direction of current"),
        48 => Some("Speed of current"),
        49 => Some("u-component of current"),
        50 => Some("v-component of current"),
        51 => Some("Specific humidity"),
        52 => Some("Relative humidity"),
        53 => Some("Humidity mixing ratio"),
        54 => Some("Precipitable water"),
        55 => Some("Vapor pressure"),
        56 => Some("Saturation deficit"),
        57 => Some("Evaporation"),
        58 => Some("Cloud ice"),
        59 => Some("Precipitation rate"),
        60 => Some("Thunderstorm probability"),
        61 => Some("Total precipitation"),
        62 => Some("Large scale precipitation"),
        63 => Some("Convective precipitation"),
        64 => Some("Snowfall rate water equivalent"),
        65 => Some("Water equivalent of accumulated snow depth"),
        66 => Some("Snow depth"),
        67 => Some("Mixed layer depth"),
        68 => Some("Transient thermocline depth"),
        69 => Some("Main thermocline depth"),
        70 => Some("Main thermocline anomaly"),
        71 => Some("Total cloud cover"),
        72 => Some("Convective cloud cover"),
        73 => Some("Low cloud cover"),
        74 => Some("Medium cloud cover"),
        75 => Some("High cloud cover"),
        76 => Some("Cloud water"),
        77 => Some("Best lifted index (to 500 hPa)"),
        78 => Some("Convective snow"),
        79 => Some("Large scale snow"),
        80 => Some("Water temperature"),
        81 => Some("Land cover"),
        82 => Some("Deviation of sea level from mean"),
        83 => Some("Surface roughness"),
        84 => Some("Albedo"),
        85 => Some("Soil temperature"),
        86 => Some("Soil moisture content"),
        87 => Some("Vegetation"),
        88 => Some("Salinity"),
        89 => Some("Density"),
        90 => Some("Water run off"),
        91 => Some("Ice cover"),
        92 => Some("Ice thickness"),
        93 => Some("Direction of ice drift"),
        94 => Some("Speed of ice drift"),
        95 => Some("u-component of ice drift"),
        96 => Some("v-component of ice drift"),
        97 => Some("Ice growth rate"),
        98 => Some("Ice divergence"),
        99 => Some("Snow melt"),
        100 => Some("Significant height of combined wind waves and swell"),
        101 => Some("Direction of wind waves"),
        102 => Some("Significant height of wind waves"),
        103 => Some("Mean period of wind waves"),
        104 => Some("Direction of swell waves"),
        105 => Some("Significant height of swell waves"),
        106 => Some("Mean period of swell waves"),
        107 => Some("Primary wave direction"),
        108 => Some("Primary wave mean period"),
        109 => Some("Secondary wave direction"),
        110 => Some("Secondary wave mean period"),
        111 => Some("Net short-wave radiation flux (surface)"),
        112 => Some("Net long-wave radiation flux (surface)"),
        113 => Some("Net short-wave radiation flux (top of atmosphere)"),
        114 => Some("Net long-wave radiation flux (top of atmosphere)"),
        115 => Some("Long wave radiation flux"),
        116 => Some("Short wave radiation flux"),
        117 => Some("Global radiation flux"),
        118 => Some("Brightness temperature"),
        121 => Some("Latent heat net flux"),
        122 => Some("Sensible heat net flux"),
        123 => Some("Boundary layer dissipation"),
        124 => Some("Momentum flux, u component"),
        125 => Some("Momentum flux, v component"),
        126 => Some("Wind mixing energy"),
        127 => Some("Image data"),
        128 => Some("Mean sea level pressure (MAPS)"),
        130 => Some("Mean sea level pressure (ETA)"),
        131 => Some("Surface lifted index"),
        132 => Some("Best (4 layer) lifted index"),
        133 => Some("K index"),
        134 => Some("Sweat index"),
        135 => Some("Horizontal moisture divergence"),
        136 => Some("Vertical speed shear"),
        137 => Some("3-hr pressure tendency"),
        140 => Some("Categorical rain"),
        141 => Some("Categorical freezing rain"),
        142 => Some("Categorical ice pellets"),
        143 => Some("Categorical snow"),
        144 => Some("Volumetric soil moisture content"),
        145 => Some("Potential evaporation rate"),
        146 => Some("Cloud work function"),
        147 => Some("Zonal flux of gravity wave stress"),
        148 => Some("Meridional flux of gravity wave stress"),
        153 => Some("Cloud water mixing ratio"),
        154 => Some("Ozone mixing ratio"),
        155 => Some("Ground heat flux"),
        156 => Some("Convective inhibition"),
        157 => Some("Convective available potential energy"),
        158 => Some("Turbulent kinetic energy"),
        159 => Some("Condensation pressure of parcel lifted from surface"),
        160 => Some("Clear sky upward solar flux"),
        176 => Some("Latitude"),
        177 => Some("Longitude"),
        190 => Some("Eta coordinate vertical velocity"),
        196 => Some("u-component of storm motion"),
        197 => Some("v-component of storm motion"),
        204 => Some("Downward short wave radiation flux"),
        205 => Some("Downward long wave radiation flux"),
        206 => Some("UV-B downward solar flux"),
        207 => Some("Moisture availability"),
        208 => Some("Exchange coefficient"),
        211 => Some("Upward short wave radiation flux"),
        212 => Some("Upward long wave radiation flux"),
        213 => Some("Amount of non-convective cloud"),
        214 => Some("Precipitation rate"),
        216 => Some("Temperature tendency by all radiation"),
        218 => Some("Precipitation index"),
        221 => Some("Planetary boundary layer height"),
        222 => Some("5-wave geopotential height"),
        223 => Some("Plant canopy surface water"),
        224 => Some("Soil type"),
        225 => Some("Vegetation type"),
        226 => Some("Blackadar mixing length scale"),
        227 => Some("Asymptotic mixing length scale"),
        228 => Some("Potential evaporation"),
        229 => Some("Snow phase-change heat flux"),
        230 => Some("5-wave geopotential height anomaly"),
        234 => Some("Baseflow-groundwater runoff"),
        235 => Some("Storm surface runoff"),
        238 => Some("Snow cover"),
        246 => Some("Rate of water dropping from canopy to ground"),
        247 => Some("Rate of water ascending from soil to canopy"),
        252 => Some("Sunshine duration"),
        253 => Some("Number of soil layers in root zone"),
        255 => Some("Missing"),
        _ => None,
    }
}

/// Returns the SI units string for a WMO table 2 parameter indicator.
pub fn parameter_units(indicator: u8) -> Option<&'static str> {
    match indicator {
        1 => Some("Pa"),
        2 => Some("Pa"),
        3 => Some("Pa/s"),
        4 => Some("K m^2 kg^-1 s^-1"),
        5 => Some("m"),
        6 => Some("m^2/s^2"),
        7 => Some("gpm"),
        8 => Some("m"),
        9 => Some("m"),
        10 => Some("DU"),
        11 | 12 | 13 | 14 | 15 | 16 | 17 | 25 => Some("K"),
        18 => Some("K"),
        19 => Some("K/m"),
        20 => Some("m"),
        24 | 77 => Some("K"),
        26 => Some("Pa"),
        27 => Some("gpm"),
        31 => Some("deg"),
        32 => Some("m/s"),
        33 | 34 => Some("m/s"),
        35 | 36 | 37 => Some("m^2/s"),
        38 => Some("1/s"),
        39 => Some("Pa/s"),
        40 => Some("m/s"),
        41 | 42 | 43 | 44 => Some("1/s"),
        45 | 46 => Some("1/s"),
        47 => Some("deg"),
        48 => Some("m/s"),
        49 | 50 => Some("m/s"),
        51 => Some("kg/kg"),
        52 => Some("%"),
        53 => Some("kg/kg"),
        54 => Some("kg/m^2"),
        55 => Some("Pa"),
        56 => Some("Pa"),
        57 => Some("kg/m^2"),
        58 => Some("kg/m^2"),
        59 => Some("kg/m^2/s"),
        60 => Some("%"),
        61 | 62 | 63 => Some("kg/m^2"),
        64 => Some("kg/m^2/s"),
        65 => Some("kg/m^2"),
        66 => Some("m"),
        67 | 68 | 69 => Some("m"),
        70 => Some("m"),
        71 | 72 | 73 | 74 | 75 => Some("%"),
        76 => Some("kg/m^2"),
        78 | 79 => Some("kg/m^2"),
        80 => Some("K"),
        81 => Some("fraction"),
        82 => Some("m"),
        83 => Some("m"),
        84 => Some("%"),
        85 => Some("K"),
        86 => Some("kg/m^2"),
        87 => Some("%"),
        88 => Some("kg/kg"),
        89 => Some("kg/m^3"),
        90 => Some("kg/m^2"),
        91 => Some("fraction"),
        92 => Some("m"),
        93 => Some("deg"),
        94 => Some("m/s"),
        95 | 96 => Some("m/s"),
        97 => Some("m/s"),
        98 => Some("1/s"),
        99 => Some("kg/m^2"),
        100 | 102 | 105 => Some("m"),
        101 | 104 | 107 | 109 => Some("deg"),
        103 | 106 | 108 | 110 => Some("s"),
        111 | 112 | 113 | 114 | 115 | 116 | 117 => Some("W/m^2"),
        118 => Some("K"),
        121 | 122 => Some("W/m^2"),
        123 => Some("W/m^2"),
        124 | 125 => Some("N/m^2"),
        126 => Some("J"),
        144 => Some("fraction"),
        145 => Some("W/m^2"),
        153 => Some("kg/kg"),
        154 => Some("kg/kg"),
        155 => Some("W/m^2"),
        156 => Some("J/kg"),
        157 => Some("J/kg"),
        158 => Some("J/kg"),
        176 => Some("deg"),
        177 => Some("deg"),
        204 | 205 | 211 | 212 => Some("W/m^2"),
        221 => Some("m"),
        228 => Some("W/m^2"),
        252 => Some("s"),
        _ => None,
    }
}

/// Returns the standard abbreviation for a WMO table 2 parameter indicator.
pub fn parameter_abbrev(indicator: u8) -> Option<&'static str> {
    match indicator {
        1 => Some("PRES"),
        2 => Some("PRMSL"),
        3 => Some("PTEND"),
        4 => Some("PVORT"),
        6 => Some("GP"),
        7 => Some("HGT"),
        8 => Some("DIST"),
        10 => Some("TOZNE"),
        11 => Some("TMP"),
        12 => Some("VTMP"),
        13 => Some("POT"),
        14 => Some("EPOT"),
        15 => Some("TMAX"),
        16 => Some("TMIN"),
        17 => Some("DPT"),
        18 => Some("DEPR"),
        19 => Some("LAPR"),
        20 => Some("VIS"),
        24 => Some("PLI"),
        25 => Some("TMPA"),
        26 => Some("PRESA"),
        27 => Some("GPA"),
        31 => Some("WDIR"),
        32 => Some("WIND"),
        33 => Some("UGRD"),
        34 => Some("VGRD"),
        35 => Some("STRM"),
        36 => Some("VPOT"),
        37 => Some("MNTSF"),
        38 => Some("SGCVV"),
        39 => Some("VVEL"),
        40 => Some("DZDT"),
        41 => Some("ABSV"),
        42 => Some("ABSD"),
        43 => Some("RELV"),
        44 => Some("RELD"),
        45 => Some("VUCSH"),
        46 => Some("VVCSH"),
        51 => Some("SPFH"),
        52 => Some("RH"),
        53 => Some("MIXR"),
        54 => Some("PWAT"),
        55 => Some("VAPP"),
        56 => Some("SATD"),
        57 => Some("EVP"),
        58 => Some("CICE"),
        59 => Some("PRATE"),
        61 => Some("APCP"),
        62 => Some("NCPCP"),
        63 => Some("ACPCP"),
        64 => Some("SRWEQ"),
        65 => Some("WEASD"),
        66 => Some("SNOD"),
        71 => Some("TCDC"),
        72 => Some("CDCON"),
        73 => Some("LCDC"),
        74 => Some("MCDC"),
        75 => Some("HCDC"),
        76 => Some("CWAT"),
        77 => Some("BLI"),
        80 => Some("WTMP"),
        81 => Some("LAND"),
        83 => Some("SFCR"),
        84 => Some("ALBDO"),
        85 => Some("TSOIL"),
        86 => Some("SOILM"),
        87 => Some("VEG"),
        90 => Some("WATR"),
        91 => Some("ICEC"),
        92 => Some("ICETK"),
        100 => Some("HTSGW"),
        111 => Some("NSWRS"),
        112 => Some("NLWRS"),
        113 => Some("NSWRT"),
        114 => Some("NLWRT"),
        117 => Some("GRAD"),
        118 => Some("BRTMP"),
        121 => Some("LHTFL"),
        122 => Some("SHTFL"),
        144 => Some("SOILW"),
        153 => Some("CLWMR"),
        154 => Some("O3MR"),
        155 => Some("GFLUX"),
        156 => Some("CIN"),
        157 => Some("CAPE"),
        158 => Some("TKE"),
        204 => Some("DSWRF"),
        205 => Some("DLWRF"),
        211 => Some("USWRF"),
        212 => Some("ULWRF"),
        221 => Some("HPBL"),
        _ => None,
    }
}

/// Level type indicator (PDS byte 10) description.
///
/// Returns a tuple of (description, units) for the level value.
pub fn level_description(level_type: u8) -> (&'static str, &'static str) {
    match level_type {
        1 => ("Ground or water surface", ""),
        2 => ("Cloud base level", ""),
        3 => ("Level of cloud tops", ""),
        4 => ("Level of 0 deg C isotherm", ""),
        5 => ("Level of adiabatic condensation lifted from surface", ""),
        6 => ("Maximum wind level", ""),
        7 => ("Tropopause", ""),
        8 => ("Nominal top of atmosphere", ""),
        9 => ("Sea bottom", ""),
        20 => ("Isothermal level", "K (1/100)"),
        100 => ("Isobaric surface", "hPa"),
        101 => ("Layer between two isobaric surfaces", "kPa"),
        102 => ("Mean sea level", ""),
        103 => ("Altitude above mean sea level", "m"),
        104 => ("Layer between two altitudes above MSL", "hm"),
        105 => ("Specified height level above ground", "m"),
        106 => (
            "Layer between two specified height levels above ground",
            "hm",
        ),
        107 => ("Sigma level", "sigma (1/10000)"),
        108 => ("Layer between two sigma levels", "sigma (1/100)"),
        109 => ("Hybrid level", ""),
        110 => ("Layer between two hybrid levels", ""),
        111 => ("Depth below land surface", "cm"),
        112 => ("Layer between two depths below land surface", "cm"),
        113 => ("Isentropic (theta) level", "K"),
        114 => ("Layer between two isentropic levels", "K (475-level)"),
        115 => (
            "Level at specified pressure difference from ground to level",
            "hPa",
        ),
        116 => (
            "Layer between two levels at specified pressure difference from ground",
            "hPa",
        ),
        117 => ("Potential vorticity surface", "10^-6 km^2/kg/s"),
        119 => ("Eta level", "eta (1/10000)"),
        120 => ("Layer between two eta levels", "eta (1/100)"),
        121 => (
            "Layer between two isobaric surfaces (high precision)",
            "1100-hPa",
        ),
        125 => ("Specified height level above ground (high precision)", "cm"),
        128 => (
            "Layer between two sigma levels (high precision)",
            "sigma (1.1-level)",
        ),
        141 => (
            "Layer between two isobaric surfaces (mixed precision)",
            "kPa, 1100-hPa",
        ),
        160 => ("Depth below sea level", "m"),
        200 => ("Entire atmosphere (considered as a single layer)", ""),
        201 => ("Entire ocean (considered as a single layer)", ""),
        204 => ("Highest tropospheric freezing level", ""),
        212 => ("Low cloud bottom level", ""),
        213 => ("Low cloud top level", ""),
        214 => ("Low cloud layer", ""),
        222 => ("Middle cloud bottom level", ""),
        223 => ("Middle cloud top level", ""),
        224 => ("Middle cloud layer", ""),
        232 => ("High cloud bottom level", ""),
        233 => ("High cloud top level", ""),
        234 => ("High cloud layer", ""),
        242 => ("Convective cloud bottom level", ""),
        243 => ("Convective cloud top level", ""),
        244 => ("Convective cloud layer", ""),
        _ => ("Unknown level type", ""),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_common_parameters() {
        assert_eq!(parameter_name(11), Some("Temperature"));
        assert_eq!(parameter_units(11), Some("K"));
        assert_eq!(parameter_abbrev(11), Some("TMP"));

        assert_eq!(parameter_name(7), Some("Geopotential height"));
        assert_eq!(parameter_units(7), Some("gpm"));
        assert_eq!(parameter_abbrev(7), Some("HGT"));

        assert_eq!(parameter_name(33), Some("u-component of wind"));
        assert_eq!(parameter_units(33), Some("m/s"));
        assert_eq!(parameter_abbrev(33), Some("UGRD"));

        assert_eq!(parameter_name(52), Some("Relative humidity"));
        assert_eq!(parameter_units(52), Some("%"));

        assert_eq!(parameter_name(61), Some("Total precipitation"));
        assert_eq!(parameter_units(61), Some("kg/m^2"));

        assert_eq!(parameter_name(1), Some("Pressure"));
        assert_eq!(parameter_units(1), Some("Pa"));
    }

    #[test]
    fn test_unknown_parameter() {
        assert_eq!(parameter_name(250), None);
        assert_eq!(parameter_units(250), None);
        assert_eq!(parameter_abbrev(250), None);
    }

    // ---- Version-aware lookup: the table a message CITES is the table ----
    // ---- it is decoded against, never silently WMO/NCEP table 2.      ----

    #[test]
    fn test_ecmwf_table_128_resolves_by_version() {
        // ECMWF (center 98) table 128: parameter 130 is Temperature and
        // parameter 134 is Surface pressure.  Decoded against NCEP table 2
        // these were "Mean sea level pressure (ETA)" and "Sweat index" --
        // the exact wrong-table decode the census flagged, live in the
        // May-1999 ERA5 campaign files.
        let t = parameter_entry(128, 98, 130).unwrap().unwrap();
        assert_eq!(t.name, "Temperature");
        assert_eq!(t.units, Some("K"));
        assert_eq!(t.abbrev, Some("t"));

        let sp = parameter_entry(128, 98, 134).unwrap().unwrap();
        assert_eq!(sp.name, "Surface pressure");
        assert_eq!(sp.units, Some("Pa"));
        assert_eq!(sp.abbrev, Some("sp"));

        let z = parameter_entry(128, 98, 129).unwrap().unwrap();
        assert_eq!(z.name, "Geopotential");
        assert_eq!(z.units, Some("m2/s2"));

        // The ERA5 soil slabs the ingest catalog reads.
        assert_eq!(
            parameter_entry(128, 98, 139).unwrap().unwrap().name,
            "Soil temperature level 1"
        );
        assert_eq!(
            parameter_entry(128, 98, 39).unwrap().unwrap().abbrev,
            Some("swvl1")
        );
    }

    #[test]
    fn test_wmo_international_region_is_center_agnostic() {
        // Indicators 1-127 are internationally assigned in versions 1-3;
        // any originating center resolves them from WMO table 2.
        for (version, center) in [(1u8, 98u8), (2, 7), (3, 34)] {
            let entry = parameter_entry(version, center, 11).unwrap().unwrap();
            assert_eq!(entry.name, "Temperature");
            assert_eq!(entry.abbrev, Some("TMP"));
        }
        // Reserved gaps inside a KNOWN table stay None, not an error.
        assert_eq!(parameter_entry(2, 7, 250).unwrap(), None);
    }

    #[test]
    fn test_ncep_extension_region_is_ncep_only() {
        // 128-254 in versions 1-3 belong to the originating center.  The
        // entries this crate carries there are NCEP's; another center's
        // message must not borrow them.
        let cape = parameter_entry(2, 7, 157).unwrap().unwrap();
        assert_eq!(cape.name, "Convective available potential energy");
        assert_eq!(cape.abbrev, Some("CAPE"));

        let err = parameter_entry(2, 34, 157).unwrap_err().to_string();
        assert!(err.contains("center 34"), "{err}");
        assert!(err.contains("version 2"), "{err}");
        assert!(err.contains("parameter 157"), "{err}");
    }

    #[test]
    fn test_unknown_local_table_fails_closed_naming_version_and_parameter() {
        let err = parameter_entry(200, 98, 250).unwrap_err().to_string();
        assert!(err.contains("version 200"), "{err}");
        assert!(err.contains("parameter 250"), "{err}");

        // NCEP's local table 128 is not ECMWF's and is not vendored.
        assert!(parameter_entry(128, 7, 129).is_err());
    }

    #[test]
    fn test_ecmwf_128_unknown_parameter_fails_closed() {
        // Parameter 11 is not a row this crate carries for ECMWF table
        // 128; answering from WMO table 2 ("Temperature") would be the
        // defect, and None would claim the table has no such row.
        let err = parameter_entry(128, 98, 11).unwrap_err().to_string();
        assert!(err.contains("version 128"), "{err}");
        assert!(err.contains("parameter 11"), "{err}");
    }

    #[test]
    fn test_level_types() {
        let (desc, units) = level_description(100);
        assert_eq!(desc, "Isobaric surface");
        assert_eq!(units, "hPa");

        let (desc, _) = level_description(1);
        assert_eq!(desc, "Ground or water surface");

        let (desc, _) = level_description(200);
        assert_eq!(desc, "Entire atmosphere (considered as a single layer)");
    }
}
