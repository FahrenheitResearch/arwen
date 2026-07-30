/// Configuration and URL generation for the GFS (Global Forecast System) model.
///
/// GFS is a global 0.25-degree model run every 6 hours (00/06/12/18z)
/// with forecasts out to 384 hours.
pub struct GfsConfig;

impl GfsConfig {
    /// Base URL for GFS GRIB2 files on the AWS Open Data bucket (0.25 degree).
    ///
    /// - `date`: format `"YYYYMMDD"` (e.g. `"20260310"`)
    /// - `hour`: model initialization hour (0, 6, 12, 18)
    /// - `fhour`: forecast hour (0-384)
    pub fn aws_url(date: &str, hour: u32, fhour: u32) -> String {
        format!(
            "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{}/{:02}/atmos/gfs.t{:02}z.pgrb2.0p25.f{:03}",
            date, hour, hour, fhour
        )
    }

    /// IDX file URL (GRIB2 URL + `.idx`).
    pub fn idx_url(date: &str, hour: u32, fhour: u32) -> String {
        format!("{}.idx", Self::aws_url(date, hour, fhour))
    }

    /// NOMADS URL (NCEP operational server).
    pub fn nomads_url(date: &str, hour: u32, fhour: u32) -> String {
        format!(
            "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{}/{:02}/atmos/gfs.t{:02}z.pgrb2.0p25.f{:03}",
            date, hour, hour, fhour
        )
    }

    // --- Grid specifications (0.25 degree global) ---

    pub fn grid_nx() -> u32 { 1440 }
    pub fn grid_ny() -> u32 { 721 }
    pub fn grid_dx() -> f64 { 0.25 } // degrees
    pub fn grid_dy() -> f64 { 0.25 }

    // --- Common variable patterns for .idx matching ---

    pub fn sfc_temp_2m() -> &'static str { "TMP:2 m above ground" }
    pub fn sfc_dewpoint_2m() -> &'static str { "DPT:2 m above ground" }
    pub fn sfc_rh_2m() -> &'static str { "RH:2 m above ground" }
    pub fn sfc_u_wind_10m() -> &'static str { "UGRD:10 m above ground" }
    pub fn sfc_v_wind_10m() -> &'static str { "VGRD:10 m above ground" }
    pub fn sfc_gust() -> &'static str { "GUST:surface" }
    pub fn sfc_mslp() -> &'static str { "PRMSL:mean sea level" }
    pub fn sfc_pressure() -> &'static str { "PRES:surface" }
    pub fn sfc_cape() -> &'static str { "CAPE:surface" }
    pub fn sfc_cin() -> &'static str { "CIN:surface" }
    pub fn sfc_precip() -> &'static str { "APCP:surface" }
    pub fn sfc_visibility() -> &'static str { "VIS:surface" }
    pub fn sfc_hgt() -> &'static str { "HGT:surface" }

    /// Build a pattern for a variable on a pressure level.
    pub fn prs_var(var: &str, level_mb: u32) -> String {
        format!("{}:{} mb", var, level_mb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aws_url_format() {
        let url = GfsConfig::aws_url("20260310", 6, 24);
        assert_eq!(
            url,
            "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20260310/06/atmos/gfs.t06z.pgrb2.0p25.f024"
        );
    }

    #[test]
    fn test_aws_url_hour_zero() {
        let url = GfsConfig::aws_url("20260310", 0, 0);
        assert!(url.contains("/00/atmos/gfs.t00z"));
        assert!(url.ends_with(".f000"));
    }

    #[test]
    fn test_idx_url() {
        let url = GfsConfig::idx_url("20260310", 12, 48);
        assert!(url.ends_with(".f048.idx"));
    }

    #[test]
    fn test_nomads_url_format() {
        let url = GfsConfig::nomads_url("20260310", 18, 384);
        assert!(url.starts_with("https://nomads.ncep.noaa.gov/"));
        assert!(url.contains("gfs.20260310/18/atmos"));
        assert!(url.ends_with(".f384"));
    }

    #[test]
    fn test_grid_specs() {
        assert_eq!(GfsConfig::grid_nx(), 1440);
        assert_eq!(GfsConfig::grid_ny(), 721);
        assert_eq!(GfsConfig::grid_dx(), 0.25);
    }

    #[test]
    fn test_prs_var() {
        assert_eq!(GfsConfig::prs_var("TMP", 850), "TMP:850 mb");
    }

    #[test]
    fn test_variable_patterns() {
        assert_eq!(GfsConfig::sfc_temp_2m(), "TMP:2 m above ground");
        assert_eq!(GfsConfig::sfc_mslp(), "PRMSL:mean sea level");
    }
}
