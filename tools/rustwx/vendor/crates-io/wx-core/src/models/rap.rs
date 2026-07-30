/// Configuration and URL generation for the RAP (Rapid Refresh) model.
///
/// RAP runs at 13km over North America, initialized every hour
/// with forecasts out to 21 hours (51 hours for 03/09/15/21z).
pub struct RapConfig;

impl RapConfig {
    /// Base URL for RAP GRIB2 files on the NOMADS server.
    ///
    /// - `date`: format `"YYYYMMDD"` (e.g. `"20260310"`)
    /// - `hour`: model initialization hour (0-23)
    /// - `fhour`: forecast hour (0-21 or 0-51)
    pub fn nomads_url(date: &str, hour: u32, fhour: u32) -> String {
        format!(
            "https://nomads.ncep.noaa.gov/pub/data/nccf/com/rap/prod/rap.{}/rap.t{:02}z.awp130pgrbf{:02}.grib2",
            date, hour, fhour
        )
    }

    /// IDX file URL.
    pub fn idx_url(date: &str, hour: u32, fhour: u32) -> String {
        format!("{}.idx", Self::nomads_url(date, hour, fhour))
    }

    /// AWS Open Data URL for RAP.
    pub fn aws_url(date: &str, hour: u32, fhour: u32) -> String {
        format!(
            "https://noaa-rap-pds.s3.amazonaws.com/rap.{}/rap.t{:02}z.awp130pgrbf{:02}.grib2",
            date, hour, fhour
        )
    }

    // --- Grid specifications (13km CONUS, grid 130) ---

    pub fn grid_nx() -> u32 { 451 }
    pub fn grid_ny() -> u32 { 337 }
    pub fn grid_dx() -> f64 { 13545.09 } // meters (Lambert conformal)
    pub fn grid_dy() -> f64 { 13545.09 }

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
    pub fn composite_refl() -> &'static str { "REFC:entire atmosphere" }
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
    fn test_nomads_url_format() {
        let url = RapConfig::nomads_url("20260310", 15, 21);
        assert_eq!(
            url,
            "https://nomads.ncep.noaa.gov/pub/data/nccf/com/rap/prod/rap.20260310/rap.t15z.awp130pgrbf21.grib2"
        );
    }

    #[test]
    fn test_aws_url_format() {
        let url = RapConfig::aws_url("20260310", 3, 6);
        assert!(url.starts_with("https://noaa-rap-pds.s3.amazonaws.com/"));
        assert!(url.contains("rap.20260310"));
        assert!(url.contains("t03z"));
    }

    #[test]
    fn test_idx_url() {
        let url = RapConfig::idx_url("20260310", 0, 0);
        assert!(url.ends_with(".grib2.idx"));
    }

    #[test]
    fn test_grid_specs() {
        assert_eq!(RapConfig::grid_nx(), 451);
        assert_eq!(RapConfig::grid_ny(), 337);
    }
}
