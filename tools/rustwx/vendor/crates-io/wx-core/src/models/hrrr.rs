/// Configuration and URL generation for the HRRR (High-Resolution Rapid Refresh) model.
///
/// HRRR is a 3km CONUS model run hourly with forecasts out to 18h (48h for 00/06/12/18z).
pub struct HrrrConfig;

impl HrrrConfig {
    /// Base URL for HRRR GRIB2 files on the AWS Open Data bucket.
    ///
    /// - `date`: format `"YYYYMMDD"` (e.g. `"20260310"`)
    /// - `hour`: model initialization hour (0-23)
    /// - `product`: `"sfc"`, `"prs"`, `"nat"`, or `"subh"`
    /// - `fhour`: forecast hour (0-48)
    pub fn aws_url(date: &str, hour: u32, product: &str, fhour: u32) -> String {
        let product_code = Self::product_code(product);
        format!(
            "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{}/conus/hrrr.t{:02}z.{}f{:02}.grib2",
            date, hour, product_code, fhour
        )
    }

    /// IDX file URL (GRIB2 URL + `.idx`).
    pub fn idx_url(date: &str, hour: u32, product: &str, fhour: u32) -> String {
        format!("{}.idx", Self::aws_url(date, hour, product, fhour))
    }

    /// NOMADS URL (NCEP operational server, rolling ~2 day availability).
    pub fn nomads_url(date: &str, hour: u32, product: &str, fhour: u32) -> String {
        let product_code = Self::product_code(product);
        format!(
            "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/hrrr.{}/conus/hrrr.t{:02}z.{}f{:02}.grib2",
            date, hour, product_code, fhour
        )
    }

    fn product_code(product: &str) -> &str {
        match product {
            "sfc" | "surface" => "wrfsfc",
            "prs" | "pressure" => "wrfprs",
            "nat" | "native" => "wrfnat",
            "subh" | "subhourly" => "wrfsubh",
            _ => "wrfsfc",
        }
    }

    // --- Grid specifications ---

    pub fn grid_nx() -> u32 { 1799 }
    pub fn grid_ny() -> u32 { 1059 }
    pub fn grid_dx() -> f64 { 3000.0 } // meters
    pub fn grid_dy() -> f64 { 3000.0 }

    // --- Common variable patterns for .idx matching ---

    pub fn sfc_temp_2m() -> &'static str { "TMP:2 m above ground" }
    pub fn sfc_dewpoint_2m() -> &'static str { "DPT:2 m above ground" }
    pub fn sfc_rh_2m() -> &'static str { "RH:2 m above ground" }
    pub fn sfc_u_wind_10m() -> &'static str { "UGRD:10 m above ground" }
    pub fn sfc_v_wind_10m() -> &'static str { "VGRD:10 m above ground" }
    pub fn sfc_gust() -> &'static str { "GUST:surface" }
    pub fn sfc_mslp() -> &'static str { "MSLMA:mean sea level" }
    pub fn sfc_pressure() -> &'static str { "PRES:surface" }
    pub fn sfc_cape() -> &'static str { "CAPE:surface" }
    pub fn sfc_cin() -> &'static str { "CIN:surface" }
    pub fn composite_refl() -> &'static str { "REFC:entire atmosphere" }
    pub fn sfc_precip() -> &'static str { "APCP:surface" }
    pub fn sfc_visibility() -> &'static str { "VIS:surface" }
    pub fn updraft_helicity() -> &'static str { "MXUPHL" }
    pub fn sfc_hgt() -> &'static str { "HGT:surface" }

    /// Build a pattern for a variable on a pressure level (e.g., `"TMP:500 mb"`).
    pub fn prs_var(var: &str, level_mb: u32) -> String {
        format!("{}:{} mb", var, level_mb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aws_url_format() {
        let url = HrrrConfig::aws_url("20260310", 12, "sfc", 6);
        assert_eq!(
            url,
            "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20260310/conus/hrrr.t12z.wrfsfcf06.grib2"
        );
    }

    #[test]
    fn test_aws_url_pressure_product() {
        let url = HrrrConfig::aws_url("20260310", 0, "prs", 18);
        assert!(url.contains("wrfprs"));
        assert!(url.contains("f18"));
    }

    #[test]
    fn test_aws_url_native_product() {
        let url = HrrrConfig::aws_url("20260310", 6, "nat", 3);
        assert!(url.contains("wrfnat"));
    }

    #[test]
    fn test_aws_url_subhourly_product() {
        let url = HrrrConfig::aws_url("20260310", 18, "subh", 1);
        assert!(url.contains("wrfsubh"));
    }

    #[test]
    fn test_idx_url_appends_idx() {
        let url = HrrrConfig::idx_url("20260310", 12, "sfc", 6);
        assert!(url.ends_with(".grib2.idx"));
    }

    #[test]
    fn test_nomads_url_format() {
        let url = HrrrConfig::nomads_url("20260310", 0, "sfc", 0);
        assert!(url.starts_with("https://nomads.ncep.noaa.gov/"));
        assert!(url.contains("hrrr.20260310"));
        assert!(url.contains("conus"));
    }

    #[test]
    fn test_product_code_aliases() {
        let url1 = HrrrConfig::aws_url("20260310", 0, "sfc", 0);
        let url2 = HrrrConfig::aws_url("20260310", 0, "surface", 0);
        assert_eq!(url1, url2);
    }

    #[test]
    fn test_grid_specs() {
        assert_eq!(HrrrConfig::grid_nx(), 1799);
        assert_eq!(HrrrConfig::grid_ny(), 1059);
        assert_eq!(HrrrConfig::grid_dx(), 3000.0);
        assert_eq!(HrrrConfig::grid_dy(), 3000.0);
    }

    #[test]
    fn test_prs_var_format() {
        assert_eq!(HrrrConfig::prs_var("TMP", 500), "TMP:500 mb");
        assert_eq!(HrrrConfig::prs_var("HGT", 250), "HGT:250 mb");
    }

    #[test]
    fn test_variable_patterns() {
        assert_eq!(HrrrConfig::sfc_temp_2m(), "TMP:2 m above ground");
        assert_eq!(HrrrConfig::composite_refl(), "REFC:entire atmosphere");
        assert_eq!(HrrrConfig::sfc_mslp(), "MSLMA:mean sea level");
    }

    #[test]
    fn test_hour_zero_padding() {
        let url = HrrrConfig::aws_url("20260310", 3, "sfc", 1);
        assert!(url.contains("t03z"));
        assert!(url.contains("f01"));
    }
}
