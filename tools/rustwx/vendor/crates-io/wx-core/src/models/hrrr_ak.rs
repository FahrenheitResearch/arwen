/// Configuration and URL generation for HRRR-Alaska.
///
/// HRRR-Alaska is the 3km Alaska domain, separate from the CONUS HRRR.
/// Runs hourly with forecasts out to 48h (for 00/06/12/18z) or 18h otherwise.
/// Uses a polar stereographic projection.
pub struct HrrrAkConfig;

impl HrrrAkConfig {
    /// AWS Open Data URL for HRRR-Alaska GRIB2 files.
    ///
    /// - `date`: format `"YYYYMMDD"` (e.g. `"20260310"`)
    /// - `hour`: model initialization hour (0-23)
    /// - `product`: `"sfc"`, `"prs"`, `"nat"`, or `"subh"`
    /// - `fhour`: forecast hour (0-48)
    pub fn aws_url(date: &str, hour: u32, product: &str, fhour: u32) -> String {
        let product_code = Self::product_code(product);
        format!(
            "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{}/alaska/hrrr.t{:02}z.{}f{:02}.ak.grib2",
            date, hour, product_code, fhour
        )
    }

    /// IDX file URL.
    pub fn idx_url(date: &str, hour: u32, product: &str, fhour: u32) -> String {
        format!("{}.idx", Self::aws_url(date, hour, product, fhour))
    }

    /// NOMADS URL for HRRR-Alaska.
    pub fn nomads_url(date: &str, hour: u32, product: &str, fhour: u32) -> String {
        let product_code = Self::product_code(product);
        format!(
            "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/hrrr.{}/alaska/hrrr.t{:02}z.{}f{:02}.ak.grib2",
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

    // --- Grid specifications (3km Alaska, polar stereographic) ---

    pub fn grid_nx() -> u32 { 1299 }
    pub fn grid_ny() -> u32 { 919 }
    pub fn grid_dx() -> f64 { 3000.0 } // meters
    pub fn grid_dy() -> f64 { 3000.0 }

    // --- Common variable patterns for .idx matching (same as CONUS HRRR) ---

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
        let url = HrrrAkConfig::aws_url("20260310", 12, "sfc", 6);
        assert_eq!(
            url,
            "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20260310/alaska/hrrr.t12z.wrfsfcf06.ak.grib2"
        );
    }

    #[test]
    fn test_aws_url_contains_alaska() {
        let url = HrrrAkConfig::aws_url("20260310", 0, "prs", 0);
        assert!(url.contains("/alaska/"));
        assert!(url.contains(".ak.grib2"));
    }

    #[test]
    fn test_nomads_url() {
        let url = HrrrAkConfig::nomads_url("20260310", 6, "sfc", 12);
        assert!(url.starts_with("https://nomads.ncep.noaa.gov/"));
        assert!(url.contains("/alaska/"));
    }

    #[test]
    fn test_idx_url() {
        let url = HrrrAkConfig::idx_url("20260310", 0, "sfc", 0);
        assert!(url.ends_with(".ak.grib2.idx"));
    }

    #[test]
    fn test_grid_specs() {
        assert_eq!(HrrrAkConfig::grid_nx(), 1299);
        assert_eq!(HrrrAkConfig::grid_ny(), 919);
        assert_eq!(HrrrAkConfig::grid_dx(), 3000.0);
    }

    #[test]
    fn test_product_code_aliases() {
        let url1 = HrrrAkConfig::aws_url("20260310", 0, "sfc", 0);
        let url2 = HrrrAkConfig::aws_url("20260310", 0, "surface", 0);
        assert_eq!(url1, url2);
    }
}
