/// Configuration and URL generation for the HREF (High-Resolution Ensemble Forecast).
///
/// HREF combines multiple high-resolution models (HRRR, NAM Nest, HiResW ARW/FV3)
/// into ensemble mean, probability, and percentile products.
/// Runs every 6 hours (00/06/12/18z), 3km CONUS grid, forecasts out to 48 hours.
pub struct HrefConfig;

impl HrefConfig {
    /// NOMADS URL for HREF GRIB2 files.
    ///
    /// - `date`: format `"YYYYMMDD"` (e.g. `"20260310"`)
    /// - `hour`: model initialization hour (0, 6, 12, 18)
    /// - `product`: `"mean"`, `"pmmn"` (probability-matched mean), `"avrg"`, `"lpmm"`, `"prob"`
    /// - `fhour`: forecast hour (1-48)
    pub fn nomads_url(date: &str, hour: u32, product: &str, fhour: u32) -> String {
        let product_code = Self::product_code(product);
        format!(
            "https://nomads.ncep.noaa.gov/pub/data/nccf/com/href/prod/href.{}/ensprod/href.t{:02}z.conus.{}.f{:02}.grib2",
            date, hour, product_code, fhour
        )
    }

    /// IDX file URL.
    pub fn idx_url(date: &str, hour: u32, product: &str, fhour: u32) -> String {
        format!("{}.idx", Self::nomads_url(date, hour, product, fhour))
    }

    /// AWS Open Data URL for HREF.
    pub fn aws_url(date: &str, hour: u32, product: &str, fhour: u32) -> String {
        let product_code = Self::product_code(product);
        format!(
            "https://noaa-href-pds.s3.amazonaws.com/href.{}/ensprod/href.t{:02}z.conus.{}.f{:02}.grib2",
            date, hour, product_code, fhour
        )
    }

    fn product_code(product: &str) -> &str {
        match product {
            "pmmn" | "prob_matched_mean" => "pmmn",
            "avrg" | "average" => "avrg",
            "lpmm" | "localized_prob" => "lpmm",
            "prob" | "probability" => "prob",
            _ => "mean",
        }
    }

    // --- Grid specifications (3km CONUS, Lambert conformal) ---

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
    fn test_nomads_url_mean() {
        let url = HrefConfig::nomads_url("20260310", 12, "mean", 24);
        assert!(url.starts_with("https://nomads.ncep.noaa.gov/"));
        assert!(url.contains("href.20260310"));
        assert!(url.contains(".mean."));
    }

    #[test]
    fn test_nomads_url_pmmn() {
        let url = HrefConfig::nomads_url("20260310", 0, "pmmn", 6);
        assert!(url.contains(".pmmn."));
    }

    #[test]
    fn test_aws_url() {
        let url = HrefConfig::aws_url("20260310", 6, "mean", 12);
        assert!(url.starts_with("https://noaa-href-pds.s3.amazonaws.com/"));
        assert!(url.contains("href.20260310"));
    }

    #[test]
    fn test_idx_url() {
        let url = HrefConfig::idx_url("20260310", 0, "mean", 1);
        assert!(url.ends_with(".grib2.idx"));
    }

    #[test]
    fn test_product_code_default_to_mean() {
        let url = HrefConfig::nomads_url("20260310", 0, "unknown", 6);
        assert!(url.contains(".mean."));
    }

    #[test]
    fn test_grid_specs() {
        assert_eq!(HrefConfig::grid_nx(), 1799);
        assert_eq!(HrefConfig::grid_ny(), 1059);
    }
}
