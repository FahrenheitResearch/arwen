/// Configuration and URL generation for the URMA (UnRestricted Mesoscale Analysis).
///
/// URMA is the companion to RTMA, providing a 2.5km CONUS analysis with a longer
/// production delay that allows more observations to be ingested. URMA is considered
/// more accurate than RTMA due to the additional data assimilation window.
/// Updated hourly, typically available ~6 hours after valid time.
pub struct UrmaConfig;

impl UrmaConfig {
    /// AWS Open Data URL for URMA GRIB2 files.
    ///
    /// - `date`: format `"YYYYMMDD"` (e.g. `"20260310"`)
    /// - `hour`: analysis hour (0-23)
    /// - `product`: `"2dvaranl"` (2D analysis) or `"2dvarges"` (2D guess)
    pub fn aws_url(date: &str, hour: u32, product: &str) -> String {
        let product_code = Self::product_code(product);
        format!(
            "https://noaa-urma-pds.s3.amazonaws.com/urma2p5.{}/urma2p5.t{:02}z.{}_ndfd.grb2_wexp",
            date, hour, product_code
        )
    }

    /// IDX file URL.
    pub fn idx_url(date: &str, hour: u32, product: &str) -> String {
        format!("{}.idx", Self::aws_url(date, hour, product))
    }

    /// NOMADS URL for URMA.
    pub fn nomads_url(date: &str, hour: u32, product: &str) -> String {
        let product_code = Self::product_code(product);
        format!(
            "https://nomads.ncep.noaa.gov/pub/data/nccf/com/urma/prod/urma2p5.{}/urma2p5.t{:02}z.{}_ndfd.grb2_wexp",
            date, hour, product_code
        )
    }

    fn product_code(product: &str) -> &str {
        match product {
            "guess" | "ges" | "2dvarges" => "2dvarges",
            _ => "2dvaranl",
        }
    }

    // --- Grid specifications (2.5km CONUS, NDFD grid — same as RTMA) ---

    pub fn grid_nx() -> u32 { 2345 }
    pub fn grid_ny() -> u32 { 1597 }
    pub fn grid_dx() -> f64 { 2539.703 } // meters (Lambert conformal)
    pub fn grid_dy() -> f64 { 2539.703 }

    // --- Common variable patterns for .idx matching ---

    pub fn sfc_temp_2m() -> &'static str { "TMP:2 m above ground" }
    pub fn sfc_dewpoint_2m() -> &'static str { "DPT:2 m above ground" }
    pub fn sfc_u_wind_10m() -> &'static str { "UGRD:10 m above ground" }
    pub fn sfc_v_wind_10m() -> &'static str { "VGRD:10 m above ground" }
    pub fn sfc_gust() -> &'static str { "GUST:surface" }
    pub fn sfc_pressure() -> &'static str { "PRES:surface" }
    pub fn sfc_visibility() -> &'static str { "VIS:surface" }
    pub fn sfc_ceiling() -> &'static str { "HGT:cloud ceiling" }
    pub fn wind_speed() -> &'static str { "WIND:10 m above ground" }
    pub fn wind_direction() -> &'static str { "WDIR:10 m above ground" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aws_url_analysis() {
        let url = UrmaConfig::aws_url("20260310", 12, "2dvaranl");
        assert!(url.starts_with("https://noaa-urma-pds.s3.amazonaws.com/"));
        assert!(url.contains("urma2p5.20260310"));
        assert!(url.contains("2dvaranl"));
    }

    #[test]
    fn test_aws_url_guess() {
        let url = UrmaConfig::aws_url("20260310", 0, "guess");
        assert!(url.contains("2dvarges"));
    }

    #[test]
    fn test_nomads_url() {
        let url = UrmaConfig::nomads_url("20260310", 6, "2dvaranl");
        assert!(url.starts_with("https://nomads.ncep.noaa.gov/"));
        assert!(url.contains("urma2p5.20260310"));
    }

    #[test]
    fn test_idx_url() {
        let url = UrmaConfig::idx_url("20260310", 0, "2dvaranl");
        assert!(url.ends_with(".idx"));
    }

    #[test]
    fn test_grid_specs() {
        assert_eq!(UrmaConfig::grid_nx(), 2345);
        assert_eq!(UrmaConfig::grid_ny(), 1597);
    }

    #[test]
    fn test_product_code_default() {
        let url = UrmaConfig::aws_url("20260310", 0, "unknown");
        assert!(url.contains("2dvaranl"));
    }
}
