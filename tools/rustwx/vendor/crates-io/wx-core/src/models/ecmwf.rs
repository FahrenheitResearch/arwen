/// Configuration and URL generation for ECMWF Open Data (IFS).
///
/// ECMWF provides open data from its Integrated Forecasting System (IFS).
/// HRES runs at 0.1° resolution, ENS at 0.2°. Open data is available at 0.25° globally.
/// Initialization times: 00z and 12z, forecasts out to 240h (HRES) / 360h (ENS).
pub struct EcmwfConfig;

impl EcmwfConfig {
    /// URL for ECMWF open data GRIB2 files.
    ///
    /// - `date`: format `"YYYYMMDD"` (e.g. `"20260310"`)
    /// - `hour`: model initialization hour (0 or 12)
    /// - `product`: `"oper"` (HRES) or `"enfo"` (ENS)
    /// - `fhour`: forecast hour
    pub fn open_data_url(date: &str, hour: u32, product: &str, fhour: u32) -> String {
        let stream = Self::product_stream(product);
        format!(
            "https://data.ecmwf.int/forecasts/{}/{:02}z/ifs/0p25/{}/{}{:02}0000-{}h-{}-fc.grib2",
            date, hour, stream, date, hour, fhour, stream
        )
    }

    /// IDX file URL (GRIB2 URL + `.idx`).
    pub fn idx_url(date: &str, hour: u32, product: &str, fhour: u32) -> String {
        format!("{}.idx", Self::open_data_url(date, hour, product, fhour))
    }

    fn product_stream(product: &str) -> &str {
        match product {
            "ens" | "enfo" | "ensemble" => "enfo",
            _ => "oper",
        }
    }

    // --- Grid specifications (0.25 degree global) ---

    pub fn grid_nx() -> u32 { 1440 }
    pub fn grid_ny() -> u32 { 721 }
    pub fn grid_dx() -> f64 { 0.25 } // degrees
    pub fn grid_dy() -> f64 { 0.25 }

    // --- Common variable patterns for .idx matching ---

    pub fn sfc_temp_2m() -> &'static str { "2t:sfc" }
    pub fn sfc_dewpoint_2m() -> &'static str { "2d:sfc" }
    pub fn sfc_u_wind_10m() -> &'static str { "10u:sfc" }
    pub fn sfc_v_wind_10m() -> &'static str { "10v:sfc" }
    pub fn sfc_gust() -> &'static str { "10fg:sfc" }
    pub fn sfc_mslp() -> &'static str { "msl:sfc" }
    pub fn sfc_pressure() -> &'static str { "sp:sfc" }
    pub fn sfc_cape() -> &'static str { "cape:sfc" }
    pub fn sfc_precip() -> &'static str { "tp:sfc" }
    pub fn sfc_hgt() -> &'static str { "orog:sfc" }

    /// Build a pattern for a variable on a pressure level (e.g., `"t:500"`).
    pub fn prs_var(var: &str, level_mb: u32) -> String {
        format!("{}:{}", var, level_mb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_open_data_url_oper() {
        let url = EcmwfConfig::open_data_url("20260310", 0, "oper", 24);
        assert!(url.starts_with("https://data.ecmwf.int/forecasts/"));
        assert!(url.contains("20260310"));
        assert!(url.contains("oper"));
    }

    #[test]
    fn test_open_data_url_ensemble() {
        let url = EcmwfConfig::open_data_url("20260310", 12, "ens", 48);
        assert!(url.contains("enfo"));
    }

    #[test]
    fn test_idx_url() {
        let url = EcmwfConfig::idx_url("20260310", 0, "oper", 6);
        assert!(url.ends_with(".idx"));
    }

    #[test]
    fn test_product_stream_default() {
        // Unknown product should default to "oper"
        let url = EcmwfConfig::open_data_url("20260310", 0, "hres", 6);
        assert!(url.contains("oper"));
    }

    #[test]
    fn test_grid_specs() {
        assert_eq!(EcmwfConfig::grid_nx(), 1440);
        assert_eq!(EcmwfConfig::grid_ny(), 721);
        assert_eq!(EcmwfConfig::grid_dx(), 0.25);
    }

    #[test]
    fn test_prs_var() {
        assert_eq!(EcmwfConfig::prs_var("t", 500), "t:500");
    }

    #[test]
    fn test_variable_patterns() {
        assert_eq!(EcmwfConfig::sfc_temp_2m(), "2t:sfc");
        assert_eq!(EcmwfConfig::sfc_mslp(), "msl:sfc");
    }
}
