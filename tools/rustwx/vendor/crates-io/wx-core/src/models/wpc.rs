/// Configuration and URL generation for WPC QPF (Weather Prediction Center
/// Quantitative Precipitation Forecasts).
///
/// WPC produces manually-analyzed QPF grids at 2.5km resolution over CONUS.
/// Products include 6-hour QPF accumulations and excessive rainfall outlook.
pub struct WpcConfig;

impl WpcConfig {
    /// FTP/HTTPS URL for WPC 2.5km QPF GRIB2 files.
    ///
    /// - `date`: format `"YYYYMMDD"` (e.g. `"20260310"`)
    /// - `hour`: issuance hour (0, 6, 12, 18)
    /// - `product`: `"6hr"` (6-hour QPF), `"day1"`, `"day2"`, `"day3"`
    /// - `fhour`: forecast hour for 6-hr period ending (e.g., 6, 12, 18, 24, ...)
    pub fn url(date: &str, hour: u32, product: &str, fhour: u32) -> String {
        match product {
            "6hr" | "6h" | "qpf" => format!(
                "https://ftp.wpc.ncep.noaa.gov/2p5km_qpf/p06m_{}{:02}f{:03}.grb",
                date, hour, fhour
            ),
            "day1" => format!(
                "https://ftp.wpc.ncep.noaa.gov/2p5km_qpf/d1_tl_{}12.grb",
                date
            ),
            "day2" => format!(
                "https://ftp.wpc.ncep.noaa.gov/2p5km_qpf/d2_tl_{}12.grb",
                date
            ),
            "day3" => format!(
                "https://ftp.wpc.ncep.noaa.gov/2p5km_qpf/d3_tl_{}12.grb",
                date
            ),
            _ => format!(
                "https://ftp.wpc.ncep.noaa.gov/2p5km_qpf/p06m_{}{:02}f{:03}.grb",
                date, hour, fhour
            ),
        }
    }

    /// NOMADS URL for WPC QPF products.
    pub fn nomads_url(date: &str, hour: u32, fhour: u32) -> String {
        format!(
            "https://nomads.ncep.noaa.gov/pub/data/nccf/com/wpc/prod/qpf/p06m_{}{:02}f{:03}.grb",
            date, hour, fhour
        )
    }

    /// WPC QPF files are typically small enough that .idx files are not used.
    /// Returns `None`.
    pub fn idx_url(_date: &str, _hour: u32, _product: &str, _fhour: u32) -> Option<String> {
        None
    }

    // --- Grid specifications (2.5km CONUS, NDFD grid) ---

    pub fn grid_nx() -> u32 { 2345 }
    pub fn grid_ny() -> u32 { 1597 }
    pub fn grid_dx() -> f64 { 2539.703 } // meters (Lambert conformal)
    pub fn grid_dy() -> f64 { 2539.703 }

    // --- Common variable patterns ---

    pub fn precip_6hr() -> &'static str { "APCP:surface" }
    pub fn precip_total() -> &'static str { "APCP:surface" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_url_6hr_qpf() {
        let url = WpcConfig::url("20260310", 12, "6hr", 18);
        assert!(url.starts_with("https://ftp.wpc.ncep.noaa.gov/"));
        assert!(url.contains("p06m_"));
        assert!(url.contains("2026031012"));
    }

    #[test]
    fn test_url_day1() {
        let url = WpcConfig::url("20260310", 0, "day1", 0);
        assert!(url.contains("d1_tl_2026031012"));
    }

    #[test]
    fn test_url_day2() {
        let url = WpcConfig::url("20260310", 0, "day2", 0);
        assert!(url.contains("d2_tl_"));
    }

    #[test]
    fn test_url_day3() {
        let url = WpcConfig::url("20260310", 0, "day3", 0);
        assert!(url.contains("d3_tl_"));
    }

    #[test]
    fn test_nomads_url() {
        let url = WpcConfig::nomads_url("20260310", 0, 6);
        assert!(url.starts_with("https://nomads.ncep.noaa.gov/"));
        assert!(url.contains("p06m_"));
    }

    #[test]
    fn test_idx_url_returns_none() {
        assert!(WpcConfig::idx_url("20260310", 0, "6hr", 6).is_none());
    }

    #[test]
    fn test_grid_specs() {
        assert_eq!(WpcConfig::grid_nx(), 2345);
        assert_eq!(WpcConfig::grid_ny(), 1597);
    }
}
