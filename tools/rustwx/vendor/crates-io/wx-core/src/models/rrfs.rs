/// Configuration and URL generation for the RRFS (Rapid Refresh Forecast System).
///
/// RRFS is the planned replacement for HRRR and RAP, running at 3km over CONUS.
/// Currently in pre-operational testing on NOMADS.
/// Initialization times vary; forecasts out to 60 hours.
pub struct RrfsConfig;

impl RrfsConfig {
    /// NOMADS URL for RRFS GRIB2 files.
    ///
    /// - `date`: format `"YYYYMMDD"` (e.g. `"20260310"`)
    /// - `hour`: model initialization hour (0-23)
    /// - `product`: `"prslev"`, `"natlev"`, or `"ififip"`
    /// - `fhour`: forecast hour (0-60)
    pub fn nomads_url(date: &str, hour: u32, product: &str, fhour: u32) -> String {
        let product_code = Self::product_code(product);
        format!(
            "https://nomads.ncep.noaa.gov/pub/data/nccf/com/rrfs/prod/rrfs.{}/{:02}/rrfs.t{:02}z.{}.f{:03}.grib2",
            date, hour, hour, product_code, fhour
        )
    }

    /// IDX file URL.
    pub fn idx_url(date: &str, hour: u32, product: &str, fhour: u32) -> String {
        format!("{}.idx", Self::nomads_url(date, hour, product, fhour))
    }

    fn product_code(product: &str) -> &str {
        match product {
            "nat" | "natlev" | "native" => "natlev",
            "ifi" | "ififip" => "ififip",
            _ => "prslev",
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
    pub fn updraft_helicity() -> &'static str { "MXUPHL" }
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
    fn test_nomads_url_prslev() {
        let url = RrfsConfig::nomads_url("20260310", 12, "prslev", 24);
        assert!(url.starts_with("https://nomads.ncep.noaa.gov/"));
        assert!(url.contains("rrfs.20260310"));
        assert!(url.contains("prslev"));
        assert!(url.contains(".f024."));
    }

    #[test]
    fn test_nomads_url_natlev() {
        let url = RrfsConfig::nomads_url("20260310", 0, "nat", 6);
        assert!(url.contains("natlev"));
    }

    #[test]
    fn test_nomads_url_default_product() {
        let url = RrfsConfig::nomads_url("20260310", 0, "unknown", 0);
        assert!(url.contains("prslev"));
    }

    #[test]
    fn test_idx_url() {
        let url = RrfsConfig::idx_url("20260310", 0, "prslev", 0);
        assert!(url.ends_with(".grib2.idx"));
    }

    #[test]
    fn test_grid_specs() {
        assert_eq!(RrfsConfig::grid_nx(), 1799);
        assert_eq!(RrfsConfig::grid_ny(), 1059);
        assert_eq!(RrfsConfig::grid_dx(), 3000.0);
    }
}
