/// Configuration and URL generation for the CFS (Climate Forecast System).
///
/// CFS provides seasonal forecasts out to 9 months at ~100km (T126) global resolution.
/// CFS v2 runs 4 times daily (00/06/12/18z) with 4 ensemble members per cycle.
/// Also provides CFS Reanalysis (CFSR) for historical data (1979-2010) and
/// CFS v2 operational analysis (2011-present).
pub struct CfsConfig;

impl CfsConfig {
    /// NOMADS URL for CFS operational forecast GRIB2 files.
    ///
    /// - `date`: format `"YYYYMMDD"` (e.g. `"20260310"`)
    /// - `hour`: model initialization hour (0, 6, 12, 18)
    /// - `product`: `"flxf"` (flux), `"pgbf"` (pressure-level), `"ocnf"` (ocean),
    ///   `"ipvf"` (isentropic PV)
    /// - `fhour`: forecast hour (6-hour intervals out to ~9 months)
    pub fn nomads_url(date: &str, hour: u32, product: &str, fhour: u32) -> String {
        let product_code = Self::product_code(product);
        format!(
            "https://nomads.ncep.noaa.gov/pub/data/nccf/com/cfs/prod/cfs.{}/{:02}/6hrly_grib_01/{}.01.{}.{:04}.grb2",
            date, hour, product_code, date, fhour
        )
    }

    /// AWS Open Data URL for CFS (NOAA CFS PDS bucket).
    pub fn aws_url(date: &str, hour: u32, product: &str, fhour: u32) -> String {
        let product_code = Self::product_code(product);
        format!(
            "https://noaa-cfs-pds.s3.amazonaws.com/cfs.{}/{:02}/6hrly_grib_01/{}.01.{}.{:04}.grb2",
            date, hour, product_code, date, fhour
        )
    }

    /// IDX file URL.
    pub fn idx_url(date: &str, hour: u32, product: &str, fhour: u32) -> String {
        format!("{}.idx", Self::nomads_url(date, hour, product, fhour))
    }

    /// CFS Reanalysis (CFSR) URL for historical data.
    ///
    /// - `year`: 4-digit year
    /// - `month`: 2-digit month
    pub fn cfsr_url(year: &str, month: &str) -> String {
        format!(
            "https://nomads.ncep.noaa.gov/pub/data/nccf/com/cfs/prod/cfsr.{}{}/",
            year, month
        )
    }

    fn product_code(product: &str) -> &str {
        match product {
            "flux" | "flxf" => "flxf",
            "pgb" | "pgbf" | "pressure" => "pgbf",
            "ocean" | "ocnf" => "ocnf",
            "ipv" | "ipvf" => "ipvf",
            _ => "pgbf",
        }
    }

    // --- Grid specifications (~100km / T126 global Gaussian grid) ---

    pub fn grid_nx() -> u32 { 384 }
    pub fn grid_ny() -> u32 { 190 }
    pub fn grid_dx() -> f64 { 0.9375 } // degrees (~100km)
    pub fn grid_dy() -> f64 { 0.9375 }

    // --- Common variable patterns for .idx matching ---

    pub fn sfc_temp_2m() -> &'static str { "TMP:2 m above ground" }
    pub fn sfc_u_wind_10m() -> &'static str { "UGRD:10 m above ground" }
    pub fn sfc_v_wind_10m() -> &'static str { "VGRD:10 m above ground" }
    pub fn sfc_mslp() -> &'static str { "PRMSL:mean sea level" }
    pub fn sfc_precip() -> &'static str { "PRATE:surface" }
    pub fn sfc_pressure() -> &'static str { "PRES:surface" }
    pub fn sfc_hgt() -> &'static str { "HGT:surface" }
    pub fn sst() -> &'static str { "TMP:surface" }

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
        let url = CfsConfig::nomads_url("20260310", 0, "pgbf", 24);
        assert!(url.starts_with("https://nomads.ncep.noaa.gov/"));
        assert!(url.contains("cfs.20260310"));
        assert!(url.contains("pgbf"));
    }

    #[test]
    fn test_aws_url_format() {
        let url = CfsConfig::aws_url("20260310", 6, "flxf", 48);
        assert!(url.starts_with("https://noaa-cfs-pds.s3.amazonaws.com/"));
        assert!(url.contains("flxf"));
    }

    #[test]
    fn test_idx_url() {
        let url = CfsConfig::idx_url("20260310", 0, "pgbf", 6);
        assert!(url.ends_with(".idx"));
    }

    #[test]
    fn test_cfsr_url() {
        let url = CfsConfig::cfsr_url("2020", "06");
        assert!(url.contains("cfsr.202006"));
    }

    #[test]
    fn test_product_code_default() {
        let url = CfsConfig::nomads_url("20260310", 0, "unknown", 6);
        assert!(url.contains("pgbf"));
    }

    #[test]
    fn test_grid_specs() {
        assert_eq!(CfsConfig::grid_nx(), 384);
        assert_eq!(CfsConfig::grid_ny(), 190);
    }
}
