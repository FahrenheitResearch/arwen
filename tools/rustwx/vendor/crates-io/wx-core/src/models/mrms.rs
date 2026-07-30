/// Configuration and URL generation for MRMS (Multi-Radar Multi-Sensor).
///
/// MRMS provides 1km CONUS radar mosaic products including composite reflectivity,
/// precipitation rate, precipitation type, and many other derived fields.
/// Updated every 2 minutes for radar products, every hour for QPE.
///
/// **Note:** MRMS files on AWS are gzip-compressed (`.grib2.gz`).
/// They must be decompressed before GRIB2 parsing.
pub struct MrmsConfig;

impl MrmsConfig {
    /// AWS Open Data URL for MRMS GRIB2 files (gzipped).
    ///
    /// - `product`: MRMS product name (e.g., `"MergedReflectivityQCComposite"`,
    ///   `"PrecipRate"`, `"PrecipFlag"`, `"RadarOnly_QPE_01H"`)
    /// - `level`: vertical level string (e.g., `"00.50"` for 0.5km, `"00.00"` for surface)
    /// - `datetime`: format `"YYYYMMDD-HHmmss"` (e.g., `"20260310-180000"`)
    ///
    /// Returns a `.grib2.gz` URL — must decompress before parsing.
    pub fn aws_url(product: &str, level: &str, datetime: &str) -> String {
        format!(
            "https://mrms-cp-pds.s3.amazonaws.com/MRMS_{}_{}_{}00.grib2.gz",
            product, level, datetime
        )
    }

    /// Convenience URL for the most common composite reflectivity product.
    ///
    /// - `datetime`: format `"YYYYMMDD-HHmmss"`
    pub fn composite_reflectivity_url(datetime: &str) -> String {
        Self::aws_url("MergedReflectivityQCComposite", "00.50", datetime)
    }

    /// Convenience URL for precipitation rate.
    pub fn precip_rate_url(datetime: &str) -> String {
        Self::aws_url("PrecipRate", "00.00", datetime)
    }

    /// Convenience URL for 1-hour QPE (Quantitative Precipitation Estimate).
    pub fn qpe_01h_url(datetime: &str) -> String {
        Self::aws_url("RadarOnly_QPE_01H", "00.00", datetime)
    }

    /// Convenience URL for precipitation flag/type.
    pub fn precip_flag_url(datetime: &str) -> String {
        Self::aws_url("PrecipFlag", "00.00", datetime)
    }

    /// MRMS does not use .idx files — each file contains a single product.
    /// Returns `None`.
    pub fn idx_url(_product: &str, _level: &str, _datetime: &str) -> Option<String> {
        None
    }

    /// Check whether a URL requires gzip decompression.
    pub fn needs_decompress(url: &str) -> bool {
        url.ends_with(".gz")
    }

    // --- Grid specifications (1km CONUS, lat/lon) ---

    pub fn grid_nx() -> u32 { 7000 }
    pub fn grid_ny() -> u32 { 3500 }
    pub fn grid_dx() -> f64 { 0.01 } // degrees (~1km)
    pub fn grid_dy() -> f64 { 0.01 }

    /// MRMS domain bounds (CONUS).
    pub fn lon_min() -> f64 { -130.0 }
    pub fn lon_max() -> f64 { -60.0 }
    pub fn lat_min() -> f64 { 20.0 }
    pub fn lat_max() -> f64 { 55.0 }

    // --- Common product names ---

    pub fn product_composite_refl() -> &'static str { "MergedReflectivityQCComposite" }
    pub fn product_precip_rate() -> &'static str { "PrecipRate" }
    pub fn product_precip_flag() -> &'static str { "PrecipFlag" }
    pub fn product_qpe_01h() -> &'static str { "RadarOnly_QPE_01H" }
    pub fn product_qpe_03h() -> &'static str { "RadarOnly_QPE_03H" }
    pub fn product_qpe_06h() -> &'static str { "RadarOnly_QPE_06H" }
    pub fn product_qpe_12h() -> &'static str { "RadarOnly_QPE_12H" }
    pub fn product_qpe_24h() -> &'static str { "RadarOnly_QPE_24H" }
    pub fn product_qpe_48h() -> &'static str { "RadarOnly_QPE_48H" }
    pub fn product_qpe_72h() -> &'static str { "RadarOnly_QPE_72H" }
    pub fn product_rotation_track_60min() -> &'static str { "RotationTrack60min" }
    pub fn product_mesh() -> &'static str { "MESH" }
    pub fn product_vil() -> &'static str { "VIL" }
    pub fn product_echo_top_18() -> &'static str { "EchoTop_18" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aws_url_format() {
        let url = MrmsConfig::aws_url("MergedReflectivityQCComposite", "00.50", "20260310-180000");
        assert_eq!(
            url,
            "https://mrms-cp-pds.s3.amazonaws.com/MRMS_MergedReflectivityQCComposite_00.50_20260310-18000000.grib2.gz"
        );
    }

    #[test]
    fn test_composite_reflectivity_url() {
        let url = MrmsConfig::composite_reflectivity_url("20260310-120000");
        assert!(url.contains("MergedReflectivityQCComposite"));
        assert!(url.contains("00.50"));
        assert!(url.ends_with(".grib2.gz"));
    }

    #[test]
    fn test_precip_rate_url() {
        let url = MrmsConfig::precip_rate_url("20260310-120000");
        assert!(url.contains("PrecipRate"));
        assert!(url.contains("00.00"));
    }

    #[test]
    fn test_qpe_01h_url() {
        let url = MrmsConfig::qpe_01h_url("20260310-120000");
        assert!(url.contains("RadarOnly_QPE_01H"));
    }

    #[test]
    fn test_precip_flag_url() {
        let url = MrmsConfig::precip_flag_url("20260310-120000");
        assert!(url.contains("PrecipFlag"));
    }

    #[test]
    fn test_idx_url_returns_none() {
        assert!(MrmsConfig::idx_url("product", "level", "datetime").is_none());
    }

    #[test]
    fn test_needs_decompress() {
        assert!(MrmsConfig::needs_decompress("file.grib2.gz"));
        assert!(!MrmsConfig::needs_decompress("file.grib2"));
    }

    #[test]
    fn test_grid_specs() {
        assert_eq!(MrmsConfig::grid_nx(), 7000);
        assert_eq!(MrmsConfig::grid_ny(), 3500);
        assert_eq!(MrmsConfig::grid_dx(), 0.01);
    }

    #[test]
    fn test_domain_bounds() {
        assert_eq!(MrmsConfig::lon_min(), -130.0);
        assert_eq!(MrmsConfig::lon_max(), -60.0);
        assert_eq!(MrmsConfig::lat_min(), 20.0);
        assert_eq!(MrmsConfig::lat_max(), 55.0);
    }

    #[test]
    fn test_product_names() {
        assert_eq!(MrmsConfig::product_composite_refl(), "MergedReflectivityQCComposite");
        assert_eq!(MrmsConfig::product_precip_rate(), "PrecipRate");
        assert_eq!(MrmsConfig::product_qpe_01h(), "RadarOnly_QPE_01H");
    }
}
