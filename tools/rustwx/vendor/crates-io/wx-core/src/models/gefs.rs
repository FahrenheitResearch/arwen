/// Configuration and URL generation for the GEFS (Global Ensemble Forecast System).
///
/// GEFS runs 31 ensemble members (1 control + 30 perturbations) at 0.5° resolution,
/// with a 0.25° control run. Also produces ensemble mean and spread products.
/// Runs every 6 hours (00/06/12/18z) with forecasts out to 384 hours.
pub struct GefsConfig;

impl GefsConfig {
    /// AWS Open Data URL for GEFS GRIB2 files.
    ///
    /// - `date`: format `"YYYYMMDD"` (e.g. `"20260310"`)
    /// - `hour`: model initialization hour (0, 6, 12, 18)
    /// - `member`: ensemble member — `"c00"` (control), `"p01"`-`"p30"` (perturbations),
    ///   `"mean"`, or `"spread"`
    /// - `product`: `"pgrb2a"` (primary) or `"pgrb2b"` (secondary variables)
    /// - `fhour`: forecast hour (0-384)
    pub fn aws_url(date: &str, hour: u32, member: &str, product: &str, fhour: u32) -> String {
        let member_code = Self::member_code(member);
        let product_code = Self::product_code(product);

        match member {
            "mean" => format!(
                "https://noaa-gefs-pds.s3.amazonaws.com/gefs.{}/{:02}/atmos/{}sp5/geavg.t{:02}z.{}.0p50.f{:03}",
                date, hour, product_code, hour, product_code, fhour
            ),
            "spread" => format!(
                "https://noaa-gefs-pds.s3.amazonaws.com/gefs.{}/{:02}/atmos/{}sp5/gespr.t{:02}z.{}.0p50.f{:03}",
                date, hour, product_code, hour, product_code, fhour
            ),
            _ => format!(
                "https://noaa-gefs-pds.s3.amazonaws.com/gefs.{}/{:02}/atmos/{}sp5/{}.t{:02}z.{}.0p50.f{:03}",
                date, hour, product_code, member_code, hour, product_code, fhour
            ),
        }
    }

    /// IDX file URL.
    pub fn idx_url(date: &str, hour: u32, member: &str, product: &str, fhour: u32) -> String {
        format!("{}.idx", Self::aws_url(date, hour, member, product, fhour))
    }

    /// NOMADS URL for GEFS.
    pub fn nomads_url(date: &str, hour: u32, member: &str, product: &str, fhour: u32) -> String {
        let member_code = Self::member_code(member);
        let product_code = Self::product_code(product);

        match member {
            "mean" => format!(
                "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod/gefs.{}/{:02}/atmos/{}sp5/geavg.t{:02}z.{}.0p50.f{:03}",
                date, hour, product_code, hour, product_code, fhour
            ),
            "spread" => format!(
                "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod/gefs.{}/{:02}/atmos/{}sp5/gespr.t{:02}z.{}.0p50.f{:03}",
                date, hour, product_code, hour, product_code, fhour
            ),
            _ => format!(
                "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod/gefs.{}/{:02}/atmos/{}sp5/{}.t{:02}z.{}.0p50.f{:03}",
                date, hour, product_code, member_code, hour, product_code, fhour
            ),
        }
    }

    /// Convert user-facing member identifier to the GEFS file naming convention.
    ///
    /// Accepts:
    /// - `"c00"` or `"control"` or `"0"` → `"gec00"` (control)
    /// - `"p01"`-`"p30"` or `"1"`-`"30"` → `"gep01"`-`"gep30"` (perturbation)
    /// - `"mean"` / `"spread"` → handled separately in URL construction
    fn member_code(member: &str) -> String {
        match member {
            "control" | "c00" | "0" => "gec00".to_string(),
            "mean" => "geavg".to_string(),
            "spread" => "gespr".to_string(),
            s if s.starts_with('p') => format!("ge{}", s),
            s => {
                // Numeric string: interpret as perturbation member number
                if let Ok(n) = s.parse::<u32>() {
                    if n == 0 {
                        "gec00".to_string()
                    } else {
                        format!("gep{:02}", n)
                    }
                } else {
                    format!("ge{}", s)
                }
            }
        }
    }

    fn product_code(product: &str) -> &str {
        match product {
            "pgrb2b" | "secondary" | "b" => "pgrb2b",
            _ => "pgrb2a",
        }
    }

    /// Number of ensemble members (control + 30 perturbations).
    pub fn num_members() -> u32 { 31 }

    /// List all member codes (`"c00"`, `"p01"`, ..., `"p30"`).
    pub fn all_member_codes() -> Vec<String> {
        let mut members = vec!["c00".to_string()];
        for i in 1..=30 {
            members.push(format!("p{:02}", i));
        }
        members
    }

    // --- Grid specifications (0.5° global for ensemble, 0.25° for control) ---

    pub fn grid_nx() -> u32 { 720 }
    pub fn grid_ny() -> u32 { 361 }
    pub fn grid_dx() -> f64 { 0.50 } // degrees (ensemble members)
    pub fn grid_dy() -> f64 { 0.50 }

    pub fn control_grid_nx() -> u32 { 1440 }
    pub fn control_grid_ny() -> u32 { 721 }
    pub fn control_grid_dx() -> f64 { 0.25 }
    pub fn control_grid_dy() -> f64 { 0.25 }

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
    pub fn sfc_precip() -> &'static str { "APCP:surface" }
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
    fn test_aws_url_control() {
        let url = GefsConfig::aws_url("20260310", 0, "c00", "pgrb2a", 24);
        assert!(url.starts_with("https://noaa-gefs-pds.s3.amazonaws.com/"));
        assert!(url.contains("gefs.20260310"));
        assert!(url.contains("gec00"));
    }

    #[test]
    fn test_aws_url_perturbation() {
        let url = GefsConfig::aws_url("20260310", 6, "p05", "pgrb2a", 48);
        assert!(url.contains("gep05"));
    }

    #[test]
    fn test_aws_url_mean() {
        let url = GefsConfig::aws_url("20260310", 12, "mean", "pgrb2a", 12);
        assert!(url.contains("geavg"));
    }

    #[test]
    fn test_aws_url_spread() {
        let url = GefsConfig::aws_url("20260310", 18, "spread", "pgrb2a", 6);
        assert!(url.contains("gespr"));
    }

    #[test]
    fn test_aws_url_numeric_member() {
        let url = GefsConfig::aws_url("20260310", 0, "5", "pgrb2a", 0);
        assert!(url.contains("gep05"));
    }

    #[test]
    fn test_aws_url_numeric_member_zero() {
        let url = GefsConfig::aws_url("20260310", 0, "0", "pgrb2a", 0);
        assert!(url.contains("gec00"));
    }

    #[test]
    fn test_nomads_url_control() {
        let url = GefsConfig::nomads_url("20260310", 0, "c00", "pgrb2a", 24);
        assert!(url.starts_with("https://nomads.ncep.noaa.gov/"));
        assert!(url.contains("gec00"));
    }

    #[test]
    fn test_nomads_url_mean() {
        let url = GefsConfig::nomads_url("20260310", 0, "mean", "pgrb2a", 12);
        assert!(url.contains("geavg"));
    }

    #[test]
    fn test_idx_url() {
        let url = GefsConfig::idx_url("20260310", 0, "c00", "pgrb2a", 0);
        assert!(url.ends_with(".idx"));
    }

    #[test]
    fn test_all_member_codes() {
        let members = GefsConfig::all_member_codes();
        assert_eq!(members.len(), 31);
        assert_eq!(members[0], "c00");
        assert_eq!(members[1], "p01");
        assert_eq!(members[30], "p30");
    }

    #[test]
    fn test_num_members() {
        assert_eq!(GefsConfig::num_members(), 31);
    }

    #[test]
    fn test_grid_specs() {
        assert_eq!(GefsConfig::grid_nx(), 720);
        assert_eq!(GefsConfig::grid_ny(), 361);
        assert_eq!(GefsConfig::control_grid_nx(), 1440);
    }

    #[test]
    fn test_product_code_secondary() {
        let url = GefsConfig::aws_url("20260310", 0, "c00", "pgrb2b", 0);
        assert!(url.contains("pgrb2b"));
    }
}
