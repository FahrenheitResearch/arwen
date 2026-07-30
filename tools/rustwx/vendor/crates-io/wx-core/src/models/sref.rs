/// Configuration and URL generation for the SREF (Short-Range Ensemble Forecast).
///
/// SREF combines ARW and NMMB (formerly NMB) dynamical cores with multiple
/// perturbations to produce short-range ensemble guidance.
/// Runs every 6 hours (03/09/15/21z), forecasts out to 87 hours.
///
/// **Note:** SREF is being phased out in favor of the RRFS ensemble.
pub struct SrefConfig;

impl SrefConfig {
    /// NOMADS URL for SREF GRIB2 files.
    ///
    /// - `date`: format `"YYYYMMDD"` (e.g. `"20260310"`)
    /// - `hour`: model initialization hour (3, 9, 15, 21)
    /// - `member`: e.g. `"arw_ctl"`, `"arw_p01"`, `"arw_n01"`, `"nmb_ctl"`, etc.
    ///   Also: `"mean"`, `"spread"`, `"prob"` for derived products
    /// - `fhour`: forecast hour (0-87, 3-hour intervals)
    pub fn nomads_url(date: &str, hour: u32, member: &str, fhour: u32) -> String {
        let member_code = Self::member_code(member);
        format!(
            "https://nomads.ncep.noaa.gov/pub/data/nccf/com/sref/prod/sref.{}/{:02}/ensprod/sref.t{:02}z.pgrb212.{}.f{:03}.grib2",
            date, hour, hour, member_code, fhour
        )
    }

    /// Ensemble product URL (mean/spread/probability).
    pub fn ensprod_url(date: &str, hour: u32, product: &str, fhour: u32) -> String {
        let product_code = match product {
            "spread" | "spr" => "spread",
            "prob" | "probability" => "prob",
            _ => "mean",
        };
        format!(
            "https://nomads.ncep.noaa.gov/pub/data/nccf/com/sref/prod/sref.{}/{:02}/ensprod/sref.t{:02}z.pgrb212.{}.f{:03}.grib2",
            date, hour, hour, product_code, fhour
        )
    }

    /// IDX file URL.
    pub fn idx_url(date: &str, hour: u32, member: &str, fhour: u32) -> String {
        format!("{}.idx", Self::nomads_url(date, hour, member, fhour))
    }

    fn member_code(member: &str) -> &str {
        match member {
            "mean" => "mean",
            "spread" | "spr" => "spread",
            "prob" | "probability" => "prob",
            other => other, // e.g. "arw_ctl", "arw_p01", "nmb_ctl", "nmb_n01"
        }
    }

    /// List available member types.
    /// ARW: ctl, p01-p07, n01-n07 (15 members)
    /// NMB: ctl, p01-p07, n01-n07 (15 members)
    /// Total: 26 unique members (some overlap handling)
    pub fn arw_members() -> Vec<String> {
        let mut members = vec!["arw_ctl".to_string()];
        for i in 1..=7 {
            members.push(format!("arw_p{:02}", i));
            members.push(format!("arw_n{:02}", i));
        }
        members
    }

    pub fn nmb_members() -> Vec<String> {
        let mut members = vec!["nmb_ctl".to_string()];
        for i in 1..=7 {
            members.push(format!("nmb_p{:02}", i));
            members.push(format!("nmb_n{:02}", i));
        }
        members
    }

    // --- Grid specifications (NCEP Grid 212, 40km Lambert conformal) ---

    pub fn grid_nx() -> u32 { 185 }
    pub fn grid_ny() -> u32 { 129 }
    pub fn grid_dx() -> f64 { 40635.0 } // meters
    pub fn grid_dy() -> f64 { 40635.0 }

    // --- Common variable patterns for .idx matching ---

    pub fn sfc_temp_2m() -> &'static str { "TMP:2 m above ground" }
    pub fn sfc_dewpoint_2m() -> &'static str { "DPT:2 m above ground" }
    pub fn sfc_rh_2m() -> &'static str { "RH:2 m above ground" }
    pub fn sfc_u_wind_10m() -> &'static str { "UGRD:10 m above ground" }
    pub fn sfc_v_wind_10m() -> &'static str { "VGRD:10 m above ground" }
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
    fn test_nomads_url_member() {
        let url = SrefConfig::nomads_url("20260310", 9, "arw_ctl", 24);
        assert!(url.starts_with("https://nomads.ncep.noaa.gov/"));
        assert!(url.contains("sref.20260310"));
        assert!(url.contains("arw_ctl"));
        assert!(url.contains(".f024."));
    }

    #[test]
    fn test_nomads_url_mean() {
        let url = SrefConfig::nomads_url("20260310", 3, "mean", 12);
        assert!(url.contains(".mean."));
    }

    #[test]
    fn test_ensprod_url_spread() {
        let url = SrefConfig::ensprod_url("20260310", 9, "spread", 24);
        assert!(url.contains(".spread."));
    }

    #[test]
    fn test_ensprod_url_default_mean() {
        let url = SrefConfig::ensprod_url("20260310", 9, "unknown", 12);
        assert!(url.contains(".mean."));
    }

    #[test]
    fn test_idx_url() {
        let url = SrefConfig::idx_url("20260310", 3, "arw_ctl", 0);
        assert!(url.ends_with(".grib2.idx"));
    }

    #[test]
    fn test_arw_members() {
        let members = SrefConfig::arw_members();
        assert_eq!(members.len(), 15); // ctl + 7 positive + 7 negative
        assert_eq!(members[0], "arw_ctl");
    }

    #[test]
    fn test_nmb_members() {
        let members = SrefConfig::nmb_members();
        assert_eq!(members.len(), 15);
        assert_eq!(members[0], "nmb_ctl");
    }

    #[test]
    fn test_grid_specs() {
        assert_eq!(SrefConfig::grid_nx(), 185);
        assert_eq!(SrefConfig::grid_ny(), 129);
    }
}
