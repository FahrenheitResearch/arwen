//! The ABI L2 cloud-product suite for DA-grade ingest: product/variable
//! tables for ACHA, ACM, ACTP, COD, CPS and CTP, and decode entry points
//! that read the product's primary variable TOGETHER with its `DQF`
//! quality-flag companion and gate every non-good pixel to NaN, with the
//! counts recorded. Nothing is ever fabricated: a pixel the quality flags
//! condemn is NaN plus a count, never a value.
//!
//! Product tokens, primary variable names and sector availability were
//! verified against live `noaa-goes19` bucket listings and downloaded
//! granules on 2026-08-05 (day 216/2026): `HT` (ACHA, m), `BCM` (ACM),
//! `Phase` (ACTP), `COD` (COD), `CPS` (CPS, µm — the older `PSD` name does
//! not appear in current files), `PRES` (CTP, hPa). COD and CTP publish no
//! mesoscale sector under any prefix variant; the other four publish
//! CONUS, full-disk and both mesoscale views.
//!
//! ## The two DQF conventions, measured
//!
//! The six products carry two incompatible DQF encodings, so the gate is
//! product-aware ([`DqfRule`]):
//!
//! * **Enumerated** (ACHA, ACM, ACTP, CTP): small enum, `0` =
//!   good-quality. Good is exactly `DQF == 0`; everything else — including
//!   an unreadable or fill DQF — gates the pixel. This is the literal
//!   fail-closed rule.
//! * **Bitfield** (COD and CPS, the DCOMP pair): a `u16` of paired
//!   provenance and degradation bits. Measured on
//!   `OR_ABI-L2-CODC-M6_G19_s20262161801170...` (and its CPS twin, whose
//!   DQF plane is bit-identical): every daytime pixel carries bit 2
//!   (`not_night_algorithm_pixel_qf`), and **every cloudy retrieval in the
//!   granule — 1,481,473 of them — carries bits 4 and 32**
//!   (`degraded_quality_qf`, `nonconvergence`). A `DQF != 0` gate, or even
//!   an any-quality-bit gate, therefore condemns 100% of actual cloud
//!   retrievals and keeps only clear sky: fail-closed to the letter and
//!   useless in fact. The default [`DqfRule::Bitfield`] mask instead
//!   condemns the three degradation causes that mark a compromised scene —
//!   snow/sea-ice surface (8), twilight (16) and sun glint (64) — plus any
//!   missing/fill DQF, and keeps the wholesale bits. That retains
//!   1,361,938 of the 1,481,473 retrievals in the measured granule while
//!   still gating the pixels the algorithm itself distrusts for a stated
//!   physical reason. [`DCOMP_CONDEMN_STRICT`] restores the literal rule
//!   for consumers who want it, and any custom mask can be passed through
//!   the `*_with_rule` entry points.

use std::error::Error;
use std::io;
use std::path::Path;

use crate::abi::{GoesAbiField, read_goes_abi_field, read_goes_abi_field_window};
use crate::s3::Sector;

/// The L2 cloud-suite products this module knows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloudProduct {
    /// ABI-L2-ACHA: cloud-top height (`HT`, m).
    CloudTopHeight,
    /// ABI-L2-ACM: clear-sky mask (`BCM`, 0 = clear, 1 = cloudy).
    ClearSkyMask,
    /// ABI-L2-ACTP: cloud-top phase (`Phase`, 0..=5).
    CloudTopPhase,
    /// ABI-L2-COD: cloud optical depth (`COD`, unitless).
    OpticalDepth,
    /// ABI-L2-CPS: cloud particle size (`CPS`, µm).
    ParticleSize,
    /// ABI-L2-CTP: cloud-top pressure (`PRES`, hPa).
    CloudTopPressure,
}

/// Default condemn mask for the DCOMP (COD/CPS) DQF bitfield: snow/sea-ice
/// surface (8), twilight (16), sun glint (64). See the module docs for the
/// measurement this rests on.
pub const DCOMP_CONDEMN_DEFAULT: u16 = 8 | 16 | 64;

/// The literal fail-closed mask for the DCOMP bitfield: every bit except
/// the two day/night provenance bits condemns. Measured to keep zero
/// cloudy retrievals on real GOES-19 granules — provided for consumers who
/// want clear-sky-only reads, not as a usable cloud gate.
pub const DCOMP_CONDEMN_STRICT: u16 = !0b11;

/// How a product's DQF plane marks a pixel good.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DqfRule {
    /// Good is exactly `DQF == 0`; anything else (including fill) gates.
    Enumerated,
    /// Good is a finite DQF with no condemned bit set; a missing/fill DQF
    /// gates.
    Bitfield { condemn: u16 },
}

impl DqfRule {
    /// Whether one decoded DQF value marks its pixel good. Decoded fill /
    /// out-of-range DQF arrives here as NaN and is never good.
    pub fn is_good(self, dqf: f32) -> bool {
        match self {
            Self::Enumerated => dqf == 0.0,
            Self::Bitfield { condemn } => {
                dqf.is_finite()
                    && (0.0..=f32::from(u16::MAX)).contains(&dqf)
                    && (dqf as u16) & condemn == 0
            }
        }
    }
}

impl CloudProduct {
    pub const ALL: [Self; 6] = [
        Self::CloudTopHeight,
        Self::ClearSkyMask,
        Self::CloudTopPhase,
        Self::OpticalDepth,
        Self::ParticleSize,
        Self::CloudTopPressure,
    ];

    pub fn parse(value: &str) -> Option<Self> {
        let normalized = value.trim().to_ascii_lowercase().replace(['-', ' '], "_");
        match normalized.as_str() {
            "acha" | "cloud_top_height" | "height" => Some(Self::CloudTopHeight),
            "acm" | "clear_sky_mask" | "cloud_mask" => Some(Self::ClearSkyMask),
            "actp" | "cloud_top_phase" | "phase" => Some(Self::CloudTopPhase),
            "cod" | "optical_depth" | "cloud_optical_depth" => Some(Self::OpticalDepth),
            "cps" | "particle_size" | "cloud_particle_size" => Some(Self::ParticleSize),
            "ctp" | "cloud_top_pressure" | "pressure" => Some(Self::CloudTopPressure),
            _ => None,
        }
    }

    /// The product family token inside S3 prefixes and filenames.
    pub fn family(self) -> &'static str {
        match self {
            Self::CloudTopHeight => "ACHA",
            Self::ClearSkyMask => "ACM",
            Self::CloudTopPhase => "ACTP",
            Self::OpticalDepth => "COD",
            Self::ParticleSize => "CPS",
            Self::CloudTopPressure => "CTP",
        }
    }

    /// Store-safe lowercase slug.
    pub fn slug(self) -> &'static str {
        match self {
            Self::CloudTopHeight => "acha",
            Self::ClearSkyMask => "acm",
            Self::CloudTopPhase => "actp",
            Self::OpticalDepth => "cod",
            Self::ParticleSize => "cps",
            Self::CloudTopPressure => "ctp",
        }
    }

    /// The primary NetCDF variable, as named in current granules
    /// (verified against real GOES-19 files, 2026-08-05).
    pub fn primary_variable(self) -> &'static str {
        match self {
            Self::CloudTopHeight => "HT",
            Self::ClearSkyMask => "BCM",
            Self::CloudTopPhase => "Phase",
            Self::OpticalDepth => "COD",
            Self::ParticleSize => "CPS",
            Self::CloudTopPressure => "PRES",
        }
    }

    /// The quality-flag companion variable. One name across the suite.
    pub fn dqf_variable(self) -> &'static str {
        "DQF"
    }

    /// The filename product token for a sector (`ABI-L2-ACHAC`,
    /// `ABI-L2-ACHAM1`, ...), or `None` where NOAA publishes no such
    /// sector: COD and CTP have no mesoscale product (verified against
    /// `noaa-goes19` listings, 2026-08-05).
    pub fn abi_product(self, sector: Sector) -> Option<String> {
        let sector_token = match sector {
            Sector::Conus => "C",
            Sector::FullDisk => "F",
            Sector::Meso1 | Sector::Meso2 => {
                if matches!(self, Self::OpticalDepth | Self::CloudTopPressure) {
                    return None;
                }
                if sector == Sector::Meso1 { "M1" } else { "M2" }
            }
        };
        Some(format!("ABI-L2-{}{}", self.family(), sector_token))
    }

    /// The product's default DQF gate. See the module docs for the
    /// measured basis of the DCOMP bitfield default.
    pub fn dqf_rule(self) -> DqfRule {
        match self {
            Self::OpticalDepth | Self::ParticleSize => DqfRule::Bitfield {
                condemn: DCOMP_CONDEMN_DEFAULT,
            },
            _ => DqfRule::Enumerated,
        }
    }
}

/// What the DQF gate did to one decoded plane.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DqfReport {
    /// Pixels in the plane.
    pub total: usize,
    /// Pixels whose primary value was already NaN before the gate (fill /
    /// out of valid range).
    pub primary_missing: usize,
    /// Pixels whose DQF itself decoded to NaN (fill / out of range).
    /// Always gated: an unreadable quality flag is not a good one.
    pub dqf_missing: usize,
    /// Pixels the rule condemned (includes `dqf_missing`).
    pub dqf_bad: usize,
    /// Pixels the gate newly forced to NaN (condemned AND previously
    /// finite).
    pub masked: usize,
    /// Finite pixels remaining after the gate.
    pub finite: usize,
}

/// A decoded, DQF-gated cloud-product plane.
#[derive(Debug, Clone, PartialEq)]
pub struct CloudProductField {
    /// The primary variable with every condemned pixel forced to NaN.
    pub field: GoesAbiField,
    pub product: CloudProduct,
    pub dqf: DqfReport,
}

/// Read and gate a full cloud-product plane using the product's default
/// DQF rule.
pub fn read_cloud_product_field(
    path: impl AsRef<Path>,
    product: CloudProduct,
) -> Result<CloudProductField, Box<dyn Error>> {
    read_cloud_product_field_with_rule(path, product, product.dqf_rule())
}

/// Read and gate a full cloud-product plane with an explicit DQF rule.
pub fn read_cloud_product_field_with_rule(
    path: impl AsRef<Path>,
    product: CloudProduct,
    rule: DqfRule,
) -> Result<CloudProductField, Box<dyn Error>> {
    let path = path.as_ref();
    let mut field = read_goes_abi_field(path, product.primary_variable())?;
    let dqf = read_goes_abi_field(path, product.dqf_variable())?;
    let report = gate_by_dqf(&mut field.values, &dqf.values, rule)?;
    Ok(CloudProductField {
        field,
        product,
        dqf: report,
    })
}

/// Read and gate a window of a cloud-product plane using the product's
/// default DQF rule. Window arguments follow
/// [`read_goes_abi_field_window`].
pub fn read_cloud_product_field_window(
    path: impl AsRef<Path>,
    product: CloudProduct,
    x_start: usize,
    x_count: usize,
    y_start: usize,
    y_count: usize,
) -> Result<CloudProductField, Box<dyn Error>> {
    read_cloud_product_field_window_with_rule(
        path,
        product,
        product.dqf_rule(),
        x_start,
        x_count,
        y_start,
        y_count,
    )
}

/// Read and gate a window of a cloud-product plane with an explicit DQF
/// rule.
pub fn read_cloud_product_field_window_with_rule(
    path: impl AsRef<Path>,
    product: CloudProduct,
    rule: DqfRule,
    x_start: usize,
    x_count: usize,
    y_start: usize,
    y_count: usize,
) -> Result<CloudProductField, Box<dyn Error>> {
    let path = path.as_ref();
    let mut field = read_goes_abi_field_window(
        path,
        product.primary_variable(),
        x_start,
        x_count,
        y_start,
        y_count,
    )?;
    let dqf = read_goes_abi_field_window(
        path,
        product.dqf_variable(),
        x_start,
        x_count,
        y_start,
        y_count,
    )?;
    let report = gate_by_dqf(&mut field.values, &dqf.values, rule)?;
    Ok(CloudProductField {
        field,
        product,
        dqf: report,
    })
}

/// Force every pixel `rule` condemns to NaN, in place, and account for
/// every pixel. Pure; the decode entry points feed it and tests pin it.
pub fn gate_by_dqf(
    values: &mut [f32],
    dqf: &[f32],
    rule: DqfRule,
) -> Result<DqfReport, Box<dyn Error>> {
    if values.len() != dqf.len() {
        return Err(boxed_error(format!(
            "DQF plane length {} does not match primary plane length {}",
            dqf.len(),
            values.len()
        )));
    }
    let mut report = DqfReport {
        total: values.len(),
        ..DqfReport::default()
    };
    for (value, &flag) in values.iter_mut().zip(dqf.iter()) {
        let primary_was_finite = value.is_finite();
        if !primary_was_finite {
            report.primary_missing += 1;
        }
        if !flag.is_finite() {
            report.dqf_missing += 1;
        }
        if !rule.is_good(flag) {
            report.dqf_bad += 1;
            if primary_was_finite {
                report.masked += 1;
            }
            *value = f32::NAN;
        }
        if value.is_finite() {
            report.finite += 1;
        }
    }
    Ok(report)
}

fn boxed_error(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidData, message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::s3::product_hour_prefix;
    use chrono::{TimeZone, Utc};

    #[test]
    fn product_tokens_match_live_bucket_keys() {
        // Pinned against listings taken from noaa-goes19 on 2026-08-05.
        assert_eq!(
            CloudProduct::CloudTopHeight
                .abi_product(Sector::Conus)
                .as_deref(),
            Some("ABI-L2-ACHAC")
        );
        assert_eq!(
            CloudProduct::CloudTopHeight
                .abi_product(Sector::Meso1)
                .as_deref(),
            Some("ABI-L2-ACHAM1")
        );
        assert_eq!(
            CloudProduct::ClearSkyMask
                .abi_product(Sector::Meso2)
                .as_deref(),
            Some("ABI-L2-ACMM2")
        );
        assert_eq!(
            CloudProduct::CloudTopPhase
                .abi_product(Sector::FullDisk)
                .as_deref(),
            Some("ABI-L2-ACTPF")
        );
        assert_eq!(
            CloudProduct::OpticalDepth
                .abi_product(Sector::Conus)
                .as_deref(),
            Some("ABI-L2-CODC")
        );
        // NOAA publishes no mesoscale COD or CTP.
        assert_eq!(CloudProduct::OpticalDepth.abi_product(Sector::Meso1), None);
        assert_eq!(
            CloudProduct::CloudTopPressure.abi_product(Sector::Meso2),
            None
        );
        assert_eq!(
            CloudProduct::CloudTopPressure
                .abi_product(Sector::Conus)
                .as_deref(),
            Some("ABI-L2-CTPC")
        );
    }

    #[test]
    fn cloud_prefixes_match_observed_key_layout() {
        use crate::goes::GoesSatellite;
        let hour = Utc.with_ymd_and_hms(2026, 8, 4, 18, 0, 0).unwrap();
        // Real key, 2026-08-05 listing:
        // ABI-L2-ACHAC/2026/216/18/OR_ABI-L2-ACHAC-M6_G19_s20262161801170_...
        assert_eq!(
            product_hour_prefix("ABI-L2-ACHAC", &GoesSatellite::G19, 6, hour),
            "ABI-L2-ACHAC/2026/216/18/OR_ABI-L2-ACHAC-M6_G19_"
        );
        // Mesoscale: shared directory prefix, sector digit kept in the
        // filename token. Real key:
        // ABI-L2-ACHAM/2026/216/18/OR_ABI-L2-ACHAM1-M6_G19_s20262161801249_...
        assert_eq!(
            product_hour_prefix("ABI-L2-ACHAM1", &GoesSatellite::G19, 6, hour),
            "ABI-L2-ACHAM/2026/216/18/OR_ABI-L2-ACHAM1-M6_G19_"
        );
    }

    #[test]
    fn parse_accepts_family_codes_and_long_names() {
        assert_eq!(CloudProduct::parse("ACHA"), Some(CloudProduct::CloudTopHeight));
        assert_eq!(
            CloudProduct::parse("cloud-top-phase"),
            Some(CloudProduct::CloudTopPhase)
        );
        assert_eq!(CloudProduct::parse("cod"), Some(CloudProduct::OpticalDepth));
        assert_eq!(CloudProduct::parse("bogus"), None);
    }

    #[test]
    fn enumerated_rule_is_literal_fail_closed() {
        let rule = DqfRule::Enumerated;
        assert!(rule.is_good(0.0));
        assert!(!rule.is_good(1.0));
        assert!(!rule.is_good(3.0));
        assert!(!rule.is_good(f32::NAN), "unreadable DQF is never good");
    }

    #[test]
    fn bitfield_rule_ignores_provenance_and_condemns_causes() {
        let rule = DqfRule::Bitfield {
            condemn: DCOMP_CONDEMN_DEFAULT,
        };
        // Measured good values from the fixture granule: clean day clear
        // sky (2) and the wholesale-flagged cloudy retrievals (134, 678).
        assert!(rule.is_good(2.0));
        assert!(rule.is_good(134.0));
        assert!(rule.is_good(678.0));
        // Glint (64), twilight (16), snow/ice surface (8) condemn.
        assert!(!rule.is_good(742.0));
        assert!(!rule.is_good(2.0 + 16.0));
        assert!(!rule.is_good(2.0 + 8.0));
        assert!(!rule.is_good(f32::NAN), "fill DQF is never good");

        let strict = DqfRule::Bitfield {
            condemn: DCOMP_CONDEMN_STRICT,
        };
        assert!(strict.is_good(2.0), "provenance bits never condemn");
        assert!(!strict.is_good(134.0), "strict condemns wholesale bits");
    }

    #[test]
    fn gate_accounts_for_every_pixel() {
        let mut values = vec![1.0, 2.0, f32::NAN, 4.0, 5.0];
        let dqf = vec![0.0, 1.0, 0.0, f32::NAN, 0.0];
        let report = gate_by_dqf(&mut values, &dqf, DqfRule::Enumerated).unwrap();
        assert_eq!(
            report,
            DqfReport {
                total: 5,
                primary_missing: 1,
                dqf_missing: 1,
                dqf_bad: 2,
                masked: 2,
                finite: 2,
            }
        );
        assert_eq!(values[0], 1.0);
        assert!(values[1].is_nan(), "condemned pixel gated to NaN");
        assert!(values[3].is_nan(), "fill DQF gates fail-closed");
        assert_eq!(values[4], 5.0);

        let mut short = vec![1.0];
        assert!(gate_by_dqf(&mut short, &dqf, DqfRule::Enumerated).is_err());
    }
}
