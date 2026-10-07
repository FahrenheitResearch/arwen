//! Physical constants (specification section 3.2) as exact binary32 words.
//!
//! The dynamical core's constants, so post-processed heights and pressures
//! agree with the forecast that wrote the history: g = 9.81, Rd = 287,
//! cp = 7 Rd / 2, Rv = 461.6, p0 = 1e5 ([R1], WRF ARW technical note), and
//! Lv = 2.501e6 at 0 C ([R2], AMS Glossary).  Saturation constants are the
//! AERK and AERKi forms of [R6]; the equivalent potential temperature
//! constants are those of [R9] eqs. 21 and 43.
//!
//! The bit patterns are written out so the CUDA source
//! (`kernels/post_common.cuh`, `WP_*`) carries the same words; the test
//! below checks each against the decimal value it stands for.

/// 9.81 m s-2.
pub const G: f32 = f32::from_bits(0x411c_f5c3);
/// 287 J kg-1 K-1.
pub const RD: f32 = f32::from_bits(0x438f_8000);
/// 461.6 J kg-1 K-1.
pub const RV: f32 = f32::from_bits(0x43e6_cccd);
/// 100000 Pa.
pub const P0: f32 = f32::from_bits(0x47c3_5000);
/// Rd / Rv, rounded once.
pub const EPS: f32 = f32::from_bits(0x3f1f_2b09);
/// 1 - Rd / Rv, rounded once.
pub const ONE_EPS: f32 = f32::from_bits(0x3ec1_a9ee);
/// Rd / cp with cp = 7 Rd / 2, rounded once.
pub const KAPPA: f32 = f32::from_bits(0x3e92_4925);
/// 2.501e6 J kg-1.
pub const LV: f32 = f32::from_bits(0x4a18_a620);
/// 273.15 K.
pub const T0C: f32 = f32::from_bits(0x4388_9333);
/// AERK: 610.94 Pa, 17.625, 243.04 C.
pub const ESW_A: f32 = f32::from_bits(0x4418_bc29);
pub const ESW_B: f32 = f32::from_bits(0x418d_0000);
pub const ESW_C: f32 = f32::from_bits(0x4373_0a3d);
/// AERKi: 611.21 Pa, 22.587, 273.86 C.
pub const ESI_A: f32 = f32::from_bits(0x4418_cd71);
pub const ESI_B: f32 = f32::from_bits(0x41b4_b22d);
pub const ESI_C: f32 = f32::from_bits(0x4388_ee14);
/// [R9] eq. 21: T_L = 2840 / (3.5 ln T - ln e - 4.805) + 55.
pub const BL_A: f32 = f32::from_bits(0x4531_8000);
pub const BL_B: f32 = f32::from_bits(0x4060_0000);
pub const BL_C: f32 = f32::from_bits(0x4099_c28f);
pub const BL_D: f32 = f32::from_bits(0x425c_0000);
/// [R9] eq. 43: exponent 0.2854 (1 - 0.28e-3 r).
pub const BK: f32 = f32::from_bits(0x3e92_1ff3);
pub const BK_R: f32 = f32::from_bits(0x3992_ccf7);
/// [R9] eq. 43: exp((3.376 / T_L - 0.00254) r (1 + 0.81e-3 r)).
pub const BE_A: f32 = f32::from_bits(0x4058_1062);
pub const BE_B: f32 = f32::from_bits(0x3b26_7621);
pub const BE_C: f32 = f32::from_bits(0x3a54_562e);
/// 0.01 hPa per Pa.
pub const HPA: f32 = f32::from_bits(0x3c23_d70a);
/// pi / 180, rounded once.
pub const DEG2RAD: f32 = f32::from_bits(0x3c8e_fa35);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_are_the_decimal_constants() {
        let cases: [(f32, f64); 26] = [
            (G, 9.81),
            (RD, 287.0),
            (RV, 461.6),
            (P0, 100000.0),
            (EPS, 287.0 / 461.6),
            (ONE_EPS, 1.0 - 287.0 / 461.6),
            (KAPPA, 2.0 / 7.0),
            (LV, 2.501e6),
            (T0C, 273.15),
            (ESW_A, 610.94),
            (ESW_B, 17.625),
            (ESW_C, 243.04),
            (ESI_A, 611.21),
            (ESI_B, 22.587),
            (ESI_C, 273.86),
            (BL_A, 2840.0),
            (BL_B, 3.5),
            (BL_C, 4.805),
            (BL_D, 55.0),
            (BK, 0.2854),
            (BK_R, 0.28e-3),
            (BE_A, 3.376),
            (BE_B, 0.00254),
            (BE_C, 0.81e-3),
            (HPA, 0.01),
            (DEG2RAD, std::f64::consts::PI / 180.0),
        ];
        for (word, value) in cases {
            let rel = ((word as f64) - value).abs() / value.abs();
            assert!(rel < 1.2e-7, "{word} vs {value}");
        }
        // The two derived words are the binary32 operations they name.
        assert_eq!(EPS, RD / RV);
        assert_eq!(KAPPA, RD / (3.5 * RD));
    }
}
