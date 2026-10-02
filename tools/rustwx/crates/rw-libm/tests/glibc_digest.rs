//! Every function against digests of the reference library's own outputs.
//!
//! THE BREAKAGE THIS PREVENTS.  The WPS orographic projection called the
//! platform C library through `f32::powf`, `f32::cos` and friends, and the
//! oracle it is graded against was captured with glibc 2.43.  On glibc 2.39
//! and the Windows UCRT the last bits differed, the Lambert inverse turned
//! one bit into hundreds of ULP, and public CI run 37036982597 failed on
//! both operating systems while the terrain-drag statics quietly differed by
//! platform.  rw-libm replaces those calls; this test holds it, on every
//! platform the gate runs on, to the exact outputs of the reference:
//!
//! * binary32 sinf, cosf, tanf, atanf, expf, logf, log10f, powf, atan2f:
//!   glibc 2.43 itself (Ubuntu 2.43-2ubuntu2.4, x86-64 with FMA, node-2);
//! * binary64 asin, acos: CORE-MATH's C cr_asin / cr_acos (glibc's own are
//!   IBM's LGPL code, not correctly rounded, and not transcribed).
//!
//! The digests were produced by tools/rw_libm_proof/glibc_digest.c, which
//! draws exactly the inputs `inputs_*` below and hashes the reference's
//! outputs the same way.  A mismatch here means a transcription changed
//! (an `a * b + c` where glibc's FMA build fuses, a constant, a branch), or
//! a platform's `mul_add` is not a fused multiply-add.

const H0: u64 = 0xcbf2_9ce4_8422_2325;

fn mix(mut h: u64, v: u64) -> u64 {
    for i in 0..8 {
        h ^= (v >> (8 * i)) & 0xff;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn canon32(x: f32) -> u64 {
    if x.is_nan() { 0x7fc0_0000 } else { x.to_bits() as u64 }
}

fn canon64(x: f64) -> u64 {
    if x.is_nan() { 0x7ff8_0000_0000_0000 } else { x.to_bits() }
}

/// SplitMix64's i-th word for a salt.
fn word(i: u64, salt: u64) -> u64 {
    let mut z = i.wrapping_add(salt).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// 2^20 encodings at a stride of 4097 (every exponent, both signs, NaN and
/// infinity included) and 2^18 random encodings.
fn inputs_f32(salt: u64) -> impl Iterator<Item = f32> {
    (0..1u64 << 20)
        .map(|i| f32::from_bits((i * 4097) as u32))
        .chain((0..1u64 << 18).map(move |i| f32::from_bits(word(i, salt) as u32)))
}

fn inputs_powf() -> impl Iterator<Item = (f32, f32)> {
    (0..1u64 << 18).map(|i| {
        let w = word(i, 21);
        if i % 2 == 0 {
            (f32::from_bits(w as u32), f32::from_bits((w >> 32) as u32))
        } else {
            // a positive base and an exponent that keeps most results finite
            let x = f32::from_bits((w as u32) & 0x7f7f_ffff);
            let y = ((w >> 32) as f64 / 4294967296.0 * 64.0 - 32.0) as f32;
            (x, y)
        }
    })
}

fn inputs_atan2f() -> impl Iterator<Item = (f32, f32)> {
    (0..1u64 << 18).map(|i| {
        let w = word(i, 22);
        if i % 2 == 0 {
            (f32::from_bits(w as u32), f32::from_bits((w >> 32) as u32))
        } else {
            // the projection's own scale: grid offsets of a few thousand
            let y = (((w as u32) as f64 / 4294967296.0 * 2.0 - 1.0) * 3000.0) as f32;
            let x = (((w >> 32) as f64 / 4294967296.0 * 2.0 - 1.0) * 3000.0) as f32;
            (y, x)
        }
    })
}

fn inputs_f64(salt: u64) -> impl Iterator<Item = f64> {
    (0..1u64 << 18).map(move |i| {
        let w = word(i, salt);
        if i % 2 == 0 {
            // every encoding with |x| <= 1, uniformly over bit patterns
            f64::from_bits(w % 0x3ff0_0000_0000_0001 | (w & (1 << 63)))
        } else {
            (w >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        }
    })
}

fn digest1(f: fn(f32) -> f32, salt: u64) -> u64 {
    inputs_f32(salt).fold(H0, |h, x| mix(h, canon32(f(x))))
}

#[test]
fn every_function_matches_the_reference_digest() {
    let mut wrong = Vec::new();
    let mut check = |name: &str, got: u64, want: u64| {
        if got != want {
            wrong.push(format!("{name}: {got:016x}, reference {want:016x}"));
        }
    };
    check("sinf", digest1(rw_libm::sinf, 1), 0xdb47_2efc_e84b_7aba);
    check("cosf", digest1(rw_libm::cosf, 2), 0x99cf_1e32_c97b_9d1a);
    check("tanf", digest1(rw_libm::tanf, 3), 0xf460_7505_29b8_0cdb);
    check("atanf", digest1(rw_libm::atanf, 4), 0x8764_fb96_0f7d_4494);
    check("expf", digest1(rw_libm::expf, 5), 0xb5ec_97f3_9f69_7846);
    check("logf", digest1(rw_libm::logf, 6), 0x1ace_9872_12ae_760a);
    check("log10f", digest1(rw_libm::log10f, 7), 0x3d3d_a09c_d283_d0da);
    check("powf", inputs_powf().fold(H0, |h, (x, y)| mix(h, canon32(rw_libm::powf(x, y)))), 0x3f5a_23de_5c96_62d3);
    check("atan2f", inputs_atan2f().fold(H0, |h, (y, x)| mix(h, canon32(rw_libm::atan2f(y, x)))), 0xc439_aa0d_9262_afed);
    check("asin", inputs_f64(23).fold(H0, |h, x| mix(h, canon64(rw_libm::asin(x)))), 0x73e1_f4ba_5814_f8db);
    check("acos", inputs_f64(24).fold(H0, |h, x| mix(h, canon64(rw_libm::acos(x)))), 0xc634_6a3a_1a9c_53f8);
    assert!(wrong.is_empty(), "outputs differ from the reference library:\n{}", wrong.join("\n"));
}

/// Points the projection and its oracle depend on, by value, so a failure
/// names an argument rather than a digest.  Values are glibc 2.43's.
#[test]
fn the_cases_that_separate_glibc_from_correct_rounding() {
    // glibc 2.43's sinf/cosf/expf/logf are not correctly rounded; these are
    // arguments where its answer and the correctly rounded one differ, and
    // the WPS oracle carries glibc's.
    for (x, want) in [(0x3a12_8600u32, 0x3a12_8600u32), (0x3ac0_0001, 0x3abf_fffd)] {
        assert_eq!(rw_libm::sinf(f32::from_bits(x)).to_bits(), want, "sinf {x:#010x}");
    }
    // the two expf arguments that need glibc's fused range reduction
    for (x, want) in [(0x4202_422fu32, 0x56fc_9f1cu32), (0xc27c_65d9, 0x11fa_2993)] {
        assert_eq!(rw_libm::expf(f32::from_bits(x)).to_bits(), want, "expf {x:#010x}");
    }
}
