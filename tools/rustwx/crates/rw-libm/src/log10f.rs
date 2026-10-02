//! Correctly rounded binary32 base-10 logarithm: CORE-MATH
//! src/binary32/log10/log10f.c (notice in lib.rs).

use crate::{hf, hf32};

/// Reciprocals of 1 + i/64, i = 0..64, rounded to 29 bits.
const TR: [f64; 65] = [
    hf("0x1p+0"), hf("0x1.f81f82p-1"), hf("0x1.f07c1fp-1"),
    hf("0x1.e9131acp-1"), hf("0x1.e1e1e1ep-1"), hf("0x1.dae6077p-1"),
    hf("0x1.d41d41dp-1"), hf("0x1.cd85689p-1"), hf("0x1.c71c71cp-1"),
    hf("0x1.c0e0704p-1"), hf("0x1.bacf915p-1"), hf("0x1.b4e81b5p-1"),
    hf("0x1.af286bdp-1"), hf("0x1.a98ef6p-1"), hf("0x1.a41a41ap-1"),
    hf("0x1.9ec8e95p-1"), hf("0x1.999999ap-1"), hf("0x1.948b0fdp-1"),
    hf("0x1.8f9c19p-1"), hf("0x1.8acb90fp-1"), hf("0x1.8618618p-1"),
    hf("0x1.8181818p-1"), hf("0x1.7d05f41p-1"), hf("0x1.78a4c81p-1"),
    hf("0x1.745d174p-1"), hf("0x1.702e05cp-1"), hf("0x1.6c16c17p-1"),
    hf("0x1.6816817p-1"), hf("0x1.642c859p-1"), hf("0x1.605816p-1"),
    hf("0x1.5c9882cp-1"), hf("0x1.58ed231p-1"), hf("0x1.5555555p-1"),
    hf("0x1.51d07ebp-1"), hf("0x1.4e5e0a7p-1"), hf("0x1.4afd6ap-1"),
    hf("0x1.47ae148p-1"), hf("0x1.446f865p-1"), hf("0x1.4141414p-1"),
    hf("0x1.3e22cbdp-1"), hf("0x1.3b13b14p-1"), hf("0x1.3813814p-1"),
    hf("0x1.3521cfbp-1"), hf("0x1.323e34ap-1"), hf("0x1.2f684bep-1"),
    hf("0x1.2c9fb4ep-1"), hf("0x1.29e412ap-1"), hf("0x1.27350b9p-1"),
    hf("0x1.2492492p-1"), hf("0x1.21fb781p-1"), hf("0x1.1f7047ep-1"),
    hf("0x1.1cf06aep-1"), hf("0x1.1a7b961p-1"), hf("0x1.1811812p-1"),
    hf("0x1.15b1e5fp-1"), hf("0x1.135c811p-1"), hf("0x1.1111111p-1"),
    hf("0x1.0ecf56cp-1"), hf("0x1.0c9715p-1"), hf("0x1.0a6810ap-1"),
    hf("0x1.0842108p-1"), hf("0x1.0624dd3p-1"), hf("0x1.041041p-1"),
    hf("0x1.0204081p-1"), 0.5,
];
/// Base-10 logarithms of the reciprocals, with an offset.
const TL: [f64; 65] = [
    hf("-0x1.2p-46"), hf("0x1.b947689310dfap-8"), hf("0x1.b5e909c96d11bp-7"),
    hf("0x1.45f4f59ed1e08p-6"), hf("0x1.af5f92cbd8bc1p-6"), hf("0x1.0ba01a606dcdep-5"),
    hf("0x1.3ed119b9a29cdp-5"), hf("0x1.714834298ed14p-5"), hf("0x1.a30a9d983564cp-5"),
    hf("0x1.d41d512670665p-5"), hf("0x1.02428c0f65442p-4"), hf("0x1.1a23444eecb67p-4"),
    hf("0x1.31b30543f4bddp-4"), hf("0x1.48f3ed39bfc2dp-4"), hf("0x1.5fe8049a0e34bp-4"),
    hf("0x1.769140a6a9f3p-4"), hf("0x1.8cf1836c98bdbp-4"), hf("0x1.a30a9d55540cap-4"),
    hf("0x1.b8de4d1ee8167p-4"), hf("0x1.ce6e4202ca20fp-4"), hf("0x1.e3bc1accacd3p-4"),
    hf("0x1.f8c9683b5aafdp-4"), hf("0x1.06cbd68ca9a03p-3"), hf("0x1.11142f19df6c5p-3"),
    hf("0x1.1b3e71fa7a913p-3"), hf("0x1.254b4d37a4677p-3"), hf("0x1.2f3b6912cbe9cp-3"),
    hf("0x1.390f68311581ap-3"), hf("0x1.42c7e7fffc53dp-3"), hf("0x1.4c65808c78cd1p-3"),
    hf("0x1.55e8c50751beap-3"), hf("0x1.5f52445dec36cp-3"), hf("0x1.68a288c3f1195p-3"),
    hf("0x1.71da17bdf0cadp-3"), hf("0x1.7af973608af6ep-3"), hf("0x1.84011952a250ep-3"),
    hf("0x1.8cf1837a7e9f4p-3"), hf("0x1.95cb2891e436ap-3"), hf("0x1.9e8e7b0f86974p-3"),
    hf("0x1.a73beaa5db121p-3"), hf("0x1.afd3e39455867p-3"), hf("0x1.b856cf060d985p-3"),
    hf("0x1.c0c5134de1f9p-3"), hf("0x1.c91f1371bc934p-3"), hf("0x1.d1652ffcd3ee8p-3"),
    hf("0x1.d997c6f635e09p-3"), hf("0x1.e1b733ab90edp-3"), hf("0x1.e9c3ceadac7ebp-3"),
    hf("0x1.f1bdeec43a29ap-3"), hf("0x1.f9a5e7a5fa392p-3"), hf("0x1.00be05ac02ef5p-2"),
    hf("0x1.04a054d81a29ep-2"), hf("0x1.087a0835957c5p-2"), hf("0x1.0c4b4570994e1p-2"),
    hf("0x1.101431aa1fe1bp-2"), hf("0x1.13d4f08b98da2p-2"), hf("0x1.178da53edb85cp-2"),
    hf("0x1.1b3e71e9f9d22p-2"), hf("0x1.1ee777defdeb7p-2"), hf("0x1.2288d7b48e205p-2"),
    hf("0x1.2622b0f52e469p-2"), hf("0x1.29b522a4c62dep-2"), hf("0x1.2d404b0e30f4ap-2"),
    hf("0x1.30c4478f3fbafp-2"), hf("0x1.34413509f78dfp-2"),
];
const B: [f64; 3] = [
    hf("0x1.bcb7b15d35067p-2"), hf("-0x1.bcbb1cd29cbafp-3"), hf("0x1.2870e2624ce4ep-3"),
];
const C: [f64; 7] = [
    hf("0x1.bcb7b1526e50ep-2"), hf("-0x1.bcb7b1526e53dp-3"), hf("0x1.287a7636f3fa2p-3"),
    hf("-0x1.bcb7b146a14b3p-4"), hf("0x1.63c627d5219cbp-4"), hf("-0x1.2880736c8762dp-4"),
    hf("0x1.fc1ecf913961ap-5"),
];
/// The binary32 encodings of 10^n indexed by bits 24..27 of the encoding
/// (0 where no power of ten lands).
const ST: [u32; 16] = [
    hf32("0x1.2a05f2p+33").to_bits(),
    hf32("0x1.4p+3").to_bits(),
    hf32("0x1.9p+6").to_bits(),
    0,
    hf32("0x1.f4p+9").to_bits(),
    0,
    hf32("0x1.388p+13").to_bits(),
    hf32("0x1.86ap+16").to_bits(),
    0,
    hf32("0x1.e848p+19").to_bits(),
    0,
    hf32("0x1.312dp+23").to_bits(),
    hf32("0x1.7d784p+26").to_bits(),
    0,
    hf32("0x1.dcd65p+29").to_bits(),
    hf32("0x1p+0").to_bits(),
];

#[inline(never)]
#[cold]
fn special(x: f32) -> f32 {
    let ux = x.to_bits();
    let ax = ux << 1;
    if ax == 0 {
        return f32::from_bits(0x1ff << 23); // -inf
    }
    if ux == 0x7f80_0000 {
        return x; // +inf
    }
    if ax > 0xff00_0000 {
        return x + x; // NaN
    }
    f32::from_bits(0x1ff8 << 19) // x < 0: NaN
}

/// Correctly rounded log10(x) for binary32 (round to nearest).
#[inline]
pub fn log10f(x: f32) -> f32 {
    // ln(2)/ln(10) in one and two parts
    const LN10: f64 = hf("0x1.34413509f79ffp-2");
    const LN10H: f64 = hf("0x1.34413509f7ap-2");
    const LN10L: f64 = hf("-0x1.0cee0ed4ca7e9p-54");
    let mut ux = x.to_bits();
    if ux >= 0x7f80_0000 {
        return special(x); // <= -0, NaN, infinity
    }
    if ux == ST[((ux >> 24) & 0xf) as usize] {
        // x = 10^n
        let mut je = (((ux as i32) >> 23) - 126) as u32;
        je = je.wrapping_mul(0x4d104d4) >> 28;
        return je as f32;
    }
    if ux < 0x0080_0000 {
        if ux == 0 {
            return special(x); // +0
        }
        // subnormal
        let n = ux.leading_zeros() - 8;
        ux <<= n;
        ux = ux.wrapping_sub(n << 23);
    }
    let e = (((ux as i32) >> 23) - 127) as f64;
    let m = (ux & ((1 << 23) - 1)) as u64;
    let j = ((m + (1 << (23 - 7))) >> (23 - 6)) as usize;
    let tz = f64::from_bits((m << (52 - 23)) | (0x3ff << 52));
    let z = tz * TR[j] - 1.0;
    let z2 = z * z;
    let mut r = ((e * LN10 + TL[j]) + z * B[0]) + z2 * (B[1] + z * B[2]);
    let mut ub = r as f32;
    let lb = (r + const { hf("0x1.af23fp-34") }) as f32;
    if ub != lb {
        let mut f = z
            * ((C[0] + z * C[1]) + z2 * ((C[2] + z * C[3]) + z2 * (C[4] + z * C[5] + z2 * C[6])));
        f += LN10L * e;
        f += TL[j] - TL[0];
        let el = e * LN10H;
        r = el + f;
        ub = r as f32;
        if r.to_bits() & ((1 << 28) - 1) == 0 {
            let dr = (el - r) + f;
            r += dr * 32.0;
            ub = r as f32;
        }
    }
    ub
}
