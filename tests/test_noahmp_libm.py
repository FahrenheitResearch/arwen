"""gpuwm.core.noahmp_libm must reproduce glibc 2.39 bit for bit.

The vectors below are not hand-written: each row is an input bit pattern and
the answer the *live* glibc 2.39 ``expf`` / ``powf`` returned for it, captured
by ``tools/noahmp_wrf461_oracle/libm_probe.c`` on the pinned WSL host
(Ubuntu 24.04, x86-64, CPU reports fma+avx2, so glibc ifunc-dispatches the
``_fma`` variants).  They are a spread of the same 320,022-case sweep that
``validate_soilwater_oracle.py --libm-sweep`` runs in full against live glibc;
this file is the offline half so the gate still holds on a Windows box with no
glibc in reach.

Coverage that matters, and which row carries it:
  expf   the ordinary band, both signs, the |x| >= 88 fall-through band, the
         overflow edge (88.72280 finite / 88.72284 -> inf), the underflow edge
         (-103.970 -> smallest subnormal / -103.972 -> 0), and +-0.
  powf   the WDFCND band (base 0.01..1, exponent 2..30) including results that
         underflow to exactly zero, the CANWATER exponent 0.667 including
         base 0.01, exact cases (1**3, 0.5**2, 2**5, 1000**3), powf(0, y),
         six rows where numpy float32 ``**`` and FP64-then-round BOTH miss
         glibc (captured 2026-08-30, glibc 2.39-0ubuntu8.7, so the negative
         controls below discriminate), and the sign_bias / subnormal-base
         paths in ``_POWF_PATH_VECTORS``.
"""

from __future__ import annotations

import struct

import pytest

from gpuwm.core.noahmp_libm import GLIBC_VERSION, expf, f32, powf


def _f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u))[0]


def _u32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


_EXPF_VECTORS = (
    (0x42082e50, 0x580acd35),   # expf(34.04522705078125)
    (0x4293c9db, 0x74c2eff4),   # expf(73.89424896240234)
    (0xc23b269a, 0x1db4f8ce),   # expf(-46.787696838378906)
    (0x41955605, 0x4cf40251),   # expf(18.667001724243164)
    (0x41536909, 0x4905aba0),   # expf(13.213143348693848)
    (0x42a2591b, 0x7a0a14df),   # expf(81.17403411865234)
    (0x4004f009, 0x40ff6a08),   # expf(2.077150583267212)
    (0x42a819b0, 0x7c192566),   # expf(84.0501708984375)
    (0xc21b53d1, 0x237c08f0),   # expf(-38.831851959228516)
    (0xc2b9663f, 0x0000998e),   # expf(-92.69969940185547)
    (0x425b7dd7, 0x670f7e79),   # expf(54.87289047241211)
    (0xc1615884, 0x354d348d),   # expf(-14.084110260009766)
    (0x4118b2cb, 0x465a0f35),   # expf(9.54365062713623)
    (0x410a5ed3, 0x45b21cc3),   # expf(8.648150444030762)
    (0xc109dbeb, 0x393df2cd),   # expf(-8.616190910339355)
    (0xc08521f6, 0x3c7f9cf3),   # expf(-4.160395622253418)
    (0x4077596d, 0x423ec800),   # expf(3.864833116531372)
    (0x40c32b40, 0x43deb670),   # expf(6.099029541015625)
    (0xbf872284, 0x3eb2247b),   # expf(-1.0557408332824707)
    (0xbf7bbd78, 0x3ebf83c6),   # expf(-0.9833598136901855)
    (0x401ebdcd, 0x413f1fc9),   # expf(2.480334520339966)
    (0xc0ca2e22, 0x3aec5d0a),   # expf(-6.318131446838379)
    (0x400da6b5, 0x41125565),   # expf(2.2132999897003174)
    (0xbd94bf17, 0x3f6e10dc),   # expf(-0.07263009995222092)
    (0xbf72f621, 0x3ec6322e),   # expf(-0.9490681290626526)
    (0xc02299c7, 0x3da16a6d),   # expf(-2.540635824203491)
    (0xbf5219c1, 0x3ee1578c),   # expf(-0.8207054734230042)
    (0xbfea1d17, 0x3e246cfd),   # expf(-1.8290127515792847)
    (0xc026b87e, 0x3d975a1f),   # expf(-2.605010509490967)
    (0xc046faf7, 0x3d36db1d),   # expf(-3.109067678451538)
    (0xc0800000, 0x3c960aae),   # expf(-4.0)
    (0x00000000, 0x3f800000),   # expf(0.0)
    (0x80000000, 0x3f800000),   # expf(-0.0)
    (0x42b00000, 0x7ef882b7),   # expf(88.0)
    (0x42b17213, 0x7f7ffd84),   # expf(88.7228012084961)
    (0x42b17218, 0x7f800000),   # expf(88.72283935546875)
    (0xc2cff0a4, 0x00000001),   # expf(-103.97000122070312)
    (0xc2cff1b7, 0x00000000),   # expf(-103.97209930419922)
    (0xc2ae0000, 0x00b33687),   # expf(-87.0)
    (0x42ae0000, 0x7e36d809),   # expf(87.0)
    (0x322bcc77, 0x3f800000),   # expf(9.99999993922529e-09)
    (0xb22bcc77, 0x3f800000),   # expf(-9.99999993922529e-09)
    (0x3f800000, 0x402df854),   # expf(1.0)
    (0xbf800000, 0x3ebc5ab2),   # expf(-1.0)
)

_POWF_VECTORS = (
    (0x3f7c7125, 0x41a32b47, 0x3f406d17),   # powf(0.9861014485359192, 20.39613151550293)
    (0x3e9e3d41, 0x4173a55f, 0x32936232),   # powf(0.30906108021736145, 15.227873802185059)
    (0x3f099587, 0x40df5c09, 0x3c56d70e),   # powf(0.5374378561973572, 6.979984760284424)
    (0x3f02978c, 0x4088e2a3, 0x3d661711),   # powf(0.510124921798706, 4.277665615081787)
    (0x3e84a076, 0x41cf15a5, 0x263c0bcb),   # powf(0.25903671979904175, 25.885568618774414)
    (0x3f281c1e, 0x41ee81d9, 0x3670b056),   # powf(0.6566790342330933, 29.81340217590332)
    (0x3ee6143f, 0x41a58c0f, 0x338b0ecf),   # powf(0.4493732154369354, 20.693387985229492)
    (0x3edc3aa6, 0x417f63c8, 0x35be4f25),   # powf(0.43013495206832886, 15.961860656738281)
    (0x3ee4a05c, 0x41b4a620, 0x325500f6),   # powf(0.44653594493865967, 22.58111572265625)
    (0x3f19b24f, 0x41066762, 0x3c617da8),   # powf(0.6003770232200623, 8.400239944458008)
    (0x3e79507d, 0x4147fe67, 0x32b80072),   # powf(0.24347110092639923, 12.49960994720459)
    (0x3f2f58dc, 0x41e27adc, 0x37bab7f8),   # powf(0.6849496364593506, 28.30998992919922)
    (0x3780987d, 0x41b57413, 0x00000000),   # powf(1.532979695184622e-05, 22.681676864624023)
    (0x3ba4f789, 0x4104dfd0, 0x1fc25df6),   # powf(0.005034391302615404, 8.304641723632812)
    (0x3704c4e0, 0x4027ca1c, 0x293dec84),   # powf(7.913651643320918e-06, 2.621710777282715)
    (0x3681acca, 0x414753ac, 0x00000000),   # powf(3.864614882331807e-06, 12.457927703857422)
    (0x37a30ef6, 0x4211f102, 0x00000000),   # powf(1.9438080926192924e-05, 36.48535919189453)
    (0x38b59ddd, 0x41ee6bdf, 0x00000000),   # powf(8.660156890982762e-05, 29.802671432495117)
    (0x37f298b0, 0x41707331, 0x00000000),   # powf(2.8919748729094863e-05, 15.028122901916504)
    (0x3d0d75cc, 0x40eb1ff9, 0x2d9ff80b),   # powf(0.03453616797924042, 7.347652912139893)
    (0x35a4744e, 0x420de1f3, 0x00000000),   # powf(1.2252801298018312e-06, 35.47065353393555)
    (0x41181c35, 0x41f4cb25, 0x712ac62d),   # powf(9.50688648223877, 30.599191665649414)
    (0x3d092d39, 0x420dcf45, 0x00000000),   # powf(0.033490393310785294, 35.45241165161133)
    (0x3f7117e0, 0x3f2ac083, 0x3f75f54e),   # powf(0.9417705535888672, 0.6669999957084656)
    (0x3e7c24d0, 0x3f2ac083, 0x3ec90c3c),   # powf(0.24623417854309082, 0.6669999957084656)
    (0x3ede387a, 0x3f2ac083, 0x3f12b5f6),   # powf(0.4340246319770813, 0.6669999957084656)
    (0x3f506a1d, 0x3f2ac083, 0x3f5f2fde),   # powf(0.8141191601753235, 0.6669999957084656)
    (0x3ed26332, 0x3f2ac083, 0x3f0d73e3),   # powf(0.41091305017471313, 0.6669999957084656)
    (0x3eb15d62, 0x3f2ac083, 0x3efc741a),   # powf(0.34641557931900024, 0.6669999957084656)
    (0x3edc70a7, 0x3f2ac083, 0x3f11ecf8),   # powf(0.43054696917533875, 0.6669999957084656)
    (0x3ec4bdb9, 0x3f2ac083, 0x3f0743cd),   # powf(0.384259968996048, 0.6669999957084656)
    (0x3f1fd87c, 0x3f2ac083, 0x3f3afcc6),   # powf(0.6243970394134521, 0.6669999957084656)
    (0x3e7aae87, 0x3f2ac083, 0x3ec844fc),   # powf(0.24480639398097992, 0.6669999957084656)
    (0x3ebc437a, 0x3f2ac083, 0x3f035949),   # powf(0.36770230531692505, 0.6669999957084656)
    (0x3e04ae72, 0x3f2ac083, 0x3e830351),   # powf(0.12957170605659485, 0.6669999957084656)
    (0x3f5838fd, 0x3f2ac083, 0x3f64badb),   # powf(0.8446195721626282, 0.6669999957084656)
    (0x3c23d70a, 0x3f2ac083, 0x3d3dd3ef),   # powf(0.009999999776482582, 0.6669999957084656)
    (0x3f800000, 0x40400000, 0x3f800000),   # powf(1.0, 3.0)
    (0x3f000000, 0x40000000, 0x3e800000),   # powf(0.5, 2.0)
    # -- discriminating rows, captured 2026-08-30 by the same libm_probe on
    #    the same pinned WSL host (glibc 2.39-0ubuntu8.7).  On each of the
    #    first five, numpy float32 ** and FP64-pow-then-round-once BOTH land
    #    1 ULP off glibc; on the sixth, both underflow to 0 where glibc
    #    returns the smallest subnormal.  They exist so the two negative
    #    controls below have something to catch: the original 44 rows all
    #    happened to sit where the three functions agree.
    (0x3e406f48, 0x41a03065, 0x275105a7),   # powf(0.18792450428009033, 20.023630142211914)
    (0x3f65de55, 0x41e0dde1, 0x3d469beb),   # powf(0.8979237675666809, 28.108339309692383)
    (0x3ea5f9a8, 0x41a8f768, 0x2e4c5c79),   # powf(0.3241703510284424, 21.120803833007812)
    (0x3f5ac1e5, 0x41ceb196, 0x3c8d0953),   # powf(0.8545210957527161, 25.836711883544922)
    (0x3f1086c2, 0x419e8c5d, 0x374955de),   # powf(0.5645562410354614, 19.81853675842285)
    (0x3c48d537, 0x41bcbb6e, 0x00000001),   # powf(0.012257865630090237, 23.59151840209961)
    (0x0da24260, 0x41d0cccd, 0x00000000),   # powf(1.0000000031710769e-30, 26.100000381469727)
    (0x1e3ce508, 0x41c80000, 0x00000000),   # powf(9.999999682655225e-21, 25.0)
    (0x00000000, 0x3f2ac083, 0x00000000),   # powf(0.0, 0.6669999957084656)
    (0x40000000, 0x40a00000, 0x42000000),   # powf(2.0, 5.0)
    (0x447a0000, 0x40400000, 0x4e6e6b28),   # powf(1000.0, 3.0)
)


def test_glibc_version_is_the_pinned_one():
    assert GLIBC_VERSION == "2.39"


@pytest.mark.parametrize("ix,want", _EXPF_VECTORS)
def test_expf_matches_glibc(ix, want):
    got = _u32(expf(_f32(ix)))
    assert got == want, (
        f"expf(0x{ix:08x} = {_f32(ix)!r}): got 0x{got:08x}, glibc 0x{want:08x}"
    )


@pytest.mark.parametrize("ix,iy,want", _POWF_VECTORS)
def test_powf_matches_glibc(ix, iy, want):
    got = _u32(powf(_f32(ix), _f32(iy)))
    assert got == want, (
        f"powf(0x{ix:08x}, 0x{iy:08x}) = powf({_f32(ix)!r}, {_f32(iy)!r}): "
        f"got 0x{got:08x}, glibc 0x{want:08x}"
    )


def test_numpy_float32_exp_is_a_different_function():
    """Negative control.

    If numpy's float32 exp happened to agree with glibc everywhere, the whole
    reason this module exists would evaporate -- and a reviewer could swap it
    in without any gate noticing.  It does not agree; prove it here so the
    substitution is permanently blocked.
    """
    import numpy as np

    disagreements = 0
    for ix, want in _EXPF_VECTORS:
        x = _f32(ix)
        with np.errstate(over="ignore", under="ignore"):
            naive = _u32(float(np.exp(np.float32(x))))
        if naive != want:
            disagreements += 1
    assert disagreements > 0, (
        "numpy float32 exp reproduced glibc on every pinned vector; the "
        "vectors no longer discriminate and the gate is not doing its job"
    )


def test_numpy_float32_power_is_a_different_function():
    """Negative control for powf, same argument as above."""
    import numpy as np

    disagreements = 0
    for ix, iy, want in _POWF_VECTORS:
        x, y = _f32(ix), _f32(iy)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            naive = _u32(float(np.float32(x) ** np.float32(y)))
        if naive != want:
            disagreements += 1
    assert disagreements > 0, (
        "numpy float32 power reproduced glibc on every pinned vector; the "
        "vectors no longer discriminate and the gate is not doing its job"
    )


def test_double_then_round_once_is_a_third_function():
    """The 'just use FP64 and round' shortcut is not glibc.  Prove it."""
    import math

    disagreements = 0
    for ix, iy, want in _POWF_VECTORS:
        x, y = _f32(ix), _f32(iy)
        try:
            naive = _u32(f32(math.pow(x, y)))
        except (OverflowError, ValueError):
            continue
        if naive != want:
            disagreements += 1
    assert disagreements > 0, (
        "FP64-then-round matched glibc on every pinned vector; widen the "
        "vector set before trusting this gate"
    )


#: The two powf paths an earlier revision of ``noahmp_libm`` refused with
#: ``NotImplementedError``: a finite negative base (the sign_bias path) and a
#: subnormal base (the normalization path).  The tracked module implements
#: both in full, so the refusal test died and these pins replaced it.  Every
#: answer is live glibc 2.39, captured 2026-08-30 by the same libm_probe on
#: the same pinned WSL host as the vectors above.
_POWF_PATH_VECTORS = (
    (0xc0000000, 0x40400000, 0xc1000000),   # powf(-2.0, 3.0) = -8.0, odd y
    (0xc0000000, 0x40000000, 0x40800000),   # powf(-2.0, 2.0) = 4.0, even y
    (0xbf000000, 0x40400000, 0xbe000000),   # powf(-0.5, 3.0) = -0.125
    (0x000116c2, 0x40000000, 0x00000000),   # powf(1e-40, 2.0) -> underflows to 0
    (0x000116c2, 0x3f000000, 0x1e3ce4e7),   # powf(1e-40, 0.5) -> normal result
    (0x000116c2, 0xbf800000, 0x7f800000),   # powf(1e-40, -1.0) -> +inf
)


@pytest.mark.parametrize("ix,iy,want", _POWF_PATH_VECTORS)
def test_powf_sign_bias_and_subnormal_paths_match_glibc(ix, iy, want):
    got = _u32(powf(_f32(ix), _f32(iy)))
    assert got == want, (
        f"powf(0x{ix:08x}, 0x{iy:08x}) = powf({_f32(ix)!r}, {_f32(iy)!r}): "
        f"got 0x{got:08x}, glibc 0x{want:08x}"
    )


def test_powf_invalid_domain_nan_pins_the_named_divergence():
    """Negative base with a non-integer exponent: NaN, with one known miss.

    Live glibc 2.39 returns 0xFFC00000 here (``__math_invalidf`` computes
    0.0/0.0, and the x86-64 default QNaN carries the sign bit SET; measured
    2026-08-30 via libm_probe).  The transcription returns the positive QNaN
    0x7FC00000, and so do all six CUDA copies of this branch (gf.cu,
    mynn_dmp_sibling.cu, mynn_pbl.cu, noahmp_fluxprep.cu, noahmp_leaves.cu,
    rrtmg_lw.cu).  The divergence is NaN *sign only*, on a call no non-buggy
    Fortran physics can make, and per the module's own rule it is reported
    here, not absorbed.  A fix must move all seven sites together; until
    then this pin holds the current behaviour still in both directions.

    GPU-VERIFIED 2026-08-31 (the "assumed to match" caveat is retired):
    each of the six modules' own translation units (module_source, same
    -std=c++17 options load_module uses) was compiled on the desktop RTX
    3080 (sm_86, NVRTC 13.0.48 CL-36260728, driver 13030, cupy 14.0.1)
    with a probe kernel calling its pow entry point -- gfk_pow,
    mynn_glibc_powf x2, r_pow x2, rlw_pow -- at (-2.0f, 0.5f).  All six
    returned 0x7FC00000 on device: per-site compiles, not the
    anchor-plus-byte-agreement shortcut, so the verification rests on no
    extraction argument.  The pin below stands verified against the artifact.
    """
    got = _u32(powf(-2.0, 0.5))
    assert got == 0x7FC00000, (
        f"powf(-2.0, 0.5): got 0x{got:08x}; the transcription's pinned answer "
        f"is 0x7fc00000 (live glibc 2.39 gives 0xffc00000 -- if this moved, "
        f"move the six CUDA copies in the same commit and repin)"
    )
